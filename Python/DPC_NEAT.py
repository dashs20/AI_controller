import numpy as np
from custom_rk45 import rk45_step
import neat
import matplotlib.pyplot as plt
import pickle

# Pendulum Dynamics Model
def DPC_dynamics(state,static):
    # unpack args
    cart_properties, F, g = static

    # unpack state
    x, dx, t1, dt1, t2, dt2 = state

    # unpack properties
    m, m1, m2, l1, l2 = cart_properties

    # Math Credit: TU Berlin "Equations of motion for an inverted double pendulum on a cart (in generalized coordinates)"

    # calculate "M(y) matrix"
    My_matrix = np.zeros([3,3])

    My_matrix[0,0] = m + m1 + m1
    My_matrix[0,1] = l1*(m1 + m2)*np.cos(t1)
    My_matrix[0,2] = m2*l2*np.cos(t2)
    My_matrix[1,0] = l1 * (m1 + m2) * np.cos(t1)
    My_matrix[1,1] = l1**2 * (m1 + m2)
    My_matrix[1,2] = l1 * l2 * m2 * np.cos(t1 - t2)
    My_matrix[2,0] = l2 * m2 * np.cos(t2)
    My_matrix[2,1] = l1 * l2 * m2 * np.cos(t1 - t2)
    My_matrix[2,2] = l2**2 * m2

    # calculate RHS components
    RHS1 = np.zeros([3,1])

    RHS1[0] = l1 * (m1 + m2) * dt1**2 * np.sin(t1) + m2 * l2 * dt2**2 * np.sin(dt2)
    RHS1[1] = -l1 * l2 * m2 * dt2**2 * np.sin(t1 - t2) + g * (m1 + m2) * l1 * np.sin(t1)
    RHS1[2] = l1 * l2 * m2 * dt1**2 * np.sin(t1 - t2) + g * l2 * m2 * np.sin(t2)

    RHS2 = np.zeros([3,1])

    RHS2[0] = F

    # calculate ddx, ddt1 and ddt2
    result = np.matmul(np.linalg.inv(My_matrix),RHS1 + RHS2)

    # return dstate

    return np.array([dx,float(result[0]),dt1,float(result[1]),dt2,float(result[2])])

# Cart Run Function
def run_cart(genomes,config):
    # Define Cart
    cart_properties = np.array([5,0.5,0.5,1,1]) # [m,m1,m2,l1,l2]

    # Define physical constants
    g = 1 # gravity
    F = 0

    static = [cart_properties,F,g]

    # Define simulation
    IC = np.array([0,0,np.pi*0.9,0,np.pi*1.1,0]) # [x, dx, t1, dt1, t2, dt2] where t is theta (rad)
    max_time = 60 # seconds
    steps = max_time * 1000
    tvec = np.linspace(0,max_time,steps)
    dt = max_time/steps
    cart_bounds = [-10,10]

    # Init NEAT
    nets = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

    # Pre-load initial contition to array of current states
    # also generate a list of time hists
    time_hists = list()
    force_hists = list()
    cur_states = np.zeros([len(genomes),6])
    for i in range(len(genomes)):
        cur_states[i,:] = IC
        time_hists.append(np.zeros([steps,6]))
        force_hists.append(np.zeros(steps))

    # Main Loop
    global generation
    generation += 1

    # loop for 1 seconds
    for i in range(steps):
        # iterate through all the pendulums
        for j in range(len(genomes)):
            # Step each system with input force from neural net
            static[1] = nets[j].activate(cur_states[j,:])[0] * 200

            # keep cart in bounds
            if(cur_states[j,0] < cart_bounds[0]):
                cur_states[j,0] = cart_bounds[0]
                cur_states[j,1] = 0
                static[1] = -(cur_states[j,0] - cart_bounds[0]) * 10000
            
            if(cur_states[j,0] > cart_bounds[1]):
                cur_states[j,0] = cart_bounds[1]
                cur_states[j,1] = 0
                static[1] = -(cur_states[j,0] - cart_bounds[1]) * 10000

            # step cart along
            cur_states[j,:] = rk45_step(DPC_dynamics,cur_states[j,:],dt,static)
            time_hists[j][i,:] = cur_states[j,:]
            force_hists[j][i] = static[1]

            # update fitness
            if(np.cos(cur_states[j,2])*cart_properties[3] + np.cos(cur_states[j,4])*cart_properties[4] > 0.9*(cart_properties[3]+cart_properties[4])):
                genomes[j][1].fitness += 1 * dt

    # Pickle time histories in case anything good comes out.
    with open(f'prev_runs/time_hists{generation}.pkl', 'wb') as f:  # open a text file
        pickle.dump(time_hists, f) # serialize the list

    with open(f'prev_runs/force_hists{generation}.pkl', 'wb') as f:  # open a text file
        pickle.dump(force_hists, f) # serialize the list

    with open(f'prev_runs/tvec{generation}.pkl', 'wb') as f:  # open a text file
        pickle.dump(tvec, f) # serialize the list

    # plot results
    # for i in range(len(genomes)):
    #     plt.plot(tvec,time_hists[i][:,0])

    # plt.figure()
    # for i in range(len(genomes)):
    #         plt.plot(tvec,time_hists[i][:,2])

    # plt.figure()
    # for i in range(len(genomes)):
    #     plt.plot(tvec,time_hists[i][:,4])

    # plt.figure()
    # for i in range(len(genomes)):
    #     plt.plot(tvec,force_hists[i][:])

    # plt.show()

if __name__ == "__main__":
    generation = 0

    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    p.run(run_cart, 1000)