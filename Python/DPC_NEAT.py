import numpy as np
from custom_rk45 import rk45_step
import neat
import pickle
import scipy.io as sio

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

    # calculate RHS2 components
    RHS2 = np.zeros([3,1])
    RHS2[0] = F

    # calculate RHS3 components
    RHS3 = np.zeros([3,1])

    d1 = 0
    d2 = 0.2
    d3 = 0.2

    RHS3[0] = d1 * dx
    RHS3[1] = d2 * dt1
    RHS3[2] = d3 * dt2

    # calculate ddx, ddt1 and ddt2
    result = np.matmul(np.linalg.inv(My_matrix),RHS1 + RHS2 - RHS3)

    # return dstate

    return np.array([dx,float(result[0]),dt1,float(result[1]),dt2,float(result[2])])

# Cart Run Function
def run_cart(genomes,config):
    # Define Cart
    cart_properties = np.array([5,0.1,0.1,1,1]) # [m,m1,m2,l1,l2]

    # Define physical constants
    g = 1 # gravity
    F = 0

    static = [cart_properties,F,g]

    # Define simulation
    IC = np.array([0,0,np.pi,0,np.pi,0]) # [x, dx, t1, dt1, t2, dt2] where t is theta (rad)
    max_time = 30 # seconds
    steps = max_time * 60
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
    for j in range(len(genomes)):
        # iterate through all the pendulums
        up = False
        for i in range(steps):
            # Step each system with input force from neural net
            # obtain input parameters for the neural net
            params = np.zeros(11)

            # [pos, vel, t1, dt1, t2, dt2]
            # [0    1    2   3    4   5  ]

            params[0] = cur_states[j,0] # cart position
            params[1] = cur_states[j,1] # cart velocity
            params[2] = cur_states[j,3] # theta 1 dot
            params[3] = cur_states[j,5] # theta 2 dot
            params[4] = np.sin(cur_states[j,2]) * cart_properties[3] # pend 1 x
            params[5] = np.cos(cur_states[j,2]) * cart_properties[3] # pend 1 y
            params[6] = params[4] + np.sin(cur_states[j,4]) * cart_properties[4] # pend 2 x
            params[7] = params[5] + np.cos(cur_states[j,4]) * cart_properties[4] # pend 2 y
            params[8] = params[4] * params[6] + params[5] * params[7] # dot(dir 1, dir 2)
            params[9] = cur_states[j,2] # theta 1
            params[10] = cur_states[j,4] # theta 2

            static[1] = nets[j].activate(params)[0] * 500

            # kill the agent if it hits the wall
            if(cur_states[j,0] < cart_bounds[0] or cur_states[j,0] > cart_bounds[1]):
                break

            # kill the agent if it tries to swing it around
            if(cur_states[j,2] > IC[2] + np.pi * 2 or cur_states[j,2] < IC[2] - np.pi * 2):
                break

            # once the pendulum is up, if it falls, kill the agent.
            tip_height = np.cos(cur_states[j,2])*cart_properties[3] + np.cos(cur_states[j,4])*cart_properties[4]
            pend_length = cart_properties[3]+cart_properties[4]
            if(tip_height > 0.95 * pend_length):
                up = True
            if(up == True and tip_height < 0.95 * pend_length):
                break

            # step cart along
            cur_states[j,:] = rk45_step(DPC_dynamics,cur_states[j,:],dt,static)
            time_hists[j][i,:] = cur_states[j,:]
            force_hists[j][i] = static[1]

            # update fitness
            def fitness_func(cur_states,dt,cart_bounds,up,force_hist):

                if(up):
                    # incentivise staying close to the center, but not as much as keeping the pendulum up
                    center_component = (-np.absolute(cur_states[j,0]) + cart_bounds[1])/cart_bounds[1]
                    # incentivise being up in the 1st place.
                    height_component = 1
                    # incentivise smooth force
                    force_component = -np.std(force_hist)/300

                    # combine components to get score
                    score = (height_component + center_component + force_component) * dt
                    return score
                else:
                    return 0

            genomes[j][1].fitness += fitness_func(cur_states,dt,cart_bounds,up,force_hists[j][0:i])

    # locate fittest agent
    fitnesses = list()
    for i in range(len(genomes)):
        fitnesses.append(genomes[i][1].fitness)
    
    best_agent_index = np.argmax(fitnesses)

    best_timehist = time_hists[best_agent_index]
    best_forcehist = force_hists[best_agent_index]

    # save every 10th generation
    if(not generation % 10):
        sio.savemat(f'../MATLAB/mat_files/result{generation}.mat', {'th':best_timehist,'tvec':tvec,'fh':best_forcehist})

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
    p.run(run_cart, 20000)