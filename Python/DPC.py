import numpy as np
from custom_rk45 import rk45_step
import matplotlib.pyplot as plt

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
    result = np.matmul(np.linalg.inv(My_matrix),RHS1) + RHS2

    # return dstate

    return np.array([dx,float(result[0]),dt1,float(result[1]),dt2,float(result[2])])

# Define Cart
cart_properties = np.array([1,1,1,1,1]) # [m,m1,m2,l1,l2]

# Define physical constants
g = 9.81 # gravity
F = 0

static = [cart_properties,F,g]

# Define simulation
IC = np.array([0,0,0.1,0,0,0]) # [x, dx, t1, dt1, t2, dt2] where t is theta (rad)
max_time = 10 # seconds
steps = max_time * 1000
tvec = np.linspace(0,max_time,steps)
dt = max_time/steps

time_hist = np.zeros([steps,6])

cur_state = IC

# Run Simulation
for i, time in enumerate(tvec):
    cur_state = rk45_step(DPC_dynamics,cur_state,dt,static)
    time_hist[i,:] = cur_state

# plot results
plt.plot(tvec,time_hist[:,0])

plt.figure()
plt.plot(tvec,time_hist[:,2])
plt.plot(tvec,time_hist[:,4])

plt.show()




