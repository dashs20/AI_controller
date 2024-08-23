import numpy as np

def rk45_step(fun,state,dt,static):
    state_len = len(state)

    # k1
    k1 = fun(state,static)

    # k2
    b_state = np.zeros(state_len)
    for i in range(state_len):
        b_state[i] = state[i] + dt / 2 * k1[i]
    k2 = fun(b_state,static)

    # k3
    c_state = np.zeros(state_len)
    for i in range(state_len):
        c_state[i] = state[i] + dt / 2 * k2[i]
    k3 = fun(c_state,static)

    # k4
    d_state = np.zeros(state_len)
    for i in range(state_len):
        d_state[i] = state[i] + dt * k3[i]
    k4 = fun(d_state,static)

    # obtain step
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)