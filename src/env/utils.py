from tracemalloc import is_tracing
import numpy as np



def reward_1(state, target, length, heigth, target_radius, is_target):
    if is_target:
        return 10
    else:
        return(-  ( (np.sqrt( np.sqrt((state[1]  - target[0])**2 + (state[2] - target[1])**2) - target_radius))/(np.sqrt( np.sqrt(length**2 + heigth**2 ))) ))


def reward_2(energy, is_target):
    if is_target:
        return 100
    else:
        return(-energy)

def energy(v_prop, mu,k = 4, n = 3):
    return(k * ( (v_prop / mu) ** n) )

def score(E, t, Es, ts, alpha = 0.5):
    return(alpha * E / Es + (1 - alpha) * t / ts)
