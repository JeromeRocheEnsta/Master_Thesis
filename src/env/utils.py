from tracemalloc import is_tracing
import numpy as np



def reward_1(state, target, length, heigth, target_radius, is_target, bonus = 10, scale = 1):
    if is_target:
        return bonus * scale
    else:
        return(- scale * ( (np.sqrt( np.sqrt((state[1]  - target[0])**2 + (state[2] - target[1])**2) - target_radius))/(np.sqrt( np.sqrt(length**2 + heigth**2 ))) ))


def reward_2(magnitude, magnitude_max, direction, position, target, is_target, bonus = 10, scale = 1):
    if is_target:
        return bonus * scale
    else:
        if (position[0] == target[0]):
            angle = direction + np.sign(position[1] - target[1]) * 90
        else:
            theta = np.arctan(abs(position[1] - target[1])/abs(position[0] - target[0]))
            angle = 0 if (position[0] < target[0]) else -180
            angle +=  direction
            if(position[0]> target[0] and position[1]>=target[1]) or (position[0] < target[0] and position[1] < target[1]):
                angle -= theta
            else:
                angle += theta
                
        return magnitude/magnitude_max * np.cos(angle * np.pi / 180) * scale
    

def reward_sparse(is_target, scale = 1):
    if is_target:
        return scale
    else:
        return 0



def energy(v_prop, mu, k = 0.1, n = 2):
    return(k * ( (v_prop) ** n) )

