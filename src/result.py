import os
import matplotlib.pyplot as plt
import numpy as np
from utils import get_data
import scipy.stats as st


def get_data(seeds):
    ## Need the current path to be in the directory containing the seeds directory
    n = len(seeds)
    mean_reward = []
    mean_length = []
    mean_energy = []
    if (n > 1):
        ci_reward = []
        ci_length = []
        ci_energy = []
    timesteps = []

    file_timestep = open('seed_1/monitoring.txt', 'r')
    ts= 0
    for _ in file_timestep:
        ts+=1
    file_timestep.close()


    for seed in seeds:
        exec('file'+str(seed)+'=open("seed_'+str(seed)+'/monitoring.txt", "r")')

    
    for _ in range(ts):
        count = 1
        rewards =[]
        lengths = []
        energies = []
        for seed in seeds:
            print(seed)
            exec('line = file'+str(seed)+'.readline()')
            exec('line = line.split()')
            
            if seed == 1:
                exec('print(line)')
                exec('timesteps.append(int(line[0]))')
            exec('rewards.append(int(line[1]))')
            exec('lengths.append(int(line[2]))')
            exec('energies.append(int(line[3]))')
            count += 1

        mean_reward.append(np.mean(rewards))
        mean_length.append(np.mean(lengths))
        mean_energy.append(np.mean(energies))
        if (n > 1) :
            t = st.t.ppf(0.975, n-1)
            ci_reward.append(t*np.sqrt(np.var(rewards)/n))
            ci_length.append(t*np.sqrt(np.var(lengths)/n))
            ci_energy.append(t*np.sqrt(np.var(energies)/n))

    for seed in seeds:
        exec('file'+str(seed)+'.close()')
    
    if(n > 1):
        return (timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy)
    else:
        return(timesteps, mean_reward, mean_length, mean_energy)


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10]
    timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy = get_data(seeds)

    up = []
    down = []
    for i in range(len(timesteps)):
        up.append(mean_reward[i] + ci_reward[i])
        down.append(mean_reward[i] - ci_reward[i])

    fig, ax = plt.subplots()
    ax.plot(timesteps,mean_reward, color = 'b')
    ax.fill_between(timesteps, down, up, color='b', alpha=.1)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Reward')
    plt.savefig('test.png')
    plt.close(fig)