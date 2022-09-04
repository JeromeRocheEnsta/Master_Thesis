import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from utils import get_data


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10]
    timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy, ref_reward, ref_length, ref_energy = get_data(seeds, bonus = 0, scale = 1)
    ### Reward plot
    up = []
    down = []
    for i in range(len(timesteps)):
        up.append(mean_reward[i] + ci_reward[i])
        down.append(mean_reward[i] - ci_reward[i])

    fig, ax = plt.subplots()
    ax.plot(timesteps,mean_reward, color = 'b')
    ax.fill_between(timesteps, down, up, color='b', alpha=.1)
    ax.axhline(y=ref_reward, color='r', linestyle='-')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Reward')
    plt.savefig('reward.png')
    plt.close(fig)


    ### Length plot

    up = []
    down = []
    for i in range(len(timesteps)):
        up.append(mean_length[i] + ci_length[i])
        down.append(mean_length[i] - ci_length[i])

    fig, ax = plt.subplots()
    ax.plot(timesteps,mean_length, color = 'b')
    ax.fill_between(timesteps, down, up, color='b', alpha=.1)
    ax.axhline(y=ref_length, color='r', linestyle='-')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Length')
    plt.savefig('length.png')
    plt.close(fig)

    ### Energy plot

    up = []
    down = []
    for i in range(len(timesteps)):
        up.append(mean_energy[i] + ci_energy[i])
        down.append(mean_energy[i] - ci_energy[i])

    fig, ax = plt.subplots()
    ax.plot(timesteps,mean_energy, color = 'b')
    ax.fill_between(timesteps, down, up, color='b', alpha=.1)
    ax.axhline(y=ref_energy, color='r', linestyle='-')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Energy')
    plt.savefig('energy.png')
    plt.close(fig)