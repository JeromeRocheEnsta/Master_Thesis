import os
from turtle import color
import matplotlib.pyplot as plt
from utils import get_data


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10]
    timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy = get_data(seeds)
    print(timesteps)

    print(mean_reward)
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