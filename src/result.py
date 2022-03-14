import os
import matplotlib.pyplot as plt
from utils import get_data


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10]
    timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy = get_data(seeds)

    plt.plot(timesteps, mean_reward)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.show()