import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib import ticker
from utils import get_data


color = {
    0 : 'blue',
    1 : 'green',
    2 : 'red',
    3 : 'black',
    4 : 'magenta',
    5 : 'cyan',
    6 : 'grey',
    7: 'purple',
    8 : 'brown'
}

if __name__ == "__main__":
    scale = 1
    bonus = 0
    seeds = [1,2,3,4,5,6,7,8,9,10]
    '''
    Configurations = {
        0 : [],
        1 : [-15, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 15],
        2 : [-15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15],
        3 : [-20, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 20],
        4 : [-30, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 30],
        5 : [-15, -5, -1, 0, 1, 5, 15],
    }
    '''
    '''
    Configurations = {
        0 : [],
        1 : [-15, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 15],
        2 : [-15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15],
        3 : [-20, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 20],
        4 : [-30, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 30],
        5 : [-50, -30, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 30, 50],
        6 : [-15, -5, -1, 0, 1, 5, 15],
    }
    '''
    Configurations = {
        0 : [15],
        1 : [30],
        2 : [40],
        3 : [-30, -15, -5, -1, 0, 1, 5, 15, 30],
        4 : [-15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15],
        5 : [-30, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 30],
        6 : [-40, -30, -15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15, 30, 40],
    }


    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(17, 7)

    xticks = ticker.MaxNLocator(6)
    ax[0].xaxis.set_major_locator(xticks)
    ax[1].xaxis.set_major_locator(xticks)
    ax[2].xaxis.set_major_locator(xticks)


    for cle, valeur in Configurations.items():
        os.chdir('configuration_'+str(cle))
        timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy, ref_reward, ref_length, ref_energy = get_data(seeds, scale = scale, bonus = bonus)
        
        ### Reward plot
        up = []
        down = []
        for i in range(len(timesteps)):
            up.append(mean_reward[i] + ci_reward[i])
            down.append(mean_reward[i] - ci_reward[i])

        ax[0].plot(timesteps,mean_reward, color = color[cle], label = 'configuration '+str(cle))
        ax[0].fill_between(timesteps, down, up, color= color[cle], alpha=.1)
        ax[0].axhline(y=ref_reward, color='r', linestyle='-')
        ax[0].set_xlabel('Timesteps')
        ax[0].set_ylabel('Average Reward')

        ### Length plot
        up = []
        down = []
        for i in range(len(timesteps)):
            up.append(mean_length[i] + ci_length[i])
            down.append(mean_length[i] - ci_length[i])


        ax[1].plot(timesteps,mean_length, color = color[cle])
        ax[1].fill_between(timesteps, down, up, color=color[cle], alpha=.1)
        ax[1].axhline(y=ref_length, color='r', linestyle='-')
        ax[1].set_xlabel('Timesteps')
        ax[1].set_ylabel('Average Length')


        ### Energy plot
        up = []
        down = []
        for i in range(len(timesteps)):
            up.append(mean_energy[i] + ci_energy[i])
            down.append(mean_energy[i] - ci_energy[i])

        ax[2].plot(timesteps,mean_energy, color = color[cle])
        ax[2].fill_between(timesteps, down, up, color=color[cle], alpha=.1)
        ax[2].axhline(y=ref_energy, color='r', linestyle='-')
        ax[2].set_xlabel('Timesteps')
        ax[2].set_ylabel('Average Energy')


        os.chdir('../')
    
   
    

    
    fig.legend()
    plt.savefig('metrics.png')
    plt.close(fig)