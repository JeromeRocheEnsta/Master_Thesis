import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib import ticker
from utils import get_data, filter


color = {
    0 : 'blue',
    1 : 'green',
    2 : 'red',
    3 : 'black',
    4 : 'magenta',
    5 : 'cyan',
    6 : 'grey'
}

Curriculum = {
    0 : {
    },
    1 : {
        'constraint' : [30000, 20000, 10000, 5000, 1000],
        'learning_rate' : [0.0005, 0.0001, 0.00005, 0.00001, 0.000005],
        'ts' : [50000, 50000, 50000, 50000, 50000]
    },
    2: {
        'constraint' : [30000, 20000, 10000, 5000, 4000, 3000, 2000, 1000],
        'learning_rate' : [0.0005, 0.0001, 0.00005, 0.00001, 0.00001, 0.00001, 0.00001, 0.000005],
        'ts' : [50000, 50000, 50000, 50000, 30000, 30000, 30000, 30000]
    }
}


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10]


    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(17, 7)

    xticks = ticker.MaxNLocator(6)
    ax[0].xaxis.set_major_locator(xticks)
    ax[1].xaxis.set_major_locator(xticks)
    ax[2].xaxis.set_major_locator(xticks)


    for cle, valeur in Curriculum.items():
        os.chdir('curriculum_'+str(cle))
        timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy, ref_reward, ref_length, ref_energy = get_data(seeds)
        

        if cle == 0:
            ### Reward plot
            up = []
            down = []
            for i in range(len(timesteps)):
                up.append(mean_reward[i] + ci_reward[i])
                down.append(mean_reward[i] - ci_reward[i])

            ax[0].plot(timesteps,mean_reward, color = color[cle], label = 'Curriculum ='+str(cle))
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
        else:
            ts = valeur['ts']
            n_curves = len(ts)
            I = [0]
            counter = 0
            for i in range(n_curves):
                counter += ts[i]
                for idx, x in enumerate(timesteps):
                    if x == counter:
                        I.append(idx)
            
            ### Reward plot
            up = []
            down = []
            for i in range(len(timesteps)):
                up.append(mean_reward[i] + ci_reward[i])
                down.append(mean_reward[i] - ci_reward[i])
                
            for i in range(len(I)-1):
                if i == 0:
                    ax[0].plot(timesteps[I[i]:I[i+1]+1],mean_reward[I[i]:I[i+1]+1], color = color[cle], label = 'Curriculum ='+str(cle))
                else:
                    ax[0].plot(timesteps[I[i]:I[i+1]+1],mean_reward[I[i]:I[i+1]+1], color = color[cle])
                ax[0].fill_between(timesteps[I[i]:I[i+1]+1], down[I[i]:I[i+1]+1], up[I[i]:I[i+1]+1], color= color[cle], alpha=.1)
            ax[0].axhline(y=ref_reward, color='r', linestyle='-')
            ax[0].set_xlabel('Timesteps')
            ax[0].set_ylabel('Average Reward')

            ### Length plot
            up = []
            down = []
            for i in range(len(timesteps)):
                up.append(mean_length[i] + ci_length[i])
                down.append(mean_length[i] - ci_length[i])

            for i in range(len(I)-1):  
                ax[1].plot(timesteps[I[i]:I[i+1]+1],mean_length[I[i]:I[i+1]+1], color = color[cle])
                ax[1].fill_between(timesteps[I[i]:I[i+1]+1], down[I[i]:I[i+1]+1], up[I[i]:I[i+1]+1], color=color[cle], alpha=.1)
            ax[1].axhline(y=ref_length, color='r', linestyle='-')
            ax[1].set_xlabel('Timesteps')
            ax[1].set_ylabel('Average Length')


            ### Energy plot
            up = []
            down = []
            for i in range(len(timesteps)):
                up.append(mean_energy[i] + ci_energy[i])
                down.append(mean_energy[i] - ci_energy[i])

            for i in range(len(I)-1):
                ax[2].plot(timesteps[I[i]:I[i+1]+1],mean_energy[I[i]:I[i+1]+1], color = color[cle])
                ax[2].fill_between(timesteps[I[i]:I[i+1]+1], down[I[i]:I[i+1]+1], up[I[i]:I[i+1]+1], color=color[cle], alpha=.1)
            ax[2].axhline(y=ref_energy, color='r', linestyle='-')
            ax[2].set_xlabel('Timesteps')
            ax[2].set_ylabel('Average Energy')

        os.chdir('../')
    
   
    

    
    fig.legend()
    plt.savefig('metrics.png')
    plt.close(fig)