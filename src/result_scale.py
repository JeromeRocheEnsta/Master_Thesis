import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib import ticker


color = {
    0 : 'blue',
    1 : 'green',
    2 : 'red',
    3 : 'black',
    4 : 'magenta',
    5 : 'cyan'
}

def get_data(seeds, scale = None):
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

    ## Extract ref paths info

    file_ref_path = open('seed_1/info.txt', 'r')
    file_ref_path.readline()
    ref_line = file_ref_path.readline()
    ref_line = ref_line.split()
    ref_reward = float(ref_line[3][:-1]) if scale == None else float(ref_line[3][:-1])/scale
    ref_length = float(ref_line[6][:-1])
    ref_energy = float(ref_line[13])
    file_ref_path.close()

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
                exec('timesteps.append(float(line[0]))')
            if scale == None:
                exec('rewards.append(float(line[1]))')
            else:
                exec('rewards.append(float(line[1])/scale)')
            exec('lengths.append(float(line[2]))')
            exec('energies.append(float(line[3]))')
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
        return (timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy, ref_reward, ref_length, ref_energy)
    else:
        return(timesteps, mean_reward, mean_length, mean_energy, ref_reward, ref_length, ref_energy)


if __name__ == "__main__":
    seeds = [1,2,3,4,5,6,7,8,9,10]
    scales = [0.01, 0.1, 1, 10, 100, 1000]

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(17, 7)

    xticks = ticker.MaxNLocator(6)
    ax[0].xaxis.set_major_locator(xticks)
    ax[1].xaxis.set_major_locator(xticks)
    ax[2].xaxis.set_major_locator(xticks)


    for idx, scale in enumerate(scales):
        os.chdir('scale_'+str(scale))
        timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy, ref_reward, ref_length, ref_energy = get_data(seeds, scale = scale)
        
        ### Reward plot
        up = []
        down = []
        for i in range(len(timesteps)):
            up.append(mean_reward[i] + ci_reward[i])
            down.append(mean_reward[i] - ci_reward[i])

        ax[0].plot(timesteps,mean_reward, color = color[idx], label = 'scale ='+str(scale))
        ax[0].fill_between(timesteps, down, up, color= color[idx], alpha=.1)
        ax[0].axhline(y=ref_reward, color='r', linestyle='-')
        ax[0].set_xlabel('Timesteps')
        ax[0].set_ylabel('Average Reward')

        ### Length plot
        up = []
        down = []
        for i in range(len(timesteps)):
            up.append(mean_length[i] + ci_length[i])
            down.append(mean_length[i] - ci_length[i])


        ax[1].plot(timesteps,mean_length, color = color[idx])
        ax[1].fill_between(timesteps, down, up, color=color[idx], alpha=.1)
        ax[1].axhline(y=ref_length, color='r', linestyle='-')
        ax[1].set_xlabel('Timesteps')
        ax[1].set_ylabel('Average Length')


        ### Energy plot
        up = []
        down = []
        for i in range(len(timesteps)):
            up.append(mean_energy[i] + ci_energy[i])
            down.append(mean_energy[i] - ci_energy[i])

        ax[2].plot(timesteps,mean_energy, color = color[idx])
        ax[2].fill_between(timesteps, down, up, color=color[idx], alpha=.1)
        ax[2].axhline(y=ref_energy, color='r', linestyle='-')
        ax[2].set_xlabel('Timesteps')
        ax[2].set_ylabel('Average Energy')


        os.chdir('../')
    
   
    

    
    fig.legend()
    plt.savefig('metrics.png')
    plt.close(fig)