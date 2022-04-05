from train import train
import os
import multiprocessing
import torch as th

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    start = (100, 900)
    target = (800, 200)
    initial_angle = 0
    radius = 30
    propulsion = 'variable'
    eval_freq = 5000
    
    bonus = 10
    scale = 1
    list_ha = ['propulsion']
    list_alpha = [15]
    list_reward_number = [1]
    list_dt = [4]
    list_gamma = [0.9]
    train_timesteps = 2000
    policy_kwargs = dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])])
    

    for i in range(len(list_reward_number)):
        for j in range(len(list_ha)):
            for k in range(len(list_alpha)):
                for l in range(len(list_dt)):
                    for m in range(len(list_gamma)):
                        reward_number =list_reward_number[i]
                        ha = list_ha[j]
                        alpha = list_alpha[k]
                        dt = list_dt[l]
                        gamma = list_gamma[m]
                        name = 'png_'+str(reward_number)+'_'+ha+'_'+str(alpha)+'_'+str(dt)+'_'+str(gamma)+'_'+str(train_timesteps)
                        
                        os.mkdir(name)
                        os.chdir(name)
                        #Multi Porcessing

                        processes = [multiprocessing.Process(target = train, args = [True, propulsion, ha, alpha, reward_number, start, target, initial_angle, radius, dt, gamma, train_timesteps, seed, eval_freq, None, 'PPO', bonus, scale]) for seed in seeds]

                        for process in processes:
                            process.start()
                        for process in processes:
                            process.join()

                        os.chdir('../')

