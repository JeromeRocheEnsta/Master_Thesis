from train import train
import os
import multiprocessing

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    seeds = [1, 2]
    start = (100, 900)
    target = (800, 200)
    initial_angle = 0
    radius = 20
    propulsion = 'variable'
    eval_freq = 1000
    
    
    list_ha = ['propulsion']
    list_alpha = [15]
    list_reward_number = [1]
    list_dt = [4]
    list_gamma = [0.9]
    train_timesteps = 10000
    

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

                        for seed in seeds:
                            print(seed)
                            exec("process"+str(seed)+"= multiprocessing.Process(target = train, args = [True, propulsion, ha, alpha, reward_number, start, target, initial_angle, radius, dt, gamma, train_timesteps, seed, eval_freq])")
                        
                        for seed in seeds:
                            exec("process"+str(seed)+".start()")

                        for seed in seeds:
                            exec("process"+str(seed)+".join()")
                        
                        
                        #train(save = True, dir_name = name, propulsion = propulsion, ha = ha, alpha = alpha, reward_number = reward_number,
                        #start = start, target = target, initial_angle = initial_angle, radius = radius, dt = dt, gamma = gamma, train_timesteps = train_timesteps, seed = seed, eval_freq = eval_freq)
                        
                        
                        os.chdir('../')

