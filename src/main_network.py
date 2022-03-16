from train import train
import os
import multiprocessing
import torch as th

Network = {
    1 : [dict(pi = [64,64], vf = [64,64])],
    2 : [dict(pi = [100,50,25], vf = [100,50,25])],
    3 : [dict(pi = [400, 300], vf = [400, 300])]
}

Activation  = {
    'relu' : th.nn.ReLU,
    'tanh' : th.nn.Tanh,
    'leakyrelu' : th.nn.LeakyReLU
}

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    train_timesteps = 500000
    method = 'SAC'

    if not os.path.exists('Exp_network'):
        os.mkdir('Exp_network')
    os.chdir('Exp_network')
    

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
                        name = method+'_'+str(reward_number)+'_'+ha+'_'+str(alpha)+'_'+str(dt)+'_'+str(gamma)+'_'+str(train_timesteps)
                        
                        os.mkdir(name)
                        os.chdir(name)

                        for cle, valeur in Activation.items():
                            for cle2, valeur2 in Network.items():
                                log = cle + '_' + str(cle2)

                                os.mkdir(log)
                                os.chdir(log)
                                #Multi Porcessing
                                
                                policy_kwargs = dict(activation_fn = valeur, net_arch = valeur2)

                                for seed in seeds:
                                    print(seed)
                                    exec("process"+str(seed)+"= multiprocessing.Process(target = train, args = [True, propulsion, ha, alpha, reward_number, start, target, initial_angle, radius, dt, gamma, train_timesteps, seed, eval_freq, policy_kwargs, method])")
                                
                                for seed in seeds:
                                    exec("process"+str(seed)+".start()")

                                for seed in seeds:
                                    exec("process"+str(seed)+".join()")

                                os.chdir('../')
                        
                        #train(save = True, dir_name = name, propulsion = propulsion, ha = ha, alpha = alpha, reward_number = reward_number,
                        #start = start, target = target, initial_angle = initial_angle, radius = radius, dt = dt, gamma = gamma, train_timesteps = train_timesteps, seed = seed, eval_freq = eval_freq)
                        
                        
                        os.chdir('../')

