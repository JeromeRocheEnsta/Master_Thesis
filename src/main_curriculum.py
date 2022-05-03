from train import train
import os
import multiprocessing
import torch as th

Curriculum = {
    1 : {
        'constraint' : [30000, 20000, 10000, 5000, 1000],
        'learning_rate' : [0.0005, 0.0001, 0.00005, 0.00005, 0.000005],
        'ts' : [30000, 30000, 40000, 40000, 50000]
    }
}


if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    #seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seeds = [1]
    start = (100, 900)
    target = (800, 200)
    initial_angle = 0
    radius = 30
    propulsion = 'variable'
    eval_freq = 5000
    
    bonus = 10
    scale = 0.01
    list_ha = ['propulsion']
    list_alpha = [15]
    list_reward_number = [1]
    list_dt = [4]
    list_gamma = [0.9]
    n_steps = 2048
    batch_size = 64
    reservoir_info = [False, None]
    train_timesteps = 200000
    method = 'PPO'
    policy_kwargs = dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])])

    if not os.path.exists('Exp_curriculum'):
        os.mkdir('Exp_curriculum')
    os.chdir('Exp_curriculum')
    

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
                        name = method+'_'+str(reward_number)+'_'+str(bonus)+'_'+str(scale)+'_'+ha+'_'+str(alpha)+'_'+str(dt)+'_'+str(gamma)
                        
                        if not os.path.exists(name):
                            os.mkdir(name)
                        os.chdir(name)

                        for cle, valeur in Curriculum.items():
                            
                            log =  'curriculum_' + str(cle)
                            os.mkdir(log)
                            os.chdir(log)
                            #Multi Porcessing
                                
                            processes = [multiprocessing.Process(target = train, args = [True, propulsion, ha, alpha, reward_number, start, target, initial_angle, radius, dt, gamma, train_timesteps, seed, eval_freq, policy_kwargs, method, bonus, scale, n_steps, batch_size, reservoir_info, valeur]) for seed in seeds]
                                
                            for process in processes:
                                process.start()
                            for process in processes:
                                process.join()

                            os.chdir('../')
                        
                        #train(save = True, dir_name = name, propulsion = propulsion, ha = ha, alpha = alpha, reward_number = reward_number,
                        #start = start, target = target, initial_angle = initial_angle, radius = radius, dt = dt, gamma = gamma, train_timesteps = train_timesteps, seed = seed, eval_freq = eval_freq)
                        
                        
                        os.chdir('../')

