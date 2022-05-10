from train import train
import os
import multiprocessing
import torch as th

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################

    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 5, 'eval_freq' : 5000}

    environment_kwargs = {
        'propulsion' : 'variable',
        'ha' : 'propulsion',
        'alpha' : 15,
        'start' : (100, 900),
        'target' : (800, 200),
        'radius' : 30,
        'dt' : 4,
        'initial_angle' : 0
    }
    
    model_kwargs = {
        'gamma' : 1,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 1000,
        'method' : 'PPO',
        'n_steps' : 2048,
        'batch_size' : 64
    }


    reward_kwargs = {
        'reward_number' : 3,
        'scale' : 0.01,
        'bonus': 10
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    #seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seeds = [1, 2]
    
    

    ########################################
    ### Run Train for all configurations ###
    ########################################
    name = 'png_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
                        
    os.mkdir(name)
    os.chdir(name)
    #Multi Porcessing

    processes = [multiprocessing.Process(target = train, args = [log_kwargs, environment_kwargs, model_kwargs, reward_kwargs, constraint_kwargs, seed]) for seed in seeds]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    os.chdir('../')

