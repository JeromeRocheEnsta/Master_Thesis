from train import train
import os
import multiprocessing
import torch as th
from utils import *

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 500, 'eval_freq' : 5000}

    environment_kwargs = {
        'propulsion' : 'variable',
        'ha' : 'propulsion',
        'alpha' : 15,
        'start' : (100, 900),
        'target' : (800, 200),
        'radius' : 30,
        'dt' : 4,
        'initial_angle' : 315,
        'wind_info' : wind_info_4,
        'continuous' : True
    }
    
    model_kwargs = {
        'gamma' : 0.9,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 500000,
        'method' : 'PPO',
        'n_steps' : 2048,
        'batch_size' : 64
    }


    reward_kwargs = {
        'reward_number' : 1,
        'scale' : 1,
        'bonus': 10
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    bonuses = [0, 1, 5, 10, 50, 100, 1000]

    if not os.path.exists('log_files'):
        os.mkdir('log_files')
    os.chdir('log_files')

    if not os.path.exists('wind_map_'+str(environment_kwargs['wind_info']['number'])):
        os.mkdir('wind_map_'+str(environment_kwargs['wind_info']['number']))
    os.chdir('wind_map_'+str(environment_kwargs['wind_info']['number']))
    

    if not os.path.exists('Exp_bonus'):
        os.mkdir('Exp_bonus')
    os.chdir('Exp_bonus')
    
    if environment_kwargs['continuous']:
        name = model_kwargs['method']+'_continuous_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
    else:
        name = model_kwargs['method']+'_discrte_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
                        
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for bonus in bonuses:
            log = 'bonus_'+str(bonus)

            reward_kwargs['bonus'] = bonus

            os.mkdir(log)
            os.chdir(log)
            #Multi Porcessing
            
            processes = [multiprocessing.Process(target = train, args = [log_kwargs, environment_kwargs, model_kwargs, reward_kwargs, constraint_kwargs, seed]) for seed in seeds]

            for process in processes:
                process.start()
            for process in processes:
                process.join()

            os.chdir('../')
    
    
    os.chdir('../')
    os.chdir('../')
    os.chdir('../')