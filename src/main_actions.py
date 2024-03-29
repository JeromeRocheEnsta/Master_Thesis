from train import train
import os
import multiprocessing
import torch as th
from utils import *

Configurations = {
    #0 : [10],
    1 : [15],
    #2 : [30],
    #3 : [-15, -1, 0, 1, 15],
    #4 : [-15, -5, -1, 0, 1, 5, 15],
    #5 : [-15, -10, -5, -1, 0, 1, 5, 10, 15],
    #6 : [-15, -10, -5, -2.5, -1, 0, 1, 2.5, 5, 10, 15],
    #7 : [-15, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 15],
    #8 : [-15, -10, -7.5, -5, -2.5, -1, 0, 1, 2.5, 5, 7.5, 10, 15]
}

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 500, 'eval_freq' : 5000, 'MonteCarlo' : False}

    environment_kwargs = {
        'propulsion' : 'variable',
        'ha' : 'propulsion',
        'alpha' : 15,
        'start' : (100, 900),
        'target' : (800, 200),
        'radius' : 30,
        'dt' : 4,
        'initial_angle' : 0,
        'wind_info' : wind_info_1,
        'continuous' : False,
        'dim_state' : 7,
        'discrete' : [-15, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 15],
        'restart' : 'fix'
    }
    
    model_kwargs = {
        'gamma' : 1,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 400000,
        'method' : 'PPO',
        'n_steps' : 2048,
        'batch_size' : 64
    }


    reward_kwargs = {
        'reward_number' : 4,
        'scale' : 1,
        'bonus': 0,
        #'power' : 1
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    #seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seeds = [1]


    if not os.path.exists('log_files'):
        os.mkdir('log_files')
    os.chdir('log_files')

    if not os.path.exists('wind_map_'+str(environment_kwargs['wind_info']['number'])):
        os.mkdir('wind_map_'+str(environment_kwargs['wind_info']['number']))
    os.chdir('wind_map_'+str(environment_kwargs['wind_info']['number']))

    if not os.path.exists('Exp_actions'):
        os.mkdir('Exp_actions')
    os.chdir('Exp_actions')
    
    
    name = model_kwargs['method']+'_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
                        
                        
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for cle, valeur in Configurations.items():
        log = 'configuration_' + str(cle)

        os.mkdir(log)
        os.chdir(log)
        #Multi Porcessing
            
        environment_kwargs['discrete'] = valeur
        if len(valeur) == 1:
            environment_kwargs['continuous'] = True
            environment_kwargs['alpha'] = float(valeur[0])
        else:
            environment_kwargs['continuous'] = False
                                
        processes = [multiprocessing.Process(target = train, args = [log_kwargs, environment_kwargs, model_kwargs, reward_kwargs, constraint_kwargs, seed]) for seed in seeds]

            
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        os.chdir('../')
                        
    os.chdir('../')
    os.chdir('../')
    os.chdir('../')

