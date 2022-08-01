from enum import auto
from train import train
import os
import multiprocessing
import torch as th
from utils import *

## Reward 1
"""
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
"""

## Reward 3 (sparse)
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
    },
    3 : {
        'type' : auto,
        'ts' : 200000,
        'factor' : 1.5
    }
}

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
        'initial_angle' : 0,
        'wind_info' : wind_info_1,
        'continuous' : True,
        'dim_state' : 3
    }
    
    model_kwargs = {
        'gamma' : 0.90,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 200000, ## For Curriculum 0
        'method' : 'PPO',
        'n_steps' : 2048,
        'batch_size' : 64
    }


    reward_kwargs = {
        'reward_number' : 3,
        'scale' : 1000,
        'bonus': 10
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if not os.path.exists('log_files'):
        os.mkdir('log_files')
    os.chdir('log_files')

    if not os.path.exists('wind_map_'+str(environment_kwargs['wind_info']['number'])):
        os.mkdir('wind_map_'+str(environment_kwargs['wind_info']['number']))
    os.chdir('wind_map_'+str(environment_kwargs['wind_info']['number']))

    if not os.path.exists('Exp_curriculum'):
        os.mkdir('Exp_curriculum')
    os.chdir('Exp_curriculum')
    

    if environment_kwargs['continuous']:
        name = model_kwargs['method']+'_continuous_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
    else:
        name = model_kwargs['method']+'_discrete_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])

    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for cle, valeur in Curriculum.items():
        log =  'curriculum_' + str(cle)
        if not os.path.exists(log):
            os.mkdir(log)
        else:
            continue
        os.chdir(log)

        if cle != 0:
            model_kwargs['Curriculum'] = valeur ## Work because the key 'Curriculum' is not affected for cle == 0

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

