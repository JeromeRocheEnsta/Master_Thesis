from train import train
import os
import multiprocessing
import torch as th
from utils import *

Network = {
    1 : [dict(pi = [64,64], vf = [64,64])],
    #2 : [dict(pi = [32,32], vf = [32,32])],
    #2 : [dict(pi = [100,50,25], vf = [100,50,25])],
    #3 : [dict(pi = [400, 300], vf = [400, 300])],
}

Activation  = {
    'relu' : th.nn.ReLU,
    'tanh' : th.nn.Tanh,
    #'leakyrelu' : th.nn.LeakyReLU,
    'sigmoig' : th.nn.Sigmoid,
    #'softplus' : th.nn.Softplus
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
        'continuous' : True,
        'dim_state' : 7,
        'discrete' : [],
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
        'bonus': 0
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

    if not os.path.exists('Exp_network'):
        os.mkdir('Exp_network')
    os.chdir('Exp_network')
    
    if environment_kwargs['continuous']:
        name = model_kwargs['method']+'_continuous_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
    else:
        name = model_kwargs['method']+'_discrete_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])

                        
                        
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for cle, valeur in Activation.items():
        for cle2, valeur2 in Network.items():
            log = cle + '_' + str(cle2)

            os.mkdir(log)
            os.chdir(log)
            #Multi Porcessing
            
            model_kwargs['policy_kwargs'] = dict(activation_fn = valeur, net_arch = valeur2)

                                
            processes = [multiprocessing.Process(target = train, args = [log_kwargs, environment_kwargs, model_kwargs, reward_kwargs, constraint_kwargs, seed]) for seed in seeds]

            
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            os.chdir('../')
                        
    os.chdir('../')
    os.chdir('../')
    os.chdir('../')

