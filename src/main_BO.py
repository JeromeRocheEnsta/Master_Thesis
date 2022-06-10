from train_BO import train
import os
import multiprocessing
import torch
from utils import *

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    log_kwargs = {
        'save' : True
    }
    
    environment_kwargs = {
        'propulsion' : 'variable',
        'ha' : 'propulsion',
        'alpha' : 15,
        'start' : (100, 900),
        'target' : (800, 200),
        'radius' : 30,
        'dt' : 4,
        'initial_angle' : 0,
        'wind_info' : wind_info_3
    }
    
    model_kwargs = {
        'gamma' : 0.99,
        'n_eval_episodes' : 1,
        'dim' : 3,
        'bounds' : torch.tensor([ [-1./360, -0.001, -0.001] , [1./360, 0.001, 0.001] ], dtype = torch.float64),
        'batch_size' : 3
    } 
    
    reward_kwargs = {
        'reward_number' : 1,
        'scale' : 0.01,
        'bonus': 10
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    seeds = [0, 77, 93747827683, 2837109374, 30293781987, 274763920, 34529873, 4786329, 504938962, 29874085, 38478930]

    ########################################
    ### Run Train for all configurations ###
    ########################################
    name = 'BO_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['n_eval_episodes'])
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    processes = [multiprocessing.Process(target = train, args = [log_kwargs, environment_kwargs, model_kwargs, reward_kwargs, constraint_kwargs, seed]) for seed in seeds]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

        

    os.chdir('../')

