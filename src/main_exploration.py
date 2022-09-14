from train import train
import os
import multiprocessing
import torch as th
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from utils import *
from env.wind_env import *
from env.wind.wind_map import *
from callback import ExplorationSpaceCallback
from train import linear_schedule

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################

    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 500, 'eval_freq' : 5000, 'MonteCarlo' : True}

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
        'restart' : 'random'
    }
    
    model_kwargs = {
        'gamma' : 1,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 150000,
        'method' : 'PPO',
        'n_steps' : 2048,
        'batch_size' : 64,
        'use_sde' : False
    }


    reward_kwargs = {
        'reward_number' : 4,
        'scale' : 1,
        'bonus': 0
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    #seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #seeds = [51+i for i in range(50)]
    seeds = [1]
    

    ########################################
    ### Run Train for all configurations ###
    ########################################
    if not os.path.exists('log_files'):
        os.mkdir('log_files')
    os.chdir('log_files')

    if not os.path.exists('wind_map_'+str(environment_kwargs['wind_info']['number'])):
        os.mkdir('wind_map_'+str(environment_kwargs['wind_info']['number']))
    os.chdir('wind_map_'+str(environment_kwargs['wind_info']['number']))
    if not os.path.exists('Exp_exploration'):
        os.mkdir('Exp_exploration')
    os.chdir('Exp_exploration')
    if environment_kwargs['continuous']:
        name = 'PPO_continuous_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
    else:
        name = 'PPO_discrete_'+str(environment_kwargs['dim_state'])+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)
        #Multi Porcessing


        #Create env and training
    wind_info = environment_kwargs['wind_info']
    discrete_maps = get_discrete_maps(wind_info['info'])
    A = WindMap(discrete_maps, wind_info['lengthscale'])

    env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale=  wind_info['lengthscale'],alpha = environment_kwargs['alpha'], start = environment_kwargs['start'], target= environment_kwargs['target'], target_radius= environment_kwargs['radius'], dt = environment_kwargs['dt'], propulsion = environment_kwargs['propulsion'], ha = environment_kwargs['ha'], reward_number = reward_kwargs['reward_number'], initial_angle=environment_kwargs['initial_angle'], bonus = reward_kwargs['bonus'], scale = reward_kwargs['scale'], reservoir_info = constraint_kwargs['reservoir_info'], continuous = environment_kwargs['continuous'], dim_state = environment_kwargs['dim_state'], discrete = environment_kwargs['discrete'], restart = environment_kwargs['restart'])

    callback = ExplorationSpaceCallback(eval_env = env, eval_freq = log_kwargs['eval_freq'], n_eval_episodes= 10)
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs = model_kwargs['policy_kwargs'], learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = model_kwargs['gamma'], seed = 2, n_steps = model_kwargs['n_steps'], batch_size = model_kwargs['batch_size'], use_sde = model_kwargs['use_sde'])
    model.learn(total_timesteps= model_kwargs['train_timesteps'], callback = callback)




    os.chdir('../../../../')