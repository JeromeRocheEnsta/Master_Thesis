import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from utils import *
from env.wind_env import *
from env.wind.wind_map import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from typing import Callable
from callback import TrackCurriculumCallback, TrackExpectedRewardCallback

gym.logger.set_level(40)


def linear_schedule(initial_value: float, end_value: float, end_progress: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining < end_progress:
            return end_value
        else:
            a = (initial_value - end_value)/(1 - end_progress)
            b = initial_value - a
            return a * progress_remaining + b

    return func


def train(log_kwargs = {'save' : False, 'n_eval_episodes_callback' : 5, 'eval_freq' : 5000, 'MonteCarlo' : False},
environment_kwargs = {'propulsion' : 'variable', 'ha' : 'propulsion', 'alpha' : 15, 'start' : (100, 900), 'target' : (800, 200),
'radius' : 20, 'dt' : 1.8, 'initial_angle' : 0, 'wind_info' : wind_info_1, 'continuous' : True, 'dim_state' : 3, 'discrete' : [],
'restart' : 'fix'},
model_kwargs = {'gamma' : 0.99, 'policy_kwargs' : None, 'train_timesteps' : 150000, 'method' : 'PPO','n_steps' : 2048,
'batch_size' : 64, 'use_sde' : False},
reward_kwargs = {'reward_number' : 1, 'scale' : 1, 'bonus': 10},
constraint_kwargs = {'reservoir_info' : [False, None]},
seed = 1):
    print("Execution de train avec seed = {}".format(seed))
    start_time = time.time()
    ### Preprocess args
    
    # log_kwargs
    save = log_kwargs['save']
    n_eval_episodes_callback = 5 if log_kwargs.get('n_eval_episodes_callback') == None else log_kwargs['n_eval_episodes_callback']
    eval_freq = log_kwargs['eval_freq'] if log_kwargs.get('eval_freq') != None else 5000
    MonteCarlo = False if log_kwargs.get('n_eval_episodes_callback') == None else log_kwargs['MonteCarlo']

    #environment_kwargs
    propulsion = environment_kwargs['propulsion']
    ha = environment_kwargs['ha']
    alpha = environment_kwargs['alpha']
    start = environment_kwargs['start']
    target = environment_kwargs['target']
    radius = environment_kwargs['radius']
    dt = environment_kwargs['dt']
    initial_angle = environment_kwargs['initial_angle']
    continuous = environment_kwargs['continuous'] if 'continuous' in environment_kwargs else True
    discrete = [] if continuous else environment_kwargs['discrete']
    dim_state = environment_kwargs['dim_state'] if 'dim_state' in environment_kwargs else 3
    restart = environment_kwargs['restart'] if 'restart' in environment_kwargs else 'fix'

    wind_info = environment_kwargs['wind_info']['info']
    wind_lengthscale = environment_kwargs['wind_info']['lengthscale']

    #model_kwargs
    gamma = model_kwargs['gamma']
    use_sde = model_kwargs['use_sde'] if model_kwargs.get('use_sde') else False
    policy_kwargs = model_kwargs.get('policy_kwargs')
    if model_kwargs.get('Curriculum') == None:
        train_timesteps = model_kwargs['train_timesteps']
        Curriculum = None
    else:
        Curriculum = model_kwargs['Curriculum']
    method = model_kwargs['method'] if model_kwargs.get('method') != None else 'PPO'
    n_steps = model_kwargs['n_steps'] if model_kwargs.get('n_steps') != None else 2048
    batch_size = model_kwargs['batch_size'] if model_kwargs.get('batch_size') != None else 64

    #rewrd_kwargs
    reward_number = reward_kwargs['reward_number'] if reward_kwargs.get('reward_number') != None else 1
    scale = reward_kwargs['scale'] if reward_kwargs.get('scale') != None else 1
    bonus = reward_kwargs['bonus'] if reward_kwargs.get('bonus') != None else 10
    if reward_number == 1:
        power = reward_kwargs['power'] if reward_kwargs.get('power') != None else 0.5

    #constraint_kwargs
    reservoir_info = constraint_kwargs['reservoir_info'] if constraint_kwargs.get('reservoir_info') != None else [False, None]

    # MKDIR to stock figures output
    if save:
        dir_name = 'seed_'+str(seed)
        os.mkdir(dir_name)
        file1 = open(dir_name+'/info.txt', 'w')
        

    # Crete WindMap Object (Wind Field modelized with a GP)
    discrete_maps = get_discrete_maps(wind_info)
    A = WindMap(discrete_maps, wind_lengthscale)

    # Save Visualization of the wind field
    fig = plot_wind_field(A, start, target)
    if save:
        plt.savefig(dir_name+'/wind_field.png')
    plt.close(fig)

    # reference path
    straight_angle = get_straight_angle(start, target)
    if reward_number == 1:
        env_ref = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius=radius, dt = dt, straight = True, ha = 'next_state', propulsion = propulsion, reward_number= reward_number, initial_angle= straight_angle, power = power, bonus = bonus, scale = scale, dim_state= dim_state, restart = 'fix')
    else:
        env_ref = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius=radius, dt = dt, straight = True, ha = 'next_state', propulsion = propulsion, reward_number= reward_number, initial_angle= straight_angle, bonus = bonus, scale = scale, dim_state= dim_state, restart = 'fix')
    env_ref.reset()
    reward_ref = 0
    while env_ref._target() == False:
        obs, reward, done, info = env_ref.step(0)
        reward_ref += reward
    ##Plot trajectory
    fig, axs = env_ref.plot_trajectory(reward_ref)
    if save:
        plt.savefig(dir_name+'/ref_path.png')
    plt.close(fig)
    file1.write('Reference Path info: \n')
    file1.write('Cumulative Reward : {}; Timesteps : {}; Time : {}; Energy Consumed : {} \n'.format(reward_ref, len(env_ref.time) - 1, round(env_ref.time[-1], 1), round(env_ref.energy[-1])))
    

    # train agent
    if Curriculum == None:
        if reward_number == 1:
            env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, power = power, bonus = bonus, scale = scale, reservoir_info = reservoir_info, continuous = continuous, dim_state = dim_state, discrete = discrete, restart = restart)
        else:
            env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, bonus = bonus, scale = scale, reservoir_info = reservoir_info, continuous = continuous, dim_state = dim_state, discrete = discrete, restart = restart)
        callback = TrackExpectedRewardCallback(eval_env = env, eval_freq = eval_freq, log_dir = dir_name, n_eval_episodes= n_eval_episodes_callback)
        if(method == 'PPO'):
            model = PPO("MlpPolicy", env, verbose=0, policy_kwargs = policy_kwargs, learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = gamma, seed = seed, n_steps = n_steps, batch_size = batch_size, use_sde = use_sde, sde_sample_freq = 100)
        else:
            raise Exception("This method is not available")
        print('Begin training with seed {}'.format(seed))
        model.learn(total_timesteps= train_timesteps, callback = callback)
        print('End training with seed {}'.format(seed))
    elif Curriculum['type'] == 'auto':
        if reward_number == 1:
            env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, power = power, bonus = bonus, scale = scale, reservoir_info = reservoir_info, continuous = continuous, dim_state = dim_state, discrete = discrete, restart = restart)
        else:
            env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, bonus = bonus, scale = scale, reservoir_info = reservoir_info, continuous = continuous, dim_state = dim_state, discrete = discrete, restart = restart)
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs = policy_kwargs, learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = gamma, seed = seed, n_steps = n_steps, batch_size = batch_size, use_sde = use_sde)
        factor = Curriculum['factor']
        callback = TrackCurriculumCallback(eval_env = env, factor = factor, eval_freq = eval_freq, log_dir = dir_name, n_eval_episodes= n_eval_episodes_callback)
        train_timesteps = Curriculum['ts']
        env.reservoir_use = True
        env.reservoir_capacity = 50000
        model.learn(total_timesteps= train_timesteps, callback = callback)
    else:
        constraint = Curriculum['constraint']
        learning_rate = Curriculum['learning_rate']
        ts = Curriculum['ts']
        if reward_number == 1:
            env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, power = power, bonus = bonus, scale = scale, reservoir_info = reservoir_info, continuous = continuous, dim_state = dim_state, discrete = discrete, restart = restart)
        else:
            env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_lengthscale, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, bonus = bonus, scale = scale, reservoir_info = reservoir_info, continuous = continuous, dim_state = dim_state, discrete = discrete, restart = restart)
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs = policy_kwargs, gamma = gamma, seed = seed, n_steps = n_steps, batch_size = batch_size, use_sde = use_sde)
        callback = TrackExpectedRewardCallback(eval_env = env, eval_freq = eval_freq, log_dir = dir_name, n_eval_episodes= n_eval_episodes_callback)
        for i in range(len(constraint)):
            env.reservoir_use = True
            env.reservoir_capacity = constraint[i]
            model.learning_rate = learning_rate[i]
            model.learn(total_timesteps = ts[i], callback = callback)
        
    ## In the case of random restart training, we have to reconfigure starting point
    if restart == 'random':
        env.restart = 'fix'
        env.start = start
        env.initial_angle = initial_angle

    # Deterministic Path
    ep_reward = 0
    for _ in range(1):
        file_path = open(dir_name+'/deterministic_path.txt', 'w')
        ep_reward = 0
        obs = env.reset()
        file_path.write(str(env.state[1])+' '+str(env.state[2])+'\n')
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            file_path.write(str(env.state[1])+' '+str(env.state[2])+'\n')
            ep_reward += reward
            env.render()
            if done:
                break
        file_path.close()

        fig, axs = env.plot_trajectory(ep_reward, ref_trajectory_x = env_ref.trajectory_x, ref_trajectory_y = env_ref.trajectory_y, ref_energy= env_ref.energy, ref_time = env_ref.time)
        if save:
            plt.savefig(dir_name+'/deterministic_path.png')
        plt.close(fig)
        file1.write('Deterministic Path info: \n')
        file1.write('Cumulative Reward : {}; Timesteps : {}; Time : {}; Energy Consumed : {} \n'.format(ep_reward, len(env.time) - 1, round(env.time[-1], 1), round(env.energy[-1])))


    # Stochastic Paths
    for episode in range(10):
        ep_reward = 0
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            env.render()
            if done:
                break
        

        fig, axs = env.plot_trajectory(ep_reward, ref_trajectory_x = env_ref.trajectory_x, ref_trajectory_y = env_ref.trajectory_y, ref_energy= env_ref.energy, ref_time = env_ref.time)
        if save:
            plt.savefig(dir_name+'/stochastic_path_'+str(episode+1)+'.png')
        plt.close(fig)
        file1.write('Stochastic Path ({}/10) info: \n'.format(episode + 1))
        file1.write('Cumulative Reward : {}; Timesteps : {}; Time : {}; Energy Consumed : {} \n'.format(ep_reward, len(env.time) - 1, round(env.time[-1], 1), round(env.energy[-1])))

    
    if MonteCarlo:
        # Monte-Carlo estimator 
        file = open(dir_name+'/MonteCarlo.txt', 'w')
        for episode in range(500):
            ep_reward = 0
            obs = env.reset()
            for i in range(1000):
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                env.render()
                if done:
                    break
            file.write(str(ep_reward)+'\n')
        file.close()

    del model
    del env
    del env_ref
    if save:
        file1.close()
    print("Training with seed number {} took --- {} seconds ---".format(seed, time.time() - start_time))
        
    

        



