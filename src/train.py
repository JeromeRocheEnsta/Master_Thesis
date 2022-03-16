import os
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
from stable_baselines3 import SAC
from typing import Callable
from callback import TrackExpectedRewardCallback

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


def train(save = False, propulsion = 'variable', ha = 'propulsion', alpha = 15, reward_number = 1,
start = (100, 900), target = (800, 200), initial_angle = 0, radius = 20, dt = 1.8, gamma = 0.99,
train_timesteps = 150000, seed = 1, eval_freq = 1000, policy_kwargs = None, method = None):
    print("Execution de train avec seed = {}".format(seed))
    
    # MKDIR to stock figures output
    if save:
        dir_name = 'seed_'+str(seed)
        os.mkdir(dir_name)
        file1 = open(dir_name+'/info.txt', 'w')
        

    # Crete WindMap Object (Wind Field modelized with a GP)
    discrete_maps = get_discrete_maps(wind_info_2)
    A = WindMap(discrete_maps)

    # Save Visualization of the wind field
    fig = plot_wind_field(A, start, target)
    if save:
        plt.savefig(dir_name+'/wind_field.png')
    plt.close(fig)

    # reference path
    straight_angle = get_straight_angle(start, target)
    env_ref = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius=radius, dt = dt, straight = True, ha = 'next_state', propulsion = propulsion, reward_number= reward_number, initial_angle= straight_angle)
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
    del env_ref

    # train agent
    env = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle)
    check_env(env)
    callback = TrackExpectedRewardCallback(eval_env = env, eval_freq = eval_freq, log_dir = dir_name, n_eval_episodes= 5)
    if(method == 'PPO'):
        print('PPO')
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs = policy_kwargs, learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = gamma, seed = seed)
    elif(method == 'SAC'):
        print('SAC')
        model = SAC("MlpPolicy", env, verbose = 0, policy_kwargs = policy_kwargs, learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = gamma, seed = seed)
    model.learn(total_timesteps= train_timesteps, callback = callback)

    # Deterministic Path
    ep_reward = 0
    for _ in range(1):
        ep_reward = 0
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            env.render()
            if done:
                break

        fig, axs = env.plot_trajectory(ep_reward)
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
        

        fig, axs = env.plot_trajectory(ep_reward)
        if save:
            plt.savefig(dir_name+'/stochastic_path_'+str(episode+1)+'.png')
        plt.close(fig)
        file1.write('Stochastic Path ({}/10) info: \n'.format(episode + 1))
        file1.write('Cumulative Reward : {}; Timesteps : {}; Time : {}; Energy Consumed : {} \n'.format(ep_reward, len(env.time) - 1, round(env.time[-1], 1), round(env.energy[-1])))

    del model
    del env
    file1.close()

    plot_monitoring(dir_name+'/monitoring.txt')

        

#train(save = True, dir_name = 'truc', train_timesteps= 5000)

'''
######################
### Control interface
######################
propulsion = 'variable'
ha = 'propulsion'
alpha = 15
reward_number = 1
start = (100, 900)
target = (800, 200)
initial_angle = 0
radius = 20
dt = 1.8

gamma = 0.99



###############
#Output directory
###############
save = False

if len(sys.argv) > 1:
    save = True
    os.mkdir(sys.argv[1])

###############
#Get the discrete map from utils.py 's wind_info
###############

discrete_maps = get_discrete_maps(wind_info_2)

A = WindMap(discrete_maps)


######################
### Visualisation 
######################
fig = plot_wind_field(A, start, target)
if save:
    plt.savefig(sys.argv[1]+'/wind_field.png')
plt.show()



######################
### Reference trajectory
######################

straight_angle = get_straight_angle(start, target)
    

env_ref = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius=radius, dt = dt, straight = True, ha = 'next_state', propulsion = propulsion, reward_number= reward_number, initial_angle= straight_angle)
env_ref.reset()
reward_ref = 0
while env_ref._target() == False:
    obs, reward, done, info = env_ref.step(0)
    reward_ref += reward

##Plot trajectory
fig, axs = env_ref.plot_trajectory(reward_ref)

if save:
    plt.savefig(sys.argv[1]+'/ref_path.png')

plt.show()


del env_ref




######################
### Test the environment
######################

env = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle)
check_env(env)

######################
### PPO Agent
######################

model = PPO("MlpPolicy", env, verbose=1, learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = gamma, seed = 1)

model.learn(total_timesteps= 150000)


######################
### Enjoy the trained agent
######################

done_count = 0
ep_reward = 0
for _ in range(1):
    ep_reward = 0
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        env.render()
        if done:
            break
    if done:
        done_count += 1

    fig, axs = env.plot_trajectory(ep_reward)

    if save:
        plt.savefig(sys.argv[1]+'/deterministic_path.png')
    
    plt.show()


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
    if done:
        done_count += 1
    

    fig, axs = env.plot_trajectory(ep_reward)
    
    if save:
        plt.savefig(sys.argv[1]+'/stochastic_path_'+str(episode+1)+'.png')

    plt.show()

del model
del env
'''


