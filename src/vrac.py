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

wind_info = wind_info_2
discrete_maps = get_discrete_maps(wind_info['info'])
A = WindMap(discrete_maps, wind_info['lengthscale'])


start = (100, 900)
target = (800, 200)
initial_angle = 0
radius = 20
propulsion = 'variable'
eval_freq = 1000
    
    
ha = 'propulsion'
reward_number = 1
alpha = 15
dt = 4
gamma = 0.9
train_timesteps = 1000

env = WindEnv_gym(wind_maps = discrete_maps, wind_lengthscale= wind_info['lengthscale'], alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, dim_state = 3)

check_env(env)

model = SAC("MlpPolicy", env, verbose = 1, learning_rate=linear_schedule(0.001, 0.000005, 0.1), gamma = gamma, seed = 1)
model.learn(total_timesteps= train_timesteps)