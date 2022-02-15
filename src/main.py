import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from env.wind_env import *
from env.wind.wind_map import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from typing import Callable

gym.logger.set_level(40)


###############
#Output directory
###############
save = False

if len(sys.argv) > 1:
    save = True
    os.mkdir(sys.argv[1])

###############
#get the environment ready
###############

discrete_maps = [[], []]

wind_info = [
    [(5, 350),(5, 315),(10, 290),(10, 270),(15, 260),(15, 250),(15, 250)],
    [(5, 350),(10, 340),(10, 315),(10, 300),(15, 260),(15, 260),(15, 240)],
    [(5, 0),(5, 350),(10, 315),(10, 300),(15, 270),(15, 250),(15, 250)],
    [(5, 0),(5, 355),(10, 340),(10, 315),(15, 280),(10, 270),(15, 260)],
    [(5, 0),(5, 0),(10, 350),(5, 330),(5, 300),(15, 280),(15, 270)],
    [(5, 0),(5, 0),(5, 350),(5, 335),(5, 290),(10, 280),(15, 270)],
    [(5, 0),(5, 350),(5, 340),(5, 330),(5, 315),(10, 280),(15, 270)]
]

wind_info_2 = [[(15, 270)]*7]*7


for i in range (49):
    row = i//7
    col =  i%7
    discrete_maps[0].append((col * 1000/6, 1000 - row * 1000/6, wind_info_2[row][col][0]))
    discrete_maps[1].append((col * 1000/6, 1000 - row * 1000/6, wind_info_2[row][col][1]))

A = WindMap(discrete_maps)

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


######################
### Visualisation 
######################

localisation = []
X = []
Y = []
for i in range(101):
    X.append(i * 1000/100)
    Y.append(i* 1000/100)
    for j in range(101):
        localisation.append( (i* 1000/100, j*1000/100) )

prediction_magnitude = A._get_magnitude(localisation)
prediction_direction = A._get_direction(localisation)
Z_magnitude = np.zeros( (len(Y), len(X)) )
# Z_direction = np.zeros( (len(Y), len(X)) )
for i in range(len(prediction_magnitude)):
    row = i % len(Y)
    col = i // len(Y)
    Z_magnitude[row, col] = prediction_magnitude[i]
    # Z_direction[row, col] = prediction_direction[i]

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z_magnitude, 20, cmap='coolwarm')
plt.colorbar()
plt.plot(start[0], start[1], 'ko', markersize = 10)
plt.plot(target[0], target[1], 'k*', markersize = 10)


localisation = []
X = []
Y = []
for i in range(21):
    for j in range(21):
        X.append(i * 1000/20)
        Y.append(j* 1000/20)
        localisation.append( (i* 1000/20, j*1000/20) )

prediction_magnitude = A._get_magnitude(localisation)
prediction_direction = A._get_direction(localisation)
U = []
V = []
for i in range(len(prediction_magnitude)):
    U.append(prediction_magnitude[i]/20 * np.cos(prediction_direction[i] * np.pi /180))
    V.append(prediction_magnitude[i]/20 * np.sin(prediction_direction[i] * np.pi /180))


plt.quiver(X, Y, U, V)

if save:
    plt.savefig(sys.argv[1]+'/wind_field.png')


plt.show()

'''
X = np.linspace(0, 1000, 1000)
Y = np.linspace(0, 1000, 1000)
Z = np.zeros( (len(Y), len(X)) )
for i in range(len(X)):
    for j in range(len(Y)):
        if( ( (X[i]-target[0])**2 + (Y[j] - target[1])**2 > radius**2)):
            Z[j][i] = (-  ( np.sqrt( (np.sqrt((X[i] - target[0])**2 + (Y[j] - target[1])**2) - radius) ))/(np.sqrt(np.sqrt(2)*1000)) )
        else:
            Z[j][i] = 1

plt.contourf(X, Y, Z, 40, cmap='RdGy')
plt.colorbar()
plt.show()
'''

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
# It will check your custom environment and output additional warnings if needed
check_env(env)

######################
### PPO Agent
######################

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
    
    if plot:
        plt.savefig(sys.argv[1]+'/stochastic_path_'+str(episode+1)+'.png')

    plt.show()

print(done_count)


del model
del env


