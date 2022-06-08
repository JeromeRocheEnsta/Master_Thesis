import os
import math
from dataclasses import dataclass

import gym
from env.wind_env import *
from env.wind.wind_map import *
from utils import *

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")



####################
##### Wind Env #####
####################


def train( log_kwargs = {'save' : False},
environment_kwargs = {'propulsion' : 'variable', 'ha' : 'propulsion', 'alpha' : 15, 'start' : (100, 900),
'target' : (800, 200), 'radius' : 20, 'dt' : 1.8, 'initial_angle' : 0},
model_kwargs = {'gamma' : 0.99, 'n_eval_episodes' : 1000, 'dim' : 4,
'bounds' : torch.tensor([ [-1, -1, -1, -1] , [1, 1, 1, 1] ], dtype = torch.float64),
'batch_size' : 4}, #method within qEI, Sobol, TuRBO
reward_kwargs = {'reward_number' : 1, 'scale' : 1, 'bonus': 10},
constraint_kwargs = {'reservoir_info' : [False, None]},
seed = 0):
    ### Preprocess args
    
    torch.manual_seed(seed)
    #np.random.seed(seed)
    
    # log_kwargs
    save = log_kwargs['save']

    #environment_kwargs
    propulsion = environment_kwargs['propulsion']
    ha = environment_kwargs['ha']
    alpha = environment_kwargs['alpha']
    start = environment_kwargs['start']
    target = environment_kwargs['target']
    radius = environment_kwargs['radius']
    dt = environment_kwargs['dt']
    initial_angle = environment_kwargs['initial_angle']

    # model_kwargs
    gamma = model_kwargs['gamma']
    n_eval_episodes = model_kwargs['n_eval_episodes']
    dim = model_kwargs['dim']
    bounds = model_kwargs['bounds']
    batch_size = model_kwargs['batch_size']
    n_init = model_kwargs['n_init'] if model_kwargs.get('n_init') != None else 2*dim

    #rewrd_kwargs
    reward_number = reward_kwargs['reward_number'] if reward_kwargs.get('reward_number') != None else 1
    scale = reward_kwargs['scale'] if reward_kwargs.get('scale') != None else 1
    bonus = reward_kwargs['bonus'] if reward_kwargs.get('bonus') != None else 10

    #constraint_kwargs
    reservoir_info = constraint_kwargs['reservoir_info'] if constraint_kwargs.get('reservoir_info') != None else [False, None]

    if save:
        dir_name = 'seed_'+str(seed)
        os.mkdir(dir_name)

    # Crete WindMap Object (Wind Field modelized with a GP)
    discrete_maps = get_discrete_maps(wind_info)
    A = WindMap(discrete_maps)
    # Save Visualization of the wind field
    fig = plot_wind_field(A, start, target)
    if save:
        plt.savefig(dir_name+'/wind_field.png')
    plt.close(fig)


    # reference path
    straight_angle = get_straight_angle(start, target)
    env_ref = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius=radius, dt = dt, straight = True, ha = 'next_state', propulsion = propulsion, reward_number= reward_number, initial_angle= straight_angle, bonus = bonus, scale = scale)
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


    #BO for RL 
    env = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, bonus = bonus, scale = scale, reservoir_info = reservoir_info)

    X_turbo, Y_turbo = TuRBO(env, dim, n_init, bounds, n_eval_episodes, batch_size)
    n_iter = len(X_turbo)
    X_ei, Y_ei = qEI(env, dim, bounds, n_eval_episodes, n_init, batch_size, n_iter)
    X_Sobol, Y_Sobol = Sobol(env, dim, bounds, n_eval_episodes, n_iter)



    # Visualization Search

    names = ["TuRBO-1", "EI", "Sobol"]
    runs = [Y_turbo, Y_ei, Y_Sobol]
    fig, ax = plt.subplots(figsize = (8,8))

    for name, run in zip(names, runs):
        fx = np.maximum.accumulate(run.cpu())
        ax.plot(fx, marker="", lw=3)

    ax.plot([0, len(Y_turbo)], [reward_ref, reward_ref], "k--", lw=3)
    ax.set_ylabel("Function value", fontsize=18)
    ax.set_xlabel("Number of evaluations", fontsize=18)
    ax.set_title("Wind Environment", fontsize=24)
    ax.set_xlim([0, len(Y_turbo)])
    #plt.ylim([-15, 600])

    ax.grid(True)
    fig.tight_layout()
    lgd = ax.legend(
        names + ["Straight Path"],
        loc="lower center",
        bbox_to_anchor=(0, -0.08, 1, 1),
        bbox_transform=plt.gcf().transFigure,
        ncol=4,
        fontsize=16,
    )
    plt.savefig(dir_name+'/methods.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

    # Visualization best paths
    ## Turbo
    thetas = [unnormalize(X_turbo[int(torch.max(Y_turbo, 0)[1])], bounds) , unnormalize(X_ei[int(torch.max(Y_ei, 0)[1])], bounds), unnormalize(X_Sobol[int(torch.max(Y_Sobol, 0)[1])], bounds)]
    for name, theta in zip(names, thetas):
        print(black_box_function(env, theta, n_eval_episodes))
        Cumulative_reward = 0
        s = env.reset()
        for _ in range(1000):
            s = torch.from_numpy(s).reshape(3,1)
            s = s.type('torch.DoubleTensor')
            action = policy_BO(theta, s)
            s, reward, done, info = env.step(action)
            Cumulative_reward += reward
            if done:
                break
        ##Plot trajectory
        fig, axs = env.plot_trajectory(Cumulative_reward, ref_trajectory_x = env_ref.trajectory_x, ref_trajectory_y = env_ref.trajectory_y, ref_energy = env_ref.energy,  ref_time = env_ref.time)
        plt.savefig(dir_name+'/trajectory_'+name+'.png')
        plt.close(fig)
    
    ## Write infos
    file_turbo = open(dir_name+'/turbo.txt', 'w')
    for i in range(len(X_turbo)):
        file_turbo.write('{} {}'.format(X_turbo[i], Y_turbo[i]))
    file_turbo.close()

    file_EI = open(dir_name+'/EI.txt', 'w')
    for i in range(len(X_ei)):
        file_EI.write('{} {}'.format(X_ei[i], Y_ei[i]))
    file_EI.close()

    file_Sobol = open(dir_name+'/Sobol.txt', 'w')
    for i in range(len(X_Sobol)):
        file_Sobol.write('{} {}'.format(X_Sobol[i], Y_Sobol[i]))
    file_Sobol.close()

    
    del env
    del env_ref