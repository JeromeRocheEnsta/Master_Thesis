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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")




propulsion = 'variable'
ha = 'propulsion'
alpha = 15
start = (100, 900)
target = (800, 200)
radius = 30
dt = 4
initial_angle = 0



reward_number = 1
scale = 0.01
bonus = 10

reservoir_info = [False, None]

# Crete WindMap Object (Wind Field modelized with a GP)
discrete_maps = get_discrete_maps(wind_info_3)
A = WindMap(discrete_maps)

env = WindEnv_gym(wind_maps = discrete_maps, alpha = alpha, start = start, target= target, target_radius= radius, dt = dt, propulsion = propulsion, ha = ha, reward_number = reward_number, initial_angle=initial_angle, bonus = bonus, scale = scale, reservoir_info = reservoir_info)

grid = np.linspace(-1, 1, 50)

file = open('objective_function.txt', 'w')

for x1 in grid:
    for x2 in grid:
        for x3 in grid:
            theta = torch.Tensor([x1, x2, x3])
            performance = black_box_function(env, theta, 1)
            file.write('{} {} {} {}\n'.format(x1, x2, x3, performance))

file.close()

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(data[0], data[1], data[2], c=data[3], cmap='coolwarm')
fig.colorbar(img)
plt.savefig('objective_function.png')
"""