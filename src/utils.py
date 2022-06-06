from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from env.wind.wind_map import WindMap
import scipy.stats as st

import os
import math
from dataclasses import dataclass


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")





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

wind_info_3 = [
    [(10, 0),(10, 0),(10, 0),(10, 0),(10, 0),(10, 0),(10, 0),(10, 0),(10, 0),(10, 0),(10, 0)],
    [(10, 90),(15, 0),(15, 0),(15, 0),(15, 0),(10, 0),(10, 350),(10, 315),(10, 320),(5, 320),(5, 200)],
    [(10, 90),(15, 90),(15, 135),(15, 120),(15, 90),(15, 60),(10, 350),(15, 315),(10, 300),(5, 300),(5, 200)],
    [(10, 90),(15, 100),(15, 135),(15, 135),(15, 135),(10, 90),(10, 280),(15, 280),(10, 280),(5, 280),(5, 200)],
    [(10, 90),(10, 100),(15, 180),(15, 135),(15, 135),(10, 135),(5, 300),(15, 280),(10, 270),(5, 270),(5, 200)],
    [(10, 90),(10, 120),(15, 140),(15, 135),(15, 135),(10, 135),(10, 230),(15, 270),(10, 270),(5, 270),(5, 200)],
    [(10, 90),(10, 120),(15, 140),(15, 135),(15, 135),(15, 135),(10, 230),(15, 270),(10, 270),(5, 270),(5, 200)],
    [(10, 90),(10, 120),(10, 140),(10, 135),(15, 135),(10, 135),(10, 230),(15, 270),(10, 270),(5, 270),(5, 200)],
    [(10, 90),(10, 120),(10, 140),(10, 135),(10, 135),(10, 180),(10, 230),(15, 270),(10, 270),(5, 270),(5, 200)],
    [(10, 90),(15, 130),(15, 150),(15, 180),(15, 200),(15, 230),(15, 250),(15, 270),(10, 270),(5, 270),(5, 200)],
    [(10, 90),(10, 180),(10, 180),(10, 180),(10, 180),(10, 200),(10, 250),(10, 270),(10, 270),(5, 200),(5, 200)],
]


def plot_wind_field(Wind, start, target):
    localisation = []
    X = []
    Y = []
    for i in range(101):
        X.append(i * 1000/100)
        Y.append(i* 1000/100)
        for j in range(101):
            localisation.append( (i* 1000/100, j*1000/100) )

    prediction_magnitude = Wind._get_magnitude(localisation)
    prediction_direction = Wind._get_direction(localisation)
    Z_magnitude = np.zeros( (len(Y), len(X)) )
    # Z_direction = np.zeros( (len(Y), len(X)) )
    for i in range(len(prediction_magnitude)):
        row = i % len(Y)
        col = i // len(Y)
        Z_magnitude[row, col] = prediction_magnitude[i]
        # Z_direction[row, col] = prediction_direction[i]

    fig = plt.figure(figsize=(6, 6))
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

    prediction_magnitude = Wind._get_magnitude(localisation)
    prediction_direction = Wind._get_direction(localisation)
    U = []
    V = []
    for i in range(len(prediction_magnitude)):
        U.append(prediction_magnitude[i]/20 * np.cos(prediction_direction[i] * np.pi /180))
        V.append(prediction_magnitude[i]/20 * np.sin(prediction_direction[i] * np.pi /180))


    plt.quiver(X, Y, U, V)
    return fig


def get_discrete_maps(wind_info):
    discrete_maps = [[], []]
    div = len(wind_info)
    for i in range (div*div):
        row = i//div
        col =  i%div
        discrete_maps[0].append((col * 1000/(div-1), 1000 - row * 1000/(div-1), wind_info[row][col][0]))
        discrete_maps[1].append((col * 1000/(div-1), 1000 - row * 1000/(div-1), wind_info[row][col][1]))

    return discrete_maps


def get_straight_angle(start, target):
    angle = np.arctan(abs(start[1] - target[1])/abs(start[0] - target[0])) * 180 / np.pi
    ## Case 1
    if(start[0] < target[0] and start[1] > target[1]):
        straight_angle = 360 - angle
    ## Case 2
    elif(start[0] > target[0] and start[1] > target[1]):
        straight_angle = 180 + angle
    ## Case 3
    elif(start[0] > target[0] and start[1] < target[1]):
        straight_angle = 180 - angle
    ## Case 4
    elif(start[0] < target[0] and start[1] < target[1]):
        straight_angle = angle
    ## Case 5
    elif(start[0] == target[0]):
        straight_angle = 90 if start[1] < target[1] else 270
    elif(start[1] == target[1]):
        straight_angle = 0 if start[0] < target[0] else 180

    return straight_angle


    

        


def plot_monitoring(file: str, log_dir = None):
    file = open(file, 'r')
    timesteps = []
    expected_rewards = []
    expected_lengths = []
    expected_energies = []
    for line in file:
        values = line.split()
        timesteps.append(int(values[0]))
        expected_rewards.append(float(values[1]))
        expected_lengths.append(float(values[2]))
        expected_energies.append(float(values[3]))

    plt.plot(timesteps, expected_rewards, '-')
    plt.xlabel('Timesteps')
    plt.ylabel('Expected Return')
    plt.show()
    plt.plot(timesteps, expected_lengths, '-')
    plt.xlabel('Timesteps')
    plt.ylabel('Expected Length')
    plt.show()
    plt.plot(timesteps, expected_energies, '-')
    plt.xlabel('Timesteps')
    plt.ylabel('Expected Energy')
    plt.show()

def filter():
    pass


def get_data(seeds, scale = None, bonus = None):
    ## Need the current path to be in the directory containing the seeds directory
    n = len(seeds)
    mean_reward = []
    mean_length = []
    mean_energy = []
    if (n > 1):
        ci_reward = []
        ci_length = []
        ci_energy = []
    timesteps = []

    ## Extract ref paths info

    file_ref_path = open('seed_1/info.txt', 'r')
    file_ref_path.readline()
    ref_line = file_ref_path.readline()
    ref_line = ref_line.split()
    ref_reward = float(ref_line[3][:-1]) if bonus == None else float(ref_line[3][:-1]) - bonus
    ref_reward = ref_reward if scale == None else ref_reward/scale
    ref_length = float(ref_line[6][:-1])
    ref_energy = float(ref_line[13])
    file_ref_path.close()

    file_timestep = open('seed_1/monitoring.txt', 'r')
    ts= 0
    for _ in file_timestep:
        ts+=1
    file_timestep.close()


    for seed in seeds:
        exec('file'+str(seed)+'=open("seed_'+str(seed)+'/monitoring.txt", "r")')

    
    for _ in range(ts):
        count = 1
        rewards =[]
        lengths = []
        energies = []
        for seed in seeds:
            print(seed)
            exec('line = file'+str(seed)+'.readline()')
            exec('line = line.split()')
            
            if seed == 1:
                exec('print(line)')
                exec('timesteps.append(float(line[0]))')

            if scale == None:
                if bonus == None:
                    exec('rewards.append(float(line[1]))')
                else:
                    exec('rewards.append(float(line[1])-bonus)')
            else:
                if bonus == None:
                    exec('rewards.append(float(line[1])/scale)')
                else:
                    exec('rewards.append((float(line[1])-bonus)/scale)')
            exec('lengths.append(float(line[2]))')
            exec('energies.append(float(line[3]))')
            count += 1

        mean_reward.append(np.mean(rewards))
        mean_length.append(np.mean(lengths))
        mean_energy.append(np.mean(energies))
        if (n > 1) :
            t = st.t.ppf(0.975, n-1)
            ci_reward.append(t*np.sqrt(np.var(rewards)/n))
            ci_length.append(t*np.sqrt(np.var(lengths)/n))
            ci_energy.append(t*np.sqrt(np.var(energies)/n))

    for seed in seeds:
        exec('file'+str(seed)+'.close()')
    
    if(n > 1):
        return (timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy, ref_reward, ref_length, ref_energy)
    else:
        return(timesteps, mean_reward, mean_length, mean_energy, ref_reward, ref_length, ref_energy)


##################################
##################################
## Function for Bayesian Opt RL ##
################################## 
##################################


def policy_BO(theta, s, deterministic = True, sigma = 0.05):
    '''
    input : two torch.tensor respectively the policy parameters and the state
    output : action
    '''
    if deterministic:
        return np.tanh(float(torch.matmul(theta, s)))
    else:
        epsilon = np.random.normal(0, sigma)
        return np.tanh(float(torch.matmul(theta, s))) + epsilon


def black_box_function(env, theta, n_eval_episodes):
    Expected_reward = 0
    for i in range(n_eval_episodes):
        state = env.reset()
        #img = plt.imshow(env.render(mode='rgb_array'))
        ep_reward = 0
        for _ in range(1000):
            #img.set_data(env.render(mode='rgb_array'))
            #display.display(plt.gcf())
            #display.clear_output(wait=True)

            s = torch.from_numpy(state).reshape(3,1)
            s = s.type('torch.DoubleTensor')
            theta = theta.type('torch.DoubleTensor')
            action = policy_BO(theta, s)

            state, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        Expected_reward += ep_reward
    Expected_reward /= n_eval_episodes
    return Expected_reward

def eval_objective(env, x, bounds, n_eval_episodes):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return black_box_function(env, unnormalize(x, bounds), n_eval_episodes = n_eval_episodes)
    #return fun(unnormalize(x, fun.bounds))



@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True) ### RANDOM ###
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )  ### RANDOM ###
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)   ### RANDOM ### ??
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

def TuRBO(env, dim, n_init, bounds, n_eval_episodes, batch_size):
    max_cholesky_size = float("inf") 
    X_turbo = get_initial_points(dim, n_init)
    Y_turbo = torch.tensor(
        [eval_objective(env, x, bounds, n_eval_episodes) for x in X_turbo], dtype=dtype, device=device
    ).unsqueeze(-1)
    print(X_turbo, Y_turbo, bounds)

    state = TurboState(dim, batch_size=batch_size)

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4


    while not state.restart_triggered:  # Run until TuRBO converges
        # Fit a GP model
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_model(mll)
        
            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf="ts",
            )

        Y_next = torch.tensor(
            [eval_objective(env, x, bounds, n_eval_episodes) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Update state
        state = update_state(state=state, Y_next=Y_next)

        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        # Print current status
        print(
            f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )

    return X_turbo, Y_turbo

def qEI(env, dim, bounds, n_eval_episodes, n_init, batch_size, n_iter):
    X_ei = get_initial_points(dim, n_init)
    Y_ei = torch.tensor(
        [eval_objective(env, x, bounds, n_eval_episodes) for x in X_ei], dtype=dtype, device=device
    ).unsqueeze(-1)

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4

    while len(Y_ei) < n_iter:
        train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        model = SingleTaskGP(X_ei, train_Y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # Create a batch
        ei = qExpectedImprovement(model, train_Y.max(), maximize=True)
        candidate, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack(
                [
                    torch.zeros(dim, dtype=dtype, device=device),
                    torch.ones(dim, dtype=dtype, device=device),
                ]
            ),
            q=batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        Y_next = torch.tensor(
            [eval_objective(env, x, bounds, n_eval_episodes) for x in candidate], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        X_ei = torch.cat((X_ei, candidate), axis=0)
        Y_ei = torch.cat((Y_ei, Y_next), axis=0)

        # Print current status
        print(f"{len(X_ei)}) Best value: {Y_ei.max().item():.2e}")

    return X_ei, Y_ei

def Sobol(env, dim, bounds, n_eval_episodes, n_iter):
    X_Sobol = SobolEngine(dim, scramble=True, seed=0).draw(n_iter).to(dtype=dtype, device=device)
    Y_Sobol = torch.tensor([eval_objective(env, x, bounds, n_eval_episodes) for x in X_Sobol], dtype=dtype, device=device).unsqueeze(-1)

    return X_Sobol, Y_Sobol