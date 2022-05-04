from train import train
import os
import multiprocessing
import torch as th


if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 5, 'eval_freq' : 5000}

    environment_kwargs = {
        'propulsion' : 'variable',
        'ha' : 'propulsion',
        'alpha' : 15,
        'start' : (100, 900),
        'target' : (800, 200),
        'radius' : 30,
        'dt' : 4,
        'initial_angle' : 0
    }
    
    model_kwargs = {
        'gamma' : 0.90,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 1000,
        'method' : 'PPO',
        'n_steps' : 2048,
        'batch_size' : 64
    }


    reward_kwargs = {
        'reward_number' : 1,
        'scale' : 0.01,
        'bonus': 10
    }

    constraint_kwargs = {
        'reservoir_info' : [False, None]
    }

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    n_steps = [1024, 2048, 4096, 8192]
    batch_size = [64, 128, 256]

    if not os.path.exists('Exp_batch'):
        os.mkdir('Exp_batch')
    os.chdir('Exp_batch')
    
    name = model_kwargs['method']+'_'+str(reward_kwargs['reward_number'])+'_'+str(reward_kwargs['bonus'])+'_'+str(reward_kwargs['scale'])+'_'+environment_kwargs['ha']+'_'+str(environment_kwargs['alpha'])+'_'+str(environment_kwargs['dt'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])
    
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for n in n_steps:
        for batch in batch_size:
            log = str(n)+'_'+str(batch)

            model_kwargs['n_steps'] = n
            model_kwargs['batch_size'] = batch

            os.mkdir(log)
            os.chdir(log)

            #Multi Porcessing
            processes = [multiprocessing.Process(target = train, args = [log_kwargs, environment_kwargs, model_kwargs, reward_kwargs, constraint_kwargs, seed]) for seed in seeds]

            for process in processes:
                process.start()
            for process in processes:
                process.join()

            os.chdir('../')
                        
                        
    os.chdir('../')