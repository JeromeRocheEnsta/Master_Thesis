from train import train

if __name__ == "__main__":
    #########################
    ### Control interface ###
    #########################
    seed = 1
    start = (100, 900)
    target = (800, 200)
    initial_angle = 0
    radius = 20
    propulsion = 'variable'
    
    ha = ['propulsion', 'next_state']
    alpha = [10, 15, 20, 40, 50]
    reward_number = [1, 2]
    dt = [1, 1.8, 2, 3, 4]
    gamma = [0.9, 0.99, 1]
    train_timesteps = 150000

    for reward_number in reward_number:
        for ha in ha:
            for alpha in alpha:
                for dt in dt:
                    for gamma in gamma:
                        name = 'png_'+str(reward_number)+'_'+ha+'_'+str(alpha)+'_'+str(dt)+'_'+str(gamma)+'_'+str(train_timesteps)
                        train(save = True, dir_name = name, propulsion = propulsion, ha = ha, alpha = alpha, reward_number = reward_number,
                        start = start, target = target, initial_angle = initial_angle, radius = radius, dt = dt, gamma = gamma, train_timesteps = 150000, seed = seed)

