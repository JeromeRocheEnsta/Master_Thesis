from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from env.wind.wind_map import WindMap
import scipy.stats as st




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
    for i in range (49):
        row = i//7
        col =  i%7
        discrete_maps[0].append((col * 1000/6, 1000 - row * 1000/6, wind_info_2[row][col][0]))
        discrete_maps[1].append((col * 1000/6, 1000 - row * 1000/6, wind_info_2[row][col][1]))

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

def get_data(seeds):
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

    for seed in seeds:
        exec('file'+seed+'=open(seed_'+seed+'/monitoring.txt, \'r\') ')

    info = True
    while info:
        count = 1
        rewards =[]
        lengths = []
        energies = []
        for seed in seeds:
            exec('line = file'+seed+'.readline()')
            if not line:
                info = False
                break
            line = line.split()
            if count == 1:
                timesteps.append(int(line[0]))
            rewards.append(line[1])
            lengths.append(line[2])
            energies.append(line[3])
            count += 1
        mean_reward.append(np.mean(rewards))
        mean_length.append(np.mean(lengths))
        mean_energy.append(np.mean(energies))
        if (n > 1) :
            t = st.t.ppf(0.975, n-1)
            ci_reward.append(t*np.sqrt(np.var(rewards)/n))
            ci_length.append(t*np.sqrt(np.var(lengths)/n))
            ci_energy.append(t*np.sqrt(np.var(energies)/n))
    
    if(n > 1):
        return (timesteps, mean_reward, ci_reward, mean_length, ci_length, mean_energy, ci_energy)
    else:
        return(timesteps, mean_reward, mean_length, mean_energy)
    

        


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