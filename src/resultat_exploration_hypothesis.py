import matplotlib.pyplot as plt
import numpy as np

#get data
trajectory_x = []
trajectory_y = []
file = open('log_files/wind_map_1/Exp_exploration/PPO_continuous_7_4_0_1_propulsion_15_4_1_150000/exploration_spaces.txt', 'r')
i = 0
for line in file:
    info = line.split()
    print(i)
    trajectory_x.append(float(info[0]))
    trajectory_y.append(float(info[1]))
    i+=1
file.close()

#Built the plot
plt.scatter(trajectory_x, trajectory_y, s = np.pi * 5**2, alpha = 0.01)
plt.xlim((0,1000))
plt.ylim((0,1000))
#plt.plot(trajectory_x, trajectory_y)
plt.savefig('log_files/wind_map_1/Exp_exploration/PPO_continuous_7_4_0_1_propulsion_15_4_1_150000/exploration_space.png')