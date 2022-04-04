import matplotlib.pyplot as plt
import numpy as np

#get data
trajectory_x = []
trajectory_y = []
file = open('Exp_exploration/exploration_spaces.txt', 'r')
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
plt.savefig('Exp_exploration/exploration_space.png')