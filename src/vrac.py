import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(start = 0, stop = 40, num = 80)
Y = []
for x in X:
    Y.append((x/20)**3)

plt.plot(X, Y)
plt.show()