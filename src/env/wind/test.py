from numpy.core.defchararray import array
from wind_map import *
import numpy as np
import matplotlib.pyplot as plt

################
## 1. Mean
################

list_1 = [1, 10, 15, 5, 7]
list_2 = [3, 9, 10, 15, 5]
list_3 = [3, 3, 3, 3, 15]


map_1 = [(0, 0, 1), (0, 0, 10), (0, 0, 15), (0, 0, 5), (0, 0, 7)]
map_2 = [(0, 0, 3), (0, 0, 9), (0, 0, 10), (0, 0, 15), (0, 0, 5)]
map_3 = [(0, 0, 3), (0, 0, 3), (0, 0, 3), (0, 0, 3), (0, 0, 15)]

lists = [list_1, list_2, list_3]
maps = [map_1, map_2, map_3]

test = True
for i in range(3):
    if (float(np.mean(lists[i])) != float(mean(maps[i]))):
        test = False

print('Test mean: OK') if test else print('Test mean: NOT OK')

################
## 2. BAYESIAN REGRESSION
################

## get_params

discrete_maps =[ [(0, 0, 0), (1, 0, 2), (3, 0, 1)], [(0, 0, 0)]  ]
A = WindMap(discrete_maps, 500)

params = A.magnitude_params
true_params = [1, np.dot( np.linalg.inv([[1, np.exp(-0.5), np.exp(-9/2)],[np.exp(-0.5), 1, np.exp(-4/2)],[np.exp(-9/2), np.exp(-4/2), 1]]), np.array([-1, 1, 0], dtype= float) )]

test = True
if(params[0] != true_params[0]):
    test = False
comparison = params[1] == true_params[1]
if not(comparison.all()):
    test = False
print('Test Bayesian Regression: OK') if test else print('Test Bayesian Regression: NOT OK')

## predict

test = True
localisation = [(2, 0)]
prediction = A._get_magnitude(localisation)
alpha = true_params[1]
kappa = np.transpose(np.array([np.exp(-4/2), np.exp(-1/2), np.exp(-1/2)]))
true_value = np.dot( alpha, kappa )
true_value = [true_value]
for i in range(len(true_value)):
    true_value[i] += true_params[0]

comparison = true_value == prediction
if not(comparison.all()):
    test = False
print('Test Bayesian Regression prediction: OK') if test else print('Test Bayesian Regression prediction: NOT OK')


## plot
localisation = []
X = []
for i in range(250):
    localisation.append((-1 + i*1/50, 0))
    X.append(-1 + i*1/50)
prediction = A._get_magnitude(localisation)

plt.plot(X, prediction)
plt.show()

##plot(2D)
discrete_maps = [[(0, 0, 5), (500, 0, 10), (1000, 0, 15), (0, 500, 5), (500, 500, 5), (1000, 500, 10), (0, 1000, 5), (500, 1000, 5), (1000, 1000, 5)], [(0,0,0)]]
A = WindMap(discrete_maps, 500)

localisation = []
X = []
Y = []
for i in range(11):
    X.append(i * 1000/10)
    Y.append(i* 1000/10)
    for j in range(11):
        localisation.append( (i* 1000/10, j*1000/10) )

prediction = A._get_magnitude(localisation)
Z = np.zeros( (len(Y), len(X)) )
for i in range(len(prediction)):
    row = i % len(Y)
    col = i // len(Y)
    Z[row, col] = prediction[i]



plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.show()

