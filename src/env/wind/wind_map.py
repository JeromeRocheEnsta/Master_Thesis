import numpy as np
from scipy.linalg import cho_factor, cho_solve

def KBF(X, Y):
    K = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            K[i][j] = np.exp(-((X[i][0] - Y[j][0])**2 + (X[i][1] - Y[j][1])**2)/(2*50**2))
    return K

def mean(map):
    s = 0
    compteur = 0
    for i in range(len(map)):
        s+= map[i][2]
        compteur += 1
    s/= compteur
    return s

class WindMap:
    def __init__(self, discrete_maps):
        self.discrete_maps = discrete_maps
        self.magnitude_params = self._get_magnitude_params()
        self.direction_params = self._get_direction_params()

    def _get_magnitude_params(self):
        m = mean(self.discrete_maps[0])
        C = KBF(self.discrete_maps[0], self.discrete_maps[0])
        l = len(self.discrete_maps[0])
        normalization = np.zeros(l)
        for i in range(l):
            normalization[i] = self.discrete_maps[0][i][2] - m
        L, low = cho_factor(C)
        alpha = np.asarray(cho_solve((L, low), normalization))
        
        return(m, alpha)


    def _get_direction_params(self):
        m = mean(self.discrete_maps[1])
        C = KBF(self.discrete_maps[1], self.discrete_maps[1])
        l = len(self.discrete_maps[1])
        normalization = np.zeros(l)
        for i in range(l):
            normalization[i] = self.discrete_maps[1][i][2] - m
        L, low = cho_factor(C)
        alpha = np.asarray(cho_solve((L, low), normalization))
        
        return(m, alpha)

    def _get_magnitude(self, X):
        ### X is of the form [(x1, y1), (x2, y2), ...]
        Kappa = KBF( X , self.discrete_maps[0])
        m, alpha = self.magnitude_params
        b = np.transpose(Kappa)
        magnitude =  np.asarray(np.dot(alpha, b))
        for i in range(len(magnitude)):
            magnitude[i] += m
        return magnitude

    def _get_direction(self, X):
        Kappa = KBF( X , self.discrete_maps[1])
        m, alpha = self.direction_params
        b = np.transpose(Kappa)
        direction = np.asarray(np.dot(alpha, b))
        for i in range(len(direction)):
            direction[i] += m
        return direction

