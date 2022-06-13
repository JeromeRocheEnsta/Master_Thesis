from dijkstar import Graph, find_path
import numpy as np
from utils import *
from env.utils import energy
from env.wind_env import *
from env.wind.wind_map import *


# graph.add_edge(1, 2, 110)
# graph.add_edge(2, 3, 125)
# graph.add_edge(3, 4, 108)
# print(find_path(graph, 1, 4))

start = (100, 900)
target = (800, 200)

mu = 20 / 3.6 # m/s
dt = 4

def number_to_coordinate(x, m):
    column = x % m
    row = x // m
    x1 = 1000/(2*m) + column * 1000/m
    y1 = (1000 * (2*m-1)/(2*m)) - row * 1000/m
    return x1, y1

def cost_function(wind_map, x, m, info):
    x1, y1 = number_to_coordinate(x, m)
    magnitude = wind_map._get_magnitude([(x1, y1)])
    direction = wind_map._get_direction([(x1, y1)])
    if info == 'right':
        angle = 0
    elif info == 'left':
        angle = 180
    elif info == 'top':
        angle = 90
    elif info == 'bottom':
        angle = 270
    v_prop = np.sqrt(( - magnitude * np.cos(direction * np.pi / 180) + mu *np.cos(angle* np.pi / 180))**2 + (( - magnitude * np.sin(direction * np.pi / 180) + mu *np.sin(angle* np.pi / 180))**2))
    print(magnitude, direction, v_prop)
    return energy(v_prop, mu)

def add_top(wind_map, x, m):
    next = x - m
    cost = cost_function(wind_map, x, m, 'top')
    return next , cost

def add_bottom(wind_map, x, m):
    next = x + m
    cost = cost_function(wind_map, x, m, 'bottom')
    return next , cost

def add_right(wind_map, x, m):
    next = x + 1
    cost = cost_function(wind_map, x, m, 'right')
    return next , cost

def add_left(wind_map, x, m):
    next = x - 1
    cost = cost_function(wind_map, x, m, 'left')
    return next , cost

if __name__ == "__main__":
    ## Wind map
    discrete_maps = get_discrete_maps(wind_info_1)
    wind_map = WindMap(discrete_maps)
    ## Find m
    m = int(1000/(mu * dt)) # to ensure relative velocity of mu * dt
    
    ### Create Graph
    graph = Graph()
    ## Create nodes
    nodes = [i for i in range(m*m)]
    ## Create edges (with edge's length given by energy consumed)
    for x in nodes:
        print(x)
        # particular cases
        if(x == 0): # corner left top
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x == m-1): # corner right top
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x // m == 0): # top edge
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x == m*m - 1): # corner right bottom
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x % m == m-1): # right edge
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x == m*(m-1)): # corner left bottom
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x // m == m-1): # bottom edge
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x % m == 0): # left edge
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next , cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
        else: # all other edges
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next , cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next , cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)

    print('m = {}\n'.format(m))
    print('\n\n The best Path is :\n')
    ##Find starting point
    row = m - 1 - int(m * start[1]/1000)
    column = int(m * start[0]/1000)
    idx_start = row * m + column
    print(column, row, idx_start)
    ## Find target point
    row = m - 1 - int(m * target[1]/1000)
    column = int(m * target[0]/1000)
    idx_target = row * m + column
    print(column, row)

    best_path = find_path(graph, idx_start, idx_target)
    print(best_path)

    ## Post processing
