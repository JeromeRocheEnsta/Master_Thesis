from dijkstar import Graph, find_path
import matplotlib.pyplot as plt
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

mu = 20 # km/h
dt = 0.5 # s

def number_to_coordinate(x, m):
    column = x % m
    row = x // m
    x1 = 1000/(2*m) + column * 1000/m
    y1 = (1000 * (2*m-1)/(2*m)) - row * 1000/m
    return x1, y1

def cost_function(wind_map, x, m, info):
    x1, y1 = number_to_coordinate(x, m)
    magnitude = wind_map._get_magnitude([(x1, y1)]) # km/h
    direction = wind_map._get_direction([(x1, y1)])
    if info == 'right':
        angle = 0
    elif info == 'left':
        angle = 180
    elif info == 'top':
        angle = 90
    elif info == 'bottom':
        angle = 270
    elif info == 'top_right':
        angle = 45
    elif info == 'top_left':
        angle = 135
    elif info == 'bottom_right':
        angle = 315
    elif info == 'bottom_left':
        angle = 225
    v_prop = np.sqrt(( - magnitude * np.cos(direction * np.pi / 180) + mu *np.cos(angle* np.pi / 180))**2 + (( - magnitude * np.sin(direction * np.pi / 180) + mu *np.sin(angle* np.pi / 180))**2))
    #print(magnitude, direction, v_prop)
    return energy(v_prop, mu, dt)

def add_top_right(wind_map, x, m):
    next = x - m + 1
    cost = np.sqrt(2) * cost_function(wind_map, x, m, 'top_right')  #the distance is greter than the horizontal or vertical one.
    return next , cost

def add_top_left(wind_map, x, m):
    next = x - m - 1
    cost = np.sqrt(2) * cost_function(wind_map, x, m, 'top_left')  #the distance is greter than the horizontal or vertical one.
    return next , cost

def add_bottom_right(wind_map, x, m):
    next = x + m + 1
    cost = np.sqrt(2) * cost_function(wind_map, x, m, 'bottom_right')  #the distance is greter than the horizontal or vertical one.
    return next , cost

def add_bottom_left(wind_map, x, m):
    next = x + m - 1
    cost = np.sqrt(2) * cost_function(wind_map, x, m, 'bottom_left')  #the distance is greter than the horizontal or vertical one.
    return next , cost

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
    wind_info = wind_info_2
    discrete_maps = get_discrete_maps(wind_info['info'])
    wind_map = WindMap(discrete_maps, wind_info['lengthscale'])
    
    ## Find m
    m = int(1000/(mu / 3.6 * dt)) # to ensure relative velocity of mu * dt || Be carefull we want second and mu is in km/h

    ## Log Files
    if not os.path.exists('log_files'):
        os.mkdir('log_files')
    os.chdir('log_files')

    if not os.path.exists('wind_map_'+str(wind_info['number'])):
        os.mkdir('wind_map_'+str(wind_info['number']))
    os.chdir('wind_map_'+str(wind_info['number']))

    if not os.path.exists('Djikstra_'+str(m)):
        os.mkdir('Djikstra_'+str(m))
    os.chdir('Djikstra_'+str(m))

    
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
            next, cost = add_bottom_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x // m == 0): # top edge
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x == m-1): # corner right top
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x % m == m-1): # right edge
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x == m*m - 1): # corner right bottom
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x // m == m-1): # bottom edge
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x == m*(m-1)): # corner left bottom
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
        elif(x % m == 0): # left edge
            next, cost = add_top(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next , cost = add_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_right(wind_map, x, m)
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
            next, cost = add_top_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_right(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_top_left(wind_map, x, m)
            graph.add_edge(x, next, cost)
            next, cost = add_bottom_left(wind_map, x, m)
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
    path = best_path[0]
    total_cost = best_path[3]
    path_x = []
    path_y = []
    for i in range(len(path)):
        x, y = number_to_coordinate(path[i], m)
        path_x.append(float(x))
        path_y.append(float(y))
    fig = plot_wind_field(wind_map, start, target)
    print(len(path_x), len(path_y))
    fig.suptitle('Energy consumed : '+str(round(float(total_cost), 2)), fontsize=16)
    plt.plot(path_x, path_y, '-', color = 'black', linewidth = 3)
    plt.savefig('DjikstraPath.png')

    os.chdir('../')
    os.chdir('../')
    os.chdir('../')

    ## Post processing
