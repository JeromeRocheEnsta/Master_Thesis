import numpy as np


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