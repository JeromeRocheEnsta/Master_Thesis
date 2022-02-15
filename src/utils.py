import numpy as np

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