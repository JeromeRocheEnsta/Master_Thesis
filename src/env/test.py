from wind_env import *

discrete_maps = [[], []]

wind_info = [
    [(5, 350),(5, 315),(10, 290),(10, 270),(15, 260),(15, 250),(15, 250)],
    [(5, 350),(10, 340),(10, 315),(10, 300),(15, 260),(15, 260),(15, 240)],
    [(5, 0),(5, 350),(10, 315),(10, 300),(15, 270),(15, 250),(15, 250)],
    [(5, 0),(5, 355),(10, 340),(10, 315),(15, 280),(10, 270),(15, 260)],
    [(5, 0),(5, 0),(10, 350),(5, 330),(5, 300),(15, 280),(15, 270)],
    [(5, 0),(5, 0),(5, 350),(5, 335),(5, 290),(10, 280),(15, 270)],
    [(5, 0),(5, 350),(5, 340),(5, 330),(5, 315),(10, 280),(15, 270)]
]

for i in range (49):
    row = i//7
    col =  i%7
    discrete_maps[0].append((col * 1000/6, 1000 - row * 1000/6, wind_info[row][col][0]))
    discrete_maps[1].append((col * 1000/6, 1000 - row * 1000/6, wind_info[row][col][1]))


#############
## CASE 1
#############
A = WindEnv_gym(wind_maps = discrete_maps, start = (500, 999) )
A.reset()

A.state[0] = 45
next_x = 502
next_y = 1001
A._next_observation(next_x, next_y)
print('Test 1: OK') if (A.state[0] == 315 and A.state[1] == 502 and A.state[2] == 999) else print('Test 1: NOT OK')


#############
## CASE 2
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (502, 999) )
A.reset()

A.state[0] = 135
next_x = 500
next_y = 1001
A._next_observation(next_x, next_y)
print('Test 2: OK') if (A.state[0] == 225 and A.state[1] == 500 and A.state[2] == 999) else print('Test 2: NOT OK')

#############
## CASE 3
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (1, 500) )
A.reset()

A.state[0] = 135
next_x = -1
next_y = 502
A._next_observation(next_x, next_y)
print('Test 3: OK') if (A.state[0] == 45 and A.state[1] == 1 and A.state[2] == 502) else print('Test 3: NOT OK')

#############
## CASE 4
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (1, 502) )
A.reset()

A.state[0] = 225
next_x = -1
next_y = 500
A._next_observation(next_x, next_y)
print('Test 4: OK') if (A.state[0] == 315 and A.state[1] == 1 and A.state[2] == 500) else print('Test 4: NOT OK')

#############
## CASE 5
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (500, 1) )
A.reset()

A.state[0] = 315
next_x = 502
next_y = -1
A._next_observation(next_x, next_y)
print('Test 5: OK') if (A.state[0] == 45 and A.state[1] == 502 and A.state[2] == 1) else print('Test 5: NOT OK')

#############
## CASE 6
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (502, 1) )
A.reset()

A.state[0] = 225
next_x = 500
next_y = -1
A._next_observation(next_x, next_y)
print('Test 6: OK') if (A.state[0] == 135 and A.state[1] == 500 and A.state[2] == 1) else print('Test 6: NOT OK')


#############
## CASE 7
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (999, 500) )
A.reset()

A.state[0] = 45
next_x = 1001
next_y = 502
A._next_observation(next_x, next_y)
print('Test 7: OK') if (A.state[0] == 135 and A.state[1] == 999 and A.state[2] == 502) else print('Test 7: NOT OK')

#############
## CASE 8
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (999, 502) )
A.reset()

A.state[0] = 315
next_x = 1001
next_y = 500
A._next_observation(next_x, next_y)
print('Test 8: OK') if (A.state[0] == 225 and A.state[1] == 999 and A.state[2] == 500) else print('Test 8: NOT OK')

#############
## CASE 9
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (999, 998) )
A.reset()

A.state[0] = 45
next_x = 1002
next_y = 1001
A._next_observation(next_x, next_y)
print('Test 9: OK') if (A.state[0] == 225 and A.state[1] == 998 and A.state[2] == 999) else print('Test 9: NOT OK')

#############
## CASE 10
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (998, 999) )
A.reset()

A.state[0] = 45
next_x = 1001
next_y = 1002
A._next_observation(next_x, next_y)
print('Test 10: OK') if (A.state[0] == 225 and A.state[1] == 999 and A.state[2] == 998) else print('Test 10: NOT OK')

#############
## CASE 11
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (999, 2) )
A.reset()

A.state[0] = 315
next_x = 1002
next_y = -1
A._next_observation(next_x, next_y)
print('Test 11: OK') if (A.state[0] == 135 and A.state[1] == 998 and A.state[2] == 1) else print('Test 11: NOT OK')

#############
## CASE 12
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (998, 1) )
A.reset()

A.state[0] = 315
next_x = 1001
next_y = -2
A._next_observation(next_x, next_y)
print('Test 12: OK') if (A.state[0] == 135 and A.state[1] == 999 and A.state[2] == 2) else print('Test 12: NOT OK')

#############
## CASE 13
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (2, 1) )
A.reset()

A.state[0] = 225
next_x = -1
next_y = -2
A._next_observation(next_x, next_y)
print('Test 13: OK') if (A.state[0] == 45 and A.state[1] == 1 and A.state[2] == 2) else print('Test 13: NOT OK')

#############
## CASE 14
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (1, 2) )
A.reset()

A.state[0] = 225
next_x = -2
next_y = -1
A._next_observation(next_x, next_y)
print('Test 14: OK') if (A.state[0] == 45 and A.state[1] == 2 and A.state[2] == 1) else print('Test 14: NOT OK')

#############
## CASE 15
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (1, 998) )
A.reset()

A.state[0] = 135
next_x = -2
next_y = 1001
A._next_observation(next_x, next_y)
print('Test 15: OK') if (A.state[0] == 315 and A.state[1] == 2 and A.state[2] == 999) else print('Test 15: NOT OK')

#############
## CASE 16
#############

A = WindEnv_gym(wind_maps = discrete_maps, start = (2, 999) )
A.reset()

A.state[0] = 135
next_x = -1
next_y = 1002
A._next_observation(next_x, next_y)
print('Test 16: OK') if (A.state[0] == 315 and A.state[1] == 1 and A.state[2] == 998) else print('Test 16: NOT OK')