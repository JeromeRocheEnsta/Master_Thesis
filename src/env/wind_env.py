import numpy as np
from env.wind.wind_map import *
import matplotlib.pyplot as plt
plt.ioff()
import random
import gym
from gym import spaces
from env.utils import reward_1, reward_2, energy



class WindEnv_gym(gym.Env):
    def __init__(self, dt = 1.8, mu = 20, alpha = 15, length = 1000, heigth = 1000, target_radius = 20, initial_angle = 0,  reward_number = 1, propulsion = 'constant', ha = 'propulsion', straight = False, wind_maps = None, start = None, target = None, bonus = 10, scale = 1, reservoir_info = [False, None]):
        super(WindEnv_gym, self).__init__()


        ### Reservoir Info
        self.reservoir_use = reservoir_info[0]
        if self.reservoir_use == True:
            self.reservoir_capacity = reservoir_info[1]


        ### Type of dynamic
        self.straight = straight
        self.propulsion = propulsion
        self.ha = ha
        self.scale = scale #Reward scaling
        self.bonus = bonus # bonus for reaching the goal
        self.magnitude_max = 20


        ### Choose the reward
        self.reward_number = reward_number

        ### Informative data about the path
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_ha = []
        self.trajectory_action = []
        self.time = []
        self.energy = []

        ### Global variable
        self.dt = dt ## timestep (s)
        self.mu = mu ## Minimal Level-Flight Velocity (km/h)


        self.propulsion_velocity= mu ## Velocity of the propulsion system
        self.wind_map = WindMap(wind_maps) ## Create the continuous wind_map

        self.alpha = alpha # action range (angle)
        
        ### Environment variables
        self.length = length
        self.heigth = heigth
        self.target_radius = target_radius


        self.initial_angle = initial_angle

        ### Starting and target points
        if start == None:
            self.start = (random.random() * length//2 , (random.random() + 1) * heigth//2 )
        else:
            self.start = start
        if target == None:
            self.target = ((random.random() +1)* length//2 , random.random() * heigth//2 )
        else:
            self.target = target
            
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(np.array([-1]), np.array([1]), shape = (1,), dtype = np.float)
        self.observation_space = spaces.Box( low = np.array([0., 0., 0.], dtype = np.float), high = np.array([360., 1000., 1000.], dtype = np.float), shape = (3,), dtype = np.float)

    def _target(self):
        dist = (self.state[1] - self.target[0])**2 + (self.state[2] - self.target[1])**2
        if ( dist <= self.target_radius**2 ):
            return True
        else: 
            return False
    



    def _next_observation(self, next_x, next_y):
        counter = 0
        if(next_x > 1000 or next_x < 0):
            counter += 1
        if(next_y >1000 or next_y < 0):
            counter += 1


        if(counter == 0):
            # zero reflexion
            self.state[1] = np.float(next_x)
            self.state[2] = np.float(next_y)
        elif(counter == 1):
            # only one reflexion
            if(next_x > 1000):
                self.state[1] = 2000 - next_x
                self.state[2] = next_y
                self.state[0] = self.state[0] + 2 * ( 90 - self.state[0] ) if self.state[0] < 90 else self.state[0] - 2 * ( self.state[0] - 270 )
            elif(next_x < 0):
                self.state[1] = - next_x
                self.state[2] = next_y
                self.state[0] = self.state[0] - 2 * ( self.state[0] - 90 ) if self.state[0] < 180 else self.state[0] + 2 * ( 270 - self.state[0] )
            elif(next_y > 1000):
                self.state[1] = next_x
                self.state[2] = 2000 - next_y
                self.state[0] = self.state[0] + 2 * ( 180 - self.state[0] ) if self.state[0] > 90 else self.state[0] - 2 * ( self.state[0] ) + 360
            elif(next_y < 0):
                self.state[1] = next_x
                self.state[2] = - next_y
                self.state[0] = self.state[0] + 2 * ( 360 - self.state[0] ) - 360 if self.state[0] > 270 else self.state[0] - 2 * ( self.state[0] - 180)
        else:
            if(next_x < 0 and next_y > 1000):
                ref_angle = np.arctan((next_y - 1000)/(-next_x))
                self.state [1] = - next_x
                self.state[2] = 2000 - next_y
                if(self.state[0]*np.pi / 180 > ref_angle):
                    self.state[0] = self.state[0] + 2 * (180 - self.state[0]) # first deviation
                    self.state[0] = self.state[0] + 2 * (270 - self.state[0]) # second deviation
                else:
                    self.state[0] = self.state[0] - 2 * (self.state[0] - 90) # first deviation
                    self.state[0] = self.state[0] - 2 * (self.state[0]) + 360 # second deviation
            elif(next_x > 1000 and next_y > 1000):
                ref_angle = np.arctan((next_y - 1000)/(next_x - 1000))
                self.state [1] = 2000 - next_x
                self.state[2] = 2000 - next_y
                if(self.state[0]*np.pi / 180 > ref_angle):
                    self.state[0] = self.state[0] + 2 * ( 90 - self.state[0] ) # first deviation
                    self.state[0] = self.state[0] + 2 * ( 180 - self.state[0] ) # second deviation
                else:
                    self.state[0] = self.state[0] - 2 * ( self.state[0] ) + 360 # first deviation
                    self.state[0] = self.state[0] - 2 * ( self.state[0] - 270 ) # second deviation
            elif(next_x > 1000 and next_y < 0):
                ref_angle = np.arctan(( - next_y )/(next_x - 1000))
                self.state [1] = 2000 - next_x
                self.state[2] = - next_y
                if(self.state[0]*np.pi / 180 > (360 - ref_angle)):
                    self.state[0] = self.state[0] + 2 * ( 360 - self.state[0] ) -360 # first deviation
                    self.state[0] = self.state[0] + 2 * ( 90 - self.state[0] ) # second deviation
                else:
                    self.state[0] = self.state[0] - 2 * ( self.state[0] - 270 ) # first deviation
                    self.state[0] = self.state[0] - 2 * ( self.state[0] - 180 ) # second deviation
            elif(next_x < 0 and next_y < 0):
                ref_angle = np.arctan(( - next_y )/( - next_x ))
                self.state [1] = - next_x
                self.state[2] = - next_y
                if( self.state[0]*np.pi /180 > (180 + ref_angle)):
                    self.state[0] = self.state[0] + 2 * ( 270 - self.state[0] )  # first deviation
                    self.state[0] = self.state[0] + 2 * ( 360 - self.state[0] ) - 360 # second deviation
                else:
                    self.state[0] = self.state[0] - 2 * ( self.state[0] - 180 ) # first deviation
                    self.state[0] = self.state[0] - 2 * ( self.state[0] - 90 ) # second deviation
        # Change the heading angle 
        # change the previous state position, next_x, next_y and find if after the correction the uav is still outside or not
        obs = np.zeros((3,), dtype = np.float)
        obs[0] = self.state[0]
        obs[1] = self.state[1]
        obs[2] = self.state[2]
        return obs


    def reset(self):
        # Execute one time step within the environment
        self.trajectory_x = [self.start[0]]
        self.trajectory_y = [self.start[1]]
        self.trajectory_ha = [self.initial_angle]
        self.trajectory_action = []
        self.energy = [0]
        self.time = [0]
        self.state = np.array([self.initial_angle, self.start[0], self.start[1]], dtype = np.float)

        self.reservoir = None if self.reservoir_use == False else self.reservoir_capacity

        obs = np.zeros((3,), dtype = np.float)
        obs[0] = self.state[0]
        obs[1] = self.state[1]
        obs[2] = self.state[2]

        return(  obs ) 

    def step(self, action):
        # Reset the state of the environment to an initial state
        previous_coordinate = self.state[1], self.state[2]
        self.state[0] = self.state[0] if self.straight else self.state[0] + action * self.alpha
        self.state[0] = self.state[0] % 360
        magnitude = float(self.wind_map._get_magnitude([(previous_coordinate[0], previous_coordinate[1])]))
        direction = float(self.wind_map._get_direction([(previous_coordinate[0], previous_coordinate[1])])) % 360

        if(self.propulsion == 'variable'):
            if (self.ha == 'propulsion'):
                cos = np.cos((direction - self.state[0])* np.pi / 180)
                delta = magnitude**2 * (cos**2 - 1) + self.mu**2
                if(delta <= 0):
                    raise Exception("The wind is too high to find a real valued propulsion velocity")
                else:
                    self.propulsion_velocity = - magnitude * cos + np.sqrt(delta)
                
                
                next_x = self.state[1] + self.dt / 3.6 * ( self.propulsion_velocity * np.cos(self.state[0] * np.pi / 180) + magnitude * np.cos(direction * np.pi / 180) ) 
                next_y = self.state[2] + self.dt / 3.6 * ( self.propulsion_velocity * np.sin(self.state[0] * np.pi / 180) + magnitude * np.sin(direction * np.pi / 180) ) 
            
            elif(self.ha == 'next_state'):
                self.propulsion_velocity = np.sqrt((self.mu * np.cos(self.state[0] * np.pi / 180) - magnitude * np.cos(direction * np.pi / 180))**2 + (self.mu * np.sin(self.state[0] * np.pi / 180) - magnitude * np.sin(direction * np.pi / 180))**2 )

                next_x = self.state[1] + self.dt/3.6 * self.mu * np.cos(self.state[0] * np.pi / 180)
                next_y = self.state[2] + self.dt/3.6 * self.mu * np.sin(self.state[0] * np.pi / 180)
            else:
                raise Exception("ha must be \' propulsion \' or \' next_state\' ")

            if(abs(np.sqrt((next_x - self.state[1])**2+(next_y - self.state[2])**2) - self.mu / 3.6 * self.dt) > 0.1):
                raise Exception("Problem with the constant relative velocity: real distance {} target distance {} for one ts.".format(np.sqrt((next_x - self.state[1])**2+(next_y - self.state[2])**2), self.mu / 3.6 * self.dt))
        else:
            raise Exception("This propulsion system is not defined yet")
        

        energy_step = energy(self.propulsion_velocity, self.mu)
        # Can we perform the next step
        if self.reservoir_use == False:
            reservoir_condition = True
        else:
            if(self.reservoir >= energy_step): #check if we have enough energy to perform the step
                reservoir_condition = True
            else:
                reservoir_condition = False

        
        if reservoir_condition:
            # We have enough energy to move
            obs = self._next_observation(np.float(next_x), np.float(next_y) )

            self.trajectory_ha.append(self.state[0])
            self.trajectory_x.append(self.state[1])
            self.trajectory_y.append(self.state[2])
            self.trajectory_action.append(action)
            self.energy.append(self.energy[-1] + energy_step)
            self.time.append(self.time[-1] + self.dt)

            if self.reservoir_use == True:
                self.reservoir -= energy_step

            return(obs, self.reward(), self._target(), {})
        else:
            # We don't have enough energy to continue moving
            obs = self._next_observation(np.float(previous_coordinate[0]), np.float(previous_coordinate[1]) )

            self.trajectory_ha.append(self.state[0])
            self.trajectory_x.append(self.state[1])
            self.trajectory_y.append(self.state[2])
            self.trajectory_action.append(action)
            self.energy.append(self.energy[-1])
            self.time.append(self.time[-1] + self.dt)

            return(obs, 0, True, {})


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def reward(self):
        if (self.reward_number == 1):
            return reward_1(self.state, self.target, self.length, self.heigth, self.target_radius, self._target(), self.bonus, self.scale)
        elif(self.reward_number == 2):
            magnitude = float(self.wind_map._get_magnitude([(self.state[1], self.state[2])]))
            direction = float(self.wind_map._get_direction([(self.state[1], self.state[2])])) % 360
            return reward_2(magnitude, self.magnitude_max, direction, (self.state[1], self.state[2]), self.target, self._target(), self.bonus, self.scale)
        else:
            raise Exception("This reward number is not available yet")

    def plot_trajectory(self, reward_ep, ref_trajectory_x = None, ref_trajectory_y = None, ref_energy = None, ref_time = None):
        fig, axs = plt.subplots(nrows = 2, ncols = 2)
        fig.set_size_inches(10, 7)

        axs[0,0].plot(self.trajectory_x, self.trajectory_y, '-')
        a_circle = plt.Circle(self.target, radius = self.target_radius)
        axs[0,0].add_artist(a_circle)
        if ref_trajectory_x != None:
            axs[0,0].plot(ref_trajectory_x, ref_trajectory_y, 'r--')
        axs[0,0].set_title('steps : {} ; reward : {}'.format(len(self.time) - 1,round(reward_ep, 3)))
        axs[0,0].set_aspect('equal', 'box')
        axs[0,0].set_xlim([0, self.length])
        axs[0,0].set_ylim([0, self.heigth])


        axs[0,1].plot(self.time, self.energy)
        if ref_trajectory_x != None:
            axs[0,1].plot(ref_time, ref_energy, 'r--')
        axs[0,1].set_title('Energy consumed ({}) v.s. time ({}s)'.format(round(self.energy[-1]), round(self.time[-1], 1)) )

        axs[1,0].scatter(self.time, self.trajectory_ha)
        axs[1,0].set_title('Heading angle versus time')

        axs[1,1].plot(self.time[1:], self.trajectory_action)
        axs[1,1].set_title('Action versus time')

        return(fig, axs)