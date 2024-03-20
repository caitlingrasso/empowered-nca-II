import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyinform

from src.ca_model import CA_MODEL
import src.constants as constants
from src.objectives import min_action_entropy, max_action_entropy, min_global_action_entropy

class Individual:

    def __init__(self, id, weights=None, init_grid=None, init_signal=None):
        self.genome = CA_MODEL(weights=weights, init_grid=init_grid, init_signal=init_signal)
        # self.target = init_grid
        self.id = id

        # objective scores
        self.age = 0
        self.error = 0
        self.error_phase1 = 0 
        self.error_phase2 = 0
        self.MI = 0
        self.grad = 0

        # Additional objectives
        self.min_action_entropy = 0 
        self.max_action_entropy = 0
        self.min_global_action_entropy = 0

    def evaluate(self, objectives, target, g):

        history, sensors, actions = self.genome.run()

        # Compute Objectives

        # Always compute error and empowerment
        self.error = self.evaluate_error(history, target) # max fitness

        self.MI = self.compute_time_lagged_MI(actions=actions, sensors=sensors)

        # self.grad = (g/constants.GENERATIONS) * (self.error) + (1-(g/constants.GENERATIONS))*(self.MI)

        if 'min_action_entropy' in objectives:
            self.min_action_entropy = min_action_entropy(actions)
        elif 'max_action_entropy' in objectives:
            self.min_prop_act = max_action_entropy(actions)
        elif 'min_global_action_entropy' in objectives:
            self.min_global_action_entropy = min_global_action_entropy(actions)

        if 'error_phase1' in objectives and 'error_phase2' in objectives:
            # TODO: split into history 1 and history 2
            split = constants.ITERATIONS//2
            history1 = history[:split]
            history2 = history[split:]
            self.error_phase1 = self.evaluate_error(history1, target)
            self.error_phase2 = self.evaluate_error(history2, target)

    def compute_time_lagged_MI(self, actions, sensors, local=False, return_series = False, max = False):
                
        """Computes time-lagged mutual information between each live cell's action/sensor pair of timeseries.
        A cell is live if it's on for at least one time step during development.
        Args:
            grids (list of numpy.ndarrays): Binary state values of each cell at each time step.
            actions (list of numpy.ndarrays): Actions values of each cell at each time step. 
                                        Length of X = # of timesteps. Dimensions of array 
                                        at X[0] = # cells x # cells.
            sensors (list of numpy.ndarrays): Sensor values of each cell at each time step. 
                                        Length of Y = # of timesteps. Dimensions of array 
                                        at Y[0] = # cells x # cells.
            k (int): history length
        """   
        # binary_states = grids[t][:,:,0] - 25x25 matrix 
        # action_states = actions[t] - 25x25x6 matrix (actions[t][i,j,5] is the action here - output signal)
        # sensor_states = sensors[t] - 25x25x10 matrix (np.mean(sensors[t][i,j,5:] is the sensor here - average of input signals from neighbors + self))

        # split into Ank and Sn2k
        A = []
        S = [] 

        # Collect action (input) states for first half of timesteps
        for action in actions[:-constants.HISTORY_LENGTH]: # original
        # for action in actions[(constants.ITERATIONS-constants.HISTORY_LENGTH)-constants.SEQUENCE_LENGTH:-constants.HISTORY_LENGTH]: # cropping for equal number of samples based on constants.SEQUENCE_LENGTH
            action = action[1:-1, 1:-1,:] # remove padding
            at = action[:,:,-1].flatten() # last value in action list (new output signal value of cell)
            A.append(list(at))
        
        # Collect sensor (output) states for second half of timesteps
        for sensor in sensors[constants.HISTORY_LENGTH:]: # original
        # for sensor in sensors[constants.ITERATIONS-constants.SEQUENCE_LENGTH:]: # cropping for equal number of samples based on constants.SEQUENCE_LENGTH
            sensor = sensor[1:-1, 1:-1,:] # remove padding
            st = np.mean(sensor[:,:,constants.NEIGHBORHOOD+1:],axis=2).flatten() # Average last 4 signal values (input signal values of neighbors + self) - these are floats because they are averages
            S.append(list(st))

        # print(len(A[i]))
        # print(len(S[i]))
        # exit()
        if max:
            MI = self.compute_MI(X=A, Y=S, local=local)*-1 # convert back to maximization
        else:
            MI = self.compute_MI(X=A, Y=S, local=local)

        if return_series:
            return MI, A, S
        else:
            return MI

    def evaluate_error(self, history, target):
        '''
        Computes error between current grid and target grid averaged over all iterations
        Normalized between 0-1
        Error is to be minimized
        '''
        all_fits = np.zeros(len(history))
        for i in range(len(history)):
            all_fits[i] = self.evaluate_grid_diff(history[i][:,:,0],target)/(constants.GRID_SIZE**2) # normalized grid difference
        return np.mean(all_fits)

    def evaluate_grid_diff(self, x, target):
        # computes L2 loss on single grid (difference measure)
        return np.sum(np.power((x-target),2))

    def compute_MI(self, X,Y,local=False):
        MI = pyinform.mutual_info(xs=X, ys=Y,local=local)
        return MI*-1

    def mutate(self):
        self.genome.mutate()

    def dominates_other(self, other, objectives):
        # Note: age is always passed in as the first objective

        if len(objectives) == 2:  # bi-objective

            obj1 = objectives[0]
            obj2 = objectives[1]

            if self.get_objective(obj1) == other.get_objective(obj1) and self.get_objective(
                    obj2) == other.get_objective(obj2):
                return self.id > other.id
            elif self.get_objective(obj1) <= other.get_objective(obj1) and self.get_objective(
                    obj2) <= other.get_objective(obj2):
                return True
            else:
                return False

        elif len(objectives) == 3:  # tri-objective

            obj1 = objectives[0]
            obj2 = objectives[1]
            obj3 = objectives[2]

            if self.get_objective(obj1) == other.get_objective(obj1) and self.get_objective(
                    obj2) == other.get_objective(obj2) and self.get_objective(obj3) == other.get_objective(obj3):
                return self.id > other.id
            elif self.get_objective(obj1) <= other.get_objective(obj1) and self.get_objective(
                    obj2) <= other.get_objective(obj2) and self.get_objective(obj3) <= other.get_objective(obj3):
                return True
            else:
                return False

    def get_objective(self, objective):
        if objective == 'age':
            return self.age
        elif objective == 'error':
            return self.error
        elif objective == 'error_phase1':
            return self.error_phase1
        elif objective == 'error_phase2':
            return self.error_phase2
        elif objective == 'MI':
            return self.MI
        elif objective == 'grad':
            return self.grad
        elif objective == 'min_action_entropy': 
            return self.min_action_entropy
        elif objective == 'max_action_entropy':
            return self.max_action_entropy
        elif objective == 'min_global_action_entropy':
            return self.min_global_action_entropy

    def print(self, objectives):
        print('[id:', self.id, end=' ')
        for objective in objectives:
            print(objective, ':', self.get_objective(objective), end=' ')
        if 'error' not in objectives:
            print('error:', self.error, end=' ')
        print(']', end='')

    def playback(self, iterations=constants.ITERATIONS, return_SA = False):

        history, sensors_timeseries, actions_timeseries = self.genome.run(iterations=iterations)
        if return_SA:
            return history, sensors_timeseries, actions_timeseries
        else:
            return history

    def continue_sim(self, init_grids, iterations, return_SA = False):

        history, sensors_timeseries, actions_timeseries = self.genome.run(continue_run=True, init_grids=init_grids, additional_iterations=iterations)

        if return_SA:
            sensors_timeseries_padding_removed = []
            actions_timeseries_padding_removed = []
            for i in range(len(sensors_timeseries)):
                sensors_timeseries_padding_removed.append(sensors_timeseries[i][1:-1, 1:-1,:])
                actions_timeseries_padding_removed.append(actions_timeseries[i][1:-1, 1:-1,:])
            return history, sensors_timeseries_padding_removed, actions_timeseries_padding_removed
        else:
            return history