import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyinform

from src.ca_model import CA_MODEL
import src.constants as constants
from src.objectives import min_action_entropy, max_action_entropy, min_global_action_entropy, compute_loss, compute_time_lagged_MI

class Individual:

    def __init__(self, id, weights=None, init_grid=None, init_signal=None):
        self.genome = CA_MODEL(weights=weights, init_grid=init_grid, init_signal=init_signal)
        # self.target = init_grid
        self.id = id

        # objective scores
        self.age = 0
        self.loss = 0
        self.loss_phase1 = 0 
        self.loss_phase2 = 0
        self.MI = 0
        self.grad = 0

        # Additional objectives
        self.min_action_entropy = 0 
        self.max_action_entropy = 0
        self.min_global_action_entropy = 0

    def evaluate(self, objectives, target, g):

        history, sensors, actions = self.genome.run()

        # Compute Objectives

        # Always compute loss and empowerment
        self.loss = compute_loss(history, target) # max fitness

        self.MI = compute_time_lagged_MI(actions=actions, sensors=sensors)

        # self.grad = (g/constants.GENERATIONS) * (self.error) + (1-(g/constants.GENERATIONS))*(self.MI)
        if 'error_phase1' in objectives and 'error_phase2' in objectives:
            # split into history 1 and history 2
            split = constants.ITERATIONS//2
            history1 = history[:split]
            history2 = history[split:]
            self.loss_phase1 = compute_loss(history1, target)
            self.loss_phase2 = compute_loss(history2, target)

        if 'min_action_entropy' in objectives:
            self.min_action_entropy = min_action_entropy(actions)

        if 'max_action_entropy' in objectives:
            self.max_action_entropy = max_action_entropy(actions)

        if 'min_global_action_entropy' in objectives:
            self.min_global_action_entropy = min_global_action_entropy(actions)

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
        elif objective == 'loss':
            return self.loss
        elif objective == 'loss_phase1':
            return self.loss_phase1
        elif objective == 'loss_phase2':
            return self.loss_phase2
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
        if 'loss' not in objectives:
            print('loss:', self.loss, end=' ')
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