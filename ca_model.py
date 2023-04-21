import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import math

import constants
from util import to_function, tanh, sin, cos, abs, sigmoid, relu
from visualizations import display_grid

class CA_MODEL:

    def __init__(self, n_channels=constants.N_CHANNELS, weights=None, init_grid=None, init_signal=None):
        self.n_channels = n_channels
        self.grids = self.initialize_grids(init_grid, init_signal)
        self.init_grid = init_grid
        self.init_signal = init_signal
        if weights is None:
            self.model, self.activations = self.initialize_model()  # list of layers (layer=matrix of weights)
        else:
            if type(weights) is list:
                self.model = weights
            else: 
                self.model = [weights]
            self.inputs = self.model[0].shape[0]
            self.outputs = self.model[-1].shape[1]
            self.activations = [sigmoid]

        self.memory = np.zeros((self.grids.shape[0], self.grids.shape[1], int(self.inputs/2)))

    def initialize_grids(self, init_grid, init_signal):
        if constants.CONTINUOUS_SIGNALING:
            # both grids are continuous
            x=np.zeros((constants.GRID_SIZE+2,constants.GRID_SIZE+2,self.n_channels), dtype=float) # +2 for padding
        else:
            # both grids are discrete
            x=np.zeros((constants.GRID_SIZE, constants.GRID_SIZE))
            x=np.zeros((constants.GRID_SIZE+2,constants.GRID_SIZE+2,self.n_channels), dtype=int) # +2 for padding

        if init_grid is not None:
            x[:,:,0] = np.pad(init_grid, pad_width=1, mode='constant', constant_values=0)
        else:
            x[(constants.GRID_SIZE+2)//2,(constants.GRID_SIZE+2)//2, 0] = 1

        if init_signal == 'random':
            x[:,:,1]=np.random.randint(0, 255, size=(x.shape[0], x.shape[1],1))
        
        return x

    def initialize_model(self):
        # initializes model to simple NN - fully connected, no hidden layers
        self.inputs = constants.NEIGHBORHOOD*2 + 2 # +2 for sense of self (state and signal)
        self.outputs = constants.NEIGHBORHOOD + 2 # grow over neighborhood and self state/signal

        if constants.MEMORY:
            self.inputs *= 2

        model = [np.random.random((self.inputs,self.outputs))*2-1] # random values [-1,1)
        activation = [sigmoid]
        return model, activation

    def add_dense_layer(self, n_hn, activation):
        if len(self.model)==1:
            weights = [np.random.random((self.inputs,n_hn))*2-1] # input layer
            weights+= [np.random.random((n_hn, self.outputs))*2-1] # output layer
            self.model=weights
            self.activations = [to_function(activation)] + self.activations
        else:
            layer_prev = self.model[-2]
            layer = [np.random.random((layer_prev.shape[1],n_hn))*2-1]
            last_layer = [np.random.random((n_hn, self.outputs))*2-1]
            self.model = self.model[:-1]+layer+last_layer
            self.activations = self.activations[:-1]+[to_function(activation)] + [self.activations[-1]]
    

    def model_summary(self):
        for l in range(len(self.model)):
            print('----------------------------')
            print('DENSE LAYER', l)
            print('input size:', self.model[l].shape[0])
            print('output size:', self.model[l].shape[1])
            print('activation:', self.activations[l].__name__)
        print('----------------------------')

    def run(self, iterations=constants.ITERATIONS, continue_run=False, additional_iterations=1, init_grids=None):
        if not continue_run:
            self.grids = self.initialize_grids(self.init_grid, self.init_signal)
            ca_iter = iterations
        elif continue_run:
            self.grids=np.pad(init_grids, pad_width=((1,1),(1,1),(0,0)), mode='constant', constant_values=0)
            ca_iter = additional_iterations

        history = []
        history.append(self.grids.copy()[1:-1, 1:-1,:])
        sensors_timeseries = []
        actions_timeseries = []
        for i in range(ca_iter):
            # print('ITERATION', i)
            self.update()
            sensors_timeseries.append(self.sensors)
            actions_timeseries.append(self.actions)
            if constants.DIFFUSE:
                self.diffuse_signal()
            history.append(self.grids.copy()[1:-1, 1:-1,:])
        return history, sensors_timeseries, actions_timeseries

    def diffuse_signal(self):
        # simulating gap junctions - signals diffuse to neighbors in all grids other than cell state
        for c in range(1,self.n_channels):
            self.grids[:,:,c] = self.grids[:,:,c] * (1-constants.DIFFUSION_RATE) + (constants.DIFFUSION_RATE/constants.NEIGHBORHOOD) \
                                * np.sum(self.get_neighbors(self.grids[:,:,c]), axis=2) # conservation of signal
            
        if not constants.CONTINUOUS_SIGNALING:
            self.grids = self.grids.astype(int)

    def update(self):
        neigh = self.get_neighbors(self.grids[:,:,0])

        ins = np.concatenate((self.get_neighbors(self.grids[:,:,0]),
                                 np.reshape(self.grids[:,:,0], (self.grids.shape[0], self.grids.shape[1], 1))), axis=2)

        for c in range(1,self.n_channels):
            ins = np.concatenate((ins, self.get_neighbors(self.grids[:,:,c]),
                                    np.reshape(self.grids[:,:,c], (self.grids.shape[0], self.grids.shape[1], 1))), axis=2)

        if constants.SIGNAL_NOISE:
            # Add Gaussian noise to the input signals
            in_signals = ins[:,:,5:]

            noisy_input = np.random.normal(in_signals, scale=constants.SIGNAL_NOISE_STD)

            # print('ORIGINAL INPUT:')
            # print(in_signals[in_signals>0])
            noisy_input[in_signals==0] = 0 # cancel out dead cells
            noisy_input = np.around(noisy_input).astype('int') # round and discritize signals
            noisy_input[noisy_input<0]=0 # min signal is 0
            noisy_input[noisy_input>255]=255 # max signal is 255
            # print('NOISY INPUT:')
            # print(noisy_input[in_signals>0])
            # print()

            ins[:,:,5:] = noisy_input

        if constants.MEMORY:
            inputs = np.concatenate((self.memory, ins,), axis=2)
        else:
            inputs = ins.copy()

        # save inputs to be returned by update function
        self.sensors = inputs

        # Initialize empty outputs array
        outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.outputs), dtype=float)

        # Compute forward pass of neural network
        outputs[self.grids[:,:,0] == 1] = self.forward_pass(inputs[self.grids[:,:,0] == 1])
        
        # save outputs to be returned by update function
        self.actions = outputs

        all_indices = np.indices((constants.GRID_SIZE + 2, constants.GRID_SIZE + 2))  # plus 2 because of the padding

        die = outputs[:, :, constants.NEIGHBORHOOD]

        # Update cell states (allows for apoptosis)
        self.grids[:,:,0] = self.grids[:,:,0] * die

        # Update signals
        for c in range(1,self.n_channels):
            self.grids[:, :, c] = outputs[:,:,constants.NEIGHBORHOOD+c] * die # if cell is dead, set signal to 0

        # Divide over neighborhood
        directions = [self.down, self.up, self.right, self.left, self.topleft, self.topright, self.bottomleft, self.bottomright]
        for i in range(constants.NEIGHBORHOOD):
            grow_dir = die * outputs[:, :, i] # mask of live cells
            x, y = directions[i](all_indices[0][grow_dir == 1], all_indices[1][grow_dir == 1])
            self.grids[x, y, 0] = 1
            for c in range(1,self.n_channels):
                self.grids[x, y, c] = outputs[grow_dir==1][:,constants.NEIGHBORHOOD+c]

        # Delete live cells on edges
        self.grids[:, 0::self.grids.shape[1] - 1, 0] = 0
        self.grids[0::self.grids.shape[0] - 1, :, 0] = 0

        self.grids[:, 0::self.grids.shape[1] - 1, 1] = 0
        self.grids[0::self.grids.shape[0] - 1, :, 1] = 0

        self.memory = ins

    def down(self, x, y):
        return x+1, y

    def up(self, x, y):
        return x-1, y

    def right(self, x, y):
        return x, y+1

    def left(self, x, y):
        return x, y-1

    def topright(self, x, y):
        return x-1, y+1

    def topleft(self, x, y):
        return x-1, y-1

    def bottomright(self, x, y):
        return x+1, y+1

    def bottomleft(self, x, y):
        return x+1, y-1

    def forward_pass(self, inputs):
        x = inputs

        for l in range(len(self.model)):
            x = np.dot(x,self.model[l])
            x = self.activations[l](x)

        cell_signal_index = constants.NEIGHBORHOOD+1
        if not constants.CONTINUOUS_SIGNALING:
            x[:,cell_signal_index] = self.rescale(x[:,cell_signal_index]).astype(int)

        # Convert the first 5 inputs to binary (cell states)
        x[:,0:cell_signal_index] = x[:,0:cell_signal_index] > 0
        return x

    def get_neighbors(self, a, neighborhood=constants.NEIGHBORHOOD):
        if neighborhood==4:
            b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
            neigh = np.concatenate((b[2:, 1:-1, None], b[:-2, 1:-1, None],
                b[1:-1, 2:, None], b[1:-1, :-2, None]), axis=2)
        elif neighborhood==8:
            b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
            neigh = np.concatenate((b[2:, 1:-1, None], b[:-2, 1:-1, None],
                                    b[1:-1, 2:, None], b[1:-1, :-2, None],
                                    b[:-2, :-2, None], b[:-2, 2:, None],
                                    b[2:, :-2, None], b[2:, 2:, None]), axis=2)
        return neigh

    # def tanh(self, x):
    #     return np.tanh(x)

    def rescale(self, x, xmin=-1, xmax=1, a=0, b=255):
        xprime = a+((x-xmin)*(b-a))/(xmax-xmin)
        return xprime

    def mutate(self):
        #TODO: include probability so on average 1/L is mutated...if rand(0,1)<mut_rate
        #this would also allow variable mutaion rates.
        layer = np.random.randint(len(self.model))
        r = np.random.randint(self.model[layer].shape[0])
        c = np.random.randint(self.model[layer].shape[1])
        self.model[layer][r,c] = random.gauss(self.model[layer][r,c], math.fabs(self.model[layer][r,c]))

    # def random_damage(self):
    #     all_indices = np.indices((constants.GRID_SIZE + 2, constants.GRID_SIZE + 2))  # plus 2 because of the padding
    #     all_r = all_indices[0][self.grids[:,:,0]==1]
    #     all_c = all_indices[1][self.grids[:,:,0]==1]

    #     index = np.random.randint(len(all_r))
    #     r = all_r[index]
    #     c = all_c[index]

    #     self.damage_at(r,c)

    # def damage_at(self, r, c):
    #     '''
    #     r,c are the coordinate points for the center of the damage
    #     '''
    #     if np.sum(self.grids[:,:,0])==0:
    #         return self.grids
    #     mask = np.zeros((constants.GRID_SIZE+2, constants.GRID_SIZE+2))
    #     mask[r-constants.DAMAGE_RADIUS:r+constants.DAMAGE_RADIUS+1, c-constants.DAMAGE_RADIUS:c+constants.DAMAGE_RADIUS+1]=1
    #     self.grids[mask==1]=0
    #     return self.grids