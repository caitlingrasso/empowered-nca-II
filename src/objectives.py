import numpy as np
from collections import Counter
import pyinform

import src.constants as constants

def compute_loss(history, target):
    '''
    Computes error between current grid and target grid averaged over all iterations
    Normalized between 0-1
    Error is to be minimized
    '''
    all_fits = np.zeros(len(history))
    for i in range(len(history)):
        all_fits[i] = compute_grid_diff(history[i][:,:,0],target)/(constants.GRID_SIZE**2) # normalized grid difference
    return np.mean(all_fits)

def compute_grid_diff(x, target):
    # computes L2 loss on single grid (difference measure)
    return np.sum(np.power((x-target),2))

def compute_time_lagged_MI(actions, sensors, local=False, return_series = False, max = False):
                
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
        MI = compute_MI(X=A, Y=S, local=local)*-1 # convert back to maximization
    else:
        MI = compute_MI(X=A, Y=S, local=local)

    if return_series:
        return MI, A, S
    else:
        return MI

def compute_MI(X,Y,local=False):
    MI = pyinform.mutual_info(xs=X, ys=Y,local=local)
    return MI*-1

def shannon_entropy(series):

    # Construct the probability distribution p
    N = len(series)
    freq_dict = dict(Counter(series))

    H = 0

    for i in freq_dict:
        H += (freq_dict[i]/N) * np.log2(1/(freq_dict[i]/N))

    return H

def min_action_entropy(actions):

    A = []

    for action in actions:
        action = action[1:-1, 1:-1,:] # remove padding
        at = action[:,:,-1].flatten() # action is the last value in the NN output (output signal)
        
        # Construct time series for individual cells 
        for i in range(len(at)):
            try:
                A[i].append(at[i])
            except:
                A.append([])
                A[i].append(at[i])

    # Iterate through the individual cell time series and compute entropy
    entropies = []
    for cell_series in A:

        entropies.append(shannon_entropy(cell_series))

    return np.mean(entropies)

def max_action_entropy(actions):

    A = []

    for action in actions:
        action = action[1:-1, 1:-1,:] # remove padding
        at = action[:,:,-1].flatten() # action is the last value in the NN output (output signal)
        
        # Construct time series for individual cells 
        for i in range(len(at)):
            try:
                A[i].append(at[i])
            except:
                A.append([])
                A[i].append(at[i])

    # Iterate through the individual cell time series and compute entropy
    entropies = []
    for cell_series in A:

        entropies.append(shannon_entropy(cell_series))

    return np.mean(entropies)*-1

def min_global_action_entropy(actions):

    A = []

    for action in actions:
        action = action[1:-1, 1:-1,:] # remove padding
        at = action[:,:,-1].flatten() # action is the last value in the NN output (output signal)
        
        # Add actions to a global time series (not split by individual cells)
        for i in range(len(at)):
            A.append(at[i])
            
    return shannon_entropy(A)

