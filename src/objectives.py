import numpy as np
from collections import Counter

def shannon_entropy(series):

    # Construct the probability distribution p
    N = len(series)
    freq_dict = dict(Counter(series))

    H = 0

    for i in freq_dict:
        H += (freq_dict[i]/N) * np.log2(1/(freq_dict[i]/N))

    return H

def min_action_entropy(actions):
    """ Using Shannon Entropy as a measure of diversity.

    Args:
        actions (list of numpy matrices): actions matrices at each time step of NCA simulation 

    Returns:
        float: mean entropy of cells' signaling channels
    """    

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
    """ Using Shannon Entropy as a measure of diversity.

    Args:
        actions (list of numpy matrices): actions matrices at each time step of NCA simulation 

    Returns:
        float: mean entropy of cells' signaling channels
    """    

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
    """ Using Shannon Entropy as a measure of diversity.

    Args:
        actions (list of numpy matrices): actions matrices at each time step of NCA simulation 

    Returns:
        float: mean entropy of cells' signaling channels
    """    

    A = []

    for action in actions:
        action = action[1:-1, 1:-1,:] # remove padding
        at = action[:,:,-1].flatten() # action is the last value in the NN output (output signal)
        
        # Add actions to a global time series (not split by individual cells)
        for i in range(len(at)):
            A.append(at[i])
            
    return shannon_entropy(A)