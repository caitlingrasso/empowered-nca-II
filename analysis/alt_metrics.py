'''
Created on 2023-02-01 12:45:53
@author: caitgrasso

Description: Other metrics used to evaluate behavior of the NCAs. 

Metrics:
1. # connected components (minimize, want cohesive shapes)
2. # cells touching a boundary (minimize)
3. Transiency of patterns
'''
import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import entropy

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import sort_by
import visualizations
import constants
from config import targets
from skimage import measure

def connected_components(history):
    final_state = history[-1][:,:,0]
    labeled_im, num_labels = measure.label(final_state, background=0, return_num=True, connectivity=1)
    return num_labels

def squareness(history):
    pass

def p_boundary_cells(history):
    final_state = history[-1][:,:,0]
    n_boundary_cells = np.sum(final_state)-np.sum(final_state[1:-1,1:-1])
    return n_boundary_cells/np.sum(final_state)

def transiency(history):
    grid = history[0][:,:,0]
    vals = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell_states = []
            alive = False
            for t in range(len(history)):
                if history[t][r,c,0]==0 and not alive:
                    alive = False
                else:
                    alive = True

                cell_states.append(history[t][r,c,0])

            # compute entropy of cell states 
            if np.sum(cell_states)!=0: # only compute for cells that are live at one point during the sim
                
                # Get number of times a cell changes states
                num_changes=0
                for i in range(1,len(cell_states)):
                    if cell_states[i-1]!=cell_states[i]:
                        num_changes+=1
                vals.append(num_changes)
             
    return np.mean(vals)

def final_state_error(history):

    final_state = history[-1][:,:,0]

    return np.sum(np.power((final_state-TARGET),2))/(constants.GRID_SIZE**2)

def perimeter(history):
    final_state = history[-1][:,:,0]
    return measure.perimeter(final_state, neighbourhood=4)/np.sum(final_state)


# Set parameters for analyses
constants.GRID_SIZE = 25
TARGET = targets(25)['square']
ITERATIONS = 50
metric = p_boundary_cells

txs = ['bi-loss', 'tri-loss', 'k=1', 'k=45']
tx_filepath_dict = {'bi-loss':'data/2023_01_03/k25/error', 'tri-loss':'data/2023_01_03/k25/error_phase1_error_phase2', 'k=1':'data/2023_01_05/k1/error_MI', 'k=45':'data/2023_01_05/k45/error_MI'}
# out_path = 'results/summary/'

metric_dict = {final_state_error:'Loss of final state', perimeter:'Perimeter Nomalized', connected_components:'Num Connected Components', p_boundary_cells:'Proportion cells on boundary', transiency:'Cell State Transiency'}
tx_color_dict = {'bi-loss':'silver', 'tri-loss':'silver', 'k=1':'dimgray', 'k=45':'dimgray'}

for i,tx in enumerate(txs):

    vals = []

    filenames = glob(tx_filepath_dict[tx]+'/*.p')

    for filename in filenames:
        
        # print(filename)

        with open(filename, 'rb') as f:
            best,stats = pickle.load(f)

        history = best.playback(iterations=ITERATIONS)

        val = metric(history)
        # print(val)

        vals.append(val)

    ci95 = 1.96 * (np.std(vals)/np.sqrt(len(vals)))
    plt.bar(i+1, np.mean(vals), yerr=ci95, color=tx_color_dict[tx], capsize=5)

plt.xticks(np.arange(1,len(txs)+1), labels=txs)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel(metric_dict[metric], fontsize=20)
plt.savefig('gecco23_figs/{}.png'.format(metric.__name__), dpi=500, bbox_inches='tight')

