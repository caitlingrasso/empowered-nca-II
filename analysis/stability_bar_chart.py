'''
Created on 2023-02-06 11:37:46
@author: caitgrasso

Description: Stability at the end of extended simulation comparing to final NCA state (not target).
'''

import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import ttest_ind

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import sort_by
import visualizations
import constants
from config import targets

def extract_all_losses_at_t(inpath):
    filenames = glob(inpath+'*.p')
    
    final_state_losses = []

    for i,filename in enumerate(filenames):
        
        with open(filename, 'rb') as f:
            best,stats = pickle.load(f)

            train_history = best.playback(iterations=TRAIN_ITERATIONS)

            final_state = train_history[-1]

            test_history = best.continue_sim(init_grids=final_state, iterations=EXTENDED_ITERATIONS)

            loss = best.evaluate_grid_diff(test_history[-1][:,:,0], final_state[:,:,0])/(constants.GRID_SIZE**2)

            final_state_losses.append(loss)


    return final_state_losses

EXTENDED_ITERATIONS = 100 # double the length of the simulation
TRAIN_ITERATIONS = 50
GRID_SIZE = 25
constants.GRID_SIZE = GRID_SIZE
TARGET = targets(GRID_SIZE)['square']

# txs = ['bi-loss', 'tri-loss', 1,5,10,17,25,32,40,45]
txs = ['bi-loss', 'tri-loss', 'k=1', 'k=45']
tx_dirs = {'bi-loss':'2023_01_03/k25', 'tri-loss':'2023_01_03/k25', 'k=1':'2023_01_05/k1', 5:'2023_01_04/k5', 10:'2023_01_03/k10', 17:'2023_01_05/k17', 25:'2023_01_03/k25', 32:'2023_01_05/k32', 40:'2023_01_03/k40', 'k=45':'2023_01_05/k45'}
tx_color_dict = {'bi-loss':'silver', 'tri-loss':'silver', 'k=1':'dimgray', 'k=45':'dimgray'}

# fig, ax = plt.subplots(1,1, figsize=(10,7))
labels = []
handles = []

all_slopes = []
all_losses_dict = {}

for i,tx in enumerate(txs):

    # Load in the appropriate runs
    if tx=='bi-loss':
        inpath = 'data/'+tx_dirs[tx]+'/error/'
    elif tx=='tri-loss':
        inpath = 'data/'+tx_dirs[tx]+'/error_phase1_error_phase2/'
    else:
        inpath = 'data/'+tx_dirs[tx]+'/error_MI/'

    losses = extract_all_losses_at_t(inpath)

    avg_loss = np.mean(losses, axis=0)
    ci95 = 1.96 * (np.std(losses)/np.sqrt(len(losses)))  

    plt.bar(i+1, avg_loss, yerr=ci95, capsize=5, color=tx_color_dict[tx])

plt.xticks(np.arange(1,len(txs)+1), labels=txs)
plt.ylabel('Instability', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Stability after {} test iterations'.format(EXTENDED_ITERATIONS))

plt.savefig('gecco23_figs/stability_bar_chart_{}test_iter.png'.format(EXTENDED_ITERATIONS), dpi=500, bbox_inches='tight')