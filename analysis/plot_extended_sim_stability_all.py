'''
Created on 2023-01-23 22:09:44
@author: caitgrasso

Description: Analyzing how the best NCAs from each treatment perform when they are run for
longer than they were trained for. 
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
    
    all_losses = np.zeros(shape=(len(filenames), EXTENDED_ITERATIONS+1))
    slopes = []
    for i,filename in enumerate(filenames):
        
        with open(filename, 'rb') as f:
            best,stats = pickle.load(f)

            history = best.playback(iterations=EXTENDED_ITERATIONS)

            grid_losses = []

            for grid in history:

                binary_grid = grid[:,:,0]

                loss_at_t = best.evaluate_grid_diff(binary_grid, TARGET)/(constants.GRID_SIZE**2)

                grid_losses.append(loss_at_t)

        # x = np.arange(len(grid_losses))
        # p_coeff = np.polyfit(x[51:], grid_losses[51:], 1)
        # slope=p_coeff[0]
        # slopes.append(slope)

        all_losses[i,:]=grid_losses

    return all_losses, slopes

EXTENDED_ITERATIONS = 100 # double the length of the simulation
GRID_SIZE = 25
constants.GRID_SIZE = GRID_SIZE
TARGET = targets(GRID_SIZE)['square']

txs = ['error', 'error_MI_k1']
tx_dirs = {'bi-loss':'data/exp1/error', 'k=1':'data/exp1/error_MI_k1'}
labels = ['bi-loss','tri-loss-empowerment (k=1)']
tx_color_dict = {'error':'tab:blue','error_MI_k1':'tab:green'}

outpath = 'results/exp1'

fig, ax = plt.subplots(1,1, figsize=(10,7))
labels = []
handles = []
pos={'bi-loss':.01, 'tri-loss':-.01, 45:0, 1:0} # position of annotation for each line of fit

all_slopes = []
all_losses_dict = {}

for i,tx in enumerate(txs):

    # Load in the appropriate runs
    inpath = tx_dirs[tx]+'/'

    all_fits, slopes = extract_all_losses_at_t(inpath)

    avg_fits = np.mean(all_fits, axis=0)
    ci95 = 1.96 * (np.std(all_fits, axis=0)/np.sqrt(len(all_fits)))  
    x = np.arange(len(avg_fits))
    line, = plt.semilogy(x,avg_fits, color=tx_color_dict[tx])
    plt.fill_between(x, (avg_fits-ci95), (avg_fits+ci95), color=tx_color_dict[tx], alpha=.25)
    if 'error_MI' in inpath:
        labels.append('k={}'.format(tx))
    else:
        labels.append(tx)
    handles.append(line)

    all_slopes.append(slopes)
    all_losses_dict[tx]=all_fits

    x = np.arange(len(avg_fits))
    coeffs = np.polyfit(x[51:], avg_fits[51:], 1)
    print(tx)
    print('slope:', coeffs[0])
    p = np.poly1d(np.polyfit(x[51:], avg_fits[51:], 1))

    x_fit = np.linspace(51,len(avg_fits))
    plt.plot(x, avg_fits, x_fit, p(x_fit),color=tx_color_dict[tx],linestyle='--')
    plt.annotate(f'm={coeffs[0]: .4f}', (101, p(x_fit)[-1]+pos[tx]), color=tx_color_dict[tx],fontsize=20, backgroundcolor='white')

ax.set_xlabel('Iterations', fontsize=25)
ax.set_ylabel('Loss',fontsize=25)
train_handle = plt.axvspan(0,50, color='lightgray',alpha=0.7)
handles.append(train_handle)
labels.append('train')
ax.legend(handles=handles, labels=labels, fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('{}/extended_sim_stability.png'.format(outpath), bbox_inches='tight', dpi=500)

