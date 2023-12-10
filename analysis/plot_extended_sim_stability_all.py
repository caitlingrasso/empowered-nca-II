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

plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

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

tx_dirs = {'bi-loss':'data/exp1_ksweep/error', 'tri-loss':'data/exp1_ksweep/error_phase1_error_phase2', 'k=1':'data/exp1_ksweep/k1/error_MI', 'k=45':'data/exp1_ksweep/k45/error_MI'}
tx_color_dict = {'bi-loss':'tab:blue','tri-loss':'tab:orange','k=1':'tab:red', 'k=45':'tab:cyan'}

outpath = 'results/exp1'

fig, ax = plt.subplots(1,1, figsize=(10,7))
labels = []
handles = []
pos={'bi-loss':.01, 'tri-loss':-.01, 'k=1':0, 'k=45':0} # position of annotation for each line of fit

all_slopes = []
all_losses_dict = {}

for lbl in tx_dirs:

    # Load in the appropriate runs
    inpath = tx_dirs[lbl]+'/'

    all_fits, slopes = extract_all_losses_at_t(inpath)

    avg_fits = np.mean(all_fits, axis=0)
    ci95 = 1.96 * (np.std(all_fits, axis=0)/np.sqrt(len(all_fits)))  
    x = np.arange(len(avg_fits))
    line, = plt.semilogy(x,avg_fits, color=tx_color_dict[lbl])
    plt.fill_between(x, (avg_fits-ci95), (avg_fits+ci95), color=tx_color_dict[lbl], alpha=.25)
    if 'error_MI' in inpath:
        labels.append('k={}'.format(lbl))
    else:
        labels.append(lbl)
    handles.append(line)

    all_slopes.append(slopes)
    all_losses_dict[lbl]=all_fits

    x = np.arange(len(avg_fits))
    coeffs = np.polyfit(x[51:], avg_fits[51:], 1)
    print(lbl)
    print('slope:', coeffs[0])
    p = np.poly1d(np.polyfit(x[51:], avg_fits[51:], 1))

    x_fit = np.linspace(51,len(avg_fits))
    plt.plot(x, avg_fits, x_fit, p(x_fit),color=tx_color_dict[lbl],linestyle='--')
    plt.annotate(f'm={coeffs[0]: .4f}', (101, p(x_fit)[-1]+pos[lbl]), color=tx_color_dict[lbl],fontsize=20, backgroundcolor='white')

ax.set_xlabel('Iterations', fontsize=25)
ax.set_ylabel('Loss',fontsize=25)
train_handle = plt.axvspan(0,50, color='lightgray',alpha=0.7)
handles.append(train_handle)
labels.append('train')
ax.legend(handles=handles, labels=labels, fontsize=20, loc=(.2,.6))
# ax.tick_params(axis='y', labelsize=30)

# plt.show()

plt.savefig('results/extended_sim_stability.png'.format(outpath), bbox_inches='tight', dpi=500)

