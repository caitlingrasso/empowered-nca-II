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

# txs = ['bi-loss', 'tri-loss', 1,5,10,17,25,32,40,45]
txs = ['bi-loss', 'tri-loss', 45, 1]
tx_dirs = {'bi-loss':'2023_01_03/k25', 'tri-loss':'2023_01_03/k25', 1:'2023_01_05/k1', 5:'2023_01_04/k5', 10:'2023_01_03/k10', 17:'2023_01_05/k17', 25:'2023_01_03/k25', 32:'2023_01_05/k32', 40:'2023_01_03/k40', 45:'2023_01_05/k45'}
tx_color_dict = {'bi-loss':'tab:blue', 'tri-loss':'tab:orange', 45:'tab:cyan', 1:'tab:red'}
fig, ax = plt.subplots(1,1, figsize=(10,7))
labels = []
handles = []
pos={'bi-loss':.01, 'tri-loss':-.01, 45:0, 1:0}

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

plt.savefig('gecco23_figs/extended_sim_stability.png', bbox_inches='tight', dpi=500)

# # Plot slope bar chart
# fig, ax = plt.subplots(1,1, figsize=(10,7))
# h = 0.005
# for i in range(len(all_slopes)):
#     if 'k=' in labels[i]:
#         color='dimgray'
#     else:
#         color='silver'

#     avg_slope = np.mean(all_slopes[i])
#     ci95 = 1.96 * (np.std(all_slopes[i])/np.sqrt(len(all_slopes[i])))  

#     # Compute significance compared to controls 
#     t, bi_loss_p = ttest_ind(all_slopes[0], all_slopes[i])
#     t, tri_loss_p = ttest_ind(all_slopes[1], all_slopes[i])

#     print(bi_loss_p, tri_loss_p)
    
#     ax.bar(i, avg_slope, yerr=ci95, color=color, capsize=5)
    
#     max_value = avg_slope+ci95

#     alpha = 0.05/(len(txs[2:])*2)
#     print(alpha)
#     # annotate plot with significance
#     bi_loss_sig = bi_loss_p<alpha
#     tri_loss_sig = tri_loss_p<alpha
#     if bi_loss_sig and tri_loss_sig:
#         print('here1')
#         ax.text(i-0.075, max_value+h, '†', ha='center', va='bottom', color='k')
#         ax.text(i+0.075, max_value+h, '‡', ha='center', va='bottom', color='k')
#     elif bi_loss_sig:
#         print('here2')
#         ax.text(i, max_value+h, '†', ha='center', va='bottom', color='k')
#     elif tri_loss_sig:
#         print('here3')
#         ax.text(i, max_value+h, '‡', ha='center', va='bottom', color='k')

# ax.set_xticks(np.arange(len(all_slopes)))
# ax.set_xticklabels(labels)
# ax.set_ylabel('Slope')
# plt.savefig('results/robustness/stability_slopes_100iter_all.png')

