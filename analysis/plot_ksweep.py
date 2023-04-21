'''
Created on 2022-12-27 12:11:58
@author: caitgrasso

Description: Plotting the average loss of the history length treatments and controls.
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
import sys, os
from scipy.stats import ttest_ind, ranksums
import matplotlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genome import Genome
import constants
from config import targets

def extract_mean_of_best_from_trials(inpath):
    filenames = glob(inpath+'*.p')
    
    best_fits_dict = {} # key = run number, value = best individual's loss from that run
    
    for filename in filenames:

        run = filename.split('run')[-1].split('.')[0]
        
        try:
            with open(filename, 'rb') as f:
                best, stats = pickle.load(f)
            
        except:
            with open(filename, 'rb') as f:
                stuff = pickle.load(f)
                best = stuff[3]

        # Plot for stability over extended iterations
        history = best.playback(iterations=ITERATIONS)
        loss = best.evaluate_error(history, TARGET)
        best_fits_dict[run] = loss
        
        # best_fits_dict[run] = best.get_objective('error')

    return best_fits_dict

def annot_stat(star, x1, x2, y, h, col='k', ax=None):
    ax = plt.gca() if ax is None else ax
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col)

# K=[1,5,10,17,25,32,40,45]
# file_paths = {1:'2023_01_05/k1', 5:'2023_01_04/k5', 10:'2023_01_03/k10', 17:'2023_01_05/k17', 25:'2023_01_03/k25', 32:'2023_01_05/k32', 40:'2023_01_03/k40', 45:'2023_01_05/k45'}
# alpha = 0.05 # significance level

# ITERATIONS = 50
# TARGET = targets()['square']

# High Res
K=[1,45,90]
file_paths = {1:'2023_01_30/square50/error_MI_k1', 45:'2023_01_30/square50/error_MI_k45', 90:'2023_01_31/square50_k90/error_MI'}
alpha = 0.05 # significance level

ITERATIONS = 100
constants.GRID_SIZE=50
TARGET = targets(grid_size=50)['square']

# Load in and compute the loss of the controls 
ctrl_dir = 'data/2023_01_30/square50'

bi_ctrl_loss_dict = extract_mean_of_best_from_trials(ctrl_dir+'/error/')
tri_ctrl_loss_dict = extract_mean_of_best_from_trials(ctrl_dir+'/error_phase1_error_phase2/')

labels = []
x_ticks = []

fig, ax = plt.subplots(1,1, figsize=(24,17))

# Compute 95% confidence interval
bi_loss_values = list(bi_ctrl_loss_dict.values())
ci95 = 1.96 * (np.std(bi_loss_values)/np.sqrt(len(bi_loss_values)))   
bi_loss_max_value = np.mean(bi_loss_values) + ci95
std = np.std(bi_loss_values) 
ax.bar(1,np.mean(bi_loss_values), color='silver', yerr=ci95,capsize=5)
labels.append('bi-loss')
x_ticks.append(1)

# Compute 95% confidence interval
tri_loss_values = list(tri_ctrl_loss_dict.values())
ci95 = 1.96 * (np.std(tri_loss_values)/np.sqrt(len(tri_loss_values)))   
tri_loss_max_value = np.mean(tri_loss_values) + ci95
std = np.std(tri_loss_values) 
ctrl_handle = ax.bar(2,np.mean(tri_loss_values), color='silver', yerr=ci95, capsize=5)
labels.append('tri-loss')
x_ticks.append(2)

index = 3
global_max_value = 0.0
h = 0.005

for i,k in enumerate(K):

    # Load in the appropriate runs
    # inpath = 'data/'+file_paths[k]+'/error_MI/'
    inpath = 'data/'+file_paths[k] +'/' # high res

    loss_dict = extract_mean_of_best_from_trials(inpath)
    values = list(loss_dict.values())

    # # Compute significance compared to controls - ttest
    # t, bi_loss_p = ttest_ind(bi_loss_values, values)
    # t, tri_loss_p = ttest_ind(tri_loss_values, values)

    # Compute significance compared to controls - ttest
    t, bi_loss_p = ranksums(bi_loss_values, values)
    t, tri_loss_p = ranksums(tri_loss_values, values)

    # Compute 95% confidence interval
    ci95 = 1.96 * (np.std(values)/np.sqrt(len(values)))   
    std = np.std(values) 
    max_value = np.mean(values)+ci95
    tx_handle = ax.bar(index, np.mean(values), color='dimgray', yerr=ci95, capsize=5)
    # annotate plot with significance
    bi_loss_sig = bi_loss_p<alpha/(len(K)*2)
    tri_loss_sig = tri_loss_p<alpha/(len(K)*2)
    if bi_loss_sig and tri_loss_sig:
        ax.text(index-0.075, max_value+h, '†', ha='center', va='bottom', color='k', fontsize=60)
        ax.text(index+0.075, max_value+h, '‡', ha='center', va='bottom', color='k', fontsize=60)
    elif bi_loss_sig:
        # global_max_value = np.max([max_value, global_max_value, bi_loss_max_value])
        # annot_stat(x1=1, x2=index, h = h, y = global_max_value+0.01, star='*')
        # max_point = global_max_value+0.01+h
        # global_max_value = np.max([global_max_value, max_point])
        ax.text(index, max_value+h, '†', ha='center', va='bottom', color='k', fontsize=60)
    elif tri_loss_sig:
        # global_max_value = np.max([max_value, global_max_value, tri_loss_max_value])
        # annot_stat(x1=2, x2=index, h = h, y = global_max_value+0.01, star='*')
        # max_point = global_max_value+0.01+h
        # global_max_value = np.max([global_max_value, max_point])
        ax.text(index, max_value+h, '‡', ha='center', va='bottom', color='k', fontsize=60)

    labels.append('k={}'.format(k))
    x_ticks.append(index)
    index+=1

# ax.legend(handles=[ctrl_handle, tx_handle], labels=['loss-only', 'loss+empowerment'], fontsize=30)
matplotlib.rcParams.update({'font.size': 60})
ax.set_xticks(x_ticks)
ax.set_xticklabels(labels, fontsize=60)
ax.set_ylabel('Loss', fontsize=60)
plt.yticks(fontsize=60)

# plt.show()
plt.savefig('gecco23_figs/ksweep_ranksums_highres.png', dpi=500, bbox_inches='tight')
plt.close()


