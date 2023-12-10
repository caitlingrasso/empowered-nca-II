'''
Created on 2023-01-12 17:28:23
@author: caitgrasso

Description: Plotting the relationship between loss and empowerment for different history lengths.
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genome import Genome
import constants
from config import targets


def extract_mean_of_best_from_trials(inpath,target,additional_obj):
    filenames = glob(inpath+'/*.p')
    
    best_fits_dict = {} # key = run number, value = best individual's loss from that run
    
    for filename in filenames:

        run = filename.split('run')[-1].split('.')[0]
        
        with open(filename, 'rb') as f:
            best,stats = pickle.load(f)

        best.evaluate(['error', additional_obj], target, 0)

        if additional_obj=='MI':
            best_fits_dict[run] = (best.get_objective(additional_obj)*-1, best.get_objective('error')) # undo minimization
        else:
            best_fits_dict[run] = (best.get_objective(additional_obj), best.get_objective('error'))

    return best_fits_dict

def compute_95_ci(xs,ys):
    meanx = np.mean(xs)
    ci95x = 1.96 * (np.std(xs)/np.sqrt(len(xs))) 

    meany = np.mean(ys)
    ci95y = 1.96 * (np.std(ys)/np.sqrt(len(ys))) 
    
    return meanx, ci95x, meany, ci95y

fig, ax = plt.subplots(1,1)
additional_obj = 'MI' # MI, min_action_entropy, min_global_action_entropy

plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15

# K=[1,5,10,17,25,32,40,45]
K=[1]
colors = ['tab:red', 'tab:green']
constants.HISTORY_LENGTH = 1
target = targets()['square']

# Load in and compute the loss of the controls 
ctrl_dir = 'data/exp1_ksweep'

bi_loss_xy_dict = extract_mean_of_best_from_trials(ctrl_dir+'/error/', target, additional_obj)
xs, ys = list(zip(*list(bi_loss_xy_dict.values())))
bi_loss_handle = ax.scatter(xs,ys,alpha=.7,color='tab:blue',edgecolors='none')
meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
ci95_handle = ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=bi_loss_handle.get_facecolor(), alpha=1)

tri_loss_xy_dict = extract_mean_of_best_from_trials(ctrl_dir+'/error_phase1_error_phase2/', target, additional_obj)
xs, ys = list(zip(*list(tri_loss_xy_dict.values())))
tri_loss_handle = ax.scatter(xs,ys,alpha=.7,color='tab:orange', edgecolors='none')
meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=tri_loss_handle.get_facecolor(), alpha=1)

labels = ['bi-loss', 'tri-loss']
handles = [bi_loss_handle, tri_loss_handle]

for i,k in enumerate(K):

    # Load in the appropriate runs
    inpath = 'data/exp1_ksweep/k{}/error_MI'.format(k)

    xy_dict = extract_mean_of_best_from_trials(inpath, target, additional_obj)
    xs, ys = list(zip(*list(xy_dict.values())))
    handle = ax.scatter(xs,ys,alpha=.7,color=colors[i],edgecolors='none')
    meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
    ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=handle.get_facecolor(), alpha=1)

    labels.append('loss + k={}'.format(k))
    handles.append(handle)

# Adding min. action entropy to the plot
MAE = extract_mean_of_best_from_trials('data/exp5_addcontrols_square/min_action_entropy', target, additional_obj)
xs, ys = list(zip(*list(MAE.values())))
MAE_handle = ax.scatter(xs,ys,alpha=.7,color='tab:green',edgecolors='none')
meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=MAE_handle.get_facecolor(), alpha=1)

handles.append(MAE_handle)
labels.append('loss + min. action entropy')

ax.legend(handles,labels, fontsize=15, ncol=2)
ax.set_ylabel('Loss', fontsize=20)
if additional_obj=='MI':
    ax.set_xlabel('Empowerment (k={})'.format(constants.HISTORY_LENGTH),fontsize=20)
# plt.show()

plt.savefig('results/exp5_additional_controls/loss_emp_scatter_k{}_MAE.png'.format(constants.HISTORY_LENGTH), dpi=300, bbox_inches='tight')
plt.close()


