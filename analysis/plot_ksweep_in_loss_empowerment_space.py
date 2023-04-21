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


def extract_mean_of_best_from_trials(inpath):
    filenames = glob(inpath+'*.p')
    
    best_fits_dict = {} # key = run number, value = best individual's loss from that run
    
    for filename in filenames:

        run = filename.split('run')[-1].split('.')[0]
        
        with open(filename, 'rb') as f:
            best,stats = pickle.load(f)

        best_fits_dict[run] = (best.get_objective('MI')*-1, best.get_objective('error'))

    return best_fits_dict

def compute_95_ci(xs,ys):
    meanx = np.mean(xs)
    ci95x = 1.96 * (np.std(xs)/np.sqrt(len(xs))) 

    meany = np.mean(ys)
    ci95y = 1.96 * (np.std(ys)/np.sqrt(len(ys))) 
    
    return meanx, ci95x, meany, ci95y

fig, ax = plt.subplots(1,1)

K=[1,5,10,17,25,32,40,45]
file_paths = {1:'2023_01_05/k1', 5:'2023_01_04/k5', 10:'2023_01_03/k10', 17:'2023_01_05/k17', 25:'2023_01_03/k25', 32:'2023_01_05/k32', 40:'2023_01_03/k40', 45:'2023_01_05/k45'}

# Load in and compute the loss of the controls 
ctrl_dir = 'data/2023_01_03/k25'

bi_loss_xy_dict = extract_mean_of_best_from_trials(ctrl_dir+'/error/')
xs, ys = list(zip(*list(bi_loss_xy_dict.values())))
bi_loss_handle = ax.scatter(xs,ys,alpha=.7,edgecolors='none')
meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
ci95_handle = ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=bi_loss_handle.get_facecolor(), alpha=1)

tri_loss_xy_dict = extract_mean_of_best_from_trials(ctrl_dir+'/error_phase1_error_phase2/')
xs, ys = list(zip(*list(tri_loss_xy_dict.values())))
tri_loss_handle = ax.scatter(xs,ys,alpha=.7,edgecolors='none')
meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=tri_loss_handle.get_facecolor(), alpha=1)

labels = ['bi-loss', '95% CI', 'tri-loss']
handles = [bi_loss_handle, ci95_handle, tri_loss_handle]

for i,k in enumerate(K):

    # Load in the appropriate runs
    inpath = 'data/'+file_paths[k]+'/error_MI/'

    xy_dict = extract_mean_of_best_from_trials(inpath)
    xs, ys = list(zip(*list(xy_dict.values())))
    handle = ax.scatter(xs,ys,alpha=.7,edgecolors='none')
    meanx, ci95x, meany, ci95y = compute_95_ci(xs,ys)
    ax.errorbar(meanx, meany, xerr=ci95x, yerr=ci95y,fmt="o", capsize=3, c=handle.get_facecolor(), alpha=1)

    labels.append('tri-loss-emp, k={}'.format(k))
    handles.append(handle)

    # Compute 95% confidence interval
    # ci95 = 1.96 * (np.std(diffs)/np.sqrt(len(diffs)))   
    # std = np.std(diffs) 
    
    # ax.scatter((k,)*len(diffs), diffs, c='orange')
    # line_handle_means = ax.scatter(k, np.mean(diffs), c='blue')
    # line_handle_err = ax.errorbar(k, np.mean(diffs), yerr=ci95, fmt="o", c='royalblue', capsize=3)

# ax.legend(handles = [line_handle_err], labels=['95% CI'])
# ax.plot(np.linspace(0, 50, num=100), (0,)*100, '-r')
ax.legend(handles,labels)
ax.set_ylabel('Loss')
ax.set_xlabel('Empowerment')

plt.savefig('results/summary/historyLengthSweep_loss_empowerment_space.png', dpi=300, bbox_inches='tight')
plt.close()


