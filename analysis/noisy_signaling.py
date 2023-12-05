'''
Created on 2023-01-30 11:40:28
@author: caitgrasso

Description: Adding Gaussian noise to signaling input during simulation. Are empowered NCAs 
more robust to noise in signals? Assess for different standard deviations. 
'''

import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualizations
import constants
from config import targets

# Set parameters
constants.ITERATIONS = 50
GRID_SIZE = 25
constants.GRID_SIZE = GRID_SIZE
TARGET = targets(GRID_SIZE)['square']
constants.SIGNAL_NOISE = True
constants.SIGNAL_NOISE_STD = .1
REPS = 10

# Square
data_dict = {0 : {'dir':'exp1_ksweep/error', 'label': 'bi-loss', 'color': 'lightblue', 'gens':2000},
             1 : {'dir':'exp1_ksweep/error_phase1_error_phase2', 'label': 'tri-loss', 'color': 'darkblue', 'gens':2000},
             2 : {'dir':'exp1_ksweep/k1/error_MI', 'label': 'tri-loss-empowerment', 'color': 'darkgreen', 'gens':2000},
             3 : {'dir':'exp5_addcontrols_square/min_action_entropy', 'label': 'tri-min_action_entropy', 'color': 'limegreen', 'gens':2000}
             }

labels = []
handles = []

for line in data_dict:

    filenames = glob('data/'+data_dict[line]['dir']+'/*.p')

    all_grid_losses = np.zeros((len(filenames)*REPS,constants.ITERATIONS+1))
    # all_grid_losses = np.zeros((len(filenames),constants.ITERATIONS+1))
    index = 0

    for i,fn in enumerate(filenames):

        try:
            with open(fn, 'rb') as f:
                best, stats = pickle.load(f)
        except:
            exit()

        for rep in range(REPS):
            
            np.random.seed(rep)

            history = best.playback()

            grid_losses = []

            for grid in history:

                binary_grid = grid[:,:,0]

                loss_at_t = best.evaluate_grid_diff(binary_grid, TARGET)/(constants.GRID_SIZE**2)

                grid_losses.append(loss_at_t)

            all_grid_losses[index,:]=grid_losses
            index+=1

    avg_grid_losses = np.mean(all_grid_losses, axis=0)
    line_handle, = plt.plot(avg_grid_losses, color=data_dict[line]['color'])
    ci = 1.96 * (np.std(all_grid_losses, axis=0)/np.sqrt(len(all_grid_losses)))
    x = np.arange(len(avg_grid_losses))
    plt.fill_between(x, (avg_grid_losses-ci), (avg_grid_losses+ci), color=data_dict[line]['color'], alpha=.25)

    handles.append(line_handle)
    labels.append(data_dict[line]['label'])

plt.title('Gaussian Noise, STD={}'.format(constants.SIGNAL_NOISE_STD))
# plt.title('No noise')
plt.legend(handles,labels)
# plt.show()
plt.savefig(f'results/noisy_signaling_{constants.SIGNAL_NOISE_STD}std.png', dpi=300, bbox_inches='tight')

