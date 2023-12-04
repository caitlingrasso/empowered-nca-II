import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys,os
from collections import Counter
import matplotlib
import matplotlib.cm as cm
from matplotlib.animation import FFMpegWriter
from collections import Counter
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualizations
import constants
from config import targets

def shannon_entropy(X):
    # H(X)
    x_freq_dict = dict(Counter(X))

    Hx = 0

    for x in list(x_freq_dict.keys()):
        px = x_freq_dict[x]/len(X)

        Hx += px * np.log2(1/px)
    
    return Hx

def conditional_shannon_entropy(X,Y):
    # H(X|Y)
    x_freq_dict = dict(Counter(X))
    y_freq_dict = dict(Counter(Y))
    joint_freq_dict = dict(Counter(zip(X,Y)))

    Hx_y = 0

    for y in list(y_freq_dict.keys()):
        for x in list(x_freq_dict.keys()):
            py = y_freq_dict[y]/len(Y)

            try:
                pxy = joint_freq_dict[(x,y)]/len(X)
            except:
                continue

            Hx_y += pxy * np.log2(py/pxy)

    return Hx_y

# Set parameters
constants.HISTORY_LENGTH=1
GRID_SIZE = 25
constants.GRID_SIZE = GRID_SIZE
TARGET = targets(GRID_SIZE)['square']
ITERATIONS = 50

data_dict = {
    0: {'dir':'data/exp1_ksweep/error', 'color':'blue', 'label': 'BL'},
    1: {'dir':'data/exp1_ksweep/error_phase1_error_phase2', 'color':'orange', 'label': 'TL'},
    2: {'dir':'data/exp1_ksweep/k1/error_MI', 'color':'red', 'label': 'TLE-k=1'},
    3: {'dir':'data/exp1_ksweep/k45/error_MI', 'color':'cyan', 'label': 'TLE-k45'},
    4: {'dir':'data/exp5_addcontrols_square/min_action_entropy', 'color':'cyan', 'label': 'MAE'}
}


all_entropies1 = []
all_entropies2 = []
labels = []

for tx in data_dict:

    filenames = glob(data_dict[tx]['dir']+'/*.p')
    
    entropies1 = []
    entropies2 = []

    for filename in filenames:
        try:
            with open(filename, 'rb') as f:
                stuff = pickle.load(f)
                best = stuff[-2]
        except:
            with open(filename, 'rb') as f:
                best,stats = pickle.load(f)
        
        best.evaluate(objectives=['error', 'MI'], target=TARGET, g=0)
        # print(best.MI*-1)

        # Run NCA
        history, sensors_timeseries, actions_timeseries = best.playback(iterations=ITERATIONS, return_SA=True)

        # extract sensory values to a matrix
        sensor_series_mat = np.zeros(shape=(constants.GRID_SIZE*constants.GRID_SIZE, len(sensors_timeseries[constants.HISTORY_LENGTH:])), dtype=int)
        for t, sensor_mat in enumerate(sensors_timeseries[constants.HISTORY_LENGTH:]):
            
            sensor_mat = sensor_mat[1:-1, 1:-1,:] # remove padding

            cell_sensor_list = np.mean(sensor_mat[:,:,constants.NEIGHBORHOOD+1:],axis=2).flatten().astype(int)
            
            for i,sensor_val in enumerate(cell_sensor_list):

                sensor_series_mat[i,t] = sensor_val

        # print(sensor_series_mat.shape)

        # extract action values to a matrix
        action_series_mat = np.zeros(shape=(constants.GRID_SIZE*constants.GRID_SIZE, len(actions_timeseries[:-constants.HISTORY_LENGTH])), dtype=int)
        for t, action_mat in enumerate(actions_timeseries[:-constants.HISTORY_LENGTH]):
            
            action_mat = action_mat[1:-1, 1:-1,:] # remove padding

            cell_action_list = action_mat[:,:,-1].flatten().astype(int)
            
            for i,action_val in enumerate(cell_action_list):

                action_series_mat[i,t] = action_val

        # print(action_series_mat.shape)

        # Construct action sensor pairs
        pairs = dict(Counter(zip(action_series_mat.flatten(), sensor_series_mat.flatten())))

        # global
        H_A = shannon_entropy(action_series_mat.flatten())
        # H_S = shannon_entropy(sensor_series_mat.flatten())
        H_AS = conditional_shannon_entropy(action_series_mat.flatten(), sensor_series_mat.flatten())
        # H_SA = conditional_shannon_entropy(sensor_series_mat.flatten(), action_series_mat.flatten())

        entropies1.append(H_A)
        entropies2.append(H_AS)
        

        # local
        # cell_entropies = []
        # for cell in range(action_series_mat.shape[0]): # iterate through rows (i.e. cells)
        #     # cell_entropies.append(shannon_entropy(sensor_series_mat[cell,:]))
        #     cell_entropies.append(conditional_shannon_entropy(sensor_series_mat[cell,:], action_series_mat[cell,:]))

        # entropies.append(np.mean(cell_entropies))

    all_entropies1.append(entropies1)
    all_entropies2.append(entropies2)
    labels.append(data_dict[tx]['label'])

fig, ax = plt.subplots(figsize=(7,3),layout='constrained')

width = 0.2
multiplier = 0

x1 = np.arange(1,len(labels)+1)
plot1 = plt.boxplot(all_entropies1, positions=x1, widths=0.2, patch_artist=True)

x2 = [x+0.25 for x in x1]
plot2 = plt.boxplot(all_entropies2,
                            positions=x2, widths=0.2, patch_artist=True)

for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(plot1[item], color='dimgray') 
plt.setp(plot1["boxes"], facecolor = 'lightgray')
plt.setp(plot1["fliers"], markeredgecolor='dimgray')

for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(plot2[item], color='darkred')
plt.setp(plot2["boxes"], facecolor = 'indianred')
plt.setp(plot2["fliers"], markeredgecolor='darkred')

xticks = [x+0.125 for x in x1]
plt.xticks(xticks, labels, fontweight='bold', fontsize=15)
# plt.ylim([0, h+0.25])
ax.tick_params(axis='y', which='major', labelsize=12)

plt.ylabel('Entropy (bits)', fontweight='bold', fontsize=10)

# Statistical annotations
for i in [0,1,3,4]:

    stat, p1 = ttest_ind(all_entropies1[i], all_entropies1[2])
    stat, p2 = ttest_ind(all_entropies2[i], all_entropies2[2])
    print(p1)
    print(p2)
    print()


# for i,p in enumerate(ps):
#     ax.plot((before_x[i],after_x[i]),(h, h),'k')

#     if p<0.001:
#         text = 'p<0.001'  
#     elif p<0.01:
#         text = 'p<0.01' 
#     elif p<0.05:
#         text = 'p<0.05' 
#     else:
#         text = 'NS'

#     ax.text(xticks[i], h+0.02, text, style='italic', ha='center')

handles = [Patch(facecolor='lightgray', edgecolor='dimgray', label='H(A)'), 
        Patch(facecolor='indianred', edgecolor='darkred', label='H(A|S)')]

# handles = [Patch(facecolor='lightgray', edgecolor='dimgray', label='before')]

ax.legend(handles=handles, handlelength=2, handleheight=1, frameon=False, bbox_to_anchor=(1.05, 1.0), loc='upper left',
            borderaxespad=0, fontsize=13)

plt.title('Global Action Entropy')

plt.savefig('results/global_HA_HAS_boxplot.png', dpi=300, bbox_inches='tight')

plt.show()