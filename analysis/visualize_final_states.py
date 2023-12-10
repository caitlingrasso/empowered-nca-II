import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualizations
import constants
from config import targets

# Set parameters
constants.HISTORY_LENGTH=1
GRID_SIZE = 25
constants.GRID_SIZE=GRID_SIZE
TARGET = targets(GRID_SIZE)['square']
ITERATIONS = 50

filenames = glob('data/exp1_ksweep/k45/error_MI/*.p')

save_path = 'results/final_states/exp1_ksweep/k45/error_MI'
os.makedirs(save_path,exist_ok=True)

for filename in filenames:
    
    save_filename = save_path+'/'+filename.split('/')[-1].split('.')[0]+'.png'

    with open(filename, 'rb') as f:
        best,stats = pickle.load(f)

    # Run NCA as in training
    train_history = best.playback(iterations=ITERATIONS)

    # Initial state
    init_grid = train_history[0]

    # Final state
    final_state = train_history[-1]

    # Visualize initial condition
    # visualizations.display_body_signal(init_grid, target=TARGET, title='initial condition', original_size=GRID_SIZE)

    # Visualize final grid state of the NCA
    visualizations.display_body_signal(final_state, target=TARGET, title='final grid condition', original_size=GRID_SIZE,save=True, show=False, fn=save_filename)
