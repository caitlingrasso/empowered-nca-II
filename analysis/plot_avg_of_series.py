import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genome import Genome
from visualizations import save_movie
import constants

SAVE_DIR = 'exp8_signaling_artifact_removed'
GENS = 2000
TITLE = ''
MEASURE = 'Loss' # 'Loss' or 'PF Size'

DIR = 'data/'+SAVE_DIR

txs = ['error', 'error_phase1_error_phase2', 'error_MI/k1']
labels = ['bi-loss','tri-loss', 'tri-loss-empowerment (k=1)']
color_tx_dict = {'error':'tab:blue','error_phase1_error_phase2':'tab:orange','error_MI/k1':'tab:green'}

line_handles=[]

for j,tx in enumerate(txs):

    results_dir = '{}/{}/*.p'.format(DIR, tx)

    filenames = glob(results_dir)

    all_vals = np.zeros(shape=(len(filenames), GENS+1))

    for i,fn in enumerate(filenames):

        print(fn)

        with open(fn, 'rb') as f:
            best, stats = pickle.load(f)

        if MEASURE == 'Loss':
            vals_per_gen = stats[0]
        elif MEASURE == 'PF Size':
            vals_per_gen = stats[-1]

        all_vals[i,:]=vals_per_gen

    avg_vals = np.mean(all_vals, axis=0)

    # 95% confidence intervals
    ci = 1.96 * (np.std(all_vals, axis=0)/np.sqrt(len(all_vals)))

    x = np.arange(len(avg_vals))
    # line, = plt.semilogy(avg_fits, color=color_tx_dict[tx], linewidth=4)
    line, = plt.plot(avg_vals, color=color_tx_dict[tx], linewidth=4)
    plt.fill_between(x, (avg_vals-ci), (avg_vals+ci), color=color_tx_dict[tx], alpha=.25)

    line_handles.append(line)

plt.legend(line_handles, labels, fontsize=15)
plt.xlabel('Generations', fontsize=25)
plt.ylabel(MEASURE, fontsize=25)
plt.title(TITLE, fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.show()


plt.show()
# os.makedirs('results/{}'.format(SAVE_DIR), exist_ok=True)

# plt.savefig('results/{}/avg_fitness_curves_CI_semilog.png'.format(SAVE_DIR), dpi=300, bbox_inches='tight')
