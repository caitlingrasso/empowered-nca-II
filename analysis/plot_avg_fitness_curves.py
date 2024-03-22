import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.individual import Individual
from src.visualizations import save_movie
import src.constants as constants

SAVE_DIR = 'exp9_SA_removed_local'
GENS = 1000
TITLE = ''

DIR = 'data/'+SAVE_DIR

txs = ['data/exp9_SA_removed_local/loss', 'data/exp9_SA_removed_local/loss_MI/k1', 'data/exp9_SA_removed_local/loss_MI/k45', 'data/exp9_SA_removed_local/loss_phase1_loss_phase2']
labels = ['bi-loss','tri-loss-empowerment (k=1)', 'tri-loss-empowerment (k=45)', 'tri-loss']
# color_tx_dict = {'loss':'tab:blue','loss_MI_k1':'tab:green'}
colors = ['blue', 'orange', 'green', 'red']

line_handles=[]

for j,tx in enumerate(txs):

    # results_dir = '{}/{}/*.p'.format(DIR, tx)
    results_dir=tx+'/*.p'

    filenames = glob(results_dir)

    if tx=='min_global_action_entropy':
        all_fits = np.zeros(shape=(len(filenames), 1000+1))
    else:
        all_fits = np.zeros(shape=(len(filenames), GENS+1))

    for i,fn in enumerate(filenames):

        print(fn)

        with open(fn, 'rb') as f:
            best, stats = pickle.load(f)
        fits_per_gen = stats[0]


        all_fits[i,:]=fits_per_gen

    avg_fits = np.mean(all_fits, axis=0)
    # 95% confidence intervals
    ci = 1.96 * (np.std(all_fits, axis=0)/np.sqrt(len(all_fits)))

    x = np.arange(len(avg_fits))
    line, = plt.semilogy(avg_fits, color=colors[j], linewidth=4)
    plt.fill_between(x, (avg_fits-ci), (avg_fits+ci), color=colors[j], alpha=.25)

    line_handles.append(line)

plt.legend(line_handles, labels, fontsize=15)
plt.xlabel('Generations', fontsize=25)
plt.ylabel('Loss', fontsize=25)
plt.title(TITLE, fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.show()

os.makedirs('results/{}'.format(SAVE_DIR), exist_ok=True)

plt.show()
# plt.savefig('results/{}/avg_fitness_curves_CI_semilog.png'.format(SAVE_DIR), dpi=300, bbox_inches='tight')
