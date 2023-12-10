import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import sort_by
from config import targets

def get_best(DIR, objective):
    return sort_by(DIR, objective)

def get_worst(DIR, objective):
    return sort_by(DIR, objective, reverse=True)

SAVE_DIR = 'exp3_upscale/'
TXS = ['error_phase1_error_phase2']
TARGET = targets(grid_size=50)['square']
METRIC = "error"

print('METRIC:', METRIC)
print()

for tx in TXS:
    
    # BEST:
    DIR = 'data/'+SAVE_DIR+'/'+tx+'/'
    sorted_dict = get_best(DIR+'*.p', METRIC)

    print('TX:', tx)

    print('BEST:', sorted_dict[0][0], ':', sorted_dict[0][1])
    print()


    # # WORST:
    # DIR = 'data/'+SAVE_DIR+'/'+tx+'/'
    # sorted_dict = get_worst(DIR+'*.p', METRIC)

    # print('WORST:', sorted_dict[0][0], ':', sorted_dict[0][1])
    # print()
