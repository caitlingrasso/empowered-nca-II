'''
Created on 2023-01-30 13:26:57
@author: caitgrasso

Description: Start evolutionary runs but set the initial population
to previously trained NCAs.
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import argparse
import time
import os
import pickle
from glob import glob

from src.targets import targets, init
import src.constants as constants
from src.optimizer import Optimizer
from src.visualizations import display_body_signal, display_grid
from src.ca_model import CA_MODEL

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--gens', default=constants.GENERATIONS, type=int)
    parser.add_argument('--popsize', default=constants.POP_SIZE, type=int)
    parser.add_argument('--target', default='square')
    parser.add_argument('--objective1', default='error')
    parser.add_argument('--objective2', default=None)
    parser.add_argument('--checkpoint_every', default=500, type=int)
    return parser.parse_args(args)

if __name__=="__main__":

    start_time = time.time()

    # Set history length 
    constants.HISTORY_LENGTH = 1

    args = sys.argv[1:]
    args = parse_args(args)

    # PATH_TO_CHECKPOINT = 'checkpoints/2023_01_03/k25/error_square25_250gens_400ps_50i_k25_run{}_checkpoint_2000gen.p'.format(args.run)
    PATH_TO_CHECKPOINT = 'checkpoints/2023_01_03/k25/error_phase1_error_phase2_square25_250gens_400ps_50i_k25_run{}_checkpoint_2000gen.p'.format(args.run)
    # PATH_TO_CHECKPOINT = 'checkpoints/2023_01_05/k1/error_MI_square25_2000gens_400ps_50i_k1_N4_run{}_checkpoint_2000gen.p'.format(args.run)
    # PATH_TO_CHECKPOINT = 'checkpoints/2023_01_05/k45/error_MI_square25_2000gens_400ps_50i_k45_N4_run{}_checkpoint_2000gen.p'.format(args.run)
    
    SAVE_TAG = 'pretrain_2023_01_03_error_phase1_error_phase2_2000gens'

    # set seed
    np.random.seed(args.run)
    random.seed(args.run)

    # set objectives
    if args.objective2 is None or args.objective2=='None':
        objectives= ['age', args.objective1] 
        obj2 = ''
    else:
        objectives = ['age', args.objective1, args.objective2]
        obj2 = args.objective2+'_'


    with open(PATH_TO_CHECKPOINT, 'rb') as f:
        optimizer, rng_state, np_rng_state = pickle.load(f)

    # Extract weights of all individuals in the population
    weights = []
    for i in optimizer.population:
        ind = optimizer.population[i]
        weights.append(ind.genome.model)

    # Save info of every individual for AFPO rainbow waterfall plot for first run only
    if args.run==1: 
        constants.SAVE_ALL=True

    # set target
    target_str = args.target
    targ=targets()[target_str]

    # # Plotting initial and target grid conditions
    # signal = np.zeros(targ.shape)
    # targ = np.concatenate((targ[:,:,None], signal[:,:,None]), axis=2)
    # signal[constants.GRID_SIZE//2,constants.GRID_SIZE//2]=255
    # init = np.concatenate((init[:,:,None], signal[:,:,None]), axis=2)
    # display_body_signal(init, save=True, fn='shapes/seed_{}_initial_condition.png'.format(constants.GRID_SIZE))
    # display_body_signal(targ, save=True, fn='shapes/{}_{}_target.png'.format(target_str, constants.GRID_SIZE))
    # exit()

    # Filenames
    prefix = '{}_finetune_{}_{}{}{}_{}gens_{}ps_{}i_k{}_N{}_run{}'.format(SAVE_TAG, args.objective1, obj2, target_str,constants.GRID_SIZE, args.gens, args.popsize, \
        constants.ITERATIONS, constants.HISTORY_LENGTH, constants.NEIGHBORHOOD, args.run)
    save_filename = 'data/{}.p'.format(prefix)
    save_all_dir = 'arwp/{}'.format(prefix)
    checkpoint_dir = 'checkpoints/{}'.format(prefix)

    # make directories for data
    os.makedirs('data/', exist_ok=True)
    os.makedirs('arwp/', exist_ok=True)
    os.makedirs('checkpoints/', exist_ok=True)

    # START RUN
    optimizer = Optimizer(target=targ, objectives=objectives, gens=args.gens, pop_size=args.popsize, 
                        checkpoint_every=args.checkpoint_every, save_all_dir=save_all_dir,
                        checkpoint_dir=checkpoint_dir, run_nbr=args.run, init_grid=init(), NN_seed_weights=weights)

    best, stats = optimizer.run() # stats = [fits_per_gen, empowerment_per_gen, pareto_front, pf_sizes_per_gen]
    
    # For tri-objective:
    # stats = [fits_per_gen, empowerment_per_gen, obj1_score_per_gen, obj2_score_per_gen, pareto_front, pf_sizes_per_gen]
    
    # Save data from run 
    f = open(save_filename, 'wb')
    pickle.dump([best,stats], f)
    f.close()

    end_time = time.time()
    print('--', (end_time-start_time)/3600, 'hours --')  
        