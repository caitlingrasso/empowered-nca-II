# Selection for short-term empowerment accelerates the evolution of homeostatic neural cellular automata

This repository contains source code for the GECCO'23 (Genetic and Evolutionary Computation Conference 2023) paper 

[Caitlin Grasso and Josh Bongard (2023). *Selection for short-term empowerment accelerates the evolution of homeostatic neural cellular automata.*](https://arxiv.org/abs/2305.15220)</br> 

<p align="center">
  <img src="https://github.com/caitlingrasso/empowered-nca-II/blob/master/methods_fig.png?raw=true"  width="600" height="550">
</p>

## Structure

- `main.py` - launches an evolutionary algorithm that performs bi- or tri-objective search of NCA capable of morphogenesis. Specify the parameters of the EA as command line input. For example:
```
python main.py --run=1 --gens=2000 --popsize=400 --target=square --objective1=error --objective2=MI --k=1
```
&ensp; &nbsp; &nbsp; See notes below regarding the objective names.

- `constants.py` - allows you to set certain constants such as the number of cellular automata (CA) iterations, the CA grid size, # EA generations, EA population size, and more. Parameters set as input to `main.py` will overwrite the values in `constants.py`.

- `config.py` - holds the target shapes. To create and test additional target shapes, add them to this file.

- `visualize_sim.ipynb` - python notebook to visualize a CA simulation and save out a video.

- `analysis/sort_by_objective.py` - sorts results from the EA and returns the best (or worst) performing NCA on the particular objective given as input.

- `analysis/` - directory containing all code to analyze various EA runs and produce the results/plots in the paper.
  - In particular, `analysis/plot_avg_fitness_curves.py` is using to generate the loss plots seen throughout the paper.

- `additional_objectives.py` contain the additional objectives that were tested in Figure 9 of the paper.

## Experiments
### 1. Morphogenesis (Fig. 2)

History length (k) sweep.

Run `main.py` for each of the evolutionary variations as follows:

**Bi-loss**
```
python main.py --runs=[0...35] --gens=2000 --popsize=400 --target=square --objective1=error
```
**Tri-loss**
```
python main.py --runs=[0...35] --gens=2000 --popsize=400 --target=square --objective1=error_phase1 --objective2=error_phase2
```
**Tri-loss-empowerment (with varying k)**
```
python main.py --runs=[0...35] --gens=2000 --popsize=400 --target=square --objective1=error --objective2=MI --k=[1,5,10,17,25,32,40,45]
```

Evolution is run for each value in the range or set of values indicated by the brackets. Thus, for the bi-loss and tri-loss trials, 35 calls to `main.py` are made. For the tri-loss-empowerment trials, 35x8=280 calls to `main.py` are made.

---
### 2. Homeostasis (Figs 3 & 4)

- `analysis/plot_extended_sim_stability_all.py` produces results related to stability if Figures 3 & 4. 

- `analysis/morphogenetic_homeostatic_characteristics.py` generates the bar plots in Figure 4 relating to # connected components, cell state transiency, and proportion of cells on boundary

---
### 3. Generalization (Figs 5 & 6)

1. Generalization to different shapes with empowerment pre-trained NCA (Fig 5). 
- `finetune.py` is similar to `main.py` but "continues" evolution from a checkpointed generation (i.e. evolution does not start with a random population)
- To produce the generalization results in Figure 5, the runs described in part 1 of this section must already be performed (for the tri-loss-empowerment variation only k=1 needs to be run). `finetune.py` can then be edited to point to a checkpointed file (by setting the `PATH_TO_CHECKPOINT` variable) from one of these runs and evolution can be continued.
- The `--target` option can be set to change the target shape to the triangle, x, or biped.

2. Generalization to larger grid (Fig 6).
- Experiments described in part 1 of this section were repeated but with the following changes to `constants.py`
```
GRID_SIZE = 50
ITERATIONS = 100
```
- Only three values of history length were tested: k=[1,45,90]

### Notes: 
1. `python3` required to run the code in this repository.
2. `error` is the name of the loss metric and `MI` is the name of the empowerment metric. Thus, runs labeled `error_MI` in the code are called tri-loss-empowerment runs in the paper (tri-objective search with the first objective as age, the second as loss, and the third as empowerment). A table mapping the objective syntax to the EA variations described in the paper is below.

<div align="center">
  
| EA Variation      | Objective 1 | Objective 2 |
| ----------- | ----------- | ----------- |
| bi-loss      | `error`       | - |
| tri-loss   | `error_phase1`        | `error_phase2` |
| tri-loss-empowerment   |  `error`  | `MI` |  

</div>

4. The `run` parameter input to `main.py` is the random seed for the evolutionary run.

