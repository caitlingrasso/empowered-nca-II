# Selection for short-term empowerment accelerates the evolution of homeostatic neural cellular automata

This repository contains source code for the GECCO'23 (Genetic and Evolutionary Computation Conference 2023) paper 

[Caitlin Grasso and Josh Bongard (2023). *Selection for short-term empowerment accelerates the evolution of homeostatic neural cellular automata.*](https://arxiv.org/abs/2305.15220)</br> 

<p align="center">
  <img src="https://github.com/caitlingrasso/empowered-nca-II/blob/master/methods_fig.png?raw=true"  width="350" height="600">
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

## Experiments

### Notes: 
1. `python3` required to run the code in this repository.
2. `error` is the name of the loss metric and `MI` is the name of the empowerment metric. Thus, runs labeled `error_MI` in the code are called tri-loss-empowerment runs in the paper (tri-objective search with the first objective as age, the second as loss, and the third as empowerment). A table mapping the objective syntax to the EA variations described in the paper is below.

<div align="center">
  
| EA Run      | Objective 1 | Objective 2 |
| ----------- | ----------- | ----------- |
| bi-loss      | `error`       | - |
| tri-loss   | `error_phase1`        | `error_phase2` |
| tri-loss-empowerment   |  `error`  | `MI` |  

</div>

4. The `run` parameter input to `main.py` is the random seed for the evolutionary run.

