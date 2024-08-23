# EMBC: Episodic Memory-Guided Behavior Cloning

## Overview
This is the official implementation of **Episodic Memory-Guided Behavior Cloning** (**EMBC**).

EMBC is a supervised learning based offline policy learning method, which can learn task-agnostic general decision components from suboptimal datasets and use them to imitate expert behavior with very few expert demonstrations.


## Files
* `main.py`: the main code of EMBC, it is the entrance for reproducing the EMBC algorithm.
* `gcsl.py (goal-conditional supersived learning)`: code implementation for training the goal-conditional actor and the reachability network.
* `planning_graph.py`: code for action inference, including code for 1) selecting subgoal from expert demonstrations and 2) running shortest-path algorithm to synthesize expert trajectories from graph.
* `construct_graph.py`: transform trajectories datasets (states) into graph.
* `dataset_utils.py`: data preprocessing.
* `./config`: hyperparameters configuration.
