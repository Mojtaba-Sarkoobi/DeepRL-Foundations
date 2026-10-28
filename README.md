# DeepRL Foundations

On here I want to learn and share my DeepRL code with others. 

This repository contains a complete, from-scratch Python implementation of core Dynamic Programming algorithms used in Reinforcement Learning, applied to a custom GridWorld environment.

## Features
* **Custom Stochastic Environment:** Configure grid dimensions, goal states, and trap states with a `slip_chance` variable to model probabilistic movement.
* **Algorithms Included:** Policy Evaluation, Policy Iteration, and Value Iteration.
* **Visualization:** Matplotlib integration to track and plot the convergence of state values over time.

## Usage
Run the script to execute the algorithms and generate the convergence charts:
```bash
python gridworld_dp.py