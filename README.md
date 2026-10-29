# DeepRL Foundations

On here I want to learn and share my DeepRL code with others. 

This repository contains a complete, from-scratch Python implementation of core Dynamic Programming algorithms used in Reinforcement Learning, applied to a custom GridWorld environment. The project is structured with modularity in mind, separating environment dynamics from agent solvers to easily support future algorithms.

## Project Structure
* `environments/`: Contains the `StochasticGridWorld` class with customizable dimensions, goals, holes, and slip probabilities.
* `agents/`: Contains the Dynamic Programming solvers (`dp_solvers.py`) and Model-Free exploration strategies (`exploration.py`).
* `utils/`: Contains Matplotlib visualization tools for plotting convergence metrics.

## Features
- **Custom Stochastic Environment:** Configure grid dimensions, goal states, and trap states with a `slip_chance` variable to model probabilistic movement.
- **Algorithms Included:** Policy Evaluation, Policy Iteration, and Value Iteration.
- **Visualization:** Matplotlib integration to track and plot the convergence of state values over time.

## Example Output
When executing a 4x4 grid with a goal at state 15 and traps at states 5, 7, 11, and 12, both algorithms successfully converge on identical optimal policies:
```text
************************************************************
Evaluating Initial Policy...
************************************************************

Initial state value (Grid View):
[[ 0.    0.    0.    0.  ]
 [ 0.    0.   -1.    0.  ]
 [ 0.    0.    0.    0.  ]
 [ 0.   -1.   -0.99  0.  ]]

Initial policy grid (Direction to move):
[['←' '←' '←' '←']
 ['←' 'X' '←' 'X']
 ['←' '←' '←' 'X']
 ['X' '←' '←' 'G']]

************************************************************
Starting Policy Iteration...
************************************************************
Policy iteration converged in 7 iterations!
Policy iteration optimal State Values (Grid View):
[[0.951 0.961 0.97  0.961]
 [0.961 0.    0.98  0.   ]
 [0.97  0.98  0.99  0.   ]
 [0.    0.99  1.    0.   ]]

Policy iteration grid (Direction to move):
[['→' '→' '↓' '←']
 ['↓' 'X' '↓' 'X']
 ['→' '→' '↓' 'X']
 ['X' '→' '→' 'G']]

************************************************************
starting value iteration...
************************************************************
Value iteration converged in 7 iterations!
Value Iteration Optimal State Values (Grid View):
[[0.951 0.961 0.97  0.961]
 [0.961 0.    0.98  0.   ]
 [0.97  0.98  0.99  0.   ]
 [0.    0.99  1.    0.   ]]

value iteration grid (Direction to move):
[['→' '→' '↓' '←']
 ['↓' 'X' '↓' 'X']
 ['→' '→' '↓' 'X']
 ['X' '→' '→' 'G']]
```

## Usage
Run the main script to execute the algorithms sequentially and generate the convergence charts:
```bash
python main.py
```