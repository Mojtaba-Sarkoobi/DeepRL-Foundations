import numpy as np
import matplotlib.pyplot as plt

from environments.gridworld import StochasticGridWorld
from agents.dp_solvers import InitialPolicy, policy_evaluation, policy_iteration, value_iteration
from utils.plotting import get_policy_grid, plot_evaluation_policy, plot_policy_iteration_progress, plot_value_iteration_progress

if __name__ == "__main__":

    # Define environment
    ROWS = 4
    COLS = 4
    env = StochasticGridWorld(
        rows=ROWS, 
        cols=COLS, 
        start=0, 
        goal=15, 
        holes=[5, 7, 11, 12], 
        slip_chance=0.0
    )

    # Define initial policy
    policy = InitialPolicy(env)
    initial_policy = policy.always_left_policy()

    ##########################################################################################################
    print("*" * 60)
    print("Evaluating Initial Policy...")
    print("*" * 60)
    state_values, history, total_iteration = policy_evaluation(initial_policy, env, gamma=0.99)
    value_grid = state_values.reshape((env.rows, env.cols))

    print ("\nInitial state value (Grid View):")
    print(np.round(value_grid, 3))

    print("\nInitial policy grid (Direction to move):")
    initial_policy_grid = get_policy_grid(initial_policy, env)
    print(initial_policy_grid)
    plot_evaluation_policy(env, history)
    ##########################################################################################################
    print( "\n" + "*" * 60)
    print("Starting Policy Iteration...")
    print("*" * 60)
    optimal_policy, optimal_values, iterations, pi_history = policy_iteration(initial_policy, env, gamma=0.99)
    optimal_values_grid = np.round(optimal_values.reshape((env.rows, env.cols)), 3)
    print(f"Policy iteration converged in {iterations} iterations!")
    print("Policy iteration optimal State Values (Grid View):")
    print(optimal_values_grid)

    print("\nPolicy iteration grid (Direction to move):")
    optimal_policy_grid = get_policy_grid(optimal_policy, env)
    print(optimal_policy_grid)
    plot_policy_iteration_progress(env, pi_history)
    ##########################################################################################################
    print("\n" + "*" * 60)
    print("starting value iteration...")
    print("*" * 60)
    vi_policy, vi_values, vi_iterations, vi_history = value_iteration(env, gamma=0.99)
    vi_value_grid = np.round(vi_values.reshape((env.rows, env.cols)), 3)
    print(f"Value iteration converged in {vi_iterations} iterations!")
    print("Value Iteration Optimal State Values (Grid View):")
    print(vi_value_grid)

    print("\nvalue iteration grid (Direction to move):")
    vi_policy_grid = get_policy_grid(optimal_policy, env)
    print(vi_policy_grid)
    plot_value_iteration_progress(env, vi_history)
    ##########################################################################################################
    plt.show()