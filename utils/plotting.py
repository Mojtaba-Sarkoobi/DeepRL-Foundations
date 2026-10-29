import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def plot_evaluation_policy(env, history):

    plt.figure(figsize=(10, 6))
    
    for s in range(env.nS):
        plt.plot(history[:, s], label=f'State {s}', linewidth=2)
        
    # Chart styling and labels
    plt.title("State Values Converging Over Time")
    plt.xlabel("Number of Iterations")
    plt.ylabel("State Value (V)")
    plt.grid(True)
    
    # Place the legend outside the chart so it doesn't cover the lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_policy_iteration_progress(env, pi_history):

    plt.figure(figsize=(10,6))

    for s in range(env.nS):
        plt.plot(pi_history[:, s], marker='o', linestyle='--', label=f'State {s}', linewidth=2)

    plt.title("Policy Iteration: Value Jumps After Each Policy Improvement")
    plt.xlabel("Macro-Iterations (Policy Updates)")
    plt.ylabel("State Value (V)")
    plt.xticks(range(len(pi_history))) # Force x-axis to show whole integers
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_value_iteration_progress(env, history):

    plt.figure(figsize=(10, 6))
    for s in range(env.nS):
        plt.plot(history[:, s], label=f'State {s}', linewidth=2)

    plt.title("Value Iteration: Optimal State Values Converging Over Time")
    plt.xlabel("Number of Iterations")
    plt.ylabel("State Value (V)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def get_policy_grid(policy, env):
    """Converts a probability policy array into a readable directional grid."""
    policy_grid = []
    action_symbols = env.get_action_symbols()
    for s in range(env.nS):
        if s in env.holes:
            policy_grid.append('X')
        elif s == env.goal: 
            policy_grid.append('G')
        else:
            if np.all(policy[s] == policy[s][0]):
                policy_grid.append("*")
            else:
                best_action = np.argmax(policy[s])
                policy_grid.append(action_symbols[best_action])

    policy_grid = np.array(policy_grid).reshape((env.rows, env.cols))

    return policy_grid