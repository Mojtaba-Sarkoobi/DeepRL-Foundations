"""
gridworld_dp.py

A complete implementation of Dynamic Programming algorithms (Policy Evaluation, 
Policy Iteration, and Value Iteration) applied to a customizable Stochastic GridWorld.

Author: Mojtaba Sarkoobi
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class StochasticGridWorld:
    """
    A customizable GridWorld environment.
    If slip_chance > 0, the agent has a small probability of slipping left 
    or right relative to its intended direction of movement.
    """

    def __init__(self, rows, cols, start, goal, holes, slip_chance = 0.1):
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4
        self.start = start
        self.goal = goal
        self.holes = holes
        self.slip_chane = slip_chance

        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}

        self.P = {s: {a : [] for a in range(self.nA)} for s in range(self.nS)}

        self._build_transition_model()

    def _get_next_state(self, s, a):
        """Calculates the resulting state given a starting state and an action."""
        row = s // self.cols
        col = s % self.cols

        if a == 0 and row > 0: # Up
            return s - self.cols
        elif a == 1 and col < self.cols - 1: # Right
            return s + 1
        elif a == 2 and row < self.rows - 1: # Down
            return s + self.cols
        elif a == 3 and col > 0: # Left
            return s - 1

        # Hit a wall, stay in a current state
        return s 

    def _build_transition_model(self):
        """Constructs the transition dynamics dictionary P[state][action]."""
        for s in range(self.nS):
            if s == self.goal or s in self.holes:
                for a in range(self.nA):
                    self.P[s][a] = [(1.0, s, 0.0, True)]
                continue

            # None terminal states
            for a in range(self.nA):
                outcomes = []
                # Calculate the probabilities of intended vs slipped movement
                intended_prob = 1.0 - (self.slip_chane * (self.nA - 1))
                intended_s = self._get_next_state(s, a)
                outcomes.append((intended_prob, intended_s))

                for a_slipped in range(self.nA):
                    if a_slipped == a:
                        continue
                    slipped_a_prime_s = self._get_next_state(s, a_slipped)    
                    outcomes.append((self.slip_chane, slipped_a_prime_s))

                # Consolidate duplicates (wall bumps)
                consolidated = {}
                for prob, next_s in outcomes:
                    if next_s not in consolidated:
                        consolidated[next_s] = 0.0
                    consolidated[next_s] += prob
                # Assign rewards
                for next_s, prob in consolidated.items():
                    if prob > 0:
                        reward = 0.0
                        done = False

                        if next_s == self.goal:
                            reward = 1.0
                            done = True
                        elif next_s in self.holes:
                            reward = -1.0
                            done = True

                        self.P[s][a].append((prob, next_s, reward, done))


    def get_action_symbols(self):
        return self.action_symbols

# ==============================================================================
# POLICY GENERATOR
# ==============================================================================

class InitialPolicy():
    """Generates baseline policies for evaluation."""
    def __init__(self, env):
        self.env = env
        
    def always_up_policy(self):
        up_policy = self.initial_policy
        for s in range(env.nS):
            up_policy[s][0] = 1.0
        return up_policy

    def always_right_policy(self):
        right_policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            right_policy[s][1] = 1.0
        return right_policy

    def always_down_policy(self):
        down_policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            down_policy[s][2] = 1.0
        return down_policy

    def always_left_policy(self):
        left_policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            left_policy[s][3] = 1.0
        return left_policy

    
    def generate_deterministic_random_policy(self):
        # Assigns 100% probability to ONE random action per state.
        random_policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            random_action = random.choice([0, 1, 2, 3])
            random_policy[s][random_action] = 1.0
        return random_policy

    # Assigns 25% probability to ALL FOUR actions per state.
    def generate_uniform_random_policy(self):
        # Assigns 25% probability to ALL FOUR actions per state.
        uniform_policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        return uniform_policy   

# ==============================================================================
# ALGORITHMS
# ==============================================================================

def policy_evaluation(policy, env, gamma=0.99, theta=1e-6):
    """Calculates the state-value function V for a given policy."""
    V = np.zeros(env.nS)
    history = [V.copy()]
    iteration = 0

    while True:
        delta = 0
        iteration += 1

        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    future_value = V[next_state] if not done else 0
                    v += action_prob * prob * (reward + gamma * future_value)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        history.append(V.copy())
        if delta < theta:
            break

    return V, np.array(history), iteration

def policy_improvement(V, env, gamma=0.99):
    """Extracts the greedy policy with respect to the current value function V."""
    new_policy = np.zeros([env.nS, env.nA])

    for s in range(env.nS):

        # Array to store the expected return (Q-value) for all for actions in this state
        q_values = np.zeros(env.nA)

        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                future_value = V[next_state] if not done else 0
                q_values[a] += prob * (reward + gamma * future_value)

        best_action = np.argmax(q_values)

        # Update the new policy to ALWAYS take the best action
        new_policy[s][best_action] = 1.0

    return new_policy

def policy_iteration(initial_policy, env, gamma=0.99, theta=1e-6):
    """Repeats evaluation and improvement until the optimal policy converges."""
    policy = initial_policy.copy()
    iteration_count = 0
    pi_history = []
    while True:
        iteration_count += 1
        
        V, _, _ = policy_evaluation(policy, env, gamma, theta)

        pi_history.append(V.copy())
        new_policy = policy_improvement(V, env, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V, iteration_count, np.array(pi_history)

def value_iteration(env, gamma=0.99, theta=1e-6):
    """Computes the optimal value function by aggressively taking the max action."""
    V = np.zeros(env.nS)
    history = [V.copy()]
    iteration_count = 0
    while True:
        delta = 0
        iteration_count += 1

        for s in range(env.nS):

            if s == env.goal or s in env.holes:
                continue

            q_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    future_value = V[next_state] if not done else 0
                    q_values[a] += prob * (reward + gamma * future_value)

            best_action_balue = np.max(q_values)
            delta = max(delta, np.abs(best_action_balue - V[s]))
            V[s] = best_action_balue

        history.append(V.copy())

        if delta < theta:
            break

    optimal_policy = policy_improvement(V, env, gamma)

    return optimal_policy, V, iteration_count, np.array(history)

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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

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
