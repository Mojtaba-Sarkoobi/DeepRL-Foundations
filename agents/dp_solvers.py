import numpy as np
import random

class InitialPolicy():
    """Generates baseline policies for evaluation."""
    def __init__(self, env):
        self.env = env
        
    def always_up_policy(self):
        up_policy = self.initial_policy
        for s in range(self.env.nS):
            up_policy[s][0] = 1.0
        return up_policy

    def always_right_policy(self):
        right_policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            right_policy[s][1] = 1.0
        return right_policy

    def always_down_policy(self):
        down_policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            down_policy[s][2] = 1.0
        return down_policy

    def always_left_policy(self):
        left_policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
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