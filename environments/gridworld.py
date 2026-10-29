
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