import numpy as np
import random
from collections import defaultdict, deque

GRID_SIZE = 8                                   #grid size (8x8)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_distance(p1, p2):
    """
    Manhattan distance
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def move(position, action):
    """
    Move agent inside the grid
    """
    x, y = position

    if ACTIONS[action] == "UP":
        x -= 1

    elif ACTIONS[action] == "DOWN":
        x += 1

    elif ACTIONS[action] == "LEFT":
        y -= 1

    elif ACTIONS[action] == "RIGHT":
        y += 1

    # Keep inside grid boundaries
    x = max(0, min(GRID_SIZE - 1, x))
    y = max(0, min(GRID_SIZE - 1, y))

    return (x, y)


# ==========================================
# REWARD FUNCTION
# ==========================================

def compute_reward(dist_before, dist_after, caught):
    """
    Reward shaping
    """

    if caught:
        return 100

    # Encourage moving closer
    return (dist_before - dist_after) * 2 - 1


# ==========================================
# ENVIRONMENT STEP FUNCTION
# ==========================================

def step_env(state, action):
    """
    State format:
    (
        pursuer_x,
        pursuer_y,
        evader_x,
        evader_y
    )
    """

    px, py, ex, ey = state

    pursuer_pos = (px, py)
    evader_pos = (ex, ey)

    # Distance before movement
    dist_before = get_distance(pursuer_pos, evader_pos)

    # Move pursuer
    pursuer_pos = move(pursuer_pos, action)

    # Random evader movement
    evader_action = random.randint(0, len(ACTIONS) - 1)
    evader_pos = move(evader_pos, evader_action)

    # Distance after movement
    dist_after = get_distance(pursuer_pos, evader_pos)

    # Check if caught
    caught = pursuer_pos == evader_pos

    # Compute reward
    reward = compute_reward(dist_before, dist_after, caught)

    # Create next state
    next_state = (
        pursuer_pos[0],
        pursuer_pos[1],
        evader_pos[0],
        evader_pos[1]
    )

    return next_state, reward, caught


# ==========================================
# SARSA AGENT
# ==========================================

class SARSAAgent:

    def __init__(
        self,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):

        self.Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)

        # Exploitation
        return np.argmax(self.Q[state])

    def update(self, state, action, reward,
               next_state, next_action):

        current_q = self.Q[state][action]

        next_q = self.Q[next_state][next_action]

        # SARSA update rule
        new_q = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )

        self.Q[state][action] = new_q

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# ==========================================
# TRAINING
# ==========================================

agent = SARSAAgent()

episodes = 2000

reward_history = deque(maxlen=100)

for episode in range(episodes):

    # Initial state
    state = (0, 0, GRID_SIZE - 1, GRID_SIZE - 1)

    action = agent.choose_action(state)

    total_reward = 0

    done = False

    max_steps = 100
    steps = 0

    while not done and steps < max_steps:

        # Take action
        next_state, reward, caught = step_env(state, action)

        # Choose next action
        next_action = agent.choose_action(next_state)

        # SARSA update
        agent.update(
            state,
            action,
            reward,
            next_state,
            next_action
        )

        # Move to next state/action
        state = next_state
        action = next_action

        total_reward += reward

        # Check terminal condition
        if caught:
            done = True

        steps += 1

    # Decay epsilon
    agent.decay_epsilon()

    reward_history.append(total_reward)

    avg_reward = np.mean(reward_history)

    print(
        f"Episode {episode+1:3d} | "
        f"Reward = {total_reward:6.1f} | "
        f"Avg Reward = {avg_reward:6.1f} | "
        f"Epsilon = {agent.epsilon:.3f}"
    )


# ==========================================
# TEST TRAINED AGENT
# ==========================================

print("\n===== TESTING TRAINED AGENT =====\n")

state = (0, 0, GRID_SIZE - 1, GRID_SIZE - 1)

done = False
steps = 0

while not done and steps < 30:

    action = np.argmax(agent.Q[state])

    next_state, reward, caught = step_env(state, action)

    px, py, ex, ey = next_state

    print(f"Step {steps+1}")
    print(f"Pursuer Position: ({px}, {py})")
    print(f"Evader Position : ({ex}, {ey})")
    print(f"Action Taken    : {ACTIONS[action]}")
    print(f"Reward          : {reward}")
    print("-" * 40)

    state = next_state

    if caught:
        print("Evader Caught!")
        done = True

    steps += 1
