import numpy as np
from collections import deque

from env import step_env, GRID_SIZE
from pursuer import Pursuer
from evader import Evader


EPISODES = 3000
MAX_STEPS = 150


pursuer = Pursuer()
evader = Evader()

history = deque(maxlen=100)

print("Training started...\n")

for episode in range(EPISODES):

    state = (
        0, 0,
        GRID_SIZE - 1,
        GRID_SIZE - 1
    )

    p_action = pursuer.choose_action(state)
    e_action = evader.choose_action(state)

    total_reward = 0

    for step in range(MAX_STEPS):

        next_state, p_reward, e_reward, caught = step_env(
            state,
            p_action,
            e_action
        )

        next_p_action = pursuer.choose_action(next_state)
        next_e_action = evader.choose_action(next_state)

        pursuer.update(
            state,
            p_action,
            p_reward,
            next_state,
            next_p_action
        )

        evader.update(
            state,
            e_action,
            e_reward,
            next_state,
            next_e_action
        )

        state = next_state
        p_action = next_p_action
        e_action = next_e_action

        total_reward += p_reward

        if caught:
            break

    pursuer.decay()
    evader.decay()

    history.append(total_reward)

    print(
        f"Episode {episode+1:4d} | "
        f"Avg Reward: {np.mean(history):7.2f} | "
        f"Epsilon: {pursuer.epsilon:.3f}"
    )

print("\nTraining complete.")