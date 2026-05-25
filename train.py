import numpy as np
from collections import deque

from pursuer import AGENT_CLASS as Pursuer
from evader import AGENT_CLASS as Evader


# Helper constants

GRID_SIZE = 10

EPISODES = 3000
MAX_STEPS = 150


# Environment 

def step_env(state, pursuer_action, evader_action):

    px, py, ex, ey = state

    actions = [
        (-1, 0),   # UP
        (1, 0),    # DOWN
        (0, -1),   # LEFT
        (0, 1)     # RIGHT
    ]

    def move(pos, action):

        x, y = pos

        dx, dy = actions[action]

        x += dx
        y += dy

        x = max(
            0,
            min(
                GRID_SIZE - 1,
                x
            )
        )

        y = max(
            0,
            min(
                GRID_SIZE - 1,
                y
            )
        )

        return (
            x,
            y
        )

    pursuer = move(
        (px, py),
        pursuer_action
    )

    evader = move(
        (ex, ey),
        evader_action
    )

    distance_before = (

        abs(px - ex)

        +

        abs(py - ey)
    )

    distance_after = (

        abs(
            pursuer[0]
            -
            evader[0]
        )

        +

        abs(
            pursuer[1]
            -
            evader[1]
        )
    )

    caught = (

        pursuer
        ==
        evader
    )

    if caught:

        p_reward = 100
        e_reward = -100

    else:

        p_reward = (

            distance_before
            -
            distance_after
        )

        e_reward = -p_reward

    next_state = (

        pursuer[0],
        pursuer[1],

        evader[0],
        evader[1]
    )

    return (

        next_state,
        p_reward,
        e_reward,
        caught
    )


# Training

pursuer = Pursuer()
evader = Evader()

history = deque(
    maxlen=100
)

print(
    "Training started...\n"
)

for episode in range(
    EPISODES
):

    state = (

        0,
        0,

        GRID_SIZE - 1,

        GRID_SIZE - 1
    )

    total_reward = 0

    for step in range(
        MAX_STEPS
    ):

        p_action = (
            pursuer.choose_action(
                state
            )
        )

        e_action = (
            evader.choose_action(
                state
            )
        )

        (
            next_state,
            p_reward,
            e_reward,
            caught

        ) = step_env(

            state,

            p_action,

            e_action
        )

        next_p = (
            pursuer.choose_action(
                next_state
            )
        )

        next_e = (
            evader.choose_action(
                next_state
            )
        )

        pursuer.update(

            state,

            p_action,

            p_reward,

            next_state,

            next_p
        )

        evader.update(

            state,

            e_action,

            e_reward,

            next_state,

            next_e
        )

        state = (
            next_state
        )

        total_reward += (
            p_reward
        )

        if caught:
            break

    pursuer.decay()
    evader.decay()

    history.append(
        total_reward
    )

    print(

        f"Episode {episode+1:4d} | "

        f"Avg Reward: "

        f"{np.mean(history):7.2f} | "

        f"Epsilon: "

        f"{pursuer.epsilon:.3f}"

    )

print(
    "\nTraining complete."
)