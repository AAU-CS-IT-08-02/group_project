import numpy as np
import random

GRID_SIZE = 10

ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]

NUM_ACTIONS = len(ACTIONS)


def get_distance(a, b):

    return (
        abs(a[0] - b[0])
        +
        abs(a[1] - b[1])
    )


def move(position, action):

    x, y = position

    if ACTIONS[action] == "UP":
        x -= 1

    elif ACTIONS[action] == "DOWN":
        x += 1

    elif ACTIONS[action] == "LEFT":
        y -= 1

    elif ACTIONS[action] == "RIGHT":
        y += 1

    x = max(0, min(GRID_SIZE - 1, x))
    y = max(0, min(GRID_SIZE - 1, y))

    return (x, y)


def step_env(
    state,
    pursuer_action,
    evader_action
):

    px, py, ex, ey = state

    pursuer = (
        px,
        py
    )

    evader = (
        ex,
        ey
    )

    dist_before = get_distance(
        pursuer,
        evader
    )

    pursuer = move(
        pursuer,
        pursuer_action
    )

    evader = move(
        evader,
        evader_action
    )

    dist_after = get_distance(
        pursuer,
        evader
    )

    caught = (
        pursuer ==
        evader
    )

    pursuer_reward = (
        100
        if caught
        else (
            (
                dist_before
                -
                dist_after
            ) * 2
        ) - 1
    )

    evader_reward = (
        -100
        if caught
        else (
            (
                dist_after
                -
                dist_before
            ) * 2
        ) + 1
    )

    next_state = (

        pursuer[0],
        pursuer[1],

        evader[0],
        evader[1]
    )

    return (
        next_state,
        pursuer_reward,
        evader_reward,
        caught
    )