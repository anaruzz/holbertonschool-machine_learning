#!/usr/bin/env python3
"""
Script that has the trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Returns: the total rewards for the episode
    """
    state = env.reset()
    done = False
    env.render()
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        env.render()
        if done is True:
            break
    return reward
