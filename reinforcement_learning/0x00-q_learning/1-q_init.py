#!/usr/bin/env python3
"""
Script that initializes the Q-table
"""
import numpy as np


def q_init(env):
    """
    Returns: the Q-table
    """
    action_space_size = env.action_space.n
    observation_size = env.observation_space.n
    table = np.zeros((observation_size, action_space_size))
    return table
