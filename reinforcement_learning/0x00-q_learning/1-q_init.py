#!/usr/bin/env python3
"""
Script that initializes the Q-table
"""
import numpy as np


def q_init(env):
    """
    Returns: the Q-table
    """
    actions_number = env.action_space.n
    states_number = env.observation_space.n
    table = np.zeros((states_number, actions_number))
    return table
