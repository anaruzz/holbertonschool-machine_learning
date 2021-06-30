#!/usr/bin/env python3
"""
Script  that uses epsilon-greedy to determine
the next action
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Returns: the next action indexf
    """
    n = np.random.uniform(0, 1)
    if n > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(Q.shape[1])
