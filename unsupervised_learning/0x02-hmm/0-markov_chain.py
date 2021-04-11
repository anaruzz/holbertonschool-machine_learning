#!/usr/bin/env python3
"""
 that determines the probability of a markov
 chain being in a particular state after a
 specified number of iterations
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Returns a np.ndarray of shape (1, n) representing
    the probability of being in a specific state after t iterations
    or None on failure
    """
