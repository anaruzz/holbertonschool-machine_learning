#!/usr/bin/env python3
"""
Function that determines the probability of a markov
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
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n = P.shape[1]
    if type(s) is not np.ndarray or s.shape != (1, n):
        return None
    if np.any(np.sum(P, axis=1)) != 1:
        return None

    # return np.matmul(s, np.linalg.matrix_power(P, t))
    while (t > 0):
        s = np.matmul(s, P)
        t -= 1
    return s
