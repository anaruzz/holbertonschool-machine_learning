#!/usr/bin/env python3
"""
Function  that determines if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    Returns True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    n, m = P.shape
    if n != m:
        return None

    if np.any(P < 0):
        return None

    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    for i in range(n):
        if P[i, i] == 1:
            return True
    return False
