#!/usr/bin/env python3
"""
Function  that determines the steady state probabilities
of a regular markov chain
"""
import numpy as np


def regular(P):
    """
    Returns a np.ndarray of shape (1, n) containing the
    steady state probabilities of a regular markov chain
    or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    n, m = P.shape
    if n != m:
        return None

    if np.any(P <= 0):
        return None

    evals, evecs = np.linalg.eig(P.T)
    evecs = evecs[:, np.isclose(evals, 1)]
    steady = (evecs / evecs.sum().T)
    return steady
