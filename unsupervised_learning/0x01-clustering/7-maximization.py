#!/usr/bin/env python3
"""
Function that calculates the maximization step in the EM
algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    returns: pi, m, S, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    if not np.all(np.isclose(g.sum(axis=0), 1)):
        return None, None, None
    # initialize
    gsum = g.sum(axis=1)
    pi = gsum / X.shape[0]
    m = np.matmul(g, X) / gsum[:, np.newaxis]
    S = np.zeros((g.shape[0], X.shape[1], X.shape[1]))

    for i in range(g.shape[0]):
        diff = X - m[i]
        S[i] = np.matmul((diff * g[i, :, np.newaxis])
                         .T, diff) / gsum[i]

    return pi, m, S
