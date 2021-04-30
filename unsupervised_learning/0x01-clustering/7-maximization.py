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

    # initialize with zeros
    pi = np.zeros((g.shape[0],))
    m = np.zeros((g.shape[0], X.shape[1]))
    S = np.zeros((g.shape[0], X.shape[1], X.shape[1]))

    for i in range(g.shape[0]):
        gsum = np.sum(g[i], axis=0)
        pi[i] = gsum / X.shape[0]
        m[i] = np.sum(np.matmul(g[i][np.newaxis, ...], X), axis=0) / gsum
        S[i] = np.matmul(g[i][np.newaxis, ...] * (X - m[i]).T, (X - m[i])) / gsum


    return pi, m, S
