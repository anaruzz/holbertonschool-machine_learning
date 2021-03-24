#!/usr/bin/env python3
"""
A function that perfors PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Returns the weights matrix, W that maintains
    var fraction of X's original variance
    """
    _, s, v = np.linalg.svd(X)
    n = 0
    sv = s.sum() * var
    t = s[0]
    while (t < sv):
        t += s[n]
        n += 1
    n += 1
    return v[:n+1].T
