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
    sv = s.sum() * var
    n = 0
    t = s[0]
    while (t < sv):
        n += 1
        t += s[n]

    return v[:n+1].T
