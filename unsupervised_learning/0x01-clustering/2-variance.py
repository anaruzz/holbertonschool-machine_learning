#!/usr/bin/env python3
"""
Function that calculates the total intra-cluster
variance for a data set
"""
import numpy as np


def variance(X, C):
    """
    Returns the total variance or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    var = np.sum((X - C[:, np.newaxis]) ** 2, axis=2)
    d = np.sqrt(var)
    min = np.min(d, axis=0)
    var = np.sum(min ** 2)
    return np.sum(var)
