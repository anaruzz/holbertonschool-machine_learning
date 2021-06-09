#!/usr/bin/env python3
"""
Function that  tests for the optimum
number of clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Returns: results, d_vars, or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(kmin) is not int or kmin < 1:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax < 1:
        return None, None

    if kmax <= kmin:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    results = []
    d_vars = []
    for i in range(kmin, kmax + 1):
        centroid, clss = kmeans(X, i, iterations)
        results.append((centroid, clss))
        var = variance(X, centroid)
        d_vars.append(var)

    for i in range(len(d_vars)):
        d_vars[i] = variance(X, results[0][0] - d_vars[i])
    return results, d_vars
