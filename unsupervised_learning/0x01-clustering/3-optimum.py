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

    if type(kmax) is not int or kmax < 1:
        return None, None

    if kmax - kmin < 2:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    d_vars = [0]
    results = [kmeans(X, kmin, iterations)]
    first_var = variance(X, results[0][0])
    while kmin < kmax:
        kmin += 1
        centroid, clss = kmeans(X, kmin, iterations)
        var = variance(X, centroid)
        results.append((centroid, clss))
        d_vars.append(first_var - var)

    return results, d_vars
