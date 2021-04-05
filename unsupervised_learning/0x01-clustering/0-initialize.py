#!/usr/bin/env python3
"""
Function that initializes cluster centroids for K-means
"""
import numpy as np


def initialize(X, k):
    """
    Returns a numpy.ndarray of shape (k, d) containing
    the initialized centroids for each cluster,
    or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    _, d = X.shape
    min = X.min(axis=0)
    max = X.max(axis=0)
    values = np.random.uniform(min, max, (k, d))

    return values
