#!/usr/bin/env python3
"""
Function that initializes variables for a
Gaussian Mixture Model
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Returns: pi, m, S, or None, None, None on failure

    pi is a numpy.ndarray of shape (k,)
    m is a numpy.ndarray of shape (k, d)
    S is a numpy.ndarray of shape (k, d, d)
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    _, d = X.shape
    m, _ = kmeans(X, k)
    if (m is None):
        return None, None, None
    p = (1 / k) * np.ones(k)
    s = np.ones((k, d, d)) * np.eye(d)
    return p, m, s
