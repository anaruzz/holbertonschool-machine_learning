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

    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster, initialized evenly
    m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster, initialized with K-means
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster, initialized as identity matrices

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
