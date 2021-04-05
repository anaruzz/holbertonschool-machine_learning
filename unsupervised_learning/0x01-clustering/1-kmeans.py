#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
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


def kmeans(X, k, iterations=1000):
    """
    Returns: C, clss, or None, None on failure

    C is a numpy.ndarray of shape (k, d) containing the
    centroid means for each cluster
    clss is a numpy.ndarray of shape (n,) containing
    the index of the cluster in C that each data point belongs to

    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations < 1:
            return None, None

    c = initialize(X, k)
    if (c.any() is None):
        return None, None

    for i in range(iterations):
        old_c = np.copy(c)
        distances = X - c[:, np.newaxis]
        distances = np.sqrt((distances ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)
        for j in range(k):
            index = np.argwhere(clss == j)
            if index.shape[0] == 0:
                c[j] = initialize(X, 1)
            else:
                c[j] = np.mean(X[index], axis=0)
        if np.all(old_c == c):
            break
    return c, clss
