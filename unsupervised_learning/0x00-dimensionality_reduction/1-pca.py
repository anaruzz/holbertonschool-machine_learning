#!/usr/bin/env python3
"""
A function that perfors PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    Returns A numpy.ndarray of shape(n, ndim)
    containing the transformed version of X
    """
    X = X - np.mean(X, axis=0)
    _, s, V = np.linalg.svd(X)
    c = np.cumsum(s)
    W = V[:ndim].T
    return np.matmul(X, W)
