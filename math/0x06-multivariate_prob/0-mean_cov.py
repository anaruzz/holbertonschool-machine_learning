#!/usr/bin/env python3
"""
A function that calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    Returns mean of the data set
    and covariance matrix of the data set
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = X.mean(axis=0)
    xi = X - mean
    cov = np.matmul(xi.T, xi) / (n - 1)
    return mean, cov
