#!/usr/bin/env python3
"""
A class that represents a Multivariate Normal distribution
"""
import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov


class MultiNormal():
    """
    Class that represents a Multivariate
    normal distribution
    """
    def __init__(self, data):
        """
        class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) < 2:
            raise TypeError("data must be a numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        xi = data - self.mean
        self.cov = np.matmul(xi, xi.T) / (n - 1)
