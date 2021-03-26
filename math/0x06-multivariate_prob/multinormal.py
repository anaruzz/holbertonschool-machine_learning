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
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        xi = data - self.mean
        self.cov = np.matmul(xi, xi.T) / (n - 1)

    def pdf(self, x):
        """
        calculate pdf at a data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        xi = x - self.mean
        sqrt = np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        inv = np.linalg.solve(self.cov, xi)
        exp = np.exp(-(inv.T.dot(xi)) / 2)
        result = 1 / sqrt * exp
        return result[0][0]
