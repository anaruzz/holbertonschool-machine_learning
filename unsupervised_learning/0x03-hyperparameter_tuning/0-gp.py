#!/usr/bin/env python3
"""
Script that represents a noiseless 1D Gaussian Process
"""
import numpy as np

class GaussianProcess():
    """
    A class that represents a gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        constructor for class
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Returns the covariance matrix
        """
        first = np.sum(X1 ** 2, 1).reshape(-1, 1)
        second = np.sum(X2 ** 2, 1)
        dist_sq = first + second - 2 * np.dot(X1, X2.T)
        kernel = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist_sq)
        return kernel
