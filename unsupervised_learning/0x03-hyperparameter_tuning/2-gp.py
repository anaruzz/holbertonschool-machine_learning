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

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a
        Gaussian Process
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu = (K_s.T.dot(K_inv).dot(self.Y)).flatten()

        cov_s = (K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, np.diag(cov_s)

    def update(self, X_new, Y_new):
        """
        updates the sample point and the
        sample function value
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
