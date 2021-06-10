#!/usr/bin/env python3
"""
hat performs Bayesian optimization on a noiseless 1D Gaussian process
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    A class that represents a Bayesian Optimization
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        constructor for class
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Returns: X_next, EI
        """
        m_sample, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            sample = np.min(self.gp.Y)
            im = sample - m_sample - self.xsi
        else:
            sample = np.max(self.gp.Y)
            im = m_sample - sample - self.xsi

        with np.errstate(divide='ignore'):
            Z = im / sigma
            EI = (im * norm.cdf(Z)) + (sigma * norm.pdf(Z))
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
