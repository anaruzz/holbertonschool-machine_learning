#!/usr/bin/env python3
""" Script that represents Exponontial distribution"""


class Exponential():
    """
    class Exponontial to represent Exponontial distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Class Constructor
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """
        Probability density function of an exponential distribution
        """
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """cumulative distribution function"""
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-x * self.lambtha)
