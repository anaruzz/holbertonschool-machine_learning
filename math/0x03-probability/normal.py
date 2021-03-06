#!/usr/bin/env python3
""" Script that represents normal distribution"""


class Normal():
    """
    class Normal to represent normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class Constructor
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            std = 0
            for x in data:
                std += (x - self.mean) ** 2
            self.stddev = (std / len(data)) ** 0.5

    def z_score(self, x):
        """
        z score of a given x value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        x value of a given z score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        calculate pdf for a given x value
        """
        a = -(x - self.mean) ** 2
        b = 2 * self.stddev ** 2
        c = self.stddev * ((2 * 3.1415926536) ** 0.5)
        return (2.7182818285 ** (a / b)) / c

    def er(self, x):
        """calculate er function"""
        return (x - ((x ** 3) / 3) + ((x ** 5) / 10) - ((x ** 7) / 42) +
                ((x ** 9) / 216)) * (2 / 3.1415926536 ** (0.5))

    def cdf(self, x):
        """cumilative density function for normal distribution"""
        a = (1 + self.er((x - self.mean) / (self.stddev * (2 ** (0.5)))))
        return 0.5 * a
