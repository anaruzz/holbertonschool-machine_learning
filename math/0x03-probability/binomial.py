#!/usr/bin/env python3
"""
Script that represents binomial distribution
"""


class Binomial():
    """
    class binomial
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        class constructor
        """
        if data is None:
            if n < 0:
                raise ValueError('n must be a positive number')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must bee a list')
            if len(data) < 2:
                raise ValueError('data must contain values')
            m = sum(data) / len(data)
            s = 0
            for x in data:
                s += (x - m) ** 2
            n = round((m ** 2) / (m - (s / len(data))))
            self.n = n
            self.p = m / n
