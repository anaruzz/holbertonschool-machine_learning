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
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            m = sum(data) / len(data)
            s = 0
            for x in data:
                s += (x - m) ** 2
            n = round((m ** 2) / (m - (s / len(data))))
            self.n = n
            self.p = m / n

    def fact(self, n):
        """
        calculate factorial
        """
        f = 1
        for i in range(1, n+1):
            f = f * i
        return f

    def pmf(self, k):
        """
        calculate pmf of binomial distribution
        """
        if k < 0:
            return 0
        try:
            a = self.n - k
            nk = self.fact(self.n) / (self.fact(k) * (self.fact(a)))
            return nk * (self.p**k * ((1 - self.p) ** (self.n - k)))
        except Exception:
            return 0
