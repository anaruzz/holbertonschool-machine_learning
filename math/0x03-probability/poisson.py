#!/usr/bin/env python3
""" Script that represents Poisson distribution"""


class Poisson():
    """
    class Poisson to represent poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        class constructor
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """
        Calculate pmf of poisson distribution
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        fct = 1
        for i in range(1, k+1):
            fct = fct * i
        return (self.lambtha ** k * 2.7182818285 ** (- self.lambtha)) / fct
