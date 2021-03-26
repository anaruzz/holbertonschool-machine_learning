#!/usr/bin/env python3
"""
A function that calculates the likelihood of obtaining
this data given various hypothetical probabilities of
developing severe side effects
"""
import numpy as np


def likelihood(x, n, P):
    """
    Returns a 1 D array containing the likelihood of
    obtaining the data, x and n for each probability
    in P
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any([v < 0 or v > 1 for v in P]):
        raise ValueError("All values in P must be in the range [0, 1]")

    f = np.math.factorial
    combination = f(n) / (f(x) * f (n - x))
    prob = np.power(P, x) * np.power(1 - P, n - x)
    return combination * prob 
