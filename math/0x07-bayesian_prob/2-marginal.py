#!/usr/bin/env python3
"""
A function that calculates the marginal probability
of obtaining the data
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Returns a 1 D array containing the intersection of
    obtaining x and n with each probability in P
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any([v < 0 or v > 1 for v in Pr]):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.any([v < 0 or v > 1 for v in P]):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    f = np.math.factorial
    combination = f(n) / (f(x) * f(n - x))
    prob = np.power(P, x) * np.power(1 - P, n - x)
    intersection = combination * prob * Pr
    return np.sum(intersection)
