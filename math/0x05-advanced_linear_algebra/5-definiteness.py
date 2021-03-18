#!/usr/bin/env python3
"""
Script that calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if not np.array_equal(matrix.T, matrix):
        return None
    if len(matrix.shape) == 1:
        return None

    e = np.linalg.eigvals(matrix)
    if np.all(e > 0):
        return ("Positive definite")
    elif np.all(e < 0):
        return ("Negative definite")
    elif np.all(e >= 0):
        return ("positive semi definite")
    elif np.all(e <= 0):
        return ("Negative definite")
    else:
        return ("Indefinite")
