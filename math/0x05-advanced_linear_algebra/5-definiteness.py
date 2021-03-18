#!/usr/bin/env python3
"""
Script that calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Returnes the defini of a matrix
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) == 1:
        return None

    if not np.array_equal(matrix.T, matrix):
        return None

    e = np.linalg.eigvals(matrix)[0]
    if np.all(e >= 0):
        if np.any(e == 0):
            return 'Positive semi-definite'
        return 'Positive definite'
    if np.all(e <= 0):
        if np.any(e == 0):
            return 'Negative semi-definite'
        return 'Negative definite'
    return 'Indefinite'
