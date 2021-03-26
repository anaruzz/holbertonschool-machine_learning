#!/usr/bin/env python3
"""
A function that calculates the correlation of a data set
"""
import numpy as np


def correlation(C):
    """
    Returns numpy array containing the correlation
    matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2  or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    var = np.sqrt(np.diag(C))
    out = np.outer(var, var)
    cor = C / out

    return cor
