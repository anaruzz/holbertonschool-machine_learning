#!/usr/bin/env python3
"""
Script that shuffles the data points
in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    returns shuffled X and Y matrices
    """
    m = X.shape[0]
    p = np.random.permutation(m)
    return X[p], Y[p]
