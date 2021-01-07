#!/usr/bin/env python3
"""
Script that converts a numeric label vector
into one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label
    vector into a one-hot matrix
    """
    try:
        # b = np.zeros((Y.size, classes))
        # b[np.arange(Y.size), Y] = 1
        if Y is None or type(Y) is not np.ndarray\
       or type(classes) is not int:
            return None
        return np.eye(classes)[Y].T
    except Exception:
        return None
