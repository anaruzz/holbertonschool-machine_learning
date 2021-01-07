#!/usr/bin/env python3
"""
Script that converts a one-hot matrix
to a numeric label vector
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix
    to a numeric label vecto
    """
    if type(one_hot) != np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
