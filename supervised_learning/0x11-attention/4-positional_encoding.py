#!/usr/bin/env python3
"""
A function that calculates the positional encoding for a transformer
"""
import numpy as np


def get_angles(position, i, d_model):
    """
    Returns value used for output vector
    """
    rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return position * rates


def positional_encoding(max_seq_len, dm):
    """
    calculates the positional encoding for a transformer
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
    the positional encoding vectors
    """

    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :], dm)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
