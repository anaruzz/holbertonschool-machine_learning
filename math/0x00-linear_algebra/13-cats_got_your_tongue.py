#!/usr/bin/env python3
"""
Script that concatenates two arrays
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    function that concatenates two arrays
    """
    new = np.concatenate((mat1, mat2), axis)
    return new
