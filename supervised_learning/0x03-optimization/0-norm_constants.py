#!/usr/bin/env python3
"""
Script that calculates the normalization
(standardization) constants of a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    Return mean and standard deviation
    """
    return X.mean(axis=0), X.std(axis=0)
