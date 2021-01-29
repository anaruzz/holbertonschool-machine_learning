#!/usr/bin/env python3
"""
Script that updates a variable in place using
inverse time decay in numpy
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    returns the updated Z matrix
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z - mean, axis=0)
    Z_norm = (Z - mean) / ((var + epsilon) ** 0.5)
    return gamma * Z_norm + beta
