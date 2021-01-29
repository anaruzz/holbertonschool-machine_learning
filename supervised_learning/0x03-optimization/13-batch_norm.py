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
    mean = numpy.mean(Z)
