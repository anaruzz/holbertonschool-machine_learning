#!/usr/bin/env python3
"""
Script that normalizes a matrix
"""
import numpy as np


def normalize(X, m, s):
    """
    returns the normalized X matrix
    """
    return (X - m) / s
