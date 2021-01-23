#!/usr/bin/env python3
"""
Script that normalizes a matrix
"""
import numpy as np


def normalize(X, m, s):
    return (X - m) / s
