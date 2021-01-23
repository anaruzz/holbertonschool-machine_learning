#!/usr/bin/env python3
"""
Script that calculates the moving average
"""
import numpy as np


def moving_average(data, beta):
    """
    returns a list containing
    the moving averages of data
    """
    avg = []
    v = 0
    for i in range(len(data)):
        v = (v * beta + (1 - beta) * data[i])
        avg.append(v / (1 - beta ** (i + 1)))
    return avg
