#!/usr/bin/env python3
"""
Script hat calculates the cost of a neural network
with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Returns the cost of the network accounting for
    L2 regularization
    """
    ws = 0
    for i in range(L):
        weight = weights['W' + str(i+1)]
        ws += np.linalg.norm(weight, "fro")
    L2 = (lambtha * ws) / (2 * m)
    return cost + L2
