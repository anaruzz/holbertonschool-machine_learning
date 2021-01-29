#!/usr/bin/env python3
"""
Script that updates a variable using the RMSProp
optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    returns the updated variable and the new moment
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var -= alpha * (grad / (np.sqrt(s) + epsilon))
    return var, s
