#!/usr/bin/env python3
"""
Script that updates a variable using the
gradient descent with momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    returns the updated variable and the new moment
    """
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
