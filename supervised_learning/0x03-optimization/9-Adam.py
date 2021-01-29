#!/usr/bin/env python3
"""
Script that updates a variable in place using
the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    returns the updated variable, the first
    new moment and the second new moment
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    vc = v / (1 - (beta1 ** t))
    sc = s / (1 - (beta2 ** t))
    var -= alpha * (vc / ((sc ** 0.5) + epsilon))
    return var, v, s
