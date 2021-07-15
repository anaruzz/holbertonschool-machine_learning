#!/usr/bin/python3
"""
Function that computes to policy with a weight of a matrix
"""
import numpy as np


def policy(matrix, weight):
    """
    computes policy
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z)
    policy = exp / np.sum(exp)
    return policy

def policy_gradient(state, weight):
    """
    Returns the action and the gradient
    """
    P = policy(state, weight)
    state = state.reshape(1, -1)
    total = 0
    for i, chance in enumerate(P[0]):
        total += chance
        action = i
        if np.random.random() < total:
            break
    gradient = state.T - (P * state.T)
    return action, gradient
