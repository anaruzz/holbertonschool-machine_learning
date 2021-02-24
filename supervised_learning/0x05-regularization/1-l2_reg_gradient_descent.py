#!/usr/bin/env python3
"""
Script that updates the weights and biases of a neural
network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases
    """
    m = len(Y[0])
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        # weight derivative
        dw = np.matmul(dz, A_prev.T) / m
        # bias derivative
        db = np.mean(dz, axis=1, keepdims=True)
        # tanh derivative
        dg = 1 - np.square(A_prev)
        # z derivative
        dz = np.matmul(weights['W' + str(i)].T, dz) * dg
        # L2 regularization
        L2 = 1 - lambtha * alpha / m
        # update weights and bias
        weights['W' + str(i)] = L2 * weights['W' + str(i)] - alpha * dw
        weights["b" + str(i)] -= alpha * db
