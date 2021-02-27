#!/usr/bin/env python3
"""
Script  that updates the weights of a neural network
with Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights
    """
    m = Y.shape[1]
    dzi = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        # weight derivative
        dwi = np.matmul(dzi, A_prev.T) / m
        # bias derivative
        dbi = np.sum(dzi, axis=1, keepdims=True) / m
        # activation function derivative
        dgi = 1 - (A_prev ** 2)
        # output
        dAi = np.matmul(weights['W'+str(i)].T, dzi)
        if i > 1:
            dAi = dAi * cache["D" + str(i - 1)]
            dAi = dAi / keep_prob
        # Z derivative
        dzi = dAi * dgi
        # update weights and bias
        weights["W" + str(i)] -= alpha * dwi
        weights["b" + str(i)] -= alpha * dbi
