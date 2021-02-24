#!/usr/bin/env python3
"""
Script that conducts forward propagation using Dropout:
"""
import numpy as np


def tanh(z):
    """
    tanh activation function
    """
    return np.tanh(z)


def softmax(z):
    """
    softmax activation function
    """
    return np.exp(z) / (np.sum(np.exp(z), axis=0))


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Returns the output of the new layer
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L+1):
        A = cache['A' + str(i-1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, A) + b
        if i == L:
            cache['A' + str(i)] = softmax(Z)
        else:
            A = tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A / keep_prob
    return cache
