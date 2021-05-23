#!/usr/bin/env python3
"""
Script that performs forward propagation
for a simple RNNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Returns H: containing the hidden states
            Y: containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    n = rnn_cell.Wy.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, n))

    for i in range(t):
        if i == 0:
            H[i] = h_0
        H[i + 1], Y[i] = rnn_cell.forward(H[i], X[i])

    return H, Y
