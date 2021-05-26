#!/usr/bin/env python3
"""
Script that performs forward propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Returns: H, Y
    H containing all of the hidden states
    Y containing all of the outputs
    """
    t = X.shape[0]
    h_prev = h_0
    H = np.array(([h_prev]))
    H = np.repeat(H, X.shape[0] + 1, axis=0)

    for i in range(t):
        for layer, cell in enumerate(rnn_cells):
            if layer == 0:
                p = X[i]
            else:
                p = h_prev

            h_prev, y = cell.forward(H[i, layer], p)
            H[i + 1, layer] = h_prev

            if (i != 0):
                Y[i] = y
            else:
                Y = np.array([y])
                Y = np.repeat(Y, t, axis=0)

    return H, Y
