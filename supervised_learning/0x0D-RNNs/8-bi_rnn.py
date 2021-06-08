#!/usr/bin/env python3
"""
Script that performs forward propagation
for a bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Returns: H, Y
    H containing all of the concatenated hidden states
    Y containing all of the outputs
    """
    t, m, _ = X.shape
    s0 = h_0.shape[1]
    s1 = h_t.shape[1]

    Hf = np.zeros((t, m, s0))
    Hb = np.zeros((t, m, s1))
    h_prev = h_0
    for i in range(t):
        Hf[i] = bi_cell.forward(h_prev, X[i])
        h_0 = Hf[i]

    for i in reversed(range(t)):
        Hb[i] = bi_cell.backward(Hb[i], X[i])

    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)

    return H, Y
