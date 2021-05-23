#!/usr/bin/env python3
"""
Script of class RNNCell that represents a cell of a
simple RNN
"""
import numpy as np


class RNNCell():
    """
    Class methods
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation
        for one time step
        Returns h_next: the next hidden state and y: the output of the cell
        """
        input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(input, self.Wh) + self.bh
        h_next = np.tanh(h_next)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
