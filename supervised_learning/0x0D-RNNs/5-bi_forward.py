#!/usr/bin/env python3
"""
Script that represents a bidirectional cell of an RNN
"""
import numpy as np


class BidirectionalCell():
    """
    Class that represents an bidirectional unit
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the forward direction for one time step
        Returns h_next: the next hidden state
        """
        # Concat h_prev and x_t to match Wh dimensions
        x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(x, self.Whf) + self.bhf)

        return h_next
