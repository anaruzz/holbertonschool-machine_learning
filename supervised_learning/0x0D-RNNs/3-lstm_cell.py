#!/usr/bin/env python3
"""
Script that represents an LSTM unit
"""
import numpy as np


class LSTMCell():
    """
    Class that represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one step
        Returns h_next: the next hidden state
                y: the output of the cell
                c_next: the next cell state
        """
        # Concat h_prev and x_t to match Wh dimensions
        x = np.concatenate((h_prev, x_t), axis=1)

        u = np.dot(x, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))

        forget = np.dot(x, self.Wf) + self.bf
        forget = 1 / (1 + np.exp(-forget))

        o = np.dot(x, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))

        c = np.tanh(np.dot(x, self.Wc) + self.bc)
        c_t = u * c + forget * c_prev
        h_t = o * np.tanh(c_t)

        y = np.dot(h_t, self.Wy) + self.by

        y = (np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True))

        return h_t, c_t, y
