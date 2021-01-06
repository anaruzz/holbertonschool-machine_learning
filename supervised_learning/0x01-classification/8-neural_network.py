#!/usr/bin/env python3
"""
that defines a neural network
with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork():
    """
    Class thet defines a neural network
    with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
