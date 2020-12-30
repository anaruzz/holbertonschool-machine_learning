#!/usr/bin/env python3
"""
Script that defines a single neuron performing
 a binary classification
"""
import numpy as np


class Neuron():
    """
    class that defines a single neuron in a neural network
    """
    def __init__(self, nx):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.W = np.random.randn(1, self.nx)
        self.b = 0
        self.A = 0
