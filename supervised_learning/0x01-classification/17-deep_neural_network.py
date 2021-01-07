#!/usr/bin/env python3
"""
Script that defines a deep neural network
performing binary classification:
"""
import numpy as np


class DeepNeuralNetwork():
    """
    deep neural network
    """
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if not all(map(lambda b: b > 0 and type(b) is int, layers)):
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                m = self.nx
            else:
                m = layers[i - 1]
            nW = 'W' + str(i + 1)
            n = layers[i]
            self.__weights[nW] = np.random.normal(size=(n, m)) * np.sqrt(2 / m)
            nb = 'b' + str(i + 1)
            self.__weights[nb] = np.zeros((n, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
