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
        self.__W = np.random.randn(1, self.nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def sigmoid(self, z):
        """
        sigmoid activation function
        """
        return (1 / (1 + np.exp(-z)))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        z = np.dot(self.__W, X) + self.b
        self.__A = self.sigmoid(z)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model with logistic regression
        """
        m = A.shape[1]
        error = (-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        return 1 / m * np.sum(error)

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        self.__A = np.where(A >= 0.5, 1, 0)
        return (self.__A, cost)
