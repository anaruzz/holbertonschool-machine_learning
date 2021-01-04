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
        self.__A = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        A = np.where(self.__A >= 0.5, 1, 0)
        return (A, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        # Update bias
        self.__b = self.__b - (alpha * np.mean(A - Y))
        # Update weight
        m = Y.shape[1]
        weight_deriv = np.matmul(X, (A-Y).T) / m
        self.__W -= alpha * weight_deriv.T

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be positive')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
