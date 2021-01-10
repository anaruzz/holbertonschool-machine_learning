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
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, self.nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, z):
        """
        sigmoid activation fuction
        """
        return (1 / (1 + np.exp(-z)))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model with logistic regression
        """
        m = Y.shape[1]
        error = (-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        return 1 / m * np.sum(error)

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        _, self.__A2 = self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        self.__A2 = np.where(self.__A2 >= 0.5, 1, 0)
        return (self.__A2, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent
        """
        m = Y.shape[1]
        d_z2 = A2 - Y
        d_w2 = np.matmul(d_z2, A1.T) * 1 / m
        d_b2 = np.sum(d_z2, axis=1, keepdims=True) * 1 / m
        d_A1 = A1 * (1 - A1)
        d_z1 = np.matmul(self.__W2.T, d_z2) * d_A1
        d_w1 = np.matmul(d_z1, X.T) * 1 / m
        d_b1 = np.sum(d_z1, axis=1, keepdims=True) * 1 / m
        self.__W1 = self.__W1 - d_w1 * alpha
        self.__b1 = self.__b1 - d_b1 * alpha
        self.__W2 = self.__W2 - d_w2 * alpha
        self.__b2 = self.__b2 - d_b2 * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
