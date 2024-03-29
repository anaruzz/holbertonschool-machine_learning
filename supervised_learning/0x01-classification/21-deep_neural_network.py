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

    def sigmoid(self, X=None, w=None, b=None, z=None):
        """
        sigmoid activation fuction
        """
        if z is None:
            z = np.dot(w, X)
            z = np.add(z, b)
        return (1 / (1 + np.exp(-z)))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        i = 0
        n = self.__L
        self.__cache['A0'] = X
        for i in range(1, n+1):
            self.__cache['A'+str(i)] = self.sigmoid(self.__cache['A'+str(i-1)],
                                                    self.weights['W'+str(i)],
                                                    self.weights['b'+str(i)])
        return self.__cache['A'+str(self.__L)], self.__cache

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
        self.forward_prop(X)
        A = self.cache['A'+str(self.__L)]
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return (A, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of Gradient descent on the NN
        """
        m = Y.shape[1]
        d_z = self.cache['A' + str(self.L)] - Y
        # reversed for loop for backword prop
        for i in range(self.L, 0, -1):
            # previous layer cache
            prev = self.cache['A' + str(i-1)]
            # weight derivative
            d_w = np.dot(d_z, prev.T) / m
            # bias derivative
            d_b = np.mean(d_z, axis=1, keepdims=True)
            # sigmoid derivative
            d_s = prev * (1 - prev)
            # z derivative
            d_z = np.dot(self.weights["W" + str(i)].T, d_z) * d_s
            # update weights and bias
            self.__weights["W" + str(i)] -= alpha * d_w
            self.__weights["b" + str(i)] -= alpha * d_b
