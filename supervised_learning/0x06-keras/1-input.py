#!/usr/bin/env python3
"""
Script that builds a neural network with the Keras library
"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Returns the built keras model
    """
    input = k.Input(shape=(nx,))
    output = input
    regularizer = k.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = (k.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=k.regularizers.l2(lambtha)))(input)
        else:
            output = (k.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=k.regularizers.l2(lambtha)))(output)
        if i != len(layers) - 1:
            output = (k.layers.Dropout(1 - keep_prob))(output)
    model = k.Model(input, output)
    return model
