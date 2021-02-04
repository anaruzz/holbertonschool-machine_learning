#!/usr/bin/env python3
"""
Script that builds a neural network with the Keras library
"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Returns the built keras model
    """
    model = k.Sequential()
    regularizer = k.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            d = nx
        elif i != len(layers) - 1:
            model.add(k.layers.Dropout(1-keep_prob))
        else:
            d = layers[i - 1]
        model.add(k.layers.Dense(layers[i], input_dim=d,
                                 activation=activations[i],
                                 kernel_regularizer=regularizer
                                 ))
    return model
