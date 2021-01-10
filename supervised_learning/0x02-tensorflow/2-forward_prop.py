#!/usr/bin/env python3
"""
Script that creates he forward propagation
graph for the neural network:r
"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates forward propagation for the neural network
    """
    for i in range(len(layer_sizes)):
        prediction = create_layer(x, layer_sizes[i], activations[i])
    return prediction
