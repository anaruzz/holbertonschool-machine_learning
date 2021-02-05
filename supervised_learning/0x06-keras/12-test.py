#!/usr/bin/env python3
"""
Script that tests a neural network
"""
import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    """
    Returns the loss and accuracy of the model
    with the testing data, respectively
    """
    test = network.evaluate(x=data, y=labels, verbose=verbose)
    return test
