#!/usr/bin/env python3
"""
Script that makes a prediciton using a neural network
"""
import tensorflow.keras as k


def predict(network, data, verbose=False):
    """
    Returns the prediciton for the data
    """
    test = network.predict(x=data, verbose=verbose)
    return test
