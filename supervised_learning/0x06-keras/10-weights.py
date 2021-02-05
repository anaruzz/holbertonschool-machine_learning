#!/usr/bin/env python3
"""
Script that saves and loads an entire model's weigths
"""
import tensorflow.keras as k


def save_weights(network, filename, save_format='h5'):
    """
    saves model weight and returns None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    loads model's weigth and returns None
    """
    network.load_weights(filename)
    return None
