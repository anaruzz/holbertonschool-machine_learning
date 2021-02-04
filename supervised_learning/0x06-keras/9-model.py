#!/usr/bin/env python3
"""
Script that saves and loads an entire model
"""
import tensorflow.keras as k


def save_model(network, filename):
    """
    saves model and returns None
    """
    network.save(filename)


def load_model(filename):
    """
    loads model and return model
    """
    net = k.models.load_model(filename)
    return net
