#!/usr/bin/env python3
"""
Script that saves and loads an entire model's configuraton
"""
import tensorflow.keras as k


def save_config(network, filename):
    """
    saves model configuration
    """
    json_format = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_format)
    return None


def load_config(filename):
    """
    loads model's configuration
    """
    with open(filename, 'r') as f:
        content = f.read()
    return k.models.model_from_json(content)
