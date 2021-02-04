#!/usr/bin/env python3
"""
Script converts a label vector
into a one-hot matrix
"""
import tensorflow.keras as k


def one_hot(labels, classes=None):
    """
    Returns one hot matrix
    """
    encoded = k.utils.to_categorical(labels, classes)
    return encoded
