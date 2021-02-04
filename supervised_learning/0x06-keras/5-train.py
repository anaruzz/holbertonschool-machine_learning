#!/usr/bin/env python3
"""
Script that trains a model using mini-batch gradient descent
"""
import tensorflow.keras as k


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                verbose=True, shuffle=False):
    """
    Returns the history object generated after training
    """
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle
                          )
    return history
