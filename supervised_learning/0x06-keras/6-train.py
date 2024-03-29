#!/usr/bin/env python3
"""
Script that trains a model using early stopping
"""
import tensorflow.keras as k


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                verbose=True,
                shuffle=False):
    """
    Returns the history object generated after training
    """
    if early_stopping:
        callback = k.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience,
                                             verbose=verbose)
    else:
        callback = None
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=[callback]
                          )
    return history
