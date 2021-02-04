#!/usr/bin/env python3
"""
Script that trains a model with learning rate decay
"""
import tensorflow.keras as k


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                verbose=True,
                save_best=False,
                filepath=None,
                shuffle=False):
    """
    Returns the history object generated after training
    """
    callback = []
    if learning_rate_decay:
        def LRD(step):
            return alpha / (1 + decay_rate * step)
    if early_stopping:
        early_s = k.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience,
                                            verbose=verbose)
        callback.append(early_s)
    if learning_rate_decay and validation_data:
        decay = k.callbacks.LearningRateScheduler(LRD, verbose=1)
        callback.append(decay)
    if save_best:
        best = k.callbacks.ModelCheckpoint(filepath)
        callback.append(best)
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=callback
                          )
    return history
