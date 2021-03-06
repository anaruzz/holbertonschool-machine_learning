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
    if validation_data:
        callbacks = []
        if learning_rate_decay:
            def LRD(step):
                return alpha / (1 + decay_rate * step)
            decay = k.callbacks.LearningRateScheduler(LRD, verbose=1)
            callbacks.append(decay)

        if early_stopping:
            early_s = k.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                verbose=verbose)
            callbacks.append(early_s)

        if save_best:
            best = k.callbacks.ModelCheckpoint(filepath,
                                               monitor="val_loss",
                                               save_best_only=True)
            callbacks.append(best)

        history = network.fit(x=data,
                              y=labels,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              callbacks=callbacks
                              )
    else:
        history = network.fit(x=data,
                              y=labels,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose,
                              validation_data=validation_data,
                              shuffle=shuffle
                              )

    return history
