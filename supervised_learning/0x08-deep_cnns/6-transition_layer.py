#!/usr/bin/env python3
"""
Script  that builds a transition layer
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Returns The output of the transition layer and the
    number of filters within the output, respectively
    """
    kernel = K.initializers.he_normal()
    nbf = int(nb_filters * compression)

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=nbf,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=kernel)(act1)
    avg = K.layers.AveragePooling2D(pool_size=(2, 2),
                                    padding='same',
                                    )(conv1)
    return avg, nbf
