#!/usr/bin/env python3
"""
Script that builds a projection block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Returns the activated output of the projection block
    """
    F11, F3, F12 = filters
    kernel = K.initializers.he_normal()

    layer1x1 = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=kernel)(A_prev)

    layer1x1 = K.layers.BatchNormalization()(layer1x1)

    layer1x1 = K.layers.Activation('relu')(layer1x1)

    layer3x3 = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=kernel)(layer1x1)

    layer3x3 = K.layers.BatchNormalization()(layer3x3)
    layer3x3 = K.layers.Activation('relu')(layer3x3)

    layer1x1 = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=kernel)(layer3x3)

    layer1x1 = K.layers.BatchNormalization()(layer1x1)
    output = K.layers.Add()([layer1x1, A_prev])

    output = K.layers.Activation('relu')(output)

    return output
