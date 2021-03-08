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

    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=kernel,
                            strides=s)(A_prev)

    norm1 = K.layers.BatchNormalization()(conv1)

    act1 = K.layers.Activation('relu')(norm1)

    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=kernel)(act1)

    norm2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(norm2)

    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=kernel)(act2)

    norm3 = K.layers.BatchNormalization()(conv3)

    conv4 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=kernel,
                            strides=s)(A_prev)

    norm4 = K.layers.BatchNormalization()(conv4)
    output = K.layers.Add()([norm3, norm4])

    output = K.layers.Activation('relu')(output)

    return output
