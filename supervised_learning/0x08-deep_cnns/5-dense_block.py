#!/usr/bin/env python3
"""
Script that builds the ResNet-50 architecture
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Returns the concatenated output of each layer within
    the Dense Block and the number of filters within
    the concatenated outputs, respectively.
    """
    output = X
    kernel = K.initializers.he_normal()

    for l in range(layers):
        norm1 = K.layers.BatchNormalization()(output)
        act1 = K.layers.Activation('relu')(norm1)
        conv1 = K.layers.Conv2D(filters=128,
                                kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=kernel)(act1)
        norm2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer=kernel)(act2)
        output = K.layers.Concatenate()([output, conv2])
        nb_filters += growth_rate
        return output, nb_filters
