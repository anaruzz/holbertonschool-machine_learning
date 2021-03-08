#!/usr/bin/env python3
"""
Script that builds an inception network
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Returns the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    l1x1 = K.layers.Conv2D(filters=F1,
                           kernel_size=(1, 1),
                           padding='same',
                           activation='relu')(A_prev)

    l1x1_1 = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu')(A_prev)

    l3x3 = K.layers.Conv2D(filters=F3,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(l1x1_1)

    l1x1_2 = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu')(A_prev)

    l5x5 = K.layers.Conv2D(filters=F5,
                           kernel_size=(5, 5),
                           padding='same',
                           activation='relu')(l1x1_2)

    l_max = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(1, 1),
                                  padding='same')(A_prev)

    l1x1_3 = K.layers.Conv2D(filters=FPP,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu')(l_max)

    output = K.layers.Concatenate()([l1x1, l3x3, l5x5, l1x1_3])

    return output
