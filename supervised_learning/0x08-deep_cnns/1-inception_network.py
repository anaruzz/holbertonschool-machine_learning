#!/usr/bin/env python3
"""
Script that builds an inception block
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Returns the concatenated output of the inception block
    """

    X = K.Input(shape=(224, 224, 3))

    l7x7 = K.layers.Conv2D(filters=64,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           padding='same',
                           activation='relu')(X)

    l_max1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same')(l7x7)

    l3x3 = K.layers.Conv2D(filters=64,
                           kernel_size=(1, 1),
                           padding='same',
                           activation='relu')(l_max1)

    l3x3 = K.layers.Conv2D(filters=192,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           activation='relu')(l3x3)

    l_max2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same')(l3x3)

    inception = inception_block(l_max2, [64, 96, 128, 16, 32, 32])
    inception = inception_block(inception, [128, 128, 192, 32, 96, 64])

    lmax3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(inception)

    inception1 = inception_block(lmax3, [192, 96, 208, 16, 48, 64])
    inception2 = inception_block(inception1, [160, 112, 224, 24, 64, 64])
    inception3 = inception_block(inception1, [128, 128, 256, 24, 64, 64])
    inception4 = inception_block(inception3, [112, 144, 288, 32, 64, 64])
    inception5 = inception_block(inception4, [256, 160, 320, 32, 128, 128])

    lmax4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(inception5)
    inception6 = inception_block(lmax4, [256, 160, 320, 32, 128, 128])
    inception7 = inception_block(inception6, [384, 192, 384, 48, 128, 128])

    avg_l = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(inception7)

    drop = K.layers.Dropout(rate=0.4)(avg_l)

    Y = K.layers.Dense(units=1000, activation='softmax')(drop)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
