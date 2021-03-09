#!/usr/bin/env python3
"""
Script that builds the DenseNet-121 architecture
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Returns the keras model
    """
    kernel = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=2*growth_rate,
                            kernel_size=(7, 7),
                            padding='same',
                            strides=(2, 2),
                            kernel_initializer=kernel)(act1)
    max_p = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv1)

    nb_filters = max_p.shape[-1].value

    dense1, nbf1 = dense_block(max_p, nb_filters, growth_rate, 6)
    trans1, nbf2 = transition_layer(dense1, nbf1, compression)

    dense2, nbf3 = dense_block(trans1, nbf2, growth_rate, 12)
    trans2, nbf4 = transition_layer(dense2, nbf3, compression)

    dense3, nbf5 = dense_block(trans2, nbf4, growth_rate, 24)
    trans3, nbf6 = transition_layer(dense3, nbf5, compression)

    dense4, nbf7 = dense_block(trans3, nbf6, growth_rate, 16)
    avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1))(dense4)
    Y = K.layers.Dense(units=1000,
                       activation="softmax",
                       kernel_initializer=kernel)(avg)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
