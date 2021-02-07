#!/usr/bin/env python3
"""
Script that  builds a modified version of
the LeNet-5 architecture using keras
"""

import tensorflow as tf
import tensorflow.keras as k


def lenet5(X):
    """
    Returns a k.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=init)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)

    flat = K.layers.Flatten()(pool2)

    f1 = K.layers.Dense(units=120, activation='relu',
                        kernel_initializer=init)(flat)

    f2 = K.layers.Dense(units=84, activation='relu',
                        kernel_initializer=init)(f1)

    y_pred = K.layers.Dense(units=10, activation='softmax',
                        kernel_initializer=init)(f2)

    model = K.models.Model(inputs=X, outputs=y_pred)
    optimizer = K.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
