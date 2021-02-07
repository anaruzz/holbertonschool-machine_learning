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
    init = k.initializers.he_normal(seed=None)

    conv1 = k.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=init)(X)

    pool1 = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = k.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)

    pool2 = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flt = k.layers.Flatten()(pool2)

    n1 = k.layers.Dense(units=120, activation='relu',
                        kernel_initializer=init)(flt)

    n2 = k.layers.Dense(units=84, activation='relu',
                        kernel_initializer=init)(n1)

    y_pred = k.layers.Dense(units=10, activation='softmax',
                        kernel_initializer=init)(n2)

    model = k.Model(inputs=X, outputs=y_pred)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model
