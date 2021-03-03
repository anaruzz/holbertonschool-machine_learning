#!/usr/bin/env python3
"""
Script that  builds a modified version of
the LeNet-5 architecture using keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Returns a k.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    kernel = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            kernel_initializer=kernel,
                            padding='same',
                            activation='relu')(X)

    pool1 = K.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            kernel_initializer=kernel,
                            padding='valid',
                            activation='relu')(pool1)

    pool2 = K.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(conv2)

    CF = K.layers.Flatten()(pool2)

    l1 = K.layers.Dense(units=120,
                        kernel_initializer=kernel,
                        activation='relu')(CF)

    l2 = K.layers.Dense(units=84,
                        kernel_initializer=kernel,
                        activation='relu')(l1)

    l3 = K.layers.Dense(units=10,
                        kernel_initializer=kernel,
                        activation='softmax')(l2)

    model = K.models.Model(inputs=X, outputs=l3)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
