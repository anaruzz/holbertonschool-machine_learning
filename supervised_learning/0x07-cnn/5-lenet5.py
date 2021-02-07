#!/usr/bin/env python3
"""
Script that  builds a modified version of
the LeNet-5 architecture using keras
"""

import tensorflow as tf
import tensorflow.keras as K


def lenet5(X):
    """
    Returns a k.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal'
    )(X)
    pool1 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal'
    )(M1)
    pool2 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)
    CF = K.layers.Flatten()(conv2)
    F1 = K.layers.Dense(
        units=120,
        kernel_initializer="he_normal",
        activation='relu'
    )(CF)
    f2 = K.layers.Dense(
        units=84,
        kernel_initializer="he_normal",
        activation='relu'
    )(f1)
    F3 = K.layers.Dense(
        units=10,
        kernel_initializer="he_normal",
        activation='softmax'
    )(F2)
    model = K.Model(inputs=X, outputs=y_pred)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
