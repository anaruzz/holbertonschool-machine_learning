#!/usr/bin/env python3
"""
Script that creates a tensorflow layer that
uses dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Returns the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, name='layer',
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)(prev)
    return layer
