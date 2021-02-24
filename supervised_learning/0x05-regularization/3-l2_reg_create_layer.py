#!/usr/bin/env python3
"""
Script that creates a tensorflow layer that
includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Returns the output of the new layer
    """
    init = (tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, name='layer',
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)(prev)
    return layer
