#!/usr/bin/env python3
"""
Script that creates a batch normalization
layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    returns a tensor of the activated output for the layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=kernel)(prev)
    mean, var = tf.nn.moments(layer, axes=[0])
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))
    epsilon = 1e-8
    z = tf.nn.batch_normalization(layer,
                                  mean,
                                  var,
                                  beta,
                                  gamma,
                                  epsilon)
    return activation(z)
