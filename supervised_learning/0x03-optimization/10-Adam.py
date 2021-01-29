#!/usr/bin/env python3
"""
Script that creates the training operation for a
neural network in tensorflow using the Adam
optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    returns the Adam optimization operation
    """
    gd = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon=epsilon)
    return gd.minimize(loss)
