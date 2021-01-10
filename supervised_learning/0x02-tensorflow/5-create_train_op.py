#!/usr/bin/env python3
"""
Script that creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Training operation
    """
    gd = tf.train.GradientDescentOptimizer(alpha)
    return gd.minimize(loss)
