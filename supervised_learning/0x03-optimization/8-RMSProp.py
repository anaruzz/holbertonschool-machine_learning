#!/usr/bin/env python3
"""
Script that creates the training operation for a
neural network in tensorflow using the RMSProp
optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    returns the RMSProp optimization operation
    """
    gd = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    return gd.minimize(loss)
