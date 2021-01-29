#!/usr/bin/env python3
"""
Script  that creates a learning rate decay operation
in tensorflow using inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    returns the learning rate decay operation
    """
    gd = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
    return gd
