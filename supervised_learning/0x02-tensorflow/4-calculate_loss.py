#!/usr/bin/env python3
"""
Script that calculates
the softmax cross-entropy loss of a prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the loss function
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
