#!/usr/bin/env python3
"""
Script that Calculates the accuracy of the prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of the prediction
    """
    prediction = tf.argmax(y_pred, 1)
    correct = tf.argmax(y, 1)
    equality = tf.equal(prediction, correct)
    accuracy = tf.cast(equality, tf.float32)
    accuracy = tf.math.reduce_mean(accuracy)
    return accuracy
