#!/usr/bin/env python3
"""
A function that randomly changes the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Rerutns the altered image
    """
    img = tf.image.random_brightness(image, max_delta)
    return img
