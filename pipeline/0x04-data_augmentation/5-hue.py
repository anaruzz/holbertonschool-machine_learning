#!/usr/bin/env python3
"""
A function that changes the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Rerutns the altered image
    """
    image = tf.image.adjust_hue(image, delta)
    return image
