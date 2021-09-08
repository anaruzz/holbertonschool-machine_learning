#!/usr/bin/env python3
"""
A function that flips an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Returns: the flipped image
    """
    image = tf.image.flip_left_right(image)
    return image
