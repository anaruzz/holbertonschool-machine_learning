#!/usr/bin/env python3
"""
A function that performs a random crop of an image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Returns cropped image
    """
    image = tf.image.random_crop(image, size, seed=None, name=None)
    return image
