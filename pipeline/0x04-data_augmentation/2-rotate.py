#!/usr/bin/env python3
"""
A function that performs a 90 degree rotation
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rerutns rotated image
    """
    img = tf.image.rot90(image, k=1)
    return img
