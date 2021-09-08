#!/usr/bin/env python3
"""
A function that randomly shears an image:
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Rerutns the sheered image
    """
    img = tf.keras.preprocessing.image.random_shear(image, intensity)
    return img
