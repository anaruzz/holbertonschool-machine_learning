#!/usr/bin/env python3
"""
Script that performs a valid convolution on
grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Returns a numpy.ndarray containing
    the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    output = np.zeros((m, h-kh+1, w-kw+1))
    for i in range(h-kh+1):
        for j in range(w-kw+1):
            output[:, i, j] = np.sum(
                kernel * images[:, i: i+kh, j: j+kw], axis=(1,2))
    return output
