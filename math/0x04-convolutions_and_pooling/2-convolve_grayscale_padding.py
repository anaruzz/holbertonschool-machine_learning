#!/usr/bin/env python3
"""
Script that performs a same convolution on
grayscale images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Returns a numpy.ndarray containing
    the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    padd_h, padd_w = padding

    h = h - kh + 2 * padd_h + 1
    w = w - kw + 2 * padd_w + 1

    output = np.zeros((m, h, w))

    padded_img = np.pad(
        array=images,
        pad_width=((0, 0), (padd_h, padd_h), (padd_w, padd_w)),
        mode="constant",
    )
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                kernel * padded_img[:, i: i+kh, j: j+kw], axis=(1, 2))
    return output
