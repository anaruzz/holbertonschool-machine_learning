#!/usr/bin/env python3
"""
Script that performs a strided convolution on
grayscale images with custom padding
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Returns a numpy.ndarray containing
    the convolved images
    """
    m, h, w = images.shape[:3]
    kh, kw = kernel.shape[:2]
    sh, sw = stride

    if type(padding) is tuple:
        padd_h, padd_w = padding
    elif padding == 'valid':
        padd_h, padd_w = 0, 0
    else:
        padd_h = (((h - 1) * sh + kh - h) // 2) + 1
        padd_w = (((w - 1) * sw + kw - w) // 2) + 1

    filter_w = (w - kw + (2 * padd_w)) // sw + 1
    filter_h = (h - kh + (2 * padd_h)) // sh + 1

    output = np.zeros((m, filter_h, filter_w))

    padded_img = np.pad(
        array=images,
        pad_width=((0, 0), (padd_h, padd_h), (padd_w, padd_w), (0, 0)),
        mode="constant",
    )
    for i in range(filter_h):
        for j in range(filter_w):
            output[:, i, j] = np.sum(
                kernel * padded_img[:, i*sh: i*sh+kh,
                                    j*sw: j*sw+kw], axis=(1, 2, 3))
    return output
