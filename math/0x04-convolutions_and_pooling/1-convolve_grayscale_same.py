#!/usr/bin/env python3
"""
Script that performs a same convolution on
grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Returns a numpy.ndarray containing
    the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    padd_h = (kh - 1) // 2 if kh % 2 else kh // 2
    padd_w = (kw - 1) // 2 if kw % 2 else kw // 2

    output = np.zeros((m, h, w))
    padded_img = np.pad(
        array=images,
        pad_width=((0,), (padd_h,), (padd_w,)),
        mode="constant",
    )
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                kernel * padded_img[:, i: i+kh, j: j+kw], axis=(1, 2))
    return output
