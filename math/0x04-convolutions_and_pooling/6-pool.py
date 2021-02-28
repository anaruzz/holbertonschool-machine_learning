#!/usr/bin/env python3
"""
Script that performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Returns a numpy.ndarray containing
    the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    filter_w = (w - kw) // sw + 1
    filter_h = (h - kh) // sh + 1

    output = np.zeros((m, filter_h, filter_w, c))

    for i in range(filter_h):
        for j in range(filter_w):
            if mode == "max":
                output[:, i, j, :] = np.max(
                    images[:, i*sh: i*sh+kh, j*sw: j*sw+kw, :],
                    axis=(1, 2))
            else:
                output[:, i, j, :] = np.average(
                    images[:, i*sh: i*sh+kh, j*sw: j*sw+kw, :],
                    axis=(1, 2))
    return output
