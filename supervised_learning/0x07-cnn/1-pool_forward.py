#!/usr/bin/env python3
""" Script that performs forward propagation
over a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Returns the output of the pooling layer
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    if mode == "max":
        op = np.max
    else:
        op = np.average

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kh) // sw + 1
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j, :] = op(
                A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                axis=(1, 2)
            )
    return output
