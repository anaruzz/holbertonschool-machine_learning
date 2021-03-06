#!/usr/bin/env python3
""" Script that performs forward propagation over
a convolutional layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    Returns the output of the convolutional layer
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw, c, nc = W.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = int(((h_prev - 1) * sh + kh - kh % 2 - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - kw % 2 - w_prev) / 2) + 1
    else:
        ph, pw = padding

    output = np.zeros((m, (h_prev + 2 * ph), (w_prev + 2 * pw), c))
    output[:, ph:h_prev + ph, pw:w_prev + pw, :] = A_prev.copy()

    nh = np.floor(((h_prev + 2 * ph - kh) / stride[0]) + 1).astype(int)
    nw = np.floor(((w_prev + 2 * pw - kw) / stride[1]) + 1).astype(int)
    S = np.zeros((m, nh, nw, nc))
    im = np.arange(0, m)

    for i in range(nh):
        for j in range(nw):
            for k in range(nc):
                S[im, i, j, k] += np.sum(output[im, sh * i:sh * i + kh,
                                         sw * j:sw * j + kw, :]
                                         * W[:, :, :, k],
                                         axis=(1, 2, 3))
    return activation(S + b)
