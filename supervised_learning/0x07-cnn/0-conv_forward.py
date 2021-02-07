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
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = int((h_prev - 1) * sh + kh % 2 - h_prev) + 1
        pw = int((w_prev - 1) * sw + kw % 2 - w_prev) + 1
    else:
        ph, pw = padding

    A_padded = np.pad(array=A_prev,
                      pad_width=((0,),
                                (ph,),
                                (pw,),
                                (0,)),
                      mode="constant",
                      constant_values=0
                      )

    output_h = int((h_prev + 2 * ph - kh) / sh + 1)
    output_w = int((w_prev + 2 * pw - kw) / sw + 1)
    image = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                image[:, i, j, k] = np.sum(W[:, :, :, k] *
                                           A_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                                           axis=(1, 2, 3)
                )
    return activation(image + b)
