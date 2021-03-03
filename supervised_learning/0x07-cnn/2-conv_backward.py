#!/usr/bin/env python3
""" Script that performs back propagation
over a pooling layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Returns the partial derivatives with respect to the
    previous layer (dA_prev), the kernels (dW),
    and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, kc, knc = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    ph, pw = 0, 0

    if padding == 'same':
        ph = (((h_new * sh) - sh + kh - h_new) // 2) + 1
        pw = (((w_new * sw) - sw + kw - w_new) // 2) + 1
    if type(padding) == tuple:
        ph, pw = padding

    new = np.pad(A_prev, ((0, 0), (ph, ph),
                          (pw, pw), (0, 0)),
                 'constant')

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    DZ_new = np.zeros(new.shape)
    dW = np.zeros_like(W)
    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(knc):
                    DZ_new[n, i*sh:i*sh+kh,
                           j*sw:j*sw+kw, :] += np.multiply(dZ[n, i, j, k],
                                                           W[..., k])
                    dW[..., k] += np.multiply(dZ[n, i, j, k],
                                              new[n,
                                                  i*sh:i*sh+kh,
                                                  j*sw:j*sw+kw, :])
    if padding == 'same':
        DZ_new = DZ_new[:, ph:-ph, pw:-pw, :]
    return DZ_new, dW, db
