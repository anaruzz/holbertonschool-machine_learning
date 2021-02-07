#!/usr/bin/env python3
"""
Script that performs back propagation
over a pooling layer of a neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Returns the partial derivatives with respect
    to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph, pw = 0, 0

    new = A_prev
    dA_new = np.zeros(new.shape)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    if mode == 'max':
                        tmp = new[n,
                                  i * sh:i * sh + kh,
                                  j * sw: j * sw + kw, k]
                        mask = tmp == np.max(tmp)
                        dA_new[n,
                               i * sh:i * sh + kh,
                               j * sw:j * sw + kw,
                               k] += np.multiply(dA[n, i, j, k],
                                                 mask)
                    elif mode == 'avg':
                        dA_new[n,
                               i * sh:i * sh + kh,
                               j * sw:j * sw + kw, k] += dA[n, i, j, k]/kh/kw
    return dA_new
