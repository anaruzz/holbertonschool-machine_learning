#!/usr/bin/env python3
"""
Function that calculates the probability density function
of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    Returns: P, or None on failure

    P is a numpy.ndarray of shape (n,) containing the
    PDF values for each data point

    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    _, d = X.shape
    if (m.shape != (d,)) or (S.shape != (d, d)):
        return None

    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    P1 = 1 / (np.sqrt((2 * np.pi) ** d * det))
    fac = np.einsum('...k,kl,...l->...', X - m, inv, X - m)
    P2 = P1 * np.exp((-1/2) * fac)
    P = np.maximum(P2, 1e-300)
    return P
