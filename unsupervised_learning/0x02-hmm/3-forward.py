#!/usr/bin/env python3
"""
Function  that performs the forward algorithm for
a hidden markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Returns: P, F, or None, None on failure

    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the forward path probabilities
        F[i, j] is the probability of being in hidden state i at time j given the previous observations
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    n, m = P.shape
    if n != m:

        return None

    if np.any(P < 0):
        return None

    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    for i in range(n):
        if P[i, i] == 1:
            return True
    return False
