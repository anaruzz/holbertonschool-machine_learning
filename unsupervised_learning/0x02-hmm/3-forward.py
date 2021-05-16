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
    F is a numpy.ndarray of shape (N, T) containing the forward
    path probabilities
    F[i, j] is the probability of being in hidden state i at time j
    given the previous observations
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:

        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:

        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    test = np.sum(Emission, axis=1)
    if not (test == 1).all():
        return None, None

    test = np.sum(Transition, axis=1)
    if not (test == 1).all():
        return None, None

    test = np.sum(Initial, axis=0)
    if not (test == 1).all():
        return None, None

    T = Observation.shape[0]
    F = np.zeros((Initial.shape[0], T))
    index = Observation[0]
    Emission_ind = Emission[:, index]
    F[:, 0] = Initial.T * Emission_ind

    for j in range(1, T):
        for i in range(Initial.shape[0]):
            F[i, j] = np.sum(Emission[i, Observation[j]]
                             * Transition[:, i] * F[:, j - 1], axis=0)

    P = np.sum(F[:, T-1:], axis=0)[0]
    return P, F
