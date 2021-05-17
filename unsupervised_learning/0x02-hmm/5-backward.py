#!/usr/bin/env python3
"""
Function that performs the backward
algorithm for a hidden markov model:
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Returns: P, B, or None, None on failure

    Pis the likelihood of the observations given the model
    B is a numpy.ndarray of shape (N, T) containing the
    backward path probabilities
        B[i, j] is the probability of generating the future
        observations from hidden state i at time j
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    N = Initial.shape[0]
    if ((Transition.shape[0] != N or Transition.shape[1] != N
         or Initial.shape[0] != N)):
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

    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    for j in range(T - 2, -1, -1):
        for i in range(N):
            B[i, j] = np.sum(B[:, j + 1] * Emission[:, Observation[j + 1]]
                             * Transition[i, :], axis=0)
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0], axis=1)[0]

    return P, B
