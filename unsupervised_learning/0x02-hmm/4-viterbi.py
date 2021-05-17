#!/usr/bin/env python3
"""
Function that calculates the most likely sequence of hidden
states for a hidden markov model
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Returns: path, P, or None, None on failure

    path is the a list of length T containing the most
    likely sequence of hidden states
    P is the probability of obtaining the path sequence

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

    V = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    B = np.zeros((N, T))

    for j in range(1, T):
        for i in range(N):
            temp = Emission[i, Observation[j]] * Transition[:, i] * V[:, j - 1]
            V[i, j] = np.max(temp, axis=0)
            B[i, j] = np.argmax(temp, axis=0)

    P = np.max(V[:, T - 1])
    S = np.argmax(V[:, T - 1])
    path = [S]

    for j in range(T - 1, 0, -1):
        S = int(B[S, j])
        path.append(S)
    path = path[:: -1]

    return path, P
