#!/usr/bin/env python3
""" Function that performs the Baum-Welch algorithm for a
hidden markov model
"""
import numpy as np


def baum_welch(Obs, Transition, Emission, Init, iterations=1000):
    """
    Returns: the converged Transitionition, Emissionion, or None, None on failure
    """
    T = Obs.shape[0]
    N, M = Emission.shape

    def forward(Obs, Transition, Emission, Init):
        """
        calculates the most likely sequence of hidden states for a
        hidden markov model
        """
        alpha = np.ndarray((N, T))
        state = Init[:, 0]
        for t in range(T):
            state = state * Emission[:, Obs[t]]
            alpha[:, t] = state
            state = np.matmul(Transition.T, state)
        return alpha[:, -1].sum(), alpha

    def backward(Obs, Transition, Emission, Init):
        """
        performs the backward algorithm for a hidden markov model
        """
        beta = np.ndarray((N, T))
        state = np.asarray([1] * N)
        beta[:, -1] = state
        for t in range(T - 2, -1, -1):
            state = np.matmul(Transition, state * Emission[:, Obs[t + 1]])
            beta[:, t] = state
        return (beta[:, 0] * Init[:, 0]
                * Emission[:, Obs[0]]).sum(), beta
    a = None
    while iterations:
        iterations -= 1
        P, alpha = forward(Obs, Transition, Emission, Init)
        P2, beta = backward(Obs, Transition, Emission, Init)

        forbackxi = alpha[:, None, :-1] * beta[None, :, 1:]
        emittedprobs = Emission[:, Obs[1:]]
        xi = forbackxi * Transition[..., None] * emittedprobs[None, :, ...]
        xi /= xi.sum(axis=(0, 1))
        forbackga = alpha * beta
        gamma = forbackga / forbackga.sum(axis=0)
        Transition = xi.sum(axis=2) / xi.sum(axis=(1, 2))
        Transition = Transition.T
        for emit in range(M):
            gammanum = gamma[:, Obs == emit]
            Emission[:, emit] = gammanum.sum(axis=1) / gamma.sum(axis=1)
        if ((np.all(a == Transition)
             and np.all(emprev == Emission)
             and np.all(initprev == Init))):
            return Transition, Emission
        a = Transition
        initprev = Init
        emprev = Emission
    return Transition, Emission
