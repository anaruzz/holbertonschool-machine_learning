#!/usr/bin/env python3
"""
Function that performs the expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    returns: pi, m, S, g, l, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(k) is not int or k < 1:
        return None

    if type(tol) is not float or tol < 0:
        return None

    if type(verbose) is not bool:
        return None

    if type(iterations) is not int or iterations < 1:
        return None

    pi, m, S = initialize(X, k)
    g, likelihood = expectation(X, pi, m, S)

    for i in range(iterations):
        prev = likelihood
        if verbose and (i % 10) == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, likelihood.round(5)))
        pi, m, S = maximization(X, g)
        g, likelihood = expectation(X, pi, m, S)
        if abs(likelihood - prev) <= tol:
            break

    if verbose:
        print("log likelihood after {} iterations: {}"
              .format(i, likelihood.round(5)))

    return pi, m, S, g, likelihood
