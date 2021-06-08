#!/usr/bin/env python3
"""
Script that finds the best number of clusters for a GMM
using the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Returns: pi, m, S, or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None

    if type(kmin) is not int or kmin < 1:
        return None, None, None, None

    if type(kmax) is not int or kmax < 1:
        return None, None, None, None

    if kmax - kmin < 1:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(iterations) is not int or iterations < 1:
        return None, None, None, None

    n, d = X.shape
    best = None
    bics = []
    loglikes = []
    for k in range(kmin, kmax + 1):
        EM = expectation_maximization(X, k,
                                      iterations,
                                      tol,
                                      verbose)
        pi, m, S, _, ll = EM
        bic = (6 * k - 1) * np.log(n) - 2 * ll
        if best is None or bics[best - kmin] > bic:
            best = k
            best_result = (pi, m, S)
        bics.append(bic)
        loglikes.append(ll)
        l_res = np.asarray(loglikes)
        b = np.asarray(bics)
    return best, best_result, l_res, b
