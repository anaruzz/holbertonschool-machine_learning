#!/usr/bin/env python3
"""
Function that calculates a GMM from a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """
    Returns: pi, m, S, clss, bic
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    c = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, c, clss, bic
