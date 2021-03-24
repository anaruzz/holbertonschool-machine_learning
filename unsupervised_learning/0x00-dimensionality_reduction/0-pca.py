#!/usr/bin/env python3
"""
A function that perfors PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """ Function that performs PCA on a dataset: """
    U, s, V = np.linalg.svd(X)
    cumulated = np.cumsum(s)
    percentage = cumulated / np.sum(s)
    r = np.argwhere(percentage >= var)[0, 0]
    return V[:r + 1].T
