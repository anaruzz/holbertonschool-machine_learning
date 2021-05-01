#!/usr/bin/env python3
"""
Function that performs agglomerative clustering on a dataset:
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Returns: clss a np.array containing cluster indices
    """
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    clss = hierarchy.fcluster(linkage, dist, criterion='distance')
    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.figure()
    plt.show()

    return clss
