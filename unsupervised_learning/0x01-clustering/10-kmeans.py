#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Returns: C, clss
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    clss = model.labels_
    C = model.cluster_centers_
    return C, clss
