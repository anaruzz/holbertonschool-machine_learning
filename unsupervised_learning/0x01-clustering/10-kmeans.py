#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    clss = model.labels_
    C = model.cluster_centers_
    return C, clss
