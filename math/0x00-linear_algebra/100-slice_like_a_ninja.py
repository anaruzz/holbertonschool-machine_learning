#!/usr/bin/env python3
"""
Script that slices a matrix along a specific axes
"""


def np_slice(matrix, axes={}):
    """
    Script that slices a matrix along a specific axes
    """
    new = []
    i = 0
    for key, value in axes.items():
        while i < key:
            new.append(slice(None))
            i += 1
        new.append(slice(*value))
        i += 1
    return matrix[tuple(new)]
