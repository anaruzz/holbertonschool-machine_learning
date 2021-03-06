#!/usr/bin/env python3
"""
Script that concatenates two matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function that concatenates two matrices
    """

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        new = []
        for i in range(len(mat1)):
            new.append(mat1[i] + mat2[i])
        return(new)
    else:
        return(None)
