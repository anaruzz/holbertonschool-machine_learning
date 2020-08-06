#!/usr/bin/env python3
"""
Script that concatenates two matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function that concatenates two matrices
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
            m = mat1 + mat2
            return(m)
    elif axis == 1 and len(mat1) == len(mat2):
            n = []
            for i in range(len(mat1)):
                n.append(mat1[i] + mat2[i])
            return(n)
    else:
            return(None)
