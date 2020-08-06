#!/usr/bin/env python3
"""
Script that multiplies two matrices
"""


def mat_mul(mat1, mat2):
    """
    function that multiplies two matrices
    """

    lines1, lines2 = len(mat1), len(mat2)
    rows1, rows2 = len(mat1[0]), len(mat2[0])

    if lines2 == rows1:
        new = []
        for i in range(len(mat1)):
            t = []
            for j in range(len(mat2[0])):
                s = 0
                for k in range(len(mat2)):
                    s += mat1[i][k] * mat2[k][j]
                t.append(s)
            new.append(t)
        return new
    else:
        return None
