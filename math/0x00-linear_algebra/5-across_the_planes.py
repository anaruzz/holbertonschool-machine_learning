#!/usr/bin/env python3
"""
Script that adds two matrices element wise
"""


def add_matrices2D(mat1, mat2):
    """
    function that adds two matrices element wise
    """
    s = len(mat1)
    if s - len(mat2) != 0 or len(mat1[0]) - len(mat2[0]) != 0:
        return None
    else:
        new = []
        j = 0
        for i in range(s):
            line = []
            for j in range(len(mat1[0])):
                line.append(mat1[i][j] + mat2[i][j])
            new.append(line)
        return new
