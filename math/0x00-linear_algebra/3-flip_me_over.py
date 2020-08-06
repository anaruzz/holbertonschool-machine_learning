#!/usr/bin/env python3
"""
Script that returns the transpose of a 2D matrix
"""
def matrix_transpose(matrix):
    """
    function that returns
    the transpose of a 2D matrix,
    """
    i = 0
    new = [[0 for i in range(len(matrix))] for j in range(len(matrix[i]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            new[j][i] = matrix[i][j]
    return new
