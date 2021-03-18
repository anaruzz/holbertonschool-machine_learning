#!/usr/bin/env python3
"""
Function that calculates the determinant of a matrix
"""


def getcofactor(m, i, j):
    """
    returns the minor of matrix
    """
    return [row[: j] + row[j+1:] for row in (m[: i] + m[i+1:])]


def determinant(matrix):
    """
    Returns the determinant of matrix
    """
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]

    if type(matrix) is not list or matrix == []:
        raise TypeError('matrix must be a list of lists')

    for i in matrix:
        if type(i) != list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(i):
            raise ValueError("matrix must be a square matrix")
            
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:
        det = 0
        for j in range(len(matrix)):
            sign = (-1) ** j
            cofactor = getcofactor(matrix, 0, j)
            det += sign * matrix[0][j] * determinant(cofactor)
    return det
