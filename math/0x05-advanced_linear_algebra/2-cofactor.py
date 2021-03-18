#!/usr/bin/env python3
"""
Function that calculates the cofactor of a matrix
"""


def minor(m, i, j):
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
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) != list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(i):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:
        det = 0
        for j in range(len(matrix)):
            sign = (-1) ** j
            cofactor = minor(matrix, 0, j)
            det += sign * matrix[0][j] * determinant(cofactor)
    return det


def cofactor(matrix):
    """
    Returns the cofactor matrix of matrix
    """
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) != list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(i):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    m = []
    for i in range(len(matrix)):
        s = []
        for j in range(len(matrix[0])):
            new = []
            for row in (matrix[:i] + matrix[i + 1:]):
                new.append(row[:j] + row[j + 1:])
            sign = (-1) ** (i + j)
            s.append(determinant(new) * sign)
        m.append(s)
    return m
