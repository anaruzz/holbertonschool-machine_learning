#!/usr/bin/env python3
# Script that calculates the shape of a matrix


def matrix_shape(matrix):
    size = [len(matrix)]
    while type(matrix[0]) is list:
        size.append(len(matrix[0]))
        matrix[0] = matrix[0][0]
    return size
