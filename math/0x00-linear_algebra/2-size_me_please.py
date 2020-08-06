#!/usr/bin/env python3
# Script that calculates the shape of a matrix


def matrix_shape(matrix):
    """
    Function that calculates the shape
    of a matrix
    """
    # size is the length of the lines in the matrix
    size = [len(matrix)]
    while type(matrix[0]) is list:
        """
        while the matrix contains an other matrix enter
        the while loop
        and then each time add the number of elements
        in that line in the certain dimension
        """
        size.append(len(matrix[0]))
        matrix[0] = matrix[0][0]
    return size
