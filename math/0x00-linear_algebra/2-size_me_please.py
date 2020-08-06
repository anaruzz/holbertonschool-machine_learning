#!/usr/bin/env python3
a = []
def matrix_shape(matrix):
    size = [len(matrix)]
    while type(matrix[0]) is list:
        size.append(len(matrix[0]))
        matrix[0] = matrix[0][0]
    return size
    # a.append(len(matrix))
    # if type(matrix[0]) is not list:
    #     return a
    # else:
    #     a.append(matrix_shape(matrix[0]))
    #     print(a)
