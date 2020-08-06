#!/usr/bin/env python3
def matrix_transpose(matrix):
    i = 0
    new = [[0 for i in range(len(matrix))] for j in range(len(matrix[i]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            new[j][i] = matrix[i][j]
    return new
