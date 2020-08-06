#!/usr/bin/env python3
matrix_shape = __import__('2-size_me_please').matrix_shape

def add_matrices2D(mat1, mat2):
    if matrix_shape(mat1) == matrix_shape(mat2):
        new = []
        for i in range(len(mat1)):
            line = []
            for j in range(a):
                b = int(mat1[i][j]) + int(mat2[i][j])
                line.append(b)
            new.append(line)
        return new
    else:
        return None
