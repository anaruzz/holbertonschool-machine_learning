#!/usr/bin/env python3
"""
Script that performs element-wise addition,
subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    function that performs element-wise addition,
    subtraction, multiplication, and division
    """
    t = (mat1+mat2, mat1-mat2, mat1*mat2, mat1/mat2)
    return t
