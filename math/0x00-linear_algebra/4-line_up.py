#!/usr/bin/env python3
"""
Script that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    function that adds two arrays element-wise
    """
    if len(arr1) == len(arr2):
        new = []
        for i in range(len(arr1)):
            new.append(arr1[i] + arr2[i])
        return new
    else:
        return None
