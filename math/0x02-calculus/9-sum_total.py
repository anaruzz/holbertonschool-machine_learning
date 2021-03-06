#!/usr/bin/env python3
"""
 Script that does the summation function of i²
"""


def summation_i_squared(n):
    """
    sum if i² from 1 to n
    """
    if type(n) is not int or n < 1:
        return None
    if n == 1:
        return 1
    else:
        return int((n * ((n + 1) * (2 * n + 1))) / 6)
