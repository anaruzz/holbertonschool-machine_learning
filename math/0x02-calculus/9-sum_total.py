#!/usr/bin/env python3

def summation_i_squared(n):
    if type(n) is not int:
        return None
    if n < 1:
        return None
    if n == 1:
        return 1
    else:
        return n ** 2 + summation_i_squared(n-1)
