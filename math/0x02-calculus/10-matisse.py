#!/usr/bin/env python3
"""
Script that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    new = []
    new.append(poly[1])
    if type(poly) is not list or len(poly) == 0:
        return None
    for i in range(len(poly)-1):
        new.append(i * poly[i])
    while new[len(new)-1] == 0:
        new.pop()
    return new
