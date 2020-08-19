#!/usr/bin/env python3
"""
Script that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    function that does derivative
    """
    new = []
    if type(poly) is not list or len(poly) == 0:
        return None
    for i in range(1, len(poly)):
        new.append(i * poly[i])
    while new[len(new)-1] == 0:
        new.pop()
    if len(new) == 0:
        new = [0]
    return new
