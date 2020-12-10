#!/usr/bin/env python3
"""
Script that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    function that does derivative
    """

    if type(poly) is not list or len(poly) == 0:
        return None
    new = []
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        new.append(i * poly[i])
    return new
