#!/usr/bin/env python3
"""
Script that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    function that does integral
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int:
        return None
    if len(poly) == 1:
        return [0]
    new = [0, poly[0]]
    for i in range(1, len(poly)):
        p = (poly[i] / (i + 1))
        new.append(p)
    return new
