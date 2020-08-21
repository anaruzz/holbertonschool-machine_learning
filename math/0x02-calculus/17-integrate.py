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
    if type(C) is not int and type(C) is not float:
        return None
    if len(poly) == 1:
        return [C]
    new = [C]
    for i in range(len(poly)):
        p = (poly[i] / (i + 1))
        if p == int(p):
            new.append(int(p))
        else:
            new.append(p)
    return new
