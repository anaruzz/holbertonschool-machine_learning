#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if len(poly) == 0 or type(poly) != list:
        return None

    drv = []
    for i in range(1, len(poly)):
        drv.append(poly[i] * i)
    if len(poly) == 1:
        return [0]
    return drv
