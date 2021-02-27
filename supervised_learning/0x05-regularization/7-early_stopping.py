#!/usr/bin/env python3
"""
Script that determines if you should
stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Returns: a boolean of whether the network should be
    stopped early, followed by the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count >= patience, count
