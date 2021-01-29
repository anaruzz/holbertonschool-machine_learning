#!/usr/bin/env python3
"""
Script that updates a variable in place using
inverse time decay in numpy
"""
import numpy as np

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    returns the updated value for alpha
    """
    alpha /= (1 + decay_rate * (global_step // decay_step))
    return alpha
