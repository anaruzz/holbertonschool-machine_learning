#!/usr/bin/env python3
"""
A function that creates a pd.DataFrame from a np.ndarray
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Returns the created pd.DataFrame
    """
    abc = list("ABCDEFGH")
    length = array.shape[1]
    return pd.DataFrame(array, columns=abc[:length])
