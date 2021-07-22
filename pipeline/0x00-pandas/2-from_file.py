#!/usr/bin/env python3
"""
A Function that creates a pd.DataFrame from a file
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Returns the loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
