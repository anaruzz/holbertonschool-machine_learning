#!/usr/bin/env python3
"""
Script that returns 2 placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Create 2 placeholders
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y
