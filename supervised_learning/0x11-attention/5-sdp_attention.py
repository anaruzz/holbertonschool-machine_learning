#!/usr/bin/env python3
"""
A function that  calculates the scaled dot product attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Returns: output, weights

    output containing the scaled dot product attention
    weights containing the attention weights
    """
    sqrt = tf.math.sqrt(tf.cast(tf.shape(K)[-1], float))
    scaled = tf.matmul(Q, K, transpose_b=True) / sqrt
    if mask is not None:
        scaled = mask * -1e9 + scaled
    scaled = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(scaled, V), scaled
