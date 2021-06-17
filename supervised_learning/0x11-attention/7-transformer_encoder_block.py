#!/usr/bin/env python3
"""
class EncoderBlock that inherits from tensorflow.keras.layers.Layer
to create an encoder block for a transformer
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class methods
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Returns:
                a tensor of shape (batch, input_seq_len, dm) containing
                the block’s output
        """
        mid, w = self.mha(x, x, x, mask)
        mid = self.dropout1(mid, training=training)
        mid = self.layernorm1(x + mid)
        output = self.dense_hidden(mid)
        output = self.dense_output(output)
        output = self.dropout2(output, training=training)
        output = self.layernorm2(mid + output)
        return output
