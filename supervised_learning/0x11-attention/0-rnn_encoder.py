#!/usr/bin/env python3
"""
a class RNNEncoder that inherits from tensorflow.keras.layers.Layer
to encode for machine translation
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class methods
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNEncoder, self).__init__()
        self.vocab = vocab

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.batch = batch
        self.gru = tf.keras.layers.GRU(units,
                                       kernel_initializer="glorot_uniform",
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True
                                       )

    def initialize_hidden_state(self):
        """
        method that initializes the hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        calls the encoders layers
        Returns: outputs, hidden
        """
        embading = self.embedding(x)
        outputs = self.gru(embading, initial_state=initial)
        return outputs
