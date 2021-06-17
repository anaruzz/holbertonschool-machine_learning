#!/usr/bin/env python3
"""
a class Encoder that inherits from tensorflow.keras.layers.Layer
to create the encoder for a transformer
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Class methods
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class Constructor
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N,
                               dm,
                               h,
                               hidden,
                               input_vocab,
                               max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N,
                               dm,
                               h,
                               hidden,
                               target_vocab,
                               max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, x, encoder_output,
             training,
             look_ahead_mask,
             padding_mask):
        """
        Returns:
            a tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        """
        enc_out = self.encoder(inputs, training, encoder_mask)
        dec_out = self.decoder(target, enc_out, training, look_ahead_mask,
                               decoder_mask)
        return self.linear(dec_out)
