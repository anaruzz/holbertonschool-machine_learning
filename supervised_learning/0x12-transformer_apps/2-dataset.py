#!/usr/bin/env python3
"""
class Dataset that loads and preps a dataset for
machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Class methods
    """
    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        a, b = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = b
        self.tokenizer_pt = a

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        """
        pp = []
        ee = []
        for pt, en in data:
            pp.append(pt.numpy())
            ee.append(en.numpy())
        tokeniz_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pp, target_vocab_size=2**15)
        tokeniz_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            ee, target_vocab_size=2**15)
        return tokeniz_pt, tokeniz_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        """
        tok_pt = [self.tokenizer_pt.vocab_size]+self.tokenizer_pt.encode(
            pt.numpy())+[(self.tokenizer_pt.vocab_size) + 1]
        tok_en = [self.tokenizer_en.vocab_size]+self.tokenizer_en.encode(
            en.numpy())+[(self.tokenizer_en.vocab_size) + 1]
        return tok_pt, tok_en

    def tf_encode(self, pt, en):
        """
        Method that acts as a tensorflow wrapper for the encode instance
        method
        """
        p, e = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        p.set_shape([None]), e.set_shape([None])

        return p, e
