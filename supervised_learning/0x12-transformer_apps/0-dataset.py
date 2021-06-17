#!/usr/bin/env python3
"""
class Dataset that loads and preps a dataset for
machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    class methods
    """
    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        portuguese, english = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = english
        self.tokenizer_pt = portuguese

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        Returns: tokenizer_pt, tokenizer_en
        """
        prt = []
        eng = []
        for pt, en in data:
            prt.append(pt.numpy())
            eng.append(en.numpy())
        tokeniz_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            prt, target_vocab_size=2**15)
        tokeniz_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            eng, target_vocab_size=2**15)
        return tokeniz_pt, tokeniz_en
