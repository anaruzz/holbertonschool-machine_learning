#!/usr/bin/env python3
"""
Script that calculates the unigram BLEU score for a sentence
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Returns: the unigram BLEU score
    """
    count = 0
    ref = set([word for ref in references for word in ref])
    for word in ref:
        if word in sentence:
            count += 1
    precision = count / ls

    best_match = None
    ls = len(sentence)
    for ref in references:
        if best_match is None:
            best_match = ref
        lm = len(best_match)
        best_diff = abs(lm - ls)
        if abs(len(ref) - ls) < best_diff:
            best_match = ref

    if ls > lm:
        bp = 1
    else:
        bp = np.exp(1 - lm / ls)

    Bleu_score = bp * precision
    return Bleu_score
