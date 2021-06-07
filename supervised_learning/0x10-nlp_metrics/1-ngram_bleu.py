#!/usr/bin/env python3
"""
Script that calculates the n-gram BLEU score for a sentence
"""
import numpy as np


def n_gram(sentence, n):
    """
    Tokenize sentence into grams
    """
    s = []
    for i in range(len(sentence) - n + 1):
        n_gram = ""
        for j in range(n):
            n_gram += sentence[i + j]
            if not j + 1 == n:
                n_gram += " "
        s.append(n_gram)
    return s


def ngram_bleu(references, sentence, n):
    """
    Returns: the n-gram BLEU score
    """
    r_list = [len(r) for r in references]
    ls = len(sentence)

    references = list(map(lambda ref: n_gram(ref, n), references))

    fl_ref = set()
    for ref in references:
        for gram in ref:
            fl_ref.add(gram)

    sentence = n_gram(sentence, n)
    count = 0
    for gram in fl_ref:
        if gram in sentence:
            count += 1
    precision = count / len(sentence)

    best_match = None
    for i, ref in enumerate(references):
        if best_match is None:
            index = i
            best_match = ref

        best_diff = len(best_match) - len(sentence)
        if abs(len(ref) - len(sentence)) < best_diff:
            index = i
            best_match = ref

    r = r_list[index]
    if ls > r:
        bp = 1
    else:
        bp = np.exp(1 - r / ls)

    Bleu_score = bp * precision
    return Bleu_score
