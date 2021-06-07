#!/usr/bin/env python3
"""
Script that creates and trains a genism fastText model
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100,
                   min_count=5, negative=5,
                   window=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    Returns: the trained model
    """
    model = FastText(sentences,
                     size=size,
                     min_count=min_count,
                     negative=negative,
                     window=window,
                     sg=not cbow,
                     iter=iterations,
                     seed=seed,
                     workers=workers
                     )
    return model
