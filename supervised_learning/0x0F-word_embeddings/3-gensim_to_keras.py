#!/usr/bin/env python3
"""
Script hat converts a gensim word2vec model to a
keras Embedding layer
"""
# import tensorflow.keras as K


def gensim_to_keras(model):
    """
    Returns: the trainable keras Embedding
    """
    k_embedding = model.wv.get_keras_embedding(train_embeddings=True)
    return k_embedding
