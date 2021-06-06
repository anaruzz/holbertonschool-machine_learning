#!/usr/bin/env python3
"""
Script that creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Returns: embeddings, features

    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
        s is the number of sentences in sentences
        f is the number of features analyzed
    features is a list of the features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences)
    embeddings = embeddings.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
