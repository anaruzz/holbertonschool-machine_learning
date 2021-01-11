#!/usr/bin/env python3
"""
Script that Evaluates the output of a neural network
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluate the output of a NN
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)
    y_pred = tf.get_collection('y_pred', scope=None)[0]
    loss = tf.get_collection('loss', scope=None)[0]
    accuracy = tf.get_collection('accuracy', scope=None)[0]
    x = tf.get_collection('x', scope=None)[0]
    y = tf.get_collection('y', scope=None)[0]
    y_pred, accuracy, loss = sess.run((y_pred, accuracy, loss), feed_dict={x: X, y: Y})
    return y_pred, accuracy, loss
