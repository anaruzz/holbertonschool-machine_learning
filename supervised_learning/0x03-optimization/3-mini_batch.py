#!/usr/bin/env python3
"""
that trains a loaded neural network model using
mini-batch gradient descent
"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid,
                    Y_valid, batch_size=32,
                    epochs=5, load_path="/tmp/model.ckpt",
                    save_path="/tmp/model.ckpt"):
    """
    Train a neural network using mini batch GD
    """
    init = tf.global_variables_initializer()
    m = X_train.shape[0]
    batches = m / batch_size
    if batches % 1 != 0:
        batches = int(batches + 1)
    else:
        batches = int(batches)
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        for ep in range(epochs+1):
            X, Y = shuffle_data(X_train, Y_train)
            train_loss, train_accuracy = sess.run((accuracy, loss), {x: X_train, y: Y_train})
            valid_loss, valid_accuracy = sess.run((accuracy, loss), {x: X_valid, y: Y_valid})
            print("After {} epochs:".format(ep))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_loss))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            if ep < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for b in range(batches):
                    start = b * batch_size
                    end = start + batch_size
                    if end > m:
                        end = m
                    X_batch = X_shuffled[start:end]
                    Y_batch = Y_shuffled[start:end]
                    sess.run((train_op), {x: X_batch,
                                             y: Y_batch})
                    batch_cost, batch_accuracy = sess.run((loss, accuracy),
                                                            {x: X_batch, y: Y_batch})
                    if (b + 1) % 100 == 0 and b > 0:
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
        return saver.save(sess, save_path)
