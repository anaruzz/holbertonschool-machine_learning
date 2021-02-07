#!/usr/bin/env python3
"""
Script that  builds a modified version of
the LeNet-5 architecture using tensorflow
"""

import tensorflow as tf


def lenet5(x, y):
    """
    Returns:

    a tensor for the softmax activated output
    a training operation that utilizes Adam
    optimization (with default hyperparameters)
    a tensor for the loss of the netowrk
    a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=init)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=init)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)

    flt = tf.layers.Flatten()(pool2)

    f1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                         kernel_initializer=init)(flt)

    f2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                         kernel_initializer=init)(f1)

    y_pred = tf.layers.Dense(units=10,
                             kernel_initializer=init)(f2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    softmax = tf.nn.softmax(y_pred)
    arg1 = tf.math.argmax(y, 1)
    arg2 = tf.math.argmax(y_pred, 1)
    eq = tf.math.equal(arg1, arg2)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    cast = tf.cast(eq, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)

    return (softmax, train_op, loss, accuracy)
