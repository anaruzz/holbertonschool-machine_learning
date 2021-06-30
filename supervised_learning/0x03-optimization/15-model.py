#!/usr/bin/env python3
"""
Script that builds, trains, and saves a neural network
model in tensorflow using Adam optimization,mini-batch
gradient descent, learning rate decay, and batch normalization
"""
import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data
update_variables_Adam = __import__('9-Adam').update_variables_Adam


def create_placeholders(nx, classes):
    """
    create x and y placeholders
    """
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, nx], name='y')
    return x, y

def create_layer(prev, n, activation):
    """
    Create custom layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=kernel,
                            activation=activation,
                            name="layer")
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    returns a tensor of the activated output for the layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=kernel,
                            activation=activation,
                            name="layer")(prev)
    mean, var = tf.nn.moments(layer, axes=[0])
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))
    epsilon = 1e-8
    z = tf.nn.batch_normalization(layer,
                                  mean,
                                  var,
                                  beta,
                                  gamma,
                                  epsilon)
    return activation(z)

def shuffle_data(X, Y):
    """
    returns shuffled X and Y matrices
    """
    m = X.shape[0]
    p = np.random.permutation(m)
    return X[p], Y[p]


def forward_prop(x, layer_sizes, activations):
    """
    returns prediction after forward propagation
    """
    for layer, activation in zip(layer_sizes, activations):
        if activation is None:
            y = create_layer(x, layer, activation)
            x = y
        else:
            y = create_batch_norm_layer(x, layer, activation)
            x = y
    return y


def calculate_accuracy(y, y_pred):
    """
    returns the accuracy of the prediction
    """
    prediction = tf.argmax(y_pred, 1)
    correct = tf.argmax(y_pred, 1)
    equality = tf.equal(prediction, correct)
    accuracy = tf.math.reduce_mean(tf.cast(equality, "float"))
    return accuracy


def calculate_loss(y, y_pred):
    """
    returns the loff function
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)





def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    returns a tensor of the activated output for the layer
    """
    init = tf.global_variables_initializer()
    m = len(Data_train)
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
            train_loss, train_accuracy = sess.run((loss, accuracy),
                                                  {x: X_train, y: Y_train})
            valid_loss, valid_accuracy = sess.run((loss, accuracy),
                                                  {x: X_valid, y: Y_valid})
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
                                                          {x: X_batch,
                                                           y: Y_batch})
                    if (b + 1) % 100 == 0 and b > 0:
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
        return saver.save(sess, save_path)



    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=kernel)(prev)
    mean, var = tf.nn.moments(layer, axes=[0])
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))
    epsilon = 1e-8
    z = tf.nn.batch_normalization(layer,
                                  mean,
                                  var,
                                  beta,
                                  gamma,
                                  epsilon)
    return activation(z)
