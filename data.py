import numpy as np
import tensorflow as tf


def data_train(num_train, num_init, num_bc, minxt, maxxt):
    """
    Generate the data needed for training the physics-informed
    neural network.

    :param num_train: number of collocation points
    :param num_init: number of points at t = 0
    :param num_bc: number of points at the two boundaries
    :param minxt: minimum of x and t
    :param maxxt: maximum of x and t
    :return: the training set, initial condition data,
             and boundary conditions
    """
    # training set
    train_data = tf.random.uniform((num_train, 2), minxt, maxxt)
    # training data at t = 0
    x_init = tf.random.uniform((num_init, 1), minxt[0], maxxt[0])
    init_data = tf.concat([x_init, tf.zeros_like(x_init)], axis=1)
    # training data at two boundaries
    t_bc = tf.random.uniform((num_bc, 1), minxt[1], maxxt[1])
    bc1_data = tf.concat([tf.zeros_like(t_bc), t_bc], axis=1)
    bc2_data = tf.concat([tf.ones_like(t_bc) * 2 * np.pi, t_bc], axis=1)
    return train_data, init_data, bc1_data, bc2_data


def data_test(num_test, minxt, maxxt):
    data = tf.linspace(minxt, maxxt, num_test)
    return data
