"""Partially learned gradient descent scheme for head data.

For legal reasons, we are sadly not able to provide the training data to the
public, but we are working on providing an open dataset for validation.
"""

import tensorflow as tf
import numpy as np
import odl
from util import random_phantom, conv2d
import os

sess = tf.InteractiveSession()

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting='const')

# Tomography
# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
# Detector: uniformly sampled, n = 1000, min = -360, max = 360
detector_partition = odl.uniform_partition(-360, 360, 1000)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)


operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint,
                                                                  'RayTransformAdjoint')

partial0 = odl.PartialDerivative(space, axis=0)
partial1 = odl.PartialDerivative(space, axis=1)
odl_op_regularizer = odl.contrib.tensorflow.as_tensorflow_layer(partial0.adjoint * partial0 +
                                                                partial1.adjoint * partial1,
                                                               'Regularizer')

# User selected paramters
n_data = 1
n_memory = 5
n_iter = 10
mu_water = 0.02
photons_per_pixel = 10000
epsilon = 1.0 / photons_per_pixel

# Helper functions to load data from disk, please excuse the ugly code.
global f
f = []
folder = '.'  # INSERT DATA FOLDER
validation_data_path = folder + 'data.npy'  # INSERT VALIDATION DATA


def next_file():
    """Get a random file from the data folder."""
    global f
    if not f:
        f = []
        for (dirpath, dirnames, filenames) in os.walk(folder):
            f.extend([os.path.join(folder, fi) for fi in filenames])

        f = np.random.permutation(f).tolist()

    return f.pop()


def generate_data(validation=False):
    """Generate a set of random data."""
    n_iter = 1 if validation else n_data

    x_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_iter):
        if validation:
            data = np.load(validation_data_path)
        else:
            data = np.load(next_file())

        phantom = space.element(np.rot90(data, -1))

        # convert go g/cm^3, data should now be normalizes around 1.0
        phantom /= 1000.0

        # Apply nonlinear operator to phantom
        data = operator(phantom)
        data = np.exp(-data * mu_water)

        # Add poisson noise
        noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel

        fbp = pseudoinverse(-np.log(noisy_data + epsilon) / mu_water)

        x_arr[i, ..., 0] = fbp
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr


with tf.name_scope('placeholders'):
    x_0 = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y")

    s = tf.fill([tf.shape(x_0)[0], size, size, n_memory], np.float32(0.0), name="s")


with tf.name_scope('variable_definitions'):
    if 0:
        w1 = tf.get_variable("w1", shape=[3, 3, n_memory + 3, 32],
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32))
        b1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 32]), name='b1')

        w2 = tf.get_variable("w2", shape=[3, 3, 32, 32],
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32))
        b2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 32]), name='b2')

        w3 = tf.get_variable("w3", shape=[3, 3, 32, n_memory + 1],
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)) * 0.1
        b3 = tf.Variable(tf.constant(0.00, shape=[1, 1, 1, n_memory + 1]), name='b3')
    else:
        # If trained network is available, re-use as starting guess
        ld = np.load('partially_learned_gradient_descent_head_parameters.npz')

        w1 = tf.Variable(tf.constant(ld['w1']), name='w1')
        b1 = tf.Variable(tf.constant(ld['b1']), name='b1')

        w2 = tf.Variable(tf.constant(ld['w2']), name='w2')
        b2 = tf.Variable(tf.constant(ld['b2']), name='b2')

        w3 = tf.Variable(tf.constant(ld['w3']), name='w3')
        b3 = tf.Variable(tf.constant(ld['b3']), name='b3')


# Implementation of the iterative scheme
x_values = [x_0]
x = x_0
for i in range(n_iter):
    with tf.name_scope('iterate_{}'.format(i)):
        nonlinear_Tx = tf.exp(-mu_water * odl_op_layer(x))
        gradx = mu_water * odl_op_layer_adjoint(y - nonlinear_Tx)
        gradreg = odl_op_regularizer(x)

        update = tf.concat([x, gradx, gradreg, s], axis=3)

        update = tf.nn.relu(conv2d(update, w1) + b1)
        update = tf.nn.relu(conv2d(update, w2) + b2)

        update = conv2d(update, w3) + b3

        s = tf.nn.relu(update[..., 1:])
        dx = update[..., 0:1]

        x = x + dx
        x_values.append(x)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum((x - x_true) ** 2, axis=(1, 2)))


with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-3
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate,
                                                global_step=global_step,
                                                decay_rate=1.0,
                                                decay_steps=500,
                                                staircase=True,
                                                name='learning_rate')

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)


# Initialize all TF variables
tf.global_variables_initializer().run()

# Solve with an ODL callback to see what happens in real time
callback = odl.solvers.CallbackShow(clim=[0.1, 0.4])

# Generate validation data
x_arr_validate, y_arr_validate, x_true_arr_validate = generate_data(validation=True)

if 0:
    # Train the network
    n_train = 100000
    for i in range(0, n_train):
        x_arr, y_arr, x_true_arr = generate_data()

        _, loss_training = sess.run([optimizer, loss],
                                  feed_dict={x_0: x_arr,
                                             x_true: x_true_arr,
                                             y: y_arr})

        # Validate on shepp-logan
        x_values_result, loss_result = sess.run([x_values, loss],
                       feed_dict={x_0: x_arr_validate,
                                  x_true: x_true_arr_validate,
                                  y: y_arr_validate})

        print('iter={}, validation loss={}'.format(i, loss_result))

        callback((space ** (n_iter + 1)).element(x_values_result))
else:
    # Validate on shepp-logan
    x_values_result, loss_result = sess.run([x_values, loss],
                   feed_dict={x_0: x_arr_validate,
                              x_true: x_true_arr_validate,
                              y: y_arr_validate})

    print('validation loss={}'.format(loss_result))

    callback((space ** (n_iter + 1)).element(x_values_result))
