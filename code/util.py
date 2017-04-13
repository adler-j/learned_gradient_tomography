"""Utilities needed for the networks."""

import numpy as np
import tensorflow as tf
import odl

def random_ellipse():
    return ((np.random.rand() - 0.3) * np.random.exponential(0.3),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            np.random.rand() - 0.5, np.random.rand() - 0.5,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc):
    n = np.random.poisson(100)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)


def conv2d(x, W, stride=(1, 1)):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')
