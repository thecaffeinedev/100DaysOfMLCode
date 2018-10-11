#!/usr/bin/env python3
import tensorflow as tf

def xavier(size):
    """
    Use Xavier Weight Initialization for 
    warm start
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class NetworkLayer():
    """
    Provides a container for a single network layer
    """

    def __init__(self, _in, _out):
        self.W = tf.Variable(xavier([_in, _out]))
        self.b = tf.Variable(tf.zeros(shape=[_out]))
