import tensorflow as tf


def down_conv(inputs, kernel_size, filters, activation=None, norm=None, initializer=None, name='down_conv'):
    with tf.variable_scope(name):
        layer = tf.layers.conv2d(inputs,
                                 kernel_size=kernel_size,
                                 filters=filters,
                                 strides=2,
                                 kernel_initializer=initializer)

        if norm is not None:
            layer = norm(layer)

        if activation is not None:
            layer = activation(layer)

    return layer


def up_conv(inputs, kernel_size, filters, activation=None, norm=None, initializer=None, name='up_conv'):
    with tf.variable_scope(name):
        layer = tf.image.resize_nearest_neighbor(inputs, [inputs.shape[1].value * 2] * 2)

        layer = tf.layers.conv2d(layer,
                                 kernel_size=kernel_size,
                                 filters=filters,
                                 padding='same',
                                 kernel_initializer=initializer)

        if norm is not None:
            layer = norm(layer)

        if activation is not None:
            layer = activation(layer)

    return layer


def flatten(inputs):
    return tf.reshape(inputs, [inputs.shape[0], -1, inputs.shape[-1]], name='flatten')


def self_attention(inputs, channel_factor=8, name='self_attention'):
    num_filters = inputs.shape[-1].value // channel_factor
    with tf.variable_scope(name):
        flat_inputs = flatten(inputs)

        f = tf.layers.conv1d(flat_inputs,
                             kernel_size=1,
                             filters=num_filters)
        g = tf.layers.conv1d(flat_inputs,
                             kernel_size=1,
                             filters=num_filters)
        h = tf.layers.conv1d(flat_inputs,
                             kernel_size=1,
                             filters=inputs.shape[-1])

        beta = tf.nn.softmax(tf.matmul(f, g, transpose_b=True))
        o = tf.matmul(beta, h)

        gamma = tf.get_variable('gamma', [], initializer=tf.zeros_initializer)
        y = gamma * o + flat_inputs
        y = tf.reshape(y, inputs.shape)

    return y
