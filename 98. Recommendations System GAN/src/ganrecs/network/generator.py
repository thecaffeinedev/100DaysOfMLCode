#!/usr/bin/env python3
import tensorflow as tf

from ganrecs.network.utils import NetworkLayer

class Generator():
    """
    Constructs a generator network for
    an adversarial context
    """

    def __init__(self, arch, _input, keep_prob):
        if len(arch) < 2:
            raise ValueError("Must provide architecture of at least one layer")
        self._layers = self._construct(arch)
        past = _input
        for i in range(len(self._layers) - 1):
            inter = tf.nn.relu(tf.matmul(past, self._layers[i].W) + self._layers[i].b)
            #dropout = tf.nn.dropout(inter, keep_prob)
            past = inter
        self.logit = tf.matmul(inter, self._layers[-1].W) + self._layers[-1].b
        self.prob = tf.nn.sigmoid(self.logit)

    def _construct(self, arch):
        layers = []
        for i in range(len(arch)-1):
            new_layer = NetworkLayer(arch[i], arch[i+1])
            layers.append(new_layer)
        return layers

    def get_var_list(self):
        weights = []
        biases = []
        for l in self._layers:
            weights.append(l.W)
            biases.append(l.b)
        return weights + biases
