#!/usr/bin/env python3
import tensorflow as tf

from ganrecs.network.utils import NetworkLayer

class Discriminator():
    """
    Constructs a discriminator network for
    an adversarial context
    """

    def __init__(self, arch, input_real, input_fake, input_real_p):
        if len(arch) < 2:
            raise ValueError("Must provide architecture of at least one layer")
        self._layers = self._construct(arch)
        past = input_real
        for i in range(len(self._layers) - 1):
            inter = tf.nn.relu(tf.matmul(past, self._layers[i].W) + self._layers[i].b)
            past = inter
        self.logit_real = tf.matmul(inter, self._layers[-1].W) + self._layers[-1].b
        self.prob_real = tf.nn.sigmoid(self.logit_real)
        past = input_fake
        for i in range(len(self._layers) - 1):
            inter = tf.nn.relu(tf.matmul(past, self._layers[i].W) + self._layers[i].b)
            past = inter
        self.logit_fake = tf.matmul(inter, self._layers[-1].W) + self._layers[-1].b
        self.prob_fake = tf.nn.sigmoid(self.logit_fake)
        # past = input_real_p
        # for i in range(len(self._layers) - 1):
        #     inter = tf.nn.relu(tf.matmul(past, self._layers[i].W) + self._layers[i].b)
        #     past = inter
        # self.logit_real_p = tf.matmul(inter, self._layers[-1].W) + self._layers[-1].b
        # self.prob_real_p = tf.nn.sigmoid(self.logit_real_p)

    def _construct(self, arch):
        layers = []
        for i in range(len(arch)-1):
            new_layer = NetworkLayer(arch[i], arch[i+1])
            layers.append(new_layer)
        return layers
    
    def _build_dis(self, some_input):
        past = some_input
        for i in range(len(self._layers) - 1):
            inter = tf.nn.relu(tf.matmul(past, self._layers[i].W) + self._layers[i].b)
            past = inter
        logit = tf.matmul(inter, self._layers[-1].W) + self._layers[-1].b
        return tf.nn.sigmoid(logit)

    def get_var_list(self):
        weights = []
        biases = []
        for l in self._layers:
            weights.append(l.W)
            biases.append(l.b)
        return weights + biases
