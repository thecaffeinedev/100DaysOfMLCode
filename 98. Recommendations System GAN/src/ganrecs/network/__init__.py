#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from ganrecs.network.discriminator import Discriminator

from ganrecs.network.generator import Generator

class GAN():

    def __init__(self, d, d_optimizer, d_net, d_loss, g, cond, g_optimizer, g_net, g_loss, keep_prob, pert):
        self.discriminator_input = d
        self.discriminator_optimizer = d_optimizer
        self.generator_input = g
        self.generator_optimizer = g_optimizer
        self.discriminator = d_net
        self.generator = g_net
        self.discriminator_loss = d_loss
        self.generator_loss = g_loss
        self.generator_condition = cond
        self.keep_prob = keep_prob,
        self.pert = pert

def gan(dis_arch, gen_arch, conditional, batch_size):
    Z = tf.placeholder(tf.float32, shape=[None, gen_arch[0]], name='noise')
    Y = tf.placeholder(tf.float32, shape=[None, conditional], name='conditional')
    X = tf.placeholder(tf.float32, shape=[None, dis_arch[0]])
    X = X + tf.random_normal(shape=tf.shape(X), mean=0., stddev=1., dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    I = tf.concat(values=[Z, Y], axis=1)
    J = tf.concat(values=[X, Y], axis=1)
    gen_arch[0] += conditional
    dis_arch[0] += conditional
    g = Generator(gen_arch, I, keep_prob)
    F = tf.concat(values=[g.prob, Y], axis=1)
    d = Discriminator(dis_arch, J, F, None)
    d_real_labels = tf.ones_like(d.logit_real)
    d_fake_labels = tf.zeros_like(d.logit_fake)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_real, labels=d_real_labels))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=d_fake_labels))
    d_total_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=tf.ones_like(d.logit_fake)))    

    d_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(d_total_loss, var_list=d.get_var_list())
    g_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g.get_var_list())
    return GAN(X, d_opt, d, d_total_loss, Z, Y, g_opt, g, g_loss, keep_prob, None)


def dragan(dis_arch, gen_arch, conditional, batch_size):
    Z = tf.placeholder(tf.float32, shape=[None, gen_arch[0]], name='noise')
    Y = tf.placeholder(tf.float32, shape=[None, conditional], name='conditional')
    X = tf.placeholder(tf.float32, shape=[None, dis_arch[0]])
    X = X + tf.random_normal(shape=tf.shape(X), mean=0., stddev=1., dtype=tf.float32)
    X_p = tf.placeholder(tf.float32, shape=[None, dis_arch[0]])
    keep_prob = tf.placeholder(tf.float32)
    I = tf.concat(values=[Z, Y], axis=1)
    J = tf.concat(values=[X, Y], axis=1)
    J_p = tf.concat(values=[X_p, Y], axis=1)
    gen_arch[0] += conditional
    dis_arch[0] += conditional
    g = Generator(gen_arch, I, keep_prob)
    F = tf.concat(values=[g.prob, Y], axis=1)
    d = Discriminator(dis_arch, J, F, J_p)
    d_real_labels = tf.ones_like(d.logit_real)
    d_fake_labels = tf.zeros_like(d.logit_fake)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_real, labels=d_real_labels))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=d_fake_labels))
    d_total_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d.logit_fake, labels=tf.ones_like(d.logit_fake)))    

    # Apply gradient penalty as here: https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    lambd = 10
    alpha = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
    differences = J_p - J
    interpolates = J + (alpha*differences)
    gradients = tf.gradients(d._build_dis(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    d_total_loss += lambd*gradient_penalty

    d_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(d_total_loss, var_list=d.get_var_list())
    g_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g.get_var_list())
    return GAN(X, d_opt, d, d_total_loss, Z, Y, g_opt, g, g_loss, keep_prob, X_p)
