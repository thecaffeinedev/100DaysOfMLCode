#!/usr/bin/env python3
# This is an adaptation of an online tutorial
# on Generative Adversarial Networks to test 
# that the customized construction constructs 
# correctly
# Original code: https://github.com/wiseodd/generative-models
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

from ganrecs.network import gan

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def get_one_hot(value):
    zeros = [0 for _ in range(10)]
    zeros[value] = 1
    return zeros

def process_args(args=None):
    parser = argparse.ArgumentParser(description="Test with MNIST data set")
    parser.add_argument('-l', '--location', help='Saved model location')
    args = parser.parse_args(args)
    location = os.path.expanduser(args.location)

    if not os.path.exists(location):
        os.makedirs(location)

    return location

def plot_losses(epochs, d_losses, g_losses):
    xs = [x for x in range(epochs)]
    plt.title('D/G Losses Over Time')
    plt.plot(xs, d_losses, label='Discriminator')
    plt.plot(xs, g_losses, label='Generator')
    plt.legend()
    plt.show()

def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

def main(args=None):
    location = process_args(args)
    model_path = os.path.join(location, "model.ckpt")

    print("Constructing network...")
    dis_arch = [784, 128, 1]
    gen_arch = [100, 128, 784]
    network = gan(dis_arch, gen_arch, 10, 128)

    saver = tf.train.Saver()

    print("Reading input data...")
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    d_losses = []
    g_losses = []
    session = tf.Session()
    if os.path.exists(model_path + ".meta"):
        print("Restoring model....")
        saver.restore(session, model_path)
    else:
        session.run(tf.global_variables_initializer())

        print("Starting run...")
        i = 0
        for it in range(500000):

            X_mb, digits = mnist.train.next_batch(128)
            _sample = sample_Z(128, 100)
            X_p = get_perturbed_batch(X_mb)
            _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss], feed_dict={network.discriminator_input: X_mb, network.pert: X_p, network.generator_input: _sample, network.generator_condition: digits})
            _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss], feed_dict={network.generator_input: _sample, network.generator_condition: digits})

            if it % 1000 == 0:
                print('current_d_loss: {:.4}'.format(D_loss_curr))
                print('current_g_loss: {:.4}'.format(G_loss_curr))
                d_losses.append(D_loss_curr)
                g_losses.append(G_loss_curr)
        
        plot_losses(500, d_losses, g_losses)


    while True:
        val = int(input("Input a number 0-9 to generate: "))
        val = get_one_hot(val)
        _sample = sample_Z(1,100)
        result = session.run(network.generator.prob, feed_dict={network.generator_input: _sample, network.generator_condition: [val]})
        fig = plot(result)
        fig.show()

if __name__ == '__main__':
    main(args)
