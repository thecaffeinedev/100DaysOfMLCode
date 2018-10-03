import tensorflow as tf
import numpy as np
from PIL import Image
import os

import sagan
import inputs


HPARAMS = tf.contrib.training.HParams(max_iter=1000,
                                      export_freq=100,
                                      batch_size=16,
                                      g_num_layers=3,
                                      g_kernel_size=3,
                                      g_filter_base=8,
                                      g_activation=tf.nn.leaky_relu,
                                      g_norm=tf.contrib.layers.instance_norm,
                                      d_num_layers=3,
                                      d_kernel_size=3,
                                      d_filter_base=8,
                                      d_activation=tf.nn.leaky_relu,
                                      d_norm=tf.contrib.layers.instance_norm,
                                      num_channels=3,
                                      resolution=32)


def train(hparams):
    if not os.path.exists('./summary'):
        os.mkdir('./summary')

    with tf.device('/cpu:0'):
        noise = tf.random_normal([hparams.batch_size, 16])
        reals = inputs.ImageInputs('./data/data.tfrecord',
                                   hparams.batch_size,
                                   hparams.resolution).get_next()

    with tf.name_scope('GAN'):
        gan = sagan.SAGAN(hparams)
        gan.build(noise, reals)

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir='./summary',
                                               graph=tf.get_default_graph())

        init_ops = tf.group([tf.global_variables_initializer(),
                             tf.local_variables_initializer()])
        sess.run(init_ops)

        for i in range(hparams.max_iter):
            summaries, step = gan.train_step(sess)
            summary_writer.add_summary(summaries, step)
            tf.logging.log_every_n(tf.logging.INFO,
                                   'Training step %d' % step,
                                   10)

            if step % hparams.export_freq == 0:
                imgs = gan.generate(sess)
                imgs = np.split(imgs, hparams.batch_size)
                for i, img in enumerate(imgs):
                    img = img.squeeze()
                    img = Image.fromarray(img)
                    img.save('./summary/img_%d_%d.png' % (step, i))


if __name__ == '__main__':
    train(HPARAMS)
