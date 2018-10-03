import tensorflow as tf
tfgan = tf.contrib.gan

import layers as ly


class SAGAN:
    def __init__(self, hparams):
        self.generator = Generator(hparams)
        self.discriminator = Discriminator(hparams)
        self.global_step = tf.train.get_or_create_global_step()

    def build(self, gen_inputs, reals):
        model = tfgan.gan_model(self.generator.build,
                                self.discriminator.build,
                                reals,
                                gen_inputs)

        self.gen_data = self._denorm(model.generated_data)

        self.gen_loss, self.dis_loss = self._build_loss(model)

        self.gen_train, self.dis_train = self._build_train_ops(self.gen_loss,
                                                               self.dis_loss,
                                                               self.global_step)

        self.summaries = tf.summary.merge_all()

    @staticmethod
    def _denorm(img):
        with tf.name_scope('denorm'):
            img -= tf.reduce_min(img, [1, 2], keepdims=True)
            img /= tf.reduce_max(img, [1, 2], keepdims=True)
            img *= 255
            img = tf.cast(img, tf.uint8)

        return img

    @staticmethod
    def _build_loss(model):
        with tf.name_scope('hinge_loss'):
            real_loss = tf.minimum(0.0, model.discriminator_real_outputs - 1.0)
            fake_loss = tf.minimum(0.0, -model.discriminator_gen_outputs - 1.0)
            dis_loss = tf.reduce_mean(-real_loss - fake_loss)

            gen_loss = tf.reduce_mean(-model.discriminator_gen_outputs)

            tf.summary.scalar('dis_real_loss', tf.reduce_mean(real_loss))
            tf.summary.scalar('dis_fake_loss', tf.reduce_mean(fake_loss))
            tf.summary.scalar('dis_loss', dis_loss)
            tf.summary.scalar('gen_loss', gen_loss)

        return gen_loss, dis_loss

    @staticmethod
    def _build_train_ops(gen_loss, dis_loss, global_step):
        with tf.name_scope('train_ops'):
            gen_optim = tf.train.AdamOptimizer(learning_rate=0.0001,
                                               beta1=0.0,
                                               beta2=0.9)
            dis_optim = tf.train.AdamOptimizer(learning_rate=0.0004,
                                               beta1=0.0,
                                               beta2=0.9)

            gen_vars = tf.trainable_variables('Generator')
            dis_vars = tf.trainable_variables('Discriminator')

            gen_train = gen_optim.minimize(gen_loss,
                                           global_step=global_step,
                                           var_list=gen_vars)
            dis_train = dis_optim.minimize(dis_loss,
                                           var_list=dis_vars)

        return gen_train, dis_train

    def train_step(self, sess):
        sess.run(self.dis_train)
        _, summaries, step = sess.run([self.gen_train,
                                       self.summaries,
                                       self.global_step])

        return summaries, step

    def generate(self, sess):
        return sess.run(self.gen_data)


class Generator:
    def __init__(self, hparams):
        self.num_layers = hparams.g_num_layers
        self.kernel_size = hparams.g_kernel_size
        self.filter_base = hparams.g_filter_base
        self.activation = hparams.g_activation
        self.norm = hparams.g_norm
        self.num_channels = hparams.num_channels

    def build(self, gen_inputs):
        initializer = None
        num_filters = self.filter_base * (2 ** self.num_layers)
        
        gen_inputs = tf.reshape(gen_inputs, [gen_inputs.shape[0], 4, 4, -1])
        layers = [tf.layers.conv2d(gen_inputs,
                                   kernel_size=1,
                                   filters=num_filters,
                                   padding='same',
                                   name='project')]
        
        for i in range(self.num_layers - 2):
            layers.append(ly.up_conv(layers[-1],
                                     kernel_size=self.kernel_size,
                                     filters=num_filters // (2 ** (i + 1)),
                                     activation=self.activation,
                                     norm=self.norm,
                                     initializer=initializer,
                                     name='up_conv_%d' % i))
            
        layers.append(ly.self_attention(layers[-1], 8, 'self_attention_0'))
        
        for i in range(self.num_layers - 2, self.num_layers):
            layers.append(ly.up_conv(layers[-1],
                                     kernel_size=self.kernel_size,
                                     filters=num_filters // (2 ** (i + 1)),
                                     activation=self.activation,
                                     norm=self.norm,
                                     initializer=initializer,
                                     name='up_conv_%d' % i))

        layers.append(ly.self_attention(layers[-1], 8, 'self_attention_1'))

        return tf.layers.conv2d(layers[-1],
                                kernel_size=1,
                                filters=self.num_channels,
                                activation=tf.nn.tanh,
                                padding='same',
                                name='output')


class Discriminator:
    def __init__(self, hparams):
        self.num_layers = hparams.d_num_layers
        self.kernel_size = hparams.d_kernel_size
        self.filter_base = hparams.d_filter_base
        self.activation = hparams.d_activation
        self.norm = hparams.d_norm

    def build(self, gen_inputs, reals):
        initializer = None

        layers = [tf.layers.conv2d(gen_inputs,
                                   kernel_size=self.kernel_size,
                                   filters=self.filter_base,
                                   padding='same',
                                   name='conv_0')]
        
        for i in range(self.num_layers - 2):
            layers.append(ly.down_conv(layers[-1],
                                       kernel_size=self.kernel_size,
                                       filters=self.filter_base ** (i + 1),
                                       activation=self.activation,
                                       norm=self.norm,
                                       initializer=initializer,
                                       name='down_conv_%d' % i))

        layers.append(ly.self_attention(layers[-1], 8, 'self_attention_0'))
        
        for i in range(self.num_layers - 2, self.num_layers):
            layers.append(ly.down_conv(layers[-1],
                                       kernel_size=self.kernel_size,
                                       filters=self.filter_base ** (i + 1),
                                       activation=self.activation,
                                       norm=self.norm,
                                       initializer=initializer,
                                       name='down_conv_%d' % i))

        layers.append(ly.self_attention(layers[-1], 8, 'self_attention_1'))
            
        return tf.layers.conv2d(layers[-1],
                                kernel_size=1,
                                filters=1,
                                padding='same',
                                name='output')
