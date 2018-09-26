import tensorflow as tf
import numpy as np

def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]

class RBM(object):
    def __init__(self, num_visble, num_hidden):
        self.dim = (num_visble, num_hidden)
        self.num_visble = num_visble
        self.num_hidden = num_hidden
        self.create_placeholder()
        self.add_model()
        self.optimizer = self.train(self.input)

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_visble],
                                    name="input")
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_visble],
                                    name="mask")
    
    def add_model(self):
        with tf.variable_scope("model"):
            self.weights = tf.get_variable(name="weights", shape=self.dim,
                                           dtype=tf.float32)
            self.hbias = tf.get_variable(name="hbias", shape=[self.num_hidden],
                                         dtype=tf.float32)
            self.vbias = tf.get_variable(name="vbias", shape=[self.num_visble],
                                         dtype=tf.float32)
            self.prev_gw = tf.get_variable(name="prev_weights", shape=self.dim,
                                           dtype=tf.float32)
            self.prev_gbh = tf.get_variable(name="prev_hbias", shape=[self.num_hidden],
                                         dtype=tf.float32)
            self.prev_gbv = tf.get_variable(name="prev_vbias", shape=[self.num_visble],
                                         dtype=tf.float32)

    def sample_hidden(self, vis):
        activations = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hbias)
        h1_sample = tf.nn.relu(tf.sign(activations - tf.random_uniform(tf.shape(activations))))
        return h1_sample, activations

    def sample_visible(self, hid, k=5):
        activations = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(self.weights)) + self.vbias)
        k_ones = tf.ones((1,k))
        partition = tf.expand_dims(tf.reduce_sum(tf.reshape(activations, (-1, self.num_visble // k, k)), 2), -1)
        #part = [tf.squeeze(x,[1]) for x in tf.split(1, self.num_visble // k, partition)]
        #print(part[0].get_shape())
        partition = partition * k_ones
        activations = activations / tf.reshape(partition , tf.shape(activations))
        v1_sample = tf.nn.relu(tf.sign(activations - tf.random_uniform(tf.shape(activations))))
       
        return v1_sample, activations

    def contrastive_divergence(self, v1):
        h1, h1a = self.sample_hidden(v1)
        v2, v2a = self.sample_visible(h1)
        self.predict = v2a
        h2, h2a = self.sample_hidden(v2)
        return [v1, h1, h1a,  v2, v2a, h2, h2a]

    def gradient(self, v1, h1, v2, h2a, masks):
        
        #gw = tf.matmul(tf.transpose(v1), h1) - tf.matmul(tf.transpose(v2), h2a)
        #gbv = tf.reduce_mean(v1 - v2, 0) 
        #gbh = tf.reduce_mean(h1 - h2a, 0)
        v1h1_mask = outer(masks, h1)
        gw = tf.reduce_mean(outer(v1, h1) * v1h1_mask - outer(v2, h2a) * v1h1_mask,axis= 0)
        gbv = tf.reduce_mean((v1 * masks) - (v2 * masks),axis= 0)
        gbh = tf.reduce_mean(h1 - h2a, axis = 0)
        return [gw, gbv, gbh]
    
    def train(self, vis,  w_lr=0.001, v_lr=0.001,
                h_lr=0.001, decay=0.0000, T=1,momentum=0.9):
        v1, h1, h1a, v2, v2a, h2, h2a = self.contrastive_divergence(vis)
        for _ in range(T-1):
            v1, h1, h1a, v2, v2a, h2, h2a = self.contrastive_divergence(v2)
        gw, gbv, gbh = self.gradient(v1, h1a, v2a, h2a, self.mask)
        if decay:
            gw -= decay * self.weights
        update_w = tf.assign(self.weights, 
                             self.weights + momentum * self.prev_gw + w_lr * gw)
        update_bh = tf.assign(self.hbias,
                              self.hbias + momentum * self.prev_gbh + h_lr * gbh)
        update_bv = tf.assign(self.vbias,
                              self.vbias + momentum * self.prev_gbv + v_lr * gbv)
        update_prev_gw = tf.assign(self.prev_gw, gw)
        update_prev_gbh = tf.assign(self.prev_gbh, gbh)
        update_prev_gbv = tf.assign(self.prev_gbv, gbv)
        optimizer = (update_w, update_bh, update_bv, update_prev_gw,
                     update_prev_gbh, update_prev_gbv)
        return optimizer



