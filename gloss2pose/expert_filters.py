import numpy as np
import tensorflow as tf

class expert_filters(object):
    def __init__(self, shape, name, rng):
        self.shape = shape
        self.rng = rng

        self.filter0 = tf.Variable(self.initial_alpha(shape), name=name + "_0")
        self.filter1 = tf.Variable(self.initial_alpha([shape[0], shape[1], shape[-1], shape[-1]]), name=name + "_1")

    def get_filt(self, classed, batch_size):
        a = tf.expand_dims(self.filter0, 1)
        a = tf.tile(a, [1, batch_size,1,  1, 1])
        w = classed
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)

        r = tf.multiply(w, a)
        f0 = tf.reduce_sum(r, axis=0)
        f0 = tf.slice(f0, [0, 0, 0, 0], [1, self.shape[1], self.shape[2], self.shape[3]])

        a = tf.expand_dims(self.filter1, 1)
        a = tf.tile(a, [1, batch_size, 1, 1, 1])
        w = classed
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)

        r = tf.multiply(w, a)
        f1 = tf.reduce_sum(r, axis=0)
        f1 = tf.slice(f1, [0, 0, 0, 0], [1, self.shape[1], self.shape[-1], self.shape[-1]])

        return tf.squeeze(f0), tf.squeeze(f1)

    def initial_alpha_np(self, shape):
        rng = self.rng
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return alpha

    def initial_alpha(self, shape):
        alpha = self.initial_alpha_np(shape)
        return tf.convert_to_tensor(alpha, dtype=tf.float32)
