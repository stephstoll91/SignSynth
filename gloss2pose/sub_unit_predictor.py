import numpy as np
import tensorflow as tf

class subunit_predictor(object):
    def __init__(self, input, input_size, output_size, hidden_size, rng, keep_prob):
        self.input = input
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rng = rng
        self.keep_prop = keep_prob

        self.w0 = tf.Variable(self.initial_weight([self.hidden_size, self.input_size]), name="w0")
        self.w1 = tf.Variable(self.initial_weight([self.hidden_size, self.hidden_size]), name="w1")
        self.w2 = tf.Variable(self.initial_weight([self.output_size, self.hidden_size]), name="w2" )

        self.b0 = tf.Variable(self.initial_bias([self.hidden_size, 1]), name="b0")
        self.b1 = tf.Variable(self.initial_bias([self.hidden_size, 1]), name="b1")
        self.b2 = tf.Variable(self.initial_bias([self.output_size, 1]), name="b2")

        self.classed = self.fp()

    def fp(self):
        H0 = tf.transpose(self.input)
        H1 = tf.matmul(self.w0, H0) + self.b0
        H1 = tf.nn.elu(H1)
        H1 = tf.nn.dropout(H1, keep_prob=self.keep_prop)

        H2 = tf.matmul(self.w1, H1) + self.b1
        H2 = tf.nn.elu(H2)
        H2 = tf.nn.dropout(H2, keep_prob=self.keep_prop)

        H3 = tf.matmul(self.w2, H2) + self.b2
        H3 = tf.nn.softmax(H3, dim=0)

        return H3

    def initial_weight(self, shape):
        rng = self.rng
        weight_bound = np.sqrt(6. / np.sum(shape[-2:]))
        weight = np.asarray(
            rng.uniform(low=-weight_bound, high=weight_bound, size=shape),
            dtype=np.float32)
        return tf.convert_to_tensor(weight, dtype = tf.float32)

    def initial_bias(self, shape):
        return tf.zeros(shape, tf.float32)
