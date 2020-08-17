import numpy as np
import tensorflow as tf

def resblock_ll(x, filters, name, padding='SAME'):
    filt0 = filters[0]
    filt1 = filters[-1]
    res = tf.layers.conv1d(x, filt0.shape[-1], 1, strides=1, name=name + '_conv0', padding=padding)
    resblock = tf.nn.conv1d(x, filt0, stride=1, name=name + '_conv1', padding=padding)
    resblock = tf.layers.batch_normalization(resblock, name=name + '_batch1')
    resblock = tf.nn.elu(resblock, name=name + '_elu')
    resblock = tf.nn.conv1d(resblock, filt1, stride=1, name=name + '_conv2', padding=padding)
    resblock = tf.layers.batch_normalization(resblock, name=name + '_batch2')
    return tf.nn.elu(resblock + res)

def resblock(x, filters, kernel_size, name):
    res = tf.layers.conv1d(x, filters, 1, strides=1, name=name+'_conv0', padding='same')
    resblock=tf.layers.conv1d(x, filters, kernel_size, strides=1, name=name+'_conv1', padding='same')
    resblock=tf.layers.batch_normalization(resblock, name=name+'_batch1')
    resblock=tf.nn.elu(resblock, name=name+'_elu')
    resblock=tf.layers.conv1d(resblock, filters, kernel_size, strides=1, name=name+'_conv2', padding='same')
    resblock=tf.layers.batch_normalization(resblock, name=name+'_batch2')
    return tf.nn.elu(resblock+res)