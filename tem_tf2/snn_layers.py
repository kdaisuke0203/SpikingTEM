import tensorflow as tf
import poisson_spike
from tensorflow.keras.layers import Layer
import numpy as np
import math


@tf.custom_gradient
def spike_function(x):
    def grad(dy):
        sigma = 2.0
        dampening_factor = 1.0
        #surrogate_grad = tf.exp(-tf.square(x) / (2 * sigma**2)) / (sigma * tf.sqrt(2.0 * tf.constant(math.pi)))
        surrogate_grad = dampening_factor * tf.maximum(0.,1 - tf.abs(x)/sigma)
        return dy * surrogate_grad
    out = tf.cast(x > 0.0, tf.float32)
    return out, grad

class SpikingDense(tf.keras.layers.Layer):
    def __init__(self, par, output_size, tau=0.5, dt=1.0, name=None):
        super().__init__()

        self.output_size = output_size
        self.tau = par['tau']
        self.dt = dt
        self.k = par.k
        self.v_min = par['v_min']
        self.v_reset = par['v_reset']
        self.v_max = par['v_max']
        self.thr = par['thr']

    def build(self, input_shape):
        x_shape, v_shape = input_shape
        n_in = x_shape[-1]
        rand_init = tf.keras.initializers.RandomNormal
        self.w = self.add_weight(shape=(n_in, self.output_size*self.k),
                                 initializer='glorot_uniform',
                                 trainable=True, name='spike_weights')
        """self.b = self.add_weight(shape=(self.output_size*self.k),
                                 initializer='zeros',
                                 trainable=True, name='spike_bias')"""
    
    def call(self, inputs):
        x, v_prev = inputs
        alpha = tf.exp(-self.dt / self.tau)
        v_new = alpha * v_prev + tf.matmul(x, self.w)
        v_new = tf.clip_by_value(v_new, clip_value_min=self.v_min, clip_value_max=self.v_max)
        #print("VVVVVVVVvv",v_new)
        spike = spike_function(v_new - self.thr)

        v_reset = tf.where(spike > 0.0, self.v_reset, v_new)
        return spike, v_reset

class SpikingDense0(tf.keras.layers.Layer):
    def __init__(self, par, output_size, tau=0.5, dt=1.0, name=None):
        super().__init__()

        self.output_size = output_size
        self.tau = par['tau']
        self.dt = dt
        self.k = par.k
        self.v_min = par['v_min']
        self.v_reset = par['v_reset']
        self.v_max = par['v_max']
        self.thr = par['thr']

    def build(self, input_shape):
        x_shape, v_shape = input_shape
        n_in = x_shape[-1]
        rand_init = tf.keras.initializers.RandomNormal
        self.w = self.add_weight(shape=(n_in, self.output_size),
                                 initializer='glorot_uniform',
                                 trainable=True, name='spike_weights')
        """self.b = self.add_weight(shape=(self.output_size*self.k),
                                 initializer='zeros',
                                 trainable=True, name='spike_bias')"""
    
    def call(self, inputs):
        x, v_prev = inputs
        
        alpha = tf.exp(-self.dt / self.tau)
        v_new = alpha * v_prev + tf.matmul(x, self.w)
        v_new = tf.clip_by_value(v_new, clip_value_min=self.v_min, clip_value_max=self.v_max)
        #print("VVVVVVVVvv",v_new)
        spike = spike_function(v_new - self.thr)

        v_reset = tf.where(spike > 0.0, self.v_reset, v_new)
        return spike, v_reset