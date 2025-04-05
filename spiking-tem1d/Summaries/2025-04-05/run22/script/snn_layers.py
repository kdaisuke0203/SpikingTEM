import tensorflow as tf
import poisson_spike
from tensorflow.keras.layers import Layer
import numpy as np
import math


def pseudo_derivative(v_scaled, dampening_factor):
  return dampening_factor * tf.maximum(0.,1 - tf.abs(v_scaled))


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):

    # This means: z = 1 if v > thr else 0
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        # This is where we overwrite the gradient
        # dy = dE/dz (total derivative) is the gradient back-propagated from the loss down to z(t)
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


class RSNN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3, name=None):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tf.exp(-1 / tau)

        self.dampening_factor = dampening_factor
        self.thr = thr

        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        self.tt = 0
        self.k = 14
        self.n_shape = 0

    def build(self, input_shape):
        #n_in = input_shape[-1]
        n_in = input_shape[-1]
        #print("NIN",n_in, input_shape)
        n = self.num_neurons
        
        rand_init = tf.keras.initializers.RandomNormal

        # define the input weight variable
        self.input_weights = self.add_weight(
            shape=(n_in,n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='input_weights')

        self.Bernoulli_weights = self.add_weight(
            shape=(n,self.k*n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='Bernoulli_weights')
        
        # define the recurrent weight variable
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=np.bool)), tf.bool)
        #self.recurrent_weights = self.add_weight(
        #    shape=(n, n), initializer=rand_init(stddev=1. / np.sqrt(n)), name='recurrent_weights')

        super().build(input_shape)

    def get_recurrent_weights(self):
      w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
      return w_rec

    def call(self, inputs, states, constants=None):
        #print("IN,S",inputs)
        old_v = states[0]
        old_spike = states[1]
        f=4.0
        random_indices = []

        # compute the input currents
        #w_rec = self.get_recurrent_weights()
        #print("IIII",inputs, states)
        self.tt += 1
        i_in = inputs @ self.input_weights + 1*math.sin(2*math.pi*f*(0.002*self.tt))#+ old_spike @ w_rec
      
        # update the voltage
        d = self.decay
        i_reset = - self.thr * old_spike
        new_v = d * old_v + (1-d) * i_in + i_reset

        v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_z_Bernoulli = new_z @ self.Bernoulli_weights
        #print("ZZZZZZZZZZz",new_z_Bernoulli)
        # sampling from q(z_t | x_<=t, z_<t)
        #random_index = tf.random.uniform((batch_size*self.num_neurons,), 0, self.k) 
                    #+ tf.arange(start=0, end=batch_size*self.num_neurons*self.k, step=self.k) #(B*C,) select 1 from every k value
        #random_index = random_index.to(x.device)
        #random_indices.append(random_index)
        random_index = np.random.randint(0,self.num_neurons*self.k,self.num_neurons)
        new_z2 = tf.gather(new_z_Bernoulli, random_index, axis=1)
        
        new_state = (new_v, new_z2)
        return (new_z2, new_v), new_state

class RSNN2(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3, name=None):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tf.exp(-1 / tau)

        self.dampening_factor = dampening_factor
        self.thr = thr

        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        self.tt = 0
        self.k = 4
        self.n_shape = 0

    def build(self, input_shape):
        #n_in = input_shape[-1]
        n_in = input_shape[-1]
        #print("NIN",n_in, input_shape)
        n = self.num_neurons
        
        rand_init = tf.keras.initializers.RandomNormal

        # define the input weight variable
        self.input_weights = self.add_weight(
            shape=(n_in,n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='input_weights')

        self.Bernoulli_weights = self.add_weight(
            shape=(n,self.k*n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='Bernoulli_weights')
        
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=np.bool)), tf.bool)

        super().build(input_shape)

    def get_recurrent_weights(self):
      w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
      return w_rec

    def call(self, inputs, states, constants=None):
        #print("IN,S",inputs)
        old_v = states[0]
        old_spike = states[1]
        f=4.0
        random_indices = []

        self.tt += 1
        i_in = inputs @ self.input_weights #+ 6.0*math.sin(2*math.pi*f*(0.002*self.tt))#+ old_spike @ w_rec
      
        # update the voltage
        d = self.decay
        i_reset = - self.thr * old_spike
        new_v = d * old_v + (1-d) * i_in + i_reset

        v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_z_Bernoulli = new_z @ self.Bernoulli_weights
        random_index = np.random.randint(0,self.num_neurons*self.k,self.num_neurons)
        new_z2 = tf.gather(new_z_Bernoulli, random_index, axis=1)
        
        new_state = (new_v, new_z2)
        return (new_z2, new_v), new_state