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

@tf.custom_gradient
def spike_function(x):
    def grad(dy):
        #sigma = 1.0
        dampening_factor = 1.0 #1s, 30a
        #surrogate_grad = tf.exp(-tf.square(x) / (2 * sigma**2)) / (sigma * tf.sqrt(2.0 * tf.constant(math.pi)))
        surrogate_grad = dampening_factor * tf.maximum(0.,1 - tf.abs(x))
        return dy * surrogate_grad
    out = tf.cast(x > 0.0, tf.float32)
    return out, grad

class SpikingDense(tf.keras.layers.Layer):
    def __init__(self, par,  output_size, dt=1.0, name=None): #tau:20a
        super().__init__()

        self.output_size = output_size
        self.tau = par['tau']
        self.dt = dt
        self.k = par['k']
        self.thr = par['thr']
        self.v_min = par['v_min']

    def build(self, input_shape):
        x_shape, v_shape = input_shape
        n_in = x_shape[-1]
        rand_init = tf.keras.initializers.RandomNormal
        """self.w = [self.add_weight(shape=(self.input_size, self.output_size),
                                 initializer='glorot_uniform',
                                 trainable=True, name='spike_weights'+ str(f)) for f in
                    range(self.k)]"""
        self.w = self.add_weight(shape=(n_in, self.output_size*self.k),
                                 initializer='glorot_uniform',
                                 trainable=True, name='spike_weights')
        self.b = self.add_weight(shape=(self.output_size*self.k,),
                                 initializer='zeros',
                                 trainable=True, name='spike_bias')
        #self.v = self.add_weight(shape=(1,self.output_size*self.k), initializer='zeros', trainable=False)
        """self.Bernoulli_weights = self.add_weight(
            shape=(self.output_size,self.k*self.output_size),
            initializer=rand_init(stddev=1. / np.sqrt(self.input_size)),
            name='Bernoulli_weights')"""    
    
    def call(self, inputs):
        x, v_prev = inputs
        alpha = tf.exp(-self.dt / self.tau)
        #print("VVVVV",self.w)
        #print("tf.matmul(x, self.w)",tf.matmul(x, self.w))
        #v_new = [alpha * v_prev[i,:,:] + tf.matmul(x, self.w[i]) - threshold for i in range(self.k)]
        #v_new = tf.convert_to_tensor(v_new, dtype=tf.float32)
        v_new = alpha * v_prev + tf.matmul(x, self.w) #+ self.b
        v_new = tf.clip_by_value(v_new, clip_value_min=self.v_min, clip_value_max=self.thr + 3e-1)
        v_new_ = v_new - self.thr
        #dv = (-self.v + tf.matmul(x, self.w) + self.b - threshold) / self.tau
        #print("DDDD",dv, self.v)
        #self.v.assign_add(self.dt * dv)
        #print("VVVVVVVVvv",v_new)
        spike = spike_function(v_new_)
        #spike = spike_function(self.v)
        #print("SSSSSSSSSSSs",spike)
        #spike_Bernoulli = spike @ self.Bernoulli_weights
        #random_index = np.random.randint(0,self.output_size*self.k,self.output_size)
        #spike_Bernoulli = tf.gather(spike_Bernoulli, random_index, axis=1)
        #self.v.assign(tf.where(spike > 0, 0.0, self.v))
        v_reset = tf.where(spike > 0.0, self.v_min, v_new)
        return spike, v_reset

class SpikingDense2(tf.keras.layers.Layer):
    def __init__(self, par,  output_size, dt=1.0, name=None): #tau:20a
        super().__init__()

        self.output_size = output_size
        self.tau = par['tau']
        self.dt = dt
        self.k = par['k']
        self.thr = par['thr']
        self.v_min = par['v_min']

    def build(self, input_shape):
        x_shape, v_shape = input_shape
        n_in = x_shape[-1]
        rand_init = tf.keras.initializers.RandomNormal
        """self.w = [self.add_weight(shape=(self.input_size, self.output_size),
                                 initializer='glorot_uniform',
                                 trainable=True, name='spike_weights'+ str(f)) for f in
                    range(self.k)]"""
        self.w = self.add_weight(shape=(n_in, self.output_size*self.k),
                                 initializer='glorot_uniform',
                                 trainable=False, name='spike_weights')
        self.b = self.add_weight(shape=(self.output_size*self.k,),
                                 initializer='zeros',
                                 trainable=False, name='spike_bias')
        #self.v = self.add_weight(shape=(1,self.output_size*self.k), initializer='zeros', trainable=False)
        """self.Bernoulli_weights = self.add_weight(
            shape=(self.output_size,self.k*self.output_size),
            initializer=rand_init(stddev=1. / np.sqrt(self.input_size)),
            name='Bernoulli_weights')"""    
    
    def call(self, inputs):
        x, v_prev = inputs
        alpha = tf.exp(-self.dt / self.tau)
        v_new = alpha * v_prev + tf.matmul(x, self.w) #+ self.b
        v_new = tf.clip_by_value(v_new, clip_value_min=self.v_min, clip_value_max=self.thr + 3e-1)
        v_new_ = v_new - self.thr
        spike = spike_function(v_new_)
        v_reset = tf.where(spike > 0.0, self.v_min, v_new)
        return spike, v_reset


class RSNN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr=1.0, tau=10., dampening_factor=50.0, name=None): #dampening_factor:30
      #tau:10a
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
        self.k = 1
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
        self.b = self.add_weight(shape=(n,),
                                 initializer='zeros',
                                 trainable=True, name='rnn_spike_bias')
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
        v_min = -1.0#-1.0
        random_indices = []

        # compute the input currents
        #w_rec = self.get_recurrent_weights()
        #print("IIII",inputs, states)
        self.tt += 1
        i_in = inputs @ self.input_weights +self.b #+ self.thr*0.1*math.sin(2*math.pi*f*(1/128*self.tt))#+ old_spike @ w_rec
        #0.06a, 0.6a
        # inputs @ self.input_weights +self.b ->a
      
        # update the voltage
        d = self.decay
        i_reset = - self.thr * old_spike
        new_v = d * old_v + i_in #+ (1-d) * i_in #+ i_reset
        new_v = tf.clip_by_value(new_v, clip_value_min=v_min, clip_value_max=self.thr+1e-2)

        v_scaled = (new_v - self.thr) #/ self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_v = tf.where(new_z > 0.0, v_min, new_v)
        new_z_Bernoulli = new_z @ self.Bernoulli_weights
        random_index = np.random.randint(0,self.num_neurons*self.k,self.num_neurons)
        new_z2 = tf.gather(new_z_Bernoulli, random_index, axis=1)
        
        new_state = (new_v, new_z2)
        #return (new_z2, new_v), new_state
        return (new_z, new_v), new_state

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


class RSNN3(tf.keras.layers.Layer):
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

        # compute the input currents
        #w_rec = self.get_recurrent_weights()
        #print("IIII",inputs, states)
        self.tt += 1
        i_in = inputs @ self.input_weights #+ 1*math.sin(2*math.pi*f*(0.002*self.tt))#+ old_spike @ w_rec
      
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
        #return (new_z2, new_v), new_state
        return (new_z, new_v), new_state