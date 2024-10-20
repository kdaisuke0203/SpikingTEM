import tensorflow as tf
from tensorflow.keras.layers import Layer

dt = 5
a = 0.25
aa = 0.5  
Vth = 0.2
tau = 0.25


class LIFSpike(Layer): #(tf.keras.layers.Layer) not work. Why?
    # add an activation paramter
    def __init__(self, units=32, activation=None, name=None):
        super(LIFSpike, self).__init__()
        self.units = units
        
        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        # initialize the weight
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name='kernel',
                             initial_value=w_init(shape=(input_shape[-1], self.units)),
                             trainable=True)
        
        # intialize the bias
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',
                             initial_value=b_init(shape=(self.units, )),
                             trainable=True)
        
    def call(self, inputs):
        # pass the computation to the activation layer
        #print("III",inputs)
        
        return tf.matmul(inputs, self.w) + self.b
        #nsteps = inputs.shape[-1]
        #u   = tf.zeros(inputs.shape[:-1])
        #out = tf.zeros(inputs.shape)
        #for step in range(nsteps):
        #    u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], tf.matmul(inputs[..., step], self.w) + self.b)
        #return out

    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, tau=tau):
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1



"""class LIFSpike(tf.keras.layers.Layer):
    
    #Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU.
    
    def __init__(self, units=32):
        super(LIFSpike, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, x):
        nsteps = tf.shape(x)[-1]
        u = tf.zeros(tf.shape(x)[:-1], dtype=x.dtype)
        out = tf.zeros(tf.shape(x), dtype=x.dtype)

        for step in range(nsteps):
            u, out[..., step] = self.state_update(u, out[..., max(step - 1, 0)], x[..., step])
        return out

    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n):
        u_t1_n1 = tau * tf.matmul(u_t_n1, (1 - o_t_n1)) + tf.matmul(inputs, self.w)
        o_t1_n1 = tf.cast(tf.greater(u_t1_n1, Vth), tf.float32)
        return u_t1_n1, o_t1_n1  """

