import tensorflow as tf
import poisson_spike
from tensorflow.keras.layers import Layer

dt = 5
a = 0.25
aa = 0.5  
tau = 0.2


class LIFSpike(Layer): #(tf.keras.layers.Layer) not work. Why?
    # add an activation paramter
    def __init__(self, units, activation=None, name=None, threshold=0.5, timewindow=10, **kwargs):
        super(LIFSpike, self).__init__(**kwargs)
        self.units = units
        #self.dense = tf.keras.layers.Dense(units)
        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)
        self.prev_output = None
        self.threshold = threshold
        self.timewindow = timewindow
        
    def build(self, input_shape):
        # initialize the weight
        #print("input_shape",input_shape, input_shape[-1])
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name='kernel',
                             initial_value=w_init(shape=(input_shape[-1], self.units)),
                             trainable=True)
        
        # intialize the bias
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',
                             initial_value=b_init(shape=(self.units, )),
                             trainable=True)

        # intialize the membrane
        #mem_init = tf.zeros_initializer()
        #self.mem = tf.Variable(name='mem',
                             #initial_value=mem_init(shape=(self.units, )),
                             #trainable=False)
        #prev_output_init = tf.zeros_initializer()
        """self.prev_output = tf.Variable(name='mem',
                             initial_value=prev_output_init(shape=(self.units, )),
                             trainable=False)
        self.prev_output = tf.Variable(name='mem',
                             initial_value=prev_output_init(shape=(self.units, )),
                             trainable=False)"""


    #@tf.function
    def call(self, inputs):
        #print("inputs", inputs)
        #inputs = poisson_spike.generate_poisson_spikes(inputs, T=self.timewindow)
        #print("inputs",inputs)
        #batch_size = tf.shape(inputs)[0]
        #if self.prev_output is None:
        #    self.prev_output = tf.zeros([16, self.units], dtype=inputs.dtype)
        # pass the computation to the activation layer
        #print("III",inputs)
        
        #print("o", o_t_n1)
        #self.mem = tau * self.mem * (1.0 - o_t_n1) + tf.matmul(inputs, self.w) + self.b
        #return self.mem
        #self.mem = self.mem + tf.matmul(inputs, self.w) + self.b
        #current_output = tau * self.prev_output * (1 - o_t_n1) + tf.matmul(inputs, self.w) + self.b 
        #prev_binary_output = tf.where(self.prev_output > self.threshold, 1.0, 0.0)
        #print("inputs", inputs) #shape(env_num, cell_num)
        #print("len(tf.shape(inputs))",inputs)
        """if len(tf.shape(inputs))==2:
            inputs2 = tf.tile(tf.expand_dims(inputs, axis=-1), [1,1,self.timewindow])
        else:
            print("len(tf.shape(inputs))",len(tf.shape(inputs)))
            inputs3 = inputs"""
        #print("inputs2", inputs2)
        #input_transformed = self.dense(inputs)
        input_transformed = tf.matmul(inputs, self.w) + self.b
        #input_transformed2 = self.dense(inputs2)
        #print("input_transformed", input_transformed)
        #print("self.prev_output", self.prev_output)
        #prev_output = tf.zeros([inputs.shape[0], self.units])
        #prev_output2 = tf.zeros([inputs2.shape[0], self.units, inputs2.shape[2]])
        #print("P",prev_output2)
        #binary_prev_output = tf.zeros([inputs.shape[0], self.units])
        for i in range(self.timewindow):
            if input_transformed.shape[0] is None:
                current_output = tau * input_transformed #+ self.prev_output
                #current_output2 = tau * input_transformed2
            else:
                current_output = input_transformed #+ tau * prev_output * (1 -binary_prev_output)
                #current_output2 = input_transformed2 + tau * prev_output2 #* (1 -binary_prev_output2)
                #prev_output = current_output
                #binary_prev_output = tf.where(prev_output > self.threshold, 1.0, 0.0)
                #tf.print("binary_prev_output", i, prev_output)
        #print("self.prev_output", self.prev_output)
        #binary_output = tf.where(current_output > self.threshold, 1.0, 0.0)
        #print("current_output", current_output2)
        #self.prev_output = current_output
        #print("inputs", inputs.shape[-1])
        #for i in range(inputs.shape[-1]):
            #self.prev_output += 1
        #tf.print("self.prev_output", self.prev_output)
        #with tf.init_scope():
            #self.prev_output = current_output
            #self.prev_output = tf.identity(current_output)

        #return binary_output
        return current_output
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

