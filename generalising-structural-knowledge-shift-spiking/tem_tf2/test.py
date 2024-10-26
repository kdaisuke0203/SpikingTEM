import model_utils

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from snn_layers import *
import poisson_spike



"""cell_num = 3
timewindow = 2
x = tf.constant([1, 2, 3, 2, 4, 6], shape=[timewindow, cell_num])
print("x",x)
x = np.add(x, 0)

x2 = poisson_spike.generate_poisson_spikes(x, T=1)
x2 = np.array(x2)
[print(len(x)) for f, x in enumerate(x2)]

print(x2)"""


"""def cond(t1):
    return tf.less(t1, 7)

def body(t1):
    tf.print(tf.add(t1, 1))
    return [tf.add(t1, 1)]

t1 = tf.constant(1)
t2 = tf.constant(5)

res = tf.while_loop(cond, body, [t1])

tf.print(res)"""

array = tf.Variable(tf.random.normal([10]))
i = tf.constant(0)
l = tf.Variable([])
rate = 2
def body(i, l):
    tf.print("i", i)
    #temp = tf.gather(array,i)
    #x = tf.random.normal([1])
    #print("xx",x[0] , array[i])
    temp = array[i] + -tf.math.log(tf.random.uniform([1]))[0] / rate
    tf.print("temp",temp)
    l = tf.concat([l, [temp]], 0)
    tf.print("l",l)
    i = tf.add(i, int(temp))
    return i, l

def cond(i,l):
   return i < 10

#x = tf.constant(1.0)
#index,list_vals = tf.while_loop(cond, body, [i,l], shape_invariants=[i.get_shape(),tf.TensorShape([None])])
index,list_vals = tf.while_loop(cond, body, [i,l], shape_invariants=[tf.TensorShape([None])])

print("index,list_vals",index,list_vals)