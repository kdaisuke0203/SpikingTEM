import model_utils

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from snn_layers import *
import poisson_spike



cell_num = 3
timewindow = 2
x = tf.constant([1, 2, 3, 2, 4, 6], shape=[timewindow, cell_num])
print("x",x)
x = np.add(x, 0)

x2 = poisson_spike.generate_poisson_spikes(x, T=1)
x2 = np.array(x2)
[print(len(x)) for f, x in enumerate(x2)]

print(x2)