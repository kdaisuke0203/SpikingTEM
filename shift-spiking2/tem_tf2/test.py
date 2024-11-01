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

"""array = tf.Variable(tf.random.normal([10]))
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

print("index,list_vals",index,list_vals)"""


"""import tensorflow as tf

# Example tensors
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6],[4,1]])

mus = tf.constant(tf.zeros([3,5,4]))

# Concatenate along axis 0 (rows)
result_axis_0 = tf.concat([tensor1, tensor2], axis=0)

# Concatenate along axis 1 (columns)
result_axis_1 = tf.concat([tensor1, tensor2], axis=1)

tf.print(mus)

tf.print(tf.concat(mus,axis=1))
#tf.print(result_axis_0)
#tf.print(result_axis_1)"""

"""t_mat = tf.constant([[[1,2],[1,1]]])
g_p = tf.constant([[[1,2],[1,1]]])
print("g",g_p)
print("tf.expand_dims(g_p, axis=2)))",tf.expand_dims(g_p, axis=2))
print("mat",tf.matmul(t_mat, tf.expand_dims(g_p, axis=2)))
print("sq",tf.squeeze(tf.matmul(t_mat, tf.expand_dims(g_p, axis=2))))

tf.squeeze(tf.matmul(t_mat, tf.expand_dims(g_p, axis=2)))"""
"""a = tf.constant(np.arange(1, 19, dtype=np.int32), shape=[2, 3, 3])
b = tf.constant(np.arange(13, 19, dtype=np.int32), shape=[2, 3, 1])
c = tf.matmul(a, b)
print("c",c)"""
"""a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
b = tf.constant([[2.0, 0.0], [1.0, 2.0]], dtype=tf.float32)

# 初期化
i = tf.constant(0)
values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # dtypeをfloat32に設定

# 条件: iが5未満のときにループを続ける
def condition(i, values):
    return i < 5

# 本体: 行列積を計算してvaluesに追加
def body(i, values):
    result = tf.matmul(a, b)  # 行列積を計算
    values = values.write(i, result)  # 計算結果をvaluesに追加
    return i + 1, values

# ループ実行
_, result = tf.while_loop(condition, body, [i, values])
result = result.stack()  # TensorArrayをテンソルに変換
print("r",result)"""
"""tensor = tf.random.uniform((4, 2, 6))

# 軸を入れ替えて形を変更
reshaped_tensor = tf.transpose(tensor, perm=[1, 2, 0])  # 軸を入れ替え
reshaped_tensor2 = tf.reshape(reshaped_tensor, (2, 6, 4))  # 形を変更

print(reshaped_tensor,reshaped_tensor2) """
def tf_repeat_axis_1(tensor, repeat, dim1):
    dim0 = tf.shape(input=tensor)[0]
    return tf.reshape(tf.tile(tf.reshape(tensor, (-1, 1)), (1, repeat)), (dim0, dim1))

#tensor = tf.constant([1, 2, 3])
tensor = tf.random.uniform((3, 2))
repeat = 2
dim1 = 4
print("1",tf.reshape(tensor, (-1, 1)))
print("2",tf.tile(tf.reshape(tensor, (-1, 1)), (1, repeat)))
result = tf_repeat_axis_1(tensor, repeat, dim1)
print(result)
