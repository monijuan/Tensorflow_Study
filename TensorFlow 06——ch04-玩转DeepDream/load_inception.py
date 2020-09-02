# coding:utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf

graph = tf.Graph()  # 创建图
sess = tf.InteractiveSession(graph=graph)  # 创建Session
model_fn = 'tensorflow_inception_graph.pb'  # 存储了inception的网络结构和对应的数据 
with tf.gfile.FastGFile(model_fn, 'rb') as f:  # 导入inception
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 输入图像需要经过处理才能送入网络中
t_input = tf.placeholder(np.float32, name='input')  # 输入的图像
imagenet_mean = 117.0  # 需要减去一个均值
# expand_dims是加一维，从[height, width, channel]变成[1, height, width, channel]
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# 找到所有卷积层，输出卷积层层数
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
print('Number of layers', len(layers))

# 输出mixed4d_3x3_bottleneck_pre_relu的形状
name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s: %s' % (name, str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))
