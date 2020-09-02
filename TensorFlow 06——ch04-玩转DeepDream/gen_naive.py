# coding: utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

# 以把一个 numpy.ndarray 保存成文件的形式。
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


# 渲染，实际上就是优化 layer_output[:, :, :, channel] 的平均值
def render_naive(t_obj, img0, iter_n=20, step=1.0, channel=139):
    t_score = tf.reduce_mean(t_obj) # t_score是优化目标，是t_obj的平均值   
    t_grad = tf.gradients(t_score, t_input)[0] # 计算 t_score 对 t_input 的梯度   
    img = img0.copy() # 创建新图
    for i in range(iter_n):
        # 在sess中计算梯度，以及当前的score，越大代表平均激活越大
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # 对 img 应用梯度， step 可以看做“学习率”
        g /= g.std() + 1e-8
        img += g * step
        print('i=%d,score(mean)=%f' % (i,score))   
        if i%10==0:
            savearray(img, 'jpg/naive_%d_%d.jpg'%(channel,i)) # 保存图片


# 导入Inception模型
graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

name = 'mixed4d_3x3_bottleneck_pre_relu' # 定义卷积层
channel = 100 # 定义通道数，总通道是144，这里用139通道举例
layer_output = graph.get_tensor_by_name("import/%s:0" % name)# 并取出对应的tensor
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0 # 定义原始的图像噪声
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=101, channel=channel) # 渲染并保存
