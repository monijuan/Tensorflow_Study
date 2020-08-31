# -*- coding: utf-8 -*-
# @Author  : 模拟卷
# @CSDN    : https://blog.csdn.net/qq_34451909
# @Software: PyCharm Tensorflow1.4.0 python3.6.9

import cifar10
import tensorflow as tf

# tf.app.flags.FLAGS是TensorFlow内部的一个全局变量存储器，同时可以用于命令行参数的处理
# f.app.flags.FLAGS.data_dir为CIFAR-10的数据路径
FLAGS = tf.app.flags.FLAGS
# 数据下载后存到cifar10_data
FLAGS.data_dir = 'cifar10_data/'
# 如果不存在数据文件，就会执行下载
cifar10.maybe_download_and_extract()
