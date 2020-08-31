# -*- coding: utf-8 -*-
# @Author  : 模拟卷
# @Software: PyCharm Tensorflow1.8.0 python3.5.6

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
weight_variable()：用来创建卷积核
tf.truncated_normal 是从截断的正态分布中输出随机值 
shape 表示生成张量的维度
stddev 是标准差
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

'''
tf.constant 创建常量 
'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
conv2d()：二维卷积
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
1.input：需要做卷积的输入图像，它要求是一个Tensor，
    具有[batch, in_height, in_width, in_channels]这样的shape，
    具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
    注意这是一个4维的Tensor，要求类型为float32和float64其中之一
2.filter：相当于CNN中的卷积核，它要求是一个Tensor，
    具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
    具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
    要求类型与参数input相同，第三维in_channels是参数input的第四维
3.strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
4.padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
5.use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
结果返回一个Tensor，这个输出就是feature map，
shape仍然是[batch, height, width, channels]这种形式。
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


'''
max_pool_2x2()：池化
tf.nn.max_pool(value, ksize, strides, padding, name=None)
1.value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
    依然是[batch, height, width, channels]这样的shape
2.ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    因为我们不想在batch和channels上做池化，所以这两个维度设为了1
3.strides：窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
4.padding：可以取’VALID’ 或者’SAME’
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 读取数据
    x = tf.placeholder(tf.float32, [None, 784]) # 占位符，代表待识别的图片 28*28
    y_ = tf.placeholder(tf.float32, [None, 10]) # 占位符，代表图像标签 10

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积、激活、池化
    W_conv1 = weight_variable([5, 5, 1, 32])  # patch=5x5,高=32的卷积核
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # 输出14*14*32

    # 第二层卷积、激活、池化
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch=5x5,高=64的卷积核
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # 输出7*7*64

    # 全连接层，输出为1024维的向量
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 用来做矩阵乘法，压平结果
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 结果是一维，长1024
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 把1024维的向量转换成10维，对应10个类别
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # 结果是一维，长10

    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 同样定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # 训练
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 50 == 0:
            print("step=%d,accuracy=%f"%(i, accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})))
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
