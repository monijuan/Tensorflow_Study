# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 13:06
# @Author  : 模拟卷
# @File    : tf_test.py
# @CSDN    : https://blog.csdn.net/qq_34451909
# @Software: PyCharm Tensorflow1.13.1 python3.5.6

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope("baises"):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        with tf.name_scope("wx_plus_b"):
            wx_plus_b = tf.matmul(inputs,weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs


def main():
    # 定义变量
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # 定义占位符
    with tf.name_scope("inputs"):
        xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
        ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

    # 添加网络
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    preduction = add_layer(l1, 10, 1, activation_function=None)

    # 计算损失
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - preduction),axis=1)) # 按行求和
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter("logs/",sess.graph)

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data)
    plt.ion() # show了之后不暂停
    plt.show()

    # 训练
    for i in range(1001):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 10 == 0:
            print("第%d步的误差为%f"%(i, sess.run(loss, feed_dict={xs: x_data, ys: y_data})))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            preduction_value = sess.run(preduction, feed_dict={xs: x_data, ys: y_data})
            lines = ax.plot(x_data,preduction_value,'r-',lw=2)
            plt.pause(0.1)


if __name__ == '__main__':
    main()
