# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.tutorials.mnist import input_data # 导入MNIST教学的模块
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 读取数据


W = tf.Variable(tf.zeros([784, 10])) # 变量，参数，将一个784维的输入转换为一个10维的输出
b = tf.Variable(tf.zeros([10])) # 变量，参数，“偏置项”（bias）
x = tf.placeholder(tf.float32, [None, 784]) # 占位符，代表待识别的图片
y_ = tf.placeholder(tf.float32, [None, 10]) # 占位符，代表图像标签
y = tf.nn.softmax(tf.matmul(x, W) + b) # 模型的输出，y=softmax(Wx + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y))) # 根据y, y_构造交叉熵损失
Optimizer = tf.train.GradientDescentOptimizer(0.01) # 随机梯度下降
train_step = Optimizer.minimize(cross_entropy) # 对模型的参数（W和b）进行优化

sess = tf.InteractiveSession() # 创建Session，只有在Session中才能运行train_step
tf.global_variables_initializer().run() # 初始化变量，分配内存

for i in range(1000):
    # batch_xs为(100, 784)的图像，对应占位符x
    # batch_ys为(100, 10)的标签，对应占位符y_
    # 在mnist.train中取100个训练数据
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行train_step，运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 都是Tensor，在Session中运行Tensor可以得到Tensor的值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # 正确的预测结果
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 预测准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
