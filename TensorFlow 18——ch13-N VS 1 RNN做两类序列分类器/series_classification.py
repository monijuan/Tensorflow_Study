# coding: utf-8
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np


##################################################################
# 这个类用于产生序列样本
class ToySequenceData(object):
    """ 生成序列数据。每个数量可能具有不同的长度。
    一共生成下面两类数据
    - 类别 0: 线性序列 (如 [0, 1, 2, 3,...])
    - 类别 1: 完全随机的序列 (i.e. [1, 3, 10, 7,...])
    注意:
    max_seq_len是最大的序列长度。对于长度小于这个数值的序列，我们将会补0。
    在送入RNN计算时，会借助sequence_length这个属性来进行相应长度的计算。
    """

    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            len = random.randint(min_seq_len, max_seq_len)  # 序列的长度是随机的
            self.seqlen.append(len)  # 用于存储所有的序列。
            # 以50%的概率，随机添加一个线性或随机的训练
            if random.random() < .5:
                # 生成一个线性序列
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in range(rand_start, rand_start + len)]
                # 长度不足max_seq_len的需要补0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                # 线性序列的label是[1, 0]（因为我们一共只有两类）
                self.labels.append([1., 0.])
            else:
                # 生成一个随机序列，长度不足max_seq_len的需要补0
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，会重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


##################################################################
# 这一部分只是测试一下如何使用上面定义的ToySequenceData
def test_ToySequenceData():
    '''
    这一部分只是测试一下如何使用上面定义的ToySequenceData
    '''
    tmp = ToySequenceData()
    batch_data, batch_labels, batch_seqlen = tmp.next(32) # 生成样本

    # batch_data 是序列数据，它是一个嵌套的list，形状为(batch_size, max_seq_len, 1)
    print(np.array(batch_data).shape)  # 输出 (32, 20, 1)
    print(batch_data[0]) # 打印第一个序列

    # batch_labels 是 label ，它也是一个嵌套的 list ，形状为(batch_size, 2)，2表示两类分类
    print(np.array(batch_labels).shape)  # (32, 2)
    print(batch_labels[0]) # 打印第一个序列的label

    # batch_seqlen 表示每个序列的实际长度，为 batch_size
    print(np.array(batch_seqlen).shape)  # (32,)
    print(batch_seqlen[0]) # 打印第一个序列的长度

# test_ToySequenceData()


##################################################################
# 定义各种参数

# 运行的参数
learning_rate = 0.01        # 学习率
training_iters = 1000000    # 最大运行步数
batch_size = 128            # batch 序列数
display_step = 10           # 每隔多少步打印信息

# 网络定义时的参数
seq_max_len = 20    # 最大的序列长度
n_hidden = 64       # RNN 隐层的size
n_classes = 2       # 类别数

# 定义三个占位符
x = tf.placeholder("float", [None, seq_max_len, 1]) # x为输入
y = tf.placeholder("float", [None, n_classes])      # y为输出
seqlen = tf.placeholder(tf.int32, [None]) # 存 x 每个序列的实际长度，None 实际为 batch_size

# weights 和 bias 在输出时会用到
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases  = {'out': tf.Variable(tf.random_normal([n_classes]))}

# 定义 trainset 和 testset
trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)


##################################################################
#定义 RNN 分类模型
def dynamicRNN(x, seqlen, weights, biases):
    '''
    输入x的形状： (batch_size, max_seq_len, n_input)
    输入seqlen的形状：(batch_size, )
    '''
    # 定义一个 BasicLSTMCell ，隐层的大小为 n_hidden(初始=64)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # 使用 tf.nn.dynamic_rnn 展开时间维度，每一个序列应该 seqlen 步
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # outputs 的形状为 (batch_size, max_seq_len, n_hidden)
    # 取出与序列长度相对应的输出，但是 TensorFlow 不支持直接对 outputs 进行索引，因此用下面的方法：
    batch_size = tf.shape(outputs)[0]    
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1) # 得到每一个序列真正的index
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']


##################################################################
# 定义损失和准确率
pred = dynamicRNN(x, seqlen, weights, biases) # pred 是 logits 而不是概率
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # 定义损失
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # 分类准确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # 平均准确率
init = tf.global_variables_initializer() # 变量初始化

##################################################################
# 训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen}) # 更新一次参数
        if step % display_step == 0:          
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen}) # 计算准确度
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen}) # 计算损失
            print("Iter " + str(step * batch_size) + \
                  ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))
