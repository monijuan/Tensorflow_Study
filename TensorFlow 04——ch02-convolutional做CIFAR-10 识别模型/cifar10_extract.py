# -*- coding: utf-8 -*-
# @Time    : 2020/8/31 8:19
# @Author  : 模拟卷
# @CSDN    : https://blog.csdn.net/qq_34451909
# @Software: PyCharm Tensorflow1.4.0 python3.6.9

# 导入当前目录的cifar10_input，这个模块负责读入cifar10数据
import cifar10_input
import tensorflow as tf
import os
import scipy.misc


def inputs_origin(data_dir):
    # 读取训练图像：filenames一共5个，从data_batch_1.bin到data_batch_5.bin，
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    # 判断文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)    
    filename_queue = tf.train.string_input_producer(filenames)  # 包装成queue
    read_input = cifar10_input.read_cifar10(filename_queue)  # 读取queue
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将图片转换为实数形式
    return reshaped_image  # 返回的reshaped_image是一张图片的tensor


if __name__ == '__main__':
    # 创建会话sess
    with tf.Session() as sess:
        # 调用inputs_origin。cifar10_data/cifar-10-batches-bin是我们下载的数据的文件夹位置
        reshaped_image = inputs_origin('cifar10_data/cifar-10-batches-bin')
        threads = tf.train.start_queue_runners(sess=sess)  # 开始填充队列        
        sess.run(tf.global_variables_initializer())  # 初始化epoch变量
        if not os.path.exists('cifar10_data/raw/'):
            os.makedirs('cifar10_data/raw/')
        # 保存30张图片
        for i in range(30):
            # 每次sess.run(reshaped_image)，都会取出一张图片
            image_array = sess.run(reshaped_image)  # 取出一张图片            
            scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg' % i)  # 将图片保存
