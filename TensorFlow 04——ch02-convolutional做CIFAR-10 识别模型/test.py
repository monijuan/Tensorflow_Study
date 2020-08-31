# coding:utf-8

import tensorflow as tf
import os
if not os.path.exists('read'):
    os.makedirs('read/')

# 新建一个Session
with tf.Session() as sess:
    filename = ['A.jpg', 'B.jpg', 'C.jpg']  # 读取三幅图片    
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)  # 文件名队列
    reader = tf.WholeFileReader()  # reader从文件名队列中读数据
    key, value = reader.read(filename_queue)
    tf.local_variables_initializer().run()  # 初始化epoch变量
    threads = tf.train.start_queue_runners(sess=sess)  # 开始填充队列
    i = 0
    while True:
        i += 1
        # 获取图片数据并保存
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
# 程序最后会抛出一个OutOfRangeError，这是epoch跑完，队列关闭的标志
