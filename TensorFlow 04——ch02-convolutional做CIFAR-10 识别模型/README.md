

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091928148.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

@[TOC](目录)
## 一、实验过程

**1. 下载CIFAR-10 数据**

```
python cifar10_download.py
```

**2. 测试TensorFlow 的数据读取机制**

```
python test.py
```

**3. 将CIFAR-10 数据集保存为图片形式**

```
python cifar10_extract.py
```

**4. 训练模型**

```
python cifar10_train.py --train_dir cifar10_train/ --data_dir cifar10_data/
```

**5. 在TensorFlow 中查看训练进度**

```
tensorboard --logdir cifar10_train/
```

**6. 测试模型效果**

```
python cifar10_eval.py --data_dir cifar10_data/ --eval_dir cifar10_eval/ --checkpoint_dir cifar10_train/
```

**7. 使用TensorBoard查看性能验证情况：**

```
tensorboard --logdir cifar10_eval/ --port 6007
```



## 二、CIFAR-10说明

该数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

## 三、正文 

### 数据下载

如果没有数据的话，就会下载到`cifar10_data/`目录下，之后直接读取就行。

如果家里下可能会比较慢，尤其是电信，可以挂学校VPN会快许多。

> cifar10_download.py

```python
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
cifar10.maybe_download_and_extract() # 这个函数放在最后

```

数据下载完成：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020083108571385.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 数据测试

> test.py

测试 TensorFlow 读取机制，报错是正常的退出，可以在while循环里面print(i)看到循环情况

```python
# -*- coding: utf-8 -*-
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

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020083108581941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



> cifar10_extract.py

读取原始数据集， 并把它们保存为原始的图片。

```python
# -*- coding: utf-8 -*-
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

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020083109060410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 开始训练模型

一开始不太懂，傻乎乎的一直跑模型，跑了两个小时还没有结束，后来发现是边跑边存结果，需要手动停止。

除此以外，训练的同时需要测试（评估）模型，也就是`cifar10_eval.py`，后面会讲到。

> cifar10_train.py

这是训练模型的接口，真正训练的过程是在`cifar10.py`里面的`inference(images)`

```python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

```

> cifar10.py/inference()

模型的结构

```python
def inference(images):
  """Build the CIFAR-10 model.
  输入是图片的tensor，输出是图片的类别
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # 第一层卷积层
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)  # 将输出报告到TensorBoard

  # 第一层卷积层的池化
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # 局部响应归一化层（LRN），现在的模型大多不采用 
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # 第二层卷积层
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)  # 将输出报告到TensorBoard

  # 局部响应归一化层（LRN）
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # 第二层卷积层的池化
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # 全连接层1
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1]) # reshape方便全连接
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)  # 输出是relu(Wx+b) 
    _activation_summary(local3)  # 将输出报告到TensorBoard

  # 全连接层2
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)  # 输出是relu(Wx+b)
    _activation_summary(local4)  # 将输出报告到TensorBoard

  # 全连接 + Softmax分类，但是不显示Softmax，输出softmax_linear
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)  # 将输出报告到TensorBoard

  return softmax_linear

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020083109141831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091424406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091433895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 查看训练情况

代码中也看到了，结果都在TensorBoard中，这也是第一次接触，觉得好强大！

在目录下起一个cmd，运行`tensorboard --logdir cifar10_train/`查看训练的情况。

与此同时要开始测试模型，不然TensorBoard只保留五个数据结果，前面的会被后面的替换，同时进行训练才能看到模型的进步。第一次我是训练到两万步了才开始测试，结果始终都在80%+。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091826451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091855486.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091859941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091907453.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091911385.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200831091915350.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)







## 四、全部代码

[https://github.com/MONI-JUAN/Tensorflow_Study](https://github.com/MONI-JUAN/Tensorflow_Study) 