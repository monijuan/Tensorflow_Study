CSDN：https://blog.csdn.net/qq_34451909/article/details/108484669




## 一、概念

- 内容损失和风格损失

  - 内容损失（Content Loss）$L_{content}$描述原始图像和生成图像在 **内容** 上的差异；
  - 风格损失（Style Loss）Gram矩阵（卷积层特征）描述原始图片中的 **风格** ；

- 用途

  - 利用内容损失还原图像内容；

  - 利用封给说你是还原图像风格；

- 风格迁移：**还原图像的时候还原令一张图像的风格**。

- 原始图像风格迁移 对比 快速图像风格

  - 原始：

    $L_{total}(\overrightarrow{p},\overrightarrow{a},\overrightarrow{x})$ 衡量 $\overrightarrow{x}$ 是否成功组合了  $\overrightarrow{p}$ 和 $\overrightarrow{a}$ 的风格，以 $L_{total}$ 为目标进行梯度下降迭代 $\overrightarrow{x}$ ，速度慢。

  - 快速：

    使用神经网络直接生成 $\overrightarrow{x}$ ，速度快。
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200909104635317.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

  | 类型 | 损失定义                                        | 是否需要训练新网络                             | 生成图像的方法                       |
  | ---- | ----------------------------------------------- | ---------------------------------------------- | ------------------------------------ |
  | 原始 | 内容损失 $L_{content}$<br>风格损失 $L_{style}$  | 否，只需要预训练好的VGGNet                     | 利用损失，通过梯度下降计算合适的图像 |
  | 快速 | 内容损失 $L_{content}$<br/>风格损失 $L_{style}$ | 是，除了预训练好的VGGNet还需要训练图像生成网络 | 利用训练好的图像生成网络直接生成     |





## 二、实现快速风格迁移

### 1.下载训练好的图像生成网络



百度云链接：https://pan.baidu.com/s/11t5Vs3GHryyF1EHwow0dkA 
提取码：amka

新建一个文件夹`models`，把七个model放进去

```
cubist.ckpt-done
denoised_starry.ckpt-done
feathers.ckpt-done
mosaic.ckpt-done
scream.ckpt-done
udnie.ckpt-done
wave.ckpt-done
```



### 2.修改代码

eval.py：

```python
def main():
	#输出res的时候记录模型和原图的名字
    name_model = FLAGS.model_file.split('/')[-1].split('.')[0]
    name_img = FLAGS.image_file.split('/')[-1].split('.')[0]
	'''
	'''
    generated_file = 'generated/res_[%s]_[%s].jpg' % (name_model, name_img)
```


model.py：

```python
def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        # kernel 是float类型的，在新版本的tensorflow需要np.int()一下才行
        x_padded = tf.pad(x, [[0, 0], [np.int(kernel / 2), np.int(kernel / 2)], [np.int(kernel / 2), np.int(kernel / 2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')
```

### 3.生成风格图像


```
python eval.py --model_file models/wave.ckpt-done --image_file img/test.jpg
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200909121604407.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/202009091210278.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

例子中还有四个原始图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200909123557455.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
其中风格分别为：

 - cubist 

   <img src="https://img-blog.csdnimg.cn/20200909123759315.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />


 - denoised_starry 

   <img src="https://img-blog.csdnimg.cn/20200909123934574.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />


 - feathers 

   <img src="https://img-blog.csdnimg.cn/20200909123943692.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

 - mosaic 

   <img src="https://img-blog.csdnimg.cn/20200909123947804.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />





 - scream 

<img src="https://img-blog.csdnimg.cn/20200909123953893.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom: 25%;" />


 - udnie 

   <img src="https://img-blog.csdnimg.cn/20200909124006309.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom: 33%;" />



 - wave

   <img src="https://img-blog.csdnimg.cn/20200909124009932.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />



这里列出了五张图片在七种模型下的所有结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200909124126245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




```
python eval.py --model_file models/wave.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/wave.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/wave.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/wave.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/wave.ckpt-done --image_file img/test5.jpg

python eval.py --model_file models/cubist.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/cubist.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/cubist.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/cubist.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/cubist.ckpt-done --image_file img/test5.jpg

python eval.py --model_file models/denoised_starry.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/denoised_starry.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/denoised_starry.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/denoised_starry.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/denoised_starry.ckpt-done --image_file img/test5.jpg

python eval.py --model_file models/feathers.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/feathers.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/feathers.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/feathers.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/feathers.ckpt-done --image_file img/test5.jpg

python eval.py --model_file models/mosaic.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/mosaic.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/mosaic.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/mosaic.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/mosaic.ckpt-done --image_file img/test5.jpg

python eval.py --model_file models/scream.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/scream.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/scream.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/scream.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/scream.ckpt-done --image_file img/test5.jpg

python eval.py --model_file models/udnie.ckpt-done --image_file img/test1.jpg
python eval.py --model_file models/udnie.ckpt-done --image_file img/test2.jpg
python eval.py --model_file models/udnie.ckpt-done --image_file img/test3.jpg
python eval.py --model_file models/udnie.ckpt-done --image_file img/test4.jpg
python eval.py --model_file models/udnie.ckpt-done --image_file img/test5.jpg
```



## 三、训练自己的模型

### 1.下载预训练模型和数据集

- 下载VGG16模型

  地址： http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 

  百度云链接：https://pan.baidu.com/s/11t5Vs3GHryyF1EHwow0dkA 
  提取码：amka

  新建文件夹 `pretrained` ，把 `vgg_16.ckpt` 放进去

  

- 下载COCO数据集，有12.6G

  地址： http://msvocds.blob.core.windows.net/coco2014/train2014.zip

  解压放到文件夹 `train2014`

  Linux可以不用移动，直接建立连接：

  ```
  ln -s ＜到 train2014 文件路径> train2014
  ```



### 2.训练wave模型

- 训练模型：

```
python train.py -c conf/wave.yml
```

​	其中 `conf/wave.yml` 是配置文件：

```yml
## Basic configuration
style_image: img/wave.jpg # 输入原始风格图像的文件名
naming: "wave" # 风格/模型的名字
model_path: models  # checkpoint 和 events 文件保存的根目录

## Weight of the loss
content_weight: 1.0  # 内容损失的权重
style_weight: 220.0  # 风格损失的权重
tv_weight: 0.0  # total variation loss

## The size, the iter number to run
image_size: 256   # 原始图片的大小
batch_size: 4     # 一次 batch 的样本数
epoch: 2  # epoch 的运行次数

## Loss Network
loss_model: "vgg_16" # 使用 vgg_16 的模型
content_layers:  # 使用 conv3_3 定义内容损失
  - "vgg_16/conv3/conv3_3"
style_layers:  # 使用 conv1_2、conv2_2、conv3_3、conv4_3 定义风格损失
  - "vgg_16/conv1/conv1_2"
  - "vgg_16/conv2/conv2_2"
  - "vgg_16/conv3/conv3_3"
  - "vgg_16/conv4/conv4_3"
checkpoint_exclude_scopes: "vgg_16/fc"  # 只需要卷积层，不需要fc层
loss_model_file: "pretrained/vgg_16.ckpt"  # 预训练对应的位置
```

​	读者如果希望训练新的“风格”，可以选取一张风格图片，并编写新的 yml 配置文件。其中，需要把 style_image 修改为新图片所在的位置，并修改对应的 naming 。 这样就可以进行训练了。最后可以使用训练完成的 checkpoint 生成图片。
​	在训练 、新的“风格”时，有可能会需要调整各个损失之间的权重。

- 查看训练情况：

```
tensorboard --logdir models/wave/
```

- 可能需要调整权重
  - content_weight 过大，图像会更接近原始图像
  - style_weight 过大，图像的风格更接近原始图像

但是因为数据集太大，一直都下不下来，这里就先放弃了。

有百度云资源的可以发一下！感激涕零！



### 3.实现细节

#### 1）生成网络的定义

 models.py

```python
def net(image, training):
    # 图片加上一圈边框，消除边缘效应
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    # 三层卷积层
    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
    
    # 仿照 ResNet 定义一些跳过的链接
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    # print(res5.get_shape())

    # 定义反卷积，先放大再卷积可以消除噪声
    with tf.variable_scope('deconv1'):
        # deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2))) #不直接转置
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        # deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2))) #不直接转置
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        # deconv_test = relu(instance_norm(conv2d(deconv2, 32, 32, 2, 1))) #不直接转置
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    # 经过了 tanh 激活，将[-1,1]缩放到[0,255]像素值范围
    y = (deconv3 + 1) * 127.5

    # 去除边框
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))
    
    return y
```

#### 2）生成网络的引用

train.py

```python
"""Build Network"""
# 损失网络
network_fn = nets_factory.get_network_fn(
	FLAGS.loss_model,
	num_classes=1,
	is_training=False) # 不需要对损失函数训练

# 图像和与处理函数，不需要训练
image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
	FLAGS.loss_model, is_training=False)

# 读入训练图像
processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
								'train2014/', image_preprocessing_fn, epochs=FLAGS.epoch)

# 引用生成网络，生成图像，这里需要训练
generated = model.net(processed_images, training=True)

# 将生成图像使用 image_preprocessing_fn 处理
processed_generated = [
	image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
	for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
]
processed_generated = tf.stack(processed_generated)

# 将原始图和生成图送到损失网络，加快速度
_, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

# Log the structure of loss network
tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
for key in endpoints_dict:
	tf.logging.info(key)
```

#### 3）内容损失

loss.py

```python
# endpoints_dict 是损失网络各层的计算结果
# content_layers 是定义使用哪些层的差距计算损失
def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)# 把生成图像分开
        size = tf.size(generated_images)
        # 生成图片的激活 与 原始图片的激活 的L^2距离
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  
    return content_loss
```



#### 4）风格损失

loss.py

```python
# endpoints_dict 是损失网络各层的计算结果
# style_features_t 是利用原始的风格图片计算的层的激活
# style_layers 是定义使用哪些层计算损失
def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {} # 为tensorboard服务的
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0) # 计算风格损失
        size = tf.size(generated_images)
        # 计算 Gram 矩阵， L^2（Loss）
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary
```

#### 5）调用损失

train.py

```python
"""Build Losses"""
# 定义内容损失
content_loss = losses.content_loss(endpoints_dict, FLAGS.content_layers)
# 定义风格损失
style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
# 定义tv损失，但是因为tv_weight=0，所以没用
tv_loss = losses.total_variation_loss(generated)  
# 总损失
loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss
```

####　6）定义训练、保存的变量

train.py

```python
"""Prepare to Train"""
global_step = tf.Variable(0, name="global_step", trainable=False)

variable_to_train = [] # 找出需要训练的变量，append进去
for variable in tf.trainable_variables():
	if not(variable.name.startswith(FLAGS.loss_model)):
		variable_to_train.append(variable)
# 定义， global_step=global_step 不会训练损失网络
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

variables_to_restore = [] # 找出需要保存的变量，append进去
for v in tf.global_variables():
	if not(v.name.startswith(FLAGS.loss_model)):
		variables_to_restore.append(v)
# 定义，只保存 variables_to_restore
saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
```







## 四、原书md.

**图像风格迁移**

**7.2.1 使用预训练模型**

在chapter_7_data/ 中提供了7 个预训练模型： wave.ckpt-done 、cubist.ckpt-done、denoised_starry.ckpt-done、mosaic.ckpt-done、scream.ckpt-done、feathers.ckpt-done。

以wave.ckpt-done的为例，在chapter_7/中新建一个models 文件
夹， 然后把wave.ckpt-done复制到这个文件夹下，运行命令：

```
python eval.py --model_file models/wave.ckpt-done --image_file img/test.jpg
```

成功风格化的图像会被写到generated/res.jpg。

**7.2.2 训练自己的模型**

准备工作：

- 在地址http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 下载VGG16模型，将下载到的压缩包解压后会得到一个vgg16.ckpt 文件。在chapter_7/中新建一个文件夹pretrained，并将vgg16.ckpt 复制到pretrained 文件夹中。最后的文件路径是pretrained/vgg16.ckpt。这个vgg16.ckpt文件在chapter_7_data/中也有提供。

- 在地址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下载COCO数据集。将该数据集解压后会得到一个train2014 文件夹，其中应该含有大量jpg 格式的图片。在chapter_7中建立到这个文件夹的符号链接：
```
ln –s <到train2014 文件夹的路径> train2014
```

训练wave模型：
```
python train.py -c conf/wave.yml
```

打开TensorBoard：
```
tensorboard --logdir models/wave/
```

训练中保存的模型在文件夹models/wave/中。

**拓展阅读**

- 关于第7.1.1 节中介绍的原始的图像风格迁移算法，可以参考论文A Neural Algorithm of Artistic Style 进一步了解其细节。关于第7.1.2 节 中介绍的快速风格迁移， 可以参考论文Perceptual Losses for Real-Time Style Transfer and Super-Resolution。

- 在训练模型的过程中，用Instance Normalization 代替了常用的Batch Normalization，这可以提高模型生成的图片质量。关于Instance Normalization 的细节，可以参考论文Instance Normalization: The Missing Ingredient for Fast Stylization。

- 尽管快速迁移可以在GPU 下实时生成风格化图片，但是它还有一个 很大的局限性，即需要事先为每一种风格训练单独的模型。论文 Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 中提出了一种“Arbitrary Style Transfer”算法，可以 为任意风格实时生成风格化图片，读者可以参考该论文了解其实现 细节。
