CSDN：https://blog.csdn.net/qq_34451909/article/details/108547934



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912135357803.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



## 一、CycleGAN 原理

### 1.损失函数

CycleGAN 与 pix2pix 的不同点在于，可以利用不成对数据训练出从 X 空间到 Y 空间的映射。例如用大量照片和油画图片可以学习到如何把照片转成油画。

算法的目标是学习从空间 X 到空间 Y 的映射，设这个映射为 F。 对应着 GAN 中的生成器， F可以将 X 中的图片 x 转为 Y 中的图片 F(x）。对于生成的图片，还需要 GAN 中的判别器来判别它是否为真实图片，由此构成对抗生成网络。设这个判别器为Dy。根据生成器和判别器可以构造 GAN 的损失了 ，该损失和原始 GAN 中的损失的形式是相同的：
$$
L_{GAN}(F,D_{y},X,Y)=E_{y \sim P_{\text {data}}(y)}[\ln D_{Y}(y)]+E_{x \sim p_{\text {data}}(x)}[\ln (1-D_{Y}(F(x)))]
$$
但只使用这一个损失是无法进行训练的。原因在于没再成对数据，映射 F可以将所有 x 都映射为 Y 空间中的同一张图片，使损失无效化。对此，作者又提出了所谓的“循环一致性损失”（ cycle consistency loss ）。让再假设一个映射 G，它可以将 Y 空间中的图片y 转换为 X 中的图片 G(y）。CycleGAN  同时学习 F 和 G 两个映射，并要求 $F(G(y))\approx y$，以及$G(F(x))\approx x$。 也是说，**将 x 的图片转换到 Y 空间后，应该还可以转换回来**。这样可以杜绝模型把所高 X 的图片都转换为 Y 空间中的同一张图片。

根据 $F(G(y))\approx y$ 和 $G(F(x))\approx x$ ，循环一致性损失定义为：
$$
L_{cyc}(F,G,X,Y)=E_{x \sim p_{\text {data}}(x)}[{\left\| G(F(x))- x \right\|_1}]+E_{y \sim p_{\text {data}}(y)}[{\left\| F(G(y))- y \right\|_2}]
$$
同时，为 G 也引入一个判别器Dx，由此可以同样定义一个 GAN 损失$L_{GAN}(G,D_{x},X,Y)$， ，最终的损失由三部分组成：
$$
L=L_{GAN}(F,D_{y},X,Y)+L_{GAN}(F,D_{x},X,Y) +\lambda L_{cyc}(F,G,X,Y)
$$
CycleGAN 的主要想法是上述的“循环一致性损失”，利用这个损失，可以巧妙地处理 X 空间和 Y 空间训练样本不一一配对的问题。



### 2.定义

#### 模型

```python
def model(self):
	# 读入X空间和Y空间的数据，保存到 x 和 y 中
	X_reader = Reader(self.X_train_file, name='X',
		image_size=self.image_size, batch_size=self.batch_size)
	Y_reader = Reader(self.Y_train_file, name='Y',
		image_size=self.image_size, batch_size=self.batch_size)
	x = X_reader.feed()
	y = Y_reader.feed()

	# 定义循环一致性损失：G:X->Y,F:Y->X
	cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

	# G: X -> Y
	fake_y = self.G(x)
	G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan) # G生成图片的loss
	G_loss =  G_gan_loss + cycle_loss
	D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan) # Y空间判别器的损失

	# F: Y -> X
	fake_x = self.F(y)
	F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan) # F生成图片的loss
	F_loss = F_gan_loss + cycle_loss
	D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan) # Y空间的损失
```

#### 循环一致性损失

```python
def cycle_consistency_loss(self, G, F, x, y):
    """
    cycle consistency loss (L1 norm)
    循环一致性损失
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss
```

#### 生成器损失

```python
def generator_loss(self, D, fake_y, use_lsgan=True):
  """  
  fool discriminator into believing that G(x) is real
  生成器损失
  """
  if use_lsgan:
    # use mean squared error
    loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
  else:
    # heuristic, non-saturating loss
    loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
  return loss
```

#### 判别器损失

```python
def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
  """ Note: default: D(y).shape == (batch_size,5,5,1),
                      fake_buffer_size=50, batch_size=1
  Args:
    G: generator object
    D: discriminator object
    y: 4D tensor (batch_size, image_size, image_size, 3)
  Returns:
    loss: scalar
  判别器损失
  """
  if use_lsgan:
    # use mean squared error
    error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
    error_fake = tf.reduce_mean(tf.square(D(fake_y)))
  else:
    # use cross entropy
    error_real = -tf.reduce_mean(ops.safe_log(D(y)))
    error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
  loss = (error_real + error_fake) / 2
  return loss
```



#### 调用损失

optimize():

```python
# G_loss、F_loss、D_Y_loss、D_X_loss
G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
return tf.no_op(name='optimizers')
```





## 二、苹果到橘子

### 1.下载数据集

下载数据集 `apple2orange.zip`

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912123954976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



网页：https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

或者用脚本：

```
bash download_dataset.sh apple2orange
```

或者百度云：

链接：https://pan.baidu.com/s/1U78OTOAi0bUuJ-bGhtWZ5A 
提取码：1r1f 



在目录创建 `data` 文件夹，把 `apple2orange` 放进去



### 2.转换成 tfrecords 格式

```
python build_data.py \
  --X_input_dir data/apple2orange/trainA \
  --Y_input_dir data/apple2orange/trainB \
  --X_output_file data/tfrecords/apple.tfrecords \
  --Y_output_file data/tfrecords/orange.tfrecords
```

```
python build_data.py --X_input_dir data/apple2orange/trainA --Y_input_dir data/apple2orange/trainB --X_output_file data/tfrecords/apple.tfrecords --Y_output_file data/tfrecords/orange.tfrecords
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912124000100.png#pic_center)




### 3.训练模型

```
python train.py \
  --X data/tfrecords/apple.tfrecords \
  --Y data/tfrecords/orange.tfrecords \
  --image_size 256
```

```
python train.py --X data/tfrecords/apple.tfrecords --Y data/tfrecords/orange.tfrecords --image_size 256
```



### 4.查看训练情况

后面的路径改成自己的

```
tensorboard --logdir checkpoints/20200912-1241
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091213521691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
花了两个小时才step100，运行之前没发现step每隔100存一下，导致模型一直看不到情况，其实可以设置小一点。

```python
if step % 100 == 0:
     train_writer.add_summary(summary, step)
     train_writer.flush()

if step % 100 == 0:
  logging.info('-----------Step %d:-------------' % step)
  logging.info('  G_loss   : {}'.format(G_loss_val))
  logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
  logging.info('  F_loss   : {}'.format(F_loss_val))
  logging.info('  D_X_loss : {}'.format(D_X_loss_val))

if step % 10000 == 0:
  save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
  logging.info("Model saved in file: %s" % save_path)

step += 1
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912135329313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912135335732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912135338845.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912135352452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 5.导出模型

后面的路径改成自己的

```
python export_graph.py \
  --checkpoint_dir checkpoints/20200912-1241 \
  --XtoY_model apple2orange.pb \
  --YtoX_model orange2apple.pb \
  --image_size 256
```

```
python export_graph.py --checkpoint_dir checkpoints/20200912-1241 --XtoY_model apple2orange.pb --YtoX_model orange2apple.pb --image_size 256
```

会在 ` 生成两个` 文件夹生成 `apple2orange.pb` 和 `orange2apple.pb`两个模型



### 6.测试模型



```
python inference.py \
  --model pretrained/apple2orange.pb \
  --input data/apple2orange/testA/n07740461_1661.jpg \
  --output data/apple2orange/output_sample.jpg \
  --image_size 256
```

```
python inference.py --model pretrained/apple2orange.pb --input data/apple2orange/testA/n07740461_1661.jpg --output data/apple2orange/output_sample.jpg --image_size 256
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091213583791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

刚刚100步的结果。。。[老爷爷看手机]







## 三、男性到女性

步骤比较类似，其他的数据集也是只要分到两个文件夹，一个是X一个是Y即可。



### 1.下载数据集



链接：https://pan.baidu.com/s/1U78OTOAi0bUuJ-bGhtWZ5A 
提取码：1r1f 





### 2.转换成 tfrecords 格式

```
python build_data.py \
  --X_input_dir data/man2woman/a_resized/ \
  --Y_input_dir data/man2woman/b_resized/ \
  --X_output_file data/man2woman/man.tfrecords \
  --Y_output_file data/man2woman/woman.tfrecords
```

```
python build_data.py --X_input_dir data/man2woman/a_resized/ --Y_input_dir data/man2woman/b_resized/ --X_output_file data/man2woman/man.tfrecords --Y_output_file data/man2woman/woman.tfrecords
```




### 3.训练模型

```
python train.py \
  --X data/man2woman/man.tfrecords \
  --Y data/man2woman/woman.tfrecords \
  --image_size 256
```

```
python train.py --X data/man2woman/man.tfrecords --Y data/man2woman/woman.tfrecords --image_size 256
```



### 4.查看训练情况

后面的路径改成自己的

```
tensorboard --logdir checkpoints/xxxxxxxxxxx
```



## 四、书上的一些效果展示
![请添加图片描述](https://img-blog.csdnimg.cn/20200912140058656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![请添加图片描述](https://img-blog.csdnimg.cn/20200912140058714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200912140139788.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



## 五、原书md

11. CycleGAN与非配对图像转换

本节的程序来源于项目https://github.com/vanhuyz/CycleGAN-TensorFlow ，并做了细微修改。

**11.2.1 下载数据集并训练**

下载一个事先准备好的数据集：

```
bash download_dataset.sh apple2orange
```

将图片转换成tfrecords格式：

```
python build_data.py \
  --X_input_dir data/apple2orange/trainA \
  --Y_input_dir data/apple2orange/trainB \
  --X_output_file data/tfrecords/apple.tfrecords \
  --Y_output_file data/tfrecords/orange.tfrecords
```

训练模型：

```
python train.py \
  --X data/tfrecords/apple.tfrecords \
  --Y data/tfrecords/orange.tfrecords \
  --image_size 256
```

打开TensorBoard(需要将--logdir checkpoints/20170715-1622 中的目录替换为自己机器中的对应目录)：

```
tensorboard --logdir checkpoints/20170715-1622
```

导出模型(同样要注意将20170715-1622 替换为自己机器中的对应目录)：

```
python export_graph.py \
  --checkpoint_dir checkpoints/20170715-1622 \
  --XtoY_model apple2orange.pb \
  --YtoX_model orange2apple.pb \
  --image_size 256
```

使用测试集中的图片进行测试：

```
python inference.py \
  --model pretrained/apple2orange.pb \
  --input data/apple2orange/testA/n07740461_1661.jpg \
  --output data/apple2orange/output_sample.jpg \
  --image_size 256
```

转换生成的图片保存在data/apple2orange/output_sample. jpg。

**11.2.2 使用自己的数据进行训练**

在chapter_11_data/中，事先提供了一个数据集man2woman.zip。，解压后共包含两个文件夹：a_resized 和b_resized。将它们放到目录~/datasets/man2woman/下。使用下面的命令将数据集转换为tfrecords：

```
python build_data.py \
  --X_input_dir ~/datasets/man2woman/a_resized/ \
  --Y_input_dir ~/datasets/man2woman/b_resized/ \
  --X_output_file ~/datasets/man2woman/man.tfrecords \
  --Y_output_file ~/datasets/man2woman/woman.tfrecords
```

训练：

```
python train.py \
  --X ~/datasets/man2woman/man.tfrecords \
  --Y ~/datasets/man2woman/woman.tfrecords \
  --image_size 256
```

导出模型和测试图片的指令可参考11.2.1。

拓展阅读

- 本章主要讲了模型CycleGAN ， 读者可以参考论文Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 了解更多细节

- CycleGAN 不需要成对数据就可以训练，具有较强的通用性，由此产生了大量有创意的应用，例如男女互换（即本章所介绍的）、猫狗互换、利用手绘地图还原古代城市等。可以参考https://zhuanlan.zhihu.com/p/28342644 以及https://junyanz.github.io/CycleGAN/ 了解这些有趣的实验

- CycleGAN 可以将将某一类图片转换成另外一类图片。如果想要把一张图片转换为另外K类图片，就需要训练K个CycleGAN，这是比较麻烦的。对此，一种名为StarGAN 的方法改进了CycleGAN， 可以只用一个模型完成K类图片的转换，有兴趣的读者可以参阅其论文StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation。

- 如果读者还想学习更多和GAN 相关的模型， 可以参考 https://github.com/hindupuravinash/the-gan-zoo 。这里列举了迄今几乎所有的名字中带有“GAN”的模型和相应的论文。