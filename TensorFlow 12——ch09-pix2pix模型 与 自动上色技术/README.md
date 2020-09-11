CSDN：https://blog.csdn.net/qq_34451909/article/details/108527128

@[TOC](目录)

## 一、概念



### 1.GAN缺陷

使用 GAN 可以对样本进行无监督学习，然后生成全新的样本。但虽然能生成新的样本，却无法确切控制新样本的类型。
如使用 GAN 生成 MNIST 数字，虽然可以生成数字，但生成的结果是随机 的，因为是根据输入的随机躁声生成图片，没再办法控制模型生成的具体数字。

### 2.cGAN

cGAN（ Conditional Generative Adversarial Networks），条件对抗生成网络，它为生成器、判别器都额外加入了一个条件y，这个条件实际是希望生成的标签。生成器G 必须要生成和条件y匹配的样本，判别器不仅要判别图像是否真实，还要判别图像和条件y 是否匹配。 

### 3.GAN 和 cGAN对比

- GAN

  - 生成器 G，输入为一个噪声 z，输出一个图像 G(z)

  - 判别器 D，输入为一个图像 x，输出该图像为真实的概率D(x)

  - 优化目标：
    $$
    V(D, G)=E_{x \sim P_{\text {data }}}[\ln D(x)]+E_{z \sim p_{z}(z)}[\ln (1-D(G(z)))]
    $$

- cGAN

  - 生成器 G，输入为一个噪声 z， 一个条件y，输出一个图像 G(z|y)

  - 判别器 D，输入为一个图像 x， 一个条件y，输出该图像在该条件下的概率D(x|y)

  - 优化目标：

  $$
  V(D, G)=E_{x \sim P_{\text {data }}}[\ln D(x|y)]+E_{z \sim p_{z}(z)}[\ln (1-D(G(z|y)))]
  $$

- 以 MNIST 为例的cGAN

  - 生成器 G，输入为一个噪声 z， 一个数字标签y（0 ~ 9），输出一个符合标签的图像 G(z|y)
  - 判别器 D，输入为一个图像 x， 一个数字标签y，输出该图像和数字符合的概率D(x|y)
  - 在训练完成后，向 G 输入某个数字标签和噪声，可以生成对应数字的图像

### 4.应用

- 将街景的标注图像变为真实照片。
- 将建筑标注图像转换为照片。
- 将卫星图像转换为地圄。
- 将白天的图片转换为夜晚的图片。
- 将边缘轮廓线转段为真实物体。

### 5.pix2pix 模型

pix2pix 和 cGAN 的结构类似，同样是由生成器 G、判别器 D 两个网络组成。 设要将 Y类型的图像转换为 X类型的图像， G、 D 的任务分别为 ：

- G 的输入是一个 Y 类图像 y  ，输出为生成图像 G(y）。 
- D 的输入为一个 X 类图像x，一个 Y 类图像y。D 需要判断 x 图像是否是真正的y对应的图像，并输出一个概率。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911113731600.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



## 二、pix2pix 



### 1.下载数据集

下载数据集`facades.rar`

用脚本：

```
python tools/download-dataset.py facades
```

或者百度云链接：[https://pan.baidu.com/s/1LhYMOCiHRr_qRryG4FcB6A](https://pan.baidu.com/s/1LhYMOCiHRr_qRryG4FcB6A) 
提取码：8x8b



### 2.查看数据集

完成后 `facades` 文件夹中有三个文件夹，分别是 `test`、`train`、`val`。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911112737580.png#pic_center)
随便打开一张：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911090914238.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



图片的左边称为 A类，右边为 B类。训练时可以指定是【A类翻译为B类】还是【B类翻译为A类】。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911134247794.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


在 Facades 数据集中，希望程序能从图像的标注出发，生成真实的建筑图像。因此，根据图像的排列顺序，应该指定将 B 类图像转换成 A 类图像。



- cGAN , 与原始 GAN 使用随机躁声生成样本不同， cGAN 可以根据指定标签生成样 本。 

- pix2pix 模型，可以看作是 cGAN 的一种特殊形式。 

### 3.训练模型

```
python pix2pix.py \
  --mode train \ 				# 训练
  --output_dir facades_train \ 	# 输出路径
  --max_epochs 200 \ 			# 最大epoch数
  --input_dir facades/train \	# 数据集路径
  --which_direction BtoA		# 学习从B类图像转换为A类
```

```
python pix2pix.py --mode train --output_dir facades_train --max_epochs 200 --input_dir facades/train --which_direction BtoA
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911153309619.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911121354151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


### 4.查看训练情况

```
tensorboard --logdir facades_train 
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911153257950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911153323233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911153326492.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




### 5.测试模型

```
python pix2pix.py \
  --mode test \					# 测试
  --output_dir facades_test \	# 输出路径
  --input_dir facades/val \		# 输入路径
  --checkpoint facades_train	# 保存模型
```

```
python pix2pix.py --mode test --output_dir facades_test --input_dir facades/val --checkpoint facades_train
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200911153451955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

训练了三四个小时的结果，还凑合。

### 6.其他数据集

百度云可以下两个数据集，也就是食物和动漫的

百度云链接：[https://pan.baidu.com/s/1LhYMOCiHRr_qRryG4FcB6A](https://pan.baidu.com/s/1LhYMOCiHRr_qRryG4FcB6A)  
提取码：8x8b

除此以外还有几个可以用脚本下载：

```
python tools/download-dataset.py facades
python tools/download-dataset.py cityscapes 
python tools/download-dataset.py maps 
python tools/download-dataset.py edges2shoes
python tools/download-dataset.py edges2handbags
```



每次训练时间太长了，这两个就没有跑，有空的可以试试



#### a)为食物图片上色

在chapter_9_data/中提供的food_resized.zip 文件解压到目录~/datasets/colorlization/下，最终形成的文件 夹结构应该是：

```
~/datasets
  colorlization/
    food_resized/
      train/
      val/
```

训练命令：

```
python pix2pix.py \
--mode train \
--output_dir colorlization_food \
--max_epochs 70 \
--input_dir ~/datasets/colorlization/food_resized/train \
--lab_colorization
```

测试命令：

```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_food_test \
  --input_dir ~/datasets/colorlization/food_resized/val \
  --checkpoint colorlization_food
```

结果在colorlization_food_test文件夹中。

#### b)为动漫图片上色

将chapter_9_data/中提供的动漫图像数据集anime_reized.zip 解压到~/datasets/colorlization/目录下，形成的文件夹结构为：

```
~/datasets
  colorlization/
    anime_resized/
      train/
      val/
```

训练命令：

```
python pix2pix.py \
  --mode train \
  --output_dir colorlization_anime \
  --max_epochs 5 \
  --input_dir ~/datasets/colorlization/anime_resized/train \
  --lab_colorization
```

测试命令：

```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_anime_test \
  --input_dir ~/datasets/colorlization/anime_resized/val \
  --checkpoint colorlization_anime
```

结果在colorlization_anime_test文件夹中。

## 三、原书md

pix2pix模型与自动上色技术

本节的程序来源于项目 https://github.com/affinelayer/pix2pix-tensorflow 。

**9.3.1 执行已有的数据集**

下载Facades数据集：
```
python tools/download-dataset.py facades
```

训练：
```
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
```

测试：
```
python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir facades/val \
  --checkpoint facades_train
```

结果在facades_test文件夹中。


**9.4.1 为食物图片上色**


在chapter_9_data/中提供的food_resized.zip 文件解压到目录~/datasets/colorlization/下，最终形成的文件
夹结构应该是：

```
~/datasets
  colorlization/
    food_resized/
      train/
      val/
```

训练命令：
```
python pix2pix.py \
--mode train \
--output_dir colorlization_food \
--max_epochs 70 \
--input_dir ~/datasets/colorlization/food_resized/train \
--lab_colorization
```

测试命令：
```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_food_test \
  --input_dir ~/datasets/colorlization/food_resized/val \
  --checkpoint colorlization_food
```

结果在colorlization_food_test文件夹中。

**9.4.2 为动漫图片上色**

将chapter_9_data/中提供的动漫图像数据集anime_reized.zip 解压到~/datasets/colorlization/目录下，形成的文件夹结构为：

```
~/datasets
  colorlization/
    anime_resized/
      train/
      val/
```

训练命令：
```
python pix2pix.py \
  --mode train \
  --output_dir colorlization_anime \
  --max_epochs 5 \
  --input_dir ~/datasets/colorlization/anime_resized/train \
  --lab_colorization
```

测试命令：
```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_anime_test \
  --input_dir ~/datasets/colorlization/anime_resized/val \
  --checkpoint colorlization_anime
```

结果在colorlization_anime_test文件夹中。


#### 拓展阅读

- 本章主要讲了cGAN 和pix2pix 两个模型。读者可以参考它们的原始 论文Conditional Generative Adversarial Nets 和Image-to-Image Translation with Conditional Adversarial Networks 学习更多细节。

- 针对pix2pix 模型，这里有一个在线演示Demo，已经预训练好了多 种模型， 可以在浏览器中直接体验pix2pix 模型的效果： https://affinelayer.com/pixsrv/ 。
