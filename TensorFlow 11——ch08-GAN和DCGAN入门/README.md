![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910184136101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



CSDN：https://blog.csdn.net/qq_34451909/article/details/108511594



@[TOC](目录)

## 一、基本概念

GAN 的全称为 Generative Adversarial Networks，意为对抗生成网络。

DCGAN 将 GAN 的概念扩展到卷积神经网络中，可以生成质量较高的图片样本 。 

### 1.GAN 的原理

有两个网络，一个是生成网络G（Generator），一个是判别网络D（Discriminator）

- G：通过噪声z生成图片，记作 G(z) ；
- D：判断图片是不是”真实的“，输入的x，输出 D(x) 代表是真实图片的概率

**训练过程**：G尽量生成真实图片去欺骗D，D尽量区分G生成的图片和真实图片。

### 2.交叉熵损失

$$
V(D, G)=E_{x \sim P_{\text {data }}(x)}[\ln D(x)]+E_{z \sim p_{z}(z)}[\ln (1-D(G(z)))]
$$

- 左边x部分代表真实图片，右边G(z)是生成的图片；
- D(x) 和 D(G(z)) 都是判断的概率；
- 生成网络 G 希望 D(G(z)) 变大，V(D, G)越大越好；
- 判别网络 D 希望 D(x) 变大，V(D, G)越小越好；

### 3.DCGAN的原理

DCGAN 的全称是 Deep Convolutional Generative Adversarial Networks，即深度卷积对抗生成网络。从名字上来看，是在 GAN 的基础上增加深度卷积网络结构，专门生成图像样本。 

事实上，GAN 并没再对D、 G 的具体结构做出任何限制 。DCGAN 中的 D、 G 的含义以及损失都和原始 GAN 中完全一，但是它在 D 和 G 中采用了较为特殊的结构，以便对图片进行高效建模。

DCGAN 中 G 的网络结构：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910121646592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



- 不采用池化层，D中用补偿（stride）的卷积代替池化；
- 在 G、 D 中均使用 Batch Normalization 帮助模型收敛。 
- 在 G 中，激活函数除了最后一层都使用 ReLU 函数，而最后一层使用 tanh 函数。
- 在 D 中，激活函数都使用 Leaky ReLU 作为激活函数。

![请添加图片描述](https://img-blog.csdnimg.cn/20200910122735587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)






![请添加图片描述](https://img-blog.csdnimg.cn/20200910122735652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![请添加图片描述](https://img-blog.csdnimg.cn/20200910122735672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

## 二、生成MNIST图像

### 1.下载数据集

用脚本下载（可能会下载失败，我也不知道为什么每次都失败）

```
python download.py mnist
```

或者百度云

链接：[https://pan.baidu.com/s/1l-IHrXYvt4M8kj_C-Blklw](https://pan.baidu.com/s/1l-IHrXYvt4M8kj_C-Blklw) 
提取码：kgrw

这个数据集和chapter 01 的一样：https://blog.csdn.net/qq_34451909/article/details/108264641

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910132841750.png#pic_center)

### 2.训练



```
python main.py --dataset mnist --input_height=28 --output_height=28 --train
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910135636474.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910135640365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

### 3.训练结果

每过100步会保存一张当前训练情况的图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910135819501.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

对比一下 0_99 和 1_106，才训练了一千步左右，已经很有数字的样子了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910135946929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

看一下书中25个epoch，也就是2.5w步之后的图像：

<img src="https://img-blog.csdnimg.cn/2020091014014575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />









## 三、使用自己的数据集训练

### 1.下载数据集

faces.zip

链接：[https://pan.baidu.com/s/1l-IHrXYvt4M8kj_C-Blklw](https://pan.baidu.com/s/1l-IHrXYvt4M8kj_C-Blklw) 
提取码：kgrw

解压`faces.zip` ，把 `anime`放进 `data` 目录



### 2.训练模型

```
python main.py --input_height 96 --input_width 96 \ # 截取中心96*96
  --output_height 48 --output_width 48 \ # 缩放到48*48
  --dataset anime --crop -–train \ # 需要执行训练
  --epoch 300 --input_fname_pattern "*.jpg" # 找出所有.jpg训练
```

```
python main.py --input_height 96 --input_width 96 --output_height 48 --output_width 48 --dataset anime --crop -–train --epoch 300 --input_fname_pattern "*.jpg"
```

这已经是训练3.7小时候的结果了，电脑太渣了


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910184021132.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910184101625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910184119684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)









对比训练模型：

- 如果是 `mnist` 数据集：

  ```python
  if config.dataset == 'mnist':
  # Update D network
  _, summary_str = self.sess.run([d_optim, self.d_sum],
  feed_dict={ 
    self.inputs: batch_images,
    self.z: batch_z,
    self.y:batch_labels,
  })
  self.writer.add_summary(summary_str, counter)
  
  # Update G network
  _, summary_str = self.sess.run([g_optim, self.g_sum],
  feed_dict={
    self.z: batch_z, 
    self.y:batch_labels,
  })
  self.writer.add_summary(summary_str, counter)
  
  # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
  _, summary_str = self.sess.run([g_optim, self.g_sum],
  feed_dict={ self.z: batch_z, self.y:batch_labels })
  self.writer.add_summary(summary_str, counter)
  
  errD_fake = self.d_loss_fake.eval({
    self.z: batch_z, 
    self.y:batch_labels
  })
  errD_real = self.d_loss_real.eval({
    self.inputs: batch_images,
    self.y:batch_labels
  })
  errG = self.g_loss.eval({
    self.z: batch_z,
    self.y: batch_labels
  })
  ```

- 如果是其他数据：

  ```python
  else:
    # Update D network
    _, summary_str = self.sess.run([d_optim, self.d_sum],
  	feed_dict={ self.inputs: batch_images, self.z: batch_z })
    self.writer.add_summary(summary_str, counter)
  
    # Update G network
    _, summary_str = self.sess.run([g_optim, self.g_sum],
  	feed_dict={ self.z: batch_z })
    self.writer.add_summary(summary_str, counter)
  
    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
    _, summary_str = self.sess.run([g_optim, self.g_sum],
  	feed_dict={ self.z: batch_z })
    self.writer.add_summary(summary_str, counter)
    
    errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
    errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
    errG = self.g_loss.eval({self.z: batch_z})
  ```

  




### 3.测试模型

```
python main.py --input_height 96 --input_width 96 \
 --output_height 48 --output_width 48 \
 --dataset anime --crop
```

```
python main.py --input_height 96 --input_width 96 --output_height 48 --output_width 48 --dataset anime --crop
```

main.py中的 OPTION 可以设置 0-4 ，在 utils.py 中的函数 visualize() 中可以看到不同的可视化选项，可以自己设置这个OPTION 

```
# Below is codes for visualization
OPTION = 0
visualize(sess, dcgan, FLAGS, OPTION)
```



### 4.测试效果

因为默认都是生成到`samples`这个文件夹，比较乱，我改了一下路径，生成到五个文件夹。

又因为模型训练的程度不够，才一千多不就已经训练了两个半小时了，只能凑合看看。

```
OPTION = 0：用模型生成一张10*10的图片
OPTION = 1：生成100张10*10的图片，都差不多样子
OPTION = 2：生成100张10*10的图片，都差不多样子
OPTION = 3：生成100张10*10的图片组成的动画
OPTION = 4：生成100张10*10的图片组成的动画，最后汇合到一个gif
```



- OPTION = 0

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910183010930.png#pic_center)


- OPTION = 1
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910182959151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




- OPTION = 2

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091018301854.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




- OPTION = 3

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910183021102.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


- OPTION = 4

  

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091018302453.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

随便放了个gif上来看 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200910183151355.gif#pic_center)

好吧，训练的太少看不出效果

## 四、原书md

 GAN与DCGAN入门

本节的程序来自于项目 https://github.com/carpedm20/DCGAN-tensorflow 。

**8.3.1 生成MNIST图像**

下载MNIST数据集：

```
python download.py mnist
```

训练：

```
python main.py --dataset mnist --input_height=28 --output_height=28 --train
```

生成图像保存在samples文件夹中。

**8.3.2 使用自己的数据集训练**

在数据目录chapter_8_data/中已经准备好了一个动漫人物头像数据集faces.zip。在源代码的data目录中再新建一个anime目录（如果没有data 目录可以自行新建），并将faces.zip 中所有的图像文件解压到anime 目录中。

训练命令：

```
python main.py --input_height 96 --input_width 96 \
  --output_height 48 --output_width 48 \
  --dataset anime --crop -–train \
  --epoch 300 --input_fname_pattern "*.jpg"
```

生成图像保存在samples文件夹中。


#### 拓展阅读

- 本章只讲了GAN 结构和训练方法，在提出GAN 的原始论文 Generative Adversarial Networks 中，还有关于GAN 收敛性的理论证明以及更多实验细节，读者可以阅读来深入理解GAN 的思想。

- 有关DCGAN的更多细节， 可以阅读其论文Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks。

- 除了本章所讲的GAN 和DCGAN 外，还有研究者对原始GAN 的损 失函数做了改进，改进后的模型可以在某些数据集上获得更稳定的 生成效果，相关的论文有：Wasserstein GAN、Least Squares Generative Adversarial Networks。

- 相比一般的神经网络，训练GAN 往往会更加困难。Github 用户 Soumith Chintala 收集了一份训练GAN 的技巧清单：https://github.com/soumith/ganhacks ，在实践中很有帮助。