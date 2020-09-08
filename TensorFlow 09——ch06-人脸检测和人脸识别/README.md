CSDN:



百度云链接：[https://pan.baidu.com/s/1i4a85oHe5huA5U7pR9N_-w](https://pan.baidu.com/s/1i4a85oHe5huA5U7pR9N_-w) 
提取码：aldv



## 一、第一部分

### 1.人脸检测和人脸对齐：MTCNN

- P-Net（输入12×12×3）
  - （输出1×1×2）人脸判断
  - （输出1×1×4）框回归
  - （输出1×1×10）人脸关键点位置：左眼、右眼、鼻子、左嘴角、右嘴角
- R-Net（输入24×24×3）
  - （输出1×1×2）人脸判断
  - （输出1×1×4）框回归
  - （输出1×1×10）人脸关键点位置：左眼、右眼、鼻子、左嘴角、右嘴角
- O-Net（输入48×48×3）
  - （输出1×1×2）人脸判断
  - （输出1×1×4）框回归
  - （输出1×1×10）人脸关键点位置：左眼、右眼、鼻子、左嘴角、右嘴角

### 2.深度卷积网络提取特征

- 三元组损失（ Triplet Loss ）

- 中心损失（ Center Loss ）

- 向量表示：

  - 对于同一个人的两张人脸图像，对应的向量之间的欧几里得距离应该比较小。
  - 对于不同人的两张人脸图像，对应的向量之间的欧几里得距离应该比较大。

### 3.应用

人脸验证、人脸识别、人脸聚类



## 二、第二部分

### 1.准备工作

- 把src目录加入PYTHONPATH

Linux：

```
export PYTHONPATH=[...]/src 
```

Windows：

参考[TensorFlow 08——ch05-TensorFlow Object Detection 深度学习目标检测](https://blog.csdn.net/qq_34451909/article/details/108382667)

把 `[...]\src` 加进去：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200908122841620.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




- 需要的库

可以测试一下：

```
＃以下是该项目中需费包库文件 
import tensorflow as tf 
import sklearn 
import scipy 
import cv2 
import hSpy 
import matplotlib 
import PIL 
import requests 
import psutil
```



==cv2 没有的话，安装不是cv2是opencv-python==



### 2.下载LFW 人脸数据库

在地址[http://vis-www.cs.umass.edu/lfw/lfw.tgz](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 下载lfw数据集，并解压到~/datasets/中：

百度云链接：[https://pan.baidu.com/s/1i4a85oHe5huA5U7pR9N_-w](https://pan.baidu.com/s/1i4a85oHe5huA5U7pR9N_-w) 
提取码：aldv



**Linux**：命令

```
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C ./lfw/raw --strip-components=1
```

**Windows**：手动

`./datasets/lfw/` 建两个：

`raw`：放解压的一大堆文件夹

`lfw_mtcnnpy_160`：空着放结果



### 3.修改代码

`src\align\detect_face.py`中的`load()`加上`allow_pickle=True`

```python
data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()  # 加上allow_pickle=True
```



### 4.对LFW进行人脸检测和对齐：

**Linux**：命令

```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/lfw/raw \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  --image_size 160 --margin 32 \
  --random_order
```

在输出目录 `~/datasets/lfw/lfw_mtcnnpy_160` 中可以找到检测、对齐后裁剪好的人脸。

**Windows**：

```
python src/align/align_dataset_mtcnn.py  ./datasets/lfw/raw ./datasets/lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order
```

### 5.验证已有模型的正确率

**Linux**：

```
python src/validate_on_lfw.py \
~/datasets/lfw/lfw_mtcnnpy_160 \ 
~/models/facenet/20170512-110547
```

**Windows**：

```
python src/validate_on_lfw.py datasets/lfw/lfw_mtcnnpy_160 src/models/facenet/20170512-110547/
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200908122825746.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

### 6.在自己的数据上使用已有模型
假设现在有三张图片`./test_imgs/1.jpg`、 `./test_imgs/2.jpg`、 `./test_imgs/3.jpg`  , 这三张图片中各含再一个人的人脸，希望计算它们两两之间的距离。 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200908124740523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



使用 compare.py 就可以实现，运行下面的代码．

**Linux**：

```
python src/compare.py \ 
~/models/facenet/20170512-110547/ \ 
./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg 
```


**Windows**：
```
python src/compare.py src/models/facenet/20170512-110547/ ./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg 
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200908124602455.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)





## 三、原书readme

 人脸检测和人脸识别

本节的程序来自于项目https://github.com/davidsandberg/facenet 。

**6.4.1 项目环境设置**

参考6.4.1小节。

**6.4.2 LFW 人脸数据库**

在地址http://vis-www.cs.umass.edu/lfw/lfw.tgz 下载lfw数据集，并解压到~/datasets/中：
```
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C ./lfw/raw --strip-components=1
```

**6.4.3 LFW 数据库上的人脸检测和对齐**

对LFW进行人脸检测和对齐：

```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/lfw/raw \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  --image_size 160 --margin 32 \
  --random_order
```

在输出目录~/datasets/lfw/lfw_mtcnnpy_160中可以找到检测、对齐后裁剪好的人脸。

**6.4.4 使用已有模型验证LFW 数据库准确率**

在百度网盘的chapter_6_data/目录或者地址https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk 下载解压得到4个模型文件夹，将它们拷贝到~/models/facenet/20170512-110547/中。

之后运行代码：
```
python src/validate_on_lfw.py \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  ~/models/facenet/20170512-110547/
```

即可验证该模型在已经裁剪好的lfw数据集上的准确率。

**6.4.5 在自己的数据上使用已有模型**

计算人脸两两之间的距离：
```
python src/compare.py \
  ~/models/facenet/20170512-110547/ \
  ./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg
```

**6.4.6 重新训练新模型**

以CASIA-WebFace数据集为例，读者需自行申请该数据集，申请地址为http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html 。获得CASIA-WebFace 数据集后，将它解压到~/datasets/casia/raw 目录中。此时文件夹~/datasets/casia/raw/中的数据结构应该类似于：
```
0000045
  001.jpg
  002.jpg
  003.jpg
  ……
0000099
  001.jpg
  002.jpg
  003.jpg
  ……
……
```

先用MTCNN进行检测和对齐：
```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/casia/raw/ \
  ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 182 --margin 44
```

再进行训练：
```
python src/train_softmax.py \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop --random_flip \
  --learning_rate_schedule_file
  data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-2 \
  --center_loss_alfa 0.9
```

打开TensorBoard的命令(<开始训练时间>需要进行替换)：
```
tensorboard --logdir ~/logs/facenet/<开始训练时间>/
```

#### 拓展阅读

- MTCNN是常用的人脸检测和人脸对齐模型，读者可以参考论文Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks 了解其细节。

- 训练人脸识别模型通常需要包含大量人脸图片的训练数据集，常用 的人脸数据集有CAISA-WebFace（http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html ）、VGG-Face（http://www.robots.ox.ac.uk/~vgg/data/vgg_face/ ）、MS-Celeb-1M（https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-millioncelebrities-real-world/ ）、MegaFace（ http://megaface.cs.washington.edu/ ）。更多数据集可以参考网站：http://www.face-rec.org/databases

- 关于Triplet Loss 的详细介绍，可以参考论文FaceNet: A Unified Embedding for Face Recognition and Clustering，关于Center Loss 的 详细介绍，可以参考论文A Discriminative Feature Learning Approach for Deep Face Recognition。
