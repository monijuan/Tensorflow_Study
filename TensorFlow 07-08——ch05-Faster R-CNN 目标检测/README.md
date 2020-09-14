

CSDN：

TensorFlow 07——ch05-理解R-CNN、Fast R-CNN、Faster R-CNN：

https://blog.csdn.net/qq_34451909/article/details/108381094

TensorFlow 08——ch05-TensorFlow Object Detection 深度学习目标检测：

https://blog.csdn.net/qq_34451909/article/details/108382667



数据集和模型较大，可以提前下！
链接：[https://pan.baidu.com/s/11E-8-AmUniHqu5WvQF82gQ](https://pan.baidu.com/s/11E-8-AmUniHqu5WvQF82gQ) 
提取码：w9bi


## 准备工作

### 1.编译 research\object_detection\protos 中的 .proto 

在`research`文件夹执行命令：

```python
protoc object_detection/protos/*.proto --python_out=.
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020090314511841.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

### 2.把Slim加入PYTHONPATH 

TensorFlow Object Detection API 是以 Slim 为基础实现的，需要将 Slim 的目录加入 PYTHONPATH 后才能正确运行。

如果是Linux直接在在`research`文件夹执行命令：

```
export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim
```

如果是windows稍微麻烦一些，在python的执行目录新建一个pth文件，每个人的电脑不一样，我的是：

在目录`C:\Anaconda3\Lib\site-packages\`，新建了一个`tensorflow_model.pth`，把research的路径放进来，我的是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200903144810793.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

```
D:\code\python\CV\Deep-Learning-21-Examples-master\chapter_5\research
D:\code\python\CV\Deep-Learning-21-Examples-master\chapter_5\research\slim
D:\code\python\CV\Deep-Learning-21-Examples-master\chapter_5\research\object_detection
```

### 3.测试安装API成功

在`research`文件夹执行命令：

```
python object_detection/builders/model_builder_test.py
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200903145000244.png#pic_center)


### 4.执行已经训练好的模型

TensorFlow Object Detection API 默认提供了 5 个预训练模型，官们都是使用 coco 数据集训练完成的，结构分别为：

- SSD+MobileNet
- SSD+Inception
- R-FCN+ResNet101 
- Faster RCNN+ResNet101 
- Faster RCNN+Inception_ResNet


在目录`research\object_detection\`，打开`jupyter`运行一遍`object_detection_tutorial.ipynb`

切换`jupyter`的目录是在`research\object_detection\` 打开终端运行`jupyter-notebook`，就会在这启动jupyter了

运行一遍`object_detection_tutorial.ipynb`之后能得到检测效果如图 。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200903144737823.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
两张图片分别是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200903144746697.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200903144750897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)






那就可以开始正题了！

## 训练新模型

### 1.下载数据集


地址是：[http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar ](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

但是这个下的比较慢，我传到百度云了。

链接：[https://pan.baidu.com/s/11E-8-AmUniHqu5WvQF82gQ](https://pan.baidu.com/s/11E-8-AmUniHqu5WvQF82gQ) 
提取码：w9bi
（另一个等下也要下）

在项目的object_detection文件夹中新建voc目录，并将解压后的数据集拷贝进来，最终形成的目录为：

```
research/
  object_detection/
    voc/
      VOCdevkit/
        VOC2012/
          JPEGImages/
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            ………………
          Annotations/
            2007_000027.xml
            2007_000032.xml
            2007_000033.xml
            2007_000039.xml
            2007_000042.xml
            ………………
          ………………
```

### 2.解压并转换

将voc 2012数据集转换为 tfrecord 恪式，转换好的 tfrecord 保存在新建的 voc 文件夹下。
在项目的object_detection文件夹中cmd运行：

```python
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
```

然后将 pascal_label_map.pbtxt 数据复制到 voc 文件夹下。


### 3.准备模型

下载  Faster RCNN+Inception_ResNet_v2 的这个模型`faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz`，


地址是：[http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz ](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz)

链接：[https://pan.baidu.com/s/11E-8-AmUniHqu5WvQF82gQ](https://pan.baidu.com/s/11E-8-AmUniHqu5WvQF82gQ) 
提取码：w9bi
（跟刚才的链接一样）

解压，在`voc`文件夹建一个文件夹`pretrained`，把这五个解压的文件放进去。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200904144001241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

==**注意：** 是这几个文件，一开始我下错了，查了好几个小时才发现数据集都匹配不上是这里导致的！==

### 4.模型配置文件

TensorFlow Object Detection API 是依赖一个特殊的设置文件进行训练的 。在object_detection/samples/configs/ 文件夹下，有一些设置文件的示例。
可以参考 faster_rcnn_inception_resnet_v2_atrous_pets.config 文件创建的设置
文件。
先将 faster_rcnn_inception_resnet_v2_atrous_pets.config 复制一份到 voc 文件夹下，修改7个地方：

```bash
 1:num_classes: 20
 2:num_examples: 5823
 3:PATH_TO_BE_CONFIGURED，需要改5个地方
     fine_tune_checkpoint: "voc/pretrained/model.ckpt"
     input_path: "voc/pascal_train.record"
     label_map_path: "voc/pascal_label_map.pbtxt"
     input_path: "voc/pascal_val.record"
     label_map_path: "voc/pascal_label_map.pbtxt"
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200904143854812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


### 5.训练新模型

在voc中创建一个文件夹`train_dir`，在object_detection中执行命令：

```python
python train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
```

查看模型训练情况

```python
tensorboard --logdir voc/train_dir/
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200904174044755.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200904174103476.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

## 原书：深度学习中的目标检测

1 安装TensorFlow Object Detection API

参考5.2.1小节完成相应操作。



3 训练新的模型

先在地址http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 下载VOC2012数据集并解压。

在项目的object_detection文件夹中新建voc目录，并将解压后的数据集拷贝进来，最终形成的目录为：

```
research/
  object_detection/
    voc/
      VOCdevkit/
        VOC2012/
          JPEGImages/
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            ………………
          Annotations/
            2007_000027.xml
            2007_000032.xml
            2007_000033.xml
            2007_000039.xml
            2007_000042.xml
            ………………
          ………………
```

在object_detection目录中执行如下命令将数据集转换为tfrecord：

```
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
```

此外，将pascal_label_map.pbtxt 数据复制到voc 文件夹下：
```
cp data/pascal_label_map.pbtxt voc/
```

下载模型文件http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz 并解压，解压后得到frozen_inference_graph.pb 、graph.pbtxt 、model.ckpt.data-00000-of-00001 、model.ckpt.index、model.ckpt.meta 5 个文件。在voc文件夹中新建一个
pretrained 文件夹，并将这5个文件复制进去。

复制一份config文件：
```
cp samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config \
  voc/voc.config
```

并在voc/voc.config中修改7处需要重新配置的地方（详见书本）。

训练模型的命令：
```
python train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
```

使用TensorBoard：
```
tensorboard --logdir voc/train_dir/
```

**5.2.4 导出模型并预测单张图片**

运行(需要根据voc/train_dir/里实际保存的checkpoint，将1582改为合适的数值)：
```
python export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path voc/voc.config \
  --trained_checkpoint_prefix voc/train_dir/model.ckpt-1582
  --output_directory voc/export/
```

导出的模型是voc/export/frozen_inference_graph.pb 文件。

**拓展阅读**

- 本章提到的R-CNN、SPPNet、Fast R-CNN、Faster R-CNN 都是基于 区域的深度目标检测方法。可以按顺序阅读以下论文了解更多细节： Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation (R-CNN) 、Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition（SPPNet）、Fast R-CNN （Fast R-CNN）、Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks（Faster R-CNN）。

- 限于篇幅，除了本章提到的这些方法外，还有一些有较高参考价值 的深度学习目标检测方法，这里同样推荐一下相关的论文：R-FCN: Object Detection via Region-based Fully Convolutional Networks （R-FCN）、You Only Look Once: Unified, Real-Time Object Detection （YOLO）、SSD: Single Shot MultiBox Detector（SSD）、YOLO9000: Better, Faster, Stronger（YOLO v2 和YOLO9000）等。
