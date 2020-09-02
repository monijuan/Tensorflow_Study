
csdn：https://blog.csdn.net/qq_34451909/article/details/108337233



## 一、准备
### 1.准备数据集
在 data_prepare 目录下用 data_convert.py 将图片转换为为 tfrecord 恪式，

```python
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```

```python
python data_convert.py -t pic/   --train-shards 2   --validation-shards 2   --num-threads 2   --dataset-name satellite
```
就可以在 pie 文件夹中找到 5 个新生成的文件，分别 
标签

> label.txt

训练数据 
> satellite_train_00000-of-00002. tfrecord 
> satellite_train_00001-of-00002. tfrecord

验证数据
> satellite_validation_00000-of-00002. tfrecord 
> satellite_validation_00001-of-00002.tfrecord

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901155455809.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 2.微调训练模型程序flowers.py为satellite.py
在目录 slim/datasets 中，复制 flowers.py，修改文件名为 satellite.py 修改好几个地方

**satellite.py** 

```python
_FILE_PATTERN = 'satellite_%s_*.tfrecord' ### 需要修改

SPLITS_TO_SIZES = {'train': 4800, 'validation': 1200} ### 需要修改

_NUM_CLASSES = 6 ### 需要修改

'image/format': tf.FixedLenFeature((), tf.string,default_value='jpg'), ### 需要修改为jpg
```


**dataset_factory. py** 


```python
from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import satellite ### 需要添加

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist, 
    'satellite': satellite, ### 需要添加
}
```


**train_image_classifier.py**

```python
###########################
# Kicks off the training. #
###########################
config = tf.ConfigProto(allow_soft_placement = True) ### 需要添加
slim.learning.train(
	train_tensor,
	logdir=FLAGS.train_dir,
	master=FLAGS.master,
	is_chief=(FLAGS.task == 0),
	init_fn=_get_init_fn(),
	summary_op=summary_op,
	number_of_steps=FLAGS.max_number_of_steps,
	log_every_n_steps=FLAGS.log_every_n_steps,
	save_summaries_secs=FLAGS.save_summaries_secs,
	save_interval_secs=FLAGS.save_interval_secs,
	sync_optimizer=optimizer if FLAGS.sync_replicas else None,
	session_config = config) ### 需要添加
```

==还有几个地方书上没有，可能是python3需要改的！==


**data_prepare/tfrecord.py**


```python
所有xrange改成range

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value=tf.compat.as_bytes(value) ### 需要添加整行
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
# Read the image file.
with open(filename, 'rb') as f: ### 需要加个b
	image_data = f.read()
	
# 函数_find_image_files():中
shuffled_index = list(range(len(filenames))) ### 需要加上list
```






### 3.准备Inception V3 模型

下载 inception_v3.ckpt
链接：https://pan.baidu.com/s/1IC5x_md8NNZr5pL3ccgjOQ 
提取码：fjct


### 4.准备训练文件夹
slim/satellite 中
- 新建文件夹`data` 放 1. 中的五个文件
- 新建文件夹`train_dir`，放结果
- 新建`pretrained`，放 Inception V3的`inception_v3.ckpt`


## 二、运行模型
在 slim 文件夹下打开终端

```python
python train_image_classifier.py \
  --train_dir=satellite/train_dir \ # 输出路径 
  --dataset_name=satellite \        # 指定训练数据集
  --dataset_split_name=train \      # 指定训练数据集
  --dataset_dir=satellite/data \    # 数据集目录路径
  --model_name=inception_v3 \       # 模型名称
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \ # 预训练模型保存位置
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \ # 恢复预训练时不恢复这两层
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \ # 微调范围
  --max_number_of_steps=100000 \# 最大步数
  --batch_size=32 \             # batch数量
  --learning_rate=0.001 \       # 学习率
  --learning_rate_decay_type=fixed \ # 固定学习率
  --save_interval_secs=300 \    # 每隔300s把模型存到 train dir 
  --save_summaries_secs=2 \     # 每隔2s把日志存到 train dir 
  --log_every_n_steps=10 \      # 每隔10步打印信息
  --optimizer=rmsprop \         # 选定的优化器
  --weight_decay=0.00004        # 所有参数的二次正则化超参数
```

```python
python train_image_classifier.py --train_dir=satellite/train_dir --dataset_name=satellite --dataset_split_name=train --dataset_dir=satellite/data --model_name=inception_v3 --checkpoint_path=satellite/pretrained/inception_v3.ckpt --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --max_number_of_steps=100000 --batch_size=32 --learning_rate=0.001 --learning_rate_decay_type=fixed --save_interval_secs=300 --save_summaries_secs=2 --log_every_n_steps=10 --optimizer=rmsprop --weight_decay=0.00004
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901155432804.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



## 三、评估模型

```python
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v3
```

```python
python eval_image_classifier.py   --checkpoint_path=satellite/train_dir   --eval_dir=satellite/eval_dir   --dataset_name=satellite   --dataset_split_name=validation   --dataset_dir=satellite/data   --model_name=inception_v3
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020090115505387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901155511609.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901155515269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020090115552831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)





## 四、对单张图片进行识别


### 1. 导出网络结构：

在 `slim 文件夹`下运行`export_inference_graph.py`，导出`inception_v3 _inf _graph.pb`

```python
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v3 \ 	# 网络结构名字
--output_file=satellite/inception_v3_inf_graph.pb \ # 网络结构文件名字
--dataset_name satellite
```

```python
python export_inference_graph.py   --alsologtostderr   --model_name=inception_v3   --output_file=satellite/inception_v3_inf_graph.pb   --dataset_name satellite
```

### 2. 导出模型参数：

在根目录运行 freeze_graph. py，导出frozen_graph.pb 

```python
python freeze_graph.py \
--input_graph slim/satellite/inception_v3_inf_graph.pb \ # 网络结构文件名字
--input_checkpoint slim/satellite/train_dir/model.ckpt-254 \ # 这个看各自train_dir的
--input_binary true \ # true 代表二进制形式
--output_node_names InceptionV3/Predictions/Reshape_1 \ # 输出模型的最后一层
--output_graph slim/satellite/frozen_graph.pb # 最后导出的模型
```

```python
python freeze_graph.py   --input_graph slim/satellite/inception_v3_inf_graph.pb   --input_checkpoint slim/satellite/train_dir/model.ckpt-254   --input_binary true   --output_node_names InceptionV3/Predictions/Reshape_1   --output_graph slim/satellite/frozen_graph.pb
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901155728543.png#pic_center)


### 3. 预测：

在根目录运行classify_image_inception_v3.py

```python
python classify_image_inception_v3.py \
--model_path slim/satellite/frozen_graph.pb \ 	# 导出的模型
--label_path data_prepare/pic/label.txt \		# 各类别名称
--image_file test_image.jpg # 需要测试的图片
```

可能由于训练不够，才几十步

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901155015967.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


```python
python classify_image_inception_v3.py   --model_path slim/satellite/frozen_graph.pb   --label_path data_prepare/pic/label.txt   --image_file test_image.jpg
```



## 五、所有代码

[https://github.com/MONI-JUAN/Tensorflow_Study/tree/master/TensorFlow%2005%E2%80%94%E2%80%94ch04-Slim%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83satellite](https://github.com/MONI-JUAN/Tensorflow_Study/tree/master/TensorFlow%2005%E2%80%94%E2%80%94ch04-Slim%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83satellite)







## 六、原文readme

#### 运行方法

**3.2 数据准备**

首先需要将数据转换成tfrecord的形式。在data_prepare文件夹下，运行：
```
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```
这样在pic文件夹下就会生成4个tfrecord文件和1个label.txt文件。

**3.3.2 定义新的datasets 文件**

参考3.3.2小节对Slim源码做修改。

**3.3.3 准备训练文件夹**

在slim文件夹下新建一个satellite目录。在这个目录下做下面几件事情：
- 新建一个data 目录，并将第3.2中准备好的5个转换好格式的训练数据复制进去。
- 新建一个空的train_dir目录，用来保存训练过程中的日志和模型。
- 新建一个pretrained目录，在slim的GitHub页面找到Inception V3 模型的下载地址http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 下载并解压后，会得到一个inception_v3.ckpt 文件，将该文件复制到pretrained 目录下（这个文件在chapter_3_data/文件中也提供了）

**3.3.4 开始训练**

（在slim文件夹下运行）训练Logits层：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=2 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

训练所有层：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

**3.3.6 验证模型准确率**

在slim文件夹下运行：
```
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v3
```

**3.3.7 TensorBoard 可视化与超参数选择**

打开TensorBoard：
```
tensorboard --logdir satellite/train_dir
```

**3.3.8 导出模型并对单张图片进行识别**

在slim文件夹下运行：
```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=satellite/inception_v3_inf_graph.pb \
  --dataset_name satellite
```

在chapter_3文件夹下运行（需将5271改成train_dir中保存的实际的模型训练步数）：
```
python freeze_graph.py \
  --input_graph slim/satellite/inception_v3_inf_graph.pb \
  --input_checkpoint slim/satellite/train_dir/model.ckpt-5271 \
  --input_binary true \
  --output_node_names InceptionV3/Predictions/Reshape_1 \
  --output_graph slim/satellite/frozen_graph.pb
```

运行导出模型分类单张图片：
```
python classify_image_inception_v3.py \
  --model_path slim/satellite/frozen_graph.pb \
  --label_path data_prepare/pic/label.txt \
  --image_file test_image.jpg
```


#### 拓展阅读

- TensorFlow Slim 是TensorFlow 中用于定义、训练和验证复杂网络的 高层API。官方已经使用TF-Slim 定义了一些常用的图像识别模型， 如AlexNet、VGGNet、Inception模型、ResNet等。本章介绍的Inception V3 模型也是其中之一， 详细文档请参考： https://github.com/tensorflow/models/tree/master/research/slim。
- 在第3.2节中，将图片数据转换成了TFRecord文件。TFRecord 是 TensorFlow 提供的用于高速读取数据的文件格式。读者可以参考博文（ http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ ）详细了解如何将数据转换为TFRecord 文件，以及 如何从TFRecord 文件中读取数据。
- Inception V3 是Inception 模型（即GoogLeNet）的改进版，可以参考论文Rethinking the Inception Architecture for Computer Vision 了解 其结构细节。
