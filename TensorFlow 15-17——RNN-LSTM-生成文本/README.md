[TensorFlow 15——ch12-RNN、LSTM基本结构](https://blog.csdn.net/qq_34451909/article/details/108558237)

[TensorFlow 16——ch12-RNN 和 LSTM 的实现方式](https://blog.csdn.net/qq_34451909/article/details/108562046)



@[TOC](目录)



## 一、函数定义

### 1.定义输入数据



model.py

```python
def build_inputs(self):
	with tf.name_scope('inputs'):
        # inputs 的形状和 targets 相同，都为（num_seqs，num_steps）
        # num_seqs 为一个 batch 内的句子个数
        # num_steps 为每个句子的长度
		self.inputs = tf.placeholder(tf.int32, shape=(
			self.num_seqs, self.num_steps), name='inputs')
		self.targets = tf.placeholder(tf.int32, shape=(
			self.num_seqs, self.num_steps), name='targets')
		# keep_prob 控制了 Dropout 层所需要的概率（训练0.5，测试1.0）
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		# 对于中文，需要使用embedding层，英文不用
		if self.use_embedding is False:
			self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
		else:
			with tf.device("/cpu:0"):
				embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
				self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
```



### 2.定有多层LSTM模型



model.py

```python
def build_lstm(self):
	# 创建单个cell并堆叠多层，每一层还加入了Dropout减少过拟合
	def get_a_cell(lstm_size, keep_prob):
		lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
		drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
		return drop

	with tf.name_scope('lstm'):
		cell = tf.nn.rnn_cell.MultiRNNCell(
			[get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
		)
		self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

		# 通过dynamic_rnn对cell展开时间维度
		self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

		# 通过lstm_outputs得到概率
		seq_output = tf.concat(self.lstm_outputs, 1)
		x = tf.reshape(seq_output, [-1, self.lstm_size])

		with tf.variable_scope('softmax'):
			softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
			softmax_b = tf.Variable(tf.zeros(self.num_classes))

		# proba_prediction = Softmax(Wx+b)
		self.logits = tf.matmul(x, softmax_w) + softmax_b
		self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')
```

### 3.定义损失

```python
def build_loss(self):
	with tf.name_scope('loss'):
		y_one_hot = tf.one_hot(self.targets, self.num_classes)
		y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
		self.loss = tf.reduce_mean(loss)
```



## 二、训练模型

### 1.生成英文

训练生成英文的模型：

```
python train.py \
  --input_file data/shakespeare.txt \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

```
python train.py  --input_file data/shakespeare.txt --name shakespeare --num_steps 50 --num_seqs 32 --learning_rate 0.01 --max_steps 20000
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914075542243.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914075548833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)








测试模型：

```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \
  --max_length 1000
```

```
python sample.py --converter_path model/shakespeare/converter.pkl --checkpoint_path model/shakespeare/ --max_length 1000
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914075556131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914075559682.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


因为每次候选下一个字母都是top5做概率归一后挑出的，所以文本生成的结果都会不同。

top5的代码看这里：[python概率选取ndarray的TOP-N](https://blog.csdn.net/qq_34451909/article/details/108570036)


真的好神奇，很难想象才20000步的效果会那么好！



### 2.生成诗词

训练写诗模型：

```
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```

```
python train.py --use_embedding --input_file data/poetry.txt --name poetry --learning_rate 0.005 --num_steps 26 --num_seqs 32 --max_steps 10000
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914080208618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914083058489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




测试模型：

```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

```
python sample.py --use_embedding --converter_path model/poetry/converter.pkl --checkpoint_path model/poetry/ --max_length 300
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200914083811548.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091408381835.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


### 3.生成C代码

训练生成C代码的模型：

```
python train.py \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

```
python train.py --input_file data/linux.txt --num_steps 100 --name linux --learning_rate 0.01 --num_seqs 32 --max_steps 20000
```



测试模型：

```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path model/linux \
  --max_length 1000
```

```
python sample.py --converter_path model/linux/converter.pkl --checkpoint_path model/linux --max_length 1000
```



## 三、原书md

 RNN基本结构与Char RNN文本生成

**12.5.4 训练模型与生成文字**

训练生成英文的模型：

```
python train.py \
  --input_file data/shakespeare.txt \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

测试模型：

```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \
  --max_length 1000
```

训练写诗模型：

```
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```


测试模型：

```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

训练生成C代码的模型：

```
python train.py \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

测试模型：

```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path model/linux \
  --max_length 1000
```

#### 拓展阅读

- 如果读者想要深入了解RNN 的结构及其训练方法，建议阅读书籍 Deep Learning（Ian Goodfellow、Yoshua Bengio 和 Aaron Courville 所著）的第10章“Sequence Modeling: Recurrent and Recursive Nets”。 此外，http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 中详细地介绍了RNN 以及Char RNN 的原理，也是很好的阅读材料。

- 如果读者想要深入了解LSTM 的结构， 推荐阅读 http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 。有网友对这篇博文做了翻译，地址为：http://blog.csdn.net/jerr__y/article/ details/58598296。

- 关于TensorFlow 中的RNN 实现，有兴趣的读者可以阅读TensorFlow 源码进行详细了解，地址为：https://github.com/tensorflow/tensorflow/ blob/master/ tensorflow/python/ops/rnn_cell_impl.py 。该源码文件中有BasicRNNCell、BasicLSTMCell、RNNCell、LSTMCell 的实现。









