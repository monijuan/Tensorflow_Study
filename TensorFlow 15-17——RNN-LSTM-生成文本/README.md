CSDN：https://blog.csdn.net/qq_34451909/article/details/108571585



[TensorFlow 15——ch12-RNN、LSTM基本结构](https://blog.csdn.net/qq_34451909/article/details/108558237)

[TensorFlow 16——ch12-RNN 和 LSTM 的实现方式](https://blog.csdn.net/qq_34451909/article/details/108562046)



@[TOC](目录)



# 第一部分 TF15



## 一、RNN

### 1.单层神经网络

单层网络的输入是 x，经过变焕 Wx+b 和激活函数f得到输出 y。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913074635557.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)




对于序列形的数据，例如：

- 自然语言处理中的 abcd 连着的单词
- 语音处理中的 abcd 连续声音信号
- 时间序列问题，股票价格之类的



### 2.引入隐状态 h



序列形的数据不太好用原始的神经网络处理。为了处理建模序列问题，RNN引入了隐状态 h(hidden state）的概念，h 可以对序列形的数据提取特征，接着再转换为输出。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913074802317.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
其中：

- 圆圈方块表示向量
- 箭头表示对向量变换
- U、W是参数矩阵
- b是偏置项参数
- f是激活函数，在RNN中通常使用tanh


计算了h1之后，h2 也是一样的，这里U、W、b都是一样的，参数共享：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913082725887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

以此类推计算所有h：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913082746644.png#pic_center)



### 3.输出y



因为 h 是对序列形的数据提取特征，所以到现在都没有输出，输出是通过 h 进行计算，例如 y1：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913082917680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

这里的 V 和 c 都是新参数。


其他的输出 y 也类似：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913083016348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

这就是 RNN 的结构，输入是 x ，输出是 y ，是等长的。



### 4.限制

因为输入输出序列必须是等长的，所以只有一些问题使用经典的 RNN 结构模型：

- 视频中每一帧的分类
- 输入字符，输出下一个字符的概率，就是 Char RNN



### 5.RNN的数学定义


$$
h_{t} = f(Ux_t+Wh_{t-1}+b)\\
y_t = Softmax(Vh_t+c)
$$

其中，$U,V,W,b,c$ ，均为参数，而 $f$ 表示激活函数，一般为 tanh 函数。





## 二、N VS 1 RNN

### 1.结构图

如果输出是单值而不是序列，就只在最后一个 h 进行输出变换，结构如图：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913084419377.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 2.数学表达

输入为：$x_1,x_2,\cdots,x_t,\cdots,x_T$ 
隐状态为：$h_1,h_2,\cdots,h_t,\cdots,h_T$ 
输出为：$Y$
运算过程为：
$$
h_t=f(Ux_t+Wh_{t-1}+b)
$$
结果$Y$的计算：

$$
Y=Softmax(Vh_r+c)
$$

## 三、1 VS N RNN

### 1.结构图

1 VS N的就是一个输入，一列输出，有两种结构

一种是：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913093609253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



另一种是：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913093645702.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

这两种图等价

### 2.数学表达

数学公式为：
$$
h_t=f(UX+Wh_{t-1}+b)\\
y_t=Softmax(Vh_t+c)
$$

### 3.应用

1 VS N 可以处理的问题有：

- 从图像生成文字（image caption），输入的是图像特征，输出的是一段句子
- 从类别生成语音或者音乐

## 四、LSTM

RNN 的改进版 LSTM ( Long Short-Term Memory，长短期记忆网络）。

LSTM 的输入和输出与 RNN 一样，所以可以无缝替换。



RNN的公式 $h_t=f(Ux_t+Wh_{t-1}+b)$ ，容易梯度爆炸和梯度消失，所以 RNN 很难处理“长程依赖”问题。



### 1.RNN 和 LSTM 的图示

RNN 的图示：



![RNN](https://img-blog.csdnimg.cn/2020091310115644.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



对应公式：
$$
h_{t} = f(Ux_t+Wh_{t-1}+b)
$$

再看 LSTM 的图示：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913101313207.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

各个符号的含义：

- 矩形：激活函数
- 圆形：逐点运算（两个矩形进行相加、相乘、等）
- 箭头：指向运算点

### 2.LSTM 的隐状态

LSTM 的隐状态有两部分：

- $h_t$ ：同 RNN
- $C_t$ ：在主干道上传递，避免梯度爆炸和梯度消失

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913103928819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

 $C_t$ 不是完全照搬 $C_{t-1}$ ，而是在$C_{t-1}$ 的基础上选择性的“遗忘”，“记住”一些内容，也就需要用到“遗忘门”和“记忆门”。



### 3.遗忘门

控制遗忘掉 $C_{t-1}$ 的那些部分。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913105832461.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

数学公式：

$$
f_t=\sigma (W_f \cdot [h_{t-1},x_t] + b_f)
$$

其中 $\sigma$ 是 Sigmoid 激活函数，输出在0 ~ 1之间。

遗忘门的输入是 $x_t$ 和 $h_{t-1}$

遗忘门的输出和  $C_{t-1}$ 相同形状的矩阵，和 $C_{t-1}$ 逐点相乘，决定遗忘哪些部分。因此全0则全遗忘，全1则全保留。

(打公式好麻烦。。。点个赞吧）

### 4.记忆门

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913110310218.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

数学公式：

$$
i_t=\sigma (W_i \cdot [h_{t-1},x_t] + b_i)\\
\widetilde{C_t}=tanh(W_C \cdot [h_{t-1},x_t] +b_C)
$$

记忆门的输入也是 $x_t$ 和 $h_{t-1}$

记忆门的输出也是 $\widetilde{C_t}$ 和 $i_t$ 逐点相乘，决定记住哪些部分。


### 5.结合“遗忘”“记忆”

$f_t$  是遗忘门输出的（0 ~ 1），$\widetilde{C_t} * i_t$ 是需要记住的新东西。

输出：
$$
C_t=f_t * C_{t-1} + i_t *\widetilde{C_t}
$$


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913113154708.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

### 6.输出门

这里有两个 $h_t$ ，一个是输出门的输出，传递给下一步用的，还有一个输出才是真正这一步的输出，是这几个门输出的函数。



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913114302550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

输出门的输出就是水平方向的那个，是根据$x_t$ 和 $h_{t-1}$ 计算，结果$h_t$ 通过 $o_t * tanh(C_t)$ 得到。

数学公式：

$$
o_t=\sigma (W_o[h_{t-1,x_t}]+b_o) \\
h_t=o_t * tanh(C_t)
$$


## 五、Char RNN

Char RNN 是 N VS N RNN 的经典结构，也就是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913124123220.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

输入是句子中的字母，输出是下一个字母，也就是预测下文。



# 第二部分 TF16



### RNNCell

RNNCell 是 Tensorflow 中的 RNN 基本单元，是一个抽象类，没有办法实体化，要用的是两个子类，一个是 `BasicRNNCell` ，一个是 `BasicLSTMCell` 。

RNNCell 有一个 `call` 函数，是 RNN 的单步计算，调用：

```python
(output, next_state) = call(input, state)
```

初始输入为 x1，初始的隐藏层为 h0，例如：

```python
(output1, h1) = cell.call(x1, h0) # 得到h1
(output2, h2) = cell.call(x2, h1) # 得到h2
```

RNNCell 的类属性 ：

- state_size 规定了隐藏层的大小
- output_size 规定了输出向量的大小

###  RNN 基本单元



```python
import tensorflow as tf
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
print(rnn_cell.state_size)
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913142116127.png#pic_center)





###  LSMT 基本单元



```python
import tensorflow as tf
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
print(lstm_cell.state_size)
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913142222644.png#pic_center)

可以看到 BasicLSTMCell 的 state_size 由 `c` 和 `h` 两部分组成。

所以一般使用 BasicLSTMCell 的时候，分开这两部分：



```python
import tensorflow as tf
import numpy as np
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = lstm_cell.zero_state(32, np.float32)
output, h1 = lstm_cell.call(inputs, h0)
print(h1.h)
print(h1.c)
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913142519699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### MultiRNNCell

单层 RNN 的能力有限，所以需要多层的 RNN，也就是第一层的输出 h 作为第二层的输入。

可以使用tf.nn.rnn_cell.MultiRNNCell 函数对 RNN 进行堆叠。测试代码如下：

```python
# 返回一个BasicLSTMCell
def get_a_cell():
	return tf.nn.rnn_cell.BasicLSTMCell(128)

# 创建3层的RNN，state_size=(128,128,128)，表示3个隐层大小都为128
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])

inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = cell.zero_state(32, np.float32)
output, h1 = cell.call(inputs, h0)
print(h1)
```



打印结果（换行是我加的）：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913172557178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



```
LSTMStateTuple(
c=<tf.Tensor 'cell_0/cell_0/basic_lstm_cell/add_1:0' shape=(32, 128) dtype=float32>, 
h=<tf.Tensor 'cell_0/cell_0/basic_lstm_cell/mul_2:0' shape=(32, 128) dtype=float32>
), 
LSTMStateTuple(
c=<tf.Tensor 'cell_1/cell_1/basic_lstm_cell/add_1:0' shape=(32, 128) dtype=float32>, 
h=<tf.Tensor 'cell_1/cell_1/basic_lstm_cell/mul_2:0' shape=(32, 128) dtype=float32>
), 
LSTMStateTuple(
c=<tf.Tensor 'cell_2/cell_2/basic_lstm_cell/add_1:0' shape=(32, 128) dtype=float32>, 
h=<tf.Tensor 'cell_2/cell_2/basic_lstm_cell/mul_2:0' shape=(32, 128) dtype=float32>
)
```



### BasicRNNCell 的 call



BasicRNNCell 的 call 的 return :

```python
def call(self, inputs, state):
    if self._linear is None:
        self._linear = _Linear([inputs, state], self._num_units, True)
        output = self._activation(self._linear([inputs, state]))
        return output, output
```

可以看出在 BasicRNNCell 中 output（输出） 和隐状态是一样的，因此需要额外对输出定义新的变换，才能得到图中真正的输出 y 。

而隐状态就是函数中的 output（函数），所以 BasicRNNCell 中 state_size 永远等于 output_size 。



### BasicLSTMCell 的 call



BasicLSTMCell 的 call 的 return :

```python
if self._state_is_tuple:
	new_state = LSTMStateTuple(new_c, new_h)
else:
	new_state = array_ops.concat([new_c, new_h], 1)
return new_h, new_state
```

其中 _state_is_tuple 是一直等于 `Ture` 的，所以返回的隐状态是 ` LSTMStateTuple(new_c, new_h)` ，而output是 `new_h` 。

因此如果处理的是分类问题，还需要对 output 添加单独的 Softmax 层才能得到最后的分类概率输出。



### 展开时间维度

对单个的 RNNCell ，如果序列长，则需要调用n次call，所以 Tensorflow 提供了一个函数 `tf.nn.dynamic_rnn`，这个函数相当于就调用了n次call。



```python
def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None):
```



输入的 inputs 数据格式：

> inputs: shape = (batch_size, time_steps, input_size) 

- batch_size：batch的大小
- time_steps：长度，也就是 time_steps 次call
- input_size：表示输入数据单个序列单个时间维度上固有的长度







# 第三部分 TF17

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

拓展阅读

- 如果读者想要深入了解RNN 的结构及其训练方法，建议阅读书籍 Deep Learning（Ian Goodfellow、Yoshua Bengio 和 Aaron Courville 所著）的第10章“Sequence Modeling: Recurrent and Recursive Nets”。 此外，http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 中详细地介绍了RNN 以及Char RNN 的原理，也是很好的阅读材料。

- 如果读者想要深入了解LSTM 的结构， 推荐阅读 http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 。有网友对这篇博文做了翻译，地址为：http://blog.csdn.net/jerr__y/article/ details/58598296。

- 关于TensorFlow 中的RNN 实现，有兴趣的读者可以阅读TensorFlow 源码进行详细了解，地址为：https://github.com/tensorflow/tensorflow/ blob/master/ tensorflow/python/ops/rnn_cell_impl.py 。该源码文件中有BasicRNNCell、BasicLSTMCell、RNNCell、LSTMCell 的实现。









