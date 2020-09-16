@[TOC](目录)

## 基本概念

### 1.词嵌入

**词嵌入**：将一个词语（word）转换成一个向量（vector），用word2vec表示词嵌入

**独热编码**：每个字母用一个向量表示，例如表示小写字母，则需要一个长度为26的向量表示每个字母，独热表示完全平等地看呆了单词表中地所有单词，忽略了词与词之前的联系。但是有些单词之间相似性更大，反而会被忽视

**word2vec** ：学习一个映射 $f$ ，把单词变成向量， $vec=f(word)$ 。如256维或512维，可以更加高效的方式表示单词，会有更丰富的有关词语的信息，性能会大大提高。

### 2.获取映射关系 $f$ 

方法：

- 基于“计数”：将经常同时出现的词映射到向量空间的相近位置，例如CBOW
- 基于“预测”：从一个词或几个词出发，预测它们可能德相邻词，通常用这种，例如Skip-Gram

## 方法一：CBOW

CBOW 的全称为 Continuous Bag of Words，即连续词袋模型，色的核心 思想是利用某个词语的上下文预测这个词语。 

### 1.一个词预测一个词

输入x，经过全连接隐藏层到h，h经过全连接层输出y。

（V 是词汇表中词的数量，独热表示的 x 的形状为(V,) ）

输出 y 相当于做 Softmax 操作前的 logits，形状也是 (V,) ，是用一个词预测另一个词。

隐层的值被当作是词的嵌入表示，即 word2vec 中的 “vec”





![在这里插入图片描述](https://img-blog.csdnimg.cn/20200915220123152.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 2.多个词预测一个词

到隐含层的时候，把全连接的值都加起来



![在这里插入图片描述](https://img-blog.csdnimg.cn/2020091522083416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



对应的网络结构：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200915220924463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



如图中的 “the cat sits on the”，预测 "mat" 

整个网络相当于是 V 类分类器，V 通常非常大，因此简单地修改网络将 V 分为两类。

**也就是判断是否为“噪声词汇”。**

优化函数：


$$
J=\ln Q_{\theta}\left(D=1 \mid \boldsymbol{w}_{t}, \boldsymbol{h}\right)+k \underset{\widetilde{w} \sim P_{\text {noise }}}{E}  \left[\ln Q_{\theta}(D=0 \mid \tilde{\boldsymbol{w}}, \boldsymbol{h})\right]
$$

其中：

- $h$：上下文
- $w_t$：真正的目标词汇
- $\widetilde{w}$：噪声词汇
- $Q_{\theta}$：Logistic 回归得到的概率

选取噪声词进行两类分类的 CBOW 模型：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200915223902279.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)

隐含层可以看作是 word2vec 中的 “vec” 向量。

对于一个单词，先将独热表示输入模型，隐含层的值对应的词的嵌入表示。

在 Tensorflow 中，损失被称为 NCE 损失，对应的函数为 tf.nn.nec_loss。








## 方法二：Skip-Gram


Skip-Gram 方法和 CBOW 方法正好相反： 使用“出现的词”来预测它“上下文文中词”。

如在之前的句子中，是使用 “woman” ，来预测“man”，“fell”等单词 。 

所以，可以把 Skip-Gram 方法看作从一个单词预测另一个单词的问题。

后面是用 Tensorflow 训练一个 Skip-Gram 方法的词嵌入。



## 训练 Skip-Gram

直接运行 `word2vec_basic.py` 即可

```
python word2vec_basic.py
```

详细看一下其中六个步骤的代码

### 第一步：下载语料库



```python
####################################################################
# 第一步: 下载语料库
print("----------------------------------")
print("第一步: 下载语料库")
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """
  这个函数的功能是：
      如果filename不存在，就在上面的地址下载它。
      如果filename存在，就跳过下载。
      最终会检查文字的字节数是否和expected_bytes相同。
  """
  if not os.path.exists(filename):
    print('start downloading...')
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

def read_data(filename):
  """
  这个函数的功能是：
      将下载好的zip文件解压并读取为word的list
  """
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

filename = maybe_download('text8.zip', 31344016) # 下载语料库text8.zip
vocabulary = read_data(filename) # 将语料库解压，并转换成一个word的list
print('Data size', len(vocabulary)) # 总长度为1700万左右
print(vocabulary[0:100]) # 输出前100个词。
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916131605915.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



text8.zip一共29.8mb，如果下不下来，可以手动下载放到当前目录：http://mattmahoney.net/dc/text8.zip



### 第二步：制作词表

```python
####################################################################
# 第二步: 制作一个词表，将不常见的词变成一个UNK标识符
print("----------------------------------")
print("第二步: 制作词表")

def build_dataset(words, n_words):
  """
  函数功能：将原始的单词表示变成index
  """
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # UNK的index为0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

vocabulary_size = 50000 # 词表的大小为5万（即我们只考虑最常出现的5万个词）
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  # 删除以节省内存
print('Most common words (+UNK)', count[:5]) # 输出最常出现的5个单词
print('Sample data（before):', [reverse_dictionary[i] for i in data[:10]]) # 原来前10个单词
print('Sample data（later) :', data[:10]) # 转换后的data
data_index = 0 # 下面使用data来制作训练集
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916131614868.png#pic_center)



### 第三步：生成batch

一个 batch 可以看作是一些“单词对”的集合，如 `woman -> man` ， `woman -> fell`

箭头左边表示“出现的单词”，右边表示该单词所在“上下文”中的单词。



第三步主要是定义一个函数，用于生成skip-gram模型用的batch

> generate_batch(batch_size, num_skips, skip_window)



输入：

- batch_size：一个batch中单词对的个数

- num_skips ：”上下文“数

  skip_window：”上下文“候选数

  生成单词对的时候，先在语料库中选取长度为（skip_window * 2+1）连续单词列表，最中间的也就是Skip-Gram 方法中”出现的单词“，其余的（skip_window * 2）是”上下文“，在这当中选取 num_skips 个词作为”上下文“放入 labels。



输出：

- batch：Skip-Gram 方法中”出现的单词“，形状为 (batch_size, )
- labels：”上下文“中的单词 (batch_size, 1)

```python
####################################################################
# 第三步：定义一个函数，用于生成skip-gram模型用的batch
print("----------------------------------")
print("第三步：生成batch")

def generate_batch(batch_size, num_skips, skip_window):
  # data_index相当于一个指针，初始为0
  # 每次生成一个batch，data_index就会相应地往后推
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  # data_index是当前数据开始的位置
  # 产生batch后就往后推1位（产生batch）
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    # 利用buffer生成batch
    # buffer是一个长度为 2 * skip_window + 1长度的word list
    # 一个buffer生成num_skips个数的样本
    # print([reverse_dictionary[i] for i in buffer])
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window] # 保证样本不重复
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    # 每利用buffer生成num_skips个样本，data_index就向后推进一位
    data_index = (data_index + 1) % len(data)
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

# 默认情况下skip_window=1, num_skips=2
# 此时就是从连续的3(3 = skip_window*2 + 1)个词中生成2(num_skips)个样本。
# 如连续的三个词['used', 'against', 'early']
# 生成两个样本：against -> used, against -> early
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916131714469.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 第四步：建立模型

变量 embeddings ，形状是（vocabulay_size, embedding_size）,对于id单词的嵌入是 embeddings[id,:]

输入数据 train_inputs，用 tf.nn.embedding_lookup 转成对应的向量 embed

再对比 embed和输入的标签 train_lavels，用 tf.nn.nec_loss 定义NCE的损失



训练模型的时候还希望对模型进行验证，所以取出”验证单词“的时候由于 embeddings 各个维度的大小可能不一样，所以需要做一次归一化，归一化后的 normalized_embeddings 计算验证词和其他单词的相似度。



```python
####################################################################
# 第四步：建立模型
print("----------------------------------")
print("第四步：建立模型")
batch_size = 128
embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量
skip_window = 1       # skip_window参数和之前保持一致
num_skips = 2         # num_skips参数和之前保持一致

# 在训练过程中，会对模型进行验证 
# 验证的方法就是找出和某个词最近的词。
# 只对前valid_window的词进行验证，因为这些词最常出现
valid_size = 16     # 每次验证16个词
valid_window = 100  # 这16个词是在前100个最常见的词中选出来的
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # 构造损失时选取的噪声词的数量

# 建立模型
graph = tf.Graph()
with graph.as_default():
  # 输入的batch
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  # 用于验证的词
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  # 在cpu上定义模型
  with tf.device('/cpu:0'):
    # 定义1个embeddings变量，相当于一行存储一个词的embedding
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 利用embedding_lookup可以轻松得到一个batch内的所有的词嵌入
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # tf.nn.nce_loss会自动选取噪声词，并且形成损失。
  # 随机选取num_sampled个噪声词
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # 得到loss后，就可以构造优化器了
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  # 计算词和词的相似度，用于验证
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  # 找出和验证词的embedding并计算它们和所有单词的相似度
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
  # 变量初始化步骤
  init = tf.global_variables_initializer()
```



### 第五步：开始训练

```python
####################################################################
# 第五步：开始训练
print("----------------------------------")
print("第五步：开始训练")
num_steps = 100001

with tf.Session(graph=graph) as session: 
  init.run() # 初始化变量
  print('Initialized')
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict) # 优化一步
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000     
      print('Average loss at step ', step, ': ', average_loss) # 2000个batch的平均损失
      average_loss = 0

    # 每1万步，我们进行一次验证
    if step % 10000 == 0:     
      sim = similarity.eval() # 验证词与所有词之间的相似度
      # 一共有valid_size个验证词
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # 输出最相邻的8个词语
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  # final_embeddings 是最后得到的 embedding 向量
  # 它的形状是[vocabulary_size, embedding_size]
  # 每一行就代表着对应index词的词嵌入表示
  final_embeddings = normalized_embeddings.eval()
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916131626764.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



从结果可以发现，一开始输出的单词是随机的，没有什么意义。

但是训练到最后的时候，输出的词汇还是比较接近的，说明embedding空间的向量表示具备了一定含义：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916131630460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)



### 第六步：可视化

得到 final_embeddings 后对嵌入空间进行可视化表示，使用 t-SNE 方法把128维变成2维，花了500个词的位置，保存为 tsne.png



```python
####################################################################
# 第六步：可视化
print("----------------------------------")
print("第六步：可视化")
# 可视化的图片会保存为“tsne.png”

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib
  matplotlib.use('agg')
  import matplotlib.pyplot as plt
  # 因为我们的embedding的大小为128维，没有办法直接可视化
  # 所以我们用t-SNE方法进行降维
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  # 只画出500个词的位置
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916132144496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)