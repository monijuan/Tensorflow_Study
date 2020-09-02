![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902162932165.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


代码链接：[https://github.com/MONI-JUAN/Tensorflow_Study/tree/master/TensorFlow%2006%E2%80%94%E2%80%94ch04-%E7%8E%A9%E8%BD%ACDeepDream](https://github.com/MONI-JUAN/Tensorflow_Study/tree/master/TensorFlow%2006%E2%80%94%E2%80%94ch04-%E7%8E%A9%E8%BD%ACDeepDream)

准备工作：下载预训练的 Inception 模型，ch4 的 tensorflow_inception_graph.pb
链接：https://pan.baidu.com/s/1IC5x_md8NNZr5pL3ccgjOQ
提取码：fjct


DeepDream是Google 开源了用来分类和整理图像的 AI 程序 Inceptionism。

这篇文章将一步步实现上图！

## 一、导入Inception模型并查看卷积层 

```python
# coding:utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf

graph = tf.Graph()  # 创建图
sess = tf.InteractiveSession(graph=graph)  # 创建Session
model_fn = 'tensorflow_inception_graph.pb'  # 存储了inception的网络结构和对应的数据 
with tf.gfile.FastGFile(model_fn, 'rb') as f:  # 导入inception
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 输入图像需要经过处理才能送入网络中
t_input = tf.placeholder(np.float32, name='input')  # 输入的图像
imagenet_mean = 117.0  # 需要减去一个均值
# expand_dims是加一维，从[height, width, channel]变成[1, height, width, channel]
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# 找到所有卷积层，输出卷积层层数
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
print('Number of layers', len(layers))

# 输出mixed4d_3x3_bottleneck_pre_relu的形状
name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s: %s' % (name, str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902130323325.png#pic_center)



## 二、生成原始的 Deep Dream 图像

```python
# coding: utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

# 以把一个 numpy.ndarray 保存成文件的形式。
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


# 渲染，实际上就是优化 layer_output[:, :, :, channel] 的平均值
def render_naive(t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # t_score是优化目标，是t_obj的平均值   
    t_grad = tf.gradients(t_score, t_input)[0] # 计算 t_score 对 t_input 的梯度   
    img = img0.copy() # 创建新图
    for i in range(iter_n):
        # 在sess中计算梯度，以及当前的score，越大代表平均激活越大
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # 对 img 应用梯度， step 可以看做“学习率”
        g /= g.std() + 1e-8
        img += g * step
        print('score(mean)=%f' % (score))   
    savearray(img, 'naive.jpg') # 保存图片


# 导入Inception模型
graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

name = 'mixed4d_3x3_bottleneck_pre_relu' # 定义卷积层
channel = 139 # 定义通道数，总通道是144，这里用139通道举例
layer_output = graph.get_tensor_by_name("import/%s:0" % name)# 并取出对应的tensor
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0 # 定义原始的图像噪声
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=20) # 渲染并保存

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902130351729.png#pic_center)

![经过20次迭代后，生成的图像](https://img-blog.csdnimg.cn/20200902130118568.jpg#pic_center)
噪声图经过20次迭代后，生成的图像

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902130153151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
可以看到，原始的噪声图经过一次次优化139通道后越来越有花的样子了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902130930591.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
顺便玩了一下通道100，这大概是鱼？

## 三、生成更大只寸的 Deep Drearn 图像

```python
# coding:utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf


# 以把一个 numpy.ndarray 保存成文件的形式。
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


# 每次将将图片放大octave_scale倍
def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img


# 计算任意大小图像的梯度
def calc_grad_tiled(img, t_grad, tile_size=512):
    # 每次只对tile_size×tile_size大小的图像计算梯度，避免内存问题
    sz = tile_size
    h, w = img.shape[:2]
    # img_shift：先在行上做整体移动，再在列上做整体移动
    # 防止在tile的边缘产生边缘效应
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    # y, x是开始位置的像素
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            # 每次对sub计算梯度。sub的大小是tile_size×tile_size
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    # 使用np.roll移动回去
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # t_score是优化目标，是t_obj的平均值   
    t_grad = tf.gradients(t_score, t_input)[0] # 计算 t_score 对 t_input 的梯度 
    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            # 每次将将图片放大octave_scale倍，共放大octave_n - 1 次            
            img = resize_ratio(img, octave_scale)
        for i in range(iter_n):            
            g = calc_grad_tiled(img, t_grad) # 计算任意大小图像的梯度
            g /= g.std() + 1e-8
            img += g * step
            print('.', end=' ')
    savearray(img, 'multiscale.jpg')


# 导入Inception模型
graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
layer_output = graph.get_tensor_by_name("import/%s:0" % name)
render_multiscale(layer_output[:, :, :, channel], img_noise, iter_n=20)

```












![octave_n=3](https://img-blog.csdnimg.cn/20200902144837597.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
octave_n=3


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902145133422.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
octave_n=5

已经很有花的样子了！


## 四、生成更高质量的 Deep Drearn 图像
### 1.怎么提高质量：放大低频梯度

生成的图像在细节部分变化还比较剧烈，而希望图像整体的风格应该比较“柔和”。之前生成图像的高频成分太多，如果低频成分应该多一些，这样生成的图像才会更加“柔和”。

>**高频成分**：是指图像中灰度、颜色、明度变化比较剧烈的地方，如边缘、细节部分。
>**低频成分**：是指图像变化不大的地方，如大块色块、整体风格。

如何让图像具高更多的低频成分而不是高频成分？一种方法是针对高频成分加入损失，这样图像在生成的时候就会因为新加入损失的作用而发生改变。但加入损失会导致计算量和收敛步数的增大。
另一种方法：放大低频梯度。

### 2.具体方法：拉普拉斯金字塔梯度标准化

之前生成图像时，使用的梯度是统一的。如果可以对梯度作分解，将之分为“高频梯度”“低频梯度’，再人为地去放大低频梯度”，就可以得到较为柔和的图像了。

在具体实践上，使用拉普拉斯金字塔（Laplacian Pyramid）对图像进行分解。这种算法可以把图像分解为多层，如图所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902150609409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)


> level1、level2对应图像的高频成分
> level3、level4对应图像的低频成分

可以对梯度也做这样的分解。分解之后，对高频的梯度和低频的梯度都做标准化，可以让梯度的低频成分和高频成分差不多，表现在图像上就会增加图像的低频成分，从而提高生成图像的质量。通常称这种方法为**拉普拉斯金字塔梯度标准化（Laplacian Pyramid Gradient Normalization）**。

### 3.代码

```python
# coding:utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf


# lap_split 和 lap_merge 要用到
k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)


# 以把一个 numpy.ndarray 保存成文件的形式。
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


# 每次将将图片放大octave_scale倍
def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img


# 计算任意大小图像的梯度
def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


# 这个函数将图像分为低频和高频成分
def lap_split(img):
    with tf.name_scope('split'):
        # 做过一次卷积相当于一次“平滑”，因此lo为低频成分
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
        # 低频成分放缩到原始图像一样大小得到lo2，再用原始图像img减去lo2，就得到高频成分hi
        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img - lo2
    return lo, hi


# 将图像img分成n层拉普拉斯金字塔
def lap_split_n(img, n):
    levels = []
    for i in range(n):
        # 递归调用lap_split将图像分为低频和高频部分    
        img, hi = lap_split(img) # 低频部分再继续分解
        levels.append(hi) # 高频部分保存到levels中
    levels.append(img)
    return levels[::-1]


# 将拉普拉斯金字塔还原到原始图像
def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img


# 对img做标准化
def normalize_std(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)


# 拉普拉斯金字塔标准化
def lap_normalize(img, scale_n=4):
    img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, scale_n) # 将图像img分成n层拉普拉斯金字塔
    tlevels = list(map(normalize_std, tlevels)) # 每一层都做标准化
    out = lap_merge(tlevels)
    return out[0, :, :, :]


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


def render_lapnorm(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # t_score是优化目标，是t_obj的平均值   
    t_grad = tf.gradients(t_score, t_input)[0] # 计算 t_score 对 t_input 的梯度 
    # 将lap_normalize转换为正常函数
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            img = resize_ratio(img, octave_scale)
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # 唯一的区别在于使用lap_norm_func来标准化g
            g = lap_norm_func(g)
            img += g * step
            print('.', end=' ')
    savearray(img, 'lapnorm.jpg')


if __name__ == '__main__':  
    graph = tf.Graph()
    model_fn = 'tensorflow_inception_graph.pb'
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})


    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    render_lapnorm(layer_output[:, :, :, channel], img_noise, iter_n=20)

```

### 4.效果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902160834835.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
终于搞定了！！！

## 五、带背景的 Deep Dream 模型

其实就是一开始传一个图片进去当作背景

```python
# coding:utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf



def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def visstd(a, s=0.1):
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img


def resize(img, hw):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, hw))
    img = img / 255 * (max - min) + min
    return img


def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  # 先在行上做整体移动，再在列上做整体移动
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


def render_deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    img = img0
    # 同样将图像进行金字塔分解
    # 此时提取高频、低频的方法比较简单。直接缩放就可以
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)
    # 先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')
    img = img.clip(0, 255)
    savearray(img, 'deepdream.jpg')


if __name__ == '__main__':    
    graph = tf.Graph()
    model_fn = 'tensorflow_inception_graph.pb'
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})


    img0 = PIL.Image.open('test.jpg')
    img0 = np.float32(img0)

    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    render_deepdream(layer_output[:, :, :, channel], img0)

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902161601376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDUxOTA5,size_16,color_FFFFFF,t_70#pic_center)
传进去的左图，输出右图