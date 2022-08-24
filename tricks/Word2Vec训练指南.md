# Word2Vec训练词嵌入指南

## 一、本内容采用第三方库gensim实现

手写指南见《word2vec手写训练指南》

word2vec介绍在这里省略，重点介绍如何在一个NLP任务中**快速训练得到word2vec的词表和词表向量来作为embedding底层输入**

## 二、安装gensim

```python
pip install gensim
if pip is availble else:
conda install gensim
```

安装完成后如能够正常导入gensim.models.word2vec即为成功

## 三、数据预处理

由于我们需要导入的文本数据中存在很多预训练词向量中并不存在的词汇，**因此需要间断训练**或者**重新开一个词表进行训练**。下面是完成处理的数据格式

```
input_data = [
['你'，’好‘]，
['在'，’干‘， ’什‘， ’么‘]，
['正'，'在'，'聊', '天', '呢']
]
```

## 四、Word2vec模型

在使用gensim库时需要导入包gensim，内部导入的模块如下:

```python
from gensim.models import word2vec
model = word2vec.Word2Vec(args)
```

```
参数24个：
参数名称	默认值	用途
sentences	None	训练的语料，一个可迭代对象。对于从磁盘加载的大型语料最好用gensim.models.word2vec.BrownCorpus，gensim.models.word2vec.Text8Corpus ，gensim.models.word2vec.LineSentence 去生成sentences
size	100	生成词向量的维度
alpha	0.025	初始学习率
window	5	句子中当前和预测单词之间的最大距离，取词窗口大小
min_count	5	文档中总频率低于此值的单词忽略
max_vocab_size	None	构建词汇表最大数，词汇大于这个数按照频率排序，去除频率低的词汇
sample	1e-3	高频词进行随机下采样的阈值，范围是(0, 1e-5)
seed	1	向量初始化的随机数种子
workers	3	几个CPU进行跑
min_alpha	0.0001	随着学习进行，学习率线性下降到这个最小数
sg	0	训练时算法选择 0:skip-gram, 1: CBOW
hs	0	0: 当这个为0 并且negative 参数不为零，用负采样，1：层次 softmax
negative	5	负采样，大于0是使用负采样，当为负数值就会进行增加噪音词
ns_exponent	0.75	负采样指数，确定负采样抽样形式：1.0：完全按比例抽，0.0对所有词均等采样，负值对低频词更多的采样。流行的是0.75
cbow_mean	1	0:使用上下文单词向量的总和，1:使用均值； 只适用于cbow
hashfxn	hash	希函数用于随机初始化权重，以提高训练的可重复性。
iter	5	迭代次数，epoch
null_word	0	空填充数据
trim_rule	None	词汇修剪规则，指定某些词语是否应保留在词汇表中，默认是 词频小于 min_count则丢弃，可以是自己定义规则
sorted_vocab	1	1：按照降序排列，0：不排序；实现方法：gensim.models.word2vec.Word2VecVocab.sort_vocab()
batch_words	10000	词数量大小，大于10000 cython会进行截断
compute_loss	False	损失(loss)值，如果是True 就会保存
callbacks	()	在训练期间的特定阶段执行的回调序列~gensim.models.callbacks.CallbackAny2Vec
max_final_vocab	None	通过自动选择匹配的min_count将词汇限制为目标词汇大小,如果min_count有参数就用给定的数值

```

其中重点使用的参数为**sentence**、**size**、**epochs**、**min_count**

## 五、完成训练和保存

### 1.完成和运行训练

运行完成model行即会正常开始运行，如果需要进行INFO检测就需要运行logging库（输出logging.INFO级别的信息）。

```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

### 2. 保存模型和保存词文件

保存模型并保存训练文件

```python
model.save(filename)
#该方法将会保存为三个文件，一个vocab和两个训练信息文件（.npy），兼具有向量信息和训练信息，
```

```python
model.wv.save_word2vec_format("filename。xxxx")
#该方法将会根据文件目标文件格式的不同保存为不同的形式，但此文件只有词向量信息但没有训练信息，因此不能断点训练
```

## 六、再加载和增量训练

### 1.再加载

#### 1°词向量再加载

```python
from gensim.word2vec import KeyedVectors
embedding = KeyedVectors.load_word2vec_format(filename or path)
#完成加载后只有一个向量对象，之后的使用可以作为词典来使用
```

### 2增量训练（相对不常用）

link：[(111条消息) Word2Vec模型增量训练_Xiaozhu_a的博客-CSDN博客_word2vec增量训练](https://blog.csdn.net/qq_43404784/article/details/83794296)



