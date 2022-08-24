



# KeyBert

keybert是一种基于语义信息的深度学习方法，用来进行相应的长/短文本关键词提取

## 核心方法——利用Bert的语义表示准确性来进行关键词找寻

Keybert的核心在于bert预训练模型的利用，bert预训练模型中对大规模文本信息进行了特征抽取，因此bert进行文本特征表示的时候生成的Text Vector是有较高的准确性的。之后利用词的Vector表示成文本整体的Embedding

## 特点——无监督自标注、预处理模型、领域独立性、简单

### 1.无监督自标注

利用双向LSTM进行文本特征标注，输出已经完成标注的文本

![image-20220709171147448](C:\Users\Trace\AppData\Roaming\Typora\typora-user-images\image-20220709171147448.png)

### 2.预训练模型

利用bert预训练模型作为embedding层

### 3.领域独立性

设计并标注收集了数据集——domain independent

![image-20220709171438807](C:\Users\Trace\AppData\Roaming\Typora\typora-user-images\image-20220709171438807.png)

### 4.简单

核心就是要找到整体向量和n-gram最近词，相似度直接利用余弦相似度进行计算，计算量比较小
$$
similarity=cos<Text embedding,n-gram embedding>
$$


## 缺点——整体文本表示法、N-gram选择

### 1.整体文本表示欠佳

bert的最大输入维度是512，过长的文本只能使用裁剪的方法，或者整体向量利用所有文本向量的平均来得到，如果是前者，则长文本的抽取效果欠佳，如果是后者，则文本的表示会因为大多数的无语意词造成偏移

### 2.N-gram选择

理论上而言文本的关键词是没有词语长度的限制的，并且是不同长度文本的混合，在预测的时候没有考虑到