# UniKeyphase——利用预训练模型以及Local Relation Layer（LR）来进行关键词提取以及生成

## 一、Motivation

提出了基于预训练生成模型UNILM（Language Model）的生成和抽取的联合模型，结构为unilm-->SRL（stacked local relation layer）-->BOW（bags of words）。

**输入原文序列，输出抽取的关键词（present Keyphrase）和生成的关键词（absent keyphrase）。**

## 二、Total Structure

![1](C:\Users\Trace\Desktop\1.jpg)

## 三、UniLM

全称为Unified Language Model

继承了Bert的总体结构，将掩盖随意的token并预测完整句子更改为了三类预训练任务，即：

​	**1.unidirectional prediction**

​	**2.bidirectional prediction**

​	**3.sequence to sequence prediction**

下面分别介绍三类预训练任务及其目标

