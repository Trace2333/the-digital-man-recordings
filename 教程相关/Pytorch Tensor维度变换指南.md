# Pytorch Tensor维度变换指南

## 一、常用的Tensor变换操作

1.维度交换

2.维度分割

3.指定维度存储

4.维度求最值index

5.维度重整

6.维度拼接

7.维度扩增和压缩

8.Tensor填充

9.张量view操作

10......

## 二、具体操作说明

### 1.维度交换

张量的维度交换是pytorch框架中的常用操作，会把一个Tensor进行维度的交换，如：[1, 2, 3]-->[3, 2, 1]

掌握维度的自由交换是DL中的必备技能。

**说明：**

```Python
torch.permute(input, dims) → Tensor
	Returns a view of the original tensor input with its dimensions permuted.

		Parameters
			input (Tensor) – the input tensor.

			dims (tuple of python:ints) – The desired ordering of dimensions
```

Example：

```Python
>>>import torch


>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])
```

### 2.维度分割

维度分割操作是如Seq2seq等端到端框架训练的时候的必备操作之一，学会进行指定维度分割可以节省很多的维度变换时间，因此该操作也比较有意义。

**说明：**

```Python
torch.split(tensor, split_size_or_sections, dim=0)[SOURCE]
	Splits the tensor into chunks. Each chunk is a view of the original tensor.

	If split_size_or_sections is an integer type, then tensor will be split into equally 	 sized chunks (if possible). Last chunk will be smaller if the tensor size along the 	 given dimension dim is not divisible by split_size.

	If split_size_or_sections is a list, then tensor will be split into 					len(split_size_or_sections) chunks with sizes in dim according to                         split_size_or_sections.

	Parameters
		tensor (Tensor) – tensor to split.
		split_size_or_sections (int) or (list(int)) – size of a single chunk or list of 		sizes for each chunk
		dim (int) – dimension along which to split the tensor.
```

**Example：**

```Python
>>> a = torch.arange(10).reshape(5,2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> torch.split(a, 2)
(tensor([[0, 1],
         [2, 3]]),
 tensor([[4, 5],
         [6, 7]]),
 tensor([[8, 9]]))
>>> torch.split(a, [1,4])
(tensor([[0, 1]]),
 tensor([[2, 3],
         [4, 5],
         [6, 7],
         [8, 9]]))
```

### 3.指定维度存储

指定维度进行数据放置的操作在实际的库中并没有对应的操作函数，但是这一步操作是需要进行序列预测所必须的，用来存储预测量并之后用于相应的操作。

由于没有对应的操作函数，因此学会怎么存还是比较有意义的。



