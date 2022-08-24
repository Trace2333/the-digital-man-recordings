# TorchMetrics的使用和评估方法

## TorchMetrics介绍

日常的学习和工作中，我们一般需要完成两方面的工作，一方面是完成模型的创新、搭建和训练，另一方面是完成模型的评估工作，**TorchMetrics**即是用来完成模型的评估工作，该第三方库可以完成如Precision、Recall、F1-score等的自动计算和分析

## TorchMetrics基本使用方法

### Install

You can install TorchMetrics using pip or conda:

```
# Python Package Index (PyPI)
pip install torchmetrics
# Conda
conda install -c conda-forge torchmetrics
```

Eventually if there is a missing PyTorch wheel for your OS or Python version you can simply compile [PyTorch from source](https://github.com/pytorch/pytorch):

```
# Optional if you do not need compile GPU support
export USE_CUDA=0  # just to keep it simple
# you can install the latest state from master
pip install git+https://github.com/pytorch/pytorch.git
# OR set a particular PyTorch release
pip install git+https://github.com/pytorch/pytorch.git@<release-tag>
# and finalize with installing TorchMetrics
pip install torchmetrics

```

### Using TorchMetrics

该第三方库支持多种数据方式的计算，如已经完成softmax等的prediction、没有进行Argmax的prediction等等，返回值统一为一个单值Tensor。

下面演示最基本的Accuracy的计算

```Python
"""
For TorchMetrics Learning
The Structure contains lots metrics
that can be used such TF/TN/F1
"""
import torch
import torchmetrics as metrics
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))
print("prediction_Size:", preds.size(), "Prediction:\n", preds)
print("Targets_Size:", target.size(), "Target:\n", target)

acc = metrics.functional.accuracy(preds, target)
print(acc)
>>>>OUT:
    
prediction_Size: torch.Size([10, 5]) Prediction:
 tensor([[0.0137, 0.0750, 0.2646, 0.4176, 0.2291],
        [0.0837, 0.1342, 0.3691, 0.2730, 0.1400],
        [0.1338, 0.2271, 0.0677, 0.2677, 0.3037],
        [0.3678, 0.0623, 0.1529, 0.2548, 0.1622],
        [0.3986, 0.1422, 0.1494, 0.2322, 0.0776],
        [0.2615, 0.1866, 0.0843, 0.1045, 0.3631],
        [0.3114, 0.2444, 0.2587, 0.0430, 0.1426],
        [0.3076, 0.2549, 0.3380, 0.0510, 0.0486],
        [0.1187, 0.3504, 0.0637, 0.4327, 0.0345],
        [0.3216, 0.2900, 0.1487, 0.1611, 0.0786]])
Targets_Size: torch.Size([10]) Target:
 tensor([1, 0, 3, 4, 1, 4, 3, 3, 4, 4])
tensor(0.1000)

```

```Python
import torch
import torchmetrics as metrics
preds = torch.randn(10, 10, 5).softmax(dim=-1).argmax(axis=2)
target = torch.randint(5, (10, 10))
print("prediction_Size:", preds.size(), "Prediction:\n", preds)
print("Targets_Size:", target.size(), "Target:\n", target)

acc = metrics.functional.accuracy(preds, target)
print(acc)

>>>OUT:
    prediction_Size: torch.Size([10, 10]) Prediction:
 tensor([[0, 0, 4, 1, 1, 1, 3, 2, 0, 1],
        [4, 2, 2, 4, 1, 1, 3, 3, 2, 3],
        [4, 1, 1, 4, 0, 4, 0, 1, 1, 0],
        [3, 2, 2, 3, 4, 3, 4, 2, 1, 1],
        [1, 3, 0, 0, 0, 4, 0, 1, 0, 1],
        [2, 0, 2, 2, 2, 2, 1, 1, 2, 2],
        [2, 3, 3, 1, 3, 0, 2, 1, 1, 2],
        [3, 3, 4, 4, 1, 0, 4, 1, 0, 2],
        [3, 3, 0, 4, 0, 3, 0, 4, 3, 3],
        [2, 2, 2, 3, 2, 4, 0, 0, 0, 2]])
Targets_Size: torch.Size([10, 10]) Target:
 tensor([[1, 3, 1, 4, 2, 3, 0, 2, 0, 2],
        [2, 1, 1, 2, 4, 4, 4, 4, 1, 0],
        [4, 3, 3, 3, 3, 0, 2, 2, 0, 0],
        [4, 2, 3, 3, 3, 3, 0, 2, 4, 2],
        [1, 0, 4, 0, 1, 0, 3, 1, 3, 4],
        [3, 2, 1, 1, 3, 2, 4, 1, 3, 4],
        [3, 3, 4, 3, 3, 3, 2, 2, 1, 1],
        [0, 1, 4, 0, 1, 4, 0, 3, 0, 4],
        [2, 2, 3, 0, 0, 4, 4, 4, 2, 4],
        [0, 1, 0, 2, 1, 4, 0, 1, 0, 1]])
tensor(0.2500)

```

由演示代码可以知道，导入计算最基本的是选择合适的评估函数和一对batch_size,num_seq相同的Tensor即可，代入计算可自动得到结果评估量

### Batch or Epoch Compute

利用库的计算特性，同样可以对多个级别进行评估



```Python
"""
For TorchMetrics Learning
The Structure contains lots metrics
that can be used such TF/TN/F1
"""
import torch
import torchmetrics
# initialize metric
metric = torchmetrics.Accuracy()

n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    # metric on current batch
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")

# metric on all batches using custom accumulation
acc = metric.compute()
print(f"Accuracy on all data: {acc}")

# Reseting internal state such that metric ready for new data
metric.reset()

>>>OUT:
	Accuracy on batch 0: 0.10000000149011612
Accuracy on batch 1: 0.4000000059604645
Accuracy on batch 2: 0.10000000149011612
Accuracy on batch 3: 0.30000001192092896
Accuracy on batch 4: 0.10000000149011612
Accuracy on batch 5: 0.10000000149011612
Accuracy on batch 6: 0.10000000149011612
Accuracy on batch 7: 0.4000000059604645
Accuracy on batch 8: 0.20000000298023224
Accuracy on batch 9: 0.0
Accuracy on all data: 0.18000000715255737
```

由代码可以看出来，metrics是一个计算对象，每一次将其用于计算将会记录之前的结果，它也包含多个类方法，如compute(计算平均)、update(更新数据)

