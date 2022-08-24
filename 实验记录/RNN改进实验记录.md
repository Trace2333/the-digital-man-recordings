# 实验记录

## 问题描述

在做RNN用于关键词提取的复现时，发现准确率总是趋于0，并且每次的输出都是全部相同的值0，loss在一个比较小的值附近波动，多次训练结果都没有什么变化，在经由胡博士指导并改进后，恢复比较正常的结果，现在做这个实验记录，为日后留下经验，同时也练习自身代码能力

## 排查实验

### 使用未更改的模型结构搭配无误的外部训练和加载代码

**实验原因**：在初期无法确定出问题的地方，因此选择将无误代码中的模型部分替换成没有改正过的模型代码（只有在这个实验里面有用，因为有没有错误的代码，之后要加强练习），看看是不是模型定义或者结构有问题（数据格式和大小一直是统一的）

**实验：**

未改正的模型代码：

```Python
class SrnnNet(nn.Module):
    """
    未修改的网络
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, embw):
        super(SrnnNet, self).__init__()
        self.shared_layer = nn.Sequential(
            RNNLayer(input_size=input_size, hidden_size=hidden_size1)
        )
        self.tow2 = nn.Sequential(
            RNNLayer(input_size=input_size, hidden_size=hidden_size2),
            # neuros related to 2 rnns will be used for sequense classification
            Selectitem(0),  # get the RNN output
            nn.Linear(in_features=hidden_size2, out_features=5),
        )  # for sequence
        self.tow1 = nn.Sequential(
            Selectitem(0),  # get the RNN output
            nn.Linear(in_features=hidden_size1, out_features=2),
        )  # for sentence
        self.dropout = nn.Dropout(0.5)
        self.embedding = embw    # 词嵌入加入参数
    def forward(self, x):
        """返回的是y_pred， z_pred"""
        x = nn.functional.embedding(torch.tensor(x[0], dtype=torch.long).to(device), self.embedding)
        out = self.shared_layer(x)  # out is a tuple
        model_out1 = self.tow1(out)
        model_out2 = self.tow2(out)
        return model_out1, model_out2

```

改后的模型代码：

```Python
class DRNN(nn.Module):
    """
    试验网络
    """
    def __init__(self, inputsize, hiddensize1, hiddensize2, inchanle, outchanle1, outchanle2, batchsize, embw):
        super(DRNN, self).__init__()
        self.hiddensize = hiddensize1
        self.batchsize = batchsize
        self.RNN1 = nn.RNN(900, hiddensize1, batch_first=True, bidirectional=False)
        self.RNN2 = nn.RNN(inputsize, hiddensize2, batch_first=True, bidirectional=False)
        self.Linear1 = nn.Linear(in_features=inchanle, out_features=outchanle1)
        self.Linear2 = nn.Linear(in_features=inchanle, out_features=outchanle2)
        self.dropout = nn.Dropout(0.5)
		self.embw = nn.Parameter(embw)
    def forward(self, inputs):
        """前向计算"""
        state1 = self.init_state(self.batchsize)
        state2 = self.init_state(self.batchsize)
        if isinstance(inputs, tuple):
            x = nn.functional.embedding(torch.tensor(contextwin_2(inputs[0], 3), dtype=torch.long).to(device),
                                        self.embw).flatten(2)
            x = self.dropout(x)
            rnnout1, state1 = self.RNN1(x)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z
        else:
            x = nn.functional.embedding(torch.tensor(contextwin_2(inputs, 3), dtype=torch.int32).to(device),
                                        self.embw).flatten(2)
            x = self.dropout(x)
            rnnout1, state1 = self.RNN1(x)
            rnnout2, state2 = self.RNN2(rnnout1)
            y = self.Linear1(rnnout1)
            z = self.Linear2(rnnout2)
            return y, z

    def init_state(self, batchsize):
        """提供零初始化"""
        return torch.zeros((1, batchsize, self.hiddensize))

```

**结果：**

keyphrase acc:0.34	keywords acc:0.33（没有把词嵌入作为参数）

keyphrase acc:0.68	keywords acc:0.71（把词嵌入加入参数）

因此可以知道，可以跑出一定的精度说明模型没有硬性的bug，所以问题出在训练和处理代码上

### 使用确认无误的模型代码+不确定问题点的训练代码

**实验原因：**已经可以确定出问题大概率出现在训练的代码上，因此要把训练代码单独拿出来，看看是不是这个问题，如果问题再出现了，那就可以确定问题一定是训练代码或者处理代码上

**实验：**

训练代码梗概

```Python
loss1 = 0
loss2 = 0
loss = 0
epochs = 1
lr = 1e-2
# 定义
lossfunction = torch.nn.CrossEntropyLoss()
# 损失函数为交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr)
# 优化器为SGD
model.train()
for epoch in range(epochs):
    iteration = tqdm(train_loader, desc=f"TRAIN on epoch {epoch}")
    for step, inputs in enumerate(iteration):
        output1, output2 = model(
            (inputs[0], torch.randn([1, batch_size, hiden_size1])))
        # 模型计算
        sentence_preds = output1.argmax(axis=2)
        sequence_preds = output2.argmax(axis=2)
        # 得到结果
 		sen_acc = acc_metrics(sentence_preds, inputs[1][0])
        seq_acc = acc_metrics(sequence_preds, inputs[1][1])
        #  指标计算
		loss1 = lossfunction(output1.permute(0, 2, 1), inputs[1][0])
        loss2 = lossfunction(output2.permute(0, 2, 1), inputs[1][1])
        # loss计算,按照NER标准计算
        loss = loss2 * 0.7 + loss1 * 0.3

```

预测：

```Python
evaluation_epochs = 2
# epoch数为2
model.eval()
for epoch in range(evaluation_epochs):
    for step, evaluation_input in enumerate(evaluation_iteration):
        with torch.no_grad():
            output1, output2 = model((evaluation_input[
                0]))  # 模型计算
            sentence_preds = output1.argmax(axis=2)
            sequence_preds = output2.argmax(axis=2)
			#  得到预测结果
            sen_acc = acc_metrics(sentence_preds, evaluation_input[1][0])   
            seq_acc = acc_metrics(sequence_preds, evaluation_input[1][1])
			#  指标计算
```

结果：

![image-7](https://s3.bmp.ovh/imgs/2022/08/14/bc072b89b88ceecd.png)

为了排除训练代码中对数据处理不同造成的影响，debug查看了各个输入流程的数据维度和内容，无误

![image-11](https://s3.bmp.ovh/imgs/2022/08/14/9fa2eb37f4e7b0d3.png)

从梯度上看没有明显的原因，但很明显梯度曲线在快速归0并且与0之间距离小于1e-5这个量级，此时学习率设置为0.01，判断是梯度消失了，再次实验，调小学习率

### 流程和其他模块不变的基础上更改学习率

**实验原因：**在网上查阅教程后并结合梯度图观察后，发现模型梯度很小，导致在优化器中乘以lr后会变得非常小，在参数量级远大于梯度量级的时候，模型参数更新不明显，因此训练没有效果。但是梯度小不能说明问题就是梯度消失，因为RNN本身不容易出现梯度消失而且训练的结果是全部优化为0，所以应该是陷入了局部最优，尝试改学习率试试

**实验：**

将学习率从1e-2~1e-5，并相应跑相同的epoch

```
-lr = 1e-1
+lr = 1e-2
```

```
-lr = 1e-2
+lr = 1e-3
```

```
-lr = 1e-3
+lr = 1e-4
```

```
-lr = 1e-4
+lr = 1e-5
```

**结果：**

结果高度一致，仍然是精度接近0，没有效果

### 只更换优化器并调节学习率实验

**实验原因：**网上查找了资料，因为这个任务里面标签值都没有很大，所以loss计算出来本来就不会很大，在梯度下降的时候可能不太能调到正确的位置，换一下优化器试试

**实验：**

将SGD更改为Adam（lr=1e-4不变）

```
-optimizer = torch.optim.SGD(model.parameters(), lr=lr)
+optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

![image-15](https://s3.bmp.ovh/imgs/2022/08/14/a4579f599e9461a7.png)

**结果：**

成功跳出局部最优，**因此问题出现在优化器上**

但是，SGD虽然是直白的随机优化，但其他优化器如这里使用的Adam也是随机优化的原理，为什么SGD却得到了局部最优？

### 定位问题在优化器，探查原因

**实验原因**：对原因不是很理解，因为已经有精度了，所以训练代码理论上不会有很大的bug导致试验失败，所以再次用SGD试试

**实验：**

```
-optimizer = torch.optim.Adam(model.parameters(), lr=lr)
+optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

**结果：**

![image-20](https://s3.bmp.ovh/imgs/2022/08/14/7c6a48c25d56a6ec.png)

问题定位应该无误，是SGD出现了问题

### 定位问题在优化器，探查原因

**实验原因：**已经知道模型没有精度是因为参数更新没有效果，所以多次对同一学习率不同的优化器进行观察，对参数着重观察

**实验：**

(只展示了一张图)

![image-19](https://s3.bmp.ovh/imgs/2022/08/14/31e1bc826649f8a0.png)

**结果：**

观察每个层的参数图，可以发现每一次除了Adam的参数有明显跳动外，都呈现一条近似直线，更新幅度很小，所以除了使用Adam都没有学到数据的特征，出现问题的根本原因还是参数更新有问题

## 结论和分析

从输入数据上是可以看出来，模型存在局部最优点，并且是比较接近于全局最优点的（当输出恒定为0的时候，loss在一个不是很大的值波动）。并且并不会由loss1和loss2相互影响（二者局部最优一致）。

在没有经过的初始点处就是一个比较平坦的位置，因此在SGD优化的时候难以感知到微小的梯度变化，那么很可能走错方向，即向局部最优优化或输出呈现随机状态，它由于没有加速度的概念，在陷入局部之后也很难跳出。参数更新程度也会非常小（**grad比较小是因为在这个任务梯度本来就不大**）

## 经验总结

1.一般来说，如果代码大体都是按照标准写出来的都不会有非常大的问题，出现问题的地方都是一些细节处的bug，所以在之后写代码时需要更加仔细，如果有不确定的细节应该去着重求证，很多时候问题就是处在这些地方，查找也不方便。

2.代码规范要有，每一次写的时候如果代码没有规范，那么和其他人交流就没有效率，每次的实验也容易无法定位，而且在未来工作中代码规范和整洁必不可少，所以代码规范也很重要。

3.需要有提交的习惯，每次的改动尽量向git提交一次，可以避免每次的改动混乱，同时出了问题也方便回档和debug
