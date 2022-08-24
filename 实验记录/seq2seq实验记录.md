# seq2seq实验记录

seq2seq模型实验以从英语到中文的端到机器翻译任务为例子，学习seq2seq模型及其attention、多层encoder-decoder的堆叠方式和堆叠作用。

## 一、RNN-RNN编码解码

### 1.实验描述

利用RNN进行初始编码解码

### 2.代码

```Python
class encoderBase(nn.Module):
    """
    基本编码器
    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers):
        super(encoderBase, self).__init__()
        self.layers = numLayers
        self.batch = batchSize
        self.hiddenSize = hiddenSize
        self.recurent = nn.RNN(inputSize, batch_first=True, bidirectional=True,
                               num_layers=numLayers, hidden_size=hiddenSize)  # Using a 2 direction RNN to read the input sentences

    def forward(self, x):
        """前向传播"""
        state = self.initZeroState()
        out = self.recurent(x, state)
        return out

    def initZeroState(self):
        """零初始化隐藏层"""
        return torch.zeros(self.layers * 2, self.batch, self.hiddenSize).to(device)

    def initNormState(self):
        """正泰分布初始化隐藏层"""
        return torch.randn([self.layers * 2, self.batch, self.hiddenSize]).to(device)


class decoderBase(nn.Module):
    """
    基本解码器
    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, dictLen):
        super(decoderBase, self).__init__()
        """初始化模型"""
        self.layers = numLayers
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.batchSize = batchSize
        self.recurrent = nn.RNN(inputSize, hiddenSize, batch_first=True, bidirectional=True, num_layers=numLayers)
        self.linear = nn.Linear(in_features=hiddenSize * 2, out_features=dictLen)

    def forward(self, x, state):
        """前向计算"""
        out, state = self.recurrent(x, state)
        y = self.linear(out)
        return y, state

    def initZeroState(self):
        """隐藏层零初始化"""
        return torch.zeros(self.layers * 2, 2, self.hiddenSize).to(device)

    def initNormState(self):
        """隐藏层正态初始化"""
        return torch.randn([self.layers * 2, 2, self.hiddenSize]).to(device)


class seq2seqBase(nn.Module):
    """
    基本seq2seq模型
    利用EncoderBase和DecoderBase搭建
    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, dictLen, embwEN, embwZH):
        """层初始化"""
        super(seq2seqBase, self).__init__()
        self.encoder = encoderBase(inputSize, hiddenSize, batchSize, numLayers)
        self.decoder = decoderBase(hiddenSize, hiddenSize, batchSize, numLayers, dictLen)  # 取最后一个输入的隐层状态作为语义向量
        self.EN = nn.Parameter(embwEN)
        self.ZH = nn.Parameter(embwZH)
        self.batchsize = batchSize

    def forward(self, x, y, ifEval=False, start_TF_rate=0.2):
        """前向计算"""
        if ifEval is not True:
            x = nn.functional.embedding(torch.tensor(x).long().to(device), self.EN)
            y = nn.functional.embedding(torch.tensor(y).long().to(device), self.ZH)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            y_in = torch.split(y, 1, dim=1)
            output1, c = self.encoder(x)
            out, c = self.decoder(y_in[0], c)
            for i in y_in[1:]:
                if start_TF_rate < random.uniform(0, 1):    # All teacher forcing
                    p, c = self.decoder(i, c)
                else:
                    p, c = self.decoder(nn.functional.embedding((out.split(1, dim=1)[-1]).argmax(2), self.ZH), c)
                out = torch.cat((out, p), dim=1)
            return out
        if ifEval:
            x = nn.functional.embedding(torch.tensor(x).long().to(device), self.EN)
            x = x.to(torch.float32)
            y = y.to(torch.float32).unsqueeze(0)
            out = torch.full([self.batchsize, 1], 1).to(device)
            EOS = torch.tensor(len(self.ZH) - 1).to(device)
            for i in range(31):
                y = torch.cat((y, self.EN[1].unsqueeze(0)), dim=0)
            y = y.reshape([self.batchsize, 1, 300])
            while not torch.equal(y[0].argmax(1), EOS):
                if out.size(1) == 1:
                    output1, c = self.encoder(x)
                y, c = self.decoder(y, c)
                out = torch.cat((out, y.argmax(2)), dim=1)
                y = nn.functional.embedding(y.argmax(2), self.ZH)
            return out

```

### 3.实验结果

实验没有实际效果，在每一步中都进行了维度的监视，此时没有应用tf机制，每一次的输入都是target的输入，所以精度是均匀上升的，**最终可以到达0.8左右的精度，但是在预测阶段直接崩掉**，预测无法停止，不能出现EOS标号。

### 4.分析

没有什么思路，猜想是teacher forcing过度导致模型的参数极端化，即每一次的参数都被每一次的target token向量都使w和h矩阵向全1矩阵优化，在一定的迭代次数后参数矩阵被优化成了全1矩阵导致每次的输出无限重复。

### 5.解决的想法

做teacher forcing概率化

## 二、teacher forcing设置概率

设置一个固定的teacher forcing概率，即有一定的概率采用target token作为输入，其余时间利用前一阶段的输出token来作为下一步的输入

### 1.实验描述

增加代码

### 2.代码

```Python
            for i in y[1:]:
                if start_TF_rate > random.uniform(0, 1):    # All teacher forcing
                    p, hidden, cell = self.decoder(i, hidden, cell)
                else:
                    decode_in = nn.functional.embedding(out.split(1, dim=1)															[-1].argmax(2), self.ZH).permute(1, 0, 2)
                    p, hidden, cell = self.decoder(decode_in, hidden, cell)
                out = torch.cat((out, p), dim=1)
```

### 3.实验结果

实验效果非常显然，但是都是负效果，预测阶段没有变化，同时出现了新的问题，即**不拟合**

![Tf-Acc](https://i.bmp.ovh/imgs/2022/08/16/4fb640e23c840c55.png)

![Tf-loss](https://i.bmp.ovh/imgs/2022/08/16/1118b3313afa50e7.png)

（teacher forcing rate=0.7，0.8,  0.9， smoth=0.8）

可以看出，在rate越大的时候loss收敛越快，同时Acc也越大，但是都处于无法收敛的状态，判定训练失败！

### 4.分析

在增加一定的teacher forcing概率之后，出现了意料之中的accuracy下降但是没有想到会出现三者都不能收敛的情况，很明显在没有输入target token的时候模型完全无法得到有效的训练

### 5.解决的想法

目前没有思路，网上查找资料

## 三、RNN->LSTM

### 1.实验描述

根据网上的教程，将双向RNN换成了LSTM来做实验，观察结果

### 2.代码

```Python
self.recurent = nn.LSTM(inputSize, batch_first=True,
                                num_layers=numLayers, hidden_size=hiddenSize)  # Using a 2 direction RNN to read the input sentences


 out, (hidden, cell) = self.recurent(x)
        return hidden, cell

    
self.recurrent = nn.LSTM(inputSize, hiddenSize, bias=True)
        self.linear = nn.Linear(in_features=hiddenSize, out_features=dictLen)
    
 
def forward(self, x, in_hidden, in_cell):
        """前向计算"""
        out = self.recurrent(x, state)
        state = out
        y = self.linear(out.unsqueeze(0).permute(1, 0, 2))
        return y, state
        out, (hidden, cell) = self.recurrent(x, (in_hidden, in_cell))
        y = self.linear(out.permute(1, 0, 2))
        return y, hidden, cell

    
  
 y = y.permute(1, 0, 2).split(1, dim=0)
            hidden, cell = self.encoder(x)
            out, hidden, cell = self.decoder(y[0], hidden, cell)
            for i in y[1:]:
                if start_TF_rate > random.uniform(0, 1):    # All teacher forcing
                    p, hidden = self.decoder(i, c)
                    p, hidden, cell = self.decoder(i, hidden, cell)
                else:
                    decode_in = nn.functional.embedding(out.split(1, dim=1)[-1].argmax(2), self.ZH).squeeze(1)
                    p, hidden = self.decoder(decode_in, c)
                    decode_in = nn.functional.embedding(out.split(1, dim=1)[-1].argmax(2), self.ZH).permute(1, 0, 2)
                    p, hidden, cell = self.decoder(decode_in, hidden, cell)
                out = torch.cat((out, p), dim=1)
            return out
```

### 3.实验结果

![TEST1](https://s3.bmp.ovh/imgs/2022/08/16/022030c16dc72ca3.png)

![TEST2](https://s3.bmp.ovh/imgs/2022/08/16/0b0128eb6263d55e.png)

**异常梯度图像**

![grad1](https://s3.bmp.ovh/imgs/2022/08/16/bc5bd913fa9fd145.png)

![grad2](https://s3.bmp.ovh/imgs/2022/08/16/ab712cc98f80a5fa.png)

![grad3](https://s3.bmp.ovh/imgs/2022/08/16/93ac9c55caf1b480.png)

### 4.分析

在具体参数分析时，梯度在极短的迭代时间内趋向于0，与学习率叠加后产生了非常小的值，理论上参数更新应该非常小幅度才对，但是参数又实际上是有效更新的，为什么会产生这种效果？同时在极小的step中隐藏层的计算矩阵被快速更新成了1，导致语义向量c每次的输出都很趋近于全1矩阵，之后的预测大量出现SOS，问题需要进一步确定。

### 5.想法

目前无

## 四、LSTM+Teacher forcing=1

### 1.实验描述

主要是为了测试在全部实验teacher forcing的时候，在evaluation过程中的表现。

分析它的参数矩阵，可以了解在优化过程中优化的趋势

### 2.代码

```Python
def forward(self, x, y, ifEval=False, start_TF_rate=1)
```

### 3.实验结果

![Acc-TF=1](https://s3.bmp.ovh/imgs/2022/08/17/399627ce13819f12.png)

从直接的图像中看不出来是不是过拟合或者局部最优，需要进一步确认

### 4.分析

没有明确的原因，只知道在全部teacher forcing的时候有效果

debug的时候有新思路，已知输出造成了这样：

```
tensor([[    1, 62683, 62683,  ..., 62683, 62683, 62683],
        [    1, 62683, 62683,  ..., 62683, 62683, 62683],
        [    1, 62683, 62683,  ..., 62683, 62683, 62683],
        ...,
        [    1, 62683, 62683,  ..., 62683, 62683, 62683],
        [    1, 62683, 62683,  ..., 62683, 62683, 62683],
        [    1, 62683, 62683,  ..., 62683, 62683, 62683]], device='cuda:0')
```

有可能是因为训练不足导致预测无法停止，即第一个预测词被无限重复了，需要查看输入的batch，比较后发现其实在输入中也并没有预测出来的这个词，所以很明显是错误预测

输入的batch查看出来没有发现什么明显的问题，但是输出的确是被趋同了，有可能是输出太偏差、、、、

查阅资料后了解到可能是因为sigmoid函数导致崩溃，但是实际的计算中却发现x并没有那么大。

cell发现比较大，平均绝对值大小在10~20之间，不知道有什么影响。

### 5.想法

仍然不能确定问题点，之后再按照RNN的情况调参测试

## 调参实验

### 1.实验描述

按照对seq2seq的理解，输入输出理论上没有问题，但是又出现了类似于RNN的情况，有可能是一样的参数问题

所以对模型进行调参实验，之后再统计实验结果。

### 2.代码

```Python
Optimizer:（Epoch=1， Teacher forcing=0.2）
    SGD
    	lr:    # 先做大learning rate的测试，尝试跳出局优
            0.2    --->ACC=0   loss不下降
            0.3    --->ACC=0	loss不下降
            0.4    --->ACC=0	loss不下降
            0.5    --->ACC=0	loss不下降
            1e-2    --->ACC=0	loss下降
            1e-3    --->ACC=0	loss下降
            1e-4    --->ACC=0	loss下降
            1e-5    --->ACC=0	loss不下降
   		Adam:
            0.2    --->ACC=0	loss下降
            0.3    --->ACC=0	loss下降
            0.4    --->ACC=0	loss下降
            0.5    --->ACC=0	loss下降
            1e-2    --->ACC=0	loss下降
            1e-3    --->ACC=0	loss下降
            1e-4    --->ACC=0	loss下降
            1e-5    --->ACC=0	loss下降
```

### 3.实验结果

实验结果表明，当前的处理流程和数据输入以及架构不能学到相应的数据分布。

当lr偏小（此处为0.2）的时候，模型精度不论学习率与优化器是什么都没有有效的预测，并且出现了不同层趋同的情况。

也有可能是模型非常依赖于target输入，在没有有效target输入的情况下是不能够有效收敛的。

两种情况需要分别进行测试。



### 4.分析

造成这种情况的原因猜想是数据在计算过程中出现了细胞死亡，有可能是在计算过程中导致sigmoid超过线性范围导致最终的结果被映射为了1，最后结果趋同的原因则与此有关，即中间计算值很大，导致sigmoid收缩为1，所以得到的就会是一样的结果。

### 5.想法

尝试减小数据大小或者对隐层状态和实际的细胞状态进行一个数值归一化，以达到最后激活函数之前的数值合适范围。

## 六、介绍一下曝光偏差

在实际的seq2seq搭建过程中，会遇到一个非常经典的问题，即曝光偏差