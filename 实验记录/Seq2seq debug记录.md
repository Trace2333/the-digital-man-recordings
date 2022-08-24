# Seq2seq debug记录

## 问题1——输出趋同且相同

### 问题1综述

语义矩阵趋同，不管输入为什么，得到的最终隐层状态都是近于一个全1矩阵，判定为Teacher forcing过度或者是陷入到局部最优。

### 排查方案

模型结构不变，加载参数。

对decoder逐步排查，比对每一个隐层的输出

方案未应用，通过思考得出在训练阶段的decoder操作有误

### 解决方案

```Python
        if ifEval is not True:
            x = nn.functional.embedding(torch.tensor(x).long().to(device), self.EN)
            y = nn.functional.embedding(torch.tensor(y).long().to(device), self.ZH)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            output1, c = self.encoder(x)
            p, _ = self.decoder(y, c)
            return p
```

​													当前训练阶段

将训练阶段的输入方式改成单步输入，即与预测阶段相同，在完成一次计算之后再进行损失计算

```Python
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

​													当前评估阶段