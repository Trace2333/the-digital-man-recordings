# 深度学习pytorch代码流程

在每次的深度学习代码写作中都要遵循一定的流程，这样每次的结果才可以排除掉自身代码导致的小细节问题，此文档用来规范 **pytorch框架** 下的深度学习代码写作流程。只有有规范合乎流程的代码才能让彼此的交流更加高校并且可以有效规避错误。

## Category

### 导入包并设置随机数种子

### 以argparse类方式定义模型超参数

### 定义earlystop（根据需求使用）

### 定义dataset和dataloader

### 实例化模型并定义损失、优化器等

### 循环迭代训练

### 绘图（根据需求使用）

### 预测

代码示例均为RNN作关键词提取任务的示例

## 一、导入包并设置随机数种子

在每次的实验中，需要手动设置一个固定的随机数种子来让结果可以复现，同时一个确定的随机数种子也让试验更加可控。

```Python
import logging
import torch
import os
import random
import dill
import wandb
from tqdm import tqdm
from evalTools import acc_metrics
from torch.utils.data import DataLoader, RandomSampler
from TorchsRNN import SrnnNet, RNNdataset, collate_fun2

seed = 42
torch.manual_seed(seed)
random.seed(seed)
# 手动初始化随机数种子确保可以复现结果
```

## 二、以argparse方式定义超参数

为了方便在每次实验和试跑中进行参数的记录和更改，使用一个类来记录参数，在实际使用中可以将其中的参数值作为输入参数，运行时运行脚本即可。

```Python
class argparse()
	"""放置参数"""
    pass


args = argparse()    # 实例化对象
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.batch_size = 64
args.input_size = 300
args.hidden_size1, args.hidden_size2 = [300, 300]
args.epochs, args.lr = [2, 1e-3]
args.dataset_file = "C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\data_set.pkl"
args.embedding_file = 
"C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\embedding.pkl"
args.check_points = './check_points'
args.save_name = "RNN.pth"
```

## 三、定义earlystop类

```Python
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
```

**参考网上代码，在试跑无误的时候再多epoch实验的时候需要用！**

## 四、定义dataset和dataloader

在pytorch框架下的训练都是循环执行  输入--->输出--->反向传播---->参数更新 的流程，而它本身又是一个迭代过程，因此每次**使用的时候都尽量使用dataset和dataloader**来传输数据。

dataset的定义：

```Python
class dataset(Dataset):
    """继承于torch.utils.data下的Dataset"""
	def __init__(self):
		super(RNNdataset, self).__init__()
		pass
    
	def __getitem__(self, index):
        """标准方法，每次都要写"""
        return self.src[index], self.trg[index]
    
    def __len__(self):
        """标准方法，每次都要写"""
        return len(self.src) 
```

dataloader的定义：

```Python
# Dataloader需要先 from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=dataset,   # 数据集对象
                          batch_size=args.batch_size,
                          shuffle=True,    # 是否打乱
                          num_workers=0,
                          drop_last=True,    # 是否要丢弃不足一个batch的数据
                          collate_fn=collate_fun2    # 自定义collate_fn，用来处理不定长batch
                          )

eval_loader = DataLoader(dataset=evaluation_dataset,
                         batch_size=args.batch_size,
                         shuffle=True,    # 是否打乱
                         num_workers=0,
                         drop_last=True,    # 是否要丢弃不足一个batch的数据
                         collate_fn=collate_fun2)    # 自定义collate_fn，用来处理不定长batch
```

## 五、实例化模型并定义损失和优化器等

每次训练都需要用框架提供的API来进行损失函数和优化器的定义，注意每次的类型选择

```Python
# 模型定义应该单独放置一个文件，此处省略具体内容
class DRNN(nn.Module):
	def __init__(self):
        super(DRNN, self).__init__()
        pass
    
    def forward(self, inputs):
        return
    
    def init_state(self, batchsize):
        return
    
    
model = DRNN(args.input_size, args.hiden_size1, args.hiden_size2, embw).to(args.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

loss1 = 0
loss2 = 0
loss = 0
alpha = 0.3
```

## 六、循环迭代训练

pytorch中的训练都是在循环中进行epoch和step的数据输入输出的

```Python
for epoch in range(args.epochs):
    iteration = enumerate(train_loader, desc=f"Running at epoch {epoch}")
    model.train()
    train_epoch_loss = []    # 区别于前面定义的train_epochs_loss
    for step, batch in iteration;
    	output1, output2 = model(
            (inputs[0], torch.randn([1, batch_size, hiden_size1])))
        	
            loss1 = criterion(output1.permute(0, 2, 1), inputs[1][0])
        	loss2 = criterion(output2.permute(0, 2, 1), inputs[1][1])
            loss = loss2 * (1-alpha) + loss1 * alpha
            
            optimizer.zero_grad()
        	loss.backward()
        	optimizer.step()
            
            sentence_preds = output1.argmax(axis=2)
        	sequence_preds = output2.argmax(axis=2)
            sen_acc = acc_metrics(sentence_preds, inputs[1][0])
        	seq_acc = acc_metrics(sequence_preds, inputs[1][1])
            
            wandb.log({"Train Sentence Precision": sen_acc})
        	wandb.log({"Train Sequence Precision": seq_acc})
            wandb.log({"train loss1": loss1})
        	wandb.log({"train loss2": loss2})
        	wandb.log({"train Totalloss": loss})
```

## 七、绘图

由于大多数时候都使用带有自动绘图功能的API如tensorboard、wandb进行绘图，因此这里的绘图功能暂时不细说。

## 八、预测

```Python
for epoch in range(1):
    iteration = tqdm(evaluation_loader, desc=f"EVALUATION on epoch {epoch + 1}")
    model.eval()
    for step, evaluation_input in enumerate(evaluation_iteration):
        with torch.no_grad():
            output1, output2 = model((evaluation_input[
                0], torch.randn([1, batch_size, hiden_size1])))
            
            sentence_preds = output1.argmax(axis=2)
            sequence_preds = output2.argmax(axis=2)
            sen_acc = acc_metrics(sentence_preds, evaluation_input[1][0])
            seq_acc = acc_metrics(sequence_preds, evaluation_input[1][1])
            
            wandb.log({"Sentence Precision": sen_acc})
            wandb.log({"Sequence Precision": seq_acc})
```

