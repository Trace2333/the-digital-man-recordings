# RNN-Bert实验记录

## 一、Bert直接初始化语料句子并将所有未登录句子作为UNK处理

### 1.实验描述

在利用bert进行测试的时候，最直接的想法是将bert作为一个embedding工具，将其抽象化后得到的词嵌入作为输入，以期得到更好的结果

### 2.实验代码

采用的方法是提前提取词向量组成embedding文件，在没有对词向量做连接、池化等操作之前保留768维度，直接修改网络维度以适应词嵌入维度、

重写了data_process

```Python
# 核心代码
def generate_embedding(bert_filename, dataset):
    """
    建立词嵌入词典并进行保存
    Args:
        bert_filename: bert配置和参数文件的文件路径
        dataset: 初步建立的dataset，经过create_dataset函数创建，是一个列表形式
    Return:
        无
    """
    model = BertModel.from_pretrained(bert_filename).to(device)
    model.eval()
    embedding = {}
    tokenizer = BertTokenizer.from_pretrained(bert_filename)
    iteration = tqdm(dataset)
    out_dataset = {}
    for i, j in zip(dataset, range(len(dataset) + 1)[1:]):
        out_dataset[i] = j
    out_dataset["[O]"] = 0
    for token, i in zip(iteration, range(len(dataset) + 1)[1:]):
        encoded_token = tokenizer.batch_encode_plus(
            [token],
            return_tensors='pt',
        )['input_ids'].to(device)
        embw = model(encoded_token)['last_hidden_state']
        embw = torch.mean(embw, dim=1).cpu().detach().numpy()
        embedding[i] = embw
    embedding["[O]"] = torch.zeros([1, 768]).cpu().detach().numpy()
    if os.path.exists(".\\data"):
        with open(".\\data\\embedding.pkl", "wb") as f1:
            pickle.dump(embedding, f1)
            print("embedding.pkl Created!")
        with open(".\\data\\dataset.pkl", "wb") as f2:
            pickle.dump(out_dataset, f2)
            print("dataset.pkl, Created!")

```

### 3.实验结果

在进行词嵌入矩阵装载时，显存多次溢出导致崩溃。

### 4.实验分析

在之后的实验中，经过计算**一共需要293GB显存**来装载240057*768维度的词嵌入张量，但是实际上，在之前的300维度下的实验中，模型却表现正常，因此可能是数据输出太大的原因

该方法认为是可以进行训练的

### 5.后续想法

先做一次增加bert作为embedding层的实验，再尝试使用降低数据精度的方法来测试。

## 二、将bert作为一个embedding层进行试验

### 1.实验描述

利用bert作为embedding层来进行计算

**Note：如何处理未登录词？**

将每个词单独放入模型进行计算得到输出，由于部分未登录词是以一个**词条**形式存在，在经过wordpiece机制优化后，会产生多个分词，所以在之后的输入中，我们将它处理成多个输出的平均值

## 三、Bert初始化，但是将未登录词以随机初始化形式得到

