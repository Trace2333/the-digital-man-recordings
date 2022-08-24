# Datasets使用

## 支持的数据集的格式

### Map-Style datasets

Map-style表示的数据类型是类似于一个字典形式的数据，即每一个数据都有对应的一个标签或者类型，即A:a的形式

在使用该类数据的时候一般需要重写 __getitem__()和__len__()两个方法，同时创建的dataset要继承于dataset

### Iterable-Style

iterable-sytle数据类型表示为流式数据，即在每一次的数据获取中都是从某一个固定的地方获得数据，其中所获得的数据可以和map-sytle类似

具体不再讨论

## 怎么使用

在实际的使用中只需要先判断需要处理的数据集的格式再选取是map or iterable即可

### getitem的写法

getitem要返回两个值，**即元素本身和元素的索引**

len返回一个值，**即数据集的长度**

可以重写继承的init方法将数据包装到dataset类上来使用

```python
def __getitem__(self, index):
        return self.src[index], self.trg[index]
```

```python
def __len__(self):
        return len(self.src) 
```

最后都是要得到这两个方法的规定返回值才行，也就是说**只需要**

**将两个方法重构得到对应的返回值即可，实现方式任意**