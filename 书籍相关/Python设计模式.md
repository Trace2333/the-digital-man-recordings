# Python设计模式

## 1.简介

### 对象的主要概念

### 封装

不能直接让外界对对象进行操作，对对象本身的操作应该是以信息发送的形式，让对象调用特定的成员函数来改变其自身的内部状态

### 多态

即同一类型的线性运算

### 继承

即一个对象可以继承它的父类，它的父类所存在的方法是可以被子类调用，同时子类也可以对其本身对象进行独立拓展

```python
class A(object):
    def __init__(self, a):
        print 'init A...'
        self.a = a

class B(A):
    def __init__(self, a):
        super(B, self).__init__(a)
        print 'init B...'

class C(A):
    def __init__(self, a):
        super(C, self).__init__(a)
        print 'init C...'

class D(B, C):
    def __init__(self, a):
        super(D, self).__init__(a)
        print 'init D...'
>>> d = D('d')
init A...
init C...
init B...
init D...

#python设计模式————多继承
class A():
    def __init__(self):
        print("这是A对象的init方法")
        self.name = "The_init_method"
    def func1(self):
        print("这是func1方法")
class C():
    def __init__(self):
        print("这是C对象的init方法")
class B(A):
    def __init__(self):
        print("B对象初始化")
        super(B, self).__init__()
    def Bfunc1(self):
        super(B, self).func1()
        print("已经过super方法重构原对象方法")
>>>testA = A()#会自动调用A中的init方法
>>>testB = B()
>>>testB.func1()#调用了第一个的func1，但是A对象的方法
>>>testB.Bfunc1()

```

#### 特别注意——super（）方法

super()方法的核心其实很简单——调用目前类中的父类中的方法，避免self的冲突，可以**用于对子类中对于父类同名的方法进行改写**

### 抽象

抽象就是类似于一个先验类——具有相应的属性和方法但是没有提前实现。使用抽象类的时候必须将对应的抽象类中的**全部方法实现**才能使用

```python
import abc


class A(metaclass=abc.ABCMeta):#metcalss=abc.ABCMeta是一个抽象的基类，如果要使用抽象类就要继承于abc.ABCMeta
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def func1(self):
        pass

    @abc.abstractmethod
    def func2(self):
        pass
class B(A):#继承于A类
    def func1(self):
        print("这是第一个函数")
    def func2(self):
        print("这是第二个函数")
    #def __init__(self):
      #  print("INIT")
>>>B = B()
>>>TypeError: Can't instantiate abstract class B with abstract methods __init__
import abc


class A(metaclass=abc.ABCMeta):#metcalss=abc.ABCMeta是一个抽象的基类，如果要使用抽象类就要继承于abc.ABCMeta
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def func1(self):
        pass

    @abc.abstractmethod
    def func2(self):
        pass
class B(A):#继承于A类
    def func1(self):
        print("这是第一个函数")
    def func2(self):
        print("这是第二个函数")
    def __init__(self):
        print("INIT")
>>>B = B()
>>>B.func1()
>>>B.func2()
>>>INIT
>>>这是第一个函数
>>>这是第二个函数
```

### 面向对象的设计原则

简而言之——对修改封闭、对拓展开放

**利用抽象基类来进行类的初始化定义**

如一个用户的类，抽象类用于通用定义，之后的拓展则在抽象类的基础上实现和拓展方法