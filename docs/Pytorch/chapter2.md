# Chapter02 Pytorch的主要组成模块
## 2.1 基本配置
- 深度学习常见的包
```python
import os 
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
```
- 常用的配置
```python
batch_size = 16                                     # 批次的大小
lr = 1e-4                                           # 优化器的学习率
max_epochs = 100                                    # 训练次数
```
- GPU设置
```python
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'                                  # 指明调用的GPU为0,1号
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")     # 指明调用的GPU为1号
```

## 2.2 数据读入
PyTorch数据读入是通过Dataset+DataLoader的方式完成的，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据。
- 自定义`Dataset`类
```python
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_file)
```
- 使用`DataLoader`按批次读入数据
```python
from torch.utils.data import DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
```
- `num_workers`：有多少个进程用于读取数据，Windows下该参数设置为0，Linux下常见的为4或者8，根据自己的电脑配置来设置
- `drop_last`：对于样本最后一部分没有达到批次数的样本，使其不再参与训练

## 2.3 构建模型
- `Module` 类是 `torch.nn` 模块里提供的一个模型构造类。
### 2.3.1 MLP类构造
- 自定义MLP 类，定义一个具有两个隐藏层的多层感知机：
```python
import torch
from torch import nn
class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类的.__init__()函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)                 # python2必须写成 super(本类名, self).__init__() ;python3 直接写为 super().__init__()
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
  # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)   
```
- `super()`用来调用父类(基类)的方法，`__init__()`是类的构造方法，`super().__init__()`就是调用父类的`init`方法， 同样可以使用`super()`去调用父类的其他方法。
- `__init__()` 是python中的构造函数，在创建对象的时"自动调用"。
> 以上的 MLP 类中⽆须定义反向传播函数。系统将通过⾃动求梯度⽽自动⽣成反向传播所需的 backward 函数。
- 实例化模型，`net(X)` 会调用 `MLP` 继承⾃自 `Module` 类的 `__call__` 函数，这个函数将调⽤ `MLP` 类定义的 `forward` 函数来完成前向计算。
```python
X = torch.rand(2,784)                                   # 设置一个随机的输入张量
net = MLP()                                             # 实例化模型
print(net)                                              # 打印模型
net(X)                                                  # 前向计算
```
```markup
MLP(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
tensor([[ 0.1020,  0.1706,  0.0030, -0.0219,  0.3967,  0.1342, -0.1981, -0.0740,
         -0.0516,  0.0757],
        [-0.0637,  0.0903,  0.0661, -0.1102,  0.1871,  0.1439, -0.2156, -0.1051,
          0.0348,  0.0091]], grad_fn=<AddmmBackward0>)
```

### 2.3.2 模型常见层
