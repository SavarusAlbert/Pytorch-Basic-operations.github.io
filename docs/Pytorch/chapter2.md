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

## 2.3 模型构建和训练
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

### 2.3.2 含模型参数的层
- `Parameter` 类是 `Tensor` 的子类，如果一个 `Tensor` 是 `Parameter` ，那么它会⾃动被添加到模型的参数列表里。
- 可以直接将参数定义成 `Parameter` ，还可以使⽤ `ParameterList` 和 `ParameterDict` 分别定义参数的列表和字典。
```python
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
```
```python
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])
```
- 一个模型的可学习参数可以通过`net.parameters()`返回(net是网络的名称)。

### 2.3.3 二维卷积层和池化层
- 二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。
- 自定义二维卷积
```python
# 卷积运算（二维互相关）
def corr2d(X, K): 
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```
- 池化操作，用相同的滑动窗口对数据进行最大池化和平均池化操作。
- 自定义池化层
```python
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

### 2.3.4 torch.nn内置卷积模块
```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```
- 参数
    - in_channels (int)：输入的通道数
    - out_channels (int)：输出通道数
    - kernel_size (int or tuple)：卷积核的大小
    - stride (int or tuple, optional)：卷积步长，默认为1
    - padding (int, tuple or str, optional)：(四周)填充，默认为0
    - padding_mode (str, optional)：填充模式，默认为'zeros'
        - 'zeros'：用'0'进行填充
        - 'reflect'：以矩阵边缘为对称轴，将矩阵中的元素对称的填充到最外围
        - 'replicate'：将矩阵的边缘复制并填充到矩阵的外围
        - 'circular'：原始矩阵循环填充到四周
    - dilation (int or tuple, optional)：卷积核映射后元素间隔，默认为1
    - groups (int, optional)：将原输入分为几组，重用out_channels/groups次，默认为1
    - bias (bool, optional) – 添加一个可学习的偏置项，默认为`True`

### 2.3.5 模型初始化
在深度学习模型的训练中，权重的初始值极为重要。设置初始值，可以使模型的收敛速度提高，还可以提高模型准确率。
- `torch.nn.init`初始化方法
```python
torch.nn.init.uniform_(tensor, a=0.0, b=1.0)                    # 均匀分布
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)                # 正态分布
torch.nn.init.constant_(tensor, val)                            # val填充
torch.nn.init.ones_(tensor)                                     # 全1填充
torch.nn.init.zeros_(tensor)                                    # 全0填充
torch.nn.init.eye_(tensor)                                      # 用单位阵填充
torch.nn.init.dirac_(tensor, groups=1)                          # 用dirac函数填充
torch.nn.init.xavier_uniform_(tensor, gain=1.0)                 # 初值更小的均匀分布
torch.nn.init.xavier_normal_(tensor, gain=1.0)                  # 初值更小的正态分布
torch.nn.init.orthogonal_(tensor, gain=1)                       # 半正交阵填充
torch.nn.init.sparse_(tensor, sparsity, std=0.01)               # 稀疏矩阵初始化
# 凯明均匀分布和正态分布初始化
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
# [a,b]区间内正态分布
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=- 2.0, b=2.0)
# 计算经过给定非线性函数后的方差变化
torch.nn.init.calculate_gain(nonlinearity, param=None)
```
- 通过`isinstance()`函数来判断模块的类型，对不同的模块进行不同的初始化。
- 初始化函数封装
```python
def initialize_weights(self):
	for m in self.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zeros_()
```

### 2.3.6 损失函数
- 常用损失函数
```python
# 二分类交叉熵损失函数
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# 交叉熵损失函数
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# L1损失函数
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
# MSE损失函数
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
# 平滑L1损失函数
torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
# 泊松分布的负对数似然损失函数
torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
# KL散度
torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
# MarginRankingLoss，计算两个向量之间的相似度
torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
# 多标签边界损失函数
torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
# 二分类损失函数，计算二分类的 logistic 损失
torch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')torch.nn.(size_average=None, reduce=None, reduction='mean')
# 多分类的折页损失
torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
# 三元组损失
torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
# 余弦相似度
torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
# CTC损失函数，用于解决时序类数据的分类
torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
```

### 2.3.7 训练和评估
- 模型训练模板
```python
def train(epoch):
    model.train()                                               # 训练状态，模型的参数支持反向传播的修改
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()                 # 将数据放到GPU上用于后续计算
        optimizer.zero_grad()                                   # 优化器的梯度清零
        output = model(data)
        loss = criterion(output, label)
        loss.backward()                                         # 将loss反向传播回网络
        optimizer.step()                                        # 使用优化器更新模型参数
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
	print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
```
- 模型验证/测试模板
```python
def val(epoch):
    model.eval()                                                # 验证/测试状态，不会修改模型参数
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)                     # 返回维度1维的最大值的序号
            loss = criterion(output, label)                     # 通过前向计算得到预测值，计算损失函数
            val_loss += loss.item()*data.size(0)                # .item()用于取出单个tensor中的值
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, val_loss))
```

### 2.3.8 Optimizer
- `Optimizer`的基类
```python
class Optimizer(object):
    def __init__(self, params, defaults):        
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
```
- 其中，`defaults`存储的是优化器的超参数，`state`是参数的缓存，`param_groups`管理的参数组。
- 常用的Optimizer
```python
# 优化器是类，要进行实例化
optimizer = torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False,)
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, maximize: bool = False,)
# 具体参数省略
optimizer = torch.optim.Adagrad()
optimizer = torch.optim.RMSprop()
optimizer = torch.optim.Adadelta()
optimizer = torch.optim.ASGD()
optimizer = torch.optim.AdamW()
optimizer = torch.optim.Adamax()
optimizer = torch.optim.LBFGS()
optimizer = torch.optim.Rprop()
optimizer = torch.optim.SparseAdam()
```
- 优化器方法
```python
optimizer.zero_grad()                                           # 清空梯度
optimizer.step()                                                # 执行一步梯度更新，参数更新
optimizer.add_param_group()                                     # 添加参数组
optimizer.load_state_dict()                                     # 加载状态参数字典，可以用来进行模型的断点续训练，继续上次的参数进行训练
optimizer.state_dict()                                          # 获取优化器当前状态信息字典
```

## 2.4 本章总结
本章主要学习如何运用Pytorch进行模型的完整训练验证流程，主要包括：
- 数据的读取和预处理
- 模型的构建
- 模型初始化
- 训练和优化