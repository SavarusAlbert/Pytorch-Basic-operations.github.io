# Chapter03 Pytorch模型定义
## 3.1 Pytorch模型定义常见方式
- 基于`nn.Module`，我们可以通过`Sequential`，`ModuleList`和`ModuleDict`三种方式定义`PyTorch`模型。
- `Sequential`适用于快速验证结果，相对简单。
- 而当我们需要之前层的信息的时候，比如 `ResNets` 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 `ModuleList`/`ModuleDict` 比较方便。
### 3.1.1 Sequential方式
- 直接排列
```python
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
```
- 使用OrderedDict
```python
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
```
### 3.1.2 ModuleList
- `nn.ModuleList` 模块接收一个子模块（或层，需属于`nn.Module`类）的列表作为输入，然后也可以类似`List`那样进行`append`和`extend`操作。
```python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
```
- 要特别注意的是，`nn.ModuleList` 并没有定义一个网络，它只是将不同的模块储存在一起。`ModuleList`中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过`forward`函数指定各个层的先后顺序后才算完成了模型的定义。具体实现时用`for`循环即可完成。
```python
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...
    ...
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```
### 3.2.3 ModuleDict
- `ModuleDict`和`ModuleList`的作用类似，只是`ModuleDict`能够更方便地为神经网络的层添加名称。
```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)
```

## 3.2 Pytorch修改模型
- 我们已经有一个现成的模型，但该模型中的部分结构不符合我们的要求，为了使用模型，我们需要对模型结构进行必要的修改。
### 3.2.1 修改模型层
- `torchvision`预定义好的模型`ResNet50`：
```python
import torchvision.models as models
net = models.resnet50()
```
```markup
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
..............
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```
- 假设我们要用这个`resnet`模型去做一个10分类的问题，就应该修改模型的`fc`层，将其输出节点数替换为10。另外，我们觉得一层全连接层可能太少了，想再加一层。可以做如下修改：
```python
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
net.fc = classifier
```
### 3.2.2 添加额外输入
- 有时候在模型训练中，除了已有模型的输入之外，还需要输入额外的信息。
- 基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在`forward`中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改。
- 例如`torchvision`的`resnet50`模型，倒数第二层增加一个额外的输入变量`add_variable`来辅助预测。
```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x = self.net(x)
        # 通过torch.cat实现了tensor的拼接
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)
        x = self.fc_add(x)
        x = self.output(x)
        return x
```
- 模型实例化
```python
import torchvision.models as models
net = models.resnet50()
model = Model(net).cuda()
# 训练中在输入数据的时候要给两个inputs
# outputs = model(inputs, add_var)
```
### 3.2.3 添加额外输出
- 有时候在模型训练中，我们需要输出模型某一中间层的结果，以施加额外的监督，获得更好的中间层结果。
- 基本的思路是修改模型定义中forward函数的return变量。
- 输出1000维的倒数第二层和10维的最后一层结果：
```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000
```
- 模型实例化
```python
import torchvision.models as models
net = models.resnet50()
model = Model(net).cuda()
# 训练中在输入数据后会有两个outputs
# out10, out1000 = model(inputs, add_var)
```

## 3.3 Pytorch模型保存与读取
### 3.3.1 模型存储
- PyTorch存储模型主要采用pkl，pt，pth三种格式。
- 模型包括模型结构和权重，存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重。
```python
from torchvision import models
model = models.resnet152(pretrained=True)
save_dir = './resnet152.pth'                                    # pkl,pt,pth使用上没有差别
# 保存整个模型
torch.save(model, save_dir)
# 保存模型权重
torch.save(model.state_dict, save_dir)
```
### 3.3.2 单卡与多卡训练下模型的保存与加载方法
- 单卡多卡模型训练(如果要使用多卡训练的话，需要对模型使用`torch.nn.DataParallel`)
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                        # 如果是多卡改成类似0,1,2
model = model.cuda()                                            # 单卡
model = torch.nn.DataParallel(model).cuda()                     # 多卡
```
- 单卡保存+单/多卡加载
```python
import os
import torch
from torchvision import models
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                        # 这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model.cuda()
save_dir = 'resnet152.pt'                                       # 保存路径

# 保存+读取整个模型
torch.save(model, save_dir)
# 单卡读取
loaded_model = torch.load(save_dir)
loaded_model.cuda()
# 多卡读取
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
loaded_model = torch.load(save_dir)
loaded_model = nn.DataParallel(loaded_model).cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)
# 单卡读取
loaded_model = models.resnet152()                               # 注意这里需要对模型结构有定义
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model.cuda()
# 多卡读取
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'                      # 这里替换成希望使用的GPU编号
loaded_model = models.resnet152()                               # 注意这里需要对模型结构有定义
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()
```
- 多卡保存+单卡加载
```python
import os
import torch
from torchvision import models
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'                      # 这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取整个模型
torch.save(model, save_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                        # 这里替换成希望使用的GPU编号
loaded_model = torch.load(save_dir).module

# 保存模型权重
torch.save(model.module.state_dict(), save_dir)
```
- 多卡保存+多卡加载
- 多卡模式下建议使用权重的方式存储和读取模型
```python
import os
import torch
from torchvision import models
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'                    # 这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()
# 保存+读取模型权重，强烈建议！！
torch.save(model.state_dict(), save_dir)
loaded_model = models.resnet152()                               # 注意这里需要对模型结构有定义
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()

# 如果只有保存的整个模型，也可以采用提取权重的方式构建新的模型
loaded_whole_model = torch.load(save_dir)
loaded_model = models.resnet152()                               # 注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_whole_model.state_dict
loaded_model = nn.DataParallel(loaded_model).cuda()
```

## 3.4 本章总结
本章主要学习Pytorch模型的基础操作，主要包括：
- 3种方式定义模型
- 修改定义好的模型
- 模型在单卡多卡上的保存和加载