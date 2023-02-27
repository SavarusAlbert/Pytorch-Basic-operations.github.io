# Chapter04 Pytorch进阶技巧
## 4.1 动态调整学习率
- `PyTorch`在`torch.optim.lr_scheduler`封装好的动态调整学习率的方法：
```python
torch.optim.lr_scheduler.LambdaLR
torch.optim.lr_scheduler.MultiplicativeLR
torch.optim.lr_scheduler.StepLR
torch.optim.lr_scheduler.MultiStepLR
torch.optim.lr_scheduler.ExponentialLR
torch.optim.lr_scheduler.CosineAnnealingLR
torch.optim.lr_scheduler.ReduceLROnPlateau
torch.optim.lr_scheduler.CyclicLR
torch.optim.lr_scheduler.OneCycleLR
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
```
- 官方代码模板
```python
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
# 需要在优化器参数更新之后再动态调整学习率
# scheduler的优化是在每一轮后面进行的
scheduler1.step() 
...
schedulern.step()
```
- 自定义scheduler学习率
```python
# 自定义学习率每30轮下降为原来的1/10
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9)
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)
```
## 4.2 模型微调
- 用`pretrained`参数来决定是否使用预训练好的权重
```python
import torchvision.models as models
# pretrained = True，意味着我们将使用在一些数据集上预训练得到的权重(pretrained默认为False)
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
...
```
- 也可以将自己的权重下载下来放到同文件夹下，然后再读取参数
```python
self.model = models.resnet50(pretrained=False)
self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
```
- 通过冻结部分层，来微调其他层的参数
```python
# 官方模板
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# resnet18将1000类改为4类
import torchvision.models as models
# 冻结参数的梯度
feature_extract = True
model = models.resnet18(pretrained=True)
set_parameter_requires_grad(model, feature_extract)
# 修改模型
num_ftrs = model.fc.in_features                             # 模型最后一层fc层in_features参数
model.fc = nn.Linear(in_features=num_ftrs, out_features=4, bias=True)
```
## 4.3 模型微调库-timm
- timm提供的预训练模型已经达到了770个
- 通过`timm.list_models()`方法查看timm提供的预训练模型
```python
import timm
avail_pretrained_models = timm.list_models(pretrained=True)
len(avail_pretrained_models)
all_densnet_models = timm.list_models("*densenet*")         # 传入想查询的模型名称（模糊查询）
```
- 查看模型具体参数
```python
model = timm.create_model('resnet34',num_classes=10,pretrained=True)
print(model.default_cfg)
```
- 可以访问这个链接查看提供的预训练模型的准确度等信息
```markup
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
 'num_classes': 1000,
 'input_size': (3, 224, 224),
 'pool_size': (7, 7),
 'crop_pct': 0.875,
 'interpolation': 'bilinear',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'first_conv': 'conv1',
 'classifier': 'fc',
 'architecture': 'resnet34'}
```
- 使用timm进行微调
```python
import timm
import torch
model = timm.create_model('resnet34',pretrained=True)
# 通过num_classes修改输出类别，in_chans改变输入通道数
# model = timm.create_model('resnet34',num_classes=10,pretrained=True,in_chans=1)
x = torch.randn(1,3,224,224)
output = model(x)
# 查看第一层模型参数
list(dict(model.named_children())['conv1'].parameters())
# timm库所创建的模型是torch.model的子类，使用torch库内置模型参数保存和加载的方法
torch.save(model.state_dict(),'./checkpoint/timm_model.pth')
model.load_state_dict(torch.load('./checkpoint/timm_model.pth'))
```
## 4.4 半精度训练
- PyTorch默认的浮点数存储方式用的是`torch.float32`，小数点后位数更多固然能保证数据的精确性，但绝大多数场景其实并不需要这么精确，只保留一半的信息也不会影响结果，也就是使用`torch.float16`格式，从而减少了显存占用。
- 使用`autocast`配置半精度训练
```python
from torch.cuda.amp import autocast
# 在模型定义中，用autocast装饰模型中的forward函数
@autocast()   
def forward(self, x):
    ...
    return x
# 在训练过程中，将数据输入模型及其之后的部分放入"with autocast():"
for x in train_loader:
	x = x.cuda()
	with autocast():
            output = model(x)
        ...
```
## 4.5 本章总结
本章学习了Pytorch的一些进阶技巧，包括：
- 动态调整学习率
- 模型微调
- 半精度训练提高显存占用