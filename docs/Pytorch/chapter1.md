# Chapter01 Pytorch基础操作总结
## 1.1 Pytorch学习资源
- [Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list)：目前已获12K Star，包含了NLP,CV,常见库，论文实现以及Pytorch的其他项目。
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)：官方发布的文档，十分丰富。
- [Pytorch-handbook](https://github.com/zergtant/pytorch-handbook)：GitHub上已经收获14.8K，pytorch手中书。
- [PyTorch官方社区](https://discuss.pytorch.org/)：PyTorch拥有一个活跃的社区，在这里你可以和开发pytorch的人们进行交流。
- [PyTorch官方tutorials](https://pytorch.org/tutorials/)：官方编写的tutorials，可以结合colab边动手边学习
- [动手学深度学习](https://zh.d2l.ai/)：动手学深度学习是由李沐老师主讲的一门深度学习入门课，拥有成熟的书籍资源和课程资源，在B站，Youtube均有回放。
- [Awesome-PyTorch-Chinese](https://github.com/INTERMT/Awesome-PyTorch-Chinese)：常见的中文优质PyTorch资源

## 1.2 Tensor相关
- 导入`torch`包
```python
import torch
```
- 随机初始化 $m{\times}n$ 矩阵，或对已有矩阵随机初始化</br>
(randn生成服从 $N(0, 1)$ 的正态分布，rand生成[0, 1)均匀分布，normal生成任意正态分布)
```python
x = torch.randn(m, n)
x1 = torch.rand(m, n)
x2 = torch.randn_like(x)
x3 = torch.normal(mean = 0, std = 1, size = (2, 2))
```
- 构建或转换为全0矩阵(全1矩阵)
```python
y = torch.zeros(m, n)
y2 = torch.ones(m, n)
z = torch.zeros_like(x)
z2 = torch.ones_like(x)
```
- 直接构建Tensor
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
```
- 获取Tensor维度信息
```python
print(x.size())
print(x.shape)
```
- 加法操作
```python
m1 = x + y
m2 = torch.add(x, y)
m3 = y.add(x)
```
- 索引操作(索引出来的结果与原数据共享内存，如果不想同时修改，可以考虑使用copy()等方法)
```python
y1 = x[:, 1]
y2 = x[2, :]
y2 += 1                                             # 源tensor也被更改了
```
- 维度变换
```python
x = torch.randa(3, 4)
# torch.view()返回的结果与源tensor共享内存
y = x.view(12)
y2 = x.view(4, 3)
y3 = x.view(-1, 6)                                  # -1表示维数由其他维决定
# torch.reshape()返回的结果不一定与源tensor共享内存，不推荐使用
y4 = x.reshape(4, 3)
# 使用clone()命令后再通过view()改变维度，可以不与源tensor共享内存
y5 = x.clone()
y6 = y5.view(4, 3)
y6 += 1
# 注：clone()的数据会被记录在计算图中，梯度回传到副本时也会传到源tensor
```
- 通过.item()命令获取单个tensor的值
```python
m = x[1:1].item()
```
- 转置、索引、切片
```python
z = x.t()                                           # 转置操作，共享内存
# 初级索引 同迭代器切片
# 高级索引
z2 = torch.index_select(input, dim, index)          # 从input的dim维的index位置挑选数据
z3 = torch.masked_select(input, mask, out=None)     # 从input中掩码mask条件的数据
```
- 数学运算、线性代数
```python
x = torch.arange(1, 13).view(3, 4)
y = torch.randn(3, 4)
# 对应元素相乘
c1 = x * y
c2 = torch.mul(a, b)
# 对应元素相除
c1 = a / b
c2 = torch.div(a, b)
# 矩阵相乘
c1 = torch.mm(x, y.t())
c2 = torch.matmul(x, y.t())
c3 = x @ y.t()
# 多维矩阵相乘，前面维度相同时可以做运算，对应位置进行后两个维度的矩阵乘法
x = torch.rand(4, 3, 28, 64)
y = torch.rand(4, 3, 64, 32)
print(torch.matmul(x, y).shape)
# 多维矩阵相乘，前面维度适用广播机制
x = torch.rand(4, 3, 28, 64)
y = torch.rand(4, 1, 64, 32)
print(torch.matmul(x, y).shape)
# 幂与开方运算
x = torch.full([2, 2], 3)
c1 = x ** 2
c2 = x.pow(2)
c3 = c2.sqrt()
# 指对运算
a = torch.exp(x)
b = torch.log(a)
# 近似，取下、取上、取整数、取小数、四舍五入
a, b, c, d, e = x.floor(), x.ceil(), x.trunc(), x.frac(), x.round()
# 裁剪：常用在梯度离散或梯度爆炸时
grad = torch.rand(2, 3)
grad_max, grad_min, grad_mid = grad.max(), grad.min(), grad.median()    # 最大值、最小值、平均值 
c1 = grad.clamp(10)                                                     # 最小是10，小于10的都变成10
c2 = grad.clamp(3, 10)                                                  # 最小是3，小于3的都变成3；最大是10，大于10的都变成10
```
- 广播机制(broadcasting)</br>
当对两个形状不同的Tensor按元素运算时，可能会触发广播机制：先适当复制元素使这两个Tensor形状相同后再按元素运算。
```python
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)
```