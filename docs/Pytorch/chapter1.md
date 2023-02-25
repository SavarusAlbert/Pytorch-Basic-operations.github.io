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
- 索引操作(索引出来的结果与原数据共享内存，如果不想同时修改，可以考虑使用`copy()`等方法)
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
- 通过`.item()`命令获取单个tensor的值
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

## 1.3 自动求导
### 1.3.1 requires_grad参数
- `torch.Tensor` 是这个包的核心类。
- 如果设置它的属性 `.requires_grad` 为 `True`(默认为`False`)，那么它将会追踪对于该张量的所有操作。
```python
x = torch.randn(3,3,requires_grad=True)
```
- 当完成计算后可以通过调用 `.backward()`，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到 `.grad` 属性。

### 1.3.2 .detach()和.detach_()方法
#### .detach()
- `.detach()`方法会返回一个新的tensor，`requires_grad`为`False`。
- 得到的这个tensor永远不需要计算其梯度，不具有grad，即使之后重新将它的`requires_grad`置为`True`，也不会具有梯度grad。
- 这个tensor是从当前计算图中分离下来的，但是仍指向原变量的存放位置，和原始的tensor共用一个内存，即一个修改另一个也会跟着改变。
```python
a = torch.tensor([1, 2, 3.], requires_grad=True)
out = a.sigmoid()
b = out.detach()
out.sum().backward()
```
```python
print("{} {}".format("a =", a))
print("{} {}".format("out =", out))
print("{} {}".format("b =", b))
```
```markup
a = tensor([1., 2., 3.], requires_grad=True)
out = tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward0>)
b = tensor([0.7311, 0.8808, 0.9526])
```
- 对这个新变量求导会报错
```python
b.sum().backward()
```
```markup
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
- 改变`.detach()`的变量后，原变量也会改变
```python
b.zero_()
print("{} {}".format("out =", out))
```
```markup
out = tensor([0., 0., 0.], grad_fn=<SigmoidBackward0>)
```
- 此时再进行反向传播会报错
```python
out.sum().backward()
```

#### .detach_()
将一个tensor从创建它的图中分离，并把它设置成叶子tensor。
- 相当于将tensor的`grad_fn`设置为`None`，将计算图中的其他部分进行变换，使其变为叶子结点，再将其`requires_grad`设置为`False`。
- `.detach_()`后无法再复原，而`.detach()`方法还可以对原计算图进行操作。

### 1.3.3 with torch.no_grad():
- 为了防止跟踪历史记录(和使用内存），可以将代码块包装在 `with torch.no_grad(): `中：
```python
x = torch.randn(10, 5, requires_grad = True)
y = torch.randn(10, 5, requires_grad = True)
z = torch.randn(10, 5, requires_grad = True)
with torch.no_grad():
    w = x + y + z
    print(w.requires_grad)
    print(w.grad_fn)
print(w.requires_grad)
```
```markup
False
None
False
```

## 1.4 梯度
### 1.4.1 多次.backward()
- 对一个变量，多次计算backward，会报错。这是因为第一次反向传播之后，这个计算图的内存就会被释放掉，这样的话再次进行反向传播就不行了，解决方法就是添加`retain_graph=True`这个参数。
```python
a = torch.ones(2, 2, requires_grad=True)
b = torch.ones(2, 2, requires_grad=True)
c = a * b
# 添加retain_graph=True这个参数后，才可以进行多次的求导
d = c.sum()
d.backward(retain_graph=True)
print(a.grad)
print(b.grad)
c1 = c * a
c1.sum().backward(retain_graph=True)
print(a.grad)
print(b.grad)
```
```markup
tensor([[1., 1.],
        [1., 1.]])
tensor([[1., 1.],
        [1., 1.]])
tensor([[3., 3.],
        [3., 3.]])
tensor([[2., 2.],
        [2., 2.]])
```
> 每次反向传播后，梯度都会叠加。所以一般在反向传播之前需把梯度清零。
```python
a.grad.data.zero_()
b.grad.data.zero_()
```
- 在模型训练时，`retain_graph=True`方法保留了计算图，却可能会使缓存快速积累，可以在第一次求导后加上`.detach()`，停止后续的求导。

### 1.4.2 非标量求导
- 在下面这个例子中，y不再是标量，`torch.autograd` 不能直接计算完整的雅可比矩阵，.`backward()`会报错。
```python
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
# 会报错
y.backward()
```
- 可以将向量v(与x形状相同，相当于权重)作为参数传给 backward：
```python
v = torch.tensor([1, 1, 1])
y.backward(v, retain_graph=True)
print(x.grad)
x.grad.data.zero_()
```
- 可以通过`tensor.data`直接对值进行修改，不会影响反向传播：
```python
x.data *= 100                               # 注意修改后保持原向量的shape
```

## 1.5 并行计算
在PyTorch框架下，CUDA的使用变得非常简单，我们只需要显式的将数据和模型通过`.cuda()`方法转移到GPU上就可加速我们的训练：
```python
model = Net()
model.cuda() # 模型显示转移到CUDA上
for image,label in dataloader:
    # 图像和标签显示转移到CUDA上
    image = image.cuda() 
    label = label.cuda()
```