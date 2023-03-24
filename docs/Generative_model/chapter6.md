# Real NVP论文泛读
## 6.1 模型基本结构
### 6.1.1 变量替换公式
- 给定可观察的数据变量 $x{\in}X$，隐变量 $z{\in}Z$ 上的先验概率分布 $p_Z$，和一个双射 $f:X{\rightarrow}Z$($g=f^{-1}$)，变量替换公式定义如下：
$$p_X(x)=p_Z(f(x))\left|{\rm{det}}(\frac{{\partial}f(x)}{{\partial}x^T})\right|$$
$${\rm{log}}(p_X(x))={\rm{log}}\left(p_Z(f(x))\right)+{\rm{log}}\left(\left|{\rm{det}}(\frac{{\partial}f(x)}{{\partial}x^T})\right|\right)$$
其中 $\frac{{\partial}f(x)}{{\partial}x^T}$ 是 $f$ 在 $x$ 上的雅克比矩阵。
- 使用逆变换采样规则可以从得到的分布中生成精确的样本。

### 6.1.2 耦合层
- 我们通过复合一系列简单的仿射耦合层，来拟合复杂的变换。
- 仿射耦合层定义：给定一个 $D$ 维($d<D$)输入 $x$，输出 $y$，满足下述公式：
$$\begin{align*}
y_{1:d}&=x_{1:d}\\
y_{d+1:D}&=x_{d+1:D}{\odot}{\rm{exp}}\left(s(x_{1:d})\right)+t(x_{1:d})
\end{align*}$$
其中 $s$ 和 $t$ 分别表示 $\mathbb{R}^d\rightarrow\mathbb{R}^{D-d}$ 上的缩放和平移，$\odot$ 是逐元素哈达玛积。

### 6.1.3 模型性质
- 变换的雅克比矩阵为：
$$\frac{{\partial}y}{{\partial}x^T}=\left[\begin{array}{l}
I_d & 0\\
\frac{{\partial}y_{d+1:D}}{{\partial}x_{1:d}} & {\rm{diag}}({\rm{exp}}[s(x_{1:d})])
\end{array}\right]$$
其中 ${\rm{diag}}({\rm{exp}}[s(x_{1:d})])$ 是一个对角矩阵，对角上的元素是向量 ${\rm{exp}}[s(x_{1:d})]$ 的元素。
- 因为雅克比矩阵是对角的，所以我们可以很快的计算其行列式为 $\sum_js(x_{1:d})_j$。
- 由于在计算雅克比行列式时不需要计算 $s$ 和 $t$ 的行列式，因此这两个函数可以进行复杂的设计。我们用深度卷积神经网络来定义 $s$ 和 $t$，它们的隐藏层可以比输入输出层有更多的特征。
- 耦合层的逆变换可以由下式计算：
$$\begin{align*}
x_{1:d}&=y_{1:d}\\
x_{d+1:D}&=(y_{d+1:D}-t(y_{1:d})){\odot}{\rm{exp}}\left(-s(y_{1:d})\right)
\end{align*}$$
因此模型的采样是高效的。同时，计算耦合层的逆也不需要计算 $s$ 和 $t$ 的逆，所以这两个函数可以做复杂的设计。

![](./img/6.1Real_NVP.png ':size=60%')

## 6.2 其他技巧
### 6.2.4 掩码卷积
- 划分可以使用二进制掩码 $b$ 实现，取如下函数形式：
$$y=b{\odot}x+(1-b){\odot}\left(x{\odot}{\rm{exp}}(s(b{\odot}x))+t(b{\odot}x)\right)$$
- 我们采用如下两种掩码方式：
    - 棋盘掩码：在空间坐标之和为奇数时取值为1，否则为0。
    - 通道方向掩码：在通道维度的前半部分为1，后半部分为0。

![](./img/6.2Real_NVP.png ':size=80%')

### 6.2.5 组合耦合层
- 耦合层的设计使得某些部分始终保持不变，所以我们设计交替变换的组合耦合层。
- 在这种情况下，根据以下几式，我们的雅克比行列式和逆变换同样是易于求解的：
$$\frac{{\partial}(f_b{\circ}f_a)}{{\partial}x_a^T}(x_a)=\frac{{\partial}f_a}{{\partial}x_a^T}(x_a)\cdot\frac{{\partial}f_b}{{\partial}x_b^T}\left(x_b=f_a(x_a)\right)$$
$${\rm{det}}(A{\cdot}B)={\rm{det}}(A){\rm{det}}(B)$$
$$(f_b{\circ}f_a)^{-1}=f_a^{-1}{\circ}f_b^{-1}$$

### 6.2.6 多尺度架构
- 我们首先应用3个具有棋盘掩码的交替组合耦合层，然后执行挤压操作来改变数据维度，最后应用3个以上的具有通道方向掩码的交替组合耦合层。
- 通过将变换进行分解计算，来降低计算的难度：
$$\begin{align*}
h^{(0)}&=x\\
(z^{(i+1)},h^{(i+1)})&=f^{(i+1)}(h^{(i)})\\
z^{(L)}&=f^{(L)}(h^{(L-1)})\\
z&=(z^{(1)},\cdots,z^{(L)})
\end{align*}$$

![](./img/6.3Real_NVP.png ':size=40%')

### 6.2.7 批规范化
- 进一步改善训练，我们对 $s$ 和 $t$ 使用残差连接、批归一化和权重归一化。
- 批归一化操作下的雅克比矩阵很容易计算，给定统计量 $\tilde{\mu}$ 和 $\tilde{\sigma}^2$，归一化操作如下：
$$x{\rightarrow}\frac{x-\tilde{\mu}}{\sqrt{\tilde{\sigma}^2+\epsilon}}$$
对应的雅克比矩阵乘上系数：
$$\left(\prod\limits_i(\tilde{\sigma}_i^2+\epsilon)\right)^{-\frac{1}{2}}$$
