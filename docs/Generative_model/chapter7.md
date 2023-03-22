# Glow论文泛读
## 7.1 背景：基于流的生成模型
- 令 $x$ 是一个具有未知真实分布的 $x{\sim}p^*(x)$ 高维随机向量。
- 我们收集一个数据集 $\mathcal{D}$，选择一个模型 $p_\theta(x)$，其中 $\theta$ 是参数。
- 在离散数据 $x$ 的情况下，对数似然目标函数等价于最小化以下函数：
$$\mathcal{L}(\mathcal{D})=\frac{1}{N}\sum\limits_{i=1}^N-{\rm{log}}p_\theta(x^{(i)})$$
- 在连续数据 $x$ 的情况下，我们最小化下面的函数：
$$\mathcal{L}(\mathcal{D})\simeq\frac{1}{N}\sum\limits_{i=1}^N-{\rm{log}}p_\theta(\tilde{x}^{(i)})+c$$
其中 $\tilde{x}^{(i)}=x^{(i)}+u$，$u\sim\mathcal{U}(0,a)$；另外 $c=-M\cdot{\rm{log}}a$，$a$ 由数据的离散程度决定，$M$ 为 $x$ 的维数。
- 在多数基于流的生成模型中，生成过程由下式定义：
$$z{\sim}p_\theta(z);{\quad}x=g_\theta(z)$$
其中 $z$ 是隐变量，$p_\theta(z)$ 是一个显性密度，如一个多元高斯分布 $p_\theta(z)=\mathcal{N}(z;\mathbf{0},\mathbf{I})$。
- 函数 $g_\theta$ 是可逆的，对于给定的数据点 $x$，隐变量可以由下式推断：
$$z=f_\theta(x)=g_\theta^{-1}(x)$$
- 我们关注变换 $f$ 的一系列分解 $f=f_1{\circ}f_2{\circ}\cdots{\circ}f_K$，使 $x$ 和 $z$ 有如下关系：
$$x\stackrel{f_1}{\longleftrightarrow}h_1\stackrel{f_2}{\longleftrightarrow}h_2\cdots\stackrel{f_K}{\longleftrightarrow}z$$
这样的可逆变换序列称为 normalizing flow。
- 在上述情况下，模型的概率密度函数可以由下式给出：
$$\begin{align*}
{\rm{log}}p_\theta(x)&={\rm{log}}p_\theta(z)+{\rm{log}}\left|{\rm{det}}(dz/dx)\right|\\
&={\rm{log}}p_\theta(z)+\sum\limits_{i=1}^K{\rm{log}}\left|{\rm{det}}(dh_i/dh_{i-1})\right|
\end{align*}$$
其中 $h_0:=x$，$h_K:=z$
- 选取合适的变换，我们可以令变换的雅克比矩阵为三角矩阵，这种情况下我们有：
$${\rm{log}}\left|{\rm{det}}(dh_i/dh_{i-1})\right|={\rm{sum}}\left({\rm{log}}\left|{\rm{diag}}(dh_i/dh_{i-1})\right|\right)$$



## 7.2 Glow