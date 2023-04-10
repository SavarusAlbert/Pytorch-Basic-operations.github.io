# MLP系列论文泛读
## 4.1 2014-Paragraph-Vec
- 句子和文档的分布式表示</br>
(Distributed Representations of Sentences and Documents)

#### 学习词的向量表示
- 给定一个训练集中的序列 $w_1,w_2,\cdots,w_T$，词向量模型最大化平均对数似然：
$$\frac{1}{T}\sum\limits_{t=k}^{T-k}{\rm{log}}p(w_t|w_{t-k},\cdots,w_{t+k})$$
概率 $p$ 通过一个softmax分类器预测：
$$p(w_t|w_{t-k},\cdots,w_{t+k})=\frac{e^{y_{w_t}}}{\sum_ie^{y_i}}$$
每个 $y_i$ 是单词 $i$ 的未归一化的对数概率：
$$y=b+Uh(w_{t-k},\cdots,w_{t+k};W)$$
其中 $U$ 和 $b$ 是分类器的参数，$h$ 是concat操作或者是一些平均操作。
- 实际训练时采用分层softmax和负采样，降低算法复杂度。本文中的分层softmax使用二元霍夫曼树编码。

#### 段落向量：分布式记忆模型
- 与一般的词向量模型架构的不同在于，引入了一个段落向量来预测父节点，具体见下图：

![](./img/4.1.1paragraph-vec.png ':size=80%')
- 因此模型中的函数 $h$ 比之前多了段落向量 $D$ 的参数：
$$y=b+Uh(w_{t-k},\cdots,w_{t+k};W,D)$$

#### 无词序的段落向量：分布式词袋
- 另一种引入段落向量的方法是：忽略上下文词，而从段落中随机采样单词来参与预测，模型架构如下图所示：

![](./img/4.1.2paragraph-vec.png ':size=80%')
- 操作上，在随机梯度下降的每次迭代中，我们采样一个文本窗口，然后从文本窗口中采样一个随机单词，给定段落向量，进行模型的预测分类任务。
- 这种方法比前面的方法所需要存储的数据量要少，在实验中将两种方法结合效果更好。

## 4.2 2015-DAN
- 在文本分类中，深度无序合成与句法方法分庭抗礼</br>
(Deep Unordered Composition Rivals Syntactic Methods for Text Classification)

### 4.2.1 无序函数与句法函数结合
#### 神经词袋模型(NBOW)
- 考虑输入序列为tokens $X$，从 $k$ 个标签中预测，首先通过函数 $g$ 对词嵌入 $v_w,w{\in}X$ 进行合成。输出一个向量 $z$，作为逻辑回归函数的输入。
- 我们定义 $g$ 如下：
$$z=g(w{\in}X)=\frac{1}{|X|}\sum\limits_{w{\in}X}v_w$$
通过softmax层得到估计概率：
$$\^{y}={\rm{softmax}}(W_s{\cdot}z+b)$$
其中 ${\rm{sofftmax}}(q)=\frac{{\rm{exp}}q}{\sum_{j=1}^k{\rm{exp}}q_j}$，$W_s$ 是一个 $k{\times}d$ 维的参数矩阵，$b$ 是偏置项。
- 我们通过最小化交叉熵损失来训练：
$$l(\^{y})=\sum\limits_{p=1}^ky_p{\rm{log}}(\^{y}_p)$$

#### 考虑与句法结合
- RecNN系列模型通过输入的顺序和结构来学习词语之间的影响，牺牲了计算效率来学习句子的语义信息。这类模型依赖于解析树，因此会学习到相对一致的语法，泛化到领域外数据集上效果会降低

### 4.2.2 DAN深度平均网络(Deep Averaging Networks)
- 我们在计算输出向量 $z$ 时，引入深度网络来训练。假定我们用 $n$ 层网络来学习 $z$，每层学习到的特征分别为 $z_{1,\cdots.n}$，我们通过下式计算：
$$z_i=g(z_{i-1})=f(W_i{\cdot}z_{i-1}+b_i)$$
然后将最后一层的输出放入softmax分类器进行预测。
- 这个深度网络称为DAN，因为每层的运算中使用矩阵乘法来统一迭代参数，算法复杂度并没有很大变化，模型性能很高。

![](./img/4.2.1DAN.png ':size=100%')
#### 词Dropout提高鲁棒性
- Dropout通过以一定概率 $p$ 将隐藏单元或输入单元随机设置为零来进行正则化。某些任务上能够提高模型性能。