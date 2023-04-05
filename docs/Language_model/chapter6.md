# RNN系列论文泛读
## 6.1 2015-Tree-LSTM
- 树结构的长短期记忆网络的改进语义表示</br>
(Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks)
### 6.1.1 长短期记忆网络
- 通常来说，RNN转换函数如下形式：
$$h_t={\rm{tanh}}(Wx_t+Uh_{t-1}+b)$$
- 在训练过程中，梯度向量的分量在长序列上会出现指数增长或衰减，这种梯度爆炸或消失的问题使得RNN模型难以学习序列中的长距离相关性。

#### 一般形式
- LSTM架构通过引入能够长期保持状态的记忆单元来解决学习长期相关性的问题。
- 我们定义LSTM单元，每一个时间步 $t$，输入门 $i_t$，遗忘门 $f_t$，输出门 $o_t$，记忆单元 $c_t$，隐藏状态 $h_t$，其中 $i_t,f_t,o_t\in[0,1]$，记 $d$ 是LSTM的记忆维度，那么LSTM方程定义如下：
$$\begin{align*}
i_t&=\sigma(W^{(i)}x_t+U^{(i)}h_{t-1}+b^{(i)})\\
f_t&=\sigma(W^{(f)}x_t+U^{(f)}h_{t-1}+b^{(f)})\\
o_t&=\sigma(W^{(o)}x_t+U^{(o)}h_{t-1}+b^{(o)})\\
u_t&={\rm{tanh}}(W^{(u)}x_t+U^{(u)}h_{t-1}+b^{(u)})\\
c_t&=i_t{\odot}u_t+f_t{\odot}c_{t-1}\\
h_t&=o_t\odot{\rm{tanh}}(c_t)
\end{align*}$$
其中 $x_t$ 是当前步的输入，$\sigma$ 是sigmoid函数，$\odot$ 是逐元素乘积。

#### 变体
- 双向LSTM：并行两个LSTM结构，一个输入正向的序列，一个输入逆向的序列，来学习过去和未来的信息。隐藏层通过将两部分的隐藏状态简单的concat起来。
- 多层LSTM：叠加多层LSTM，$l$ 层的隐藏状态输出作为 $l+1$ 层的输入，来提取高维特征。
- 两种变体也可以结合起来。

### 6.1.2 树结构LSTM
#### 子树LSTM
- 给定一个树，令 $C(j)$ 表示节点 $j$ 的孩子节点。子树LSTM结构通过下式定义：
$$\begin{align*}
\tilde{h}_j&=\sum\limits_{k{\in}C(j)}h_k\\
i_j&=\sigma(W^{(i)}x_j+U^{(i)}\tilde{h}_j+b^{(i)})\\
f_{jk}&=\sigma(W^{(f)}x_j+U^{(f)}h_{k}+b^{(f)})\\
o_j&=\sigma(W^{(o)}x_j+U^{(o)}\tilde{h}_j+b^{(o)})\\
u_j&={\rm{tanh}}(W^{(u)}x_j+U^{(u)}\tilde{h}_j+b^{(u)})\\
c_j&=i_j{\odot}u_j+\sum\limits_{k{\in}C(j)}f_{jk}{\odot}c_k\\
h_j&=o_j\odot{\rm{tanh}}(c_j)
\end{align*}$$
- 由于子树LSTM的成分依赖于子节点的隐状态之和，因此非常适合分支因子高的或孩子无序的树，例如依赖树。我们将应用于依赖树的子树LSTM称为依赖树LSTM。

#### N元树LSTM
- N元树LSTM应用在最多N个分支且孩子节点有序的树结构。因此所有节点可以被1到N排序。通过下式定义：
$$\begin{align*}
i_j&=\sigma(W^{(i)}x_j+\sum\limits_{l=1}^NU^{(i)}_lh_{jl}+b^{(i)})\\
f_{jk}&=\sigma(W^{(f)}x_j+\sum\limits_{l=1}^NU^{(f)}_{kl}h_{jl}+b^{(f)})\\
o_j&=\sigma(W^{(o)}x_j+\sum\limits_{l=1}^NU^{(o)}_lh_{jl}+b^{(o)})\\
u_j&={\rm{tanh}}(W^{(u)}x_j+\sum\limits_{l=1}^NU^{(u)}_lh_{jl}+b^{(u)})\\
c_j&=i_j{\odot}u_j+\sum\limits_{l=1}^Nf_{jl}{\odot}c_{jl}\\
h_j&=o_j\odot{\rm{tanh}}(c_j)
\end{align*}$$
- 注意到子树LSTM和N元树LSTM在树只是一个单链的情况下reduce为普通的LSTM。
- 为每个子节点引入单独的参数矩阵，可以使得N元树LSTM能够比子树LSTM学习到更细致的子状态条件。
- 遗忘门 $f_{jk}$ 引入非对角参数矩阵 $U_{kl}^{(f)},k{\neq}l$，这种参数化允许更灵活地从子代到父代的信息传播。但当N非常大时可能有很多参数会归0。
- 因为左右节点是明确区分的，所以我们可以将二元树LSTM看作二元成分树来操作。我们将这个二元树LSTM称为成分树LSTM。

### 6.1.3 模型
#### 树LSTM分类器
- 我们希望从一个类别离散集 $\mathcal{Y}$ 组成的树中预测标签 $\^{y}$。
- 在每一个节点 $j$，以 $j$ 为根节点的子树 $\{x\}_j$ 作为输入，我们用一个softmax分类器来预测节点标签 $\^{y}_j$。分类器以隐藏状态 $h_j$ 作为输入：
$$\begin{align*}
\^{p}_\theta(y|\{x\}_j)&={\rm{softmax}}(W^{(s)}h_j+b^{(s)})\\
\^{y}_j&={\rm{arg}}\mathop{\rm{max}}\limits_{y}\^{p}_\theta(y|\{x\}_j)
\end{align*}$$
- 损失函数是真实类别 $y^{(k)}$ 的负对数似然：
$$J(\theta)=-\frac{1}{m}\sum\limits_{k=1}^m{\rm{log}}\^{p}_\theta(y^{(k)}|\{x\}^{(k)})+\frac{\lambda}{2}\|\theta\|^2_2$$
其中 $m$ 是训练集中标签节点的个数，$k$ 表示第 $k$ 个标签节点，$\lambda$ 是L2正则的超参数。

#### 句子对的语义相关性
- 给定句子对，我们希望预测一个 $[1,K],K>1$ 之间的实值相似分数，序列 $\{1,2,\cdots,K\}$ 是相似性度量，较高的分数表示更大的相似度。
- 我们首先在每个句子的分析树上使用树LSTM模型来得到句子对表示 $h_L$ 和 $h_R$。给定句子对表示，我们使用神经网络来预测相似性分数 $\^{y}$：
$$\begin{align*}
h_{\times}&=h_L{\odot}h_R\\
h_{+}&=|h_L-h_R|\\
h_s&=\sigma(W^{(\times)}h_{\times}+W^{(+)}h_++b^{(h)})\\
\^{p}_\theta&={\rm{softmax}}(W^{(p)}h_s+b^{(p)})\\
\^{y}&=r^T\^{p}_\theta
\end{align*}$$
其中 $r^T=[1,2,\cdots,K]$，绝对值函数是逐元素绝对值。根据经验我们设计了 $h_{\times}$ 和 $h_+$，我们发现将这两个组合起来效果更好。
- 我们希望在预测分布 $\^{p}_\theta$ 下的评分接近黄金评分 $y\in[1,K]$：
$$\^{y}=r^T\^{p}_\theta{\approx}y$$
因此我们定义一个稀疏目标分布 $p$ ，是使得 $y=r^Tp$：
$$p_i=\left\{\begin{array}{ll}
y-{\lfloor}y{\rfloor}, & i={\lfloor}y{\rfloor}+1\\
{\lfloor}y{\rfloor}-y+1, & i={\lfloor}y{\rfloor}\\
0 & {\rm{otherwise}}
\end{array}\right.$$
其中 $i\in[1,K]$。
- 损失函数定义为分布 $p$ 和 $\^{p}_\theta$ 的正则KL散度：
$$J(\theta)=\frac{1}{m}\sum\limits_{k=1}^m{\rm{KL}}(p^{(k)}\|\^{p}_\theta^{(k)})+\frac{\lambda}{2}\|\theta\|^2_2$$
其中 $m$ 是训练的句子对总数，$k$ 表示第 $k$ 对句子对。


## 6.2 2015-S-LSWM
- 递归结构上的长短期记忆</br>
(Long Short-Term Memory Over Recursive Structures)
#### 模型简介
- 本文将LSTM结构扩展为一个层级结构的网络，每个记忆单元可以反映多个子节点的状态。
- 下图是一个二元情况的例子：

![](./img/6.2.1S-LSTM.png ':size=40%')

其中 $\circ$ 和 $-$ 分别代表信息的通过和阻塞。

#### 记忆单元
- 前面例子中的每一个节点都是由一个S-LSTM记忆块构成，其具体结构见下图：

![](./img/6.2.2S-LSTM.png ':size=60%')

- 每个存储块包含一个输入门和一个输出门。遗忘门的数量取决于子节点数量。在本文中，我们假设每个节点上有两个子节点，也就是说，我们有两个遗忘门。
- 图中，两个孩子节点的隐藏向量作为块的输入，分别表示为 $h_{t-1}^L$(左孩子)和 $h_{t-1}^R$(右孩子)。输入门 $i_t$ 处理隐藏向量和cell向量($c_{t-1}^L$ 和 $c_{t-1}^R$)，左右遗忘门 $f_t^L$ 和 $f_t^R$ 同样以这四个向量为输入。左右遗忘门是相对独立的，有独立的参数。输出门$o_t$ 考虑来自子代的隐藏向量和当前的cell向量。
- 反过来，当前块的隐藏向量 $h_t$ 和cell向量 $c_t$ 被传递给父代，并根据当前块是父代的左子节点还是右子节点来使用。这样，记忆块通过合并子代的门控向量，可以直接或间接的反映多个后代细胞。因此，结构上的长距离相互作用可以被捕获。
- 具体公式关系如下：
$$\begin{align*}
i_t&=\sigma(W_{hi}^Lh_{t-1}^L+W_{hi}^Rh_{t-1}^R+W_{ci}^Lc_{t-1}^L+W_{ci}^Rc_{t-1}^R+b_i)\\
f_t^L&=\sigma(W_{hf_l}^Lh_{t-1}^L+W_{hf_l}^Rh_{t-1}^R+W_{cf_l}^Lc_{t-1}^L+W_{cf_l}^Rc_{t-1}^R+b_{f_l})\\
f_t^R&=\sigma(W_{hf_r}^Lh_{t-1}^L+W_{hf_r}^Rh_{t-1}^R+W_{cf_r}^Lc_{t-1}^L+W_{cf_r}^Rc_{t-1}^R+b_{f_r})\\
x_t&=W_{hx}^Lh_{t-1}^L+W_{hx}^Rh_{t-1}^R+b_x\\
c_t&=f_t^L{\otimes}c_{t-1}^L+f_t^R{\otimes}c_{t-1}^R+i_t{\otimes}{\rm{tanh}}(x_t)\\
o_t&=\sigma(W_{ho}^Lh_{t-1}^L+W_{ho}^Rh_{t-1}^R+W_{co}c_t+b_o)\\
h_t&=o_t\otimes{\rm{tanh}}(c_t)
\end{align*}$$
其中 $\sigma$ 是sigmoid函数，$f^L$ 和 $f^R$ 是左右遗忘门，$b$ 是偏置，$W$ 是权重矩阵，$\otimes$ 是哈达玛积，矩阵下标表示了它们用的位置。

#### 反向传播
- 反向传播过程比较复杂，需要区分左右后代。我们给出具体式子，对于一个记忆块，假定传给隐藏向量的误差为 $\epsilon_t^h$，输出门、左右遗忘门、输入门的导数分别为 $\delta_t^o,\delta_t^{f_l},\delta_t^{f_r},\delta_t^i$，那么反向传播式子如下：
$$\begin{align*}
\epsilon_t^h&=\frac{{\partial}O}{{\partial}h_t}\\
\delta_t^o&=\epsilon_t^h{\otimes}{\rm{tanh}}(c_t){\otimes}\sigma^{\prime}(o_t)\\
\delta_t^{f_l}&=\epsilon_t^c{\otimes}c_{t-1}^L{\otimes}\sigma^{\prime}(f_t^L)\\
\delta_t^{f_r}&=\epsilon_t^c{\otimes}c_{t-1}^R{\otimes}\sigma^{\prime}(f_t^R)\\
\delta_t^i&=\epsilon_t^c{\otimes}{\rm{tanh}}(x_t){\otimes}\sigma^{\prime}(i_t)\\
\end{align*}$$
其中 $\sigma^{\prime}(x)$ 是sigmoid函数逐元素对 $x$ 求导，$\epsilon_t^c$ 是指在cell向量上求导。因此左右子节点的 $\epsilon_t^c$ 分别为：
$$\begin{align*}
\epsilon_t^c=&\epsilon_t^h{\otimes}o_t{\otimes}g^{\prime}(c_t)+\epsilon_{t+1}^c{\otimes}f_{t+1}^L+(W_{ci}^L)^T\delta_{t+1}^i+\\
&(W_{cf_l}^L)^T\delta_{t+1}^{f_l}+(W_{cf_r}^L)^T\delta_{t+1}^{f_r}+(W_{co})^T\delta_t^o\\
\epsilon_t^c=&\epsilon_t^h{\otimes}o_t{\otimes}g^{\prime}(c_t)+\epsilon_{t+1}^c{\otimes}f_{t+1}^R+(W_{ci}^R)^T\delta_{t+1}^i+\\
&(W_{cf_l}^R)^T\delta_{t+1}^{f_l}+(W_{cf_r}^R)^T\delta_{t+1}^{f_r}+(W_{co})^T\delta_t^o
\end{align*}$$
其中 $g^{\prime}(x)$ 是 ${\rm{tanh}}$ 函数的逐元素求导。

#### 树上的目标函数
- 如果考虑与问题相关的输出，目标函数可以被定义的很复杂。在本文中，只简单的使用最小化正则交叉熵损失以及所有节点的损失和：
$$E(\theta)=\sum\limits_i\sum\limits_jt_j^i{\rm{log}}{y^{sen_i}}_j+\lambda\|\theta\|^2_2$$
其中 $y^{sen_i}\in\mathbb{R}^{c{\times}1}$ 是预测分布，$t^i\in\mathbb{R}^{c{\times}1}$ 是目标分布，$c$ 是类别数，$j{\in}c$ 是多元目标分布的第 $j$ 个元素。


## 6.3 2015-TextRCNN
- 用于文本分类的循环卷积神经网络</br>
(Recurrent Convolutional Neural Networks for Text Classification)
- 我们提出一个深度神经网络来捕获文本的语义。下图展示了模型的网络结构：

![](./img/6.3.1TextRCNN.png ':size=100%')

- 网络输入一个文档 $D$，其中 $w_1,w_2,\cdots,w_n$ 是词序列。网络输出一个类元素。我们用 $p(k|D,\theta)$ 来表示文档属于类别 $k$ 的概率($\theta$ 是网络中的参数)。

### 6.3.1 词表示学习
- 我们通过词和词的语境来表示一个词，语境能够帮助我们获得更精确的词意。我们使用双向循环结构。
- 我们定义 $c_l(w_i)$ 是左侧语境，$c_r(w_i)$ 是右侧语境，两个向量都是稠密的，且 $|c|$ 是实值元素。具体计算如下：
$$\begin{align*}
c_l(w_i)&=f(W^{(l)}c_l(w_{i-1})+W^{(sl)}e(w_{i-1}))\\
c_r(w_i)&=f(W^{(r)}c_r(w_{i+1})+W^{(sr)}e(w_{i+1}))
\end{align*}$$
其中 $e(w_{i-1})$ 是单词 $w_{i-1}$ 的词嵌入，$c_l(w_{i-1})$ 是前一个词 $w_{i-1}$ 的左侧语境。右侧同样。$W$ 是参数矩阵，$f$ 是非线性激活函数。最右侧的单词 $w_n$ 的下一个词的参数与 $w_n$ 共享 $c_r(w_n)$。
- 我们根据下式定义词 $w_i$ 的表示 $x_i$ 为三个表示的concat：
$$x_i=[c_l(w_i);e(w_i);c_r(w_i)]$$
- 接下来我们输入到下面的网络得到潜在语义向量 $y_i^{(2)}$：
$$y_i^{(2)}={\rm{tanh}}(W^{(2)}x_i+b^{(2)})$$

### 6.3.2 文本表示学习
- 我们模型中的卷积神经网络是用来表达文本的。从卷积神经网络的角度来看，我们前面提到的循环结构就是卷积层。
- 在计算所有词的表示时，我们采用最大池化层：
$$y^{(3)}=\mathop{\rm{max}}\limits_{i=1}^ny_i^{(2)}$$
其中 ${\rm{max}}$ 函数是逐项最大化。
- 池化层将不同长度的文本转化为固定长度的向量。通过池化层，我们可以捕获贯穿整个文本的信息。还有其他类型的池化层，如平均池化。这里我们不使用平均池化，因为只有少数单词及其组合对于捕获文档的含义是有用的。最大池化层试图找到文档中最重要的潜在语义因素。
- 模型最后的输出层类似于传统的神经网络，定义为：
$$y^{(4)}=W^{(4)}y^{(3)}+b^{(4)}$$
- 最后输入softmax分类器得到一个概率：
$$p_i=\frac{{\rm{exp}}(y_i^{(4)})}{\sum_{k=1}^n{\rm{exp}}(y_k^{(4)})}$$

### 6.3.3 模型训练
#### 训练网络参数
- 我们将所有可训练的参数统一表示为 $\theta$，网络的目标函数使用最大化对数似然：
$$\theta\rightarrow\sum\limits_{D\in\mathbb{D}}{\rm{log}}(class_D|D,\theta)$$
其中 $\mathbb{D}$ 是参与训练的文件的集合，$class_D$ 是正确分类的文件。
- 我们使用SGD来优化目标函数，每一步随机选择一个 $(D,class_D)$ 进行一个步骤的梯度：
$$\theta\leftarrow\theta+\alpha\frac{\partial{\rm{log}}p(class_D|D,\theta)}{\partial\theta}$$
其中 $\alpha$ 是学习率。
- 所有的参数由均匀分布随机初始化。

#### 预训练词嵌入
- 我们使用Skip-gram模型来预训练词向量。


## 6.4 2015-MTLSTM
- 用于句子和文档建模的多时间尺度长短期记忆神经网络</br>
(Multi-Timescale Long Short-Term Memory Neural Network for Modelling Sentences and Documents)
- LSTM可以捕获序列中的长期短期的依赖关系。但是长期依赖关系需要沿着序列逐步传递。一些重要的信息在长文本的传递过程中会丢失。
- 此外，当我们使用时间反向传播(BPTT)算法时，误差信号通过多个时间步反向传播。对于长文本，训练效率也可能较低。例如，如果一个有价值的特征出现在一个长文档的开头，我们需要通过整个文档来对误差进行反向传播。

### 6.4.1 MT-LSTM神经网络
- 我们将LSTM单元分成若干组。不同组捕捉不同时间尺度的依赖关系。如下图所示：

![](./img/6.4.1MT-LSTM.png ':size=60%')

其中虚节点表示当前时刻失活的单元，实节点表示当前时刻被激活的单元。虚线表示保持不变的单元，实线表示下一时间步将更新的单元。
- 具体来说，LSTM单元分为 $g$ 个群组 $\{G_1,\cdots,G_g\}$，每个 $G_k$ 在不同的时间步 $T_k$ 激活。因此，门和权重矩阵也是分段的。只有一个分支的MT-LSTM就是标准的LSTM。
- 每个时间步 $t$，满足 $t\ {\rm{mod}}\ T_k=0$ 的 $G_k$ 生效。时间段集合的选择 $T_k\in\{T_1,\cdots,T_g\}$ 是任意的，我们这里定义 $T_k=2^{k-1}$。
- 注意到 $G_1$ 每个步都会激活，类似于标准的LSTM单元。
- 在时间步 $t$，记忆cell向量和隐状态向量有两种情况：
    - $G_k$ 是激活的，那么LSTM单元按照下式计算：
    $$\begin{align*}
    \mathbf{i}_t^k&=\sigma(\mathbf{W}_i^k\mathbf{x}_t+\sum\limits_{j=1}^g\mathbf{U}_i^{j{\rightarrow}k}\mathbf{h}_{t-1}^j+\sum\limits_{j=1}^g\mathbf{V}_i^{j{\rightarrow}k}\mathbf{c}_{t-1}^j)\\
    \mathbf{f}_t^k&=\sigma(\mathbf{W}_f^k\mathbf{x}_t+\sum\limits_{j=1}^g\mathbf{U}_f^{j{\rightarrow}k}\mathbf{h}_{t-1}^j+\sum\limits_{j=1}^g\mathbf{V}_f^{j{\rightarrow}k}\mathbf{c}_{t-1}^j)\\
    \mathbf{o}_t^k&=\sigma(\mathbf{W}_o^k\mathbf{x}_t+\sum\limits_{j=1}^g\mathbf{U}_o^{j{\rightarrow}k}\mathbf{h}_{t-1}^j+\sum\limits_{j=1}^g\mathbf{V}_o^{j{\rightarrow}k}\mathbf{c}_t^j)\\
    \tilde{\mathbf{c}}_t^k&={\rm{tanh}}(\mathbf{W}_c^k\mathbf{x}_t+\sum\limits_{j=1}^g\mathbf{U}_c^{j{\rightarrow}k}\mathbf{h}_{t-1}^j)\\
    \mathbf{c}_t^k&=\mathbf{f}_t^k{\odot}\mathbf{c}_{t-1}^k+\mathbf{i}_t^k{\odot}\tilde{\mathbf{c}}_t^k\\
    \mathbf{h}_t^k&=\mathbf{o}_t^k{\odot}{\rm{tanh}}(\mathbf{c}_t^k)
    \end{align*}$$
    其中 $\mathbf{i}_t^k,\mathbf{f}_t^k,\mathbf{o}_t^k$ 是对应时间步 $t$ 和群 $G_k$ 的输入门、遗忘门和输出门，$\mathbf{c}_t^k$ 和 $\mathbf{h}_t^k$ 是对应的记忆cell向量和隐状态向量。
    - $G_k$ 未激活，LSTM单元保持不变：
    $$\begin{align*}
    \mathbf{c}_t^k&=\mathbf{c}_{t-1}^k\\
    \mathbf{h}_t^k&=\mathbf{h}_{t-1}^k
    \end{align*}$$

### 6.4.2 两种反馈策略
- LSTM的反馈机制通过从时间步 $t-1$ 到 $t$ 的循环连接来实现。由于MT-LSTM群组的更新频率不同，我们可以将不同的组看作人类的记忆。快速组是短期记忆，而慢速组是长期记忆。
- 因此，我们设计了两种反馈策略来定义不同组之间的连接模式，如下图：

![](./img/6.4.2MT-LSTM.png ':size=60%')

虚线表示反馈连接，实线表示当前时刻的连接。
#### 快到慢策略
- 直觉上，当我们将短时记忆积累到一定程度时，我们将一些有价值的信息从短时记忆存储到长时记忆中。因此，我们首先定义了一种从快到慢的策略，即用较快的组更新较慢的组。
- 当且仅当 $T_j{\leq}T_k$ 时存在群 $j$ 到群 $k$ 的连接。当 $T_j>T_k$ 时，权重矩阵 $\mathbf{U}_i^{j{\rightarrow}k},\mathbf{U}_f^{j{\rightarrow}k},\mathbf{U}_o^{j{\rightarrow}k}$,$\mathbf{U}_c^{j{\rightarrow}k},\mathbf{V}_i^{j{\rightarrow}k},\mathbf{V}_f^{j{\rightarrow}k},\mathbf{V}_o^{j{\rightarrow}k}$ 置为0。

#### 慢到快策略
- 这种策略的动机是，长时记忆可以"蒸馏"成短时记忆。
- 当且仅当 $T_j{\geq}T_i$ 时存在群 $j$ 到群 $i$ 的连接。当 $T_j<T_k$ 时，权重矩阵 $\mathbf{U}_i^{j{\rightarrow}k},\mathbf{U}_f^{j{\rightarrow}k},\mathbf{U}_o^{j{\rightarrow}k}$,$\mathbf{U}_c^{j{\rightarrow}k},\mathbf{V}_i^{j{\rightarrow}k},\mathbf{V}_f^{j{\rightarrow}k},\mathbf{V}_o^{j{\rightarrow}k}$ 置为0。

### 6.4.3 动态选择MT-LSTM单元群组数
- 另一个考虑是需要使用多少群组。一个直观的方法是，对于长文本，我们需要比短文本更多的组。组的数量取决于文本的长度。
- 我们使用一个动态策略来选择最大的组数 $g$，然后根据不同的任务选择最佳的超参数 $g$。组数的上界由下式给出：
$$g={\rm{log}}_2L-1$$
其中 $L$ 为语料的平均长度。
- 因此，最慢的组至少被激活两次。

### 6.4.4 训练
- 最后时刻的隐藏层传入一个全连接层和一个softmax分类器，预测给定句子类别的概率分布。
- 通过最小化交叉熵损失来优化，目标函数包含一个L2正则项。
- 通过反向传播训练网络，使用AdaGrad优化更新。

## 6.5 2016-oh-2LSTMp


## 6.6 2016-BLSTM-2DCNN

## 6.7 2016-Multi-Task