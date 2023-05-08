# RNN系列论文
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
- 半监督文本分类的对抗训练方法</br>
(Adversarial Training Methods for Semi-Supervised Text Classification)

### 6.5.1 模型
- 我们记 $T$ 个单词的序列为 $\{w^{(t)}|t=1,\cdots,T\}$，序列对应的标签为 $y$。
- 定义词嵌入矩阵 $\mathbf{V}\in\mathbb{R}^{(K+1){\times}D}$，其中 $K$是词汇表的大小，每一行 $v_k$ 对应于第 $k$ 个单词的词嵌入。定义第 $(K+1)$ 个词嵌入为特殊的句子结尾词嵌入 $v_{\rm{eos}}$。
- 我们使用一个简单的LSTM神经网络模型来做文本分类，在时间步 $t$，输入离散单词 $w^{(t)}$ 和对应的词嵌入 $v^{(t)}$。如下图：

![](./img/6.5.1oh-2LSTMp.png ':size=60%')
- 我们尝试引入双向LSTM，通过concat模型输出来预测标签。
- 在对抗和虚拟对抗训练中，我们训练分类器使其对嵌入的扰动具有鲁棒性，如下图所示：

![](./img/6.5.2oh-2LSTMp.png ':size=60%')
- 该模型可以通过学习具有非常大范数的嵌入来使扰动失效，为了防止这种无效的情况，当我们将对抗和虚拟对抗训练应用于上面定义的模型时，将嵌入 $v_k$ 替换为归一化的嵌入 $\bar{v}_k$：
$$\bar{v}_k=\frac{v_k-{\rm{E}}(v)}{\sqrt{{\rm{Var}}(v)}},\quad{\rm{E}}(v)=\sum\limits_{j=1}^Kf_jv_j,\quad{\rm{Var}}(v)=\sum\limits_{j=1}^Kf_j(v_j-{\rm{E}}(v))^2$$
其中 $f_j$ 是第 $j$ 个词在训练集中的频率。

### 6.5.2 对抗和虚拟对抗训练
- 对抗训练是一种新颖的分类器正则化方法，以提高对小的、近似的最坏情况扰动的鲁棒性。
- 我们记输入为 $x$，分类器的参数为 $\theta$。训练时，在损失函数中加上下面这项：
$$-{\rm{log}}p(y|x+r_{\rm{adv}};\theta)$$
其中 $r_{\rm{adv}}=\mathop{\rm{argmin}}\limits_{r,\|r\|\leq\epsilon}{\rm{log}}p(y|x+r;\^{\theta})$，$r$ 是输入上的扰动，$\^{\theta}$ 是当前分类器参数的不变的集合。使用常数拷贝而不是 $\theta$ 表示不使用反向传播算法通过对抗样本构造过程传播梯度。
- 在训练的每一步，我们识别出当前模型 $p(y|x;\^{\theta})$ 中的最坏扰动 $r_{\rm{adv}}$，通过最小化上述式子来训练模型对这些扰动的鲁棒性。然而，一般情况下我们不能精确地计算这个值，因为对于许多模型，关于 $r$ 的精确极小化是困难的。
- Goodfellow et al. 提出了一种逼近方法，通过对 ${\rm{log}}p(y|x;\^{\theta})$ 在 $x$ 附近线性逼近，以及一个L2范数，可以得到：
$$r_{\rm{adv}}=-\frac{g}{\|g\|_2}\epsilon,{\quad}g=\nabla_x{\rm{log}}p(y|x;\^{\theta})$$
通过反向传播可以轻易计算这个扰动。
- 虚拟对抗训练是对抗训练的正则版本，通过下式计算损失函数：
$${\rm{KL}}[p(\cdot|x;\^{\theta})\|p(\cdot|x+r_{\rm{v-adv}};\theta)]$$
其中 $r_{\rm{v-adv}}=\mathop{\rm{argmax}}\limits_{r,\|r\|\leq\epsilon}{\rm{KL}}[p(\cdot|x;\^{\theta})\|p(\cdot|x+r;\^{\theta})]$。
- 式中不需要实际标签 $y$，因此虚拟对抗训练能够应用于半监督学习。

#### 对抗训练
- 我们对词嵌入应用对抗扰动。记一系列词嵌入 $[\bar{v}^{(1)},\bar{v}^{(2)},\cdots,\bar{v}^{(T)}]$ 的concatenation为 $s$，模型条件概率为 $p(y|s;\theta)$。那么我们定义对抗扰动为：
$$r_{\rm{adv}}=-\frac{g}{\|g\|_2}\epsilon,{\quad}g=\nabla_s{\rm{log}}p(y|s;\^{\theta})$$
- 为了使对抗扰动更具鲁棒性，我们定义对抗损失为：
$$L_{\rm{adv}}(\theta)=-\frac{1}{N}\sum\limits_{n=1}^N{\rm{log}}p(y_n|s_n+r_{\rm{adv,n}};\theta)$$
其中 $N$ 是有标签样例数。
- 在对抗训练中，通过SGD来最小化负对数似然和 $L_{\rm{adv}}$ 的和。

#### 虚拟对抗训练
- 在虚拟对抗训练中，每一步我们计算虚拟对抗扰动的逼近式：
$$r_{\rm{v-adv}}=-\frac{g}{\|g\|_2}\epsilon,{\quad}g=\nabla_{s+d}{\rm{KL}}[p(\cdot|s;\^{\theta})\|p(\cdot|s+d;\^{\theta})]$$
其中 $d$ 是一个 $TD$ 维的很小的随机向量。
- 虚拟对抗损失函数定义为：
$$L_{\rm{v-adv}}(\theta)=-\frac{1}{N^\prime}\sum\limits_{n\prime=1}^{N^\prime}{\rm{KL}}[p(\cdot|s_{n^\prime};\^{\theta})\|p(\cdot|s_{n^\prime}+r_{\rm{v-adv},n^\prime};\^{\theta})]$$
其中 $N^\prime$ 是有标签和无标签的样例数和。


## 6.6 2016-BLSTM-2DCNN
- 结合双向Lstm和二维最大池化改进文本分类</br>
(Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling)
- 模型由4部分组成，包括BLSTM层、二维卷积层、二维最大池化层和输出层。如下图所示：

![](./img/6.6.1BLSTM-2DCNN.png ':size=90%')

### 6.6.1 BLSTM层
- 给定一个长度为 $l$ 的文本序列 $S=\{x_1,x_2,\cdots,x_l\}$，每一时间步 $t$，记忆cell $c_t$ 和隐藏状态 $h_t$ 由下式更新：
$$\quad\left[\begin{array}{c}
i_t\\
f_t\\
o_t\\
\^{c}_t
\end{array}\right]=\left[\begin{array}{c}
\sigma\\
\sigma\\
\sigma\\
{\rm{tanh}}
\end{array}\right]W\cdot[h_{t-1},x_t]$$
$$\begin{align*}
c_t&=f_t{\odot}c_{t-1}+i_t{\odot}\^{c}_t\\
h_t&=o_t{\odot}{\rm{tanh}}(c_t)
\end{align*}$$
其中 $x_t$ 是 $t$ 步的输入，$i,f,o$ 是输入门、遗忘门和输出门，$\^{c}$ 是当前的cell状态，$\sigma$ 是sigmoid函数，$\odot$ 是哈达玛积。
- 双向LSTM通过将不同时间方向的隐状态concat起来，来学习未来的信息。第 $i$ 个单词的输出如下：
$$h_i=[\mathop{h_i}\limits ^{\rightarrow}{\oplus}\mathop{h_i}\limits ^{\leftarrow}]$$

### 6.6.2 卷积神经网络
- 由于BLSTM可以访问未来和过去的上下文，因此 $h_i$ 与文本中所有其他单词相关，可以将特征向量组成的矩阵视为"图像"来处理。
#### 二维卷积层
- 由BLSTM层得到的矩阵 $H=\{h_1,h_2,\cdots,h_l\},H\in\mathbb{R}^{l{\times}d^w}$，其中 $d^w$ 是词向量的大小。利用窄卷积提取 $H$ 上的局部特征。
- 卷积算子包括一个2D过滤器 $\mathbf{m}\in\mathbb{R}^{k{\times}d}$，应用在 $k$ 个单词和 $d$ 个特征向量的窗口上。具体如下式：
$$o_{i,j}=f(\mathbf{m}{\cdot}H_{i:i+k-1,j:j+d-1}+b)$$
其中 $i\in[1,l-k+1],j\in[1,d^w-d+1]$，$\cdot$ 是点积，$b\in\mathbb{R}$ 是偏置项，$f$ 是非线性函数。
- 应用于矩阵 $H$ 的每个窗口得到特征图 $O$：
$$O=[o_{1,1},o_{1.2},\cdots,o_{l-k+1,d^w-d+1}]$$
- 上面描述了一个过滤器，卷积层可以设置多个相同大小的过滤器学习互补特征，也可以设置多个不同大小的过滤器。

#### 二维最大池化层
- 利用2D最大池化得到固定长度的向量。2D最大池化 $p\in\mathbb{R}^{p_1{\times}p_2}$ 对矩阵 $O$ 的每个窗口提取最大值：
$$p_{i,j}=down(O_{i:i+p_1,j:j+p_2})$$
其中 $down(\cdot)$ 表示2D最大池化函数，$i=(1,1+p_1,\cdots,1+(l-k+1/p_1-1){\cdot}p_1)$，$j=(1,1+p_2,\cdots,1+(d^w-d+1/p_2-1){\cdot}p_2)$。
- 池化的结果合并为：
$$h^*=[p_{1,1},p_{1,1+p_2},\cdots,p_{1+(l-k+1/p_1-1),1+(d^w-d+1/p_2-1){\cdot}p_2}]$$
其中 $h^*\in\mathbb{R}$，$h^*$ 的长度为 ${\lfloor}l-k+1/p_1{\rfloor}\times{\lfloor}d^w-d+1/p_2{\rfloor}$。

### 6.6.3 输出层
- 对于文本分类，2D最大池化层的输出 $h^*$ 是输入文本 $S$ 的整体表示。然后传给softmax分类器，预测语义关系标签 $\^{y}$：
$$\begin{align*}
\^{p}(y|s)=&{\rm{softmax}}(W^{(s)}h^*+b^{(s)})\\
\^{y}&={\rm{arg}}\mathop{\rm{max}}\limits_y\^{p}(y|s)
\end{align*}$$
- 目标函数为最小化正则交叉熵损失：
$$J(\theta)=-\frac{1}{m}\sum\limits_{i=1}^mt_i{\rm{log}}(y_i)+\lambda\|\theta\|^2_F$$
其中 $t\in\mathbb{R}^m$ 是ground truth独热表示，$y\in\mathbb{R}^m$ 是softmax分类出的每个类别的估计概率，$m$ 是目标类别的个数，$\lambda$ 是L2正则超参数。
- 通过小批量SGD训练，AdaDelta优化更新。

## 6.7 2017-DeepMoji
- 使用数以百万计的emoji学习任意域表示来检测情感、情绪和讽刺</br>
(Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm)

### 6.7.1 预训练
- 在很多情况下，表情符号是文本情感内容的代替。因此，在预测哪个表情最初是文本的一部分的分类任务上进行预训练，可以提高在目标任务上的表现。社交媒体中包含大量带有emoji符号的短文本，可以作为噪音标签进行预训练。
- 我们使用了Twitter从2013年1月1日到2017年6月1日的数据，相对的，使用任何带有emoji的数据集都可以。
- 预训练数据集只使用没有URL的英文推文。我们的假设是，从URL中获取的内容对于理解推文中的情感内容很可能是重要的。因此，我们移除了URL。
- 恰当的标记化对泛化很重要。具有2个或2个以上重复字符的单词被缩短为同一个token(例如'loool'和'looooool'被标记为相同)。类似地，我们对所有URL、艾特@、编号等使用同一个特殊的token来替代。
- 必须包含至少1个非标点、非emoji、非特殊token的词句才会纳入训练集。
- 许多推文包含同一表情符号的多次重复或多个不同表情符号。在训练数据中，我们对于每个独特的表情符号类型，保存一个单独的tweet，以该表情符号为标签进行预训练，多次重复只保留1个。

### 6.7.2 模型
- 模型结构如下图：

![](./img/6.7.1DeepMoji.png ':size=60%')
- 包括双向LSTM层，Attention层，以及残差连接，最后接入softmax分类器。
- 模型在构建文本的表示时，注意力机制能够权衡每个单词对于预测任务的重要性：
$$\begin{align*}
e_t&=h_tw_a\\
a_t&=\frac{{\rm{exp}}(e_t)}{\sum_{i=1}^T{\rm{exp}}(e_i)}\\
v&=\sum\limits_{i=1}^Ta_ih_i
\end{align*}$$
其中 $h_t$ 是时间步 $t$ 的词表示，$w_a$ 是注意力权重。$a_t$ 是注意力分数，$v$ 是文本的向量表示。
- 注意力机制和残差连接提高了模型性能。

### 6.7.3 迁移学习
- 通过禁用参数更新来防止过拟合，我们的预训练模型可以通过多种方式对目标任务进行微调，并使用一些方法"冻结"层。
- 一种常见的做法是使用网络作为特征提取器，在对目标任务进行微调时，除了最后一层外，模型中的所有层都被冻结，简记为last。或者，另一种常见的方法是使用预训练的模型作为初始化，其中完整的模型是未冻结的，简记为full。
- 我们提出了一种新的简单的迁移学习方法："链式解冻"(chain-thaw)，每次依次解冻和微调单层。该方法以牺牲微调所需的额外计算能力为代价，提高了目标任务的准确性。通过分别训练每一层，模型能够调整整个网络中的模式，降低过拟合的风险。在无监督学习的情况下，序列微调似乎具有类似于逐层训练的正则化效果。
- 具体地说，链式解冻方法首先将任何新的层(往往只有一个Softmax层)微调到目标任务，直到在验证集上收敛。然后从网络中的第一层开始逐层微调。最后，利用所有层对整个模型进行训练。每次模型在验证集上收敛时，权重被重新加载到最佳设置，从而以类似于提前停止的方式防止过拟合。如下图所示：

![](./img/6.7.2DeepMoji.png ':size=60%')

仅执行步骤a与 last 方法相同，只做步骤d与 full 方法相同。
- 链式解冻方法的一个好处是能够将词汇表扩展到新的领域，而不会有过拟合的风险。对于给定的数据集，多达10000个来自训练集的新词被添加到词汇表中。

## 6.8 2017-TopicRNN
- 一个具有长程语义依赖的循环神经网络</br>
(A Recurrent Neural Network with Long-Range Semantic Dependency)

### 6.8.1 概率主题模型
- 概率主题模型是用来捕捉全局语义一致性的模型族。它们为文档集合的总结、组织和导航提供了强有力的工具。
- 这类模型的一个基本目标是找到一组词，它们倾向于共同出现在同一文档中。这些词组被称为主题，代表一种概率分布。然后，文档被表示为这些潜在主题的混合物，通过后验推断，学习到的主题捕获了词聚类的语义一致性。
- 最简单的概率主题模型是潜在狄利克雷分配(LDA)。假定 $K$ 个主题 $\beta=\{\beta_1,\cdots,\beta_K\}$，每一个都是词汇表上的一个概率分布。LDA生成过程如下：
    - 首先随机生成 $K$ 个话题的单词分布 $\beta_k\sim_{iid}{\rm{Dirichlet}}(\tau)$，服从狄利克雷分布。
    - 然后对于每个文档中包含的词 $y_{1:T}$，独立的生成文档级别的变量和数据：
        - 根据超参数 $\alpha$ 随机生成参数向量 $\theta\sim{\rm{Dirichlet}}(\alpha)$；
        - 对文档的第 $t$ 个单词：
            - 生成话题 $z_t\sim{\rm{Dirichlet}}(\theta)$；
            - 然后生成单词 $y_t\sim{\rm{Dirichlet}}(\beta_{z_t})$；
    - 忽略 $z_t$，我们可以得到关于单词的概率，具体为：
    $$p(y_{1:T}|\beta)={\int}p(\theta)\prod\limits_{t=1}^T\sum\limits_{z_t}p(z_t|\theta)p(y_t|z_t,\beta)d\theta={\int}p(\theta)\prod\limits_{t=1}^T(\beta\theta)_{y_t}d\theta$$
- 很多主题模型是词袋模型，缺点是忽略了词序。另一个问题是它们不能很好地建模停用词。这是因为停用词通常不携带语义，它们的出现主要是为了根据语言的语法使句子更具可读性。停用词频繁地出现在几乎每一个文档中，并且几乎可以与任何词共现。

### 6.8.2 TopicRNN模型
#### 模型
- TopicRNN是一个生成模型，对于文本中包含的词 $y_{1:T}$：
    - 从一个均匀分布生成话题向量 $\theta\sim\mathcal{N}(\mathbf{0},\mathbf{1})$；
    - 给定词 $y_{1:t-1}$，对于文本中第 $t$ 个词 $y_t$：
        - 计算隐藏状态 $h_t=f_W(x_t,h_{t-1})$，其中我们令 $x_t{\triangleq}y_{t-1}$；
        - 从伯努利分布中生成停用词指示物 $l_t\sim{\rm{Bernoulli}}(\sigma(\Gamma^{\top}h_t))$，$\sigma$ 是sigmoid函数；
        - 生成单词 $y_t{\sim}p(y_t|h_t,\theta,l_t,B)$，其中：
        $$p(y_t=i|h_t,\theta,l_t,B)\propto{\rm{exp}}(v_i^{\top}h_t+(1-l_t)b_i^{\top}\theta)$$
- 停用词指示物 $l_t$ 控制 $\theta$ 对输出的影响。如果 $l_t=1$，说明 $y_t$ 是一个停用词，$\theta$ 对输出没有影响。
- 我们选择使用主题向量作为偏置而不是将其传递到RNN的隐藏状态的主要原因是，它使我们能够清晰地分离全局语义和局部动力的贡献。全局语义来自于排除停用词时有意义的主题。然而这些停用词是语言模型的局部动力学所需要的。因此，我们通过停用词的二元决策模型来实现这种全局与局部的分离。
- 展开的模型结构如下图：

![](./img/6.8.1TopicRNN.png ':size=60%')
- 我们用 $\Theta=\{\Gamma,V,B,W,W_c\}$ 来表示所有的模型参数，$W_c$ 是推断网络的参数(后面会介绍)。词序列为 $y_{1:T}$，停用词指示物为 $l_{1:T}$，对数边际似然目标函数为：
$${\rm{log}}p(y_{1:T},l_{1:T}|h_t)={\rm{log}}{\int}p(\theta)\prod\limits_{t=1}^Tp(y_t|h_t,l_t,\theta)p(l_t|h_t)d\theta$$

#### 模型推断
- 直接优化目标函数是困难的，我们使用变分推断来逼近这个边际函数。令 $q(\theta)$ 是 $\theta$ 上的变分分布。证据下界ELBO为：
$$\begin{align*}
\mathcal{L}(y_{1:T},l_{1:T}|q(\theta),\Theta)&\triangleq\mathbb{E}_{q(\theta)}\left[\sum\limits_{t=1}^T{\rm{log}}p(y_t|h_t,l_t,\theta)+{\rm{log}}p(l_t|h_t)+{\rm{log}}p(\theta)-{\rm{log}}q(\theta)\right]\\
&\leq{\rm{log}}p(y_{1:T},l_{1:T}|h_t,\Theta)
\end{align*}$$
- 根据变分自编码器技术，我们选择 $q(\theta)$ 的形式作为推理网络。令 $X_c\in\mathcal{N}_+^{|V_c|}$ 是不包括停用词的 $y_{1:T}$ 的词频表示，$V_c$ 是不包括停用词的词汇表大小。
- 变分自编码推断网络 $q(\theta|X_c,W_c)$ 将 $X_c$ 映到一个 $K$ 维隐变量空间中，具体为：
$$\begin{align*}
q(\theta|X_c,W_c)&=\mathcal{N}(\theta;\mu(X_c),{\rm{diag}}(\sigma^2(X_c)))\\
\mu(X_c)&=W_1g(X_c)+a_1\\
{\rm{log}}\sigma(X_c)&=W_2g(X_c)+a_2
\end{align*}$$
其中 $g(\cdot)$ 表示前馈神经网络，权重矩阵 $W_1,W_2$ 和偏置项 $a_1,a_2$ 在文档中共享参数。每个文档有其自己的 $\mu(X_c)$ 和 $\sigma(X_c)$，服从分布 $q(\theta|X_c)$。
- 推断网络输出一个分布作为语义信息的汇总，训练中使用Adam算法联合优化并更新参数，我们从 $q(\theta|X_c)$ 中随机采样，并使用重参数化技巧。具体网络见下图：

![](./img/6.8.2TopicRNN.png ':size=60%')

#### 生成序列文本
- 给定词序列 $y_{1:t-1}$，我们有 $q(\theta|X_c)$ 的一个初始估计。为了生成下一个词 $y_t$，我们在线计算其概率分布。我们选择 $\theta$ 作为当前分布 $q(\theta|X_c)$ 的均值的点估计 $\^{\theta}$。
- 边缘化停用词指示物 $l_t$，$y_t$ 的逼近分布为：
$$p(y_t|y_{1:t-1})\approx\sum\limits_{l_t}p(y_t|h_t,\^{\theta},l_t)p(l_t|h_t)$$
- 预测词 $y_t$ 就是来自这个预测分布的样本。如果 $y_t$ 不是停用词，我们通过将 $y_t$ 加入到 $X_c$ 来更新 $q(\theta|X_c)$。然而，在每个单词预测之后更新 $q(\theta|X_c)$ 是昂贵的，所以我们使用了一个滑动窗口。

#### 模型复杂度
- TopicRNN的算法复杂度为 $O(H{\times}H+H{\times}(C+K)+W_c)$，其中 $H$ 是RNN隐藏层的大小，$C$ 是词汇表大小，$K$ 是话题向量的维度，$W_c$ 是推断网络的参数个数。

## 6.9 2017- Miyato et al.
- 半监督文本分类的对抗训练方法</br>
(Adversarial Training Methods for Semi-Supervised Text Classification)

### 6.9.1 模型
- 我们记 $T$ 个单词的序列为 $\{w^{(t)}|t=1,\cdots,T\}$，对应的目标为 $y$。为了将离散的单词变为连续的向量，我们定义词嵌入矩阵 $V\in\mathbb{R}^{(K+1){\times}D}$，其中 $K$ 是词汇表的单词数，矩阵的每一行 $v_i$ 对应于第 $i$ 个单词的词嵌入。第 $K+1$ 个词嵌入为end of sequence的token，记为 $v_{eos}$。
- 我们使用一个基于LSTM的神经网络模型，在时间步 $t$ 的输入是离散的单词 $w^{(t)}$，对应的词嵌入为 $v^{(t)}$。如下图：

![](./img/6.9.1Miyato%20et%20al..png ':size=60%')
- 我们使用双向LSTM，将结果concat到一起进行预测。
- 在对抗和虚拟对抗训练中，我们训练分类器使其对嵌入的扰动具有鲁棒性，如下图：

![](./img/6.9.2Miyato%20et%20al..png ':size=60%')
- 为了防止具有大范数的嵌入使扰动变得不重要，我们通过下式归一化向量 $v_k$，记为 $\bar{v}_k$：
$$\bar{v}_k=\frac{v_k-{\rm{E}}(v)}{\sqrt{{\rm{Var}}(v)}},{\quad}{\rm{E}}(v)=\sum\limits_{j=1}^Kf_jv_j,{\rm{Var}}(v)=\sum\limits_{j=1}^Kf_j(v_j-{\rm{E}}(v))^2$$
其中 $f_i$ 是第 $i$ 个单词在所有单词中的频率。

### 6.9.2 对抗和虚拟对抗训练
#### 对抗训练
- 记 $x$ 为输入，$\theta$ 是分类器的参数，对抗训练将下述的项加入到损失函数中：
$$-{\rm{log}}p(y|x+r_{\rm{adv}};\theta)$$
其中 $r_{\rm{adv}}=\mathop{\rm{argmin}}\limits_{r,\|r\|\leq\epsilon}{\rm{log}}p(y|x+r;\^{\theta})$，$r$ 是扰动项，$\^{\theta}$ 是当前分类器参数的不变的拷贝集合。使用拷贝而不是 $\theta$ 说明不使用反向传播算法传播梯度。
- 在训练的每一步，我们都识别出当前模型 $p(y|x;\^{\theta})$ 在等式中的最坏情况扰动 $r_{\rm{adv}}$，并通过argmin来训练模型对这些扰动的鲁棒性。然而，关于 $r$ 的精确极小化是困难的。我们使用以下逼近式：
$$r_{\rm{adv}}=-{\epsilon}g/\|g\|_2,{\quad}g=\nabla_x{\rm{log}}p(y|x;\^{\theta})$$

#### 虚拟对抗训练
- 虚拟对抗训练与对抗训练类似，是一种正则化方法。目标函数如下：
$${\rm{KL}}[p(\cdot|x;\^{\theta})\|p(\cdot|x+r_{{\rm{v}}\text{-}{\rm{adv}}};\theta)]$$
$$r_{{\rm{v}}\text{-}{\rm{adv}}}=\mathop{\rm{argmax}}\limits_{r,\|r\|\leq\epsilon}{\rm{KL}}[p(\cdot|x;\^{\theta})\|p(\cdot|x+r;\^{\theta})]$$
- 与对抗扰动最大的区别是，式中不需要真实标签 $y$，因此可以应用于半监督学习。

#### 词嵌入扰动
- 我们在词嵌入上进行对抗扰动，而不是直接在输入上进行。我们将词嵌入的concatenation $[\bar{v}^{(1)},\bar{v}^{(2)},\cdots,\bar{v}^{(T)}]$ 记为 $s$，关于 $y$ 的条件概率为 $p(y|s;\theta)$。
- 然后我们在 $s$ 上定义对抗扰动为：
$$r_{\rm{adv}}=-{\epsilon}g/\|g\|_2,{\quad}g=\nabla_s{\rm{log}}p(y|s;\^{\theta})$$
- 为了使上式更具鲁棒性，我们定义对抗损失为：
$$L_{\rm{adv}}(\theta)=-\frac{1}{N}\sum\limits_{n=1}^N{\rm{log}}p(y_n|s_n+r_{{\rm{adv}},n};\theta)$$
其中 $N$ 是有标签样本数。
- 在虚拟对抗训练中我们计算下列逼近式：
$$r_{{\rm{v}}\text{-}{\rm{adv}}}={\epsilon}g/\|g\|_2,{\quad}g=\nabla_{s+d}{\rm{KL}}[p(\cdot|s;\^{\theta})\|p(\cdot|s+d;\^{\theta})]$$
其中 $d$ 是一个 $TD$ 维的小随机向量，逼近式对应于2阶泰勒展开。
- 虚拟对抗损失函数定义为：
$$L_{{\rm{v}}\text{-}{\rm{adv}}}(\theta)=\frac{1}{N^\prime}\sum\limits_{n^\prime=1}^{N^\prime}{\rm{KL}}[p(\cdot|s_{n^\prime};\^{\theta})\|p(\cdot|s_{n^\prime}+r_{{\rm{v}}\text{-}{\rm{adv}},n^\prime};\theta)]$$
其中 $N^\prime$ 是有标签和无标签样本的总数。