# Attention系列论文泛读
## 7.1 2016-HAN
- 用于文档分类的层次注意力网络</br>
(Hierarchical Attention Networks for Document Classification)
- 层次注意力网络的整体架构如图，它由几个部分组成：词序列编码器、词级注意力层、句子编码器和句子级注意力层。

![](./img/7.1.1HAN.png ':size=60%')

### 7.1.1 基于GRU的序列编码器
- GRU使用门控机制来跟踪序列的状态，而不使用单独的记忆单元。门有两种类型：重置门 $r_t$ 和更新门 $z_t$，它们共同控制信息更新。在时间 $t$，GRU通过下式计算：
$$h_t=(1-z_t){\odot}h_{t-1}+z_t{\odot}\tilde{h}_t$$
- $z_t$ 通过下式更新：
$$z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z)$$
其中 $x_t$ 是时间 $t$ 的序列向量。
- 候选状态 $\tilde{h}_t$ 由下式计算：
$$\tilde{h}_t={\rm{tanh}}(W_hx_t+r_t{\odot}(U_hh_{t-1})+b_h)$$
其中 $r_t$ 是重置门，控制过去状态对候选状态的贡献程度。如果 $r_t$ 为零，则忘记之前的状态。重置门按照下式更新：
$$r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)$$

### 7.1.2 分层注意力机制
- 我们关注文档分类任务。假定一个文档有 $L$ 个句子 $s_i$，每个句子有 $T_i$ 个单词。第 $i$ 个句子的单词为 $w_{it},t\in[1,T]$。模型输出一个向量表示，放入一个分类器来执行文档分类。
#### 词编码器
- 首先通过嵌入矩阵 $W_e$ 得到词嵌入 $x_{ij}=W_ew_{ij}$。
- 使用双向GRU，将两个方向的信息进行汇总得到单词的注释，从而将上下文信息融入到注释中。
- 双向GRU包括前向 $\mathop{f}\limits^{\rightarrow}$ 和后向 $\mathop{f}\limits ^{\leftarrow}$，从不同的方向输入句子向量：
$$\begin{align*}
x_{it}&=W_ew_{it},&t\in[1,T]\\
\mathop{h_{it}}\limits ^{\rightarrow}&=\mathop{\rm{GRU}}\limits ^{\longrightarrow}(x_{it}),&t\in[1,T]\\
\mathop{h_{it}}\limits ^{\leftarrow}&=\mathop{\rm{GRU}}\limits ^{\longleftarrow}(x_{it}),&t\in[T,1]
\end{align*}$$
- 通过将前向后向隐藏状态concat起来，得到一个注释，聚合了以 $w_{it}$为中心的整个序列的信息：
$$h_{it}=[\mathop{h_{it}}\limits ^{\rightarrow},\mathop{h_{it}}\limits ^{\leftarrow}]$$

#### 词注意力
- 对于一个句子的意义，并非所有单词都有相同的贡献。因此，我们引入注意力机制来提取对句子意义重要的词，并聚合这些词的表示，形成句子向量。
- 具体来说，我们首先将得到的词注释 $h_{it}$ 放到一个一层的MLP中得到 $u_{it}$ 作为隐藏表示，然后我们评估它和词级上下文向量 $u_w$ 的相似度，通过softmax分类器来获得一个重要性权重 $\alpha_{it}$。在这之后，我们通过权重和来计算句子向量 $s_i$：
$$\begin{align*}
u_{it}&={\rm{tanh}}(W_wh_{it}+b_w)\\
\alpha_{it}&=\frac{{\rm{exp}}(u_{it}^{\top}u_w)}{\sum_t{\rm{exp}}(u_{it}^{\top}u_w)}\\
s_i&=\sum\limits_t\alpha_{it}h_{it}
\end{align*}$$
- 上下文向量 $u_w$ 可以看作是这类固定的query "what is the informative word" 的高维表示，在训练过程中随机初始化并联合学习。

#### 句子编码器
- 给定句子向量 $s_i$，我们同样通过一个双向GRU来编码句子：
$$\begin{align*}
\mathop{h_i}\limits ^{\rightarrow}&=\mathop{\rm{GRU}}\limits ^{\longrightarrow}(s_i),&i\in[1,L]\\
\mathop{h_i}\limits ^{\leftarrow}&=\mathop{\rm{GRU}}\limits ^{\longleftarrow}(s_i),&i\in[L,1]
\end{align*}$$
- 将前向后向隐藏状态concat起来，得到一个注释，聚合了附近句子的信息，但重点仍在中心句子 $i$ 上：
$$h_i=[\mathop{h_i}\limits ^{\rightarrow},\mathop{h_i}\limits ^{\leftarrow}]$$

#### 句子注意力
- 为了奖励作为正确分类文档线索的句子，我们再次使用注意力机制并引入句子级别的上下文向量 $u_s$，并使用该向量度量句子的重要性。
- 具体来说：
$$\begin{align*}
u_i&={\rm{tanh}}(W_sh_i+b_s)\\
\alpha_i&=\frac{{\rm{exp}}(u_i^{\top}u_s)}{\sum_i{\rm{exp}}(u_i^{\top}u_s)}\\
v&=\sum\limits_i\alpha_ih_i
\end{align*}$$
- $v$ 是文档向量，它概括了文档中句子的所有信息。
- 同样，句子级别的上下文向量 $u_s$ 可以在训练过程中随机初始化并联合学习。

### 7.1.3 文档分类
- 文档向量 $v$ 是文档的高维表示，可以作为文档分类的特征，通过MLP和softmax分类器得到分类概率：
$$p={\rm{softmax}}(W_cv+b_c)$$
- 损失函数使用正确标签的负对数似然来训练：
$$L=-\sum\limits_d{\rm{log}}p_{dj}$$
其中 $j$ 是文档 $d$ 的标签，


## 7.2 2016-BI-Attention
- 用于跨语言情感分类的基于注意力机制的LSTM网络</br>
(Attention-based LSTM Network for Cross-Lingual Sentiment Classification)
### 7.2.1 准备工作
#### 问题定义
- 跨语言情感分类任务的目标是使用源语言的训练集数据来构建一个模型，能够适用于目标语言的测试集。
- 我们的训练集是有标签的英语数据集：
$$L_{EN}=\{x_i,y_i\}^N_{i=1}$$
其中 $x_i$ 为评论文本，$y_i$ 为情感标签向量，$(1,0)$ 为积极的，$(0,1)$ 为消极的。
- 目标语言为中文，测试集为：
$$T_{CN}=\{x_i\}^T_{i=1}$$
- 以及中文无标签数据集：
$$U_{CN}=\{x_i\}^M_{i=1}$$
- 任务是通过 $L_{EN}$ 和 $U_{CN}$ 去学习一个模型来对 $T_{CN}$ 进行情感分类。
- 我们通过在线翻译将有标签、无标签和测试数据都翻译为另一种语言。我们将一篇文档及其对应的译文称为一对平行文档。

#### LSTM
- 我们使用标准的LSTM网络，结构如下图：

![](./img/7.2.1BI-Attention.png ':size=50%')

### 7.2.2 模型架构
#### 模型结构
- 模型的主要结构如图所示(图中展示了中文部分的结构，英文部分结构与之相同，只是模型参数不同)：

![](./img/7.2.2BI-Attention.png ':size=80%')
- 对于一对平行文档 $x_{cn}$ 和 $x_{en}$，分别送入基于注意力机制的LSTM网络中(如图中的中文网络结构)。
- 整个模型分为4层。
    - 输入层中，文档被表示为词的序列，通过预训练模型得到词嵌入；
    - LSTM层中，我们由一个双向LSTM网络得到高维表示；
    - 在文档表示层，我们引入attention模型来得到最终的文档表示；
    - 在输出层，我们将平行文档的中文部分和英文部分的表示concat起来，然后使用softmax分类器来预测情感标签。
###### 输入层
- 中文或英文文档 $x$，包含了句子序列 $\{s_i\}_{i=1}^{|x|}$，每个句子包含了一些单词 $s_i=\{w_{i,j}\}_{j=1}^{|s_i|}$，我们通过一个预训练词嵌入模型来将文档中的每个词表示为一个固定大小的词向量。

###### LSTM层
- 在每种语言中，我们使用双向LSTM对输入序列进行建模。我们从前向LSTM网络中得到前向隐藏状态 ${\mathop{h}\limits^{\rightarrow}}_{i,j}$，从后向LSTM网络中得到后向隐藏状态 ${\mathop{h}\limits^{\leftarrow}}_{i,j}$。
- 将两个隐藏状态concat到一起，得到第 $i$ 个句子的第 $j$ 个单词的状态 $h_{i,j}$:
$$h_{i,j}={\mathop{h}\limits^{\rightarrow}}_{i,j}\|{\mathop{h}\limits^{\leftarrow}}_{i,j}$$
其中 $\|$ 表示concat操作。

###### 文档表示层
- 文档的不同部分对于整体情感通常具有不同的重要性。有些句子或词语可以是决定性的，而另一些句子或词语则是无关的。
- 我们使用分层注意力机制，为每个单词分配一个真实值分数，为每个句子分配一个真实值分数。
- 假定我们对每个句子 $s_i{\in}x$ 有一个注意力分数 $A_i$，对每个单词 $w_{i,j}{\in}s_i$ 有一个注意力分数 $a_{i,j}$，我们对分数进行归一化后，得到：
$$\sum\limits_iA_i=1,\quad\sum\limits_ja_{i,j}=1$$
- 句子注意力衡量的是哪个句子对整体情感更重要，而单词注意力捕捉的是每个句子中的情感信号。因此，文档 $x$ 的文档表示 $r$ 计算如下：
$$r=\sum\limits_i[A_i\cdot\sum\limits_j(a_{i,j}{\cdot}h_{i,j})]$$

###### 输出层
- 在输出层，需要对文档的整体情感进行预测。对于每一个英文文档 $x_{en}$ 及其对应的译文 $x_{cn}$，假设它们的文档表示为 $r_{en}$ 和 $r_{cn}$，我们通过concat操作来得到最终的特征向量，并使用softmax分类器预测最终的情感标签：
$$\^{y}={\rm{softmax}}(r_{cn}\|r_{en})$$

#### 分层注意力机制
- 结合双语言LSTM网络的分层注意力机制。
- 第一个层次是句子注意力模型，衡量哪些句子对文档的整体情感更重要。对每个句子 $s_i=\{w_{i,j}\}_{j=1}^{|s_i|}$，我们通过前向和后向隐藏状态来得到句子表示：
$$s_i={\mathop{h}\limits^{\rightarrow}}_{i,|s_i|}\|{\mathop{h}\limits^{\leftarrow}}_{i,1}$$
- 我们使用两层前馈神经网络 $f$ 来预测注意力分数，$\theta_s$ 是网络 $f$ 的参数：
$$\^{A}=f(s_i;\theta_s)$$
$$A_i=\frac{{\rm{exp}}(\^{A}_j)}{\sum_j{\rm{exp}}(\^{A}_j)}$$
- 在词的层次，我们通过词本身的词嵌入，以及前向后向隐藏状态来得到词表示：
$$e_{i,j}=w_{i,j}\|{\mathop{h}\limits^{\rightarrow}}_{i,j}\|{\mathop{h}\limits^{\leftarrow}}_{i,j}$$
- 同样使用两层网络 $f$ 来预测注意力分数，$\theta_w$ 是网络 $f$ 的参数：
$$\^{a}_{i,j}=f(e_{i,j};\theta_w)$$
$$a_{i,j}=\frac{{\rm{exp}}(\^{a}_{i,j})}{\sum_j{\rm{exp}}(\^{a}_{i,j})}$$

### 7.2.3 模型训练
- 模型以半监督的方式进行训练。在监督部分，我们通过最小化交叉熵损失来训练：
$$L_1=\sum\limits_{(x_{en},x_{cn})}\sum\limits_i-y_i{\rm{log}}(\^{y}_i)$$
其中 $x_{en}$ 和 $x_{cn}$ 是训练集中的平行文档，$y$ 是标签的情感向量，$\^{y}$ 是模型的预测向量。
- 无监督部分通过最小化并行数据之间的文档表示：
$$L_2=\sum\limits_{(x_{en},x_{cn})}\|r_{en}-r_{cn}\|^2$$
其中 $x_{en}$ 和 $x_{cn}$ 是有标签和无标签数据集中的并行文档。
- 最终的目标函数是两部分的权重和，$\alpha$ 是超参数：
$$L=L_1+{\alpha}L_2$$
- 模型使用Adadelta来更新参数。
- 在测试阶段，将 $T_{CN}$ 中的测试文档与 $T_{EN}$ 中对应的机器翻译文本一起发送到我们的模型中。最后的情感预测通过一个softmax分类器对双语文本的concat表示进行预测，如前所述。


## 7.3 2016-LSTMN
- 用于机器阅读的长短期记忆网络</br>
(Long Short-Term Memory-Networks for Machine Reading)

### 7.3.1 机器阅读
- 我们模型的核心是一个LSTM单元，它带有一个扩展的记忆磁盘，显式地模拟人类的记忆广度。
- 该模型在每一时间步以基于注意力的记忆寻址机制来执行tokens之间的隐式关系分析。

#### LSTM
- 通过在单个记忆槽中增量地添加新内容来处理可变长度序列 $x=(x_1,x_2,\cdots,x_n)$，每一时间步 $t$，记忆cell $c_t$ 和隐藏状态 $h_t$ 由下式更新：
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

#### LSTM网络
- LSTM的第一个问题是：在递归过程中，它们能够记忆序列的程度。
    - LSTMs可以产生一个状态表示列表，但是下一个状态总是从当前状态计算得到。也就是说，给定当前状态 $h_t$，下一个状态 $h_{t+1}$ 条件独立于状态 $h_1,\cdots,h_t$ 和tokens $x_1,\cdots,x_t$。
    - 当递归状态以马尔科夫方式更新时，假设LSTMs保持无界记忆(仅当前状态就很好地概括了迄今为止所看到的特征)。这种假设在实践中可能会失效，例如当序列较长或内存大小不够大时。
- LSTMs的另一个不理想的特性是关于结构化的输入。LSTM按照顺序逐个聚合信息，但没有明确的机制来推理tokens之间的结构和关系。
- 我们的模型旨在解决这两个问题。我们的解决方案是修改标准的LSTM结构，将记忆单元替换为记忆网络。由此产生的LSTMN以唯一的记忆槽存储每个输入token的上下文表示，记忆的大小随时间增长，直到达到记忆跨度的上界。
- 该设计使LSTM能够通过神经注意力层推理tokens之间的关系，然后执行非马尔科夫状态更新。具体的网络结构如下图：

![](./img/7.3.1LSTMN.png ':size=60%')

- 模型维护两套向量，分别存储在用于与环境交互的隐藏状态磁盘，和用于表示实际存储内容的记忆磁盘。因此，每个token都与一个隐藏向量和一个记忆向量相关联。
- 令 $x_t$ 表示当前输入，$C_{t-1}=(c_1,\cdots,c_{t-1})$ 表示当前的记忆磁盘，$H_{t-1}=(h_1,\cdots,h_{t-1})$ 表示之前的隐藏转态磁盘。在时间 $t$，模型根据一个注意力层来进行计算：
$$a_i^t=v^T{\rm{tanh}}(W_hh_i+W_xx_t+W_{\tilde{h}}\tilde{h}_{t-1})$$
$$s_i^t={\rm{softmax}}(a_i^t)$$
- 上式给出了tokens的隐藏状态向量上的概率分布。然后，我们可以分别为之前的隐藏磁盘和记忆磁盘计算一个自适应概要向量，记为 $\tilde{h}_t$ 和 $\tilde{c}_t$：
$$\left[\begin{array}{c}
\tilde{h}_t\\
\tilde{c}_t
\end{array}\right]=\sum\limits_{i=1}^{t-1}s_i^t\cdot\left[\begin{array}{c}
h_i\\
c_i
\end{array}\right]$$
- 然后循环更新：
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
\end{array}\right]W\cdot[\tilde{h}_t,x_t]$$
$$\begin{align*}
c_t&=f_t{\odot}\tilde{c}_t+i_t{\odot}\^{c}_t\\
h_t&=o_t{\odot}{\rm{tanh}}(c_t)
\end{align*}$$
其中 $v,W_h,W_x,W_{\tilde{h}}$ 是网络的权重项。
- LSTMN背后的一个关键思想是利用注意力来诱导tokens之间的关系。这些关系是可微的，并且是一个更大的表示学习网络的组成部分。
- 通过交替堆叠多个记忆层和隐藏层，也有可能拥有更结构化的关系推理模块，类似于堆叠LSTM或多跳记忆网络。这可以通过将 $k$ 层输出 $h_t^k$ 作为 $k+1$ 层的输入，那么第 $k+1$ 层的注意力可以通过下式计算：
$$a_{i,k+1}^t=v^T{\rm{tanh}}(W_hh_i^{k+1}+W_lh_t^k+W_{\tilde{h}}\tilde{h}_{t-1}^{k+1})$$
- 也可以通过残差将 $x_t$ 送到上层。

### 7.3.2 用LSTMN建模两个序列
- 机器翻译、文本蕴涵等自然语言处理任务关注的是两个序列的建模，而不是单个序列的建模。使用循环网络建模两个序列的标准工具是编码器-解码器架构。
- 我们将两种架构结合起来，一个是应用注意力机制进行内在关系推理的LSTMN，另一个是通过注意力模块学习两个序列之间的相互关系的编码器-解码器网络。如下图：

![](./img/7.3.2LSTMN.png ':size=100%')

#### 浅层注意力融合
- 浅层融合只是将LSTMN作为一个单独的模块，可以很容易地用于编码器-解码器架构，而不是标准的RNN或LSTM。如上图，编码器和解码器都被建模为具有内含注意力的LSTMNs。同时，当解码器读取目标token时，会触发相互注意力。

#### 深层注意力融合
- 深度融合在计算状态更新时结合了相互注意力和内含注意力。我们使用不同的符号来表示这两组注意力。
- $C$ 和 $H$ 分别表示目标记忆磁盘和隐藏磁盘，存储到目前为止已经处理过的目标符号的表示。内含注意力遵循上节定义的注意力的计算式。
- 我们使用 $A=[\alpha_1,\cdots,\alpha_m]$ 和 $Y=[\gamma_1,\cdots,\gamma_m]$ 来表示源内存磁盘和隐藏磁盘，其中 $m$ 是源序列的长度。
- 我们通过下式计算 $t$ 步输入和整个源序列中的tokens之间的相互注意力：
$$b_j^t=u^T{\rm{tanh}}(W_{\gamma}\gamma_j+W_xx_t+W_{\tilde{\gamma}}\tilde{\gamma}_{t-1})$$
$$p_j^t={\rm{softmax}}(b_j^t)$$
- 其中，自适应表示通过下式计算：
$$\left[\begin{array}{c}
\tilde{\gamma}_t\\
\tilde{\alpha}_t\\
\end{array}\right]=\sum\limits_{j=1}^mp_j^t\cdot\left[\begin{array}{c}
\gamma_j\\
\alpha_j
\end{array}\right]$$
- 通过另一个门操作 $r_t$ 将自适应源表示 $\tilde{\alpha}_t$ 转移到目标记忆中：
$$r_t=\sigma(W_r\cdot[\tilde{\gamma}_t,x_t])$$
最后得到新的目标记忆：
$$c_t=r_t{\odot}\tilde{\alpha}_t+f_t{\odot}\tilde{c}_t+i_t{\odot}\^{c}_t$$
$$h_t=o_t{\odot}{\rm{tanh}}(c_t)$$


## 7.4 2017-Lin et al
- 一个结构化的自注意力句子嵌入</br>
(a Structured Self-Attentive Sentence Embedding)

### 7.4.1 模型结构
- 模型由两部分组成。一部分是双向LSTM，另一部分是自注意力机制。
- 假定我们有一个句子序列，有 $n$ 个tokens，第 $i$ 个词的词嵌入记为 $w_i$，则句子嵌入为一个二维矩阵 $S$：
$$S=(w_1,w_2,\cdots,w_n)$$
- 我们使用双向LSTM得到句子的特征：
$$\begin{align*}
\mathop{h_t}\limits^{\rightarrow}&=\mathop{\rm{LSTM}}\limits^{\longrightarrow}(w_t,\mathop{h_{t-1}}\limits^{\longrightarrow})\\
\mathop{h_t}\limits^{\leftarrow}&=\mathop{\rm{LSTM}}\limits^{\longleftarrow}(w_t,\mathop{h_{t-1}}\limits^{\longleftarrow})
\end{align*}$$
- concat双向特征得到隐藏状态 $h_t$。所有的 $h_t$ 合并为一个特征图：
$$H=(h_1,h_2,\cdots,h_n)$$
- 将 $H$ 作为注意力机制的输入，通过下式得到一个权重向量 $a$：
$$a={\rm{softmax}}(w_{s_2}{\rm{tanh}}(W_{s_1}H^\top))$$
其中 $W_{s_1}$ 是一个 $d_a{\times}n$ 维的权重矩阵，$w_{s_2}$ 是一个 $d_a$ 维的权重向量，$d_a$ 是一个超参数。得到的 $a$ 是一个 $n$ 维向量。
- 通过权重和得到输入序列的向量表示：
$$m={\sum}aH^\top$$
- 通过多跳注意力机制来捕获长句子中的多个重要部分，我们将 $w_{s_2}$ 扩展为一个 $r{\times}d_a$ 维的矩阵，记为 $W_{s_2}$，因此我们通过下式得到一个权重矩阵 $A$：
$$A={\rm{softmax}}(W_{s_2}{\rm{tanh}}(W_{s_1}H^\top))$$
其中 ${\rm{softmax}}$ 沿着输入的第二维进行。我们可以将上式看为一个具有 $d_a$ 个隐藏单元的2层的MLP，参数为 $\{W_{s_2},W_{s_1}\}$。
- 嵌入向量 $m$ 变为一个 $r{\times}n$ 维的嵌入矩阵 $M$，具体计算为：
$$M=AH$$

### 7.4.2 惩罚项
- 如果注意力机制总是为所有 $r$ 跳提供相似的求和权重，嵌入矩阵 $M$ 可能会出现冗余问题。因此，我们需要一个惩罚项来鼓励不同注意力跳数的求和权重向量的多样性。
- 评价多样性的最好方法肯定是KL散度，但实践中并不稳定，我们猜测是因为需要优化多个KL散度的问题，在优化矩阵 $A$ 时出现很多0值使训练不稳定。另一方面，我们希望每一行关注语义的一个方面，但KL散度不能做到这一点。
- 为此我们引入一个新的惩罚项：
$$P=\|(AA^\top-I)\|_F^2$$
其中 $\|\cdot\|_F$ 是矩阵的Frobenius norm。与L2正则项类似，我们对其加上权重系数 $\lambda$ 然后与原始loss合一起来优化。


## 7.5 2018-SGM
- 多标签分类的序列生成模型</br>
(SGM: Sequence Generation Model for Multi-Label Classification)

### 7.5.1 模型概述
- 给定 $L$ 个标签 $\mathcal{L}=\{l_1,l_2,\cdots,l_L\}$，包含 $m$ 个单词的文本序列 $x$。任务是指定一个包含 $n$ 个标签的 $\mathcal{L}$ 的子集 $y$。
- 与传统的单标签分类中每个样本只分配一个标签不同，多标签分类(MLC)任务中的每个样本可以有多个标签。从序列生成的角度来看，MLC任务可以通过最大化条件概率 $p(y|x)$ 来寻找最优标签序列 $y^*$，其计算公式如下：
$$p(y|x)=\prod\limits_{i=1}^np(y_i|y_1,y_2,\cdots,y_{i-1},x)$$
- 模型如图所示：

![](./img/7.5.1SGM.png ':size=100%')
- 首先，我们根据标签在训练集中出现的频率对每个样本的标签序列进行排序。此外，在标签序列的头尾分别添加 $bos$ 和 $eos$ 符号。
- 文本序列 $x$ 被编码到隐藏状态，在时间步 $t$ 通过注意力机制聚合为上下文向量 $c_t$。
- 解码器将上下文向量 $c_t$ 以及解码器的最后一个隐藏状态 $s_{t-1}$ 和嵌入向量 $g(y_{t-1})$ 作为输入，在时间步 $t$ 得到隐藏状态 $s_t$。其中，$y_{t-1}$ 为 $t-1$ 时刻标签空间 $\mathcal{L}$ 上的预测概率分布。
- 函数 $g$ 以 $y_{t-1}$ 为输入，产生嵌入向量并传递给解码器。
- 最后，利用掩码softmax层输出概率分布 $y_t$。

### 7.5.2 序列生成
- 整个序列生成模型由具有注意力机制的编码器和解码器组成。
#### 编码器
- 令 $(w_1,w_2,\cdots,w_m)$ 是具有 $m$ 个单词的句子，其中 $w_i$ 是第 $i$ 个单词的独热表示。
- 我们首先通过一个嵌入矩阵 $E\in\mathbb{R}^{k{\times}|\mathcal{V}|}$ 将 $w_i$ 变为一组稠密嵌入向量 $x_i$，其中 $|\mathcal{V}|$ 是词汇表的大小，$k$ 是嵌入向量的维度。
- 我们使用双向LSTM得到双向特征：
$$\begin{align*}
\mathop{h_i}\limits^{\rightarrow}&=\mathop{\rm{LSTM}}\limits^{\longrightarrow}(\mathop{h_{i-1}}\limits^{\longrightarrow},x_i)\\
\mathop{h_i}\limits^{\leftarrow}&=\mathop{\rm{LSTM}}\limits^{\longleftarrow}(\mathop{h_{i+1}}\limits^{\longleftarrow},x_i)
\end{align*}$$
- 通过concat操作得到最终的特征：
$$h_i=[\mathop{h_i}\limits^{\rightarrow};\mathop{h_i}\limits^{\leftarrow}]$$

#### 注意力机制
- 当模型预测不同的标签时，并不是所有的文本词都做出相同的贡献。注意力机制通过关注文本序列的不同部分并聚合这些信息词的隐藏表示来产生上下文向量。
- 在时间步 $t$ 通过下式为第 $i$ 个单词分配权重 $\alpha_{ti}$：
$$\begin{align*}
e_{ti}&=v_a^{\top}{\rm{tanh}}(W_as_t+U_ah_i)\\
\alpha_{ti}&=\frac{{\rm{exp}}(e_{ti})}{\sum_{j=1}^m{\rm{exp}}(e_{tj})}
\end{align*}$$
其中 $W_a,U_a,v_a$ 是权重参数，$s_t$ 是解码器在时间步 $t$ 的当前隐藏状态。
- 最后传入解码器的文本向量 $c_t$ 为：
$$c_t=\sum\limits_{i=1}^m\alpha_{ti}h_i$$

#### 解码器
- 解码器在时间步 $t$ 的隐藏状态 $s_t$ 计算如下： 
$$s_t={\rm{LSTM}}(s_{t-1},[g(y_{t-1});c_{t-1}])$$
其中 $[g(y_{t-1});c_{t-1}]$ 是指将 $g(y_{t-1})$ 和 $c_{t-1}$ concat起来。
- $g(y_{t-1})$ 是在分布 $y_{t-1}$ 下概率最大的标签嵌入，$y_{t-1}$ 是在时间步 $t-1$ 时的标签空间 $\mathcal{L}$ 上的概率分布，具体由下式计算：
$$\begin{align*}
o_t&=W_of(W_ds_t+V_dc_t)\\
y_t&={\rm{softmax}}(o_t+I_t)
\end{align*}$$
其中 $W_o,W_d,V_d$ 是权重参数，$f$ 是一个非线性激活函。
- $I_t\in\mathbb{R}^L$ 是掩码向量，用于防止解码器预测重复标签数：
$$(I_t)_i=\left\{\begin{array}{ll}
-\infty &\text{if the label } l_i \text{ has been predicted at previous } t-1 \text{ time steps.}\\
0 &\text{otherwise.}
\end{array}\right.$$
- 在训练阶段，损失函数为交叉熵损失函数。我们采用束搜索算法来寻找推断时排名靠前的预测路径。

### 7.5.3 全局嵌入
- $t$ 步的解码器隐藏状态向量是贪心的求得的，通过分布 $y_{t-1}$ 下概率最大的标签嵌入求得。如果在时间步 $t$ 的预测是错误的，那么我们很可能会在接下来的时间步得到一系列错误的标签预测，这也被称为曝光偏差。
- 实际上，前面所有时间步的信息都是有价值的，所以我们应该考虑所有的信息来缓解曝光偏差。
- 基于此，我们提出了一种新的解码器结构。我们引入全局嵌入。
- 令 $e$ 表示在分布 $y_{t-1}$ 下概率最大的标签嵌入，$\bar{e}$ 为 $t$ 时刻的加权平均嵌入：
$$\bar{e}=\sum\limits_{i=1}^Ly_{t-1}^{(i)}e_i$$
其中 $y_{t-1}^{(i)}$ 是 $y_{t-1}$ 的第 $i$ 个元素，$e_i$ 是第 $i$ 个标签的嵌入向量。
- 时间步 $t$ 传入解码器的全局嵌入 $g(y_{t-1})$ 为：
$$g(y_{t-1})=(1-H){\odot}e+H{\odot}\bar{e}$$
其中 $H$ 是控制加权平均嵌入比例的变换门：
$$H=W_1e+W_2\bar{e}$$
其中 $W_1,W_2\in\mathbb{R}^{L{\times}L}$ 是权重矩阵。
- 全局嵌入 $g(y_{t-1})$ 是原始嵌入和使用变换门 $H$ 的加权平均嵌入的优化组合，可以自动确定每个维度上的组合因子。通过考虑每个标签出现的概率，该模型能够减少上一时间步错误预测造成的损失。这使得模型能够更准确地预测标签序列。

## 7.6 2018-ELMo
- 深度语境化词语表示</br>
(Deep contextualized word representations)
- ELMo(Embeddings from Language Models)词表示是整个输入句子的函数。我们通过4个部分来介绍。
### 7.6.1 双向语言模型
- 给定 $N$ 个tokens的序列 $(t_1,t_2,\cdots,t_N)$，前向语言模型通过建模 $t_k$ 的概率来计算序列的概率：
$$p(t_1,t_2,\cdots,t_N)=\prod\limits_{k=1}^Np(t_k|t_1,t_2,\cdots,t_{k-1})$$
- 最近的神经语言模型，计算一个与上下文无关的token表示 $x_k^{LM}$，然后放入 $L$ 层前向LSTMs网络中。在每个位置 $k$，每个LSTM层输出一个上下文独立的表示 $\mathop{h_{k,j}^{LM}}\limits^{\longrightarrow}$，其中 $j=1,\cdots,L$。最上层的输出放入softmax分类器中来预测下一个token $t_{k+1}$。
- 一个后向LM与前向类似，只是逆序文本序列，通过未来的文本预测之前的token：
$$p(t_1,t_2,\cdots,t_N)=\prod\limits_{k=1}^Np(t_k|t_{k+1},t_{k+2},\cdots,t_N)$$
- 双向语言模型通过联合前向和后向过程，通过最大化对数似然来训练：
$$\sum\limits_{k=1}^N({\rm{log}}p(t_k|t_1,\cdots,t_{k-1};\Theta_x,{\mathop{\Theta}\limits^{\rightarrow}}_{\rm{LSTM}},\Theta_s)+{\rm{log}}p(t_k|t_{k+1},\cdots,t_N;\Theta_x,{\mathop{\Theta}\limits^{\leftarrow}}_{\rm{LSTM}},\Theta_s))$$
其中 $\Theta_x$ 是token表示的参数，$\Theta_s$ 是softmax层参数，我们在前向后向过程中共享这两部分参数。

### 7.6.2 ELMo
- 对于每个token $t_k$，一个 $L$ 层biLM计算了 $2L+1$ 个表示：
$$\begin{align*}
R_k&=\{x_k^{LM},\mathop{h_{k,j}^{LM}}\limits^{\longrightarrow},\mathop{h_{k,j}^{LM}}\limits^{\longleftarrow}|j=1,\cdots,L\}\\
&=\{h_{k,j}^{LM}|j=0,\cdots,L\}
\end{align*}$$
其中 $h_{k,j}^{LM}=[\mathop{h_{k,j}^{LM}}\limits^{\longrightarrow};\mathop{h_{k,j}^{LM}}\limits^{\longleftarrow}]$。
- ELMo将这些层的表示变为一个向量：
$${\rm{ELMo}}_k=E(R_k;\Theta_c)$$
    - 最简单的情况，是用最后一层的输出来作为向量：$E(R_k)=h_{k,L}^{LM}$。
    - 更一般的，我们计算所有biLM层的特定任务权重：
    $${\rm{ELMo}}_k^{task}=E(R_k;\Theta^{task})=\gamma^{task}\sum\limits_{j=0}^Ls_j^{task}h_{k,j}^{LM}$$
    其中，$s^{task}$ 是softmax-normalized权重，$\gamma^{task}$ 是标量参数，允许模型对整个ELMo向量进行缩放。

### 7.6.3 使用biLMs进行有监督的NLP任务
- 给定一个预训练的biLM和一个目标NLP任务的监督架构，使用biLM改进任务模型是一个简单的过程。我们运行biLM并记录每个单词的所有层表示。然后，我们让任务模型学习这些表示的线性组合。
- 首先考虑监督模型的最低层。多数监督NLP模型在最低层有一个共同的体系结构，使得我们能以一个统一的方式添加ELMo。
- 给定tokens序列 $(t_1,\cdots,t_N)$，使用预训练的词嵌入和可选的基于字符的表示为每个token形成一个上下文无关的token表示 $x_k$。然后，使用双向RNNs、CNNs或前馈网络形成上下文敏感的表示 $h_k$。
- 为了将ELMo添加到监督模型中，我们首先冻结biLM的权重，然后将ELMo向量与 $x_k$ concat起来得到 $[x_k;{\rm{ELMo}}_k^{task}]$，再放入任务模型中。
- 另一些任务中，我们在任务模型后引入ELMo，将 $h_k$ 替换为 $[h_k;{\rm{ELMo}}_k^{task}]$。
- 最后，Dropout操作是有效的。在损失中添加正则项 $\lambda\|w\|_2^2$ 也是有效的。

### 7.6.4 预训练的双向语言模型架构
- 最终模型使用 $L=2$ 个biLSTM层，包含4096个单元和512维投影，以及从第一层到第二层的残差连接。上下文不敏感的表示使用2048个字符的n-gram卷积滤波器，然后使用两个highway层和一个线性投影层，输出一个512维的表示。
- 在1B的单词上训练10个epochs后，前向和后向平均困惑度为39.7，而前向CNN-BIG-LSTM为30.0。总体而言，我们发现前后向困惑度近似相等，后向值略低。
- 预训练后，biLM可以为任何任务计算表示。某些情况下，在特定领域的数据上微调biLM会导致困惑度的显著下降以及下游任务性能的增加。

## 7.7 2018-BiBloSA

## 7.8 2019-AttentionXML

## 7.9 2019-HAPN

## 7.10 2019-Proto-HATT

## 7.11 2019-STCKA