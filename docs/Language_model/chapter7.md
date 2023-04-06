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


## 7.3 2016-LSTMN