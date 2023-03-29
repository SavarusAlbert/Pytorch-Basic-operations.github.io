# 词嵌入
word embedding
## 2.1 Word2Vec
- Word2vec是一种学习词向量的框架。包括CBOW和skip-gram两类，CBOW通过上下文词来预测中心词，skip-gram通过中心词预测上下文词。
- 对于每个位置 $t=1,\cdots,T$，固定窗口大小 $m$，给定中心词 $w_j$，句子的概率为：
$${\rm{likelihood}}=L(\theta)=\prod\limits_{t=1}^T\prod\limits_{-m{\leq}j{\leq}m\ j{\neq}0}P(w_{t+j}|w_t;\theta)$$
简单变换得到需要优化的函数：
$$J(\theta)=-\frac{1}{T}{\rm{log}}L(\theta)=-\frac{1}{T}\sum\limits_{t=1}^T\sum\limits_{-m{\leq}j{\leq}m\ j{\neq}0}{\rm{log}}P(w_{t+j}|w_t;\theta)$$
- 两个词向量的点积能够表明单词之间的相似度：
$$u^{\top}v=u{\cdot}v=\sum\limits_{i-1}^nu_iv_i$$
- 通过softmax将其转换为概率分布，我们可以得到单词 $o$(outside上下文词)在单词 $c$(center中心词)条件下发生的概率：
$$P(o|c)=\frac{exp(u_o^{\top}v_c)}{\sum_{w{\epsilon}V}exp(u_w^{\top}v_c)}$$
- 然后通过随机梯度下降来优化概率 $P(o|c)$ 直到概率趋近真实值。
### 2.1.1 算法代码
```python
import gensim
from gensim.models import Word2Vec
# 导入数据集，通过正则表达式进行分词，得到词的列表lines
# 通过gensim.models.Word2Vec构建模型，向量维度为vector_size，前后窗口大小为window，最低词频为min_count，模型训练迭代次数为epochs，负采样negative
model = Word2Vec(lines, vector_size=20, window=2, min_count=3, epochs=7, negative=10)
# 输出词的词向量
model.wv.get_vector('word')
# 输出相似度最高的20个词
model.wv.most_similar('word', topn=20)
```

## 2.2 GloVe
#### 构建共现矩阵
- 共现矩阵：由语料库中所有不重复单词构成矩阵以存储单词的共现次数。
- 构建共现矩阵 $X$ 的方式一般有：全局共现；在设定好的窗口大小内部的共现。共现矩阵维度较高，可以通过SVD进行矩阵分解降维，或通过PCA主成分分析进行降维。
- GloVe共现矩阵采用另一种方式，引入衰减系数来代替窗口大小的限制，距离越远的单词会有更小的权重系数。
- 上下文词 $k$ 在中心词 $i$ 出现时的出现概率为：
$$P_{ik}=P(k|i)=\frac{X_{ik}}{X_i}$$
#### 词语类比
- 通过概率之间的比值，能够表达和同一个词都相关或都不相关，因此考虑构建一个函数来表示比值关系：
$$F(w_i,w_j,\tilde{w}_k)=\frac{P_{ik}}{P_{jk}}$$
为了保持对称性(词语类比具有对称关系)，通过下式构建：
$$F((w_i-w_j)^{\top}\tilde{w}_k)=\frac{F(w_i^{\top}\tilde{w}_k)}{F(w_j^{\top}\tilde{w}_k)}$$
- 由上式以及概率函数的定义，我们可以通过下式来定义 $F$ 函数：
$$F(w_i^{\top}\tilde{w}_k)=P_{ik}=\frac{X_{ik}}{X_i}$$
- 令 $F={\rm{exp}}$，则有：
$$w_i^{\top}\tilde{w}_k={\rm{log}}(P_{ik})={\rm{log}}(X_{ik})-{\rm{log}}(X_i)$$
- 为了满足对称性，我们引入偏置项 $\tilde{b}_k$ 并用 $b_i$ 来代替 ${\rm{log}}(X_i)$，最终得到：
$$w_i^{\top}\tilde{w}_k+b_i+\tilde{b}_k={\rm{log}}(X_{ik})$$
#### 损失函数
- 我们知道在一个语料库中，肯定存在很多单词他们在一起出现的次数是很多的，那么我们希望：
    - 这些单词的权重要大于那些很少在一起出现的单词，因此这个函数要是非递减函数；
    - 但这个权重也不能过大，当到达一定程度之后当不再增加；
    - 如果两个单词没有在一起出现，也就是 $X_{ij}$，那么他们应该不参与到loss function的计算当中去，也就是 $f(x)$ 要满足 $f(x)=0$。
- 为此，作者提出了以下权重函数：
$$f(x)=\left\{\begin{array}{ll}
(x/x_{\rm{max}})^{\alpha} & {\rm{if}}x<x_{\rm{max}}\\
1 & {\rm{ otherwise}}
\end{array}\right.$$
- 增加了上述权重函数后，构建损失函数：
$$J=\sum\limits_{i,j=1}^{V}f\left(X_{i j}\right)\left(w_{i}^{T}\tilde{w}_{j}+b_{i}+\tilde{b}_{j}-{\rm{log}}X_{ij}\right)^2$$



## Polyglot
## Senna
## SSKIP