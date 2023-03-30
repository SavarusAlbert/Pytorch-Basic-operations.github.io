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

## 2.3 FastText
#### word-level Model
- 基于word单词作为基本单位。能够很好的对词库中每一个词进行向量表示。
    - 问题描述：容易出现单词不存在于词汇库中的情况
    - 解决方法：最佳语料规模，使系统能够获得更多的词汇量
    - 问题描述：如果遇到了不正式的拼写, 系统很难进行处理
    - 解决方法：矫正或加规则约束
#### Character-Level Model
- 基于 Character 作为基本单位。能够很好的对字库中每一个 Char 进行向量表示。
    - 问题描述：输入句子变长，使得数据变得稀疏，而且对于远距离的依赖难以学到，训练速度降低
    - 解决方法：利用多层卷积、池化、残差等方法
#### Subword Model
介于 word-level Model 和 Character-level 之间的 Model。
- BPE：将经常出现的byte pair用一个新的byte来代替，构成新的文本库。
- sentencepiece model：将词间的空白也当成一种标记，可以直接处理sentence，而不需要将其pre-tokenize成单词。
#### Hybrid Model
- 混合模型，对 word-level 和 character-level 进行加权求和。
#### FastText
- n-gram：将每个 word 表示成 bag of character n-gram 以及单词本身的集合，例如对于where这个单词和n=3的情况，它可以表示为 <wh,whe,her,ere,re>。
- 模型结构：
    - 输入层：输入单词的 n-gram 特征
    - 隐藏层：对多个词向量加权平均，得到 center word $w$ 与 context word $c$ 的分数：
    $$s(w,c)=\sum\limits_{g{\in}G_w}\vec{z_g}^{\top}\vec{v_c}$$
    其中 $G_w$ 是单词 $w$ 的 n-gram 集合。
    - 输出层：softmax 多分类。当类别过多时，可以用Hierarchical softmax 降低算法复杂度。

![](./img/2.1FastText.png ':size=80%')

- 算法代码
```python
import fasttext
# 训练模型
classifier = fasttext.train_supervised(input, lr=0.1, dim=100, 
                   ws=5, epoch=5, minCount=1, 
                   minCountLabel=0, minn=0, 
                   maxn=0, neg=5, wordNgrams=1, 
                   loss="softmax", bucket=2000000, 
                   thread=12, lrUpdateRate=100,
                   t=1e-4, label="__label__", 
                   verbose=2, pretrainedVectors="")
# 保存模型
classifier.save_model('./model/fasttext.bin')
# 模型预测
labels = classifier.predict(texts)
# 模型加载
model = fasttext.load_model(path)
```