# 基础代码部分
## 1.1 设置模型参数
- `argparse` 模块：命令行选项、参数和子命令解析器。能够将代码和参数分离开来，使代码更简洁，适用性更强，另外，它能够自动生成帮助文档，当用户输入无效的参数时能给出错误信息。
- 具体见 [argparse文档](https://docs.python.org/zh-cn/3/library/argparse.html#argumentparser-objects)
```python
class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True)
```
- `argparse.ArgumentParser` 用法
```python
import argparse
# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning Rate')
# 解析参数，将参数字符串转换为对象并将其设为命名空间的属性，返回带有成员的命名空间。
params = parser.parse_args()                                # Namespace(lr=0.001)
# 调用对象方法
params.lr                                                   # 0.001
# 参数赋值
params.lr = 0.1
print(params.lr)                                            # 0.1
```

## 1.2 分词
### 1.2.1 基于字符串匹配的分词方法
- 正向最大匹配法：首先给定一个最大的词条长度，假设为max_num=3，首先取出句子的前3个字符，看前3个字符是否存在于词库中，如果存在，则返回第一个分词，滑动窗口向后滑动3个位置；如果不存在，我们把滑动窗口从右向左缩小1，判断前两个字符是否存在于词库。以此类推，直至遍历完整个句子。
```python
corpus = ['我们', '我', '今天', '特别', '特别想', '非常', '非常想', '吃', '芒果']
sentence = '我今天特别想吃芒果啊'
max_num = 3
def word_segment1(sentence, corpus, max_num):
    result = []
    start = 0
    end = min(start + max_num,len(sentence))
    while(start!=len(sentence)):
        if start==end:
            result.append(sentence[start])
            if sentence[start] not in corpus:
                corpus.append(sentence[start])  
                # 假如词库中找不到单词，则返回这个单词，并往词库中添加；看具体的场景，如果词库中没有这个词则也可以直接返回空值，表示分词失败
            start += 1
            end = min(start + max_num, len(sentence))
        if sentence[start:end] in corpus:
            result.append(sentence[start:end])
            start += len(result[-1])
            end = min(start + max_num,len(sentence))
        else:
            end -= 1
    return corpus, result
newcorpus, result = word_segment1(sentence,corpus,max_num)
print(newcorpus)
print(result)
# ['我们', '我', '今天', '特别', '特别想', '非常', '非常想', '吃', '芒果', '啊']
# ['我', '今天', '特别想', '吃', '芒果', '啊']
```
- 逆向最大匹配法：与正向相反，窗口从后向前滑。
```python
def word_segment2(sentence, corpus, max_num):
    result = []
    end = len(sentence)
    start = min(end - max_num, 0)
    while(end != 0):
        if start==end:
            result.append(sentence[start])
            if sentence[start] not in corpus:
                corpus.append(sentence[start])
            end -=1
            start = min(end - max_num, 0)
        if sentence[start:end] in corpus:
            result.append(sentence[start:end])
            end -= len(result[-1])
            start = min(end - max_num, 0)
        else:
            start +=1
    return corpus, result[::-1]
newcorpus, result = word_segment2(sentence,corpus,max_num)
print(newcorpus)
print(result)
# ['我们', '我', '今天', '特别', '特别想', '非常', '非常想', '吃', '芒果', '啊']
# ['我', '今天', '特别想', '吃', '芒果', '啊']
```
- 最少切分：使每一句中切出的词数最小。可以使用动态规划，划分成局部问题。
- 双向最大匹配法，将正向逆向匹配算法进行比较：
    - 如果词数不同，取分词数量较少的作为结果；
    - 如果词数相同：
        - 分词结果相同，说明没有歧义，返回任意一个；
        - 分词结果不同，返回单字较少的那个。

### 1.2.2 基于统计的分词方法
#### N-gram分词
- 自定义函数提取N-gram
```python
import re
def n_gram_extractor(sentence, n):
    tokens = clean_text(sentence).split()                                               # 清洗文本并分割，clean_text函数在后续数据清洗章节定义
    n_gram = []
    for i in range(len(tokens)-n+1):
        n_gram.append(tokens[i:i+n])
    return n_gram
```
- 使用 `NLTK` 模块提取N-gram
```python
from nltk import ngrams
list(ngrams(sentence, n))
```
- 使用 `TextBlob` 模块提取N-gram
```python
from textblob import TextBlob
blob = TextBlob(sentence)
blob.ngrams(n)
```

#### 隐马尔可夫模型(HMM)
- HMM联合概率函数为：
$$P(O_1,\cdots,O_T,X_1,\cdots,X_T)=P(X_1)P(O_1|X_1)\prod\limits_{t=2}^TP(X_t|X_{t-1})P(O_t|X_t)$$
其中 $X_t$ 是 $t$ 时刻的状态，$O_t$ 是 $t$ 时刻的观测值。
- HMM要满足三大假设：
    - 马尔可夫假设：$P(X_t|X_{t-1}X_{t-2}{\cdots}X_1)=P(X_t|X_{t-1})$
    - 齐次性假设：$P(X_t|X_{t-1})=P(X_s|X_{s-1})$
    - 观察独立性假设：$P(O_t|X_tX_{t-1}{\cdots}X_1,O_{t-1}O_{t-2}{\cdots}O_1)=P(O_t|X_t)$
```python
import numpy as np
import os
from tqdm import tqdm
import pickle
class HMM():
    def __init__(self,file_text,file_state):
        self.all_texts = open(file_text, 'r', encoding='utf-8').read().split('\n')      # 按行获取所有文本
        self.all_states = open(file_state, 'r', encoding='utf-8').read().split('\n')    # 按行获取所有状态
        self.states_to_index = {'B': 0, 'M': 1, 'S': 2, 'E': 3}                         # 每个状态定义一个索引，可以通过状态获取索引
        self.index_to_states = ['B', 'M', 'S', 'E']                                     # 通过索引获取状态
        self.len_states = len(self.states_to_index)                                     # 状态长度
    
        self.init_matrix = np.zeros((self.len_states))                                  # 初始化初始概率矩阵 P(X_1)
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))             # 初始化状态转移矩阵 P(X_t|X_t-1)
        self.emit_matrix = {'B': {'total':0}, 'M': {'total':0}, 
                            'S': {'total':0}, 'E': {'total':0}}                         # 初始化发射概率矩阵 P(O_t|X_t)
    
    def cal_init_matrix(self, state):
        self.init_matrix[self.states_to_index[state[0]]] += 1                           # BMSE 四种状态，对应状态出现1次就 +1
    
    def cal_transfer_matrix(self, states):
        sta_join = ''.join(states)                                                      # 状态转移，从当前状态转移到后一状态，即从sta1转移到sta2
        sta1 = sta_join[:-1]
        sta2 = sta_join[1:]
        for s1, s2 in zip(sta1, sta2):
            self.transfer_matrix[self.states_to_index[s1],self.states_to_index[s2]] += 1
    
    def cal_emit_matrix(self, words, states):
        for word, state in zip(''.join(words), ''.join(states)):
            self.emit_matrix[state][word] = self.emit_matrix[state].get(word,0) + 1     # 状态的次数 +1
            self.emit_matrix[state]['total'] += 1                                       # 总次数 +1
    
    def normalize(self):
        self.init_matrix = self.init_matrix/np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix/np.sum(self.transfer_matrix,axis=1,keepdims=True)
        self.emit_matrix = {state:{word:t/word_times['total']*1000 for word,t in word_times.items() if word != 'total'} for state, word_times in self.emit_matrix.items()}

    def train(self):
        if os.path.exists('three_matrix.pkl'):                                          # 如果存在参数就不需要训练了
            self.init_matrix, self.transfer_matrix, self.emit_matrix = pickle.load(open('three_matrix.pkl','rb'))
            return
        for words, states in tqdm(zip(self.all_texts, self.all_states)):
            words = words.split(' ')
            states = states.split(' ')
            self.cal_init_matrix(states[0])
            self.cal_transfer_matrix(states)
            self.cal_emit_matrix(words, states)
        self.normalize()
        pickle.dump([self.init_matrix, self.transfer_matrix, self.emit_matrix], open('three_matrix.pkl', 'wb'))
# viterbi算法进行推断，代码略
```

#### 条件随机场模型(CRF)
- 4-tag CRF标注：B(Begin，词首), E(End，词尾), M(Middle，词中), S(Single,单字词)
```python
# 定义标注函数，将形如 "人们 常 说 生活 是 一 部 教科书" 的句子进行4-tag标注
def character_tagging(input_file, output_file):
    input_data = open(input_file, 'r', encoding='utf-8')
    output_data = open(output_file, 'w', encoding='utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "\tS\n")
            else:
                output_data.write(word[0] + "\tB\n")
                for w in word[1:len(word)-1]:
                    output_data.write(w + "\tM\n")
                output_data.write(word[len(word)-1] + "\tE\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()
```
- 使用 [CRF++](http://taku910.github.io/crfpp/) 工具包训练

### 1.2.3 分词常用评价指标
- 正负样本预测情况：
    - True Positive(TP)：表示将正样本预测为正样本，即预测正确；
    - False Positive(FP)：表示将负样本预测为正样本，即预测错误；
    - False Negative(FN)：表示将正样本预测为负样本，即预测错误；
    - True Negative(TN)：表示将负样本预测为负样本，即预测正确
- 正确率：
$${\rm{Accuracy}}=\frac{TP+TN}{TP+FP+FN+TN}$$
- 精确率：
$${\rm{Precision}}=\frac{TP}{TP+FP}$$
- 召回率：
$${\rm{Recall}}=\frac{TP}{TP+FN}$$
- F-score：
$${\rm{F}}_{\rm{score}}=(1+\beta^2)\frac{{\rm{Precision}}\cdot{\rm{Recall}}}{\beta^2\cdot{\rm{Precision}}+{\rm{Recall}}}$$
当 $\beta=1$ 时称为 ${\rm{F}}_1$ 值。


## 1.3 数据清洗
### 1.3.1 Normalization
- 文本大小写转换
```python
text = text.lower()
```
- 移除不需要的特殊符号
```python
import re
def clean_text(text):
    # 根据需求去掉特殊符号
    return re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", " ", text)
```

### 1.3.2 Tokenization
- 最简单的方法是，通过 `split()` 方法返回词列表
```python
# 默认按照空格拆分，可根据需求更改
words = text.split()
```
- `nltk` 库进行拆分文本(比 `split()` 方法更优化，能够根据标点符号位置进行不同处理)
```python
from nltk.tokenize import word_tokenize
words = word_tokenize(text)
```

### 1.3.3 Stop Word 停止词
- 根据需求设置语料库的停止词(无意义词)，`nltk` 库有英语常用停用词
```python
from nltk.corpus import stopwords
stopwords.words('english')
```
- 过滤停用词
```python
words = [w for w in words if w not in stopwords.words('english')]
```

### 1.3.4 Part-of-Speech Tagging 词性标注
- 使用 `nltk` 库进行词性标注
```python
from nltk import pos_tag
pos_tag(words)
```

### 1.3.5 Named Entity Recognition 名词实体识别
- Named Entity 一般是名词短语，又来指代某些特定对象、人、或地点 可以使用 `ne_chunk()` 方法标注文本中的命名实体。在进行这一步前，必须先进行 `Tokenization` 并进行 `PoS Tagging`。
```python
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
ne_chunk(pos_tag(word_tokenize(words)))
```

### 1.3.6 Stemming and Lemmatization 词根还原
- 通过 `PorterStemmer()` 方法还原词干词根
```python
from nltk.stem.porter import PorterStemmer
stemmed = [PorterStemmer().stem(w) for w in words]
```
- 通过 `WordNetLemmatizer()` 方法，利用词典将词还原成标准化形式(需要字典，对内存要求比Stemming方法高)
```python
from nltk.stem.wordnet import WordNetLemmatizer
lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]                     # pos参数可以指定标准化形式的词性
```

### 其他操作
- Rare word replacement：去除词频小的单词
- 添加 `<BOS>`, `<EOS>` 位置编码
- Long Sentence Cut-Off：长句子截断

## 1.4 统计特征
### 1.4.1 TF-IDF
- TF(Term Frequency)是词频，表示词条在文本中出现的频率：
$${\rm{TF}}_w=\frac{\text{在某类中词条}{w}\text{出现的次数}}{\text{该类中所有的词条数目}}$$
- IDF(Inverse Document Frequency)是逆文本频率，IDF越大，包含词条的文档越少，则说明词条具有很好的类别区分能力：
$${\rm{IDF}}_w={\rm{log}}\frac{\text{语料库文档总数}}{\text{包含词条}{w}\text{的文档数}+1}$$
- TF-IDF，某一特定文件内的高频词，该词在整个文件集合中有低文件频率。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语：
$${\rm{TF}}\text{-}{\rm{IDF}}={\rm{TF}}\times{\rm{IDF}}$$
- 简单实现
```python
from collections import defaultdict
import math
import operator
def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1
    # 计算每个词的TF值
    word_tf = {}
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i]/sum(doc_frequency.values())
    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}
    word_doc = defaultdict(int)
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num/(word_doc[i]+1))
    # 计算每个词的TF-IDF值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]
    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select
```
- `nltk` 实现
```python
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
# 根据text构建语料库
corpus = TextCollection(text)
# 计算单词word的TF、IDF、TF-IDF
tf = corpus.tf(word, corpus)
idf = corpus.idf(word)
tf_idf = corpus.tf_idf(word, corpus)
```
- `sklearn` 实现
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# 将文本中的词语转换为词频矩阵，矩阵元素a[i][j]，表示j词在i类文本下的词频
vectorizer = CountVectorizer()
# 通过fit_transform计算词语出现次数
word_times = vectorizer.fit_transform(text)
# 统计每个词语的tf-idf权值
tf_idf_transformer = TfidfTransformer()
# 计算词频矩阵的tf-idf
tf_idf = tf_idf_transformer.fit_transform(word_times)
# 转换为矩阵进行输出
tf_idf.toarray()
```
- `Jieba` 实现
```python
import jieba.analyse
# 输出文本text中的前5个td-idf重要单词
keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=False, allowPOS=())
```