# NLP常用工具包学习
## 2.1 nltk API部分
- 工具包安装及下载
```python
!pip install nltk
import nltk
# 选择需要的工具包进行下载
nltk.download()
```
- 分词操作
```python
from nltk.tokenize import word_tokenize
input_str = "Today's weather is good."
tokens = word_tokenize(input_str)                           # 进行英文分词
tokens = [word.lower() for word in tokens]                  # 大写转换为小写
```
- 创建 `Text` 对象
```python
from nltk.text import Text
text_obj = Text(tokens)                                     # 创建Text对象
text_obj.count(word)                                        # word单词的计数
text_obj.index(word)                                        # word单词的索引
text_obj.plot(n)                                            # 前n多的单词进行画图展示
```
- 停用词
```python
from nltk.corpus import stopwords
stopwords.fileids()                                         # 查看停用词表支持语言种类
stopwords.raw('english')                                    # 查看英文停用词表
tokens_set = set(tokens)                                    # 通过set函数去重
st_tokens = tokens_set.intersection(set(stopwords.words('english')))    # 查看tokens集合与英文停用词集合的交集
# 过滤停用词
filtered = [w for w in tokens_set if w not in st_tokens]
```
- 词性标注
```python
from nltk import pos_tag
tags = pos_tag(tokens)                                      # 词性标注，词性符号意义见相关文档
```
- 句子分块
```python
from nltk.chunk import RegexpParser
sentence = [('the', 'DT'), ('little', 'JJ'), ('yellow', 'JJ'), ('dog', 'NN'), ('died', 'VBD')]
grammer = 'MY_NP: {<DT>?<JJ>*<NN>}'                         # 设置分块的规则
cp = RegexpParser(grammer)                                  # 生成规则
result = cp.parse(sentence)                                 # 解析句子，进行分块
result.draw()                                               # 调用matplotlib进行画图
```
- 命名实体识别
```python
from nltk import ne_chunk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
sentence = "Edison went to Tsinghua University today"
# 首先对输入进行分词，然后标注词性，最后进行命名实体识别
ne_chunk(pos_tag(word_tokenize(sentence)))
```
- 构建数据清洗函数
```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# 举例说明
s = '   RT @Amila #Test\nTom\'s newly listed Co  &amp; Mary\'s unlisted     Group to supply tech for nlTK.\nh $TSLA $AAPL https://t.co/x34afsfQsh'
def text_clean(text):
    print('原始数据：', text, '\n')
    # 去掉HTML标签(e.g. &amp;)
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', text)
    print('去掉特殊标签后的:', text_no_special_entities, '\n')
    # 去掉一些价值符号
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    print('去掉价值符号后的:', text_no_tickers, '\n')
    # 去掉超链接
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    print('去掉超链接后的:', text_no_hyperlinks, '\n')
    # 去掉一些专门名词缩写，这里我们简单定义为字母比较少的词
    text_no_small_words = re.sub(r'\b\w{1,2}\b', '', text_no_hyperlinks)
    print('去掉专门名词缩写后:', text_no_small_words, '\n')
    # 去掉多余的空格
    text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_small_words)
    text_no_whitespace = text_no_whitespace.lstrip(' ')
    print('去掉空格后:', text_no_whitespace, '\n')
    # 分词
    tokens = word_tokenize(text_no_whitespace)
    print('分词结果:', tokens, '\n')
    # 去停用词
    list_no_stopwords = [i for i in tokens if i not in stopwords.words('english')]
    print('去停用词后结果:', list_no_stopwords, '\n')
    # 过滤后结果
    text_filtered = ' '.join(list_no_stopwords)
    print('过滤后:', text_filtered)
text_clean(s)
```

## 2.2 spaCy API部分
[官方文档](https://spacy.io/api)
- 工具包安装及英文模型下载
```powershell
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
- 导入工具包和英文模型
```python
import spacy
nlp = spacy.load('en_core_web_sm')
text = 'Weather is good, very windy and sunny. We have no classes in the afternoon.'
```
- 文本处理
```python
doc = nlp(text)
# 分词
for token in doc:
    print(token)
# 分句
for sent in doc.sents:
    print(sent)
# 词性 .pos_
for token in doc:
    print('{}-{}'.format(token, token.pos_))
```
[词性符号对照表](https://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/)
- 命名实体识别
```python
doc_2 = nlp("I went to Paris where I met my old friend Jack from uni")
# 命名实体 .ents
for ent in doc_2.ents:
    print('{}-{}'.format(ent, ent.label_))
```
```jupyter-notebook
from spacy import displacy
displacy.render(doc_2, style='ent', jupyter=True)
```