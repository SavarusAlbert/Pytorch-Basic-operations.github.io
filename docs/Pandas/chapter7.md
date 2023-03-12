# Chapter07 文本数据
## 7.1 str对象
- `str` 对象是定义在 `Index` 或 `Series` 上的属性，专门用于处理每个元素的文本内容，其内部定义了大量方法，因此对一个序列进行文本处理，首先需要获取其 `str` 对象。
- 在Python标准库中也有 `str` 模块，为了使用上的便利，有许多函数的用法 `pandas` 照搬了它的设计。
### 7.1.1 `[]`索引器
- 对 `str` 对象使用 `[]` 索引器，可以取出某个位置的元素，也能通过切片得到子串，如果超出范围则返回缺失值。
```python
import numpy as np
import pandas as pd
s = pd.Series(['abcd', 'efg', 'hi'])
```
```markup
0    abcd
1     efg
2      hi
dtype: object
```
```python
s.str[-1: 0: -2]
```
```markup
0    db
1     g
2     i
dtype: object
```
```python
s.str[2]
```
```markup
0      c
1      g
2    NaN
dtype: object
```
### 7.1.2 string类型
- 绝大多数对于 `object` 和 `string` 类型的序列使用 `str` 对象方法产生的结果是一致。但对于一个可迭代对象， `string` 类型的 `str` 对象和 `object` 类型的 `str` 对象返回结果可能是不同的。
```python
s = pd.Series([{1: 'temp_1', 2: 'temp_2'}, ['a', 'b'], 0.5, 'my_string'])
s.str[1]                                                    # 对可迭代对象做迭代
```
```markup
0    temp_1
1         b
2       NaN
3         y
dtype: object
```
```python
s.astype('string').str[1]                                   # 将对象变为字符串后再对位置迭代
```
```markup
0    1
1    '
2    .
3    y
dtype: string
```
- `string` 类型是 `Nullable` 类型，但 `object` 不是。这意味着 `string` 类型的序列，如果调用的 `str` 方法返回值为整数 `Series` 和布尔 `Series` 时，其分别对应的 `dtype` 是 `Int` 和 `boolean` 的 `Nullable` 类型，而 `object` 类型则会分别返回 `int/float` 和 `bool/object` ，取决于缺失值的存在与否。同时，字符串的比较操作，也具有相似的特性， `string` 返回 `Nullable` 类型，但 `object` 不会。
- 最后需要注意的是，对于全体元素为数值类型的序列，即使其类型为 `object` 或者 `category` 也不允许直接使用 `str` 属性。

## 7.2 文本处理的五类操作
### 7.2.1 拆分
- `str.split` 能够把字符串的列进行拆分，其中第一个参数为正则表达式，可选参数包括从左到右的最大拆分次数 `n` ，是否展开为多个列 `expand` 。
```python
s = pd.Series(['上海市黄浦区方浜中路249号',
            '上海市宝山区密山路5号'])
s.str.split('[市区路]')
```
```markup
0    [上海, 黄浦, 方浜中, 249号]
1       [上海, 宝山, 密山, 5号]
dtype: object
```
```python
s.str.split('[市区路]', n=2, expand=True)
```
```markup
    0   1         2
0  上海  黄浦  方浜中路249号
1  上海  宝山     密山路5号
```
### 7.2.2 合并
- 关于合并一共有两个函数，分别是 `str.join` 和 `str.cat` 。 
- `str.join` 表示用某个连接符把 `Series` 中的字符串列表连接起来，如果列表中出现了非字符串元素则返回缺失值。
```python
s = pd.Series([['a','b'], [1, 'a'], [['a', 'b'], 'c']])
s.str.join('-')
```
```markup
0    a-b
1    NaN
2    NaN
dtype: object
```
- `str.cat` 用于合并两个序列，主要参数为连接符 `sep` 、连接形式 `join` 以及缺失值替代符号 `na_rep` ，其中连接形式默认为以索引为键的左连接。
```python
s1 = pd.Series(['a','b'])
s2 = pd.Series(['cat','dog'])
s1.str.cat(s2,sep='-')
```
```markup
0    a-cat
1    b-dog
dtype: object
```
```python
s2.index = [1, 2]
s1.str.cat(s2, sep='-', na_rep='?', join='outer')           # 外连接，见第5章
```
```markup
0      a-?
1    b-cat
2    ?-dog
dtype: object
```
### 7.2.3 匹配
- `str.contains` 返回了每个字符串是否包含正则模式的布尔序列。
```python
s = pd.Series(['my cat', 'he is fat', 'railway station'])
s.str.contains('\s\wat')                                    # 匹配正则表达式
```
```markup
0     True
1     True
2    False
dtype: bool
```
- `str.startswith` 和 `str.endswith` 返回了每个字符串以给定模式为开始和结束的布尔序列，它们都不支持正则表达式。
```python
s.str.startswith('my')
```
```markup
0     True
1    False
2    False
dtype: bool
```
```python
s.str.endswith('t')
```
```markup
0     True
1     True
2    False
dtype: bool
```
- 如果需要用正则表达式来检测开始或结束字符串的模式，可以使用 `str.match` ，其返回了每个字符串起始处是否符合给定正则模式的布尔序列。
```python
s.str.match('m|h')
```
```markup
0     True
1     True
2    False
dtype: bool
```
```python
s.str[::-1].str.match('ta[f|g]|n')                          # 反转后匹配
```
```markup
0    False
1     True
2     True
dtype: bool
```
- `str.find` 和 `str.rfind` ，分别返回从左到右和从右到左第一次匹配的位置的索引，未找到则返回-1。需要注意的是这两个函数不支持正则匹配，只能用于字符子串的匹配：
```python
s = pd.Series(['This is an apple. That is not an apple.'])
s.str.find('apple')
```
```markup
0    11
dtype: int64
```
```python
s.str.rfind('apple')
```
```markup
0    33
dtype: int64
```
### 7.2.4 替换
- `str.replace` 和 `replace` 并不是一个函数，在使用字符串替换时应当使用前者。
```python
s = pd.Series(['a_1_b','c_?'])
s.str.replace('\d|\?', 'new', regex=True)
```
```markup
0    a_new_b
1      c_new
dtype: object
```
- 当需要对不同部分进行有差别的替换时，可以利用 `子组` 的方法，并且此时可以通过传入自定义的替换函数来分别进行处理，注意 `group(k)` 代表匹配到的第 k 个子组(圆括号之间的内容)。
```python
s = pd.Series(['上海市黄浦区方浜中路249号',
               '上海市宝山区密山路5号',
               '北京市昌平区北农路2号'])
pat = '(\w+市)(\w+区)(\w+路)(\d+号)'
def my_func(m):
    return '匹配'+m.group(1)
s.str.replace(pat, my_func, regex=True)
```
```markup
0    匹配上海市
1    匹配上海市
2    匹配北京市
dtype: object
```
### 7.2.5 提取
- 提取既可以认为是一种返回具体元素(而不是布尔值或元素对应的索引位置)的匹配操作，也可以认为是一种特殊的拆分操作。通过 `str.extract` 进行提取。
```python
pat = '(\w+市)(\w+区)(\w+路)(\d+号)'
s.str.extract(pat)
```
```markup
     0      1       2       3
0  上海市  黄浦区  方浜中路  249号
1  上海市  宝山区   密山路    5号
2  北京市  昌平区   北农路    2号
```
- 通过子组的命名，可以直接对新生成 `DataFrame` 的列命名。
```python
pat = '(?P<市名>\w+市)(?P<区名>\w+区)(?P<路名>\w+路)(?P<编号>\d+号)'
s.str.extract(pat)
```
```markup
    市名   区名    路名    编号
0  上海市  黄浦区  方浜中路  249号
1  上海市  宝山区   密山路    5号
2  北京市  昌平区   北农路    2号
```
- `str.extractall` 不同于 `str.extract` 只匹配一次，它会把所有符合条件的模式全部匹配出来，如果存在多个结果，则以多级索引的方式存储。
```python
s = pd.Series(['A135T15,A26S5','B674S2,B25T6'], index = ['my_A','my_B'])
pat = '[A|B](\d+)[T|S](\d+)'
s.str.extractall(pat)
```
```markup
              0   1
     match         
my_A 0      135  15
     1       26   5
my_B 0      674   2
     1       25   6
```
- `str.findall` 的功能类似于 `str.extractall` ，区别在于前者把结果存入列表中，而后者处理为多级索引，每个行只对应一组匹配，而不是把所有匹配组合构成列表。
```python
s.str.findall(pat)
```
```markup
my_A    [(135, 15), (26, 5)]
my_B     [(674, 2), (25, 6)]
dtype: object
```

## 7.3 常用字符串函数
### 7.3.1 字母型函数
- `upper`, `lower`, `title`, `capitalize`, `swapcase` 这五个函数主要用于字母的大小写转化，从下面的例子中就容易领会其功能。
```python
s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
s.str.upper()                                               # 大写
s.str.lower()                                               # 小写
s.str.title()                                               # 每个单词首字母大写
s.str.capitalize()                                          # 句子首字母大写
s.str.swapcase()                                            # 大小写互换
```
### 7.3.2 数值型函数
- 这里着重需要介绍的是 `pd.to_numeric` 方法，它虽然不是 `str` 对象上的方法，但是能够对字符格式的数值进行快速转换和筛选。其主要参数包括 `errors` 和 `downcast` 分别代表了非数值的处理模式和转换类型。其中，对于不能转换为数值的有三种 `errors` 选项， `raise`, `coerce`, `ignore` 分别表示直接报错、设为缺失以及保持原来的字符串。
```python
s = pd.Series(['1', '2.2', '2e', '??', '-2.1', '0'])
pd.to_numeric(s, errors='ignore')
```
```markup
0       1
1     2.2
2      2e
3      ??
4    -2.1
5       0
dtype: object
```
```python
pd.to_numeric(s, errors='coerce')
```
```markup
0    1.0
1    2.2
2    NaN
3    NaN
4   -2.1
5    0.0
dtype: float64
```
- 在数据清洗时，可以利用 `coerce` 的设定，快速查看非数值型的行。
```python
s[pd.to_numeric(s, errors='coerce').isna()]
```
```markup
2    2e
3    ??
dtype: object
```
### 7.3.3 统计型函数
- `count` 和 `len` 的作用分别是返回出现正则模式的次数和字符串的长度。
```python
s = pd.Series(['cat rat fat at', 'get feed sheet heat'])
s.str.count('[r|f]at|ee')
```
```markup
0    2
1    2
dtype: int64
```
```python
s.str.len()
```
```markup
0    14
1    19
dtype: int64
```
### 7.3.4 格式型函数
- 格式型函数主要分为两类，第一种是除空型，第二种是填充型。
- 第一类函数一共有三种，它们分别是 `strip`, `rstrip`, `lstrip` ，分别代表去除两侧空格、右侧空格和左侧空格。这些函数在数据清洗时是有用的，特别是列名含有非法空格的时候。
```python
my_index = pd.Index([' col1', 'col2 ', ' col3 '])
my_index.str.strip().str.len()
# my_index.str.rstrip().str.len()
# my_index.str.lstrip().str.len()
```
```markup
Int64Index([4, 4, 4], dtype='int64')
```
- 对于填充型函数而言， `pad` 是最灵活的，它可以选定字符串长度、填充的方向和填充内容。
```python
s = pd.Series(['a','b','c'])
s.str.pad(5,'left','*')
# s.str.pad(5,'right','*')
# s.str.pad(5,'both','*')
```
```markup
0    ****a
1    ****b
2    ****c
dtype: object
```
- 上述的三种情况可以分别用 `rjust`, `ljust`, `center` 来等效完成，需要注意 `ljust` 是指右侧填充而不是左侧填充。
```python
s.str.rjust(5, '*')
# s.str.ljust(5, '*')
# s.str.center(5, '*')
```
```markup
0    ****a
1    ****b
2    ****c
dtype: object
```
- 在读取 excel 文件时，经常会出现数字前补0的需求，例如证券代码读入的时候会把”000007”作为数值7来处理， `pandas` 中除了可以使用上面的左侧填充函数进行操作之外，还可用 `zfill` 来实现。
```python
s = pd.Series([7, 155, 303000]).astype('string')
s.str.pad(6,'left','0')
```
```markup
0    000007
1    000155
2    303000
dtype: string
```
```python
s.str.rjust(6,'0')
```
```markup
0    000007
1    000155
2    303000
dtype: string
```
```python
s.str.zfill(6)
```
```markup
0    000007
1    000155
2    303000
dtype: string
```