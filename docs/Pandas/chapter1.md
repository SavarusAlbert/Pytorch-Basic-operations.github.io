# Chapter01 Pandas基础
- 导入包
```python
import numpy as np
import pandas as pd
```
## 1.1 文件的读取和写入
### 1.1.1 数据读取
- `pandas` 可以读取`.csv`, `.excel`, `.txt` 文件。
```python
df_csv = pd.read_csv('data/my_csv.csv')                 # 读取.csv文件
df_txt = pd.read_table('data/my_table.txt')             # 读取.txt文件
df_excel = pd.read_excel('data/my_excel.xlsx')          # 读取.xlsx文件
```
- 常用参数：
    - `header=None` 表示第一行不作为列名
    - `index_col` 表示把某一列或几列作为索引
    - `usecols` 表示读取列的集合，默认读取所有的列
    - `parse_dates` 表示需要转化为时间的列
    - `nrows` 表示读取的数据行数
    - `read_table` 有一个分割参数 `sep` ，它使得用户可以自定义分割符号，进行 `txt` 数据的读取，同时需要指定引擎 `engine='python'`
### 1.1.2 数据写入
- `DataFrame.to_csv()` 可以保存为 `.csv` 文件，也可以保存为 `.txt` 文件，通过 `seq`自定义分隔符
- `DataFrame.to_excel()` 可以保存为 `.xlsx` 文件
- 常用参数：
    - `index = False` 不保存索引
```python
df_csv.to_csv('data/my_csv_saved.csv', index=False)
df_excel.to_excel('data/my_excel_saved.xlsx', index=False)
df_txt.to_csv('data/my_txt_saved.txt', sep='\t', index=False)
```
- 如果想要把表格快速转换为 `markdown` 和 `latex` 语言，可以使用 `to_markdown` 和 `to_latex` 函数，此处需要安装 `tabulate` 包

## 1.2 基本数据结构
- `pandas` 中具有两种基本的数据存储结构，存储一维 `values` 的 `Series` 和存储二维 `values` 的 `DataFrame` ，在这两种结构上定义了很多的属性和方法。
### 1.2.1 Series
- `Series` 一般由四个部分组成，分别是序列的值 `data` 、索引 `index` 、存储类型 `dtype` 、序列的名字 `name` 。其中，索引也可以指定它的名字，默认为空。
```python
s = pd.Series(data = [100, 'a', {'dic1':5}],
              index = pd.Index(['id1', 20, 'third'], name='my_idx'),
              dtype = 'object',
              name = 'my_name')
x1 = s.values                                           # array([100, 'a', {'dic1': 5}], dtype=object)
x2 = s.shape                                            # 获取序列长度 (3,)
x3 = s['third']                                         # 索引操作 {'dic1': 5}
```
### 1.2.2 DataFrame
- `DataFrame` 在 `Series` 的基础上增加了列索引，一个数据框可以由二维的 `data` 与行列索引来构造。
```python
# 方法1
data = [[1, 'a', 1.2], [2, 'b', 2.2], [3, 'c', 3.2]]
df = pd.DataFrame(data = data,
                  index = ['row_%d'%i for i in range(3)],
                  columns=['col_0', 'col_1', 'col_2'])
# 方法2
df = pd.DataFrame(data = {'col_0': [1,2,3], 'col_1':list('abc'),
                          'col_2': [1.2, 2.2, 3.2]},
                  index = ['row_%d'%i for i in range(3)])
```
```markdown
       col_0 col_1  col_2
row_0      1     a    1.2
row_1      2     b    2.2
row_2      3     c    3.2
```
- 在 `DataFrame` 中可以用 `[col_name]` 与 `[col_list]` 来取出相应的列与由多个列组成的表，结果分别为 `Series` 和 `DataFrame` 
```python
df['col_0']
```
```markdown 
row_0    1
row_1    2
row_2    3
Name: col_0, dtype: int64
```
```python
df[['col_0', 'col_1']]
```
```markdown 
       col_0 col_1
row_0      1     a
row_1      2     b
row_2      3     c
```
```python
# 将数据转置
df.T
```
```markdown
      row_0 row_1 row_2
col_0     1     2     3
col_1     a     b     c
col_2   1.2   2.2   3.2
```

## 1.3 常用基本函数
- 导入 `learn_pandas.csv` 的虚拟数据集
```python
df = pd.read_csv('data/learn_pandas.csv')
print(df.columns)
```
```markup
Index(['School', 'Grade', 'Name', 'Gender', 'Height', 'Weight', 'Transfer',
       'Test_Number', 'Test_Date', 'Time_Record'],
      dtype='object')
```
- 取数据集的前7列
```python
df = df[df.columns[:7]]
```
### 1.3.1 汇总函数
- `head`, `tail` 函数分别表示返回表或者序列的前 n 行和后 n 行，其中 n 默认为5
```python
df.head(2)
df.tail(3)
```
- `info`, `describe` 分别返回表的 信息概况和表中数值列对应的主要统计量(均值方差最大最小值等信息)
```python
df.info()
df.describe()
```
### 1.3.2 特征统计函数
- 在 `Series` 和 `DataFrame` 上定义了许多统计函数，最常见的是 `sum`, `mean`, `median`, `var`, `std`, `max`, `min`
- `quantile`, `count`, `idxmax` ，分别返回分位数、非缺失值个数、最大值对应的索引
```python
df_demo = df[['Height', 'Weight']]
a1, a2, a3 = df_demo.quantile(), df_demo.count(), df_demo.idxmax()
```
- 上面这些所有的函数，由于操作后返回的是标量，所以又称为聚合函数，它们有一个公共参数 `axis` ，默认为0代表逐列聚合，如果设置为1则表示逐行聚合。
### 1.3.3 唯一值函数
- 对序列使用 `unique` 和 `nunique` 可以分别得到其唯一值组成的列表和唯一值的个数
```python
df['School'].unique()                                   # ['Shanghai Jiao Tong University', 'Peking University', 'Fudan University', 'Tsinghua University']
df['School'].nunique()                                  # 4
```
- `value_counts` 可以得到唯一值和其对应出现的频数
```python
df['School'].value_counts()
```
```markup
Tsinghua University              69
Shanghai Jiao Tong University    57
Fudan University                 40
Peking University                34
Name: School, dtype: int64
```
- 如果想要观察多个列组合的唯一值，可以使用 `DataFrame.drop_duplicates()`
    - `keep=first` 保留第一次出现的所在行
    - `keep=last` 保留最后一次出现的所在行
    - `keep=False` 表示把所有重复组合所在的行剔除。
- `DataFrame.duplicated()` 返回是否为唯一值的布尔列表，`keep` 参数与 `DataFrame.drop_duplicates()` 一致。其返回的序列，把重复元素设为 `True` ，否则为 `False` 。
### 1.3.4 替换函数
#### 映射替换
- `replace` 可以通过字典构造，或者传入两个列表来进行替换
```python
df['Gender'].replace({'Female':0, 'Male':1}).head()
df['Gender'].replace(['Female', 'Male'], [0, 1]).head()
```
- `replace` 还有一种特殊的方向替换，指定 method 参数为 ffill 则为用前面一个最近的未被替换的值进行替换， bfill 则使用后面最近的未被替换的值进行替换
```python
s = pd.Series(['a', 1, 'b', 2, 1, 1, 'a'])
s1 = s.replace([1, 2], method='ffill')
s2 = s.replace([1, 2], method='bfill')
print(s1, s2)
```
```markup
0    a
1    a
2    b
3    b
4    b
5    b
6    a
dtype: object 
0    a
1    b
2    b
3    a
4    a
5    a
6    a
dtype: object
```
#### 逻辑替换
- 包括 `where` 和 `mask` ，这两个函数是完全对称的： `where` 函数在传入条件为 `False` 的对应行进行替换，而 `mask` 在传入条件为 `True` 的对应行进行替换，当不指定替换值时，替换为缺失值
```python
s = pd.Series([-1, 1.2345, 100, -50])
s1 = s.where(s<0)
s2 = s.where(s<0, 100)
s3 = s.mask(s<0)
s4 = s.mask(s<0, -50)
s_condition= pd.Series([True,False,False,True],index=s.index)
s5 = s.mask(s_condition, -50)
print(s1, s2, s3, s4, s5)
```
```markup
0    -1.0
1     NaN
2     NaN
3   -50.0
dtype: float64 
0     -1.0
1    100.0
2    100.0
3    -50.0
dtype: float64 
0         NaN
1      1.2345
2    100.0000
3         NaN
dtype: float64 
0    -50.0000
1      1.2345
2    100.0000
3    -50.0000
dtype: float64
0    -50.0000
1      1.2345
2    100.0000
3    -50.0000
dtype: float64
```
#### 数值替换
- `round`, `abs`, `clip` 方法，分别表示按照给定精度四舍五入、取绝对值和截断
```python
s = pd.Series([-1, 1.2345, 100, -50])
s1 = s.round(2)
s2 = s.abs()
s3 = s.clip(0, 2)                                            # 前两个数分别表示上下截断边界
print(s1, s2, s3)
```
```markup
0     -1.00
1      1.23
2    100.00
3    -50.00
dtype: float64 
0      1.0000
1      1.2345
2    100.0000
3     50.0000
dtype: float64 
0    0.0000
1    1.2345
2    2.0000
3    0.0000
dtype: float64
```
### 1.3.5 排序函数
- 演示用 `set_index` 方法把年级和姓名两列作为索引，另两列为值
```python
df_demo = df[['Grade', 'Name', 'Height',
              'Weight']].set_index(['Grade','Name'])
```
#### 值排序
- 通过 `sort_values` 函数对身高进行排序，默认参数 `ascending=True` 为升序
```python
df_demo.sort_values('Height')
df_demo.sort_values('Height', ascending=False)
# 升序排序Weight，再降序排序Height
df_demo.sort_values(['Weight','Height'],ascending=[True,False])
```
#### 索引排序
- `sort_index` 的用法和值排序完全一致，只不过元素的值在索引中，此时需要指定索引层的名字或者层号，用参数 `level` 表示。另外，需要注意的是字符串的排列顺序由字母顺序决定
```python
df_demo.sort_index(level=['Grade','Name'],ascending=[True,False])
```
### 1.3.6 apply方法
- `apply` 方法常用于 `DataFrame` 的行迭代或者列迭代，参数是一个以序列为输入的函数
```python
df_demo = df[['Height', 'Weight']]
def my_mean(x):
    res = x.mean()
    return res
df_demo.apply(my_mean)
# lambda写法也可以
df_demo.apply(lambda x:x.mean())
df_demo.apply(lambda x:x.mean(), axis=1)                    # 指定沿行应用函数
```
```markup
Height    163.218033
Weight     55.015873
dtype: float64
```
- 得益于传入自定义函数的处理， `apply` 的自由度很高，但这是以性能为代价的。一般而言，使用 `pandas` 的内置函数处理和 `apply` 来处理同一个任务，其速度会相差较多，因此只有在确实存在自定义需求的情境下才考虑使用 `apply` 。

## 1.4 窗口对象
### 1.4.1 滑动窗口
- 对一个序列使用 `.rolling` 得到滑窗对象，其最重要的参数为窗口大小 `window`
```python
s = pd.Series([1,2,3,4,5])
roller = s.rolling(window = 3)
```
- 使用相应的聚合函数进行计算，需要注意的是窗口包含当前行所在的元素，例如在第四个位置进行均值运算时，应当计算(2+3+4)/3，而不是(1+2+3)/3
```python
r1 = roller.mean()
# 等价于 roller.apply(lambda x:x.mean())
r2 = roller.sum()
print(r1, r2)
```
```markup
0    NaN
1    NaN
2    2.0
3    3.0
4    4.0
dtype: float64 
0     NaN
1     NaN
2     6.0
3     9.0
4    12.0
dtype: float64
```
- 对于另一个序列的滑动相关系数或滑动协方差
```python
s2 = pd.Series([1,2,6,16,30])
rs1 = roller.cov(s2)
rs2 = roller.corr(s2)
print(rs1, rs2)
```
```markup
0     NaN
1     NaN
2     2.5
3     7.0
4    12.0
dtype: float64 
0         NaN
1         NaN
2    0.944911
3    0.970725
4    0.995402
dtype: float64
```
- `shift`, `diff`, `pct_change` 是一组类滑窗函数，它们的公共参数为 `periods=n` ，默认为1，分别表示取向前第 n 个元素的值、与向前第 n 个元素做差（与 `Numpy` 中不同，后者表示 n 阶差分）、与向前第 n 个元素相比计算增长率。这里的 n 可以为负，表示反方向的类似操作。
```python
s = pd.Series([1,3,6,10,15])
s1 = s.shift(2)
s2 = s.diff(3)
s3 = s.pct_change()
s4 = s.shift(-1)
s5 = s.diff(-2)
print(s1, s2, s3, s4, s5)
```
```markup
0    NaN
1    NaN
2    1.0
3    3.0
4    6.0
dtype: float64 
0     NaN
1     NaN
2     NaN
3     9.0
4    12.0
dtype: float64 
0         NaN
1    2.000000
2    1.000000
3    0.666667
4    0.500000
dtype: float64 
0     3.0
1     6.0
2    10.0
3    15.0
4     NaN
dtype: float64 
0   -5.0
1   -7.0
2   -9.0
3    NaN
4    NaN
dtype: float64
```
### 1.4.2 扩张窗口
- 扩张窗口又称累计窗口，可以理解为一个动态长度的窗口，其窗口的大小就是从序列开始处到具体操作的对应位置，其使用的聚合函数会作用于这些逐步扩张的窗口上。具体地说，设序列为a1, a2, a3, a4，则其每个位置对应的窗口即[a1]、[a1, a2]、[a1, a2, a3]、[a1, a2, a3, a4]
```python
s = pd.Series([1, 3, 6, 10])
s_mean = s.expanding().mean()
print(s_mean)
```
```markup
0    1.000000
1    2.000000
2    3.333333
3    5.000000
dtype: float64
```