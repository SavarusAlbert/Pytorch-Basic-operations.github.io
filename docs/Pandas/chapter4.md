# Chapter04 变形
## 4.1 长宽表的变形
- 对于表的某一特征来说，数据的呈现形式决定了其为长表还是宽表。举例来说：
```python
import numpy as np
import pandas as pd
# 关于性别的长表
pd.DataFrame({'Gender':['F','F','M','M'],
              'Height':[163, 160, 175, 180]})
```
```markup
  Gender  Height
0      F     163
1      F     160
2      M     175
3      M     180
```
```python
# 关于性别的宽表
pd.DataFrame({'Height: F':[163, 160],
              'Height: M':[175, 180]})
```
```markup
   Height: F  Height: M
0        163        175
1        160        180
```
### 4.1.1 pivot函数
- 通过 `pivot` 函数变形长宽表，其中的 `index`, `columns`, `values` 参数，表示变形后的行索引、列索引和对应的数值。利用 `pivot` 进行变形操作需要满足唯一性的要求，即由于在新表中的行列索引对应了唯一的 `value` ，因此原表中的 `index` 和 `columns` 对应两个列的行组合必须唯一。
```python
# 定义样例表
df = pd.DataFrame({'Class':[1,1,2,2],
                   'Name':['San Zhang','San Zhang','Si Li','Si Li'],
                   'Subject':['Chinese','Math','Chinese','Math'],
                   'Grade':[80,75,90,85]})
print(df)
```
```markup
   Class       Name  Subject  Grade
0      1  San Zhang  Chinese     80
1      1  San Zhang     Math     75
2      2      Si Li  Chinese     90
3      2      Si Li     Math     85
```
```python
df.pivot(index='Name', columns='Subject', values='Grade')
```
```markup
Subject    Chinese  Math
Name                    
San Zhang       80    75
Si Li           90    85
```
- `pivot` 相关的三个参数允许被设置为列表，会返回多级索引。
```python
df = pd.DataFrame({'Class':[1, 1, 2, 2, 1, 1, 2, 2],
                  'Name':['San Zhang', 'San Zhang', 'Si Li', 'Si Li',
                           'San Zhang', 'San Zhang', 'Si Li', 'Si Li'],
                  'Examination': ['Mid', 'Final', 'Mid', 'Final',
                                 'Mid', 'Final', 'Mid', 'Final'],
                  'Subject':['Chinese', 'Chinese', 'Chinese', 'Chinese',
                              'Math', 'Math', 'Math', 'Math'],
                  'Grade':[80, 75, 85, 65, 90, 85, 92, 88],
                  'rank':[10, 15, 21, 15, 20, 7, 6, 2]})
print(df)
```
```markup
   Class       Name Examination  Subject  Grade  rank
0      1  San Zhang         Mid  Chinese     80    10
1      1  San Zhang       Final  Chinese     75    15
2      2      Si Li         Mid  Chinese     85    21
3      2      Si Li       Final  Chinese     65    15
4      1  San Zhang         Mid     Math     90    20
5      1  San Zhang       Final     Math     85     7
6      2      Si Li         Mid     Math     92     6
7      2      Si Li       Final     Math     88     2
```
```python
df.pivot(index = ['Class', 'Name'],
        columns = ['Subject','Examination'],
        values = ['Grade','rank'])
```
```markup
                  Grade                     rank                 
Subject         Chinese       Math       Chinese       Math      
Examination         Mid Final  Mid Final     Mid Final  Mid Final
Class Name                                                       
1     San Zhang      80    75   90    85      10    15   20     7
2     Si Li          85    65   92    88      21    15    6     2
```
### 4.1.2 pivot_table函数
- 如果变换的表不满足唯一性，就不能使用 `pivot` 函数，对于需要聚合多个值的情况，可以用 `pivot_table` 函数。
```python
df = pd.DataFrame({'Name':['San Zhang', 'San Zhang',
                           'San Zhang', 'San Zhang',
                           'Si Li', 'Si Li', 'Si Li', 'Si Li'],
                  'Subject':['Chinese', 'Chinese', 'Math', 'Math',
                              'Chinese', 'Chinese', 'Math', 'Math'],
                  'Grade':[80, 90, 100, 90, 70, 80, 85, 95]})
print(df)
```
```markup
        Name  Subject  Grade
0  San Zhang  Chinese     80
1  San Zhang  Chinese     90
2  San Zhang     Math    100
3  San Zhang     Math     90
4      Si Li  Chinese     70
5      Si Li  Chinese     80
6      Si Li     Math     85
7      Si Li     Math     95
```
```python
# 对多次成绩求均值
df.pivot_table(index = 'Name',
               columns = 'Subject',
               values = 'Grade',
               aggfunc = 'mean')
# 也可以传入以序列为输入标量为输出的聚合函数来实现自定义操作
df.pivot_table(index = 'Name',
               columns = 'Subject',
               values = 'Grade',
               aggfunc = lambda x:x.mean())
```
```markup
Subject    Chinese  Math
Name                    
San Zhang       85    95
Si Li           75    90
```
- 此外， `pivot_table` 具有边际汇总的功能，可以通过设置 `margins=True` 来实现，其中边际的聚合方式与 `aggfunc` 中给出的聚合方法一致。
```python
df.pivot_table(index = 'Name',
               columns = 'Subject',
               values = 'Grade',
               aggfunc='mean',
               margins=True)
```
```markup
Subject    Chinese  Math    All
Name                           
San Zhang       85  95.0  90.00
Si Li           75  90.0  82.50
All             80  92.5  86.25
```
### 4.1.3 melt函数
-  `pivot` 把长表转为宽表，`melt` 通过相应的逆操作把宽表转为长表。
```python
# 样例表
df = pd.DataFrame({'Class':[1,2],
                  'Name':['San Zhang', 'Si Li'],
                  'Chinese':[80, 90],
                  'Math':[80, 75]})
print(df)
```
```markup
   Class       Name  Chinese  Math
0      1  San Zhang       80    80
1      2      Si Li       90    75
```
```python
df_melted = df.melt(id_vars = ['Class', 'Name'],
                    value_vars = ['Chinese', 'Math'],
                    var_name = 'Subject',
                    value_name = 'Grade')
```
```markup
   Class       Name  Subject  Grade
0      1  San Zhang  Chinese     80
1      2      Si Li  Chinese     90
2      1  San Zhang     Math     80
3      2      Si Li     Math     75
```
### 4.1.4 wide_to_long函数
- 如果列中包含了交叉类别，就需要使用 `wide_to_long` 函数来完成。
```python
df = pd.DataFrame({'Class':[1,2],'Name':['San Zhang', 'Si Li'],
                   'Chinese_Mid':[80, 75], 'Math_Mid':[90, 85],
                   'Chinese_Final':[80, 75], 'Math_Final':[90, 85]})
print(df)
```
```markup
   Class       Name  Chinese_Mid  Math_Mid  Chinese_Final  Math_Final
0      1  San Zhang           80        90             80          90
1      2      Si Li           75        85             75          85
```
```python
pd.wide_to_long(df,
                stubnames=['Chinese', 'Math'],
                i = ['Class', 'Name'],
                j='Examination',
                sep='_',
                suffix='.+')
```
```markup
                             Chinese  Math
Class Name      Examination               
1     San Zhang Mid               80    90
                Final             80    90
2     Si Li     Mid               75    85
                Final             75    85
```

## 4.2 索引的变形
### 4.2.1 unstack函数
- 在第二章中提到了利用 `swaplevel` 或者 `reorder_levels` 进行索引内部的层交换，下面就要讨论 行列索引之间 的交换，由于这种交换带来了 `DataFrame` 维度上的变化，因此属于变形操作。
- `unstack` 函数的作用是把行索引转为列索引。`unstack` 的主要参数是移动的层号，默认转化最内层，移动到列索引的最内层，同时支持同时转化多个层：
```python
df = pd.DataFrame(np.ones((4,2)),
                  index = pd.Index([('A', 'cat', 'big'),
                                    ('A', 'dog', 'small'),
                                    ('B', 'cat', 'big'),
                                    ('B', 'dog', 'small')]),
                  columns=['col_1', 'col_2'])
print(df)
```
```markup
             col_1  col_2
A cat big      1.0    1.0
  dog small    1.0    1.0
B cat big      1.0    1.0
  dog small    1.0    1.0
```
```python
df.unstack()
# 等价于df.unstack(2)
```
```markup
      col_1       col_2      
        big small   big small
A cat   1.0   NaN   1.0   NaN
  dog   NaN   1.0   NaN   1.0
B cat   1.0   NaN   1.0   NaN
  dog   NaN   1.0   NaN   1.0
```
```python
df.unstack([0,2])
```
```markup
    col_1                  col_2                 
        A          B           A          B      
      big small  big small   big small  big small
cat   1.0   NaN  1.0   NaN   1.0   NaN  1.0   NaN
dog   NaN   1.0  NaN   1.0   NaN   1.0  NaN   1.0
```
- 类似于 `pivot` 中的唯一性要求，在 `unstack` 中必须保证 被转为列索引的行索引层和被保留的行索引层构成的组合是唯一的。
### 4.2.2 stack函数
- `stack` 的作用就是把列索引的层压入行索引。
```python
df.unstack().stack().equals(df)                             # True
df.unstack([1,2]).stack([1,2]).equals(df)                   # True
```

## 4.3 其他变形函数
### 4.3.1 crosstab函数
- `crosstab` 能实现的所有功能 `pivot_table` 都能完成。在默认状态下， `crosstab` 可以统计元素组合出现的频数，即 `count` 操作。
```python
df = pd.read_csv('data/learn_pandas.csv')
pd.crosstab(index = df.School, columns = df.Transfer)
# 这等价于如下写法，aggfunc即聚合参数
pd.crosstab(index = df.School, columns = df.Transfer,
            values = [0]*df.shape[0], aggfunc = 'count')
```
```markup
Transfer                        N  Y
School                              
Fudan University               38  1
Peking University              28  2
Shanghai Jiao Tong University  53  0
Tsinghua University            62  4
```
- 可以利用 `pivot_table` 进行等价操作，由于这里统计的是组合的频数，因此 `values` 参数无论传入哪一个列都不会影响最后的结果。
```python
df.pivot_table(index = 'School',
               columns = 'Transfer',
               values = 'Name',
               aggfunc = 'count')
```
### 4.3.2 explode函数
- `explode` 参数能够对某一列的元素进行纵向的展开，被展开的单元格必须存储 `list`, `tuple`, `Series`, `np.ndarray` 中的一种类型。
```python
df_ex = pd.DataFrame({'A': [[1, 2],
                         'my_str',
                         {1, 2},
                         pd.Series([3, 4])],
                      'B': 1})
print(df)
```
```markup
	A	                    B
0	[1, 2]	                1
1	my_str	                1
2	{1, 2}	                1
3	0 3 1 4 dtype: int64	1
```
```python
df_ex.explode('A')
```
```markup
        A  B
0       1  1
0       2  1
1  my_str  1
2       1  1
2       2  1
3       3  1
3       4  1
```
### 4.3.3 get_dummies函数
- `get_dummies` 是用于特征构建的重要函数之一，其作用是把类别特征转为指示变量。
```python
# 对年级一列转为指示变量，属于某一个年级的对应列标记为1，否则为0
pd.get_dummies(df.Grade)
```
```markup
   Freshman  Junior  Senior  Sophomore
0         1       0       0          0
1         1       0       0          0
2         0       0       1          0
3         0       0       0          1
4         0       0       0          1
```