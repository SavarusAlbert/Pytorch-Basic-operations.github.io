# Chapter02 索引
## 2.1 索引器
### 2.1.1 表的列索引
- 列索引是最常见的索引形式，一般通过 `[]` 来实现。通过 `[列名]` 可以从 `DataFrame` 中取出相应的列，返回值为 `Series`。
- 如果要取出多个列，则可以通过 [列名组成的列表] ，其返回值为一个 `DataFrame` ，例如从表中取出性别和姓名两列。
- 此外，若要取出单列，且列名中不包含空格，则可以用 `.列名` 取出，这和 `[列名]` 是等价的。
```python
import numpy as np
import pandas as pd
df = pd.read_csv('data/learn_pandas.csv',
                 usecols = ['School', 'Grade', 'Name', 'Gender',
                            'Weight', 'Transfer'])
df['Name']
df[['Gender', 'Name']]
df.Name
```
### 2.1.2 序列的行索引
#### 以字符串为索引的 `Series`
- 如果取出单个索引的对应元素，则可以使用 `[item]` ，若 `Series` 只有单个值对应，则返回这个标量值，如果有多个值对应，则返回一个 `Series`。
- 如果取出多个索引的对应元素，则可以使用 `[items的列表]`。
- 如果想要取出某两个索引之间的元素，并且这两个索引是在整个索引中唯一出现，则可以使用切片，同时需要注意这里的切片会包含两个端点。
- 如果前后端点的值存在重复，即非唯一值，那么需要经过排序才能使用切片。
```python
s = pd.Series([1, 2, 3, 4, 5, 6],
               index=['a', 'b', 'a', 'a', 'a', 'c'])
s['a']
s['b']
s[['c', 'b']]
s['c': 'b': -2]
s.sort_index()['a': 'b']
```
#### 以整数为索引的 `Series`
```python
s = pd.Series(['a', 'b', 'c', 'd', 'e', 'f'],
              index=[1, 3, 1, 2, 5, 4])
s[1]                                                        # 生成从0开始的所有整数索引
s[[2,3]]                                                    # 取出对应索引的值
s[1:-1:2]                                                   # 同python切片
```
### 2.1.3 loc索引器
- 对 `DataFrame` 的行进行选取，分为两种，一种是基于元素的 `loc` 索引器，另一种是基于位置的 `iloc` 索引器。
- `loc` 索引器的一般形式是 `loc[*, *]` ，其中第一个 * 代表行的选择，第二个 * 代表列的选择，如果省略第二个位置写作 `loc[*]` ，这个 * 是指行的筛选。
- 其中， * 的位置一共有五类合法对象，分别是：单个元素、元素列表、元素切片、布尔列表以及函数。
```python
# 利用 set_index 方法把 Name 列设为索引，便于演示
df_demo = df.set_index('Name')
```
#### * 为单个元素
- 此时，直接取出相应的行或列，如果该元素在索引中重复则结果为 `DataFrame`，否则为 `Series`。
- 也可以同时选择行和列。
```python
df_demo.loc['Qiang Sun']                                    # 多个人叫此名字
df_demo.loc['Quan Zhao']                                    # 名字唯一
df_demo.loc['Qiang Sun', 'School']                          # 返回Series
df_demo.loc['Quan Zhao', 'School']                          # 返回单个元素
```
#### * 为元素列表
- 取出列表中所有元素值对应的行或列。
```python
df_demo.loc[['Qiang Sun','Quan Zhao'], ['School','Gender']]
```
#### * 为切片
- 同 `Series` 切片，如果是唯一值的起点和终点字符，那么就可以使用切片，并且包含两个端点，如果不唯一则报错。
```python
df_demo.loc['Gaojuan You':'Gaoqiang Qian', 'School':'Gender']
```
- 如果 `DataFrame` 使用整数索引，其使用整数切片的时候和上面**字符串索引**的要求一致，都是**元素切片**，包含端点且起点、终点，不允许有重复值。
```python
df_loc_slice_demo = df_demo.copy()
df_loc_slice_demo.index = range(df_demo.shape[0],0,-1)
df_loc_slice_demo.loc[5:3]                                  # 返回索引5到索引3的部分
df_loc_slice_demo.loc[3:5]                                  # 返回空表
```
#### * 为布尔列表
- 在实际的数据处理中，根据条件来筛选行是极其常见的，此处传入 `loc` 的布尔列表与 `DataFrame` 长度相同，且列表为 `True` 的位置所对应的行会被选中， `False` 则会被剔除。
- 可以通过 `isin` 方法返回的布尔列表
- 对于复合条件而言，可以用 `|`(或), `&`(且), `~`(取反) 的组合来实现。
```python
df_demo.loc[df_demo.Weight>70]
df_demo.loc[df_demo.Grade.isin(['Freshman', 'Senior'])]
df_demo.loc[condition_1 | condition_2]                      # condition 可以为复合条件
```
#### * 为函数
- 这里的函数，必须以前面的四种合法形式之一为返回值，并且函数的输入值为 `DataFrame` 本身，支持使用 `lambda` 表达式。
- 由于函数无法返回如 `start: end: step` 的切片形式，故返回切片时要用 `slice` 对象进行包装。
```python
def condition(x):
    condition_1 = x.School == 'Fudan University'
    condition_2 = x.School == 'Peking University'
    result = condition_1 | condition_2
    return result
df_demo.loc[condition]
df_demo.loc[lambda x:'Quan Zhao', lambda x:'Gender']
df_demo.loc[lambda x: slice('Gaojuan You', 'Gaoqiang Qian')]
```
### 2.1.4 iloc索引器
- `iloc` 的使用与 `loc` 完全类似，只不过是针对位置进行筛选，在相应的 * 位置处一共也有五类合法对象，分别是：整数、整数列表、整数切片、布尔列表以及函数，函数的返回值必须是前面的四类合法对象中的一个，其输入同样也为 `DataFrame` 本身。
```python
df_demo.iloc[1, 1]                                          # 第二行第二列
df_demo.iloc[[0, 1], [0, 1]]                                # 前两行前两列
df_demo.iloc[1: 4, 2:4]                                     # 切片不包含结束端点
df_demo.iloc[lambda x: slice(1, 4)]                         # 传入切片为返回值的函数
```
- 在使用布尔列表的时候要特别注意，不能传入 `Series` 而必须传入序列的 `values` ，否则会报错。
```python
df_demo.iloc[(df_demo.Weight>80).values]
```
### 2.1.5 query方法
- 在 `pandas` 中，支持把字符串形式的查询表达式传入 `query` 方法来查询数据，其表达式的执行结果必须返回布尔列表。在进行复杂索引时，由于这种检索方式无需像普通方法一样重复使用 `DataFrame` 的名字来引用列名，一般而言会使代码长度在不降低可读性的前提下有所减少。
```python
df.query('((School == "Fudan University")&'
         ' (Grade == "Senior")&'
         ' (Weight > 70))|'
         '((School == "Peking University")&'
         ' (Grade != "Senior")&'
         ' (Weight > 80))')
```
- 在 `query` 表达式中，帮用户注册了所有来自 `DataFrame` 的列名，所有属于该 `Series` 的方法都可以被调用，和正常的函数调用并没有区别，还注册了若干英语的字面用法，帮助提高可读性
```python
df.query('Weight > Weight.mean()')
df.query('(Grade not in ["Freshman", "Sophomore"]) and'
         '(Gender == "Male")')
df.query('Grade == ["Junior", "Senior"]')
```
- 对于 `query` 中的字符串，如果要引用外部变量，只需在变量名前加 `@` 符号。
```python
low, high =70, 80
df.query('(Weight >= @low) & (Weight <= @high)')
```
### 2.1.6 随机抽样
- 如果把 `DataFrame` 的每一行看作一个样本，或把每一列看作一个特征，再把整个 `DataFrame` 看作总体，想要对样本或特征进行随机抽样就可以用 `sample` 函数。有时在拿到大型数据集后，想要对统计特征进行计算来了解数据的大致分布，但是这很费时间。同时，由于许多统计特征在等概率不放回的简单随机抽样条件下，是总体统计特征的无偏估计，比如样本均值和总体均值，那么就可以先从整张表中抽出一部分来做近似估计。
- `sample` 函数中的主要参数为 `n`, `axis`, `frac`, `replace`, `weights` ，前三个分别是指抽样数量、抽样的方向(0为行、1为列)和抽样比例(0.3则为从总体中抽出30%的样本)。
- `replace` 和 `weights` 分别是指是否放回和每个样本的抽样相对概率，当 `replace = True` 则表示有放回抽样。
```python
df_sample = pd.DataFrame({'id': list('abcde'),
                          'value': [1, 2, 3, 4, 90]})
df_sample.sample(3, replace = True, weights = df_sample.value)
```

## 2.2 多级索引
### 2.2.1 多级索引及其表的结构
- 构造多级索引的 `DataFrame` 用来举例。
```python
np.random.seed(0)
multi_index = pd.MultiIndex.from_product([list('ABCD'),
              df.Gender.unique()], names=('School', 'Gender'))
multi_column = pd.MultiIndex.from_product([['Height', 'Weight'],
               df.Grade.unique()], names=('Indicator', 'Grade'))
df_multi = pd.DataFrame(np.c_[(np.random.randn(8,4)*5 + 163).tolist(),
                              (np.random.randn(8,4)*5 + 65).tolist()],
                        index = multi_index,
                        columns = multi_column).round(1)
df_multi
```
```markup
Indicator       Height                           Weight                        
Grade         Freshman Senior Sophomore Junior Freshman Senior Sophomore Junior
School Gender                                                                  
A      Female    171.8  165.0     167.9  174.2     60.6   55.1      63.3   65.8
       Male      172.3  158.1     167.8  162.2     71.2   71.0      63.1   63.5
B      Female    162.5  165.1     163.7  170.3     59.8   57.9      56.5   74.8
       Male      166.8  163.6     165.2  164.7     62.5   62.8      58.7   68.9
C      Female    170.5  162.0     164.6  158.7     56.9   63.9      60.5   66.9
       Male      150.2  166.3     167.3  159.3     62.4   59.1      64.9   67.1
D      Female    174.3  155.7     163.2  162.1     65.3   66.5      61.8   63.2
       Male      170.7  170.3     163.8  164.9     61.6   63.2      60.9   56.4
```
- 与单层索引的表一样，具备元素值、行索引和列索引三个部分。其中，这里的行索引和列索引都是 `MultiIndex` 类型，只不过 索引中的一个元素是元组 而不是单层索引中的标量。例如，行索引的第四个元素为 `("B", "Male")` ，列索引的第二个元素为 `("Height", "Senior")` ，这里需要注意，外层连续出现相同的值时，第一次之后出现的会被隐藏显示，使结果的可读性增强。
- 与单层索引类似， `MultiIndex` 也具有名字属性，图中的 `School` 和 `Gender` 分别对应了表的第一层和第二层行索引的名字， `Indicator` 和 `Grade` 分别对应了第一层和第二层列索引的名字。索引的名字和值属性分别可以通过 `names` 和 `values` 获得。
```python
df_multi.index.names                                        # FrozenList(['School', 'Gender'])
df_multi.columns.names                                      # FrozenList(['Indicator', 'Grade'])
df_multi.index.values                                       # array([('A', 'Female'), ('A', 'Male'), ('B', 'Female'), ('B', 'Male'), ('C', 'Female'), ('C', 'Male'), ('D', 'Female'), ('D', 'Male')], dtype=object)
df_multi.columns.values                                     # array([('Height', 'Freshman'), ('Height', 'Senior'), ('Height', 'Sophomore'), ('Height', 'Junior'), ('Weight', 'Freshman'), ('Weight', 'Senior'), ('Weight', 'Sophomore'), ('Weight', 'Junior')], dtype=object)
df_multi.index.get_level_values(0)                          # Index(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'], dtype='object', name='School')
```
### 2.2.2 多级索引中的loc索引器
- 回到原表，将学校和年级设为索引，此时的行为多级索引，列为单级索引。
```python
df_multi = df.set_index(['School', 'Grade'])
print(df_multi.head())
```
```markup
                                                   Name  Gender  Weight Transfer
School                        Grade                                             
Shanghai Jiao Tong University Freshman     Gaopeng Yang  Female    46.0        N
Peking University             Freshman   Changqiang You    Male    70.0        N
Shanghai Jiao Tong University Senior            Mei Sun    Male    89.0        N
Fudan University              Sophomore    Xiaojuan Sun  Female    41.0        N
                              Sophomore     Gaojuan You    Male    74.0        N
```
- 由于多级索引中的单个元素以元组为单位，因此之前在2.1中介绍的 `loc` 和 `iloc` 方法完全可以照搬，只需把标量的位置替换成对应的元组。
```python
df_sorted = df_multi.sort_index()
df_sorted.loc[('Fudan University', 'Junior')]
df_sorted.loc[[('Fudan University', 'Senior'),
              ('Shanghai Jiao Tong University', 'Freshman')]]
df_sorted.loc[df_sorted.Weight > 70]
df_sorted.loc[lambda x:('Fudan University','Junior')]
```
- 当使用切片时需要注意，在单级索引中只要切片端点元素是唯一的，那么就可以进行切片，但在多级索引中，无论元组在索引中是否重复出现，都必须经过排序才能使用切片，否则报错。
```python
df_sorted.loc[('Fudan University', 'Senior'):]
```
- 在多级索引中的元组有一种特殊的用法，可以对多层的元素进行交叉组合后索引，但同时需要指定 `loc` 的列，全选则用 `:` 表示
```python
# 选出所有北大和复旦的大二大三学生
df_multi.loc[(['Peking University', 'Fudan University'],
                    ['Sophomore', 'Junior']), :]
# 选出北大的大三学生和复旦的大二学生
df_multi.loc[[('Peking University', 'Junior'),
                    ('Fudan University', 'Sophomore')]]
```
### 2.2.3 IndexSlice对象
- 构造一个 索引不重复的 `DataFrame`。
```python
np.random.seed(0)
L1,L2 = ['A','B','C'],['a','b','c']
mul_index1 = pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
L3,L4 = ['D','E','F'],['d','e','f']
mul_index2 = pd.MultiIndex.from_product([L3,L4],names=('Big', 'Small'))
df_ex = pd.DataFrame(np.random.randint(-9,10,(9,9)),
                    index=mul_index1,
                    columns=mul_index2)
print(df_ex)
```
```markup
Big          D        E        F      
Small        d  e  f  d  e  f  d  e  f
Upper Lower                           
A     a      3  6 -9 -6 -6 -2  0  9 -5
      b     -3  3 -8 -3 -2  5  8 -4  4
      c     -1  0  7 -4  6  6 -9  9 -6
B     a      8  5 -2 -9 -8  0 -9  1 -6
      b      2  9 -7 -9 -9 -5 -4 -3 -1
      c      8  6 -5  0  1 -8 -8 -2  0
C     a     -6 -3  2  5  9 -9  5 -6  3
      b      1  2 -5 -3 -5  6 -6  3 -5
      c     -1  5  6 -6  6  4  7  8 -4
```
- 为了使用 `silce` 对象，先要进行定义：
```python
idx = pd.IndexSlice
```
#### `loc[idx[*,*]]` 型
- 这种情况并不能进行多层分别切片，前一个 * 表示行的选择，后一个 * 表示列的选择，与单纯的 `loc` 类似。
```python
df_ex.loc[idx['C':, ('D', 'f'):]]
df_ex.loc[idx[:'A', lambda x:x.sum()>0]]
df_ex.sum() > 0
```
#### `loc[idx[*,*],idx[*,*]]` 型
- 这种情况能够分层进行切片，前一个 idx 指代的是行索引，后一个是列索引，但需要注意的是，此时不支持使用函数。
```python
df_ex.loc[idx[:'A', 'b':], idx['E':, 'e':]]
```
### 2.2.4 多级索引的构造
- `from_tuples` 指根据传入由元组组成的列表进行构造。
```python
my_tuple = [('a','cat'),('a','dog'),('b','cat'),('b','dog')]
pd.MultiIndex.from_tuples(my_tuple, names=['First','Second'])
```
```markup
MultiIndex([('a', 'cat'),
            ('a', 'dog'),
            ('b', 'cat'),
            ('b', 'dog')],
           names=['First', 'Second'])
```
- `from_arrays` 指根据传入列表中，对应层的列表进行构造。
```python
my_array = [list('aabb'), ['cat', 'dog']*2]
pd.MultiIndex.from_arrays(my_array, names=['First','Second'])
```
```markup
MultiIndex([('a', 'cat'),
            ('a', 'dog'),
            ('b', 'cat'),
            ('b', 'dog')],
           names=['First', 'Second'])
```
- `from_product` 指根据给定多个列表的笛卡尔积进行构造。
```python
my_list1 = ['a','b']
my_list2 = ['cat','dog']
pd.MultiIndex.from_product([my_list1,
                            my_list2],
                           names=['First','Second'])
```
```markup
MultiIndex([('a', 'cat'),
            ('a', 'dog'),
            ('b', 'cat'),
            ('b', 'dog')],
           names=['First', 'Second'])
```

## 2.3 索引的常用方法
### 2.3.1 索引层的交换和删除
- 构造一个三级索引的例子：
```python
np.random.seed(0)
L1,L2,L3 = ['A','B'],['a','b'],['alpha','beta']
mul_index1 = pd.MultiIndex.from_product([L1,L2,L3],
             names=('Upper', 'Lower','Extra'))
L4,L5,L6 = ['C','D'],['c','d'],['cat','dog']
mul_index2 = pd.MultiIndex.from_product([L4,L5,L6],
             names=('Big', 'Small', 'Other'))
df_ex = pd.DataFrame(np.random.randint(-9,10,(8,8)),
                        index=mul_index1,
                        columns=mul_index2)
print(df_ex)
```
```markup
Big                 C               D            
Small               c       d       c       d    
Other             cat dog cat dog cat dog cat dog
Upper Lower Extra                                
A     a     alpha   3   6  -9  -6  -6  -2   0   9
            beta   -5  -3   3  -8  -3  -2   5   8
      b     alpha  -4   4  -1   0   7  -4   6   6
            beta   -9   9  -6   8   5  -2  -9  -8
B     a     alpha   0  -9   1  -6   2   9  -7  -9
            beta   -9  -5  -4  -3  -1   8   6  -5
      b     alpha   0   1  -8  -8  -2   0  -6  -3
            beta    2   5   9  -9   5  -6   3   1
```
- 索引层的交换由 `swaplevel` 和 `reorder_levels` 完成，前者只能交换两个层，而后者可以交换任意层，两者都可以指定交换的是轴是哪一个，即行索引或列索引。
```python
df_ex.swaplevel(0,2,axis=1)                                 # 列索引的第一层和第三层交换
df_ex.reorder_levels([2,0,1],axis=0)                        # 列表数字指代原来索引中的层
```
- 若想要删除某一层的索引，可以使用 `droplevel` 方法。
```python
df_ex.droplevel(1,axis=1)
df_ex.droplevel([0,1],axis=0)
```
### 2.3.2 索引属性的修改
- 通过 `rename_axis` 可以对索引层的名字进行修改，常用的修改方式是传入字典的映射。
```python
df_ex.rename_axis(index={'Upper':'Changed_row'},
                  columns={'Other':'Changed_Col'})
```
- 通过 `rename` 可以对索引的值进行修改，如果是多级索引需要指定修改的层号 `level`。
```python
df_ex.rename(columns={'cat':'not_cat'},
             level=2).head()
```
- 传入参数也可以是函数，其输入值就是索引元素。
```python
df_ex.rename(index=lambda x:str.upper(x),
             level=2).head()
```
- 对于整个索引的元素替换，可以利用迭代器实现。
```python
new_values = iter(list('abcdefgh'))
df_ex.rename(index=lambda x:next(new_values),
             level=2)
```
- 通过 `map` 函数修改索引。
```python
df_temp = df_ex.copy()
new_idx = df_temp.index.map(lambda x: (x[0],
                                       x[1],
                                       str.upper(x[2])))
df_temp.index = new_idx
```
- 多层索引转单层
```python
df_temp = df_ex.copy()
new_idx = df_temp.index.map(lambda x: (x[0]+'-'+
                                       x[1]+'-'+
                                       x[2]))
df_temp.index = new_idx
# 反向展开
new_idx = df_temp.index.map(lambda x:tuple(x.split('-')))
df_temp.index = new_idx
```
### 2.3.3 索引的设置与重置
- 构造一个新表举例说明：
```python
df_new = pd.DataFrame({'A':list('aacd'),
                       'B':list('PQRT'),
                       'C':[1,2,3,4]})
```
- 索引的设置可以使用 `set_index` 完成，这里的主要参数是 `append` ，表示是否来保留原来的索引，直接把新设定的添加到原索引的内层。
```python
df_new.set_index('A')
df_new.set_index('A', append=True)
df_new.set_index(['A', 'B'])                                # 同时指定多个列作为索引
```
- 如果想要添加索引的列没有出现在其中，那么可以直接在参数中传入相应的 `Series` 。
```python
my_index = pd.Series(list('WXYZ'), name='D')
df_new = df_new.set_index(['A', my_index])
```
- `reset_index` 是 `set_index` 的逆函数，其主要参数是 `drop` ，表示是否要把去掉的索引层丢弃，而不是添加到列中。
```python
df_new.reset_index(['D'])
df_new.reset_index(['D'], drop=True)
```
- 如果重置了所有的索引，那么 `pandas` 会直接重新生成一个默认索引。
### 2.3.4 索引的变形
- 给定一个新的索引，把原表中相应的索引对应元素填充到新索引构成的表中。
```python
df_reindex = pd.DataFrame({"Weight":[60,70,80],
                           "Height":[176,180,179]},
                           index=['1001','1003','1002'])
df_reindex.reindex(index=['1001','1002','1003','1004'],
                   columns=['Weight','Gender'])             # 原来表中的数据和新表中会根据索引自动对齐
```
- 与 `reindex` 功能类似的函数是 `reindex_like` ，其功能是仿照传入的表索引来进行被调用表索引的变形。
```python
df_existed = pd.DataFrame(index=['1001','1002','1003','1004'],
                          columns=['Weight','Gender'])
df_reindex.reindex_like(df_existed)
```

## 2.4 索引运算
### 2.4.1 一般的索引运算
利用集合运算来取出符合条件行，此时通过 `Index` 上的运算操作就很容易实现。
- 先用 `unique` 去重后再进行集合运算。
```python
df_set_1 = pd.DataFrame([[0,1],[1,2],[3,4]],
                        index = pd.Index(['a','b','a'],name='id1'))
df_set_2 = pd.DataFrame([[4,5],[2,6],[7,1]],
                        index = pd.Index(['b','b','c'],name='id2'))
id1, id2 = df_set_1.index.unique(), df_set_2.index.unique()
# 集合操作
id1.intersection(id2)
id1.union(id2)
id1.difference(id2)
id1.symmetric_difference(id2)
```
- 若两张表需要做集合运算的列并没有被设置索引，一种办法是先转成索引，运算后再恢复，另一种方法是利用 `isin` 函数。
```python
# 例如在重置索引的第一张表中选出id列交集的所在行
df_set_in_col_1 = df_set_1.reset_index()
df_set_in_col_2 = df_set_2.reset_index()
df_set_in_col_1[df_set_in_col_1.id1.isin(df_set_in_col_2.id2)]
```