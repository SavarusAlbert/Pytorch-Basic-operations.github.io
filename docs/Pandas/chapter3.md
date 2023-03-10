# Chapter03 分组
## 3.1 分组模式及其对象
### 3.1.1 分组的一般模式
- 想要实现分组操作，必须明确三个要素：分组依据、数据来源、操作及其返回结果。
- 分组代码的一般模式：`df.groupby(分组依据)[数据来源].使用操作`
```python
import numpy as np
import pandas as pd
df = pd.read_csv('data/learn_pandas.csv')
# 按照性别统计身高中位数
df.groupby('Gender')['Height'].median()
```
```markup
Gender
Female    159.6
Male      173.4
Name: Height, dtype: float64
```
### 3.1.2 分组依据的本质
- 根据多个维度进行分组，只需在 groupby 中传入相应列名构成的列表即可。
```python
df.groupby(['School', 'Gender'])['Height'].mean()
```
- 按照复杂逻辑来分组。
```python
condition = df.Weight > df.Weight.mean()
df.groupby(condition)['Height'].mean()
```
- 传入列名只是一种简便的记号，事实上等价于传入的是一个或多个列，最后分组的依据来自于数据来源组合的unique值。
```python
# 查看unique组类别
df[['School', 'Gender']].drop_duplicates()
# 按照唯一组类别进行分组
df.groupby([df['School'], df['Gender']])['Height'].mean()
```
### 3.1.3 Groupby对象
- 分组调用的方法都来自于 `pandas` 中的 `groupby` 对象，对象上定义了许多方法，也具有一些方便的属性。
```python
gb = df.groupby(['School', 'Grade'])
gb.ngroups                                                  # 分组个数
gb.groups.keys()                                            # 返回组索引列表
gb.size()                                                   # 统计每个组的元素个数
gb.get_group(('Fudan University', 'Freshman'))              # 获取所在组对应的行
```
## 3.2 聚合操作
### 3.2.1 内置聚合函数
`max/min/mean/median/count/all/any/idxmax/idxmin/mad/nunique/skew/quantile/sum/std/var/sem/size/prod`
```python
gb = df.groupby('Gender')['Height']
gb.idxmin()
gb.quantile(0.95)
df.groupby('Gender')[['Height', 'Weight']].max()            # 包含多个列时，将按照列进行迭代计算
```
### 3.2.2 `agg`方法
#### 使用多个函数
- 当使用多个聚合函数时，需要用列表的形式把内置聚合函数对应的字符串传入，先前提到的所有字符串都是合法的。
```python
gb = df.groupby('Gender')[['Height', 'Weight']]
gb.agg(['sum', 'idxmax', 'skew'])
```
```markup
         Height                   Weight                 
            sum idxmax      skew     sum idxmax      skew
Gender                                                   
Female  21014.0     28 -0.219253  6469.0     28 -0.268482
Male     8854.9    193  0.437535  3929.0      2 -0.332393
```
#### 对特定的列使用特定的聚合函数
- 对于方法和列的特殊对应，可以通过构造字典传入 `agg` 中实现，其中字典以列名为键，以聚合字符串或字符串列表为值。
```python
gb.agg({'Height':['mean','max'], 'Weight':'count'})
```
```markup
           Height        Weight
             mean    max  count
Gender                         
Female  159.19697  170.2    135
Male    173.62549  193.9     54
```
#### 使用自定义函数
- 在 `agg` 中可以使用具体的自定义函数， 需要注意传入函数的参数是之前数据源中的列，逐列进行计算。
```python
gb.agg(lambda x: x.max()-x.min())
```
```markup
        Height  Weight
Gender                
Female    24.8    29.0
Male      38.2    38.0
```
- 由于传入的是序列，因此序列上的方法和属性都是可以在函数中使用的，只需保证返回值是标量即可。
```python
def my_func(s):
    res = 'High'
    if s.mean() <= df[s.name].mean():
        res = 'Low'
    return res
gb.agg(my_func)
```
#### 聚合结果重命名
- 如果想要对聚合结果的列名进行重命名，只需要将上述函数的位置改写成元组，元组的第一个元素为新的名字，第二个位置为原来的函数，包括聚合字符串和自定义函数。
```python
gb.agg([('range', lambda x: x.max()-x.min()), ('my_sum', 'sum')])
gb.agg({'Height': [('my_func', my_func), 'sum'],
        'Weight': lambda x:x.max()})
gb.agg({'Height': [('my_func', my_func), 'sum'],
        'Weight': [('range', lambda x:x.max())]})
```



## 3.3 变换操作
- 变换函数的返回值为同长度的序列，最常用的内置变换函数是累计函数：`cumcount/cumsum/cumprod/cummax/cummin` ，它们的使用方式和聚合函数类似，只不过完成的是组内累计操作。
```python
gb.cummax()
```
- 当用自定义变换时需要使用 `transform` 方法，被调用的自定义函数， **其传入值为数据源的序列** ，与 `agg` 的传入类型是一致的，其最后的返回结果是行列索引与数据源一致的 `DataFrame`
```python
gb.transform(lambda x: (x-x.mean())/x.std())
```
- 前面提到了 `transform` 只能返回同长度的序列，但事实上还可以返回一个标量，这会使得结果被广播到其所在的整个组，这种 标量广播 的技巧在特征工程中是非常常见的。
```python
gb.transform('mean')
```

## 3.4 组索引与过滤
- 组过滤作为行过滤(行索引可以看作行过滤)的推广，指的是如果对一个组的全体所在行进行统计的结果返回 `True` 则会被保留， `False` 则该组会被过滤，最后把所有未被过滤的组其对应的所在行拼接起来作为 `DataFrame` 返回。
- 在 `groupby` 对象中，定义了 `filter` 方法进行组的筛选，其中自定义函数的输入参数为数据源构成的 `DataFrame` 本身，所有表方法和属性都可以在自定义函数中相应地使用，同时只需保证自定义函数的返回为布尔值即可。
```python
# 通过过滤得到所有容量大于100的组
gb.apply(lambda x:x.shape)
gb.filter(lambda x: x.shape[0] > 100)
```

## 3.5 跨列分组
- `apply` 的自定义函数传入参数与 `filter` 完全一致，只不过后者只允许返回布尔值。
```python
# 分组计算组BMI=Weight/Height^2 的均值，其中体重和身高的单位分别为千克和米
def BMI(x):
    Height = x['Height']/100
    Weight = x['Weight']
    BMI_value = Weight/Height**2
    return BMI_value.mean()
gb.apply(BMI)
```
```markup
Gender
Female    18.860930
Male      24.318654
dtype: float64
```
### 3.5.1 apply函数
- `apply` 方法可以返回标量、一维 `Series` 和二维 `DataFrame`。
####  返回标量
- 结果得到的是 `Series` ，索引与 `agg` 的结果一致。
```python
gb = df.groupby(['Gender','Test_Number'])[['Height','Weight']]
gb.apply(lambda x: 0)
gb.apply(lambda x: [0, 0])
```
#### 返回Series
- 结果得到的是 `DataFrame` ，行索引与标量情况一致，列索引为 `Series` 的索引。
```python
gb.apply(lambda x: pd.Series([0,0],index=['a','b']))
```
```markup
                    a  b
Gender Test_Number      
Female 1            0  0
       2            0  0
       3            0  0
Male   1            0  0
       2            0  0
       3            0  0
```
#### 返回DataFrame
- 结果得到的是 `DataFrame` ，行索引最内层在每个组原先 `agg` 的结果索引上，再加一层返回的 `DataFrame` 行索引，同时分组结果 `DataFrame` 的列索引和返回的 `DataFrame` 列索引一致。
```python
gb.apply(lambda x: pd.DataFrame(np.ones((2,2)),
                                index = ['a','b'],
                                columns=pd.Index([('w','x'),('y','z')])))
```
```markup
                        w    y
                        x    z
Gender Test_Number            
Female 1           a  1.0  1.0
                   b  1.0  1.0
       2           a  1.0  1.0
                   b  1.0  1.0
       3           a  1.0  1.0
                   b  1.0  1.0
Male   1           a  1.0  1.0
                   b  1.0  1.0
       2           a  1.0  1.0
                   b  1.0  1.0
       3           a  1.0  1.0
                   b  1.0  1.0
```
### 3.5.2 性能
- 最后需要强调的是， `apply` 函数的灵活性是以牺牲一定性能为代价换得的，除非需要使用跨列处理的分组处理，否则应当使用其他专门设计的 `groupby` 对象方法，否则在性能上会存在较大的差距。同时，在使用聚合函数和变换函数时，也应当优先使用内置函数，它们经过了高度的性能优化，一般而言在速度上都会快于用自定义函数来实现。