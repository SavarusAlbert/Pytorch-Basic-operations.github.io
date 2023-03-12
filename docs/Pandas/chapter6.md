# Chapter06 缺失数据
## 6.1 缺失值的统计和删除
### 6.1.1 缺失信息的统计
- 缺失数据可以使用 `isna` 或 `isnull` （两个函数没有区别）来查看每个单元格是否缺失，结合 `mean` 可以计算出每列缺失值的比例。
```python
import numpy as np
import pandas as pd
df = pd.read_csv('data/learn_pandas.csv',
                 usecols = ['Grade', 'Name', 'Gender', 'Height',
                            'Weight', 'Transfer'])
df.isna()
df.isna().mean()                                            # 查看缺失的比例
```
```markup
Grade       0.000
Name        0.000
Gender      0.000
Height      0.085
Weight      0.055
Transfer    0.060
dtype: float64
```
- 如果想要查看某一列缺失或者非缺失的行，可以利用 `Series` 上的 `isna` 或者 `notna` 进行布尔索引。
```python
df[df.Height.isna()].head()
```
```markup
        Grade          Name  Gender  Height  Weight Transfer
3   Sophomore  Xiaojuan Sun  Female     NaN    41.0        N
12     Senior      Peng You  Female     NaN    48.0      NaN
26     Junior     Yanli You  Female     NaN    48.0        N
36   Freshman  Xiaojuan Qin    Male     NaN    79.0        Y
60   Freshman    Yanpeng Lv    Male     NaN    65.0        N
```
- 如果想要同时对几个列，检索出全部为缺失或者至少有一个缺失或者没有缺失的行，可以使用 `isna`, `notna` 和 `any`, `all` 的组合。`any` 一个序列中满足一个`True`，则返回`True`；`all` 一个序列中所有值为`True`时，返回`True`，否则为`False`。
```python
sub_set = df[['Height', 'Weight', 'Transfer']]
df[sub_set.isna().all(1)]                                   # axis=1 判断全部
df[sub_set.isna().any(1)]                                   # axis=1 判断存在
df[sub_set.notna().all(1)]                                  # axis=1 判断没有缺失
```
### 6.1.2 缺失信息的删除
- 数据处理中经常需要根据缺失值的大小、比例或其他特征来进行行样本或列特征的删除。
- `dropna` 的主要参数为轴方向 `axis` (默认为0，即删除行)、删除方式 `how` 、删除的非缺失值个数阈值 `thresh` (非缺失值 没有达到这个数量的相应维度会被删除)、备选的删除子集 `subset` ，其中 `how` 主要有 `any` 和 `all` 两种参数可以选择。
```python
df.dropna(how = 'any', subset = ['Height', 'Weight'])       # 删除身高体重至少有一个缺失的行
# df.loc[df[['Height', 'Weight']].notna().all(1)]           # 等价索引操作
df.dropna(1, thresh=df.shape[0]-15)                         # 删除超过15个缺失值的列
# df.loc[:, ~(df.isna().sum()>15)]                          # 等价索引操作
```

## 6.2 缺失值的填充和插值
### 6.2.1 利用fillna进行填充
- 在 `fillna` 中有三个参数是常用的： `value`, `method`, `limit` 。其中， `value` 为填充值，可以是标量，也可以是索引到元素的字典映射； `method` 为填充方法，有用前面的元素填充 `ffill` 和用后面的元素填充 `bfill` 两种类型， `limit` 参数表示连续缺失值的最大填充次数。
```python
# 构造样例
s = pd.Series([np.nan, 1, np.nan, np.nan, 2, np.nan],
               list('aaabcd'))
```
```markup
a    NaN
a    1.0
a    NaN
b    NaN
c    2.0
d    NaN
dtype: float64
```
```python
s.fillna(method='ffill')                                    # 用前面的值向后填充
s.fillna(method='ffill', limit=1)                           # 连续出现的缺失，最多填充一次
s.fillna(s.mean())                                          # 由s.mean()填充
s.fillna({'a': 100, 'd': 200})                              # 通过索引映射填充的值
df.groupby('Grade')['Height'].transform(
                     lambda x: x.fillna(x.mean())).head()   # 先分组，再根据年级进行身高缺失值的填充
```
### 6.2.2 插值函数
- `interpolate` 插值函数，插值方法(默认为 `linear` 线性插值)，控制方向的 `limit_direction`(默认为前向限制插值 `forward`，后向限制插值或者双向限制插值可以指定为 `backward` 或 `both`) ，`limit` 控制最大连续缺失值插值个数。
```python
# 构造样例
s = pd.Series([np.nan, np.nan, 1,
               np.nan, np.nan, np.nan,
               2, np.nan, np.nan]
# array([nan, nan,  1., nan, nan, nan,  2., nan, nan])
s.interpolate(limit_direction='backward', limit=1).values   # 后向线性插值，最大插值个数1
# array([ nan, 1.  , 1.  ,  nan,  nan, 1.75, 2.  ,  nan,  nan])
s.interpolate(limit_direction='both', limit=1).values       # 双向线性插值，最大插值个数1
# array([ nan, 1.  , 1.  , 1.25,  nan, 1.75, 2.  , 2.  ,  nan])
```
- 最近邻插值，缺失值的元素和离它最近的非缺失值元素一样。
```python
s.interpolate('nearest').values
# array([nan, nan,  1.,  1.,  1.,  2.,  2., nan, nan])
```
- 索引插值，即根据索引大小进行线性插值(对于时间戳索引也可以使用)。
```python
# 构造不等间距的索引样例
s = pd.Series([0,np.nan,10],index=[0,1,10])
s.interpolate(method='index')
```
```markup
0      0.0
1      1.0
10    10.0
dtype: float64
```

## 6.3 Nullable类型
### 6.3.1 缺失记号及其缺陷
- 在 `python` 中的缺失值用 `None` 表示，该元素除了等于自己本身之外，与其他任何元素不相等。
- 在 `numpy` 中利用 `np.nan` 来表示缺失值，该元素除了不和其他任何元素相等之外，和自身的比较结果也返回 `False` 。
- 在使用 `equals` 函数进行两张表或两个序列的相同性检验时，会自动跳过两侧表都是缺失值的位置，直接返回 `True` 。
#### pd.Na
- 在 `pandas` 中可以看到 `object` 类型的对象，而 `object` 是一种混杂对象类型，如果出现了多个类型的元素同时存储在 `Series` 中，它的类型就会变成 `object` 。
-  `np.nan` 的本身是一种浮点类型，而如果浮点和时间类型混合存储，如果不设计新的内置缺失类型来处理，就会变成含糊不清的 `object` 类型，这显然是不希望看到的。同时，由于 `np.nan` 的浮点性质，如果在一个整数的 `Series` 中出现缺失，那么其类型会转变为 `float64` ；而如果在一个布尔类型的序列中出现缺失，那么其类型就会转为 `object` 而不是 `bool` 。
-  因此，`pandas` 尝试设计了一种新的缺失类型 `pd.NA` 以及三种 `Nullable` 序列类型来应对这些缺陷，它们分别是 `Int`, `boolean` 和 `string` 。
- 在时间序列的对象中， `pandas` 利用 `pd.NaT` 来指代缺失值。
### 6.3.2 Nullable类型的性质
- 从字面意义上看 `Nullable` 就是可空的，言下之意就是序列类型不受缺失值的影响。例如，在上述三个 `Nullable` 类型中存储缺失值，都会转为 `pandas` 内置的 `pd.NA` 。
```python
pd.Series([np.nan, 1], dtype = 'Int64')
```
```markup
0    <NA>
1       1
dtype: Int64
```
```python
pd.Series([np.nan, True], dtype = 'boolean')
```
```markup
0    <NA>
1    True
dtype: boolean
```
```python
pd.Series([np.nan, 'my_str'], dtype = 'string')
```
```markup
0      <NA>
1    my_str
dtype: string
```
#### Int类型
- 在 Int 的序列中，返回的结果会尽可能地成为 Nullable 的类型。
```python
pd.Series([np.nan, 0], dtype = 'Int64') + 1
```
```markup
0    <NA>
1       1
dtype: Int64
```
```python
pd.Series([np.nan, 0], dtype = 'Int64') == 0
```
```markup
0    <NA>
1    True
dtype: boolean
```
```python
pd.Series([np.nan, 0], dtype = 'Int64') * 0.5               # 只能是浮点
```
```markup
0    <NA>
1     0.0
dtype: Float64
```
#### boolean类型
对于 `boolean` 类型的序列而言，其和 `bool` 序列的行为主要有两点区别。
- 第一点是带有缺失的布尔列表无法进行索引器中的选择，而 `boolean` 会把缺失值看作 `False` 。
```python
s = pd.Series(['a', 'b'])
s_bool = pd.Series([True, np.nan])
s_boolean = pd.Series([True, np.nan]).astype('boolean')
# s[s_bool]  会报错
s[s_boolean]
```
```markup
0    a
dtype: object
```
- 第二点是在进行逻辑运算时， `bool` 类型在缺失处返回的永远是 `False` ，而 `boolean` 会根据逻辑运算是否能确定唯一结果来返回相应的值。
- `True | pd.NA` 中无论缺失值为什么值，必然返回 `True` ； `False | pd.NA` 中的结果会根据缺失值取值的不同而变化，此时返回 `pd.NA` ； `False & pd.NA` 中无论缺失值为什么值，必然返回 `False` 。
```python
s_boolean & True
```
```markup
0    True
1    <NA>
dtype: boolean
```
```python
s_bool & True
```
```markup
0     True
1    False
dtype: bool
```
#### 类型转换
- 一般在实际数据处理时，可以在数据集读入后，先通过 `convert_dtypes` 转为 `Nullable` 类型。
```python
df = pd.read_csv('data/learn_pandas.csv')
df = df.convert_dtypes()
print(df.dtypes)
```
```markup
School          string
Grade           string
Name            string
Gender          string
Height         Float64
Weight           Int64
Transfer        string
Test_Number      Int64
Test_Date       string
Time_Record     string
dtype: object
```
### 6.3.3 缺失数据的计算和分组
- 当调用函数 `sum`, `prod` 使用加法和乘法的时候，缺失数据等价于被分别视作`0`和`1`，即不改变原来的计算结果。
```python
s = pd.Series([2,3,np.nan,4,5])
s.sum()                                                     # 14.0
s.prod()                                                    # 120.0
```
- 当使用累计函数时，会自动跳过缺失值所处的位置。
```python
s.cumsum()
```
```markup
0     2.0
1     5.0
2     NaN
3     9.0
4    14.0
dtype: float64
```
- 当进行单个标量运算的时候，除了 `np.nan ** 0` 和 `1 ** np.nan` 这两种情况为确定的值之外，所有运算结果全为缺失(`pd.NA` 的行为与此一致)，并且 `np.nan` 在比较操作时一定返回 `False` ，而 `pd.NA` 返回 `pd.NA` 。
```python
np.nan == 0                                                 # False
pd.NA == 0                                                  # <NA>
np.nan > 0                                                  # False
pd.NA > 0                                                   # <NA>
np.nan + 1                                                  # nan
np.log(np.nan)                                              # nan
np.add(np.nan, 1)                                           # nan
np.nan ** 0                                                 # 1.0
pd.NA ** 0                                                  # 1
1 ** np.nan                                                 # 1.0
1 ** pd.NA                                                  # 1
```
- 对于一些函数而言，缺失可以作为一个类别处理，例如在 `groupby`, `get_dummies` 中可以设置相应的参数来进行增加缺失类别。
```python
# 构造样例
df_nan = pd.DataFrame({'category':['a','a','b',np.nan,np.nan],
                       'value':[1,3,5,7,9]})
df_nan.groupby('category',
                dropna=False)['value'].mean()
```
```markup
category
a      2
b      5
NaN    8
Name: value, dtype: int64
```
```python
pd.get_dummies(df_nan.category, dummy_na=True)
```
```markup
   a  b  NaN
0  1  0    0
1  1  0    0
2  0  1    0
3  0  0    1
4  0  0    1
```
```python
pd.get_dummies(df_nan.category)
```
```markup
   a  b
0  1  0
1  1  0
2  0  1
3  0  0
4  0  0
```