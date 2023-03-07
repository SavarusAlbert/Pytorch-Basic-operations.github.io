# Chapter02 Numpy数组的基本操作
## 2.1 切片
- `Numpy`数组的切片操作与 Python 的切片操作一致，但`Numpy`数组切片操作返回的对象只是原数组的视图(view)。
- 在 `Numpy` 中，所有赋值运算不会为数组和数组中的任何元素创建副本。
- `numpy.ndarray.copy()` 函数创建一个副本(copy)。 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
## 2.2 数组迭代
- 使用 `apply_along_axis(func1d, axis, arr)` 代替`for`循环：
```python
def my_func(x):
    return (x[0] + x[-1]) * 0.5
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.apply_along_axis(my_func, 0, x)
print(y)
```
```markup
[21. 22. 23. 24. 25.]
```
## 2.3 数组操作
- 更改形状
```python
import numpy as np
# numpy.ndarray.shape表示数组的维度
x = np.array(range(10))
x.shape                                                     # (10,)
# numpy.ndarray.flat 将数组转换为一维的迭代器，可以用for访问数组每一个元素
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y1 = x.flat
for i in y1:
    print(i, end=' ')
# numpy.ndarray.flatten([order='C']) 将数组的副本转换为一维数组，并返回
# order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序
y2 = x.flatten()
y3 = x.flatten(order='F')
# numpy.ndarray.ravel([order='C']) 与 flatten 函数用法相同，但当 'order'=C 时返回的是原数组的视图(view)
y4 = np.ravel(x)                                            # 视图
y5 = np.ravel(x, order='F')                                 # 副本
# numpy.reshape(a, newshape[, order='C'])在不更改数据的情况下为数组赋予新的形状
x = np.arange(12)
y = np.reshape(x, [3, 4])
y = np.reshape(x, [3, -1])
# 当参数newshape = -1时，表示将数组降为一维
x = np.random.randint(12, size=[2, 2, 3])
y = np.reshape(x, -1)
```
- 数组转置
```python
# numpy.ndarray.T 对数组进行转置
x = np.random.randint(1, 100, [2, 2, 3])
y = x.T
# numpy.transpose(a, axes=None) 按轴转置，axes = None 时与 numpy.ndarray.T 一样
x = np.ones((1, 2, 3))
np.transpose(x, (1, 0, 2)).shape                            # (2, 1, 3)
```
- 更改维度
```python
# 通过numpy.newaxis = None增加数组维度
x = np.array([1, 2, 9, 4, 5, 6, 7, 8])                      # shape为(8,)
y1 = x[np.newaxis, :]                                       # shape为(1, 8)
y2 = x[:, np.newaxis]                                       # shape为(8, 1)
# 多维数组在画图时有可能因为某些维度为1而无法画图，此时降维操作就有了意义
# numpy.squeeze(a, axis=None) 从数组中删除单维(axis选多维会报错)
x = np.array([[[0], [1], [2]]])                             # shape为(1, 3, 1)
y = np.squeeze(x)                                           # shape为(3,)
```
- 数组拼接
```python
# numpy.concatenate((a1, a2, ...), axis=0, out=None) 将多个数组在指定维度拼接
x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.concatenate([x, y])                                  # [[ 1  2  3] [ 7  8  9]]
z = np.concatenate([x, y], axis=1)                          # [[ 1  2  3  7  8  9]]
# numpy.stack(arrays, axis=0, out=None)，沿着新添加的轴拼接
x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.stack([x, y])                                        # [[1 2 3] [7 8 9]]
z = np.stack([x, y], axis=1)                                # [[1 7] [2 8] [3 9]]
# numpy.hstack(),numpy.vstack()分别表示水平和竖直的拼接方式。在数据维度等于1时，比较特殊。而当维度大于或等于2时，它们的作用相当于concatenate，用于在已有轴上进行操作。
```
- 数组拆分
    - `numpy.split(ary, indices_or_sections, axis=0)` ，`ary`是待拆分的数组
    - `indices_or_sections`的类型为`int`或者一维数组，如果是一个整数的话，就用这个数平均分割原数组；如果是一个数组的话，就以数组中的数字为索引拆分
    - 垂直拆分numpy.vsplit(ary, indices_or_sections)
    - 水平拆分numpy.hsplit(ary, indices_or_sections)
```python
x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.split(x, [1, 3])
```
```markup
[array([[11, 12, 13, 14]]),
 array([[16, 17, 18, 19],
        [21, 22, 23, 24]]),
 array([], shape=(0, 4), dtype=int32)]
```
- 数组平铺
    - `numpy.tile(A, reps)`将原矩阵横向、纵向地复制
    - `numpy.repeat(a, repeats, axis=None)`将数或矩阵沿axis轴重复，repeats可以是数或矩阵
```python
x = np.array([[1, 2], [3, 4]])
y = np.tile(x, (1, 3))
print(y)
```
```markup
[[1 2 1 2 1 2]
 [3 4 3 4 3 4]]
```
```python
x = np.array([[1, 2], [3, 4]])
y1 = np.repeat(x, 2)                                        # [1 1 2 2 3 3 4 4]
y2 = np.repeat(x, [2, 3], axis=0)                           # [[1, 2],[1, 2],[3, 4],[3, 4],[3, 4]]
```
- 添加或删除元素
    - `numpy.unique(ar, return_index=False, return_inverse=False,return_counts=False, axis=None)` 去重并排序
    - `return_index=True`表示返回新列表元素在旧列表中的位置，并以列表形式储存
    - `return_inverse=True`表示返回旧列表元素在新列表中的位置，并以列表形式储存
    - `return_counts=True`表示返回新列表元素在旧列表中的个数，并以列表形式储存
```python
a=np.array([1,1,2,3,3,4,4])
b=np.unique(a,return_index=True, return_inverse=True,return_counts=True)
print(b)
```
```markup
(array([1, 2, 3, 4]),
 array([0, 2, 3, 5], dtype=int64),
 array([0, 0, 1, 2, 2, 3, 3], dtype=int64),
 array([2, 1, 2, 2], dtype=int64))
```
## 2.4 数学函数
### 2.4.1 广播机制
- 广播机制的规则有三个：
    - 如果两个数组的维度(dim)不相同，那么小维度数组的形状将会在左边补1。
    - 如果shape维度不匹配，但是有维度是1，那么可以扩展维度是1的维度匹配另一个数组；
    - 如果shape维度不匹配，但是没有任何一个维度是1，则匹配引发错误；
```python
# 二维数组加一维数组
x = np.arange(4)
y = np.ones((3, 4))
print((x+y).shape)                                          # (3, 4)
# 两个数组均需要广播
x = np.arange(4).reshape(4, 1)
y = np.ones(5)
print((x+y).shape)                                          # (4, 5)
```
### 2.4.2 数学函数
在广播机制的作用下，向量间的数学运算能够适应维度进行扩展。
- 简单函数
    - `numpy.add(x1, x2, *args, **kwargs)` 逐元素相加
    - `numpy.subtract(x1, x2, *args, **kwargs)` 逐元素相减
    - `numpy.multiply(x1, x2, *args, **kwargs)` 逐元素相除
    - `numpy.divide(x1, x2, *args, **kwargs)` 逐元素返回真除数
    - `numpy.floor_divide(x1, x2, *args, **kwargs)` 逐元素返回地板除结果
    - `numpy.power(x1, x2, *args, **kwargs)` 逐元素幂次
    - `numpy.sqrt(x, *args, **kwargs)` 返回非负平方根
    - `numpy.square(x, *args, **kwargs)` 返回逐元素平方
```python
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.arange(1, 6)
np.add(x, y) == x + y
np.subtract(x, y) == x - y
np.multiply(x, y) == x * y
np.divide(x, y) == x / y
np.floor_divide(x, y) == x // y
np.power(x, np.full([1, 5], 2)) == x ** np.full([1, 5], 2)
```
- 三角函数(逐元素运算)
    - `numpy.sin(x, *args, **kwargs)`
    - `numpy.cos(x, *args, **kwargs)`
    - `numpy.tan(x, *args, **kwargs)`
    - `numpy.arcsin(x, *args, **kwargs)`
    - `numpy.arccos(x, *args, **kwargs)`
    - `numpy.arctan(x, *args, **kwargs)`
- 指数和对数
    - `numpy.exp(x, *args, **kwargs)`
    - `numpy.log(x, *args, **kwargs)`
    - `numpy.exp2(x, *args, **kwargs)`
    - `numpy.log2(x, *args, **kwargs)`
    - `numpy.log10(x, *args, **kwargs)`
- 加法和累加
    - `numpy.sum(a[, axis=None, dtype=None, out=None, …])` 求和，如果设置axis则对特定轴求和
    - `numpy.cumsum(a, axis=None, dtype=None, out=None)` 逐元素返回累加和，设置轴则对特定轴进行逐元素累加和
```python
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y1 = np.sum(x)                                              # 575
y2 = np.sum(x, axis=0)                                      # [105 110 115 120 125]
y3 = np.cumsum(x)                                           # [ 11  23  36  50  65  81  98 116 135 155 176 198 221 245 270 296 323 351 380 410 441 473 506 540 575]
y4 = np.cumsum(x, axis=0)                                   # [[ 11  12  13  14  15] [ 27  29  31  33  35] [ 48  51  54  57  60] [ 74  78  82  86  90] [105 110 115 120 125]]
```
- 乘积和累乘
    - `numpy.prod(a[, axis=None, dtype=None, out=None, …])` 乘积，如果设置axis则对特定轴求积
    - `numpy.cumprod(a, axis=None, dtype=None, out=None)` 逐元素返回累乘，设置轴则对特定轴进行逐元素累乘
```python
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y1 = np.prod(x)                                             # 788529152
y2 = np.prod(x, axis=0)                                     # [2978976 3877632 4972968 6294624 7875000]
y3 = np.cumprod(x)
y4 = np.cumprod(x, axis=0)
```
- 离散差
    - `numpy.diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue)` 逐轴作差out[i] = a[i+1] - a[i]
```python
A = np.arange(2, 14).reshape((3, 4))
A[1, 1] = 8
np.diff(A)                                                  # [[1 1 1] [2 0 1] [1 1 1]]
np.diff(A, axis=0)                                          # [[4 5 4 4] [4 3 4 4]]
```
- 四舍五入
    - `numpy.around(a, decimals=0, out=None)` 四舍五入
    - `numpy.ceil(x, *args, **kwargs)` 向上取整
    - `numpy.floor(x, *args, **kwargs)` 向下取整
- 裁剪
    - `numpy.clip(a, a_min, a_max, out=None, **kwargs)` 裁剪至最大最小值
```python
x = np.array([11, 17, 23, 29, 35])
y = np.clip(x, a_min=20, a_max=30)                          # [20, 20, 23, 29, 30]
```
- 绝对值
    - `numpy.absolute(x, *args, **kwargs)` 或 `numpy.abs(x, *args, **kwargs)`
- 返回符号
    - `numpy.sign(x, *args, **kwargs)`
### 2.4.3 逻辑函数
- 真值测试
    - `numpy.all(a, axis=None, out=None, keepdims=np._NoValue)` 沿axis轴返回是否全为真
    - `numpy.any(a, axis=None, out=None, keepdims=np._NoValue)` 沿axis轴返回是否存在真
    - `numpy.isnan(x, *args, **kwargs)` 返回数组各元素是否为Nan
- 逻辑运算
    - `numpy.logical_not(x, *args, **kwargs)` 逻辑非x
    - `numpy.logical_and(x1, x2, *args, **kwargs)` x1、x2逻辑和
    - `numpy.logical_or(x1, x2, *args, **kwargs)` x1、x2逻辑或
    - `numpy.logical_xor(x1, x2, *args, **kwargs)` x1、x2逻辑异或
- 比较运算
    - `numpy.greater(x1, x2, *args, **kwargs)` 逐元素比较x1 > x2
    - `numpy.greater_equal(x1, x2, *args, **kwargs)` 逐元素比较x1 >= x2
    - `numpy.equal(x1, x2, *args, **kwargs)` 逐元素比较x1 == x2
    - `numpy.not_equal(x1, x2, *args, **kwargs)` 逐元素比较x1 != x2
    - `numpy.less(x1, x2, *args, **kwargs)` 逐元素比较x1 < x2
    - `numpy.less_equal(x1, x2, *args, **kwargs)` 逐元素比较x1 <= x2
- 误差内判断是否相等，判断式：`np.absolute(a - b) <= (atol + rtol * absolute(b))`
    - `numpy.isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)` 逐元素判断是否在误差范围内相等
    - `numpy.allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)` 判断所有元素是否在误差范围内
    - `equal_nan=True` 则Nan==Nan
## 2.5 排序，搜索和计数
### 2.5.1 排序
- `numpy.sort(a[, axis=-1, kind='quicksort', order=None])` 返回copy的排序后数组
    - axis：排序沿数组的(轴)方向，0表示按行，1表示按列，None表示展开来排序，默认为-1，表示沿最后的轴排序。
    - kind：排序的算法，提供了快排'quicksort'、混排'mergesort'、堆排'heapsort'， 默认为‘quicksort'。
    - order：排序的字段名，可指定字段排序，默认为None。
- `numpy.argsort(a[, axis=-1, kind='quicksort', order=None])` 返回排序后原数组的index组成的数组
- `numpy.take(a, indices, axis=None, out=None, mode='raise')` 沿轴从数组中获取元素
```python
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
y1 = np.array([np.take(x[i], np.argsort(x[i])) for i in range(5)])
y2 = np.sort(x)
y1 == y2
```
- `numpy.lexsort(keys[, axis=-1])` 使用键序列执行间接稳定排序(给定多个可以在电子表格中解释为列的排序键，lexsort返回一个整数索引数组，该数组描述了按多个列排序的顺序。序列中的最后一个键用于主排序顺序，倒数第二个键用于辅助排序顺序，依此类推)
- `keys`参数必须是可以转换为相同形状的数组的对象序列。如果为keys参数提供了2D数组，则将其行解释为排序键，并根据最后一行，倒数第二行等进行排序。
```python
# 按照第一列的升序或者降序对整体数据进行排序
np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
index = np.lexsort([x[:, 0]])
y = x[index]
```
- `numpy.partition(a, kth, axis=-1, kind='introselect', order=None)` 以索引是 kth 的元素为基准，将元素分成两部分，即大于该元素的放在其后面，小于该元素的放在其前面，这里有点类似于快排。
```python
x = np.array([5, 1, 6, 7, 8, 4])
y = np.partition(x, kth=2, axis=0)                          # [1, 4, 5, 7, 8, 6]
```
- `numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)` 返回partition后的index数组
### 2.5.2 索引
- `numpy.argmax(a[, axis=None, out=None])` 返回axis轴的最大值的index
- `numpy.argmin(a[, axis=None, out=None])` 返回axis轴的最小值的index
- `numppy.nonzero(a)` 返回非零元素的索引
```python
x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
y = np.nonzero(x)                                           # [[0 1 2 2] [0 1 0 1]]
x[y]                                                        # [3, 4, 5, 6] # 取x[索引]
np.transpose(y)                                             # [[0, 0], [1, 1], [2, 0], [2, 1]] 转换轴，得到坐标
```
- `numpy.where(condition, [x=None, y=None])` 满足condition，输出x，不满足输出y，若没有x、y，则返回满足condition的元素index，`np.transpose()`后为坐标
```python
x = np.arange(10)                                           # [0 1 2 3 4 5 6 7 8 9]
y = np.where(x < 5, x, 10 * x)                              # [0 1 2 3 4 50 60 70 80 90]
z = np.where(x > 7)                                         # [8, 9]
```
- `numpy.searchsorted(a, v[, side='left', sorter=None])` 返回v插入到数组a的index，a非升序时，sorter非空，存放a中升序index，side为从左或从右插入
```python
x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
y1 = np.searchsorted(x, [-1, 0, 11, 15, 33, 35])                    # [0 0 4 5 7 8]
y2 = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right')      # [0 1 5 5 8 8]
```
### 2.5.3 计数
- `numpy.count_nonzero(a, axis=None)` 计数数组中非零元素个数
```python
x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])           # 5
x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=1)   # [2 3]
```
## 2.6 集合操作
- `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)` 找出数组中唯一值并返回已排序结果
    - return_index=True 表示返回新列表元素在旧列表中的位置。
    - return_inverse=True表示返回旧列表元素在新列表中的位置。
    - return_counts=True表示返回新列表元素在旧列表中出现的次数
```python
x = np.array([[1, 1], [2, 3]])
y = np.unique(x)                                            # [1 2 3]
x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
y = np.unique(x, axis=0)                                    # [[1 0 0] [2 3 4]]
x = np.array(['a', 'b', 'b', 'c', 'a'])
y, index, inverse, counts = np.unique(x, return_index=True, return_inverse=True, return_counts=True)
# y = ['a', 'b', 'c'], index = [0, 1, 3], inverse = [0, 1, 1, 2, 0], count = ([2, 2, 1]
```
- `numpy.in1d(ar1, ar2, assume_unique=False, invert=False)` 前面的数组是否包含于后面的数组，返回布尔值，返回的值是针对第一个参数的数组的，所以维数和第一个参数一致，布尔值与数组的元素位置也一一对应
```python
test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)                                # [True False True False True]
test[mask]                                                  # [0 2 0]
mask = np.in1d(test, states, invert=True)                   # [False True False True False]
```
- `numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)` 返回两数组交集
```python
x = np.array([1, 1, 2, 3, 4])
y = np.array([2, 1, 4, 6])
xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
# xy = [1 2 4], x_ind = [0 2 4], y_ind = [1 0 2]
from functools import reduce
reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))  # [3]
```
- `numpy.union1d(ar1, ar2)` 返回两数组并集，唯一化并排序
```python
x = np.union1d([-1, 0, 1], [-2, 0, 2])                              # [-2 -1  0  1  2]
from functools import reduce
x = reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))  # [1 2 3 4 6]
```
- `numpy.setdiff1d(ar1, ar2, assume_unique=False` 集合的差，即元素存在于ar1不存在于ar2
```python
x = np.array([1, 2, 3, 2, 4, 1])
y = np.array([3, 4, 5, 6])
z = np.setdiff1d(a, b)                                      # [1 2]
```
- `numpy.setxor1d(ar1, ar2, assume_unique=False)` 集合的对称差，即两个集合的交集的补集。简言之，就是两个数组中各自独自拥有的元素的集合
```python
x = np.array([1, 2, 3, 2, 4, 1])
y = np.array([3, 4, 5, 6])
z = np.setxor1d(a, b)                                       # [1 2 5 6]
```