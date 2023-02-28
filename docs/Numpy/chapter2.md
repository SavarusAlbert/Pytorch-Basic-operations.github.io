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