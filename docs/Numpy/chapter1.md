# Chapter01 Numpy数组的创建
## 1.1 常量
```python
import numpy as np
a, b, c, d = np.nan, np.inf, np.pi, np.e                    # 空值、无穷大、圆周率、自然常数
```
- 判断数组是否为空
```python
x = np.array([1, 1, 8, np.nan, 10])
y = np.isnan(x)                                             # 返回 y = [False False False True False]
z = np.count_nonzero(y)                                     # 返回 z = 1
```
```python
# 两个np.nan不相等
print(np.nan == np.nan)                                     # False
```

## 1.2 数组的创建
- `numpy` 提供的最重要的数据结构是 `ndarray`，它是 Python 中`list`的扩展。
- 通过`array()`函数进行创建
```python
# 创建一维数组
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
# 创建二维数组
c = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
# 创建三维数组
d = np.array([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
```
- 通过`asarray()`函数进行创建(与`array()`相同)
```python
a = np.asarray([0, 1, 2, 3, 4])
...
d = np.asarray([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
```
- `array()`和`asarray()`主要区别就是当数据源是 `ndarray` 时，`array()`仍然会 `copy` 出一个副本，占用新的内存，但不改变 `dtype` 时 `asarray()`不会。
```python
x = np.array([1, 2, 3])
y = np.array(x)
z = np.asarray(x)
w = np.asarray(x, dtype=np.float64)
a = [y == z] + [y == w]
x[2] = 1
b = [y == z] + [y == w]
print(a, b)
```
```markup
[array([ True,  True,  True]), array([ True,  True,  True])] [array([ True,  True, False]), array([ True,  True,  True])]
```
- 通过`fromfunction()`函数进行创建
```python
def f(x, y):
    return 10 * x + y
x = np.fromfunction(f, (5, 4), dtype=int)
print(x)
```
```markup
[[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]
```
```python
x = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
print(x)
```
```markup
[[0 1 2]
 [1 2 3]
 [2 3 4]]
```

## 1.3 填充数组
- 全0数组、全1数组、空数组、全m数组填充方式
```python
# 全0数组
x = np.zeros(5)
x = np.zeros([2, 3])
y = np.zeros_like(np.array(range(4)))
# 全1数组
x = np.ones(5)
x = np.ones([2, 3])
y = np.ones_like(np.array(range(4)))
# 空数组
x = np.empty(5)
x = np.empty([2, 3])
y = np.empty_like(np.array(range(4)))
# 全m数组，full(shape, fill_value)
m = 7
x = np.full((2,), m)
x = np.full(2, m)
x = np.full((2, 7), m)
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.full_like(x, m)
```
- 单位阵、对角矩阵
```python
x = np.eye(5)
x = np.eye(2, 3)
y = np.identity(4)
x = np.arange(9).reshape((3, 3))
y = np.diag(x, k=0)                                         # k 对角线的位置，大于零位于对角线上面，小于零则在下面。
```

## 1.4 利用数值范围来创建ndarray
```python
# np.arange()函数：返回给定间隔内的均匀间隔的值。
x = np.arange(3, 7, 2)                                      # [3 5]
# np.linspace()函数：返回指定间隔内的等间隔数字。
x = np.linspace(start=0, stop=2, num=9)                     # [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]
# np.logspace()函数：返回数以对数刻度均匀分布。
x = np.logspace(0, 1, 5)                                    # [ 1.    1.78  3.16  5.62 10.  ]
# np.random.rand() 返回一个由[0,1)内的随机数组成的数组。
x = np.random.random(5)
x = np.random.random([2, 3])
# np.around(a, decimals=0, out=None)，取整
x = np.around(x, decimals=2)
```

## 1.5 结构数组的创建
```python
# 利用字典来定义结构
personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
# 利用包含多个元组的列表来定义结构
personType = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
# 可以使用字段名作为下标获取对应的值
print(a['name'])
```

## 1.6 数组属性
```python
a = np.array([[1, 2, 3], [4, 5, 6.0]])
# numpy.ndarray.ndim用于返回数组的维数
a1 = a.ndim                                                 # 2
# numpy.ndarray.shape表示数组的shape
a2 = a.shape                                                # (2, 3)
# numpy.ndarray.size返回数组中所有元素的总量，相当于数组的shape中所有元素的乘积
a3 = a.size                                                 # 6
# numpy.ndarray.dtype 返回 ndarray 对象的元素类型
a4 = a.dtype                                                # float64
# numpy.ndarray.itemsize 以字节的形式返回数组中每一个元素的大小
a5 = a.itemsize                                             # 8
```
- 在`ndarray`中所有元素必须是同一类型，否则会自动向下转换，`int`->`float`->`str`
```python
b = np.array([1, 2, 3, 4, '5'])
print(b)
```
```markup
['1' '2' '3' '4' '5']
```