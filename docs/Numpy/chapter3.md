# Chapter03 Numpy在模型训练中的操作
## 3.1 输入和输出
### 3.1.1 保存和读取
- `numpy.save(file, arr, allow_pickle=True, fix_imports=True)` 以`.npy`格式将数组保存到二进制文件中
    - `file`：文件名
    - `arr`：待存储数据
    - `allow_pickle`：是否允许使用pickle存储
    - `fix_imports=True`：若为True，pickle将尝试将旧的python2名称映射到python3中使用的新名称
- `numpy.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000)` 从`.npy`、`.npz`或 pickled文件加载数组或pickled对象
    - `mmap_mode: {None, ‘r+’, ‘r’, ‘w+’, ‘c’};`：读取文件的方式
    - `allow_pickle=False`：允许加载存储在`.npy`文件中的pickled对象数组
    - `fix_imports=True`：若为True，pickle将尝试将旧的python2名称映射到python3中使用的新名称
    - `encoding='ASCII'`：制定编码格式，默认为“ASCII”
```python
import numpy as np
outfile = r'.\test.npy'
x = np.random.uniform(low=0, high=1,size = [3, 5])
np.save(outfile, x)
y = np.load(outfile)
```
- `numpy.savez(file, *args, **kwds)` 以未压缩的`.npz`格式将多个数组保存到单个文件中
    - `.npz`格式：以压缩打包的方式存储文件，可以用压缩软件解压。
    - `savez()`函数：第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为`arr_0, arr_1, …`。
    - `savez()`函数：输出的是一个压缩文件(扩展名为`.npz`)，其中每个文件都是一个`save()`保存的`.npy`文件，文件名对应于数组名。`load()`自动识别`.npz`文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容
```python
import numpy as np
outfile = r'.\test.npz'
x = np.linspace(0, np.pi, 5)
y = np.sin(x)
z = np.cos(x)
np.savez(outfile, x, y, z_d=z)
data = np.load(outfile)
np.set_printoptions(suppress=True)                          # 用于控制Python中小数的显示精度，suppress=True表示不用科学计数法
print(data.files)                                           # ['z_d', 'arr_0', 'arr_1']
print(data['arr_0'])                                        # [0.         0.78539816 1.57079633 2.35619449 3.14159265]
print(data['arr_1'])                                        # [0.         0.70710678 1.         0.70710678 0.        ]
print(data['z_d'])                                          # [ 1.          0.70710678  0.         -0.70710678 -1.        ]
```
### 3.1.2 文本文件的保存和读取
- `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)` 用来存储文本文件(如`.TXT`，`.CSV`等)
    - `fname`：文件路径
    - `X`：存入文件的数组
    - `fmt='%.18e'`：写入文件中每个元素的字符串格式，默认'%.18e'(保留18位小数的浮点数形式)
    - `delimiter=' '`：分割字符串，默认以空格分隔
- `numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None, like=None)` 用来读取文本文件(如`.TXT`，`.CSV`等)
    - `fname`：文件路径
    - `dtype=float`：数据类型，默认为float
    - `comments='#'`: 字符串或字符串组成的列表，默认为'#'，表示注释字符集开始的标志
    - `skiprows=0`：跳过多少行，一般跳过第一行表头
    - `usecols=None`：元组（元组内数据为列的数值索引）， 用来指定要读取数据的列(第一列为0)
    - `unpack=False`：当加载多列数据时是否需要将数据列进行解耦赋值给不同的变量
```python
import numpy as np
outfile1 = r'.\test.txt'
x1 = np.arange(0, 10).reshape(2, -1)
np.savetxt(outfile1, x1)
y1 = np.loadtxt(outfile1)
outfile2 = r'.\test.csv'
x2 = np.arange(0, 10, 0.5).reshape(4, -1)
np.savetxt(outfile2, x2, fmt='%.3f', delimiter=',')
y2 = np.loadtxt(outfile2, delimiter=',')
```
- `numpy.genfromtxt()` 从文本文件加载数据，并按指定方式处理缺失值
### 3.1.3 文本格式选项
- `numpy.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None, *, legacy=None)` 设置打印选项，这些选项决定浮点数、数组和其它Numpy对象的显示方式
    - `precision=8`：设置浮点精度，控制输出的小数点个数，默认是8
    - `threshold=1000`：概略显示，超过该值则以“…”的形式来表示，默认是1000
    - `linewidth=75`：用于确定每行多少字符数后插入换行符，默认为75
    - `suppress=False`：当`suppress=True`，表示小数不需要以科学计数法的形式输出，默认是False
    - `nanstr=nan`：浮点非数字的字符串表示形式，默认`nan`
    - `infstr=inf`：浮点无穷大的字符串表示形式，默认`inf`
    - `formatter`：一个字典，自定义格式化用于显示的数组元素。键为需要格式化的类型，值为格式化的字符串
- `numpy.get_printoptions()` 获取当前打印选项

## 3.2 随机抽样
- `numpy.random` 模块对 Python 内置的 `random` 进行了补充，增加了一些用于高效生成多种概率分布的样本值的函数，如正态分布、泊松分布等。
- `numpy.random.seed(seed=None)` 设置随机种子
### 3.2.1 离散型随机变量
- 二项分布，概率函数：
$$P\{X=k\}={n \choose k}p^k(1-p)^{n-k}$$
    - `numpy.random.binomial(n, p, size=None)` 对二项分布进行采样，size表示采样的次数，n表示做了n重伯努利试验，p表示成功的概率，函数的返回值表示n中成功的次数。
- 泊松分布，概率函数：
$$P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda},k=0,1,\cdots$$
    - `numpy.random.poisson(lam=1.0, size=None)` 对一个泊松分布进行采样，size表示采样的次数，lam表示一个单位内发生事件的平均值，函数的返回值表示一个单位内事件发生的次数。
- 超几何分布，概率函数：
$$p(k,M,n,N)=\frac{{n \choose k}{M-n \choose N-k}}{{M \choose N}}$$
    - `numpy.random.hypergeometric(ngood, nbad, nsample, size=None)` 对一个超几何分布进行采样，size表示采样的次数，ngood表示总体中具有成功标志的元素个数，nbad表示总体中不具有成功标志的元素个数，ngood+nbad表示总体样本容量，nsample表示抽取元素的次数（小于或等于总体样本容量），函数的返回值表示抽取nsample个元素中具有成功标识的元素个数。
### 3.2.2 连续型随机变量
- 均匀分布
    - `numpy.random.uniform(low=0.0, high=1.0, size=None)` 从区间[low, high)的均匀分布采样，size表示采样的次数。
    - `numpy.random.rand(d0, d1, ..., dn)` 可以得到[0,1)之间的均匀分布的随机数。
    - `numpy.random.randint(low, high=None, size=None, dtype='l')` 可以得到[low,high)之间均匀分布的随机整数
- 标准正态分布，概率函数：
$$f(x)=\frac{{\rm{exp}}(-x^2/2)}{\sqrt{2\pi}}$$
    - `numpy.random.randn(d0, d1, ..., dn)` 返回标准正态分布(均值为0，标准差为1)的一个采样。
- 正态(高斯)分布，$X{\sim}N(\mu,\sigma^2)$，概率函数：
$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}{\rm{exp}}\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
    - `numpy.random.normal(loc=0.0, scale=1.0, size=None)` 从正态(高斯)分布中抽取随机样本，loc表示均值，scale表示标准差，size表示采样的次数。
- 指数分布，$X{\sim}{\rm{EXP}}(\frac{1}{\lambda})$，概率函数：
$$\begin{equation*}
f(x)=\begin{cases}
{\lambda}e^{-{\lambda}x}&x>0\\
0&x\leq0
\end{cases}
\end{equation*}$$
    - `numpy.random.exponential(scale=1.0, size=None)` 对指数分布进行采样，$scale=\frac{1}{\lambda}$，size表示采样的次数。
### 3.2.3 其他随机函数
- `numpy.random.choice(a, size=None, replace=True, p=None)` 从序列中获取元素，若a为整数，元素取值从`np.range(a)`中随机获取；若a为数组，取值从a数组元素中随机获取，replace控制生成数组中的元素是否重复，p为选取元素概率，size表示采样的次数。
```python
np.random.seed(20200614)
x = np.random.choice(10, 3, replace=False, p=[0.05, 0, 0.05, 0.9, 0, 0, 0, 0, 0, 0])
print(x)                                                    # [3 0 2]
```
- `numpy.random.shuffle(x)` 对x进行重排序，如果x为多维数组，只沿第 0 轴洗牌，改变原来的数组，输出为None。
- `numpy.random.permutation(x)` 与shuffle相同，打乱第0轴的数据，但是它不改变原来的数组，返回一个新数组。
## 3.3 统计相关
### 3.3.1 次序统计
- `numpy.amin(a[, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue])` 返回最小值，或返回沿轴axis的最小值数组。
- `numpy.amax(a[, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue])` 返回最大值，或返回沿轴axis的最大值数组。
- `numpy.ptp(a, axis=None, out=None, keepdims=np._NoValue)` 返回极差，或返回沿轴axis的极差数组。
- `numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)` 返回a数组排序后数组中q分位的元素，axis设置则沿轴进行。
```python
np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
np.percentile(x, [25, 50])                                  # [ 2. 10.]
np.percentile(x, [25, 50], axis=1)                          # [[ 1. 10.  8.  2.] [ 2. 11.  9. 15.]]
```
### 3.3.2 均值与方差
- `numpy.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)` 返回中位数，axis设置则沿轴返回中位数数组。
- `numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)` 返回平均值，axis设置则沿轴返回平均值数组。
- `numpy.average(a, axis=None, weights=None, returned=False, *, keepdims=<no value>)` 计算加权平均值，
```python
x = np.array(range(11, 36)).reshape(5,5)
y1 = np.average(x)                                          # 23.0
y2 = np.average(x, axis=0)                                  # [21. 22. 23. 24. 25.]
x2 = np.arange(1, 26).reshape([5, 5])
z1 = np.average(x, weights=x2)                              # 27.0
z2 = np.average(x, axis=0, weights=x2)                      # [25.54545455, 26.16666667, 26.84615385, 27.57142857, 28.33333333]
```
- `numpy.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)[source]` 返回数组方差，axis设置则沿轴返回方差数组。ddof=0时为方差(分母为 $n$)，ddof=1时为样本方差(分母为 $n-1$)
- `numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)` 返回数组标准差，axis设置则沿轴返回标准差数组。
### 3.3.3 其他统计量
- 计算协方差矩阵，期望为 $E[X]$ 和 $E[Y]$ 的两个实随机变量 $X$ 与 $Y$ 之间的协方差 $\rm{Cov}(X,Y)$ 定义为：
$${\rm{Cov}}(X,Y)=E[(X-E[X])(Y-E[Y])]=E[XY]-E[X]E[Y]$$
    - `numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)` 返回数组协方差，可带权重。
- 计算相关系数，正则化的协方差：
$$r(X,Y)=\frac{{\rm{Cov}}(X,Y)}{\sqrt{{\rm{Var}}[X]{\rm{Var}}[Y]}}$$
    - `numpy.corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue)` 返回数组相关系数。
- 直方图
    - `numpy.digitize(x, bins, right=False)` 返回x中每个值在升序数组bins中的位置。
```python
x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)                                 # [1 4 3 2]
```
## 3.4 线性代数
- Numpy 定义了 `matrix` 类型，使用该 `matrix` 类型创建的是矩阵对象，它们的加减乘除运算采用矩阵方式计算，因此用法和Matlab十分类似。但是由于 NumPy 中同时存在 `ndarray` 和 `matrix` 对象，因此用户很容易将两者弄混。这有违 Python 的"显式优于隐式"的原则，因此官方并不推荐在程序中使用 matrix。在这里，我们仍然用 ndarray 来介绍。
### 3.4.1 矩阵和向量积
- `numpy.dot(a, b[, out])` 计算两个矩阵的乘积，如果是一维数组则是它们的内积。
- `numpy.linalg.eig(a)` 计算方阵的特征值和特征向量。
- `numpy.linalg.eigvals(a)` 计算方阵的特征值。
```python
x = np.diag((1, 2, 3))
y = np.linalg.eigvals(x)                                    # [1. 2. 3.]
a, b = np.linalg.eig(x)                                     # a = [1. 2. 3.], b = [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
```
### 3.4.2 矩阵分解
- 奇异值分解 `u, s, v = numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)`
    - a 是一个形如(M,N)矩阵
    - `full_matrices`的取值是为`False`或者`True`，为`True`时u的大小为(M,M)，v的大小为(N,N)。否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)。
    - `compute_uv`的取值是为`False`或者`True`，为`True`时表示计算u,s,v。为False的时候只计算s。
    - 总共有三个返回值u,s,v，u大小为(M,M)，s大小为(M,N)，v大小为(N,N)，a = u*s*v。
    其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值。
- QR分解 `q,r = numpy.linalg.qr(a, mode='reduced')` 计算矩阵a的QR分解。
    - a是一个(M, N)的待分解矩阵。
    - mode = reduced：返回(M, N)的列向量两两正交的矩阵q，和(N, N)的三角阵r(Reduced QR分解)
    - mode = complete：返回(M, M)的正交矩阵q，和(M, N)的三角阵r(Full QR分解)
- Cholesky分解
    - `L = numpy.linalg.cholesky(a)` 返回正定矩阵a的 Cholesky 分解 $a=L{\cdot}L^\top$，其中$L$是下三角矩阵。
### 3.4.3 范数等操作
- `numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)` 计算向量或者矩阵的范数。根据`ord`参数的不同，计算不同的范数：
![](./img/linalg_norm.png ':size=80%')
- `numpy.linalg.det(a)` 返回矩阵行列式
- `numpy.linalg.matrix_rank(M, tol=None, hermitian=False)` 返回矩阵的秩
- `numpy.trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)` 返回矩阵的迹，方阵的迹就是主对角元素之和
### 3.4.4 逆矩阵
- `numpy.linalg.inv(a)` 计算矩阵a的逆矩阵
- `numpy.linalg.solve(a, b)` 求解线性方程组或矩阵方程
```python
# 求解线性方程组
#  x + 2y +  z = 7
# 2x -  y + 3z = 7
# 3x +  y + 2z =18
A = np.array([[1, 2, 1], [2, -1, 3], [3, 1, 2]])
b = np.array([7, 7, 18])
x = np.linalg.solve(A, b)                                   # [ 7.  1. -2.]
x = np.linalg.inv(A).dot(b)                                 # [ 7.  1. -2.]
```