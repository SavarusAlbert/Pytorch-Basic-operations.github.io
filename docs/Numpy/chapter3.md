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
### 3.2.1 