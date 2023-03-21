# VAE论文精读
## 2.1 问题描述
- 由 $N$ 个连续或离散变量 $x$ 组成的数据集 $X=\{x^{(i)}\}^N_{i=1}$，我们假定 $X$ 由包含隐变量 $z$ 的随机过程生成，包括两步：
    - 由一个先验分布 $p_{\theta^*}(z)$ 生成的 $z^{(i)}$
    - 由条件分布 $p_{\theta^*}(x|z)$ 生成的 $x^{(i)}$
- 我们假定 $p_{\theta^*}(z)$ 和 $p_{\theta^*}(x|z)$ 是来自 $p_{\theta}(z)$ 和 $p_{\theta}(x|z)$，并且概率密度函数对于 $\theta$ 和 $z$ 是处处可微的。
- 另一方面，$\theta^*$ 和 $z^{(i)}$ 是未知的参数和变量。
- 我们考虑一种通用算法，该算法能够处理以下两个情况：
    - Intractability：边际似然函数的积分 $p_\theta(x)={\int}p_\theta(z)p_\theta(x|z)dz$ 难以计算，所以我们不能估计似然函数；后验分布 $p_\theta(z|x)=p_\theta(x|z)p_\theta(z)/p_\theta(x)$ 难以计算，所以不能使用EM算法；均值场假设中的积分也同样难以计算。这种难以计算的情况在一些复杂似然函数时非常常见，例如一个带有非线性隐藏层的神经网络。
    - 超大数据集：我们需要处理海量的数据集，批优化算法也同样昂贵；我们希望使用很小的小批量甚至单个数据去更新参数。这种情况下蒙特卡洛EM算法由于需要循环采样而低效。
- 我们关心并要解决的三个相关问题如下：
    - 参数 $\theta$ 的有效逼近ML或者MAP：我们可能就是求这些参数本身，或者模仿隐藏的随机过程并生成类似于真实数据的人工数据。
    - 对给定参数 $\theta$ 和观测值 $x$ 的隐变量 $z$ 进行有效的近似后验推理。这在编码任务和数据表示任务中很有用。
    - 对变量 $x$ 进行有效的边际推理逼近。这使我们能够进行所有需要先验 $x$ 的推理任务。通常来说，图像去噪、图像修复和超分辨率任务有这部分的应用。
- 为了解决上述问题，论文介绍了一个识别模型 $q_\phi(z|x)$：对难以计算的真正后验分布 $p_\theta(z|x)$ 的逼近。与均值场方法不同的是，模型不需要因子分解，且参数 $\theta$ 不是从某些封闭形式的期望中计算出来的。

## 2.2 变分下界
- 边际似然函数是由各个独立的数据的边际似然求和得到：
$${\rm{log}}p_\theta(x^{(1)},\cdots,x^{(N)})=\sum\limits_{i=1}^N{\rm{log}}p_\theta(x^{(i)})$$
可以重写上式为：
$${\rm{log}}p_\theta(x^{(i)})=D_{KL}\left(q_\phi(z|x^{(i)})\|p_\theta(z|x^{(i)})\right)+\mathcal{L}(\theta,\phi;x^{(i)})$$
- 由于KL散度是非负的，所以 $\mathcal{L}(\theta,\phi;x^{(i)})$ 这一项被叫做数据点 $i$ 的边际似然的变分下界：
$${\rm{log}}p_\theta(x^{(i)})\geq\mathcal{L}(\theta,\phi;x^{(i)})=\mathbb{E}_{q_\phi(z|x)}[-{\rm{log}}q_\phi(z|x)+{\rm{log}}p_\theta(x,z)]$$
可以重写上式为(推导略)：
$$\mathcal{L}(\theta,\phi;x^{(i)})=-D_{KL}\left(q_\phi(z|x^{(i)})\|p_\theta(z)\right)+\mathbb{E}_{q_\phi(z|x^{(i)})}\left[{\rm{log}}p_\theta(x^{(i)}|z)\right]$$
- 我们希望根据变分参数 $\phi$ 和生成参数 $\theta$ 对下界进行优化，但变分下界的梯度估计有很高的方差。

## 2.3 SGVB估计和AEVB算法
- 我们可以引入噪声变量 $\epsilon$ 和分布转换函数 $g_\phi(\epsilon,x)$ 来重写随机变量 $\tilde{z}{\sim}q_\phi(z|x)$：
$$\tilde{z}=g_\phi(\epsilon,x),\quad\epsilon{\sim}p(\epsilon)$$
- 对函数 $f(z)$ 的期望进行蒙特卡洛估计：
$$\mathbb{E}_{q_\phi(z|x^{(i)})}[f(z)]=\mathbb{E}_{p(\epsilon)}\left[f(g_\phi(\epsilon,x^{(i)}))\right]\simeq\frac{1}{L}\sum\limits_{l=1}^Lf(g_\phi(\epsilon^{(l)},x^{(i)}))$$
其中 $\epsilon^{(l)}{\sim}p(\epsilon)$。
#### SGVB估计
- 随机梯度变分贝叶斯(SGVB)估计 $\tilde{\mathcal{L}}^A(\theta,\phi;x^{(i)})\simeq\mathcal{L}(\theta,\phi;x^{(i)})$：
$$\tilde{\mathcal{L}}^A(\theta,\phi;x^{(i)})=\frac{1}{L}\sum\limits_{l=1}^L{\rm{log}}p_\theta(x^{(i)},z^{(i,l)})-{\rm{log}}q_\phi(z^{(i,l)}|x^{(i)})$$
其中 $z^{(i,l)}=g_\phi(\epsilon^{(i,l)},x^{(i)})$，$\epsilon^{(l)}{\sim}p(\epsilon)$。
- 由于KL散度总是能够根据其中先验和后验的分布来综合得到(见原始论文附录B)，因此只需对期望项进行采样估计，这给出了第二种SGVB估计形式：
$$\tilde{\mathcal{L}}^B(\theta,\phi;x^{(i)})=-D_{KL}(q_\phi(z|x^{(i)})\|p_\theta(z))+\frac{1}{L}\sum\limits_{l=1}^L{\rm{log}}p_\theta(x^{(i)}|z^{(i,l)})$$
其中 $z^{(i,l)}=g_\phi(\epsilon^{(i,l)},x^{(i)})$，$\epsilon^{(l)}{\sim}p(\epsilon)$。
#### 小批量估计
- 给定 $N$ 个数据的数据集 ${\rm{X}}$，通过小批量(每个batch $M$ 个数据)进行训练，我们可以构建整个数据集的边际似然下界的估计：
$$\mathcal{L}(\theta,\phi;{\rm{X}})\simeq\tilde{\mathcal{L}}(\theta,\phi;{\rm{X}}^M)=\frac{N}{M}\sum\limits_{i=1}^M\tilde{\mathcal{L}}(\theta,\phi;x^{(i)})$$
其中minibatch ${\rm{X}}^M=\{x^{(i)}\}^M_{i=1}$ 是整个数据集中的随机抽样。
#### AEVB算法
- 算法1：小批量版本的Auto-Encoding VB(AEVB)算法，两种SGVB都可以使用，在实验中我们取 $M=100$，$L=1$。
> $\theta,\phi\leftarrow$ 参数初始化  
**repeat**  
&emsp;&emsp;${\rm{X}}^M\leftarrow$ 从数据集中随机采样 $M$ 个数据  
&emsp;&emsp;$\epsilon\leftarrow$ 从噪声分布 $p(\epsilon)$ 随机采样  
&emsp;&emsp;$g\leftarrow\nabla_{\theta,\phi}\tilde{\mathcal{L}}^M(\theta,\phi;{\rm{X}}^M,\epsilon)$ 小批量估计  
&emsp;&emsp;$\theta,\phi\leftarrow$ 用梯度 $g$ 更新参数(SGD或Adagrad等优化算法)  
**until** 参数 $(\theta,\phi)$ 收敛  
**return** $\theta,\phi$

## 2.4 重参数化技巧
- 重参数化技巧被频繁用在SGVB估计的推导中。在训练过程中，我们需要对连续随机变量 $z$ 进行采样才能通过逼近来近似求得期望，但采样的步骤使得梯度不能回传，无法进行优化算法。因此我们引入重参数化技巧，将 $z$ 变为确定性变量，取而代之的是采样一个 $\epsilon$，这样就解决了梯度不能回传的问题，这也是这个技巧的核心。
- 具体来说，假定 $z$ 是一个连续随机变量，服从条件概率分布 $z{\sim}q_\phi(z|x)$，通常可以将随机变量 $z$ 表示为一个确定性变量 $z=g_\phi(\epsilon,x)$，其中 $\epsilon{\sim}p(\epsilon)$，$g_\phi(.)$ 是由 $\phi$ 参数化的向量值函数。
- 我们有：
$$q_\phi(z|x)\prod_idz_i=p(\epsilon)\prod_id\epsilon_i$$
因此有：
$${\int}q_\phi(z|x)f(z)dz={\int}p(\epsilon)f(z)d\epsilon={\int}p(\epsilon)f(g_\phi(\epsilon,x))d\epsilon$$
- 根据上式构建一个微分估计：
$${\int}q_\phi(z|x)f(z)dz\simeq\frac{1}{L}\sum\limits_{l=1}^Lf(g_\phi(x,\epsilon^{(l)}))$$
其中 $\epsilon^{(l)}{\sim}p(\epsilon)$。