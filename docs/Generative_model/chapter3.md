# DDPM论文精读
- 扩散概率模型通过变分推断进行参数化马尔科夫链训练，在有限步后生成与数据匹配的样本。

## 3.1 模型概览
- 扩散模型是隐变量模型，概率密度函数定义为：
$$p_{\theta}(x_0):={\int}p_{\theta}(x_{0:T})dx_{1:T}$$
其中 $x_1,\cdots,x_T$ 是和数据 $x_0{\sim}q(x_0)$同维度的隐变量。

### 3.1.1 模型的逆向过程和前向过程
- 如下图所示，我们构造两个相反的过程，通过向图片中添加噪声，再反向去噪，来达到生成图像的任务。

![](./img/3.1DDPM.png ':size=100%')

#### 逆向过程(reverse process)
- 联合分布 $p_{\theta}(x_{0:T})$ 定义为一个马尔科夫链：
$$p_{\theta}(x_{0:T}):=p(x_T)\prod\limits_{t=1}^Tp_{\theta}(x_{t-1}|x_t)$$
$$p_{\theta}(x_{t-1}|x_t):=\mathcal{N}\left(x_{t-1};\mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t,t)\right)$$
其中 $T$ 步的分布是一个高斯分布 $p(x_T)=\mathcal{N}(x_T;\mathbf{0},\mathbf{I})$，我们的目的是从这个高斯分布通过逐步去掉高斯噪声来还原图像。

#### 前向(扩散)过程(forward process/diffusion process)
- 根据变量 $\beta_1,\cdots,\beta_T$ 向数据中逐步添加高斯噪声来逼近 $q(x_{1:T}|x_0)$：
$$q(x_{1:T}|x_0):=\prod_{t=1}^Tq(x_t|x_{t-1})$$
$$q(x_t|x_{t-1}):=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t{\mathbf{I}})$$

### 3.1.2 模型的优化训练
- 通过优化变分下界的似然函数来训练：
$$\mathbb{E}[-{\rm{log}}p_{\theta}(x_0)]\leq\mathbb{E}_q\left[-{\rm{log}}\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}\right]=\mathbb{E}_q\left[-{\rm{log}}p(x_T)-\sum\limits_{t{\geq}1}{\rm{log}}\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]=:\mathcal{L}$$
- 前向过程中，我们向数据中逐步添加高斯噪声，多步添加可以合并，我们可以得到一个隐式解，使用符号替换 $\alpha_t:=1-\beta_t$ 和 $\bar{\alpha}_t:=\prod\limits_{s=1}^t\alpha_s$，我们有：
$$q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t){\mathbf{I}})$$
然后我们就可以通过随机梯度下降来优化 $\mathcal{L}$。

#### 变分下界似然函数
- 我们结合上述式子，通过变量替换重写 $\mathcal{L}$：
$$\mathcal{L}=\mathbb{E}_q\left[D_{\rm{KL}}(q(x_T|x_0)\|p(x_T))+\sum\limits_{t>1}D_{\rm{KL}}(q(x_{t-1}|x_t,x_0)\|p_{\theta}(x_{t-1}|x_t))-{\rm{log}}p_{\theta}(x_0|x_1)\right]$$
- 之所以将 $\mathcal{L}$ 写成KL散度的形式，是为了将 $\mathcal{L}$ 分为以下三个部分来学习：
    - 我们记 $\mathcal{L}$ 期望中的第一部分为 $\mathcal{L}_T$：
    $$\mathcal{L}_T:=D_{\rm{KL}}(q(x_T|x_0)\|p(x_T))$$
    - 我们记 $\mathcal{L}$ 期望中的第二部分为 $\mathcal{L}_{t-1}$：
    $$\mathcal{L}_{t-1}:=D_{\rm{KL}}(q(x_{t-1}|x_t,x_0)\|p_{\theta}(x_{t-1}|x_t))$$
    - 我们记 $\mathcal{L}$ 期望中的第三部分为 $\mathcal{L}_{0}$：
    $$\mathcal{L}_0=-{\rm{log}}p_\theta(x_0|x_1)$$
- 上述式子通过KL散度直接比较 $p_\theta(x_{t-1}|x_t)$ 和 $q(x_{t-1}|x_t,x_0)$。我们可以证明在 $x_0$ 条件下它是一个高斯分布：
$$q(x_{t-1}|x_t,x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t\mathbf{I})$$
其中 $\tilde{\mu}_t(x_t,x_0):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$，$\tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$。

## 3.2 扩散模型和去噪自编码器
Diffusion models and denoising autoencoders
### 3.2.1 前向过程和 $\mathcal{L}_T$
- 在训练中，我们不用重参数化技巧去替换 $\beta_t$，而是将它固定为常数。这样，在我们的假设中，近似后验 $q$ 就没有可学习的参数，前向过程完全是已知的，所以 $\mathcal{L}_T$ 在训练中就是一个常量，不参与学习过程。

### 3.2.2 逆向过程和 $\mathcal{L}_{1:T-1}$
- 我们讨论逆向过程中 $p_{\theta}(x_{t-1}|x_t)=\mathcal{N}\left(x_{t-1};\mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t,t)\right)$ 的选择。
    - 首先，我们令 $\Sigma_{\theta}(x_t,t)=\sigma_t^2\mathbf{I}$。其中 $\sigma_t^2=\beta_t$ 和 $\sigma_t^2=\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ 两种极端情况分别对应了逆向过程中熵的上下界。
    - 由于我们对 $p_{\theta}(x_{t-1}|x_t)$ 的重写，我们可以对 $\mathcal{L}_{t-1}$ 进行下述变换：
    $$\mathcal{L}_{t-1}=\mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\mu}_t(x_t,x_0)-\mu_{\theta}(x_t,t)\|^2\right]+C$$
    其中 $C$ 是一个与 $\theta$ 无关的常数。

- 对 $q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t){\mathbf{I}})$ 进行重参数化，用 $x_0$ 和 $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 来描述 $x_t$：
$$x_t(x_0,\epsilon)=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$
- 再将 $\tilde{\mu}_t(x_t,x_0)$ 展开，可以得到：
$$\begin{align*}
\mathcal{L}_{t-1}-C&=\mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma_t^2}\bigg|\bigg|\tilde{\mu}_t\left(x_t(x_0,\epsilon),\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t(x_0,\epsilon)-\sqrt{1-\bar{\alpha}_t}\epsilon)\right)-\mu_\theta(x_t(x_0,\epsilon),t)\bigg|\bigg|^2\right]\\
&=\mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma_t^2}\bigg|\bigg|\frac{1}{\sqrt{\alpha_t}}\left(x_t(x_0,\epsilon)-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)-\mu_\theta(x_t(x_0,\epsilon),t)\bigg|\bigg|^2\right]
\end{align*}$$
- 因为 $x_t$ 可以作为模型输入，所以我们保留 $x_t$ 作为输入量。然后根据 $\tilde{\mu}_t$ 的展开式，可以得到如下参数变换：
$$\mu_\theta(x_t,t)=\tilde{\mu}_t\left(x_t,\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t))\right)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)$$
其中 $\epsilon_\theta$ 是从 $x_t$ 预测 $\epsilon$ 的函数近似。
- 联立上述两式得到：
$$\mathcal{L}_{t-1}-C=\mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)\|^2\right]$$

### 3.2.3 训练和采样算法
- 训练算法
> **repeat**  
&emsp;&emsp;$x_0{\sim}q(x_0)$  
&emsp;&emsp;$t{\sim}{\rm{Uniform}}(\{1,\cdots,T\})$  
&emsp;&emsp;$\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$  
&emsp;&emsp;梯度下降步骤  
&emsp;&emsp;&emsp;&emsp;$\nabla_\theta\|\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)\|^2$  
**until** 收敛

- 采样算法
> $x_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$  
**for** $t=T,\cdots,1$ **do**  
&emsp;&emsp;$z\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ if $t>1$, else $z=0$  
&emsp;&emsp;$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_tz$  
**end for**  
**return** $x_0$

### 3.2.4 数据缩放、逆向过程解码器、$\mathcal{L}_0$
- 我们将逆向过程的最后一项设置为从高斯分布 $\mathcal{N}(x_0;\mu_\theta(x_1,1),\sigma_1^2\mathbf{I})$ 生成的独立离散解码器：
$$p_\theta(x_0|x_1)=\prod\limits_{i=1}^D\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)}\mathcal{N}(x;\mu_\theta^i(x_1,1),\sigma_1^2)dx$$
$$\delta_+(x)=\left\{
    \begin{array}{ll}
        \infty & {\rm{if}}\ x=1\\
        x+\frac{1}{255} & {\rm{if}}\ x<1
    \end{array}
    \right.\quad\delta_-(x)=\left\{
    \begin{array}{ll}
        -\infty & {\rm{if}}\ x=-1\\
        x-\frac{1}{255} & {\rm{if}}\ x>-1
    \end{array}
    \right.$$
其中 $D$ 是数据的维度。
- 这样可以保证数据的无损，同时也不再需要向数据中添加噪声或者将缩放操作的雅克比矩阵合并到对数似然中。

### 3.2.5 简化训练对象
- 在实验中，我们发现下式对变分下界的简化能增加样本的质量且更容易实施(其中 $t$ 在1到 $T$ 之间)：
$$\mathcal{L}_{\rm{simple}}(\theta):=\mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)\|^2\right]$$
上式只是去掉了一些不影响 $L_2$ 损失的常量参数和变量系数，使得 $\mathcal{L}$ 更简单，训练起来就会更高效。