# $\xi6$ 逻辑斯谛回归与最大熵模型

[TOC]

## $\xi6.1$ 逻辑斯谛回归模型

### $\xi6.1.1$ 逻辑斯谛分布

+ 分布函数
  $$
  F(x)=P(X\leq x)=\frac{1}{1+\exp(-(x-\mu)/\gamma)}
  $$

+ 该曲线关于$(\mu,\frac{1}{2})$中心对称，曲线在中心附近增长速度较快，在两端增长速度较慢。形状参数$\gamma$的值越小，曲线在中心附近增长得越快。

+ 关于逻辑斯谛， 更常见的一种表达是Logistic function
  $$
  \sigma(z)=\frac{1}{1+\exp(-z)}
  $$

+ 这个函数把实数域映射到(0, 1)区间，这个范围正好是概率的范围， 而且可导，对于0输入， 得到的是0.5，可以用来表示等可能性。

### $\xi6.1.2$ 二项逻辑斯谛回归模型

+ 二项逻辑斯谛回归模型是如下的条件概率分布：
  $$
  \begin{aligned}
  P(Y=1|x)&=\frac{\exp(w\cdot x)}{1+\exp(w\cdot x)}  \\
  &=\frac{\exp(w\cdot x)/\exp(w\cdot x)}{(1+\exp(w\cdot x))/(\exp(w\cdot x))}  \\
  &=\frac{1}{e^{-(w\cdot x)}+1}  \\
  P(Y=0|x)&=\frac{1}{1+\exp(w\cdot x)}\\
  &=1-\frac{1}{1+e^{-(w\cdot x)}}  \\
  &=\frac{e^{-(w\cdot x)}}{1+e^{-(w\cdot x)}}
  \end{aligned}
  $$

+ **逻辑斯谛分布**对应了一种**概率**, **几率**为指数形式 $e^z$,  $z$ 为**对数几率**$logit$.

$$
logit(p)=\log(o)=\log\frac{p}{1-p}
$$

+ 上面是对数几率的定义， 这里对应了事件， 要么发生， 要么不发生。所以逻辑斯谛回归模型就表示成

$$
\log\frac{P(Y=1|x)}{1-P(Y=1|x)}=\color{red}\log\frac{P(Y=1|x)} {P(Y=0|x)}\color{black}=w\cdot x
$$

### $\xi6.1.3$ 模型参数估计

- 参数估计这里， 似然函数书中的表达
  $$
  \prod^N_{i=1}[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}
  $$

+ 使用对数似然会更简单， 会将上面表达式的连乘形式会转换成求和形式。对数函数为单调递增函数，最大化对数似然等价于最大化似然函数。
  $$
  \begin{aligned}
  \log \prod_{i=1}^N[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}&=\sum_{i=1}^N[y_i\log(\pi(x_i))+(1-y_i)\log(1-\pi(x_i))]  \\
  &=\sum_{i=1}^N[y_i\log(\frac{\pi(x_i)}{1-\pi(x_i)})+\log(1-\pi(x_i))]  \\
  &=\sum_{i=1}^N[y_i(w\cdot x_i)-\log(1+\exp(w\cdot x_i))]
  \end{aligned}
  $$

+ 这样，问题就变成了以对数似然函数为目标函数的最优化问题。逻辑斯蒂回归学习中通常采用的方法是梯度下降法及拟牛顿法。

### $\xi6.1.4$ 多项逻辑斯谛回归模型

+ 假设离散型随机变量$Y$的取值集合是${1,2,\dots,K}$, 多项逻辑斯谛回归模型是
  $$
  \begin{aligned}
  P(Y=k|x)&=\frac{\exp(w_k\cdot x)}{1+\sum_{k=1}^{K-1}\exp(w_k\cdot x)}, k=1,2,\dots,K-1\\
  P(Y=K|x)&=\frac{1}{1+\sum_{k=1}^{K-1}\exp(w_k\cdot x)}\\
  \end{aligned}
  $$

+ 参数估计，似然函数
  $$
  \prod\limits_{i=1}\limits^NP(y_i|x_i,W)=\prod\limits_{i=1}\limits^N\prod\limits_{l=1}^K \left(\frac{\exp(w_k\cdot x_i)}{\sum_{k=1}^K\exp(w_k\cdot x_i)}\right)^{I(y_i=l)}
  $$
  

## $\xi6.2$ 最大熵模型

### $\xi6.2.1$ 最大熵原理

+ 最大熵原理(Maxent principle)是**概率模型**学习的一个准则。
+ 最大熵原理通常用约束条件来确定概率模型的集合。

### $\xi6.2.2$ 最大熵模型的定义

- 最大熵原理也可以表述为在满足约束条件的模型集合中选取最大熵的模型。
- 几何解释：在单纯形中满足直线约束条件的模型中选取最优的模型，而最大熵原理则给出了选择最优模型的一个准则。

### $\xi6.2.2$ 最大熵模型的定义

+ 假设分类模型是一个条件概率分布，$P(Y|X)$, $X\in \mathcal {X} \sub \mathbf R^n$

  给定一个训练集 $T=\{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$

  $N$是训练样本容量，$x \in \mathbf R^n$ 

  联合分布$P(X,Y)$与边缘分布P(X)的经验分布分别为$\widetilde P(X, Y)和\widetilde P(X)$
  $$
  \begin{aligned}
  &\widetilde P (X=x, Y=y)=\frac{\nu(X=x, Y=y)}{N} \\
  &\widetilde P (X=x)=\frac {\nu (X=x)}{N}
  \end{aligned}
  $$
  上面两个就是不同的数据样本，在训练数据集中的比例。

+ 特征函数$f(x,y)$关于经验分布$\widetilde P (X, Y)$的期望值用$E_{\widetilde P}(f)$表示：
  $$
  E_{\widetilde P}(f)=\sum\limits_{x,y}\widetilde P(x,y)f(x,y)
  $$
  特征函数$f(x,y)$关于模型$P(Y|X)$与经验分布$\widetilde P (X)$的期望值, 用$E_{P}(f)$表示
  $$
  E_{P}(f)=\sum\limits_{x,y}{\widetilde P(x)P(y|x)f(x,y)}
  $$
  如果模型能够获取训练数据中的信息，那么就有
  $$
  \widetilde{P}(x,y)=P(y|x)\widetilde{P}(x)
  $$
  就可以假设这两个期望值相等，即$$E_P(f)=E_{\widetilde P}(f)$$

+ 如果增加$n$个**特征函数**, 就可以增加$n$个**约束条件**，特征也对应增加了一列。

  + 假设满足所有约束条件的模型集合为

  $\mathcal {C} \equiv \ \{P \in \mathcal {P}|E_P(f_i)=E_{\widetilde {P}}(f_i) {, i=1,2,\dots,n}\} $

  + 定义在条件概率分布$P(Y|X)$上的条件熵为

  $H(P)=-\sum \limits _{x, y} \widetilde {P}(x)P(y|x)\log {P(y|x)}$

  则模型集合$\cal {C}$中条件熵$H(P)$最大的模型称为最大熵模型，上式中对数为自然对数。

### $\xi6.2.3$ 最大熵模型的学习

- 最大熵模型的学习可以形式化为约束最优化问题。
  $$
  \begin{eqnarray*}
  \min \limits_{P\in \mathcal {C}}-H(P)=\sum\limits_{x,y}\widetilde P(x)P(y|x)\log P(y|x) \tag{6.14}  \\
  s.t. E_P(f_i)-E_{\widetilde P}(f_i)=0, i =1,2,\dots,n \tag{6.15}  \\
  \sum \limits_y P(y|x)=1 \tag{6.16}
  \end{eqnarray*}
  $$

- 最优化原始问题可以等价位对偶问题。而最大熵模型的学习归结为对偶函数的极大化。

- 步骤

  1. 根据约束条件，引进拉格朗日乘子，定义拉格朗日函数。
  2. 根据拉格朗日对偶性，通过对偶最优化问题得到原始最优化问题的解。

  $$
  \max_w\min_P L(P,w)
  $$

  3. 先求解$L(P,w)$关于P的极小化问题。分别对$P_i$求偏导。
  4. 再求解$$\min_P L(P,w)$$的极大化问题。分别对$w_i$求偏导。
  5. 最后得到所要的概率分布。

### $\xi6.2.4$ 极大似然估计

- 对偶函数的极大化等价于最大熵模型的极大似然估计。证明见书P102。

## $\xi6.3$ 模型学习的最优化算法

### $\xi6.3.1$ 改进的迭代尺度法

- 改进的迭代尺度法是一种最大熵模型学习的最优化算法。

### $\xi6.3.2$ 拟牛顿法

- 拟牛顿法也是一种最大熵模型学习的最优化算法。见附录B

### $\xi6.3.3$ 目标函数

#### 逻辑斯谛回归模型

$$
\begin{aligned}
L(w)&=\sum\limits^{N}_{i=1}[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))]\\
&=\sum\limits^{N}_{i=1}[y_i\log{\frac{\pi(x_i)}{1-\pi(x_i)}}+\log(1-\pi(x_i))]\\
&=\sum\limits^{N}_{i=1}[y_i(w\cdot x_i)-\log(1+\exp(w\cdot{x_i})]
\end{aligned}
$$

#### 最大熵模型

$$
\begin{align}
L_{\widetilde {P}}(P_w)&=\sum \limits_{x,y}\widetilde {P}(x,y)\log{P}(y|x)  \\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x,y)\log{(Z_w(x))}  \\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x)P(y|x)\log{(Z_w(x))}  \\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}\sum_{y}P(y|x)  \\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}
\end{align}
$$

以上推导用到了$\sum\limits_yP(y|x)=1$

## $\xi6.4$ 其他

+ 最大熵模型与逻辑斯谛回归模型有类似的形式，它们又称为对数线性模型。模型学习就是再给定的训练数据条件下对模型进行极大似然估计或正则化的极大似然估计。