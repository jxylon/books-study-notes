# $\xi2$ 感知机

[TOC]

## $\xi2.1$ 感知机模型

+ 定义：

  + 输入空间：$\mathcal X \sube \bf R^n$

    输出空间：$\mathcal Y=\{{+1,-1\}}$

    > 从输入空间到输出空间的决策函数：$f(x)=sign (w\cdot x+b)$
    >
    > 其中$w \in R^n$叫作权值，$w \in R$叫作偏置

  称为感知机。

  + sign是符号函数，即$$sign(x) = \begin{cases}+1, & x\geq 0 \\ -1, & x<0 \\ \end{cases}$$
  
+ 感知机是一种线性分类模型，属于判别模型，其假设空间定义在特征空间的所有线性分类模型或线性分类器。
+ 几何解释：线性方程$ w \cdot x + b =0$ 对应于特征空间中的一个超平面S，其中w是超平面的法向量，b是超平面的截距。这个超平面将特征空间划分为两部分，两部分的点被分为正负两类，因此超平面S称为分离超平面。
+ 求感知机模型本质是求模型参数w，b。

## $\xi2.2$感知机学习策略

### $\xi2.2.1$ 数据集的线性可分性

+ 如果存在一个超平面S将数据集的正负实例点完全正确地划分到超平面的两侧，则数据集T为线性可分数据集。

### $\xi2.2.2$ 感知机学习策略

+ 损失函数选择

> 损失函数的一个自然选择是误分类点的总数，但是，这样的损失函数**不是参数$w,b$的连续可导函数，不易优化**
>
> 损失函数的另一个选择是误分类点到超平面$S$的总距离，这是感知机所采用的

感知机学习的经验风险函数(损失函数)
$$
L(w,b)=-\sum_{x_i\in M}y_i(w\cdot x_i+b)
$$
其中$M$是误分类点的集合

给定训练数据集$T$，损失函数$L(w,b)$是$w$和$b$的连续可导函数。

+ 感知机学习的策略是在假设空间中选取使损失函数式最小的模型参数w，b，即感知机模型。

## $\xi2.3$ 感知机学习算法

 ### $\xi2.3.1$ 原始形式

> 输入：$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}\\ x_i\in \cal X=\bf R^n\mit , y_i\in \cal Y\it =\{-1,+1\}, i=1,2,\dots,N; \ \ 0<\eta\leq 1$
>
> 输出：$w,b;f(x)=sign(w\cdot x+b)$
>
> 1. 选取初值$w_0,b_0$
> 2. 训练集中选取数据$(x_i,y_i)$
> 3. 如果$y_i(w\cdot x_i+b)\leq 0$
>
> $$
> w\leftarrow w+\eta y_ix_i \nonumber\\
> b\leftarrow b+\eta y_i
> $$
>
> 4. 转至(2)，直至训练集中没有误分类点

注意这个原始形式中的迭代公式，可以对$x$补1，将$w$和$b$合并在一起，合在一起的这个叫做扩充权重向量，书上有提到。

###  $\xi2.3.2$ 对偶形式

+ 对偶形式的基本思想是将$w$和$b$表示为实例$x_i$和标记$y_i$的线性组合的形式，通过求解其系数而求得$w$和$b$。

> 输入：$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}\\ x_i\in \cal{X}=\bf{R}^n , y_i\in \cal{Y} =\{-1,+1\}, i=1,2,\dots, N; 0< \eta \leq 1$
>
> 输出：
> $$
> \alpha ,b; f(x)=sign\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right)\nonumber\\
> \alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^T
> $$
>
> 1. $\alpha \leftarrow 0,b\leftarrow 0$
> 2. 训练集中选取数据$(x_i,y_i)$
> 3. 如果$y_i\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right) \leq 0$
>
> $$
> \alpha_i\leftarrow \alpha_i+\eta \nonumber\\
> b\leftarrow b+\eta y_i
> $$
>
> 4. 转至(2)，直至训练集中没有误分类点

### $\xi2.3.3$ 感知机算法的收敛性

+ 定理2.1(Novikoff）

  1. 存在满足条件$\|\hat{w}_{opt}\| = 1$的超平面$\hat{w}_{opt} \cdot \hat{x} = w_{opt} \cdot x+b_{opt} = 0$将训练数据完全正确分开，且存在$\gamma>0$，对所有 $i = 1,2,...,N$,
     $$
     y_i(\hat{w}_{opt} \cdot \hat{x}) = y_i(w_{opt} \cdot x_i+b_{opt}) > \gamma
     $$

  2. 令 $ R = max\|\hat{x}_{i}\|$，则感知机算法在训练集上的误分类次数k满足不等式
     $$
     k \leq (\frac{R}{\gamma})^2
     $$

+ 以上定理的证明在P42、P43，定理表明误分类的次数是有上界的，故感知机的学习算法原始形式是收敛的。
+ 感知机学习算法存在许多解，这些解取决于初值的选择，也依赖误分类点选择顺序。
+ 当训练集线性不可分时，感知机学习算法不收敛，迭代结果会发生震荡。

## $\xi2.4$ 其他

+ Gram Matrix ：$G=[x_i\cdot x_j]_{N\times N}$

+ 误分点条件：$y_i(w \cdot x_i+b) \le 0$