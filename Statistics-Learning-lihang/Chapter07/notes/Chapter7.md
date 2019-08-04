# $\xi7$ 支持向量机 - PART1

[TOC]

## $\xi7.1$ 线性可分支持向量机与硬间隔最大化

### $\xi7.1.1$ 线性可分支持向量机

+ 定义

  + 给定线性可分训练数据集，通过间隔最大化或等价地求解相应的凸二次规划问题学习得到的分离超平面
    $$
    w^*\cdot x+b^*=0
    $$
    

    相应的分类决策函数
    $$
    f(x)=sign(w^*\cdot x+b^*)
    $$

+ 感知机利用误分类最小的策略，求得分离超平面，解有无穷多种。
+ 线性可分支持向量机利用间隔最大化求最优分离超平面，解是唯一的。

### $\xi7.1.2$ 函数间隔和几何间隔

- 函数间隔

  - 对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的函数间隔为
    $$
    \hat \gamma_i=y_i(w\cdot x_i+b)
    $$
    定义超平面$(w,b)$关于训练数据集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的函数间隔之最小值，即
    $$
    \hat \gamma=\min_{i=1,\cdots,N}\hat\gamma_i
    $$

  - 函数间隔可以表示分类预测的**正确性**及**确信度**。

- 几何间隔

  - 对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔为
    $$
    \hat \gamma_i=y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})
    $$
    定义超平面$(w,b)$关于训练数据集$T$的几何间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的几何间隔之最小值，即
    $$
    \hat \gamma=\min_{i=1,\cdots,N}\hat\gamma_i
    $$

  - 超平面样本点的几何间隔一般是实例点到超平面的带符号的距离，当样本点被正确分类时就是实例点到超平面的距离

- 函数间隔和几何间隔的关系

  - 当$$||w|| = 1$$，那么函数间隔和几何间隔相等。
  - 当w，b成比例变化时，超平面不变，但函数间隔改变，几何间隔不变。

### $\xi7.1.3$ 间隔最大化

- 直观解释：对训练数据集找到几何间隔最大的超平面意味着充分大的确信度对训练数据进行划分。(点到超平面的距离表示确信度。)

+ 最大间隔分离超平面

  + 我们希望最大化超平面关于训练数据集的几何间隔，约束条件表示超平面关于每个点的几何间隔至少是$\gamma$

  $$
  \max_{w,b}\:\:\:\:\frac{\hat\gamma}{||w||}  \\
  s.t.\:\:\:\:y_i(w\cdot x_i+b) \geq \hat\gamma, \:\: i=1,2,...N
  $$

  + 又因为$$\hat\gamma$$的取值不影响最优化问题的解，取$$\hat\gamma = 1$$，则最大化$$\frac{1}{||w||}$$问题可以等价为最小化$$\frac{1}{2}||w||^2$$的问题。
    $$
    \min_{w,b}\:\:\:\:\frac{1}{2}||w||^2  \\
    s.t.\:\:\:\:y_i(w\cdot x_i+b)-1 \geq 0, \:\: i=1,2,...N
    $$

  + 这是一个凸二次规划问题。

+ 算法步骤

  1. 构造并求解约束最优化问题。`（但是并没有说明怎么求解约束最优化问题）`
  2. 由此得到分离超平面和分类决策函数。

+ 线性可分训练集的最大间隔分离超平面是存在且唯一的。（证明见P117-P118）

+ 在线性可分情况下，训练数据集的样本点中与分离超平面距离最近的样本点的实例称为支持向量。两边支持向量的距离称为间隔，等于$$\frac{2}{||w||}$$。

+ 由于支持向量在确定分离超平面中起着决定作用，所以将这种分类模型称为支持向量机。

### $\xi7.1.4$ 学习的对偶算法`（求解约束最优化问题的方法）`

- 对偶问题往往更容易求解
- 自然引入核函数，进而推广到非线性分类问题

+ 针对每个不等式约束，定义拉格朗日乘子$\alpha_i\ge0$，定义拉格朗日函数

$$
\begin{align}
L(w,b,\alpha)&=\frac{1}{2}w\cdot w-\left[\sum_{i=1}^N\alpha_i[y_i(w\cdot x_i+b)-1]\right]\\
&=\frac{1}{2}\left\|w\right\|^2-\left[\sum_{i=1}^N\alpha_i[y_i(w\cdot x_i+b)-1]\right]\\
&=\frac{1}{2}\left\|w\right\|^2-\sum_{i=1}^N\alpha_iy_i(w\cdot x_i+b)+\sum_{i=1}^N\alpha_i
\end{align}\\
\alpha_i \geq0, i=1,2,\dots,N
$$

​	其中$\alpha=(\alpha_1,\alpha_2,\dots,\alpha_N)^T$为拉格朗日乘子向量

+ **原始问题是极小极大问题，**根据**拉格朗日对偶性**，原始问题的**对偶问题是极大极小问题**:

$$
\max\limits_\alpha\min\limits_{w,b}L(w,b,\alpha)
$$

+ 算法步骤

  + 转换后的对偶问题

  $$
  \min\limits_\alpha \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
  s.t. \ \ \ \sum_{i=1}^N\alpha_iy_i=0\\
  \alpha_i\geq0, i=1,2,\dots,N
  $$

  + 计算下面两个公式

  $$
  \begin{align}
  w^*&=\sum_{i=1}^{N}\alpha_i^*y_ix_i\\
  b^*&=\color{red}y_j\color{black}-\sum_{i=1}^{N}\alpha_i^*y_i(x_i\cdot \color{red}x_j\color{black})
  \end{align}
  $$

  

  + 求得分离超平面和分类决策函数

$\alpha$不为零的点对应的实例为支持向量，通过支持向量可以求得$b$值

## $\xi7.2$ 线性支持向量机与软间隔最大化

### $\xi7.2.1$ 线性支持向量机

- 定义
  $$
  \begin{align}
  \min_{w,b,\xi} &\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i\\
  s.t. \ \ \ &y_i(w\cdot x_i+b)\geqslant1-\xi_i, i=1,2,\dots,N\\
  &\xi_i\geqslant0,i=1,2,\dots,N
  \end{align}
  $$

### $\xi7.2.2$ 学习的对偶算法

- 原始问题里面有两部分约束，涉及到两个拉格朗日乘子向量
  $$
  \begin{align}
  \min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
  s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
  &0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
  \end{align}
  $$

- 通过求解对偶问题， 得到$\alpha$，然后求解$w,b$的过程和之前一样

- 注意， 书后总结部分，有这样一句描述：**线性支持向量机的解$w^*$唯一但$b^*$不一定唯一**

  线性支持向量机是线性可分支持向量机的超集。

### $\xi7.2.3$ 支持向量

- 在线性不可分的情况下，支持向量的情况更为复杂一点，实例$x_i$到间隔边界的距离是$\frac{\xi_i}{||w||}$。
- 软间隔的支持向量或者在间隔边界上，或者在间隔边界与分离超平面之间，或者在分离超平面误分一侧。
  1. 若$\alpha^*_i <C$，则$\xi_i=0$，恰好落在间隔边界上。
  2. 若$\alpha^*_i =C$，则$0<\xi_i<1$，在间隔边界与分离超平面之间。
  3. 若$\alpha^*_i =C$，则$\xi_i=1$，在分离超平面上。
  4. 若$\alpha^*_i <C$，则$\xi_i>1$，在分离超平面误分一侧。

### $\xi7.2.4$ 合页损失函数

- 另一种解释，最小化目标函数

  $$\min\limits_{w,b} \sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2$$

  其中

  - 第一项是经验损失或经验风险，函数$L(y(w\cdot x+b))=[1-y(w\cdot x+b)]_+$称为合页损失，可以表示成$L = \max(1-y(w\cdot x+b), 0)$
  - 第二项是**系数为$\lambda$的$w$的$L_2$范数的平方**，是正则化项

- 书中这里通过定理7.4说明了用合页损失表达的最优化问题和线性支持向量机原始最优化问题的关系。
  $$
  \begin{align}
  \min_{w,b,\xi} &\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i\\
  s.t. \ \ \ &y_i(w\cdot x_i+b)\geqslant1-\xi_i, i=1,2,\dots,N\\
  &\xi_i\geqslant0,i=1,2,\dots,N
  \end{align}
  $$
  等价于
  $$
  \min\limits_{w,b} \sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2
  $$

+ 其他
  + 0-1损失函数不是连续可导
  + 合页损失认为是0-1损失函数的上界，在[AdaBoost](../CH08/README.md)中也有说明，指数损失也是0-1损失函数的上界，在[感知机](../CH02/README.md)中有提到`损失函数的自然选择是误分类点的个数`，这句在最开始见到的时候，可能不一定有上面图片的直觉。注意在本书[CH12](../CH12/README.md)中也有这个图，可以对比理解下。
  + 感知机误分类驱动， 选择函数间隔作为损失考虑分类的正确性，合页损失不仅要考虑分类正确， 还要考虑确信度足够高时损失才是0。

## $\xi7.3$ 非线性可分支持向量机

### $\xi7.3.1$ 核技巧

- 核技巧的想法是在学习和预测中只定义核函数$K(x,z)$，而不是显式的定义映射函数$\phi$

  通常，直接计算$K(x,z)$比较容易， 而通过$\phi(x)$和$\phi(z)$计算$K(x,z)$并不容易。
  $$
  \begin{align}
  W(\alpha)=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
  f(x)=sign\left(\sum_{i=1}^{N_s}\alpha_i^*y_i\phi(x_i)\cdot \phi(x)+b^*\right)=sign\left(\sum_{i=1}^{N_s}\alpha_i^*y_iK(x_i,x)+b^*\right) 
  \end{align}
  $$
  学习是隐式地在特征空间进行的，不需要显式的定义特征空间和映射函数。这样的技巧称为核技巧，核技巧不仅引用于支持向量机，而且应用于其他统计学习问题。

### $\xi7.3.2$ 正定核

- 正定核的充要条件是$K(x,z)$对应的Gram矩阵是半正定的。（证明见P139-P140）

### $\xi7.3.3$ 常用核函数

- 多项式核函数
- 高斯核函数
- 字符串核函数

### $\xi7.3.4$ 非线性支持向量机

- 算法

  - 构建最优化问题：
    $$
    \begin{align}
    \min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
    s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
    &0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
    \end{align}
    $$
    求解得到$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$

  - 选择$\alpha^*$的一个正分量计算
    $$
    b^*=y_j-\sum_{i=1}^N\alpha_i^*y_iK(x_i,x_j)
    $$

  - 构造决策函数
    $$
    f(x)=sign\left(\sum_{i=1}^N\alpha_i^*y_iK(x,x_i)+b^*\right)
    $$
    

## $\xi7.4$ 序列最小最优化算法

+ 支持向量机的学习问题可以形式化为求解凸二次规划问题]
  $$
  \begin{aligned}
  \min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
  s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
  &0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
  \end{aligned}
  $$

  + 这个问题中，变量是$\alpha$，一个变量$\alpha_i$对应一个样本点$(x_i,y_i)$，变量总数等于$N$

+ KKT 条件
  
  + KKT条件是该最优化问题的充分必要条件。

### $\xi7.4.1$ 两个变量二次规划的求解方法

- 整个SMO算法包括两~~步骤~~**部分**：

  1. 求解两个变量二次规划的解析方法
  2. 选择变量的启发式方法

  $$
  \begin{aligned}
  \min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
  s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
  &0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
  \end{aligned}
  $$

- 注意，这里是**两个部分**，而不是先后的两个步骤。

- 两变量二次规划求解

  选择两个变量$\alpha_1,\alpha_2$

  由等式约束可以得到

  $\alpha_1=-y_1\sum\limits_{i=2}^N\alpha_iy_i$

  所以这个问题实质上是单变量优化问题。
  $$
  \begin{align}
  \min_{\alpha_1,\alpha_2} W(\alpha_1,\alpha_2)=&\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2\nonumber\\
  &-(\alpha_1+\alpha_2)+y_1\alpha_1\sum_{i=3}^Ny_i\alpha_iK_{il}+y_2\alpha_2\sum_{i=3}^Ny_i\alpha_iK_{i2}\\
  s.t. \ \ \ &\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^Ny_i\alpha_i=\varsigma\\
  &0\leqslant\alpha_i\leqslant C, i=1,2
  \end{align}
  $$
  上面存在两个约束：

  1. **线性**等式约束
  2. 边界约束

### $\xi7.4.2$ 变量的选择方法

- 变量的选择方法
  1. 第一个变量$\alpha_1$
     外层循环
     违反KKT条件**最严重**的样本点
  2. 第二个变量$\alpha_2$
     内层循环
     希望能使$\alpha_2$有足够大的变化
  3. 计算阈值$b$和差值$E_i$

### $\xi7.4.3$ SMO算法

- 算法

  - 输入：训练数据集$T={(x_1,y_1),(x_2,y_2),\dots, (x_N,y_N)}$，其中$x_i\in\mathcal X=\bf R^n, y_i\in\mathcal Y=\{-1,+1\}, i=1,2,\dots,N$,精度$\epsilon$

    输出：近似解$\hat\alpha$

    1. 取初值$\alpha_0=0$，令$k=0$
    2. **选取**优化变量$\alpha_1^{(k)},\alpha_2^{(k)}$，解析求解两个变量的最优化问题，求得最优解$\alpha_1^{(k+1)},\alpha_2^{(k+1)}$，更新$\alpha$为$\alpha^{k+1}$
    3. 若在精度$\epsilon$范围内满足停机条件

    $$
    \sum_{i=1}^{N}\alpha_iy_i=0\\
    0\leqslant\alpha_i\leqslant C,i=1,2,\dots,N\\
    y_i\cdot g(x_i)=
    \begin{cases}
    \geqslant1,\{x_i|\alpha_i=0\}\\
    =1,\{x_i|0<\alpha_i<C\}\\
    \leqslant1,\{x_i|\alpha_i=C\}
    \end{cases}\\
    g(x_i)=\sum_{j=1}^{N}\alpha_jy_jK(x_j,x_i)+b
    $$

    则转4,否则，$k=k+1$，转2

    4. 取$\hat\alpha=\alpha^{(k+1)}$
