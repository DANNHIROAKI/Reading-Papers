# A Near-Linear Time Algorithm for the Chamfer Distance

一种近线性时间计算倒角距离的算法

Ainesh Bakshi

艾尼什·巴克希

MIT

ainesh@mit.edu

Piotr Indyk

皮奥特·因迪克

MIT

indyk@mit.edu

Rajesh Jayaram

拉杰什·贾亚拉姆

Google Research

谷歌研究院

rkjayaram@google.com

Sandeep Silwal

桑迪普·西尔瓦尔

MIT

silwal@mit.edu

Erik Waingarten

埃里克·温加滕

University of Pennsylvania

宾夕法尼亚大学

ewaingar@seas.upenn.edu

## Abstract

## 摘要

For any two point sets $A,B \subset  {\mathbb{R}}^{d}$ of size up to $n$ ,the Chamfer distance from $A$ to $B$ is defined as $\mathrm{{CH}}\left( {A,B}\right)  = \mathop{\sum }\limits_{{a \in  A}}\mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$ ,where ${d}_{X}$ is the underlying distance measure (e.g., the Euclidean or Manhattan distance). The Chamfer distance is a popular measure of dissimilarity between point clouds, used in many machine learning, computer vision, and graphics applications, and admits a straightforward $\mathcal{O}\left( {d{n}^{2}}\right)$ -time brute force algorithm. Further,the Chamfer distance is often used as a proxy for the more computationally demanding Earth-Mover (Optimal Transport) Distance. However,the quadratic dependence on $n$ in the running time makes the naive approach intractable for large datasets.

对于任意两个规模不超过$n$的点集$A,B \subset  {\mathbb{R}}^{d}$，从$A$到$B$的倒角距离定义为$\mathrm{{CH}}\left( {A,B}\right)  = \mathop{\sum }\limits_{{a \in  A}}\mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$，其中${d}_{X}$是基础距离度量（如欧氏距离或曼哈顿距离）。倒角距离是点云间差异性的常用度量，广泛应用于机器学习、计算机视觉和图形学领域，其暴力算法时间复杂度为$\mathcal{O}\left( {d{n}^{2}}\right)$。此外，倒角距离常被用作计算复杂度更高的地球移动（最优传输）距离的替代指标。然而，运行时间中对$n$的二次依赖使得朴素方法难以处理大规模数据集。

We overcome this bottleneck and present the first $\left( {1 + \varepsilon }\right)$ -approximate algorithm for estimating the Chamfer distance with a near-linear running time. Specifically, our algorithm runs in time $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ and is implementable. Our experiments demonstrate that it is both accurate and fast on large high-dimensional datasets. We believe that our algorithm will open new avenues for analyzing large high-dimensional point clouds. We also give evidence that if the goal is to report a $\left( {1 + \varepsilon }\right)$ -approximate mapping from $A$ to $B$ (as opposed to just its value),then any sub-quadratic time algorithm is unlikely to exist.

我们突破了这一瓶颈，首次提出能在近线性时间内估算倒角距离的$\left( {1 + \varepsilon }\right)$近似算法。具体而言，我们的算法运行时间为$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$且具备可实施性。实验表明该算法在处理大规模高维数据集时兼具准确性与高效性。我们相信该算法将为分析高维点云数据开辟新途径。同时我们证明，若需输出从$A$到$B$的$\left( {1 + \varepsilon }\right)$近似映射（而非仅计算距离值），则任何亚二次时间算法都难以存在。

## 1 Introduction

## 1 引言

For any two point sets $A,B \subset  {\mathbb{R}}^{d}$ of sizes up to $n$ ,the Chamfer distance ${}^{1}$ from $A$ to $B$ is defined as

对于任意规模不超过$n$的两个点集$A,B \subset  {\mathbb{R}}^{d}$，从$A$到$B$的倒角距离${}^{1}$定义为

$$
\mathrm{{CH}}\left( {A,B}\right)  = \mathop{\sum }\limits_{{a \in  A}}\mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right) 
$$

where ${d}_{X}$ is the underlying distance measure,such as the Euclidean or Manhattan distance. The Chamfer distance, and its weighted generalization called Relaxed Earth Mover Distance [KSKW15 AM19, are popular measures of dissimilarity between point clouds. They are widely used in machine learning (e.g.,[KSKW15,WCL ${}^{ + }{19}$ ]),computer vision (e.g.,[AS03,SMFW04,FSG17,JSQJ18]) and computer graphics $\left\lbrack  {{\mathrm{{LSS}}}^{ + }{19}}\right\rbrack$ . Subroutines for computing Chamfer distances are available in popular libraries, such as Tensorflow [ten23], Pytorch [pyt23] and PDAL [pda23]. In many of those applications (e.g., [KSKW15]) Chamfer distance is used as a faster proxy for the more computationally demanding Earth-Mover (Optimal Transport) Distance. Despite the popularity of Chamfer distance,the naïve algorithm for computing it has quadratic $\mathcal{O}\left( {n}^{2}\right)$ running time, which makes it difficult to use for large datasets. Faster approximate algorithms can be obtained by performing $n$ exact or approximate nearest neighbor queries,one for each point in $A$ . By utilizing the state of the art approximate nearest neighbor algorithms,this leads to $\left( {1 + \varepsilon }\right)$ - approximate estimators with running times of $\mathcal{O}\left( {n{\left( 1/\varepsilon \right) }^{\mathcal{O}\left( d\right) }\log n}\right)$ in low dimensions $\left\lbrack  {{\mathrm{{AMN}}}^{ + }{98}}\right\rbrack$ or roughly $\mathcal{O}\left( {d{n}^{1 + \frac{1}{2{\left( 1 + \varepsilon \right) }^{2} - 1}}}\right)$ in high dimensions [AR15]. Alas,the first bound suffers from exponential dependence on the dimension, while the second bound is significantly subquadratic only for relatively large approximation factors.

其中${d}_{X}$表示基础距离度量，如欧氏距离或曼哈顿距离。倒角距离（Chamfer distance）及其加权泛化形式——松弛推土机距离（Relaxed Earth Mover Distance）[KSKW15 AM19]，是点云差异度量的常用方法。这些方法广泛应用于机器学习（如[KSKW15,WCL ${}^{ + }{19}$]）、计算机视觉（如[AS03,SMFW04,FSG17,JSQJ18]）和计算机图形学$\left\lbrack  {{\mathrm{{LSS}}}^{ + }{19}}\right\rbrack$领域。主流计算库如TensorFlow [ten23]、PyTorch [pyt23]和PDAL [pda23]均提供倒角距离计算子程序。在多数应用场景中（如[KSKW15]），倒角距离被用作计算复杂度更高的推土机距离（最优传输距离）的快速替代方案。尽管倒角距离应用广泛，其朴素算法的$\mathcal{O}\left( {n}^{2}\right)$时间复杂度为平方级，难以处理大规模数据集。通过执行$n$精确或近似最近邻查询（对$A$中的每个点执行一次），可获得更快的近似算法。利用最先进的近似最近邻算法，可在低维空间$\left\lbrack  {{\mathrm{{AMN}}}^{ + }{98}}\right\rbrack$获得$\left( {1 + \varepsilon }\right)$近似估计器，其运行时间为$\mathcal{O}\left( {n{\left( 1/\varepsilon \right) }^{\mathcal{O}\left( d\right) }\log n}\right)$；在高维空间约为$\mathcal{O}\left( {d{n}^{1 + \frac{1}{2{\left( 1 + \varepsilon \right) }^{2} - 1}}}\right)$[AR15]。然而，前者存在维度指数依赖问题，后者仅在较大近似因子时才能显著优于平方复杂度。

---

<!-- Footnote -->

${}^{1}$ This is the definition adopted,e.g.,in [AS03]. Some other papers,e.g.,[FSG17],replace each distance term ${d}_{X}\left( {a,b}\right)$ with its square,e.g.,instead of $\parallel a - b{\parallel }_{2}$ they use $\parallel a - b{\parallel }_{2}^{2}$ . In this paper we focus on the first definition, as it emphasizes the connection to Earth Mover Distance and its relaxed weighted version in [KSKW15, AM19].

${}^{1}$该定义采用自[AS03]等文献。其他研究如[FSG17]将距离项${d}_{X}\left( {a,b}\right)$替换为其平方值，即使用$\parallel a - b{\parallel }_{2}^{2}$而非$\parallel a - b{\parallel }_{2}$。本文聚焦第一种定义，因其更能体现与推土机距离及其在[KSKW15, AM19]中提出的加权松弛版本的关联性。

<!-- Footnote -->

---

### 1.1 Our Results

### 1.1 研究成果

In this paper we overcome this bottleneck and present the first $\left( {1 + \varepsilon }\right)$ -approximate algorithm for estimating Chamfer distance that has a near-linear running time, both in theory and in practice. Concretely, our contributions are as follows:

本文突破该瓶颈，首次提出兼具理论意义与实践价值的$\left( {1 + \varepsilon }\right)$近似倒角距离估计算法，其时间复杂度接近线性。具体贡献如下：

- When the underlying metric ${d}_{X}$ is defined by the ${\ell }_{1}$ or ${\ell }_{2}$ norm,we give an algorithm that runs in time $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ and estimates the Chamfer distance up to $1 \pm  \varepsilon$ with ${99}\%$ probability (see Theorem 2.1). In general,our algorithm works for any metric ${d}_{X}$ supported by Locality-Sensitive Hash functions (see Definition A.1), with the algorithm running time depending on the parameters of those functions. Importantly, the algorithm is quite easy to implement, see Figures 1 and 2

- 当基础度量${d}_{X}$由${\ell }_{1}$或${\ell }_{2}$范数定义时，我们提出的算法运行时间为$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$，能以${99}\%$概率实现$1 \pm  \varepsilon$精度的倒角距离估计（见定理2.1）。该算法适用于任何支持局部敏感哈希函数（定义A.1）的度量${d}_{X}$，其时间复杂度取决于哈希函数参数。值得注意的是，该算法实现简便（见图1和图2）

- For the more general problem of reporting a mapping $g : A \rightarrow  B$ whose cost $\mathop{\sum }\limits_{{a \in  A}}{d}_{X}\left( {a,g\left( a\right) }\right)$ is within a factor of $1 + \varepsilon$ from $\operatorname{CH}\left( {A,B}\right)$ ,we show that,under a popular complexity-theoretic conjecture, an algorithm with a running time analogous to that of our estimation algorithm does not exist,even when ${d}_{X}\left( {a,b}\right)  = \parallel a - b{\parallel }_{1}$ . Specifically,under a Hitting Set Conjecture [Wil18], any such algorithm must run in time $\Omega \left( {n}^{2 - \delta }\right)$ for any constant $\delta  > 0$ ,even when the dimension $d = \Theta \left( {{\log }^{2}n}\right)$ and $\varepsilon  = \frac{\Theta \left( 1\right) }{d}$ . (In contrast,our estimation algorithm runs in near-linear time for such parameters). This demonstrates that, for the Chamfer distance, estimation is significantly easier than reporting.

- 针对报告一个成本$\mathop{\sum }\limits_{{a \in  A}}{d}_{X}\left( {a,g\left( a\right) }\right)$在$\operatorname{CH}\left( {A,B}\right)$的$1 + \varepsilon$倍范围内的映射$g : A \rightarrow  B$这一更普遍问题，我们证明在主流复杂性理论猜想下，即使当${d}_{X}\left( {a,b}\right)  = \parallel a - b{\parallel }_{1}$时，也不存在运行时间类似于我们估计算法的解决方案。具体而言，根据命中集猜想[Wil18]，任何此类算法必须在$\Omega \left( {n}^{2 - \delta }\right)$时间内运行（对于任意常数$\delta  > 0$），即使维度$d = \Theta \left( {{\log }^{2}n}\right)$和$\varepsilon  = \frac{\Theta \left( 1\right) }{d}$时亦然（相比之下，我们的估计算法在此类参数下运行于近线性时间）。这表明对于Chamfer距离而言，估算远比报告简单。

- We experimentally evaluate our algorithm on real and synthetic data sets. Our experiments demonstrate the effectiveness of our algorithm for both low and high dimensional datasets and across different dataset scales. Overall,it is much faster $\left( { > 5\mathbf{x}}\right)$ than brute force (even accelerated with KD-trees) and both faster and more sample efficient (5-10x) than simple uniform sampling We demonstrate the scalability of our method by running it on billion-scale Big-ANN-Benchmarks datasets $\left\lbrack  {{\mathrm{{SWA}}}^{ + }{22}}\right\rbrack$ ,where it runs up to $\mathbf{{50x}}$ faster than optimized brute force. In addition,our method is robust to different datasets: while uniform sampling performs reasonably well for some datasets in our experiments,it performs poorly on datasets where the distances from points in $A$ to their neighbors in $B$ vary significantly. In such cases,our algorithm is able to adapt its importance sampling probabilities appropriately and obtain significant improvements over uniform sampling.

- 我们在真实与合成数据集上实验评估算法性能。实验证明该算法在低维与高维数据集、不同规模场景下均表现优异。总体而言，其速度远超暴力搜索（即使采用KD树加速），比简单均匀采样快5-10倍且样本效率更高。通过在十亿级Big-ANN-Benchmarks数据集$\left\lbrack  {{\mathrm{{SWA}}}^{ + }{22}}\right\rbrack$上的测试，我们验证了方法的可扩展性——其运行速度最高可比优化暴力搜索快$\mathbf{{50x}}$倍。此外，本方法对数据集差异具有鲁棒性：虽然均匀采样在某些实验中表现尚可，但在$A$到$B$点间距离差异显著的数据集上效果欠佳。此时我们的算法能自适应调整重要性采样概率，较均匀采样获得显著提升。

## 2 Algorithm and Analysis

## 2 算法与分析

In this section, we establish our main result for estimating Chamfer distance:

本节我们建立Chamfer距离估算的主要结论：

Theorem 2.1 (Estimating Chamfer Distance in Nearly Linear Time). Given as input two datasets $A,B \subset  {\mathbb{R}}^{d}$ such that $\left| A\right| ,\left| B\right|  \leq  n$ ,and an accuracy parameter $0 < \varepsilon  < 1$ ,Chamfer-Estimate runs in time $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ and outputs an estimator $\eta$ such that with probability at least ${99}/{100}$ ,

定理2.1（近线性时间估算Chamfer距离）。给定输入数据集$A,B \subset  {\mathbb{R}}^{d}$（满足$\left| A\right| ,\left| B\right|  \leq  n$）和精度参数$0 < \varepsilon  < 1$，Chamfer-Estimate在$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$时间内运行，并以至少${99}/{100}$概率输出估计量$\eta$满足：

$$
\left( {1 - \varepsilon }\right) \mathrm{{CH}}\left( {A,B}\right)  \leq  \eta  \leq  \left( {1 + \varepsilon }\right) \mathrm{{CH}}\left( {A,B}\right) ,
$$

when the underlying metric is Euclidean $\left( {\ell }_{2}\right)$ or Manhattan $\left( {\ell }_{1}\right)$ distance.

当底层度量采用欧氏$\left( {\ell }_{2}\right)$或曼哈顿$\left( {\ell }_{1}\right)$距离时。

For ease of exposition, we make the simplifying assumption that the underlying metric is Manhattan distance,i.e. ${d}_{X}\left( {a,b}\right)  = \parallel a - b{\parallel }_{1}$ . Our algorithm still succeeds whenever the underlying metric admits a locality-sensitive hash function (see Definition A.1).

为便于阐述，我们简化假设底层度量采用曼哈顿距离（即${d}_{X}\left( {a,b}\right)  = \parallel a - b{\parallel }_{1}$）。只要底层度量允许局部敏感哈希函数（见定义A.1），本算法仍可成立。

Uniform vs Importance Sampling. A natural algorithm for estimating $\mathrm{{CH}}\left( {A,B}\right)$ proceeds by uniform sampling: sample an $a \in  A$ uniformly at random and explicitly compute $\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ . In general,we can compute the estimator $\widehat{z}$ for $\mathrm{{CH}}\left( {A,B}\right)$ by averaging over $s$ uniformly chosen samples,resulting in runtime $\mathcal{O}\left( {nds}\right)$ . It is easy to see that the resulting estimator is un-biased,i.e. $\mathbf{E}\left\lbrack  \widehat{z}\right\rbrack   = \mathrm{{CH}}\left( {A,B}\right)$ . However,if a small constant fraction of elements in $A$ contribute significantly to

均匀采样与重要性采样的对比。估算$\mathrm{{CH}}\left( {A,B}\right)$的自然算法采用均匀采样：随机均匀采样$a \in  A$并显式计算$\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$。通常，我们可以通过对$s$个均匀选取的样本取平均值来计算$\mathrm{{CH}}\left( {A,B}\right)$的估计量$\widehat{z}$，从而得到运行时间$\mathcal{O}\left( {nds}\right)$。易见该估计量是无偏的，即$\mathbf{E}\left\lbrack  \widehat{z}\right\rbrack   = \mathrm{{CH}}\left( {A,B}\right)$。然而，若$A$中仅有少量元素对结果贡献显著

<!-- Media -->

---

Subroutine Chamfer-Estimate(A,B,T)

子程序Chamfer-Estimate(A,B,T)

Input: Two subsets $A,B \subset  {\mathbb{R}}^{d}$ of size at most $n$ ,and a parameter $T \in  \mathbb{N}$ .

输入：两个最大规模为$n$的子集$A,B \subset  {\mathbb{R}}^{d}$，及参数$T \in  \mathbb{N}$

Output: A number $\mathbf{\eta } \in  {\mathbb{R}}_{ \geq  0}$ .

输出：数值$\mathbf{\eta } \in  {\mathbb{R}}_{ \geq  0}$

1. Execute the algorithm CrudeNN(A,B),and let the output be a set of positive real numbers

1. 执行CrudeNN(A,B)算法，输出为一组始终满足<b1></b1>的正实数集

	${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ which always satisfy ${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ . Let $\mathbf{D} \mathrel{\text{:=}} \mathop{\sum }\limits_{{a \in  A}}{\mathbf{D}}_{a}$ .

	${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$。令$\mathbf{D} \mathrel{\text{:=}} \mathop{\sum }\limits_{{a \in  A}}{\mathbf{D}}_{a}$

2. Construct the probability distribution $\mathcal{D}$ ,supported on the set $A$ ,which satisfies that for every

2. 在集合$A$上构建概率分布$\mathcal{D}$，确保对每个元素

	$a \in  A,$

$$
\mathop{\Pr }\limits_{{\mathbf{x} \sim  \mathcal{D}}}\left\lbrack  {\mathbf{x} = a}\right\rbrack   \mathrel{\text{:=}} \frac{{\mathbf{D}}_{a}}{\mathbf{D}}.
$$

3. For $\ell  \in  \left\lbrack  T\right\rbrack$ ,sample ${\mathbf{x}}_{\ell } \sim  \mathcal{D}$ and spend $\mathcal{O}\left( {\left| B\right| d}\right)$ time to compute

3. 对于$\ell  \in  \left\lbrack  T\right\rbrack$，采样${\mathbf{x}}_{\ell } \sim  \mathcal{D}$并耗费$\mathcal{O}\left( {\left| B\right| d}\right)$时间计算

$$
{\mathbf{\eta }}_{\ell } \mathrel{\text{:=}} \frac{\mathbf{D}}{{\mathbf{D}}_{{\mathbf{x}}_{\ell }}} \cdot  \mathop{\min }\limits_{{b \in  B}}{\begin{Vmatrix}{\mathbf{x}}_{\ell } - b\end{Vmatrix}}_{1}.
$$

4. Output

4. 输出

$$
\mathbf{\eta } \mathrel{\text{:=}} \frac{1}{T}\mathop{\sum }\limits_{{\ell  = 1}}^{T}{\mathbf{\eta }}_{\ell }.
$$

---

Figure 1: The Chamfer-Estimate Algorithm.

图1：Chamfer-Estimate算法

<!-- Media -->

$\mathrm{{CH}}\left( {A,B}\right)$ ,then $s = \Omega \left( n\right)$ samples could be necessary to obtain,say,a 1% relative error estimate with constant probability. Since each sample requires a linear scan to find the nearest neighbor, this would result in a quadratic runtime.

$\mathrm{{CH}}\left( {A,B}\right)$，则可能需要$s = \Omega \left( n\right)$次采样才能以恒定概率获得1%相对误差的估计。由于每次采样需线性扫描寻找最近邻，这将导致平方级时间复杂度

While such an approach has good empirical performance for well-behaved datasets, it does not work for data sets where the distribution of the distances from points in $A$ to their nearest neighbors in $B$ is skewed. Further, it is computationally prohibitive to verify the quality of the approximation given by uniform sampling. Towards proving Theorem 2.1, it is paramount to obtain an algorithm that works regardless of the structure of the input dataset.

虽然该方法在数据分布良好时具有优越的实证性能，但对于$A$中各点到$B$中最近邻距离分布倾斜的数据集则失效。此外，验证均匀采样所得近似解的质量在计算上是不可行的。为证明定理2.1，关键在于获得不受输入数据集结构影响的普适算法

A more nuanced approach is to perform importance sampling where we sample $a \in  A$ with probability proportional to its contribution to $\operatorname{CH}\left( {A,B}\right)$ . In particular,if we had access to a distribution, ${\mathbf{D}}_{a}$ ,over elements $a \in  A$ such that, $\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1} \leq  {\mathbf{D}}_{a} \leq  \lambda \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ ,for some parameter $\lambda  > 1$ , then sampling $O\left( \lambda \right)$ samples results in an estimator $\widehat{z}$ that is within $1\%$ relative error to the true answer with probability at least ${99}\%$ . Formally,we consider the estimator defined in Algorithm 1,where we assume access to $\operatorname{CrudeNN}\left( {A,B}\right)$ ,a sub-routine which receives as input $A$ and $B$ and outputs estimates ${\mathbf{D}}_{a} \in  {\mathbb{R}}_{ \geq  0}$ for each $a \in  A$ which is guaranteed to be an upper bound for $\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ . Based on the values ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ we construct an importance sampling distribution $\mathcal{D}$ supported on $A$ . As a result, we obtain the following lemma:

一种更为精细的方法是采用重要性采样，即以与$a \in  A$对$\operatorname{CH}\left( {A,B}\right)$贡献度成比例的概率进行采样。具体而言，若我们能获得元素$a \in  A$上的分布${\mathbf{D}}_{a}$，使得对于某参数$\lambda  > 1$满足$\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1} \leq  {\mathbf{D}}_{a} \leq  \lambda \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$，则采样$O\left( \lambda \right)$次可得到估计量$\widehat{z}$，该估计量以至少${99}\%$的概率将真实答案的相对误差控制在$1\%$范围内。形式化地，我们考虑算法1中定义的估计量，其中假设可调用子程序$\operatorname{CrudeNN}\left( {A,B}\right)$——该子程序接收输入$A$和$B$，并为每个$a \in  A$输出估计值${\mathbf{D}}_{a} \in  {\mathbb{R}}_{ \geq  0}$，该值保证是$\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$的上界。基于数值${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$，我们构建了在$A$上支撑的重要性采样分布$\mathcal{D}$。由此，我们得到如下引理：

Lemma 2.2 (Variance Bounds for Chamfer Estimate). Let $n,d \in  \mathbb{N}$ and suppose $A,B$ are two subsets of ${\mathbb{R}}^{d}$ of size at most $n$ . For any $T \in  \mathbb{N}$ ,the output $\eta$ of Chamfer-Estimate(A,B,T) satisfies

引理2.2（倒角估计的方差界限）。设$n,d \in  \mathbb{N}$且假设$A,B$是${\mathbb{R}}^{d}$的两个子集，其大小最多为$n$。对于任意$T \in  \mathbb{N}$，倒角估计算法Chamfer-Estimate(A,B,T)的输出$\eta$满足

$$
\mathbf{E}\left\lbrack  \mathbf{\eta }\right\rbrack   = \mathrm{{CH}}\left( {A,B}\right) ,
$$

$$
\operatorname{Var}\left\lbrack  \mathbf{\eta }\right\rbrack   \leq  \frac{1}{T} \cdot  \mathrm{{CH}}{\left( A,B\right) }^{2}\left( {\frac{\mathbf{D}}{\mathrm{{CH}}\left( {A,B}\right) } - 1}\right) ,
$$

for $D$ from Line 1 in Figure 1 The expectations and variance are over the randomness in the samples of Line 3 of Chamfer-Estimate(A,B,T). In particular,

对于图1中第1行的$D$，期望和方差是针对Chamfer-Estimate(A,B,T)第3行样本随机性计算的。具体而言，

$$
\Pr \left\lbrack  {\left| {\eta  - \mathrm{{CH}}\left( {A,B}\right) }\right|  \geq  \varepsilon  \cdot  \mathrm{{CH}}\left( {A,B}\right) }\right\rbrack   \leq  \frac{1}{{\varepsilon }^{2} \cdot  T}\left( {\frac{D}{\mathrm{{CH}}\left( {A,B}\right) } - 1}\right) .
$$

The proof follows from a standard analysis of importance sampling and is deferred to Appendix A Observe,if $\mathbf{D} \leq  \lambda \mathrm{{CH}}\left( {A,B}\right)$ ,it suffices to sample $T = O\left( {\lambda /{\varepsilon }^{2}}\right)$ points in $A$ ,leading to a running time of $O\left( {{nd\lambda }/{\varepsilon }^{2}}\right)$ .

该证明源自重要性抽样的标准分析过程，具体推导详见附录A。注意，若$\mathbf{D} \leq  \lambda \mathrm{{CH}}\left( {A,B}\right)$成立，则只需在$A$中采样$T = O\left( {\lambda /{\varepsilon }^{2}}\right)$个点即可，这将使运行时间降至$O\left( {{nd\lambda }/{\varepsilon }^{2}}\right)$。

---

Subroutine CrudeNN(A,B)

子程序CrudeNN(A,B)

Input: Two subsets $A,B$ of a metric space $\left( {X,\parallel  \cdot  {\parallel }_{1}}\right)$ of size at most $n$ such that all non-zero

输入：度量空间(metric space) $\left( {X,\parallel  \cdot  {\parallel }_{1}}\right)$ 中两个大小不超过 $n$ 的子集 $A,B$，使得所有非零

distances between any point in $A$ and any point in $B$ is between 1 and poly $\left( {n/\varepsilon }\right)$ . We assume

$A$ 中任意一点与 $B$ 中任意一点的距离介于1与多项式$\left( {n/\varepsilon }\right)$之间。我们假设

access to a locality-sensitive hash family at every scale $\mathcal{H}\left( r\right)$ for any $r \geq  0$ satisfying conditions

获取各尺度下的局部敏感哈希族(Locality-Sensitive Hash Family) $\mathcal{H}\left( r\right)$ 对于满足条件的任意$r \geq  0$

of Definition A.1 (We show in Appendix A that,for ${\ell }_{1}$ and ${\ell }_{2}$ ,the desired hash families exist,

根据定义A.1（我们在附录A中证明，对于${\ell }_{1}$和${\ell }_{2}$，所需的哈希族存在，

and that distances between 1 and poly $\left( {n/\varepsilon }\right)$ is without loss of generality).

且1与多项式$\left( {n/\varepsilon }\right)$之间的距离不失一般性）。

Output: A list of numbers ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ where ${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ .

输出：一个数字列表 ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$，其中 ${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ 。

			1. We instantiate $L = \mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$ and for $i \in  \{ 0,\ldots ,L\}$ ,we let ${r}_{i} = {2}^{i}$ .

			1. 实例化$L = \mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$，对于$i \in  \{ 0,\ldots ,L\}$，我们设${r}_{i} = {2}^{i}$。

			2. For each $i \in  \{ 0,\ldots ,L\}$ sample a hash function ${\mathbf{h}}_{i} : X \rightarrow  U$ from ${\mathbf{h}}_{i} \sim  \mathcal{H}\left( {r}_{i}\right)$ .

			2. 为每个$i \in  \{ 0,\ldots ,L\}$样本从${\mathbf{h}}_{i} \sim  \mathcal{H}\left( {r}_{i}\right)$中采样哈希函数${\mathbf{h}}_{i} : X \rightarrow  U$。

			3. For each $a \in  A$ ,find the smallest $i \in  \{ 0,\ldots ,L\}$ for which there exists a point $b \in  B$

			3. 对于每个$a \in  A$，找到存在对应点$b \in  B$的最小$i \in  \{ 0,\ldots ,L\}$

				with ${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( b\right)$ ,and set ${\mathbf{D}}_{a} = \parallel a - b{\parallel }_{1}$ .

				满足${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( b\right)$，并设${\mathbf{D}}_{a} = \parallel a - b{\parallel }_{1}$。

						- The above may be done by first hashing each point $b \in  B$ and $i \in  \{ 0,\ldots ,L\}$

						- - 上述操作可通过先对每个点$b \in  B$和$i \in  \{ 0,\ldots ,L\}$进行哈希处理实现

							according to ${\mathbf{h}}_{i}\left( b\right)$ . Then,for each $a \in  A$ ,we iterate through $i \in  \{ 0,\ldots ,L\}$

							依据${\mathbf{h}}_{i}\left( b\right)$。接着，对于每个$a \in  A$，我们遍历$i \in  \{ 0,\ldots ,L\}$

							while hashing $a$ according to ${\mathbf{h}}_{i}\left( a\right)$ until the first $b \in  B$ with ${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( b\right)$ is

							同时根据${\mathbf{h}}_{i}\left( a\right)$对$a$进行哈希，直到找到首个满足${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( b\right)$的$b \in  B$

							found.

							。

---

Figure 2: The CrudeNN Algorithm.

图2：CrudeNN算法。

Obtaining importance sampling probabilities. It remains to show how to implement the $\operatorname{CrudeNN}\left( {A,B}\right)$ subroutine to obtain the distribution over elements in $A$ which is a reasonable over-estimator of the true probabilities. A natural first step is to consider performing an $\mathcal{O}\left( {\log n}\right)$ -approximate nearest neighbor search (NNS): for every ${a}^{\prime } \in  A$ ,find ${b}^{\prime } \in  B$ satisfying ${\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}/\mathop{\min }\limits_{{b \in  B}}{\begin{Vmatrix}{a}^{\prime } - b\end{Vmatrix}}_{1} = \mathcal{O}\left( {\log n}\right)$ . This leads to the desired guarantees on ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ . Unfortunately,the state of the art algorithms for $\mathcal{O}\left( {\log n}\right)$ -approximate NNS,even under the ${\ell }_{1}$ norm, posses extraneous poly $\left( {\log n}\right)$ factors in the runtime,resulting in a significantly higher running time. These factors are even higher for the ${\ell }_{2}$ norm. Therefore,instead of performing a direct reduction to approximate NNS, we open up the approximate NNS black-box and give a simple algorithm which directly satisfies our desired guarantees on ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ .

获取重要性采样概率。仍需说明如何实现$\operatorname{CrudeNN}\left( {A,B}\right)$子程序来获得$A$中元素的分布，这是对真实概率的合理高估。自然的初步考虑是进行$\mathcal{O}\left( {\log n}\right)$-近似最近邻搜索(NNS)：对每个${a}^{\prime } \in  A$，找到满足${\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}/\mathop{\min }\limits_{{b \in  B}}{\begin{Vmatrix}{a}^{\prime } - b\end{Vmatrix}}_{1} = \mathcal{O}\left( {\log n}\right)$的${b}^{\prime } \in  B$。这将产生${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$的预期保证。遗憾的是，即便在${\ell }_{1}$范数下，当前最先进的$\mathcal{O}\left( {\log n}\right)$-近似NNS算法在运行时间中仍存在多余的多项式$\left( {\log n}\right)$因子，导致运行时间显著增加。对于${\ell }_{2}$范数，这些因子更为突出。因此，我们没有直接进行近似NNS的归约，而是打开这个近似NNS黑盒，给出一个直接满足${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$预期保证的简单算法。

To begin with, we assume that the aspect ratio of all pair-wise distances is bounded by a fixed polynomial,poly $\left( {n/\varepsilon }\right)$ (we defer the reduction from an arbitrary input to one with polynomially bounded aspect ratio to Lemma A.3). We proceed via computing $\mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$ different (randomized) partitions of the dataset $A \cup  B$ . The $i$ -th partition,for $1 \leq  i \leq  \mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$ ,can be written as $A \cup  B = { \cup  }_{j}{\mathcal{P}}_{j}^{i}$ and approximately satisfies the property that points in $A \cup  B$ that are at distance at most ${2}^{i}$ will be in the same partition ${\mathcal{P}}_{j}^{i}$ with sufficiently large probability. To obtain these components, we use a family of locality-sensitive hash functions, whose formal properties are given in Definition A.1. Intuitively, these hash functions guarantee that:

首先，我们假设所有成对距离的纵横比受限于一个固定多项式poly$\left( {n/\varepsilon }\right)$（将任意输入归约至多项式有界纵横比的过程详见引理A.3）。我们通过计算$\mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$种不同的（随机化）数据集$A \cup  B$划分来推进。对于$1 \leq  i \leq  \mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$而言，第$i$个划分可表示为$A \cup  B = { \cup  }_{j}{\mathcal{P}}_{j}^{i}$，其近似满足以下性质：在$A \cup  B$中相距不超过${2}^{i}$的点将以足够高的概率被划分至同一分区${\mathcal{P}}_{j}^{i}$。为获得这些组件，我们使用一族局部敏感哈希函数，其形式化性质见定义A.1。直观而言，这些哈希函数保证：

1. For each ${a}^{\prime } \in  A$ ,its true nearest neighbor ${b}^{\prime } \in  B$ falls into the same component as ${a}^{\prime }$ in the ${i}_{0}$ -th partition,where ${2}^{{i}_{0}} = \Theta {\left( {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}\right) }^{2}$ ,and

1. 对于每个${a}^{\prime } \in  A$，其真实最近邻${b}^{\prime } \in  B$在第${i}_{0}$个划分中与${a}^{\prime }$同属一个组件，其中${2}^{{i}_{0}} = \Theta {\left( {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}\right) }^{2}$；且

2. Every other extraneous $b \neq  {b}^{\prime }$ is not in the same component as ${a}^{\prime }$ for each $i < {i}_{0}$ .

2. 对于每个$i < {i}_{0}$，其他无关点$b \neq  {b}^{\prime }$均不与${a}^{\prime }$同属一个组件。

It is easy to check that any hash function that satisfies the aforementioned guarantees yields a valid set of distances ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ as follows: for every ${a}^{\prime } \in  A$ ,find the smallest ${i}_{0}$ for which there exists a ${b}^{\prime } \in  B$ in the same component as ${a}^{\prime }$ in the ${i}_{0}$ -th partition. Then set ${\mathbf{D}}_{{a}^{\prime }} = {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}$ . Intuitively,the ${b}^{\prime }$ we find for any fixed ${a}^{\prime }$ in this procedure will have distance that is at least the closest neighbor in $B$ and with good probability,it won’t be too much larger. A caveat here is that we cannot show the above guarantee holds for ${2}^{{i}_{0}} = \Theta \left( {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}\right)$ . Instead,we obtain the slightly weaker guarantee that, in the expectation,the partition ${b}^{\prime }$ lands in is a $\mathcal{O}\left( {\log n}\right)$ -approximation to the minimum distance,i.e. ${2}^{{i}_{0}} = \Theta \left( {\log n \cdot  {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}}\right)$ . Therefore,after running CrudeNN(A,B),setting $\lambda  = \log n$ suffices for our $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ time algorithm. We formalize this argument in the following lemma:

容易验证，任何满足上述保证的哈希函数均可生成有效距离集${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$：对于每个${a}^{\prime } \in  A$，找到最小的${i}_{0}$使得在第${i}_{0}$个划分中存在与${a}^{\prime }$同属一个组件的${b}^{\prime } \in  B$，随后设定${\mathbf{D}}_{{a}^{\prime }} = {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}$。直观上，本流程中为任意固定${a}^{\prime }$找到的${b}^{\prime }$，其距离至少等于$B$中的最近邻，且大概率不会过大。需注意，我们无法证明上述保证对${2}^{{i}_{0}} = \Theta \left( {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}\right)$成立，而是获得稍弱的保证：在期望意义上，划分${b}^{\prime }$所属的分区是最小距离的$\mathcal{O}\left( {\log n}\right)$近似，即${2}^{{i}_{0}} = \Theta \left( {\log n \cdot  {\begin{Vmatrix}{a}^{\prime } - {b}^{\prime }\end{Vmatrix}}_{1}}\right)$。因此，运行CrudeNN(A,B)后，设定$\lambda  = \log n$即可满足我们$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$时间算法的需求。我们在下述引理中形式化该论证：

Lemma 2.3 (Oversampling with bounded Aspect Ratio). Let $\left( {X,{d}_{X}}\right)$ be a metric space with a locality-sensitive hash family at every scale (see Definition A.1). Consider two subsets $A,B \subset  X$ of

引理2.3（有界纵横比下的过采样）。设$\left( {X,{d}_{X}}\right)$为在每个尺度都具有局部敏感哈希族的度量空间（见定义A.1）。考虑其两个子集$A,B \subset  X$

---

<!-- Footnote -->

${}^{2}$ Recall we assumed all distances are between 1 and poly(n)resulting in only $\mathcal{O}\left( {\log n}\right)$ different partitions

${}^{2}$ 回顾我们假设所有距离值在1到poly(n)之间，因此仅产生$\mathcal{O}\left( {\log n}\right)$种不同划分

<!-- Footnote -->

---

size at most $n$ and any $\varepsilon  \in  \left( {0,1}\right)$ satisfying

大小至多为$n$且满足任意$\varepsilon  \in  \left( {0,1}\right)$

$$
1 \leq  \mathop{\min }\limits_{\substack{{a \in  A,b \in  B} \\  {a \neq  b} }}{d}_{X}\left( {a,b}\right)  \leq  \mathop{\max }\limits_{{a \in  A,b \in  B}}{d}_{X}\left( {a,b}\right)  \leq  \operatorname{poly}\left( {n/\varepsilon }\right) .
$$

Algorithm $\overline{2}$ CrudeNN(A,B),outputs a list of (random) positive numbers ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ which satisfy the following two guarantees:

算法$\overline{2}$ CrudeNN(A,B)输出一组满足以下两个保证的（随机）正数列表${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$：

- With probability 1,every $a \in  A$ satisfies ${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$ .

- 概率为1时，每个$a \in  A$都满足${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$

- For every $a \in  A,\mathbf{E}\left\lbrack  {\mathbf{D}}_{a}\right\rbrack   \leq  \mathcal{O}\left( {\log n}\right)  \cdot  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$ .

- 对于每个$a \in  A,\mathbf{E}\left\lbrack  {\mathbf{D}}_{a}\right\rbrack   \leq  \mathcal{O}\left( {\log n}\right)  \cdot  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$

Further,Algorithm 2,runs in time $\mathcal{O}\left( {\operatorname{dn}\log \left( {n/\varepsilon }\right) }\right)$ time,assuming that each function used in the algorithm can be evaluated in $\mathcal{O}\left( d\right)$ time.

此外，算法2在$\mathcal{O}\left( {\operatorname{dn}\log \left( {n/\varepsilon }\right) }\right)$时间内运行，前提是算法中使用的每个函数可在$\mathcal{O}\left( d\right)$时间内完成计算

Proof Sketch for Theorem 2.1 Given the lemmas above, it is straight-forward to complete the proof of Theorem 2.1. First,we reduce to the setting where the aspect ratio is $\operatorname{poly}\left( {n/\varepsilon }\right)$ (see Lemma A. 3 for a formal reduction). We then invoke Lemma 2.3 and apply Markov's inequality to obtain a set of distances ${\mathbf{D}}_{a}$ such that with probability at least ${99}/{100}$ ,for each $a \in  A,\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1} \leq  {\mathbf{D}}_{a}$ and $\mathop{\sum }\limits_{{a \in  A}}{\mathbf{D}}_{a} \leq  \mathcal{O}\left( {\log \left( n\right) }\right) \mathrm{{CH}}\left( {A,B}\right)$ . We then invoke Lemma 2.2 and set the number of samples, $T = \mathcal{O}\left( {\log \left( n\right) /{\varepsilon }^{2}}\right)$ . The running time of our algorithm is then given by the time of CrudeNN(A,B), which is $O\left( {{nd}\log \left( {n/\varepsilon }\right) }\right)$ ,and the time needed to evaluate the estimator in Lemma 2.2,requiring $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ time. Refer to Section A for the full proof.

定理2.1的证明概要 根据上述引理，完成定理2.1的证明是直截了当的。首先我们将问题归约到纵横比为$\operatorname{poly}\left( {n/\varepsilon }\right)$的场景（正式归约过程见引理A.3）。接着调用引理2.3并应用马尔可夫不等式，得到一组距离值${\mathbf{D}}_{a}$，使得至少有${99}/{100}$的概率，对于每个$a \in  A,\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1} \leq  {\mathbf{D}}_{a}$和$\mathop{\sum }\limits_{{a \in  A}}{\mathbf{D}}_{a} \leq  \mathcal{O}\left( {\log \left( n\right) }\right) \mathrm{{CH}}\left( {A,B}\right)$成立。随后调用引理2.2并设置采样次数$T = \mathcal{O}\left( {\log \left( n\right) /{\varepsilon }^{2}}\right)$。算法运行时间由CrudeNN(A,B)的$O\left( {{nd}\log \left( {n/\varepsilon }\right) }\right)$时间，以及评估引理2.2中估计量所需的$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$时间共同决定。完整证明参见附录A节

## 3 Experiments

## 3 实验

We perform an empirical evaluation of our Chamfer distance estimation algorithm.

我们对倒角距离估计算法进行了实证评估

Summary of Results Our experiments demonstrate the effectiveness of our algorithm for both low and high dimensional datasets and across different dataset sizes. Overall, it is much faster than brute force (even accelerated with KD-trees). Further, our algorithm is both faster and more sample-efficient than uniform sampling. It is also robust to different datasets: while uniform sampling performs well for most datasets in our experiments, it performs poorly on datasets where the distances from points in $A$ to their neighbors in $B$ vary significantly. In such cases,our algorithm is able to adapt its importance sampling probabilities appropriately and obtain significant improvements over uniform sampling.

结果概要 实验证明我们的算法在低维和高维数据集、不同规模数据集上均表现优异。总体而言，其速度远快于暴力搜索（即使采用KD树加速）。此外，相比均匀采样，我们的算法兼具更快速度和更高样本效率。算法对不同数据集具有鲁棒性：虽然均匀采样在我们大多数实验数据集中表现良好，但在$A$到$B$的邻点距离差异显著的数据集上表现欠佳。这种情况下，我们的算法能自适应调整重要性采样概率，相较均匀采样获得显著提升

<!-- Media -->

<table><tr><td>Dataset</td><td>$\left| A\right| ,\left| B\right|$</td><td>$d$</td><td>Experiment</td><td>$\mathbf{{Metric}}$</td><td>Reference</td></tr><tr><td>ShapeNet</td><td>$\sim  8 \cdot  {10}^{3}, \sim  8 \cdot  {10}^{3}$</td><td>3</td><td>Small Scale</td><td>${\ell }_{1}$</td><td>CFG+15</td></tr><tr><td>Text Embeddings</td><td>${2.5} \cdot  {10}^{3},{1.8} \cdot  {10}^{3}$</td><td>300</td><td>Small Scale</td><td>${\ell }_{1}$</td><td>KSKW15</td></tr><tr><td>Gaussian Points</td><td>$5 \cdot  {10}^{4},5 \cdot  {10}^{4}$</td><td>2</td><td>Outliers</td><td>${\ell }_{1}$</td><td>-</td></tr><tr><td>DEEP1B</td><td>${10}^{4},{10}^{9}$</td><td>96</td><td>Large Scale</td><td>${\ell }_{2}$</td><td>BL16</td></tr><tr><td>Microsoft-Turing</td><td>${10}^{5},{10}^{9}$</td><td>100</td><td>Large Scale</td><td>${\ell }_{2}$</td><td>$\left\lbrack  \mathrm{{SW}{A}^{ + }{22}}\right\rbrack$</td></tr></table>

<table><tbody><tr><td>数据集</td><td>$\left| A\right| ,\left| B\right|$</td><td>$d$</td><td>实验</td><td>$\mathbf{{Metric}}$</td><td>参考文献</td></tr><tr><td>ShapeNet(形状网络)</td><td>$\sim  8 \cdot  {10}^{3}, \sim  8 \cdot  {10}^{3}$</td><td>3</td><td>小规模</td><td>${\ell }_{1}$</td><td>CFG+15(文献编号)</td></tr><tr><td>文本嵌入</td><td>${2.5} \cdot  {10}^{3},{1.8} \cdot  {10}^{3}$</td><td>300</td><td>小规模</td><td>${\ell }_{1}$</td><td>KSKW15(文献编号)</td></tr><tr><td>高斯点</td><td>$5 \cdot  {10}^{4},5 \cdot  {10}^{4}$</td><td>2</td><td>离群值</td><td>${\ell }_{1}$</td><td>-</td></tr><tr><td>DEEP1B(十亿级深度数据集)</td><td>${10}^{4},{10}^{9}$</td><td>96</td><td>大规模</td><td>${\ell }_{2}$</td><td>BL16(文献编号)</td></tr><tr><td>微软图灵</td><td>${10}^{5},{10}^{9}$</td><td>100</td><td>大规模</td><td>${\ell }_{2}$</td><td>$\left\lbrack  \mathrm{{SW}{A}^{ + }{22}}\right\rbrack$</td></tr></tbody></table>

Table 1: Summary of our datasets. For ShapeNet,the value of $\left| A\right|$ and $\left| B\right|$ is averaged across different point clouds in the dataset.

表1：我们的数据集概览。对于ShapeNet数据集，$\left| A\right|$和$\left| B\right|$的值是该数据集中不同点云的平均值。

<!-- Media -->

### 3.1 Experimental Setup

### 3.1 实验设置

We use three different experimental setups, small scale, outlier, and large scale. They are designed to 'stress test' our algorithm, and relevant baselines, under vastly different parameter regimes. The datasets we use are summarized in Table 1. For all experiments, we introduce uniform sampling as a competitive baseline for estimating the Chamfer distance, as well as (accelerated) brute force computation. All results are averaged across ${20} +$ trials and 1 standard deviation error bars are shown when relevant.

我们采用三种实验设置：小规模、异常值和大规模场景。这些设计旨在极端参数条件下对我们的算法及相关基线方法进行"压力测试"。所用数据集详见表1。所有实验均引入均匀采样作为估算倒角距离（Chamfer distance）的竞争基线，同时采用（加速版）暴力计算法。所有结果均取${20} +$次试验平均值，相关数据附有1个标准差误差线。

Small Scale These experiments are motivated from common use cases of Chamfer distance in the computer vision and NLP domains. In our small scale experiments, we use two different datasets: (a) the ShapeNet dataset,a collection of point clouds of objects in three dimensions $\left\lbrack  {{\mathrm{{CFG}}}^{ + }{15}}\right\rbrack$ . ShapeNet is a common benchmark dataset frequently used in computer graphics, computer vision, robotics and Chamfer distance is a widely used measure of similarity between different ShapeNet point clouds $\left\lbrack  {{\mathrm{{CFG}}}^{ + }{15}}\right\rbrack$ . (b) We create point clouds of words from text documents from [KSKW15]. Each point represents a word embedding obtained from the word-to-vec model of $\left\lbrack  {{\mathrm{{MSC}}}^{ + }{13}}\right\rbrack$ in ${\mathbb{R}}^{300}$ applied to the Federalist Papers corpus. As mentioned earlier, a popular relaxation of the common Earth Mover Distance is exactly the (weighted) version of the Chamfer distance [KSKW15, AM19].

小规模实验 这些实验源于计算机视觉和自然语言处理领域使用倒角距离的典型场景。我们采用两个数据集：(a) ShapeNet数据集——包含三维物体点云集合$\left\lbrack  {{\mathrm{{CFG}}}^{ + }{15}}\right\rbrack$。该数据集是计算机图形学、计算机视觉和机器人学领域的常用基准，倒角距离则是衡量不同ShapeNet点云间相似度的通用指标$\left\lbrack  {{\mathrm{{CFG}}}^{ + }{15}}\right\rbrack$。(b) 基于[KSKW15]文献文本生成的词向量点云，每个点代表应用word2vec模型$\left\lbrack  {{\mathrm{{MSC}}}^{ + }{13}}\right\rbrack$处理《联邦党人文集》语料库得到的词嵌入向量。如前言所述，对经典推土机距离（Earth Mover Distance）的常用松弛方法正是倒角距离的加权版本[KSKW15, AM19]。

Since ShapenNet is in three dimensions, we implement nearest neighbor queries using KD-trees to accelerate the brute force baseline as KD-trees can perform exact nearest neighbor search quickly in small dimensions. However, they have runtime exponential in dimension meaning they cannot be used for the text embedding dataset, for which we use a standard naive brute force computation. For both these datasets, we implement our algorithms using Python 3.9.7 on an M1 MacbookPro with 32GB of RAM. We also use an efficient implementation of KD trees in Python and use Numpy and Numba whenever relevant. Since the point clouds in the dataset have approximately the same $n$ value, we compute the symmetric version $\mathrm{{CH}}\left( {A,B}\right)  + \mathrm{{CH}}\left( {B,A}\right)$ . For these experiments,we use the ${\ell }_{1}$ distance function.

由于ShapeNet是三维数据，我们采用KD树加速暴力基线法的最近邻查询——KD树能在低维空间快速完成精确最近邻搜索。但其时间复杂度随维度指数增长，故不适用于文本嵌入数据集（该场景使用标准暴力计算法）。所有算法均在配备32GB内存的M1 MacbookPro上通过Python 3.9.7实现，使用Python高效KD树实现库，适时调用Numpy和Numba加速。鉴于数据集中点云的$n$值近似相等，我们计算对称版本$\mathrm{{CH}}\left( {A,B}\right)  + \mathrm{{CH}}\left( {B,A}\right)$。本实验采用${\ell }_{1}$距离函数。

Outliers This experiment is meant to showcase the robustness of our algorithm. We consider two point clouds, $A$ and $B$ ,each sampled from Gaussian points in ${\mathbb{R}}^{100}$ with identity covariance. Furthermore,we add an "outlier" point to $A$ equal to ${0.5n} \cdot  \mathbf{1}$ ,where1is the all ones vector.

异常值实验 本实验用于验证算法鲁棒性。我们构建两个点云$A$和$B$，均采样自${\mathbb{R}}^{100}$维协方差矩阵为单位阵的高斯分布，并向$A$添加一个等于${0.5n} \cdot  \mathbf{1}$的"异常点"（其中1表示全1向量）。

This example models scenarios where the distances from points in $A$ to their nearest neighbors in $B$ vary significantly, and thus uniform sampling might not accurately account for all distances, missing a small fraction of large ones.

该示例模拟了$A$中各点到$B$中最近邻点距离差异显著的情况，此时均匀采样可能因遗漏少量大距离值而导致估算失准。

Large Scale The purpose of these experiments is to demonstrate that our method scales to datasets with billions of points in hundreds of dimensions. We use two challenging approximate nearest neighbor search datasets: DEEP1B [BL16] and Microsoft Turing-ANNS [SWA ${}^{ + }{22}$ ]. For these datasets,the set $A$ is the query data associated with the datasets. Due to the asymmetric sizes,we compute $\mathrm{{CH}}\left( {A,B}\right)$ . These datasets are normalized to have unit norm and we consider the ${\ell }_{2}$ distance function.

大规模实验 本实验旨在证明我们的方法可扩展至数百维、数十亿规模的数据集。我们使用两个高难度近似最近邻搜索数据集：DEEP1B [BL16]和微软图灵ANNS[SWA${}^{ + }{22}$]。其中$A$为数据集关联的查询数据。由于数据规模不对称，我们计算$\mathrm{{CH}}\left( {A,B}\right)$。这些数据集经归一化处理，采用${\ell }_{2}$距离函数。

These datasets are too large to handle using the prior configurations. Thus, we use a proprietary in-memory parallel implementation of the SimHash algorithm,which is an ${\ell }_{2}$ LSH family for normalized vectors according to Definition A. 1 [Cha02], on a shared virtual compute cluster with 2x64 core AMD Epyc 7763 CPUs (Zen3) with 2.45Ghz - 3.5GHz clock frequency, 2TB DDR4 RAM and 256 MB L3 cache. We also utilize parallization on the same compute cluster for naive brute force search.

这些数据集超出前述配置的处理能力，因此我们在配备2×64核AMD Epyc 7763处理器(主频2.45Ghz-3.5GHz，2TB DDR4内存，256MB L3缓存)的虚拟计算集群上，使用专有内存并行SimHash算法实现——根据定义A.1[Cha02]，该算法是针对归一化向量的${\ell }_{2}$局部敏感哈希族(LSH family)。暴力搜索同样在该集群上并行化实现。

### 3.2 Results

### 3.2 实验结果

Small Scale First we discuss configuring parameters. Recall that in our theoretical results, we use $\mathcal{O}\left( {\log n}\right)$ different scales of the LSH family in CrudeNN. CrudeNN then computes (over) estimates of the nearest neighbor distance from points in $A$ to $B$ (in near linear time) which is then used for importance sampling by Chamfer-Estimate. Concretely for the ${\ell }_{1}$ case,this the LSH family corresponds to imposing $\mathcal{O}\left( {\log n}\right)$ grids with progressively smaller side lengths. In our experiments, we treat the number of levels of grids to use as a tuneable parameter in our implementation and find that a very small number suffices for high quality results in the importance sampling phase.

小规模场景 首先讨论参数配置。理论结果表明，我们在CrudeNN中使用了$\mathcal{O}\left( {\log n}\right)$种不同尺度的LSH族。该算法以近线性时间计算从$A$到$B$的最近邻距离（过估计值），随后用于Chamfer-Estimate的重要性采样。具体到${\ell }_{1}$案例，LSH族对应逐步缩小边长的$\mathcal{O}\left( {\log n}\right)$层网格。实验中将网格层数作为可调参数，发现极少数层级即可在重要性采样阶段获得优质结果。

Figure 6(b) shows that only using 3 grid levels is sufficient for the crude estimates ${\mathbf{D}}_{a}$ to be within a factor of 2 away from the true nearest neighbor values for the ShapeNet dataset, averaged across different point clouds in the dataset. Thus for the rest of the Small Scale experiments, we fix the number of grid levels to be 3 .

图6(b)显示，对于ShapeNet数据集，仅需3层网格就可使粗略估计值${\mathbf{D}}_{a}$与真实最近邻值的误差保持在2倍以内（数据集内不同点云的平均值）。因此在后续小规模实验中，固定采用3层网格。

Figure 3 (a) shows the sample complexity vs accuracy trade offs of our algorithm, which uses importance sampling, compared to uniform sampling. Accuracy is measured by the relative error to the true value. We see that our algorithm possesses a better trade off as we obtain the same relative error using only 10 samples as uniform sampling does using ${50} +$ samples,resulting in at least a $5\mathbf{x}$ improvement in sample complexity. For the text embedding dataset, the performance gap between our importance sampling algorithm and uniform sampling grows even wider, as demonstrated by Figure 3(b),leading to $> \mathbf{{10x}}$ improvement in sample complexity.

图3(a)对比了重要性采样与均匀采样的样本复杂度-准确率权衡曲线，准确率以真实值的相对误差衡量。可见本算法仅需10个样本即可达到均匀采样${50} +$个样本的同等误差水平，样本复杂度至少提升$5\mathbf{x}$倍。对于文本嵌入数据集（图3(b)），重要性采样优势更显著，样本复杂度提升达$> \mathbf{{10x}}$倍。

In terms of runtimes, we expect the brute force search to be much slower than either importance sampling and uniform sampling. Furthermore, our algorithm has the overhead of first estimating the values ${\mathbf{D}}_{a}$ for $a \in  A$ using an LSH family,which uniform sampling does not. However,this is compensated by the fact that our algorithm requires much fewer samples to get accurate estimates. Indeed, Figure 4 (a) shows the average time of 100 Chamfer distance computations between randomly chosen pairs of point clouds in the ShapeNet dataset. We set the number of samples for uniform sampling and importance sampling (our algorithm) such that they both output estimates with (close to) 2% relative error. Note that our runtime includes the time to build our LSH data structures. This means we used 100 samples for importance sampling and 500 for uniform. The brute force KD Tree algorithm (which reports exact answers) is approximately $5\mathrm{x}$ slower than our algorithm. At the same time,our algorithm is ${50}\%$ faster than uniform sampling. For the Federalist Papers dataset (Figure 4 (b)), our algorithm only required 20 samples to get a 2% relative error approximation, whereas uniform sampling required at least 450 samples. As a result,our algorithm achieved $\mathbf{{2x}}$ speedup compared to uniform sampling.

运行时间方面，暴力搜索明显慢于两种采样方法。虽然本算法需额外通过LSH族估算$a \in  A$的${\mathbf{D}}_{a}$值（均匀采样无此开销），但所需样本量大幅减少。图4(a)展示ShapeNet数据集中随机点云对进行100次Chamfer距离计算的平均耗时：设置两种采样方法的样本量使其输出误差接近2%（含构建LSH数据结构时间），重要性采样用100样本，均匀采样需500样本。暴力KD树算法（精确解）比本算法慢约$5\mathrm{x}$倍，同时本算法比均匀采样快${50}\%$倍。联邦党人文集数据集（图4(b)）中，本算法仅需20样本即达2%误差，而均匀采样需450样本以上，最终实现$\mathbf{{2x}}$倍加速。

<!-- Media -->

<!-- figureText: ShapeNet Dataset Federalist Papers Dataset Synthetic Dataset Importance Sampling Uniform Sampling 0.8 Relative Error 0.6 Importance Sampling Uniform Sampling 0.4 0.2 100 #Samples #Samples (c) Gaussian Points Turing Dataset 0.8 Importance Sampling 0.7 Uniform Sampling Relative Error 0.5 0.2 0.1 0.0 20 60 #Samples (e) Turing 0.8 Importance Sampling 0.8 Uniform Sampling Relative Error 0.6 Relative Error 0.6 0.4 0.4 0.2 0.2 0.0 #Samples (a) ShapeNet (b) Federalist Papers DEEP Dataset Importance Sampling Uniform Sampling Relative Error 0.6 0.4 0.0 20 60 100 (d) DEEP -->

<img src="https://cdn.noedgeai.com/019625a9-7e32-71c3-9be2-e6920620ac11_6.jpg?x=306&y=229&w=1193&h=768&r=0"/>

Figure 3: Sample complexity vs relative error curves.

图3：样本复杂度与相对误差关系曲线

<!-- figureText: ShapeNet Dataset Federalist Papers Dataset 1 Chamfer Computation (ms) 800 600 400 200 KD Tree Uniform Sampling (b) Federalist Papers 6000 100 Chamfer Computations (m 5000 4000 3000 2000 1000 KD Tree Uniform Sampling (a) ShapeNet -->

<img src="https://cdn.noedgeai.com/019625a9-7e32-71c3-9be2-e6920620ac11_6.jpg?x=437&y=1104&w=913&h=367&r=0"/>

Figure 4: Runtime experiments. We set the number of samples for uniform and importance sampling such that the relative errors of their respective approximations are similar.

图4：运行时间实验。调整两种采样方法的样本量使其近似误差相当

<!-- Media -->

Outliers We performed similar experiments as above. Figure 3 (c) shows the sample complexity vs accuracy trade off curves of our algorithm and uniform sampling. Uniform sampling has a very large error compared to our algorithm, as expected. While the relative error of our algorithm decreases smoothly as the sample size grows, uniform sampling has the same high relative error. In fact, the relative error will stay high until the outlier is sampled,which typically requires $\Omega \left( n\right)$ samples.

离群值分析 我们进行了类似实验。图3(c)显示两种算法的样本复杂度-准确率权衡曲线。如预期，均匀采样误差远高于本算法。随着样本量增加，本算法误差平稳下降，而均匀采样保持高误差——其误差仅当采样到离群点（通常需$\Omega \left( n\right)$样本）后才会降低。

<!-- Media -->

<!-- figureText: DEEP Dataset Turing Dataset 64x1M 0.10 16x500k 0.09 16x250k $8 \times  {100}\mathrm{k}$ Relative Error 0.08 Uniform Sampling 0.07 0.06 0.05 0.04 0.03 20 40 50 60 100 #Samples (b) Turing 32x200k 0.10 Uniform Sampling ${16} \times  {100}\mathrm{k}$ 8x100k Relative Error 16x250k ${16} \times  {500}\mathrm{k}$ 64x500k ${64} \times  1\mathrm{M}$ 0.06 0.04 20 30 40 70 80 90 100 #Samples (a) DEEP -->

<img src="https://cdn.noedgeai.com/019625a9-7e32-71c3-9be2-e6920620ac11_7.jpg?x=436&y=231&w=916&h=391&r=0"/>

Figure 5: The figures show sample complexity vs relative error curves as we vary the number of LSH data structures and window sizes. Each curve maps $k \times  W$ where $k$ is the number of LSH data structures we use to repeatedly hash points in $B$ and $W$ is the window size,the number of points retrieved from $B$ that hash closest to any given $a$ at the smallest possible distance scales.

图5：展示改变LSH数据结构数量和窗口大小时的样本复杂度-相对误差曲线。每条曲线对应$k \times  W$，其中$k$表示用于重复哈希$B$中点的LSH数据结构数量，$W$表示窗口大小（即在最小距离尺度下从$B$中检索出的、与给定$a$哈希值最接近的点数）。

<!-- Media -->

Large Scale We consider two modifications to our algorithm to optimize the performance of CrudeNN on the two challenging datasets that we are using; namely, note that both datasets are standard for benchmarking billion-scale nearest neighbor search. First, in the CrudeNN algorithm, when computing ${\mathbf{D}}_{a}$ for $a \in  A$ ,we search through the hash buckets ${h}_{1}\left( a\right) ,{h}_{2}\left( a\right) ,\ldots$ containing $a$ in increasing order of $i$ (i.e.,smallest scale first),and retrieve the first $W$ (window size) distinct points in $B$ from these buckets. Then,the whole process is repeated $k$ times,with $k$ independent LSH data structures,and ${\mathbf{D}}_{a}$ is set to be the distance from $a$ to the closest among all ${Wk}$ retrieved points.

大规模优化 我们对算法进行了两项改进以提升CrudeNN在两个具有挑战性的数据集上的性能；需注意这两个数据集都是用于十亿级最近邻搜索基准测试的标准数据集。首先，在CrudeNN算法中，当计算${\mathbf{D}}_{a}$时，我们按$i$的升序（即最小尺度优先）检索包含$a$的哈希桶${h}_{1}\left( a\right) ,{h}_{2}\left( a\right) ,\ldots$，并从这些桶中获取前$W$（窗口大小）个不同的$B$点。该过程重复$k$次（使用$k$个独立的LSH数据结构），最终将${\mathbf{D}}_{a}$设定为$a$到所有${Wk}$个检索点中最近点的距离。

Note that previously,for our smaller datasets,we set ${\mathbf{D}}_{a}$ to be the distance to the first point in $B$ colliding with $a$ ,and repeated the LSH data structure once,corresponding to $W = k = 1$ . In our figures,we refer to these parameter choices as $k \times  W$ and test our algorithm across several choices.

需说明的是，此前针对较小数据集时，我们将${\mathbf{D}}_{a}$设置为$B$中首个与$a$碰撞点的距离，并仅重复一次LSH数据结构（对应$W = k = 1$）。在图表中，我们将这些参数组合标注为$k \times  W$，并测试了算法在不同参数下的表现。

For the DEEP and Turing datasets, Figures 3 (d) and 3 (e) show the sample complexity vs relative error trade-offs for the best parameter choice (both ${64} \times  {10}^{6}$ ) compared to uniform sampling. Qualitatively, we observe the same behavior as before: importance sampling requires fewer samples to obtain the same accuracy as uniform sampling. Regarding the other parameter choices, we see that, as expected,if we decrease $k$ (the number of LSH data structures),or if we decrease $W$ (the window size),the quality of the approximations ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ decreases and importance sampling has worse sample complexity trade-offs. Nevertheless, for all parameter choices, we see that we obtain superior sample complexity trade-offs compared to uniform sampling, as shown in Figure 5. A difference between these parameter choices are the runtimes required to construct the approximations ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ . For example for the DEEP dataset, the naive brute force approach (which is also optimized using parallelization) took approximately ${1.3} \cdot  {10}^{4}$ seconds,whereas the most expensive parameter choice of ${64} \times  {10}^{6}$ took approximately half the time at ${6.4} \times  {10}^{3}$ and the cheapest parameter choice of $8 \times  {10}^{5}$ took 225 seconds,leading to a 2x-50x factor speedup. The runtime differences between brute force and our algorithm were qualitative similar for the Turing dataset.

对于DEEP和图灵数据集，图3(d)和3(e)展示了最佳参数选择（均为${64} \times  {10}^{6}$）与均匀采样在样本复杂度与相对误差权衡上的对比。定性来看，我们观察到了与此前相同的行为：重要性采样只需更少样本即可达到与均匀采样相同的精度。其他参数组合中，如预期那样，当减少$k$（LSH数据结构数量）或缩小$W$（窗口大小）时，近似结果${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$的质量会下降，重要性采样的样本复杂度权衡表现变差。但如图5所示，所有参数组合均优于均匀采样。不同参数选择的主要差异在于构建近似${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$所需的运行时——例如DEEP数据集中，原始暴力搜索法（经并行优化）耗时约${1.3} \cdot  {10}^{4}$秒，而最耗时的${64} \times  {10}^{6}$参数组合耗时减半（${6.4} \times  {10}^{3}$秒），最经济的$8 \times  {10}^{5}$组合仅需225秒，实现了2-50倍的加速。图灵数据集的运行时差异趋势与之相似。

Similar to the small scale dataset, our method also outperforms uniform sampling in terms of runtime if we require they both output high quality approximations. If we measure the runtime to get a 1% relative error,the ${16} \times  2 \cdot  {10}^{5}$ version of our algorithm for the DEEP dataset requires approximately 980 samples with total runtime approximately 1785 seconds, whereas uniform sampling requires $> {1750}$ samples and runtime $> {2200}$ seconds,which is $> {23}\%$ slower. The gap in runtime increases if we desire approximations with even smaller relative error, as the overhead of obtaining the approximations ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ becomes increasingly overwhelmed by the time needed to compute the exact answer for our samples.

与小规模数据集类似，当要求输出高质量近似时，我们的方法在运行时方面同样优于均匀采样。若要获得1%相对误差，DEEP数据集上${16} \times  2 \cdot  {10}^{5}$版本的算法需约980个样本（总运行时1785秒），而均匀采样需要$> {1750}$个样本和$> {2200}$秒运行时，速度慢了$> {23}\%$倍。当追求更小相对误差时，运行时差距会进一步扩大，因为构建近似${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$的开销逐渐被样本精确计算所需的时间所掩盖。

Additional Experimental Results We perform additional experiments to show the utility of our approximation algorithm for the Chamfer distance for downstream tasks. For the ShapeNet dataset, we show we can efficiently recover the true exact nearest neighbor of a fixed point cloud $A$ in Chamfer distance among a large collect of different point clouds. In other words, it is beneficial for finding the ’nearest neighboring point cloud’. Recall the ShapeNet dataset,contains approximately $5 \cdot  {10}^{4}$ different point clouds. We consider the following simple (and standard) two step pipeline: (1) use our algorithm to compute an approximation of the Chamfer distance from $A$ to every other point cloud $B$ in our dataset. More specifically,compute an approximation to $\mathrm{{CH}}\left( {A,B}\right)  + \mathrm{{CH}}\left( {B,A}\right)$ for all $B$ using 50 samples and the same parameter configurations as the small scale experiments. Then filter the dataset of points clouds and prune down to the top $k$ closest point cloud candidates according to our approximate distances. (2) Find the closest point cloud in the top $k$ candidates via exact computation.

补充实验结果 我们通过额外实验验证所提出的倒角距离近似算法在下游任务中的实用性。在ShapeNet数据集上，我们证明能够高效地从大量点云集合中，根据倒角距离恢复固定点云$A$的精确最近邻。换言之，该方法对寻找"最近邻点云"具有显著优势。需知ShapeNet数据集包含约$5 \cdot  {10}^{4}$个不同点云。我们采用以下标准两阶段流程：(1)使用本算法计算$A$与数据集中其他所有点云$B$的倒角距离近似值，具体而言，采用50个样本及与小规模实验相同的参数配置，计算所有$B$对应$\mathrm{{CH}}\left( {A,B}\right)  + \mathrm{{CH}}\left( {B,A}\right)$的近似值，随后根据近似距离筛选出前$k$个候选点云。(2)通过精确计算从前$k$个候选中确定最近点云。

We measure the accuracy of this via the standard recall $@k$ measure,which computes the fraction of times the exact nearest neighbor $B$ of $A$ ,averaged over multiple $A$ ’s,is within the top $k$ choices. Figure 6 (a) shows that the true exact nearest neighbor of $A$ ,that is the point cloud $B$ which minimizes $\mathrm{{CH}}\left( {A,B}\right)  + \mathrm{{CH}}\left( {B,A}\right)$ among our collection of multiple point clouds,is within the top 30 candidates $> {98}\%$ ,time (averaged over multiple different choices of $A$ ). This represents a more than $\mathbf{{1000}}\mathbf{x}$ reduction in the number of point clouds we do exact computation over compared to the naive brute force method, demonstrating the utility of our algorithm for downstream tasks.

我们采用标准召回率$@k$衡量准确性，该指标计算在多次$A$实验中，精确最近邻$B$位于前$k$候选中的概率均值。图6(a)显示，在多个点云集合中使$\mathrm{{CH}}\left( {A,B}\right)  + \mathrm{{CH}}\left( {B,A}\right)$最小化的真实最近邻$B$，有$> {98}\%$的概率位于前30候选之列（基于不同$A$选择的多次实验均值）。相较于暴力穷举法，该方法使需要精确计算的点云数量减少超过$\mathbf{{1000}}\mathbf{x}$倍，充分证明了算法在下游任务中的实用价值。

<!-- Media -->

<!-- figureText: 1.0 ShapeNet Dataset 7 6 Ratio 2 10 #Levels (b) Quality of approximations ${\mathbf{D}}_{a}$ vs the number of levels of LSH data structure 0.8 0.2 0.0 10 20 30 40 50 (a) ShapeNet NNS pipeline experiments -->

<img src="https://cdn.noedgeai.com/019625a9-7e32-71c3-9be2-e6920620ac11_8.jpg?x=434&y=687&w=925&h=429&r=0"/>

Figure 6: Additional figures for the ShapeNet dataset.

图6：ShapeNet数据集的补充图示

<!-- Media -->

## 4 Lower Bound for Reporting the Alignment

## 4 对齐报告的下界证明

We presented an algorithm that,in time $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ ,produces a $\left( {1 + \varepsilon }\right)$ -approximation to $\mathrm{{CH}}\left( {A,B}\right)$ . It is natural to ask whether it is also possible to report a mapping $g : A \rightarrow  B$ whose $\operatorname{cost}\mathop{\sum }\limits_{{a \in  A}}\parallel a - g\left( a\right) {\parallel }_{1}$ is within a factor of $1 + \varepsilon$ from $\operatorname{CH}\left( {A,B}\right)$ . (Our algorithm uses on random sampling and thusdoes not give such a mapping). This section shows that, under a popular complexity-theoretic conjecture called the Hitting Set Conjecture [Wil18], such an algorithm does not exists. For simplicity,we focus on the case when the underlying metric ${d}_{X}$ is induced by the Manhattan distance, i.e., ${d}_{X}\left( {a,b}\right)  = \parallel a - b{\parallel }_{1}$ . The argument is similar for the Euclidean distance,Euclidean distance squared, etc. To state our result formally, we first define the Hitting Set (HS) problem.

我们提出的算法可在$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$时间内生成$\mathrm{{CH}}\left( {A,B}\right)$的$\left( {1 + \varepsilon }\right)$近似解。自然引出的问题是：能否同时输出一个映射$g : A \rightarrow  B$，其$\operatorname{cost}\mathop{\sum }\limits_{{a \in  A}}\parallel a - g\left( a\right) {\parallel }_{1}$与$\operatorname{CH}\left( {A,B}\right)$的偏差在$1 + \varepsilon$倍以内？（本算法采用随机采样策略故不提供此类映射）。本节将证明，在计算复杂性理论中流行的"击中集猜想"[Wil18]成立的前提下，此类算法不存在。为简化论述，我们重点讨论曼哈顿距离${d}_{X}$诱导的度量空间，即${d}_{X}\left( {a,b}\right)  = \parallel a - b{\parallel }_{1}$。该论证同样适用于欧氏距离、平方欧氏距离等情形。为严谨表述结论，我们首先定义击中集(HS)问题。

Definition 4.1 (Hitting Set (HS) problem). The input to the problem consists of two sets of vectors $A,B \subseteq  \{ 0,1{\} }^{d}$ ,and the goal is to determine whether there exists some $a \in  A$ such that $a \cdot  b \neq  0$ for every $b \in  B$ . If such an $a \in  A$ exists,we say that $a$ hits $B$ .

定义4.1（击中集问题）。问题输入包含两个向量集合$A,B \subseteq  \{ 0,1{\} }^{d}$，目标是判定是否存在某个$a \in  A$使得对于每个$b \in  B$都满足$a \cdot  b \neq  0$。若存在这样的$a \in  A$，则称$a$击中了$B$。

It is easy to see that the Hitting Set problem can be solved in time $\mathcal{O}\left( {{n}^{2}d}\right)$ . The Hitting Set Conjecture [Wil18] postulates that this running time is close to the optimal. Specifically:

显然，击中集问题可在$\mathcal{O}\left( {{n}^{2}d}\right)$时间内求解。击中集猜想[Wil18]认为该时间复杂度已接近最优，具体表现为：

Conjecture 4.2. Suppose $d = \Theta \left( {{\log }^{2}n}\right)$ . Then for every constant $\delta  > 0$ ,no randomized algorithm can solve the Hitting Set problem in $\mathcal{O}\left( {n}^{2 - \delta }\right)$ time.

猜想4.2：假设$d = \Theta \left( {{\log }^{2}n}\right)$成立，则对于任意常数$\delta  > 0$，不存在随机算法能在$\mathcal{O}\left( {n}^{2 - \delta }\right)$时间内求解命中集问题。

Our result can be now phrased as follows.

我们的研究结果可表述如下。

Theorem 4.3 (Hardness for reporting a mapping). Let $T\left( {N,D,\varepsilon }\right)$ be the running time of an algorithm ${ALG}$ that,given sets of $A$ ", $B$ " $\subset  \{ 0,1{\} }^{D}$ of sizes at most $N$ ,reports a mapping $g : A$ " $\rightarrow  B$ " with cost $\left( {1 + \varepsilon }\right) \mathrm{{CH}}\left( {A\text{",}B\text{"),for}D = \Theta \left( {{\log }^{2}N}\right) \text{and}\varepsilon  = \frac{\Theta \left( 1\right) }{D}}\right)$ . Assuming the Hitting Set Conjecture, we have that $T\left( {N,D,\varepsilon }\right)$ is at least $\Omega \left( {N}^{2 - \delta }\right)$ for any constant $\delta  > 0$ .

定理4.3（映射报告的计算复杂度下界）：设$T\left( {N,D,\varepsilon }\right)$为算法${ALG}$的运行时间，该算法在输入规模不超过$N$的集合$A$与$B$$\subset  \{ 0,1{\} }^{D}$时，能报告成本为$\left( {1 + \varepsilon }\right) \mathrm{{CH}}\left( {A\text{",}B\text{"),for}D = \Theta \left( {{\log }^{2}N}\right) \text{and}\varepsilon  = \frac{\Theta \left( 1\right) }{D}}\right)$的映射$g : A$$\rightarrow  B$。基于命中集猜想，对于任意常数$\delta  > 0$，$T\left( {N,D,\varepsilon }\right)$至少为$\Omega \left( {N}^{2 - \delta }\right)$。

## 5 Conclusion

## 5 结论

We present an efficient approximation algorithm for estimating the Chamfer distance up to a $1 + \varepsilon$ factor in time $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ . The result is complemented with a conditional lower bound which shows that reporting a Chamfer distance mapping of similar quality requires nearly quadratic time Our algorithm is easy to implement in practice and compares favorably to brute force computation and uniform sampling. We envision our main tools of obtaining fast estimates of coarse nearest neighbor distances combined with importance sampling can have additional applications in the analysis of high-dimensional, large scale data.

我们提出了一种高效近似算法，可在$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$时间内以$1 + \varepsilon$因子估计Chamfer距离。该结果与条件性下界相辅相成，表明报告具有相似质量的Chamfer距离映射需要近二次方时间。本算法易于实践实现，在性能上优于暴力计算与均匀采样。我们预见，结合重要性采样的快速粗粒度最近邻距离估计方法，将有助于高维海量数据分析的其他应用。

## References

## 参考文献

[AM19] Kubilay Atasu and Thomas Mittelholzer. Linear-complexity data-parallel earth mover's

[AM19] Kubilay Atasu与Thomas Mittelholzer。线性复杂度数据并行地球移动器

distance approximations. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 364-373. PMLR, 09-15 Jun 2019.

[AMN ${}^{ + }{98}$ ] Sunil Arya,David M Mount,Nathan S Netanyahu,Ruth Silverman,and Angela Y Wu. An optimal algorithm for approximate nearest neighbor searching fixed dimensions. Journal of the ACM (JACM), 45(6):891-923, 1998.

[AR15] Alexandr Andoni and Ilya Razenshteyn. Optimal data-dependent hashing for approximate near neighbors. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing, pages 793-801, 2015.

[AS03] Vassilis Athitsos and Stan Sclaroff. Estimating 3d hand pose from a cluttered image. In 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings., volume 2, pages II-432. IEEE, 2003.

[BL16] Artem Babenko and Victor Lempitsky. Efficient indexing of billion-scale datasets of deep descriptors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2055-2063, 2016.

$\left\lbrack  {{\mathrm{{CFG}}}^{ + }{15}}\right\rbrack$ Angel X Chang,Thomas Funkhouser,Leonidas Guibas,Pat Hanrahan,Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[Cha02] Moses S Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing, pages 380-388, 2002.

[FSG17] Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 605-613, 2017.

[Ind06] Piotr Indyk. Stable distributions, pseudorandom generators, embeddings, and data stream computation. Journal of the ACM (JACM), 53(3):307-323, 2006.

[JSQJ18] Li Jiang, Shaoshuai Shi, Xiaojuan Qi, and Jiaya Jia. Gal: Geometric adversarial loss for single-view 3d-object reconstruction. In Proceedings of the European conference on computer vision (ECCV), pages 802-816, 2018.

[KSKW15] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From word embeddings to document distances. In International conference on machine learning, pages 957-966. PMLR, 2015.

[LSS ${}^{ + }$ 19] Chun-Liang Li,Tomas Simon,Jason Saragih,Barnabás Póczos,and Yaser Sheikh. Lbs autoencoder: Self-supervised fitting of articulated meshes to point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11967-11976, 2019.

[Mat13] Jiri Matousek. Lectures on discrete geometry, volume 212. Springer Science & Business Media, 2013.

$\left\lbrack  {{\mathrm{{MSC}}}^{ + }{13}}\right\rbrack$ Tomas Mikolov,Ilya Sutskever,Kai Chen,Greg S Corrado,and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.

[pda23] Pdal: Chamfer. https://pdal.io/en/2.4.3/apps/chamfer.html, 2023. Accessed: 2023-05-12.

[pyt23] Pytorch3d: Loss functions. https://pytorch3d.readthedocs.io/en/latest/ modules/loss.html, 2023. Accessed: 2023-05-12.

[Roh19] Dhruv Rohatgi. Conditional hardness of earth mover distance. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (APPROX/RANDOM 2019). Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.

[SMFW04] Erik B Sudderth, Michael I Mandel, William T Freeman, and Alan S Willsky. Visual hand tracking using nonparametric belief propagation. In 2004 Conference on Computer Vision and Pattern Recognition Workshop, pages 189-189. IEEE, 2004.

[SWA ${}^{ + }{22}$ ] Harsha Vardhan Simhadri,George Williams,Martin Aumüller,Matthijs Douze,Artem Babenko, Dmitry Baranchuk, Qi Chen, Lucas Hosseini, Ravishankar Krishnaswamny, Gopal Srinivasa, et al. Results of the neurips'21 challenge on billion-scale approximate nearest neighbor search. In NeurIPS 2021 Competitions and Demonstrations Track, pages 177-189. PMLR, 2022.

[ten23] Tensorflow graphics: Chamfer distance. https://www.tensorflow.org/graphics/ api_docs/python/tfg/nn/loss/chamfer_distance/evaluate, 2023. Accessed: 2023-05-12.

[WCL ${}^{ + }$ 19] Ziyu Wan,Dongdong Chen,Yan Li,Xingguang Yan,Junge Zhang,Yizhou Yu,and Jing Liao. Transductive zero-shot learning with visual structure constraint. Advances in neural information processing systems, 32, 2019.

[Wil18] Virginia Vassilevska Williams. On some fine-grained questions in algorithms and complexity. In Proceedings of the international congress of mathematicians: Rio de janeiro 2018, pages 3447-3487. World Scientific, 2018.

## A Deferred Analysis from Section 2

## 附录A 第2节延后分析

Proof of Lemma 2.2. The proof follows from a standard analysis of importance sampling. The fact that our estimator $\mathbf{\eta }$ is unbiased holds from the definition of ${\mathbf{\eta }}_{\ell }$ ,since we are re-weighting samples according to the probability with which they are sampled in $\mathcal{D}$ (in particular,the estimator is unbiased for all distributions $\mathcal{D}$ where ${\mathbf{D}}_{a} > 0$ for all $a \in  A$ ). The bound on the variance is then a simple calculation:

引理2.2证明：该证明遵循重要性采样的标准分析流程。估计量$\mathbf{\eta }$的无偏性源于${\mathbf{\eta }}_{\ell }$的定义——我们根据样本在$\mathcal{D}$中的抽样概率进行重新加权（特别地，该估计量对所有满足${\mathbf{D}}_{a} > 0$（当$a \in  A$时）的分布$\mathcal{D}$均无偏）。方差界的推导则为简单计算：

$$
\operatorname{Var}\left\lbrack  \mathbf{\eta }\right\rbrack   \leq  \frac{1}{T} \cdot  \left( {\left\lbrack  {\mathop{\sum }\limits_{{a \in  A}}\left( \frac{\mathbf{D}}{{\mathbf{D}}_{a}}\right) \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{2}^{2}}\right\rbrack   - \mathrm{{CH}}{\left( A,B\right) }^{2}}\right) 
$$

$$
 \leq  \frac{1}{T} \cdot  \left\lbrack  {\mathop{\sum }\limits_{{a \in  A}}\mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{2} \cdot  \mathbf{D}}\right\rbrack   - \frac{\mathrm{{CH}}{\left( A,B\right) }^{2}}{T}
$$

$$
 \leq  \frac{1}{T} \cdot  \mathrm{{CH}}{\left( A,B\right) }^{2}\left( {\frac{\mathbf{D}}{\mathrm{{CH}}\left( {A,B}\right) } - 1}\right) .
$$

The final probability bound follows from Chebyshev’s inequality.

最终概率界由切比雪夫不等式得出。

Locality Sensitive Hashing at every scale. We now discuss how to find such partitions. For the ${\ell }_{1}$ distance,each partition $i$ is formed by imposing a (randomly shifted) grid of side length ${2}^{i}$ on the dataset. Note that while the grid partitions the entire space ${\mathbb{R}}^{d}$ into infinitely many components, we can efficiently enumerate over the non empty components which actually contain points in our dataset. To this end, we introduce the following definition:

多尺度局部敏感哈希：现讨论如何构建此类划分。对于${\ell }_{1}$距离，每个划分$i$通过施加边长为${2}^{i}$的（随机偏移）网格实现。需注意虽然网格将整个空间${\mathbb{R}}^{d}$划分为无限组件，但可高效枚举实际包含数据点的非空组件。为此引入以下定义：

Definition A. 1 (Hashing at every scale). There exists a fixed constant ${c}_{1} > 0$ and a parameterized family $\mathcal{H}\left( r\right)$ of functions from $X$ to some universe $U$ such that for all $r > 0$ ,and for every $x,y \in  X$

定义A.1（全尺度哈希）：存在固定常数${c}_{1} > 0$与参数化函数族$\mathcal{H}\left( r\right)$（将$X$映射至某宇宙$U$），使得对于所有$r > 0$及每个$x,y \in  X$

1. Close points collide frequently:

1. 邻近点高频碰撞：

$$
\mathop{\Pr }\limits_{{\mathbf{h} \sim  \mathcal{H}\left( r\right) }}\left\lbrack  {\mathbf{h}\left( x\right)  \neq  \mathbf{h}\left( y\right) }\right\rbrack   \leq  \frac{\parallel x - y{\parallel }_{1}}{r},
$$

2. Far points collide infrequently:

2. 远距点低频碰撞：

$$
\mathop{\Pr }\limits_{{\mathbf{h} \sim  \mathcal{H}\left( r\right) }}\left\lbrack  {\mathbf{h}\left( x\right)  = \mathbf{h}\left( y\right) }\right\rbrack   \leq  \exp \left( {-{c}_{1} \cdot  \frac{\parallel x - y{\parallel }_{1}}{r}}\right) .
$$

We are now ready to make this approach concrete via the following lemma:

现通过以下引理具体实现该方法：

Lemma A. 2 (Oversampling with bounded Aspect Ratio). Let $\left( {X,{d}_{X}}\right)$ be a metric space with a locality-sensitive hash family at every scale (see Definition A.1). Consider two subsets $A,B \subset  X$ of size at most $n$ and $\varepsilon  \in  \left( {0,1}\right)$ satisfying

引理A.2（有界纵横比的过采样）：设$\left( {X,{d}_{X}}\right)$为具有全尺度局部敏感哈希族的度量空间（见定义A.1）。考虑两个子集$A,B \subset  X$（规模分别不超过$n$与$\varepsilon  \in  \left( {0,1}\right)$）满足

$$
1 \leq  \mathop{\min }\limits_{\substack{{a \in  A,b \in  B} \\  {a \neq  b} }}{d}_{X}\left( {a,b}\right)  \leq  \mathop{\max }\limits_{{a \in  A,b \in  B}}{d}_{X}\left( {a,b}\right)  \leq  \operatorname{poly}\left( {n/\varepsilon }\right) .
$$

Algorithm $\overline{2}$ CrudeNN(A,B),outputs a list of (random) positive numbers ${\left\{  {D}_{a}\right\}  }_{a \in  A}$ which satisfy the following two guarantees:

算法$\overline{2}$ CrudeNN(A,B)输出一组满足以下两项保证的(随机)正数列表${\left\{  {D}_{a}\right\}  }_{a \in  A}$：

- With probability 1,every $a \in  A$ satisfies ${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$ .

- 在概率1条件下，每个$a \in  A$都满足${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$。

- For every $a \in  A,\mathbf{E}\left\lbrack  {\mathbf{D}}_{a}\right\rbrack   \leq  \mathcal{O}\left( {\log n}\right)  \cdot  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$ .

- 对于每个$a \in  A,\mathbf{E}\left\lbrack  {\mathbf{D}}_{a}\right\rbrack   \leq  \mathcal{O}\left( {\log n}\right)  \cdot  \mathop{\min }\limits_{{b \in  B}}{d}_{X}\left( {a,b}\right)$。

Further,Algorithm 2,runs in time $\mathcal{O}\left( {\operatorname{dn}\log \left( {n/\varepsilon }\right) }\right)$ time,assuming that each function used in the algorithm can be evaluated in $\mathcal{O}\left( d\right)$ time.

此外，假设算法中每个函数可在$\mathcal{O}\left( d\right)$时间内完成计算，则算法2的运行时间为$\mathcal{O}\left( {\operatorname{dn}\log \left( {n/\varepsilon }\right) }\right)$。

Finally, we show that it always suffices to assume bounded aspect ratio:

最后，我们证明始终可以限定纵横比范围：

Lemma A. 3 (Reduction to bounded Aspect Ratio). Given an instance $A,B \subset  {\mathbb{R}}^{d}$ such that $\left| A\right| ,\left| B\right|  \leq  n$ ,and $0 < \varepsilon  < 1$ there exists an algorithm that runs in time $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ and outputs a partition ${A}_{1},{A}_{2},\ldots {A}_{T}$ of $A$ and ${B}_{1},{B}_{2},\ldots {B}_{T}$ of $B$ such that $T = \mathcal{O}\left( n\right)$ and for each

引理A.3(有界纵横比归约)。给定实例$A,B \subset  {\mathbb{R}}^{d}$使得$\left| A\right| ,\left| B\right|  \leq  n$且$0 < \varepsilon  < 1$，存在一个在$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$时间内运行的算法，可输出$A$的划分${A}_{1},{A}_{2},\ldots {A}_{T}$和$B$的划分${B}_{1},{B}_{2},\ldots {B}_{T}$，满足$T = \mathcal{O}\left( n\right)$且对每个

$t \in  \left\lbrack  T\right\rbrack$ ,

$$
1 \leq  \mathop{\min }\limits_{\substack{{a \in  {A}_{t},b \in  {B}_{t}} \\  {a \neq  b} }}\parallel a - b{\parallel }_{1} \leq  \mathop{\max }\limits_{{a \in  {A}_{t},b \in  {B}_{t}}}\parallel a - b{\parallel }_{1} \leq  \operatorname{poly}\left( {n/\varepsilon }\right) .
$$

Further,

此外，

$$
\left( {1 - \varepsilon }\right) \mathrm{{CH}}\left( {A,B}\right)  \leq  \mathop{\sum }\limits_{{t \in  \left\lbrack  T\right\rbrack  }}\mathrm{{CH}}\left( {{A}_{t},{B}_{t}}\right)  \leq  \left( {1 + \varepsilon }\right) \mathrm{{CH}}\left( {A,B}\right) .
$$

We defer the proofs of Lemma A. 2 and Lemma A.3 to sub-sections A.1 and A.3 respectively. We are now ready to complete the proof of Theorem 2.1

我们将引理A.2和引理A.3的证明分别推迟到A.1和A.3子节。现在可以完成定理2.1的证明了

Proof of Theorem 2.1 Observe,by Lemma A.3,we can partition the input into pairs ${\left( {A}_{t},{B}_{t}\right) }_{t \in  \left\lbrack  T\right\rbrack  }$ such that each pair has aspect ratio at most poly $\left( {n/\varepsilon }\right)$ and the $\mathrm{{CH}}\left( {A,B}\right)$ is well-approximated by the direct sum of $\mathrm{{CH}}\left( {{A}_{t},{B}_{t}}\right)$ . Next,repeating the construction from Lemma A.2,and applying Markov’s inequality,we have list ${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$ such that with probability at least ${99}/{100}$ ,for all $a \in  A$ , ${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$ and $\mathbf{D} = \mathop{\sum }\limits_{{a \in  A}}{\mathbf{D}}_{a} \leq  \mathcal{O}\left( {\log \left( n\right) }\right) \mathrm{{CH}}\left( {A,B}\right)$ . Invoking Lemma 2.2 with the aforementioned parameters,and $T = \mathcal{O}\left( {\log \left( n\right) /{\varepsilon }^{2}}\right)$ suffices to obtain an estimator $\eta$ which is a $\left( {1 \pm  \varepsilon }\right)$ relative-error approximation to $\operatorname{CH}\left( {A,B}\right)$ . Since we require computing the exact nearest neighbor for at most $\mathcal{O}\left( {\log \left( n\right) /{\varepsilon }^{2}}\right)$ points,the running time is dominated by $\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$ ,which completes the proof.

定理2.1的证明 根据引理A.3，我们可以将输入数据划分为若干对${\left( {A}_{t},{B}_{t}\right) }_{t \in  \left\lbrack  T\right\rbrack  }$，使得每对的宽高比至多为多项式$\left( {n/\varepsilon }\right)$，且$\mathrm{{CH}}\left( {A,B}\right)$能很好地近似为$\mathrm{{CH}}\left( {{A}_{t},{B}_{t}}\right)$的直和。接着，重复引理A.2的构造并应用马尔可夫不等式，我们得到列表${\left\{  {\mathbf{D}}_{a}\right\}  }_{a \in  A}$，其中至少有${99}/{100}$的概率，对所有$a \in  A$、${\mathbf{D}}_{a} \geq  \mathop{\min }\limits_{{b \in  B}}\parallel a - b{\parallel }_{1}$和$\mathbf{D} = \mathop{\sum }\limits_{{a \in  A}}{\mathbf{D}}_{a} \leq  \mathcal{O}\left( {\log \left( n\right) }\right) \mathrm{{CH}}\left( {A,B}\right)$成立。使用上述参数调用引理2.2，且$T = \mathcal{O}\left( {\log \left( n\right) /{\varepsilon }^{2}}\right)$足以获得估计量$\eta$，这是$\operatorname{CH}\left( {A,B}\right)$的一个$\left( {1 \pm  \varepsilon }\right)$相对误差近似。由于我们最多只需为$\mathcal{O}\left( {\log \left( n\right) /{\varepsilon }^{2}}\right)$个点计算精确最近邻，运行时间主要取决于$\mathcal{O}\left( {{nd}\log \left( n\right) /{\varepsilon }^{2}}\right)$，至此完成证明。

### A.1 Analysis for CrudeNN

### A.1 CrudeNN算法分析

In this subsection, we focus analyze the CrudeNN algorithm and provide a proof for Lemma A.2. A construction of hash family satisfying Definition A. 1 is given in Section A.2. Each function from the family can be evaluated in $\mathcal{O}\left( d\right)$ time per point. We are now ready to prove Lemma A.2

本小节我们重点分析CrudeNN算法，并为引理A.2提供证明。满足定义A.1的哈希族构造将在A.2节给出。该族中每个函数对每个点的计算时间为$\mathcal{O}\left( d\right)$。现在我们可以开始证明引理A.2

Proof of Lemma $\left| \overline{A.2}\right|$ We note that the first item is trivially true,since $\operatorname{CrudeNN}\left( {A,B}\right)$ always sets ${\mathbf{D}}_{a}$ to be some distance between $a$ and a point in $B$ . Thus,this distance can only be larger than the true minimum distance. The more challenging aspect is obtaining an upper bound on the expected value of ${\mathbf{D}}_{a}$ . Consider a fixed setting of $a \in  A$ ,and the following setting of parameters:

引理$\left| \overline{A.2}\right|$的证明 注意到第一条显然成立，因为$\operatorname{CrudeNN}\left( {A,B}\right)$总是将${\mathbf{D}}_{a}$设为$a$与$B$中某点之间的距离。因此这个距离只会大于真实最小距离。更具挑战性的是获得${\mathbf{D}}_{a}$期望值的上界。考虑固定参数$a \in  A$及以下参数设置：

$$
b = \underset{{b}^{\prime } \in  B}{\arg \min }{d}_{X}\left( {a,{b}^{\prime }}\right) \;{\gamma }_{a} = {d}_{X}\left( {a,b}\right) \;{i}_{0} = \left\lceil  {{\log }_{2}{\gamma }_{a}}\right\rceil  ,
$$

and notice that since ${\gamma }_{a}$ is between 1 and poly $\left( {n/\varepsilon }\right)$ ,we have ${i}_{0}$ is at least 0 and at most $L =$ $\mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$ . We will upper bound the expectation of ${\mathbf{D}}_{a}$ by considering a parameter $c > 1$ (which will later be set to $\mathcal{O}\left( {\log n}\right)$ )),and integrating over the probability that ${\mathbf{D}}_{a}$ is at least $\gamma$ ,for all $\gamma  \geq  c \cdot  {\gamma }_{a} :$

注意到由于${\gamma }_{a}$介于1和多项式$\left( {n/\varepsilon }\right)$之间，我们有${i}_{0}$至少为0且至多为$L =$$\mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$。我们将通过考虑参数$c > 1$（后续将设为$\mathcal{O}\left( {\log n}\right)$）来上界${\mathbf{D}}_{a}$的期望，并对所有$\gamma  \geq  c \cdot  {\gamma }_{a} :$积分${\mathbf{D}}_{a}$至少为$\gamma$的概率

$$
\mathbf{E}\left\lbrack  {\mathbf{D}}_{a}\right\rbrack   \leq  c \cdot  {\gamma }_{a} + {\int }_{c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{D}}_{a} \geq  \gamma }\right\rbrack  {d\gamma }. \tag{1}
$$

We now show that for any $a \in  A$ ,the probability that ${\mathbf{D}}_{a}$ is larger than $\gamma$ can be appropriately bounded. Consider the following two bad events.

现在证明对于任意$a \in  A$，${\mathbf{D}}_{a}$大于$\gamma$的概率可以被适当限制。考虑以下两个不良事件。

- ${\mathbf{E}}_{1}\left( \gamma \right)$ : This event occurs when there exists a point ${b}^{\prime } \in  B$ at distance at least $\gamma$ from $a$ and there exists an index $i \leq  {i}_{0}$ for which ${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$ .

- ${\mathbf{E}}_{1}\left( \gamma \right)$：当存在距离$a$至少$\gamma$的点${b}^{\prime } \in  B$，且存在索引$i \leq  {i}_{0}$使得${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$时，此事件发生。

- ${\mathbf{E}}_{2}\left( \gamma \right)$ : This event occurs when there exists an index $i > {i}_{0}$ such that:

- ${\mathbf{E}}_{2}\left( \gamma \right)$：当存在满足以下条件的索引$i > {i}_{0}$时，此事件发生：

$$
\text{- For every}{i}^{\prime } \in  \left\{  {{i}_{0},\ldots ,i - 1}\right\}  \text{,we have}{\mathbf{h}}_{{i}^{\prime }}\left( a\right)  \neq  {\mathbf{h}}_{{i}^{\prime }}\left( b\right) \text{for all}b \in  B\text{.}
$$

- There exists ${b}^{\prime } \in  B$ at distance at least $\gamma$ from $a$ where ${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$ .

- 存在距离$a$至少$\gamma$的${b}^{\prime } \in  B$，其中${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$。

We note that whenever $\operatorname{CrudeNN}\left( {A,B}\right)$ set ${\mathbf{D}}_{a}$ larger than $\gamma$ ,one of the two events, ${\mathbf{E}}_{1}\left( \gamma \right)$ or ${\mathbf{E}}_{2}\left( \gamma \right)$ ,must have been triggered. To see why,suppose $\operatorname{CrudeNN}\left( {A,B}\right)$ set ${\mathbf{D}}_{a}$ to be larger than $\gamma$ because a point ${b}^{\prime } \in  B$ with ${d}_{X}\left( {a,{b}^{\prime }}\right)  \geq  \gamma$ happened to have ${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$ ,for an index $i \in  \{ 0,\ldots ,L\}$ ,and that the index $i$ was the first case where it happened. If $i \leq  {i}_{0}$ ,this is event ${\mathbf{E}}_{1}\left( \gamma \right)$ . If $i > {i}_{0}$ ,we claim event ${\mathbf{E}}_{2}\left( \gamma \right)$ occurred: in addition to ${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$ ,it must have been the case that,for all ${i}^{\prime } \in  \left\{  {{i}_{0},\ldots ,i - 1}\right\}  ,{\mathbf{h}}_{{i}^{\prime }}\left( a\right)  \neq  {\mathbf{h}}_{{i}^{\prime }}\left( b\right)$ (otherwise, $i$ would not be the first index). We will upper bound the probability that either event ${\mathbf{E}}_{1}\left( \gamma \right)$ or ${\mathbf{E}}_{2}\left( \gamma \right)$ occurs. We make use of the tail bounds as stated in Definition A.1. The upper bound for the probability that ${\mathbf{E}}_{1}\left( \gamma \right)$ is simple, since it suffices to union bound over at most $n$ points at distance larger than $\gamma$ ,using the fact that ${r}_{{i}_{0}} = {2}^{{i}_{0}}$ is at most $2 \cdot  {\gamma }_{a}$ :

需注意，每当$\operatorname{CrudeNN}\left( {A,B}\right)$将${\mathbf{D}}_{a}$设为大于$\gamma$时，两个事件${\mathbf{E}}_{1}\left( \gamma \right)$或${\mathbf{E}}_{2}\left( \gamma \right)$必触发其一。原因在于：假设$\operatorname{CrudeNN}\left( {A,B}\right)$因索引$i \in  \{ 0,\ldots ,L\}$对应的点${b}^{\prime } \in  B$（具有属性${d}_{X}\left( {a,{b}^{\prime }}\right)  \geq  \gamma$）恰好满足${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$而调大${\mathbf{D}}_{a}$，且$i$是首个满足此条件的索引。若$i \leq  {i}_{0}$，则对应事件${\mathbf{E}}_{1}\left( \gamma \right)$；若$i > {i}_{0}$，则事件${\mathbf{E}}_{2}\left( \gamma \right)$必然发生——除${\mathbf{h}}_{i}\left( a\right)  = {\mathbf{h}}_{i}\left( {b}^{\prime }\right)$外，所有${i}^{\prime } \in  \left\{  {{i}_{0},\ldots ,i - 1}\right\}  ,{\mathbf{h}}_{{i}^{\prime }}\left( a\right)  \neq  {\mathbf{h}}_{{i}^{\prime }}\left( b\right)$都必须满足条件（否则$i$就不是首个索引）。我们将对事件${\mathbf{E}}_{1}\left( \gamma \right)$或${\mathbf{E}}_{2}\left( \gamma \right)$的发生概率进行上界估计，应用定义A.1所述的尾界理论。对于事件${\mathbf{E}}_{1}\left( \gamma \right)$的概率上界计算较为简单，只需对距离超过$\gamma$的至多$n$个点进行联合界估计，并利用${r}_{{i}_{0}} = {2}^{{i}_{0}}$不超过$2 \cdot  {\gamma }_{a}$的特性。

$$
\Pr \left\lbrack  {{\mathbf{E}}_{1}\left( \gamma \right) }\right\rbrack   \leq  n \cdot  \exp \left( {-{c}_{1} \cdot  \frac{\gamma }{2{\gamma }_{a}}}\right) . \tag{2}
$$

We will upper bound the probability that event ${\mathbf{E}}_{2}\left( \gamma \right)$ a bit more carefully. We will use the fact that for all $i$ ,the parameter ${r}_{i}$ is always between ${2}^{i - {i}_{0}}{\gamma }_{a}$ and ${2}^{i - {i}_{0} + 1}{\gamma }_{a}$ .

我们将更谨慎地估计事件${\mathbf{E}}_{2}\left( \gamma \right)$的概率上界。利用的关键性质是：对于所有$i$，参数${r}_{i}$始终介于${2}^{i - {i}_{0}}{\gamma }_{a}$与${2}^{i - {i}_{0} + 1}{\gamma }_{a}$之间。

$$
\Pr \left\lbrack  {{\mathbf{E}}_{2}\left( \gamma \right) }\right\rbrack   \leq  \mathop{\sum }\limits_{{i > {i}_{0}}}\left( {\mathop{\prod }\limits_{{{i}^{\prime } = {i}_{0}}}^{{i - 1}}\frac{{\gamma }_{a}}{{r}_{{i}^{\prime }}}}\right)  \cdot  \max \left\{  {n \cdot  \exp \left( {-{c}_{1} \cdot  \frac{\gamma }{{r}_{i}}}\right) ,1}\right\}  
$$

$$
 \leq  \mathop{\sum }\limits_{{i > {i}_{0}}}{2}^{-\left( {0 + \cdots  + \left( {i - 1 - {i}_{0}}\right) }\right) }\max \left\{  {\exp \left( {\ln \left( n\right)  - {c}_{1} \cdot  \frac{\gamma }{{2}^{i - {i}_{0} + 1} \cdot  {\gamma }_{a}}}\right) ,1}\right\}  
$$

$$
 \leq  \mathop{\sum }\limits_{{k \geq  0}}{2}^{-\Omega \left( {k}^{2}\right) } \cdot  \max \left\{  {\exp \left( {\ln \left( n\right)  - {c}_{1} \cdot  \frac{\gamma }{{2}^{k + 2} \cdot  {\gamma }_{a}}}\right) ,1}\right\}  . \tag{3}
$$

With the above two upper bounds in place, we upper bound (1) by dividing the integral into the two contributing summands,from ${\mathbf{E}}_{1}\left( \gamma \right)$ and ${\mathbf{E}}_{2}\left( \gamma \right)$ ,and then upper bounding each individually. Namely, we have

基于上述两个上界，我们通过将积分拆分为来自${\mathbf{E}}_{1}\left( \gamma \right)$和${\mathbf{E}}_{2}\left( \gamma \right)$的两个贡献项来对(1)式进行上界限定，并分别对每项进行上界估计。具体而言，可得

$$
{\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{D}}_{a} \geq  \gamma }\right\rbrack  {d\gamma } \leq  {\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{E}}_{1}\left( \gamma \right) }\right\rbrack  {d\gamma } + {\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{E}}_{2}\left( \gamma \right) }\right\rbrack  {d\gamma }.
$$

The first summand can be simply upper bounded by using the upper bound from (2), where we have

第一项可直接采用(2)式中的上界进行限定，其中满足

$$
{\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{E}}_{1}\left( \gamma \right) }\right\rbrack  {d\gamma } \leq  {\int }_{\gamma  : c{\gamma }_{a}}^{\infty }n\exp \left( {-{c}_{1} \cdot  \frac{\gamma }{2{\gamma }_{a}}}\right) {d\gamma } \leq  \frac{n \cdot  2{\gamma }_{a}}{{c}_{1}} \cdot  {e}^{-{c}_{1}c/2} \leq  {\gamma }_{a}
$$

for a large enough $c = \Theta \left( {\log n}\right)$ . The second summand is upper bounded by the upper bound in (3), while being slightly more careful in the computation. In particular, we first commute the summation over $k$ and the integral; then,for each $k \geq  0$ ,we define

当$c = \Theta \left( {\log n}\right)$足够大时。第二项则采用(3)式中的上界，但在计算过程中需更为谨慎。特别地，我们首先交换$k$求和与积分的顺序；随后，对于每个$k \geq  0$，定义

$$
{\alpha }_{k} \mathrel{\text{:=}} {2}^{k + 3}\ln \left( n\right)  \cdot  {\gamma }_{a}/{c}_{1},
$$

and we break up the integral into the interval $\left\lbrack  {c \cdot  {\gamma }_{a},{\alpha }_{k}}\right\rbrack$ (if ${\alpha }_{k} < c{\gamma }_{a}$ ,the interval is empty),as well as $\left\lbrack  {{\alpha }_{k},\infty }\right)$ :

并将积分区间划分为$\left\lbrack  {c \cdot  {\gamma }_{a},{\alpha }_{k}}\right\rbrack$（若${\alpha }_{k} < c{\gamma }_{a}$，则该区间为空集）以及$\left\lbrack  {{\alpha }_{k},\infty }\right)$：

$$
{\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{E}}_{2}\left( \gamma \right) }\right\rbrack  {d\gamma } \leq  \mathop{\sum }\limits_{{k \geq  0}}{2}^{-\Omega \left( {k}^{2}\right) }{\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\max \left\{  {\exp \left( {\ln \left( n\right)  - {c}_{1} \cdot  \frac{\gamma }{{2}^{k + 2} \cdot  {\gamma }_{a}}}\right) ,1}\right\}  {d\gamma }
$$

$$
 \leq  \mathop{\sum }\limits_{{k \geq  0}}{2}^{-\Omega \left( {k}^{2}\right) }\left( {{\left( {\alpha }_{k} - c \cdot  {\gamma }_{a}\right) }^{ + } + {\int }_{\gamma  : {\alpha }_{k}}^{\infty }\exp \left( {-\frac{{c}_{1}}{2} \cdot  \frac{\gamma }{{2}^{k + 2}{\gamma }_{a}}}\right) {d\gamma }}\right) ,
$$

where in the second inequality,we used the fact that the setting of ${\alpha }_{k}$ ,the additional $\ln \left( n\right)$ factor in the exponent can be removed up to a factor of two. Thus,

在第二个不等式中，我们利用了${\alpha }_{k}$的设定条件，使得指数部分的额外$\ln \left( n\right)$因子可在系数为2的范围内消除。因此，

$$
{\int }_{\gamma  : c{\gamma }_{a}}^{\infty }\Pr \left\lbrack  {{\mathbf{E}}_{2}\left( \gamma \right) }\right\rbrack  {d\gamma } \leq  \mathop{\sum }\limits_{{k \geq  0}}{2}^{-\Omega \left( {k}^{2}\right) }\left( {{\gamma }_{a} \cdot  \mathcal{O}\left( {{2}^{k}\log n}\right)  + {\gamma }_{a} \cdot  \mathcal{O}\left( {2}^{k}\right) }\right)  = \mathcal{O}\left( {\log n}\right)  \cdot  {\gamma }_{a}.
$$

Finally,the running time is dominated by the cost of evaluating $\mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$ functions on $n$ points in dimension $d$ . Since each evaluation takes $\mathcal{O}\left( d\right)$ time,the bound follows.

最终，运行时间主要取决于在$d$维空间中$n$个点上评估$\mathcal{O}\left( {\log \left( {n/\varepsilon }\right) }\right)$函数的成本。由于每次评估耗时$\mathcal{O}\left( d\right)$，故可得该界。

### A.2 Locality-Sensitive Hashing at Every Scale

### A.2 多尺度局部敏感哈希

Lemma A. 4 (Constructing a LSH at every scale). For any $r \geq  0$ and any $d \in  \mathbb{N}$ ,there exists a hash family $\mathcal{H}\left( r\right)$ such that for any two points $x,y \in  {\mathbb{R}}^{d}$ ,

引理A.4（构建多尺度LSH）。对于任意$r \geq  0$和$d \in  \mathbb{N}$，存在哈希族$\mathcal{H}\left( r\right)$使得对于任意两点$x,y \in  {\mathbb{R}}^{d}$，

$$
\mathop{\Pr }\limits_{{\mathbf{h} \sim  \mathcal{H}\left( r\right) }}\left\lbrack  {\mathbf{h}\left( x\right)  \neq  \mathbf{h}\left( y\right) }\right\rbrack   \leq  \frac{\parallel x - y{\parallel }_{1}}{r}
$$

$$
\mathop{\Pr }\limits_{{\mathbf{h} \sim  \mathcal{H}\left( r\right) }}\left\lbrack  {\mathbf{h}\left( x\right)  = \mathbf{h}\left( y\right) }\right\rbrack   \leq  \exp \left( {-\frac{\parallel x - y{\parallel }_{1}}{r}}\right) .
$$

In addition,for any $\mathbf{h} \sim  \mathcal{H}\left( r\right) ,\mathbf{h}\left( x\right)$ may be computed in $\mathcal{O}\left( d\right)$ time.

此外，任意$\mathbf{h} \sim  \mathcal{H}\left( r\right) ,\mathbf{h}\left( x\right)$可在$\mathcal{O}\left( d\right)$时间内完成计算。

Proof. The construction proceeds in the following way: in order to generate a function $\mathbf{h} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{Z}}^{d}$ sampled from $\mathcal{H}\left( r\right)$ ,

证明。构造过程如下：为生成从$\mathcal{H}\left( r\right)$采样的函数$\mathbf{h} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{Z}}^{d}$，

- We sample a random vector $\mathbf{z} \sim  {\left\lbrack  0,r\right\rbrack  }^{d}$ .

- 采样随机向量$\mathbf{z} \sim  {\left\lbrack  0,r\right\rbrack  }^{d}$

- We let

- 令

$$
\mathbf{h}\left( x\right)  = \left( {\left\lceil  \frac{{x}_{1} + {\mathbf{z}}_{1}}{r}\right\rceil  ,\left\lceil  \frac{{x}_{2} + {\mathbf{z}}_{2}}{r}\right\rceil  ,\ldots ,\left\lceil  \frac{{x}_{d} + {\mathbf{z}}_{d}}{r}\right\rceil  }\right) .
$$

Fix $x,y \in  {\mathbb{R}}^{d}$ . If $\mathbf{h}\left( x\right)  \neq  \mathbf{h}\left( y\right)$ ,there exists some coordinate $k \in  \left\lbrack  d\right\rbrack$ on which $\mathbf{h}{\left( x\right) }_{k} \neq  \mathbf{h}{\left( y\right) }_{k}$ . This occurs whenever (i) $\left| {{x}_{k} - {y}_{k}}\right|  > r$ ,or (ii) $\left| {{x}_{k} - {y}_{k}}\right|  \leq  r$ ,but ${z}_{k}$ happens to fall within an interval of length $\left| {{x}_{k} - {y}_{k}}\right|$ ,thereby separating $x$ from $y$ . By a union bound,

固定$x,y \in  {\mathbb{R}}^{d}$。若$\mathbf{h}\left( x\right)  \neq  \mathbf{h}\left( y\right)$，则存在某个坐标$k \in  \left\lbrack  d\right\rbrack$使得$\mathbf{h}{\left( x\right) }_{k} \neq  \mathbf{h}{\left( y\right) }_{k}$。该情况发生在：(i)$\left| {{x}_{k} - {y}_{k}}\right|  > r$，或(ii)$\left| {{x}_{k} - {y}_{k}}\right|  \leq  r$但${z}_{k}$恰好落入长度为$\left| {{x}_{k} - {y}_{k}}\right|$的区间，从而将$x$与$y$分离。根据联合界原理，

$$
\mathop{\Pr }\limits_{{\mathbf{h} \sim  \mathcal{H}\left( r\right) }}\left\lbrack  {\mathbf{h}\left( x\right)  \neq  \mathbf{h}\left( y\right) }\right\rbrack   \leq  \mathop{\sum }\limits_{{k = 1}}^{d}\frac{\left| {x}_{k} - {y}_{k}\right| }{r} = \frac{\parallel x - y{\parallel }_{1}}{r}.
$$

On the other hand,in order for $\mathbf{h}\left( x\right)  = \mathbf{h}\left( y\right)$ ,it must be the case that every $\left| {{x}_{k} - {y}_{k}}\right|  \leq  r$ ,and in addition,the threshold ${\mathbf{z}}_{k}$ always avoids an interval of length $\left| {{x}_{k} - {y}_{k}}\right|$ . The probability that this occurs is

另一方面，为了使$\mathbf{h}\left( x\right)  = \mathbf{h}\left( y\right)$成立，必须满足每个$\left| {{x}_{k} - {y}_{k}}\right|  \leq  r$，此外阈值${\mathbf{z}}_{k}$始终避开长度为$\left| {{x}_{k} - {y}_{k}}\right|$的区间。此情况发生的概率为

$$
\mathop{\Pr }\limits_{{\mathbf{h} \sim  \mathcal{H}\left( r\right) }}\left\lbrack  {\mathbf{h}\left( x\right)  = \mathbf{h}\left( y\right) }\right\rbrack   = \mathop{\prod }\limits_{{k = 1}}^{d}\max \left\{  {0,1 - \frac{\left| {x}_{k} - {y}_{k}\right| }{r}}\right\}   \leq  \exp \left( {-\mathop{\sum }\limits_{{k = 1}}^{d}\frac{\left| {x}_{k} - {y}_{k}\right| }{r}}\right) 
$$

$$
 \leq  \exp \left( {-\frac{\parallel x - y{\parallel }_{1}}{r}}\right) .
$$

Extending the above construction to ${\ell }_{2}$ follows from embedding the points $A \cup  B$ into ${\ell }_{1}$ via a standard construction.

将上述构造扩展至${\ell }_{2}$，可通过标准构造将点$A \cup  B$嵌入${\ell }_{1}$来实现。

Theorem A. 5 ([Mat13]). Let $\varepsilon  \in  \left( {0,1}\right)$ and define $\mathbf{T} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{k}$ by

定理A.5（[Mat13]）。设$\varepsilon  \in  \left( {0,1}\right)$并定义$\mathbf{T} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{k}$为

$$
\mathbf{T}{\left( x\right) }_{i} = \frac{1}{\beta k}\mathop{\sum }\limits_{{j = 1}}^{d}{Z}_{ij}{x}_{j},\;i = 1,\ldots ,k
$$

where $\beta  = \sqrt{2/\pi }$ . Then for every vector $x \in  {\mathbb{R}}^{d}$ ,we have

其中$\beta  = \sqrt{2/\pi }$。则对任意向量$x \in  {\mathbb{R}}^{d}$，有

$$
\Pr \left\lbrack  {\left( {1 - \varepsilon }\right) \parallel x{\parallel }_{2} \leq  \parallel \mathbf{T}\left( x\right) {\parallel }_{1} \leq  \left( {1 + \varepsilon }\right) \parallel x{\parallel }_{2}}\right\rbrack   \geq  1 - {e}^{c{\varepsilon }^{2}k},
$$

where $c > 0$ is a constant.

其中$c > 0$为常数。

The map $\mathbf{T} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{k}$ with $k = \mathcal{O}\left( {\log n/{\varepsilon }^{2}}\right)$ gives an embedding of $A \cup  B$ into ${\ell }_{1}^{k}$ of distortion $\left( {1 \pm  \varepsilon }\right)$ with high probability. Formally,with probability at least $1 - 1/n$ over the draw of $\mathbf{T}$ with $t = \mathcal{O}\left( {\log n/{\varepsilon }^{2}}\right)$ ,every $a \in  A$ and $b \in  B$ satisfies

映射$\mathbf{T} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{k}$与$k = \mathcal{O}\left( {\log n/{\varepsilon }^{2}}\right)$以高概率将$A \cup  B$嵌入${\ell }_{1}^{k}$，畸变率不超过$\left( {1 \pm  \varepsilon }\right)$。形式化地说，在$\mathbf{T}$的抽取过程中（设$t = \mathcal{O}\left( {\log n/{\varepsilon }^{2}}\right)$），以至少$1 - 1/n$的概率，所有$a \in  A$与$b \in  B$均满足

$$
\left( {1 - \varepsilon }\right) \parallel a - b{\parallel }_{2} \leq  \parallel \mathbf{T}\left( a\right)  - \mathbf{T}\left( b\right) {\parallel }_{1} \leq  \left( {1 + \varepsilon }\right) \parallel a - b{\parallel }_{2}.
$$

This embedding has the effect of reducing ${\ell }_{2}$ to ${\ell }_{1}$ without affecting the Chamfer distance of the mapped points by more than a $\left( {1 \pm  \varepsilon }\right)$ -factor. In addition,the embedding incurs an extra additive factor of $\mathcal{O}\left( {{nd}\log n/{\varepsilon }^{2}}\right)$ to the running time in order to perform the embedding for all points.

该嵌入方法可将${\ell }_{2}$降至${\ell }_{1}$，同时确保映射点集的Chamfer距离变化不超过$\left( {1 \pm  \varepsilon }\right)$倍。此外，对所有点执行嵌入操作会给运行时间带来$\mathcal{O}\left( {{nd}\log n/{\varepsilon }^{2}}\right)$的额外加性因子。

### A.3 Reduction to poly $\left( {n/\varepsilon }\right)$ Aspect Ratio for ${\ell }_{p},p \in  \left\lbrack  {1,2}\right\rbrack$

### A.3 将${\ell }_{p},p \in  \left\lbrack  {1,2}\right\rbrack$问题归约为多$\left( {n/\varepsilon }\right)$纵横比情形

In this section,we discuss how to reduce to the case of a $\operatorname{poly}\left( {n/\varepsilon }\right)$ aspect ratio. The reduction proceeds by first obtaining a very crude estimate of $\mathrm{{CH}}\left( {A,B}\right)$ (which will be a poly(n)-approximation), applying a locality-sensitive hash function in order to partition points of $A$ and $B$ which are significantly farther than $\operatorname{poly}\left( n\right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$ . Finally,we add $\mathcal{O}\left( {\log n}\right)$ coordinates and add random vector of length $\operatorname{poly}\left( {\varepsilon /n}\right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$ in order to guarantee that the minimum distance is at least $\operatorname{poly}\left( {\varepsilon /n}\right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$ without changing $\mathrm{{CH}}\left( {A,B}\right)$ significantly.

本节讨论如何将问题归约为有限纵横比情形。首先通过极粗略估计$\mathrm{{CH}}\left( {A,B}\right)$（即多项式(n)近似解），应用局部敏感哈希函数划分$A$与$B$中距离显著超过$\operatorname{poly}\left( n\right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$的点集。最后添加$\mathcal{O}\left( {\log n}\right)$个坐标及长度为$\operatorname{poly}\left( {\varepsilon /n}\right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$的随机向量，在保证最小距离不小于$\operatorname{poly}\left( {\varepsilon /n}\right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$的同时，使$\mathrm{{CH}}\left( {A,B}\right)$不发生显著变化。

Proof of Lemma $\overline{A.3}$ Partitioning Given Very Crude Estimates. In particular,suppose that with an $\mathcal{O}\left( {nd}\right)  + \mathcal{O}\left( {n\overline{\log }n}\right)$ time computation,we can achieve a value of $\mathbf{\eta } \in  {\mathbb{R}}_{ \geq  0}$ which satisfies

引理$\overline{A.3}$证明：基于极粗略估计的划分。特别地，假设通过$\mathcal{O}\left( {nd}\right)  + \mathcal{O}\left( {n\overline{\log }n}\right)$时间计算可获得满足下式的$\mathbf{\eta } \in  {\mathbb{R}}_{ \geq  0}$值：

$$
\mathrm{{CH}}\left( {A,B}\right)  \leq  \eta  \leq  c \cdot  \mathrm{{CH}}\left( {A,B}\right) ,
$$

with high probability (which we will show how to do briefly with $c = \operatorname{poly}\left( n\right)$ ). Then,consider sampling $\mathbf{h} \sim  \mathcal{H}\left( {{cn} \cdot  \mathbf{\eta }}\right)$ and partitioning $A$ and $B$ into equivalence classes according to where they hash to under $\mathbf{h}$ . The probability that two points at distance farther than $\mathcal{O}\left( {\log n}\right)  \cdot  {cn} \cdot  \mathbf{\eta }$ collide under $\mathbf{h}$ is small enough to union bound over at most ${n}^{2}$ many possible pairs of vectors. In addition,the probability that there exists $a \in  A$ for which $b \in  B$ minimizing $\parallel a - b{\parallel }_{p}$ satisfies $\mathbf{h}\left( a\right)  \neq  \mathbf{h}\left( b\right)$ is at most $\mathrm{{CH}}\left( {A,B}\right) /\left( {{cn} \cdot  \mathbf{\eta }}\right)  \leq  1/n$ . This latter inequality implies that computing the Chamfer distance of the corresponding parts in the partition and summing them is equivalent to computing $\mathrm{{CH}}\left( {A,B}\right)$

以高概率（我们将简要演示如何通过$c = \operatorname{poly}\left( n\right)$实现）。接着，考虑对$\mathbf{h} \sim  \mathcal{H}\left( {{cn} \cdot  \mathbf{\eta }}\right)$进行采样，并根据$\mathbf{h}$下的哈希位置将$A$和$B$划分为等价类。距离超过$\mathcal{O}\left( {\log n}\right)  \cdot  {cn} \cdot  \mathbf{\eta }$的两点在$\mathbf{h}$下发生碰撞的概率足够小，可通过最多${n}^{2}$个向量对的并集约束来覆盖。此外，存在满足$b \in  B$最小化$\parallel a - b{\parallel }_{p}$的$a \in  A$且符合$\mathbf{h}\left( a\right)  \neq  \mathbf{h}\left( b\right)$的概率至多为$\mathrm{{CH}}\left( {A,B}\right) /\left( {{cn} \cdot  \mathbf{\eta }}\right)  \leq  1/n$。后一个不等式意味着，计算划分中对应部分的Chamfer距离并求和，等价于计算$\mathrm{{CH}}\left( {A,B}\right)$

Getting Very Crude Estimates. We now show how to obtain a poly(n)-approximation to $\operatorname{CH}\left( {A,B}\right)$ in time $\mathcal{O}\left( {nd}\right)  + \mathcal{O}\left( {n\log n}\right)$ for points in ${\mathbb{R}}^{d}$ with ${\ell }_{p}$ distance. This is done via the $p$ -stable sketch of Indyk [Ind06]. In particular,we sample a vector $\mathbf{g} \in  {\mathbb{R}}^{d}$ by independent $p$ -stable random variables (for instance, $\mathbf{g}$ is a standard Gaussian vector for $p = 2$ and a vector of independent Cauchy random variables for $p = 1$ ). We may then compute the scalar random variables $\{ \langle a,\mathbf{g}\rangle {\} }_{a \in  A}$ and $\{ \langle b,\mathbf{g}\rangle {\} }_{b \in  B}$ , which give a projection onto a one-dimensional space. By $p$ -stability,for any $a \in  A$ and $b \in  B$ ,the distribution of $\langle a,\mathbf{g}\rangle  - \langle b,\mathbf{g}\rangle$ is exactly as $\parallel a - b{\parallel }_{p} \cdot  {\mathbf{g}}^{\prime }$ ,where ${\mathbf{g}}^{\prime }$ is an independent $p$ -stable random variable. Hence,we will have that for every $a \in  A$ and $b \in  B$ ,

获取极粗略估计。我们现在展示如何在$\mathcal{O}\left( {nd}\right)  + \mathcal{O}\left( {n\log n}\right)$时间内，对${\mathbb{R}}^{d}$空间中具有${\ell }_{p}$距离的点获得poly(n)量级的$\operatorname{CH}\left( {A,B}\right)$近似解。这通过Indyk[Ind06]提出的$p$稳定草图实现。具体而言，我们通过独立$p$稳定随机变量（例如当$p = 2$时采用标准高斯向量，$p = 1$时采用独立柯西随机变量向量）采样向量$\mathbf{g} \in  {\mathbb{R}}^{d}$。随后可计算标量随机变量$\{ \langle a,\mathbf{g}\rangle {\} }_{a \in  A}$和$\{ \langle b,\mathbf{g}\rangle {\} }_{b \in  B}$，它们给出了一维空间上的投影。根据$p$稳定性，对于任意$a \in  A$和$b \in  B$，$\langle a,\mathbf{g}\rangle  - \langle b,\mathbf{g}\rangle$的分布完全等同于$\parallel a - b{\parallel }_{p} \cdot  {\mathbf{g}}^{\prime }$，其中${\mathbf{g}}^{\prime }$是独立的$p$稳定随机变量。因此，对于每个$a \in  A$和$b \in  B$，我们将有

$$
\frac{\parallel a - b{\parallel }_{p}}{\operatorname{poly}\left( n\right) } \leq  \left| {\langle a,\mathbf{g}\rangle -\langle b,\mathbf{g}\rangle }\right|  \leq  \parallel a - b{\parallel }_{p} \cdot  \operatorname{poly}\left( n\right) ,
$$

with probability $1 - 1/\operatorname{poly}\left( n\right)$ and hence $\operatorname{CH}\left( {\{ \langle a,\mathbf{g}\rangle {\} }_{a \in  A},\{ \langle b,\mathbf{g}\rangle {\} }_{b \in  B}}\right)$ ,which is computable by 1-dimensional nearest neighbor search (i.e., repeatedly querying a binary search tree), gives a poly(n)-approximation to $\operatorname{CH}\left( {A,B}\right)$ .

以$1 - 1/\operatorname{poly}\left( n\right)$概率成立，因此$\operatorname{CH}\left( {\{ \langle a,\mathbf{g}\rangle {\} }_{a \in  A},\{ \langle b,\mathbf{g}\rangle {\} }_{b \in  B}}\right)$（可通过一维最近邻搜索计算，即重复查询二叉搜索树）给出$\operatorname{CH}\left( {A,B}\right)$的poly(n)量级近似。

Adding Distance Finally,we now note that $\eta /c$ gives us a lower bound on $\mathrm{{CH}}\left( {A,B}\right)$ . Suppose we append $\mathcal{O}\left( {\log n}\right)$ coordinates to each point and in those coordinates,we add a random vector of norm $\varepsilon  \cdot  \mathbf{\eta }/\left( {cn}\right)$ . With high probability,every pair of points is now at distance at least $\varepsilon  \cdot  \mathbf{\eta }/\left( {cn}\right)$ . In addition, the Chamfer distance between the new set of points increases by at most an additive $\mathcal{O}\left( {\varepsilon \mathbf{\eta }/c}\right)$ ,which is at most $\mathcal{O}\left( \varepsilon \right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$ ,proving Lemma A. 3

距离扩展 最后，我们注意到$\eta /c$给出了$\mathrm{{CH}}\left( {A,B}\right)$的下界。假设我们为每个点追加$\mathcal{O}\left( {\log n}\right)$个坐标维度，并在这些维度中添加一个模长为$\varepsilon  \cdot  \mathbf{\eta }/\left( {cn}\right)$的随机向量。以高概率计算，此时每对点之间的最小距离为$\varepsilon  \cdot  \mathbf{\eta }/\left( {cn}\right)$。此外，新点集之间的Chamfer距离最多增加$\mathcal{O}\left( {\varepsilon \mathbf{\eta }/c}\right)$的加性量，该值不超过$\mathcal{O}\left( \varepsilon \right)  \cdot  \mathrm{{CH}}\left( {A,B}\right)$，由此得证引理A.3。

## B Deferred Analysis from Section 4

## B 第4节的延迟分析

Proof. To set the notation,we let ${n}_{A} = \left| A\right| ,{n}_{B} = \left| B\right|$ .

证明。为建立符号体系，设${n}_{A} = \left| A\right| ,{n}_{B} = \left| B\right|$。

The proof mimics the argument from [Roh19], which proved a similar hardness result for the problem of computing the Earth-Mover Distance. In particular, Lemma 4.3 from that paper shows the following claim.

该证明沿用了[Roh19]中的论证方法，该论文针对计算地球移动距离（Earth-Mover Distance）问题证明了类似的硬度结果。特别地，该论文中的引理4.3展示了以下论断。

Claim B.1. For any two sets $A,B \subseteq  \{ 0,1{\} }^{d}$ ,there is a mapping $f : \{ 0,1{\} }^{d} \rightarrow  \{ 0,1{\} }^{d}$ ,and a vector $v \in  \{ 0,1{\} }^{d}$ ,such that $d$ " $= \mathcal{O}\left( d\right)$ and for any $a \in  A,b \in  B$ :

论断B.1 对于任意两个集合$A,B \subseteq  \{ 0,1{\} }^{d}$，存在映射$f : \{ 0,1{\} }^{d} \rightarrow  \{ 0,1{\} }^{d}$和向量$v \in  \{ 0,1{\} }^{d}$，使得$d$"$= \mathcal{O}\left( d\right)$且对于任意$a \in  A,b \in  B$：

$$
\text{- If}a \cdot  b = 0\text{then}\parallel f\left( a\right)  - f\left( b\right) {\parallel }_{1} = {4d} + 2\text{,}
$$

$$
\text{- If}a \cdot  b > 0\text{then}\parallel f\left( a\right)  - f\left( b\right) {\parallel }_{1} \geq  {4d} + 4\text{,}
$$

$$
\text{-}\parallel f\left( a\right)  - v{\parallel }_{1} = {4d} + 4\text{.}
$$

Furthermore,each evaluation $f\left( a\right)$ can be performed in $\mathcal{O}\left( d\right)$ time.

此外，每次$f\left( a\right)$求值可在$\mathcal{O}\left( d\right)$时间内完成。

We will be running ${ALG}$ on sets $A$ " $= \{ f\left( a\right)  : a \in  A\}$ and $B$ " $= \{ f\left( b\right)  : b \in  B\}  \cup  \{ v\}$ . It can be seen that,given a reported mapping $g$ ,we can assume that for all $a$ " $\in  A$ " we have ${\begin{Vmatrix}{a}^{n} - g\left( {a}^{n}\right) \end{Vmatrix}}_{1} \leq  {4d} + 4$ ,as otherwise $g$ can map ${a}^{\prime \prime }$ to $v$ . If for all $a \in  A$ there exists $b \in  B$ such that $a \cdot  b = 0$ ,i.e., $A$ does not contain a hitting vector,then the optimal mapping cost is ${n}_{A}\left( {{4d} + 2}\right)$ . More generally,let $H$ be the set of vectors $a \in  A$ hitting $B$ ,and let $h = \left| H\right|$ . It can be seen that

我们将在集合${ALG}$上运行$A$"$= \{ f\left( a\right)  : a \in  A\}$与$B$"$= \{ f\left( b\right)  : b \in  B\}  \cup  \{ v\}$。可以看出，给定一个已报告的映射$g$，我们可以假设对于所有$a$"$\in  A$"都有${\begin{Vmatrix}{a}^{n} - g\left( {a}^{n}\right) \end{Vmatrix}}_{1} \leq  {4d} + 4$，否则$g$能将${a}^{\prime \prime }$映射到$v$。若对所有$a \in  A$都存在$b \in  B$使得$a \cdot  b = 0$（即$A$不包含命中向量），则最优映射成本为${n}_{A}\left( {{4d} + 2}\right)$。更一般地，设$H$为命中$B$的向量$a \in  A$的集合，并设$h = \left| H\right|$。由此可见

$$
\mathrm{{CH}}\left( {{A}^{\prime \prime },{B}^{\prime \prime }}\right)  = h\left( {{4d} + 4}\right)  + \left( {{n}_{A} - h}\right) \left( {{4d} + 2}\right)  = {n}_{A}\left( {{4d} + 2}\right)  + {2h}.
$$

Thus,if we could compute $\mathrm{{CH}}\left( {A",B"}\right)$ exactly,we would determine if $h = 0$ and solve HS. In what follows we show that even an approximate solution can be used to accomplish this task as long as $\varepsilon$ is small enough.

因此，若能精确计算$\mathrm{{CH}}\left( {A",B"}\right)$，我们就能判定$h = 0$并解决HS问题。下文将证明，只要$\varepsilon$足够小，近似解也能完成该任务。

Let $t = c\log \left( n\right) /\varepsilon$ for some large enough constant $c > 1$ . Consider the algorithm HittingSet(A,B) that solves HS by invoking the algorithm ${ALG}$ .

设$t = c\log \left( n\right) /\varepsilon$为某个足够大的常数$c > 1$。考虑通过调用算法${ALG}$来解决HS问题的HittingSet(A,B)算法。

---

Subroutine HittingSet(A,B)

子程序HittingSet(A,B)

Input: Two sets $A,B \subset  \{ 0,1{\} }^{d}$ of size at most $n$ ,and an oracle access to ${ALG}$ that computes

输入：两个最多包含$n$个元素的集合$A,B \subset  \{ 0,1{\} }^{d}$，及用于计算${ALG}$的预言机访问权限

$\left( {1 + \varepsilon }\right)$ -approximate CH.

$\left( {1 + \varepsilon }\right)$-近似CH。

Output: Determines whether there exists $a \in  A$ such that $a \cdot  b > 0$ for all $b \in  B$

输出：判定是否存在$a \in  A$使得对所有$b \in  B$都有$a \cdot  b > 0$

		1. Sample (uniformly,without replacement) $\min \left( {t,\left| A\right| }\right)$ distinct vectors $a \in  A$ ,and for

		1. 均匀无放回地抽样$\min \left( {t,\left| A\right| }\right)$个不同向量$a \in  A$，并检查

			each of them check if $a \cdot  b > 0$ for all $B$ . If such an $a$ is found,return YES.

			是否存在$a$使得对所有$B$都有$a \cdot  b > 0$。若找到则返回YES。

		2. Construct $A$ ", $B$ " as in Claim B.1,and invoke ${ALG}$ . Let $g : A$ " $\rightarrow  B$ " be the returned

		2. 如权利要求B.1所述构建$A$"$B$"，调用${ALG}$。设返回的

			map.

			映射为<b3></b3>"<b4></b4>"。

		3. Identify the set $M$ containing all $a \in  A$ such that $\parallel g\left( {f\left( a\right) }\right)  - f\left( a\right) {\parallel }_{1} = {4d} + 2$ . Note

		3. 识别包含所有满足$\parallel g\left( {f\left( a\right) }\right)  - f\left( a\right) {\parallel }_{1} = {4d} + 2$的$a \in  A$的集合$M$。注意

			that $a \cdot  b = 0$ for $b \in  B$ such that $f\left( b\right)  = g\left( {f\left( a\right) }\right)$ .

			对于满足$f\left( b\right)  = g\left( {f\left( a\right) }\right)$的$b \in  B$有$a \cdot  b = 0$。

		4. Recursively execute HittingSet(A - M,B)

		4. 递归执行HittingSet(A - M,B)

---

Figure 7: Reduction from Hitting Set to $\left( {1 + \varepsilon }\right)$ -approximate $\mathrm{{CH}}$ ,implemented using algorithm ${ALG}$ .

图7：通过算法${ALG}$实现的从命中集到$\left( {1 + \varepsilon }\right)$-近似$\mathrm{{CH}}$的归约。

It can be seen that the first three steps of the algorithm take at most $\mathcal{O}\left( {ntd}\right)$ time. Furthermore,if the algorithm terminates,it reports the correct answer,as only vectors $a$ that are guaranteed not to be hitting are removed in the recursion. It remains to bound the total number and cost of the recursive steps. To this end,we will show that,with high probability,in each recursive call we have $\left| {A - M}\right|  \leq$ $\left| A\right| /2$ . This will yield a total time of $\log n\left\lbrack  {\left( {ntd}\right)  + T\left( {n + 1,\mathcal{O}\left( d\right) ,\varepsilon }\right) }\right\rbrack$ . Since $t = c\log \left( n\right) /\varepsilon$ , $d = {\log }^{2}n$ and $\varepsilon  = \frac{\Theta \left( 1\right) }{d}$ ,it follows that the time is at most $n{\log }^{5}\left( n\right)  + \log \left( n\right) T\left( {n + 1,\mathcal{O}\left( d\right) ,\varepsilon }\right)$ , and the theorem follows.

可以看出，该算法的前三个步骤最多耗时$\mathcal{O}\left( {ntd}\right)$。此外，若算法终止，其报告的结果必然正确，因为在递归过程中仅会移除被确保非命中向量的$a$。接下来需要界定递归步骤的总次数及计算成本。为此，我们将证明在每次递归调用中，以高概率满足$\left| {A - M}\right|  \leq$$\left| A\right| /2$。这将使得总时间复杂度为$\log n\left\lbrack  {\left( {ntd}\right)  + T\left( {n + 1,\mathcal{O}\left( d\right) ,\varepsilon }\right) }\right\rbrack$。由于$t = c\log \left( n\right) /\varepsilon$、$d = {\log }^{2}n$及$\varepsilon  = \frac{\Theta \left( 1\right) }{d}$，可推得耗时上限为$n{\log }^{5}\left( n\right)  + \log \left( n\right) T\left( {n + 1,\mathcal{O}\left( d\right) ,\varepsilon }\right)$，定理得证。

To show that $\left| {A - M}\right|  \leq  \left| A\right| /2$ ,first observe that if the algorithm reaches step (2),then for a large enough constant $c > 1$ it holds,with high probability,that the set $H$ of hitting vectors $a$ has cardinality at most $\varepsilon  \cdot  {n}_{A}$ ,as otherwise one such vector would have been sampled. Thus,the subroutine ${ALG}$ returns a map where the vast majority of the points $f\left( a\right)$ have been matched to a point $f\left( b\right)$ such that $\parallel f\left( a\right)  - f\left( b\right) {\parallel }_{1} = {4d} + 2$ . More formally,the cost of the mapping $g$ is

为证明$\left| {A - M}\right|  \leq  \left| A\right| /2$，首先观察到若算法执行至步骤(2)，对于足够大的常数$c > 1$，以高概率成立的是：命中向量集$H$$a$的基数至多为$\varepsilon  \cdot  {n}_{A}$，否则应已抽样到此类向量。因此，子程序${ALG}$返回的映射中，绝大多数点$f\left( a\right)$已匹配至满足$\parallel f\left( a\right)  - f\left( b\right) {\parallel }_{1} = {4d} + 2$的点$f\left( b\right)$。更形式化地说，该映射$g$的成本可表示为

$$
C = \mathop{\sum }\limits_{{{a}^{\prime \prime } \in  {A}^{\prime \prime }}}{\begin{Vmatrix}{a}^{\prime \prime } - g\left( {a}^{\prime \prime }\right) \end{Vmatrix}}_{1}
$$

$$
 \leq  \left( {1 + \varepsilon }\right) \left\lbrack  {{n}_{A}\left( {{4d} + 2}\right)  + 2\left| H\right| }\right\rbrack  
$$

$$
 \leq  \left( {1 + \varepsilon }\right) \left\lbrack  {{n}_{A}\left( {{4d} + 2}\right)  + {2\varepsilon }{n}_{A}}\right\rbrack  
$$

$$
 \leq  {n}_{A}\left( {{4d} + 2}\right)  + {4\varepsilon }{n}_{A}\left( {d + 2}\right) 
$$

$$
 \leq  {n}_{A}\left( {{4d} + 2}\right)  + {n}_{A}
$$

where in the last step we used the assumption about $\varepsilon$ .

其中最后一步运用了关于$\varepsilon$的假设。

Denote $m = \left| M\right|$ . Observe that the cost $C$ of the mapping $g$ can be alternatively written as:

记作$m = \left| M\right|$。注意到映射$g$的成本$C$可改写为：

$$
C = m\left( {{4d} + 2}\right)  + \left( {{n}_{A} - m}\right) \left( {{4d} + 4}\right)  = {n}_{A}\left( {{4d} + 4}\right)  - {2m}
$$

This implies $m = \left( {{n}_{A}\left( {{4d} + 4}\right)  - C}\right) /2$ . Since we showed earlier that $C \leq  {n}_{A}\left( {{4d} + 2}\right)  + {n}_{A}$ ,we

由此可得$m = \left( {{n}_{A}\left( {{4d} + 4}\right)  - C}\right) /2$。鉴于前文已证$C \leq  {n}_{A}\left( {{4d} + 2}\right)  + {n}_{A}$，我们

conclude that

可得出结论：

$$
m = \left( {{n}_{A}\left( {{4d} + 4}\right)  - C}\right) /2 \geq  \left( {2{n}_{A} - {n}_{A}}\right) /2 = {n}_{A}/2.
$$

Thus, $\left| {A - M}\right|  = {n}_{A} - m \leq  {n}_{A}/2$ ,completing the proof.

因此$\left| {A - M}\right|  = {n}_{A} - m \leq  {n}_{A}/2$，证明完成。