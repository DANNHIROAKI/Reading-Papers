# Query-Aware Locality-Sensitive Hashing for Approximate Nearest Neighbor Search

# 用于近似最近邻搜索的查询感知局部敏感哈希

Qiang Huang,

黄强

Jianlin Feng,Yikai Zhang School of Software Sun Yat-sen University Guangzhou, China

冯建林、张逸凯 中山大学软件学院 中国广州

huangq2011@gmail.com fengjlin@mail.sysu.edu.cn echo_evenop@yahoo.com

huangq2011@gmail.com fengjlin@mail.sysu.edu.cn echo_evenop@yahoo.com

Qiong Fang

方琼

School of Software Engineering South China University of Technology Guangzhou, China

华南理工大学软件工程学院 中国广州

sefangq@scut.edu.cn

Wilfred Ng

威尔弗雷德·吴（Wilfred Ng）

Department of Computer

计算机系

Science and Engineering

科学与工程

Hong Kong University of

香港

Science and Technology

科技大学

Hong Kong, China

中国香港

wilfred@cse.ust.hk

## ABSTRACT

## 摘要

Locality-Sensitive Hashing (LSH) and its variants are the well-known indexing schemes for the $c$ -Approximate Nearest Neighbor ( $c$ -ANN) search problem in high-dimensional Euclidean space. Traditionally, LSH functions are constructed in a query-oblivious manner in the sense that buckets are partitioned before any query arrives. However, objects closer to a query may be partitioned into different buckets, which is undesirable. Due to the use of query-oblivious bucket partition, the state-of-the-art LSH schemes for external memory, namely C2LSH and LSB-Forest, only work with approximation ratio of integer $c \geq  2$ .

局部敏感哈希（Locality-Sensitive Hashing，LSH）及其变体是高维欧几里得空间中$c$ - 近似最近邻（$c$ - ANN）搜索问题的著名索引方案。传统上，LSH 函数以与查询无关的方式构建，即在任何查询到来之前就对桶进行划分。然而，与查询更接近的对象可能会被划分到不同的桶中，这是不理想的。由于使用了与查询无关的桶划分，用于外部内存的最先进的 LSH 方案，即 C2LSH 和 LSB - 森林，仅适用于整数$c \geq  2$的近似比。

In this paper, we introduce a novel concept of query-aware bucket partition which uses a given query as the "anchor" for bucket partition. Accordingly, a query-aware LSH function is a random projection coupled with query-aware bucket partition, which removes random shift required by traditional query-oblivious LSH functions. Notably, query-aware bucket partition can be easily implemented so that query performance is guaranteed. We propose a novel query-aware LSH scheme named QALSH for $c$ -ANN search over external memory. Our theoretical studies show that QALSH enjoys a guarantee on query quality. The use of query-aware LSH function enables QALSH to work with any approximation ratio $c > 1$ . Extensive experiments show that QALSH outperforms C2LSH and LSB-Forest, especially in high-dimensional space. Specifically,by using a ratio $c < 2$ , QALSH can achieve much better query quality.

在本文中，我们引入了一种新颖的查询感知桶划分（query-aware bucket partition）概念，该概念使用给定查询作为桶划分的“锚点”。因此，查询感知局部敏感哈希（LSH）函数是一种与查询感知桶划分相结合的随机投影，它消除了传统的不考虑查询的LSH函数所需的随机偏移。值得注意的是，查询感知桶划分可以轻松实现，从而保证查询性能。我们提出了一种名为QALSH的新颖查询感知LSH方案，用于外部内存上的$c$ - 近似最近邻（ANN）搜索。我们的理论研究表明，QALSH在查询质量上有保证。查询感知LSH函数的使用使QALSH能够以任何近似比$c > 1$ 工作。大量实验表明，QALSH的性能优于C2LSH和LSB - 森林，尤其是在高维空间中。具体而言，通过使用比率$c < 2$ ，QALSH可以实现更好的查询质量。

## 1. INTRODUCTION

## 1. 引言

The problem of Nearest Neighbor (NN) search in Euclidean space has wide applications, such as image and video databases, information retrieval, and data mining. In many applications, data objects are typically represented as Euclidean vectors (or points). For example, in image search applications, images can be naturally mapped into high-dimensional feature vectors with one dimension per pixel.

欧几里得空间中的最近邻（Nearest Neighbor，NN）搜索问题有着广泛的应用，如图像和视频数据库、信息检索以及数据挖掘等。在许多应用中，数据对象通常表示为欧几里得向量（或点）。例如，在图像搜索应用中，图像可以自然地映射为高维特征向量，每个像素对应一个维度。

To bypass the difficulty of finding exact query answers in high-dimensional space, the approximate version of the problem,called the $c$ -Approximate Nearest Neighbor ( $c$ - ANN) search, has attracted extensive studies [13, 10, 3, 7, 15.4. For a given approximation ratio $c\left( {c > 1}\right)$ and a query object $q,c$ -ANN search returns the object within distance $c$ times the distance of $q$ to its exact nearest neighbor. Since the approximation ratio $c$ is an upper bound,a smaller $c$ means a better guarantee of query quality.

为了绕过在高维空间中寻找精确查询答案的困难，该问题的近似版本，即$c$ - 近似最近邻（$c$ - Approximate Nearest Neighbor，$c$ - ANN）搜索，吸引了广泛的研究[13, 10, 3, 7, 15.4]。对于给定的近似比率$c\left( {c > 1}\right)$和查询对象$q,c$，$c$ - ANN搜索返回的对象与查询对象的距离是查询对象与其精确最近邻距离的$c$倍。由于近似比率$c$是一个上限，因此$c$越小意味着对查询质量的保证越好。

Locality-Sensitive Hashing (LSH) [7, 2] and its variants 12,15,4 are the well-known indexing schemes for $c$ -ANN search in high-dimensional space. The seminal work on LSH scheme for Euclidean space was first presented by Datar et al. 2, which is named E2LSH ${}^{1}$ later. E2LSH constructs LSH functions based on $p$ -stable distributions. For Euclidean space, 2-stable distribution, i.e., standard normal distribution $\mathcal{N}\left( {0,1}\right)$ ,is used in E2LSH and its variants,such as Entropy-LSH [12], LSB-Forest [15] and C2LSH [4].

局部敏感哈希（Locality-Sensitive Hashing，LSH）[7, 2]及其变体[12,15,4]是高维空间中$c$ -近似最近邻（ANN）搜索的著名索引方案。关于欧几里得空间LSH方案的开创性工作最早由达塔尔（Datar）等人[2]提出，该方案后来被命名为E2LSH ${}^{1}$。E2LSH基于$p$ -稳定分布构建LSH函数。对于欧几里得空间，E2LSH及其变体（如熵LSH（Entropy-LSH）[12]、最低有效位森林（LSB-Forest）[15]和C2LSH [4]）使用2 -稳定分布，即标准正态分布$\mathcal{N}\left( {0,1}\right)$。

Under an LSH function for Euclidean space, the probability of collision (or simply collision probability) between two objects decreases monotonically as their Euclidean distance increases. An LSH function of E2LSH has the basic form as follows: ${h}_{\overrightarrow{a},b}\left( o\right)  = \left\lfloor  \frac{\overrightarrow{a} \cdot  \overrightarrow{o} + b}{w}\right\rfloor$ . Such an LSH function partitions an object into a bucket in the following manner: first it projects object $o$ along the random line identified by $\overrightarrow{a}$ (or simply the random line $\overrightarrow{a}$ ),and then gives the projection $\overrightarrow{a} \cdot  \overrightarrow{o}$ a random shift of $b$ ,and finally uses the floor function to locate the interval of width $w$ in which the shifted projection falls. The interval is simply taken as the bucket of object $o$ . In this approach,bucket partition is carried out before any query arrives, and hence it is said to be query-oblivious. Accordingly, the corresponding LSH function is called a query-oblivious LSH function. An illustration of query-oblivious bucket partition is given in Figure 1, where the random line is segmented into buckets $\lbrack 0,w),\lbrack  - w,0)$ , $\lbrack w,{2w}),\lbrack  - {2w}, - w)$ ,and so on. Due to the use of the floor function, here the origin (i.e., 0) of the random line can be viewed as the "anchor" for locating the boundary of each interval. Query-oblivious bucket partition has the advantage of leaving the overhead of bucket partition to the preprocessing step. However, query-oblivious bucket partition may lead to some undesirable situation, i.e., objects closer to a query may be partitioned into different buckets. For example,as shown in Figure 1,although ${o}_{1}$ is closer to $q$ than ${o}_{2},{o}_{1}$ and $q$ are segmented into different buckets.

在欧几里得空间的局部敏感哈希（LSH）函数下，两个对象之间的碰撞概率（或简称为碰撞可能性）会随着它们的欧几里得距离增加而单调递减。E2LSH的LSH函数基本形式如下：${h}_{\overrightarrow{a},b}\left( o\right)  = \left\lfloor  \frac{\overrightarrow{a} \cdot  \overrightarrow{o} + b}{w}\right\rfloor$ 。这样的LSH函数按以下方式将一个对象划分到一个桶中：首先，它将对象$o$ 沿着由$\overrightarrow{a}$ 确定的随机线（或简称为随机线$\overrightarrow{a}$ ）进行投影，然后对投影$\overrightarrow{a} \cdot  \overrightarrow{o}$ 进行$b$ 的随机偏移，最后使用向下取整函数来确定偏移后的投影所在的宽度为$w$ 的区间。该区间就被简单地视为对象$o$ 的桶。在这种方法中，桶划分是在任何查询到来之前进行的，因此它被称为与查询无关的。相应地，对应的LSH函数被称为与查询无关的LSH函数。图1给出了与查询无关的桶划分的示例，其中随机线被分割成桶$\lbrack 0,w),\lbrack  - w,0)$ 、$\lbrack w,{2w}),\lbrack  - {2w}, - w)$ 等等。由于使用了向下取整函数，这里随机线的原点（即0）可以被视为定位每个区间边界的“锚点”。与查询无关的桶划分的优点是将桶划分的开销留给预处理步骤。然而，与查询无关的桶划分可能会导致一些不理想的情况，即与查询更接近的对象可能被划分到不同的桶中。例如，如图1所示，尽管${o}_{1}$ 比${o}_{2},{o}_{1}$ 更接近$q$ ，但$q$ 被划分到了不同的桶中。

---

<!-- Footnote -->

of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.For any use beyond those covered by this license, obtain permission by emailing

要查看本许可协议的详细内容，请访问 http://creativecommons.org/licenses/by-nc-nd/4.0/。如需进行超出本许可协议范围的使用，请通过发送电子邮件获取许可

Copyright 2015 VLDB Endowment 2150-8097/15/09.

版权所有 2015 年超大型数据库捐赠基金（VLDB Endowment）2150 - 8097/15/09。

http://www.mit.edu/~andoni/LSH

<!-- Footnote -->

---

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_1.jpg?x=243&y=156&w=521&h=114&r=0"/>

Figure 1: Query-Oblivious Bucket Partition

图 1：查询无关的桶分区

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_1.jpg?x=238&y=365&w=531&h=122&r=0"/>

Figure 2: Query-Aware Bucket Partition

图 2：查询感知的桶分区

<!-- Media -->

The basic form of ${h}_{\overrightarrow{a},b}\left( o\right)$ has been used by the variants of E2LSH, such as Entropy-LSH and C2LSH. In LSB-Forest, even though the LSH functions $\left( {{h}_{\overrightarrow{a},b}\left( o\right)  = \overrightarrow{a} \cdot  \overrightarrow{o} + b}\right)$ only explicitly involve random projection and random shift, its encoding hash values by Z-order also implicitly use the origin as the "anchor". Random shift along the random line is a prerequisite for the query-oblivious hash functions to be locality-sensitive. In a word, the state-of-the-art LSH schemes for external memory, namely C2LSH and LSB-Forest, are both built on query-oblivious bucket partition. As analyzed in Section 5.1, due to the use of query-oblivious bucket partition, $\mathrm{C}2\mathrm{{LSH}}$ and LSB-Forest only work with integer $c \geq  2$ for $c$ -ANN search,which is limited for applications that prefer a ratio as strong as $c < 2$ .

${h}_{\overrightarrow{a},b}\left( o\right)$的基本形式已被E2LSH的变体所采用，如熵局部敏感哈希（Entropy-LSH）和C2局部敏感哈希（C2LSH）。在最低有效位森林（LSB-Forest）中，尽管局部敏感哈希（LSH）函数$\left( {{h}_{\overrightarrow{a},b}\left( o\right)  = \overrightarrow{a} \cdot  \overrightarrow{o} + b}\right)$仅显式地涉及随机投影和随机偏移，但其通过Z序对哈希值进行编码也隐式地使用原点作为“锚点”。沿随机直线进行随机偏移是与查询无关的哈希函数具有局部敏感性的先决条件。总之，用于外部内存的最先进的局部敏感哈希方案，即C2局部敏感哈希（C2LSH）和最低有效位森林（LSB-Forest），均基于与查询无关的桶划分构建。正如5.1节所分析的，由于使用了与查询无关的桶划分，$\mathrm{C}2\mathrm{{LSH}}$和最低有效位森林（LSB-Forest）仅适用于整数$c \geq  2$的$c$ -近似最近邻（ANN）搜索，这对于希望达到$c < 2$这样强比率的应用来说是有局限性的。

Motivated by the limitations of query-oblivious bucket partition, we propose a novel concept of query-aware bucket partition and develop novel query-aware LSH functions accordingly. Given a pre-specified bucket width $w$ ,a hash function ${h}_{\overrightarrow{a}}\left( o\right)  = \overrightarrow{a} \cdot  \overrightarrow{o}$ first projects object $o$ along the random line $\overrightarrow{a}$ as before. When a query $q$ arrives,we compute the projection of $q$ (i.e., ${h}_{\overrightarrow{a}}\left( q\right)$ ) and take the query projection (or simply the query) as the "anchor" for bucket partition. Specifically,the interval $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ ,i.e.,a bucket of width $w$ centered at ${h}_{\overrightarrow{a}}\left( q\right)$ (or simply at $q$ ),is first imposed along the random line $\overrightarrow{a}$ . And if necessary,we can impose buckets with any larger bucket width, in the same manner of using the query as the "anchor". This approach of bucket partition is said to be query-aware. In Section 3, we show that the hash function ${h}_{\overrightarrow{a}}\left( o\right)$ coupled with query-aware bucket partition is indeed locality-sensitive, and hence is called a query-aware LSH function. An example of query-aware bucket partition is illustrated in Figure 2,where $h\left( q\right)$ evenly splits the buckets into two half-buckets of width $\frac{w}{2}$ . By applying the query-aware bucket partition, ${o}_{1}$ and $q$ are partitioned into the same bucket, the undesirable situation illustrated in Figure 1 is then avoided.

受与查询无关的桶划分方法的局限性所驱动，我们提出了一种新颖的与查询相关的桶划分概念，并相应地开发了新颖的与查询相关的局部敏感哈希（LSH）函数。给定一个预先指定的桶宽度 $w$，哈希函数 ${h}_{\overrightarrow{a}}\left( o\right)  = \overrightarrow{a} \cdot  \overrightarrow{o}$ 首先像之前一样将对象 $o$ 沿着随机直线 $\overrightarrow{a}$ 进行投影。当一个查询 $q$ 到来时，我们计算 $q$ 的投影（即 ${h}_{\overrightarrow{a}}\left( q\right)$），并将查询投影（或简称为查询）作为桶划分的“锚点”。具体而言，首先沿着随机直线 $\overrightarrow{a}$ 划分区间 $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$，即一个以 ${h}_{\overrightarrow{a}}\left( q\right)$（或简称为以 $q$）为中心、宽度为 $w$ 的桶。如有必要，我们可以以将查询作为“锚点”的相同方式划分任意更大宽度的桶。这种桶划分方法被称为与查询相关的桶划分。在第3节中，我们证明了与查询相关的桶划分相结合的哈希函数 ${h}_{\overrightarrow{a}}\left( o\right)$ 确实具有局部敏感性，因此被称为与查询相关的LSH函数。图2展示了一个与查询相关的桶划分示例，其中 $h\left( q\right)$ 将桶均匀地划分为两个宽度为 $\frac{w}{2}$ 的半桶。通过应用与查询相关的桶划分，${o}_{1}$ 和 $q$ 被划分到同一个桶中，从而避免了图1中所示的不理想情况。

Notice that random shift is not necessary for query-aware bucket partition. Thus, compared to query-oblivious LSH functions, query-aware LSH functions are simpler to compute. However, we need to dynamically do query-aware bucket partition. Given a query-aware LSH function ${h}_{\overrightarrow{a}}\left( o\right)  =$ $\overrightarrow{a} \cdot  \overrightarrow{o}$ ,in the pre-processing step,we compute the projections of all the data objects along the random line, and index all the data projections by a ${B}^{ + }$ -tree. When a query object $q$ arrives,we compute the query projection and use the ${B}^{ + }$ -tree to locate objects falling in the interval $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - }\right.$ $\left. {\frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ . And if required by our search algorithm,we can gradually locate data objects even farther away from the query,just like performing a ${B}^{ + }$ -tree range search. In other words, we do not need to physically partition the whole random line at all. Therefore, the overhead of query-aware bucket partition is affordable.

请注意，对于查询感知的桶划分，随机偏移并非必要。因此，与查询无关的局部敏感哈希（LSH）函数相比，查询感知的LSH函数计算起来更简单。然而，我们需要动态地进行查询感知的桶划分。给定一个查询感知的LSH函数${h}_{\overrightarrow{a}}\left( o\right)  =$ $\overrightarrow{a} \cdot  \overrightarrow{o}$，在预处理步骤中，我们计算所有数据对象沿随机直线的投影，并通过${B}^{ + }$ -树对所有数据投影进行索引。当一个查询对象$q$到来时，我们计算查询投影，并使用${B}^{ + }$ -树来定位落在区间$\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - }\right.$ $\left. {\frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$内的对象。如果我们的搜索算法有要求，我们甚至可以逐步定位离查询对象更远的数据对象，就像执行一次${B}^{ + }$ -树范围搜索一样。换句话说，我们根本不需要对整个随机直线进行物理划分。因此，查询感知的桶划分的开销是可以承受的。

Based on query-aware LSH functions, we propose a novel Query-Aware LSH scheme called QALSH for $c$ -ANN search in high-dimensional Euclidean space. Interestingly, as analyzed in Section 5.1, query-aware bucket partition enables QALSH to work with any $c > 1$ . In this paper,we also develop a novel approach to setting the bucket width $w$ automatically, as shown in Section 5.3. In contrast, the state-of-the-art query-oblivious LSH schemes depend on manually setting $w$ . For example,both E2LSH and LSB-Forest manually set $w = {4.0}$ ,while C2LSH manually sets $w = {1.0}$ .

基于查询感知的局部敏感哈希（LSH）函数，我们提出了一种名为QALSH的新型查询感知LSH方案，用于高维欧几里得空间中的$c$ -近似最近邻（ANN）搜索。有趣的是，正如5.1节所分析的，查询感知的桶划分使QALSH能够适用于任何$c > 1$ 。在本文中，我们还开发了一种自动设置桶宽度$w$ 的新方法，如5.3节所示。相比之下，目前最先进的查询无关LSH方案依赖于手动设置$w$ 。例如，E2LSH和LSB - 森林（LSB - Forest）都手动设置$w = {4.0}$ ，而C2LSH手动设置$w = {1.0}$ 。

In summary, we introduce a novel concept of query-aware bucket partition and develop novel query-aware LSH functions accordingly. We propose a novel query-aware LSH scheme QALSH for high-dimensional $c$ -ANN search over external memory. QALSH works with any approximation ratio $c > 1$ and enjoys a theoretical guarantee on query quality. QALSH also solves the problem of $c$ -approximate $k$ - nearest neighbors(c - k - ANN)search. Extensive experiments on four real datasets show that in high-dimensional Euclidean space QALSH outperforms C2LSH and LSB-Forest which also have guarantee on query quality.

综上所述，我们引入了一种新颖的查询感知桶划分（query-aware bucket partition）概念，并相应地开发了新颖的查询感知局部敏感哈希（LSH，Locality-Sensitive Hashing）函数。我们提出了一种新颖的查询感知LSH方案QALSH，用于外部内存上的高维$c$ - 近似最近邻（ANN，Approximate Nearest Neighbor）搜索。QALSH适用于任何近似比$c > 1$，并在查询质量上有理论保证。QALSH还解决了$c$ - 近似$k$ - 最近邻（c - k - ANN）搜索问题。在四个真实数据集上的大量实验表明，在高维欧几里得空间中，QALSH的性能优于同样在查询质量上有保证的C2LSH和LSB - 森林（LSB - Forest）。

The rest of this paper is organized as follows. We first discuss preliminaries in Section 2. Then we introduce the query-aware LSH family in Section 3 The QALSH scheme is presented in Section 4 and its theoretical analysis is given in Section 5. Experimental studies are presented in Section 6. Related work is discussed in Section 7. Finally, we conclude our work in Section 8.

本文的其余部分组织如下。我们首先在第2节讨论预备知识。然后在第3节介绍查询感知LSH族。QALSH方案在第4节介绍，其理论分析在第5节给出。实验研究在第6节呈现。相关工作在第7节讨论。最后，我们在第8节对工作进行总结。

## 2. PRELIMINARIES

## 2. 预备知识

### 2.1 Problem Setting

### 2.1 问题设定

Let $D$ be a database of $n$ data objects in $d$ -dimensional Euclidean space ${\mathcal{R}}^{d}$ and let $\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix}$ denote the Euclidean distance between two objects ${o}_{1}$ and ${o}_{2}$ . Given a query object $q$ in ${\mathcal{R}}^{d}$ and an approximation ratio $c\left( {c > 1}\right) ,c$ - ANN search is to find an object $o \in  D$ such that $\parallel o,q\parallel  \leq$ $c\begin{Vmatrix}{{o}^{ * },q}\end{Vmatrix}$ ,where ${o}^{ * }$ is the exact $\mathrm{{NN}}$ of $q$ in $D$ . Similarly, $c$ - $k$ -ANN is to find $k$ objects ${o}_{i} \in  D\left( {1 \leq  i \leq  k}\right)$ such that $\begin{Vmatrix}{{o}_{i},q}\end{Vmatrix} \leq  c\begin{Vmatrix}{{o}_{i}^{ * },q}\end{Vmatrix}$ ,where ${o}_{i}^{ * }$ is the exact $i$ -th NN of $q$ in $D$ .

设 $D$ 为 $d$ 维欧几里得空间 ${\mathcal{R}}^{d}$ 中包含 $n$ 个数据对象的数据库，并用 $\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix}$ 表示两个对象 ${o}_{1}$ 和 ${o}_{2}$ 之间的欧几里得距离。给定 ${\mathcal{R}}^{d}$ 中的一个查询对象 $q$ 和一个近似比 $c\left( {c > 1}\right) ,c$ ，$c\left( {c > 1}\right) ,c$ - 近似最近邻（ANN）搜索是要找到一个对象 $o \in  D$ ，使得 $\parallel o,q\parallel  \leq$ $c\begin{Vmatrix}{{o}^{ * },q}\end{Vmatrix}$ ，其中 ${o}^{ * }$ 是 $q$ 在 $D$ 中的精确最近邻（NN）。类似地，$c$ - $k$ - 近似最近邻（ANN）是要找到 $k$ 个对象 ${o}_{i} \in  D\left( {1 \leq  i \leq  k}\right)$ ，使得 $\begin{Vmatrix}{{o}_{i},q}\end{Vmatrix} \leq  c\begin{Vmatrix}{{o}_{i}^{ * },q}\end{Vmatrix}$ ，其中 ${o}_{i}^{ * }$ 是 $q$ 在 $D$ 中的精确第 $i$ 近邻（NN）。

### 2.2 Query-Oblivious LSH Family

### 2.2 与查询无关的局部敏感哈希（LSH）族

A family of LSH functions is able to partition "closer" objects into the same bucket with an accordingly higher probability. If two objects $o$ and $q$ are partitioned into the same bucket by a hash function $h$ ,we say $o$ and $q$ collide under $h$ . Formally, an LSH function family (or simply an LSH family) in Euclidean space is defined as:

局部敏感哈希（LSH）函数族能够以相应更高的概率将“更接近”的对象划分到同一个桶中。如果两个对象 $o$ 和 $q$ 被哈希函数 $h$ 划分到同一个桶中，我们就说 $o$ 和 $q$ 在 $h$ 下发生碰撞。形式上，欧几里得空间中的局部敏感哈希（LSH）函数族（或简称为 LSH 族）定义如下：

Definition 1. Given a search radius $r$ and approximation ratio $c$ ,an LSH function family $H = \left\{  {h : {\mathcal{R}}^{d} \rightarrow  U}\right\}$ is said to be $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -sensitive,if,for any $o,q \in  {\mathcal{R}}^{d}$ we have

定义 1. 给定搜索半径 $r$ 和近似比 $c$，如果对于任意 $o,q \in  {\mathcal{R}}^{d}$ 都有，则称局部敏感哈希（LSH）函数族 $H = \left\{  {h : {\mathcal{R}}^{d} \rightarrow  U}\right\}$ 是 $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -敏感的。

- if $\parallel o,q\parallel  \leq  r$ ,then $\mathop{\Pr }\limits_{H}\left\lbrack  {o\text{and}q\text{collide under}h}\right\rbrack   \geq  {p}_{1}$ ;

- 如果 $\parallel o,q\parallel  \leq  r$，那么 $\mathop{\Pr }\limits_{H}\left\lbrack  {o\text{and}q\text{collide under}h}\right\rbrack   \geq  {p}_{1}$；

- if $\parallel o,q\parallel  > {cr}$ ,then $\mathop{\Pr }\limits_{H}\left\lbrack  {o\text{and}q\text{collide under}h}\right\rbrack   \leq  {p}_{2}$ . where $c > 1$ and ${p}_{1} > {p}_{2}$ . For ease of reference, ${p}_{1}$ and ${p}_{2}$ are called positively-colliding probability and negatively-colliding probability, respectively.

- 如果 $\parallel o,q\parallel  > {cr}$，那么 $\mathop{\Pr }\limits_{H}\left\lbrack  {o\text{and}q\text{collide under}h}\right\rbrack   \leq  {p}_{2}$，其中 $c > 1$ 和 ${p}_{1} > {p}_{2}$。为便于参考，${p}_{1}$ 和 ${p}_{2}$ 分别称为正碰撞概率（positively-colliding probability）和负碰撞概率（negatively-colliding probability）。

A query-oblivious LSH family is an LSH family $H =$ $\left\{  {h : {\mathcal{R}}^{d} \rightarrow  \mathcal{Z}}\right\}$ where each hash function $h$ exploits query-oblivious bucket partition, i.e., buckets in the hash table of $h$ are statically determined before any query arrives. Normally,for a query-oblivious LSH function $h$ ,two objects $o$ and $q$ collide under $h$ means $h\left( o\right)  = h\left( q\right)$ ,where $h\left( o\right)$ identifies the bucket of $o$ . A typical query-oblivious LSH function is formally defined as follows [2].

查询无关的局部敏感哈希（LSH）族是一个LSH族$H =$ $\left\{  {h : {\mathcal{R}}^{d} \rightarrow  \mathcal{Z}}\right\}$，其中每个哈希函数$h$采用查询无关的桶划分，即$h$的哈希表中的桶在任何查询到达之前就已静态确定。通常，对于一个查询无关的LSH函数$h$，两个对象$o$和$q$在$h$下发生冲突意味着$h\left( o\right)  = h\left( q\right)$，其中$h\left( o\right)$标识$o$所在的桶。一个典型的查询无关的LSH函数正式定义如下[2]。

$$
{h}_{\overrightarrow{a},b}\left( o\right)  = \left\lfloor  \frac{\overrightarrow{a} \cdot  \overrightarrow{o} + b}{w}\right\rfloor  , \tag{1}
$$

where $\overrightarrow{o}$ is a $d$ -dimensional Euclidean vector representing object $o,\overrightarrow{a}$ is a $d$ -dimensional random vector with each entry drawn independently from standard normal distribution $\mathcal{N}\left( {0,1}\right)$ . $w$ is the pre-specified bucket width,and $b$ is a real number uniformly drawn from $\lbrack 0,w)$ .

其中 $\overrightarrow{o}$ 是一个 $d$ 维欧几里得向量（Euclidean vector），表示对象 $o,\overrightarrow{a}$ 是一个 $d$ 维随机向量，其每个元素均独立地从标准正态分布 $\mathcal{N}\left( {0,1}\right)$ 中抽取。$w$ 是预先指定的桶宽度（bucket width），$b$ 是一个从 $\lbrack 0,w)$ 中均匀抽取的实数。

For two objects ${o}_{1}$ and ${o}_{2}$ ,and a uniformly randomly chosen hash function ${h}_{\overrightarrow{a},b}$ ,let $s = \begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix}$ ,and then their collision probability is computed as follows [2]:

对于两个对象 ${o}_{1}$ 和 ${o}_{2}$，以及一个均匀随机选择的哈希函数 ${h}_{\overrightarrow{a},b}$，设 $s = \begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix}$，然后按照如下方式计算它们的碰撞概率 [2]：

$$
\xi \left( s\right)  = P{r}_{\overrightarrow{a},b}\left\lbrack  {{h}_{\overrightarrow{a},b}\left( {o}_{1}\right)  = {h}_{\overrightarrow{a},b}\left( {o}_{2}\right) }\right\rbrack   \tag{2}
$$

$$
 = {\int }_{0}^{w}\frac{1}{s}{f}_{2}\left( \frac{t}{s}\right) \left( {1 - \frac{t}{w}}\right) {dt}
$$

where ${f}_{2}\left( x\right)  = \frac{2}{\sqrt{2\pi }}{e}^{-\frac{{x}^{2}}{2}}$ . For a fixed $w,\xi \left( s\right)$ decreases monotonically as $s$ increases. With ${\xi }_{1} = \xi \left( r\right)$ and ${\xi }_{2} = \xi \left( {cr}\right)$ , the family of hash functions ${h}_{\overrightarrow{a},b}$ is $\left( {r,{cr},{\xi }_{1},{\xi }_{2}}\right)$ -sensitive. Specifically,if we set $r = 1$ and ${cr} = c$ ,we have Lemma 1 as follows [2] :

其中 ${f}_{2}\left( x\right)  = \frac{2}{\sqrt{2\pi }}{e}^{-\frac{{x}^{2}}{2}}$ 。对于固定的 $w,\xi \left( s\right)$ ，其随 $s$ 的增大而单调递减。当 ${\xi }_{1} = \xi \left( r\right)$ 且 ${\xi }_{2} = \xi \left( {cr}\right)$ 时，哈希函数族 ${h}_{\overrightarrow{a},b}$ 具有 $\left( {r,{cr},{\xi }_{1},{\xi }_{2}}\right)$ -敏感性。具体而言，如果我们令 $r = 1$ 且 ${cr} = c$ ，则有如下引理 1 [2] ：

LEMMA 1. The query-oblivious LSH family identified by Equation 1 is $\left( {1,c,{\xi }_{1},{\xi }_{2}}\right)$ -sensitive,where ${\xi }_{1} = \xi \left( 1\right)$ and ${\xi }_{2} = \xi \left( c\right)$ .

引理1. 由公式1确定的与查询无关的局部敏感哈希（LSH）族是$\left( {1,c,{\xi }_{1},{\xi }_{2}}\right)$敏感的，其中${\xi }_{1} = \xi \left( 1\right)$且${\xi }_{2} = \xi \left( c\right)$。

## 3. QUERY-AWARE LSH FAMILY

## 3. 与查询相关的局部敏感哈希（LSH）族

In this section we first introduce the concept of query-aware LSH functions. Then we make a computational comparison of positively- and negatively-colliding probabilities between query-oblivious and query-aware LSH families. Finally, we show that query-aware LSH family is able to support virtual rehashing in a simple and quick manner.

在本节中，我们首先介绍与查询相关的局部敏感哈希（LSH）函数的概念。然后，我们对与查询无关和与查询相关的局部敏感哈希（LSH）族的正碰撞概率和负碰撞概率进行计算比较。最后，我们表明与查询相关的局部敏感哈希（LSH）族能够以简单快捷的方式支持虚拟重哈希。

### 3.1 $\left( {1,c,{p}_{1},{p}_{2}}\right)$ -sensitive LSH Family

### 3.1 $\left( {1,c,{p}_{1},{p}_{2}}\right)$敏感的局部敏感哈希（LSH）族

Constructing LSH functions in a query-aware manner consists of two steps: random projection and query-aware bucket partition. Formally,a query-aware hash function ${h}_{\overrightarrow{a}}\left( o\right)$ : ${\mathcal{R}}^{d} \rightarrow  \mathcal{R}$ maps a $d$ -dimensional object $\overrightarrow{o}$ to a number along the real line identified by a random vector $\overrightarrow{a}$ ,whose entries are drawn independently from $\mathcal{N}\left( {0,1}\right)$ . For a fixed $\overrightarrow{a}$ ,the corresponding hash function ${h}_{\overrightarrow{a}}\left( o\right)$ is defined as follows:

以查询感知的方式构建局部敏感哈希（LSH）函数包括两个步骤：随机投影和查询感知的桶划分。形式上，一个查询感知的哈希函数 ${h}_{\overrightarrow{a}}\left( o\right)$ : ${\mathcal{R}}^{d} \rightarrow  \mathcal{R}$ 将一个 $d$ 维对象 $\overrightarrow{o}$ 沿着由随机向量 $\overrightarrow{a}$ 确定的实数轴映射为一个数字，该随机向量的元素是从 $\mathcal{N}\left( {0,1}\right)$ 中独立抽取的。对于一个固定的 $\overrightarrow{a}$，相应的哈希函数 ${h}_{\overrightarrow{a}}\left( o\right)$ 定义如下：

$$
{h}_{\overrightarrow{a}}\left( o\right)  = \overrightarrow{a} \cdot  \overrightarrow{o} \tag{3}
$$

For all the data objects, their projections along the random line $\overrightarrow{a}$ are computed in the pre-processing step. When a query object $q$ arrives,we obtain the query projection by computing ${h}_{\overrightarrow{a}}\left( q\right)$ . Then,we use the query as the "anchor" to locate the anchor bucket with width $w$ (defined by ${h}_{\overrightarrow{a}}\left( \cdot \right)$ ), i.e.,the interval $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ . If the projection of an object $o$ (i.e., ${h}_{\overrightarrow{a}}\left( o\right)$ ),falls in the anchor bucket with width $w$ ,i.e., $\left| {{h}_{\overrightarrow{a}}\left( o\right)  - {h}_{\overrightarrow{a}}\left( q\right) }\right|  \leq  \frac{w}{2}$ ,we say $o$ collides with $q$ under ${h}_{\overrightarrow{a}}$ .

对于所有数据对象，在预处理步骤中计算它们在随机直线$\overrightarrow{a}$上的投影。当一个查询对象$q$到来时，我们通过计算${h}_{\overrightarrow{a}}\left( q\right)$得到查询投影。然后，我们使用该查询作为“锚点”来定位宽度为$w$（由${h}_{\overrightarrow{a}}\left( \cdot \right)$定义）的锚桶，即区间$\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$。如果一个对象$o$的投影（即${h}_{\overrightarrow{a}}\left( o\right)$）落在宽度为$w$的锚桶中，即$\left| {{h}_{\overrightarrow{a}}\left( o\right)  - {h}_{\overrightarrow{a}}\left( q\right) }\right|  \leq  \frac{w}{2}$，我们称在${h}_{\overrightarrow{a}}$下$o$与$q$发生碰撞。

We now show that the family of hash functions ${h}_{\overrightarrow{a}}\left( o\right)$ coupled with query-aware bucket partition is locality-sensitive. In this sense,each ${h}_{\overrightarrow{a}}\left( o\right)$ in the family is said to be a query-aware LSH function. For objects $o$ and $q$ ,let $s = \parallel o,q\parallel$ . Due to the stability of standard normal distribution $\mathcal{N}\left( {0,1}\right)$ , we have that $\left( {\overrightarrow{a} \cdot  \overrightarrow{o} - \overrightarrow{a} \cdot  \overrightarrow{q}}\right)$ is distributed as ${sX}$ ,where $X$ is a random variable drawn from $\mathcal{N}\left( {0,1}\right)$ [2]. Let $\varphi \left( x\right)$ be the probability density function (PDF) of $\mathcal{N}\left( {0,1}\right)$ ,i.e., $\varphi \left( x\right)  = \frac{1}{\sqrt{2\pi }}{e}^{-\frac{{x}^{2}}{2}}$ . The collision probability between $o$ and $q$ under ${h}_{\overrightarrow{a}}$ is computed as follows:

我们现在证明，与查询感知桶划分相结合的哈希函数族 ${h}_{\overrightarrow{a}}\left( o\right)$ 具有局部敏感性。从这个意义上说，该族中的每个 ${h}_{\overrightarrow{a}}\left( o\right)$ 都被称为查询感知的局部敏感哈希（LSH）函数。对于对象 $o$ 和 $q$，设 $s = \parallel o,q\parallel$。由于标准正态分布 $\mathcal{N}\left( {0,1}\right)$ 的稳定性，我们可知 $\left( {\overrightarrow{a} \cdot  \overrightarrow{o} - \overrightarrow{a} \cdot  \overrightarrow{q}}\right)$ 服从 ${sX}$ 分布，其中 $X$ 是从 $\mathcal{N}\left( {0,1}\right)$ 中抽取的随机变量 [2]。设 $\varphi \left( x\right)$ 为 $\mathcal{N}\left( {0,1}\right)$ 的概率密度函数（PDF），即 $\varphi \left( x\right)  = \frac{1}{\sqrt{2\pi }}{e}^{-\frac{{x}^{2}}{2}}$。$o$ 和 $q$ 在 ${h}_{\overrightarrow{a}}$ 下的碰撞概率计算如下：

$$
p\left( s\right)  = P{r}_{\overrightarrow{a}}\left\lbrack  {\left| {{h}_{\overrightarrow{a}}\left( o\right)  - {h}_{\overrightarrow{a}}\left( q\right) }\right|  \leq  \frac{w}{2}}\right\rbrack   = {Pr}\left\lbrack  {\left| {sX}\right|  \leq  \frac{w}{2}}\right\rbrack  
$$

$$
 = \Pr \left\lbrack  {-\frac{w}{2s} \leq  X \leq  \frac{w}{2s}}\right\rbrack   = {\int }_{-\frac{w}{2s}}^{\frac{w}{2s}}\varphi \left( x\right) {dx} \tag{4}
$$

Accordingly, we have Lemma 2 as follows:

因此，我们有如下引理2：

LEMMA 2. The query-aware hash family of all the hash functions ${h}_{\overrightarrow{a}}\left( o\right)$ that are identified by Equation 3 and coupled with query-aware bucket partition is $\left( {1,c,{p}_{1},{p}_{2}}\right)$ -sensitive, where ${p}_{1} = p\left( 1\right)$ and ${p}_{2} = p\left( c\right)$ .

引理2. 由方程3确定并与查询感知桶划分相结合的所有哈希函数${h}_{\overrightarrow{a}}\left( o\right)$的查询感知哈希族是$\left( {1,c,{p}_{1},{p}_{2}}\right)$敏感的，其中${p}_{1} = p\left( 1\right)$且${p}_{2} = p\left( c\right)$。

Proof. Referring to Equation 4, a simple calculation shows that $p\left( s\right)  = 1 - 2\operatorname{norm}\left( {-\frac{w}{2s}}\right)$ ,where $\operatorname{norm}\left( x\right)  =$ ${\int }_{-\infty }^{x}\varphi \left( t\right) {dt}$ . Note that $\operatorname{norm}\left( x\right)$ is simply the cumulative distribution function (CDF) of $\mathcal{N}\left( {0,1}\right)$ ,which increases monotonically as $x$ increases. For a fixed $w,\operatorname{norm}\left( {-\frac{w}{2s}}\right)$ increases monotonically as $s$ increases,and hence $p\left( s\right)$ decreases monotonically as $s$ increases. Therefore,according to Definition 1, the query-aware hash family identified by Equation 3,is $\left( {1,c,{p}_{1},{p}_{2}}\right)$ -sensitive,where ${p}_{1} = p\left( 1\right)$ and ${p}_{2} = p\left( c\right)$ ,respectively.

证明。参考公式4，简单计算可得$p\left( s\right)  = 1 - 2\operatorname{norm}\left( {-\frac{w}{2s}}\right)$ ，其中$\operatorname{norm}\left( x\right)  =$ ${\int }_{-\infty }^{x}\varphi \left( t\right) {dt}$ 。注意，$\operatorname{norm}\left( x\right)$ 正是$\mathcal{N}\left( {0,1}\right)$ 的累积分布函数（CDF），它会随着$x$ 的增大而单调递增。对于固定的$w,\operatorname{norm}\left( {-\frac{w}{2s}}\right)$ ，其会随着$s$ 的增大而单调递增，因此$p\left( s\right)$ 会随着$s$ 的增大而单调递减。所以，根据定义1，由公式3确定的查询感知哈希族是$\left( {1,c,{p}_{1},{p}_{2}}\right)$ -敏感的，其中${p}_{1} = p\left( 1\right)$ 和${p}_{2} = p\left( c\right)$ 分别为相应参数。

### 3.2 Comparison of Colliding Probabilities

### 3.2 碰撞概率比较

The effectiveness of an $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -sensitive hash family depends on the difference between the positively-colliding probability and negatively-colliding probability,i.e., $\left( {{p}_{1} - }\right.$ ${p}_{2}$ ),since the difference measures the degree that positively-colliding data objects of a query $q$ can be discriminated from negatively-colliding ones. We now show that the novel query-aware hash family leads to larger $\left( {{p}_{1} - {p}_{2}}\right)$ under typical settings of bucket width $w$ . For query-aware LSH family, from the proof of Lemma 2,we have ${p}_{1} = 1 - \operatorname{2norm}\left( {-\frac{w}{2}}\right)$ and ${p}_{2} = 1 - 2\operatorname{norm}\left( {-\frac{w}{2c}}\right)$ . For query-oblivious LSH family,we have ${\xi }_{1} = 1 - \operatorname{2norm}\left( {-w}\right)  - \frac{2}{\sqrt{2\pi }w}\left( {1 - {e}^{-\left( {{w}^{2}/2}\right) }}\right)$ and ${\xi }_{2} = 1 - 2\operatorname{norm}\left( {-w/c}\right)  - \frac{2}{\sqrt{2\pi }w/c}\left( {1 - {e}^{-\left( {{w}^{2}/2{c}^{2}}\right) }}\right) \left\lbrack  2\right\rbrack$ .

一个$\left( {r,{cr},{p}_{1},{p}_{2}}\right)$敏感哈希族的有效性取决于正碰撞概率和负碰撞概率之间的差异，即$\left( {{p}_{1} - }\right.$ ${p}_{2}$ ），因为该差异衡量了查询$q$的正碰撞数据对象与负碰撞数据对象的可区分程度。我们现在证明，在桶宽度$w$的典型设置下，新颖的查询感知哈希族会导致更大的$\left( {{p}_{1} - {p}_{2}}\right)$。对于查询感知的局部敏感哈希（LSH）族，根据引理2的证明，我们有${p}_{1} = 1 - \operatorname{2norm}\left( {-\frac{w}{2}}\right)$和${p}_{2} = 1 - 2\operatorname{norm}\left( {-\frac{w}{2c}}\right)$。对于查询无关的LSH族，我们有${\xi }_{1} = 1 - \operatorname{2norm}\left( {-w}\right)  - \frac{2}{\sqrt{2\pi }w}\left( {1 - {e}^{-\left( {{w}^{2}/2}\right) }}\right)$和${\xi }_{2} = 1 - 2\operatorname{norm}\left( {-w/c}\right)  - \frac{2}{\sqrt{2\pi }w/c}\left( {1 - {e}^{-\left( {{w}^{2}/2{c}^{2}}\right) }}\right) \left\lbrack  2\right\rbrack$。

Bucket width $w$ is a critical parameter of an LSH function. While E2LSH and LSB-Forest manually set $w = {4.0},\mathrm{C}2\mathrm{{LSH}}$ manually sets $w = {1.0}$ . For $w$ in the range $\left\lbrack  {0,{10}}\right\rbrack$ ,starting from 0.5 and with a step of 0.5 , we show the variations of the colliding probabilities ${p}_{1},{p}_{2},{\xi }_{1}$ ,and ${\xi }_{2}$ for two different $c$ values in Figure 3. We find that all the colliding probabilities monotonically increase as $w$ increases,and get very close to 1 as $w$ gets close to 10 . In addition, ${p}_{1}$ and ${p}_{2}$ are consistently larger than ${\xi }_{1}$ and ${\xi }_{2}$ ,respectively. Thus,we also show the two differences $\left( {{p}_{1} - {p}_{2}}\right)$ and $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ with respect to $w$ in Figure 4. We have two interesting observations: (1) $\left( {{p}_{1} - {p}_{2}}\right)$ is larger than $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ under typical bucket widths, namely $w = {4.0}$ and $w = {1.0}$ . (2) Both $\left( {{p}_{1} - {p}_{2}}\right)$ and $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ tend to have maximum values in the $w$ range $\left\lbrack  {0,{10}}\right\rbrack$ . Observation (1) indicates that our novel query-aware LSH family can be used to improve the performance of query-oblivious LSH schemes such as C2LSH by leveraging a larger $\left( {{p}_{1} - {p}_{2}}\right)$ . Observation (2) inspires us to automatically set bucket width $w$ by maximizing the difference $\left( {{p}_{1} - {p}_{2}}\right)$ ,which actually leads to the minimization of the number of hash tables in QALSH as analyzed in Section 5.3.1.

桶宽度 $w$ 是局部敏感哈希（LSH）函数的一个关键参数。E2LSH 和 LSB - 森林手动设置 $w = {4.0},\mathrm{C}2\mathrm{{LSH}}$ 手动设置 $w = {1.0}$ 。对于范围在 $\left\lbrack  {0,{10}}\right\rbrack$ 内的 $w$ ，从 0.5 开始，步长为 0.5 ，我们在图 3 中展示了两种不同 $c$ 值下碰撞概率 ${p}_{1},{p}_{2},{\xi }_{1}$ 和 ${\xi }_{2}$ 的变化情况。我们发现，所有碰撞概率都随着 $w$ 的增大而单调增加，并且当 $w$ 接近 10 时，非常接近 1 。此外，${p}_{1}$ 和 ${p}_{2}$ 分别始终大于 ${\xi }_{1}$ 和 ${\xi }_{2}$ 。因此，我们在图 4 中还展示了关于 $w$ 的两个差值 $\left( {{p}_{1} - {p}_{2}}\right)$ 和 $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ 。我们有两个有趣的观察结果：（1）在典型的桶宽度，即 $w = {4.0}$ 和 $w = {1.0}$ 下，$\left( {{p}_{1} - {p}_{2}}\right)$ 大于 $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ 。（2）$\left( {{p}_{1} - {p}_{2}}\right)$ 和 $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ 都倾向于在 $w$ 范围 $\left\lbrack  {0,{10}}\right\rbrack$ 内取得最大值。观察结果（1）表明，我们新颖的查询感知 LSH 族可以通过利用更大的 $\left( {{p}_{1} - {p}_{2}}\right)$ 来提高诸如 C2LSH 等查询无关 LSH 方案的性能。观察结果（2）启发我们通过最大化差值 $\left( {{p}_{1} - {p}_{2}}\right)$ 来自动设置桶宽度 $w$ ，实际上，如第 5.3.1 节所分析的，这会导致查询感知局部敏感哈希（QALSH）中哈希表数量的最小化。

<!-- Media -->

<!-- figureText: (a) $\mathrm{c} = {2.0}$ -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_3.jpg?x=161&y=168&w=690&h=252&r=0"/>

Figure 3: Positively-colliding probability and negatively-colliding probability

图3：正碰撞概率和负碰撞概率

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_3.jpg?x=160&y=561&w=690&h=250&r=0"/>

Figure 4: Difference between positively-colliding probability and negatively-colliding probability

图4：正碰撞概率与负碰撞概率之差

<!-- Media -->

Since C2LSH has been shown to outperform LSB-Forest in high-dimensional space, and the query-aware bucket partition is easy to implement for a single query-aware LSH function, in this paper we propose to follow C2LSH's general framework to demonstrate the desirability of query-aware LSH families.

由于已有研究表明，在高维空间中C2LSH（基于碰撞的局部敏感哈希）的性能优于LSB - Forest（最低有效位森林），并且对于单个查询感知的局部敏感哈希（LSH）函数而言，查询感知的桶划分易于实现，因此在本文中，我们建议遵循C2LSH的总体框架来证明查询感知的LSH族的可行性。

### 3.3 Virtual Rehashing

### 3.3 虚拟重哈希

LSH schemes such as C2LSH do not solve the $c$ -ANN search problem directly. This is because an $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ - sensitive LSH family requires $R$ to be pre-specified so as to compute ${p}_{1}$ and ${p}_{2}$ . It is the decision version of the $c$ -ANN search problem,i.e.,the(R,c)-NN search problem,that can be directly solved by exploiting an $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -sensitive LSH family. Given a query object $q$ and a search radius $R$ ,the(R,c)-NN search problem is to find a data object ${o}_{1}$ whose distance to $q$ is at most ${cR}$ if there exists a data object ${o}_{2}$ whose distance to $q$ is at most $R$ .

像C2LSH这样的局部敏感哈希（LSH）方案并不能直接解决$c$ -近似最近邻（ANN）搜索问题。这是因为一个$\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -敏感的LSH族需要预先指定$R$ ，以便计算${p}_{1}$ 和${p}_{2}$ 。正是$c$ -ANN搜索问题的判定版本，即(R,c)-最近邻（NN）搜索问题，可以通过利用一个$\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -敏感的LSH族直接解决。给定一个查询对象$q$ 和一个搜索半径$R$ ，(R,c)-NN搜索问题是：如果存在一个数据对象${o}_{2}$ 到$q$ 的距离至多为$R$ ，则找到一个数据对象${o}_{1}$ ，其到$q$ 的距离至多为${cR}$ 。

The $c$ -ANN search of a query $q$ is reduced to a series of the (R,c)-NN search of $q$ with properly increasing search radius $R \in  \left\{  {1,c,{c}^{2},{c}^{3},\ldots }\right\}$ . Therefore,for each $R$ ,we need an $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -sensitive LSH family. For each $R \in  \left\{  {c,{c}^{2},\ldots }\right\}$ , by deriving an $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -sensitive hash family from the $\left( {1,c,{p}_{1},{p}_{2}}\right)$ -sensitive hash family for $R = 1$ ,hash tables for all the subsequent radii can be virtually imposed on the physical hash tables for $R = 1$ . This is the underlying idea of virtual rehashing of C2LSH.

查询 $q$ 的 $c$ -ANN搜索被简化为一系列对 $q$ 进行的 (R,c) -NN搜索，其中搜索半径 $R \in  \left\{  {1,c,{c}^{2},{c}^{3},\ldots }\right\}$ 会适当增大。因此，对于每个 $R$ ，我们需要一个 $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -敏感的LSH（局部敏感哈希，Locality-Sensitive Hashing）族。对于每个 $R \in  \left\{  {c,{c}^{2},\ldots }\right\}$ ，通过从用于 $R = 1$ 的 $\left( {1,c,{p}_{1},{p}_{2}}\right)$ -敏感哈希族中推导出一个 $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -敏感哈希族，所有后续半径的哈希表实际上可以叠加在用于 $R = 1$ 的物理哈希表上。这就是C2LSH（一种局部敏感哈希算法）虚拟重哈希的基本思想。

We now show that QALSH can also do virtual rehashing by deriving $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -sensitive functions from Equation 3 Virtual rehashing of QALSH enables it to work with any $c > 1$ ,while both C2LSH and LSB-Forest only work with integer $c \geq  2$ . A formal proof of this advantage is given in Section 5.1

我们现在证明，QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）还可以通过从方程3推导$\left( {R,{cR},{p}_{1},{p}_{2}}\right)$敏感函数来进行虚拟重哈希。QALSH的虚拟重哈希使其能够处理任意的$c > 1$，而C2LSH（一种局部敏感哈希算法）和LSB - Forest（最低有效位森林）仅能处理整数$c \geq  2$。第5.1节给出了这一优势的正式证明。

Proposition 1. The query-aware hash family

命题1. 查询感知哈希族

$$
{H}_{\overrightarrow{a}}^{R}\left( o\right)  = \frac{{h}_{\overrightarrow{a}}\left( o\right) }{R}
$$

is $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -sensitive, $\underline{w}$ here $c,{p}_{1},{p}_{2}$ and ${h}_{\overrightarrow{a}}\left( \cdot \right)$ are the same as defined in Lemma 2,and $R$ is a power of $c$ (i.e., ${c}^{k}$ for some integer $k \geq  1$ ).

是$\left( {R,{cR},{p}_{1},{p}_{2}}\right)$敏感的，这里$\underline{w}$，$c,{p}_{1},{p}_{2}$和${h}_{\overrightarrow{a}}\left( \cdot \right)$与引理2中的定义相同，并且$R$是$c$的幂（即，对于某个整数$k \geq  1$，有${c}^{k}$）。

Proof. Let $\overrightarrow{{o}^{\prime }} = \frac{\overrightarrow{o}}{R}$ ,from Equation 3,we have ${H}_{\overrightarrow{a}}^{R}\left( o\right)  =$ $\frac{\overrightarrow{a} \cdot  \overrightarrow{o}}{R} = \overrightarrow{a} \cdot  \overrightarrow{{o}^{\prime }} = {h}_{\overrightarrow{a}}\left( {o}^{\prime }\right)$ . By Lemma 2,we assert that ${h}_{\overrightarrow{a}}\left( {o}^{\prime }\right)$ is $\left( {1,c,{p}_{1},{p}_{2}}\right)$ -sensitive. The assertion implies,objects ${o}_{1}^{\prime }$ and ${o}_{2}^{\prime }$ collide under ${h}_{\overrightarrow{a}}\left( \cdot \right)$ with a probability at least ${p}_{1}$ if $\begin{Vmatrix}{{o}_{1}^{\prime },{o}_{2}^{\prime }}\end{Vmatrix} \leq  1.\;\begin{Vmatrix}{{o}_{1}^{\prime },{o}_{2}^{\prime }}\end{Vmatrix} \leq  1$ is equivalent to $\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix} \leq  R$ . Thus,it follows that ${o}_{1}$ and ${o}_{2}$ collide with a probability at least ${p}_{1}$ under ${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$ if $\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix} \leq  R$ . Similarly,the assertion also implies, ${o}_{1}$ and ${o}_{2}$ collide under ${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$ with a probability at most ${p}_{2}$ if $\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix} \geq  {cR}$ . Therefore,the query-aware hash family ${H}_{\overrightarrow{a}}^{R}\left( o\right)$ is $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -sensitive.

证明。设$\overrightarrow{{o}^{\prime }} = \frac{\overrightarrow{o}}{R}$，由方程3，我们有${H}_{\overrightarrow{a}}^{R}\left( o\right)  =$ $\frac{\overrightarrow{a} \cdot  \overrightarrow{o}}{R} = \overrightarrow{a} \cdot  \overrightarrow{{o}^{\prime }} = {h}_{\overrightarrow{a}}\left( {o}^{\prime }\right)$。根据引理2，我们断言${h}_{\overrightarrow{a}}\left( {o}^{\prime }\right)$是$\left( {1,c,{p}_{1},{p}_{2}}\right)$ -敏感的。该断言意味着，如果$\begin{Vmatrix}{{o}_{1}^{\prime },{o}_{2}^{\prime }}\end{Vmatrix} \leq  1.\;\begin{Vmatrix}{{o}_{1}^{\prime },{o}_{2}^{\prime }}\end{Vmatrix} \leq  1$等同于$\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix} \leq  R$，则对象${o}_{1}^{\prime }$和${o}_{2}^{\prime }$在${h}_{\overrightarrow{a}}\left( \cdot \right)$下发生碰撞的概率至少为${p}_{1}$。因此，如果$\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix} \leq  R$成立，那么${o}_{1}$和${o}_{2}$在${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$下发生碰撞的概率至少为${p}_{1}$。类似地，该断言还意味着，如果$\begin{Vmatrix}{{o}_{1},{o}_{2}}\end{Vmatrix} \geq  {cR}$成立，那么${o}_{1}$和${o}_{2}$在${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$下发生碰撞的概率至多为${p}_{2}$。因此，查询感知哈希族${H}_{\overrightarrow{a}}^{R}\left( o\right)$是$\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ -敏感的。

Given a query $q$ and a pre-specified bucket width $w$ ,for $R \in  \left\{  {1,c,{c}^{2},\ldots }\right\}$ ,we now define the round- $R$ anchor bucket ${B}^{R}$ as the anchor bucket with width $w$ defined by ${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$ , i.e.,the interval $\left\lbrack  {{H}_{\overrightarrow{a}}^{R}\left( q\right)  - \frac{w}{2},{H}_{\overrightarrow{a}}^{R}\left( q\right)  + \frac{w}{2}}\right\rbrack$ ,which is centered at $q$ ’s projection ${H}_{\overrightarrow{a}}^{R}\left( q\right)$ along the random line $\overrightarrow{a}$ . In other words,the round- $R$ anchor bucket is located by query-aware bucket partition with bucket width $w$ as before. Specifically, ${B}^{1}$ is simply the interval $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ ,which is the anchor bucket with width $w$ defined by ${h}_{\overrightarrow{a}}\left( \cdot \right)$ .

给定一个查询 $q$ 和一个预先指定的桶宽度 $w$，对于 $R \in  \left\{  {1,c,{c}^{2},\ldots }\right\}$，我们现在将第 $R$ 轮锚定桶 ${B}^{R}$ 定义为宽度为 $w$ 的锚定桶，由 ${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$ 定义，即区间 $\left\lbrack  {{H}_{\overrightarrow{a}}^{R}\left( q\right)  - \frac{w}{2},{H}_{\overrightarrow{a}}^{R}\left( q\right)  + \frac{w}{2}}\right\rbrack$，该区间以 $q$ 沿随机直线 $\overrightarrow{a}$ 的投影 ${H}_{\overrightarrow{a}}^{R}\left( q\right)$ 为中心。换句话说，第 $R$ 轮锚定桶通过与之前相同的、桶宽度为 $w$ 的查询感知桶划分来定位。具体而言，${B}^{1}$ 就是区间 $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$，它是由 ${h}_{\overrightarrow{a}}\left( \cdot \right)$ 定义的宽度为 $w$ 的锚定桶。

As shown in Section 4.1,to find the(R,c)-NN of a query $q$ ,we only need to check the round- $R$ anchor bucket ${B}^{R}$ for the specific $R$ . To find the $c$ -ANN of $q$ ,we check the round- $R$ anchor buckets round by round for gradually increasing $R\left( {R \in  \left\{  {1,c,{c}^{2},\ldots }\right\}  }\right)$ . All the round- $R$ anchor buckets defined by ${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$ are centered at $q$ along the same $\overrightarrow{a}$ ,and can be located along $\overrightarrow{a}$ with properly adjusted bucket width. Therefore, we only need to keep one physical copy of the data projections along $\overrightarrow{a}$ . Using the results of the following Propositions 2 and 3,we can virtually impose ${B}^{R}$ over ${B}^{1}$ , and hence ${B}^{cR}$ over ${B}^{R}$ . This is the underlying idea of virtual rehashing of QALSH. Here we only show the proof of Proposition 2 since the proof of Proposition 3 is similar.

如4.1节所示，要找到查询$q$的(R,c)-最近邻（(R,c)-NN），我们只需检查特定$R$对应的第$R$轮锚定桶${B}^{R}$。要找到$q$的$c$-近似最近邻（$c$-ANN），我们逐轮检查第$R$轮锚定桶，同时逐渐增大$R\left( {R \in  \left\{  {1,c,{c}^{2},\ldots }\right\}  }\right)$。由${H}_{\overrightarrow{a}}^{R}\left( \cdot \right)$定义的所有第$R$轮锚定桶都沿着相同的$\overrightarrow{a}$以$q$为中心，并且可以沿着$\overrightarrow{a}$通过适当调整桶宽度来定位。因此，我们只需要保留沿着$\overrightarrow{a}$的数据投影的一个物理副本。利用下面命题2和命题3的结果，我们可以在${B}^{1}$上虚拟地施加${B}^{R}$，进而在${B}^{R}$上施加${B}^{cR}$。这就是QALSH（快速近似最近邻搜索哈希算法）虚拟重哈希的基本思想。这里我们只展示命题2的证明，因为命题3的证明与之类似。

Proposition 2. Given $q$ and $w,{B}^{R}$ contains ${B}^{1}$ ,and the width of ${B}^{R}$ is $R$ times the width of ${B}^{1}$ ,i.e., ${wR}$ .

命题2。给定$q$且$w,{B}^{R}$包含${B}^{1}$，并且${B}^{R}$的宽度是${B}^{1}$宽度的$R$倍，即${wR}$。

Proof. According to the definition of ${B}^{R}$ ,for each object $o$ in ${B}^{R}$ ,we have $\left| {{H}_{\overrightarrow{a}}^{R}\left( o\right)  - {H}_{\overrightarrow{a}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}$ ,i.e., $\left| {\frac{{h}_{\overrightarrow{a}}\left( o\right) }{R} - \frac{{h}_{\overrightarrow{a}}\left( q\right) }{R}}\right|  \leq$ $\frac{w}{2}$ . Hence,we have $\left| {{h}_{\overrightarrow{a}}\left( o\right)  - {h}_{\overrightarrow{a}}\left( q\right) }\right|  \leq  \frac{wR}{2}$ ,which means that $o$ falls into the interval $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{wR}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{wR}{2}}\right\rbrack$ along the random line $\overrightarrow{a}$ . This interval is simply the anchor bucket with width ${wR}$ defined by ${h}_{\overrightarrow{a}}\left( \cdot \right)$ ,which obviously contains the sub-interval $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ ,i.e., ${B}^{1}$ . And the width of ${B}^{R}$ is $R$ times the width of ${B}^{1}$

证明。根据${B}^{R}$的定义，对于${B}^{R}$中的每个对象$o$，我们有$\left| {{H}_{\overrightarrow{a}}^{R}\left( o\right)  - {H}_{\overrightarrow{a}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}$，即$\left| {\frac{{h}_{\overrightarrow{a}}\left( o\right) }{R} - \frac{{h}_{\overrightarrow{a}}\left( q\right) }{R}}\right|  \leq$$\frac{w}{2}$。因此，我们有$\left| {{h}_{\overrightarrow{a}}\left( o\right)  - {h}_{\overrightarrow{a}}\left( q\right) }\right|  \leq  \frac{wR}{2}$，这意味着$o$沿着随机线$\overrightarrow{a}$落入区间$\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{wR}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{wR}{2}}\right\rbrack$。这个区间就是由${h}_{\overrightarrow{a}}\left( \cdot \right)$定义的宽度为${wR}$的锚桶，它显然包含子区间$\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$，即${B}^{1}$。并且${B}^{R}$的宽度是${B}^{1}$宽度的$R$倍

Proposition 3. Given $q$ and $w,{B}^{cR}$ contains ${B}^{R}$ ,and the width of ${B}^{cR}$ is $c$ times the width of ${B}^{R}$ .

命题3。给定$q$且$w,{B}^{cR}$包含${B}^{R}$，并且${B}^{cR}$的宽度是${B}^{R}$宽度的$c$倍。

Referring to Figure 5,on the random line $\overrightarrow{{a}_{1}}$ ,the interval of width $w$ centered at $q$ is ${B}^{1}$ ,which is indicated by "00". ${B}^{2}$ and ${B}^{4}$ are indicated by "1001" and "32100123", respectively. Virtual rehashing of QALSH is equal to symmetrically searching half-buckets of length $\frac{w}{2}$ one by one on both sides of $q$ . A detailed example is given in Section 4.2

参考图5，在随机直线$\overrightarrow{{a}_{1}}$上，以$q$为中心、宽度为$w$的区间是${B}^{1}$，用“00”表示。${B}^{2}$和${B}^{4}$分别用“1001”和“32100123”表示。QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）的虚拟重哈希等同于在$q$的两侧逐个对称地搜索长度为$\frac{w}{2}$的半桶。第4.2节给出了一个详细示例。

<!-- Media -->

<!-- figureText: $\overrightarrow{{a}_{1}} = \frac{{v}_{0}{\gamma }_{0}{v}_{1}{\gamma }_{1}{v}_{2}{\gamma }_{2}{v}_{3}{\gamma }_{1}{v}_{2}{\gamma }_{3}}{3\;2\;1\;0\;0\;1\;2\;3}$ $\overrightarrow{{a}_{2}} = \overset{ \parallel  }{\overbrace{3\;2}}\overset{{0}_{1} \downarrow  }{\overbrace{1\;0}}\overset{{0}_{2} \uparrow  }{\overbrace{0\;1}}\overset{{0}_{3} \uparrow  }{\overbrace{0\;1}}\overset{{0}_{4} \uparrow  }{\overbrace{0\;1}}\overset{{0}_{3}}{\overbrace{0\;1}}\overset{{0}_{4}}{\overbrace{0\;1}}\overset{{0}_{5}}{\overbrace{0\;1}}\overset{{0}_{6}}{\overbrace{0\;1}}.$ -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_4.jpg?x=199&y=150&w=609&h=215&r=0"/>

Figure 5: Virtual Rehashing of QALSH for $c = 2$

图5：$c = 2$的QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）虚拟重哈希

<!-- Media -->

### 3.4 Preparing for Bucket Partition

### 3.4 准备桶划分

Given a query-aware LSH function ${h}_{\overrightarrow{a}}$ ,to perform virtual rehashing along $\overrightarrow{a}$ ,we need to quickly locate a series of anchor buckets via query-aware bucket partition. Therefore, in the pre-processing step,we prepare the hash table $T$ of ${h}_{\overrightarrow{a}}.T$ is a list of the pairs $\left( {{h}_{\overrightarrow{a}}\left( o\right) ,I{D}_{o}}\right)$ for each object $o$ in the database $D$ ,where $I{D}_{o}$ is the object id referring to $o$ . The list is sorted in ascending order of ${h}_{\overrightarrow{a}}\left( o\right)$ ,and is then indexed by a ${B}^{ + }$ -tree.

给定一个查询感知的局部敏感哈希（LSH）函数${h}_{\overrightarrow{a}}$，为了沿着$\overrightarrow{a}$执行虚拟重哈希，我们需要通过查询感知的桶划分快速定位一系列锚桶。因此，在预处理步骤中，我们准备${h}_{\overrightarrow{a}}.T$的哈希表$T$，对于数据库$D$中的每个对象$o$，$T$是成对元素$\left( {{h}_{\overrightarrow{a}}\left( o\right) ,I{D}_{o}}\right)$的列表，其中$I{D}_{o}$是指向$o$的对象ID。该列表按${h}_{\overrightarrow{a}}\left( o\right)$的升序排序，然后通过${B}^{ + }$树进行索引。

Given a pre-specified bucket width $w$ . When a query $q$ arrives,to conduct an(R,c)-NN search,we perform a range search $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{wR}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{wR}{2}}\right\rbrack$ to locate the round- $R$ anchor bucket using the ${B}^{ + }$ -tree over the hash table $T$ . To conduct an $c$ -ANN search,we first perform a range search $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ to locate the round-1 anchor bucket ${B}^{1}$ . Then we use virtual rehashing to check both sides of ${B}^{1}$ to locate the round- $R$ anchor buckets in need. In this manner, we can implement query-aware bucket partition quickly, without physically partitioning the whole random line into buckets of width $w$ .

给定一个预先指定的桶宽度 $w$。当查询 $q$ 到达时，为了进行 (R,c)-最近邻（(R,c)-NN）搜索，我们使用哈希表 $T$ 上的 ${B}^{ + }$ -树执行范围搜索 $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{wR}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{wR}{2}}\right\rbrack$ 来定位第 $R$ 轮锚定桶。为了进行 $c$ -近似最近邻（$c$ -ANN）搜索，我们首先执行范围搜索 $\left\lbrack  {{h}_{\overrightarrow{a}}\left( q\right)  - \frac{w}{2},{h}_{\overrightarrow{a}}\left( q\right)  + \frac{w}{2}}\right\rbrack$ 来定位第 1 轮锚定桶 ${B}^{1}$。然后我们使用虚拟重新哈希来检查 ${B}^{1}$ 的两侧，以定位所需的第 $R$ 轮锚定桶。通过这种方式，我们可以快速实现查询感知的桶划分，而无需将整个随机线物理划分为宽度为 $w$ 的桶。

Essentially, a hash table of QALSH can be viewed as a Secondary ${B}^{ + }$ -tree,which enables QALSH to support updates and to enhance the performance of relational databases.

本质上，QALSH（查询感知局部敏感哈希）的哈希表可以看作是一个二级 ${B}^{ + }$ -树，这使得QALSH能够支持更新操作并提升关系型数据库的性能。

## 4. QUERY-AWARE LSH SCHEME

## 4. 查询感知局部敏感哈希方案

Given a query-aware LSH function $h$ ,if a data object $o$ is close to a query $q$ in the original Euclidean space,then it is very likely they will collide in the anchor bucket with width $w$ defined by $h$ . However,under a specific function, they may not collide at all. Therefore, QALSH exploits a collection of $m$ independent query-aware LSH functions to achieve quality guarantee. A good candidate $o$ for query answers is expected to collide with $q$ frequently under the $m$ functions. QALSH identifies final query answers from a collection of such candidates.

给定一个查询感知局部敏感哈希函数 $h$ ，如果一个数据对象 $o$ 在原始欧几里得空间中接近一个查询 $q$ ，那么它们很可能会在由 $h$ 定义的宽度为 $w$ 的锚桶中发生碰撞。然而，在特定函数下，它们可能根本不会发生碰撞。因此，QALSH利用一组 $m$ 个独立的查询感知局部敏感哈希函数来保证查询质量。对于查询答案而言，一个合适的候选对象 $o$ 有望在 $m$ 个函数下频繁地与 $q$ 发生碰撞。QALSH从这样的候选对象集合中确定最终的查询答案。

### 4.1 QALSH for (R, c)-NN Search

### 4.1 用于 (R, c)-最近邻搜索的QALSH

QALSH directly solves the(R,c)-NN problem by exploiting a base $\mathcal{B}$ of $m$ query-aware LSH functions $\left\{  {{H}_{\overrightarrow{{a}_{1}}}^{R}\left( \cdot \right) ,{H}_{\overrightarrow{{a}_{2}}}^{R}\left( \cdot \right) }\right.$ , $\left. {\ldots ,{H}_{\overrightarrow{{a}_{m}}}^{R}\left( \cdot \right) }\right\}$ . Those LSH functions are mutually independent,and are uniformly selected from an $\left( {R,{cR},{p}_{1},{p}_{2}}\right)$ - sensitive query-aware LSH family. For each ${H}_{\overrightarrow{{a}_{i}}}^{R}\left( \cdot \right)$ ,we build a hash table ${T}_{i}$ which is indexed by a ${B}^{ + }$ -tree,as described in Section 3.4.

QALSH（查询感知局部敏感哈希）通过利用由$m$个查询感知局部敏感哈希（LSH）函数$\left\{  {{H}_{\overrightarrow{{a}_{1}}}^{R}\left( \cdot \right) ,{H}_{\overrightarrow{{a}_{2}}}^{R}\left( \cdot \right) }\right.$、$\left. {\ldots ,{H}_{\overrightarrow{{a}_{m}}}^{R}\left( \cdot \right) }\right\}$构成的基$\mathcal{B}$，直接解决了(R,c)-最近邻（NN）问题。这些LSH函数相互独立，并且是从一个$\left( {R,{cR},{p}_{1},{p}_{2}}\right)$敏感的查询感知LSH族中均匀选取的。对于每个${H}_{\overrightarrow{{a}_{i}}}^{R}\left( \cdot \right)$，我们构建一个哈希表${T}_{i}$，该哈希表由一个${B}^{ + }$树进行索引，如3.4节所述。

To find the(R,c)-NN of a query $q$ ,we first compute the hash values ${H}_{\overrightarrow{{a}_{i}}}^{R}\left( q\right)$ for $i = 1,2,\ldots ,m$ ,and then use the ${B}^{ + }$ - trees over ${T}_{i}$ s to locate the $m$ round- $R$ anchor buckets. For each object $o$ that appears in some of the $m$ anchor buckets, we collect its collision number $\# \operatorname{Col}\left( o\right)$ ,which is formally defined as follows:

为了找到查询 $q$ 的 (R,c)-近邻（(R,c)-NN），我们首先计算 $i = 1,2,\ldots ,m$ 的哈希值 ${H}_{\overrightarrow{{a}_{i}}}^{R}\left( q\right)$，然后使用基于 ${T}_{i}$ 的 ${B}^{ + }$ 树来定位第 $m$ 轮 $R$ 锚桶。对于出现在某些 $m$ 锚桶中的每个对象 $o$，我们收集其碰撞数 $\# \operatorname{Col}\left( o\right)$，其正式定义如下：

$$
\# \operatorname{Col}\left( o\right)  = \left| \left\{  {{H}_{\overrightarrow{a}}^{R}\left| {{H}_{\overrightarrow{a}}^{R} \in  \mathcal{B} \land  }\right| {H}_{\overrightarrow{a}}^{R}\left( o\right)  - {H}_{\overrightarrow{a}}^{R}\left( q\right)  \mid   \leq  \frac{w}{2}}\right\}  \right|  \tag{5}
$$

Given a pre-specified collision threshold $l$ ,object $o$ is called frequent (with respect to $q,w$ and $\mathcal{B}$ ) if $\# \operatorname{Col}\left( o\right)  \geq  l$ . We prefer to collecting collision numbers first for objects whose projections are closer to the query projection. We only need to find the "first" ${\beta n}$ frequent objects (where $\beta$ is clarified later and $n$ is $D$ ’s cardinality) and compute the Euclidean distances to $q$ for them. If there is some frequent object whose distance to $q$ is less than or equal to ${cR}$ ,we return YES and the object; Otherwise, we return NO.

给定一个预先指定的碰撞阈值 $l$ ，若 $\# \operatorname{Col}\left( o\right)  \geq  l$ 成立，则称对象 $o$ 是频繁的（相对于 $q,w$ 和 $\mathcal{B}$ 而言）。我们倾向于先收集投影更接近查询投影的对象的碰撞数。我们只需要找到 “前” ${\beta n}$ 个频繁对象（其中 $\beta$ 稍后会明确说明， $n$ 是 $D$ 的基数），并计算它们到 $q$ 的欧几里得距离。如果存在某个频繁对象到 $q$ 的距离小于或等于 ${cR}$ ，则返回 “是” 以及该对象；否则，返回 “否”。

The base cardinality $m$ is one of the key parameters for QALSH, which need to be properly chosen so as to ensure that the following two properties hold at the same time with a constant probability:

基础基数 $m$ 是QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）的关键参数之一，需要对其进行适当选择，以确保以下两个属性能以恒定概率同时成立：

- ${\mathcal{P}}_{1}$ : If there exists an object $o$ whose distance to $q$ is within $R$ ,then $o$ is a frequent object.

- ${\mathcal{P}}_{1}$ ：如果存在一个对象 $o$，其与 $q$ 的距离在 $R$ 以内，那么 $o$ 就是一个频繁对象。

- ${\mathcal{P}}_{2}$ : The total number of false positives is less than ${\beta n}$ , where each false positive is a frequent object whose distance to $q$ is larger than ${cR}$ .

- ${\mathcal{P}}_{2}$ ：误报的总数少于 ${\beta n}$，其中每个误报都是一个频繁对象，其与 $q$ 的距离大于 ${cR}$。

The above assertion is assured by Lemma 3 as follows, which guarantees correctness of QALSH for the(R,c)-NN search. Let $l$ be the collision threshold, $\alpha$ be the collision threshold in percentage,we have $l = {\alpha m}$ . Let $\delta$ be the error probability, $\beta$ be the percentage of false positives.

上述断言由以下引理 3（Lemma 3）保证，该引理确保了 QALSH 在 (R,c)-近邻搜索（(R,c)-NN search）中的正确性。设 $l$ 为碰撞阈值，$\alpha$ 为碰撞阈值的百分比，我们有 $l = {\alpha m}$。设 $\delta$ 为错误概率，$\beta$ 为误报率。

LEMMA 3. Given ${p}_{1} = p\left( 1\right)$ and ${p}_{2} = p\left( c\right)$ ,where $p\left( \cdot \right)$ is defined by Equation 4. Let $\alpha ,\beta$ and $\delta$ be defined as above. For ${p}_{2} < \alpha  < {p}_{1},0 < \beta  < 1$ and $0 < \delta  < \frac{1}{2},{\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ hold at the same time with probability at least $\frac{1}{2} - \delta$ ,provided the base cardinality $m$ is given as below:

引理3。给定${p}_{1} = p\left( 1\right)$和${p}_{2} = p\left( c\right)$，其中$p\left( \cdot \right)$由方程4定义。设$\alpha ,\beta$和$\delta$如上所定义。对于${p}_{2} < \alpha  < {p}_{1},0 < \beta  < 1$、$0 < \delta  < \frac{1}{2},{\mathcal{P}}_{1}$和${\mathcal{P}}_{2}$同时成立的概率至少为$\frac{1}{2} - \delta$，前提是基基数$m$如下给出：

$$
m = \left\lceil  {\max \left( {\frac{1}{2{\left( {p}_{1} - \alpha \right) }^{2}}\ln \frac{1}{\delta },\frac{1}{2{\left( \alpha  - {p}_{2}\right) }^{2}}\ln \frac{2}{\beta }}\right) }\right\rceil   \tag{6}
$$

Lemma 3 is a slightly different version of Lemma 1 of $\mathrm{C}2\mathrm{{LSH}}$ in the sense that the joint probability of ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ is explicitly bounded from below. Therefore, we only give a sketch of the proof in Appendix A.

引理3是文献$\mathrm{C}2\mathrm{{LSH}}$中引理1的一个略有不同的版本，具体而言，${\mathcal{P}}_{1}$和${\mathcal{P}}_{2}$的联合概率明确地从下方有界。因此，我们仅在附录A中给出证明的概要。

### 4.2 QALSH for c-ANN Search

### 4.2 用于c - 近似最近邻（c - ANN）搜索的QALSH算法

Given a query $q$ and a pre-specified bucket width $w$ ,in order to find the $c$ -ANN of $q$ ,QALSH first collects frequent objects from round-1 anchor buckets using $R = 1$ ; if frequent objects collected so far are not enough, QALSH automatically updates $R$ ,and hence collects more frequent objects from the round- $R$ anchor buckets via virtual rehashing,and etc., until finally enough frequent objects have been found or a good enough frequent object has been identified. The $c$ -ANN of $q$ must be one of the frequent objects.

给定一个查询$q$和一个预先指定的桶宽度$w$，为了找到$q$的$c$ - 近似最近邻，QALSH算法首先使用$R = 1$从第一轮锚定桶中收集频繁对象；如果到目前为止收集的频繁对象数量不足，QALSH会自动更新$R$，从而通过虚拟重哈希从第$R$轮锚定桶中收集更多频繁对象，依此类推，直到最终找到足够多的频繁对象或识别出一个足够好的频繁对象。$q$的$c$ - 近似最近邻必定是这些频繁对象之一。

QALSH is quite straightforward, as shown in Algorithm 1. A candidate set $C$ is used to store the frequent objects found so far, and is empty at the beginning.

QALSH（快速近似局部敏感哈希）算法非常简单，如算法1所示。候选集 $C$ 用于存储目前已找到的频繁对象，初始时为空。

Terminating condition. QALSH terminates in one of the two following cases which are supported by the two properties ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ of Lemma 3 respectively:

终止条件。QALSH（快速近似局部敏感哈希）算法在以下两种情况之一终止，这两种情况分别由引理3的两个性质 ${\mathcal{P}}_{1}$ 和 ${\mathcal{P}}_{2}$ 支持：

- ${\mathcal{T}}_{1}$ : At round- $R$ ,there exists at least 1 frequent object whose Euclidean distance to $q$ is less than or equal to ${cR}$ (referring to Lines 9-11 in Algorithm 1).

- ${\mathcal{T}}_{1}$ ：在第 $R$ 轮，至少存在1个频繁对象，其与 $q$ 的欧几里得距离小于或等于 ${cR}$（参见算法1中的第9 - 11行）。

- ${\mathcal{T}}_{2}$ : At round- $R$ ,at least ${\beta n}$ frequent objects have been found (referring to Line 2 and Line 13 in Algorithm 1).

- ${\mathcal{T}}_{2}$ ：在第 $R$ 轮，至少已找到 ${\beta n}$ 个频繁对象（参见算法1中的第2行和第13行）。

<!-- Media -->

## Algorithm 1 QALSH

## 算法1 QALSH（快速近似局部敏感哈希）

---

Input:

		$c$ is the approximation ratio, $\beta$ is the percentage of false

		positives, $\delta$ is the error probability. $m$ is the number of

		hash tables, $l$ is the collision threshold.

Output:

		the nearest object ${o}_{min}$ in the set $C$ of frequent objects.

		$R = 1;C = \varnothing ;$

		while $\left| C\right|  < {\beta n}$ do

			for each $i = 1$ to $m$ do

				increase $\# \operatorname{Col}\left( o\right)$ by 1 if $o$ is found in the round- $R$

				anchor bucket,i.e., $\left| {{H}_{{\overrightarrow{a}}_{i}}^{R}\left( o\right)  - {H}_{{\overrightarrow{a}}_{i}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}$ ;

				if $\# \operatorname{Col}\left( o\right)  \geq  l$ then

					$C = C \cup  o;$

				end if

			end for

			if $\left| {\{ o \mid  o \in  C \land  \parallel o,q\parallel  \leq  c \times  R\} }\right|  \geq  1$ then

				break;

			end if

			update radius $R$ ;

		end while

		return the nearest object ${o}_{min} \in  C$ ;

---

<!-- Media -->

Update of Search Radius $R$ . It can be checked that, in Algorithm 1,if the terminating condition ${\mathcal{T}}_{1}$ is still not satisfied at the moment, i.e., we have not found a good enough frequent object,then we need to update $R$ in Line 12. For ease of reference,let $R$ and ${R}^{\prime }$ denote current and next search radius, respectively.

搜索半径 $R$ 的更新。可以验证，在算法 1 中，如果此时终止条件 ${\mathcal{T}}_{1}$ 仍未满足，即我们尚未找到足够好的频繁对象，那么我们需要在第 12 行更新 $R$。为便于引用，分别用 $R$ 和 ${R}^{\prime }$ 表示当前搜索半径和下一个搜索半径。

Since C2LSH statically set ${R}^{\prime } = c \times  R$ ,it conducts one by one a series of(R,c)-NN search with $R \in  \left\{  {1,c,{c}^{2},\ldots }\right\}$ . Actually, some round of the search could be wasteful. Given $c = 2$ and $l = 2$ ,an example can be illustrated in Figure 5 and Algorithm 1. After Algorithm 1 s first round (i.e., after the $\left( {R = 1,c = 2}\right)$ -NN search),we have $\# \operatorname{Col}\left( {o}_{2}\right)  = 1$ and $\# \operatorname{Col}\left( {o}_{3}\right)  = 1$ ,since only ${o}_{2}$ and ${o}_{3}$ respectively appears once in the two round-1 anchor buckets labeled by " 00 ". Since both ${o}_{2}$ and ${o}_{3}$ are not frequent at the moment,Algorithm 1 needs to update $R$ . Since there is no new data object in round-2 anchor buckets which are labeled by "1001", we do not need to update $R$ to be $R = 2$ (i.e., $R = c$ ) as what C2LSH chooses to do.

由于C2LSH（基于计数的局部敏感哈希）静态设置了${R}^{\prime } = c \times  R$，它会使用$R \in  \left\{  {1,c,{c}^{2},\ldots }\right\}$逐个进行一系列的(R,c)-最近邻（NN）搜索。实际上，某些轮次的搜索可能是浪费的。给定$c = 2$和$l = 2$，图5和算法1给出了一个示例。在算法1的第一轮（即$\left( {R = 1,c = 2}\right)$-最近邻搜索之后），我们得到$\# \operatorname{Col}\left( {o}_{2}\right)  = 1$和$\# \operatorname{Col}\left( {o}_{3}\right)  = 1$，因为只有${o}_{2}$和${o}_{3}$分别在标记为“00”的两个第一轮锚定桶中出现了一次。由于此时${o}_{2}$和${o}_{3}$都不频繁，算法1需要更新$R$。由于在标记为“1001”的第二轮锚定桶中没有新的数据对象，我们不需要像C2LSH选择的那样将$R$更新为$R = 2$（即$R = c$）。

In contrast, QALSH chooses to skip such wasteful rounds by leveraging the projections of data objects to properly update $R$ . Recall that each of the $m$ hash tables of QALSH is simply a ${B}^{ + }$ -tree,we can easily find the object $o$ which is closest to $q$ and exists outside of the current round- $R$ anchor bucket. Thus we have $m$ such objects in total. Suppose their distances to $q$ (in terms of projections) are sorted in ascending order and denoted as ${d}_{1},{d}_{2},\ldots$ ,and ${d}_{m}$ ,i.e., ${d}_{1}$ is the smallest and ${d}_{m}$ is the biggest. Let ${d}_{med}$ denote the median of ${d}_{1},{d}_{2},\ldots$ ,and ${d}_{m}$ . QALSH automatically set ${R}^{\prime }$ to be ${R}^{\prime } = {c}^{k}$ such that $\frac{w{R}^{\prime }}{2} \geq  {d}_{med}$ and integer $k$ is as small as possible. Therefore,there are at least $\frac{m}{2}$ objects for collecting collision number in the next round of search.

相比之下，快速近似局部敏感哈希（QALSH）方法通过利用数据对象的投影来适当地更新$R$，从而选择跳过这种浪费的轮次。回顾一下，QALSH的$m$个哈希表中的每一个都只是一棵${B}^{ + }$树，我们可以很容易地找到距离$q$最近且存在于当前第$R$轮锚定桶之外的对象$o$。因此，我们总共拥有$m$个这样的对象。假设它们到$q$的距离（就投影而言）按升序排列，并表示为${d}_{1},{d}_{2},\ldots$和${d}_{m}$，即${d}_{1}$是最小的，${d}_{m}$是最大的。令${d}_{med}$表示${d}_{1},{d}_{2},\ldots$和${d}_{m}$的中位数。QALSH自动将${R}^{\prime }$设置为${R}^{\prime } = {c}^{k}$，使得$\frac{w{R}^{\prime }}{2} \geq  {d}_{med}$且整数$k$尽可能小。因此，在下一轮搜索中，至少有$\frac{m}{2}$个对象用于收集碰撞次数。

The underlying intuition is as follows. If we set ${R}^{\prime }$ according to ${d}_{1}$ ,the round- ${R}^{\prime }$ anchor buckets may contain too few data objects for collecting collision number and hence we waste the scan of the round- ${R}^{\prime }$ anchor buckets. On the other hand,if we set ${R}^{\prime }$ according to ${d}_{m}$ ,since ${d}_{m}$ may be too large,round- ${R}^{\prime }$ anchor buckets may contain too many data objects, and hence we may do unnecessary collision number collection and Euclidean distance computation. In addition, ${R}^{\prime }$ has to be ${R}^{\prime } = {c}^{k}$ for integer $k$ ,so that the theoretical framework of QALSH is still assured.

其基本直觉如下。如果我们根据${d}_{1}$来设置${R}^{\prime }$，第${R}^{\prime }$轮锚桶可能包含的数据对象太少，无法收集碰撞数，因此我们会浪费对第${R}^{\prime }$轮锚桶的扫描。另一方面，如果我们根据${d}_{m}$来设置${R}^{\prime }$，由于${d}_{m}$可能太大，第${R}^{\prime }$轮锚桶可能包含太多的数据对象，因此我们可能会进行不必要的碰撞数收集和欧几里得距离计算。此外，对于整数$k$，${R}^{\prime }$必须为${R}^{\prime } = {c}^{k}$，这样才能保证QALSH（快速近似最近邻哈希）的理论框架仍然成立。

### 4.3 QALSH for c-k-ANN Search

### 4.3 用于c - k - 近似最近邻（c - k - ANN）搜索的QALSH（快速近似最近邻哈希）

To support the $c - k$ -ANN search,QALSH only needs to change its terminating conditions of $c$ -ANN:

为了支持$c - k$ - 近似最近邻（ANN）搜索，QALSH（快速近似最近邻哈希）只需要改变其$c$ - 近似最近邻（ANN）的终止条件：

- ${\mathcal{T}}_{1}^{\prime }$ : At round- $R$ ,there exist at least $k$ frequent objects whose Euclidean distance to $q$ is within ${cR}$ .

- ${\mathcal{T}}_{1}^{\prime }$ ：在第 $R$ 轮，至少存在 $k$ 个频繁对象，其与 $q$ 的欧几里得距离在 ${cR}$ 以内。

- ${\mathcal{T}}_{2}^{\prime }$ : At round- $R$ ,there are at least ${\beta n} + k - 1$ frequent objects that have been found.

- ${\mathcal{T}}_{2}^{\prime }$ ：在第 $R$ 轮，至少已找到 ${\beta n} + k - 1$ 个频繁对象。

## 5. THEORETICAL ANALYSIS

## 5. 理论分析

In this section, we first show that QALSH works with any approximation ratio $c > 1$ ,and then we give the bound on approximation ratio for $c$ -ANN search. Then we discuss the parameter setting of QALSH and propose an automatic way to set bucket width $w$ . Finally,we show the time and space complexity of QALSH.

在本节中，我们首先证明QALSH（量化自适应局部敏感哈希，Quantized Adaptive Locality-Sensitive Hashing）适用于任何近似比 $c > 1$ ，然后给出 $c$ -最近邻搜索（$c$ -ANN search）的近似比界限。接着，我们讨论QALSH的参数设置，并提出一种自动设置桶宽度 $w$ 的方法。最后，我们展示QALSH的时间和空间复杂度。

### 5.1 Working with Any Approximation Ratio

### 5.1 适用于任何近似比

C2LSH physically builds hash tables for search radius $R = 1$ ,where the buckets are called level-1 buckets and are statically partitioned before any query arrives. Each level-1 bucket is identified by an integer called bid. Referring to Observation 3 of C2LSH, when C2LSH performs virtual rehashing for search radius $R \in  \left\{  {c,{c}^{2},{c}^{3},\ldots }\right\}$ ,each level- $R$ bucket consists of exactly $R$ level-1 buckets identified by consecutive level-1 bids. When approximation ratio $c$ is not an integer, search radius $R$ is not an integer either,which implies some level-1 bucket must be further partitioned. However, the level-1 bucket has already been set up as the smallest granularity of bucket partition. Therefore, C2LSH only works with integer $c \geq  2$ .

C2LSH会为搜索半径$R = 1$实际构建哈希表，其中的桶被称为一级桶（level-1 buckets），并且在任何查询到来之前就进行了静态分区。每个一级桶由一个称为桶ID（bid）的整数标识。参考C2LSH的观察结果3，当C2LSH为搜索半径$R \in  \left\{  {c,{c}^{2},{c}^{3},\ldots }\right\}$执行虚拟重新哈希时，每个$R$级桶恰好由$R$个一级桶组成，这些一级桶由连续的一级桶ID标识。当近似比$c$不是整数时，搜索半径$R$也不是整数，这意味着某些一级桶必须进一步分区。然而，一级桶已经被设置为桶分区的最小粒度。因此，C2LSH仅适用于整数$c \geq  2$。

LSB-Forest suffers from the same problem of static bucket partition as C2LSH. Before $k$ -dimensional objects are converted into Z-order values, grids must be imposed on the $k$ coordinates of the $k$ -dimensional space. Each cell of the grids is equal to a bucket. A Z-order value virtually imposes grids at different levels. A high-level cell (bucket) is used for larger search radius and consists of an integral number of low-level cells (buckets) which are used for smaller search radius. Therefore, LSB-Forest also only works with integer $c \geq  2$ . Now we show:

LSB森林（LSB-Forest）与C2LSH一样存在静态桶划分的问题。在将$k$维对象转换为Z序值之前，必须在$k$维空间的$k$个坐标上划分网格。网格的每个单元格相当于一个桶。Z序值实际上在不同级别上划分了网格。高级单元格（桶）用于较大的搜索半径，它由整数个低级单元格（桶）组成，低级单元格（桶）用于较小的搜索半径。因此，LSB森林也仅适用于整数$c \geq  2$。现在我们证明：

LEMMA 4. Algorithm 1 works with any approximation ratio $c > 1$ .

引理4. 算法1适用于任何近似比$c > 1$。

Proof. As shown in Algorithm 1, only anchor buckets at different rounds are needed. Instead of using a fixed bucket id to identify a pre-partitioned bucket, all the anchor buckets at different rounds are virtually imposed by specifying a bucket range,i.e., $\left| {{H}_{\overrightarrow{a}}^{R}\left( o\right)  - {H}_{\overrightarrow{a}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}$ . We can use any bucket range to decide an anchor bucket, since the hash values, i.e., the projections, have been recorded in the hash tables. According to Proposition 3, the enlargement of a round- $R$ anchor bucket by $c$ times,which generates a round- ${cR}$ anchor bucket,is realized by enlarging the corresponding bucket range by $c$ times,where $c$ is not required to be an integer. Therefore,Algorithm 1 works with any $c > 1$ .

证明。如算法1所示，仅需要不同轮次的锚桶（anchor buckets）。我们不使用固定的桶ID来标识预划分的桶，而是通过指定一个桶范围，即 $\left| {{H}_{\overrightarrow{a}}^{R}\left( o\right)  - {H}_{\overrightarrow{a}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}$ ，来虚拟地施加所有不同轮次的锚桶。由于哈希值（即投影）已记录在哈希表中，我们可以使用任意桶范围来确定一个锚桶。根据命题3，将第 $R$ 轮的锚桶扩大 $c$ 倍，从而生成第 ${cR}$ 轮的锚桶，这是通过将相应的桶范围扩大 $c$ 倍来实现的，其中 $c$ 不必为整数。因此，算法1适用于任意 $c > 1$ 。

### 5.2 Bound on Approximation Ratio

### 5.2 近似比的界

For $c$ -ANN search,we now present the bound on approximation ratio for Algorithm 1

对于$c$ - 最近邻（ANN）搜索，我们现在给出算法1的近似比界限

THEOREM 1. Algorithm 1 returns a ${c}^{2}$ -approximate NN with probability at least $\frac{1}{2} - \delta$ .

定理1. 算法1以至少$\frac{1}{2} - \delta$的概率返回一个${c}^{2}$ - 近似最近邻（NN）。

Since both QALSH and C2LSH use the technique of virtual rehashing, this theorem is a stronger version of Theorem 1 of C2LSH in the sense that the probability in this theorem is explicitly bounded from below. This theorem simply follows from the combination of Theorem 1 of C2LSH and Lemma 3 of QALSH.

由于QALSH（快速近似最近邻哈希）和C2LSH（基于聚类的局部敏感哈希）都使用了虚拟重哈希技术，从该定理中的概率明确有下界这一意义上来说，此定理是C2LSH定理1的更强版本。该定理直接由C2LSH的定理1和QALSH的引理3组合得出。

### 5.3 Parameter Settings

### 5.3 参数设置

The accuracy of QALSH is controlled by error probability $\delta$ ,approximation ratio $c$ and false positive percentage $\beta$ , where $\delta ,c$ and $\beta$ are constants specified by users. $\delta$ controls the success rate of any LSH-based method for $c$ -ANN search. In this paper,we set $\delta  = \frac{1}{e}$ . A smaller $c$ means a higher accuracy. Intuitively,a bigger $\beta$ allows $\mathrm{C}2\mathrm{{LSH}}$ and QALSH to check more frequent objects, and hence enables them to achieve a better search quality, with higher costs in terms of random I/Os. Similar to C2LSH,QALSH sets $\beta  = {100}/n$ to restrict the number of random I/Os.

QALSH的准确性由误差概率$\delta$、近似比$c$和误报率$\beta$控制，其中$\delta ,c$和$\beta$是用户指定的常量。$\delta$控制任何基于LSH的方法进行$c$ - 近似最近邻（ANN）搜索的成功率。在本文中，我们设置$\delta  = \frac{1}{e}$。较小的$c$意味着更高的准确性。直观地说，较大的$\beta$允许$\mathrm{C}2\mathrm{{LSH}}$和QALSH检查更多频繁出现的对象，从而使它们能够实现更好的搜索质量，但会在随机输入/输出（I/O）方面产生更高的成本。与C2LSH类似，QALSH设置$\beta  = {100}/n$来限制随机I/O的数量。

We now consider the base cardinality $m$ ,collision threshold percentage $\alpha$ and collision threshold $l$ . Referring to Equation 6 of Lemma 3,let ${m}_{1} = \left\lceil  {\frac{1}{2{\left( {p}_{1} - \alpha \right) }^{2}}\ln \frac{1}{\delta }}\right\rceil$ and ${m}_{2} =$ $\left\lceil  {\frac{1}{2{\left( \alpha  - {p}_{2}\right) }^{2}}\ln \frac{2}{\beta }}\right\rceil$ ,we have $m = \max \left( {{m}_{1},{m}_{2}}\right)$ . Since ${p}_{2} <$ $\alpha  < {p}_{1},{m}_{1}$ increases monotonically with $\alpha$ and ${m}_{2}$ decreases monotonically with $\alpha$ . Since $m = \max \left( {{m}_{1},{m}_{2}}\right) ,m$ is smallest when ${m}_{1} = {m}_{2}$ . Then, $\alpha$ can be determined by:

我们现在考虑基数下限 $m$、冲突阈值百分比 $\alpha$ 和冲突阈值 $l$。参考引理 3 的公式 6，设 ${m}_{1} = \left\lceil  {\frac{1}{2{\left( {p}_{1} - \alpha \right) }^{2}}\ln \frac{1}{\delta }}\right\rceil$ 且 ${m}_{2} =$ $\left\lceil  {\frac{1}{2{\left( \alpha  - {p}_{2}\right) }^{2}}\ln \frac{2}{\beta }}\right\rceil$，我们有 $m = \max \left( {{m}_{1},{m}_{2}}\right)$。由于 ${p}_{2} <$ $\alpha  < {p}_{1},{m}_{1}$ 随 $\alpha$ 单调递增，且 ${m}_{2}$ 随 $\alpha$ 单调递减。由于当 ${m}_{1} = {m}_{2}$ 时 $m = \max \left( {{m}_{1},{m}_{2}}\right) ,m$ 最小。那么，$\alpha$ 可由以下方式确定：

$$
\alpha  = \frac{\eta  \cdot  {p}_{1} + {p}_{2}}{1 + \eta },\text{ where }\eta  = \sqrt{\frac{\ln \frac{2}{\beta }}{\ln \frac{1}{\delta }}} \tag{7}
$$

Replacing $\alpha$ in ${m}_{1}$ by Equation 7,we have:

将式7代入${m}_{1}$中的$\alpha$，我们得到：

$$
m = \left\lceil  \frac{{\left( \sqrt{\ln \frac{2}{\beta }} + \sqrt{\ln \frac{1}{\delta }}\right) }^{2}}{2{\left( {p}_{1} - {p}_{2}\right) }^{2}}\right\rceil   \tag{8}
$$

After setting the values of $m$ and $\alpha$ ,we compute the integer collision threshold $l$ as follows:

在设定$m$和$\alpha$的值后，我们按如下方式计算整数碰撞阈值$l$：

$$
l = \lceil {\alpha m}\rceil  \tag{9}
$$

The base cardinality $m$ is simply the number of hash tables in QALSH. A small $m$ leads to small time and space overhead in QALSH,as shown in Section 5.4. However, $m$ must be set to satisfy the requirement of Lemma 3 for quality guarantee. It follows from Equation 8 that $m$ decreases monotonically with the difference $\left( {{p}_{1} - {p}_{2}}\right)$ for fixed $\delta$ and $\beta$ . From Section 3.2,we know there is a value of $w$ in the range $\left\lbrack  {0,{10}}\right\rbrack$ to maximize $\left( {{p}_{1} - {p}_{2}}\right)$ . Both E2LSH and LSB-Forest manually set bucket width $w = {4.0}$ ,while C2LSH manually set $w = {1.0}$ . In the next section,we propose to automatically decide $w$ so as to minimize the base cardinality $m$ .

基础基数 $m$ 简单来说就是QALSH（快速近似最近邻搜索哈希算法）中哈希表的数量。如5.4节所示，较小的 $m$ 会使QALSH的时间和空间开销较小。然而，为保证质量，必须设置 $m$ 以满足引理3的要求。由公式8可知，对于固定的 $\delta$ 和 $\beta$ ， $m$ 随差值 $\left( {{p}_{1} - {p}_{2}}\right)$ 单调递减。从3.2节可知，在区间 $\left\lbrack  {0,{10}}\right\rbrack$ 内存在一个 $w$ 的值能使 $\left( {{p}_{1} - {p}_{2}}\right)$ 达到最大。E2LSH（欧几里得空间局部敏感哈希）和LSB - Forest（最低有效位森林）都手动设置桶宽 $w = {4.0}$ ，而C2LSH（余弦空间局部敏感哈希）手动设置 $w = {1.0}$ 。在下一节中，我们提议自动确定 $w$ ，以便最小化基础基数 $m$ 。

#### 5.3.1 Automatically Setting $w$ by Minimizing $m$

#### 5.3.1 通过最小化 $m$ 自动设置 $w$

The strategy of minimizing $m$ is to select the value of $w$ that maximizes the difference $\left( {{p}_{1} - {p}_{2}}\right)$ . Formally,we have Lemma 5 to minimize $m$ .

最小化 $m$ 的策略是选择能使差值 $\left( {{p}_{1} - {p}_{2}}\right)$ 最大化的 $w$ 值。形式上，我们有引理 5 来最小化 $m$。

LEMMA 5. Suppose $\delta$ and $\beta$ are user-specified constants, for any approximation ratio $c > 1$ ,the base cardinality $m$ of ${QALSH}$ is minimized by setting

引理 5. 假设 $\delta$ 和 $\beta$ 是用户指定的常数，对于任何近似比 $c > 1$，通过设置可使 ${QALSH}$ 的基数 $m$ 最小化

$$
w = \sqrt{\frac{8{c}^{2}\ln c}{{c}^{2} - 1}} \tag{10}
$$

Proof. Let $\mu \left( w\right)  = {p}_{1} - {p}_{2}$ . From Equation 4,we have:

证明. 设 $\mu \left( w\right)  = {p}_{1} - {p}_{2}$。由方程 4，我们有：

$$
\mu \left( w\right)  = {p}_{1} - {p}_{2}
$$

$$
 = {\int }_{-\frac{w}{2}}^{\frac{w}{2}}\frac{1}{\sqrt{2\pi }}{e}^{-\frac{{t}^{2}}{2}}{dt} - {\int }_{-\frac{w}{2c}}^{\frac{w}{2c}}\frac{1}{\sqrt{2\pi }}{e}^{-\frac{{t}^{2}}{2}}{dt}
$$

$$
 = \frac{2}{\sqrt{2\pi }}{\int }_{-\infty }^{\frac{w}{2}}{e}^{-\frac{{t}^{2}}{2}}{dt} - \frac{2}{\sqrt{2\pi }}{\int }_{-\infty }^{\frac{w}{2c}}{e}^{-\frac{{t}^{2}}{2}}{dt}
$$

Using the basic techniques of calculus, we take the derivative and obtain the following equation:

运用微积分的基本技巧，我们求导并得到以下方程：

$$
{\mu }^{\prime }\left( w\right)  = \frac{1}{\sqrt{2\pi }}\left( {{e}^{-\frac{{w}^{2}}{8}} - \frac{1}{c} \cdot  {e}^{-\frac{{w}^{2}}{8{c}^{2}}}}\right) 
$$

Let ${\mu }^{\prime }\left( w\right)  = 0$ . Since $w > 0$ and $c > 1$ ,we have the expression ${w}^{ * } = \sqrt{\frac{8{c}^{2}\ln c}{{c}^{2} - 1}}$ . When $0 < w < {w}^{ * },{\mu }^{\prime }\left( w\right)  > 0$ and when $w > {w}^{ * },{\mu }^{\prime }\left( w\right)  < 0$ . Thus, $\mu \left( w\right)$ monotonically increases with $w$ for $0 < w < {w}^{ * }$ ,and monotonically decreases with $w$ for $w > {w}^{ * }$ . Therefore, $\mu \left( w\right)  = {p}_{1} - {p}_{2}$ achieves its maximum value when $w = {w}^{ * }$ . From Equation 8, $m$ decreases monotonically with the difference $\left( {{p}_{1} - {p}_{2}}\right)$ since $\beta$ and $\delta$ are constants. Thus, $m$ achieves its minimum value when $w = {w}^{ * }$ . Since Equation 8 is derived from Lemma 3 the minimum value of $m$ satisfies the quality guarantee.

设${\mu }^{\prime }\left( w\right)  = 0$。由于$w > 0$且$c > 1$，我们有表达式${w}^{ * } = \sqrt{\frac{8{c}^{2}\ln c}{{c}^{2} - 1}}$。当$0 < w < {w}^{ * },{\mu }^{\prime }\left( w\right)  > 0$时以及当$w > {w}^{ * },{\mu }^{\prime }\left( w\right)  < 0$时。因此，对于$0 < w < {w}^{ * }$，$\mu \left( w\right)$随$w$单调递增，对于$w > {w}^{ * }$，$\mu \left( w\right)$随$w$单调递减。所以，当$w = {w}^{ * }$时，$\mu \left( w\right)  = {p}_{1} - {p}_{2}$取得其最大值。由方程8可知，由于$\beta$和$\delta$为常数，$m$随差值$\left( {{p}_{1} - {p}_{2}}\right)$单调递减。因此，当$w = {w}^{ * }$时，$m$取得其最小值。由于方程8是由引理3推导得出的，所以$m$的最小值满足质量保证。

### 5.4 Time and Space Complexity

### 5.4 时间和空间复杂度

Since we set $\beta  = \frac{100}{n},{\beta n}$ is constant. From Equations 8 and 9,we have $m = O\left( {\log n}\right)$ and $l = O\left( {\log n}\right)$ ,respectively.

由于我们设定 $\beta  = \frac{100}{n},{\beta n}$ 为常数。根据公式 8 和 9，我们分别得到 $m = O\left( {\log n}\right)$ 和 $l = O\left( {\log n}\right)$。

The time cost of QALSH consists of four parts: First, first, computing the projection of a query for $m$ hash tables costs ${md} = O\left( {d\log n}\right)$ ; Second,locating the $m$ round-1 anchor buckets in ${B}^{ + }$ -tree costs $m\log n = O\left( {\left( \log n\right) }^{2}\right)$ ; Third,in the worst case, finding the frequent objects as candidates needs to do collision counting for all the $n$ objects over each hash table,which costs $\ln  = O\left( {n\log n}\right)$ ; Finally,calculating Euclidean distance for candidates costs ${\beta nd} = O\left( d\right)$ . Therefore,the time complexity of QALSH is $O\left( {d\log n + {\left( \log n\right) }^{2} + }\right.$ $n\log n + d) = O\left( {d\log n + n\log n}\right) .$

QALSH（快速近似最近邻哈希）的时间成本由四部分组成：第一，计算查询在 $m$ 个哈希表上的投影，成本为 ${md} = O\left( {d\log n}\right)$；第二，在 ${B}^{ + }$ -树中定位 $m$ 个第一轮锚桶，成本为 $m\log n = O\left( {\left( \log n\right) }^{2}\right)$；第三，在最坏情况下，找出频繁对象作为候选对象需要对每个哈希表上的所有 $n$ 个对象进行冲突计数，成本为 $\ln  = O\left( {n\log n}\right)$；最后，计算候选对象的欧几里得距离，成本为 ${\beta nd} = O\left( d\right)$。因此，QALSH 的时间复杂度为 $O\left( {d\log n + {\left( \log n\right) }^{2} + }\right.$ $n\log n + d) = O\left( {d\log n + n\log n}\right) .$

The space complexity of QALSH consists of two parts: the space of dataset $O\left( {nd}\right)$ and the space of index ${mn} =$ $O\left( {n\log n}\right)$ for $m$ hash tables which store $n$ data objects’ id and projection. Thus, the total space consumption of QALSH is $O\left( {{nd} + n\log n}\right)$ .

QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）的空间复杂度由两部分组成：数据集 $O\left( {nd}\right)$ 的空间以及用于存储 $n$ 个数据对象的 ID 和投影的 $m$ 个哈希表的索引空间 ${mn} =$ $O\left( {n\log n}\right)$。因此，QALSH 的总空间消耗为 $O\left( {{nd} + n\log n}\right)$。

## 6. EXPERIMENTS

## 6. 实验

In this section, we study the performance of QALSH using four real datasets. Since QALSH has quality guarantee and is designed for external memory, we take two state-of-the-art schemes of the same kind as the benchmark, namely, LSB-Forest and C2LSH.

在本节中，我们使用四个真实数据集来研究 QALSH 的性能。由于 QALSH 具有质量保证且是为外部内存设计的，我们选取两种同类的最先进方案作为基准，即 LSB 森林（LSB-Forest）和 C2LSH。

### 6.1 Experiment Setup

### 6.1 实验设置

#### 6.1.1 Benchmark Methods

#### 6.1.1 基准方法

- LSB-Forest. LSB-Forest uses a set of $L$ LSB-Trees to achieve quality guarantee, which has a success probability at least $\frac{1}{2} - \frac{1}{e}$ . LSB-Forest requires ${2L}$ buffer pages for $c$ -ANN search. Since LSB-Forest has been shown to outperform iDistance [8] and MEDRANK [3], they are omitted for comparison here.

- LSB森林（LSB-Forest）。LSB森林使用一组$L$个LSB树（LSB-Trees）来实现质量保证，其成功概率至少为$\frac{1}{2} - \frac{1}{e}$。LSB森林进行$c$ - 近似最近邻（ANN）搜索需要${2L}$个缓冲页。由于已证明LSB森林的性能优于iDistance [8]和MEDRANK [3]，因此这里省略它们以进行比较。

- C2LSH. C2LSH is most related to QALSH. It requires a buffer of ${2m}$ pages for $c$ -ANN search,where $m$ is the number of hash tables used in C2LSH. We consider C2LSH with $l$ as the collision threshold,as only under this case it has quality guarantee.

- C2LSH。C2LSH与QALSH最为相关。它进行$c$ - 近似最近邻（ANN）搜索需要一个${2m}$页的缓冲区，其中$m$是C2LSH中使用的哈希表数量。我们将$l$作为碰撞阈值来考虑C2LSH，因为只有在这种情况下它才有质量保证。

Our method is implemented in $\mathrm{C} +  +$ . All methods are compiled with gcc 4.8 with -O3. All experiments were done on a PC with Intel Core i7-2670M 2.20GHz CPU, 8 GB memory and 1 TB hard disk, running Linux 3.11.

我们的方法在 $\mathrm{C} +  +$ 中实现。所有方法均使用 gcc 4.8 并开启 -O3 选项进行编译。所有实验均在一台配备英特尔酷睿 i7 - 2670M 2.20GHz 处理器、8GB 内存和 1TB 硬盘、运行 Linux 3.11 系统的个人电脑上进行。

#### 6.1.2 Datasets and Queries

#### 6.1.2 数据集与查询

We use four real datasets in our experiments. We scale up values to integers as required by LSB-Forest and C2LSH, while QALSH is able to handle real numbers directly. We set page size $B$ according to what LSB-Forest requires for best performance.

我们在实验中使用了四个真实数据集。由于 LSB - 森林（LSB - Forest）和 C2LSH 的要求，我们将数值缩放为整数，而 QALSH 能够直接处理实数。我们根据 LSB - 森林实现最佳性能的要求设置页面大小 $B$。

- Mnist ${}^{2}$ . This 784-dimensional dataset has ${60},{000}\mathrm{{ob}}$ - jects. We follow [15, 4] and consider the top-50 dimensions with the largest variance. $B$ is set to be $4\mathrm{{KB}}$ .

- 手写数字数据集（Mnist） ${}^{2}$。这个 784 维的数据集有 ${60},{000}\mathrm{{ob}}$ 个对象。我们遵循文献 [15, 4] 的方法，考虑方差最大的前 50 个维度。$B$ 被设置为 $4\mathrm{{KB}}$。

- Sift ${}^{3}$ We use1,000,000128-dimensional base vectors of Sift as dataset. $B$ is set to be $4\mathrm{{KB}}$ .

- SIFT特征 ${}^{3}$ 我们使用1000000个128维的SIFT（尺度不变特征变换，Scale-Invariant Feature Transform）基础向量作为数据集。$B$ 设置为 $4\mathrm{{KB}}$。

- LabelMe ${}^{4}$ . This 512-dimensional dataset has181,093 objects. The coordinates are normalized to be integers in a range of $\left\lbrack  {0,{58104}}\right\rbrack  .B$ is set to be $8\mathrm{{KB}}$ .

- LabelMe数据集 ${}^{4}$。这个512维的数据集包含181093个对象。坐标被归一化为 $\left\lbrack  {0,{58104}}\right\rbrack  .B$ 范围内的整数，$\left\lbrack  {0,{58104}}\right\rbrack  .B$ 设置为 $8\mathrm{{KB}}$。

- P53 ${}^{5}$ . The 5,408-dimensional biological dataset in 2012 version has 31,420 objects. We removed all objects that have missing values, so that the cardinality of the dataset is reduced to 31,159 . The coordinates are normalized to be integers in a range of $\left\lbrack  {0,{10000}}\right\rbrack  .B$ is set to be ${64}\mathrm{{KB}}$ .

- P53数据集 ${}^{5}$。2012版的5408维生物数据集有31420个对象。我们移除了所有包含缺失值的对象，因此数据集的基数减少到31159。坐标被归一化为 $\left\lbrack  {0,{10000}}\right\rbrack  .B$ 范围内的整数，$\left\lbrack  {0,{10000}}\right\rbrack  .B$ 设置为 ${64}\mathrm{{KB}}$。

Both LSB-Forest and C2LSH study the performance by averaging the query results of 50 random queries, while SRS uses 100 random queries. We conduct the experiments using three sets of queries, which,respectively, contain 50, 100, and 200 queries. Since the experimental results over the three query sets exhibit similar trends, we only report the results over the set of 100 queries due to space limitation. For the datasets Mnist and Sift, the queries are uniformly randomly chosen from their corresponding test sets. For the datasets LabelMe and P53, the queries are uniformly randomly chosen from the data objects. Mnist and Sift are regarded as low-dimensional datasets. LabelMe and P53 are regarded as medium- and high-dimensional datasets, respectively.

LSB - Forest和C2LSH均通过对50次随机查询的结果求平均值来研究性能，而SRS使用100次随机查询。我们使用三组查询进行实验，这三组查询分别包含50、100和200次查询。由于三组查询集的实验结果呈现出相似的趋势，受篇幅限制，我们仅报告100次查询集的结果。对于Mnist（手写数字数据集）和Sift（尺度不变特征变换数据集）数据集，查询是从其对应的测试集中均匀随机选取的。对于LabelMe（图像标注数据集）和P53（P53基因数据集）数据集，查询是从数据对象中均匀随机选取的。Mnist和Sift被视为低维数据集，LabelMe和P53分别被视为中维和高维数据集。

#### 6.1.3 Evaluation Metrics

#### 6.1.3 评估指标

We use the following metrics for performance evaluation.

我们使用以下指标进行性能评估。

- Index Size. Since the size of datasets are constant for all methods, we use the size of the index generated by a method to evaluate the space overhead of the method.

- 索引大小。由于所有方法的数据集大小是固定的，我们使用一种方法生成的索引大小来评估该方法的空间开销。

- Overall Ratio. Overall ratio [15, 4] is used to measure the accuracy of a method. For the $c - k$ -ANN search,it is defined as $\frac{1}{k}\mathop{\sum }\limits_{{i = 1}}^{k}\frac{\begin{Vmatrix}{o}_{i},q\end{Vmatrix}}{\begin{Vmatrix}{o}_{i}^{ * },q\end{Vmatrix}}$ ,where ${o}_{i}$ is the $i$ -th object returned by a method and ${o}_{i}^{ * }$ is the true $i$ -th nearest object, $i = 1,2,\ldots ,k$ . Intuitively,a smaller overall ratio means a higher accuracy.

- 总体比率。总体比率[15, 4]用于衡量一种方法的准确性。对于$c - k$ -最近邻搜索，其定义为$\frac{1}{k}\mathop{\sum }\limits_{{i = 1}}^{k}\frac{\begin{Vmatrix}{o}_{i},q\end{Vmatrix}}{\begin{Vmatrix}{o}_{i}^{ * },q\end{Vmatrix}}$ ，其中${o}_{i}$ 是一种方法返回的第$i$ 个对象，${o}_{i}^{ * }$ 是真正的第$i$ 个最近邻对象，$i = 1,2,\ldots ,k$ 。直观地说，总体比率越小意味着准确性越高。

<!-- Media -->

Table 1: Index Size of QALSH vs. Bucket Width $w$

表1：QALSH的索引大小与桶宽度$w$ 的关系

<table><tr><td>$w$</td><td>Mnist</td><td>Sift</td><td>LabelMe</td><td>P53</td></tr><tr><td>1.000</td><td>49.6 MB</td><td>1.0 GB</td><td>163.8 MB</td><td>68.6 MB</td></tr><tr><td>2.000</td><td>19.1 MB</td><td>388.6 MB</td><td>63.7 MB</td><td>26.5 MB</td></tr><tr><td>2.719</td><td>16.5 MB</td><td>336.0 MB</td><td>54.6 MB</td><td>23.1 MB</td></tr><tr><td>3.000</td><td>16.8 MB</td><td>344.1 MB</td><td>56.1 MB</td><td>23.5 MB</td></tr><tr><td>4.000</td><td>23.2 MB</td><td>473.6 MB</td><td>77.4 MB</td><td>${32.2}\mathrm{{MB}}$</td></tr></table>

<table><tbody><tr><td>$w$</td><td>手写数字数据集（Mnist）</td><td>尺度不变特征变换（Sift）</td><td>标签我（LabelMe）</td><td>P53</td></tr><tr><td>1.000</td><td>49.6兆字节</td><td>1.0吉字节</td><td>163.8兆字节</td><td>68.6兆字节</td></tr><tr><td>2.000</td><td>19.1兆字节</td><td>388.6兆字节</td><td>63.7兆字节</td><td>26.5兆字节</td></tr><tr><td>2.719</td><td>16.5兆字节</td><td>336.0兆字节</td><td>54.6兆字节</td><td>23.1兆字节</td></tr><tr><td>3.000</td><td>16.8兆字节</td><td>344.1兆字节</td><td>56.1兆字节</td><td>23.5兆字节</td></tr><tr><td>4.000</td><td>23.2兆字节</td><td>473.6兆字节</td><td>77.4兆字节</td><td>${32.2}\mathrm{{MB}}$</td></tr></tbody></table>

Table 2: Index Size of C2LSH vs. Bucket Width $w$

表2：C2LSH（碰撞计数局部敏感哈希）的索引大小与桶宽度对比 $w$

<table><tr><td>$w$</td><td>Mnist</td><td>Sift</td><td>LabelMe</td><td>P53</td></tr><tr><td>1.000</td><td>61.2 MB</td><td>1.2 GB</td><td>435.2 MB</td><td>83.5 MB</td></tr><tr><td>2.000</td><td>29.9 MB</td><td>597.7 MB</td><td>193.1 MB</td><td>41.6 MB</td></tr><tr><td>2.184</td><td>29.5 MB</td><td>589.6 MB</td><td>188.5 MB</td><td>41.0 MB</td></tr><tr><td>3.000</td><td>33.1 MB</td><td>669.4 MB</td><td>197.5 MB</td><td>45.4 MB</td></tr><tr><td>4.000</td><td>46.1 MB</td><td>945.6 MB</td><td>258.3 MB</td><td>62.1 MB</td></tr></table>

<table><tbody><tr><td>$w$</td><td>手写数字数据集（Mnist）</td><td>尺度不变特征变换（Sift）</td><td>标注图像数据集（LabelMe）</td><td>P53</td></tr><tr><td>1.000</td><td>61.2兆字节</td><td>1.2吉字节</td><td>435.2兆字节</td><td>83.5兆字节</td></tr><tr><td>2.000</td><td>29.9兆字节</td><td>597.7兆字节</td><td>193.1兆字节</td><td>41.6兆字节</td></tr><tr><td>2.184</td><td>29.5兆字节</td><td>589.6兆字节</td><td>188.5兆字节</td><td>41.0兆字节</td></tr><tr><td>3.000</td><td>33.1兆字节</td><td>669.4兆字节</td><td>197.5兆字节</td><td>45.4兆字节</td></tr><tr><td>4.000</td><td>46.1兆字节</td><td>945.6兆字节</td><td>258.3兆字节</td><td>62.1兆字节</td></tr></tbody></table>

<!-- Media -->

- I/O Cost. We follow LSB-Forest and C2LSH to use I/O cost to evaluate the efficiency of a method. It is defined as the number of pages to be accessed. I/O cost consists of two parts: the cost of finding candidates (i.e. frequent objects) and the cost of distance computation of candidates in the original space.

- I/O成本。我们遵循LSB - 森林（LSB - Forest）和C2LSH的方法，使用I/O成本来评估一种方法的效率。它被定义为需要访问的页面数量。I/O成本由两部分组成：查找候选对象（即频繁对象）的成本和在原始空间中对候选对象进行距离计算的成本。

- Running Time. Since query-aware bucket partition introduces extra overhead, we also consider the running time cost for processing a query. It is defined as the wall-clock time for a method to solve the $c - k$ -ANN problem.

- 运行时间。由于查询感知桶划分会引入额外的开销，我们还考虑处理查询的运行时间成本。它被定义为一种方法解决$c - k$ - 近似最近邻（ANN）问题的实际时钟时间。

### 6.2 Parameter Settings

### 6.2 参数设置

For the sake of fairness, the success probability of all methods is set to $\frac{1}{2} - \frac{1}{e}$ ,i.e., $\delta$ of QALSH and C2LSH is set to $\frac{1}{e}$ . We use setting $c = {2.0}$ ,so that LSB-Forest and C2LSH can achieve their best performance. Both QALSH and C2LSH set false positive percentage $\beta$ to be ${100}/n$ to limit the number of candidates and hence the corresponding number of random I/Os. Other parameters of LSB-Forest and C2LSH are set to their default values [15, 4].

为了公平起见，所有方法的成功概率均设为$\frac{1}{2} - \frac{1}{e}$，即QALSH（查询感知局部敏感哈希）和C2LSH（基于聚类的局部敏感哈希）的$\delta$设为$\frac{1}{e}$。我们采用设置$c = {2.0}$，以便LSB - 森林（最低有效位森林）和C2LSH能达到其最佳性能。QALSH和C2LSH均将误报率$\beta$设为${100}/n$，以限制候选对象的数量，从而限制相应的随机输入/输出操作数量。LSB - 森林和C2LSH的其他参数设为其默认值[15, 4]。

We compute bucket width $w$ for QALSH by Equation 10, and get $w = {2.719}$ for $c = 2$ . Since $w$ is manually set to $\overline{1.0}$ and 4.0 in C2LSH and LSB-Forest respectively, we also consider two intermediate values $w = {2.0}$ and $w = {3.0}$ . Table 1 shows the index size of QALSH under the five settings of $w$ . We observe that the index size under setting $w = {2.719}$ is indeed the smallest. Since each hash table has the same size, the difference in index size reflects the difference in the number of hash tables,i.e.,the base cardinality $m$ . In other words,setting $w = {2.719}$ minimizes $m$ among the five settings of $w$ . We also evaluate the overall ratio, $\mathrm{I}/\mathrm{O}$ cost and running time of QALSH under the five settings of $w$ . We observe that the overall ratios under different settings are basically equal to each other. Due to the smallest index size under setting $w = {2.719}$ ,both the I/O cost and running time under this setting are the smallest. Due to space limitation, we omit those results here.

我们通过公式10计算QALSH（快速近似最近邻哈希，Quick Approximate Nearest Neighbor Hashing）的桶宽度$w$，并针对$c = 2$得到$w = {2.719}$。由于在C2LSH（基于碰撞的局部敏感哈希，Collision-based Locality-Sensitive Hashing）和LSB - Forest（最低有效位森林）中，$w$分别手动设置为$\overline{1.0}$和4.0，我们还考虑了两个中间值$w = {2.0}$和$w = {3.0}$。表1展示了在$w$的五种设置下QALSH的索引大小。我们观察到，在设置$w = {2.719}$下的索引大小确实是最小的。由于每个哈希表的大小相同，索引大小的差异反映了哈希表数量的差异，即基数$m$。换句话说，在$w$的五种设置中，设置$w = {2.719}$使$m$最小化。我们还评估了在$w$的五种设置下QALSH的总体比率、$\mathrm{I}/\mathrm{O}$成本和运行时间。我们观察到，不同设置下的总体比率基本相等。由于在设置$w = {2.719}$下索引大小最小，因此该设置下的I/O成本和运行时间也是最小的。由于篇幅限制，我们在此省略这些结果。

Since the base cardinality $m$ of both QALSH and C2LSH is computed by Equation 8, we also automatically compute $w$ for C2LSH to minimize $m$ (or to maximize $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ ), and get $w = {2.184}$ for $c = 2$ . Table 2 shows the index size of $\mathrm{C}2\mathrm{{LSH}}$ under the five settings of $w$ . Interestingly, our experimental results show that C2LSH performs better under the setting $w = {2.184}$ than $w = {1.0}$ ,which is the default value of C2LSH [4]. Due to space limitation, we also omit the results here.

由于QALSH和C2LSH的基础基数$m$均由公式8计算得出，我们也会自动为C2LSH计算$w$，以最小化$m$（或最大化$\left( {{\xi }_{1} - {\xi }_{2}}\right)$），并得到$c = 2$对应的$w = {2.184}$。表2展示了在$w$的五种设置下$\mathrm{C}2\mathrm{{LSH}}$的索引大小。有趣的是，我们的实验结果表明，在设置$w = {2.184}$下，C2LSH的性能优于$w = {1.0}$，而$w = {1.0}$是C2LSH的默认值[4]。由于篇幅限制，我们在此也省略了相关结果。

---

<!-- Footnote -->

http://yann.lecun.com/exdb/mnist/

http://corpus-texmex.irisa.fr/

http://labelme.csail.mit.edu/inctructions.html

${}^{5}$ http://archive.ics.uci.edu/ml/datasets/p53+ Mutants

${}^{5}$ http://archive.ics.uci.edu/ml/datasets/p53+ 突变体

<!-- Footnote -->

---

<!-- Media -->

Table 3: Statistics of Index Size

表3：索引大小统计

<table><tr><td/><td>Mnist</td><td>Sift</td><td>LabelMe</td><td>P53</td></tr><tr><td>$L$</td><td>55</td><td>354</td><td>213</td><td>102</td></tr><tr><td>LSB-Forest</td><td>858.1 MB</td><td>246.3 GB</td><td>106.6 GB</td><td>69.4 GB</td></tr><tr><td>$m$</td><td>115</td><td>147</td><td>128</td><td>107</td></tr><tr><td>C2LSH</td><td>29.5 MB</td><td>589.6 MB</td><td>188.5 MB</td><td>41.0 MB</td></tr><tr><td>$m$</td><td>65</td><td>83</td><td>72</td><td>61</td></tr><tr><td>QALSH</td><td>16.5 MB</td><td>${336.0}\mathrm{{MB}}$</td><td>54.6 MB</td><td>23.1 MB</td></tr></table>

<table><tbody><tr><td></td><td>手写数字数据集（Mnist）</td><td>尺度不变特征变换（Sift）</td><td>标签我（LabelMe）</td><td>P53</td></tr><tr><td>$L$</td><td>55</td><td>354</td><td>213</td><td>102</td></tr><tr><td>最低有效位森林（LSB - Forest）</td><td>858.1兆字节</td><td>246.3吉字节</td><td>106.6吉字节</td><td>69.4吉字节</td></tr><tr><td>$m$</td><td>115</td><td>147</td><td>128</td><td>107</td></tr><tr><td>C2局部敏感哈希（C2LSH）</td><td>29.5兆字节</td><td>589.6兆字节</td><td>188.5兆字节</td><td>41.0兆字节</td></tr><tr><td>$m$</td><td>65</td><td>83</td><td>72</td><td>61</td></tr><tr><td>卡尔什（QALSH）</td><td>16.5兆字节</td><td>${336.0}\mathrm{{MB}}$</td><td>54.6兆字节</td><td>23.1兆字节</td></tr></tbody></table>

<!-- Media -->

Our experiments demonstrate the effectiveness of automatically determining the bucket width $w$ by minimizing the base cardinality $m$ . In the subsequent experiments,we only show the results of both QALSH and C2LSH with $w$ set to the automatically determined values. Specifically, we have $w = {2.719}$ for $c = 2$ for QALSH,and $w = {2.184}$ for $c = 2$ for $\mathrm{C}2\mathrm{{LSH}}$ . Since the number of hash functions of LSB-Forest is not affected by $w$ ,we still use its manually set value $w = {4.0}$ .

我们的实验证明了通过最小化基础基数 $m$ 自动确定桶宽度 $w$ 的有效性。在后续实验中，我们仅展示将 $w$ 设置为自动确定值时 QALSH 和 C2LSH 的结果。具体而言，对于 QALSH，当 $c = 2$ 时，我们有 $w = {2.719}$；对于 $\mathrm{C}2\mathrm{{LSH}}$，当 $c = 2$ 时，我们有 $w = {2.184}$。由于 LSB - 森林的哈希函数数量不受 $w$ 影响，我们仍使用其手动设置的值 $w = {4.0}$。

### 6.3 Index Size and Indexing Time

### 6.3 索引大小和索引时间

We list the index sizes of all the three methods over the four datasets in Table 3,where $L$ is the number of LSB-Trees used by LSB-Forest,and $m$ is the number of hash tables used by C2LSH and QALSH. Each method needs ${2m}$ or ${2L}$ buffer pages for performing $c$ -ANN search,in the experiments we set the number of buffer pages to be $2\max \left( {m,L}\right)$ so as to make LSB-Forest or C2LSH have enough buffer pages. Referring to Table 3,the $m$ value of QALSH is consistently smaller than that of C2LSH, and is also consistently smaller than the $L$ value of LSB-Forest except on the dataset Mnist. In other words, QALSH only needs a smaller number of buffer pages.

我们在表3中列出了三种方法在四个数据集上的索引大小，其中$L$是LSB森林（LSB-Forest）使用的LSB树（LSB-Trees）的数量，$m$是C2LSH和QALSH使用的哈希表的数量。每种方法在执行$c$ - 近似最近邻（ANN）搜索时需要${2m}$或${2L}$个缓冲页，在实验中我们将缓冲页的数量设置为$2\max \left( {m,L}\right)$，以便让LSB森林或C2LSH有足够的缓冲页。参考表3，QALSH的$m$值始终小于C2LSH的$m$值，并且除了在Mnist数据集上之外，也始终小于LSB森林的$L$值。换句话说，QALSH只需要较少数量的缓冲页。

For each dataset, the index sizes of QALSH and C2LSH are smaller than the index size of LSB-Forest by about two or three orders of magnitude. LSB-Forest stores coordinates of objects and Z-order values in leaf pages in each LSB-Tree. Large data dimensionality $d$ leads to large overhead for storing coordinates. Moreover, each Z-order value has ${uv}$ bits where $u = O\left( {{\log }_{2}d}\right)$ and $v = O\left( {\log {dn}}\right)$ . In total, the index size of LSB-Forest grows at the rate of $O\left( {{d}^{1.5}{n}^{1.5}}\right)$ . Therefore,LSB-Forest incurs extremely large space overhead on high-dimensional datasets. In contrast, the index sizes of QALSH and C2LSH are independent of $d$ . Meanwhile,QALSH and C2LSH only store object ids and projections in their hash tables, at the expense of using random I/Os to access coordinates for computing Euclidean distance. The index size of QALSH is about ${29}\%$ to ${57}\%$ of that of C2LSH. The difference between their index size is mainly due to the different number of hash tables needed by each method. Simpler query-aware LSH functions used by QALSH result in smaller number of hash tables.

对于每个数据集，QALSH（查询感知局部敏感哈希）和C2LSH（基于聚类的局部敏感哈希）的索引大小比LSB - 森林（LSB - Forest）的索引大小小大约两到三个数量级。LSB - 森林在每个LSB - 树（LSB - Tree）的叶页中存储对象的坐标和Z序值。较大的数据维度$d$会导致存储坐标的开销较大。此外，每个Z序值有${uv}$位，其中$u = O\left( {{\log }_{2}d}\right)$且$v = O\left( {\log {dn}}\right)$。总体而言，LSB - 森林的索引大小以$O\left( {{d}^{1.5}{n}^{1.5}}\right)$的速率增长。因此，LSB - 森林在高维数据集上会产生极大的空间开销。相比之下，QALSH和C2LSH的索引大小与$d$无关。同时，QALSH和C2LSH仅在其哈希表中存储对象ID和投影，代价是使用随机I/O来访问坐标以计算欧几里得距离。QALSH的索引大小约为C2LSH索引大小的${29}\%$到${57}\%$。它们索引大小的差异主要是由于每种方法所需的哈希表数量不同。QALSH使用的更简单的查询感知局部敏感哈希函数导致所需的哈希表数量更少。

The wall-clock time for building the index, i.e., the indexing time, is generally proportional to the index size. On every dataset, the indexing time of QALSH is the smallest while that of LSB-Forest is the largest. Specifically, on the dataset Sift with one million data objects, LSB-Forest takes more than 2.5 hours in building the index, and C2LSH takes about 3 minutes, while QALSH only takes about 50 seconds.

构建索引的挂钟时间，即索引构建时间，通常与索引大小成正比。在每个数据集上，QALSH（快速近似局部敏感哈希）的索引构建时间最短，而LSB - Forest（最低有效位森林）的索引构建时间最长。具体而言，在包含一百万个数据对象的Sift（尺度不变特征变换）数据集上，LSB - Forest构建索引需要超过2.5小时，C2LSH（一种局部敏感哈希算法）大约需要3分钟，而QALSH仅需约50秒。

### 6.4 Overall Ratio

### 6.4 总体比率

We evaluate the overall ratio for $2 - k$ -ANN search by varying $k$ from 1 to 100 . Results are shown in Figure 6

我们通过将$k$从1变化到100来评估$2 - k$ - 近似最近邻（ANN）搜索的总体比率。结果如图6所示

All the methods get satisfactory overall ratios, which are much smaller than the theoretical bound ${c}^{2} = 4$ . Compared to LSB-Forest, QALSH and C2LSH achieve significantly higher accuracy. The overall ratios of QALSH and C2LSH are always smaller than 1.05 , while the smallest overall ratio of LSB-Forest on the four datasets is still larger than 1.24. The overall ratios of QALSH are basically the same as those of C2LSH. This is because the parameters which affect accuracy are set to be the same for both methods.

所有方法都获得了令人满意的总体比率，这些比率远小于理论界限${c}^{2} = 4$。与LSB - 森林（LSB - Forest）相比，QALSH和C2LSH实现了显著更高的准确率。QALSH和C2LSH的总体比率始终小于1.05，而LSB - 森林在四个数据集上的最小总体比率仍大于1.24。QALSH的总体比率与C2LSH的基本相同。这是因为影响准确率的参数对这两种方法设置为相同。

As $k$ increases,the overall ratios of QALSH and C2LSH tend to increase while the overall ratio of LSB-Forest tends to decrease. In fact, both QALSH and C2LSH return the best $k$ objects out of a candidate set of size ${\beta n} + k - 1$ . As $k$ increases,only $k - 1$ additional candidates are checked for possible improvement on the ratios. In contrast, LSB-Forest tends to check relatively more objects.

随着$k$的增加，QALSH和C2LSH的总体比率趋于增加，而LSB - 森林（LSB - Forest）的总体比率趋于降低。实际上，QALSH和C2LSH都会从大小为${\beta n} + k - 1$的候选集中返回最佳的$k$个对象。随着$k$的增加，仅会检查$k - 1$个额外的候选对象以可能改善比率。相比之下，LSB - 森林（LSB - Forest）倾向于检查相对更多的对象。

### 6.5 I/O Cost

### 6.5 输入/输出成本

We evaluate the I/O cost for 2- $k$ -ANN search by varying $k$ from 1 to 100 . The results ${}^{6}$ are shown in Figure 7,

我们通过将 $k$ 从 1 变化到 100 来评估 2 - $k$ -近似最近邻（ANN）搜索的输入/输出（I/O）成本。结果 ${}^{6}$ 如图 7 所示。

Compared to QALSH and C2LSH, LSB-Forest requires much smaller I/O costs on low- and medium-dimensional datasets, i.e., Mnist, Sift and LabelMe. However, its overall ratio is much larger than those of QALSH and C2LSH. For the high-dimensional dataset P53, the I/O cost of QALSH is smaller than that of LSB-Forest. This is because the I/O cost of LSB-Forest monotonically increases as data dimensionality $d$ increases,while the I/O costs of QALSH and C2LSH are independent of $d$ . Compared to C2LSH,QALSH uses about ${49}\%$ to ${76}\%$ of the I/O costs of C2LSH,but still achieves the same accuracy.

与快速近似最近邻哈希（QALSH）和 C2LSH 相比，最低有效位森林（LSB - Forest）在低维和中维数据集（即 Mnist、Sift 和 LabelMe）上所需的输入/输出（I/O）成本要小得多。然而，其总体比率远大于 QALSH 和 C2LSH。对于高维数据集 P53，QALSH 的输入/输出（I/O）成本小于 LSB - Forest。这是因为 LSB - Forest 的输入/输出（I/O）成本随着数据维度 $d$ 的增加而单调增加，而 QALSH 和 C2LSH 的输入/输出（I/O）成本与 $d$ 无关。与 C2LSH 相比，QALSH 的输入/输出（I/O）成本约为 C2LSH 的 ${49}\%$ 到 ${76}\%$，但仍能达到相同的精度。

When $k$ increases,the I/O cost of LSB-Forest increases gently, while the I/O costs of QALSH and C2LSH increase more apparently. This is because LSB-Forest already stores the coordinates of objects in each LSB-Tree, and hence it computes the Euclidean distance without extra I/O costs. However, neither QALSH nor C2LSH stores the coordinates in hash tables,and thus one random $\mathrm{I}/\mathrm{O}$ is needed for every candidate in the worst case. As $k$ increases,the number of candidates increases,and accordingly the $\mathrm{I}/\mathrm{O}$ costs of both QALSH and C2LSH increase.

当 $k$ 增大时，LSB 森林（LSB-Forest）的输入/输出（I/O）成本缓慢增加，而快速近似最近邻哈希（QALSH）和基于码本的局部敏感哈希（C2LSH）的 I/O 成本增加得更为明显。这是因为 LSB 森林已经在每个 LSB 树（LSB-Tree）中存储了对象的坐标，因此它在计算欧几里得距离时无需额外的 I/O 成本。然而，QALSH 和 C2LSH 都没有在哈希表中存储坐标，因此在最坏的情况下，每个候选对象都需要一次随机 $\mathrm{I}/\mathrm{O}$ 操作。随着 $k$ 的增大，候选对象的数量增加，相应地，QALSH 和 C2LSH 的 $\mathrm{I}/\mathrm{O}$ 成本也会增加。

### 6.6 Running Time

### 6.6 运行时间

We study the running time for $2 - k$ -ANN search by varying $k$ from 1 to 100 . The results are shown in Figure 8

我们通过将 $k$ 从 1 变化到 100 来研究 $2 - k$ -近似最近邻（ANN）搜索的运行时间。结果如图 8 所示

Interestingly, the running time of LSB-Forest is larger than that of QALSH on the medium-dimensional dataset LabelMe, even though its I/O cost is smaller than that of QALSH. While the I/O cost of LSB-Forest is slightly larger than that of QALSH on the high-dimensional dataset P53, the running time of LSB-Forest is surprisingly larger than that of QALSH by more than two orders of magnitude. In fact,as data dimensionality $d$ increases,LSB-Forest tends to use more CPU time for finding the candidates whose Z-order values are closest to the Z-order value of the query. As already explained in Section 6.3,larger $d$ leads to longer $\mathrm{Z}$ -order values and hence leads to more time cost for processing Z-order values. It is worth mentioning that while QALSH is more efficient than LSB-forest on the medium-and high-dimensional datasets, it also achieves much higher searching accuracy than LSB-Forest.

有趣的是，在中等维度数据集LabelMe（标签我）上，LSB - Forest（最低有效位森林）的运行时间比QALSH（快速近似最近邻哈希）长，尽管其I/O成本比QALSH低。在高维数据集P53上，虽然LSB - Forest的I/O成本略高于QALSH，但其运行时间却比QALSH长两个数量级以上，这令人惊讶。实际上，随着数据维度$d$的增加，LSB - Forest倾向于花费更多的CPU时间来寻找Z序值最接近查询Z序值的候选对象。正如第6.3节所解释的，更大的$d$会导致更长的$\mathrm{Z}$序值，从而导致处理Z序值的时间成本更高。值得一提的是，虽然在中等和高维数据集上QALSH比LSB - Forest更高效，但它的搜索准确率也比LSB - Forest高得多。

---

<!-- Footnote -->

${}^{6}\mathrm{I}/\mathrm{O}$ costs of brute-force linear scan method over the datasets of Mnist, Sift, LabelMe and P53 are 3000, 125000, 45249 and 10353, respectively.

${}^{6}\mathrm{I}/\mathrm{O}$ 在Mnist（手写数字数据集）、Sift（尺度不变特征变换数据集）、LabelMe（标注图像数据集）和P53（P53蛋白质数据集）数据集上的暴力线性扫描方法的成本分别为3000、125000、45249和10353。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (a) Ratio on Mnist (b) Ratio on Sift (c) Ratio on LabelMe (d) Ratio on P53 -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_9.jpg?x=161&y=169&w=1464&h=264&r=0"/>

Figure 6: Overall Ratio of QALSH, C2LSH and LSB-Forest

图6：QALSH（快速近似最近邻哈希）、C2LSH（一种局部敏感哈希方法）和LSB - Forest（最低有效位森林）的总体比率

<!-- figureText: 1000 6000 $\begin{array}{llllllllll} {10} & {20} & {30} & {40} & {50} & {60} & {70} & {80} & {90} & {100} \end{array}$ (c) I/O on LabelMe (d) I/O on P53 $\begin{array}{llllllllll} {10} & {20} & {30} & {40} & {50} & {60} & {70} & {80} & {90} & {100} \end{array}$ $\begin{array}{lllllllllll} 1 & {10} & {20} & {30} & {40} & {50} & {60} & {70} & {80} & {90} & {100} \end{array}$ (a) I/O on Mnist (b) I/O on Sift -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_9.jpg?x=159&y=536&w=1469&h=262&r=0"/>

Figure 7: I/O Cost of QALSH, C2LSH and LSB-Forest

图7：QALSH（快速近似最近邻哈希）、C2LSH（一种局部敏感哈希方法）和LSB - Forest（最低有效位森林）的I/O成本

<!-- Media -->

The running time of QALSH is larger than that of LSB-Forest on the low-dimensional datasets, i.e., Mnist and Sift, but the searching accuracy of QALSH is much higher than that of LSB-Forest. Note that in this set of experiments, we set $c$ to 2.0 so that LSB-Forest can achieve the best performance. Actually, we can trade the accuracy of QALSH for efficiency by setting a larger $c$ value. More explanation will be given in Section 6.7.

在低维数据集（即Mnist（手写数字数据集）和Sift（尺度不变特征变换数据集））上，QALSH（快速近似最近邻哈希）的运行时间比LSB - Forest（最低有效位森林）长，但QALSH的搜索准确率比LSB - Forest高得多。请注意，在这组实验中，我们将$c$设置为2.0，以便LSB - Forest能够达到最佳性能。实际上，我们可以通过设置更大的$c$值来牺牲QALSH的准确率以提高效率。更多解释将在6.7节给出。

The running time of QALSH is consistently smaller than that of C2LSH on all the four datasets. Although QALSH may use more time in locating anchor buckets, its I/O cost is significantly smaller than that of C2LSH as shown in Figure 7. As the I/O cost is the main overhead, the total running time of QALSH is smaller than that of C2LSH.

在所有四个数据集上，QALSH（快速近似局部敏感哈希）的运行时间始终小于C2LSH（一种局部敏感哈希算法）。尽管QALSH在定位锚桶时可能会花费更多时间，但其I/O成本明显小于C2LSH，如图7所示。由于I/O成本是主要开销，因此QALSH的总运行时间小于C2LSH。

### 6.7 Performance vs. Approximation Ratio

### 6.7 性能与近似率

We study how approximation ratio $c$ affects the performance of QALSH. Due to space limitation, we only show results on Mnist and P53 in Figure 9. We observe similar trends from the results on the other two datasets.

我们研究了近似率 $c$ 如何影响QALSH（快速近似局部敏感哈希）的性能。由于篇幅限制，我们仅在图9中展示了Mnist（手写数字数据集）和P53（一种蛋白质相关数据集）的结果。我们从另外两个数据集的结果中也观察到了类似的趋势。

QALSH achieves better query quality with smaller $c$ value. From Figures 9(a) and 9(b), the overall ratio of QALSH decreases monotonically as $c$ decreases. When $c$ is set to 1.5, the overall ratio of QALSH is very close to 1.0, even for $k = {100}$ . This means,by using $c < {2.0}$ ,QALSH is able to return extremely accurate results. Meanwhile,when $c$ is set to 3.0, the overall ratios of QALSH on both datasets are still smaller than 1.07.

QALSH（查询感知局部敏感哈希）在 $c$ 值较小时能实现更好的查询质量。从图9(a)和9(b)可以看出，随着 $c$ 的减小，QALSH的总体比率单调下降。当 $c$ 设置为1.5时，即使对于 $k = {100}$ ，QALSH的总体比率也非常接近1.0。这意味着，通过使用 $c < {2.0}$ ，QALSH能够返回极其准确的结果。同时，当 $c$ 设置为3.0时，QALSH在两个数据集上的总体比率仍小于1.07。

From Figures 9(c) to 9(f), both the I/O cost and the running time of QALSH decrease monotonically as $c$ increases. Specifically,the I/O costs under setting $c = {3.0}$ are about ${25}\%$ and ${50}\%$ of the I/O costs under setting $c = {1.5}$ over the datasets Mnist and P53, respectively. Similar trends can be observed for the running time of QALSH. Therefore, under certain circumstances where the searching efficiency is a critical requirement, we can trade the accuracy of QALSH for efficiency by setting a larger $c$ value. For example,for the low-dimensional dataset Mnist, the running time of QALSH with $c = {3.0}$ is comparable to the running time of LSB-Forest shown in Figure 8(a), but the overall ratio of QALSH with $c = {3.0}$ is still much smaller than that of LSB-Forest.

从图9(c)到图9(f)可以看出，随着$c$的增大，QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）的输入/输出（I/O）成本和运行时间均单调递减。具体而言，在数据集Mnist和P53上，设置$c = {3.0}$时的I/O成本分别约为设置$c = {1.5}$时I/O成本的${25}\%$和${50}\%$。QALSH的运行时间也呈现出类似的趋势。因此，在搜索效率是关键要求的特定情况下，我们可以通过设置更大的$c$值来牺牲QALSH的准确性以换取效率。例如，对于低维数据集Mnist，设置$c = {3.0}$时QALSH的运行时间与图8(a)中LSB - 森林（Least Significant Bit Forest）的运行时间相当，但设置$c = {3.0}$时QALSH的整体比率仍远小于LSB - 森林的整体比率。

### 6.8 QALSH vs. C2LSH

### 6.8 QALSH与C2LSH（压缩碰撞局部敏感哈希，Compressed Collision Locality-Sensitive Hashing）的比较

We study the performance of QALSH and C2LSH on the four datasets by setting $m$ and $l$ of C2LSH to be the same as those of QALSH. Due to space limitation, we only show results on Mnist and P53 in Figure 10. Similar trends are observed from the results on the other two datasets.

我们通过将C2LSH的$m$和$l$设置为与QALSH相同，来研究QALSH（快速近似局部敏感哈希）和C2LSH（基于聚类的局部敏感哈希）在四个数据集上的性能。由于篇幅限制，我们仅在图10中展示了Mnist（手写数字数据集）和P53（一种蛋白质相关数据集）的结果。从另外两个数据集的结果中也观察到了类似的趋势。

From Figures 10(a) and 10(b), the overall ratio of QALSH is much smaller than that of C2LSH. In fact, by setting the same values of $m$ and $\beta$ ,the maximum value of $\left( {{\xi }_{1} - {\xi }_{2}}\right)$ is smaller than that of $\left( {{p}_{1} - {p}_{2}}\right)$ according to Equation 8 as discussed in Section 3.2. Hence,the error probability $\bar{\delta }$ of C2LSH is forced to increase. QALSH accordingly enjoys higher accuracy under the same $m$ and $l$ . From Figures 10(c) to ${10}\left( \mathrm{f}\right)$ ,QALSH also enjoys less $\mathrm{I}/\mathrm{O}$ cost and running time. For the special case in P53, the running time of C2LSH is slightly less than that of QALSH when $k \leq  {30}$ . This is because their I/O costs are close to each other, but QALSH needs more time in locating the anchor buckets.

从图10(a)和图10(b)可以看出，QALSH（查询感知局部敏感哈希，Query-Aware Locality-Sensitive Hashing）的总体比率远小于C2LSH（基于聚类的局部敏感哈希，Clustering-based Locality-Sensitive Hashing）。实际上，通过设置相同的$m$和$\beta$值，根据3.2节中讨论的公式8，$\left( {{\xi }_{1} - {\xi }_{2}}\right)$的最大值小于$\left( {{p}_{1} - {p}_{2}}\right)$的最大值。因此，C2LSH的错误概率$\bar{\delta }$被迫增加。在相同的$m$和$l$条件下，QALSH相应地具有更高的准确性。从图10(c)到${10}\left( \mathrm{f}\right)$，QALSH的$\mathrm{I}/\mathrm{O}$成本和运行时间也更少。对于P53中的特殊情况，当$k \leq  {30}$时，C2LSH的运行时间略少于QALSH。这是因为它们的输入/输出（I/O）成本相近，但QALSH在定位锚桶时需要更多时间。

We also study the performance of QALSH and C2LSH by setting $m$ and $l$ of QALSH to be the same as those of $\mathrm{C}2\mathrm{{LSH}}$ ,and observe similar trends from the results on the four datasets.

我们还通过将QALSH（查询感知局部敏感哈希）的$m$和$l$设置为与$\mathrm{C}2\mathrm{{LSH}}$相同的值来研究QALSH和C2LSH（一种局部敏感哈希算法）的性能，并从四个数据集的结果中观察到了相似的趋势。

### 6.9 Summary

### 6.9 总结

Based on the experiment results, we have the following findings. First, to achieve the same query quality, QALSH consumes much smaller space for index construction than C2LSH. In addition, QALSH is much more efficient than C2LSH since both its I/O cost and running time are much smaller than those of C2LSH. Second, when QALSH and C2LSH use the index of the same size, QALSH enjoys less $\mathrm{I}/\mathrm{O}$ cost and running time,and achieves higher accuracy. Third,QALSH works with any $c > 1$ . More accurate query results can be found by setting $c < {2.0}$ ,at the expense of $\mathrm{I}/\mathrm{O}$ . In contrast,LSB-Forest and $\mathrm{C}2\mathrm{{LSH}}$ only work with integer $c \geq  2$ . Finally,compared to LSB-Forest,QALSH uses much smaller index to achieve much higher accuracy, although it uses more $\mathrm{I}/\mathrm{O}$ and running time on low- and medium-dimensional datasets. For high-dimensional datasets, QALSH outperforms LSB-Forest in terms of all the four evaluation metrics. This is because data dimensionality affects LSB-Forest. In general, data dimensionality affects any method depending on space-filling curve such as Z-order.

基于实验结果，我们有以下发现。首先，为达到相同的查询质量，QALSH（量化近似局部敏感哈希）在索引构建方面所需的空间比C2LSH（一种局部敏感哈希方法）小得多。此外，QALSH比C2LSH效率高得多，因为它的I/O成本和运行时间都比C2LSH小得多。其次，当QALSH和C2LSH使用相同大小的索引时，QALSH的$\mathrm{I}/\mathrm{O}$成本和运行时间更低，并且准确率更高。第三，QALSH适用于任何$c > 1$。通过设置$c < {2.0}$可以找到更准确的查询结果，但会增加$\mathrm{I}/\mathrm{O}$成本。相比之下，LSB - Forest（最低有效位森林）和$\mathrm{C}2\mathrm{{LSH}}$仅适用于整数$c \geq  2$。最后，与LSB - Forest相比，QALSH使用小得多的索引就能实现高得多的准确率，尽管在低维和中维数据集上它会使用更多的$\mathrm{I}/\mathrm{O}$成本和运行时间。对于高维数据集，QALSH在所有四个评估指标方面都优于LSB - Forest。这是因为数据维度会影响LSB - Forest。一般来说，数据维度会影响任何依赖于空间填充曲线（如Z序）的方法。

<!-- Media -->

<!-- figureText: $\begin{array}{lllllllllll} 1 & {10} & {20} & {30} & {40} & {50} & {60} & {70} & {80} & {90} & {100} \end{array}$ (c) Time on LabelMe (d) Time on P53 (a) Time on Mnist (b) Time on Sift -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_10.jpg?x=161&y=168&w=1466&h=266&r=0"/>

Figure 8: Running Time of QALSH, C2LSH and LSB-Forest

图8：QALSH、C2LSH和LSB - 森林的运行时间

Figure 9: Performance of QALSH vs. $c$

图9：QALSH与$c$的性能对比

<!-- figureText: (a) Mnist, Ratio vs. c (b) P53, Ratio vs. c (e) Mnist, Time vs. c (b) Ratio on P53 $\begin{array}{lllllllllll} 1 & {10} & {20} & {30} & {40} & {50} & {60} & {70} & {80} & {90} & {100} \end{array}$ (c) I/O on Mnist (e) Time on Mnist (f) Time on P53 -->

<img src="https://cdn.noedgeai.com/01957c01-1383-7a6f-9eb5-42cce97a32f7_10.jpg?x=155&y=553&w=1487&h=795&r=0"/>

<!-- Media -->

## 7. RELATED WORK

## 7. 相关工作

LSH functions are first introduced for use in Hamming space by Indyk and Motwani [7. LSH functions based on $p$ -stable distribution in Euclidean space are introduced by Datar et al. 2, which leads to E2LSH for processing memory dataset. E2LSH builds physical hash tables for a series of search radii, and hence results in a big consumption of storage space. One space saving alternative is to use a single "magic" radius to process different queries [5]. However, such a "magic" radius is hard to decide [15].

Indyk和Motwani首次引入LSH（局部敏感哈希）函数用于汉明空间 [7]。Datar等人引入了基于欧几里得空间中$p$ - 稳定分布的LSH函数 [2]，由此产生了用于处理内存数据集的E2LSH（欧几里得空间局部敏感哈希）。E2LSH为一系列搜索半径构建物理哈希表，因此会消耗大量的存储空间。一种节省空间的替代方法是使用单个“神奇”半径来处理不同的查询 [5]。然而，这样的“神奇”半径很难确定 [15]。

<!-- Media -->

Figure 10: QALSH vs. C2LSH

图10：QALSH与C2LSH的对比

<!-- Media -->

Virtual rehashing is implicitly or explicitly used in LSB-Forest [15] and C2LSH [4] to avoid building physical hash tables for each search radius. Virtual rehashing used in QALSH is much simpler and more effective than that of C2LSH due to the use of query-aware LSH function. Specifically, virtual rehashing of QALSH does not involve any random shift and floor function, and is carried out in a symmetrical manner. LSB-Forest, C2LSH and QALSH all have theoretical guarantee on query quality. Recently, a variant of LSB-Forest named SK-LSH [11] exploits linear order instead of Z-order for encoding hash values, without any theoretical guarantee on query quality.

在LSB - 森林（LSB - Forest）[15]和C2LSH [4]中隐式或显式地使用了虚拟重哈希（Virtual rehashing），以避免为每个搜索半径构建物理哈希表。由于使用了查询感知局部敏感哈希（LSH）函数，QALSH中使用的虚拟重哈希比C2LSH中的要简单得多且更有效。具体而言，QALSH的虚拟重哈希不涉及任何随机移位和向下取整函数，并且是以对称方式进行的。LSB - 森林（LSB - Forest）、C2LSH和QALSH在查询质量上都有理论保证。最近，一种名为SK - LSH [11]的LSB - 森林（LSB - Forest）变体采用线性顺序而非Z顺序来编码哈希值，但其查询质量没有任何理论保证。

An LSH function for Euclidean space, no matter query-oblivious or query-aware, involves random projection. Random projection is also used in MEDRANK [3] to project objects over a set of $m$ random lines. However,MEDRANK does not segment a random line into buckets. An object that is found closest to a query along at least $\frac{m}{2}$ random lines,is reported as the $c$ -ANN of the query. The median threshold of $\frac{m}{2}$ is generalized by collision threshold for finding frequent objects in both C2LSH and QALSH. A classic result on random projection is the Johnson-Lindenstrass Lemma 9,which states that by projecting objects in $d$ - dimensional Euclidean space along $m$ random lines,the distance in the original $d$ -dimensions can be approximately preserved in the $m$ -dimensions. In a recent work on LSH for memory dataset in Euclidean space, Andoni et al. 1 propose to replace random projection (i.e., data-oblivious projection) by data-aware projection. However, the LSH scheme is still query-oblivious. Recently, Sun et al. 14 introduce another projection-based method named SRS. SRS uses only 6 random projections to convert high-dimensional data objects into low-dimensional ones so that they can be indexed by a single $R$ -tree. While C2LSH has better overall ratio than SRS, SRS uses a rather small index and also incurs much less I/O cost. Since SRS exploits only 6 to 10 random projections, it is natural to expect one is able to perform several groups of such projections. However, it is not clear which group of projections in SRS would lead to the best overall ratio. Intuitively, SRS is less stable than C2LSH and QALSH, since SRS is based on less than 10 projections but C2LSH and QALSH take advantage of more projections.

欧几里得空间的局部敏感哈希（LSH）函数，无论是否与查询无关，都涉及随机投影。随机投影也被用于MEDRANK [3]中，以将对象投影到一组$m$随机线上。然而，MEDRANK并不将随机线分割成桶。沿着至少$\frac{m}{2}$条随机线被发现与查询最接近的对象，会被报告为查询的$c$ -近似最近邻（ANN）。在C2LSH和QALSH中，用于查找频繁对象的碰撞阈值推广了$\frac{m}{2}$的中位数阈值。关于随机投影的一个经典结果是约翰逊 - 林登斯特劳斯引理9，该引理指出，通过将$d$维欧几里得空间中的对象沿着$m$条随机线进行投影，原始$d$维空间中的距离可以在$m$维空间中近似保留。在最近一项关于欧几里得空间中内存数据集的LSH研究中，安多尼（Andoni）等人[1]提议用数据感知投影取代随机投影（即与数据无关的投影）。然而，该LSH方案仍然与查询无关。最近，孙（Sun）等人[14]引入了另一种基于投影的方法，名为SRS。SRS仅使用6次随机投影将高维数据对象转换为低维对象，以便可以用单个$R$ -树对它们进行索引。虽然C2LSH的整体比率比SRS好，但SRS使用的索引相当小，并且I/O成本也低得多。由于SRS仅利用6到10次随机投影，自然可以预期能够执行几组这样的投影。然而，尚不清楚SRS中的哪一组投影会导致最佳的整体比率。直观地说，SRS比C2LSH和QALSH更不稳定，因为SRS基于少于10次投影，而C2LSH和QALSH利用了更多的投影。

## 8. CONCLUSIONS

## 8. 结论

In this paper, we introduce a novel concept of query-aware LSH function and accordingly propose a novel LSH scheme QALSH for $c$ -ANN search in high-dimensional Euclidean space. A query-aware LSH function is a random projection coupled with query-aware bucket partition. The function needs no random shift that is a prerequisite of traditional LSH functions. Query-aware LSH functions also enables QALSH to work with any approximation ratio $c > 1$ . In contrast, the state-of-the-art LSH schemes such as C2LSH and LSB-Forest only work with integer $c \geq  2$ . Our theoretical analysis shows that QALSH achieves a quality guarantee for the $c$ -ANN search. We also propose an automatic way to decide the bucket width $w$ used in QALSH. Experimental results on four real datasets demonstrate that QALSH outperforms C2LSH and LSB-Forest, especially in high-dimensional space.

在本文中，我们引入了查询感知局部敏感哈希（LSH）函数的全新概念，并相应地提出了一种新颖的LSH方案——查询感知局部敏感哈希（QALSH），用于高维欧几里得空间中的$c$ - 近似最近邻（ANN）搜索。查询感知LSH函数是一种结合了查询感知桶划分的随机投影。该函数不需要随机偏移，而随机偏移是传统LSH函数的一个先决条件。查询感知LSH函数还使QALSH能够适用于任何近似比率$c > 1$ 。相比之下，诸如C2LSH和LSB - 森林等最先进的LSH方案仅适用于整数$c \geq  2$ 。我们的理论分析表明，QALSH为$c$ - ANN搜索提供了质量保证。我们还提出了一种自动确定QALSH中使用的桶宽度$w$ 的方法。在四个真实数据集上的实验结果表明，QALSH的性能优于C2LSH和LSB - 森林，尤其是在高维空间中。

## 9. ACKNOWLEDGMENTS

## 9. 致谢

This work is partially supported by China NSF Grant 60970043, HKUST FSGRF13EG22 and FSGRF14EG31. We thank Wei Wang (UNSW) for his insightful comments.

本工作得到了中国国家自然科学基金（China NSF）项目60970043、香港科技大学（HKUST）前沿研究种子基金（FSGRF）13EG22和14EG31的部分资助。我们感谢新南威尔士大学（UNSW）的王巍提出的深刻见解。

## 10. REFERENCES

## 10. 参考文献

[1] A. Andoni, P. Indyk, H. L. Nguyen, and I. Razenshteyn. Beyond locality-sensitive hashing. In SODA, pages 1018-1028, 2014.

[1] A. 安多尼（A. Andoni）、P. 因迪克（P. Indyk）、H. L. 阮（H. L. Nguyen）和I. 拉曾施泰因（I. Razenshteyn）。超越局部敏感哈希。收录于《离散算法研讨会论文集》（SODA），第1018 - 1028页，2014年。

[2] M. Datar, N. Immorlica, P. Indyk, and V. S. Mirrokni. Locality-sensitive hashing scheme based on p-stable distributions. In ${SoCG}$ ,pages ${253} - {262},{2004}$ .

[2] M. 达塔尔（M. Datar）、N. 伊莫利卡（N. Immorlica）、P. 因迪克（P. Indyk）和V. S. 米罗克尼（V. S. Mirrokni）。基于p - 稳定分布的局部敏感哈希方案。收录于${SoCG}$，第${253} - {262},{2004}$页。

[3] R. Fagin, R. Kumar, and D. Sivakumar. Efficient similarity search and classification via rank aggregation. In ${ACM}{SIGMOD}$ ,pages 301-312,2003.

[3] R. 法金（R. Fagin）、R. 库马尔（R. Kumar）和D. 西瓦库马尔（D. Sivakumar）。通过排名聚合实现高效的相似性搜索和分类。收录于${ACM}{SIGMOD}$，第301 - 312页，2003年。

[4] J. Gan, J. Feng, Q. Fang, and W. Ng. Locality-sensitive hashing scheme based on dynamic collision counting. In SIGMOD, pages 541-552, 2012.

[4] 甘（Gan）、冯（Feng）、方（Fang）和吴（Ng）。基于动态碰撞计数的局部敏感哈希方案。发表于《管理数据的特别兴趣小组会议论文集》（SIGMOD），第541 - 552页，2012年。

[5] A. Gionis, P. Indyk, R. Motwani, et al. Similarity search in high dimensions via hashing. In VLDB, volume 99, pages 518-529. VLDB Endowment, 1999.

[5] 吉奥尼斯（Gionis）、因迪克（Indyk）、莫特瓦尼（Motwani）等人。通过哈希进行高维空间中的相似性搜索。发表于《大型数据库会议论文集》（VLDB），第99卷，第518 - 529页。大型数据库会议基金组织（VLDB Endowment），1999年。

[6] W. Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13-30, 1963.

[6] 霍夫丁（Hoeffding）。有界随机变量之和的概率不等式。《美国统计协会杂志》（Journal of the American Statistical Association），58(301):13 - 30，1963年。

[7] P. Indyk and R. Motwani. Approximate nearest neighbors: towards removing the curse of dimensionality. In ${ACM}\;{STOC}$ ,pages ${604} - {613},{1998}$ .

[7] 因迪克（Indyk）和莫特瓦尼（Motwani）。近似最近邻：消除维度灾难的探索。发表于${ACM}\;{STOC}$，第${604} - {613},{1998}$页。

[8] H. Jagadish, B. C. Ooi, K.-L. Tan, C. Yu, and R. Zhang. idistance: an adaptive b+-tree based indexing method for nearest neighbor search. ${ACM}$ TODS, 30(2):364-397, 2005.

[8] H. 贾加迪什（H. Jagadish）、B. C. 奥伊（B. C. Ooi）、K.-L. 谭（K.-L. Tan）、C. 于（C. Yu）和 R. 张（R. Zhang）。idistance：一种基于自适应B + 树的最近邻搜索索引方法。${ACM}$ 《ACM数据库系统汇刊》（TODS），30(2):364 - 397，2005年。

[9] W. Johnson and J. Lindenstrauss. Extensions of lipshitz mapping into hilbert space. Contemporary Mathematics, 26:189-206, 1984.

[9] W. 约翰逊（W. Johnson）和 J. 林登施特劳斯（J. Lindenstrauss）。利普希茨映射到希尔伯特空间的扩展。《当代数学》（Contemporary Mathematics），26:189 - 206，1984年。

[10] J. M. Kleinberg. Two algorithms for nearest-neighbor search in high dimensions. In ${ACM}\;{STOC}$ ,pages 599-608, 1997.

[10] J. M. 克莱因伯格（J. M. Kleinberg）。高维空间中最近邻搜索的两种算法。见${ACM}\;{STOC}$，第599 - 608页，1997年。

[11] Y. Liu, J. Cui, Z. Huang, H. Li, and H. T. Shen. Sk-lsh: An efficient index structure for approximate nearest neighbor search. VLDB, 7(9), 2014.

[11] Y. 刘（Y. Liu）、J. 崔（J. Cui）、Z. 黄（Z. Huang）、H. 李（H. Li）和 H. T. 沈（H. T. Shen）。Sk - LSH：一种用于近似最近邻搜索的高效索引结构。《非常大数据库会议论文集》（VLDB），7(9)，2014年。

[12] R. Panigrahy. Entropy based nearest neighbor search in high dimensions. In ${ACM} - {SIAM}\;{SODA}$ ,pages 1186-1195, 2006.

[12] R. 帕尼格拉希（R. Panigrahy）。基于熵的高维最近邻搜索。见${ACM} - {SIAM}\;{SODA}$，第1186 - 1195页，2006年。

[13] H. Samet. Foundations of multidimensional and metric data structures. Morgan Kaufmann, 2006.

[13] H. 萨梅特（H. Samet）。多维和度量数据结构基础。摩根·考夫曼出版社（Morgan Kaufmann），2006年。

[14] Y. Sun, W. Wang, J. Qin, Y. Zhang, and X. Lin. Srs: Solving c-approximate nearest neighbor queries in high dimensional euclidean space with a tiny index. VLDB, 8(1), 2014.

[14] 孙Y（Y. Sun）、王W（W. Wang）、秦J（J. Qin）、张Y（Y. Zhang）和林X（X. Lin）。Srs：使用微小索引解决高维欧几里得空间中的c - 近似最近邻查询。《非常大数据库会议论文集》（VLDB），8(1)，2014年。

[15] Y. Tao, K. Yi, C. Sheng, and P. Kalnis. Efficient and accurate nearest neighbor and closest pair search in high-dimensional space. ACM TODS, 35(3):20, 2010.

[15] 陶Y（Y. Tao）、易K（K. Yi）、盛C（C. Sheng）和卡尔尼斯P（P. Kalnis）。高维空间中高效准确的最近邻和最近点对搜索。《美国计算机协会数据库系统汇刊》（ACM TODS），35(3):20，2010年。

## APPENDIX

## 附录

## A. PROOF OF LEMMA 3

## A. 引理3的证明

Proof. Before bounding $\Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack$ from below and hence proving Lemma 3, we have to prove lower bounds on ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ .

证明。在从下方界定$\Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack$并进而证明引理3之前，我们必须先证明${\mathcal{P}}_{1}$和${\mathcal{P}}_{2}$的下界。

We now show some details of proving $\Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   \geq  1 - \delta$ . Let ${S}_{1} = \{ o \mid  \parallel o - q\parallel  \leq  R\}$ . For $\forall o \in  {S}_{1},\Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   =$ $\Pr \left\lbrack  {\# \operatorname{Col}\left( o\right)  \geq  {\alpha m}}\right\rbrack   = \mathop{\sum }\limits_{{i = \lceil {\alpha m}\rceil }}^{m}{C}_{m}^{i}{p}^{i}{\left( 1 - p\right) }^{m - i}$ ,where $p =$ $\Pr \left\lbrack  {\left| {{H}_{\overrightarrow{{a}_{j}}}^{R}\left( o\right)  - {H}_{\overrightarrow{{a}_{j}}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}}\right\rbrack   \geq  {p}_{1} > \alpha ,j = 1,2,\ldots ,m$ . Then by following the same reasoning based on Hoeffding's Inequality 6 from Lemma 1 of C2LSH,we have $\Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   \geq$ $1 - \delta$ ,when $m = \left\lceil  {\max \left( {\frac{1}{2{\left( {p}_{1} - \alpha \right) }^{2}}\ln \frac{1}{\delta },\frac{1}{2{\left( \alpha  - {p}_{2}\right) }^{2}}\ln \frac{2}{\beta }}\right) }\right\rceil$ .

我们现在展示证明$\Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   \geq  1 - \delta$的一些细节。设${S}_{1} = \{ o \mid  \parallel o - q\parallel  \leq  R\}$。对于$\forall o \in  {S}_{1},\Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   =$ $\Pr \left\lbrack  {\# \operatorname{Col}\left( o\right)  \geq  {\alpha m}}\right\rbrack   = \mathop{\sum }\limits_{{i = \lceil {\alpha m}\rceil }}^{m}{C}_{m}^{i}{p}^{i}{\left( 1 - p\right) }^{m - i}$，其中$p =$ $\Pr \left\lbrack  {\left| {{H}_{\overrightarrow{{a}_{j}}}^{R}\left( o\right)  - {H}_{\overrightarrow{{a}_{j}}}^{R}\left( q\right) }\right|  \leq  \frac{w}{2}}\right\rbrack   \geq  {p}_{1} > \alpha ,j = 1,2,\ldots ,m$。然后根据C2LSH引理1中的霍夫丁不等式6（Hoeffding's Inequality 6）进行相同的推理，当$m = \left\lceil  {\max \left( {\frac{1}{2{\left( {p}_{1} - \alpha \right) }^{2}}\ln \frac{1}{\delta },\frac{1}{2{\left( \alpha  - {p}_{2}\right) }^{2}}\ln \frac{2}{\beta }}\right) }\right\rceil$时，我们有$\Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   \geq$ $1 - \delta$。

Similarly,using the same $m$ ,we have $\Pr \left\lbrack  {\mathcal{P}}_{2}\right\rbrack   > \frac{1}{2}$ .

同样地，使用相同的 $m$，我们有 $\Pr \left\lbrack  {\mathcal{P}}_{2}\right\rbrack   > \frac{1}{2}$。

For the(R,c)-NN search,since QALSH terminates when either ${\mathcal{P}}_{1}$ or ${\mathcal{P}}_{2}$ holds,we have $\Pr \left\lbrack  {{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\right\rbrack   = 1$ . We also have the formula: $\Pr \left\lbrack  {{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\right\rbrack   = \Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   + \Pr \left\lbrack  {\mathcal{P}}_{2}\right\rbrack   - \Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack$ . Therefore,we can bound $\Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack$ from below as follows:

对于(R,c)-最近邻搜索，由于QALSH（快速近似局部敏感哈希）在${\mathcal{P}}_{1}$或${\mathcal{P}}_{2}$成立时终止，我们有$\Pr \left\lbrack  {{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\right\rbrack   = 1$。我们还有公式：$\Pr \left\lbrack  {{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\right\rbrack   = \Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   + \Pr \left\lbrack  {\mathcal{P}}_{2}\right\rbrack   - \Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack$。因此，我们可以如下从下方界定$\Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack$：

$$
\Pr \left\lbrack  {{\mathcal{P}}_{1} \cap  {\mathcal{P}}_{2}}\right\rbrack   = \Pr \left\lbrack  {\mathcal{P}}_{1}\right\rbrack   + \Pr \left\lbrack  {\mathcal{P}}_{2}\right\rbrack   - \Pr \left\lbrack  {{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\right\rbrack  
$$

$$
 \geq  1 - \delta  + \frac{1}{2} - 1 = \frac{1}{2} - \delta 
$$

And hence Lemma 3 is proved.

因此，引理3得证。