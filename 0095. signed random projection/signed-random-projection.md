# Similarity Estimation Techniques from Rounding Algorithms

基于舍入算法的相似性估计技术

Moses S. Charikar

摩西·S·查理卡

Dept. of Computer Science

计算机科学系

Princeton University

普林斯顿大学

35 Olden Street

奥尔登街35号

Princeton, NJ 08544

新泽西州普林斯顿 08544

moses@cs.princeton.edu

## ABSTRACT

## 摘要

A locality sensitive hashing scheme is a distribution on a family $\mathcal{F}$ of hash functions operating on a collection of objects,such that for two objects $x,y$ ,

局部敏感哈希方案是作用于对象集合的哈希函数族$\mathcal{F}$上的概率分布，使得对于两个对象$x,y$，

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   = \operatorname{sim}\left( {x,y}\right) ,
$$

where $\operatorname{sim}\left( {x,y}\right)  \in  \left\lbrack  {0,1}\right\rbrack$ is some similarity function defined on the collection of objects. Such a scheme leads to a compact representation of objects so that similarity of objects can be estimated from their compact sketches, and also leads to efficient algorithms for approximate nearest neighbor search and clustering. Min-wise independent permutations provide an elegant construction of such a locality sensitive hashing scheme for a collection of subsets with the set similarity measure $\operatorname{sim}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\left| A \cup  B\right| }$ .

其中$\operatorname{sim}\left( {x,y}\right)  \in  \left\lbrack  {0,1}\right\rbrack$是在对象集合上定义的相似性函数。该方案能生成对象的紧凑表示，从而通过其压缩概要估算对象相似度，并为近似最近邻搜索和聚类提供高效算法。最小独立置换为采用集合相似度度量$\operatorname{sim}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\left| A \cup  B\right| }$的子集集合，提供了一种优雅的局部敏感哈希构造方案。

We show that rounding algorithms for LPs and SDPs used in the context of approximation algorithms can be viewed as locality sensitive hashing schemes for several interesting collections of objects. Based on this insight, we construct new locality sensitive hashing schemes for:

我们证明，近似算法中使用的线性规划和半定规划舍入算法，可视为针对多个有趣对象集合的局部敏感哈希方案。基于此洞见，我们构建了新的局部敏感哈希方案：

1. A collection of vectors with the distance between $\overrightarrow{u}$ and $\overrightarrow{v}$ measured by $\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right) /\pi$ ,where $\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right)$ is the angle between $\overrightarrow{u}$ and $\overrightarrow{v}$ . This yields a sketching scheme for estimating the cosine similarity measure between two vectors, as well as a simple alternative to minwise independent permutations for estimating set similarity.

1. 向量集合中$\overrightarrow{u}$与$\overrightarrow{v}$的距离通过$\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right) /\pi$度量，其中$\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right)$表示两者夹角。这产生了用于估计向量间余弦相似度的概要方案，以及替代最小独立置换来估算集合相似度的简易方法。

2. A collection of distributions on $n$ points in a metric space, with distance between distributions measured by the Earth Mover Distance (EMD), (a popular distance measure in graphics and vision). Our hash functions map distributions to points in the metric space such that,for distributions $P$ and $Q$ ,

2. 度量空间中$n$个点上分布的集合，采用推土机距离(EMD)（图形学与视觉领域常用度量）衡量分布间距离。我们的哈希函数将分布映射至度量空间中的点，使得对于分布$P$和$Q$，

$$
\mathbf{{EMD}}\left( {P,Q}\right)  \leq  {\mathbf{E}}_{h \in  \mathcal{F}}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack  
$$

$$
 \leq  O\left( {\log n\log \log n}\right)  \cdot  \mathbf{{EMD}}\left( {P,Q}\right) .
$$

## 1. INTRODUCTION

## 1. 引言

The current information explosion has resulted in an increasing number of applications that need to deal with large volumes of data. While traditional algorithm analysis assumes that the data fits in main memory, it is unreasonable to make such assumptions when dealing with massive data sets such as data from phone calls collected by phone companies, multimedia data, web page repositories and so on. This new setting has resulted in an increased interest in algorithms that process the input data in restricted ways, including sampling a few data points, making only a few passes over the data, and constructing a succinct sketch of the input which can then be efficiently processed.

当前信息爆炸导致需要处理海量数据的应用激增。传统算法分析假设数据可存入主存，但处理电话公司收集的通话记录、多媒体数据、网页库等大规模数据集时，这种假设已不成立。这一新形势促使人们更关注以受限方式处理输入数据的算法，包括采样少量数据点、仅对数据进行少量遍历，以及构建可高效处理的输入压缩概要。

There has been a lot of recent work on streaming algorithms, i.e. algorithms that produce an output by making one pass (or a few passes) over the data while using a limited amount of storage space and time. To cite a few examples, Alon et al [2] considered the problem of estimating frequency moments and Guha et al [25] considered the problem of clustering points in a streaming fashion. Many of these streaming algorithms need to represent important aspects of the data they have seen so far in a small amount of space; in other words they maintain a compact sketch of the data that encapsulates the relevant properties of the data set. Indeed, some of these techniques lead to sketching algorithms - algorithms that produce a compact sketch of a data set so that various measurements on the original data set can be estimated by efficient computations on the compact sketches. Building on the ideas of [2], Alon et al [1] give algorithms for estimating join sizes. Gibbons and Matias [18] give sketching algorithms producing so called synopsis data structures for various problems including maintaining approximate histograms, hot lists and so on. Gilbert et al [19] give algorithms to compute sketches for data streams so as to estimate any linear projection of the data and use this to get individual point and range estimates. Recently, Gilbert et al [21] gave efficient algorithms for the dynamic maintenance of histograms. Their algorithm processes a stream of updates and maintains a small sketch of the data from which the optimal histogram representation can be approximated very quickly.

近年来涌现大量流算法研究，即使用有限存储空间和时间，通过单次（或少量）遍历数据生成输出的算法。例如阿隆等人[2]研究频率矩估计问题，古哈等人[25]探索流式聚类问题。许多流算法需要用少量空间表示已观测数据的关键特征，即维护封装数据集相关属性的压缩概要。事实上，部分技术催生了概要算法——生成数据集紧凑概要，使得通过对概要的高效计算可估计原始数据的各种度量。基于[2]思想，阿隆等人[1]提出连接大小估计算法。吉本斯和马蒂亚斯[18]针对近似直方图维护、热点列表等问题，提出生成所谓概要数据结构的算法。吉尔伯特等人[19]给出数据流概要计算算法，用于估计数据的任意线性投影，进而获得单点与范围估计。近期，吉尔伯特等人[21]提出动态维护直方图的高效算法，该算法处理更新流并维护数据的小型概要，可据此快速近似最优直方图表示。

In this work, we focus on sketching algorithms for estimating similarity, i.e. the construction of functions that produce succinct sketches of objects in a collection, such that the similarity of objects can be estimated efficiently from their sketches. Here,similarity $\operatorname{sim}\left( {x,y}\right)$ is a function that maps pairs of objects $x,y$ to a number in $\left\lbrack  {0,1}\right\rbrack$ ,measuring the degree of similarity between $x$ and $y.\operatorname{sim}\left( {x,y}\right)  = 1$ corresponds to objects $x,y$ that are identical while $\operatorname{sim}\left( {x,y}\right)  = 0$ corresponds to objects that are very different.

本研究中，我们专注于相似性估计的草图算法，即构建能生成集合中对象简洁草图（sketch）的函数，使得对象间的相似度可通过其草图高效估算。此处的相似性$\operatorname{sim}\left( {x,y}\right)$是将对象对$x,y$映射到$\left\lbrack  {0,1}\right\rbrack$区间数值的函数，当$x$与$y.\operatorname{sim}\left( {x,y}\right)  = 1$对应完全相同的对象$x,y$时取最大值，$\operatorname{sim}\left( {x,y}\right)  = 0$则对应差异极大的对象。

Broder et al $\left\lbrack  {8,5,7,6}\right\rbrack$ introduced the notion of min-wise independent permutations, a technique for constructing such sketching functions for a collection of sets. The similarity measure considered there was

Broder等人$\left\lbrack  {8,5,7,6}\right\rbrack$提出了最小独立排列（min-wise independent permutations）的概念，这是一种为集合构建草图函数的技术。其所考虑的相似性度量是

$$
\operatorname{sim}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\left| A \cup  B\right| }.
$$

We note that this is exactly the Jaccard coefficient of similarity used in information retrieval.

我们注意到这正是信息检索中使用的杰卡德相似系数（Jaccard coefficient）。

The min-wise independent permutation scheme allows the construction of a distribution on hash functions $h : {2}^{U} \rightarrow  U$ such that

最小独立排列方案允许构建哈希函数$h : {2}^{U} \rightarrow  U$的概率分布，使得

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( A\right)  = h\left( B\right) }\right\rbrack   = \operatorname{sim}\left( {A,B}\right) .
$$

Here $\mathcal{F}$ denotes the family of hash functions (with an associated probability distribution) operating on subsets of the universe $U$ . By choosing say $t$ hash functions ${h}_{1},\ldots {h}_{t}$ from this family,a set $S$ could be represented by the hash vector $\left( {{h}_{1}\left( S\right) ,\ldots {h}_{t}\left( S\right) }\right)$ . Now,the similarity between two sets can be estimated by counting the number of matching coordinates in their corresponding hash vectors. ${}^{1}$

此处$\mathcal{F}$表示作用于全域$U$子集的哈希函数族（附带概率分布）。通过从该族中选择$t$个哈希函数${h}_{1},\ldots {h}_{t}$，集合$S$可表示为哈希向量$\left( {{h}_{1}\left( S\right) ,\ldots {h}_{t}\left( S\right) }\right)$。此时，两集合的相似度可通过比较对应哈希向量中匹配坐标的数量来估算。${}^{1}$

The work of Broder et al was originally motivated by the application of eliminating near-duplicate documents in the Altavista index. Representing documents as sets of features with similarity between sets determined as above, the hashing technique provided a simple method for estimating similarity of documents, thus allowing the original documents to be discarded and reducing the input size significantly.

Broder等人的研究最初源于Altavista索引中消除近重复文档的应用。将文档表示为特征集合后，采用上述集合相似度判定方法，该哈希技术为文档相似度估算提供了简洁方案，从而允许丢弃原始文档并大幅缩减输入规模。

In fact, the minwise independent permutations hashing scheme is a particular instance of a locality sensitive hashing scheme introduced by Indyk and Motwani [31] in their work on nearest neighbor search in high dimensions.

事实上，最小独立排列哈希方案是Indyk与Motwani[31]在高维最近邻搜索研究中提出的局部敏感哈希（locality sensitive hashing）方案的特例。

DEFINITION 1. A locality sensitive hashing scheme is a distribution on a family $\mathcal{F}$ of hash functions operating on a collection of objects,such that for two objects $x,y$ ,

定义1. 局部敏感哈希方案是在作用于对象集合的哈希函数族$\mathcal{F}$上的概率分布，使得对于任意两个对象$x,y$满足

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   = \operatorname{sim}\left( {x,y}\right)  \tag{1}
$$

Here $\operatorname{sim}\left( {x,y}\right)$ is some similarity function defined on the collection of objects.

此处$\operatorname{sim}\left( {x,y}\right)$是定义在对象集合上的某种相似性函数。

Given a hash function family $\mathcal{F}$ that satisfies (1),we will say that $\mathcal{F}$ is a locality sensitive hash function family corresponding to similarity function $\operatorname{sim}\left( {x,y}\right)$ . Indyk and Mot-wani showed that such a hashing scheme facilitates the construction of efficient data structures for answering approximate nearest-neighbor queries on the collection of objects.

给定满足(1)的哈希函数族$\mathcal{F}$，我们称$\mathcal{F}$是对应相似函数$\operatorname{sim}\left( {x,y}\right)$的局部敏感哈希函数族。Indyk与Motwani证明此类哈希方案能有效构建用于回答对象集合近似最近邻查询的数据结构。

In particular, using the hashing scheme given by minwise independent permutations results in efficient data structures for set similarity queries and leads to efficient clustering algorithms. This was exploited later in several experimental papers: Cohen et al [14] for association-rule mining, Haveli-wala et al [27] for clustering web documents, Chen et al [13] for selectivity estimation of boolean queries, Chen et al [12] for twig queries,and Gionis et al [22] for indexing set value attributes. All of this work used the hashing technique for set similarity together with ideas from [31].

特别地，采用最小独立排列的哈希方案可构建高效的集合相似性查询数据结构，并衍生出高效聚类算法。该技术后续被多篇实验论文应用：Cohen等[14]用于关联规则挖掘，Haveliwala等[27]用于网页文档聚类，Chen等[13]用于布尔查询选择性估计，Chen等[12]用于枝状查询，Gionis等[22]用于集合值属性索引。这些研究均结合了[31]的哈希技术与集合相似性方法。

We note that the definition of locality sensitive hashing used by [31] is slightly different, although in the same spirit as our definition. Their definition involves parameters ${r}_{1} >$ ${r}_{2}$ and ${p}_{1} > {p}_{2}$ . A family $\mathcal{F}$ is said to be $\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$ - sensitive for a similarity measure $\operatorname{sim}\left( {x,y}\right)$ if $\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{h \in  \mathcal{F}}}\lbrack h\left( x\right)  =$ $h\left( y\right) \rbrack  \geq  {p}_{1}$ when $\operatorname{sim}\left( {x,y}\right)  \geq  {r}_{1}$ and $\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \leq$ ${p}_{2}$ when $\operatorname{sim}\left( {x,y}\right)  \leq  {r}_{2}$ . Despite the difference in the precise definition, we chose to retain the name locality sensitive hashing in this work since the two notions are essentially the same. Hash functions with closely related properties were investigated earlier by Linial and Sasson [34] and Indyk et al $\left\lbrack  {32}\right\rbrack$ .

我们注意到[31]所使用的局部敏感哈希定义虽与我们的定义精神一致，但存在细微差异。他们的定义涉及参数${r}_{1} >$${r}_{2}$和${p}_{1} > {p}_{2}$。若满足$\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{h \in  \mathcal{F}}}\lbrack h\left( x\right)  =$$h\left( y\right) \rbrack  \geq  {p}_{1}$当$\operatorname{sim}\left( {x,y}\right)  \geq  {r}_{1}$且$\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \leq$${p}_{2}$当$\operatorname{sim}\left( {x,y}\right)  \leq  {r}_{2}$时，则称族$\mathcal{F}$对于相似性度量$\operatorname{sim}\left( {x,y}\right)$是$\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$敏感的。尽管精确定义存在差异，我们选择在本研究中保留"局部敏感哈希"这一名称，因为两者的核心概念本质相同。Linial和Sasson[34]以及Indyk等人$\left\lbrack  {32}\right\rbrack$早前已研究过具有密切关联特性的哈希函数。

### 1.1 Our Results

### 1.1 研究成果

In this paper, we explore constructions of locality sensitive hash functions for various other interesting similarity functions. The utility of such hash function schemes (for nearest neighbor queries and clustering) crucially depends on the fact that the similarity estimation is based on a test of equality of the hash function values. We make an interesting connection between constructions of similarity preserving hash-functions and rounding procedures used in the design of approximation algorithms. We show that procedures used for rounding fractional solutions from linear programs and vector solutions to semidefinite programs can be used to derive similarity preserving hash functions for interesting classes of similarity functions.

本文探索了针对多种重要相似性函数的局部敏感哈希函数构造方法。此类哈希函数方案（用于最近邻查询和聚类）的实用性关键取决于相似性估计基于哈希函数值的相等性测试这一特性。我们揭示了相似性保持哈希函数构造与近似算法设计中舍入程序之间的有趣联系，证明线性规划分数解和半定规划向量解的舍入方法可用于推导特定相似函数类的相似性保持哈希函数。

In Section 2, we prove some necessary conditions on similarity measures $\operatorname{sim}\left( {x,y}\right)$ for the existence of locality sensitive hash functions satisfying (1). Using this, we show that such locality sensitive hash functions do not exist for certain commonly used similarity measures in information retrieval, the Dice coefficient and the Overlap coefficient.

第2节证明了相似性度量$\operatorname{sim}\left( {x,y}\right)$存在满足(1)式的局部敏感哈希函数的必要条件，据此表明信息检索中常用的Dice系数和Overlap系数等相似性度量不存在此类局部敏感哈希函数。

In seminal work, Goemans and Williamson [24] introduced semidefinite programming relaxations as a tool for approximation algorithms. They used the random hyperplane rounding technique to round vector solutions for the MAX-CUT problem. We will see in Section 3 that the random hyperplane technique naturally gives a family of hash functions $\mathcal{F}$ for vectors such that

Goemans和Williamson[24]的开创性工作首次将半定规划松弛引入近似算法设计，他们采用随机超平面舍入技术处理MAX-CUT问题的向量解。第3节将展示该技术自然导出一族向量哈希函数$\mathcal{F}$，使得...

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( \overrightarrow{u}\right)  = h\left( \overrightarrow{v}\right) }\right\rbrack   = 1 - \frac{\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right) }{\pi }.
$$

Here $\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right)$ refers to the angle between vectors $\overrightarrow{u}$ and $\overrightarrow{v}$ . Note that the function $1 - \frac{\theta }{\pi }$ is closely related to the function $\cos \left( \theta \right)$ . (In fact it is always within a factor 0.878 from it. Moreover, $\cos \left( \theta \right)$ can be estimated from an estimate of $\theta$ .) Thus this similarity function is very closely related to the cosine similarity measure, commonly used in information retrieval. (In fact, Indyk and Motwani [31] describe how the set similarity measure can be adapted to measure dot product between binary vectors in $d$ -dimensional Hamming space. Their approach breaks up the data set into $O\left( {\log d}\right)$ groups, each consisting of approximately the same weight. Our approach, based on estimating the angle between vectors is more direct and is also more general since it applies to general vectors.) We also note that the cosine between vectors can be estimated from known techniques based on random projections $\left\lbrack  {2,1,{20}}\right\rbrack$ . However,the advantage of a locality sensitive hashing based scheme is that this directly yields techniques for nearest neighbor search for the cosine similarity measure.

此处$\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right)$表示向量$\overrightarrow{u}$与$\overrightarrow{v}$间的夹角。注意函数$1 - \frac{\theta }{\pi }$与$\cos \left( \theta \right)$密切相关（实际偏差不超过0.878倍，且$\cos \left( \theta \right)$可通过$\theta$估计值推算）。因此该相似函数与信息检索常用的余弦相似度量高度相关（Indyk和Motwani[31]曾阐述如何调整集合相似度量以计算$d$维汉明空间中二值向量的点积，其方法将数据集分割为$O\left( {\log d}\right)$个权重近似的组别。而我们的向量夹角估计方法更直接且普适，适用于一般向量）。基于随机投影$\left\lbrack  {2,1,{20}}\right\rbrack$的现有技术虽可估计向量余弦值，但局部敏感哈希方案的优势在于能直接为余弦相似度量提供最近邻搜索技术。

---

<!-- Footnote -->

${}^{1}$ One question left open in [7] was the issue of compact representation of hash functions in this family; this was settled by Indyk [28], who gave a construction of a small family of minwise independent permutations.

${}^{1}$ 文献[7]中遗留的一个开放性问题，是关于该哈希函数族紧凑表示的问题；这一问题由Indyk[28]解决，他构建了一个小型的最小独立置换族。

<!-- Footnote -->

---

An attractive feature of the hash functions obtained from the random hyperplane method is that the output is a single bit; thus the output of $t$ hash functions can be concatenated very easily to produce a $t$ -bit vector. ${}^{2}$ Estimating similarity between vectors amounts to measuring the Hamming distance between the corresponding $t$ -bit hash vectors. We can represent sets by their characteristic vectors and use this locality sensitive hashing scheme for measuring similarity between sets. This yields a slightly different similarity measure for sets, one that is linearly proportional to the angle between their characteristic vectors.

通过随机超平面方法获得的哈希函数具有一个显著特点，即其输出为单比特；因此，$t$个哈希函数的输出可以非常容易地串联起来，生成一个$t$比特向量。${}^{2}$向量间相似度的估算，等同于测量对应$t$比特哈希向量间的汉明距离。我们可以用特征向量表示集合，并利用这种局部敏感哈希方案来度量集合间的相似度。这为集合提供了一种略有不同的相似度度量方式，该度量与它们特征向量间的夹角呈线性比例关系。

In Section 4, we present a locality sensitive hashing scheme for a certain metric on distributions on points, called the Earth Mover Distance. We are given a set of points $L =$ $\left\{  {{l}_{1},\ldots {l}_{n}}\right\}$ ,with a distance function $d\left( {i,j}\right)$ defined on them. A probability distribution $P\left( X\right)$ (or distribution for short) is a set of weights ${p}_{1},\ldots {p}_{n}$ on the points such that ${p}_{i} \geq  0$ and $\sum {p}_{i} = 1$ . (We will often refer to distribution $P\left( X\right)$ as simply $P$ ,implicitly referring to an underlying set $X$ of points.) The Earth Mover Distance $\mathbf{{EMD}}\left( {P,Q}\right)$ between two distributions $P$ and $Q$ is defined to be the cost of the min cost matching that transforms one distribution to another. (Imagine each distribution as placing a certain amount of earth on each point. $\mathbf{{EMD}}\left( {P,Q}\right)$ measures the minimum amount of work that must be done in transforming one distribution to the other.) This is a popular metric for images and is used for image similarity, navigating image databases and so on $\left\lbrack  {{37},{38},{39},{40},{36},{15},{16},{41},{42}}\right\rbrack$ . The idea is to represent an image as a distribution on features with an underlying distance metric on features (e.g. colors in a color spectrum). Since the earth mover distance is expensive to compute (requiring a solution to a minimum transportation problem), applications typically use an approximation of the earth mover distance. (e.g. representing distributions by their centroids).

在第4节中，我们针对点分布上的某种度量——称为推土机距离(Earth Mover Distance)，提出了一种局部敏感哈希方案。给定一组点$L =$$\left\{  {{l}_{1},\ldots {l}_{n}}\right\}$，以及定义在这些点上的距离函数$d\left( {i,j}\right)$。概率分布$P\left( X\right)$（简称分布）是点上的一组权重${p}_{1},\ldots {p}_{n}$，满足${p}_{i} \geq  0$且$\sum {p}_{i} = 1$。（我们常将分布$P\left( X\right)$简称为$P$，隐含指代基础点集$X$。）两个分布$P$与$Q$间的推土机距离$\mathbf{{EMD}}\left( {P,Q}\right)$，定义为将一个分布转换为另一个分布的最小成本匹配的成本。（想象每个分布都在各点上放置了一定量的土。$\mathbf{{EMD}}\left( {P,Q}\right)$衡量了将一个分布转换为另一个分布所需的最小工作量。）这是一种常用于图像的度量，用于图像相似性、图像数据库导航等$\left\lbrack  {{37},{38},{39},{40},{36},{15},{16},{41},{42}}\right\rbrack$。其思路是将图像表示为特征上的分布，这些特征具有基础的距离度量（如色谱中的颜色）。由于推土机距离计算成本高昂（需要解决最小运输问题），应用通常使用其近似值（例如通过质心表示分布）。

We construct a hash function family for estimating the earth mover distance. Our family is based on rounding algorithms for LP relaxations for the problem of classification with pairwise relationships studied by Kleinberg and Tar-dos [33], and further studied by Calinescu et al [10] and Chekuri et al [11]. Combining a new LP formulation described by Chekuri et al together with a rounding technique of Kleinberg and Tardos, we show a construction of a hash function family which approximates the earth mover distance to a factor of $O\left( {\log n\log \log n}\right)$ . Each hash function in this family maps a distribution on points $L = \left\{  {{l}_{1},\ldots ,{l}_{n}}\right\}$ to some point ${l}_{i}$ in the set. For two distributions $P\left( X\right)$ and $Q\left( X\right)$ on the set of points,our family of hash functions $\mathcal{F}$ satisfies the property that:

我们构建了一个用于估算推土机距离的哈希函数族。该函数族基于Kleinberg和Tardos[33]研究的带成对关系分类问题的LP松弛舍入算法，并得到Calinescu等人[10]和Chekuri等人[11]的进一步研究。结合Chekuri等人描述的新LP公式与Kleinberg和Tardos的舍入技术，我们展示了一个哈希函数族的构建，该族将推土机距离近似到$O\left( {\log n\log \log n}\right)$因子。此族中的每个哈希函数将点上的分布$L = \left\{  {{l}_{1},\ldots ,{l}_{n}}\right\}$映射到集合中的某个点${l}_{i}$。对于点集上的两个分布$P\left( X\right)$和$Q\left( X\right)$，我们的哈希函数族$\mathcal{F}$满足以下性质：

$$
\mathbf{{EMD}}\left( {P,Q}\right)  \leq  {\mathbf{E}}_{h \in  \mathcal{F}}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack  
$$

$$
 \leq  O\left( {\log n\log \log n}\right)  \cdot  \mathbf{{EMD}}\left( {P,Q}\right) .
$$

We also show an interesting fact about a rounding algorithm in Kleinberg and Tardos [33] applying to the case where the underlying metric on points is a uniform metric. In this case, we show that their rounding algorithm can be viewed as a generalization of min-wise independent permutations extended to a continuous setting. Their rounding procedure yields a locality sensitive hash function for vectors whose coordinates are all non-negative. Given two vectors $\overrightarrow{a} = \left( {{a}_{1},\ldots {a}_{n}}\right)$ and $\overrightarrow{b} = \left( {{b}_{1},\ldots {b}_{n}}\right)$ ,the similarity function is

我们还展示了Kleinberg和Tardos[33]中舍入算法的一个有趣事实，适用于点上基础度量是均匀度量的情况。在此情况下，我们证明他们的舍入算法可视为扩展到连续设置的最小独立置换的泛化。他们的舍入过程为坐标均为非负的向量产生了一个局部敏感哈希函数。给定两个向量$\overrightarrow{a} = \left( {{a}_{1},\ldots {a}_{n}}\right)$和$\overrightarrow{b} = \left( {{b}_{1},\ldots {b}_{n}}\right)$，相似性函数为

$$
\operatorname{sim}\left( {\overrightarrow{a},\overrightarrow{b}}\right)  = \frac{\mathop{\sum }\limits_{i}\min \left( {{a}_{i},{b}_{i}}\right) }{\mathop{\sum }\limits_{i}\max \left( {{a}_{i},{b}_{i}}\right) }.
$$

(Note that when $\overrightarrow{a}$ and $\overrightarrow{b}$ are the characteristic vectors for sets $A$ and $B$ ,this expression reduces to the set similarity measure for min-wise independent permutations.)

（注意当$\overrightarrow{a}$和$\overrightarrow{b}$分别是集合$A$与$B$的特征向量时，该表达式可简化为最小独立排列的集合相似度度量。）

Applications of locality sensitive hash functions to solving nearest neighbor queries typically reduce the problem to the Hamming space. Indyk and Motwani [31] give a data structure that solves the approximate nearest neighbor problem on the Hamming space. Their construction is a reduction to the so called PLEB (Point Location in Equal Balls) problem, followed by a hashing technique concatenating the values of several locality sensitive hash functions. We give a simple technique that achieves the same performance as the Indyk Motwani result in Section 5. The basic idea is as follows: Given bit vectors consisting of $d$ bits each,we choose a number of random permutations of the bits. For each random permutation $\sigma$ ,we maintain a sorted order of the bit vectors, in lexicographic order of the bits permuted by $\sigma$ . To find a nearest neighbor for a query bit vector $q$ we do the following: For each permutation $\sigma$ ,we perform a binary search on the sorted order corresponding to $\sigma$ to locate the bit vectors closest to $q$ (in the lexicographic order obtained by bits permuted by $\sigma$ ). Further,we search in each of the sorted orders proceeding upwards and downwards from the location of $q$ , according to a certain rule. Of all the bit vectors examined, we return the one that has the smallest Hamming distance to the query vector. The performance bounds we can prove for this simple scheme are identical to that proved by Indyk and Motwani for their scheme.

局部敏感哈希函数在解决最近邻查询问题时通常将问题转化到汉明空间。Indyk和Motwani[31]提出了一种数据结构，用于解决汉明空间上的近似最近邻问题。其构造方法先将问题归约为所谓的PLEB（等球面点定位）问题，再通过串联多个局部敏感哈希函数值的哈希技术实现。我们在第5节提出了一种能达到与Indyk-Motwani方案相同性能的简易技术，其核心思路是：给定由$d$位组成的比特向量，我们随机选择若干位排列方式。对于每个随机排列$\sigma$，我们按$\sigma$置换后的字典序维护比特向量的排序列表。为查询向量$q$寻找最近邻时：对每个排列$\sigma$，在对应排序列表中进行二分搜索以定位最接近$q$的比特向量（按$\sigma$置换后的字典序）。此外，根据特定规则在排序列表中沿$q$位置向上和向下搜索。在所有检测的比特向量中，返回与查询向量汉明距离最小的那个。该简易方案的性能边界与Indyk和Motwani所证明的结果完全一致。

## 2. EXISTENCE OF LOCALITY SENSITIVE HASH FUNCTIONS

## 2. 局部敏感哈希函数的存在性

In this section, we discuss certain necessary properties for the existence of locality sensitive hash function families for given similarity measures.

本节将讨论给定相似度度量下，局部敏感哈希函数族存在的必要属性。

LEMMA 1. For any similarity function $\operatorname{sim}\left( {x,y}\right)$ that admits a locality sensitive hash function family as defined in (1),the distance function $1 - \operatorname{sim}\left( {x,y}\right)$ satisfies triangle inequality.

引理1. 对于任何满足(1)式定义的局部敏感哈希函数族的相似函数$\operatorname{sim}\left( {x,y}\right)$，其距离函数$1 - \operatorname{sim}\left( {x,y}\right)$必满足三角不等式。

Proof. Suppose there exists a locality sensitive hash function family such that

证明. 假设存在满足如下条件的局部敏感哈希函数族：

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   = \operatorname{sim}\left( {x,y}\right) .
$$

Then,

则有：

$$
1 - \operatorname{sim}\left( {x,y}\right)  = \mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  \neq  h\left( y\right) }\right\rbrack  .
$$

Let ${\Delta }_{h}\left( {x,y}\right)$ be an indicator variable for the event $h\left( x\right)  \neq$ $h\left( y\right)$ . We claim that ${\Delta }_{h}\left( {x,y}\right)$ satisfies the triangle inequality, i.e.

设${\Delta }_{h}\left( {x,y}\right)$为事件$h\left( x\right)  \neq$$h\left( y\right)$的指示变量。我们断言${\Delta }_{h}\left( {x,y}\right)$满足三角不等式，即

$$
{\Delta }_{h}\left( {x,y}\right)  + {\Delta }_{h}\left( {y,z}\right)  \geq  {\Delta }_{h}\left( {x,z}\right) .
$$

Since ${\Delta }_{h}\left( \right)$ takes values in the set $\{ 0,1\}$ ,the only case when the above inequality could be violated would be when ${\Delta }_{h}\left( {x,y}\right)  = {\Delta }_{h}\left( {y,z}\right)  = 0$ . But in this case $h\left( x\right)  = h\left( y\right)$ and $h\left( y\right)  = h\left( z\right)$ . Thus, $h\left( x\right)  = h\left( z\right)$ implying that ${\Delta }_{h}\left( {x,z}\right)  = 0$ and the inequality is satisfied. This proves the claim. Now,

由于${\Delta }_{h}\left( \right)$取值于集合$\{ 0,1\}$，上述不等式唯一可能被违反的情况是${\Delta }_{h}\left( {x,y}\right)  = {\Delta }_{h}\left( {y,z}\right)  = 0$。但此时$h\left( x\right)  = h\left( y\right)$且$h\left( y\right)  = h\left( z\right)$。因此$h\left( x\right)  = h\left( z\right)$意味着${\Delta }_{h}\left( {x,z}\right)  = 0$，不等式成立。由此得证。现可得：

$$
1 - \operatorname{sim}\left( {x,y}\right)  = {\mathbf{E}}_{h \in  \mathcal{F}}\left\lbrack  {{\Delta }_{h}\left( {x,y}\right) }\right\rbrack  
$$

---

<!-- Footnote -->

${}^{2}$ In Section 2,we will show that we can convert any locality sensitive hashing scheme to one that maps objects to $\{ 0,1\}$ with a slight change in similarity measure. However, the modified hash functions convey less information, e.g. the collision probability for the modified hash function family is at least $1/2$ even for a pair of objects with original similarity 0.

${}^{2}$ 在第2节中，我们将展示如何通过微调相似度度量，将任意局部敏感哈希方案转换为映射对象到$\{ 0,1\}$的方案。但修改后的哈希函数传递的信息量更少，例如对于原始相似度为0的对象对，修改后哈希函数族的碰撞概率仍不低于$1/2$。

<!-- Footnote -->

---

Since ${\Delta }_{h}\left( {x,y}\right)$ satisfies the triangle inequality, ${\mathbf{E}}_{h \in  \mathcal{F}}\left\lbrack  {{\Delta }_{h}\left( {x,y}\right) }\right\rbrack$ must also satisfy the triangle inequality. This proves the lemma.

既然${\Delta }_{h}\left( {x,y}\right)$满足三角不等式，${\mathbf{E}}_{h \in  \mathcal{F}}\left\lbrack  {{\Delta }_{h}\left( {x,y}\right) }\right\rbrack$必然也满足三角不等式。引理得证。

This gives a very simple proof of the fact that for the set similarity measure $\operatorname{sim}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\left| A \cup  B\right| },1 - \operatorname{sim}\left( {A,B}\right)$ satisfies the triangle inequality. This follows from Lemma 1 and the fact that a set similarity measure admits a locality sensitive hash function family, namely that given by minwise independent permutations.

这为集合相似度度量$\operatorname{sim}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\left| A \cup  B\right| },1 - \operatorname{sim}\left( {A,B}\right)$满足三角不等式的事实提供了一个极其简洁的证明。该结论源自引理1，以及集合相似度度量允许采用局部敏感哈希函数族这一特性，具体而言即由最小独立排列给出的哈希函数。

One could ask the question whether locality sensitive hash functions satisfying the definition (1) exist for other commonly used set similarity measures in information retrieval. For example, Dice's coefficient is defined as

人们可能会质疑，对于信息检索中其他常用集合相似度度量，是否存在满足定义(1)的局部敏感哈希函数。例如，Dice系数定义为

$$
{\operatorname{sim}}_{\text{Dice }}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\frac{1}{2}\left( {\left| A\right|  + \left| B\right| }\right) }
$$

The Overlap coefficient is defined as

重叠系数定义为

$$
{\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {A,B}\right)  = \frac{\left| A \cap  B\right| }{\min \left( {\left| A\right| ,\left| B\right| }\right) }
$$

We can use Lemma 1 to show that there is no such locality sensitive hash function family for Dice's coefficient and the Overlap measure by showing that the corresponding distance function does not satisfy triangle inequality.

通过引理1可证明，由于Dice系数和重叠度量对应的距离函数不满足三角不等式，故不存在相应的局部敏感哈希函数族。

Consider the sets $A = \{ a\} ,B = \{ b\} ,C = \{ a,b\}$ . Then,

考虑集合$A = \{ a\} ,B = \{ b\} ,C = \{ a,b\}$，则有：

$$
{\operatorname{sim}}_{{\mathrm{D}}_{\text{ice }}}\left( {A,C}\right)  = \frac{2}{3},\;{\operatorname{sim}}_{{\mathrm{D}}_{\text{ice }}}\left( {C,B}\right)  = \frac{2}{3},
$$

$$
{\operatorname{sim}}_{\text{Dice }}\left( {A,B}\right)  = 0
$$

$$
1 - {\operatorname{sim}}_{{\mathrm{D}}_{\text{ice }}}\left( {A,C}\right)  + 1 - {\operatorname{sim}}_{{\mathrm{D}}_{\text{ice }}}\left( {C,B}\right) 
$$

$$
 < 1 - {\operatorname{sim}}_{\text{Dice }}\left( {A,B}\right) 
$$

Similarly, the values for the Overlap measure are as follows:

类似地，重叠度量的取值如下：

$$
{\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {A,C}\right)  = 1,\;{\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {C,B}\right)  = 1,\;{\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {A,B}\right)  = 0
$$

$$
1 - {\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {A,C}\right)  + 1 - {\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {C,B}\right)  < 1 - {\operatorname{sim}}_{{\mathrm{O}}_{vl}}\left( {A,B}\right) 
$$

This shows that there is no locality sensitive hash function family corresponding to Dice's coefficient and the Overlap measure.

这表明不存在对应于Dice系数和重叠度量的局部敏感哈希函数族。

It is often convenient to have a hash function family that maps objects to $\{ 0,1\}$ . In that case,the output of $t$ different hash functions can simply be concatenated to obtain a $t$ -bit hash value for an object. In fact, we can always obtain such a binary hash function family with a slight change in the similarity measure. A similar result was used and proved by Gionis et al [22]. We include a proof for completeness.

若哈希函数族能将对象映射到$\{ 0,1\}$则通常更为便利。此时，只需串联$t$个不同哈希函数的输出即可获得对象的$t$位哈希值。实际上，我们总能通过对相似度度量进行微调来获得此类二元哈希函数族。Gionis等人[22]曾使用并证明了类似结论，此处为完备性补充证明。

LEMMA 2. Given a locality sensitive hash function family $\mathcal{F}$ corresponding to a similarity function $\operatorname{sim}\left( {x,y}\right)$ ,we can obtain a locality sensitive hash function family ${\mathcal{F}}^{\prime }$ that maps objects to $\{ 0,1\}$ and corresponds to the similarity function $\frac{1 + \operatorname{sim}\left( {x,y}\right) }{2}$ .

引理2. 给定对应于相似函数$\operatorname{sim}\left( {x,y}\right)$的局部敏感哈希函数族$\mathcal{F}$，可构造出将对象映射到$\{ 0,1\}$且对应于相似函数$\frac{1 + \operatorname{sim}\left( {x,y}\right) }{2}$的局部敏感哈希函数族${\mathcal{F}}^{\prime }$。

Proof. Suppose we have a hash function family such that

证明. 假设已有满足如下条件的哈希函数族：

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   = \operatorname{sim}\left( {x,y}\right) .
$$

Let $\mathcal{B}$ be a pairwise independent family of hash functions that operate on the domain of the functions in $\mathcal{F}$ and map elements in the domain to $\{ 0,1\}$ . Then $\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{b \in  \mathcal{B}}}\left\lbrack  {b\left( u\right)  = b\left( v\right) }\right\rbrack   =$ $1/2$ if $u \neq  v$ and $\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{b \in  \mathcal{B}}}\left\lbrack  {b\left( u\right)  = b\left( v\right) }\right\rbrack   = 1$ if $u = v$ . Consider the hash function family obtained by composing a hash function from $\mathcal{F}$ with one from $\mathcal{B}$ . This maps objects to $\{ 0,1\}$ and we claim that it has the required properties.

令$\mathcal{B}$为作用于$\mathcal{F}$函数域上的两两独立哈希函数族，其将定义域元素映射到$\{ 0,1\}$。则当$u \neq  v$时$\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{b \in  \mathcal{B}}}\left\lbrack  {b\left( u\right)  = b\left( v\right) }\right\rbrack   =$$1/2$，当$u = v$时$\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{b \in  \mathcal{B}}}\left\lbrack  {b\left( u\right)  = b\left( v\right) }\right\rbrack   = 1$。考虑由$\mathcal{F}$的哈希函数与$\mathcal{B}$的哈希函数复合而成的函数族，该函数族将对象映射到$\{ 0,1\}$且具备所需性质。

$$
\mathop{\Pr }\limits_{{h \in  \mathcal{F},b \in  \mathcal{B}}}\left\lbrack  {b\left( {h\left( x\right) }\right)  = b\left( {h\left( y\right) }\right) }\right\rbrack   = \frac{1 + \operatorname{sim}\left( {x,y}\right) }{2}
$$

With probability $\operatorname{sim}\left( {x,y}\right) ,h\left( x\right)  = h\left( y\right)$ and hence $b(h\left( x\right)  =$ $b\left( {h\left( y\right) }\right)$ . With probability $1 - \operatorname{sim}\left( {x,y}\right) ,h\left( x\right)  \neq  h\left( y\right)$ and in this case, $\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{b \in  \mathcal{B}}}\left\lbrack  {b(h\left( x\right)  = b\left( {h\left( y\right) }\right) \rbrack  = \frac{1}{2}}\right.$ . Thus,

以概率$\operatorname{sim}\left( {x,y}\right) ,h\left( x\right)  = h\left( y\right)$成立时，有$b(h\left( x\right)  =$$b\left( {h\left( y\right) }\right)$；以概率$1 - \operatorname{sim}\left( {x,y}\right) ,h\left( x\right)  \neq  h\left( y\right)$成立时，则有$\mathop{\operatorname{\mathbf{P} \mathbf{r} }}\limits_{{b \in  \mathcal{B}}}\left\lbrack  {b(h\left( x\right)  = b\left( {h\left( y\right) }\right) \rbrack  = \frac{1}{2}}\right.$。因此：

$$
\Pr \left\lbrack  {b\left( {h\left( x\right) }\right)  = b\left( {h\left( y\right) }\right) }\right\rbrack   = \operatorname{sim}\left( {x,y}\right)  + \left( {1 - \operatorname{sim}\left( {x,y}\right) }\right) /2
$$

$$
 = \left( {1 + \operatorname{sim}\left( {x,y}\right) }\right) /2\text{.}
$$

This can be used to show a stronger condition for the existence of a locality sensitive hash function family.

该结论可用于证明局部敏感哈希函数族存在的更强约束条件。

LEMMA 3. For any similarity function $\operatorname{sim}\left( {x,y}\right)$ that admits a locality sensitive hash function family as defined in (1),the distance function $1 - \operatorname{sim}\left( {x,y}\right)$ is isometrically embeddable in the Hamming cube.

引理3. 对于任何允许存在如(1)式定义的局部敏感哈希函数族的相似性函数$\operatorname{sim}\left( {x,y}\right)$，其距离函数$1 - \operatorname{sim}\left( {x,y}\right)$可等距嵌入汉明立方体。

Proof. Firstly, we apply Lemma 2 to construct a binary locality sensitive hash function family corresponding to similarity function ${\operatorname{sim}}^{\prime }\left( {x,y}\right)  = \left( {1 + \operatorname{sim}\left( {x,y}\right) }\right) /2$ . Note that such a binary hash function family gives an embedding of objects into the Hamming cube (obtained by concatenating the values of all the hash functions in the family). For object $x$ ,let $v\left( x\right)$ be the element in the Hamming cube $x$ is mapped to. $1 - {\operatorname{sim}}^{\prime }\left( {x,y}\right)$ is simply the fraction of bits that do not agree in $v\left( x\right)$ and $v\left( y\right)$ ,which is proportional to the Hamming distance between $v\left( x\right)$ and $v\left( y\right)$ . Thus this embedding is an isometric embedding of the distance function $1 - {\operatorname{sim}}^{\prime }\left( {x,y}\right)$ in the Hamming cube. But

证明. 首先应用引理2构建与相似性函数${\operatorname{sim}}^{\prime }\left( {x,y}\right)  = \left( {1 + \operatorname{sim}\left( {x,y}\right) }\right) /2$对应的二元局部敏感哈希函数族。注意到该二元哈希函数族实现了对象向汉明立方体的嵌入（通过连接该族所有哈希函数值获得）。对于对象$x$，设$v\left( x\right)$为其映射至汉明立方体的对应元素。$1 - {\operatorname{sim}}^{\prime }\left( {x,y}\right)$即为$v\left( x\right)$与$v\left( y\right)$中不匹配比特的比例，该值与两者间的汉明距离成正比。因此该嵌入实现了距离函数$1 - {\operatorname{sim}}^{\prime }\left( {x,y}\right)$在汉明立方体中的等距嵌入。但

$$
1 - {\operatorname{sim}}^{\prime }\left( {x,y}\right)  = 1 - \left( {1 + \operatorname{sim}\left( {x,y}\right) }\right) /2 = \left( {1 - \operatorname{sim}\left( {x,y}\right) }\right) /2.
$$

This implies that $1 - \operatorname{sim}\left( {x,y}\right)$ can be isometrically embedded in the Hamming cube.

这意味着$1 - \operatorname{sim}\left( {x,y}\right)$可被等距嵌入汉明立方体。

We note that Lemma 3 has a weak converse, i.e. for a similarity measure $\operatorname{sim}\left( {x,y}\right)$ any isometric embedding of the distance function $1 - \operatorname{sim}\left( {x,y}\right)$ in the Hamming cube yields a locality sensitive hash function family corresponding to the similarity measure $\left( {\alpha  + \operatorname{sim}\left( {x,y}\right) }\right) /\left( {\alpha  + 1}\right)$ for some $\alpha  > 0$ .

我们注意到引理3存在弱逆命题，即对于相似性度量$\operatorname{sim}\left( {x,y}\right)$，距离函数$1 - \operatorname{sim}\left( {x,y}\right)$在汉明立方体中的任何等距嵌入，都会产生对应于该相似性度量$\left( {\alpha  + \operatorname{sim}\left( {x,y}\right) }\right) /\left( {\alpha  + 1}\right)$的局部敏感哈希函数族（对于某个$\alpha  > 0$）。

## 3. RANDOM HYPERPLANE BASED HASH FUNCTIONS FOR VECTORS

## 3. 基于随机超平面的向量哈希函数

Given a collection of vectors in ${R}^{d}$ ,we consider the family of hash functions defined as follows: We choose a random vector $\overrightarrow{r}$ from the $d$ -dimensional Gaussian distribution (i.e. each coordinate is drawn the 1-dimensional Gaussian distribution). Corresponding to this vector $\overrightarrow{r}$ ,we define a hash function ${h}_{\overrightarrow{r}}$ as follows:

给定${R}^{d}$中的向量集合，我们考虑如下定义的哈希函数族：从$d$维高斯分布中选取随机向量$\overrightarrow{r}$（即每个坐标从一维高斯分布中抽取）。对应于该向量$\overrightarrow{r}$，我们定义哈希函数${h}_{\overrightarrow{r}}$如下：

$$
{h}_{\overrightarrow{r}}\left( \overrightarrow{u}\right)  = \left\{  \begin{array}{ll} 1 & \text{ if }\overrightarrow{r} \cdot  \overrightarrow{u} \geq  0 \\  0 & \text{ if }\overrightarrow{r} \cdot  \overrightarrow{u} < 0 \end{array}\right. 
$$

Then for vectors $\overrightarrow{u}$ and $\overrightarrow{v}$ ,

则对于向量$\overrightarrow{u}$和$\overrightarrow{v}$，

$$
\Pr \left\lbrack  {{h}_{\overrightarrow{r}}\left( \overrightarrow{u}\right)  = {h}_{\overrightarrow{r}}\left( \overrightarrow{v}\right) }\right\rbrack   = 1 - \frac{\theta \left( {\overrightarrow{u},\overrightarrow{v}}\right) }{\pi }.
$$

This was used by Goemans and Williamson [24] in their rounding scheme for the semidefinite programming relaxation of MAX-CUT.

该方法曾被Goemans和Williamson[24]用于MAX-CUT半定规划松弛的舍入方案。

Picking a random hyperplane amounts to choosing a normally distributed random variable for each dimension. Thus even representing a hash function in this family could require a large number of random bits. However,for $n$ vectors,the hash functions can be chosen by picking $O\left( {{\log }^{2}n}\right)$ random bits, i.e. we can restrict the random hyperplanes to be in a family of size ${2}^{O\left( {{\log }^{2}n}\right) }$ . This follows from the techniques in Indyk [30] and Engebretsen et al [17], which in turn use Nisan's pseudorandom number generator for space bounded computations [35]. We omit the details since they are similar to those in $\left\lbrack  {{30},{17}}\right\rbrack$ .

选择随机超平面等价于为每个维度选取正态分布的随机变量。因此即使表示该族中的哈希函数也可能需要大量随机比特。然而对于$n$向量，可通过选取$O\left( {{\log }^{2}n}\right)$个随机比特来选择哈希函数，即我们可以将随机超平面限制在大小为${2}^{O\left( {{\log }^{2}n}\right) }$的族中。这源于Indyk[30]和Engebretsen等人[17]的技术，后者又使用了Nisan针对空间受限计算的伪随机数生成器[35]。因与$\left\lbrack  {{30},{17}}\right\rbrack$中内容相似，此处省略细节。

Using this random hyperplane based hash function, we obtain a hash function family for set similarity, for a slightly different measure of similarity of sets. Suppose sets are represented by their characteristic vectors. Then, applying the above scheme gives a locality sensitive hashing scheme where

利用这种基于随机超平面的哈希函数，我们获得了适用于集合相似性的哈希函数族（针对稍有不同的集合相似性度量）。假设集合由其特征向量表示，则应用上述方案可得到局部敏感哈希方案，其中

$$
\Pr \left\lbrack  {h\left( A\right)  = h\left( B\right) }\right\rbrack   = 1 - \frac{\theta }{\pi },\text{ where }
$$

$$
\theta  = {\cos }^{-1}\left( \frac{\left| A \cap  B\right| }{\sqrt{\left| A\right|  \cdot  \left| B\right| }}\right) 
$$

Also, this hash function family facilitates easy incorporation of element weights in the similarity calculation, since the values of the coordinates of the characteristic vectors could be real valued element weights. Later, in Section 4.1 we will present another technique to define and estimate similarity of weighted sets.

此外，该哈希函数族便于在相似性计算中整合元素权重，因为特征向量坐标值可以是实数型的元素权重。后文第4.1节将介绍另一种定义和估计加权集相似性的技术。

### 4.THE EARTH MOVER DISTANCE

### 4. 推土机距离

Consider a set of points $L = \left\{  {{l}_{1},\ldots {l}_{n}}\right\}$ with a distance function $d\left( {i,j}\right)$ (assumed to be a metric). A distribution $P\left( L\right)$ on $L$ is a collection of non-negative weights $\left( {{p}_{1},\ldots {p}_{n}}\right)$ for points in $X$ such that $\sum {p}_{i} = 1$ . The distance between two distributions $P\left( L\right)$ and $Q\left( L\right)$ is defined to be the optimal cost of the following minimum transportation problem:

考虑一个点集$L = \left\{  {{l}_{1},\ldots {l}_{n}}\right\}$及其距离函数$d\left( {i,j}\right)$（假设为度量）。在$L$上的分布$P\left( L\right)$是指为$X$中各点分配非负权重$\left( {{p}_{1},\ldots {p}_{n}}\right)$的集合，且满足$\sum {p}_{i} = 1$。两个分布$P\left( L\right)$与$Q\left( L\right)$之间的距离定义为以下最小运输问题的最优成本：

$$
\min \mathop{\sum }\limits_{{i,j}}{\mathbf{f}}_{i,j} \cdot  d\left( {i,j}\right)  \tag{2}
$$

$$
\forall i\;\mathop{\sum }\limits_{j}{\mathbf{f}}_{i,j} = {p}_{i} \tag{3}
$$

$$
\forall j\;\mathop{\sum }\limits_{i}{\mathbf{f}}_{i,j} = {q}_{j} \tag{4}
$$

$$
\forall i,j\;{\mathbf{f}}_{i,j} \geq  0 \tag{5}
$$

Note that we define a somewhat restricted form of the Earth Mover Distance. The general definition does not assume that the sum of the weights is identical for distributions $P\left( L\right)$ and $Q\left( L\right)$ . This is useful for example in matching a small image to a portion of a larger image.

需注意此处定义的是地球移动距离（Earth Mover Distance）的限制形式。一般定义不要求分布$P\left( L\right)$与$Q\left( L\right)$的权重总和相同。这种特性在例如将小图像匹配到大图像局部区域时非常有用。

We will construct a hash function family for estimating the Earth Mover Distance based on rounding algorithms for the problem of classification with pairwise relationships, introduced by Kleinberg and Tardos [33]. (A closely related problem was also studied by Broder et al [9]). In designing hash functions to estimate the Earth Mover Distance, we will relax the definition of locality sensitive hashing (1) in three ways.

我们将基于Kleinberg和Tardos[33]提出的成对关系分类问题的舍入算法，构建用于估计地球移动距离的哈希函数族（Broder等人[9]也研究了密切相关的问题）。在设计哈希函数时，我们将从三个方面放宽局部敏感哈希(1)的定义：

1. Firstly, the quantity we are trying to estimate is a distance measure,not a similarity measure in $\left\lbrack  {0,1}\right\rbrack$ .

1. 首先，我们试图估计的是距离度量而非$\left\lbrack  {0,1}\right\rbrack$中的相似性度量。

2. Secondly, we will allow the hash functions to map objects to points in a metric space and measure

2. 其次，允许哈希函数将对象映射到度量空间中的点并测量

$\mathbf{E}\left\lbrack  {d\left( {h\left( x\right) ,h\left( y\right) }\right) }\right\rbrack$ . (A locality sensitive hash function for a similarity measure $\operatorname{sim}\left( {x,y}\right)$ can be viewed as a scheme to estimate the distance $1 - \operatorname{sim}\left( {x,y}\right)$ by $\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  \neq  h\left( y\right) }\right\rbrack$ . This is equivalent to having a uniform metric on the hash values).

$\mathbf{E}\left\lbrack  {d\left( {h\left( x\right) ,h\left( y\right) }\right) }\right\rbrack$。（相似性度量$\operatorname{sim}\left( {x,y}\right)$的局部敏感哈希函数可视为通过$\mathop{\Pr }\limits_{{h \in  \mathcal{F}}}\left\lbrack  {h\left( x\right)  \neq  h\left( y\right) }\right\rbrack$估计距离$1 - \operatorname{sim}\left( {x,y}\right)$的方案，这等价于对哈希值采用统一度量）。

3. Thirdly, our estimator for the Earth Mover Distance will not be an unbiased estimator, i.e. our estimate will approximate the Earth Mover Distance to within a small factor.

3. 第三，我们对地球移动距离的估计量并非无偏估计，即估计值将以较小系数近似真实距离。

We now describe the problem of classification with pairwise relationships. Given a collection of objects $V$ and labels $L = \left\{  {{l}_{1},\ldots ,{l}_{n}}\right\}$ ,the goal is to assign labels to objects. The cost of assigning label $l$ to object $u \in  V$ is $c\left( {u,l}\right)$ . Certain pairs of objects(u,v)are related; such pairs form the edges of a graph over $V$ . Each edge $e = \left( {u,v}\right)$ is associated with a non-negative weight ${w}_{e}$ . For edge $e = \left( {u,v}\right)$ ,if $u$ is assigned label $h\left( u\right)$ and $v$ is assigned label $h\left( v\right)$ ,then the cost paid is ${w}_{e}d\left( {h\left( u\right) ,h\left( v\right) }\right)$ .

现在描述成对关系分类问题：给定对象集合$V$和标签集$L = \left\{  {{l}_{1},\ldots ,{l}_{n}}\right\}$，目标是为对象分配标签。将标签$l$赋予对象$u \in  V$的成本为$c\left( {u,l}\right)$。特定对象对(u,v)存在关联关系，这些关系构成$V$上的图边集。每条边$e = \left( {u,v}\right)$关联非负权重${w}_{e}$。若边$e = \left( {u,v}\right)$关联的对象$u$被分配标签$h\left( u\right)$，$v$被分配标签$h\left( v\right)$，则产生成本${w}_{e}d\left( {h\left( u\right) ,h\left( v\right) }\right)$。

The problem is to come up with an assignment of labels $h : V \rightarrow  L$ ,so as to minimize the cost of the labeling $h$ given by

该问题的目标是找到标签分配方案$h : V \rightarrow  L$，以最小化由下式给出的标注成本$h$：

$$
\mathop{\sum }\limits_{{u \in  V}}c\left( {v,h\left( v\right) }\right)  + \mathop{\sum }\limits_{{e = \left( {u,v}\right)  \in  E}}{w}_{e}d\left( {h\left( u\right) ,h\left( v\right) }\right) 
$$

The approximation algorithms for this problem use an LP to assign,for every $u \in  V$ ,a probability distribution over labels in $L$ (i.e. a set of non-negative weights that sum up to 1). Given a distribution $P$ over labels in $L$ ,the rounding algorithm of Kleinberg and Tardos gave a randomized procedure for assigning label $h\left( P\right)$ to $P$ with the following properties:

该问题的近似算法通过线性规划为每个$u \in  V$分配$L$中标签的概率分布（即总和为1的非负权重集）。给定$L$上的标签分布$P$，Kleinberg和Tardos的舍入算法提供了随机分配标签$h\left( P\right)$至$P$的流程，具有以下特性：

1. Given distribution $P\left( L\right)  = \left( {{p}_{1},\ldots {p}_{n}}\right)$ ,

1. 给定分布$P\left( L\right)  = \left( {{p}_{1},\ldots {p}_{n}}\right)$，

$$
\Pr \left\lbrack  {h\left( P\right)  = {l}_{i}}\right\rbrack   = {p}_{i}. \tag{6}
$$

2. Suppose $P$ and $Q$ are probability distributions over $L$ .

2. 假设$P$和$Q$是定义在$L$上的概率分布。

$$
\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack   \leq  O\left( {\log n\log \log n}\right) \mathbf{{EMD}}\left( {P,Q}\right)  \tag{7}
$$

We note that the second property (7) is not immediately obvious from [33], since they do not describe LP relaxations for general metrics. Their LP relaxations are defined for Hierarchically well Separated Trees (HSTs). They convert a general metric to such an HST using Bartal's results [3, 4] on probabilistic approximation of metric spaces via tree metrics. However, it follows from combining ideas in [33] with those in Chekuri et al [11]. Chekuri et al do in fact give an LP relaxation for general metrics. The LP relaxation does indeed produce distributions over labels for every object $u \in  V$ . The fractional distance between two labelings is expressed as the min cost transshipment between $P$ and $Q$ , which is identical to the Earth Mover Distance $\mathbf{{EMD}}\left( {P,Q}\right)$ . Now, this fractional solution can be used in the rounding algorithm developed by Kleinberg and Tardos to obtain the second property (7) claimed above. In fact, Chekuri et al use this fact to claim that the gap of their LP relaxation is at most $O\left( {\log n\log \log n}\right)$ (Theorem 5.1 in [11]).

我们注意到第二个性质(7)并不能直接从文献[33]中得出，因为该文献未描述一般度量空间的线性规划松弛方法。他们的线性规划松弛仅针对层次化分离树(Hierarchically well Separated Trees, HSTs)定义。通过Bartal关于树度量空间概率逼近的研究成果[3,4]，他们将一般度量空间转换为HST。但结合文献[33]与Chekuri等人[11]的研究思路可以推导出该性质。Chekuri团队确实给出了适用于一般度量空间的线性规划松弛方法，该松弛算法会为每个对象$u \in  V$生成标签分布。两个标记方案之间的分数距离表示为$P$与$Q$之间的最小成本转运问题，这等同于地球移动距离(Earth Mover Distance)$\mathbf{{EMD}}\left( {P,Q}\right)$。该分数解可用于Kleinberg和Tardos开发的舍入算法，从而获得前述第二个性质(7)。事实上，Chekuri团队利用此结论证明其线性规划松弛的间隙至多为$O\left( {\log n\log \log n}\right)$（参见文献[11]定理5.1）。

We elaborate some more on why the property (7) holds. Kleinberg and Tardos first (probabilistically) approximate the metric on $L$ by an HST using $\left\lbrack  {3,4}\right\rbrack$ . This is a tree with all vertices in the original metric at the leaves. The pairwise distance between any two vertices does no decrease and all pairwise distances are increased by a factor of at most $O\left( {\log n\log \log n}\right)$ (in expectation). For this tree metric, they use an LP formulation which can be described as follows. Suppose we have a rooted tree. For subtree $T$ ,let ${\ell }_{T}$ denote the length of the edge that $T$ hangs off of,i.e. the first edge on the path from $T$ to the root. Further,for distribution $P$ on the vertices of the original metric,let $P\left( T\right)$ denote the total probability mass that $P$ assigns to leaves in $T;Q\left( T\right)$ is similarly defined. The distance between distributions $P$ and $Q$ is measured by $\mathop{\sum }\limits_{T}{\ell }_{T}\left| {P\left( T\right)  - Q\left( T\right) }\right|$ ,where the summation is computed over all subtrees $T$ . The Klein-berg Tardos rounding scheme ensures that $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$ is within a constant factor of $\mathop{\sum }\limits_{T}{\ell }_{T}\left| {P\left( T\right)  - Q\left( T\right) }\right|$ .

我们将进一步阐述性质(7)的成立原理。Kleinberg和Tardos首先利用$\left\lbrack  {3,4}\right\rbrack$对$L$上的度量空间进行HST概率逼近，该树结构将所有原始度量空间的顶点置于叶节点。任意两顶点间的距离不会缩短，且所有成对距离预期最多增加$O\left( {\log n\log \log n}\right)$倍。对于该树度量，他们采用如下线性规划模型：假设存在根树，对于子树$T$，令${\ell }_{T}$表示$T$悬挂边的长度（即该子树到根节点路径的首条边）。对于原始度量顶点上的分布$P$，令$P\left( T\right)$表示$P$分配给$T;Q\left( T\right)$叶节点的总概率质量（$Q$的定义类似）。分布$P$与$Q$间的距离通过$\mathop{\sum }\limits_{T}{\ell }_{T}\left| {P\left( T\right)  - Q\left( T\right) }\right|$度量（求和运算遍历所有子树$T$）。Kleinberg-Tardos舍入方案保证$\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$与$\mathop{\sum }\limits_{T}{\ell }_{T}\left| {P\left( T\right)  - Q\left( T\right) }\right|$的比值保持在常数范围内。

Suppose instead, we measured the distance between distributions by $\mathbf{{EMD}}\left( {P,Q}\right)$ ,defined on the original metric. By probabilistically approximating the original metric by a tree metric ${T}^{\prime }$ ,the expected value of the distance ${\mathbf{{EMD}}}_{{T}^{\prime }}\left( {P,Q}\right)$ (on the tree metric ${T}^{\prime }$ ) is at most a factor of $O\left( {\log n\log \log n}\right)$ times $\mathbf{{EMD}}\left( {P,Q}\right)$ . This follows since all distances increase by $O\left( {\log n\log \log n}\right)$ in expectation. Now note that the tree distance measure used by Kleinberg and Tardos $\mathop{\sum }\limits_{T}{\ell }_{T} \mid  P\left( T\right)  -$ $Q\left( T\right)  \mid$ is a lower bound on (and in fact exactly equal to) ${\mathbf{{EMD}}}_{{T}^{\prime }}\left( {P,Q}\right)$ . To see that this is a lower bound,note that in the min cost transportation between $P$ and $Q$ on ${T}^{\prime }$ , the flow on the edge leading upwards from subtree $T$ must be at least $\left| {P\left( T\right)  - Q\left( T\right) }\right|$ . Since the rounding scheme ensures that $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$ is within a constant factor of $\mathop{\sum }\limits_{T}{\ell }_{T}\left| {P\left( T\right)  - Q\left( T\right) }\right|$ ,we have that

假设我们改用原度量上定义的$\mathbf{{EMD}}\left( {P,Q}\right)$来衡量分布间距离。通过概率性地用树度量${T}^{\prime }$逼近原度量，距离${\mathbf{{EMD}}}_{{T}^{\prime }}\left( {P,Q}\right)$（在树度量${T}^{\prime }$上）的期望值至多是$\mathbf{{EMD}}\left( {P,Q}\right)$的$O\left( {\log n\log \log n}\right)$倍。这是因为所有距离在期望中都会增加$O\left( {\log n\log \log n}\right)$倍。需注意Kleinberg和Tardos$\mathop{\sum }\limits_{T}{\ell }_{T} \mid  P\left( T\right)  -$$Q\left( T\right)  \mid$使用的树距离度量是${\mathbf{{EMD}}}_{{T}^{\prime }}\left( {P,Q}\right)$的下界（实际上完全相等）。要理解这是下界，请注意在${T}^{\prime }$上$P$与$Q$之间的最小成本运输中，从子树$T$向上流动的边流量至少为$\left| {P\left( T\right)  - Q\left( T\right) }\right|$。由于舍入方案确保$\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$与$\mathop{\sum }\limits_{T}{\ell }_{T}\left| {P\left( T\right)  - Q\left( T\right) }\right|$相差常数倍，因此可得：

$$
\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack   \leq  O\left( 1\right) {\mathbf{{EMD}}}_{{T}^{\prime }}\left( {P,Q}\right) 
$$

$$
 \leq  O\left( {\log n\log \log n}\right) \mathbf{{EMD}}\left( {P,Q}\right) 
$$

where the expectation is over the random choice of the HST and the random choices made by the rounding procedure.

其中期望值取自HST的随机选择及舍入过程的随机决策。

THEOREM 1. The Kleinberg Tardos rounding scheme yields a locality sensitive hashing scheme such that

定理1. Kleinberg-Tardos舍入方案产生的局部敏感哈希方案满足：

$$
\mathbf{{EMD}}\left( {P,Q}\right)  \leq  \mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack  
$$

$$
 \leq  O\left( {\log n\log \log n}\right) \mathbf{{EMD}}\left( {P,Q}\right) .
$$

Proof. The upper bound on $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$ follows directly from the second property (7) of the rounding scheme stated above.

证明. $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$的上界直接由前述舍入方案的第二个性质(7)得出。

We show that the lower bound follows from the first property (6). Let ${y}_{i,j}$ be the joint probability that $h\left( P\right)  = {l}_{i}$ and $h\left( Q\right)  = {l}_{j}$ . Note that $\mathop{\sum }\limits_{j}{y}_{i,j} = {p}_{i}$ ,since this is simply the probability that $h\left( P\right)  = {l}_{i}$ . Similarly $\mathop{\sum }\limits_{i}{y}_{i,j} = {q}_{j}$ , since this is simply the probability that $h\left( Q\right)  = {l}_{j}$ . Now,if $h\left( P\right)  = {l}_{i}$ and $h\left( Q\right)  = {l}_{j}$ ,then $d\left( {h\left( P\right) h\left( Q\right) }\right)  = d\left( {i,j}\right)$ . Hence $\mathbf{E}\left\lbrack  {d\left( {f\left( P\right) ,f\left( Q\right) }\right) }\right\rbrack   = \mathop{\sum }\limits_{{i,j}}{y}_{i,j} \cdot  d\left( {i,j}\right)$ . Let us write down the expected cost and the constraints on ${y}_{i,j}$ .

我们证明下界来自第一个性质(6)。设${y}_{i,j}$为$h\left( P\right)  = {l}_{i}$与$h\left( Q\right)  = {l}_{j}$的联合概率。注意$\mathop{\sum }\limits_{j}{y}_{i,j} = {p}_{i}$，因为这仅是$h\left( P\right)  = {l}_{i}$的概率。同理$\mathop{\sum }\limits_{i}{y}_{i,j} = {q}_{j}$，因其仅是$h\left( Q\right)  = {l}_{j}$的概率。若$h\left( P\right)  = {l}_{i}$且$h\left( Q\right)  = {l}_{j}$，则$d\left( {h\left( P\right) h\left( Q\right) }\right)  = d\left( {i,j}\right)$。因此$\mathbf{E}\left\lbrack  {d\left( {f\left( P\right) ,f\left( Q\right) }\right) }\right\rbrack   = \mathop{\sum }\limits_{{i,j}}{y}_{i,j} \cdot  d\left( {i,j}\right)$。现列出${y}_{i,j}$的期望成本及约束条件。

$$
\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack   = \mathop{\sum }\limits_{{i,j}}{\mathbf{y}}_{i,j} \cdot  d\left( {i,j}\right) 
$$

$$
\forall i\;\mathop{\sum }\limits_{j}{y}_{i,j} = {p}_{i}
$$

$$
\forall j\;\mathop{\sum }\limits_{i}{y}_{i,j} = {q}_{j}
$$

$$
\forall i,j\;{y}_{i,j} \geq  0
$$

Comparing this with the LP for $\mathbf{{EMD}}\left( {P,Q}\right)$ ,we see that the values of ${f}_{i,j} = {y}_{i,j}$ is a feasible solution to the LP (2) to (5) and $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$ is exactly the value of this solution. Since $\mathbf{{EMD}}\left( {P,Q}\right)$ is the minimum value of a feasible solution,it follows that $\operatorname{EMD}\left( {P,Q}\right)  \leq  \mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$ .

将此与$\mathbf{{EMD}}\left( {P,Q}\right)$的线性规划(LP)比较，可见${f}_{i,j} = {y}_{i,j}$的值是LP(2)至(5)的可行解，而$\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$正是该解的值。由于$\mathbf{{EMD}}\left( {P,Q}\right)$是可行解的最小值，故可推出$\operatorname{EMD}\left( {P,Q}\right)  \leq  \mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$。

Calinescu et al $\left\lbrack  {10}\right\rbrack$ study a variant of the classification problem with pairwise relationships called the 0 -extension problem. This is the version without assignment costs where some objects are assigned labels apriori and this labeling must be extended to the other objects (a generalization of multiway cut). For this problem, they design a rounding scheme to get a $O\left( {\log n}\right)$ approximation. Again,their technique does not explicitly use an LP that gives probability distributions on labels. However in hindsight, their rounding scheme can be interpreted as a randomized procedure for assigning labels to distributions such that

Calinescu等人$\left\lbrack  {10}\right\rbrack$研究了一种带有成对关系的分类问题变体——0-延展问题。这是无分配成本的版本，其中某些对象被预先分配标签，且必须将此标记延展至其他对象（多路切割的广义形式）。针对该问题，他们设计了一种舍入方案以获得$O\left( {\log n}\right)$近似解。需注意的是，其技术并未显式使用给出标签概率分布的线性规划。但后见之明表明，他们的舍入方案可解读为一种随机化分配标签至分布的流程，使得

## $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack   \leq  O\left( {\log n}\right) \mathbf{{EMD}}\left( {P,Q}\right) .$

Thus their rounding scheme gives a tighter guarantee than (7). However, they do not ensure (6). Thus the previous proof showing that $\mathbf{{EMD}}\left( {P,Q}\right)  \leq  \mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$ does not apply. In fact one can construct examples such that $\mathbf{{EMD}}\left( {P,Q}\right)  > 0$ ,yet $\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack   = 0$ . Hence,the resulting hash function family provides an upper bound on $\mathbf{{EMD}}\left( {P,Q}\right)$ within a factor $O\left( {\log n}\right)$ but does not provide a good lower bound.

因此其舍入方案提供了比(7)更严格的保证。然而他们并未确保(6)成立。故先前证明$\mathbf{{EMD}}\left( {P,Q}\right)  \leq  \mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack$不适用。实际上可构造出$\mathbf{{EMD}}\left( {P,Q}\right)  > 0$但$\mathbf{E}\left\lbrack  {d\left( {h\left( P\right) ,h\left( Q\right) }\right) }\right\rbrack   = 0$的示例。因此，所得哈希函数族能在$O\left( {\log n}\right)$因子内为$\mathbf{{EMD}}\left( {P,Q}\right)$提供上界，但无法给出良好的下界。

We mention that the hashing scheme described provides an approximation to the Earth Mover Distance where the quality of the approximation is exactly the factor by which the underlying metric can be probabilistically approximated by HSTs. In particular, if the underlying metric itself is an HST, this yields an estimate within a constant factor. This could have applications in compactly representing distributions over hierarchical classes. For example, documents can be assigned a probability distribution over classes in the Open Directory Project (ODP) hierarchy. This hierarchy could be thought of as an HST and documents can be mapped to distributions over this HST. The distance between two distributions can be measured by the Earth Mover Distance. In this case, the hashing scheme described gives a way to estimate this distance measure to within a constant factor.

需说明的是，所述哈希方案为地球移动距离(EMD)提供了近似解，其近似质量恰好等于底层度量被层次树(HST)概率近似的因子。特别地，若底层度量本身即为HST，则能获得常数因子内的估计值。这在层次类别的紧凑分布表示中具有应用潜力。例如，文档可被分配开放目录项目(ODP)层次结构中各类别的概率分布。该层次结构可视作HST，文档则可映射至该HST上的分布。两个分布间的距离可通过地球移动距离度量。此时，所述哈希方案能以常数因子估计该距离度量。

### 4.1 Weighted Sets

### 4.1 加权集合

We show that the Kleinberg Tardos [33] rounding scheme for the case of the uniform metric actually is an extension of min-wise independent permutations to the weighted case.

我们证明，Kleinberg与Tardos[33]针对均匀度量情形提出的舍入方案，实际上是min-wise独立排列在加权情况下的扩展。

First we recall the hashing scheme given by min-wise independent permutations. Given a universe $U$ ,consider a random permutation $\pi$ of $U$ . Assume that the elements of $U$ are totally ordered. Given a subset $A \subseteq  U$ ,the hash function ${h}_{\pi }$ is defined as follows:

首先回顾min-wise独立排列给出的哈希方案。给定全集$U$，考虑$U$的一个随机排列$\pi$。假设$U$的元素具有全序关系。对于子集$A \subseteq  U$，哈希函数${h}_{\pi }$定义如下：

$$
{h}_{\pi }\left( A\right)  = \min \{ \pi \left( A\right) \} 
$$

Then the property satisfied by this hash function family is that

该哈希函数族满足的性质是

$$
\mathop{\Pr }\limits_{\pi }\left\lbrack  {{h}_{\pi }\left( A\right)  = {h}_{\pi }\left( B\right) }\right\rbrack   = \frac{\left| A \cap  B\right| }{\left| A \cup  B\right| }
$$

We now review the Kleinberg Tardos rounding scheme for the uniform metric: Firstly, imagine that we pick an infinite sequence ${\left\{  \left( {i}_{t},{\alpha }_{t}\right) \right\}  }_{t = 1}^{\infty }$ where for each $t,{i}_{t}$ is picked uniformly and at random in $\{ 1,\ldots n\}$ and ${\alpha }_{t}$ is picked uniformly and at random in $\left\lbrack  {0,1}\right\rbrack$ . Given a distribution $P = \left( {{p}_{1},\ldots ,{p}_{n}}\right)$ , the assignment of labels is done in phases. In the $i$ th phase, we check whether ${\alpha }_{i} \leq  {p}_{{i}_{t}}$ . If this is the case and $P$ has not been assigned a label yet,it is assigned label ${i}_{t}$ .

现在重审Kleinberg-Tardos针对均匀度量的舍入方案：首先设想选取无限序列${\left\{  \left( {i}_{t},{\alpha }_{t}\right) \right\}  }_{t = 1}^{\infty }$，其中每个$t,{i}_{t}$在$\{ 1,\ldots n\}$上均匀随机选取，${\alpha }_{t}$在$\left\lbrack  {0,1}\right\rbrack$上均匀随机选取。给定分布$P = \left( {{p}_{1},\ldots ,{p}_{n}}\right)$，标签分配分阶段进行。在第$i$阶段，检查是否满足${\alpha }_{i} \leq  {p}_{{i}_{t}}$。若满足且$P$尚未分配标签，则为其分配标签${i}_{t}$。

Now,we can think of these distributions as sets in ${R}^{2}$ (see Figure 1).

现在可将这些分布视为${R}^{2}$中的集合（见图1）。

<!-- Media -->

<!-- figureText: 1 3 4 5 7 8 -->

<img src="https://cdn.noedgeai.com/019629ed-d415-7da2-8f5e-6edcf2e3d1e2_6.jpg?x=180&y=730&w=655&h=188&r=0"/>

Figure 1: Viewing a distribution as a continuous set.

图1：将分布视为连续集合的示意图

<!-- Media -->

The set $S\left( P\right)$ corresponding to distribution $P$ consists of the union of the rectangles $\left\lbrack  {i - 1,i}\right\rbrack   \times  \left\lbrack  {0,{p}_{i}}\right\rbrack$ . The elements of the universe are $\left\lbrack  {i - 1,i}\right\rbrack   \times  \alpha .\left\lbrack  {i - 1,i}\right\rbrack   \times  \alpha$ belongs to $S\left( P\right)$ iff $\alpha  \leq  {p}_{i}$ . The notion of cardinality of union and intersection of sets is replaced by the area of the intersection and union of two such sets in ${R}^{2}$ . Note that the Kleinberg Tar-dos rounding scheme can be interpreted as constructing a permutation of the universe and assigning to a distribution $P$ ,the value $i$ such that $\left( {i,\alpha }\right)$ is the minimum in the permutation amongst all elements contained in $S\left( P\right)$ . Suppose instead,we assign to $P$ ,the element $\left( {i,\alpha }\right)$ which is the minimum in the permutation of $S\left( P\right)$ . Let $h$ be a hash function derived from this scheme (a slight modification of the one in [33]). Then,

与分布$P$对应的集合$S\left( P\right)$由矩形$\left\lbrack  {i - 1,i}\right\rbrack   \times  \left\lbrack  {0,{p}_{i}}\right\rbrack$的并集构成。全集中元素$\left\lbrack  {i - 1,i}\right\rbrack   \times  \alpha .\left\lbrack  {i - 1,i}\right\rbrack   \times  \alpha$属于$S\left( P\right)$当且仅当$\alpha  \leq  {p}_{i}$。集合交并的基数概念被替换为${R}^{2}$中两个此类集合交并的面积。注意Kleinberg-Tardos舍入方案可解释为：构造全集的一个排列，并为分布$P$分配值$i$，使得$\left( {i,\alpha }\right)$是该排列中包含在$S\left( P\right)$的所有元素中的最小值。若改为分配$S\left( P\right)$排列中最小值元素$\left( {i,\alpha }\right)$给$P$，设$h$为该方案导出的哈希函数（对文献[33]方案的轻微修改），则有：

$$
\Pr \left\lbrack  {h\left( P\right)  = h\left( Q\right) }\right\rbrack   = \frac{\left| S\left( P\right)  \cap  S\left( Q\right) \right| }{\left| S\left( P\right)  \cup  S\left( Q\right) \right| } = \frac{\mathop{\sum }\limits_{i}\min \left( {{p}_{i},{q}_{i}}\right) }{\mathop{\sum }\limits_{i}\max \left( {{p}_{i},{q}_{i}}\right) } \tag{8}
$$

For the Kleinberg Tardos rounding scheme, the probability of collision is at least the probability of collision for the modified scheme (since two objects hashed to $\left( {i,{\alpha }_{1}}\right)$ and $\left( {i,{\alpha }_{2}}\right)$ respectively in the modified scheme would be both mapped to $i$ in the original scheme). Hence

对于Kleinberg-Tardos舍入方案，冲突概率至少等于修改方案的冲突概率（因修改方案中分别被哈希到$\left( {i,{\alpha }_{1}}\right)$和$\left( {i,{\alpha }_{2}}\right)$的两个对象，在原方案中会被同时映射到$i$）。因此

$$
\mathop{\Pr }\limits_{{KT}}\left\lbrack  {h\left( P\right)  = h\left( Q\right) }\right\rbrack   \geq  \frac{\mathop{\sum }\limits_{i}\min \left( {{p}_{i},{q}_{i}}\right) }{\mathop{\sum }\limits_{i}\max \left( {{p}_{i},{q}_{i}}\right) }
$$

$$
\mathop{\Pr }\limits_{{KT}}\left\lbrack  {h\left( P\right)  \neq  h\left( Q\right) }\right\rbrack   \leq  1 - \frac{\mathop{\sum }\limits_{i}\min \left( {{p}_{i},{q}_{i}}\right) }{\mathop{\sum }\limits_{i}\max \left( {{p}_{i},{q}_{i}}\right) }
$$

$$
 = \frac{\mathop{\sum }\limits_{i}\left| {{p}_{i} - {q}_{i}}\right| }{\mathop{\sum }\limits_{i}\max \left( {{p}_{i},{q}_{i}}\right) } \leq  \mathop{\sum }\limits_{i}\left| {{p}_{i} - {q}_{i}}\right| 
$$

The last inequality follows from the fact that $\sum {p}_{i} = \sum {q}_{i} =$ 1 in the Kleinberg Tardos setting. This was exactly the property used in [33] to obtain a 2-approximation for the uniform metric case.

末项不等式源于Kleinberg-Tardos设定中$\sum {p}_{i} = \sum {q}_{i} =$1的性质。这正是文献[33]用于获得均匀度量情形2-近似解的关键特性。

Note that the hashing scheme given by (8) is a generalization of min-wise independent permutations to the weighted setting where elements in sets are associated with weights $\in  \left\lbrack  {0,1}\right\rbrack$ . Min-wise independent permutations are a special case of this scheme when the weights are $\{ 0,1\}$ . This scheme could be useful in a setting where a weighted set similarity notion is desired. We note that the original min-wise independent permutations can be used in the setting of integer weights by simply duplicating elements according to their weight. The present scheme would work for any nonnegative real weights.

请注意，(8)式给出的哈希方案是将最小独立排列推广到加权场景的泛化形式，其中集合中的元素具有权重$\in  \left\lbrack  {0,1}\right\rbrack$。当权重为$\{ 0,1\}$时，最小独立排列是该方案的特例。该方案适用于需要定义加权集合相似度的场景。需说明的是，原始最小独立排列可通过按权重复制元素的方式应用于整数权重场景，而本方案适用于任意非负实数权重。

## 5. APPROXIMATE NEAREST NEIGHBOR SEARCH IN HAMMING SPACE.

## 5. 汉明空间中的近似最近邻搜索

Applications of locality sensitive hash functions to solving nearest neighbor queries typically reduce the problem to the Hamming space. Indyk and Motwani [31] give a data structure that solves the approximate nearest neighbor problem on the Hamming space ${H}^{d}$ . Their construction is a reduction to the so called PLEB (Point Location in Equal Balls) problem, followed by a hashing technique concatenating the values of several locality sensitive hash functions.

局部敏感哈希函数在解决最近邻查询时的应用通常将问题规约到汉明空间。Indyk和Motwani[31]提出了一种数据结构，可解决汉明空间${H}^{d}$上的近似最近邻问题。其构造方法先规约到所谓的PLEB(等球体中的点定位)问题，再通过连接多个局部敏感哈希函数值的哈希技术实现。

THEOREM 2 ([31]). For any $\epsilon  > 0$ ,there exists an algorithm for $\epsilon$ -PLEB in ${H}^{d}$ using $O\left( {{dn} + {n}^{1 + 1/\left( {1 + \epsilon }\right) }}\right)$ space and $O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$ hash function evaluations for each query.

定理2([31])。对于任意$\epsilon  > 0$，存在求解$\epsilon$-PLEB的算法，该算法在${H}^{d}$空间内运行，每次查询需进行$O\left( {{dn} + {n}^{1 + 1/\left( {1 + \epsilon }\right) }}\right)$次哈希函数计算。

We give a simple technique that achieves the same performance as the Indyk Motwani result:

我们提出了一种能达到与Indyk-Motwani方案相同性能的简单技术：

Given bit vectors consisting of $d$ bits each,we choose $N = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$ random permutations of the bits. For each random permutation $\sigma$ ,we maintain a sorted order ${O}_{\sigma }$ of the bit vectors, in lexicographic order of the bits permuted by $\sigma$ . Given a query bit vector $q$ ,we find the approximate nearest neighbor by doing the following: For each permutation $\sigma$ ,we perform a binary search on ${O}_{\sigma }$ to locate the two bit vectors closest to $q$ (in the lexicographic order obtained by bits permuted by $\sigma$ ). We now search in each of the sorted orders ${O}_{\sigma }$ examining elements above and below the position returned by the binary search in order of the length of the longest prefix that matches $q$ . This can be done by maintaining two pointers for each sorted order ${O}_{\sigma }$ (one moves up and the other down). At each step we move one of the pointers up or down corresponding to the element with the longest matching prefix. (Here the length of the longest matching prefix in ${O}_{\sigma }$ is computed relative to $q$ with its bits permuted by $\sigma$ ). We examine ${2N} = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$ bit vectors in this way. Of all the bit vectors examined, we return the one that has the smallest Hamming distance to $q$ . The performance bounds we can prove for this simple scheme are identical to that proved by Indyk and Motwani for their scheme. An advantage of this scheme is that we do not need a reduction to many instances of PLEB for different values of radius $r$ ,i.e. we solve the nearest neighbor problem simultaneously for all values of radius $r$ using a single data structure.

给定由$d$位组成的比特向量，我们随机选择$N = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$个位排列组合。对于每个随机排列$\sigma$，我们按该排列的字典序维护比特向量的排序列表${O}_{\sigma }$。查询比特向量$q$时，通过以下步骤寻找近似最近邻：对每个排列$\sigma$，在${O}_{\sigma }$上执行二分查找定位最接近$q$的两个比特向量（根据$\sigma$排列后的字典序）。随后在每个排序列表${O}_{\sigma }$中，按照与$q$匹配的最长前缀长度顺序，检查二分查找返回位置上下方的元素。这可通过为每个排序列表${O}_{\sigma }$维护两个指针实现（一个上移一个下移）。每步操作中，我们根据最长匹配前缀元素移动对应指针（此处${O}_{\sigma }$中的最长匹配前缀长度是相对于经$\sigma$排列后的$q$计算）。我们以此方式检查${2N} = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$个比特向量，最终返回与$q$汉明距离最小的向量。该简单方案的性能边界与Indyk和Motwani所证明的结果一致，其优势在于无需针对不同半径值$r$规约到多个PLEB实例，即通过单一数据结构即可同时求解所有半径值$r$的最近邻问题。

We outline the main ideas of the analysis. In fact, the proof follows along similar lines to the proofs of Theorem 5 and Corollary 3 in [31]. Suppose the nearest neighbor of $q$ is at a Hamming distance of $r$ from $q$ . Set ${p}_{1} = 1 - \frac{r}{d}$ , ${p}_{2} = 1 - \frac{r\left( {1 + \epsilon }\right) }{d}$ and $k = {\log }_{1/{p}_{2}}n$ . Let $\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}}$ . Then ${n}^{\rho } = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$ . We can show that with constant probability,from amongst $N = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$ permutations,there exists a permutation such that the nearest neighbor agrees with $p$ on the first $k$ coordinates in $\sigma$ . Further,over all $L$ permutations, the number of bit vectors that are at Hamming distance of more than $r\left( {1 + \epsilon }\right)$ from $q$ and agree on the first $k$ coordinates is at most ${2N}$ with constant probability. This implies that for this permutation $\sigma$ ,one of the ${2L}$ bit vectors near $q$ in the ordering ${O}_{\sigma }$ and examined by the algorithm will be a $\left( {1 + \epsilon }\right)$ -approximate nearest neighbor. The probability calculations are similar to those in [31], and we only sketch the main ideas.

我们概述了分析的主要思路。实际上，该证明遵循与文献[31]中定理5和推论3相似的证明路径。假设$q$的最近邻点与之汉明距离为$r$。设${p}_{1} = 1 - \frac{r}{d}$、${p}_{2} = 1 - \frac{r\left( {1 + \epsilon }\right) }{d}$和$k = {\log }_{1/{p}_{2}}n$。令$\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}}$，则${n}^{\rho } = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$。我们可以证明，在$N = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right)$个排列中，存在一个排列使得最近邻点与$p$在$\sigma$的前$k$个坐标上一致的概率为常数。此外，在所有$L$个排列中，与$q$汉明距离超过$r\left( {1 + \epsilon }\right)$且前$k$个坐标一致的比特向量数量，以常数概率不超过${2N}$。这意味着对于排列$\sigma$，算法在排序${O}_{\sigma }$中检查的$q$附近${2L}$个比特向量之一，将是一个$\left( {1 + \epsilon }\right)$近似最近邻。概率计算与文献[31]类似，此处仅简述核心思想。

For any point ${q}^{\prime }$ at distance at least $r\left( {1 + \epsilon }\right)$ from $q$ ,the probability that a random coordinate agrees with $q$ is at most ${p}_{2}$ . Thus the probability that the first $k$ coordinates agree is at most ${p}_{2}^{k} = \frac{1}{n}$ . For the $N$ permutations,the expected number of such points that agree in the first $k$ coordinates is at most $N$ . The probability that this number is $\leq  {2N}$ is $> 1/2$ . Further,for a random permutation $\sigma$ ,the probability that the nearest neighbor agrees in $k$ coordinates is ${p}_{1}^{k} = {n}^{-\rho }$ . Hence the probability that there exists one permutation amongst the $N = {n}^{\rho }$ permutations where the nearest neighbor agrees in $k$ coordinates is at least $1 - {\left( 1 - {n}^{-\rho }\right) }^{{n}^{\rho }} > 1/2$ . This establishes the correctness of the procedure.

对于与$q$距离至少为$r\left( {1 + \epsilon }\right)$的任意点${q}^{\prime }$，其随机坐标与$q$一致的概率至多为${p}_{2}$。因此前$k$个坐标一致的概率上限为${p}_{2}^{k} = \frac{1}{n}$。对于$N$个排列，前$k$个坐标一致的期望点数不超过$N$。该数值达到$\leq  {2N}$的概率为$> 1/2$。进一步，随机排列$\sigma$中最近邻前$k$个坐标一致的概率是${p}_{1}^{k} = {n}^{-\rho }$。因此在$N = {n}^{\rho }$个排列中存在至少一个排列使最近邻前$k$个坐标一致的概率不低于$1 - {\left( 1 - {n}^{-\rho }\right) }^{{n}^{\rho }} > 1/2$。这验证了算法的正确性。

As we stated earlier, a nice property of this data structure is that it automatically adjusts to the correct distance $r$ to the nearest neighbor, i.e. we do not need to maintain separate data structures for different values of $r$ .

如前所述，该数据结构的一个优良特性是能自动适应最近邻的正确距离$r$，即无需为不同$r$值维护独立数据结构。

## 6. CONCLUSIONS

## 6. 结论

We have demonstrated an interesting relationship between rounding algorithms used for rounding fractional solutions of LPs and vector solutions of SDPs on the one hand, and the construction of locality sensitive hash functions for interesting classes of objects, on the other.

我们揭示了一个有趣的双向关系：一方面是将线性规划(LP)分数解和半定规划(SDP)向量解取整的算法，另一方面是针对特定对象类构造局部敏感哈希函数的方法。

Rounding algorithms yield new constructions of locality sensitive hash functions that were not known previously. Conversely (at least in hindsight), locality sensitive hash functions lead to rounding algorithms (as in the case of min-wise independent permutations and the uniform metric case in Kleinberg and Tardos [33]).

取整算法产生了先前未知的新型局部敏感哈希函数构造。反观之（至少事后看来），局部敏感哈希函数也能导向取整算法（如Kleinberg和Tardos[33]中最小独立排列与均匀度量空间的案例）。

An interesting direction to pursue would be to investigate the construction of sketching functions that allow one to estimate information theoretic measures of distance between distributions such as the KL-divergence, commonly used in statistical learning theory. Since the KL-divergence is neither symmetric nor satisfies triangle inequality, new ideas would be required in order to design a sketch function to approximate it. Such a sketch function, if one exists, would be a very valuable tool in compactly representing complex distributions.

一个值得探索的方向是研究能估计分布间信息理论距离度量（如统计学习理论常用的KL散度）的素描函数构造。由于KL散度既不满足对称性也不符合三角不等式，设计其近似素描函数需要新思路。此类函数若能实现，将成为紧凑表示复杂分布的宝贵工具。

## 7. REFERENCES

## 7. 参考文献

[1] N. Alon, P. B. Gibbons, Y. Matias, and M. Szegedy. Tracking Join and Self-Join Sizes in Limited Storage. Proc. 18th PODS pp. 10-20, 1999.

[2] N. Alon, Y. Matias, and M. Szegedy. The Space Complexity of Approximating the Frequency Moments. JCSS 58(1): 137-147, 1999

[3] Y. Bartal. Probabilistic approximation of metric spaces and its algorithmic application. Proc. 37th FOCS, pages 184-193, 1996.

[4] Y. Bartal. On approximating arbitrary metrics by tree

[4] Y. Bartal. 关于用树近似任意度量

metrics. In Proc. 30th STOC, pages 161-168, 1998.

[5] A. Z. Broder. On the resemblance and containment of documents. Proc. Compression and Complexity of ${SEQUENCES}$ ,pp. 21-29. IEEE Computer Society, 1997.

[6] A. Z. Broder. Filtering near-duplicate documents. Proc. FUN 98, 1998.

[7] A. Z. Broder, M. Charikar, A. Frieze, and M. Mitzenmacher. Min-wise independent permutations. Proc. 30th STOC, pp. 327-336, 1998.

[8] A. Z. Broder, S. C. Glassman, M. S. Manasse, and G. Zweig. Syntactic clustering of the Web. Proc. 6th Int'l World Wide Web Conference, pp. 391-404, 1997.

[9] A. Z. Broder, R. Krauthgamer, and M. Mitzenmacher. Improved classification via connectivity information. Proc. 11th SODA, pp. 576-585, 2000.

[10] G. Calinescu, H. J. Karloff, and Y. Rabani. Approximation algorithms for the 0 -extension problem. Proc. 11th SODA, pp. 8-16, 2000.

[11] C. Chekuri, S. Khanna, J. Naor, and L. Zosin. Approximation algorithms for the metric labeling problem via a new linear programming formulation. Proc. 12th SODA, pp. 109-118, 2001.

[12] Z. Chen, H. V. Jagadish, F. Korn, N. Koudas, S. Muthukrishnan, R. T. Ng, and D. Srivastava. Counting Twig Matches in a Tree. Proc. 17th ICDE pp. 595-604. 2001.

[13] Z. Chen, F. Korn, N. Koudas, and S. Muthukrishnan. Selectivity Estimation for Boolean Queries. Proc. 19th ${PODS}$ ,pp. 216-225, 2000.

[14] E. Cohen, M. Datar, S. Fujiwara, A. Gionis, P. Indyk, R. Motwani, J. D. Ullman, and C. Yang. Finding Interesting Associations without Support Pruning. Proc. 16th ICDE pp. 489-499, 2000.

[15] S. Cohen and L. Guibas. The Earth Mover's Distance under Transformation Sets. Proc. 7th IEEE Intnl. Conf. Computer Vision, 1999.

[16] S. Cohen and L. Guibas. The Earth Mover's Distance: Lower Bounds and Invariance under Translation. Tech. report STAN-CS-TR-97-1597, Dept. of Computer Science, Stanford University, 1997.

[17] L. Engebretsen, P. Indyk and R. O'Donnell. Derandomized dimensionality reduction with applications. To appear in Proc. 13th SODA, 2002.

[18] P. B. Gibbons and Y. Matias. Synopsis Data Structures for Massive Data Sets. Proc. 10th SODA pp. 909-910, 1999.

[19] A. C. Gilbert, Y. Kotidis, S. Muthukrishnan, and M. J. Strauss. Surfing Wavelets on Streams: One-Pass Summaries for Approximate Aggregate Queries. Proc. 27th VLDB pp. 79-88, 2001.

[20] A. C. Gilbert, Y. Kotidis, S. Muthukrishnan, and M. J. Strauss. QuickSAND: Quick Summary and Analysis of Network Data. DIMACS Technical Report 2001-43, November 2001.

[21] A. C. Gilbert, S. Guha, P. Indyk, Y. Kotidis, S. Muthukrishnan, and M. J. Strauss. Fast, Small-Space Algorithms for Approximate Histogram Maintenance. these proceedings.

[22] A. Gionis, D. Gunopulos, and N. Koudas. Efficient

and Tunable Similar Set Retrieval. Proc. SIGMOD Conference 2001.

及可调相似集检索。SIGMOD会议论文集2001。

[23] A. Gionis, P. Indyk, and R. Motwani. Similarity Search in High Dimensions via Hashing. Proc. 25th VLDB pp. 518-529, 1999.

[23] A. Gionis、P. Indyk与R. Motwani。通过哈希实现高维相似性搜索。《第25届VLDB会议论文集》第518-529页，1999年。

[24] M. X. Goemans and D. P. Williamson. Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming. JACM 42(6): 1115-1145, 1995.

[24] M. X. Goemans与D. P. Williamson。利用半定规划改进最大割与可满足性问题的近似算法。《ACM期刊》42(6): 1115-1145，1995年。

[25] S. Guha, N. Mishra, R. Motwani, and L. O'Callaghan. Clustering data streams. Proc. 41st FOCS, pp. 359-366, 2000.

[25] S. Guha、N. Mishra、R. Motwani与L. O'Callaghan。数据流聚类。《第41届FOCS会议论文集》第359-366页，2000年。

[26] A. Gupta and Eva Tardos. A constant factor approximation algorithm for a class of classification problems. Proc. 32nd STOC, pp. 652-658, 2000.

[26] A. Gupta与Eva Tardos。一类分类问题的常数因子近似算法。《第32届STOC会议论文集》第652-658页，2000年。

[27] T. H. Haveliwala, A. Gionis, and P. Indyk. Scalable Techniques for Clustering the Web. Proc. 3rd WebDB, pp. 129-134, 2000.

[27] T. H. Haveliwala、A. Gionis与P. Indyk。可扩展的万维网聚类技术。《第3届WebDB会议论文集》第129-134页，2000年。

[28] P. Indyk. A small approximately min-wise independent family of hash functions. Proc. 10th ${SODA}$ ,pp. 454-456, 1999.

[28] P. Indyk。近似最小独立哈希函数的小型族。《第10届${SODA}$会议论文集》第454-456页，1999年。

[29] P. Indyk. On approximate nearest neighbors in non-Euclidean spaces. Proc. 40th FOCS, pp. 148-155, 1999.

[29] P. Indyk。非欧几里得空间中的近似最近邻。《第40届FOCS会议论文集》第148-155页，1999年。

[30] P. Indyk. Stable Distributions, Pseudorandom Generators, Embeddings and Data Stream Computation. Proc. 41st FOCS, 189-197, 2000.

[30] P. Indyk。稳定分布、伪随机生成器、嵌入与数据流计算。《第41届FOCS会议论文集》第189-197页，2000年。

[31] Indyk, P., Motwani, R. Approximate nearest neighbors: towards removing the curse of dimensionality. Proc. 30th STOC pp. 604-613, 1998.

[31] P. Indyk与R. Motwani。近似最近邻：消除维度灾难的路径。《第30届STOC会议论文集》第604-613页，1998年。

[32] P. Indyk, R. Motwani, P. Raghavan, and S. Vempala. Locality-Preserving Hashing in Multidimensional Spaces. Proc. 29th STOC, pp. 618-625, 1997.

[32] P. Indyk、R. Motwani、P. Raghavan与S. Vempala。多维空间中保持局部性的哈希。《第29届STOC会议论文集》第618-625页，1997年。

[33] J. M. Kleinberg and Éva Tardos Approximation Algorithms for Classification Problems with Pairwise Relationships: Metric Labeling and Markov Random Fields. Proc. 40th FOCS, pp. 14-23, 1999.

[34] N. Linial and O. Sasson. Non-Expansive Hashing. Combinatorica 18(1): 121-132, 1998.

[35] N. Nisan. Pseudorandom sequences for space bounded computations. Combinatorica, 12:449-461, 1992.

[36] Y. Rubner. Perceptual Metrics for Image Database Navigation. Phd Thesis, Stanford University, May 1999

[37] Y. Rubner, L. J. Guibas, and C. Tomasi. The Earth Mover's Distance, Multi-Dimensional Scaling, and Color-Based Image Retrieval. Proc. of the ARPA Image Understanding Workshop, pp. 661-668, 1997.

[38] Y. Rubner, C. Tomasi. Texture Metrics. Proc. IEEE International Conference on Systems, Man, and Cybernetics, 1998, pp. 4601-4607.

[39] Y. Rubner, C. Tomasi, and L. J. Guibas. A Metric for Distributions with Applications to Image Databases. Proc. IEEE Int. Conf. on Computer Vision, pp. 59-66, 1998.

[40] Y. Rubner, C. Tomasi, and L. J. Guibas. The Earth Mover's Distance as a Metric for Image Retrieval. Tech. Report STAN-CS-TN-98-86, Dept. of Computer Science, Stanford University, 1998.

[41] M. Ruzon and C. Tomasi. Color edge detection with the compass operator. Proc. IEEE Conf. Computer Vision and Pattern Recognition, 2:160-166, 1999.

[42] M. Ruzon and C. Tomasi. Corner detection in textured color images. Proc. IEEE Int. Conf. Computer Vision, 2:1039-1045, 1999.