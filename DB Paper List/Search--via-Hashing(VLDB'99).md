# Similarity Search in High Dimensions via Hashing

# 基于哈希的高维相似性搜索

ARISTIDES GIONIS * Piotr Indyk ${}^{ \dagger  }$ RAJEEV MOTWANI ${}^{ \ddagger  }$

阿里斯蒂德斯·吉奥尼斯（Aristides Gionis）* 彼得·因迪克（Piotr Indyk） ${}^{ \dagger  }$ 拉杰夫·莫特瓦尼（Rajeev Motwani） ${}^{ \ddagger  }$

Department of Computer Science

计算机科学系

Stanford University

斯坦福大学

Stanford, CA 94305

加利福尼亚州斯坦福市，邮编 94305

\{gionis,indyk,rajeev\}@cs.stanford.edu

\{gionis,indyk,rajeev\}@cs.stanford.edu

## Abstract

## 摘要

The nearest- or near-neighbor query problems arise in a large variety of database applications, usually in the context of similarity searching. Of late, there has been increasing interest in building search/index structures for performing similarity search over high-dimensional data, e.g., image databases, document collections, time-series databases, and genome databases. Unfortunately, all known techniques for solving this problem fall prey to the "curse of dimensionality." That is, the data structures scale poorly with data dimensionality; in fact, if the number of dimensions exceeds 10 to 20,searching in $k$ -d trees and related structures involves the inspection of a large fraction of the database, thereby doing no better than brute-force linear search. It has been suggested that since the selection of features and the choice of a distance metric in typical applications is rather heuristic, determining an approximate nearest neighbor should suffice for most practical purposes. In this paper, we examine a novel scheme for approximate similarity search based on hashing. The basic idea is to hash the points from the database so as to ensure that the probability of collision is much higher for objects that are close to each other than for those that are far apart. We provide experimental evidence that our method gives significant improvement in running time over other methods for searching in high-dimensional spaces based on hierarchical tree decomposition. Experimental results also indicate that our scheme scales well even for a relatively large number of dimensions (more than 50).

最近邻或近邻查询问题出现在各种各样的数据库应用中，通常是在相似性搜索的背景下。近年来，人们对构建用于在高维数据（例如图像数据库、文档集合、时间序列数据库和基因组数据库）上执行相似性搜索的搜索/索引结构越来越感兴趣。不幸的是，所有已知的解决此问题的技术都受到“维度诅咒”的影响。也就是说，数据结构随着数据维度的增加而扩展性很差；实际上，如果维度数超过 10 到 20，在 $k$ -d 树和相关结构中进行搜索需要检查数据库的很大一部分，因此并不比暴力线性搜索好多少。有人建议，由于在典型应用中特征的选择和距离度量的选择相当依赖启发式方法，因此对于大多数实际目的而言，确定近似最近邻就足够了。在本文中，我们研究了一种基于哈希的近似相似性搜索的新方案。基本思想是对数据库中的点进行哈希处理，以确保彼此接近的对象发生碰撞的概率比相距较远的对象高得多。我们通过实验证明，与其他基于层次树分解的高维空间搜索方法相比，我们的方法在运行时间上有显著改进。实验结果还表明，即使对于相对较多的维度（超过 50），我们的方案也具有良好的扩展性。

## 1 Introduction

## 1 引言

A similarity search problem involves a collection of objects (e.g., documents, images) that are characterized by a collection of relevant features and represented as points in a high-dimensional attribute space; given queries in the form of points in this space, we are required to find the nearest (most similar) object to the query. The particularly interesting and well-studied case is the $d$ -dimensional Euclidean space. The problem is of major importance to a variety of applications; some examples are: data compression [20]; databases and data mining [21]; information retrieval [11, 16, 38]; image and video databases $\left\lbrack  {{15},{17},{37},{42}}\right\rbrack$ ; machine learning [7]; pattern recognition [9, 13]; and, statistics and data analysis $\left\lbrack  {{12},{27}}\right\rbrack$ . Typically,the features of the objects of interest are represented as points in ${\Re }^{d}$ and a distance metric is used to measure similarity of objects. The basic problem then is to perform indexing or similarity searching for query objects. The number of features (i.e., the dimensionality) ranges anywhere from tens to thousands. For example, in multimedia applications such as IBM's QBIC (Query by Image Content), the number of features could be several hundreds $\left\lbrack  {{15},{17}}\right\rbrack$ . In information retrieval for text documents, vector-space representations involve several thousands of dimensions, and it is considered to be a dramatic improvement that dimension-reduction techniques, such as the Karhunen-Loéve transform [26, 30] (also known as principal components analysis [22] or latent semantic indexing [11]), can reduce the dimensionality to a mere few hundreds!

相似性搜索问题涉及一组对象（例如文档、图像），这些对象由一组相关特征来表征，并表示为高维属性空间中的点；给定该空间中以点的形式表示的查询，我们需要找到与查询最接近（最相似）的对象。特别有趣且研究充分的情况是 $d$ 维欧几里得空间。这个问题对各种应用都非常重要；一些例子包括：数据压缩 [20]；数据库和数据挖掘 [21]；信息检索 [11, 16, 38]；图像和视频数据库 $\left\lbrack  {{15},{17},{37},{42}}\right\rbrack$；机器学习 [7]；模式识别 [9, 13]；以及统计和数据分析 $\left\lbrack  {{12},{27}}\right\rbrack$。通常，感兴趣的对象的特征表示为 ${\Re }^{d}$ 中的点，并使用距离度量来衡量对象的相似性。那么基本问题就是对查询对象进行索引或相似性搜索。特征的数量（即维度）从几十到几千不等。例如，在像 IBM 的 QBIC（基于图像内容的查询）这样的多媒体应用中，特征的数量可能有几百个 $\left\lbrack  {{15},{17}}\right\rbrack$。在文本文档的信息检索中，向量空间表示涉及数千个维度，而诸如卡尔胡宁 - 洛埃夫变换 [26, 30]（也称为主成分分析 [22] 或潜在语义索引 [11]）这样的降维技术能够将维度降低到仅仅几百个，这被认为是一个巨大的改进！

---

<!-- Footnote -->

*Supported by NAVY N00014-96-1-1221 grant and NSF Grant IIS-9811904.

*由海军 N00014 - 96 - 1 - 1221 资助以及美国国家科学基金会 IIS - 9811904 资助。

${}^{ \dagger  }$ Supported by Stanford Graduate Fellowship and NSF NYI Award CCR-9357849.

${}^{ \dagger  }$ 由斯坦福研究生奖学金和美国国家科学基金会 NYI 奖 CCR - 9357849 资助。

*Supported by ARO MURI Grant DAAH04-96-1-0007, NSF Grant IIS-9811904, and NSF Young Investigator Award CCR- 9357849, with matching funds from IBM, Mitsubishi, Schlum-berger Foundation, Shell Foundation, and Xerox Corporation.

*由陆军研究办公室多学科大学研究计划（ARO MURI）资助 DAAH04 - 96 - 1 - 0007、美国国家科学基金会 IIS - 9811904 资助以及美国国家科学基金会青年研究员奖 CCR - 9357849 资助，同时获得了来自 IBM、三菱、斯伦贝谢基金会、壳牌基金会和施乐公司的配套资金。

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the VLDB copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Very Large Data Base Endowment. To copy otherwise, or to republish, requires a fee and/or special permission from the Endowment. Proceedings of the 25th VLDB Conference, Edinburgh, Scotland, 1999.

允许免费复制本材料的全部或部分内容，前提是复制的目的不是为了直接商业利益，要显示 VLDB 版权声明、出版物的标题及其日期，并注明复制是经大型数据库捐赠基金许可的。否则，进行复制或重新发布需要支付费用和/或获得捐赠基金的特别许可。第 25 届 VLDB 会议论文集，苏格兰爱丁堡，1999 年。

<!-- Footnote -->

---

The low-dimensional case (say,for $d$ equal to 2 or 3) is well-solved [14], so the main issue is that of dealing with a large number of dimensions, the so-called "curse of dimensionality." Despite decades of intensive effort, the current solutions are not entirely satisfactory; in fact,for large enough $d$ ,in theory or in practice, they provide little improvement over a linear algorithm which compares a query to each point from the database. In particular, it was shown in [45] that, both empirically and theoretically, all current indexing techniques (based on space partitioning) degrade to linear search for sufficiently high dimensions. This situation poses a serious obstacle to the future development of large scale similarity search systems. Imagine for example a search engine which enables content-based image retrieval on the World-Wide Web. If the system was to index a significant fraction of the web, the number of images to index would be at least of the order tens (if not hundreds) of million. Clearly, no indexing method exhibiting linear (or close to linear) dependence on the data size could manage such a huge data set.

低维情况（例如，当$d$等于2或3时）已得到很好的解决[14]，因此主要问题在于处理大量维度，即所谓的“维度灾难”。尽管经过了数十年的深入研究，目前的解决方案仍不尽如人意；事实上，对于足够大的$d$，无论是在理论上还是实践中，它们相较于将查询与数据库中的每个点进行比较的线性算法，改进甚微。特别是，文献[45]表明，从经验和理论两方面来看，所有当前的索引技术（基于空间划分）在维度足够高时都会退化为线性搜索。这种情况对大规模相似性搜索系统的未来发展构成了严重障碍。例如，设想一个搜索引擎能够在万维网上进行基于内容的图像检索。如果该系统要对网络上相当一部分图像进行索引，那么需要索引的图像数量至少会达到数千万（甚至数亿）。显然，任何对数据规模呈线性（或接近线性）依赖的索引方法都无法处理如此庞大的数据集。

The premise of this paper is that in many cases it is not necessary to insist on the exact answer; instead, determining an approximate answer should suffice (refer to Section 2 for a formal definition). This observation underlies a large body of recent research in databases, including using random sampling for histogram estimation [8] and median approximation [33], using wavelets for selectivity estimation [34] and approximate SVD [25]. We observe that there are many applications of nearest neighbor search where an approximate answer is good enough. For example, it often happens (e.g., see [23]) that the relevant answers are much closer to the query point than the irrelevant ones; in fact, this is a desirable property of a good similarity measure. In such cases, the approximate algorithm (with a suitable approximation factor) will return the same result as an exact algorithm. In other situations, an approximate algorithm provides the user with a time-quality tradeoff - the user can decide whether to spend more time waiting for the exact answer, or to be satisfied with a much quicker approximation (e.g., see [5]).

本文的前提是，在许多情况下，不必坚持要求得到精确答案；相反，确定一个近似答案就足够了（关于正式定义，请参考第2节）。这一观点是近期数据库领域大量研究的基础，包括使用随机抽样进行直方图估计[8]和中位数近似[33]，使用小波进行选择性估计[34]和近似奇异值分解（SVD）[25]。我们注意到，在最近邻搜索的许多应用中，近似答案就已经足够好了。例如，经常会出现（例如，参见[23]）相关答案比不相关答案更接近查询点的情况；事实上，这是一个良好的相似性度量所应具备的理想属性。在这种情况下，近似算法（具有合适的近似因子）将返回与精确算法相同的结果。在其他情况下，近似算法为用户提供了时间 - 质量的权衡——用户可以决定是花费更多时间等待精确答案，还是满足于更快得到的近似答案（例如，参见[5]）。

The above arguments rely on the assumption that approximate similarity search can be performed much faster than the exact one. In this paper we show that this is indeed the case. Specifically, we introduce a new indexing method for approximate nearest neighbor with a truly sublinear dependence on the data size even for high-dimensional data. Instead of using space partitioning, it relies on a new method called locality-sensitive hashing (LSH). The key idea is to hash the points using several hash functions so as to ensure that, for each function, the probability of collision is much higher for objects which are close to each other than for those which are far apart. Then, one can determine near neighbors by hashing the query point and retrieving elements stored in buckets containing that point. We provide such locality-sensitive hash functions that are simple and easy to implement; they can also be naturally extended to the dynamic setting, i.e., when insertion and deletion operations also need to be supported. Although in this paper we are focused on Euclidean spaces, different LSH functions can be also used for other similarity measures, such as dot product $\left\lbrack  5\right\rbrack$ .

上述论点依赖于这样一个假设，即近似相似性搜索的执行速度可以比精确搜索快得多。在本文中，我们证明了情况确实如此。具体来说，我们引入了一种新的近似最近邻索引方法，即使对于高维数据，该方法对数据规模的依赖也是真正的亚线性的。它不使用空间划分，而是依赖于一种称为局部敏感哈希（LSH）的新方法。其关键思想是使用多个哈希函数对这些点进行哈希处理，以确保对于每个函数，彼此接近的对象发生碰撞的概率远高于彼此远离的对象。然后，可以通过对查询点进行哈希处理并检索存储在包含该点的桶中的元素来确定近邻。我们提供了这样的局部敏感哈希函数，它们简单且易于实现；它们也可以自然地扩展到动态环境中，即当还需要支持插入和删除操作时。尽管在本文中我们主要关注欧几里得空间，但不同的LSH函数也可用于其他相似性度量，如点积$\left\lbrack  5\right\rbrack$。

Locality-Sensitive Hashing was introduced by Indyk and Motwani [24] for the purposes of devising main memory algorithms for nearest neighbor search; in particular,it enabled us to achieve worst-case $O\left( {d{n}^{1/\epsilon }}\right)$ - time for approximate nearest neighbor query over an $n$ -point database. In this paper we improve that technique and achieve a significantly improved query time of $O\left( {d{n}^{1/\left( {1 + \epsilon }\right) }}\right)$ . This yields an approximate nearest neighbor algorithm running in sublinear time for any $\epsilon  > 0$ . Furthermore,we generalize the algorithm and its analysis to the case of external memory.

Indyk和Motwani[24]引入了局部敏感哈希（Locality - Sensitive Hashing），用于设计用于最近邻搜索的主存算法；特别是，它使我们能够在一个包含$n$个点的数据库上对近似最近邻查询实现最坏情况下的$O\left( {d{n}^{1/\epsilon }}\right)$时间复杂度。在本文中，我们改进了该技术，将查询时间显著提高到$O\left( {d{n}^{1/\left( {1 + \epsilon }\right) }}\right)$。这产生了一种对于任何$\epsilon  > 0$都能以亚线性时间运行的近似最近邻算法。此外，我们将该算法及其分析推广到外部内存的情况。

We support our theoretical arguments by empirical evidence. We performed experiments on two data sets. The first contains 20,000 histograms of color images, where each histogram was represented as a point in $d$ -dimensional space,for $d$ up to 64 . The second contains around 270,000 points representing texture information of blocks of large aerial photographs. All our tables were stored on disk. We compared the performance of our algorithm with the performance of the Sphere/Rectangle-tree (SR-tree) [28], a recent data structure which was shown to be comparable to or significantly more efficient than other tree-decomposition-based indexing methods for spatial data. The experiments show that our algorithm is significantly faster than the earlier methods, in some cases even by several orders of magnitude. It also scales well as the data size and dimensionality increase. Thus, it enables a new approach to high-performance similarity search - fast retrieval of approximate answer, possibly followed by a slower but more accurate computation in the few cases where the user is not satisfied with the approximate answer.

我们用实证证据支持我们的理论观点。我们在两个数据集上进行了实验。第一个数据集包含20,000张彩色图像的直方图，其中每个直方图都表示为$d$维空间中的一个点，$d$最大为64。第二个数据集包含大约270,000个点，这些点代表了大幅航空照片中各个区块的纹理信息。我们所有的表格都存储在磁盘上。我们将我们算法的性能与球体/矩形树（SR树）[28]的性能进行了比较，SR树是一种最近提出的数据结构，已被证明与其他基于树分解的空间数据索引方法相当，甚至在效率上有显著提升。实验表明，我们的算法比早期的方法快得多，在某些情况下甚至快几个数量级。随着数据规模和维度的增加，它的扩展性也很好。因此，它为高性能相似性搜索提供了一种新方法——快速检索近似答案，在用户对近似答案不满意的少数情况下，可能随后进行较慢但更精确的计算。

The rest of this paper is organized as follows. In Section 2 we introduce the notation and give formal definitions of the similarity search problems. Then in Section 3 we describe locality-sensitive hashing and show how to apply it to nearest neighbor search. In Section 4 we report the results of experiments with LSH. The related work is described in Section 5. Finally, in Section 6 we present conclusions and ideas for future research.

本文的其余部分组织如下。在第2节中，我们介绍符号并给出相似性搜索问题的正式定义。然后在第3节中，我们描述局部敏感哈希并展示如何将其应用于最近邻搜索。在第4节中，我们报告局部敏感哈希（LSH）的实验结果。相关工作在第5节中描述。最后，在第6节中，我们给出结论和未来研究的思路。

## 2 Preliminaries

## 2 预备知识

We use ${l}_{p}^{d}$ to denote the Euclidean space ${\Re }^{d}$ under the ${l}_{p}$ norm,i.e.,when the length of a vector $\left( {{x}_{1},\ldots {x}_{d}}\right)$ is defined as ${\left( {\left| {x}_{1}\right| }^{p} + \ldots  + {\left| {x}_{d}\right| }^{p}\right) }^{1/p}$ . Further, ${d}_{p}\left( {p,q}\right)  =$ $\parallel p - q{\parallel }_{p}$ denotes the distance between the points $p$ and $q$ in ${l}_{p}^{d}$ . We use ${H}^{d}$ to denote the Hamming metric space of dimension $d$ ,i.e.,the space of binary vectors of length $d$ under the standard Hamming metric. We use ${d}_{H}\left( {p,q}\right)$ denote the Hamming distance,i.e.,the number of bits on which $p$ and $q$ differ.

我们用${l}_{p}^{d}$表示在${l}_{p}$范数下的欧几里得空间${\Re }^{d}$，即当向量$\left( {{x}_{1},\ldots {x}_{d}}\right)$的长度定义为${\left( {\left| {x}_{1}\right| }^{p} + \ldots  + {\left| {x}_{d}\right| }^{p}\right) }^{1/p}$时。此外，${d}_{p}\left( {p,q}\right)  =$ $\parallel p - q{\parallel }_{p}$表示${l}_{p}^{d}$中$p$和$q$两点之间的距离。我们用${H}^{d}$表示维度为$d$的汉明度量空间，即在标准汉明度量下长度为$d$的二进制向量空间。我们用${d}_{H}\left( {p,q}\right)$表示汉明距离，即$p$和$q$不同的比特位数。

The nearest neighbor search problem is defined as follows:

最近邻搜索问题定义如下：

Definition 1 (Nearest Neighbor Search (NNS)) Given a set $P$ of $n$ objects represented as points in a normed space ${l}_{p}^{d}$ ,preprocess $P$ so as to efficiently answer queries by finding the point in $P$ closest to a query point $q$ .

定义1（最近邻搜索（NNS）） 给定一组由$n$个对象组成的集合$P$，这些对象在赋范空间${l}_{p}^{d}$中表示为点，对$P$进行预处理，以便通过找到$P$中最接近查询点$q$的点来高效地回答查询。

The definition generalizes naturally to the case where we want to return $K > 1$ points. Specifically,in the $K$ -Nearest Neighbors Search ( $K$ -NNS),we wish to return the $K$ points in the database that are closest to the query point. The approximate version of the NNS problem is defined as follows:

该定义自然地推广到我们希望返回$K > 1$个点的情况。具体来说，在$K$ -最近邻搜索（$K$ -NNS）中，我们希望返回数据库中最接近查询点的$K$个点。最近邻搜索问题的近似版本定义如下：

Definition 2 ( $\epsilon$ -Nearest Neighbor Search ( $\epsilon$ -NNS) Given a set $P$ of points in a normed space ${l}_{p}^{d}$ ,preprocess $P$ so as to efficiently return a point $p \in  P$ for any given query point $q$ ,such that $d\left( {q,p}\right)  \leq  \left( {1 + \epsilon }\right) d\left( {q,P}\right)$ , where $d\left( {q,P}\right)$ is the distance of $q$ to the its closest point in $P$ .

定义2（$\epsilon$ -最近邻搜索（$\epsilon$ -NNS）） 给定赋范空间${l}_{p}^{d}$中的一组点$P$，对$P$进行预处理，以便对于任何给定的查询点$q$，高效地返回一个点$p \in  P$，使得$d\left( {q,p}\right)  \leq  \left( {1 + \epsilon }\right) d\left( {q,P}\right)$，其中$d\left( {q,P}\right)$是$q$到其在$P$中最近点的距离。

Again, this definition generalizes naturally to finding $K > 1$ approximate nearest neighbors. In the ${Ap}$ - proximate $K$ -NNS problem,we wish to find $K$ points ${p}_{1},\ldots ,{p}_{K}$ such that the distance of ${p}_{i}$ to the query $q$ is at most $\left( {1 + \epsilon }\right)$ times the distance from the $i$ th nearest point to $q$ .

同样，这个定义自然地推广到寻找$K > 1$个近似最近邻。在${Ap}$ -近似$K$ -NNS问题中，我们希望找到$K$个点${p}_{1},\ldots ,{p}_{K}$，使得${p}_{i}$到查询点$q$的距离至多是第$i$个最近点到$q$的距离的$\left( {1 + \epsilon }\right)$倍。

## 3 The Algorithm

## 3 算法

In this section we present efficient solutions to the approximate versions of the NNS problem. Without significant loss of generality, we will make the following two assumptions about the data:

在本节中，我们将为最近邻搜索（NNS）问题的近似版本提供高效的解决方案。在不失一般性的前提下，我们对数据做出以下两个假设：

1. the distance is defined by the ${l}_{1}$ norm (see comments below),

1. 距离由${l}_{1}$范数定义（见下文注释），

2. all coordinates of points in $P$ are positive integers.

2. $P$中所有点的坐标均为正整数。

The first assumption is not very restrictive, as usually there is no clear advantage in, or even difference between,using ${l}_{2}$ or ${l}_{1}$ norm for similarity search. For example, the experiments done for the Webseek [43] project (see [40], chapter 4) show that comparing color histograms using ${l}_{1}$ and ${l}_{2}$ norms yields very similar results $\left( {l}_{1}\right.$ is marginally better). Both our data sets (see Section 4) have a similar property. Specifically, we observed that a nearest neighbor of an average query point computed under the ${l}_{1}$ norm was also an $\epsilon$ -approximate neighbor under the ${l}_{2}$ norm with an average value of $\epsilon$ less than $3\%$ (this observation holds for both data sets). Moreover, in most cases (i.e., for ${67}\%$ of the queries in the first set and ${73}\%$ in the second set) the nearest neighbors under ${l}_{1}$ and ${l}_{2}$ norms were exactly the same. This observation is interesting in its own right, and can be partially explained via the theorem by Figiel et al (see [19] and references therein). They showed analytically that by simply applying scaling and random rotation to the space ${l}_{2}$ , we can make the distances induced by the ${l}_{1}$ and ${l}_{2}$ norms almost equal up to an arbitrarily small factor. It seems plausible that real data is already randomly rotated,thus the difference between ${l}_{1}$ and ${l}_{2}$ norm is very small. Moreover, for the data sets for which this property does not hold, we are guaranteed that after performing scaling and random rotation our algorithms can be used for the ${l}_{2}$ norm with arbitrarily small loss of precision.

第一个假设的限制并不大，因为在相似性搜索中，使用${l}_{2}$范数或${l}_{1}$范数通常没有明显的优势，甚至没有区别。例如，为Webseek [43]项目所做的实验（见[40]，第4章）表明，使用${l}_{1}$范数和${l}_{2}$范数比较颜色直方图会得到非常相似的结果（$\left( {l}_{1}\right.$略好一些）。我们的两个数据集（见第4节）也有类似的性质。具体来说，我们观察到，在${l}_{1}$范数下计算得到的平均查询点的最近邻，在${l}_{2}$范数下也是一个$\epsilon$ -近似邻点，且$\epsilon$的平均值小于$3\%$（这一观察结果对两个数据集都成立）。此外，在大多数情况下（即，第一组中${67}\%$的查询和第二组中${73}\%$的查询），${l}_{1}$范数和${l}_{2}$范数下的最近邻完全相同。这一观察结果本身就很有趣，并且可以通过Figiel等人的定理（见[19]及其中的参考文献）部分解释。他们通过分析表明，只需对空间${l}_{2}$进行缩放和随机旋转，我们就可以使${l}_{1}$范数和${l}_{2}$范数所诱导的距离在任意小的因子范围内几乎相等。似乎真实数据已经是随机旋转的，因此${l}_{1}$范数和${l}_{2}$范数之间的差异非常小。此外，对于不具备这一性质的数据集，我们可以保证，在进行缩放和随机旋转后，我们的算法可以用于${l}_{2}$范数，且精度损失任意小。

As far as the second assumption is concerned, clearly all coordinates can be made positive by properly translating the origin of ${\Re }^{d}$ . We can then convert all coordinates to integers by multiplying them by a suitably large number and rounding to the nearest integer. It can be easily verified that by choosing proper parameters, the error induced by rounding can be made arbitrarily small. Notice that after this operation the minimum interpoint distance is 1 .

至于第二个假设，显然可以通过适当地平移${\Re }^{d}$的原点，使所有坐标变为正数。然后，我们可以将所有坐标乘以一个足够大的数并四舍五入到最接近的整数，从而将其转换为整数。可以很容易地验证，通过选择合适的参数，四舍五入所引起的误差可以任意小。请注意，经过此操作后，点与点之间的最小距离为1。

### 3.1 Locality-Sensitive Hashing

### 3.1 局部敏感哈希

In this section we present locality-sensitive hashing (LSH). This technique was originally introduced by Indyk and Motwani [24] for the purposes of devising main memory algorithms for the $\epsilon$ -NNS problem. Here we give an improved version of their algorithm. The new algorithm is in many respects more natural than the earlier one: it does not require the hash buckets to store only one point; it has better running time guarantees; and, the analysis is generalized to the case of secondary memory.

在本节中，我们将介绍局部敏感哈希（LSH）。这项技术最初由Indyk和Motwani [24]引入，用于为$\epsilon$ -最近邻搜索问题设计主存算法。在这里，我们给出他们算法的一个改进版本。新算法在很多方面比早期的算法更自然：它不要求哈希桶只存储一个点；它有更好的运行时间保证；并且，分析被推广到了二级存储的情况。

Let $C$ be the largest coordinate in all points in $P$ . Then,as per [29],we can embed $P$ into the Hamming cube ${H}^{{d}^{\prime }}$ with ${d}^{\prime } = {Cd}$ ,by transforming each point $p = \left( {{x}_{1},\ldots {x}_{d}}\right)$ into a binary vector

设$C$是$P$中所有点的最大坐标。那么，根据[29]，我们可以通过将每个点$p = \left( {{x}_{1},\ldots {x}_{d}}\right)$转换为一个二进制向量，将$P$嵌入到汉明立方体${H}^{{d}^{\prime }}$中，其中${d}^{\prime } = {Cd}$ 。

$$
v\left( p\right)  = {\operatorname{Unary}}_{C}\left( {x}_{1}\right) \ldots {\operatorname{Unary}}_{C}\left( {x}_{d}\right) ,
$$

where ${\operatorname{Unary}}_{C}\left( x\right)$ denotes the unary representation of $x$ ,i.e.,is a sequence of $x$ ones followed by $C - x$ zeroes.

其中${\operatorname{Unary}}_{C}\left( x\right)$表示$x$的一元表示，即，是一个由$x$个1后面跟着$C - x$个0组成的序列。

Fact 1 For any pair of points $p,q$ with coordinates in the set $\{ 1\ldots C\}$ ,

事实1 对于坐标在集合$\{ 1\ldots C\}$中的任意一对点$p,q$，

$$
{d}_{1}\left( {p,q}\right)  = {d}_{H}\left( {v\left( p\right) ,v\left( q\right) }\right) .
$$

That is, the embedding preserves the distances between the points. Therefore, in the sequel we can concentrate on solving $\epsilon$ -NNS in the Hamming space ${H}^{{d}^{\prime }}$ . However,we emphasize that we do not need to actually convert the data to the unary representation, which could be expensive when $C$ is large; in fact,all our algorithms can be made to run in time independent on $C$ . Rather,the unary representation provides us with a convenient framework for description of the algorithms which would be more complicated otherwise.

也就是说，嵌入操作保留了点之间的距离。因此，在后续内容中，我们可以专注于在汉明空间 ${H}^{{d}^{\prime }}$ 中解决 $\epsilon$ -近邻搜索（Nearest Neighbor Search，NNS）问题。然而，我们强调不需要实际将数据转换为一元表示，当 $C$ 很大时，这种转换可能代价高昂；实际上，我们所有的算法都可以在与 $C$ 无关的时间内运行。相反，一元表示为我们提供了一个方便的框架来描述算法，否则这些算法会更加复杂。

We define the hash functions as follows. For an integer $l$ to be specified later,choose $l$ subsets ${I}_{1},\ldots ,{I}_{l}$ of $\left\{  {1,\ldots ,{d}^{\prime }}\right\}$ . Let ${p}_{\mid I}$ denote the projection of vector $p$ on the coordinate set $I$ ,i.e.,we compute ${p}_{\mid I}$ by selecting the coordinate positions as per $I$ and concatenating the bits in those positions. Denote ${g}_{j}\left( p\right)  = {p}_{\mid {I}_{j}}$ . For the preprocessing,we store each $p \in  P$ in the bucket ${g}_{j}\left( p\right)$ ,for $j = 1,\ldots ,l$ . As the total number of buckets may be large, we compress the buckets by resorting to standard hashing. Thus, we use two levels of hashing: the LSH function maps a point $p$ to bucket ${g}_{j}\left( p\right)$ , and a standard hash function maps the contents of these buckets into a hash table of size $M$ . The maximal bucket size of the latter hash table is denoted by $B$ . For the algorithm’s analysis,we will assume hashing with chaining, i.e., when the number of points in a bucket exceeds $B$ ,a new bucket (also of size $B$ ) is allocated and linked to and from the old bucket. However, our implementation does not employ chaining, but relies on a simpler approach: if a bucket in a given index is full, a new point cannot be added to it, since it will be added to some other index with high probability. This saves us the overhead of maintaining the link structure.

我们如下定义哈希函数。对于稍后要指定的整数 $l$，选择 $\left\{  {1,\ldots ,{d}^{\prime }}\right\}$ 的 $l$ 个子集 ${I}_{1},\ldots ,{I}_{l}$。令 ${p}_{\mid I}$ 表示向量 $p$ 在坐标集 $I$ 上的投影，即我们根据 $I$ 选择坐标位置并连接这些位置上的比特来计算 ${p}_{\mid I}$。记为 ${g}_{j}\left( p\right)  = {p}_{\mid {I}_{j}}$。在预处理阶段，对于 $j = 1,\ldots ,l$，我们将每个 $p \in  P$ 存储在桶 ${g}_{j}\left( p\right)$ 中。由于桶的总数可能很大，我们通过采用标准哈希来压缩这些桶。因此，我们使用两级哈希：局部敏感哈希（Locality-Sensitive Hashing，LSH）函数将点 $p$ 映射到桶 ${g}_{j}\left( p\right)$，而标准哈希函数将这些桶的内容映射到大小为 $M$ 的哈希表中。后一个哈希表的最大桶大小记为 $B$。为了对算法进行分析，我们将假设采用链式哈希，即当一个桶中的点数超过 $B$ 时，会分配一个新的桶（大小也为 $B$）并与旧桶建立链接。然而，我们的实现并不采用链式方法，而是依赖于一种更简单的方法：如果给定索引中的一个桶已满，就不能再向其中添加新点，因为该点很可能会被添加到其他索引中。这样可以节省维护链接结构的开销。

The number $n$ of points,the size $M$ of the hash table,and the maximum bucket size $B$ are related by the following equation:

点的数量 $n$、哈希表的大小 $M$ 和最大桶大小 $B$ 由以下方程关联：

$$
M = \alpha \frac{n}{B}
$$

where $\alpha$ is the memory utilization parameter,i.e.,the ratio of the memory allocated for the index to the size of the data set.

其中 $\alpha$ 是内存利用率参数，即分配给索引的内存与数据集大小的比率。

To process a query $q$ ,we search all indices ${g}_{1}\left( q\right) ,\ldots ,{g}_{l}\left( q\right)$ until we either encounter at least $c \cdot  l$ points (for $c$ specified later) or use all $l$ indices. Clearly, the number of disk accesses is always upper bounded by the number of indices,which is equal to $l$ . Let ${p}_{1},\ldots ,{p}_{t}$ be the points encountered in the process. For Approximate $K$ -NNS,we output the $K$ points ${p}_{i}$ closest to $q$ ; in general,we may return fewer points if the number of points encountered is less than $K$ .

为了处理查询 $q$，我们搜索所有索引 ${g}_{1}\left( q\right) ,\ldots ,{g}_{l}\left( q\right)$，直到我们遇到至少 $c \cdot  l$ 个点（$c$ 稍后指定）或者使用完所有 $l$ 个索引。显然，磁盘访问次数总是上限为索引的数量，即 $l$。令 ${p}_{1},\ldots ,{p}_{t}$ 为该过程中遇到的点。对于近似 $K$ -近邻搜索，我们输出与 $q$ 最接近的 $K$ 个点 ${p}_{i}$；一般来说，如果遇到的点数少于 $K$，我们可能返回更少的点。

It remains to specify the choice of the subsets ${I}_{j}$ . For each $j \in  \{ 1,\ldots ,l\}$ ,the set ${I}_{j}$ consists of $k$ elements from $\left\{  {1,\ldots ,{d}^{\prime }}\right\}$ sampled uniformly at random with replacement. The optimal value of $k$ is chosen to maximize the probability that a point $p$ "close" to $q$ will fall into the same bucket as $q$ ,and also to minimize the probability that a point ${p}^{\prime }$ "far away" from $q$ will fall into the same bucket. The choice of the values of $l$ and $k$ is deferred to the next section.

还需要指定子集 ${I}_{j}$ 的选择。对于每个 $j \in  \{ 1,\ldots ,l\}$，集合 ${I}_{j}$ 由从 $\left\{  {1,\ldots ,{d}^{\prime }}\right\}$ 中带放回地均匀随机采样的 $k$ 个元素组成。选择 $k$ 的最优值是为了最大化与 $q$ “接近”的点 $p$ 落入与 $q$ 相同桶的概率，同时最小化与 $q$ “远离”的点 ${p}^{\prime }$ 落入相同桶的概率。$l$ 和 $k$ 的值的选择将推迟到下一节。

<!-- Media -->

---

Algorithm Preprocessing

算法 预处理

Input A set of points $P$ ,

输入 一组点 $P$，

	$l$ (number of hash tables),

	$l$（哈希表的数量），

Output Hash tables ${\mathcal{T}}_{i},i = 1,\ldots ,l$

输出哈希表 ${\mathcal{T}}_{i},i = 1,\ldots ,l$

Foreach $i = 1,\ldots ,l$

遍历 $i = 1,\ldots ,l$

	Initialize hash table ${\mathcal{T}}_{i}$ by generating

	通过生成随机哈希函数 ${\mathcal{T}}_{i}$ 初始化哈希表 ${\mathcal{T}}_{i}$

	a random hash function ${g}_{i}\left( \cdot \right)$

	一个随机哈希函数 ${g}_{i}\left( \cdot \right)$

Foreach $i = 1,\ldots ,l$

遍历 $i = 1,\ldots ,l$

	Foreach $j = 1,\ldots ,n$

	遍历 $j = 1,\ldots ,n$

		Store point ${p}_{j}$ on bucket ${g}_{i}\left( {p}_{j}\right)$ of hash table ${\mathcal{T}}_{i}$

			将点 ${p}_{j}$ 存储在哈希表 ${\mathcal{T}}_{i}$ 的桶 ${g}_{i}\left( {p}_{j}\right)$ 上

---

Figure 1: Preprocessing algorithm for points already embedded in the Hamming cube.

图1：已嵌入汉明立方体（Hamming cube）的点的预处理算法。

---

Algorithm Approximate Nearest Neighbor Query

近似最近邻查询算法

Input A query point $q$ ,

输入一个查询点 $q$ ，

	$K$ (number of appr. nearest neighbors)

	$K$ （近似最近邻的数量）

Access To hash tables ${\mathcal{T}}_{i},i = 1,\ldots ,l$

访问由预处理算法生成的哈希表 ${\mathcal{T}}_{i},i = 1,\ldots ,l$

	generated by the preprocessing algorithm

	由预处理算法生成

Output $K$ (or less) appr. nearest neighbors

输出 $K$ （或更少）个近似最近邻

$S \leftarrow  \varnothing$

Foreach $i = 1,\ldots ,l$

遍历 $i = 1,\ldots ,l$

	$S \leftarrow  S \cup  \{$ points found in ${g}_{i}\left( q\right)$ bucket of table ${\mathcal{T}}_{i}\}$

	在表 ${\mathcal{T}}_{i}\}$ 的桶 ${g}_{i}\left( q\right)$ 中找到的 $S \leftarrow  S \cup  \{$ 个点

Return the $K$ nearest neighbors of $q$ found in set $S$

返回在集合$S$中找到的$q$的$K$个最近邻

/* Can be found by main memory linear search */

/* 可以通过主存线性搜索找到 */

---

Figure 2: Approximate Nearest Neighbor query answering algorithm.

图2：近似最近邻查询应答算法。

<!-- Media -->

Although we are mainly interested in the $1/\mathrm{O}$ complexity of our scheme, it is worth pointing out that the hash functions can be efficiently computed if the data set is obtained by mapping ${l}_{1}^{d}$ into ${d}^{\prime }$ -dimensional Hamming space. Let $p$ be any point from the data set and let ${p}^{\prime }$ denote its image after the mapping. Let $I$ be the set of coordinates and recall that we need to compute ${p}_{\mid I}^{\prime }$ . For $i = 1,\ldots ,d$ ,let ${I}_{\mid i}$ denote,in sorted order,the coordinates in $I$ which correspond to the $i$ th coordinate of $p$ . Observe,that projecting ${p}^{\prime }$ on ${I}_{\mid i}$ results in a sequence of bits which is monotone, i.e., consists of a number,say ${o}_{i}$ ,of ones followed by zeros. Therefore,in order to represent ${p}_{I}^{\prime }$ it is sufficient to compute ${o}_{i}$ for $i = 1,\ldots ,d$ . However,the latter task is equivalent to finding the number of elements in the sorted array ${I}_{\mid i}$ which are smaller than a given value,i.e.,the $i$ th coordinate of $p$ . This can be done via binary search in $\log C$ time,or even in constant time using a precomputed array of $C$ bits. Thus,the total time needed to compute the function is either $O\left( {d\log C}\right)$ or $O\left( d\right)$ ,depending on resources used. In our experimental section,the value of $C$ can be made very small, and therefore we will resort to the second method.

尽管我们主要关注我们方案的$1/\mathrm{O}$复杂度，但值得指出的是，如果数据集是通过将${l}_{1}^{d}$映射到${d}^{\prime }$维汉明空间（Hamming space）得到的，那么哈希函数可以被高效计算。设$p$是数据集中的任意一点，${p}^{\prime }$表示其映射后的像。设$I$是坐标集合，并且回顾我们需要计算${p}_{\mid I}^{\prime }$。对于$i = 1,\ldots ,d$，设${I}_{\mid i}$按排序顺序表示$I$中对应于$p$的第$i$个坐标的坐标。观察可知，将${p}^{\prime }$投影到${I}_{\mid i}$上会得到一个单调的比特序列，即由若干个（设为${o}_{i}$个）1后面跟着0组成。因此，为了表示${p}_{I}^{\prime }$，只需为$i = 1,\ldots ,d$计算${o}_{i}$即可。然而，后一个任务等价于在排序数组${I}_{\mid i}$中找到小于给定值（即$p$的第$i$个坐标）的元素个数。这可以通过二分查找在$\log C$时间内完成，或者使用预先计算的$C$位的数组在常数时间内完成。因此，计算该函数所需的总时间要么是$O\left( {d\log C}\right)$，要么是$O\left( d\right)$，这取决于所使用的资源。在我们的实验部分，$C$的值可以变得非常小，因此我们将采用第二种方法。

For quick reference we summarize the preprocessing and query answering algorithms in Figures 1 and 2.

为了快速参考，我们在图1和图2中总结了预处理和查询应答算法。

### 3.2 Analysis of Locality-Sensitive Hashing

### 3.2 局部敏感哈希（Locality-Sensitive Hashing）分析

The principle behind our method is that the probability of collision of two points $p$ and $q$ is closely related to the distance between them. Specifically, the larger the distance, the smaller the collision probability. This intuition is formalized as follows [24]. Let $D\left( {\cdot , \cdot  }\right)$ be a distance function of elements from a set $S$ ,and for any $p \in  S$ let $\mathcal{B}\left( {p,r}\right)$ denote the set of elements from $S$ within the distance $r$ from $p$ .

我们方法背后的原理是，两个点$p$和$q$发生碰撞的概率与它们之间的距离密切相关。具体来说，距离越大，碰撞概率越小。这种直觉可以形式化表述如下[24]。设$D\left( {\cdot , \cdot  }\right)$是集合$S$中元素的距离函数，对于任意$p \in  S$，设$\mathcal{B}\left( {p,r}\right)$表示$S$中与$p$的距离在$r$以内的元素集合。

Definition 3 $A$ family $\mathcal{H}$ of functions from $S$ to $U$ is called $\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$ -sensitive for $D\left( {\cdot , \cdot  }\right)$ if for any $q,p \in  S$

定义3 从$S$到$U$的函数族$\mathcal{H}$，如果对于任意$q,p \in  S$，则称其对于$D\left( {\cdot , \cdot  }\right)$是$\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$敏感的

- if $p \in  \mathcal{B}\left( {q,{r}_{1}}\right)$ then $\mathop{\Pr }\limits_{\mathcal{H}}\left\lbrack  {h\left( q\right)  = h\left( p\right) }\right\rbrack   \geq  {p}_{1}$ ,

- 如果$p \in  \mathcal{B}\left( {q,{r}_{1}}\right)$，那么$\mathop{\Pr }\limits_{\mathcal{H}}\left\lbrack  {h\left( q\right)  = h\left( p\right) }\right\rbrack   \geq  {p}_{1}$，

- if $p \notin  \mathcal{B}\left( {q,{r}_{2}}\right)$ then $\mathop{\Pr }\limits_{\mathcal{H}}\left\lbrack  {h\left( q\right)  = h\left( p\right) }\right\rbrack   \leq  {p}_{2}$ .

- 如果$p \notin  \mathcal{B}\left( {q,{r}_{2}}\right)$，那么$\mathop{\Pr }\limits_{\mathcal{H}}\left\lbrack  {h\left( q\right)  = h\left( p\right) }\right\rbrack   \leq  {p}_{2}$。

In the above definition, probabilities are considered with respect to the random choice of a function $h$ from the family $\mathcal{H}$ . In order for a locality-sensitive family to be useful,it has to satisfy the inequalities ${p}_{1} > {p}_{2}$ and ${r}_{1} < {r}_{2}$ .

在上述定义中，概率是相对于从函数族$\mathcal{H}$中随机选择一个函数$h$而言的。为了使局部敏感函数族有用，它必须满足不等式${p}_{1} > {p}_{2}$和${r}_{1} < {r}_{2}$。

Observe that if $D\left( {\cdot , \cdot  }\right)$ is the Hamming distance ${d}_{H}\left( {\cdot , \cdot  }\right)$ ,then the family of projections on one coordinate is locality-sensitive. More specifically:

观察可知，如果 $D\left( {\cdot , \cdot  }\right)$ 是汉明距离 ${d}_{H}\left( {\cdot , \cdot  }\right)$ ，那么在一个坐标上的投影族是局部敏感的。更具体地说：

Fact 2 Let $S$ be ${H}^{{d}^{\prime }}$ (the ${d}^{\prime }$ -dimensional Hamming cube) and $D\left( {p,q}\right)  = {d}_{H}\left( {p,q}\right)$ for $p,q \in$ ${H}^{{d}^{\prime }}$ . Then for any $r,\epsilon  > 0$ ,the family ${\mathcal{H}}_{{d}^{\prime }} =$ $\left\{  {{h}_{i} : {h}_{i}\left( \left( {{b}_{1},\ldots ,{b}_{{d}^{\prime }}}\right) \right)  = {b}_{i},\text{ for }i = 1,\ldots ,{d}^{\prime }}\right\}$ is $\left( {r,r\left( {1 + \epsilon }\right) ,1 - \frac{r}{{d}^{\prime }},1 - \frac{r\left( {1 + \epsilon }\right) }{{d}^{\prime }}}\right)$ -sensitive.

事实 2 设 $S$ 为 ${H}^{{d}^{\prime }}$ （ ${d}^{\prime }$ 维汉明立方体），且对于 $p,q \in$ ${H}^{{d}^{\prime }}$ 有 $D\left( {p,q}\right)  = {d}_{H}\left( {p,q}\right)$ 。那么对于任意 $r,\epsilon  > 0$ ，族 ${\mathcal{H}}_{{d}^{\prime }} =$ $\left\{  {{h}_{i} : {h}_{i}\left( \left( {{b}_{1},\ldots ,{b}_{{d}^{\prime }}}\right) \right)  = {b}_{i},\text{ for }i = 1,\ldots ,{d}^{\prime }}\right\}$ 是 $\left( {r,r\left( {1 + \epsilon }\right) ,1 - \frac{r}{{d}^{\prime }},1 - \frac{r\left( {1 + \epsilon }\right) }{{d}^{\prime }}}\right)$ 敏感的。

We now generalize the algorithm from the previous section to an arbitrary locality-sensitive family $\mathcal{H}$ . Thus,the algorithm is equally applicable to other locality-sensitive hash functions (e.g., see [5]). The generalization is simple: the functions $g$ are now defined to be of the form

现在，我们将上一节的算法推广到任意局部敏感族 $\mathcal{H}$ 。因此，该算法同样适用于其他局部敏感哈希函数（例如，参见 [5]）。推广很简单：函数 $g$ 现在定义为如下形式

$$
{g}_{i}\left( p\right)  = \left( {{h}_{{i}_{1}}\left( p\right) ,{h}_{{i}_{2}}\left( p\right) ,\ldots ,{h}_{{i}_{k}}\left( p\right) }\right) ,
$$

where the functions ${h}_{{i}_{1}},\ldots ,{h}_{{i}_{k}}$ are randomly chosen from $\mathcal{H}$ with replacement. As before,we choose $l$ such functions ${g}_{1},\ldots ,{g}_{l}$ . In the case when the family ${\mathcal{H}}_{{d}^{\prime }}$ is used, i.e., each function selects one bit of an argument, the resulting values of ${g}_{j}\left( p\right)$ are essentially equivalent to ${p}_{\mid {I}_{j}}$ .

其中函数 ${h}_{{i}_{1}},\ldots ,{h}_{{i}_{k}}$ 是从 $\mathcal{H}$ 中有放回地随机选取的。和之前一样，我们选取 $l$ 个这样的函数 ${g}_{1},\ldots ,{g}_{l}$ 。当使用族 ${\mathcal{H}}_{{d}^{\prime }}$ 时，即每个函数选择一个参数的一位， ${g}_{j}\left( p\right)$ 的结果值本质上等同于 ${p}_{\mid {I}_{j}}$ 。

We now show that the LSH algorithm can be used to solve what we call the $\left( {r,\epsilon }\right)$ -Neighbor problem: determine whether there exists a point $p$ within a fixed distance ${r}_{1} = r$ of $q$ ,or whether all points in the database are at least a distance ${r}_{2} = r\left( {1 + \epsilon }\right)$ away from $q$ ; in the first case, the algorithm is required to return a point ${p}^{\prime }$ within distance at most $\left( {1 + \epsilon }\right) r$ from $q$ . In particular, we argue that the LSH algorithm solves this problem for a proper choice of $k$ and $l$ ,depending on $r$ and $\epsilon$ . Then we show how to apply the solution to this problem to solve $\epsilon$ -NNS.

现在我们证明，局部敏感哈希（LSH）算法可用于解决我们所谓的 $\left( {r,\epsilon }\right)$ -近邻问题：确定是否存在一个点 $p$ 与 $q$ 的距离在固定距离 ${r}_{1} = r$ 之内，或者数据库中的所有点与 $q$ 的距离是否至少为 ${r}_{2} = r\left( {1 + \epsilon }\right)$ ；在第一种情况下，要求算法返回一个与 $q$ 的距离至多为 $\left( {1 + \epsilon }\right) r$ 的点 ${p}^{\prime }$ 。特别地，我们认为，根据 $r$ 和 $\epsilon$ ，通过适当选择 $k$ 和 $l$ ，LSH 算法可以解决这个问题。然后我们展示如何将这个问题的解决方案应用于解决 $\epsilon$ -最近邻搜索（NNS）问题。

Denote by ${P}^{\prime }$ the set of all points ${p}^{\prime } \in  P$ such that $d\left( {q,{p}^{\prime }}\right)  > {r}_{2}$ . We observe that the algorithm correctly solves the $\left( {r,\epsilon }\right)$ -Neighbor problem if the following two properties hold:

用 ${P}^{\prime }$ 表示所有满足 $d\left( {q,{p}^{\prime }}\right)  > {r}_{2}$ 的点 ${p}^{\prime } \in  P$ 的集合。我们观察到，如果满足以下两个性质，该算法就能正确解决 $\left( {r,\epsilon }\right)$ -近邻问题：

P1 If there exists ${p}^{ * }$ such that ${p}^{ * } \in  \mathcal{B}\left( {q,{r}_{1}}\right)$ ,then ${g}_{j}\left( {p}^{ * }\right)  = {g}_{j}\left( q\right)$ for some $j = 1,\ldots ,l$ .

P1 如果存在 ${p}^{ * }$ 使得 ${p}^{ * } \in  \mathcal{B}\left( {q,{r}_{1}}\right)$ ，那么对于某个 $j = 1,\ldots ,l$ 有 ${g}_{j}\left( {p}^{ * }\right)  = {g}_{j}\left( q\right)$ 。

P2 The total number of blocks pointed to by $q$ and containing only points from ${P}^{\prime }$ is less than ${cl}$ .

P2 由 $q$ 指向且仅包含来自 ${P}^{\prime }$ 的点的块的总数小于 ${cl}$ 。

Assume that $\mathcal{H}$ is a $\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$ -sensitive family; define $\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}}$ . The correctness of the LSH algorithm follows from the following theorem.

假设 $\mathcal{H}$ 是一个 $\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$ -敏感族；定义 $\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}}$。局部敏感哈希（LSH）算法的正确性由以下定理得出。

Theorem 1 Setting $k = {\log }_{1/{p}_{2}}\left( {n/B}\right)$ and $l = {\left( \frac{n}{B}\right) }^{\rho }$ guarantees that properties $\mathbf{{P1}}$ and $\mathbf{{P2}}$ hold with probability at least $\frac{1}{2} - \frac{1}{e} \geq  {0.132}$ .

定理 1 设置 $k = {\log }_{1/{p}_{2}}\left( {n/B}\right)$ 和 $l = {\left( \frac{n}{B}\right) }^{\rho }$ 可确保属性 $\mathbf{{P1}}$ 和 $\mathbf{{P2}}$ 至少以 $\frac{1}{2} - \frac{1}{e} \geq  {0.132}$ 的概率成立。

Remark 1 Note that by repeating the LSH algorithm $O\left( {1/\delta }\right)$ times,we can amplify the probability of success in at least one trial to $1 - \delta$ ,for any $\delta  > 0$ .

注记 1 注意，通过将局部敏感哈希（LSH）算法重复 $O\left( {1/\delta }\right)$ 次，对于任意 $\delta  > 0$，我们可以将至少一次试验成功的概率提高到 $1 - \delta$。

Proof: Let property $\mathbf{{P1}}$ hold with probability ${P}_{1}$ , and property $\mathbf{{P2}}$ hold with probability ${P}_{2}$ . We will show that both ${P}_{1}$ and ${P}_{2}$ are large. Assume that there exists a point ${p}^{ * }$ within distance ${r}_{1}$ of $q$ ; the proof is quite similar otherwise. Set $k = {\log }_{1/{p}_{2}}\left( {n/B}\right)$ . The probability that $g\left( {p}^{\prime }\right)  = g\left( q\right)$ for $p \in  P - \mathcal{B}\left( {q,{r}_{2}}\right)$ is at most ${p}_{2}^{k} = \frac{B}{n}$ . Denote the set of all points ${p}^{\prime } \notin  \mathcal{B}\left( {q,{r}_{2}}\right)$ by ${P}^{\prime }$ . The expected number of blocks allocated for ${g}_{j}$ which contain exclusively points from ${P}^{\prime }$ does not exceed 2. The expected number of such blocks allocated for all ${g}_{j}$ is at most ${2l}$ . Thus,by the Markov inequality [35],the probability that this number exceeds ${4l}$ is less than $1/2$ . If we choose $c = 4$ ,the probability that the property $\mathrm{P}2$ holds is ${P}_{2} > 1/2$ .

证明：设属性 $\mathbf{{P1}}$ 以 ${P}_{1}$ 的概率成立，属性 $\mathbf{{P2}}$ 以 ${P}_{2}$ 的概率成立。我们将证明 ${P}_{1}$ 和 ${P}_{2}$ 都很大。假设存在一个点 ${p}^{ * }$ 与 $q$ 的距离在 ${r}_{1}$ 以内；否则证明过程非常相似。设 $k = {\log }_{1/{p}_{2}}\left( {n/B}\right)$。对于 $p \in  P - \mathcal{B}\left( {q,{r}_{2}}\right)$，$g\left( {p}^{\prime }\right)  = g\left( q\right)$ 成立的概率至多为 ${p}_{2}^{k} = \frac{B}{n}$。用 ${P}^{\prime }$ 表示所有点 ${p}^{\prime } \notin  \mathcal{B}\left( {q,{r}_{2}}\right)$ 的集合。为 ${g}_{j}$ 分配的仅包含来自 ${P}^{\prime }$ 的点的块的期望数量不超过 2。为所有 ${g}_{j}$ 分配的此类块的期望数量至多为 ${2l}$。因此，根据马尔可夫不等式 [35]，这个数量超过 ${4l}$ 的概率小于 $1/2$。如果我们选择 $c = 4$，属性 $\mathrm{P}2$ 成立的概率为 ${P}_{2} > 1/2$。

Consider now the probability of ${g}_{j}\left( {p}^{ * }\right)  = {g}_{j}\left( q\right)$ . Clearly, it is bounded from below by

现在考虑 ${g}_{j}\left( {p}^{ * }\right)  = {g}_{j}\left( q\right)$ 的概率。显然，它有如下下界

$$
{p}_{1}^{k} = {p}_{1}^{{\log }_{1/{p}_{2}}n/B} = {\left( n/B\right) }^{-\frac{\log 1/{p}_{1}}{\log 1/{p}_{2}}} = {\left( n/B\right) }^{-\rho }.
$$

By setting $l = {\left( \frac{n}{B}\right) }^{\rho }$ ,we bound from above the probability that ${g}_{j}\left( {p}^{ * }\right)  \neq  {g}_{j}\left( q\right)$ for all $j = 1,\ldots ,l$ by $1/e$ . Thus the probability that one such ${g}_{j}$ exists is at least ${P}_{1} \geq  1 - 1/e$ .

通过设置 $l = {\left( \frac{n}{B}\right) }^{\rho }$，我们将对于所有 $j = 1,\ldots ,l$ 都有 ${g}_{j}\left( {p}^{ * }\right)  \neq  {g}_{j}\left( q\right)$ 成立的概率上界限定为 $1/e$。因此，存在一个这样的 ${g}_{j}$ 的概率至少为 ${P}_{1} \geq  1 - 1/e$。

Therefore, the probability that both properties P1 and $\mathrm{P}2$ hold is at least $1 - \left\lbrack  {\left( {1 - {P}_{1}}\right)  + \left( {1 - {P}_{2}}\right) }\right\rbrack   =$ ${P}_{1} + {P}_{2} - 1 \geq  \frac{1}{2} - \frac{1}{e}$ . The theorem follows.

因此，属性 P1 和 $\mathrm{P}2$ 都成立的概率至少为 $1 - \left\lbrack  {\left( {1 - {P}_{1}}\right)  + \left( {1 - {P}_{2}}\right) }\right\rbrack   =$ ${P}_{1} + {P}_{2} - 1 \geq  \frac{1}{2} - \frac{1}{e}$。定理得证。

In the following we consider the LSH family for the Hamming metric of dimension ${d}^{\prime }$ as specified in Fact 2 . For this case,we show that $\rho  \leq  \frac{1}{1 + \epsilon }$ assuming that $r < \frac{{d}^{\prime }}{\ln n}$ ; the latter assumption can be easily satisfied by increasing the dimensionality by padding a sufficiently long string of0s at the end of each point's representation.

接下来，我们考虑如事实2中所规定的、维度为${d}^{\prime }$的汉明度量（Hamming metric）的局部敏感哈希（LSH）族。对于这种情况，我们证明在假设$r < \frac{{d}^{\prime }}{\ln n}$成立的条件下有$\rho  \leq  \frac{1}{1 + \epsilon }$；通过在每个点的表示末尾填充足够长的0字符串来增加维度，后一个假设很容易满足。

Fact 3 Let $r < \frac{{d}^{\prime }}{\ln n}$ . If ${p}_{1} = 1 - \frac{r}{{d}^{\prime }}$ and ${p}_{2} = 1 - \frac{r\left( {1 + \epsilon }\right) }{{d}^{\prime }}$ , then $\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}} \leq  \frac{1}{1 + \epsilon }$ .

事实3 设$r < \frac{{d}^{\prime }}{\ln n}$。若${p}_{1} = 1 - \frac{r}{{d}^{\prime }}$且${p}_{2} = 1 - \frac{r\left( {1 + \epsilon }\right) }{{d}^{\prime }}$，则$\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}} \leq  \frac{1}{1 + \epsilon }$。

Proof: Observe that

证明：注意到

$$
\rho  = \frac{\ln 1/{p}_{1}}{\ln 1/{p}_{2}} = \frac{\ln \frac{1}{1 - r/{d}^{\prime }}}{\ln \frac{1}{1 - \left( {1 + \epsilon }\right) r/{d}^{\prime }}} = \frac{\ln \left( {1 - r/{d}^{\prime }}\right) }{\ln \left( {1 - \left( {1 + \epsilon }\right) r/{d}^{\prime }}\right) }.
$$

Multiplying both the numerator and the denominator by $\frac{{d}^{\prime }}{r}$ ,we obtain:

将分子和分母同时乘以$\frac{{d}^{\prime }}{r}$，我们得到：

$$
\rho  = \frac{\frac{{d}^{\prime }}{r}\ln \left( {1 - r/{d}^{\prime }}\right) }{\frac{{d}^{\prime }}{r}\ln \left( {1 - \left( {1 + \epsilon }\right) r/{d}^{\prime }}\right) }
$$

$$
 = \frac{\ln {\left( 1 - r/{d}^{\prime }\right) }^{{d}^{\prime }/r}}{\ln {\left( 1 - \left( 1 + \epsilon \right) r/{d}^{\prime }\right) }^{{d}^{\prime }/r}} = \frac{U}{L}
$$

In order to upper bound $\rho$ ,we need to bound $U$ from below and $L$ from above; note that both $U$ and $L$ are negative. To this end we use the following inequalities [35]:

为了对$\rho$进行上界估计，我们需要对$U$进行下界估计，对$L$进行上界估计；注意$U$和$L$均为负数。为此，我们使用以下不等式[35]：

$$
{\left( 1 - \left( 1 + \epsilon \right) r/{d}^{\prime }\right) }^{{d}^{\prime }/r} < {e}^{-\left( {1 + \epsilon }\right) }
$$

and

以及

$$
{\left( 1 - \frac{r}{{d}^{\prime }}\right) }^{{d}^{\prime }/r} > {e}^{-1}\left( {1 - \frac{1}{{d}^{\prime }/r}}\right) .
$$

Therefore,

因此，

$$
\frac{U}{L} < \frac{\ln \left( {{e}^{-1}\left( {1 - \frac{1}{{d}^{\prime }/r}}\right) }\right) }{\ln {e}^{-\left( {1 + \epsilon }\right) }}
$$

$$
 = \frac{-1 + \ln \left( {1 - \frac{1}{{d}^{\prime }/r}}\right) }{-\left( {1 + \epsilon }\right) }
$$

$$
 = 1/\left( {1 + \epsilon }\right)  - \frac{\ln \left( {1 - \frac{1}{{d}^{\prime }/r}}\right) }{1 + \epsilon }
$$

$$
 < 1/\left( {1 + \epsilon }\right)  - \ln \left( {1 - 1/\ln n}\right) 
$$

where the last step uses the assumptions that $\epsilon  > 0$ and $r < \frac{{d}^{\prime }}{\ln n}$ . We conclude that

最后一步使用了$\epsilon  > 0$和$r < \frac{{d}^{\prime }}{\ln n}$的假设。我们得出结论

$$
{n}^{\rho } < {n}^{1/\left( {1 + \epsilon }\right) }{n}^{-\ln \left( {1 - 1/\ln n}\right) }
$$

$$
 = {n}^{1/\left( {1 + \epsilon }\right) }{\left( 1 - 1/\ln n\right) }^{-\ln n} = O\left( {n}^{1/\left( {1 + \epsilon }\right) }\right) 
$$

We now return to the $\epsilon$ -NNS problem. First,we observe that we could reduce it to the $\left( {r,\epsilon }\right)$ -Neighbor problem by building several data structures for the latter problem with different values of $r$ . More specifically,we could explore $r$ equal to ${r}_{0},{r}_{0}\left( {1 + \epsilon }\right)$ , ${r}_{0}{\left( 1 + \epsilon \right) }^{2},\ldots ,{r}_{\max }$ ,where ${r}_{0}$ and ${r}_{\max }$ are the smallest and the largest possible distance between the query and the data point, respectively. We remark that the number of different radii could be further reduced [24] at the cost of increasing running time and space requirement. On the other hand, we observed that in practice choosing only one value of $r$ is sufficient to produce answers of good quality. This can be explained as in [10] where it was observed that the distribution of distances between a query point and the data set in most cases does not depend on the specific query point, but on the intrinsic properties of the data set. Under the assumption of distribution invariance, the same parameter $r$ is likely to work for a vast majority of queries. Therefore in the experimental section we adopt a fixed choice of $r$ and therefore also of $k$ and $l$ .

现在我们回到$\epsilon$ - 近邻搜索（NNS）问题。首先，我们注意到可以通过为$\left( {r,\epsilon }\right)$ - 近邻问题构建几个具有不同$r$值的数据结构，将其简化为该问题。更具体地说，我们可以探索$r$等于${r}_{0},{r}_{0}\left( {1 + \epsilon }\right)$、${r}_{0}{\left( 1 + \epsilon \right) }^{2},\ldots ,{r}_{\max }$的情况，其中${r}_{0}$和${r}_{\max }$分别是查询点与数据点之间可能的最小和最大距离。我们注意到，以增加运行时间和空间需求为代价，可以进一步减少不同半径的数量[24]。另一方面，我们在实践中发现，仅选择一个$r$值就足以产生高质量的答案。这可以如文献[10]中所解释的那样，在大多数情况下，查询点与数据集之间的距离分布并不取决于特定的查询点，而是取决于数据集的内在属性。在分布不变性的假设下，相同的参数$r$很可能适用于绝大多数查询。因此，在实验部分，我们采用固定的$r$选择，从而也采用固定的$k$和$l$选择。

## 4 Experiments

## 4 实验

In this section we report the results of our experiments with locality-sensitive hashing method. We performed experiments on two data sets. The first one contains up to 20,000 histograms of color images from COREL Draw library, where each histogram was represented as a point in $d$ -dimensional space,for $d$ up to 64 . The second one contains around 270,000 points of dimension 60 representing texture information of blocks of large large aerial photographs. We describe the data sets in more detail later in the section.

在本节中，我们报告了使用局部敏感哈希方法进行实验的结果。我们在两个数据集上进行了实验。第一个数据集包含来自COREL Draw库的多达20,000个彩色图像直方图，每个直方图在$d$维空间中表示为一个点，其中$d$最大为64。第二个数据集包含约270,000个60维的点，这些点代表大型航空照片块的纹理信息。我们将在本节后面更详细地描述这些数据集。

We decided not to use randomly-chosen synthetic data in our experiments. Though such data is often used to measure the performance of exact similarity search algorithms, we found it unsuitable for evaluation of approximate algorithms for the high data dimensionality. The main reason is as follows. Assume a data set consisting of points chosen independently at random from the same distribution. Most distributions (notably uniform) used in the literature assume that all coordinates of each point are chosen independently. In such a case,for any pair of points $p,q$ the distance $d\left( {p,q}\right)$ is sharply concentrated around the mean; for example, for the uniform distribution over the unit cube,the expected distance is $O\left( d\right)$ ,while the standard deviation is only $O\left( \sqrt{d}\right)$ . Thus almost all pairs are approximately within the same distance, so the notion of approximate nearest neighbor is not meaningful - almost every point is an approximate nearest neighbor.

我们决定在实验中不使用随机选择的合成数据。尽管此类数据常用于衡量精确相似性搜索算法的性能，但我们发现它不适用于评估高维数据的近似算法。主要原因如下。假设一个数据集由从同一分布中独立随机选择的点组成。文献中使用的大多数分布（特别是均匀分布）假设每个点的所有坐标都是独立选择的。在这种情况下，对于任意一对点$p,q$，距离$d\left( {p,q}\right)$会紧密集中在均值附近；例如，对于单位立方体上的均匀分布，期望距离为$O\left( d\right)$，而标准差仅为$O\left( \sqrt{d}\right)$。因此，几乎所有点对的距离都大致相同，所以近似近邻的概念就没有意义了——几乎每个点都是近似近邻。

Implementation. We implement the LSH algorithm as specified in Section 3. The LSH functions can be computed as described in Section 3.1. Denote the resulting vector of coordinates by $\left( {{v}_{1},\ldots ,{v}_{k}}\right)$ . For the second level mapping we use functions of the form

实现。我们按照第3节的规定实现局部敏感哈希（LSH）算法。可以按照第3.1节的描述计算LSH函数。将得到的坐标向量记为$\left( {{v}_{1},\ldots ,{v}_{k}}\right)$。对于第二层映射，我们使用以下形式的函数

$$
h\left( {{v}_{1},\ldots ,{v}_{k}}\right)  = {a}_{1} \cdot  {v}_{1} + \cdots  + {a}_{d} \cdot  {v}_{k}{\;\operatorname{mod}\;M},
$$

where $M$ is the size of the hash table and ${a}_{1},\ldots ,{a}_{k}$ are random numbers from interval $\left\lbrack  {0\ldots M - 1}\right\rbrack$ . These functions can be computed using only ${2k} - 1$ operations, and are sufficiently random for our purposes, i.e., give low probability of collision. Each second level bucket is then directly mapped to a disk block. We assumed that each block is $8\mathrm{{KB}}$ of data. As each coordinate in our data sets can be represented using 1 byte, we can store up to ${8192}/{dd}$ -dimensional points per block. Therefore,we assume the bucket size $B = {100}$ for $d = {64}$ or $d = {60},B = {300}$ for $d = {27}$ and $B = {1000}$ for $d = 8$ .

其中$M$是哈希表的大小，${a}_{1},\ldots ,{a}_{k}$是区间$\left\lbrack  {0\ldots M - 1}\right\rbrack$内的随机数。这些函数仅需${2k} - 1$次运算即可计算，并且对于我们的目的来说具有足够的随机性，即碰撞概率较低。然后，每个第二层桶直接映射到一个磁盘块。我们假设每个块包含$8\mathrm{{KB}}$的数据。由于我们数据集中的每个坐标可以用1字节表示，因此每个块最多可以存储${8192}/{dd}$维的点。因此，我们假设对于$d = {64}$桶大小为$B = {100}$，对于$d = {27}$为$d = {60},B = {300}$，对于$d = 8$为$B = {1000}$。

For the SR-tree, we used the implementation by Katayama, available from his web page [28]. As above, we allow it to store about 8192 coordinates per disk block.

对于SR树，我们使用了片山（Katayama）的实现，可从他的网页[28]获取。如上所述，我们允许它每个磁盘块存储约8192个坐标。

Performance measures. The goal of our experiments was to estimate two performance measures: speed (for both SR-tree and LSH) and accuracy (for LSH). The speed is measured by the number of disk blocks accessed in order to answer a query. We count all disk accesses, thus ignoring the issue of caching. Observe that in case of LSH this number is easy to predict as it is clearly equal to the number of indices used. As the number of indices also determines the storage overhead, it is a natural parameter to optimize.

性能指标。我们实验的目标是评估两个性能指标：速度（针对SR树和局部敏感哈希（LSH））和准确性（针对LSH）。速度通过为回答查询而访问的磁盘块数量来衡量。我们统计所有磁盘访问，因此忽略缓存问题。请注意，对于LSH，这个数量很容易预测，因为它显然等于使用的索引数量。由于索引数量也决定了存储开销，因此它是一个自然的优化参数。

The error of LSH is measured as follows. Following [2] we define (for the Approximate 1-NNS problem) the effective error as

LSH的误差测量如下。遵循文献[2]，我们（针对近似1-最近邻搜索（1-NNS）问题）将有效误差定义为

$$
E = \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{\text{query }q \in  Q}}\frac{{d}_{LSH}}{{d}^{ * }},
$$

where ${d}_{LSH}$ denotes the distance from a query point $q$ to a point found by LSH, ${d}^{ * }$ is the distance from $q$ to the closest point, and the sum is taken of all queries for which a nonempty index was found. We also measure the (small) fraction of queries for which no nonempty bucket was found; we call this quantity miss ratio. For the Approximate $K$ -NNS we measure separately the distance ratios between the closest points found to the nearest neighbor, the 2nd closest one to the 2nd nearest neighbor and so on; then we average the ratios. The miss ratio is defined to be the fraction of cases when less than $K$ points were found.

其中${d}_{LSH}$表示从查询点$q$到LSH找到的点的距离，${d}^{ * }$是从$q$到最近点的距离，并且该求和是对所有找到非空索引的查询进行的。我们还测量未找到非空桶的查询的（小）比例；我们将此数量称为未命中比率。对于近似$K$ -最近邻搜索（NNS），我们分别测量找到的最近点与最近邻、第二近点与第二近邻等之间的距离比率；然后对这些比率求平均值。未命中比率定义为找到少于$K$个点的情况的比例。

Data Sets. Our first data set consists of 20,000 histograms of color thumbnail-sized images of various contents taken from the COREL library. The histograms were extracted after transforming the pixels of the images to the 3-dimensional CIE-Lab color space [44]; the property of this space is that the distance between each pair of points corresponds to the perceptual dissimilarity between the colors that the two points represent. Then we partitioned the color space into a grid of smaller cubes, and given an image, we create the color histogram of the image by counting how many pixels fall into each of these cubes. By dividing each axis into $u$ intervals we obtain a total of ${u}^{3}$ cubes. For most experiments,we assumed $u = 4$ obtaining a 64-dimensional space. Each histogram cube (i.e., color) then corresponds to a dimension of space representing the images. Finally, quantization is performed in order to fit each coordinate in 1 byte. For each point representing an image each coordinate effectively counts the number of the image's pixels of a specific color. All coordinates are clearly non-negative integers, as assumed in Section 3. The distribution of interpoint distances in our point sets is shown in Figure 3. Both graphs were obtained by computing all interpoint distances of random subsets of 200 points, normalizing the maximal value to 1 .

数据集。我们的第一个数据集由20,000个彩色缩略图大小的图像直方图组成，这些图像内容各异，取自COREL库。在将图像的像素转换到三维CIE-Lab颜色空间[44]后提取直方图；该空间的特性是每对点之间的距离对应于这两个点所代表的颜色之间的感知差异。然后我们将颜色空间划分为更小立方体的网格，给定一幅图像，我们通过计算有多少像素落入每个立方体来创建图像的颜色直方图。通过将每个轴划分为$u$个区间，我们总共得到${u}^{3}$个立方体。对于大多数实验，我们假设$u = 4$，得到一个64维的空间。然后每个直方图立方体（即颜色）对应于表示图像的空间的一个维度。最后，进行量化以将每个坐标拟合到1字节中。对于表示图像的每个点，每个坐标实际上统计了图像中特定颜色的像素数量。如第3节所假设的，所有坐标显然都是非负整数。我们点集中点间距离的分布如图3所示。两个图都是通过计算200个点的随机子集的所有点间距离，并将最大值归一化为1得到的。

<!-- Media -->

<!-- figureText: 1 Point set distance distribution 4000 5000 6000 7000 8000 Interpoint distance (a) Color histograms Texture data set point distribution 400 500 600 700 Interpoint distance (b) Texture features Normalized frequency 0.8 0.6 0.4 0.2 1000 2000 3000 Normalized frequency 0.6 0.2 100 200 300 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_6.jpg?x=946&y=178&w=654&h=1162&r=0"/>

Figure 3: The profiles of the data sets.

图3：数据集的分布情况。

<!-- Media -->

The second data set contains 275,465 feature vectors of dimension 60 representing texture information of blocks of large aerial photographs. This data set was provided by B.S. Manjunath [31, 32]; its size and dimensionality "provides challenging problems in high dimensional indexing" [31]. These features are obtained from Gabor filtering of the image tiles. The Gabor filter bank consists of 5 scales and 6 orientations of filters, thus the total number of filters is $5 \times  6 = {30}$ . The mean and standard deviation of each filtered output are used to constructed the feature vector $\left( {d = {30} \times  2 = {60}}\right)$ . These texture features are extracted from 40 large air photos. Before the feature extraction, each airphoto is first partitioned into nonoverlapping tiles of size 64 times 64, from which the feature vectors are computed.

第二个数据集包含275,465个60维的特征向量，代表大型航空照片块的纹理信息。这个数据集由B.S.曼朱纳特（Manjunath）提供[31, 32]；其大小和维度“在高维索引方面带来了具有挑战性的问题”[31]。这些特征是通过对图像块进行Gabor滤波得到的。Gabor滤波器组由5个尺度和6个方向的滤波器组成，因此滤波器的总数为$5 \times  6 = {30}$。每个滤波输出的均值和标准差用于构建特征向量$\left( {d = {30} \times  2 = {60}}\right)$。这些纹理特征是从40张大航空照片中提取的。在特征提取之前，每张航空照片首先被划分为大小为64×64的不重叠块，从中计算特征向量。

Query Sets. The difficulty in evaluating similarity searching algorithms is the lack of a publicly available database containing typical query points. Therefore, we had to construct the query set from the data set itself. Our construction is as follows: we split the data set randomly into two disjoint parts (call them ${S}_{1}$ and $\left. {S}_{2}\right)$ . For the first data set the size of ${S}_{1}$ is 19,000 while the size of ${S}_{2}$ is 1000 . The set ${S}_{1}$ forms a database of images,while the first 500 points from ${S}_{2}$ (denoted by $Q$ ) are used as query points (we use the other 500 points for various verification purposes). For the second data set we chose ${S}_{1}$ to be of size270,000,and we use 1000 of the remaining 5,465 points as a query set. The numbers are slightly different for the scala-bility experiments as they require varying the size of the data set. In this case we chose a random subset of ${S}_{1}$ of required size.

查询集。评估相似性搜索算法的难点在于缺乏一个包含典型查询点的公开可用数据库。因此，我们不得不从数据集本身构建查询集。我们的构建方式如下：我们将数据集随机划分为两个不相交的部分（分别称为${S}_{1}$和$\left. {S}_{2}\right)$）。对于第一个数据集，${S}_{1}$的大小为19000，而${S}_{2}$的大小为1000。集合${S}_{1}$构成一个图像数据库，而从${S}_{2}$中选取的前500个点（用$Q$表示）用作查询点（我们使用另外500个点用于各种验证目的）。对于第二个数据集，我们选择${S}_{1}$的大小为270000，并使用剩余的5465个点中的1000个作为查询集。在可扩展性实验中，这些数字略有不同，因为它们需要改变数据集的大小。在这种情况下，我们选择了所需大小的${S}_{1}$的一个随机子集。

### 4.1 Experimental Results

### 4.1 实验结果

In this section we describe the results of our experiments. For both data sets they consist essentially of the following three steps. In the first phase we have to make the following choice: the value of $k$ (the number of sampled bits) to choose for a given data set and the given number of indices $l$ in order to minimize the effective error. It turned out that the optimal value of $k$ is essentially independent of $n$ and $d$ and thus we can use the same value for different values of these parameters. In the second phase, we estimate the influence of the number of indices $l$ on the error. Finally,we measure the performance of LSH by computing (for a variety of data sets) the minimal number of indices needed to achieve a specified value of error. When applicable, we also compare this performance with that of SR-trees.

在本节中，我们描述了实验结果。对于这两个数据集，实验基本上包括以下三个步骤。在第一阶段，我们必须做出以下选择：为给定的数据集和给定的索引数量$l$选择$k$的值（采样比特数），以最小化有效误差。结果表明，$k$的最优值基本上与$n$和$d$无关，因此我们可以为这些参数的不同值使用相同的值。在第二阶段，我们估计索引数量$l$对误差的影响。最后，我们通过计算（针对各种数据集）达到指定误差值所需的最小索引数量来衡量LSH（局部敏感哈希）的性能。在适用的情况下，我们还将这种性能与SR树的性能进行比较。

### 4.2 Color histograms

### 4.2 颜色直方图

For this data set, we performed several experiments aimed at understanding the behavior of LSH algorithm and its performance relative to SR-tree. As mentioned above, we started with an observation that the optimal value of sampled bits $k$ is essentially independent of $n$ and $d$ and approximately equal to 700 for $d = {64}$ . The lack of dependence on $n$ can be explained by the fact that the smaller data sets were obtained by sampling the large one and therefore all of the sets have similar structure; we believe the lack of dependence on $d$ is also influenced by the structure of the data. Therefore the following experiments were done assuming $k = {700}$ .

对于这个数据集，我们进行了几个实验，旨在了解LSH算法的行为及其相对于SR树的性能。如上所述，我们首先观察到采样比特$k$的最优值基本上与$n$和$d$无关，并且对于$d = {64}$大约等于700。与$n$无关可以解释为，较小的数据集是通过对大的数据集进行采样得到的，因此所有数据集都具有相似的结构；我们认为与$d$无关也受到数据结构的影响。因此，以下实验是在假设$k = {700}$的情况下进行的。

Our next observation was that the value of storage overhead $\alpha$ does not exert much influence over the performance of the algorithm (we tried $\alpha$ ’s from the interval $\left\lbrack  {2,5}\right\rbrack$ ); thus,in the following we set $\alpha  = 2$ .

我们的下一个观察结果是，存储开销$\alpha$的值对算法的性能影响不大（我们尝试了区间$\left\lbrack  {2,5}\right\rbrack$内的$\alpha$值）；因此，在接下来的实验中，我们设置$\alpha  = 2$。

In the next step we estimated the influence of $l$ on $E$ . The results (for $n = {19},{000},d = {64},K = 1$ ) are shown on Figure 4. As expected, one index is not sufficient to achieve reasonably small error - the effective error can easily exceed ${50}\%$ . The error however decreases very fast as $l$ increases. This is due to the fact that the probabilities of finding empty bucket are independent for different indices and therefore the probability that all buckets are empty decays exponentially in $l$ .

在下一步中，我们估计了$l$对$E$的影响。（对于$n = {19},{000},d = {64},K = 1$的）结果如图4所示。正如预期的那样，一个索引不足以实现足够小的误差——有效误差很容易超过${50}\%$。然而，随着$l$的增加，误差下降得非常快。这是因为对于不同的索引，找到空桶的概率是相互独立的，因此所有桶都为空的概率随着$l$呈指数衰减。

<!-- Media -->

<!-- figureText: 0.6 alpha=2, n=19000, d=64, k=700 8 9 10 Number of indices 0.5 0.4 0.3 0.2 0.1 0 2 3 4 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_7.jpg?x=922&y=180&w=650&h=550&r=0"/>

Figure 4: Error vs. the number of indices.

图4：误差与索引数量的关系。

<!-- Media -->

In order to compare the performance of LSH with SR-tree, we computed (for a variety of data sets) the minimal number of indices needed to achieve a specified value of error $E$ equal to $2\% ,5\% ,{10}\%$ or ${20}\%$ . Then we investigated the performance of the two algorithms while varying the dimension and data size.

为了比较LSH和SR树的性能，我们计算了（针对各种数据集）达到指定误差值$E$（等于$2\% ,5\% ,{10}\%$或${20}\%$）所需的最小索引数量。然后，我们在改变维度和数据大小的情况下研究了这两种算法的性能。

Dependence on Data Size. We performed the simulations for $d = {64}$ and the data sets of sizes 1000 , 2000,5000,10000and 19000 . To achieve better understanding of scalability of our algorithms, we did run the experiments twice: for Approximate 1-NNS and for Approximate 10-NNS. The results are presented on Figure 5.

对数据大小的依赖性。我们对$d = {64}$以及大小为1000、2000、5000、10000和19000的数据集进行了模拟。为了更好地理解我们算法的可扩展性，我们进行了两次实验：一次是近似1-最近邻搜索（Approximate 1-NNS），另一次是近似10-最近邻搜索（Approximate 10-NNS）。结果如图5所示。

Notice the strongly sublinear dependence exhibited by LSH: although for small $E = 2\%$ it matches SR-tree for $n = {1000}$ with 5 blocks accessed (for $K = 1$ ),it requires 3 accesses more for a data set 19 times larger. At the same time the I/O activity of SR-tree increases by more than ${200}\%$ . For larger errors the LSH curves are nearly flat, i.e., exhibit little dependence on the size of the data. Similar or even better behavior occurs for Approximate 10-NNS.

注意局部敏感哈希（LSH）呈现出的强次线性依赖关系：尽管对于较小的$E = 2\%$，在访问5个块（对于$K = 1$）的情况下，它在$n = {1000}$方面与SR树表现相当，但对于大19倍的数据集，它需要多访问3次。与此同时，SR树的I/O活动增加了超过${200}\%$。对于较大的误差，LSH曲线几乎是平坦的，即对数据大小的依赖性很小。近似10近邻搜索（Approximate 10 - NNS）也有类似甚至更好的表现。

We also computed the miss ratios, i.e., the fraction of queries for which no answer was found. The results are presented on Figure 6. We used the parameters from the previous experiment. On can observe that for say $E = 5\%$ and Approximate 1-NNS,the miss ratios are quite high $\left( {{10}\% }\right)$ for small $n$ ,but decrease to around $1\%$ for $n = {19},{000}$ .

我们还计算了未命中比率，即未找到答案的查询所占的比例。结果如图6所示。我们使用了上一个实验中的参数。可以观察到，例如对于$E = 5\%$和近似1近邻搜索（Approximate 1 - NNS），在较小的$n$时，未命中比率相当高（$\left( {{10}\% }\right)$），但在$n = {19},{000}$时降至约$1\%$。

<!-- Media -->

<!-- figureText: 20 LSH, error=.02 alpha $= 2,1 - \mathrm{{NNS}}$ Number of database points $\times  {10}^{4}$ (a) Approximate 1-NNS alpha $= 2,{10} - \mathrm{{NNS}}$ 1.9 Number of database points $\times  {10}^{4}$ (b) Approximate 10-NNS LSH, error=.05 LSH, error=.1 LSH, error=.2 Disk accesses 5 0.10.2 0.5 40 SR-Tree 35 LSH, error=.02 LSH, error=.05 30 LSH, error=.1 LSH, error=.2 Disk accesses 20 15 10 5 0 0.10.2 0.5 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_8.jpg?x=204&y=179&w=657&h=1205&r=0"/>

Figure 5: Number of indices vs. data size.

图5：索引数量与数据大小的关系。

<!-- Media -->

Dependence on Dimension. We performed the simulations for $d = {2}^{3},{3}^{3}$ and ${4}^{3}$ ; the choice of $d$ ’s was limited to cubes of natural numbers because of the way the data has been created. Again, we performed the comparison for Approximate 1-NNS and Approximate 10-NNS; the results are shown on Figure 7. Note that LSH scales very well with the increase of dimensionality: for $E = 5\%$ the change from $d = 8$ to $d = {64}$ increases the number of indices only by 2 . The miss ratio was always below $6\%$ for all dimensions.

对维度的依赖性。我们对$d = {2}^{3},{3}^{3}$和${4}^{3}$进行了模拟；由于数据的创建方式，$d$的选择仅限于自然数的立方。同样，我们对近似1近邻搜索（Approximate 1 - NNS）和近似10近邻搜索（Approximate 10 - NNS）进行了比较；结果如图7所示。注意，LSH随着维度的增加扩展性非常好：对于$E = 5\%$，从$d = 8$到$d = {64}$的变化仅使索引数量增加了2。对于所有维度，未命中比率始终低于$6\%$。

This completes the comparison of LSH with SR-tree. For a better understanding of the behavior of LSH, we performed an additional experiment on LSH only. Figure 8 presents the performance of LSH when the number of nearest neighbors to retrieve vary from 1 to 100 .

至此完成了局部敏感哈希（LSH）与SR树的比较。为了更好地理解LSH的性能，我们仅对LSH进行了一项额外的实验。图8展示了当要检索的最近邻数量从1到100变化时LSH的性能。

<!-- Media -->

<!-- figureText: 0.25 alpha=2, n=19000, d=64, 1-NNS Error=.05 Error=.1 1 1.9 Number of database points $\times  {10}^{4}$ (a) Approximate 1-NNS alpha=2, n=19000, d=64, 10-NNS Error=.05 Error=.1 1.9 Number of database points $\times  {10}^{4}$ (b) Approximate 10-NNS 0.2 Miss ratio 0.1 0.05 0.10.2 0.5 0.5 0.4 Miss ratio 0.2 0.1 0.10.2 0.5 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_8.jpg?x=944&y=180&w=657&h=1187&r=0"/>

Figure 6: Miss ratio vs. data size.

图6：未命中比率与数据大小的关系。

<!-- Media -->

### 4.3 Texture features

### 4.3 纹理特征

The experiments with texture feature data were designed to measure the performance of the LSH algorithm on large data sets; note that the size of the texture file(270,000points)is an order of magnitude larger than the size of the histogram data set $({20},{000}$ points). The first step (i.e., the choice of the number of sampled bits $k$ ) was very similar to the previous experiment, therefore we skip the detailed description here. We just state that we assumed that the number of sampled bits $k = {65}$ ,with other parameters being: the storage overhead $\alpha  = 1$ ,block size $B = {100}$ ,and the number of nearest neighbors equal to 10 . As stated above,the value of $n$ was equal to270,000.

使用纹理特征数据进行的实验旨在衡量局部敏感哈希（LSH）算法在大数据集上的性能；注意，纹理文件的大小（270,000个点）比直方图数据集（$({20},{000}$个点）大一个数量级。第一步（即采样比特数$k$的选择）与上一个实验非常相似，因此我们在此跳过详细描述。我们仅说明我们假设采样比特数为$k = {65}$，其他参数为：存储开销$\alpha  = 1$、块大小$B = {100}$以及最近邻数量等于10。如上所述，$n$的值等于270,000。

We varied the number of indices from 3 to 100 , which resulted in error from ${50}\%$ to ${15}\%$ (see Figure 9 (a)). The shape of the curve is similar as in the previous experiment. The miss ratio was roughly $4\%$ for 3 indices, $1\%$ for 5 indices,and $0\%$ otherwise.

我们将索引数量从3变化到100，这导致误差从${50}\%$到${15}\%$（见图9 (a)）。曲线的形状与上一个实验相似。未命中比率大致为：3个索引时为$4\%$，5个索引时为$1\%$，其他情况为$0\%$。

<!-- Media -->

<!-- figureText: 20 alpha $= 2,1 - \mathrm{{NNS}}$ 64 (a) Approximate 1-NNS alpha $= 2,{10} - \mathrm{{NNS}}$ 64 Dimensions (b) Approximate 10-NNS 18 SR-Tree LSH, error=.02 16 LSH, error=.05 14 LSH, error=.1 LSH, error=.2 Disk accesses 12 10 8 6 4 2 0 35 30 SR-Tree LSH, error=.02 LSH, error=.05 LSH, error=.1 Disk accesses LSH, error=.2 20 10 5 0 8 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_9.jpg?x=205&y=179&w=654&h=1190&r=0"/>

Figure 7: Number of indices vs. dimension.

图7：索引数量与维度的关系。

<!-- Media -->

To compare with SR-tree, we implemented that latter on random subsets of the whole data set of sizes from 10,000 to 200,000 . For $n = {200},{000}$ the average number of blocks accessed per query by SR-tree was 1310, which is one to two orders of magnitude larger than the number of blocks accessed by our algorithm (see Figure 9 (b) where we show the running times of LSH for effective error ${15}\%$ ). Observe though that an SR-tree computes exact answers while LSH provides only an approximation. Thus in order to perform an accurate evaluation of LSH, we decided to compare it with a modified SR-tree algorithm which produces approximate answers. The modification is simple: instead of running SR-tree on the whole data set, we run it on a randomly chosen subset of it. In this way we achieve a speed-up of the algorithm (as the random sample of the data set is smaller than the original set) while incurring some error.

为了与SR树进行比较，我们在整个数据集大小从10,000到200,000的随机子集上实现了SR树。对于$n = {200},{000}$，SR树每次查询平均访问的块数为1310，这比我们的算法访问的块数大1到2个数量级（见图9 (b)，其中我们展示了有效误差为${15}\%$时LSH的运行时间）。不过要注意，SR树计算的是精确答案，而LSH仅提供近似答案。因此，为了准确评估LSH，我们决定将其与一种产生近似答案的改进SR树算法进行比较。修改很简单：我们不是在整个数据集上运行SR树，而是在其随机选择的子集上运行。通过这种方式，我们在产生一定误差的同时实现了算法的加速（因为数据集的随机样本比原始集小）。

<!-- Media -->

<!-- figureText: 15 alpha=2, n=19000, d=64 100 Error=.05 Error=.1 Error=.2 Disk accesses 0 1 10 20 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_9.jpg?x=950&y=183&w=650&h=554&r=0"/>

Figure 8: Number of indices vs. number of nearest neighbors.

图8：索引数量与最近邻数量的关系。

<!-- Media -->

The query cost versus error tradeoff obtained in this way (for the entire data set) is depicted on Figure 9; we also include a similar graph for LSH.

以这种方式（针对整个数据集）获得的查询成本与误差权衡结果如图9所示；我们还给出了局部敏感哈希（LSH）的类似图表。

Observe that using random sampling results in considerable speed-up for the SR-tree algorithm, while keeping the error relatively low. However, even in this case the LSH algorithm offers considerably outperforms SR-trees, being up to an order of magnitude faster.

可以观察到，对SR树算法使用随机采样能显著提高速度，同时保持相对较低的误差。然而，即使在这种情况下，局部敏感哈希（LSH）算法的性能也明显优于SR树，速度快达一个数量级。

## 5 Previous Work

## 5 相关工作

There is considerable literature on various versions of the nearest neighbor problem. Due to lack of space we omit detailed description of related work; the reader is advised to read [39] for a survey of a variety of data structures for nearest neighbors in geometric spaces, including variants of $k$ -d trees, $R$ -trees,and structures based on space-filling curves. The more recent results are surveyed in [41]; see also an excellent survey by [4]. Recent theoretical work in nearest neighbor search is briefly surveyed in [24].

关于最近邻问题的各种变体有大量文献。由于篇幅限制，我们省略了相关工作的详细描述；建议读者阅读文献[39]，以了解几何空间中用于最近邻搜索的各种数据结构，包括$k$ -d树、$R$ -树以及基于空间填充曲线的结构。文献[41]对较新的研究成果进行了综述；另见文献[4]的精彩综述。文献[24]简要综述了最近邻搜索的最新理论研究。

## 6 Conclusions

## 6 结论

We presented a novel scheme for approximate similarity search based on locality-sensitive hashing. We compared the performance of this technique and SR-tree, a good representative of tree-based spatial data structures. We showed that by allowing small error and additional storage overhead, we can considerably improve the query time. Experimental results also indicate that our scheme scales well to even a large number of dimensions and data size. An additional advantage of our data structure is that its running time is essentially determined in advance. All these properties make LSH a suitable candidate for high-performance and real-time systems.

我们提出了一种基于局部敏感哈希（LSH）的近似相似性搜索新方案。我们比较了该技术与SR树（一种基于树的空间数据结构的优秀代表）的性能。我们表明，通过允许较小的误差和额外的存储开销，可以显著提高查询时间。实验结果还表明，我们的方案在处理大量维度和大规模数据时具有良好的扩展性。我们的数据结构的另一个优点是其运行时间基本上可以预先确定。所有这些特性使局部敏感哈希（LSH）成为高性能和实时系统的合适选择。

<!-- Media -->

<!-- figureText: 450 Performance vs error SR-Tree LSH 30 35 45 50 Error (%) (a) 100 150 200 Data Set Size (b) 400 350 Disk accesses 300 250 200 150 100 50 10 15 20 25 1400 1200 SR-Tree Number of Disk Accesses LSH 1000 600 400 200 0 50 -->

<img src="https://cdn.noedgeai.com/0195c90b-2218-7c6f-901b-0b72356e3ae8_10.jpg?x=206&y=184&w=650&h=1124&r=0"/>

Figure 9: (a) number of indices vs. error and (b) number of indices vs. size.

图9：(a) 索引数量与误差的关系；(b) 索引数量与大小的关系。

<!-- Media -->

In recent work $\left\lbrack  {5,{23}}\right\rbrack$ ,we explore applications of LSH-type techniques to data mining and search for copyrighted video data. Our experience suggests that there is a lot of potential for further improvement of the performance of the LSH algorithm. For example, our data structures are created using a randomized procedure. It would be interesting if there was a more systematic method for performing this task; such a method could take additional advantage of the structure of the data set. We also believe that investigation of hybrid data structures obtained by merging the tree-based and hashing-based approaches is a fruitful direction for further research.

在最近的工作$\left\lbrack  {5,{23}}\right\rbrack$中，我们探索了局部敏感哈希（LSH）类型技术在数据挖掘和版权视频数据搜索中的应用。我们的经验表明，局部敏感哈希（LSH）算法的性能还有很大的提升潜力。例如，我们的数据结构是通过随机化过程创建的。如果有一种更系统的方法来完成这项任务，那将很有趣；这种方法可以进一步利用数据集的结构。我们还认为，研究将基于树的方法和基于哈希的方法相结合的混合数据结构是一个富有成果的进一步研究方向。

## References

## 参考文献

[1] S. Arya, D.M. Mount, and O. Narayan, Accounting for boundary effects in nearest-neighbor searching. Discrete and Computational Geometry, 16 (1996), pp. 155-176.

[2] S. Arya, D.M. Mount, N.S. Netanyahu, R. Silverman, and A. Wu. An optimal algorithm for approximate nearest neighbor searching, In Proceedings of the 5th Annual ACM-SIAM Symposium on Discrete Algorithms, 1994, pp. 573-582.

[3] J.L. Bentley. Multidimensional binary search trees used for associative searching. Communications of the ${ACM},{18}\left( {1975}\right)$ ,pp. 509-517.

[4] S. Berchtold and D.A. Keim. High-dimensional Index Structures. In Proceedings of SIGMOD, 1998, p. 501. See http://www.informatik.uni-halle.de/ keim/ SIGMOD98Tutorial.ps.gz

[5] E. Cohen. M. Datar, S. Fujiwara, A. Gionis, P. Indyk, R. Motwani, J. D. Ullman. C. Yang. Finding Interesting Associations without Support Pruning. Technical Report, Computer Science Department, Stanford University.

[6] T.M. Chan. Approximate Nearest Neighbor Queries Revisited. In Proceedings of the 13th Annual ACM Symposium on Computational Geometry, 1997, pp. 352-358.

[7] S. Cost and S. Salzberg. A weighted nearest neighbor algorithm for learning with symbolic features. Machine Learning, 10 (1993), pp. 57-67.

[8] S. Chaudhuri, R. Motwani and V. Narasayya. "Random Sampling for Histogram Construction: How much is enough?". In Proceedings of SIGMOD'98, pp. 436-447.

[9] T.M. Cover and P.E. Hart. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13 (1967), pp. 21-27.

[10] P. Ciaccia, M. Patella and P. Zezula, A cost model for similarity queries in metric spaces In Proceedings of PODS'98, pp. 59-68.

[11] S. Deerwester, S. T. Dumais, T.K. Landauer, G.W. Furnas, and R.A. Harshman. Indexing by latent semantic analysis. Journal of the Society for Information Sciences, 41 (1990), pp. 391-407.

[12] L. Devroye and T.J. Wagner. Nearest neighbor methods in discrimination. Handbook of Statistics, vol. 2, P.R. Krishnaiah and L.N. Kanal, eds., North-Holland, 1982.

[13] R.O. Duda and P.E. Hart. Pattern Classification and Scene Analysis. John Wiley & Sons, NY, 1973.

[14] H. Edelsbrunner. Algorithms in Combinatorial Geometry. Springer-Verlag, 1987.

[15] C. Faloutsos, R. Barber, M. Flickner, W. Niblack, D. Petkovic, and W. Equitz. Efficient and effective querying by image content. Journal of Intelligent Information Systems, 3 (1994), pp. 231-262.

[16] C. Faloutsos and D.W. Oard. A Survey of Information Retrieval and Filtering Methods. Technical Re-

[16] C. Faloutsos和D.W. Oard。信息检索和过滤方法综述。技术报告 -

port CS-TR-3514, Department of Computer Science, University of Maryland, College Park, 1995.

[17] M. Flickner, H. Sawhney, W. Niblack, J. Ashley, Q. Huang, B. Dom, M. Gorkani, J. Hafner, D. Lee, D. Petkovic, D. Steele, and P. Yanker. Query by image and video content: the QBIC system. IEEE Computer, 28 (1995), pp. 23-32.

[18] J.K. Friedman, J.L. Bentley, and R.A. Finkel. An algorithm for finding best matches in logarithmic expected time. ACM Transactions on Mathematical Software, 3 (1977), pp. 209-226.

[19] T. Figiel, J. Lindenstrauss, V. D. Milman. The dimension of almost spherical sections of convex bodies. Acta Math. 139 (1977), no. 1-2, 53-94.

[20] A. Gersho and R.M. Gray. Vector Quantization and Data Compression. Kluwer, 1991.

[21] T. Hastie and R. Tibshirani. Discriminant adaptive nearest neighbor classification. In Proceedings of the First International Conference on Knowledge Discovery & Data Mining, 1995, pp. 142-149.

[22] H. Hotelling. Analysis of a complex of statistical variables into principal components. Journal of Educational Psychology, 27 (1933). pp. 417-441.

[23] P. Indyk, G. Iyengar, N. Shivakumar. Finding pirated video sequences on the Internet. Technical Report, Computer Science Department, Stanford University.

[24] P. Indyk and R. Motwani. Approximate Nearest Neighbor - Towards Removing the Curse of Dimensionality. In Proceedings of the 30th Symposium on Theory of Computing, 1998, pp. 604-613.

[25] K.V. Ravi Kanth, D. Agrawal, A. Singh. "Dimensionality Reduction for Similarity Searching in Dynamic Databases". In Proceedings of SIGMOD'98, 166-176.

[26] K. Karhunen. Über lineare Methoden in der Wahrscheinlichkeitsrechnung. Ann. Acad. Sci. Fen-nicae, Ser. A137, 1947.

[27] V. Koivune and S. Kassam. Nearest neighbor filters for multivariate data. IEEE Workshop on Nonlinear Signal and Image Processing, 1995.

[28] N. Katayama and S. Satoh. The SR-tree: an index structure for high-dimensional nearest neighbor queries. In Proc. SIGMOD'97, pp. 369-380. The code is available from http://www.rd.nacsis.ac.jp/ ~katayama/homepage/research/srtree/English.html

[29] N. Linial, E. London, and Y. Rabinovich. The geometry of graphs and some of its algorithmic applications. In Proceedings of 35th Annual IEEE Symposium on Foundations of Computer Science, 1994, pp. 577-591.

[30] M. Loéve. Fonctions aleastoires de second ordere. Processus Stochastiques et mouvement Brownian, Hermann, Paris, 1948.

[31] B. S. Manjunath. Airphoto dataset. http://vivaldi.ece.ucsb.edu/Manjunath/research.htm

[32] B. S. Manjunath and W. Y. Ma. Texture features for browsing and retrieval of large image data. IEEE

[32] B. S. Manjunath和W. Y. Ma。用于大型图像数据浏览和检索的纹理特征。IEEE

Transactions on Pattern Analysis and Machine Intelligence, (Special Issue on Digital Libraries), 18 (8), pp. 837-842.

[33] G.S. Manku, S. Rajagopalan, and B.G. Lindsay. Approximate Medians and other Quantiles in One Pass and with Limited Memory. In Proceedings of SIG-MOD'98, pp. 426-435.

[34] Y. Matias, J.S. Vitter, and M. Wang. Wavelet-based Histograms for Selectivity Estimations. In Proceedings of SIGMOD'98, pp. 448-459.

[35] R. Motwani and P. Raghavan. Randomized Algorithms. Cambridge University Press, 1995.

[36] M. Otterman. Approximate matching with high dimensionality R-trees. M.Sc. Scholarly paper, Dept. of Computer Science, Univ. of Maryland, College Park, MD, 1992.

[37] A. Pentland, R.W. Picard, and S. Sclaroff. Photo-book: tools for content-based manipulation of image databases. In Proceedings of the SPIE Conference on Storage and Retrieval of Image and Video Databases II, 1994.

[38] G. Salton and M.J. McGill. Introduction to Modern Information Retrieval. McGraw-Hill Book Company, New York, NY, 1983.

[39] H. Samet. The Design and Analysis of Spatial Data Structures. Addison-Wesley, Reading, MA, 1989.

[40] J. R. Smith. Integrated Spatial and Feature Image Systems: Retrieval, Analysis and Compression. Ph.D. thesis, Columbia University, 1997. Available at ftp://ftp.ctr.columbia.edu/CTR-Research/advent/ public/public/jrsmith/thesis

[41] T. Sellis, N. Roussopoulos, and C. Faloutsos. Multidimensional Access Methods: Trees Have Grown Everywhere. In Proceedings of the 23rd International Conference on Very Large Data Bases, 1997, 13-15.

[42] A.W.M. Smeulders and R. Jain, eds. Image Databases and Multi-media Search. Proceedings of the First International Workshop, IDB-MMS'96, Amsterdam University Press, Amsterdam, 1996.

[43] J.R. Smith and S.F. Chang. Visually Searching the Web for Content. IEEE Multimedia 4 (1997): pp. 12-20. See also http://disney.ctr.columbia.edu/WebSEEk

[44] G. Wyszecki and W.S. Styles. Color science: concepts and methods, quantitative data and formulae. John Wiley and Sons, New York, NY, 1982.

[45] R. Weber, H. Schek, and S. Blott. A quantitative analysis and performance study for Similarity Search Methods in High Dimensional Spaces. In Proceedings of the 24th International Conference on Very Large Data Bases (VLDB), 1998, pp. 194-205.