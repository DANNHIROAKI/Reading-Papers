<!-- Media -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_0.jpg?x=1348&y=9&w=230&h=225&r=0"/>

<!-- Media -->

# CIVET: Exploring Compact Index for Variable-Length Subsequence Matching on Time Series

# CIVET：探索用于时间序列可变长度子序列匹配的紧凑索引

Haoran Xiong

熊浩然

Fudan University

复旦大学

hrxiong20@fudan.edu.cn

Hang Zhang

张航

Fudan University

复旦大学

zhanghang21@m.fudan.edu.cn

Zeyu Wang

王泽宇

Fudan University

复旦大学

zeyuwang21@m.fudan.edu.cn

Zhenying ${\mathrm{{He}}}^{ * }$

甄莹 ${\mathrm{{He}}}^{ * }$

Fudan University

复旦大学

zhenying@fudan.edu.cn

Peng Wang*

王鹏*

Fudan University

复旦大学

pengwang5@fudan.edu.cn

X. Sean Wang

王肖恩

Fudan University

复旦大学

xywangcs@fudan.edu.cn

## ABSTRACT

## 摘要

Nowadays the demands for managing and analyzing substantially increasing collections of time series are becoming more challenging. Subsequence matching, as a core subroutine in time series analysis, has drawn significant research attention. Most of the previous works only focus on matching the subsequences with equal length to the query. However, many scenarios require support for efficient variable-length subsequence matching. In this paper, we propose a new representation, Uniform Piecewise Aggregate Approximation (UPAA) with the capability of aligning features for variable-length time series while remaining the lower bounding property. Based on UPAA, we present a compact index structure by grouping adjacent subsequences and similar subsequences respectively. Moreover, we propose an index pruning algorithm and a data filtering strategy to efficiently support variable-length subsequence matching without false dismissals. The experiments conducted on both real and synthetic datasets demonstrate that our approach achieves considerably better efficiency, scalability, and effectiveness than existing approaches.

如今，管理和分析数量大幅增长的时间序列集合的需求变得更具挑战性。子序列匹配作为时间序列分析中的核心子程序，已引起了大量的研究关注。以往的大多数工作仅专注于匹配与查询长度相等的子序列。然而，许多场景需要支持高效的可变长度子序列匹配。在本文中，我们提出了一种新的表示方法——统一分段聚合近似（Uniform Piecewise Aggregate Approximation，UPAA），它能够对可变长度时间序列的特征进行对齐，同时保留下界特性。基于UPAA，我们分别通过对相邻子序列和相似子序列进行分组，提出了一种紧凑的索引结构。此外，我们还提出了一种索引剪枝算法和一种数据过滤策略，以在不产生漏检的情况下高效支持可变长度子序列匹配。在真实和合成数据集上进行的实验表明，与现有方法相比，我们的方法在效率、可扩展性和有效性方面有显著提升。

## PVLDB Reference Format:

## PVLDB 引用格式：

Haoran Xiong, Hang Zhang, Zeyu Wang, Zhenying He, Peng Wang, and X. Sean Wang. CIVET: Exploring Compact Index for Variable-Length Subsequence Matching on Time Series. PVLDB, 17(9): 2123-2135, 2024. doi:10.14778/3665844.3665845

熊浩然、张航、王泽宇、何振英、王鹏和王肖恩（X. Sean Wang）。CIVET：探索用于时间序列可变长度子序列匹配的紧凑索引。《大型数据库会议论文集》（PVLDB），17(9)：2123 - 2135，2024。doi:10.14778/3665844.3665845

## PVLDB Artifact Availability:

## 《大型数据库会议论文集》（PVLDB）工件可用性：

The source code, data, and/or other artifacts have been made available at https://github.com/CIVET-TS/CIVET.

源代码、数据和/或其他工件已在https://github.com/CIVET-TS/CIVET上提供。

## 1 INTRODUCTION

## 1 引言

Time series has become prevalent due to numerous applications generating extensive collections of time-stamped data [13, 15, 42], necessitating advanced analysis techniques for valuable insights. As a core subroutine of time series analysis, subsequence matching has attracted significant attention and research effort $\left\lbrack  {{25},{30},{31},{39}}\right\rbrack$ .

由于众多应用生成了大量带时间戳的数据集合，时间序列变得十分普遍[13, 15, 42]，这就需要先进的分析技术来获取有价值的见解。作为时间序列分析的核心子程序，子序列匹配吸引了大量的关注和研究工作$\left\lbrack  {{25},{30},{31},{39}}\right\rbrack$。

Informally, subsequence matching finds subsequences from a long sequence that are similar to a given query sequence. Many subsequence matching approaches focus on matching the subsequences with the same length as the query $\left\lbrack  {8,{29}}\right\rbrack$ . However,it has been demonstrated that many applications call for sequence matching approaches that allow for variable length $\left\lbrack  {{12},{18}}\right\rbrack$ . We illustrate it with an example below.

通俗地说，子序列匹配是从一个长序列中找出与给定查询序列相似的子序列。许多子序列匹配方法专注于匹配与查询序列长度相同的子序列$\left\lbrack  {8,{29}}\right\rbrack$。然而，事实证明，许多应用需要允许可变长度的序列匹配方法$\left\lbrack  {{12},{18}}\right\rbrack$。下面我们通过一个例子来说明。

<!-- Media -->

<!-- figureText: (c) Head-and-Shoulders (d) Head-and-Shoulder: 187 days -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_0.jpg?x=929&y=917&w=713&h=365&r=0"/>

Figure 1: Variable-length Subsequence Matching on four Stock datasets. The Triangles pattern $\left( {Q}_{1}\right)$ searches datasets (a) and (b),yielding results ${S}_{1}$ and ${S}_{2}$ . The Head-and-shoulders pattern $\left( {Q}_{2}\right)$ searches datasets (c) and (d),yielding results ${S}_{3}$ and ${S}_{4}$ . (e) shows the process of variable-length subsequence matching between ${Q}_{1}$ and ${S}_{1},{S}_{2}$ .

图1：四个股票数据集上的可变长度子序列匹配。三角形模式$\left( {Q}_{1}\right)$搜索数据集(a)和(b)，得到结果${S}_{1}$和${S}_{2}$。头肩模式$\left( {Q}_{2}\right)$搜索数据集(c)和(d)，得到结果${S}_{3}$和${S}_{4}$。(e)展示了${Q}_{1}$和${S}_{1},{S}_{2}$之间可变长度子序列匹配的过程。

<!-- Media -->

Example 1. In stock trading, technical analysis is a financial method to identify investigation opportunities by studying historical trading activity [1]. As a concrete example, Figure 1 presents four sequences of stock prices, which conform to two classic chart patterns in technical analysis,triangles $\left( {{S}_{1},{S}_{2}}\right.$ in Figure 1(a) and (b)) and head-and-shoulders $\left( {{S}_{3},{S}_{4}}\right.$ in Figure 1(c) and (d)) [9],depicted with highlighted lines. ${Q}_{1}$ and ${Q}_{2}$ are utilized as query sequences for two chart patterns, employed for subsequence matching.

示例1。在股票交易中，技术分析是一种通过研究历史交易活动来识别投资机会的金融方法[1]。作为一个具体的例子，图1展示了四个股票价格序列，它们符合技术分析中的两种经典图表模式，即图1(a)和(b)中的三角形$\left( {{S}_{1},{S}_{2}}\right.$以及图1(c)和(d)中的头肩模式$\left( {{S}_{3},{S}_{4}}\right.$[9]，用高亮线表示。${Q}_{1}$和${Q}_{2}$被用作两种图表模式的查询序列，用于子序列匹配。

Different subsequences though adhering to the same pattern, may have different lengths. We name this phenomenon as global scaling. In this case, only finding subsequences of the same length as query will lose many meaningful results. To solve this problem, uniform scaling has been proposed to align variable-length subsequences $\left\lbrack  {{18},{31}}\right\rbrack$ . With uniform scaling,the query sequence is uniformly stretched or shrunk to different lengths and aligned with the target subsequence, thus solving global scaling in the time dimension. This technique is widely employed in actual application $\left\lbrack  {{10},{18},{33},{41}}\right\rbrack$ . In addition,to focus on the shape of subsequences and eliminate the effect offset shifting and amplitude, z-normalization $\left\lbrack  {{17},{30}}\right\rbrack$ is often used before distance calculation, which still works in the context of uniform scaling.

尽管遵循相同的模式，但不同的子序列可能具有不同的长度。我们将这种现象称为全局缩放。在这种情况下，只查找与查询长度相同的子序列会丢失许多有意义的结果。为了解决这个问题，人们提出了均匀缩放来对齐可变长度的子序列$\left\lbrack  {{18},{31}}\right\rbrack$。通过均匀缩放，查询序列被均匀地拉伸或收缩到不同的长度，并与目标子序列对齐，从而解决了时间维度上的全局缩放问题。这种技术在实际应用中被广泛使用$\left\lbrack  {{10},{18},{33},{41}}\right\rbrack$。此外，为了关注子序列的形状并消除偏移和幅度的影响，在距离计算之前通常会使用z - 归一化$\left\lbrack  {{17},{30}}\right\rbrack$，这在均匀缩放的情况下仍然有效。

---

<!-- Footnote -->

*Corresponding authors

*通讯作者

This work is licensed under the Creative Commons BY-NC-ND 4.0 International License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of this license. For any use beyond those covered by this license, obtain permission by emailing info@vldb.org.Copyright is held by the owner/author(s). Publication rights licensed to the VLDB Endowment.

本作品采用知识共享署名 - 非商业性使用 - 禁止演绎4.0国际许可协议进行许可。请访问https://creativecommons.org/licenses/by - nc - nd/4.0/查看此许可协议的副本。如需进行超出此许可范围的使用，请通过发送电子邮件至info@vldb.org获得许可。版权归所有者/作者所有。出版权授予大型数据库会议基金会（VLDB Endowment）。

Proceedings of the VLDB Endowment, Vol. 17, No. 9 ISSN 2150-8097.

《大型数据库会议论文集》（Proceedings of the VLDB Endowment），第17卷，第9期，国际标准连续出版物编号：2150 - 8097。

doi:10.14778/3665844.3665845

doi:10.14778/3665844.3665845

<!-- Footnote -->

---

In this paper, we focus on the variable-length subsequence matching problem under uniform scaling and z-normalization. During the variable-length subsequence matching, the query sequence is first uniformly-scaled to variable lengths and then find similar subsequences of the corresponding lengths [12]. Figure 1 shows two concrete examples. Before calculating the distances,the query ${Q}_{1}$ is scaled to sequences ${Q}_{1}^{\prime }$ and ${Q}_{1}^{\prime \prime }$ whose lengths are the same as ${S}_{1}$ and ${S}_{2}$ and then computing distance (say,Euclidean Distance) between them after z-normalization. The hat symbols (e.g., $\widehat{{Q}_{1}^{\prime }}$ ) indicate z-normalized sequences.

在本文中，我们聚焦于均匀缩放和z-归一化条件下的可变长度子序列匹配问题。在可变长度子序列匹配过程中，首先将查询序列均匀缩放到不同长度，然后找出相应长度的相似子序列[12]。图1展示了两个具体示例。在计算距离之前，将查询序列${Q}_{1}$缩放为序列${Q}_{1}^{\prime }$和${Q}_{1}^{\prime \prime }$，使其长度分别与${S}_{1}$和${S}_{2}$相同，然后在z-归一化后计算它们之间的距离（例如，欧几里得距离）。帽符号（例如，$\widehat{{Q}_{1}^{\prime }}$）表示z-归一化后的序列。

Compared with fixed-length subsequence matching, the variable-length subsequence matching problem is more challenging. First, the variable-length case has a much larger search space. Given a length- $l$ query subsequence,fixed-length matching only needs to verify all equi-length subsequences in the database, whereas variable-length matching needs all the subsequences in the concerned range of lengths. Second, calculating distance between subsequences of different lengths is more time-consuming and it involves a large number of redundant calculations when computing distances between query and different subsequences of long time series. Therefore, it is essential to design an efficient index and query algorithm to avoid unnecessary computations.

与固定长度子序列匹配相比，可变长度子序列匹配问题更具挑战性。首先，可变长度情况下的搜索空间要大得多。给定一个长度为$l$的查询子序列，固定长度匹配只需验证数据库中所有等长的子序列，而可变长度匹配则需要考虑相关长度范围内的所有子序列。其次，计算不同长度子序列之间的距离更耗时，并且在计算查询序列与长时间序列的不同子序列之间的距离时会涉及大量冗余计算。因此，设计一种高效的索引和查询算法以避免不必要的计算至关重要。

Numerous works have studied variable-length sequence matching problem under uniform scaling $\left\lbrack  {{10},{12},{18},{41}}\right\rbrack$ . However,these approaches do not consider the subsequences of the time series in the database, despite that the length of the subsequences are in the user's concerned range. Thus, the search space of this problem is substantially smaller than ours. Other works [31, 33] study the variable-length subsequence matching problem. Nonetheless, these methods cannot take effect with z-normalization due to the design of distance bounds. Thus, their results are prone to get disturbed by the offset shifting and amplitude of subsequences.

许多研究已经探讨了均匀缩放$\left\lbrack  {{10},{12},{18},{41}}\right\rbrack$条件下的可变长度序列匹配问题。然而，这些方法并未考虑数据库中时间序列的子序列，尽管这些子序列的长度在用户关注的范围内。因此，该问题的搜索空间比我们的要小得多。其他研究[31, 33]则研究了可变长度子序列匹配问题。尽管如此，由于距离边界的设计，这些方法在z-归一化时无法生效。因此，它们的结果容易受到子序列的偏移和幅度的干扰。

In this paper, we propose an extended PAA that can eliminate the influence of global scaling between time series. Besides, we have meticulously designed a compact index structure, which stores similar but variable-length subsequences closely, possessing the ability to handle scaled queries of different lengths at once. Moreover, our methodologies in representation and index structures have been meticulously designed. They can efficiently facilitate exact top-K queries without incurring any false dismissals combined with the respective lower bound distances. In addition, we formulate enveloping sequences by utilizing the monotonicity of the mean and standard deviation values in z-normalization. Leveraging these enveloping sequences, we introduce a robust lower bounding distance designed to efficaciously eliminate redundant distance computations in the context of uniform scaling and normalization.

在本文中，我们提出了一种扩展的分段聚合近似（PAA）方法，该方法可以消除时间序列之间全局缩放的影响。此外，我们精心设计了一种紧凑的索引结构，该结构可以紧密存储相似但长度可变的子序列，具备一次性处理不同长度缩放查询的能力。而且，我们在表示方法和索引结构方面进行了精心设计。结合各自的下界距离，它们可以有效地实现精确的前K查询，且不会产生任何误判。此外，我们利用z-归一化中均值和标准差的单调性来构建包络序列。利用这些包络序列，我们引入了一种稳健的下界距离，旨在有效地消除均匀缩放和归一化情况下的冗余距离计算。

In summary, facilitated by a new representation technique, we design a compact index structure supporting efficient variable-length subsequence matching with the help of index pruning and data filtering strategies. We call the approach as CIVET (Compact Index for Variable-length subsequencE matching on Time series). CIVET is experimentally proved to be more advanced than the SOTA approaches in terms of approximate matching accuracy and exact matching performance under uniform scaling and normalization.

综上所述，借助一种新的表示技术，我们结合索引剪枝和数据过滤策略，设计了一种紧凑的索引结构，以支持高效的可变长度子序列匹配。我们将该方法称为CIVET（时间序列可变长度子序列匹配的紧凑索引）。实验证明，在均匀缩放和归一化条件下，CIVET在近似匹配精度和精确匹配性能方面比现有最优方法更具优势。

<!-- Media -->

Table 1: Table of Symbols

表1：符号表

<table><tr><td>Symbols</td><td>Description</td></tr><tr><td>$T = \left( {{t}_{1},{t}_{2},\ldots ,{t}_{n}}\right)$</td><td>Time series</td></tr><tr><td>| S |</td><td>Length of $S$</td></tr><tr><td>${T}_{i,l}$</td><td>Subsequence of $T$ (from $i$ to $i + l - 1$ )</td></tr><tr><td>$\widehat{S}$</td><td>Z-normalized $S$</td></tr><tr><td>${Q}^{p}$</td><td>$Q$ scaled to length $p$ with uniform scaling</td></tr><tr><td>${S}_{i}$</td><td>The $i$ -th segment of $S$ in UniSeg</td></tr><tr><td>${r}_{i}$</td><td>The last point index of $i$ -th segment</td></tr><tr><td>${D}_{ed}\left( {\cdot , \cdot  }\right)$</td><td>Euclidean distance</td></tr><tr><td>${D}_{dtw}\left( {\cdot , \cdot  }\right)$</td><td>DTW distance</td></tr><tr><td>${D}_{usn}^{ - }\left( \cdot \right)$</td><td>Uniform scaling distance with normalization</td></tr></table>

<table><tbody><tr><td>符号</td><td>描述</td></tr><tr><td>$T = \left( {{t}_{1},{t}_{2},\ldots ,{t}_{n}}\right)$</td><td>时间序列</td></tr><tr><td>| S |</td><td>$S$的长度</td></tr><tr><td>${T}_{i,l}$</td><td>$T$的子序列（从$i$到$i + l - 1$）</td></tr><tr><td>$\widehat{S}$</td><td>经过Z归一化处理的$S$</td></tr><tr><td>${Q}^{p}$</td><td>$Q$通过均匀缩放调整为长度$p$</td></tr><tr><td>${S}_{i}$</td><td>UniSeg中$S$的第$i$个分段</td></tr><tr><td>${r}_{i}$</td><td>第$i$个分段的最后一个点的索引</td></tr><tr><td>${D}_{ed}\left( {\cdot , \cdot  }\right)$</td><td>欧几里得距离</td></tr><tr><td>${D}_{dtw}\left( {\cdot , \cdot  }\right)$</td><td>动态时间规整距离（DTW距离）</td></tr><tr><td>${D}_{usn}^{ - }\left( \cdot \right)$</td><td>经过归一化处理的均匀缩放距离</td></tr></tbody></table>

<!-- Media -->

The contributions of this paper can be summarized as follows:

本文的贡献可总结如下：

- The Uniform Piecewise Aggregate Approximation (UPAA) is introduced to manage variable lengths, aligning feature representations while retaining essential properties of PAA, thus enhancing robustness against global scaling.

- 引入统一分段聚合近似（Uniform Piecewise Aggregate Approximation，UPAA）来处理可变长度，在保留分段聚合近似（PAA）基本属性的同时对齐特征表示，从而增强对全局缩放的鲁棒性。

- A new indexing method is designed, constructing a compact index structure by grouping adjacent subsequences and subsequently grouping subsequences with similar features.

- 设计了一种新的索引方法，通过对相邻子序列进行分组，然后对具有相似特征的子序列进行分组，构建了一个紧凑的索引结构。

- Leveraging lower bounding properties, we propose effective index pruning and data filtering techniques, both tailored for global scaling and z-normalization and compatible with ED and DTW distances.

- 利用下界属性，我们提出了有效的索引剪枝和数据过滤技术，这些技术既适用于全局缩放和z - 归一化，又与欧几里得距离（ED）和动态时间规整距离（DTW）兼容。

The paper is organized as follows: Section 2 presents the formal problem statement and method overview. Section 3 extends the PAA to represent variable-length time series. Sections 4 and 5 introduce the details of index construction and matching algorithms. Section 6 presents and discusses the experimental results. Section 7 discusses related works. Section 8 concludes the paper.

本文的组织结构如下：第2节给出正式的问题陈述和方法概述。第3节扩展了分段聚合近似（PAA）以表示可变长度的时间序列。第4节和第5节介绍索引构建和匹配算法的细节。第6节展示并讨论实验结果。第7节讨论相关工作。第8节对本文进行总结。

## 2 PROBLEM STATEMENT

## 2 问题陈述

### 2.1 Preliminaries and Problem Formulation

### 2.1 预备知识和问题表述

Time series is a sequence of values listed in time order, denoted as $T = \left( {{t}_{1},{t}_{2},\cdots ,{t}_{n}}\right)$ ,where $n = \left| T\right|$ is the length of $T$ . Subsequence ${T}_{i,l}$ of time series $T$ is a length- $l$ contiguous sequence within $T$ ,which starts from position $i$ . Formally,we denoted ${T}_{i,l}$ as ${T}_{i,l} = \left( {{t}_{i},{t}_{i + 1},\cdots ,{t}_{i + l - 1}}\right)$ ,where $1 \leq  i \leq  \left| T\right|  - l + 1$ . Later in this paper,we refer to query sequence and subsequence as $Q$ and $S$ for distinction. $T$ is used to refer to a long sequence specifically. For any subsequences $S = \left( {{s}_{1},{s}_{2},\cdots ,{s}_{n}}\right)$ ,we use ${\mu }^{S}$ and ${\sigma }^{S}$ to denote the mean value and standard deviation of $S$ respectively.

时间序列是按时间顺序列出的值的序列，记为 $T = \left( {{t}_{1},{t}_{2},\cdots ,{t}_{n}}\right)$ ，其中 $n = \left| T\right|$ 是 $T$ 的长度。时间序列 $T$ 的子序列 ${T}_{i,l}$ 是 $T$ 内长度为 $l$ 的连续序列，它从位置 $i$ 开始。形式上，我们将 ${T}_{i,l}$ 记为 ${T}_{i,l} = \left( {{t}_{i},{t}_{i + 1},\cdots ,{t}_{i + l - 1}}\right)$ ，其中 $1 \leq  i \leq  \left| T\right|  - l + 1$ 。在本文后面，为了区分，我们将查询序列和子序列分别称为 $Q$ 和 $S$ 。 $T$ 专门用于指代长序列。对于任何子序列 $S = \left( {{s}_{1},{s}_{2},\cdots ,{s}_{n}}\right)$ ，我们分别使用 ${\mu }^{S}$ 和 ${\sigma }^{S}$ 来表示 $S$ 的均值和标准差。

Definition 1 (Euclidean Distance (ED)). Given $Q$ and $S$ with the same length $l,{ED}$ between them is ${D}_{ed}\left( {Q,S}\right)  = \sqrt{\mathop{\sum }\limits_{{i = 1}}^{l}{\left( {q}_{i} - {s}_{i}\right) }^{2}}$ .

定义1（欧几里得距离（Euclidean Distance，ED））。给定 $Q$ 和 $S$ ，它们之间长度为 $l,{ED}$ 的距离为 ${D}_{ed}\left( {Q,S}\right)  = \sqrt{\mathop{\sum }\limits_{{i = 1}}^{l}{\left( {q}_{i} - {s}_{i}\right) }^{2}}$ 。

ED computes the distance between two sequences with one-to-one map, while DTW eliminates local misalignment with one-to-many map. The warping path is used to describe the mapping relation between two sequences.

欧几里得距离（ED）通过一对一映射计算两个序列之间的距离，而动态时间规整（DTW）通过一对多映射消除局部不对齐。弯曲路径用于描述两个序列之间的映射关系。

Definition 2 (WARPING PATH). Given two length-l sequences, $Q$ and $S$ ,a warping path is denoted as $A = \left( {{a}_{1},{a}_{2},\ldots ,{a}_{\left| A\right| }}\right)$ . The $x$ -th element ${a}_{x} = \left( {i,j}\right)$ is a pair of values representing the mapping between ${Q}_{i}$ and ${S}_{j}$ . We use ${a}_{x}$ . fst and ${a}_{x}$ .snd to refer to the first and second values of ${a}_{x}$ . A warping path satisfies the following constraints: (1) $1 \leq  i,j \leq  l,{a}_{1} = \left( {1,1}\right)$ ,and ${a}_{\left| A\right| } = \left( {l,l}\right) ,\left( 2\right) 0 \leq  {a}_{x + 1} \cdot  {fst} -$ ${a}_{x}$ . fst $\leq  1$ and $0 \leq  {a}_{x + 1}$ .snd $- {a}_{x}$ .snd $\leq  1$ .

定义2（弯曲路径（WARPING PATH））。给定两个长度为l的序列 $Q$ 和 $S$ ，弯曲路径记为 $A = \left( {{a}_{1},{a}_{2},\ldots ,{a}_{\left| A\right| }}\right)$ 。第 $x$ 个元素 ${a}_{x} = \left( {i,j}\right)$ 是一对值，表示 ${Q}_{i}$ 和 ${S}_{j}$ 之间的映射。我们使用 ${a}_{x}$ .fst 和 ${a}_{x}$ .snd 来指代 ${a}_{x}$ 的第一个和第二个值。弯曲路径满足以下约束条件：（1） $1 \leq  i,j \leq  l,{a}_{1} = \left( {1,1}\right)$ ，并且 ${a}_{\left| A\right| } = \left( {l,l}\right) ,\left( 2\right) 0 \leq  {a}_{x + 1} \cdot  {fst} -$ ${a}_{x}$ .fst $\leq  1$ 以及 $0 \leq  {a}_{x + 1}$ .snd $- {a}_{x}$ .snd $\leq  1$ 。

Definition 3 (Constrained Dynamic Time Warping Distance). Given two length-l sequences $Q$ and $S$ ,and the time warping constraint $c$ ,the constrained dynamic time warping distance between them is defined as, ${D}_{dtw}\left( {Q,S}\right)  = \underset{A}{\arg \min }\sqrt{\mathop{\sum }\limits_{i}^{\left| A\right| }{\left( {q}_{{a}_{i}.{fst}} - {s}_{{a}_{i}.{snd}}\right) }^{2}}$ . According to the Sakoe-Chiba constraint [32],any element ${a}_{x}$ in the warping path $A$ satisfies that $\left| {{a}_{x}\text{. fst-}{a}_{x}\text{.snd}}\right|  \leq  c$ .

定义3（受限动态时间规整距离）。给定两个长度为l的序列$Q$和$S$，以及时间规整约束$c$，它们之间的受限动态时间规整距离定义为${D}_{dtw}\left( {Q,S}\right)  = \underset{A}{\arg \min }\sqrt{\mathop{\sum }\limits_{i}^{\left| A\right| }{\left( {q}_{{a}_{i}.{fst}} - {s}_{{a}_{i}.{snd}}\right) }^{2}}$。根据佐伯 - 千叶约束[32]，规整路径$A$中的任何元素${a}_{x}$都满足$\left| {{a}_{x}\text{. fst-}{a}_{x}\text{.snd}}\right|  \leq  c$。

Definition 4 (Uniform Scaling). Given a length-n sequence $Q$ and a length $p$ ,the uniform scaling stretches up (if $n < p$ ) or shrinks down (if $n > p$ ) $S$ to a length-p time series ${Q}^{p} = \left( {{q}_{1}^{p},{q}_{2}^{p},\cdots ,{q}_{p}^{p}}\right)$ , where

定义4（均匀缩放）。给定一个长度为n的序列$Q$和一个长度$p$，均匀缩放会将$S$拉伸（如果$n < p$）或收缩（如果$n > p$）为一个长度为p的时间序列${Q}^{p} = \left( {{q}_{1}^{p},{q}_{2}^{p},\cdots ,{q}_{p}^{p}}\right)$，其中

$$
{q}_{i}^{p} = {q}_{\left\lceil  i * \frac{n}{p}\right\rceil  },1 \leq  i \leq  p
$$

DEFINITION 5 (Z-NORMALIZED SERIES). Given a length-n time series $S$ ,a normalized series of $S$ ,is defined as, $\widehat{S} = \left( {{\widehat{s}}_{1},{\widehat{s}}_{2},\cdots ,{\widehat{s}}_{n}}\right)$ , where ${\widehat{s}}_{i} = \frac{{s}_{i} - {\mu }^{S}}{{\sigma }^{S}},1 \leq  i \leq  n$ .

定义5（Z - 归一化序列）。给定一个长度为n的时间序列$S$，$S$的归一化序列定义为$\widehat{S} = \left( {{\widehat{s}}_{1},{\widehat{s}}_{2},\cdots ,{\widehat{s}}_{n}}\right)$，其中${\widehat{s}}_{i} = \frac{{s}_{i} - {\mu }^{S}}{{\sigma }^{S}},1 \leq  i \leq  n$。

Under the influence of time series length, ED between shorter time series is more likely small, even though they could be less similar. Therefore, the length norm is adopted to eliminate the influence of different lengths. It divides the distance by $\sqrt{l}$ ,where $l$ is the length of the sequences [26].

在时间序列长度的影响下，较短时间序列之间的欧几里得距离（ED）更有可能较小，即使它们的相似度可能较低。因此，采用长度归一化来消除不同长度的影响。它将距离除以$\sqrt{l}$，其中$l$是序列的长度[26]。

Definition 6 (Uniform Scaling Distance with Norm). Given $Q$ and $S$ ,the uniform scaling distance with both $z$ -norm and length-norm between them is defined as follows:

定义6（带归一化的均匀缩放距离）。给定$Q$和$S$，它们之间同时具有$z$ - 范数和长度归一化的均匀缩放距离定义如下：

$$
{D}_{\text{usn }}\left( {Q,S}\right)  = \frac{D\left( {\overset{⏜}{{Q}^{l}},\widehat{S}}\right) }{\sqrt{l}},l = \left| S\right| 
$$

Definition 6 scales $Q$ to length $\left| S\right|$ ,while it is also possible to scale $S$ to $\left| Q\right|$ . Both approaches possess similar capabilities in eliminating global scaling. Here we stay consistent with the preceding works [31]. Distance $D$ in ${D}_{usn}$ can be either ${D}_{ed}$ or ${D}_{dtw}$ depending on the concrete scenario. We denote them as ${D}_{usn}^{ed}$ and ${D}_{usn}^{dtw}$ . In this paper, when saying two sequences are similar or have similar patterns,we mean that they have a small distance under ${D}_{usn}$ . That is, when scaling to the same length, the two sequences have a very small ED/DTW.

定义6将$Q$缩放到长度$\left| S\right|$，同时也可以将$S$缩放到$\left| Q\right|$。这两种方法在消除全局缩放方面具有相似的能力。在这里，我们与之前的工作[31]保持一致。${D}_{usn}$中的距离$D$可以是${D}_{ed}$或${D}_{dtw}$，具体取决于具体场景。我们将它们表示为${D}_{usn}^{ed}$和${D}_{usn}^{dtw}$。在本文中，当说两个序列相似或具有相似的模式时，我们指的是它们在${D}_{usn}$下的距离较小。也就是说，当缩放到相同长度时，这两个序列的欧几里得距离（ED）/动态时间规整距离（DTW）非常小。

Problem 1 (Top-K Subsequence Matching within Dusn). Given a time series $T$ ,a length range $\left\lbrack  {{l}_{\min },{l}_{\max }}\right\rbrack$ and an integer $K$ ,for any $Q$ ,the top- $K$ matching is to find a set of subsequences $\mathbb{R} =$ $\left\{  {{S}_{1},{S}_{2},\cdots ,{S}_{K}}\right\}   \subseteq  \mathbb{A}$ ,where $\mathbb{A}$ contains all subsequences of $T$ whose lengths satisfy the length range,such that, $\forall S \in  \mathbb{R}$ and $\forall {S}^{\prime } \in  \mathbb{A} - \mathbb{R}$ , ${D}_{usn}\left( {Q,S}\right)  \leq  {D}_{usn}\left( {Q,{S}^{\prime }}\right)$ .

问题1（Dusn内的前K个子序列匹配）。给定一个时间序列$T$、一个长度范围$\left\lbrack  {{l}_{\min },{l}_{\max }}\right\rbrack$和一个整数$K$，对于任何$Q$，前$K$匹配是要找到一个子序列集合$\mathbb{R} =$ $\left\{  {{S}_{1},{S}_{2},\cdots ,{S}_{K}}\right\}   \subseteq  \mathbb{A}$，其中$\mathbb{A}$包含$T$的所有长度满足该长度范围的子序列，使得$\forall S \in  \mathbb{R}$且$\forall {S}^{\prime } \in  \mathbb{A} - \mathbb{R}$，${D}_{usn}\left( {Q,S}\right)  \leq  {D}_{usn}\left( {Q,{S}^{\prime }}\right)$。

### 2.2 iSAX Index Family

### 2.2 iSAX索引族

Our work preserves the main structure of iSAX index, thus we briefly review the related techniques in this part.

我们的工作保留了iSAX索引的主要结构，因此我们在这部分简要回顾相关技术。

2.2.1 Representation. The representation technique summarizes time series into a lower-dimensional representation to estimate the approximate distance between them efficiently. We review the PAA used by iSAX index [16, 40].

2.2.1 表示方法。表示技术将时间序列总结为低维表示，以有效估计它们之间的近似距离。我们回顾了iSAX索引所使用的分段聚合近似（PAA）方法 [16, 40]。

The Piecewise Aggregate Approximation (PAA) [16, 40] splits a sequence $S$ into disjoint equal-length segments and represents each segment with the mean of its values,which transforms $S$ into a $m$ - dimensional representation ${PAA}\left( S\right)$ ,where $m = \lfloor \frac{\left| S\right| }{\text{ length of segment }}\rfloor$ .

分段聚合近似（Piecewise Aggregate Approximation，PAA） [16, 40] 将序列 $S$ 分割成不相交的等长片段，并以每个片段值的均值来表示该片段，这将 $S$ 转换为 $m$ 维表示 ${PAA}\left( S\right)$，其中 $m = \lfloor \frac{\left| S\right| }{\text{ length of segment }}\rfloor$。

Referring to the proposition $\left\lbrack  {{11},{16}}\right\rbrack$ ,PAA gives a lower bounding distance for the Euclidean distance between two sequences.

参考命题 $\left\lbrack  {{11},{16}}\right\rbrack$，PAA为两个序列之间的欧几里得距离提供了一个下界距离。

Proposition 1 (PAA Lower Bound). Given two time series $Q$ and $S$ such that $\left| Q\right|  = \left| S\right|$ ,we have

命题1（PAA下界）。给定两个时间序列 $Q$ 和 $S$，使得 $\left| Q\right|  = \left| S\right|$，我们有

$$
{D}_{ed}\left( {Q,S}\right)  \geq  \sqrt{{l}_{seg} \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}{\left( PAA{\left( Q\right) }_{i} - PAA{\left( S\right) }_{i}\right) }^{2}}. \tag{1}
$$

where ${l}_{\text{seg }}$ is the length of segment, $m = \left\lfloor  {\left| S\right| /{l}_{\text{seg }}}\right\rfloor$ .

其中 ${l}_{\text{seg }}$ 是片段的长度， $m = \left\lfloor  {\left| S\right| /{l}_{\text{seg }}}\right\rfloor$。

The SAX [24] and iSAX [34] representations are also adopted in iSAX index to to reduce storage space and facilitate index construction. Briefly described, they discretize each coefficient of PAA(S) as a binary string,referred to as ${SAX}\left( S\right)$ and ${iSAX}\left( S\right)$ .

符号聚合近似（SAX） [24] 和改进的符号聚合近似（iSAX） [34] 表示方法也被应用于iSAX索引中，以减少存储空间并便于索引构建。简而言之，它们将PAA(S)的每个系数离散化为一个二进制字符串，分别记为 ${SAX}\left( S\right)$ 和 ${iSAX}\left( S\right)$。

2.2.2 Index Structure. When given a predefined segment length, iSAX index $\left\lbrack  {{34},{38}}\right\rbrack$ follows the same behavior to represent all sequences as PAAs and construct a tree-like index structure on top of them. The index consists of three types of nodes (root node, inner node,and leaf node). The root node has at most ${2}^{m}$ child nodes, while the inner node has only 2 child nodes. Each node has a distinct iSAX representation representing all the sequences in its subtree. Combining the Proposition 1, iSAX index can guarantee no false dismissals when pruning tree nodes during the matching procedure. The index also supports incremental data insertion and dynamic update of tree structure [2].

2.2.2 索引结构。当给定预定义的片段长度时，iSAX索引 $\left\lbrack  {{34},{38}}\right\rbrack$ 采用相同的方式将所有序列表示为PAA，并在此基础上构建一个树状索引结构。该索引由三种类型的节点（根节点、内部节点和叶节点）组成。根节点最多有 ${2}^{m}$ 个子节点，而内部节点只有2个子节点。每个节点都有一个独特的iSAX表示，代表其所有子树中的序列。结合命题1，iSAX索引可以保证在匹配过程中修剪树节点时不会出现漏检情况。该索引还支持增量数据插入和树结构的动态更新 [2]。

### 2.3 Approach Overview

### 2.3 方法概述

Our work systematically tackles this problem from three interconnected aspects: data representation, indexing, and query processing. The framework and basic design approach are illustrated in Figure 2.

我们的工作从三个相互关联的方面系统地解决了这个问题：数据表示、索引和查询处理。该框架和基本设计方法如图2所示。

<!-- Media -->

<!-- figureText: Subsequences Index Building (c2) Enhanced Scanning with (c) Query Processing (c1) Lower Bounding for Tree Structure -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_2.jpg?x=952&y=1498&w=673&h=518&r=0"/>

Figure 2: CIVET Framework

图2：CIVET框架

<!-- Media -->

UPAA Representation. As shown in Figure 2(a), we propose Uniform Piecewise Aggregate Approximation (UPAA), which effectively summarizes sequences of different lengths and provides an exact lower bound of the uniform scaling distance with norm between two sequences for pruning.

统一分段聚合近似（UPAA）表示。如图2(a)所示，我们提出了统一分段聚合近似（Uniform Piecewise Aggregate Approximation，UPAA）方法，该方法能有效总结不同长度的序列，并为两个序列之间具有范数的统一缩放距离提供一个精确的下界，用于剪枝。

Index Construction. We design a compact iSAX-based index to manage massive subsequences based on UPAA, where adjacent subsequences are organized into blocks (e.g., ${B}_{i},{B}_{j}$ in Figure 2(b)), and similar blocks are organized into envelops (e.g., ${E}_{k}$ ). In this way, subsequences are compactly stored in our index and can be efficiently accessed based on similarity.

索引构建。我们基于UPAA设计了一个紧凑的基于iSAX的索引来管理大量子序列，其中相邻的子序列被组织成块（例如，图2(b)中的 ${B}_{i},{B}_{j}$），相似的块被组织成包络（例如， ${E}_{k}$）。通过这种方式，子序列被紧凑地存储在我们的索引中，并且可以基于相似度进行高效访问。

Query Processing. We further design exact and approximate subsequence querying algorithms that can efficiently prune the subtrees based on the lower bound provided by UPAA when traversing the index (see Figure 2(c1)), and can also filter out unpromising subsequences when scanning the subsequences inside a node without any false dismissals (see Figure 2(c2)).

查询处理。我们进一步设计了精确和近似子序列查询算法，这些算法在遍历索引时可以根据UPAA提供的下界有效地修剪子树（见图2(c1)），并且在扫描节点内的子序列时可以过滤掉没有希望的子序列，而不会出现漏检情况（见图2(c2)）。

## 3 EXTENDING PAA FOR GLOBAL SCALING

## 3 扩展PAA以处理全局缩放

We extend PAA as UPAA to handle the global scaling among sequences and present the lower bound properties of UPAA.

我们将PAA扩展为UPAA，以处理序列之间的全局缩放问题，并给出UPAA的下界性质。

### 3.1 Uniform PAA

### 3.1 统一PAA

Since PAA splits both query and database subsequences into equal-length segments, it cannot solve the global scaling phenomenon. As shown in Figure 3(a),although ${S}_{1}$ and ${S}_{2}$ exhibit the same pattern, the equi-length segmentation of PAA fails to capture this similarity.

由于分段聚合近似（PAA）方法将查询子序列和数据库子序列都分割成等长的片段，因此它无法解决全局缩放现象。如图3(a)所示，尽管${S}_{1}$和${S}_{2}$呈现出相同的模式，但PAA的等长分割无法捕捉到这种相似性。

To tackle this problem, we adopt a new segmentation strategy. Instead of fixing the length of each segment, we fix the total number of segments. Formally, we first define the segmentation method and then extend the PAA with this method.

为了解决这个问题，我们采用了一种新的分割策略。我们不固定每个片段的长度，而是固定片段的总数。形式上，我们首先定义分割方法，然后用这种方法扩展PAA。

Definition 7 (Uniform Segmentation (UniSEG)). Given a sequence $S = \left( {{s}_{1},{s}_{2},\cdots ,{s}_{n}}\right)$ and the number of segments $m,S$ is segmented as $m$ parts,denoted as $\operatorname{UniSeg}\left( S\right)  = \left( {{S}_{1},{S}_{2},\cdots ,{S}_{m}}\right)$ . The $i$ -th segment is defined as ${S}_{i} = \left( {{s}_{{r}_{i - 1} + 1},\cdots ,{s}_{{r}_{i}}}\right)$ ,where ${r}_{i} = \left\lfloor  \frac{i \times  n}{m}\right\rfloor$ , for $1 \leq  i \leq  m$ ,and initially, ${r}_{0} = 0$ .

定义7（均匀分割（UniSEG））。给定一个序列$S = \left( {{s}_{1},{s}_{2},\cdots ,{s}_{n}}\right)$，将其分割成$m,S$个片段，记为$m$部分，记为$\operatorname{UniSeg}\left( S\right)  = \left( {{S}_{1},{S}_{2},\cdots ,{S}_{m}}\right)$。第$i$个片段定义为${S}_{i} = \left( {{s}_{{r}_{i - 1} + 1},\cdots ,{s}_{{r}_{i}}}\right)$，其中${r}_{i} = \left\lfloor  \frac{i \times  n}{m}\right\rfloor$，对于$1 \leq  i \leq  m$，并且初始时，${r}_{0} = 0$。

Definition 8 (Uniform PAA (UPAA)). Given a sequence $S =$ $\left( {{s}_{1},{s}_{2},\cdots ,{s}_{n}}\right)$ and the number of segments $m$ ,we compress and represent $\operatorname{UniSeg}\left( S\right)$ as a m-dimension vector,denoted as $\operatorname{UPAA}\left( S\right)  =$ $\left( {{\mu }_{1},{\mu }_{2},\cdots ,{\mu }_{k}}\right)$ ,where ${\mu }_{i}$ is the mean value of ${S}_{i}$ . The i-th coefficient ${UPAA}{\left( S\right) }_{i}$ is denoted as ${\mu }_{i}\left( S\right)$ interchangeably.

定义8（均匀分段聚合近似（UPAA））。给定一个序列$S =$ $\left( {{s}_{1},{s}_{2},\cdots ,{s}_{n}}\right)$和片段数量$m$，我们将$\operatorname{UniSeg}\left( S\right)$压缩并表示为一个m维向量，记为$\operatorname{UPAA}\left( S\right)  =$ $\left( {{\mu }_{1},{\mu }_{2},\cdots ,{\mu }_{k}}\right)$，其中${\mu }_{i}$是${S}_{i}$的平均值。第i个系数${UPAA}{\left( S\right) }_{i}$也可互换地记为${\mu }_{i}\left( S\right)$。

Given a dataset, $m$ is the same for database and query subsequences, despite different lengths. In this way, for similar sequences with global scaling, the corresponding segments after UniSeg will tend to be similar. As demonstrated in Figure 3(b), UPAA (with $m = 3$ ) effectively transforms ${S}_{1}$ and ${S}_{2}$ into similar and same-dimensional representations.

给定一个数据集，尽管数据库子序列和查询子序列的长度不同，但$m$对于它们是相同的。这样，对于具有全局缩放的相似序列，均匀分割（UniSeg）后的相应片段将趋于相似。如图3(b)所示，UPAA（使用$m = 3$）有效地将${S}_{1}$和${S}_{2}$转换为相似且同维度的表示。

Similar to PAA, UPAA also possesses the lower bound property.

与PAA类似，UPAA也具有下界性质。

THEOREM 1 (UPAA Lower BOUND). Given two time series $Q$ and $S$ such that $\left| Q\right|  = \left| S\right|$ ,the number of segments $m$ ,we have that,

定理1（UPAA下界）。给定两个时间序列$Q$和$S$，使得$\left| Q\right|  = \left| S\right|$，片段数量为$m$，我们有：

$$
{D}_{ed}\left( {Q,S}\right)  \geq  \sqrt{\left\lfloor  \frac{\left| S\right| }{m}\right\rfloor   \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}{\left( UPAA{\left( Q\right) }_{i} - UPAA{\left( S\right) }_{i}\right) }^{2}}. \tag{2}
$$

<!-- Media -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_3.jpg?x=1029&y=239&w=515&h=215&r=0"/>

Figure 3: PAA and UPAA. ${S}_{1}$ and ${S}_{2}$ have a small ${D}_{usn}$ (a) but PAA summarize them with values of large differences. (b) UPAA summarize them with closer values.

图3：PAA和UPAA。${S}_{1}$和${S}_{2}$的${D}_{usn}$值较小（a），但PAA用差异较大的值来概括它们。（b）UPAA用更接近的值来概括它们。

<!-- Media -->

Proof. According to the definition of UniSeg, a given sequence might be split into segments of different lengths. But the difference among the lengths of all segments does not exceed one. Let ${l}_{seg} =$ $\lfloor \left| S\right| /m\rfloor$ ,we have that,

证明。根据均匀分割（UniSeg）的定义，给定的序列可能会被分割成不同长度的片段。但所有片段长度之间的差异不超过1。设${l}_{seg} =$ $\lfloor \left| S\right| /m\rfloor$，我们有：

$$
{r}_{i} - {r}_{i - 1} = {l}_{\text{seg }}\text{or}{l}_{\text{seg }} + 1\text{,for}1 \leq  i \leq  m\text{.} \tag{3}
$$

According to the corollary in [40],for the $i$ -th segment,we have,

根据文献[40]中的推论，对于第$i$个片段，我们有：

$$
\left( {{r}_{i} - {r}_{i - 1}}\right)  \cdot  {\left( {\mu }_{i}\left( Q\right)  - {\mu }_{i}\left( S\right) \right) }^{2} \leq  \mathop{\sum }\limits_{{j = {r}_{i - 1} + 1}}^{{r}_{i}}{\left( {q}_{j} - {s}_{j}\right) }^{2} \tag{4}
$$

Now, we can easily prove the correctness by scaling the polynomial coefficient as follows,

现在，我们可以通过如下缩放多项式系数轻松证明其正确性：

$$
{D}_{ed}\left( {Q,S}\right)  \geq  \sqrt{\mathop{\sum }\limits_{{i = 1}}^{m}\left( {{r}_{i} - {r}_{i - 1}}\right)  \cdot  {\left( {\mu }_{i}\left( Q\right)  - {\mu }_{i}\left( S\right) \right) }^{2}} \tag{5}
$$

$$
 \geq  \sqrt{\left\lfloor  \frac{\left| S\right| }{m}\right\rfloor   \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}{\left( {\mu }_{i}\left( Q\right)  - {\mu }_{i}\left( S\right) \right) }^{2}}.
$$

Note that when the sequence length is divisible by $m$ ,the formula in Theorem 1 and the formula in Proposition 1 have the same meaning. Consequently, UPAA enhances PAA with capabilities to represent and align variable-length sequences without losing the original properties of PAA.

注意，当序列长度能被$m$整除时，定理1中的公式和命题1中的公式具有相同的含义。因此，UPAA在不损失PAA原有性质的前提下，增强了PAA表示和对齐可变长度序列的能力。

### 3.2 Lower Bound for a Set of Time Series

### 3.2 一组时间序列的下界

Usually, we need to estimate the distance between a query sequence and a set of variable-length subsequences. So, we infer the lower bounding distance for this situation. First, we scale the query to all possible lengths and use two vectors to delimit minimal and maximal UPAA representations of the scaled query sequences. Formally, for a query sequence $Q$ and a set of sequences $\mathbb{S}$ ,and the number of segments $m$ ,we denote the minimal and maximal UPAAs as ${L}^{Q}$ and ${U}^{Q}$ ,respectively,such that for $1 \leq  i \leq  m$ .

通常，我们需要估计一个查询序列与一组可变长度子序列之间的距离。因此，我们推导出了这种情况下的下界距离。首先，我们将查询序列缩放到所有可能的长度，并使用两个向量来界定缩放后查询序列的最小和最大UPAA（均匀分段聚合近似，Uniform Piecewise Aggregate Approximation）表示。形式上，对于一个查询序列$Q$和一组序列$\mathbb{S}$，以及分段数量$m$，我们分别将最小和最大UPAA表示为${L}^{Q}$和${U}^{Q}$，使得对于$1 \leq  i \leq  m$成立。

$$
{L}_{i}^{Q} = \min \left( \left\{  {{\mu }_{i}\left( {Q}^{\left| S\right| }\right) }\right\}  \right) ,{U}_{i}^{Q} = \max \left( \left\{  {{\mu }_{i}\left( {Q}^{\left| S\right| }\right) }\right\}  \right) ,\forall S \in  \mathbb{S}, \tag{6}
$$

where ${Q}^{\left| S\right| }$ means scaling the query $Q$ to the length of sequence $S$ using uniform scaling.

其中${Q}^{\left| S\right| }$表示使用均匀缩放将查询序列$Q$缩放到序列$S$的长度。

Similarly,for the sequences in $\mathbb{S}$ ,we can use two vectors, ${L}^{\mathbb{S}}$ and ${U}^{\mathbb{S}}$ ,to enclose the minimal and maximal UPAA coefficients of all the sequences in the set $\mathbb{S}$ ,

类似地，对于$\mathbb{S}$中的序列，我们可以使用两个向量${L}^{\mathbb{S}}$和${U}^{\mathbb{S}}$来包含集合$\mathbb{S}$中所有序列的最小和最大UPAA系数。

$$
{L}_{i}^{\mathbb{S}} = \min \left( {{\mu }_{i}\left( S\right) }\right) ,{U}_{i}^{\mathbb{S}} = \max \left( {{\mu }_{i}\left( S\right) }\right) ,\forall S \in  \mathbb{S}. \tag{7}
$$

Now we have the lower bound for a set of time series.

现在我们得到了一组时间序列的下界。

<!-- Media -->

<!-- figureText: (a) Grouping Adjacent Subsequences ${B}_{1}{Blk}\left( {s + W,{l}_{min} + H}\right)$ $\operatorname{SAX}\left( {U}^{{B}_{1}}\right)  : \{ {01},{10},{01}\}$ ${iSAX}\left( {U}^{N}\right)  = \{ 0,{10},0\}$ ${iSAX}\left( {L}^{N}\right)  = \{ {0.10.0}\}$ ${B}_{2}\;{Blk}\left( {s + {2W},{l}_{min}}\right)$ $\operatorname{SAX}\left( {U}^{{B}_{2}}\right)  : \{ {10},{11},{01}\}$ $\operatorname{SAX}\left( {L}^{{B}_{2}}\right)  : \{ {01},{10},{01}\}$ ${iSAX}\left( {U}^{N}\right)  = \{ 1,{11},0\}$ $\operatorname{iSAX}\left( {L}^{N}\right)  = \{ 0,{10},0\}$ ${E}_{i}{SAX}\left( {U}^{{E}_{i}}\right)  = \{ {10},{11},{01}\}$ ${SAX}\left( {L}^{{E}_{i}}\right)  = \{ {00},{10},{00}\}$ ${SAX}\left( {L}^{{E}_{i}}\right) {SAX}\left( {U}^{{E}_{i}}\right)$ (b) Grouping Similar Subsequences -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_4.jpg?x=161&y=241&w=698&h=356&r=0"/>

Figure 4: CIVET Index Construction

图4：CIVET索引构建

<!-- Media -->

THEOREM 2 (UPAA LOWER BOUND ON SET). Given a sequence $Q$ and a set of time series $\mathbb{S}$ ,the number of segments $m$ ,we have that,

定理2（集合上的UPAA下界）。给定一个序列$Q$和一组时间序列$\mathbb{S}$，分段数量为$m$，我们有：

$$
\mathop{\min }\limits_{{S \in  \mathbb{S}}}\left\{  {{D}_{ed}\left( {{Q}^{\left| S\right| },S}\right) }\right\}   \geq  \sqrt{\left\lfloor  \frac{{l}_{min}}{m}\right\rfloor   \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}\left\{  \begin{array}{l} {\left( {L}_{i}^{\mathbb{S}} - {U}_{i}^{Q}\right) }^{2},\text{ if }{L}_{i}^{\mathbb{S}} > {U}_{i}^{Q} \\  {\left( {U}_{i}^{\mathbb{S}} - {L}_{i}^{Q}\right) }^{2},\text{ if }{U}_{i}^{\mathbb{S}} < {L}_{i}^{Q} \\  0\;,\text{ otherwise } \end{array}\right. }
$$

(8)

where ${l}_{\min }$ is the minimal length of sequences in $\mathbb{S}$ .

其中${l}_{\min }$是$\mathbb{S}$中序列的最小长度。

Proof. Without loss of generality, we consider a random sequence $S$ in $\mathbb{S}$ here. Before calculating the distance, $Q$ is scaled to the length of $S$ . And it is easy to know that $\left| S\right|  \geq  {l}_{\min }$ . So,combining with the Theorem 1, we have that,

证明。不失一般性，我们在此考虑$\mathbb{S}$中的一个随机序列$S$。在计算距离之前，将$Q$缩放到$S$的长度。并且很容易知道$\left| S\right|  \geq  {l}_{\min }$。因此，结合定理1，我们有：

$$
{D}_{ed}\left( {{Q}^{\left| S\right| },S}\right)  \geq  \sqrt{\left\lfloor  \frac{{l}_{\min }}{m}\right\rfloor   \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}{\left( {\mu }_{i}\left( {Q}^{\left| S\right| }\right)  - {\mu }_{i}\left( S\right) \right) }^{2}}. \tag{9}
$$

According to the definition of $L$ and $U$ for the query and the set of sequences, we easily prove the correctness of Equation 8.

根据查询序列和序列集合的$L$和$U$的定义，我们很容易证明等式8的正确性。

Therefore, UPAA enables the iSAX family index to obtain the capability to process variable-length data effectively while retaining the ability of index pruning without false dismissals.

因此，UPAA使iSAX族索引能够在保留无错误排除的索引剪枝能力的同时，有效处理可变长度数据。

## 4 INDEX CONSTRUCTION

## 4 索引构建

In this section, we present our compact index. We provide two techniques named block summarization (Section 4.1) and envelope summarization (Section 4.2) to compact the redundant information of subsequence. Then, we describe the procedure of building CIVET index (Section 4.3).

在本节中，我们介绍我们的紧凑索引。我们提供了两种技术，分别是块汇总（第4.1节）和包络汇总（第4.2节），以压缩子序列的冗余信息。然后，我们描述构建CIVET索引的过程（第4.3节）。

### 4.1 Grouping Adjacent Subsequences

### 4.1 相邻子序列分组

In this part, we provide a representation method to summarize sets of overlapping subsequences succinctly.

在这部分，我们提供一种表示方法来简洁地汇总一组重叠的子序列。

A specific subsequence is determined by its start position and length. Considering subsequences as points on a two-dimensional coordinate, all the subsequences of a long time series form a two-dimensional space. we depict it as the space of subsequences in Figure 4(a). Thus, subsequences can be divided into small rectangles with width $W$ and height $H$ . The two user-defined parameters, $W$ and $H$ ,represent the stepsizes of start position and length,respectively. We refer to the rectangle as Block (Blk). Since there are many overlaps among the subsequences in the same block, their UPAAs tend to be similar. We summarize them with a higher-level representation.

一个特定的子序列由其起始位置和长度确定。将子序列视为二维坐标上的点，一个长时序列的所有子序列构成一个二维空间。我们在图4(a)中将其描述为子序列空间。因此，子序列可以被划分为宽度为$W$、高度为$H$的小矩形。两个用户定义的参数$W$和$H$分别表示起始位置和长度的步长。我们将该矩形称为块（Blk）。由于同一块中的子序列之间存在许多重叠，它们的UPAA往往相似。我们用更高级别的表示来汇总它们。

Definition 9 (Block). Given two parameters $W$ and $H$ ,a block ${Blk}\left( {s,l}\right)$ groups a set of adjacent subsequences in the long sequence $S$ and delimits the UPAA coefficients of these subsequences with two m-dimension vectors, ${L}^{B}$ and ${U}^{B}$ . The set of subsequences in ${Blk}\left( {s,l}\right)$ is defined as follows,

定义9（块）。给定两个参数$W$和$H$，一个块${Blk}\left( {s,l}\right)$将长序列$S$中的一组相邻子序列分组，并使用两个m维向量${L}^{B}$和${U}^{B}$来界定这些子序列的UPAA系数。${Blk}\left( {s,l}\right)$中的子序列集合定义如下：

$$
{\mathbb{S}}^{B} = \left\{  {{S}_{i,{l}^{\prime }} \mid  s \leq  i < s + W\text{ and }l \leq  {l}^{\prime } < l + H}\right\}  . \tag{10}
$$

And two vectors, ${L}^{B}$ and ${U}^{B}$ ,satisfy that,

并且两个向量，${L}^{B}$ 和 ${U}^{B}$ ，满足以下条件：

$$
{L}_{i}^{B} = \min \left( {{\mu }_{i}\left( {S}^{\prime }\right) }\right) ,{U}_{i}^{B} = \max \left( {{\mu }_{i}\left( {S}^{\prime }\right) }\right) ,\forall {S}^{\prime } \in  {\mathbb{S}}^{B}\text{.} \tag{11}
$$

Specific boundary corner cases are not explicitly mentioned in Equation 10 for brevity. But note that the subsequences in the set ${\mathbb{S}}^{B}$ must adhere to the sequence length constraints $\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$ .

为简洁起见，方程10中未明确提及特定的边界角点情况。但请注意，集合 ${\mathbb{S}}^{B}$ 中的子序列必须遵守序列长度约束 $\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$ 。

Now, we can summarize all subsequences within the space of subsequences using continuous but non-overlapping blocks, as illustrated in Figure 4(a). Each block effectively summarizes the information of the corresponding adjacent sequences using only two low-dimensional vectors and a pointer that references the raw data. We depict two concrete examples of Blocks, ${B}_{1}$ and ${B}_{2}$ .

现在，我们可以使用连续但不重叠的块来总结子序列空间内的所有子序列，如图4(a)所示。每个块仅使用两个低维向量和一个引用原始数据的指针，有效地总结了相应相邻序列的信息。我们描绘了两个具体的块示例，${B}_{1}$ 和 ${B}_{2}$ 。

Block summarization allows users to adjust $W$ and $H$ to trade space and time. However, we can only summarize the subsequences with close positions and lengths. The similar subsequences may not be arranged rectangularly in the space of subsequences. Intuitively, in Figure 4, we use similar colors to mark the Blocks with similar representations. For example,the blocks ${B}_{1}$ and ${B}_{2}$ are similar but not adjacent. So, we propose a new method to rearrange and pack up Blocks with similar features to improve the compactness of the index further.

块总结允许用户调整 $W$ 和 $H$ 以权衡空间和时间。然而，我们只能总结位置和长度相近的子序列。相似的子序列在子序列空间中可能不会呈矩形排列。直观地说，在图4中，我们使用相似的颜色来标记具有相似表示的块。例如，块 ${B}_{1}$ 和 ${B}_{2}$ 相似但不相邻。因此，我们提出了一种新方法，用于重新排列和打包具有相似特征的块，以进一步提高索引的紧凑性。

### 4.2 Grouping Similar Blocks

### 4.2 对相似块进行分组

Block summarizes the information of adjacent subsequences. This part focuses on the blocks with similar UPAA representations but different positions. We call the grouped blocks an Envelope. We refine the sortable representation invSAX [20] for block summarization and then introduce the procedure to group blocks.

块总结了相邻子序列的信息。这部分重点关注具有相似UPAA表示但位置不同的块。我们将分组后的块称为包络（Envelope）。我们改进了可排序表示invSAX [20] 用于块总结，然后介绍对块进行分组的过程。

4.2.1 InvSAX for block summarization. The invSAX provides the ability to convert the SAX representation of a time series into a sortable representation [20]. Sorted by invSAX, time series with similar SAX will be placed in close positions. So we utilize the invSAX to rearrange blocks in our work.

4.2.1 用于块总结的invSAX。invSAX能够将时间序列的SAX表示转换为可排序表示 [20] 。按invSAX排序后，具有相似SAX的时间序列将被放置在相邻位置。因此，我们在工作中利用invSAX来重新排列块。

Given a time series $S,{SAX}\left( S\right)$ represents $S$ with a length- $m$ vector,whose $i$ -th value ${SA}{X}_{i}\left( S\right)$ is a binary number. The bits in the binary number indicate the possible value range of ${PA}{A}_{i}\left( S\right)$ , and the higher bit has more impact on this range. The key idea of invSAX is to sort the time series according to the more important bits, i.e., the higher bits have higher sorting priority. For example, the invSAX representation of SAX(011,101,001)is represented as '010100111',

给定一个时间序列 $S,{SAX}\left( S\right)$ 用一个长度为 $m$ 的向量表示 $S$ ，其第 $i$ 个值 ${SA}{X}_{i}\left( S\right)$ 是一个二进制数。二进制数中的位表示 ${PA}{A}_{i}\left( S\right)$ 的可能取值范围，且高位对该范围的影响更大。invSAX的关键思想是根据更重要的位对时间序列进行排序，即高位具有更高的排序优先级。例如，SAX(011,101,001)的invSAX表示为'010100111'。

Here, we refine the invSAX for block summarization. As introduced above,block summarization delimits a block ${Blk}$ with two vectors ${L}^{B}$ and ${U}^{B}$ . We can merge these two vectors into one named $L{U}^{B}$ as follows,

在这里，我们改进了invSAX用于块总结。如上所述，块总结用两个向量 ${L}^{B}$ 和 ${U}^{B}$ 界定一个块 ${Blk}$ 。我们可以将这两个向量合并为一个名为 $L{U}^{B}$ 的向量，如下所示：

$$
L{U}^{B}\left( {Blk}\right)  = \left( {{L}_{1}^{B},{U}_{1}^{B},\cdots ,{L}_{m}^{B},{U}_{m}^{B}}\right) , \tag{12}
$$

where $m$ is the number of segments.

其中 $m$ 是段的数量。

Now, the invSAX can be easily applied to block summarization straightforwardly. We transform the $L{U}^{B}\left( {Blk}\right)$ into SAX representation and then convert the SAX into sortable summarization using the same logic of invSAX. We refer to this sortable summarization as inv ${SA}{X}^{B}$ for simplicity. For instance, ${L}^{{B}_{1}}$ and ${U}^{{B}_{1}}$ in Figure 4(a) are respectively given by(00,10,00)and $\left( {\underline{\mathbf{{01}}},\underline{\mathbf{{10}}},\underline{01}}\right)$ . Consequently, the invSA ${X}^{B}$ representation for ${B}_{1}$ would be $\mathbf{0}\underline{\mathbf{0}}\underline{\mathbf{1}}\underline{\mathbf{1}}0\underline{\mathbf{0}}\underline{\mathbf{1}}0\underline{\mathbf{0}}0\underline{\mathbf{1}}$ .

现在，invSAX可以直接轻松地应用于块总结。我们将 $L{U}^{B}\left( {Blk}\right)$ 转换为SAX表示，然后使用invSAX的相同逻辑将SAX转换为可排序总结。为简单起见，我们将这种可排序总结称为inv ${SA}{X}^{B}$ 。例如，图4(a)中的 ${L}^{{B}_{1}}$ 和 ${U}^{{B}_{1}}$ 分别由(00,10,00)和 $\left( {\underline{\mathbf{{01}}},\underline{\mathbf{{10}}},\underline{01}}\right)$ 给出。因此， ${B}_{1}$ 的invSA ${X}^{B}$ 表示将是 $\mathbf{0}\underline{\mathbf{0}}\underline{\mathbf{1}}\underline{\mathbf{1}}0\underline{\mathbf{0}}\underline{\mathbf{1}}0\underline{\mathbf{0}}0\underline{\mathbf{1}}$ 。

#### 4.2.2 Envelope construction. In this part, we further compact the blocks with similar UPAA representations facilitated by invSA ${X}^{B}$ .

#### 4.2.2 包络（Envelope）构建。在这部分，我们借助invSA ${X}^{B}$ 进一步压缩具有相似UPAA表示的块。

Given the blocks constructed from a long time series, we append them into an array and sort the array by invSA ${X}^{B}$ . Thus,the blocks with similar ${L}^{B}$ and ${U}^{B}$ tend to be placed in close positions of the array. We use an envelope to summarize the blocks in a sliding window.

给定由长时间序列构建的块，我们将它们追加到一个数组中，并按逆后缀数组（invSA ${X}^{B}$）对该数组进行排序。因此，具有相似 ${L}^{B}$ 和 ${U}^{B}$ 的块往往会被放置在数组的相邻位置。我们使用一个包络（envelope）来总结滑动窗口中的块。

DEFINITION 10 (ENVELOPE). Given a length-n array of blocks sorted by inv ${SA}{X}^{B}\left( {{B}_{1},{B}_{2},\cdots ,{B}_{n}}\right)$ and window size ws,we group every ws blocks as an envelope and delimit their UPAAs with two $m$ - dimensional vectors, ${L}^{E}$ and ${U}^{E}$ . Formally,the set of blocks in the i-th envelope is defined as ${\mathbb{S}}^{{E}_{i}} = \left\{  {{B}_{j} \mid  \left( {i - 1}\right)  * {ws} < j \leq  \min \left( {n,i * {ws}}\right) }\right\}$ . And two vectors ${L}^{E}$ and ${U}^{E}$ of ${E}_{i}$ satisfy that,

定义 10（包络（ENVELOPE））。给定一个按逆 ${SA}{X}^{B}\left( {{B}_{1},{B}_{2},\cdots ,{B}_{n}}\right)$ 排序的长度为 n 的块数组和窗口大小 ws，我们将每 ws 个块归为一个包络，并使用两个 $m$ 维向量 ${L}^{E}$ 和 ${U}^{E}$ 来界定它们的 UPAA。形式上，第 i 个包络中的块集定义为 ${\mathbb{S}}^{{E}_{i}} = \left\{  {{B}_{j} \mid  \left( {i - 1}\right)  * {ws} < j \leq  \min \left( {n,i * {ws}}\right) }\right\}$。并且 ${E}_{i}$ 的两个向量 ${L}^{E}$ 和 ${U}^{E}$ 满足：

$$
{L}_{i}^{E} = \min \left( {L}_{i}^{B}\right) ,{U}_{i}^{E} = \max \left( {U}_{i}^{B}\right) ,\forall B \in  {\mathbb{S}}^{{E}_{i}}. \tag{13}
$$

A concrete example is depicted in Figure 4(b). We set the size of the sliding window to 4 . To aid in understanding, we provide a more intuitive example in Figure 2(b). When grouping adjacent subsequences, the construction algorithm collects subsequences with similar lengths and starting positions into the same block, as the figure illustrates by ${B}_{i}$ and ${B}_{j}$ . When grouping similar blocks, ${B}_{i}$ and ${B}_{j}$ ,which have similar UPAA features,are sorted into nearby positions for constructing the envelope.

图 4(b) 展示了一个具体示例。我们将滑动窗口的大小设置为 4。为了便于理解，我们在图 2(b) 中提供了一个更直观的示例。在对相邻子序列进行分组时，构建算法将长度和起始位置相似的子序列收集到同一个块中，如图中 ${B}_{i}$ 和 ${B}_{j}$ 所示。在对相似块进行分组时，具有相似 UPAA 特征的 ${B}_{i}$ 和 ${B}_{j}$ 被排序到相邻位置以构建包络。

So far, we compact the subsequences with similar UPAAs in an Env with the help of the summarization methods above. For one envelope,we only need to store two vectors, ${L}^{E}$ and ${U}^{E}$ ,and the pointers of blocks in the envelope. Combining these grouping methods, we can construct a compact and efficient index to support subsequence matching.

到目前为止，借助上述总结方法，我们将具有相似 UPAA 的子序列压缩到一个包络（Env）中。对于一个包络，我们只需要存储两个向量 ${L}^{E}$ 和 ${U}^{E}$ 以及包络中块的指针。结合这些分组方法，我们可以构建一个紧凑且高效的索引来支持子序列匹配。

### 4.3 Index Building

### 4.3 索引构建

Here in this part, we tend to present the procedure to index the envelopes based on iSAX index [34], called CIVET index.

在这部分，我们将介绍基于 iSAX 索引 [34] 对包络进行索引的过程，称为 CIVET 索引。

Similar to ULISSE [25], we maintain minimal and maximal iSAX symbols in each node of CIVET index,denoted by ${iSAX}\left( {L}^{N}\right)$ and ${iSAX}\left( {U}^{N}\right)$ . Besides that,in the leaf node,we additionally store the SAX representations of two vectors, ${L}^{E}$ and ${U}^{E}$ ,and pointers to the blocks. CIVET does not contain real data since subsequences overlap extensively.

与 ULISSE [25] 类似，我们在 CIVET 索引的每个节点中维护最小和最大的 iSAX 符号，分别用 ${iSAX}\left( {L}^{N}\right)$ 和 ${iSAX}\left( {U}^{N}\right)$ 表示。此外，在叶节点中，我们额外存储两个向量 ${L}^{E}$ 和 ${U}^{E}$ 的 SAX 表示以及指向块的指针。由于子序列广泛重叠，CIVET 不包含真实数据。

Before constructing the index structure, we first build the envelopes for subsequences as described in Sections 4.1 and 4.2. Then, we insert these envelopes into CIVET index according to the ${SAX}\left( {L}^{E}\right)$ one by one. That is,finding the target leaf node whose ${iSAX}\left( {L}^{N}\right)$ contains ${SAX}\left( {L}^{E}\right)$ . Then we update the nodes in the route from the root node to the leaf node with respect to ${iSAX}\left( {U}^{N}\right)$ .

在构建索引结构之前，我们首先按照 4.1 节和 4.2 节的描述为子序列构建包络。然后，我们根据 ${SAX}\left( {L}^{E}\right)$ 逐个将这些包络插入到 CIVET 索引中。即，找到其 ${iSAX}\left( {L}^{N}\right)$ 包含 ${SAX}\left( {L}^{E}\right)$ 的目标叶节点。然后，我们根据 ${iSAX}\left( {U}^{N}\right)$ 更新从根节点到叶节点路径上的节点。

Figure 4(c) shows a concrete example of envelope inserting. The ${E}_{i}$ is inserted according to the ${SAX}\left( {L}^{{E}_{i}}\right)$ . Notably,the ${iSAX}\left( {U}^{N}\right)$ is also updated to ensure the property of the node representation vectors. Besides,we also depict the pointers of blocks ${B}_{1}$ and ${B}_{2}$ .

图 4(c) 展示了一个包络插入的具体示例。根据 ${SAX}\left( {L}^{{E}_{i}}\right)$ 插入 ${E}_{i}$。值得注意的是，还更新了 ${iSAX}\left( {U}^{N}\right)$ 以确保节点表示向量的属性。此外，我们还描绘了块 ${B}_{1}$ 和 ${B}_{2}$ 的指针。

When building the CIVET index, we adopt the efficient algorithm for block construction [25] and then sort blocks to construct envelopes. However, it will consume a lot of memory to sort all the blocks of long time series. So we utilize the buffer mechanism. We sequentially load part of the raw time series into a fixed-size buffer and then conduct blocks and envelopes in bulk.

在构建 CIVET 索引时，我们采用高效的块构建算法 [25]，然后对块进行排序以构建包络。然而，对长时间序列的所有块进行排序会消耗大量内存。因此，我们利用缓冲机制。我们将原始时间序列的一部分依次加载到一个固定大小的缓冲区中，然后批量处理块和包络。

Complexity Analysis. For convenience,we let $M$ be the value ${l}_{\max } - {l}_{\min },m$ be the segment number, $k$ be the number of blocks in one envelope and $N$ is the length of long time series. Before utilizing any grouping technique, the magnitude of subsequences is $O\left( {MN}\right)$ . while the CIVET shrinks the space complexity to $O\left( \frac{bMN}{kWH}\right)$ , where $b$ means the bytes of each envelope. The time complexity of block construction is $O\left( \frac{{M}^{2}{Nm}}{W}\right)$ . The time complexity of envelope construction is $O\left( {\frac{MN}{WH}\lg \left( \frac{MN}{WH}\right) }\right)$ .

复杂度分析。为方便起见，我们设$M$为值，${l}_{\max } - {l}_{\min },m$为分段数量，$k$为一个包络中的块数，$N$为长时间序列的长度。在使用任何分组技术之前，子序列的数量级为$O\left( {MN}\right)$。而CIVET（压缩区间值时间序列）将空间复杂度缩小至$O\left( \frac{bMN}{kWH}\right)$，其中$b$表示每个包络的字节数。块构建的时间复杂度为$O\left( \frac{{M}^{2}{Nm}}{W}\right)$。包络构建的时间复杂度为$O\left( {\frac{MN}{WH}\lg \left( \frac{MN}{WH}\right) }\right)$。

## 5 QUERY PROCESSING

## 5 查询处理

This section provides the matching algorithm, including the lower bounding distance, details on matching algorithms, optimization of scanning to reduce unnecessary distance calculations, and extends these techniques to support DTW.

本节提供匹配算法，包括下界距离、匹配算法的详细信息、减少不必要距离计算的扫描优化，并将这些技术扩展以支持动态时间规整（DTW）。

### 5.1 Lower Bounding for Envelope and Node

### 5.1 包络和节点的下界

In this part, we propose a lower bounding distance between the query and envelope to prune the candidate envelopes and tree nodes during the exact matching.

在这部分，我们提出查询与包络之间的下界距离，以便在精确匹配过程中修剪候选包络和树节点。

First of all, similar to envelope summarization, we use two vectors to summarize the information of a given query. Specifically, We scale the query to all possible lengths in the range $\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$ , calculate the UPAAs, and delimit them as lower and upper bounds to represent the query. Formally,given $m$ as the number of segment and a query $Q$ ,the lower and upper bounds of $Q$ are denoted as ${L}^{Q}$ and ${U}^{Q}$ respectively,such that,

首先，与包络汇总类似，我们使用两个向量来汇总给定查询的信息。具体来说，我们将查询缩放到范围$\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$内的所有可能长度，计算统一分段聚合近似（UPAA），并将其界定为下界和上界以表示查询。形式上，给定$m$作为分段数量和一个查询$Q$，$Q$的下界和上界分别表示为${L}^{Q}$和${U}^{Q}$，使得

$$
{L}_{i}^{Q} = \min \left( \left\{  {{\mu }_{i}\left( \widehat{{Q}^{l}}\right) }\right\}  \right) ,{U}_{i}^{Q} = \max \left( \left\{  {{\mu }_{i}\left( \widehat{{Q}^{l}}\right) }\right\}  \right) , \tag{14}
$$

where $1 \leq  i \leq  m$ and $l \in  \left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$ .

其中$1 \leq  i \leq  m$和$l \in  \left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$。

The distance between ${PAA}$ and SAX is provided for the lower bound of the Euclidean distance [34]. Similarly, we refer to the lower and upper breakpoints of SAX value as ${\beta }_{L}\left( \cdot \right)$ and ${\beta }_{U}\left( \cdot \right)$ . Given a query $Q$ with bounds ${L}^{Q}$ and ${U}^{Q}$ and an envelope $E$ with bounds ${L}^{E}$ and ${U}^{E}$ ,we define a lower bounding distance between them as,

为欧几里得距离的下界提供了${PAA}$与符号聚合近似（SAX）之间的距离[34]。类似地，我们将SAX值的下断点和上断点称为${\beta }_{L}\left( \cdot \right)$和${\beta }_{U}\left( \cdot \right)$。给定一个具有边界${L}^{Q}$和${U}^{Q}$的查询$Q$以及一个具有边界${L}^{E}$和${U}^{E}$的包络$E$，我们将它们之间的下界距离定义为

$$
L{B}_{env}\left( {Q,E}\right)  =  \tag{15}
$$

$$
\sqrt{\frac{\eta }{m}}\sqrt{\mathop{\sum }\limits_{{i = 1}}^{m}\left\{  \begin{matrix} {\left( {\beta }_{L}\left( SAX{\left( {L}^{E}\right) }_{i}\right)  - {U}_{i}^{Q}\right) }^{2},\text{ if }{\beta }_{L}\left( {{SAX}{\left( {L}^{E}\right) }_{i}}\right)  > {U}_{i}^{Q} \\  {\left( {\beta }_{U}\left( SAX{\left( {U}^{E}\right) }_{i}\right)  - {L}_{i}^{Q}\right) }^{2},\text{ if }{\beta }_{U}\left( {{SAX}{\left( {U}^{E}\right) }_{i}}\right)  < {L}_{i}^{Q} \\  0\;,\text{ otherwise } \end{matrix}\right. }
$$

As mentioned earlier, UniSeg may produce segments of different lengths. To eliminate the influence of this phenomenon and ensure the correctness of the lower bounding distance, we import the scaling factor $\eta$ ,such that,

如前所述，统一分段（UniSeg）可能会产生不同长度的分段。为消除这种现象的影响并确保下界距离的正确性，我们引入缩放因子$\eta$，使得

$$
\eta  = \frac{{l}^{\prime }}{{l}^{\prime } + 1}\text{,where}{l}^{\prime } = \left\lfloor  \frac{l}{m}\right\rfloor   \tag{16}
$$

$l$ refers to the minimal length among all subsequences in this envelope,and ${l}^{\prime }$ indicates the minimal length of segments.

$l$ 指的是该包络中所有子序列的最小长度，${l}^{\prime }$ 表示线段的最小长度。

THEOREM 3. Given an envelope $E$ and a query $Q$ ,for any subsequence $S$ in the envelope,we have that,

定理 3。给定一个包络 $E$ 和一个查询 $Q$，对于包络中的任何子序列 $S$，我们有：

$$
L{B}_{env}\left( {Q,E}\right)  \leq  {D}_{usn}^{ed}\left( {Q,S}\right) . \tag{17}
$$

Proof. According to the definition of PAA and SAX in [34], we know that ${\beta }_{L}\left( {{SAX}{\left( {L}^{E}\right) }_{i}}\right)  \leq  {L}_{i}^{E} \leq  {U}_{i}^{E} \leq  {\beta }_{U}\left( {{SAX}{\left( {U}^{E}\right) }_{i}}\right)$ ,where ${\beta }_{L}$ and ${\beta }_{U}$ are the lower and higher breakpoint of iSAX [34]. We have,

证明。根据文献 [34] 中 PAA（分段聚合近似，Piecewise Aggregate Approximation）和 SAX（符号聚合近似，Symbolic Aggregate approXimation）的定义，我们知道 ${\beta }_{L}\left( {{SAX}{\left( {L}^{E}\right) }_{i}}\right)  \leq  {L}_{i}^{E} \leq  {U}_{i}^{E} \leq  {\beta }_{U}\left( {{SAX}{\left( {U}^{E}\right) }_{i}}\right)$，其中 ${\beta }_{L}$ 和 ${\beta }_{U}$ 是 iSAX（改进的符号聚合近似，improved Symbolic Aggregate approXimation）[34] 的下限和上限断点。我们有：

$$
L{B}_{\text{env }}\left( {Q,E}\right)  \leq  \sqrt{\frac{\eta }{m}}\sqrt{\mathop{\sum }\limits_{{i = 1}}^{m}\left\{  \begin{array}{l} {\left( {L}_{i}^{E} - {U}_{i}^{Q}\right) }^{2},\text{ if }{L}_{i}^{E} > {U}_{i}^{Q} \\  {\left( {U}_{i}^{E} - {L}_{i}^{Q}\right) }^{2},\text{ if }{U}_{i}^{E} < {L}_{i}^{Q} \\  0\;,\text{ otherwise } \end{array}\right. } \tag{18}
$$

Then we consider the scaling factor $\eta$ . According to the monotonicity of the function $f\left( x\right)  = 1/\left( {x + 1}\right)$ ,combining with $0 <$ ${l}_{\min } < \left| S\right|$ ,we have,

然后我们考虑缩放因子 $\eta$。根据函数 $f\left( x\right)  = 1/\left( {x + 1}\right)$ 的单调性，结合 $0 <$ ${l}_{\min } < \left| S\right|$，我们有：

$$
\eta  = \frac{{l}^{\prime }}{{l}^{\prime } + 1} = \frac{\left\lfloor  \frac{{l}_{\min }}{m}\right\rfloor  }{\left\lfloor  \frac{{l}_{\min }}{m}\right\rfloor   + 1} \leq  \frac{\left\lfloor  \frac{\left| S\right| }{m}\right\rfloor  }{\left\lfloor  \frac{\left| S\right| }{m}\right\rfloor   + 1}. \tag{19}
$$

Besides, it is easy to know,

此外，很容易知道：

$$
\left| S\right| /m \leq  \lfloor \left| S\right| /m\rfloor  + 1 \tag{20}
$$

From Equation 19 and Equation 20 we infer that,

从方程 19 和方程 20 我们可以推断出：

$$
\frac{\eta }{m} \leq  \frac{\left\lfloor  \frac{\left| S\right| }{m}\right\rfloor  }{\left| S\right| } \tag{21}
$$

Now, according to Equation 8, Equation 18 can be derived as,

现在，根据方程 8，方程 18 可以推导为：

$$
L{B}_{\text{env }}\left( {Q,E}\right)  \leq  \sqrt{\frac{1}{\left| S\right| }}\sqrt{\left\lbrack  \frac{\left| S\right| }{m}\right\rbrack   \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}\left\{  \begin{array}{l} {\left( {L}_{i}^{E} - {U}_{i}^{Q}\right) }^{2},\text{ if }{L}_{i}^{E} > {U}_{i}^{Q} \\  {\left( {U}_{i}^{E} - {L}_{i}^{Q}\right) }^{2},\text{ if }{U}_{i}^{E} < {L}_{i}^{Q} \\  0\;,\text{ otherwise } \end{array}\right. }
$$

$$
 \leq  \sqrt{\frac{1}{\left| S\right| }} \cdot  {D}_{ed}\left( {\widehat{{Q}^{\left| S\right| }},\widehat{S}}\right)  = {D}_{usn}^{ed}\left( {Q,S}\right) 
$$

Moreover,we apply the same logic of $L{B}_{env}\left( {Q,E}\right)$ to lower bound the distance between query $Q$ and node $N$ in CIVET. We only need to replace the envelope's SAX representation with the node's iSAX representation. Formally,given a query $Q$ with bounds ${L}^{Q}$ and ${U}^{Q}$ and a node $N$ with bounds ${L}^{N}$ and ${U}^{N}$ ,we define a lower bounding distance between them as,

此外，我们应用与 $L{B}_{env}\left( {Q,E}\right)$ 相同的逻辑来对 CIVET 中查询 $Q$ 和节点 $N$ 之间的距离进行下界估计。我们只需要用节点的 iSAX 表示替换包络的 SAX 表示。形式上，给定一个具有边界 ${L}^{Q}$ 和 ${U}^{Q}$ 的查询 $Q$ 以及一个具有边界 ${L}^{N}$ 和 ${U}^{N}$ 的节点 $N$，我们将它们之间的下界距离定义为：

$$
L{B}_{\text{node }}\left( {Q,N}\right)  =  \tag{22}
$$

$$
\sqrt{\frac{{\eta }^{\prime }}{m}}\sqrt{\mathop{\sum }\limits_{{i = 1}}^{m}\left\{  \begin{matrix} {\left( {\beta }_{L}\left( iSAX{\left( {L}^{N}\right) }_{i}\right)  - {U}_{i}^{Q}\right) }^{2},{if}{\beta }_{L}\left( {{iSAX}{\left( {L}^{N}\right) }_{i}}\right)  > {U}_{i}^{Q} & \\  {\left( {\beta }_{U}\left( iSAX{\left( {U}^{N}\right) }_{i}\right)  - {L}_{i}^{Q}\right) }^{2},{if}{\beta }_{U}\left( {{iSAX}{\left( {U}^{N}\right) }_{i}}\right)  < {L}_{i}^{Q} & \\  0 & ,\text{ otherwise } \end{matrix}\right. }
$$

Here,the ${\eta }^{\prime }$ is set as $\frac{{l}^{\prime }}{{l}^{\prime } + 1}$ ,where ${l}^{\prime } = \left\lfloor  \frac{{l}_{\min }}{m}\right\rfloor$ . Similar to $L{B}_{env}$ , $L{B}_{\text{node }}$ also retains lower-bound properties. Its logic is akin to that of Theorem 3, and thus will not be elaborated here. We depict the detailed calculation of $L{B}_{\text{node }}$ in Figure 2(c1) for an intuitive illustration. The black bar and the gray area, respectively, represent the minimal and maximal UPAAs of scaled queries $\left( {{L}^{Q},{U}^{Q}}\right.$ in Equation 6) and subsequences in the node $\left( {{L}^{N},{U}^{N}}\right)$ . The blue bar represents the numerical calculations of $L{B}_{\text{node }}$ in Equation 22.

这里，${\eta }^{\prime }$ 设为 $\frac{{l}^{\prime }}{{l}^{\prime } + 1}$，其中 ${l}^{\prime } = \left\lfloor  \frac{{l}_{\min }}{m}\right\rfloor$。与 $L{B}_{env}$ 类似，$L{B}_{\text{node }}$ 也保留了下界性质。其逻辑与定理 3 类似，因此这里不再详述。为了直观说明，我们在图 2(c1) 中描述了 $L{B}_{\text{node }}$ 的详细计算过程。黑色条和灰色区域分别表示缩放查询 $\left( {{L}^{Q},{U}^{Q}}\right.$（方程 6 中）和节点 $\left( {{L}^{N},{U}^{N}}\right)$ 中子序列的最小和最大 UPAAs（未指定，可能是某种聚合近似）。蓝色条表示方程 22 中 $L{B}_{\text{node }}$ 的数值计算结果。

### 5.2 Search Algorithm

### 5.2 搜索算法

We utilize the lower bounding distances in the previous subsection to prune unnecessary sub-trees and indicate the visiting order of tree nodes, which supports an efficient approximate search. Then, we refine the results with an exact search procedure.

我们利用上一小节中的下界距离来修剪不必要的子树并指示树节点的访问顺序，这支持高效的近似搜索。然后，我们通过精确搜索过程来优化结果。

Algorithm 1 matches the top- $K$ nearest neighbors of query $Q$ and returns the distance of $\mathrm{K}$ -th nearest neighbor, ${KThBsf}$ ,and the top- $\mathrm{K}$ results, ${R}^{K}$ . Firstly,we initialize variables. ${R}^{K}$ is a max heap used to record the current top-K optimal results, and the variable KThBsf records the largest distance in ${R}^{K}$ . The heap accepts a node and a $L{B}_{\text{node }}$ distance. The heap orders the inserted nodes in descending order of the lower bounding distances. Initially, we insert the root node into the heap with a zero distance (Line 1-3).In the main loop, we first get the closest node from the heap (Line 5). If the $L{B}_{\text{node }}$ of the node is greater than or equal to the ${KThBsf}$ ,the ${R}^{K}$ is already the exact result of the top-K search (Line 6-7). We also adopt early-stopping logic for approximate matching procedure, controlling the total number of visiting leaf nodes (Line 8-9). If the node is terminal, we iterate all the envelopes and check the lower bounding distance between $E$ and $Q$ . The envelope is skipped directly if the lower bounding distance equals or exceeds the KThBsf. Otherwise, we calculate the exact results using checkEnv. Here, the checkEnv calculates distances for every subsequence in the envelope and updates the top-K results in ${KThBsf}$ and ${R}^{K}$ (Line 10-13). Later in Section 5.3, Algorithm 2 will enhance this brute-force procedure with an effective filtering strategy. If the node is an internal or root node, we insert its child nodes into the heap with the lower bounding distances $L{B}_{\text{node }}$ (Line 14-16).

算法1匹配查询 $Q$ 的前 $K$ 个最近邻，并返回第 $\mathrm{K}$ 个最近邻的距离 ${KThBsf}$ 以及前 $\mathrm{K}$ 个结果 ${R}^{K}$ 。首先，我们初始化变量。 ${R}^{K}$ 是一个最大堆，用于记录当前的前K个最优结果，变量KThBsf记录 ${R}^{K}$ 中的最大距离。该堆接受一个节点和一个 $L{B}_{\text{node }}$ 距离。堆按照下界距离降序对插入的节点进行排序。最初，我们以零距离将根节点插入堆中（第1 - 3行）。在主循环中，我们首先从堆中获取最接近的节点（第5行）。如果该节点的 $L{B}_{\text{node }}$ 大于或等于 ${KThBsf}$ ，则 ${R}^{K}$ 已经是前K个搜索的精确结果（第6 - 7行）。我们还为近似匹配过程采用提前停止逻辑，控制访问叶节点的总数（第8 - 9行）。如果该节点是终端节点，我们遍历所有的包络，并检查 $E$ 和 $Q$ 之间的下界距离。如果下界距离等于或超过KThBsf，则直接跳过该包络。否则，我们使用checkEnv计算精确结果。这里，checkEnv为包络中的每个子序列计算距离，并更新 ${KThBsf}$ 和 ${R}^{K}$ 中的前K个结果（第10 - 13行）。在后面的5.3节中，算法2将使用有效的过滤策略来改进这种暴力过程。如果该节点是内部节点或根节点，我们将其子节点及其下界距离 $L{B}_{\text{node }}$ 插入堆中（第14 - 16行）。

<!-- Media -->

Algorithm 1: searchAlgorithm

算法1：搜索算法

Data: $K,Q,{USI},\max {Visit}$ .

数据： $K,Q,{USI},\max {Visit}$ 。

---

Result: ${KThBsf},{R}^{K}$ .

${KThBsf} \leftarrow  \infty$ ,Initialize ${R}^{K};//$ Max-heap with capability $\mathrm{K}$

Initialize heap; // Min-heap

heap.add(USI.root, 0);

// Getting approximate results

4 while heap is not empty do

		$n \leftarrow$ heap.pop(   );

		if $n$ .dist $>  = {KThBsf}$ then

			return ${KThBsf},{R}^{K};//$ Got the exact top- $K$ results

		if Number of visited leaves $>  =$ maxVisit then

			break;

		if $n$ .node is leaf node then

			for $E$ in n.node.envs do

				if $\operatorname{LBenv}\left( {Q,E}\right)  < {KThBsf}$ then

					${KThBsf},{R}^{K} \leftarrow$ checkEnv $\left( {Q,E,{KThBsf},{R}^{K}}\right)$ ;

		else if n.node is internal or root node then

			for childNode in n.node.children do

				heap.add(childNode, LBnode (Q, childNode));

// Getting exact results

for $E$ in USI.sequentialEnvs do

		if $\operatorname{LBenv}\left( {Q,E}\right)  < {KThBsf}$ then

			${KThBsf},{R}^{K} \leftarrow$ checkEnv $\left( {Q,E,{KThBsf},{R}^{K}}\right)$ ;

	return ${KThBsf},{R}^{K}$ ;

---

<!-- Media -->

Till now, if the algorithm does not obtain the exact results, we adopt the sequential checking procedure to refine the final results. We maintain sorted envelopes, which allows us to scan the raw data only once to get the final results. The algorithm sequentially processes the envelopes. If an envelope can not be filtered by $L{B}_{env}$ , we calculate the exact distance between the query and subsequences in the envelope (Line 17-19).

到目前为止，如果算法没有得到精确结果，我们采用顺序检查过程来优化最终结果。我们维护已排序的包络，这使我们只需扫描一次原始数据就能得到最终结果。该算法按顺序处理包络。如果一个包络不能被 $L{B}_{env}$ 过滤，我们计算查询与包络中子序列之间的精确距离（第17 - 19行）。

### 5.3 Enhanced Scanning with Lower Bounding

### 5.3 利用下界进行增强扫描

In Algorithms 1,if an envelope cannot be filtered by $L{B}_{env}$ ,we must calculate all the distances between query and subsequences in the envelopes. In this part, we propose a new lower bounding distance to accelerate the distance calculation.

在算法1中，如果一个包络不能被 $L{B}_{env}$ 过滤，我们必须计算查询与包络中子序列之间的所有距离。在这部分，我们提出一种新的下界距离来加速距离计算。

Lower Bounding. Subsequences at the same position but with varying lengths yield subtle differences after z-normalization, which results in a large amount of redundant calculation. By exploiting the monotonicity in z-normalization, we construct enveloping sequences to delimit the maximum and minimum values of normalized subsequences. Then, we propose a new lower-bound distance $L{B}_{s}$ based on these enveloping sequences to skip unnecessary distance calculation.

下界。相同位置但长度不同的子序列在z - 归一化后会产生细微差异，这导致大量的冗余计算。通过利用z - 归一化中的单调性，我们构造包络序列来界定归一化子序列的最大值和最小值。然后，我们基于这些包络序列提出一种新的下界距离 $L{B}_{s}$ ，以跳过不必要的距离计算。

Given a query $Q$ and a block $B\left( {s,l}\right)$ ,and parameters $W$ and $H$ . When calculating the distances between the query and the subsequences in this block, query only has to be scaled into lengths within the range $\left\lbrack  {l,l + H - 1}\right\rbrack$ rather than all the possible lengths. Therefore, we can delimit a tighter lower and upper bound for query,denoted as ${l}^{Q}$ and ${u}^{Q}$ ,respectively,such that,

给定一个查询 $Q$ 和一个块 $B\left( {s,l}\right)$ ，以及参数 $W$ 和 $H$ 。在计算查询与该块中子序列之间的距离时，查询只需缩放到 $\left\lbrack  {l,l + H - 1}\right\rbrack$ 范围内的长度，而不是所有可能的长度。因此，我们可以为查询界定更严格的下界和上界，分别表示为 ${l}^{Q}$ 和 ${u}^{Q}$ ，使得

$$
{l}_{i}^{Q} = \min \left( \left\{  {\overset{⏜}{{Q}_{i}^{{l}^{\prime }}} \mid  {l}^{\prime } \in  \left\lbrack  {l,l + H - 1}\right\rbrack  }\right\}  \right) , \tag{23}
$$

$$
{u}_{i}^{Q} = \max \left( \left\{  {{Q}_{i}^{{l}^{\prime }} \mid  {l}^{\prime } \in  \left\lbrack  {l,l + H - 1}\right\rbrack  }\right\}  \right) 
$$

where $1 \leq  i \leq  l$ .

其中 $1 \leq  i \leq  l$ 。

Now, we consider all the subsequences with the same start position ${s}^{\prime }$ in the block $B$ . So we have ${s}^{\prime } \in  \left\lbrack  {s,s + W - 1}\right\rbrack$ ,and the length of these subsequences ${l}^{\prime } \in  \left\lbrack  {l,l + H - 1}\right\rbrack$ . We enclose these subsequences with two enveloping sequences, ${l}^{S}$ and ${u}^{S}$ . For brevity,we use $S$ to denote ${T}_{{s}^{\prime },{l}^{\prime }}$ . Here,we show the bounds as follows,

现在，我们考虑块 $B$ 中所有起始位置为 ${s}^{\prime }$ 的子序列。因此，我们有 ${s}^{\prime } \in  \left\lbrack  {s,s + W - 1}\right\rbrack$，且这些子序列的长度为 ${l}^{\prime } \in  \left\lbrack  {l,l + H - 1}\right\rbrack$。我们用两个包络序列 ${l}^{S}$ 和 ${u}^{S}$ 来包围这些子序列。为简洁起见，我们用 $S$ 表示 ${T}_{{s}^{\prime },{l}^{\prime }}$。这里，我们给出如下边界：

$$
{l}_{i}^{S} = \left\{  {\begin{array}{l} \frac{{S}_{i} - {\mu }_{max}}{{\sigma }_{max}},\text{ if }{S}_{i} > {\mu }_{max} \\  \frac{{S}_{i} - {\mu }_{max}}{{\sigma }_{min}},\text{ if }{S}_{i} \leq  {\mu }_{max} \end{array}{u}_{i}^{S} = \left\{  \begin{array}{l} \frac{{S}_{i} - {\mu }_{min}}{{\sigma }_{min}},\text{ if }{S}_{i} > {\mu }_{min} \\  \frac{{S}_{i} - {\mu }_{min}}{{\sigma }_{max}},\text{ if }{S}_{i} \leq  {\mu }_{min} \end{array}\right. }\right. 
$$

(24)

where $1 \leq  i \leq  l$ .

其中 $1 \leq  i \leq  l$。

Till now,we build the lower bound $L{B}_{s}\left( {Q,B,{s}^{\prime }}\right)$ between query and subsequences starting at the same position ${s}^{\prime }$ in the block $B$ ,

到目前为止，我们构建了查询与块 $B$ 中起始位置为 ${s}^{\prime }$ 的子序列之间的下界 $L{B}_{s}\left( {Q,B,{s}^{\prime }}\right)$。

$$
L{B}_{s}\left( {Q,B,{s}^{\prime }}\right)  = \sqrt{\frac{1}{l + H - 1}}\sqrt{\mathop{\sum }\limits_{{i = 1}}^{l}\left\{  \begin{array}{r} {\left( {l}_{i}^{S} - {u}_{i}^{Q}\right) }^{2},\text{ if }{l}_{i}^{S} > {u}_{i}^{Q} \\  {\left( {u}_{i}^{S} - {l}_{i}^{Q}\right) }^{2},\text{ if }{u}_{i}^{S} < {l}_{i}^{Q}. \\  0,\text{ otherwise } \end{array}\right. } \tag{25}
$$

THEOREM 4. Having a block $B\left( {s,l}\right)$ and a query $Q$ ,for any subsequence $S$ starting at the position ${s}^{\prime }$ in this block $B$ ,the following inequality is satisfied,

定理 4。给定一个块 $B\left( {s,l}\right)$ 和一个查询 $Q$，对于该块 $B$ 中起始位置为 ${s}^{\prime }$ 的任意子序列 $S$，满足以下不等式：

$$
L{B}_{s}\left( {Q,B,{s}^{\prime }}\right)  \leq  {D}_{usn}^{ed}\left( {Q,S}\right) . \tag{26}
$$

Proof. First,we prove the correctness of bounds ${l}^{S}$ and ${u}^{S}$ ,using the monotonicity of $\mu$ and $\sigma$ in the definition of Z-normalization. Without loss of generality,we consider $\sigma  > 0$ . Let $f\left( x\right)  = \frac{x - \mu }{\sigma }$ . By taking the partial derivative of $f\left( x\right)$ ,we have

证明。首先，我们利用 Z - 归一化定义中 $\mu$ 和 $\sigma$ 的单调性来证明边界 ${l}^{S}$ 和 ${u}^{S}$ 的正确性。不失一般性，我们考虑 $\sigma  > 0$。设 $f\left( x\right)  = \frac{x - \mu }{\sigma }$。对 $f\left( x\right)$ 求偏导数，我们得到

$$
\frac{\partial f\left( x\right) }{\partial \mu } =  - \frac{1}{\sigma },\frac{\partial f\left( x\right) }{\partial \sigma } =  - \frac{x - \mu }{{\sigma }^{2}}.
$$

For the lower bound ${l}_{i}^{S}$ ,we first consider the case of ${S}_{i} > {\mu }_{\max }$ . Thus,we have $\frac{\partial f\left( x\right) }{\partial \sigma } < 0$ . To obtain the lower bound,we set $\sigma$ to the maximal value ${\sigma }_{\max }$ . For $\mu$ ,we have $\frac{\partial f\left( x\right) }{\partial \mu } < 0$ . So, $\mu$ needs to take the maximal value ${\mu }_{\max }$ . Therefore,in the case of ${S}_{i} - {\mu }_{\max } > 0$ , ${L}_{i}^{{s}^{\prime }}$ takes the value of $\frac{{S}_{i} - {\mu }_{\max }}{{\sigma }_{\max }}$ . Similarly,we can prove the case of ${S}_{i} \leq  {\mu }_{\max }$ and the upper bound ${u}_{i}^{S}$ .

对于下界 ${l}_{i}^{S}$，我们首先考虑 ${S}_{i} > {\mu }_{\max }$ 的情况。因此，我们有 $\frac{\partial f\left( x\right) }{\partial \sigma } < 0$。为了得到下界，我们将 $\sigma$ 设为最大值 ${\sigma }_{\max }$。对于 $\mu$，我们有 $\frac{\partial f\left( x\right) }{\partial \mu } < 0$。所以，$\mu$ 需要取最大值 ${\mu }_{\max }$。因此，在 ${S}_{i} - {\mu }_{\max } > 0$ 的情况下，${L}_{i}^{{s}^{\prime }}$ 取值为 $\frac{{S}_{i} - {\mu }_{\max }}{{\sigma }_{\max }}$。类似地，我们可以证明 ${S}_{i} \leq  {\mu }_{\max }$ 的情况和上界 ${u}_{i}^{S}$。

Thus,considering any subsequence $S = \left( {{s}_{1},{s}_{2},\cdots ,{s}_{\left| S\right| }}\right)$ ,starting at ${s}^{\prime }$ in the block $B$ ,we have ${l}_{i}^{S} \leq  {\widehat{s}}_{i} \leq  {u}_{i}^{S}$ . It is easy to prove that the $L{B}_{s}\left( {Q,B,{s}^{\prime }}\right)$ is a lower bounding distance for ${D}_{usn}^{ed}\left( {Q,S}\right)$ .

因此，考虑块 $B$ 中起始于 ${s}^{\prime }$ 的任意子序列 $S = \left( {{s}_{1},{s}_{2},\cdots ,{s}_{\left| S\right| }}\right)$，我们有 ${l}_{i}^{S} \leq  {\widehat{s}}_{i} \leq  {u}_{i}^{S}$。很容易证明 $L{B}_{s}\left( {Q,B,{s}^{\prime }}\right)$ 是 ${D}_{usn}^{ed}\left( {Q,S}\right)$ 的一个下界距离。

Figure 2(c2) presents an illustration of $L{B}_{s}$ . The grey stripes represent the bounds calculated by Equation 23. The red dashed lines indicate the upper and lower bounds in Equation 24, and the blue shaded area represents the schematic of the $L{B}_{s}$ calculation.

图 2(c2) 展示了 $L{B}_{s}$ 的示意图。灰色条纹表示由方程 23 计算得到的边界。红色虚线表示方程 24 中的上界和下界，蓝色阴影区域表示 $L{B}_{s}$ 计算的示意图。

Searching Algorithm. Now, the procedure of searching an envelope can be accelerated using $L{B}_{s}$ ,presented in Algorithm 2. For each block contained by an envelope (Line 1), we iterate every start position in the block (Line 2-3). Before calculating the exact distances,we check whether to skip the calculation using $L{B}_{s}$ (Line 4). The calculation of concrete distance is conducted for each subsequence starting at the specific position (Line 5-6). We adopt the online normalization technique [30] when computing the $L{B}_{s}$ . Now, we can replace the function checkEnv in Algorithms 1 with checkEnvEnhanced in Algorithm 2 for faster searching.

搜索算法。现在，可以使用算法2中提出的$L{B}_{s}$来加速信封（envelope）搜索过程。对于信封包含的每个块（第1行），我们遍历该块中的每个起始位置（第2 - 3行）。在计算精确距离之前，我们使用$L{B}_{s}$检查是否跳过计算（第4行）。针对从特定位置开始的每个子序列进行具体距离的计算（第5 - 6行）。在计算$L{B}_{s}$时，我们采用在线归一化技术[30]。现在，我们可以用算法2中的checkEnvEnhanced替换算法1中的checkEnv函数，以实现更快的搜索。

<!-- Media -->

Algorithm 2: checkEnvEnhanced

算法2：checkEnvEnhanced

---

Data: $Q,E,{KThBsf},{R}^{K}$ .

Result: ${KThBsf},{R}^{K}$ .

for $B$ in $E$ .blocks do

	$s \leftarrow$ B.startPos;

	for $i \leftarrow  0$ to $W - 1$ do

		if ${LBs}\left( {Q,B,s + i}\right)  < {bsf}$ then

			for $S$ starting at $s + i$ in the block $B$ do

				Calculate $\operatorname{Dusn}\left( {Q,S}\right)$ ,update ${KThBsf}$ and ${R}^{K}$ ;

return ${KThBsf},{R}^{K}$ ;

---

<!-- Media -->

Complexity analysis. We use ${l}_{avg}$ to represent the average length of subsequences in a block. The time complexity of calculating $L{B}_{s}$ is $O\left( {l}_{avg}\right)$ for each start position. Using $\alpha$ as the pruning ratio of $L{B}_{s}$ ,the average time complexity of checkEnvEnhanced is $O\left( {{wsW}\left( {l + \left( {1 - \alpha }\right) {Hl}}\right) }\right)$ ,while checkEnv consumes $O\left( {wsWHl}\right)$ time. Therefore,when the pruning ratio of $L{B}_{s}$ is high,some pre-computation can help us save a amount of real distance calculation.

复杂度分析。我们使用${l}_{avg}$表示块中子序列的平均长度。对于每个起始位置，计算$L{B}_{s}$的时间复杂度为$O\left( {l}_{avg}\right)$。使用$\alpha$作为$L{B}_{s}$的剪枝率，checkEnvEnhanced的平均时间复杂度为$O\left( {{wsW}\left( {l + \left( {1 - \alpha }\right) {Hl}}\right) }\right)$，而checkEnv消耗$O\left( {wsWHl}\right)$的时间。因此，当$L{B}_{s}$的剪枝率较高时，一些预计算可以帮助我们节省大量的实际距离计算。

### 5.4 Supporting DTW Distance

### 5.4 支持DTW距离

To handle the local misalignment between scaled queries and target subsequences,adopting the concept of $L{B}_{\text{Keogh }}\left\lbrack  {19}\right\rbrack$ (which constructs upper and lower-bound sequences for the query sequence to incorporate variations in time axis), we construct a boundary envelope for the query to enable UPAA tolerant the temporal misalignment and support the cDTW distance.

为了处理缩放查询与目标子序列之间的局部未对齐问题，采用$L{B}_{\text{Keogh }}\left\lbrack  {19}\right\rbrack$的概念（它为查询序列构建上下界序列以纳入时间轴上的变化），我们为查询构建一个边界信封，使UPAA能够容忍时间上的未对齐并支持cDTW距离。

Given a length- $n$ query sequence $Q$ and the time warping constraint $c$ ,according to [19],the enveloping sequences for cDTW distance are constructed as,

给定一个长度为$n$的查询序列$Q$和时间规整约束$c$，根据文献[19]，cDTW距离的包络序列构建如下：

$$
{u}_{i}^{\text{keogh }}\left( Q\right)  = \max \left( {{Q}_{\max \left( {1,i - c}\right) },\ldots ,{Q}_{\min \left( {i + c,n}\right) }}\right) , \tag{27}
$$

$$
{l}_{i}^{\text{keogh }}\left( Q\right)  = \min \left( {{Q}_{\max \left( {1,i - c}\right) },\ldots ,{Q}_{\min \left( {i + c,n}\right) }}\right) 
$$

,where $1 \leq  i \leq  n$ . These two sequences ${u}^{\text{keogh }}$ and ${l}^{\text{keogh }}$ form a length- $n$ envelope to enclose the original sequence $Q$ ,which helps to calculate the lower bounding distance for cDTW to accelerate the query processing.

，其中$1 \leq  i \leq  n$。这两个序列${u}^{\text{keogh }}$和${l}^{\text{keogh }}$形成一个长度为$n$的信封，将原始序列$Q$包裹起来，这有助于计算cDTW的下界距离，从而加速查询处理。

Now we reformulate the lower and upper bounds for query in Section 5.1 and Section 5.3 with a similar idea of [19].

现在，我们用与文献[19]类似的思路重新表述第5.1节和第5.3节中查询的上下界。

First, we reconstruct the lower and upper bounds in Equation 14, denoted as ${L}^{Qdtw}$ and ${U}^{Qdtw}$ ,such that,

首先，我们重构公式14中的上下界，记为${L}^{Qdtw}$和${U}^{Qdtw}$，使得：

$$
{L}_{i}^{Qdtw} = \min \left( \left\{  {{\mu }_{i}\left( {{l}^{\text{keogh }}\left( \widehat{{Q}^{l}}\right) }\right)  \mid  l \in  \left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack  }\right\}  \right) , \tag{28}
$$

$$
{U}_{i}^{Qdtw} = \max \left( \left\{  {{\mu }_{i}\left( {{u}^{\text{keogh }}\left( \widehat{{Q}^{l}}\right) }\right)  \mid  l \in  \left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack  }\right\}  \right) .
$$

Similarly, we restate the lower and upper enveloping sequences in Equation 23,denoted as ${l}^{Qdtw}$ and ${u}^{Qdtw}$ ,satisfying that,

类似地，我们重述公式23中的上下包络序列，记为${l}^{Qdtw}$和${u}^{Qdtw}$，满足：

$$
{l}_{i}^{Qdtw} = \min \left( \left\{  {{l}^{\text{keogh }}\left( \widehat{{Q}_{i}^{{l}^{\prime }}}\right)  \mid  {l}^{\prime } \in  \left\lbrack  {l,l + H - 1}\right\rbrack  }\right\}  \right) , \tag{29}
$$

$$
{u}_{i}^{Qdtw} = \max \left( \left\{  {{u}^{\text{keogh }}\left( \overset{⏜}{{Q}_{i}^{{l}^{\prime }}}\right)  \mid  {l}^{\prime } \in  \left\lbrack  {l,l + H - 1}\right\rbrack  }\right\}  \right) .
$$

By replacing the enveloping sequences in $L{B}_{env}$ and $L{B}_{s}$ with ${l}^{Qdtw}$ and ${u}^{Qdtw}$ ,we obtain the new $L{B}_{env}^{dtw}$ and $L{B}_{s}^{dtw}$ suitable for ${D}_{usn}^{dtw}$ . Referring to the property of enveloping sequences [19],we can easily prove the correctness of these lower bounding distances similar to the Theorem 3 and Theorem 4. Therefore, the CIVET index and query algorithms can be adapted to the cDTW distance.

通过用${l}^{Qdtw}$和${u}^{Qdtw}$替换$L{B}_{env}$和$L{B}_{s}$中的包络序列，我们得到适用于${D}_{usn}^{dtw}$的新的$L{B}_{env}^{dtw}$和$L{B}_{s}^{dtw}$。参考包络序列的性质[19]，我们可以像定理3和定理4那样轻松证明这些下界距离的正确性。因此，CIVET索引和查询算法可以适应cDTW距离。

### 5.5 Discussion

### 5.5 讨论

CIVET is designed for finding similar subsequences with the closest ${D}_{usn}$ . Users can use a pattern as the query sequence and find subsequences with the same pattern. CIVET does not support constrained matching and non-normalized matching, as well as ad-hoc semantic search. Nonetheless, CIVET can be extended to many real-world applications. For example, one can use CIVET to find subsequences containing multiple patterns by extracting and searching single patterns and then filtering the returned results to obtain the final answer. CIVET can also be extended to support range search by combining the lower bounding distances in Section 5.1 with the user-provided range threshold for timely abandonment.

CIVET旨在寻找具有最接近${D}_{usn}$的相似子序列。用户可以使用一个模式作为查询序列，查找具有相同模式的子序列。CIVET不支持约束匹配和非归一化匹配，以及临时语义搜索。尽管如此，CIVET可以扩展到许多实际应用中。例如，用户可以使用CIVET通过提取和搜索单个模式来查找包含多个模式的子序列，然后过滤返回的结果以获得最终答案。通过将第5.1节中的下界距离与用户提供的范围阈值相结合以实现及时放弃，CIVET还可以扩展以支持范围搜索。

## 6 EXPERIMENT

## 6 实验

We evaluate the efficiency and effectiveness of CIVET on real and synthetic datasets. All experiments are conducted on a computer running Ubuntu 18.04.6 LTS 64-bit with an Intel(R) Xeon(R) Gold 5215 CPU @ 2.50GHz multi 80 CPU, 64GB RAM, and 4TB Dell PERC H730P disk. All methods are implemented with C++.

我们在真实和合成数据集上评估了CIVET（连续区间可变长度子序列匹配算法）的效率和有效性。所有实验均在一台运行Ubuntu 18.04.6 LTS 64位操作系统的计算机上进行，该计算机配备了英特尔（Intel）至强（Xeon）Gold 5215 CPU（主频2.50GHz，80核）、64GB内存和4TB戴尔PERC H730P磁盘。所有方法均使用C++实现。

### 6.1 Experimental Setup

### 6.1 实验设置

#### 6.1.1 Datasets. Datasets used in our experiments are listed below.

#### 6.1.1 数据集。我们实验中使用的数据集如下所列。

AGW is a gesture recognition dataset that contains 10 types of gestures acquired by a three-axis accelerometer. GMA contains 3D hand trajectories collected with Leap Motion device. PLAID contains current and voltage measurements from different appliance types. GAP records the active energy consumed from 2006 to 2008 in France [14]. CAP contains a periodic EEG activity occurring during NREM sleep phase [36]. SYN is a synthetic dataset generated as the sum of a sequence of random steps extracted from a standard Gaussian distribution $\mathcal{N}\left( {0,1}\right)$ .

AGW是一个手势识别数据集，包含由三轴加速度计采集的10种手势。GMA包含使用Leap Motion设备收集的3D手部轨迹。PLAID包含不同电器类型的电流和电压测量值。GAP记录了2006年至2008年法国的有功能耗[14]。CAP包含非快速眼动睡眠阶段出现的周期性脑电图活动[36]。SYN是一个合成数据集，由从标准高斯分布中提取的随机步长序列求和生成$\mathcal{N}\left( {0,1}\right)$。

The AGW, GMA, and PLAID are three small and real datasets containing variable-length sequences provided by UCR Archive [7]. We randomly select sequences for each dataset as queries; the rest are shuffled and concatenated as a long target sequence. Their lengths are ${370}\mathrm{\;K},{170}\mathrm{\;K}$ ,and ${340}\mathrm{\;K}$ ,respectively. Length ranges of the query workload are $\left\lbrack  {{32},{385}}\right\rbrack  ,\left\lbrack  {{80},{360}}\right\rbrack$ ,and $\left\lbrack  {{200},{684}}\right\rbrack$ .

AGW、GMA和PLAID是三个小型真实数据集，包含UCR存档[7]提供的可变长度序列。我们为每个数据集随机选择序列作为查询；其余序列进行打乱并连接成一个长目标序列。它们的长度分别为${370}\mathrm{\;K},{170}\mathrm{\;K}$和${340}\mathrm{\;K}$。查询工作负载的长度范围为$\left\lbrack  {{32},{385}}\right\rbrack  ,\left\lbrack  {{80},{360}}\right\rbrack$和$\left\lbrack  {{200},{684}}\right\rbrack$。

For GAP, CAP, and SYN, queries are remolded from random subsequences. We scale subsequences into random lengths within $\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$ to generate queries. The additional Gaussian noise is added to the scaled sequences. In the experiments, we set the length range as $\left\lbrack  {{256},{512}}\right\rbrack$ by default. These three datasets (SYN, GAP, CAP) are large datasets, where the size of each dataset is ${10}\mathrm{M}$ . Note that the number of candidate subsequences in the long sequence is about 2 billion under these settings.

对于GAP、CAP和SYN，查询是由随机子序列重塑而来。我们将子序列缩放为$\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$内的随机长度以生成查询。在缩放后的序列中添加额外的高斯噪声。在实验中，我们默认将长度范围设置为$\left\lbrack  {{256},{512}}\right\rbrack$。这三个数据集（SYN、GAP、CAP）是大型数据集，每个数据集的大小为${10}\mathrm{M}$。请注意，在这些设置下，长序列中的候选子序列数量约为20亿。

#### 6.1.2 The Comparison Methods. Three baselines are adopted.

#### 6.1.2 对比方法。采用了三种基线方法。

UCR Suite [30] (UCR for short) searches the most similar normalized subsequence by scanning the whole time series and speeds up the search processing using some pruning techniques. During the subsequence matching, we scale the given query to every possible length within the length range $\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$ and then match top-k subsequences with the UCR Suite.

UCR套件[30]（简称UCR）通过扫描整个时间序列来搜索最相似的归一化子序列，并使用一些剪枝技术加速搜索过程。在子序列匹配过程中，我们将给定的查询缩放到长度范围$\left\lbrack  {{l}_{min},{l}_{max}}\right\rbrack$内的每个可能长度，然后使用UCR套件匹配前k个子序列。

UCR-US [31] is a representative subsequence matching method that supports normalized distance under uniform scaling. It refines the lower-bound techniques in UCR Suite to suit the scenario of uniform scaling. UCR-US is omitted for comparison under DTW due to lack of support. Besides, UCR and UCR-US can directly execute queries without building indexes.

UCR - US[31]是一种具有代表性的子序列匹配方法，支持均匀缩放下的归一化距离。它改进了UCR套件中的下界技术以适应均匀缩放场景。由于缺乏支持，在动态时间规整（DTW）下不进行UCR - US的对比。此外，UCR和UCR - US可以直接执行查询，无需构建索引。

ULISSE [25], as state of the art in subsequence matching, supports matching the subsequences within the constraint on length range by constructing an iSAX-based index. Similarly, we scale the given query to every possible length and conduct the subsequence matchings with ULISSE. Both CIVET and ULISSE support control the number of visited leaf nodes during the approximate matching procedure. Unless otherwise specified, we set it to 5 , the default parameter provided by ULISSE.

ULISSE[25]作为子序列匹配领域的先进技术，通过构建基于iSAX的索引支持在长度范围约束内进行子序列匹配。同样，我们将给定的查询缩放到每个可能的长度，并使用ULISSE进行子序列匹配。CIVET和ULISSE都支持在近似匹配过程中控制访问的叶节点数量。除非另有说明，我们将其设置为5，这是ULISSE提供的默认参数。

UCR and UCR-US are scanning algorithms without an index, while ULISSE and CIVET require constructing indexes. Later, we show that a few queries will amortize the index-building cost.

UCR和UCR - US是无索引的扫描算法，而ULISSE和CIVET需要构建索引。稍后我们将表明，少量查询将分摊索引构建成本。

6.1.3 Parameter Default Setting and Influences. CIVET requires four parameters during the indexing. We provide default values of the parameters that ensure both optimal performance and fair comparison. By default,we set $W$ as ${0.1}\left( {{l}_{\max } - {l}_{\min }}\right) ,H$ as 16,ws as $\frac{{l}_{\max } - {l}_{\min }}{8},m$ as 8 .

6.1.3 参数默认设置及影响。CIVET在索引过程中需要四个参数。我们提供了确保最佳性能和公平比较的参数默认值。默认情况下，我们将$W$设置为${0.1}\left( {{l}_{\max } - {l}_{\min }}\right) ,H$为16，ws设置为$\frac{{l}_{\max } - {l}_{\min }}{8},m$为8。

<!-- Media -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_9.jpg?x=193&y=240&w=1413&h=275&r=0"/>

Figure 5: Influence of Parameters Figure 6: Initial Testing of $m$

图5：参数的影响 图6：$m$的初始测试

<!-- Media -->

We study the influences of parameters on multiple datasets (SYN, CAP, GAP). We adjust one parameter and keep the other as default values to show the changes in query time and index size. Since the structure and size of the index under the same parameter setting are similar, we only show the average size here. The results are depicted in Figure 5.

我们研究了参数对多个数据集（SYN、CAP、GAP）的影响。我们调整一个参数，将其他参数保持为默认值，以展示查询时间和索引大小的变化。由于相同参数设置下索引的结构和大小相似，我们在此仅展示平均大小。结果如图5所示。

$W$ and ${ws}$ ,the step size of starting positions and the window size of envelopes. These two parameters are not sensitive to the query time and can fine-tune the time-space tradeoff (see Figure 5 (a) and (b)). Users can decide them according to the space constraint. In our experiments, we use them to control the index size to be nearly equal to the baseline ULISSE in order to ensure a fair comparison.

$W$和${ws}$，即起始位置的步长和包络的窗口大小。这两个参数对查询时间不敏感，可以微调时空权衡（见图5 (a)和(b)）。用户可以根据空间限制来确定它们。在我们的实验中，我们使用它们来控制索引大小，使其接近基线ULISSE，以确保公平比较。

$H$ ,the step size of lengths. The proper $H$ optimizes the effectiveness of both index pruning and scan filtering. In CIVET, there is a sweet point when varying $H$ where the query performance can achieve the optimum. We decide it by pre-experiments. As shown in Figure 5 (c), $H = {16}$ is the best point for all three datasets.

$H$，即长度的步长。合适的$H$可以优化索引剪枝和扫描过滤的效果。在CIVET中，改变$H$时存在一个最佳点，此时查询性能可以达到最优。我们通过预实验来确定它。如图5 (c)所示，$H = {16}$是所有三个数据集的最佳点。

$m$ ,the number of segments in each subsequence. Experimental results (see Figure 5(d) in the paper) indicate that query efficiency changes slightly when $m$ varies. As $m$ increases,the expressive capability of PAA (and UPAA) improves, but this also leads to higher computational costs [38]. Thus, increasing the number of segments is beneficial up to a point, after which query efficiency starts to decrease [5]. To determine the most effective value of $m$ , we conducted initial testing on a subset of our data. Specifically, we sampled $5\%$ of the subsequences to construct indexes with different $m$ values $\left( {\{ 4,8,{12},{16}\} }\right)$ . Then,we executed a series of random queries to assess the efficiency of these indexes. The $m$ value yielding the highest efficiency was selected to construct the complete index. The results of our initial tests, presented in Figure 6, indicate that for our datasets, $m = 8$ offers the best configuration.

$m$，即每个子序列中的分段数量。实验结果（见论文中的图5(d)）表明，当$m$变化时，查询效率变化不大。随着$m$的增加，PAA（和UPAA）的表达能力提高，但这也会导致更高的计算成本[38]。因此，增加分段数量在一定程度上是有益的，超过这个点后，查询效率开始下降[5]。为了确定$m$的最有效值，我们对数据的一个子集进行了初步测试。具体来说，我们对$5\%$的子序列进行采样，以构建具有不同$m$值$\left( {\{ 4,8,{12},{16}\} }\right)$的索引。然后，我们执行了一系列随机查询，以评估这些索引的效率。选择产生最高效率的$m$值来构建完整的索引。我们初步测试的结果如图6所示，表明对于我们的数据集，$m = 8$提供了最佳配置。

The parameter $W$ in CIVET and $\gamma$ in ULISSE share the same meanings (step size of starting positions). Therefore, we vary them to provide additional details about the index structure, as shown in Table 2. It is evident that, owing to our more compact construction logic, CIVET achieves superior compactness.

CIVET中的参数$W$和ULISSE中的$\gamma$具有相同的含义（起始位置的步长）。因此，我们改变它们以提供有关索引结构的更多细节，如表2所示。显然，由于我们更紧凑的构建逻辑，CIVET实现了更好的紧凑性。

### 6.2 Evaluation of Exact Top-1 Matching

### 6.2 精确Top - 1匹配评估

In this part, we have thoroughly explored Exact Top-1 Matching to analyze the efficiency and effectiveness of our method. All experiments in this subsection are conducted on all datasets.

在这部分，我们深入探索了精确Top - 1匹配，以分析我们方法的效率和有效性。本小节中的所有实验均在所有数据集上进行。

<!-- Media -->

Table 2: Details of Index Structure

表2：索引结构详情

<table><tr><td colspan="2" rowspan="2"/><td colspan="6">W for CIVET $/\gamma$ for ULISSE</td></tr><tr><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td rowspan="4">CIVET</td><td>Height</td><td>8.7</td><td>7.3</td><td>6.5</td><td>5.4</td><td>4.2</td><td>3.1</td></tr><tr><td>#Envs</td><td>1.3M</td><td>620K</td><td>314K</td><td>156K</td><td>78K</td><td>39K</td></tr><tr><td>#Leaf Nodes</td><td>1843</td><td>938</td><td>460</td><td>227</td><td>115</td><td>60</td></tr><tr><td>Index Size(MB)</td><td>269</td><td>138</td><td>68</td><td>32</td><td>16</td><td>7.3</td></tr><tr><td rowspan="4">ULISSE</td><td>Height</td><td>9.5</td><td>8.6</td><td>7.6</td><td>6.9</td><td>6.1</td><td>5.4</td></tr><tr><td>#Envs</td><td>2.0M</td><td>1.1M</td><td>588K</td><td>303K</td><td>153K</td><td>78K</td></tr><tr><td>#Leaf Nodes</td><td>3870</td><td>2243</td><td>1218</td><td>724</td><td>403</td><td>204</td></tr><tr><td>Index Size(MB)</td><td>720</td><td>361</td><td>180</td><td>66</td><td>33</td><td>23</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2"></td><td colspan="6">用于CIVET（西维特）的W $/\gamma$ 用于ULISSE（尤利塞）</td></tr><tr><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td rowspan="4">西维特（CIVET）</td><td>高度</td><td>8.7</td><td>7.3</td><td>6.5</td><td>5.4</td><td>4.2</td><td>3.1</td></tr><tr><td>环境数量</td><td>1.3M</td><td>620K</td><td>314K</td><td>156K</td><td>78K</td><td>39K</td></tr><tr><td>叶节点数量</td><td>1843</td><td>938</td><td>460</td><td>227</td><td>115</td><td>60</td></tr><tr><td>索引大小（兆字节）</td><td>269</td><td>138</td><td>68</td><td>32</td><td>16</td><td>7.3</td></tr><tr><td rowspan="4">尤利塞（ULISSE）</td><td>高度</td><td>9.5</td><td>8.6</td><td>7.6</td><td>6.9</td><td>6.1</td><td>5.4</td></tr><tr><td>环境数量</td><td>2.0M</td><td>1.1M</td><td>588K</td><td>303K</td><td>153K</td><td>78K</td></tr><tr><td>叶节点数量</td><td>3870</td><td>2243</td><td>1218</td><td>724</td><td>403</td><td>204</td></tr><tr><td>索引大小（兆字节）</td><td>720</td><td>361</td><td>180</td><td>66</td><td>33</td><td>23</td></tr></tbody></table>

<!-- Media -->

We record the average exact query time and the pruning power of 100 queries. The pruning power refers to the percentage of the total number of subsequences that do not need to calculate concrete distance. Note that ULISSE skips envelopes for efficient matching while UCR-US prunes sets of subsequences with the help of lower bounding distance. CIVET has two steps of pruning procedure. We refer to the index pruning as the first stage (ST1) and the filtering in checkEnv as the second stage (ST2).

我们记录了100次查询的平均精确查询时间和剪枝能力。剪枝能力是指不需要计算具体距离的子序列总数的百分比。请注意，ULISSE为了实现高效匹配而跳过包络，而UCR - US借助下界距离对一组子序列进行剪枝。CIVET有两个剪枝步骤。我们将索引剪枝称为第一阶段（ST1），将checkEnv中的过滤称为第二阶段（ST2）。

Performance of Exact Top-1 Matching with ED. We first test CIVET and baselines using exact Top-1 matching with ED.

使用欧氏距离（ED）进行精确Top - 1匹配的性能。我们首先使用欧氏距离（ED）的精确Top - 1匹配来测试CIVET和基线方法。

Figure 7(a) and (b) show the average exact Top-1 query time. As depicted, CIVET achieves the acceleration of 2.5x-7.8x compared with UCR-US and ${7.3}\mathrm{x} - {11.5}\mathrm{x}$ compared with ULISSE on different datasets. Figure 7(c) and (d) report the average pruning power of three methods on these datasets. Due to the grouping of subsequences with similar features during index building, the pruning power of ST1 in our indexing method surpasses that of ULISSE. One distinction is important: unlike ULISSE, we consider all possible lengths at once during the index pruning process. Yet, we can still achieve a higher pruning rate in ST1. This also reflects the capability of UPAA to eliminate the influence of global scaling. In ST2, when setting an appropriate value for $H$ ,the algorithm constructs a compact lower-bounding distance for z-normalized subsequences with varying lengths. Therefore, the pruning rate of the scanning algorithm consistently remains very high on different datasets.

图7（a）和（b）展示了平均精确Top - 1查询时间。如图所示，在不同数据集上，与UCR - US相比，CIVET实现了2.5倍至7.8倍的加速，与ULISSE相比实现了${7.3}\mathrm{x} - {11.5}\mathrm{x}$倍的加速。图7（c）和（d）报告了这三种方法在这些数据集上的平均剪枝能力。由于在索引构建过程中对具有相似特征的子序列进行了分组，我们的索引方法中ST1的剪枝能力超过了ULISSE。有一个重要的区别：与ULISSE不同，我们在索引剪枝过程中一次性考虑所有可能的长度。然而，我们仍然可以在ST1中实现更高的剪枝率。这也反映了UPAA消除全局缩放影响的能力。在ST2中，当为$H$设置一个合适的值时，该算法为不同长度的z - 归一化子序列构建了一个紧凑的下界距离。因此，扫描算法的剪枝率在不同数据集上始终保持很高。

To illustrate the overall time cost of indexing and querying, we show the cumulative time to build the index and answer 20 queries. As shown in Figures 7(e) and 7(f), the index-building overhead brought by CIVET index building is very small. After executing 5 queries, the cumulative time cost of CIVET gets smaller than UCR and UCR-US. Additionally, the results indicate that the Disk I/O time required for CIVET is substantially smaller than ULISSE.

为了说明索引构建和查询的总体时间成本，我们展示了构建索引和回答20次查询的累积时间。如图7（e）和7（f）所示，CIVET索引构建带来的索引构建开销非常小。在执行5次查询后，CIVET的累积时间成本小于UCR和UCR - US。此外，结果表明CIVET所需的磁盘I/O时间明显小于ULISSE。

<!-- Media -->

<!-- figureText: (a) Efficiency on Small Datasets Syn./Real Datasets (b) Efficiency on Large Datasets Syn./Real Datasets (d) Pruning Power on Large Datasets (f) Time Cost on Large Datasets UCR Archive Datasets (c) Pruning Power on Small Datasets UCR Archive Datasets (e) Time Cost on Small Datasets -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_10.jpg?x=158&y=251&w=703&h=889&r=0"/>

Figure 7: Exact Top-1 Matching with ED

图7：使用欧氏距离（ED）进行精确Top - 1匹配

<!-- Media -->

Performance of Exact Top-1 Matching with DTW. CIVET is also extended to support variable-length matching within cDTW distance. So we conduct the exact Top-1 matching with cDTW distance here. Figure 8 shows average query time and pruning effectiveness. As illustrated, CIVET is 5.5x-16.6x faster than ULISSE and 8.1x-22.7x than UCR on different datasets.

使用动态时间规整（DTW）进行精确Top - 1匹配的性能。CIVET还进行了扩展，以支持在受限动态时间规整（cDTW）距离内的可变长度匹配。因此，我们在这里使用cDTW距离进行精确Top - 1匹配。图8展示了平均查询时间和剪枝效果。如图所示，在不同数据集上，CIVET比ULISSE快5.5倍至16.6倍，比UCR快8.1倍至22.7倍。

In summary,our approach achieves an average of $5\mathrm{x}$ speedups than UCR-US, 11x speed-ups than ULISSE, and 15x on UCR on different datasets. The efficiency of matching underscores our method's superiority in variable-length subsequence matching. In addition, the pruning rate and other statistical data also reflect the high effectiveness of UPAA in representing variable-length sequences, and the efficiency and robustness of the query algorithm.

综上所述，我们的方法在不同数据集上比UCR - US平均实现了$5\mathrm{x}$倍的加速，比ULISSE实现了11倍的加速，比UCR实现了15倍的加速。匹配效率凸显了我们的方法在可变长度子序列匹配方面的优越性。此外，剪枝率和其他统计数据也反映了UPAA在表示可变长度序列方面的高效性，以及查询算法的效率和鲁棒性。

### 6.3 Exploratory Experiments

### 6.3 探索性实验

To analyze more beneficial aspects of CIVET, we conduct a series of exploratory experiments in this part. All experiments are performed on large and real datasets, specifically GAP and CAP.

为了分析CIVET更多有益的方面，我们在这部分进行了一系列探索性实验。所有实验均在大型真实数据集上进行，具体为GAP和CAP。

Performance of Top-K Matching. CIVET supports exact Top- $\mathrm{K}$ query as well. We also show the query time of the exact Top-K query varying $K$ in Figure 9. Both ED and DTW distances are tested. The experimental results show that CIVET keeps a stable performance as the number of nearest neighbors increases.

Top - K匹配的性能。CIVET也支持精确Top - $\mathrm{K}$查询。我们还在图9中展示了精确Top - K查询的查询时间随$K$的变化情况。同时测试了欧氏距离（ED）和动态时间规整（DTW）距离。实验结果表明，随着最近邻数量的增加，CIVET保持了稳定的性能。

Performance when Varying ${l}_{\max } - {l}_{\min }$ . We explore the query time and the pruning power of different methods on GAP and CAP when varying the length range ${l}_{max} - {l}_{min}$ in Figure 10. The index pruning of CIVET (CIVET-ST1) outperforms that of ULISSE, and pruning during the scanning (CIVET-ST2) also exhibits superior capability compared to the pruning efficiency of UCR-US. For CIVET, due to the effective grouping strategies of subsequences, subsequences in the same blocks and envelopes have more similar features. Therefore, the pruning power drops with a slower trend, and the query time grows more slowly.

改变${l}_{\max } - {l}_{\min }$时的性能。我们在图10中展示了在GAP和CAP数据集上，改变长度范围${l}_{max} - {l}_{min}$时不同方法的查询时间和剪枝能力。CIVET的索引剪枝（CIVET - ST1）优于ULISSE，扫描过程中的剪枝（CIVET - ST2）与UCR - US的剪枝效率相比也表现出更优的能力。对于CIVET，由于子序列的有效分组策略，同一块和包络中的子序列具有更多相似的特征。因此，剪枝能力下降的趋势更慢，查询时间增长也更慢。

<!-- Media -->

<!-- figureText: UCR Archive Datasets (b) Efficiency on Large Datasets (d) Pruning Power on Large Data (a) Efficiency on Small Datasets UCR Archive Datasets (c) Pruning Power on Small Data -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_10.jpg?x=944&y=254&w=685&h=600&r=0"/>

Figure 8: Exact Top-1 Matching with DTW

图8：使用动态时间规整（DTW）进行精确Top - 1匹配

<!-- Media -->

Accuracy of Approximate Matching In this part, we show the accuracy of the approximate matching compared with ULISSE. For ULISSE, we get the results from scaled queries of all possible lengths. We record how many approximate results belong to the top-100 exact results (Recall of Top-100). Figure 11 demonstrates how the accuracy of approximate matching changes as the number of visited subsequences increases. CIVET achieves a higher recall on both datasets than ULISSE.

近似匹配的准确性 在这部分，我们展示了与ULISSE相比，近似匹配的准确性。对于ULISSE，我们从所有可能长度的缩放查询中获取结果。我们记录了有多少近似结果属于前100个精确结果（前100召回率）。图11展示了随着访问子序列数量的增加，近似匹配的准确性是如何变化的。在两个数据集上，CIVET的召回率都高于ULISSE。

<!-- Media -->

<!-- figureText: (a) Top-K Matching on GAP with ED (c) Top-K Matching on CAP with EI -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_10.jpg?x=938&y=1425&w=701&h=587&r=0"/>

Figure 9: Exact Top-K Matching

图9：精确的前K匹配

<!-- figureText: (a) Efficiency on GAP (ED) (c) Pruning Power on GAP (ED) -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_11.jpg?x=165&y=253&w=692&h=537&r=0"/>

Figure 10: Exact Top-1 Matching Varying ${l}_{\max } - {l}_{\min }$

图10：精确的前1匹配，${l}_{\max } - {l}_{\min }$可变

<!-- figureText: 50M 100M 150M 200M 250M 50M 100M 150M 200M 250M #Subsequences Accessed (b) Accuracy on CAP #Subsequences Accessed (a) Accuracy on GAP -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_11.jpg?x=162&y=903&w=692&h=278&r=0"/>

Figure 11: Approximate Matching

图11：近似匹配

<!-- Media -->

### 6.4 Scalability

### 6.4 可扩展性

To test the scalability of CIVET, we conduct experiments on SYN with different sizes from ${10}^{5}$ to ${10}^{9}$ . We record the index building time and the query time of CIVET in Figure 12. The time of CIVET building increases linearly with the amount of data. The efficient indexing algorithm proposed by ULISSE is also suitable for our index. So we can construct CIVET with the same time complexity (with a few more constants). In addition, we also depict the exact matching time of different methods. As the size of the dataset grows, the query time of our method grows linearly. Again, CIVET is faster than baselines by about one order of magnitude. Therefore, CIVET has the ability to index and query efficiently with great scalability.

为了测试CIVET的可扩展性，我们在大小从${10}^{5}$到${10}^{9}$不等的SYN数据集上进行了实验。我们在图12中记录了CIVET的索引构建时间和查询时间。CIVET的构建时间随数据量线性增加。ULISSE提出的高效索引算法也适用于我们的索引。因此，我们可以以相同的时间复杂度（多几个常数）构建CIVET。此外，我们还描绘了不同方法的精确匹配时间。随着数据集大小的增长，我们方法的查询时间线性增长。同样，CIVET比基线方法快约一个数量级。因此，CIVET能够高效地进行索引和查询，具有很强的可扩展性。

## 7 RELATED WORK

## 7 相关工作

Fixed-length Subsequence Matching. UCR SUITE [30] devise several techniques for computing lower bounds for the efficient similarity query. Many works have improved time series indexing through various summarization techniques [3, 11, 34, 37]. However, all of these indexes require queries with a preset and fixed length. Recent research has tried to tackle this limitation. Some studies propose efficient approaches to support constrained normalized subsequence matching, which is different from our problem. [4, 39]. ULISSE extends iSAX with envelope summarization so that it can index subsequences of variable length [25]. However, the methods above only support equal-length queries, meaning that the lengths of matched subsequences and queries are the same. As a result, they are not good at handling global scaling in the time dimension.

固定长度子序列匹配。UCR SUITE [30]设计了几种计算下界的技术，用于高效的相似性查询。许多工作通过各种汇总技术改进了时间序列索引[3, 11, 34, 37]。然而，所有这些索引都要求查询具有预设的固定长度。最近的研究试图解决这一限制。一些研究提出了支持受限归一化子序列匹配的高效方法，这与我们的问题不同[4, 39]。ULISSE通过包络汇总扩展了iSAX，使其能够对可变长度的子序列进行索引[25]。然而，上述方法仅支持等长查询，即匹配子序列和查询的长度相同。因此，它们不擅长处理时间维度上的全局缩放。

<!-- Media -->

<!-- figureText: (a) Efficiency of Index Building (b) Exact Top-1 Matching with ED -->

<img src="https://cdn.noedgeai.com/019594aa-b155-7695-8d43-16c01cad9409_11.jpg?x=940&y=254&w=692&h=282&r=0"/>

Figure 12: Scalability

图12：可扩展性

<!-- Media -->

Variable-Length Subsequence Matching. The SpADe distance function is an elastic distance function that can handle time drift, amplitude drift, and shape scaling [6]. Lian et al. create the multi-scaled segment mean (MSM), which can be computed gradually and is suited to the stream features [23]. Kotsifakos et al. modify the edit distance for query by humming problem [21, 22]. These elastic measures do allow for variation in length. However, they only provide tolerance to local shifts in two sequences without the ability to deal with large global scaling. Recently, the sketch-based method has attracted the attention of many researchers $\left\lbrack  {{27},{28},{35}}\right\rbrack$ . While these methods are capable of matching variable-length data, their main focus is on the whole matching.

可变长度子序列匹配。SpADe距离函数是一种弹性距离函数，能够处理时间漂移、幅度漂移和形状缩放[6]。Lian等人创建了多尺度段均值（MSM），它可以逐步计算，适合流特征[23]。Kotsifakos等人修改了哼唱查询问题的编辑距离[21, 22]。这些弹性度量确实允许长度变化。然而，它们仅对两个序列中的局部偏移提供容差，而无法处理大的全局缩放。最近，基于草图的方法吸引了许多研究人员的关注$\left\lbrack  {{27},{28},{35}}\right\rbrack$。虽然这些方法能够匹配可变长度的数据，但它们的主要重点是整体匹配。

Uniform Scaling. Keogh et al. [18] utilize R-tree to index and accelerate uniform scaling distance calculation for whole matching. Ada et al. motivate the accommodation of US and DTW [12]. The extended work on UCR Suite proposes a lower bound distance to accelerate sequential match under uniform scaling [31]. A recent work refines the distance with a tighter bound [33]. However, these methods have not thoroughly investigated the properties of Uniform Scaling, and they have not been meticulously designed for subsequence queries under normalized conditions. There is still significant room for improvement in terms of efficiency.

均匀缩放。Keogh等人[18]利用R树进行索引，并加速整体匹配的均匀缩放距离计算。Ada等人推动了对均匀缩放（US）和动态时间规整（DTW）的融合[12]。UCR Suite的扩展工作提出了一种下界距离，以加速均匀缩放条件下的顺序匹配[31]。最近的一项工作用更严格的边界细化了距离[33]。然而，这些方法没有深入研究均匀缩放的特性，也没有针对归一化条件下的子序列查询进行精心设计。在效率方面仍有很大的改进空间。

## 8 CONCLUSION

## 8 结论

In this paper, we extend the PAA as UPAA with uniform segmentation, which possesses the ability to tolerate global scaling. Besides combining two grouping strategies, our approach constructs a compact and effective index structure that supports efficient subsequence matching facilitated by index pruning and data filtering. Experimental results on different synthetic and real datasets demonstrate the efficiency, scalability, and effectiveness of our approach.

在本文中，我们将分段聚合近似（PAA）扩展为均匀分段聚合近似（UPAA），它具有容忍全局缩放的能力。除了结合两种分组策略外，我们的方法构建了一个紧凑而有效的索引结构，通过索引剪枝和数据过滤支持高效的子序列匹配。在不同的合成和真实数据集上的实验结果证明了我们方法的效率、可扩展性和有效性。

## ACKNOWLEDGMENTS

## 致谢

The authors thank the anonymous reviewers for their insightful comments and suggestions. This work was supported by the National Key R&D Program of China (No. 2021YFB3300502).

作者感谢匿名审稿人提出的深刻见解和建议。本工作得到了国家重点研发计划（编号：2021YFB3300502）的支持。

## REFERENCES

## 参考文献

[1] Steven B. Achelis. 2001. Technical Analysis from A to $Z$ (2nd ed.). McGraw Hill Professional.

[1] 史蒂文·B·阿切利斯（Steven B. Achelis）. 2001年. 《技术分析从A到$Z$》（第2版）. 麦格劳·希尔专业出版公司.

[2] Alessandro Camerra, Themis Palpanas, Jin Shieh, and Eamonn Keogh. 2010. iSAX 2.0: Indexing and Mining One Billion Time Series. In Proceedings of the 2010 IEEE International Conference on Data Mining (ICDM). IEEE Computer Society, 58-67.

[2] 亚历山德罗·卡梅拉（Alessandro Camerra）、西弥斯·帕尔帕纳斯（Themis Palpanas）、金·谢（Jin Shieh）和埃蒙·基奥（Eamonn Keogh）. 2010年. iSAX 2.0：对十亿级时间序列进行索引和挖掘. 收录于《2010年IEEE国际数据挖掘会议（ICDM）论文集》. IEEE计算机协会，第58 - 67页.

[3] Alessandro Camerra, Jin Shieh, Themis Palpanas, Thanawin Rakthanmanon, and Eamonn Keogh. 2014. Beyond one billion time series: indexing and mining very large time series collections with $i$ sax2+. Knowledge and information systems 39, 1 (2014), 123-151.

[3] 亚历山德罗·卡梅拉（Alessandro Camerra）、金·谢（Jin Shieh）、西弥斯·帕尔帕纳斯（Themis Palpanas）、塔纳温·拉赫坦马农（Thanawin Rakthanmanon）和埃蒙·基奥（Eamonn Keogh）. 2014年. 超越十亿级时间序列：使用$i$ sax2+对超大型时间序列集合进行索引和挖掘. 《知识与信息系统》39, 1（2014年），第123 - 151页.

[4] Zemin Chao, Hong Gao, Yinan An, and Jianzhong Li. 2022. The inherent time complexity and an efficient algorithm for subsequence matching problem. Proceedings of the VLDB Endowment 15, 7 (2022), 1453-1465.

[4] 晁泽敏（Zemin Chao）、高宏（Hong Gao）、安怡楠（Yinan An）和李建中（Jianzhong Li）. 2022年. 子序列匹配问题的内在时间复杂度及高效算法. 《VLDB捐赠基金会议录》15, 7（2022年），第1453 - 1465页.

[5] Georgios Chatzigeorgakidis, Dimitrios Skoutas, Kostas Patroumpas, Themis Palpanas, Spiros Athanasiou, and Spiros Skiadopoulos. 2023. Efficient Range and kNN Twin Subsequence Search in Time Series. IEEE Transactions on Knowledge and Data Engineering 35, 6 (2023), 5794-5807.

[5] 乔治奥斯·查齐乔治亚基斯（Georgios Chatzigeorgakidis）、迪米特里奥斯·斯科塔斯（Dimitrios Skoutas）、科斯塔斯·帕楚姆帕斯（Kostas Patroumpas）、西弥斯·帕尔帕纳斯（Themis Palpanas）、斯皮罗斯·阿塔纳索乌（Spiros Athanasiou）和斯皮罗斯·斯基亚多普洛斯（Spiros Skiadopoulos）. 2023年. 时间序列中高效的范围和k近邻孪生子序列搜索. 《IEEE知识与数据工程汇刊》35, 6（2023年），第5794 - 5807页.

[6] Yueguo Chen, Mario A Nascimento, Beng Chin Ooi, and Anthony KH Tung. 2007. Spade: On shape-based pattern detection in streaming time series. In 2007 IEEE 23rd International conference on data engineering (ICDE). IEEE, 786-795.

[6] 陈跃国（Yueguo Chen）、马里奥·A·纳西门托（Mario A Nascimento）、翁炳钦（Beng Chin Ooi）和董家鸿（Anthony KH Tung）. 2007年. Spade：流式时间序列中基于形状的模式检测. 收录于《2007年IEEE第23届国际数据工程会议（ICDE）》. IEEE，第786 - 795页.

[7] Hoang Anh Dau, Anthony Bagnall, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, and Eamonn Keogh. 2019. The UCR time series archive. IEEE/CAA Journal of Automatica Sinica 6, 6 (2019), 1293-1305.

[7] 黄安·道（Hoang Anh Dau）、安东尼·巴格内尔（Anthony Bagnall）、卡维·卡姆加尔（Kaveh Kamgar）、叶志佳（Chin-Chia Michael Yeh）、朱燕（Yan Zhu）、沙加耶格·加尔加比（Shaghayegh Gharghabi）、乔蒂拉特·安·拉塔纳马哈塔纳（Chotirat Ann Ratanamahatana）和埃蒙·基奥（Eamonn Keogh）. 2019年. UCR时间序列存档. 《IEEE/CAA自动化学报》6, 6（2019年），第1293 - 1305页.

[8] Hui Ding, Goce Trajcevski, Peter Scheuermann, Xiaoyue Wang, and Eamonn Keogh. 2008. Querying and mining of time series data: experimental comparison of representations and distance measures. Proceedings of the VLDB Endowment 1, 2 (2008), 1542-1552.

[8] 丁辉（Hui Ding）、戈采·特拉伊切夫斯基（Goce Trajcevski）、彼得·舍尔曼（Peter Scheuermann）、王晓月（Xiaoyue Wang）和埃蒙·基奥（Eamonn Keogh）. 2008年. 时间序列数据的查询与挖掘：表示方法和距离度量的实验比较. 《VLDB捐赠基金会议录》1, 2（2008年），第1542 - 1552页.

[9] Robert D Edwards, John Magee, and WH Charles Bassetti. 2018. Technical analysis of stock trends. CRC press.

[9] 罗伯特·D·爱德华兹（Robert D Edwards）、约翰·马吉（John Magee）和W·H·查尔斯·巴西蒂（WH Charles Bassetti）. 2018年. 《股票趋势技术分析》. CRC出版社.

[10] Waiyawuth Euachongprasit and Chotirat Ann Ratanamahatana. 2008. Efficient multimedia time series data retrieval under uniform scaling and normalisation. In Advances in Information Retrieval: 30th European Conference on IR Research (ECIR). Springer, 506-513.

[10] 瓦亚乌特·尤阿冲普拉西特（Waiyawuth Euachongprasit）和乔蒂拉特·安·拉塔纳马哈塔纳（Chotirat Ann Ratanamahatana）. 2008年. 统一缩放和归一化下的高效多媒体时间序列数据检索. 收录于《信息检索进展：第30届欧洲信息检索研究会议（ECIR）》. 施普林格出版社，第506 - 513页.

[11] Christos Faloutsos, Mudumbai Ranganathan, and Yannis Manolopoulos. 1994. Fast subsequence matching in time-series databases. Acm Sigmod Record 23, 2 (1994), 419-429.

[11] 克里斯托斯·法鲁索斯（Christos Faloutsos）、穆丹比·兰加纳坦（Mudumbai Ranganathan）和扬尼斯·马诺洛普洛斯（Yannis Manolopoulos）. 1994年. 时间序列数据库中的快速子序列匹配. 《ACM SIGMOD记录》23, 2（1994年），第419 - 429页.

[12] Ada Wai-Chee Fu, Eamonn Keogh, Leo Yung Hang Lau, Chotirat Ann Ratanama-hatana, and Raymond Chi-Wing Wong. 2008. Scaling and time warping in time series querying. The VLDB Journal 17, 4 (2008), 899-921.

[12] 傅慧慈（Ada Wai-Chee Fu）、埃蒙·基奥（Eamonn Keogh）、刘永辉（Leo Yung Hang Lau）、乔蒂拉特·安·拉塔纳马哈塔纳（Chotirat Ann Ratanama - hatana）和黄志荣（Raymond Chi-Wing Wong）. 2008年. 时间序列查询中的缩放和时间规整. 《VLDB杂志》17, 4（2008年），第899 - 921页.

[13] Søren Kejser Jensen, Torben Bach Pedersen, and Christian Thomsen. 2017. Time series management systems: A survey. IEEE Transactions on Knowledge and Data Engineering 29, 11 (2017), 2581-2600.

[13] 索伦·凯泽·延森（Søren Kejser Jensen）、托本·巴赫·佩德森（Torben Bach Pedersen）和克里斯蒂安·汤姆森（Christian Thomsen）. 2017年. 时间序列管理系统：综述. 《IEEE知识与数据工程汇刊》29, 11（2017年），第2581 - 2600页.

[14] Markelle Kelly, Rachel Longjohn, and Kolby Nottingham. 2021. The UCI Machine Learning Repository. https://archive.ics.uci.edu

[14] 马克尔·凯利（Markelle Kelly）、雷切尔·朗约翰（Rachel Longjohn）和科尔比·诺丁汉（Kolby Nottingham）. 2021年. UCI机器学习库. https://archive.ics.uci.edu

[15] Eamonn Keogh. 2006. A decade of progress in indexing and mining large time series databases. In Proceedings of the 32nd International Conference on Very Large Data Bases (VLDB). VLDB Endowment, 1268-1268.

[15] 埃蒙·基奥（Eamonn Keogh）. 2006年. 大型时间序列数据库索引和挖掘十年进展. 收录于《第32届国际超大型数据库会议（VLDB）论文集》. VLDB捐赠基金，第1268 - 1268页.

[16] Eamonn Keogh, Kaushik Chakrabarti, Michael Pazzani, and Sharad Mehrotra. 2001. Dimensionality reduction for fast similarity search in large time series databases. Knowledge and information Systems 3, 3 (2001), 263-286.

[16] 埃蒙·基奥（Eamonn Keogh）、考希克·查克拉巴蒂（Kaushik Chakrabarti）、迈克尔·帕扎尼（Michael Pazzani）和沙拉德·梅赫罗特拉（Sharad Mehrotra）. 2001年. 大型时间序列数据库中用于快速相似性搜索的降维方法. 《知识与信息系统》3, 3（2001年），第263 - 286页.

[17] Eamonn Keogh and Shruti Kasetty. 2003. On the Need for Time Series Data Mining Benchmarks: A Survey and Empirical Demonstration. Data Mining and Knowledge Discovery 7 (2003), 349-371.

[17] 埃蒙·基奥（Eamonn Keogh）和什鲁蒂·卡塞蒂（Shruti Kasetty）。2003年。关于时间序列数据挖掘基准的必要性：一项调查与实证演示。《数据挖掘与知识发现》（Data Mining and Knowledge Discovery）7（2003），349 - 371。

[18] Eamonn Keogh, Themistoklis Palpanas, Victor B Zordan, Dimitrios Gunopulos, and Marc Cardle. 2004. Indexing large human-motion databases. In Proceedings of the Thirtieth international conference on Very large data bases-Volume 30.780-791.

[18] 埃蒙·基奥（Eamonn Keogh）、西奥多克里斯·帕尔帕纳斯（Themistoklis Palpanas）、维克多·B·佐尔丹（Victor B Zordan）、迪米特里奥斯·古诺普洛斯（Dimitrios Gunopulos）和马克·卡德尔（Marc Cardle）。2004年。大型人体运动数据库的索引。收录于《第三十届国际超大型数据库会议论文集 - 第30卷》（Proceedings of the Thirtieth international conference on Very large data bases - Volume 30）。780 - 791。

[19] Eamonn Keogh and Chotirat Ann Ratanamahatana. 2005. Exact indexing of dynamic time warping. Knowledge and information systems 7 (2005), 358-386.

[19] 埃蒙·基奥（Eamonn Keogh）和乔蒂拉特·安·拉塔纳马哈塔纳（Chotirat Ann Ratanamahatana）。2005年。动态时间规整的精确索引。《知识与信息系统》（Knowledge and information systems）7（2005），358 - 386。

[20] Haridimos Kondylakis, Niv Dayan, Kostas Zoumpatianos, and Themis Palpanas. 2018. Coconut: a scalable bottom-up approach for building data series indexes. Proceedings of the VLDB Endowment 11, 6 (2018), 677-690.

[20] 哈里迪莫斯·孔迪拉基斯（Haridimos Kondylakis）、尼夫·戴扬（Niv Dayan）、科斯塔斯·祖姆帕蒂亚诺斯（Kostas Zoumpatianos）和西姆斯·帕尔帕纳斯（Themis Palpanas）。2018年。椰子（Coconut）：一种可扩展的自底向上构建数据序列索引的方法。《VLDB捐赠会议论文集》（Proceedings of the VLDB Endowment）11, 6（2018），677 - 690。

[21] Alexios Kotsifakos, Panagiotis Papapetrou, Jaakko Hollmén, and Dimitrios Gunopulos. 2011. A subsequence matching with gaps-range-tolerances framework: a query-by-humming application. Proceedings of the VLDB Endowment 4, 11 (2011), 761-771.

[21] 阿列克西奥斯·科齐法科斯（Alexios Kotsifakos）、帕纳约蒂斯·帕帕佩特鲁（Panagiotis Papapetrou）、亚科·霍尔门（Jaakko Hollmén）和迪米特里奥斯·古诺普洛斯（Dimitrios Gunopulos）。2011年。一种带间隙 - 范围容差的子序列匹配框架：哼唱查询应用。《VLDB捐赠会议论文集》（Proceedings of the VLDB Endowment）4, 11（2011），761 - 771。

[22] Alexios Kotsifakos, Panagiotis Papapetrou, Jaakko Hollmén, Dimitrios Gunop-ulos, Vassilis Athitsos, and George Kollios. 2012. Hum-a-song: a subsequence matching with gaps-range-tolerances query-by-humming system. Proceedings of the VLDB Endowment 5, 12 (2012), 1930-1933.

[22] 阿列克西奥斯·科齐法科斯（Alexios Kotsifakos）、帕纳约蒂斯·帕帕佩特鲁（Panagiotis Papapetrou）、亚科·霍尔门（Jaakko Hollmén）、迪米特里奥斯·古诺普洛斯（Dimitrios Gunop - ulos）、瓦西利斯·阿西索斯（Vassilis Athitsos）和乔治·科利奥斯（George Kollios）。2012年。哼唱歌曲（Hum - a - song）：一种带间隙 - 范围容差的哼唱查询子序列匹配系统。《VLDB捐赠会议论文集》（Proceedings of the VLDB Endowment）5, 12（2012），1930 - 1933。

[23] Xiang Lian, Lei Chen, Jeffrey Xu Yu, Guoren Wang, and Ge Yu. 2007. Similarity match over high speed time-series streams. In 2007 IEEE 23rd International Conference on Data Engineering (ICDE). IEEE, 1086-1095.

[23] 连翔（Xiang Lian）、陈雷（Lei Chen）、徐宇杰弗里（Jeffrey Xu Yu）、王国仁（Guoren Wang）和余歌（Ge Yu）。2007年。高速时间序列流上的相似性匹配。收录于《2007年IEEE第23届国际数据工程会议》（2007 IEEE 23rd International Conference on Data Engineering (ICDE)）。IEEE，1086 - 1095。

[24] Jessica Lin, Eamonn Keogh, Stefano Lonardi, and Bill Chiu. 2003. A symbolic representation of time series, with implications for streaming algorithms. In Proceedings of the 8th ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery (DMKD). ACM, 2-11.

[24] 杰西卡·林（Jessica Lin）、埃蒙·基奥（Eamonn Keogh）、斯特凡诺·洛纳尔迪（Stefano Lonardi）和比尔·邱（Bill Chiu）。2003年。时间序列的一种符号表示及其对流式算法的影响。收录于《第8届ACM SIGMOD数据挖掘与知识发现研究问题研讨会论文集》（Proceedings of the 8th ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery (DMKD)）。ACM，2 - 11。

[25] Michele Linardi and Themis Palpanas. 2018. Scalable, variable-length similarity search in data series: The ULISSE approach. Proceedings of the VLDB Endowment 11, 13 (2018), 2236-2248.

[25] 米歇尔·洛纳尔迪（Michele Linardi）和西姆斯·帕尔帕纳斯（Themis Palpanas）。2018年。数据序列中可扩展的可变长度相似性搜索：尤利西斯（ULISSE）方法。《VLDB捐赠会议论文集》（Proceedings of the VLDB Endowment）11, 13（2018），2236 - 2248。

[26] Michele Linardi, Yan Zhu, Themis Palpanas, and Eamonn Keogh. 2018. Matrix Profile X: VALMOD - Scalable Discovery of Variable-Length Motifs in Data Series. In Proceedings of the 2018 International Conference on Management of Data (SIGMOD). ACM, 1053-1066.

[26] 米歇尔·洛纳尔迪（Michele Linardi）、朱燕（Yan Zhu）、西姆斯·帕尔帕纳斯（Themis Palpanas）和埃蒙·基奥（Eamonn Keogh）。2018年。矩阵轮廓X：VALMOD - 数据序列中可变长度模式的可扩展发现。收录于《2018年国际数据管理会议论文集》（Proceedings of the 2018 International Conference on Management of Data (SIGMOD)）。ACM，1053 - 1066。

[27] Miro Mannino and Azza Abouzied. 2018. Expressive Time Series Querying with Hand-Drawn Scale-Free Sketches. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI). ACM, 1-13.

[27] 米罗·曼尼诺（Miro Mannino）和阿扎·阿布齐德（Azza Abouzied）。2018年。使用手绘无标度草图进行富有表现力的时间序列查询。收录于《2018年人机交互大会论文集》（Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI)）。ACM，1 - 13。

[28] Prithiviraj K Muthumanickam, Katerina Vrotsou, Matthew Cooper, and Jimmy Johansson. 2016. Shape grammar extraction for efficient query-by-sketch pattern matching in long time series. In 2016 IEEE Conference on Visual Analytics Science and Technology (VAST). IEEE, 121-130.

[28] 普里蒂维拉杰·K·穆图马尼卡姆（Prithiviraj K Muthumanickam）、卡特里娜·弗罗佐（Katerina Vrotsou）、马修·库珀（Matthew Cooper）和吉米·约翰松（Jimmy Johansson）。2016年。长时序列中用于高效草图查询模式匹配的形状语法提取。收录于《2016年IEEE可视化分析科学与技术会议》（2016 IEEE Conference on Visual Analytics Science and Technology (VAST)）。IEEE，121 - 130。

[29] John Paparrizos, Chunwei Liu, Aaron J. Elmore, and Michael J. Franklin. 2020. Debunking Four Long-Standing Misconceptions of Time-Series Distance Measures. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD). ACM, 1887-1905.

[29] 约翰·帕帕里佐斯（John Paparrizos）、刘春伟（Chunwei Liu）、亚伦·J·埃尔莫尔（Aaron J. Elmore）和迈克尔·J·富兰克林（Michael J. Franklin）。2020年。揭穿时间序列距离度量的四个长期误解。收录于《2020年ACM SIGMOD国际数据管理会议论文集》（Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD)）。ACM，1887 - 1905。

[30] Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, Brandon Westover, Qiang Zhu, Jesin Zakaria, and Eamonn Keogh. 2012. Searching and mining trillions of time series subsequences under dynamic time warping. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD). ACM, 262-270.

[30] 塔纳温·拉赫坦马农（Thanawin Rakthanmanon）、比尔森·坎帕纳（Bilson Campana）、阿卜杜拉·穆恩（Abdullah Mueen）、古斯塔沃·巴蒂斯塔（Gustavo Batista）、布兰登·韦斯托弗（Brandon Westover）、朱强（Qiang Zhu）、杰辛·扎卡里亚（Jesin Zakaria）和埃蒙·基奥（Eamonn Keogh）。2012年。动态时间规整下对数万亿时间序列子序列的搜索与挖掘。收录于《第18届ACM SIGKDD国际知识发现与数据挖掘会议论文集》（Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD)）。ACM，262 - 270。

[31] Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, Brandon Westover, Qiang Zhu, Jesin Zakaria, and Eamonn Keogh. 2013. Addressing Big Data Time Series: Mining Trillions of Time Series Subsequences Under Dynamic Time Warping. ACM Transactions on Knowledge Discovery from Data 7, 3 (2013), 1-31.

[31] 塔纳温·拉克坦马农（Thanawin Rakthanmanon）、比尔森·坎帕纳（Bilson Campana）、阿卜杜拉·穆恩（Abdullah Mueen）、古斯塔沃·巴蒂斯塔（Gustavo Batista）、布兰登·韦斯托弗（Brandon Westover）、朱强（Qiang Zhu）、杰辛·扎卡里亚（Jesin Zakaria）和埃蒙·基奥（Eamonn Keogh）。2013年。应对大数据时间序列：在动态时间规整下挖掘数万亿个时间序列子序列。《ACM数据知识发现汇刊》7, 3 (2013)，1 - 31。

[32] Hiroaki Sakoe and Seibi Chiba. 1978. Dynamic programming algorithm optimization for spoken word recognition. IEEE transactions on acoustics, speech, and signal processing 26, 1 (1978), 43-49.

[32] 酒井浩明（Hiroaki Sakoe）和千叶诚比（Seibi Chiba）。1978年。用于语音识别的动态规划算法优化。《IEEE声学、语音和信号处理汇刊》26, 1 (1978)，43 - 49。

[33] Yilin Shen, Yanping Chen, Eamonn Keogh, and Hongxia Jin. 2018. Accelerating time series searching with large uniform scaling. In Proceedings of the 2018 SIAM International Conference on Data Mining (ICDM). SIAM, 234-242.

[33] 沈怡琳（Yilin Shen）、陈艳萍（Yanping Chen）、埃蒙·基奥（Eamonn Keogh）和金红霞（Hongxia Jin）。2018年。利用大均匀缩放加速时间序列搜索。《2018年SIAM国际数据挖掘会议论文集》（ICDM）。SIAM，234 - 242。

[34] Jin Shieh and Eamonn Keogh. 2008. iSAX: indexing and mining terabyte sized time series. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). ACM, 623-631.

[34] 谢晋（Jin Shieh）和埃蒙·基奥（Eamonn Keogh）。2008年。iSAX：用于索引和挖掘TB级时间序列。《第14届ACM SIGKDD国际知识发现与数据挖掘会议论文集》（KDD）。ACM，623 - 631。

[35] Tarique Siddiqui, Paul Luh, Zesheng Wang, Karrie Karahalios, and Aditya Parameswaran. 2020. Shapesearch: A flexible and efficient system for shape-based exploration of trendlines. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD). ACM, 51-65.

[35] 塔里克·西迪基（Tarique Siddiqui）、保罗·卢（Paul Luh）、王泽生（Zesheng Wang）、卡里·卡拉哈廖斯（Karrie Karahalios）和阿迪亚·帕拉梅斯瓦兰（Aditya Parameswaran）。2020年。Shapesearch：一种灵活高效的基于形状的趋势线探索系统。《2020年ACM SIGMOD国际数据管理会议论文集》（SIGMOD）。ACM，51 - 65。

[36] Mario Giovanni Terzano, Liborio Parrino, Adriano Sherieri, Ronald Chervin, Sudhansu Chokroverty, Christian Guilleminault, Max Hirshkowitz, Mark Ma-howald, Harvey Moldofsky, Agostino Rosa, et al. 2001. Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep. Sleep medicine 2, 6 (2001), 537-554.

[36] 马里奥·乔瓦尼·特尔扎诺（Mario Giovanni Terzano）、利博里奥·帕里诺（Liborio Parrino）、阿德里亚诺·谢里耶里（Adriano Sherieri）、罗纳德·切尔文（Ronald Chervin）、苏汉苏·乔克罗弗蒂（Sudhansu Chokroverty）、克里斯蒂安·吉列米诺（Christian Guilleminault）、马克斯·赫什科维茨（Max Hirshkowitz）、马克·马霍尔德（Mark Ma - howald）、哈维·莫尔多夫斯基（Harvey Moldofsky）、阿戈斯蒂诺·罗莎（Agostino Rosa）等。2001年。人类睡眠中周期性交替模式（CAP）评分的图谱、规则和记录技术。《睡眠医学》2, 6 (2001)，537 - 554。

[37] Yang Wang, Peng Wang, Jian Pei, Wei Wang, and Sheng Huang. 2013. A data-adaptive and dynamic segmentation index for whole matching on time series. Proceedings of the VLDB Endowment 6, 10 (2013), 793-804.

[37] 王洋（Yang Wang）、王鹏（Peng Wang）、裴健（Jian Pei）、王伟（Wei Wang）和黄胜（Sheng Huang）。2013年。一种用于时间序列整体匹配的数据自适应动态分割索引。《VLDB捐赠会议论文集》6, 10 (2013)，793 - 804。

[38] Zeyu Wang, Qitong Wang, Peng Wang, Themis Palpanas, and Wei Wang. 2023. Dumpy: A compact and adaptive index for large data series collections. Proceedings of the ACM on Management of Data 1, 1 (2023), 1-27.

[38] 王泽宇（Zeyu Wang）、王启同（Qitong Wang）、王鹏（Peng Wang）、西弥斯·帕尔帕纳斯（Themis Palpanas）和王伟（Wei Wang）。2023年。Dumpy：一种用于大数据序列集合的紧凑自适应索引。《ACM数据管理汇刊》1, 1 (2023)，1 - 27。

[39] Jiaye Wu, Peng Wang, Ningting Pan, Chen Wang, Wei Wang, and Jianmin Wang. 2019. Kv-match: A subsequence matching approach supporting normalization and time warping. In IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 866-877.

[39] 吴佳烨（Jiaye Wu）、王鹏（Peng Wang）、潘宁婷（Ningting Pan）、王晨（Chen Wang）、王伟（Wei Wang）和王建民（Jianmin Wang）。2019年。Kv - match：一种支持归一化和时间规整的子序列匹配方法。《IEEE第35届国际数据工程会议》（ICDE）。IEEE，866 - 877。

[40] Byoung-Kee Yi and Christos Faloutsos. 2000. Fast Time Sequence Indexing for Arbitrary Lp Norms. In Proceedings of the 26th International Conference on Very Large Data Bases (VLDB). Morgan Kaufmann Publishers Inc., 385-394.

[40] 柳炳基（Byoung - Kee Yi）和克里斯托斯·法沃索斯（Christos Faloutsos）。2000年。任意Lp范数下的快速时间序列索引。《第26届国际超大型数据库会议论文集》（VLDB）。摩根·考夫曼出版社，385 - 394。

[41] Yunyue Zhu and Dennis Shasha. 2003. Warping indexes with envelope transforms for query by humming. In Proceedings of the 2003 ACM SIGMOD international conference on Management of data (SIGMOD). ACM, 181-192.

[41] 朱云岳（Yunyue Zhu）和丹尼斯·沙莎（Dennis Shasha）。2003年。用于哼唱查询的带包络变换的规整索引。《2003年ACM SIGMOD国际数据管理会议论文集》（SIGMOD）。ACM，181 - 192。

[42] Kostas Zoumpatianos and Themis Palpanas. 2018. Data series management: Fulfilling the need for big sequence analytics. In IEEE 34th International Conference on Data Engineering (ICDE). IEEE, 1677-1678.

[42] 科斯塔斯·祖姆帕蒂亚诺斯（Kostas Zoumpatianos）和西弥斯·帕尔帕纳斯（Themis Palpanas）。2018年。数据序列管理：满足大数据序列分析的需求。《IEEE第34届国际数据工程会议》（ICDE）。IEEE，1677 - 1678。