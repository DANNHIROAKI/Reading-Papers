# Bottom-Up Computation of Sparse and Iceberg CUBEs

# 稀疏冰山数据立方体（Sparse and Iceberg CUBEs）的自底向上计算

Kevin Beyer

凯文·拜尔（Kevin Beyer）

Computer Sciences Department University of Wisconsin - Madison

威斯康星大学麦迪逊分校计算机科学系

beyer@cs.wisc.edu

Raghu Ramakrishnan

拉古·拉马克里什南（Raghu Ramakrishnan）

Computer Sciences Department

计算机科学系

University of Wisconsin - Madison

威斯康星大学麦迪逊分校

raghu@cs.wisc.edu

## Abstract

## 摘要

We introduce the Iceberg-CUBE problem as a reformulation of the datacube (CUBE) problem. The Iceberg-CUBE problem is to compute only those group-by partitions with an aggregate value (e.g., count) above some minimum support threshold. The result of Iceberg-CUBE can be used (1) to answer group-by queries with a clause such as HAVING COUNT(*) $>  = X$ ,where $X$ is greater than the threshold, (2) for mining multidimensional association rules, and (3) to complement existing strategies for identifying interesting subsets of the CUBE for precomputation.

我们将冰山数据立方体（Iceberg-CUBE）问题作为数据立方体（CUBE）问题的一种重新表述引入。冰山数据立方体问题是仅计算那些聚合值（例如计数）高于某个最小支持阈值的分组分区。冰山数据立方体的结果可用于（1）回答带有诸如 HAVING COUNT(*) $>  = X$ 子句的分组查询，其中 $X$ 大于该阈值；（2）挖掘多维关联规则；（3）补充现有的用于识别数据立方体中有趣子集以进行预计算的策略。

We present a new algorithm (BUC) for Iceberg-CUBE computation. BUC builds the CUBE bottom-up; i.e., it builds the CUBE by starting from a group-by on a single attribute, then a group-by on a pair of attributes, then a group-by on three attributes, and so on. This is the opposite of all techniques proposed earlier for computing the CUBE, and has an important practical advantage: BUC avoids computing the larger group-bys that do not meet minimum support. The pruning in BUC is similar to the pruning in the Apriori algorithm for association rules, except that BUC trades some pruning for locality of reference and reduced memory requirements. BUC uses the same pruning strategy when computing sparse, complete CUBEs.

我们提出了一种用于冰山数据立方体计算的新算法（BUC）。BUC 自底向上构建数据立方体；即，它从对单个属性进行分组开始构建数据立方体，然后对一对属性进行分组，接着对三个属性进行分组，依此类推。这与之前提出的所有计算数据立方体的技术相反，并且具有一个重要的实际优势：BUC 避免计算不满足最小支持度的较大分组。BUC 中的剪枝与关联规则的 Apriori 算法中的剪枝类似，不同之处在于 BUC 为了引用局部性和减少内存需求而进行了一些剪枝权衡。BUC 在计算稀疏、完整的数据立方体时使用相同的剪枝策略。

We present a thorough performance evaluation over a broad range of workloads. Our evaluation demonstrates that (in contrast to earlier assumptions) minimizing the aggregations or the number of sorts is not the most important aspect of the sparse CUBE problem. The pruning in BUC, combined with an efficient sort method, enables BUC to outperform all previous algorithms for sparse CUBEs, even for computing entire CUBEs, and to dramatically improve Iceberg-CUBE computation.

我们对广泛的工作负载进行了全面的性能评估。我们的评估表明（与之前的假设相反），最小化聚合或排序次数并不是稀疏数据立方体问题最重要的方面。BUC 中的剪枝与高效的排序方法相结合，使 BUC 在计算稀疏数据立方体时优于所有先前的算法，即使是计算整个数据立方体，也能显著改进冰山数据立方体的计算。

## 1 Introduction

## 1 引言

Decision support systems frequently precompute many aggregates to improve the response time of aggregation queries. The datacube (CUBE) operator [6] generalizes the standard GROUP-BY operator to compute aggregates for every combination of GROUP BY attributes. For example, consider a relation Transaction(Product, Store, Customer, Sales). The sum of Sales over the CUBE of Product, Store, and Customer produces the sum of Sales for the entire relation (i.e., no GROUP BY), for each Product (GROUP BY Product), for each Store, for each Customer, for each pair: (Product, Store), (Product, Customer), and (Store, Customer), and finally for each (Product, Store, Customer) combination. In OLAP parlance, the grouping attributes are called dimensions, the attributes that are aggregated are called measures, and one particular GROUP BY (e.g., Product, Store) in a CUBE computation is sometimes called a cuboid or simply a group-by.

决策支持系统经常预计算许多聚合值以提高聚合查询的响应时间。数据立方体（CUBE）运算符 [6] 将标准的 GROUP - BY 运算符进行了推广，以计算 GROUP BY 属性的每个组合的聚合值。例如，考虑一个关系 Transaction(Product, Store, Customer, Sales)。对 Product、Store 和 Customer 的数据立方体计算 Sales 的总和，会得出整个关系的 Sales 总和（即不进行 GROUP BY）、每个 Product 的 Sales 总和（GROUP BY Product）、每个 Store 的 Sales 总和、每个 Customer 的 Sales 总和、每对 (Product, Store)、(Product, Customer) 和 (Store, Customer) 的 Sales 总和，最后是每个 (Product, Store, Customer) 组合的 Sales 总和。在联机分析处理（OLAP）术语中，分组属性称为维度，被聚合的属性称为度量，数据立方体计算中的一个特定 GROUP BY（例如 Product, Store）有时称为方体或简称为分组。

The basic CUBE problem is to compute all of the aggregates as efficiently as possible. Simultaneously computing the aggregates offers the opportunity to share partitioning and aggregation costs between various group-bys. The chief difficulty is that the CUBE problem is exponential in the number of dimensions: for $d$ dimensions, ${2}^{d}$ group-bys are computed. In addition, the size of each group-by depends upon the cardinality of its dimensions. If every store sold every product, then the (Product, Store) group-by would have $\left| \text{Product}\right|  \times  \left| \text{Store}\right|$ result tuples. However,as the number of dimensions or the cardinalities increase, the product of the cardinalities grossly exceeds the (fixed) size of the input relation for many of the group-bys. Even in our small example, if the data comes from a large department store, it is highly unlikely that a given customer purchased even $5\%$ of the products, or shopped in more than 1% of the stores.

基本的数据立方体问题是尽可能高效地计算所有聚合值。同时计算聚合值提供了在各种分组之间共享分区和聚合成本的机会。主要困难在于数据立方体问题的维度数量是指数级的：对于 $d$ 个维度，要计算 ${2}^{d}$ 个分组。此外，每个分组的大小取决于其维度的基数。如果每个商店都销售每种产品，那么 (Product, Store) 分组将有 $\left| \text{Product}\right|  \times  \left| \text{Store}\right|$ 个结果元组。然而，随着维度数量或基数的增加，许多分组的基数乘积会大大超过输入关系的（固定）大小。即使在我们的小例子中，如果数据来自一家大型百货公司，那么一个给定的客户购买甚至 $5\%$ 的产品或者在超过 1% 的商店购物的可能性非常小。

When the product of the cardinalities for a group-by is large relative to the number of tuples that actually appear in the result, we say the group-by is sparse. When the number of sparse group-bys is large relative to the number of total number of group-bys, we say the CUBE is sparse. As is well-recognized, given the large result size for the entire CUBE, especially on sparse datasets, it is important to identify (and precompute) subsets of interest.

当一个分组的基数乘积相对于实际出现在结果中的元组数量较大时，我们称该分组是稀疏的。当稀疏分组的数量相对于分组总数较大时，我们称数据立方体是稀疏的。众所周知，鉴于整个数据立方体的结果规模很大，尤其是在稀疏数据集上，识别（并预计算）感兴趣的子集非常重要。

This paper addresses CUBE computation over sparse CUBEs and makes the following contributions:

本文讨论了稀疏数据立方体上的数据立方体计算，并做出了以下贡献：

1. We introduce a variant of the CUBE problem, called Iceberg-CUBE in the spirit of [5], that allows us to selectively compute only those partitions that satisfy a user-specified aggregate condition (similar to SQL's HAVING clause). The Iceberg-CUBE formulation can be viewed as a new dynamic subset selection strategy, complementing earlier approaches that statically identify group-bys (rather than partitions) to be precomputed $\left\lbrack  {{10},8,7,3,{18}}\right\rbrack$ . The Iceberg-CUBE formulation also identifies precisely the subset of the CUBE that is required to mine multidimensional association rules in the framework described in [11].

1. 我们引入了CUBE问题的一个变体，按照文献[5]的思路将其称为冰山CUBE（Iceberg - CUBE），它允许我们有选择地仅计算那些满足用户指定聚合条件（类似于SQL的HAVING子句）的分区。冰山CUBE公式可以被视为一种新的动态子集选择策略，它补充了早期的方法，早期方法是静态地识别要预计算的分组依据（而不是分区） $\left\lbrack  {{10},8,7,3,{18}}\right\rbrack$ 。冰山CUBE公式还精确地确定了在文献[11]所描述的框架中挖掘多维关联规则所需的CUBE子集。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fce provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers or to redistribute to lists. requires prior specific permission and/or a fee.

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和第一页的完整引用信息。否则，如需复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。

SIGMOD '99 Philadelphia PA

1999年SIGMOD会议 宾夕法尼亚州费城

Copyright ACM 1999 1-58113-084-8/99/05...\$5.00

版权归美国计算机协会（ACM）1999年所有 1 - 58113 - 084 - 8/99/05... 5.00美元

<!-- Footnote -->

---

2. We present a simple and efficient algorithm, called BUC, for the Iceberg-CUBE problem. BUC proceeds bottom-up (it builds the CUBE by starting from a group-by on a single attribute, then a group-by on a pair of attributes, and so on) in contrast to all prior CUBE algorithms. This feature enables the Iceberg aggregate selections to be pushed into the CUBE computation easily. We also outline an extension to compute precisely the CUBE subset identified for precomputation by the PBS algorithm [18].

2. 我们为冰山CUBE问题提出了一种简单而高效的算法，称为自底向上立方体算法（Bottom - Up Cube，BUC）。与所有先前的CUBE算法不同，BUC采用自底向上的方式（它从对单个属性进行分组开始构建CUBE，然后对一对属性进行分组，依此类推）。这一特性使得冰山聚合选择能够轻松地融入CUBE计算中。我们还概述了一种扩展方法，用于精确计算由PBS算法[18]确定要预计算的CUBE子集。

3. We present an extensive performance evaluation based upon a complete implementation of BUC. We focus our comparison on the MemoryCube algorithm presented in [14], which was shown to be superior to earlier algorithms for computing sparse CUBEs. We show that BUC outperforms MemoryCube on a wide range of synthetic and real datasets, even when computing the full CUBE. Further, since MemoryCube does not exploit aggregate selections, BUC outperforms it significantly for Iceberg-CUBE computation.

3. 我们基于BUC的完整实现进行了广泛的性能评估。我们将比较重点放在文献[14]中提出的内存立方体算法（MemoryCube）上，该算法在计算稀疏CUBE方面优于早期算法。我们表明，在各种合成和真实数据集上，即使是计算完整的CUBE，BUC的性能也优于MemoryCube。此外，由于MemoryCube没有利用聚合选择，因此在冰山CUBE计算中，BUC的性能明显优于它。

The rest of this paper is organized as follows: Section 2 discusses sparse CUBEs, and we argue that the basic CUBE problem is not appropriate for high dimensions. In Section 3, we define the Iceberg-CUBE problem. Section 4 describes the previous work on computing the CUBE. Most previous work does not scale to sparse, high-dimensional CUBEs [14], and none can directly take advantage of the minimum support threshold in Iceberg-CUBE. In Section 5, we present a new algorithm for CUBE and Iceberg-CUBE computation called Bottom-Up Cube (BUC). Our performance analysis is described in Section 6. Our evaluation demonstrates that (in contrast to earlier assumptions) minimizing the aggregations or the number of sorts is not the most important aspect of the sparse CUBE problem. The pruning in BUC, combined with an efficient sort method, turns out to be the key to the performance gains, even for full CUBEs. We present our conclusions in Section 7.

本文的其余部分组织如下：第2节讨论稀疏CUBE，我们认为基本的CUBE问题不适用于高维情况。在第3节中，我们定义了冰山CUBE问题。第4节描述了先前在计算CUBE方面的工作。大多数先前的工作无法扩展到稀疏、高维的CUBE[14]，并且没有一种方法可以直接利用冰山CUBE中的最小支持阈值。在第5节中，我们提出了一种用于计算CUBE和冰山CUBE的新算法，称为自底向上立方体算法（BUC）。我们的性能分析在第6节中描述。我们的评估表明（与早期的假设相反），最小化聚合或排序次数并不是稀疏CUBE问题最重要的方面。事实证明，BUC中的剪枝与高效的排序方法相结合，是实现性能提升的关键，即使对于完整的CUBE也是如此。我们在第7节中给出结论。

## 2 Motivation

## 2 动机

Ross and Srivastava computed the full CUBE on a real nine-dimensional dataset containing weather conditions at various weather stations on land for September 1985 [14, 9]. The dataset had 1,015,367 tuples ( $\sim  {39}\mathrm{{MB}}$ ). The CUBE on this dataset produces 210,343,580 tuples $\left( { \sim  8\mathrm{{GB}}}\right)$ -more than 200 times the input size!

罗斯（Ross）和斯里瓦斯塔瓦（Srivastava）对一个真实的九维数据集计算了完整的CUBE，该数据集包含1985年9月陆地上各个气象站的天气状况[14, 9]。该数据集有1,015,367个元组 ( $\sim  {39}\mathrm{{MB}}$ )。对这个数据集计算CUBE会产生210,343,580个元组 $\left( { \sim  8\mathrm{{GB}}}\right)$ —— 是输入大小的200多倍！

We break down the output size distribution for this dataset in Figure 1. In both the graphs,the $\mathrm{x}$ -axis is the number of output tuples in a group-by relative to the input size. Figure 1a shows the number of group-bys that are smaller than a given relative size, and Figure 1b shows the amount of space (relative to the input size) needed to build all the group-bys that are less than a given size. The graphs convey a number of interesting points:

我们在图1中分解了这个数据集的输出大小分布。在这两个图中， $\mathrm{x}$ 轴表示分组依据中相对于输入大小的输出元组数量。图1a显示了小于给定相对大小的分组依据的数量，图1b显示了构建所有小于给定大小的分组依据所需的空间（相对于输入大小）。这些图传达了一些有趣的信息：

- About ${20}\%$ of the group-bys performed very little aggregation (all of the group-bys with relative size of nearly 1). These group-bys are simply a projection of the input. (If the data were uniform and uncorrelated, then nearly ${60}\%$ of the group-bys would perform little to no aggregation.) About ${60}\%$ of the group-bys aggregated an average of no more than 4 tuples.

- 大约 ${20}\%$ 的分组依据几乎没有进行聚合（所有相对大小接近1的分组依据）。这些分组依据只是输入的投影。（如果数据是均匀且不相关的，那么几乎 ${60}\%$ 的分组依据将几乎不进行聚合。）大约 ${60}\%$ 的分组依据平均聚合不超过4个元组。

- Computing all group-bys that aggregate at least two input tuples on average (group-bys with relative size of 0.5 ) requires about 50 times the input size versus the 200 times for the full CUBE. Requiring an average of 10 tuples to be aggregated (i.e., 0.001% of the input tuples) shrinks the space requirement to 5 times the input.

- 计算所有平均聚合至少两个输入元组的分组（相对大小为 0.5 的分组）所需的空间约为输入大小的 50 倍，而完整的 CUBE 则需要 200 倍。若要求平均聚合 10 个元组（即输入元组的 0.001%），则空间需求会缩减至输入大小的 5 倍。

- As noted in [14], simply writing the entire output to disk can take an inordinate amount of time, and can easily dominate the cost of computing the CUBE. By selecting the group-bys (or group-by partitions) that perform at least a little aggregation, the output time can be significantly reduced.

- 正如文献 [14] 中所指出的，简单地将整个输出写入磁盘可能会花费大量时间，并且很容易在计算 CUBE 的成本中占据主导地位。通过选择至少进行少量聚合的分组（或分组分区），可以显著减少输出时间。

Because the space requirements for large CUBEs are so high, often we cannot realistically compute the full CUBE. We need a way to choose what portion of the CUBE to compute. A number of researchers have proposed computing a subset of the group-bys instead of the entire datacube $\lbrack {10}$ , $8,7,3,{18}\rbrack$ . The algorithms choose the group-bys to compute based on the available disk space, the expected size of the group-by, and the expected benefit of precomputing the aggregate. Under certain common conditions, [18] presents an algorithm called PBS that chooses to materialize the smallest group-bys (i.e., the fewest result tuples, which is the same as the group-bys that perform the most aggregation).

由于大型 CUBE 的空间需求非常高，通常我们实际上无法计算完整的 CUBE。我们需要一种方法来选择计算 CUBE 的哪一部分。许多研究人员提议计算分组的一个子集，而不是整个数据立方体 $\lbrack {10}$ , $8,7,3,{18}\rbrack$ 。这些算法根据可用磁盘空间、分组的预期大小以及预计算聚合的预期收益来选择要计算的分组。在某些常见条件下，文献 [18] 提出了一种名为 PBS 的算法，该算法选择物化最小的分组（即结果元组最少的分组，这与执行最多聚合的分组相同）。

Choosing a subset of the group-bys to compute is certainly a reasonable way to reduce the output of full CUBE computation, and we show in Section 6 that BUC works particularly well with PBS. However, statically choosing a subset of the group-bys is not the only way to reduce the output; the next section describes a new alternative.

选择要计算的分组子集无疑是减少完整 CUBE 计算输出的合理方法，并且我们将在第 6 节中表明，BUC 与 PBS 配合使用效果特别好。然而，静态地选择分组子集并不是减少输出的唯一方法；下一节将介绍一种新的替代方法。

## 3 The Iceberg-CUBE Problem

## 3 冰山 - CUBE 问题

The Iceberg-CUBE problem is to compute all group-by partitions for every combination of the grouping attributes that satisfy an aggregate selection condition, as in the HAVING clause of an SQL query. For concreteness, we will discuss the condition that a partition contain at least $N$ tuples; other aggregate selections can be handled as well, as described in Section 5.2. The parameter $N$ is called the minimum support of a partition, or minsup for short. Iceberg-CUBE with minsup of 1 is exactly the same as the original CUBE problem. Iceberg-CUBE with a minsup of $N$ is easily expressed in SQL with the CUBE BY clause:

冰山 - CUBE 问题是为满足聚合选择条件的分组属性的每个组合计算所有分组分区，就像 SQL 查询的 HAVING 子句中那样。为了具体说明，我们将讨论一个分区至少包含 $N$ 个元组的条件；其他聚合选择也可以处理，如第 5.2 节所述。参数 $N$ 称为分区的最小支持度，简称 minsup。minsup 为 1 的冰山 - CUBE 与原始的 CUBE 问题完全相同。minsup 为 $N$ 的冰山 - CUBE 可以很容易地用 SQL 的 CUBE BY 子句表示：

SELECT A,B,C,COUNT (*),SUM (X)

SELECT A,B,C,COUNT (*),SUM (X)

FROM R

CUBE BY A,B,C

CUBE BY A,B,C

HAVING COUNT $\left( *\right)  >  =$ N

HAVING COUNT $\left( *\right)  >  =$ N

<!-- Media -->

<!-- figureText: 500 200 Sum of Group-bys Sizes $< X$ 150 100 50 0 0 0.2 0.4 0.6 0.8 Group-by Size / Input Size (b) Total Space Blow-up Count of Group-bys Sizes $< X$ 400 300 200 100 0 0.2 0.4 0.6 0.8 1 Group-by Size / Input Size (a) Group-by Size -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_2.jpg?x=106&y=70&w=1500&h=548&r=0"/>

Figure 1: Space requirements for the weather dataset

图 1：天气数据集的空间需求

<!-- Media -->

The precomputed result of an Iceberg-CUBE can be used to answer GROUP BY queries on any combination of the dimensions $(A,B$ ,and $C$ ,in this case) that also contains a HAVING COUNT(*) $>  = M$ ,where $M >  = N$ . The count does not need to be stored if all queries (explicitly or implicitly) contain HAVING COUNT $\left( *\right)  >  = \mathrm{N}$ . That is,the user is simply not interested in small group-by partitions.

冰山 - CUBE 的预计算结果可用于回答关于维度 $(A,B$ 和 $C$ （在这种情况下）的任何组合的 GROUP BY 查询，这些查询还包含 HAVING COUNT(*) $>  = M$ ，其中 $M >  = N$ 。如果所有查询（显式或隐式）都包含 HAVING COUNT $\left( *\right)  >  = \mathrm{N}$ ，则无需存储计数。也就是说，用户根本不关心小的分组分区。

Iceberg-CUBE is not an entirely new problem, although this is the first time that it has been proposed for CUBE queries. The mining of multi-dimensional association rules (MDAR) uses a notion of minimum support that is equivalent to Iceberg-CUBE [11]. MDARs have the form $X \Rightarrow  Y.X$ is called the body of the rule,and $Y$ is called the head. $X$ and $Y$ are sets of conjunctive predicates. For the purposes of this discussion, each predicate essentially has the form attribute $=$ value,although they are a bit more general in [11]. The support for a rule $X \Rightarrow  Y$ in a relation $R$ is the probability that a tuple of $R$ contains both $X$ and $Y$ . In other words,the support is the count of tuples that are true for both $X$ and $Y$ divided by the number of tuples in $R$ . The confidence of a rule is the probability that a tuple of $R$ contains $Y$ given that the tuple contains $X$ (i.e., $\operatorname{Prob}\left( {Y \mid  X}\right)  = \operatorname{count}\left( {X\text{and}Y}\right) /\operatorname{count}\left( X\right) )$ . The goal of mining MDARs is to find rules with a support of at least the minsup and a confidence of at least minconf.

冰山立方体（Iceberg-CUBE）并非一个全新的问题，尽管这是它首次被应用于立方体（CUBE）查询。多维关联规则（Multi-dimensional Association Rules，MDAR）的挖掘使用了一种最小支持度的概念，该概念等同于冰山立方体 [11]。多维关联规则具有 $X \Rightarrow  Y.X$ 的形式，其中 $X \Rightarrow  Y.X$ 被称为规则的主体，$Y$ 被称为规则的头部。$X$ 和 $Y$ 是合取谓词的集合。在本次讨论中，每个谓词本质上具有属性 $=$ 值的形式，不过在文献 [11] 中它们更为通用。关系 $R$ 中规则 $X \Rightarrow  Y$ 的支持度是 $R$ 中的一个元组同时包含 $X$ 和 $Y$ 的概率。换句话说，支持度是同时满足 $X$ 和 $Y$ 的元组数量除以 $R$ 中的元组总数。规则的置信度是在 $R$ 中的一个元组包含 $X$ 的条件下，该元组包含 $Y$ 的概率（即 $\operatorname{Prob}\left( {Y \mid  X}\right)  = \operatorname{count}\left( {X\text{and}Y}\right) /\operatorname{count}\left( X\right) )$）。挖掘多维关联规则的目标是找到支持度至少为最小支持度（minsup）且置信度至少为最小置信度（minconf）的规则。

The first step in mining MDARs, as with traditional association rules ${}^{1}\left\lbrack  2\right\rbrack$ ,is to find the rules that meet minimum support. This is precisely the Iceberg-CUBE problem where $N = \left| R\right|  * {\text{minsup}}_{\text{MDAR }}$ . The results from Iceberg-CUBE can then be combined to find the rules that meet minimum confidence.

与传统关联规则 ${}^{1}\left\lbrack  2\right\rbrack$ 一样，挖掘多维关联规则的第一步是找出满足最小支持度的规则。这正是冰山立方体问题，其中 $N = \left| R\right|  * {\text{minsup}}_{\text{MDAR }}$。然后可以将冰山立方体的结果进行组合，以找出满足最小置信度的规则。

A precomputed Iceberg-CUBE is also useful for computing iceberg queries [5]. An iceberg query computes a single group-by and eliminates all tuples with an aggregate value below some threshold. For example:

预先计算的冰山立方体（Iceberg-CUBE）对于计算冰山查询 [5] 也很有用。冰山查询计算单个分组操作，并消除所有聚合值低于某个阈值的元组。例如：

$$
\text{SELECT A,B,C,COUNT (*)}
$$

FROM R

$$
\text{GROUP BY A, B, C}
$$

$$
\text{HAVING COUNT}\left( *\right)  >  = \mathrm{N}
$$

So the Iceberg-CUBE is essentially an iceberg query over a CUBE. Although we do not discuss this point further, the techniques of [5] can be used to refine BUC in some cases with large partitions.

因此，冰山立方体（Iceberg-CUBE）本质上是对立方体（CUBE）进行的冰山查询。尽管我们不再进一步讨论这一点，但在某些具有大分区的情况下，可以使用文献 [5] 中的技术来改进 BUC 算法。

## 4 Previous CUBE Algorithms

## 4 先前的立方体（CUBE）算法

The CUBE was introduced in [6], and they outlined some useful properties for CUBE computation: (1) minimize data movement, (2) map string dimension attributes [and other types] to integers between zero and the cardinality of the attribute, and (3) use parallelism. Mapping dimensions to integers reduces space requirements (i.e., no long strings), eliminates expensive type interpretation in the CUBE code, and packing the domain between zero and the cardinality allows the dimensions to be used as array subscripts.

立方体（CUBE）在文献 [6] 中被引入，并且文中概述了一些对立方体计算有用的属性：（1）最小化数据移动；（2）将字符串维度属性（以及其他类型）映射到零和该属性的基数之间的整数；（3）使用并行性。将维度映射为整数可以减少空间需求（即，无需长字符串），消除立方体代码中昂贵的类型解释，并且将域范围限制在零和基数之间允许将维度用作数组下标。

Three types of aggregate functions were identified. Consider aggregating a set of tuples $T$ . Let $\left\{  {{S}_{i} \mid  i = 1\ldots n}\right\}$ be a any complete set of disjoint subsets of $T$ such that $\mathop{\bigcup }\limits_{i}{S}_{i} = T$ ,and $\mathop{\bigcap }\limits_{i}{S}_{i} = \{ \} .$

确定了三种类型的聚合函数。考虑对一组元组 $T$ 进行聚合。设 $\left\{  {{S}_{i} \mid  i = 1\ldots n}\right\}$ 是 $T$ 的任意一组完全不相交的子集，使得 $\mathop{\bigcup }\limits_{i}{S}_{i} = T$，并且 $\mathop{\bigcap }\limits_{i}{S}_{i} = \{ \} .$

- An aggregate function $F$ is distributive if there is a function $G$ such that $F\left( T\right)  = G\left( \left\{  {F\left( {S}_{i}\right)  \mid  i = 1\ldots n}\right\}  \right)$ . SUM,MIN,and MAX are distributive with $G = F$ . COUNT is distributive with $G = \mathrm{{SUM}}$ .

- 如果存在一个函数 $G$ 使得 $F\left( T\right)  = G\left( \left\{  {F\left( {S}_{i}\right)  \mid  i = 1\ldots n}\right\}  \right)$，则聚合函数 $F$ 是可分配的。SUM、MIN 和 MAX 是可分配的，其中 $G = F$。COUNT 是可分配的，其中 $G = \mathrm{{SUM}}$。

- An aggregate function $F$ is algebraic if there is a $M$ - tuple valued function $G$ and a function $H$ such that $F\left( T\right)  = H\left( \left\{  {G\left( {S}_{i}\right)  \mid  i = 1\ldots n}\right\}  \right)$ ,and $M$ is constant regardless of $\left| T\right|$ and $n$ . All distributive functions are algebraic, as are Average, standard deviation, MaxN, and MinN. For Average, $G$ produces the sum and count, and $H$ divides the result.

- 如果存在一个 $M$ 元值函数 $G$ 和一个函数 $H$ 使得 $F\left( T\right)  = H\left( \left\{  {G\left( {S}_{i}\right)  \mid  i = 1\ldots n}\right\}  \right)$，并且 $M$ 与 $\left| T\right|$ 和 $n$ 无关，则聚合函数 $F$ 是代数的。所有可分配函数都是代数的，平均值、标准差、MaxN 和 MinN 也是代数的。对于平均值，$G$ 产生总和和计数，$H$ 对结果进行除法运算。

---

<!-- Footnote -->

${}^{1}$ Note that the traditional association rule problem can be mapped to the MDAR problem be making each item an attribute (i.e., a dimension) with a value of zero or one (and using predicates with a value of 1 ). Unfortunately, BUC is ill-suited for this problem because (1) the dimensionality is extremely high (50,000 items is not uncommon), (2) the cardinality of every dimension is extremely low (only two values), and (3) the dimensions tend to be highly skewed.

${}^{1}$ 请注意，传统的关联规则问题可以通过将每个项作为一个取值为 0 或 1 的属性（即一个维度）（并使用取值为 1 的谓词）映射到多维关联规则挖掘（MDAR）问题。不幸的是，BUC 算法不太适合这个问题，因为（1）维度极高（50,000 个项并不罕见），（2）每个维度的基数极低（只有两个值），以及（3）各维度往往高度倾斜。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: ABCD ACD BCD BC BD CD C all ABC ABD AB AC AD A B -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_3.jpg?x=110&y=79&w=499&h=382&r=0"/>

Figure 2: 4-Dimensional Lattice

图 2：四维格

<!-- figureText: biggest $\rightarrow$ ABCD $\rightarrow$ smallest ACD BCD BC BD CD C D all ABC ABD AB AC AD A B -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_3.jpg?x=628&y=76&w=495&h=386&r=0"/>

Figure 3: Sample Processing Tree

图 3：示例处理树

<!-- figureText: 5 ABCD 8 ACD 12 BCD 11 BC 13 BD 15 CD 14 C 16 D 1 all 4 ABC 6 ABD 7 AC 9 AD 2 A 10 B -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_3.jpg?x=1141&y=89&w=497&h=373&r=0"/>

Figure 4: BUC Processing Tree

图 4：BUC 处理树

<!-- Media -->

- An aggregate function $F$ is holistic if it not algebraic. For example, Median and Rank are holistic.

- 如果聚合函数 $F$ 不是代数函数，则它是整体函数。例如，中位数（Median）和排名（Rank）就是整体函数。

Algebraic functions have the key property that more detailed aggregates (i.e., more dimensions) can be used to compute less detailed aggregates. This property induces a partial ordering (i.e., a lattice) on all of the group-bys of the CUBE. A group-by is called a child of some parent group-by if the parent can be used to compute the child (and no other group-by is between the parent and the child). Figure 2 depicts a sample lattice where $\mathrm{A},\mathrm{B},\mathrm{C}$ ,and $\mathrm{D}$ are dimensions, nodes represent group-bys, and the arcs show the parent-child relationship.

代数函数具有这样的关键特性：更详细的聚合（即更多的维度）可用于计算不太详细的聚合。这一特性在 CUBE 的所有分组依据上诱导出一个偏序（即一个格）。如果某个父分组依据可用于计算子分组依据（且在父分组依据和子分组依据之间没有其他分组依据），则该子分组依据被称为该父分组依据的子项。图 2 展示了一个示例格，其中 $\mathrm{A},\mathrm{B},\mathrm{C}$ 和 $\mathrm{D}$ 是维度，节点表示分组依据，弧线表示父子关系。

All of the algorithms use this lattice view of the problem. The goal is to take advantage of as much commonality as possible between a parent and child. In general, the algorithms recognize that group-bys with common attributes can share partitions, sorts, or partial sorts. The algorithms differ on exactly how they exploit this commonality. The following subsections provide details on each of the previous proposed algorithms.

所有算法都使用问题的这种格视图。目标是尽可能利用父子之间的共性。一般来说，这些算法认识到具有共同属性的分组依据可以共享分区、排序或部分排序。这些算法在如何利用这种共性方面有所不同。以下小节详细介绍之前提出的每个算法。

### 4.1 PipeSort, PipeHash, and Overlap

### 4.1 管道排序（PipeSort）、管道哈希（PipeHash）和重叠算法（Overlap）

- PipeSort, proposed in [16, 1], searches the space of possible sort orders for the best set of sorts that convert the CUBE lattice into a processing tree (e.g., Figure 3). The search attempts to minimize the number of sorts, while at the same time it seeks to compute a group-by from its smallest parent. The authors recognized that many of the sorts will have a common prefix, so they optimized the sorting procedures to take advantage of partial sorts. Except when sorting, PipeSort uses at most $d + 1$ memory cells for each of the simultaneous aggregates,where $d$ is the number of dimensions.

- [16, 1] 中提出的管道排序（PipeSort）算法在可能的排序顺序空间中搜索最佳的排序集，以将 CUBE 格转换为处理树（例如，图 3）。该搜索尝试最小化排序次数，同时试图从其最小的父分组依据计算一个分组依据。作者认识到许多排序会有一个共同的前缀，因此他们优化了排序过程以利用部分排序。除了排序时，管道排序（PipeSort）为每个同时进行的聚合最多使用 $d + 1$ 个内存单元，其中 $d$ 是维度的数量。

[14] points out that PipeSort performs at least $\left( \begin{matrix} d \\  \lceil d/2\rceil  \end{matrix}\right)$ sorts,where $d$ is the number of dimensions. When computing sparse CUBEs, many of the intermediate results that are sorted cannot fit in memory, so external sort will be used.

[14] 指出，管道排序（PipeSort）至少执行 $\left( \begin{matrix} d \\  \lceil d/2\rceil  \end{matrix}\right)$ 次排序，其中 $d$ 是维度的数量。在计算稀疏 CUBE 时，许多经过排序的中间结果无法放入内存，因此将使用外部排序。

- PipeHash,also proposed in $\left\lbrack  {{16},1}\right\rbrack$ ,computes a group-by from its smallest parent in the lattice. For example, if the attributes in Figure 2 are ordered such that $A \leq  B \leq  C \leq  D$ ,and group-by size estimates are proportional to the product of the cardinalities (as is the case with attribute independence assumptions), then the processing tree produced is shown in Figure 3. PipeHash uses a hash table for every simultaneously computed group-by. If all of the hash tables cannot fit in memory, PipeHash partitions the data on some attribute and processes each partition independently. PipeHash suffers from two problems. First, it does not overlap as much computation as PipeSort because PipeSort computes multiple group-bys with one sort where as PipeHash must re-hash the data for every group-by. Second, PipeHash requires a significant amount of memory to store the hash tables for the group-bys even after partitioning.

- $\left\lbrack  {{16},1}\right\rbrack$ 中也提出的管道哈希（PipeHash）算法从格中其最小的父分组依据计算一个分组依据。例如，如果图 2 中的属性按 $A \leq  B \leq  C \leq  D$ 排序，并且分组依据大小估计与基数的乘积成正比（在属性独立假设的情况下就是如此），那么生成的处理树如图 3 所示。管道哈希（PipeHash）为每个同时计算的分组依据使用一个哈希表。如果所有哈希表无法放入内存，管道哈希（PipeHash）会根据某个属性对数据进行分区，并独立处理每个分区。管道哈希（PipeHash）存在两个问题。首先，它不像管道排序（PipeSort）那样有大量的计算重叠，因为管道排序（PipeSort）通过一次排序计算多个分组依据，而管道哈希（PipeHash）必须为每个分组依据重新哈希数据。其次，即使在分区之后，管道哈希（PipeHash）也需要大量内存来存储分组依据的哈希表。

- Overlap, proposed in [4, 1], aims to overlap as much sorting as possible by computing a group-by from a parent with the maximum sort-order overlap. The algorithm recognizes that if a group-by shares a prefix with its parent, then the parent consists of a number of partitions, one for each value of the prefix. For example, the ${ABC}$ group-by has $\left| A\right|$ partitions that can be sorted independently on $C$ to produce the ${AC}$ sort order.

- [4, 1] 中提出的重叠算法（Overlap）旨在通过从具有最大排序顺序重叠的父分组依据计算一个分组依据，尽可能多地重叠排序。该算法认识到，如果一个分组依据与其父分组依据共享一个前缀，那么父分组依据由多个分区组成，每个分区对应前缀的一个值。例如，${ABC}$ 分组依据有 $\left| A\right|$ 个分区，这些分区可以在 $C$ 上独立排序以产生 ${AC}$ 排序顺序。

Overlap chooses a sort order for the root of the processing tree, and then all subsequent sorts are some suffix of this order. Once the processing tree is formed, Overlap tries to fit as many partitions in memory as possible to avoid writing intermediate results. If enough memory is available, Overlap can make one pass over a sorted input file.

重叠算法（Overlap）为处理树的根选择一个排序顺序，然后所有后续排序都是该顺序的某个后缀。一旦处理树形成，重叠算法（Overlap）会尝试将尽可能多的分区放入内存，以避免写入中间结果。如果有足够的内存，重叠算法（Overlap）可以对排序后的输入文件进行一次遍历。

With the same assumptions on the lattice as we used for PipeHash, Overlap also produces the processing tree in Figure 3. Actually, under these assumptions, choosing the smallest parent minimizes the partition sizes and the number of partitions, which is ideal for Overlap. For example, ${BD}$ could be computed from ${ABD}$ ,or ${BCD}$ , but ${BCD}$ produces partitions that are proportional to $\left| B\right|$ while ${ABD}$ produces partitions proportional to $\left| {BD}\right|$ . ${CD}$ could be computed from either ${ACD}$ or ${BCD}$ ,either would produce the same partition sizes, but the number of partitions in ${BCD}$ is proportional to $\left| B\right|$ while ${ACD}$ is proportional to $\left| A\right|$ ,so ${BCD}$ produces fewer partitions. Thus, picking the smallest parent is quite similar to picking for the most overlap.

与我们在PipeHash中对格的假设相同，Overlap算法也会生成图3中的处理树。实际上，在这些假设下，选择最小的父节点可以最小化分区大小和分区数量，这对Overlap算法来说是理想的。例如，${BD}$可以从${ABD}$或${BCD}$计算得到，但${BCD}$产生的分区与$\left| B\right|$成比例，而${ABD}$产生的分区与$\left| {BD}\right|$成比例。${CD}$可以从${ACD}$或${BCD}$计算得到，两者产生的分区大小相同，但${BCD}$中的分区数量与$\left| B\right|$成比例，而${ACD}$与$\left| A\right|$成比例，因此${BCD}$产生的分区更少。因此，选择最小的父节点与选择重叠最多的节点非常相似。

[14] argues that the Overlap on sparse CUBEs also produces a large amount of $\mathrm{I}/\mathrm{O}$ by sorting intermediate

[14]认为，对稀疏立方体（CUBEs）使用Overlap算法通过对中间结果进行排序也会产生大量的$\mathrm{I}/\mathrm{O}$

results (at least quadratic in the number of dimensions).

（至少与维度数量成二次方关系）。

Because these algorithms can generate significant $\mathrm{I}/\mathrm{O}$ for intermediate results or require large amounts of main memory, we do not consider them further.

由于这些算法可能会为中间结果生成大量的$\mathrm{I}/\mathrm{O}$，或者需要大量的主存，因此我们不再进一步考虑它们。

### 4.2 ArrayCube

### 4.2 数组立方体算法（ArrayCube）

An array-based algorithm for computing the CUBE was described in [19]; we call this ArrayCube. The algorithm is very similar to Overlap, except that it uses in-memory arrays to store the partitions and to avoid sorting. The algorithm expects its input to be in an array-based file-structure, but it can be extend to use relational input and output with little or no performance penalty. The input order for ArrayCube is slightly different from the input order for Overlap because the array file-structure is "chunked". A chunk of a n-dimensional array is a n-dimensional subarray that corresponds loosely to a page. The array is stored in units of chunks to provide multidimensional clustering; chunking is not useful for computing the cube.

[19]中描述了一种基于数组的计算立方体（CUBE）的算法，我们称之为数组立方体算法（ArrayCube）。该算法与Overlap算法非常相似，不同之处在于它使用内存数组来存储分区并避免排序。该算法期望其输入采用基于数组的文件结构，但它可以扩展为使用关系型输入和输出，且几乎不会造成性能损失。数组立方体算法（ArrayCube）的输入顺序与Overlap算法的输入顺序略有不同，因为数组文件结构是“分块”的。n维数组的一个块是一个n维子数组，大致对应于一个页面。数组以块为单位存储，以提供多维聚类；分块对于计算立方体并无用处。

This algorithm is unique in two regards. First, it requires no tuple comparisons, only array indexing. Second, the array file-structure offers compression as well as indexing. The other methods could benefit from compression as well, and with the amount of redundancy in the CUBE output, we expect high compression ratios. The algorithm is most effective when the product of the cardinalities of the dimensions is moderate. Unfortunately, if the data is too sparse, this method becomes infeasible because the in-memory arrays become too large to fit in main-memory.

该算法在两个方面具有独特性。首先，它不需要进行元组比较，只需要进行数组索引。其次，数组文件结构既提供压缩功能，也提供索引功能。其他方法也可以从压缩中受益，并且由于立方体输出中存在大量冗余，我们预计压缩率会很高。当各维度基数的乘积适中时，该算法最为有效。不幸的是，如果数据过于稀疏，这种方法就不可行了，因为内存数组会变得太大而无法装入主存。

### 4.3 PartitionedCube and MemoryCube

### 4.3 分区立方体算法（PartitionedCube）和内存立方体算法（MemoryCube）

PartitionedCube and MemoryCube, described in [14], are designed to work together. PartitionedCube partitions the data on some attribute into memory-sized units (similar to PipeHash but for a different reason), and MemoryCube computes the CUBE on each in-memory partition. The key observation they made is that buffering intermediate group-bys in memory - something all the previous algorithms do (except PipeSort, which does not buffer anything) - requires too much memory for large sparse CUBEs. Instead, they chose to buffer the partitioned input data for repeated in-memory sorts, similar to PipeSort, although they present a new algorithm that picks the minimum number of sorts (which is exactly the largest tier in the lattice $= \left( \begin{matrix} d \\  \lceil d/2\rceil  \end{matrix}\right)$ ). Once each partition for the first partitioning attribute is processed (which is half of the CUBE), the input is repartitioned on the next attribute.

[14]中描述的分区立方体算法（PartitionedCube）和内存立方体算法（MemoryCube）是设计用于协同工作的。分区立方体算法（PartitionedCube）根据某个属性将数据划分为内存大小的单元（类似于PipeHash算法，但原因不同），而内存立方体算法（MemoryCube）对每个内存分区计算立方体（CUBE）。他们的关键发现是，将中间分组结果缓存在内存中——之前的所有算法（除了不缓存任何内容的PipeSort算法）都是这样做的——对于大型稀疏立方体（CUBEs）来说需要太多的内存。相反，他们选择缓存分区后的输入数据，以便进行重复的内存排序，这与PipeSort算法类似，不过他们提出了一种新的算法来选择最少的排序次数（这恰好是格$= \left( \begin{matrix} d \\  \lceil d/2\rceil  \end{matrix}\right)$中的最大层级）。一旦处理完第一个分区属性的每个分区（这是立方体的一半），就会根据下一个属性对输入进行重新分区。

Since this algorithm is designed for sparse CUBEs, we consider this to be the best existing algorithm for the CUBE problems discussed in this paper (sparse CUBEs and Iceberg-CUBEs). Therefore, we only compare BUC to this algorithm.

由于该算法是为稀疏立方体（CUBEs）设计的，我们认为它是本文所讨论的立方体问题（稀疏立方体和冰山立方体）中现有的最佳算法。因此，我们只将BUC算法与该算法进行比较。

In recent independent work, researchers at Columbia University also found that pushing the HAVING clause into CUBE computation is beneficial [15]. They describe how to take advantage of HAVING predicates in PartitionedCube/ MemoryCube.

在最近的独立研究中，哥伦比亚大学的研究人员也发现，将HAVING子句融入立方体计算是有益的[15]。他们描述了如何在分区立方体算法（PartitionedCube）/内存立方体算法（MemoryCube）中利用HAVING谓词。

<!-- Media -->

---

Procedure BottomUpCube(input, dim)

自底向上立方体算法（BottomUpCube）过程（输入，维度）

Inputs:

输入：

		input: The relation to aggregate.

		  输入：要进行聚合的关系。

		dim: The starting dimension for this iteration.

		  维度：本次迭代的起始维度。

Globals:

全局变量：

		constant numDims: The total number of dimensions.

		常量numDims：维度的总数。

		constant cardinality[numDims]: The cardinality of

		常量cardinality[numDims]：

					each dimension.

					每个维度的基数。

		constant minsup: The minimum number of tuples in a

		常量minsup：一个分区中要输出的元组的最小数量。

					partition for it to be output.

					为了使其能够输出。

		outputRec: The current output record.

		outputRec：当前的输出记录。

		dataCount[numDims]: Stores the size of each partition.

		dataCount[numDims]：存储每个分区的大小。

					dataCount[i] is a list of integers of size

					dataCount[i]是一个大小为

					cardinality[i].

					cardinality[i]的整数列表。

Outputs:

输出：

		One record that is the aggregation of input.

		一条记录，它是输入的聚合结果。

		Recursively, outputs CUBE(dim, ..., numDims) on

		递归地，在

					input (with minimum support).

					输入（具有最小支持度）上输出CUBE(dim, ..., numDims)。

Method:

方法：

	1: Aggregate(input); // Places result in outputRec

	1: 聚合(input); // 将结果放入outputRec

	2: if input.count(   ) == 1 then // Optimization

	2: 如果input.count(   ) == 1 则 // 优化

					WriteAncestors(input[0], dim); return;

					写入祖先节点（输入[0]，维度）；返回;

			write outputRec;

			写入输出记录;

			for $\mathrm{d} = \dim ;\mathrm{d} <$ numDims $;\mathrm{d} +  + \mathrm{{do}}$

			对于 $\mathrm{d} = \dim ;\mathrm{d} <$ 个维度 $;\mathrm{d} +  + \mathrm{{do}}$

					let $C =$ cardinality[d];

					设 $C =$ 为基数[d];

					Partition(input, d, C, dataCount[d]);

					对输入、维度d、C和数据计数[d]进行分区;

					let $\mathrm{k} = 0$ ;

					令 $\mathrm{k} = 0$ ;

					for $\mathrm{i} = 0;\mathrm{i} < \mathrm{C};\mathrm{i} +  +$ do // For each partition

					对于 $\mathrm{i} = 0;\mathrm{i} < \mathrm{C};\mathrm{i} +  +$ 执行 // 对于每个分区

						let $c =$ dataCount $\left\lbrack  d\right\rbrack  \left\lbrack  i\right\rbrack$

						令 $c =$ 数据计数 $\left\lbrack  d\right\rbrack  \left\lbrack  i\right\rbrack$

						if $c >  =$ minsup then $//$ The BUC stops here

						如果 $c >  =$ 小于最小支持度（minsup），则 $//$ BUC 在此处停止

								outputRec.dim[d] $=$ input[k].dim[d];

								输出记录的维度 [d] $=$ 输入 [k] 的维度 [d];

								BottomUpCube(input[k ... k+c], d+1);

								自底向上立方体算法（BottomUpCube）（输入 [k ... k + c]，d + 1）;

						end if

						结束条件判断

						$\mathrm{k} +  = \mathrm{c}$ ;

					end for

					结束循环

				outputRec.dim[d] $=$ ALL;

				输出记录的维度 [d] $=$ 全部（ALL）;

			end for

			结束循环

---

Figure 5: Algorithm BottomUpCube (BUC)

图 5：自底向上立方体算法（BottomUpCube，BUC）

<!-- Media -->

## 5 Algorithm Bottom-Up Cube

## 5 自底向上立方体算法

We propose a new algorithm called BottomUpCube (BUC) for sparse CUBE and Iceberg-CUBE computation. The idea in BUC is to combine the I/O efficiency of PartitionedCube/ MemoryCube, but to take advantage of minimum support pruning like Apriori [2]. BUC was inspired by the algorithms in [14], particularly PartitionedCube. BUC is similar to a version of algorithm PartitionedCube that never calls MemoryCube.

我们提出了一种名为自底向上立方体算法（BottomUpCube，BUC）的新算法，用于稀疏立方体（CUBE）和冰山立方体（Iceberg - CUBE）的计算。BUC 的思路是结合分区立方体算法（PartitionedCube）/内存立方体算法（MemoryCube）的输入/输出（I/O）效率，同时像先验算法（Apriori）[2]一样利用最小支持度进行剪枝。BUC 受到文献[14]中算法的启发，特别是分区立方体算法。BUC 类似于分区立方体算法的一个版本，但从不调用内存立方体算法。

To achieve pruning, BUC proceeds from the bottom of the lattice (i.e., the smallest / most aggregated group-bys), and works its way up towards the larger, less aggregated group-bys. All of the previous algorithms compute in the opposite direction. Since parent group-bys are used to compute child group-bys, the algorithms cannot avoid computing the parents.

为了实现剪枝，BUC 从格的底部（即最小/最聚合的分组依据）开始，逐步向上处理更大、聚合程度更低的分组依据。之前的所有算法都是以相反的方向进行计算。由于父分组依据用于计算子分组依据，这些算法无法避免计算父分组依据。

The details of BUC are in Figure 5. The first step is to aggregate the entire input (line 1) and write the result (line 3). (Line 2 is an optimization that we discuss below. For now we ignore that line.) For each dimension $d$ between dim and numDims,the input is partitioned on dimension $d$ (line 6). On return from Partition(   ),dataCount contains the number of records for each distinct value of the $d$ -th dimension. Line 8 iterates through the partitions (i.e., each distinct value). If the partition meets minimum support (which is always true for full CUBEs because minsup is one), the partition becomes the input relation in the next recursive call to BottomUpCube, which computes the (Iceberg) CUBE on the partition for dimensions $d + 1$ to numDims. Upon return from the recursive call, we continue with the next partition of dimension $d$ . Once all the partitions are processed, we repeat the whole process for the next dimension.

BUC 的详细步骤如图 5 所示。第一步是对整个输入进行聚合（第 1 行）并写入结果（第 3 行）。（第 2 行是我们下面要讨论的一种优化，目前我们忽略这一行。）对于维度 $d$ 到维度数量（numDims）之间的每个维度 $d$，输入在维度 $d$ 上进行分区（第 6 行）。从分区函数（Partition()）返回时，数据计数（dataCount）包含第 $d$ 个维度每个不同值的记录数量。第 8 行遍历各个分区（即每个不同的值）。如果分区满足最小支持度（对于完整立方体，这总是成立的，因为最小支持度为 1），则该分区成为下一次递归调用自底向上立方体算法（BottomUpCube）的输入关系，该算法会在该分区上计算从维度 $d + 1$ 到维度数量（numDims）的（冰山）立方体。递归调用返回后，我们继续处理维度 $d$ 的下一个分区。一旦所有分区都处理完毕，我们对下一个维度重复整个过程。

<!-- Media -->

<!-- figureText: b2 d1 d2 c2 b1 al b3 b4 a2 a3 a4 -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_5.jpg?x=232&y=63&w=481&h=534&r=0"/>

Figure 6: BUC Partitioning

图 6：BUC 分区

<!-- Media -->

Figure 4 shows the BUC processing tree (i.e., how it covers the lattice). The numbers indicate the order in which BUC visits the group-bys. Figure 6 illustrates how the input is partitioned during the first four calls to BottomUpCube (assuming minsup is one). First BUC produces the empty group-by. Next,it partitions on dimension $A$ ,producing partitions ${a1}$ to ${a4}$ ,and then it recurses on partition ${a1}$ . The ${a1}$ partition is aggregated and produces a single tuple for the $A$ group-by. Next,it partitions the ${a1}$ partition on dimension $B$ . It recurses on the $\langle {a1},{b1}\rangle$ partition and writes a $\langle {a1},{b1}\rangle$ tuple for the ${AB}$ group-by. Similarly for $\langle {a1},{b1},{c1}\rangle$ ,and then $\langle {a1},{b1},{c1},{d1}\rangle$ ,but this time it does not enter the loop at line 4. Instead it simply returns only to recurse again on the $\langle {a1},{b1},{c1},{d2}\rangle$ partition. BUC then returns twice and then recurses on the $\langle {a1},{b1},{c2}\rangle$ partition. When this is complete,it partitions the $\langle {a1},{b1}\rangle$ partition on $D$ to produce the $\langle {a1},{b1},D\rangle$ aggregates.

图4展示了BUC处理树（即它如何覆盖格）。数字表示BUC访问分组依据（group-by）的顺序。图6说明了在对BottomUpCube进行前四次调用期间输入是如何划分的（假设最小支持度为1）。首先，BUC生成空的分组依据。接下来，它在维度$A$上进行划分，生成划分${a1}$到${a4}$，然后在划分${a1}$上进行递归。${a1}$划分被聚合，并为$A$分组依据生成一个单一元组。接下来，它在维度$B$上对${a1}$划分进行划分。它在$\langle {a1},{b1}\rangle$划分上进行递归，并为${AB}$分组依据写入一个$\langle {a1},{b1}\rangle$元组。对于$\langle {a1},{b1},{c1}\rangle$和$\langle {a1},{b1},{c1},{d1}\rangle$也是类似的情况，但这次它不会进入第4行的循环。相反，它只是返回，然后再次在$\langle {a1},{b1},{c1},{d2}\rangle$划分上进行递归。然后BUC返回两次，接着在$\langle {a1},{b1},{c2}\rangle$划分上进行递归。完成此操作后，它在$D$上对$\langle {a1},{b1}\rangle$划分进行划分，以生成$\langle {a1},{b1},D\rangle$聚合。

Once the $\langle {a1},{b1}\rangle$ partition is completely processed,BUC proceeds to $\langle {a1},{b2}\rangle$ . This partition consists of a single tuple. If we ignore line 2 for the moment, then BUC will recurse for $\langle {a1},{b2},c\rangle ,\langle {a1},{b2},c,d\rangle$ ,and $\langle {a1},{b2},d\rangle$ aggregating and partitioning a single tuple. While the result is correct, it's a fruitless exercise. We add line 2 to the algorithm so that the aggregates on this tuple are computed just once, and then the result tuple is written to each of its ancestor group-bys in WriteAncestors(   ) by simply setting the dimension values appropriately (in this case: $\langle {a1},{b2}\rangle ,\langle {a1},{b2},c\rangle ,\langle {a1},{b2},c,d\rangle$ , and $\langle {a1},{b2},d\rangle )$ .

一旦$\langle {a1},{b1}\rangle$划分被完全处理，BUC就会处理$\langle {a1},{b2}\rangle$。这个划分由一个单一元组组成。如果我们暂时忽略第2行，那么BUC将对$\langle {a1},{b2},c\rangle ,\langle {a1},{b2},c,d\rangle$和$\langle {a1},{b2},d\rangle$进行递归，对一个单一元组进行聚合和划分。虽然结果是正确的，但这是一项徒劳的工作。我们在算法中添加第2行，以便对这个元组的聚合只计算一次，然后通过适当地设置维度值（在这种情况下：$\langle {a1},{b2}\rangle ,\langle {a1},{b2},c\rangle ,\langle {a1},{b2},c,d\rangle$和$\langle {a1},{b2},d\rangle )$），将结果元组写入WriteAncestors()中的每个祖先分组依据。

The elimination of the aggregation and partitioning of single tuple partitions is a key factor in the success of BUC on sparse CUBEs because many partitions have a single tuple. Figure 1 shows that ${20}\%$ of the group-bys consisted almost entirely of single tuple partitions! On one generated dataset, this optimization improved the computation by more than ${40}\%$ .

消除单一元组划分的聚合和划分是BUC在稀疏立方体（CUBE）上取得成功的一个关键因素，因为许多划分只有一个单一元组。图1显示，${20}\%$的分组依据几乎完全由单一元组划分组成！在一个生成的数据集上，这种优化将计算效率提高了超过${40}\%$。

### 5.1 Iceberg-CUBE with BUC

### 5.1 结合BUC的冰山立方体（Iceberg-CUBE）

To this point, we considered only full CUBE computation (i.e., minsup $= 1$ ). The optimization for a single tuple partition is similar to how BUC processes an Iceberg-CUBE. When a small partition is found, instead of writing for all of the group-bys, BUC simply skips the partition (line 10) and does not consider any of the partition's ancestors. The pruning is correct because the partition sizes are always decreasing when BUC recurses, and therefore none of the ancestors can have minimum support.

到目前为止，我们只考虑了完整立方体的计算（即最小支持度$= 1$）。对单一元组划分的优化类似于BUC处理冰山立方体的方式。当找到一个小划分时，BUC不会为所有的分组依据进行写入操作，而是直接跳过该划分（第10行），并且不考虑该划分的任何祖先。这种剪枝是正确的，因为在BUC递归时，划分的大小总是在减小，因此没有一个祖先可以满足最小支持度。

The pruning is similar to the pruning in Apriori [2]. The major difference between Apriori (appropriately adapted for Iceberg-CUBE computation) and BUC is that Apriori processes the lattice breadth first instead of depth first. In our example,Apriori would compute the $A,B,C$ ,and $D$ group-bys in one pass of the input. A candidate set would be created from all the partitions that meet minimum support. For example,if $\langle {a3}\rangle$ and $\langle {b2}\rangle$ made minimum support,then $\langle {a3},{b2}\rangle$ would be added to the candidate set. The input would be read a second time and the candidate pairs would be pruned for minimum support. Now the remaining pairs are combined to form the candidate triples. If $\langle {a3},{b2}\rangle ,\langle {a3},{c5}\rangle$ ,and $\langle {b2},{c5}\rangle$ all made minimum support, then $\langle {a3},{b2},{c5}\rangle$ is a candidate in the third pass.

剪枝操作与Apriori算法[2]中的剪枝类似。Apriori算法（针对冰山立方体（Iceberg - CUBE）计算进行适当调整）和BUC算法的主要区别在于，Apriori算法是按广度优先而非深度优先的方式处理格结构。在我们的示例中，Apriori算法会在对输入数据进行一次遍历的过程中计算$A,B,C$和$D$分组操作。会从所有满足最小支持度的分区中创建候选集。例如，如果$\langle {a3}\rangle$和$\langle {b2}\rangle$满足最小支持度，那么$\langle {a3},{b2}\rangle$将被添加到候选集中。会对输入数据进行第二次读取，并对候选对进行剪枝以满足最小支持度。现在，剩余的对会组合形成候选三元组。如果$\langle {a3},{b2}\rangle ,\langle {a3},{c5}\rangle$和$\langle {b2},{c5}\rangle$都满足最小支持度，那么$\langle {a3},{b2},{c5}\rangle$将是第三次遍历中的一个候选。

Apriori can prune group-bys one step earlier than BUC. For example,if the partition $\langle {a3}\rangle$ met minimum support but $\langle {b2}\rangle$ did not,BUC will consider the $\langle {a3},{b2}\rangle$ partition but Apriori will not. The problem with using Apriori is the candidate set usually cannot fit in memory because little pruning is expected during first few passes of the input. BUC trades pruning for locality of reference and reduced memory requirements.

Apriori算法比BUC算法能提前一步对分组操作进行剪枝。例如，如果分区$\langle {a3}\rangle$满足最小支持度，但$\langle {b2}\rangle$不满足，BUC算法会考虑$\langle {a3},{b2}\rangle$分区，而Apriori算法则不会。使用Apriori算法的问题在于，候选集通常无法全部存入内存，因为在对输入数据进行前几次遍历时，预计不会有太多剪枝操作。BUC算法通过牺牲剪枝操作来换取引用局部性和降低内存需求。

We implemented the modified Apriori algorithm and compared it to BUC. As expected, when the output size is large, Apriori needs too much memory and performs terribly. We then compared the algorithms with extremely skewed input: all duplicates. Apriori needed very little memory, but it still performed significantly worse than BUC.

我们实现了改进后的Apriori算法，并将其与BUC算法进行了比较。正如预期的那样，当输出规模较大时，Apriori算法需要大量内存，并且性能极差。然后，我们使用极度倾斜的输入数据（所有数据都是重复的）对这两种算法进行了比较。Apriori算法只需要很少的内存，但它的性能仍然明显比BUC算法差。

### 5.2 Additional Pruning Functions

### 5.2 额外的剪枝函数

As described in [13], functions other than count can be used for pruning. The pruning function must be monotonic. ${}^{2}$ For any two sets (of tuples,in our case) $S$ and $T$ such that $S \subseteq  T$ , a function $f$ is monotonicly decreasing if $f\left( S\right)  <  = f\left( T\right)$ . (If $f$ is monotonically increasing,the inequality in the prune expression must be reversed. We are actually interested in monotonically decreasing boolean functions,true $<$ false: if $f\left( T\right)$ is false then $f\left( S\right)$ is false for any subset $S$ of $T$ ). Count is monotonicly decreasing, since the count of a subset is certainly larger than the count of the superset. Min and max are monotonic,as is the sum of positive numbers. Also,if $f$ and $g$ are two monotonically decreasing boolean functions, then $f \land  g$ and $f \vee  g$ are both monotonically decreasing boolean functions.

正如文献[13]中所描述的，除计数函数外，其他函数也可用于剪枝。剪枝函数必须是单调的。${}^{2}$对于任意两个集合（在我们的例子中是元组集合）$S$和$T$，如果$S \subseteq  T$，那么当$f\left( S\right)  <  = f\left( T\right)$时，函数$f$是单调递减的。（如果$f$是单调递增的，那么剪枝表达式中的不等式必须反转。我们实际上关注的是单调递减的布尔函数，即从真$<$到假：如果$f\left( T\right)$为假，那么对于$T$的任意子集$S$，$f\left( S\right)$也为假）。计数函数是单调递减的，因为子集的计数肯定小于超集的计数。最小值、最大值函数是单调的，正数的和函数也是单调的。此外，如果$f$和$g$是两个单调递减的布尔函数，那么$f \land  g$和$f \vee  g$也都是单调递减的布尔函数。

---

<!-- Footnote -->

${}^{2}$ The functions are called anti-monotonic in [13].

${}^{2}$在文献[13]中，这些函数被称为反单调函数。

<!-- Footnote -->

---

Average, however, is not monotonic. [13] describes how some non-monotonic functions can still be used to prune by replacing it with a conservative monotonic function. For example, if the average of positive numbers must be above $X$ ,we can prune with the sum above $X$ and the minimum below $X$ . If the sum of both positive and negative numbers must be above $X$ ,then we can prune with the sum of only the positive numbers above $X$ .

然而，平均值函数不是单调的。文献[13]描述了如何通过用一个保守的单调函数替换非单调函数，仍然可以使用一些非单调函数进行剪枝。例如，如果正数的平均值必须大于$X$，我们可以使用总和大于$X$且最小值小于$X$的条件进行剪枝。如果正数和负数的总和必须大于$X$，那么我们可以使用仅正数的总和大于$X$的条件进行剪枝。

These additional pruning functions can be quite useful in practice. For example, we can prune aggregates with little sales (HAVING SUM(sales) $>  = \mathrm{S}$ ),or those aggregates that do not include any young or old people (HAVING MIN(age) <= young OR MAX(age) >= old). These functions can not only be used for aggregate precomputation, but also for mining multi-dimensional association rules.

这些额外的剪枝函数在实际应用中非常有用。例如，我们可以对销售额较低的聚合结果进行剪枝（HAVING SUM(sales) $>  = \mathrm{S}$），或者对不包含任何年轻人或老年人的聚合结果进行剪枝（HAVING MIN(age) <= 年轻人年龄 OR MAX(age) >= 老年人年龄）。这些函数不仅可用于聚合预计算，还可用于挖掘多维关联规则。

To use additional pruning predicates in BUC, add:

要在BUC算法中使用额外的剪枝谓词，需添加：

if CanPrune(   ) then return;

if CanPrune(   ) then return;

after the call to Aggregate(   ) and before line 2 (where CanPrune(   ) is the pruning predicate).

在调用Aggregate(   )之后且在第2行之前（其中CanPrune(   )是剪枝谓词）。

The class of predicates that BUC can use for pruning is by no means the only useful HAVING predicates. For example,consider HAVING COUNT(*) $< \mathrm{X}$ . In this case, the user wants the groups with little aggregation, which occur towards the top of the lattice. Now, the previous CUBE algorithms can prune their computation, but BUC cannot. When computing large CUBEs however, this predicate will not reduce the output size much, so the output time will dominate the cost of computation. Note that even if a predicate cannot be used for pruning in BUC, it can still be used for reducing the CUBE output.

BUC可用于剪枝的谓词类别绝不是唯一有用的HAVING谓词。例如，考虑HAVING COUNT(*) $< \mathrm{X}$ 。在这种情况下，用户希望获取聚合值较小的分组，这些分组位于格结构的顶部。此时，之前的CUBE算法可以对计算进行剪枝，但BUC却不能。然而，在计算大型CUBE时，这个谓词不会大幅减少输出规模，因此输出时间将主导计算成本。请注意，即使某个谓词不能在BUC中用于剪枝，它仍然可以用于减少CUBE的输出。

### 5.3 Partitioning

### 5.3 分区

The majority of the time in BUC is spent partitioning the data, so optimizing Partition(   ) is important. When 'input' does not fit in main memory, the data must be partitioned to disk. This can be done with hash-partitioning, followed by a partitioning within each bucket (which hopefully now fits in main memory), or external-sort can be used. When performing an external partitioning, the aggregation step (at line 1 of BUC) can be combined with the partitioning.

BUC的大部分时间都花在对数据进行分区上，因此优化Partition( )函数很重要。当“输入”数据无法全部存入主内存时，必须将数据分区存储到磁盘上。这可以通过哈希分区来实现，然后在每个桶内再进行分区（希望此时桶内数据能存入主内存），或者也可以使用外部排序。在进行外部分区时，聚合步骤（BUC算法的第1行）可以与分区操作结合进行。

Once 'input' fits in main memory, which hopefully occurs after partitioning on the first dimension, we can use in-memory sorting or hashing to partition the data. Note that once 'input' fits into memory on some call to BUC, for all recursive calls from that point, 'input' will fit in memory. Therefore, an implementation of BUC should have BUC-External and BUC-Internal, where processing starts with BUC-External and switches to BUC-Internal when the input fits in memory. Our current implementation does not perform external partitioning.

一旦“输入”数据能够存入主内存（希望在对第一个维度进行分区后能达到这种情况），我们就可以使用内存内排序或哈希来对数据进行分区。请注意，一旦在某次调用BUC时“输入”数据能够存入内存，那么从这一点开始的所有递归调用中，“输入”数据都将能存入内存。因此，BUC的实现应该包含BUC-External和BUC-Internal两部分，处理过程从BUC-External开始，当输入数据能存入内存时切换到BUC-Internal。我们目前的实现没有进行外部分区。

Our implementation uses a linear sorting method called CountingSort [17]. CountingSort excels at sorting large lists that have a sort key of moderate cardinality (i.e., many duplicates). The algorithm requires that the sort key be an integer value between zero and its cardinality, and that the cardinality is known in advance. When an attribute of a relation meets this property, we say the attribute is packed.

我们的实现使用了一种称为计数排序（CountingSort）[17]的线性排序方法。计数排序在对具有中等基数排序键（即有许多重复项）的大型列表进行排序时表现出色。该算法要求排序键是一个介于零和其基数之间的整数值，并且基数需要预先已知。当关系的某个属性满足这一特性时，我们称该属性是紧凑的（packed）。

Our implementation of BUC assumes that all of the dimensions are packed. This same assumption is used in [19]. This assumption is reasonable because strings and other types are usually mapped into integers to save space and eliminate type interpretation. Also, in a star-schema [12], the dimension values are often system generated integers that are used as keys to the dimension tables. If the dimensions are not packed in the input, they can be packed when the input is first read by creating a hashed symbol table for each dimension as described in [6], and the mapping can be reversed when tuples are output (or a simple pre- and post-processing pass can be used).

我们对BUC的实现假设所有维度都是紧凑的。文献[19]中也采用了相同的假设。这个假设是合理的，因为字符串和其他类型通常会被映射为整数，以节省空间并消除类型解释。此外，在星型模式[12]中，维度值通常是系统生成的整数，用作维度表的键。如果输入中的维度不是紧凑的，可以在首次读取输入时，通过为每个维度创建一个哈希符号表（如文献[6]中所述）来将其紧凑化，并且在输出元组时可以将映射反转（或者可以使用简单的预处理和后处理步骤）。

We found the use of CountingSort to be an important optimization to BUC. For example, when sorting one million records with widely varied key cardinality and skew, QuickSort ran between 3 and 10 times slower than CountingSort. CountingSort is faster not only because it sorts in $\mathrm{O}\left( \mathrm{N}\right)$ time,but also because it does not perform any key comparisons. When using CountingSort, we do not even need comparisons to find the partition boundaries, because the counts computed in CountingSort can be saved for use in BUC.

我们发现使用计数排序是对BUC的一项重要优化。例如，在对一百万条记录进行排序时，这些记录的键基数和偏斜度差异很大，快速排序（QuickSort）的运行速度比计数排序慢3到10倍。计数排序更快不仅是因为它的排序时间复杂度为$\mathrm{O}\left( \mathrm{N}\right)$，还因为它不进行任何键比较。使用计数排序时，我们甚至不需要进行比较来确定分区边界，因为计数排序中计算的计数可以保存下来供BUC使用。

CountingSort cannot be easily used in other CUBE algorithms because they perform sorts on several dimensions (composite keys). (However, CountingSort is a stable sort. This means that a sort on a composite key can be achieved by calling CountingSort for each key attribute in reverse order.)

计数排序不容易用于其他CUBE算法，因为其他算法会在多个维度（复合键）上进行排序。（不过，计数排序是一种稳定排序。这意味着可以通过按逆序对每个键属性调用计数排序来实现对复合键的排序。）

Unfortunately, the advantage of CountingSort over Quick-Sort slowly degrades as the ratio of the number of tuples to the cardinality decreases. When the number of tuples is significantly less than the cardinality of the partitioning dimension, QuickSort is faster than CountingSort. BUC produces many sorts with small partitions, so our implementation switches to QuickSort when the number of tuples in the partition is less than $1/4$ the cardinality. (Also,QuickSort switches to InsertionSort when the number of tuples is less than 12.)

不幸的是，随着元组数量与基数的比率降低，计数排序相对于快速排序的优势会逐渐减弱。当元组数量明显少于分区维度的基数时，快速排序比计数排序更快。BUC会产生许多小分区的排序操作，因此我们的实现会在分区中的元组数量少于$1/4$倍基数时切换到快速排序。（此外，当元组数量少于12时，快速排序会切换到插入排序。）

#### 5.3.1 Dimension Ordering

#### 5.3.1 维度排序

The performance of BUC is sensitive to the ordering of the dimensions. The goal of BUC is to prune as early as possible; i.e., BUC wants to find partitions that do not meet minimum support (or the other pruning criteria, or that only have one tuple). For best performance, the most discriminating dimensions should be used first. Remember that the first dimension is used in half of the group-bys, so it has the most potential for savings.

BUC的性能对维度的排序很敏感。BUC的目标是尽早进行剪枝；即，BUC希望找到不满足最小支持度（或其他剪枝标准，或者只有一个元组）的分区。为了获得最佳性能，应该首先使用最具区分性的维度。请记住，第一个维度会在一半的分组操作中使用，因此它最有可能节省计算量。

How discriminating a dimensions is depends upon several factors:

一个维度的区分性取决于几个因素：

- Cardinality: The cardinality of a dimension (the number of distinct values) determines the number of partitions that are created. The higher the cardinality, the smaller the partitions, and therefore the closer BUC is to pruning some computation.

- 基数：维度的基数（不同值的数量）决定了创建的分区数量。基数越高，分区越小，因此BUC越有可能减少一些计算量。

- Skew: The skew in a dimension affects the size of each partition. Skewed dimensions also have a smaller effective cardinality when used as the second or later partitioning attribute because it is likely that infrequent values will not appear in some partition. The more uniform a dimension (i.e., the less skew), the better it is for pruning.

- 倾斜度：维度的倾斜度会影响每个分区的大小。当倾斜的维度用作第二个或后续的分区属性时，其有效基数也会较小，因为不常见的值可能不会出现在某些分区中。维度越均匀（即倾斜度越小），对剪枝越有利。

- Correlation: If a dimension is correlated with an earlier partitioning dimension, then its effective cardinality is reduced. Correlation decreases pruning.

- 相关性：如果一个维度与较早的分区维度相关，则其有效基数会降低。相关性会减少剪枝效果。

We experimented with two heuristics for ordering the dimensions. The first heuristic is to order the dimensions based on decreasing cardinality. The second heuristic is to order the dimensions based on increasing maximum number of duplicates. When the data is not skewed, the two heuristics are equivalent. Section 6.5 gives a synthetic example where the second heuristic out-performs the first, but on we found little difference on real datasets.

我们对两种维度排序启发式方法进行了实验。第一种启发式方法是根据基数递减对维度进行排序。第二种启发式方法是根据重复项的最大数量递增对维度进行排序。当数据没有倾斜时，这两种启发式方法是等效的。第6.5节给出了一个合成示例，其中第二种启发式方法的性能优于第一种，但在真实数据集上我们发现差异不大。

### 5.4 Collapsing Duplicates

### 5.4 合并重复项

In the presence of high skew or correlation, a few group-by values can account for most of the tuples in the input. For example,when computing $\operatorname{CUBE}\left( {A,B,C,D}\right)$ ,the partition $\langle {a3},{b2},{c7},{d4}\rangle$ could contain ${90}\%$ of the original input. When this occurs, it is worthwhile to collapse the duplicate partitioning values to a singe tuple (using aggregation).

在存在高度倾斜或相关性的情况下，少数分组依据值可能占输入中大多数元组。例如，在计算 $\operatorname{CUBE}\left( {A,B,C,D}\right)$ 时，分区 $\langle {a3},{b2},{c7},{d4}\rangle$ 可能包含 ${90}\%$ 的原始输入。出现这种情况时，值得将重复的分区值合并为单个元组（使用聚合）。

If skewed data is expected to be common, we suggest changing the top-level call to BUC to collapse duplicates. This can be done by making a copy of the BUC procedure called BUC-Dedup and starting the computation at BUC-Dedup. Then, replace the Partition(   ) function at line 6 of BUC-Dedup with a function that not only partitions the data on dimension $d$ but collapses all of the duplicates on dimensions $d$ ... numDims -1 .

如果预计倾斜数据很常见，我们建议更改对BUC的顶级调用以合并重复项。这可以通过创建一个名为BUC - Dedup的BUC过程副本并从BUC - Dedup开始计算来实现。然后，将BUC - Dedup第6行的Partition( )函数替换为一个不仅在维度 $d$ 上对数据进行分区，还能合并维度 $d$ ... numDims - 1 上所有重复项的函数。

Collapsing duplicates has three disadvantages. First, if the data has few duplicates, there is a modest extra cost of trying to eliminate them. Second, the 'input' to BUC is now the result of aggregation, and if a large number of aggregates are computed on a small number of measure fields, less tuples will fit in memory. Third, and most importantly, holistic aggregate functions can no longer be computed because the Aggregate(   ) function at line 1 receives partially aggregated data.

合并重复项有三个缺点。首先，如果数据中重复项很少，尝试消除它们会有适度的额外成本。其次，现在BUC的“输入”是聚合的结果，如果在少量度量字段上计算大量聚合，内存中能容纳的元组会减少。第三，也是最重要的一点，由于第1行的Aggregate( )函数接收到的是部分聚合的数据，因此无法再计算整体聚合函数。

#### 5.4.1 Switching To ArrayCube

#### 5.4.1 切换到ArrayCube

BUC can perform poorly when each recursive partitioning does not significantly reduce the input size. For example, consider computing the CUBE on a relation that is 64 times the size of main memory, with ten dimensions that each have two distinct values. In this case, BUC needs to partition on six attributes before the input fits in memory. The CUBE is actually dense, not sparse, so previous algorithms, in particular ArrayCube [19], will perform better than BUC.

当每次递归分区不能显著减少输入大小时，BUC的性能可能较差。例如，考虑在一个大小是主存64倍的关系上计算CUBE，该关系有十个维度，每个维度有两个不同的值。在这种情况下，BUC需要对六个属性进行分区，输入才能放入内存。实际上CUBE是密集的，而不是稀疏的，因此以前的算法，特别是ArrayCube [19]，会比BUC性能更好。

However, this effect can occur even when computing a sparse CUBE. Consider the previous example again, but say one dimension (call it $A$ ) has a cardinality of 100,000 . The result CUBE is much more sparse, but half the group-bys are just as dense as they were before (i.e., the ones that do not use $A$ ). If $A$ is used as the first partitioning attribute, then BUC efficiently computes all the group-bys on $A$ ,but performs poorly on the remaining group-bys.

然而，即使在计算稀疏CUBE时也可能出现这种情况。再次考虑前面的例子，但假设一个维度（称其为 $A$ ）的基数为100,000。结果CUBE更加稀疏，但一半的分组依据仍然和以前一样密集（即不使用 $A$ 的那些）。如果将 $A$ 用作第一个分区属性，那么BUC可以有效地计算所有关于 $A$ 的分组依据，但对其余分组依据的计算性能较差。

We suggest switching from BUC to ArrayCube whenever the product of the remaining cardinalities is reasonably small but the number of input tuples is still large. The switch can occur at any point, but a logical choice would be to use ArrayCube on the last dimensions during to topmost call to BUC. For example, when computing $\operatorname{CUBE}\left( {A,B,C,D}\right)$ ,if $A$ is large but $B,C$ ,and $D$ are small, use BUC to compute all the group-bys that use $A$ ,and use ArrayCube to compute the remaining group-bys starting at group-by ${BCD}$ (refer to Figure 4).

我们建议每当剩余基数的乘积相当小，但输入元组的数量仍然很大时，从BUC切换到ArrayCube。切换可以在任何时候进行，但一个合理的选择是在对BUC的最顶层调用中对最后几个维度使用ArrayCube。例如，在计算 $\operatorname{CUBE}\left( {A,B,C,D}\right)$ 时，如果 $A$ 很大，但 $B,C$ 和 $D$ 很小，使用BUC计算所有使用 $A$ 的分组依据，并使用ArrayCube从分组依据 ${BCD}$ 开始计算其余分组依据（参见图4）。

When switching to ArrayCube, we can no longer prune any of the computation, although the output can still be pruned. However, all of the group-bys must be relatively small (and therefore aggregate many records) to fit in memory, which implies that most of those group-bys are likely to be computed in any case. Another downside to using ArrayCube is that it does not support holistic aggregate functions.

当切换到数组立方体（ArrayCube）时，我们无法再对任何计算进行剪枝，尽管输出仍可进行剪枝。然而，所有的分组操作（group - by）必须相对较小（从而聚合大量记录）才能装入内存，这意味着在任何情况下，大多数分组操作都可能会被计算。使用数组立方体的另一个缺点是它不支持整体聚合函数。

This optimization is at odds with collapsing duplicates. If we reconsider the first example in this section ( 10 dimensions each with cardinality of 2 ), collapsing duplicates on the original input will reduce the relation to at most ${2}^{10} = {1024}$ tuples. More work needs to be done to determine which strategy is best.

这种优化与合并重复项相矛盾。如果我们重新考虑本节中的第一个示例（10 个维度，每个维度的基数为 2），对原始输入进行重复项合并将使关系最多减少到 ${2}^{10} = {1024}$ 个元组。需要做更多工作来确定哪种策略最佳。

### 5.5 Using BUC with PBS

### 5.5 将 BUC 与 PBS 结合使用

BUC works well with the PickBySize (PBS) algorithm for choosing group-bys to precompute [18]. PBS chooses the group-bys with the smallest expected (output) size (i.e., the group-bys that perform the most aggregation). PBS chooses entire group-bys, not partitions like Iceberg-CUBE.

BUC 与按大小选择（PickBySize，PBS）算法配合良好，该算法用于选择要预计算的分组操作 [18]。PBS 选择预期（输出）大小最小的分组操作（即执行最多聚合操作的分组操作）。PBS 选择整个分组操作，而不像冰山立方体（Iceberg - CUBE）那样选择分区。

Since PBS chooses by the size of the group-by, if some node is chosen, then all of its children must have been chosen. For example,if ${ABC}$ is selected,then ${AB},{AC}$ , ${BC},A,B,C$ ,and the empty group-by must all be selected. Selecting group-bys this way produces a frontier in the lattice. Every group-by below the frontier is selected. Another interesting point from [18] is that the BPUS algorithm in [10] also tends to pick group-bys towards the bottom of the lattice.

由于 PBS 按分组操作的大小进行选择，如果选择了某个节点，那么它的所有子节点也必须被选择。例如，如果选择了 ${ABC}$，那么 ${AB},{AC}$、${BC},A,B,C$ 和空分组操作都必须被选择。以这种方式选择分组操作会在格中产生一个边界。边界以下的每个分组操作都会被选择。文献 [18] 中另一个有趣的点是，文献 [10] 中的 BPUS 算法也倾向于选择格底部的分组操作。

BUC can easily be extended to compute only selected aggregates. When PBS is used, BUC can stop when it hits the frontier. This means that BUC can compute only the selected group-bys, an no others.

BUC 可以很容易地扩展为仅计算选定的聚合。当使用 PBS 时，BUC 遇到边界时可以停止。这意味着 BUC 可以仅计算选定的分组操作，而不计算其他分组操作。

### 5.6 Minimizing Aggregations

### 5.6 最小化聚合

Since BUC proceeds bottom-up, it does not take advantage of algebraic functions to reduce the aggregation costs. On the positive side, BUC can efficiently compute holistic functions, unlike most of the previous algorithms (MemoryCube can efficiently compute holistic functions as well.) In our experiments we found that aggregation costs were only small percent of the processing costs. Even when computing 16 aggregates on a full CUBE,less than $1/4$ of the processing time was attributed to aggregation (see Section 6.3). However, BUC can take advantage of algebraic functions by aggregating the results of the recursive call from any one iteration of the loop at line 8 . This complicates the algorithm a bit, and when we implemented it, BUC actually ran more slowly!

由于 BUC 是自底向上进行的，它没有利用代数函数来降低聚合成本。从积极的方面来看，与大多数先前的算法不同，BUC 可以有效地计算整体函数（内存立方体（MemoryCube）也可以有效地计算整体函数）。在我们的实验中，我们发现聚合成本仅占处理成本的一小部分。即使在完整立方体上计算 16 个聚合时，不到 $1/4$ 的处理时间用于聚合（见 6.3 节）。然而，BUC 可以通过聚合循环第 8 行中任何一次递归调用的结果来利用代数函数。这会使算法稍微复杂一些，而且当我们实现它时，BUC 实际上运行得更慢了！

### 5.7 Memory Requirements

### 5.7 内存需求

BUC relies on a significant, but reasonable, amount of working memory. As mentioned previously, BUC tries to fit a partition of tuples in memory as soon as possible. Say the partition has $N$ tuples,and each tuple requires $T$ bytes. Our implementation uses pointers to the tuples, and CountingSort requires a second set of pointers for temporary use. Let ${C}_{1}\ldots {C}_{d}$ be the cardinality of each dimension,and ${C}_{\max }$ be the maximum cardinality. CountingSort uses ${C}_{\max }$ counters,and BUC uses $\sum {C}_{i}$ counters. If the counters and pointers are each four bytes, the total memory requirement in bytes for BUC is:

BUC 依赖于大量但合理的工作内存。如前所述，BUC 会尽快将元组的一个分区装入内存。假设该分区有 $N$ 个元组，每个元组需要 $T$ 字节。我们的实现使用指向元组的指针，计数排序（CountingSort）需要第二组指针用于临时使用。设 ${C}_{1}\ldots {C}_{d}$ 为每个维度的基数，${C}_{\max }$ 为最大基数。计数排序使用 ${C}_{\max }$ 个计数器，BUC 使用 $\sum {C}_{i}$ 个计数器。如果计数器和指针每个都是 4 字节，那么 BUC 的总内存需求（以字节为单位）为：

$$
N\left( {T + 8}\right)  + 4\mathop{\sum }\limits_{{i = 1}}^{d}{C}_{i} + 4{C}_{\max }
$$

The memory requirements can be reduced by switching to QuickSort. When using QuickSort, the second set of pointers and all of the counters are not needed. Also, the counters in BUC are not necessary to use CountingSort. If the counters are not used, BUC will have to search for the partition boundaries.

通过切换到快速排序（QuickSort）可以减少内存需求。使用快速排序时，不需要第二组指针和所有计数器。此外，BUC 中的计数器对于使用计数排序也不是必需的。如果不使用计数器，BUC 将不得不搜索分区边界。

## 6 Performance Analysis

## 6 性能分析

We received an executable for MemoryCube from Prof. Ken Ross. Receiving his executable not only saved us time, but also allowed us to do a fair comparison with code that they optimized. Their implementation included a number of performance improvements that are not described in [14], but they are expected to appear in the forthcoming journal version of the paper. It is sufficient to say that they report this version is three times faster than the original version used in [14].

我们从肯·罗斯（Ken Ross）教授那里获得了内存立方体（MemoryCube）的可执行文件。获得他的可执行文件不仅为我们节省了时间，还使我们能够与他们优化后的代码进行公平比较。他们的实现包含了一些文献 [14] 中未描述的性能改进，但预计这些改进将出现在该论文即将发表的期刊版本中。可以说，他们报告这个版本比文献 [14] 中使用的原始版本快三倍。

We implemented BUC for main memory only (no external partitioning). The implementation of MemoryCube had the same restriction because it did not come with Partitioned-Cube. This is not a problem because PartitionedCube / MemoryCube and BUC have equivalent I/O performance. We ran our tests on a ${300}\mathrm{{MHz}}$ Sun Ultra 10 Workstation with 256MB of RAM. We measured the elapsed time, but since the implementation of MemoryCube reads text files, we did not count the time to read the file, and we did not output any results. The input to the programs was all integer data, and all the dimensions were packed between 0 and their cardinality. We estimated the I/O time based upon the number of tuples in the input and output, assuming no partitioning was required. Our system had a sequential I/O rate of $5\mathrm{{MB}}/\mathrm{{sec}}$ ,so we used that figure in estimating $\mathrm{I}/\mathrm{O}$ times.

我们仅针对主内存实现了BUC算法（不进行外部分区）。MemoryCube的实现也有同样的限制，因为它没有采用分区立方体（Partitioned - Cube）。这不是问题，因为分区立方体/内存立方体（PartitionedCube / MemoryCube）和BUC具有相当的I/O性能。我们在一台配备256MB内存的${300}\mathrm{{MHz}}$ Sun Ultra 10工作站上运行测试。我们测量了经过的时间，但由于MemoryCube的实现需要读取文本文件，所以我们没有计算读取文件的时间，也没有输出任何结果。程序的输入均为整数数据，所有维度都被压缩在0到其基数之间。我们根据输入和输出中的元组数量来估算I/O时间，假设不需要进行分区。我们的系统顺序I/O速率为$5\mathrm{{MB}}/\mathrm{{sec}}$，因此我们使用该数值来估算$\mathrm{I}/\mathrm{O}$时间。

### 6.1 Full CUBE Computation

### 6.1 全立方体（Full CUBE）计算

The first experiment compares BUC with MemoryCube for full CUBE computation. We randomly generated one million tuples. We varied the number of dimensions (group-by attributes) from 2 to 11 (11 was a compiled limit for MemoryCube). We repeated the experiment with three different cardinalities: 10,100 and 1000 (all dimensions had the same cardinality). The results are shown in Figure 7. The graphs show a number of interesting points:

第一个实验比较了BUC和MemoryCube在全立方体计算方面的性能。我们随机生成了一百万个元组。我们将维度（分组属性）的数量从2变化到11（11是MemoryCube的编译限制）。我们使用三种不同的基数（10、100和1000，所有维度的基数相同）重复进行了实验。结果如图7所示。这些图表显示了一些有趣的点：

- The CUBE gets more sparse as the number of dimensions increases and as the cardinality increases. As a result, the output gets extremely large, so the output time dominates the cost of computation. The same result was observed in [14].

- 随着维度数量和基数的增加，立方体（CUBE）变得更加稀疏。因此，输出变得极其庞大，所以输出时间主导了计算成本。在文献[14]中也观察到了相同的结果。

- With a cardinality of 10, BUC and MemoryCube have comparable performance. BUC begins to improve on MemoryCube at 10 dimensions.

- 当基数为10时，BUC和MemoryCube的性能相当。在10个维度时，BUC的性能开始优于MemoryCube。

- As cardinality increases, the time for MemoryCube marginally increases or stays about the same. BUC, however, dramatically improves as the cardinality increases. With 11 dimensions and a cardinality of 1000 , BUC is over 4 times faster than MemoryCube. The reason that BUC improves so much is that the CUBE is getting significantly more sparse as cardinality increases, ${}^{3}$ so BUC can stop partitioning and aggregating and simply write the answer to all ancestors. (Even though we do not output the result, we still make the correct number of calls the output function.)

- 随着基数的增加，MemoryCube的计算时间略有增加或基本保持不变。然而，BUC的性能随着基数的增加而显著提高。在11个维度且基数为1000的情况下，BUC比MemoryCube快4倍多。BUC性能大幅提升的原因是，随着基数的增加，立方体变得明显更加稀疏，${}^{3}$因此BUC可以停止分区和聚合操作，直接将答案写入所有祖先节点。（即使我们不输出结果，我们仍然会正确调用输出函数。）

### 6.2 Iceberg-CUBE Computation

### 6.2 冰山立方体（Iceberg - CUBE）计算

This experiment explores the effect of minimum support pruning in BUC. The input is the 11 dimensional data from the previous section. The cardinality is again 10,100 , or 1000. The minimum support was 1 (i.e., full CUBE), 2, 10 , or 100. (Remember that the minimum support is the minimum number of records for a group-by partition to be output. A minimum support of 10 is ${0.001}\%$ of the data.)

这个实验探讨了BUC中最小支持度剪枝的效果。输入是上一节中的11维数据。基数仍然是10、100或1000。最小支持度为1（即全立方体）、2、10或100。（请记住，最小支持度是分组分区输出所需的最小记录数。最小支持度为10是数据的${0.001}\%$。）

The results are shown in Figure 8 and Figure 9. A minimum support of 10 decreases the time for BUC significantly: ${37}\% ,{75}\%$ ,and ${85}\%$ for cardinalities10,100, and 1000 respectively. In addition, MemoryCube now takes twice as long as BUC for a cardinality of 10 . The other major effect is the I/O time no longer dominates the computation, even with a minimum support of 2 .

结果如图8和图9所示。最小支持度为10时，BUC的计算时间显著减少：对于基数分别为10、100和1000的情况，分别减少了${37}\% ,{75}\%$和${85}\%$。此外，当基数为10时，MemoryCube的计算时间是BUC的两倍。另一个主要影响是，即使最小支持度为2，I/O时间也不再主导计算。

### 6.3 Additional Aggregates

### 6.3 额外聚合

BUC does not try to share the computation of aggregates between parent and child group-bys, only the partitioning costs. To verify that partitioning is the major expense, not aggregation, we computed a full CUBE on 10 dimensional data with a cardinality of 100 and one million tuples. The results are shown in Figure 10. Computing one aggregate accounts for less than $7\%$ of the total cost. Computing 16 aggregates is still only ${23}\%$ of the total cost. If any algorithm sacrifices partitioning to try and overlap the aggregate computations, these percentages will only decrease. This suggests that optimizing the partitioning is the right approach for sparse CUBEs.

BUC不会尝试在父分组和子分组之间共享聚合计算，仅共享分区成本。为了验证分区是主要开销，而不是聚合，我们对基数为100、包含一百万个元组的10维数据计算了全立方体。结果如图10所示。计算一个聚合的成本不到总成本的$7\%$。计算16个聚合的成本仍然仅占总成本的${23}\%$。如果任何算法为了尝试重叠聚合计算而牺牲分区，这些百分比只会降低。这表明，优化分区是处理稀疏立方体的正确方法。

### 6.4 PBS

### 6.4 PBS

We ran an experiment to determine how BUC and Mem-oryCube compare when used with PBS. We generated a 10 dimensional dataset with a cardinality of 100 . Since all the dimensions are the same, every group-by with the same number of dimensions has the same estimated size (e.g., $\left| {AB}\right|  = \left| {AC}\right|  = \left| {BC}\right|$ ). The implementation of Mem-oryCube had an option to limit the maximum number of attributes used in a group-by (i.e., the maximum number of non-ALL values), and we implemented the same feature in BUC. We varied the maximum number of grouping dimensions from 0 to 10 . Figure 11 shows that MemoryCube and BUC followed a similar trend, but that BUC was always significantly faster. The performance of MemoryCube did not change between 0 and 2 dimensions, probably because their implementation is not optimized for this case.

我们进行了一项实验，以确定在与PBS（并行位图扫描，Parallel Bitmap Scanning）一起使用时，BUC（Bottom-Up Cubing）和MemoryCube的性能对比情况。我们生成了一个基数为100的10维数据集。由于所有维度都相同，因此具有相同维度数量的每个分组操作（group-by）的估计大小都相同（例如，$\left| {AB}\right|  = \left| {AC}\right|  = \left| {BC}\right|$ ）。MemoryCube的实现有一个选项，可以限制分组操作中使用的最大属性数量（即非ALL值的最大数量），我们在BUC中也实现了相同的功能。我们将分组维度的最大数量从0变化到10。图11显示，MemoryCube和BUC呈现出相似的趋势，但BUC始终明显更快。MemoryCube在0到2个维度之间的性能没有变化，可能是因为其实现没有针对这种情况进行优化。

---

<!-- Footnote -->

${}^{3}$ Even though the CUBE gets more sparse as dimensionality increases, the problem is still exponentially harder, so BUC can never get faster with added dimensions.

${}^{3}$ 尽管随着维度的增加，数据立方体（CUBE）变得更加稀疏，但问题的复杂度仍然呈指数级增长，因此BUC不会因为增加维度而变得更快。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Cardinality $= {10}$ Cardinality $= {100}$ Cardinality $= {1000}$ BUC 2500 BUC MemoryCube 2000 MemoryCube Est. I/O Time/ Est. I/O Time Time (sec) 1500 1000 500 0 8 10 8 10 Dimensions Dimensions 2500 2500 2000 MemoryCube 2000 Est. I/O Time Time (sec) 1500 Time (sec) 1500 1000 1000 500 500 0 0 2 4 8 10 2 Dimensions -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_9.jpg?x=74&y=0&w=1531&h=418&r=0"/>

Figure 7: Full CUBE computation

图7：全数据立方体（Full CUBE）计算

<!-- figureText: Dimensions = 11 Dimensions = 11 600 Card $= {100}$ 500 Card=1000 Time (sec) 400 300 200 100 0 70 80 90100 0 8 12 16 Minimum Support Number of Aggregates Figure 9: Est. I/O with min. support Figure 10: Additional aggregates 1000 BUC(1) Time (sec) 500 BUC(100) BUC(1)-dup Skew First MemoryCube Skew Last 2 0 2 6 10 Skew Number of Skewed Dimensions Figure 12: Increasing skew Figure 13: Skewed dimension order BUC BUC-Dedup MemoryCube Est. I/O Time 40 60 80 100 Minimum Support 2000 2000 Card $= {10}$ Card=100 1500 Card=1000 1500 Time (sec) 1000 500 Time (sec) 1000 500 0 0 0 10 20 30 40 60 80 90 100 10 20 30 Minimum Support Figure 8: BUC with min. support Cardinality $= {1000}$ 1500 BUC 1000 MemoryCube Time (sec) 1000 Time (sec) 500 500 2 4 6 8 10 0 Max Group-by Attributes Figure 11: Limited dimensions 500 2000 400 1500 Time (sec) 300 BUC Time (sec) 1000 200 MemoryCube Est. I/O Time 100 500 0 0 20 40 60 80 100 20 Minimum Support -->

<img src="https://cdn.noedgeai.com/0195c911-dbd0-7407-a430-dcafc8db3855_9.jpg?x=54&y=543&w=1545&h=1382&r=0"/>

Figure 15: Mail-order sales data

图15：邮购销售数据

Figure 14: Weather data

图14：气象数据

<!-- Media -->

### 6.5 Skew

### 6.5 数据倾斜（Skew）

As mentioned previously, BUC is sensitive to skew in the data. In all of the previous experiments, the data was generated uniformly (i.e., no skew). We ran an experiment on 10 dimensional data with cardinality of 100 that varied the skew simultaneously in all dimensions. We used a Zipf distribution to generate the data. Zipf uses a parameter $\alpha$ to determine the degree of skew. When $\alpha  = 0$ ,the data is uniform,and as $\alpha$ increases,the skew increases rapidly: at $\alpha  = 3$ ,the most frequent value occurred in about ${83}\%$ of the tuples.

如前所述，BUC对数据中的倾斜情况很敏感。在之前的所有实验中，数据都是均匀生成的（即没有倾斜）。我们对基数为100的10维数据进行了一项实验，在所有维度上同时改变倾斜程度。我们使用齐普夫分布（Zipf distribution）来生成数据。齐普夫分布使用一个参数 $\alpha$ 来确定倾斜程度。当 $\alpha  = 0$ 时，数据是均匀的，并且随着 $\alpha$ 的增加，倾斜程度迅速增加：在 $\alpha  = 3$ 时，最频繁的值出现在大约 ${83}\%$ 的元组中。

The results in Figure 12 show that the performance of BUC does degrade as skew increases. BUC with a minimum support of 100 even converges on BUC for full CUBE. The performance of MemoryCube, however, improved with skew because the implementation collapses duplicate group-by values. We added the duplicate collapsing code to BUC as described in Section 5.4. This version is called BUC-Dedup in the graph. With this modification, BUC degraded until the deduplication compensated for the loss of pruning. At which point, BUC and MemoryCube have similar performance.

图12中的结果表明，随着倾斜程度的增加，BUC的性能确实会下降。最小支持度为100的BUC甚至会收敛到全数据立方体的BUC。然而，MemoryCube的性能随着倾斜程度的增加而提高，因为其实现会合并重复的分组值。我们按照第5.4节的描述，为BUC添加了合并重复项的代码。这个版本在图中称为BUC-Dedup。通过这种修改，BUC的性能会下降，直到去重操作弥补了剪枝损失。此时，BUC和MemoryCube的性能相似。

We ran another experiment were we varied the number of skew dimensions, each with the same cardinality. Figure 13 shows that placing the skewed dimensions last in the dimension ordering is significantly better than placing the skewed dimensions first.

我们进行了另一项实验，改变倾斜维度的数量，每个维度的基数相同。图13显示，将倾斜维度放在维度排序的最后，明显比将倾斜维度放在最前面要好。

### 6.6 Weather Data

### 6.6 气象数据

Figure 14 shows the time for MemoryCube and BUC on a real nine-dimensional dataset containing weather conditions at various weather stations on land for September 1985 [9]. The dataset contained 1,015,367 tuples. The attributes were ordered by cardinality: station-id (7037), longitude (352), solar-altitude (179), latitude (152), present-weather (101), day (30), weather-change-code (10), hour (8), and brightness (2). Many of the attributes were highly skewed, and some of the attributes were significantly correlated (e.g., only one station was at one (latitude, longitude)).

图14显示了MemoryCube和BUC在一个真实的九维数据集上的运行时间，该数据集包含1985年9月陆地上各个气象站的气象条件 [9]。该数据集包含1,015,367个元组。属性按基数排序：站点ID（7037）、经度（352）、太阳高度（179）、纬度（152）、当前天气（101）、日期（30）、天气变化代码（10）、小时（8）和亮度（2）。许多属性存在高度倾斜，并且一些属性之间存在显著的相关性（例如，只有一个站点位于一个（纬度，经度）位置）。

This experiment shows that BUC is effective on real data, even with high skew and correlation. BUC is 2 times faster than MemoryCube for full CUBE computation, and 3.5 times faster when minimum support is 10 . The graph also shows that a minimum support of just 2 tuples significantly reduces the I/O cost (4.3 times faster). With a minimum support of 10 , the I/O costs drop drastically ( 39 times faster than full CUBE). We also ran BUC with the code to collapse duplicates. For full CUBE, this version of BUC ran in 167 seconds,which is a ${20}\%$ improvement.

这个实验表明，即使在存在高度倾斜和相关性的真实数据上，BUC仍然有效。在全数据立方体计算中，BUC比MemoryCube快2倍，当最小支持度为10时，快3.5倍。该图还显示，仅2个元组的最小支持度就显著降低了I/O成本（快4.3倍）。当最小支持度为10时，I/O成本急剧下降（比全数据立方体快39倍）。我们还运行了带有合并重复项代码的BUC。对于全数据立方体，这个版本的BUC运行时间为167秒，这是一个 ${20}\%$ 的改进。

### 6.7 Mail-order Data

### 6.7 邮购数据

We ran MemoryCube and BUC on a second real dataset. This data is sales data from a mail-order clothing company. We limited the dataset to two million tuples to keep the relation in memory. The dataset has ten dimensions: the first three digits of the customer's zip code (920), product number (793), add space in the catalog (361), order date (319), page in the catalog (212), category (40), colors (21), gender of the product (8), catalog id (2), and focus indicator (2). This dataset contains extreme correlation. The product number, page, category, colors, gender, and focus attributes are all strongly correlated. Collapsing duplicates on all of the group-bys (i.e.,creating the $\left\langle  {{D}_{1},{D}_{2},\ldots ,{D}_{10}}\right\rangle$ ) produced less than 1.4 million distinct tuples.

我们在第二个真实数据集上运行了MemoryCube和BUC算法。该数据是一家邮购服装公司的销售数据。为了将关系存储在内存中，我们将数据集限制为二百万条元组。该数据集有十个维度：客户邮政编码的前三位（920）、产品编号（793）、产品目录中的空白处（361）、订单日期（319）、产品目录中的页码（212）、类别（40）、颜色（21）、产品性别（8）、产品目录编号（2）和重点指标（2）。这个数据集存在极强的相关性。产品编号、页码、类别、颜色、性别和重点属性都高度相关。对所有分组依据进行去重操作（即创建$\left\langle  {{D}_{1},{D}_{2},\ldots ,{D}_{10}}\right\rangle$）后，得到的不同元组数量少于一百四十万条。

The results of the experiment are depicted in Figure 15. Even with the correlation, BUC is still 2 times faster than MemoryCube for a full CUBE. With duplicate elimination, BUC becomes 8 times faster than MemoryCube, and with a minimum support of 10 , BUC is 14.6 times faster!

实验结果如图15所示。即使存在相关性，对于完整的CUBE操作，BUC算法仍比MemoryCube算法快两倍。在进行去重操作后，BUC算法比MemoryCube算法快八倍；当最小支持度为10时，BUC算法比MemoryCube算法快14.6倍！

## 7 Conclusions

## 7 结论

We introduced the Iceberg-CUBE problem and demonstrated its viability as an alternative to static selection of group-bys. We discussed how Iceberg-CUBE relates to full CUBE computation, multi-dimensional association rules, and iceberg queries.

我们引入了冰山立方体（Iceberg-CUBE）问题，并证明了它作为分组静态选择替代方案的可行性。我们讨论了冰山立方体与全立方体计算、多维关联规则和冰山查询之间的关系。

We presented a novel algorithm called BUC for Iceberg-CUBE and sparse CUBE computation. BUC builds the CUBE from the most aggregated group-bys to the least aggregated, which allows BUC to share partitioning costs and to prune the computation. We also described how BUC complements group-by selection algorithms like PBS. BUC can be extended to support dimension hierarchies, and it can be easily parallelized. Exactly the best way to implement these features is left for future research.

我们提出了一种名为BUC的新颖算法，用于冰山立方体和稀疏立方体计算。BUC从聚合程度最高的分组构建到聚合程度最低的分组，这使得BUC能够共享分区成本并对计算进行剪枝。我们还描述了BUC如何补充像PBS这样的分组选择算法。BUC可以扩展以支持维度层次结构，并且可以轻松并行化。具体实现这些功能的最佳方法留待未来研究。

Our experiments demonstrated that BUC is significantly faster at computing full sparse CUBEs than its closest competitor, MemoryCube. For example, BUC was eight times faster than MemoryCube on one real dataset. For Iceberg-CUBE queries, our experiments also showed that BUC improves upon its own performance, with speedups of up to four times with a minimum support of ten tuples.

我们的实验表明，在计算全稀疏立方体时，BUC比其最接近的竞争对手MemoryCube显著更快。例如，在一个真实数据集上，BUC比MemoryCube快八倍。对于冰山立方体查询，我们的实验还表明，BUC在自身性能上有所提升，在最小支持度为十个元组的情况下，速度提升可达四倍。

## Acknowledgements

## 致谢

We thank Prof. Ed Robertson for his comments on a previous version of this paper. We are particularly grateful to Prof. Ken Ross for his helpful comments and for supplying his implementation of MemoryCube. We also thank the anonymous referees for their comments. This work was supported in part by ORD contract 144-ET33 and NSF research grant IIS-9802882

我们感谢埃德·罗伯逊（Ed Robertson）教授对本文前一版本的评论。我们特别感谢肯·罗斯（Ken Ross）教授的有益评论以及提供他实现的MemoryCube。我们也感谢匿名评审人员的评论。这项工作部分得到了ORD合同144 - ET33和美国国家科学基金会（NSF）研究资助IIS - 9802882的支持。

## References

## 参考文献

[1] S. Agarwal, R. Agrawal, P. M. Deshpande, A. Gupta, J. F. Naughton, R. Ramakrishnan, and S. Sarawagi. On the computation of multidimensional aggregates. In Proc. of the 22nd VLDB Conf., pages 506-521, 1996.

[2] R. Agrawal and R. Srikant. Fast algorithms for mining association rules. In Proc. of the 20th VLDB Conf., pages 487-499, Santiago, Chile, Sept. 1994.

[2] R. Agrawal和R. Srikant。挖掘关联规则的快速算法。见《第20届VLDB会议论文集》，第487 - 499页，智利圣地亚哥，1994年9月。

[3] E. Baralis, S. Paraboschi, and E. Teniente. Materialized view selection in a multidimensional database. In Proc. of the 23rd VLDB Conf., pages 98-112, Delphi, Greece, 1997.

[4] P. M. Deshpande, S. Agarwal, J. F. Naughton, and R. Ramakrishnan. Computation of multidimensional aggregates. Technical Report 1314, University of Wisconsin - Madison, 1996.

[5] M. Fang, N. Shivakumar, H. Garcia-Molina, R. Mot-wani, and J. D. Ullman. Computing iceberg queries efficiently. In Proc. of the 24th VLDB Conf., pages 299-310, New York, New York, August 1998.

[6] J. Gray, A. Bosworth, A. Layman, and H. Pirahesh. Datacube: A relational aggregation operator generalizing group-by, cross-tab, and sub-totals. In Proc. of the IEEE ICDE, pages 152-159, 1996.

[7] H. Gupta. Selection of views to materialize in a data warehouse. In Proc. of the 6th ICDT, pages 98-112, Delphi, Greece, 1997.

[8] H. Gupta, V. Harinarayan, A. Rajaraman, and J. D. Ullman. Index selection for OLAP. In Proc. of the 13th ICDE, pages 208-219, Manchester, UK, 1997.

[9] C. Hahn, S. Warren, and J. London. Edited synoptic cloud reports from ships and land stations over the globe, 1982-1991. http://cdiac.esd.ornl.gov/- cdiac/ndps/ndp026b.html, http://cdiac.esd.ornl.gov/- ftp/ndp026b/SEP85L.Z, 1994.

[10] V. Harinarayan, A. Rajaraman, and J. D. Ullman. Implementing data cubes effeciently. In Proc. of the ACM SIGMOD Conf., pages 205-216, 1996.

[11] M. Kamber, J. Han, and J. Y. Chiang. Metarule-guided mining of multi-dimensional association rules using data cubes. In Proceeding of the 3rd Intl. Conf. on Knowledge Discovery and Data Mining, Newport Beach, CA, Aug. 1997. Also Techical Report CS-TR 97-10, School of Computing Science, Simon Fraser University, May 1997.

[12] R. Kimball. The Data Warehouse Toolkit. John Wiley and Sons, Inc, 1996.

[13] R. T. Ng, L. V. Lakshmanan, J. Han, and A. Pang. Exploratory mining and pruning optimizations of constrained associations rules. In Proc. of the ACM-SIGMOD Conf. on Management of Data, pages 13-24, Seattle, WA, June 1998.

[14] K. A. Ross and D. Srivastava. Fast computation of sparse datacubes. In Proc. of the 23rd VLDB Conf., pages 116-125, Athens, Greece, 1997.

[15] K. A. Ross and K. A. Zaman. Optimizing selections over data cubes. Technical Report CUCS-018-98, Columbia University, Nov 1998. http://www.cs.- columbia.edu/ library/1998.html.

[16] S. Sarawagi, R. Agrawal, and A. Gupta. On computing the data cube. Technical Report RJ10026, IBM Almaden Research Center, San Jose, CA, 1996.

[17] R. Sedgewick. Algorithms in $C$ ,chapter Chapter 8,page 112. Addison-Wesley Publishing Company, 1990.

[18] A. Shukla, P. M. Deshpande, and J. F. Naughton. Materialized view selection for multidimensional datasets. In Proc. of the 24th VLDB Conf., pages 488-499, New York, New York, August 1998.

[19] Y. Zhao, P. M. Deshpande, and J. F. Naughton. An array-based algorithm for simultaneous multidimensional aggregates. In Proc. of the ACM SIGMOD Conf., pages 159-170, 1997.