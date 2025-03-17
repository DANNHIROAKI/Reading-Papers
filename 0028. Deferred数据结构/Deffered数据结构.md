# DEFERRED DATA STRUCTURING*

# 延迟数据结构化*

RICHARD M. KARP†, RAJEEV MOTWANI†, AND PRABHAKAR RAGHAVAN‡

理查德·M·卡普（RICHARD M. KARP）†、拉吉夫·莫特瓦尼（RAJEEV MOTWANI）†和普拉巴卡尔·拉加万（PRABHAKAR RAGHAVAN）‡

Abstract. We consider the problem of answering a series of on-line queries on a static data set. The conventional approach to such problems involves a preprocessing phase which constructs a data structure with good search behavior. The data structure representing the data set then remains fixed throughout the processing of the queries. Our approach involves dynamic or query-driven structuring of the data set; our algorithm processes the data set only when doing so is required for answering a query. A data structure constructed progressively in this fashion is called a deferred data structure.

摘要。我们考虑在静态数据集上回答一系列在线查询的问题。解决此类问题的传统方法包括一个预处理阶段，该阶段构建一个具有良好搜索性能的数据结构。表示数据集的数据结构在查询处理过程中保持固定。我们的方法涉及对数据集进行动态或由查询驱动的结构化；我们的算法仅在回答查询需要时才处理数据集。以这种方式逐步构建的数据结构称为延迟数据结构。

We develop the notion of deferred data structures by solving the problem of answering membership queries on an ordered set. We obtain a randomized algorithm which achieves asymptotically optimal performance with high probability. We then present optimal deferred data structures for the following problems in the plane: testing convex-hull membership, half-plane intersection queries and fixed-constraint multi-objective linear programming. We also apply the deferred data structuring technique to multidimensional dominance query problems.

我们通过解决在有序集上回答成员查询的问题来发展延迟数据结构的概念。我们得到了一种随机算法，该算法很有可能实现渐近最优性能。然后，我们针对平面中的以下问题提出了最优延迟数据结构：测试凸包成员性、半平面相交查询和固定约束多目标线性规划。我们还将延迟数据结构化技术应用于多维支配查询问题。

Key words. data structure, preprocessing, query processing, lower bound, randomized algorithm, computational geometry, convex hull, linear programming, half-space intersection, dominance counting

关键词。数据结构、预处理、查询处理、下界、随机算法、计算几何、凸包、线性规划、半空间相交、支配计数

## AMS(MOS) subject classifications. ${68}\mathrm{P}{05},{68}\mathrm{P}{10},{68}\mathrm{P}{20},{68}\mathrm{Q}{20},{68}\mathrm{U}{05}$

## 美国数学学会（MOS）学科分类。${68}\mathrm{P}{05},{68}\mathrm{P}{10},{68}\mathrm{P}{20},{68}\mathrm{Q}{20},{68}\mathrm{U}{05}$

1. Introduction. We consider several search problems where we are given a set of $n$ elements,which we call the data set. We are required to answer a sequence of queries about the data set.

1. 引言。我们考虑几个搜索问题，在这些问题中，我们会得到一组 $n$ 个元素，我们将其称为数据集。我们需要回答一系列关于该数据集的查询。

The conventional approach to search problems consists of preprocessing the data set in time $p\left( n\right)$ ,thus building up a search structure that enables queries to be answered efficiently. Subsequently,each query can be answered in time $q\left( n\right)$ . The time needed for answering $r$ queries is thus $p\left( n\right)  + r \cdot  q\left( n\right)$ . Very often,a single query can be answered without preprocessing in time $o\left( {p\left( n\right) }\right)$ . The preprocessing approach is thus uneconomical unless the number of queries $r$ is sufficiently large.

解决搜索问题的传统方法是在时间$p\left( n\right)$内对数据集进行预处理，从而构建一个搜索结构，以便能够高效地回答查询。随后，每个查询都可以在时间$q\left( n\right)$内得到解答。因此，回答$r$个查询所需的时间为$p\left( n\right)  + r \cdot  q\left( n\right)$。很多时候，单个查询可以在不进行预处理的情况下在时间$o\left( {p\left( n\right) }\right)$内得到解答。因此，除非查询数量$r$足够大，否则预处理方法并不经济。

We present here an alternative to preprocessing, in which the search structure is built up "on-the-fly" as queries are answered. Throughout this paper we assume that an adversary generates a stream of queries which can cease at any point. Each query must be answered on-line, before the next one is received. If the adversary generates sufficiently many queries, we will show that we build up the complete search structure in time $O\left( {p\left( n\right) }\right)$ so that further queries can be answered in time $q\left( n\right)$ . If on the other hand the adversary generates few queries, we will show that the total work we expend in the process of answering them (which includes building the search structure partially) is asymptotically smaller than $p\left( n\right)  + r \cdot  q\left( n\right)$ . We thus perform at least as well as the preprocessing approach,and in fact better when $r$ is small. We do so with no a priori knowledge of $r$ . We call our approach deferred data structuring since we build up the search structure gradually as queries arrive, rather than all at once. In some cases we show that our deferred data structuring algorithm is of nearly optimal efficiency, even in comparison with algorithms that know $r$ ,the number of queries,in advance.

我们在此提出一种预处理的替代方法，在该方法中，搜索结构是在回答查询时“即时”构建的。在本文中，我们假设存在一个对手生成一系列查询，这些查询可能在任何时刻停止。每个查询都必须在收到下一个查询之前在线回答。如果对手生成了足够多的查询，我们将证明我们能在时间$O\left( {p\left( n\right) }\right)$内构建出完整的搜索结构，以便后续查询能在时间$q\left( n\right)$内得到回答。另一方面，如果对手生成的查询较少，我们将证明在回答这些查询的过程中（包括部分构建搜索结构）所花费的总工作量渐近小于$p\left( n\right)  + r \cdot  q\left( n\right)$。因此，我们的方法至少和预处理方法表现一样好，实际上，当$r$较小时，我们的方法表现更好。而且我们在事先不知道$r$的情况下就能做到这一点。我们将这种方法称为延迟数据结构化，因为我们是随着查询的到来逐步构建搜索结构，而不是一次性构建。在某些情况下，我们证明了我们的延迟数据结构化算法几乎具有最优效率，即使与那些事先知道查询数量$r$的算法相比也是如此。

---

<!-- Footnote -->

* Received by the editors December 22, 1986, accepted for publication (in revised form) November 10, 1987.

* 编辑于1986年12月22日收到，于1987年11月10日接受发表（修订版）。

† Computer Science Division, University of California, Berkeley, California 94720. The work of the first two authors was supported in part by the National Science Foundation under grant DCR-8411954. The results in § 4 first appeared in R. Motwani and P. Raghavan, Deferred data structures: query-driven preprocessing for geometric search problems, in Proc. 2nd Annual ACM Symposium on Computational Geometry, Yorktown Heights, NY, June 1986, pp. 303-312.

† 加利福尼亚大学伯克利分校计算机科学系，加利福尼亚州伯克利市，邮编94720。前两位作者的工作部分得到了美国国家科学基金会（National Science Foundation）资助，资助编号为DCR - 8411954。第4节中的结果首次发表于R. 莫特瓦尼（R. Motwani）和P. 拉加万（P. Raghavan）的论文《延迟数据结构：几何搜索问题的查询驱动预处理》，该论文发表于1986年6月在纽约约克敦海茨（Yorktown Heights）举行的第二届美国计算机协会计算几何年度研讨会（2nd Annual ACM Symposium on Computational Geometry）会议录，第303 - 312页。

‡IBM Thomas J. Watson Research Center, Yorktown Heights, New York 10598. The work of this author was supported in part by an IBM Doctoral Fellowship while he was a graduate student at the Computer Science Division, University of California, Berkeley, California 94720.

‡ 国际商业机器公司（IBM）托马斯·J·沃森研究中心（IBM Thomas J. Watson Research Center），纽约州约克敦海茨，邮编10598。这位作者的工作部分得到了IBM博士奖学金的支持，当时他是加利福尼亚大学伯克利分校计算机科学系的研究生，该校位于加利福尼亚州伯克利市，邮编94720。

<!-- Footnote -->

---

In $§2$ we exemplify our approach through the membership query problem. We determine the complexity of answering $r$ queries on $n$ elements under the comparison tree model. In $§3$ we present a randomized algorithm for the membership query problem whose performance matches an information-theoretic lower bound (ignoring asymptotically smaller additive terms). We then proceed to exhibit deferred data structure for several geometry problems. In $\$ 4$ we show that deferred data structuring is optimal for the following two-dimensional geometric problems: (1) Given $n$ points in the plane, to determine whether a query point lies inside their convex hull. (2) Given $n$ half-planes,to determine whether a query point lies in their common intersection. (3) Given $n$ linear constraints in two variables,to optimize a query objective function (also linear). Our algorithms are proved optimal by means of a tight lower bound (under the algebraic computation tree model) in $§{4.4}$ . In $§5$ we consider dominance problems in $d$ -space. We present theorems about the deferred construction of Bentley’s ECDF search tree [2].

在$§2$中，我们通过成员查询问题来举例说明我们的方法。我们确定了在比较树模型下，对$n$个元素回答$r$个查询的复杂度。在$§3$中，我们提出了一种针对成员查询问题的随机算法，其性能与信息论下界相匹配（忽略渐近较小的附加项）。然后，我们将为几个几何问题展示延迟数据结构。在$\$ 4$中，我们表明延迟数据结构对于以下二维几何问题是最优的：(1) 给定平面上的$n$个点，确定一个查询点是否位于它们的凸包内。(2) 给定$n$个半平面，确定一个查询点是否位于它们的公共交集内。(3) 给定两个变量的$n$个线性约束，优化一个查询目标函数（也是线性的）。在$§{4.4}$中，我们通过一个严格的下界（在代数计算树模型下）证明了我们的算法是最优的。在$§5$中，我们考虑$d$维空间中的支配问题。我们给出了关于本特利（Bentley）的经验累积分布函数（ECDF）搜索树[2]的延迟构建的定理。

In this paper all logarithms are to the base two.

在本文中，所有对数均以2为底。

2. General principles of deferred data structuring. In this section we develop the basic ideas involved in deferred data structuring. Let $X = \left\{  {{x}_{1},{x}_{2},\cdots ,{x}_{n}}\right\}$ be a set of $n$ elements drawn from a totally ordered set $U$ . Consider a series of queries where each query ${q}_{j}$ is an element of $U$ ; for each query,we must determine whether it is present in $X$ .

2. 延迟数据结构化的一般原则。在本节中，我们将阐述延迟数据结构化所涉及的基本概念。设$X = \left\{  {{x}_{1},{x}_{2},\cdots ,{x}_{n}}\right\}$是从全序集$U$中选取的$n$个元素组成的集合。考虑一系列查询，其中每个查询${q}_{j}$都是$U$中的一个元素；对于每个查询，我们必须确定它是否存在于$X$中。

If we had to answer just one query,we could simply compare the query ${q}_{1}$ to every member of $X$ and answer the query in $O\left( n\right)$ comparison operations. This would be the preferred method for answering a small number of queries. On the other hand, if we knew that the number of queries $r$ were large,we could first sort the elements of $X$ in $p\left( n\right)  = O\left( {n\log n}\right)$ operations,these building up a binary search tree ${T}_{X}$ for the elements of $X$ . We could then do a binary search costing $Q\left( n\right)  = O\left( {\log n}\right)$ comparisons for each query; this takes $O\left( {\left( {n + r}\right)  \cdot  \log n}\right)$ comparisons.

如果我们只需回答一个查询，我们可以简单地将查询 ${q}_{1}$ 与 $X$ 中的每个元素进行比较，并通过 $O\left( n\right)$ 次比较操作来回答该查询。这将是回答少量查询的首选方法。另一方面，如果我们知道查询数量 $r$ 很大，我们可以首先通过 $p\left( n\right)  = O\left( {n\log n}\right)$ 次操作对 $X$ 中的元素进行排序，从而为 $X$ 中的元素构建一个二叉搜索树 ${T}_{X}$。然后，对于每个查询，我们可以进行一次成本为 $Q\left( n\right)  = O\left( {\log n}\right)$ 次比较的二分查找；这需要 $O\left( {\left( {n + r}\right)  \cdot  \log n}\right)$ 次比较。

We proceed to determine the complexity (number of comparisons) of answering $r$ queries on the set $X$ ; we do not know $r$ a priori,and each query is to be answered before we know of the next one.

我们接下来确定在集合$X$上回答$r$个查询的复杂度（比较次数）；我们事先并不知道$r$的值，并且每个查询都要在得知下一个查询之前给出答案。

2.1. The lower bound. We first prove an information-theoretic lower bound for this problem.

2.1. 下界。我们首先为这个问题证明一个信息论下界。

THEOREM 1. The number of comparisons needed to process $r$ queries is at least $\left( {n + r}\right)  \cdot  \log \left( {\min \{ n,r\} }\right)  - O\left( {\min \{ n,r\} }\right)$ in the worst case.

定理1. 在最坏情况下，处理$r$个查询所需的比较次数至少为$\left( {n + r}\right)  \cdot  \log \left( {\min \{ n,r\} }\right)  - O\left( {\min \{ n,r\} }\right)$。

Remark. Note that neither of the strategies mentioned above (linear search, or sorting followed by binary search) achieves this bound for all $r \leqq  n$ .

注记。注意，上述提到的两种策略（线性搜索，或先排序再进行二分搜索）都无法对所有的$r \leqq  n$达到这个下界。

Proof. If we could collect the $r$ queries and process them off-line,we would have an instance of the SET INTERSECTION problem where we have to find the elements common to the sets $X = \left\{  {{x}_{1},{x}_{2},\cdots ,{x}_{n}}\right\}$ and $Q = \left\{  {{q}_{1},\cdots ,{q}_{r}}\right\}$ . We will prove a lower bound of $\Omega \left( {\left( {n + r}\right)  \cdot  \log \left( {\min \{ n,r\} }\right) }\right)$ comparisons for determining the intersection of two sets of cardinalities $n$ and $r$ . This off-line lower bound will hold a fortiori for the on-line case in which we are interested. We present the argument for the case $r \leqq  n$ ; the other case is symmetrical.

证明。如果我们能收集 $r$ 个查询并离线处理它们，我们就会得到一个集合交集问题的实例，在该问题中，我们必须找出集合 $X = \left\{  {{x}_{1},{x}_{2},\cdots ,{x}_{n}}\right\}$ 和 $Q = \left\{  {{q}_{1},\cdots ,{q}_{r}}\right\}$ 的公共元素。我们将证明，对于确定基数分别为 $n$ 和 $r$ 的两个集合的交集，需要进行 $\Omega \left( {\left( {n + r}\right)  \cdot  \log \left( {\min \{ n,r\} }\right) }\right)$ 次比较的下界。这个离线下界对于我们所关注的在线情况更是成立。我们针对 $r \leqq  n$ 的情况给出论证；另一种情况是对称的。

Since we are interested in lower bounds on this problem, we can restrict our attention to only those cases where $X \cap  Q = \varnothing$ . In this case the algorithm has to determine the relation of each element in $X$ to each element in $Q$ . An adversary can ensure that for any two elements in $Q$ there will be at least one in $X$ whose value lies between them. In other words,the elements of $Q$ will partition $X$ into at least $r - 1$ nonempty classes. Each such class will consist of all those members of $X$ which lie between two consecutive values in the total ordering of $Q$ . We shall give an information-theoretic lower bound by counting some ways of arranging $X$ and $Q$ to satisfy the above constraint.

由于我们对该问题的下界感兴趣，因此可以将注意力仅局限于那些满足$X \cap  Q = \varnothing$的情况。在这种情况下，算法必须确定$X$中的每个元素与$Q$中的每个元素之间的关系。对手可以确保，对于$Q$中的任意两个元素，$X$中至少会有一个元素的值介于它们之间。换句话说，$Q$中的元素将把$X$划分为至少$r - 1$个非空类。每个这样的类将由$X$中所有介于$Q$的全序中两个连续值之间的成员组成。我们将通过计算满足上述约束条件的$X$和$Q$的某些排列方式，给出一个信息论下界。

There are $r$ ! ways of ordering the elements in $Q$ . Given a total order on $Q$ ,there are(r - 1)! ways of separating the elements in $Q$ by some arbitrary $r - 1$ elements from $X$ . The remaining elements of $X$ can be placed arbitrarily. There are $r + 1$ available slots as determined by the $r$ ordered elements of $Q$ . This can be done in ${\left( r + 1\right) }^{n - r + 1}$ ways. Let $I$ be the total number of interleavings (of $X$ and $Q$ ) possible when $S \cap  Q = \varnothing$ . Then the number of possible arrangements specified above is a lower bound on the value of $I$ :

对$Q$中的元素进行排序有$r$!种方式。给定$Q$上的一个全序，用来自$X$的任意$r - 1$个元素分隔$Q$中的元素有(r - 1)!种方式。$X$中剩余的元素可以任意放置。由$Q$的$r$个有序元素确定有$r + 1$个可用的空位。这可以用${\left( r + 1\right) }^{n - r + 1}$种方式完成。设$I$为当$S \cap  Q = \varnothing$时（$X$和$Q$的）所有可能的交错排列的总数。那么上述指定的可能排列数是$I$值的一个下界：

$$
I \geqq  r! \cdot  \left( {r - 1}\right) ! \cdot  {\left( r + 1\right) }^{n - r + 1}.
$$

Since the algorithm has to identify one out of (at least) this many possible arrangements the lower bound is given by $\log I$ :

由于该算法必须从（至少）这么多可能的排列中识别出一种，因此下界由$\log I$给出：

$$
\log I \geqq  \left( {n + r}\right)  \cdot  \log r - {2r}\log e.
$$

Here $e$ represents the base of the natural logarithms.

这里$e$表示自然对数的底数。

2.2. Upper bounds. We now present two approaches to obtaining an upper bound which comes within a multiplicative constant factor of the lower bound. The first approach is based on merge-sort, while the second is based on recursively finding medians.

2.2. 上界。我们现在介绍两种获取上界的方法，该上界与下界相差一个乘法常数因子。第一种方法基于归并排序，而第二种方法基于递归地查找中位数。

2.2.1. An approach based on merge-sort. The following algorithm comes within a constant factor of the lower bound. It uses a recursive merge-sort technique to totally order the elements in $X$ . The merge-sort proceeds in $\log n$ stages. At the end of a stage the set $X$ is partitioned into a number of equal-sized totally ordered subsets called runs. Each stage pairs off all the runs resulting from the previous stage and merges them to create longer runs. These stages are interleaved with the processing of a set of queries, until a single totally ordered run remains, whereafter no more comparisons between elements of $X$ are required. To process a query implies a binary search through each of the existing runs. The number of queries processed between consecutive merging stages or,equivalently,the minimum length of a run before the $i$ th query,are chosen appropriately.

2.2.1. 基于归并排序的方法。以下算法得到的结果与下界相差一个常数因子。它使用递归归并排序技术对集合 $X$ 中的元素进行全排序。归并排序分 $\log n$ 个阶段进行。在每个阶段结束时，集合 $X$ 被划分为若干个大小相等的全序子集，称为游程（run）。每个阶段将上一阶段得到的所有游程两两配对并合并，以创建更长的游程。这些阶段与一组查询的处理过程交替进行，直到只剩下一个全序游程，此后就不再需要对集合 $X$ 中的元素进行比较。处理一个查询意味着对每个现有的游程进行二分查找。连续合并阶段之间处理的查询数量，或者等价地，第 $i$ 次查询之前游程的最小长度，都经过了适当的选择。

This algorithm ensures that the size of each run is at least $L\left( i\right)$ before the $i$ th query. A suitable choice for $L\left( i\right)$ is $\Theta \left( {i\log i}\right)$ . Since the length of a run must be a power of 2 we will choose

该算法确保在第 $i$ 次查询之前，每个游程（run）的大小至少为 $L\left( i\right)$。$L\left( i\right)$ 的一个合适选择是 $\Theta \left( {i\log i}\right)$。由于游程的长度必须是 2 的幂，我们将选择

$$
L\left( i\right)  = {2}^{\left\lceil  \log \left( i\log i\right) \right\rceil  }.
$$

The processing cost of going from a stage with runs of length 1 to runs of length $L\left( i\right)$ is $O\left( {n\log L\left( i\right) }\right)$ . Thus the total cost of processing in answering $r$ queries is $O\left( {n\log r}\right)$ . The search cost for the $i$ th query is upper bounded by $n \cdot  \lceil \log \left( {L\left( i\right)  + 1}\right) \rceil /L\left( i\right)$ . Summing over the first $r$ queries,the search cost is bounded by

从游程长度为 1 的阶段过渡到游程长度为 $L\left( i\right)$ 的阶段的处理成本为 $O\left( {n\log L\left( i\right) }\right)$。因此，回答 $r$ 次查询的总处理成本为 $O\left( {n\log r}\right)$。第 $i$ 次查询的搜索成本上限为 $n \cdot  \lceil \log \left( {L\left( i\right)  + 1}\right) \rceil /L\left( i\right)$。对前 $r$ 次查询求和，搜索成本的上限为

$$
\mathop{\sum }\limits_{{i = 1}}^{r}\frac{n}{L\left( i\right) } \cdot  \lceil \log \left( {L\left( i\right)  + 1}\right) \rceil  = O\left( {n\log r}\right) .
$$

THEOREM 2. For $r \leqq  n$ ,the total cost of answering $r$ queries is $O\left( {n\log r}\right)$ .

定理2。对于$r \leqq  n$，回答$r$个查询的总成本为$O\left( {n\log r}\right)$。

When $r > n$ ,we note that the set $X$ will be completely ordered by our strategy. All queries are then answered in time $O\left( {\log n}\right)$ by binary search.

当 $r > n$ 时，我们注意到集合 $X$ 将按照我们的策略被完全排序。然后，所有查询都可以通过二分查找在时间 $O\left( {\log n}\right)$ 内得到解答。

Proof. The processing cost and the search cost are each $O\left( {n\log r}\right)$ ,so that the total cost of answering the first $r$ queries is $O\left( {n\log r}\right)$ .

证明。处理成本和搜索成本均为 $O\left( {n\log r}\right)$，因此回答前 $r$ 个查询的总成本为 $O\left( {n\log r}\right)$。

2.2.2. An approach based on recursive median finding. We now describe an alternative approach based on median finding; a specification of the algorithm in "pseudo-pascal" follows. The algorithm builds the binary search tree ${T}_{X}$ in a query-driven fashion. Each internal node $v$ of ${T}_{X}$ is viewed as representing a subset $X\left( v\right)$ of $X$ -the root represents $X$ ,its left and right children represent the smallest $\left( {n - 1}\right) /2$ and the biggest $\left( {n - 1}\right) /2$ elements of $X$ ,respectively,and so on. Let $\operatorname{LSon}\left( v\right)$ and $\operatorname{RSon}\left( v\right)$ represent the left and right children of $v$ ,respectively. We can now think of building ${T}_{X}$ as follows. For each internal node $v$ ,expansion consists of partitioning $X\left( v\right)$ into two subsets of equal size-elements smaller than the median of $X\left( v\right)$ ,which will constitute $X\left( {\text{Lson}\left( v\right) }\right)$ ,and elements larger than the median,which will make up $X\left( {\text{Rson}\left( v\right) }\right)$ . We label $v$ by the median of $X\left( v\right)$ . Thus a node at level $i$ represents at most $n/{2}^{i}$ elements of $X.{}^{1}$ Subsequently,LSon(v)and $\operatorname{RSon}\left( v\right)$ may be expanded. Since the median of $X\left( v\right)$ can be found in $3\left| {X\left( v\right) }\right|$ comparisons [12],the expansion of node $v$ takes $3\left| {X\left( v\right) }\right|$ comparisons. If we begin by expanding the root of ${T}_{X}$ (which represents the entire set $X$ ),and then expand every node created, ${T}_{X}$ can be built up in ${3n}\log n$ comparisons.

2.2.2. 基于递归寻找中位数的方法。我们现在描述一种基于寻找中位数的替代方法；以下是该算法的“伪Pascal”规范。该算法以查询驱动的方式构建二叉搜索树${T}_{X}$。${T}_{X}$的每个内部节点$v$被视为代表$X$的一个子集$X\left( v\right)$ —— 根节点代表$X$，其左子节点和右子节点分别代表$X$中最小的$\left( {n - 1}\right) /2$个元素和最大的$\left( {n - 1}\right) /2$个元素，依此类推。设$\operatorname{LSon}\left( v\right)$和$\operatorname{RSon}\left( v\right)$分别代表$v$的左子节点和右子节点。我们现在可以按如下方式考虑构建${T}_{X}$。对于每个内部节点$v$，扩展操作包括将$X\left( v\right)$划分为两个大小相等的子集 —— 小于$X\left( v\right)$中位数的元素将构成$X\left( {\text{Lson}\left( v\right) }\right)$，大于中位数的元素将构成$X\left( {\text{Rson}\left( v\right) }\right)$。我们用$X\left( v\right)$的中位数标记$v$。因此，第$i$层的节点最多代表$X.{}^{1}$中的$n/{2}^{i}$个元素。随后，可以对$v$的左子节点（LSon(v)）和$\operatorname{RSon}\left( v\right)$进行扩展。由于可以通过$3\left| {X\left( v\right) }\right|$次比较找到$X\left( v\right)$的中位数[12]，因此节点$v$的扩展需要$3\left| {X\left( v\right) }\right|$次比较。如果我们从扩展${T}_{X}$的根节点（它代表整个集合$X$）开始，然后扩展创建的每个节点，那么可以通过${3n}\log n$次比较构建出${T}_{X}$。

The search for a query can be thought of as tracing a root-to-leaf path in ${T}_{X}$ . The key observation is that for any given query ${q}_{j}$ ,we need only expand those nodes visited by the search for ${q}_{j}$ ; this is the query-driven tree construction referred to earlier. After each expansion, at most one of the resulting offspring will be visited. The first query ${q}_{1}$ is answered in $O\left( {n + n/2 + \cdots }\right)  = O\left( n\right)$ operations while building up one root-to-leaf path of ${T}_{X}$ . The time taken to answer ${q}_{1}$ is thus within a constant factor of the time for a linear search. In the process of answering ${q}_{1}$ ,we have developed some structure that will be useful in answering subsequent queries; any future search that visits a node that is already expanded will only cost us a single comparison to proceed to the next level of the search; there is no further expansion cost at this node. Nodes that remain unexpanded will be expanded when other queries visit them. When $n$ queries that visit all $n$ leaves have been answered, ${T}_{X}$ will have been completely built up. In essence, we are dispensing with an explicit preprocessing phase, i.e., we are doing "preprocessing" operations only when needed. The cost of building the data structure is distributed over several queries.

对一个查询的搜索可以被视为在${T}_{X}$中追踪一条从根节点到叶子节点的路径。关键的观察结果是，对于任何给定的查询${q}_{j}$，我们只需要扩展在搜索${q}_{j}$时访问过的那些节点；这就是前面提到的基于查询的树构建方法。每次扩展后，最多只有一个生成的子节点会被访问。第一个查询${q}_{1}$在$O\left( {n + n/2 + \cdots }\right)  = O\left( n\right)$次操作内得到解答，同时构建了${T}_{X}$的一条从根节点到叶子节点的路径。因此，解答${q}_{1}$所花费的时间与线性搜索的时间相差一个常数因子。在解答${q}_{1}$的过程中，我们构建了一些结构，这些结构在解答后续查询时会很有用；任何未来的搜索如果访问到已经扩展过的节点，只需要进行一次比较就可以进入搜索的下一层；在这个节点上不会有进一步的扩展成本。未扩展的节点会在其他查询访问它们时进行扩展。当所有访问$n$个叶子节点的$n$个查询都得到解答后，${T}_{X}$就会完全构建好。本质上，我们省去了显式的预处理阶段，即我们只在需要时进行“预处理”操作。构建数据结构的成本分摊到了多个查询上。

DETAILED DESCRIPTION OF THE ALGORITHM. With every node in the tree we associate a set of values and a label, both of which may at times be undefined.

算法的详细描述。我们为树中的每个节点关联一组值和一个标签，这两者有时可能未定义。

## Main body

## 主体

Step 1. Initialize the tree, ${T}_{X}$ ,with the $n$ data keys at the root.

步骤1. 用根节点处的 $n$ 数据键初始化树 ${T}_{X}$。

Step 2. Get a query $q$ .

步骤2. 获取一个查询 $q$。

Step 3. Result $\leftarrow$ SEARCH (root, $q$ ).

步骤3. 结果 $\leftarrow$ 搜索（根节点，$q$）。

Step 4. Output the result.

步骤4. 输出结果。

Step 5. Goto Step 2

步骤5. 转到步骤2

procedure SEARCH ( $v$ : node; $q$ :query): boolean;

过程SEARCH ( $v$ : 节点; $q$ : 查询): 布尔型;

Step 1. If ( $v$ is not labeled) then EXPAND ( $v$ ).

步骤1. 如果 ( $v$ 未被标记)，则执行EXPAND ( $v$ )。

Step 2. If $\left( {\operatorname{label}\left( v\right)  = q}\right)$ then return true.

步骤2. 如果 $\left( {\operatorname{label}\left( v\right)  = q}\right)$ 成立，则返回真。

Step 3. If ( $v$ is a leaf node) then return false.

步骤3. 如果 ( $v$ 是叶节点)，则返回假。

---

<!-- Footnote -->

${}^{1}$ Actually it represents slightly fewer elements,since each node picks up one element of $\mathrm{X}$ as its label. This does not matter, as we are deriving an upper bound.

${}^{1}$ 实际上，它表示的元素数量略少，因为每个节点会选取 $\mathrm{X}$ 中的一个元素作为其标签。这无关紧要，因为我们正在推导一个上界。

<!-- Footnote -->

---

Step 4. If $\left( {q < \operatorname{label}\left( v\right) }\right)$ then return $\operatorname{SEARCH}\left( {\text{left_child}\left( v\right) ,q}\right)$ .

步骤4. 如果 $\left( {q < \operatorname{label}\left( v\right) }\right)$ ，则返回 $\operatorname{SEARCH}\left( {\text{left_child}\left( v\right) ,q}\right)$ 。

Step 5. If $\left( {q > \operatorname{label}\left( v\right) }\right)$ then return $\operatorname{SEARCH}\left( {\text{right_child}\left( v\right) ,q}\right)$ .

步骤5. 如果 $\left( {q > \operatorname{label}\left( v\right) }\right)$ ，则返回 $\operatorname{SEARCH}\left( {\text{right_child}\left( v\right) ,q}\right)$ 。

procedure EXPAND ( $v$ : node);

过程 EXPAND ( $v$ : 节点);

Step 1. $S \leftarrow  \operatorname{set}\left( v\right)$ .

步骤1. $S \leftarrow  \operatorname{set}\left( v\right)$ 。

Step 2. $m \leftarrow$ MEDIAN_FIND(S).

步骤2. $m \leftarrow$ 中位数查找(S)。

Step 3. label $\left( v\right)  \leftarrow  m$ .

步骤3. 标记 $\left( v\right)  \leftarrow  m$ 。

Step 4. if $\left( {\left| S\right|  = 1}\right)$ then return.

步骤4. 如果 $\left( {\left| S\right|  = 1}\right)$ 成立，则返回。

Step 5. ${S}_{l} \leftarrow  \left\lbrack  {x \mid  x \in  S\text{and}x < m}\right\rbrack$ .

步骤5. ${S}_{l} \leftarrow  \left\lbrack  {x \mid  x \in  S\text{and}x < m}\right\rbrack$ 。

Step 6. ${S}_{r} \leftarrow  \left\lbrack  {x \mid  x \in  S\text{and}x > m}\right\rbrack$ .

步骤6. ${S}_{r} \leftarrow  \left\lbrack  {x \mid  x \in  S\text{and}x > m}\right\rbrack$ 。

Step 7. set $\left( {\operatorname{leftchild}\left( v\right) }\right)  \leftarrow  {S}_{l}$ .

步骤7. 设置 $\left( {\operatorname{leftchild}\left( v\right) }\right)  \leftarrow  {S}_{l}$ 。

Step 8. set $\left( {\text{rightchild}\left( v\right) }\right)  \leftarrow  {S}_{r}$ .

步骤8. 设置 $\left( {\text{rightchild}\left( v\right) }\right)  \leftarrow  {S}_{r}$ 。

It should be noted that the two subsets, ${S}_{l}$ and ${S}_{r}$ ,are computed by the procedure MEDIAN_FIND as part of the process of finding the median. There is no extra work associated with determining these two sets once the median has been found.

应当注意的是，两个子集 ${S}_{l}$ 和 ${S}_{r}$ 是在寻找中位数的过程中由中位数查找（MEDIAN_FIND）程序计算得出的。一旦找到中位数，确定这两个集合无需额外的工作。

In order to analyze our algorithm,let us define a function on $n$ and $r$ as follows:

为了分析我们的算法，让我们对 $n$ 和 $r$ 定义一个函数如下：

$$
\Lambda \left( {n,r}\right)  = \left\{  \begin{array}{ll} {3n}\log r + r\log n, & r \leqq  n, \\  \left( {{3n} + r}\right)  \cdot  \log n, & r > n. \end{array}\right. 
$$

Note that $\Lambda \left( {n,r}\right)  = \Theta \left( {\left( {n + r}\right)  \cdot  \log \min \left( {n,r}\right) }\right)$ since $r \cdot  \log n \leqq  n \cdot  \log r$ for $r \leqq  n$ .

注意，由于对于 $r \leqq  n$ 有 $r \cdot  \log n \leqq  n \cdot  \log r$，所以 $\Lambda \left( {n,r}\right)  = \Theta \left( {\left( {n + r}\right)  \cdot  \log \min \left( {n,r}\right) }\right)$。

THEOREM 3. The number of operations needed for processing $r$ queries is no more than $\Lambda \left( {n,r}\right)$ .

定理3. 处理$r$个查询所需的操作次数不超过$\Lambda \left( {n,r}\right)$。

Proof. Consider the case $r \leqq  n$ . No more than $r$ nodes will be expanded at any level of ${T}_{X}$ ,after $r$ queries. For nodes in the top log $r$ levels,the total cost is thus less than ${3n}\log r$ . This is because all nodes may be expanded at each of the first $\log \mathrm{r}$ levels. The expansion of a node $v$ entails finding the median of $X\left( v\right)$ and this requires at least $3\left| {X\left( v\right) }\right|$ comparisons in the worst case [12]. For $i > \lceil \log r\rceil$ the node-expansion cost at level $i$ is $O\left( {r \cdot  n/{2}^{i}}\right)$ . This is because the cost of expanding a node at level $i$ is at most $3 \cdot  n/2$ . Summing over all but the first $\lceil \log r\rceil$ levels,the cost of node expansion at these levels is $O\left( n\right)$ . In addition to the expansion cost,we have to consider the cost associated with search; this is at most $\log n$ comparisons per query. The search component of the cost is thus always less than $r\log n$ .

证明。考虑情况 $r \leqq  n$。在进行 $r$ 次查询后，${T}_{X}$ 的任何层级扩展的节点数都不会超过 $r$ 个。对于前 log $r$ 层的节点，总成本小于 ${3n}\log r$。这是因为在前 $\log \mathrm{r}$ 层的每一层，所有节点都可能被扩展。扩展一个节点 $v$ 需要找到 $X\left( v\right)$ 的中位数，在最坏情况下，这至少需要 $3\left| {X\left( v\right) }\right|$ 次比较 [12]。对于 $i > \lceil \log r\rceil$，第 $i$ 层的节点扩展成本为 $O\left( {r \cdot  n/{2}^{i}}\right)$。这是因为第 $i$ 层扩展一个节点的成本至多为 $3 \cdot  n/2$。对除前 $\lceil \log r\rceil$ 层之外的所有层级求和，这些层级的节点扩展成本为 $O\left( n\right)$。除了扩展成本，我们还需要考虑与搜索相关的成本；每次查询的比较次数至多为 $\log n$。因此，成本中的搜索部分始终小于 $r\log n$。

When $r$ exceeds $n$ ,the expansion cost can never exceed the cost of constructing ${T}_{X}$ completely; this cost is ${3n}\log n$ . Again,note that the factor of 3 comes from the median-finding procedure.

当 $r$ 超过 $n$ 时，扩展成本永远不会超过完全构建 ${T}_{X}$ 的成本；该成本为 ${3n}\log n$。再次强调，3 这个系数来自于中位数查找过程。

2.3. A general paradigm for deferred data structuring. We are now ready to state the general paradigm for deferred data structuring. This paradigm will isolate some features essential for a search problem to be amenable to this approach, and will simplify our description of the geometric search problems considered in $§§4$ and 5 . It also enables us to identify some problems where this approach is not likely to work.

2.3. 延迟数据结构的通用范式。我们现在准备阐述延迟数据结构的通用范式。该范式将分离出搜索问题适用此方法的一些关键特征，并将简化我们对 $§§4$ 和第 5 节中所考虑的几何搜索问题的描述。它还使我们能够识别出一些此方法可能不适用的问题。

Let $\Pi$ be a search problem with the following properties. (1) The search is on a set $S$ of $n$ data points (in the above example, $S = X$ ). (2) A query $q$ can be answered in $O\left( n\right)$ time. (3) In time $O\left( n\right)$ ,we can partition $S$ into two equal-sized subsets ${S}_{1}$ and ${S}_{2}$ such that (i) the answer to query $q$ on set $S$ is equal to the answer to $q$ on either ${S}_{1}$ or ${\mathrm{S}}_{2}$ ; (ii) in the course of partitioning $S$ we can compute a function on $S$ , $f\left( S\right)$ ,such that there is a constant time procedure,TEST $\left( {f\left( S\right) ,q}\right)$ ,which will determine whether the answer to $q$ on $S$ is to be found in ${S}_{1}$ or ${S}_{2}$ . (In the above example $f\left( S\right)  =$ MEDIAN(S)and TEST is a simple comparison operation.)

设$\Pi$为具有以下性质的搜索问题。(1) 搜索是在一个包含$n$个数据点的集合$S$上进行（在上述示例中，为$S = X$）。(2) 一个查询$q$可以在$O\left( n\right)$时间内得到解答。(3) 在$O\left( n\right)$时间内，我们可以将$S$划分为两个大小相等的子集${S}_{1}$和${S}_{2}$，使得：(i) 在集合$S$上对查询$q$的解答等于在${S}_{1}$或${\mathrm{S}}_{2}$上对$q$的解答；(ii) 在划分$S$的过程中，我们可以计算一个关于$S$的函数$f\left( S\right)$，使得存在一个常数时间的过程TEST $\left( {f\left( S\right) ,q}\right)$，该过程能够确定在$S$上对$q$的解答是在${S}_{1}$中还是在${S}_{2}$中。（在上述示例中，$f\left( S\right)  =$为MEDIAN(S)，且TEST是一个简单的比较操作。）

Under these conditions, we can adopt the deferred data structuring approach that builds the search tree gradually. We illustrate this paradigm by several geometric examples in $§§4$ and 5 .

在这些条件下，我们可以采用延迟数据结构化方法，该方法可逐步构建搜索树。我们在$§§4$和5中通过几个几何示例来说明这一范式。

3. A randomized algorithm. In the last section we saw a deterministic algorithm to answer $r$ queries in $O\left( {\left( {n + r}\right)  \cdot  \log \min \{ n,r\} }\right)$ time using deferred data structures. The upper bound of Theorem 3 exceeds the information-theoretic lower bound by a factor of 3 if we use the median algorithm given in [12]. Finding the median of $n$ elements takes ${3n}$ comparisons and this is what leads to the gap between the upper and lower bounds. A careful implementation would reduce the constant factor to 2.5 by passing down certain partial orders generated in the median-finding algorithm from parent to children nodes. More easily implemented algorithms given in [3] would yield even higher constant factors. There is an algorithm due to Floyd [7] which computes the median in ${3n}/2$ expected time. Its use would reduce our constant to $3/2$ . Here we present a randomized algorithm in which the number of comparisons will be optimal (with high probability).

3. 一种随机算法。在上一节中，我们看到了一种确定性算法，它使用延迟数据结构在$O\left( {\left( {n + r}\right)  \cdot  \log \min \{ n,r\} }\right)$时间内回答$r$查询。如果我们使用文献[12]中给出的中位数算法，定理3的上界比信息论下界高出3倍。找出$n$个元素的中位数需要${3n}$次比较，这就是导致上界和下界之间存在差距的原因。通过将中位数查找算法中生成的某些偏序从父节点传递到子节点，精心实现的算法可以将常数因子降低到2.5。文献[3]中给出的更易于实现的算法会产生更高的常数因子。有一种由弗洛伊德（Floyd）[7]提出的算法，它能在${3n}/2$的期望时间内计算中位数。使用该算法会将我们的常数降低到$3/2$。这里我们介绍一种随机算法，其中比较次数将是最优的（具有高概率）。

The randomized algorithm differs from the one in $§2$ in just one respect. The median of the set of values stored at a node was used earlier to get a partition for the purposes of node expansion. Here we will use a mediocre element for the same purpose. The mediocre element will be chosen to be quite close to the median. More precisely, the rank of a mediocre element from a set of size $t$ will lie in the range $t/2 \pm  {t}^{2/3}$ . We will use randomized techniques to compute a mediocre element efficiently. First, a random subset of size $O\left( {t}^{5/6}\right)$ is chosen from the $t$ elements. The median of this random subset is a good candidate for being a mediocre element. It takes $t + O\left( {t}^{5/6}\right)$ comparisons to pick a random sample and test its median element for "mediocrity" (see Step 5 below). This sampling is repeated until a mediocre element is found. The call to the procedure MEDIAN_FIND, in the algorithm outlined in § 2, should be replaced by a call to the procedure MEDIOCRE_FIND outlined below.

随机算法与文献$§2$中的算法仅在一个方面有所不同。早期，为了进行节点扩展，使用存储在节点处的一组值的中位数来进行划分。在这里，我们将使用一个近似中位数的元素（平庸元素）来达到相同的目的。近似中位数的元素将被选择为非常接近中位数。更准确地说，从大小为$t$的集合中选取的近似中位数元素的排名将位于区间$t/2 \pm  {t}^{2/3}$内。我们将使用随机化技术来高效地计算近似中位数元素。首先，从$t$个元素中随机选择一个大小为$O\left( {t}^{5/6}\right)$的子集。这个随机子集的中位数是成为近似中位数元素的一个不错候选。选择一个随机样本并测试其中位数元素是否为“近似中位数”需要进行$t + O\left( {t}^{5/6}\right)$次比较（见下面的步骤5）。重复这个抽样过程，直到找到一个近似中位数元素。在§ 2中概述的算法里，对过程MEDIAN_FIND的调用应该替换为对下面概述的过程MEDIOCRE_FIND的调用。

procedure MEDIOCRE_FIND (T: set of values): value;

过程MEDIOCRE_FIND（T：值的集合）：值;

Step 1 Let $t = \left| T\right|$ .

步骤1 设 $t = \left| T\right|$ 。

Step 2 Pick a random sample $S$ of size $2 \cdot  \left\lceil  {t}^{5/6}\right\rceil   + 1$ from $T$ .

步骤2 从 $T$ 中随机选取一个大小为 $2 \cdot  \left\lceil  {t}^{5/6}\right\rceil   + 1$ 的样本 $S$ 。

Step ${3m} \leftarrow$ MEDIAN_FIND(S).

步骤 ${3m} \leftarrow$ 中位数查找（S）。

Step 4 Compute rank(m)by comparing with each element of $T - S$ .

步骤4 通过与 $T - S$ 中的每个元素进行比较来计算m的排名。

Step 5 If rank(m)is not in the range $\left( {t/2}\right)  \pm  {t}^{2/3}$ then goto Step 2.

步骤5 如果m的排名不在范围 $\left( {t/2}\right)  \pm  {t}^{2/3}$ 内，则转到步骤2。

Step 6 Return $m$ .

步骤 6：返回 $m$。

Note that in Step 4 we need not compare $m$ with elements of $S$ since we assume that the procedure MEDIAN_FIND implicitly gives us the partition of $S$ with respect to $m$ . At the last few levels we will revert to deterministic median finding since the node sizes will be too small to justify randomization. A good choice is to use procedure MEDIOCRE_FIND for the first $\log n - 5$ levels and procedure MEDIAN_FIND thereafter. The randomized algorithm leads to the following theorem.

注意，在步骤 4 中，我们无需将 $m$ 与 $S$ 中的元素进行比较，因为我们假设中位数查找程序（MEDIAN_FIND）隐式地为我们提供了 $S$ 相对于 $m$ 的划分。在最后几层，我们将采用确定性的中位数查找方法，因为节点规模太小，不值得进行随机化操作。一个不错的选择是在前 $\log n - 5$ 层使用普通查找程序（MEDIOCRE_FIND），之后使用中位数查找程序（MEDIAN_FIND）。该随机算法引出了以下定理。

THEOREM 4. Let $T\left( {n,r}\right)$ be the total number of comparisons made by the randomized algorithm in answering $r$ queries on $n$ elements. Then the following holds with probability greater than $1 - \log r/\beta  \cdot  n$ ,

定理 4：设 $T\left( {n,r}\right)$ 为随机算法在对 $n$ 个元素进行 $r$ 次查询时所进行的比较总次数。那么，以下不等式以大于 $1 - \log r/\beta  \cdot  n$ 的概率成立：

$$
T\left( {n,r}\right)  \leqq  \left\{  \begin{array}{ll} \left( {1 + \alpha }\right) \left( {n\log r + r\log n}\right) , & r \leqq  n, \\  \left( {1 + \alpha }\right) \left( {n + r}\right) \log n, & r > n, \end{array}\right. 
$$

where $\beta$ depends on the value of the constant $\alpha$ .

其中，$\beta$ 取决于常数 $\alpha$ 的值。

The remainder of this section is devoted to the proof of this theorem. The proof will be organized into five lemmas.

本节的其余部分将致力于证明该定理。证明将分为五个引理进行。

The use of mediocre elements (instead of the median) may result in uneven splits, causing an imbalance in the binary search tree being created. Nevertheless, the following lemma shows that the higher of the new binary search tree cannot be much worse than $\log n$ . We also show that the number of elements associated with each node at level $i$ is close to $n/{2}^{i}$ . Let the size of a node in the search tree be the number of elements associated with that node.

使用普通元素（而非中位数）可能会导致分割不均匀，从而使正在创建的二叉搜索树出现不平衡。不过，以下引理表明，新二叉搜索树的高度不会比 $\log n$ 差太多。我们还表明，在第 $i$ 层上与每个节点关联的元素数量接近 $n/{2}^{i}$ 。设搜索树中某个节点的大小为与该节点关联的元素数量。

LEMMA 1. Let ${s}_{i}$ be the size of some node at level $i$ . Then,

引理 1。设 ${s}_{i}$ 为第 $i$ 层上某个节点的大小。那么，

$$
{n}_{i}\left( {1 - \frac{20}{{n}_{i}^{1/3}}}\right)  \leqq  {s}_{i} \leqq  {n}_{i}\left( {1 + \frac{20}{{n}_{i}^{1/3}}}\right) ,
$$

provided ${n}_{i} \geqq  {22}$ ,where ${n}_{i} = n/{2}^{i}$ .

假设 ${n}_{i} \geqq  {22}$ ，其中 ${n}_{i} = n/{2}^{i}$ 。

Proof. We will prove one side of the inequality by means of induction on the levels. The inequality is clearly true at the root $\left( {i = 0}\right)$ since ${s}_{0} = n$ . Suppose the inequality holds up to level $i - 1$ ,i.e., ${s}_{i - 1} \leqq  {n}_{i - 1} \cdot  \left( {1 + {20}/{n}_{i - 1}^{1/3}}\right)$ . We now partition the ${s}_{i - 1}$ elements about the mediocre element. Let ${s}_{i}$ denote the larger of the two partition sets. By the definition of the mediocre element we have ${s}_{i} \leqq  {s}_{i - 1}/2 + {s}_{i - 1}^{2/3}$ . Using the fact that ${\left( 1 + x\right) }^{a} \leqq  1 + a \cdot  x,0 \leqq  a \leqq  1$ we get,

证明。我们将通过对层级进行归纳来证明该不等式的一侧。该不等式在根节点 $\left( {i = 0}\right)$ 处显然成立，因为 ${s}_{0} = n$ 。假设该不等式在层级 $i - 1$ 之前都成立，即 ${s}_{i - 1} \leqq  {n}_{i - 1} \cdot  \left( {1 + {20}/{n}_{i - 1}^{1/3}}\right)$ 。现在我们围绕中间元素对 ${s}_{i - 1}$ 个元素进行划分。设 ${s}_{i}$ 表示两个划分子集里较大的那个。根据中间元素的定义，我们有 ${s}_{i} \leqq  {s}_{i - 1}/2 + {s}_{i - 1}^{2/3}$ 。利用 ${\left( 1 + x\right) }^{a} \leqq  1 + a \cdot  x,0 \leqq  a \leqq  1$ 这一事实，我们可得

$$
{s}_{i} \leqq  {n}_{i}\left( {1 + \frac{1}{{n}_{i}^{1/3}}\left( {{11} \cdot  {2}^{2/3} + \frac{40}{3} \cdot  \frac{{2}^{1/3}}{{n}_{i}^{1/3}}}\right) }\right) .
$$

For ${n}_{i} \geqq  {22}$ ,we note that,

对于 ${n}_{i} \geqq  {22}$，我们注意到，

$$
\left( {{11} \cdot  {2}^{2/3} + \frac{40}{3} \cdot  \frac{{2}^{1/3}}{{n}_{i}^{1/3}}}\right)  \leqq  {20}.
$$

This implies the desired result,

这意味着得到了所需的结果，

$$
{s}_{i} \leqq  {n}_{i}\left( {1 + \frac{20}{{n}_{i}^{1/3}}}\right) 
$$

provided ${n}_{i} \geqq  {22}$ .

前提是 ${n}_{i} \geqq  {22}$。

LEMMA 2. The height of the binary search tree in the randomized algorithm will be $\log n + O\left( 1\right)$ .

引理 2. 随机算法中的二叉搜索树的高度将为 $\log n + O\left( 1\right)$。

Proof. At level $k = \log n - 5$ we will no longer be using mediocre elements to expand a node. Instead, we use the median of the set of elements stored at a node to partition those elements. At this point Lemma 1 is still applicable since ${n}_{k} = {32} \geqq  {22}$ and we have,

证明。在第 $k = \log n - 5$ 层，我们将不再使用普通元素来扩展节点。相反，我们使用存储在节点处的元素集合的中位数来划分这些元素。此时，引理 1 仍然适用，因为 ${n}_{k} = {32} \geqq  {22}$，并且我们有，

$$
{s}_{k} \leqq  {n}_{k}\left( {1 + \frac{20}{{n}_{k}^{1/3}}}\right)  \leqq  {2}^{8}.
$$

Thus,the total number of levels is no more than $k + 8$ . The height of the tree is bounded by $\log n + 3$ .

因此，总层数不超过 $k + 8$。树的高度受 $\log n + 3$ 限制。

From Lemma 2 it follows that the cost of searching in the randomized binary search tree will be close to optimal. Let us now consider the cost of constructing the tree, in particular the total cost of node expansions. The following result shows that the median of the random sample is a mediocre element for the entire set with very high probability.

由引理2可知，在随机二叉搜索树中进行搜索的成本将接近最优。现在让我们考虑构建该树的成本，特别是节点扩展的总成本。以下结果表明，随机样本的中位数极有可能是整个集合中的中等元素。

LEMMA 3. Let $p\left( t\right)$ be the probability that a single iteration of the random sampling does not come up with a mediocre element for a set of size t. Then,

引理3. 设$p\left( t\right)$为随机抽样的单次迭代未从规模为t的集合中选出中等元素的概率。那么，

$$
p\left( t\right)  \leqq  2 \cdot  {t}^{1/2} \cdot  \exp \left( {-4 \cdot  {t}^{1/6}}\right)  \leqq  \frac{1}{4t}.
$$

Proof. Let $T$ be a set of size $t$ to which a single iteration of the random sampling process has been applied. First,a random subset $S$ of size $s\left( t\right)  = 2 \cdot  f\left( t\right)  + 1$ is chosen, where $f\left( t\right)  = \left\lceil  {t}^{5/6}\right\rceil$ . The median of $S$ is tested for being a mediocre element of $T$ . In other words,the rank of the median of $S$ should be in the range $t/2 \pm  {t}^{2/3}$ in $T$ . Let $P\left( {x}_{r}\right)$ be the event that the element ${x}_{r}$ (the element of rank $r$ in $T$ ) is the median of $S$ :

证明。设 $T$ 为一个规模为 $t$ 的集合，已对其应用了一次随机抽样过程。首先，选取一个规模为 $s\left( t\right)  = 2 \cdot  f\left( t\right)  + 1$ 的随机子集 $S$，其中 $f\left( t\right)  = \left\lceil  {t}^{5/6}\right\rceil$ 。检验 $S$ 的中位数是否为 $T$ 的中等元素。换句话说，$S$ 的中位数在 $T$ 中的排名应在区间 $t/2 \pm  {t}^{2/3}$ 内。设 $P\left( {x}_{r}\right)$ 为元素 ${x}_{r}$（在 $T$ 中排名为 $r$ 的元素）是 $S$ 的中位数这一事件：

$$
P\left( {x}_{r}\right)  = \left( \begin{matrix} r - 1 \\  f\left( t\right)  \end{matrix}\right)  \cdot  \left( \frac{t - r}{f\left( t\right) }\right) /\left( \begin{matrix} t \\  s\left( t\right)  \end{matrix}\right) ,\;f\left( t\right)  < r \leqq  t - f\left( t\right) .
$$

Let $d\left( t\right)  = {t}^{2/3}$ . We will refer to $f\left( t\right)$ and $d\left( t\right)$ as $f$ and $d$ ,respectively,to simplify the following description. Clearly,

设 $d\left( t\right)  = {t}^{2/3}$ 。为简化后续描述，我们将分别把 $f\left( t\right)$ 和 $d\left( t\right)$ 称为 $f$ 和 $d$ 。显然，

$$
p\left( t\right)  = \mathop{\sum }\limits_{{r = f + 1}}^{{t/2 - d}}P\left( {x}_{r}\right)  + \mathop{\sum }\limits_{{r = t/2 + d}}^{{t - f}}P\left( {x}_{r}\right)  = \frac{2s}{t - {2f}}\mathop{\sum }\limits_{{r = f + 1}}^{{t/2 - d}}\left( \frac{r - f}{r}\right)  \cdot  \left( \begin{matrix} r \\  f \end{matrix}\right)  \cdot  \left( \begin{matrix} t - r \\  f \end{matrix}\right) /\left( \begin{matrix} t \\  2 \cdot  f \end{matrix}\right) .
$$

We make use of Stirling's formula:

我们使用斯特林公式（Stirling's formula）：

$$
n! = {\left( 2\pi n\right) }^{1/2}{\left( \frac{n}{e}\right) }^{n}{e}^{{k}_{n}},\;\frac{1}{{12n} + 1} < {k}_{n} < \frac{1}{12n}
$$

to derive the following inequality upon considerable simplification:

经过大量简化后，推导出以下不等式：

$$
p\left( t\right)  < 2 \cdot  {f}^{1/2} \cdot  \exp \left( \frac{-{4f}{d}^{2}}{{t}^{2}}\right) .
$$

Given the choices for $f\left( t\right)$ and $d\left( t\right)$ the bound on $p\left( t\right)$ follows immediately. The second part of the inequality given below is also easy to verify:

根据 $f\left( t\right)$ 和 $d\left( t\right)$ 的取值，$p\left( t\right)$ 的边界可立即得出。下面不等式的第二部分也很容易验证：

$$
p\left( t\right)  < 2 \cdot  {t}^{1/2} \cdot  \exp \left( {-4{t}^{1/6}}\right)  < \frac{1}{4t}.
$$

Consider now the overall cost of expanding the nodes in the randomized algorithm. First, there is the cost of finding the medians of the small random samples. Lemma 5 will show that the cost of finding the medians of the small random samples is small even when summed over the entire tree. More important is the cost of deciding whether the median for the sample is a mediocre element for the entire set. There is no cost associated with the actual partitioning since the testing for "mediocrity" implicitly determines the precise partition (see Step 5 of the procedure MEDIOCRE_FIND). Consider the $i$ th level in the tree being constructed. Let $m = {2}^{i}$ denote the maximum number of nodes at this level. The sizes of the sets associated with the nodes at this level must lie in the range $\left( {{n}_{i}/2}\right)  \pm  {20} \cdot  {n}^{2/3}$ ,where ${n}_{i} = n/m$ is the average size of these sets. Supppose each application of the random sampling yielded a mediocre element. This would imply that the total cost of testing for mediocrity is $n$ . However,there will be some bad instances where we do not generate a mediocre element. Let the number of such instances be $s$ at the $i$ th level. The next lemma shows that with high probability $s$ is bounded by ${\varepsilon m}$ ,where $\varepsilon$ is an appropriately small constant. Let ${c}_{i}$ denote the cost of testing for mediocrity at level $i$ . When $s \leqq  \varepsilon  \cdot  m$ we have

现在考虑随机算法中扩展节点的总体成本。首先，存在寻找小随机样本中位数的成本。引理5将表明，即使对整棵树求和，寻找小随机样本中位数的成本也很小。更重要的是，要确定样本的中位数是否为整个集合的中等元素，这需要一定成本。实际的划分操作没有相关成本，因为对“中等性”的测试会隐式地确定精确的划分（见MEDIOCRE_FIND过程的步骤5）。考虑正在构建的树的第$i$层。设$m = {2}^{i}$表示该层的最大节点数。与该层节点相关联的集合的大小必须在$\left( {{n}_{i}/2}\right)  \pm  {20} \cdot  {n}^{2/3}$范围内，其中${n}_{i} = n/m$是这些集合的平均大小。假设每次随机抽样都产生一个中等元素。这意味着测试中等性的总成本为$n$。然而，会有一些糟糕的情况，即我们没有生成中等元素。设第$i$层此类情况的数量为$s$。下一个引理表明，$s$以高概率被${\varepsilon m}$界定，其中$\varepsilon$是一个适当小的常数。设${c}_{i}$表示第$i$层测试中等性的成本。当$s \leqq  \varepsilon  \cdot  m$时，我们有

$$
{c}_{i} \leqq  n + n \cdot  \varepsilon \left( {1 + \frac{20}{{n}_{i}^{1/3}}}\right)  = \left( {1 + \alpha }\right)  \cdot  n.
$$

Since ${n}_{i} > 1$ at all levels it is clear the $\alpha  \leqq  {21} \cdot  \varepsilon$ .

由于在各级上，显然有${n}_{i} > 1$ $\alpha  \leqq  {21} \cdot  \varepsilon$。

LEMMA 4. Let $C$ denote the sum of ${c}_{i}$ over all but the last $O\left( {1/\varepsilon }\right)$ levels, $P(C \geqq$ $\left. {\left( {1 + \alpha }\right)  \cdot  n \cdot  \log r}\right)  \leqq  \log r/{k}^{2} \cdot  n$ .

引理4. 设$C$表示除最后$O\left( {1/\varepsilon }\right)$层外所有层上${c}_{i}$的和，即$P(C \geqq$ $\left. {\left( {1 + \alpha }\right)  \cdot  n \cdot  \log r}\right)  \leqq  \log r/{k}^{2} \cdot  n$。

Proof. Let the random variable ${\zeta }_{i}$ denote the number of bad instances in $l =$ $\left( {1 + \varepsilon }\right)  \cdot  m$ iterations of the random sampling at level $i$ . We already have bounds on $p\left( t\right)$ ,the probability of a single iteration on a node of size $t$ being bad. The $l$ iterations at level $i$ do not use equal-sized sets. Therefore let $p$ denote the largest value taken by $p\left( t\right)$ at the nodes of that level. Let $E\left( \zeta \right)$ and $D\left( \zeta \right)$ denote the mean and deviation of some random variable $\zeta$ . The Chebyshev inequality states that $P(\left| {\zeta  - E\left( \zeta \right) }\right|  \geqq$ $\lambda  \cdot  D\left( \zeta \right) ) \leqq  1/{\lambda }^{2}$ . Since $E\left( {\zeta }_{i}\right)  = l \cdot  p$ and $D\left( {\zeta }_{i}\right)  = {\left( l \cdot  p \cdot  \left( 1 - p\right) \right) }^{1/2}$ we have the following:

证明。设随机变量 ${\zeta }_{i}$ 表示在第 $i$ 层的随机抽样的 $l =$ $\left( {1 + \varepsilon }\right)  \cdot  m$ 次迭代中不良实例的数量。我们已经得到了 $p\left( t\right)$ 的边界，即大小为 $t$ 的节点上单次迭代出现不良情况的概率。第 $i$ 层的 $l$ 次迭代不使用等大小的集合。因此，设 $p$ 表示该层节点上 $p\left( t\right)$ 所取的最大值。设 $E\left( \zeta \right)$ 和 $D\left( \zeta \right)$ 分别表示某个随机变量 $\zeta$ 的均值和标准差。切比雪夫不等式表明 $P(\left| {\zeta  - E\left( \zeta \right) }\right|  \geqq$ $\lambda  \cdot  D\left( \zeta \right) ) \leqq  1/{\lambda }^{2}$ 。由于 $E\left( {\zeta }_{i}\right)  = l \cdot  p$ 且 $D\left( {\zeta }_{i}\right)  = {\left( l \cdot  p \cdot  \left( 1 - p\right) \right) }^{1/2}$ ，我们有以下结论：

$$
P\left( {{\zeta }_{i} \geqq  l - m}\right)  \leqq  \frac{1 \cdot  p \cdot  \left( {1 - p}\right) }{m \cdot  {\varepsilon }^{2}}\;\text{ when }p \leqq  \frac{\varepsilon }{2 \cdot  \left( {1 + \varepsilon }\right) }.
$$

Using the bounds on $p\left( t\right)$ and the lower bound on the size of a node at level $i$ we get, $P\left( {s > {\varepsilon m}}\right)  = P\left( {{c}_{i} \geqq  \left( {1 + \alpha }\right)  \cdot  n}\right)  \leqq  k/{\varepsilon }^{2} \cdot  n$ for all but the last $O\left( {\log 1/\varepsilon }\right)$ levels, $k$ is a small constant. Choosing $\beta  = {\varepsilon }^{2}/k$ and summing the probability over the first $\log r$ levels yields the required bound.

利用$p\left( t\right)$的边界以及第$i$层节点大小的下界，我们得到，除了最后$O\left( {\log 1/\varepsilon }\right)$层外，对于所有层，$k$是一个小常数。选择$\beta  = {\varepsilon }^{2}/k$并对前$\log r$层的概率求和，即可得到所需的边界。

LEMMA 5. When $r < n$ ,the total cost of finding the medians of the random samples is $O\left( {{n}^{5/6} \cdot  {r}^{1/6}}\right)$ with probability $1 - \log r/\beta  \cdot  n$ .

引理5. 当$r < n$时，找到随机样本中位数的总成本以概率$1 - \log r/\beta  \cdot  n$为$O\left( {{n}^{5/6} \cdot  {r}^{1/6}}\right)$。

Proof. The cost of finding the median at a node of size $t$ is ${3t}$ . Let the sizes of the two children of this node be $k \cdot  t$ and $\left( {1 - k}\right)  \cdot  t$ ,where $k$ lies between $\frac{1}{2}$ and 1 . The cost of finding the medians for the children will be proportional to $C\left( k\right)  \cdot  {t}^{5/6}$ ,where $C\left( k\right)  = \left( {{k}^{5/6} + {\left( 1 - k\right) }^{5/6}}\right)$ . Clearly, $C\left( k\right)$ is maximized at $k = \frac{1}{2}$ . Define $C = C\left( \frac{1}{2}\right)  = {2}^{1/6}$ . Thus, the cost of finding the medians at a single level increases by at most a factor of $C$ in going from level $i$ to $i + 1$ . We know that the cost of median finding at the first level is $3 \cdot  {n}^{5/6}$ . Hence,the total median-finding cost for the first $\log r$ levels is

证明。在规模为 $t$ 的节点处寻找中位数的成本是 ${3t}$。设该节点的两个子节点的规模分别为 $k \cdot  t$ 和 $\left( {1 - k}\right)  \cdot  t$，其中 $k$ 介于 $\frac{1}{2}$ 和 1 之间。为子节点寻找中位数的成本将与 $C\left( k\right)  \cdot  {t}^{5/6}$ 成正比，其中 $C\left( k\right)  = \left( {{k}^{5/6} + {\left( 1 - k\right) }^{5/6}}\right)$。显然，$C\left( k\right)$ 在 $k = \frac{1}{2}$ 处取得最大值。定义 $C = C\left( \frac{1}{2}\right)  = {2}^{1/6}$。因此，从第 $i$ 层到第 $i + 1$ 层，在单层寻找中位数的成本最多增加 $C$ 倍。我们知道，在第一层寻找中位数的成本是 $3 \cdot  {n}^{5/6}$。因此，前 $\log r$ 层寻找中位数的总成本是

$$
3 \cdot  {n}^{5/6} \cdot  \left( {1 + C + {C}^{2}\cdots {C}^{\log r - 1}}\right) .
$$

This sums to $O\left( {{n}^{5/6} \cdot  {r}^{1/6}}\right)$ since $C \leqq  {2}^{1/6}$ . When $r > n$ the bound on the median-finding cost becomes $O\left( n\right)$ . In our analysis so far we have ignored the repetitions in the median finding for a given node. This will be necessary since not every median of the random sample will be a mediocre element for the entire set. However, the analysis in Lemma 4 also applies to the median-finding cost since it just bounds the number of repetitions of the mediocre finding process at a level.

由于$C \leqq  {2}^{1/6}$，其总和为$O\left( {{n}^{5/6} \cdot  {r}^{1/6}}\right)$。当$r > n$时，寻找中位数的成本上限变为$O\left( n\right)$。到目前为止，在我们的分析中，我们忽略了给定节点在寻找中位数时的重复情况。这是必要的，因为并非随机样本的每个中位数对于整个集合来说都是中等元素。然而，引理4中的分析也适用于寻找中位数的成本，因为它只是对某一层级上寻找中等元素过程的重复次数进行了界定。

Theorem 4 follows immediately from Lemmas 2, 4, and 5.

定理4可直接由引理2、引理4和引理5推出。

## 4. Planar convex hull and linear programming problems.

## 4. 平面凸包与线性规划问题。

4.1. Point membership in a convex hull. In this section we consider the following problem. We are given a set $P = \left\{  {{p}_{1},{p}_{2},\cdots ,{p}_{n}}\right\}$ of $n$ data points in the plane. Data point ${p}_{i}$ is specified by its two coordinates ${p}_{i} = \left( {{p}_{ix},{p}_{iy}}\right)$ . The convex hull of $P$ will be denoted by $\mathrm{{CH}}\left( P\right)$ . We are required to answer a series of queries: "Is the query point ${q}_{j} = \left( {{q}_{jx},{q}_{jy}}\right)$ included in $\mathrm{{CH}}\left( P\right)$ ?"

4.1. 点在凸包中的隶属关系。在本节中，我们考虑以下问题。给定平面上由 $n$ 个数据点组成的集合 $P = \left\{  {{p}_{1},{p}_{2},\cdots ,{p}_{n}}\right\}$。数据点 ${p}_{i}$ 由其两个坐标 ${p}_{i} = \left( {{p}_{ix},{p}_{iy}}\right)$ 确定。$P$ 的凸包将用 $\mathrm{{CH}}\left( P\right)$ 表示。我们需要回答一系列查询：“查询点 ${q}_{j} = \left( {{q}_{jx},{q}_{jy}}\right)$ 是否包含在 $\mathrm{{CH}}\left( P\right)$ 中？”

We first present two solutions based on the preprocessing approach. Neither of these is optimal for all values of $r$ . Let $\operatorname{BCH}\left( P\right)$ denote those points of $P$ which lie on the boundary of $\mathrm{{CH}}\left( P\right)$ . A single query can be answered in $O\left( n\right)$ time as follows. Compute the polar angles from ${q}_{j}$ to all the data points. The query point is included in $\mathrm{{CH}}\left( P\right)$ if and only if the range of angles $\geqq  {180}^{ \circ  }$ . Alternatively,we can answer $r$ queries by first constructing $\mathrm{{CH}}\left( P\right)$ in time $O\left( {n\log h}\right)$ ,where $h$ is the number of points in $\mathrm{{BCH}}\left( P\right) \left\lbrack  6\right\rbrack  ,\left\lbrack  {11}\right\rbrack$ . Now choose a point, $O$ ,in the interior of $\mathrm{{CH}}\left( P\right)$ and divide the plane into $h$ wedges by means of $h$ semi-infinite lines originating at $O$ and going through each of the $h$ vertices of $\mathrm{{CH}}\left( P\right)$ . Each wedge contains exactly one edge from the boundary of $\mathrm{{CH}}\left( P\right)$ . In any wedge,all points on the same side of this edge as $O$ must lie inside $\mathrm{{CH}}\left( P\right)$ . To answer a query we first determine the wedge in which it lies in $O\left( {\log h}\right)$ time by doing a binary search with respect to the angles subtended at $O$ . We can now test the query point with respect to the edge of the $\mathrm{{CH}}\left( P\right)$ which lies in that particular wedge to decide the membership in $\mathrm{{CH}}\left( P\right)$ . This requires a total of $O\left( {\left( {n + r}\right)  \cdot  \log h}\right)$ operations to answer $r$ queries.

我们首先介绍基于预处理方法的两种解决方案。这两种方案都并非对 $r$ 的所有取值都是最优的。设 $\operatorname{BCH}\left( P\right)$ 表示 $P$ 中位于 $\mathrm{{CH}}\left( P\right)$ 边界上的那些点。单个查询可以按如下方式在 $O\left( n\right)$ 时间内得到解答。计算从 ${q}_{j}$ 到所有数据点的极角。当且仅当角度范围为 $\geqq  {180}^{ \circ  }$ 时，查询点包含在 $\mathrm{{CH}}\left( P\right)$ 中。或者，我们可以通过首先在 $O\left( {n\log h}\right)$ 时间内构建 $\mathrm{{CH}}\left( P\right)$ 来解答 $r$ 个查询，其中 $h$ 是 $\mathrm{{BCH}}\left( P\right) \left\lbrack  6\right\rbrack  ,\left\lbrack  {11}\right\rbrack$ 中的点数。现在，在 $\mathrm{{CH}}\left( P\right)$ 的内部选择一个点 $O$，并通过从 $O$ 出发并经过 $\mathrm{{CH}}\left( P\right)$ 的 $h$ 个顶点的 $h$ 条半无限直线将平面划分为 $h$ 个楔形区域。每个楔形区域恰好包含 $\mathrm{{CH}}\left( P\right)$ 边界上的一条边。在任何楔形区域中，与 $O$ 位于这条边同一侧的所有点必定位于 $\mathrm{{CH}}\left( P\right)$ 内部。为了回答一个查询，我们首先通过对在 $O$ 处张成的角度进行二分查找，在 $O\left( {\log h}\right)$ 时间内确定查询点所在的楔形区域。现在，我们可以针对位于该特定楔形区域内的 $\mathrm{{CH}}\left( P\right)$ 的边来测试查询点，以确定其是否属于 $\mathrm{{CH}}\left( P\right)$。回答 $r$ 个查询总共需要 $O\left( {\left( {n + r}\right)  \cdot  \log h}\right)$ 次操作。

Our approach to solving the point membership problem using deferred data structuring is based on the Kirkpatrick-Seidel top-down convex-hull algorithm [6]. The edges on the boundary of $\mathrm{{CH}}\left( P\right)$ consist of an upper chain and a lower chain. Each of these is a sequence of edges going from the leftmost to the rightmost point in $P$ . Consider a vertical line which partitions $P$ into two nonempty subsets. Such a line will intersect with exactly one edge of each chain; these edges will be referred to as the upper tangent and the lower tangent corresponding to the line. The tangents corresponding to a vertical line which partitions $P$ into subsets of equal size (which we call the median line) are called the tangents of $P$ . Kirkpatrick and Seidel show that a tangent can be computed in $O\left( \left| P\right| \right)$ operations.

我们使用延迟数据结构解决点成员问题的方法基于柯克帕特里克 - 赛德尔（Kirkpatrick - Seidel）自顶向下的凸包算法 [6]。$\mathrm{{CH}}\left( P\right)$ 边界上的边由上链和下链组成。每条链都是从 $P$ 中最左边的点到最右边的点的一系列边。考虑一条将 $P$ 划分为两个非空子集的垂直线。这样的线将恰好与每条链的一条边相交；这些边将被称为对应于该线的上切线和下切线。将 $P$ 划分为大小相等的子集的垂直线（我们称之为中线）所对应的切线称为 $P$ 的切线。柯克帕特里克（Kirkpatrick）和赛德尔（Seidel）证明，可以通过 $O\left( \left| P\right| \right)$ 次运算计算出一条切线。

We now describe our deferred data structure. In the following description we only refer to the upper chain and tangents; analogous reasoning applies to the lower chain and tangents. The data structure consists of a binary search tree ${T}_{P}$ in which each internal node $v$ represents a subset $P\left( v\right)$ of $P$ (where $P\left( \text{root}\right)  = P$ ). Associated with $v$ is an $x$ -interval ${R}_{v} = \left\lbrack  {{x}_{L}\left( v\right) ,{x}_{R}\left( v\right) }\right\rbrack  ;P\left( v\right)$ consists of exactly those data points whose $x$ -coordinates lie in ${R}_{v}$ . We expand a node by computing the median line of $P\left( v\right)$ . The members of $P\left( v\right)$ are partitioned into two subsets: points lying to the left of the median line and points lying to its right. These are associated with the two children of $v$ . The tangent for $P\left( v\right)$ can now be computed in $O\left( \left| {P\left( v\right) }\right| \right)$ operations. It is possible that the tangents corresponding to the two vertical lines demarcating ${R}_{v}$ may be adjacent in the chain. In fact, the two tangents may be the same. In these degenerate cases we do not need to compute the tangent of $P\left( v\right)$ . Such degeneracies can be identified from the tangents corresponding to the vertical lines bounding ${R}_{v}$ (these tangents will have been computed by ancestors of $v$ ). If at a node we find that both the upper and the lower tangent are degenerate, we will not expand the node; such a node is a leaf of ${T}_{P}$ . Since at least one new tangent is discovered each time we expand a node,the number of internal nodes of ${T}_{P}$ (and hence the number of leaves of ${T}_{P}$ ) will never exceed $h$ .

我们现在描述我们的延迟数据结构。在以下描述中，我们仅提及上链和切线；类似的推理适用于下链和切线。该数据结构由一棵二叉搜索树 ${T}_{P}$ 组成，其中每个内部节点 $v$ 表示 $P$ 的一个子集 $P\left( v\right)$（其中 $P\left( \text{root}\right)  = P$ ）。与 $v$ 关联的是一个 $x$ -区间 ${R}_{v} = \left\lbrack  {{x}_{L}\left( v\right) ,{x}_{R}\left( v\right) }\right\rbrack  ;P\left( v\right)$ ，它恰好由那些 $x$ -坐标位于 ${R}_{v}$ 内的数据点组成。我们通过计算 $P\left( v\right)$ 的中线来扩展一个节点。$P\left( v\right)$ 的成员被划分为两个子集：位于中线左侧的点和位于中线右侧的点。这些子集与 $v$ 的两个子节点相关联。现在可以通过 $O\left( \left| {P\left( v\right) }\right| \right)$ 次操作计算 $P\left( v\right)$ 的切线。有可能划分 ${R}_{v}$ 的两条垂直线对应的切线在链中是相邻的。实际上，这两条切线可能是同一条。在这些退化情况下，我们不需要计算 $P\left( v\right)$ 的切线。可以从界定 ${R}_{v}$ 的垂直线对应的切线（这些切线将由 $v$ 的祖先节点计算得出）中识别出此类退化情况。如果在某个节点处我们发现上切线和下切线都是退化的，我们将不扩展该节点；这样的节点是 ${T}_{P}$ 的叶子节点。由于每次扩展一个节点时至少会发现一条新的切线，${T}_{P}$ 的内部节点数量（因此 ${T}_{P}$ 的叶子节点数量）永远不会超过 $h$ 。

The search for a query traverses a root-to-leaf path in the search tree. A node is expanded when it is first visited. At any node $v$ the search progresses to its left or right child depending on the $x$ -coordinate of the query point. In addition,we test whether the query point lies below the upper tangent (extended to infinity in both directions) of $P\left( v\right)$ . If this test fails at any node along the search path we know that the query point lies outside $\mathrm{{CH}}\left( P\right)$ . Similar tests apply to the lower chain/tangent.

对一个查询的搜索会遍历搜索树中从根节点到叶节点的路径。一个节点在首次被访问时会被展开。在任意节点 $v$ 处，搜索会根据查询点的 $x$ 坐标向其左子节点或右子节点推进。此外，我们会测试查询点是否位于 $P\left( v\right)$ 的上切线（向两个方向延伸至无穷远）下方。如果在搜索路径上的任意节点处该测试不通过，我们就知道查询点位于 $\mathrm{{CH}}\left( P\right)$ 之外。类似的测试也适用于下链/切线。

Figure 1 shows an example in which two queries ${q1}$ and ${q2}$ have resulted in the expansion of the root and its two children. The query ${q1}$ lay to the left of the median line of $P$ ,and above the lower tangent of $P$ (extended to the left by dotted lines). This caused LSon (root) to be expanded; at this point we find that ${q1}$ lies below the lower tangent of the left child and is thus outside $\mathrm{{CH}}\left( P\right)$ . Note that the lower tangents of root and LSon (root) meet at a point of $P$ ; this means that we will never again compute a lower tangent in the right-subtree of LSon (root). Similarly, q2 expands the right child of the root node; it is found to lie between the upper and lower tangents of RSon (root),and is thus in $\mathrm{{CH}}\left( P\right)$ .

图1展示了一个示例，其中两个查询 ${q1}$ 和 ${q2}$ 导致根节点及其两个子节点的扩展。查询 ${q1}$ 位于 $P$ 中线的左侧，且在 $P$ 的下切线（用虚线向左延伸）上方。这使得左子节点（根节点）被扩展；此时我们发现 ${q1}$ 位于左子节点下切线的下方，因此在 $\mathrm{{CH}}\left( P\right)$ 之外。注意，根节点和左子节点（根节点）的下切线在 $P$ 的某一点相交；这意味着我们将不再计算左子节点（根节点）右子树中的下切线。类似地，q2扩展了根节点的右子节点；发现它位于右子节点（根节点）的上切线和下切线之间，因此在 $\mathrm{{CH}}\left( P\right)$ 之内。

THEOREM 5. The number of operations for processing $r$ hull-membership queries is $O\left( {\Lambda \left( {n,r}\right) }\right)$ .

定理5. 处理$r$凸包成员查询的操作次数为$O\left( {\Lambda \left( {n,r}\right) }\right)$。

Proof. The depth of ${T}_{P}$ never exceeds $\log n$ . Moreover,a node at level $i$ can be expanded in time $O\left( {n/{2}^{i}}\right)$ . This fits our paradigm. An analysis similar to the proof of Theorem 2 establishes the result. [

证明。${T}_{P}$的深度永远不会超过$\log n$。此外，层级为$i$的节点可以在$O\left( {n/{2}^{i}}\right)$时间内展开。这符合我们的范式。与定理2的证明类似的分析可以得出该结果。[

<!-- Media -->

<!-- figureText: — Tangent lines _____ Median lines ⊕ Query points -->

<img src="https://cdn.noedgeai.com/01957d2c-9f57-79ae-931a-23a4109a0bf5_10.jpg?x=191&y=211&w=1178&h=476&r=0"/>

FIG. 1. Membership in a hull; two queries and the resulting development of ${T}_{P}$ .

图1. 凸包中的成员关系；两个查询以及${T}_{P}$的最终展开情况。

<!-- Media -->

4.2. Intersection of half-spaces. We consider the problem of determining whether a query point ${q}_{j} = \left( {{q}_{jx},{q}_{jy}}\right)$ lies in the intersection of $n$ half-planes. Let $H =$ $\left\{  {{h}_{1},{h}_{2},\cdots {h}_{n}}\right\}$ denote the set of lines which bound the half-planes. We assume that each half-plane contains the origin. If not, we can apply a suitable linear transformation in $O\left( n\right)$ time to bring the origin into the common intersection (provided the intersection of the ${h}_{i}$ is nonempty). This can be done by finding a point in the interior of the intersection [8] and mapping the origin onto this feasible point. We can also test in linear time whether the intersection is empty [8]. Let ${H}_{i}$ denote the half-plane (containing the origin) which is bounded by the line ${h}_{i}$ . We assume in this section that the intersection of the ${H}_{i}$ is bounded-in $\$ {4.3}$ we will show that the case of an unbounded intersection region is easily handled.

4.2. 半空间的交集。我们考虑确定一个查询点 ${q}_{j} = \left( {{q}_{jx},{q}_{jy}}\right)$ 是否位于 $n$ 个半平面的交集内的问题。设 $H =$ $\left\{  {{h}_{1},{h}_{2},\cdots {h}_{n}}\right\}$ 表示界定这些半平面的直线集合。我们假设每个半平面都包含原点。如果不包含，我们可以在 $O\left( n\right)$ 时间内应用一个合适的线性变换，将原点移到公共交集内（前提是 ${h}_{i}$ 的交集非空）。这可以通过找到交集内部的一个点 [8] 并将原点映射到这个可行点来实现。我们还可以在线性时间内测试交集是否为空 [8]。设 ${H}_{i}$ 表示由直线 ${h}_{i}$ 界定的（包含原点的）半平面。在本节中，我们假设 ${H}_{i}$ 的交集是有界的——在 $\$ {4.3}$ 中我们将证明，无界交集区域的情况很容易处理。

The notion of geometric duality (or polarity) [4], [11] will prove extremely useful in the solution of the next two problems. In the plane this reduces to a transformation between points and lines. The dual of a point $p = \left( {a,b}\right)$ is the line ${l}_{p}$ whose equation is ${ax} + {by} + 1 = 0$ ,and vice versa. A more intuitive definition is illustrated in Fig. 2. The line ${l}_{p}$ is perpendicular to the line joining the origin to the point $p$ . If the distance between $p$ and the origin is $d$ ,then the dual line ${l}_{p}$ lies at a distance $1/d$ from the origin in the opposite direction.

几何对偶性（或极性）的概念 [4]、[11] 将在解决接下来的两个问题中被证明极为有用。在平面中，这可简化为点和直线之间的一种变换。点 $p = \left( {a,b}\right)$ 的对偶是直线 ${l}_{p}$，其方程为 ${ax} + {by} + 1 = 0$，反之亦然。图 2 展示了一个更直观的定义。直线 ${l}_{p}$ 垂直于连接原点和点 $p$ 的直线。如果 $p$ 与原点之间的距离为 $d$，那么对偶直线 ${l}_{p}$ 位于与原点距离为 $1/d$ 的相反方向上。

<!-- Media -->

<!-- figureText: X ${l}_{p} : \mathrm{{ax}} + \mathrm{{by}} + 1 = 0$ -->

<img src="https://cdn.noedgeai.com/01957d2c-9f57-79ae-931a-23a4109a0bf5_10.jpg?x=428&y=1428&w=707&h=672&r=0"/>

FIG. 2. Duality of points and lines.

图 2. 点和直线的对偶性。

<!-- Media -->

We will now apply the duality transformation to the intersection of the half-planes under consideration. The dual of the line ${h}_{i}$ is a point,which we will denote by ${p}_{i}$ ; we denote by $P$ the set of these points. The dual of the intersection of the ${H}_{i}$ is the set of all points in ${\mathbf{R}}^{2}$ not in $\mathrm{{CH}}\left( P\right)$ . The dual of ${q}_{j}$ is a line ${L}_{j}$ . The query point ${q}_{j}$ is in the intersection of the ${H}_{i}$ if and only if ${L}_{j}$ does not intersect $\mathrm{{CH}}\left( P\right)$ . Thus our problem reduces to determining whether each of a series of query lines intersects the convex hull of a set of points.

现在，我们将对偶变换应用于所考虑的半平面的交集。直线 ${h}_{i}$ 的对偶是一个点，我们将其记为 ${p}_{i}$；我们用 $P$ 表示这些点的集合。${H}_{i}$ 的交集的对偶是 ${\mathbf{R}}^{2}$ 中所有不在 $\mathrm{{CH}}\left( P\right)$ 中的点的集合。${q}_{j}$ 的对偶是一条直线 ${L}_{j}$。查询点 ${q}_{j}$ 在 ${H}_{i}$ 的交集中，当且仅当 ${L}_{j}$ 不与 $\mathrm{{CH}}\left( P\right)$ 相交。因此，我们的问题归结为确定一系列查询直线中的每一条是否与一组点的凸包相交。

The search tree and the node-expansion process are exactly the same as in § 4.1. At each node $v$ ,we compute the intersection of ${L}_{j}$ with the median line of $P\left( v\right)$ . We know that ${L}_{j}$ must intersect $\mathrm{{CH}}\left( P\right)$ if one of the following holds: (1) the intersection point lies between the upper and lower tangents of $P\left( v\right) ;\left( 2\right) {L}_{j}$ intersects one of the tangents of the current node. If not, we must continue the search in the left or right child of $v$ ,depending on the slopes of ${L}_{j}$ and the tangent. These three possibilities are illustrated in Fig. 3 by lines ${L1},{L2}$ ,and ${L3}$ ,respectively. In the case of ${L3}$ ,we see that any intersection of ${L3}$ with $\mathrm{{CH}}\left( P\right)$ must lie to the left of the median line; we therefore continue the search in LSon ( $v$ ).

搜索树和节点扩展过程与§ 4.1中完全相同。在每个节点$v$处，我们计算${L}_{j}$与$P\left( v\right)$的中线的交点。我们知道，如果满足以下条件之一，${L}_{j}$必定与$\mathrm{{CH}}\left( P\right)$相交：(1)交点位于$P\left( v\right) ;\left( 2\right) {L}_{j}$的上下切线之间；(2)${L}_{j}$与当前节点的某条切线相交。如果不满足上述条件，则必须根据${L}_{j}$和切线的斜率，在$v$的左子节点或右子节点中继续搜索。图3分别用直线${L1},{L2}$和${L3}$说明了这三种可能性。对于${L3}$的情况，我们发现${L3}$与$\mathrm{{CH}}\left( P\right)$的任何交点必定位于中线左侧；因此，我们在LSon($v$)中继续搜索。

The following theorem results.

得出以下定理。

THEOREM 6. The number of operations for processing $r$ half-plane intersection queries is $O\left( {\Lambda \left( {n,r}\right) }\right)$ .

定理6：处理$r$个半平面相交查询的操作次数为$O\left( {\Lambda \left( {n,r}\right) }\right)$。

4.3. Two-variable linear programming. Let $L\left( f\right)$ be a two-variable linear programming problem with $n$ constraints and the objective function $f$ ,which is to be minimized subject to these constraints. The algorithms of Dyer [5] and Megiddo [8] can find the optimum for a single objective function in time $O\left( n\right)$ . We consider a query version of the linear programming problem. Each query is an objective function ${f}_{i}$ ,and we are asked to solve $L\left( {f}_{i}\right)$ .

4.3. 双变量线性规划。设$L\left( f\right)$是一个具有$n$个约束条件且目标函数为$f$的双变量线性规划问题，该目标函数需在这些约束条件下求最小值。戴尔（Dyer）[5]和梅吉多（Megiddo）[8]提出的算法能够在$O\left( n\right)$时间内为单个目标函数找到最优解。我们考虑线性规划问题的查询版本。每个查询是一个目标函数${f}_{i}$，我们需要求解$L\left( {f}_{i}\right)$。

The preprocessing approach to this problem consists of finding the intersection of the half-planes defined by the constraints. This can be done in $O\left( {n\log n}\right)$ time by divide-and-conquer. The set of half-planes is partitioned into two sets of almost equal sizes. The intersection of half-planes in each subproblem can be found recursively; the two intersections can then be merged in linear time [11]. A binary search for the slope of the objective function then answers each query in $O\left( {\log n}\right)$ time.

针对这个问题的预处理方法包括找出由约束条件定义的半平面的交集。通过分治法可以在$O\left( {n\log n}\right)$时间内完成此操作。半平面集合被划分为两个大小几乎相等的集合。每个子问题中半平面的交集可以递归地找到；然后可以在线性时间内将这两个交集合并[11]。接着，对目标函数的斜率进行二分查找，就能在$O\left( {\log n}\right)$时间内回答每个查询。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957d2c-9f57-79ae-931a-23a4109a0bf5_11.jpg?x=205&y=1523&w=1179&h=582&r=0"/>

FIG. 3. Example for testing line intersection with a hull.

图3. 测试直线与凸包相交的示例。

<!-- Media -->

As before, we resort to the geometric dual to solve the problem. We may again assume without loss of generality that the feasible region ${R}_{L}$ is nonempty and contains the origin. Each of the $n$ constraints defines a half-plane ${H}_{i};{R}_{L}$ is the intersection of these half-planes. Using the notation of $§{4.1}$ ,the dual of ${R}_{L}$ is the exterior of $\mathrm{{CH}}\left( P\right)$ .

和之前一样，我们借助几何对偶来解决这个问题。我们同样可以不失一般性地假设可行域 ${R}_{L}$ 非空且包含原点。$n$ 个约束条件中的每一个都定义了一个半平面 ${H}_{i};{R}_{L}$ 是这些半平面的交集。使用 $§{4.1}$ 的符号表示，${R}_{L}$ 的对偶是 $\mathrm{{CH}}\left( P\right)$ 的外部。

To begin with,we will assume that ${R}_{L}$ is bounded. This implies that the origin in the dual plane lies in $\mathrm{{CH}}\left( P\right)$ . The objective function ${f}_{i}$ can be looked upon as a family of parallel lines in the primal. Depending on the slope of ${f}_{i}$ ,we need only consider the set of parallel lines above or below the origin. This set of lines dualizes to a semi-infinite straight line with the origin as one endpoint. We call this the objective line ${g}_{i}$ ,and note that it intersects the boundary of $\mathrm{{CH}}\left( P\right)$ at one point which corresponds to the optimum solution.

首先，我们假设 ${R}_{L}$ 是有界的。这意味着对偶平面中的原点位于 $\mathrm{{CH}}\left( P\right)$ 内。目标函数 ${f}_{i}$ 可以看作是原问题中的一族平行线。根据 ${f}_{i}$ 的斜率，我们只需考虑原点上方或下方的平行线集合。这组直线对偶化为一条以原点为一个端点的半无限直线。我们将其称为目标直线 ${g}_{i}$，并注意到它与 $\mathrm{{CH}}\left( P\right)$ 的边界相交于一点，该点对应于最优解。

The search tree and node expansion are as in § 4.2. While searching at a node $v$ , we compute the intersection,if any,of ${g}_{i}$ with the median line of $P\left( v\right)$ . If there is no intersection or if the point of intersection does not lie between the tangents, the search proceeds to the left (right) child of $v$ if the origin lies to the left (right) of the median line. Otherwise,we proceed in the opposite direction. The search terminates if ${g}_{i}$ intersects a tangent of $P\left( v\right)$ .

搜索树和节点扩展如§ 4.2所述。在节点$v$处进行搜索时，我们计算${g}_{i}$与$P\left( v\right)$的中线的交集（如果存在）。如果没有交集，或者交点不在切线之间，若原点位于中线左侧（右侧），则搜索将继续到$v$的左（右）子节点。否则，我们将朝相反方向继续搜索。如果${g}_{i}$与$P\left( v\right)$的一条切线相交，则搜索终止。

When ${R}_{L}$ is unbounded,the origin in the dual plane does not lie in $\mathrm{{CH}}\left( P\right)$ . If ${g}_{i}$ does not intersect $\mathrm{{CH}}\left( P\right)$ ,the solution to the problem is unbounded. This can be detected by computing in $O\left( n\right)$ time the polar angle from the origin to all points in $P$ ; this is done once,at the beginning. If ${g}_{i}$ lies outside the cone defined by this range of angles,it does not intersect $\mathrm{{CH}}\left( P\right)$ . If ${g}_{i}$ intersects $\mathrm{{CH}}\left( P\right)$ ,we use the same search procedure as in the bounded case. The two points in $\mathrm{{BCH}}\left( P\right)$ which subtend the extreme angles at the origin are joined by a tangent. Intersection with this tangent is ignored for the termination criterion above.

当 ${R}_{L}$ 无界时，对偶平面中的原点不在 $\mathrm{{CH}}\left( P\right)$ 内。如果 ${g}_{i}$ 与 $\mathrm{{CH}}\left( P\right)$ 不相交，则该问题的解是无界的。这可以通过在 $O\left( n\right)$ 时间内计算从原点到 $P$ 中所有点的极角来检测；此操作在开始时进行一次。如果 ${g}_{i}$ 位于由该角度范围定义的圆锥之外，则它与 $\mathrm{{CH}}\left( P\right)$ 不相交。如果 ${g}_{i}$ 与 $\mathrm{{CH}}\left( P\right)$ 相交，我们使用与有界情况相同的搜索过程。$\mathrm{{BCH}}\left( P\right)$ 中在原点处张成极限角的两个点由一条切线连接。对于上述终止准则，忽略与该切线的交点。

Figure 4 shows an unbounded feasible region, and the corresponding convex hull in the dual. Two objective functions ${f}_{1}$ and ${f}_{2}$ and their dual objective lines are shown. The arc in the dual indicates the locus of objective lines (e.g., ${g}_{2}$ ) that do not intersect $\mathrm{{CH}}\left( P\right)$ ,and hence have unbounded optima.

图4展示了一个无界可行域以及对偶空间中对应的凸包。图中显示了两个目标函数${f}_{1}$和${f}_{2}$及其对偶目标线。对偶空间中的弧线表示不与$\mathrm{{CH}}\left( P\right)$相交的目标线（例如${g}_{2}$）的轨迹，因此这些目标线具有无界最优解。

<!-- Media -->

<!-- figureText: Primal Plane Dual Plane -->

<img src="https://cdn.noedgeai.com/01957d2c-9f57-79ae-931a-23a4109a0bf5_12.jpg?x=198&y=1468&w=1216&h=642&r=0"/>

FIG. 4. Unbounded linear-programming search example.

图4. 无界线性规划搜索示例。

<!-- Media -->

THEOREM 7. The number of operations for processing $r$ two-variable linear programming queries is $O\left( {\Lambda \left( {n,r}\right) }\right)$ .

定理7. 处理$r$个双变量线性规划查询的操作次数为$O\left( {\Lambda \left( {n,r}\right) }\right)$。

4.4. Lower bounds under the algebraic tree model. The information-theoretic lower bound of $§2$ is not valid for the geometric problems we have been considering in this section. In $§2$ we were working with the comparison-tree model of computation, whereas we are allowing arithmetic operations here. We therefore use the algebraic tree model of computation [1].

4.4. 代数树模型下的下界。$§2$的信息论下界对于本节所考虑的几何问题并不适用。在$§2$中，我们使用的是比较树计算模型，而这里我们允许进行算术运算。因此，我们采用代数树计算模型[1]。

An algebraic computation tree is an algorithm to decicide whether an input vector, a point in ${\mathbf{R}}^{n}$ ,lies in a point set $W \subseteq  {\mathbf{R}}^{n}$ . The nodes in the tree are of three types: computation nodes, branching nodes, and leaves. A computation node has exactly one child and it can perform one of the usual arithmetic operations or compute a square root. A branching node behaves like a node in a comparison tree, i.e., it can perform comparisons with previously computed values. It has exactly two children corresponding to the possible outcomes of the comparison. A leaf is labeled either "Accept" or "Reject," and it has no children. Each addition operation, subtraction operation or multiplication by a constant costs zero. Every other operation or comparison has a unit cost. The complexity of an algebraic computation tree is the maximum sum of costs along a root-leaf path in the tree. If $W \subseteq  {\mathbf{R}}^{n}$ ,then $C\left( W\right)$ ,the complexity of $W$ , is the minimum complexity of a tree that accepts precisely the set $W$ . For any point set $S \subseteq  {\mathbf{R}}^{n}$ ,let $\# \left( S\right)$ denote the number of connected components of $W$ . It was shown in [1] that $C\left( W\right)  = \Omega \left( {\log \# \left( W\right) }\right)$ .

代数计算树是一种用于判定输入向量（即${\mathbf{R}}^{n}$中的一个点）是否位于点集$W \subseteq  {\mathbf{R}}^{n}$中的算法。树中的节点有三种类型：计算节点、分支节点和叶节点。计算节点恰好有一个子节点，它可以执行常见的算术运算之一或计算平方根。分支节点的行为类似于比较树中的节点，即它可以与之前计算的值进行比较。它恰好有两个子节点，分别对应比较的两种可能结果。叶节点标记为“接受”或“拒绝”，且没有子节点。每次加法运算、减法运算或与常数的乘法运算的成本为零。其他每个运算或比较的成本为一个单位。代数计算树的复杂度是树中从根节点到叶节点路径上成本的最大总和。如果是$W \subseteq  {\mathbf{R}}^{n}$，那么$C\left( W\right)$（即$W$的复杂度）是恰好接受集合$W$的树的最小复杂度。对于任何点集$S \subseteq  {\mathbf{R}}^{n}$，用$\# \left( S\right)$表示$W$的连通分量的数量。文献[1]中证明了$C\left( W\right)  = \Omega \left( {\log \# \left( W\right) }\right)$。

We now show a lower bound of $\Omega \left( {\left( {n + r}\right)  \cdot  \log \min \{ n,r\} }\right)$ algebraic operations for processing $r$ hull-membership queries on $n$ data points. We will in fact show that this bound holds when the $r$ queries are processed off-line. The bound is obtained through a reduction from the SET DISJOINTNESS problem, defined as follows. Given two sets $X = \left\{  {{x}_{1},{x}_{2}\cdots {x}_{n}}\right\}$ and $Q = \left\{  {{q}_{1},{q}_{2}\cdots {q}_{r}}\right\}$ ,determine whether their intersection is nonempty. This problem is a simpler version of the SET INTERSECTION problem mentioned in $§2$ . We first prove a lower bound on SET DISJOINTNESS.

我们现在展示处理关于$n$个数据点的$r$个凸包成员查询（hull - membership queries）所需的代数运算次数的下界为$\Omega \left( {\left( {n + r}\right)  \cdot  \log \min \{ n,r\} }\right)$。实际上，我们将证明当离线处理这$r$个查询时，该下界仍然成立。这个下界是通过从集合不相交问题（SET DISJOINTNESS problem）归约得到的，该问题定义如下。给定两个集合$X = \left\{  {{x}_{1},{x}_{2}\cdots {x}_{n}}\right\}$和$Q = \left\{  {{q}_{1},{q}_{2}\cdots {q}_{r}}\right\}$，判断它们的交集是否为空。这个问题是文献$§2$中提到的集合相交问题（SET INTERSECTION problem）的一个简化版本。我们首先证明集合不相交问题的一个下界。

THEOREM 8. Any algebraic computation tree that solves SET DISJOINTNESS must have a complexity of $\Omega \left( {\left( {n + r}\right)  \cdot  \log \min \{ n,r\} }\right)$ .

定理8. 任何解决集合不相交问题（SET DISJOINTNESS）的代数计算树的复杂度必定为 $\Omega \left( {\left( {n + r}\right)  \cdot  \log \min \{ n,r\} }\right)$。

Proof. Assume without loss of generality that $r \leq  n$ . Every instance of SET DISJOINTNESS can be represented as a point $\beta  = \left( {{x}_{1},\cdots ,{x}_{n},{q}_{1},\cdots ,{q}_{r}}\right)$ in ${\mathbf{R}}^{n + r}$ . Let $W \subseteq  {\mathbf{R}}^{n + r}$ be the set of all points representing disjoint sets. The complexity of the problem is $\Omega \left( {\log  * \left( W\right) }\right)$ ,where $* \left( W\right)$ is the number of connected components of $W$ [1]. Consider instances for which the ${q}_{i}$ are distinct. The elements of $Q$ can be ordered as $\left\{  {{q}_{\left( 1\right) } < {q}_{\left( 2\right) } < \cdots  < {q}_{\left( r\right) }}\right\}$ ,where(i)represents the index of the $i$ th smallest value in $\left\{  {{q}_{1},\cdots ,{q}_{r}}\right\}$ . Let ${S}_{\beta }\left( i\right)  = \left\{  {{x}_{k} : {q}_{\left( i\right) } < {x}_{k} < {q}_{\left( i + 1\right) }}\right\}$ ,for $1 \leqq  i \leqq  r - 1$ . Define ${W}^{ * } =$ $\left\{  {\beta  : \left| {{S}_{\beta }\left( i\right)  = \lfloor n/\left( {r - 1}\right) }\right| ,1 \leqq  i \leqq  r - 1}\right\}  ,{W}^{ * } \subseteq  W$ . The subsets of ${W}^{ * }$ corresponding to different choices of ${S}_{\beta }$ ’s are separated by hyperplanes of the form ${x}_{i} = {q}_{j}$ . These hyperplanes are entirely disjoint from $W$ . This means that if two points in ${W}^{ * }$ are separated by these hyperplanes then they must also be separated in $W$ . Hence,the number of components of $W$ is at least as large as the number of ways of partitioning $\left\{  {{x}_{1},{x}_{2},\cdots ,{x}_{n}}\right\}$ into the ${S}_{\beta }$ ’s as per the definition of ${W}^{ * }$ . A counting argument shows that this is at least as large as

证明。不失一般性，假设 $r \leq  n$ 。集合不相交问题（SET DISJOINTNESS）的每个实例都可以表示为 ${\mathbf{R}}^{n + r}$ 中的一个点 $\beta  = \left( {{x}_{1},\cdots ,{x}_{n},{q}_{1},\cdots ,{q}_{r}}\right)$ 。设 $W \subseteq  {\mathbf{R}}^{n + r}$ 为表示不相交集合的所有点的集合。该问题的复杂度为 $\Omega \left( {\log  * \left( W\right) }\right)$ ，其中 $* \left( W\right)$ 是 $W$ 的连通分量的数量 [1]。考虑 ${q}_{i}$ 互不相同的实例。$Q$ 中的元素可以按 $\left\{  {{q}_{\left( 1\right) } < {q}_{\left( 2\right) } < \cdots  < {q}_{\left( r\right) }}\right\}$ 排序，其中 (i) 表示 $\left\{  {{q}_{1},\cdots ,{q}_{r}}\right\}$ 中第 $i$ 小值的索引。对于 $1 \leqq  i \leqq  r - 1$ ，设 ${S}_{\beta }\left( i\right)  = \left\{  {{x}_{k} : {q}_{\left( i\right) } < {x}_{k} < {q}_{\left( i + 1\right) }}\right\}$ 。定义 ${W}^{ * } =$ $\left\{  {\beta  : \left| {{S}_{\beta }\left( i\right)  = \lfloor n/\left( {r - 1}\right) }\right| ,1 \leqq  i \leqq  r - 1}\right\}  ,{W}^{ * } \subseteq  W$ 。${W}^{ * }$ 中对应于 ${S}_{\beta }$ 不同选择的子集被形如 ${x}_{i} = {q}_{j}$ 的超平面分隔。这些超平面与 $W$ 完全不相交。这意味着如果 ${W}^{ * }$ 中的两个点被这些超平面分隔，那么它们在 $W$ 中也一定被分隔。因此，根据 ${W}^{ * }$ 的定义，$W$ 的分量数量至少与将 $\left\{  {{x}_{1},{x}_{2},\cdots ,{x}_{n}}\right\}$ 划分为 ${S}_{\beta }$ 的方式数量一样多。通过计数论证可知，这个数量至少为

$$
r!\frac{n!}{{\left( \lfloor \left( n/r - 1\right) !{)}^{r - 1}\right) }^{r - 1}}.
$$

From this it follows that the complexity is $\Omega \left( {\left( {n + r}\right)  \cdot  \log r}\right)$ .

由此可知，复杂度为 $\Omega \left( {\left( {n + r}\right)  \cdot  \log r}\right)$。

THEOREM 9. The complexity of processing $r$ hull-membership queries is $\Omega ((n +$ $r) \cdot  \log \min \{ n,r\} )$ .

定理 9. 处理 $r$ 凸包成员查询的复杂度为 $\Omega ((n +$ $r) \cdot  \log \min \{ n,r\} )$。

Proof. By reduction from SET DISJOINTNESS in $O\left( {n + r}\right)$ time. Without loss of generality,assume that the elements of both sets lie in the interval $\lbrack 0,{2\pi })$ . Each element ${x}_{i}$ maps onto a point ${p}_{i}$ on the unit circle with polar coordinates $\left( {1,{x}_{i}}\right)$ . This constitutes our data set $P$ ; note that $\operatorname{BCH}\left( P\right)  = P$ . Each element ${q}_{j}$ of $Q$ maps onto a point ${r}_{j}$ with polar coordinates $\left( {1,{q}_{j}}\right)$ . The point ${r}_{j}$ lies in $\mathrm{{CH}}\left( P\right)$ if and only if ${q}_{j} \in  X$ . Thus SET DISJOINTNESS ${\alpha }_{n + r}$ HULL_MEMBERSHIP. [

证明。通过在$O\left( {n + r}\right)$时间内从集合不相交问题（SET DISJOINTNESS）进行归约。不失一般性，假设两个集合的元素都位于区间$\lbrack 0,{2\pi })$内。每个元素${x}_{i}$映射到单位圆上极坐标为$\left( {1,{x}_{i}}\right)$的点${p}_{i}$。这构成了我们的数据集$P$；注意$\operatorname{BCH}\left( P\right)  = P$。$Q$的每个元素${q}_{j}$映射到极坐标为$\left( {1,{q}_{j}}\right)$的点${r}_{j}$。当且仅当${q}_{j} \in  X$时，点${r}_{j}$位于$\mathrm{{CH}}\left( P\right)$内。因此，集合不相交问题（SET DISJOINTNESS）可归约为凸包成员问题（HULL_MEMBERSHIP）。[

The lower bound extends to the problems in $§§{4.2}$ and 4.3.

该下界适用于$§§{4.2}$和4.3中的问题。

4.5. Effect of the number of points on the convex hull. In this section we return to the problem of determining whether a query point lies within the convex hull of $n$ given data points. We show that a substantial improvement is possible when $h$ ,the number of data points on the boundary of the convex hull,is much smaller than $n$ . It is clear that the guarantees of Theorem 4 are too weak in such a case, since it is possible to find $\mathrm{{CH}}\left( P\right)$ in $O\left( {n\log h}\right)$ operations by the Kirkpatrick-Seidel algorithm; subsequently,queries can be answered in time $O\left( {\log h}\right)$ each. This gives a time bound of $O\left( {\left( {n + r}\right) \log h}\right)$ for answering $r$ queries. This may seem to contradict the lower bound of Theorem 9 but recall that in the lower bound reduction all $n$ data points were on the boundary of the convex hull. When $r$ exceeds $h$ ,the algorithm of $§{4.1}$ achieves a time bound of $O\left( {n\log h + r\log n}\right)$ ,since node expansion costs add up to only $O\left( {n\log h}\right)$ . The cost of searching,however,unfortunately grows as $r\log n$ because the depth of ${T}_{P}$ may grow as $\log n$ even though the number of leaves is only $h$ .

4.5. 点数对凸包的影响。在本节中，我们回到确定查询点是否位于给定的 $n$ 个数据点的凸包内的问题。我们表明，当 $h$（凸包边界上的数据点数量）远小于 $n$ 时，可能会有显著的改进。显然，在这种情况下，定理 4 的保证太弱了，因为通过柯克帕特里克 - 赛德尔算法（Kirkpatrick - Seidel algorithm）可以在 $O\left( {n\log h}\right)$ 次操作中找到 $\mathrm{{CH}}\left( P\right)$；随后，每次查询可以在 $O\left( {\log h}\right)$ 时间内得到答案。这给出了回答 $r$ 次查询的时间界限为 $O\left( {\left( {n + r}\right) \log h}\right)$。这似乎与定理 9 的下界相矛盾，但请记住，在该下界归约中，所有 $n$ 个数据点都在凸包的边界上。当 $r$ 超过 $h$ 时，$§{4.1}$ 的算法实现了 $O\left( {n\log h + r\log n}\right)$ 的时间界限，因为节点扩展成本总计仅为 $O\left( {n\log h}\right)$。然而，不幸的是，搜索成本会随着 $r\log n$ 增长，因为即使叶子节点的数量仅为 $h$，${T}_{P}$ 的深度也可能随着 $\log n$ 增长。

To get around this difficulty we construct, in a dovetailed fashion, two binary search trees ${T}_{P}$ and ${T}_{D}$ . Let $T$ be the fully expanded version of the search tree constructed by the algorithm of $§{4.1}$ . It has $h$ leaves and can be constructed in $O\left( {n\log h}\right)$ time. The two trees ${T}_{P}$ and ${T}_{D}$ will be partially expanded versions of $T$ . ${T}_{P}$ is the version obtained by processing queries according to the algorithm of $§{4.1}$ . The other tree ${T}_{D}$ is obtained by partially constructing $T$ through a deferred depth-first traversal.

为了克服这一困难，我们以交错的方式构建两棵二叉搜索树${T}_{P}$和${T}_{D}$。设$T$为采用$§{4.1}$算法构建的搜索树的完全展开版本。它有$h$个叶子节点，并且可以在$O\left( {n\log h}\right)$时间内构建完成。两棵树${T}_{P}$和${T}_{D}$将是$T$的展开版本。${T}_{P}$是根据$§{4.1}$算法处理查询所得到的版本。另一棵树${T}_{D}$是通过延迟深度优先遍历部分构建$T$而得到的。

The depth-first traversal of a tree with $l$ leaves can be looked upon as consisting of $l$ phases,each of which ends when a new leaf is reached. Similarly,the depth-first construction of ${T}_{D}$ can be broken down into $h$ phases. These $h$ phases are interleaves with the processing of the first $h$ queries on the search tree ${T}_{P}$ . Each phase can also be looked upon as the processing of a judiciously chosen query on the tree ${T}_{D}$ . Thus the cost of the deferred construction of ${T}_{D}$ has the same upper bound as that for ${T}_{P}$ .

具有 $l$ 片叶子的树的深度优先遍历可以看作由 $l$ 个阶段组成，每个阶段在到达一片新叶子时结束。类似地，${T}_{D}$ 的深度优先构造可以分解为 $h$ 个阶段。这 $h$ 个阶段与搜索树 ${T}_{P}$ 上的前 $h$ 个查询的处理相互交织。每个阶段也可以看作是对树 ${T}_{D}$ 上一个经过精心选择的查询的处理。因此，${T}_{D}$ 的延迟构造的成本与 ${T}_{P}$ 的成本具有相同的上界。

When $r$ exceeds $h$ ,the tree ${T}_{D}$ will be fully constructed after the first $h$ queries have been processed on ${T}_{P}$ . At this point ${T}_{P}$ itself may not be fully expanded; in fact only one leaf may have been exposed in it. Since the $\mathrm{{CH}}\left( P\right)$ is now completely determined by ${T}_{D}$ we can do away with the two search trees for further query processing. We now resort to the wedge method to answer each query in time $O\left( {\log h}\right)$ (see $§{4.1}$ ). Since the cost of constructing ${T}_{D}$ is $O\left( {n\log h}\right)$ the following theorem results.

当 $r$ 超过 $h$ 时，在 ${T}_{P}$ 上处理完前 $h$ 个查询后，树 ${T}_{D}$ 将完全构建好。此时，${T}_{P}$ 本身可能并未完全展开；实际上，其中可能仅暴露了一个叶节点。由于 $\mathrm{{CH}}\left( P\right)$ 现在完全由 ${T}_{D}$ 确定，我们可以在后续查询处理中不再使用这两棵搜索树。现在，我们采用楔形法（wedge method）在时间 $O\left( {\log h}\right)$ 内回答每个查询（见 $§{4.1}$）。由于构建 ${T}_{D}$ 的成本为 $O\left( {n\log h}\right)$，因此得出以下定理。

THEOREM 10. The cost of processing $r$ hull-membership queries is $O\left( {{\Lambda }^{\prime }\left( {n,r,h}\right) }\right)$ , where

定理 10. 处理 $r$ 个凸包成员查询的成本为 $O\left( {{\Lambda }^{\prime }\left( {n,r,h}\right) }\right)$，其中

$$
{\Lambda }^{\prime }\left( {n,r,h}\right)  = \left\{  \begin{array}{ll} n\log r, & r \leqq  h, \\  \left( {n + r}\right)  \cdot  \log h, & r > h. \end{array}\right. 
$$

Analogous results hold for the problems in $§§{4.2}$ and 4.3.

类似的结果适用于$§§{4.2}$和4.3中的问题。

5. Domination problems. In this section we investigate a problem related to point domination in $k$ -dimensional space. This problem does not fit directly into the paradigm presented at the end of $§2$ . However,a higher-dimensional analogue of divide-and-conquer enables us to adapt our technique to such problems.

5. 支配问题。在本节中，我们研究一个与$k$维空间中的点支配相关的问题。这个问题并不直接符合$§2$结尾处提出的范式。然而，分治法（divide-and-conquer）的高维类比使我们能够将我们的技术应用于此类问题。

Let ${p}_{i}$ denote the $i$ th coordinate of a point $p$ in $k$ -space. We say that $p$ dominates $q$ if and only if ${p}_{i} \geqq  {q}_{i}$ for all $i,1 \leqq  i \leqq  k$ . Bentley [2] considers the dominance counting problem which is also called the ECDF Searching Problem. In this problem we are given a set $P = \left\{  {{p}_{1},{p}_{2}\cdots {p}_{n}}\right\}$ of $n$ points in $k$ -space. For each query point $q$ ,we are asked to report the number of points of $P$ dominated by $q$ .

用 ${p}_{i}$ 表示 $k$ 维空间中一点 $p$ 的第 $i$ 个坐标。当且仅当对于所有 $i,1 \leqq  i \leqq  k$ 都有 ${p}_{i} \geqq  {q}_{i}$ 时，我们称 $p$ 支配 $q$。本特利（Bentley）[2] 研究了支配计数问题，该问题也被称为经验累积分布函数搜索问题（ECDF Searching Problem）。在这个问题中，我们给定了 $k$ 维空间中的一个包含 $n$ 个点的集合 $P = \left\{  {{p}_{1},{p}_{2}\cdots {p}_{n}}\right\}$。对于每个查询点 $q$，我们需要报告 $P$ 中被 $q$ 支配的点的数量。

Bentley uses a multidimensional divide-and-conquer strategy to solve this problem. He constructs a data structure,the ECDF tree,which answers each query in $O\left( {{\log }^{k}n}\right)$ time following a preprocessing phase requiring $O\left( {n{\log }^{k - 1}n}\right)$ time. This result holds for fixed number of dimensions(k)and for $n$ a power of 2 . However,a more detailed analysis due to Monier [9] shows the validity of this result for arbitrary $n$ and $k$ . In fact,Monier shows that the constant implicit in the $O$ result is $1/\left( {k - 1}\right) !$ . In the following analysis we too will assume that the number of dimensions is fixed and that $n$ is a power of 2 . Our results can be generalized to allow for arbitrary $k$ and $n$ by invoking the results due to Monier.

本特利（Bentley）采用多维分治策略来解决这个问题。他构建了一种数据结构，即经验累积分布函数（ECDF）树，该数据结构在经过一个需要$O\left( {n{\log }^{k - 1}n}\right)$时间的预处理阶段后，能在$O\left( {{\log }^{k}n}\right)$时间内回答每个查询。这一结果适用于固定维度数（k）以及$n$为2的幂的情况。然而，莫尼尔（Monier）[9]进行的更详细分析表明，这一结果对于任意的$n$和$k$都有效。实际上，莫尼尔证明了$O$结果中隐含的常数为$1/\left( {k - 1}\right) !$。在接下来的分析中，我们也将假设维度数是固定的，并且$n$是2的幂。通过引用莫尼尔的研究结果，我们的结果可以推广到允许任意的$k$和$n$的情况。

The basic paradigm of multidimensional divide-and-conquer is as follows: given a problem involving $n$ points in $k$ -space,first divide into (and recursively solve) two subproblems each of $n/2$ points in $k$ -space,and then recursively solve one problem of at most $n$ points in(k - 1)-space. When applied to the dominance counting problem, this paradigm yields the following search or counting strategy:

多维分治法的基本范式如下：给定一个涉及 $n$ 个位于 $k$ 空间中的点的问题，首先将其划分为（并递归求解）两个子问题，每个子问题包含 $n/2$ 个位于 $k$ 空间中的点，然后递归求解一个最多包含 $n$ 个位于 (k - 1) 维空间中的点的问题。当应用于支配计数问题时，该范式产生以下搜索或计数策略：

(1) Find a(k - 1)-dimensional hyperplane $M$ dividing $P$ into two subsets ${P}_{1}$ and ${P}_{2}$ ,each of cardinality $n/2$ . We will assume that $M$ is of the form ${x}_{k} = c$ . Hence,all points in ${P}_{1}$ have their $k$ th coordinate less than $c$ ,while those in ${P}_{2}$ have their $k$ th coordinate greater than $c$ .

(1) 找到一个 (k - 1) 维超平面 $M$，将 $P$ 划分为两个子集 ${P}_{1}$ 和 ${P}_{2}$，每个子集的基数为 $n/2$。我们假设 $M$ 的形式为 ${x}_{k} = c$。因此，${P}_{1}$ 中的所有点的第 $k$ 个坐标小于 $c$，而 ${P}_{2}$ 中的点的第 $k$ 个坐标大于 $c$。

(2) If the query point $q$ lies on the same side of $M$ as ${P}_{1}$ (i.e., ${q}_{k} < c$ ) then recursively search in ${P}_{1}$ only. It is clear that the query point cannot dominate any point in ${P}_{2}$ .

(2) 如果查询点 $q$ 与 ${P}_{1}$ 位于 $M$ 的同一侧（即 ${q}_{k} < c$ ），则仅在 ${P}_{1}$ 中进行递归搜索。显然，查询点不可能支配 ${P}_{2}$ 中的任何点。

(3) Otherwise, $q$ lies on the same side of $M$ as ${P}_{2}$ (i.e., ${q}_{k} > c$ ) and we know that $q$ dominates every point of ${P}_{1}$ in the $k$ th-coordinate. Now we project ${P}_{1}$ and $q$ onto $M$ and recursively search in(k - 1)-space. We also search ${P}_{2}$ in $k$ -space. In Fig. 5 we illustrate this strategy for two-dimensional space.

(3) 否则，$q$ 与 ${P}_{2}$ 位于 $M$ 的同一侧（即 ${q}_{k} > c$ ），并且我们知道 $q$ 在第 $k$ 个坐标上支配 ${P}_{1}$ 中的每个点。现在，我们将 ${P}_{1}$ 和 $q$ 投影到 $M$ 上，并在 (k - 1) 维空间中进行递归搜索。我们还在 $k$ 维空间中搜索 ${P}_{2}$ 。在图 5 中，我们展示了这种在二维空间中的策略。

In one-dimensional space the ECDF searching problem reduces to finding the rank of a query value in the given data-set. The one-dimensional ECDF search tree is an optimal binary search tree on the $n$ points in $P$ . The $k$ -dimensional ECDF tree for the $n$ points in $P$ is a recursively built data structure. The root of this tree contains $M$ ,the median hyperplane for the $k$ th dimension. The left subtree is a $k$ -dimensional ECDF tree for the $n/2$ points in ${P}_{1}$ ,the points in $P$ which lie below $M$ . Similarly,the right subtree is a $k$ -dimensional ECDF tree for the $n/2$ points in ${P}_{2}$ ,the points in $P$ which lie above $M$ . The root also contains a(k - 1)-dimensional ECDF tree representing the points in the ${P}_{1}$ projected onto $M$ .

在一维空间中，经验累积分布函数（ECDF）搜索问题可简化为在给定数据集中查找查询值的排名。一维ECDF搜索树是基于$n$个位于$P$中的点构建的最优二叉搜索树。针对$P$中$n$个点的$k$维ECDF树是一种递归构建的数据结构。该树的根节点包含$M$，即第$k$维的中位数超平面。左子树是基于${P}_{1}$中$n/2$个点构建的$k$维ECDF树，${P}_{1}$中的点是$P$中位于$M$下方的点。类似地，右子树是基于${P}_{2}$中$n/2$个点构建的$k$维ECDF树，${P}_{2}$中的点是$P$中位于$M$上方的点。根节点还包含一个(k - 1)维ECDF树，它表示${P}_{1}$中的点投影到$M$上的结果。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957d2c-9f57-79ae-931a-23a4109a0bf5_15.jpg?x=173&y=1557&w=1227&h=548&r=0"/>

FIG. 5. The two cases for dominance counting in 2-space.

图5. 二维空间中优势计数的两种情况。

<!-- Media -->

To answer a query $q$ ,the search algorithm compares ${q}_{k}$ to $c$ ,the value defining the median plane $M$ stored at the root. If ${q}_{k}$ is less than $c$ then the search is restricted to the points in ${P}_{1}$ only. The algorithm then recursively searches in the left subtree. If,on the other hand, ${q}_{k}$ is greater than $c$ then the algorithm recursively searches in the right subtree as well as the(k - 1)-dimensional ECDF tree stored at the root. For the one-dimensional ECDF tree the algorithm is the standard binary tree search. For fixed $k$ ,the preprocessing time to build the $k$ -dimensional ECDF tree is $p\left( n\right)  =$ $O\left( {n{\log }^{k}n}\right)$ ,and the time required to answer a single query is $q\left( n\right)  = O\left( {{\log }^{k}n}\right)$ .

为了回答查询$q$，搜索算法将${q}_{k}$与存储在根节点处定义中值平面$M$的值$c$进行比较。如果${q}_{k}$小于$c$，则搜索仅限制在${P}_{1}$中的点。然后，该算法在左子树中进行递归搜索。另一方面，如果${q}_{k}$大于$c$，则算法在右子树以及存储在根节点处的(k - 1)维经验累积分布函数（ECDF）树中进行递归搜索。对于一维ECDF树，该算法是标准的二叉树搜索。对于固定的$k$，构建$k$维ECDF树的预处理时间为$p\left( n\right)  =$$O\left( {n{\log }^{k}n}\right)$，回答单个查询所需的时间为$q\left( n\right)  = O\left( {{\log }^{k}n}\right)$。

We now apply the deferred data structuring technique to the $k$ -dimensional ECDF tree. As before, we do not perform any preprocessing to construct the search tree. The ECDF tree is constructed on-the-fly in the process of answering the queries. Initially, all the points are stored at the root of the $k$ -dimensional ECDF tree. In general,when a query search reaches an unexpanded node $v$ we compute the median hyperplane, ${M}_{v}$ ,and partition the data points around ${M}_{v}$ . The two sets are then passed down to the two descendant nodes of $v$ . We also initialize the(k - 1)-dimensional ECDF tree which is to be created at $v$ . Even these lower-dimensional trees are created in a deferred fashion depending upon the queries being answered. The application of deferred data structuring to the ECDF tree results in the following theorem.

我们现在将延迟数据结构化技术应用于$k$维经验累积分布函数（ECDF）树。和之前一样，我们不进行任何预处理来构建搜索树。ECDF树是在回答查询的过程中动态构建的。最初，所有点都存储在$k$维ECDF树的根节点处。一般来说，当查询搜索到达一个未展开的节点$v$时，我们计算中位数超平面${M}_{v}$，并围绕${M}_{v}$对数据点进行划分。然后将这两个集合传递到$v$的两个子节点。我们还初始化将在$v$处创建的(k - 1)维ECDF树。即使是这些低维树也是根据正在回答的查询以延迟方式创建的。将延迟数据结构化应用于ECDF树会得到以下定理。

THEOREM 11. The cost of answering $r$ dominance search queries in $k$ -space is $O\left( {F\left( {n,r,k}\right) }\right)$ ,where

定理11. 在$k$空间中回答$r$个支配搜索查询的代价是$O\left( {F\left( {n,r,k}\right) }\right)$，其中

$$
F\left( {n,r,k}\right)  = \left\{  \begin{array}{ll} n{\log }^{k}r + r{\log }^{k}n, & r \leqq  n, \\  n{\log }^{k}n + r{\log }^{k}n, & r > n. \end{array}\right. 
$$

Proof. The proof will be by induction over both $k$ and $n$ . It is easy to see that the time required to answer a query remains unchanged by the process of deferring the construction of the ECDF tree. This proof will concentrate on the node-expansion component of the processing cost. Clearly,we need not consider the case where $r > n$ since the node-expansion cost cannot exceed the total preprocessing cost of the nondeferred ECDF tree. Let $f\left( {n,r,k}\right)$ denote the worst-case node-expansion cost for answering $r$ queries over $n$ data points in $k$ dimensions using a $k$ -dimensional ECDF tree. When $r$ exceeds $n$ we have $f\left( {n,r,k}\right)  = O\left( {n \cdot  {\log }^{k}n}\right)$ since $n$ queries,each leading to a different leaf, are sufficient to fully expand the ECDF tree. We will now prove that $f\left( {n,r,k}\right)  = O\left( {n \cdot  {\log }^{k}r}\right)$ when $r \leqq  n$ .

证明。本证明将对 $k$ 和 $n$ 进行归纳。不难看出，通过推迟经验累积分布函数（ECDF）树的构建过程，回答查询所需的时间保持不变。本证明将聚焦于处理成本中的节点扩展部分。显然，我们无需考虑 $r > n$ 的情况，因为节点扩展成本不会超过未推迟构建的 ECDF 树的总预处理成本。设 $f\left( {n,r,k}\right)$ 表示使用 $k$ 维 ECDF 树在 $k$ 维空间中对 $n$ 个数据点回答 $r$ 个查询时的最坏情况下的节点扩展成本。当 $r$ 超过 $n$ 时，我们有 $f\left( {n,r,k}\right)  = O\left( {n \cdot  {\log }^{k}n}\right)$，因为 $n$ 个查询（每个查询指向不同的叶子节点）足以完全扩展 ECDF 树。现在我们将证明当 $r \leqq  n$ 时，$f\left( {n,r,k}\right)  = O\left( {n \cdot  {\log }^{k}r}\right)$ 成立。

The basis of this induction is the case where $k = 1$ . Consider the one-dimensional ECDF tree. It is an optimal binary search tree and we can invoke Theorem 3 to show the validity of this theorem. This establishes the base case of our induction over $k$ ,in other words, $f\left( {n,r,1}\right)  = O\left( {n \cdot  \log r}\right)$ when $r \leqq  n$ . The induction hypothesis is that the above result is valid for up to $k - 1$ dimensions,i.e., $f\left( {n,r,k - 1}\right)  = O\left( {n \cdot  {\log }^{k - 1}r}\right)$ when $r \leqq  n$ . We now prove that it must be valid for $k$ dimensions also. At the second level of our nested induction we concentrate on the $k$ -dimensional ECDF tree and use induction over $n$ . It is clear that the $k$ -dimensional ECDF tree for $n = 1$ points will satisfy the above theorem for $r \leqq  n$ . We now assume that the result is valid for up to $n - 1$ points in $k$ dimensions. To complete the proof we show that,under the given assumptions,the result can be extended to $n$ points in $k$ dimensions.

此归纳法的基础是 $k = 1$ 的情况。考虑一维经验累积分布函数（ECDF）树。它是一棵最优二叉搜索树，我们可以引用定理 3 来证明该定理的有效性。这就确立了我们对 $k$ 进行归纳的基础情况，换句话说，当 $r \leqq  n$ 时，$f\left( {n,r,1}\right)  = O\left( {n \cdot  \log r}\right)$ 成立。归纳假设是上述结果在至多 $k - 1$ 维时成立，即当 $r \leqq  n$ 时，$f\left( {n,r,k - 1}\right)  = O\left( {n \cdot  {\log }^{k - 1}r}\right)$ 成立。现在我们证明它在 $k$ 维时也一定成立。在我们嵌套归纳的第二层，我们专注于 $k$ 维的 ECDF 树，并对 $n$ 进行归纳。显然，对于 $n = 1$ 个点的 $k$ 维 ECDF 树，在 $r \leqq  n$ 的情况下将满足上述定理。现在我们假设在 $k$ 维中，对于至多 $n - 1$ 个点，该结果成立。为完成证明，我们表明，在给定假设下，该结果可以扩展到 $k$ 维中的 $n$ 个点。

Consider the root node,say $V$ ,of the $k$ -dimensional ECDF tree for the $n$ points in $P$ . It contains a median hyperplane,say ${M}_{V}$ ,which partitions the $n$ points in $P$ into two equal subsets, ${P}_{1}$ and ${P}_{2}$ . Recall that ${P}_{1}$ is the set of all those points in $P$ which lie below ${M}_{V};{P}_{2}$ is the set of those points in $P$ which lie above ${M}_{V}$ . The left and right subtrees of $V$ are the $k$ -dimensional ECDF trees for ${P}_{1}$ and ${P}_{2}$ ,respectively. We also store at $V$ a(k - 1)-dimensional ECDF tree,say ${T}_{1}$ ,for the projections of the points in ${P}_{1}$ onto ${M}_{V}$ . This lower-dimension tree creates a kind of asymmetry between ${P}_{1}$ and ${P}_{2}$ . This asymmetry can complicate our proof considerably. Therefore,for the purposes of this proof only, we will make a simplifying assumption about the structure of the ECDF tree. We assume that $V$ also contains a(k - 1)-dimensional ECDF tree, say ${T}_{2}$ ,for the projections of the points in ${P}_{2}$ onto ${M}_{V}$ .

考虑$P$中$n$个点的$k$维经验累积分布函数（ECDF）树的根节点，设为$V$。它包含一个中位数超平面，设为${M}_{V}$，该超平面将$P$中的$n$个点划分为两个相等的子集${P}_{1}$和${P}_{2}$。回顾一下，${P}_{1}$是$P$中位于${M}_{V}$下方的所有点的集合，${M}_{V};{P}_{2}$是$P$中位于${M}_{V}$上方的所有点的集合。$V$的左子树和右子树分别是${P}_{1}$和${P}_{2}$的$k$维ECDF树。我们还在$V$处存储了一个(k - 1)维ECDF树，设为${T}_{1}$，用于存储${P}_{1}$中的点在${M}_{V}$上的投影。这种低维树在${P}_{1}$和${P}_{2}$之间造成了一种不对称性。这种不对称性会使我们的证明变得相当复杂。因此，仅出于本证明的目的，我们将对ECDF树的结构做一个简化假设。我们假设$V$还包含一个(k - 1)维ECDF树，设为${T}_{2}$，用于存储${P}_{2}$中的点在${M}_{V}$上的投影。

The search procedure for the ECDF tree is also modified to introduce symmetry. Given a query $q$ ,we first test it with respect to the median hyperplane ${M}_{V}$ . If it lies above ${M}_{V}$ the search continues in the right subtree of $V$ and in ${T}_{1}$ . On the other hand, if $q$ lies below ${M}_{V}$ we continue the search in the left subtree of $V$ as well as ${T}_{2}$ . The search in ${T}_{2}$ is redundant because $q$ ,lying below ${M}_{V}$ ,cannot dominate any point in ${P}_{2}$ . These modifications are made not just at the root but at all nodes in an ECDF tree. It is not very hard to see that these modifications can only increase the running times of our node-expansion algorithm. Moreover, these changes entail performing redundant operations which do not change the outcome of our algorithm. It is clear, therefore, that any upper bounds on the node-expansion costs for the modified ECDF tree also apply to the original deferred data structure.

ECDF树的搜索过程也进行了修改以引入对称性。给定一个查询$q$，我们首先根据中位数超平面${M}_{V}$对其进行测试。如果它位于${M}_{V}$上方，则搜索在$V$的右子树和${T}_{1}$中继续进行。另一方面，如果$q$位于${M}_{V}$下方，则我们在$V$的左子树以及${T}_{2}$中继续搜索。在${T}_{2}$中的搜索是冗余的，因为位于${M}_{V}$下方的$q$不可能支配${P}_{2}$中的任何点。这些修改不仅在根节点进行，而且在ECDF树的所有节点上都进行。不难看出，这些修改只会增加我们的节点扩展算法的运行时间。此外，这些更改需要执行冗余操作，而这些操作不会改变我们算法的结果。因此，很明显，修改后的ECDF树的节点扩展成本的任何上界也适用于原始的延迟数据结构。

We now proceed to complete the induction proof for $r \leqq  n$ . Let ${r}_{1}$ denote the number of queries which lie below the median hyperplane ${M}_{V}$ . These queries continue the search down the left subtree of the root. Let ${r}_{2} = r - {r}_{1}$ denote the remaining queries which continue the search down the right subtree as they lie above the median hyperplane ${M}_{V}$ . Consider the node-expansion costs involved in processing these queries. Finding the median hyperplane ${M}_{V}$ requires $O\left( n\right)$ operations. The ${r}_{1}$ queries lying below ${M}_{V}$ are processed in the left subtree of $V$ (a $k$ -dimensional ECDF tree on $n/2$ points) and in ${T}_{2}$ (a(k - 1)-dimensional ECDF tree on $n/2$ points). The remaining ${r}_{2}$ queries are processed in the right subtree of $V$ (a $k$ -dimensional ECDF tree on $n/2$ points) and in ${T}_{1}$ (a(k - 1)-dimensional ECDF tree on $n/2$ points). This gives us the following bound on the total node-expansion cost entailed by processing $r$ queries:

我们现在继续完成关于$r \leqq  n$的归纳证明。设${r}_{1}$表示位于中位数超平面${M}_{V}$下方的查询数量。这些查询会继续在根节点的左子树中进行搜索。设${r}_{2} = r - {r}_{1}$表示其余的查询，由于它们位于中位数超平面${M}_{V}$上方，因此会继续在右子树中进行搜索。考虑处理这些查询所涉及的节点扩展成本。找到中位数超平面${M}_{V}$需要$O\left( n\right)$次操作。位于${M}_{V}$下方的${r}_{1}$个查询会在$V$（一个基于$n/2$个点的$k$维经验累积分布函数（ECDF）树）的左子树和${T}_{2}$（一个基于$n/2$个点的(k - 1)维ECDF树）中进行处理。其余的${r}_{2}$个查询会在$V$（一个基于$n/2$个点的$k$维ECDF树）的右子树和${T}_{1}$（一个基于$n/2$个点的(k - 1)维ECDF树）中进行处理。这为我们提供了处理$r$个查询所产生的总节点扩展成本的以下边界：

$$
f\left( {n,r,k}\right)  = \mathop{\max }\limits_{{{r}_{1} + {r}_{2} = r}}\left\{  {f\left( {\frac{n}{2},{r}_{1},k}\right)  + f\left( {\frac{n}{2},{r}_{2},k}\right)  + f\left( {\frac{n}{2},{r}_{1},k - 1}\right) }\right. 
$$

$$
\left. {+f\left( {\frac{n}{2},{r}_{2},k - 1}\right)  + O\left( n\right) }\right\}  \text{.}
$$

Using the induction hypotheses we know the exact form of the functions on the right-hand side of the inequality. In particular, we know that these functions are convex. This implies that the right-hand side of the inequality is maximized when ${r}_{1} = {r}_{2} = r/2$ . Putting together all this we have the desired result

利用归纳假设，我们知道不等式右侧函数的精确形式。特别地，我们知道这些函数是凸函数。这意味着当${r}_{1} = {r}_{2} = r/2$时，不等式右侧取得最大值。综合所有这些，我们得到了所需的结果

$$
f\left( {n,r,k}\right)  = O\left( {n \cdot  {\log }^{k}r}\right) ,\;r \leqq  n.
$$

Again,note that this result is valid only for fixed $k$ and $n$ a power of 2 . The constant implicit in the $O$ will,in general,depend on $k$ . Monier’s detailed analyses [9] of Bentley’s algorithm also extends our result to arbitrary $n$ and $k$ .

再次强调，请注意这个结果仅对固定的$k$和2的幂次$n$有效。$O$中隐含的常数通常会依赖于$k$。莫尼尔（Monier）对本特利（Bentley）算法的详细分析[9]也将我们的结果推广到了任意的$n$和$k$。

Bentley [2] actually has a slightly better bound on the preprocessing time for constructing ECDF trees. He makes use of a presorting technique to improve the bound to $O\left( {n \cdot  {\log }^{k - 1}n}\right)$ for $k$ -dimensional ECDF trees on $n$ points. He first sorts all $n$ points by the first coordinate in $O\left( {n \cdot  \log n}\right)$ time. This ordering is maintained at every step, especially when dividing the points into two sets about a median hyperplane for some other coordinate. Consider the two-dimensional ECDF tree. Initially,all $n$ points are stored at the root in order by the first coordinate. After the first query,these $n$ points are partitioned about a median hyperplane and passed down to the children nodes. The ordering by the first coordinate is maintained during this partition. Let ${P}_{1}$ denote the points being passed down to the left subtree, ${P}_{2}$ denotes the points passed down to the right subtree. In the original ECDF tree we would have constructed a one-dimensional ECDF tree for the points in ${P}_{1}$ and stored it at the root. Instead,we now just store the points of ${P}_{1}$ ,in order by the first coordinate,at the root. This process is repeated at every node in the two-dimensional ECDF tree. We now use the two-dimensional ECDF tree as the basic data structure in our recursive construction of a $k$ -dimensional ECDF tree. In effect,we have done away with the one-dimensional ECDF tree. The preprocessing cost for constructing the presorted $k$ -dimensional ECDF tree becomes $O\left( {n \cdot  {\log }^{k - 1}n}\right)  + O\left( {n \cdot  \log n}\right)$ . The new data structure is as easily deferred as the previous one and we have the following result.

本特利（Bentley）[2]实际上对构建经验累积分布函数（ECDF）树的预处理时间有一个稍好的边界。他利用预排序技术将$k$维、$n$个点的ECDF树的边界改进为$O\left( {n \cdot  {\log }^{k - 1}n}\right)$。他首先在$O\left( {n \cdot  \log n}\right)$时间内按第一个坐标对所有$n$个点进行排序。在每一步都保持这种排序，特别是在根据某个其他坐标的中位数超平面将这些点划分为两个集合时。考虑二维ECDF树。最初，所有$n$个点按第一个坐标的顺序存储在根节点。在第一次查询之后，这$n$个点根据中位数超平面进行划分，并传递到子节点。在这个划分过程中，按第一个坐标的排序得以保持。设${P}_{1}$表示传递到左子树的点，${P}_{2}$表示传递到右子树的点。在原始的ECDF树中，我们会为${P}_{1}$中的点构建一个一维ECDF树并将其存储在根节点。相反，我们现在只是将${P}_{1}$中的点按第一个坐标的顺序存储在根节点。在二维ECDF树的每个节点都重复这个过程。我们现在在递归构建$k$维ECDF树时使用二维ECDF树作为基本数据结构。实际上，我们已经不再使用一维ECDF树。构建预排序的$k$维ECDF树的预处理成本变为$O\left( {n \cdot  {\log }^{k - 1}n}\right)  + O\left( {n \cdot  \log n}\right)$。新的数据结构和之前的数据结构一样易于延迟处理，我们得到以下结果。

THEOREM 12. The cost of answering $r$ dominance search queries in $k$ -space is $O\left( {G\left( {n,r,k}\right) }\right)$ ,where

定理12. 在$k$空间中回答$r$支配搜索查询的代价为$O\left( {G\left( {n,r,k}\right) }\right)$，其中

$$
G\left( {n,r,k}\right)  = \left\{  \begin{array}{ll} n\log n + n{\log }^{k - 1}r + r{\log }^{k}n, & r \leqq  n, \\  n{\log }^{k - 1}n + r{\log }^{k}n, & r > n. \end{array}\right. 
$$

Proof. The proof follows from a straightforward modification of the proof for Theorem 11. Note that cost of presorting is subsumed by the node-expansion cost when $r > n$ .

证明. 该证明可通过对定理11的证明进行直接修改得到。注意，当$r > n$时，预排序的代价包含在节点扩展代价中。

6. Conclusion. The paradigm of deferred data structuring has been applied to some search problems. In all cases, we considered on-line queries and developed the search tree as queries were processed. For the problems studied, our method improves on existing strategies involving a preprocessing phase followed by a search phase. An interesting open problem is to design deferred data structures for dynamic data sets in which insertions and deletions are allowed concurrently with query processing.

6. 结论. 延迟数据结构范式已应用于一些搜索问题。在所有情况下，我们考虑在线查询，并在处理查询时构建搜索树。对于所研究的问题，我们的方法改进了现有的先进行预处理阶段再进行搜索阶段的策略。一个有趣的开放性问题是为动态数据集设计延迟数据结构，其中允许在查询处理的同时进行插入和删除操作。

The nearest-neighbor problem [13] asks for the nearest of $n$ data points to a query point. The problem is solved using Voronoi diagrams in $O\left( {\log n}\right)$ search time; the Voronoi diagram can be constructed in $O\left( {n\log n}\right)$ time. There is no known top-down divide-and-conquer algorithm for constructing the Voronoi diagram optimally. The obvious top-down method of constructing the bisector of the left and the right $n/2$ points (see [14] for a definition of the bisector of two sets of points) fails, since sorting reduces to computing this bisector. It remains an interesting open problem whether a deferred data structure can be devised for the nearest-neighbor search problem. Note that the techniques of $§2$ can be used to solve the one-dimensional nearest-neighbor problem.

最近邻问题 [13] 是要找出 $n$ 个数据点中距离查询点最近的点。该问题可使用沃罗诺伊图（Voronoi diagrams）在 $O\left( {\log n}\right)$ 搜索时间内解决；沃罗诺伊图可在 $O\left( {n\log n}\right)$ 时间内构建。目前尚无已知的自顶向下分治算法能最优地构建沃罗诺伊图。构建左右 $n/2$ 个点的平分线（关于两组点的平分线的定义见 [14]）的明显自顶向下方法是行不通的，因为排序可归结为计算这条平分线。能否为最近邻搜索问题设计出一种延迟数据结构仍是一个有趣的开放性问题。注意，$§2$ 中的技术可用于解决一维最近邻问题。

## REFERENCES

## 参考文献

[1] M. BEN-OR, Lower bounds for algebraic computation trees, in Proc. 15th Annual ACM Symposium on Theory of Computing, May 1983, pp. 80-86.

[1] M. 本 - 奥尔（M. BEN - OR），代数计算树的下界，收录于《第 15 届 ACM 计算理论年度研讨会会议录》，1983 年 5 月，第 80 - 86 页。

[2] J. L. BENTLEY, Multidimensional divide and conquer, Comm. ACM, 23 (1980), pp. 214-229.

[2] J. L. 本特利（J. L. BENTLEY），《多维分治法》，《美国计算机协会通讯》（Comm. ACM），第23卷（1980年），第214 - 229页。

[3] M. Blum, R. Floyd, V. PRATT, R. RIVEST, AND R. TARJAN, Time bounds for selection, J. Comput. System. Sci., 7 (1973), pp. 448-461.

[3] M. 布卢姆（M. Blum）、R. 弗洛伊德（R. Floyd）、V. 普拉特（V. PRATT）、R. 李维斯特（R. RIVEST）和R. 塔扬（R. TARJAN），《选择问题的时间界限》，《计算机与系统科学杂志》（J. Comput. System. Sci.），第7卷（1973年），第448 - 461页。

[4] B. M. Chazelle, L. J. GUIBAS, AND D. T. LEE, The power of geometric duality, in Proc. 24th Annual IEEE Annual Symposium on Foundations of Computer Science, November 1983, pp. 217-225.

[4] B. M. 查泽尔（B. M. Chazelle）、L. J. 吉巴斯（L. J. GUIBAS）和D. T. 李（D. T. LEE），《几何对偶性的力量》，收录于《第24届IEEE计算机科学基础年度研讨会论文集》（Proc. 24th Annual IEEE Annual Symposium on Foundations of Computer Science），1983年11月，第217 - 225页。

[5] M. E. DYER, Linear time algorithms for two- and three-variable linear programs, SIAM J. Comput., 13 (1984), pp. 31-45.

[5] M. E. 戴尔（M. E. DYER），《二变量和三变量线性规划的线性时间算法》，《工业与应用数学学会计算杂志》（SIAM J. Comput.），第13卷（1984年），第31 - 45页。

[6] D. G. KIRKPATRICK AND R. SEIDEL, The ultimate planar convex hull algorithm?, SIAM J. Comput., 15 (1986), pp. 287-299.

[6] D. G. 柯克帕特里克（D. G. KIRKPATRICK）和R. 赛德尔（R. SEIDEL），《终极平面凸包算法？》，《工业与应用数学学会计算杂志》（SIAM J. Comput.），第15卷（1986年），第287 - 299页。

[7] D. E. KNUTH, The Art of Computer Programming: Sorting and Searching, 3, Addison-Wesley, New York, 1973, pp. 217-219.

[7] D. E. 克努斯（D. E. KNUTH），《计算机程序设计艺术：排序与查找》（The Art of Computer Programming: Sorting and Searching），第3卷，艾迪生 - 韦斯利出版社（Addison - Wesley），纽约，1973年，第217 - 219页。

[8] N. MEGIDDO,Linear time algorithm for linear programming in ${R}^{3}$ and related problems,SIAM J. Comput., 12 (1983), pp. 759-776.

[8] N. 梅吉多（N. MEGIDDO），“${R}^{3}$中线性规划的线性时间算法及相关问题”（Linear time algorithm for linear programming in ${R}^{3}$ and related problems），《工业与应用数学学会计算杂志》（SIAM J. Comput.），第12卷（1983年），第759 - 776页。

[9] L. MONIER, Combinatorial solutions of multidimensional divide-and-conquer recurrences, J. Algorithms, 1 (1986), pp. 60-74.

[9] L. 莫尼尔（L. MONIER），多维分治递归的组合解法，《算法杂志》（J. Algorithms），1 (1986)，第60 - 74页。

[10] R. MOTWANI AND P. RAGHAVAN, Deferred data structures: query-driven preprocessing for geometric search problems, in Proc. 2nd Annual ACM Symposium on Computational Geometry, Yorktown Heights, NY, June 1986, pp. 303-312.

[10] R. 莫特瓦尼（R. MOTWANI）和P. 拉加万（P. RAGHAVAN），延迟数据结构：几何搜索问题的查询驱动预处理，收录于《第二届ACM计算几何年度研讨会论文集》（Proc. 2nd Annual ACM Symposium on Computational Geometry），纽约约克敦海茨（Yorktown Heights, NY），1986年6月，第303 - 312页。

[11] F. P. Preparata AND M. I. SHAMOS, Computational Geometry: An Introduction, Springer-Verlag, Berlin, New York, 1985.

[11] F. P. 普雷帕拉塔（F. P. Preparata）和M. I. 沙莫斯（M. I. SHAMOS），《计算几何导论》（Computational Geometry: An Introduction），施普林格出版社（Springer - Verlag），柏林、纽约，1985年。

[12] A. Schönkage, M. Paterson, and N. PippenGER, Finding the median, J. Comput. System Sci., 13 (1981), pp. 184-199.

[12] A. 舍恩卡格（A. Schönkage）、M. 帕特森（M. Paterson）和N. 皮彭格（N. PippenGER），寻找中位数，《计算机系统科学杂志》（J. Comput. System Sci.），13 (1981)，第184 - 199页。

[13] M. I. Shamos and D. Hoey, Closest-point problems, in Proc. 16th Annual IEEE Annual Symposium on Foundations of Computer Science, October 1975, pp. 151-162.

[13] M. I. 沙莫斯（M. I. Shamos）和D. 霍伊（D. Hoey），最近点问题，收录于《第16届IEEE计算机科学基础年度研讨会论文集》（Proc. 16th Annual IEEE Annual Symposium on Foundations of Computer Science），1975年10月，第151 - 162页。

[14] M. I. SHAMOS, Computational geometry, Ph.D. thesis, Yale University, New Haven, CT, 1977.

[14] M. I. 沙莫斯（M. I. SHAMOS），《计算几何》，博士学位论文，耶鲁大学（Yale University），美国康涅狄格州纽黑文（New Haven, CT），1977年。