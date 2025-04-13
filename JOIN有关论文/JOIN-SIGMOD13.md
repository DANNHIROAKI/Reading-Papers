# Skew Strikes Back: New Developments in the Theory of Join Algorithms

# 偏斜卷土重来：连接算法理论的新进展

Hung Q. Ngo

吴洪 Q.（Hung Q. Ngo）

University at Buffalo, SUNY

纽约州立大学布法罗分校

hungngo@buffalo.edu

Christopher Ré

克里斯托弗·雷（Christopher Ré）

Stanford University

斯坦福大学

chrismre@cs.stanford.edu

Atri Rudra

阿特里·鲁德拉（Atri Rudra）

University at Buffalo, SUNY

纽约州立大学布法罗分校

atri@buffalo.edu

October 17, 2013

2013年10月17日

## 1 Introduction

## 1 引言

Evaluating the relational join is one of the central algorithmic and most well-studied problems in database systems. A staggering number of variants have been considered including Block-Nested loop join, Hash-Join, Grace, Sort-merge (see Grafe [20] for a survey, and [5, 8, 27] for discussions of more modern issues). Commercial database engines use finely tuned join heuristics that take into account a wide variety of factors including the selectivity of various predicates, memory, IO, etc. In spite of this study of join queries, the textbook description of join processing is suboptimal. This survey describes recent results on join algorithms that have provable worst-case optimality runtime guarantees. We survey recent work and provide a simpler and unified description of these algorithms that we hope is useful for theory-minded readers, algorithm designers, and systems implementors.

评估关系连接是数据库系统中核心的算法问题之一，也是研究最为深入的问题之一。人们已经考虑了数量惊人的变体，包括块嵌套循环连接、哈希连接、格雷斯连接、排序合并连接（有关综述请参阅格拉夫 [20]，有关更现代问题的讨论请参阅 [5, 8, 27]）。商业数据库引擎使用经过精细调整的连接启发式方法，这些方法会考虑各种因素，包括各种谓词的选择性、内存、输入输出等。尽管对连接查询进行了大量研究，但教科书对连接处理的描述并非最优。本综述介绍了连接算法的最新成果，这些成果具有可证明的最坏情况最优运行时间保证。我们对近期的工作进行了综述，并对这些算法进行了更简单、统一的描述，希望对理论研究者、算法设计者和系统实现者有所帮助。

Much of this progress can be understood by thinking about a simple join evaluation problem that we illustrate with the so-called triangle query, a query that has become increasingly popular in the last decade with the advent of social networks, biological motifs, and graph databases [39,40]

通过思考一个简单的连接评估问题，可以理解这一进展的大部分内容。我们用所谓的三角形查询来说明这个问题，随着社交网络、生物基序和图数据库的出现，这个查询在过去十年中越来越受欢迎 [39,40]

Suppose that one is given a graph with $N$ edges,how many distinct triangles can there be in the graph?

假设给定一个有 $N$ 条边的图，该图中最多可以有多少个不同的三角形？

A first bound is to say that there are at most $N$ edges,and hence at most $O\left( {N}^{3}\right)$ triangles. A bit more thought suggests that every triangle is indexed by any two of its sides and hence there at most $O\left( {N}^{2}\right)$ triangles. However,the correct,tight,and non-trivial asymptotic is $O\left( {N}^{3/2}\right)$ . An example of the questions this survey is how do we list all the triangles in time $O\left( {N}^{3/2}\right)$ ? Such an algorithm would have a worst-case optimal running time. In contrast, traditional databases evaluate joins pairwise, and as has been noted by several authors,this forces them to run in time $\Omega \left( {N}^{2}\right)$ on some instance of the triangle query. This survey gives an overview of recent developments that establish such non-trivial bounds for all join queries and algorithms that meet these bounds, which we call worst-case optimal join algorithms.

第一个界限是说最多有 $N$ 条边，因此最多有 $O\left( {N}^{3}\right)$ 个三角形。再深入思考一下，每个三角形都可以由它的任意两条边来索引，因此最多有 $O\left( {N}^{2}\right)$ 个三角形。然而，正确、严格且非平凡的渐近界限是 $O\left( {N}^{3/2}\right)$。本综述要解决的一个问题示例是，我们如何在 $O\left( {N}^{3/2}\right)$ 时间内列出所有三角形？这样的算法将具有最坏情况最优运行时间。相比之下，传统数据库是成对评估连接的，正如几位作者所指出的，这使得它们在某些三角形查询实例上的运行时间为 $\Omega \left( {N}^{2}\right)$。本综述概述了近期的进展，这些进展为所有连接查询建立了这种非平凡的界限，并介绍了满足这些界限的算法，我们称之为最坏情况最优连接算法。

Estimates on the output size of join have been known since the 1990s, thanks to the work of Friedgut and Kahn [14] in the context of bounding the number of occurrences of a given small hypergraph inside a large hypergraph. More recently and more generally, tight estimates for the natural join problem were derived by Grohe-Marx [23] and Atserias-Grohe-Marx [3] (AGM henceforth). In fact, similar bounds can be traced back to the 1940s in geometry, where it was known as the famous Loomis-Whitney inequality [29]. The most general geometric bound is by Bollobás-Thomason in the 1990s [6]. We proved (with Porat) that AGM and the discrete version of Bollobás-Thomason are equivalent [32], and so the connection between these areas is deep.

自20世纪90年代以来，由于弗里德古特和卡恩 [14] 在限制给定小超图在大超图中出现次数的背景下所做的工作，人们就已经知道了对连接输出大小的估计。最近，更一般地，格罗赫 - 马克思 [23] 和阿塞里亚斯 - 格罗赫 - 马克思 [3]（以下简称AGM）推导出了自然连接问题的严格估计。事实上，类似的界限可以追溯到20世纪40年代的几何学，当时它被称为著名的卢米斯 - 惠特尼不等式 [29]。最一般的几何界限是20世纪90年代博洛巴斯 - 托马森给出的 [6]。我们（与波拉特一起）证明了AGM和博洛巴斯 - 托马森的离散版本是等价的 [32]，因此这些领域之间的联系非常深刻。

Connections of join size to arcane geometric bounds may reasonably lead a practitioner to believe that the cause of suboptimality is a mysterious force wholly unknown to them-but it is not; it is the old enemy of the database optimizer, skew. We hope to highlight two conceptual messages with this survey:

连接大小与晦涩的几何界限之间的联系可能会让从业者合理地认为，次优性的原因是一种他们完全未知的神秘力量——但事实并非如此；它是数据库优化器的老对手，偏斜。我们希望通过本综述强调两个概念性信息：

- The main ideas of the the algorithms presented here are an optimal way of avoiding skew - something database practitioners have been fighting with for decades. We describe a theoretical basis for one family of techniques to cope with skew by relating them to geometry.

- 这里介绍的算法的主要思想是一种避免偏斜的最优方法——这是数据库从业者几十年来一直在与之斗争的问题。我们通过将一类处理偏斜的技术与几何学联系起来，为其提供了理论基础。

- The second idea is a challenge to the database dogma of doing "one join at a time," as is done in traditional database systems. We show that there are classes of queries for which any join-project plan is destined to be slower than the best possible run time by a polynomial factor in the data size.

- 第二个观点是对传统数据库系统中“一次进行一次连接”这一数据库教条的挑战。我们表明，对于某些类型的查询，任何连接 - 投影计划的运行时间注定会比最佳可能运行时间慢一个与数据大小成多项式关系的因子。

Outline of the Survey. We begin with a short (and necessarily incomplete) history of join processing with a focus on recent history. In Section 2, we describe how these algorithms work for the triangle query. In Section 3, we describe how to use these new size bounds for join queries. In Section 4, we provide new simplified proofs of these bounds and join algorithms. Finally, we describe two open questions in Section 5

综述大纲。我们首先简要（必然不完整）回顾一下连接处理的历史，重点关注近期历史。在第2节中，我们描述这些算法如何处理三角形查询。在第3节中，我们描述如何将这些新的大小边界用于连接查询。在第4节中，我们为这些边界和连接算法提供新的简化证明。最后，我们在第5节中描述两个开放性问题。

## A Brief History of Join Processing

## 连接处理简史

Conjunctive query evaluation in general and join query evaluation in particular have a very long history and deep connections to logic and constraint satisfaction [7, 9, 13, 17, 19, 28, 33, 41]. Most of the join algorithms with provable performance guarantees work for specific classes of queries ${}^{1}$ As we describe, there are two major approaches for join processing: using structural information of the query and using cardinality information. As we explain, the AGM bounds are exciting because they bring together both types of information.

一般的合取查询评估，特别是连接查询评估，有着悠久的历史，并且与逻辑和约束满足问题有着深刻的联系 [7, 9, 13, 17, 19, 28, 33, 41]。大多数具有可证明性能保证的连接算法适用于特定类型的查询 ${}^{1}$ 正如我们所描述的，连接处理主要有两种方法：利用查询的结构信息和利用基数信息。正如我们所解释的，AGM边界之所以令人兴奋，是因为它们将这两种类型的信息结合在了一起。

The Structural Approaches On the theoretical side, many algorithms use some structural property of the query such as acyclicity or bounded "width." For example, when the query is acyclic, the classic algorithm of Yannakakis [45] runs in time essentially linear in the input plus output size. A query is acyclic if and only if it has a join tree, which can be constructed using the textbook GYO-reduction [21, 46]

结构方法 在理论方面，许多算法利用查询的某些结构属性，如无环性或有界“宽度”。例如，当查询是无环的时，Yannakakis [45] 的经典算法的运行时间基本上与输入和输出的大小呈线性关系。当且仅当查询有一个连接树时，它才是无环的，这个连接树可以使用教科书上的GYO约简 [21, 46] 来构建。

Subsequent work further expand the classes of queries which can be evaluated in polynomial time. These work define progressively more general notions of "width" for a query, which intuitively measures how far a query is from being acyclic. Roughly, these results state that if the corresponding notion of "width" is bounded by a constant, then the query is "tractable," i.e. there is a polynomial time algorithm to evaluate it. For example, Gyssens et al. [24, 25] showed that queries with bounded "degree of acyclicity" are tractable. Then come query width (qw) from Chekuri and Rajaraman [9], hypertree width and generalized hypertree width (ghw) from Gottlob et al. [18, 37]. These are related to the treewidth (tw) of a query's hypergraph, rooted in Robertson and Seymour on graph minors [36]. Acyclic queries are exactly those with $\mathrm{{qw}} = 1$ .

后续工作进一步扩展了可以在多项式时间内评估的查询类别。这些工作为查询逐步定义了更通用的“宽度”概念，直观地衡量了一个查询与无环查询的偏离程度。大致来说，这些结果表明，如果相应的“宽度”概念有一个常数界，那么该查询是“可处理的”，即存在一个多项式时间算法来评估它。例如，Gyssens等人 [24, 25] 表明具有有界“无环度”的查询是可处理的。然后是Chekuri和Rajaraman [9] 提出的查询宽度（qw），Gottlob等人 [18, 37] 提出的超树宽度和广义超树宽度（ghw）。这些与查询超图的树宽（tw）有关，其根源在于Robertson和Seymour关于图子式的研究 [36]。无环查询恰好是那些 $\mathrm{{qw}} = 1$ 的查询。

Cardinality-based Approaches Width only tells half of the story, as was wonderfully articulated in Scar-cello's SIGMOD Record paper [37]:

基于基数的方法 正如Scar - cello在SIGMOD Record论文 [37] 中精彩阐述的那样，宽度只说明了问题的一半：

decomposition methods focus "only" on structural features, while they completely disregard "quantitative" aspects of the query, that may dramatically affect the query-evaluation time. Said another way, the width approach disregards the input relation sizes and summarizes them in a single number, $N$ . As a result,the run time of these structural approaches is $O\left( {{N}^{w + 1}\log N}\right)$ ,where $N$ is the input size and $w$ is the corresponding width measure. On the other hand,commercial RDBMSs seem to place little emphasis on the structural property of the query and tremendous emphasis on the cardinality side of join processing. Commercial databases often process a join query by breaking a complex multiway join into a series of pairwise joins; an approach first described in the seminal System R, Selinger-style optimizer from the 1970 [38]. However, throwing away this structural information comes at a cost: any join-project plan is destined to be slower than the best possible run time by a polynomial factor in the data size.

分解方法“仅”关注结构特征，而完全忽略了查询的“定量”方面，这些方面可能会极大地影响查询评估时间。换句话说，宽度方法忽略了输入关系的大小，并将它们总结为一个单一的数字 $N$。因此，这些结构方法的运行时间是 $O\left( {{N}^{w + 1}\log N}\right)$，其中 $N$ 是输入大小，$w$ 是相应的宽度度量。另一方面，商业关系数据库管理系统（RDBMS）似乎很少强调查询的结构属性，而非常强调连接处理的基数方面。商业数据库通常通过将复杂的多路连接分解为一系列的成对连接来处理连接查询；这种方法最早在20世纪70年代具有开创性的System R、Selinger风格的优化器中被描述 [38]。然而，丢弃这些结构信息是有代价的：任何连接 - 投影计划的运行时间注定会比最佳可能运行时间慢一个与数据大小成多项式关系的因子。

---

<!-- Footnote -->

${}^{1}$ Throughout this survey,we will measure the run time of join algorithms in terms of the input data,assuming the input query has constant size; this is known as the data complexity measure, which is standard in database theory [41].

${}^{1}$ 在整个综述中，我们将根据输入数据来衡量连接算法的运行时间，假设输入查询的大小是常数；这被称为数据复杂度度量，在数据库理论中是标准的 [41]。

<!-- Footnote -->

---

Bridging This Gap A major recent result from AGM [3, 23] is the key to bridging this gap: AGM derived a tight bound on the output size of a join query as a function of individual input relation sizes and a much finer notion of "width". The AGM bound leads to the notion of fractional query number and eventually fractional hypertree width (fhw) which is strictly more general than all of the above width notions [31]. To summarize, for the same query, it can be shown that

弥合这一差距 AGM [3, 23] 最近的一个主要成果是弥合这一差距的关键：AGM推导出了连接查询输出大小的一个紧密边界，该边界是各个输入关系大小和一个更精细的“宽度”概念的函数。AGM边界引出了分数查询数的概念，并最终引出了分数超树宽度（fhw），它比上述所有宽度概念都更具一般性 [31]。总结来说，对于同一个查询，可以证明

$$
\text{ fhw } \leq  \text{ ghw } \leq  \text{ qw } \leq  \text{ tw } + 1,
$$

and the join-project algorithm from AGM runs in time $O\left( {{N}^{\mathrm{{fhw}} + 1}\log N}\right)$ . AGM’s bound is sharp enough to take into account cardinality information, and they can be much better when the input relation sizes vary. The bound takes into account both the input relation statistics and the structural properties of the query. The question is whether it is possible and how to turn the bound into join algorithms,with runtime $O\left( {N}^{\text{fwh }}\right)$ and much better when input relations do not have the same size.

并且来自AGM（原文未明确全称）的连接 - 投影算法的运行时间为$O\left( {{N}^{\mathrm{{fhw}} + 1}\log N}\right)$。AGM的边界足够精确，能够考虑到基数信息，并且当输入关系的大小不同时，其效果可能会好得多。该边界同时考虑了输入关系的统计信息和查询的结构属性。问题在于是否有可能以及如何将这个边界转化为连接算法，使其运行时间为$O\left( {N}^{\text{fwh }}\right)$，并且在输入关系大小不同时效果更好。

The first such worst-case optimal join algorithm was designed by the authors (and Porat) in 2012 [32]. Soon after, an algorithm (with a simpler description) with a similar optimality guarantee was presented soon after called "Leapfrog Triejoin" [42]. Remarkably this algorithm was implemented in a commercial database system before its optimality guarantees were discovered. A key idea in the algorithms is handling skew in a theoretically optimal way, and uses many of the same techniques that database management systems have used for decades heuristically [12, 43, 44]

第一个这样的最坏情况最优连接算法是由作者（以及波拉特）在2012年设计的[32]。不久之后，一个具有类似最优性保证（描述更简单）的算法被提出，称为“Leapfrog Triejoin”（蛙跳式字典树连接）[42]。值得注意的是，这个算法在其最优性保证被发现之前就已经在一个商业数据库系统中实现了。这些算法的一个关键思想是以理论上最优的方式处理数据倾斜，并且使用了许多数据库管理系统几十年来一直启发式使用的相同技术[12, 43, 44]

A technical contribution of this survey is to describe the algorithms from [32] and [42] and their analyses in one unifying (and simplified) framework. In particular, we make the observation that these join algorithms are in fact special cases of a single join algorithm. This result is new and serves to explain the common link between these join algorithms. We also illustrate some unexpected connections with geometry,which ${w}^{c}$ believe are interesting in their own right and may be the basis for further theoretical development.

本综述的一项技术贡献是在一个统一（且简化）的框架中描述文献[32]和[42]中的算法及其分析。特别是，我们观察到这些连接算法实际上是单个连接算法的特殊情况。这一结果是新的，有助于解释这些连接算法之间的共同联系。我们还说明了与几何学的一些意想不到的联系，我们认为这些联系本身就很有趣，并且可能是进一步理论发展的基础。

## 2 Much ado about the triangle

## 2 关于三角形的诸多讨论

We begin with the triangle query

我们从三角形查询开始

$$
{Q}_{ \bigtriangleup  } = R\left( {A,B}\right)  \bowtie  S\left( {B,C}\right)  \bowtie  T\left( {A,C}\right) .
$$

The above query is the simplest cyclic query and is rich enough to illustrate most of the ideas in the new join algorithms 2 We first describe the traditional way to evaluate this query and how skew impacts this query. We then develop two closely related algorithmic ideas allowing us to mitigate the impact of skew in these examples; they are the key ideas behind the recent join processing algorithms.

上述查询是最简单的循环查询，足以说明新连接算法中的大多数思想。我们首先描述评估此查询的传统方法以及数据倾斜如何影响此查询。然后，我们提出两个密切相关的算法思路，使我们能够减轻这些示例中数据倾斜的影响；它们是近期连接处理算法背后的关键思想。

---

<!-- Footnote -->

${}^{2}$ This query can be used to list all triangles in a given graph $G = \left( {V,E}\right)$ ,if we set $R,S$ and $T$ to consist of all pairs(u,v)and (v,u)for which ${uv}$ is an edge. Due to symmetry,each triangle in $G$ will be listed 6 times in the join.

${}^{2}$ 如果我们将$R,S$和$T$设置为包含所有满足${uv}$为边的对(u, v)和(v, u)，则此查询可用于列出给定图$G = \left( {V,E}\right)$中的所有三角形。由于对称性，$G$中的每个三角形在连接中都会被列出6次。

<!-- Footnote -->

---

### 2.1 Why traditional join plans are suboptimal

### 2.1 为什么传统连接计划不是最优的

The textbook way to evaluate any join query,including ${Q}_{\bigtriangleup }$ ,is to determine the best pair-wise join plan [35, Ch. 15]. Figure 1 illustrates three plans that a conventional RDBMS would use for this query. For example, the first plan is to compute the intermediate join $P = R \boxtimes  T$ and then compute $P \boxtimes  S$ as the final output.

评估任何连接查询（包括${Q}_{\bigtriangleup }$）的教科书式方法是确定最佳的两两连接计划[35, 第15章]。图1展示了传统关系型数据库管理系统（RDBMS）会用于此查询的三种计划。例如，第一个计划是计算中间连接$P = R \boxtimes  T$，然后计算$P \boxtimes  S$作为最终输出。

<!-- Media -->

<!-- figureText: ✘ ✘ ✘ ✘ $R$ ✘ ✘ -->

<img src="https://cdn.noedgeai.com/0195cc9c-3871-780e-84a7-1cc8384f8f6e_3.jpg?x=612&y=426&w=572&h=217&r=0"/>

Figure 1: The three pair-wise join plans for ${Q}_{\bigtriangleup }$ .

图1：${Q}_{\bigtriangleup }$的三种两两连接计划。

<!-- Media -->

We next give a family of instances for which any of the above three join plans must run in time $\Omega \left( {N}^{2}\right)$ because the intermediate relation $P$ is too large. Let $m \geq  1$ be a positive integer. The instance family is illustrated in Figure 2,where the domain of the attributes $A,B$ and $C$ are $\left\{  {{a}_{0},{a}_{1},\ldots ,{a}_{m}}\right\}  ,\left\{  {{b}_{0},{b}_{1},\ldots ,{b}_{m}}\right\}$ , and $\left\{  {{c}_{0},{c}_{1},\ldots ,{c}_{m}}\right\}$ respectively. In Figure 2,the unfilled circles denote the values ${a}_{0},{b}_{0}$ and ${c}_{0}$ respectively while the black circles denote the rest of the values.

接下来，我们给出一类实例，对于这类实例，上述三种连接计划中的任何一种都必须在时间$\Omega \left( {N}^{2}\right)$内运行，因为中间关系$P$太大了。设$m \geq  1$为一个正整数。该实例类如图2所示，其中属性$A,B$和$C$的域分别为$\left\{  {{a}_{0},{a}_{1},\ldots ,{a}_{m}}\right\}  ,\left\{  {{b}_{0},{b}_{1},\ldots ,{b}_{m}}\right\}$和$\left\{  {{c}_{0},{c}_{1},\ldots ,{c}_{m}}\right\}$。在图2中，空心圆分别表示值${a}_{0},{b}_{0}$和${c}_{0}$，而实心圆表示其余的值。

For this instance each relation has $N = {2m} + 1$ tuples and $\left| {Q}_{ \bigtriangleup  }\right|  = {3m} + 1$ ; however,any pair-wise join has size ${m}^{2} + m$ . Thus,for large $m$ ,any of the three join plans will take $\Omega \left( {N}^{2}\right)$ time. In fact,it can be shown that even if we allow projections in addition to joins,the $\Omega \left( {N}^{2}\right)$ bound still holds. (See Lemma 4.2.) By contrast,the two algorithms shown in the next section runs in time $O\left( N\right)$ ,which is optimal because the output itself has $\Omega \left( N\right)$ tuples!

对于这个实例，每个关系有 $N = {2m} + 1$ 个元组且 $\left| {Q}_{ \bigtriangleup  }\right|  = {3m} + 1$ ；然而，任意两两连接的结果大小为 ${m}^{2} + m$ 。因此，对于较大的 $m$ ，三种连接计划中的任何一种都将花费 $\Omega \left( {N}^{2}\right)$ 的时间。事实上，可以证明，即使我们除了连接操作还允许投影操作， $\Omega \left( {N}^{2}\right)$ 这个界限仍然成立。（见引理 4.2）相比之下，下一节展示的两种算法的运行时间为 $O\left( N\right)$ ，这是最优的，因为输出本身就有 $\Omega \left( N\right)$ 个元组！

### 2.2 Algorithm 1: The Power of Two Choices

### 2.2 算法 1：二选一的力量

Inspecting the bad example above,one can see a root cause for the large intermediate relation: ${a}_{0}$ has "high degree" or in the terminology to follow it is heavy. In other words, it is an example of skew. To cope with skew, we shall take a strategy often employed in database systems: we deal with nodes of high and low skew using different join techniques [12, 44]. The first goal then is to understand when a value has high skew. To shorten notations,for each ${a}_{i}$ define

审视上述不良示例，可以看出中间关系规模较大的一个根本原因： ${a}_{0}$ 具有“高度数”，或者用后续术语来说，它是“重的”。换句话说，这是一个数据倾斜的例子。为了应对数据倾斜，我们将采用数据库系统中常用的一种策略：我们使用不同的连接技术来处理高倾斜和低倾斜的节点 [12, 44]。那么，首要目标是理解一个值何时具有高倾斜度。为了简化符号，对于每个 ${a}_{i}$ ，定义

$$
{Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack   \mathrel{\text{:=}} {\pi }_{B,C}\left( {{\sigma }_{A = {a}_{i}}\left( {Q}_{ \bigtriangleup  }\right) }\right) .
$$

We will call ${a}_{i}$ heavy if

如果满足以下条件，我们将称 ${a}_{i}$ 为“重的”：

$$
\left| {{\sigma }_{A = {a}_{i}}\left( {R \bowtie  T}\right) }\right|  \geq  \left| {{Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack  }\right| .
$$

In other words,the value ${a}_{i}$ is heavy if its contribution to the size of intermediate relation $R \bowtie  S$ is greater than its contribution to the size of the output. Since

换句话说，如果值 ${a}_{i}$ 对中间关系 $R \bowtie  S$ 规模的贡献大于它对输出规模的贡献，那么该值就是“重的”。因为

$$
\left| {{\sigma }_{A = {a}_{i}}\left( {R \bowtie  S}\right) }\right|  = \left| {{\sigma }_{A = {a}_{i}}R}\right|  \cdot  \left| {{\sigma }_{A = {a}_{i}}S}\right| ,
$$

we can easily compute the left hand side of the above inequality from an appropriate index of the input relations. Of course,we do not know $\left| {{Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack  }\right|$ until after we have computed ${Q}_{ \bigtriangleup  }$ . However,note that we always have ${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack   \subseteq  S$ . Thus,we will use $\left| S\right|$ as a proxy for $\left| {{Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack  }\right|$ . The two choices come from the following two ways of computing ${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack$ :

我们可以从输入关系的适当索引轻松计算出上述不等式的左边。当然，在计算出 ${Q}_{ \bigtriangleup  }$ 之前，我们并不知道 $\left| {{Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack  }\right|$ 的值。然而，请注意，我们始终有 ${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack   \subseteq  S$ 。因此，我们将使用 $\left| S\right|$ 作为 $\left| {{Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack  }\right|$ 的代理。这两种选择来自以下两种计算 ${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack$ 的方式：

(i) Compute ${\sigma }_{A = {a}_{i}}\left( R\right)  \bowtie  {\sigma }_{A = {a}_{i}}\left( T\right)$ and filter the results by probing against $S$ or

(i) 计算 ${\sigma }_{A = {a}_{i}}\left( R\right)  \bowtie  {\sigma }_{A = {a}_{i}}\left( T\right)$ 并通过与 $S$ 进行探查来过滤结果；或者

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195cc9c-3871-780e-84a7-1cc8384f8f6e_4.jpg?x=1031&y=200&w=423&h=266&r=0"/>

$$
R = \left\{  {a}_{0}\right\}   \times  \left\{  {{b}_{0},\ldots ,{b}_{m}}\right\}   \cup  \left\{  {{a}_{0},\ldots ,{a}_{m}}\right\}   \times  \left\{  {b}_{0}\right\}  
$$

$$
S = \left\{  {b}_{0}\right\}   \times  \left\{  {{c}_{0},\ldots ,{c}_{m}}\right\}   \cup  \left\{  {{b}_{0},\ldots ,{b}_{m}}\right\}   \times  \left\{  {c}_{0}\right\}  
$$

$$
T = \left\{  {a}_{0}\right\}   \times  \left\{  {{c}_{0},\ldots ,{c}_{m}}\right\}   \cup  \left\{  {{a}_{0},\ldots ,{a}_{m}}\right\}   \times  \left\{  {c}_{0}\right\}  
$$

Figure 2: Counter-example for join-project only plans for the triangles (left) and an illustration for $m = 4$ (right). The pairs connected by the red/green/blue edges form the tuples in the relations $R/S/T$ respectively. Note that the in this case each relation has $N = {2m} + 1 = 9$ tuples and there are ${3m} + 1 = {13}$ output tuples in ${Q}_{ \bigtriangleup  }$ . Any pair-wise join however has size ${m}^{2} + m = {20}$ .

图 2：三角形仅连接 - 投影计划的反例（左）和 $m = 4$ 的图示（右）。由红/绿/蓝边连接的对分别构成了关系 $R/S/T$ 中的元组。请注意，在这种情况下，每个关系有 $N = {2m} + 1 = 9$ 个元组，并且 ${Q}_{ \bigtriangleup  }$ 中有 ${3m} + 1 = {13}$ 个输出元组。然而，任意两两连接的结果大小为 ${m}^{2} + m = {20}$ 。

<!-- Media -->

(ii) Consider each tuple in $\left( {b,c}\right)  \in  S$ and check if $\left( {{a}_{i},b}\right)  \in  R$ and $\left( {{a}_{i},c}\right)  \in  T$ .

(ii) 考虑 $\left( {b,c}\right)  \in  S$ 中的每个元组，并检查是否满足 $\left( {{a}_{i},b}\right)  \in  R$ 和 $\left( {{a}_{i},c}\right)  \in  T$ 。

We pick option (i) when ${a}_{i}$ is light (low skew) and pick option (ii) when ${a}_{i}$ is heavy (high skew).

当 ${a}_{i}$ 是“轻的”（低倾斜度）时，我们选择选项 (i)；当 ${a}_{i}$ 是“重的”（高倾斜度）时，我们选择选项 (ii)。

Example 1. Let us work through the motivating example from Figure 2. When we compute ${Q}_{\Delta }\left\lbrack  {a}_{0}\right\rbrack$ ,we realize that ${a}_{0}$ is heavy and hence,we use option (ii) above. Since here we just scan tuples in $S$ ,computing ${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{0}\right\rbrack$ takes $O\left( m\right)$ time. On the other hand,when we want to compute ${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack$ for $i \geq  1$ ,we realize that these ${a}_{i}$ ’s are light and so we take option (i). In these cases $\left| {{\sigma }_{A = {a}_{i}}R}\right|  = \left| {{\sigma }_{A = {a}_{i}}T}\right|  = 1$ and hence the algorithm runs in time $O\left( 1\right)$ . As there are $m$ such light ${a}_{i}$ ’s,the algorithm overall takes $O\left( m\right)$ each on the heavy and light vertices and thus $O\left( m\right)  = O\left( N\right)$ overall which is best possible since the output size is $\Theta \left( N\right)$ .

示例1。让我们来分析图2中的激励示例。当我们计算${Q}_{\Delta }\left\lbrack  {a}_{0}\right\rbrack$时，我们发现${a}_{0}$是“重的”（heavy），因此，我们选择上述选项（ii）。由于这里我们只是扫描$S$中的元组，计算${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{0}\right\rbrack$需要$O\left( m\right)$的时间。另一方面，当我们想为$i \geq  1$计算${Q}_{ \bigtriangleup  }\left\lbrack  {a}_{i}\right\rbrack$时，我们发现这些${a}_{i}$是“轻的”（light），所以我们选择选项（i）。在这些情况下$\left| {{\sigma }_{A = {a}_{i}}R}\right|  = \left| {{\sigma }_{A = {a}_{i}}T}\right|  = 1$，因此该算法的运行时间为$O\left( 1\right)$。由于有$m$个这样的轻${a}_{i}$，该算法在重顶点和轻顶点上分别花费$O\left( m\right)$的时间，因此总体上花费$O\left( m\right)  = O\left( N\right)$的时间，这是最优的，因为输出规模为$\Theta \left( N\right)$。

Algorithm and Analysis Algorithm 1 fully specifies how to compute ${Q}_{ \bigtriangleup  }$ using the above idea of two choices. Given that the relations $R,S$ ,and $T$ are already indexed appropriately,computing $L$ in line 2 can easily be done in time $O\left( {\min \{ \left| R\right| ,\left| S\right| ,\left| T\right| \} }\right)$ . Then,for each $a \in  L$ ,the body of the for loop from line 4 to line 11 clearly takes time in the order of

算法与分析 算法1详细说明了如何使用上述两种选择的思想来计算${Q}_{ \bigtriangleup  }$。假设关系$R,S$和$T$已经进行了适当的索引，那么在第2行计算$L$可以很容易地在$O\left( {\min \{ \left| R\right| ,\left| S\right| ,\left| T\right| \} }\right)$的时间内完成。然后，对于每个$a \in  L$，从第4行到第11行的for循环体显然花费的时间量级为

$$
\min \left( {\left| {{\sigma }_{A = a}R}\right|  \cdot  \left| {{\sigma }_{A = a}T}\right| ,\left| S\right| }\right) ,
$$

thanks to the power of two choices! Thus, the overall time spent by the algorithm is up to constant factors

多亏了两种选择的力量！因此，该算法总体花费的时间在常数因子范围内

$$
\mathop{\sum }\limits_{{a \in  L}}\min \left( {\left| {{\sigma }_{A = a}R}\right|  \cdot  \left| {{\sigma }_{A = a}T}\right| ,\left| S\right| }\right) . \tag{1}
$$

We bound the sum above by using two inequalities. The first is the simple observation that for any $x,y \geq  0$

我们通过使用两个不等式来界定上述和式。第一个是一个简单的观察结果，即对于任何$x,y \geq  0$

$$
\min \left( {x,y}\right)  \leq  \sqrt{xy}\text{.} \tag{2}
$$

The second is the famous Cauchy-Schwarz inequality ${}^{3}$ :

第二个是著名的柯西 - 施瓦茨不等式（Cauchy - Schwarz inequality）${}^{3}$：

$$
\mathop{\sum }\limits_{{a \in  L}}{x}_{a} \cdot  {y}_{a} \leq  \sqrt{\mathop{\sum }\limits_{{a \in  L}}{x}_{a}^{2}} \cdot  \sqrt{\mathop{\sum }\limits_{{a \in  L}}{y}_{a}^{2}}, \tag{3}
$$

where ${\left( {x}_{a}\right) }_{a \in  L}$ and ${\left( {y}_{a}\right) }_{a \in  L}$ are vectors of real values. Applying (2) to (1),we obtain

其中${\left( {x}_{a}\right) }_{a \in  L}$和${\left( {y}_{a}\right) }_{a \in  L}$是实值向量。将（2）应用于（1），我们得到

$$
\mathop{\sum }\limits_{{a \in  L}}\sqrt{\left| {{\sigma }_{A = a}R}\right|  \cdot  \left| {{\sigma }_{A = a}T}\right|  \cdot  \left| S\right| } \tag{4}
$$

$$
 = \sqrt{\left| S\right| } \cdot  \mathop{\sum }\limits_{{a \in  L}}\sqrt{\left| {\sigma }_{A = a}R\right| } \cdot  \sqrt{\left| {\sigma }_{A = a}T\right| } \tag{5}
$$

$$
 \leq  \sqrt{\left| S\right| } \cdot  \sqrt{\mathop{\sum }\limits_{{a \in  L}}\left| {{\sigma }_{A = a}R}\right| } \cdot  \sqrt{\mathop{\sum }\limits_{{a \in  L}}\left| {{\sigma }_{A = a}T}\right| }
$$

$$
 \leq  \sqrt{\left| S\right| } \cdot  \sqrt{\mathop{\sum }\limits_{{a \in  {\pi }_{A}\left( R\right) }}\left| {{\sigma }_{A = a}R}\right| } \cdot  \sqrt{\mathop{\sum }\limits_{{a \in  {\pi }_{A}\left( T\right) }}\left| {{\sigma }_{A = a}T}\right| }
$$

$$
 = \sqrt{\left| S\right| } \cdot  \sqrt{\left| R\right| } \cdot  \sqrt{\left| T\right| }\text{.}
$$

---

<!-- Footnote -->

${}^{3}$ The inner product of two vectors is at most the product of their length

${}^{3}$ 两个向量的内积至多等于它们长度的乘积

<!-- Footnote -->

---

<!-- Media -->

Algorithm 1 Computing ${Q}_{ \bigtriangleup  }$ with power of two choices.

算法1 使用两种选择的力量计算${Q}_{ \bigtriangleup  }$。

---

Input: $R\left( {A,B}\right) ,S\left( {B,C}\right) ,T\left( {A,C}\right)$ in sorted order

输入：按排序顺序排列的$R\left( {A,B}\right) ,S\left( {B,C}\right) ,T\left( {A,C}\right)$

	${Q}_{\Delta } \leftarrow  \varnothing$

	$L \leftarrow  {\pi }_{A}\left( R\right)  \cap  {\pi }_{A}\left( T\right)$

	For each $a \in  L$ do

	对于每个$a \in  L$执行

		If $\left| {{\sigma }_{A = a}R}\right|  \cdot  \left| {{\sigma }_{A = a}T}\right|  \geq  \left| S\right|$ then

		如果$\left| {{\sigma }_{A = a}R}\right|  \cdot  \left| {{\sigma }_{A = a}T}\right|  \geq  \left| S\right|$则

			For each $\left( {b,c}\right)  \in  S$ do

			对于每个$\left( {b,c}\right)  \in  S$执行

				If $\left( {a,b}\right)  \in  R$ and $\left( {a,c}\right)  \in  T$ then

				如果$\left( {a,b}\right)  \in  R$且$\left( {a,c}\right)  \in  T$则

					Add(a,b,c)to ${Q}_{\Delta }$

					将(a,b,c)添加到${Q}_{\Delta }$中

		else

			 否则

			For each $b \in  {\pi }_{B}\left( {{\sigma }_{A = a}R}\right)  \land  c \in  {\pi }_{C}\left( {{\sigma }_{A = a}T}\right)$ do

			对于每个$b \in  {\pi }_{B}\left( {{\sigma }_{A = a}R}\right)  \land  c \in  {\pi }_{C}\left( {{\sigma }_{A = a}T}\right)$执行

				If $\left( {b,c}\right)  \in  S$ then

				如果 $\left( {b,c}\right)  \in  S$，那么

					Add(a,b,c)to ${Q}_{\Delta }$

					将 Add(a,b,c) 添加到 ${Q}_{\Delta }$ 中

	Return $Q$

	返回 $Q$

---

<!-- Media -->

If $\left| R\right|  = \left| S\right|  = \left| T\right|  = N$ ,then the above is $O\left( {N}^{3/2}\right)$ as claimed in the introduction. We will generalize the above algorithm beyond triangles to general join queries in Section 4. Before that, we present a second algorithm that has exactly the same worst-case run-time and a similar analysis to illustrate the recursive structure of the generic worst-case join algorithm described in Section 4.

如果 $\left| R\right|  = \left| S\right|  = \left| T\right|  = N$，那么上述内容即为引言中所声称的 $O\left( {N}^{3/2}\right)$。我们将在第 4 节把上述算法从三角形推广到一般的连接查询。在此之前，我们将介绍第二种算法，该算法具有完全相同的最坏情况运行时间和相似的分析，以说明第 4 节中描述的通用最坏情况连接算法的递归结构。

### 2.3 Algorithm 2: Delaying the Computation

### 2.3 算法 2：延迟计算

Now we present a second way to compute ${Q}_{ \land  }\left\lbrack  {a}_{i}\right\rbrack$ that differentiates between heavy and light values ${a}_{i} \in  A$ in a different way. We don’t try to estimate the heaviness of ${a}_{i}$ right off the bat. Algorithm 2 "looks deeper" into what pair(b,c)can go along with ${a}_{i}$ in the output by computing $c$ for each candidate $b$ .

现在，我们介绍第二种计算 ${Q}_{ \land  }\left\lbrack  {a}_{i}\right\rbrack$ 的方法，该方法以不同的方式区分重值和轻值 ${a}_{i} \in  A$。我们不会立即尝试估计 ${a}_{i}$ 的“重”程度。算法 2 通过为每个候选 $b$ 计算 $c$，“更深入地”探究哪些对 (b,c) 可以与 ${a}_{i}$ 一起出现在输出中。

Algorithm 2 works as follows. By computing the intersection ${\pi }_{B}\left( {{\sigma }_{A = {a}_{i}}R}\right)  \cap  {\pi }_{B}S$ ,we only look at the candidates $b$ that can possibly participate with ${a}_{i}$ in the output $\left( {{a}_{i},b,c}\right)$ . Then,the candidate set for $c$ is ${\pi }_{C}\left( {{\sigma }_{B = b}S}\right)  \cap  {\pi }_{C}\left( {{\sigma }_{A = {a}_{i}}T}\right)$ . When ${a}_{i}$ is really skewed toward the heavy side,the candidates $b$ and then $c$ help gradually reduce the skew toward building up the final solution ${Q}_{\bigtriangleup }$ .

算法 2 的工作方式如下。通过计算交集 ${\pi }_{B}\left( {{\sigma }_{A = {a}_{i}}R}\right)  \cap  {\pi }_{B}S$，我们只考虑那些可能与 ${a}_{i}$ 一起出现在输出 $\left( {{a}_{i},b,c}\right)$ 中的候选 $b$。然后，$c$ 的候选集为 ${\pi }_{C}\left( {{\sigma }_{B = b}S}\right)  \cap  {\pi }_{C}\left( {{\sigma }_{A = {a}_{i}}T}\right)$。当 ${a}_{i}$ 确实偏向重的一侧时，候选 $b$ 以及随后的 $c$ 有助于逐步减少在构建最终解 ${Q}_{\bigtriangleup }$ 时的偏差。

Example 2. Let us now see how delaying computation works on the bad example. As we have observed in using the power of two choices, computing the intersection of two sorted sets takes time at most the minimum of the two sizes.

示例 2。现在让我们看看延迟计算在这个糟糕的示例中是如何工作的。正如我们在利用二选一的能力时所观察到的，计算两个有序集合的交集所需的时间最多为两个集合大小的最小值。

<!-- Media -->

Algorithm 2 Computing ${Q}_{ \bigtriangleup  }$ by delaying computation.

算法 2 通过延迟计算来计算 ${Q}_{ \bigtriangleup  }$。

---

Input: $R\left( {A,B}\right) ,S\left( {B,C}\right) ,T\left( {A,C}\right)$ in sorted order

输入：按排序顺序排列的 $R\left( {A,B}\right) ,S\left( {B,C}\right) ,T\left( {A,C}\right)$

	$Q \leftarrow  \varnothing$

	${L}_{A} \leftarrow  {\pi }_{A}R \cap  {\pi }_{A}T$

	For each $a \in  {L}_{A}$ do

	对于每个 $a \in  {L}_{A}$ 执行

		${L}_{B}^{a} \leftarrow  {\pi }_{B}{\sigma }_{A = a}R \cap  {\pi }_{B}S$

		For each $b \in  {L}_{B}^{a}$ do

		对于每个 $b \in  {L}_{B}^{a}$ 执行

			${L}_{C}^{a,b} \leftarrow  {\pi }_{C}{\sigma }_{B = b}S \cap  {\pi }_{C}{\sigma }_{A = a}T$

			For each $c \in  {L}_{C}^{a,b}$ do

			对于每个 $c \in  {L}_{C}^{a,b}$ 执行

				Add(a,b,c)to $Q$

				将 Add(a,b,c) 添加到 $Q$ 中

	Return $Q$

	返回 $Q$

---

<!-- Media -->

For ${a}_{0}$ ,we consider all $b \in  \left\{  {{b}_{0},{b}_{1},\ldots ,{b}_{m}}\right\}$ . When $b = {b}_{0}$ ,we have

对于 ${a}_{0}$，我们考虑所有的 $b \in  \left\{  {{b}_{0},{b}_{1},\ldots ,{b}_{m}}\right\}$。当 $b = {b}_{0}$ 时，我们有

$$
{\pi }_{C}\left( {{\sigma }_{B = {b}_{0}}S}\right)  = {\pi }_{C}\left( {{\sigma }_{A = {a}_{0}}T}\right)  = \left\{  {{c}_{0},\ldots ,{c}_{m}}\right\}  ,
$$

so we output the $m + 1$ triangles in total time $O\left( m\right)$ . For the pairs $\left( {{a}_{0},{b}_{i}}\right)$ when $i \geq  1$ ,we have $\left| {{\sigma }_{B = {b}_{i}}S}\right|  = 1$ and hence we spend $O\left( 1\right)$ time on each such pair,for a total of $O\left( m\right)$ overall.

因此，我们在总时间 $O\left( m\right)$ 内输出了 $m + 1$ 个三角形。对于 $\left( {{a}_{0},{b}_{i}}\right)$ 这样的对，当 $i \geq  1$ 时，我们有 $\left| {{\sigma }_{B = {b}_{i}}S}\right|  = 1$，因此在每一对这样的对上花费 $O\left( 1\right)$ 的时间，总体上总共花费 $O\left( m\right)$ 的时间。

Now consider ${a}_{i}$ for $i \geq  1$ . In this case, $b = {b}_{0}$ is the only candidate. Further,for $\left( {{a}_{i},{b}_{0}}\right)$ ,we have $\left| {{\sigma }_{A = {a}_{i}}T}\right|  = 1$ ,so we can handle each such ${a}_{i}$ in $O\left( 1\right)$ time leading to an overall run time of $O\left( m\right)$ . Thus on this bad example Algorithm 2 runs in $O\left( N\right)$ time.

现在考虑当 $i \geq  1$ 时的 ${a}_{i}$。在这种情况下，$b = {b}_{0}$ 是唯一的候选。此外，对于 $\left( {{a}_{i},{b}_{0}}\right)$，我们有 $\left| {{\sigma }_{A = {a}_{i}}T}\right|  = 1$，所以我们可以在 $O\left( 1\right)$ 的时间内处理每一个这样的 ${a}_{i}$，从而导致总的运行时间为 $O\left( m\right)$。因此，在这个糟糕的例子中，算法 2 的运行时间为 $O\left( N\right)$。

Appendix B has the full analysis of Algorithm 2: its worst-case runtime is exactly the same as that of Algorithm 1. What is remarkable is that both of these algorithms follow exactly the same recursive structure and they are special cases of a generic worst-case optimal join algorithm.

附录 B 对算法 2 进行了全面分析：其最坏情况下的运行时间与算法 1 完全相同。值得注意的是，这两种算法遵循完全相同的递归结构，它们是通用最坏情况最优连接算法的特殊情况。

## 3 A User's Guide to the AGM bound

## 3 AGM 界的用户指南

We now describe one way to generalize the bound of the output size of a join (mirroring the $O\left( {N}^{3/2}\right)$ bound we saw for the triangle query) and illustrate its use with a few examples.

我们现在描述一种推广连接输出大小界的方法（类似于我们在三角形查询中看到的 $O\left( {N}^{3/2}\right)$ 界），并通过几个例子来说明其用法。

### 3.1 AGM Bound

### 3.1 AGM 界

To state the AGM bound, we need some notation. The natural join problem can be defined as follows. We are given a collection of $m$ relations. Each relation is over a collection of attributes. We use $\mathcal{V}$ to denote the set of attributes; let $n = \left| \mathcal{V}\right|$ . The join query $Q$ is modeled as a hypergraph $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$ ,where for each hyperedge $F \in  \mathcal{E}$ there is a relation ${R}_{F}$ on attribute set $F$ . Figure 3 shows several example join queries,their associated hypergraphs, and illustrates the bounds below.

为了陈述 AGM 界，我们需要一些符号。自然连接问题可以定义如下。我们给定了 $m$ 个关系的集合。每个关系都基于一组属性。我们用 $\mathcal{V}$ 表示属性集；设 $n = \left| \mathcal{V}\right|$。连接查询 $Q$ 被建模为一个超图 $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$，其中对于每条超边 $F \in  \mathcal{E}$，在属性集 $F$ 上都有一个关系 ${R}_{F}$。图 3 展示了几个连接查询的例子、它们相关的超图，并说明了下面的界。

Atserias-Grohe-Marx [3] and Grohe-Marx [23] proved the following remarkable inequality, which shall be referred to as the AGM’s inequality henceforth. Let $\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$ be any point in the following polyhe-

阿塞里亚斯 - 格罗赫 - 马克思（Atserias - Grohe - Marx）[3] 和格罗赫 - 马克思（Grohe - Marx）[23] 证明了以下显著的不等式，此后我们将其称为 AGM 不等式。设 $\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$ 是以下多面体中的任意一点：

dron:

多面体

$$
\left\{  {\mathbf{x} \mid  \mathop{\sum }\limits_{{F : v \in  F}}{x}_{F} \geq  1,\forall v \in  \mathcal{V},\mathbf{x} \geq  \mathbf{0}}\right\}  .
$$

Such a point $\mathbf{x}$ is called a fractional edge cover of the hypergraph $\mathcal{H}$ . Then,AGM’s inequality states that the

这样的点 $\mathbf{x}$ 被称为超图 $\mathcal{H}$ 的分数边覆盖。那么，AGM 不等式表明

join size can be bounded by

连接大小可以被界定为

$$
\left| Q\right|  = \left| {{ \bowtie  }_{F \in  \mathcal{E}}{R}_{F}}\right|  \leq  \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}. \tag{6}
$$

<!-- Media -->

<!-- figureText: $\left( {A}_{1}\right.$ ${R}_{1}$ ${R}_{-3}$ $\left( {A}_{2}\right.$ ${R}_{-1}$ ${R}_{2,3}$ ${R}_{2,4}$ ${R}_{-2}$ ${R}_{3,4}$ ${A}_{3}$ $\left( {A}_{4}\right.$ ${R}_{1,4}$ ${x}_{{R}_{-i}} = \frac{1}{3}\forall i$ ${x}_{{R}_{i,j}} = \frac{1}{3}\forall \left( {i,j}\right)$ ${x}_{{R}_{-1}} = {x}_{{R}_{-2}} = 1$ $\begin{matrix} {x}_{{R}_{1,4}} = {x}_{{R}_{2,3}} = 1 \\  {\mathbf{K}}_{4} \end{matrix}$ ${\mathrm{{LW}}}_{4}$ ${x}_{T} = \frac{1}{2}$ ${A}_{3}$ B $S$ (c ${x}_{S} + {\grave{x}}_{T} = 1$ ${x}_{S} + {x}_{T} = 1$ ${\mathbf{Q}}_{ \bigtriangleup  }$ -->

<img src="https://cdn.noedgeai.com/0195cc9c-3871-780e-84a7-1cc8384f8f6e_7.jpg?x=452&y=205&w=887&h=348&r=0"/>

Figure 3: A handful of queries and their covers.

图 3：一些查询及其覆盖。

<!-- Media -->

### 3.2 Example Bounds

### 3.2 示例界

We now illustrate the AGM bound on some specific join queries. We begin with the triangle query ${Q}_{\Delta }$ . In this case the corresponding hypergraph $\mathcal{H}$ is as in the left part of Figure 3 . We consider two covers (which are also marked in Figure 3). The first one is ${x}_{R} = {x}_{T} = {x}_{S} = \frac{1}{2}$ . This is a valid cover since the required inequalities are satisfied for every vertex. For example,for vertex $C$ ,the two edges incident on it are $S$ and $T$ and we have ${x}_{S} + {x}_{T} = 1 \geq  1$ as required. In this case the bound (6) states that

我们现在通过一些特定的连接查询来说明 AGM 界。我们从三角形查询 ${Q}_{\Delta }$ 开始。在这种情况下，对应的超图 $\mathcal{H}$ 如图 3 的左半部分所示。我们考虑两种覆盖（在图 3 中也有标记）。第一种是 ${x}_{R} = {x}_{T} = {x}_{S} = \frac{1}{2}$。这是一个有效的覆盖，因为对于每个顶点，所需的不等式都得到满足。例如，对于顶点 $C$，与其关联的两条边是 $S$ 和 $T$，并且我们有 ${x}_{S} + {x}_{T} = 1 \geq  1$，满足要求。在这种情况下，界 (6) 表明

$$
\left| {Q}_{\Delta }\right|  \leq  \sqrt{\left| R\right|  \cdot  \left| S\right|  \cdot  \left| T\right| }. \tag{7}
$$

Another valid cover is ${x}_{R} = {x}_{T} = 1$ and ${x}_{S} = 0$ (this cover is also marked in Figure 3). This is a valid cover, e.g. since for $C$ we have ${x}_{S} + {x}_{T} = 1 \geq  1$ and for vertex $A$ ,we have ${x}_{R} + {x}_{T} = 2 \geq  1$ as required. For this cover, bound (6) gives

另一个有效的覆盖是${x}_{R} = {x}_{T} = 1$和${x}_{S} = 0$（此覆盖也在图3中标记）。这是一个有效的覆盖，例如，对于$C$，我们有${x}_{S} + {x}_{T} = 1 \geq  1$，对于顶点$A$，我们有满足要求的${x}_{R} + {x}_{T} = 2 \geq  1$。对于这个覆盖，边界条件(6)给出

$$
\left| {Q}_{ \bigtriangleup  }\right|  \leq  \left| R\right|  \cdot  \left| T\right| . \tag{8}
$$

These two bounds can be better in different scenarios. E.g. when $\left| R\right|  = \left| S\right|  = \left| T\right|  = N$ ,then (7) gives an upper bound of ${N}^{3/2}$ (which is the tight answer) while ⑧ gives a bound of ${N}^{2}$ ,which is worse. However,if $\left| R\right|  = \left| T\right|  = 1$ and $\left| S\right|  = N$ ,then (7) gives a bound of $\sqrt{N}$ ,which has a lot of slack; while [8] gives a bound of 1 , which is tight.

这两个边界条件在不同场景下可能更优。例如，当$\left| R\right|  = \left| S\right|  = \left| T\right|  = N$时，那么(7)给出的上界为${N}^{3/2}$（这是精确答案），而⑧给出的边界为${N}^{2}$，这个结果更差。然而，如果$\left| R\right|  = \left| T\right|  = 1$且$\left| S\right|  = N$，那么(7)给出的边界为$\sqrt{N}$，这个边界有很大的松弛度；而[8]给出的边界为1，这是精确的。

For another class of examples,consider the "clique" query. In this case there are $n \geq  3$ attributes and $m = \left( \begin{array}{l} n \\  2 \end{array}\right)$ relations: one ${R}_{i,j}$ for every $i < j \in  \left\lbrack  n\right\rbrack$ : we will call this query ${K}_{n}$ . Note that ${K}_{3}$ is ${Q}_{\Delta }$ . The middle part of Figure 3 considers the ${K}_{4}$ query. We highlight one cover: ${x}_{{R}_{i,j}} = \frac{1}{n - 1}$ for every $i < j \in  \left\lbrack  n\right\rbrack$ . This is a valid cover since every attribute is contained in $n - 1$ relations. Further,in this case (6) gives a bound of $\sqrt[{n - 1}]{\mathop{\prod }\limits_{{i < j}}\left| {R}_{i,j}\right| }$ ,which simplifies to ${N}^{n/2}$ for the case when every relation has size $N$ .

对于另一类示例，考虑“团”查询。在这种情况下，有$n \geq  3$个属性和$m = \left( \begin{array}{l} n \\  2 \end{array}\right)$个关系：对于每个$i < j \in  \left\lbrack  n\right\rbrack$都有一个${R}_{i,j}$：我们将这个查询称为${K}_{n}$。注意，${K}_{3}$是${Q}_{\Delta }$。图3的中间部分考虑了${K}_{4}$查询。我们突出显示一个覆盖：对于每个$i < j \in  \left\lbrack  n\right\rbrack$都有${x}_{{R}_{i,j}} = \frac{1}{n - 1}$。这是一个有效的覆盖，因为每个属性都包含在$n - 1$个关系中。此外，在这种情况下，(6)给出的边界为$\sqrt[{n - 1}]{\mathop{\prod }\limits_{{i < j}}\left| {R}_{i,j}\right| }$，当每个关系的大小为$N$时，该边界简化为${N}^{n/2}$。

Finally,we consider the Loomis-Whitney $L{W}_{n}$ queries. In this case there are $n$ attributes and there are $m = n$ relations. In particular,for every $i \in  \left\lbrack  n\right\rbrack$ there is a relation ${R}_{-i} = {R}_{\left\lbrack  n\right\rbrack  \smallsetminus \{ i\} }$ . Note that $L{W}_{3}$ is ${Q}_{ \vartriangle  }$ . See the right of Figure 3 for the $L{W}_{4}$ query. We highlight one cover: ${x}_{{R}_{i,j}} = \frac{1}{n - 1}$ for every $i < j \in  \left\lbrack  n\right\rbrack$ . This is a valid cover since every attribute is contained in $n - 1$ relations. Further,in this case (6) gives a bound of $\sqrt[{n - 1}]{\mathop{\prod }\limits_{i}\left| {R}_{-i}\right| }$ ,which simplifies to ${N}^{1 + \frac{1}{n - 1}}$ for the case when every relation has size $N$ . Note that this bound approaches $N$ as $n$ becomes larger.

最后，我们考虑卢米斯 - 惠特尼（Loomis - Whitney）$L{W}_{n}$查询。在这种情况下，有$n$个属性和$m = n$个关系。特别地，对于每个$i \in  \left\lbrack  n\right\rbrack$，都存在一个关系${R}_{-i} = {R}_{\left\lbrack  n\right\rbrack  \smallsetminus \{ i\} }$。注意，$L{W}_{3}$是${Q}_{ \vartriangle  }$。有关$L{W}_{4}$查询，请参见图3的右侧。我们突出显示一个覆盖：对于每个$i < j \in  \left\lbrack  n\right\rbrack$，有${x}_{{R}_{i,j}} = \frac{1}{n - 1}$。这是一个有效的覆盖，因为每个属性都包含在$n - 1$个关系中。此外，在这种情况下，公式(6)给出了一个界$\sqrt[{n - 1}]{\mathop{\prod }\limits_{i}\left| {R}_{-i}\right| }$，当每个关系的大小为$N$时，该界可简化为${N}^{1 + \frac{1}{n - 1}}$。注意，当$n$变得更大时，这个界趋近于$N$。

### 3.3 The Tightest AGM Bound

### 3.3 最紧的AGM界

As we just saw, the optimal edge cover for the AGM bound depends on the relation sizes. To minimize the right hand side of [6], we can solve the following linear program:

正如我们刚刚看到的，AGM界的最优边覆盖取决于关系的大小。为了最小化[6]式的右侧，我们可以求解以下线性规划问题：

$$
\min \mathop{\sum }\limits_{{F \in  \mathcal{E}}}\left( {{\log }_{2}\left| {R}_{F}\right| }\right)  \cdot  {x}_{F}
$$

$$
\text{s.t.}\;\mathop{\sum }\limits_{{F : v \in  F}}{x}_{F} \geq  1,v \in  \mathcal{V}
$$

$$
\mathbf{x} \geq  \mathbf{0}
$$

Implicitly,the objective function above depends on the database instance $\mathcal{D}$ on which the query is applied. Let ${\rho }^{ * }\left( {Q,\mathcal{D}}\right)$ denote the optimal objective value to the above linear program. We refer to ${\rho }^{ * }\left( {Q,\mathcal{D}}\right)$ as the fractional edge cover number of the query $Q$ with respect to the database instance $\mathcal{D}$ ,following Grohe [22]. The AGM’s inequality can be summarized simply by $\left| Q\right|  \leq  {2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) }$ .

隐式地，上述目标函数依赖于应用查询的数据库实例$\mathcal{D}$。令${\rho }^{ * }\left( {Q,\mathcal{D}}\right)$表示上述线性规划的最优目标值。按照格罗赫（Grohe）[22]的说法，我们将${\rho }^{ * }\left( {Q,\mathcal{D}}\right)$称为查询$Q$相对于数据库实例$\mathcal{D}$的分数边覆盖数。AGM不等式可以简单地总结为$\left| Q\right|  \leq  {2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) }$。

### 3.4 Applying AGM bound on conjunctive queries with simple functional dependencies

### 3.4 在具有简单函数依赖的合取查询上应用AGM界

Thus far we have been describing bounds and algorithms for natural join queries. A super-class of natural join queries called conjunctive queries. A conjunctive query is a query of the form

到目前为止，我们一直在描述自然连接查询的界和算法。自然连接查询的一个超类称为合取查询。合取查询是如下形式的查询

$$
C = {R}_{0}\left( {\bar{X}}_{0}\right)  \leftarrow  {R}_{1}\left( {\bar{X}}_{1}\right)  \land  \cdots  \land  {R}_{m}\left( {\bar{X}}_{m}\right) 
$$

where $\left\{  {{R}_{1},\ldots ,{R}_{m}}\right\}$ is a multi-set of relation symbols,i.e. some relation might occur more than once in the query,and ${\bar{X}}_{0},\ldots ,{\bar{X}}_{m}$ are tuples of variables,and each variable occurring in the query’s head $R\left( {\bar{X}}_{0}\right)$ must also occur in the body. It is important to note that the same variable might occur more than once in the same tuple ${\bar{X}}_{i}$ .

其中$\left\{  {{R}_{1},\ldots ,{R}_{m}}\right\}$是关系符号的多重集，即某个关系可能在查询中出现多次，并且${\bar{X}}_{0},\ldots ,{\bar{X}}_{m}$是变量元组，查询头部$R\left( {\bar{X}}_{0}\right)$中出现的每个变量也必须出现在查询体中。需要注意的是，同一个变量可能在同一个元组${\bar{X}}_{i}$中出现多次。

We will use vars(C)to denote the set of all variables occurring in $C$ . Note that ${\bar{X}}_{0} \subseteq  \operatorname{vars}\left( C\right)$ and it is entirely possible for ${\bar{X}}_{0}$ to be empty (Boolean conjunctive query). For example,the following are conjunctive queries:

我们将使用vars(C)来表示在$C$中出现的所有变量的集合。注意，${\bar{X}}_{0} \subseteq  \operatorname{vars}\left( C\right)$，并且${\bar{X}}_{0}$完全有可能为空（布尔合取查询）。例如，以下是合取查询：

$$
{R}_{0}\left( {WXYZ}\right)  \leftarrow  S\left( {WXY}\right)  \land  S\left( {WWW}\right)  \land  T\left( {YZ}\right) 
$$

$$
{R}_{0}\left( Z\right)  \leftarrow  S\left( {WXY}\right)  \land  S\left( {WWW}\right)  \land  T\left( {YZ}\right) .
$$

The former query is a full conjunctive query because the head atom contains all the query's variable.

前一个查询是完全合取查询，因为头部原子包含了查询的所有变量。

Following Gottlob, Lee, Valiant, and Valiant (GLVV hence forth) [15, 16], we also know that the AGM bound can be extended to general conjunctive queries even with simple functional dependencies. ${}^{4}$ In this survey, our presentation closely follows Grohe's presentation of GLVV [22].

按照戈特洛布（Gottlob）、李（Lee）、瓦利安特（Valiant）和瓦利安特（Valiant）（以下简称GLVV）[15, 16]的研究，我们还知道，即使存在简单的函数依赖，AGM界也可以扩展到一般的合取查询。${}^{4}$在本综述中，我们的表述紧密遵循格罗赫（Grohe）对GLVV的表述[22]。

To illustrate what can go "wrong" when we are moving from natural join queries to conjunctive queries, let us first consider a few example conjunctive queries, introducing one issue at a time. In all examples below,relations are assumed to have the same size $N$ .

为了说明从自然连接查询过渡到合取查询时可能出现的“问题”，让我们首先考虑几个合取查询的例子，一次引入一个问题。在以下所有例子中，假设关系的大小都为$N$。

Example 3 (Projection). Consider

示例3（投影）。考虑

$$
{C}_{1} = {R}_{0}\left( W\right)  \leftarrow  R\left( {WX}\right)  \land  S\left( {WY}\right)  \land  T\left( {WZ}\right) .
$$

In the (natural) join query, $R\left( {WX}\right)  \land  S\left( {WY}\right)  \land  T\left( {WZ}\right)$ AGM bound gives ${N}^{3}$ ; but because ${R}_{0}\left( W\right)  \subseteq  {\pi }_{W}\left( R\right)  \bowtie$ ${\pi }_{W}\left( S\right)  \bowtie  {\pi }_{W}\left( T\right)$ ,AGM bound can be adapted to the instance restricted only to the output variables yielding an upper bound of $N$ on the output size.

在（自然）连接查询中，$R\left( {WX}\right)  \land  S\left( {WY}\right)  \land  T\left( {WZ}\right)$ AGM界给出${N}^{3}$；但由于${R}_{0}\left( W\right)  \subseteq  {\pi }_{W}\left( R\right)  \bowtie$ ${\pi }_{W}\left( S\right)  \bowtie  {\pi }_{W}\left( T\right)$，AGM界可以应用于仅限制在输出变量上的实例，从而得出输出规模的上界为$N$。

---

<!-- Footnote -->

${}^{4}$ GLVV also have fascinating bounds for the general functional dependency and composite keys cases,and characterization of treewidth-preserving queries; both of those topics are beyond the scope of this survey, in particular because they require different machinery from what we have developed thus far.

${}^{4}$ GLVV对于一般函数依赖和复合键的情况也有引人入胜的界，以及对保持树宽查询的刻画；这两个主题都超出了本综述的范围，特别是因为它们需要与我们目前所开发的不同的工具。

<!-- Footnote -->

---

Example 4 (Repeated variables). Consider the query

示例4（重复变量）。考虑查询

$$
{C}_{2} = {R}_{0}\left( {WY}\right)  \leftarrow  R\left( {WW}\right)  \land  S\left( {WY}\right)  \land  T\left( {YY}\right) .
$$

This is a full conjunctive query as all variables appear in the head atom ${R}_{0}$ . In this case,we can replace $R\left( {WW}\right)$ and $T\left( {YY}\right)$ by keeping only tuples $\left( {{t}_{1},{t}_{2}}\right)  \in  R$ for which ${t}_{1} = {t}_{2}$ and tuples $\left( {{t}_{1},{t}_{2}}\right)  \in  T$ for which ${t}_{1} = {t}_{2}$ ; essentially,we turn the query into a natural join query of the form ${R}^{\prime }\left( W\right)  \land  S\left( {WY}\right)  \land  {T}^{\prime }\left( Y\right)$ . For this query, ${x}_{{R}^{\prime }} = {x}_{{T}^{\prime }} = 0$ and ${x}_{S} = 1$ is a fractional cover and thus by AGM bound $N$ is an upperbound on the output size.

这是一个完全合取查询，因为所有变量都出现在头原子${R}_{0}$中。在这种情况下，我们可以通过仅保留满足${t}_{1} = {t}_{2}$的元组$\left( {{t}_{1},{t}_{2}}\right)  \in  R$和满足${t}_{1} = {t}_{2}$的元组$\left( {{t}_{1},{t}_{2}}\right)  \in  T$来替换$R\left( {WW}\right)$和$T\left( {YY}\right)$；本质上，我们将该查询转换为形式为${R}^{\prime }\left( W\right)  \land  S\left( {WY}\right)  \land  {T}^{\prime }\left( Y\right)$的自然连接查询。对于这个查询，${x}_{{R}^{\prime }} = {x}_{{T}^{\prime }} = 0$和${x}_{S} = 1$是一个分数覆盖，因此根据AGM界，$N$是输出规模的一个上界。

Example 5 (Introducing the chase). Consider the query

示例5（引入追逐法）。考虑查询

$$
{C}_{3} = {R}_{0}\left( {WXY}\right)  \leftarrow  R\left( {WX}\right)  \land  R\left( {WW}\right)  \land  S\left( {XY}\right) .
$$

Without additional information,the best bound we can get for this query is $O\left( {N}^{2}\right)$ : we can easily turn it into a natural join query of the form $R\left( {WX}\right)  \land  {R}^{\prime }\left( W\right)  \land  S\left( {XY}\right)$ ,where ${R}^{\prime }$ is obtained from $R$ by keeping all tuples $\left( {{t}_{1},{t}_{2}}\right)  \in  R$ for which ${t}_{1} = {t}_{2}$ . Then, $\left( {{x}_{R},{x}_{{R}^{\prime }},{x}_{S}}\right)$ is a fractional edge cover for this query if and only if ${x}_{R} + {x}_{{R}^{\prime }} \geq  1$ (to cover $W$ ), ${x}_{R} + {x}_{S} \geq  1$ (to cover $X$ ), ${x}_{S} \geq  1$ (to cover $Y$ ); So, ${x}_{S} = {x}_{{R}^{\prime }} = 1$ and ${x}_{R} = 0$ is a fractional cover,yielding the $O\left( {N}^{2}\right)$ bound. Furthermore,it is easy to construct input instances for which the output size is $\Omega \left( {N}^{2}\right)$ :

在没有额外信息的情况下，我们能为这个查询得到的最佳界是$O\left( {N}^{2}\right)$：我们可以轻松地将其转换为形式为$R\left( {WX}\right)  \land  {R}^{\prime }\left( W\right)  \land  S\left( {XY}\right)$的自然连接查询，其中${R}^{\prime }$是通过保留$R$中所有满足${t}_{1} = {t}_{2}$的元组$\left( {{t}_{1},{t}_{2}}\right)  \in  R$得到的。那么，$\left( {{x}_{R},{x}_{{R}^{\prime }},{x}_{S}}\right)$是这个查询的一个分数边覆盖，当且仅当${x}_{R} + {x}_{{R}^{\prime }} \geq  1$（以覆盖$W$），${x}_{R} + {x}_{S} \geq  1$（以覆盖$X$），${x}_{S} \geq  1$（以覆盖$Y$）；所以，${x}_{S} = {x}_{{R}^{\prime }} = 1$和${x}_{R} = 0$是一个分数覆盖，得出$O\left( {N}^{2}\right)$界。此外，很容易构造输出规模为$\Omega \left( {N}^{2}\right)$的输入实例：

$$
R = \{ \left( {i,i}\right)  \mid  i \in  \left\lbrack  {N/2}\right\rbrack  \} \bigcup \{ \left( {i,0}\right)  \mid  i \in  \left\lbrack  {N/2}\right\rbrack  \} 
$$

$$
S = \{ \left( {0,j}\right)  \mid  j \in  \left\lbrack  N\right\rbrack  \} .
$$

Every tuple(i,0,j)for $i \in  \left\lbrack  {N/2}\right\rbrack  ,j \in  \left\lbrack  N\right\rbrack$ is in the output.

对于$i \in  \left\lbrack  {N/2}\right\rbrack  ,j \in  \left\lbrack  N\right\rbrack$的每个元组(i,0,j)都在输出中。

Next,suppose we have an additional piece of information that the first attribute in relation $R$ is its key, i.e. if $\left( {{t}_{1},{t}_{2}}\right)$ and $\left( {{t}_{1},{t}_{2}^{\prime }}\right)$ are in $R$ ,then ${t}_{2} = {t}_{2}^{\prime }$ . Then we can significantly reduce the output size bound because we can infer the following about the output tuples:(w,x,y)is an output tuple iff(w,x)and(w,w) are in $R$ ,and(x,y)are in $S$ . The functional dependency tells us that $x = w$ . Hence,the query is equivalent to

接下来，假设我们有一条额外的信息，即关系 $R$ 中的第一个属性是其键，也就是说，如果 $\left( {{t}_{1},{t}_{2}}\right)$ 和 $\left( {{t}_{1},{t}_{2}^{\prime }}\right)$ 在 $R$ 中，那么 ${t}_{2} = {t}_{2}^{\prime }$ 。然后我们可以显著降低输出规模的上限，因为我们可以对输出元组做出如下推断：(w, x, y) 是一个输出元组，当且仅当 (w, x) 和 (w, w) 在 $R$ 中，并且 (x, y) 在 $S$ 中。函数依赖告诉我们 $x = w$ 。因此，该查询等价于

$$
{C}_{3}^{\prime } = {R}_{0}\left( {WY}\right)  \leftarrow  R\left( {WW}\right)  \land  S\left( {WY}\right) .
$$

The AGM bound for this (natural) join query is $N$ . The transformation from ${C}_{3}$ to ${C}_{3}^{\prime }$ we just described is, of course, the famous chase operation [2, 4, 30], which is much more powerful than what conveyed by this example.

此（自然）连接查询的 AGM 上限是 $N$ 。我们刚刚描述的从 ${C}_{3}$ 到 ${C}_{3}^{\prime }$ 的转换，当然就是著名的追赶操作 [2, 4, 30]，它比这个例子所传达的功能要强大得多。

Example 6 (Taking advantage of FDs). Consider the following query

示例 6（利用函数依赖）。考虑以下查询

$$
{C}_{4} = {R}_{0}\left( {X{Y}_{1},\ldots ,{Y}_{k},Z}\right)  \leftarrow  \mathop{\bigwedge }\limits_{{i = 1}}^{k}{R}_{i}\left( {X{Y}_{i}}\right)  \land  \mathop{\bigwedge }\limits_{{i = 1}}^{k}{S}_{i}\left( {{Y}_{i}Z}\right) .
$$

First,without any functional dependency,AGM bound gives ${N}^{k}$ for this query,because the fractional cover constraints are

首先，在没有任何函数依赖的情况下，AGM 上限为此查询给出 ${N}^{k}$ ，因为分数覆盖约束为

$$
\mathop{\sum }\limits_{{i = 1}}^{k}{x}_{{R}_{i}} \geq  1\left( {\operatorname{cover}X}\right) 
$$

$$
{x}_{{R}_{i}} + {x}_{{S}_{i}} \geq  1\left( {\operatorname{cover}{Y}_{i}}\right) i \in  \left\lbrack  k\right\rbrack  
$$

$$
\mathop{\sum }\limits_{{i = 1}}^{k}{x}_{{S}_{i}} \geq  1\left( {\operatorname{cover}Z}\right) 
$$

The AGM bound is ${N}^{\mathop{\sum }\limits_{i}\left( {{x}_{{R}_{i}} + {x}_{{S}_{i}}}\right) } \geq  {N}^{k}$ .

AGM 上限是 ${N}^{\mathop{\sum }\limits_{i}\left( {{x}_{{R}_{i}} + {x}_{{S}_{i}}}\right) } \geq  {N}^{k}$ 。

Second,suppose we know $k + 1$ functional dependencies: each of the first attributes of relations ${R}_{1},\ldots ,{R}_{k}$ is a key for the corresponding relation,and the first attribute of ${S}_{1}$ is its key. Then,we have the following sets of functional dependencies: $X \rightarrow  {Y}_{i},i \in  \left\lbrack  k\right\rbrack$ ,and ${Y}_{1} \rightarrow  Z$ . Now,construct a fictitious relation ${R}^{\prime }\left( {X,{Y}_{1},\ldots ,{Y}_{k},Z}\right)$ as follows: $\left( {x,{y}_{1},\ldots ,{y}_{k},z}\right)  \in  {R}^{\prime }$ iff $\left( {x,{y}_{i}}\right)  \in  {R}_{i}$ for all $i \in  \left\lbrack  k\right\rbrack$ and $\left( {{y}_{1},z}\right)  \in  {S}_{1}$ . Then,obviously $\left| {R}^{\prime }\right|  \leq  N$ . More importantly,the output does not change if we add ${R}^{\prime }$ to the body query ${C}_{4}$ to obtain a new conjunctive query ${C}_{4}^{\prime }$ . However,this time we can set ${x}_{{R}^{\prime }} = 1$ and all other variables in the fractional cover to be 0 and obtain an upper bound of $N$ .

其次，假设我们知道 $k + 1$ 个函数依赖：关系 ${R}_{1},\ldots ,{R}_{k}$ 的每个第一个属性都是对应关系的键，并且 ${S}_{1}$ 的第一个属性是其键。那么，我们有以下函数依赖集：$X \rightarrow  {Y}_{i},i \in  \left\lbrack  k\right\rbrack$ ，以及 ${Y}_{1} \rightarrow  Z$ 。现在，按如下方式构造一个虚拟关系 ${R}^{\prime }\left( {X,{Y}_{1},\ldots ,{Y}_{k},Z}\right)$ ：当且仅当对于所有 $i \in  \left\lbrack  k\right\rbrack$ 和 $\left( {{y}_{1},z}\right)  \in  {S}_{1}$ 有 $\left( {x,{y}_{i}}\right)  \in  {R}_{i}$ 时，$\left( {x,{y}_{1},\ldots ,{y}_{k},z}\right)  \in  {R}^{\prime }$ 成立。显然，$\left| {R}^{\prime }\right|  \leq  N$ 。更重要的是，如果我们将 ${R}^{\prime }$ 添加到主体查询 ${C}_{4}$ 中以获得一个新的合取查询 ${C}_{4}^{\prime }$ ，输出不会改变。然而，这次我们可以将 ${x}_{{R}^{\prime }} = 1$ 以及分数覆盖中的所有其他变量设为 0，并得到上限 $N$ 。

We present a more formal treatment of the steps needed to convert a conjunctive query with simple functional dependencies to a join query in Appendix E.

我们在附录 E 中对将具有简单函数依赖的合取查询转换为连接查询所需的步骤进行了更正式的处理。

## 4 Worst-case-optimal algorithms

## 4 最坏情况最优算法

We first show how to analyze the upper bound that proves AGM and from which we develop a generalized join algorithm that captures both algorithms from Ngo-Porat-Ré-Rudra [32] (henceforth NPRR) and Leapfrog Triejoin [42]. Then, we describe the limitation of any join-project plan.

我们首先展示如何分析证明 AGM 的上限，并由此开发一种广义连接算法，该算法涵盖了 Ngo - Porat - Ré - Rudra [32]（以下简称 NPRR）的算法和跳跃式字典树连接 [42] 算法。然后，我们描述任何连接 - 投影计划的局限性。

Henceforth,we need the following notation. Let $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$ be any hypergraph and $I \subseteq  \mathcal{V}$ be an arbitrary subset of vertices of $\mathcal{H}$ . Then,we define

此后，我们需要以下符号。设 $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$ 为任意超图，$I \subseteq  \mathcal{V}$ 为 $\mathcal{H}$ 的顶点的任意子集。那么，我们定义

$$
{\mathcal{E}}_{I} \mathrel{\text{:=}} \{ F \in  \mathcal{E} \mid  F \cap  I \neq  \varnothing \} .
$$

Example 7. For the query ${Q}_{ \bigtriangleup  }$ from Section 2,we have ${\mathcal{H}}_{ \bigtriangleup  } = \left( {{\mathcal{V}}_{ \bigtriangleup  },{\mathcal{E}}_{ \bigtriangleup  }}\right)$ ,where

示例 7。对于第 2 节中的查询 ${Q}_{ \bigtriangleup  }$ ，我们有 ${\mathcal{H}}_{ \bigtriangleup  } = \left( {{\mathcal{V}}_{ \bigtriangleup  },{\mathcal{E}}_{ \bigtriangleup  }}\right)$ ，其中

$$
{\mathcal{V}}_{\Delta } = \{ A,B,C\} 
$$

$$
{\mathcal{E}}_{ \bigtriangleup  } = \{ \{ A,B\} ,\{ B,C\} ,\{ A,C\} \} .
$$

Let ${I}_{1} = \{ A\}$ and ${I}_{2} = \{ A,B\}$ ,then ${\mathcal{E}}_{{I}_{1}} = \{ \{ A,B\} ,\{ A,C\} \}$ ,and ${\mathcal{E}}_{{I}_{2}} = {\mathcal{E}}_{\Delta }$ .

设 ${I}_{1} = \{ A\}$ 和 ${I}_{2} = \{ A,B\}$ ，则 ${\mathcal{E}}_{{I}_{1}} = \{ \{ A,B\} ,\{ A,C\} \}$ ，且 ${\mathcal{E}}_{{I}_{2}} = {\mathcal{E}}_{\Delta }$ 。

### 4.1 A proof of the AGM bound

### 4.1 算术 - 几何平均（AGM）界的证明

We prove the AGM inequality in two steps: a query decomposition lemma, and then a succinct inductive proof, which we then use to develop a generic worst-case optimal join algorithm.

我们分两步证明算术 - 几何平均（AGM）不等式：首先是一个查询分解引理，然后是一个简洁的归纳证明，之后我们将利用这些来开发一种通用的最坏情况最优连接算法。

#### 4.1.1 The query decomposition lemma

#### 4.1.1 查询分解引理

Ngo-Porat-Ré-Rudra [32] gave an inductive proof of AGM bound (6) using Hölder inequality. (AGM proved the bound using an entropy based argument: see Appendix D for more details.) The proof has an inductive structure leading naturally to recursive join algorithms. NPRR's strategy is a generalization of the strategy in [6] to prove the Bollobás-Thomason inequality, shown in [32] to be equivalent to AGM's bound.

恩戈（Ngo） - 波拉特（Porat） - 雷（Ré） - 鲁德拉（Rudra）[32] 使用赫尔德不等式（Hölder inequality）对算术 - 几何平均（AGM）界（6）进行了归纳证明。（算术 - 几何平均（AGM）使用基于熵的论证证明了该界：更多细节见附录 D。）该证明具有归纳结构，自然地引出了递归连接算法。恩戈 - 波拉特 - 雷 - 鲁德拉（NPRR）的策略是对 [6] 中证明博洛巴什 - 托马森不等式（Bollobás - Thomason inequality）的策略的推广，[32] 中表明该不等式等价于算术 - 几何平均（AGM）界。

Implicit in NPRR is the following key lemma, which will be crucial in proving bounds on general join queries (as well as proving upper bounds on the runtime of the new join algorithms).

恩戈 - 波拉特 - 雷 - 鲁德拉（NPRR）的证明中隐含着以下关键引理，该引理对于证明一般连接查询的界（以及证明新连接算法的运行时间上界）至关重要。

Lemma 4.1 (Query decomposition lemma). Let $Q = { \bowtie  }_{F \in  \mathcal{E}}{R}_{F}$ be a natural join query represented by a hypergraph $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$ ,and $\mathbf{x}$ be any fractional edge cover for $\mathcal{H}$ . Let $\mathcal{V} = I \uplus  J$ be an arbitrary partition of $\mathcal{V}$ such that $1 \leq  \left| I\right|  < \left| \mathcal{V}\right|$ ; and,

引理 4.1（查询分解引理）。设 $Q = { \bowtie  }_{F \in  \mathcal{E}}{R}_{F}$ 是由超图 $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$ 表示的自然连接查询，$\mathbf{x}$ 是 $\mathcal{H}$ 的任意分数边覆盖。设 $\mathcal{V} = I \uplus  J$ 是 $\mathcal{V}$ 的任意划分，使得 $1 \leq  \left| I\right|  < \left| \mathcal{V}\right|$ ；并且，

$$
L = { \bowtie  }_{F \in  {\mathcal{E}}_{I}}{\pi }_{I}\left( {R}_{F}\right) .
$$

Then,

那么，

$$
\mathop{\sum }\limits_{{{\mathbf{t}}_{I} \in  L}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}} \leq  \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}. \tag{9}
$$

Before we prove the lemma above, we outline how we have already used the lemma above specialized to ${Q}_{ \bigtriangleup  }$ in Section 2 to bound the runtime of Algorithm 1 . We use the lemma with $\mathbf{x} = \left( {1/2,1/2,1/2}\right)$ ,which is a valid fractional edge cover for ${\mathcal{H}}_{ \vartriangle  }$ .

在证明上述引理之前，我们先概述一下在第 2 节中如何针对 ${Q}_{ \bigtriangleup  }$ 专门使用上述引理来界定算法 1 的运行时间。我们使用 $\mathbf{x} = \left( {1/2,1/2,1/2}\right)$ 应用该引理，它是 ${\mathcal{H}}_{ \vartriangle  }$ 的一个有效分数边覆盖。

For Algorithm 1 we use Lemma 4.1 with $I = {I}_{1}$ . Note that $L$ in Lemma 4.1 is the same as

对于算法 1，我们使用引理 4.1 并结合 $I = {I}_{1}$ 。注意，引理 4.1 中的 $L$ 与

$$
{\pi }_{A}\left( R\right)  \bowtie  {\pi }_{A}\left( T\right)  = {\pi }_{A}\left( R\right)  \cap  {\pi }_{A}\left( T\right) ,
$$

i.e. this $L$ is exactly the same as the $L$ in Algorithm 1,We now consider the left hand side (LHS) in (9). Note that we have ${\mathcal{E}}_{J} = \{ \{ A,B\} ,\{ B,C\} ,\{ A,C\} \}$ . Thus,the LHS is the same as

即这个 $L$ 与算法 1 中的 $L$ 完全相同。现在我们考虑（9）式的左边（LHS）。注意到我们有 ${\mathcal{E}}_{J} = \{ \{ A,B\} ,\{ B,C\} ,\{ A,C\} \}$ 。因此，左边与

$$
\mathop{\sum }\limits_{{a \in  L}}\sqrt{\left| R \ltimes  \left( a\right) \right| } \cdot  \sqrt{\left| T \ltimes  \left( a\right) \right| } \cdot  \sqrt{\left| S \ltimes  \left( a\right) \right| }
$$

$$
 = \mathop{\sum }\limits_{{a \in  L}}\sqrt{\left| {\sigma }_{A = a}R\right| } \cdot  \sqrt{\left| {\sigma }_{A = a}T\right| } \cdot  \sqrt{\left| S\right| }.
$$

Note that the last expression is exactly the same as [4],which is at most $\sqrt{\left| R\right|  \cdot  \left| S\right|  \cdot  \left| T\right| }$ by Lemma 4.1. This was what shown in Section 2.

注意，最后一个表达式与 [4] 完全相同，根据引理 4.1，它至多为 $\sqrt{\left| R\right|  \cdot  \left| S\right|  \cdot  \left| T\right| }$ 。这就是第 2 节中所展示的内容。

Proof of Lemma 4.1. The plan is to "unroll" the sum of products on the left hand side using Hölder inequality as follows. Let $j \in  I$ be an arbitrary attribute. Define

引理 4.1 的证明。我们的计划是使用赫尔德不等式（Hölder inequality）按如下方式“展开”左边的乘积之和。设 $j \in  I$ 是任意属性。定义

$$
{I}^{\prime } = I - \{ j\} 
$$

$$
{J}^{\prime } = J \cup  \{ j\} 
$$

$$
{L}^{\prime } = { \bowtie  }_{F \in  {\mathcal{E}}_{{I}^{\prime }}}{\pi }_{{I}^{\prime }}\left( {R}_{F}\right) .
$$

We will show that

我们将证明

$$
\mathop{\sum }\limits_{{{\mathbf{t}}_{I} \in  L}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}} \leq  \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{{I}^{\prime }}\right| }^{{x}_{F}}. \tag{10}
$$

Then,by repeated applications of (10) we will bring ${I}^{\prime }$ down to empty and the right hand side is that of (9).

然后，通过反复应用（10）式，我们将把 ${I}^{\prime }$ 缩减为空集，此时右边就是（9）式的右边。

To prove (10) we write ${\mathbf{t}}_{I} = \left( {{\mathbf{t}}_{{I}^{\prime }},{t}_{j}}\right)$ for some ${\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }$ and decompose a sum over $L$ to a double sum over ${L}^{\prime }$ and ${t}_{j}$ ,where the second sum is only over ${t}_{j}$ for which $\left( {{\mathbf{t}}_{{I}^{\prime }},{t}_{j}}\right)  \in  L$ .

为了证明（10）式，我们将 ${\mathbf{t}}_{I} = \left( {{\mathbf{t}}_{{I}^{\prime }},{t}_{j}}\right)$ 写为某个 ${\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }$ 的形式，并将对 $L$ 的求和分解为对 ${L}^{\prime }$ 和 ${t}_{j}$ 的双重求和，其中第二个求和仅针对满足 $\left( {{\mathbf{t}}_{{I}^{\prime }},{t}_{j}}\right)  \in  L$ 的 ${t}_{j}$ 。

$$
\mathop{\sum }\limits_{{{\mathbf{t}}_{I} \in  L}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}} = \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\sum }\limits_{{t}_{j}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  \left( {\mathbf{t}}_{{I}^{\prime }},{t}_{j}\right) \right| }^{{x}_{F}}
$$

$$
 = \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\sum }\limits_{{t}_{j}}\left( {\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  \left( {\mathbf{t}}_{{I}^{\prime }},{t}_{j}\right) \right| }^{{x}_{F}}}\right)  \cdot  \left( {\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }} - {\mathcal{E}}_{J}}}{1}^{{x}_{F}}}\right) 
$$

$$
 = \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\sum }\limits_{{t}_{j}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }}}}{\left| {R}_{F} \ltimes  \left( {\mathbf{t}}_{{I}^{\prime }},{t}_{j}\right) \right| }^{{x}_{F}}
$$

$$
 = \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }} - {\mathcal{E}}_{\{ j\} }}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{{I}^{\prime }}\right| }^{{x}_{F}}\mathop{\sum }\limits_{{t}_{j}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{\{ j\} }}}{\left| {R}_{F} \ltimes  \left( {\mathbf{t}}_{{I}^{\prime }},{t}_{j}\right) \right| }^{{x}_{F}}
$$

$$
 \leq  \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }} - {\mathcal{E}}_{\{ j\} }}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{{I}^{\prime }}\right| }^{{x}_{F}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{\{ j\} }}}{\left( \mathop{\sum }\limits_{{t}_{j}}\left| {R}_{F} \ltimes  \left( {\mathbf{t}}_{{I}^{\prime }},{t}_{j}\right) \right| \right) }^{{x}_{F}}
$$

$$
 \leq  \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }} - {\mathcal{E}}_{\{ j\} }}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{{I}^{\prime }}\right| }^{{x}_{F}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{\{ j\} }}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{{I}^{\prime }}\right| }^{{x}_{F}}
$$

$$
 = \mathop{\sum }\limits_{{{\mathbf{t}}_{{I}^{\prime }} \in  {L}^{\prime }}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{{J}^{\prime }}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{{I}^{\prime }}\right| }^{{x}_{F}}.
$$

In the above,the third equality follows from fact that for any $F \in  {\mathcal{E}}_{{J}^{\prime }} - {\mathcal{E}}_{J}$ ,we have $F \subseteq  {I}^{\prime } \cup  \{ j\}$ . The first inequality is an application of Hölder inequality,which holds because $\mathop{\sum }\limits_{{F \in  {\mathcal{E}}_{\left( i\right) }}}{x}_{F} \geq  1$ . The second inequality is true because the sum is only over ${t}_{j}$ for which $\left( {{\mathbf{t}}_{{I}^{\prime }},{t}_{j}}\right)  \in  L$ .

在上述内容中，第三个等式源于这样一个事实：对于任意$F \in  {\mathcal{E}}_{{J}^{\prime }} - {\mathcal{E}}_{J}$，我们有$F \subseteq  {I}^{\prime } \cup  \{ j\}$。第一个不等式是赫尔德不等式（Hölder inequality）的应用，该不等式成立是因为$\mathop{\sum }\limits_{{F \in  {\mathcal{E}}_{\left( i\right) }}}{x}_{F} \geq  1$。第二个不等式成立是因为求和仅针对满足$\left( {{\mathbf{t}}_{{I}^{\prime }},{t}_{j}}\right)  \in  L$的${t}_{j}$进行。

It is quite remarkable that from the query decomposition lemma, we can prove AGM inequality (6), and describe and analyze two join algorithms succinctly.

值得注意的是，从查询分解引理出发，我们可以证明算术 - 几何平均不等式（AGM inequality）(6)，并且能够简洁地描述和分析两种连接算法。

#### 4.1.2 An inductive proof of AGM inequality

#### 4.1.2 算术 - 几何平均不等式的归纳证明

Base case. In the base case $\left| \mathcal{V}\right|  = 1$ ,we are computing the join of $\left| \mathcal{E}\right|$ unary relations. Let $\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$ be a fractional edge cover for this instance. Then,

基础情形。在基础情形$\left| \mathcal{V}\right|  = 1$中，我们正在计算$\left| \mathcal{E}\right|$个一元关系的连接。设$\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$是该实例的一个分数边覆盖。那么，

$$
\left| {{ \bowtie  }_{F \in  \mathcal{E}}{R}_{F}}\right|  \leq  \mathop{\min }\limits_{{F \in  \mathcal{E}}}\left| {R}_{F}\right| 
$$

$$
 \leq  {\left( \mathop{\min }\limits_{{F \in  \mathcal{E}}}\left| {R}_{F}\right| \right) }^{\mathop{\sum }\limits_{{F \in  \mathcal{E}}}{x}_{F}}
$$

$$
 = \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left( \mathop{\min }\limits_{{F \in  \mathcal{E}}}\left| {R}_{F}\right| \right) }^{{x}_{F}}
$$

$$
 \leq  \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}
$$

Inductive step. Now,assume $n = \left| \mathcal{V}\right|  \geq  2$ . Let $\mathcal{V} = I \uplus  J$ be any partition of $\mathcal{V}$ such that $1 \leq  \left| I\right|  < \left| \mathcal{V}\right|$ . Define $L = { \bowtie  }_{F \in  {\mathcal{E}}_{I}}{\pi }_{I}\left( {R}_{F}\right)$ as in Lemma 4.1. For each tuple ${\mathbf{t}}_{I} \in  L$ we define a new join query

归纳步骤。现在，假设$n = \left| \mathcal{V}\right|  \geq  2$。设$\mathcal{V} = I \uplus  J$是$\mathcal{V}$的任意一个划分，使得$1 \leq  \left| I\right|  < \left| \mathcal{V}\right|$。按照引理4.1的方式定义$L = { \bowtie  }_{F \in  {\mathcal{E}}_{I}}{\pi }_{I}\left( {R}_{F}\right)$。对于每个元组${\mathbf{t}}_{I} \in  L$，我们定义一个新的连接查询

$$
Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack   \mathrel{\text{:=}} { \bowtie  }_{F \in  {\mathcal{E}}_{J}}{\pi }_{J}\left( {{R}_{F} \ltimes  {\mathbf{t}}_{I}}\right) .
$$

Then,obviously we can write the original query $Q$ as

那么，显然我们可以将原始查询$Q$写为

$$
Q = \mathop{\bigcup }\limits_{{{\mathbf{t}}_{I} \in  L}}\left( {\left\{  {\mathbf{t}}_{I}\right\}   \times  Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack  }\right) . \tag{11}
$$

The vector ${\left( {x}_{F}\right) }_{F \in  {\mathcal{E}}_{J}}$ is a fractional edge cover for the hypergraph of $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ . Hence,the induction hypothesis

向量${\left( {x}_{F}\right) }_{F \in  {\mathcal{E}}_{J}}$是$Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$的超图的一个分数边覆盖。因此，根据归纳假设

gives us

我们得到

$$
\left| {Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack  }\right|  \leq  \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {\pi }_{J}\left( {R}_{F} \ltimes  {\mathbf{t}}_{I}\right) \right| }^{{x}_{F}} = \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}}. \tag{12}
$$

From (11), (12), and (9) we obtain AGM inequality:

由(11)、(12)和(9)，我们得到算术 - 几何平均不等式：

$$
\left| Q\right|  = \mathop{\sum }\limits_{{{\mathbf{t}}_{I} \in  L}}\left| {Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack  }\right|  \leq  \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}.
$$

### 4.2 Worst-case optimal join algorithms

### 4.2 最坏情况最优连接算法

From the proof of Lemma 4.1 and the query decomposition (11), it is straightforward to design a class of recursive join algorithms which is optimal in the worst case. (See Algorithm 3.)

从引理4.1的证明和查询分解(11)出发，很容易设计出一类在最坏情况下最优的递归连接算法。（见算法3。）

A mild assumption which is not very crucial is to pre-index all the relations so that the inputs to the sub-queries $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ can readily be available when the time comes to compute it. Both NPRR and Leapfrog Triejoin algorithms do this by fixing a global attribute order and build a B-tree-like index structure for each input relation consistent with this global attribute order. NPRR also described an hash-based indexing structure so as to remove a log-factor from the final run time. We will not delve on this point here, except to emphasize the fact that we do not include the linear time pre-processing step in the final runtime formula.

一个不太关键的温和假设是对所有关系进行预索引，以便在需要计算子查询$Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$时，其输入能够随时可用。NPRR算法和Leapfrog Triejoin算法都是通过固定一个全局属性顺序，并为每个输入关系构建一个类似于B树的索引结构来实现这一点的。NPRR算法还描述了一种基于哈希的索引结构，以便从最终运行时间中消除一个对数因子。我们在这里不会深入讨论这一点，只是强调我们在最终运行时间公式中不包括线性时间的预处理步骤这一事实。

<!-- Media -->

Algorithm 3 Generic-Join $\left( {{ \bowtie  }_{F \in  \mathcal{E}}{R}_{F}}\right)$

算法3 通用连接 $\left( {{ \bowtie  }_{F \in  \mathcal{E}}{R}_{F}}\right)$

---

Input: Query $Q$ ,hypergraph $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$

输入：查询 $Q$，超图 $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$

	$Q \leftarrow  \varnothing$

	If $\left| \mathcal{V}\right|  = 1$ then

	如果 $\left| \mathcal{V}\right|  = 1$ 则

		return $\mathop{\bigcap }\limits_{{F \in  \mathcal{E}}}{R}_{F}$

		返回 $\mathop{\bigcap }\limits_{{F \in  \mathcal{E}}}{R}_{F}$

	Pick $I$ arbitrarily such that $1 \leq  \left| I\right|  < \left| \mathcal{V}\right|$

	任意选取 $I$ 使得 $1 \leq  \left| I\right|  < \left| \mathcal{V}\right|$

	$L \leftarrow$ Generic-Join $\left( {{ \bowtie  }_{F \in  {\mathcal{E}}_{I}}{\pi }_{I}\left( {R}_{F}\right) }\right)$

	$L \leftarrow$ 通用连接 $\left( {{ \bowtie  }_{F \in  {\mathcal{E}}_{I}}{\pi }_{I}\left( {R}_{F}\right) }\right)$

	For every ${\mathbf{t}}_{I} \in  L$ do

	对每个 ${\mathbf{t}}_{I} \in  L$ 执行

		$Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack   \leftarrow$ Generic-Join $\left( {{ \bowtie  }_{F \in  {\mathcal{E}}_{J}}{\pi }_{J}\left( {{R}_{F} \ltimes  {\mathbf{t}}_{I}}\right) }\right)$

		$Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack   \leftarrow$ 通用连接 $\left( {{ \bowtie  }_{F \in  {\mathcal{E}}_{J}}{\pi }_{J}\left( {{R}_{F} \ltimes  {\mathbf{t}}_{I}}\right) }\right)$

		$Q \leftarrow  Q \cup  \left\{  {\mathbf{t}}_{I}\right\}   \times  Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$

	Return $Q$

	返回 $Q$

---

<!-- Media -->

Given the indices,when $\left| \mathcal{V}\right|  = 1$ computing $\mathop{\bigcap }\limits_{{F \in  \mathcal{E}}}{R}_{F}$ can easily be done in time

给定索引后，当 $\left| \mathcal{V}\right|  = 1$ 时，计算 $\mathop{\bigcap }\limits_{{F \in  \mathcal{E}}}{R}_{F}$ 可以在规定时间内轻松完成

$$
\widetilde{O}\left( {m\min \left| {R}_{F}\right| }\right)  = \widetilde{O}\left( {m\mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}}\right) .
$$

Then, given this base-case runtime guarantee, we can show by induction that the overall runtime of Algorithm 3 is $\widetilde{O}\left( {{mn}\mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}}\right)$ ,where $\widetilde{O}$ hides a potential log-factor of the input size. This is because,by induction the time it takes to compute $L$ is $\widetilde{O}\left( {m\left| I\right| \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{I}}}{\left| {R}_{F}\right| }^{{x}_{F}}}\right)$ ,and the time it takes to compute $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ is

然后，基于这个基本情况的运行时间保证，我们可以通过归纳法证明算法 3 的总体运行时间为 $\widetilde{O}\left( {{mn}\mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}}\right)$，其中 $\widetilde{O}$ 隐藏了输入规模的潜在对数因子。这是因为，通过归纳法，计算 $L$ 所需的时间为 $\widetilde{O}\left( {m\left| I\right| \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{I}}}{\left| {R}_{F}\right| }^{{x}_{F}}}\right)$，而计算 $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ 所需的时间为

$$
\widetilde{O}\left( {m\left( {n - \left| I\right| }\right) \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}}}\right) 
$$

Hence,from Lemma 4.1,the total run time is $\widetilde{O}$ of

因此，根据引理 4.1，总运行时间为 $\widetilde{O}$ 的

$$
m\left| I\right| \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{I}}}{\left| {R}_{F}\right| }^{{x}_{F}} + m\left( {n - \left| I\right| }\right) \mathop{\sum }\limits_{{{\mathbf{t}}_{I} \in  L}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}}
$$

$$
 \leq  m\left| I\right| \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{I}}}{\left| {R}_{F}\right| }^{{x}_{F}} + m\left( {n - \left| I\right| }\right) \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}
$$

$$
 \leq  {mn}\mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{{x}_{F}}.
$$

The NPRR algorithm is an instantiation of Algorithm 3 where it picks $J \in  \mathcal{E},I = \mathcal{V} - J$ ,and solves the sub-queries $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ in a different way,making use of the power of two choices idea. Since $J \in  \mathcal{E}$ ,we write

NPRR 算法是算法 3 的一个实例，它选取 $J \in  \mathcal{E},I = \mathcal{V} - J$，并以不同的方式解决子查询 $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$，利用了二选一的思想。由于 $J \in  \mathcal{E}$，我们记为

$$
Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack   = {R}_{J} \bowtie  \left( {{ \bowtie  }_{F \in  {\mathcal{E}}_{J}-\{ J\} }{\pi }_{J}\left( {{R}_{F} \bowtie  {\mathbf{t}}_{I}}\right) }\right) .
$$

Now,if ${x}_{J} \geq  1$ then we solve for $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ by checking for every tuple in ${R}_{J}$ whether it can be part of $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ .

现在，如果 ${x}_{J} \geq  1$，那么我们通过检查 ${R}_{J}$ 中的每个元组是否可以成为 $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$ 的一部分来求解 $Q\left\lbrack  {\mathbf{t}}_{I}\right\rbrack$。

The run time is $\widetilde{O}$ of

运行时间为 $\widetilde{O}$ 的

$$
\left( {n - \left| I\right| }\right) \left| {R}_{J}\right|  \leq  \left( {n - \left| I\right| }\right) \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}}.
$$

When ${x}_{J} < 1$ ,we will make use of an extremely simple observation: for any real numbers $p,q \geq  0$ and $z \in  \left\lbrack  {0,1}\right\rbrack  ,\min \{ p,q\}  \leq  {p}^{z}{q}^{1 - z}$ (note that ② is the special case of $z = 1/2$ ). In particular,define

当 ${x}_{J} < 1$ 时，我们将利用一个极其简单的观察结果：对于任意实数 $p,q \geq  0$ 和 $z \in  \left\lbrack  {0,1}\right\rbrack  ,\min \{ p,q\}  \leq  {p}^{z}{q}^{1 - z}$（注意②是 $z = 1/2$ 的特殊情况）。特别地，定义

$$
p = \left| {R}_{J}\right| 
$$

$$
q = \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}-\{ J\} }}{\left| {\pi }_{J}\left( {R}_{F} \ltimes  {\mathbf{t}}_{I}\right) \right| }^{\frac{{x}_{F}}{1 - {x}_{J}}}
$$

Then,

那么，

$$
\min \{ p,q\}  \leq  {\left| {R}_{J}\right| }^{{x}_{J}}\mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}-\{ J\} }}{\left| {\pi }_{J}\left( {R}_{F} \ltimes  {\mathbf{t}}_{I}\right) \right| }^{{x}_{F}}
$$

$$
 = \mathop{\prod }\limits_{{F \in  {\mathcal{E}}_{J}}}{\left| {R}_{F} \ltimes  {\mathbf{t}}_{I}\right| }^{{x}_{F}}.
$$

From there,when ${x}_{J} < 1$ and $p \leq  q$ ,we go through each tuple in ${R}_{J}$ and check as in the case ${x}_{J} \geq  1$ . And when $p > q$ ,we solve the subquery ${ \bowtie  }_{F \in  {\mathcal{E}}_{J}-\{ J\} }{\pi }_{J}\left( {{R}_{F} \bowtie  {\mathbf{t}}_{I}}\right)$ first using ${\left( \frac{{x}_{F}}{1 - {x}_{J}}\right) }_{F \in  {\mathcal{E}}_{J}-\{ J\} }$ as its fractional edge cover; and then check for each tuple in the result whether it is in ${R}_{J}$ . In either case,the run time $\widetilde{O}\left( {\min \{ p,q\} }\right)$ which along with the observation above completes the proof.

从那里开始，当${x}_{J} < 1$和$p \leq  q$时，我们遍历${R}_{J}$中的每个元组，并按照${x}_{J} \geq  1$的情况进行检查。而当$p > q$时，我们首先使用${\left( \frac{{x}_{F}}{1 - {x}_{J}}\right) }_{F \in  {\mathcal{E}}_{J}-\{ J\} }$作为其分数边覆盖来求解子查询${ \bowtie  }_{F \in  {\mathcal{E}}_{J}-\{ J\} }{\pi }_{J}\left( {{R}_{F} \bowtie  {\mathbf{t}}_{I}}\right)$；然后检查结果中的每个元组是否在${R}_{J}$中。无论哪种情况，运行时间为$\widetilde{O}\left( {\min \{ p,q\} }\right)$，结合上述观察结果，证明完毕。

Next we outline how Algorithm 1 is Algorithm 3 with the above modification for NPRR for the triangle query ${Q}_{ \bigtriangleup  }$ . In particular,we will use $\mathbf{x} = \left( {1/2,1/2,1/2}\right)$ and $I = \{ A\}$ . Note that this choice of $I$ implies that $J = \{ B,C\}$ ,which means in Step 5 Algorithm 3 computes

接下来，我们概述算法1如何是对三角形查询${Q}_{ \bigtriangleup  }$的NPRR进行上述修改后的算法3。具体来说，我们将使用$\mathbf{x} = \left( {1/2,1/2,1/2}\right)$和$I = \{ A\}$。注意，这种对$I$的选择意味着$J = \{ B,C\}$，这意味着在算法3的第5步中计算

$$
L = {\pi }_{A}\left( R\right)  \bowtie  {\pi }_{A}\left( T\right)  = {\pi }_{A}\left( R\right)  \cap  {\pi }_{A}\left( T\right) ,
$$

which is exactly the same $L$ as in Algorithm 1 . Thus,in the remaining part of Algorithm 3 one would cycle through all $a \in  L$ (as one does in Algorithm 1). In particular,by the discussion above,since ${x}_{S} = 1/2 < 1$ , we will try the best of two choices. In particular, we have

这与算法1中的$L$完全相同。因此，在算法3的其余部分，将遍历所有的$a \in  L$（就像在算法1中那样）。具体来说，根据上述讨论，由于${x}_{S} = 1/2 < 1$，我们将尝试两种选择中的最优方案。具体而言，我们有

$$
{ \bowtie  }_{F \in  {\mathcal{E}}_{J}-\{ J\} }{\pi }_{J}\left( {{R}_{F} \bowtie  \left( a\right) }\right)  = {\pi }_{B}\left( {{\sigma }_{A = a}R}\right)  \times  {\pi }_{C}\left( {{\sigma }_{A = a}T}\right) ,
$$

$$
p = \left| I\right| 
$$

$$
q = \left| {{\sigma }_{A = a}R}\right|  \cdot  \left| {{\sigma }_{A = a}T}\right| .
$$

Hence, the NPRR algorithm described exactly matches Algorithm 1,

因此，所描述的NPRR算法与算法1完全匹配。

The Leapfrog Triejoin algorithm [42] is an instantiation of Algorithm 3 where $\mathcal{V} = \left\lbrack  n\right\rbrack$ and $I =$ $\{ 1,\ldots ,n - 1\}$ (or equivalently $I = \{ 1\}$ ). To illustrate,we outline how Algorithm 2 is Algorithm 3 with $I = \{ A,B\}$ when specialized to ${Q}_{ \bigtriangleup  }$ . Consider the run of Algorithm 3 on ${\mathcal{H}}_{ \bigtriangleup  }$ ,and the first time Step 4 is executed. The call to Generic-Join in Step 5 returns $L = \left\{  {\left( {a,b}\right)  \mid  a \in  {L}_{A},b \in  {L}_{B}^{a}}\right\}$ ,where ${L}_{A}$ and ${L}_{B}^{a}$ are as defined in Algorithm 2. The rest of Algorithm 3 is to do the following for every $\left( {a,b}\right)  \in  L.Q\left\lbrack  \left( {a,b}\right) \right\rbrack$ is computed by the recursive call to Algorithm 3 to obtain $\left\{  {\left( {a,b}\right)  \times  {L}_{C}^{a,b}}\right.$ ,where

Leapfrog Triejoin算法[42]是算法3的一个实例，其中$\mathcal{V} = \left\lbrack  n\right\rbrack$且$I =$ $\{ 1,\ldots ,n - 1\}$（或者等价地$I = \{ 1\}$）。为了说明这一点，我们概述当专门针对${Q}_{ \bigtriangleup  }$时，算法2如何是带有$I = \{ A,B\}$的算法3。考虑算法3在${\mathcal{H}}_{ \bigtriangleup  }$上的运行，以及第一次执行第4步的情况。第5步中对Generic - Join的调用返回$L = \left\{  {\left( {a,b}\right)  \mid  a \in  {L}_{A},b \in  {L}_{B}^{a}}\right\}$，其中${L}_{A}$和${L}_{B}^{a}$的定义与算法2中的相同。算法3的其余部分是对每个$\left( {a,b}\right)  \in  L.Q\left\lbrack  \left( {a,b}\right) \right\rbrack$执行以下操作，通过对算法3的递归调用计算得到$\left\{  {\left( {a,b}\right)  \times  {L}_{C}^{a,b}}\right.$，其中

$$
{L}_{C}^{a,b} = {\pi }_{C}{\sigma }_{B = b}S \bowtie  {\pi }_{C}{\sigma }_{A = a}T = {\pi }_{C}{\sigma }_{B = b}S \cap  {\pi }_{C}{\sigma }_{A = a}T,
$$

exactly as was done in Algorithm 2. Finally,we get back to $L$ in Step 5 being as claimed above. Note that in during the recursive call of Algorithm 3 on ${Q}_{\infty } = R \bowtie  {\pi }_{B}\left( S\right)  \bowtie  {\pi }_{A}\left( T\right)$ . The claim follows by picking $I = \{ A\}$ in Step 4 when Algorithm 3 is run on ${Q}_{\infty }$ (and tracing through rest of Algorithm 3).

这与算法2中所做的完全相同。最后，我们回到第5步中的$L$，如上述所声称的那样。注意，在算法3对${Q}_{\infty } = R \bowtie  {\pi }_{B}\left( S\right)  \bowtie  {\pi }_{A}\left( T\right)$进行递归调用期间。通过在算法3对${Q}_{\infty }$运行时在第4步选择$I = \{ A\}$（并遍历算法3的其余部分），该声明成立。

### 4.3 On the limitation of any join-project plan

### 4.3 关于任何连接 - 投影计划的局限性

AGM proved that there are classes of queries for which join-only plans are significantly worse than their join-project plan. In particular,they showed that for every $M,N \in  \mathbb{N}$ ,there is a query $Q$ of size at least $M$ and a database $\mathcal{D}$ of size at least $N$ such that ${2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) } \leq  {N}^{2}$ and every join-only plan runs in time at least $N\frac{1}{5}{\log }_{2}\left| Q\right|$

AGM（阿尔乔姆夫 - 加德纳 - 米勒）证明，对于某些类别的查询，仅连接计划（join-only plans）明显比连接 - 投影计划（join-project plan）差。具体而言，他们表明，对于每个 $M,N \in  \mathbb{N}$，存在一个规模至少为 $M$ 的查询 $Q$ 和一个规模至少为 $N$ 的数据库 $\mathcal{D}$，使得 ${2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) } \leq  {N}^{2}$ 成立，并且每个仅连接计划的运行时间至少为 $N\frac{1}{5}{\log }_{2}\left| Q\right|$

NPRR continued with the story and noted that for the class of $L{W}_{n}$ queries from Section 3.2 every join-project plan runs in time polynomially worse than the AGM bound. The proof of the following lemma can be found in Appendix F.

NPRR（纳拉亚南 - 帕帕康斯坦丁努 - 拉贾拉曼 - 苏达拉扬）继续探讨该问题，并指出对于 3.2 节中 $L{W}_{n}$ 类查询，每个连接 - 投影计划的运行时间比 AGM 界限（AGM bound）在多项式意义上更差。以下引理的证明可在附录 F 中找到。

Lemma 4.2. Let $n \geq  2$ be an arbitrary integer. For any LW-query $Q$ with corresponding hypergraph $\mathcal{H} =$ $\left( {\left\lbrack  n\right\rbrack  ,\left( \begin{matrix} \left\lbrack  n\right\rbrack  \\  n - 1 \end{matrix}\right) }\right)$ ,and any positive integer $N \geq  2$ ,there exist $n$ relations ${R}_{i},i \in  \left\lbrack  n\right\rbrack$ such that $\left| {R}_{i}\right|  = N,\forall i \in  \left\lbrack  n\right\rbrack$ , the attribute set for ${R}_{i}$ is $\left\lbrack  n\right\rbrack   - \{ i\}$ ,and that any join-project plan for $Q$ on these relations has run-time at least $\Omega \left( {{N}^{2}/{n}^{2}}\right)$ .

引理 4.2。设 $n \geq  2$ 为任意整数。对于任何具有对应超图 $\mathcal{H} =$ $\left( {\left\lbrack  n\right\rbrack  ,\left( \begin{matrix} \left\lbrack  n\right\rbrack  \\  n - 1 \end{matrix}\right) }\right)$ 的 LW 查询（LW-query）$Q$，以及任何正整数 $N \geq  2$，存在 $n$ 个关系 ${R}_{i},i \in  \left\lbrack  n\right\rbrack$，使得 $\left| {R}_{i}\right|  = N,\forall i \in  \left\lbrack  n\right\rbrack$ 成立，${R}_{i}$ 的属性集为 $\left\lbrack  n\right\rbrack   - \{ i\}$，并且 $Q$ 在这些关系上的任何连接 - 投影计划的运行时间至少为 $\Omega \left( {{N}^{2}/{n}^{2}}\right)$。

Note that both the traditional join-tree-based algorithms and AGM's algorithm described in Appendix D.2 are join-project plans. Consequently, they run in time asymptotically worse than the best AGM bound for this instance, which is

请注意，传统的基于连接树的算法和附录 D.2 中描述的 AGM 算法都是连接 - 投影计划。因此，对于这个实例，它们的运行时间在渐近意义上比最佳 AGM 界限更差，该界限为

$$
\left| {{ \bowtie  }_{i = 1}^{n}{R}_{i}}\right|  \leq  \mathop{\prod }\limits_{{i = 1}}^{n}{\left| {R}_{i}\right| }^{1/\left( {n - 1}\right) } = {N}^{1 + 1/\left( {n - 1}\right) }.
$$

On the other hand,both algorithms described in Section 4.2 take $O\left( {N}^{1 + 1/\left( {n - 1}\right) }\right)$ -time because their run times match the AGM bound. In fact, the NPRR algorithm in Section 4.2 can be shown to run in linear data-complexity time $O\left( {{n}^{2}N}\right)$ for this query [32].

另一方面，4.2 节中描述的两种算法都需要 $O\left( {N}^{1 + 1/\left( {n - 1}\right) }\right)$ 时间，因为它们的运行时间与 AGM 界限相匹配。事实上，4.2 节中的 NPRR 算法对于此查询 [32] 可以证明在线性数据复杂度时间 $O\left( {{n}^{2}N}\right)$ 内运行。

## 5 Open Questions

## 5 开放性问题

We conclude this survey with two open questions: one for systems researchers and one for theoreticians:

我们以两个开放性问题结束本次综述：一个是针对系统研究人员的，另一个是针对理论研究人员的：

1. A natural question to ask is whether the algorithmic ideas that were presented in this survey can gain runtime efficiency in databases systems. This is an intriguing open question: on one hand we have shown asymptotic improvements in join algorithms, but on the other there are several decades of engineering refinements and research contributions in the traditional dogma.

1. 一个自然的问题是，本次综述中提出的算法思想是否能在数据库系统中提高运行时效率。这是一个引人入胜的开放性问题：一方面，我们已经展示了连接算法的渐近改进；但另一方面，传统教条中有几十年的工程改进和研究成果。

2. Worst-case results, as noted by several authors, may only give us information about pathological instances. Thus, there is a natural push toward more refined measures of complexity. For example, current complexity measures are too weak to explain why indexes are used or give insight into the average case. For example, could one design an adaptive join algorithm whose run time is somehow dictated by the "difficulty" of the input instance (instead of the input size as in the currently known results)?

2. 正如几位作者所指出的，最坏情况结果可能仅能为我们提供关于病态实例的信息。因此，自然会推动采用更精细的复杂度度量。例如，当前的复杂度度量太弱，无法解释为什么使用索引，也无法洞察平均情况。例如，能否设计一种自适应连接算法，其运行时间以某种方式由输入实例的“难度”（而不是像目前已知结果那样由输入规模）决定？

## Acknowledgements

## 致谢

HN's work is partly supported by NSF grant CCF-1319402 and a gift from Logicblox. CR's work on this project is generously supported by NSF CAREER Award under No. IIS-1353606, NSF award under No. CCF-1356918, the ONR under awards No. N000141210041 and No. N000141310129, Sloan Research Fellowship, Oracle, and Google. AR's work is partly supported by NSF CAREER Award CCF-CCF-0844796, NSF grant CCF-1319402 and a gift from Logicblox.

HN 的工作部分得到了美国国家科学基金会（NSF）资助项目 CCF - 1319402 以及 Logicblox 公司的捐赠支持。CR 在该项目上的工作得到了美国国家科学基金会职业发展奖（NSF CAREER Award）（编号 IIS - 1353606）、美国国家科学基金会资助项目（编号 CCF - 1356918）、美国海军研究办公室（ONR）资助项目（编号 N000141210041 和 N000141310129）、斯隆研究奖学金（Sloan Research Fellowship）、甲骨文公司（Oracle）和谷歌公司（Google）的慷慨支持。AR 的工作部分得到了美国国家科学基金会职业发展奖 CCF - CCF - 0844796、美国国家科学基金会资助项目 CCF - 1319402 以及 Logicblox 公司的捐赠支持。

## References

## 参考文献

[1] ABITEBOUL, S., HULL, R., AND VIANU, V. Foundations of Databases. Addison-Wesley, 1995.

[2] Aho, A. V., Beeri, C., and ULMAN, J. D. The theory of joins in relational databases. ACM Trans. Database Syst. 4, 3 (1979), 297-314.

[3] ATSERIAS, A., Grohe, M., AND Marx, D. Size bounds and query plans for relational joins. In FOCS (2008), IEEE Computer Society, pp. 739-748.

[4] BeerI, C., AND VarDI, M. Y. A proof procedure for data dependencies. J. ACM 31, 4 (1984), 718-741.

[4] 比尔（BeerI），C.，和瓦尔迪（VarDI），M. Y. 数据依赖的证明过程。《美国计算机协会期刊》（J. ACM）31 卷，第 4 期（1984 年），718 - 741 页。

[5] BLANAS, S., LI, Y., AND PATEL, J. M. Design and evaluation of main memory hash join algorithms for multi-core cpus. In SIGMOD (2011), ACM, pp. 37-48.

[6] Bollobás, B., AND THOMASON, A. Projections of bodies and hereditary properties of hypergraphs. Bull. London Math. Soc. 27, 5 (1995), 417-424.

[7] Chandra, A. K., and Merlin, P. M. Optimal implementation of conjunctive queries in relational data bases. In STOC (1977), J. E. Hopcroft, E. P. Friedman, and M. A. Harrison, Eds., ACM, pp. 77-90.

[8] Chaudhuri, S. An overview of query optimization in relational systems. In PODS (1998), ACM, pp. 34-43.

[9] Chekuri, C., and Rajaraman, A. Conjunctive query containment revisited. Theor. Comput. Sci. 239, 2 (2000), 211-229.

[10] Chung, F. R. K., Graham, R. L., FrankL, P., and Shearer, J. B. Some intersection theorems for ordered sets and graphs. J. Combin. Theory Ser. A 43, 1 (1986), 23-37.

[11] Cover, T. M., and Thomas, J. A. Elements of information theory, second ed. Wiley-Interscience [John Wiley & Sons], Hoboken, NJ, 2006.

[12] DeWitt, D. J., Naughton, J. F., Schneider, D. A., and Seshadri, S. Practical skew handling in parallel joins. In Proceedings of the 18th International Conference on Very Large Data Bases (San Francisco, CA, USA, 1992), VLDB '92, Morgan Kaufmann Publishers Inc., pp. 27-40.

[13] FAGIN, R. Degrees of acyclicity for hypergraphs and relational database schemes. J. ACM 30, 3 (1983), 514-550.

[14] FRIEDGUT, E., AND KAHN, J. On the number of copies of one hypergraph in another. Israel J. Math. 105 (1998), 251-256.

[15] Gorttos, G., Lee, S. T., and Valiant, G. Size and treewidth bounds for conjunctive queries. In PODS (2009), J. Paredaens and J. Su, Eds., ACM, pp. 45-54.

[16] Gorttob, G., Lee, S. T., Vallant, G., and Vallant, P. Size and treewidth bounds for conjunctive queries. J. ACM 59, 3 (2012), 16.

[17] Gorttob, G., Leone, N., and Scarcello, F. Hypertree decompositions and tractable queries. J. Comput. Syst. Sci. 64, 3 (2002), 579-627.

[18] Gorttos, G., Leone, N., and Scarcello, F. Robbers, marshals, and guards: game theoretic and logical characterizations of hypertree width. J. Comput. Syst. Sci. 66, 4 (2003), 775-808.

[19] Gorthos, G., Mıкцбs, Z., and Schwentick, T. Generalized hypertree decompositions: Np-hardness and tractable variants. J. ACM 56, 6 (2009).

[20] Graefe, G. Query evaluation techniques for large databases. ACM Computing Surveys 25, 2 (June 1993), 73-170.

[21] Graндм, M. H. On the universal relation, 1980. Tech. Report.

[22] Grohe, M. Bounds and algorithms for joins via fractional edge covers. http://www.automata.rwth-aachen.de/~grohe/pub/gro12+.pdf, 2012. Manuscript.

[23] Grohe, M., and Marx, D. Constraint solving via fractional edge covers. In SODA (2006), ACM Press, pp. 289-298.

[24] GYSSENs, M., JEAVONS, P., AND CohEN, D. A. Decomposing constraint satisfaction problems using database techniques. Artif. Intell. 66, 1 (1994), 57-89.

[25] Gyssens, M., and Paredaens, J. A decomposition methodology for cyclic databases. In Advances in Data Base Theory (1982), pp. 85-122.

[26] Hardy, G. H., Littlewood, J. E., and Pólya, G. Inequalities. Cambridge University Press, Cambridge, 1988. Reprint of the 1952 edition.

[27] Kim, C., Kaldewey, T., Lee, V. W., SEDLar, E., Nguyen, A. D., Satish, N., ChHuGani, J., Di BLas, A., AND DUBEY, P. Sort vs. hash revisited: fast join implementation on modern multi-core cpus. Proc. VLDB Endow. 2, 2 (Aug. 2009), 1378-1389.

[28] Kolairis, P. G., and Varpi, M. Y. Conjunctive-query containment and constraint satisfaction. J. Com-put. Syst. Sci. 61, 2 (2000), 302-332.

[29] Loomis, L. H., AND WHITNEY, H. An inequality related to the isoperimetric inequality. Bull. Amer. Math. Soc 55 (1949), 961-962.

[30] Maier, D., Mendelzon, A. O., and Sagiv, Y. Testing implications of data dependencies. ACM Trans. Database Syst. 4, 4 (Dec. 1979), 455-469.

[31] Marx, D. Approximating fractional hypertree width. ACM Trans. Algorithms 6, 2 (Apr. 2010), 29:1- 29:17.

[32] Ngo, H. Q., Porant, E., Ré, C., and Rubna, A. Worst-case optimal join algorithms: [extended abstract]. In ${PODS}\left( {2012}\right)$ ,pp. 37-48.

[33] Papabimitratou, C. H., and Yannakakis, M. On the complexity of database queries. In PODS (1997), A. O. Mendelzon and Z. M. Özsoyoglu, Eds., ACM Press, pp. 12-19.

[34] RADHAKRISHNAN, J. Entropy and counting. Computational Mathematics, Modelling and Algorithms (2003), 146.

[35] RamakRISHNAN, R., AND GERRKE, J. Database Management Systems, 3 ed. McGraw-Hill, Inc., New York, NY, USA, 2003.

[36] Robertson, N., and Seymour, P. D. Graph minors. II. Algorithmic aspects of tree-width. J. Algorithms 7, 3 (1986), 309-322.

[37] Scarcello, F. Query answering exploiting structural properties. SIGMOD Record 34, 3 (2005), 91-99.

[38] Selinger, P. G., Astrahan, M. M., Chamberlin, D. D., Lorie, R. A., and Price, T. G. Access path selection in a relational database management system. In Proceedings of the 1979 ACM SIGMOD international conference on Management of data (New York, NY, USA, 1979), SIGMOD '79, ACM, pp. 23-34.

[39] Suri, S., AND VASSILVITSKII, S. Counting triangles and the curse of the last reducer. In WWW (2011), pp. 607-614.

[40] Tsourakkakis, C. E. Fast counting of triangles in large real networks without counting: Algorithms and laws. In ICDM (2008), IEEE Computer Society, pp. 608-617.

[41] Vardi, M. Y. The complexity of relational query languages (extended abstract). In STOC (1982), H. R. Lewis, B. B. Simons, W. A. Burkhard, and L. H. Landweber, Eds., ACM, pp. 137-146.

[42] VELDHUIZEN, T. L. Leapfrog Triejoin: a worst-case optimal join algorithm. ArXiv e-prints (Oct. 2012).

[43] Walton, C. B., Dale, A. G., and Jenevein, R. M. A taxonomy and performance model of data skew effects in parallel joins. In Proceedings of the 17th International Conference on Very Large Data Bases (San Francisco, CA, USA, 1991), VLDB '91, Morgan Kaufmann Publishers Inc., pp. 537-548.

[44] Xu, Y., Kostamaa, P., Zhou, X., and Chen, L. Handling data skew in parallel joins in shared-nothing systems. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data (New York, NY, USA, 2008), SIGMOD '08, ACM, pp. 1043-1052.

[45] Yannakakis, M. Algorithms for acyclic database schemes. In VLDB (1981), IEEE Computer Society, pp. 82-94.

[46] Yu, C., AND OzsoyoGLU, M. On determining tree-query membership of a distributed query. Informatica 22, 3 (1984), 261-282.

## A Relation Algebra Notation

## 关系代数符号

We assume the existence of a set of attribute names $\mathcal{A} = {A}_{1},\ldots ,{A}_{n}$ with associated domains ${\mathbf{D}}_{1},\ldots ,{\mathbf{D}}_{n}$ and infinite set of relational symbols $\mathcal{R}$ . A relational schema for the symbol $R \in  \mathcal{R}$ of arity $k$ is a tuple $\bar{A}\left( R\right)  = \left( {{A}_{{i}_{1}},\ldots ,{A}_{{i}_{k}}}\right)$ of distinct attributes that defines the attributes of the relation. A relational database schema is a set of relational symbols and associated schemas denoted by $R\left( {\bar{A}\left( R\right) }\right) ,R \in  \mathcal{R}$ . A relational instance for $R\left( {{A}_{{i}_{1}},\ldots ,{A}_{{i}_{k}}}\right)$ is a subset of ${\mathbf{D}}_{{i}_{1}} \times  \cdots  \times  {\mathbf{D}}_{{i}_{k}}$ . A relational database $\mathcal{D}$ is a collection of instances, one for each relational symbol in schema,denoted by ${R}^{\mathcal{D}}$ .

我们假设存在一组属性名$\mathcal{A} = {A}_{1},\ldots ,{A}_{n}$，其关联的域为${\mathbf{D}}_{1},\ldots ,{\mathbf{D}}_{n}$，以及一个无限的关系符号集$\mathcal{R}$。对于元数为$k$的符号$R \in  \mathcal{R}$，其关系模式是一个由不同属性组成的元组$\bar{A}\left( R\right)  = \left( {{A}_{{i}_{1}},\ldots ,{A}_{{i}_{k}}}\right)$，该元组定义了该关系的属性。关系数据库模式是一组关系符号和相关模式的集合，用$R\left( {\bar{A}\left( R\right) }\right) ,R \in  \mathcal{R}$表示。对于$R\left( {{A}_{{i}_{1}},\ldots ,{A}_{{i}_{k}}}\right)$的关系实例是${\mathbf{D}}_{{i}_{1}} \times  \cdots  \times  {\mathbf{D}}_{{i}_{k}}$的一个子集。关系数据库$\mathcal{D}$是实例的集合，模式中的每个关系符号对应一个实例，用${R}^{\mathcal{D}}$表示。

A natural join query (or simply join query) $Q$ is specified by a finite subset of relational symbols atoms $\left( Q\right)  \subseteq  \mathcal{R}$ ,denoted by ${ \bowtie  }_{R \in  \operatorname{atoms}\left( Q\right) }R$ . Let $\bar{A}\left( Q\right)$ denote the set of all attributes that appear in some relation in $Q$ ,that is $\bar{A}\left( Q\right)  = \{ A \mid  A \in  \bar{A}\left( R\right)$ for some $R \in  \operatorname{atoms}\left( Q\right) \}$ . Given a tuple $\mathbf{t}$ we will write ${\mathbf{t}}_{\bar{A}}$ to emphasize that its support is the attribute set $\bar{A}$ . Further,for any $\bar{S} \subset  \bar{A}$ we let ${\mathbf{t}}_{\bar{S}}$ denote $\mathbf{t}$ restricted to $\bar{S}$ . Given a database instance $\mathcal{D}$ ,the output of the query $Q$ on the database instance $\mathcal{D}$ is denoted $Q\left( \mathcal{D}\right)$ and is

自然连接查询（或简称为连接查询）$Q$由关系符号原子的一个有限子集$\left( Q\right)  \subseteq  \mathcal{R}$指定，用${ \bowtie  }_{R \in  \operatorname{atoms}\left( Q\right) }R$表示。设$\bar{A}\left( Q\right)$表示在$Q$的某个关系中出现的所有属性的集合，即对于某个$R \in  \operatorname{atoms}\left( Q\right) \}$有$\bar{A}\left( Q\right)  = \{ A \mid  A \in  \bar{A}\left( R\right)$。给定一个元组$\mathbf{t}$，我们将用${\mathbf{t}}_{\bar{A}}$来强调其支撑集是属性集$\bar{A}$。此外，对于任何$\bar{S} \subset  \bar{A}$，我们用${\mathbf{t}}_{\bar{S}}$表示$\mathbf{t}$在$\bar{S}$上的限制。给定一个数据库实例$\mathcal{D}$，查询$Q$在数据库实例$\mathcal{D}$上的输出用$Q\left( \mathcal{D}\right)$表示，且为

defined as

定义为

$$
Q\left( \mathcal{D}\right) \overset{\text{ def }}{ = }\left\{  {\mathbf{t} \in  {\mathbf{D}}^{\bar{A}\left( Q\right) } \mid  {\mathbf{t}}_{\bar{A}\left( R\right) } \in  {R}^{\mathcal{D}}\text{ for each }R \in  \operatorname{atoms}\left( Q\right) }\right\}  
$$

where ${\mathbf{D}}^{\bar{A}\left( Q\right) }$ is a shorthand for ${ \times  }_{i : {A}_{i} \in  \bar{A}\left( Q\right) }{\mathbf{D}}_{i}$ . When the instance is clear from the context we will refer to $Q\left( \mathcal{D}\right)$ by just $Q$ .

其中${\mathbf{D}}^{\bar{A}\left( Q\right) }$是${ \times  }_{i : {A}_{i} \in  \bar{A}\left( Q\right) }{\mathbf{D}}_{i}$的简写。当实例在上下文中明确时，我们将仅用$Q$来指代$Q\left( \mathcal{D}\right)$。

We also use the notion of a semijoin: Given two relations $R\left( \bar{A}\right)$ and $S\left( \bar{B}\right)$ their semijoin $R \ltimes  S$ is defined by

我们还使用半连接的概念：给定两个关系$R\left( \bar{A}\right)$和$S\left( \bar{B}\right)$，它们的半连接$R \ltimes  S$定义为

$$
R \ltimes  S\overset{\text{ def }}{ = }\left\{  {\mathbf{t} \in  R : \exists \mathbf{u} \in  S\text{ s.t. }{\mathbf{t}}_{\bar{A} \cap  \bar{B}} = {\mathbf{u}}_{\bar{A} \cap  \bar{B}}}\right\}  .
$$

For any relation $R\left( \bar{A}\right)$ ,and any subset $\bar{S} \subseteq  \bar{A}$ of its attributes,let ${\pi }_{\bar{S}}\left( R\right)$ denote the projection of $R$ onto $\bar{S}$ , i.e.

对于任何关系$R\left( \bar{A}\right)$，以及其属性的任何子集$\bar{S} \subseteq  \bar{A}$，用${\pi }_{\bar{S}}\left( R\right)$表示$R$在$\bar{S}$上的投影，即

$$
{\pi }_{\bar{S}}\left( R\right)  = \left\{  {{\mathbf{t}}_{\bar{S}}\mid \exists {\mathbf{t}}_{\bar{A} \smallsetminus  \bar{S}},\left( {{\mathbf{t}}_{\bar{S}},{\mathbf{t}}_{\bar{A} \smallsetminus  \bar{S}}}\right)  \in  R}\right\}  .
$$

For any relation $R\left( \bar{A}\right)$ ,any subset $\bar{S} \subseteq  \bar{A}$ of its attributes,and a vector $\mathbf{s} \in  \mathop{\prod }\limits_{{i \in  \bar{S}}}{\mathbf{D}}_{i}$ ,let ${\sigma }_{\bar{S} = \mathbf{s}}\left( R\right)$ denote the selection of $R$ with $\bar{S}$ and $\mathbf{s}$ ,i.e.

对于任意关系$R\left( \bar{A}\right)$，其属性的任意子集$\bar{S} \subseteq  \bar{A}$，以及向量$\mathbf{s} \in  \mathop{\prod }\limits_{{i \in  \bar{S}}}{\mathbf{D}}_{i}$，令${\sigma }_{\bar{S} = \mathbf{s}}\left( R\right)$表示对$R$进行选择，选择条件为$\bar{S}$和$\mathbf{s}$，即

$$
{\sigma }_{\bar{S} = \mathbf{s}}\left( R\right)  = \left\{  {\mathbf{t} \mid  {\mathbf{t}}_{\bar{S}} = \mathbf{s}}\right\}  .
$$

## B Analysis of Algorithm 2

## B 算法2分析

It turns out that the run time of Algorithm 2 is dominated by the time spent in computing the set ${L}_{C}^{a,b}$ for every $a \in  {L}_{A}$ and $b \in  {L}_{B}^{a}$ . We now analyze this time "inside out." We first note that for a given $a \in  {L}_{A}$ and $b \in  {L}_{B}^{a}$ ,the time it takes to compute ${L}_{C}^{a,b}$ is at most

事实证明，算法2的运行时间主要取决于为每个$a \in  {L}_{A}$和$b \in  {L}_{B}^{a}$计算集合${L}_{C}^{a,b}$所花费的时间。我们现在“由内而外”地分析这个时间。首先，我们注意到对于给定的$a \in  {L}_{A}$和$b \in  {L}_{B}^{a}$，计算${L}_{C}^{a,b}$所需的时间至多为

$$
\min \left\{  {\left| {{\pi }_{C}{\sigma }_{B = b}S}\right| ,\left| {{\pi }_{C}{\sigma }_{A = a}T}\right| }\right\}   \leq  \sqrt{\left| {{\pi }_{C}{\sigma }_{B = b}S}\right|  \cdot  \left| {{\pi }_{C}{\sigma }_{A = a}T}\right| },
$$

where the inequality follows from (2). We now go one level up and sum the run time over all $b \in  {L}_{B}^{a}$ (but for the same fixed $a \in  {L}_{A}$ as above). This leads to the time taken to compute ${L}_{C}^{a,b}$ over all $b \in  {L}_{B}^{a}$ to be upper bounded by

其中不等式由(2)得出。现在我们再上一层，对所有$b \in  {L}_{B}^{a}$（但与上述相同的固定$a \in  {L}_{A}$）的运行时间求和。这导致计算所有$b \in  {L}_{B}^{a}$上的${L}_{C}^{a,b}$所需的时间上限为

$$
\sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| } \cdot  \mathop{\sum }\limits_{{b \in  {L}_{B}^{a}}}\sqrt{\left| {\pi }_{C}{\sigma }_{B = b}S\right| }
$$

$$
 \leq  \sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| } \cdot  \sqrt{\left| {L}_{B}^{a}\right| } \cdot  \sqrt{\mathop{\sum }\limits_{{b \in  {L}_{B}^{a}}}\left| {{\pi }_{C}{\sigma }_{B = b}S}\right| }
$$

$$
 \leq  \sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| } \cdot  \sqrt{\left| {L}_{B}^{a}\right| } \cdot  \sqrt{\mathop{\sum }\limits_{{b \in  {\pi }_{B}S}}\left| {{\pi }_{C}{\sigma }_{B = b}S}\right| }
$$

$$
 = \sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| } \cdot  \sqrt{\left| {L}_{B}^{a}\right| } \cdot  \sqrt{\left| S\right| }
$$

$$
 = \sqrt{\left| S\right| } \cdot  \sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| } \cdot  \sqrt{\left| {L}_{B}^{a}\right| }
$$

$$
 \leq  \sqrt{\left| S\right| } \cdot  \sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| } \cdot  \sqrt{\left| {\pi }_{B}{\sigma }_{A = a}R\right| },
$$

where the first inequality follows from (3) (where we have $L = {L}_{B}^{a}$ and the vectors are the all 1’s vector and the vector ${\left( \sqrt{\left| {\pi }_{C}{\sigma }_{B = b}S\right| }\right) }_{b \in  {L}_{B}^{a}}$ . The second inequality follows from the fact that ${L}_{B}^{a} \subseteq  {\pi }_{B}S$ . The final inequality follows from the fact that ${L}_{B}^{a} \subseteq  {\pi }_{B}{\sigma }_{A = a}R$ .

其中第一个不等式由(3)得出（其中我们有$L = {L}_{B}^{a}$，向量为全1向量和向量${\left( \sqrt{\left| {\pi }_{C}{\sigma }_{B = b}S\right| }\right) }_{b \in  {L}_{B}^{a}}$）。第二个不等式由${L}_{B}^{a} \subseteq  {\pi }_{B}S$这一事实得出。最后一个不等式由${L}_{B}^{a} \subseteq  {\pi }_{B}{\sigma }_{A = a}R$这一事实得出。

To complete the runtime analysis,we need to sum up the last expression above for every $a \in  {L}_{A}$ . Iowever,note that this sum is exactly the expression (5),which we have already seen is ${N}^{3/2}$ ,as desired.

为了完成运行时间分析，我们需要对每个$a \in  {L}_{A}$对上述最后一个表达式求和。然而，注意到这个和恰好就是表达式(5)，我们已经知道它是${N}^{3/2}$，符合要求。

For completeness, we show how the analysis of Algorithm 2 above follows directly from Lemma 4.1, In this case we use Lemma 4.1 with $I = {I}_{2}$ ,which implies

为了完整起见，我们展示上述算法2的分析如何直接由引理4.1得出。在这种情况下，我们使用引理4.1，其中$I = {I}_{2}$，这意味着

$$
L = R \bowtie  {\pi }_{B}S \bowtie  {\pi }_{A}T = \left\{  {\left( {a,b}\right)  \mid  a \in  {L}_{A},b \in  {L}_{B}^{a}}\right\}  ,
$$

where ${L}_{A}$ and ${L}_{B}^{a}$ is as defined in Algorithm 2,Note that in this case ${\mathcal{E}}_{J} = \{ \{ B,C\} ,\{ A,C\} \}$ . Thus,we have that the LHS in (9) is the same as

其中${L}_{A}$和${L}_{B}^{a}$如算法2中所定义。注意在这种情况下${\mathcal{E}}_{J} = \{ \{ B,C\} ,\{ A,C\} \}$。因此，(9)式的左边与

$$
\mathop{\sum }\limits_{{a \in  {L}_{A}}}\mathop{\sum }\limits_{{b \in  {L}_{B}^{a}}}\sqrt{\left| S \ltimes  \left( a,b\right) \right| } \cdot  \sqrt{T \ltimes  \left( {a,b}\right) }
$$

$$
 = \mathop{\sum }\limits_{{a \in  {L}_{A}}}\mathop{\sum }\limits_{{b \in  {L}_{B}^{a}}}\sqrt{\left| {\pi }_{C}{\sigma }_{B = b}S\right| } \cdot  \sqrt{\left| {\pi }_{C}{\sigma }_{A = a}T\right| }.
$$

Note that the last expression is the same as the one in (13). Lemma 4.1 argues that the above is at most ${N}^{3/2}$ , which is exactly what we proved above.

注意到最后一个表达式与(13)中的表达式相同。引理4.1表明上述表达式至多为${N}^{3/2}$，这正是我们上面所证明的。

## C Technical Tools

## C 技术工具

The following form of Hölder's inequality (also historically attributed to Jensen) can be found in any standard texts on inequalities. The reader is referred to the classic book "Inequalities" by Hardy, Littlewood, and Pólya [26] (Theorem 22 on page 29).

以下形式的赫尔德不等式（历史上也归功于延森）可以在任何关于不等式的标准教材中找到。读者可参考哈代、利特尔伍德和波利亚所著的经典书籍《不等式》[26]（第29页的定理22）。

Lemma C.1 (Hölder inequality). Let $m,n$ be positive integers. Let ${y}_{1},\ldots ,{y}_{n}$ be non-negative real numbers such that ${y}_{1} + \cdots  + {y}_{n} \geq  1$ . Let ${a}_{ij} \geq  0$ be non-negative real numbers,for $i \in  \left\lbrack  m\right\rbrack$ and $j \in  \left\lbrack  n\right\rbrack$ . With the

引理C.1（赫尔德不等式）。设$m,n$为正整数。设${y}_{1},\ldots ,{y}_{n}$为非负实数，使得${y}_{1} + \cdots  + {y}_{n} \geq  1$。设${a}_{ij} \geq  0$为非负实数，其中$i \in  \left\lbrack  m\right\rbrack$且$j \in  \left\lbrack  n\right\rbrack$。根据

convention ${0}^{0} = 0$ ,we have:

约定${0}^{0} = 0$，我们有：

$$
\mathop{\sum }\limits_{{i = 1}}^{m}\mathop{\prod }\limits_{{j = 1}}^{n}{a}_{ij}^{{y}_{j}} \leq  \mathop{\prod }\limits_{{j = 1}}^{n}{\left( \mathop{\sum }\limits_{{i = 1}}^{m}{a}_{ij}\right) }^{{y}_{j}}. \tag{13}
$$

## D Entropy and Alternate Derivations

## D 熵与其他推导

### D.1 Entropy and Shearer's inequality

### D.1 熵与希勒不等式

For basic background knowledge on information theory and the entropy function in particular, the reader is referred to Thomas and Cover [11]. We are necessarily brief in this section.

关于信息论的基础背景知识，尤其是熵函数的相关知识，读者可参考托马斯（Thomas）和科弗（Cover）的文献[11]。本节内容必然较为简略。

Let $X$ be a discrete random variable taking on values from a domain $\mathcal{X}$ with probability mass function ${p}_{X}$ . The (binary) entropy function $H\left\lbrack  X\right\rbrack$ is a measure of the degree of uncertainty associated with $X$ ,and is

设$X$为一个离散随机变量，其取值来自定义域$\mathcal{X}$，概率质量函数为${p}_{X}$。（二进制）熵函数$H\left\lbrack  X\right\rbrack$是衡量与$X$相关的不确定程度的指标，其定义为

defined by

定义如下

$$
H\left\lbrack  X\right\rbrack   \mathrel{\text{:=}}  - \mathop{\sum }\limits_{{x \in  \mathcal{X}}}{p}_{X}\left( x\right) {\log }_{2}{p}_{X}\left( x\right) .
$$

We can replace $X$ by a tuple $\mathbf{X} = \left( {{X}_{1},\ldots ,{X}_{n}}\right)$ of random variables and define the joint entropy in exactly the same way. The only difference in the formula above is the replacement of ${p}_{X}$ by the joint probability mass function.

我们可以用随机变量的元组$\mathbf{X} = \left( {{X}_{1},\ldots ,{X}_{n}}\right)$替换$X$，并以完全相同的方式定义联合熵。上述公式唯一的区别在于用联合概率质量函数替换${p}_{X}$。

Let $X,Y$ be two discrete random variables on domains $\mathcal{X},\mathcal{Y}$ ,respectively. The conditional entropy function of $X$ given $Y$ measures the degree of uncertainty about $X$ given that we knew $Y$ :

设$X,Y$分别为定义域$\mathcal{X},\mathcal{Y}$上的两个离散随机变量。给定$Y$时$X$的条件熵函数衡量了在已知$Y$的情况下关于$X$的不确定程度：

$$
H\left\lbrack  {X \mid  Y}\right\rbrack   \mathrel{\text{:=}}  - \mathop{\sum }\limits_{\substack{{x \in  \mathcal{X}} \\  {y \in  \mathcal{Y}} }}\mathrm{P}\left\lbrack  {X = x,Y = y}\right\rbrack  {\log }_{2}\mathrm{P}\left\lbrack  {X = x \mid  Y = y}\right\rbrack  .
$$

Again,we extend the above definition in the natural way when $X$ and $Y$ are replaced by tuples of random variables. Many simple relations regarding entropy and conditional entropy can be derived from first principle, and they often have very intuitive interpretation. For example, the inequality

同样，当$X$和$Y$被随机变量的元组替换时，我们以自然的方式扩展上述定义。许多关于熵和条件熵的简单关系可以从基本原理推导得出，并且它们通常具有非常直观的解释。例如，不等式

$$
H\left\lbrack  {X \mid  Y,Z}\right\rbrack   \leq  H\left\lbrack  {X \mid  Y}\right\rbrack  
$$

can be intuitively "explained" by thinking that knowing less (only $Y$ as opposed to knowing both $Y$ and $Z$ ) leads to more uncertainty about $X$ . Similarly,the following formula can easily be derived from first principles.

可以直观地“解释”为，知道的信息越少（只知道$Y$而不是同时知道$Y$和$Z$）会导致对$X$的不确定性越大。类似地，以下公式可以很容易地从基本原理推导得出。

$$
H\left\lbrack  {{X}_{1},\ldots ,{X}_{n}}\right\rbrack   = \mathop{\sum }\limits_{{j = 1}}^{n}H\left\lbrack  {{H}_{j} \mid  {X}_{1},\ldots ,{X}_{j - 1}}\right\rbrack  . \tag{14}
$$

The above basic observation can be used to prove Shearer's inequality below. Our proof here follows the conditional entropy approach from Radhakrishnan [34]. The version of Shearer's lemma below is slightly more direct than the version used in [3, 22], leading to a shorter proof of AGM's inequality in Section D.2.

上述基本观察结果可用于证明下面的希勒不等式（Shearer's inequality）。我们这里的证明采用了拉达克里什南（Radhakrishnan）[34]提出的条件熵方法。下面的希勒引理（Shearer's lemma）版本比文献[3, 22]中使用的版本更直接，从而使得在D.2节中对AGM不等式的证明更简短。

Lemma D.1 (Shearer [10]). Let ${X}_{1},\ldots ,{X}_{n}$ be $n$ random variables. For each subset $F \subseteq  \left\lbrack  n\right\rbrack$ ,let ${\mathbf{X}}_{F} =$ ${\left( {X}_{i}\right) }_{i \in  F}$ ; and,let $\mathbf{X} = {X}_{\left\lbrack  n\right\rbrack  }$ . Let $\mathcal{H} = \left( {\mathcal{V} = \left\lbrack  n\right\rbrack  ,\mathcal{E}}\right)$ be a hypergraph,and $\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$ be any fractional edge cover of the hypergraph. Then, the following inequality holds

引理D.1（希勒（Shearer）[10]）。设${X}_{1},\ldots ,{X}_{n}$为$n$个随机变量。对于每个子集$F \subseteq  \left\lbrack  n\right\rbrack$，设${\mathbf{X}}_{F} =$ ${\left( {X}_{i}\right) }_{i \in  F}$；并且，设$\mathbf{X} = {X}_{\left\lbrack  n\right\rbrack  }$。设$\mathcal{H} = \left( {\mathcal{V} = \left\lbrack  n\right\rbrack  ,\mathcal{E}}\right)$为一个超图，$\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$为该超图的任意分数边覆盖。则以下不等式成立

$$
H\left\lbrack  \mathbf{X}\right\rbrack   \leq  \mathop{\sum }\limits_{{F \in  \mathcal{E}}}{x}_{F} \cdot  H\left\lbrack  {\mathbf{X}}_{F}\right\rbrack   \tag{15}
$$

Proof. For every $F \in  \mathcal{E}$ and $j \in  F$ ,we have

证明。对于每一个$F \in  \mathcal{E}$和$j \in  F$，我们有

$$
H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j}\right\rbrack   \leq  H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j,i \in  F}\right\rbrack  
$$

because the entropy on left hand side is conditioned on a (perhaps non-strict) superset of the variables conditioned on the right hand side. Additionally,because the vector $\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$ is a fractional edge cover, for every $j \in  \left\lbrack  n\right\rbrack$ we have $\mathop{\sum }\limits_{{F \in  \mathcal{E},j \in  F}}{x}_{F} \geq  1$ . It follows that,for every $j \in  \left\lbrack  n\right\rbrack$ ,

因为左边的熵是基于右边所基于的变量的（可能是非严格的）超集进行条件设定的。此外，由于向量$\mathbf{x} = {\left( {x}_{F}\right) }_{F \in  \mathcal{E}}$是一个分数边覆盖，对于每一个$j \in  \left\lbrack  n\right\rbrack$，我们有$\mathop{\sum }\limits_{{F \in  \mathcal{E},j \in  F}}{x}_{F} \geq  1$。由此可知，对于每一个$j \in  \left\lbrack  n\right\rbrack$，

$$
H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j}\right\rbrack   \leq  \mathop{\sum }\limits_{{F \in  \mathcal{E},j \in  F}}{x}_{F} \cdot  H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j,i \in  F}\right\rbrack  .
$$

From this inequality and formula (14), we obtain

从这个不等式和公式（14），我们可以得到

$$
H\left\lbrack  \mathbf{X}\right\rbrack   = \mathop{\sum }\limits_{{j = 1}}^{n}H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j}\right\rbrack  
$$

$$
 \leq  \mathop{\sum }\limits_{{j = 1}}^{n}\mathop{\sum }\limits_{{F \in  \mathcal{E},j \in  F}}{x}_{F} \cdot  H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j,i \in  F}\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{F \in  \mathcal{E}}}{x}_{F} \cdot  \mathop{\sum }\limits_{{j \in  F}}H\left\lbrack  {{X}_{j} \mid  {X}_{i},i < j,i \in  F}\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{F \in  \mathcal{E}}}{x}_{F} \cdot  H\left\lbrack  {\mathbf{X}}_{F}\right\rbrack  
$$

### D.2 AGM's proof based on Shearer's entropy inequality and a join-project plan

### D.2 基于希勒熵不等式和连接 - 投影计划的AGM证明

AGM's inequality was shown in [3, 23] as follows. A similar approach was used in [14] to prove an essentially equivalent inequality regarding the number of copies of a hypergraph in another. Let $\mathbf{X} = {\left( {X}_{v}\right) }_{v \in  \mathcal{V}}$ denote a uniformly chosen random tuple from the output of the query $Q$ . Then, $H\left\lbrack  \mathbf{X}\right\rbrack   = {\log }_{2}\left| Q\right|$ . Note that each ${X}_{v},v \in  \mathcal{V}$ is a random variable. For each $F \in  \mathcal{E}$ ,let ${\mathbf{X}}_{F} = {\left( {X}_{v}\right) }_{v \in  F}$ . Then,the random variables ${\mathbf{X}}_{F}$ takes on values in ${R}_{F}$ . Because the uniform distribution has the maximum entropy,we have $H\left\lbrack  {\mathbf{X}}_{F}\right\rbrack   \leq  {\log }_{2}\left| {R}_{F}\right|$ , for all $F \in  \mathcal{E}$ .

AGM不等式（AGM's inequality）在文献[3, 23]中表述如下。文献[14]采用了类似的方法来证明一个关于一个超图在另一个超图中副本数量的本质上等价的不等式。设$\mathbf{X} = {\left( {X}_{v}\right) }_{v \in  \mathcal{V}}$表示从查询$Q$的输出中均匀选择的一个随机元组。那么，$H\left\lbrack  \mathbf{X}\right\rbrack   = {\log }_{2}\left| Q\right|$。注意，每个${X}_{v},v \in  \mathcal{V}$都是一个随机变量。对于每个$F \in  \mathcal{E}$，设${\mathbf{X}}_{F} = {\left( {X}_{v}\right) }_{v \in  F}$。那么，随机变量${\mathbf{X}}_{F}$的取值范围是${R}_{F}$。由于均匀分布具有最大熵，对于所有的$F \in  \mathcal{E}$，我们有$H\left\lbrack  {\mathbf{X}}_{F}\right\rbrack   \leq  {\log }_{2}\left| {R}_{F}\right|$。

Let $\mathbf{x}$ denote any fractional edge cover for the hypergraph $\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$ of the query $Q$ ,then from Shearer's inequality (15) we obtain

设$\mathbf{x}$表示查询$Q$的超图$\mathcal{H} = \left( {\mathcal{V},\mathcal{E}}\right)$的任意分数边覆盖（fractional edge cover），那么根据希勒不等式（Shearer's inequality）(15)，我们可以得到

$$
{\log }_{2}\left| Q\right|  = H\left\lbrack  \mathbf{X}\right\rbrack   \leq  \mathop{\sum }\limits_{{F \in  \mathcal{E}}}{x}_{F} \cdot  H\left\lbrack  {\mathbf{X}}_{F}\right\rbrack   \leq  \mathop{\sum }\limits_{{F \in  \mathcal{E}}}{x}_{F} \cdot  {\log }_{2}\left| {R}_{F}\right| ,
$$

which is exactly AGM's inequality (6).

这正是AGM不等式(6)。

Proposition D.2. For any query $Q$ ,there is a join-project plan with runtime $O\left( {{\left| Q\right| }^{2} \cdot  {2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) } \cdot  N}\right)$ for evaluating $Q$ ,where $N$ is the input size. Proof. We define the join-project plan recursively. Let ${A}_{1},\ldots ,{A}_{n}$ be the attributes and ${R}_{1},\ldots ,{R}_{m}$ be the relations of $Q$ . Let ${\bar{B}}_{n - 1} = \left( {{A}_{1},\ldots ,{A}_{n - 1}}\right)$ . We first recursively compute the join $P = { \bowtie  }_{F \in  \mathcal{E}}{\pi }_{{\bar{B}}_{n - 1}}\left( {R}_{F}\right)$ . Then, we output

命题D.2。对于任何查询$Q$，存在一个连接 - 投影计划（join - project plan），其运行时间为$O\left( {{\left| Q\right| }^{2} \cdot  {2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) } \cdot  N}\right)$，用于计算$Q$，其中$N$是输入规模。证明：我们递归地定义连接 - 投影计划。设${A}_{1},\ldots ,{A}_{n}$是$Q$的属性，${R}_{1},\ldots ,{R}_{m}$是$Q$的关系。设${\bar{B}}_{n - 1} = \left( {{A}_{1},\ldots ,{A}_{n - 1}}\right)$。我们首先递归地计算连接$P = { \bowtie  }_{F \in  \mathcal{E}}{\pi }_{{\bar{B}}_{n - 1}}\left( {R}_{F}\right)$。然后，我们输出

$$
Q = \left( {\cdots \left( {\left( {P \bowtie  {\pi }_{{A}_{n}}\left( {R}_{1}\right) }\right)  \bowtie  {\pi }_{{A}_{n}}\left( {R}_{2}\right) }\right)  \bowtie  \cdots  \bowtie  {\pi }_{{A}_{n}}\left( {R}_{m}\right) }\right) .
$$

The base case is when $n = 1$ which is the simple $m$ set intersection problem. It is easy to see that ${\rho }^{ * }\left( {P,\mathcal{D}}\right)  \leq$ ${\rho }^{ * }\left( {Q,\mathcal{D}}\right)$ ,because any fractional edge cover for $Q$ is also a fractional edge cover for $P$ . Hence,all the intermediate results in computing $Q$ from $P$ has sizes at most $\left| P\right|  \cdot  N \leq  {2}^{{\rho }^{ * }\left( {P,\mathcal{D}}\right) }N \leq  {2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) } \cdot  N$ . From there, the claimed run-time follows.

基本情况是当$n = 1$时，这是一个简单的$m$集合交集问题。很容易看出${\rho }^{ * }\left( {P,\mathcal{D}}\right)  \leq$ ${\rho }^{ * }\left( {Q,\mathcal{D}}\right)$，因为$Q$的任何分数边覆盖也是$P$的分数边覆盖。因此，从$P$计算$Q$的所有中间结果的规模至多为$\left| P\right|  \cdot  N \leq  {2}^{{\rho }^{ * }\left( {P,\mathcal{D}}\right) }N \leq  {2}^{{\rho }^{ * }\left( {Q,\mathcal{D}}\right) } \cdot  N$。由此可得所声称的运行时间。

## E A more formal description of GLVV results

## E 对GLVV结果的更正式描述

From examples in Section 3.4, we can describe GLVV's strategy for obtaining size bounds for general conjunctive queries with simple functional dependencies. Let $C$ be such query. The general idea is to construct a natural join query $Q$ such that $\left| C\right|  \leq  \left| Q\right|$ and thus we can simply apply AGM bound on $Q$ .

从3.4节的例子中，我们可以描述GLVV用于获得具有简单函数依赖的一般合取查询（conjunctive queries）规模边界的策略。设$C$是这样的查询。总体思路是构造一个自然连接查询$Q$，使得$\left| C\right|  \leq  \left| Q\right|$，这样我们就可以直接对$Q$应用AGM边界。

- The first step is to turn the conjunctive query $C$ into ${C}_{1} = \operatorname{chase}\left( C\right)$ .

- 第一步是将合取查询$C$转换为${C}_{1} = \operatorname{chase}\left( C\right)$。

This step can be done in $O\left( {\left| C\right| }^{4}\right)$ -time [1,2,4,30]. This is the step in the spirit of Example 5,

这一步可以在$O\left( {\left| C\right| }^{4}\right)$时间内完成 [1,2,4,30]。这一步遵循示例 5 的思路。

- Next,let ${C}_{2}$ be obtained from ${C}_{1}$ by replacing each repeated relation symbol by a fresh relation symbol. Thus,in ${C}_{2}$ there is no duplicate relation. (In the join algorithm itself,all we have to do is to have them point to the same underlying index.)

- 接下来，通过用新的关系符号替换${C}_{1}$中的每个重复关系符号来得到${C}_{2}$。因此，在${C}_{2}$中没有重复的关系。（在连接算法本身中，我们所要做的就是让它们指向相同的底层索引。）

- The next step is in the spirit of Example 6. Recall that a simple functional dependency (FD) is of the form $R\left\lbrack  i\right\rbrack   \rightarrow  R\left\lbrack  j\right\rbrack$ ,which means the $j$ th attribute of $R$ is functionally determined by the $i$ th attribute. From the set of given simple FDs, we can obtain a set of FDs in terms of the variables in the query ${C}_{2}$ . For example,if $R\left\lbrack  3\right\rbrack   \rightarrow  R\left\lbrack  1\right\rbrack$ is a given FD and $R\left( {XYZW}\right)$ occurs in the body of the query ${C}_{2}$ , then we obtain a FD of the form $Z \rightarrow  X$ . Now,WLOG assume that $\operatorname{vars}\left( {C}_{2}\right)  = \left\{  {{X}_{1},\ldots ,{X}_{n}}\right\}$ . We repeatedly apply the following steps for each (variable) FD of the form ${X}_{i} \rightarrow  {X}_{j}$ :

- 下一步遵循示例 6 的思路。回想一下，简单函数依赖（FD，Functional Dependency）的形式为$R\left\lbrack  i\right\rbrack   \rightarrow  R\left\lbrack  j\right\rbrack$，这意味着$R$的第$j$个属性由第$i$个属性函数确定。从给定的简单函数依赖集合中，我们可以根据查询${C}_{2}$中的变量得到一组函数依赖。例如，如果$R\left\lbrack  3\right\rbrack   \rightarrow  R\left\lbrack  1\right\rbrack$是一个给定的函数依赖，并且$R\left( {XYZW}\right)$出现在查询${C}_{2}$的主体中，那么我们可以得到一个形式为$Z \rightarrow  X$的函数依赖。现在，不失一般性地假设$\operatorname{vars}\left( {C}_{2}\right)  = \left\{  {{X}_{1},\ldots ,{X}_{n}}\right\}$。我们对每个形式为${X}_{i} \rightarrow  {X}_{j}$的（变量）函数依赖重复应用以下步骤：

- For every atom $R$ in which ${X}_{i}$ appears but ${X}_{j}$ does not,add ${X}_{j}$ to $R$ . (This is a fictitious relation - and in the join algorithm we do not need to physically add a new field to the data; rather, we keep a pointer to where ${X}_{j}$ can be found when needed.)

- 对于每个包含${X}_{i}$但不包含${X}_{j}$的原子$R$，将${X}_{j}$添加到$R$中。（这是一个虚拟关系——在连接算法中，我们不需要在物理上向数据中添加新字段；相反，我们保留一个指针，以便在需要时可以找到${X}_{j}$的位置。）

- For every FD of the form ${X}_{k} \rightarrow  {X}_{i}$ ,we add ${X}_{k} \rightarrow  {X}_{j}$ as a new FD.

- 对于每个形式为${X}_{k} \rightarrow  {X}_{i}$的函数依赖，我们添加${X}_{k} \rightarrow  {X}_{j}$作为一个新的函数依赖。

- Remove the FD ${X}_{i} \rightarrow  {X}_{j}$ .

- 移除函数依赖${X}_{i} \rightarrow  {X}_{j}$。

It is easy to see that performing the above in a canonical order will terminate in about $O\left( {n}^{2}\right)$ time,and the new query ${C}_{3}$ (with some fictitious relations "blown up" with more attributes) is exactly equivalent to the the old query ${C}_{2}$ .

很容易看出，按规范顺序执行上述操作将在大约$O\left( {n}^{2}\right)$时间内终止，并且新查询${C}_{3}$（其中一些虚拟关系通过更多属性“扩展”）与旧查询${C}_{2}$完全等价。

- Next, we can remove repeated variables from relations by keeping only tuples that match the repetition patterns. For example,if there is an atom $R\left( {XXY}\right)$ in the query ${C}_{3}$ ,then we only have to keep tuples $\left( {{t}_{1},{t}_{2},{t}_{3}}\right)$ in $R$ for which ${t}_{1} = {t}_{2}$ . After this step,we can replace $R$ by a relation ${R}^{\prime }\left( {XY}\right)$ . This step is in the spirit of Example 4. We call the resulting query ${C}_{4}$ .

- 接下来，我们可以通过仅保留与重复模式匹配的元组来从关系中移除重复变量。例如，如果查询${C}_{3}$中有一个原子$R\left( {XXY}\right)$，那么我们只需要保留$R$中满足${t}_{1} = {t}_{2}$的元组$\left( {{t}_{1},{t}_{2},{t}_{3}}\right)$。在这一步之后，我们可以用关系${R}^{\prime }\left( {XY}\right)$替换$R$。这一步遵循示例 4 的思路。我们将得到的查询称为${C}_{4}$。

- Finally,we turn ${C}_{4}$ into $Q$ by "projecting out" all the attributes not in the head atom as was shown in Example 3. Since $Q$ is now a join query, $\mathrm{{AGM}}$ bounds applies. What is remarkable is that GLVV showed that the bound obtained this way is essentially tight up to a factor which is data-independent.

- 最后，如示例 3 所示，通过“投影出”头原子中不存在的所有属性，我们将${C}_{4}$转换为$Q$。由于$Q$现在是一个连接查询，$\mathrm{{AGM}}$界适用。值得注意的是，GLVV 表明，以这种方式得到的界在一个与数据无关的因子范围内基本上是紧的。

Note again that our description above is essentially what was obtained from GLVV, without the coloring number machinery. Furthermore,if $C$ was a full conjunctive query,then we don’t have to project away any attribute and thus any worst-case join algorithm described in the previous section can be applied which is still worst-case optimal!

再次注意，我们上面的描述本质上是从 GLVV 那里得到的，没有使用着色数机制。此外，如果$C$是一个完全合取查询，那么我们不需要投影掉任何属性，因此上一节中描述的任何最坏情况连接算法都可以应用，并且仍然是最坏情况最优的！

## F Proof of Lemma 4.2

## F 引理 4.2 的证明

Proof of Lemma 4.2 In the instances below the domain of any attribute is

引理 4.2 的证明 在以下实例中，任何属性的域是

$$
\mathcal{D} = \{ 0,1,\ldots ,\left( {N - 1}\right) /\left( {n - 1}\right) \} ,
$$

where we ignore the integrality issue for the sake of clarify. For $i \in  \left\lbrack  n\right\rbrack$ ,let ${R}_{i}$ denote the set of all tuples in ${\mathcal{D}}^{\left\lbrack  n\right\rbrack  -\{ i\} }$ each of which has at most one non-zero value. It follows that,for all $i \in  \left\lbrack  n\right\rbrack$ ,

为了清晰起见，我们忽略整数性问题。对于$i \in  \left\lbrack  n\right\rbrack$，设${R}_{i}$表示${\mathcal{D}}^{\left\lbrack  n\right\rbrack  -\{ i\} }$中所有元组的集合，其中每个元组最多有一个非零值。由此可知，对于所有的$i \in  \left\lbrack  n\right\rbrack$，

$$
\left| {R}_{i}\right|  = \left( {n - 1}\right) \left\lbrack  {\left( {N - 1}\right) /\left( {n - 1}\right)  + 1}\right\rbrack   - \left( {n - 2}\right)  = N,
$$

and that

并且

$$
\left| {{ \bowtie  }_{i = 1}^{n}{R}_{i}}\right|  = n\left\lbrack  {\left( {N - 1}\right) /\left( {n - 1}\right)  + 1}\right\rbrack   - \left( {n - 1}\right) 
$$

$$
 = N + \left( {N - 1}\right) /\left( {n - 1}\right) \text{.}
$$

(We remark that the instance above was specialized to $n = 3$ in Section 2.)

（我们注意到，上述实例在第2节中专门针对$n = 3$进行了讨论。）

A relation $R$ on attribute set $\bar{A} \subseteq  \left\lbrack  n\right\rbrack$ is called "simple" if $R$ is the set of all tuples in ${\mathcal{D}}^{\bar{A}}$ each of which has at most one non-zero value. Then, we observe the following properties.

如果属性集$\bar{A} \subseteq  \left\lbrack  n\right\rbrack$上的关系$R$是${\mathcal{D}}^{\bar{A}}$中所有元组的集合，且每个元组最多有一个非零值，则称该关系为“简单”关系。然后，我们观察到以下性质。

(a) The input relations ${R}_{i}$ are simple.

(a) 输入关系${R}_{i}$是简单关系。

(b) An arbitrary projection of a simple relation is simple.

(b) 简单关系的任意投影都是简单关系。

(c) Let $S$ and $T$ be any two simple relations on attribute sets ${\bar{A}}_{S}$ and ${\bar{A}}_{T}$ ,respectively. If ${\bar{A}}_{S}$ is contained in ${\bar{A}}_{T}$ or vice versa,then $S \bowtie  T$ is simple. If neither ${\bar{A}}_{S}$ nor ${\bar{A}}_{T}$ is contained in the other,then $\left| {S \bowtie  T}\right|  \geq  {\left( 1 + \left( N - 1\right) /\left( n - 1\right) \right) }^{2} = \Omega \left( {{N}^{2}/{n}^{2}}\right) .$

(c) 设$S$和$T$分别是属性集${\bar{A}}_{S}$和${\bar{A}}_{T}$上的任意两个简单关系。如果${\bar{A}}_{S}$包含于${\bar{A}}_{T}$，或者反之，则$S \bowtie  T$是简单关系。如果${\bar{A}}_{S}$和${\bar{A}}_{T}$都不包含于对方，则$\left| {S \bowtie  T}\right|  \geq  {\left( 1 + \left( N - 1\right) /\left( n - 1\right) \right) }^{2} = \Omega \left( {{N}^{2}/{n}^{2}}\right) .$

For an arbitrary join-project plan starting from the simple relations ${R}_{i}$ ,we eventually must join two relations whose attribute sets are not contained in one another,which right then requires $\Omega \left( {{N}^{2}/{n}^{2}}\right)$ run time.

对于从简单关系${R}_{i}$开始的任意连接 - 投影计划，我们最终必须连接两个属性集互不包含的关系，此时需要$\Omega \left( {{N}^{2}/{n}^{2}}\right)$的运行时间。