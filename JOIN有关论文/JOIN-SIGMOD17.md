# The Dynamic Yannakakis Algorithm: Compact and Efficient Query Processing Under Updates

# 动态亚纳卡基斯算法：更新环境下的紧凑高效查询处理

Muhammad Idris

穆罕默德·伊德里斯

Université Libre de Bruxelles and TU Dresden

布鲁塞尔自由大学和德累斯顿工业大学

muhammad.idris@ulb.ac.be

Martín Ugarte

马丁·乌加特

Université Libre de Bruxelles

布鲁塞尔自由大学

mugartec@ulb.ac.be

Stijn Vansummeren

斯廷·范萨默伦

Université Libre de Bruxelles

布鲁塞尔自由大学

stijn.vansummeren@ulb.ac.be

## ABSTRACT

## 摘要

Modern computing tasks such as real-time analytics require refresh of query results under high update rates. Incremental View Maintenance (IVM) approaches this problem by materializing results in order to avoid recomputation. IVM naturally induces a trade-off between the space needed to maintain the materialized results and the time used to process updates. In this paper, we show that the full materialization of results is a barrier for more general optimization strategies. In particular, we present a new approach for evaluating queries under updates. Instead of the materialization of results, we require a data structure that allows: (1) linear time maintenance under updates, (2) constant-delay enumeration of the output, (3) constant-time lookups in the output, while (4) using only linear space in the size of the database. We call such a structure a Dynamic Constant-delay Linear Representation (DCLR) for the query. We show that Dyn, a dynamic version of the Yannakakis algorithm, yields DCLRs for the class of free-connex acyclic CQs. We show that this is optimal in the sense that no DCLR can exist for CQs that are not free-connex acyclic. Moreover, we identify a sub-class of queries for which DYN features constant-time update per tuple and show that this class is maximal. Finally, using the TPC-H and TPC-DS benchmarks, we experimentally compare DYN and a higher-order IVM (HIVM) engine. Our approach is not only more efficient in terms of memory consumption (as expected), but is also consistently faster in processing updates.

诸如实时分析等现代计算任务需要在高更新率下刷新查询结果。增量视图维护（Incremental View Maintenance，IVM）通过物化结果来避免重新计算，从而解决这一问题。IVM自然会在维护物化结果所需的空间和处理更新所用的时间之间产生权衡。在本文中，我们表明结果的完全物化是更通用优化策略的障碍。特别是，我们提出了一种在更新环境下评估查询的新方法。我们不进行结果的物化，而是需要一种数据结构，该结构允许：（1）在更新时进行线性时间维护；（2）对输出进行恒定延迟枚举；（3）在输出中进行恒定时间查找；（4）仅使用与数据库大小成线性关系的空间。我们将这种结构称为查询的动态恒定延迟线性表示（Dynamic Constant-delay Linear Representation，DCLR）。我们证明，亚纳卡基斯算法的动态版本Dyn为自由连通无环合取查询（free-connex acyclic CQs）类生成DCLR。我们表明，从非自由连通无环的合取查询不存在DCLR的意义上来说，这是最优的。此外，我们确定了一类查询，对于这类查询，DYN具有每个元组恒定时间更新的特点，并表明这类查询是最大的。最后，使用TPC - H和TPC - DS基准测试，我们通过实验比较了DYN和高阶增量视图维护（Higher - Order IVM，HIVM）引擎。我们的方法不仅在内存消耗方面更高效（如预期），而且在处理更新时也始终更快。

## KEYWORDS

## 关键词

Incremental View Maintenance; Dynamic query processing; Acyclic joins

增量视图维护；动态查询处理；无环连接

## 1. INTRODUCTION

## 1. 引言

Real-time analytics find applications in Financial Systems , Industrial Control Systems, Business Intelligence and Online Machine Learning, among many others (see [15] for a survey). Generally, the analytical results that need to be kept up-to-date, or at least their basic elements, are specified in a query language. The main task is then to efficiently update the query results under frequent data updates.

实时分析在金融系统、工业控制系统、商业智能和在线机器学习等众多领域都有应用（相关综述见[15]）。通常，需要保持最新状态的分析结果，或者至少是其基本元素，是用查询语言指定的。那么主要任务就是在频繁的数据更新下高效地更新查询结果。

In this paper, we focus on the problem of dynamic query evaluation,where a given query $Q$ has to be evaluated against a database that is constantly updated. In this setting, when database ${db}$ is updated to database ${db} + u$ under update $u$ ,the objective is to efficiently compute $Q\left( {{db} + u}\right)$ ,taking into consideration that $Q\left( {db}\right)$ was already evaluated and re-computations could be avoided. Dynamic query evaluation has traditionally been approached from Incremental View Maintenance (IVM) [13]. IVM techniques materialize $Q\left( {db}\right)$ and evaluate delta queries. These take as input ${db}$ , $u$ and the materialized $Q\left( {db}\right)$ ,and return the set of tuples to add/delete from $Q\left( {db}\right)$ to obtain $Q\left( {{db} + u}\right)$ . If $u$ is small w.r.t. ${db}$ ,this is expected to be faster than recomputing $Q\left( {{db} + u}\right)$ from scratch. Research in this area has recently received a big boost with the introduction of Higher-Order IVM (HIVM) [25,26,30]. Given a query $Q$ ,HIVM not only defines the delta query ${\Delta Q}$ ,but also materializes it. Mo re-over, it defines higher-order delta queries (i.e., delta queries for delta queries,denoted ${\Delta }^{2}Q,{\Delta }^{3}Q,\ldots$ ),where every ${\Delta }^{j}Q$ describes how the materialization of ${\Delta }^{j - 1}Q$ should change under updates. This method is highly efficient in practice, and is formally in a lower complexity class than IVM [25].

在本文中，我们关注动态查询评估问题，即需要针对不断更新的数据库评估给定查询$Q$。在这种情况下，当数据库${db}$在更新$u$的作用下更新为数据库${db} + u$时，目标是高效地计算$Q\left( {{db} + u}\right)$，同时要考虑到$Q\left( {db}\right)$已经被评估过，并且可以避免重新计算。传统上，动态查询评估是从增量视图维护（IVM）[13]的角度来处理的。IVM技术物化$Q\left( {db}\right)$并评估增量查询。这些增量查询以${db}$、$u$和物化的$Q\left( {db}\right)$为输入，并返回要从$Q\left( {db}\right)$中添加/删除的元组集合，以得到$Q\left( {{db} + u}\right)$。如果相对于${db}$而言，$u$较小，那么预计这比从头重新计算$Q\left( {{db} + u}\right)$要快。随着高阶增量视图维护（Higher - Order IVM，HIVM）[25,26,30]的引入，该领域的研究最近得到了极大的推动。给定一个查询$Q$，HIVM不仅定义了增量查询${\Delta Q}$，还对其进行物化。此外，它还定义了高阶增量查询（即增量查询的增量查询，记为${\Delta }^{2}Q,{\Delta }^{3}Q,\ldots$），其中每个${\Delta }^{j}Q$描述了在更新时${\Delta }^{j - 1}Q$的物化应该如何变化。这种方法在实践中非常高效，并且在形式上比IVM具有更低的复杂度类[25]。

(H)IVM present important drawbacks, however. First, materialization of $Q\left( {db}\right)$ requires $\Omega \left( {\parallel Q\left( {db}\right) \parallel }\right)$ space,where $\parallel {db}\parallel$ denotes the size of ${db}$ . Therefore,when $Q\left( {db}\right)$ is large compared to ${db}$ ,materializing $Q\left( {db}\right)$ quickly becomes impractical, especially for main-memory based systems. HIVM is even more affected by this problem than IVM since it not only materializes the result of $Q$ but also the results of the higher-order delta queries. Second, IVM and HIVM only exploit the information provided by the materialized views to process updates, while additional forms of information could result in better update rates. Consider for example the query $Q = R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$ and a database with $N$ tuples in $R$ and $N$ tuples in $S$ ,all with the same $B$ value. The materialization of $Q\left( {db}\right)$ in this case uses $\Theta \left( {N}^{2}\right)$ space and is useless for re-computing $Q$ under updates. In contrast,a simple index on $B$ for $R$ and $S$ would allow for efficient enumeration of the set of tuples that need to be added/removed from $Q\left( {db}\right)$ to obtain $Q\left( {{db} + u}\right)$ . It is important to note that even for queries whose result is smaller than the database, aggressive materialization of higher-order delta queries in HIVM can still cause these problems to appear. Indeed, some higher-order delta queries are partial join results,which can be larger than both ${db}$ and $Q\left( {db}\right)$ .

然而，(高阶)增量视图维护((H)IVM)存在重要的缺点。首先，物化$Q\left( {db}\right)$需要$\Omega \left( {\parallel Q\left( {db}\right) \parallel }\right)$的空间，其中$\parallel {db}\parallel$表示${db}$的大小。因此，当$Q\left( {db}\right)$相对于${db}$很大时，物化$Q\left( {db}\right)$很快就变得不切实际，特别是对于基于主内存的系统。高阶增量视图维护(HIVM)比增量视图维护(IVM)更受这个问题的影响，因为它不仅物化$Q$的结果，还物化高阶增量查询的结果。其次，增量视图维护(IVM)和高阶增量视图维护(HIVM)仅利用物化视图提供的信息来处理更新，而其他形式的信息可能会带来更好的更新率。例如，考虑查询$Q = R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$和一个数据库，其中$R$中有$N$个元组，$S$中也有$N$个元组，所有元组的$B$值都相同。在这种情况下，物化$Q\left( {db}\right)$使用$\Theta \left( {N}^{2}\right)$的空间，并且在更新时重新计算$Q$时毫无用处。相比之下，为$R$和$S$的$B$建立一个简单的索引，就可以有效地枚举需要从$Q\left( {db}\right)$中添加/删除的元组集合，以得到$Q\left( {{db} + u}\right)$。需要注意的是，即使对于结果比数据库小的查询，高阶增量视图维护(HIVM)中高阶增量查询的激进物化仍然可能导致这些问题出现。实际上，一些高阶增量查询是部分连接结果，它们可能比${db}$和$Q\left( {db}\right)$都大。

While these problems are inherent to (H)IVM methods based on materialization, they can be avoided by taking a different approach to dynamic query evaluation: instead of maintaining $Q\left( {db}\right)$ ,we could maintain a data structure from which $Q\left( {db}\right)$ can be generated as efficiently as if it were materialized. This notion is formalized by the theoretical database community by requiring that the output $Q\left( {db}\right)$ can be enumerated from the data structure with constant delay [5]. Intuitively, data structures that feature constant-delay enumeration (CDE for short) are aimed at representing data in compressed form yet have a streaming decompression algorithm that can spend only a constant amount of work to produce each new output tuple [37].

虽然这些问题是基于物化的(高阶)增量视图维护((H)IVM)方法所固有的，但可以通过采用不同的动态查询评估方法来避免这些问题：我们可以维护一个数据结构，从中可以像物化$Q\left( {db}\right)$一样高效地生成$Q\left( {db}\right)$，而不是直接维护$Q\left( {db}\right)$。理论数据库界通过要求可以从数据结构中以恒定延迟枚举输出$Q\left( {db}\right)$ [5]，将这一概念形式化。直观地说，具有恒定延迟枚举(CDE，简称)特性的数据结构旨在以压缩形式表示数据，同时拥有一种流式解压缩算法，该算法在生成每个新的输出元组时只需花费恒定的工作量 [37]。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和第一页上的完整引用信息。必须尊重本作品中除作者之外的其他所有者的版权。允许进行带引用的摘要。如需以其他方式复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。

SIGMOD '17 May 14-19, 2017, Chicago, IL, USA

2017年5月14 - 19日，美国伊利诺伊州芝加哥，2017年管理数据国际会议(SIGMOD '17)

(C) 2017 Copyright held by the owner/author(s). Publication rights licensed to ACM. ISBN 978-1-4503-4197-4/17/05...\$15.00

(C) 2017 版权归所有者/作者所有。出版权授权给美国计算机协会(ACM)。国际标准书号(ISBN) 978 - 1 - 4503 - 4197 - 4/17/05... 15.00美元

DOI: http://dx.doi.org/10.1145/3035918.3064027

数字对象标识符(DOI)：http://dx.doi.org/10.1145/3035918.3064027

<!-- Footnote -->

---

While there is increasing work on query evaluation with constant-delay enumeration $\left\lbrack  {5,7,{10},{31},{37}}\right\rbrack$ ,known results either present (involved) theoretical algorithms, or study the static setting without updates. In this paper, therefore, we are concerned with designing a practical algorithm for dynamic query evaluation based on constant-delay enumeration. In particular,to dynamically process a query $Q$ we desire a Dynamic Constant-delay Linear Representation (DCLR) of $Q$ ,meaning that for every database ${db}$ we can compute a data structure ${\mathcal{D}}_{db}$ with the following properties:

虽然关于使用恒定延迟枚举进行查询评估$\left\lbrack  {5,7,{10},{31},{37}}\right\rbrack$的研究越来越多，但已知的结果要么是(复杂的)理论算法，要么是在不考虑更新的静态环境下进行研究。因此，在本文中，我们关注的是设计一种基于恒定延迟枚举的实用动态查询评估算法。具体来说，为了动态处理查询$Q$，我们希望得到$Q$的动态恒定延迟线性表示(DCLR)，这意味着对于每个数据库${db}$，我们可以计算出一个具有以下属性的数据结构${\mathcal{D}}_{db}$：

- $\left( {P}_{1}\right) {\mathcal{D}}_{db}$ allows to enumerate $Q\left( {db}\right)$ with constant delay.

- $\left( {P}_{1}\right) {\mathcal{D}}_{db}$允许以恒定延迟枚举$Q\left( {db}\right)$。

- $\left( {P}_{2}\right)$ For any tuple $\overrightarrow{t}$ ,we can use ${\mathcal{D}}_{db}$ to check whether $\overrightarrow{t} \in  Q\left( {db}\right)$ in constant time.

- $\left( {P}_{2}\right)$ 对于任意元组 $\overrightarrow{t}$ ，我们可以使用 ${\mathcal{D}}_{db}$ 在常量时间内检查 $\overrightarrow{t} \in  Q\left( {db}\right)$ 是否成立。

- $\left( {P}_{3}\right) {\mathcal{D}}_{db}$ requires only $O\left( {\parallel {db}\parallel }\right)$ space. As such ${\mathcal{D}}_{db}$ depends only on ${db}$ and is independent of the size of $Q\left( {db}\right)$ .

- $\left( {P}_{3}\right) {\mathcal{D}}_{db}$ 仅需要 $O\left( {\parallel {db}\parallel }\right)$ 的空间。因此，${\mathcal{D}}_{db}$ 仅取决于 ${db}$ ，并且与 $Q\left( {db}\right)$ 的大小无关。

- $\left( {P}_{4}\right) {\mathcal{D}}_{db}$ features efficient maintenance under updates: given ${\mathcal{D}}_{db}$ and update $u$ to database ${db}$ ,we can compute ${\mathcal{D}}_{{db} + u}$ in time $O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$ . In contrast,both IVM and HIVM may require $\Omega \left( {\parallel u\parallel  + \parallel Q\left( {{db} + u}\right) \parallel }\right)$ time in the worst case.

- $\left( {P}_{4}\right) {\mathcal{D}}_{db}$ 在更新操作下具有高效的维护特性：给定 ${\mathcal{D}}_{db}$ 以及对数据库 ${db}$ 的更新 $u$ ，我们可以在时间 $O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$ 内计算出 ${\mathcal{D}}_{{db} + u}$ 。相比之下，IVM 和 HIVM 在最坏情况下可能需要 $\Omega \left( {\parallel u\parallel  + \parallel Q\left( {{db} + u}\right) \parallel }\right)$ 的时间。

It is important to note that we consider query evaluation in main memory and measure time and space under data complexity [38]. That is, the query is considered to be fixed and not part of the input. This makes sense under dynamic query evaluation, where the query is known in advance and the data is constantly changing. In particular, the number of relations to be queried, their arity, and the length of the query are all constant.

需要注意的是，我们考虑的是主存中的查询评估，并在数据复杂度的框架下衡量时间和空间[38]。也就是说，查询被视为固定的，不属于输入的一部分。这在动态查询评估中是合理的，因为查询是预先已知的，而数据在不断变化。特别是，要查询的关系数量、它们的元数以及查询的长度都是常量。

Our contributions are as follows. We focus on the class of conjunctive aggregate queries (CAQ) evaluated under mul-tiset semantics. Conjunctive aggregate queries are queries that compute aggregates (e.g., SUM, AVG, ...) over the result of a conjunctive query (CQ, also known as select-project-join query). As a first contribution we discuss how to modify the Yannakakis algorithm [39] to obtain a static query evaluation algorithm that satisfies properties ${P}_{1}$ and ${P}_{3}$ for the restricted class of acyclic join queries. We call this variant CDY, for Constant Delay Yannakakis. We then introduce Dyn, a dynamic version of CDY, and show that it yields DCLRs (properties ${P}_{1} - {P}_{4}$ ) for the acyclic join queries. DYN is a good algorithmic core to build practical dynamic algorithms on, for the following reasons.

我们的贡献如下。我们专注于在多重集语义下评估的合取聚合查询（Conjunctive Aggregate Queries，CAQ）类。合取聚合查询是对合取查询（Conjunctive Query，CQ，也称为选择 - 投影 - 连接查询）的结果计算聚合（例如，SUM、AVG 等）的查询。作为第一项贡献，我们讨论了如何修改 Yannakakis 算法[39]，以获得一个满足无环连接查询受限类的属性 ${P}_{1}$ 和 ${P}_{3}$ 的静态查询评估算法。我们将这个变体称为 CDY，即常量延迟 Yannakakis 算法。然后，我们引入了 Dyn，它是 CDY 的动态版本，并证明它能为无环连接查询产生动态常量延迟响应（Dynamic Constant - Delay Response，DCLR，属性 ${P}_{1} - {P}_{4}$ ）。出于以下原因，DYN 是构建实用动态算法的良好算法核心。

(1) Like standard Yannakakis, DYN is a conceptually simple algorithm, and therefore easy to implement.

(1) 与标准的 Yannakakis 算法一样，DYN 是一个概念上简单的算法，因此易于实现。

(2) We show that Dyn can support not only join queries (which do not allow projection), but also CQs (with projection) that belong to the class of free-connex acyclic CQs. This is optimal, in the sense that results by Bagan et al. [5] and Brault-Baron [10] for the static setting imply that, under certain complexity-theoretic assumptions, a DCLR can exist for a CQ $Q$ only if $Q$ is free-connex acyclic. In other words, DYN is able to evaluate the most general subclass of conjunctive queries satisfying ${P}_{1} - {P}_{4}$ .

(2) 我们证明了 Dyn 不仅可以支持连接查询（不允许投影），还可以支持属于自由连通无环 CQ 类的 CQ（带有投影）。这是最优的，因为 Bagan 等人[5]和 Brault - Baron[10]在静态设置下的结果表明，在某些复杂度理论假设下，只有当 CQ $Q$ 是自由连通无环时，才可能存在 DCLR。换句话说，DYN 能够评估满足 ${P}_{1} - {P}_{4}$ 的最通用的合取查询子类。

(3) Furthermore, in very recent work Berkholz et al [7] have characterized the class of self-join free CQs that feature CDE and that can be maintained in $O\left( 1\right)$ time under single-tuple updates. They show that this class corresponds to the class of so-called $q$ -hierarchical queries,a strict subclass of the free-connex acyclic queries. We match their lower bound: for (not necessarily self-join free) q-hierarchical CQs the DYN algorithm processes single-tuple updates in constant time. For non q-hierarchical queries, Berkholz et al.'s result yields it unlikely that single-tuple updates can be processsed in constant time. While for such queries, DYN hence naturally also requires more processing time, our experiments show that it remains highly effective.

(3) 此外，在最近的工作中，Berkholz 等人[7]刻画了具有常量延迟枚举（Constant - Delay Enumeration，CDE）特性且在单元组更新下可以在 $O\left( 1\right)$ 时间内维护的无自连接 CQ 类。他们表明这个类对应于所谓的 $q$ - 分层查询类，这是自由连通无环查询的一个严格子类。我们达到了他们的下界：对于（不一定无自连接的）q - 分层 CQ，DYN 算法可以在常量时间内处理单元组更新。对于非 q - 分层查询，Berkholz 等人的结果表明不太可能在常量时间内处理单元组更新。虽然对于这类查询，DYN 自然也需要更多的处理时间，但我们的实验表明它仍然非常有效。

(4) For single-tuple updates, DYN also allows us to enumerate the delta result $Q\left( {{db} + u}\right)  - Q\left( {db}\right)$ with constant delay. This result is relevant for push-based query processing systems, where users do not ping the system for the complete current query answer, but instead ask to be notified of the changes to the query results when the database changes.

(4) 对于单元组更新，DYN 还允许我们以恒定延迟枚举增量结果 $Q\left( {{db} + u}\right)  - Q\left( {db}\right)$。这一结果与基于推送的查询处理系统相关，在该系统中，用户不会主动向系统询问当前完整的查询答案，而是要求在数据库发生更改时收到查询结果变化的通知。

Building on DYN, we present an extended algorithm that allows for dynamic evaluation of acyclic CAQs. In particular,for an CAQ $Q$ whose join ${Q}^{\prime }$ is acyclic but not free-connex, we obtain a dynamic query processing method that is based on delta-enumeration of a free-connex projection of ${Q}^{\prime }$ to materialize the resulting aggregates $Q\left( {db}\right)$ . We hence require $O\left( {\parallel {db}\parallel  + \parallel Q\left( {db}\right) \parallel }\right)$ memory in this case,just like IVM methods.

在 DYN 的基础上，我们提出了一种扩展算法，用于对无环聚合连接查询（CAQ）进行动态评估。具体而言，对于一个连接 ${Q}^{\prime }$ 无环但非自由连通的 CAQ $Q$，我们得到了一种动态查询处理方法，该方法基于对 ${Q}^{\prime }$ 的自由连通投影进行增量枚举，以物化得到的聚合结果 $Q\left( {db}\right)$。因此，在这种情况下，我们需要 $O\left( {\parallel {db}\parallel  + \parallel Q\left( {db}\right) \parallel }\right)$ 的内存，这与增量视图维护（IVM）方法相同。

Finally, we experimentally compare our approach against HIVM on the industry-standard benchmarks TPC-H and TPC-DS. Our experiments show that, for the class of acyclic CAQs, our method is up to one order of magnitude more efficient than HIVM, both in terms of update time and memory consumption. At the same time, our experiments show that the enumeration of $Q\left( {db}\right)$ from ${\mathcal{D}}_{db}$ is as fast (and sometimes,even faster) as when $Q\left( {db}\right)$ was materialized as an array.

最后，我们在行业标准基准测试 TPC - H 和 TPC - DS 上对我们的方法与混合增量视图维护（HIVM）方法进行了实验比较。我们的实验表明，对于无环 CAQ 类查询，无论是在更新时间还是内存消耗方面，我们的方法比 HIVM 方法效率高出一个数量级。同时，我们的实验表明，从 ${\mathcal{D}}_{db}$ 枚举 $Q\left( {db}\right)$ 的速度与将 $Q\left( {db}\right)$ 物化为数组时一样快（有时甚至更快）。

Organization. This paper is further organized as follows. We discuss additional related work in Section 2 and introduce background concepts in Section 3. DYN is developed in Section 4 and experimentally evaluated in Section 5. Because of space constraints, full proofs are deferred to the full version of this paper, but cruxes of some key results may be found in the Appendix.

结构安排。本文的其余部分安排如下。我们在第 2 节讨论其他相关工作，并在第 3 节介绍背景概念。DYN 在第 4 节进行开发，并在第 5 节进行实验评估。由于篇幅限制，完整的证明将推迟到本文的完整版本中给出，但一些关键结果的要点可在附录中找到。

## 2. RELATED WORK

## 2. 相关工作

Incremental View Maintenance. The problem of incrementally maintaining materialized answers to conjunctive queries under updates has been extensively studied $\lbrack 3,8,9$ , ${11},{12},{20},{26},{30},{33}\rbrack$ ,and has been adapted to support different types of aggregates $\left\lbrack  {{28},{35}}\right\rbrack$ . A natural extension of IVM is the maintenance of auxiliary views $\left\lbrack  {{24},{34}}\right\rbrack$ ,which have been also adapted to allow for aggregate queries [21]. IVM has been influential to several other areas of databases (see [13] for a recent survey). Our work differs from IVM in that we maintain data structures that do not fully materialize query results.

增量视图维护。在更新操作下增量维护连接查询的物化答案这一问题已得到广泛研究 $\lbrack 3,8,9$、${11},{12},{20},{26},{30},{33}\rbrack$，并且已进行了扩展以支持不同类型的聚合操作 $\left\lbrack  {{28},{35}}\right\rbrack$。增量视图维护（IVM）的一个自然扩展是辅助视图 $\left\lbrack  {{24},{34}}\right\rbrack$ 的维护，这些辅助视图也已进行了调整以支持聚合查询 [21]。IVM 对数据库的其他几个领域产生了影响（有关最新综述，请参阅 [13]）。我们的工作与 IVM 的不同之处在于，我们维护的数据结构不会完全物化查询结果。

Factorized Databases. Factorized Databases are ingenious succinct representations of relational tables [31]. They allow for constant-delay enumeration and do not only reduce memory consumption, but can also avoid redundancy and speed up query processing $\left\lbrack  {6,{31},{36}}\right\rbrack$ . While factorized query evaluation is not limited to acyclic queries (as we are), it has only been studied in the static setting without updates.

因式分解数据库。因式分解数据库是关系表的一种巧妙的简洁表示形式 [31]。它们允许以恒定延迟进行枚举，不仅可以减少内存消耗，还可以避免冗余并加速查询处理 $\left\lbrack  {6,{31},{36}}\right\rbrack$。虽然因式分解查询评估并不局限于无环查询（与我们的工作不同），但它仅在无更新的静态环境中进行了研究。

Join algorithms. The well-known Yannakakis algorithm evaluates acyclic join queries in $O\left( {\parallel {db}\parallel  + \parallel Q\left( {db}\right) \parallel }\right)$ by using a join tree of such query [39]. Worst-case optimal algorithms have been developed for more general classes of queries by inspecting other forms of query decompositions (see [29] for a survey). Recently, join algorithms derived from query decompositions have been identified to use intermediate data structures similar to factorized databases [31]. These data structures, however, are designed for the static setting and not to react to updates. Recent work has also extended known join algorithms to allow for multiple aggregations on top of join queries $\left\lbrack  {2,{23}}\right\rbrack$ in the static setting.

连接算法。著名的亚纳卡基斯（Yannakakis）算法通过使用无环连接查询的连接树，在 $O\left( {\parallel {db}\parallel  + \parallel Q\left( {db}\right) \parallel }\right)$ 时间内评估此类查询 [39]。通过检查查询的其他分解形式，已经为更一般的查询类开发了最坏情况最优算法（有关综述，请参阅 [29]）。最近，已发现从查询分解派生的连接算法使用的中间数据结构与因式分解数据库 [31] 类似。然而，这些数据结构是为静态环境设计的，无法对更新做出反应。最近的工作还扩展了已知的连接算法，以允许在静态环境下对连接查询进行多次聚合操作 $\left\lbrack  {2,{23}}\right\rbrack$。

The Generalized Distributed Law. The Generalized Distributed Law (GDL) is an algorithm for solving the marginalize a product function (MPF) problem [4]. It has been recently shown to be equivalent to algorithms for computing aggregate-join queries with one aggregate [23]. The algorithm DYN developed in this paper can be seen as a strategy for solving the MPF problem under the dynamic setting.

广义分配律。广义分配律（GDL）是一种用于解决边缘化乘积函数（MPF）问题的算法 [4]。最近的研究表明，它等同于计算具有一个聚合操作的聚合连接查询的算法 [23]。本文开发的算法 DYN 可以看作是在动态环境下解决 MPF 问题的一种策略。

## 3. PRELIMINARIES

## 3. 预备知识

We adopt the data model of Generalized Multiset Relations (GMRs for short) [25,26]. A GMR is a relation in which each tuple is associated to an integer in $\mathbb{Z}$ . Figure 1 shows several examples. Note that in a GMR, in contrast to classical multisets, the multiplicity of a tuple can be negative. This allows to treat insertions and deletions symmetrically, as we will later see. To avoid ambiguity, we give a formal definition of GMRs that will be used throughout this paper.

我们采用广义多重集关系（Generalized Multiset Relations，简称GMRs）的数据模型 [25,26]。GMR是一种关系，其中每个元组都与一个属于$\mathbb{Z}$的整数相关联。图1展示了几个示例。请注意，与经典多重集不同，在GMR中，元组的重数可以为负。正如我们稍后将看到的，这使得我们能够对称地处理插入和删除操作。为避免歧义，我们给出GMR的正式定义，该定义将在本文中一直使用。

Tuples. We first introduce some notation for tuples. Let $\bar{x}$ be a set of variables (also commonly known as column names or attributes). We write $\mathbb{T}\left\lbrack  \bar{x}\right\rbrack$ for the universe of all possible tuples over $\bar{x}$ . If $\overrightarrow{t} \in  \mathbb{T}\left\lbrack  \bar{x}\right\rbrack$ and $y$ is a variable in $\bar{x}$ then we write $\overrightarrow{t}\left( y\right)$ for the value assigned to $y$ by $\overrightarrow{t}$ . If $\bar{y} \subseteq  \bar{x}$ then we write $\overrightarrow{t}\left\lbrack  \bar{y}\right\rbrack$ for the tuple over $\bar{y}$ obtained from $\overrightarrow{t}$ by removing all variables in $\bar{x} \smallsetminus  \bar{y}$ . For example,if $\overrightarrow{t} = \langle A : 5,B : 4,C : 3\rangle$ then $\overrightarrow{t}\left( A\right)  = 5$ and $\overrightarrow{t}\left\lbrack  {B,C}\right\rbrack   = \langle B : 4,C : 3\rangle$ .

元组。我们首先介绍一些关于元组的符号。设$\bar{x}$为一个变量集（通常也称为列名或属性）。我们用$\mathbb{T}\left\lbrack  \bar{x}\right\rbrack$表示所有可能的基于$\bar{x}$的元组的全集。如果$\overrightarrow{t} \in  \mathbb{T}\left\lbrack  \bar{x}\right\rbrack$且$y$是$\bar{x}$中的一个变量，那么我们用$\overrightarrow{t}\left( y\right)$表示$\overrightarrow{t}$赋给$y$的值。如果$\bar{y} \subseteq  \bar{x}$，那么我们用$\overrightarrow{t}\left\lbrack  \bar{y}\right\rbrack$表示通过从$\overrightarrow{t}$中移除$\bar{x} \smallsetminus  \bar{y}$中的所有变量而得到的基于$\bar{y}$的元组。例如，如果$\overrightarrow{t} = \langle A : 5,B : 4,C : 3\rangle$，那么$\overrightarrow{t}\left( A\right)  = 5$且$\overrightarrow{t}\left\lbrack  {B,C}\right\rbrack   = \langle B : 4,C : 3\rangle$。

GMRs. A generalized multiset relation (GMR) over $\bar{x}$ is a function $R : \mathbb{T}\left\lbrack  \bar{x}\right\rbrack   \rightarrow  \mathbb{Z}$ from relation tuples over $\bar{x}$ to integers. Every GMR $R$ is a total function from the (possibly infinite) set $\mathbb{T}\left\lbrack  \bar{x}\right\rbrack$ to $\mathbb{Z}$ and hence,conceptually,is an infinite object. However, every GMR is required to have finite support $\operatorname{supp}\left( R\right)  \mathrel{\text{:=}} \{ \overrightarrow{t} \in  \mathbb{T}\left\lbrack  \bar{x}\right\rbrack   \mid  R\left( \overrightarrow{t}\right)  \neq  0\}$ . Intuitively, $R\left( \overrightarrow{t}\right)  = 0$ indicates that $\overrightarrow{t}$ is absent from $R$ . The fact that $R$ must have finite support indicates that $R$ is a finite relation. To illustrate,in Figure $1,R\left( {\langle a,b\rangle }\right)  = 2$ ,hence present,while $R\left( \left\langle  {a,{b}^{\prime }}\right\rangle  \right)  = 0$ ,hence absent (and not shown). In what follows,we abuse notation and write $\left( {\overrightarrow{t},\mu }\right)  \in  R$ to indicate that $\overrightarrow{t} \in  \operatorname{supp}\left( R\right)$ and $R\left( \overrightarrow{t}\right)  = \mu ;\overrightarrow{t} \in  R$ to indicate $\overrightarrow{T} \in  \operatorname{supp}\left( R\right)$ ; and $\left| R\right|$ for $\left| {\operatorname{supp}\left( R\right) }\right|$ . We say that $R$ is empty if $\operatorname{supp}\left( R\right)  = \varnothing$ . The set of all GMRs over $\bar{x}$ is denoted by $\mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack$ . A GMR is positive if $R\left( \overrightarrow{t}\right)  > 0$ for all $\overrightarrow{t} \in  \operatorname{supp}\left( R\right)$ .

广义多重集关系（GMRs）。在$\bar{x}$上的广义多重集关系（Generalized Multiset Relation，GMR）是一个从$\bar{x}$上的关系元组到整数的函数$R : \mathbb{T}\left\lbrack  \bar{x}\right\rbrack   \rightarrow  \mathbb{Z}$。每个GMR $R$是一个从（可能无限的）集合$\mathbb{T}\left\lbrack  \bar{x}\right\rbrack$到$\mathbb{Z}$的全函数，因此，从概念上讲，它是一个无限对象。然而，每个GMR都要求具有有限支撑$\operatorname{supp}\left( R\right)  \mathrel{\text{:=}} \{ \overrightarrow{t} \in  \mathbb{T}\left\lbrack  \bar{x}\right\rbrack   \mid  R\left( \overrightarrow{t}\right)  \neq  0\}$。直观地说，$R\left( \overrightarrow{t}\right)  = 0$表示$\overrightarrow{t}$不在$R$中。$R$必须具有有限支撑这一事实表明$R$是一个有限关系。为了说明这一点，在图$1,R\left( {\langle a,b\rangle }\right)  = 2$中，因此存在，而$R\left( \left\langle  {a,{b}^{\prime }}\right\rangle  \right)  = 0$，因此不存在（且未显示）。接下来，我们滥用符号，用$\left( {\overrightarrow{t},\mu }\right)  \in  R$表示$\overrightarrow{t} \in  \operatorname{supp}\left( R\right)$，用$R\left( \overrightarrow{t}\right)  = \mu ;\overrightarrow{t} \in  R$表示$\overrightarrow{T} \in  \operatorname{supp}\left( R\right)$；用$\left| R\right|$表示$\left| {\operatorname{supp}\left( R\right) }\right|$。我们说如果$\operatorname{supp}\left( R\right)  = \varnothing$，则$R$为空。在$\bar{x}$上的所有GMR的集合用$\mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack$表示。如果对于所有的$\overrightarrow{t} \in  \operatorname{supp}\left( R\right)$都有$R\left( \overrightarrow{t}\right)  > 0$，则称一个GMR为正的。

Operations on GMRs. Let $R$ and $S$ be GMRs over $\bar{x},T$ a GMR over $\bar{y}$ ,and $\bar{z} \subseteq  \bar{x}$ . The operations union $\left( {R + S}\right)$ , minus(R - S),join $\left( {R \boxtimes  T}\right)$ and projection $\left( {\pi \bar{z}R}\right)$ over GMRs are defined as follows.

对GMR的操作。设$R$和$S$是在$\bar{x},T$上的GMR，$\bar{y}$上的一个GMR，以及$\bar{z} \subseteq  \bar{x}$。GMR上的并运算$\left( {R + S}\right)$、差运算（R - S）、连接运算$\left( {R \boxtimes  T}\right)$和投影运算$\left( {\pi \bar{z}R}\right)$定义如下。

$$
R + S \in  \mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack   : \overrightarrow{t} \mapsto  R\left( \overrightarrow{t}\right)  + S\left( \overrightarrow{t}\right) 
$$

$$
R - S \in  \mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack   : \overrightarrow{t} \mapsto  R\left( \overrightarrow{t}\right)  - S\left( \overrightarrow{t}\right) 
$$

$$
R \boxtimes  T \in  \mathbb{{GMR}}\left\lbrack  {\bar{x} \cup  \bar{y}}\right\rbrack   : \overrightarrow{t} \mapsto  R\left( {\overrightarrow{t}\left\lbrack  \bar{x}\right\rbrack  }\right)  \times  S\left( {\overrightarrow{t}\left\lbrack  \bar{y}\right\rbrack  }\right) 
$$

$$
{\pi }_{\bar{z}}R \in  \mathbb{{GMR}}\left\lbrack  \bar{z}\right\rbrack  \; : \overrightarrow{t} \mapsto  \mathop{\sum }\limits_{{\overrightarrow{s} \in  \mathbb{T}\left\lbrack  \bar{x}\right\rbrack  ,\overrightarrow{s}\left\lbrack  \bar{z}\right\rbrack   = \overrightarrow{t}}}R\left( \overrightarrow{s}\right) 
$$

<!-- Media -->

<!-- figureText: $R$ $S$ $T$ ${\pi }_{A}\left( S\right)$ $A$ $R \boxtimes  T$ $A$ -3 3 15 -4 $B$ $R + S$ $R - S$ $A$ $A$ $B$ 3 -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_2.jpg?x=988&y=154&w=588&h=260&r=0"/>

Figure 1: Operations on GMRs

图1：对GMR的操作

<!-- Media -->

Figure 1 illustrates these operations. Note that GMRs there are positive, modeling standard multisets. Hence union, join, and projection correspond to the classical operations from relational algebra under multiset (i.e., bag) semantics. Minus is not relational difference, since it simply subtracts multiplicities (notice this could yield negative multiplicities).

图1展示了这些操作。注意，那里的GMR是正的，用于建模标准多重集。因此，并、连接和投影对应于多重集（即包）语义下关系代数中的经典操作。差运算不是关系差，因为它只是简单地减去重数（注意这可能会产生负重数）。

Query Language. Conjunctive Queries (CQs) are expressions of the form

查询语言。合取查询（Conjunctive Queries，CQs）是如下形式的表达式

$$
Q = {\pi }_{\bar{y}}\left( {{r}_{1}\left( \overline{{x}_{1}}\right)  \boxtimes  \cdots  \boxtimes  {r}_{n}\left( \overline{{x}_{n}}\right) }\right) .
$$

Here, ${r}_{1},\ldots ,{r}_{n}$ are relation symbols; $\overline{{x}_{1}},\ldots ,\overline{{x}_{n}}$ are sets of variables,and $\bar{y} \subseteq  \overline{{x}_{1}} \cup  \cdots  \cup  \overline{{x}_{n}}$ is the set of output variables, also denoted by $\operatorname{out}\left( Q\right)$ . If $\bar{y} = \overline{{x}_{1}} \cup  \cdots  \cup  \overline{{x}_{n}}$ then $Q$ is a join query and simply denoted as ${r}_{1}\left( \overline{{x}_{1}}\right)  \boxtimes  \cdots  \boxtimes  {r}_{n}\left( \overline{{x}_{n}}\right)$ . The pairs ${r}_{i}\left( \overline{{x}_{i}}\right)$ are called atomic queries (or simply atoms).

这里，${r}_{1},\ldots ,{r}_{n}$ 是关系符号；$\overline{{x}_{1}},\ldots ,\overline{{x}_{n}}$ 是变量集合，并且 $\bar{y} \subseteq  \overline{{x}_{1}} \cup  \cdots  \cup  \overline{{x}_{n}}$ 是输出变量集合，也用 $\operatorname{out}\left( Q\right)$ 表示。如果 $\bar{y} = \overline{{x}_{1}} \cup  \cdots  \cup  \overline{{x}_{n}}$，那么 $Q$ 是一个连接查询（join query），并简单地表示为 ${r}_{1}\left( \overline{{x}_{1}}\right)  \boxtimes  \cdots  \boxtimes  {r}_{n}\left( \overline{{x}_{n}}\right)$。对 ${r}_{i}\left( \overline{{x}_{i}}\right)$ 被称为原子查询（atomic queries，或简称为原子）。

A database over a set $A$ of atoms is a function ${db}$ that maps every atom $r\left( \bar{x}\right)  \in  A$ to a positive GMR $d{b}_{r\left( \bar{x}\right) }$ over $\bar{x}$ . Given a database ${db}$ over the atoms occurring in query $Q$ , the evaluation of $Q$ over ${db}$ ,denoted $Q\left( {db}\right)$ ,is the GMR over $\bar{y}$ constructed in the expected way: substitute each atom $r\left( \bar{x}\right)$ in $Q$ by $d{b}_{r\left( \bar{x}\right) }$ ,and subsequently apply the operations according to the structure of $Q$ .

基于原子集合 $A$ 的数据库是一个函数 ${db}$，它将每个原子 $r\left( \bar{x}\right)  \in  A$ 映射到基于 $\bar{x}$ 的正广义多重关系（positive GMR）$d{b}_{r\left( \bar{x}\right) }$。给定一个基于查询 $Q$ 中出现的原子的数据库 ${db}$，在 ${db}$ 上对 $Q$ 进行评估，记为 $Q\left( {db}\right)$，是按照预期方式构造的基于 $\bar{y}$ 的广义多重关系（GMR）：用 $d{b}_{r\left( \bar{x}\right) }$ 替换 $Q$ 中的每个原子 $r\left( \bar{x}\right)$，然后根据 $Q$ 的结构应用相应的操作。

Discussion. For ease of notation in the rest of the paper, we have not included relational selection ${\sigma }_{\theta }\left( {r\left( \bar{x}\right) }\right)$ in queries. This is without loss of generality, as to dynamically process a Select-Project-Join query we can always filter out irrelevant tuples. For example,for $Q = {\pi }_{\bar{z}}\left( {{\sigma }_{{\theta }_{1}}\left( {r\left( \bar{x}\right) }\right)  \boxtimes  {\sigma }_{{\theta }_{2}}\left( {s\left( \bar{y}\right) }\right) }\right)$ we can consider new relation symbols ${r}^{\prime }$ and ${s}^{\prime }$ and dynamically process ${Q}^{\prime } = {\pi }_{\bar{z}}\left( {{r}^{\prime }\left( \bar{x}\right)  \boxtimes  {s}^{\prime }\left( \bar{y}\right) }\right)$ instead. Then,whenever $r$ and/or $s$ are updated,it suffices to discard the tuples that do not satisfy the corresponding filter, and propagate the rest of the updates to relations ${r}^{\prime }$ and ${s}^{\prime }$ to update ${Q}^{\prime }$ .

讨论。为了便于本文其余部分的符号表示，我们在查询中未包含关系选择 ${\sigma }_{\theta }\left( {r\left( \bar{x}\right) }\right)$。这并不失一般性，因为要动态处理选择 - 投影 - 连接查询（Select - Project - Join query），我们总是可以过滤掉不相关的元组。例如，对于 $Q = {\pi }_{\bar{z}}\left( {{\sigma }_{{\theta }_{1}}\left( {r\left( \bar{x}\right) }\right)  \boxtimes  {\sigma }_{{\theta }_{2}}\left( {s\left( \bar{y}\right) }\right) }\right)$，我们可以考虑新的关系符号 ${r}^{\prime }$ 和 ${s}^{\prime }$，并动态处理 ${Q}^{\prime } = {\pi }_{\bar{z}}\left( {{r}^{\prime }\left( \bar{x}\right)  \boxtimes  {s}^{\prime }\left( \bar{y}\right) }\right)$ 来代替。然后，每当 $r$ 和/或 $s$ 被更新时，只需丢弃不满足相应过滤器的元组，并将其余更新传播到关系 ${r}^{\prime }$ 和 ${s}^{\prime }$ 以更新 ${Q}^{\prime }$。

Updates and deltas. An update to a GMR $R$ is simply a GMR ${\Delta R}$ over the same variables as $R$ . Applying update ${\Delta R}$ to $R$ yields the GMR $R + {\Delta R}$ . An update to a database ${db}$ is a collection $u$ of (not necessarily positive) GMRs,one GMR ${u}_{r\left( \bar{x}\right) }$ for every atom $r\left( \bar{x}\right)$ of ${db}$ ,such that $d{b}_{r\left( \bar{x}\right) } + {u}_{r\left( \bar{x}\right) }$ is positive. We write ${db} + u$ for the database obtained by applying $u$ to each atom of ${db}$ ,i.e., ${\left( db + u\right) }_{r\left( \bar{x}\right) } = d{b}_{r\left( \bar{x}\right) } + {u}_{r\left( \bar{x}\right) }$ , for every atom $r\left( \bar{x}\right)$ of ${db}$ . For every query $Q$ ,every database ${db}$ and every update $u$ to ${db}$ ,we define the delta query ${\Delta Q}\left( {{db},u}\right)$ of $Q$ w.r.t. ${db}$ and $u$ by

更新与增量。对广义匹配规则（GMR） $R$ 的一次更新，简单来说就是一个与 $R$ 作用于相同变量的广义匹配规则 ${\Delta R}$。将更新 ${\Delta R}$ 应用于 $R$ 会得到广义匹配规则 $R + {\Delta R}$。对数据库 ${db}$ 的一次更新是一组（不一定为正的）广义匹配规则 $u$，对于 ${db}$ 中的每个原子 $r\left( \bar{x}\right)$ 都有一个广义匹配规则 ${u}_{r\left( \bar{x}\right) }$，使得 $d{b}_{r\left( \bar{x}\right) } + {u}_{r\left( \bar{x}\right) }$ 为正。我们用 ${db} + u$ 表示将 $u$ 应用于 ${db}$ 中的每个原子后得到的数据库，即对于 ${db}$ 中的每个原子 $r\left( \bar{x}\right)$，有 ${\left( db + u\right) }_{r\left( \bar{x}\right) } = d{b}_{r\left( \bar{x}\right) } + {u}_{r\left( \bar{x}\right) }$。对于每个查询 $Q$、每个数据库 ${db}$ 以及对 ${db}$ 的每次更新 $u$，我们定义 $Q$ 相对于 ${db}$ 和 $u$ 的增量查询 ${\Delta Q}\left( {{db},u}\right)$ 为

$$
{\Delta Q}\left( {{db},u}\right)  \mathrel{\text{:=}} Q\left( {{db} + u}\right)  - Q\left( {db}\right) .
$$

As such, ${\Delta Q}\left( {{db},u}\right)$ is the update that we need to apply to $Q\left( {db}\right)$ in order to obtain $Q\left( {{db} + u}\right)$ .

因此，为了从 $Q\left( {db}\right)$ 得到 $Q\left( {{db} + u}\right)$，我们需要对 $Q\left( {db}\right)$ 应用更新 ${\Delta Q}\left( {{db},u}\right)$。

<!-- Media -->

<!-- figureText: $\left( {T}_{1}\right)$ $\left( {T}_{2}\right)$ $\left( {T}_{4}\right)$ $\left\lbrack  y\right\rbrack$ $\left\lbrack  {y,v}\right\rbrack$ $\left\lbrack  {x,y}\right\rbrack$ $\left\lbrack  {y,v}\right\rbrack$ $\left( {T}_{3}\right)$ $R\left( {x,y,z}\right)$ $R\left( {x,y,z}\right)$ $\left\lbrack  y\right\rbrack$ $S\left( {x,y,u}\right) T\left( {y,v,w}\right)$ $S\left( {x,y,u}\right)$ $\left\lbrack  {y,v}\right\rbrack$ $\left\lbrack  {x,y,z}\right\rbrack$ $U\left( {y,v,p}\right)$ -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_3.jpg?x=152&y=148&w=1487&h=240&r=0"/>

Figure 2: Width-one GHDs for $\{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) \} .{T}_{1}$ is a traditional join tree, ${T}_{3}$ and ${T}_{4}$ are generalized join trees. In addition, ${T}_{4}$ is simple.

图 2：$\{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) \} .{T}_{1}$ 的宽度为 1 的广义超树分解（GHD）是传统的连接树，${T}_{3}$ 和 ${T}_{4}$ 是广义连接树。此外，${T}_{4}$ 是简单的。

<!-- Media -->

### 3.1 Acyclicity

### 3.1 无环性

Throughout the paper we focus on the class of acyclic queries. While there are many equivalent ways of defining acyclic queries [1] we will use here a characterization of the acyclic queries in terms of those queries that have a Generalized Hy-pertree Decomposition (GHD for short) of width one [19]. Width-one GHDs generalize traditional join trees [1] by also allowing partial hyperedges to occur as nodes in the tree. Intuitively, these partial hyperedges represent projections of single atoms. The importance of this feature will become clear at the end of Section 4, where we show the existence of acyclic queries for which traditional join trees (where only full hyperedges can occur) do not induce optimal complexity algorithms under the setting of dynamic query evaluation.

在本文中，我们主要关注无环查询这一类别。虽然有许多等价的方法来定义无环查询 [1]，但在这里我们将使用一种基于宽度为 1 的广义超树分解（GHD，简称）的无环查询特征描述 [19]。宽度为 1 的广义超树分解通过允许部分超边作为树中的节点，对传统的连接树 [1] 进行了推广。直观地说，这些部分超边表示单个原子的投影。这一特性的重要性将在第 4 节末尾变得清晰，在那里我们将展示存在一些无环查询，对于这些查询，传统的连接树（其中只允许出现完整的超边）在动态查询评估的设置下无法导出最优复杂度的算法。

To simplify notation, we denote the set of all variables (resp. the set of all atoms) that occur in a mathematical object $X$ (such as a query) by $\operatorname{var}\left( X\right)$ (resp. $\operatorname{at}\left( X\right)$ ). In particular,if $X$ is itself a set of variables,then $\operatorname{var}\left( X\right)  = X$ .

为了简化符号，我们用 $\operatorname{var}\left( X\right)$（分别地，$\operatorname{at}\left( X\right)$）表示出现在数学对象 $X$（如查询）中的所有变量的集合（分别地，所有原子的集合）。特别地，如果 $X$ 本身就是一个变量集合，那么 $\operatorname{var}\left( X\right)  = X$。

Definition 3.1 (Width-1 GHD). Let $A$ be a finite set of atoms. A hyperedge in $A$ is a set $\bar{x}$ of variables such that $\bar{x} \subseteq  \operatorname{var}\left( \mathbf{a}\right)$ for some atom $\mathbf{a} \in  A$ . We call $\bar{x}$ full in $A$ if $\bar{x} = \operatorname{var}\left( \mathbf{a}\right)$ for some $\mathbf{a} \in  A$ ,and partial otherwise. A Generalized Hypertree Decomposition (GHD) of width one for $A$ is a directed tree $T = \left( {V,E}\right)$ such that: ${}^{1}$

定义 3.1（宽度为 1 的广义超树分解）。设 $A$ 是一个有限的原子集合。$A$ 中的一个超边是一个变量集合 $\bar{x}$，使得对于某个原子 $\mathbf{a} \in  A$ 有 $\bar{x} \subseteq  \operatorname{var}\left( \mathbf{a}\right)$。如果对于某个 $\mathbf{a} \in  A$ 有 $\bar{x} = \operatorname{var}\left( \mathbf{a}\right)$，我们称 $\bar{x}$ 在 $A$ 中是完整的，否则称其为部分的。$A$ 的宽度为 1 的广义超树分解（GHD）是一棵有向树 $T = \left( {V,E}\right)$，使得：${}^{1}$

- All nodes of $T$ are either atoms or hyperedges in $A$ . Moreover,every atom in $A$ occurs in $T$ .

- $T$的所有节点要么是$A$中的原子，要么是$A$中的超边。此外，$A$中的每个原子都出现在$T$中。

- Whenever the same variable $x$ occurs in two nodes $m$ and $n$ of $T$ ,then $x$ occurs in each node on the unique undirected path linking $m$ and $n$ .

- 只要同一个变量$x$出现在$T$的两个节点$m$和$n$中，那么$x$就会出现在连接$m$和$n$的唯一无向路径上的每个节点中。

If all nodes in $T$ are atoms,then $T$ is a traditional join tree.

如果$T$中的所有节点都是原子，那么$T$就是一棵传统的连接树。

To illustrate, Figure 2 shows four width-one GHDs for $\{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) \} .{T}_{1}$ is traditional while the others are not.

为了说明这一点，图2展示了$\{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) \} .{T}_{1}$的四个宽度为1的广义超树分解（GHD），其中一个是传统的，而其他的不是。

Definition 3.2 (Acyclicity). A CQ $Q$ is acyclic if there exists a width-one GHD $T$ for $\operatorname{at}\left( Q\right)$ ,and is cyclic otherwise.

定义3.2（无环性）。如果存在一个宽度为1的广义超树分解$T$用于$\operatorname{at}\left( Q\right)$，则合取查询（CQ）$Q$是无环的，否则是有环的。

For example the width-one GHDs of Figure 2 show that $R\left( {x,y,z}\right)  \boxtimes  S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right)$ is acyclic. In contrast, $R\left( {x,y}\right)  \boxtimes  S\left( {y,z}\right)  \boxtimes  T\left( {x,z}\right)$ ,the triangle query,is the prototypical cyclic join query.

例如，图2中的宽度为1的广义超树分解表明$R\left( {x,y,z}\right)  \boxtimes  S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right)$是无环的。相比之下，$R\left( {x,y}\right)  \boxtimes  S\left( {y,z}\right)  \boxtimes  T\left( {x,z}\right)$（三角形查询）是典型的有环连接查询。

For the rest of the paper, it will be convenient to focus on width-one GHDs of a particular form. We call such restricted GHDs generalized join trees.

在本文的其余部分，关注特定形式的宽度为1的广义超树分解会很方便。我们将这种受限的广义超树分解称为广义连接树。

Definition 3.3 (Generalized Join Tree). A generalized join tree for set of atoms $A$ is a width-one GHD $T$ for $A$ in which all atoms occur as leafs. Moreover,every interior node $n$ must have at least one child $c$ such that $\operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( c\right)$ .

定义3.3（广义连接树）。原子集合$A$的广义连接树是$A$的一个宽度为1的广义超树分解$T$，其中所有原子都作为叶子节点出现。此外，每个内部节点$n$必须至少有一个子节点$c$，使得$\operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( c\right)$。

In Figure 2,trees ${T}_{3}$ and ${T}_{4}$ are generalized join trees; trees ${T}_{1}$ and ${T}_{2}$ are not. The following proposition (proof in Appendix A) shows that we may restrict our attention to generalized join trees without loss of generality.

在图2中，树${T}_{3}$和${T}_{4}$是广义连接树；树${T}_{1}$和${T}_{2}$不是。以下命题（证明见附录A）表明，我们可以不失一般性地将注意力限制在广义连接树上。

Proposition 3.4. If there exists a width-one GHD for set of atoms $A$ ,then there also exists a generalized join tree for $A$ . Consequently,a ${CQQ}$ is acyclic iff at(Q)has a generalized join tree.

命题3.4。如果存在原子集合$A$的宽度为1的广义超树分解，那么也存在$A$的广义连接树。因此，一个${CQQ}$是无环的，当且仅当at(Q)有一棵广义连接树。

In what follows, we will refer to generalized join trees simply as join trees. When $n$ is a node of a join tree $T,c$ is a child of $n$ ,and $\operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( c\right)$ ,we call $c$ a guard of $n$ . By definition, there is a guard for every hyperedge. We denote by $\operatorname{grd}\left( n\right)$ the set of guards of $n$ ,by $\operatorname{ch}\left( n\right)$ the set of children of $n$ ,and by $\operatorname{ng}\left( n\right)$ the set $\operatorname{ch}\left( n\right)  \smallsetminus  \operatorname{grd}\left( n\right)$ of non-guards of $n$ . Finally,we define $\operatorname{pvar}\left( c\right)$ to be the set of variables that $c$ has in common with its parent (pvar $\left( c\right)  = \varnothing$ for the root). For example,in ${T}_{3}$ of Figure 2,pvar $\left( {S\left( {x,y,u}\right) }\right)  = \{ x,y\}$ .

在接下来的内容中，我们将广义连接树简称为连接树。当$n$是连接树的一个节点，$T,c$是$n$的一个子节点，并且$\operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( c\right)$时，我们称$c$为$n$的一个守卫。根据定义，每个超边都有一个守卫。我们用$\operatorname{grd}\left( n\right)$表示$n$的守卫集合，用$\operatorname{ch}\left( n\right)$表示$n$的子节点集合，用$\operatorname{ng}\left( n\right)$表示$n$的非守卫集合$\operatorname{ch}\left( n\right)  \smallsetminus  \operatorname{grd}\left( n\right)$。最后，我们将$\operatorname{pvar}\left( c\right)$定义为$c$与其父节点共有的变量集合（对于根节点为pvar $\left( c\right)  = \varnothing$）。例如，在图2的${T}_{3}$中，pvar $\left( {S\left( {x,y,u}\right) }\right)  = \{ x,y\}$。

### 3.2 Computational Model

### 3.2 计算模型

We focus on dynamic query evaluation in main-memory and analyze performance under data complexity [38]. We assume a model of computation where tuple values and integers take $O\left( 1\right)$ space and arithmetic operations on integers as well as memory lookups are $O\left( 1\right)$ operations. We further assume that every GMR $R$ can be represented by a data structure that allows (1) enumeration of $R$ with constant delay (as defined in Section 4.1); (2) multiplicity lookups $R\left( \overrightarrow{t}\right)$ in $O\left( 1\right)$ time given $\overrightarrow{t}$ ; (3) single-tuple insertions and deletions in $O\left( 1\right)$ time; while (4) having size that is proportional to the number of tuples in the support of $R$ . We further assume the existence of dynamic data structures that can be used to index GMRs on a subset of their variables. Concretely if $R$ is a GMR over $\bar{x}$ and $I$ is an index of $R$ on $\bar{y} \subseteq  \bar{x}$ then we assume that for every $\bar{y}$ -tuple $\overrightarrow{s}$ we can retrieve in $O\left( 1\right)$ time a pointer to the GMR $I\left( \overrightarrow{s}\right)  \in  \mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack$ consisting of all tuples that project to $\overrightarrow{s}$ ,as formally defined by

我们专注于主内存中的动态查询评估，并分析数据复杂度下的性能 [38]。我们假设一种计算模型，其中元组值和整数占用 $O\left( 1\right)$ 空间，对整数的算术运算以及内存查找都是 $O\left( 1\right)$ 操作。我们进一步假设每个广义多关系（Generalized Multi-Relation，GMR） $R$ 都可以由一个数据结构表示，该数据结构允许：（1）以恒定延迟枚举 $R$（如第 4.1 节所定义）；（2）在给定 $\overrightarrow{t}$ 的情况下，在 $O\left( 1\right)$ 时间内进行多重性查找 $R\left( \overrightarrow{t}\right)$；（3）在 $O\left( 1\right)$ 时间内进行单元素元组的插入和删除；（4）其大小与 $R$ 支持集中的元组数量成正比。我们进一步假设存在可用于对 GMR 在其变量子集上进行索引的动态数据结构。具体而言，如果 $R$ 是关于 $\bar{x}$ 的 GMR，并且 $I$ 是 $R$ 在 $\bar{y} \subseteq  \bar{x}$ 上的索引，那么我们假设对于每个 $\bar{y}$ - 元组 $\overrightarrow{s}$，我们可以在 $O\left( 1\right)$ 时间内检索到指向由所有投影到 $\overrightarrow{s}$ 的元组组成的 GMR $I\left( \overrightarrow{s}\right)  \in  \mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack$ 的指针，如正式定义的那样

$$
I\left( \overrightarrow{s}\right)  \in  \mathbb{{GMR}}\left\lbrack  \overrightarrow{x}\right\rbrack   : \overrightarrow{t} \mapsto  \left\{  \begin{array}{ll} R\left( \overrightarrow{t}\right) & \text{ if }\overrightarrow{t}\left\lbrack  \overrightarrow{y}\right\rbrack   = \overrightarrow{s} \\  0 & \text{ otherwise } \end{array}\right. 
$$

Moreover, we assume that single-tuple insertions and deletions to $R$ can be reflected in the index in $O\left( 1\right)$ time and that an index takes space linear in the support of $R$ . Essentially, our assumptions amount to perfect hashing of linear size [14]. Although this is not realistic for practical computers [32], it is well known that complexity results for this model can be translated, through amortized analysis, to average complexity in real-life implementations [14].

此外，我们假设对 $R$ 进行单元素元组的插入和删除操作可以在 $O\left( 1\right)$ 时间内反映到索引中，并且索引占用的空间与 $R$ 的支持集呈线性关系。本质上，我们的假设相当于线性大小的完美哈希 [14]。尽管这对于实际计算机来说不现实 [32]，但众所周知，通过摊还分析，该模型的复杂度结果可以转化为实际实现中的平均复杂度 [14]。

---

<!-- Footnote -->

${}^{1}$ Readers familiar with the usual definition of GHDs of arbitrary width may find a discussion of the correspondence between our definition and the usual one in Appendix A.

${}^{1}$ 熟悉任意宽度广义超树分解（Generalized Hypertree Decomposition，GHD）通常定义的读者可以在附录 A 中找到关于我们的定义与通常定义之间对应关系的讨论。

<!-- Footnote -->

---

## 4. DYNAMIC YANNAKAKIS

## 4. 动态扬纳卡基斯算法

In this section we develop DYN, a dynamic version of the Yannakakis algorithm. In Section 4.1, we introduce the notion of constant-delay enumeration. Then, in Section 4.2 we show that, for acyclic join queries, a representation satisfying properties ${P}_{1}$ and ${P}_{3}$ of the Introduction can be obtained by slightly modifying the Yannakakis algorithm. We introduce DYN in Section 4.3, and show in Sections 4.4- 4.5 that this also gives DCLRs for CQs that are free-connex acyclic. We show that this is optimal in two distinct ways in Section 4.6.

在本节中，我们开发了 DYN，即扬纳卡基斯（Yannakakis）算法的动态版本。在第 4.1 节中，我们介绍了恒定延迟枚举的概念。然后，在第 4.2 节中，我们表明，对于无环连接查询，通过对扬纳卡基斯算法进行轻微修改，可以得到满足引言中属性 ${P}_{1}$ 和 ${P}_{3}$ 的表示。我们在第 4.3 节中介绍 DYN，并在第 4.4 - 4.5 节中表明，这也为无连接环的合取查询（Conjunctive Query，CQ）提供了动态约束逻辑表示（Dynamic Constraint Logic Representation，DCLR）。我们在第 4.6 节中以两种不同的方式表明这是最优的。

### 4.1 Constant delay enumeration

### 4.1 恒定延迟枚举

Definition 4.1. A data structure $D$ supports enumeration of a set $E$ if there is a routine ENUM such that $\operatorname{ENUM}\left( D\right)$ outputs each element of $E$ exactly once. Moreover, $D$ supports constant-delay enumeration (CDE) of $E$ if,when $\operatorname{ENUM}\left( D\right)$ is invoked, the time until the output of the first element; the time between any two consecutive elements; and the time between the output of the last element and the termination of $\operatorname{ENUM}\left( D\right)$ ,are all constant. In particular,these times cannot depend on the size of $D$ nor on the size of $E$ . We say that $D$ supports constant-delay enumeration of a GMR $R$ if $D$ supports constant-delay enumeration of the set ${E}_{R} = \{ \left( {\overrightarrow{t},R\left( \overrightarrow{t}\right) }\right)  \mid  \overrightarrow{t} \in  \operatorname{supp}\left( R\right) \}$ .

定义 4.1。如果存在一个例程 ENUM 使得 $\operatorname{ENUM}\left( D\right)$ 恰好输出集合 $E$ 中的每个元素一次，则数据结构 $D$ 支持对集合 $E$ 的枚举。此外，如果在调用 $\operatorname{ENUM}\left( D\right)$ 时，直到输出第一个元素的时间、任意两个连续元素之间的时间以及输出最后一个元素到 $\operatorname{ENUM}\left( D\right)$ 终止之间的时间都是恒定的，则 $D$ 支持对 $E$ 的恒定延迟枚举（Constant - Delay Enumeration，CDE）。特别地，这些时间不能依赖于 $D$ 的大小或 $E$ 的大小。如果 $D$ 支持对集合 ${E}_{R} = \{ \left( {\overrightarrow{t},R\left( \overrightarrow{t}\right) }\right)  \mid  \overrightarrow{t} \in  \operatorname{supp}\left( R\right) \}$ 的恒定延迟枚举，我们就说 $D$ 支持对广义多关系（Generalized Multi - Relation，GMR） $R$ 的恒定延迟枚举。

As a trivial example of CDE of a GMR $R$ ,assume that the pairs $\left( {\overrightarrow{t},R\left( \overrightarrow{t}\right) }\right)$ of ${E}_{R}$ are stored in an array $A$ (without duplicates). Then $A$ supports CDE of $R$ : $\operatorname{ENUM}\left( A\right)$ simply iterates over each element in $A$ ,one by one,always outputting the current element. To see that this is correct, first observe that all pairs of ${E}_{R}$ will be output exactly once. Moreover, the time required to output the first pair is the time required to fetch the first array element, hence constant. Similarly, the time required to produce each subsequent output tuple is the time required to fetch the next array element, again constant. Finally, checking whether we have reached the end of ${E}_{R}$ amounts to checking whether we have reached the end of the array, again taking constant time.

作为广义匹配关系（GMR）$R$的连续延迟枚举（CDE）的一个简单示例，假设${E}_{R}$的对$\left( {\overrightarrow{t},R\left( \overrightarrow{t}\right) }\right)$存储在一个数组$A$中（无重复）。那么$A$支持$R$的连续延迟枚举：$\operatorname{ENUM}\left( A\right)$只需逐个遍历$A$中的每个元素，并始终输出当前元素。为了验证这是正确的，首先观察到${E}_{R}$的所有对都将被精确输出一次。此外，输出第一对所需的时间就是获取第一个数组元素所需的时间，因此是常数时间。同样，生成每个后续输出元组所需的时间就是获取下一个数组元素所需的时间，同样是常数时间。最后，检查是否已到达${E}_{R}$的末尾等同于检查是否已到达数组的末尾，同样只需常数时间。

This example shows that in order to do CDE of the result $Q\left( {db}\right)$ of a query $Q$ on input database ${db}$ ,we can always (naively) materialize $Q\left( {db}\right)$ in an in-memory array $A$ . Unfortunately, $A$ then requires memory proportional to $\parallel Q\left( {db}\right) \parallel$ which, depending on the query, can be of size polynomial in $\parallel {db}\parallel$ . We hence search for other data structures that can represent $Q\left( {db}\right)$ using less space,while still allowing enumeration with the same (worst-case) complexity as enumeration from a materialized array $A$ : namely,with constant delay. The key idea to obtain this is delayed evaluation. To illustrate this, consider that we are asked to compute the Cartesian product of $R$ and $S$ . Then it suffices to simply store $R$ and $S$ ,requiring $O\left( {\parallel R\parallel  + \parallel S\parallel }\right)  = O\left( {\parallel {db}\parallel }\right)$ memory. To enumerate $R \times  S$ ,ENUM simply executes a nested-loop based Cartesian product over $R$ and $S$ . This satisfies the properties of CDE. Indeed,every element of $R \times  S$ will be output exactly once. Moreover, the time required to output the first element of $R \times  S$ is the time required to initialize a pointer to the first elements of $R$ and $S$ (hence constant). The time required to produce each subsequent element is bounded by the time required to either advance the pointer in $S$ ,or advance the pointer in $R$ and reset the pointer in $S$ to the beginning. In both cases,this is constant. Finally, checking whether we have reached the end of $R \times  S$ again takes constant time.

这个示例表明，为了对输入数据库${db}$上的查询$Q$的结果$Q\left( {db}\right)$进行连续延迟枚举，我们总是可以（简单地）将$Q\left( {db}\right)$物化到一个内存数组$A$中。不幸的是，$A$所需的内存与$\parallel Q\left( {db}\right) \parallel$成正比，根据查询的不同，其大小可能是$\parallel {db}\parallel$的多项式。因此，我们寻找其他数据结构，这些数据结构可以用更少的空间表示$Q\left( {db}\right)$，同时仍然允许以与从物化数组$A$进行枚举相同的（最坏情况）复杂度进行枚举，即具有常数延迟。实现这一点的关键思想是延迟求值。为了说明这一点，假设要求我们计算$R$和$S$的笛卡尔积。那么只需简单地存储$R$和$S$，所需内存为$O\left( {\parallel R\parallel  + \parallel S\parallel }\right)  = O\left( {\parallel {db}\parallel }\right)$。为了枚举$R \times  S$，ENUM只需对$R$和$S$执行基于嵌套循环的笛卡尔积。这满足连续延迟枚举的属性。实际上，$R \times  S$的每个元素都将被精确输出一次。此外，输出$R \times  S$的第一个元素所需的时间是初始化指向$R$和$S$的第一个元素的指针所需的时间（因此是常数时间）。生成每个后续元素所需的时间受限于推进$S$中的指针，或者推进$R$中的指针并将$S$中的指针重置到开头所需的时间。在这两种情况下，这都是常数时间。最后，检查是否已到达$R \times  S$的末尾同样只需常数时间。

The situation becomes more complex for queries that involve joins instead of Cartesian products. Consider for example the query $Q = R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$ . Simply delaying evaluation does not yield constant-delay enumeration. Indeed,suppose that we evaluate $Q$ using a simple in-memory hash join with $R$ as build relation and $S$ as probe relation. Assume that the corresponding index of $R$ on $B$ (i.e. the hash table) has already been computed. When iterating over $S$ to probe the hash table,we may have to visit an unbounded number of $S$ -tuples that do not join with any of the $R$ -tuples. Consequently,there is no constant that bounds the delay between consecutive outputs. A similar analysis shows that other join algorithms, such as the sort-merge join, do not yield enumeration with constant delay.

对于涉及连接操作而非笛卡尔积的查询，情况变得更加复杂。例如，考虑查询$Q = R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$。仅仅延迟求值并不能实现常数延迟枚举。实际上，假设我们使用简单的内存哈希连接来计算$Q$，其中$R$作为构建关系，$S$作为探测关系。假设$R$在$B$上的相应索引（即哈希表）已经计算好。当遍历$S$来探测哈希表时，我们可能需要访问数量不确定的$S$元组，这些元组与任何$R$元组都不连接。因此，不存在一个常数来限制连续输出之间的延迟。类似的分析表明，其他连接算法，如排序合并连接，也不能实现常数延迟枚举。

In essence, therefore, a data structure that allows CDE of $Q\left( {db}\right)$ must be able to produce all output tuples and their multiplicities without spending any extra time in building auxiliary data structures to help in enumeration (such as hash tables or sorted versions of the input relations), nor can it afford to waste time in processing input tuples that in the end do not appear in $Q\left( {db}\right)$ .

因此，从本质上讲，允许对$Q\left( {db}\right)$进行连续延迟枚举的数据结构必须能够生成所有输出元组及其重数，而无需花费额外的时间来构建辅助数据结构以帮助枚举（如哈希表或输入关系的排序版本），也不能浪费时间处理最终不会出现在$Q\left( {db}\right)$中的输入元组。

How do we obtain CDE for $R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$ ? Intuitively speaking, if in our hash join algorithm we can ensure to only iterate over those $S$ -tuples that have matching $R$ - records, we trivially obtain a CDE algorithm. In a broader sense, we need to maintain under updates, for the relations that are used as probe relations, the set of tuples that will match the corresponding build relation(s). We call these tuples the live tuples. In the following sections we gradually devise a more general algorithm that follows this idea. Intuitively, this algorithm dynamically maintains the hash tables and the live values for a query in a DCLR.

我们如何为 $R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$ 获得恒定延迟枚举（CDE）呢？直观地说，如果在我们的哈希连接算法中，我们能确保只遍历那些与 $R$ 记录匹配的 $S$ 元组，那么我们就能轻松得到一个 CDE 算法。从更广泛的意义上讲，对于用作探测关系的那些关系，我们需要在更新时维护与相应构建关系匹配的元组集合。我们将这些元组称为活动元组。在接下来的章节中，我们将逐步设计出一个遵循此思路的更通用的算法。直观地说，该算法会动态维护分布式并发逻辑规则（DCLR）中查询的哈希表和活动值。

### 4.2 Constant Delay Yannakakis

### 4.2 恒定延迟的亚纳卡基斯算法

Acyclic full join queries are evaluated in $O\left( {\parallel {db}\parallel  + \parallel Q\left( {db}\right) \parallel }\right)$ time by the well-known Yannakakis algorithm. For future reference, we recall the operation of the Yannakakis algorithm [39], formulated in our setting. We first need to introduce the semi-join operation for GMRs.

无环全连接查询可以通过著名的亚纳卡基斯（Yannakakis）算法在 $O\left( {\parallel {db}\parallel  + \parallel Q\left( {db}\right) \parallel }\right)$ 时间内完成评估。为了便于后续参考，我们回顾一下在我们的设定下亚纳卡基斯算法 [39] 的操作。首先，我们需要为广义多关系（GMR）引入半连接操作。

Definition 4.2. The semijoin $R \ltimes  S$ of a GMR $R\left\lbrack  \bar{x}\right\rbrack$ by a GMR $S$ is the GMR over $\bar{x}$ defined by

定义 4.2。广义多关系 $R\left\lbrack  \bar{x}\right\rbrack$ 与广义多关系 $S$ 的半连接 $R \ltimes  S$ 是定义在 $\bar{x}$ 上的广义多关系，其定义如下

$$
R \ltimes  S \in  \mathbb{{GMR}}\left\lbrack  \bar{x}\right\rbrack   : \overrightarrow{s} \mapsto  \left\{  \begin{array}{ll} R\left( \overrightarrow{s}\right) & \text{ if }\overrightarrow{s} \in  {\pi }_{\bar{x}}\left( {R \boxtimes  S}\right) \\  0 & \text{ otherwise. } \end{array}\right. 
$$

Classical Yannakakis. In its standard formulation, Yan-nakakis takes as input a traditional join tree $T$ for a join query $Q$ and a database ${db}$ on $Q$ . The algorithm starts by assigning a GMR ${R}_{n}$ over $\operatorname{var}\left( n\right)$ to each node $n$ in $T$ . Initially, ${R}_{n} \mathrel{\text{:=}} d{b}_{n}$ . The algorithm then works in three stages.

经典的亚纳卡基斯算法。在其标准形式中，亚纳卡基斯算法将连接查询 $Q$ 的传统连接树 $T$ 和基于 $Q$ 的数据库 ${db}$ 作为输入。该算法首先为 $T$ 中的每个节点 $n$ 分配一个基于 $\operatorname{var}\left( n\right)$ 的广义多关系 ${R}_{n}$。初始时，${R}_{n} \mathrel{\text{:=}} d{b}_{n}$。然后，该算法分三个阶段进行工作。

(1) The nodes of $T$ are visited in some bottom-up traversal order of $T$ . When node $n$ is visited in this order,its parent $p$ is considered and ${R}_{p}$ is updated to ${R}_{p} \mathrel{\text{:=}} {R}_{p} \ltimes  {R}_{n}$ .

(1) 按照 $T$ 的某种自底向上的遍历顺序访问 $T$ 的节点。当按此顺序访问节点 $n$ 时，会考虑其父节点 $p$，并将 ${R}_{p}$ 更新为 ${R}_{p} \mathrel{\text{:=}} {R}_{p} \ltimes  {R}_{n}$。

(2) The nodes of $T$ are visited in a top-down traversal order. When node $n$ is visited in this order,each child $c$ of $n$ is considered,and ${R}_{c}$ is updated to ${R}_{c} \mathrel{\text{:=}} {R}_{c} \ltimes  {R}_{n}$ .

(2) 按照自顶向下的遍历顺序访问 $T$ 的节点。当按此顺序访问节点 $n$ 时，会考虑 $n$ 的每个子节点 $c$，并将 ${R}_{c}$ 更新为 ${R}_{c} \mathrel{\text{:=}} {R}_{c} \ltimes  {R}_{n}$。

(3) The interior nodes of $T$ are again visited in a bottom-up order. In this stage, however, the actual join results are computed: when node $n$ with children ${c}_{1},\ldots ,{c}_{k}$ is visited,its GMR is updated to ${R}_{n} \mathrel{\text{:=}} {R}_{{c}_{1}} \boxtimes  \cdots  \boxtimes  {R}_{{c}_{k}}$ .

(3) 再次按照自底向上的顺序访问 $T$ 的内部节点。然而，在这个阶段，会计算实际的连接结果：当访问具有子节点 ${c}_{1},\ldots ,{c}_{k}$ 的节点 $n$ 时，其广义多关系会更新为 ${R}_{n} \mathrel{\text{:=}} {R}_{{c}_{1}} \boxtimes  \cdots  \boxtimes  {R}_{{c}_{k}}$。

After the final stage, the GMR materialized at the root is precisely $Q\left( {db}\right)$ . The initialization together with stages (1) and (2) run in time $O\left( {\parallel {db}\parallel }\right)$ while stage 3 can be shown to run in time $O\left( {\parallel Q\left( {db}\right) \parallel }\right)$ . It is worth noting that the Yan-nakakis algorithm fully materializes the query result $Q\left( {db}\right)$ at the root,requiring $O\left( {\parallel Q\left( {db}\right) \parallel }\right)$ space. Notice also that this algorithm works over the static setting, and does not consider updates.

在最后一个阶段之后，在根节点具体化的广义多关系恰好是 $Q\left( {db}\right)$。初始化以及阶段 (1) 和 (2) 的运行时间为 $O\left( {\parallel {db}\parallel }\right)$，而阶段 3 的运行时间可以证明为 $O\left( {\parallel Q\left( {db}\right) \parallel }\right)$。值得注意的是，亚纳卡基斯算法会在根节点完全具体化查询结果 $Q\left( {db}\right)$，需要 $O\left( {\parallel Q\left( {db}\right) \parallel }\right)$ 的空间。还要注意，该算法是在静态环境下工作的，没有考虑更新操作。

To extend Yannakakis to work on generalized join trees in addition to traditional join trees, one only needs to modify the initialization step as follows. If $n$ is a hyperedge,simply set ${R}_{n} \mathrel{\text{:=}} {\pi }_{\operatorname{var}\left( n\right) }{R}_{c}$ for some arbitrary but fixed $c \in  \operatorname{grd}\left( n\right)$ (which we may assume to have been initialized before if we initialize in a bottom-up fashion). It is not difficult to see that,with this initalization,every hyperedge $n$ has ${R}_{n} =$ ${\pi }_{\operatorname{var}\left( n\right) }d{b}_{\mathbf{a}}$ for some descendant atom $\mathbf{a}$ of $n$ . In other words, ${R}_{n}$ is the projection of some input atom. This ensures that, even on generalized join trees, Yannakakis exhibits the same complexity guarantees.

为了让亚纳卡基斯算法（Yannakakis）除了能处理传统连接树之外，还能处理广义连接树，只需按如下方式修改初始化步骤。如果 $n$ 是一条超边，只需为某个任意但固定的 $c \in  \operatorname{grd}\left( n\right)$ 设置 ${R}_{n} \mathrel{\text{:=}} {\pi }_{\operatorname{var}\left( n\right) }{R}_{c}$（如果我们采用自底向上的方式进行初始化，可假设 $c \in  \operatorname{grd}\left( n\right)$ 在此之前已完成初始化）。不难看出，通过这种初始化方式，每条超边 $n$ 都存在某个 $n$ 的后代原子 $\mathbf{a}$ 使得 ${R}_{n} =$ ${\pi }_{\operatorname{var}\left( n\right) }d{b}_{\mathbf{a}}$ 成立。换句话说，${R}_{n}$ 是某个输入原子的投影。这就确保了，即使在广义连接树上，亚纳卡基斯算法（Yannakakis）也能保证相同的复杂度。

Yannakakis with constant delay enumeration (CDY). Our dynamic query processing algorithm is based on the simple observation that, after the first bottom-up traversal stage,the join query result $Q\left( {db}\right)$ can be enumerated with constant delay. As such, there is no need to materialize the query result in stage 3. To illustrate this claim, consider the following variant of the Classical Yannakakis algorithm, called CDY for Constant Delay Yannakakis.

具有恒定延迟枚举功能的亚纳卡基斯算法（CDY）。我们的动态查询处理算法基于一个简单的观察结果：在第一个自底向上的遍历阶段之后，连接查询结果 $Q\left( {db}\right)$ 可以以恒定延迟进行枚举。因此，在阶段 3 无需物化查询结果。为了说明这一观点，考虑经典亚纳卡基斯算法（Classical Yannakakis）的以下变体，称为具有恒定延迟的亚纳卡基斯算法（CDY，Constant Delay Yannakakis）。

(1) Do the first stage of Classical Yannakakis.

(1) 执行经典亚纳卡基斯算法（Classical Yannakakis）的第一阶段。

(2) For each node $n$ construct an index ${L}_{n}$ of ${R}_{n}$ on $\operatorname{pvar}\left( n\right)$ .

(2) 为每个节点 $n$ 构建 ${R}_{n}$ 在 $\operatorname{pvar}\left( n\right)$ 上的索引 ${L}_{n}$。

Given this pre-processing, the constant-delay enumeration method ENUM is essentially a multi-way hash join, where the GMR materialized at the root is used as probe relation, and the other ${R}_{n}$ as build relations,with the hash tables given by ${L}_{n}$ . Because of the way in which ${R}_{n}$ is computed,we are ensured that for every probe we will have matching join tuples, ensuring constant-delay enumeration. Note, moreover, that the GMRs materialized after the first step of the Yannakakis algorithm, as well as the constructed indexes, require $O\left( {\parallel {db}\parallel }\right)$ space. We delay a formal definition of the enumeration algorithm until Section 4.5, but illustrate its working by means of the following example.

有了这种预处理，恒定延迟枚举方法 ENUM 本质上是一种多路哈希连接，其中在根节点物化的广义匹配关系（GMR）用作探测关系，其他 ${R}_{n}$ 用作构建关系，哈希表由 ${L}_{n}$ 给出。由于 ${R}_{n}$ 的计算方式，我们可以确保每次探测都能找到匹配的连接元组，从而保证恒定延迟枚举。此外，请注意，在亚纳卡基斯算法（Yannakakis）的第一步之后物化的广义匹配关系（GMR）以及构建的索引需要 $O\left( {\parallel {db}\parallel }\right)$ 的空间。我们将枚举算法的正式定义推迟到第 4.5 节，但通过以下示例来说明其工作原理。

Example 4.3. Consider generalized join tree ${T}_{3}$ of Figure 2. ENUM works as follows. Let $\overrightarrow{s}$ be the empty tuple. Then ENUM is defined by:

示例 4.3。考虑图 2 中的广义连接树 ${T}_{3}$。ENUM 的工作方式如下。设 $\overrightarrow{s}$ 为空元组。则 ENUM 定义如下：

for each ${\overrightarrow{t}}_{\left\lbrack  y\right\rbrack  } \in  {L}_{\left\lbrack  y\right\rbrack  }\left( \overrightarrow{s}\right)$ do

对于每个 ${\overrightarrow{t}}_{\left\lbrack  y\right\rbrack  } \in  {L}_{\left\lbrack  y\right\rbrack  }\left( \overrightarrow{s}\right)$ 执行

for each ${\overrightarrow{t}}_{\left\lbrack  x,y,z\right\rbrack  } \in  {L}_{\left\lbrack  x,y,z\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y\right\rbrack  }\right)$ do

对于每个 ${\overrightarrow{t}}_{\left\lbrack  x,y,z\right\rbrack  } \in  {L}_{\left\lbrack  x,y,z\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y\right\rbrack  }\right)$ 执行

for each $\left( {{\overrightarrow{t}}_{R},{\mu }_{R}}\right)  \in  {L}_{R\left( {x,y,z}\right) }\left( {\overrightarrow{t}}_{\left\lbrack  x,y,z\right\rbrack  }\right)$ do

对于每个 $\left( {{\overrightarrow{t}}_{R},{\mu }_{R}}\right)  \in  {L}_{R\left( {x,y,z}\right) }\left( {\overrightarrow{t}}_{\left\lbrack  x,y,z\right\rbrack  }\right)$ 执行

for each $\left( {{\overrightarrow{t}}_{S},{\mu }_{S}}\right)  \in  {L}_{S\left( {y,v,w}\right) }\left( {{\overrightarrow{t}}_{\left\lbrack  y,v,w\right\rbrack  }\left\lbrack  {y,v}\right\rbrack  }\right)$ do

对于每个 $\left( {{\overrightarrow{t}}_{S},{\mu }_{S}}\right)  \in  {L}_{S\left( {y,v,w}\right) }\left( {{\overrightarrow{t}}_{\left\lbrack  y,v,w\right\rbrack  }\left\lbrack  {y,v}\right\rbrack  }\right)$ 执行

for each ${\overrightarrow{t}}_{\left\lbrack  y,v\right\rbrack  } \in  {L}_{\left\lbrack  y,v\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y\right\rbrack  }\right)$ do

对于每个 ${\overrightarrow{t}}_{\left\lbrack  y,v\right\rbrack  } \in  {L}_{\left\lbrack  y,v\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y\right\rbrack  }\right)$ 执行

for each $\left( {{\overrightarrow{t}}_{T},{\mu }_{T}}\right)  \in  {L}_{T\left\lbrack  {y,v,w}\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y,v\right\rbrack  }\right)$ do

对于每个 $\left( {{\overrightarrow{t}}_{T},{\mu }_{T}}\right)  \in  {L}_{T\left\lbrack  {y,v,w}\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y,v\right\rbrack  }\right)$ 执行

for each $\left( {{\overrightarrow{t}}_{U},{\mu }_{T}}\right)  \in  {L}_{U\left\lbrack  {y,v,p}\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y,v\right\rbrack  }\right)$ do

对于每个 $\left( {{\overrightarrow{t}}_{U},{\mu }_{T}}\right)  \in  {L}_{U\left\lbrack  {y,v,p}\right\rbrack  }\left( {\overrightarrow{t}}_{\left\lbrack  y,v\right\rbrack  }\right)$ 执行

output $\left( {\overrightarrow{{t}_{R}}\overrightarrow{{t}_{S}}\overrightarrow{{t}_{T}}\overrightarrow{{t}_{U}},{\mu }_{R} * {\mu }_{S} * {\mu }_{T} * {\mu }_{U}}\right)$

输出 $\left( {\overrightarrow{{t}_{R}}\overrightarrow{{t}_{S}}\overrightarrow{{t}_{T}}\overrightarrow{{t}_{U}},{\mu }_{R} * {\mu }_{S} * {\mu }_{T} * {\mu }_{U}}\right)$

From our discussion so far, we obain:

从我们目前的讨论中，我们得到：

Proposition 4.4. Given an acyclic full join query $Q$ ,a join tree $T$ of $Q$ and a database ${db},{CDY}\left( {T,{db}}\right)$ runs in time $O\left( {\parallel {db}\parallel }\right)$ using space $O\left( {\parallel {db}\parallel }\right)$ . Once ${CDY}$ has completed, ENUM effectively enumerates $Q\left( {db}\right)$ with constant delay.

命题 4.4。给定一个无环全连接查询 $Q$、$Q$ 的一个连接树 $T$ 和一个数据库 ${db},{CDY}\left( {T,{db}}\right)$，使用空间 $O\left( {\parallel {db}\parallel }\right)$ 在时间 $O\left( {\parallel {db}\parallel }\right)$ 内运行。一旦 ${CDY}$ 完成，ENUM 以恒定延迟有效地枚举 $Q\left( {db}\right)$。

### 4.3 Dynamic Yannakakis

### 4.3 动态扬纳卡基斯算法

Definition 4.5. Let $T$ be a join tree and ${db}$ a database. Let ${R}_{n}$ ,for $n \in  T$ ,be the GMR associated to $n$ after executing the first stage of the Yannakakis algorithm. A tuple $\overrightarrow{t}$ is called live in(db,n)w.r.t. $T$ if $\overrightarrow{t} \in  {R}_{n}{.}^{2}$

定义 4.5。设 $T$ 为一个连接树，${db}$ 为一个数据库。设 ${R}_{n}$（对于 $n \in  T$）为执行扬纳卡基斯算法第一阶段后与 $n$ 相关联的广义匹配关系（GMR）。如果 $\overrightarrow{t} \in  {R}_{n}{.}^{2}$，则元组 $\overrightarrow{t}$ 相对于 $T$ 在 (db,n) 中被称为活跃的。

CDY shows that we can suitably index the live tuples to enumerate $Q\left( {db}\right)$ with constant delay. To turn CDY into a dynamic algorithm, it hence suffices to maintain the live tuples and indices under updates. A naive approach for doing this would be to re-run CDY from scratch whenever the database is updated. This would spend time linear in the size of the updated database. Of course, this naive approach introduces unnecessary overhead. Indeed, consider an update that inserts a single tuple to an atom $\mathbf{a}$ . In that case, only the set of live tuples associated to $\mathbf{a}$ and its ancestors in join tree $T$ can change,while the rest of the nodes would remain unchanged. Moreover, the new set of live tuples of $\mathbf{a}$ and its ancestors can be computed incrementally. In At the end of Section 5, we will see in particular that avoiding naive recomputation is highly effective in practice.

CDY算法表明，我们可以对活跃元组进行适当的索引，以恒定延迟枚举$Q\left( {db}\right)$。为了将CDY算法转变为动态算法，因此只需在更新时维护活跃元组和索引即可。一种简单的做法是，每当数据库更新时，从头重新运行CDY算法。这将花费与更新后数据库大小成线性关系的时间。当然，这种简单的方法会引入不必要的开销。实际上，考虑一个向原子$\mathbf{a}$插入单个元组的更新操作。在这种情况下，只有与$\mathbf{a}$及其在连接树$T$中的祖先相关联的活跃元组集合会发生变化，而其余节点将保持不变。此外，$\mathbf{a}$及其祖先的新活跃元组集合可以增量计算。在第5节末尾，我们将特别看到，在实践中避免简单的重新计算非常有效。

In order to be able to explain how we maintain the live tuples incrementally, we require the following definitions.

为了能够解释我们如何增量维护活跃元组，我们需要以下定义。

Definition 4.6. Let $T$ be a join tree. To every node $n$ of $T$ we associate two queries, ${\Lambda }_{n}^{T}$ and ${\Psi }_{n}^{T}$ ,over $\operatorname{var}\left( n\right)$ and pvar(n),respectively. To every hyperedge $n$ of $T$ we also associate an additional query ${\Gamma }_{n}^{T}$ over $\operatorname{var}\left( n\right)$ . The definition of these queries is recursive: for each atom $\mathbf{a}$ we define ${\Lambda }_{\mathbf{a}}^{T}$ simply as $\mathbf{a}$ ,and ${\Psi }_{\mathbf{a}}^{T} \mathrel{\text{:=}} {\pi }_{\operatorname{pvar}\left( \mathbf{a}\right) }\mathbf{a}$ . Then,in a bottom-up traversal order,for every hyperedge $n$ we define

定义4.6。设$T$为一个连接树。对于$T$的每个节点$n$，我们分别在$\operatorname{var}\left( n\right)$和pvar(n)上关联两个查询${\Lambda }_{n}^{T}$和${\Psi }_{n}^{T}$。对于$T$的每条超边$n$，我们还在$\operatorname{var}\left( n\right)$上关联一个额外的查询${\Gamma }_{n}^{T}$。这些查询的定义是递归的：对于每个原子$\mathbf{a}$，我们简单地将${\Lambda }_{\mathbf{a}}^{T}$定义为$\mathbf{a}$，以及${\Psi }_{\mathbf{a}}^{T} \mathrel{\text{:=}} {\pi }_{\operatorname{pvar}\left( \mathbf{a}\right) }\mathbf{a}$。然后，按照自底向上的遍历顺序，对于每条超边$n$，我们定义

$$
{\Lambda }_{n}^{T} \mathrel{\text{:=}} {\Gamma }_{n}^{T} \boxtimes  {\mathbb{M}}_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}^{T}\;{\Psi }_{n}^{T} \mathrel{\text{:=}} {\pi }_{\operatorname{pvar}\left( n\right) }{\Lambda }_{n}^{T}
$$

$$
{\Gamma }_{n}^{T} \mathrel{\text{:=}} {\mathbb{M}}_{c \in  \operatorname{grd}\left( n\right) }{\Psi }_{c}
$$

We often omit the superscript if it is clear from the context. Intuitively, ${\Lambda }_{n}$ contains the set of live tuples of $n$ ,while ${\Psi }_{n}$ and ${\Gamma }_{n}$ are auxiliary queries that will help maintain ${\Lambda }_{n}$ under updates. The following proposition shows that, indeed,the queries ${\Lambda }_{n}$ characterize the live tuples. The proof (omitted) is by induction on the height of node $n$ in $T$ .

如果上下文明确，我们通常会省略上标。直观地说，${\Lambda }_{n}$包含$n$的活跃元组集合，而${\Psi }_{n}$和${\Gamma }_{n}$是辅助查询，将有助于在更新时维护${\Lambda }_{n}$。以下命题表明，实际上，查询${\Lambda }_{n}$刻画了活跃元组。证明（省略）是通过对$T$中节点$n$的高度进行归纳得出的。

Proposition 4.7. A tuple $\overrightarrow{t}$ is live in(db,n)w.r.t. join tree $T$ if,and only if, $\overrightarrow{t} \in  {\Lambda }_{n}^{T}\left( {db}\right)$ .

命题4.7。元组$\overrightarrow{t}$在(db,n)中相对于连接树$T$是活跃的，当且仅当$\overrightarrow{t} \in  {\Lambda }_{n}^{T}\left( {db}\right)$。

We can now define the data structure maintained by DYN.

我们现在可以定义DYN算法维护的数据结构。

Definition 4.8 (T-representation). Let $T$ be a join tree and let ${db}$ be a database. A $T$ -representation ( $T$ -rep for short) of ${db}$ is a data structure $\mathcal{D}$ that for each node $n$ of $T$ contains: - an index ${L}_{n}$ of ${\Lambda }_{n}\left( {db}\right)$ on $\operatorname{pvar}\left( n\right)$ ;

定义4.8（T表示）。设$T$为一个连接树，${db}$为一个数据库。${db}$的$T$表示（简称$T$ - rep）是一个数据结构$\mathcal{D}$，对于$T$的每个节点$n$，它包含： - ${\Lambda }_{n}\left( {db}\right)$在$\operatorname{pvar}\left( n\right)$上的一个索引${L}_{n}$；

- a GMR ${P}_{n}$ that materializes ${\Psi }_{n}\left( {db}\right)$ ,i.e., ${P}_{n} = {\Psi }_{n}\left( {db}\right)$ ;

- 一个物化${\Psi }_{n}\left( {db}\right)$的GMR ${P}_{n}$，即${P}_{n} = {\Psi }_{n}\left( {db}\right)$；

- a GMR ${G}_{n}$ that materializes ${\Gamma }_{n}\left( {db}\right)$ ,i.e., ${G}_{n} = {\Gamma }_{n}\left( {db}\right)$ ;

- 一个物化${\Gamma }_{n}\left( {db}\right)$的GMR ${G}_{n}$，即${G}_{n} = {\Gamma }_{n}\left( {db}\right)$；

- for every non-guard child $c \in  \mathrm{{ng}}\left( n\right)$ ,an index ${G}_{n,c}$ of ${G}_{n}$ on pvar(c).

- 对于每个非保护子节点$c \in  \mathrm{{ng}}\left( n\right)$，${G}_{n}$在pvar(c)上的一个索引${G}_{n,c}$。

Example 4.9. Consider join tree ${T}_{3}$ from Figure 2. In Figure 3,we show the ${T}_{3}$ -rep $\mathcal{D}$ for the database ${db}$ consisting of the GMRs $R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right)$ presented at the leaves of Figure 3. For each node $n$ ,the live tuples ${L}_{n} = {\Lambda }_{n}\left( {db}\right)$ are given by the white-colored tables (shown below $n$ ) while ${P}_{n} = {\Psi }_{n}\left( {db}\right)$ is given by the gray-colored tables (shown above $n$ on the edge from $n$ to its parent). For reasons of parsimony, we do not show the ${G}_{n}$ : for $\left\lbrack  y\right\rbrack$ and $\left\lbrack  {y,v}\right\rbrack$ this equals ${L}_{n}$ ; for $\left\lbrack  {x,y,z}\right\rbrack$ this equals ${L}_{R\left( {x,y,z}\right) }$ . The indexes are likewise not shown.

示例4.9。考虑图2中的连接树${T}_{3}$。在图3中，我们展示了由图3叶子节点处给出的广义匹配规则（GMR）$R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right)$组成的数据库${db}$的${T}_{3}$ - 表示$\mathcal{D}$。对于每个节点$n$，活动元组${L}_{n} = {\Lambda }_{n}\left( {db}\right)$由白色表格给出（显示在$n$下方），而${P}_{n} = {\Psi }_{n}\left( {db}\right)$由灰色表格给出（显示在从$n$到其父节点的边上，位于$n$上方）。出于简洁性考虑，我们不展示${G}_{n}$：对于$\left\lbrack  y\right\rbrack$和$\left\lbrack  {y,v}\right\rbrack$，其等于${L}_{n}$；对于$\left\lbrack  {x,y,z}\right\rbrack$，其等于${L}_{R\left( {x,y,z}\right) }$。索引同样未展示。

A first important feature of $T$ -representations is that they use only linear space. Proposition 4.10. Let $T$ be a join tree and $\mathcal{D}$ a $T$ -rep of ${db}$ . Then $\parallel \mathcal{D}\parallel  = O\left( {\parallel {db}\parallel }\right)$ .

$T$ - 表示的第一个重要特征是它们仅使用线性空间。命题4.10。设$T$为一个连接树，$\mathcal{D}$为${db}$的一个$T$ - 表示。那么$\parallel \mathcal{D}\parallel  = O\left( {\parallel {db}\parallel }\right)$。

---

<!-- Footnote -->

${}^{2}$ Recall that $\overrightarrow{t} \in  S$ indicates that $S\left( \overrightarrow{t}\right)  \neq  0$ .

${}^{2}$ 回顾一下，$\overrightarrow{t} \in  S$ 表示 $S\left( \overrightarrow{t}\right)  \neq  0$。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 2 | 2 | $\left\lbrack  y\right\rbrack$ (   ) 8 $\left\lbrack  {y,v}\right\rbrack$ $\left\lbrack  {x,y,z}\right\rbrack$ $\begin{array}{lll} 1 & 2 & 3 \end{array}$ $R\left( {x,y,z}\right)$ $S\left( {x,y,u}\right)$ -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_6.jpg?x=225&y=146&w=557&h=339&r=0"/>

Figure 3: Illustration of $T$ -representations (Example 4.9).

图3：$T$ - 表示的图示（示例4.9）。

<!-- Media -->

The crux to prove this proposition lies in observing that, as illustrated in Figure 3,for all nodes $n$ ,if $\overrightarrow{t}$ is in ${\Lambda }_{n}\left( {db}\right)$ , ${\Psi }_{n}\left( {db}\right)$ ,or ${\Gamma }_{n}\left( {db}\right)$ ,then there is some descendant atom $\mathbf{a} \in$ $T$ such that $\overrightarrow{t} \in  {\pi }_{\bar{x}}\left( {d{b}_{a}}\right)$ with $\bar{x} = \operatorname{var}\left( n\right)$ or $\bar{x} = \operatorname{pvar}\left( n\right)$ . Therefore, ${\Lambda }_{n}\left( {db}\right) ,{\Psi }_{n}\left( {db}\right)$ ,and ${\Gamma }_{n}\left( {db}\right)$ as well as indexes thereon,all take space $O\left( {\parallel {db}\parallel }\right)$ .

证明这个命题的关键在于观察到，如图3所示，对于所有节点$n$，如果$\overrightarrow{t}$在${\Lambda }_{n}\left( {db}\right)$、${\Psi }_{n}\left( {db}\right)$或${\Gamma }_{n}\left( {db}\right)$中，那么存在某个后代原子$\mathbf{a} \in$ $T$，使得$\overrightarrow{t} \in  {\pi }_{\bar{x}}\left( {d{b}_{a}}\right)$ 且 $\bar{x} = \operatorname{var}\left( n\right)$ 或 $\bar{x} = \operatorname{pvar}\left( n\right)$。因此，${\Lambda }_{n}\left( {db}\right) ,{\Psi }_{n}\left( {db}\right)$ 以及 ${\Gamma }_{n}\left( {db}\right)$ 及其上的索引，都占用空间 $O\left( {\parallel {db}\parallel }\right)$。

Dynamic Yannakakis. We now describe the Dynamic Yan-nakakis algorithm (DYN) presented in Algorithm 1. DYN maintains $T$ -representations under updates. To explicitly assert the join-tree over which DYN operates we write ${\mathrm{{DYN}}}_{T}$ .

动态亚纳卡基斯算法。我们现在描述算法1中给出的动态亚纳卡基斯算法（DYN）。DYN在更新操作下维护$T$ - 表示。为了明确指出DYN所操作的连接树，我们记为${\mathrm{{DYN}}}_{T}$。

Like classical Yannakakis, ${\mathrm{{DYN}}}_{T}$ traverses the nodes of $T$ in a bottom-up fashion upon update $u$ . During this traversal,the goal is to materialize,for each node $n$ ,the deltas $\Delta {\Lambda }_{n}\left( {{db},u}\right) ,\Delta {\Psi }_{n}\left( {{db},u}\right)$ ,and $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ into GMRs $\Delta {L}_{n}$ , $\Delta {P}_{n}$ ,and $\Delta {G}_{n}$ ,respectively. These represent the updates that we need to apply to $\mathcal{D}$ ’s components in order to obtain a $T$ -rep for ${db} + u$ . This application happens in lines $6 - {10}$ .

与经典的亚纳卡基斯算法类似，${\mathrm{{DYN}}}_{T}$ 在更新 $u$ 时以自底向上的方式遍历 $T$ 的节点。在这个遍历过程中，目标是分别将每个节点 $n$ 的增量 $\Delta {\Lambda }_{n}\left( {{db},u}\right) ,\Delta {\Psi }_{n}\left( {{db},u}\right)$、$\Delta {\Gamma }_{n}\left( {{db},u}\right)$ 实例化为广义多关系（GMR）$\Delta {L}_{n}$、$\Delta {P}_{n}$ 和 $\Delta {G}_{n}$。这些表示为了获得 ${db} + u$ 的 $T$ -表示，我们需要应用到 $\mathcal{D}$ 的组件上的更新。这个应用过程在第 $6 - {10}$ 行完成。

The delta GMRs are computed as follows. When $n$ is an atom $\mathbf{a},{\mathrm{{DYN}}}_{T}$ uses the update $u$ to compute $\Delta {L}_{\mathbf{a}} = {u}_{\mathbf{a}}$ and $\Delta {P}_{\mathbf{a}} = {\pi }_{\operatorname{pvar}\left( \mathbf{a}\right) }{u}_{\mathbf{a}}$ . The latter projection can be done using a simple hash-based aggregation algorithm. When $n$ is a hyperedge, ${\mathrm{{DYN}}}_{T}$ uses Algorithm 2 to compute $\Delta {L}_{n}$ , $\Delta {P}_{n}$ and $\Delta {G}_{n}$ . This algorithm uses the materialized index of ${\Lambda }_{n}\left( {db}\right)$ on $\operatorname{pvar}\left( n\right)$ ,and the materialized GMRs ${P}_{n} =$ ${\Psi }_{n}\left( {db}\right)$ and ${G}_{n} = {\Gamma }_{n}\left( {db}\right)$ ,which are already available in $\mathcal{D}$ . In addition,it uses the delta GMRs $\Delta {P}_{c}$ for each child $c$ of $n$ ,which was previously computed when visiting $c$ . In order to compute $\Delta {L}_{n},\Delta {P}_{n}$ ,and $\Delta {G}_{n}$ efficiently,we use the following insight, proven in the Appendix B.

增量广义多关系（GMR）的计算方式如下。当 $n$ 是一个原子时，$\mathbf{a},{\mathrm{{DYN}}}_{T}$ 使用更新 $u$ 来计算 $\Delta {L}_{\mathbf{a}} = {u}_{\mathbf{a}}$ 和 $\Delta {P}_{\mathbf{a}} = {\pi }_{\operatorname{pvar}\left( \mathbf{a}\right) }{u}_{\mathbf{a}}$。后一个投影可以使用简单的基于哈希的聚合算法来完成。当 $n$ 是一条超边时，${\mathrm{{DYN}}}_{T}$ 使用算法 2 来计算 $\Delta {L}_{n}$、$\Delta {P}_{n}$ 和 $\Delta {G}_{n}$。该算法使用 ${\Lambda }_{n}\left( {db}\right)$ 在 $\operatorname{pvar}\left( n\right)$ 上的实例化索引，以及已经在 $\mathcal{D}$ 中可用的实例化广义多关系（GMR）${P}_{n} =$、${\Psi }_{n}\left( {db}\right)$ 和 ${G}_{n} = {\Gamma }_{n}\left( {db}\right)$。此外，它还使用 $n$ 的每个子节点 $c$ 的增量广义多关系（GMR）$\Delta {P}_{c}$，这些增量广义多关系（GMR）是在访问 $c$ 时之前计算得到的。为了高效地计算 $\Delta {L}_{n},\Delta {P}_{n}$ 和 $\Delta {G}_{n}$，我们使用附录 B 中证明的以下见解。

Lemma 4.11. If $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)$ then $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ for some guard $c \in  \operatorname{grd}\left( n\right)$ . Moreover,if $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$ then either (1) $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ for some guard $c \in  \operatorname{grd}\left( n\right)$ or (2) $\overrightarrow{t} \in  \left( {{\Gamma }_{n}\left( {db}\right)  \ltimes  \Delta {\Psi }_{c}\left( {{db},u}\right) }\right)$ for some child $c \in  \operatorname{ng}\left( n\right)$ .

引理 4.11。如果 $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)$，那么对于某个保护条件 $c \in  \operatorname{grd}\left( n\right)$ 有 $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$。此外，如果 $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$，那么要么（1）对于某个保护条件 $c \in  \operatorname{grd}\left( n\right)$ 有 $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$，要么（2）对于某个子节点 $c \in  \operatorname{ng}\left( n\right)$ 有 $\overrightarrow{t} \in  \left( {{\Gamma }_{n}\left( {db}\right)  \ltimes  \Delta {\Psi }_{c}\left( {{db},u}\right) }\right)$。

Algorithm 2 uses Lemma 4.11 to compute a bound on $\operatorname{supp}\left( {\Delta {\Lambda }_{n}\left( {{db},u}\right) }\right)$ . In particular,in lines 2-4 it computes

算法 2 使用引理 4.11 来计算 $\operatorname{supp}\left( {\Delta {\Lambda }_{n}\left( {{db},u}\right) }\right)$ 的一个边界。具体来说，在第 2 - 4 行中，它计算

$$
U = \mathop{\bigcup }\limits_{{c \in  \operatorname{grd}\left( n\right) }}\operatorname{supp}\left( {\Delta {P}_{c}}\right)  \cup  \mathop{\bigcup }\limits_{{c \in  \operatorname{ng}\left( n\right) }}\operatorname{supp}\left( {{G}_{n} \ltimes  \Delta {P}_{c}}\right) .
$$

As such, $U$ contains all tuples that can appear in $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ or $\Delta {\Lambda }_{n}\left( {{db},u}\right)$ . Lines 5-8 compute $\Delta {G}_{n} = \Delta {\Gamma }_{n}\left( {{db},u}\right)$ , $\Delta {L}_{n} = \Delta {\Lambda }_{n}\left( {{db},u}\right)$ and $\Delta {P}_{n} = \Delta {\Psi }_{n}\left( {{db},u}\right)$ by iterating over the tuples in $U$ and using the fact that ${\Psi }_{c}\left( {{db} + u}\right) \left( \overrightarrow{s}\right)  =$ ${P}_{c}\left( \overrightarrow{s}\right)  + \Delta {P}_{c}\left( \overrightarrow{s}\right)$ ,for every tuple $\overrightarrow{s}$ . From Lemma 4.11 and our explanation so far, we hence obtain:

因此，$U$ 包含了所有可能出现在 $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ 或 $\Delta {\Lambda }_{n}\left( {{db},u}\right)$ 中的元组。第 5 - 8 行通过遍历 $U$ 中的元组，并利用对于每个元组 $\overrightarrow{s}$ 都有 ${\Psi }_{c}\left( {{db} + u}\right) \left( \overrightarrow{s}\right)  =$ ${P}_{c}\left( \overrightarrow{s}\right)  + \Delta {P}_{c}\left( \overrightarrow{s}\right)$ 这一事实，来计算 $\Delta {G}_{n} = \Delta {\Gamma }_{n}\left( {{db},u}\right)$、$\Delta {L}_{n} = \Delta {\Lambda }_{n}\left( {{db},u}\right)$ 和 $\Delta {P}_{n} = \Delta {\Psi }_{n}\left( {{db},u}\right)$。根据引理 4.11 以及我们到目前为止的解释，我们因此得到：

<!-- Media -->

Algorithm $1{\mathrm{{DYN}}}_{T}$ : Update trigger maintaining $T$ -rep $\mathcal{D}$ under update $u$

算法 $1{\mathrm{{DYN}}}_{T}$：在更新 $u$ 的情况下维护 $T$ - 表示 $\mathcal{D}$ 的更新触发器

---

Assume: $T$ is a join tree

假设：$T$ 是一个连接树

Input: $T$ -rep $\mathcal{D}$ for(db); and update $u$

输入：数据库（db）的 $T$ - 表示 $\mathcal{D}$；以及更新 $u$

Output: T-rep for ${db} + u$ .

输出：${db} + u$ 的 T - 表示。

for each node $n \in  T$ ,visited in bottom-up order do

对于以自底向上顺序访问的每个节点 $n \in  T$ 执行

	compute $\Delta {L}_{n},\Delta {P}_{n}$ ,and $\Delta {G}_{n}$ (if applicable)

	计算 $\Delta {L}_{n},\Delta {P}_{n}$，以及 $\Delta {G}_{n}$（如果适用）

for each node $n \in  T$ do

对于每个节点 $n \in  T$ 执行

	${L}_{n} \mathrel{\text{+=}} \Delta {L}_{n};\;{P}_{n} +  = \Delta {P}_{n}$

	if $n$ is a hyperedge then

	如果 $n$ 是一个超边，则

		${G}_{n} +  = \Delta {G}_{n}$

		for each $c \in  \operatorname{ng}\left( n\right)$ do ${G}_{n,c} +  = \Delta {G}_{n}$

		对于每个 $c \in  \operatorname{ng}\left( n\right)$ 执行 ${G}_{n,c} +  = \Delta {G}_{n}$

---

Algorithm 2 Delta computation for hyperedge $n$

算法 2 超边 $n$ 的增量计算

---

1: Initialize $\Delta {L}_{n},\Delta {P}_{n}$ and $\Delta {G}_{n}$ to the empty GMRs

1: 将 $\Delta {L}_{n},\Delta {P}_{n}$ 和 $\Delta {G}_{n}$ 初始化为空的广义多集表示（GMR）

	Initialize $U \mathrel{\text{:=}} \mathop{\bigcup }\limits_{{c \in  \operatorname{grd}\left( n\right) }}\operatorname{supp}\left( {\Delta {P}_{c}}\right)$

	初始化 $U \mathrel{\text{:=}} \mathop{\bigcup }\limits_{{c \in  \operatorname{grd}\left( n\right) }}\operatorname{supp}\left( {\Delta {P}_{c}}\right)$

	for each $c \in  \operatorname{ng}\left( n\right)$ and each ${\overrightarrow{t}}_{c} \in  \Delta {P}_{c}$ do

	对于每个 $c \in  \operatorname{ng}\left( n\right)$ 和每个 ${\overrightarrow{t}}_{c} \in  \Delta {P}_{c}$ 执行

		$U \mathrel{\text{:=}} U \cup  \operatorname{supp}\left( {{G}_{n,c}\left\lbrack  \overrightarrow{{t}_{c}}\right\rbrack  }\right)$

	for each $\overrightarrow{t} \in  U$ do

	对于每个 $\overrightarrow{t} \in  U$ 执行

		$\Delta {G}_{n}\left\lbrack  \overrightarrow{t}\right\rbrack   \mathrel{\text{:=}} \mathop{\prod }\limits_{{c \in  \operatorname{grd}\left( n\right) }}\left( {{P}_{c} + \Delta {P}_{c}}\right) \left\lbrack  \overrightarrow{t}\right\rbrack   - {G}_{n}\left\lbrack  \overrightarrow{t}\right\rbrack$

		$\Delta {L}_{n}\left\lbrack  \overrightarrow{t}\right\rbrack   \mathrel{\text{:=}} \mathop{\prod }\limits_{{c \in  \operatorname{ch}\left( n\right) }}\left( {{P}_{c} + \Delta {P}_{c}}\right) \left\lbrack  {\overrightarrow{t}\left\lbrack  {\operatorname{pvar}\left( c\right) }\right\rbrack  }\right\rbrack   - {L}_{n}\left\lbrack  \overrightarrow{t}\right\rbrack$

		$\Delta {P}_{n}\left\lbrack  {\overrightarrow{t}\left\lbrack  {\operatorname{pvar}\left( n\right) }\right\rbrack  }\right\rbrack   +  = \Delta {L}_{n}\left\lbrack  \overrightarrow{t}\right\rbrack$

---

<!-- Media -->

Theorem 4.12. Let $\mathcal{D}$ be a $T$ -rep for ${db}$ and let $u$ be an update to ${db}$ . ${\operatorname{DYN}}_{T}\left( {\mathcal{D},u}\right)$ produces a $T$ -rep for ${db} + u$ .

定理 4.12。设 $\mathcal{D}$ 是 ${db}$ 的 $T$ - 表示，并且设 $u$ 是对 ${db}$ 的一次更新。${\operatorname{DYN}}_{T}\left( {\mathcal{D},u}\right)$ 会生成 ${db} + u$ 的 $T$ - 表示。

### 4.4 Complexity of Dynamic Yannakakis

### 4.4 动态亚纳卡基斯算法的复杂度

Next,we study the efficiency with which ${\mathrm{{DYN}}}_{T}$ maintains $T$ -reps under updates. Towards this end,we first illustrate ${\mathrm{{DYN}}}_{T}$ ’s operation by example.

接下来，我们研究 ${\mathrm{{DYN}}}_{T}$ 在更新操作下维护 $T$ -代表（$T$ -reps）的效率。为此，我们首先通过示例来说明 ${\mathrm{{DYN}}}_{T}$ 的操作。

Example 4.13. Consider join tree ${T}_{3}$ from Figure 2 and the ${T}_{3}$ -rep $\mathcal{D}$ shown in Figure 3 that we discussed in Example 4.9. Consider the update $\{ \left( {1,5,9}\right)  \mapsto  2\}$ on $S$ (i.e., tuple(1,5,9)is inserted into $S\left( {x,y,u}\right)$ with multiplicity 2 ). When ${\mathrm{{DYN}}}_{{T}_{3}}$ executes,it will compute empty delta GMRs for all nodes,except for $S\left( {x,y,u}\right)$ and its ancestors. In particular,when Algorithm 2 is run on hyperedge $\left\lbrack  {x,y,z}\right\rbrack$ ,it operates as follows. First observe that $\left\lbrack  {x,y,z}\right\rbrack$ has only one guard child,namely $R\left( {x,y,z}\right)$ . Hence ${G}_{\left\lbrack  x,y,z\right\rbrack  } = {P}_{R\left( {x,y,z}\right) } =$ ${L}_{R\left( {x,y,z}\right) } = R$ . Since $R$ is not updated, $U$ is initialized to $\varnothing$ in line 2. Then,in lines 3 and 4,Algorithm 2 uses the index ${G}_{\left\lbrack  {x,y,z}\right\rbrack  ,S\left\lbrack  {x,y,u}\right\rbrack  }$ of ${G}_{\left\lbrack  x,y,z\right\rbrack  }$ on $\left\lbrack  {x,y}\right\rbrack$ to directly retrieve all tuples in $R\left( {x,y,z}\right)$ that satisfy $x = 1$ and $y = 5$ . This is the main purpose of the indexes ${G}_{n,c}$ : we do not have to iterate over the entire GMR ${G}_{n}$ nor check for equalities. During the rest of its computation, Algorithm 2 then calculates $\Delta {G}_{\left\lbrack  x,y,z\right\rbrack  } = \varnothing ,\Delta {L}_{\left\lbrack  x,y,z\right\rbrack  } = \{ \left( {1,5,6}\right)  \mapsto  2,\left( {1,5,7}\right)  \mapsto  2\}$ , and $\Delta {P}_{\left\lbrack  x,y,z\right\rbrack  } = \{ \left( 5\right)  \mapsto  4\}$ . When processing $\left\lbrack  y\right\rbrack$ ,we have to propagate $\Delta {P}_{\left\lbrack  x,y,z\right\rbrack  }$ from $\left\lbrack  {x,y,z}\right\rbrack$ to $\left\lbrack  y\right\rbrack$ . This is done by initializing $U = \{ \left( 5\right) \}$ in line 2 of Algorithm 2,after which $\Delta {G}_{\left\lbrack  y\right\rbrack  } = \Delta {L}_{\left\lbrack  y\right\rbrack  } = \{ \left( 5\right)  \mapsto  {24}\}$ and $\Delta {P}_{\left\lbrack  y\right\rbrack  } = \{ \left( \right)  \mapsto  {24}\} .$

示例 4.13。考虑图 2 中的连接树 ${T}_{3}$ 以及图 3 中所示的 ${T}_{3}$ -代表 $\mathcal{D}$，我们在示例 4.9 中讨论过该代表。考虑对 $S$ 进行的更新 $\{ \left( {1,5,9}\right)  \mapsto  2\}$（即，元组 (1, 5, 9) 以重数 2 插入到 $S\left( {x,y,u}\right)$ 中）。当 ${\mathrm{{DYN}}}_{{T}_{3}}$ 执行时，它将为所有节点计算空的增量广义最大表示（delta GMRs），除了 $S\left( {x,y,u}\right)$ 及其祖先节点。具体而言，当对超边 $\left\lbrack  {x,y,z}\right\rbrack$ 运行算法 2 时，其操作如下。首先观察到 $\left\lbrack  {x,y,z}\right\rbrack$ 只有一个守护子节点，即 $R\left( {x,y,z}\right)$。因此 ${G}_{\left\lbrack  x,y,z\right\rbrack  } = {P}_{R\left( {x,y,z}\right) } =$ ${L}_{R\left( {x,y,z}\right) } = R$。由于 $R$ 未更新，在第 2 行将 $U$ 初始化为 $\varnothing$。然后，在第 3 行和第 4 行，算法 2 使用 ${G}_{\left\lbrack  x,y,z\right\rbrack  }$ 在 $\left\lbrack  {x,y}\right\rbrack$ 上的索引 ${G}_{\left\lbrack  {x,y,z}\right\rbrack  ,S\left\lbrack  {x,y,u}\right\rbrack  }$ 直接检索 $R\left( {x,y,z}\right)$ 中满足 $x = 1$ 和 $y = 5$ 的所有元组。这就是索引 ${G}_{n,c}$ 的主要目的：我们不必遍历整个广义最大表示（GMR）${G}_{n}$，也不必检查相等性。在算法 2 的其余计算过程中，它会计算 $\Delta {G}_{\left\lbrack  x,y,z\right\rbrack  } = \varnothing ,\Delta {L}_{\left\lbrack  x,y,z\right\rbrack  } = \{ \left( {1,5,6}\right)  \mapsto  2,\left( {1,5,7}\right)  \mapsto  2\}$ 和 $\Delta {P}_{\left\lbrack  x,y,z\right\rbrack  } = \{ \left( 5\right)  \mapsto  4\}$。在处理 $\left\lbrack  y\right\rbrack$ 时，我们必须将 $\Delta {P}_{\left\lbrack  x,y,z\right\rbrack  }$ 从 $\left\lbrack  {x,y,z}\right\rbrack$ 传播到 $\left\lbrack  y\right\rbrack$。这是通过在算法 2 的第 2 行初始化 $U = \{ \left( 5\right) \}$ 来完成的，之后 $\Delta {G}_{\left\lbrack  y\right\rbrack  } = \Delta {L}_{\left\lbrack  y\right\rbrack  } = \{ \left( 5\right)  \mapsto  {24}\}$ 和 $\Delta {P}_{\left\lbrack  y\right\rbrack  } = \{ \left( \right)  \mapsto  {24}\} .$

It is important to observe that in this example the single-tuple update $\{ \left( {1,5,9}\right)  \mapsto  2\}$ on $S$ triggers multiple tuples to become live in $\left\lbrack  {x,y,z}\right\rbrack$ . This occurs simply because the variable $z$ of $\left\lbrack  {x,y,z}\right\rbrack$ is not in $S\left( {x,y,u}\right)$ ; therefore a single-tuple update to $S$ can match many tuples in ${G}_{\left\lbrack  x,y,z\right\rbrack  }$ with different $z$ values. In fact,in the worst case,it may cause as many tuples to become live in $\left\lbrack  {x,y,z}\right\rbrack$ as there are tuples in ${G}_{\left\lbrack  x,y,z\right\rbrack  } = R$ . In contrast,single-tuple updates into $R\left( {x,y,z}\right) ,T\left( {y,v,w}\right)$ ,or $U\left( {y,v,p}\right)$ ,can cause at most 1 tuple to become live in any of their parents. This is because $R$ ’s (resp. $T$ ’s and $U$ ’s) parent contains only variables that are also mentioned in $R$ (resp. $T$ ,resp. $U$ ). Likewise,updates to $\left\lbrack  {x,y,z}\right\rbrack$ (resp. $\left\lbrack  {y,v}\right\rbrack$ ) that we need to propagate to $\left\lbrack  y\right\rbrack$ can only cause as many $\left\lbrack  y\right\rbrack$ tuples to become live as have become live in $\left\lbrack  {x,y,z}\right\rbrack$ (resp. $\left\lbrack  {y,v}\right\rbrack$ ).

需要注意的是，在这个例子中，对$S$的单元组更新 $\{ \left( {1,5,9}\right)  \mapsto  2\}$ 会触发 $\left\lbrack  {x,y,z}\right\rbrack$ 中有多个元组变为活跃状态。这种情况的发生仅仅是因为 $\left\lbrack  {x,y,z}\right\rbrack$ 的变量 $z$ 不在 $S\left( {x,y,u}\right)$ 中；因此，对 $S$ 的单元组更新可以与 ${G}_{\left\lbrack  x,y,z\right\rbrack  }$ 中具有不同 $z$ 值的许多元组匹配。实际上，在最坏的情况下，它可能会导致 $\left\lbrack  {x,y,z}\right\rbrack$ 中变为活跃状态的元组数量与 ${G}_{\left\lbrack  x,y,z\right\rbrack  } = R$ 中的元组数量一样多。相比之下，对 $R\left( {x,y,z}\right) ,T\left( {y,v,w}\right)$ 或 $U\left( {y,v,p}\right)$ 的单元组更新，最多只能使它们的任何一个父节点中有 1 个元组变为活跃状态。这是因为 $R$（分别对应 $T$ 和 $U$）的父节点仅包含在 $R$（分别对应 $T$、$U$）中也被提及的变量。同样，我们需要传播到 $\left\lbrack  y\right\rbrack$ 的对 $\left\lbrack  {x,y,z}\right\rbrack$（分别对应 $\left\lbrack  {y,v}\right\rbrack$）的更新，只能使 $\left\lbrack  y\right\rbrack$ 中变为活跃状态的元组数量与 $\left\lbrack  {x,y,z}\right\rbrack$（分别对应 $\left\lbrack  {y,v}\right\rbrack$）中变为活跃状态的元组数量相同。

So, the fact that a node contains all variables mentioned in its parent makes it efficient to propagate updates from that node to its parent. Trees for which all nodes (except for the root) contain all variables mentioned in their parent are called simple trees.

因此，一个节点包含其父节点中提及的所有变量这一事实，使得从该节点向其父节点传播更新变得高效。所有节点（除根节点外）都包含其父节点中提及的所有变量的树被称为简单树。

Definition 4.14 (Simplicity). A width-one GHD $T$ is $\operatorname{sim}$ - ple if every child node in $T$ is a guard of its parent. A query $Q$ is simple if it has a simple join tree.

定义 4.14（简单性）。如果 $T$ 中的每个子节点都是其父节点的保护节点，则宽度为 1 的广义超树分解（GHD） $T$ 是 $\operatorname{sim}$ - 简单的。如果查询 $Q$ 有一个简单连接树，则该查询是简单的。

For example, ${T}_{4}$ of Figure 2 is simple,but ${T}_{3}$ is not since $S\left( {x,y,u}\right)$ is a child but not a guard of $\left\lbrack  {x,y,z}\right\rbrack$ .

例如，图 2 中的 ${T}_{4}$ 是简单的，但 ${T}_{3}$ 不是，因为 $S\left( {x,y,u}\right)$ 是 $\left\lbrack  {x,y,z}\right\rbrack$ 的子节点，但不是其保护节点。

Because in simple trees the number of tuples that can propagate from a child update to its parent is bounded by the size of the child update, we obtain that, for a simple tree $T,{\mathrm{{DYN}}}_{T}$ maintains a $T$ -rep under update $u$ in time linear in $u$ ,independent of the databse ${db}$ .

因为在简单树中，从子节点更新传播到其父节点的元组数量受子节点更新大小的限制，我们可以得出，对于简单树 $T,{\mathrm{{DYN}}}_{T}$，在更新 $u$ 时能以与 $u$ 成线性关系的时间维护一个 $T$ - 表示，且与数据库 ${db}$ 无关。

Theorem 4.15. ${\operatorname{DYN}}_{T}\left( {\mathcal{D},u}\right)$ produces a $T$ -rep for ${db} + u$ in time $O\left( {\parallel u\parallel }\right)$ for every database ${db}$ and every update $u$ if, and only if, $T$ is simple.

定理 4.15。当且仅当 $T$ 是简单的时，对于每个数据库 ${db}$ 和每个更新 $u$，${\operatorname{DYN}}_{T}\left( {\mathcal{D},u}\right)$ 能在时间 $O\left( {\parallel u\parallel }\right)$ 内为 ${db} + u$ 生成一个 $T$ - 表示。

The technical proof is deferred to the full version of this paper because of space constraints.

由于篇幅限制，技术证明将推迟到本文的完整版本中给出。

Theorem 4.15 indicates that, before using DYN to dynamically process $Q$ ,it is important to check for the existence of a generalized simple join tree for $Q$ .

定理 4.15 表明，在使用动态查询处理系统（DYN）对 $Q$ 进行动态处理之前，检查 $Q$ 是否存在广义简单连接树是很重要的。

On non-simple trees $T$ ,such as the tree ${T}_{3}$ from Example 4.13, ${\mathrm{{DYN}}}_{T}$ is less efficient in the worst case. Indeed,as already illustrated above, a single-tuple update can trigger multiple-tuple updates to its ancestors and in the worst case the parent update may be as big as $\parallel {db}\parallel$ . In principle,the multiple-tuple update to the parent may cause an even bigger update to the grand-parent (assuming that the latter is not a guard of the grand-parent). The number of tuples in an update to a node can be shown to be always bounded by $\parallel {db}\parallel  + \parallel u\parallel$ ,however. Using this observation,we can show:

在非简单树$T$上，例如示例4.13中的树${T}_{3}$，${\mathrm{{DYN}}}_{T}$在最坏情况下效率较低。实际上，正如上面已经说明的，单元组更新可能会触发对其祖先的多单元组更新，并且在最坏情况下，父节点的更新可能会达到$\parallel {db}\parallel$那么大。原则上，对父节点的多单元组更新可能会导致对祖父节点进行更大的更新（假设祖父节点不是其父节点的保护节点）。然而，可以证明，对一个节点的更新中的元组数量总是受$\parallel {db}\parallel  + \parallel u\parallel$限制。利用这一观察结果，我们可以证明：

Proposition 4.16. Let $T$ be a join tree, $\mathcal{D}$ a $T$ -rep for ${db}$ and $u$ an update to ${db}.{\mathrm{{DYN}}}_{T}\left( {\mathcal{D},u}\right)$ produces a $T$ -rep for ${db} + u$ in time $O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$ .

命题4.16。设$T$为连接树，$\mathcal{D}$是${db}$的$T$ - 表示，并且$u$对${db}.{\mathrm{{DYN}}}_{T}\left( {\mathcal{D},u}\right)$的更新在时间$O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$内产生${db} + u$的$T$ - 表示。

In other words,in the worst case, ${\mathrm{{DYN}}}_{T}$ runs in time $O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$ ,which is unfortunately similar to recomputing everything from scratch using CDY after every update. Fortunately, while recomputing everything from scratch will always cost $\Omega \left( {\parallel {db} + u\parallel }\right)$ time,in practice DYN performs much better than its $O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$ upper bound. This is discussed at the end of Section 5.

换句话说，在最坏情况下，${\mathrm{{DYN}}}_{T}$的运行时间为$O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$，不幸的是，这类似于每次更新后使用CDY从头重新计算所有内容。幸运的是，虽然从头重新计算所有内容总是需要$\Omega \left( {\parallel {db} + u\parallel }\right)$的时间，但实际上DYN的性能比其$O\left( {\parallel {db}\parallel  + \parallel u\parallel }\right)$的上界要好得多。这将在第5节末尾进行讨论。

### 4.5 Enumeration

### 4.5 枚举

In this section,we show that a $T$ -rep $\mathcal{D}$ for ${db}$ ,with $T$ a join tree for join query $Q$ can be used not only to enumerate $Q\left( {db}\right)$ with constant delay,but also some of its projections ${\pi }_{\bar{x}}Q\left( {db}\right)$ . In particular,CDE of projections is possible if there exists a subtree of $T$ that includes the root and contains precisely the set $\bar{x}$ of projected variables. Intuitively,if such subtree exists, the enumeration algorithm will be able to traverse $\mathcal{D}$ to find the required values of $\bar{x}$ without traversing tuples containing variables in $\operatorname{var}\left( Q\right)  \smallsetminus  \bar{x}$ ,which may cause unbounded delays in the enumeration. Essentially the same idea has been used before in the context of static CDE [5] (as discussed in Section 4.6), factorized databases [6], and worst-case optimal algorithms [23]. We proceed formally.

在本节中，我们表明，对于连接查询$Q$的连接树$T$，${db}$的$T$ - 表示$\mathcal{D}$不仅可用于以恒定延迟枚举$Q\left( {db}\right)$，还可用于枚举其某些投影${\pi }_{\bar{x}}Q\left( {db}\right)$。特别地，如果存在$T$的一个包含根节点且恰好包含投影变量集合$\bar{x}$的子树，那么投影的CDE是可行的。直观地说，如果存在这样的子树，枚举算法将能够遍历$\mathcal{D}$以找到$\bar{x}$的所需值，而无需遍历包含$\operatorname{var}\left( Q\right)  \smallsetminus  \bar{x}$中变量的元组，因为这些元组可能会导致枚举过程中出现无界延迟。本质上，相同的思想之前已经在静态CDE [5]（如第4.6节所讨论）、因式分解数据库 [6]和最坏情况最优算法 [23]的背景下被使用过。我们进行正式推导。

<!-- Media -->

Algorithm 3 ENUM ${}_{\left( T,N\right) }\left( \mathcal{D}\right)$

算法3 枚举${}_{\left( T,N\right) }\left( \mathcal{D}\right)$

---

for each $\left( {{\overrightarrow{t}}_{{n}_{1}},{\mu }_{{n}_{1}}}\right)  \in  {L}_{{n}_{1}}\left( \left\lbrack  \right\rbrack  \right)$ do

对于每个$\left( {{\overrightarrow{t}}_{{n}_{1}},{\mu }_{{n}_{1}}}\right)  \in  {L}_{{n}_{1}}\left( \left\lbrack  \right\rbrack  \right)$执行

	for each $\left( {{\overrightarrow{t}}_{{n}_{2}},{\mu }_{{n}_{2}}}\right)  \in  {L}_{{n}_{2}}\left( {{\overrightarrow{t}}_{p\left( {n}_{2}\right) }\left\lbrack  \overline{{x}_{{n}_{2}}}\right\rbrack  }\right)$ do

	对于每个$\left( {{\overrightarrow{t}}_{{n}_{2}},{\mu }_{{n}_{2}}}\right)  \in  {L}_{{n}_{2}}\left( {{\overrightarrow{t}}_{p\left( {n}_{2}\right) }\left\lbrack  \overline{{x}_{{n}_{2}}}\right\rbrack  }\right)$执行

		for each $\left( {{\overrightarrow{t}}_{{n}_{3}},{\mu }_{{n}_{3}}}\right)  \in  {L}_{{n}_{3}}\left( {{\overrightarrow{t}}_{p\left( {n}_{3}\right) }\left\lbrack  \overline{{x}_{{n}_{3}}}\right\rbrack  }\right)$ do

		对于每个$\left( {{\overrightarrow{t}}_{{n}_{3}},{\mu }_{{n}_{3}}}\right)  \in  {L}_{{n}_{3}}\left( {{\overrightarrow{t}}_{p\left( {n}_{3}\right) }\left\lbrack  \overline{{x}_{{n}_{3}}}\right\rbrack  }\right)$执行

			...

			for each $\left( {{\overrightarrow{t}}_{{n}_{k}},{\mu }_{{n}_{k}}}\right)  \in  {L}_{{n}_{k}}\left( {{\overrightarrow{t}}_{p\left( {n}_{k}\right) }\left\lbrack  \overline{{x}_{{n}_{k}}}\right\rbrack  }\right)$ do

			对每个 $\left( {{\overrightarrow{t}}_{{n}_{k}},{\mu }_{{n}_{k}}}\right)  \in  {L}_{{n}_{k}}\left( {{\overrightarrow{t}}_{p\left( {n}_{k}\right) }\left\lbrack  \overline{{x}_{{n}_{k}}}\right\rbrack  }\right)$ 执行

				let $\mu  = {\mu }_{{c}_{1}} * \cdots  * {\mu }_{{c}_{l}} * {\Pi }_{m \in  M}{P}_{m}\left( {{\overrightarrow{t}}_{p\left( m\right) }\left\lbrack  \overline{{x}_{m}}\right\rbrack  }\right)$

				令 $\mu  = {\mu }_{{c}_{1}} * \cdots  * {\mu }_{{c}_{l}} * {\Pi }_{m \in  M}{P}_{m}\left( {{\overrightarrow{t}}_{p\left( m\right) }\left\lbrack  \overline{{x}_{m}}\right\rbrack  }\right)$

					output $\left( {{\overrightarrow{t}}_{{c}_{1}}{\overrightarrow{t}}_{{c}_{2}}\ldots {\overrightarrow{t}}_{{c}_{l}},\mu }\right)$

					输出 $\left( {{\overrightarrow{t}}_{{c}_{1}}{\overrightarrow{t}}_{{c}_{2}}\ldots {\overrightarrow{t}}_{{c}_{l}},\mu }\right)$

---

<!-- Media -->

Definition 4.17. Let $T = \left( {V,E}\right)$ be a join tree. A subset $N \subseteq  V$ is connex if it includes the root and the subgraph of $T$ induced by $N$ is a tree.

定义 4.17。设 $T = \left( {V,E}\right)$ 为一个连接树（join tree）。若子集 $N \subseteq  V$ 包含根节点，且由 $N$ 所诱导的 $T$ 的子图是一棵树，则称 $N \subseteq  V$ 是连通的（connex）。

To illustrate, $\{ \left\lbrack  y\right\rbrack  ,\left\lbrack  {x,y,z}\right\rbrack  ,\left\lbrack  {y,v}\right\rbrack  \}$ is a connex subset of the join tree ${T}_{3}$ of Figure 2,but $\{ \left\lbrack  y\right\rbrack  ,S\left( {x,y,u}\right) \}$ is not.

举例来说，$\{ \left\lbrack  y\right\rbrack  ,\left\lbrack  {x,y,z}\right\rbrack  ,\left\lbrack  {y,v}\right\rbrack  \}$ 是图 2 中连接树 ${T}_{3}$ 的一个连通子集，但 $\{ \left\lbrack  y\right\rbrack  ,S\left( {x,y,u}\right) \}$ 不是。

For each join tree $T$ and each connex subset $N$ of its nodes we define enumeration algorithm ${\operatorname{ENUM}}_{\left( T,N\right) }$ as follows. Let ${T}^{\prime }$ be the subtree of $T$ induced by $N$ and let $\left( {{n}_{k},{n}_{k - 1},\ldots ,{n}_{1}}\right)$ be a topological sort of ${T}^{\prime }$ . In particular, ${n}_{1}$ is the root of $T$ . Let ${c}_{1},\ldots ,{c}_{l}$ be the leaf nodes of ${T}^{\prime }$ . Let $p\left( n\right)$ denote the parent of node $n$ and let $\overline{{x}_{n}}$ denote pvar(n). Finally,let $M$ be the subset of all nodes in $T$ that are not in $N$ ,but for which some sibling is in $N$ . With this notation, ${\operatorname{ENUM}}_{\left( T,N\right) }$ is shown in Algorithm 3. Example 4.3 shows ${\operatorname{ENUM}}_{\left( {T}_{3},N\right) }\left( \mathcal{D}\right)$ for the join tree ${T}_{3}$ of Figure 2 with $N$ consisting of all nodes in ${T}_{3}$ .

对于每个连接树 $T$ 及其节点的每个连通子集 $N$，我们按如下方式定义枚举算法 ${\operatorname{ENUM}}_{\left( T,N\right) }$。设 ${T}^{\prime }$ 是由 $N$ 所诱导的 $T$ 的子树，设 $\left( {{n}_{k},{n}_{k - 1},\ldots ,{n}_{1}}\right)$ 是 ${T}^{\prime }$ 的一个拓扑排序。特别地，${n}_{1}$ 是 $T$ 的根节点。设 ${c}_{1},\ldots ,{c}_{l}$ 是 ${T}^{\prime }$ 的叶节点。设 $p\left( n\right)$ 表示节点 $n$ 的父节点，设 $\overline{{x}_{n}}$ 表示 pvar(n)。最后，设 $M$ 是 $T$ 中所有不在 $N$ 里，但有某个兄弟节点在 $N$ 中的节点组成的子集。使用这些符号，${\operatorname{ENUM}}_{\left( T,N\right) }$ 如算法 3 所示。示例 4.3 展示了图 2 中连接树 ${T}_{3}$ 在 $N$ 由 ${T}_{3}$ 中所有节点组成时的 ${\operatorname{ENUM}}_{\left( {T}_{3},N\right) }\left( \mathcal{D}\right)$。

Proposition 4.18. Let $T$ be a join tree for join query $Q$ . Assume that $N$ is a connex subset of $T$ . Then ${\operatorname{ENUM}}_{\left( T,N\right) }\left( \mathcal{D}\right)$ enumerates ${Q}^{\prime }\left( {db}\right)  \mathrel{\text{:=}} {\pi }_{\operatorname{var}\left( N\right) }Q\left( {db}\right)$ with constant delay,for every database ${db}$ and every $T$ -rep $\mathcal{D}$ of ${db}$ . Moreover,for an arbitrary tuple $\overrightarrow{t}$ ,its multiplicity in ${Q}^{\prime }\left( {db}\right)$ can be calculated from $\mathcal{D}$ in constant time.

命题 4.18。设 $T$ 是连接查询 $Q$ 的一个连接树。假设 $N$ 是 $T$ 的一个连通子集。那么对于每个数据库 ${db}$ 以及 ${db}$ 的每个 $T$ -表示 $\mathcal{D}$，${\operatorname{ENUM}}_{\left( T,N\right) }\left( \mathcal{D}\right)$ 以恒定延迟枚举 ${Q}^{\prime }\left( {db}\right)  \mathrel{\text{:=}} {\pi }_{\operatorname{var}\left( N\right) }Q\left( {db}\right)$。此外，对于任意元组 $\overrightarrow{t}$，它在 ${Q}^{\prime }\left( {db}\right)$ 中的重数可以在恒定时间内从 $\mathcal{D}$ 计算得出。

The technical proof is deferred until the full version of this paper. Note that ${\operatorname{ENUM}}_{T,V}\left( \mathcal{D}\right)$ with $V$ the set of all nodes in $T$ enumerates $Q\left( {db}\right)$ . Combining all of our results so far we obtain that join-tree representations are DCLRs for the class of all free-connex acyclic ${CQs}$ ,which is defined as follows.

技术证明将推迟到本文的完整版本中给出。注意，当$V$为$T$中所有节点的集合时，${\operatorname{ENUM}}_{T,V}\left( \mathcal{D}\right)$可枚举$Q\left( {db}\right)$。综合我们目前得到的所有结果，我们得出：连接树表示是所有自由连通无环${CQs}$类的可延迟常数时间枚举表示（DCLR），其定义如下。

Definition 4.19. (Compatible,Free-Connex Acyclic) Let $T$ be a join tree. A CQ $Q$ is compatible with $T$ if $T$ is a join tree for $Q$ and $T$ has a connex subset $N$ with $\operatorname{var}\left( N\right)  =$ out(Q). A CQ is free-connex acyclic if it has a compatible join tree.

定义4.19.（兼容的、自由连通无环）设$T$为一棵连接树。如果$T$是$Q$的连接树，并且$T$有一个连通子集$N$，且$\operatorname{var}\left( N\right)  =$包含于$Q$的输出（out(Q)），则称一个合取查询（CQ）$Q$与$T$兼容。如果一个合取查询有一棵兼容的连接树，则称其为自由连通无环的。

In particular, every acyclic join query is free-connex acyclic. Let $T$ be a join tree. It now follows that the class of all $T$ - reps form a DCLR of every $\mathrm{{CQ}}Q$ compatible with $T$ .

特别地，每个无环连接查询都是自由连通无环的。设$T$为一棵连接树。由此可知，所有$T$ - 表示的类构成了与$T$兼容的每个$\mathrm{{CQ}}Q$的可延迟常数时间枚举表示（DCLR）。

Delta-enumeration. Using ${\mathrm{{DYN}}}_{T}$ we can actually also enumerate deltas ${\Delta Q}\left( {{db},u}\right)$ of $Q\left( {db}\right)$ under single-tuple update $u$ . This result is relevant for push-based query processing systems, where users do not ping the system for the complete current query answer, but instead ask to be notified of the changes to the query results when the database changes. In addition, as we will discuss in Section 5, it also provides a key method for dynamic processing of CAQs.

增量枚举。利用${\mathrm{{DYN}}}_{T}$，我们实际上还可以枚举在单元组更新$u$下$Q\left( {db}\right)$的增量${\Delta Q}\left( {{db},u}\right)$。这一结果与基于推送的查询处理系统相关，在该系统中，用户不会主动向系统请求完整的当前查询答案，而是要求在数据库发生变化时收到查询结果变更的通知。此外，正如我们将在第5节中讨论的，它还为合取聚合查询（CAQ）的动态处理提供了一种关键方法。

Definition 4.20 ( ${\Delta T}$ -rep). Let $T$ be a join tree, ${db}$ be a database and let $u$ be an update to ${db}$ . A ${\Delta T}$ -representation of(db,u)is a data structure $\Delta \mathcal{D}$ that contains (1) a $T$ -rep for ${db}$ ; (2) an index $\Delta {L}_{n}$ of $\Delta {\Lambda }_{n}\left( {{db},u}\right)$ on $\operatorname{pvar}\left( n\right)$ ,for every $n \in  T$ ; and (3) a GMR $\Delta {P}_{n}$ that materializes $\Delta {\Psi }_{n}\left( {{db},u}\right)$ , for every $n \in  T$ .

定义4.20（${\Delta T}$ - 表示）。设$T$为一棵连接树，${db}$为一个数据库，$u$为对${db}$的一次更新。(db,u)的${\Delta T}$ - 表示是一个数据结构$\Delta \mathcal{D}$，它包含：（1）${db}$的$T$ - 表示；（2）对于每个$n \in  T$，$\Delta {\Lambda }_{n}\left( {{db},u}\right)$在$\operatorname{pvar}\left( n\right)$上的一个索引$\Delta {L}_{n}$；（3）对于每个$n \in  T$，一个物化$\Delta {\Psi }_{n}\left( {{db},u}\right)$的广义匹配表示（GMR）$\Delta {P}_{n}$。

Note that ${\mathrm{{DYN}}}_{T}$ needs to compute $\Delta {L}_{n} = \Delta {\Lambda }_{n}\left( {{db},u}\right)$ and $\Delta {P}_{n} = \Delta {\Psi }_{n}\left( {{db},u}\right)$ anyway when processing $T$ -rep $\mathcal{D}$ under update $u$ . Hence,after the bottom-up pass in lines $4 - 5$ of ${\mathrm{{DYN}}}_{T}$ we obtain a ${\Delta T}$ -rep $\Delta \mathcal{D}$ ,provided that we represent $\Delta {L}_{n}$ as an index on $\operatorname{pvar}\left( n\right)$ .

注意，在更新$u$下处理$T$ - 表示$\mathcal{D}$时，${\mathrm{{DYN}}}_{T}$无论如何都需要计算$\Delta {L}_{n} = \Delta {\Lambda }_{n}\left( {{db},u}\right)$和$\Delta {P}_{n} = \Delta {\Psi }_{n}\left( {{db},u}\right)$。因此，在${\mathrm{{DYN}}}_{T}$的第$4 - 5$行进行自底向上的遍历之后，只要我们将$\Delta {L}_{n}$表示为$\operatorname{pvar}\left( n\right)$上的一个索引，就可以得到一个${\Delta T}$ - 表示$\Delta \mathcal{D}$。

Theorem 4.21. Let $u$ be a single-tuple update to database db. Let $\Delta \mathcal{D}$ be a ${\Delta T}$ -rep of(db,u)and let $Q$ be compatible with $T$ . Then ${\Delta Q}\left( {{db},u}\right)$ can be enumerated with constant delay from $\Delta \mathcal{D}$ .

定理4.21。设$u$为对数据库db的一次单元组更新。设$\Delta \mathcal{D}$为(db,u)的${\Delta T}$ - 表示，且$Q$与$T$兼容。那么可以从$\Delta \mathcal{D}$以常数延迟枚举${\Delta Q}\left( {{db},u}\right)$。

Due to space constraints, we defer the definition of the delta-enumeration algorithm to the full version of this paper. How to enumerate ${\Delta Q}\left( {{db},u}\right)$ with constant delay for general, multiple-tuple updates remains an open problem.

由于篇幅限制，我们将增量枚举算法的定义推迟到本文的完整版本中给出。对于一般的多单元组更新，如何以常数延迟枚举${\Delta Q}\left( {{db},u}\right)$仍是一个开放问题。

### 4.6 Optimalitiy

### 4.6 最优性

In this section we show that Dynamic Yannakakis is optimal in two aspects. (1) It is able to dynamically process the largest subclass of CQs for which DCLRs can reasonably be expected to exist. (2) The class of queries for which Dyn processes updates in $O\left( {\parallel u\parallel }\right)$ time (Theorem 4.15) is the largest class of queries for which we can reasonably expect to have such update processing time as well as CDE of results.

在本节中，我们将证明动态亚纳卡基斯算法（Dynamic Yannakakis）在两个方面具有最优性。（1）它能够动态处理最大的合取查询（CQs）子类，对于该子类，我们有理由期望存在动态连接线性表示（DCLRs）。（2）动态算法（Dyn）能在$O\left( {\parallel u\parallel }\right)$时间内处理更新的查询类（定理4.15），是我们有理由期望既能有这样的更新处理时间，又能有结果的简洁数据枚举（CDE）的最大查询类。

DCLR-optimality. In the static setting without updates, a query $Q$ is said to be in class $\mathrm{{CD}} \circ  \mathrm{{LIN}}$ if there exists an algorithm that,for each database ${db}$ does an $O\left( {\parallel {db}\parallel }\right)$ - time precomputation and then proceeds in CDE of $Q\left( {db}\right)$ , evaluated under set semantics. Bagan et al. showed that, under the so-called binary matrix multiplication conjecture, an acyclic $\mathrm{{CQ}}$ is in $\mathrm{{CD}} \circ  \mathrm{{LIN}}$ if and only if it is free-connex [5]. Recently, Brault-Baron extended this result under two assumptions: the triangle hypothesis (checking the presence of a triangle in a hypergraph with $n$ nodes cannot be done in time $O\left( {n}^{2}\right)$ ) and the tetrahedron hypothesis (for each $k > 2$ , checking whether a hypergraph contains a $k$ -simplex cannot be done in time $O\left( n\right)$ ). Under these assumptions,he shows that a CQ is in $\mathrm{{CD}} \circ  \mathrm{{LIN}}$ if and only if it is free-connex acyclic [10]. In Appendix C we prove that this implies:

动态连接线性表示最优性（DCLR - optimality）。在无更新的静态设置中，如果存在一种算法，对于每个数据库${db}$进行$O\left( {\parallel {db}\parallel }\right)$时间的预计算，然后以集合语义对$Q\left( {db}\right)$进行简洁数据枚举（CDE），则称查询$Q$属于类$\mathrm{{CD}} \circ  \mathrm{{LIN}}$。巴甘（Bagan）等人表明，在所谓的二进制矩阵乘法猜想下，一个无环的$\mathrm{{CQ}}$属于$\mathrm{{CD}} \circ  \mathrm{{LIN}}$当且仅当它是无连接环的（free - connex）[5]。最近，布劳尔特 - 巴伦（Brault - Baron）在两个假设下扩展了这一结果：三角形假设（在具有$n$个节点的超图中检查是否存在三角形无法在$O\left( {n}^{2}\right)$时间内完成）和四面体假设（对于每个$k > 2$，检查超图是否包含$k$ - 单纯形无法在$O\left( n\right)$时间内完成）。在这些假设下，他证明了一个合取查询（CQ）属于$\mathrm{{CD}} \circ  \mathrm{{LIN}}$当且仅当它是无连接环的无环查询[10]。在附录C中，我们证明了这意味着：

Proposition 4.22. Under the above-mentioned hypotheses, a DCLR exists for ${CQQ}$ if,and only if, $Q$ is free-connex acyclic.

命题4.22。在上述假设下，对于${CQQ}$存在动态连接线性表示（DCLR）当且仅当$Q$是无连接环的无环查询。

Processing-time optimality. Berkholz et al. have recently characterized the class of self-join free CQs that feature a representation that allows both CDE of results and $O\left( 1\right)$ maintenance under single-tuple updates [7]. In particular, they show that, under the assumption of hardness of the Online Matrix-Vector Multiplication problem [22], the following dichotomy holds. When evaluated under set semantics, a CQ $Q$ without self joins features a representation that supports CDE and maintenance in $O\left( 1\right)$ time under single-tuple updates if,and only if, $Q$ is $q$ -hierarchical. Notice $Q$ can be maintained in $O\left( 1\right)$ time under single-tuple updates if,and only if,it can also be maintained in $O\left( {\parallel u\parallel }\right)$ time under arbitrary updates. The definition of $q$ -hierarchical queries is the following.

处理时间最优性。伯克霍尔茨（Berkholz）等人最近刻画了无自连接的合取查询（CQs）类，这类查询具有一种表示形式，允许在单元组更新下同时进行结果的简洁数据枚举（CDE）和$O\left( 1\right)$维护[7]。特别地，他们表明，在在线矩阵 - 向量乘法问题具有难解性的假设下[22]，存在以下二分法。在集合语义下进行评估时，一个无自连接的合取查询$Q$具有一种表示形式，支持在单元组更新下以$O\left( 1\right)$时间进行简洁数据枚举（CDE）和维护，当且仅当$Q$是$q$ - 分层的。注意，$Q$能在单元组更新下以$O\left( 1\right)$时间维护，当且仅当它也能在任意更新下以$O\left( {\parallel u\parallel }\right)$时间维护。$q$ - 分层查询的定义如下。

Definition 4.23 ( $q$ -hierarchicality). Given a CQ $Q$ and a variable $x \in  \operatorname{var}\left( Q\right)$ ,let ${at}\left( x\right)$ denote the set of all atoms in which $x$ occurs in $Q.Q$ is called hierarchical if for every pair of variables $x,y \in  \operatorname{var}\left( Q\right)$ ,either ${at}\left( x\right)  \subseteq  {at}\left( y\right)$ or ${at}\left( y\right)  \subseteq$ ${at}\left( x\right)$ or ${at}\left( x\right)  \cap  {at}\left( y\right)  = \varnothing$ . A CQ $Q$ is q-hierarchical if it is hierarchical and for every two variables $x,y \in  \operatorname{var}\left( Q\right)$ ,if $x \in  \operatorname{out}\left( Q\right)$ and $\operatorname{at}\left( x\right)  \varsubsetneq  \operatorname{at}\left( y\right)$ ,then $y \in  \operatorname{out}\left( Q\right)$ .

定义4.23（$q$ - 分层性）。给定一个合取查询$Q$和一个变量$x \in  \operatorname{var}\left( Q\right)$，令${at}\left( x\right)$表示$x$在$Q.Q$中出现的所有原子的集合。如果对于每对变量$x,y \in  \operatorname{var}\left( Q\right)$，要么${at}\left( x\right)  \subseteq  {at}\left( y\right)$，要么${at}\left( y\right)  \subseteq$ ${at}\left( x\right)$，要么${at}\left( x\right)  \cap  {at}\left( y\right)  = \varnothing$，则称$Q.Q$是分层的。如果一个合取查询$Q$是分层的，并且对于任意两个变量$x,y \in  \operatorname{var}\left( Q\right)$，如果$x \in  \operatorname{out}\left( Q\right)$且$\operatorname{at}\left( x\right)  \varsubsetneq  \operatorname{at}\left( y\right)$，那么$y \in  \operatorname{out}\left( Q\right)$，则称该合取查询$Q$是q - 分层的。

Example 4.24. Consider the join query $Q = R\left( {x,y,z}\right)  \boxtimes$ $S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right) .Q$ is hierarchical. Moreover, ${\pi }_{x,y}Q$ is $q$ -hierarchical. In contrast, ${\pi }_{u}Q$ is not $q$ - hierarchical since $u \in  \operatorname{out}\left( Q\right) ,{at}\left( u\right)  \varsubsetneq  {at}\left( y\right)$ ,yet $y \notin  \operatorname{out}\left( Q\right)$ .

示例4.24。考虑连接查询$Q = R\left( {x,y,z}\right)  \boxtimes$ $S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right) .Q$是分层的。此外，${\pi }_{x,y}Q$是$q$ - 分层的。相比之下，${\pi }_{u}Q$不是$q$ - 分层的，因为$u \in  \operatorname{out}\left( Q\right) ,{at}\left( u\right)  \varsubsetneq  {at}\left( y\right)$，但$y \notin  \operatorname{out}\left( Q\right)$。

Observe that a join query is q-hierarchical iff it is hierarchical. The hierarchical property has actually already played a central role for efficient query evaluation in various contexts $\left\lbrack  {{16},{17},{27}}\right\rbrack$ ,see [7] for a discussion.

观察可知，一个连接查询是q - 分层的当且仅当它是分层的。实际上，分层属性在各种环境下的高效查询评估中已经起到了核心作用$\left\lbrack  {{16},{17},{27}}\right\rbrack$，相关讨论见文献[7]。

The following two propositions (proven in Appendix C) establish the relationship between DYN and the dichotomy of Berkholz et al.

以下两个命题（在附录C中证明）确立了DYN与伯克霍尔茨（Berkholz）等人的二分法之间的关系。

Proposition 4.25. If a ${CQQ}$ is q-hierarchical,then it has a join tree which is both simple and compatible.

命题4.25。如果一个${CQQ}$是q - 分层的，那么它有一棵既简单又兼容的连接树。

It then follows from Theorem 4.15 and Proposition 4.18 that,for $q$ -hierarchical queries,DYN also processes single-tuple updates in $O\left( 1\right)$ time while allowing the query result to be enumerated with constant delay (given the join tree). This hence matches the algorithm provided by Berkholz et al. for processing $q$ -hierarchical queries under updates. Note that,by Proposition 4.25,all $q$ -hierarchical queries must be free-connex acyclic.

由定理4.15和命题4.18可知，对于$q$ - 分层查询，DYN也能在$O\left( 1\right)$时间内处理单元组更新，同时允许以恒定延迟枚举查询结果（给定连接树）。因此，这与伯克霍尔茨等人提供的处理更新情况下的$q$ - 分层查询的算法相匹配。注意，根据命题4.25，所有$q$ - 分层查询必须是自由连通无环的。

Proposition 4.26. If a ${CQQ}$ has a join tree $T$ which is both simple and compatible with $Q$ ,then $Q$ is $q$ -hierarchical.

命题4.26。如果一个${CQQ}$有一棵连接树$T$，它既简单又与$Q$兼容，那么$Q$是$q$ - 分层的。

This result is to be expected, since from Theorem 4.15 and Proposition 4.18 we know that for such $T$ we can do CDE of $Q$ and do maintenance under updates in $O\left( {\parallel u\parallel }\right)$ time. If other than $q$ -hierarhical queries had simple compatible join trees, Berkholz et al.'s dichotomy would fail. Also observe that, as seen in Section 4.4, DYN may process updates in $\omega \left( {\parallel u\parallel }\right)$ time on non-simple join trees. Berkholz et al.’s dichotomy implies that this is unavoidable in the worst case.

这个结果是可以预料到的，因为从定理4.15和命题4.18我们知道，对于这样的$T$，我们可以对$Q$进行CDE（上下文依赖编码），并在$O\left( {\parallel u\parallel }\right)$时间内进行更新维护。如果除了$q$ - 分层查询之外的其他查询有简单兼容的连接树，那么伯克霍尔茨等人的二分法就会失效。另请注意，如第4.4节所示，DYN可能在非简单连接树上以$\omega \left( {\parallel u\parallel }\right)$时间处理更新。伯克霍尔茨等人的二分法意味着在最坏情况下这是不可避免的。

At this point, we can explain why it is important to work with join trees based on width-one GHDs rather than classical join trees (which do not allow partial hyperedges to occur). Indeed, the following proposition (proven in Appendix $\mathrm{C}$ ) shows that there are hierarchical queries for which no classical join tree is simple. Therefore, if we restrict ourselves to classical join trees we will fail to obtain an $O\left( {\parallel u\parallel }\right)$ update time for some $q$ -hierarchical queries.

此时，我们可以解释为什么基于宽度为1的广义超树分解（GHD）的连接树比经典连接树（不允许出现部分超边）更重要。实际上，以下命题（在附录$\mathrm{C}$中证明）表明，存在一些分层查询，对于这些查询，没有经典连接树是简单的。因此，如果我们局限于经典连接树，对于某些$q$ - 分层查询，我们将无法获得$O\left( {\parallel u\parallel }\right)$的更新时间。

Proposition 4.27. Let $Q$ be the hierarchical join query $R\left( {x,y,z}\right)  \boxtimes  S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right)$ . Every simple width-one GHD for $Q$ has at least one partial hyperedge.

命题4.27。设$Q$为分层连接查询$R\left( {x,y,z}\right)  \boxtimes  S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right)$。$Q$的每个简单宽度为1的广义超树分解（GHD）至少有一个部分超边。

## 5. IMPLEMENTATION AND EXPERIMEN- TAL VALIDATION

## 5. 实现与实验验证

In this section, we experimentally measure the performance of DYN, focusing on both throughput and memory consumption. We start by describing how our implementation addresses some practical issues, then we describe in detail the operational setup, and finally present the experimental results.

在本节中，我们通过实验来衡量DYN的性能，重点关注吞吐量和内存消耗。我们首先描述我们的实现如何解决一些实际问题，然后详细描述操作设置，最后展示实验结果。

### 5.1 Practical Implementation

### 5.1 实际实现

We have described how DYN processes free-connex acyclic CQs under updates. In this subsection, we first explain how to use DYN as an algorithmic core for practical dynamic query evaluation of the more general class of acyclic conjunctive aggregate queries (not necessarily free-connex).

我们已经描述了DYN如何在更新情况下处理自由连通无环合取查询（CQs）。在本小节中，我们首先解释如何将DYN用作更一般类别的无环合取聚合查询（不一定是自由连通的）的实际动态查询评估的算法核心。

Definition 5.1. A conjunctive aggregate query (CAQ) is a query of the form ${Q}^{\prime } = \left( {\bar{x},{f}_{1},\ldots ,{f}_{n}}\right) Q$ ,where $Q$ is a $\mathrm{{CQ}}$ ; $\bar{x} \subseteq  \operatorname{out}\left( Q\right)$ and ${f}_{i}$ is an aggregate function over $\operatorname{out}\left( Q\right)$ for $1 \leq  i \leq  n.{Q}^{\prime }$ is acyclic if its $\mathrm{{CQ}}Q$ is so.

定义5.1。合取聚合查询（CAQ）是形式为${Q}^{\prime } = \left( {\bar{x},{f}_{1},\ldots ,{f}_{n}}\right) Q$的查询，其中$Q$是一个$\mathrm{{CQ}}$；$\bar{x} \subseteq  \operatorname{out}\left( Q\right)$且${f}_{i}$是一个关于$\operatorname{out}\left( Q\right)$的聚合函数，对于$1 \leq  i \leq  n.{Q}^{\prime }$，如果其$\mathrm{{CQ}}Q$是无环的，则该查询是无环的。

Example aggregate functions are $\operatorname{SUM}\left( u\right)  \times  3$ or $\operatorname{AVG}\left( x\right)$ , assuming $u,x$ in $\operatorname{out}\left( Q\right)$ . The semantics of CAQs is best illustrated by example. Consider ${Q}^{\prime } = \left( {x,y,\operatorname{AVG}\left( v\right) }\right) {\pi }_{x,y,v}$ $\left( {R\left( {x,y,z}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$ . This query groups the result of ${\pi }_{x,y,v}\left( {R\left( {x,y,z}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$ by $x,y$ ,and computes $\operatorname{AVG}\left( v\right)$ (under multiset semantics) for each group. It should be noted that we assume that the aggregate functions need to be streamable. This means that one should be able to update the aggregate function results by only inspecting the updates to $Q\left( {db}\right)$ and the previous aggregate value plus,possibly,a constant amount of extra information per tuple

假设$\operatorname{out}\left( Q\right)$中的$u,x$，示例聚合函数有$\operatorname{SUM}\left( u\right)  \times  3$或$\operatorname{AVG}\left( x\right)$。合取聚合查询（CAQs）的语义最好通过示例来说明。考虑${Q}^{\prime } = \left( {x,y,\operatorname{AVG}\left( v\right) }\right) {\pi }_{x,y,v}$ $\left( {R\left( {x,y,z}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$。该查询按$x,y$对${\pi }_{x,y,v}\left( {R\left( {x,y,z}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$的结果进行分组，并为每个组计算$\operatorname{AVG}\left( v\right)$（在多重集语义下）。需要注意的是，我们假设聚合函数需要是可流式处理的。这意味着应该能够仅通过检查对$Q\left( {db}\right)$的更新以及先前的聚合值，再加上可能每个元组的固定数量的额外信息，来更新聚合函数的结果。

We can dynamically process an acyclic CAQ ${Q}^{\prime }$ using DYN by means of a simple strategy: use DYN to maintain a DCLR for the acyclic $\mathrm{{CQ}}Q$ of ${Q}^{\prime }$ ,but materialize the output of the CAQ in an array. Use delta-enumeration on $Q$ to maintain this array under single-tuple updates. Note that, in order to support delta-enumeration with constant delay, we require that $Q$ is free-connex (Theorem 4.21). If this is not the case, (which frequently occurs in practice), we let Dyn maintain a DCLR for a free-connex acyclic approximation ${Q}_{F}$ of $Q$ . ${Q}_{F}$ can always be obtained from $Q$ by extending the set of output variables of $Q$ (in the worst case by adding all variables to the output). Of course, under this strategy, we require $\Omega \left( \begin{Vmatrix}{{Q}^{\prime }\left( {db}\right) }\end{Vmatrix}\right)$ space,just like $\left( \mathrm{H}\right) \mathrm{{IVM}}$ ,but we avoid the (partial) materialization of $Q$ and its deltas. As shown in Section 5.3, this property actually make DYN outperform HIVM in both processing time and space.

我们可以通过一种简单的策略，使用DYN动态处理无环约束聚合查询（CAQ）${Q}^{\prime }$：利用DYN为${Q}^{\prime }$的无环$\mathrm{{CQ}}Q$维护一个动态连接关系表示（DCLR），但将CAQ的输出存储在一个数组中。对$Q$使用增量枚举法，以在单元组更新的情况下维护这个数组。请注意，为了支持具有恒定延迟的增量枚举，我们要求$Q$是自由连通的（定理4.21）。如果情况并非如此（在实际应用中经常出现这种情况），我们让Dyn为$Q$的自由连通无环近似${Q}_{F}$维护一个DCLR。${Q}_{F}$总是可以通过扩展$Q$的输出变量集从$Q$得到（在最坏的情况下，将所有变量添加到输出中）。当然，在这种策略下，我们需要$\Omega \left( \begin{Vmatrix}{{Q}^{\prime }\left( {db}\right) }\end{Vmatrix}\right)$的空间，就像$\left( \mathrm{H}\right) \mathrm{{IVM}}$一样，但我们避免了$Q$及其增量的（部分）物化。如第5.3节所示，这一特性实际上使DYN在处理时间和空间上都优于HIVM。

An important optimization that our implementation applies in this context, is that of early computation of aggregate functions that are restricted to variables of a single atom. For example,consider ${Q}^{\prime } = \left( {x,y,\operatorname{SUM}\left( t\right) }\right) {\pi }_{x,y,t}$ $\left( {R\left( {x,y,t}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$ . Our implementation will actually run Dyn on ${\pi }_{x,y}\left( {{R}^{\prime }\left( {x,y}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$ where ${R}^{\prime }$ is the GMR that maps tuple $\left( {x,y}\right)  \mapsto  \mathop{\sum }\limits_{t}t \times  R\left( {x,y,t}\right)$ . Note that ${R}^{\prime }$ can be maintained under updates to $R$ .

我们的实现方案在这种情况下应用的一个重要优化是对限制在单个原子变量上的聚合函数进行提前计算。例如，考虑${Q}^{\prime } = \left( {x,y,\operatorname{SUM}\left( t\right) }\right) {\pi }_{x,y,t}$ $\left( {R\left( {x,y,t}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$。我们的实现方案实际上会在${\pi }_{x,y}\left( {{R}^{\prime }\left( {x,y}\right)  \boxtimes  S\left( {y,z,v}\right) }\right)$上运行Dyn，其中${R}^{\prime }$是将元组$\left( {x,y}\right)  \mapsto  \mathop{\sum }\limits_{t}t \times  R\left( {x,y,t}\right)$进行映射的广义多关系（GMR）。请注意，${R}^{\prime }$可以在对$R$进行更新时得到维护。

Sub-queries Before proceeding to the experimental evaluation of Dyn, we briefly discuss how to evaluate queries with sub-queries. Recall from Proposition 4.18 that $T$ -reps have a particularly interesting property: If $\mathcal{D}$ is a $T$ -rep and $Q$ is compatible with $T$ ,then the multiplicity of an arbitrary tuple $\overrightarrow{t}$ in $Q\left( {db}\right)$ can be calculated in constant time from $\mathcal{D}$ . This is highly relevant in practice,since when evaluating queries with IN or EXIST sub-queries, it suffices to maintain two DCLRs, one for the subquery and one for the outer query. From the viewpoint of the outer query, the subquery DCLR then behaves as an input GMR.

子查询在对Dyn进行实验评估之前，我们简要讨论一下如何评估带有子查询的查询。回顾命题4.18可知，$T$ - 表示具有一个特别有趣的性质：如果$\mathcal{D}$是一个$T$ - 表示，并且$Q$与$T$兼容，那么任意元组$\overrightarrow{t}$在$Q\left( {db}\right)$中的重数可以从$\mathcal{D}$在常数时间内计算得出。这在实际应用中非常重要，因为在评估带有IN或EXIST子查询的查询时，只需维护两个DCLR，一个用于子查询，一个用于外部查询。从外部查询的角度来看，子查询的DCLR就像一个输入的GMR。

Generalized Join Trees When dynamically processing a $\mathrm{{CQ}}$ ,the join tree under consideration can impact the performance of Dyn. For example, one would expect that when processing a $q$ -hiearchical query,DYN performs better using a simple tree than a non-simple tree. One could measure how simple a tree is by estimating the amount of single-tuple updates that will be processed in constant time by Dyn. Although there are well-known algorithms for heuristic search of hypertree decompositions [18], their objective is to find low-width decompositions, and therefore are not well-suited for our setting. We have developed a simple cost model for generalized join trees and have used minimum-cost trees for experimentation. For the sake of space, the details of this cost model are left to the full version of the paper.

广义连接树在动态处理$\mathrm{{CQ}}$时，所考虑的连接树会影响Dyn的性能。例如，人们可能会期望在处理$q$ - 层次查询时，使用简单树时DYN的性能比使用非简单树时更好。可以通过估计Dyn将在常数时间内处理的单元组更新的数量来衡量一棵树的简单程度。虽然有一些众所周知的用于超树分解启发式搜索的算法[18]，但它们的目标是找到低宽度的分解，因此不太适合我们的场景。我们为广义连接树开发了一个简单的成本模型，并使用最小成本树进行实验。由于篇幅限制，这个成本模型的详细信息将在论文的完整版本中给出。

<!-- Media -->

<table><tr><td colspan="2">Benchmark</td><td>Query</td><td>#of tuples</td></tr><tr><td rowspan="13">TPC-H</td><td rowspan="4">Full joins</td><td>FQ1</td><td>2,833,827</td></tr><tr><td>FQ2</td><td>2,617,163</td></tr><tr><td>FQ3</td><td>2,820,494</td></tr><tr><td>FQ4</td><td>2,270,494</td></tr><tr><td rowspan="9">Aggregate queries</td><td>Q1</td><td>7,999,406</td></tr><tr><td>Q3</td><td>10,199,406</td></tr><tr><td>Q4</td><td>9,999,406</td></tr><tr><td>Q6</td><td>7,999,406</td></tr><tr><td>Q9</td><td>11,346,069</td></tr><tr><td>Q12</td><td>9,999,406</td></tr><tr><td>Q13</td><td>2,200,000</td></tr><tr><td>Q16</td><td>1,333,330</td></tr><tr><td>Q18</td><td>10,199,406</td></tr><tr><td rowspan="5">TPC-DS</td><td>Full joins</td><td>FQ5</td><td>10,669,570</td></tr><tr><td rowspan="4">Aggregate queries</td><td>Q3</td><td>11,638,073</td></tr><tr><td>Q7</td><td>13,559,239</td></tr><tr><td>Q19</td><td>11,987,115</td></tr><tr><td>Q22</td><td>36,138,621</td></tr></table>

<table><tbody><tr><td colspan="2">基准测试</td><td>查询</td><td>元组数量</td></tr><tr><td rowspan="13">TPC - H（事务处理性能委员会 - 混合基准测试）</td><td rowspan="4">全连接</td><td>FQ1</td><td>2,833,827</td></tr><tr><td>FQ2</td><td>2,617,163</td></tr><tr><td>FQ3</td><td>2,820,494</td></tr><tr><td>FQ4</td><td>2,270,494</td></tr><tr><td rowspan="9">聚合查询</td><td>Q1</td><td>7,999,406</td></tr><tr><td>Q3</td><td>10,199,406</td></tr><tr><td>Q4</td><td>9,999,406</td></tr><tr><td>Q6</td><td>7,999,406</td></tr><tr><td>Q9</td><td>11,346,069</td></tr><tr><td>Q12</td><td>9,999,406</td></tr><tr><td>Q13</td><td>2,200,000</td></tr><tr><td>Q16</td><td>1,333,330</td></tr><tr><td>Q18</td><td>10,199,406</td></tr><tr><td rowspan="5">TPC - DS（事务处理性能委员会 - 决策支持基准测试）</td><td>全连接</td><td>FQ5</td><td>10,669,570</td></tr><tr><td rowspan="4">聚合查询</td><td>Q3</td><td>11,638,073</td></tr><tr><td>Q7</td><td>13,559,239</td></tr><tr><td>Q19</td><td>11,987,115</td></tr><tr><td>Q22</td><td>36,138,621</td></tr></tbody></table>

Table 1: Number of tuples in the stream file of each query

表1：每个查询的流文件中的元组数量

<!-- Media -->

### 5.2 Experimental Setup

### 5.2 实验设置

Queries and update streams. We evaluate the subset of queries available in the industry-standard benchmarks $\mathrm{{TPC}} - \mathrm{H}$ and $\mathrm{{TPC}} - \mathrm{{DS}}$ that can be evaluated by the methods described throughout this paper. In particular, we evaluate those queries involving only equijoins, whose FROM-WHERE caluses are acyclic. Queries are divided into acyclic full-join queries (called FQs) and acyclic aggregate queries. Acyclic full join queries are generated by taking the FROM clause of the corresponding queries on the benchmarks. It is important to mention that we omit the ORDER BY and LIMIT clauses, we replaced the left-outer join in query Q13 by an equijoin, and modified Q16 to remove an inequality. We discard those queries using the MIN and MAX aggregate functions as this is not supported by our current implementation. We report all the evaluated queries in Appendix D.

查询和更新流。我们评估了行业标准基准测试 $\mathrm{{TPC}} - \mathrm{H}$ 和 $\mathrm{{TPC}} - \mathrm{{DS}}$ 中可用的查询子集，这些查询可以通过本文中描述的方法进行评估。具体而言，我们评估那些仅涉及等值连接（equijoin）的查询，其FROM - WHERE子句是无环的。查询分为无环全连接查询（称为FQ）和无环聚合查询。无环全连接查询是通过采用基准测试中相应查询的FROM子句生成的。需要注意的是，我们省略了ORDER BY和LIMIT子句，将查询Q13中的左外连接替换为等值连接，并修改了Q16以消除一个不等式。我们舍弃了那些使用MIN和MAX聚合函数的查询，因为我们当前的实现不支持这些函数。我们在附录D中报告了所有评估的查询。

Our update streams consist of single-tuple insertions only and are generated as follows. We use the data-generating utilities of the benchmarks, namely dbgen for TPC-H and dsdgen for TPC-DS ${}^{3}$ . We used scale factor 0.5 and 2 for the FQs from TPC-H and TPC-DS, respectively, and scale factor 2 and 4 for the aggregate queries from TPC-H and TPC-DS, respectively. Notice that the data-generating tools create datasets for a fixed schema, while most queries do not use the complete set of relations. The update streams are generated by randomly selecting the tuples to be inserted from the relations that occur in each query. To use the same update streams for evaluating both DYN and HIVM, each stream is stored in a file. The number of tuples on each file is depicted in Table 1.

我们的更新流仅由单元组插入组成，生成方式如下。我们使用基准测试的数据生成工具，即TPC - H的dbgen和TPC - DS的dsdgen ${}^{3}$。对于来自TPC - H和TPC - DS的FQ，我们分别使用了0.5和2的缩放因子；对于来自TPC - H和TPC - DS的聚合查询，我们分别使用了2和4的缩放因子。请注意，数据生成工具为固定模式创建数据集，而大多数查询并不使用完整的关系集。更新流是通过从每个查询中出现的关系中随机选择要插入的元组来生成的。为了使用相同的更新流来评估DYN和HIVM，每个流都存储在一个文件中。每个文件中的元组数量如表1所示。

---

<!-- Footnote -->

${}^{3}$ dbgen and dsgen are available at http://www.tpc.org/

${}^{3}$ dbgen和dsgen可从http://www.tpc.org/获取

<!-- Footnote -->

---

Comparison to HIVM. As discussed in the introduction, HIVM is an efficient method for dynamic query evaluation that highly improves processing time over IVM [26]. We compare our implementation against DBToaster [26], a state-of-the-art HIVM engine. DBToaster is particularly meticulous in that it materializes only useful views, and therefore it is an interesting implementation for comparison in both throughput and memory usage. Moreover, DBToaster has been extensively tested and proven to be more efficient than a commercial database management system, a commercial stream processing system and an IVM implementation [26]. DBToaster compiles SQL queries into trigger programs for different programming languages. We compare against those in Scala, the same programming language used in our implementation. It is important to mention that programs compiled by DBToaster use the so-called ${akka}{\text{actors}}^{4}$ to generate update tuples. During our experiments, we have found that this creates an unnecessary memory overhead by creating many temporary objects. For a fair comparison we have therefore removed these actors from DBToaster.

与HIVM的比较。如引言中所述，HIVM是一种用于动态查询评估的高效方法，与增量视图维护（IVM）相比，它显著提高了处理时间 [26]。我们将我们的实现与DBToaster [26] 进行比较，DBToaster是一种先进的HIVM引擎。DBToaster特别精细，因为它只物化有用的视图，因此它在吞吐量和内存使用方面都是一个值得比较的有趣实现。此外，DBToaster已经过广泛测试，并被证明比商业数据库管理系统、商业流处理系统和IVM实现更高效 [26]。DBToaster将SQL查询编译成不同编程语言的触发器程序。我们与使用Scala编写的程序进行比较，Scala也是我们实现中使用的编程语言。需要注意的是，DBToaster编译的程序使用所谓的 ${akka}{\text{actors}}^{4}$ 来生成更新元组。在我们的实验中，我们发现这会通过创建许多临时对象产生不必要的内存开销。因此，为了进行公平比较，我们从DBToaster中移除了这些参与者。

Operational setup. The experiments are performed on a machine running GNU/Linux with an Intel Core i7 processor running at ${3.07}\mathrm{{GHz}}$ . We use version 2.11.8 of the Scala programming language, version 1.8.0_101 of the Java Virtual Machine, and version 2.2 of the DBToaster compiler. Each query is evaluated 10 times against each of the two engines for measuring time, and two times for measuring memory; the presented results are the average measurements over those runs. Every time a query is evaluated, ${16}\mathrm{{GB}}$ of main memory are freshly allocated to the corresponding program. To measure memory usage we use the Java Virtual Machine (JVM) System calls. We measure the memory consumption every 1000 updates, and consider the maximum value. For a fair comparison, we call the garbage collector before each memory measurement. The time used by the garbage collector is not considered in the measurements of throughput.

操作设置。实验在运行GNU/Linux的机器上进行，该机器配备了运行频率为 ${3.07}\mathrm{{GHz}}$ 的英特尔酷睿i7处理器。我们使用Scala编程语言的2.11.8版本、Java虚拟机的1.8.0_101版本和DBToaster编译器的2.2版本。每个查询针对两个引擎分别评估10次以测量时间，评估2次以测量内存；呈现的结果是这些运行的平均测量值。每次评估查询时，都会为相应的程序新分配 ${16}\mathrm{{GB}}$ 的主内存。为了测量内存使用情况，我们使用Java虚拟机（JVM）系统调用。我们每1000次更新测量一次内存消耗，并考虑最大值。为了进行公平比较，我们在每次内存测量之前调用垃圾回收器。垃圾回收器使用的时间不包含在吞吐量的测量中。

### 5.3 Experimental results

### 5.3 实验结果

Figure 4 depicts the resources used by Dyn as a percentage of the resources used by DBToaster. For each query, we plot the percentage of memory used by DYN considering that ${100}\%$ is to the memory used by DBToaster,and the same is done for processing time. This improves readability and normalizes the chart. To present the absolute values, on top of the bars corresponding to each query we write the memory and time used by DBToaster. Some executions of DBToaster failed because they required more than ${16}\mathrm{{GB}}$ of main memory. In those cases, we report 16GB of memory and the time it took the execution to raise an exception. We mark such queries with an asterisk (*) in Figure 4. Note that DYN never runs out of memory, and times reported for DYN are the times required to process the entire update stream.

图4展示了Dyn使用的资源占DBToaster使用资源的百分比。对于每个查询，我们绘制了Dyn使用的内存百分比，假设${100}\%$是DBToaster使用的内存，处理时间的绘制方式相同。这提高了可读性并对图表进行了归一化处理。为了展示绝对值，在对应每个查询的条形图上方，我们标注了DBToaster使用的内存和时间。DBToaster的一些执行失败了，因为它们需要超过${16}\mathrm{{GB}}$的主内存。在这些情况下，我们报告使用了16GB的内存以及执行引发异常所需的时间。我们在图4中用星号（*）标记了这些查询。请注意，Dyn永远不会耗尽内存，并且报告的Dyn时间是处理整个更新流所需的时间。

Full-join queries. For full join queries (FQ1-FQ5), Figure 4 shows that DYN outperforms DBToaster by close to one order of magnitude, both in memory consumption and processing time. The difference in memory consumption is expected, since the result of full-join queries can be poly-nomially larger than the input dataset, and DBToaster materializes these results. The difference in processing time, then,is a consequence of DYN’s maintenance of $T$ -reps rather than the results themselves. The average processing time for DBToaster over FQ1-FQ5 is 128.49 seconds, while for DYN it is 29.85 seconds. This includes FQ1, FQ3, FQ4 and FQ5, for which DBToaster reached the memory limit. Then, 128.49 seconds is only a lower bound for the average processing time of DBToaster over FQ1-FQ5. Regarding memory consumption, DBToaster requires in average 14.68 GB for FQ1-FQ5 (considering a limit of ${16}\mathrm{{GB}}$ ),compared to the ${2.74}\mathrm{{GB}}$ required by Dyn. Note that the query presenting the biggest difference, FQ4, is a q-hierarchical query (see Section 4.4).

全连接查询。对于全连接查询（FQ1 - FQ5），图4显示，在内存消耗和处理时间方面，Dyn的性能比DBToaster高出近一个数量级。内存消耗的差异是可以预期的，因为全连接查询的结果可能比输入数据集大很多，而DBToaster会物化这些结果。处理时间的差异则是由于Dyn维护的是$T$ - 表示（$T$ - reps）而不是结果本身。DBToaster在FQ1 - FQ5上的平均处理时间为128.49秒，而Dyn为29.85秒。这包括了DBToaster达到内存限制的FQ1、FQ3、FQ4和FQ5。因此，128.49秒只是DBToaster在FQ1 - FQ5上平均处理时间的下限。关于内存消耗，DBToaster在FQ1 - FQ5上平均需要14.68GB（考虑到${16}\mathrm{{GB}}$的限制），而Dyn只需要${2.74}\mathrm{{GB}}$。请注意，差异最大的查询FQ4是一个q - 分层查询（见第4.4节）。

Aggregate queries. For aggregate queries, Figure 4 shows that DYN can significantly improve the memory consumption of HIVM while improving processing time up to one order of magnitude for TPC-H Q13' and TPC-DS Q7.

聚合查询。对于聚合查询，图4显示，Dyn可以显著改善HIVM的内存消耗，同时对于TPC - H Q13'和TPC - DS Q7，处理时间最多可提高一个数量级。

For TPC-H queries Q1, Q3, and Q6, DYN equals DBToast-ers memory consumption. For these queries, the algorithms used by Dyn and DBToaster are nearly identical which is why DYN and DBToaster require the same amount of memory. The difference in execution time for these queries is due to implementation specifics. For example we have detected that DBToaster parses tuple attributes before filtering particular attributes in the WHERE clause. Our implementation, in contrast, does lazy parsing, meaning that each attribute is parsed only when it is used. In particular, if a certain attribute fails its local condition, then subsequent attributes are not parsed.

对于TPC - H查询Q1、Q3和Q6，Dyn的内存消耗与DBToaster相同。对于这些查询，Dyn和DBToaster使用的算法几乎相同，这就是为什么Dyn和DBToaster需要相同数量的内存。这些查询执行时间的差异是由于实现细节造成的。例如，我们发现DBToaster在WHERE子句中过滤特定属性之前会解析元组属性。相比之下，我们的实现采用了惰性解析，即每个属性仅在使用时才进行解析。特别是，如果某个属性未通过其局部条件，则后续属性不会被解析。

The biggest difference in processing time is observed for TPC-H query Q13' and TPC-DS query Q7. Q13' has a sub-query that computes the amount of orders processed for each customer. It then counts the number of customers for which $k$ orders were processed,for each $k$ . To process this, DBToaster almost fully recomputes the sub-query each time a new update arrives, which basically yields a quadratic algorithm. In contrast, our implementation also uses DYN to maintain the sub-query as a $T$ -rep,supporting,for this particular case, constant update time. For Q7, the aggressive materialization of delta queries causes DBToaster to maintain 88 different GMRs. In contrast,to maintain its $T$ -rep, DYN only needs to store 5 GMRs and 5 indexes.

在处理时间上，TPC - H查询Q13'和TPC - DS查询Q7的差异最大。Q13'有一个子查询，用于计算每个客户处理的订单数量。然后，对于每个$k$，它会统计处理了$k$个订单的客户数量。为了处理这个问题，每当有新的更新到来时，DBToaster几乎会完全重新计算子查询，这基本上产生了一个二次算法。相比之下，我们的实现也使用Dyn将子查询维护为一个$T$ - 表示（$T$ - rep），在这种特定情况下，支持恒定的更新时间。对于Q7，增量查询的激进物化导致DBToaster维护88个不同的GMR（广义匹配规则，Generalized Match Rule）。相比之下，为了维护其$T$ - 表示，Dyn只需要存储5个GMR和5个索引。

Scalability. To show that DYN performs in a consistent way against streams of different sizes, we report the processing time and memory consumption each time a ${10}\%$ of the stream is processed (Figure 6). The results show that for all queries the memory and time increase linearly with the amount of tuples processed. We can see that Dyn is constantly faster and scales more consistently. The same phenomena occur for memory consumption. Due to space constraints, we only report the measurements for the TPC-H queries FQ1, Q3 and Q18.

可扩展性。为了表明Dyn在处理不同大小的流时表现一致，我们报告了每次处理流的${10}\%$时的处理时间和内存消耗（图6）。结果显示，对于所有查询，内存和时间随着处理的元组数量线性增加。我们可以看到，Dyn始终更快，并且扩展性更稳定。内存消耗也出现了相同的现象。由于篇幅限制，我们仅报告了TPC - H查询FQ1、Q3和Q18的测量结果。

Enumeration of query results. We know from Section 4.1 that $T$ -reps feature constant delay enumeration,but this theoretical notion hides a constant factor that could decrease performance in practice. To show that this is not the case, we have measured the time needed for enumerating and writing to secondary memory the results of FQ1 to FQ4 from their corresponding DCLRs. We use update streams of different sizes, and for comparison we measure the time needed to iterate over the materialized set of results (from an in-memory array) and write them to secondary memory. The results are depicted in Figure 5. Interestingly, for larger result sizes, enumerating from a T-rep was slightly more efficient than enumerating from an in-memory array. A possible explanation is illustrated by the following example. Consider the full-join query $R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$ ,and assume there are several tuples in the join result. It is not hard to see that given a fixed $B$ value,from a $T$ -rep we can iterate over the $C$ values corresponding to each $A$ value. This way,the $A$ and $B$ values are not re-assigned while generating several tuples. In contrast, every time a tuple is read from an array each value needs to be read again.

查询结果的枚举。从4.1节我们知道，$T$ -表示（$T$ -rep）具有常量延迟枚举的特性，但这一理论概念隐藏了一个可能在实际中降低性能的常量因子。为了证明实际并非如此，我们测量了从相应的DCLR（数据依赖约束语言表示）中枚举并将FQ1到FQ4的结果写入二级存储所需的时间。我们使用不同大小的更新流，并且为了进行比较，我们还测量了遍历物化结果集（来自内存数组）并将其写入二级存储所需的时间。结果如图5所示。有趣的是，对于较大的结果集，从T - 表示（T - rep）中枚举比从内存数组中枚举略有效率。下面的例子说明了一种可能的解释。考虑全连接查询$R\left( {A,B}\right)  \boxtimes  S\left( {B,C}\right)$ ，并假设连接结果中有多个元组。不难看出，给定一个固定的$B$ 值，从$T$ - 表示中我们可以遍历每个$A$ 值对应的$C$ 值。这样，在生成多个元组时，$A$ 和$B$ 值不会被重新赋值。相反，每次从数组中读取一个元组时，每个值都需要重新读取。

---

<!-- Footnote -->

${}^{4}$ http://doc.akka.io/docs/akka/snapshot/scala/actors.html

${}^{4}$ http://doc.akka.io/docs/akka/snapshot/scala/actors.html

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: % Processing Time % Memory Consumption * DBToaster memory consumption exceeded 16GB and its execution was halted 11,125ME 231MB 546MB 2,146ME 1,071MB 1,742MB 16.000MB 5,149MB 10,599MB 132.05s 28.0s 30635.0s 437.07s 100.0 83.9 82.0 75.8 71.5 64.5 39.4 38.2 31.4 9.4 .0 .7 Q16’ TPC-H Aggregate queries TPC-DS Aggregate queries 16,000MB 9,389MB 16,000MB 16,000MB 16,000MB 49MB 336MB 221MB 40MB 70.0s 26.47s 90.0s 100 98.0 103.6 80 69.3 60 45.8 40 41.8 20 16.1 $\begin{array}{ll} {16.8} & {10.0} \end{array}$ $\begin{array}{ll} {11.6} & {4.0} \end{array}$ FQ3* FQ4* FQ5* Full-join queries -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_11.jpg?x=151&y=143&w=1485&h=536&r=0"/>

Figure 4: DYN usage of resources as a percentage of the resources consumed by DBToaster (lower is better).

图4：DYN的资源使用情况，以占DBToaster消耗资源的百分比表示（数值越低越好）。

<!-- figureText: T-rep Array FQ1 - 1.3GB FQ3 - 1.6gb Query - Output size (MB) Enumeration Time (sec) 40 30 20 10 0 FQ4 - 215.6MB FQ2 - 424.3MB -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_11.jpg?x=151&y=794&w=723&h=370&r=0"/>

Figure 5: Time for enumerating output (lower is better)

图5：枚举输出所需的时间（数值越低越好）

<!-- Media -->

Full query recomputation. In Section 4.3 we mentioned that, in theory, the worst-case complexity for updating a $T$ -rep when $T$ is not simple is the same as that of recomputing the Yannakakis algorithm from scratch. However, we can expect DYN to be much faster than the naive full-recomputation algorithm as it only updates those portions of the $T$ -rep that are affected. This is indeed the case in practice. We tested both strategies over different datasets for FQ1 and FQ4. In average, the naive recomputation turned out to process updates 190 times slower than Dyn. Due to space constraints we do not report the full results.

全查询重新计算。在4.3节中我们提到，理论上，当$T$ 不简单时，更新$T$ - 表示的最坏情况复杂度与从头重新计算Yannakakis算法的复杂度相同。然而，我们可以预期DYN比简单的全重新计算算法快得多，因为它只更新$T$ - 表示中受影响的部分。实际上确实如此。我们在不同的数据集上对FQ1和FQ4测试了这两种策略。平均而言，简单的重新计算处理更新的速度比Dyn慢190倍。由于篇幅限制，我们没有报告完整的结果。

## ACKNOWLEDGEMENTS

## 致谢

M. Idris is supported by the Erasmus Mundus Joint Doctorate in "Information Technologies for Business Intelligence - Doctoral College (IT4BI-DC)". M. Ugarte is supported by the Brussels Captial Region-Innoviris (project SPICES). S. Vansummeren acknowledges gracious support from the Wiener-Anspach foundation.

M. Idris由伊拉斯谟世界联合博士项目“商业智能信息技术 - 博士学院（IT4BI - DC）”资助。M. Ugarte由布鲁塞尔首都大区 - Innoviris（SPICES项目）资助。S. Vansummeren感谢维纳 - 安斯巴赫基金会的慷慨支持。

<!-- Media -->

<!-- figureText: DYN (time) DBToaster (time) DYN (memory) DBToaster (memory) 30 20 10 60% 70% 80% 100% 48 12 6 60% 70% 80% 90% 100% 80 70 10 0 60% 70% 80% 90% 100% 16000 14000 FQ1 12000 10000 8000 6000 4000 2000 0 10% 20% 30% 50% 1600 1400 Memory consumption (MB) Q18 - TPC-H 1200 1000 800 600 400 0 10% 20% 30% 40% 50% 2000 1750 Q3 - TPC-DS 1500 1250 750 500 250 10% 20% 30% 40% 50% -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_11.jpg?x=925&y=839&w=723&h=1151&r=0"/>

Figure 6: Resource utilization v/s% of tuples processed

图6：资源利用率与处理元组的百分比对比

<!-- Media -->

## 6. REFERENCES

## 6. 参考文献

[1] S. Abiteboul, R. Hull, and V. Vianu. Foundations of databases. Addison-Wesley, 1995.

[2] M. Abo Khamis, H. Q. Ngo, and A. Rudra. FAQ: Questions asked frequently. In Proc. of PODS, pages 13-28, 2016.

[3] M. E. Adiba and B. G. Lindsay. Database snapshots. In Proc. of VLDB 1980, pages 86-91, 1980.

[4] S. M. Aji and R. J. McEliece. The generalized distributive law. IEEE Trans. Information Theory, 46(2):325-343, 2006.

[5] G. Bagan, A. Durand, and E. Grandjean. On acyclic conjunctive queries and constant delay enumeration. In Proc. of CSL, pages 208-222, 2007.

[6] N. Bakibayev, T. Kočiský, D. Olteanu, and J. Závodný. Aggregation and ordering in factorised databases. Proc. of VLDB, 6(14):1990-2001, 2013.

[7] C. Berkholz, J. Keppeler, and N. Schweikardt. Answering conjunctive queries under updates. In Proc. of PODS, 2017. To appear.

[8] J. A. Blakeley, N. Coburn, and P.-V. Larson. Updating derived relations: Detecting irrelevant and autonomously computable updates. ACM TODS, (3):369-400, 1989.

[9] J. A. Blakeley, P.-A. Larson, and F. W. Tompa. Efficiently updating materialized views. In Proc. of SIGMOD, pages 61-71, 1986.

[10] J. Brault-Baron. De la pertinence de l'énumération: complexité en logiques. PhD thesis, Université de Caen, 2013.

[11] O. P. Buneman and E. K. Clemons. Efficiently monitoring relational databases. ACM TODS, (3):368-382, 1979.

[12] S. Ceri and J. Widom. Deriving production rules for incremental view maintenance. In Proc. of VLDB, pages 577-589, 1991.

[13] R. Chirkova and J. Yang. Materialized Views. Now Publishers Inc., Hanover, MA, USA, 2012.

[14] T. Cormen. Introduction to Algorithms, 3rd Edition:. MIT Press, 2009.

[15] G. Cugola and A. Margara. Processing flows of information: From data stream to complex event processing. ACM CSUR, 44(3):15:1-15:62, 2012.

[16] N. Dalvi and D. Suciu. The dichotomy of probabilistic inference for unions of conjunctive queries. $J.{ACM}$ , 59(6):30:1-30:87, 2013.

[17] R. Fink and D. Olteanu. Dichotomies for queries with negation in probabilistic databases. ACM TODS, 41(1):4:1-4:47, 2016.

[18] G. Gottlob, M. Grohe, n. Musliu, M. Samer, and F. Scarcello. Hypertree decompositions: Structure, algorithms, and applications. In Proc. of WG, pages 1-15, 2005.

[19] G. Gottlob, N. Leone, and F. Scarcello. Robbers, marshals, and guards: game theoretic and logical characterizations of hypertree width. J. Comput. Syst. Sci., 66(4):775-808, 2003.

[20] A. Gupta, I. S. Mumick, and V. S. Subrahmanian. Maintaining views incrementally. In Proc. of SIGMOD, pages 157-166, 1993.

[21] H. Gupta and I. S. Mumick. Incremental maintenance

of aggregate and outerjoin expressions. Information

聚合和外连接表达式的。信息

Systems, (6):435-464, 2006.

[22] M. Henzinger, S. Krinninger, D. Nanongkai, and T. Saranurak. Unifying and strengthening hardness for dynamic problems via the online matrix-vector multiplication conjecture. In Proc. of STOC, pages 21-30, 2015.

[23] M. R. Joglekar, R. Puttagunta, and C. Ré. Ajar: Aggregations and joins over annotated relations. In Proc. of PODS, pages 91-106, 2016.

[24] A. Kawaguchi, D. Lieuwen, I. Mumick, and K. Ross. Implementing incremental view maintenance in nested data models. In Proc. of DBPL, pages 202-221, 1997.

[25] C. Koch. Incremental query evaluation in a ring of databases. In Proc. of PODS, pages 87-98, 2010.

[26] C. Koch, Y. Ahmad, O. Kennedy, M. Nikolic, A. Nötzli, D. Lupei, and A. Shaikhha. Dbtoaster: higher-order delta processing for dynamic, frequently fresh views. VLDB Journal, pages 253-278, 2014.

[27] P. Koutris and D. Suciu. Parallel evaluation of conjunctive queries. In Proc. of PODS, pages 223-234, 2011.

[28] I. S. Mumick, D. Quass, and B. S. Mumick. Maintenance of data cubes and summary tables in a warehouse. SIGMOD Records, 26(2):100-111, 1997.

[29] H. Q. Ngo, C. Ré, and A. Rudra. Skew strikes back: New developments in the theory of join algorithms. SIGMOD Records, 42(4):5-16, 2014.

[30] M. Nikolic, M. Dashti, and C. Koch. How to win a hot dog eating contest: Distributed incremental view maintenance with batch updates. In Proc. of SIGMOD, pages 511-526, 2016.

[31] D. Olteanu and J. Závodný. Size bounds for factorised representations of query results. ACM TODS, 40(1):2:1-2:44, 2015.

[32] C. H. Papadimitriou. Computational complexity. In Encyclopedia of Computer Science, pages 260-265. 2003.

[33] X. Qian and G. Wiederhold. Incremental recomputation of active relational expressions. IEEE Trans. on Knowl. and Data Eng., 3(3):337-341, 1991.

[34] K. A. Ross, D. Srivastava, and S. Sudarshan. Materialized view maintenance and integrity constraint checking: Trading space for time. In Proc. of SIGMOD, pages 447-458, 1996.

[35] N. Roussopoulos. Materialized views and data warehouses. SIGMOD Records, 27(1):21-26, 1998.

[36] M. Schleich, D. Olteanu, and R. Ciucanu. Learning linear regression models over factorized joins. In Proc. of SIGMOD, pages 3-18, 2016.

[37] L. Segoufin. Constant delay enumeration for conjunctive queries. SIGMOD Record, 44(1):10-17, 2015.

[38] M. Y. Vardi. The complexity of relational query languages (extended abstract). In Proc. of STOC, pages 137-146, 1982.

[39] M. Yannakakis. Algorithms for acyclic database schemes. In Proc. of VLDB, pages 82-94, 1981.

## APPENDIX

## 附录

## A. PROOFS FROM SECTION 3

## A. 第3节的证明

Readers familiar with GHDs of arbitrary width may observe that GHDs are normally defined as triples $\left( {T,\chi ,\lambda }\right)$ with $T$ a tree; $\chi$ a function that assigns a set of variables to each node, and $\lambda$ a function that assigns a set of atoms to each node (see [19]). Since we focus on GHDs of width one, and hence do not need the full richness of GHDs,we omit $\chi$ and $\lambda$ from our definition. These can be recovered by fixing $\chi  : n \rightarrow  \operatorname{var}\left( n\right)$ and $\lambda$ to be the function that maps atoms $\mathbf{a} \mapsto  \{ \mathbf{a}\}$ and hyperedges $\bar{x} \mapsto  \{ \mathbf{b}\}$ where $\mathbf{b}$ is some atom with $\bar{x} \subseteq  \operatorname{var}\left( \mathbf{b}\right)$ . Note that,since under this definition $\left| {\lambda \left( n\right) }\right|  = 1$ for all nodes, this indeed yields a GHD of width 1 .

熟悉任意宽度广义超树分解（GHDs）的读者可能会注意到，GHDs通常被定义为三元组$\left( {T,\chi ,\lambda }\right)$ ，其中$T$ 是一棵树；$\chi$ 是一个为每个节点分配一组变量的函数，$\lambda$ 是一个为每个节点分配一组原子的函数（见[19]）。由于我们关注的是宽度为1的GHDs，因此不需要GHDs的全部特性，我们在定义中省略了$\chi$ 和$\lambda$ 。可以通过将$\chi  : n \rightarrow  \operatorname{var}\left( n\right)$ 和$\lambda$ 固定为将原子$\mathbf{a} \mapsto  \{ \mathbf{a}\}$ 和超边$\bar{x} \mapsto  \{ \mathbf{b}\}$ 映射的函数来恢复它们，其中$\mathbf{b}$ 是某个具有$\bar{x} \subseteq  \operatorname{var}\left( \mathbf{b}\right)$ 的原子。注意，根据这个定义，对于所有节点都有$\left| {\lambda \left( n\right) }\right|  = 1$ ，这确实产生了一个宽度为1的GHD。

Proposition 3.4. If there exists a width-one GHD for set of atoms $A$ ,then there also exists a generalized join tree for $A$ . Consequently,a ${CQQ}$ is acyclic iff at(Q)has a generalized join tree.

命题3.4。如果对于原子集$A$ 存在一个宽度为1的GHD，那么对于$A$ 也存在一个广义连接树。因此，一个${CQQ}$ 是无环的当且仅当at(Q)有一个广义连接树。

Crux. A traditional join tree for $A$ is a width-one GHD $T$ for $A$ in which all the nodes are atoms in $A$ (no hyperedges allowed). It is well-known that a width-one GHD exists for $A$ if,and only if,a traditional join tree $T$ exists for $A$ [19]. It hence suffices to show that every traditional join tree $T$ for $A$ can be transformed into a generalized join tree ${T}^{\prime }$ for $A$ . We do this by this by recursively applying the following transformation rule to nodes in $T$ ,starting at the root:

关键。对于$A$的传统连接树是一个宽度为 1 的广义超树分解（Generalized Hypergraph Decomposition，GHD）$T$，其中所有节点都是$A$中的原子（不允许有超边）。众所周知，当且仅当存在$A$的传统连接树$T$时，$A$才存在宽度为 1 的广义超树分解[19]。因此，只需证明$A$的每个传统连接树$T$都可以转换为$A$的广义连接树${T}^{\prime }$。我们通过从根节点开始，对$T$中的节点递归应用以下转换规则来实现这一点：

Let $n$ be the current node being transformed. If hyperedge $\operatorname{var}\left( n\right)$ is not yet in ${T}^{\prime }$ ,then add $\operatorname{var}\left( n\right)$ to ${T}^{\prime }$ and add an edge from $\operatorname{var}\left( n\right)$ to $\operatorname{var}\left( p\right)$ with $p$ the parent of $n$ in $T$ . Finally (and even if $\operatorname{var}\left( n\right)$ were already in ${T}^{\prime }$ ),add $n$ (which is an atom) to $T$ and add an edge from $n$ to $\operatorname{var}\left( n\right)$ . Then, recursively apply this procedure to each child of $n$ in $T$ .

设$n$是当前正在转换的节点。如果超边$\operatorname{var}\left( n\right)$尚未在${T}^{\prime }$中，则将$\operatorname{var}\left( n\right)$添加到${T}^{\prime }$中，并从$\operatorname{var}\left( n\right)$到$\operatorname{var}\left( p\right)$添加一条边，其中$p$是$n$在$T$中的父节点。最后（即使$\operatorname{var}\left( n\right)$已经在${T}^{\prime }$中），将$n$（它是一个原子）添加到$T$中，并从$n$到$\operatorname{var}\left( n\right)$添加一条边。然后，对$n$在$T$中的每个子节点递归应用此过程。

To illustrate, if we apply this procedure to traditional join tree ${T}_{1}$ of Figure 2 then we obtain the following generalized join tree.

为了说明这一点，如果我们对图 2 中的传统连接树${T}_{1}$应用此过程，那么我们将得到以下广义连接树。

<!-- Media -->

<!-- figureText: $R\left( {x,y,z}\right) S\left( {x,y,u}\right) \left\lbrack  {y,v,w}\right\rbrack$ $T\left( {y,v,w}\right) U\left( {y,v,p}\right)$ -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_13.jpg?x=298&y=1321&w=420&h=185&r=0"/>

<!-- Media -->

It is a standard exercise to show that this transformation indeed always yields a generalized join tree.

证明此转换确实总是能得到一个广义连接树是一个标准练习。

## B. PROOFS FROM SECTION 4.3

## B. 第 4.3 节的证明

Lemma 4.11. If $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)$ then $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ for some guard $c \in  \operatorname{grd}\left( n\right)$ . Moreover,if $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$ then either (1) $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ for some guard $c \in  \operatorname{grd}\left( n\right)$ or (2) $\overrightarrow{t} \in  \left( {{\Gamma }_{n}\left( {db}\right)  \ltimes  \Delta {\Psi }_{c}\left( {{db},u}\right) }\right)$ for some child $c \in  \operatorname{ng}\left( n\right)$ .

引理 4.11。如果$\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)$，那么对于某个保护式$c \in  \operatorname{grd}\left( n\right)$有$\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$。此外，如果$\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$，那么要么（1）对于某个保护式$c \in  \operatorname{grd}\left( n\right)$有$\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$，要么（2）对于某个子节点$c \in  \operatorname{ng}\left( n\right)$有$\overrightarrow{t} \in  \left( {{\Gamma }_{n}\left( {db}\right)  \ltimes  \Delta {\Psi }_{c}\left( {{db},u}\right) }\right)$。

Proof. We only show the reasoning when $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$ . The reasoning when $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)$ is similar.

证明。我们仅展示当$\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$时的推理过程。当$\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)$时的推理过程类似。

Suppose that $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$ . By definition, ${\Lambda }_{n} \mathrel{\text{:=}} {\Gamma }_{n} \boxtimes$ ${ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}$ . By definition of join tree, $\operatorname{grd}\left( n\right)$ is non-empty. If $\mathrm{{ng}}\left( n\right)$ is empty,then in particular, $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)  =$ $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ . In that case,by the first part of the lemma,we hence obtain $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ for some $c \in  \operatorname{grd}\left( n\right)$ . It remains to confirm the result when $\mathrm{{ng}}\left( n\right)$ is non-empty. Hereto,first observe that taking deltas distributes over joins as follows.

假设 $\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)$ 。根据定义，${\Lambda }_{n} \mathrel{\text{:=}} {\Gamma }_{n} \boxtimes$ ${ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}$ 。根据连接树（join tree）的定义，$\operatorname{grd}\left( n\right)$ 非空。如果 $\mathrm{{ng}}\left( n\right)$ 为空，那么特别地，$\overrightarrow{t} \in  \Delta {\Lambda }_{n}\left( {{db},u}\right)  =$ $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ 。在这种情况下，根据引理的第一部分，我们因此得到对于某个 $c \in  \operatorname{grd}\left( n\right)$ 有 $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ 。当 $\mathrm{{ng}}\left( n\right)$ 非空时，仍需确认该结果。为此，首先观察到取差分（delta）对连接（join）的分配方式如下。

$$
\Delta \left( {r\left( \bar{x}\right)  \boxtimes  s\left( \bar{y}\right) }\right) \left( {{db},u}\right)  = {\Delta r}\left( \bar{x}\right) \left( {{db},u}\right)  \boxtimes  s\left( \bar{y}\right) \left( {db}\right) 
$$

$$
 + \left( {{\Delta r}\left( \bar{x}\right) \left( {{db},u}\right)  \boxtimes  {\Delta s}\left( \bar{y}\right) \left( {{db},u}\right) }\right. 
$$

$$
 + \left( {r\left( \bar{x}\right) \left( {{db},u}\right)  \boxtimes  {\Delta s}\left( \bar{y}\right) \left( {{db},u}\right) }\right. 
$$

By application of this equality to $\Delta {\Lambda }_{n}\left( {{db},u}\right)$ ,we obtain that there are three cases possible.

将此等式应用于 $\Delta {\Lambda }_{n}\left( {{db},u}\right)$ ，我们得到有三种可能的情况。

- Case $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)  \boxtimes  \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {db}\right)$ . Then, $\overrightarrow{t} \in$ $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ since ${\Gamma }_{n}$ has the same schema as ${\Lambda }_{n}$ . By the first part of the lemma,we hence obtain $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ for some $c \in  \operatorname{grd}\left( n\right)$ .

- 情况 $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)  \boxtimes  \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {db}\right)$ 。那么，$\overrightarrow{t} \in$ $\Delta {\Gamma }_{n}\left( {{db},u}\right)$ ，因为 ${\Gamma }_{n}$ 与 ${\Lambda }_{n}$ 具有相同的模式（schema）。根据引理的第一部分，我们因此得到对于某个 $c \in  \operatorname{grd}\left( n\right)$ 有 $\overrightarrow{t} \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ 。

- The case $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)  \boxtimes  \Delta \left( {{ \boxtimes  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$ is similar.

- 情况 $\overrightarrow{t} \in  \Delta {\Gamma }_{n}\left( {{db},u}\right)  \boxtimes  \Delta \left( {{ \boxtimes  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$ 类似。

- Case $\overrightarrow{t} \in  {\Gamma }_{n}\left( {db}\right)  \boxtimes  \Delta \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$ . Then in particular, $\overrightarrow{t} \in  {\Gamma }_{n}\left( {db}\right)$ . Moreover, $\overrightarrow{t}\left\lbrack  {\mathop{\bigcup }\limits_{{c \in  \operatorname{ng}\left( n\right) }}\operatorname{pvar}\left( c\right) }\right\rbrack$ is in $\Delta \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$ . Then by application of the above distribution of delta's over joins on expression $\Delta \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$ we obtain that there is at least one $c \in  \operatorname{ng}\left( n\right)$ such that $\overrightarrow{t}\left\lbrack  {\operatorname{pvar}\left( c\right) }\right\rbrack   \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$ . Therefore, $\overrightarrow{t} \in  \operatorname{supp}\left( {{\Gamma }_{c}\left( {db}\right)  \ltimes  \Delta {\Psi }_{c}\left( {{db},u}\right) }\right)$ ,as desired.

- 情形 $\overrightarrow{t} \in  {\Gamma }_{n}\left( {db}\right)  \boxtimes  \Delta \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$。那么特别地，$\overrightarrow{t} \in  {\Gamma }_{n}\left( {db}\right)$。此外，$\overrightarrow{t}\left\lbrack  {\mathop{\bigcup }\limits_{{c \in  \operatorname{ng}\left( n\right) }}\operatorname{pvar}\left( c\right) }\right\rbrack$ 属于 $\Delta \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$。然后通过对表达式 $\Delta \left( {{ \bowtie  }_{c \in  \operatorname{ng}\left( n\right) }{\Psi }_{c}}\right) \left( {{db},u}\right)$ 应用上述 δ 在连接上的分配律，我们得到至少存在一个 $c \in  \operatorname{ng}\left( n\right)$ 使得 $\overrightarrow{t}\left\lbrack  {\operatorname{pvar}\left( c\right) }\right\rbrack   \in  \Delta {\Psi }_{c}\left( {{db},u}\right)$。因此，正如所期望的，$\overrightarrow{t} \in  \operatorname{supp}\left( {{\Gamma }_{c}\left( {db}\right)  \ltimes  \Delta {\Psi }_{c}\left( {{db},u}\right) }\right)$。

## C. PROOFS FROM SECTION 4.6

## C. 第4.6节的证明

Proposition 4.22. Under the above-mentioned hypotheses, a DCLR exists for ${CQQ}$ if,and only if, $Q$ is free-connex acyclic.

命题4.22。在上述假设下，${CQQ}$ 存在一个动态连接线性表示（DCLR）当且仅当 $Q$ 是自由连通无环的。

Crux. The if direction follows from all of our results so far. For the only if direction,assume that query $Q$ is not free-connex acyclic and suppose that a DCLR exists for query $Q$ . In particular,we can compute,for every database ${db}$ a data structure $\mathbb{D}$ that represents $Q\left( {db}\right)$ ,for every database ${db}$ . Let $U$ be the algorithm that maintains these datastructures under updates and let $\epsilon$ be the DCLR that represents the empty query result (which is obtained when $Q$ is evaluated on the empty database). Then,starting from $\epsilon ,U\left( {\epsilon ,{db}}\right)$ must construct a DCLR in time $O\left( {\parallel \epsilon \parallel  + \parallel {db}\parallel }\right)  = O\left( {\parallel {db}\parallel }\right)$ since $\epsilon$ is constant. Now enumerate $Q\left( {db}\right)$ from $U\left( {\epsilon ,{db}}\right)$ with constant delay but do not output tuple multiplicities. This enumerates $Q\left( {db}\right)$ evaluated under set semantics. Then $Q \in  \mathrm{{CD}} \circ  \mathrm{{LIN}}$ ,contradicting Brault-Baron [10].

关键思路。“如果”方向可由我们目前所有的结果推出。对于“仅当”方向，假设查询 $Q$ 不是自由连通无环的，并且假设查询 $Q$ 存在一个动态连接线性表示（DCLR）。特别地，对于每个数据库 ${db}$，我们可以计算一个表示 $Q\left( {db}\right)$ 的数据结构 $\mathbb{D}$。设 $U$ 是在更新操作下维护这些数据结构的算法，设 $\epsilon$ 是表示空查询结果（当 $Q$ 在空数据库上求值时得到）的动态连接线性表示（DCLR）。那么，从 $\epsilon ,U\left( {\epsilon ,{db}}\right)$ 开始必须在时间 $O\left( {\parallel \epsilon \parallel  + \parallel {db}\parallel }\right)  = O\left( {\parallel {db}\parallel }\right)$ 内构造出一个动态连接线性表示（DCLR），因为 $\epsilon$ 是常量。现在以恒定延迟从 $U\left( {\epsilon ,{db}}\right)$ 枚举 $Q\left( {db}\right)$，但不输出元组的重数。这是以集合语义对 $Q\left( {db}\right)$ 进行枚举。然后 $Q \in  \mathrm{{CD}} \circ  \mathrm{{LIN}}$，这与布劳尔特 - 巴伦 [10] 的结论矛盾。

Proposition 4.25. If a ${CQQ}$ is q-hierarchical,then it has a join tree which is both simple and compatible.

命题4.25。如果一个 ${CQQ}$ 是 q - 分层的，那么它有一棵既简单又兼容的连接树。

Proof. A CQ $Q$ is connected if for any two $x,y \in  \operatorname{var}\left( Q\right)$ there is a path $x = {z}_{0},\ldots ,{z}_{l} = y$ such that for each $j < l$ there is an atom $\mathbf{a}$ in $Q$ such that $\left\{  {{z}_{j},{z}_{j + 1}}\right\}   \subseteq  \operatorname{var}\left( \mathbf{a}\right)$ . It is a standard observation that every $\mathrm{{CQ}}$ can be written as a join ${Q}_{1} \boxtimes  \cdots  \boxtimes  {Q}_{k}$ of connected CQs with pairwise disjoint sets of output variables. Call these ${Q}_{i}$ the connected components of $Q$ . Berkholz et al. show that $\mathrm{{CQ}}Q$ is hierarchical if and only if every connected component ${Q}_{i}$ of $Q$ has a $q$ -tree, which is defined as follows.

证明。如果对于任意两个$x,y \in  \operatorname{var}\left( Q\right)$，存在一条路径$x = {z}_{0},\ldots ,{z}_{l} = y$，使得对于每个$j < l$，在$Q$中存在一个原子$\mathbf{a}$，满足$\left\{  {{z}_{j},{z}_{j + 1}}\right\}   \subseteq  \operatorname{var}\left( \mathbf{a}\right)$，则合取查询（Conjunctive Query，CQ）$Q$是连通的。一个标准的结论是，每个$\mathrm{{CQ}}$都可以写成具有两两不相交的输出变量集的连通合取查询的连接${Q}_{1} \boxtimes  \cdots  \boxtimes  {Q}_{k}$。将这些${Q}_{i}$称为$Q$的连通分量。伯克霍尔茨（Berkholz）等人表明，当且仅当$Q$的每个连通分量${Q}_{i}$都有一个$q$ - 树时，$\mathrm{{CQ}}Q$是分层的，$q$ - 树的定义如下。

Definition C.1. Let ${Q}_{i}$ be a connected ${CQ}$ . A $q$ -tree for ${Q}_{i}$ is a rooted directed tree ${F}_{{Q}_{i}} = \left( {V,E}\right)$ with $V = \operatorname{var}\left( Q\right)$ s.t. (1) for all atoms $\mathbf{a}$ in ${Q}_{i}$ the set $\operatorname{var}\left( \mathbf{a}\right)$ forms a directed path in ${F}_{Q}$ starting at the root,and (2) if $\operatorname{out}\left( {Q}_{i}\right)  \neq  \varnothing$ ,then $\operatorname{out}\left( {Q}_{i}\right)$ is a connected subset in ${F}_{{Q}_{i}}$ containing the root.

定义C.1。设${Q}_{i}$是一个连通的${CQ}$。${Q}_{i}$的$q$ - 树是一棵有根的有向树${F}_{{Q}_{i}} = \left( {V,E}\right)$，其中$V = \operatorname{var}\left( Q\right)$，使得（1）对于${Q}_{i}$中的所有原子$\mathbf{a}$，集合$\operatorname{var}\left( \mathbf{a}\right)$在${F}_{Q}$中形成一条从根节点开始的有向路径；（2）如果$\operatorname{out}\left( {Q}_{i}\right)  \neq  \varnothing$，那么$\operatorname{out}\left( {Q}_{i}\right)$是${F}_{{Q}_{i}}$中包含根节点的一个连通子集。

<!-- Media -->

<!-- figureText: $\left( {F}_{Q}\right)$ (T) 0 $\left\lbrack  x\right\rbrack$ $\left\lbrack  {x,y}\right\rbrack$ $\left\lbrack  {x,y,u}\right\rbrack$ $\left\lbrack  {x,y,w}\right\rbrack$ $\left\lbrack  {x,y,u,v}\right\rbrack  R\left( {x,y,w}\right)$ $S\left( {x,y,u,v}\right)$ $w$ $E\left( {x,y}\right)$ -->

<img src="https://cdn.noedgeai.com/0195cc9d-20ab-7231-961a-fb75922b669d_14.jpg?x=249&y=153&w=511&h=387&r=0"/>

Figure 7: Illustration of the proof of Proposition 4.25

图7：命题4.25证明的图示

<!-- Media -->

To show the proposition,assume that $\mathrm{{CQ}}Q$ is hierarchical. From the $q$ -trees for the connected components of $Q$ we can construct a simple join tree $T$ for $Q$ that is compatible with $Q$ ,as follows. For ease of exposition,let us assume that $Q$ has a single connected component; the general case is similar. Let ${F}_{Q}$ be the $q$ -tree for $Q$ . Then,for every node $x$ in ${F}_{Q}$ ,define $p\left( x\right)$ to be set of variables that occur in the unique path from $x$ to the root in ${F}_{Q}$ . In particular, $p\left( x\right)$ contains $x$ . By definition of $q$ -trees, $p\left( x\right)$ must be a partial hyperedge of $A$ ,the set of atoms in $Q$ . Construct $T$ as follows. Initially, $T$ contains only the empty hyperedge $\varnothing$ . For all variables $x \in  \operatorname{var}\left( Q\right)$ ,add hyperedge $p\left( x\right)$ to $T$ . For every edge $x \rightarrow  y$ in ${F}_{Q}$ ,add an edge $p\left( x\right)  \rightarrow  p\left( y\right)$ to $T$ . If $x$ is the root in ${F}_{Q}$ ,then also add the edge $p\left( x\right)  \rightarrow  \varnothing$ to the root $\varnothing$ in $T$ . Next,add all atoms of $Q$ to $T$ ,and for each atom $\mathbf{a}$ ,add an edge from $\mathbf{a}$ to the hyperedge $h$ in $T$ with $h = \operatorname{var}\left( \mathbf{a}\right)$ . (This hyperedge has been generated by $p\left( x\right)$ with $x$ the variable in $\operatorname{var}\left( \mathbf{a}\right)$ that is the lowest among all variables of $\operatorname{var}\left( \mathbf{a}\right)$ in $\left. {F}_{Q}\right)$ . Figure 7 illustrates this construction for $Q = {\pi }_{x,y,u}\left( {E\left( {x,y}\right)  \boxtimes  R\left( {x,y,w}\right)  \boxtimes  S\left( {x,y,u,v}\right) }\right)$ and the $q$ -tree ${F}_{Q}$ shown in Figure 7. Note that in this example, $T$ is indeed a simple generalized join tree. It can be shown that this is always the case. It remains to show that $T$ is compatible with $Q$ . To do so,observe that,by definition of $q$ -tree, $\operatorname{out}\left( Q\right)$ is a connected subset of ${F}_{Q}$ that contains the root. Then $N = \{ \varnothing \}  \cup  \{ p\left( x\right)  \mid  x \in  \operatorname{out}\left( Q\right) \}$ must be a connex subset of $T$ with $\operatorname{var}\left( N\right)  = \operatorname{out}\left( Q\right)$ ,as desired.

为了证明该命题，假设$\mathrm{{CQ}}Q$是分层的（hierarchical）。从$Q$的连通分量的$q$ - 树，我们可以为$Q$构造一个与$Q$兼容的简单连接树$T$，具体如下。为便于阐述，我们假设$Q$只有一个连通分量；一般情况类似。设${F}_{Q}$为$Q$的$q$ - 树。那么，对于${F}_{Q}$中的每个节点$x$，定义$p\left( x\right)$为在${F}_{Q}$中从$x$到根节点的唯一路径上出现的变量集合。特别地，$p\left( x\right)$包含$x$。根据$q$ - 树的定义，$p\left( x\right)$必须是$A$（$Q$中的原子集合）的一个部分超边（partial hyperedge）。按如下方式构造$T$。初始时，$T$仅包含空超边$\varnothing$。对于所有变量$x \in  \operatorname{var}\left( Q\right)$，将超边$p\left( x\right)$添加到$T$中。对于${F}_{Q}$中的每条边$x \rightarrow  y$，将边$p\left( x\right)  \rightarrow  p\left( y\right)$添加到$T$中。如果$x$是${F}_{Q}$中的根节点，那么还将边$p\left( x\right)  \rightarrow  \varnothing$添加到$T$的根节点$\varnothing$。接下来，将$Q$的所有原子添加到$T$中，并且对于每个原子$\mathbf{a}$，从$\mathbf{a}$向$T$中满足$h = \operatorname{var}\left( \mathbf{a}\right)$的超边$h$添加一条边。（这个超边是由$p\left( x\right)$生成的，其中$x$是$\operatorname{var}\left( \mathbf{a}\right)$中的变量，且在$\left. {F}_{Q}\right)$中是$\operatorname{var}\left( \mathbf{a}\right)$所有变量中层次最低的。图7展示了针对$Q = {\pi }_{x,y,u}\left( {E\left( {x,y}\right)  \boxtimes  R\left( {x,y,w}\right)  \boxtimes  S\left( {x,y,u,v}\right) }\right)$和图7中所示的$q$ - 树${F}_{Q}$的这种构造过程。注意，在这个例子中，$T$确实是一个简单的广义连接树。可以证明情况总是如此。接下来需要证明$T$与$Q$兼容。为此，观察可知，根据$q$ - 树的定义，$\operatorname{out}\left( Q\right)$是${F}_{Q}$的一个包含根节点的连通子集。那么$N = \{ \varnothing \}  \cup  \{ p\left( x\right)  \mid  x \in  \operatorname{out}\left( Q\right) \}$必定是$T$的一个满足$\operatorname{var}\left( N\right)  = \operatorname{out}\left( Q\right)$的连通子集，符合要求。

Proposition 4.26. If a ${CQQ}$ has a join tree $T$ which is both simple and compatible with $Q$ ,then $Q$ is $q$ -hierarchical.

命题4.26。如果一个${CQQ}$有一个既简单又与$Q$兼容的连接树$T$，那么$Q$是$q$ - 分层的（$q$ - hierarchical）。

Proof. Assume that $T$ is a simple join tree for $Q$ that is also compatible with $Q$ . We first show that $Q$ is hiearchical. Let $x$ and $y$ be two variables in $Q$ . If ${at}\left( x\right)  \cap  {at}\left( y\right)  = \varnothing$ we are done. Hence,assume ${at}\left( x\right)  \cap  {at}\left( y\right)  \neq  \varnothing$ . Let $c \in  {at}\left( x\right)  \cap  {at}\left( y\right)$ . We need to show that either ${at}\left( x\right)  \subseteq  {at}\left( y\right)$ or ${at}\left( y\right)  \subseteq  {at}\left( x\right)$ . Assume for the purpose of contradiction that neither holds. Then there exists $\mathbf{a} \in  {at}\left( x\right)  \smallsetminus  {at}\left( y\right)$ and similarly an atom $\mathbf{b} \in$ ${at}\left( y\right)  \smallsetminus  {at}\left( x\right)$ . Since $T$ is a join tree,and since $x$ occurs both in $\mathbf{a}$ and $\mathbf{c}$ ,we know that $x$ must occur in every node on the unique undirected path between $\mathbf{a}$ and $\mathbf{c}$ in $T$ . In particular, let $n$ be the least common ancestor of $\mathbf{a}$ and $\mathbf{c}$ . Then $x \in$ $\operatorname{var}\left( n\right)$ . Similarly, $y$ must occur in every node on the unique undirected path between $\mathbf{b}$ and $\mathbf{c}$ in $T$ . In particular,let $m$ be the least common ancestor of $\mathbf{b}$ and $\mathbf{c}$ . Then $y \in  \operatorname{var}\left( m\right)$ . Now there are two possibilities. Either (1) $n$ is an ancestor of $m$ . But then,since $T$ is simple, $x \in  \operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( m\right)$ . Since $\mathbf{b}$ is a descendant of $m$ then by simplicity of $T$ hence $x \in$ $\operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( m\right)  \subseteq  \operatorname{var}\left( \mathbf{b}\right)$ . This contradicts our assumption that $\mathbf{b} \in  {at}\left( y\right)  \smallsetminus  {at}\left( x\right)$ . Otherwise,(2) $m$ is an ancestor of $m$ and we similarly obtain a contradiction to our assumption that $a \in  {at}\left( x\right)  \smallsetminus  {at}\left( y\right)$ .

证明。假设$T$是$Q$的一个简单连接树（simple join tree），且与$Q$兼容。我们首先证明$Q$是分层的（hiearchical）。设$x$和$y$是$Q$中的两个变量。如果${at}\left( x\right)  \cap  {at}\left( y\right)  = \varnothing$，则证明完成。因此，假设${at}\left( x\right)  \cap  {at}\left( y\right)  \neq  \varnothing$。设$c \in  {at}\left( x\right)  \cap  {at}\left( y\right)$。我们需要证明要么${at}\left( x\right)  \subseteq  {at}\left( y\right)$，要么${at}\left( y\right)  \subseteq  {at}\left( x\right)$。为了推出矛盾，假设两者都不成立。那么存在$\mathbf{a} \in  {at}\left( x\right)  \smallsetminus  {at}\left( y\right)$，类似地，存在一个原子$\mathbf{b} \in$${at}\left( y\right)  \smallsetminus  {at}\left( x\right)$。由于$T$是一个连接树，并且由于$x$同时出现在$\mathbf{a}$和$\mathbf{c}$中，我们知道$x$必须出现在$T$中$\mathbf{a}$和$\mathbf{c}$之间唯一无向路径上的每个节点中。特别地，设$n$是$\mathbf{a}$和$\mathbf{c}$的最近公共祖先（least common ancestor）。那么$x \in$$\operatorname{var}\left( n\right)$。类似地，$y$必须出现在$T$中$\mathbf{b}$和$\mathbf{c}$之间唯一无向路径上的每个节点中。特别地，设$m$是$\mathbf{b}$和$\mathbf{c}$的最近公共祖先。那么$y \in  \operatorname{var}\left( m\right)$。现在有两种可能性。要么（1）$n$是$m$的祖先。但由于$T$是简单的，$x \in  \operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( m\right)$。由于$\mathbf{b}$是$m$的后代，根据$T$的简单性，因此$x \in$$\operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( m\right)  \subseteq  \operatorname{var}\left( \mathbf{b}\right)$。这与我们假设的$\mathbf{b} \in  {at}\left( y\right)  \smallsetminus  {at}\left( x\right)$相矛盾。否则，（2）$m$是$m$的祖先，我们类似地得到与我们假设的$a \in  {at}\left( x\right)  \smallsetminus  {at}\left( y\right)$相矛盾的结果。

It remains to show $q$ -hierachicality. Hereto,assume that ${at}\left( x\right)  \varsubsetneq  {at}\left( y\right)$ and $x \in  \operatorname{out}\left( Q\right)$ . We need to show that $y \in$ out(Q). Let $\mathbf{a} \in  {at}\left( x\right)$ and let $\mathbf{b} \in  {at}\left( y\right)  \smallsetminus  {at}\left( x\right)$ . In particular, $\mathbf{a}$ contains both $x$ and $y$ ,while $\mathbf{b}$ contains only $y$ . From compatibility of $T$ with $Q$ ,it follows that there is a connex subset $N$ of $T$ such that $\operatorname{var}\left( N\right)  = \operatorname{out}\left( Q\right)$ . Let $n$ be the lowest ancestor node of $\mathbf{a}$ in $N$ that contains $x$ . Because $T$ is simple,all descendants of $n$ hence also have $x$ . As a consequence, $\mathbf{b}$ cannot be a descendant of $n$ . Since $\mathbf{a}$ and $\mathbf{b}$ share $y$ ,this implies that the unique undirected path between $\mathbf{a}$ and $\mathbf{b}$ must pass through $n$ . Because all nodes on this path must share all variables in common between $\mathbf{a}$ and $\mathbf{b}$ ,it follows that $y \in  \operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( N\right)  = \operatorname{out}\left( Q\right)$ .

接下来需要证明 $q$ -层次性（$q$ -hierachicality）。为此，假设 ${at}\left( x\right)  \varsubsetneq  {at}\left( y\right)$ 且 $x \in  \operatorname{out}\left( Q\right)$。我们需要证明 $y \in$ 属于（out(Q)）。设 $\mathbf{a} \in  {at}\left( x\right)$ 并设 $\mathbf{b} \in  {at}\left( y\right)  \smallsetminus  {at}\left( x\right)$。特别地，$\mathbf{a}$ 同时包含 $x$ 和 $y$，而 $\mathbf{b}$ 仅包含 $y$。由 $T$ 与 $Q$ 的相容性可知，存在 $T$ 的一个连通子集 $N$ 使得 $\operatorname{var}\left( N\right)  = \operatorname{out}\left( Q\right)$。设 $n$ 为 $N$ 中包含 $x$ 的 $\mathbf{a}$ 的最低祖先节点。因为 $T$ 是简单的，所以 $n$ 的所有后代节点也都包含 $x$。因此，$\mathbf{b}$ 不可能是 $n$ 的后代节点。由于 $\mathbf{a}$ 和 $\mathbf{b}$ 共享 $y$，这意味着 $\mathbf{a}$ 和 $\mathbf{b}$ 之间的唯一无向路径必须经过 $n$。因为这条路径上的所有节点必须共享 $\mathbf{a}$ 和 $\mathbf{b}$ 之间的所有公共变量，所以可得 $y \in  \operatorname{var}\left( n\right)  \subseteq  \operatorname{var}\left( N\right)  = \operatorname{out}\left( Q\right)$。

Proposition 4.27. Let $Q$ be the hierarchical join query $R\left( {x,y,z}\right)  \boxtimes  S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right)$ . Every simple width-one GHD for $Q$ has at least one partial hyperedge.

命题 4.27。设 $Q$ 为层次连接查询（hierarchical join query）$R\left( {x,y,z}\right)  \boxtimes  S\left( {x,y,u}\right)  \boxtimes  T\left( {y,v,w}\right)  \boxtimes  U\left( {y,v,p}\right)$。$Q$ 的每个简单宽度为 1 的广义超树分解（GHD，Generalized Hypergraph Decomposition）至少有一个部分超边。

Proof. Let $T$ be a simple width-one GHD for $Q$ and assume,for the purpose of contradiction that $T$ contains only atoms and full hyperedges. $T$ ’s nodes are hence elements of $\{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) ,\left\lbrack  {x,y,z}\right\rbrack  ,\left\lbrack  {x,y,u}\right\rbrack  ,\lbrack y,$ $v,w\rbrack ,\left\lbrack  {y,v,p}\right\rbrack  \}$ . Partition this set into

证明。设 $T$ 为 $Q$ 的一个简单宽度为 1 的广义超树分解，并为了推出矛盾，假设 $T$ 仅包含原子和完全超边。因此，$T$ 的节点是 $\{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) ,\left\lbrack  {x,y,z}\right\rbrack  ,\left\lbrack  {x,y,u}\right\rbrack  ,\lbrack y,$ $v,w\rbrack ,\left\lbrack  {y,v,p}\right\rbrack  \}$ 的元素。将这个集合划分为

$$
{XY} = \{ R\left( {x,y,z}\right) ,S\left( {x,y,u}\right) ,\left\lbrack  {x,y,z}\right\rbrack  ,\left\lbrack  {x,y,u}\right\rbrack  \} 
$$

$$
{YV} = \{ T\left( {y,v,w}\right) ,U\left( {y,v,p}\right) ,\left\lbrack  {y,v,w}\right\rbrack  ,\left\lbrack  {y,v,p}\right\rbrack  \} .
$$

Now consider the unique undirected path $m,{n}_{1},{n}_{2},\ldots ,{n}_{k},p$ between $m = R\left( {x,y,z}\right)$ and $p = T\left( {y,v,w}\right)$ . There are two possibilities: either this undirected path shows that some node in ${XY}$ is a parent of a node in ${YV}$ ,or it shows that some node in ${YV}$ is a parent of some node in ${XY}$ . In either case,hierarchicality is violated since nodes in ${XY}$ all have variable $x$ while nodes in ${YV}$ don’t and,conversely,nodes in ${YV}$ all have variable $v$ while nodes in ${XY}$ don’t.

现在考虑 $m = R\left( {x,y,z}\right)$ 和 $p = T\left( {y,v,w}\right)$ 之间的唯一无向路径 $m,{n}_{1},{n}_{2},\ldots ,{n}_{k},p$。有两种可能性：要么这条无向路径表明 ${XY}$ 中的某个节点是 ${YV}$ 中某个节点的父节点，要么它表明 ${YV}$ 中的某个节点是 ${XY}$ 中某个节点的父节点。在这两种情况下，层次性都被违反了，因为 ${XY}$ 中的节点都有变量 $x$，而 ${YV}$ 中的节点没有；反之，${YV}$ 中的节点都有变量 $v$，而 ${XY}$ 中的节点没有。

## D. QUERIES

## D. 查询

Full join queries

完全连接查询

FQ1

SELECT * FROM orders o, lineitem 1, part p , partsupp ps

从订单表 o、行项目表 1、零件表 p、零件供应商表 ps 中选择所有列

WHERE O.orderkey = 1.orderkey, AND 1.partkey = p.partkey

条件是 O.订单键 = 1.订单键，并且 1.零件键 = p.零件键

AND l.partkey $=$ ps.partkey AND l.suppkey $=$ ps.suppkey

并且 l.零件键 $=$ ps.零件键 并且 l.供应商键 $=$ ps.供应商键

F02

SELECT * FROM lineitem 1,orders,customer c,part $\mathrm{p}$ ,nation $\mathrm{n}$

从表lineitem 1、orders、customer c、part $\mathrm{p}$ 、nation $\mathrm{n}$ 中选择所有列

WHERE 1.orderkey = o.orderkey AND o.custkey = c.custkey

其中1.orderkey等于o.orderkey，并且o.custkey等于c.custkey

AND l.partkey $=$ p.partkey AND c.nationkey $=$ n.nationkey

并且l.partkey $=$ p.partkey，并且c.nationkey $=$ n.nationkey

FQ3

SELECT * FROM orders o, lineitem 1,

从表orders o、lineitem 1中选择所有列

partsupp ps, supplier s, customer c

partsupp ps、supplier s、customer c

WHERE o.orderkey $= 1$ .orderkey AND

其中o.orderkey $= 1$ .orderkey，并且

1.suppkey = ps.suppkey AND

1.suppkey等于ps.suppkey，并且

1.suppkey = s.suppkey AND o.custkey = c.custkey

1.suppkey等于s.suppkey，并且o.custkey等于c.custkey

FQ4

SELECT * FROM lineitem 1, supplier s, partsupp ps

从表lineitem 1、supplier s、partsupp ps中选择所有列

WHERE 1.suppkey $=$ s.suppkey

其中1.suppkey $=$ s.suppkey

AND 1.suppkey $=$ ps.suppkey

并且1.suppkey $=$ ps.suppkey

FQ5

SELECT * SELECT * FROM date_dim dd, store_sales ss, item i

从表date_dim dd、store_sales ss、item i中选择所有列

WHERE ss.s_item_sk = i.i_item_sk

其中ss.s_item_sk等于i.i_item_sk

AND ss.s_date_sk = dd.d_date_sk

并且ss.s_date_sk等于dd.d_date_sk

Q1

SELECT 1_returnflag, 1_linestatus, SUM(1_quantity)

选择1_returnflag、1_linestatus，对1_quantity求和

AS sum_qty, SUM(1_extendedprice) AS sum_base_price,

作为sum_qty，对1_extendedprice求和作为sum_base_price

SUM(1_extendedprice * (1 - 1_discount)) AS sum_disc_price,

1_扩展价格(1_extendedprice) * (1 - 1_折扣(1_discount))的总和作为折扣后总价(sum_disc_price),

SUM(1_extendedprice * (1 - 1_discount) * (1 + 1_tax)) AS

1_扩展价格(1_extendedprice) * (1 - 1_折扣(1_discount)) * (1 + 1_税(1_tax))的总和作为

sum_charge, AVG(1_quantity) AS AVG_qty, AVG(1_extendedprice)

总费用(sum_charge)，1_数量(1_quantity)的平均值作为平均数量(AVG_qty)，1_扩展价格(1_extendedprice)的平均值

AS AVG_price, AVG(1_discount) AS AVG_disc, count(*)

作为平均价格(AVG_price)，1_折扣(1_discount)的平均值作为平均折扣(AVG_disc)，记录数量

AS count_order

作为订单数量(count_order)

FROM lineitem WHERE

从行项目(lineitem)表中选取，条件为

l_shipdate <= date '1998-12-01' - interval '108' day

1_发货日期(l_shipdate) <= 日期 '1998 - 12 - 01' - 间隔 '108' 天

group by 1_returnflag,1_linestatus

按1_退货标志(1_returnflag)，1_行状态(1_linestatus)分组

Q3

SELECT 1_orderkey,SUM(1_extendedprice * (1 - 1_discount))

选择1_订单键(1_orderkey)，1_扩展价格(1_extendedprice) * (1 - 1_折扣(1_discount))的总和

AS revenue,o_orderdate,o_shippriority

作为收入(revenue)，订单日期(o_orderdate)，发货优先级(o_shippriority)

FROM customer,orders,lineitem WHERE c_mktsegment = 'AUTOMOBILE'

从客户(customer)表、订单(orders)表、行项目(lineitem)表中选取，条件为客户市场细分(c_mktsegment) = '汽车(AUTOMOBILE)'

AND c_custkey $= o$ _custkey AND l_orderkey $= o$ _orderkey

并且客户键(c_custkey) $= o$ 客户键(_custkey) 并且行项目订单键(l_orderkey) $= o$ 订单键(_orderkey)

AND O_orderdate < date '1995-03-13' AND

并且订单日期(O_orderdate) < 日期 '1995 - 03 - 13' 并且

l_shipdate > date '1995-03-13'

行项目发货日期(l_shipdate) > 日期 '1995 - 03 - 13'

group by 1 orderkey,o_orderdate,o_shippriority

按1_订单键(orderkey)，订单日期(o_orderdate)，发货优先级(o_shippriority)分组

Q4

SELECT o_orderpriority, count(*) AS order_count

选择订单优先级(o_orderpriority)，记录数量作为订单数量(order_count)

FROM orders WHERE o_orderdate $>  =$ date ’1995-01-01’ AND

从订单表中选取，条件为订单日期 $>  =$ 早于日期 '1995-01-01' 并且

o_orderdate < date '1995-01-01' + interval '3' month

订单日期小于日期 '1995-01-01' 加上间隔 '3' 个月

AND exists ( SELECT * FROM lineitem WHERE

并且存在（从行项目表中选取所有记录，条件为

l_orderkey = o_orderkey AND 1_commitdate < l_receiptdate)

行项目订单键等于订单键且交货承诺日期小于收货日期）

group by o_orderpriority

按订单优先级分组

Q6

SELECT SUM ( 1_extendedprice * 1_discount ) AS revenue

选择 1_扩展价格 * 1_折扣的总和作为收入

FROM lineitem

从行项目表中选取

WHERE 1_shipdate >= date '1994-01-01' AND

其中 1_发货日期大于等于日期 '1994-01-01' 并且

l_shipdate < date '1994-01-01' + interval '1' year

1_发货日期小于日期 '1994-01-01' 加上间隔 '1' 年

AND 1_discount between 0.06 - 0.01 AND

并且 1_折扣在 0.06 - 0.01 到

0.06 + 0.01 AND 1_quantity < 24;

0.06 + 0.01 之间，并且 1_数量小于 24;

Q9

SELECT nation,o_year,SUM(amount) AS sum_profit

选择国家、订单年份、金额总和作为总利润

FROM ( SELECT n_name

从 ( 选择 n_名称

AS nation,extract (year FROM o_orderdate) AS o_year,

作为国家，从 o_订单日期中提取年份作为订单年份，

1_extendedprice * (1 - 1_discount) - ps_supplycost * 1_quantity

1_扩展价格 * (1 - 1_折扣) - 零件供应成本 * 1_数量

AS amount

作为金额

FROM part,supplier,lineitem,partsupp,

从零件表、供应商表、行项目表、零件供应表、

orders,nation WHERE s_suppkey $= 1$ _suppkey

订单表、国家表中选取，其中 s_供应商键 $= 1$ _供应商键

AND ps_suppkey $= 1$ _suppkey AND

并且 ps_供应商键 $= 1$ _供应商键 并且

ps_partkey = 1_partkey AND p_partkey = 1_partkey

ps_零件键 = 1_零件键 并且 p_零件键 = 1_零件键

AND o_orderkey = l_orderkey

并且 o_订单键 = 1_订单键

AND s_nationkey = n_nationkey

并且 s_nationkey 等于 n_nationkey

AND p_name like '%dim%') AS profit

并且 p_name 类似于 '%dim%') 作为利润

group by nation,o_year

按国家、年份分组

Q12

SELECT 1_shipmode, SUM(case when o_orderpriority = '1-URGENT'

选择 1_shipmode，对（当 o_orderpriority 等于 '1-紧急'

or o_orderpriority = '2-HIGH'

或者 o_orderpriority 等于 '2-高'

then 1 else 0 end) AS high_line_count,

则为 1 否则为 0 的情况）求和作为高优先级行数

SUM(case when o_orderpriority <> '1-URGENT'

对（当 o_orderpriority 不等于 '1-紧急'

AND o_orderpriority <> '2-HIGH' then 1 else 0 end)

并且 o_orderpriority 不等于 '2-高' 则为 1 否则为 0 的情况）求和

AS low_line_count

作为低优先级行数

FROM orders,lineitem

从订单表、行项目表中选取

WHERE o_orderkey $= 1$ _orderkey AND

其中 o_orderkey $= 1$ _orderkey 并且

l_shipmode in ('RAIL', 'FOB') AND

l_shipmode 在 ('铁路', '离岸价') 之中 并且

l_commitdate < l_receiptdate AND

l_commitdate 小于 l_receiptdate 并且

l_shipdate < l_commitdate AND

l_shipdate 小于 l_commitdate 并且

l_receiptdate >= date '1997-01-01' AND

l_receiptdate 大于等于日期 '1997-01-01' 并且

l_receiptdate < date '1997-01-01' + interval '1' year

l_receiptdate 小于日期 '1997-01-01' 加上间隔 '1' 年

group by l_shipmode

按l_shipmode分组

Q13’

SELECT c_count, COUNT(*) AS custdist

选择c_count，将COUNT(*) 作为custdist

FROM (   )

从 (   )

SELECT c.custkey AS c_custkey, COUNT(o.orderkey) AS c_count

选择c.custkey 作为c_custkey，将COUNT(o.orderkey) 作为c_count

FROM customer c, orders o

从customer表（c），orders表（o）

WHERE c.custkey $=$ o.custkey

其中c.custkey $=$ o.custkey

AND (o.comment NOT LIKE '%special%requests%')

并且 (o.comment 不包含 '%特殊%请求%')

group by c.custkey) c_orders

按c.custkey分组) c_orders

group by c_count;

按c_count分组;

Q16’

SELECT p_brand, p_type, p_size, count (distinct ps_suppkey) as supplier_cnt

选择p_品牌，p_类型，p_尺寸，将count (不同的ps_供应商键) 作为供应商数量

FROM partsupp, part

从partsupp表，part表

WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#34'

其中p_零件键 = ps_零件键 并且 p_品牌 <> '品牌#34'

AND p_type not like 'LARGE BRUSHED%'

并且 p_类型 不包含 '大型刷式%'

AND p_size in(48,19,12,4,41,7,21,39)

并且 p_尺寸 在(48,19,12,4,41,7,21,39)范围内

AND ps_suppkey in ( SELECT s_suppkey FROM supplier

并且 ps_供应商键 在 ( 选择s_供应商键 从supplier表

WHERE s_comment not like '%Customer%Complaints%' )

其中s_备注 不包含 '%客户%投诉%' )

GROUP BY p_brand, p_type, p_size

按p_brand（品牌）、p_type（类型）、p_size（尺寸）分组

Q18

SELECT c_name, c_custkey, o_orderkey, o_orderdate,

选择c_name（客户名称）、c_custkey（客户键）、o_orderkey（订单键）、o_orderdate（订单日期）

o_totalprice,SUM(l_quantity)

o_totalprice（订单总价）、l_quantity（行项目数量）的总和

FROM customer,orders,lineitem

从customer（客户表）、orders（订单表）、lineitem（行项目表）中选取

WHERE o_orderkey in (SELECT l_orderkey

条件是o_orderkey（订单键）在（选择l_orderkey（行项目订单键）

FROM lineitem

从lineitem（行项目表）中

group by 1 orderkey having SUM(1_quantity) > 314)

按1_orderkey（此处可能有误，推测为l_orderkey）分组，且l_quantity（行项目数量）的总和大于314）

AND c_custkey $= o$ _custkey AND o_orderkey $= 1$ _orderkey

并且c_custkey（客户键） $= o$ _custkey 以及o_orderkey（订单键） $= 1$ _orderkey

group by

分组依据

c_name,c_custkey,o_orderkey,o_orderdate,o_totalprice

c_name（客户名称）、c_custkey（客户键）、o_orderkey（订单键）、o_orderdate（订单日期）、o_totalprice（订单总价）

## TPC-DS

## TPC - DS（事务处理性能委员会决策支持基准测试）

Q3 SELECT dt.d_year, i.i_brand_id, i.i_brand , SUM(ss.ss_ext_sales_price) AS sum_agg FROM date_dim dt, store_sales ss, item i WHERE dt.d_date_sk = ss.ss_sold_date_sk AND ss.ss_item_sk = i.i_item_sk AND dt.d_moy = 12 AND i.i_manufact_id = 436 group by dt.d_year, i.i_brand , i.i_brand_id; Q7 SELECT i.i_item_id, AVG(ss.ss_quantity) AS agg1, AVG(ss.ss_list_price) AS agg2, AVG(ss.ss_coupon_amt) AS agg3, AVG(ss.ss_sales_price) AS agg4 FROM store_sales ss, customer_demographics cd, date_dim d, item i, promotion p WHERE ss.ss_item_sk = i.i_item_sk AND ss.ss_sold_date_sk = d.d_date_sk AND ss.ss_cdemo_sk $=$ cd.cd_demo_sk AND ss.ss_promo_sk $=$ p.p_promo_sk AND cd.cd_gender = 'F' AND cd.cd_marital_status = 'W' AND (p.p_channel_email = 'N' OR p.p_channel_event = 'N') AND d.d_year = 1998 group by i.i_item_id; Q19 SELECT i.i_brand_id, i.i_brand , i.i_manufact_id, i.i_manufact, SUM(ss.ss_ext_sales_price) AS ext_price FROM date_dim dd, store_sales ss, item i, customer c, customer_address ca, store s WHERE dd.d_date_sk = ss.ss_sold_date_sk AND ss.ss_item_sk = i.i_item_sk AND i.i_manager_id = 7 AND dd.d_moy = 11 AND dd.d_year = 1999 AND ss.ss_customer_sk = c.c_customer_sk AND c.c_current_addr_sk $=$ ca.ca_address_sk AND ss.ss_store_sk = s.s_store_sk group by i.i_br , i.i_br, _id, i.i_manufact_id, i.i_manufact; Q22 SELECT i.i_product_name, i.i_brand , i.i_class, i.i_category, SUM(inv.inv_quantity_on_hand) AS qoh FROM date_dim dd, inventory inv, item i, warehouse wh WHERE dd.d_date_sk = inv.inv_date_sk AND inv.inv_item_sk = i.i_item_sk AND inv.inv_warehouse_sk = wh.w_warehouse_sk AND dd.d_month_seq between 1193 AND 1204 group by i.i_product_name, i.i_brand , i.i_clASs, i.i_category;

Q3 选择dt.d_year（日期维度的年份）、i.i_brand_id（商品品牌ID）、i.i_brand（商品品牌），将ss.ss_ext_sales_price（店铺销售的扩展销售价格）的总和作为sum_agg（汇总值）。从date_dim（日期维度表）dt、store_sales（店铺销售表）ss、item（商品表）i中选取，条件是dt.d_date_sk（日期维度的日期键）等于ss.ss_sold_date_sk（店铺销售的销售日期键），且ss.ss_item_sk（店铺销售的商品键）等于i.i_item_sk（商品的商品键），并且dt.d_moy（日期维度的月份）为12，且i.i_manufact_id（商品的制造商ID）为436，按dt.d_year（日期维度的年份）、i.i_brand（商品品牌）、i.i_brand_id（商品品牌ID）分组；Q7 选择i.i_item_id（商品ID），将ss.ss_quantity（店铺销售的数量）的平均值作为agg1（汇总值1），将ss.ss_list_price（店铺销售的标价）的平均值作为agg2（汇总值2），将ss.ss_coupon_amt（店铺销售的优惠券金额）的平均值作为agg3（汇总值3），将ss.ss_sales_price（店铺销售的销售价格）的平均值作为agg4（汇总值4）。从store_sales（店铺销售表）ss、customer_demographics（客户人口统计信息表）cd、date_dim（日期维度表）d、item（商品表）i、promotion（促销表）p中选取，条件是ss.ss_item_sk（店铺销售的商品键）等于i.i_item_sk（商品的商品键），且ss.ss_sold_date_sk（店铺销售的销售日期键）等于d.d_date_sk（日期维度的日期键），且ss.ss_cdemo_sk $=$ cd.cd_demo_sk（店铺销售的客户人口统计信息键 $=$ 客户人口统计信息表的人口统计信息键），且ss.ss_promo_sk $=$ p.p_promo_sk（店铺销售的促销键 $=$ 促销表的促销键），并且cd.cd_gender（客户人口统计信息的性别）为'F'（女性），且cd.cd_marital_status（客户人口统计信息的婚姻状况）为'W'（丧偶），并且（p.p_channel_email（促销的电子邮件渠道）为'N'（否）或者p.p_channel_event（促销的活动渠道）为'N'（否）），且d.d_year（日期维度的年份）为1998，按i.i_item_id（商品ID）分组；Q19 选择i.i_brand_id（商品品牌ID）、i.i_brand（商品品牌）、i.i_manufact_id（商品的制造商ID）、i.i_manufact（商品的制造商），将ss.ss_ext_sales_price（店铺销售的扩展销售价格）的总和作为ext_price（扩展价格）。从date_dim（日期维度表）dd、store_sales（店铺销售表）ss、item（商品表）i、customer（客户表）c、customer_address（客户地址表）ca、store（店铺表）s中选取，条件是dd.d_date_sk（日期维度的日期键）等于ss.ss_sold_date_sk（店铺销售的销售日期键），且ss.ss_item_sk（店铺销售的商品键）等于i.i_item_sk（商品的商品键），且i.i_manager_id（商品的经理ID）为7，且dd.d_moy（日期维度的月份）为11，且dd.d_year（日期维度的年份）为1999，且ss.ss_customer_sk（店铺销售的客户键）等于c.c_customer_sk（客户表的客户键），且c.c_current_addr_sk $=$ ca.ca_address_sk（客户表的当前地址键 $=$ 客户地址表的地址键），且ss.ss_store_sk（店铺销售的店铺键）等于s.s_store_sk（店铺表的店铺键），按i.i_br（此处可能有误，推测为i.i_brand）、i.i_br（此处可能有误，推测为i.i_brand）、_id（此处可能有误）、i.i_manufact_id（商品的制造商ID）、i.i_manufact（商品的制造商）分组；Q22 选择i.i_product_name（商品名称）、i.i_brand（商品品牌）、i.i_class（商品类别）、i.i_category（商品分类），将inv.inv_quantity_on_hand（库存的现有数量）的总和作为qoh（现有库存数量）。从date_dim（日期维度表）dd、inventory（库存表）inv、item（商品表）i、warehouse（仓库表）wh中选取，条件是dd.d_date_sk（日期维度的日期键）等于inv.inv_date_sk（库存的日期键），且inv.inv_item_sk（库存的商品键）等于i.i_item_sk（商品的商品键），且inv.inv_warehouse_sk（库存的仓库键）等于wh.w_warehouse_sk（仓库表的仓库键），且dd.d_month_seq（日期维度的月份序列）在1193到1204之间，按i.i_product_name（商品名称）、i.i_brand（商品品牌）、i.i_clASs（此处可能有误，推测为i.i_class）、i.i_category（商品分类）分组。