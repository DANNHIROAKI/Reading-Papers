# Join Sampling under Acyclic Degree Constraints and (Cyclic) Subgraph Sampling

# 无环度约束下的连接采样与（循环）子图采样

Ru Wang Yufei Tao

王茹 陶宇飞

Department of Computer Science and Engineering Chinese University of Hong Kong \{rwang21, taoyf\}@cse.cuhk.edu.hk

香港中文大学计算机科学与工程系 \{rwang21, taoyf\}@cse.cuhk.edu.hk

December 20, 2023

2023年12月20日

## Abstract

## 摘要

Given a (natural) join with an acyclic set of degree constraints (the join itself does not need to be acyclic), we show how to draw a uniformly random sample from the join result in $O\left( {\text{polymat}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time (assuming data complexity) after a preprocessing phase of $O\left( \mathrm{{IN}}\right)$ expected time,where IN,OUT,and polymat are the join’s input size,output size, and polymatroid bound, respectively. This compares favorably with the state of the art (Deng et al. and Kim et al., both in PODS'23), which states that, in the absence of degree constraints,a uniformly random sample can be drawn in $\widetilde{O}\left( {{AGM}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time after a preprocessing phase of $\widetilde{O}\left( \mathrm{{IN}}\right)$ expected time,where ${AGM}$ is the join’s AGM bound and $\widetilde{O}\left( \text{.}\right)$ hides a polylog(IN) factor. Our algorithm applies to every join supported by the solutions of Deng et al. and Kim et al. Furthermore, since the polymatroid bound is at most the AGM bound, our performance guarantees are never worse, but can be considerably better, than those of Deng et al. and Kim et al.

给定一个带有无环度约束集的（自然）连接（连接本身不需要是无环的），我们展示了如何在经过期望时间为 $O\left( \mathrm{{IN}}\right)$ 的预处理阶段后，以期望时间 $O\left( {\text{polymat}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 从连接结果中抽取一个均匀随机样本（假设为数据复杂度），其中 IN、OUT 和 polymat 分别是连接的输入大小、输出大小和多拟阵界（polymatroid bound）。这与现有技术（邓等人和金等人，均发表于 PODS'23）相比更具优势，现有技术表明，在没有度约束的情况下，经过期望时间为 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 的预处理阶段后，可以以期望时间 $\widetilde{O}\left( {{AGM}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 抽取一个均匀随机样本，其中 ${AGM}$ 是连接的 AGM 界（AGM bound），$\widetilde{O}\left( \text{.}\right)$ 隐藏了一个 polylog(IN) 因子。我们的算法适用于邓等人和金等人的解决方案所支持的每一个连接。此外，由于多拟阵界至多为 AGM 界，我们的性能保证绝不会比邓等人和金等人的差，而且可能会好得多。

We then utilize our techniques to tackle directed subgraph sampling, a problem that has extensive database applications and bears close relevance to joins. Let $G = \left( {V,E}\right)$ be a directed data graph where each vertex has an out-degree at most $\lambda$ ,and let $P$ be a directed pattern graph with a constant number of vertices. The objective is to uniformly sample an occurrence of $P$ in $G$ . The problem can be modeled as join sampling with input size $\operatorname{IN} = \Theta \left( \left| E\right| \right)$ but,whenever $P$ contains cycles,the converted join has cyclic degree constraints. We show that it is always possible to throw away certain degree constraints such that (i) the remaining constraints are acyclic and (ii) the new join has asymptotically the same polymatroid bound polymat as the old one. Combining this finding with our new join sampling solution yields an algorithm to sample from the original (cyclic) join (thereby yielding a uniformly random occurrence of $P$ ) in $O\left( {\text{polymat}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time after $O\left( \left| E\right| \right)$ expected-time preprocessing,where $\mathrm{{OUT}}$ is the number of occurrences. We also prove similar results for undirected subgraph sampling and demonstrate how our techniques can be significantly simplified in that scenario. Previously, the state of the art for (undirected/directed) subgraph sampling uses $O\left( {{\left| E\right| }^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ time to draw a sample (after $O\left( \left| E\right| \right)$ expected-time preprocessing) where ${\rho }^{ * }$ is the fractional edge cover number of $P$ . Our results are more favorable because polymat never exceeds but can be considerably lower than ${\left| E\right| }^{{\rho }^{ * }}$ .

然后，我们利用我们的技术来解决有向子图采样问题，这是一个在数据库中有广泛应用且与连接密切相关的问题。设 $G = \left( {V,E}\right)$ 是一个有向数据图，其中每个顶点的出度至多为 $\lambda$，设 $P$ 是一个顶点数量固定的有向模式图。目标是从 $G$ 中均匀采样 $P$ 的一个出现实例。该问题可以建模为输入大小为 $\operatorname{IN} = \Theta \left( \left| E\right| \right)$ 的连接采样问题，但是，只要 $P$ 包含循环，转换后的连接就具有循环度约束。我们表明，总是可以舍弃某些度约束，使得（i）剩余的约束是无环的，并且（ii）新连接的多拟阵界 polymat 与旧连接的渐近相同。将这一发现与我们新的连接采样解决方案相结合，得到了一个算法，该算法在经过期望时间为 $O\left( \left| E\right| \right)$ 的预处理后，以期望时间 $O\left( {\text{polymat}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 从原始（循环）连接中采样（从而得到 $P$ 的一个均匀随机出现实例），其中 $\mathrm{{OUT}}$ 是出现实例的数量。我们还证明了无向子图采样的类似结果，并展示了在该场景下我们的技术如何显著简化。此前，（无向/有向）子图采样的现有技术使用 $O\left( {{\left| E\right| }^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 时间来抽取一个样本（经过期望时间为 $O\left( \left| E\right| \right)$ 的预处理），其中 ${\rho }^{ * }$ 是 $P$ 的分数边覆盖数（fractional edge cover number）。我们的结果更具优势，因为 polymat 从不超过但可能远低于 ${\left| E\right| }^{{\rho }^{ * }}$。

## 1 Introduction

## 1 引言

In relational database systems, (natural) joins are acknowledged as notably computation-intensive, with its cost surging drastically in response to expanding data volumes. In the current big data era, the imperative to circumvent excessive computation increasingly overshadows the requirement for complete join results. A myriad of applications, including machine learning algorithms, online analytical processing, and recommendation systems, can operate effectively with random samples. This situation has sparked research initiatives focused on devising techniques capable of producing samples from a join result significantly faster than executing the join in its entirety. In the realm of graph theory, the significance of join operations is mirrored in their intrinsic connections to subgraph listing,a classical problem that seeks to pinpoint all the occurrences of a pattern $P$ (for instance, a directed 3-vertex cycle) within a data graph $G$ (such as a social network where a directed edge symbolizes a "follow" relationship). Analogous to joins, subgraph listing demands a vast amount of computation time,which escalates rapidly with the sizes of $G$ and $P$ . Fortunately,many social network analyses do not require the full set of occurrences of $P$ ,but can function well with only samples from those occurrences. This has triggered the development of methods that can extract samples considerably faster than finding all the occurrences.

在关系数据库系统中，（自然）连接被认为是计算量极大的操作，随着数据量的增加，其成本会急剧上升。在当前的大数据时代，避免过度计算的需求日益超过对完整连接结果的需求。包括机器学习算法、在线分析处理和推荐系统在内的众多应用程序，使用随机样本就可以有效运行。这种情况引发了相关研究，旨在设计出能够比执行完整连接快得多地从连接结果中生成样本的技术。在图论领域，连接操作的重要性体现在它们与子图列举（subgraph listing）的内在联系上，子图列举是一个经典问题，旨在找出数据图 $G$（例如，有向边表示“关注”关系的社交网络）中所有模式 $P$（例如，有向三顶点环）的出现位置。与连接操作类似，子图列举需要大量的计算时间，并且随着 $G$ 和 $P$ 的规模增大而迅速增加。幸运的是，许多社交网络分析并不需要 $P$ 的所有出现位置，只需要这些出现位置的样本就可以很好地运行。这促使了一些方法的发展，这些方法提取样本的速度比找出所有出现位置要快得多。

This paper will revisit join sampling and subgraph sampling under a unified "degree-constrained framework". Next, we will first describe the framework formally in Section 1.1, review the previous results in Section 1.2, and then overview our results in Section 1.3.

本文将在统一的“度约束框架”下重新探讨连接采样和子图采样问题。接下来，我们将在 1.1 节正式描述该框架，在 1.2 节回顾以往的研究成果，然后在 1.3 节概述我们的研究结果。

### 1.1 Problem Definitions

### 1.1 问题定义

Join Sampling. Let att be a finite set, with each element called an attribute, and dom be a countably infinite set,with each element called a value. For a non-empty set $\mathcal{X} \subseteq$ att of attributes, a tuple over $X$ is a function $\mathbf{u} : \mathcal{X} \rightarrow$ dom. For any non-empty subset $\mathcal{Y} \subseteq  \mathcal{X}$ ,we define $\mathbf{u}\left\lbrack  \mathcal{Y}\right\rbrack   -$ the projection of $\mathbf{u}$ on $Y$ - as the tuple $\mathbf{v}$ over $\mathcal{Y}$ satisfying $\mathbf{v}\left( Y\right)  = \mathbf{u}\left( Y\right)$ for every attribute $Y \in  \mathcal{Y}$ .

连接采样。设 att 为一个有限集，其每个元素称为一个属性；设 dom 为一个可数无限集，其每个元素称为一个值。对于非空属性集 $\mathcal{X} \subseteq$ att，定义在 $X$ 上的一个元组是一个函数 $\mathbf{u} : \mathcal{X} \rightarrow$ dom。对于任意非空子集 $\mathcal{Y} \subseteq  \mathcal{X}$，我们定义 $\mathbf{u}\left\lbrack  \mathcal{Y}\right\rbrack   -$（即 $\mathbf{u}$ 在 $Y$ 上的投影）为定义在 $\mathcal{Y}$ 上的元组 $\mathbf{v}$，使得对于每个属性 $Y \in  \mathcal{Y}$ 都满足 $\mathbf{v}\left( Y\right)  = \mathbf{u}\left( Y\right)$。

A relation $R$ is a set of tuples over the same set $\mathcal{Z}$ of attributes; we refer to $\mathcal{Z}$ as the schema of $R$ and represent it as $\operatorname{schema}\left( R\right)$ . The arity of $R$ is the size of $\operatorname{schema}\left( R\right)$ . For any subsets $\mathcal{X}$ and $\mathcal{Y}$ of schema(R)satisfying $\mathcal{X} \subset  \mathcal{Y}$ (note: $\mathcal{X}$ is a proper subset of $\mathcal{Y}$ ),define:

关系 $R$ 是定义在同一属性集 $\mathcal{Z}$ 上的元组集合；我们将 $\mathcal{Z}$ 称为 $R$ 的模式（schema），并将其表示为 $\operatorname{schema}\left( R\right)$。$R$ 的元数（arity）是 $\operatorname{schema}\left( R\right)$ 的大小。对于模式(R)的任意子集 $\mathcal{X}$ 和 $\mathcal{Y}$，若满足 $\mathcal{X} \subset  \mathcal{Y}$（注意：$\mathcal{X}$ 是 $\mathcal{Y}$ 的真子集），定义：

$$
{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)  = \mathop{\max }\limits_{{\text{tuple }\mathbf{u}\text{ over }\mathcal{X}}}\left| \left\{  {\mathbf{v}\left\lbrack  \mathcal{Y}\right\rbrack   \mid  \mathbf{v} \in  R,\mathbf{v}\left\lbrack  \mathcal{X}\right\rbrack   = \mathbf{u}}\right\}  \right| . \tag{1}
$$

For an intuitive explanation,imagine grouping the tuples of $R$ by $\mathcal{X}$ and counting,for each group, how many distinct $\mathcal{Y}$ -projections are formed by the tuples therein. Then,the value ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)$ corresponds to the maximum count of all groups. It is worth pointing out that,when $\mathcal{X} = \varnothing$ ,then ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)$ is simply $\left| {{\Pi }_{\mathcal{Y}}\left( R\right) }\right|$ where $\Pi$ is the standard "projection" operator in relational algebra. If in addition $\mathcal{Y} = \operatorname{schema}\left( R\right)$ ,then ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)$ equals $\left| R\right|$ .

为了直观解释，想象将 $R$ 中的元组按 $\mathcal{X}$ 进行分组，并对每个组计算其中的元组形成了多少个不同的 $\mathcal{Y}$ -投影。那么，值 ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)$ 对应于所有组中的最大计数。值得指出的是，当 $\mathcal{X} = \varnothing$ 时，${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)$ 就是 $\left| {{\Pi }_{\mathcal{Y}}\left( R\right) }\right|$，其中 $\Pi$ 是关系代数中的标准“投影”运算符。此外，如果 $\mathcal{Y} = \operatorname{schema}\left( R\right)$，那么 ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)$ 等于 $\left| R\right|$。

We define a join as a set $\mathcal{Q}$ of relations (some of which may have the same schema). Let schema(Q)be the union of the attributes of the relations in $\mathcal{Q}$ ,i.e.,schema $\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{R \in  Q}}$ schema(R). Focusing on "data complexity",we consider only joins where both $\mathcal{Q}$ and schema(Q)have constant sizes. The result of $\mathcal{Q}$ is a relation over $\operatorname{schema}\left( \mathcal{Q}\right)$ formalized as:

我们将连接（join）定义为一个关系集合$\mathcal{Q}$（其中一些关系可能具有相同的模式）。设schema(Q)为集合$\mathcal{Q}$中各关系的属性的并集，即schema $\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{R \in  Q}}$ schema(R)。聚焦于“数据复杂度”，我们仅考虑$\mathcal{Q}$和schema(Q)大小均为常量的连接。$\mathcal{Q}$的结果是一个基于$\operatorname{schema}\left( \mathcal{Q}\right)$的关系，形式化表示如下：

$$
\text{join}\left( \mathcal{Q}\right)  = \{ \text{tuple}\mathbf{u}\text{over schema}\left( \mathcal{Q}\right)  \mid  \forall R \in  \mathcal{Q} : \mathbf{u}\left\lbrack  {\text{schema}\left( R\right) }\right\rbrack   \in  R\} \text{.}
$$

Define $\operatorname{IN} = \mathop{\sum }\limits_{{R \in  \mathcal{Q}}}\left| R\right|$ and $\operatorname{OUT} = \left| {\operatorname{join}\left( \mathcal{Q}\right) }\right|$ . We will refer to $\operatorname{IN}$ and OUT as the input size and output size of $Q$ ,respectively.

定义$\operatorname{IN} = \mathop{\sum }\limits_{{R \in  \mathcal{Q}}}\left| R\right|$和$\operatorname{OUT} = \left| {\operatorname{join}\left( \mathcal{Q}\right) }\right|$。我们将分别把$\operatorname{IN}$和OUT称为$Q$的输入规模和输出规模。

A join sampling operation returns a tuple drawn uniformly at random from join(Q)or declares ${join}\left( \mathcal{Q}\right)  = \varnothing$ . All such operations must be mutually independent. The objective of the join sampling problem is to preprocess the input relations of $\mathcal{Q}$ into an appropriate data structure that can be used to perform join-sampling operations repeatedly.

连接抽样操作会从连接(Q)中均匀随机抽取一个元组，或者声明${join}\left( \mathcal{Q}\right)  = \varnothing$。所有此类操作必须相互独立。连接抽样问题的目标是将$\mathcal{Q}$的输入关系预处理成一个合适的数据结构，以便能重复执行连接抽样操作。

We study the problem in the scenario where $\mathcal{Q}$ conforms to a set $\mathrm{{DC}}$ of degree constraints. Specifically,each degree constraint has the form $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$ where $\mathcal{X}$ and $\mathcal{Y}$ are subsets of schema(Q)satisfying $\mathcal{X} \subset  \mathcal{Y}$ and ${N}_{\mathcal{Y} \mid  \mathcal{X}} \geq  1$ is an integer. A relation $R \in  \mathcal{Q}$ is said to guard the constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$ if

我们研究$\mathcal{Q}$符合一组度约束$\mathrm{{DC}}$的场景下的问题。具体而言，每个度约束的形式为$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$，其中$\mathcal{X}$和$\mathcal{Y}$是schema(Q)的子集，满足$\mathcal{X} \subset  \mathcal{Y}$，且${N}_{\mathcal{Y} \mid  \mathcal{X}} \geq  1$是一个整数。若一个关系$R \in  \mathcal{Q}$满足以下条件，则称其守护约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$：

$$
\mathcal{Y} \subseteq  \operatorname{schema}\left( R\right) \text{,and}{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( R\right)  \leq  {N}_{\mathcal{Y} \mid  \mathcal{X}}\text{.}
$$

The join $\mathcal{Q}$ is consistent with $\mathrm{{DC}}$ - written as $\mathcal{Q} \vDash  \mathrm{{DC}}$ - if every degree constraint in $\mathrm{{DC}}$ is guarded by at least one relation in $\mathcal{Q}$ . It is safe to assume that $\mathrm{{DC}}$ does not have two constraints $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$ and $\left( {{\mathcal{X}}^{\prime },{\mathcal{Y}}^{\prime },{N}_{{\mathcal{Y}}^{\prime } \mid  {\mathcal{X}}^{\prime }}}\right)$ with $\mathcal{X} = {\mathcal{X}}^{\prime }$ and $\mathcal{Y} = {\mathcal{Y}}^{\prime }$ ; otherwise,assuming ${N}_{\mathcal{Y} \mid  \mathcal{X}} \leq  {N}_{{\mathcal{Y}}^{\prime } \mid  {\mathcal{X}}^{\prime }}$ the constraint $\left( {{\mathcal{X}}^{\prime },{\mathcal{Y}}^{\prime },{N}_{{\mathcal{Y}}^{\prime } \mid  {\mathcal{X}}^{\prime }}}\right)$ is redundant and can be removed from DC.

若连接$\mathcal{Q}$与$\mathrm{{DC}}$一致（记为$\mathcal{Q} \vDash  \mathrm{{DC}}$），则意味着$\mathrm{{DC}}$中的每个度约束都至少由$\mathcal{Q}$中的一个关系守护。可以放心假设$\mathrm{{DC}}$中不存在两个约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$和$\left( {{\mathcal{X}}^{\prime },{\mathcal{Y}}^{\prime },{N}_{{\mathcal{Y}}^{\prime } \mid  {\mathcal{X}}^{\prime }}}\right)$，使得$\mathcal{X} = {\mathcal{X}}^{\prime }$且$\mathcal{Y} = {\mathcal{Y}}^{\prime }$；否则，假设${N}_{\mathcal{Y} \mid  \mathcal{X}} \leq  {N}_{{\mathcal{Y}}^{\prime } \mid  {\mathcal{X}}^{\prime }}$，约束$\left( {{\mathcal{X}}^{\prime },{\mathcal{Y}}^{\prime },{N}_{{\mathcal{Y}}^{\prime } \mid  {\mathcal{X}}^{\prime }}}\right)$是冗余的，可以从DC中移除。

In this work, we concentrate on "acyclic" degree dependency. To formalize this notion, let us define a constraint dependency graph ${G}_{\mathrm{{DC}}}$ as follows. This is a directed graph whose vertex set is $\operatorname{schema}\left( \mathcal{Q}\right)$ (i.e.,each vertex of ${G}_{\mathrm{{DC}}}$ is an attribute in schema(Q)). For each degree constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$ such that $\mathcal{X} \neq  \varnothing$ ,we add a (directed) edge(X,Y)to ${G}_{\mathrm{{DC}}}$ for every pair $\left( {X,Y}\right)  \in  \mathcal{X} \times  \left( {\mathcal{Y} - \mathcal{X}}\right)$ . We say that the set $\mathrm{{DC}}$ is acyclic if ${G}_{\mathrm{{DC}}}$ is an acyclic graph; otherwise, $\mathrm{{DC}}$ is cyclic.

在这项工作中，我们专注于“无环”度依赖。为了形式化这个概念，让我们如下定义一个约束依赖图${G}_{\mathrm{{DC}}}$。这是一个有向图，其顶点集为$\operatorname{schema}\left( \mathcal{Q}\right)$（即，${G}_{\mathrm{{DC}}}$的每个顶点都是模式(Q)中的一个属性）。对于每个度约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$，使得$\mathcal{X} \neq  \varnothing$，对于每一对$\left( {X,Y}\right)  \in  \mathcal{X} \times  \left( {\mathcal{Y} - \mathcal{X}}\right)$，我们向${G}_{\mathrm{{DC}}}$中添加一条（有向）边(X, Y)。我们称集合$\mathrm{{DC}}$是无环的，如果${G}_{\mathrm{{DC}}}$是一个无环图；否则，$\mathrm{{DC}}$是有环的。

It is important to note that each relation $R \in  \mathcal{Q}$ implicitly defines a special degree constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$ where $\mathcal{X} = \varnothing ,\mathcal{Y} = \operatorname{schema}\left( R\right)$ ,and ${N}_{\mathcal{Y} \mid  \mathcal{X}} = \left| R\right|$ . Such a constraint - known as a cardinality constraint - is always assumed to be present in DC. As all cardinality constraints have $\mathcal{X} = \varnothing$ ,they do not affect the construction of ${G}_{\mathrm{{DC}}}$ . Consequently,if DC only contains cardinality constraints,then ${G}_{\mathrm{{DC}}}$ is empty and hence trivially acyclic. Moreover,readers should avoid the misconception that "an acyclic ${G}_{\mathrm{{DC}}}$ implies $\mathcal{Q}$ being an acyclic join"; these two acyclicity notions are unrelated. While the definition of an acyclic join is not needed for our discussion, readers unfamiliar with this term may refer to [2, Chapter 6.4].

需要注意的是，每个关系$R \in  \mathcal{Q}$都隐式地定义了一个特殊的度约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$，其中$\mathcal{X} = \varnothing ,\mathcal{Y} = \operatorname{schema}\left( R\right)$，且${N}_{\mathcal{Y} \mid  \mathcal{X}} = \left| R\right|$。这样的约束——称为基数约束——总是被假定存在于DC中。由于所有基数约束都有$\mathcal{X} = \varnothing$，它们不会影响${G}_{\mathrm{{DC}}}$的构建。因此，如果DC仅包含基数约束，那么${G}_{\mathrm{{DC}}}$为空，从而显然是无环的。此外，读者应避免产生“无环的${G}_{\mathrm{{DC}}}$意味着$\mathcal{Q}$是无环连接”的误解；这两个无环性概念是不相关的。虽然我们的讨论不需要无环连接的定义，但不熟悉这个术语的读者可以参考[2，第6.4章]。

Directed Graph Sampling. We are given a data graph $G = \left( {V,E}\right)$ and a pattern graph $P =$ $\left( {{V}_{P},{E}_{P}}\right)$ ,both being simple directed graphs. The pattern graph is weakly-connected ${}^{1}$ and has a constant number of vertices. A simple directed graph ${G}_{\text{sub }} = \left( {{V}_{\text{sub }},{E}_{\text{sub }}}\right)$ is a subgraph of $G$ if ${V}_{sub} \subseteq  V$ and ${E}_{sub} \subseteq  E$ . The subgraph ${G}_{sub}$ is an occurrence of $P$ if they are isomorphic,namely, there is a bijection $f : {V}_{\text{sub }} \rightarrow  {V}_{P}$ such that,for any distinct vertices ${u}_{1},{u}_{2} \in  {V}_{\text{sub }}$ ,there is an edge $\left( {{u}_{1},{u}_{2}}\right)  \in  {E}_{\text{sub }}$ if and only if $\left( {f\left( {u}_{1}\right) ,f\left( {u}_{2}\right) }\right)$ is an edge in ${E}_{P}$ . We will refer to $f$ as a isomorphism bijection between $P$ and ${G}_{\text{sub }}$ .

有向图采样。给定一个数据图$G = \left( {V,E}\right)$和一个模式图$P =$ $\left( {{V}_{P},{E}_{P}}\right)$，它们都是简单有向图。模式图是弱连通的${}^{1}$，并且具有固定数量的顶点。如果${V}_{sub} \subseteq  V$且${E}_{sub} \subseteq  E$，则简单有向图${G}_{\text{sub }} = \left( {{V}_{\text{sub }},{E}_{\text{sub }}}\right)$是$G$的子图。如果子图${G}_{sub}$与$P$同构，即存在一个双射$f : {V}_{\text{sub }} \rightarrow  {V}_{P}$，使得对于任意不同的顶点${u}_{1},{u}_{2} \in  {V}_{\text{sub }}$，当且仅当$\left( {f\left( {u}_{1}\right) ,f\left( {u}_{2}\right) }\right)$是${E}_{P}$中的一条边时，存在一条边$\left( {{u}_{1},{u}_{2}}\right)  \in  {E}_{\text{sub }}$，那么${G}_{sub}$是$P$的一个实例。我们将$f$称为$P$和${G}_{\text{sub }}$之间的同构双射。

A subgraph sampling operation returns an occurrence of $P$ in $G$ uniformly at random or declares the absence of any occurrence. All such operations need to be mutually independent. The objective of the subgraph sampling problem is to preprocess $G$ into a data structure that can support every subgraph-sampling operation efficiently. We will study the problem under a degree constraint: every vertex in $G$ has an out-degree at most $\lambda$ .

子图采样操作会从$G$中均匀随机地返回一个$P$的实例，或者声明不存在任何实例。所有此类操作必须相互独立。子图采样问题的目标是将$G$预处理成一种数据结构，以便能够高效地支持每一次子图采样操作。我们将在度约束条件下研究该问题：$G$中的每个顶点的出度至多为$\lambda$。

Undirected Graph Sampling. The setup of this problem is the same as the previous problem except that (i) both the data graph $G = \left( {V,E}\right)$ and the pattern graph $P = \left( {{V}_{P},{E}_{P}}\right)$ are simple undirected graphs,with $P$ being connected; (ii) a subgraph ${G}_{\text{sub }}$ of $G$ is an occurrence of $P$ if ${G}_{\text{sub }}$ and $P$ are isomorphic in the undirected sense: there is a isomorphism bijection $f : {V}_{\text{sub }} \rightarrow  {V}_{P}$ between $P$ and ${G}_{sub}$ such that,for any distinct ${u}_{1},{u}_{2} \in  {V}_{sub}$ ,an edge $\left\{  {{u}_{1},{u}_{2}}\right\}$ exists in ${E}_{sub}$ if and only if $\left\{  {f\left( {u}_{1}\right) ,f\left( {u}_{2}\right) }\right\}   \in  {E}_{P}{;}^{2}$ and (iii) the degree constraint becomes: every vertex in $G$ has a degree at most $\lambda$ .

无向图采样。该问题的设定与上一个问题相同，除了：（i）数据图$G = \left( {V,E}\right)$和模式图$P = \left( {{V}_{P},{E}_{P}}\right)$均为简单无向图，且$P$是连通的；（ii）若${G}_{\text{sub }}$和$P$在无向意义下是同构的，即存在一个从$P$到${G}_{sub}$的同构双射$f : {V}_{\text{sub }} \rightarrow  {V}_{P}$，使得对于任意不同的${u}_{1},{u}_{2} \in  {V}_{sub}$，当且仅当$\left\{  {f\left( {u}_{1}\right) ,f\left( {u}_{2}\right) }\right\}   \in  {E}_{P}{;}^{2}$时，${E}_{sub}$中存在边$\left\{  {{u}_{1},{u}_{2}}\right\}$，则$G$的子图${G}_{\text{sub }}$是$P$的一个实例；（iii）度约束变为：$G$中的每个顶点的度至多为$\lambda$。

---

<!-- Footnote -->

${}^{1}$ Namely,if we ignore the edge directions,then $P$ becomes a connected undirected graph.

${}^{1}$ 即，如果我们忽略边的方向，那么$P$就变成了一个连通的无向图。

${}^{2}$ We represent a directed edge as an ordered pair and an undirected edge as a set.

${}^{2}$ 我们将有向边表示为有序对，将无向边表示为集合。

<!-- Footnote -->

---

Math Conventions. For an integer $x \geq  1$ ,the notation $\left\lbrack  x\right\rbrack$ denotes the set $\{ 1,2,\ldots ,x\}$ ; as a special case, $\left\lbrack  0\right\rbrack$ represents the empty set. Every logarithm $\log \left( \cdot \right)$ has base 2,and function ${\exp }_{2}\left( x\right)$ is defined to be ${2}^{x}$ . We use double curly braces to represent multi-sets,e.g., $\{ \{ 1,1,1,2,2,3\} \}$ is a multi-set with 6 elements.

数学约定。对于整数$x \geq  1$，符号$\left\lbrack  x\right\rbrack$表示集合$\{ 1,2,\ldots ,x\}$；特别地，$\left\lbrack  0\right\rbrack$表示空集。每个对数$\log \left( \cdot \right)$的底数均为2，函数${\exp }_{2}\left( x\right)$定义为${2}^{x}$。我们使用双花括号来表示多重集，例如，$\{ \{ 1,1,1,2,2,3\} \}$是一个包含6个元素的多重集。

### 1.2 Related Work

### 1.2 相关工作

Join Computation. Any algorithm correctly answering a join query $\mathcal{Q}$ must incur $\Omega$ (OUT) time just to output the OUT tuples in join(Q). Hence,finding the greatest possible value of OUT is an imperative step towards unraveling the time complexity of join evaluation. A classical result in this regard is the ${AGM}$ bound [7]. To describe this bound,let us define the schema graph of $\mathcal{Q}$ as a multi-hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ where

连接计算。任何能正确回答连接查询$\mathcal{Q}$的算法，仅输出连接结果join(Q)中的OUT元组就必须花费$\Omega$（OUT）的时间。因此，找出OUT的最大可能值是揭示连接评估时间复杂度的关键一步。在这方面的一个经典结果是${AGM}$界[7]。为了描述这个界，我们将$\mathcal{Q}$的模式图定义为一个多重超图$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$，其中

$$
\mathcal{V} = \operatorname{schema}\left( \mathcal{Q}\right) \text{,and}\mathcal{E} = \{ \{ \operatorname{schema}\left( R\right)  \mid  R \in  \mathcal{Q}\} \} \text{.} \tag{2}
$$

Note that $\mathcal{E}$ is a multi-set because the relations in $\mathcal{Q}$ may have identical schemas. A fractional edge cover of $\mathcal{G}$ is a function $w : \mathcal{E} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ such that,for any $X \in  \mathcal{V}$ ,it holds that $\mathop{\sum }\limits_{{F \in  \mathcal{E} : X \in  F}}w\left( F\right)  \geq  1$ (namely,the total weight assigned to the hyperedges covering $X$ is at least 1). Atserias,Grohe,and Marx [7] showed that,given any fractional edge cover,it always holds that OUT $\leq  \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{w\left( F\right) }$ , where ${R}_{F}$ is the relation in $\mathcal{Q}$ whose schema corresponds to the hyperedge $F$ . The AGM bound is defined as $\operatorname{AGM}\left( \mathcal{Q}\right)  = \mathop{\min }\limits_{w}\mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{w\left( F\right) }$ .

请注意，$\mathcal{E}$ 是一个多重集，因为 $\mathcal{Q}$ 中的关系可能具有相同的模式。$\mathcal{G}$ 的一个分数边覆盖是一个函数 $w : \mathcal{E} \rightarrow  \left\lbrack  {0,1}\right\rbrack$，使得对于任何 $X \in  \mathcal{V}$，都有 $\mathop{\sum }\limits_{{F \in  \mathcal{E} : X \in  F}}w\left( F\right)  \geq  1$ 成立（即，覆盖 $X$ 的超边所分配的总权重至少为 1）。Atserias、Grohe 和 Marx [7] 表明，给定任何分数边覆盖，总有 OUT $\leq  \mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{w\left( F\right) }$ 成立，其中 ${R}_{F}$ 是 $\mathcal{Q}$ 中模式对应于超边 $F$ 的关系。AGM 界定义为 $\operatorname{AGM}\left( \mathcal{Q}\right)  = \mathop{\min }\limits_{w}\mathop{\prod }\limits_{{F \in  \mathcal{E}}}{\left| {R}_{F}\right| }^{w\left( F\right) }$。

The AGM bound is tight: given any hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ and any set of positive integers $\left\{  {{N}_{F} \mid  F \in  \mathcal{E}}\right\}$ ,there is always a join $\mathcal{Q}$ such that $\mathcal{Q}$ has $\mathcal{G}$ as the schema graph, $\left| {R}_{F}\right|  = \left| {N}_{F}\right|$ for each $F \in  \mathcal{E}$ ,and the output size OUT is $\Theta \left( {{AGM}\left( \mathcal{Q}\right) }\right)$ . This has motivated the development of algorithms $\left\lbrack  {5,{14},{21},{23},{25},{28},{31} - {34},{38}}\right\rbrack$ that can compute $\operatorname{join}\left( \mathcal{Q}\right)$ in $\widetilde{O}\left( {{AGM}\left( \mathcal{Q}\right) }\right)$ time - where $\widetilde{O}\left( \text{.}\right) {hidesafafactorpolylogarithmictotheinputsizeIN}$ of $\mathcal{Q} -$ and therefore are worst-case optimal up to an $\widetilde{O}\left( 1\right)$ factor.

AGM 界是紧的：给定任何超图 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ 和任何正整数集合 $\left\{  {{N}_{F} \mid  F \in  \mathcal{E}}\right\}$，总是存在一个连接 $\mathcal{Q}$，使得 $\mathcal{Q}$ 以 $\mathcal{G}$ 作为模式图，对于每个 $F \in  \mathcal{E}$ 都有 $\left| {R}_{F}\right|  = \left| {N}_{F}\right|$，并且输出大小 OUT 为 $\Theta \left( {{AGM}\left( \mathcal{Q}\right) }\right)$。这推动了算法 $\left\lbrack  {5,{14},{21},{23},{25},{28},{31} - {34},{38}}\right\rbrack$ 的发展，这些算法可以在 $\widetilde{O}\left( {{AGM}\left( \mathcal{Q}\right) }\right)$ 时间内计算 $\operatorname{join}\left( \mathcal{Q}\right)$ —— 其中 $\widetilde{O}\left( \text{.}\right) {hidesafafactorpolylogarithmictotheinputsizeIN}$ 是 $\mathcal{Q} -$ 的 ，因此在 $\widetilde{O}\left( 1\right)$ 因子范围内是最坏情况下的最优算法。

However, the tightness of the AGM bound relies on the assumption that all the degree constraints on $\mathcal{Q}$ are purely cardinality constraints. In reality,general degree constraints are prevalent,and their inclusion could dramatically decrease the maximum output size OUT. This observation has sparked significant interest $\left\lbrack  {{13},{17},{21},{22},{24},{25},{30},{36}}\right\rbrack$ in establishing refined upper bounds on OUT tailored for more complex degree constraints. Most notably, Khamis et al. [25] proposed the entropic bound, which is applicable to any set DC of degree constraints and is tight in a strong sense (see Theorem 5.5 of [36]). Unfortunately, the entropic bound is difficult to compute because it requires solving a linear program (LP) involving infinitely many constraints (it remains an open problem whether the computation is decidable). Not coincidentally, no join algorithm is known to have a running time matching the entropic bound.

然而，AGM 界的紧性依赖于一个假设，即 $\mathcal{Q}$ 上的所有度约束都是纯粹的基数约束。在现实中，一般的度约束很常见，包含这些约束可能会显著降低最大输出大小 OUT。这一观察引发了人们对为更复杂的度约束建立针对 OUT 的精细上界的浓厚兴趣 $\left\lbrack  {{13},{17},{21},{22},{24},{25},{30},{36}}\right\rbrack$。最值得注意的是，Khamis 等人 [25] 提出了熵界，它适用于任何度约束集合 DC，并且在很强的意义上是紧的（见 [36] 中的定理 5.5）。不幸的是，熵界很难计算，因为它需要求解一个涉及无限多个约束的线性规划（LP）问题（计算是否可判定仍是一个开放问题）。并非巧合的是，目前还没有已知的连接算法的运行时间能与熵界相匹配。

To circumvent the above issue, Khamis et al. [25] introduced the polymatroid bound as an alternative, which we represent as polymat(DC) because this bound is fully decided by DC (i.e., any join $\mathcal{Q} \vDash  \mathrm{{DC}}$ must satisfy $\mathrm{{OUT}} \leq$ polymat(DC)). Section 2 will discuss polymat(DC) in detail; for now, it suffices to understand that (i) the polymatroid bound, although possibly looser than the entropic bound,never exceeds the AGM bound,and (ii) polymat (DC) can be computed in $O\left( 1\right)$ time under data complexity. Khamis et al. [25] proposed an algorithm named PANDA that can evaluate an arbitrary join $\mathcal{Q} \vDash  \mathrm{{DC}}$ in time $\widetilde{O}\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ .

为了规避上述问题，卡米斯（Khamis）等人 [25] 引入了多拟阵界（polymatroid bound）作为替代方案，我们将其表示为 polymat(DC)，因为该界完全由 DC 决定（即，任何连接 $\mathcal{Q} \vDash  \mathrm{{DC}}$ 都必须满足 $\mathrm{{OUT}} \leq$ polymat(DC)）。第 2 节将详细讨论 polymat(DC)；目前，只需理解以下两点即可：（i）多拟阵界虽然可能比熵界（entropic bound）宽松，但绝不会超过算术 - 几何平均界（AGM bound）；（ii）在数据复杂度下，polymat(DC) 可以在 $O\left( 1\right)$ 时间内计算得出。卡米斯等人 [25] 提出了一种名为 PANDA 的算法，该算法可以在 $\widetilde{O}\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ 时间内评估任意连接 $\mathcal{Q} \vDash  \mathrm{{DC}}$。

Interestingly, when DC is acyclic, the entropic bound is equivalent to the polymatroid bound [30]. In this scenario,Ngo [30] presented a simple algorithm to compute any join $\mathcal{Q} \vDash  \mathrm{{DC}}$ in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ time,after a preprocessing of $O\left( \mathrm{{IN}}\right)$ expected time.

有趣的是，当 DC 为无环时，熵界等同于多拟阵界 [30]。在这种情况下，恩戈（Ngo）[30] 提出了一种简单的算法，在经过 $O\left( \mathrm{{IN}}\right)$ 期望时间的预处理后，可以在 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ 时间内计算任何连接 $\mathcal{Q} \vDash  \mathrm{{DC}}$。

Join Sampling. For an acyclic join (not to be confused with a join having an acyclic set of degree constraints), it is possible to sample from the join result in constant time, after a preprocessing of $O\left( \mathrm{{IN}}\right)$ expected time [39]. The problem becomes more complex when dealing with an arbitrary (cyclic) join $\mathcal{Q}$ ,with the latest advancements presented in two PODS’23 papers [14,26]. Specifically, Kim et al. [26] described how to sample in $\widetilde{O}\left( {{AGM}\left( \mathcal{Q}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,after a preprocessing of $\widetilde{O}\left( \mathrm{{IN}}\right)$ time. Deng et al. [14] achieved the same guarantees using different approaches,and offered a rationale explaining why the expected sample time $O\left( {{AGM}\left( \mathcal{Q}\right) /\mathrm{{OUT}}}\right)$ can no longer be significantly improved,even when $0 < \mathrm{{OUT}} \ll  \operatorname{AGM}\left( \mathcal{Q}\right)$ ,subject to commonly accepted conjectures. We refer readers to $\left\lbrack  {3,{10},{11},{14},{26},{39}}\right\rbrack$ and the references therein for other results (now superseded) on join sampling.

连接采样。对于无环连接（不要与具有无环度约束集的连接相混淆），在经过 $O\left( \mathrm{{IN}}\right)$ 期望时间的预处理后，可以在常数时间内从连接结果中进行采样 [39]。当处理任意（循环）连接 $\mathcal{Q}$ 时，问题变得更加复杂，2023 年 PODS 会议的两篇论文 [14,26] 介绍了最新进展。具体而言，金（Kim）等人 [26] 描述了如何在经过 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 时间的预处理后，在 $\widetilde{O}\left( {{AGM}\left( \mathcal{Q}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ 期望时间内进行采样。邓（Deng）等人 [14] 使用不同的方法实现了相同的保证，并给出了一个理由，解释了为什么在普遍接受的猜想下，即使 $0 < \mathrm{{OUT}} \ll  \operatorname{AGM}\left( \mathcal{Q}\right)$，期望采样时间 $O\left( {{AGM}\left( \mathcal{Q}\right) /\mathrm{{OUT}}}\right)$ 也无法再显著提高。我们建议读者参考 $\left\lbrack  {3,{10},{11},{14},{26},{39}}\right\rbrack$ 及其参考文献，以了解关于连接采样的其他结果（现已被取代）。

Subgraph Listing. Let us start by clarifying the fractional edge cover number ${\rho }^{ * }\left( P\right)$ of a simple undirected pattern graph $P = \left( {{V}_{P},{E}_{P}}\right)$ . Given a fractional edge cover of $P$ (i.e.,function $w : {E}_{P} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ such that,for any vertex $X \in  {V}_{P}$ ,we have $\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}w\left( F\right)  \geq  1$ ),define $\mathop{\sum }\limits_{{F \in  {E}_{P}}}w\left( F\right)$ as the total weight of $w$ . The value of ${\rho }^{ * }\left( P\right)$ is the smallest total weight of all fractional edge covers of $P$ . Given a directed pattern graph $P$ ,we define its fractional edge cover number ${\rho }^{ * }\left( P\right)$ as the value ${\rho }^{ * }\left( {P}^{\prime }\right)$ of the corresponding undirected graph ${P}^{\prime }$ that is obtained from $P$ by ignoring all the edge directions.

子图列举。让我们首先明确一个简单无向模式图 $P = \left( {{V}_{P},{E}_{P}}\right)$ 的分数边覆盖数 ${\rho }^{ * }\left( P\right)$。给定 $P$ 的一个分数边覆盖（即，函数 $w : {E}_{P} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 满足对于任何顶点 $X \in  {V}_{P}$，都有 $\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}w\left( F\right)  \geq  1$），将 $\mathop{\sum }\limits_{{F \in  {E}_{P}}}w\left( F\right)$ 定义为 $w$ 的总权重。${\rho }^{ * }\left( P\right)$ 的值是 $P$ 的所有分数边覆盖的最小总权重。给定一个有向模式图 $P$，我们将其分数边覆盖数 ${\rho }^{ * }\left( P\right)$ 定义为通过忽略 $P$ 中所有边的方向而得到的相应无向图 ${P}^{\prime }$ 的值 ${\rho }^{ * }\left( {P}^{\prime }\right)$。

When $P$ has a constant size,it is well-known $\left\lbrack  {4,7}\right\rbrack$ that any data graph $G = \left( {V,E}\right)$ can encompass $O\left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$ occurrences of $P$ . This holds true both when $P$ and $G$ are directed and when they are undirected. This upper bound is tight: in both the directed and undirected scenarios, for any integer $m$ ,there is a data graph $G = \left( {V,E}\right)$ with $\left| E\right|  = m$ edges that has $\Omega \left( {m}^{{\rho }^{ * }\left( P\right) }\right)$ occurrences of $P$ . Thus,a subgraph listing algorithm is considered worst-case optimal if it finishes in $\widetilde{O}\left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$ time.

当$P$的规模固定时，众所周知$\left\lbrack  {4,7}\right\rbrack$，任何数据图$G = \left( {V,E}\right)$都可以包含$P$的$O\left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$个实例。无论$P$和$G$是有向图还是无向图，这一点都成立。这个上界是紧的：在有向和无向两种情况下，对于任意整数$m$，都存在一个具有$\left| E\right|  = m$条边的数据图$G = \left( {V,E}\right)$，它包含$P$的$\Omega \left( {m}^{{\rho }^{ * }\left( P\right) }\right)$个实例。因此，如果一个子图列举算法能在$\widetilde{O}\left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$时间内完成，那么它就被认为是最坏情况下的最优算法。

It is well-known that directed/undirected subgraph listing can be converted to a join $\mathcal{Q}$ on binary relations (namely,relations of arity 2). The join $\mathcal{Q}$ has an input size of $\mathrm{{IN}} = \Theta \left( \left| E\right| \right)$ ,and its $\mathrm{{AGM}}$ bound is ${AGM}\left( \mathcal{Q}\right)  = \Theta \left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$ . All occurrences of $P$ in $G$ can be derived from $\operatorname{join}\left( \mathcal{Q}\right)$ for free. Thus,any $\widetilde{O}\left( {{AGM}\left( \mathcal{Q}\right) }\right)$ -time join algorithm is essentially worst-case optimal for subgraph listing.

众所周知，有向/无向子图列举问题可以转化为对二元关系（即元数为2的关系）的连接操作$\mathcal{Q}$。连接操作$\mathcal{Q}$的输入规模为$\mathrm{{IN}} = \Theta \left( \left| E\right| \right)$，其$\mathrm{{AGM}}$界为${AGM}\left( \mathcal{Q}\right)  = \Theta \left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$。$G$中$P$的所有实例都可以从$\operatorname{join}\left( \mathcal{Q}\right)$中免费推导出来。因此，任何$\widetilde{O}\left( {{AGM}\left( \mathcal{Q}\right) }\right)$时间的连接算法本质上都是子图列举问题在最坏情况下的最优算法。

Assuming $P$ and $G$ to be directed,Jayaraman et al. [19] presented interesting enhancement over the above transformation in the scenario where each vertex of $G$ has an out-degree at most $\lambda$ . The key lies in examining the polymatroid bound of the join $\mathcal{Q}$ derived from subgraph listing. As will be explained in Section 4,this join $\mathcal{Q}$ has a set DC of degree constraints whose constraint dependency graph ${G}_{\mathrm{{DC}}}$ coincides with $P$ . Jayaraman et al. developed an algorithm that lists all occurrences of $\mathcal{Q}$ in $G$ in $O$ (polymat(DC)) time (after a preprocessing of $O\left( \mathrm{{IN}}\right)$ expected time) and confirmed that this is worst-case optimal. Their findings are closely related to our work, and we will delve into them further when their specifics become crucial to our discussion.

假设$P$和$G$是有向图，贾亚拉曼（Jayaraman）等人[19]在$G$的每个顶点的出度至多为$\lambda$的情况下，对上述转换提出了有趣的改进。关键在于研究从子图列举问题导出的连接操作$\mathcal{Q}$的多拟阵界。正如第4节将解释的那样，这个连接操作$\mathcal{Q}$有一个度约束集合DC，其约束依赖图${G}_{\mathrm{{DC}}}$与$P$重合。贾亚拉曼等人开发了一种算法，该算法能在$O$（多拟阵(DC)）时间内（经过$O\left( \mathrm{{IN}}\right)$期望时间的预处理后）列举出$G$中$\mathcal{Q}$的所有实例，并证实这是最坏情况下的最优算法。他们的研究结果与我们的工作密切相关，当这些细节对我们的讨论至关重要时，我们将进一步深入探讨。

There is a substantial body of literature on bounding the cost of subgraph listing using parameters distinct from those already mentioned. These studies typically concentrate on specific patterns (such as paths, cycles, and cliques) or particular graphs (for instance, those that are sparse under a suitable metric). We refer interested readers to $\left\lbrack  {1,8,9,{12},{15},{18},{20},{27},{29},{37}}\right\rbrack$ and the references therein.

有大量文献使用与上述不同的参数来界定子图列举的成本。这些研究通常集中在特定的模式（如路径、循环和团）或特定的图（例如，在合适的度量下是稀疏的图）上。我们建议感兴趣的读者参考$\left\lbrack  {1,8,9,{12},{15},{18},{20},{27},{29},{37}}\right\rbrack$及其参考文献。

Subgraph Sampling. Fichtenberger, Gao, and Peng [16] described how to sample an occurrence of the pattern $P$ in the data graph $G$ in $O\left( {{\left| E\right| }^{{\rho }^{ * }\left( P\right) }/\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,where OUT is the number of occurrences of $P$ in $G$ ,after a preprocessing of $O\left( \left| E\right| \right)$ expected time. In [14],Deng et al. clarified how to deploy an arbitrary join sampling algorithm to perform subgraph sampling; their approach ensures the same guarantees as in [16],baring an $\widetilde{O}\left( 1\right)$ factor. The methods of $\left\lbrack  {{14},{16}}\right\rbrack$ are applicable in both undirected and directed scenarios.

子图采样。菲希滕贝格尔（Fichtenberger）、高（Gao）和彭（Peng）[16]描述了如何在$O\left( {{\left| E\right| }^{{\rho }^{ * }\left( P\right) }/\max \{ 1,\mathrm{{OUT}}\} }\right)$的期望时间内，在数据图$G$中对模式$P$的出现情况进行采样，其中OUT是$P$在$G$中的出现次数，这是在经过$O\left( \left| E\right| \right)$期望时间的预处理之后实现的。在文献[14]中，邓（Deng）等人阐明了如何部署任意连接采样算法来执行子图采样；他们的方法能保证与文献[16]有相同的效果，但会有一个$\widetilde{O}\left( 1\right)$的系数。文献$\left\lbrack  {{14},{16}}\right\rbrack$中的方法适用于无向和有向两种场景。

### 1.3 Our Results

### 1.3 我们的成果

For any join $\mathcal{Q}$ with an acyclic set $\mathrm{{DC}}$ of degree constraints,we will demonstrate in Section 3 how to extract a uniformly random sample from join(Q)in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,following an initial preprocessing of $O\left( \mathrm{{IN}}\right)$ expected time. This performance is favorable when compared to the recent results of $\left\lbrack  {{14},{26}}\right\rbrack$ (reviewed in Section 1.2),which examined settings where DC consists only of cardinality constraints and is therefore trivially acyclic. As polymat(DC) is at most but can be substantially lower than ${AGM}\left( \mathcal{Q}\right)$ ,our guarantees are never worse,but can be considerably better,than those in $\left\lbrack  {{14},{26}}\right\rbrack$ .

对于任何具有无环度约束集$\mathrm{{DC}}$的连接$\mathcal{Q}$，我们将在第3节中展示如何在经过$O\left( \mathrm{{IN}}\right)$期望时间的初始预处理后，在$O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$期望时间内从连接(Q)中提取一个均匀随机样本。与文献$\left\lbrack  {{14},{26}}\right\rbrack$（在1.2节中回顾）的最新成果相比，这一性能更优，文献$\left\lbrack  {{14},{26}}\right\rbrack$研究的是度约束集（DC）仅由基数约束组成，因此显然是无环的情况。由于多面体（polymat(DC)）至多为但可能远低于${AGM}\left( \mathcal{Q}\right)$，我们的保证结果绝不会比文献$\left\lbrack  {{14},{26}}\right\rbrack$中的差，而且可能会好得多。

What if DC is cyclic? An idea, proposed in [30], is to discard enough constraints to make the remaining set ${\mathrm{{DC}}}^{\prime }$ of constraints acyclic (while ensuring $\mathcal{Q} \vDash  {\mathrm{{DC}}}^{\prime }$ ). Our algorithm can then be applied to draw a sample in $O\left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ time. However,this can be unsatisfactory because polymat $\left( {\mathrm{{DC}}}^{\prime }\right)$ can potentially be much larger than polymat(DC).

如果度约束集（DC）是有环的怎么办？文献[30]提出了一个想法，即舍弃足够多的约束，使剩余的约束集${\mathrm{{DC}}}^{\prime }$无环（同时确保$\mathcal{Q} \vDash  {\mathrm{{DC}}}^{\prime }$）。然后我们的算法就可以在$O\left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$时间内抽取样本。然而，这可能并不令人满意，因为多面体$\left( {\mathrm{{DC}}}^{\prime }\right)$可能会比多面体（polymat(DC)）大得多。

Our next contribution is to prove that, interestingly, the issue does not affect subgraph listing/sampling. Consider first directed subgraph listing,defined by a pattern graph $P$ and a data graph $G$ where every vertex has an out-degree at most $\lambda$ . This problem can be converted to a join $\mathcal{Q}$ on binary relations,which is associated with a set DC of degree constraints such that the constraint dependency graph ${G}_{\mathrm{{DC}}}$ is exactly $P$ . Consequently,whenever $P$ contains a cycle,so does ${G}_{\mathrm{{DC}}}$ ,making DC cyclic. Nevertheless,we will demonstrate in Section 4 the existence of an acyclic set ${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$ ensuring $Q \vDash  {\mathrm{{DC}}}^{\prime }$ and polymat $\left( \mathrm{{DC}}\right)  = \Theta \left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$ . This "magical" ${\mathrm{{DC}}}^{\prime }$ has an immediate implication: Ngo’s join algorithm in [30],when applied to $Q$ and ${\mathrm{{DC}}}^{\prime }$ directly, already solves directed subgraph listing optimally in $O\left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)  = O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ time. This dramatically simplifies - in terms of both procedure and analysis - an algorithm of Jayaraman et al. [19] (for directed subgraph listing, reviewed in Section 1.2) that has the same guarantees.

我们的下一个贡献是证明，有趣的是，该问题并不影响子图列举/采样。首先考虑有向子图列举问题，它由一个模式图 $P$ 和一个数据图 $G$ 定义，其中每个顶点的出度至多为 $\lambda$。这个问题可以转化为对二元关系的连接操作 $\mathcal{Q}$，该操作与一组度约束集合 DC 相关联，使得约束依赖图 ${G}_{\mathrm{{DC}}}$ 恰好就是 $P$。因此，只要 $P$ 包含一个环，${G}_{\mathrm{{DC}}}$ 也会包含一个环，从而使 DC 具有循环性。然而，我们将在第 4 节中证明存在一个无环集合 ${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$ 能确保 $Q \vDash  {\mathrm{{DC}}}^{\prime }$ 和多面体 $\left( \mathrm{{DC}}\right)  = \Theta \left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$。这个“神奇的” ${\mathrm{{DC}}}^{\prime }$ 有一个直接的推论：Ngo 在文献 [30] 中的连接算法，当直接应用于 $Q$ 和 ${\mathrm{{DC}}}^{\prime }$ 时，已经能在 $O\left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)  = O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ 时间内最优地解决有向子图列举问题。这在过程和分析两方面都极大地简化了 Jayaraman 等人 [19] 的算法（用于有向子图列举，在 1.2 节中回顾），且具有相同的保证。

The same elegance extends to directed subgraph sampling: by applying our new join sampling algorithm to $\mathcal{Q}$ and the "magical" ${\mathrm{{DC}}}^{\prime }$ ,we can sample an occurrence of $P$ in $G$ using $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,after a preprocessing of $O\left( \left| E\right| \right)$ expected time. As polymat(DC)never exceeds but can be much lower than $\operatorname{AGM}\left( \mathcal{Q}\right)  = \Theta \left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$ ,our result compares favorably with the state of the art $\left\lbrack  {{14},{16},{26}}\right\rbrack$ reviewed in Section 1.2.

同样的精妙之处也适用于有向子图采样：通过将我们新的连接采样算法应用于 $\mathcal{Q}$ 和“神奇的” ${\mathrm{{DC}}}^{\prime }$，在经过 $O\left( \left| E\right| \right)$ 期望时间的预处理后，我们可以在 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ 期望时间内对 $G$ 中 $P$ 的一个出现进行采样。由于多面体(DC)的值从不超过但可能远低于 $\operatorname{AGM}\left( \mathcal{Q}\right)  = \Theta \left( {\left| E\right| }^{{\rho }^{ * }\left( P\right) }\right)$，我们的结果与 1.2 节中回顾的现有技术水平相比更具优势。

Undirected subgraph sampling (where both $P$ and $G$ are undirected and each vertex in $G$ has a degree at most $\lambda$ ) is a special version of its directed counterpart and can be settled by a slightly modified version of our directed subgraph sampling algorithm. However, it is possible to do better by harnessing the undirected nature. In Section 5, we will first solve the polymatroid bound into a closed-form expression, which somewhat unexpectedly exhibits a crucial relationship to a well-known graph decomposition method. This relationship motivates a surprisingly simple algorithm for undirected subgraph sampling that offers guarantees analogous to those in the directed scenario.

无向子图采样（其中 $P$ 和 $G$ 都是无向图，且 $G$ 中的每个顶点的度至多为 $\lambda$）是其有向版本的一个特殊情况，可以通过对我们的有向子图采样算法进行略微修改来解决。然而，利用无向图的特性有可能做得更好。在第 5 节中，我们将首先将多面体边界求解为一个封闭形式的表达式，该表达式出人意料地与一种著名的图分解方法呈现出关键关系。这种关系启发了一种用于无向子图采样的极其简单的算法，该算法提供了与有向场景中类似的保证。

By virtue of the power of sampling, our findings have further implications on other fundamental problems including output-size estimation, output permutation, and small-delay enumeration. We will elaborate on the details in Section 6.

凭借采样的强大能力，我们的发现对其他基本问题（包括输出大小估计、输出排列和小延迟枚举）有进一步的影响。我们将在第 6 节中详细阐述这些细节。

## 2 Preliminaries

## 2 预备知识

Set Functions,Polymatroid Bounds,and Modular Bounds. Suppose that $\mathcal{S}$ is a finite set. We refer to a function $h : {2}^{\mathcal{S}} \rightarrow  {\mathbb{R}}_{ \geq  0}$ as a set function over $\mathcal{S}$ ,where ${\mathbb{R}}_{ \geq  0}$ is the set of non-negative

集合函数、多面体边界和模边界。假设 $\mathcal{S}$ 是一个有限集。我们将函数 $h : {2}^{\mathcal{S}} \rightarrow  {\mathbb{R}}_{ \geq  0}$ 称为 $\mathcal{S}$ 上的集合函数，其中 ${\mathbb{R}}_{ \geq  0}$ 是非负实数集

real values. Such a function $h$ is said to be

这样的函数 $h$ 被称为

- zero-grounded if $h\left( \varnothing \right)  = 0$ ;

- 如果 $h\left( \varnothing \right)  = 0$ ，则为零基函数；

- monotone if $h\left( \mathcal{X}\right)  \leq  h\left( \mathcal{Y}\right)$ for all $\mathcal{X},\mathcal{Y}$ satisfying $\mathcal{X} \subseteq  \mathcal{Y} \subseteq  \mathcal{S}$ ;

- 若对于所有满足 $\mathcal{X} \subseteq  \mathcal{Y} \subseteq  \mathcal{S}$ 的 $\mathcal{X},\mathcal{Y}$ 都有 $h\left( \mathcal{X}\right)  \leq  h\left( \mathcal{Y}\right)$，则为单调的；

- modular if $h\left( \mathcal{X}\right)  = \mathop{\sum }\limits_{{A \in  \mathcal{X}}}h\left( {\{ A\} }\right)$ holds for any $\mathcal{X} \subseteq  \mathcal{S}$ ;

- 若对于任意 $\mathcal{X} \subseteq  \mathcal{S}$ 都有 $h\left( \mathcal{X}\right)  = \mathop{\sum }\limits_{{A \in  \mathcal{X}}}h\left( {\{ A\} }\right)$ 成立，则为模的（模块化的）；

- submodular if $h\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + h\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  \leq  h\left( \mathcal{X}\right)  + h\left( \mathcal{Y}\right)$ holds for any $\mathcal{X},\mathcal{Y} \subseteq  \mathcal{S}$ .

- 若对于任意 $\mathcal{X},\mathcal{Y} \subseteq  \mathcal{S}$ 都有 $h\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + h\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  \leq  h\left( \mathcal{X}\right)  + h\left( \mathcal{Y}\right)$ 成立，则为次模的（子模块化的）。

Define:

定义：

${\mathrm{M}}_{\mathcal{S}} =$ the set of modular set functions over $\mathcal{S}$

${\mathrm{M}}_{\mathcal{S}} =$ 是定义在 $\mathcal{S}$ 上的模集函数（模块化集合函数）的集合

${\Gamma }_{\mathcal{S}} =$ the set of set functions over $\mathcal{S}$ that are zero-grounded,monotone,submodular

${\Gamma }_{\mathcal{S}} =$ 是定义在 $\mathcal{S}$ 上的零基、单调且次模（子模块化）的集函数的集合

Note that every modular function must be zero-grounded and monotone. Clearly, ${\mathrm{M}}_{\mathcal{S}} \subseteq  {\Gamma }_{\mathcal{S}}$ .

注意，每个模函数（模块化函数）必定是零基且单调的。显然，${\mathrm{M}}_{\mathcal{S}} \subseteq  {\Gamma }_{\mathcal{S}}$。

Consider $\mathcal{C}$ to be a set of triples,each having the form $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$ where $X \subset  \mathcal{Y} \subseteq  \mathcal{S}$ and ${N}_{\mathcal{Y} \mid  \mathcal{X}} \geq  1$ is an integer. We will refer to $\mathcal{C}$ as a rule collection over $\mathcal{S}$ and to each triple therein as a rule. Intuitively, the presence of a rule collection is to instruct us to focus only on certain restricted set functions. Formally, these are the set functions in:

将 $\mathcal{C}$ 视为一个三元组的集合，每个三元组的形式为 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)$，其中 $X \subset  \mathcal{Y} \subseteq  \mathcal{S}$ 且 ${N}_{\mathcal{Y} \mid  \mathcal{X}} \geq  1$ 是一个整数。我们将 $\mathcal{C}$ 称为定义在 $\mathcal{S}$ 上的规则集合，其中的每个三元组称为一条规则。直观地说，规则集合的存在是为了指示我们仅关注某些受限的集函数。形式上，这些集函数属于：

$$
{\mathcal{H}}_{\mathcal{C}} = \left\{  {\operatorname{set}\text{ function }h\text{ over }\mathcal{S} \mid  h\left( \mathcal{Y}\right)  - h\left( \mathcal{X}\right)  \leq  \log {N}_{\mathcal{Y} \mid  \mathcal{X}},\;\forall \left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathcal{C}}\right\}  . \tag{3}
$$

The polymatroid bound of $\mathcal{C}$ can now be defined as

现在可以将 $\mathcal{C}$ 的多面体边界（多拟阵边界）定义为

$$
\operatorname{polymat}\left( \mathcal{C}\right)  = {\exp }_{2}\left( {\mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{S}} \cap  {\mathcal{H}}_{\mathcal{C}}}}h\left( \mathcal{S}\right) }\right) . \tag{4}
$$

Recall that ${\exp }_{2}\left( x\right)  = {2}^{x}$ . Similarly,the modular bound of $\mathcal{C}$ is defined as

回顾 ${\exp }_{2}\left( x\right)  = {2}^{x}$。类似地，$\mathcal{C}$ 的模边界（模块化边界）定义为

$$
\operatorname{modular}\left( \mathcal{C}\right)  = {\exp }_{2}\left( {\mathop{\max }\limits_{{h \in  {\mathbb{M}}_{\mathcal{S}} \cap  {\mathcal{H}}_{\mathcal{C}}}}h\left( \mathcal{S}\right) }\right) . \tag{5}
$$

Join Output Size Bounds. Let us fix a join $\mathcal{Q}$ whose schema graph is $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ . Suppose that $\mathcal{Q}$ is consistent with a set $\mathrm{{DC}}$ of degree constraints,i.e., $\mathcal{Q} \vDash  \mathrm{{DC}}$ . As explained in Section 1.1,we follow the convention that each relation of $\mathcal{Q}$ implicitly inserts a cardinality constraint (i.e.,a special degree constraint) to DC. Note that the set DC is merely a rule collection over $\mathcal{V}$ . The following lemma was established by Khamis et al. [25]:

连接输出大小边界。让我们固定一个连接 $\mathcal{Q}$，其模式图为 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$。假设 $\mathcal{Q}$ 与一组度约束 $\mathrm{{DC}}$ 一致，即 $\mathcal{Q} \vDash  \mathrm{{DC}}$。如第 1.1 节所述，我们遵循这样的约定：$\mathcal{Q}$ 的每个关系都会隐式地向 DC 插入一个基数约束（即一种特殊的度约束）。注意，集合 DC 仅仅是定义在 $\mathcal{V}$ 上的一个规则集合。以下引理由 Khamis 等人 [25] 建立：

Lemma 2.1 ([25]). The output size OUT of $\mathcal{Q}$ is at most polymat(DC),i.e.,the polymatroid bound of DC (as defined in (4)).

引理 2.1 ([25])。$\mathcal{Q}$ 的输出大小 OUT 至多为 polymat(DC)，即 DC 的多面体边界（多拟阵边界）（如 (4) 中所定义）。

How about $\operatorname{modular}\left( \mathrm{{DC}}\right)$ ,i.e.,the modular bound of $\mathcal{V}$ ? As ${\mathrm{M}}_{\mathcal{V}} \subseteq  {\Gamma }_{\mathcal{V}}$ ,we have $\operatorname{modular}\left( \mathrm{{DC}}\right)  \leq$ polymat(DC) and the inequality can be strict in general. However, an exception arises when DC is acyclic, as proved in [30]:

$\operatorname{modular}\left( \mathrm{{DC}}\right)$ 即 $\mathcal{V}$ 的模边界（模块化边界）情况如何呢？由于 ${\mathrm{M}}_{\mathcal{V}} \subseteq  {\Gamma }_{\mathcal{V}}$，我们有 $\operatorname{modular}\left( \mathrm{{DC}}\right)  \leq$ ≤ polymat(DC)，并且一般情况下该不等式可能是严格的。然而，当 DC 是无环的时会出现例外情况，如 [30] 中所证明：

Lemma 2.2 ( [30]). When DC is acyclic, it always holds that modular(DC) = polymat(DC), namely, $\mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)  = \mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right) .$

引理2.2（[30]）。当有向图DC（Directed Graph DC）无环时，始终有模块化（modular）(DC) = 多拟阵（polymatroid）(DC)，即$\mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)  = \mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right) .$

As a corollary, when DC is acyclic, the value of modular(DC) always serves as an upper bound of OUT. In our technical development,we will need to analyze the set functions ${h}^{ * } \in  {\Gamma }_{\mathcal{V}}$ that realize the polymatriod bound,i.e., ${h}^{ * }\left( \mathcal{V}\right)  = \mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ . A crucial advantage provided by Lemma 2.2 is that we can instead scrutinize those set functions ${h}^{ * } \in  {\mathrm{M}}_{\mathcal{V}}$ realizing the modular

作为一个推论，当有向图（DC）无环时，模函数值（modular(DC)）始终是输出（OUT）的上界。在我们的技术推导过程中，我们需要分析实现多拟阵界的集合函数${h}^{ * } \in  {\Gamma }_{\mathcal{V}}$，即${h}^{ * }\left( \mathcal{V}\right)  = \mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$。引理2.2提供的一个关键优势是，我们可以转而仔细研究那些实现模函数的集合函数${h}^{ * } \in  {\mathrm{M}}_{\mathcal{V}}$

bound,i.e., ${h}^{ * }\left( \mathcal{V}\right)  = \mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ . Compared to their submodular counterparts,modular set functions exhibit more regularity because every $h \in  {\mathrm{M}}_{\mathcal{V}}$ is fully determined by its value $h\left( {\{ A\} }\right)$ on each individual attribute $A \in  \mathcal{V}$ . In particular,for any $h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}$ ,it holds true that $h\left( \mathcal{Y}\right)  - h\left( \mathcal{X}\right)  = \mathop{\sum }\limits_{{A \in  \mathcal{Y} - \mathcal{X}}}h\left( A\right)$ for any $\mathcal{X} \subset  \mathcal{Y} \subseteq  \mathcal{V}$ . If we associate each $A \in  \mathcal{V}$ with a variable ${\nu }_{A}$ , then $\mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ - hence,also $\mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ - is precisely the optimal value of the following LP:

边界，即 ${h}^{ * }\left( \mathcal{V}\right)  = \mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ 。与次模函数（submodular function）相比，模集函数（modular set function）表现出更强的规律性，因为每个 $h \in  {\mathrm{M}}_{\mathcal{V}}$ 完全由其在每个单独属性 $A \in  \mathcal{V}$ 上的值 $h\left( {\{ A\} }\right)$ 决定。特别地，对于任何 $h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}$ ，对于任何 $\mathcal{X} \subset  \mathcal{Y} \subseteq  \mathcal{V}$ ，都有 $h\left( \mathcal{Y}\right)  - h\left( \mathcal{X}\right)  = \mathop{\sum }\limits_{{A \in  \mathcal{Y} - \mathcal{X}}}h\left( A\right)$ 成立。如果我们将每个 $A \in  \mathcal{V}$ 与一个变量 ${\nu }_{A}$ 关联起来，那么 $\mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ ——因此， $\mathop{\max }\limits_{{h \in  {\Gamma }_{\mathcal{V}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right)$ 也是——恰好是以下线性规划（LP）的最优值：

modular LP max $\mathop{\sum }\limits_{{A \in  \mathcal{V}}}{\nu }_{A}$ subject to

模块化线性规划（LP）最大值 $\mathop{\sum }\limits_{{A \in  \mathcal{V}}}{\nu }_{A}$，约束条件为

$$
\begin{array}{ll} \mathop{\sum }\limits_{{A \in  \mathcal{Y} - \mathcal{X}}}{\nu }_{A} \leq  \log {N}_{\mathcal{Y} \mid  \mathcal{X}} & \forall \left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}} \\  {\nu }_{A} \geq  0 & \forall A \in  \mathcal{V} \end{array}
$$

We will also need to work with the LP’s dual. Specifically,if we associate a variable ${\delta }_{\mathcal{Y} \mid  \mathcal{X}}$ for every degree constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ ,then the dual $\mathrm{{LP}}$ is:

我们还需要处理线性规划（LP）的对偶问题。具体而言，如果我们为每个度约束 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ 关联一个变量 ${\delta }_{\mathcal{Y} \mid  \mathcal{X}}$，那么对偶问题 $\mathrm{{LP}}$ 为：

dual modular LP $\min \mathop{\sum }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \cdot  \log {N}_{\mathcal{Y} \mid  \mathcal{X}}$ subject to

对偶模块化线性规划 $\min \mathop{\sum }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \cdot  \log {N}_{\mathcal{Y} \mid  \mathcal{X}}$ 满足

$$
\begin{array}{l} \mathop{\sum }\limits_{\substack{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y}} \mid  \mathcal{X}}\right)  \in  \mathrm{{DC}}} \\  {A \in  \mathcal{Y} - \mathcal{X}} }}{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \geq  1\;\forall A \in  \mathcal{V} \\  {\delta }_{\mathcal{Y} \mid  \mathcal{X}} \geq  0\;\forall \left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}} \end{array}
$$

## 3 Join Sampling under Acyclic Degree Dependency

## 3 无环度依赖下的连接采样

This section serves as a proof of our first main result:

本节用于证明我们的第一个主要结果：

Theorem 3.1. For any join $\mathcal{Q}$ consistent with an acyclic set $\mathrm{{DC}}$ of degree constraints,we can build in $O\left( \mathrm{{IN}}\right)$ expected time a data structure that supports each join sampling operation in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,where $\mathrm{{IN}}$ and OUT are the input and out sizes of $\mathcal{Q}$ ,respectively,and polymat(DC)is the polymatroid bound of $\mathrm{{DC}}$ .

定理3.1。对于与无环度约束集 $\mathrm{{DC}}$ 一致的任何连接 $\mathcal{Q}$，我们可以在 $O\left( \mathrm{{IN}}\right)$ 期望时间内构建一个数据结构，该结构支持每次连接采样操作的期望时间为 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$，其中 $\mathrm{{IN}}$ 和 OUT 分别是 $\mathcal{Q}$ 的输入和输出大小，并且 polymat(DC) 是 $\mathrm{{DC}}$ 的多拟阵界。

Basic Definitions. Let $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ be the schema graph of $\mathcal{Q}$ ,and ${G}_{\mathrm{{DC}}}$ be the constraint dependency graph determined by DC. For each hyperedge $F \in  \mathcal{E}$ ,we denote by ${R}_{F}$ the relation whose schema corresponds to $F$ . Recall that every constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ is guarded by at least one relation in $\mathcal{Q}$ . Among them,we arbitrarily designate one relation as the constraint’s main guard,whose schema is represented as $F\left( {\mathcal{X},\mathcal{Y}}\right)$ (the main guard can then be conveniently identified as ${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }$ ).

基本定义。设 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ 为 $\mathcal{Q}$ 的模式图，${G}_{\mathrm{{DC}}}$ 为由 DC 确定的约束依赖图。对于每条超边 $F \in  \mathcal{E}$，我们用 ${R}_{F}$ 表示其模式对应于 $F$ 的关系。回顾一下，每个约束 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ 都由 $\mathcal{Q}$ 中的至少一个关系所守护。在这些关系中，我们任意指定一个关系作为该约束的主守护关系，其模式表示为 $F\left( {\mathcal{X},\mathcal{Y}}\right)$（这样就可以方便地将主守护关系标识为 ${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }$）。

Set $k = \left| \mathcal{V}\right|$ . As ${G}_{\mathrm{{DC}}}$ is a DAG (acyclic directed graph),we can order its $k$ vertices (i.e., attributes) into a topological order: ${A}_{1},{A}_{2},\ldots ,{A}_{k}$ . For each $i \in  \left\lbrack  k\right\rbrack$ ,define ${\mathcal{V}}_{i} = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{i}}\right\}$ ; specially,define ${\mathcal{V}}_{0} = \varnothing$ . For any $i \in  \left\lbrack  k\right\rbrack$ ,define

设 $k = \left| \mathcal{V}\right|$。由于 ${G}_{\mathrm{{DC}}}$ 是一个有向无环图（DAG），我们可以将其 $k$ 个顶点（即属性）按拓扑顺序排列：${A}_{1},{A}_{2},\ldots ,{A}_{k}$。对于每个 $i \in  \left\lbrack  k\right\rbrack$，定义 ${\mathcal{V}}_{i} = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{i}}\right\}$；特别地，定义 ${\mathcal{V}}_{0} = \varnothing$。对于任何 $i \in  \left\lbrack  k\right\rbrack$，定义

$$
\mathrm{{DC}}\left( {A}_{i}\right)  = \left\{  {\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}} \mid  {A}_{i} \in  \mathcal{Y} - \mathcal{X}}\right\}   \tag{6}
$$

Fix an arbitrary $i \in  \left\lbrack  k\right\rbrack$ and an arbitrary constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right)$ . Given a tuple $\mathbf{w}$ over ${\mathcal{V}}_{i - 1}$ (note: if $i = 1$ ,then ${\mathcal{V}}_{i - 1} = \varnothing$ and $\mathbf{w}$ is a null tuple) and a value $a \in  \mathbf{{dom}}$ ,we define a "relative degree" for $a$ as:

固定任意的 $i \in  \left\lbrack  k\right\rbrack$ 和任意的约束 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right)$。给定一个在 ${\mathcal{V}}_{i - 1}$ 上的元组 $\mathbf{w}$（注意：如果 $i = 1$，则 ${\mathcal{V}}_{i - 1} = \varnothing$ 且 $\mathbf{w}$ 为空元组）和一个值 $a \in  \mathbf{{dom}}$，我们为 $a$ 定义一个“相对度”如下：

$$
{\operatorname{reldeg}}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)  = \frac{\left| {\sigma }_{{A}_{i} = a}\left( {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  \mathbf{w}\right) \right) \right| }{\left| {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  \mathbf{w}\right) \right| } \tag{7}
$$

where $\sigma$ and $\ltimes$ are the standard selection and semi-join operators in relational algebra,respectively. To understand the intuition behind relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)$ ,imagine drawing a tuple $\mathbf{u}$ from ${\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }\right.  \ltimes$

其中 $\sigma$ 和 $\ltimes$ 分别是关系代数（relational algebra）中的标准选择运算符和半连接运算符。为了理解 relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)$ 的原理，想象从 ${\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }\right.  \ltimes$ 中抽取一个元组 $\mathbf{u}$

---

## ADC-sample

## 模数转换采样（ADC - sample）

0. ${A}_{1},{A}_{2},\ldots ,{A}_{k} \leftarrow$ a topological order of ${G}_{\mathrm{{DC}}}$

0. ${A}_{1},{A}_{2},\ldots ,{A}_{k} \leftarrow$ 是 ${G}_{\mathrm{{DC}}}$ 的一个拓扑排序

1. ${\mathbf{w}}_{0} \leftarrow$ a null tuple

1. ${\mathbf{w}}_{0} \leftarrow$ 是一个空元组

2. for $i = 1$ to $k$ do

2. 对于从 $i = 1$ 到 $k$ 执行

3. pick a constraint $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)$ uniformly at random from $\mathrm{{DC}}\left( {A}_{i}\right)$

3. 从 $\mathrm{{DC}}\left( {A}_{i}\right)$ 中均匀随机地选取一个约束 $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)$

4. ${\mathbf{u}}^{ \circ  } \leftarrow$ a tuple chosen uniformly at random from ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$

4. ${\mathbf{u}}^{ \circ  } \leftarrow$ 是从 ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ 中均匀随机选取的一个元组

	/* note: if $i = 1$ ,then ${R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1} = {R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } *$ /

	/* 注意：如果 $i = 1$ ，那么 ${R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1} = {R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } *$ */

5. $\;{a}_{i} \leftarrow  {\mathbf{u}}^{ \circ  }\left( {A}_{i}\right)$

6. if $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)  \neq$ constrain ${t}_{i - 1}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right)$ then declare failure

6. 如果 $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)  \neq$ 约束 ${t}_{i - 1}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right)$ ，则宣告失败

7. $\;{\mathbf{w}}_{i} \leftarrow$ the tuple over ${\mathcal{V}}_{i}$ formed by concatenating ${\mathbf{w}}_{i - 1}$ with ${a}_{i}$

7. $\;{\mathbf{w}}_{i} \leftarrow$ 是通过将 ${\mathbf{w}}_{i - 1}$ 与 ${a}_{i}$ 连接而形成的关于 ${\mathcal{V}}_{i}$ 的元组

8. declare failure with probability $1 - {p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)$ ,where ${p}_{\text{pass }}$ is given in (12)

8. 以概率 $1 - {p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)$ 宣告失败，其中 ${p}_{\text{pass }}$ 在 (12) 中给出

9. if ${\mathbf{w}}_{k}\left\lbrack  F\right\rbrack   \in  {R}_{F}$ for $\forall F \in  \mathcal{E}$ then /* that is, ${\mathbf{w}}_{k} \in  \operatorname{join}\left( \mathcal{Q}\right)$ */

9. 如果对于 $\forall F \in  \mathcal{E}$ 有 ${\mathbf{w}}_{k}\left\lbrack  F\right\rbrack   \in  {R}_{F}$ ，那么 /* 即 ${\mathbf{w}}_{k} \in  \operatorname{join}\left( \mathcal{Q}\right)$ */

10. return ${\mathbf{w}}_{k}$

10. 返回 ${\mathbf{w}}_{k}$

---

Figure 1: Our sampling algorithm

图 1：我们的采样算法

$\mathbf{w}$ ) uniformly at random; then relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)$ is the probability to see $\mathbf{u}\left( {A}_{i}\right)  = a$ . Given a tuple $\mathbf{w}$ over ${\mathcal{V}}_{i - 1}$ and a value $a \in  \mathbf{{dom}}$ ,define

$\mathbf{w}$ ) 均匀随机地；那么 relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)$ 是观察到 $\mathbf{u}\left( {A}_{i}\right)  = a$ 的概率。给定一个关于 ${\mathcal{V}}_{i - 1}$ 的元组 $\mathbf{w}$ 和一个值 $a \in  \mathbf{{dom}}$ ，定义

$$
{\operatorname{reldeg}}_{i}^{ * }\left( {\mathbf{w},a}\right)  = \mathop{\max }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \operatorname{DC}\left( {A}_{i}\right) }}{\operatorname{reldeg}}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)  \tag{8}
$$

$$
{\operatorname{constraint}}_{i}^{ * }\left( {\mathbf{w},a}\right)  = \underset{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right) }{\arg \max }{\operatorname{reldeg}}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right) . \tag{9}
$$

Specifically,constraint ${t}_{i}^{ * }\left( {\mathbf{w},a}\right)$ returns the constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right)$ satisfying the condition relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)  = {\operatorname{reldeg}}_{i}^{ * }\left( {\mathbf{w},a}\right)$ . If more than one constraint meets this condition,define ${\text{constraint}}_{i}^{ * }\left( {\mathbf{w},a}\right)$ to be an arbitrary one among those constraints.

具体而言，约束条件 ${t}_{i}^{ * }\left( {\mathbf{w},a}\right)$ 返回满足关系条件 relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)  = {\operatorname{reldeg}}_{i}^{ * }\left( {\mathbf{w},a}\right)$ 的约束 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right)$。如果有多个约束满足此条件，则将 ${\text{constraint}}_{i}^{ * }\left( {\mathbf{w},a}\right)$ 定义为这些约束中的任意一个。

Henceforth,we will fix an arbitrary optimal solution $\left\{  {{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \mid  \left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}\right\}$ to the dual modular LP in Section 2. Thus:

此后，我们将固定第2节中对偶模线性规划（dual modular LP）的任意一个最优解 $\left\{  {{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \mid  \left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}\right\}$。因此：

$$
\mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{N}_{\mathcal{Y} \mid  \mathcal{X}}^{\delta \mathcal{Y} \mid  \mathcal{X}} = {\exp }_{2}\left( {\mathop{\sum }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \cdot  \log {N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  = {\exp }_{2}\left( {\mathop{\max }\limits_{{h \in  {\mathrm{M}}_{\mathcal{Y}} \cap  {\mathcal{H}}_{\mathrm{{DC}}}}}h\left( \mathcal{V}\right) }\right) 
$$

$$
\text{(by (5)) = modular(DC)}
$$

$$
\text{(by Lemma 2.2)} = \text{polymat(DC).} \tag{10}
$$

Finally,for any $i \in  \left\lbrack  {0,k}\right\rbrack$ and any tuple $\mathbf{w}$ over ${\mathcal{V}}_{i}$ ,define:

最后，对于任意 $i \in  \left\lbrack  {0,k}\right\rbrack$ 以及 ${\mathcal{V}}_{i}$ 上的任意元组 $\mathbf{w}$，定义：

$$
{B}_{i}\left( \mathbf{w}\right)  = \mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\left( {\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  \mathbf{w}\right) \right) }^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}. \tag{11}
$$

Two observations will be useful later:

以下两个观察结果后续会很有用：

- If $i = 0$ ,then $\mathbf{w}$ is a null tuple and ${B}_{0}\left( \text{null}\right)  = \mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\left( {\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }\right) \right) }^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}$ ,which is at most $\mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{N}_{\mathcal{Y} \mid  \mathcal{X}}^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}} =$ polymat(DC).

- 如果$i = 0$，那么$\mathbf{w}$是一个空元组，并且${B}_{0}\left( \text{null}\right)  = \mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\left( {\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }\right) \right) }^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}$，它至多是$\mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{N}_{\mathcal{Y} \mid  \mathcal{X}}^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}} =$多集（DC）。

- If $i = k$ and $\mathbf{w} \in  \operatorname{join}\left( \mathcal{Q}\right)$ ,then ${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  \mathbf{w}$ contains exactly one tuple for any $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in$ DC and thus ${B}_{k}\left( \mathbf{w}\right)  = 1$ .

- 如果$i = k$且$\mathbf{w} \in  \operatorname{join}\left( \mathcal{Q}\right)$，那么对于任何$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in$ DC，${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  \mathbf{w}$恰好包含一个元组，因此${B}_{k}\left( \mathbf{w}\right)  = 1$。

Algorithm. Our sampling algorithm, named ADC-sample, is presented in Figure 1. At a high level, it processes one attribute at a time according to the topological order ${A}_{1},{A}_{2},\ldots ,{A}_{k}$ . The for-loop in Lines 2-9 finds a value ${a}_{i}$ for attribute ${A}_{i}\left( {i \in  \left\lbrack  k\right\rbrack  }\right)$ . The algorithm may fail to return anything, but when it succeeds (i.e.,reaching Line 10),the values ${a}_{1},{a}_{2},\ldots ,{a}_{k}$ will make a uniformly random tuple from join(Q).

算法。我们的采样算法名为ADC - 采样，如图1所示。总体而言，它根据拓扑顺序${A}_{1},{A}_{2},\ldots ,{A}_{k}$一次处理一个属性。第2 - 9行的for循环为属性${A}_{i}\left( {i \in  \left\lbrack  k\right\rbrack  }\right)$找到一个值${a}_{i}$。该算法可能无法返回任何结果，但当它成功时（即到达第10行），值${a}_{1},{a}_{2},\ldots ,{a}_{k}$将从连接（Q）中生成一个均匀随机元组。

Next,we explain the details of the for-loop. The loop starts with values ${a}_{1},{a}_{2},\ldots ,{a}_{i - 1}$ already stored in a tuple ${\mathbf{w}}_{i - 1}$ (i.e., ${\mathbf{w}}_{i - 1}\left( {A}_{j}\right)  = {a}_{j}$ for all $j \in  \left\lbrack  {i - 1}\right\rbrack$ ). Line 3 randomly chooses a degree constraint $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)$ from $\mathrm{{DC}}\left( {A}_{i}\right)$ ; see (6). Conceptually,next we identify the main guard ${R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) }$ of this constraint,semi-join the relation with ${\mathbf{w}}_{i - 1}$ ,and project the semi-join result on ${\mathcal{Y}}^{ \circ  }$ to obtain ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ . Then,Line 4 randomly chooses a tuple ${\mathbf{u}}^{ \circ  }$ from ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ and Line 5 takes ${\mathbf{u}}^{ \circ  }\left( {A}_{i}\right)$ as the value of ${a}_{i}$ (note: ${A}_{i} \in  {\mathcal{Y}}^{ \circ  } - {\mathcal{X}}^{ \circ  } \subseteq  {\mathcal{Y}}^{ \circ  }$ ). Physically,however,we do not compute ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ during the sample process. Instead, with proper preprocessing (discussed later),we can acquire the value ${a}_{i}$ in $O\left( 1\right)$ time. Continuing, Line 6 may declare failure and terminate ADC-sample,but if we get past this line, $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)$ must be exactly constrain ${t}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right)$ ; see (9). As clarified later,the check at Line 6 can be performed in $O\left( 1\right)$ time. We now form a tuple ${\mathbf{w}}_{i}$ that takes value ${a}_{j}$ on attribute ${A}_{j}$ for each $j \in  \left\lbrack  i\right\rbrack$ (Line 7). Line 8 allows us to pass with probability

接下来，我们详细解释这个for循环。循环开始时，值 ${a}_{1},{a}_{2},\ldots ,{a}_{i - 1}$ 已存储在元组 ${\mathbf{w}}_{i - 1}$ 中（即，对于所有 $j \in  \left\lbrack  {i - 1}\right\rbrack$ ，有 ${\mathbf{w}}_{i - 1}\left( {A}_{j}\right)  = {a}_{j}$ ）。第3行从 $\mathrm{{DC}}\left( {A}_{i}\right)$ 中随机选择一个度约束 $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)$ ；参见(6)。从概念上讲，接下来我们确定该约束的主保护条件 ${R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) }$ ，将该关系与 ${\mathbf{w}}_{i - 1}$ 进行半连接，并在 ${\mathcal{Y}}^{ \circ  }$ 上对半连接结果进行投影以获得 ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ 。然后，第4行从 ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ 中随机选择一个元组 ${\mathbf{u}}^{ \circ  }$ ，第5行将 ${\mathbf{u}}^{ \circ  }\left( {A}_{i}\right)$ 作为 ${a}_{i}$ 的值（注意： ${A}_{i} \in  {\mathcal{Y}}^{ \circ  } - {\mathcal{X}}^{ \circ  } \subseteq  {\mathcal{Y}}^{ \circ  }$ ）。然而，在物理实现上，我们在采样过程中并不计算 ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ 。相反，通过适当的预处理（稍后讨论），我们可以在 $O\left( 1\right)$ 时间内获取 ${a}_{i}$ 的值。接着，第6行可能会宣告失败并终止ADC采样，但如果我们通过了这一行， $\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)$ 必定恰好约束 ${t}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right)$ ；参见(9)。如稍后所阐明的，第6行的检查可以在 $O\left( 1\right)$ 时间内完成。现在，我们形成一个元组 ${\mathbf{w}}_{i}$ ，对于每个 $j \in  \left\lbrack  i\right\rbrack$ ，该元组在属性 ${A}_{j}$ 上取值 ${a}_{j}$ （第7行）。第8行使我们能够以一定概率继续执行

$$
{p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)  = \frac{{B}_{i}\left( {\mathbf{w}}_{i}\right) }{{B}_{i - 1}\left( {\mathbf{w}}_{i - 1}\right) } \cdot  \frac{1}{{\operatorname{reldeg}}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}\left( {A}_{i}\right) }\right) } \tag{12}
$$

or otherwise terminate the algorithm by declaring failure. As proved later, ${p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)$ cannot exceed 1 (Lemma 3.2); moreover,this value can be computed in $O\left( 1\right)$ time. The overall execution time of ADC-sample is constant.

否则通过宣告失败来终止算法。如稍后所证明的， ${p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)$ 不会超过1（引理3.2）；此外，这个值可以在 $O\left( 1\right)$ 时间内计算得出。ADC采样的总体执行时间是常数。

Analysis. Next we prove that the value in (12) serves as a legal probability value.

分析。接下来，我们证明式(12)中的值可作为合法的概率值。

Lemma 3.2. For every $i \in  \left\lbrack  k\right\rbrack$ ,we have ${p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)  \leq  1$ .

引理3.2。对于任意$i \in  \left\lbrack  k\right\rbrack$，我们有${p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)  \leq  1$。

Proof. Consider an arbitrary constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right)$ . Recall that ADC-sample processes the attributes by the topological order ${A}_{1},\ldots ,{A}_{k}$ . In the constrained dependency graph ${G}_{\mathrm{{DC}}}$ ,every attribute of $\mathcal{X}$ has an out-going edge to ${A}_{i}$ . Hence,all the attributes in $\mathcal{X}$ must be processed prior to ${A}_{i}$ . This implies that all the tuples in ${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}$ must have the same projection on $\mathcal{X}$ . Therefore, ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ equals $\left| {{\Pi }_{\mathcal{Y}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right) }\right|$ . By the same reasoning, ${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right)$ equals $\left| {{\Pi }_{\mathcal{Y}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right) }\right|$ . We thus have:

证明。考虑任意约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right)$。回顾一下，ADC采样（ADC-sample）按照拓扑顺序${A}_{1},\ldots ,{A}_{k}$处理属性。在约束依赖图${G}_{\mathrm{{DC}}}$中，$\mathcal{X}$的每个属性都有一条指向${A}_{i}$的出边。因此，$\mathcal{X}$中的所有属性必须在${A}_{i}$之前处理。这意味着${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}$中的所有元组在$\mathcal{X}$上的投影必须相同。因此，${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$等于$\left| {{\Pi }_{\mathcal{Y}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right) }\right|$。同理，${\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right)$等于$\left| {{\Pi }_{\mathcal{Y}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right) }\right|$。因此，我们有：

$$
\frac{{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right) }{{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right) } = \frac{\left| {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}\right) \right| }{\left| {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}\right) \right| }
$$

$$
 = \frac{\left| {\sigma }_{{A}_{i} = {a}_{i}}\left( {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}\right) \right) \right| }{\left| {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}\right) \right| }
$$

$$
 = {\operatorname{reldeg}}_{i,\mathcal{X},\mathcal{Y}}\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right) 
$$

$$
 \leq  {\text{reldeg}}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right) \text{.} \tag{13}
$$

On the other hand,for any constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \notin  \mathrm{{DC}}\left( {A}_{i}\right)$ ,it trivially holds that

另一方面，对于任意约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \notin  \mathrm{{DC}}\left( {A}_{i}\right)$，显然有

$$
{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right)  \leq  {\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)  \tag{14}
$$

because ${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}$ is a subset of ${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}$ .

因为${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}$是${R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}$的子集。

We can now derive

我们现在可以推导出

$$
{p}_{\text{pass }}\left( {i,{\mathbf{w}}_{i - 1},{\mathbf{w}}_{i}}\right)  = \frac{1}{\operatorname{relde}{g}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right) }\mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}}}{\left( \frac{{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right) }{{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right) }\right) }^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}
$$

$$
\text{by (14))} \leq  \frac{1}{\operatorname{relde}{g}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right) }\mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right) }}{\left( \frac{{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i}}\right) }{{\deg }_{\mathcal{Y} \mid  \mathcal{X}}\left( {{R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right) }\right) }^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}
$$

$$
\text{by (13))} \leq  \frac{1}{{\operatorname{reldeg}}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},{a}_{i}}\right) }\mathop{\prod }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right) }}{\operatorname{reldeg}}_{i}^{ * }{\left( {\mathbf{w}}_{i - 1},{a}_{i}\right) }^{{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}
$$

$$
 = {\operatorname{reldeg}}_{i}^{ * }{\left( {\mathbf{w}}_{i - 1},{a}_{i}\right) }^{\left( {\mathop{\sum }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right) }}{\delta }_{\mathcal{Y} \mid  \mathcal{X}}}\right)  - 1} \leq  1.
$$

The last step used $\mathop{\sum }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right) }}{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \geq  1$ guaranteed by the dual modular LP.

最后一步使用了对偶模块化线性规划（dual modular LP）保证的$\mathop{\sum }\limits_{{\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}\left( {A}_{i}\right) }}{\delta }_{\mathcal{Y} \mid  \mathcal{X}} \geq  1$。

Next,we argue that every result tuple $v \in  \operatorname{join}\left( \mathcal{Q}\right)$ is returned by ADC-sample with the same probability. For this purpose,let us define two random events for each $i \in  \left\lbrack  k\right\rbrack$ :

接下来，我们证明ADC采样以相同的概率返回每个结果元组$v \in  \operatorname{join}\left( \mathcal{Q}\right)$。为此，对于每个$i \in  \left\lbrack  k\right\rbrack$，我们定义两个随机事件：

- event $\mathbf{E}\mathbf{1}\left( i\right)  : \left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)  = {\operatorname{constraint}}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},\mathbf{v}\left( {A}_{i}\right) }\right)$ in the $i$ -th loop of ADC-sample;

- 事件 $\mathbf{E}\mathbf{1}\left( i\right)  : \left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)  = {\operatorname{constraint}}_{i}^{ * }\left( {{\mathbf{w}}_{i - 1},\mathbf{v}\left( {A}_{i}\right) }\right)$ 发生在ADC采样的第 $i$ 次循环中；

- event $\mathbf{{E2}}\left( i\right)$ : Line 8 does not declare failure in the $i$ -th loop of ADC-sample.

- 事件 $\mathbf{{E2}}\left( i\right)$ ：在ADC采样的第 $i$ 次循环中，第8行未声明失败。

The probability for ADC-sample to return $\mathbf{v}$ can be derived as follows.

ADC采样返回 $\mathbf{v}$ 的概率推导如下。

$$
\mathbf{{Pr}}\left\lbrack  {\mathbf{v}\text{ returned }}\right\rbrack   = \mathop{\prod }\limits_{{i = 1}}^{k}\mathbf{{Pr}}\left\lbrack  {{a}_{i} = \mathbf{v}\left( {A}_{i}\right) ,\mathbf{E}\mathbf{1}\left( i\right) ,\mathbf{E}\mathbf{2}\left( i\right)  \mid  {\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack  
$$

(if $i = 1$ ,then ${\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack$ becomes ${\mathbf{w}}_{0} = \mathbf{v}\left\lbrack  \varnothing \right\rbrack$ ,which is vacuously true)

（若 $i = 1$ ，则 ${\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack$ 变为 ${\mathbf{w}}_{0} = \mathbf{v}\left\lbrack  \varnothing \right\rbrack$ ，这显然成立）

$$
 = \mathop{\prod }\limits_{{i = 1}}^{k}\left( {\mathbf{{Pr}}\left\lbrack  {{a}_{i} = \mathbf{v}\left( {A}_{i}\right) ,\mathbf{E}\mathbf{1}\left( i\right)  \mid  {\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack  .}\right. 
$$

$$
\left. \left. {\mathbf{{Pr}}\left\lbrack  {\mathbf{{E2}}\left( i\right)  \mid  \mathbf{{E1}}\left( i\right) ,{a}_{i} = \mathbf{v}\left( {A}_{i}\right) ,{\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack  }\right\rbrack  \right) \text{.} \tag{15}
$$

Observe

观察

$$
\Pr \left\lbrack  {{a}_{i} = \mathbf{v}\left( {A}_{i}\right) ,\mathbf{E}\mathbf{1}\left( i\right)  \mid  {\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack  
$$

$$
 = \Pr \left\lbrack  {\mathbf{{E1}}\left( i\right)  \mid  {\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack   \cdot  \Pr \left\lbrack  {{a}_{i} = \mathbf{v}\left( {A}_{i}\right)  \mid  \mathbf{{E1}}\left( i\right) ,{\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack  
$$

$$
 = \frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| } \cdot  \frac{\left| {\sigma }_{{A}_{i} = \mathbf{v}\left( {A}_{i}\right) }\left( {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  \right) \right) \right| }{\left| {\Pi }_{\mathcal{Y}}\left( {R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  \right) \right| }
$$

$$
\text{(note:}\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  },{N}_{{\mathcal{Y}}^{ \circ  } \mid  {\mathcal{X}}^{ \circ  }}}\right)  = {\operatorname{constraint}}_{i}^{ * }\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  ,\mathbf{v}\left( {A}_{i}\right) }\right) \text{,due to}\mathbf{E}\mathbf{1}\left( i\right) \text{and}{\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  \text{))}
$$

$$
 = \frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| } \cdot  {\operatorname{reldeg}}_{i,{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  ,\mathbf{v}\left( {A}_{i}\right) }\right)  = \frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| } \cdot  {\operatorname{reldeg}}_{i}^{ * }\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  ,\mathbf{v}\left( {A}_{i}\right) }\right) . \tag{16}
$$

On the other hand:

另一方面：

$$
\Pr \left\lbrack  {\mathbf{{E2}}\left( i\right)  \mid  \mathbf{{E1}}\left( i\right) ,{a}_{i} = \mathbf{v}\left( {A}_{i}\right) ,{\mathbf{w}}_{i - 1} = \mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right\rbrack  
$$

$$
 = {p}_{\text{pass }}\left( {i,\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  ,\mathbf{v}\left\lbrack  {\mathcal{V}}_{i}\right\rbrack  }\right) 
$$

$$
\text{(by (12))} = \frac{{B}_{i}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i}\right\rbrack  }\right) }{{B}_{i - 1}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right) } \cdot  \frac{1}{{\operatorname{reldeg}}_{i}^{ * }\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  ,\mathbf{v}\left( {A}_{i}\right) }\right) }\text{.} \tag{17}
$$

Plugging (16) and (17) into (15) yields

将(16)和(17)代入(15)可得

$$
\Pr \left\lbrack  {\mathbf{v}\text{ returned }}\right\rbrack   = \mathop{\prod }\limits_{{i = 1}}^{k}\frac{{B}_{i}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i}\right\rbrack  }\right) }{{B}_{i - 1}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{i - 1}\right\rbrack  }\right) } \cdot  \frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| } = \frac{{B}_{k}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{k}\right\rbrack  }\right) }{{B}_{0}\left( {\mathbf{v}\left\lbrack  {\mathcal{V}}_{0}\right\rbrack  }\right) } \cdot  \mathop{\prod }\limits_{{i = 1}}^{k}\frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| }
$$

$$
 = \frac{1}{{B}_{0}\left( \text{ null }\right) } \cdot  \mathop{\prod }\limits_{{i = 1}}^{k}\frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| }.
$$

As the above is identical for every $\mathbf{v} \in  \operatorname{join}\left( \mathcal{Q}\right)$ ,we can conclude that each tuple in the join result gets returned by ADC-sample with the same probability. As an immediate corollary, each run of ADC-sample successfully returns a sample from join(Q)with probability

由于上述情况对每个 $\mathbf{v} \in  \operatorname{join}\left( \mathcal{Q}\right)$ 都相同，我们可以得出连接结果中的每个元组被ADC采样以相同概率返回。作为直接推论，ADC采样的每次运行都以一定概率成功从连接(Q)中返回一个样本。

$$
\frac{\mathrm{{OUT}}}{{B}_{0}\left( \text{ null }\right) } \cdot  \mathop{\prod }\limits_{{i = 1}}^{k}\frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| } \geq  \frac{\mathrm{{OUT}}}{\text{ polymat }\left( \mathrm{{DC}}\right) } \cdot  \mathop{\prod }\limits_{{i = 1}}^{k}\frac{1}{\left| \mathrm{{DC}}\left( {A}_{i}\right) \right| } = \Omega \left( \frac{\mathrm{{OUT}}}{\text{ polymat }\left( \mathrm{{DC}}\right) }\right) .
$$

In Appendix A,we will explain how to preprocess the relations of $\mathcal{Q}$ in $O\left( \mathrm{{IN}}\right)$ expected time to ensure that ADC-sample completes in $O\left( 1\right)$ time.

在附录A中，我们将解释如何在 $O\left( \mathrm{{IN}}\right)$ 的期望时间内对 $\mathcal{Q}$ 的关系进行预处理，以确保ADC采样在 $O\left( 1\right)$ 时间内完成。

Performing a Join Sampling Operation. Recall that this operation must either return a uniformly random sample of $\operatorname{join}\left( \mathcal{Q}\right)$ or declare $\operatorname{join}\left( \mathcal{Q}\right)  = \varnothing$ . To support this operation,we execute two threads concurrently. The first thread repeatedly invokes ADC-sample until it successfully returns a sample. The other thread runs Ngo’s algorithm in [30] to compute ${join}\left( \mathcal{Q}\right)$ in full,after which we can declare $\operatorname{join}\left( \mathcal{Q}\right)  \neq  \varnothing$ or sample from $\operatorname{join}\left( \mathcal{Q}\right)$ in constant time. As soon as one thread finishes, we manually terminate the other one.

执行连接采样操作。回顾一下，此操作必须要么返回 $\operatorname{join}\left( \mathcal{Q}\right)$ 的均匀随机样本，要么声明 $\operatorname{join}\left( \mathcal{Q}\right)  = \varnothing$ 。为支持此操作，我们同时执行两个线程。第一个线程反复调用ADC采样，直到成功返回一个样本。另一个线程运行文献[30]中的Ngo算法来完整计算 ${join}\left( \mathcal{Q}\right)$ ，之后我们可以声明 $\operatorname{join}\left( \mathcal{Q}\right)  \neq  \varnothing$ 或在常量时间内从 $\operatorname{join}\left( \mathcal{Q}\right)$ 中采样。一旦一个线程完成，我们就手动终止另一个线程。

This strategy guarantees that the join operation completes in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ time. To explain why,consider first the scenario where $\mathrm{{OUT}} \geq  1$ . In this case,we expect to find a sample with $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\mathrm{{OUT}}}\right)$ repeats of $\mathrm{{ADC}}$ -sample. Hence,the first thread finishes in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\mathrm{{OUT}}}\right)$ expected sample time. On the other hand,if $\mathrm{{OUT}} = 0$ ,the second thread will finish in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ time. This concludes the proof of Theorem 3.1.

此策略保证连接操作在 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ 时间内完成。为解释原因，首先考虑 $\mathrm{{OUT}} \geq  1$ 的情况。在这种情况下，我们期望通过 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\mathrm{{OUT}}}\right)$ 次 $\mathrm{{ADC}}$ 采样找到一个样本。因此，第一个线程在 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\mathrm{{OUT}}}\right)$ 的期望采样时间内完成。另一方面，如果 $\mathrm{{OUT}} = 0$ ，第二个线程将在 $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ 时间内完成。至此，定理3.1证明完毕。

Remarks. When DC has only cardinality constraints (is thus "trivially" acyclic), ADC-sample simplifies into the sampling algorithm of Kim et al. [26]. In retrospect, two main obstacles prevent an obvious extension of their algorithm to an arbitrary acyclic DC. The first is identifying an appropriate way to deal with constraints $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ where $\mathcal{X} \neq  \varnothing$ (such constraints are absent in the degenerated context of [26]). The second obstacle involves determining how to benefit from a topological order (attribute ordering is irrelevant in [26]); replacing the order with a non-topological one may ruin the correctness of ADC-sample.

备注。当依赖约束（DC）仅存在基数约束（因此是“平凡”无环的）时，ADC 采样可简化为 Kim 等人 [26] 提出的采样算法。回顾来看，有两个主要障碍阻碍了将他们的算法直接扩展到任意无环的依赖约束。第一个障碍是找到一种合适的方法来处理约束 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$，其中 $\mathcal{X} \neq  \varnothing$（在文献 [26] 的退化情形中不存在此类约束）。第二个障碍涉及确定如何利用拓扑序（在文献 [26] 中属性排序无关紧要）；若用非拓扑序替代该顺序，可能会破坏 ADC 采样的正确性。

## 4 Directed Subgraph Sampling

## 4 有向子图采样

Given a directed pattern graph $P = \left( {{V}_{P},{E}_{P}}\right)$ and a directed data graph $G = \left( {V,E}\right)$ ,we use $\operatorname{occ}\left( {G,P}\right)$ to represent the set of occurrences of $P$ in $G$ . Every vertex in $G$ has an out-degree at most $\lambda$ . Our goal is to design an algorithm to sample from $\operatorname{occ}\left( {G,P}\right)$ efficiently.

给定一个有向模式图 $P = \left( {{V}_{P},{E}_{P}}\right)$ 和一个有向数据图 $G = \left( {V,E}\right)$，我们用 $\operatorname{occ}\left( {G,P}\right)$ 表示 $P$ 在 $G$ 中的出现集合。$G$ 中的每个顶点的出度至多为 $\lambda$。我们的目标是设计一种算法，以便从 $\operatorname{occ}\left( {G,P}\right)$ 中高效采样。

Let us formulate the "polymatroid bound" for this problem. Given an integer $m \geq  1$ ,an integer $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$ ,and a pattern $P = \left( {{V}_{P},{E}_{P}}\right)$ ,first build a rule collection $\mathcal{C}$ over ${V}_{P}$ as follows: for each edge $\left( {X,Y}\right)  \in  {E}_{P}$ ,add to $\mathcal{C}$ two rules: $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ and $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ . Then,the directed polymatriod bound of $m,\lambda$ ,and $P$ can be defined as

让我们为这个问题构建“多拟阵界”。给定一个整数 $m \geq  1$、一个整数 $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$ 和一个模式 $P = \left( {{V}_{P},{E}_{P}}\right)$，首先按如下方式在 ${V}_{P}$ 上构建一个规则集合 $\mathcal{C}$：对于每条边 $\left( {X,Y}\right)  \in  {E}_{P}$，向 $\mathcal{C}$ 中添加两条规则：$\left( {\varnothing ,\{ X,Y\} ,m}\right)$ 和 $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$。然后，$m,\lambda$ 和 $P$ 的有向多拟阵界可以定义为

$$
{\text{polymat}}_{\text{dir }}\left( {m,\lambda ,P}\right)  = \text{polymat}\left( \mathcal{C}\right)  \tag{18}
$$

where polymat(C)follows the definition in (4).

其中多拟阵（polymat(C)）遵循 (4) 中的定义。

This formulation reflects how directed subgraph listing can be processed as a join. Consider a companion join $\mathcal{Q}$ constructed from $G$ and $P$ as follows. The schema graph of $\mathcal{Q}$ ,denoted as $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ ,is exactly $P = \left( {{V}_{P},{E}_{P}}\right)$ (i.e., $\mathcal{V} = {V}_{P}$ and $\mathcal{E} = {E}_{P}$ ). For every edge $F = \left( {X,Y}\right)  \in  {E}_{P}$ , create a relation ${R}_{F} \in  \mathcal{Q}$ by inserting,for each edge(x,y)in the data graph $G$ ,a tuple $\mathbf{u}$ with $\mathbf{u}\left( X\right)  = x$ and $\mathbf{u}\left( Y\right)  = y$ into ${R}_{F}$ . The rule collection $\mathcal{C}$ can now be regarded as a set DC of degree constraints with which $\mathcal{Q}$ is consistent,i.e., $\mathcal{Q} \vDash  \mathrm{{DC}} = \mathcal{C}$ . The constraint dependence graph ${G}_{\mathrm{{DC}}}$ is precisely $P$ . It is immediate that ${\operatorname{polymat}}_{\text{dir }}\left( {\left| E\right| ,\lambda ,P}\right)  =$ polymat(DC). To find all the occurrences in $\operatorname{occ}\left( {G,P}\right)$ ,it suffices to compute $\operatorname{join}\left( \mathcal{Q}\right)$ . Specifically,every tuple $\mathbf{u} \in  \operatorname{join}\left( \mathcal{Q}\right)$ that uses a distinct value on every attribute in $\mathcal{V}\left( { = {V}_{P}}\right)$ matches a unique occurrence in $\operatorname{occ}\left( {G,P}\right)$ . Conversely, every occurrence in $\operatorname{occ}\left( {G,P}\right)$ matches the same number $c$ of tuples in $\operatorname{join}\left( \mathcal{Q}\right)$ ,where $c \geq  1$ is a constant equal to the number of automorphisms of $P$ . If we denote $\mathrm{{OUT}} = \left| {\operatorname{occ}\left( {G,P}\right) }\right|$ and ${\text{OUT}}_{\mathcal{Q}} = \left| {\text{join}\left( \mathcal{Q}\right) }\right|$ ,it follows that $c \cdot$ OUT $\leq  {\text{OUT}}_{\mathcal{Q}} \leq$ polymat $\left( \text{DC}\right)  = {\text{polymat}}_{\text{dir}}\left( {\left| E\right| ,\lambda ,P}\right)$ .

这种表述反映了有向子图列举如何可以作为连接操作来处理。考虑按如下方式由$G$和$P$构建的一个伴随连接$\mathcal{Q}$。$\mathcal{Q}$的模式图（记为$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$）恰好是$P = \left( {{V}_{P},{E}_{P}}\right)$（即$\mathcal{V} = {V}_{P}$和$\mathcal{E} = {E}_{P}$）。对于每条边$F = \left( {X,Y}\right)  \in  {E}_{P}$，通过在数据图$G$中为每条边(x, y)插入一个元组$\mathbf{u}$（其中$\mathbf{u}\left( X\right)  = x$且$\mathbf{u}\left( Y\right)  = y$）到${R}_{F}$中，来创建一个关系${R}_{F} \in  \mathcal{Q}$。规则集合$\mathcal{C}$现在可以被视为一个度约束集合DC，$\mathcal{Q}$与该集合一致，即$\mathcal{Q} \vDash  \mathrm{{DC}} = \mathcal{C}$。约束依赖图${G}_{\mathrm{{DC}}}$恰好是$P$。显然有${\operatorname{polymat}}_{\text{dir }}\left( {\left| E\right| ,\lambda ,P}\right)  =$多面体(DC)。要找出$\operatorname{occ}\left( {G,P}\right)$中的所有出现情况，只需计算$\operatorname{join}\left( \mathcal{Q}\right)$即可。具体而言，在$\mathcal{V}\left( { = {V}_{P}}\right)$的每个属性上使用不同值的每个元组$\mathbf{u} \in  \operatorname{join}\left( \mathcal{Q}\right)$都与$\operatorname{occ}\left( {G,P}\right)$中的一个唯一出现情况相匹配。反之，$\operatorname{occ}\left( {G,P}\right)$中的每个出现情况都与$\operatorname{join}\left( \mathcal{Q}\right)$中的相同数量$c$的元组相匹配，其中$c \geq  1$是一个等于$P$的自同构数量的常数。如果我们记$\mathrm{{OUT}} = \left| {\operatorname{occ}\left( {G,P}\right) }\right|$和${\text{OUT}}_{\mathcal{Q}} = \left| {\text{join}\left( \mathcal{Q}\right) }\right|$，那么有$c \cdot$输出 $\leq  {\text{OUT}}_{\mathcal{Q}} \leq$ 多面体 $\left( \text{DC}\right)  = {\text{polymat}}_{\text{dir}}\left( {\left| E\right| ,\lambda ,P}\right)$。

The above observation suggests how directed subgraph sampling can be reduced to join sampling. First,sample a tuple $\mathbf{u}$ from join(Q)uniformly at random. Then,check whether $\mathbf{u}\left( A\right)  = \mathbf{u}\left( {A}^{\prime }\right)$ for any two distinct attributes $A,{A}^{\prime } \in  \mathcal{V}$ . If so,declare failure; otherwise,declare success and return the unique occurrence matching the tuple $\mathbf{u}$ . The success probability equals $c \cdot  \mathrm{{OUT}}/{\mathrm{{OUT}}}_{\mathcal{Q}}$ . In a success event,every occurrence in $\operatorname{occ}\left( {G,P}\right)$ has the same probability to be returned.

上述观察结果表明了有向子图采样如何可以简化为连接采样。首先，从连接(Q)中均匀随机地采样一个元组$\mathbf{u}$。然后，检查对于任意两个不同的属性$A,{A}^{\prime } \in  \mathcal{V}$是否有$\mathbf{u}\left( A\right)  = \mathbf{u}\left( {A}^{\prime }\right)$。如果是，则宣告失败；否则，宣告成功并返回与元组$\mathbf{u}$匹配的唯一出现情况。成功概率等于$c \cdot  \mathrm{{OUT}}/{\mathrm{{OUT}}}_{\mathcal{Q}}$。在成功事件中，$\operatorname{occ}\left( {G,P}\right)$中的每个出现情况都有相同的概率被返回。

When $P$ is acyclic,so is ${G}_{\mathrm{{DC}}}$ ,and thus our algorithm in Theorem 3.1 can be readily applied to handle a subgraph sampling operation. To analyze the performance,consider first OUT $\geq  1$ . We expect to draw $O\left( {{\mathrm{{OUT}}}_{\mathcal{Q}}/\mathrm{{OUT}}}\right)$ samples from $\operatorname{join}\left( \mathcal{Q}\right)$ until a success event. As Theorem 3.1 guarantees retrieving a sample from join(Q)in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /{\mathrm{{OUT}}}_{\mathcal{Q}}}\right)$ expected time,overall we expect to sample an occurrence from $\operatorname{occ}\left( {G,P}\right)$ in

当$P$为无环图时，${G}_{\mathrm{{DC}}}$同样如此，因此我们在定理3.1中的算法可轻松用于处理子图采样操作。为分析性能，首先考虑OUT $\geq  1$。我们期望从$\operatorname{join}\left( \mathcal{Q}\right)$中抽取$O\left( {{\mathrm{{OUT}}}_{\mathcal{Q}}/\mathrm{{OUT}}}\right)$个样本，直至出现成功事件。由于定理3.1保证在$O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /{\mathrm{{OUT}}}_{\mathcal{Q}}}\right)$的期望时间内从连接（Q）中获取一个样本，总体而言，我们期望在以下时间内从$\operatorname{occ}\left( {G,P}\right)$中采样到一个出现实例

$$
O\left( {\frac{\text{ polymat }\left( \mathrm{{DC}}\right) }{{\mathrm{{OUT}}}_{\mathcal{Q}}} \cdot  \frac{{\mathrm{{OUT}}}_{\mathcal{Q}}}{\mathrm{{OUT}}}}\right)  = O\left( \frac{\text{ polymat }\left( \mathrm{{DC}}\right) }{\mathrm{{OUT}}}\right) 
$$

time. To prepare for the possibility of OUT $= 0$ ,we apply the "two-thread approach" in Section 3. We run a concurrent thread that executes Ngo’s algorithm in [30], which finds the whole join(Q), and hence $\operatorname{occ}\left( {G,P}\right)$ ,in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ time,after which we can declare $\operatorname{occ}\left( {G,P}\right)  = \varnothing$ or sample from $\operatorname{occ}\left( {G,P}\right)$ in $O\left( 1\right)$ time. By accepting whichever thread finishes earlier,we ensure that the operation completes in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ time.

时间。为应对OUT $= 0$的可能性，我们采用第3节中的“双线程方法”。我们运行一个并发线程来执行文献[30]中的Ngo算法，该算法能在$O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$时间内找到整个连接（Q），进而得到$\operatorname{occ}\left( {G,P}\right)$，之后我们可以声明$\operatorname{occ}\left( {G,P}\right)  = \varnothing$或者在$O\left( 1\right)$时间内从$\operatorname{occ}\left( {G,P}\right)$中采样。通过接受先完成的线程，我们确保该操作在$O\left( {\text{polymat}\left( \mathrm{{DC}}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$时间内完成。

The main challenge arises when $P$ is cyclic. In this case, ${G}_{\mathrm{{DC}}}$ (which equals $P$ ) is cyclic. Thus, DC becomes a cyclic set of degree constraints, rendering neither Theorem 3.1 nor Ngo's algorithm in [30] applicable. We overcome this challenge with the lemma below.

当$P$为循环图时，主要挑战便出现了。在这种情况下，${G}_{\mathrm{{DC}}}$（等于$P$）是循环图。因此，DC成为一组循环的度约束，使得定理3.1和文献[30]中的Ngo算法都无法适用。我们通过下面的引理来克服这一挑战。

Lemma 4.1. If $\mathrm{{DC}}$ is cyclic,we can always find an acyclic subset ${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$ satisfying polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  =$ $\Theta \left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ .

引理4.1。如果$\mathrm{{DC}}$是循环图，我们总能找到一个满足多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  =$ $\Theta \left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$的无环子集${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$。

The proof is presented in Appendix B. Because $\mathcal{Q} \vDash  \mathrm{{DC}}$ and ${\mathrm{{DC}}}^{\prime }$ is a subset of $\mathrm{{DC}}$ ,we know that $\mathcal{Q}$ must be consistent with ${\mathrm{{DC}}}^{\prime }$ as well,i.e., $\mathcal{Q} \vDash  {\mathrm{{DC}}}^{\prime }$ . Therefore,our Theorem 3.1 can now be used to extract a sample from $\operatorname{join}\left( \mathcal{Q}\right)$ in $O\left( {{\operatorname{polymat}}_{\text{dir }}\left( {\mathrm{{DC}}}^{\prime }\right) /\max \left\{  {1,{\mathrm{{OUT}}}_{\mathcal{Q}}}\right\}  }\right)$ time. Importantly,Lemma 4.1 also permits us to directly apply Ngo’s algorithm in [30] to compute ${join}\left( \mathcal{Q}\right)$ in $O\left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$ time. Therefore,we can now apply the two-thread technique to sample from $\operatorname{occ}\left( {G,P}\right)$ in

证明见附录B。因为$\mathcal{Q} \vDash  \mathrm{{DC}}$且${\mathrm{{DC}}}^{\prime }$是$\mathrm{{DC}}$的子集，我们知道$\mathcal{Q}$也必定与${\mathrm{{DC}}}^{\prime }$一致，即$\mathcal{Q} \vDash  {\mathrm{{DC}}}^{\prime }$。因此，我们现在可以使用定理3.1在$O\left( {{\operatorname{polymat}}_{\text{dir }}\left( {\mathrm{{DC}}}^{\prime }\right) /\max \left\{  {1,{\mathrm{{OUT}}}_{\mathcal{Q}}}\right\}  }\right)$时间内从$\operatorname{join}\left( \mathcal{Q}\right)$中提取一个样本。重要的是，引理4.1还允许我们直接应用文献[30]中的Ngo算法在$O\left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$时间内计算${join}\left( \mathcal{Q}\right)$。因此，我们现在可以应用双线程技术在以下时间内从$\operatorname{occ}\left( {G,P}\right)$中采样

$$
O\left( \frac{\text{ polymat }\left( {\mathrm{{DC}}}^{\prime }\right) }{\max \{ 1,\mathrm{{OUT}}\} }\right)  = O\left( \frac{\text{ polymat }\left( \mathrm{{DC}}\right) }{\max \{ 1,\mathrm{{OUT}}\} }\right)  = O\left( \frac{{\text{ polymat }}_{\text{dir }}\left( {\left| E\right| ,\lambda ,P}\right) }{\max \{ 1,\mathrm{{OUT}}\} }\right) 
$$

time. We thus have arrived yet:

时间。因此我们得出结论：

Theorem 4.2. Let $G = \left( {V,E}\right)$ be a simple directed data graph,where each vertex has an out-degree at most $\lambda$ . Let $P = \left( {{V}_{P},{E}_{P}}\right)$ be a simple weakly-connected directed pattern graph with a constant number of vertices. We can build in $O\left( \left| E\right| \right)$ expected time a data structure that supports each subgraph sampling operation in $O\left( {{\text{polymat}}_{\text{dir }}\left( {\left| E\right| ,\lambda ,P}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,where $\mathrm{{OUT}}$ is the number of occurrences of $P$ in $G$ ,and polymat ${}_{dir}\left( {\left| E\right| ,\lambda ,P}\right)$ is the directed polymatrioid bound in (18).

定理4.2。设$G = \left( {V,E}\right)$为一个简单有向数据图，其中每个顶点的出度至多为$\lambda$。设$P = \left( {{V}_{P},{E}_{P}}\right)$为一个顶点数量固定的简单弱连通有向模式图。我们可以在$O\left( \left| E\right| \right)$的期望时间内构建一个数据结构，该结构支持在$O\left( {{\text{polymat}}_{\text{dir }}\left( {\left| E\right| ,\lambda ,P}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$的期望时间内进行每次子图采样操作，其中$\mathrm{{OUT}}$是$P$在$G$中出现的次数，并且有向多拟阵界（directed polymatrioid bound）polymat ${}_{dir}\left( {\left| E\right| ,\lambda ,P}\right)$ 如式(18)所示。

Remarks. For subgraph listing, Jayaraman et al. [19] presented a sophisticated method that also enables the application of Ngo’s algorithm in [30] to a cyclic $P$ . Given the companion join $\mathcal{Q}$ ,they employ the degree uniformization technique [21] to generate $t = O\left( {\text{polylog}\left| E\right| }\right)$ new joins ${\mathcal{Q}}_{1},{\mathcal{Q}}_{2},\ldots ,{\mathcal{Q}}_{t}$ such that $\operatorname{join}\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{i = 1}}^{t}\operatorname{join}\left( {\mathcal{Q}}_{i}\right)$ . For each $i \in  \left\lbrack  t\right\rbrack$ ,they construct an acyclic set ${\mathrm{{DC}}}_{i}$ of degree constraints (which is not always a subset of $\mathrm{{DC}}$ ) with the property $\mathop{\sum }\limits_{{i = 1}}^{t}$ polymat $\left( {\mathrm{{DC}}}_{i}\right)  \leq$ polymat(DC). Each join ${\mathcal{Q}}_{i}\left( {i \in  \left\lbrack  t\right\rbrack  }\right)$ can then be processed by Ngo’s algorithm in $O\left( {\text{polymat}\left( {\mathrm{{DC}}}_{i}\right) }\right)$ time,thus giving an algorithm for computing $\operatorname{join}\left( \mathcal{Q}\right)$ (and hence $\operatorname{occ}\left( {G,P}\right)$ ) in $O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ time. On the other hand,Lemma 4.1 facilitates a direct application of Ngo’s algorithm to $\mathcal{Q}$ ,implying the non-necessity of degree uniformization in subgraph listing. We believe that this simplification is noteworthy and merits its own dedicated exposition, considering the critical nature of the subgraph listing problem. In the absence of Lemma 4.1, integrating our join-sampling algorithm with the methodology of [19] for the purpose of subgraph sampling would require substantially more effort. Our proof of Lemma 4.1 does draw upon the analysis of [19], as discussed in depth in Appendix B.

备注。对于子图列举问题，Jayaraman等人[19]提出了一种复杂的方法，该方法还能将Ngo在文献[30]中的算法应用于循环$P$。给定伴随连接$\mathcal{Q}$，他们采用度均匀化技术[21]生成$t = O\left( {\text{polylog}\left| E\right| }\right)$个新的连接${\mathcal{Q}}_{1},{\mathcal{Q}}_{2},\ldots ,{\mathcal{Q}}_{t}$，使得$\operatorname{join}\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{i = 1}}^{t}\operatorname{join}\left( {\mathcal{Q}}_{i}\right)$成立。对于每个$i \in  \left\lbrack  t\right\rbrack$，他们构造一个无环的度约束集合${\mathrm{{DC}}}_{i}$（该集合并不总是$\mathrm{{DC}}$的子集），其性质为$\mathop{\sum }\limits_{{i = 1}}^{t}$多拟阵（polymat） $\left( {\mathrm{{DC}}}_{i}\right)  \leq$ 多拟阵(DC)。然后，Ngo的算法可以在$O\left( {\text{polymat}\left( {\mathrm{{DC}}}_{i}\right) }\right)$的时间内处理每个连接${\mathcal{Q}}_{i}\left( {i \in  \left\lbrack  t\right\rbrack  }\right)$，从而给出一个在$O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$的时间内计算$\operatorname{join}\left( \mathcal{Q}\right)$（进而计算$\operatorname{occ}\left( {G,P}\right)$）的算法。另一方面，引理4.1有助于将Ngo的算法直接应用于$\mathcal{Q}$，这意味着在子图列举中不需要进行度均匀化。考虑到子图列举问题的重要性，我们认为这种简化值得关注，并且值得专门阐述。如果没有引理4.1，为了进行子图采样而将我们的连接采样算法与文献[19]的方法相结合将需要付出更多的努力。正如附录B中深入讨论的那样，我们对引理4.1的证明确实借鉴了文献[19]的分析。

## 5 Undirected Subgraph Sampling

## 5 无向子图采样

Given an undirected pattern graph $P = \left( {{V}_{P},{E}_{P}}\right)$ and an undirected data graph $G = \left( {V,E}\right)$ ,we use $\operatorname{occ}\left( {G,P}\right)$ to represent the set of occurrences of $P$ in $G$ . Every vertex in $G$ has a degree at most $\lambda$ . Our goal is to design an algorithm to sample from $\operatorname{occ}\left( {G,P}\right)$ efficiently.

给定一个无向模式图$P = \left( {{V}_{P},{E}_{P}}\right)$和一个无向数据图$G = \left( {V,E}\right)$，我们用$\operatorname{occ}\left( {G,P}\right)$表示$P$在$G$中出现的集合。$G$中的每个顶点的度至多为$\lambda$。我们的目标是设计一种算法，以便从$\operatorname{occ}\left( {G,P}\right)$中高效地进行采样。

We formulate the "polymatroid bound" of this problem through a reduction to its directed counterpart. For $P = \left( {{V}_{P},{E}_{P}}\right)$ ,create a directed pattern ${P}^{\prime } = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ as follows. First,set ${V}_{P}^{\prime } = {V}_{P}$ . Second,for every edge $\{ X,Y\}  \in  {E}_{P}$ ,add to ${E}_{P}$ two directed edges(X,Y)and(Y,X). Now,given an integer $m \geq  1$ ,an integer $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$ ,and an undirected pattern $P = \left( {{V}_{P},{E}_{P}}\right)$ ,the undirected polymatroid bound of $m,\lambda$ ,and $P$ is defined as

我们通过将该问题归约为其有向版本来构建此问题的“多拟阵界（polymatroid bound）”。对于$P = \left( {{V}_{P},{E}_{P}}\right)$，按如下方式创建有向模式${P}^{\prime } = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$。首先，设定${V}_{P}^{\prime } = {V}_{P}$。其次，对于每条边$\{ X,Y\}  \in  {E}_{P}$，向${E}_{P}$中添加两条有向边(X, Y)和(Y, X)。现在，给定一个整数$m \geq  1$、一个整数$\lambda  \in  \left\lbrack  {1,m}\right\rbrack$和一个无向模式$P = \left( {{V}_{P},{E}_{P}}\right)$，$m,\lambda$和$P$的无向多拟阵界定义为

$$
{\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)  = {\text{polymat}}_{\text{dir }}\left( {m,\lambda ,{P}^{\prime }}\right)  \tag{19}
$$

where the function polymat ${}_{\text{dir }}$ is defined in (18).

其中函数多拟阵${}_{\text{dir }}$在(18)中定义。

Our formulation highlights how undirected subgraph sampling can be reduced to the directed version. From $G = \left( {V,E}\right)$ ,construct a directed graph ${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$ by setting ${V}^{\prime } = V$ ,and for every edge $\{ x,y\}  \in  E$ ,adding to ${E}^{\prime }$ two directed edges(x,y)and(y,x). Every occurrence in $\operatorname{occ}\left( {G,P}\right)$ matches the same number (a constant) of occurrences in $\operatorname{occ}\left( {{G}^{\prime },{P}^{\prime }}\right)$ . By resorting to Theorem 4.2, we can obtain an algorithm to sample from $\operatorname{occ}\left( {G,P}\right)$ in $O\left( {{\text{polymat }}_{\text{undir }}\left( {2\left| E\right| ,\lambda ,P}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ time (where $\mathrm{{OUT}} = \left| {\operatorname{occ}\left( {G,P}\right) }\right|$ ) after a preprocessing of $O\left( \left| E\right| \right)$ expected time. We omit the details because they will be superseded by another simpler approach to be described later.

我们的构建方式凸显了如何将无向子图采样归约为有向版本。从$G = \left( {V,E}\right)$出发，通过设定${V}^{\prime } = V$来构建有向图${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$，并且对于每条边$\{ x,y\}  \in  E$，向${E}^{\prime }$中添加两条有向边(x, y)和(y, x)。$\operatorname{occ}\left( {G,P}\right)$中的每个出现次数与$\operatorname{occ}\left( {{G}^{\prime },{P}^{\prime }}\right)$中的相同数量（一个常数）的出现次数相匹配。借助定理4.2，我们可以在经过期望时间为$O\left( \left| E\right| \right)$的预处理后，得到一个在$O\left( {{\text{polymat }}_{\text{undir }}\left( {2\left| E\right| ,\lambda ,P}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$时间内从$\operatorname{occ}\left( {G,P}\right)$中采样的算法（其中$\mathrm{{OUT}} = \left| {\operatorname{occ}\left( {G,P}\right) }\right|$）。我们省略细节，因为它们将被稍后描述的另一种更简单的方法所取代。

Unlike polymat ${}_{\text{dir }}\left( {m,\lambda ,P}\right)$ ,which is defined through an LP,we can solve the undirected counterpart ${\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$ into a closed form. It is always possible $\left\lbrack  {6,{33},{35}}\right\rbrack$ to decompose $P$ into vertex-disjoint subgraphs ${\partial }_{1},{\partial }_{2},\ldots ,{\partial }_{\alpha },{ \star  }_{1},{ \star  }_{2},\ldots$ ,and ${ \star  }_{\beta }$ (for some integers $\alpha ,\beta  \geq  0$ ) such that

与通过线性规划（LP）定义的多拟阵${}_{\text{dir }}\left( {m,\lambda ,P}\right)$不同，我们可以将无向版本${\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$求解为封闭形式。总是有可能$\left\lbrack  {6,{33},{35}}\right\rbrack$将$P$分解为顶点不相交的子图${\partial }_{1},{\partial }_{2},\ldots ,{\partial }_{\alpha },{ \star  }_{1},{ \star  }_{2},\ldots$和${ \star  }_{\beta }$（对于某些整数$\alpha ,\beta  \geq  0$），使得

- ${\square }_{i}$ is an odd-length cycle for each $i \in  \left\lbrack  \alpha \right\rbrack$ ;

- 对于每个$i \in  \left\lbrack  \alpha \right\rbrack$，${\square }_{i}$是一个奇数长度的环；

- ${ \star  }_{j}$ is a star ${}^{3}$ for each $j \in  \left\lbrack  \beta \right\rbrack$ ;

- 对于每个$j \in  \left\lbrack  \beta \right\rbrack$，${ \star  }_{j}$是一个星图${}^{3}$；

---

<!-- Footnote -->

${}^{3}$ A star is an undirected graph where one vertex,called the center,has an edge to every other vertex,and each non-center vertex has degree 1 .

${}^{3}$星图是一种无向图，其中一个被称为中心的顶点与其他每个顶点都有一条边相连，并且每个非中心顶点的度为1。

<!-- Footnote -->

---

- $\mathop{\sum }\limits_{{i = 1}}^{\alpha }{\rho }^{ * }\left( {\partial }_{i}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{\rho }^{ * }\left( { \star  }_{j}\right)  = {\rho }^{ * }\left( P\right)$ ; see Section 1.2 for the definition of the fractional edge cover number function ${\rho }^{ * }\left( \text{.}\right) .$

- $\mathop{\sum }\limits_{{i = 1}}^{\alpha }{\rho }^{ * }\left( {\partial }_{i}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{\rho }^{ * }\left( { \star  }_{j}\right)  = {\rho }^{ * }\left( P\right)$ ；关于分数边覆盖数函数 ${\rho }^{ * }\left( \text{.}\right) .$ 的定义，请参见1.2节

We will refer to $\left( {{\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ as a fractional edge-cover decomposition of $P$ . Define:

我们将 $\left( {{\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ 称为 $P$ 的分数边覆盖分解。定义：

$$
{k}_{\text{cycle }} = \text{ total number of vertices in }{\bigtriangleup }_{1},\ldots ,{\bigtriangleup }_{\alpha } \tag{20}
$$

$$
{k}_{\text{star }} = \text{total number of vertices in}{ \star  }_{1},\ldots ,{ \star  }_{\beta }\text{.} \tag{21}
$$

We establish the lemma below in Appendix C.

我们在附录C中证明以下引理。

Lemma 5.1. If $k$ is the number of vertices in $P$ ,then

引理5.1。如果 $k$ 是 $P$ 中的顶点数，那么

$$
{\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)  = \left\{  \begin{array}{ll} m \cdot  {\lambda }^{k - 2} & \text{ if }\lambda  \leq  \sqrt{m}, \\  {m}^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }} & \text{ if }\lambda  > \sqrt{m} \end{array}\right.  \tag{22}
$$

As shown in Appendix D,for $k = O\left( 1\right)$ ,the expression in (22) asymptotically matches an upper bound of $\left| {\operatorname{occ}\left( {{G}^{\prime },P}\right) }\right|$ - for any ${G}^{\prime }$ with $m$ edges and maximum vertex-degree at most $\lambda$ - easily derived from a fractional edge-cover decomposition. The same appendix will also prove that, for a wide range of $m$ and $\lambda$ values,there exists a graph ${G}^{\prime }$ with $m$ edges and maximum vertex degree at most $\lambda$ guaranteeing the following simultaneously for all patterns $P$ with $k$ vertices: $\left| {\operatorname{occ}\left( {{G}^{\prime },P}\right) }\right|$ is no less than the expression’s value,up to a factor of $1/{\left( 4k\right) }^{k}$ . This implies that,when $k$ is a constant, the expression asymptotically equals ${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$ . More effort is required to prove that the expression equals ${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right) )$ precisely,as we demonstrate in Appendix C.

如附录D所示，对于 $k = O\left( 1\right)$ ，(22) 中的表达式渐近匹配 $\left| {\operatorname{occ}\left( {{G}^{\prime },P}\right) }\right|$ 的一个上界 —— 对于任何具有 $m$ 条边且最大顶点度至多为 $\lambda$ 的 ${G}^{\prime }$ —— 该上界可从分数边覆盖分解轻松推导得出。同一附录还将证明，对于广泛的 $m$ 和 $\lambda$ 值，存在一个具有 $m$ 条边且最大顶点度至多为 $\lambda$ 的图 ${G}^{\prime }$ ，能同时保证对于所有具有 $k$ 个顶点的模式 $P$ 满足以下条件： $\left| {\operatorname{occ}\left( {{G}^{\prime },P}\right) }\right|$ 不小于该表达式的值，误差因子为 $1/{\left( 4k\right) }^{k}$ 。这意味着，当 $k$ 为常数时，该表达式渐近等于 ${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$ 。要证明该表达式精确等于 ${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right) )$ ，还需要更多的工作，我们将在附录C中进行证明。

Lemma 5.1 points to an alternative strategy to perform undirected subgraph sampling without joins. Identify an arbitrary spanning tree $\mathcal{T}$ of the pattern $P = \left( {{V}_{P},{E}_{P}}\right)$ . Order the vertices of ${V}_{P}$ as ${A}_{1},{A}_{2},\ldots ,{A}_{k}$ such that,for any $i \in  \left\lbrack  {2,k}\right\rbrack$ ,vertex ${A}_{i}$ is adjacent in $\mathcal{T}$ to a (unique) vertex ${A}_{j}$ with $j \in  \left\lbrack  {i - 1}\right\rbrack$ . Now construct a map $f : {V}_{P} \rightarrow  V$ as follows (recall that $V$ is the vertex set of the data graph $G = \left( {V,E}\right)$ ). First,take an edge $\{ u,v\}$ uniformly at random from $E$ ,and choose one of the following with an equal probability: (i) $f\left( {A}_{1}\right)  = u,f\left( {A}_{2}\right)  = v$ ,or (ii) $f\left( {A}_{1}\right)  = v,f\left( {A}_{2}\right)  = u$ . Then,for every $i \in  \left\lbrack  {3,k}\right\rbrack$ ,decide $f\left( {A}_{i}\right)$ as follows. Suppose that ${A}_{i}$ is adjacent to ${A}_{j}$ for some (unique) $j \in  \left\lbrack  {1,i - 1}\right\rbrack$ . Toss a coin with probability $\deg \left( {f\left( {A}_{j}\right) }\right) /\lambda$ ,where $\deg \left( {f\left( {A}_{j}\right) }\right)$ is the degree of vertex $f\left( {A}_{j}\right)$ in $G$ . If the coin comes up tails,declare failure and terminate. Otherwise,set $f\left( {A}_{i}\right)$ to a neighbor of $f\left( {A}_{j}\right)$ in $G$ picked uniformly at random. After finalizing the map $f\left( .\right)$ , check whether $\left\{  {f\left( {A}_{i}\right)  \mid  i \in  \left\lbrack  k\right\rbrack  }\right\}$ induces a subgraph of $G$ isomorphic to $P$ . If so,accept $f$ and return this subgraph; otherwise,reject $f$ . To guarantee a sample or declare $\operatorname{occ}\left( {G,P}\right)  = \varnothing$ ,apply the "two-thread approach" by (i) repeating the algorithm until acceptance and (ii) concurrently running an algorithm for computing the whole $\operatorname{occ}\left( {G,P}\right)$ in $O\left( {{\operatorname{polymat}}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) }\right)$ time ${}^{4}$ . As proved in Appendix E,this ensures the expected sample time $O\left( {\left| E\right|  \cdot  {\lambda }^{k - 2}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ after a preprocessing of $O\left( \left| E\right| \right)$ expected time.

引理5.1指出了一种无需连接操作即可进行无向子图采样的替代策略。确定模式$P = \left( {{V}_{P},{E}_{P}}\right)$的任意一棵生成树$\mathcal{T}$。将${V}_{P}$的顶点排序为${A}_{1},{A}_{2},\ldots ,{A}_{k}$，使得对于任意$i \in  \left\lbrack  {2,k}\right\rbrack$，顶点${A}_{i}$在$\mathcal{T}$中与一个（唯一的）顶点${A}_{j}$相邻，且$j \in  \left\lbrack  {i - 1}\right\rbrack$。现在按如下方式构造一个映射$f : {V}_{P} \rightarrow  V$（回顾一下，$V$是数据图$G = \left( {V,E}\right)$的顶点集）。首先，从$E$中均匀随机选取一条边$\{ u,v\}$，并以相等的概率选择以下两种情况之一：(i) $f\left( {A}_{1}\right)  = u,f\left( {A}_{2}\right)  = v$，或(ii) $f\left( {A}_{1}\right)  = v,f\left( {A}_{2}\right)  = u$。然后，对于每个$i \in  \left\lbrack  {3,k}\right\rbrack$，按如下方式确定$f\left( {A}_{i}\right)$。假设对于某个（唯一的）$j \in  \left\lbrack  {1,i - 1}\right\rbrack$，${A}_{i}$与${A}_{j}$相邻。抛掷一枚正面朝上概率为$\deg \left( {f\left( {A}_{j}\right) }\right) /\lambda$的硬币，其中$\deg \left( {f\left( {A}_{j}\right) }\right)$是顶点$f\left( {A}_{j}\right)$在$G$中的度数。如果硬币反面朝上，则宣告失败并终止。否则，将$f\left( {A}_{i}\right)$设为$G$中$f\left( {A}_{j}\right)$的一个均匀随机选取的邻居。在确定好映射$f\left( .\right)$之后，检查$\left\{  {f\left( {A}_{i}\right)  \mid  i \in  \left\lbrack  k\right\rbrack  }\right\}$是否诱导出一个与$P$同构的$G$的子图。如果是，则接受$f$并返回这个子图；否则，拒绝$f$。为了保证得到一个样本或宣告$\operatorname{occ}\left( {G,P}\right)  = \varnothing$，采用“双线程方法”，即(i)重复该算法直到接受样本，以及(ii)同时运行一个用于在$O\left( {{\operatorname{polymat}}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) }\right)$时间${}^{4}$内计算整个$\operatorname{occ}\left( {G,P}\right)$的算法。如附录E所证明的，这确保了在经过期望时间为$O\left( \left| E\right| \right)$的预处理之后，期望的采样时间为$O\left( {\left| E\right|  \cdot  {\lambda }^{k - 2}/\max \{ 1,\mathrm{{OUT}}\} }\right)$。

The above algorithm suffices for the case $\lambda  \leq  \sqrt{\left| E\right| }$ . Consider now the case $\lambda  > \sqrt{\left| E\right| }$ . We will construct a map $f : {V}_{P} \rightarrow  V$ according to a given fractional edge-cover decomposition $\left( {{\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ . For each $i \in  \left\lbrack  \alpha \right\rbrack$ ,use the algorithm of [16] to uniformly sample an occurrence of ${\partial }_{i}$ - denoted as ${G}_{\text{sub }}\left( {\partial }_{i}\right)$ - in $O\left( {{\left| E\right| }^{{\rho }^{ * }\left( {\partial }_{i}\right) }/\max \left\{  {1,\left| {\operatorname{occ}\left( {G,{\partial }_{i}}\right) }\right| }\right\}  }\right)$ expected time. Let ${f}_{{\square }_{i}}$ be a map from the vertex set of ${\partial }_{i}$ to that of ${G}_{\text{sub }}\left( {\partial }_{i}\right)$ ,chosen uniformly at random from all the isomorphism bijections (defined in Section 1.1) between ${ ○ }_{i}$ and ${G}_{\text{sub }}\left( { ○ }_{i}\right)$ . For each $j \in  \left\lbrack  \beta \right\rbrack$ , apply our earlier algorithm to uniformly sample an occurrence of ${ \star  }_{j}$ - denoted as ${G}_{\text{sub }}\left( { \star  }_{j}\right)$ - in $O\left( {\left| E\right|  \cdot  {\lambda }^{{k}_{j}}/\max \left\{  {1,\left| {\operatorname{occ}\left( {G,{ \star  }_{j}}\right) }\right| }\right\}  }\right)$ expected time,where ${k}_{j}$ is the number of vertices in ${ \star  }_{j}$ . Let ${f}_{{ \star  }_{j}}$ be a map from the vertex set of ${ \star  }_{j}$ to that of ${G}_{\text{sub }}\left( { \star  }_{j}\right)$ ,chosen uniformly at random from all the polymorphism bijections between ${ \star  }_{j}$ and ${G}_{\text{sub }}\left( { \star  }_{j}\right)$ . If any of ${\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }$ has no occurrences,declare $\operatorname{occ}\left( {G,P}\right)  = \varnothing$ and terminate. Otherwise,the functions in $\left\{  {{f}_{{\square }_{i}} \mid  i \in  \left\lbrack  \alpha \right\rbrack  }\right\}$ and $\left\{  {{f}_{{ \star  }_{j}} \mid  j \in  \left\lbrack  \beta \right\rbrack  }\right\}$ together determine the map $f$ we aim to build. Check whether $\{ f\left( A\right)  \mid  A \in  V\}$ induces a subgraph of $G$ isomorphic to $P$ . If so,accept $f$ and return this subgraph; otherwise,reject $f$ . Repeat until acceptance and concurrently run an algorithm for computing the whole $\operatorname{occ}\left( {G,P}\right)$ in $O\left( {{\text{polymat}}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) }\right)$ time. As proved in Appendix E,this ensures the expected sample time $O\left( {{\left| E\right| }^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ after a preprocessing of $O\left( \left| E\right| \right)$ expected time.

上述算法适用于情形 $\lambda  \leq  \sqrt{\left| E\right| }$。现在考虑情形 $\lambda  > \sqrt{\left| E\right| }$。我们将根据给定的分数边覆盖分解 $\left( {{\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ 构造一个映射 $f : {V}_{P} \rightarrow  V$。对于每个 $i \in  \left\lbrack  \alpha \right\rbrack$，使用文献 [16] 中的算法在 $O\left( {{\left| E\right| }^{{\rho }^{ * }\left( {\partial }_{i}\right) }/\max \left\{  {1,\left| {\operatorname{occ}\left( {G,{\partial }_{i}}\right) }\right| }\right\}  }\right)$ 期望时间内均匀采样 ${\partial }_{i}$ 的一个出现（记为 ${G}_{\text{sub }}\left( {\partial }_{i}\right)$）。设 ${f}_{{\square }_{i}}$ 是从 ${\partial }_{i}$ 的顶点集到 ${G}_{\text{sub }}\left( {\partial }_{i}\right)$ 的顶点集的一个映射，它是从 ${ ○ }_{i}$ 和 ${G}_{\text{sub }}\left( { ○ }_{i}\right)$ 之间的所有同构双射（在 1.1 节中定义）中均匀随机选取的。对于每个 $j \in  \left\lbrack  \beta \right\rbrack$，应用我们之前的算法在 $O\left( {\left| E\right|  \cdot  {\lambda }^{{k}_{j}}/\max \left\{  {1,\left| {\operatorname{occ}\left( {G,{ \star  }_{j}}\right) }\right| }\right\}  }\right)$ 期望时间内均匀采样 ${ \star  }_{j}$ 的一个出现（记为 ${G}_{\text{sub }}\left( { \star  }_{j}\right)$），其中 ${k}_{j}$ 是 ${ \star  }_{j}$ 中的顶点数。设 ${f}_{{ \star  }_{j}}$ 是从 ${ \star  }_{j}$ 的顶点集到 ${G}_{\text{sub }}\left( { \star  }_{j}\right)$ 的顶点集的一个映射，它是从 ${ \star  }_{j}$ 和 ${G}_{\text{sub }}\left( { \star  }_{j}\right)$ 之间的所有同态双射中均匀随机选取的。如果 ${\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }$ 中的任何一个没有出现，则声明 $\operatorname{occ}\left( {G,P}\right)  = \varnothing$ 并终止。否则，$\left\{  {{f}_{{\square }_{i}} \mid  i \in  \left\lbrack  \alpha \right\rbrack  }\right\}$ 和 $\left\{  {{f}_{{ \star  }_{j}} \mid  j \in  \left\lbrack  \beta \right\rbrack  }\right\}$ 中的函数共同确定我们要构建的映射 $f$。检查 $\{ f\left( A\right)  \mid  A \in  V\}$ 是否诱导出 $G$ 的一个与 $P$ 同构的子图。如果是，则接受 $f$ 并返回这个子图；否则，拒绝 $f$。重复此过程直到接受，并同时运行一个在 $O\left( {{\text{polymat}}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) }\right)$ 时间内计算整个 $\operatorname{occ}\left( {G,P}\right)$ 的算法。如附录 E 中所证明的，这确保了在经过 $O\left( \left| E\right| \right)$ 期望时间的预处理后，期望采样时间为 $O\left( {{\left| E\right| }^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$。

---

<!-- Footnote -->

${}^{4}$ For this purpose,use Ngo’s algorithm in [30] to find all occurrences in $\operatorname{occ}\left( {{G}^{\prime },{P}^{\prime }}\right)$ — see our earlier definitions of ${G}^{\prime }$ and ${P}^{\prime } -$ which is possible due to Lemma 4.1.

${}^{4}$ 为此，使用文献[30]中的 Ngo 算法（Ngo's algorithm）来找出$\operatorname{occ}\left( {{G}^{\prime },{P}^{\prime }}\right)$中的所有出现情况 —— 见我们之前对${G}^{\prime }$和${P}^{\prime } -$的定义，由于引理 4.1，这是可行的。

<!-- Footnote -->

---

We can now conclude with our last main result:

我们现在可以得出最后一个主要结果：

Theorem 5.2. Let $G = \left( {V,E}\right)$ be a simple undirected data graph,where each vertex has a degree at most $\lambda$ . Let $P = \left( {{V}_{P},{E}_{P}}\right)$ be a simple connected pattern graph with a constant number of vertices. We can build in $O\left( \left| E\right| \right)$ expected time a data structure that supports each subgraph sampling operation in $O\left( {{\text{polymat}}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ expected time,where $\mathrm{{OUT}}$ is the number of occurrences of $P$ in $G$ ,and polymat ${}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right)$ is the undirected polymatrioid bound in (22).

定理 5.2。设$G = \left( {V,E}\right)$为一个简单无向数据图，其中每个顶点的度数至多为$\lambda$。设$P = \left( {{V}_{P},{E}_{P}}\right)$为一个具有固定顶点数的简单连通模式图。我们可以在$O\left( \left| E\right| \right)$的期望时间内构建一个数据结构，该结构支持每次子图采样操作的期望时间为$O\left( {{\text{polymat}}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$，其中$\mathrm{{OUT}}$是$P$在$G$中的出现次数，并且多拟阵（polymat）${}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right)$是式(22)中的无向多拟阵界。

## 6 Concluding Remarks

## 6 结论

Our new sampling algorithms imply new results on several other fundamental problems. We will illustrate this with respect to evaluating a join $\mathcal{Q}$ consistent with an acyclic set $\mathrm{{DC}}$ of degree constraints. Similar implications also apply to subgraph sampling.

我们的新采样算法在其他几个基本问题上产生了新的结果。我们将以评估与度约束的无环集$\mathrm{{DC}}$一致的连接$\mathcal{Q}$为例进行说明。类似的结果也适用于子图采样。

- By standard techniques $\left\lbrack  {{11},{14}}\right\rbrack$ ,we can estimate the output size OUT up to a relative error $\epsilon$ with high probability (i.e.,at least $1 - 1/{\mathrm{{IN}}}^{c}$ for an arbitrarily large constant $c$ ) in time $\widetilde{O}\left( {\frac{1}{{\epsilon }^{2}}\frac{\text{polymat}\left( \mathrm{{DC}}\right) }{\max \{ 1,\mathrm{{OUT}}\} }}\right)$ after a preprocessing of $O\left( \mathrm{{IN}}\right)$ expected time.

- 通过标准技术$\left\lbrack  {{11},{14}}\right\rbrack$，在经过$O\left( \mathrm{{IN}}\right)$的期望时间预处理后，我们可以在$\widetilde{O}\left( {\frac{1}{{\epsilon }^{2}}\frac{\text{polymat}\left( \mathrm{{DC}}\right) }{\max \{ 1,\mathrm{{OUT}}\} }}\right)$的时间内以高概率（即，对于任意大的常数$c$，至少为$1 - 1/{\mathrm{{IN}}}^{c}$）将输出大小 OUT 估计到相对误差$\epsilon$以内。

- Employing a technique in [14],we can,with high probability,report all the tuples in $\operatorname{join}\left( \mathcal{Q}\right)$ with a delay of $\widetilde{O}\left( \frac{\text{polymat(DC)}}{\max \{ 1,\mathrm{{OUT}}\} }\right)$ . In this context,'delay' refers to the maximum interval between the reporting of two successive result tuples, assuming the presence of a placeholder tuple at the beginning and another at the end.

- 采用文献[14]中的一种技术，我们可以高概率地以$\widetilde{O}\left( \frac{\text{polymat(DC)}}{\max \{ 1,\mathrm{{OUT}}\} }\right)$的延迟报告$\operatorname{join}\left( \mathcal{Q}\right)$中的所有元组。在此上下文中，“延迟”指的是报告两个连续结果元组之间的最大间隔，假设在开始和结束处各有一个占位元组。

- In addition to the delay guarantee, our algorithm in the second bullet can, with high probability, report the tuples of $\operatorname{join}\left( \mathcal{Q}\right)$ in a random permutation. This means that each of the OUT! possible permutations has an equal probability of being the output.

- 除了延迟保证外，第二个要点中的算法还可以高概率地以随机排列的方式报告$\operatorname{join}\left( \mathcal{Q}\right)$的元组。这意味着 OUT! 种可能的排列中的每一种都有相等的概率成为输出。

All of the results presented above compare favorably with the current state of the art as presented in [14]. This is primarily due to the superiority of polymat (DC) over ${AGM}\left( \mathcal{Q}\right)$ . In addition,our findings in the last two bullet points also complement Ngo's algorithm as described in [30] in a satisfying manner.

上述所有结果与文献[14]中呈现的当前技术水平相比具有优势。这主要是由于多拟阵（polymat）(DC)优于${AGM}\left( \mathcal{Q}\right)$。此外，我们在最后两个要点中的发现也很好地补充了文献[30]中描述的 Ngo 算法。

## Appendix

## 附录

## A Implementing ADC-Sample with Indexes

## A 使用索引实现 ADC - 采样

We preprocess each constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ as follows. Let $R \in  Q$ be its main guard,i.e., $R = {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }$ . For each $i \in  \left\lbrack  k\right\rbrack$ and each tuple $\mathbf{w} \in  {\Pi }_{\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}}\left( R\right)$ ,define

我们按如下方式对每个约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$进行预处理。设$R \in  Q$为其主保护条件，即$R = {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }$。对于每个$i \in  \left\lbrack  k\right\rbrack$和每个元组$\mathbf{w} \in  {\Pi }_{\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}}\left( R\right)$，定义

$$
{R}_{\mathcal{Y}}\left( {i,\mathbf{w}}\right)  = \left\{  {\mathbf{u}\left\lbrack  \mathcal{Y}\right\rbrack   \mid  \mathbf{u} \in  R,\mathbf{u}\left\lbrack  {\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}}\right\rbrack   = \mathbf{w}}\right\}  
$$

which we refer to as a fragment.

我们将其称为片段。

During preprocessing,we compute and store ${R}_{\mathcal{Y}}\left( {i,\mathbf{w}}\right)$ for every $i \in  \left\lbrack  k\right\rbrack$ and $\mathbf{w} \in  {\Pi }_{\text{schema }\left( R\right)  \cap  {\mathcal{V}}_{i}}\left( R\right)$ . Next,we will explain how to do so for an arbitrary $i \in  \left\lbrack  k\right\rbrack$ . First,group all the tuples of $R$ by the attributes in schema $\left( R\right)  \cap  {\mathcal{V}}_{i}$ ,which can be done in $O\left( \mathrm{{IN}}\right)$ expected time by hashing. Then,perform the following steps for each group in turn. Let $\mathbf{w}$ be the group’s projection on $\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}$ . We compute the group tuples’ projections onto $\mathcal{Y}$ and eliminate duplicate projections,the outcome of which is precisely ${Ry}\left( {i,\mathbf{w}}\right)$ and is stored using an array. With hashing,this requires expected time only linear to the group’s size. Therefore,the total cost of generating the fragments ${R}_{\mathcal{Y}}\left( {i,\mathbf{w}}\right)$ of all $\mathbf{w} \in  {\Pi }_{\text{schema }\left( R\right)  \cap  {\mathcal{V}}_{i}}\left( R\right)$ is $O\left( \mathrm{{IN}}\right)$ expected.

在预处理期间，我们为每个$i \in  \left\lbrack  k\right\rbrack$和$\mathbf{w} \in  {\Pi }_{\text{schema }\left( R\right)  \cap  {\mathcal{V}}_{i}}\left( R\right)$计算并存储${R}_{\mathcal{Y}}\left( {i,\mathbf{w}}\right)$。接下来，我们将解释如何为任意的$i \in  \left\lbrack  k\right\rbrack$进行此操作。首先，通过模式$\left( R\right)  \cap  {\mathcal{V}}_{i}$中的属性对$R$的所有元组进行分组，这可以通过哈希在$O\left( \mathrm{{IN}}\right)$期望时间内完成。然后，依次对每个组执行以下步骤。设$\mathbf{w}$为该组在$\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}$上的投影。我们计算该组元组在$\mathcal{Y}$上的投影并消除重复的投影，其结果恰好是${Ry}\left( {i,\mathbf{w}}\right)$，并使用数组进行存储。通过哈希，这只需要与组大小呈线性关系的期望时间。因此，生成所有$\mathbf{w} \in  {\Pi }_{\text{schema }\left( R\right)  \cap  {\mathcal{V}}_{i}}\left( R\right)$的片段${R}_{\mathcal{Y}}\left( {i,\mathbf{w}}\right)$的总成本的期望为$O\left( \mathrm{{IN}}\right)$。

After the above preprocessing,given any $i \in  \left\lbrack  k\right\rbrack$ ,constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$ ,tuple $\mathbf{w}$ over ${\mathcal{V}}_{i - 1}$ ,and value $a \in  \mathbf{{dom}}$ ,we can compute relde ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)$ defined in (7) in constant time. For convenience,let $R = {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }$ . To compute $\left| {{\Pi }_{\mathcal{Y}}\left( {R \ltimes  \mathbf{w}}\right) }\right|$ (the denominator of (7)),first obtain ${\mathbf{w}}_{1} = \mathbf{w}\left\lbrack  {\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i - 1}}\right\rbrack$ . Then, $\Pi \mathcal{Y}\left( {R \ltimes  \mathbf{w}}\right)$ is just the fragment $R\mathcal{Y}\left( {i - 1,{\mathbf{w}}_{1}}\right)$ ,which has been pre-stored. The size of this fragment can be retrieved using ${\mathbf{w}}_{1}$ in $O\left( 1\right)$ time. Similarly,to compute $\left| {{\sigma }_{{A}_{i} = a}\left( {{\Pi y}\left( {R \ltimes  \mathbf{w}}\right) }\right) }\right|$ (the numerator of (7)),we can first obtain ${\mathbf{w}}_{2}$ ,which is a tuple over $\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}$ that shares the values of ${\mathbf{w}}_{1}$ on all the attributes in $\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i - 1}$ and additionally uses value $a$ on attribute ${A}_{i}$ . Then, ${\sigma }_{{A}_{i} = a}\left( {{\Pi }_{\mathcal{Y}}\left( {R \ltimes  \mathbf{w}}\right) }\right)$ is just the fragment ${R}_{\mathcal{Y}}\left( {i,{\mathbf{w}}_{2}}\right)$ , which has been pre-stored. The size of this fragment can be fetched using ${\mathbf{w}}_{2}$ in $O\left( 1\right)$ time.

经过上述预处理后，给定任意的 $i \in  \left\lbrack  k\right\rbrack$、约束 $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  \mathrm{{DC}}$、在 ${\mathcal{V}}_{i - 1}$ 上的元组 $\mathbf{w}$ 以及值 $a \in  \mathbf{{dom}}$，我们可以在常数时间内计算出 (7) 中定义的相对依赖度 ${g}_{i,\mathcal{X},\mathcal{Y}}\left( {\mathbf{w},a}\right)$。为方便起见，设 $R = {R}_{F\left( {\mathcal{X},\mathcal{Y}}\right) }$。要计算 $\left| {{\Pi }_{\mathcal{Y}}\left( {R \ltimes  \mathbf{w}}\right) }\right|$（(7) 的分母），首先获取 ${\mathbf{w}}_{1} = \mathbf{w}\left\lbrack  {\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i - 1}}\right\rbrack$。然后，$\Pi \mathcal{Y}\left( {R \ltimes  \mathbf{w}}\right)$ 就是预先存储的片段 $R\mathcal{Y}\left( {i - 1,{\mathbf{w}}_{1}}\right)$。可以使用 ${\mathbf{w}}_{1}$ 在 $O\left( 1\right)$ 时间内检索该片段的大小。类似地，要计算 $\left| {{\sigma }_{{A}_{i} = a}\left( {{\Pi y}\left( {R \ltimes  \mathbf{w}}\right) }\right) }\right|$（(7) 的分子），我们可以首先获取 ${\mathbf{w}}_{2}$，它是在 $\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i}$ 上的一个元组，该元组在 $\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i - 1}$ 中的所有属性上共享 ${\mathbf{w}}_{1}$ 的值，并且在属性 ${A}_{i}$ 上额外使用值 $a$。然后，${\sigma }_{{A}_{i} = a}\left( {{\Pi }_{\mathcal{Y}}\left( {R \ltimes  \mathbf{w}}\right) }\right)$ 就是预先存储的片段 ${R}_{\mathcal{Y}}\left( {i,{\mathbf{w}}_{2}}\right)$。可以使用 ${\mathbf{w}}_{2}$ 在 $O\left( 1\right)$ 时间内获取该片段的大小。

As a corollary,given any $i \in  \left\lbrack  k\right\rbrack$ ,tuple $\mathbf{w}$ over ${\mathcal{V}}_{i - 1}$ ,and value $a \in  \mathbf{{dom}}$ ,we can compute relde ${g}_{i}^{ * }\left( {\mathbf{w},a}\right)$ and constraint ${}_{i}^{ * }\left( {\mathbf{w},a}\right)$ - defined in (8) and (9),respectively - in constant time.

作为推论，给定任意的 $i \in  \left\lbrack  k\right\rbrack$、在 ${\mathcal{V}}_{i - 1}$ 上的元组 $\mathbf{w}$ 以及值 $a \in  \mathbf{{dom}}$，我们可以在常数时间内分别计算出 (8) 和 (9) 中定义的相对依赖度 ${g}_{i}^{ * }\left( {\mathbf{w},a}\right)$ 和约束 ${}_{i}^{ * }\left( {\mathbf{w},a}\right)$。

It remains to explain how to implement Line 4 of ADC-sample (Figure 1). Here, we want to randomly sample a tuple from ${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$ . Again,for convenience,let $R = {R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) }$ . Obtain ${\mathbf{w}}^{\prime } = {\mathbf{w}}_{i - 1}\left\lbrack  {\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i - 1}}\right\rbrack$ . Then, ${\Pi }_{\mathcal{Y}}\left( {R \ltimes  {\mathbf{w}}_{i - 1}}\right)$ is just the fragment ${Ry}\left( {i - 1,{\mathbf{w}}^{\prime }}\right)$ , which has been stored in an array. The starting address and size of the array can be acquired using ${\mathbf{w}}^{\prime }$ in $O\left( 1\right)$ time,after which a sample can be drawn from the fragment in constant time.

接下来需要解释如何实现ADC采样（图1）的第4行。在这里，我们要从${\Pi }_{{\mathcal{Y}}^{ \circ  }}\left( {{R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) } \ltimes  {\mathbf{w}}_{i - 1}}\right)$中随机采样一个元组。同样，为了方便起见，设$R = {R}_{F\left( {{\mathcal{X}}^{ \circ  },{\mathcal{Y}}^{ \circ  }}\right) }$。得到${\mathbf{w}}^{\prime } = {\mathbf{w}}_{i - 1}\left\lbrack  {\operatorname{schema}\left( R\right)  \cap  {\mathcal{V}}_{i - 1}}\right\rbrack$。然后，${\Pi }_{\mathcal{Y}}\left( {R \ltimes  {\mathbf{w}}_{i - 1}}\right)$就是片段${Ry}\left( {i - 1,{\mathbf{w}}^{\prime }}\right)$，它已存储在一个数组中。可以使用${\mathbf{w}}^{\prime }$在$O\left( 1\right)$时间内获取该数组的起始地址和大小，之后可以在常数时间内从该片段中抽取一个样本。

## B Proof of Lemma 4.1

## B 引理4.1的证明

Let us rephrase the problem as follows. Let $P = \left( {{V}_{P},{E}_{P}}\right)$ be a cyclic pattern graph. Given an integer $m \geq  1$ and an integer $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$ ,define $\mathrm{{DC}}$ to be a set of degree constraints over ${V}_{P}$ that contains two constraints for each edge $\left( {X,Y}\right)  \in  {E}_{P} : \left( {\varnothing ,\{ X,Y\} ,m}\right)$ and $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ . The constraint dependence graph ${G}_{\mathrm{{DC}}}$ is exactly $P$ and,hence,is cyclic. We want to prove the existence of an acyclic ${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$ such that polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  =$ polymat(DC). We will first tackle the situation where $\lambda  > \sqrt{m}$ before proceeding to the opposite scenario. The former case presents a more intriguing line of argumentation than the latter.

让我们将问题重新表述如下。设$P = \left( {{V}_{P},{E}_{P}}\right)$为一个循环模式图。给定一个整数$m \geq  1$和一个整数$\lambda  \in  \left\lbrack  {1,m}\right\rbrack$，定义$\mathrm{{DC}}$为关于${V}_{P}$的一组度约束，对于每条边$\left( {X,Y}\right)  \in  {E}_{P} : \left( {\varnothing ,\{ X,Y\} ,m}\right)$和$\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$包含两个约束。约束依赖图${G}_{\mathrm{{DC}}}$恰好是$P$，因此是循环的。我们要证明存在一个无环的${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$，使得多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  =$ 多面体(DC)。我们将先处理$\lambda  > \sqrt{m}$的情况，然后再处理相反的情况。前一种情况比后一种情况呈现出更有趣的论证思路。

Case $\lambda  > \sqrt{m}$ . For every edge $\left( {X,Y}\right)  \in  {G}_{\mathrm{{DC}}} = \left( {{V}_{P},{E}_{P}}\right)$ ,define two variables: ${x}_{X,Y}$ and ${z}_{X,Y}$ . Jayaraman et al. [19] showed that,for $\lambda  > \sqrt{m}$ ,polymat(DC) is,up to a constant factor,the optimal value of the following LP (named ${\mathrm{{LP}}}^{\left( +\right) }$ following [19]): ${\mathbf{{LP}}}^{\left( +\right) }\left\lbrack  {19}\right\rbrack  \min \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}\log m + {z}_{X,Y}\log \lambda$ subject to

情况$\lambda  > \sqrt{m}$。对于每条边$\left( {X,Y}\right)  \in  {G}_{\mathrm{{DC}}} = \left( {{V}_{P},{E}_{P}}\right)$，定义两个变量：${x}_{X,Y}$和${z}_{X,Y}$。贾亚拉曼等人[19]表明，对于$\lambda  > \sqrt{m}$，多面体(DC)在一个常数因子范围内，是以下线性规划（按照[19]命名为${\mathrm{{LP}}}^{\left( +\right) }$）的最优值：${\mathbf{{LP}}}^{\left( +\right) }\left\lbrack  {19}\right\rbrack  \min \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}\log m + {z}_{X,Y}\log \lambda$ 满足

$$
\mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}}}\left( {{x}_{X,A} + {z}_{X,A}}\right)  + \mathop{\sum }\limits_{{\left( {A,Y}\right)  \in  {E}_{P}}}{x}_{A,Y} \geq  1\;\forall A \in  {V}_{P}
$$

$$
{x}_{X,Y} \geq  0,{z}_{X,Y} \geq  0\;\forall \left( {X,Y}\right)  \in  {E}_{P}
$$

Lemma B.1. There exists an optimal solution to $L{P}^{\left( +\right) }$ satisfying the condition that the edges in $\left\{  {\left( {X,Y}\right)  \in  {E}_{P} \mid  {z}_{X,Y} > 0}\right\}$ induce an acyclic subgraph of ${G}_{\mathrm{{DC}}}$ .

引理B.1。存在$L{P}^{\left( +\right) }$的一个最优解，满足$\left\{  {\left( {X,Y}\right)  \in  {E}_{P} \mid  {z}_{X,Y} > 0}\right\}$中的边诱导出${G}_{\mathrm{{DC}}}$的一个无环子图这一条件。

We note that while the above lemma is not expressly stated in [19], it can be extrapolated from the analysis presented in Section H. 2 of [19]. Nevertheless, the argument laid out in [19] is quite intricate. Our proof, which will be presented below, incorporates news ideas beyond their argument and is considerably shorter. Specifically, these new ideas are evidenced in the way we formulate a novel LP optimal solution in (23)-(26).

我们注意到，虽然上述引理未在文献[19]中明确表述，但可从文献[19]的H. 2节所呈现的分析中推断得出。然而，文献[19]中给出的论证相当复杂。我们将在下文给出的证明融入了超越其论证的新思想，并且篇幅明显更短。具体而言，这些新思想体现在我们在式(23) - (26)中构建一个新颖的线性规划（LP）最优解的方式上。

Proof of Lemma B.1. Consider an arbitrary optimal solution to ${\mathrm{{LP}}}^{\left( +\right) }$ that sets ${x}_{X,Y} = {x}_{X,Y}^{ * }$ and ${z}_{X,Y} = {z}_{X,Y}^{ * }$ for each $\left( {X,Y}\right)  \in  {E}_{P}$ . If the edge set $\left\{  {\left( {X,Y}\right)  \in  {E}_{P} \mid  {z}_{X,Y}^{ * } > 0}\right\}$ induces an acyclic graph,we are done. Next,we consider that ${G}_{\mathrm{{DC}}}$ contains a cycle.

引理B.1的证明。考虑${\mathrm{{LP}}}^{\left( +\right) }$的任意一个最优解，该解为每个$\left( {X,Y}\right)  \in  {E}_{P}$设定了${x}_{X,Y} = {x}_{X,Y}^{ * }$和${z}_{X,Y} = {z}_{X,Y}^{ * }$。如果边集$\left\{  {\left( {X,Y}\right)  \in  {E}_{P} \mid  {z}_{X,Y}^{ * } > 0}\right\}$构成一个无环图，那么证明完成。接下来，我们考虑${G}_{\mathrm{{DC}}}$包含一个环的情况。

Suppose that $\left( {{A}_{1},{A}_{2}}\right)$ is the edge in the cycle with the smallest ${z}_{{A}_{1},{A}_{2}}^{ * }$ (breaking ties arbitrarily). Let $\left( {{A}_{2},{A}_{3}}\right)$ be the edge succeeding $\left( {{A}_{1},{A}_{2}}\right)$ in the cycle. It thus follows that ${z}_{{A}_{2},{A}_{3}}^{ * } \geq  {z}_{{A}_{1},{A}_{2}}^{ * }$ . Define

假设$\left( {{A}_{1},{A}_{2}}\right)$是环中${z}_{{A}_{1},{A}_{2}}^{ * }$值最小的边（若有多个最小边则任意打破平局）。设$\left( {{A}_{2},{A}_{3}}\right)$是环中$\left( {{A}_{1},{A}_{2}}\right)$的后继边。由此可得${z}_{{A}_{2},{A}_{3}}^{ * } \geq  {z}_{{A}_{1},{A}_{2}}^{ * }$。定义

$$
{x}_{{A}_{2},{A}_{3}}^{\prime } = {x}_{{A}_{2},{A}_{3}}^{ * } + {z}_{{A}_{1},{A}_{2}}^{ * } \tag{23}
$$

$$
{x}_{{A}_{1},{A}_{2}}^{\prime } = {x}_{{A}_{1},{A}_{2}}^{ * } \tag{24}
$$

$$
{z}_{{A}_{2},{A}_{3}}^{\prime } = 0 \tag{25}
$$

$$
{z}_{{A}_{1},{A}_{2}}^{\prime } = 0 \tag{26}
$$

For every edge $\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  \left\{  {\left( {{A}_{1},{A}_{2}}\right) ,\left( {{A}_{2},{A}_{3}}\right) }\right\}$ ,set ${x}_{X,Y}^{\prime } = {x}_{X,Y}^{ * }$ and ${z}_{X,Y}^{\prime } = {z}_{X,Y}^{ * }$ . It is easy to verify that,for every vertex $A \in  {V}_{P}$ ,we have

对于每条边$\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  \left\{  {\left( {{A}_{1},{A}_{2}}\right) ,\left( {{A}_{2},{A}_{3}}\right) }\right\}$，设定${x}_{X,Y}^{\prime } = {x}_{X,Y}^{ * }$和${z}_{X,Y}^{\prime } = {z}_{X,Y}^{ * }$。很容易验证，对于每个顶点$A \in  {V}_{P}$，我们有

$$
\mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}}}\left( {{x}_{X,A}^{\prime } + {z}_{X,A}^{\prime }}\right)  + \mathop{\sum }\limits_{{\left( {A,Y}\right)  \in  {E}_{P}}}{x}_{A,Y}^{\prime } \geq  \mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}}}\left( {{x}_{X,A}^{ * } + {z}_{X,A}^{ * }}\right)  + \mathop{\sum }\limits_{{\left( {A,Y}\right)  \in  {E}_{P}}}{x}_{A,Y}^{ * }.
$$

Therefore, $\left\{  {{x}_{X,Y}^{\prime },{z}_{X,Y}^{\prime } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ serves as a feasible solution to ${\mathrm{{LP}}}^{\left( +\right) }$ . However:

因此，$\left\{  {{x}_{X,Y}^{\prime },{z}_{X,Y}^{\prime } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$是${\mathrm{{LP}}}^{\left( +\right) }$的一个可行解。然而：

$$
\left( {\mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}^{\prime }\log m + {z}_{X,Y}^{\prime }\log \lambda }\right)  - \left( {\mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}^{ * }\log m + {z}_{X,Y}^{ * }\log \lambda }\right) 
$$

$$
 = {z}_{{A}_{1},{A}_{2}}^{ * }\log m - \left( {{z}_{{A}_{1},{A}_{2}}^{ * } + {z}_{{A}_{2},{A}_{3}}^{ * }}\right) \log \lambda 
$$

$$
 \leq  {z}_{{A}_{1},{A}_{2}}^{ * }\log m - 2 \cdot  {z}_{{A}_{1},{A}_{2}}^{ * }\log \lambda 
$$

$$
\text{< 0} \tag{27}
$$

where the last step used the fact ${\lambda }^{2} > m$ . This contradicts the optimality of $\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  }\right.$ $\left. {E}_{P}\right\}$ .

其中最后一步利用了${\lambda }^{2} > m$这一事实。这与$\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  }\right.$ $\left. {E}_{P}\right\}$的最优性相矛盾。

We now build a set ${\mathrm{{DC}}}^{\prime }$ of degree constraints as follows. First,take an optimal solution $\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ to ${\mathrm{{LP}}}^{\left( +\right) }$ promised by Lemma B.1. Add to ${\mathrm{{DC}}}^{\prime }$ a constraint $\left( {X,\{ X,Y\} ,\lambda }\right)$ for every $\left( {X,Y}\right)  \in  {E}_{P}$ satisfying ${z}_{X,Y}^{ * } > 0$ . Then,for every edge $\left( {X,Y}\right)  \in  {E}_{P}$ ,add to ${\mathrm{{DC}}}^{\prime }$ a constraint $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ . The ${\mathrm{{DC}}}^{\prime }$ thus constructed must be acyclic. Denote by ${G}_{{\mathrm{{DC}}}^{\prime }} = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ the degree constraint graph of ${\mathrm{{DC}}}^{\prime }$ . Note that ${V}_{P} = {V}_{P}^{\prime }$ and ${E}_{P}^{\prime } \subset  {E}_{P}$ .

我们现在按如下方式构建一组度约束${\mathrm{{DC}}}^{\prime }$。首先，取引理B.1所保证的${\mathrm{{LP}}}^{\left( +\right) }$的一个最优解$\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$。对于每个满足${z}_{X,Y}^{ * } > 0$的$\left( {X,Y}\right)  \in  {E}_{P}$，将约束$\left( {X,\{ X,Y\} ,\lambda }\right)$添加到${\mathrm{{DC}}}^{\prime }$中。然后，对于每条边$\left( {X,Y}\right)  \in  {E}_{P}$，将约束$\left( {\varnothing ,\{ X,Y\} ,m}\right)$添加到${\mathrm{{DC}}}^{\prime }$中。如此构建的${\mathrm{{DC}}}^{\prime }$必定是无环的。用${G}_{{\mathrm{{DC}}}^{\prime }} = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$表示${\mathrm{{DC}}}^{\prime }$的度约束图。注意${V}_{P} = {V}_{P}^{\prime }$和${E}_{P}^{\prime } \subset  {E}_{P}$。

Lemma B.2. The ${\mathrm{{DC}}}^{\prime }$ constructed in the above manner satisfies polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  = \Theta \left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ .

引理B.2. 以上述方式构建的${\mathrm{{DC}}}^{\prime }$满足多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  = \Theta \left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$。

Proof. We will first establish polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  \geq$ polymat(DC). Remember that polymat $\left( {\mathrm{{DC}}}^{\prime }\right)$ is the optimal value of the modular LP (in its primal form) defined by ${\mathrm{{DC}}}^{\prime }$ ,as described in Section 2. Similarly,polymat(DC) is the optimal value of the modular LP defined by DC. Given that ${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$ , the LP defined by ${\mathrm{{DC}}}^{\prime }$ incorporates only a subset of the constraints found in the LP defined by $\mathrm{{DC}}$ . Therefore,it must be the case that polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  \geq$ polymat(DC).

证明。我们首先将证明多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  \geq$ ≤ 多面体(DC)。请记住，多面体$\left( {\mathrm{{DC}}}^{\prime }\right)$是由${\mathrm{{DC}}}^{\prime }$定义的模块化线性规划（其原始形式）的最优值，如第2节所述。类似地，多面体(DC)是由DC定义的模块化线性规划的最优值。鉴于${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$，由${\mathrm{{DC}}}^{\prime }$定义的线性规划仅包含由$\mathrm{{DC}}$定义的线性规划中约束的一个子集。因此，必然有 多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  \geq$ ≤ 多面体(DC)。

The rest of the proof will show polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  = O$ (polymat(DC)),which will establish the lemma Consider the following LP:

证明的其余部分将证明多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  = O$ ≥ (多面体(DC))，这将证明该引理。考虑以下线性规划：

${\mathbf{{LP}}}_{\mathbf{1}}^{\left( +\right) }\;\min \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}\log m + {z}_{X,Y}\log \lambda$ subject to

${\mathbf{{LP}}}_{\mathbf{1}}^{\left( +\right) }\;\min \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}\log m + {z}_{X,Y}\log \lambda$ 满足

$$
\mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}}}{x}_{X,A} + \mathop{\sum }\limits_{{\left( {A,Y}\right)  \in  {E}_{P}}}{x}_{A,Y} + \mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}^{\prime }}}{z}_{X,A} \geq  1\;\forall A \in  {\mathcal{V}}_{P}
$$

$$
{x}_{X,Y} \geq  0,{z}_{X,Y} \geq  0\;\forall \left( {X,Y}\right)  \in  {E}_{P}
$$

The condition $\left( {X,A}\right)  \in  {E}_{P}^{\prime }$ in the first inequality marks the difference between ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ and ${\mathrm{{LP}}}^{\left( +\right) }$ . Note that the two LPs have the same objective function.

第一个不等式中的条件$\left( {X,A}\right)  \in  {E}_{P}^{\prime }$标志着${\mathrm{{LP}}}_{1}^{\left( +\right) }$和${\mathrm{{LP}}}^{\left( +\right) }$之间的差异。注意，这两个线性规划具有相同的目标函数。

Claim 1: ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ and ${\mathrm{{LP}}}^{\left( +\right) }$ have the same optimal value.

命题1：${\mathrm{{LP}}}_{1}^{\left( +\right) }$和${\mathrm{{LP}}}^{\left( +\right) }$具有相同的最优值。

To prove the claim,first observe that any feasible solution $\left\{  {{x}_{X,Y},{z}_{X,Y} \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ to ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ is also a feasible solution to ${\mathrm{{LP}}}^{\left( +\right) }$ . Hence,the optimal value of ${\mathrm{{LP}}}^{\left( +\right) }$ cannot exceed that of ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ . On the other hand,recall that earlier we have identified an optimal solution $\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ to ${\mathrm{{LP}}}^{\left( +\right) }$ . By how ${\mathrm{{DC}}}^{\prime }$ is built from that solution and how ${G}_{{\mathrm{{DC}}}^{\prime }} = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ is built from ${\mathrm{{DC}}}^{\prime }$ ,it must hold that ${z}_{X,Y}^{ * } = 0$ for every $\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$ . Hence, $\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ makes a feasible solution to ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ . This implies that $\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ must be an optimal solution to ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ . Claim 1 now follows.

为证明该命题，首先观察到，${\mathrm{{LP}}}_{1}^{\left( +\right) }$的任何可行解$\left\{  {{x}_{X,Y},{z}_{X,Y} \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$也是${\mathrm{{LP}}}^{\left( +\right) }$的可行解。因此，${\mathrm{{LP}}}^{\left( +\right) }$的最优值不会超过${\mathrm{{LP}}}_{1}^{\left( +\right) }$的最优值。另一方面，回顾一下，我们之前已经确定了${\mathrm{{LP}}}^{\left( +\right) }$的一个最优解$\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$。根据如何从该解构建${\mathrm{{DC}}}^{\prime }$以及如何从${\mathrm{{DC}}}^{\prime }$构建${G}_{{\mathrm{{DC}}}^{\prime }} = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$，对于每个$\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$，必然有${z}_{X,Y}^{ * } = 0$成立。因此，$\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$构成了${\mathrm{{LP}}}_{1}^{\left( +\right) }$的一个可行解。这意味着$\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$必定是${\mathrm{{LP}}}_{1}^{\left( +\right) }$的一个最优解。命题1得证。

Consider another LP:

考虑另一个线性规划问题（LP）：

${\mathbf{{LP}}}_{\mathbf{2}}^{\left( +\right) }\;\min \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}\log m + \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}^{\prime }}}{z}_{X,Y}\log \lambda$ subject to

${\mathbf{{LP}}}_{\mathbf{2}}^{\left( +\right) }\;\min \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}}}{x}_{X,Y}\log m + \mathop{\sum }\limits_{{\left( {X,Y}\right)  \in  {E}_{P}^{\prime }}}{z}_{X,Y}\log \lambda$ 满足约束条件

$$
\mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}}}{x}_{X,A} + \mathop{\sum }\limits_{{\left( {A,Y}\right)  \in  {E}_{P}}}{x}_{A,Y} + \mathop{\sum }\limits_{{\left( {X,A}\right)  \in  {E}_{P}^{\prime }}}{z}_{X,A} \geq  1\;\forall A \in  {\mathcal{V}}_{P}
$$

$$
{x}_{X,Y} \geq  0\;\forall \left( {X,Y}\right)  \in  {E}_{P}
$$

$$
{z}_{X,Y} \geq  0\;\forall \left( {X,Y}\right)  \in  {E}_{P}^{\prime }
$$

${\mathrm{{LP}}}_{2}^{\left( +\right) }$ differs from ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ in that the former drops the variables ${z}_{X,Y}$ of those edges $\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$ . This happens both in the constraints and the objective function.

${\mathrm{{LP}}}_{2}^{\left( +\right) }$与${\mathrm{{LP}}}_{1}^{\left( +\right) }$的不同之处在于，前者去掉了那些边$\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$对应的变量${z}_{X,Y}$。这在约束条件和目标函数中均有体现。

Claim 2: ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ and ${\mathrm{{LP}}}_{2}^{\left( +\right) }$ have the same optimal value.

命题2：${\mathrm{{LP}}}_{1}^{\left( +\right) }$和${\mathrm{{LP}}}_{2}^{\left( +\right) }$具有相同的最优值。

To prove the claim,first observe that,given a feasible solution $\left\{  {{x}_{X,Y} \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}   \cup  \left\{  {{z}_{X,Y} \mid  (X}\right.$ , $\left. {\left. Y\right)  \in  {E}_{P}^{\prime }}\right\}$ to ${\mathrm{{LP}}}_{2}^{\left( +\right) }$ ,we can extend it into a feasible solution to ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ by padding ${Z}_{X,Y} = 0$ for each $\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$ . Hence,the optimal value of ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ cannot exceed that of ${\mathrm{{LP}}}_{2}^{\left( +\right) }$ . On the other hand,as mentioned before, $\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$ is an optimal solution to ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ . In this solution, ${z}_{X,Y}^{ * } = 0$ for every $\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$ . Thus, $\left\{  {{x}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}   \cup  \left\{  {{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}^{\prime }}\right\}$ makes a feasible solution to ${\mathrm{{LP}}}_{2}^{\left( +\right) }$ ,achieving the same objective function value as the optimal value of ${\mathrm{{LP}}}_{1}^{\left( +\right) }$ . Claim 2 now follows.

为证明该命题，首先注意到，给定${\mathrm{{LP}}}_{2}^{\left( +\right) }$的一个可行解$\left\{  {{x}_{X,Y} \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}   \cup  \left\{  {{z}_{X,Y} \mid  (X}\right.$、$\left. {\left. Y\right)  \in  {E}_{P}^{\prime }}\right\}$，我们可以通过为每个$\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$填充${Z}_{X,Y} = 0$，将其扩展为${\mathrm{{LP}}}_{1}^{\left( +\right) }$的一个可行解。因此，${\mathrm{{LP}}}_{1}^{\left( +\right) }$的最优值不会超过${\mathrm{{LP}}}_{2}^{\left( +\right) }$的最优值。另一方面，如前所述，$\left\{  {{x}_{X,Y}^{ * },{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}$是${\mathrm{{LP}}}_{1}^{\left( +\right) }$的一个最优解。在这个解中，对于每个$\left( {X,Y}\right)  \in  {E}_{P} \smallsetminus  {E}_{P}^{\prime }$都有${z}_{X,Y}^{ * } = 0$。因此，$\left\{  {{x}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}}\right\}   \cup  \left\{  {{z}_{X,Y}^{ * } \mid  \left( {X,Y}\right)  \in  {E}_{P}^{\prime }}\right\}$构成了${\mathrm{{LP}}}_{2}^{\left( +\right) }$的一个可行解，其目标函数值与${\mathrm{{LP}}}_{1}^{\left( +\right) }$的最优值相同。命题2得证。

Finally,notice that ${\mathrm{{LP}}}_{2}^{\left( +\right) }$ is exactly the dual modular LP defined by ${\mathrm{{DC}}}^{\prime }$ . Hence, $\log \left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$ is exactly the optimal value of ${\mathrm{{LP}}}_{2}^{\left( +\right) }$ . Thus,polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  = O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ can now be derived from the above discussion and the fact that $\log \left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ is asymptotically the optimal value of ${\mathrm{{LP}}}^{\left( +\right) }$ .

最后，注意到${\mathrm{{LP}}}_{2}^{\left( +\right) }$恰好是由${\mathrm{{DC}}}^{\prime }$定义的对偶模线性规划（Dual Modular Linear Programming）。因此，$\log \left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$恰好是${\mathrm{{LP}}}_{2}^{\left( +\right) }$的最优值。因此，多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  = O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$现在可以从上述讨论以及$\log \left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$渐近于${\mathrm{{LP}}}^{\left( +\right) }$的最优值这一事实推导得出。

Case $\lambda  \leq  \sqrt{m}$ . Let us first define several concepts. A directed star refers to a directed graph where there are $t \geq  2$ vertices,among which one vertex,designated the center,has $t - 1$ edges (in-coming and out-going edges combined), and every other vertex, called a petal, has only one edge (which can be an in-coming or out-going edge). Now,consider a directed bipartite graph between ${U}_{1}$ and ${U}_{2}$ , each being an independent sets of vertices (an edge may point from one vertex in ${U}_{1}$ to a vertex in ${U}_{2}$ ,or vice versa). A directed star cover of the bipartite graph is a set of directed stars such that

情况 $\lambda  \leq  \sqrt{m}$ 。让我们首先定义几个概念。有向星（directed star）指的是一个有 $t \geq  2$ 个顶点的有向图，其中一个顶点被指定为中心，有 $t - 1$ 条边（入边和出边之和），而其他每个顶点，称为花瓣，只有一条边（可以是入边或出边）。现在，考虑 ${U}_{1}$ 和 ${U}_{2}$ 之间的有向二分图， ${U}_{1}$ 和 ${U}_{2}$ 各自是独立的顶点集（一条边可以从 ${U}_{1}$ 中的一个顶点指向 ${U}_{2}$ 中的一个顶点，反之亦然）。二分图的有向星覆盖是一组有向星，使得

- each directed star is a subgraph of the bipartite graph,

- 每个有向星都是二分图的子图，

- no two directed stars share a common edge, and

- 任意两个有向星没有公共边，并且

- every vertex in ${U}_{1} \cup  {U}_{2}$ appears in exactly one directed star.

- ${U}_{1} \cup  {U}_{2}$ 中的每个顶点恰好出现在一个有向星中。

A directed star cover is minimum if it has the least number of edges, counting all directed stars in the cover.

如果一个有向星覆盖包含的所有有向星的边数最少，那么它就是最小有向星覆盖。

Next, we review an expression about polymat(DC) derived in [19]. Find all the strongly connected components (SCCs) of ${G}_{\mathrm{{DC}}} = \left( {{V}_{P},{E}_{P}}\right)$ . Adopting terms from [19],an SCC is classified as (i) a source if it has no in-coming edge from another SCC, or a non-source otherwise; (ii) trivial if it consists of a single vertex, or non-trivial otherwise. Define:

接下来，我们回顾一下文献 [19] 中推导的关于多面体（polymat，DC）的一个表达式。找出 ${G}_{\mathrm{{DC}}} = \left( {{V}_{P},{E}_{P}}\right)$ 的所有强连通分量（SCCs，strongly connected components）。采用文献 [19] 中的术语，一个强连通分量被分类为：（i）如果它没有来自其他强连通分量的入边，则为源（source），否则为非源；（ii）如果它由单个顶点组成，则为平凡的（trivial），否则为非平凡的。定义：

- $S =$ the set of vertices in ${G}_{\mathrm{{DC}}}$ each forming a trivial source SCC by itself.

- $S =$ ： ${G}_{\mathrm{{DC}}}$ 中每个自身构成一个平凡源强连通分量的顶点的集合。

- $T =$ the set of vertices in ${G}_{\mathrm{{DC}}}$ receiving an in-coming edge from at least one vertex in $S$ .

- $T =$ ： ${G}_{\mathrm{{DC}}}$ 中至少从 $S$ 中的一个顶点接收一条入边的顶点的集合。

Take a minimum directed star cover of the directed bipartite graph induced by $S$ and $T$ . Define

取由 $S$ 和 $T$ 诱导的有向二分图的最小有向星覆盖。定义

- ${S}_{1} =$ the set of vertices in $S$ each serving as the center of some directed star in the cover.

- ${S}_{1} =$ ： $S$ 中每个在覆盖中作为某个有向星中心的顶点的集合。

- ${S}_{2} = S \smallsetminus  {S}_{1}$ .

- ${T}_{2} =$ the set of vertices in $T$ each serving as the center of some directed star in the cover.

- ${T}_{2} =$ ： $T$ 中每个在覆盖中作为某个有向星中心的顶点的集合。

- ${T}_{1} = T \smallsetminus  {T}_{2}$ .

Note that the meanings of the symbols ${S}_{1},{S}_{2},{T}_{1}$ ,and ${T}_{2}$ follow exactly those in [19] for the reader’s convenience (in particular,note the semantics of ${T}_{1}$ and ${T}_{2}$ ).

注意，为了方便读者，符号 ${S}_{1},{S}_{2},{T}_{1}$ 和 ${T}_{2}$ 的含义与文献 [19] 中完全一致（特别要注意 ${T}_{1}$ 和 ${T}_{2}$ 的语义）。

We now introduce three quantities:

我们现在引入三个量：

- ${c}_{1}$ : the number of non-trivial source SCCs;

- ${c}_{1}$ ：非平凡源强连通分量的数量；

- ${n}_{1}$ : the total number of vertices in non-trivial source SCCs;

- ${n}_{1}$ ：非平凡源强连通分量中顶点的总数；

- ${n}_{2} = \left| {V}_{P}\right|  - {n}_{1} - \left| S\right|  - \left| T\right|$ .

Jayaraman et al. [19] showed:

贾亚拉曼（Jayaraman）等人 [19] 证明了：

$$
{\text{ polymat }}_{\text{dir }}\left( {m,\lambda ,P}\right)  = \Theta \left( {{m}^{{c}_{1} + \left| S\right| } \cdot  {\lambda }^{{n}_{1} + {n}_{2} + \left| {T}_{1}\right|  - 2{c}_{1} - \left| {S}_{1}\right| }}\right) . \tag{28}
$$

Let ${G}_{\mathrm{{DC}}}^{\prime } = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ be an arbitrary weakly-connected acyclic subgraph of ${G}_{\mathrm{{DC}}}$ satisfying all the conditions below.

设 ${G}_{\mathrm{{DC}}}^{\prime } = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ 为 ${G}_{\mathrm{{DC}}}$ 的任意一个弱连通无环子图，且满足以下所有条件。

- ${V}_{P} = {V}_{P}^{\prime }$ .

- ${E}_{P}^{\prime }$ contains all the edges in the minimum directed star cover identified earlier.

- ${E}_{P}^{\prime }$ 包含先前确定的最小有向星覆盖中的所有边。

- In each non-trivial source SCC, every vertex, except for one, has one in-coming edge included in ${E}_{P}^{\prime }$ . We will refer to the vertex $X$ with no in-coming edges in ${E}_{P}^{\prime }$ as the SCC’s root. The fact that every other vertex $Y$ in the SCC has an in-coming edge in ${E}_{P}^{\prime }$ implies $\left( {X,Y}\right)  \in  {E}_{P}^{\prime }$ for at least one $Y$ . We designate one such(X,Y)as the SCC’s main edge.

- 在每个非平凡源强连通分量（SCC，Strongly Connected Component）中，除一个顶点外，每个顶点都有一条入边包含在 ${E}_{P}^{\prime }$ 中。我们将 ${E}_{P}^{\prime }$ 中没有入边的顶点 $X$ 称为该强连通分量的根。强连通分量中其他每个顶点 $Y$ 在 ${E}_{P}^{\prime }$ 中有入边这一事实意味着对于至少一个 $Y$ 有 $\left( {X,Y}\right)  \in  {E}_{P}^{\prime }$ 。我们指定这样的一个 (X, Y) 作为该强连通分量的主边。

- In each non-trivial non-source SCC,every vertex has an in-coming edge included in ${E}_{P}^{\prime }$ .

- 在每个非平凡非源强连通分量中，每个顶点都有一条入边包含在 ${E}_{P}^{\prime }$ 中。

It is rudimentary to verify that such a subgraph ${G}_{\mathrm{{DC}}}^{\prime }$ must exist.

很容易验证这样的子图 ${G}_{\mathrm{{DC}}}^{\prime }$ 一定存在。

From ${G}_{\mathrm{{DC}}} = \left( {{V}_{P},{E}_{P}}\right)$ and ${G}_{\mathrm{{DC}}}^{\prime } = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ ,we create a set ${\mathrm{{DC}}}^{\prime }$ of degree constraints as follows.

根据 ${G}_{\mathrm{{DC}}} = \left( {{V}_{P},{E}_{P}}\right)$ 和 ${G}_{\mathrm{{DC}}}^{\prime } = \left( {{V}_{P}^{\prime },{E}_{P}^{\prime }}\right)$ ，我们按如下方式创建一个度约束集合 ${\mathrm{{DC}}}^{\prime }$ 。

- For each edge $\left( {X,Y}\right)  \in  {E}_{P}$ (note: not ${E}_{P}^{\prime }$ ),add a constraint $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ to ${\mathrm{{DC}}}^{\prime }$ .

- 对于每条边 $\left( {X,Y}\right)  \in  {E}_{P}$ （注意：不是 ${E}_{P}^{\prime }$ ），向 ${\mathrm{{DC}}}^{\prime }$ 中添加一个约束 $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ 。

- We inspect each directed star in the minimum directed star cover and distinguish two possibilities.

- 我们检查最小有向星覆盖中的每个有向星，并区分两种可能性。

- Scenario 1: The star’s center $X$ comes from ${S}_{1}$ . Let the star’s petals be ${Y}_{1},{Y}_{2},\ldots ,{Y}_{t}$ for some $t \geq  1$ ; the ordering of the petals does not matter. For each $i \in  \left\lbrack  {t - 1}\right\rbrack$ ,we add a constraint $\left( {\{ X\} ,\left\{  {X,{Y}_{i}}\right\}  ,\lambda }\right)$ to ${\mathrm{{DC}}}^{\prime }$ . We will refer to $\left( {X,{Y}_{t}}\right)$ as the star’s main edge.

- 场景 1：星的中心 $X$ 来自 ${S}_{1}$ 。设星的花瓣为对于某个 $t \geq  1$ 的 ${Y}_{1},{Y}_{2},\ldots ,{Y}_{t}$ ；花瓣的顺序无关紧要。对于每个 $i \in  \left\lbrack  {t - 1}\right\rbrack$ ，我们向 ${\mathrm{{DC}}}^{\prime }$ 中添加一个约束 $\left( {\{ X\} ,\left\{  {X,{Y}_{i}}\right\}  ,\lambda }\right)$ 。我们将 $\left( {X,{Y}_{t}}\right)$ 称为星的主边。

- Scenario 2: The star’s center $X$ comes from ${T}_{2}$ . Nothing needs to be done.

- 场景 2：星的中心 $X$ 来自 ${T}_{2}$ 。无需做任何操作。

- Consider now each non-trivial source SCC. Remember that every vertex $Y$ ,other than the SCC’s root,has an in-coming edge $\left( {X,Y}\right)  \in  {E}_{P}^{\prime }$ . For every such $Y$ ,if(X,Y)is not the SCC’s main edge,add a constraint $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ to ${\mathrm{{DC}}}^{\prime }$ .

- 现在考虑每个非平凡源强连通分量。请记住，除强连通分量的根之外的每个顶点 $Y$ 都有一条入边 $\left( {X,Y}\right)  \in  {E}_{P}^{\prime }$ 。对于每个这样的 $Y$ ，如果 (X, Y) 不是该强连通分量的主边，则向 ${\mathrm{{DC}}}^{\prime }$ 中添加一个约束 $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ 。

- Finally,we examine each non-source SCC. As mentioned,every vertex $Y$ in such an SCC has an in-coming edge $\left( {X,Y}\right)  \in  {E}_{P}^{\prime }$ . For every $Y$ ,add a constraint $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ to ${\mathrm{{DC}}}^{\prime }$ .

- 最后，我们检查每个非源强连通分量。如前所述，这样的强连通分量中的每个顶点 $Y$ 都有一条入边 $\left( {X,Y}\right)  \in  {E}_{P}^{\prime }$ 。对于每个 $Y$ ，向 ${\mathrm{{DC}}}^{\prime }$ 中添加一个约束 $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ 。

The rest of the proof will show polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  = \Theta$ (polymat(DC)). As ${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$ ,we must have polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  \geq$ polymat(DC)following the same reasoning used in the $\lambda  > \sqrt{m}$ case.

证明的其余部分将证明多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  = \Theta$（多面体(DC)）。由于${\mathrm{{DC}}}^{\prime } \subset  \mathrm{{DC}}$，按照在$\lambda  > \sqrt{m}$情形中使用的相同推理，我们必定有多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  \geq$ 多面体(DC)。

We will now proceed to argue that polymat $\left( {\mathrm{{DC}}}^{\prime }\right)  = O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$ . Recall that $\log \left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$ is the optimal value of the dual modular LP of ${\mathrm{{DC}}}^{\prime }$ (see Section 2). On the other hand,the value of polymat(DC) satisfies (28). In the following, we will construct a feasible solution to the dual modular LP of ${\mathrm{{DC}}}^{\prime }$ under which the LP’s objective function achieves the value of

现在我们将着手论证多面体$\left( {\mathrm{{DC}}}^{\prime }\right)  = O\left( {\text{polymat}\left( \mathrm{{DC}}\right) }\right)$。回想一下，$\log \left( {\text{polymat}\left( {\mathrm{{DC}}}^{\prime }\right) }\right)$是${\mathrm{{DC}}}^{\prime }$的对偶模线性规划（见第2节）的最优值。另一方面，多面体(DC)的值满足(28)。接下来，我们将构造${\mathrm{{DC}}}^{\prime }$的对偶模线性规划的一个可行解，在此解下该线性规划的目标函数达到

$$
\left( {\left( {{c}_{1} + \left| S\right| }\right)  \cdot  \log m}\right)  + \left( {{n}_{1} + {n}_{2} + \left| {T}_{1}\right|  - 2{c}_{1} - \left| {S}_{1}\right| }\right)  \cdot  \log \lambda  \tag{29}
$$

which will be sufficient for proving Lemma B.1.

这足以证明引理B.1。

The dual modular LP associates every constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  {\mathrm{{DC}}}^{\prime }$ with a variable ${\delta }_{\mathcal{Y} \mid  \mathcal{X}}$ . We determine these variables' values as follows.

对偶模线性规划将每个约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  {\mathrm{{DC}}}^{\prime }$与一个变量${\delta }_{\mathcal{Y} \mid  \mathcal{X}}$关联起来。我们按如下方式确定这些变量的值。

- For every constraint $\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  {\mathrm{{DC}}}^{\prime }$ where ${N}_{\mathcal{Y} \mid  \mathcal{X}} = \lambda$ ,set ${\delta }_{\mathcal{Y} \mid  \mathcal{X}} = 1$ .

- 对于每个约束$\left( {\mathcal{X},\mathcal{Y},{N}_{\mathcal{Y} \mid  \mathcal{X}}}\right)  \in  {\mathrm{{DC}}}^{\prime }$，其中${N}_{\mathcal{Y} \mid  \mathcal{X}} = \lambda$，令${\delta }_{\mathcal{Y} \mid  \mathcal{X}} = 1$。

- Consider each directed star in the minimum directed star.

- 考虑最小有向星中的每个有向星。

- Scenario 1: The star’s center $X$ comes from ${S}_{1}$ . For the star’s main edge(X,Y),the constraint $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ exists in ${\mathrm{{DC}}}^{\prime }$ . Set ${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$ .

- 场景1：星的中心$X$来自${S}_{1}$。对于星的主边(X,Y)，约束$\left( {\varnothing ,\{ X,Y\} ,m}\right)$存在于${\mathrm{{DC}}}^{\prime }$中。令${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$。

- Scenario 2: The star’s center $X$ comes from ${T}_{2}$ . For every petal $Y$ of the star,the constraint $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ exists in ${\mathrm{{DC}}}^{\prime }$ . Set ${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$ .

- 场景2：星的中心$X$来自${T}_{2}$。对于星的每个花瓣$Y$，约束$\left( {\varnothing ,\{ X,Y\} ,m}\right)$存在于${\mathrm{{DC}}}^{\prime }$中。令${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$。

- Consider each non-trivial source SCC. Let(X,Y)be the main edge of the SCC. The constraint $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ exists in ${\mathrm{{DC}}}^{\prime }$ . Set ${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$ .

- 考虑每个非平凡源强连通分量(SCC)。设(X,Y)为该强连通分量的主边。约束$\left( {\varnothing ,\{ X,Y\} ,m}\right)$存在于${\mathrm{{DC}}}^{\prime }$中。令${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$。

The other variables that have not yet been mentioned are all set to 0 .

尚未提及的其他变量都设为0。

It is tedious but straightforward to verify that all the constraints of the dual modular LP are fulfilled. To confirm that the objective function indeed evaluates to (29), observe:

验证对偶模线性规划的所有约束都得到满足是繁琐但直接的。为确认目标函数确实计算为(29)，观察：

- There are ${c}_{1} + \left| S\right|$ constraints of the form $\left( {\varnothing ,\{ X,Y\} ,m}\right)$ with ${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$ . Specifically, ${c}_{1}$ of them come from the roots of the non-trivial source SCCs, $\left| {S}_{1}\right|$ of them come from the star center vertices in ${S}_{1}$ ,and $\left| {S}_{2}\right|$ of them come from the petal vertices in ${S}_{2}$ .

- 有${c}_{1} + \left| S\right|$个形如$\left( {\varnothing ,\{ X,Y\} ,m}\right)$且${\delta }_{\{ X,Y\}  \mid  \varnothing } = 1$的约束。具体而言，其中${c}_{1}$个来自非平凡源强连通分量的根，$\left| {S}_{1}\right|$个来自${S}_{1}$中的星中心顶点，$\left| {S}_{2}\right|$个来自${S}_{2}$中的花瓣顶点。

- There are ${n}_{1} + {n}_{2} + \left| {T}_{1}\right|  - 2{c}_{1} - \left| {S}_{1}\right|$ of the form $\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$ with ${\delta }_{\{ X,Y\} \mid \{ X\} } = 1$ . Specifically, ${n}_{1} - 2{c}_{1}$ of them come from the non-main edges of the non-trivial source SCCs, ${n}_{2}$ of them come from the vertices that are not in any non-trivial source SCC and are not in $S \cup  T$ ,and $\left| {T}_{1}\right|  - \left| {S}_{1}\right|$ of them come from the petal vertices that are (i) in ${T}_{1}$ but (ii) not in the main edges of their respective stars.

- 存在${n}_{1} + {n}_{2} + \left| {T}_{1}\right|  - 2{c}_{1} - \left| {S}_{1}\right|$个形如$\left( {\{ X\} ,\{ X,Y\} ,\lambda }\right)$且满足${\delta }_{\{ X,Y\} \mid \{ X\} } = 1$的元素。具体而言，其中${n}_{1} - 2{c}_{1}$个来自非平凡源强连通分量（SCC）的非主边，${n}_{2}$个来自不在任何非平凡源强连通分量中且不在$S \cup  T$中的顶点，$\left| {T}_{1}\right|  - \left| {S}_{1}\right|$个来自满足以下条件的花瓣顶点：（i）在${T}_{1}$中，但（ii）不在其各自星型结构的主边上。

We now conclude the whole proof of Lemma 4.1.

我们现在完成引理4.1的整个证明。

## C Proof of Lemma 5.1

## C 引理5.1的证明

By definition, $\log \left( {{\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right) }\right)$ equals the optimal value of the following LP:

根据定义，$\log \left( {{\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right) }\right)$等于以下线性规划（LP）的最优值：

Polymatroid LP: maximize $h\left( {V}_{P}\right)$ ,from all set functions $h\left( \cdot \right)$ over ${V}_{P}$ ,subject to

多面体线性规划：在所有定义于${V}_{P}$上的集合函数$h\left( \cdot \right)$中最大化$h\left( {V}_{P}\right)$，满足

(I) $h\left( \varnothing \right)  = 0$

(II) $h\left( {\{ X,Y\} }\right)  \leq  \log m\;\forall \{ X,Y\}  \in  {E}_{P}$

(II) $h\left( {\{ X,Y\} }\right)  \leq  \log m\;\forall \{ X,Y\}  \in  {E}_{P}$

(III) $h\left( {\{ X,Y\} }\right)  - h\left( {\{ X\} }\right)  \leq  \log \lambda \;\forall \{ X,Y\}  \in  {E}_{P}$

(III) $h\left( {\{ X,Y\} }\right)  - h\left( {\{ X\} }\right)  \leq  \log \lambda \;\forall \{ X,Y\}  \in  {E}_{P}$

(IV) $h\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + h\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  \leq  h\left( \mathcal{X}\right)  + h\left( \mathcal{Y}\right) \;\forall \mathcal{X},\mathcal{Y} \subseteq  {V}_{P}$

(IV) $h\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + h\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  \leq  h\left( \mathcal{X}\right)  + h\left( \mathcal{Y}\right) \;\forall \mathcal{X},\mathcal{Y} \subseteq  {V}_{P}$

(V) $h\left( \mathcal{X}\right)  \leq  h\left( \mathcal{Y}\right) \;\forall \mathcal{X} \subseteq  \mathcal{Y} \subseteq  {V}_{P}$

To see that the above is an instance of linear programming,one can view a set function $h$ over ${V}_{P}$ as a point in a ${2}^{\left| {V}_{P}\right| }$ -dimensional space,where each dimension is a different subset of ${V}_{P}$ . Consequently, for any subset $\mathcal{X} \subseteq  {V}_{P},h\left( \mathcal{X}\right)$ can be regarded as the "coordinate" of this point on the dimension $\mathcal{X}$ .

为了说明上述是线性规划的一个实例，可以将定义在${V}_{P}$上的集合函数$h$视为${2}^{\left| {V}_{P}\right| }$维空间中的一个点，其中每个维度对应${V}_{P}$的一个不同子集。因此，对于任何子集$\mathcal{X} \subseteq  {V}_{P},h\left( \mathcal{X}\right)$，可以将其视为该点在维度$\mathcal{X}$上的“坐标”。

### C.1 Upper Bounding the Objective Function

### C.1 目标函数的上界

First,we will show that (22) is an upper bound of ${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$ . More precisely,for every feasible solution $h\left( \text{.}\right) {tothepolymatroidLP},{wewillprove} :$

首先，我们将证明(22)是${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$的一个上界。更准确地说，对于每个可行解$h\left( \text{.}\right) {tothepolymatroidLP},{wewillprove} :$

$$
h\left( {V}_{P}\right)  \leq  \log m + \left( {k - 2}\right) \log \lambda  \tag{30}
$$

$$
h\left( {V}_{P}\right)  \leq  \left( {{k}_{\text{cycle }}/2 + \beta }\right) \log m + \left( {{k}_{\text{star }} - {2\beta }}\right) \log \lambda . \tag{31}
$$

This implies that ${\exp }_{2}\left( {h\left( {V}_{P}\right) }\right)$ is always bounded by the right hand side of (22) and,hence,so is ${\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$ .

这意味着${\exp }_{2}\left( {h\left( {V}_{P}\right) }\right)$始终受(22)式右边的约束，因此${\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$也是如此。

Proof of (30). Let us order the vertices in ${V}_{P}$ as ${A}_{1},{A}_{2},\ldots ,{A}_{k}$ with the property that,for each $i \in  \left\lbrack  {2,k}\right\rbrack$ ,the vertex ${A}_{i}$ is adjacent in $P$ to at least one vertex ${A}_{j}$ with $j < i$ . We will denote this value of $j$ as $\operatorname{back}\left( i\right)$ . Such an ordering definitely exists because $P$ is connected. For $i \geq  3$ ,define ${A}_{\left\lbrack  i\right\rbrack  } = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{i}}\right\}  .$

(30)的证明。我们将${V}_{P}$中的顶点按${A}_{1},{A}_{2},\ldots ,{A}_{k}$排序，使得对于每个$i \in  \left\lbrack  {2,k}\right\rbrack$，顶点${A}_{i}$在$P$中与至少一个满足$j < i$的顶点${A}_{j}$相邻。我们将$j$的这个值记为$\operatorname{back}\left( i\right)$。这样的排序肯定存在，因为$P$是连通的。对于$i \geq  3$，定义${A}_{\left\lbrack  i\right\rbrack  } = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{i}}\right\}  .$

To start with, let us observe that, by the constraint (IV) of the polymatroid LP, the inequality

首先，让我们注意到，根据多拟阵线性规划的约束条件（IV），不等式

$$
h\left( {A}_{\left\lbrack  i\right\rbrack  }\right)  + h\left( {A}_{\text{back }\left( i\right) }\right)  \leq  h\left( {A}_{\left\lbrack  i - 1\right\rbrack  }\right)  + h\left( {{A}_{\text{back }\left( i\right) },{A}_{i}}\right)  \tag{32}
$$

holds for all $i \in  \left\lbrack  {2,k}\right\rbrack$ . Thus:

对所有$i \in  \left\lbrack  {2,k}\right\rbrack$都成立。因此：

$$
h\left( {V}_{P}\right)  = h\left( {{A}_{1},{A}_{2}}\right)  + \mathop{\sum }\limits_{{i = 3}}^{k}h\left( {A}_{\left\lbrack  i\right\rbrack  }\right)  - h\left( {A}_{\left\lbrack  i - 1\right\rbrack  }\right) 
$$

$$
\text{(by (32))} \leq  h\left( {{A}_{1},{A}_{2}}\right)  + \mathop{\sum }\limits_{{i = 3}}^{k}h\left( {{A}_{{back}\left( i\right) },{A}_{i}}\right)  - h\left( {A}_{{back}\left( i\right) }\right) 
$$

$$
 \leq  \log m + \left( {k - 2}\right) \log \lambda \text{.}
$$

The last step used constraints (II) and (III) of the polymatroid LP.

最后一步使用了多拟阵线性规划的约束条件（II）和（III）。

Proof of (31). For each cycle ${ ○ }_{i}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ in the fractional edge-cover decomposition of $P$ ,define $V\left( {\square }_{i}\right)$ as the set of vertices in ${\square }_{i}$ and set $k\left( {\square }_{i}\right)  = \left| {V\left( {\square }_{i}\right) }\right|$ . Likewise,for each star ${ \star  }_{j}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)$ , define $V\left( { \star  }_{j}\right)$ as the set of vertices in ${ \star  }_{j}$ and set $k\left( { \star  }_{j}\right)  = \left| {V\left( { \star  }_{j}\right) }\right|$ . We aim to prove

(31)的证明。对于$P$的分数边覆盖分解中的每个圈${ ○ }_{i}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$，将$V\left( {\square }_{i}\right)$定义为${\square }_{i}$中的顶点集，并设$k\left( {\square }_{i}\right)  = \left| {V\left( {\square }_{i}\right) }\right|$。同样，对于每个星${ \star  }_{j}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)$，将$V\left( { \star  }_{j}\right)$定义为${ \star  }_{j}$中的顶点集，并设$k\left( { \star  }_{j}\right)  = \left| {V\left( { \star  }_{j}\right) }\right|$。我们的目标是证明

$$
\text{for each}i \in  \left\lbrack  \alpha \right\rbrack   : h\left( {V\left( {\square }_{i}\right) }\right)  \leq  \left( {k\left( {\square }_{i}\right) /2}\right) \log m\text{;} \tag{33}
$$

$$
\text{for each}j \in  \left\lbrack  \beta \right\rbrack   : h\left( {V\left( { \star  }_{j}\right) }\right)  \leq  \log m + \left( {k\left( { \star  }_{j}\right)  - 2}\right) \log \lambda \text{.} \tag{34}
$$

Once this is done, we can establish (31) as follows. First, from constraint (IV), we know $h\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  \leq  h\left( \mathcal{X}\right)  + h\left( \mathcal{Y}\right)$ for any disjoint $\mathcal{X},\mathcal{Y} \subseteq  {V}_{P}$ ; call this the disjointness rule. Then,we can derive

一旦完成这一步，我们可以按如下方式证明(31)。首先，根据约束条件（IV），我们知道对于任何不相交的$\mathcal{X},\mathcal{Y} \subseteq  {V}_{P}$，有$h\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  \leq  h\left( \mathcal{X}\right)  + h\left( \mathcal{Y}\right)$；我们称此为不相交规则。然后，我们可以推导出

$$
h\left( {V}_{P}\right)  \leq  \mathop{\sum }\limits_{{i = 1}}^{\alpha }h\left( {V\left( {\square }_{i}\right) }\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }h\left( {V\left( { \star  }_{j}\right) }\right) \text{ (disjointness rule) }
$$

$$
\left( {\text{by}\left( {33}\right) ,\left( {34}\right) }\right)  \leq  \mathop{\sum }\limits_{{i = 1}}^{\alpha }\frac{k\left( {\partial }_{i}\right) }{2}\log m + \mathop{\sum }\limits_{{j = 1}}^{\beta }\left( {\log m + \left( {k\left( { \star  }_{j}\right)  - 2}\right) \log \lambda }\right) 
$$

$$
 = \left( {{k}_{\text{cycle }}/2 + \beta }\right) \log m + \left( {{k}_{\text{star }} - {2\beta }}\right) \log \lambda 
$$

as desired.

正如我们所期望的。

We proceed to prove (33). Let us arrange the vertices of ${\partial }_{i}$ in clockwise order as ${X}_{1},{X}_{2},\ldots ,{X}_{k\left( {\circlearrowleft }_{i}\right) }$ . For $t \geq  3$ ,define ${X}_{\left\lbrack  t\right\rbrack  } = \left\{  {{X}_{1},{X}_{2},\ldots ,{X}_{t}}\right\}$ . Applying the disjointness rule,we get

我们接着证明(33)。我们将${\partial }_{i}$的顶点按顺时针顺序排列为${X}_{1},{X}_{2},\ldots ,{X}_{k\left( {\circlearrowleft }_{i}\right) }$。对于$t \geq  3$，定义${X}_{\left\lbrack  t\right\rbrack  } = \left\{  {{X}_{1},{X}_{2},\ldots ,{X}_{t}}\right\}$。应用不相交规则，我们得到

$$
h\left( {V\left( {\partial }_{i}\right) }\right)  \leq  h\left( {{X}_{1},{X}_{k\left( {\bigtriangleup }_{i}\right) }}\right)  + \mathop{\sum }\limits_{{t = 2}}^{{k\left( {\bigtriangleup }_{i}\right)  - 1}}h\left( {X}_{t}\right) . \tag{35}
$$

Equipped with the above, we can derive

有了上述结果，我们可以推导出

$$
k\left( {\circlearrowright }_{i}\right)  \cdot  \log m\text{(by constraint (II))}
$$

$$
 \geq  \left( {\mathop{\sum }\limits_{{t = 1}}^{{k\left( {\bigtriangleup }_{i}\right)  - 1}}h\left( {{X}_{t},{X}_{t + 1}}\right) }\right)  + h\left( \left\{  {{X}_{k\left( {\bigtriangleup }_{i}\right) },{X}_{1}}\right\}  \right) 
$$

(the next few steps will apply constraint (IV))

（接下来的几步将应用约束条件（IV））

$$
 \geq  h\left( {X}_{\left\lbrack  3\right\rbrack  }\right)  + h\left( {X}_{2}\right)  + \left( {\mathop{\sum }\limits_{{t = 3}}^{{k\left( {\bigtriangleup }_{i}\right)  - 1}}h\left( {{X}_{t},{X}_{t + 1}}\right) }\right)  + h\left( {{X}_{k\left( {\bigtriangleup }_{i}\right) },{X}_{1}}\right) 
$$

...

$$
 \geq  h\left( {X}_{\left\lbrack  4\right\rbrack  }\right)  + h\left( {X}_{2}\right)  + h\left( {X}_{3}\right)  + \left( {\mathop{\sum }\limits_{{t = 4}}^{{k\left( {\bigtriangleup }_{i}\right)  - 1}}h\left( {{X}_{t},{X}_{t + 1}}\right) }\right)  + h\left( {{X}_{k\left( {\bigtriangleup }_{i}\right) },{X}_{1}}\right) 
$$

$$
 \geq  h\left( {X}_{\left\lbrack  k\left( {\bigtriangleup }_{i}\right) \right\rbrack  }\right)  + \left( {\mathop{\sum }\limits_{{t = 2}}^{{k\left( {\bigtriangleup }_{i}\right)  - 1}}h\left( {X}_{t}\right) }\right)  + h\left( {{X}_{k\left( {\bigtriangleup }_{i}\right) },{X}_{1}}\right) 
$$

(the next step applies (35) and $V\left( {\square }_{i}\right)  = {X}_{\left\lbrack  k\left( {\square }_{i}\right) \right\rbrack  }$ )

（下一步应用(35)和$V\left( {\square }_{i}\right)  = {X}_{\left\lbrack  k\left( {\square }_{i}\right) \right\rbrack  }$）

$$
 \geq  2 \cdot  h\left( {V\left( {\square }_{i}\right) }\right) 
$$

as claimed in (33).

正如(33)中所声称的。

It remains to prove (34). Let us label the vertices in ${ \star  }_{j}$ as ${Y}_{1},{Y}_{2},\ldots ,{Y}_{k\left( { \star  }_{j}\right) }$ with ${Y}_{1}$ being the center vertex. For $t \geq  3$ ,define ${Y}_{\left\lbrack  t\right\rbrack  } = \left\{  {{Y}_{1},{Y}_{2},\ldots ,{Y}_{t}}\right\}$ . Then:

还需要证明(34)。我们将${ \star  }_{j}$中的顶点标记为${Y}_{1},{Y}_{2},\ldots ,{Y}_{k\left( { \star  }_{j}\right) }$，其中${Y}_{1}$是中心顶点。对于$t \geq  3$，定义${Y}_{\left\lbrack  t\right\rbrack  } = \left\{  {{Y}_{1},{Y}_{2},\ldots ,{Y}_{t}}\right\}$。然后：

$$
h\left( {V\left( { \star  }_{j}\right) }\right)  = h\left( {{Y}_{1},{Y}_{2}}\right)  + \mathop{\sum }\limits_{{t = 3}}^{{k\left( { \star  }_{j}\right) }}h\left( {Y}_{\left\lbrack  t\right\rbrack  }\right)  - h\left( {Y}_{\left\lbrack  t - 1\right\rbrack  }\right) 
$$

$$
\text{(by constraint (IV))} \leq  h\left( {{Y}_{1},{Y}_{2}}\right)  + \mathop{\sum }\limits_{{t = 3}}^{{k\left( { \star  }_{j}\right) }}\left( {h\left( {{Y}_{1},{Y}_{t}}\right)  - h\left( {Y}_{1}\right) }\right) 
$$

$$
\text{(by constraints (II),(III))} \leq  \log m + \left( {k\left( { \star  }_{j}\right)  - 2}\right) \log \lambda 
$$

as claimed in (34).

正如(34)中所声称的。

### C.2 Constructing an Optimal Set Function

### C.2 构造一个最优集函数

To prove Lemma 5.1,we still need to show that ${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$ is at least the right hand side of (22). For this purpose,it suffices to prove (i) when $\lambda  \leq  \sqrt{m}$ ,there is a feasible set function ${h}^{ * }$ whose ${h}^{ * }\left( {V}_{P}\right)$ equals the right hand side of (30),and (ii) when $\lambda  > \sqrt{m}$ ,there is a feasible set function ${h}^{ * }$ whose ${h}^{ * }\left( {V}_{P}\right)$ equals the right hand side of (31). This subsection will construct these set functions explicitly.

为了证明引理5.1，我们仍需证明${\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right)$至少等于(22)式的右侧。为此，只需证明 (i) 当$\lambda  \leq  \sqrt{m}$时，存在一个可行的集合函数${h}^{ * }$，其${h}^{ * }\left( {V}_{P}\right)$等于(30)式的右侧；(ii) 当$\lambda  > \sqrt{m}$时，存在一个可行的集合函数${h}^{ * }$，其${h}^{ * }\left( {V}_{P}\right)$等于(31)式的右侧。本小节将明确构造这些集合函数。

Case $\lambda  \leq  \sqrt{m}$ . In this scenario,the function ${h}^{ * }$ is easy to design. For each $\mathcal{X} \subseteq  {V}_{P}$ ,set

情形$\lambda  \leq  \sqrt{m}$。在这种情况下，函数${h}^{ * }$很容易设计。对于每个$\mathcal{X} \subseteq  {V}_{P}$，设定

$$
{h}^{ * }\left( \mathcal{X}\right)  = \left\{  \begin{array}{ll} 0 & \text{ if }\mathcal{X} = \varnothing \\  \log m + \left( {\left| \mathcal{X}\right|  - 2}\right) \log \lambda & \text{ otherwise } \end{array}\right.  \tag{36}
$$

Obviously, ${h}^{ * }\left( {V}_{P}\right)  = \log m + \left( {\left| {V}_{P}\right|  - 2}\right) \log \lambda$ ,as needed. It remains to explain why this ${h}^{ * }$ is a feasible solution to the polymatroid LP. We prove as follows.

显然，如所需的那样，有${h}^{ * }\left( {V}_{P}\right)  = \log m + \left( {\left| {V}_{P}\right|  - 2}\right) \log \lambda$。接下来需要解释为什么这个${h}^{ * }$是多拟阵线性规划的一个可行解。我们证明如下。

Constraints (I), (II), and (III) are trivial to verify and omitted. Regarding (IV), first note that the constraint is obviously true if $\mathcal{X}$ or $\mathcal{Y}$ is empty. If neither of them is empty,we can derive:

约束条件 (I)、(II) 和 (III) 很容易验证，在此省略。对于 (IV)，首先注意到，如果$\mathcal{X}$或$\mathcal{Y}$为空，该约束条件显然成立。如果它们都不为空，我们可以推导得出：

$$
{h}^{ * }\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + {h}^{ * }\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  = 2\log m + \left( {\left| {\mathcal{X} \cup  \mathcal{Y}}\right|  + \left| {\mathcal{X} \cap  \mathcal{Y}}\right|  - 4}\right) \log \lambda 
$$

$$
 = 2\log m + \left( {\left| \mathcal{X}\right|  + \left| \mathcal{Y}\right|  - 4}\right) \log \lambda 
$$

$$
 = {h}^{ * }\left( \mathcal{X}\right)  + {h}^{ * }\left( \mathcal{Y}\right)  \tag{37}
$$

as needed. Now,consider constraint (V). If $\mathcal{X} = \varnothing$ ,the constraint holds because ${h}^{ * }\left( \mathcal{Y}\right)  = \log m +$ $\left( {\left| \mathcal{Y}\right|  - 2}\right) \log \lambda  \geq  \log m - \log \lambda  \geq  0$ . If $X \neq  \varnothing$ ,then

如所需的那样。现在，考虑约束条件 (V)。如果$\mathcal{X} = \varnothing$，该约束条件成立，因为${h}^{ * }\left( \mathcal{Y}\right)  = \log m +$ $\left( {\left| \mathcal{Y}\right|  - 2}\right) \log \lambda  \geq  \log m - \log \lambda  \geq  0$。如果$X \neq  \varnothing$，那么

$$
{h}^{ * }\left( \mathcal{X}\right)  = \log m + \left( {\left| \mathcal{X}\right|  - 2}\right) \log \lambda 
$$

$$
 \leq  \log m + \left( {\left| \mathcal{Y}\right|  - 2}\right) \log \lambda 
$$

$$
 = {h}^{ * }\left( \mathcal{Y}\right) \text{.}
$$

Case $\lambda  > \sqrt{m}$ . Let us look at the fractional edge-cover decomposition of $P : \left( {{ \oplus  }_{1},\ldots ,{ \oplus  }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ . As before,for each $i \in  \left\lbrack  \alpha \right\rbrack$ ,define $V\left( {\square }_{i}\right)$ as the set of vertices in the cycle ${\square }_{i}$ ; for each $j \in  \left\lbrack  \beta \right\rbrack$ ,define $V\left( { \star  }_{j}\right)$ as the set of vertices in the star ${ \star  }_{j}$ .

情形$\lambda  > \sqrt{m}$。让我们来看$P : \left( {{ \oplus  }_{1},\ldots ,{ \oplus  }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$的分数边覆盖分解。和之前一样，对于每个$i \in  \left\lbrack  \alpha \right\rbrack$，将$V\left( {\square }_{i}\right)$定义为循环${\square }_{i}$中的顶点集；对于每个$j \in  \left\lbrack  \beta \right\rbrack$，将$V\left( { \star  }_{j}\right)$定义为星型${ \star  }_{j}$中的顶点集。

To design the function ${h}^{ * }$ ,for each $\mathcal{X} \subseteq  {V}_{P}$ ,we choose the value ${h}^{ * }\left( \mathcal{X}\right)$ by the following rules.

为了设计函数${h}^{ * }$，对于每个$\mathcal{X} \subseteq  {V}_{P}$，我们根据以下规则选择值${h}^{ * }\left( \mathcal{X}\right)$。

- (Rule 1): If $\mathcal{X} = \varnothing$ ,then ${h}^{ * }\left( \mathcal{X}\right)  = 0$ .

- （规则1）：如果$\mathcal{X} = \varnothing$，那么${h}^{ * }\left( \mathcal{X}\right)  = 0$。

- (Rule 2): if $\mathcal{X} \subseteq  V\left( {\partial }_{i}\right)$ for some $i \in  \left\lbrack  \alpha \right\rbrack$ but $\mathcal{X} \neq  \varnothing$ ,then ${h}^{ * }\left( \mathcal{X}\right)  = \frac{\left| \mathcal{X}\right| }{2}\log m$ .

- （规则2）：如果对于某个$i \in  \left\lbrack  \alpha \right\rbrack$有$\mathcal{X} \subseteq  V\left( {\partial }_{i}\right)$，但$\mathcal{X} \neq  \varnothing$，那么${h}^{ * }\left( \mathcal{X}\right)  = \frac{\left| \mathcal{X}\right| }{2}\log m$。

- (Rule 3): if $\mathcal{X} \subseteq  V\left( { \star  }_{j}\right)$ for some $j \in  \left\lbrack  \beta \right\rbrack$ but $\mathcal{X} \neq  \varnothing$ ,then ${h}^{ * }\left( \mathcal{X}\right)  = \log m + \left( {\left| \mathcal{X}\right|  - 2}\right) \log \lambda$ .

- （规则3）：若对于某个$j \in  \left\lbrack  \beta \right\rbrack$有$\mathcal{X} \subseteq  V\left( { \star  }_{j}\right)$，但$\mathcal{X} \neq  \varnothing$，则${h}^{ * }\left( \mathcal{X}\right)  = \log m + \left( {\left| \mathcal{X}\right|  - 2}\right) \log \lambda$。

- (Rule 4): Suppose that none of the above rules applies. For each $i \in  \left\lbrack  \alpha \right\rbrack$ ,define ${\mathcal{Y}}_{i} = \mathcal{X} \cap  V\left( { \ominus  }_{i}\right)$ ; similarly,for each $j \in  \left\lbrack  \beta \right\rbrack$ ,define ${\mathcal{Z}}_{j} = \mathcal{X} \cap  V\left( { \star  }_{j}\right)$ . Then:

- （规则4）：假设上述规则均不适用。对于每个$i \in  \left\lbrack  \alpha \right\rbrack$，定义${\mathcal{Y}}_{i} = \mathcal{X} \cap  V\left( { \ominus  }_{i}\right)$；类似地，对于每个$j \in  \left\lbrack  \beta \right\rbrack$，定义${\mathcal{Z}}_{j} = \mathcal{X} \cap  V\left( { \star  }_{j}\right)$。那么：

$$
{h}^{ * }\left( \mathcal{X}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {\mathcal{Y}}_{i}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {\mathcal{Z}}_{j}\right) . \tag{38}
$$

The above equation is well-defined because each ${h}^{ * }\left( {\mathcal{Y}}_{i}\right)$ can be computed using Rules 1 and 2, and each ${h}^{ * }\left( {\mathcal{Z}}_{j}\right)$ can be computed using Rules 1 and 3 .

上述方程是良定义的，因为每个${h}^{ * }\left( {\mathcal{Y}}_{i}\right)$都可以使用规则1和规则2计算得出，每个${h}^{ * }\left( {\mathcal{Z}}_{j}\right)$都可以使用规则1和规则3计算得出。

As a remark,our construction ensures that (38) holds for all $\mathcal{X} \subseteq  {V}_{P}$ .

作为备注，我们的构造确保了（38）式对所有$\mathcal{X} \subseteq  {V}_{P}$都成立。

It is easy to check that ${h}^{ * }\left( {V}_{P}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {V\left( {\circlearrowleft }_{i}\right) }\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {V\left( { \star  }_{j}\right) }\right)$ ,which is $\left( {{k}_{\text{cycle }}/2 + \beta }\right) \log m +$ $\left( {{k}_{\text{star }} - {2\beta }}\right) \log \lambda$ ,as needed. It suffices to verify the feasibility of ${h}^{ * }$ .

很容易验证${h}^{ * }\left( {V}_{P}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {V\left( {\circlearrowleft }_{i}\right) }\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {V\left( { \star  }_{j}\right) }\right)$，即$\left( {{k}_{\text{cycle }}/2 + \beta }\right) \log m +$$\left( {{k}_{\text{star }} - {2\beta }}\right) \log \lambda$，符合要求。只需验证${h}^{ * }$的可行性即可。

Constraint (I) is guaranteed by Rule 1. Next, we will discuss constraints (II) and (III) together. The verification is easy (and hence omitted) if $\{ X,Y\}$ is a cycle edge or a star edge. Now,consider that $\{ X,Y\}$ is neither a cycle edge nor a star edge. By the properties of fractional edge-cover decomposition, one of the following must occur:

约束条件（I）由规则1保证。接下来，我们将一起讨论约束条件（II）和（III）。如果$\{ X,Y\}$是环边或星边，验证很容易（因此省略）。现在，考虑$\{ X,Y\}$既不是环边也不是星边的情况。根据分数边覆盖分解的性质，必然会出现以下情况之一：

- (C-1) $X$ and $Y$ are in two different cycles;

- （C - 1）$X$和$Y$位于两个不同的环中；

- $\left( {\mathrm{C} - 2}\right) X$ and $Y$ are in two different stars;

- $\left( {\mathrm{C} - 2}\right) X$和$Y$位于两个不同的星中；

- (C-3) one of $X$ and $Y$ is in a cycle and the other is in a star.

- （C - 3）$X$和$Y$中有一个在环中，另一个在星中。

In all the above scenarios,it holds that ${h}^{ * }\left( {\{ X,Y\} }\right)  = {h}^{ * }\left( {\{ X\} }\right)  + {h}^{ * }\left( {\{ Y\} }\right)$ . Thus,to confirm (II),it suffices to show ${h}^{ * }\left( {\{ X\} }\right)  + {h}^{ * }\left( {\{ Y\} }\right)  \leq  \log m$ ,and to confirm (III),it suffices to show $h\left( {\{ Y\} }\right)  \leq  \log \lambda$ . It is rudimentary to verify both of these inequalities using Rules 2 and 3 and the fact $\log m < 2\log \lambda$ .

在上述所有情形中，都有${h}^{ * }\left( {\{ X,Y\} }\right)  = {h}^{ * }\left( {\{ X\} }\right)  + {h}^{ * }\left( {\{ Y\} }\right)$成立。因此，要确认（II），只需证明${h}^{ * }\left( {\{ X\} }\right)  + {h}^{ * }\left( {\{ Y\} }\right)  \leq  \log m$；要确认（III），只需证明$h\left( {\{ Y\} }\right)  \leq  \log \lambda$。使用规则2和规则3以及事实$\log m < 2\log \lambda$来验证这两个不等式是很基础的。

Constraint (IV) is trivially true if either $\mathcal{X}$ or $\mathcal{Y}$ is empty. Now,assume that neither is empty. If $\mathcal{X},\mathcal{Y} \subseteq  \mathcal{V}\left( { \ominus  }_{i}\right)$ for some $i \in  \left\lbrack  \alpha \right\rbrack$ ,we have:

如果$\mathcal{X}$或$\mathcal{Y}$为空，约束条件（IV）显然成立。现在，假设两者都不为空。如果对于某个$i \in  \left\lbrack  \alpha \right\rbrack$有$\mathcal{X},\mathcal{Y} \subseteq  \mathcal{V}\left( { \ominus  }_{i}\right)$，我们有：

$$
{h}^{ * }\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + {h}^{ * }\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  = \frac{\left| {\mathcal{X} \cap  \mathcal{Y}}\right|  + \left| {\mathcal{X} \cup  \mathcal{Y}}\right| }{2}\log m
$$

$$
 = \frac{\left| \mathcal{X}\right|  + \left| \mathcal{Y}\right| }{2}\log m
$$

$$
 = {h}^{ * }\left( \mathcal{X}\right)  + {h}^{ * }\left( \mathcal{Y}\right) \text{.}
$$

If $\mathcal{X},\mathcal{Y} \subseteq  \mathcal{V}\left( { \star  }_{j}\right)$ for some $j \in  \left\lbrack  \beta \right\rbrack$ ,the reader can verify (IV) with the same derivation in (37).

如果对于某个$j \in  \left\lbrack  \beta \right\rbrack$有$\mathcal{X},\mathcal{Y} \subseteq  \mathcal{V}\left( { \star  }_{j}\right)$，读者可以用（37）中的相同推导来验证（IV）。

Next,we consider arbitrary $\mathcal{X},\mathcal{Y} \subseteq  \mathcal{V}$ . Define for each $i \in  \left\lbrack  \alpha \right\rbrack$ and $j \in  \left\lbrack  \beta \right\rbrack$ :

接下来，我们考虑任意的$\mathcal{X},\mathcal{Y} \subseteq  \mathcal{V}$。对于每个$i \in  \left\lbrack  \alpha \right\rbrack$和$j \in  \left\lbrack  \beta \right\rbrack$，定义：

$$
{\mathcal{X}}_{i}^{\text{cycle }} = \mathcal{X} \cap  \mathcal{V}\left( { \circ  }_{i}\right)  \tag{39}
$$

$$
{\mathcal{X}}_{j}^{\text{star }} = \mathcal{X} \cap  \mathcal{V}\left( { \star  }_{j}\right)  \tag{40}
$$

$$
{\mathcal{Y}}_{i}^{\text{cycle }} = \mathcal{Y} \cap  \mathcal{V}\left( { \circ  }_{i}\right)  \tag{41}
$$

$$
{\mathcal{Y}}_{j}^{\text{star }} = \mathcal{Y} \cap  \mathcal{V}\left( { \star  }_{j}\right) . \tag{42}
$$

We can derive:

我们可以推导出：

$$
{h}^{ * }\left( {\mathcal{X} \cup  \mathcal{Y}}\right) 
$$

$$
\text{(by Rule 4)} = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  \cap  \mathcal{V}\left( {\partial }_{i}\right) }\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  \cap  \mathcal{V}\left( { \star  }_{j}\right) }\right) 
$$

$$
 = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {{\mathcal{X}}_{i}^{\text{cycle }} \cup  {\mathcal{Y}}_{i}^{\text{cycle }}}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {{\mathcal{X}}_{j}^{\text{star }} \cup  {\mathcal{Y}}_{j}^{\text{star }}}\right) 
$$

and similarly:

类似地：

$$
{h}^{ * }\left( {\mathcal{X} \cap  \mathcal{Y}}\right) 
$$

$$
 = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  \cap  \mathcal{V}\left( { \circ  }_{i}\right) }\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {\left( {\mathcal{X} \cap  \mathcal{Y}}\right)  \cap  \mathcal{V}\left( { \star  }_{j}\right) }\right) 
$$

$$
 = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {{\mathcal{X}}_{i}^{\text{cycle }} \cap  {\mathcal{Y}}_{i}^{\text{cycle }}}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {{\mathcal{X}}_{j}^{\text{star }} \cap  {\mathcal{Y}}_{j}^{\text{star }}}\right) .
$$

Recall that (IV) has been validated in the scenario where (i) $\mathcal{X}$ and $\mathcal{Y}$ are contained in the same cycle,or (ii) they are contained in the same star. Thus,it holds for each $i \in  \left\lbrack  \alpha \right\rbrack$ and $j \in  \left\lbrack  \beta \right\rbrack$ that

回想一下，在以下两种情况下（IV）已被验证成立：（i）$\mathcal{X}$和$\mathcal{Y}$包含在同一个环中，或者（ii）它们包含在同一个星型结构中。因此，对于每个$i \in  \left\lbrack  \alpha \right\rbrack$和$j \in  \left\lbrack  \beta \right\rbrack$，有

$$
{h}^{ * }\left( {{\mathcal{X}}_{i}^{\text{cycle }} \cup  {\mathcal{Y}}_{i}^{\text{cycle }}}\right)  + {h}^{ * }\left( {{\mathcal{X}}_{i}^{\text{cycle }} \cap  {\mathcal{Y}}_{i}^{\text{cycle }}}\right)  \leq  {h}^{ * }\left( {\mathcal{X}}_{i}^{\text{cycle }}\right)  + {h}^{ * }\left( {\mathcal{Y}}_{i}^{\text{cycle }}\right)  \tag{43}
$$

$$
{h}^{ * }\left( {{\mathcal{X}}_{j}^{\text{star }} \cup  {\mathcal{Y}}_{j}^{\text{star }}}\right)  + {h}^{ * }\left( {{\mathcal{X}}_{j}^{\text{star }} \cap  {\mathcal{Y}}_{j}^{\text{star }}}\right)  \leq  {h}^{ * }\left( {\mathcal{X}}_{j}^{\text{star }}\right)  + {h}^{ * }\left( {\mathcal{Y}}_{j}^{\text{star }}\right) . \tag{44}
$$

Equipped with these facts, we get:

有了这些事实，我们得到：

$$
{h}^{ * }\left( {\mathcal{X} \cup  \mathcal{Y}}\right)  + {h}^{ * }\left( {\mathcal{X} \cap  \mathcal{Y}}\right) 
$$

$$
 = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {{\mathcal{X}}_{i}^{\text{cycle }} \cup  {\mathcal{Y}}_{i}^{\text{cycle }}}\right)  + {h}^{ * }\left( {{\mathcal{X}}_{i}^{\text{cycle }} \cap  {\mathcal{Y}}_{i}^{\text{cycle }}}\right)  + 
$$

$$
\mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {{\mathcal{X}}_{j}^{\text{star }} \cup  {\mathcal{Y}}_{j}^{\text{star }}}\right)  + {h}^{ * }\left( {{\mathcal{X}}_{j}^{\text{star }} \cap  {\mathcal{Y}}_{j}^{\text{star }}}\right) 
$$

$$
\text{(by (43) and (44))} \leq  \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {\mathcal{X}}_{i}^{\text{cycle }}\right)  + {h}^{ * }\left( {\mathcal{Y}}_{i}^{\text{cycle }}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {\mathcal{X}}_{j}^{\text{star }}\right)  + {h}^{ * }\left( {\mathcal{Y}}_{j}^{\text{star }}\right) 
$$

$$
\text{(by Rule 4)} = {h}^{ * }\left( \mathcal{X}\right)  + {h}^{ * }\left( \mathcal{Y}\right) \text{.}
$$

This verifies the correctness of constraint (IV).

这验证了约束条件（IV）的正确性。

Finally,let us look at (V). This constraint is trivially met if $\mathcal{X} = \varnothing$ . Next,we assume that $\mathcal{X}$ is not empty. If $\mathcal{Y}$ is a subset of $\mathcal{V}\left( {\square }_{i}\right)$ for some $i \in  \left\lbrack  \alpha \right\rbrack$ ,then,by Rule $2,{h}^{ * }\left( \mathcal{X}\right)  =$ $\frac{\left| \mathcal{X}\right| }{2}\log m \leq  \frac{\left| \mathcal{Y}\right| }{2}\log m = {h}^{ * }\left( \mathcal{Y}\right)$ . If $\mathcal{Y}$ is a subset of $\mathcal{V}\left( { \star  }_{j}\right)$ for some $j \in  \left\lbrack  \beta \right\rbrack$ ,then,by Rule 3, ${h}^{ * }\left( \mathcal{X}\right)  = \log m + \left( {\left| \mathcal{X}\right|  - 2}\right) \log \lambda  \leq  \log m + \left( {\left| \mathcal{Y}\right|  - 2}\right) \log \lambda  = {h}^{ * }\left( \mathcal{Y}\right) .$

最后，让我们来看（V）。如果$\mathcal{X} = \varnothing$，这个约束条件显然满足。接下来，我们假设$\mathcal{X}$不为空。如果对于某个$i \in  \left\lbrack  \alpha \right\rbrack$，$\mathcal{Y}$是$\mathcal{V}\left( {\square }_{i}\right)$的子集，那么根据规则$2,{h}^{ * }\left( \mathcal{X}\right)  =$有$\frac{\left| \mathcal{X}\right| }{2}\log m \leq  \frac{\left| \mathcal{Y}\right| }{2}\log m = {h}^{ * }\left( \mathcal{Y}\right)$。如果对于某个$j \in  \left\lbrack  \beta \right\rbrack$，$\mathcal{Y}$是$\mathcal{V}\left( { \star  }_{j}\right)$的子集，那么根据规则3，有${h}^{ * }\left( \mathcal{X}\right)  = \log m + \left( {\left| \mathcal{X}\right|  - 2}\right) \log \lambda  \leq  \log m + \left( {\left| \mathcal{Y}\right|  - 2}\right) \log \lambda  = {h}^{ * }\left( \mathcal{Y}\right) .$

It remains to consider the situation where $\mathcal{Y}$ can be any subset of $\mathcal{V}$ . Using the definitions in (39)-(42), we have:

还需要考虑$\mathcal{Y}$可以是$\mathcal{V}$的任意子集的情况。使用（39） - （42）中的定义，我们有：

$$
{h}^{ * }\left( \mathcal{X}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {\mathcal{X}}_{i}^{\text{cycle }}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {\mathcal{X}}_{j}^{\text{star }}\right) \text{ (by Rule 4) }
$$

$$
 \leq  \mathop{\sum }\limits_{{i = 1}}^{\alpha }{h}^{ * }\left( {\mathcal{Y}}_{i}^{\text{cycle }}\right)  + \mathop{\sum }\limits_{{j = 1}}^{\beta }{h}^{ * }\left( {\mathcal{Y}}_{j}^{\text{star }}\right) 
$$

(we have verified (V) in the scenario where $\mathcal{Y}$ is contained in a cycle or a star)

（我们已经在$\mathcal{Y}$包含在一个环或一个星型结构的场景中验证了条件 (V)）

(by Rule 4) $= {h}^{ * }\left( \mathcal{Y}\right)$

（根据规则 4）$= {h}^{ * }\left( \mathcal{Y}\right)$

as needed to verify constraint (V).

以验证约束条件 (V) 所需。

## D Equation (22) as an Output Size Bound and Its Tightness

## D 方程 (22) 作为输出规模上界及其紧性

We will start by proving that (22) is an asymptotic upper bound on $\left| {\operatorname{occ}\left( {G,P}\right) }\right|$ for any graph $G$ that has $m$ edges and a maximum vertex-degree at most $\lambda$ . First,we will prove that $\left| {\operatorname{occ}\left( {G,P}\right) }\right|  =$ $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$ (recall that $k$ is the number of vertices in $P$ ). Identify an arbitrary spanning tree $\mathcal{T}$ of the pattern $P$ . It is rudimentary to verify that $\mathcal{T}$ has $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$ occurrences in $G,{}^{5}$ implying that the number of occurrences of $P$ is also $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$ . Next,we will demonstrate that $\left| {\operatorname{occ}\left( {G,P}\right) }\right|  = O\left( {{m}^{\frac{{k}_{\text{cycle }}}{2}} + \beta  \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}}\right)$ . For each $i \in  \left\lbrack  \alpha \right\rbrack$ ,let $k\left( {\partial }_{i}\right)$ be the number of vertices in ${\partial }_{i}$ ; for each $j \in  \left\lbrack  \beta \right\rbrack$ ,let $k\left( { \star  }_{j}\right)$ be the number of vertices in ${ \star  }_{j}$ . The fractional edge cover number of ${ \circ  }_{i}$ $\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ is ${\rho }^{ * }\left( {\circlearrowleft }_{i}\right)  = k\left( {\circlearrowleft }_{i}\right) /2$ ; this means $\mathop{\sum }\limits_{{i = 1}}^{k}{\rho }^{ * }\left( {\circlearrowleft }_{i}\right)  = k\left( {\circlearrowleft }_{i}\right) /2 = {k}_{\text{cycle }}/2$ . For each $i \in  \left\lbrack  \alpha \right\rbrack$ ,the pattern ${\partial }_{i}$ can have $O\left( {m}^{{\rho }^{ * }\left( {\square }_{i}\right) }\right)$ occurrences in $G$ . For each $j \in  \left\lbrack  \beta \right\rbrack$ ,by our earlier analysis,the pattern ${ \star  }_{j}$ can have $O\left( {m}^{k\left( { \star  }_{j}\right)  - 2}\right)$ occurrences in $G$ . Thus,the number of occurrences of $P$ must be asymptotically bounded by

我们首先证明，对于任何具有 $m$ 条边且最大顶点度至多为 $\lambda$ 的图 $G$，方程 (22) 是 $\left| {\operatorname{occ}\left( {G,P}\right) }\right|$ 的渐近上界。首先，我们将证明 $\left| {\operatorname{occ}\left( {G,P}\right) }\right|  =$ $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$（回顾一下，$k$ 是 $P$ 中的顶点数）。确定模式 $P$ 的任意一棵生成树 $\mathcal{T}$。很容易验证 $\mathcal{T}$ 在 $G,{}^{5}$ 中有 $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$ 个出现，这意味着 $P$ 的出现次数也是 $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$。接下来，我们将证明 $\left| {\operatorname{occ}\left( {G,P}\right) }\right|  = O\left( {{m}^{\frac{{k}_{\text{cycle }}}{2}} + \beta  \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}}\right)$。对于每个 $i \in  \left\lbrack  \alpha \right\rbrack$，设 $k\left( {\partial }_{i}\right)$ 是 ${\partial }_{i}$ 中的顶点数；对于每个 $j \in  \left\lbrack  \beta \right\rbrack$，设 $k\left( { \star  }_{j}\right)$ 是 ${ \star  }_{j}$ 中的顶点数。${ \circ  }_{i}$ $\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ 的分数边覆盖数是 ${\rho }^{ * }\left( {\circlearrowleft }_{i}\right)  = k\left( {\circlearrowleft }_{i}\right) /2$；这意味着 $\mathop{\sum }\limits_{{i = 1}}^{k}{\rho }^{ * }\left( {\circlearrowleft }_{i}\right)  = k\left( {\circlearrowleft }_{i}\right) /2 = {k}_{\text{cycle }}/2$。对于每个 $i \in  \left\lbrack  \alpha \right\rbrack$，模式 ${\partial }_{i}$ 在 $G$ 中可以有 $O\left( {m}^{{\rho }^{ * }\left( {\square }_{i}\right) }\right)$ 个出现。对于每个 $j \in  \left\lbrack  \beta \right\rbrack$，根据我们之前的分析，模式 ${ \star  }_{j}$ 在 $G$ 中可以有 $O\left( {m}^{k\left( { \star  }_{j}\right)  - 2}\right)$ 个出现。因此，$P$ 的出现次数在渐近意义上一定有界于

$$
\mathop{\prod }\limits_{{i = 1}}^{\alpha }{m}^{{\rho }^{ * }\left( {\bigtriangleup }_{i}\right) } \cdot  \mathop{\prod }\limits_{{j = 1}}^{\beta }m \cdot  {\lambda }^{k\left( { \star  }_{j}\right)  - 2} = {m}^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}.
$$

The rest of this section will concentrate on the tightness of (22) as an upper bound of $\left| {\operatorname{occ}\left( {G,P}\right) }\right|$ . Our objective is to prove:

本节的其余部分将集中讨论方程 (22) 作为 $\left| {\operatorname{occ}\left( {G,P}\right) }\right|$ 的上界的紧性。我们的目标是证明：

Theorem D.1. Fix an arbitrary integer $k \geq  2$ . For any values of $m$ and $\lambda$ satisfying $m \geq$ $\max \left\{  {{16}{k}^{2},{64}}\right\}$ and $k \leq  \lambda  \leq  m/\left( {4k}\right)$ ,there is an undirected simple graph ${G}^{ * }$ satisfying all the conditions below:

定理 D.1。固定任意整数 $k \geq  2$。对于满足 $m \geq$ $\max \left\{  {{16}{k}^{2},{64}}\right\}$ 和 $k \leq  \lambda  \leq  m/\left( {4k}\right)$ 的任意 $m$ 和 $\lambda$ 的值，存在一个无向简单图 ${G}^{ * }$ 满足以下所有条件：

- ${G}^{ * }$ has at most $m$ edges and and a maximum vertex degree at most $\lambda$ ;

- ${G}^{ * }$最多有$m$条边，且最大顶点度至多为$\lambda$；

- For any undirected simple pattern graph $P = \left( {{V}_{P},{E}_{P}}\right)$ that has $k$ vertices,the number of occurrences of $P$ in ${G}^{ * }$ is at least $\frac{1}{{\left( 4k\right) }^{k}}$ -polymat ${}_{\text{undir }}\left( {m,\lambda ,P}\right)$ ,where polymat ${}_{\text{undir }}\left( {m,\lambda ,P}\right)$ is given in (22).

- 对于任何具有$k$个顶点的无向简单模式图$P = \left( {{V}_{P},{E}_{P}}\right)$，$P$在${G}^{ * }$中出现的次数至少为$\frac{1}{{\left( 4k\right) }^{k}}$ - 多胞体${}_{\text{undir }}\left( {m,\lambda ,P}\right)$，其中多胞体${}_{\text{undir }}\left( {m,\lambda ,P}\right)$在(22)中给出。

It is worth noting that we aim to construct a single ${G}^{ * }$ that is a "bad" input for all possible patterns (with $k$ vertices) simultaneously. Our proof will deal with the case $k \leq  \lambda  < \sqrt{m}$ and the case $\sqrt{m} \leq  \lambda  \leq  m/\left( {4k}\right)$ separately.

值得注意的是，我们的目标是构造一个单一的${G}^{ * }$，它同时是所有可能模式（具有$k$个顶点）的“不良”输入。我们的证明将分别处理$k \leq  \lambda  < \sqrt{m}$的情况和$\sqrt{m} \leq  \lambda  \leq  m/\left( {4k}\right)$的情况。

---

<!-- Footnote -->

${}^{5}$ There are $O\left( m\right)$ choices to map an arbitrary edge $\mathcal{T}$ to an edge in $G$ ,and then $\lambda$ choices to map each of the $k - 2$ remaining vertices of $\mathcal{T}$ to a vertex in $G$ .

${}^{5}$ 将任意一条边$\mathcal{T}$映射到$G$中的一条边有$O\left( m\right)$种选择，然后将$\mathcal{T}$中剩余的$k - 2$个顶点中的每个顶点映射到$G$中的一个顶点有$\lambda$种选择。

<!-- Footnote -->

---

Remark. The main challenge in the proof is to establish the factor $\frac{1}{{\left( 4k\right) }^{k}}$ . There exist lower bound arguments $\left\lbrack  {{19},{36}}\right\rbrack$ that can be used in our context to build a hard instance for each individual pattern $P$ . A naive way to form a single hard input ${G}^{ * }$ is to combine the hard instances of all possible patterns having $k$ vertices. But this would introduce a gigantic factor roughly $1/{2}^{\Omega \left( {k}^{2}\right) }$ .

注：证明中的主要挑战是确定因子$\frac{1}{{\left( 4k\right) }^{k}}$。存在一些下界论证$\left\lbrack  {{19},{36}}\right\rbrack$，可在我们的情境中用于为每个单独的模式$P$构建一个困难实例。形成单个困难输入${G}^{ * }$的一种简单方法是将所有具有$k$个顶点的可能模式的困难实例组合起来。但这将引入一个大约为$1/{2}^{\Omega \left( {k}^{2}\right) }$的巨大因子。

D. 1 Case $k \leq  \lambda  < \sqrt{m}$

D. 1 情况$k \leq  \lambda  < \sqrt{m}$

In this scenario, ${G}^{ * }$ only needs to be the graph that consists of $\left\lfloor  {m/\left( \begin{array}{l} \lambda \\  2 \end{array}\right) }\right\rfloor$ independent $\lambda$ -cliques. ${}^{6}$ An occurrence of $P$ can be formed by mapping ${V}_{P}$ to $k$ arbitrary distinct vertices in one $\lambda$ -clique. The number of occurrences of $P$ in ${G}^{ * }$ is at least

在这种情况下，${G}^{ * }$只需是由$\left\lfloor  {m/\left( \begin{array}{l} \lambda \\  2 \end{array}\right) }\right\rfloor$个独立的$\lambda$ - 团组成的图。${}^{6}$ $P$的一次出现可以通过将${V}_{P}$映射到一个$\lambda$ - 团中的$k$个任意不同的顶点来形成。$P$在${G}^{ * }$中出现的次数至少为

$$
\left\lfloor  \frac{m}{\left( \begin{array}{l} \lambda \\  2 \end{array}\right) }\right\rfloor   \cdot  \left( \begin{array}{l} \lambda \\  k \end{array}\right) 
$$

$$
\left( {\text{ applying }\left( \begin{array}{l} \lambda \\  k \end{array}\right)  \geq  {\left( \lambda /k\right) }^{k}}\right)  \geq  \left( {\frac{2m}{{\lambda }^{2} - \lambda } - 1}\right)  \cdot  {\left( \lambda /k\right) }^{k}
$$

$$
\left( {\text{ as }m > {\lambda }^{2} - \lambda }\right)  > \frac{m}{{\lambda }^{2} - \lambda } \cdot  {\left( \lambda /k\right) }^{k}
$$

$$
\text{(as}k \geq  2\text{)} > \frac{1}{{k}^{k}} \cdot  m \cdot  {\lambda }^{k - 2} = \frac{1}{{k}^{k}} \cdot  {\text{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right) \text{.}
$$

D.2Case $\sqrt{m} \leq  \lambda  \leq  m/\left( {4k}\right)$

D.2 情况$\sqrt{m} \leq  \lambda  \leq  m/\left( {4k}\right)$

We construct ${G}^{ * }$ as follows. First,create three disjoint sets of vertices: ${V}_{A},{V}_{B}$ ,and ${V}_{C}$ ,whose sizes are $\lceil \lambda /4\rceil ,\lceil m/\left( {4\lambda }\right) \rceil$ ,and $\lceil \sqrt{m}/4\rceil$ . Because $\lambda  \leq  \frac{m}{4k}$ and $m \geq  {16}{k}^{2}$ ,each of ${V}_{A},{V}_{B}$ ,and ${V}_{C}$ has a size at least $k$ . Then,decide the edges of ${G}^{ * }$ in the way below:

我们按如下方式构造${G}^{ * }$。首先，创建三个不相交的顶点集：${V}_{A},{V}_{B}$和${V}_{C}$，它们的大小分别为$\lceil \lambda /4\rceil ,\lceil m/\left( {4\lambda }\right) \rceil$和$\lceil \sqrt{m}/4\rceil$。因为$\lambda  \leq  \frac{m}{4k}$和$m \geq  {16}{k}^{2}$，所以${V}_{A},{V}_{B}$和${V}_{C}$的大小至少为$k$。然后，按以下方式确定${G}^{ * }$的边：

- Add an edge between each pair of vertices in ${V}_{B} \cup  {V}_{C}$ (thereby producing a clique of $\lceil m/\left( {4\lambda }\right) \rceil  +$ $\left\lceil  {\sqrt{m}/4}\right\rceil$ vertices).

- 在${V}_{B} \cup  {V}_{C}$中的每对顶点之间添加一条边（从而产生一个由$\lceil m/\left( {4\lambda }\right) \rceil  +$ $\left\lceil  {\sqrt{m}/4}\right\rceil$个顶点组成的团）。

- Add an edge $\{ u,v\}$ for each vertex pair $\left( {u,v}\right)  \in  {V}_{A} \times  {V}_{B}$ .

- 为每对顶点$\left( {u,v}\right)  \in  {V}_{A} \times  {V}_{B}$添加一条边$\{ u,v\}$。

Next,we show that ${G}^{ * }$ has at most $m$ edges through a careful calculation. First,the number of edges between ${V}_{A}$ and ${V}_{B}$ is

接下来，我们通过仔细计算证明 ${G}^{ * }$ 最多有 $m$ 条边。首先，${V}_{A}$ 和 ${V}_{B}$ 之间的边数为

$$
\left\lbrack  {\lambda /4}\right\rbrack   \cdot  \left\lbrack  {m/\left( {4\lambda }\right) }\right\rbrack  
$$

$$
 \leq  \left( {\lambda /4 + 1}\right)  \cdot  \left( {m/\left( {4\lambda }\right)  + 1}\right) 
$$

$$
 = m/{16} + \lambda /4 + m/\left( {4\lambda }\right)  + 1
$$

$$
\left( {\text{ as }\sqrt{m} \leq  \lambda  \leq  m}\right)  \leq  {5m}/{16} + \sqrt{m}/4 + 1
$$

$$
\text{(as}{16} \leq  m\text{)} \leq  {7m}/{16}\text{.}
$$

The number of edges between ${V}_{B}$ and ${V}_{C}$ is:

${V}_{B}$ 和 ${V}_{C}$ 之间的边数为：

$$
\lceil m/\left( {4\lambda }\right) \rceil  \cdot  \lceil \sqrt{m}/4\rceil 
$$

$$
 \leq  \left( {m/\left( {4\lambda }\right)  + 1}\right)  \cdot  \left( {\sqrt{m}/4 + 1}\right) 
$$

$$
 = {m}^{1.5}/\left( {16\lambda }\right)  + m/\left( {4\lambda }\right)  + \sqrt{m}/4 + 1
$$

$$
\left( {\text{as}\sqrt{m} \leq  \lambda }\right)  \leq  m/{16} + \sqrt{m}/2 + 1
$$

$$
\text{(as}{16} \leq  m\text{)} \leq  m/4\text{.}
$$

---

<!-- Footnote -->

${}^{6}$ When $\lambda  < \sqrt{m},\left( \begin{array}{l} \lambda \\  2 \end{array}\right)  = \frac{{\lambda }^{2} - \lambda }{2} < m$ . Hence, ${G}^{ * }$ contains at least one clique.

${}^{6}$ 当 $\lambda  < \sqrt{m},\left( \begin{array}{l} \lambda \\  2 \end{array}\right)  = \frac{{\lambda }^{2} - \lambda }{2} < m$ 时。因此，${G}^{ * }$ 至少包含一个团（clique）。

<!-- Footnote -->

---

The number of edges between vertices in ${V}_{B}$ is:

${V}_{B}$ 中顶点之间的边数为：

$$
\lceil m/\left( {4\lambda }\right) \rceil  \cdot  \left( {\lceil m/\left( {4\lambda }\right) \rceil  - 1}\right) /2
$$

$$
 \leq  \left( {m/\left( {4\lambda }\right)  + 1}\right)  \cdot  \left( {m/\left( {4\lambda }\right) }\right) /2
$$

$$
 = {m}^{2}/\left( {{32}{\lambda }^{2}}\right)  + m/\left( {8\lambda }\right) 
$$

$$
\left( {\text{ as }\sqrt{m} \leq  \lambda }\right)  \leq  m/{32} + \sqrt{m}/8 \leq  {5m}/{32}
$$

The number of edges between vertices in ${V}_{C}$ is:

${V}_{C}$ 中顶点之间的边数为：

$$
\lceil \sqrt{m}/4\rceil  \cdot  \left( {\lceil \sqrt{m}/4 - 1}\right) /2
$$

$$
 \leq  \left( {\sqrt{m}/4 + 1}\right)  \cdot  \left( {\sqrt{m}/4}\right) /2
$$

$$
 = m/{32} + \sqrt{m}/8 \leq  {5m}/{32}\text{.}
$$

Thus,the total number of edges in ${G}^{ * }$ is at most ${7m}/{16} + m/4 + {5m}/{32} + {5m}/{32} = m$ .

因此，${G}^{ * }$ 中的边总数最多为 ${7m}/{16} + m/4 + {5m}/{32} + {5m}/{32} = m$。

The maximum vertex degree in ${G}^{ * }$ is decided by the vertices in ${V}_{B}$ and equals

${G}^{ * }$ 中的最大顶点度由 ${V}_{B}$ 中的顶点决定，且等于

$$
\lceil \lambda /4\rceil  + \lceil m/\left( {4\lambda }\right) \rceil  - 1 + \lceil \sqrt{m}/4\rceil 
$$

$$
 \leq  \lambda /4 + m/\left( {4\lambda }\right)  + \sqrt{m}/4 + 2
$$

$$
\text{(as}\lambda  \geq  \sqrt{m}\text{)} \leq  {3\lambda }/4 + 2 \leq  \lambda \text{.}
$$

Therefore, ${G}^{ * }$ satisfies the requirement in the first bullet of Theorem D.1.

因此，${G}^{ * }$ 满足定理 D.1 第一个要点中的要求。

The rest of the proof will focus on the theorem’s second bullet. Let $P$ be an arbitrary pattern with $k$ vertices,and take a fractional edge-cover decomposition of $P : \left( {{ \ominus  }_{1},\ldots ,{ \ominus  }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ . Define ${k}_{\text{star }}$ and ${k}_{\text{cycle }}$ in the same way as in (21) and (20). We claim:

证明的其余部分将集中在定理的第二个要点上。设 $P$ 是一个具有 $k$ 个顶点的任意模式，并对 $P : \left( {{ \ominus  }_{1},\ldots ,{ \ominus  }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ 进行分数边覆盖分解。按照 (21) 和 (20) 中的相同方式定义 ${k}_{\text{star }}$ 和 ${k}_{\text{cycle }}$。我们断言：

Lemma D.2. There is an integer $s \in  \left\lbrack  {0,\beta }\right\rbrack$ under which we can divide the vertices of $P$ into disjoint sets ${U}_{A},{U}_{B}$ and ${U}_{C}$ such that

引理 D.2。存在一个整数 $s \in  \left\lbrack  {0,\beta }\right\rbrack$，在该整数下，我们可以将 $P$ 的顶点划分为不相交的集合 ${U}_{A},{U}_{B}$ 和 ${U}_{C}$，使得

- $\left| {U}_{A}\right|  = {k}_{\text{star }} - \beta  - s,\left| {U}_{B}\right|  = \beta  - s,\left| {U}_{C}\right|  = {k}_{\text{cycle }} + {2s}$ ;

- $P$ has no edges between ${U}_{A}$ and ${U}_{C}$ ;

- $P$ 在 ${U}_{A}$ 和 ${U}_{C}$ 之间没有边；

- $P$ has no edges between any two vertices in ${U}_{A}$ .

- $P$ 在 ${U}_{A}$ 中的任意两个顶点之间没有边。

The proof of the lemma is non-trivial and deferred to the end of this section. An occurrence of $P$ in ${G}^{ * }$ can be formed through the three steps below:

该引理的证明并非易事，将推迟到本节末尾进行。$P$ 在 ${G}^{ * }$ 中的一次出现可以通过以下三个步骤形成：

1. Map ${U}_{A}$ to $\left| {U}_{A}\right|$ distinct vertices in ${V}_{A}$ .

1. 将 ${U}_{A}$ 映射到 ${V}_{A}$ 中的 $\left| {U}_{A}\right|$ 个不同顶点。

2. Map ${U}_{B}$ to $\left| {U}_{B}\right|$ distinct vertices in ${V}_{B}$ .

2. 将 ${U}_{B}$ 映射到 ${V}_{B}$ 中的 $\left| {U}_{B}\right|$ 个不同顶点。

3. Map ${U}_{C}$ to $\left| {U}_{C}\right|$ distinct vertices in ${V}_{C}$ .

3. 将 ${U}_{C}$ 映射到 ${V}_{C}$ 中的 $\left| {U}_{C}\right|$ 个不同顶点。

Each step is carried out independently from the other steps. Hence,the number of occurrences of $P$ in ${G}^{ * }$ is at least

每一步都独立于其他步骤进行。因此，$P$在${G}^{ * }$中出现的次数至少为

$$
\left( \begin{matrix} \lambda /4 \\  {k}_{\text{star }} - \beta  - s \end{matrix}\right)  \cdot  \left( \begin{matrix} \frac{m}{4\lambda } \\  \beta  - s \end{matrix}\right)  \cdot  \left( \begin{matrix} \sqrt{m}/4 \\  {k}_{\text{cycle }} + {2s} \end{matrix}\right) 
$$

$$
 \geq  {\left( \frac{\lambda /4}{{k}_{\text{star }} - \beta  - s}\right) }^{{k}_{\text{star }} - \beta  - s} \cdot  {\left( \frac{\frac{m}{4\lambda }}{\beta  - s}\right) }^{\beta  - s} \cdot  {\left( \frac{\sqrt{m}/4}{{k}_{\text{cycle }} + {2s}}\right) }^{{k}_{\text{cycle }} + {2s}}
$$

$$
 \geq  \frac{{\lambda }^{{k}_{\text{star }} - \beta  - s}}{{4}^{{k}_{\text{star }} - \beta  - s} \cdot  {k}^{{k}_{\text{star }} - \beta  - s}} \cdot  \frac{{m}^{\beta  - s}}{{4}^{\beta  - s} \cdot  {\lambda }^{\beta  - s} \cdot  {k}^{\beta  - s}} \cdot  \frac{{m}^{{k}_{\text{cycle }}/2 + s}}{{4}^{{k}_{\text{cycle }} + {2s}} \cdot  {k}^{{k}_{\text{cycle }} + {2s}}}
$$

$$
 = \frac{1}{{\left( 4k\right) }^{k}} \cdot  {m}^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }} = \frac{1}{{\left( 4k\right) }^{k}} \cdot  {\operatorname{polymat}}_{\text{undir }}\left( {m,\lambda ,P}\right) .
$$

Proof of Lemma D.2. With each vertex $X$ in the pattern graph $P = \left( {{V}_{P},{E}_{P}}\right)$ ,we associate a variable $\nu \left( X\right)  \geq  0$ (equivalently, $\nu$ is a function from ${V}_{P}$ to ${\mathbb{R}}_{ \geq  0}$ ). Consider the following LP defined on these variables.

引理D.2的证明。对于模式图$P = \left( {{V}_{P},{E}_{P}}\right)$中的每个顶点$X$，我们关联一个变量$\nu \left( X\right)  \geq  0$（等价地，$\nu$是一个从${V}_{P}$到${\mathbb{R}}_{ \geq  0}$的函数）。考虑基于这些变量定义的以下线性规划问题。

vertex-pack LP max $\mathop{\sum }\limits_{{X \in  {V}_{P}}}\nu \left( X\right)$ subject to

顶点打包线性规划 最大化$\mathop{\sum }\limits_{{X \in  {V}_{P}}}\nu \left( X\right)$，约束条件为

$$
\mathop{\sum }\limits_{{X : X \in  F}}\nu \left( X\right)  \leq  1\;\forall F \in  {E}_{P}
$$

$$
\nu \left( X\right)  \geq  0\;\forall X \in  {V}_{P}
$$

A feasible solution $\nu$ is said to be half-integral if $\nu \left( X\right)  \in  \left\{  {0,\frac{1}{2},1}\right\}$ for each $X \in  {V}_{P}$ .

如果对于每个$X \in  {V}_{P}$都有$\nu \left( X\right)  \in  \left\{  {0,\frac{1}{2},1}\right\}$，则称可行解$\nu$是半整数解。

Consider any fractional edge-cover decomposition of $P : \left( {{\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$ . Recall that each star ${ \star  }_{j}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)$ contains a vertex designated as the center. We refer to every other vertex in the star as a petal.

考虑$P : \left( {{\partial }_{1},\ldots ,{\partial }_{\alpha },{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right)$的任意分数边覆盖分解。回想一下，每个星型结构${ \star  }_{j}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)$都包含一个被指定为中心的顶点。我们将星型结构中的其他每个顶点称为花瓣。

Lemma D.3. For the vertex-pack ${LP}$ ,there exists a half-integral optimal solution $\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$ satisfying all the conditions below.

引理D.3。对于顶点打包${LP}$，存在一个满足以下所有条件的半整数最优解$\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$。

- ${\nu }^{ * }\left( X\right)  = {0.5}$ for every vertex $X$ in ${\partial }_{1},\ldots {\partial }_{\alpha }$ .

- 对于${\partial }_{1},\ldots {\partial }_{\alpha }$中的每个顶点$X$，有${\nu }^{ * }\left( X\right)  = {0.5}$。

- For every $j \in  \left\lbrack  \beta \right\rbrack$ ,the function ${\nu }^{ * }$ has the properties below.

- 对于每个$j \in  \left\lbrack  \beta \right\rbrack$，函数${\nu }^{ * }$具有以下性质。

- If ${ \star  }_{j}$ has at least two edges,then ${\nu }^{ * }\left( X\right)  = 0$ holds for the star’s center $X$ and ${\nu }^{ * }\left( Y\right)  = 1$ holds for every petal $Y$ .

- 如果${ \star  }_{j}$至少有两条边，那么对于星型结构的中心$X$有${\nu }^{ * }\left( X\right)  = 0$成立，并且对于每个花瓣$Y$有${\nu }^{ * }\left( Y\right)  = 1$成立。

- If ${ \star  }_{j}$ has only one edge $\{ X,Y\}$ - in which case we call ${ \star  }_{j}$ a "one-edge star" - then ${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  = 1$ .

- 如果${ \star  }_{j}$只有一条边$\{ X,Y\}$（在这种情况下，我们称${ \star  }_{j}$为“单边星型结构”），那么${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  = 1$。

Proof. Let us first define the dual of vertex-pack LP. Associate each edge $F \in  {E}_{P}$ with a variable $w\left( F\right)  \geq  0$ (equivalently, $w$ is a function from ${E}_{P}$ to ${\mathbb{R}}_{ \geq  0}$ ). Then,the dual is:

证明。我们首先定义顶点打包线性规划的对偶问题。将每条边$F \in  {E}_{P}$与一个变量$w\left( F\right)  \geq  0$关联（等价地，$w$是一个从${E}_{P}$到${\mathbb{R}}_{ \geq  0}$的函数）。那么，对偶问题为：

edge-cover LP $\;\min \mathop{\sum }\limits_{{F \in  {E}_{P}}}w\left( F\right)$ subject to

边覆盖线性规划 $\;\min \mathop{\sum }\limits_{{F \in  {E}_{P}}}w\left( F\right)$，约束条件为

$$
\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}w\left( F\right)  \geq  1\;\forall X \in  {V}_{P}
$$

$$
w\left( F\right)  \geq  0\;\forall F \in  {E}_{P}
$$

The given fractional edge-cover decomposition implies an optimal solution $\left\{  {{w}^{ * }\left( F\right)  \mid  F \in  {E}_{p}}\right\}$ to the edge-cover LP satisfying:

给定的分数边覆盖分解意味着边覆盖线性规划存在一个满足以下条件的最优解$\left\{  {{w}^{ * }\left( F\right)  \mid  F \in  {E}_{p}}\right\}$：

- ${w}^{ * }\left( F\right)  = {0.5}$ for every edge $F$ in ${\partial }_{1},\ldots {\partial }_{\alpha }$ ;

- 对于${\partial }_{1},\ldots {\partial }_{\alpha }$中的每条边$F$，有${w}^{ * }\left( F\right)  = {0.5}$；

- ${w}^{ * }\left( F\right)  = 1$ for every edge $F$ in ${ \star  }_{1},\ldots ,{ \star  }_{\beta }$ ;

- 对于${ \star  }_{1},\ldots ,{ \star  }_{\beta }$中的每条边$F$，有${w}^{ * }\left( F\right)  = 1$；

- ${w}^{ * }\left( F\right)  = 0$ for every other edge $F \in  {E}_{p}$ .

- 对于其他每条边$F \in  {E}_{p}$，有${w}^{ * }\left( F\right)  = 0$。

By the complementary slackness theorem,the function ${w}^{ * }$ implies an optimal solution $\left\{  {{\nu }^{\prime }\left( X\right)  \mid  X \in  }\right.$ $\left. {V}_{P}\right\}$ to the vertex-pack LP with the properties below.

根据互补松弛定理，函数${w}^{ * }$意味着顶点包装线性规划（vertex-pack LP）存在一个具有以下性质的最优解$\left\{  {{\nu }^{\prime }\left( X\right)  \mid  X \in  }\right.$ $\left. {V}_{P}\right\}$。

- P1: For every edge $\{ X,Y\}  \in  {E}_{p}$ ,if ${w}^{ * }\left( {\{ X,Y\} }\right)  > 0$ ,then ${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  = 1$ .

- P1：对于每条边$\{ X,Y\}  \in  {E}_{p}$，若${w}^{ * }\left( {\{ X,Y\} }\right)  > 0$，则${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  = 1$。

- P2: For every vertex $X$ satisfying $\mathop{\sum }\limits_{{X \in  F}}{w}^{ * }\left( F\right)  > 1,{\nu }^{\prime }\left( X\right)  = 0$ .

- P2：对于每个满足$\mathop{\sum }\limits_{{X \in  F}}{w}^{ * }\left( F\right)  > 1,{\nu }^{\prime }\left( X\right)  = 0$的顶点$X$。

From the above,we can derive additional properties of the solution $\left\{  {{\nu }^{\prime }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$ .

由上述内容，我们可以推导出解$\left\{  {{\nu }^{\prime }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$的其他性质。

- P3: ${\nu }^{\prime }\left( X\right)  = {0.5}$ for every vertex $X$ in ${ \ominus  }_{1},\ldots ,{ \ominus  }_{\alpha }$ . To see why,consider any ${ \ominus  }_{i}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ . Let ${X}_{1},{X}_{2},\ldots ,{X}_{\ell }$ (for some $\ell  \geq  3$ ) be the vertices of ${\partial }_{i}$ in clockwise order. Property P1 yields $\ell$ equations: ${\nu }^{\prime }\left( {X}_{j}\right)  + {\nu }^{\prime }\left( {X}_{j + 1}\right)  = 1$ for $j \in  \left\lbrack  {\ell  - 1}\right\rbrack$ and ${\nu }^{\prime }\left( {X}_{\ell }\right)  + {\nu }^{\prime }\left( {X}_{1}\right)  = 1$ . Solving the system of these equations gives ${\nu }^{\prime }\left( {X}_{j}\right)  = {0.5}$ for each $j \in  \left\lbrack  \ell \right\rbrack$ .

- P3：对于${ \ominus  }_{1},\ldots ,{ \ominus  }_{\alpha }$中的每个顶点$X$，有${\nu }^{\prime }\left( X\right)  = {0.5}$。为了说明原因，考虑任意的${ \ominus  }_{i}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$。设${X}_{1},{X}_{2},\ldots ,{X}_{\ell }$（对于某个$\ell  \geq  3$）是以顺时针顺序排列的${\partial }_{i}$的顶点。性质P1产生$\ell$个方程：对于$j \in  \left\lbrack  {\ell  - 1}\right\rbrack$和${\nu }^{\prime }\left( {X}_{\ell }\right)  + {\nu }^{\prime }\left( {X}_{1}\right)  = 1$，有${\nu }^{\prime }\left( {X}_{j}\right)  + {\nu }^{\prime }\left( {X}_{j + 1}\right)  = 1$。求解这些方程组成的方程组，可得对于每个$j \in  \left\lbrack  \ell \right\rbrack$，有${\nu }^{\prime }\left( {X}_{j}\right)  = {0.5}$。

- For every $j \in  \left\lbrack  \beta \right\rbrack$ ,we have:

- 对于每个$j \in  \left\lbrack  \beta \right\rbrack$，我们有：

- P4-1: If ${ \star  }_{j}$ has at least two edges,then ${\nu }^{\prime }\left( X\right)  = 0$ holds for the star’s center $X$ and ${\nu }^{\prime }\left( Y\right)  = 1$ for every petal $Y$ . To see why,notice that $\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}{w}^{ * }\left( F\right)$ is precisely the number of edges in ${ \star  }_{j}$ and,hence, $\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}{w}^{ * }\left( F\right)  > 1$ . Thus, $\mathbf{{P2}}$ asserts ${\nu }^{\prime }\left( X\right)  = 0$ . It then follows from $\mathbf{{P1}}$ that ${\nu }^{\prime }\left( Y\right)  = 1 - {\nu }^{\prime }\left( X\right)  = 1$ for every petal $Y$ .

- P4 - 1：若${ \star  }_{j}$至少有两条边，则对于星型结构的中心$X$，${\nu }^{\prime }\left( X\right)  = 0$成立，并且对于每个花瓣$Y$，有${\nu }^{\prime }\left( Y\right)  = 1$。为了说明原因，注意到$\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}{w}^{ * }\left( F\right)$恰好是${ \star  }_{j}$中的边的数量，因此，$\mathop{\sum }\limits_{{F \in  {E}_{P} : X \in  F}}{w}^{ * }\left( F\right)  > 1$。于是，$\mathbf{{P2}}$表明${\nu }^{\prime }\left( X\right)  = 0$成立。然后根据$\mathbf{{P1}}$可知，对于每个花瓣$Y$，有${\nu }^{\prime }\left( Y\right)  = 1 - {\nu }^{\prime }\left( X\right)  = 1$。

- P4-2: If ${ \star  }_{j}$ has only one edge $\{ X,Y\}$ ,then ${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  = 1$ . This directly follows from $\mathbf{{P1}}$ and the fact that ${w}^{ * }\left( {\{ X,Y\} }\right)  = 1$ .

- P4 - 2：如果${ \star  }_{j}$只有一条边$\{ X,Y\}$，那么${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  = 1$。这直接由$\mathbf{{P1}}$以及${w}^{ * }\left( {\{ X,Y\} }\right)  = 1$这一事实得出。

Now,we show how to construct ${\nu }^{ * }\left( \text{.}\right)$ . If ${\nu }^{\prime }\left( \text{.}\right)$ is already half-integral,then we set ${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)$ for all $X \in  {V}_{P}$ and finish. Otherwise,set

现在，我们展示如何构造${\nu }^{ * }\left( \text{.}\right)$。如果${\nu }^{\prime }\left( \text{.}\right)$已经是半整数的，那么我们对所有的$X \in  {V}_{P}$设置${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)$并结束。否则，设置

- ${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)$ for every $X \in  {V}_{P}$ with ${\nu }^{\prime }\left( X\right)  \in  \left\{  {0,1,\frac{1}{2}}\right\}$ ;

- 对于每个满足${\nu }^{\prime }\left( X\right)  \in  \left\{  {0,1,\frac{1}{2}}\right\}$的$X \in  {V}_{P}$，设置${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)$；

- ${\nu }^{ * }\left( X\right)  = 0$ for every $X \in  {V}_{P}$ with $0 < {\nu }^{\prime }\left( X\right)  < 1/2$ ;

- 对于每个满足$0 < {\nu }^{\prime }\left( X\right)  < 1/2$的$X \in  {V}_{P}$，设置${\nu }^{ * }\left( X\right)  = 0$；

- ${\nu }^{ * }\left( X\right)  = 1$ for every $X \in  {V}_{P}$ with $1/2 < {\nu }^{\prime }\left( X\right)  < 1$ .

- 对于每个满足$1/2 < {\nu }^{\prime }\left( X\right)  < 1$的$X \in  {V}_{P}$，设置${\nu }^{ * }\left( X\right)  = 1$。

We need to verify that ${\nu }^{ * }\left( \text{.}\right) {meetstheconditionslistedinthestatementofLemma\ D.3.\ Clearly,}$ ${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)  = 1/2$ for every vertex $X$ in ${\partial }_{1},\ldots {\partial }_{\alpha }$ (property $\mathbf{{P3}}$ ). For each $j \in  \left\lbrack  \beta \right\rbrack$ ,if ${ \star  }_{j}$ has at least two edges,then ${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)  = 0$ holds for the star’s center $X$ and ${\nu }^{ * }\left( Y\right)  = {\nu }^{\prime }\left( Y\right)  = 1$ holds for every petal $Y$ (property P4-1). If ${ \star  }_{j}$ has only one edge $\{ X,Y\}$ - that is, ${ \star  }_{j}$ is a one-edge star - property P4-2 tells us that ${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  = 1$ . If ${\nu }^{\prime }\left( X\right)  = {\nu }^{\prime }\left( Y\right)  = 1/2$ , then ${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  = 1/2 + 1/2 = 1$ . Otherwise,one vertex of $\{ X,Y\}  -$ say $X -$ satisfies $0 < {\nu }^{\prime }\left( X\right)  < 1/2$ ,and the other vertex satisfies $1/2 < {\nu }^{\prime }\left( Y\right)  < 1$ . Our construction ensures that ${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  = 0 + 1 = 1$

我们需要验证对于${\partial }_{1},\ldots {\partial }_{\alpha }$中的每个顶点$X$，${\nu }^{ * }\left( \text{.}\right) {meetstheconditionslistedinthestatementofLemma\ D.3.\ Clearly,}$ ${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)  = 1/2$（性质$\mathbf{{P3}}$）。对于每个$j \in  \left\lbrack  \beta \right\rbrack$，如果${ \star  }_{j}$至少有两条边，那么对于星的中心$X$，${\nu }^{ * }\left( X\right)  = {\nu }^{\prime }\left( X\right)  = 0$成立，并且对于每个花瓣$Y$，${\nu }^{ * }\left( Y\right)  = {\nu }^{\prime }\left( Y\right)  = 1$成立（性质P4 - 1）。如果${ \star  }_{j}$只有一条边$\{ X,Y\}$ —— 即${ \star  }_{j}$是单边星 —— 性质P4 - 2告诉我们${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  = 1$。如果${\nu }^{\prime }\left( X\right)  = {\nu }^{\prime }\left( Y\right)  = 1/2$，那么${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  = 1/2 + 1/2 = 1$。否则，$\{ X,Y\}  -$的一个顶点，比如说$X -$，满足$0 < {\nu }^{\prime }\left( X\right)  < 1/2$，而另一个顶点满足$1/2 < {\nu }^{\prime }\left( Y\right)  < 1$。我们的构造确保了${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  = 0 + 1 = 1$

It remains to check that $\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$ is indeed an optimal solution to the vertex-pack LP. First,we prove the the solution’s feasibility. For this purpose,given any edge $F = \{ X,Y\}  \in  {E}_{P}$ , we must show ${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  \leq  1$ . Assume there exists an edge $F = \{ X,Y\}  \in  {E}_{P}$ such that ${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  > 1$ . Because ${\nu }^{ * }\left( \text{.}\right) {ishalf}$ -integral $,{inthissituationatleastonevertexin}\{ X,Y\}  -$ say $X$ - must receive value ${\nu }^{ * }\left( X\right)  = 1$ . We proceed differently depending on the value of ${\nu }^{ * }\left( Y\right)$ .

接下来需要验证$\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$确实是顶点打包线性规划（vertex-pack LP）的最优解。首先，我们证明该解的可行性。为此，给定任意边$F = \{ X,Y\}  \in  {E}_{P}$，我们必须证明${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  \leq  1$。假设存在一条边$F = \{ X,Y\}  \in  {E}_{P}$使得${\nu }^{ * }\left( X\right)  + {\nu }^{ * }\left( Y\right)  > 1$。因为${\nu }^{ * }\left( \text{.}\right) {ishalf}$是整数的（integral），$,{inthissituationatleastonevertexin}\{ X,Y\}  -$，不妨设$X$，其值必须为${\nu }^{ * }\left( X\right)  = 1$。我们根据${\nu }^{ * }\left( Y\right)$的值进行不同的处理。

- ${\nu }^{ * }\left( Y\right)  = 1/2$ . By how ${\nu }^{ * }$ is constructed from ${\nu }^{\prime }$ ,in this case we must have ${\nu }^{\prime }\left( X\right)  > 1/2$ and ${\nu }^{\prime }\left( Y\right)  = 1/2$ . But then ${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  > 1$ ,violating the fact that ${\nu }^{\prime }\left( .\right)$ is a feasible solution to the vertex-pack LP.

- ${\nu }^{ * }\left( Y\right)  = 1/2$。根据${\nu }^{ * }$是如何由${\nu }^{\prime }$构造的，在这种情况下，我们必然有${\nu }^{\prime }\left( X\right)  > 1/2$和${\nu }^{\prime }\left( Y\right)  = 1/2$。但这样一来${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  > 1$，这与${\nu }^{\prime }\left( .\right)$是顶点打包线性规划的可行解这一事实相矛盾。

- ${\nu }^{ * }\left( Y\right)  = 1$ . By how ${\nu }^{ * }$ is constructed,we must have ${\nu }^{\prime }\left( X\right)  > 1/2$ and ${\nu }^{\prime }\left( Y\right)  > 1/2$ . Again, ${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  > 1$ ,violating the feasibility of ${\nu }^{\prime }\left( \text{.}\right) .$

- ${\nu }^{ * }\left( Y\right)  = 1$。根据${\nu }^{ * }$的构造方式，我们必然有${\nu }^{\prime }\left( X\right)  > 1/2$和${\nu }^{\prime }\left( Y\right)  > 1/2$。同样，${\nu }^{\prime }\left( X\right)  + {\nu }^{\prime }\left( Y\right)  > 1$，这违反了${\nu }^{\prime }\left( \text{.}\right) .$的可行性。

Finally,we will prove that $\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$ achieves the optimal value for the vertex-pack LP's objective function. This is true because

最后，我们将证明$\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$能使顶点打包线性规划的目标函数达到最优值。这是因为

$$
\mathop{\sum }\limits_{{X \in  {V}_{P}}}{\nu }^{ * }\left( X\right)  = \left( {\mathop{\sum }\limits_{{X : X\text{ not in any one-edge star }}}{\nu }^{ * }\left( X\right) }\right)  + \mathop{\sum }\limits_{{X : X\text{ in a one-edge star }}}{\nu }^{ * }\left( X\right) 
$$

$= \left( {\mathop{\sum }\limits_{{X : X\text{ not in any one-edge star }}}{\nu }^{\prime }\left( X\right) }\right)  +$ number of one-edge stars

$= \left( {\mathop{\sum }\limits_{{X : X\text{ not in any one-edge star }}}{\nu }^{\prime }\left( X\right) }\right)  +$单边星的数量

(property $\mathbf{{P4}} - \mathbf{2}$ ) $= \left( {\mathop{\sum }\limits_{{X : X\text{ not in any one-edge star }}}{\nu }^{\prime }\left( X\right) }\right)  + \mathop{\sum }\limits_{{X : X\text{ in a one-edge star }}}{\nu }^{\prime }\left( X\right)$

（性质$\mathbf{{P4}} - \mathbf{2}$）$= \left( {\mathop{\sum }\limits_{{X : X\text{ not in any one-edge star }}}{\nu }^{\prime }\left( X\right) }\right)  + \mathop{\sum }\limits_{{X : X\text{ in a one-edge star }}}{\nu }^{\prime }\left( X\right)$

$$
 = \mathop{\sum }\limits_{{X \in  {V}_{P}}}{\nu }^{\prime }\left( X\right) 
$$

and ${\nu }^{\prime }\left( \text{.}\right) {isanoptimalsolutiontothevertex} - {packLP}.$

并且${\nu }^{\prime }\left( \text{.}\right) {isanoptimalsolutiontothevertex} - {packLP}.$

We are now ready to prove Lemma D.2. Let $\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$ be an optimal solution to the vertex-pack LP problem promised by Lemma D.3. Divide ${V}_{P}$ into three subsets

现在我们准备证明引理D.2。设$\left\{  {{\nu }^{ * }\left( X\right)  \mid  X \in  {V}_{P}}\right\}$是引理D.3所保证的顶点打包线性规划问题的最优解。将${V}_{P}$划分为三个子集

$$
{U}_{A} = \left\{  {X \in  {V}_{P} \mid  {\nu }^{ * }\left( X\right)  = 1}\right\}  
$$

$$
{U}_{B} = \left\{  {X \in  {V}_{P} \mid  {\nu }^{ * }\left( X\right)  = 0}\right\}  
$$

$$
{U}_{C} = \left\{  {X \in  {V}_{P} \mid  {\nu }^{ * }\left( X\right)  = 1/2}\right\}  .
$$

By the feasibility of ${\nu }^{ * }\left( \text{.}\right) {tothesvertex} - {packLP}$ ,no vertex in ${U}_{A}{canbecenttoanyvertex}$ in ${U}_{C}$ ,and no two vertices in ${U}_{A}$ can be adjacent to each other. It remains to verify that the sizes of ${U}_{A},{U}_{B}$ ,and ${U}_{C}$ meet the requirement in the first bullet of Lemma D.2. To facilitate our subsequent argument,given a one-edge star ${ \star  }_{j}$ (for some $j \in  \left\lbrack  \beta \right\rbrack$ ),we call it a half-half one-edge star if ${\nu }^{ * }\left( X\right)  = {\nu }^{ * }\left( Y\right)  = 1/2$ ,where $\{ X,Y\}$ is the (only) edge in ${ \star  }_{j}$ . Define

根据${\nu }^{ * }\left( \text{.}\right) {tothesvertex} - {packLP}$的可行性，在${U}_{C}$中${U}_{A}{canbecenttoanyvertex}$里没有顶点，并且${U}_{A}$中任意两个顶点都不能相邻。接下来需要验证${U}_{A},{U}_{B}$和${U}_{C}$的大小是否满足引理D.2第一条的要求。为了便于后续论证，给定一个单边星${ \star  }_{j}$（对于某个$j \in  \left\lbrack  \beta \right\rbrack$），如果${\nu }^{ * }\left( X\right)  = {\nu }^{ * }\left( Y\right)  = 1/2$，我们称其为半对半单边星，其中$\{ X,Y\}$是${ \star  }_{j}$中的（唯一）边。定义

$$
s = \text{number of half-half one-edge stars in}\left\{  {{ \star  }_{1},\ldots ,{ \star  }_{\beta }}\right\}  \text{.}
$$

On the other hand,if a one-edge star ${ \star  }_{j}$ is not half-half,we call it a 0-1 one-edge star.

另一方面，如果一个单边星${ \star  }_{j}$不是半对半单边星，我们称其为0 - 1单边星。

By Lemma D.3, ${U}_{C}$ includes all the vertices in ${\partial }_{1},\ldots ,{\partial }_{\alpha }$ and all the vertices in the half-half one-edge stars. Hence, $\left| {U}_{C}\right|  = {k}_{\text{cycle }} + {2s}$ . On the other hand, ${U}_{B}$ includes (i) the center of every star that has at least two edges and (ii) exactly one vertex from every 0-1 one-edge star. In other words,every star contributes 1 to the size of ${U}_{B}$ except for the half-half one-edge stars,implying $\left| {U}_{B}\right|  = \beta  - s$ . Finally, $\left| {U}_{A}\right|  = k - \left| {U}_{B}\right|  - \left| {U}_{C}\right|  = {k}_{\text{star }} - \beta  - s$ . This completes the proof of Lemma D.2.

根据引理D.3，${U}_{C}$包含${\partial }_{1},\ldots ,{\partial }_{\alpha }$中的所有顶点以及所有半对半单边星中的所有顶点。因此，$\left| {U}_{C}\right|  = {k}_{\text{cycle }} + {2s}$。另一方面，${U}_{B}$包含：(i) 每个至少有两条边的星的中心；(ii) 每个0 - 1单边星中恰好一个顶点。换句话说，除了半对半单边星之外，每个星对${U}_{B}$的大小贡献为1，这意味着$\left| {U}_{B}\right|  = \beta  - s$。最后，$\left| {U}_{A}\right|  = k - \left| {U}_{B}\right|  - \left| {U}_{C}\right|  = {k}_{\text{star }} - \beta  - s$。至此，引理D.2证明完毕。

## E Analysis of the Sampling Algorithm in Section 5

## E 第5节中采样算法的分析

The purpose of preprocessing is essentially reorganizing the data graph $G = \left( {V,E}\right)$ in the adjacency-list format,which can be easily done in $O\left( \left| E\right| \right)$ expected time. The following discussion focuses on the cost of extracting a sample. We will first consider $\lambda  \leq  \sqrt{\left| E\right| }$ before attending to $\lambda  > \sqrt{\left| E\right| }$ .

预处理的目的本质上是以邻接表格式重新组织数据图$G = \left( {V,E}\right)$，这可以在$O\left( \left| E\right| \right)$的期望时间内轻松完成。接下来的讨论将重点关注提取样本的成本。我们将先考虑$\lambda  \leq  \sqrt{\left| E\right| }$，再考虑$\lambda  > \sqrt{\left| E\right| }$。

Case $\lambda  \leq  \sqrt{\left| E\right| }$ . Let ${G}_{\text{sub }} = \left( {{V}_{\text{sub }},{E}_{\text{sub }}}\right)$ be an occurrence of $P$ in $G$ . There exist a constant number $c$ of isomorphism bijections $g : {V}_{P} \rightarrow  {V}_{\text{sub }}$ between $P$ and ${G}_{\text{sub }}$ (the number $c$ is the number of automorphisms of $P$ ). Fix any such bijection $g$ . Each repeat of our algorithm builds a map $f : {V}_{p} \rightarrow  V$ . It is rudimentary to verify that $\Pr \left\lbrack  {f = g}\right\rbrack   = \frac{1}{2\left| E\right| } \cdot  \frac{1}{{\lambda }^{k - 2}}$ . Hence,each occurrence is returned with probability $\frac{c}{2\left| E\right| } \cdot  \frac{1}{{\lambda }^{k - 2}}$ . It is now straightforward to prove that our algorithm has expected sample time: $O\left( {\left| E\right|  \cdot  {\lambda }^{k - 2}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ .

情况$\lambda  \leq  \sqrt{\left| E\right| }$。设${G}_{\text{sub }} = \left( {{V}_{\text{sub }},{E}_{\text{sub }}}\right)$是$P$在$G$中的一个实例。在$P$和${G}_{\text{sub }}$之间存在常数个同构双射$g : {V}_{P} \rightarrow  {V}_{\text{sub }}$（这个常数$c$是$P$的自同构的数量）。固定任意一个这样的双射$g$。我们算法的每次重复都会构建一个映射$f : {V}_{p} \rightarrow  V$。很容易验证$\Pr \left\lbrack  {f = g}\right\rbrack   = \frac{1}{2\left| E\right| } \cdot  \frac{1}{{\lambda }^{k - 2}}$。因此，每个实例被返回的概率为$\frac{c}{2\left| E\right| } \cdot  \frac{1}{{\lambda }^{k - 2}}$。现在很容易证明我们的算法的期望采样时间为：$O\left( {\left| E\right|  \cdot  {\lambda }^{k - 2}/\max \{ 1,\mathrm{{OUT}}\} }\right)$。

Case $\mathbf{\lambda } > \sqrt{\left| \mathbf{E}\right| }$ . If OUT $= 0$ ,the two-thread approach allows our algorithm to terminate in $O\left( {{\text{ polymat }}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) }\right)  = O\left( {{m}^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}}\right)$ time. Next,we consider only OUT $\geq  1$ .

情况 $\mathbf{\lambda } > \sqrt{\left| \mathbf{E}\right| }$ 。如果输出 $= 0$ ，双线程方法允许我们的算法在 $O\left( {{\text{ polymat }}_{\text{undir }}\left( {\left| E\right| ,\lambda ,P}\right) }\right)  = O\left( {{m}^{\frac{{k}_{\text{cycle }}}{2} + \beta } \cdot  {\lambda }^{{k}_{\text{star }} - {2\beta }}}\right)$ 时间内终止。接下来，我们仅考虑输出 $\geq  1$ 。

For each $i \in  \left\lbrack  \alpha \right\rbrack$ ,denote by $k\left( {\square }_{i}\right)$ the number of vertices in ${\square }_{i}$ ; for each $j \in  \left\lbrack  \beta \right\rbrack$ ,denote by $k\left( { \star  }_{j}\right)$ the number of vertices in ${ \star  }_{j}$ . The fractional edge cover number of ${ \odot  }_{i}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ is ${\rho }^{ * }\left( { \odot  }_{i}\right)  = k\left( { \odot  }_{i}\right) /2$ . This means $\mathop{\sum }\limits_{{i = 1}}^{k}{\rho }^{ * }\left( {\square }_{i}\right)  = k\left( {\square }_{i}\right) /2 = {k}_{\text{cycle }}/2$ .

对于每个 $i \in  \left\lbrack  \alpha \right\rbrack$ ，用 $k\left( {\square }_{i}\right)$ 表示 ${\square }_{i}$ 中的顶点数量；对于每个 $j \in  \left\lbrack  \beta \right\rbrack$ ，用 $k\left( { \star  }_{j}\right)$ 表示 ${ \star  }_{j}$ 中的顶点数量。 ${ \odot  }_{i}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ 的分数边覆盖数是 ${\rho }^{ * }\left( { \odot  }_{i}\right)  = k\left( { \odot  }_{i}\right) /2$ 。这意味着 $\mathop{\sum }\limits_{{i = 1}}^{k}{\rho }^{ * }\left( {\square }_{i}\right)  = k\left( {\square }_{i}\right) /2 = {k}_{\text{cycle }}/2$ 。

Fix an arbitrary occurrence ${G}_{sub} = \left( {{V}_{sub},{E}_{sub}}\right)$ of $P$ in $G$ . Let $g$ be any of the $c$ isomorphism bijections between $P$ and ${G}_{\text{sub }}$ . Each repeat of our algorithm constructs a map $f : {V}_{P} \rightarrow  V$ through isomorphism bijections ${f}_{{\square }_{1}},\ldots ,{f}_{{\square }_{\alpha }},{f}_{{ \star  }_{1}},\ldots ,{f}_{{ \star  }_{\beta }}$ . The event $g = f$ happens if and only if all of the following events occur.

固定 $P$ 在 $G$ 中的任意一个出现 ${G}_{sub} = \left( {{V}_{sub},{E}_{sub}}\right)$ 。设 $g$ 是 $P$ 和 ${G}_{\text{sub }}$ 之间的 $c$ 个同构双射中的任意一个。我们算法的每次重复都通过同构双射 ${f}_{{\square }_{1}},\ldots ,{f}_{{\square }_{\alpha }},{f}_{{ \star  }_{1}},\ldots ,{f}_{{ \star  }_{\beta }}$ 构造一个映射 $f : {V}_{P} \rightarrow  V$ 。事件 $g = f$ 发生当且仅当以下所有事件都发生。

- Event ${\mathbf{E}}_{{\square }_{i}}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)  : g\left( A\right)  = {f}_{{\square }_{i}}\left( A\right)$ for each vertex $A$ of ${\square }_{i}$ ;

- 对于 ${\square }_{i}$ 的每个顶点 $A$ ，事件 ${\mathbf{E}}_{{\square }_{i}}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)  : g\left( A\right)  = {f}_{{\square }_{i}}\left( A\right)$ ；

- Event ${\mathbf{E}}_{{ \star  }_{j}}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)  : g\left( A\right)  = {f}_{{ \star  }_{i}}\left( A\right)$ for each vertex $A$ of ${ \star  }_{j}$ .

- 对于 ${ \star  }_{j}$ 的每个顶点 $A$ ，事件 ${\mathbf{E}}_{{ \star  }_{j}}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)  : g\left( A\right)  = {f}_{{ \star  }_{i}}\left( A\right)$ 。

If ${c}_{{\circlearrowright }_{i}}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ is the number of automorphisms of ${\circlearrowright }_{i}$ ,we have

如果 ${c}_{{\circlearrowright }_{i}}\left( {i \in  \left\lbrack  \alpha \right\rbrack  }\right)$ 是 ${\circlearrowright }_{i}$ 的自同构数量，我们有

$$
\Pr \left\lbrack  {\mathbf{E}}_{{\square }_{i}}\right\rbrack   = \frac{1}{{c}_{{\square }_{i}} \cdot  \left| {\operatorname{occ}\left( {G,{\square }_{i}}\right) }\right| }. \tag{45}
$$

Likewise,if ${c}_{{ \star  }_{j}}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)$ is the number of automorphisms of ${ \star  }_{j}$ ,we have

同样地，如果 ${c}_{{ \star  }_{j}}\left( {j \in  \left\lbrack  \beta \right\rbrack  }\right)$ 是 ${ \star  }_{j}$ 的自同构数量，我们有

$$
\Pr \left\lbrack  {\mathbf{E}}_{{ \star  }_{j}}\right\rbrack   = \frac{1}{{c}_{{ \star  }_{j}} \cdot  \left| {\operatorname{occ}\left( {G,{ \star  }_{j}}\right) }\right| }. \tag{46}
$$

It follows from (45) and (46) that

由 (45) 和 (46) 可得

$$
\Pr \left\lbrack  {g = f}\right\rbrack   = \left( {\mathop{\prod }\limits_{{i = 1}}^{\alpha }\frac{1}{{c}_{{\bigtriangleup }_{i}} \cdot  \left| {\operatorname{occ}\left( {G,{\bigtriangleup }_{i}}\right) }\right| }}\right)  \cdot  \left( {\mathop{\prod }\limits_{{j = 1}}^{\beta }\frac{1}{{c}_{{ \star  }_{j}} \cdot  \left| {\operatorname{occ}\left( {G,{ \star  }_{j}}\right) }\right| }}\right) .
$$

Therefore:

因此：

$$
\Pr \left\lbrack  {{G}_{\text{sub }}\text{ sampled }}\right\rbrack   = c \cdot  \left( {\mathop{\prod }\limits_{{i = 1}}^{\alpha }\frac{1}{{c}_{{\bigtriangleup }_{i}} \cdot  \left| {\operatorname{occ}\left( {G,{\bigtriangleup }_{i}}\right) }\right| }}\right)  \cdot  \left( {\mathop{\prod }\limits_{{j = 1}}^{\beta }\frac{1}{{c}_{{ \star  }_{j}} \cdot  \left| {\operatorname{occ}\left( {G,{ \star  }_{j}}\right) }\right| }}\right) . \tag{47}
$$

As this probability is identical for all ${G}_{sub}$ ,we know that each occurrence of $P$ is sampled with the same probability.

由于这个概率对于所有 ${G}_{sub}$ 都是相同的，我们知道 $P$ 的每个出现都以相同的概率被采样。

The expected number of repeats to obtain a sample occurrence of $P$ is

获得 $P$ 的一个样本出现的期望重复次数是

$$
O\left( \frac{\mathop{\prod }\limits_{{i = 1}}^{\alpha }\left| {\operatorname{occ}\left( {G,{\square }_{i}}\right) }\right|  \cdot  \mathop{\prod }\limits_{{j = 1}}^{\beta }\left| {\operatorname{occ}\left( {G,{ \star  }_{j}}\right) }\right| }{\mathrm{{OUT}}}\right) 
$$

whereas each repeat runs in time at the order of

而每次重复的运行时间量级为

$$
\left( {\mathop{\prod }\limits_{{i = 1}}^{\alpha }\frac{{\left| E\right| }^{{\rho }^{ * }\left( {\partial }_{i}\right) }}{\left| \operatorname{occ}\left( G,{\partial }_{i}\right) \right| }}\right)  \cdot  \left( {\mathop{\prod }\limits_{{j = 1}}^{\beta }\frac{\left| E\right|  \cdot  {\lambda }^{k\left( { \star  }_{j}\right)  - 2}}{\left| \operatorname{occ}\left( G,{ \star  }_{j}\right) \right| }}\right) . \tag{48}
$$

We can now conclude that the expected sample time is at the order of

我们现在可以得出结论，期望采样时间的量级为

$$
\frac{1}{\mathrm{{OUT}}} \cdot  \left( {\mathop{\prod }\limits_{{i = 1}}^{\alpha }{\left| E\right| }^{{\rho }^{ * }\left( {\bigtriangleup }_{i}\right) }}\right)  \cdot  \left( {\mathop{\prod }\limits_{{j = 1}}^{\beta }\left| E\right|  \cdot  {\lambda }^{k\left( { \star  }_{j}\right)  - 2}}\right)  = \frac{{\left| E\right| }^{{k}_{\text{cycle }}/2} \cdot  {\lambda }^{{k}_{\text{star }} - 2}}{\mathrm{{OUT}}}.
$$

References

[1] Amir Abboud, Seri Khoury, Oree Leibowitz, and Ron Safier. Listing 4-cycles. CoRR, abs/2211.10022, 2022.

[2] Serge Abiteboul, Richard Hull, and Victor Vianu. Foundations of Databases. Addison-Wesley, 1995.

[3] Swarup Acharya, Phillip B. Gibbons, Viswanath Poosala, and Sridhar Ramaswamy. Join synopses for approximate query answering. In Proceedings of ACM Management of Data (SIGMOD), pages 275-286, 1999.

[4] Noga Alon. On the number of subgraphs of prescribed type of graphs with a given number of edges. Israel Journal of Mathematics, 38:116-130, 1981.

[5] Kaleb Alway, Eric Blais, and Semih Salihoglu. Box covers and domain orderings for beyond worst-case join processing. In Proceedings of International Conference on Database Theory (ICDT), pages 3:1-3:23, 2021.

[6] Sepehr Assadi, Michael Kapralov, and Sanjeev Khanna. A simple sublinear-time algorithm for counting arbitrary subgraphs via edge sampling. In Innovations in Theoretical Computer Science (ITCS), pages 6:1-6:20, 2019.

[7] Albert Atserias, Martin Grohe, and Daniel Marx. Size bounds and query plans for relational joins. SIAM Journal on Computing, 42(4):1737-1767, 2013.

[8] Matthias Bentert, Till Fluschnik, Andre Nichterlein, and Rolf Niedermeier. Parameterized aspects of triangle enumeration. Journal of Computer and System Sciences (JCSS), 103:61-77, 2019.

[9] Andreas Bjorklund, Rasmus Pagh, Virginia Vassilevska Williams, and Uri Zwick. Listing triangles. In Proceedings of International Colloquium on Automata, Languages and Programming (ICALP), pages 223-234, 2014.

[10] Surajit Chaudhuri, Rajeev Motwani, and Vivek R. Narasayya. On random sampling over joins. In Proceedings of ACM Management of Data (SIGMOD), pages 263-274, 1999.

[11] Yu Chen and Ke Yi. Random sampling and size estimation over cyclic joins. In Proceedings of International Conference on Database Theory (ICDT), pages 7:1-7:18, 2020.

[12] N. Chiba and T. Nishizeki. Arboricity and subgraph listing algorithms. SIAM Journal of Computing, 14(1):210-223, 1985.

[13] Kyle Deeds, Dan Suciu, Magda Balazinska, and Walter Cai. Degree sequence bound for join cardinality estimation. In Proceedings of International Conference on Database Theory (ICDT), volume 255, pages 8:1-8:18, 2023.

[14] Shiyuan Deng, Shangqi Lu, and Yufei Tao. On join sampling and the hardness of combinatorial output-sensitive join algorithms. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 99-111, 2023.

[15] Davide Eppstein. Subgraph isomorphism in planar graphs and related problems. J. Graph Algorithms Appl., 3(3):1-27, 1999.

[16] Hendrik Fichtenberger, Mingze Gao, and Pan Peng. Sampling arbitrary subgraphs exactly uniformly in sublinear time. In Proceedings of International Colloquium on Automata, Languages and Programming (ICALP), pages 45:1-45:13, 2020.

[17] Tomasz Gogacz and Szymon Torunczyk. Entropy bounds for conjunctive queries with functional dependencies. In Proceedings of International Conference on Database Theory (ICDT), volume 68, pages 15:1-15:17, 2017.

[18] Chinh T. Hoang, Marcin Kaminski, Joe Sawada, and R. Sritharan. Finding and listing induced paths and cycles. Discrete Applied Mathematics, 161(4-5):633-641, 2013.

[19] Sai Vikneshwar Mani Jayaraman, Corey Ropell, and Atri Rudra. Worst-case optimal binary join algorithms under general ${\ell }_{p}$ constraints. CoRR,abs/2112.01003,2021.

[20] Ce Jin and Yinzhan Xu. Removing additive structure in 3sum-based reductions. In Proceedings of ACM Symposium on Theory of Computing (STOC), pages 405-418, 2023.

[21] Manas Joglekar and Christopher Re. It's all a matter of degree - using degree information to optimize multiway joins. Theory Comput. Syst., 62(4):810-853, 2018.

[22] Mahmoud Abo Khamis, Vasileios Nakos, Dan Olteanu, and Dan Suciu. Join size bounds using lp-norms on degree sequences. CoRR, abs/2306.14075, 2023.

[23] Mahmoud Abo Khamis, Hung Q. Ngo, Christopher Re, and Atri Rudra. Joins via geometric resolutions: Worst case and beyond. ACM Transactions on Database Systems (TODS), 41(4):22:1-22:45, 2016.

[24] Mahmoud Abo Khamis, Hung Q. Ngo, and Dan Suciu. Computing join queries with functional dependencies. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 327-342, 2016.

[25] Mahmoud Abo Khamis, Hung Q. Ngo, and Dan Suciu. What do shannon-type inequalities, submodular width, and disjunctive datalog have to do with one another? In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 429-444, 2017.

[26] Kyoungmin Kim, Jaehyun Ha, George Fletcher, and Wook-Shin Han. Guaranteeing the $\widetilde{O}$ (AGM/OUT) runtime for uniform sampling and size estimation over joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 113-125, 2023.

[27] George Manoussakis. Listing all fixed-length simple cycles in sparse graphs in optimal time. In Fundamentals of Computation Theory, pages 355-366, 2017.

[28] Gonzalo Navarro, Juan L. Reutter, and Javiel Rojas-Ledesma. Optimal joins using compact data structures. In Proceedings of International Conference on Database Theory (ICDT), volume 155, pages 21:1-21:21, 2020.

[29] Jaroslav Nesetril and Svatopluk Poljak. On the complexity of the subgraph problem. Commen-tationes Mathematicae Universitatis Carolinae, 26(2):415-419, 1985.

[30] Hung Q. Ngo. Worst-case optimal join algorithms: Techniques, results, and open problems. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 111-124, 2018.

[31] Hung Q. Ngo, Dung T. Nguyen, Christopher Re, and Atri Rudra. Beyond worst-case analysis for joins with minesweeper. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 234-245, 2014.

[32] Hung Q. Ngo, Ely Porat, Christopher Ré, and Atri Rudra. Worst-Case Optimal Join Algorithms: [Extended Abstract]. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 37-48, 2012.

[33] Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. Worst-case optimal join algorithms. Journal of the ${ACM}\left( {JACM}\right) ,{65}\left( 3\right)  : {16} : 1 - {16} : {40},{2018}$ .

[34] Hung Q. Ngo, Christopher Re, and Atri Rudra. Skew strikes back: new developments in the theory of join algorithms. SIGMOD Rec., 42(4):5-16, 2013.

[35] Alexander Schrijver. Combinatorial Optimization: Polyhedra and Efficiency. Springer-Verlag, 2003.

[36] Dan Suciu. Applications of information inequalities to database theory problems. CoRR, abs/2304.11996, 2023.

[37] Maciej M. Syslo. An efficient cycle vector space algorithm for listing all cycles of a planar graph. SIAM Journal of Computing, 10(4):797-808, 1981.

[38] Todd L. Veldhuizen. Triejoin: A simple, worst-case optimal join algorithm. In Proceedings of International Conference on Database Theory (ICDT), pages 96-106, 2014.

[39] Zhuoyue Zhao, Robert Christensen, Feifei Li, Xiao Hu, and Ke Yi. Random sampling over joins revisited. In Proceedings of ACM Management of Data (SIGMOD), pages 1525-1539, 2018.