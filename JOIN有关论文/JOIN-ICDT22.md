# Parallel Acyclic Joins with Canonical Edge Covers

# 基于规范边覆盖的并行无环连接

Yufei Tao

陶宇飞

Department of Computer Science and Engineering

计算机科学与工程系

Chinese University of Hong Kong

香港中文大学

taoyf@cse.cuhk.edu.hk

January 11, 2022

2022年1月11日

## Abstract

## 摘要

In PODS'21, Hu presented an algorithm in the massively parallel computation (MPC) model that processes any acyclic join with an asymptotically optimal load. In this paper, we present an alternative analysis of her algorithm. The novelty of our analysis is in the revelation of a new mathematical structure - which we name canonical edge cover - for acyclic hypergraphs. We prove non-trivial properties for canonical edge covers that offer us a graph-theoretic perspective about why Hu's algorithm works.

在2021年的数据库系统原理研讨会（PODS'21）上，胡（Hu）提出了一种在大规模并行计算（MPC）模型中处理任意无环连接的算法，该算法具有渐近最优负载。在本文中，我们对她的算法进行了另一种分析。我们分析的新颖之处在于揭示了无环超图的一种新的数学结构——我们将其命名为规范边覆盖。我们证明了规范边覆盖的非平凡性质，这些性质为我们提供了一个图论视角，解释了胡的算法为何有效。

Keywords: Joins, Conjunctive Queries, MPC Algorithms, Parallel Computing.

关键词：连接、合取查询、MPC算法、并行计算。

Acknowledgments: This work was partially supported by GRF projects 142078/20 and 142034/21 from HKRGC.

致谢：本工作部分得到了香港研究资助局（HKRGC）的研资局研究项目142078/20和142034/21的支持。

## 1 Introduction

## 1 引言

Massively parallel join processing has attracted considerable attention in recent years. This line of research makes two types of contributions. The first consists of algorithms that promise excellent performance. The second, more subtle, type of contributions comprises knowledge revealing mathematical structures in the underlying problems. The latter is a necessary side-product of the former. In general, as human beings switch to a more generic setting, their knowledge from restrictive settings often proves insufficient, which then necessitates deeper investigation into the problem characteristics. Traditional studies have focused on joins in the RAM computation model $\lbrack 4,{14},{16},{18}$ , 19], a degenerated "parallel" setup having only one machine. Designing algorithms to work with any number of machines poses serious challenges and demands novel findings $\left\lbrack  {3,7,8,{10},{12},{13},{15},{20},{21}}\right\rbrack$ beyond the RAM literature.

近年来，大规模并行连接处理引起了广泛关注。这一研究方向有两类贡献。第一类是能保证卓越性能的算法。第二类贡献更为微妙，包括揭示底层问题中数学结构的知识。后者是前者的必要副产品。一般来说，当人们转向更通用的场景时，他们在受限场景中的知识往往显得不足，这就需要对问题特征进行更深入的研究。传统研究主要集中在随机存取存储器（RAM）计算模型中的连接 $\lbrack 4,{14},{16},{18}$ , 19]，这是一种退化的“并行”设置，只有一台机器。设计适用于任意数量机器的算法面临着严峻挑战，需要超越RAM文献的新发现 $\left\lbrack  {3,7,8,{10},{12},{13},{15},{20},{21}}\right\rbrack$。

This paper will focus on acyclic joins, a class of joins with profound importance in database systems $\left\lbrack  {1,7 - 9,{11},{22}}\right\rbrack$ . Recently,Hu $\left\lbrack  8\right\rbrack$ developed a worst-case optimal massively parallel algorithm for acyclic joins. In the current work, we will provide an alternative, hopefully more accessible, analysis of her elegant algorithm. The real excitement from our analysis is the identification of a new mathematical structure - we call "canonical edge cover" - for acyclic hypergraphs. The structure reveals a unique characteristic of acyclic joins and is a core reason why Hu's algorithm works.

本文将聚焦于无环连接，这是一类在数据库系统中具有重要意义的连接 $\left\lbrack  {1,7 - 9,{11},{22}}\right\rbrack$。最近，胡 $\left\lbrack  8\right\rbrack$ 为无环连接开发了一种最坏情况下最优的大规模并行算法。在当前的工作中，我们将对她的精妙算法进行另一种分析，希望这种分析更容易理解。我们分析的真正亮点在于识别出了无环超图的一种新的数学结构——我们称之为“规范边覆盖”。这种结构揭示了无环连接的独特特征，也是胡的算法有效的核心原因。

### 1.1 Problem Definition

### 1.1 问题定义

Acyclic Joins. Let att be a set where each element is called an attribute. Let dom be another set where each element is called a value. We assume a total order on dom; if not, manually impose one by ordering the values arbitrarily.

无环连接。设att是一个集合，其中每个元素称为一个属性。设dom是另一个集合，其中每个元素称为一个值。我们假设dom上有一个全序；如果没有，则通过任意对值进行排序来手动施加一个全序。

A tuple over a set $U \subseteq$ att is a function $\mathbf{u} : U \rightarrow$ dom. For each attribute $X \in  U$ ,we refer to $\mathbf{u}\left( X\right)$ as the value of $\mathbf{u}$ on $X$ . Given a subset ${U}^{\prime } \subseteq  U$ ,define $\mathbf{u}\left\lbrack  {U}^{\prime }\right\rbrack   -$ the projection of $\mathbf{u}$ on ${U}^{\prime } -$ as the tuple ${\mathbf{u}}^{\prime }$ over ${U}^{\prime }$ such that ${\mathbf{u}}^{\prime }\left( X\right)  = \mathbf{u}\left( X\right)$ for every $X \in  {U}^{\prime }$ . A relation is a set $R$ of tuples over the same set $U$ of attributes. We call $U$ the scheme of $R$ ,a fact denoted as $\operatorname{scheme}\left( R\right)  = U$ . If $U$ is the empty set $\varnothing$ ,then $R$ is also $\varnothing$ .

集合 $U \subseteq$ att上的一个元组是一个函数 $\mathbf{u} : U \rightarrow$ dom。对于每个属性 $X \in  U$，我们将 $\mathbf{u}\left( X\right)$ 称为 $\mathbf{u}$ 在 $X$ 上的值。给定一个子集 ${U}^{\prime } \subseteq  U$，定义 $\mathbf{u}\left\lbrack  {U}^{\prime }\right\rbrack   -$ 为 $\mathbf{u}$ 在 ${U}^{\prime } -$ 上的投影，即 ${U}^{\prime }$ 上的元组 ${\mathbf{u}}^{\prime }$，使得对于每个 $X \in  {U}^{\prime }$ 都有 ${\mathbf{u}}^{\prime }\left( X\right)  = \mathbf{u}\left( X\right)$。一个关系是同一属性集 $U$ 上元组的集合 $R$。我们称 $U$ 为 $R$ 的模式，记为 $\operatorname{scheme}\left( R\right)  = U$。如果 $U$ 是空集 $\varnothing$，那么 $R$ 也是 $\varnothing$。

We represent a join query (henceforth,simply a "join" or "query") as a set $Q$ of relations. Define attset $\left( Q\right)  = \mathop{\bigcup }\limits_{{R \in  Q}}$ scheme(R). The query result - denoted as $\operatorname{Join}\left( Q\right)$ - is the following relation over attset(Q)

我们将一个连接查询（此后简称为“连接”或“查询”）表示为一个关系集合 $Q$。定义attset $\left( Q\right)  = \mathop{\bigcup }\limits_{{R \in  Q}}$ scheme(R)。查询结果——记为 $\operatorname{Join}\left( Q\right)$——是attset(Q)上的以下关系

$$
\operatorname{Join}\left( Q\right)  = \{ \text{ tuple }\mathbf{u}\text{ over attset }\left( Q\right)  \mid  \forall R \in  Q,\mathbf{u}\left\lbrack  {\operatorname{scheme}\left( R\right) }\right\rbrack   \in  R\} .
$$

If the relations in $Q$ are ${R}_{1},{R}_{2},\ldots ,{R}_{\left| Q\right| }$ ,we may represent $\operatorname{Join}\left( Q\right)$ also as ${R}_{1} \bowtie  {R}_{2} \bowtie  \ldots  \bowtie  {R}_{\left| Q\right| }$ .

如果$Q$中的关系为${R}_{1},{R}_{2},\ldots ,{R}_{\left| Q\right| }$，我们也可以将$\operatorname{Join}\left( Q\right)$表示为${R}_{1} \bowtie  {R}_{2} \bowtie  \ldots  \bowtie  {R}_{\left| Q\right| }$。

$Q$ can be characterized by a hypergraph $G = \left( {V,E}\right)$ where each vertex in $V$ is a distinct attribute in attset(Q),and each hyperedge in $E$ is the scheme of a distinct relation in $Q.E$ may contain identical hyperedges because two (or more) relations in $Q$ can have the same scheme. The term "hyper" suggests that a hyperedge can have more than two attributes.

$Q$可以用超图$G = \left( {V,E}\right)$来表征，其中$V$中的每个顶点是attset(Q)中一个不同的属性，$E$中的每条超边是$Q.E$中一个不同关系的模式。$Q$中的两个（或更多）关系可以具有相同的模式，因此$Q.E$可能包含相同的超边。“超”这个术语表明一条超边可以有两个以上的属性。

A query is acyclic if its hypergraph is acyclic. Specifically,a hypergraph $G = \left( {V,E}\right)$ is acyclic if we can create a tree $T$ where

如果一个查询的超图是无环的，那么该查询就是无环的。具体来说，如果我们可以创建一棵树$T$，使得超图$G = \left( {V,E}\right)$是无环的

- every node in $T$ stores (and,hence,"corresponds to") a distinct hyperedge in $E$ ;

- $T$中的每个节点存储（因此“对应于”）$E$中的一条不同的超边；

- (connectedness requirement) for every attribute $X \in  V$ ,the set $S$ of nodes whose corresponding hyperedges contain $X$ forms a connected subtree in $T$ .

- （连通性要求）对于每个属性$X \in  V$，其对应超边包含$X$的节点集合$S$在$T$中形成一个连通子树。

<!-- Media -->

<!-- figureText: CEJ HN HK (EHJ) AI CLM CEF EFG BCE BO ABC BD -->

<img src="https://cdn.noedgeai.com/0195ccba-87e9-7860-ae95-5e250889b8c6_2.jpg?x=684&y=201&w=409&h=351&r=0"/>

Figure 1: A hyperedge tree example

图1：一个超边树示例

<!-- Media -->

We will call $T$ a hyperedge tree of $G$ (also known as the join tree of $Q$ in the literature).

我们将把$T$称为$G$的超边树（在文献中也称为$Q$的连接树）。

Example 1.1. Consider the hypergraph $G = \left( {V,E}\right)$ where $V = \{ \mathrm{A},\mathrm{B},\ldots ,0\}$ and $E = \{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{BO}}$ , EFG,BCE,CEF,CEJ,HI,LM,EHJ,KL,HK,HN\}. Figure 1 shows a hyperedge tree $T$ of $G$ . To understand the connectedness requirement, observe the connected subtree formed by the five hyperedges involving E.

示例1.1。考虑超图$G = \left( {V,E}\right)$，其中$V = \{ \mathrm{A},\mathrm{B},\ldots ,0\}$且$E = \{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{BO}}$，{EFG, BCE, CEF, CEJ, HI, LM, EHJ, KL, HK, HN}。图1展示了$G$的一个超边树$T$。为了理解连通性要求，观察涉及E的五条超边所形成的连通子树。

As $G$ and $T$ both contain "vertices" and "edges",for better clarity we will obey several conventions throughout the paper. A vertex in $G$ will always be referred to as an attribute,while the term node is reserved for the vertices in $T$ . Furthermore,to avoid confusion with hyperedges,we will always refer to an edge in $T$ as a link.

由于$G$和$T$都包含“顶点”和“边”，为了更清晰起见，我们在整篇论文中将遵循几个约定。$G$中的顶点将始终称为属性，而“节点”这个术语则专门用于$T$中的顶点。此外，为了避免与超边混淆，我们将始终把$T$中的边称为链接。

We use $m$ to denote the input size of $Q$ ,defined as $\mathop{\sum }\limits_{{R \in  Q}}\left| R\right|$ ,namely,the total number of tuples in the relations participating in the join.

我们使用$m$来表示$Q$的输入规模，定义为$\mathop{\sum }\limits_{{R \in  Q}}\left| R\right|$，即参与连接的关系中的元组总数。

Computation Model. We assume the massively parallel computation (MPC) model which is popular in designing massively parallel algorithms $\left\lbrack  {3,7,8,{10},{12},{13},{15},{20},{21}}\right\rbrack$ . In this model, we have $p$ machines,each storing $\Theta \left( {m/p}\right)$ tuples from the relations of a query $Q$ initially. An algorithm executes in rounds, each having two phases: in the first phase, each machine performs local computation; in the second, the machines exchange messages (every message must have been generated at the end of the first phase). An algorithm must finish in a constant number of rounds, and when it does,every tuple in $\operatorname{Join}\left( Q\right)$ must reside on at least one machine. The load of a round is the largest number of words received by a machine in that round. The load of an algorithm is the maximum load of all the rounds. The objective is to design an algorithm with the smallest load.

计算模型。我们假设采用大规模并行计算（MPC）模型，该模型在设计大规模并行算法$\left\lbrack  {3,7,8,{10},{12},{13},{15},{20},{21}}\right\rbrack$中很流行。在这个模型中，我们有$p$台机器，每台机器最初存储来自查询$Q$的关系中的$\Theta \left( {m/p}\right)$个元组。一个算法按轮执行，每一轮有两个阶段：在第一阶段，每台机器进行本地计算；在第二阶段，机器之间交换消息（每条消息必须在第一阶段结束时生成）。一个算法必须在固定的轮数内完成，并且完成时，$\operatorname{Join}\left( Q\right)$中的每个元组必须至少驻留在一台机器上。一轮的负载是该轮中一台机器接收到的最大单词数。一个算法的负载是所有轮次的最大负载。目标是设计一个负载最小的算法。

Math Conventions. The number $p$ of machines is considered to be at most ${m}^{1 - \epsilon }$ ,for some arbitrarily small constant $\epsilon  > 0$ . Every value in $\mathbf{{dom}}$ can be represented with $O\left( 1\right)$ words. Our discussion focuses on data complexities,namely,we are interested in the influence of $m$ on algorithm performance. For that reason,we assume that the hypergraph $G$ of $Q$ has $O\left( 1\right)$ vertices. Given an integer $x \geq  1$ ,the notation $\left\lbrack  x\right\rbrack$ represents the set $\{ 1,2,\ldots ,x\}$ .

数学约定。机器的数量 $p$ 被认为至多为 ${m}^{1 - \epsilon }$，其中 $\epsilon  > 0$ 是任意小的常数。$\mathbf{{dom}}$ 中的每个值都可以用 $O\left( 1\right)$ 个字来表示。我们的讨论集中在数据复杂度上，即，我们关注 $m$ 对算法性能的影响。因此，我们假设 $Q$ 的超图 $G$ 有 $O\left( 1\right)$ 个顶点。给定一个整数 $x \geq  1$，符号 $\left\lbrack  x\right\rbrack$ 表示集合 $\{ 1,2,\ldots ,x\}$。

### 1.2 Previous Results

### 1.2 先前的结果

Fractional Edge Coverings and the AGM bound. Consider a query $Q$ (which may or may not be acyclic) with hypergraph $G = \left( {V,E}\right)$ . Associate every hyperedge $e \in  E$ with a real-valued weight ${w}_{e}$ ,which falls between 0 and 1 . Impose a constraint on every attribute $X \in  V : \mathop{\sum }\limits_{{e \in  E : X \in  e}}{w}_{e} \geq  1$ , i.e.,the total weight of all the hyperedges covering $X$ must be at least 1 . A set of weights $\left\{  {{w}_{e} \mid  e \in  E}\right\}$ fulfilling all the constraints is a fractional edge covering of $G$ . If we define $\mathop{\sum }\limits_{{e \in  E}}{w}_{e}$ as the total weight of the fractional edge covering,the fractional edge covering number of $G$ - denoted as $\rho$ - is the minimum total weight of all possible fractional edge coverings. A fractional edge covering is optimal if its total weight equals $\rho$ .

分数边覆盖与AGM界。考虑一个查询 $Q$（它可能是无环的，也可能不是），其超图为 $G = \left( {V,E}\right)$。为每条超边 $e \in  E$ 关联一个实值权重 ${w}_{e}$，该权重介于0和1之间。对每个属性 $X \in  V : \mathop{\sum }\limits_{{e \in  E : X \in  e}}{w}_{e} \geq  1$ 施加一个约束，即，覆盖 $X$ 的所有超边的总权重必须至少为1。一组满足所有约束的权重 $\left\{  {{w}_{e} \mid  e \in  E}\right\}$ 是 $G$ 的一个分数边覆盖。如果我们将 $\mathop{\sum }\limits_{{e \in  E}}{w}_{e}$ 定义为分数边覆盖的总权重，那么 $G$ 的分数边覆盖数（记为 $\rho$）是所有可能的分数边覆盖的最小总权重。如果一个分数边覆盖的总权重等于 $\rho$，则它是最优的。

The ${AGM}$ bound,proved by Atserias,Grohe,and Marx [5],states that the size of $\operatorname{Join}\left( Q\right)$ is always bounded by $O\left( {m}^{\rho }\right)$ ; recall that $m$ is the input size of $Q$ . Furthermore,the bound is tight: in the worst case, $\left| {\operatorname{Join}\left( Q\right) }\right|$ can indeed reach $\Omega \left( {m}^{\rho }\right) \left\lbrack  5\right\rbrack$ .

由Atserias、Grohe和Marx [5] 证明的 ${AGM}$ 界表明，$\operatorname{Join}\left( Q\right)$ 的大小总是有界于 $O\left( {m}^{\rho }\right)$；回顾一下，$m$ 是 $Q$ 的输入大小。此外，这个界是紧的：在最坏的情况下，$\left| {\operatorname{Join}\left( Q\right) }\right|$ 确实可以达到 $\Omega \left( {m}^{\rho }\right) \left\lbrack  5\right\rbrack$。

Simplification for Acyclic Queries: Edge Covers. When $Q$ is acyclic, $G = \left( {V,E}\right)$ always admits an optimal fractional edge covering with integral weights [8]. Recall that all the weights ${w}_{e}\left( {e \in  E}\right)$ must fall between 0 and 1 . Hence,every weight in an optimal fractional edge covering must be either 0 or 1 . This pleasant property allows the reader to connect $\rho$ to edge "covers". A subset $S \subseteq  E$ is an edge cover ${r}^{1}$ of $G$ if every attribute of $V$ appears in at least one hyperedge of $S$ . Thus,the value of $\rho$ is simply the minimum size of all edge covers,namely,the smallest number of hyperedges that we must pick to cover all the attributes.

无环查询的简化：边覆盖。当 $Q$ 是无环的时，$G = \left( {V,E}\right)$ 总是存在一个具有整数值权重的最优分数边覆盖 [8]。回顾一下，所有的权重 ${w}_{e}\left( {e \in  E}\right)$ 必须介于0和1之间。因此，最优分数边覆盖中的每个权重必须是0或1。这个良好的性质使读者能够将 $\rho$ 与边“覆盖”联系起来。如果 $V$ 的每个属性至少出现在 $S$ 的一条超边中，则子集 $S \subseteq  E$ 是 $G$ 的一个边覆盖 ${r}^{1}$。因此，$\rho$ 的值就是所有边覆盖的最小规模，即，我们为了覆盖所有属性必须选取的最少超边数量。

Join Algorithms in RAM. An algorithm able to answer $Q$ using $O\left( {m}^{\rho }\right)$ time in the RAM model is worst-case optimal. Indeed,as $\left| {\operatorname{Join}\left( Q\right) }\right|$ can be $\Omega \left( {m}^{\rho }\right)$ ,we need $\Theta \left( {m}^{\rho }\right)$ time just to output $\operatorname{Join}\left( Q\right)$ in the worst case. Ngo et al. [17] designed the first algorithm that guarantees a running time of $O\left( {m}^{\rho }\right)$ for all queries. Since then,the community has discovered more algorithms $\left\lbrack  {4,{14},{16},{18},{19}}\right\rbrack$ that are all worst-cast optimal (sometimes up to a polylogarithmic factor) but differ in their own features. For an acyclic $Q$ ,an algorithm due to Yannakakis [22] achieves a stronger sense of optimality: his algorithm runs in $O\left( {m + \left| {\operatorname{Join}\left( Q\right) }\right| }\right)$ time,which is clearly the best regardless of $\left| {\operatorname{Join}\left( Q\right) }\right|$ .

随机存取存储器（RAM）中的连接算法。在随机存取存储器模型中，能够在$O\left( {m}^{\rho }\right)$时间内回答$Q$的算法是最坏情况最优的。实际上，由于$\left| {\operatorname{Join}\left( Q\right) }\right|$可以是$\Omega \left( {m}^{\rho }\right)$，在最坏情况下，仅输出$\operatorname{Join}\left( Q\right)$就需要$\Theta \left( {m}^{\rho }\right)$时间。Ngo等人[17]设计了第一个能保证所有查询的运行时间为$O\left( {m}^{\rho }\right)$的算法。从那时起，学界发现了更多$\left\lbrack  {4,{14},{16},{18},{19}}\right\rbrack$算法，这些算法都是最坏情况最优的（有时相差一个多对数因子），但各自的特点不同。对于无环的$Q$，Yannakakis[22]提出的算法实现了更强意义上的最优性：他的算法运行时间为$O\left( {m + \left| {\operatorname{Join}\left( Q\right) }\right| }\right)$，显然这是最优的，与$\left| {\operatorname{Join}\left( Q\right) }\right|$无关。

Join Algorithms in MPC. Koutris, Beame, and Suciu [15] showed that, in the MPC model, the AGM bound implies a worst-case lower bound of $\Omega \left( {m/{p}^{1/\rho }}\right)$ on the load of any algorithm that answers a query $Q$ ,where $m$ is the input size of $Q$ and $\rho$ is the fractional edge covering number of the hypergraph $G = \left( {V,E}\right)$ defined by $Q$ .

大规模并行计算（MPC）中的连接算法。Koutris、Beame和Suciu[15]表明，在大规模并行计算模型中，AGM边界意味着任何回答查询$Q$的算法的负载在最坏情况下的下界为$\Omega \left( {m/{p}^{1/\rho }}\right)$，其中$m$是$Q$的输入大小，$\rho$是由$Q$定义的超图$G = \left( {V,E}\right)$的分数边覆盖数。

The above negative result has motivated considerable research looking for MPC algorithms whose loads are bounded by $O\left( {m/{p}^{1/\rho }}\right)$ ,ignoring polylogarithmic factors; such algorithms are worst-case optimal. The goal has been realized only on four query classes. The first consists of all the cartesian-product queries (i.e.,the relations in $Q$ have disjoint schemes); see $\left\lbrack  {3,6,{13}}\right\rbrack$ for several optimal algorithms on such queries. The second is the so-called Loomis-Whitney join,where $E$ consists of all the $\left| V\right|$ possible hyperedges of $\left| V\right|  - 1$ attributes; see [15] for an optimal algorithm for such joins. The third class includes every query where all the hyperedges in $G$ contain at most two attributes; see $\left\lbrack  {{12},{13},{21}}\right\rbrack$ for the optimal algorithms. The fourth class comprises all the acyclic queries,which were recently solved by $\mathrm{{Hu}}\left\lbrack  8\right\rbrack$ optimally. It is worth pointing out that $\mathrm{{Hu}}$ ’s algorithm subsumes an earlier algorithm of [9] which is worst-case optimal on a subclass of acyclic queries.

上述负面结果促使人们进行了大量研究，以寻找负载受$O\left( {m/{p}^{1/\rho }}\right)$限制（忽略多对数因子）的大规模并行计算算法；这样的算法是最坏情况最优的。这一目标仅在四类查询中得以实现。第一类包括所有笛卡尔积查询（即$Q$中的关系具有不相交的模式）；有关此类查询的几种最优算法见$\left\lbrack  {3,6,{13}}\right\rbrack$。第二类是所谓的卢米斯 - 惠特尼连接（Loomis - Whitney join），其中$E$由$\left| V\right|  - 1$个属性的所有$\left| V\right|$种可能的超边组成；有关此类连接的最优算法见[15]。第三类包括$G$中所有超边最多包含两个属性的每个查询；有关最优算法见$\left\lbrack  {{12},{13},{21}}\right\rbrack$。第四类包括所有无环查询，最近$\mathrm{{Hu}}\left\lbrack  8\right\rbrack$对其进行了最优求解。值得指出的是，$\mathrm{{Hu}}$的算法包含了[9]早期的一个算法，该算法在无环查询的一个子类上是最坏情况最优的。

Although it still remains elusive what other query classes can be settled with load $O\left( {m/{p}^{1/\rho }}\right)$ , now we know that this is unachievable for certain queries. In [8], Hu constructed a class of queries for which every algorithm must incur a load of $\omega \left( {m/{p}^{1/\rho }}\right)$ in the worst case. The result of [8] suggests that additional parameters - other than $m,p$ ,and $\rho$ - are needed to describe the worst-case optimality of an ideal MPC algorithm. We will not delve into the issue further because it does not apply to acyclic queries (the focus of this paper), but the reader may consult the recent works $\left\lbrack  {8,{20}}\right\rbrack$ for the latest development on that issue. Finally,we remark that several algorithms $\left\lbrack  {2,9,{10}}\right\rbrack$ are able to achieve a load sensitive to the join size $\left| {\operatorname{Join}\left( Q\right) }\right|$ .

尽管目前仍不清楚还有哪些查询类可以用负载$O\left( {m/{p}^{1/\rho }}\right)$来解决，但现在我们知道，对于某些查询，这是无法实现的。在[8]中，Hu构造了一类查询，对于这类查询，每个算法在最坏情况下都必须产生$\omega \left( {m/{p}^{1/\rho }}\right)$的负载。[8]的结果表明，除了$m,p$和$\rho$之外，还需要其他参数来描述理想的大规模并行计算算法的最坏情况最优性。我们不会进一步深入探讨这个问题，因为它不适用于无环查询（本文的重点），但读者可以参考近期的研究$\left\lbrack  {8,{20}}\right\rbrack$以了解该问题的最新进展。最后，我们注意到，有几种算法$\left\lbrack  {2,9,{10}}\right\rbrack$能够实现对连接大小$\left| {\operatorname{Join}\left( Q\right) }\right|$敏感的负载。

---

<!-- Footnote -->

${}^{1}$ In case the reader is wondering,the literature uses the words "covering" and "cover" exactly the way they are used in our paper.

${}^{1}$ 以防读者有疑问，相关文献中“覆盖（covering）”和“覆盖集（cover）”的用法与本文完全一致。

<!-- Footnote -->

---

### 1.3 Our Contributions

### 1.3 我们的贡献

The first, easy-to-discern, contribution of our paper is a new analysis of Hu's algorithm [8] for acyclic queries. Our second contribution is the introduction of canonical edge cover as a mathematical structure inherent in acyclic queries. We prove a suite of graph-theoretic properties for canonical edge covers and use them to give a more fundamental interpretation of the design choices in Hu's algorithm. The rest of the section will provide an overview of our results and techniques.

本文第一个易于识别的贡献是对胡氏（Hu）针对无环查询的算法 [8] 进行了新的分析。我们的第二个贡献是引入了规范边覆盖（canonical edge cover）这一无环查询中固有的数学结构。我们证明了规范边覆盖的一系列图论性质，并利用这些性质对胡氏算法中的设计选择给出了更基础的解释。本节其余部分将概述我们的研究结果和技术。

Clustering, $k$ -Groups, $k$ -Products,and Induced Loads. We first create a conceptual framework to state Hu’s and our results on a common ground. Define a clustering of $E$ (the hyperedge set of $G$ ) as a set $\left\{  {{E}_{1},{E}_{2},\ldots ,{E}_{s}}\right\}$ for some $s \geq  1$ where (i) each ${E}_{i}$ is a subset of $E,i \in  \left\lbrack  s\right\rbrack$ ,and (ii) $\mathop{\bigcup }\limits_{i}{E}_{i} = E$ . We call each ${E}_{i}$ a cluster; note that the clusters need not be disjoint.

聚类、$k$ -组、$k$ -积和诱导负载。我们首先创建一个概念框架，以便在共同基础上阐述胡氏的结果和我们的结果。将 $E$（$G$ 的超边集）的聚类定义为集合 $\left\{  {{E}_{1},{E}_{2},\ldots ,{E}_{s}}\right\}$，其中 $s \geq  1$ 满足：（i）每个 ${E}_{i}$ 是 $E,i \in  \left\lbrack  s\right\rbrack$ 的子集；（ii）$\mathop{\bigcup }\limits_{i}{E}_{i} = E$。我们称每个 ${E}_{i}$ 为一个簇；注意，这些簇不必互不相交。

Fix an arbitrary clustering $C = \left\{  {{E}_{1},{E}_{2},\ldots ,{E}_{s}}\right\}$ . Given an integer $k \geq  1$ ,define a $k$ -group of $C$ as a collection of $k$ hyperedges,each taken from a distinct cluster.

固定一个任意的聚类 $C = \left\{  {{E}_{1},{E}_{2},\ldots ,{E}_{s}}\right\}$。给定一个整数 $k \geq  1$，将 $C$ 的一个 $k$ -组定义为 $k$ 条超边的集合，每条超边取自不同的簇。

Example 1.2. Let $G = \left( {V,E}\right)$ be the hypergraph in Example 1.1 (Figure 1). $C = \{ \{ \mathrm{B}0,\mathrm{{BCE}}$ , CEJ\}, \{ABC, BCE, CEJ\}, \{BD, BCE, CEJ\}, \{EFG, CEF, CEJ\}, \{HI\}, \{EMJ\}, \{LM, KL\}, \{HK\}, \{LM, KL\}, \{HK\}, \{HN\}\} is a clustering of $E$ . A 3-group example is $\{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{EFG}}\}$ . Note that the hyperedges in a $k$ -group need not be distinct. For example, $\{ \mathrm{{CEJ}},\mathrm{{CEJ}},\mathrm{{CEJ}}\}$ is also a 3-group: the first $\mathrm{{CEJ}}$ is taken from the cluster $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$ ,the second from $\{ \mathrm{{BD}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$ ,and the third from $\{ \mathrm{{EFG}},\mathrm{{CEF}},\mathrm{{CEJ}}\}$ . For a non-example, $\{ \mathrm{{ABC}},\mathrm{{LM}},\mathrm{{KL}}\}$ is not a 3-group.

示例 1.2。设 $G = \left( {V,E}\right)$ 为例 1.1（图 1）中的超图。$C = \{ \{ \mathrm{B}0,\mathrm{{BCE}}$，{CEJ}，{ABC, BCE, CEJ}，{BD, BCE, CEJ}，{EFG, CEF, CEJ}，{HI}，{EMJ}，{LM, KL}，{HK}，{LM, KL}，{HK}，{HN}} 是 $E$ 的一个聚类。一个 3 - 组的示例是 $\{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{EFG}}\}$。注意，一个 $k$ - 组中的超边不必是不同的。例如，$\{ \mathrm{{CEJ}},\mathrm{{CEJ}},\mathrm{{CEJ}}\}$ 也是一个 3 - 组：第一条 $\mathrm{{CEJ}}$ 取自簇 $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$，第二条取自 $\{ \mathrm{{BD}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$，第三条取自 $\{ \mathrm{{EFG}},\mathrm{{CEF}},\mathrm{{CEJ}}\}$。作为一个反例，$\{ \mathrm{{ABC}},\mathrm{{LM}},\mathrm{{KL}}\}$ 不是一个 3 - 组。

For each hyperedge $e \in  E$ ,let $R\left( e\right)$ represent the relation in $Q$ whose scheme is $e$ . Given a $k$ -group $K$ of the clustering $C$ ,we define the $Q$ -product of $K$ as $\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( e\right) }\right|$ (i.e.,the cartesian-product size of all the relevant relations). Given an integer $k$ ,we define the $\max \left( {k,Q}\right)$ -product of $C$ — denoted as ${P}_{k}\left( {Q,C}\right)$ — as the maximum $Q$ -product of all the $k$ -groups of $C$ .

对于每条超边 $e \in  E$，设 $R\left( e\right)$ 表示 $Q$ 中模式为 $e$ 的关系。给定聚类 $C$ 的一个 $k$ - 组 $K$，我们将 $K$ 的 $Q$ - 积定义为 $\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( e\right) }\right|$（即，所有相关关系的笛卡尔积的大小）。给定一个整数 $k$，我们将 $C$ 的 $\max \left( {k,Q}\right)$ - 积（记为 ${P}_{k}\left( {Q,C}\right)$）定义为 $C$ 的所有 $k$ - 组的最大 $Q$ - 积。

Example 1.3. Continuing on the previous example,the $Q$ -product of the 3-group $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$ is $\left| {R\left( \mathrm{{ABC}}\right) }\right|  \cdot  \left| {R\left( \mathrm{{BCE}}\right) }\right|  \cdot  \left| {R\left( \mathrm{{CEJ}}\right) }\right|$ ,while that of the 3-group $\{ \mathrm{{CEJ}},\mathrm{{CEJ}},\mathrm{{CEJ}}\}$ is ${\left| R\left( \mathrm{{CEJ}}\right) \right| }^{3}$ .

示例1.3。延续上一个示例，3 - 群$\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$的$Q$ - 积是$\left| {R\left( \mathrm{{ABC}}\right) }\right|  \cdot  \left| {R\left( \mathrm{{BCE}}\right) }\right|  \cdot  \left| {R\left( \mathrm{{CEJ}}\right) }\right|$，而3 - 群$\{ \mathrm{{CEJ}},\mathrm{{CEJ}},\mathrm{{CEJ}}\}$的$Q$ - 积是${\left| R\left( \mathrm{{CEJ}}\right) \right| }^{3}$。

Define the $Q$ -induced load of $C$ as

将$C$的$Q$ - 诱导负载定义为

$$
\mathop{\max }\limits_{{k = 1}}^{s}{\left( {P}_{k}\left( Q,C\right) /p\right) }^{1/k} \tag{1}
$$

As ${P}_{k}\left( {Q,C}\right)  \leq  {m}^{k}$ for any $k \in  \left\lbrack  s\right\rbrack$ ,it must hold that $\left( 1\right)  \leq  m/{p}^{1/s}$ .

由于对于任意$k \in  \left\lbrack  s\right\rbrack$都有${P}_{k}\left( {Q,C}\right)  \leq  {m}^{k}$，那么必然有$\left( 1\right)  \leq  m/{p}^{1/s}$。

We can now give a more detailed account of Hu's result [8]. She proved that the load of her algorithm is bounded by $O\left( L\right)$ ,where $L$ is the $Q$ -induced load of a certain clustering with size $s = \rho$ , and $\rho$ is the fractional edge covering number of $G$ . It thus follows immediately that $L \leq  m/{p}^{1/s}$ . In [8],Hu presented a recursive procedure to identify the clustering $C$ whose $Q$ -induced load equals the target $L$ . The procedure,however,is somewhat sophisticated,making it difficult to describe the target $C$ in a succinct manner. Such difficulty is unjustified,especially given the algorithm’s elegance, and indicates the existence of a hidden mathematical structure.

现在我们可以更详细地阐述胡（Hu）的结果[8]。她证明了其算法的负载受$O\left( L\right)$限制，其中$L$是规模为$s = \rho$的某个聚类的$Q$ - 诱导负载，$\rho$是$G$的分数边覆盖数。因此，立即可以得出$L \leq  m/{p}^{1/s}$。在文献[8]中，胡提出了一个递归过程来确定聚类$C$，其$Q$ - 诱导负载等于目标值$L$。然而，该过程有些复杂，难以简洁地描述目标聚类$C$。这种困难是不合理的，特别是考虑到该算法的精妙性，这表明存在一种隐藏的数学结构。

Our Results and Techniques. A hypergraph $G$ can have many optimal edge covers (all of which must have size $\rho$ ). While Hu’s analysis [8] assumes an arbitrary optimal edge cover,we will be choosy about what we work with. In Figure 1, the 9 circled nodes constitute a canonical edge cover $F$ of $G$ . Let us give an informal but intuitive explanation of how to construct this $F$ . After rooting the tree in Figure 1 at HN,we add to $F$ all the leaf nodes: BO,ABC,BD,EFG,HI,LM. Then,we process the non-leaf nodes bottom up. In processing BCE, we ask: which attributes will disappear as we ascend further in the tree? The answer is B, which is thus a "disappearing" attribute of BCE. Then, we ask: does $F$ already cover $\mathrm{B}$ ? The answer is yes,due to the existence of $\mathrm{{BO}}$ ; we therefore do not include BCE in $F$ . We process BCE,CEF,and CEJ similarly,none of which enters $F$ . At EHJ,we find disappearing attributes $\mathrm{E}$ and $\mathrm{J}$ . In general,as long as one disappearing attribute has not been covered by $F$ ,we pick the node; this is why EHJ is in $F$ . The other nodes HK and HN in $F$ are chosen based on the same reasoning.

我们的结果与技术。超图$G$可以有许多最优边覆盖（所有这些边覆盖的规模必定为$\rho$）。虽然胡的分析[8]假设了一个任意的最优边覆盖，但我们会对所使用的边覆盖进行选择。在图1中，9个带圆圈的节点构成了$G$的一个规范边覆盖$F$。让我们对如何构造这个$F$给出一个非正式但直观的解释。在将图1中的树以HN为根节点后，我们将所有叶节点：BO、ABC、BD、EFG、HI、LM添加到$F$中。然后，我们自下而上处理非叶节点。在处理BCE时，我们会问：当我们在树中进一步向上移动时，哪些属性会消失？答案是B，因此B是BCE的一个“消失”属性。接着，我们会问：$F$是否已经覆盖了$\mathrm{B}$？答案是肯定的，因为存在$\mathrm{{BO}}$；因此，我们不将BCE包含在$F$中。我们以类似的方式处理BCE、CEF和CEJ，它们都不会进入$F$。在EHJ处，我们发现了消失属性$\mathrm{E}$和$\mathrm{J}$。一般来说，只要有一个消失属性尚未被$F$覆盖，我们就选择该节点；这就是为什么EHJ在$F$中。$F$中的其他节点HK和HN也是基于同样的推理被选中的。

We show that a canonical edge cover determined this way has appealing properties which fit the recursive strategy behind Hu's algorithm very well. At a high level, Hu's algorithm works by simplifying $G$ into a number of "residual" hypergraphs to be processed recursively. Interestingly, with trivial modifications (such as removing the attributes that have become irrelevant), a canonical edge cover of $G$ remains canonical on every residual hypergraph. This is the most crucial property we utilize to relate the load of the original query to those of the "residual queries" in forming up a working recurrence.

我们表明，以这种方式确定的规范边覆盖具有吸引人的性质，非常适合胡的算法背后的递归策略。从高层次来看，胡的算法通过将$G$简化为多个要递归处理的“剩余”超图来工作。有趣的是，经过一些微小的修改（例如移除那些变得无关紧要的属性），$G$的规范边覆盖在每个剩余超图上仍然是规范的。这是我们在形成一个有效的递推关系时，用于将原始查询的负载与“剩余查询”的负载相关联的最关键性质。

Our techniques also provide a simple and natural way to pinpoint a clustering $C$ that can be used to bound the algorithm’s load. Consider the canonical edge cover $F$ shown in Figure 1 (the circled nodes). For each node in $F$ ,take a "signature path" by walking up and stopping right before reaching its lowest proper ancestor in $F$ . For example,the signature path of ABC is $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$ (note: the path does not contain EHJ). Likewise, the signature path of LM is $\{ \mathrm{{LM}},\mathrm{{KL}}\}$ . The signature paths of all the nodes in $F$ together produce the clustering $C$ given in Example 1.2. Our main result (Theorem 9) states that the $Q$ -induced load of $C$ is an upper bound on the load of Hu’s algorithm. Because $C$ has a size at most $\rho$ ,the algorithm’s load is thus bounded by $O\left( {m/{p}^{1/\rho }}\right)$ .

我们的技术还提供了一种简单自然的方法来确定一个聚类$C$，该聚类可用于界定算法的负载。考虑图1中所示的规范边覆盖$F$（带圆圈的节点）。对于$F$中的每个节点，通过向上遍历并在到达其在$F$中最低的真祖先之前停止，获取一条“特征路径”。例如，ABC的特征路径是$\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$（注意：该路径不包含EHJ）。同样，LM的特征路径是$\{ \mathrm{{LM}},\mathrm{{KL}}\}$。$F$中所有节点的特征路径共同产生了示例1.2中给出的聚类$C$。我们的主要结果（定理9）表明，$C$的$Q$ - 诱导负载是胡氏算法负载的上界。因为$C$的大小至多为$\rho$，所以该算法的负载因此被$O\left( {m/{p}^{1/\rho }}\right)$所界定。

## 2 Canonical Edge Covers for Acyclic Hypergraphs

## 2 无环超图的规范边覆盖

This section is purely graph theoretic: we will establish several new properties for acyclic hypergraphs. Let $G = \left( {V,E}\right)$ be an acyclic hypergraph. A hyperedge ${e}_{1} \in  E$ is subsumed if it is a subset of another hyperedge ${e}_{2} \in  E$ ,i.e., ${e}_{1} \subseteq  {e}_{2}$ . If an attribute $X$ appears in only a single hyperedge,we call $X$ an exclusive attribute; otherwise, $X$ is non-exclusive. Unless otherwise stated,we allow $G$ to be an arbitrary acyclic hypergraph. In particular,this means that $E$ can contain two or more hyperedges with the same attributes (nonetheless, they are still distinct hyperedges) and may even have empty hyperedges (i.e.,with no attributes at all). $G$ is clean if $E$ has no subsumed edges. Some of our results will apply only to clean hypergraphs.

本节纯粹是图论内容：我们将为无环超图建立几个新的性质。设$G = \left( {V,E}\right)$为一个无环超图。如果一条超边${e}_{1} \in  E$是另一条超边${e}_{2} \in  E$的子集，即${e}_{1} \subseteq  {e}_{2}$，则称${e}_{1} \in  E$被包含。如果一个属性$X$仅出现在一条超边中，我们称$X$为排他属性；否则，$X$为非排他属性。除非另有说明，我们允许$G$为任意无环超图。特别地，这意味着$E$可以包含两条或更多具有相同属性的超边（尽管如此，它们仍然是不同的超边），甚至可能有空超边（即，根本没有属性）。如果$E$没有被包含的边，则称$G$是干净的。我们的一些结果仅适用于干净的超图。

Denote by $T$ a hyperedge tree of $G$ (the existence of $T$ is guaranteed; see Section 1.1). By rooting $T$ at an arbitrary leaf,we can regard $T$ as a rooted tree. Make all the links ${}^{2}$ of $T$ point downwards,i.e.,from parent to child. This way, $T$ becomes a directed acyclic graph.

用$T$表示$G$的超边树（$T$的存在是有保证的；见1.1节）。通过将$T$的根设为任意一个叶子节点，我们可以将$T$视为一棵有根树。使$T$的所有链接${}^{2}$都指向下，即从父节点指向子节点。这样，$T$就变成了一个有向无环图。

Now that there are two views of $T$ (i.e.,undirected and directed),we ought to be careful with terminology. By default,we will treat $T$ as a directed tree. Accordingly,a leaf of $T$ is a node with out-degree 0 , a path is a sequence of nodes where each node has a link pointing to the next node, and a subtree rooted at a node $e$ is the directed tree induced by the nodes reachable from $e$ in $T$ . Sometimes,we may revert back to the undirected view of $T$ . In that case,we use the term raw leaf for a leaf in the undirected $T$ (i.e.,a raw leaf can be a leaf or the root under the directed view)

既然$T$有两种视图（即，无向和有向），我们应该谨慎使用术语。默认情况下，我们将把$T$视为有向树。因此，$T$的叶子节点是出度为0的节点，路径是一系列节点，其中每个节点都有一个指向后续节点的链接，以节点$e$为根的子树是由$T$中从$e$可达的节点所诱导的有向树。有时，我们可能会回到$T$的无向视图。在这种情况下，我们使用“原始叶子节点”来表示无向$T$中的叶子节点（即，原始叶子节点在有向视图中可以是叶子节点或根节点）

---

<!-- Footnote -->

${}^{2}$ Remember that we refrain from saying "edges" of $T$ ; see Section 1.1.

${}^{2}$请记住，我们避免说$T$的“边”；见1.1节。

<!-- Footnote -->

---

### 2.1 Fundamental Definitions and Properties

### 2.1 基本定义和性质

Summits and Disappearing Attributes. We say that the root of $T$ is the highest node in $T$ and, in general, a node is higher (or lower) than any of its proper descendants (or ancestors). For each attribute $X \in  V$ ,we define the summit of $X$ as the highest node (a.k.a. a hyperedge) that contains $X$ . If node $e$ is the summit of $X$ ,we call $X$ a disappearing attribute in $e$ . By acyclicity’s connectedness requirement (Section 1.1), $X$ can appear only in the subtree rooted at $e$ and hence "disappears" as soon as we leave the subtree.

顶点和消失属性。我们称$T$的根节点是$T$中最高的节点，一般来说，一个节点比其任何真后代（或祖先）都高（或低）。对于每个属性$X \in  V$，我们将$X$的顶点定义为包含$X$的最高节点（也称为超边）。如果节点$e$是$X$的顶点，我们称$X$是$e$中的消失属性。根据无环性的连通性要求（1.1节），$X$只能出现在以$e$为根的子树中，因此一旦我们离开该子树，$X$就“消失”了。

Example 2.1. Let $G = \left( {V,E}\right)$ be the hypergraph in Example 1.1 whose (rooted) hypergraph tree $T$ is shown in Figure 1. The summit of $\mathrm{C}$ is node CEJ. Thus, $\mathrm{C}$ is a disappearing attribute of CEJ. Node EHJ is the summit of E and J. Hence,both E and J are disappearing attributes of EHJ.

示例2.1。设$G = \left( {V,E}\right)$为示例1.1中的超图，其（有根）超图树$T$如图1所示。$\mathrm{C}$的顶点是节点CEJ。因此，$\mathrm{C}$是CEJ的一个消失属性。节点EHJ是E和J的顶点。因此，E和J都是EHJ的消失属性。

Canonical Edge Cover. We say that a subset $S \subseteq  E$ covers an attribute $X \in  V$ if $S$ has a hyperedge containing $X$ . Recall that an optimal edge cover of $G$ is the smallest $S$ covering every attribute in $V$ . Optimal edge covers are not unique. Some are of particular importance to us; and we will identify them as "canonical". Towards a procedural definition, consider the following algorithm:

规范边覆盖。如果$S$有一条包含$X$的超边，我们就说子集$S \subseteq  E$覆盖属性$X \in  V$。回顾一下，$G$的最优边覆盖是覆盖$V$中每个属性的最小$S$。最优边覆盖不是唯一的。有些对我们特别重要；我们将把它们识别为“规范的”。为了给出一个程序性的定义，考虑以下算法：

edge-cover $\left( T\right) {/}^{ * }T$ is rooted */

边覆盖$\left( T\right) {/}^{ * }T$是有根的 */

1. ${F}_{\text{tmp }} = \varnothing$

2. obtain a reverse topological order ${e}_{1},{e}_{2},\ldots ,{e}_{\left| E\right| }$ of the nodes (i.e.,hyperedges) in $T$

2. 获得$T$中节点（即超边）的逆拓扑排序${e}_{1},{e}_{2},\ldots ,{e}_{\left| E\right| }$

3. for $i = 1$ to $\left| E\right|$ do

3. 从$i = 1$到$\left| E\right|$执行

4. if ${e}_{i}$ has a disappearing attribute not covered by ${F}_{\text{tmp }}$ then add ${e}_{i}$ to ${F}_{\text{tmp }}$

4. 如果${e}_{i}$有一个未被${F}_{\text{tmp }}$覆盖的消失属性，则将${e}_{i}$添加到${F}_{\text{tmp }}$中

5. return ${F}_{\text{tmp }}$

5. 返回${F}_{\text{tmp }}$

Lemma 1. The output of edge-cover - denoted as $F$ - is an optimal edge cover of $G$ ,and does not depend on the reverse topological order at Line 2. Furthermore,if $G$ is clean, $F$ includes all the raw leaves of $T$ .

引理1。边覆盖的输出（记为$F$）是$G$的最优边覆盖，并且不依赖于第2行的逆拓扑排序。此外，如果$G$是干净的，$F$包含$T$的所有原始叶节点。

All the missing proofs can be found in the appendix. We refer to $F$ as the canonical edge cover (CEC) of $G$ induced by $T$ . The size of $F$ is precisely the fractional edge covering number $\rho$ of $Q$ .

所有缺失的证明可以在附录中找到。我们将$F$称为由$T$诱导的$G$的规范边覆盖（CEC）。$F$的大小恰好是$Q$的分数边覆盖数$\rho$。

Example 2.2. Continuing on the previous example,consider the reverse topological order of $T$ : ABC, BD, BO, BCE, EFG, CEF, CEJ, HI, EHJ, LM, KL, HK, HN. When processing ABC, edge-cover adds it to ${F}_{\mathrm{{tmp}}}$ because $\mathrm{{ABC}}$ has a disappearing attribute $\mathrm{A}$ and yet ${F}_{\mathrm{{tmp}}} = \varnothing$ . When processing $\mathrm{{BCE}}$ , ${F}_{\text{tmp }} = \{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{BO}}\}$ . BCE has a disappearing attribute $\mathrm{B}$ ,which,however,has been covered by ${F}_{\mathrm{{tmp}}}$ . Thus,B is not added to ${F}_{\mathrm{{tmp}}}$ . The final output of the algorithm is $F = \{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{BO}},\mathrm{{EFG}}$ , HI,LM,EHJ,HK,HN\},which is the CEC of $G$ induced by $T$ .

示例2.2。继续上一个示例，考虑$T$的逆拓扑排序：ABC、BD、BO、BCE、EFG、CEF、CEJ、HI、EHJ、LM、KL、HK、HN。处理ABC时，边覆盖将其添加到${F}_{\mathrm{{tmp}}}$中，因为$\mathrm{{ABC}}$有一个消失属性$\mathrm{A}$，而${F}_{\mathrm{{tmp}}} = \varnothing$。处理$\mathrm{{BCE}}$时，${F}_{\text{tmp }} = \{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{BO}}\}$。BCE有一个消失属性$\mathrm{B}$，然而，它已经被${F}_{\mathrm{{tmp}}}$覆盖。因此，B不会被添加到${F}_{\mathrm{{tmp}}}$中。该算法的最终输出是$F = \{ \mathrm{{ABC}},\mathrm{{BD}},\mathrm{{BO}},\mathrm{{EFG}}$、HI、LM、EHJ、HK、HN}，这是由$T$诱导的$G$的CEC。

Signature Paths. Whenever $F$ includes the root of $T$ ,we can define a signature path - denoted as $\operatorname{sigpath}\left( {f,T}\right)$ - for each node (i.e.,hyperedge) $f \in  F$ . Specifically, $\operatorname{sigpath}\left( {f,T}\right)$ is a set of nodes defined as follows:

签名路径。只要$F$包含$T$的根节点，我们就可以为每个节点（即超边）$f \in  F$定义一个签名路径（记为$\operatorname{sigpath}\left( {f,T}\right)$）。具体来说，$\operatorname{sigpath}\left( {f,T}\right)$是一个节点集合，定义如下：

- If $f$ is the root of $T$ ,sigpath $\left( {f,T}\right)  = \{ f\}$ .

- 如果$f$是$T$的根节点，则签名路径$\left( {f,T}\right)  = \{ f\}$。

- Otherwise,let $\widehat{f}$ be the lowest node in $F$ that is a proper ancestor of $f$ . Then,sigpath(f,T) is the set of nodes on the path from $\widehat{f}$ to $f$ ,except $\widehat{f}$ .

- 否则，设$\widehat{f}$是$F$中$f$的最低真祖先节点。那么，签名路径(f, T)是从$\widehat{f}$到$f$的路径上的节点集合，但不包括$\widehat{f}$。

Example 2.3. Consider the set $F$ obtained in the previous example. If $f = \mathrm{{HN}}$ ,then the signature path of $f$ is $\{ \mathrm{{HN}}\}$ . If $f = \mathrm{{ABC}}$ ,then $\widehat{f} = \mathrm{{EHJ}}$ ; and the signature path of $f$ is $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$ .

示例2.3。考虑上一个示例中得到的集合$F$。如果$f = \mathrm{{HN}}$，那么$f$的签名路径是$\{ \mathrm{{HN}}\}$。如果$f = \mathrm{{ABC}}$，那么$\widehat{f} = \mathrm{{EHJ}}$；并且$f$的签名路径是$\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$。

(Clean $G$ ) Clustering,Anchor Leaf,and Anchor Attribute. Consider $G = \left( {V,E}\right)$ now as a clean hypergraph. Let $F$ be the CEC of $G$ induced by a hyperedge tree $T$ of $G$ . As $F$ contains the root and leaves of $T$ (Lemma 1), $\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$ is a clustering of $E$ . If $f$ is not the root of $T$ ,we call $\operatorname{sigpath}\left( {f,T}\right)$ a non-root cluster. ${}^{3}$

（清洁$G$）聚类、锚叶和锚属性。现在将$G = \left( {V,E}\right)$视为一个清洁超图。设$F$是由$G$的超边树$T$所诱导的$G$的连通边分量（CEC）。由于$F$包含$T$的根节点和叶节点（引理1），$\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$是$E$的一个聚类。如果$f$不是$T$的根节点，我们称$\operatorname{sigpath}\left( {f,T}\right)$为非根聚类。${}^{3}$

Let ${f}^{ \circ  }$ be a leaf node in $F$ ,and $\widehat{f}$ be the lowest proper ancestor of ${f}^{ \circ  }$ in $F$ . We call ${f}^{ \circ  }$ an anchor leaf of $T$ if two conditions are satisfied:

设${f}^{ \circ  }$是$F$中的一个叶节点，$\widehat{f}$是${f}^{ \circ  }$在$F$中的最低真祖先节点。如果满足两个条件，我们称${f}^{ \circ  }$为$T$的一个锚叶：

- $\widehat{f}$ has no non-leaf proper descendants in $F$ .

- $\widehat{f}$在$F$中没有非叶真后代节点。

- ${f}^{ \circ  }$ has an attribute ${A}^{ \circ  }$ such that ${A}^{ \circ  } \notin  \widehat{f}$ but ${A}^{ \circ  } \in  e$ for every node $e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ .

- ${f}^{ \circ  }$有一个属性${A}^{ \circ  }$，使得${A}^{ \circ  } \notin  \widehat{f}$，但对于每个节点$e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$有${A}^{ \circ  } \in  e$。

${A}^{ \circ  }$ will be referred to as an anchor attribute of ${f}^{ \circ  }$ .

${A}^{ \circ  }$将被称为${f}^{ \circ  }$的一个锚属性。

Lemma 2. If $G$ is clean, $F$ always contains an anchor leaf.

引理2。如果$G$是清洁的，$F$总是包含一个锚叶。

Example 2.4. From the $F$ constructed earlier,we obtain the clustering $C = \{ \{ \mathrm{B}0,\mathrm{{BCE}},\mathrm{{CEJ}}\}$ , $\{$ ABC, BCE, CEJ $\} ,\;\{$ BD, BCE, CEJ $\} ,\;\{$ EFG, CEF, CEJ $\} ,\;\{$ HI $\} ,\;\{$ EHJ $\} ,\;\{$ LM, KL $\} ,\;\{$ HN $\} \} .\;$ Other than $\{ \mathrm{{HN}}\}$ ,all the clusters in $C$ are non-root clusters. ABC is an anchor leaf of $T$ with an anchor attribute C. HI is another anchor leaf with an anchor attribute I. For a non-example, BD is not an anchor leaf because it does not have an attribute that exists in all the nodes in sigpath $\left( {\mathrm{{BD}},T}\right)  = \{ \mathrm{{BD}},\mathrm{{BCE}}$ , CEJ\}. Furthermore,LM is not an anchor leaf because HK,the lowest proper ancestor of LM in $F$ ,has a non-leaf proper descendant in $F$ (i.e.,EHJ).

示例2.4。从前面构造的$F$中，我们得到聚类$C = \{ \{ \mathrm{B}0,\mathrm{{BCE}},\mathrm{{CEJ}}\}$，$\{$ ABC、BCE、CEJ $\} ,\;\{$ BD、BCE、CEJ $\} ,\;\{$ EFG、CEF、CEJ $\} ,\;\{$ HI $\} ,\;\{$ EHJ $\} ,\;\{$ LM、KL $\} ,\;\{$ HN $\} \} .\;$ 除了$\{ \mathrm{{HN}}\}$之外，$C$中的所有聚类都是非根聚类。ABC是以属性C为锚属性的$T$的一个锚叶。HI是以属性I为锚属性的另一个锚叶。举个反例，BD不是锚叶，因为它没有一个存在于签名路径$\left( {\mathrm{{BD}},T}\right)  = \{ \mathrm{{BD}},\mathrm{{BCE}}$（即{CEJ}）中所有节点的属性。此外，LM不是锚叶，因为LM在$F$中的最低真祖先节点HK在$F$中有一个非叶真后代节点（即EHJ）。

### 2.2 (Clean $G$ ) Properties on Residual Hypergraphs

### 2.2 （清洁$G$）残差超图的性质

This subsection assumes $G = \left( {V,E}\right)$ to be clean. Let $T$ be a hyperedge tree of $G$ and $F$ be the CEC induced by $T$ . Fix an arbitrary anchor leaf ${f}^{ \circ  }$ of $T$ and an anchor attribute ${A}^{ \circ  }$ of ${f}^{ \circ  }$ . We will analyze how the CEC changes as $G$ is simplified based on ${f}^{ \circ  }$ and ${A}^{ \circ  }$ .

本小节假设$G = \left( {V,E}\right)$是清洁的。设$T$是$G$的一个超边树，$F$是由$T$所诱导的连通边分量（CEC）。固定$T$的任意一个锚叶${f}^{ \circ  }$和${f}^{ \circ  }$的一个锚属性${A}^{ \circ  }$。我们将分析当基于${f}^{ \circ  }$和${A}^{ \circ  }$对$G$进行简化时，连通边分量（CEC）是如何变化的。

#### 2.2.1 Simplification 1

#### 2.2.1 简化1

The first simplification is based on removing attribute ${A}^{ \circ  }$ from $G$ .

第一个简化基于从$G$中移除属性${A}^{ \circ  }$。

Residual Hypergraph. Let ${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$ be the residual hypergraph obtained by eliminating ${A}^{ \circ  }$ from $G : {V}^{\prime } = V \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ ,and ${E}^{\prime }$ collects a hyperedge ${e}^{\prime } = e \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ for every $e \in  E{.}^{4}$ We characterize the one-one correspondence between ${E}^{\prime }$ and $E$ by introducing a function $\operatorname{map}\left( e\right)  = {e}^{\prime }$ and its inverse function ${ma}{p}^{-1}\left( {e}^{\prime }\right)  = e$ . Let ${T}^{\prime }$ be the hyperedge tree of ${G}^{\prime }$ obtained by discarding ${A}^{ \circ  }$ from every node in $T$ (note: ${G}^{\prime }$ is not necessarily clean).

残差超图。设${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$是通过从$G : {V}^{\prime } = V \smallsetminus  \left\{  {A}^{ \circ  }\right\}$中消除${A}^{ \circ  }$得到的残差超图，并且对于每个$e \in  E{.}^{4}$，${E}^{\prime }$收集一条超边${e}^{\prime } = e \smallsetminus  \left\{  {A}^{ \circ  }\right\}$。我们通过引入一个函数$\operatorname{map}\left( e\right)  = {e}^{\prime }$及其反函数${ma}{p}^{-1}\left( {e}^{\prime }\right)  = e$来刻画${E}^{\prime }$和$E$之间的一一对应关系。设${T}^{\prime }$是通过从$T$的每个节点中丢弃${A}^{ \circ  }$而得到的${G}^{\prime }$的超边树（注意：${G}^{\prime }$不一定是干净的）。

Canonical Edge Cover. Define

规范边覆盖。定义

$$
{F}^{\prime } = \left\{  \begin{array}{ll} F \smallsetminus  \left\{  {f}^{ \circ  }\right\}  & \text{if }\operatorname{map}\left( {f}^{ \circ  }\right) \text{ is subsumed in }{G}^{\prime } \\  \{ \operatorname{map}\left( f\right)  \mid  f \in  F\} & \text{ otherwise } \end{array}\right.  \tag{2}
$$

Example 2.5. Continuing on the previous example,if we choose ${f}^{ \circ  } = \mathrm{{ABC}}$ with ${A}^{ \circ  } = \mathrm{C}$ and eliminate $\mathrm{C}$ from the tree $T$ in Figure 1,we obtain the hyperedge tree ${T}^{\prime }$ in Figure 2a,where the circled nodes constitute the set ${F}^{\prime }$ . Similarly,if we choose ${f}^{ \circ  } = \mathrm{{HI}}$ with ${A}^{ \circ  } = \mathrm{I}$ ,then ${T}^{\prime }$ and ${F}^{\prime }$ are as demonstrated in Figure 2b.

示例2.5。延续上一个示例，如果我们选择带有${A}^{ \circ  } = \mathrm{C}$的${f}^{ \circ  } = \mathrm{{ABC}}$，并从图1中的树$T$中消除$\mathrm{C}$，我们得到图2a中的超边树${T}^{\prime }$，其中带圆圈的节点构成集合${F}^{\prime }$。类似地，如果我们选择带有${A}^{ \circ  } = \mathrm{I}$的${f}^{ \circ  } = \mathrm{{HI}}$，那么${T}^{\prime }$和${F}^{\prime }$如图2b所示。

---

<!-- Footnote -->

${}^{3}$ If $f$ is the root of $T$ ,sigpath(f,T)contains just $f$ itself.

${}^{3}$ 如果$f$是$T$的根，sigpath(f,T)仅包含$f$本身。

${}^{4}$ If $e = \left\{  {A}^{ \circ  }\right\}  ,{E}^{\prime }$ collects ${e}^{\prime } = \varnothing$ .

${}^{4}$ 如果$e = \left\{  {A}^{ \circ  }\right\}  ,{E}^{\prime }$收集${e}^{\prime } = \varnothing$。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: HN HN HK (EHJ) CEJ CLM BCE CEF EFG Figure 2: Residual hypergraphs ${e}_{\text{small}}$ ${e}_{\text{big }}$ ${e}_{\text{big }}$ (b) ${e}_{\text{small }}$ parents ${e}_{\text{big }}$ CHK (EHJ) EJ HI CLM BE EF EFG ${e}_{\text{big }}$ ${e}_{\text{small }}$ (a) ${e}_{\text{big }}$ parents ${e}_{\text{small }}$ -->

<img src="https://cdn.noedgeai.com/0195ccba-87e9-7860-ae95-5e250889b8c6_8.jpg?x=412&y=195&w=953&h=703&r=0"/>

Figure 3: Two cases of cleansing

图3：清理的两种情况

<!-- Media -->

Lemma 3. If $G$ is clean, ${F}^{\prime }$ is the CEC of ${G}^{\prime }$ induced by ${T}^{\prime }$ . Furthermore,if $\operatorname{map}\left( {f}^{ \circ  }\right)$ is subsumed in ${G}^{\prime }$ ,then ${A}^{ \circ  }$ must be an exclusive attribute in ${f}^{ \circ  }$ .

引理3。如果$G$是干净的，${F}^{\prime }$是由${T}^{\prime }$诱导的${G}^{\prime }$的规范边覆盖（CEC）。此外，如果$\operatorname{map}\left( {f}^{ \circ  }\right)$被包含在${G}^{\prime }$中，那么${A}^{ \circ  }$必须是${f}^{ \circ  }$中的一个排他属性。

As a corollary,if $\operatorname{map}\left( {f}^{ \circ  }\right)$ is subsumed in ${G}^{\prime }$ ,then every hyperedge of $G$ ,except ${f}^{ \circ  }$ ,is directly retained in ${G}^{\prime }$ ; furthermore, $\operatorname{map}\left( {f}^{ \circ  }\right)$ is the only subsumed edge in ${G}^{\prime }$ . The next lemma gives another property of ${F}^{\prime }$ that holds no matter if $G$ is clean.

作为推论，如果$\operatorname{map}\left( {f}^{ \circ  }\right)$被包含在${G}^{\prime }$中，那么$G$的除${f}^{ \circ  }$之外的每条超边都直接保留在${G}^{\prime }$中；此外，$\operatorname{map}\left( {f}^{ \circ  }\right)$是${G}^{\prime }$中唯一被包含的边。下一个引理给出了${F}^{\prime }$的另一个性质，无论$G$是否干净都成立。

Lemma 4. If a hyperedge ${e}^{\prime }$ of ${G}^{\prime }$ is subsumed,then ${e}^{\prime } \notin  {F}^{\prime }$

引理4。如果${G}^{\prime }$的一条超边${e}^{\prime }$被包含，那么${e}^{\prime } \notin  {F}^{\prime }$

Cleansing. Even though $G$ is clean,the residual hypergraph ${G}^{\prime }$ may contain subsumed hyperedges. Next,we describe a cleansing procedure which converts ${G}^{\prime }$ into a clean hypergraph ${G}^{ * } = \left( {{V}^{\prime },{E}^{ * }}\right)$ (note that ${G}^{ * }$ has the same vertices as ${G}^{\prime }$ ) and converts ${T}^{\prime }$ into a rooted hyperedge tree ${T}^{ * }$ of ${G}^{ * }$ .

清理。即使$G$是干净的，剩余超图${G}^{\prime }$仍可能包含被包含的超边。接下来，我们描述一个清理过程，该过程将${G}^{\prime }$转换为一个干净的超图${G}^{ * } = \left( {{V}^{\prime },{E}^{ * }}\right)$（注意，${G}^{ * }$与${G}^{\prime }$具有相同的顶点），并将${T}^{\prime }$转换为${G}^{ * }$的有根超边树${T}^{ * }$。

Cleansing is simple if $\operatorname{map}\left( {f}^{ \circ  }\right)$ is subsumed in ${G}^{\prime }$ . In this case, ${G}^{ * }$ is the hypergraph obtained by removing $\operatorname{map}\left( {f}^{ \circ  }\right)$ from ${G}^{\prime }$ ,and ${T}^{ * }$ is the tree obtained by removing the leaf $\operatorname{map}\left( {f}^{ \circ  }\right)$ from ${T}^{\prime }$ . If $\operatorname{map}\left( {f}^{ \circ  }\right)$ is not subsumed,the cleansing algorithm is:

如果$\operatorname{map}\left( {f}^{ \circ  }\right)$被包含在${G}^{\prime }$中，清理过程很简单。在这种情况下，${G}^{ * }$是通过从${G}^{\prime }$中移除$\operatorname{map}\left( {f}^{ \circ  }\right)$得到的超图，而${T}^{ * }$是通过从${T}^{\prime }$中移除叶子节点$\operatorname{map}\left( {f}^{ \circ  }\right)$得到的树。如果$\operatorname{map}\left( {f}^{ \circ  }\right)$未被包含，清理算法如下：

<!-- Media -->

---

cleanse $\left( {{G}^{\prime },{T}^{\prime }}\right) / *$ condition: $\operatorname{map}\left( {f}^{ \circ  }\right)$ not subsumed $* /$

清理$\left( {{G}^{\prime },{T}^{\prime }}\right) / *$ 条件：$\operatorname{map}\left( {f}^{ \circ  }\right)$未被包含 $* /$

1. ${G}^{ * } = {G}^{\prime },{T}^{ * } = {T}^{\prime }$

2. while ${G}^{ * }$ has hyperedges ${e}_{\text{small }}$ and ${e}_{\text{big }}$ such that ${e}_{\text{small }} \subseteq  {e}_{\text{big }}$ and they are connected by

2. 当${G}^{ * }$有超边${e}_{\text{small }}$和${e}_{\text{big }}$，使得${e}_{\text{small }} \subseteq  {e}_{\text{big }}$，并且它们通过

		a link in ${T}^{ * }$ do

		${T}^{ * }$中的一条链接相连时，执行

3. remove ${e}_{\text{small }}$ from ${G}^{ * }$ and ${T}^{ * }$

3. 从 ${G}^{ * }$ 和 ${T}^{ * }$ 中移除 ${e}_{\text{small }}$

					/* ${e}_{\text{small }} \notin  {F}^{\prime }$ by Lemma 4 */

					/* 根据引理 4 移除 ${e}_{\text{small }} \notin  {F}^{\prime }$ */

4. if ${e}_{\text{big }}$ was the parent of ${e}_{\text{small }}$ in ${T}^{ * }$ then

4. 如果在 ${T}^{ * }$ 中 ${e}_{\text{big }}$ 是 ${e}_{\text{small }}$ 的父节点，则

							make ${e}_{\text{big }}$ the new parent for all the child nodes of ${e}_{\text{small }}$ ; see Figure 3a

													 让 ${e}_{\text{big }}$ 成为 ${e}_{\text{small }}$ 所有子节点的新父节点；见图 3a

					else

					否则

			make ${e}_{\text{big }}$ the new parent for the child nodes of ${e}_{\text{small }}$ ,and

					 让 ${e}_{\text{big }}$ 成为 ${e}_{\text{small }}$ 子节点的新父节点，并且

							make ${e}_{\text{big }}$ a child of the (original) parent of ${e}_{\text{small }}$ in ${T}^{ * }$ ; see Figure $3\mathrm{\;b}$

													 让 ${e}_{\text{big }}$ 成为 ${T}^{ * }$ 中 ${e}_{\text{small }}$（原始）父节点的子节点；见图 $3\mathrm{\;b}$

7. return ${G}^{ * }$ and ${T}^{ * }$

7. 返回 ${G}^{ * }$ 和 ${T}^{ * }$

---

At the end of cleansing,we always set ${F}^{ * } = {F}^{\prime }$ ,regardless of whether $\operatorname{map}\left( {f}^{ \circ  }\right)$ is subsumed.

在清理结束时，无论 $\operatorname{map}\left( {f}^{ \circ  }\right)$ 是否被包含，我们总是设置 ${F}^{ * } = {F}^{\prime }$。

<!-- figureText: HN HN HK (EHJ) BE CLM (b) After removing EF HK CHJ BE EF ( $\overset{⏜}{\mathrm{{HI}}}$ CLM AB BD EFG (a) After removing EJ -->

<img src="https://cdn.noedgeai.com/0195ccba-87e9-7860-ae95-5e250889b8c6_9.jpg?x=452&y=199&w=872&h=344&r=0"/>

Figure 4: Simplification 1

图 4：简化 1

<!-- Media -->

Lemma 5. After cleansing, ${F}^{ * }$ is the CEC of ${G}^{ * }$ induced by ${T}^{ * }$ .

引理 5. 清理后，${F}^{ * }$ 是由 ${T}^{ * }$ 诱导的 ${G}^{ * }$ 的连通导出子图（CEC，Connected Exported Component）。

Example 2.6. In Example 2.5,the residual hypergraph ${G}^{\prime }$ in Figure 2a has two subsumed hyperedges EJ and EF, each removed by an iteration of cleanse. Suppose that the first iteration sets ${e}_{\text{small }} = \mathrm{{EJ}}$ and ${e}_{\mathrm{{big}}} = \mathrm{{EHJ}}$ (this is a case of Figure 3a). Figure 4a illustrates the ${T}^{ * }$ after removing EJ. The next iteration sets ${e}_{\text{small }} = \mathrm{{EF}}$ and ${e}_{\mathrm{{big}}} = \mathrm{{EFG}}$ (a case of Figure 4b). Figure 4b illustrates the ${T}^{ * }$ after removing EF. In both Figure 4a and 4b,the circled nodes constitute the CEC of ${G}^{ * }$ induced by ${T}^{ * }$ .

例 2.6. 在例 2.5 中，图 2a 中的剩余超图 ${G}^{\prime }$ 有两条被包含的超边 EJ 和 EF，每条超边都通过一次清理迭代被移除。假设第一次迭代设置了 ${e}_{\text{small }} = \mathrm{{EJ}}$ 和 ${e}_{\mathrm{{big}}} = \mathrm{{EHJ}}$（这是图 3a 的情况）。图 4a 展示了移除 EJ 后的 ${T}^{ * }$。下一次迭代设置了 ${e}_{\text{small }} = \mathrm{{EF}}$ 和 ${e}_{\mathrm{{big}}} = \mathrm{{EFG}}$（图 4b 的情况）。图 4b 展示了移除 EF 后的 ${T}^{ * }$。在图 4a 和图 4b 中，带圆圈的节点构成了由 ${T}^{ * }$ 诱导的 ${G}^{ * }$ 的连通导出子图（CEC）。

Distinct Clusters Lemma. The next property concerns the hypergraph ${G}^{ * } = \left( {{V}^{\prime },{E}^{ * }}\right)$ after cleansing and the original hypergraph $G = \left( {V,E}\right)$ . Recall that ${T}^{ * }$ and $T$ are hyperedge trees of ${G}^{ * }$ and $G$ ,respectively. Before proceeding,the reader should recall that every hyperedge ${e}^{ * } \in  {E}^{ * }$ corresponds to a distinct hyperedge $e \in  E$ ,which is the hyperedge given by ${\operatorname{map}}^{-1}\left( {e}^{ * }\right)$ .

不同簇引理。下一个性质涉及清理后的超图 ${G}^{ * } = \left( {{V}^{\prime },{E}^{ * }}\right)$ 和原始超图 $G = \left( {V,E}\right)$。回想一下，${T}^{ * }$ 和 $T$ 分别是 ${G}^{ * }$ 和 $G$ 的超边树。在继续之前，读者应该回想一下，每条超边 ${e}^{ * } \in  {E}^{ * }$ 都对应一条不同的超边 $e \in  E$，它是由 ${\operatorname{map}}^{-1}\left( {e}^{ * }\right)$ 给出的超边。

Consider once again the CEC $F$ of $G$ ,i.e.,the original hypergraph,induced by $T$ . As mentioned in Section 2.1, $C = \{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$ is a clustering of $E$ . By the same reasoning,because ${F}^{ * }$ is the CEC of ${G}^{ * }$ induced by ${T}^{ * }$ (Lemma 5), ${C}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}^{ * }}\right)  \mid  {f}^{ * } \in  {F}^{ * }}\right\}$ must be a clustering of ${E}^{ * }$ . The following lemma draws a connection between $C$ and ${C}^{ * }$ :

再次考虑由 $T$ 诱导的原始超图 $G$ 的连通导出子图（CEC）$F$。如 2.1 节所述，$C = \{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$ 是 $E$ 的一个聚类。基于同样的推理，因为 ${F}^{ * }$ 是由 ${T}^{ * }$ 诱导的 ${G}^{ * }$ 的连通导出子图（引理 5），所以 ${C}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}^{ * }}\right)  \mid  {f}^{ * } \in  {F}^{ * }}\right\}$ 一定是 ${E}^{ * }$ 的一个聚类。以下引理建立了 $C$ 和 ${C}^{ * }$ 之间的联系：

Lemma 6 (Distinct Clusters Lemma). For any $1 \leq  k \leq  \left| {F}^{ * }\right|$ ,if $\left\{  {{e}_{1}^{ * },\ldots ,{e}_{k}^{ * }}\right\}$ is a $k$ -group of ${C}^{ * }$ ,then $\left\{  {{ma}{p}^{-1}\left( {e}_{1}^{ * }\right) ,\ldots ,{ma}{p}^{-1}\left( {e}_{k}^{ * }\right) }\right\}$ is a $k$ -group of $C$ .

引理6（不同簇引理）。对于任意$1 \leq  k \leq  \left| {F}^{ * }\right|$，若$\left\{  {{e}_{1}^{ * },\ldots ,{e}_{k}^{ * }}\right\}$是${C}^{ * }$的一个$k$ - 组，则$\left\{  {{ma}{p}^{-1}\left( {e}_{1}^{ * }\right) ,\ldots ,{ma}{p}^{-1}\left( {e}_{k}^{ * }\right) }\right\}$是$C$的一个$k$ - 组。

By definition of $k$ -group, ${e}_{1}^{ * },\ldots ,{e}_{k}^{ * }$ originate from $k$ distinct clusters in ${C}^{ * }$ . The lemma promises $k$ different clusters in $C$ each containing a distinct hyperedge in $\left\{  {{ma}{p}^{-1}\left( {e}_{1}^{ * }\right) ,\ldots ,{ma}{p}^{-1}\left( {e}_{k}^{ * }\right) }\right\}$ .

根据$k$ - 组的定义，${e}_{1}^{ * },\ldots ,{e}_{k}^{ * }$源自${C}^{ * }$中的$k$个不同簇。该引理保证了$C$中有$k$个不同的簇，每个簇都包含$\left\{  {{ma}{p}^{-1}\left( {e}_{1}^{ * }\right) ,\ldots ,{ma}{p}^{-1}\left( {e}_{k}^{ * }\right) }\right\}$中的一条不同超边。

Example 2.7. Consider the ${T}^{ * }$ (and hence ${G}^{ * }$ ) and ${F}^{ * }$ illustrated in Figure 4b. The clustering ${C}^{ * }$ is $\{ \{$ AB, BE $\} ,\{$ B0, BE $\} ,\{$ BD, BE $\} ,\{$ EFG $\} ,\{$ EFG $\} ,\{$ EHJ $\} ,\{$ LM, KL $\} ,\{$ HK $\} ,\{$ HN $\} \} .\;$ Because $\;\{$ BE, EFG, KL $\}$ is a 3-group of ${C}^{ * }$ ,Lemma 6 asserts that $\left\{  {{ma}{p}^{-1}\left( \mathrm{{BE}}\right) ,{ma}{p}^{-1}\left( \mathrm{{EFG}}\right) ,{ma}{p}^{-1}\left( \mathrm{{KL}}\right) \}  = \{ \mathrm{{BCE}},\mathrm{{EFG}},\mathrm{{KL}}}\right. \}$ must be a 3-group of the clustering $C$ in Example 2.4.

示例2.7。考虑图4b中所示的${T}^{ * }$（因此也是${G}^{ * }$）和${F}^{ * }$。聚类${C}^{ * }$为$\{ \{$AB, BE $\} ,\{$B0, BE $\} ,\{$BD, BE $\} ,\{$EFG $\} ,\{$EFG $\} ,\{$EHJ $\} ,\{$LM, KL $\} ,\{$HK $\} ,\{$HN $\} \} .\;$ 因为$\;\{$BE, EFG, KL $\}$是${C}^{ * }$的一个3 - 组，引理6断言$\left\{  {{ma}{p}^{-1}\left( \mathrm{{BE}}\right) ,{ma}{p}^{-1}\left( \mathrm{{EFG}}\right) ,{ma}{p}^{-1}\left( \mathrm{{KL}}\right) \}  = \{ \mathrm{{BCE}},\mathrm{{EFG}},\mathrm{{KL}}}\right. \}$必定是示例2.4中聚类$C$的一个3 - 组。

#### 2.2.2 Simplification 2

#### 2.2.2 简化2

The second simplification decomposes $G$ into multiple hypergraphs based on sigpath $\left( {{f}^{ \circ  },T}\right)$ .

第二种简化方法基于信号路径$\left( {{f}^{ \circ  },T}\right)$将$G$分解为多个超图。

Decomposition. Define $Z$ to be the set of nodes $z$ in $T$ satisfying: $z$ is not in sigpath $\left( {{f}^{ \circ  },T}\right)$ but the parent of $z$ is. For each $z \in  Z$ ,define a rooted tree ${T}_{z}^{ * }$ as follows:

分解。定义$Z$为$T$中满足以下条件的节点$z$的集合：$z$不在信号路径$\left( {{f}^{ \circ  },T}\right)$中，但$z$的父节点在。对于每个$z \in  Z$，按如下方式定义一棵有根树${T}_{z}^{ * }$：

- The root of ${T}_{z}^{ * }$ is the parent of $z$ in $T$ .

- ${T}_{z}^{ * }$的根节点是$z$在$T$中的父节点。

- The root of ${T}_{z}^{ * }$ has only one child in ${T}_{z}^{ * }$ ,which is $z$ .

- ${T}_{z}^{ * }$的根节点在${T}_{z}^{ * }$中只有一个子节点，即$z$。

<!-- Media -->

<!-- figureText: HN HN HK CEF (BCE) (BCE) CHJ EFG BO AI CLM (b) ${T}_{\mathrm{{CEF}}}^{ * }$ (c) ${T}_{\mathrm{B}0}^{ * }$ (d) ${T}_{\mathrm{{BD}}}^{ * }$ (e) ${\bar{T}}^{ * }$ HK EHJ CEJ AI LM CEJ BCE CEF BO ABC BD EFG (a) The dotted path is sigpath(ABC,T) -->

<img src="https://cdn.noedgeai.com/0195ccba-87e9-7860-ae95-5e250889b8c6_10.jpg?x=262&y=201&w=1257&h=405&r=0"/>

Figure 5: Decomposition

图5：分解

<!-- Media -->

- The subtree rooted at $z$ in ${T}_{z}^{ * }$ is the same as the subtree rooted at $z$ in $T$ .

- ${T}_{z}^{ * }$中以$z$为根的子树与$T$中以$z$为根的子树相同。

Separately,define ${\bar{T}}^{ * }$ as the rooted tree obtained by removing from $T$ the subtree rooted at the highest node in sigpath $\left( {{f}^{ \circ  },T}\right)$ .

另外，定义${\bar{T}}^{ * }$为从$T$中移除信号路径$\left( {{f}^{ \circ  },T}\right)$中最高节点为根的子树后得到的有根树。

From each ${T}_{z}^{ * }$ ,generate a hypergraph ${G}_{z}^{ * } = \left( {{V}_{z}^{ * },{E}_{z}^{ * }}\right)$ . Specifically, ${E}_{z}^{ * }$ includes all and only the nodes (each being a hyperedge) in ${T}_{z}^{ * }$ ,and ${V}_{z}^{ * }$ is the set of attributes appearing in at least one hyperedge in ${E}_{z}^{ * }$ . Likewise,from ${\bar{T}}^{ * }$ ,generate a hypergraph ${\bar{G}}^{ * } = \left( {{\bar{V}}^{ * },{\bar{E}}^{ * }}\right)$ where ${\bar{E}}^{ * }$ includes all and only the nodes in ${\bar{T}}^{ * }$ ,and ${\bar{V}}^{ * }$ is the set of attributes appearing in at least one hyperedge in ${\bar{E}}^{ * }$ .

从每个${T}_{z}^{ * }$生成一个超图${G}_{z}^{ * } = \left( {{V}_{z}^{ * },{E}_{z}^{ * }}\right)$。具体而言，${E}_{z}^{ * }$包含且仅包含${T}_{z}^{ * }$中的所有节点（每个节点都是一条超边），并且${V}_{z}^{ * }$是出现在${E}_{z}^{ * }$中至少一条超边中的属性集合。同样，从${\bar{T}}^{ * }$生成一个超图${\bar{G}}^{ * } = \left( {{\bar{V}}^{ * },{\bar{E}}^{ * }}\right)$，其中${\bar{E}}^{ * }$包含且仅包含${\bar{T}}^{ * }$中的所有节点，并且${\bar{V}}^{ * }$是出现在${\bar{E}}^{ * }$中至少一条超边中的属性集合。

Because $G$ is clean,so must be all the generated hypergraphs. Furthermore,each of them has fewer edges than $G.{}^{5}$ For each $z \in  Z,{T}_{z}^{ * }$ is a hyperedge tree of ${G}_{z}^{ * }$ ; similarly, ${\bar{T}}^{ * }$ is a hyperedge tree of ${\bar{G}}^{ * }$ .

因为$G$是干净的，所以所有生成的超图也必须是干净的。此外，它们中的每一个的边数都比$G.{}^{5}$少。因为每个$z \in  Z,{T}_{z}^{ * }$是${G}_{z}^{ * }$的超边树；类似地，${\bar{T}}^{ * }$是${\bar{G}}^{ * }$的超边树。

Example 2.8. In our running example, ${f}^{ \circ  } = \mathrm{{ABC}}$ ,whose signature path is $\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)  =$ $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$ ; see Figure 5a. $Z = \{ \mathrm{{BO}},\mathrm{{BD}},\mathrm{{CEF}}\}$ . Figure 5b,5c,and 5d illustrate ${T}_{z}^{ * }$ for $z = \mathrm{{CEF}}$ , BO,and BD,respectively. Figure 5e gives ${\bar{T}}_{z}^{ * }$ .

示例2.8。在我们的运行示例中，${f}^{ \circ  } = \mathrm{{ABC}}$，其签名路径为$\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)  =$ $\{ \mathrm{{ABC}},\mathrm{{BCE}},\mathrm{{CEJ}}\}$；见图5a。$Z = \{ \mathrm{{BO}},\mathrm{{BD}},\mathrm{{CEF}}\}$。图5b、5c和5d分别展示了${T}_{z}^{ * }$对于$z = \mathrm{{CEF}}$、BO和BD的情况。图5e给出了${\bar{T}}_{z}^{ * }$。

Canonical Edge Covers. Recall that $F$ is the CEC of $G$ induced by $T$ . Next,we derive the CECs of the hypergraphs generated from the decomposition. For each $z \in  Z$ ,define

规范边覆盖。回顾一下，$F$是由$T$诱导的$G$的规范边覆盖（CEC）。接下来，我们推导从分解中生成的超图的规范边覆盖。对于每个$z \in  Z$，定义

$$
{F}_{z}^{ * } = \{ \text{ parent of }z\}  \cup  \left( {F \cap  {E}_{z}^{ * }}\right) . \tag{3}
$$

Also, define

此外，定义

$$
{\bar{F}}^{ * } = F \cap  {\bar{E}}^{ * }. \tag{4}
$$

Lemma 7. For each node $z \in  Z,{F}_{z}^{ * }$ is the CEC of ${G}_{z}^{ * }$ induced by ${T}_{z}^{ * }$ . Furthermore, ${\bar{F}}^{ * }$ is the CEC of ${\bar{G}}^{ * }$ induced by ${\bar{T}}^{ * }$ .

引理7。对于每个节点$z \in  Z,{F}_{z}^{ * }$是由${T}_{z}^{ * }$诱导的${G}_{z}^{ * }$的规范边覆盖。此外，${\bar{F}}^{ * }$是由${\bar{T}}^{ * }$诱导的${\bar{G}}^{ * }$的规范边覆盖。

Example 2.9. We have circled the nodes in ${F}_{z}^{ * }$ in Figure 5b,5c,and 5d for $z =$ CEF,BO,and BD, respectively. Similarly,the circled nodes in Figure 5e constitute ${\bar{F}}_{z}^{ * }$ .

示例2.9。我们在图5b、5c和5d中分别圈出了${F}_{z}^{ * }$中对应于$z =$、CEF、BO和BD的节点。类似地，图5e中圈出的节点构成了${\bar{F}}_{z}^{ * }$。

Distinct Clusters Lemma 2. We close the section with a property resembling Lemma 6.

不同簇引理2。我们以一个类似于引理6的性质来结束本节。

Consider any $z \in  Z$ . Because ${G}_{z}^{ * } = \left( {{V}_{z}^{ * },{E}_{z}^{ * }}\right)$ is clean and ${F}_{z}^{ * }$ is the CEC of ${G}_{z}^{ * }$ induced by ${T}_{z}^{ * },{C}_{z}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}_{z}^{ * }}\right)  \mid  {f}^{ * } \in  {F}_{z}^{ * }}\right\}$ is a clustering of ${E}_{z}^{ * }$ . Similarly,regarding ${\bar{G}}^{ * } = \left( {{\bar{V}}^{ * },{\bar{E}}^{ * }}\right)$ , ${\bar{C}}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{\bar{T}}^{ * }}\right)  \mid  {f}^{ * } \in  {\bar{F}}^{ * }}\right\}$ is a clustering of ${\bar{E}}^{ * }$ .

考虑任意$z \in  Z$。因为${G}_{z}^{ * } = \left( {{V}_{z}^{ * },{E}_{z}^{ * }}\right)$是干净的，并且由${T}_{z}^{ * },{C}_{z}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}_{z}^{ * }}\right)  \mid  {f}^{ * } \in  {F}_{z}^{ * }}\right\}$诱导的${G}_{z}^{ * }$的连通分量聚类（CEC，Connected-Component Clustering）${F}_{z}^{ * }$是${E}_{z}^{ * }$的一个聚类。类似地，对于${\bar{G}}^{ * } = \left( {{\bar{V}}^{ * },{\bar{E}}^{ * }}\right)$，${\bar{C}}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{\bar{T}}^{ * }}\right)  \mid  {f}^{ * } \in  {\bar{F}}^{ * }}\right\}$是${\bar{E}}^{ * }$的一个聚类。

Define a super- $k$ -group to be a set of hyperedges $K = \left\{  {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right\}$ satisfying:

定义一个超$k$ - 组为满足以下条件的超边集合$K = \left\{  {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right\}$：

---

<!-- Footnote -->

${}^{5}$ Because ${f}^{ \circ  }$ does not appear in any of the generated hypergraphs.

${}^{5}$ 因为${f}^{ \circ  }$不出现在任何生成的超图中。

<!-- Footnote -->

---

- Each ${e}_{i},i \in  \left\lbrack  k\right\rbrack$ ,is taken from a cluster of ${\bar{C}}^{ * }$ or a non-root cluster ${}^{6}$ of ${C}_{z}^{ * }$ for some $z \in  Z$ .

- 每个${e}_{i},i \in  \left\lbrack  k\right\rbrack$取自${\bar{C}}^{ * }$的一个聚类，或者对于某个$z \in  Z$取自${C}_{z}^{ * }$的一个非根聚类${}^{6}$。

- No two hyperedges in $K$ are taken from the same cluster.

- $K$中没有两条超边取自同一个聚类。

Before delving into the next lemma,the reader should recall that $\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$ is a clustering of $E$ .

在深入研究下一个引理之前，读者应该回想一下$\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$是$E$的一个聚类。

Lemma 8 (Distinct Clusters Lemma 2). If $\left\{  {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right\}$ is a super- $k$ -group,then $\left\{  {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right\}$ must be a $k$ -group of the clustering $\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$ .

引理8（不同聚类引理2）。如果$\left\{  {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right\}$是一个超$k$ - 组，那么$\left\{  {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right\}$一定是聚类$\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$的一个$k$ - 组。

Example 2.10. In Figure 5, ${C}_{\mathrm{{CEF}}}^{ * } = \{ \{ \mathrm{{EFG}},\mathrm{{CEF}}\} ,\{ \mathrm{{CEJ}}\} \} ,{C}_{\mathrm{B}0}^{ * } = \{ \{ \mathrm{B}0\} ,\{ \mathrm{{BCE}}\} \} ,{C}_{\mathrm{{BD}}}^{ * } =$ $\{ \{ \mathrm{{BD}}\} ,\{ \mathrm{{BCE}}\} \} ,{\bar{C}}^{ * } = \{ \{ \mathrm{{HI}}\} ,\{ \mathrm{{EHJ}}\} ,\{ \mathrm{{HK}}\} ,\{ \mathrm{{HN}}\} ,\{ \mathrm{{LM}},\mathrm{{KL}}\} \} .\;$ A super-4-group is $\{ \mathrm{{CEF}},\mathrm{{BO}},\mathrm{{BD}},\mathrm{{KL}}\}$ . Lemma 8 assures us that $\{ \mathrm{{CEF}},\mathrm{{BO}},\mathrm{{BD}},\mathrm{{KL}}\}$ must be a 4-group in the clustering $C$ given in Example 2.4.

例2.10。在图5中，${C}_{\mathrm{{CEF}}}^{ * } = \{ \{ \mathrm{{EFG}},\mathrm{{CEF}}\} ,\{ \mathrm{{CEJ}}\} \} ,{C}_{\mathrm{B}0}^{ * } = \{ \{ \mathrm{B}0\} ,\{ \mathrm{{BCE}}\} \} ,{C}_{\mathrm{{BD}}}^{ * } =$ $\{ \{ \mathrm{{BD}}\} ,\{ \mathrm{{BCE}}\} \} ,{\bar{C}}^{ * } = \{ \{ \mathrm{{HI}}\} ,\{ \mathrm{{EHJ}}\} ,\{ \mathrm{{HK}}\} ,\{ \mathrm{{HN}}\} ,\{ \mathrm{{LM}},\mathrm{{KL}}\} \} .\;$ 一个超4 - 组是$\{ \mathrm{{CEF}},\mathrm{{BO}},\mathrm{{BD}},\mathrm{{KL}}\}$。引理8向我们保证，$\{ \mathrm{{CEF}},\mathrm{{BO}},\mathrm{{BD}},\mathrm{{KL}}\}$一定是例2.4中给出的聚类$C$中的一个4 - 组。

## 3 An MPC Algorithm

## 3 一种多方计算（MPC，Multi-Party Computation）算法

The rest of the paper will apply the theory of CECs to solve acyclic queries in the MPC model. We will describe a variant of $\mathrm{{Hu}}$ ’s algorithm [8] in this section ${}^{7}$ and present our analysis in the next section. Denote by $Q$ the acyclic query to be answered. Let $G = \left( {V,E}\right)$ be the hypergraph of $Q$ . We assume $G$ to be clean; otherwise, $Q$ can be converted to a clean query having the same result with load $O\left( {m/p}\right)$ [8]. We will also assume that $Q$ has at least two relations; otherwise,the query is trivial and requires no communication.

本文的其余部分将应用连通边覆盖（CECs）理论来解决消息传递计算（MPC）模型中的无环查询问题。我们将在本节${}^{7}$中描述$\mathrm{{Hu}}$算法[8]的一个变体，并在下一节中给出我们的分析。用$Q$表示待回答的无环查询。设$G = \left( {V,E}\right)$为$Q$的超图。我们假设$G$是干净的；否则，$Q$可以转换为一个具有相同结果且负载为$O\left( {m/p}\right)$的干净查询[8]。我们还假设$Q$至少有两个关系；否则，该查询很简单，不需要通信。

### 3.1 Configurations

### 3.1 配置

Let $T$ be a hyperedge tree of $G$ and $F$ be the CEC of $G$ induced by $T$ . The size of $F$ is precisely $\rho$ ,the fractional edge covering number of $Q$ (Section 1.2). As explained in Section 2.1,when $G$ is clean,

设$T$为$G$的超边树，$F$为由$T$诱导的$G$的连通边覆盖（CEC）。$F$的大小恰好为$\rho$，即$Q$的分数边覆盖数（第1.2节）。如第2.1节所述，当$G$是干净的时

$$
C = \{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}  \tag{5}
$$

is a clustering of $E$ . Let ${f}^{ \circ  }$ be an anchor leaf of $T$ and ${A}^{ \circ  }$ an anchor attribute of ${f}^{ \circ  }$ (Section 2.1); remember that ${A}^{ \circ  }$ appears in all the hyperedges of sigpath $\left( {{f}^{ \circ  },T}\right)$ . Define

是$E$的一个聚类。设${f}^{ \circ  }$为$T$的一个锚叶节点，${A}^{ \circ  }$为${f}^{ \circ  }$的一个锚属性（第2.1节）；请记住，${A}^{ \circ  }$出现在签名路径$\left( {{f}^{ \circ  },T}\right)$的所有超边中。定义

$$
L = \text{the}Q\text{-induced load of}C\text{.} \tag{6}
$$

The reader can review Equation (1) for the definition of " $Q$ -induced load".

读者可以回顾方程（1）中“$Q$ -诱导负载”的定义。

For each hyperedge $e \in  E$ ,as before $R\left( e\right)$ denotes the relation in $Q$ corresponding to $e$ . Fix a value $x \in$ dom. Given an $e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ ,we define the ${A}^{ \circ  }$ -frequency of $x$ in $R\left( e\right)$ as the number of tuples $\mathbf{u} \in  R\left( e\right)$ such that $\mathbf{u}\left( {A}^{ \circ  }\right)  = x$ . Further define the signature-path ${A}^{ \circ  }$ -frequency of $x$ as the sum of its ${A}^{ \circ  }$ -frequencies in the $R\left( e\right)$ of all $e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ . A value $x \in  \mathbf{{dom}}$ is

对于每个超边$e \in  E$，和之前一样，$R\left( e\right)$表示$Q$中与$e$对应的关系。固定一个值$x \in$∈dom。给定一个$e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$，我们将$x$在$R\left( e\right)$中的${A}^{ \circ  }$ -频率定义为满足$\mathbf{u}\left( {A}^{ \circ  }\right)  = x$的元组$\mathbf{u} \in  R\left( e\right)$的数量。进一步将$x$的签名路径${A}^{ \circ  }$ -频率定义为它在所有$e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$对应的$R\left( e\right)$中的${A}^{ \circ  }$ -频率之和。一个值$x \in  \mathbf{{dom}}$是

- heavy,if its signature-path ${A}^{ \circ  }$ -frequency is at least $L$ ;

- 重的，如果它的签名路径${A}^{ \circ  }$ -频率至少为$L$；

- light, otherwise.

- 轻的，否则。

---

<!-- Footnote -->

${}^{6}$ Namely, ${e}_{i}$ cannot be the root of ${T}_{z}^{ * }$ .

${}^{6}$ 即，${e}_{i}$不能是${T}_{z}^{ * }$的根节点。

${}^{7}$ Our algorithm follows Hu’s ideas [8] but differs in certain details. For example,Hu’s algorithm takes an arbitrary optimal edge cover of $G$ as the input,while we insist on working with a CEC.

${}^{7}$ 我们的算法遵循胡（Hu）的思路[8]，但在某些细节上有所不同。例如，胡的算法将$G$的任意最优边覆盖作为输入，而我们坚持使用连通边覆盖（CEC）。

<!-- Footnote -->

---

Divide dom into disjoint intervals such that the light values in each interval have a total signature-path ${A}^{ \circ  }$ -frequency of $\Theta \left( L\right)$ . We will refer to those intervals as the light intervals of ${A}^{ \circ  }$ . The total number of heavy values and light intervals is at most

将dom划分为不相交的区间，使得每个区间中的轻值的总签名路径${A}^{ \circ  }$ -频率为$\Theta \left( L\right)$。我们将这些区间称为${A}^{ \circ  }$的轻区间。重值和轻区间的总数最多为

$$
\mathop{\sum }\limits_{{e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right) }}\frac{\left| R\left( e\right) \right| }{L} = O\left( {\mathop{\max }\limits_{{e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right) }}\frac{\left| R\left( e\right) \right| }{L}}\right)  = O\left( \frac{\max \left( {1,Q}\right) \text{-product of }C}{L}\right)  = O\left( p\right)  \tag{7}
$$

where the first equality used the fact that sigpath $\left( {{f}^{ \circ  },T}\right)$ has $O\left( 1\right)$ edges and the second equality applied the definition of max(k,Q)-product (see Section 1.3).

其中第一个等式利用了签名路径$\left( {{f}^{ \circ  },T}\right)$有$O\left( 1\right)$条边这一事实，第二个等式应用了最大(k,Q) -积的定义（见第1.3节）。

A configuration $\eta$ is either a heavy value or a light interval. Equation (7) implies that the number of configurations is $O\left( p\right)$ . For each hypergraph $e \in  E$ ,define a relation $R\left( {e,\eta }\right)$ as follows:

一个配置$\eta$要么是一个重值，要么是一个轻区间。方程（7）表明配置的数量为$O\left( p\right)$。对于每个超图$e \in  E$，定义一个关系$R\left( {e,\eta }\right)$如下：

- if $\eta$ is a heavy value, $R\left( {e,\eta }\right)$ includes all and only the tuples $\mathbf{u} \in  R\left( e\right)$ satisfying $\mathbf{u}\left( {A}^{ \circ  }\right)  = \eta$ ;

- 如果 $\eta$ 是一个重值（heavy value），则 $R\left( {e,\eta }\right)$ 包含且仅包含满足 $\mathbf{u}\left( {A}^{ \circ  }\right)  = \eta$ 的所有元组 $\mathbf{u} \in  R\left( e\right)$；

- if $\eta$ is a light interval, $R\left( {e,\eta }\right)$ includes all and only the tuples $\mathbf{u} \in  R\left( e\right)$ where $\mathbf{u}\left( {A}^{ \circ  }\right)$ is a light value in $\eta$ .

- 如果 $\eta$ 是一个轻区间（light interval），则 $R\left( {e,\eta }\right)$ 包含且仅包含 $\mathbf{u}\left( {A}^{ \circ  }\right)$ 是 $\eta$ 中的轻值（light value）的所有元组 $\mathbf{u} \in  R\left( e\right)$。

Note that $R\left( {e,\eta }\right)  = R\left( e\right)$ if ${A}^{ \circ  } \notin  e$ . Let ${Q}_{\eta }$ be the query defined by $\{ R\left( {e,\eta }\right)  \mid  e \in  E\}$ . Our objective is to compute $\operatorname{Join}\left( {Q}_{\eta }\right)$ for all $\eta$ in parallel. The final result $\operatorname{Join}\left( Q\right)$ is simply $\mathop{\bigcup }\limits_{\eta }\operatorname{Join}\left( {Q}_{\eta }\right)$ .

注意，如果 ${A}^{ \circ  } \notin  e$，则 $R\left( {e,\eta }\right)  = R\left( e\right)$。设 ${Q}_{\eta }$ 是由 $\{ R\left( {e,\eta }\right)  \mid  e \in  E\}$ 定义的查询。我们的目标是并行计算所有 $\eta$ 的 $\operatorname{Join}\left( {Q}_{\eta }\right)$。最终结果 $\operatorname{Join}\left( Q\right)$ 就是 $\mathop{\bigcup }\limits_{\eta }\operatorname{Join}\left( {Q}_{\eta }\right)$。

The rest of the section will explain how to solve $\operatorname{Join}\left( {Q}_{\eta }\right)$ for an arbitrary $\eta$ . We allocate

本节的其余部分将解释如何为任意的 $\eta$ 求解 $\operatorname{Join}\left( {Q}_{\eta }\right)$。我们分配

$$
{p}_{\eta } = \Theta \left( {1 + \mathop{\max }\limits_{{k = 1}}^{\left| F\right| }\frac{{P}_{k}\left( {{Q}_{\eta },C}\right) }{{L}^{k}}}\right)  \tag{8}
$$

machines for this purpose,where ${P}_{k}\left( {{Q}_{\eta },C}\right)$ is the $\max \left( {k,{Q}_{\eta }}\right)$ -product of $C$ .

为此目的的机器，其中 ${P}_{k}\left( {{Q}_{\eta },C}\right)$ 是 $C$ 的 $\max \left( {k,{Q}_{\eta }}\right)$ -积（$\max \left( {k,{Q}_{\eta }}\right)$ -product）。

### 3.2 Solving ${Q}_{\eta }$ When $\eta$ is a Heavy Value

### 3.2 当 $\eta$ 是重值（heavy value）时求解 ${Q}_{\eta }$

Define the residual hypergraph ${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$ after removing ${A}^{ \circ  }$ ,and also functions $\operatorname{map}\left( \text{.}\right) {and}$ ${ma}{p}^{-1}\left( \text{.}\right) {asinSection2.2.1}$ . We compute $\operatorname{Join}\left( {Q}_{\eta }\right)$ in five steps.

定义移除 ${A}^{ \circ  }$ 后的剩余超图 ${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$，以及函数 $\operatorname{map}\left( \text{.}\right) {and}$ ${ma}{p}^{-1}\left( \text{.}\right) {asinSection2.2.1}$。我们分五步计算 $\operatorname{Join}\left( {Q}_{\eta }\right)$。

Step 1. Send the tuples of $R\left( {e,\eta }\right)$ ,for all $e \in  E$ ,to the ${p}_{\eta }$ allocated machines such that each machine receives $\Theta \left( {\frac{1}{{p}_{\eta }}\mathop{\sum }\limits_{{e \in  E}}\left| {R\left( {e,\eta }\right) }\right| }\right)$ tuples.

步骤 1. 将所有 $e \in  E$ 的 $R\left( {e,\eta }\right)$ 元组发送到分配的 ${p}_{\eta }$ 台机器，使得每台机器接收 $\Theta \left( {\frac{1}{{p}_{\eta }}\mathop{\sum }\limits_{{e \in  E}}\left| {R\left( {e,\eta }\right) }\right| }\right)$ 个元组。

Step 2. For each $e \in  E$ ,convert $R\left( {e,\eta }\right)$ to ${R}^{ * }\left( {{e}^{\prime },\eta }\right)$ where ${e}^{\prime } = \operatorname{map}\left( e\right)  = e \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ . Specifically, ${R}^{ * }\left( {{e}^{\prime },\eta }\right)$ is a copy of $R\left( {e,\eta }\right)$ but with ${A}^{ \circ  }$ discarded,or formally, ${R}^{ * }\left( {{e}^{\prime },\eta }\right)  = \left\{  {\mathbf{u}\left\lbrack  {e}^{\prime }\right\rbrack   \mid  }\right.$ tuple $\mathbf{u} \in$ $R\left( {e,\eta }\right) \}$ . No communication occurs as each machine simply discards ${A}^{ \circ  }$ from every tuple $\mathbf{u} \in  R\left( {e,\eta }\right)$ in the local storage.

步骤 2. 对于每个 $e \in  E$，将 $R\left( {e,\eta }\right)$ 转换为 ${R}^{ * }\left( {{e}^{\prime },\eta }\right)$，其中 ${e}^{\prime } = \operatorname{map}\left( e\right)  = e \smallsetminus  \left\{  {A}^{ \circ  }\right\}$。具体来说，${R}^{ * }\left( {{e}^{\prime },\eta }\right)$ 是 $R\left( {e,\eta }\right)$ 的一个副本，但丢弃了 ${A}^{ \circ  }$，或者形式上，${R}^{ * }\left( {{e}^{\prime },\eta }\right)  = \left\{  {\mathbf{u}\left\lbrack  {e}^{\prime }\right\rbrack   \mid  }\right.$ 元组 $\mathbf{u} \in$ $R\left( {e,\eta }\right) \}$。由于每台机器只是从本地存储的每个元组 $\mathbf{u} \in  R\left( {e,\eta }\right)$ 中丢弃 ${A}^{ \circ  }$，因此不会发生通信。

Step 3. Cleanse ${G}^{\prime }$ into ${G}^{ * } = \left( {{V}^{\prime },{E}^{ * }}\right)$ . As explained in Section 2.2.1,this may or may not require calling algorithm cleanse. If called,cleanse identifies in each iteration two hyperedges ${e}_{\text{small }}$ and ${e}_{\text{big }}$ in the current ${G}^{ * }$ and removes ${e}_{\text{small }}$ . Accordingly,we perform a semi-join between ${R}^{ * }\left( {{e}_{\text{small }},\eta }\right)$ and ${R}^{ * }\left( {{e}_{\mathrm{{big}}},\eta }\right)$ ,which removes every tuple $\mathbf{u}$ from ${R}^{ * }\left( {{e}_{\mathrm{{big}}},\eta }\right)$ with the property that $\mathbf{u}\left\lbrack  {e}_{\text{small }}\right\rbrack$ is absent from ${R}^{ * }\left( {{e}_{\text{small }},\eta }\right) .{R}^{ * }\left( {{e}_{\text{small }},\eta }\right)$ is discarded after the semi-join.

步骤3. 将${G}^{\prime }$净化为${G}^{ * } = \left( {{V}^{\prime },{E}^{ * }}\right)$。如2.2.1节所述，这可能需要也可能不需要调用净化算法。如果调用，净化算法会在每次迭代中识别当前${G}^{ * }$中的两条超边${e}_{\text{small }}$和${e}_{\text{big }}$，并移除${e}_{\text{small }}$。因此，我们在${R}^{ * }\left( {{e}_{\text{small }},\eta }\right)$和${R}^{ * }\left( {{e}_{\mathrm{{big}}},\eta }\right)$之间执行半连接，这会从${R}^{ * }\left( {{e}_{\mathrm{{big}}},\eta }\right)$中移除每个满足$\mathbf{u}\left\lbrack  {e}_{\text{small }}\right\rbrack$不在${R}^{ * }\left( {{e}_{\text{small }},\eta }\right) .{R}^{ * }\left( {{e}_{\text{small }},\eta }\right)$中的元组$\mathbf{u}$，这些元组在半连接后会被丢弃。

Step 4. Let ${Q}_{\eta }^{ * }$ be the query defined by the relation set $\left\{  {{R}^{ * }\left( {{e}^{ * },\eta }\right)  \mid  {e}^{ * } \in  {E}^{ * }}\right\}$ . Compute $\operatorname{Join}\left( {Q}_{\eta }^{ * }\right)$ using ${p}_{\eta }$ machines recursively. Note that the number of participating attributes has decreased by 1 for the recursion.

步骤4. 令${Q}_{\eta }^{ * }$为由关系集$\left\{  {{R}^{ * }\left( {{e}^{ * },\eta }\right)  \mid  {e}^{ * } \in  {E}^{ * }}\right\}$定义的查询。使用${p}_{\eta }$台机器递归地计算$\operatorname{Join}\left( {Q}_{\eta }^{ * }\right)$。注意，递归时参与的属性数量减少了1。

Step 5. We output $\operatorname{Join}\left( {Q}_{\eta }\right)$ by augmenting each tuple $\mathbf{u} \in  \operatorname{Join}\left( {Q}_{\eta }^{ * }\right)$ with $\mathbf{u}\left( {A}^{ \circ  }\right)  = \eta$ . No communication is needed.

步骤5. 我们通过为每个元组$\mathbf{u} \in  \operatorname{Join}\left( {Q}_{\eta }^{ * }\right)$添加$\mathbf{u}\left( {A}^{ \circ  }\right)  = \eta$来输出$\operatorname{Join}\left( {Q}_{\eta }\right)$。无需通信。

### 3.3 Solving ${Q}_{\eta }$ When $\eta$ is a Light Interval

### 3.3 当$\eta$是轻区间时求解${Q}_{\eta }$

Define $Z,{G}_{z}^{ * } = \left( {{V}_{z}^{ * },{E}_{z}^{ * }}\right)$ (for each $z \in  Z$ ), ${C}_{z}^{ * },{\bar{G}}^{ * } = \left( {{\bar{V}}^{ * },{\bar{E}}^{ * }}\right)$ ,and ${\bar{C}}^{ * }$ all in the way described in Section 2.2.2. We compute $\operatorname{Join}\left( {Q}_{\eta }\right)$ in four steps.

按照2.2.2节中描述的方式定义$Z,{G}_{z}^{ * } = \left( {{V}_{z}^{ * },{E}_{z}^{ * }}\right)$（对于每个$z \in  Z$）、${C}_{z}^{ * },{\bar{G}}^{ * } = \left( {{\bar{V}}^{ * },{\bar{E}}^{ * }}\right)$和${\bar{C}}^{ * }$。我们分四个步骤计算$\operatorname{Join}\left( {Q}_{\eta }\right)$。

Step 1. Same as Step 1 of the algorithm in Section 3.2.

步骤1. 与3.2节中算法的步骤1相同。

Step 2. For each $e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ ,broadcast $R\left( {e,\eta }\right)$ to all ${p}_{\eta }$ machines. By definition of light interval,the size of $R\left( {e,\eta }\right)$ is at most $L$ .

步骤2. 对于每个$e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$，将$R\left( {e,\eta }\right)$广播到所有${p}_{\eta }$台机器。根据轻区间的定义，$R\left( {e,\eta }\right)$的大小至多为$L$。

Step 3. For each $z \in  Z$ ,define a query ${Q}_{\eta ,z}^{ * } = \left\{  {R\left( {e,\eta }\right)  \mid  e \in  {E}_{z}^{ * }}\right\}$ . Similarly,for ${\bar{G}}^{ * }$ ,define a query ${\bar{Q}}_{\eta }^{ * } = \left\{  {R\left( {e,\eta }\right)  \mid  e \in  {\bar{E}}^{ * }}\right\}$ . Next,we compute the cartesian product of $\operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$ and the $\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right)$ of all the $z \in  Z$ - namely $\left( {{ \times  }_{z \in  Z}\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right) }\right)  \times  \operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$ - using ${p}_{\eta }$ machines. Towards that purpose,define for each $z \in  Z$

步骤3. 对于每个 $z \in  Z$，定义一个查询 ${Q}_{\eta ,z}^{ * } = \left\{  {R\left( {e,\eta }\right)  \mid  e \in  {E}_{z}^{ * }}\right\}$。类似地，对于 ${\bar{G}}^{ * }$，定义一个查询 ${\bar{Q}}_{\eta }^{ * } = \left\{  {R\left( {e,\eta }\right)  \mid  e \in  {\bar{E}}^{ * }}\right\}$。接下来，我们使用 ${p}_{\eta }$ 台机器计算 $\operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$ 与所有 $z \in  Z$ 的 $\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right)$（即 $\left( {{ \times  }_{z \in  Z}\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right) }\right)  \times  \operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$）的笛卡尔积。为此，为每个 $z \in  Z$ 进行定义

$$
{p}_{\eta ,z} = \Theta \left( {1 + \mathop{\max }\limits_{{k = 1}}^{\left| {F}_{z}^{ * }\right| }\frac{{P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) }{{L}^{k}}}\right)  \tag{9}
$$

where ${P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)$ is the max $\left( {k,{Q}_{\eta ,z}^{ * }}\right)$ -product of the clustering ${C}_{z}^{ * }$ . Similarly,define

其中 ${P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)$ 是聚类 ${C}_{z}^{ * }$ 的最大 $\left( {k,{Q}_{\eta ,z}^{ * }}\right)$ 积。类似地，定义

$$
{\bar{p}}_{\eta } = \Theta \left( {1 + \mathop{\max }\limits_{{k = 1}}^{\left| {\bar{F}}^{ * }\right| }\frac{{P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right) }{{L}^{k}}}\right)  \tag{10}
$$

where ${P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right)$ is the max $\left( {k,{\bar{Q}}_{\eta }^{ * }}\right)$ -product of the clustering ${\bar{C}}^{ * }$ . We will prove later that each ${Q}_{\eta ,z}^{ * }$ can be answered with load $O\left( L\right)$ using ${p}_{\eta ,z}$ machines,and ${\bar{Q}}_{\eta }^{ * }$ can be answered with load $O\left( L\right)$ using ${\bar{p}}_{\eta }$ machines. Therefore,applying the cartesian product algorithm given in Lemma 6 of [12] (see also Lemma 4 of [13]),we can compute $\left( {{ \times  }_{z \in  Z}\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right) }\right)  \times  \operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$ with load $O\left( L\right)$ using ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z}$ machines. As proved later,we can adjust the constants in (9) and (10) to make sure ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} \leq  {p}_{\eta }$ ,where ${p}_{\eta }$ is given in (8).

其中 ${P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right)$ 是聚类 ${\bar{C}}^{ * }$ 的最大 $\left( {k,{\bar{Q}}_{\eta }^{ * }}\right)$ 积。我们稍后将证明，每个 ${Q}_{\eta ,z}^{ * }$ 可以使用 ${p}_{\eta ,z}$ 台机器以负载 $O\left( L\right)$ 进行响应，并且 ${\bar{Q}}_{\eta }^{ * }$ 可以使用 ${\bar{p}}_{\eta }$ 台机器以负载 $O\left( L\right)$ 进行响应。因此，应用文献 [12] 引理6（另见文献 [13] 引理4）中给出的笛卡尔积算法，我们可以使用 ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z}$ 台机器以负载 $O\left( L\right)$ 计算 $\left( {{ \times  }_{z \in  Z}\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right) }\right)  \times  \operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$。如后文所证，我们可以调整 (9) 和 (10) 中的常数，以确保 ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} \leq  {p}_{\eta }$，其中 ${p}_{\eta }$ 在 (8) 中给出。

Step 4. We combine the cartesian product $\left( {{ \times  }_{z \in  Z}\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right) }\right)  \times  \operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$ with the tuples broadcast in Step 2 to derive $\operatorname{Join}\left( {Q}_{\eta }\right)$ with no more communication. Specifically,for each tuple $\mathbf{u}$ in the cartesian product,the machine where $\mathbf{u}$ resides outputs $\{ \mathbf{u}\}  \bowtie  \left( {{ \bowtie  }_{e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right) }R\left( {e,\eta }\right) }\right)$ . It is rudimentary to verify that all the tuples of $\operatorname{Join}\left( {Q}_{\eta }\right)$ will be produced this way.

步骤4. 我们将笛卡尔积 $\left( {{ \times  }_{z \in  Z}\operatorname{Join}\left( {Q}_{\eta ,z}^{ * }\right) }\right)  \times  \operatorname{Join}\left( {\bar{Q}}_{\eta }^{ * }\right)$ 与步骤2中广播的元组合并，无需更多通信即可得到 $\operatorname{Join}\left( {Q}_{\eta }\right)$。具体而言，对于笛卡尔积中的每个元组 $\mathbf{u}$，$\mathbf{u}$ 所在的机器输出 $\{ \mathbf{u}\}  \bowtie  \left( {{ \bowtie  }_{e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right) }R\left( {e,\eta }\right) }\right)$。验证 $\operatorname{Join}\left( {Q}_{\eta }\right)$ 的所有元组都将以这种方式生成是很基础的。

## 4 Analysis of the Algorithm

## 4 算法分析

This section will establish:

本节将证明：

Theorem 9. Consider any join query $Q$ defined in Section 1.1 whose hypergraph is $G$ . The algorithm of Section 3 answers $Q$ with load $O\left( L\right)$ ,where $L$ (given in (6)) is the $Q$ -induced load of the clustering obtained from a canonical edge cover of $G$ .

定理9. 考虑第1.1节中定义的任意连接查询 $Q$，其超图为 $G$。第3节的算法以负载 $O\left( L\right)$ 回答 $Q$，其中 $L$（在(6)中给出）是从 $G$ 的规范边覆盖得到的聚类的 $Q$ -诱导负载。

We will prove the theorem by induction on the number of participating attributes (i.e., $\left| V\right|$ ) and the number of participating relations (i.e., $\left| Q\right|$ ). If $\left| Q\right|  = 1$ ,the theorem trivially holds. If $\left| V\right|  = 1,Q$ has only one relation (because $Q$ is clean) and the theorem also holds. Next,assuming that the theorem holds on any query with either strictly less participating attributes or strictly less participating relations than $Q$ ,we will prove the theorem’s correctness on $Q$ .

我们将通过对参与属性的数量（即 $\left| V\right|$）和参与关系的数量（即 $\left| Q\right|$）进行归纳来证明该定理。如果 $\left| Q\right|  = 1$，该定理显然成立。如果 $\left| V\right|  = 1,Q$ 只有一个关系（因为 $Q$ 是干净的），该定理也成立。接下来，假设该定理对任何参与属性严格少于 $Q$ 或参与关系严格少于 $Q$ 的查询都成立，我们将证明该定理在 $Q$ 上的正确性。

Our analysis will answer three questions. First, why do we have enough machines to handle all configurations in parallel? In particular,we must show that $\mathop{\sum }\limits_{\eta }{p}_{\eta } \leq  p$ ,where ${p}_{\eta }$ is given in (8). Second,why does each step in Section 3.2 and 3.3 entail a load of $O\left( L\right)$ ? Third,why do we have ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} \leq  {p}_{\eta }$ in Step 3 of Section 3.3? Settling these questions will complete the proof of Theorem 9.

我们的分析将回答三个问题。首先，为什么我们有足够的机器来并行处理所有配置？具体而言，我们必须证明 $\mathop{\sum }\limits_{\eta }{p}_{\eta } \leq  p$，其中 ${p}_{\eta }$ 在(8)中给出。其次，为什么第3.2节和第3.3节中的每个步骤都会产生 $O\left( L\right)$ 的负载？第三，为什么在第3.3节的步骤3中我们有 ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} \leq  {p}_{\eta }$？解决这些问题将完成定理9的证明。

All the notations in this section follow those in Section 3.

本节中的所有符号都遵循第3节中的符号。

### 4.1 Total Number of Machines for All Configurations

### 4.1 所有配置所需的机器总数

It suffices to prove $\mathop{\sum }\limits_{\eta }{p}_{\eta } = O\left( p\right)$ because adjusting the hidden constants then ensures $\mathop{\sum }\limits_{\eta }{p}_{\eta } \leq  p$ . For every $k \in  \left\lbrack  \left| F\right| \right\rbrack$ ,we will show

证明 $\mathop{\sum }\limits_{\eta }{p}_{\eta } = O\left( p\right)$ 就足够了，因为调整隐藏常数可以确保 $\mathop{\sum }\limits_{\eta }{p}_{\eta } \leq  p$。对于每个 $k \in  \left\lbrack  \left| F\right| \right\rbrack$，我们将证明

$$
\frac{1}{{L}^{k}}\mathop{\sum }\limits_{\eta }{P}_{k}\left( {{Q}_{\eta },C}\right)  = O\left( p\right)  \tag{11}
$$

which will yield

这将得出

$$
\mathop{\sum }\limits_{\eta }{p}_{\eta } = \mathop{\sum }\limits_{\eta }O\left( {1 + \mathop{\max }\limits_{{k = 1}}^{\left| F\right| }\frac{{P}_{k}\left( {{Q}_{\eta },C}\right) }{{L}^{k}}}\right) 
$$

$$
 = \mathop{\sum }\limits_{\eta }O\left( {1 + \mathop{\sum }\limits_{{k = 1}}^{\left| F\right| }\frac{{P}_{k}\left( {{Q}_{\eta },C}\right) }{{L}^{k}}}\right)  = O\left( p\right)  + \mathop{\sum }\limits_{{k = 1}}^{\left| F\right| }O\left( {\mathop{\sum }\limits_{\eta }\frac{{P}_{k}\left( {{Q}_{\eta },C}\right) }{{L}^{k}}}\right) 
$$

$$
 = O\left( p\right) 
$$

where the second equality used $\left| F\right|  = O\left( 1\right)$ and the third equality used $\mathop{\sum }\limits_{\eta }1 = O\left( p\right) {.}^{8}$

其中第二个等式使用了 $\left| F\right|  = O\left( 1\right)$，第三个等式使用了 $\mathop{\sum }\limits_{\eta }1 = O\left( p\right) {.}^{8}$

Henceforth,fix the value of $k$ . For any $\eta$ ,the hypergraph of ${Q}_{\eta }$ is always $G$ (i.e.,the hypergraph of $Q$ ). Consider an arbitrary $k$ -group $K$ of the clustering $C$ (given in Equation 5). The ${Q}_{\eta }$ -product of $K$ is $\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \cdot  {}^{9}$ For any $K$ ,we will prove

此后，固定 $k$ 的值。对于任何 $\eta$，${Q}_{\eta }$ 的超图始终是 $G$（即 $Q$ 的超图）。考虑聚类 $C$（在方程5中给出）的任意 $k$ -组 $K$。$K$ 的 ${Q}_{\eta }$ -积是 $\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \cdot  {}^{9}$ 对于任何 $K$，我们将证明

$$
\frac{1}{{L}^{k}}\mathop{\sum }\limits_{\eta }\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  = O\left( p\right) . \tag{12}
$$

As $C$ has $O\left( 1\right) k$ -groups $K$ ,the above yields

由于 $C$ 具有 $O\left( 1\right) k$ -群 $K$，上述内容得出

$$
\mathop{\sum }\limits_{\eta }\frac{{P}_{k}\left( {{Q}_{\eta },C}\right) }{{L}^{k}} = \mathop{\sum }\limits_{\eta }\frac{1}{{L}^{k}}\mathop{\max }\limits_{K}\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right| 
$$

$$
 = O\left( {\mathop{\sum }\limits_{\eta }\frac{1}{{L}^{k}}\mathop{\sum }\limits_{K}\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right| }\right)  = O\left( {\mathop{\sum }\limits_{K}\frac{1}{{L}^{k}}\mathop{\sum }\limits_{\eta }\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right| }\right) 
$$

$$
 = \mathop{\sum }\limits_{K}O\left( p\right)  = O\left( p\right) 
$$

as claimed in (11).

如 (11) 中所声称的那样。

Let us first consider the case where $K \cap  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)  \neq  \varnothing$ ,namely, $K$ has a hyperedge ${e}_{0}$ picked from the cluster $\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ . We have:

让我们首先考虑 $K \cap  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)  \neq  \varnothing$ 的情况，即 $K$ 有一条从簇 $\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ 中选取的超边 ${e}_{0}$。我们有：

$$
\mathop{\sum }\limits_{\eta }\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  = \mathop{\sum }\limits_{\eta }\left( {\left| {R\left( {{e}_{0},\eta }\right) }\right|  \cdot  \mathop{\prod }\limits_{{e \in  K \smallsetminus  \left\{  {e}_{0}\right\}  }}\left| {R\left( {e,\eta }\right) }\right| }\right)  \tag{13}
$$

For each $e \in  K \smallsetminus  \left\{  {e}_{0}\right\}$ ,obviously $\left| {R\left( {e,\eta }\right) }\right|  \leq  \left| {R\left( e\right) }\right|$ . Regarding ${e}_{0}$ ,because ${A}^{ \circ  }$ must be an attribute of ${e}_{0}$ ,the relations $R\left( {{e}_{0},\eta }\right)$ of all the configurations $\eta$ form a partition of $R\left( {e}_{0}\right) .{}^{10}$ Hence:

对于每个 $e \in  K \smallsetminus  \left\{  {e}_{0}\right\}$，显然有 $\left| {R\left( {e,\eta }\right) }\right|  \leq  \left| {R\left( e\right) }\right|$。关于 ${e}_{0}$，因为 ${A}^{ \circ  }$ 必须是 ${e}_{0}$ 的一个属性，所有配置 $\eta$ 的关系 $R\left( {{e}_{0},\eta }\right)$ 构成了 $R\left( {e}_{0}\right) .{}^{10}$ 的一个划分。因此：

$$
 \leq  \left( {\mathop{\prod }\limits_{{e \in  K \smallsetminus  \left\{  {e}_{0}\right\}  }}\left| {R\left( e\right) }\right| }\right) \left( {\mathop{\sum }\limits_{\eta }\left| {R\left( {{e}_{0},\eta }\right) }\right| }\right)  = \left( {\mathop{\prod }\limits_{{e \in  K \smallsetminus  \left\{  {e}_{0}\right\}  }}\left| {R\left( e\right) }\right| }\right)  \cdot  \left| {R\left( {e}_{0}\right) }\right|  = \mathop{\prod }\limits_{{e \in  K}}\left| {R\left( e\right) }\right| 
$$

$$
 \leq  \max \left( {k,Q}\right) \text{-product of}C\text{.}
$$

---

<!-- Footnote -->

${}^{8}\mathop{\sum }\limits_{\eta }1$ is the number of configurations which is $O\left( p\right)$ as shown in (7).

${}^{8}\mathop{\sum }\limits_{\eta }1$ 是配置的数量，如 (7) 中所示为 $O\left( p\right)$。

${}^{9}$ For the definition of "a k-group’s $Q$ -product",review Section 1.3.

${}^{9}$ 关于 “一个 k - 群的 $Q$ -积” 的定义，请回顾 1.3 节。

<!-- Footnote -->

---

Therefore,the left hand side of (12) is bounded by $\frac{\max \left( {k,Q}\right)  - \operatorname{product\ of}C}{{L}^{k}}$ ,which is at most $p$ (by definition of $L$ ).

因此，(12) 的左边以 $\frac{\max \left( {k,Q}\right)  - \operatorname{product\ of}C}{{L}^{k}}$ 为界，而 $\frac{\max \left( {k,Q}\right)  - \operatorname{product\ of}C}{{L}^{k}}$ 至多为 $p$（根据 $L$ 的定义）。

Next,we consider $K \cap  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)  = \varnothing$ . In this case,we must have $k = \left| K\right|  \leq  \left| F\right|  - 1$ ,because the hyperedges in $K$ need to come from distinct clusters of $C$ ,and $C$ has $\left| F\right|$ clusters (one of them is sigpath $\left( {{f}^{ \circ  },T}\right)$ ,which now must be excluded). Applying the trivial fact $\left| {R\left( {e,\eta }\right) }\right|  \leq  \left| {R\left( e\right) }\right|$ (for any $e)$ and the fact that $\mathop{\sum }\limits_{\eta }1$ is bounded by (7),we have

接下来，我们考虑 $K \cap  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)  = \varnothing$。在这种情况下，我们必定有 $k = \left| K\right|  \leq  \left| F\right|  - 1$，因为 $K$ 中的超边需要来自 $C$ 的不同簇，并且 $C$ 有 $\left| F\right|$ 个簇（其中一个是信号路径 $\left( {{f}^{ \circ  },T}\right)$，现在必须排除）。应用平凡事实 $\left| {R\left( {e,\eta }\right) }\right|  \leq  \left| {R\left( e\right) }\right|$（对于任何 $e)$）以及 $\mathop{\sum }\limits_{\eta }1$ 以 (7) 为界这一事实，我们有

$$
\frac{1}{{L}^{k}}\mathop{\sum }\limits_{\eta }\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \leq  \frac{1}{{L}^{k}}\mathop{\sum }\limits_{\eta }\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( e\right) }\right|  = O\left( {\frac{1}{{L}^{k}}\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( e\right) }\right|  \cdot  \mathop{\max }\limits_{{e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right) }}\frac{\left| R\left( e\right) \right| }{L}}\right) 
$$

$$
 = O\left( \frac{\max \left( {k + 1,Q}\right)  - \text{ product of }C}{{L}^{k + 1}}\right) 
$$

which is at most $p$ . This completes the proof of $\mathop{\sum }\limits_{\eta }{p}_{\eta } = O\left( p\right)$ .

其至多为 $p$。这就完成了 $\mathop{\sum }\limits_{\eta }{p}_{\eta } = O\left( p\right)$ 的证明。

### 4.2 Heavy ${Q}_{\eta }$

### 4.2 重 ${Q}_{\eta }$

This subsection will prove that the algorithm in Section 3.2 has load $O\left( L\right)$ . Step 2 and 5 demand no communication. The loads of Step 1 and 3 can all be bounded ${}^{11}$ by $O\left( {\frac{1}{{p}_{\eta }}\mathop{\sum }\limits_{{e \in  E}}\left| {R\left( {e,\eta }\right) }\right| }\right)  =$ $O\left( {\frac{1}{{p}_{\eta }}\mathop{\max }\limits_{{e \in  E}}\left| {R\left( {e,\eta }\right) }\right| }\right)  = O\left( {{P}_{1}\left( {{Q}_{\eta },C}\right) /{p}_{\eta }}\right)  = O\left( L\right) .$

本小节将证明3.2节中的算法具有负载$O\left( L\right)$。步骤2和步骤5无需通信。步骤1和步骤3的负载都可以由$O\left( {\frac{1}{{p}_{\eta }}\mathop{\sum }\limits_{{e \in  E}}\left| {R\left( {e,\eta }\right) }\right| }\right)  =$ $O\left( {\frac{1}{{p}_{\eta }}\mathop{\max }\limits_{{e \in  E}}\left| {R\left( {e,\eta }\right) }\right| }\right)  = O\left( {{P}_{1}\left( {{Q}_{\eta },C}\right) /{p}_{\eta }}\right)  = O\left( L\right) .$界定为${}^{11}$

To analyze Step 4,let ${T}^{ * }$ be the hyperedge tree of ${G}^{ * }$ (produced by cleansing) and ${F}^{ * }$ be the CEC of ${G}^{ * }$ . By definition,the ${Q}_{\eta }^{ * }$ -induced load of the clustering ${C}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}^{ * }}\right)  \mid  {f}^{ * } \in  {F}^{ * }}\right\}$ is

为了分析步骤4，设${T}^{ * }$为${G}^{ * }$的超边树（由清理操作生成），${F}^{ * }$为${G}^{ * }$的连通边分量（CEC）。根据定义，聚类${C}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}^{ * }}\right)  \mid  {f}^{ * } \in  {F}^{ * }}\right\}$的${Q}_{\eta }^{ * }$ - 诱导负载为

$$
{L}_{\eta }^{ * } = \mathop{\max }\limits_{{k = 1}}^{\left| {F}^{ * }\right| }{\left( \frac{{P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right) }{{p}_{\eta }}\right) }^{1/k} \tag{14}
$$

where ${P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)$ is the $\max \left( {k,{Q}_{\eta }^{ * }}\right)$ -product of ${C}^{ * }$ . By our inductive assumption (that Theorem 9 holds on $\left. {Q}_{\eta }^{ * }\right)$ ,Step 4 incurs load $O\left( {L}_{\eta }^{ * }\right)$ . We will prove ${P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right)$ for every $k$ which, together with (8) and (14),will tell us ${L}_{\eta }^{ * } = O\left( L\right)$ .

其中${P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)$是${C}^{ * }$的$\max \left( {k,{Q}_{\eta }^{ * }}\right)$ - 积。根据我们的归纳假设（即定理9在$\left. {Q}_{\eta }^{ * }\right)$上成立），步骤4产生的负载为$O\left( {L}_{\eta }^{ * }\right)$。我们将证明对于每个$k$都有${P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right)$，结合(8)式和(14)式，这将告诉我们${L}_{\eta }^{ * } = O\left( L\right)$。

Before proceeding,the reader should recall that,for any hyperedge ${e}^{ * }$ of ${G}^{ * },{ma}{p}^{-1}\left( {e}^{ * }\right)$ gives a hyperedge in $G$ . We must have $\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  \left| {R\left( {{ma}{p}^{-1}\left( {e}^{ * }\right) ,\eta }\right) }\right|$ . To see why,note that this is true when $\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|$ is created in Step 2,whereas ${R}^{ * }\left( {{e}^{ * },\eta }\right)$ can only shrink in Steps 3-5.

在继续之前，读者应该回想一下，对于${G}^{ * },{ma}{p}^{-1}\left( {e}^{ * }\right)$的任何超边${e}^{ * }$，都会在$G$中产生一个超边。我们必须有$\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  \left| {R\left( {{ma}{p}^{-1}\left( {e}^{ * }\right) ,\eta }\right) }\right|$。要明白为什么，请注意，当$\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|$在步骤2中创建时这是成立的，而在步骤3 - 5中${R}^{ * }\left( {{e}^{ * },\eta }\right)$只会缩小。

To prove ${P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right)$ ,consider any $k$ -group ${K}^{ * }$ of ${C}^{ * }$ . By Lemma 6, $K =$ $\left\{  {{ma}{p}^{-1}\left( {e}^{ * }\right)  \mid  {e}^{ * } \in  {K}^{ * }}\right\}$ must be a $k$ -group of $C$ . Since $\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  \left| {R\left( {{ma}{p}^{-1}\left( {e}^{ * }\right) ,\eta }\right) }\right|$ for any ${e}^{ * } \in  {K}^{ * }$ ,we have $\mathop{\prod }\limits_{{{e}^{ * } \in  {K}^{ * }}}\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right)$ . Therefore:

为了证明${P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right)$，考虑${C}^{ * }$的任意$k$ - 群${K}^{ * }$。根据引理6，$K =$ $\left\{  {{ma}{p}^{-1}\left( {e}^{ * }\right)  \mid  {e}^{ * } \in  {K}^{ * }}\right\}$必定是$C$的一个$k$ - 群。由于对于任意${e}^{ * } \in  {K}^{ * }$都有$\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  \left| {R\left( {{ma}{p}^{-1}\left( {e}^{ * }\right) ,\eta }\right) }\right|$，我们有$\mathop{\prod }\limits_{{{e}^{ * } \in  {K}^{ * }}}\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right)$。因此：

$$
{P}_{k}\left( {{Q}_{\eta }^{ * },{C}^{ * }}\right)  = \mathop{\max }\limits_{{K}^{ * }}\mathop{\prod }\limits_{{{e}^{ * } \in  {K}^{ * }}}\left| {{R}^{ * }\left( {{e}^{ * },\eta }\right) }\right|  \leq  {P}_{k}\left( {{Q}_{\eta },C}\right) .
$$

---

<!-- Footnote -->

${}^{10}$ The $R\left( {{e}_{0},\eta }\right)$ of all the $\eta$ are mutually disjoint and their union equals $R\left( {e}_{0}\right)$ .

${}^{10}$ 所有$\eta$的$R\left( {{e}_{0},\eta }\right)$是互不相交的，并且它们的并集等于$R\left( {e}_{0}\right)$。

${}^{11}$ Step 3 performs $O\left( 1\right)$ semi joins,each of which can be performed by sorting. For sorting in the MPC model,see Section 2.2.1 of [10]. The stated bound for Step 1 and 3 requires the assumption $p \leq  {m}^{1 - \epsilon }$ .

${}^{11}$ 步骤3执行$O\left( 1\right)$次半连接，每次半连接都可以通过排序来执行。关于在多方计算（MPC，Multi - Party Computation）模型中的排序，请参阅文献[10]的第2.2.1节。步骤1和步骤3所陈述的界限需要假设$p \leq  {m}^{1 - \epsilon }$。

<!-- Footnote -->

---

### 4.3 Light ${Q}_{\eta }$

### 4.3 轻量级${Q}_{\eta }$

This subsection will concentrate on the algorithm of Section 3.3.

本小节将专注于第3.3节的算法。

Load. Step 1 incurs load $O\left( L\right)$ (same analysis as in Section 3.2). Step 2 also requires a load of $O\left( L\right)$ because every broadcast relation has a size of at most $L$ . Step 4 needs no communication.

负载。步骤1产生的负载为$O\left( L\right)$（与第3.2节的分析相同）。步骤2也需要$O\left( L\right)$的负载，因为每个广播关系的大小至多为$L$。步骤4不需要通信。

To analyze Step 3,let us first consider ${\bar{Q}}_{\eta }^{ * }$ . The ${\bar{Q}}_{\eta }^{ * }$ -induced load of the clustering ${\bar{C}}^{ * }$ is

为了分析步骤3，让我们首先考虑${\bar{Q}}_{\eta }^{ * }$。聚类${\bar{C}}^{ * }$的由${\bar{Q}}_{\eta }^{ * }$引起的负载为

$$
{\bar{L}}_{\eta }^{ * } = \mathop{\max }\limits_{{k = 1}}^{\left| {\bar{C}}^{ * }\right| }{\left( \frac{{P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right) }{{\bar{p}}_{\eta }}\right) }^{1/k}
$$

where ${P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right)$ as the $\max \left( {k,{\bar{Q}}_{\eta }^{ * }}\right)$ -product of ${\bar{C}}^{ * }$ . By our inductive assumption (that Theorem 9 holds on ${\bar{Q}}_{\eta }^{ * }$ ),answering ${\bar{Q}}_{\eta }^{ * }$ with ${\bar{p}}_{\eta }$ machines requires load $O\left( {\bar{L}}_{\eta }^{ * }\right)$ ,which is $O\left( L\right)$ given the ${\bar{p}}_{\eta }$ in (10). A similar argument shows that answering each ${Q}_{\eta ,z}^{ * }$ with ${p}_{\eta ,z}$ machines - with ${p}_{\eta ,z}$ given in (9) - incurs a load of $O\left( L\right)$ . Thus,the cartesian product at Step 3 can be computed with load $O\left( L\right)$ .

其中${P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right)$为${\bar{C}}^{ * }$的$\max \left( {k,{\bar{Q}}_{\eta }^{ * }}\right)$ - 积。根据我们的归纳假设（即定理9在${\bar{Q}}_{\eta }^{ * }$上成立），用${\bar{p}}_{\eta }$台机器回答${\bar{Q}}_{\eta }^{ * }$需要负载$O\left( {\bar{L}}_{\eta }^{ * }\right)$，在(10)式给定的${\bar{p}}_{\eta }$的情况下，该负载为$O\left( L\right)$。类似的论证表明，用(9)式给定的${p}_{\eta ,z}$台机器回答每个${Q}_{\eta ,z}^{ * }$会产生$O\left( L\right)$的负载。因此，步骤3的笛卡尔积可以在负载为$O\left( L\right)$的情况下计算得出。

Number of machines in Step 3. Next,we will prove that ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} \leq  {p}_{\eta }$ always holds in Step 3. It suffices to show ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} = O\left( {p}_{\eta }\right)$ which,as we will see,relies on Lemma 8 and the fact that $\left| {R\left( {e,\eta }\right) }\right|  \leq  L$ for every node $e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ .

步骤3中的机器数量。接下来，我们将证明${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} \leq  {p}_{\eta }$在步骤3中始终成立。只需证明${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} = O\left( {p}_{\eta }\right)$即可，正如我们将看到的，这依赖于引理8以及对于每个节点$e \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$都有$\left| {R\left( {e,\eta }\right) }\right|  \leq  L$这一事实。

Consider an arbitrary $z \in  Z$ . The root of ${T}_{z}^{ * } -$ denoted as ${e}_{\text{root }} -$ must belong to $\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ . Recall that a $k$ -group $K$ of ${C}_{z}^{ * }$ takes a hyperedge from a distinct cluster in ${C}_{z}^{ * }$ . Call $K$ a non-root $k$ -group if ${e}_{\text{root }} \notin  K$ ,or a root $k$ -group,otherwise. Define

考虑任意的$z \in  Z$。${T}_{z}^{ * } -$的根（记为${e}_{\text{root }} -$）必定属于$\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$。回顾一下，${C}_{z}^{ * }$的一个$k$ - 组$K$从${C}_{z}^{ * }$中的一个不同簇中选取一条超边。如果${e}_{\text{root }} \notin  K$，则称$K$为非根$k$ - 组；否则，称其为根$k$ - 组。定义

$$
{P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  = \max \left( {k,{Q}_{\eta ,z}^{ * }}\right) \text{-product of}{C}_{z}^{ * }
$$

${P}_{k}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  = \max \left( {k,{Q}_{\eta ,z}^{ * }}\right)$ -product of all the non-root $k$ -groups of ${C}_{z}^{ * }$ .

${P}_{k}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  = \max \left( {k,{Q}_{\eta ,z}^{ * }}\right)$ - ${C}_{z}^{ * }$的所有非根$k$ - 组的乘积。

As a special case,define ${P}_{0}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  = 1$ . For any $k$ ,we observe

作为一种特殊情况，定义${P}_{0}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  = 1$。对于任意的$k$，我们观察到

$$
{P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  \leq  \max \left\{  {{P}_{k}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) ,L \cdot  {P}_{k - 1}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) }\right\}  . \tag{15}
$$

To prove the inequality,fix $K$ to the $k$ -group with the largest ${Q}_{\eta ,z}^{ * }$ -product $\left( { = {P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) }\right)$ . If $K$ is a non-root $k$ -group,(15) obviously holds. Consider,instead,that $K$ is a root $k$ -group. Since ${e}_{\text{root }} \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ ,we know $\left| {R\left( {{e}_{\text{root }},\eta }\right) }\right|  \leq  L$ and hence $\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \leq  L \cdot  \mathop{\prod }\limits_{{e \in  K \smallsetminus  \left\{  {e}_{\text{root }}\right\}  }}\left| {R\left( {e,\eta }\right) }\right|$ . As $K \smallsetminus  \left\{  {e}_{\text{root }}\right\}$ is a non-root(k - 1)-group, ${P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  \leq  L \cdot  {P}_{k - 1}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)$ holds.

为了证明这个不等式，将$K$固定为具有最大${Q}_{\eta ,z}^{ * }$ - 乘积$\left( { = {P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) }\right)$的$k$ - 组。如果$K$是一个非根$k$ - 组，那么(15)显然成立。相反，考虑$K$是一个根$k$ - 组的情况。由于${e}_{\text{root }} \in  \operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$，我们知道$\left| {R\left( {{e}_{\text{root }},\eta }\right) }\right|  \leq  L$，因此$\mathop{\prod }\limits_{{e \in  K}}\left| {R\left( {e,\eta }\right) }\right|  \leq  L \cdot  \mathop{\prod }\limits_{{e \in  K \smallsetminus  \left\{  {e}_{\text{root }}\right\}  }}\left| {R\left( {e,\eta }\right) }\right|$。由于$K \smallsetminus  \left\{  {e}_{\text{root }}\right\}$是一个非根(k - 1) - 组，所以${P}_{k}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)  \leq  L \cdot  {P}_{k - 1}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)$成立。

Equipped with (15), we can now derive from (9):

有了(15)，我们现在可以从(9)推导出：

$$
{p}_{\eta ,z} = O\left( {1 + \mathop{\max }\limits_{{k = 1}}^{\left| {F}_{z}^{ * }\right| }\frac{\max \left\{  {{P}_{k}^{non}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) ,L \cdot  {P}_{k - 1}^{non}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) }\right\}  }{{L}^{k}}}\right) 
$$

$$
 = O\left( {1 + \mathop{\max }\limits_{{k = 1}}^{{\left| {F}_{z}^{ * }\right|  - 1}}\frac{{P}_{k}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) }{{L}^{k}}}\right)  \tag{16}
$$

where the second equality used the fact that,when $k = \left| {F}_{z}^{ * }\right|$ ,a $k$ -group must be a root $k$ -group.

其中第二个等式利用了这样一个事实，即当$k = \left| {F}_{z}^{ * }\right|$时，一个$k$ - 组必定是一个根$k$ - 组。

We are now ready to prove ${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} = O\left( {p}_{\eta }\right)$ . For each $z \in  Z$ ,define integer ${k}_{z}$ and a set ${K}_{z}$ of hyperedges as follows:

我们现在准备证明${\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} = O\left( {p}_{\eta }\right)$。对于每个$z \in  Z$，定义整数${k}_{z}$和一个超边集合${K}_{z}$如下：

- If $\left( {16}\right)  = \Theta \left( {{P}_{k}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) /{L}^{k}}\right)$ for some $k \in  \left\lbrack  {1,\left| {F}_{z}^{ * }\right|  - 1}\right\rbrack$ ,set ${k}_{z} = k$ and ${K}_{z}$ to the non-root $k$ -group whose ${Q}_{\eta ,z}^{ * }$ -product equals ${P}_{k}^{non}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)$ .

- 如果对于某个$k \in  \left\lbrack  {1,\left| {F}_{z}^{ * }\right|  - 1}\right\rbrack$有$\left( {16}\right)  = \Theta \left( {{P}_{k}^{\text{non }}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right) /{L}^{k}}\right)$，则将${k}_{z} = k$和${K}_{z}$设为非根$k$ - 组，其${Q}_{\eta ,z}^{ * }$ - 积等于${P}_{k}^{non}\left( {{Q}_{\eta ,z}^{ * },{C}_{z}^{ * }}\right)$。

- Otherwise (we must have ${p}_{\eta ,z} = \Theta \left( 1\right)$ ),set ${k}_{z} = 0$ and ${K}_{z} = \varnothing$ ; furthermore,define the ${Q}_{\eta ,z}^{ * }$ -product of ${K}_{z}$ to be 1 .

- 否则（我们必定有${p}_{\eta ,z} = \Theta \left( 1\right)$），设${k}_{z} = 0$和${K}_{z} = \varnothing$；此外，定义${K}_{z}$的${Q}_{\eta ,z}^{ * }$ - 积为 1。

Similarly,regarding ${\bar{p}}_{\eta }$ in (10),define integer $\bar{k}$ and a set $\bar{K}$ of hyperedges as follows:

类似地，对于(10)中的${\bar{p}}_{\eta }$，定义整数$\bar{k}$和一个超边集合$\bar{K}$如下：

- If $\left( {10}\right)  = \Theta \left( {{P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right) /{L}^{k}}\right)$ for some $k \in  \left\lbrack  {1,\left| {\bar{F}}^{ * }\right| }\right\rbrack$ ,set $\bar{k} = k$ and $\bar{K}$ to the $k$ -group of the clustering ${\bar{C}}^{ * }$ whose ${\bar{Q}}_{\eta }^{ * }$ -product equals ${P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right)$ .

- 如果对于某个$k \in  \left\lbrack  {1,\left| {\bar{F}}^{ * }\right| }\right\rbrack$有$\left( {10}\right)  = \Theta \left( {{P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right) /{L}^{k}}\right)$，则将$\bar{k} = k$和$\bar{K}$设为聚类${\bar{C}}^{ * }$的$k$ - 组，其${\bar{Q}}_{\eta }^{ * }$ - 积等于${P}_{k}\left( {{\bar{Q}}_{\eta }^{ * },{\bar{C}}^{ * }}\right)$。

- Otherwise,set $\bar{k} = 0$ and $\bar{K} = \varnothing$ ; furthermore,define the ${\bar{Q}}_{\eta }^{ * }$ -product of $\bar{K}$ to be 1 .

- 否则，设$\bar{k} = 0$和$\bar{K} = \varnothing$；此外，定义$\bar{K}$的${\bar{Q}}_{\eta }^{ * }$ - 积为 1。

Define ${K}_{\text{super }} = \bar{K} \cup  \left( {\mathop{\bigcup }\limits_{{z \in  Z}}{K}_{z}}\right)$ . If ${K}_{\text{super }} = \varnothing$ ,then ${p}_{\eta ,z} = \Theta \left( 1\right)$ for all $z \in  Z$ and ${\bar{p}}_{\eta } = \Theta \left( 1\right)$ , which leads to

定义${K}_{\text{super }} = \bar{K} \cup  \left( {\mathop{\bigcup }\limits_{{z \in  Z}}{K}_{z}}\right)$。如果${K}_{\text{super }} = \varnothing$，那么对于所有的$z \in  Z$和${\bar{p}}_{\eta } = \Theta \left( 1\right)$有${p}_{\eta ,z} = \Theta \left( 1\right)$，这导致

$$
{\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} = O\left( 1\right)  = O\left( {p}_{\eta }\right) .
$$

If ${K}_{\text{super }} \neq  \varnothing ,{K}_{\text{super }}$ is a super- $\left| {K}_{\text{super }}\right|$ -group ${}^{12}$ . By Lemma 8, ${K}_{\text{super }}$ is a $\left| {K}_{\text{super }}\right|$ -group of $T$ . We thus have:

如果${K}_{\text{super }} \neq  \varnothing ,{K}_{\text{super }}$是一个超$\left| {K}_{\text{super }}\right|$ - 组${}^{12}$。根据引理 8，${K}_{\text{super }}$是$T$的一个$\left| {K}_{\text{super }}\right|$ - 组。因此我们有：

$$
{\bar{p}}_{\eta } \cdot  \mathop{\prod }\limits_{{z \in  Z}}{p}_{\eta ,z} = \frac{{\bar{Q}}_{\eta }^{ * } - \text{ product of }\bar{K}}{{L}^{\bar{k}}}\mathop{\prod }\limits_{{z \in  Z}}\frac{{Q}_{\eta ,z}^{ * } - \text{ product of }{K}_{z}}{{L}^{{k}_{z}}}
$$

$$
 = \frac{\mathop{\prod }\limits_{{e \in  {K}_{\text{super }}}}\left| {R\left( e\right) }\right| }{{L}^{\left| {K}_{\text{super }}\right| }} \leq  \frac{\max \left( {\left| {K}_{\text{super }}\right| ,{Q}_{\eta }}\right)  - \text{ product of }C}{{L}^{\left| {K}_{\text{super }}\right| }} = O\left( {p}_{\eta }\right) .
$$

This completes the whole proof of Theorem 9.

至此完成了定理 9 的整个证明。

## References

## 参考文献

[1] Serge Abiteboul, Richard Hull, and Victor Vianu. Foundations of Databases. Addison-Wesley, 1995.

[2] Foto N. Afrati, Manas R. Joglekar, Christopher Ré, Semih Salihoglu, and Jeffrey D. Ullman. GYM: A multiround distributed join algorithm. In Proceedings of International Conference on Database Theory (ICDT), pages 4:1-4:18, 2017.

[3] Foto N. Afrati and Jeffrey D. Ullman. Optimizing multiway joins in a map-reduce environment. IEEE Transactions on Knowledge and Data Engineering (TKDE), 23(9):1282-1298, 2011.

[4] Kaleb Alway, Eric Blais, and Semih Salihoglu. Box covers and domain orderings for beyond worst-case join processing. In Proceedings of International Conference on Database Theory (ICDT), pages 3:1-3:23, 2021.

[5] Albert Atserias, Martin Grohe, and Daniel Marx. Size bounds and query plans for relational joins. SIAM Journal of Computing, 42(4):1737-1767, 2013.

[6] Paul Beame, Paraschos Koutris, and Dan Suciu. Communication steps for parallel query processing. Journal of the ACM (JACM), 64(6):40:1-40:58, 2017.

[7] Christoph Berkholz, Jens Keppeler, and Nicole Schweikardt. Answering conjunctive queries under updates. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 303-318, 2017.

---

<!-- Footnote -->

${}^{12}$ For the definition of super- $k$ -group,review Section 2.2.2.

${}^{12}$ 关于超$k$ - 组的定义，请回顾 2.2.2 节。

<!-- Footnote -->

---

[8] Xiao Hu. Cover or pack: New upper and lower bounds for massively parallel joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 181-198, 2021.

[9] Xiao Hu and Ke Yi. Instance and output optimal parallel algorithms for acyclic joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 450-463, 2019.

[10] Xiao Hu, Ke Yi, and Yufei Tao. Output-optimal massively parallel algorithms for similarity joins. ACM Transactions on Database Systems (TODS), 44(2):6:1-6:36, 2019.

[11] Muhammad Idris, Martín Ugarte, and Stijn Vansummeren. The dynamic yannakakis algorithm: Compact and efficient query processing under updates. In Proceedings of ACM Management of Data (SIGMOD), pages 1259-1274. ACM, 2017.

[12] Bas Ketsman and Dan Suciu. A worst-case optimal multi-round algorithm for parallel computation of conjunctive queries. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 417-428, 2017.

[13] Bas Ketsman, Dan Suciu, and Yufei Tao. A near-optimal parallel algorithm for joining binary relations. CoRR, abs/2011.14482, 2020.

[14] Mahmoud Abo Khamis, Hung Q. Ngo, Christopher Ré, and Atri Rudra. Joins via geometric resolutions: Worst-case and beyond. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 213-228, 2015.

[15] Paraschos Koutris, Paul Beame, and Dan Suciu. Worst-case optimal algorithms for parallel query processing. In Proceedings of International Conference on Database Theory (ICDT), pages $8 : 1 - 8 : {18},{2016}$ .

[16] Hung Q. Ngo, Dung T. Nguyen, Christopher Re, and Atri Rudra. Beyond worst-case analysis for joins with minesweeper. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 234-245, 2014.

[17] Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. Worst-case optimal join algorithms: [extended abstract]. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 37-48, 2012.

[18] Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. Worst-case optimal join algorithms. Journal of the ${ACM}\left( {JACM}\right) ,{65}\left( 3\right)  : {16} : 1 - {16} : {40},{2018}$ .

[19] Hung Q. Ngo, Christopher Re, and Atri Rudra. Skew strikes back: new developments in the theory of join algorithms. SIGMOD Rec., 42(4):5-16, 2013.

[20] Miao Qiao and Yufei Tao. Two-attribute skew free, isolated CP theorem, and massively parallel joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 166-180, 2021.

[21] Yufei Tao. A simple parallel algorithm for natural joins on binary relations. In Proceedings of International Conference on Database Theory (ICDT), pages 25:1-25:18, 2020.

[22] Mihalis Yannakakis. Algorithms for acyclic database schemes. In Proceedings of Very Large Data Bases (VLDB), pages 82-94, 1981.

## Appendix

## 附录

## A Proof of Lemma 1

## A 引理 1 的证明

We first show that $F$ is an edge cover of $G$ . Each attribute $X \in  V$ is a disappearing attribute of some hyperedge $e \in  E$ . When $e$ is processed at Line 4 of edge-cover,either $X$ is already covered or $e$ itself will be added to ${F}_{\text{tmp }}$ (which will then cover $X$ ).

我们首先证明 $F$ 是 $G$ 的一个边覆盖（edge cover）。每个属性 $X \in  V$ 都是某个超边 $e \in  E$ 的消失属性（disappearing attribute）。当在边覆盖算法的第 4 行处理 $e$ 时，要么 $X$ 已经被覆盖，要么 $e$ 本身会被添加到 ${F}_{\text{tmp }}$ 中（这样就会覆盖 $X$）。

Next,we argue that $F$ is an optimal edge cover (i.e.,having the smallest size). Let ${F}^{\prime }$ be an arbitrary optimal edge cover of $G$ . We will establish a one-one mapping between $F$ and ${F}^{\prime }$ ,which implies the optimality of $F$ .

接下来，我们证明 $F$ 是一个最优边覆盖（即，具有最小的规模）。设 ${F}^{\prime }$ 是 $G$ 的任意一个最优边覆盖。我们将在 $F$ 和 ${F}^{\prime }$ 之间建立一一映射，这意味着 $F$ 的最优性。

Fix an arbitrary hyperedge $e \in  F$ . If $e$ also belongs to ${F}^{\prime }$ ,we map $e$ to its copy in ${F}^{\prime }$ . Consider the opposite case where $e \notin  {F}^{\prime }$ . The fact $e \in  F$ indicates that when edge-cover processes $e,e$ must contain a disappearing attribute $X$ that has not been covered by ${F}_{\mathrm{{tmp}}}$ . Let ${e}^{\prime } \in  {F}^{\prime }$ be an arbitrary hyperedge containing $X$ ; we map $e$ to ${e}^{\prime }$ . As explained in Section 2.1, ${e}^{\prime }$ must be a proper descendant of $e$ in $T$ .

固定任意一个超边 $e \in  F$。如果 $e$ 也属于 ${F}^{\prime }$，我们将 $e$ 映射到 ${F}^{\prime }$ 中的其副本。考虑相反的情况，即 $e \notin  {F}^{\prime }$。事实 $e \in  F$ 表明，当边覆盖算法处理 $e,e$ 时，$e,e$ 必须包含一个尚未被 ${F}_{\mathrm{{tmp}}}$ 覆盖的消失属性 $X$。设 ${e}^{\prime } \in  {F}^{\prime }$ 是包含 $X$ 的任意一个超边；我们将 $e$ 映射到 ${e}^{\prime }$。如第 2.1 节所述，${e}^{\prime }$ 必须是 $e$ 在 $T$ 中的一个真后代（proper descendant）。

We argue that no two $e$ and $\widehat{e}$ in $F$ can be mapped to the same hyperedge ${e}^{\prime } \in  F$ . If this happens, ${e}^{\prime }$ is a descendant of both $e$ and $\widehat{e}$ . Assume,without loss of generality,that $e$ is a proper descendant of $\widehat{e}$ . Since $\widehat{e}$ is mapped to ${e}^{\prime }$ ,there is an attribute $Y$ such that

我们证明 $F$ 中没有两个 $e$ 和 $\widehat{e}$ 可以映射到同一个超边 ${e}^{\prime } \in  F$。如果发生这种情况，${e}^{\prime }$ 是 $e$ 和 $\widehat{e}$ 的后代。不失一般性，假设 $e$ 是 $\widehat{e}$ 的真后代。由于 $\widehat{e}$ 被映射到 ${e}^{\prime }$，存在一个属性 $Y$ 使得

- $Y$ is a disappearing attribute in $\widehat{e}$ not covered by ${F}_{\text{tmp }}$ when edge-cover adds $\widehat{e}$ to ${F}_{\text{tmp }}$ ;

- 当边覆盖算法将 $\widehat{e}$ 添加到 ${F}_{\text{tmp }}$ 时，$Y$ 是 $\widehat{e}$ 中未被 ${F}_{\text{tmp }}$ 覆盖的消失属性；

- $Y \in  {e}^{\prime }$ .

Because $e$ is on the path from $\widehat{e}$ to ${e}^{\prime }$ in $T$ ,connectedness of acyclicity guarantees $Y \in  e$ . On the other hand, $e \in  F$ and $e$ is processed before $\widehat{e}$ (reverse topological order). Thus,when $\widehat{e}$ is processed, $e \in  {F}_{\mathrm{{tmp}}}$ and hence $Y$ must be covered by ${F}_{\mathrm{{tmp}}}$ ,giving a contradiction.

因为 $e$ 在 $T$ 中从 $\widehat{e}$ 到 ${e}^{\prime }$ 的路径上，无环连通性保证了 $Y \in  e$。另一方面，$e \in  F$ 且 $e$ 在 $\widehat{e}$ 之前被处理（逆拓扑序）。因此，当处理 $\widehat{e}$ 时，$e \in  {F}_{\mathrm{{tmp}}}$ 并且因此 $Y$ 必须被 ${F}_{\mathrm{{tmp}}}$ 覆盖，这产生了矛盾。

We now proceed to show that $F$ does not depend on the reverse topological order at Line 2 . Recall that,when processing a node $e$ ,edge-cover adds it to ${F}_{\mathrm{{tmp}}}$ if and only if ${F}_{\mathrm{{tmp}}}$ does not cover a disappearing attribute $X$ of $e$ . All the nodes containing $X$ must appear in the subtree of $T$ rooted at $e$ and thus must be processed before $e$ . Hence,whether $e \in  F$ is determined by which of those nodes are selected into ${F}_{\mathrm{{tmp}}}$ . The observation gives rise to an inductive argument. First,if $e$ is a leaf, $e$ enters ${F}_{\mathrm{{tmp}}}$ if and only if it has a disappearing attribute (which must be exclusive), independent of the reverse topological order used. For a non-leaf node $e$ ,inductively,once we have decided whether ${e}^{\prime } \in  {F}_{\text{tmp }}$ for every proper descendent ${e}^{\prime }$ of $e$ ,whether $e \in  {F}_{\text{tmp }}$ has also been decided. We thus conclude that the reverse topological order has no influence on the output.

我们现在来证明$F$不依赖于第2行的逆拓扑排序。回顾一下，在处理节点$e$时，当且仅当${F}_{\mathrm{{tmp}}}$不覆盖$e$的一个消失属性$X$时，边覆盖（edge - cover）算法才会将其添加到${F}_{\mathrm{{tmp}}}$中。所有包含$X$的节点都必须出现在以$e$为根的$T$的子树中，因此必须在处理$e$之前进行处理。因此，$e \in  F$是否被选中是由哪些节点被选入${F}_{\mathrm{{tmp}}}$决定的。这一观察结果引出了一个归纳论证。首先，如果$e$是一个叶子节点，当且仅当它有一个消失属性（该属性必须是排他的）时，$e$才会进入${F}_{\mathrm{{tmp}}}$，这与所使用的逆拓扑排序无关。对于一个非叶子节点$e$，通过归纳法，一旦我们确定了$e$的每个真后代节点${e}^{\prime }$是否满足${e}^{\prime } \in  {F}_{\text{tmp }}$，那么$e \in  {F}_{\text{tmp }}$是否满足也已经确定。因此，我们得出结论：逆拓扑排序对输出没有影响。

It remains to show that when $G$ is clean, $F$ must include all the raw leaf nodes $e$ of $T$ . If $e$ is not the root of $T$ ,it must have an attribute $X$ absent from the parent node of $e$ (otherwise, $e$ is subsumed by its parent and $G$ is not clean). Similarly,if $e$ is the root of $T$ ,it must have an attribute $X$ absent from its child (there is only one child because $e$ is a raw leaf). In both cases,the attribute $X$ is exclusive at $e$ and will force edge-cover to add $e$ to ${F}_{\mathrm{{tmp}}}$ .

还需要证明的是，当$G$是干净的时，$F$必须包含$T$的所有原始叶子节点$e$。如果$e$不是$T$的根节点，那么它一定有一个属性$X$不存在于$e$的父节点中（否则，$e$被其父节点所包含，且$G$不是干净的）。类似地，如果$e$是$T$的根节点，它一定有一个属性$X$不存在于它的子节点中（因为$e$是一个原始叶子节点，所以只有一个子节点）。在这两种情况下，属性$X$在$e$处是排他的，这将迫使边覆盖（edge - cover）算法将$e$添加到${F}_{\mathrm{{tmp}}}$中。

## B Proof of Lemma 2

## B 引理2的证明

Identify an arbitrary non-leaf node $\widehat{f} \in  F$ such that no other non-leaf node in $F$ is lower than $\widehat{f}$ . The existence of $\widehat{f}$ is guaranteed because $F$ includes the root of $T$ . Consider any child node $e$ of $\widehat{f}$ in $T$ . Since $G$ is clean, $e$ must have an attribute ${A}^{ \circ  }$ that does not appear in $\widehat{f}$ . Let ${f}^{ \circ  }$ be any node in $F$ that contains ${A}^{ \circ  }$ . By the connectedness requirement of acyclicity, ${f}^{ \circ  }$ must be in the subtree of $T$ rooted at $e$ and,therefore,must be a leaf.

确定一个任意的非叶子节点$\widehat{f} \in  F$，使得$F$中没有其他非叶子节点比$\widehat{f}$更低。$\widehat{f}$的存在是有保证的，因为$F$包含$T$的根节点。考虑$T$中$\widehat{f}$的任意一个子节点$e$。由于$G$是干净的，$e$一定有一个属性${A}^{ \circ  }$不出现在$\widehat{f}$中。设${f}^{ \circ  }$是$F$中包含${A}^{ \circ  }$的任意一个节点。根据无环性的连通性要求，${f}^{ \circ  }$必须在以$e$为根的$T$的子树中，因此它一定是一个叶子节点。

We argue that ${f}^{ \circ  }$ is an anchor leaf. The signature path of ${f}^{ \circ  }$ includes all the nodes on the path from $e$ to ${f}^{ \circ  }$ . Because ${A}^{ \circ  } \in  e$ and ${A}^{ \circ  } \in  {f}^{ \circ  },{A}^{ \circ  }$ must appear in all the nodes on the path (connectedness requirement) and is thus an anchor attribute of ${f}^{ \circ  }$ .

我们认为${f}^{ \circ  }$是一个锚定叶子节点。${f}^{ \circ  }$的签名路径包含从$e$到${f}^{ \circ  }$路径上的所有节点。因为${A}^{ \circ  } \in  e$和${A}^{ \circ  } \in  {f}^{ \circ  },{A}^{ \circ  }$必须出现在路径上的所有节点中（连通性要求），因此它是${f}^{ \circ  }$的一个锚定属性。

## C Proof of Lemma 3

## C 引理3的证明

## C. $1\;\operatorname{map}\left( {f}^{ \circ  }\right)$ Subsumed in ${G}^{\prime }$

## C. $1\;\operatorname{map}\left( {f}^{ \circ  }\right)$ 被包含于 ${G}^{\prime }$

Let $\widehat{e}$ be the parent of ${f}^{ \circ  }$ in $T$ . If $\operatorname{map}\left( {f}^{ \circ  }\right)  = {f}^{ \circ  } \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ is subsumed in ${G}^{\prime }$ ,then $\operatorname{map}\left( {f}^{ \circ  }\right)$ must be a subset of $\operatorname{map}\left( \widehat{e}\right)$ ,which indicates ${A}^{ \circ  } \notin  \widehat{e}$ (otherwise, ${f}^{ \circ  } \subseteq  \widehat{e}$ and $G$ is not clean). Because ${A}^{ \circ  }$ needs to appear in all the nodes of sigpath $\left( {{f}^{ \circ  },T}\right) ,{A}^{ \circ  } \notin  \widehat{e}$ indicates that $\operatorname{signath}\left( {{f}^{ \circ  },T}\right)$ has only a single node ${f}^{ \circ  }$ . It thus follows that $\widehat{e} \in  F$ and ${A}^{ \circ  }$ is an exclusive attribute in ${f}^{ \circ  }$ . Hence,the removal of ${A}^{ \circ  }$ does not affect any hyperedge except ${f}^{ \circ  }$ .

设$\widehat{e}$是$T$中${f}^{ \circ  }$的父节点。如果$\operatorname{map}\left( {f}^{ \circ  }\right)  = {f}^{ \circ  } \smallsetminus  \left\{  {A}^{ \circ  }\right\}$被包含在${G}^{\prime }$中，那么$\operatorname{map}\left( {f}^{ \circ  }\right)$必定是$\operatorname{map}\left( \widehat{e}\right)$的一个子集，这意味着${A}^{ \circ  } \notin  \widehat{e}$（否则，${f}^{ \circ  } \subseteq  \widehat{e}$和$G$就不清晰了）。因为${A}^{ \circ  }$需要出现在签名路径（sigpath）的所有节点中，$\left( {{f}^{ \circ  },T}\right) ,{A}^{ \circ  } \notin  \widehat{e}$表明$\operatorname{signath}\left( {{f}^{ \circ  },T}\right)$只有一个节点${f}^{ \circ  }$。因此，$\widehat{e} \in  F$且${A}^{ \circ  }$是${f}^{ \circ  }$中的一个排他属性。所以，移除${A}^{ \circ  }$除了${f}^{ \circ  }$之外不会影响任何超边。

Next,we show that ${F}^{\prime } = F \smallsetminus  \left\{  {f}^{ \circ  }\right\}$ is the CEC of ${G}^{\prime }$ induced by ${T}^{\prime }$ . It suffices to prove that ${F}^{\prime }$ is the output of edge-cover $\left( {T}^{\prime }\right)$ on some reverse topological order of ${T}^{\prime }$ . For this purpose,consider ${\sigma }_{0}$ as an arbitrary reverse topological order of $T$ where $\widehat{e}$ succeeds ${f}^{ \circ  }$ . Let ${\sigma }_{1}$ be the sequence obtained by removing ${f}^{ \circ  }$ from ${\sigma }_{0};{\sigma }_{1}$ must be a reverse topological order of ${T}^{\prime }$ . Let ${e}_{\text{before }}$ be the node preceding ${f}^{ \circ  }$ in ${\sigma }_{0}$ (and hence preceding $\widehat{e}$ in ${\sigma }_{1}$ ); define ${e}_{\text{before }}$ to be a dummy node if ${f}^{ \circ  }$ is the first in ${\sigma }_{0}$ .

接下来，我们证明${F}^{\prime } = F \smallsetminus  \left\{  {f}^{ \circ  }\right\}$是由${T}^{\prime }$诱导的${G}^{\prime }$的最小边覆盖（CEC）。只需证明${F}^{\prime }$是在${T}^{\prime }$的某个逆拓扑序上执行边覆盖算法$\left( {T}^{\prime }\right)$的输出即可。为此，考虑${\sigma }_{0}$作为$T$的任意一个逆拓扑序，其中$\widehat{e}$在${f}^{ \circ  }$之后。设${\sigma }_{1}$是从${\sigma }_{0};{\sigma }_{1}$中移除${f}^{ \circ  }$后得到的序列，它必定是${T}^{\prime }$的一个逆拓扑序。设${e}_{\text{before }}$是${\sigma }_{0}$中${f}^{ \circ  }$的前一个节点（因此也是${\sigma }_{1}$中$\widehat{e}$的前一个节点）；如果${f}^{ \circ  }$是${\sigma }_{0}$中的第一个节点，则将${e}_{\text{before }}$定义为一个虚拟节点。

Let us compare the execution of edge-cover(T)on ${\sigma }_{0}$ to that of edge-cover $\left( {T}^{\prime }\right)$ on ${\sigma }_{1}$ . The two executions are identical till the moment when ${e}_{\text{before }}$ has been processed. By the fact that edge-cover(T)adds $\widehat{e}$ to ${F}_{\text{tmp }}$ (we have proved earlier $\widehat{e} \in  F$ ), $\widehat{e}$ has a disappearing attribute not covered by ${F}_{\text{tmp }}$ when $\widehat{e}$ is processed. Hence,when $\widehat{e}$ is processed by edge-cover $\left( {T}^{\prime }\right)$ ,it must also have a disappearing attribute not covered by ${F}_{\mathrm{{tmp}}}$ and thus is added to ${F}_{\mathrm{{tmp}}}$ . The rest execution of edge-cover(T)is the same as that of edge-cover $\left( {T}^{\prime }\right)$ because every non-exclusive attribute of ${f}^{ \circ  }$ is in $\widehat{e}$ . Therefore,the output of edge-cover $\left( {T}^{\prime }\right)$ is the same as that of edge-cover(T),except that the former does not include ${f}^{ \circ  }$ .

让我们比较在${\sigma }_{0}$上执行边覆盖算法edge - cover(T)与在${\sigma }_{1}$上执行边覆盖算法edge - cover $\left( {T}^{\prime }\right)$的情况。在处理完${e}_{\text{before }}$之前，这两次执行是完全相同的。由于边覆盖算法edge - cover(T)会将$\widehat{e}$添加到${F}_{\text{tmp }}$中（我们之前已经证明了$\widehat{e} \in  F$），当处理$\widehat{e}$时，$\widehat{e}$有一个未被${F}_{\text{tmp }}$覆盖的消失属性。因此，当边覆盖算法edge - cover $\left( {T}^{\prime }\right)$处理$\widehat{e}$时，它也一定有一个未被${F}_{\mathrm{{tmp}}}$覆盖的消失属性，从而会被添加到${F}_{\mathrm{{tmp}}}$中。边覆盖算法edge - cover(T)的其余执行过程与边覆盖算法edge - cover $\left( {T}^{\prime }\right)$相同，因为${f}^{ \circ  }$的每个非排他属性都在$\widehat{e}$中。因此，边覆盖算法edge - cover $\left( {T}^{\prime }\right)$的输出与边覆盖算法edge - cover(T)的输出相同，只是前者不包含${f}^{ \circ  }$。

## C. $2\;\operatorname{map}\left( {f}^{ \circ  }\right)$ Not Subsumed in ${G}^{\prime }$

## C. $2\;\operatorname{map}\left( {f}^{ \circ  }\right)$ 不被${G}^{\prime }$包含

Let ${\sigma }_{0} = \left( {{e}_{1},{e}_{2},\ldots ,{e}_{\left| E\right| }}\right)$ be an arbitrary reverse topological order of $T$ . Define ${e}_{i}^{\prime } = \operatorname{map}\left( {e}_{i}\right)  =$ ${e}_{i} \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ for $i \in  \left\lbrack  \left| E\right| \right\rbrack$ . The sequence ${\sigma }_{1} = \left( {{e}_{1}^{\prime },{e}_{2}^{\prime },\ldots ,{e}_{\left| E\right| }^{\prime }}\right)$ is a reverse topological order of ${T}^{\prime }$ . We will compare the execution of edge-cover(T)on ${\sigma }_{0}$ to that of edge-cover $\left( {T}^{\prime }\right)$ on ${\sigma }_{1}$ . Define ${F}_{0}\left( {e}_{i}\right)$ (resp., ${F}_{1}\left( {e}_{i}^{\prime }\right)$ ) as the content of ${F}_{\text{tmp }}$ after edge-cover(T)(resp.,edge-cover $\left( {T}^{\prime }\right)$ ) has processed ${e}_{i}$ $\left( {\text{resp.,}{e}_{i}^{\prime }}\right)$ .

设${\sigma }_{0} = \left( {{e}_{1},{e}_{2},\ldots ,{e}_{\left| E\right| }}\right)$是$T$的任意一个逆拓扑序。对于$i \in  \left\lbrack  \left| E\right| \right\rbrack$，定义${e}_{i}^{\prime } = \operatorname{map}\left( {e}_{i}\right)  =$ ${e}_{i} \smallsetminus  \left\{  {A}^{ \circ  }\right\}$。序列${\sigma }_{1} = \left( {{e}_{1}^{\prime },{e}_{2}^{\prime },\ldots ,{e}_{\left| E\right| }^{\prime }}\right)$是${T}^{\prime }$的一个逆拓扑序。我们将比较在${\sigma }_{0}$上执行边覆盖算法edge - cover(T)与在${\sigma }_{1}$上执行边覆盖算法edge - cover $\left( {T}^{\prime }\right)$的情况。定义${F}_{0}\left( {e}_{i}\right)$（分别地，${F}_{1}\left( {e}_{i}^{\prime }\right)$）为在边覆盖算法edge - cover(T)（分别地，边覆盖算法edge - cover $\left( {T}^{\prime }\right)$）处理完${e}_{i}$ $\left( {\text{resp.,}{e}_{i}^{\prime }}\right)$之后${F}_{\text{tmp }}$的内容。

Claim 1: For any leaf $e$ of $T$ ,edge-cover $\left( {T}^{\prime }\right)$ must add ${e}^{\prime } = \operatorname{map}\left( e\right)$ to ${F}_{\text{tmp }}$ .

命题1：对于$T$的任意叶子节点$e$，边覆盖算法edge - cover $\left( {T}^{\prime }\right)$必须将${e}^{\prime } = \operatorname{map}\left( e\right)$添加到${F}_{\text{tmp }}$中。

Let us prove the claim. Because $e$ is a leaf of $T$ and $G$ is clean, $e$ must have an exclusive attribute $X$ . If edge-cover does not add ${e}^{\prime }$ to ${F}_{\mathrm{{tmp}}},{e}^{\prime }$ has no exclusive attributes in ${T}^{\prime }$ . This implies $X = {A}^{ \circ  }$ , which further implies ${f}^{ \circ  } = e$ (otherwise, ${A}^{ \circ  }$ appears in two distinct nodes and cannot be exclusive). However,in that case, ${e}^{\prime }$ must contain an exclusive attribute in ${T}^{\prime }\left( {{e}^{\prime } = \operatorname{map}\left( {f}^{ \circ  }\right) }\right.$ is not subsumed in ${G}^{\prime }$ ). We thus have reached a contradiction.

让我们来证明这个命题。因为$e$是$T$的一个叶子节点，且$G$是干净的，所以$e$必定有一个排他属性$X$。如果边覆盖没有将${e}^{\prime }$添加到${F}_{\mathrm{{tmp}}},{e}^{\prime }$，那么${F}_{\mathrm{{tmp}}},{e}^{\prime }$在${T}^{\prime }$中就没有排他属性。这意味着$X = {A}^{ \circ  }$，进而意味着${f}^{ \circ  } = e$（否则，${A}^{ \circ  }$会出现在两个不同的节点中，就不能是排他的了）。然而，在这种情况下，${e}^{\prime }$在${T}^{\prime }\left( {{e}^{\prime } = \operatorname{map}\left( {f}^{ \circ  }\right) }\right.$中必定包含一个排他属性（该属性在${G}^{\prime }$中未被包含）。这样我们就得出了矛盾。

To establish Lemma 3, it suffices to prove:

要证明引理3，只需证明：

Claim 2: For each $i,{e}_{i} \in  {F}_{0}\left( {e}_{i}\right)$ if and only if ${e}_{i}^{\prime } \in  {F}_{1}\left( {e}_{i}^{\prime }\right)$ .

命题2：对于每个$i,{e}_{i} \in  {F}_{0}\left( {e}_{i}\right)$，当且仅当${e}_{i}^{\prime } \in  {F}_{1}\left( {e}_{i}^{\prime }\right)$。

We prove the claim by induction on $i$ . Because ${e}_{1}$ is a leaf of $T$ ,Lemma 1 and Claim 1 guarantee ${e}_{1} \in  {F}_{0}\left( {e}_{1}\right)$ and ${e}_{1}^{\prime } \in  {F}_{1}\left( {e}_{1}^{\prime }\right)$ ,respectively. Thus,Claim 2 holds for $i = 1$ .

我们通过对$i$进行归纳来证明这个命题。因为${e}_{1}$是$T$的一个叶子节点，引理1和命题1分别保证了${e}_{1} \in  {F}_{0}\left( {e}_{1}\right)$和${e}_{1}^{\prime } \in  {F}_{1}\left( {e}_{1}^{\prime }\right)$。因此，命题2对于$i = 1$成立。

Next,we prove the correctness on $i > 1$ ,assuming that it holds on ${e}_{i - 1}$ and ${e}_{i - 1}^{\prime }$ . The inductive assumption implies that ${F}_{0}\left( {e}_{i - 1}\right)$ covers an attribute $X \neq  {A}^{ \circ  }$ if and only if ${F}_{1}\left( {e}_{i - 1}^{\prime }\right)$ covers $X$ . If ${e}_{i} \notin  {F}_{0}\left( {e}_{i}\right)$ ,every disappearing attribute of ${e}_{i}$ must be covered by ${F}_{0}\left( {e}_{i - 1}\right)$ . Hence, ${F}_{1}\left( {e}_{i - 1}^{\prime }\right)$ must cover all the disappearing attributes of ${e}_{i}^{\prime }$ and thus ${e}_{i}^{\prime } \notin  {F}_{1}\left( {e}_{i}^{\prime }\right)$ .

接下来，我们假设命题在${e}_{i - 1}$和${e}_{i - 1}^{\prime }$上成立，来证明其在$i > 1$上的正确性。归纳假设意味着${F}_{0}\left( {e}_{i - 1}\right)$覆盖一个属性$X \neq  {A}^{ \circ  }$当且仅当${F}_{1}\left( {e}_{i - 1}^{\prime }\right)$覆盖$X$。如果${e}_{i} \notin  {F}_{0}\left( {e}_{i}\right)$，${e}_{i}$的每个消失属性都必须被${F}_{0}\left( {e}_{i - 1}\right)$覆盖。因此，${F}_{1}\left( {e}_{i - 1}^{\prime }\right)$必须覆盖${e}_{i}^{\prime }$的所有消失属性，从而${e}_{i}^{\prime } \notin  {F}_{1}\left( {e}_{i}^{\prime }\right)$。

The rest of the proof assumes ${e}_{i} \in  {F}_{0}\left( {e}_{i}\right)$ ,i.e., ${e}_{i}$ has a disappearing attribute $X$ not covered by ${F}_{0}\left( {e}_{i - 1}\right)$ . If $X \neq  {A}^{ \circ  },X$ is a disappearing attribute in ${e}_{i}^{\prime }$ not covered by ${F}_{1}\left( {e}_{i - 1}^{\prime }\right)$ and thus ${e}_{i}^{\prime } \in  {F}_{1}\left( {e}_{i}^{\prime }\right)$ . It remains to discuss the scenario $X = {A}^{ \circ  }$ . As ${A}^{ \circ  }$ is disappearing at ${e}_{i},{A}^{ \circ  }$ cannot exist in the parent of ${e}_{i}$ . On the other hand,because ${A}^{ \circ  } \in  {f}^{ \circ  }$ ,acyclicity’s connectedness requirement forces ${f}^{ \circ  }$ to be a descendant of ${e}_{i}$ . We can safely conclude that ${f}^{ \circ  } = {e}_{i}$ ; otherwise,the leaf ${f}^{ \circ  }$ is processed before ${e}_{i}$ and must exist in ${F}_{0}\left( {e}_{i - 1}\right)$ (Lemma 1),contradicting the fact that ${A}^{ \circ  }$ is not covered by ${F}_{0}\left( {e}_{i - 1}\right)$ . Then, ${e}_{i}^{\prime } \in  {F}_{1}\left( {e}_{i}^{\prime }\right)$ follows from Claim 1 .

证明的其余部分假设${e}_{i} \in  {F}_{0}\left( {e}_{i}\right)$，即${e}_{i}$有一个消失属性$X$未被${F}_{0}\left( {e}_{i - 1}\right)$覆盖。如果$X \neq  {A}^{ \circ  },X$是${e}_{i}^{\prime }$中未被${F}_{1}\left( {e}_{i - 1}^{\prime }\right)$覆盖的消失属性，因此${e}_{i}^{\prime } \in  {F}_{1}\left( {e}_{i}^{\prime }\right)$。接下来需要讨论$X = {A}^{ \circ  }$的情况。由于${A}^{ \circ  }$在${e}_{i},{A}^{ \circ  }$处消失，${e}_{i}$的父节点中不可能存在${A}^{ \circ  }$。另一方面，因为${A}^{ \circ  } \in  {f}^{ \circ  }$，无环性的连通性要求迫使${f}^{ \circ  }$是${e}_{i}$的后代。我们可以有把握地得出${f}^{ \circ  } = {e}_{i}$；否则，叶子节点${f}^{ \circ  }$会在${e}_{i}$之前被处理，并且必须存在于${F}_{0}\left( {e}_{i - 1}\right)$中（引理1），这与${A}^{ \circ  }$未被${F}_{0}\left( {e}_{i - 1}\right)$覆盖这一事实相矛盾。然后，根据命题1可得${e}_{i}^{\prime } \in  {F}_{1}\left( {e}_{i}^{\prime }\right)$。

## D Proof of Lemma 4

## D 引理4的证明

Define $e = {ma}{p}^{-1}\left( {e}^{\prime }\right)$ . Because ${e}^{\prime }$ is subsumed,we know that $e$ must contain ${A}^{ \circ  }$ (otherwise, $e$ is subsumed and $G$ is not clean). In other words, $e = {e}^{\prime } \cup  \left\{  {A}^{ \circ  }\right\}$ . Furthermore,if $e = {f}^{ \circ  }$ ,then $\operatorname{map}\left( {f}^{ \circ  }\right)  = \operatorname{map}\left( e\right)  = {e}^{\prime }$ is subsumed in ${G}^{\prime }$ ,in which case we must have ${e}^{\prime } \notin  {F}^{\prime }$ (by the way we define ${F}^{\prime }$ in (2)). Next,we assume $e \neq  {f}^{ \circ  }$ . To complete the proof,it suffices to show that $e \notin  F$ , where $F$ is the CEC of $G$ induced by $T$ .

定义$e = {ma}{p}^{-1}\left( {e}^{\prime }\right)$。因为${e}^{\prime }$被包含，我们知道$e$必须包含${A}^{ \circ  }$（否则，$e$被包含且$G$不干净）。换句话说，$e = {e}^{\prime } \cup  \left\{  {A}^{ \circ  }\right\}$。此外，如果$e = {f}^{ \circ  }$，那么$\operatorname{map}\left( {f}^{ \circ  }\right)  = \operatorname{map}\left( e\right)  = {e}^{\prime }$在${G}^{\prime }$中被包含，在这种情况下，我们必须有${e}^{\prime } \notin  {F}^{\prime }$（根据我们在(2)中对${F}^{\prime }$的定义）。接下来，我们假设$e \neq  {f}^{ \circ  }$。为了完成证明，只需证明$e \notin  F$，其中$F$是由$T$诱导的$G$的CEC（连通等价类）。

Let $\widehat{f}$ be the lowest proper ancestor of ${f}^{ \circ  }$ in $F$ (here,"ancestor" is defined with respect to $T$ ). By definition of ${A}^{ \circ  },{A}^{ \circ  } \notin  \widehat{f}$ . Because ${A}^{ \circ  } \in  {f}^{ \circ  }$ and ${A}^{ \circ  } \in  e,e$ must be a proper descendant of $\widehat{f}$ in $T$ . Assume,for contradiction purposes,that $e \in  F$ . As $\widehat{f}$ (by definition of ${f}^{ \circ  }$ ) cannot have any non-leaf proper descendant in $F,e$ must be a leaf of $T$ .

设$\widehat{f}$是${f}^{ \circ  }$在$F$中的最低真祖先（这里，“祖先”是相对于$T$定义的）。根据${A}^{ \circ  },{A}^{ \circ  } \notin  \widehat{f}$的定义。因为${A}^{ \circ  } \in  {f}^{ \circ  }$且${A}^{ \circ  } \in  e,e$必须是$\widehat{f}$在$T$中的真后代。为了推出矛盾，假设$e \in  F$。由于$\widehat{f}$（根据${f}^{ \circ  }$的定义）在$F,e$中不能有任何非叶子真后代，$F,e$必须是$T$的叶子节点。

Because ${A}^{ \circ  }$ appears in two distinct nodes $e$ and ${f}^{ \circ  }$ ,acyclicity’s connected requirement demands that ${A}^{ \circ  }$ should also exist in the parent $\widehat{e}$ of $e$ . Because $G$ is clean,we know that $e$ must have at least one attribute $X$ that does not appear in $\widehat{e}$ and thus must be exclusive. It follows that $X \neq  {A}^{ \circ  }$ However,in that case ${e}^{\prime } = e \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ contains $X$ and thus cannot be subsumed in ${G}^{\prime }(X$ remains exclusive in ${G}^{\prime }$ ),giving a contradiction.

因为 ${A}^{ \circ  }$ 出现在两个不同的节点 $e$ 和 ${f}^{ \circ  }$ 中，无环性的连通要求规定 ${A}^{ \circ  }$ 也应存在于 $e$ 的父节点 $\widehat{e}$ 中。由于 $G$ 是干净的，我们知道 $e$ 必须至少有一个属性 $X$ 不出现在 $\widehat{e}$ 中，因此该属性必定是排他的。由此可得 $X \neq  {A}^{ \circ  }$ 然而，在这种情况下，${e}^{\prime } = e \smallsetminus  \left\{  {A}^{ \circ  }\right\}$ 包含 $X$，因此不能被 ${G}^{\prime }(X$ 所包含（$X$ 在 ${G}^{\prime }$ 中仍然是排他的），这就产生了矛盾。

## E Proof of Lemma 5

## 引理 5 的证明

We discuss only the scenario where $\operatorname{map}\left( {f}^{ \circ  }\right)$ is not subsumed in ${G}^{\prime }$ (the opposite case is easy and omitted). Our proof will establish a stronger claim:

我们仅讨论 $\operatorname{map}\left( {f}^{ \circ  }\right)$ 不被 ${G}^{\prime }$ 所包含的情况（相反的情况很简单，在此省略）。我们的证明将确立一个更强的命题：

Claim: ${F}^{ * } = {F}^{\prime }$ is the CEC of ${G}^{ * }$ induced by ${T}^{ * }$ every time Line 2 of cleanse is executed.

命题：每次执行清理（cleanse）操作的第 2 行时，${F}^{ * } = {F}^{\prime }$ 是由 ${T}^{ * }$ 诱导的 ${G}^{ * }$ 的最小边覆盖（CEC，Connected Edge Cover）。

${G}^{ * } = {G}^{\prime }$ and ${T}^{ * } = {T}^{\prime }$ at Line 1. ${F}^{ * } = {F}^{\prime }$ is the CEC of ${G}^{ * }$ induced by ${T}^{ * }$ at this moment (Lemma 3). Hence, the claim holds on the first execution of Line 2.

在第 1 行有 ${G}^{ * } = {G}^{\prime }$ 和 ${T}^{ * } = {T}^{\prime }$。此时，${F}^{ * } = {F}^{\prime }$ 是由 ${T}^{ * }$ 诱导的 ${G}^{ * }$ 的最小边覆盖（引理 3）。因此，该命题在第 2 行首次执行时成立。

Inductively, assuming that the claim holds currently, we will show that it still does after cleanse deletes the next ${e}_{\text{small }}$ from ${G}^{ * }$ . Let ${G}_{0}^{ * }$ and ${T}_{0}^{ * }$ (resp., ${G}_{1}^{ * }$ and ${T}_{1}^{ * }$ ) be the ${G}^{ * }$ and ${T}^{ * }$ before (resp., after) the deletion of ${e}_{\text{small }}$ ,respectively. The fact ${e}_{\text{small }}$ being subsumed in ${G}^{ * }$ suggests ${e}_{\text{small }}$ being subsumed in ${G}^{\prime }$ . By Lemma $4,{e}_{\text{small }} \notin  {F}^{\prime } = {F}^{ * }$ .

通过归纳法，假设该命题当前成立，我们将证明在清理操作从 ${G}^{ * }$ 中删除下一个 ${e}_{\text{small }}$ 之后，该命题仍然成立。设 ${G}_{0}^{ * }$ 和 ${T}_{0}^{ * }$（分别地，${G}_{1}^{ * }$ 和 ${T}_{1}^{ * }$）分别是删除 ${e}_{\text{small }}$ 之前（分别地，之后）的 ${G}^{ * }$ 和 ${T}^{ * }$。${e}_{\text{small }}$ 被 ${G}^{ * }$ 所包含这一事实表明 ${e}_{\text{small }}$ 被 ${G}^{\prime }$ 所包含。根据引理 $4,{e}_{\text{small }} \notin  {F}^{\prime } = {F}^{ * }$。

Case 1: ${e}_{\text{big }}$ parents ${e}_{\text{small }}$ . Let ${\sigma }_{0}$ be a reverse topological order of ${T}_{0}^{ * }$ where ${e}_{\text{big }}$ succeeds ${e}_{\text{small }}$ . As ${F}^{ * }$ is the CEC of ${G}_{0}^{ * }$ induced by ${T}_{0}^{ * }$ ,edge-cover $\left( {T}_{0}^{ * }\right)$ produces ${F}^{ * }$ if executed on ${\sigma }_{0}$ (Lemma 1).

情况 1：${e}_{\text{big }}$ 是 ${e}_{\text{small }}$ 的父节点。设 ${\sigma }_{0}$ 是 ${T}_{0}^{ * }$ 的一个逆拓扑排序，其中 ${e}_{\text{big }}$ 在 ${e}_{\text{small }}$ 之后。由于 ${F}^{ * }$ 是由 ${T}_{0}^{ * }$ 诱导的 ${G}_{0}^{ * }$ 的最小边覆盖，如果在 ${\sigma }_{0}$ 上执行边覆盖操作 $\left( {T}_{0}^{ * }\right)$，则会产生 ${F}^{ * }$（引理 1）。

Let ${\sigma }_{1}$ be a copy of ${\sigma }_{0}$ but with ${e}_{\text{small }}$ removed; ${\sigma }_{1}$ is a reverse topological order of ${T}_{1}^{ * }$ . Every node in ${T}_{1}^{ * }$ retains the same disappearing attributes as in ${T}_{0}^{ * }$ (see Figure 3a). For every node $e \neq  {e}_{\text{small }}$ ,running edge-cover $\left( {T}_{1}^{ * }\right)$ on ${\sigma }_{1}$ has the same effect as running edge-cover $\left( {T}_{0}^{ * }\right)$ on ${\sigma }_{0}$ . Therefore,edge-cover $\left( {T}_{1}^{ * }\right)$ also outputs ${F}^{ * }$ .

令${\sigma }_{1}$为${\sigma }_{0}$的一个副本，但移除了${e}_{\text{small }}$；${\sigma }_{1}$是${T}_{1}^{ * }$的一个逆拓扑序。${T}_{1}^{ * }$中的每个节点保留与${T}_{0}^{ * }$中相同的消失属性（见图3a）。对于每个节点$e \neq  {e}_{\text{small }}$，在${\sigma }_{1}$上运行边覆盖$\left( {T}_{1}^{ * }\right)$与在${\sigma }_{0}$上运行边覆盖$\left( {T}_{0}^{ * }\right)$具有相同的效果。因此，边覆盖$\left( {T}_{1}^{ * }\right)$也输出${F}^{ * }$。

Case 2: ${e}_{\text{small }}$ parents ${e}_{\text{big }}$ . Let ${\sigma }_{0}$ be a reverse topological order of ${T}_{0}^{ * }$ where ${e}_{\text{small }}$ succeeds ${e}_{\text{big }}$ . Let ${\sigma }_{1}$ be a copy of ${\sigma }_{0}$ but with ${e}_{\text{small }}$ removed; ${\sigma }_{1}$ is a reverse topological order of ${T}_{1}^{ * }$ . We will argue that running edge-cover $\left( {T}_{1}^{ * }\right)$ on ${\sigma }_{1}$ returns ${F}^{ * }$ .

情况2：${e}_{\text{small }}$是${e}_{\text{big }}$的父节点。令${\sigma }_{0}$为${T}_{0}^{ * }$的一个逆拓扑序，其中${e}_{\text{small }}$在${e}_{\text{big }}$之后。令${\sigma }_{1}$为${\sigma }_{0}$的一个副本，但移除了${e}_{\text{small }}$；${\sigma }_{1}$是${T}_{1}^{ * }$的一个逆拓扑序。我们将证明在${\sigma }_{1}$上运行边覆盖$\left( {T}_{1}^{ * }\right)$会返回${F}^{ * }$。

The reader should note several preliminary facts about disappearing attributes. If an attribute has ${e}_{\text{small }}$ as the summit in ${T}_{0}^{ * }$ ,the attribute’s summit in ${T}_{1}^{ * }$ becomes ${e}_{\text{big }}$ (see Figure 3b). If an attribute has $e \neq  {e}_{\text{small }}$ as the summit in ${T}_{0}^{ * }$ ,its summit in ${T}_{1}^{ * }$ is still $e$ . Hence,every node in ${T}_{1}^{ * }$ except ${e}_{\text{big }}$ retains the same disappearing attributes as in ${T}_{0}^{ * }$ ,whereas the disappearing attributes of ${e}_{\text{big }}$ in ${T}_{1}^{ * }$ contain those of ${e}_{\text{big }}$ and ${e}_{\text{small }}$ in ${T}_{0}^{ * }$ .

读者应该注意关于消失属性的几个初步事实。如果一个属性在${T}_{0}^{ * }$中以${e}_{\text{small }}$为顶点，那么该属性在${T}_{1}^{ * }$中的顶点变为${e}_{\text{big }}$（见图3b）。如果一个属性在${T}_{0}^{ * }$中以$e \neq  {e}_{\text{small }}$为顶点，那么它在${T}_{1}^{ * }$中的顶点仍然是$e$。因此，${T}_{1}^{ * }$中除${e}_{\text{big }}$之外的每个节点保留与${T}_{0}^{ * }$中相同的消失属性，而${e}_{\text{big }}$在${T}_{1}^{ * }$中的消失属性包含${e}_{\text{big }}$和${e}_{\text{small }}$在${T}_{0}^{ * }$中的消失属性。

For each node $e$ in ${\sigma }_{0}$ (resp. ${\sigma }_{1}$ ),denote by ${F}_{0}\left( e\right)$ (resp. ${F}_{1}\left( e\right)$ ) the content of ${F}_{\text{tmp }}$ after edge-cover $\left( {T}_{0}^{ * }\right)$ (resp. edge-cover $\left( {T}_{1}^{ * }\right)$ ) has processed $e$ . Let ${e}_{\text{before }}$ be the node before ${e}_{\text{big }}$ in ${\sigma }_{0}{}^{.13}$ It is easy to see that edge-cover $\left( {T}_{0}^{ * }\right)$ and edge-cover $\left( {T}_{1}^{ * }\right)$ behave the same way until finishing with ${e}_{\text{before }}$ , which gives ${F}_{0}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{before }}\right)$ . It must hold that ${e}_{\text{small }} \notin  {F}_{0}\left( {e}_{\text{small }}\right) {.}^{14}$ Two possibilities apply to ${e}_{\mathrm{{big}}}$ :

对于${\sigma }_{0}$（分别地，${\sigma }_{1}$）中的每个节点$e$，用${F}_{0}\left( e\right)$（分别地，${F}_{1}\left( e\right)$）表示在边覆盖$\left( {T}_{0}^{ * }\right)$（分别地，边覆盖$\left( {T}_{1}^{ * }\right)$）处理完$e$之后${F}_{\text{tmp }}$的内容。令${e}_{\text{before }}$为${\sigma }_{0}{}^{.13}$中${e}_{\text{big }}$之前的节点。很容易看出，边覆盖$\left( {T}_{0}^{ * }\right)$和边覆盖$\left( {T}_{1}^{ * }\right)$在处理完${e}_{\text{before }}$之前的行为是相同的，这得到${F}_{0}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{before }}\right)$。必然有${e}_{\text{small }} \notin  {F}_{0}\left( {e}_{\text{small }}\right) {.}^{14}$。对于${e}_{\mathrm{{big}}}$有两种可能的情况：

1. ${e}_{\text{big }} \in  {F}_{0}\left( {e}_{\text{big }}\right)$ . Hence, ${e}_{\text{big }}$ has a disappearing attribute in ${T}_{0}^{ * }$ not covered by ${F}_{0}\left( {e}_{\text{before }}\right)$ . This means that ${e}_{\text{big }}$ also has a disappearing attribute in ${T}_{1}^{ * }$ not covered by ${F}_{1}\left( {e}_{\text{before }}\right)  = {F}_{0}\left( {e}_{\text{before }}\right)$ . It follows that ${e}_{\text{big }} \in  {F}_{1}\left( {e}_{\text{big }}\right)$ ,meaning ${F}_{1}\left( {e}_{\text{big }}\right)  = {F}_{0}\left( {e}_{\text{big }}\right)  = {F}_{0}\left( {e}_{\text{small }}\right)$ .

1. ${e}_{\text{big }} \in  {F}_{0}\left( {e}_{\text{big }}\right)$ 。因此，${e}_{\text{big }}$ 在 ${T}_{0}^{ * }$ 中有一个未被 ${F}_{0}\left( {e}_{\text{before }}\right)$ 覆盖的消失属性。这意味着 ${e}_{\text{big }}$ 在 ${T}_{1}^{ * }$ 中也有一个未被 ${F}_{1}\left( {e}_{\text{before }}\right)  = {F}_{0}\left( {e}_{\text{before }}\right)$ 覆盖的消失属性。由此可得 ${e}_{\text{big }} \in  {F}_{1}\left( {e}_{\text{big }}\right)$，即 ${F}_{1}\left( {e}_{\text{big }}\right)  = {F}_{0}\left( {e}_{\text{big }}\right)  = {F}_{0}\left( {e}_{\text{small }}\right)$ 。

2. ${e}_{\text{big }} \notin  {F}_{0}\left( {e}_{\text{big }}\right)$ . All the disappearing attributes of ${e}_{\text{big }}$ and ${e}_{\text{small }}$ in ${T}_{0}^{ * }$ are covered by ${F}_{0}\left( {e}_{\text{before }}\right)$ . Hence,the disappearing attributes of ${e}_{\text{big }}$ in ${T}_{1}^{ * }$ are covered by ${F}_{1}\left( {e}_{\text{before }}\right)  = {F}_{0}\left( {e}_{\text{before }}\right)$ . Therefore, ${e}_{\text{big }} \notin  {F}_{1}\left( {e}_{\text{big }}\right)$ ,meaning ${F}_{0}\left( {e}_{\text{small }}\right)  = {F}_{0}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{big }}\right)$ .

2. ${e}_{\text{big }} \notin  {F}_{0}\left( {e}_{\text{big }}\right)$ 。${e}_{\text{big }}$ 和 ${e}_{\text{small }}$ 在 ${T}_{0}^{ * }$ 中的所有消失属性都被 ${F}_{0}\left( {e}_{\text{before }}\right)$ 覆盖。因此，${e}_{\text{big }}$ 在 ${T}_{1}^{ * }$ 中的消失属性被 ${F}_{1}\left( {e}_{\text{before }}\right)  = {F}_{0}\left( {e}_{\text{before }}\right)$ 覆盖。所以，${e}_{\text{big }} \notin  {F}_{1}\left( {e}_{\text{big }}\right)$，即 ${F}_{0}\left( {e}_{\text{small }}\right)  = {F}_{0}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{big }}\right)$ 。

We now conclude that ${F}_{1}\left( {e}_{\text{big }}\right)  = {F}_{0}\left( {e}_{\text{small }}\right)$ always holds. Every remaining node in ${\sigma }_{0}$ and ${\sigma }_{1}$ has the same disappearing attributes in ${T}_{0}^{ * }$ and ${T}_{1}^{ * }$ . The rest execution of edge-cover $\left( {T}_{0}^{ * }\right)$ is identical to that of edge-cover $\left( {T}_{1}^{ * }\right)$ .

我们现在得出结论，${F}_{1}\left( {e}_{\text{big }}\right)  = {F}_{0}\left( {e}_{\text{small }}\right)$ 始终成立。${\sigma }_{0}$ 和 ${\sigma }_{1}$ 中每个剩余节点在 ${T}_{0}^{ * }$ 和 ${T}_{1}^{ * }$ 中具有相同的消失属性。边覆盖 $\left( {T}_{0}^{ * }\right)$ 的其余执行过程与边覆盖 $\left( {T}_{1}^{ * }\right)$ 的相同。

## F Proof of Lemma 6

## F 引理 6 的证明

We will discuss only the scenario where $\operatorname{map}\left( {f}^{ \circ  }\right)$ is not subsumed (the opposite scenario is easy and omitted).

我们将只讨论 $\operatorname{map}\left( {f}^{ \circ  }\right)$ 未被包含的情况（相反的情况很简单，省略）。

Departing from acyclic queries,let us consider a more general problem on a rooted tree $\mathcal{T}$ where (i) every node is colored black or white,and (ii) the root and all the leaves are black. Denote by $B$ the set of black nodes. Each black node $b \in  B$ is associated with a signature path:

从无环查询出发，让我们考虑一个关于有根树 $\mathcal{T}$ 的更一般的问题，其中 (i) 每个节点被染成黑色或白色，并且 (ii) 根节点和所有叶子节点都是黑色的。用 $B$ 表示黑色节点的集合。每个黑色节点 $b \in  B$ 都关联着一条签名路径：

- If $b$ is the root of $\mathcal{T}$ ,its signature path contains just $b$ itself.

- 如果 $b$ 是 $\mathcal{T}$ 的根节点，其签名路径仅包含 $b$ 本身。

- Otherwise,let $\widehat{b}$ be the lowest ancestor of $b$ among all the nodes in $B$ ; the signature path of $b$ is the set of nodes on the path from $\widehat{b}$ to $b$ ,except $\widehat{b}$ .

- 否则，设 $\widehat{b}$ 是 $B$ 中所有节点里 $b$ 的最低祖先；$b$ 的签名路径是从 $\widehat{b}$ 到 $b$ 的路径上的节点集合，但不包括 $\widehat{b}$ 。

---

<!-- Footnote -->

${}^{13}$ In the special case where ${e}_{\text{big }}$ is the first in ${\sigma }_{0}$ ,define ${e}_{\text{before }}$ as a dummy node with ${F}_{0}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{before }}\right)  = \varnothing$ . ${}^{14}$ Otherwise, ${e}_{\text{small }} \in  {F}^{ * }$ ,contradicting Lemma 4.

${}^{13}$ 在特殊情况下，若 ${e}_{\text{big }}$ 是 ${\sigma }_{0}$ 中的第一个元素，则将 ${e}_{\text{before }}$ 定义为一个具有 ${F}_{0}\left( {e}_{\text{before }}\right)  = {F}_{1}\left( {e}_{\text{before }}\right)  = \varnothing$ 属性的虚拟节点。 ${}^{14}$ 否则， ${e}_{\text{small }} \in  {F}^{ * }$ ，这与引理4矛盾。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: $\Rightarrow$ (b) Type 2 (d) Type 4 (a) Type 1 (c) Type 3 -->

<img src="https://cdn.noedgeai.com/0195ccba-87e9-7860-ae95-5e250889b8c6_23.jpg?x=489&y=194&w=798&h=543&r=0"/>

Figure 6: Four types of contraction

图6：四种类型的收缩

<!-- Media -->

We define four types of contractions:

我们定义四种类型的收缩：

- Type 1: We are given two white nodes ${v}_{1}$ and ${v}_{2}$ such that ${v}_{1}$ parents ${v}_{2}$ . The contraction removes ${v}_{2}$ from $\mathcal{T}$ and makes ${v}_{1}$ the new parent for all the child nodes of ${v}_{2}$ . See Figure 6a.

- 类型1：给定两个白色节点 ${v}_{1}$ 和 ${v}_{2}$ ，使得 ${v}_{1}$ 是 ${v}_{2}$ 的父节点。该收缩操作将 ${v}_{2}$ 从 $\mathcal{T}$ 中移除，并使 ${v}_{1}$ 成为 ${v}_{2}$ 所有子节点的新父节点。见图6a。

- Type 2: We are given two white nodes ${v}_{1}$ and ${v}_{2}$ such that ${v}_{1}$ parents ${v}_{2}$ . The contraction removes ${v}_{1}$ from $\mathcal{T}$ ,makes ${v}_{2}$ the new parent for all the child nodes of ${v}_{1}$ ,and makes ${v}_{2}$ a child of the original parent of ${v}_{1}$ . See Figure 6b.

- 类型2：给定两个白色节点 ${v}_{1}$ 和 ${v}_{2}$ ，使得 ${v}_{1}$ 是 ${v}_{2}$ 的父节点。该收缩操作将 ${v}_{1}$ 从 $\mathcal{T}$ 中移除，使 ${v}_{2}$ 成为 ${v}_{1}$ 所有子节点的新父节点，并使 ${v}_{2}$ 成为 ${v}_{1}$ 原父节点的子节点。见图6b。

- Type 3: Same as Type 1,except that ${v}_{1}$ is black and ${v}_{2}$ is white. See Figure 6c.

- 类型3：与类型1相同，不同之处在于 ${v}_{1}$ 是黑色节点， ${v}_{2}$ 是白色节点。见图6c。

- Type 4: Same as Type 2,except that ${v}_{1}$ is white and ${v}_{2}$ is black. See Figure 6d.

- 类型4：与类型2相同，不同之处在于 ${v}_{1}$ 是白色节点， ${v}_{2}$ 是黑色节点。见图6d。

The facts below are evident:

以下事实显而易见：

- The number of black nodes remains the same after a contraction.

- 收缩操作后，黑色节点的数量保持不变。

- After a contraction, each signature path either remains the same or shrinks.

- 收缩操作后，每个签名路径要么保持不变，要么缩短。

We now draw correspondence between a contraction and a hyperedge deletion in cleanse. $\mathcal{T}$ corresponds to the current hyperedge tree ${T}^{ * }$ in cleanse. The set $B$ of black nodes equals ${F}^{ * } = {F}^{\prime }$ for the entire execution of cleanse. The set $\left\{  {{v}_{1},{v}_{2}}\right\}$ corresponds to $\left\{  {{e}_{\text{small }},{e}_{\text{big }}}\right\}$ . As shown in Lemma 4, ${e}_{\text{small }}$ cannot exist in ${F}^{ * }$ and thus cannot correspond to a black node. If we denote by $C$ (resp., ${C}^{ * }$ ) the set of signature paths at the beginning (resp.,end) of cleanse,each signature path in ${C}^{ * }$ is obtained by continuously shrinking a distinct signature path in $C$ . This implies Lemma 6,noticing that $C = \left\{  {\operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F}\right\}$ and ${C}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}^{ * }}\right)  \mid  {f}^{ * } \in  {F}^{ * }}\right\}$ .

我们现在建立收缩操作与清理过程中超级边删除操作之间的对应关系。 $\mathcal{T}$ 对应清理过程中的当前超级边树 ${T}^{ * }$ 。在整个清理执行过程中，黑色节点的集合 $B$ 等于 ${F}^{ * } = {F}^{\prime }$ 。集合 $\left\{  {{v}_{1},{v}_{2}}\right\}$ 对应于 $\left\{  {{e}_{\text{small }},{e}_{\text{big }}}\right\}$ 。如引理4所示， ${e}_{\text{small }}$ 不可能存在于 ${F}^{ * }$ 中，因此不可能对应于一个黑色节点。如果我们用 $C$ （分别地， ${C}^{ * }$ ）表示清理开始（分别地，结束）时的签名路径集合，那么 ${C}^{ * }$ 中的每个签名路径都是通过不断缩短 $C$ 中一个不同的签名路径得到的。考虑到 $C = \left\{  {\operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F}\right\}$ 和 ${C}^{ * } = \left\{  {\operatorname{sigpath}\left( {{f}^{ * },{T}^{ * }}\right)  \mid  {f}^{ * } \in  {F}^{ * }}\right\}$ ，这意味着引理6成立。

## G Proof of Lemma 7

## G 引理7的证明

We will first prove that,for any $z \in  Z,{F}_{z}^{ * }$ is the CEC of ${G}_{z}^{ * }$ induced by ${T}_{z}^{ * }$ . Let $\widehat{z}$ be the parent of $z$ . Recall that $F$ is the CEC of $G$ induced by $T$ . Consider a reverse topological order ${\sigma }_{z}$ of $T$ satisfying the following condition: a prefix of ${\sigma }_{z}$ is a permutation of the nodes in the subtree of $T$ rooted at $z$ . In other words,in ${\sigma }_{z}$ ,every node in the aforementioned subtree must rank before every node outside the subtree. Define ${\sigma }_{z}^{ * }$ to be the sequence obtained by deleting from ${\sigma }_{z}$ all the nodes $e$ such that $e \neq  \widehat{z}$ and $e$ is outside the subtree of $T$ rooted at $z$ . It is clear that ${\sigma }_{z}^{ * }$ is a reverse topological order of ${T}_{z}^{ * }$ .

我们首先证明，对于任意$z \in  Z,{F}_{z}^{ * }$是由${T}_{z}^{ * }$诱导的${G}_{z}^{ * }$的连通边覆盖（CEC，Connected Edge Cover）。设$\widehat{z}$是$z$的父节点。回顾可知，$F$是由$T$诱导的$G$的连通边覆盖。考虑$T$的一个逆拓扑序${\sigma }_{z}$，它满足以下条件：${\sigma }_{z}$的一个前缀是$T$中以$z$为根的子树中节点的一个排列。换句话说，在${\sigma }_{z}$中，上述子树中的每个节点的排名必须在子树外的每个节点之前。定义${\sigma }_{z}^{ * }$为从${\sigma }_{z}$中删除所有满足$e \neq  \widehat{z}$且$e$在$T$中以$z$为根的子树之外的节点$e$后得到的序列。显然，${\sigma }_{z}^{ * }$是${T}_{z}^{ * }$的一个逆拓扑序。

Let us compare the execution of edge-cover(T)on $\sigma$ to that of edge-cover $\left( {T}_{z}^{ * }\right)$ on ${\sigma }_{z}^{ * }$ . They are exactly the same until $z$ has been processed. Hence,every node in the ${F}_{\text{tmp }}$ of edge-cover(T)at this moment must have been added to ${F}_{\text{tmp }}$ by edge-cover $\left( {T}_{z}^{ * }\right)$ . This means that all the nodes in ${F}_{z}^{ * }$ , except $\widehat{z}$ ,must appear in the final ${F}_{\text{tmp }}$ output by edge-cover $\left( {T}_{z}^{ * }\right)$ . Finally,the final ${F}_{\text{tmp }}$ must also contain $\widehat{z}$ as well due to Lemma 1 (notice that $\widehat{z}$ is a raw leaf of ${T}_{z}^{ * }$ ). This shows that ${F}_{z}^{ * }$ is the CEC of ${G}_{z}^{ * }$ induced by ${T}_{z}^{ * }$ .

让我们比较在$\sigma$上执行边覆盖算法（edge - cover(T)）和在${\sigma }_{z}^{ * }$上执行边覆盖算法$\left( {T}_{z}^{ * }\right)$的情况。在处理完$z$之前，它们的执行过程完全相同。因此，此时边覆盖算法（edge - cover(T)）的${F}_{\text{tmp }}$中的每个节点必定已被边覆盖算法$\left( {T}_{z}^{ * }\right)$添加到${F}_{\text{tmp }}$中。这意味着，除了$\widehat{z}$之外，${F}_{z}^{ * }$中的所有节点都必须出现在边覆盖算法$\left( {T}_{z}^{ * }\right)$最终输出的${F}_{\text{tmp }}$中。最后，根据引理1，最终的${F}_{\text{tmp }}$也必须包含$\widehat{z}$（注意，$\widehat{z}$是${T}_{z}^{ * }$的一个原始叶子节点）。这表明${F}_{z}^{ * }$是由${T}_{z}^{ * }$诱导的${G}_{z}^{ * }$的连通边覆盖。

Next,we prove that ${\bar{F}}^{ * }$ is the CEC of ${\bar{G}}^{ * }$ induced by ${\bar{T}}^{ * }$ . Let $\bar{e}$ be the highest node in $\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$ . Consider a reverse topological order $\bar{\sigma }$ of $T$ satisfying the following condition: a prefix of $\bar{\sigma }$ is a permutation of the nodes in the subtree of $T$ rooted at $\bar{e}$ . Define ${\bar{\sigma }}^{ * }$ to be the sequence obtained by deleting that prefix from $\bar{\sigma }$ . It is clear that ${\bar{\sigma }}^{ * }$ is a reverse topological order of ${\bar{T}}^{ * }$ . Define $\widehat{\bar{e}}$ to be the parent of $\bar{e}$ in $T$ . Note that $\widehat{\bar{e}}$ must belong to $F$ due to the definitions of $\bar{e}$ and $\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$

接下来，我们证明${\bar{F}}^{ * }$是由${\bar{T}}^{ * }$诱导的${\bar{G}}^{ * }$的连通边覆盖。设$\bar{e}$是$\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$中最高的节点。考虑$T$的一个逆拓扑序$\bar{\sigma }$，它满足以下条件：$\bar{\sigma }$的一个前缀是$T$中以$\bar{e}$为根的子树中节点的一个排列。定义${\bar{\sigma }}^{ * }$为从$\bar{\sigma }$中删除该前缀后得到的序列。显然，${\bar{\sigma }}^{ * }$是${\bar{T}}^{ * }$的一个逆拓扑序。定义$\widehat{\bar{e}}$为$\bar{e}$在$T$中的父节点。注意，根据$\bar{e}$和$\operatorname{sigpath}\left( {{f}^{ \circ  },T}\right)$的定义，$\widehat{\bar{e}}$必定属于$F$。

We will compare the execution of edge-cover(T)on $\sigma$ to that of edge-cover $\left( {\bar{T}}^{ * }\right)$ on ${\bar{\sigma }}^{ * }$ . For each $e$ in $\sigma$ ,define ${F}_{0}\left( e\right)$ as the content of ${F}_{\text{tmp }}$ after edge-cover(T)has finished processing $e$ . Similarly, for each $e$ in ${\bar{\sigma }}^{ * }$ ,define ${F}_{1}\left( e\right)$ as the content of ${F}_{\text{tmp }}$ after edge-cover $\left( {\bar{T}}^{ * }\right)$ has finished processing $e$ . Divide $\sigma$ into three segments: (i) ${\sigma }_{1}$ ,which includes the prefix of $\sigma$ ending at (and including) $\bar{e}$ , (ii) ${\sigma }_{2}$ ,which starts right after ${\sigma }_{1}$ and ends at (and includes) $\widehat{\bar{e}}$ ,and (iii) ${\sigma }_{3}$ ,which is the rest of $\sigma$ . Note that ${\bar{\sigma }}^{ * }$ is the concatenation of ${\sigma }_{2}$ and ${\sigma }_{3}$ .

我们将比较在$\sigma$上执行边覆盖算法edge - cover(T)与在${\bar{\sigma }}^{ * }$上执行边覆盖算法edge - cover $\left( {\bar{T}}^{ * }\right)$的情况。对于$\sigma$中的每个$e$，将${F}_{0}\left( e\right)$定义为边覆盖算法edge - cover(T)处理完$e$后${F}_{\text{tmp }}$的内容。类似地，对于${\bar{\sigma }}^{ * }$中的每个$e$，将${F}_{1}\left( e\right)$定义为边覆盖算法edge - cover $\left( {\bar{T}}^{ * }\right)$处理完$e$后${F}_{\text{tmp }}$的内容。将$\sigma$划分为三个部分：(i) ${\sigma }_{1}$，它包含$\sigma$中以$\bar{e}$结尾（包括$\bar{e}$）的前缀；(ii) ${\sigma }_{2}$，它从${\sigma }_{1}$之后开始，以$\widehat{\bar{e}}$结尾（包括$\widehat{\bar{e}}$）；(iii) ${\sigma }_{3}$，它是$\sigma$的其余部分。注意，${\bar{\sigma }}^{ * }$是${\sigma }_{2}$和${\sigma }_{3}$的连接。

Claim 1: For any $e$ in ${\sigma }_{2},e \in  {F}_{0}\left( e\right)$ if and only if $e \in  {F}_{1}\left( e\right)$ .

命题1：对于${\sigma }_{2},e \in  {F}_{0}\left( e\right)$中的任何$e$，当且仅当$e \in  {F}_{1}\left( e\right)$。

We prove the claim by induction. As the base case,consider $e$ as the first element in ${\sigma }_{2}$ . In ${\bar{T}}^{ * }$ , $e$ must be a leaf and,by Lemma 1,must be in ${F}_{1}\left( e\right)$ . In $T,e$ is either a leaf or $\widehat{e}$ . In the former case,Lemma 1 assures us $e \in  {F}_{0}\left( e\right)$ . In the latter case, $e$ is also in ${F}_{0}\left( e\right)$ because $\widehat{e} \in  F$ .

我们通过归纳法来证明这一命题。作为基础情况，考虑 $e$ 作为 ${\sigma }_{2}$ 中的第一个元素。在 ${\bar{T}}^{ * }$ 中，$e$ 必定是一个叶子节点，并且根据引理 1，必定在 ${F}_{1}\left( e\right)$ 中。在 $T,e$ 中，$e$ 要么是一个叶子节点，要么是 $\widehat{e}$。在前一种情况下，引理 1 保证 $e \in  {F}_{0}\left( e\right)$。在后一种情况下，$e$ 也在 ${F}_{0}\left( e\right)$ 中，因为 $\widehat{e} \in  F$。

Next,we prove the claim on every other node $e$ in ${\sigma }_{2}$ ,assuming the claim’s correctness on the node ${e}_{\text{before }}$ preceding $e$ in ${\sigma }_{2}$ . This inductive assumption implies ${F}_{1}\left( {e}_{\text{before }}\right)  \subseteq  {F}_{0}\left( {e}_{\text{before }}\right)$ . If $e \in  {F}_{0}\left( e\right)$ , then $e$ has a disappearing attribute $X$ not covered by ${F}_{0}\left( {e}_{\text{before }}\right)$ . As ${F}_{1}\left( {e}_{\text{before }}\right)  \subseteq  {F}_{0}\left( {e}_{\text{before }}\right)$ , ${F}_{1}\left( {e}_{\text{before }}\right)$ does not cover $X$ ,either. Hence,edge-cover $\left( {\bar{T}}^{ * }\right)$ adds $e$ to ${F}_{\text{tmp }}$ ,namely, $e \in  {F}_{1}\left( e\right)$ .

接下来，我们证明 ${\sigma }_{2}$ 中其他每个节点 $e$ 的命题，假设该命题在 ${\sigma }_{2}$ 中先于 $e$ 的节点 ${e}_{\text{before }}$ 上是正确的。这一归纳假设意味着 ${F}_{1}\left( {e}_{\text{before }}\right)  \subseteq  {F}_{0}\left( {e}_{\text{before }}\right)$。如果 $e \in  {F}_{0}\left( e\right)$，那么 $e$ 有一个消失属性 $X$ 未被 ${F}_{0}\left( {e}_{\text{before }}\right)$ 覆盖。由于 ${F}_{1}\left( {e}_{\text{before }}\right)  \subseteq  {F}_{0}\left( {e}_{\text{before }}\right)$，${F}_{1}\left( {e}_{\text{before }}\right)$ 也未覆盖 $X$。因此，边覆盖 $\left( {\bar{T}}^{ * }\right)$ 将 $e$ 添加到 ${F}_{\text{tmp }}$ 中，即 $e \in  {F}_{1}\left( e\right)$。

Let us now focus on the case where $e \in  {F}_{1}\left( e\right)$ . If $e = \widehat{\widehat{e}}$ ,the fact $\widehat{e} \in  F$ indicates $e \in  {F}_{0}\left( e\right)$ . Next, we consider $e \neq  \widehat{\bar{e}}$ ,meaning that $e$ is a proper descendant of $\widehat{\bar{e}}$ . The fact $e \in  {F}_{1}\left( e\right)$ suggests that $e$ has a disappearing attribute $X$ not covered by ${F}_{1}\left( {e}_{\text{before }}\right)$ . If $e \notin  {F}_{0}\left( e\right) ,{F}_{0}\left( {e}_{\text{before }}\right)$ must have a node ${e}^{\prime }$ containing $X$ . Node ${e}^{\prime }$ must come from ${\sigma }_{1}$ (the inductive assumption prohibits ${e}^{\prime }$ from appearing in ${\sigma }_{2}$ ) and hence must be a descendant of $\bar{e}$ . By acyclicity’s connectedness requirement, $X$ appearing in both $e$ and ${e}^{\prime }$ means that $X$ must belong to $\widehat{\widehat{e}}$ . But this contradicts $X$ disappearing at $e$ . We thus conclude that $e \in  {F}_{0}\left( e\right)$ .

现在让我们关注 $e \in  {F}_{1}\left( e\right)$ 的情况。如果 $e = \widehat{\widehat{e}}$，事实 $\widehat{e} \in  F$ 表明 $e \in  {F}_{0}\left( e\right)$。接下来，我们考虑 $e \neq  \widehat{\bar{e}}$，这意味着 $e$ 是 $\widehat{\bar{e}}$ 的一个真后代节点。事实 $e \in  {F}_{1}\left( e\right)$ 表明 $e$ 有一个消失属性 $X$ 未被 ${F}_{1}\left( {e}_{\text{before }}\right)$ 覆盖。如果 $e \notin  {F}_{0}\left( e\right) ,{F}_{0}\left( {e}_{\text{before }}\right)$ 必定有一个包含 $X$ 的节点 ${e}^{\prime }$。节点 ${e}^{\prime }$ 必定来自 ${\sigma }_{1}$（归纳假设禁止 ${e}^{\prime }$ 出现在 ${\sigma }_{2}$ 中），因此必定是 $\bar{e}$ 的后代节点。根据无环性的连通性要求，$X$ 同时出现在 $e$ 和 ${e}^{\prime }$ 中意味着 $X$ 必定属于 $\widehat{\widehat{e}}$。但这与 $X$ 在 $e$ 处消失相矛盾。因此我们得出结论 $e \in  {F}_{0}\left( e\right)$。

Claim 2: For any $e$ in ${\sigma }_{3},e \in  {F}_{0}\left( e\right)$ if and only if $e \in  {F}_{1}\left( e\right)$ .

命题 2：对于 ${\sigma }_{3},e \in  {F}_{0}\left( e\right)$ 中的任何 $e$，当且仅当 $e \in  {F}_{1}\left( e\right)$。

Claim 1 assures us that ${F}_{1}\left( \widehat{\bar{e}}\right)  \subseteq  {F}_{0}\left( \widehat{\bar{e}}\right)$ . Note also that $\widehat{\bar{e}}$ belongs to ${F}_{0}\left( \widehat{\bar{e}}\right)$ (as explained before, $\widehat{e} \in  F)$ and hence also to ${F}_{1}\left( \widehat{e}\right)$ (Claim 1). Any node ${e}^{\prime } \in  {F}_{0}\left( \widehat{e}\right)  \smallsetminus  {F}_{1}\left( \widehat{e}\right)$ must appear in the subtree rooted at $\widehat{\widehat{e}}$ in $T$ ,whereas any node $e$ in ${\sigma }_{3}$ must be outside that subtree. By acyclicity’s connectedness requirement,if ${e}^{\prime }$ contains an attribute $X$ in $e$ ,then $X \in  \widehat{\widehat{e}}$ for sure. This means that ${F}_{1}\left( \widehat{e}\right)$ covers a disappearing attribute of $e$ if and only if ${F}_{0}\left( \widehat{e}\right)$ does so. Therefore,edge-cover $\left( {\bar{T}}^{ * }\right)$ processes each node of ${\sigma }_{3}$ in the same way as edge-cover(T). This proves the correctness of Claim 2.

断言1向我们保证了${F}_{1}\left( \widehat{\bar{e}}\right)  \subseteq  {F}_{0}\left( \widehat{\bar{e}}\right)$。还要注意，$\widehat{\bar{e}}$属于${F}_{0}\left( \widehat{\bar{e}}\right)$（如前所述，$\widehat{e} \in  F)$，因此也属于${F}_{1}\left( \widehat{e}\right)$（断言1）。任何节点${e}^{\prime } \in  {F}_{0}\left( \widehat{e}\right)  \smallsetminus  {F}_{1}\left( \widehat{e}\right)$必定出现在$T$中以$\widehat{\widehat{e}}$为根的子树中，而${\sigma }_{3}$中的任何节点$e$必定在该子树之外。根据无环性的连通性要求，如果${e}^{\prime }$包含$e$中的一个属性$X$，那么肯定有$X \in  \widehat{\widehat{e}}$。这意味着，当且仅当${F}_{0}\left( \widehat{e}\right)$覆盖$e$的一个消失属性时，${F}_{1}\left( \widehat{e}\right)$才会覆盖该属性。因此，边覆盖$\left( {\bar{T}}^{ * }\right)$处理${\sigma }_{3}$中每个节点的方式与边覆盖(T)相同。这证明了断言2的正确性。

By putting Claim 1 and 2 together,we conclude that edge-cover $\left( {\bar{T}}^{ * }\right)$ returns all and only the attributes in ${\sigma }_{2} \cup  {\sigma }_{3}$ output by edge-cover(T). Therefore,the output of edge-cover $\left( {\bar{T}}^{ * }\right)$ is $F \cap  {\bar{E}}^{ * } = {\bar{F}}^{ * }.$

将断言1和断言2结合起来，我们得出结论：边覆盖$\left( {\bar{T}}^{ * }\right)$返回的恰好是边覆盖(T)输出的${\sigma }_{2} \cup  {\sigma }_{3}$中的所有属性。因此，边覆盖$\left( {\bar{T}}^{ * }\right)$的输出是$F \cap  {\bar{E}}^{ * } = {\bar{F}}^{ * }.$

## H Proof of Lemma 8

## H 引理8的证明

For any ${f}^{ * } \in  {F}_{z}^{ * }$ and any $z \in  Z$ that is not the root of ${T}_{z}^{ * }$ ,it holds that $\operatorname{sigpath}\left( {{f}^{ * },{T}_{z}^{ * }}\right)  \subseteq$ $\operatorname{sigpath}\left( {{f}^{ * },T}\right)$ . Similarly,for any ${f}^{ * } \in  {\bar{F}}^{ * }$ ,it holds that $\operatorname{sigpath}\left( {{f}^{ * },{\bar{T}}^{ * }}\right)  \subseteq  \operatorname{sigpath}\left( {{f}^{ * },T}\right)$ . To prove the lemma,it suffices to show that,given a super- $k$ -group $K = \left\{  {{e}_{1},\ldots ,{e}_{k}}\right\}$ ,we can always assign each ${e}_{i},i \in  \left\lbrack  k\right\rbrack$ ,to a distinct cluster in $\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$ . This is easy: if ${e}_{i}$ is picked from sigpath $\left( {{f}^{ * },{T}_{z}^{ * }}\right)$ for some $z \in  F$ and ${f}^{ * } \in  {F}_{z}^{ * }$ ,assign ${e}_{i}$ to sigpath $\left( {{f}^{ * },T}\right)$ ; if ${e}_{i}$ is picked from $\operatorname{sigpath}\left( {{f}^{ * },{\bar{T}}^{ * }}\right)$ for some ${f}^{ * } \in  {\bar{F}}^{ * }$ ,assign ${e}_{i}$ to $\operatorname{sigpath}\left( {{f}^{ * },T}\right)$ .

对于任意的${f}^{ * } \in  {F}_{z}^{ * }$和任意不是${T}_{z}^{ * }$根节点的$z \in  Z$，有$\operatorname{sigpath}\left( {{f}^{ * },{T}_{z}^{ * }}\right)  \subseteq$ $\operatorname{sigpath}\left( {{f}^{ * },T}\right)$。类似地，对于任意的${f}^{ * } \in  {\bar{F}}^{ * }$，有$\operatorname{sigpath}\left( {{f}^{ * },{\bar{T}}^{ * }}\right)  \subseteq  \operatorname{sigpath}\left( {{f}^{ * },T}\right)$。为了证明该引理，只需证明，给定一个超$k$ - 组$K = \left\{  {{e}_{1},\ldots ,{e}_{k}}\right\}$，我们总能将每个${e}_{i},i \in  \left\lbrack  k\right\rbrack$分配到$\{ \operatorname{sigpath}\left( {f,T}\right)  \mid  f \in  F\}$中的一个不同的簇中。这很容易：如果${e}_{i}$是从某个$z \in  F$和${f}^{ * } \in  {F}_{z}^{ * }$的签名路径$\left( {{f}^{ * },{T}_{z}^{ * }}\right)$中选取的，则将${e}_{i}$分配到签名路径$\left( {{f}^{ * },T}\right)$；如果${e}_{i}$是从某个${f}^{ * } \in  {\bar{F}}^{ * }$的$\operatorname{sigpath}\left( {{f}^{ * },{\bar{T}}^{ * }}\right)$中选取的，则将${e}_{i}$分配到$\operatorname{sigpath}\left( {{f}^{ * },T}\right)$。