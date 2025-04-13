# A Simple Parallel Algorithm for Natural Joins on Binary Relations

# 二元关系自然连接的简单并行算法

Yufei Tao

陶宇飞

Chinese University of Hong Kong, Hong Kong

中国香港中文大学，香港

taoyf@cse.cuhk.edu.hk

## — Abstract

## — 摘要

In PODS'17, Ketsman and Suciu gave an algorithm in the MPC model for computing the result of any natural join where every input relation has two attributes. Achieving an optimal load $O\left( {m/{p}^{1/\rho }}\right)$ - where $m$ is the total size of the input relations, $p$ the number of machines,and $\rho$ the fractional edge covering number of the join - their algorithm requires 7 rounds to finish. This paper presents a simpler algorithm that ensures the same load with 3 rounds (in fact, the second round incurs only a load of $O\left( {p}^{2}\right)$ to transmit certain statistics to assist machine allocation in the last round). Our algorithm is made possible by a new theorem that provides fresh insight on the structure of the problem, and brings us closer to understanding the intrinsic reason why joins on binary relations can be settled with load $O\left( {m/{p}^{1/\rho }}\right)$ .

在PODS'17会议上，凯茨曼（Ketsman）和苏丘（Suciu）在大规模并行计算（MPC）模型中提出了一种算法，用于计算每个输入关系都有两个属性的任意自然连接的结果。为了实现最优负载 $O\left( {m/{p}^{1/\rho }}\right)$ ——其中 $m$ 是输入关系的总大小， $p$ 是机器数量， $\rho$ 是连接的分数边覆盖数——他们的算法需要7轮才能完成。本文提出了一种更简单的算法，该算法在3轮内就能确保相同的负载（实际上，第二轮仅产生 $O\left( {p}^{2}\right)$ 的负载，用于传输某些统计信息，以协助最后一轮的机器分配）。我们的算法得益于一个新定理，该定理为问题的结构提供了全新的见解，并使我们更接近理解为什么二元关系的连接可以以负载 $O\left( {m/{p}^{1/\rho }}\right)$ 解决的内在原因。

2012 ACM Subject Classification Theory of computation $\rightarrow$ Database query processing and optimization (theory)

2012年美国计算机协会（ACM）学科分类 计算理论 $\rightarrow$ 数据库查询处理与优化（理论）

Keywords and phrases Natural Joins, Conjunctive Queries, MPC Algorithms, Parallel Computing

关键词和短语 自然连接、合取查询、大规模并行计算（MPC）算法、并行计算

Digital Object Identifier 10.4230/LIPIcs.ICDT.2020.25

数字对象标识符 10.4230/LIPIcs.ICDT.2020.25

## 1 Introduction

## 1 引言

Understanding the computational hardness of joins has always been a central topic in database theory. Traditionally, research efforts (see $\left\lbrack  {1,4,8,{11},{12},{14},{15}}\right\rbrack$ and the references therein) have focused on discovering fast algorithms for processing joins in the random access machine (RAM) model. Nowadays, massively parallel systems such as Hadoop [6] and Spark [2] (https://spark.apache.org) have become the mainstream computing architecture for performing analytical tasks on gigantic volumes of data, where direct implementation of RAM join algorithms rarely gives satisfactory performance. A main reason behind this phenomenon is that, while a RAM algorithm is designed to reduce the CPU time, in systems like Hadoop and Spark it is much more important to minimize the amount of communication across the participating machines, because the overhead of delivering all the necessary messages typically overwhelms the cost of CPU calculation. This has motivated a line of research - which also includes this work - that aims to understand the communication complexities of join problems.

理解连接操作的计算难度一直是数据库理论中的核心话题。传统上，研究工作（见 $\left\lbrack  {1,4,8,{11},{12},{14},{15}}\right\rbrack$ 及其参考文献）主要集中在随机访问机器（RAM）模型中寻找处理连接的快速算法。如今，像Hadoop [6] 和Spark [2] （https://spark.apache.org）这样的大规模并行系统已成为处理海量数据分析任务的主流计算架构，而直接实现RAM连接算法很少能获得令人满意的性能。这一现象的主要原因是，虽然RAM算法旨在减少CPU时间，但在Hadoop和Spark等系统中，最小化参与机器之间的通信量更为重要，因为传输所有必要消息的开销通常会超过CPU计算成本。这推动了一系列旨在理解连接问题通信复杂性的研究——本文的工作也属于此类研究。

### 1.1 Problem Definition

### 1.1 问题定义

In this subsection, we will first give a formal definition of natural join - the type of joins studied in this paper - and then elaborate on the computation model assumed.

在本小节中，我们将首先给出自然连接（本文所研究的连接类型）的正式定义，然后详细阐述所采用的计算模型。

Natural Joins. Let att be a countably infinite set where each element is called an attribute. Let $\mathbf{{dom}}$ be another countably infinite set. A tuple over a set $U \subseteq$ att is a function $\mathbf{u} : U \rightarrow$ dom. Given a subset $V$ of $U$ ,define $\mathbf{u}\left\lbrack  V\right\rbrack$ as the tuple $\mathbf{v}$ over $V$ such that $\mathbf{v}\left( X\right)  = \mathbf{u}\left( X\right)$ for every $X \in  V$ . We say that $\mathbf{u}\left\lbrack  V\right\rbrack$ is the projection of $\mathbf{u}$ on $V$ .

自然连接。设att是一个可数无限集，其中每个元素称为一个属性。设 $\mathbf{{dom}}$ 是另一个可数无限集。集合 $U \subseteq$ att上的一个元组是一个函数 $\mathbf{u} : U \rightarrow$ dom。给定 $U$ 的一个子集 $V$ ，将 $\mathbf{u}\left\lbrack  V\right\rbrack$ 定义为 $V$ 上的元组 $\mathbf{v}$ ，使得对于每个 $X \in  V$ 都有 $\mathbf{v}\left( X\right)  = \mathbf{u}\left( X\right)$ 。我们称 $\mathbf{u}\left\lbrack  V\right\rbrack$ 是 $\mathbf{u}$ 在 $V$ 上的投影。

A relation is defined to be a set $R$ of tuples over the same set $U$ of attributes. We say that $R$ is over $U$ ,and that the scheme of $R$ is $U$ ,represented with the notation scheme $\left( R\right)  = U$ .

关系被定义为同一属性集 $U$ 上元组的集合 $R$ 。我们称 $R$ 基于 $U$ ，并且 $R$ 的模式是 $U$ ，用符号scheme $\left( R\right)  = U$ 表示。

The arity of $R$ ,denoted as arity(R),equals $\mid$ scheme $\left( R\right)  \mid  .R$ is unary if arity $\left( R\right)  = 1$ ,and binary if $\operatorname{arity}\left( R\right)  = 2$ .

$R$ 的元数，记为arity(R)，等于 $\mid$ 。如果arity $\left( R\right)  = 1$ ，则scheme $\left( R\right)  \mid  .R$ 是一元的；如果 $\operatorname{arity}\left( R\right)  = 2$ ，则是二元的。

A join query is defined as a set $\mathcal{Q}$ of relations. If we let attset $\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{R \in  \mathcal{Q}}}\operatorname{scheme}\left( R\right)$ , the result of the query,denoted as $\operatorname{Join}\left( \mathcal{Q}\right)$ ,is the following relation over attset(Q)

连接查询被定义为一个关系集合$\mathcal{Q}$。如果我们设属性集为$\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{R \in  \mathcal{Q}}}\operatorname{scheme}\left( R\right)$，该查询的结果（表示为$\operatorname{Join}\left( \mathcal{Q}\right)$）是属性集(Q)上的如下关系

$$
\left\{  {\text{tuple }\mathbf{u}\text{ over attset }\left( \mathcal{Q}\right) \mid \forall R \in  \mathcal{Q},\mathbf{u}\left\lbrack  {\text{ scheme }\left( R\right) }\right\rbrack   \in  R}\right\}  .
$$

If $\mathcal{Q}$ has only two relations $R$ and $S$ ,we may also use $R \bowtie  S$ as an alternative representation of $\operatorname{Join}\left( Q\right)$ . The integer $m = \mathop{\sum }\limits_{{R \in  \mathcal{Q}}}\left| R\right|$ is the input size of the query. Concentrating on data complexity,we will assume that both $\left| \mathcal{Q}\right|$ and $\left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$ are constants.

如果$\mathcal{Q}$中只有两个关系$R$和$S$，我们也可以用$R \bowtie  S$作为$\operatorname{Join}\left( Q\right)$的另一种表示形式。整数$m = \mathop{\sum }\limits_{{R \in  \mathcal{Q}}}\left| R\right|$是该查询的输入规模。专注于数据复杂度，我们将假设$\left| \mathcal{Q}\right|$和$\left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$均为常量。

A join query $\mathcal{Q}$ is

连接查询$\mathcal{Q}$是

- scheme-clean if no distinct $R,S \in  \mathcal{Q}$ satisfy scheme $\left( R\right)  =$ scheme(S);

- 模式干净的（scheme - clean），如果没有不同的$R,S \in  \mathcal{Q}$满足模式$\left( R\right)  =$ = 模式(S)；

- simple if (i) $\mathcal{Q}$ is scheme-clean,and (ii) every $R \in  \mathcal{Q}$ is binary.

- 简单的，如果 (i) $\mathcal{Q}$ 是模式干净的，并且 (ii) 每个$R \in  \mathcal{Q}$都是二元的。

The primary goal of this paper is to design parallel algorithms for processing simple queries efficiently.

本文的主要目标是设计用于高效处理简单查询的并行算法。

Computation Model. We will assume the massively parallel computation (MPC) model which has been widely accepted as a reasonable abstraction of the massively parallel systems that exist today. In this model,there are $p$ machines such that at the beginning the input elements are evenly distributed across these machines. For a join query, this means that each machine stores $\Theta \left( {m/p}\right)$ tuples from the input relations.

计算模型。我们将采用大规模并行计算（MPC，Massively Parallel Computation）模型，该模型已被广泛认为是对当今现有大规模并行系统的合理抽象。在这个模型中，有$p$台机器，在开始时输入元素均匀分布在这些机器上。对于连接查询，这意味着每台机器存储来自输入关系的$\Theta \left( {m/p}\right)$个元组。

An algorithm is executed in rounds, each of which has two phases:

算法按轮次执行，每一轮有两个阶段：

- In the first phase, each machine does computation on the data of its local storage.

- 在第一阶段，每台机器对其本地存储的数据进行计算。

- In the second phase, the machines communicate by sending messages to each other.

- 在第二阶段，机器之间通过相互发送消息进行通信。

It is important to note that all the messages sent out in the second phase must have already been prepared in the first phase. This prevents a machine from, for example, sending information based on what has been received during the second phase. Another round is launched only if the problem has not been solved by the current round. In our context, solving a join query means that every tuple in the join result has been produced on at least one machine.

需要注意的是，在第二阶段发送的所有消息必须在第一阶段就已准备好。这可以防止一台机器，例如，根据在第二阶段接收到的信息发送信息。只有当当前轮次未能解决问题时，才会启动下一轮。在我们的上下文中，解决一个连接查询意味着连接结果中的每个元组至少在一台机器上被生成。

The load of a round is defined by the largest number of words that is received by a machine in this round,that is,if machine $i \in  \left\lbrack  {1,p}\right\rbrack$ receives ${x}_{i}$ words,then the load is $\mathop{\max }\limits_{{i = 1}}^{p}{x}_{i}$ . The performance of an algorithm is measured by two metrics: (i) the number of rounds, and (ii) the load of the algorithm, defined to be the largest load incurred by a round, among all the rounds. CPU computation, which takes place in the first phase of each round, is for free.

一轮的负载定义为该轮中一台机器接收到的最大单词数，即，如果机器$i \in  \left\lbrack  {1,p}\right\rbrack$接收到${x}_{i}$个单词，那么负载就是$\mathop{\max }\limits_{{i = 1}}^{p}{x}_{i}$。算法的性能通过两个指标来衡量：(i) 轮数，以及 (ii) 算法的负载，定义为所有轮次中一轮所产生的最大负载。每一轮第一阶段进行的CPU计算是免费的。

The number $p$ of machines is assumed to be significantly less than $m$ ,which in this paper means ${p}^{3} \leq  m$ specifically. All the algorithms to be mentioned,including those reviewed in the next subsection and the ones we propose, are randomized. Their loads are all bounded in a "high probability manner". Henceforth, whenever we say that an algorithm has load at most $L$ ,we mean that its load is bounded by $L$ with probability at least $1 - 1/{p}^{2}$ . Finally, we consider that every value in $\mathbf{{dom}}$ can be encoded in a single word.

假设机器的数量$p$显著小于$m$，在本文中具体指${p}^{3} \leq  m$。所有要提及的算法，包括下一小节回顾的算法和我们提出的算法，都是随机化的。它们的负载都以“高概率方式”有界。此后，每当我们说一个算法的负载至多为$L$时，我们的意思是其负载以至少$1 - 1/{p}^{2}$的概率被$L$所界定。最后，我们认为$\mathbf{{dom}}$中的每个值都可以用一个单词编码。

### 1.2 Previous Results

### 1.2 先前的研究成果

Afrati and Ullman [3] showed that any join query can be solved in a single round with load $\widetilde{O}\left( {m/{p}^{1/\min \{ k,\left| \mathcal{Q}\right| \} }}\right)$ where $k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|$ ,and the notation $\widetilde{O}$ hides polylogarithmic factors. Improving upon the earlier work [5], Koutris, Beame, and Suciu [10] presented another single-round algorithm that solves a join query with load $\widetilde{O}\left( {m/{p}^{1/\psi }}\right)$ where $\psi$ is the fractional quasi-packing number of the query. They also proved that $\Omega \left( {m/{p}^{1/\psi }}\right)$ is a lower bound on the load of any one-round algorithm, under certain restrictions on the statistics that the algorithm knows about the input relations.

阿夫拉蒂（Afrati）和厄尔曼（Ullman）[3]表明，任何连接查询都可以在一轮内以负载$\widetilde{O}\left( {m/{p}^{1/\min \{ k,\left| \mathcal{Q}\right| \} }}\right)$解决，其中$k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|$ ，符号$\widetilde{O}$隐藏了多项式对数因子。在早期工作[5]的基础上进行改进，库特里斯（Koutris）、比姆（Beame）和苏丘（Suciu）[10]提出了另一种单轮算法，该算法以负载$\widetilde{O}\left( {m/{p}^{1/\psi }}\right)$解决连接查询，其中$\psi$是查询的分数拟填充数。他们还证明，在算法已知的输入关系统计信息的某些限制下，$\Omega \left( {m/{p}^{1/\psi }}\right)$是任何单轮算法负载的下限。

For algorithms that perform more than one,but still $O\left( 1\right)$ ,rounds, $\Omega \left( {m/{p}^{1/\rho }}\right)$ has been shown [10] to be a lower bound on the load,where $\rho$ is the factional edge covering number of the query. The value of $\rho$ never exceeds,but can be strictly smaller than, $\psi$ ,which implies that multi-round algorithms may achieve significantly lower loads than one-round counterparts, thus providing strong motivation for studying the former.

对于执行超过一轮但仍为$O\left( 1\right)$轮的算法，[10]已证明$\Omega \left( {m/{p}^{1/\rho }}\right)$是负载的下限，其中$\rho$是查询的分数边覆盖数。$\rho$的值永远不会超过，但可能严格小于$\psi$ ，这意味着多轮算法可能比单轮算法实现显著更低的负载，因此为研究前者提供了强大的动力。

Though matching the lower bound of $\Omega \left( {m/{p}^{1/\rho }}\right)$ algorithmically still remains open for arbitrary join queries,this has been achieved for several special query classes $\left\lbrack  {3,7,9,{10}}\right\rbrack$ . In particular, Ketsman and Suciu [9] gave an algorithm, henceforth referred to as "the KS algorithm",that solves a simple query in 7 rounds with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ . The class of simple queries bears unique significance due to their relevance to subgraph enumeration, which is the problem of finding all occurrences of a subgraph ${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$ in a graph $G = \left( {V,E}\right)$ . Regarding $E$ as a relation of two attributes,i.e.,source vertex and destination vertex,we can convert subgraph enumeration to a join query on $\left| {E}^{\prime }\right|$ copies of the "relation" $E$ ,and renaming the attributes in each relation to reflect how the vertices of ${V}^{\prime }$ are connected in ${G}^{\prime }$ ; see [12] for an example where ${G}^{\prime }$ is a clique of 3 vertices.

尽管对于任意连接查询，在算法上达到$\Omega \left( {m/{p}^{1/\rho }}\right)$的下限仍然是一个未解决的问题，但对于几个特殊的查询类$\left\lbrack  {3,7,9,{10}}\right\rbrack$已经实现了这一点。特别是，凯茨曼（Ketsman）和苏丘（Suciu）[9]给出了一种算法，此后称为“KS算法”，该算法在7轮内以负载$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$解决简单查询。简单查询类具有独特的意义，因为它们与子图枚举相关，子图枚举是在图$G = \left( {V,E}\right)$中找到子图${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$的所有出现位置的问题。将$E$视为两个属性（即源顶点和目标顶点）的关系，我们可以将子图枚举转换为对$\left| {E}^{\prime }\right|$个“关系”$E$副本的连接查询，并对每个关系中的属性进行重命名，以反映${V}^{\prime }$的顶点在${G}^{\prime }$中是如何连接的；有关${G}^{\prime }$是3个顶点的团的示例，请参见[12]。

### 1.3 Our Contributions

### 1.3 我们的贡献

Our main result is that any simple join query can be solved in the MPC model with the optimal load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ using 3 rounds. Our algorithm is in fact similar to a subroutine deployed in the KS algorithm, which, however, also demands several other subroutines that entail a larger number of rounds, and are proved to be unnecessary by our solution. The improvement owes to a new theorem that reveals an intrinsic property of the problem, which will be explained shortly with an example. In retrospect, the algorithm of Kestman and Suciu [9] can be regarded as using sophisticated graph-theoretic ideas to compensate for not knowing that property. It is not surprising that their algorithm can be simplified substantially once our understanding on the structure of the problem has been strengthened.

我们的主要结果是，任何简单连接查询都可以在MPC模型中使用3轮以最优负载$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$解决。我们的算法实际上类似于KS算法中部署的一个子程序，但KS算法还需要其他几个子程序，这些子程序需要更多的轮数，而我们的解决方案证明这些子程序是不必要的。这种改进归功于一个新定理，该定理揭示了问题的内在属性，下面将通过一个示例简要说明。回顾一下，凯茨曼（Kestman）和苏丘（Suciu）[9]的算法可以被视为使用复杂的图论思想来弥补对该属性的未知。一旦我们对问题结构的理解得到加强，他们的算法能够大幅简化也就不足为奇了。

To gain an overview of our techniques,let us consider the join query $\mathcal{Q}$ illustrated by the graph in Figure 1a. An edge connecting vertices $X$ and $Y$ represents a relation ${R}_{\{ X,Y\} }$ with scheme $\left( {R}_{\{ X,Y\} }\right)  = \{ X,Y\} .\mathcal{Q}$ is defined by the set of relations represented by the 18 edges in Figure 1a. Notice that attest $\left( \mathcal{Q}\right)  = \{ \mathrm{A},\mathrm{B},\ldots ,\mathrm{L}\}$ has a size of 12 .

为了全面了解我们的技术，让我们考虑图1a中的图所示的连接查询$\mathcal{Q}$。连接顶点$X$和$Y$的边表示一个关系${R}_{\{ X,Y\} }$，其模式为$\left( {R}_{\{ X,Y\} }\right)  = \{ X,Y\} .\mathcal{Q}$，该关系由图1a中18条边所代表的关系集定义。注意，证明$\left( \mathcal{Q}\right)  = \{ \mathrm{A},\mathrm{B},\ldots ,\mathrm{L}\}$的大小为12。

We adopt an idea that is behind nearly all the join algorithms in the MPC model $\left\lbrack  {7,9,{10}}\right\rbrack$ , namely,to divide the join result based on "heavy hitters". Let $\lambda$ be an integer parameter whose choice will be clarified later. A value $x \in  \mathbf{{dom}}$ is heavy if an input relation $R \in  \mathcal{Q}$ has at least $m/\lambda$ tuples carrying this value on an attribute $X \in$ scheme(R). The number of heavy values is $O\left( \lambda \right)$ . A value $x \in  \mathbf{{dom}}$ is light if $x$ appears in at least one relation $R \in  \mathcal{Q}$ but is not heavy. A tuple in the join result may take a heavy or light value on each of the 12 attributes $\mathrm{A},\ldots ,\mathrm{K}$ . As there are at most $O\left( \lambda \right)$ choices on each attribute (namely,light value or one of the $O\left( \lambda \right)$ heavy values),there are $O\left( {\lambda }^{12}\right)$ "combinations" of choices from all attributes; we will refer to each combination as a configuration. If we manage to design an algorithm to find the result tuples under each configuration, executing this algorithm for all configurations solves the query.

我们采用了一种几乎所有MPC模型$\left\lbrack  {7,9,{10}}\right\rbrack$中的连接算法背后的思想，即根据“高频项（heavy hitters）”划分连接结果。设$\lambda$为一个整数参数，其选择将在后面说明。如果输入关系$R \in  \mathcal{Q}$在属性$X \in$（模式(R)）上至少有$m/\lambda$个元组携带某个值$x \in  \mathbf{{dom}}$，则该值为高频值。高频值的数量为$O\left( \lambda \right)$。如果$x$出现在至少一个关系$R \in  \mathcal{Q}$中但不是高频值，则该值为低频值。连接结果中的一个元组在12个属性$\mathrm{A},\ldots ,\mathrm{K}$的每个属性上可能取高频值或低频值。由于每个属性最多有$O\left( \lambda \right)$种选择（即低频值或$O\left( \lambda \right)$个高频值之一），因此所有属性的选择有$O\left( {\lambda }^{12}\right)$种“组合”；我们将每种组合称为一种配置。如果我们设法设计一种算法来找出每种配置下的结果元组，那么对所有配置执行该算法就能解决该查询。

Figure 1b illustrates what happens in one of the possible configurations where we constrain attributes $\mathrm{D},\mathrm{E},\mathrm{F}$ ,and $\mathrm{K}$ to take heavy values $\mathrm{d},\mathrm{e},\mathrm{f}$ ,and $\mathrm{k}$ respectively,and the other attributes to take light values. Accordingly, vertices D, E, F, and K are colored black in the figure. This configuration gives rise to a residue query ${\mathcal{Q}}^{\prime }$ whose input relations are decided as follows:

图1b展示了在一种可能的配置中发生的情况，在该配置中，我们约束属性$\mathrm{D},\mathrm{E},\mathrm{F}$和$\mathrm{K}$分别取高频值$\mathrm{d},\mathrm{e},\mathrm{f}$和$\mathrm{k}$，而其他属性取低频值。因此，图中顶点D、E、F和K被涂成黑色。这种配置产生了一个剩余查询${\mathcal{Q}}^{\prime }$，其输入关系确定如下：

- For each edge $\{ X,Y\}$ with two white vertices, ${\mathcal{Q}}^{\prime }$ has a relation ${R}_{\{ X,Y\} }^{\prime }$ that contains only the tuples in ${R}_{\{ X,Y\} } \in  \mathcal{Q}$ using light values on both $X$ and $Y$ ;

- 对于每条连接两个白色顶点的边$\{ X,Y\}$，${\mathcal{Q}}^{\prime }$有一个关系${R}_{\{ X,Y\} }^{\prime }$，该关系仅包含${R}_{\{ X,Y\} } \in  \mathcal{Q}$中在$X$和$Y$上都使用低频值的元组；

<!-- Media -->

<!-- figureText: G C G O O L O H (c) After deleting (d) After semi-join black verticess reduction E=e Ok PK=k (a) A join query (b) A residue query -->

<img src="https://cdn.noedgeai.com/0195ce8e-ebf0-7dc5-b30b-c8220ede8665_3.jpg?x=320&y=259&w=1049&h=328&r=0"/>

Figure 1 Processing a join by constraining heavy values.

图1 通过约束高频值处理连接。

<!-- Media -->

- For each edge $\{ X,Y\}$ with a white vertex $X$ and a black vertex $Y,{\mathcal{Q}}^{\prime }$ has a relation ${R}_{\{ X,Y\} }^{\prime }$ that contains only the tuples in ${R}_{\{ X,Y\} } \in  \mathcal{Q}$ each of which uses a light value on $X$ and the constrained heavy value on $Y$ ;

- 对于每条连接一个白色顶点$X$和一个黑色顶点$Y,{\mathcal{Q}}^{\prime }$的边$\{ X,Y\}$，$X$有一个关系${R}_{\{ X,Y\} }^{\prime }$，该关系仅包含${R}_{\{ X,Y\} } \in  \mathcal{Q}$中每个在$X$上使用低频值且在$Y$上使用约束高频值的元组；

- For each edge $\{ X,Y\}$ with two black vertices, ${\mathcal{Q}}^{\prime }$ has a relation ${R}_{\{ X,Y\} }^{\prime }$ with only one tuple that takes the constrained heavy values on $X$ and $Y$ ,respectively.

- 对于每条连接两个黑色顶点的边$\{ X,Y\}$，${\mathcal{Q}}^{\prime }$有一个关系${R}_{\{ X,Y\} }^{\prime }$，该关系只有一个元组，该元组分别在$X$和$Y$上取约束高频值。

For example,in ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ ,a tuple must use light values on both $\mathrm{A}$ and $\mathrm{B}$ ; in ${R}_{\{ \mathrm{D},\mathrm{G}\} }^{\prime }$ ,a tuple must use value $\mathrm{d}$ on $\mathrm{D}$ and a light value on $\mathrm{G};{R}_{\{ \mathrm{D},\mathrm{K}\} }^{\prime }$ has only a single tuple with value $\mathrm{d}$ on $\mathrm{D}$ and $\mathrm{k}$ on $\mathrm{K}$ . Finding all result tuples for $\mathcal{Q}$ under the designated configuration amounts to evaluating the residue query ${\mathcal{Q}}^{\prime }$ .

例如，在${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$中，一个元组必须在$\mathrm{A}$和$\mathrm{B}$上都使用轻值；在${R}_{\{ \mathrm{D},\mathrm{G}\} }^{\prime }$中，一个元组必须在$\mathrm{D}$上使用值$\mathrm{d}$，并且在$\mathrm{G};{R}_{\{ \mathrm{D},\mathrm{K}\} }^{\prime }$上使用轻值，在$\mathrm{D}$上只有一个值为$\mathrm{d}$且在$\mathrm{K}$上为$\mathrm{k}$的元组。在指定配置下为$\mathcal{Q}$找到所有结果元组相当于评估残差查询${\mathcal{Q}}^{\prime }$。

Since the black attributes have had their values fixed in the configuration, they can be deleted from the residue query,after which some relations in ${\mathcal{Q}}^{\prime }$ become unary or even disappear. Relation ${R}_{\{ \mathrm{A},\mathrm{D}\} }^{\prime } \in  {\mathcal{Q}}^{\prime }$ ,for example,is now regarded as a unary relation over $\{ \mathrm{A}\}$ , with the understanding that every tuple is "piggybacked" the value d on D. Let us denote this unary relation as ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ ,which is illustrated in Figure $1\mathrm{c}$ with a dotted edge extending from vertex $\mathrm{A}$ and carrying the label $\mathrm{d}$ . The deletion of $\mathrm{D},\mathrm{E},\mathrm{F}$ ,and $\mathrm{K}$ results in 13 unary relations (e.g.,two of them are over $\{ \mathtt{A}\}$ ,namely, ${R}_{\{ \mathtt{A}\}  \mid  \mathtt{d}}^{\prime }$ and ${R}_{\{ \mathtt{A}\}  \mid  \mathtt{e}}^{\prime }$ ). Attributes $\mathtt{G},\mathtt{H}$ ,and $\mathtt{L}$ now become isolated because they are not connected to any other vertices by solid edges. Relations ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime }$ ,and ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime }$ still have arity 2 because their schemes do not have black attributes. ${R}_{\{ \mathrm{D},\mathrm{K}\} }^{\prime }$ ,on the other hand,has disappeared.

由于黑色属性的值在配置中已固定，因此可以从残差查询中删除它们，之后${\mathcal{Q}}^{\prime }$中的一些关系会变为一元关系甚至消失。例如，关系${R}_{\{ \mathrm{A},\mathrm{D}\} }^{\prime } \in  {\mathcal{Q}}^{\prime }$现在被视为关于$\{ \mathrm{A}\}$的一元关系，理解为每个元组在D上“搭载”值d。我们将这个一元关系表示为${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$，如图$1\mathrm{c}$所示，从顶点$\mathrm{A}$延伸出一条虚线边并带有标签$\mathrm{d}$。删除$\mathrm{D},\mathrm{E},\mathrm{F}$和$\mathrm{K}$会产生13个一元关系（例如，其中两个是关于$\{ \mathtt{A}\}$的，即${R}_{\{ \mathtt{A}\}  \mid  \mathtt{d}}^{\prime }$和${R}_{\{ \mathtt{A}\}  \mid  \mathtt{e}}^{\prime }$）。属性$\mathtt{G},\mathtt{H}$和$\mathtt{L}$现在变得孤立，因为它们没有通过实线边与任何其他顶点相连。关系${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime }$和${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime }$的元数仍然为2，因为它们的模式中没有黑色属性。另一方面，${R}_{\{ \mathrm{D},\mathrm{K}\} }^{\prime }$已经消失。

Our algorithm solves the residue query ${\mathcal{Q}}^{\prime }$ of Figure 1c in two steps:

我们的算法分两步解决图1c中的残差查询${\mathcal{Q}}^{\prime }$：

1. Perform a semi-join reduction which involves two substeps:

1. 执行半连接约简，包括两个子步骤：

- For every vertex $X$ in Figure 1c,intersect all the unary relations over $\{ X\}$ - if any - into a single list ${R}_{\{ X\} }^{\prime \prime }$ . For example,the two unary relations ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ and ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ of $\mathrm{A}$ are intersected on A to produce ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ . Note that only the values in ${\bar{R}}_{\{ \mathrm{A}\} }^{\prime \prime }$ can appear in the final join result.

- 对于图1c中的每个顶点$X$，将所有关于$\{ X\}$的一元关系（如果有的话）相交成一个单一列表${R}_{\{ X\} }^{\prime \prime }$。例如，$\mathrm{A}$的两个一元关系${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$和${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$在A上相交以产生${R}_{\{ \mathrm{A}\} }^{\prime \prime }$。请注意，只有${\bar{R}}_{\{ \mathrm{A}\} }^{\prime \prime }$中的值才能出现在最终的连接结果中。

- For every non-isolated attribute $X$ in Figure 1c,use ${R}_{\{ X\} }^{\prime \prime }$ to shrink each non-unary relation ${R}_{\{ X,Y\} }^{\prime }$ ,for all relevant $Y$ ,to kick out those tuples whose $X$ -values do not appear in ${R}_{\{ X\} }^{\prime \prime }$ . This reduces ${R}_{\{ X,Y\} }^{\prime }$ to a subset ${R}_{\{ X,Y\} }^{\prime \prime }$ . For example,after the shrinking,every tuple in ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ uses a value in ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ on attribute $\mathrm{A}$ ,and a value in ${R}_{\{ \mathrm{B}\} }^{\prime \prime }$ on attribute $\mathrm{B}$ .

- 对于图1c中每个非孤立属性 $X$，使用 ${R}_{\{ X\} }^{\prime \prime }$ 对每个非一元关系 ${R}_{\{ X,Y\} }^{\prime }$ 进行缩减，对于所有相关的 $Y$，剔除那些 $X$ 值不在 ${R}_{\{ X\} }^{\prime \prime }$ 中出现的元组。这会将 ${R}_{\{ X,Y\} }^{\prime }$ 缩减为一个子集 ${R}_{\{ X,Y\} }^{\prime \prime }$。例如，缩减后，${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ 中的每个元组在属性 $\mathrm{A}$ 上使用 ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ 中的一个值，在属性 $\mathrm{B}$ 上使用 ${R}_{\{ \mathrm{B}\} }^{\prime \prime }$ 中的一个值。

2. Perform a cartesian product. To see how,first observe that the residue query ${\mathcal{Q}}^{\prime }$ can now be further simplified into a join query ${\mathcal{Q}}^{\prime \prime }$ which includes:

2. 执行笛卡尔积。为了了解具体做法，首先观察到残差查询 ${\mathcal{Q}}^{\prime }$ 现在可以进一步简化为一个连接查询 ${\mathcal{Q}}^{\prime \prime }$，该查询包括：

- The relation ${R}_{\{ X\} }^{\prime \prime }$ for each isolated attribute $X$ ;

- 每个孤立属性 $X$ 对应的关系 ${R}_{\{ X\} }^{\prime \prime }$；

- The relation ${R}_{\{ X,Y\} }^{\prime \prime }$ for each solid edge in Figure 1c.

- 图1c中每条实边对应的关系 ${R}_{\{ X,Y\} }^{\prime \prime }$。

Figure 1d gives a neater view of ${\mathcal{Q}}^{\prime \prime }$ ,from which it is easy to see that $\operatorname{Join}\left( {\mathcal{Q}}^{\prime \prime }\right)$ equals the cartesian product of (i) three unary relations ${R}_{\{ \mathrm{G}}^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ ,and ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$ ,(ii) a binary relation ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }$ ,and (iii) the result of the "triangle join" $\left\{  {{R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime \prime }}\right\}$ . This cartesian product can be generated in one round using the hypercube algorithm of [3], leveraging the fact that only light values are present in these relations. An optimal load can be achieved by setting $\lambda  = \Theta \left( {p}^{1/\left( {2\rho }\right) }\right)$ .

图1d更清晰地展示了 ${\mathcal{Q}}^{\prime \prime }$，从图中很容易看出 $\operatorname{Join}\left( {\mathcal{Q}}^{\prime \prime }\right)$ 等于 (i) 三个一元关系 ${R}_{\{ \mathrm{G}}^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ 和 ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$、(ii) 一个二元关系 ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }$ 以及 (iii) “三角形连接” $\left\{  {{R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime \prime }}\right\}$ 的结果的笛卡尔积。利用这些关系中仅存在轻量级值这一事实，可以使用文献[3]中的超立方体算法在一轮中生成这个笛卡尔积。通过设置 $\lambda  = \Theta \left( {p}^{1/\left( {2\rho }\right) }\right)$ 可以实现最优负载。

The KS algorithm deploys a similar procedure to deal with a primitive type of configurations (see Lemma 14 of [9]). The main difference, however, is that while the KS algorithm resorts to more sophisticated and round-intensive procedures to tackle other configurations (e.g., the one in Figure 1b), we proceed in the same manner for all configurations anyway. This simplification is the side product of a theorem established in this paper, which shows that the cartesian product of all the unary relations ${R}_{\{ X\} }^{\prime \prime }$ - one for each isolated attribute $X$ (in our example, ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ ,and ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$ ) - is not too large on average,over all the possible configurations. This property allows us to duplicate such cartesian products onto a large number of machines, which in turn is the key reason why the hypercube algorithm can be invoked to finish Step 2 in one round. In fact, handling those unary relations has been the main challenge in all the algorithms $\left\lbrack  {7,9,{10}}\right\rbrack$ applying the "decomposition by heavy-hitters" idea, because the binary relations obtained from the decomposition are so-called "skew-free", and hence, easy to process. In light of this, our theorem provides deeper insight into the reason why simple join queries can be processed with the optimal load.

KS算法采用类似的过程来处理一种基本类型的配置（见文献[9]中的引理14）。然而，主要的区别在于，虽然KS算法采用更复杂且轮次密集的过程来处理其他配置（例如，图1b中的配置），但我们无论如何都以相同的方式处理所有配置。这种简化是本文所建立的一个定理的附带产物，该定理表明，所有一元关系${R}_{\{ X\} }^{\prime \prime }$（每个孤立属性$X$对应一个，在我们的示例中为${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$和${R}_{\{ \mathrm{L}\} }^{\prime \prime }$）的笛卡尔积，在所有可能的配置上平均而言不会太大。这一特性使我们能够将这些笛卡尔积复制到大量的机器上，而这反过来也是可以调用超立方体算法在一轮内完成步骤2的关键原因。事实上，处理这些一元关系一直是所有应用“基于频繁项分解”思想的算法$\left\lbrack  {7,9,{10}}\right\rbrack$的主要挑战，因为从分解中得到的二元关系是所谓的“无偏斜”关系，因此易于处理。鉴于此，我们的定理更深入地揭示了为什么简单连接查询可以以最优负载进行处理的原因。

It is worth mentioning that while our algorithm performs 3 rounds, the second round, which transmits certain statistics to assist machine allocation in the last round, incurs only a small load that is a polynomial of $p$ and does not depend on $m$ . In other words,the algorithm performs only 2 rounds whose loads are sensitive to $m$ . This brings us very close to finally settling the problem with the optimal load using genuinely only 2 rounds, and leaves open the question: is the transmission of those statistics absolutely necessary?

值得一提的是，虽然我们的算法执行3轮，但第二轮（传输某些统计信息以协助最后一轮的机器分配）产生的负载很小，该负载是$p$的多项式，并且不依赖于$m$。换句话说，该算法实际上只有2轮的负载对$m$敏感。这使我们非常接近最终以最优负载真正仅用2轮解决该问题，同时也留下了一个悬而未决的问题：传输那些统计信息是否绝对必要？

## 2 Preliminaries

## 2 预备知识

### 2.1 Hypergraphs

### 2.1 超图

We define a hypergraph $\mathcal{G}$ as a pair(V,E)where:

我们将超图$\mathcal{G}$定义为一个二元组(V, E)，其中：

- $\mathcal{V}$ is a finite set,where each element is called a vertex;

- $\mathcal{V}$是一个有限集，其每个元素称为一个顶点；

- $\mathcal{E}$ is a set of non-empty subsets of $\mathcal{V}$ ,where each subset is called a hyperedge.

- $\mathcal{E}$是$\mathcal{V}$的非空子集的集合，其每个子集称为一条超边。

Given a vertex $X \in  \mathcal{V}$ and a hyperedge $e \in  \mathcal{E}$ ,we say that $X$ and $e$ are incident to each other if $X \in  e$ . A vertex $X \in  \mathcal{V}$ is dangling if it is not incident on any hyperedge in $\mathcal{E}$ . In this paper, we consider only hypergraphs where there are no dangling vertices.

给定一个顶点$X \in  \mathcal{V}$和一条超边$e \in  \mathcal{E}$，如果$X \in  e$，我们称$X$和$e$相互关联。如果一个顶点$X \in  \mathcal{V}$与$\mathcal{E}$中的任何超边都不关联，则称该顶点为悬挂顶点。在本文中，我们仅考虑不存在悬挂顶点的超图。

Two distinct vertices $X,Y \in  V$ are adjacent to each other if there is an $e \in  \mathcal{E}$ containing both $X$ and $Y$ . An edge $e$ is unary if $\left| e\right|  = 1$ ,or binary if $\left| e\right|  = 2$ . A binary hypergraph is one that has only binary edges. Given a subset ${\mathcal{V}}^{\prime }$ of $\mathcal{V}$ ,we define the subgraph induced by ${\mathcal{V}}^{\prime }$ as $\left( {{\mathcal{V}}^{\prime },{\mathcal{E}}^{\prime }}\right)$ where ${\mathcal{E}}^{\prime } = \left\{  {{\mathcal{V}}^{\prime } \cap  e \mid  e \in  \mathcal{E} \land  {\mathcal{V}}^{\prime } \cap  e \neq  \varnothing }\right\}$ .

如果存在一条包含$X$和$Y$的$e \in  \mathcal{E}$，则两个不同的顶点$X,Y \in  V$相互邻接。如果$\left| e\right|  = 1$，则边$e$为一元边；如果$\left| e\right|  = 2$，则边$e$为二元边。二元超图是仅包含二元边的超图。给定$\mathcal{V}$的一个子集${\mathcal{V}}^{\prime }$，我们将由${\mathcal{V}}^{\prime }$诱导的子图定义为$\left( {{\mathcal{V}}^{\prime },{\mathcal{E}}^{\prime }}\right)$，其中${\mathcal{E}}^{\prime } = \left\{  {{\mathcal{V}}^{\prime } \cap  e \mid  e \in  \mathcal{E} \land  {\mathcal{V}}^{\prime } \cap  e \neq  \varnothing }\right\}$。

### 2.2 Fractional Edge Coverings and Packings

### 2.2 分数边覆盖与边填充

Let $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ be a hypergraph,and $W$ a function mapping $\mathcal{E}$ to real values in $\left\lbrack  {0,1}\right\rbrack$ . We call $W\left( e\right)$ the weight of the hyperedge $e$ ,and $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ the total weight of $W$ . Given a vertex $X \in  \mathcal{V}$ ,we refer to $\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)$ ,which is the sum of the weights of the edges incident to $X$ ,as the weight of $X$ .

设$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$为一个超图，$W$是一个将$\mathcal{E}$映射到$\left\lbrack  {0,1}\right\rbrack$中实数值的函数。我们称$W\left( e\right)$为超边$e$的权重，称$\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$为$W$的总权重。给定一个顶点$X \in  \mathcal{V}$，我们将$\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)$（即与$X$关联的边的权重之和）称为$X$的权重。

$W$ is a fractional edge covering of $\mathcal{G}$ if the weight of every vertex $X \in  \mathcal{V}$ is at least 1 . The fractional edge covering number of $\mathcal{G}$ ,which is denoted as $\rho \left( \mathcal{G}\right)$ ,equals the smallest total weight of all the fractional edge coverings. $W$ is a fractional edge packing if the weight of every vertex $X \in  \mathcal{V}$ is at most 1 . The fractional edge packing number of $\mathcal{G}$ ,which is denoted as $\tau \left( \mathcal{G}\right)$ ,equals the largest total weight of all the fractional edge packings. A fractional edge packing $W$ is tight if it is simultaneously also a fractional edge covering. Likewise,a fractional edge covering $W$ is tight if it is simultaneously also a fractional edge packing. Note that in both cases it must hold that the weight of every vertex $X \in  \mathcal{V}$ is exactly 1 .

如果每个顶点$X \in  \mathcal{V}$的权重至少为1，则$W$是$\mathcal{G}$的一个分数边覆盖。$\mathcal{G}$的分数边覆盖数（记为$\rho \left( \mathcal{G}\right)$）等于所有分数边覆盖的最小总权重。如果每个顶点$X \in  \mathcal{V}$的权重至多为1，则$W$是一个分数边填充。$\mathcal{G}$的分数边填充数（记为$\tau \left( \mathcal{G}\right)$）等于所有分数边填充的最大总权重。如果一个分数边填充$W$同时也是一个分数边覆盖，则称其为紧的。同样地，如果一个分数边覆盖$W$同时也是一个分数边填充，则称其为紧的。注意，在这两种情况下，每个顶点$X \in  \mathcal{V}$的权重都必须恰好为1。

The lemma below lists several useful properties of binary hypergraphs:

下面的引理列出了二元超图的几个有用性质：

- Lemma 1. If $\mathcal{G}$ is binary,then:

- 引理1。如果$\mathcal{G}$是二元的，则：

- $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$ and $\rho \left( \mathcal{G}\right)  \geq  \tau \left( \mathcal{G}\right)$ ,where the two equalities hold if and only if $\mathcal{G}$ admits a tight fractional edge packing (or covering).

- $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$和$\rho \left( \mathcal{G}\right)  \geq  \tau \left( \mathcal{G}\right)$，当且仅当$\mathcal{G}$存在一个紧的分数边填充（或覆盖）时，这两个等式成立。

- $\mathcal{G}$ admits a fractional edge packing $W$ of total weight $\tau \left( \mathcal{G}\right)$ such that

- $\mathcal{G}$存在一个总权重为$\tau \left( \mathcal{G}\right)$的分数边填充$W$，使得

1. The weight of every vertex $X \in  \mathcal{V}$ is either 0 or 1 .

1. 每个顶点$X \in  \mathcal{V}$的权重要么为0，要么为1。

2. If $\mathcal{Z}$ is the set of vertices in $\mathcal{V}$ with weight0,then $\rho \left( \mathcal{G}\right)  - \tau \left( \mathcal{G}\right)  = \left| \mathcal{Z}\right|$ .

2. 如果$\mathcal{Z}$是$\mathcal{V}$中权重为0的顶点集，则$\rho \left( \mathcal{G}\right)  - \tau \left( \mathcal{G}\right)  = \left| \mathcal{Z}\right|$。

Proof. The first bullet is proved in Theorem 2.2.7 of [13]. The fractional edge packing $W$ in Theorem 2.1.5 of [13] satisfies Condition 1 of the second bullet. This $W$ also fulfills Condition 2 , as is proved in Lemma 16 of [9].

证明。第一个要点在文献[13]的定理2.2.7中已证明。文献[13]的定理2.1.5中的分数边填充$W$满足第二个要点的条件1。如文献[9]的引理16所证明的，这个$W$也满足条件2。

Example. Suppose that $\mathcal{G}$ is the binary hypergraph in Figure 1a. It has a fractional edge covering number $\rho \left( \mathcal{G}\right)  = {6.5}$ ,as is achieved by the function ${W}_{1}$ that maps $\{ \mathrm{G},\mathrm{F}\} ,\{ \mathrm{D},\mathrm{K}\}$ , $\{ \mathrm{I},\mathrm{J}\} ,\{ \mathrm{E},\mathrm{H}\}$ ,and $\{ \mathrm{E},\mathrm{L}\}$ to $1,\{ \mathrm{A},\mathrm{B}\} ,\{ \mathrm{A},\mathrm{C}\}$ ,and $\{ \mathrm{B},\mathrm{C}\}$ to $1/2$ ,and the other edges to0. Its fractional edge packing number is $\tau \left( \mathcal{G}\right)  = {5.5}$ ,achieved by the function ${W}_{2}$ which is the same as ${W}_{1}$ except that ${W}_{2}$ maps $\{ \mathrm{E},\mathrm{L}\}$ to $0.{W}_{2}$ also satisfies both conditions of the second bullet (notice that $\mathcal{Z} = \{ \mathrm{L}\}$ ).

示例。假设$\mathcal{G}$是图1a中的二元超图。它有一个分数边覆盖数$\rho \left( \mathcal{G}\right)  = {6.5}$，这可以通过函数${W}_{1}$实现，该函数将$\{ \mathrm{G},\mathrm{F}\} ,\{ \mathrm{D},\mathrm{K}\}$、$\{ \mathrm{I},\mathrm{J}\} ,\{ \mathrm{E},\mathrm{H}\}$和$\{ \mathrm{E},\mathrm{L}\}$映射到$1,\{ \mathrm{A},\mathrm{B}\} ,\{ \mathrm{A},\mathrm{C}\}$，将$\{ \mathrm{B},\mathrm{C}\}$映射到$1/2$，并将其他边映射到0。它的分数边填充数是$\tau \left( \mathcal{G}\right)  = {5.5}$，由函数${W}_{2}$实现，该函数与${W}_{1}$相同，只是${W}_{2}$将$\{ \mathrm{E},\mathrm{L}\}$映射到$0.{W}_{2}$，同时也满足第二个要点的两个条件（注意$\mathcal{Z} = \{ \mathrm{L}\}$）。

### 2.3 Hypergraph of a Join Query and the AGM Bound

### 2.3 连接查询的超图与AGM边界

Every join query $\mathcal{Q}$ defines a hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ where $\mathcal{V} =$ attset(Q)and $\mathcal{E} =$ $\{$ scheme $\left( R\right)  \mid  R \in  \mathcal{Q}\}$ . When $\mathcal{Q}$ is scheme-clean,for each hyperedge $e \in  \mathcal{E}$ we denote by ${R}_{e}$ the input relation $R \in  \mathcal{Q}$ with $e =$ scheme(R). Note also that $\mathcal{G}$ must be binary if $\mathcal{Q}$ is simple. The following result is known as the AGM bound:

每个连接查询$\mathcal{Q}$都定义了一个超图$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$，其中$\mathcal{V} =$为attset(Q)，$\mathcal{E} =$ $\{$为scheme $\left( R\right)  \mid  R \in  \mathcal{Q}\}$。当$\mathcal{Q}$是方案清洁（scheme - clean）时，对于每个超边$e \in  \mathcal{E}$，我们用${R}_{e}$表示输入关系$R \in  \mathcal{Q}$，其具有$e =$ scheme(R)。另需注意，如果$\mathcal{Q}$是简单的，那么$\mathcal{G}$必须是二元的。以下结果被称为AGM边界：

Lemma 2 ( [4]). Let $\mathcal{Q}$ be a scheme-clean join query,and $W$ be a fractional edge covering of the hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ defined by $\mathcal{Q}$ . Then, $\left| {\operatorname{Join}\left( \mathcal{Q}\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\right| }^{W\left( e\right) }$ .

引理2（[4]）。设$\mathcal{Q}$是一个方案清洁的连接查询，$W$是由$\mathcal{Q}$定义的超图$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$的一个分数边覆盖。那么，$\left| {\operatorname{Join}\left( \mathcal{Q}\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\right| }^{W\left( e\right) }$。

### 2.4 MPC Building Blocks

### 2.4 MPC构建模块

Cartesian Products. Suppose that $R$ and $S$ are relations with disjoint schemes. Their cartesian product,denoted as $R \times  S$ ,is a relation over scheme $\left( R\right)  \cup$ scheme(S)which consists of all the tuples $\mathbf{u}$ over $\operatorname{scheme}\left( R\right)  \cup$ scheme(S)such that $\mathbf{u}\left\lbrack  {\operatorname{scheme}\left( R\right) }\right\rbrack   \in  R$ and $\mathbf{u}\left\lbrack  {\text{scheme}\left( S\right) }\right\rbrack   \in  S$ . Sometimes,we need to compute the cartesian product of a set of relations $\mathcal{Q} = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{t}}\right\}  \left( {t \geq  2}\right)$ with mutually disjoint schemes. For convenience,define ${CP}\left( \mathcal{Q}\right)$ as a short form for ${R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{t}$ . Note that ${CP}\left( \mathcal{Q}\right)$ can also be regarded as the result $\operatorname{Join}\left( \mathcal{Q}\right)$ of the join query $\mathcal{Q}$ .

笛卡尔积。假设 $R$ 和 $S$ 是具有不相交模式的关系。它们的笛卡尔积，记为 $R \times  S$，是模式 $\left( R\right)  \cup$ 方案(S)上的一个关系，该关系由 $\operatorname{scheme}\left( R\right)  \cup$ 方案(S)上的所有元组 $\mathbf{u}$ 组成，使得 $\mathbf{u}\left\lbrack  {\operatorname{scheme}\left( R\right) }\right\rbrack   \in  R$ 且 $\mathbf{u}\left\lbrack  {\text{scheme}\left( S\right) }\right\rbrack   \in  S$。有时，我们需要计算一组具有互不相交模式的关系 $\mathcal{Q} = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{t}}\right\}  \left( {t \geq  2}\right)$ 的笛卡尔积。为方便起见，将 ${CP}\left( \mathcal{Q}\right)$ 定义为 ${R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{t}$ 的简写形式。注意，${CP}\left( \mathcal{Q}\right)$ 也可以看作是连接查询 $\mathcal{Q}$ 的结果 $\operatorname{Join}\left( \mathcal{Q}\right)$。

Two results regarding cartesian products will be useful:

关于笛卡尔积的两个结果将很有用：

- Lemma 3 ( [3]). Given $\mathcal{Q} = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{t}}\right\}$ ,we can compute ${CP}\left( \mathcal{Q}\right)$ in one round with load $\widetilde{O}\left( {{\left| CP\left( \mathcal{Q}\right) \right| }^{\frac{1}{t}}/{p}^{\frac{1}{t}}}\right)$ using $p$ machines.

- 引理 3（[3]）。给定 $\mathcal{Q} = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{t}}\right\}$，我们可以使用 $p$ 台机器在一轮内以负载 $\widetilde{O}\left( {{\left| CP\left( \mathcal{Q}\right) \right| }^{\frac{1}{t}}/{p}^{\frac{1}{t}}}\right)$ 计算出 ${CP}\left( \mathcal{Q}\right)$。

Lemma 4 ( [9]). Let ${\mathcal{Q}}_{1}$ and ${\mathcal{Q}}_{2}$ be two join queries that have input sizes at most $m$ , and satisfy the condition that attset $\left( {\mathcal{Q}}_{1}\right)  \cap$ attset $\left( {\mathcal{Q}}_{2}\right)  = \varnothing$ . Suppose that $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ can be computed in one round with load $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$ using ${p}_{1}$ machines,and similarly, $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ can be computed in one round with load $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$ using ${p}_{2}$ machines. Then $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ can be computed in one round with load $\widetilde{O}\left( {m/\min \left\{  {{p}^{1/{t}_{1}},{p}^{1/{t}_{2}}}\right\}  }\right)$ using ${p}_{1}{p}_{2}$ machines.

引理 4（[9]）。设 ${\mathcal{Q}}_{1}$ 和 ${\mathcal{Q}}_{2}$ 是两个输入大小至多为 $m$ 的连接查询，并且满足属性集 $\left( {\mathcal{Q}}_{1}\right)  \cap$ 属性集 $\left( {\mathcal{Q}}_{2}\right)  = \varnothing$ 的条件。假设 $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ 可以使用 ${p}_{1}$ 台机器在一轮内以负载 $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$ 计算得出，类似地，$\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ 可以使用 ${p}_{2}$ 台机器在一轮内以负载 $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$ 计算得出。那么 $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ 可以使用 ${p}_{1}{p}_{2}$ 台机器在一轮内以负载 $\widetilde{O}\left( {m/\min \left\{  {{p}^{1/{t}_{1}},{p}^{1/{t}_{2}}}\right\}  }\right)$ 计算得出。

Skew-Free Queries. Let $\mathcal{Q}$ be a join query on binary relations. Regardless of whether $\mathcal{Q}$ is simple, it can be solved in a single round with a small load if no value appears too often in its input relations. Denote by $m$ the input size of $\mathcal{Q}$ . Set $k = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$ ,and list out the attributes in attset(Q)as ${X}_{1},\ldots ,{X}_{k}$ . Let ${p}_{i}$ be a positive integer, $i \in  \left\lbrack  {1,k}\right\rbrack$ ,which is referred to as the share of ${X}_{i}$ . A relation $R \in  \mathcal{Q}$ with scheme $\left\{  {{X}_{i},{X}_{j}}\right\}$ is skew-free if every value $x \in$ dom fulfills both conditions below:

无倾斜查询。设 $\mathcal{Q}$ 为二元关系上的连接查询。无论 $\mathcal{Q}$ 是否简单，如果其输入关系中没有值频繁出现，那么它可以在一轮内以较小的负载解决。用 $m$ 表示 $\mathcal{Q}$ 的输入大小。设 $k = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$，并将 attset(Q) 中的属性列为 ${X}_{1},\ldots ,{X}_{k}$。设 ${p}_{i}$ 为正整数，$i \in  \left\lbrack  {1,k}\right\rbrack$，这被称为 ${X}_{i}$ 的份额。如果每个值 $x \in$ 定义域都满足以下两个条件，则具有模式 $\left\{  {{X}_{i},{X}_{j}}\right\}$ 的关系 $R \in  \mathcal{Q}$ 是无倾斜的：

- $R$ has $O\left( {m/{p}_{i}}\right)$ tuples $\mathbf{u}$ with $\mathbf{u}\left( {X}_{i}\right)  = x$ ;

- $R$ 有 $O\left( {m/{p}_{i}}\right)$ 个元组 $\mathbf{u}$ 且 $\mathbf{u}\left( {X}_{i}\right)  = x$；

- $R$ has $O\left( {m/{p}_{j}}\right)$ tuples $\mathbf{u}$ with $\mathbf{u}\left( {X}_{j}\right)  = x$ .

- $R$ 有 $O\left( {m/{p}_{j}}\right)$ 个元组 $\mathbf{u}$ 且 $\mathbf{u}\left( {X}_{j}\right)  = x$。

Define share $\left( R\right)  = {p}_{i} \cdot  {p}_{j}$ . If every $R \in  \mathcal{Q}$ is skew-free, $\mathcal{Q}$ is skew-free,and can be solved with the following guarantee:

定义份额 $\left( R\right)  = {p}_{i} \cdot  {p}_{j}$。如果每个 $R \in  \mathcal{Q}$ 都是无倾斜的，那么 $\mathcal{Q}$ 是无倾斜的，并且可以通过以下保证来解决：

- Lemma 5 ( [5]). A skew-free query $\mathcal{Q}$ can be answered in one round with load $\widetilde{O}\left( {m/\mathop{\min }\limits_{{R \in  \mathcal{Q}}}\operatorname{share}\left( R\right) }\right)$ using $\mathop{\prod }\limits_{{i = 1}}^{k}{p}_{i}$ machines.

- 引理 5（[5]）。一个无倾斜查询 $\mathcal{Q}$ 可以使用 $\mathop{\prod }\limits_{{i = 1}}^{k}{p}_{i}$ 台机器在一轮内以负载 $\widetilde{O}\left( {m/\mathop{\min }\limits_{{R \in  \mathcal{Q}}}\operatorname{share}\left( R\right) }\right)$ 得到答案。

One-Attribute Reduction. Let $X \in$ att be an attribute. We have $a \geq  1$ unary relations ${R}_{1},\ldots ,{R}_{a}$ over $\{ X\}$ ,and $b \geq  1$ binary relations ${S}_{1},\ldots ,{S}_{b}$ such that ${S}_{i}\left( {1 \leq  i \leq  b}\right)$ is a relation over $\left\{  {X,{Y}_{i}}\right\}$ where ${Y}_{i}$ is an attribute in att different from $X$ . Here,both $a$ and $b$ are constants. Our objective is to compute ${S}_{i}^{\# }$ which includes all tuples $\mathbf{u} \in  {S}_{i}$ satisfying the condition that $\mathbf{u}\left( X\right)  \in  \mathop{\bigcap }\limits_{{j = 1}}^{a}{R}_{j}$ . We will refer to this operation as one-attribute reduction. Let $n = \mathop{\sum }\limits_{{j = 1}}^{a}\left| {R}_{i}\right|  + \mathop{\sum }\limits_{{i = 1}}^{b}\left| {S}_{i}\right|$ . A value $x \in  \mathbf{{dom}}$ is a heavy-hitter if at least $n/p$ tuples in some ${S}_{i}\left( {1 \leq  i \leq  b}\right)$ use $x$ as their $X$ -values,where $p$ is the number of machines assigned to the operation.

单属性归约。设 $X \in$ att 为一个属性。我们有 $a \geq  1$ 个一元关系 ${R}_{1},\ldots ,{R}_{a}$ 定义在 $\{ X\}$ 上，以及 $b \geq  1$ 个二元关系 ${S}_{1},\ldots ,{S}_{b}$，使得 ${S}_{i}\left( {1 \leq  i \leq  b}\right)$ 是定义在 $\left\{  {X,{Y}_{i}}\right\}$ 上的关系，其中 ${Y}_{i}$ 是 att 中不同于 $X$ 的一个属性。这里，$a$ 和 $b$ 都是常数。我们的目标是计算 ${S}_{i}^{\# }$，它包含所有满足条件 $\mathbf{u}\left( X\right)  \in  \mathop{\bigcap }\limits_{{j = 1}}^{a}{R}_{j}$ 的元组 $\mathbf{u} \in  {S}_{i}$。我们将此操作称为单属性归约。设 $n = \mathop{\sum }\limits_{{j = 1}}^{a}\left| {R}_{i}\right|  + \mathop{\sum }\limits_{{i = 1}}^{b}\left| {S}_{i}\right|$。如果某个 ${S}_{i}\left( {1 \leq  i \leq  b}\right)$ 中至少有 $n/p$ 个元组使用 $x$ 作为它们的 $X$ 值，则值 $x \in  \mathbf{{dom}}$ 是高频项，其中 $p$ 是分配给该操作的机器数量。

- Lemma 6. One-attribute reduction can be performed in one round with load $\widetilde{O}\left( {p + n/p}\right)$ using $p$ machines,provided that each machine knows all the heavy-hitters.

- 引理 6。只要每台机器都知道所有高频项，单属性归约就可以使用 $p$ 台机器在一轮内以负载 $\widetilde{O}\left( {p + n/p}\right)$ 执行。

Proof. See Appendix A.

证明。见附录 A。

It is worth mentioning that the above lemma is an extension of a result in [10]. The $p$ term in the load can actually be eliminated, if the machine knows also additional statistics of the heavy-hitters. We do not need to be bothered with such details because the term is affordable for our purposes.

值得一提的是，上述引理是文献[10]中一个结果的扩展。如果机器还了解重击项（heavy - hitters）的额外统计信息，负载中的$p$项实际上是可以消除的。我们无需为这些细节烦恼，因为就我们的目的而言，该项是可以接受的。

## 3 A Taxonomy of the Join Result

## 3 连接结果的分类

Recall that Section 1 outlined a method to partition the join result based on heavy and light values. In this section, we formalize this method and establish some fundamental properties. Denote by $\mathcal{Q}$ a simple join query,by $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ the hypergraph defined by $\mathcal{Q}$ ,by $m$ the input size of $\mathcal{Q}$ ,and by $k$ the number of attributes in attset(Q).

回顾一下，第1节概述了一种基于重值和轻值对连接结果进行划分的方法。在本节中，我们将这种方法形式化，并建立一些基本性质。用$\mathcal{Q}$表示一个简单的连接查询，用$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$表示由$\mathcal{Q}$定义的超图，用$m$表示$\mathcal{Q}$的输入规模，用$k$表示attset(Q)中的属性数量。

Heavy and Light Values. Fix an arbitrary integer $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$ . A value $x \in  \mathbf{{dom}}$ is

重值和轻值。固定一个任意整数$\lambda  \in  \left\lbrack  {1,m}\right\rbrack$。一个值$x \in  \mathbf{{dom}}$是

- heavy if there exists a relation $R \in  \mathcal{Q}$ and some attribute $X \in  \operatorname{scheme}\left( R\right)$ such that $\left| {\{ \mathbf{u} \in  R \mid  \mathbf{u}\left( X\right)  = x\} }\right|  \geq  m/\lambda$

- 重值，如果存在一个关系$R \in  \mathcal{Q}$和某个属性$X \in  \operatorname{scheme}\left( R\right)$，使得$\left| {\{ \mathbf{u} \in  R \mid  \mathbf{u}\left( X\right)  = x\} }\right|  \geq  m/\lambda$

- light if $x$ is not heavy,but appears in at least one relation $R \in  \mathcal{Q}$ .

- 轻值，如果$x$不是重值，但至少出现在一个关系$R \in  \mathcal{Q}$中。

Since each relation has $O\left( 1\right)$ attributes,the number of heavy values is $O\left( \lambda \right)$ .

由于每个关系有$O\left( 1\right)$个属性，重值的数量是$O\left( \lambda \right)$。

Configurations. Let $\mathcal{H}$ be an arbitrary (possibly empty) subset of attset(Q). A configuration of $\mathcal{H}$ is a tuple $\mathbf{\eta }$ over $\mathcal{H}$ such that $\mathbf{\eta }\left( X\right)$ is heavy for every $X \in  \mathcal{H}$ . Obviously,each $\mathcal{H} \subseteq  \operatorname{attset}\left( \mathcal{Q}\right)$ has at most $O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)$ configurations.

配置。设$\mathcal{H}$是attset(Q)的任意（可能为空）子集。$\mathcal{H}$的一个配置是一个在$\mathcal{H}$上的元组$\mathbf{\eta }$，使得对于每个$X \in  \mathcal{H}$，$\mathbf{\eta }\left( X\right)$是重值。显然，每个$\mathcal{H} \subseteq  \operatorname{attset}\left( \mathcal{Q}\right)$最多有$O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)$个配置。

Residue Relations and Residue Queries. Now,let us fix a configuration $\eta$ of $\mathcal{H}$ ,and aim to produce all the result tuples $\mathbf{u} \in  \operatorname{Join}\left( \mathcal{Q}\right)$ consistent with the configuration,namely, $\mathbf{u}$ satisfies

残差关系和残差查询。现在，让我们固定$\mathcal{H}$的一个配置$\eta$，并旨在生成所有与该配置一致的结果元组$\mathbf{u} \in  \operatorname{Join}\left( \mathcal{Q}\right)$，即$\mathbf{u}$满足

- $\mathbf{u}\left( X\right)  = \mathbf{\eta }\left( X\right)$ for every $X \in  \mathcal{H}$ ,and

- 对于每个 $X \in  \mathcal{H}$，有 $\mathbf{u}\left( X\right)  = \mathbf{\eta }\left( X\right)$，并且

- $u\left( X\right)$ is light for every $X \in  \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$ .

- 对于每个 $X \in  \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$，$u\left( X\right)$ 是轻的。

We will take a few steps to define what is the residue query under $\mathbf{\eta }$ ,which is denoted as ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ ,whose result is precisely the set of all the qualifying $\mathbf{u}$ .

我们将分几步来定义在 $\mathbf{\eta }$ 下的残差查询（residue query）是什么，用 ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ 表示，其结果恰好是所有符合条件的 $\mathbf{u}$ 的集合。

Let $e$ be a hyperedge in $\mathcal{E}$ that is not subsumed by $\mathcal{H}$ ,i.e., $e$ has at least one attribute outside $\mathcal{H}$ . This hyperedge is said to be active on $\mathbf{\eta }$ . Define ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ ,namely,the set of attributes in $e$ that are outside $\mathcal{H}$ . The relation ${R}_{e} \in  \mathcal{Q}$ defines a residue relation under $\mathbf{\eta }$ which

设 $e$ 是 $\mathcal{E}$ 中的一条超边（hyperedge），它不被 $\mathcal{H}$ 包含，即 $e$ 至少有一个属性在 $\mathcal{H}$ 之外。称这条超边在 $\mathbf{\eta }$ 上是活跃的。定义 ${e}^{\prime } = e \smallsetminus  \mathcal{H}$，即 $e$ 中在 $\mathcal{H}$ 之外的属性集合。关系 ${R}_{e} \in  \mathcal{Q}$ 定义了在 $\mathbf{\eta }$ 下的一个残差关系（residue relation），该关系

- is over ${e}^{\prime }$ and

- 基于 ${e}^{\prime }$，并且

- consists of every tuple $\mathbf{v}$ that is the projection of some tuple $\mathbf{w} \in  {R}_{e}$ "consistent" with $\mathbf{\eta }$ , namely: (i) $\mathbf{w}\left( X\right)  = \mathbf{\eta }\left( X\right)$ for every $X \in  e \cap  \mathcal{H}$ ,(ii) $\mathbf{w}\left( Y\right)$ is light for every $Y \in  {e}^{\prime }$ ,and (iii) $v = w\left\lbrack  {e}^{\prime }\right\rbrack$ .

- 由每个元组 $\mathbf{v}$ 组成，$\mathbf{v}$ 是某个与 $\mathbf{\eta }$ “一致” 的元组 $\mathbf{w} \in  {R}_{e}$ 的投影，即：(i) 对于每个 $X \in  e \cap  \mathcal{H}$，有 $\mathbf{w}\left( X\right)  = \mathbf{\eta }\left( X\right)$；(ii) 对于每个 $Y \in  {e}^{\prime }$，$\mathbf{w}\left( Y\right)$ 是轻的；(iii) $v = w\left\lbrack  {e}^{\prime }\right\rbrack$。

The residue relation is denoted as ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$ ,where $\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack$ is the projection of $\mathbf{\eta }$ on $e \smallsetminus  {e}^{\prime }$ , as was introduced in Section 1.1.

残差关系用 ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$ 表示，其中 $\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack$ 是 $\mathbf{\eta }$ 在 $e \smallsetminus  {e}^{\prime }$ 上的投影，如 1.1 节所述。

We can now define the residue query as

我们现在可以将残差查询定义为

$$
{\mathcal{Q}}_{\mathbf{\eta }}^{\prime } = \left\{  {{R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  } \mid  e \in  \mathcal{E},e\text{ active on }\mathbf{\eta }}\right\}  .
$$

Example. Suppose that $\mathcal{Q}$ is the query discussed in Section 1.3 with its hypergraph $\mathcal{G}$ given in Figure 1a. Consider the configuration $\mathbf{\eta }$ of $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ where $\mathbf{\eta }\left\lbrack  \mathrm{D}\right\rbrack   = \mathrm{d},\mathbf{\eta }\left\lbrack  \mathrm{E}\right\rbrack   = \mathrm{e}$ , $\mathbf{\eta }\left\lbrack  \mathrm{F}\right\rbrack   = \mathbf{f}$ ,and $\mathbf{\eta }\left\lbrack  \mathrm{K}\right\rbrack   = \mathrm{k}$ . If $e$ is the edge $\{ \mathrm{A},\mathrm{D}\}$ ,then ${e}^{\prime } = \{ \mathrm{A}\}$ and $\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack   = \mathbf{\eta }\left\lbrack  {\{ \mathrm{D}\} }\right\rbrack   = \mathrm{d}$ ,such that ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$ is the relation ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ mentioned in Section 1.3. If $e$ is the edge $\{ \mathrm{A},\mathrm{B}\}$ ,on the other hand,then ${e}^{\prime } = \{ \mathrm{A},\mathrm{B}\}$ and $\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack   = \varnothing$ ,so that ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$ can be written as ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \varnothing }^{\prime }$ , and is the relation ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ in Section 1.3. The residue query ${\mathcal{Q}}_{\eta }^{\prime }$ is precisely the query ${\mathcal{Q}}^{\prime }$ described in Section 1.3. (to be continued) $\blacktriangle$

示例。假设$\mathcal{Q}$是1.3节中讨论的查询，其超图$\mathcal{G}$如图1a所示。考虑$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$的配置$\mathbf{\eta }$，其中$\mathbf{\eta }\left\lbrack  \mathrm{D}\right\rbrack   = \mathrm{d},\mathbf{\eta }\left\lbrack  \mathrm{E}\right\rbrack   = \mathrm{e}$、$\mathbf{\eta }\left\lbrack  \mathrm{F}\right\rbrack   = \mathbf{f}$和$\mathbf{\eta }\left\lbrack  \mathrm{K}\right\rbrack   = \mathrm{k}$。如果$e$是边$\{ \mathrm{A},\mathrm{D}\}$，那么${e}^{\prime } = \{ \mathrm{A}\}$和$\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack   = \mathbf{\eta }\left\lbrack  {\{ \mathrm{D}\} }\right\rbrack   = \mathrm{d}$，使得${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$是1.3节中提到的关系${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$。另一方面，如果$e$是边$\{ \mathrm{A},\mathrm{B}\}$，那么${e}^{\prime } = \{ \mathrm{A},\mathrm{B}\}$和$\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack   = \varnothing$，因此${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$可以写成${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \varnothing }^{\prime }$，并且是1.3节中的关系${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$。残差查询${\mathcal{Q}}_{\eta }^{\prime }$恰好是1.3节中描述的查询${\mathcal{Q}}^{\prime }$。（待续）$\blacktriangle$

It is rudimentary to verify

验证这一点是很基础的

$$
\operatorname{Join}\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{\mathcal{H}}\left( {\mathop{\bigcup }\limits_{{\text{config. }\mathbf{\eta }\text{ of }\mathcal{H}}}\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right) \times \{ \mathbf{\eta }\} }\right) . \tag{1}
$$

Denote by ${m}_{\eta }$ the input size of ${\mathcal{Q}}_{\eta }^{\prime }$ . The next proposition says that total input size of all the residue queries is not too large:

用${m}_{\eta }$表示${\mathcal{Q}}_{\eta }^{\prime }$的输入规模。下一个命题表明，所有残差查询的总输入规模不会太大：

Proposition 7. If $\mathcal{Q}$ is simple, $\mathop{\sum }\limits_{\text{config. }}{}_{\eta }$ of $\mathcal{H}{m}_{\eta } = O\left( {m \cdot  {\lambda }^{k - 2}}\right)$ holds for every $\mathcal{H} \subseteq$ attset(Q).

命题7。如果$\mathcal{Q}$是简单的，那么对于每个$\mathcal{H} \subseteq$ attset(Q)，$\mathcal{H}{m}_{\eta } = O\left( {m \cdot  {\lambda }^{k - 2}}\right)$的$\mathop{\sum }\limits_{\text{config. }}{}_{\eta }$成立。

Proof. See Appendix B. Y. Tao

证明。见附录B。Y. 陶

<!-- Media -->

<!-- figureText: Q ${}_{\mathrm{H}}$ -->

<img src="https://cdn.noedgeai.com/0195ce8e-ebf0-7dc5-b30b-c8220ede8665_8.jpg?x=648&y=259&w=297&h=297&r=0"/>

Figure 2 Subgraph induced by $\mathcal{L}$ .

图2 由$\mathcal{L}$诱导的子图。

<!-- Media -->

## 4 A Join Computation Framework

## 4 连接计算框架

Given a simple join query $\mathcal{Q}$ ,we will concentrate on an arbitrary subset $\mathcal{H}$ of attset(Q)in this section. In Sections 4.1-4.2, we will generalize the strategy illustrated in Section 1.3 into a formal framework for producing

给定一个简单连接查询$\mathcal{Q}$，在本节中我们将关注attset(Q)的任意子集$\mathcal{H}$。在4.1 - 4.2节中，我们将把1.3节中说明的策略推广到一个正式的框架中以进行生成

$$
\mathop{\bigcup }\limits_{{\text{config. }\mathbf{\eta }\text{ of }\mathcal{H}}}\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right) . \tag{2}
$$

Section 4.3 will then establish a theorem on this framework, which is the core of the techniques proposed in this paper.

然后4.3节将针对这个框架建立一个定理，这是本文提出的技术的核心。

### 4.1 Removing the Attributes in $\mathcal{H}$

### 4.1 移除$\mathcal{H}$中的属性

We will refer to each attribute in $\mathcal{H}$ as a heavy attribute. Define $\mathcal{L} = \operatorname{scheme}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$ ,where each attribute is called a light attribute. Denote by $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ the hypergraph defined by $\mathcal{Q}$ . An edge $e \in  \mathcal{E}$ is (i) a light edge if $e$ contains two light attributes,or (ii) a cross edge if $e$ contains a heavy attribute and a light attribute. A light attribute $X \in  \mathcal{L}$ is a border attribute if it appears in at least one cross edge $e$ of $\mathcal{G}$ ; note that this implies $e \smallsetminus  \mathcal{H} = \{ X\}$ . Denote by ${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$ the subgraph of $\mathcal{G}$ induced by $\mathcal{L}$ . A vertex $X \in  \mathcal{L}$ is isolated if $\{ X\}$ is the only edge in ${\mathcal{E}}^{\prime }$ incident to $X$ . Define $\mathcal{I}$ to be the set of isolated vertices in ${\mathcal{G}}^{\prime }$ .

我们将把$\mathcal{H}$中的每个属性称为重属性。定义$\mathcal{L} = \operatorname{scheme}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$，其中每个属性称为轻属性。用$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$表示由$\mathcal{Q}$定义的超图。一条边$e \in  \mathcal{E}$满足：（i）若$e$包含两个轻属性，则它是一条轻边；（ii）若$e$包含一个重属性和一个轻属性，则它是一条交叉边。如果一个轻属性$X \in  \mathcal{L}$出现在$\mathcal{G}$的至少一条交叉边$e$中，那么它就是一个边界属性；注意，这意味着$e \smallsetminus  \mathcal{H} = \{ X\}$。用${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$表示由$\mathcal{L}$所诱导的$\mathcal{G}$的子图。如果$\{ X\}$是${\mathcal{E}}^{\prime }$中与$X$关联的唯一边，则顶点$X \in  \mathcal{L}$是孤立的。将$\mathcal{I}$定义为${\mathcal{G}}^{\prime }$中孤立顶点的集合。

Example (cont.). As before,let $\mathcal{Q}$ be the join query whose hypergraph $\mathcal{G}$ is shown in Figure 1a,and set $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\} .\mathcal{L}$ includes all the white vertices in Figure 1b. $\{ \mathrm{A},\mathrm{B}\}$ is a light edge, $\{ \mathrm{A},\mathrm{D}\}$ is a cross edge,while $\{ \mathrm{D},\mathrm{K}\}$ is neither a light edge nor a cross edge. All the vertices in $\mathcal{L}$ except $\mathrm{J}$ are border vertices. Figure 2 shows the subgraph of $\mathcal{G}$ induced by $\mathcal{L}$ , where a unary edge is represented by a box and a binary edge by a segment. Notice that no unary edge covers J. Vertices G, H, and L are the only isolated vertices. (to be continued) $\blacktriangle$

示例（续）。和之前一样，设$\mathcal{Q}$为连接查询，其超图$\mathcal{G}$如图1a所示，并且集合$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\} .\mathcal{L}$包含图1b中所有的白色顶点。$\{ \mathrm{A},\mathrm{B}\}$是一条轻边，$\{ \mathrm{A},\mathrm{D}\}$是一条交叉边，而$\{ \mathrm{D},\mathrm{K}\}$既不是轻边也不是交叉边。除$\mathrm{J}$之外，$\mathcal{L}$中的所有顶点都是边界顶点。图2展示了由$\mathcal{L}$所诱导的$\mathcal{G}$的子图，其中一元边用方框表示，二元边用线段表示。注意，没有一元边覆盖J。顶点G、H和L是仅有的孤立顶点。（待续）$\blacktriangle$

### 4.2 Semi-Join Reduction

### 4.2 半连接约简

Recall that every configuration $\mathbf{\eta }$ of $\mathcal{H}$ gives rise to a residue query ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ . Next,we will transform ${\mathcal{Q}}_{\eta }^{\prime }$ into an alternative join query ${\mathcal{Q}}_{\eta }^{\prime \prime }$ which,as shown in the next section,can be processed in a single round in the MPC model.

回顾一下，$\mathcal{H}$的每个配置$\mathbf{\eta }$都会产生一个残差查询${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$。接下来，我们将把${\mathcal{Q}}_{\eta }^{\prime }$转换为一个替代连接查询${\mathcal{Q}}_{\eta }^{\prime \prime }$，如在下一节中所示，该查询可以在MPC（大规模并行计算，Massively Parallel Computation）模型的单轮计算中进行处理。

First of all,observe that the hypergraph defined by ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ is always ${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$ ,regardless of $\mathbf{\eta }$ . Consider a border attribute $X \in  \mathcal{L}$ ,and a cross edge $e$ of $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ incident to $X$ . As explained in Section 3,the input relation ${R}_{e} \in  \mathcal{Q}$ defines a unary residue relation ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime } \in  {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ where ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ . Since ${e}^{\prime } = \{ X\}$ ,we can as well write the relation as ${R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }$ . Now that every such relation has the same scheme $\{ X\}$ ,we can define:

首先，观察可知，由${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$定义的超图始终是${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$，与$\mathbf{\eta }$无关。考虑一个边界属性$X \in  \mathcal{L}$，以及$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$中与$X$相关联的一条交叉边$e$。正如第3节所解释的，输入关系${R}_{e} \in  \mathcal{Q}$定义了一个一元残差关系${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime } \in  {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$，其中${e}^{\prime } = e \smallsetminus  \mathcal{H}$。由于${e}^{\prime } = \{ X\}$，我们也可以将该关系写为${R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }$。既然每个这样的关系都有相同的模式$\{ X\}$，我们可以定义：

$$
{R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime } = \mathop{\bigcap }\limits_{{\text{cross edge }e\text{ containing }X}}{R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }. \tag{3}
$$

Example (cont.). Let $\mathcal{H}$ and $\mathbf{\eta }$ be the same as in the earlier description of this example. Set $X$ to the border attribute $\mathrm{A}$ . If $e$ is the cross edge $\{ \mathrm{A},\mathrm{D}\}$ ,the example in Section 3 has shown that ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$ is the relation ${R}_{\{ \mathbf{A}\}  \mid  \mathbf{d}}^{\prime }$ obtained in Section 1.3. Similarly,if $e$ is the cross edge $\{ \mathrm{A},\mathrm{E}\} ,{R}_{{e}^{\prime }\left| {\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }\right| }^{\prime }$ is the relation ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ obtained in Section 1.3. As $\mathrm{A}$ is contained only in these two cross edges, ${R}_{\{ A\}  \mid  \eta }^{\prime \prime }$ is the intersection of ${R}_{\{ A\}  \mid  d}^{\prime }$ and ${R}_{\{ A\}  \mid  e}^{\prime }$ ,and corresponds to the relation ${R}_{\{ A\} }^{\prime \prime }$ given in Section 1.3. (to be continued) $\blacktriangle$

示例（续）。设$\mathcal{H}$和$\mathbf{\eta }$与本示例前面描述的相同。将$X$设为边界属性$\mathrm{A}$。如果$e$是交叉边$\{ \mathrm{A},\mathrm{D}\}$，第3节中的示例已表明${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime }$是在第1.3节中得到的关系${R}_{\{ \mathbf{A}\}  \mid  \mathbf{d}}^{\prime }$。类似地，如果$e$是交叉边$\{ \mathrm{A},\mathrm{E}\} ,{R}_{{e}^{\prime }\left| {\mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }\right| }^{\prime }$，则${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$是在第1.3节中得到的关系。由于$\mathrm{A}$仅包含在这两条交叉边中，${R}_{\{ A\}  \mid  \eta }^{\prime \prime }$是${R}_{\{ A\}  \mid  d}^{\prime }$和${R}_{\{ A\}  \mid  e}^{\prime }$的交集，并且对应于第1.3节中给出的关系${R}_{\{ A\} }^{\prime \prime }$。（待续）$\blacktriangle$

Consider a light edge $e = \{ X,Y\}$ in $\mathcal{G}$ . Recall that ${R}_{e}$ defines a residue relation ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime } \in  {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ ,which can be written as ${R}_{e \mid  \varnothing }^{\prime }$ because ${e}^{\prime } = e \smallsetminus  \mathcal{H} = e$ . We define ${R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$ as a relation over $e$ which consists of every tuple $\mathbf{u} \in  {R}_{e\left( \cdot \right) }^{\prime }$ satisfying both conditions below:

考虑$\mathcal{G}$中的一条轻边$e = \{ X,Y\}$。回顾一下，${R}_{e}$定义了一个剩余关系${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime } \in  {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$，由于${e}^{\prime } = e \smallsetminus  \mathcal{H} = e$，该关系可以写成${R}_{e \mid  \varnothing }^{\prime }$的形式。我们将${R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$定义为$e$上的一个关系，它由满足以下两个条件的每个元组$\mathbf{u} \in  {R}_{e\left( \cdot \right) }^{\prime }$组成：

- (applicable only if $X$ is a border attribute) $\mathbf{u}\left( X\right)  \in  {R}_{X \mid  \mathbf{\eta }}^{\prime \prime }$ ;

- （仅当$X$是边界属性时适用）$\mathbf{u}\left( X\right)  \in  {R}_{X \mid  \mathbf{\eta }}^{\prime \prime }$；

- (applicable only if $Y$ is a border attribute) $\mathbf{u}\left( Y\right)  \in  {R}_{Y \mid  \mathbf{\eta }}^{\prime \prime }$ .

- （仅当$Y$是边界属性时适用）$\mathbf{u}\left( Y\right)  \in  {R}_{Y \mid  \mathbf{\eta }}^{\prime \prime }$。

Note that if neither $X$ nor $Y$ is a border attribute,then ${R}_{e \mid  \eta }^{\prime \prime } = {R}_{e \mid  \varnothing }^{\prime }$ .

注意，如果$X$和$Y$都不是边界属性，那么${R}_{e \mid  \eta }^{\prime \prime } = {R}_{e \mid  \varnothing }^{\prime }$。

Example (cont.). Let us concentrate the light edge $e = \{ \mathrm{A},\mathrm{B}\}$ . The example in Section 3 has explained that ${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime } = {R}_{\{ \mathbf{A},\mathbf{B}\}  \mid  \varnothing }^{\prime }$ is the relation ${R}_{\{ \mathbf{A},\mathbf{B}\} }^{\prime }$ obtained in Section 1.3. As $\mathbf{A}$ and $\mathrm{B}$ are both border attributes, ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ includes all the tuples in ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ that take a value in ${R}_{\{ \mathrm{A}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ on attribute $\mathrm{A}$ and a value in ${R}_{\{ \mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ on attribute $\mathrm{B}$ . Note that ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ corresponds to the relation ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ given in Section 1.3.

示例（续）。让我们关注轻边$e = \{ \mathrm{A},\mathrm{B}\}$。第3节中的示例已经解释过，${R}_{{e}^{\prime } \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }^{\prime } = {R}_{\{ \mathbf{A},\mathbf{B}\}  \mid  \varnothing }^{\prime }$是在第1.3节中得到的关系${R}_{\{ \mathbf{A},\mathbf{B}\} }^{\prime }$。由于$\mathbf{A}$和$\mathrm{B}$都是边界属性，${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$包含${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$中所有在属性$\mathrm{A}$上取值于${R}_{\{ \mathrm{A}\}  \mid  \mathbf{\eta }}^{\prime \prime }$且在属性$\mathrm{B}$上取值于${R}_{\{ \mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$的元组。注意，${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$对应于第1.3节中给出的关系${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$。

Every vertex $X \in  \mathcal{I}$ must be a border attribute,and thus must have ${R}_{X \mid  \eta }^{\prime \prime }$ defined. Therefore, we can legally define:

每个顶点$X \in  \mathcal{I}$都必须是边界属性，因此必须定义${R}_{X \mid  \eta }^{\prime \prime }$。所以，我们可以合法地定义：

$$
{\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\prime \prime } = \left\{  {{R}_{e \mid  \mathbf{\eta }}^{\prime \prime } \mid  \text{ light edge }e \in  \mathcal{E}}\right\}  
$$

$$
{\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime } = \left\{  {{R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime } \mid  X \in  \mathcal{I}}\right\}  
$$

$$
{\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime } = {\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\prime \prime } \cup  {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }.
$$

Notice that the join queries ${\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime },{\mathcal{Q}}_{\text{light} \mid  \mathbf{\eta }}^{\prime \prime }$ ,and ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ are all scheme-clean.

注意，连接查询${\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime },{\mathcal{Q}}_{\text{light} \mid  \mathbf{\eta }}^{\prime \prime }$和${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$都是模式干净的。

Example (cont.). ${\mathcal{Q}}_{\text{light} \mid  \eta }^{\prime \prime }$ consists of ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }$ ,and ${R}_{\{ \mathrm{I},\mathrm{J}\} }$ ,and ${\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }$ consists of ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ ,and ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$ ,where all the relation names follow those given in Section 1.3.

示例（续）。${\mathcal{Q}}_{\text{light} \mid  \eta }^{\prime \prime }$ 由 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }$ 以及 ${R}_{\{ \mathrm{I},\mathrm{J}\} }$ 组成，并且 ${\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }$ 由 ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ 以及 ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$ 组成，其中所有关系名遵循 1.3 节中给出的定义。

Proposition 8. $\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)  = \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{\operatorname{light} \mid  \mathbf{\eta }}^{\prime \prime }\right)$ .

命题 8。$\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)  = \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{\operatorname{light} \mid  \mathbf{\eta }}^{\prime \prime }\right)$。

Proof. See Appendix C.

证明。见附录 C。

We will refer to the above process of converting ${\mathcal{Q}}_{\eta }^{\prime }$ to ${\mathcal{Q}}_{\eta }^{\prime \prime }$ as semi-join reduction,and call ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ the reduced query of $\mathbf{\eta }$ .

我们将上述把 ${\mathcal{Q}}_{\eta }^{\prime }$ 转换为 ${\mathcal{Q}}_{\eta }^{\prime \prime }$ 的过程称为半连接约简，并将 ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ 称为 $\mathbf{\eta }$ 的约简查询。

### 4.3 The Isolated Cartesian Product Theorem

### 4.3 孤立笛卡尔积定理

We are ready to present the first main result of this paper:

我们准备给出本文的第一个主要结果：

Theorem 9 (The Isolated Cartesian Product Theorem).

定理 9（孤立笛卡尔积定理）。

$$
\mathop{\sum }\limits_{{\text{config. }\mathbf{\eta }}}\left| {\operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right) }\right|  = O\left( {{\lambda }^{2\left( {\rho  - \left| \mathcal{I}\right| }\right)  - \left| {\mathcal{L} \smallsetminus  \mathcal{I}}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }}\right)  \tag{4}
$$

where $\rho$ is the fractional edge covering number of $\mathcal{Q}$ .

其中 $\rho$ 是 $\mathcal{Q}$ 的分数边覆盖数。

The rest of the section serves as a proof of the above theorem. To start with,define $\mathcal{F}$ to be the set of attributes in $\mathcal{H}$ that are adjacent to at least one isolated vertex in $\mathcal{G}$ . The left hand side of (4) can be bounded by looking at only the configurations of $\mathcal{F}$ :

本节的其余部分用于证明上述定理。首先，将 $\mathcal{F}$ 定义为 $\mathcal{H}$ 中与 $\mathcal{G}$ 中至少一个孤立顶点相邻的属性集。通过仅考虑 $\mathcal{F}$ 的配置，可以对 (4) 的左侧进行界定：

- Lemma 10. $\mathop{\sum }\limits_{{\text{config. }\mathbf{\eta }\text{ of }\mathcal{H}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right) }\right|  = O\left( {\lambda }^{\left| \mathcal{H}\left| -\right| \mathcal{F}\right| }\right)  \cdot  \mathop{\sum }\limits_{{\text{config. }{\mathbf{\eta }}^{\prime }\text{ of }\mathcal{F}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }\right) }\right|$ .

- 引理 10。$\mathop{\sum }\limits_{{\text{config. }\mathbf{\eta }\text{ of }\mathcal{H}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right) }\right|  = O\left( {\lambda }^{\left| \mathcal{H}\left| -\right| \mathcal{F}\right| }\right)  \cdot  \mathop{\sum }\limits_{{\text{config. }{\mathbf{\eta }}^{\prime }\text{ of }\mathcal{F}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }\right) }\right|$。

Proof. See Appendix D.

证明。见附录 D。

Now,let us take a fractional edge packing $W$ of the hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ that obeys the second bullet of Lemma 1. Denote by $\tau$ the total weight of $W$ which,by definition of $W$ , is the fractional edge packing number of $\mathcal{G}$ . Define:

现在，让我们取超图 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ 的一个分数边填充 $W$，它满足引理 1 的第二个要点。用 $\tau$ 表示 $W$ 的总权重，根据 $W$ 的定义，它是 $\mathcal{G}$ 的分数边填充数。定义：

$$
\mathcal{Z} = \left\{  {X \in  \mathcal{V}\left| {\;\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)  = 0}\right. }\right\}  
$$

that is, $Z$ is the set of vertices with weight 0 under $W$ . Set ${\mathcal{I}}_{0} = \mathcal{I} \cap  \mathcal{Z}$ and ${\mathcal{I}}_{1} = \mathcal{I} \smallsetminus  {\mathcal{I}}_{0}$ . Since $W$ satisfies Condition 1 of the second bullet in Lemma 1,we know that every vertex in ${\mathcal{I}}_{1}$ has weight 1,while every vertex in ${\mathcal{I}}_{0}$ has weight 0 .

也就是说，$Z$ 是在 $W$ 下权重为 0 的顶点集合。设 ${\mathcal{I}}_{0} = \mathcal{I} \cap  \mathcal{Z}$ 和 ${\mathcal{I}}_{1} = \mathcal{I} \smallsetminus  {\mathcal{I}}_{0}$。由于 $W$ 满足引理 1 中第二点的条件 1，我们知道 ${\mathcal{I}}_{1}$ 中的每个顶点权重为 1，而 ${\mathcal{I}}_{0}$ 中的每个顶点权重为 0。

Example. Let $\mathcal{G}$ be the hypergraph in Figure 1a. As explained by the example in Section 2.2, the fractional edge packing number $\tau$ of $\mathcal{G}$ is achieved by the function $W$ that maps $\{ \mathbf{G}$ , $\mathrm{F}\} ,\{ \mathrm{D},\mathrm{K}\} ,\{ \mathrm{I},\mathrm{J}\}$ ,and $\{ \mathrm{E},\mathrm{H}\}$ to $1,\{ \mathrm{A},\mathrm{B}\} ,\{ \mathrm{A},\mathrm{C}\}$ ,and $\{ \mathrm{B},\mathrm{C}\}$ to $1/2$ ,and the other edges to 0; $\mathcal{Z}$ contains a single vertex $\mathrm{L}$ . Setting $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ yields $\mathcal{I} = \{ \mathrm{G},\mathrm{H},\mathrm{L}\} ,{\mathcal{I}}_{0} = \{ \mathrm{L}\}$ , and ${\mathcal{I}}_{1} = \{ \mathrm{G},\mathrm{H}\} .\mathcal{F} = \{ \mathrm{D},\mathrm{E},\mathrm{F}\}$ ,noticing that $\mathrm{K}$ is not adjacent to any isolated vertex. (to be continued) $\blacktriangle$

示例。设 $\mathcal{G}$ 为图 1a 中的超图。正如 2.2 节中的示例所解释的，$\mathcal{G}$ 的分数边填充数 $\tau$ 由函数 $W$ 实现，该函数将 $\{ \mathbf{G}$、$\mathrm{F}\} ,\{ \mathrm{D},\mathrm{K}\} ,\{ \mathrm{I},\mathrm{J}\}$ 和 $\{ \mathrm{E},\mathrm{H}\}$ 映射到 $1,\{ \mathrm{A},\mathrm{B}\} ,\{ \mathrm{A},\mathrm{C}\}$，将 $\{ \mathrm{B},\mathrm{C}\}$ 映射到 $1/2$，将其他边映射到 0；$\mathcal{Z}$ 包含单个顶点 $\mathrm{L}$。设 $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ 可得 $\mathcal{I} = \{ \mathrm{G},\mathrm{H},\mathrm{L}\} ,{\mathcal{I}}_{0} = \{ \mathrm{L}\}$，以及 ${\mathcal{I}}_{1} = \{ \mathrm{G},\mathrm{H}\} .\mathcal{F} = \{ \mathrm{D},\mathrm{E},\mathrm{F}\}$，注意到 $\mathrm{K}$ 不与任何孤立顶点相邻。（待续）$\blacktriangle$

We now present a crucial lemma which is in fact a stronger version of Theorem 9:

我们现在给出一个关键引理，实际上它是定理 9 的一个更强版本：

Lemma 11. $\mathop{\sum }\limits_{{\text{config. }{\mathbf{\eta }}^{\prime }\text{ of }\mathcal{F}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }\right) }\right|  = O\left( {{\lambda }^{\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }}\right)$ .

引理 11。$\mathop{\sum }\limits_{{\text{config. }{\mathbf{\eta }}^{\prime }\text{ of }\mathcal{F}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }\right) }\right|  = O\left( {{\lambda }^{\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }}\right)$。

Before proving the above lemma, let us first see how it can be used to complete the proof of Theorem 9. By combining Lemmas 10 and 11, we know that the left hand side of (4) is $O\left( {{\lambda }^{\left| \mathcal{H}\right|  - \left| {\mathcal{I}}_{1}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }}\right)$ . Hence,it suffices to prove

在证明上述引理之前，让我们先看看如何用它来完成定理 9 的证明。结合引理 10 和 11，我们知道 (4) 式的左边是 $O\left( {{\lambda }^{\left| \mathcal{H}\right|  - \left| {\mathcal{I}}_{1}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }}\right)$。因此，只需证明

$$
\left| \mathcal{H}\right|  - \left| {\mathcal{I}}_{1}\right|  \leq  2\left( {\rho  - \left| \mathcal{I}\right| }\right)  - \left| {\mathcal{L} \smallsetminus  \mathcal{I}}\right|  \Leftrightarrow  
$$

$$
\left| \mathcal{H}\right|  + \left| {\mathcal{L} \smallsetminus  \mathcal{I}}\right|  + \left| \mathcal{I}\right|  + \left| \mathcal{I}\right|  - \left| {\mathcal{I}}_{1}\right|  \leq  {2\rho } \Leftrightarrow  
$$

$$
\left| \mathcal{V}\right|  - \rho  + \left| {\mathcal{I}}_{0}\right|  \leq  \rho \left( {\text{ note: }\left| \mathcal{V}\right|  = \left| \mathcal{H}\right|  + \left| {\mathcal{L} \smallsetminus  \mathcal{I}}\right|  + \left| \mathcal{I}\right| }\right)  \Leftrightarrow  
$$

$$
\tau  + \left| {\mathcal{I}}_{0}\right|  \leq  \rho \text{(note:}\rho  + \tau  = \left| \mathcal{V}\right| \text{by Lemma 1)}
$$

which is true because $\rho  - \tau  = \left| \mathcal{Z}\right|$ by Condition 2 of the second bullet in Lemma 1,and ${\mathcal{I}}_{0} \subseteq  \mathcal{Z}$ .

这是成立的，因为根据引理 1 中第二点的条件 2 有 $\rho  - \tau  = \left| \mathcal{Z}\right|$，并且 ${\mathcal{I}}_{0} \subseteq  \mathcal{Z}$。

Proof of Lemma 11. Our idea is to construct a set ${\mathcal{Q}}^{ * }$ of relations such that $\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ has a result size at least the left hand side of the inequality in Lemma 11. Then, we will prove that the hypergraph of ${\mathcal{Q}}^{ * }$ has a certain fractional edge covering which,together with the AGM bound,yields an upper bound on $\left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$ which happens to be the right hand side of the inequality.

引理11的证明。我们的思路是构造一个关系集合${\mathcal{Q}}^{ * }$，使得$\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$的结果大小至少为引理11中不等式的左侧。然后，我们将证明${\mathcal{Q}}^{ * }$的超图具有某种分数边覆盖，结合AGM界（AGM bound），可以得到$\left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$的一个上界，而这个上界恰好是不等式的右侧。

We construct ${\mathcal{Q}}^{ * }$ as follows. Initially,set ${\mathcal{Q}}^{ * }$ to $\varnothing$ . For every cross edge $e \in  \mathcal{E}$ incident to an isolated vertex,add to ${\mathcal{Q}}^{ * }$ a relation ${R}_{e}^{ * } = {R}_{e}$ . For every $X \in  \mathcal{F}$ ,add a unary relation ${R}_{\{ X\} }^{ * }$ to ${\mathcal{Q}}^{ * }$ which consists of all the heavy values on attribute $X$ . Note that ${R}_{\{ X\} }^{ * }$ has $O\left( \lambda \right)$ tuples. Finally,for every $Y \in  {\mathcal{I}}_{0}$ ,add a unary relation ${R}_{\{ Y\} }^{ * }$ to ${\mathcal{Q}}^{ * }$ which contains all the heavy and light values on attribute $Y$ .

我们按如下方式构造${\mathcal{Q}}^{ * }$。初始时，将${\mathcal{Q}}^{ * }$设为$\varnothing$。对于每条与孤立顶点相关联的交叉边$e \in  \mathcal{E}$，向${\mathcal{Q}}^{ * }$中添加一个关系${R}_{e}^{ * } = {R}_{e}$。对于每个$X \in  \mathcal{F}$，向${\mathcal{Q}}^{ * }$中添加一个一元关系${R}_{\{ X\} }^{ * }$，该关系由属性$X$上的所有重值（heavy values）组成。注意，${R}_{\{ X\} }^{ * }$有$O\left( \lambda \right)$个元组。最后，对于每个$Y \in  {\mathcal{I}}_{0}$，向${\mathcal{Q}}^{ * }$中添加一个一元关系${R}_{\{ Y\} }^{ * }$，该关系包含属性$Y$上的所有重值和轻值（light values）。

Define ${\mathcal{G}}^{ * } = \left( {{\mathcal{V}}^{ * },{\mathcal{E}}^{ * }}\right)$ as the hypergraph defined by ${\mathcal{Q}}^{ * }$ . Note that ${\mathcal{V}}^{ * } = \mathcal{I} \cup  \mathcal{F}$ ,while ${\mathcal{E}}^{ * }$ consists of all the cross edges in $\mathcal{G},\left| \mathcal{F}\right|$ unary edges $\{ X\}$ for every $X \in  \mathcal{F}$ ,and $\left| {\mathcal{I}}_{0}\right|$ unary edges $\{ Y\}$ for every $Y \in  {\mathcal{I}}_{0}$ . 25:12 A Simple Parallel Algorithm for Natural Joins on Binary Relations

将${\mathcal{G}}^{ * } = \left( {{\mathcal{V}}^{ * },{\mathcal{E}}^{ * }}\right)$定义为由${\mathcal{Q}}^{ * }$所定义的超图。注意，${\mathcal{V}}^{ * } = \mathcal{I} \cup  \mathcal{F}$，而${\mathcal{E}}^{ * }$由$\mathcal{G},\left| \mathcal{F}\right|$中的所有交叉边、每个$X \in  \mathcal{F}$对应的一元边$\{ X\}$以及每个$Y \in  {\mathcal{I}}_{0}$对应的一元边$\{ Y\}$组成。25:12 二元关系自然连接的简单并行算法

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195ce8e-ebf0-7dc5-b30b-c8220ede8665_11.jpg?x=678&y=261&w=328&h=297&r=0"/>

Figure 3 Illustration of ${\mathcal{Q}}^{ * }$ .

图3 ${\mathcal{Q}}^{ * }$的图示。

<!-- Media -->

Example (cont.). Figure 3 shows the hypergraph of the ${\mathcal{Q}}^{ * }$ constructed,where as before a box and a segment represent a unary and a binary edge,respectively. (to be continued) $\blacktriangle$

示例（续）。图3展示了所构造的${\mathcal{Q}}^{ * }$的超图，和之前一样，方框和线段分别表示一元边和二元边。（待续）$\blacktriangle$

- Lemma 12. $\mathop{\sum }\limits_{{\text{config. }{\eta }^{\prime }\text{ of }\mathcal{F}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\eta }^{\prime }}^{\prime \prime }\right) }\right|  \leq  \left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$ .

- 引理12。$\mathop{\sum }\limits_{{\text{config. }{\eta }^{\prime }\text{ of }\mathcal{F}}}\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\eta }^{\prime }}^{\prime \prime }\right) }\right|  \leq  \left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$。

Proof. See Appendix E.

证明。见附录E。

- Lemma 13. ${\mathcal{G}}^{ * }$ admits a tight fractional edge covering ${\mathcal{W}}^{ * }$ satisfying $\mathop{\sum }\limits_{{X \in  F}}{W}^{ * }\left( {\{ X\} }\right)  =$ $\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right|$ .

- 引理13。${\mathcal{G}}^{ * }$存在一个紧分数边覆盖（tight fractional edge covering）${\mathcal{W}}^{ * }$，满足$\mathop{\sum }\limits_{{X \in  F}}{W}^{ * }\left( {\{ X\} }\right)  =$ $\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right|$。

Proof. Recall that our proof of Theorem 9 began with a fractional edge packing $W$ of $\mathcal{G}$ . We construct a desired function ${W}^{ * }$ from $W$ as follows. First,for every cross edge $e \in  \mathcal{E}$ ,set ${W}^{ * }\left( e\right)  = W\left( e\right)$ . Observe that every edge in $\mathcal{E}$ incident to $Y \in  \mathcal{I}$ must be a cross edge. Hence, $\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)$ is precisely the weight of $Y$ under $W$ . By definition of $W$ ,we thus have ensured $\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)  = 1$ for each $Y \in  {\mathcal{I}}_{1}$ ,and $\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)  = 0$ for each $Y \in  {\mathcal{I}}_{0}$ . As a second step,we set ${W}^{ * }\left( {\{ Y\} }\right)  = 1$ for each $Y \in  {\mathcal{I}}_{0}$ so that the edges in ${\mathcal{E}}^{ * }$ containing $Y$ have a total weight of 1 .

证明。回顾一下，我们对定理9的证明始于$\mathcal{G}$的一个分数边填充$W$。我们从$W$出发，按如下方式构造所需的函数${W}^{ * }$。首先，对于每一条交叉边$e \in  \mathcal{E}$，令${W}^{ * }\left( e\right)  = W\left( e\right)$。注意，$\mathcal{E}$中与$Y \in  \mathcal{I}$关联的每一条边必定是交叉边。因此，$\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)$恰好是$Y$在$W$下的权重。根据$W$的定义，我们由此确保了对于每个$Y \in  {\mathcal{I}}_{1}$有$\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)  = 1$，并且对于每个$Y \in  {\mathcal{I}}_{0}$有$\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)  = 0$。第二步，对于每个$Y \in  {\mathcal{I}}_{0}$，我们令${W}^{ * }\left( {\{ Y\} }\right)  = 1$，使得${\mathcal{E}}^{ * }$中包含$Y$的边的总权重为1。

It remains to make sure that each attribute $X \in  \mathcal{F}$ has a weight 1 under ${W}^{ * }$ . Since $W$ is a fractional edge packing of $\mathcal{G}$ ,it must hold that $\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right)  \leq  1$ . This permits us to assign the following weight to the unary edge $\{ X\}$ :

还需要确保每个属性$X \in  \mathcal{F}$在${W}^{ * }$下的权重为1。由于$W$是$\mathcal{G}$的一个分数边填充，那么必定有$\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right)  \leq  1$。这使我们能够为一元边$\{ X\}$分配如下权重：

$$
{W}^{ * }\left( {\{ X\} }\right)  = 1 - \mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right) .
$$

This finishes the design of ${W}^{ * }$ which is now a tight fractional edge covering of ${\mathcal{G}}^{ * }$ . Clearly:

这就完成了${W}^{ * }$的设计，此时${W}^{ * }$是${\mathcal{G}}^{ * }$的一个紧分数边覆盖。显然：

$$
\mathop{\sum }\limits_{{X \in  \mathcal{F}}}{W}^{ * }\left( {\{ X\} }\right)  = \left| \mathcal{F}\right|  - \mathop{\sum }\limits_{{X \in  \mathcal{F}}}\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right) . \tag{5}
$$

Every binary edge $e \in  {\mathcal{E}}^{ * }$ contains a vertex in $\mathcal{F}$ and a vertex in $\mathcal{I}$ . Therefore:

每一条二元边$e \in  {\mathcal{E}}^{ * }$都包含$\mathcal{F}$中的一个顶点和$\mathcal{I}$中的一个顶点。因此：

$$
\mathop{\sum }\limits_{{X \in  \mathcal{F}}}\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right)  = \mathop{\sum }\limits_{{Y \in  \mathcal{I}}}\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}W\left( e\right)  = \left| {\mathcal{I}}_{1}\right| .
$$

Putting together the above equation with (5) completes the proof.

将上述等式与(5)结合起来，就完成了证明。

Example (cont.). For the ${\mathcal{G}}^{ * }$ in Figure 3,a fractional edge covering in Lemma 13 is given by the function ${W}^{ * }$ that maps $\{ \mathtt{G},\mathtt{F}\} ,\{ \mathtt{E},\mathtt{H}\} ,\{ \mathtt{D}\}$ ,and $\{ \mathtt{L}\}$ to 1,and the other edges to 0 . Note that $\mathop{\sum }\limits_{{X \in  F}}{W}^{ * }\left( {\{ X\} }\right)  = {W}^{ * }\left( {\{ \mathrm{D}\} }\right)  = 1$ ,same as $\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right|  = 3 - 2 = 1$ .

示例（续）。对于图3中的${\mathcal{G}}^{ * }$，引理13中的一个分数边覆盖由函数${W}^{ * }$给出，该函数将$\{ \mathtt{G},\mathtt{F}\} ,\{ \mathtt{E},\mathtt{H}\} ,\{ \mathtt{D}\}$和$\{ \mathtt{L}\}$映射到1，将其他边映射到0。注意，$\mathop{\sum }\limits_{{X \in  F}}{W}^{ * }\left( {\{ X\} }\right)  = {W}^{ * }\left( {\{ \mathrm{D}\} }\right)  = 1$与$\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right|  = 3 - 2 = 1$相同。

The AGM bound in Lemma 2 tells us that

引理2中的AGM（算术 - 几何平均）界告诉我们

$$
\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)  \leq  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * }}}{\left| {R}_{e}^{ * }\right| }^{{W}^{ * }\left( e\right) } = \left( {\mathop{\prod }\limits_{{X \in  \mathcal{F}}}{\left| {R}_{\{ X\} }^{ * }\right| }^{\left. {W}^{ * }\left( \{ X\} \right) \right) }}\right) \left( {\mathop{\prod }\limits_{{Y \in  \mathcal{I}}}\mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * } : Y \in  e}}{\left| {R}_{e}^{ * }\right| }^{{W}^{ * }\left( e\right) }}\right) 
$$

$$
 = \left( {\mathop{\prod }\limits_{{X \in  \mathcal{F}}}{\left( O\left( \lambda \right) \right) }^{{W}^{ * }\left( {\{ X\} }\right) }}\right) \left( {\mathop{\prod }\limits_{{Y \in  \mathcal{I}}}\mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * } : Y \in  e}}{m}^{{W}^{ * }\left( e\right) }}\right) 
$$

$$
\text{(by Lemma 13)} = O\left( {\lambda }^{\left| \mathcal{F}\right|  - \left| {\mathcal{I}}_{1}\right| }\right)  \cdot  {m}^{\left| \mathcal{I}\right| }
$$

which completes the proof of Lemma 11.

这就完成了引理11的证明。

## 5 A 5-Round MPC Algorithm

## 5 一个5轮多方计算（MPC）算法

We now proceed to implement the strategy discussed in the previous section under the MPC model. Our objective in this section is to explain how the isolated cartesian product theorem can be utilized to answer a simple join query $\mathcal{Q}$ with the optimal load $O\left( {m/{p}^{1/\rho }}\right)$ in a rather straightforward manner. Hence, we intentionally leave out the optimization tricks to reduce the number of rounds, but even so, our algorithm finishes in only 5 rounds. Those tricks are the topic of the next section.

我们现在着手在多方计算（MPC）模型下实现上一节讨论的策略。本节的目标是解释如何利用孤立笛卡尔积定理，以一种相当直接的方式，用最优负载 $O\left( {m/{p}^{1/\rho }}\right)$ 来回答一个简单的连接查询 $\mathcal{Q}$。因此，我们有意省略了减少轮数的优化技巧，但即便如此，我们的算法也只需 5 轮即可完成。这些技巧将在下一节讨论。

A statistical record is defined as a tuple(R,X,x, cnt),where $R$ is a relation in $\mathcal{Q},X$ an attribute in scheme $\left( R\right) ,x$ a value in $\mathbf{{dom}}$ ,and cnt the number of tuples $\mathbf{u} \in  R$ with $\mathbf{u}\left( X\right)  = x$ . Specially, $\left( {R,\varnothing ,{nil},{cnt}}\right)$ is also regarded as a statistical record where ${cnt}$ gives the number of tuples in $R$ that use only light values. A histogram is defined as the set of statistical records for all possible $R,X$ ,and $x$ satisfying (i) cn $t = \Omega \left( {m/{p}^{1/\rho }}\right)$ ,or (ii) $X = \varnothing$ (and,hence $x = {nil}$ ); note that there are only $O\left( {p}^{1/\rho }\right)$ such records. We assume that every machine has a local copy of the histogram. It is worth mentioning that all existing join algorithms $\left\lbrack  {5,9}\right\rbrack$ ,which strive to finish in a specifically small - rather than just asymptotically constant - number of rounds,demand that each machine should be preloaded with ${p}^{O\left( 1\right) }$ statistical records.

统计记录定义为一个元组 (R, X, x, cnt)，其中 $R$ 是 $\mathcal{Q},X$ 中的一个关系，$\left( R\right) ,x$ 是模式中的一个属性，$\mathbf{{dom}}$ 中的一个值，cnt 是满足 $\mathbf{u}\left( X\right)  = x$ 的元组 $\mathbf{u} \in  R$ 的数量。特别地，$\left( {R,\varnothing ,{nil},{cnt}}\right)$ 也被视为一个统计记录，其中 ${cnt}$ 给出了 $R$ 中仅使用轻值的元组数量。直方图定义为所有可能的 $R,X$ 和 $x$ 满足以下条件的统计记录集合：(i) cn $t = \Omega \left( {m/{p}^{1/\rho }}\right)$，或 (ii) $X = \varnothing$（因此 $x = {nil}$）；注意，这样的记录只有 $O\left( {p}^{1/\rho }\right)$ 条。我们假设每台机器都有直方图的本地副本。值得一提的是，所有现有的连接算法 $\left\lbrack  {5,9}\right\rbrack$ 都力求在特定的小轮数（而非仅渐近常数轮数）内完成，要求每台机器预先加载 ${p}^{O\left( 1\right) }$ 条统计记录。

Henceforth,the value of $\lambda$ will be fixed to $\Theta \left( {p}^{1/\left( {2\rho }\right) }\right)$ . We focus on explaining how to compute (2) for an arbitrary subset $\mathcal{H}$ of attset(Q). Set $k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|$ . As attset(Q)has ${2}^{k} = O\left( 1\right)$ subsets,processing all of them in parallel increases the load only by a constant factor,and definitely discovers the entire $\operatorname{Join}\left( \mathcal{Q}\right)$ ,as is guaranteed by (1). Our algorithm produces (2) in three steps:

此后，$\lambda$ 的值将固定为 $\Theta \left( {p}^{1/\left( {2\rho }\right) }\right)$。我们专注于解释如何为 attset(Q) 的任意子集 $\mathcal{H}$ 计算 (2)。设 $k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|$。由于 attset(Q) 有 ${2}^{k} = O\left( 1\right)$ 个子集，并行处理所有子集只会使负载增加一个常数因子，并且肯定能发现整个 $\operatorname{Join}\left( \mathcal{Q}\right)$，正如 (1) 所保证的那样。我们的算法分三步计算 (2)：

1. Generate the input relations of the residue query ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ of every configuration $\mathbf{\eta }$ of $\mathcal{H}$ .

1. 为 $\mathcal{H}$ 的每个配置 $\mathbf{\eta }$ 生成残差查询 ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ 的输入关系。

2. Generate the input relations of the reduced query ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ of every $\mathbf{\eta }$ .

2. 为每个 $\mathbf{\eta }$ 生成简化查询 ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ 的输入关系。

3. Evaluate ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ for every $\mathbf{\eta }$ .

3. 为每个 $\mathbf{\eta }$ 计算 ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$。

The number of configurations of $\mathcal{H}$ is $O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)  = O\left( {\lambda }^{k}\right)  = O\left( {p}^{k/\left( {2\rho }\right) }\right)$ ,which is $O\left( p\right)$ because $\rho  \geq  k/2$ by the first bullet of Lemma 1. Next,we elaborate on the details of each step.

$\mathcal{H}$ 的配置数量为 $O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)  = O\left( {\lambda }^{k}\right)  = O\left( {p}^{k/\left( {2\rho }\right) }\right)$，根据引理 1 的第一点，由于 $\rho  \geq  k/2$，所以该数量为 $O\left( p\right)$。接下来，我们详细阐述每一步的细节。

Step 1. Proposition 7 tells us that the input relations of all the residue queries have $O\left( {m \cdot  {\lambda }^{k - 2}}\right)$ tuples in total. We allocate ${p}_{\mathbf{\eta }}^{\prime } = \left\lceil  {p \cdot  \frac{{m}_{\mathbf{\eta }}}{\Theta \left( {m \cdot  {\lambda }^{k - 2}}\right) }}\right\rceil$ machines to store the relations of ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ ,so that each machine assigned to ${Q}_{\mathbf{\eta }}^{\prime }$ keeps on average $O\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = O\left( {m \cdot  {\lambda }^{k - 2}/p}\right)  =$ $O\left( {m/{p}^{1/\rho }}\right)$ tuples,where the last equality used $\rho  \geq  k/2$ . Since each machine $i \in  \left\lbrack  {1,p}\right\rbrack$ can use the histogram to calculate ${m}_{\mathbf{\eta }}$ precisely for each $\mathbf{\eta }$ ,it can compute locally the id range of the ${m}_{\mathbf{\eta }}$ machines responsible for ${Q}_{\mathbf{\eta }}^{\prime }$ . If a tuple $\mathbf{u}$ in the local storage of machine $i$ belongs

步骤1. 命题7告诉我们，所有残差查询的输入关系总共具有$O\left( {m \cdot  {\lambda }^{k - 2}}\right)$个元组。我们分配${p}_{\mathbf{\eta }}^{\prime } = \left\lceil  {p \cdot  \frac{{m}_{\mathbf{\eta }}}{\Theta \left( {m \cdot  {\lambda }^{k - 2}}\right) }}\right\rceil$台机器来存储${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$的关系，使得分配给${Q}_{\mathbf{\eta }}^{\prime }$的每台机器平均保存$O\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = O\left( {m \cdot  {\lambda }^{k - 2}/p}\right)  =$ $O\left( {m/{p}^{1/\rho }}\right)$个元组，其中最后一个等式使用了$\rho  \geq  k/2$。由于每台机器$i \in  \left\lbrack  {1,p}\right\rbrack$可以使用直方图为每个$\mathbf{\eta }$精确计算${m}_{\mathbf{\eta }}$，因此它可以在本地计算负责${Q}_{\mathbf{\eta }}^{\prime }$的${m}_{\mathbf{\eta }}$台机器的id范围。如果机器$i$本地存储中的一个元组$\mathbf{u}$属于

to ${Q}_{\eta }^{\prime }$ ,the machine sends $\mathbf{u}$ to a random machine within that id range. Standard analysis shows that each of the ${m}_{\eta }$ machines receives roughly the same number of tuples,such that this step can be done in a single round with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ .

${Q}_{\eta }^{\prime }$，则该机器将$\mathbf{u}$发送到该id范围内的一台随机机器。标准分析表明，${m}_{\eta }$台机器中的每台机器接收的元组数量大致相同，因此这一步可以在一轮内完成，负载为$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$。

Step 2. Now that all the input relations of each ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$ have been stored on ${p}_{\mathbf{\eta }}^{\prime }$ machines,the semi-join reduction that converts ${\mathcal{Q}}_{\eta }^{\prime }$ to ${\mathcal{Q}}_{\eta }^{\prime \prime }$ becomes a standard process [9] that can be accomplished in 2 rounds with load $\widetilde{O}\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ .

步骤2. 现在，每个${\mathcal{Q}}_{\mathbf{\eta }}^{\prime }$的所有输入关系都已存储在${p}_{\mathbf{\eta }}^{\prime }$台机器上，将${\mathcal{Q}}_{\eta }^{\prime }$转换为${\mathcal{Q}}_{\eta }^{\prime \prime }$的半连接约简成为一个标准过程[9]，可以在两轮内完成，负载为$\widetilde{O}\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$。

Step 3. This step starts by letting each machine know about the value of $\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }\right) }\right|$ for every $\mathbf{\eta }$ . For this purpose,each machine can broadcast to all machines how many tuples it has in ${R}_{\{ X\}  \mid  \eta }^{\prime \prime }$ for every $X \in  \mathcal{I}$ and every $\mathbf{\eta }$ . Since there are $O\left( p\right)$ different $\mathbf{\eta }$ ,at most $O\left( p\right)$ numbers are broadcast by each machine,such that the load of this round is $O\left( {p}^{2}\right)$ . With all these numbers,each machine can figure out independently the values of all $\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }\right) }\right|$ . We will call this round the statistical round henceforth.

步骤3. 这一步首先让每台机器了解每个$\mathbf{\eta }$对应的$\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }\right) }\right|$的值。为此，每台机器可以向所有机器广播它在${R}_{\{ X\}  \mid  \eta }^{\prime \prime }$中针对每个$X \in  \mathcal{I}$和每个$\mathbf{\eta }$拥有的元组数量。由于有$O\left( p\right)$个不同的$\mathbf{\eta }$，每台机器最多广播$O\left( p\right)$个数字，因此这一轮的负载为$O\left( {p}^{2}\right)$。有了所有这些数字，每台机器可以独立计算出所有$\left| {{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }\right) }\right|$的值。此后，我们将这一轮称为统计轮。

We allocate

我们分配

$$
{p}_{\mathbf{\eta }}^{\prime \prime } = \left\lbrack  {p \cdot  \frac{\left| CP\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right) \right| }{\Theta \left( {{\lambda }^{2\left( {\rho  - \left| \mathcal{I}\right| }\right) } - \left| {\mathcal{L} \smallsetminus  \mathcal{I}}\right|  \cdot  {m}^{\left| \mathcal{I}\right| }}\right) }}\right\rbrack   \tag{6}
$$

machines to computing ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ . Theorem 9 guarantees that the total number of machines needed by all the configurations is at most $p$ . We complete the algorithm with the lemma below:

台机器来计算${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$。定理9保证所有配置所需的机器总数最多为$p$。我们用下面的引理完成该算法：

Lemma 14. ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ can be answered in one round with load $O\left( {m/{p}^{1/\rho }}\right)$ using ${p}_{\mathbf{\eta }}^{\prime \prime }$ machines.

引理14. ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ 可以在一轮内使用 ${p}_{\mathbf{\eta }}^{\prime \prime }$ 台机器以负载 $O\left( {m/{p}^{1/\rho }}\right)$ 进行解答。

Proof. $\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ is the cartesian product of ${CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)$ and $\operatorname{Join}\left( {\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }\right)$ ,as shown Proposition 8. By Lemma 3,if we deploy ${p}_{\mathbf{\eta }}^{\prime \prime }/{\lambda }^{\mathcal{L} \smallsetminus  \mathcal{I}}$ machines to compute $\operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)$ in one round, the load is

证明。如命题8所示，$\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ 是 ${CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)$ 和 $\operatorname{Join}\left( {\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }\right)$ 的笛卡尔积。根据引理3，如果我们部署 ${p}_{\mathbf{\eta }}^{\prime \prime }/{\lambda }^{\mathcal{L} \smallsetminus  \mathcal{I}}$ 台机器在一轮内计算 $\operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)$，负载为

$$
\widetilde{O}\left( \frac{{CP}{\left( {\mathcal{Q}}_{\mathcal{I} \mid  \eta }^{\prime \prime }\right) }^{1/\left| \mathcal{I}\right| }}{{\left( \frac{{p}_{\eta }^{\prime \prime }}{{\mathcal{K}}^{\left| \mathcal{I}\right| }}\right) }^{1/\left| \mathcal{I}\right| }}\right)  = \widetilde{O}\left( \frac{m \cdot  {\lambda }^{\frac{2\left( {\rho  - \left| \mathcal{I}\right| }\right) }{\left| \mathcal{I}\right| }}}{{p}^{1/\left| \mathcal{I}\right| }}\right)  = \widetilde{O}\left( \frac{m \cdot  {p}^{\frac{2\left( {\rho  - \left| \mathcal{I}\right| }\right) }{{2\rho }\left| \mathcal{I}\right| }}}{{p}^{1/\left| \mathcal{I}\right| }}\right)  = \widetilde{O}\left( \frac{m}{{p}^{1/\rho }}\right) .
$$

Regarding ${\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }$ ,first verify that attset $\left( {\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }\right)  = \mathcal{L} \smallsetminus  \mathcal{I}$ . Recall that the input relations of ${\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }$ contain only light values. Hence,this join query is skew-free if we assign a share of $\lambda$ to each attribute in $\mathcal{L} \smallsetminus  \mathcal{I}$ . By Lemma 5,we can solve it in one round with load $\widetilde{O}\left( {m/{\lambda }^{2}}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ using ${\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }$ machines.

关于 ${\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }$，首先验证属性集 $\left( {\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }\right)  = \mathcal{L} \smallsetminus  \mathcal{I}$。回想一下，${\mathcal{Q}}_{{light} \mid  \mathbf{\eta }}^{\prime \prime }$ 的输入关系仅包含轻值。因此，如果我们将 $\lambda$ 的一部分分配给 $\mathcal{L} \smallsetminus  \mathcal{I}$ 中的每个属性，这个连接查询就是无偏斜的。根据引理5，我们可以使用 ${\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }$ 台机器在一轮内以负载 $\widetilde{O}\left( {m/{\lambda }^{2}}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ 解决它。

Lemma 4 now tells us that $\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ can be computed in one round with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ using $\left( {{p}_{\mathbf{\eta }}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }}\right)  \cdot  {\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| } = {p}_{\mathbf{\eta }}^{\prime \prime }$ machines.

现在引理4告诉我们，$\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ 可以在一轮内使用 $\left( {{p}_{\mathbf{\eta }}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }}\right)  \cdot  {\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| } = {p}_{\mathbf{\eta }}^{\prime \prime }$ 台机器以负载 $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ 进行计算。

## 6 A 3-Round MPC Algorithm

## 6 一种三轮多方计算算法

Next, we will improve the algorithm in Section 5 by reducing the number of rounds to 3 .

接下来，我们将通过将轮数减少到3轮来改进第5节中的算法。

### 6.1 A New Approach to Handle Light Edges

### 6.1 处理轻边的新方法

Let $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ be the hypergraph defined by the simple join query $\mathcal{Q}$ given. Fix an arbitrary subset $\mathcal{H}$ of attset(Q). Recall that,for every light edge $e \in  \mathcal{E}$ ,our 5-round algorithm needed to generate ${R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$ for every configuration $\mathbf{\eta }$ of $\mathcal{H}$ . Next,we will describe an alternative approach to perform the join without explicitly computing ${R}_{e \mid  \mathbf{p}}^{\prime \prime }$ ,which is crucial for obtaining a 3-round algorithm (to be presented in the next subsection).

设 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ 为由给定的简单连接查询 $\mathcal{Q}$ 定义的超图。固定属性集(Q)的任意子集 $\mathcal{H}$。回想一下，对于每条轻边 $e \in  \mathcal{E}$，我们的五轮算法需要为 $\mathcal{H}$ 的每个配置 $\mathbf{\eta }$ 生成 ${R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$。接下来，我们将描述一种在不明确计算 ${R}_{e \mid  \mathbf{p}}^{\prime \prime }$ 的情况下执行连接的替代方法，这对于获得三轮算法（将在下一小节介绍）至关重要。

Fix a configuration $\mathbf{\eta }$ of $\mathcal{H}$ . Consider a light edge $e$ of $\mathcal{G}$ ,and an attribute $X \in  e$ . Define ${R}_{e \mid  \mathbf{\eta }}^{\# X}$ as follows:

固定$\mathcal{H}$的一个配置$\mathbf{\eta }$。考虑$\mathcal{G}$的一条轻边$e$，以及一个属性$X \in  e$。按如下方式定义${R}_{e \mid  \mathbf{\eta }}^{\# X}$：

- if $X$ is not a border attribute, ${R}_{e \mid  \mathbf{\eta }}^{\# X} = {R}_{e \mid  \varnothing }^{\prime }$ ;

- 如果$X$不是边界属性，则为${R}_{e \mid  \mathbf{\eta }}^{\# X} = {R}_{e \mid  \varnothing }^{\prime }$；

- otherwise, ${R}_{e \mid  \mathbf{\eta }}^{\# X}$ is a relation over $e$ that consists of every tuple $\mathbf{u} \in  {R}_{e \mid  \varnothing }^{\prime }$ satisfying $\mathbf{u}\left( X\right)  \in  {R}_{X \mid  \mathbf{\eta }}^{\prime \prime }$

- 否则，${R}_{e \mid  \mathbf{\eta }}^{\# X}$是$e$上的一个关系，它由满足$\mathbf{u}\left( X\right)  \in  {R}_{X \mid  \mathbf{\eta }}^{\prime \prime }$的每个元组$\mathbf{u} \in  {R}_{e \mid  \varnothing }^{\prime }$组成

- Proposition 15. ${R}_{e \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{e \mid  \mathbf{\eta }}^{\# X} \bowtie  {R}_{e \mid  \mathbf{\eta }}^{\# Y}$ .

- 命题15。${R}_{e \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{e \mid  \mathbf{\eta }}^{\# X} \bowtie  {R}_{e \mid  \mathbf{\eta }}^{\# Y}$。

Proof. See Appendix F.

证明。见附录F。

Example. Returning to the query $\mathcal{Q}$ in Figure 1a,consider again $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ ,and the configuration $\mathbf{\eta }$ of $\mathcal{H}$ with $\mathbf{\eta }\left( \mathrm{D}\right)  = \mathrm{d},\mathbf{\eta }\left( \mathrm{E}\right)  = \mathrm{e},\mathbf{\eta }\left( \mathrm{F}\right)  = \mathrm{f}$ ,and $\mathbf{\eta }\left( \mathrm{K}\right)  = \mathrm{k}$ . We will illustrate the above definitions by showing how to avoid explicitly computing ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ and ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ (namely, ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ and ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }$ in the description of Section 1.3). We will generate ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{A}},{R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{B}}$ , ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{I}}$ ,and ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{J}}$ such that ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{A}} \bowtie  {R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{B}}$ and ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{I}} \bowtie$ ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \eta }^{\# \mathrm{J}}$ .

示例。回到图1a中的查询$\mathcal{Q}$，再次考虑$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$，以及$\mathcal{H}$的配置$\mathbf{\eta }$，其中$\mathbf{\eta }\left( \mathrm{D}\right)  = \mathrm{d},\mathbf{\eta }\left( \mathrm{E}\right)  = \mathrm{e},\mathbf{\eta }\left( \mathrm{F}\right)  = \mathrm{f}$且$\mathbf{\eta }\left( \mathrm{K}\right)  = \mathrm{k}$。我们将通过展示如何避免显式计算${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime }$和${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\prime \prime }$（即第1.3节描述中的${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$和${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }$）来说明上述定义。我们将生成${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{A}},{R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{B}}$、${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{I}}$和${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{J}}$，使得${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{A}} \bowtie  {R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \mathbf{\eta }}^{\# \mathrm{B}}$且${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{I}} \bowtie$${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \eta }^{\# \mathrm{J}}$。

Among the four relations to compute, ${R}_{\{ \mathbf{I},\mathbf{J}\}  \mid  \mathbf{\eta }}^{\# \mathbf{J}}$ is the simplest because $\mathbf{J}$ is not a border attribute; hence, ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{J}}$ equals ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \varnothing }^{\prime }$ (i.e., ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime }$ in Section 1.3). Regarding the other three relations,we will elaborate only on the generation of ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\# \mathrm{A}}$ because the same ideas apply to ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\# \mathrm{B}}$ and ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\# \mathrm{I}}$ .

在要计算的四个关系中，${R}_{\{ \mathbf{I},\mathbf{J}\}  \mid  \mathbf{\eta }}^{\# \mathbf{J}}$ 是最简单的，因为 $\mathbf{J}$ 不是边界属性；因此，${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \mathbf{\eta }}^{\# \mathrm{J}}$ 等于 ${R}_{\{ \mathrm{I},\mathrm{J}\}  \mid  \varnothing }^{\prime }$（即第 1.3 节中的 ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime }$）。对于另外三个关系，我们将仅详细阐述 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\# \mathrm{A}}$ 的生成，因为相同的思路也适用于 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\# \mathrm{B}}$ 和 ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\# \mathrm{I}}$。

Under $\mathbf{\eta }$ ,there are two unary residue relations defined over $\{ \mathrm{A}\}$ ,namely, ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ and ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ . The intersection of those two relations yields the unary relation ${R}_{\{ \mathrm{A}\}  \mid  \eta }^{\prime \prime }$ (i.e., ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ in Section 1.3). Then, ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\# \mathrm{A}}$ consists of every tuple $\mathbf{u}$ in the residue relation ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \varnothing }^{\prime }$ (i.e., ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ in Section 1.3) whose $\mathbf{u}\left( A\right)$ appears in ${R}_{\{ \mathrm{A}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ .

在 $\mathbf{\eta }$ 条件下，在 $\{ \mathrm{A}\}$ 上定义了两个一元残差关系，即 ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ 和 ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$。这两个关系的交集产生一元关系 ${R}_{\{ \mathrm{A}\}  \mid  \eta }^{\prime \prime }$（即第 1.3 节中的 ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$）。然后，${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\# \mathrm{A}}$ 由残差关系 ${R}_{\{ \mathrm{A},\mathrm{B}\}  \mid  \varnothing }^{\prime }$（即第 1.3 节中的 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$）中每个元组 $\mathbf{u}$ 组成，其中 $\mathbf{u}\left( A\right)$ 出现在 ${R}_{\{ \mathrm{A}\}  \mid  \mathbf{\eta }}^{\prime \prime }$ 中。

Define:

定义：

$$
{\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\# } = \left\{  {{R}_{e \mid  \mathbf{\eta }}^{\# X} \mid  \text{ every light edge }e \in  \mathcal{E},\text{ every attribute }X \in  e}\right\}  
$$

$$
{\mathcal{Q}}_{\mathbf{\eta }}^{\# } = {\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\# } \cup  {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }.
$$

Proposition 15 immediately implies $\operatorname{Join}\left( {Q}_{\mathbf{\eta }}^{\prime \prime }\right)  = \operatorname{Join}\left( {Q}_{\mathbf{\eta }}^{\# }\right)  = {CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{\operatorname{light} \mid  \mathbf{\eta }}^{\# }\right)$ .

命题15直接蕴含$\operatorname{Join}\left( {Q}_{\mathbf{\eta }}^{\prime \prime }\right)  = \operatorname{Join}\left( {Q}_{\mathbf{\eta }}^{\# }\right)  = {CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{\operatorname{light} \mid  \mathbf{\eta }}^{\# }\right)$。

### 6.2 The Algorithm

### 6.2 算法

We are now ready to clarify how to solve $\mathcal{Q}$ in 3 rounds,concentrating on a specific subset $\mathcal{H}$ of attset(Q)(set $k =  \mid$ attset $\left( \mathcal{Q}\right)  \mid  )$ :

现在我们准备阐明如何在三轮内求解$\mathcal{Q}$，重点关注attset(Q)的一个特定子集$\mathcal{H}$（集合$k =  \mid$ attset $\left( \mathcal{Q}\right)  \mid  )$：

- Round 1: Generate the input relations of ${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$ for every configuration $\mathbf{\eta }$ of $\mathcal{H}$ .

- 第一轮：为$\mathcal{H}$的每个配置$\mathbf{\eta }$生成${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$的输入关系。

- Round 2: Same as the statistical round in Section 5.

- 第二轮：与第5节中的统计轮相同。

- Round 3: Evaluate ${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$ for every $\mathbf{\eta }$ .

- 第三轮：为每个$\mathbf{\eta }$计算${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$。

It remains to elaborate on the details of Round 1 and 3 .

接下来需要详细阐述第一轮和第三轮的细节。

Round 1. Allocate ${p}_{\mathbf{\eta }}^{\prime } = \left\lceil  {p \cdot  \frac{{m}_{\mathbf{\eta }}}{\Theta \left( {m \cdot  {\lambda }^{k - 2}}\right) }}\right\rceil$ machines to computing the relations of ${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$ . Let us focus on a specific configuration $\mathbf{\eta }$ . To generate the relations in ${\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\# }$ ,we carry out one-attribute reduction (see Section 2.4) for every border attribute $X \in  \mathcal{L}$ . Specifically,this operation is performed on

第一轮。分配${p}_{\mathbf{\eta }}^{\prime } = \left\lceil  {p \cdot  \frac{{m}_{\mathbf{\eta }}}{\Theta \left( {m \cdot  {\lambda }^{k - 2}}\right) }}\right\rceil$台机器来计算${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$的关系。让我们关注一个特定的配置$\mathbf{\eta }$。为了生成${\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\# }$中的关系，我们对每个边界属性$X \in  \mathcal{L}$进行单属性约简（见第2.4节）。具体来说，此操作在以下对象上执行

- the unary relations ${R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }$ for all cross edges $e$ of $\mathcal{G}$ incident to $X$ ,and

- 对于$\mathcal{G}$中所有与$X$相关联的交叉边$e$的一元关系${R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }$，以及

- the binary relations ${R}_{e \mid  \varnothing }^{\prime }$ for all light edges $e$ of $\mathcal{G}$ incident to $X$ .

- 对于$\mathcal{G}$中所有与$X$相关联的轻边$e$的二元关系${R}_{e \mid  \varnothing }^{\prime }$。

It generates the ${R}_{e \mid  \mathbf{\eta }}^{\# X}$ for every light edge $e$ incident to $X$ . Note that a value $x \in  \mathbf{{dom}}$ is a heavy-hitter for this operation only if $x$ appears in some input relation of $\mathcal{Q}{m}_{\eta }/{p}_{\eta }^{\prime } =$ $\Omega \left( \frac{m \cdot  {\lambda }^{k - 2}}{p}\right)  = \Omega \left( {m/{p}^{1/\rho }}\right)$ times. Therefore,every machine can independently figure out the heavy-hitters from its histogram, and send each tuple in its local storage directly to the corresponding machines where the tuple is needed to perform one-attribute reductions. By Lemma 6,all the one-attribute reductions entail an overall load of $\widetilde{O}\left( {p + {m}_{\mathbf{\eta }}/{p}_{\mathbf{n}}^{\prime }}\right)  =$ $\widetilde{O}\left( {p + m/{p}^{1/\rho }}\right)$ .

它为与$X$相关联的每条轻边$e$生成${R}_{e \mid  \mathbf{\eta }}^{\# X}$。请注意，只有当$x$在$\mathcal{Q}{m}_{\eta }/{p}_{\eta }^{\prime } =$的某些输入关系中出现$\Omega \left( \frac{m \cdot  {\lambda }^{k - 2}}{p}\right)  = \Omega \left( {m/{p}^{1/\rho }}\right)$次时，值$x \in  \mathbf{{dom}}$才是此操作的频繁项。因此，每台机器都可以从其直方图中独立找出频繁项，并将其本地存储中的每个元组直接发送到需要该元组来执行单属性约简的相应机器。根据引理6，所有单属性约简的总负载为$\widetilde{O}\left( {p + {m}_{\mathbf{\eta }}/{p}_{\mathbf{n}}^{\prime }}\right)  =$ $\widetilde{O}\left( {p + m/{p}^{1/\rho }}\right)$。

The relations in ${\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }$ can be easily produced by set intersection. Specifically,for every isolated attribute $X \in  \mathcal{I}$ ,we obtain ${R}_{\{ X\}  \mid  \eta }^{\prime \prime }$ as the intersection of all the unary relations ${R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }$ ,where $e$ ranges over all cross edges of $\mathcal{G}$ incident to $X$ . This can be done by standard hashing in one round with load $\widetilde{O}\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ .

${\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }$中的关系可以通过集合交集轻松得出。具体而言，对于每个孤立属性$X \in  \mathcal{I}$，我们将${R}_{\{ X\}  \mid  \eta }^{\prime \prime }$作为所有一元关系${R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }^{\prime }$的交集，其中$e$遍历$\mathcal{G}$中与$X$相关的所有交叉边。这可以通过标准哈希在一轮内完成，负载为$\widetilde{O}\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$。

Round 3. Allocating ${p}_{\mathbf{\eta }}^{\prime \prime }$ ,as is given in (6),machines to each configuration $\mathbf{\eta }$ of $\mathcal{H}$ ,we compute ${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$ in exactly the same way Lemma 14 computes ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ . In fact,the statement of Lemma 14,as well as the proof,holds verbatim by replacing every ${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$ with ${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$ and every ${\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\prime \prime }$ with ${\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\# }$ .

第三轮。按照(6)中给出的方式，为$\mathcal{H}$的每个配置$\mathbf{\eta }$分配${p}_{\mathbf{\eta }}^{\prime \prime }$台机器，我们计算${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$的方式与引理14计算${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$的方式完全相同。事实上，引理14的陈述及其证明，只需将每个${\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }$替换为${\mathcal{Q}}_{\mathbf{\eta }}^{\# }$，每个${\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\prime \prime }$替换为${\mathcal{Q}}_{\text{light } \mid  \mathbf{\eta }}^{\# }$，即可逐字适用。

We thus have obtained a 3-round algorithm for answering a simple join query with load $\widetilde{O}\left( {{p}^{2} + m/{p}^{1/\rho }}\right)$ which is $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ under our assumption $m \geq  {p}^{3}$ . This establishes the second main result of this paper:

因此，我们得到了一个用于回答简单连接查询的三轮算法，其负载为$\widetilde{O}\left( {{p}^{2} + m/{p}^{1/\rho }}\right)$，在我们的假设$m \geq  {p}^{3}$下为$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$。这确立了本文的第二个主要结果：

Theorem 16. Given a simple join query with input size $m$ and a fractional edge covering number $\rho$ ,we can answer it in the MPC model using $p$ machines in two rounds with load $O\left( {m/{p}^{1/\rho }}\right)$ ,assuming that $m \geq  {p}^{3}$ ,and that each machine has been preloaded with a histogram as is prescribed in Section 5.

定理16。给定一个输入大小为$m$且分数边覆盖数为$\rho$的简单连接查询，假设$m \geq  {p}^{3}$成立，并且每台机器已按照第5节的规定预加载了直方图，我们可以在MPC模型中使用$p$台机器在两轮内以负载$O\left( {m/{p}^{1/\rho }}\right)$回答该查询。

It is worth mentioning that Round 2 of our algorithm (i.e., the statistical round) has a load of $O\left( {p}^{2}\right)$ such that only the first and third rounds of the algorithm entail a load sensitive to $m$ .

值得一提的是，我们算法的第二轮（即统计轮）的负载为$O\left( {p}^{2}\right)$，因此只有算法的第一轮和第三轮的负载对$m$敏感。

## — References

## — 参考文献

1 Serge Abiteboul, Richard Hull, and Victor Vianu. Foundations of Databases. Addison-Wesley, 1995.

2 Azza Abouzeid, Kamil Bajda-Pawlikowski, Daniel J. Abadi, Alexander Rasin, and Avi Sil-berschatz. HadoopDB: An Architectural Hybrid of MapReduce and DBMS Technologies for Analytical Workloads. Proceedings of the VLDB Endowment (PVLDB), 2(1):922-933, 2009.

3 Foto N. Afrati and Jeffrey D. Ullman. Optimizing Multiway Joins in a Map-Reduce Environment. IEEE Transactions on Knowledge and Data Engineering (TKDE), 23(9):1282-1298, 2011.

4 Albert Atserias, Martin Grohe, and Daniel Marx. Size Bounds and Query Plans for Relational Joins. SIAM J. Comput., 42(4):1737-1767, 2013.

5 Paul Beame, Paraschos Koutris, and Dan Suciu. Communication Steps for Parallel Query Processing. Journal of the ACM (JACM), 64(6):40:1-40:58, 2017.

6 Jeffrey Dean and Sanjay Ghemawat. MapReduce: Simplified Data Processing on Large Clusters. In Proceedings of USENIX Symposium on Operating Systems Design and Implementation (OSDI), pages 137-150, 2004.

7 Xiao Hu, Paraschos Koutris, and Ke Yi. An External-Memory Work-Depth Model and Its Applications to Massively Parallel Join Algorithms. Manuscript, 2018.

8 Xiaocheng Hu, Miao Qiao, and Yufei Tao. I/O-efficient join dependency testing, loomis-whitney join, and triangle enumeration. Journal of Computer and System Sciences (JCSS), 82(8):1300-1315, 2016.

9 Bas Ketsman and Dan Suciu. A Worst-Case Optimal Multi-Round Algorithm for Parallel Computation of Conjunctive Queries. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 417-428, 2017.

10 Paraschos Koutris, Paul Beame, and Dan Suciu. Worst-Case Optimal Algorithms for Parallel Query Processing. In Proceedings of International Conference on Database Theory (ICDT), pages $8 : 1 - 8 : {18},{2016}$ .

11 Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. Worst-case Optimal Join Algorithms. Journal of the ${ACM}\left( {JACM}\right) ,{65}\left( 3\right)  : {16} : 1 - {16} : {40},{2018}$ .

12 Rasmus Pagh and Francesco Silvestri. The input/output complexity of triangle enumeration. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 224-233, 2014.

13 Edward R. Scheinerman and Daniel H. Ullman. Fractional Graph Theory: A Rational Approach to the Theory of Graphs. Wiley, New York, 1997.

14 Todd L. Veldhuizen. Triejoin: A Simple, Worst-Case Optimal Join Algorithm. In Proceedings of International Conference on Database Theory (ICDT), pages 96-106, 2014.

15 Mihalis Yannakakis. Algorithms for Acyclic Database Schemes. In Very Large Data Bases, 7th International Conference, September 9-11, 1981, Cannes, France, Proceedings, pages 82-94, 1981.

## A Proof of Lemma 6

## 引理6的证明

For each $i \in  \left\lbrack  {1,b}\right\rbrack$ ,divide ${S}_{i}$ into (i) ${S}_{i}^{1}$ ,which includes the tuples of $\mathbf{u} \in  {S}_{i}$ where $\mathbf{u}\left( X\right)$ is a heavy-hitter,and (ii) ${S}_{i}^{2} = {S}_{i} \smallsetminus  {S}_{i}^{1}$ . Accordingly,divide ${S}_{i}^{\# }$ into (i) ${S}_{i}^{\# 1}$ ,which includes the tuples of $\mathbf{u} \in  {S}_{i}^{\# }$ where $\mathbf{u}\left( X\right)$ is a heavy-hitter,and (ii) ${S}_{i}^{\# 2} = {S}_{i}^{\# } \smallsetminus  {S}_{i}^{\# 1}$ . We will compute ${S}_{i}^{\# 1}$ and ${S}_{i}^{\# 2}$ ,separately.

对于每个$i \in  \left\lbrack  {1,b}\right\rbrack$，将${S}_{i}$划分为：(i) ${S}_{i}^{1}$，它包含$\mathbf{u} \in  {S}_{i}$中$\mathbf{u}\left( X\right)$为频繁项（heavy - hitter）的元组；(ii) ${S}_{i}^{2} = {S}_{i} \smallsetminus  {S}_{i}^{1}$。相应地，将${S}_{i}^{\# }$划分为：(i) ${S}_{i}^{\# 1}$，它包含$\mathbf{u} \in  {S}_{i}^{\# }$中$\mathbf{u}\left( X\right)$为频繁项的元组；(ii) ${S}_{i}^{\# 2} = {S}_{i}^{\# } \smallsetminus  {S}_{i}^{\# 1}$。我们将分别计算${S}_{i}^{\# 1}$和${S}_{i}^{\# 2}$。

The computation of ${S}_{1}^{\# 1},\ldots ,{S}_{b}^{\# 1}$ is trivial. Since there are at most $p$ heavy-hitters,each machine storing a heavy-hitter $x$ in some ${R}_{j}\left( {j \in  \left\lbrack  {1,a}\right\rbrack  }\right)$ simply broadcasts the pair(x,j) to all machines. This takes one round with load $O\left( p\right)$ . A machine holding a tuple $\mathbf{u}$ with $\mathbf{u}\left( X\right)  = x$ in some ${S}_{i}\left( {i \in  \left\lbrack  {1,a}\right\rbrack  }\right)$ adds $\mathbf{u}$ to ${S}_{i}^{\# 1}$ only if it has received(x,j)for all $j \in  \left\lbrack  {1,a}\right\rbrack$ .

${S}_{1}^{\# 1},\ldots ,{S}_{b}^{\# 1}$的计算很简单。由于最多有$p$个频繁项，每台在某个${R}_{j}\left( {j \in  \left\lbrack  {1,a}\right\rbrack  }\right)$中存储频繁项$x$的机器只需将对(x, j)广播给所有机器。这需要一轮，负载为$O\left( p\right)$。某台在某个${S}_{i}\left( {i \in  \left\lbrack  {1,a}\right\rbrack  }\right)$中持有元组$\mathbf{u}$且$\mathbf{u}\left( X\right)  = x$的机器，只有在它接收到所有$j \in  \left\lbrack  {1,a}\right\rbrack$的(x, j)时，才将$\mathbf{u}$添加到${S}_{i}^{\# 1}$中。

${S}_{1}^{\# 2},\ldots ,{S}_{b}^{\# 2}$ ,on the other hand,can be produced using Lemma 5 . Assign a share of $p$ to $X$ and a share of 1 to every other attribute. By definition,the join query $\left\{  {{R}_{1},\ldots ,{R}_{a},{S}_{1}^{2},\ldots ,{S}_{b}^{2}}\right\}$ is skew-free,and therefore,can be solved in round round with load $\widetilde{O}\left( {n/p}\right)$ . For each $i \in  \left\lbrack  {1,b}\right\rbrack$ , ${S}_{i}^{\# 2}$ can then be easily obtained from the result of this query.

另一方面，可以使用引理5来计算${S}_{1}^{\# 2},\ldots ,{S}_{b}^{\# 2}$。将$p$的一部分分配给$X$，将1的一部分分配给其他每个属性。根据定义，连接查询$\left\{  {{R}_{1},\ldots ,{R}_{a},{S}_{1}^{2},\ldots ,{S}_{b}^{2}}\right\}$是无倾斜（skew - free）的，因此可以在一轮内以负载$\widetilde{O}\left( {n/p}\right)$解决。对于每个$i \in  \left\lbrack  {1,b}\right\rbrack$，然后可以从该查询的结果中轻松获得${S}_{i}^{\# 2}$。

## B Proof of Proposition 7

## B 命题7的证明

Let us first introduce a definition. Suppose that $\mathcal{S}$ is a subset of $\mathcal{H}$ . We say that a configuration ${\mathbf{\eta }}_{\mathcal{H}}$ of $\mathcal{H}$ extends a configuration ${\mathbf{\eta }}_{\mathcal{S}}$ of $\mathcal{S}$ if ${\mathbf{\eta }}_{\mathcal{S}} = {\mathbf{\eta }}_{\mathcal{H}}\left\lbrack  \mathcal{S}\right\rbrack  .O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| \mathcal{S}\right| }\right)$ configurations ${\mathbf{\eta }}_{\mathcal{H}}$ of $\mathcal{H}$ can extend ${\mathbf{\eta }}_{\mathcal{S}}$ ,because every attribute in $\mathcal{H} \smallsetminus  \mathcal{S}$ has $O\left( \lambda \right)$ heavy values.

让我们首先引入一个定义。假设$\mathcal{S}$是$\mathcal{H}$的一个子集。我们说$\mathcal{H}$的一个配置${\mathbf{\eta }}_{\mathcal{H}}$扩展了$\mathcal{S}$的一个配置${\mathbf{\eta }}_{\mathcal{S}}$，如果${\mathbf{\eta }}_{\mathcal{S}} = {\mathbf{\eta }}_{\mathcal{H}}\left\lbrack  \mathcal{S}\right\rbrack  .O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| \mathcal{S}\right| }\right)$。$\mathcal{H}$的配置${\mathbf{\eta }}_{\mathcal{H}}$可以扩展${\mathbf{\eta }}_{\mathcal{S}}$，因为$\mathcal{H} \smallsetminus  \mathcal{S}$中的每个属性都有$O\left( \lambda \right)$个频繁值。

Returning to the proof of the proposition,Let $\mathbf{\eta }$ be an arbitrary configuration of $\mathcal{H}$ ,and $e \in  \mathcal{E}$ an arbitrary hyperedge that is active on $\mathbf{\eta }$ . Define ${\mathcal{H}}^{\prime } = e \cap  \mathcal{H}$ and ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ . A tuple $\mathbf{u} \in  {R}_{e}$ belongs to ${R}_{e \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }$ only if $\mathbf{\eta }$ extends the configuration $\mathbf{u}\left\lbrack  {\mathcal{H}}^{\prime }\right\rbrack$ of ${\mathcal{H}}^{\prime }$ . There are $O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right| }\right)$ such $\mathbf{\eta }$ . As $\left| \mathcal{E}\right|  = O\left( 1\right) ,\mathbf{u}$ can contribute $O\left( 1\right)$ to the term ${m}_{\mathbf{\eta }}$ for at most $O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right| }\right)$ different $\mathbf{\eta }$ .

回到该命题的证明，设$\mathbf{\eta }$为$\mathcal{H}$的任意一个配置，$e \in  \mathcal{E}$为在$\mathbf{\eta }$上活跃的任意一条超边。定义${\mathcal{H}}^{\prime } = e \cap  \mathcal{H}$和${e}^{\prime } = e \smallsetminus  \mathcal{H}$。元组$\mathbf{u} \in  {R}_{e}$属于${R}_{e \mid  \mathbf{\eta }\left\lbrack  {e \smallsetminus  {e}^{\prime }}\right\rbrack  }$当且仅当$\mathbf{\eta }$扩展了${\mathcal{H}}^{\prime }$的配置$\mathbf{u}\left\lbrack  {\mathcal{H}}^{\prime }\right\rbrack$。这样的$\mathbf{\eta }$有$O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right| }\right)$个。因为$\left| \mathcal{E}\right|  = O\left( 1\right) ,\mathbf{u}$最多可以对$O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right| }\right)$个不同的$\mathbf{\eta }$的项${m}_{\mathbf{\eta }}$贡献$O\left( 1\right)$。

It remains to prove that $\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right|  \leq  k - 2$ . Observe that $\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right|$ is the number of attributes in $\mathcal{H}$ that do not belong to $e$ . This number is at most $k - 2$ because $e$ has two attributes.

还需证明$\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right|  \leq  k - 2$。注意到$\left| \mathcal{H}\right|  - \left| {\mathcal{H}}^{\prime }\right|$是$\mathcal{H}$中不属于$e$的属性的数量。这个数量最多为$k - 2$，因为$e$有两个属性。

## C Proof of Proposition 8

## C 命题8的证明

The second equality follows directly from the fact that the scheme of each relation in ${\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{n}}^{\prime \prime }$ is disjoint with the scheme of any other relation in ${\mathcal{Q}}_{\eta }^{\prime \prime }$ . Next we focus on proving the first equality.

第二个等式直接由以下事实得出：${\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{n}}^{\prime \prime }$中每个关系的模式与${\mathcal{Q}}_{\eta }^{\prime \prime }$中任何其他关系的模式不相交。接下来我们专注于证明第一个等式。

We first show $\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)  \subseteq  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ . Consider an arbitrary tuple $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)$ . For any attribute $X \in  \mathcal{L}$ and any cross edge $e$ of $\mathcal{G}$ containing $X$ ,since $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }$ , it must hold that $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \eta }^{\prime \prime }$ . For any light edge $e = \{ X,Y\}  \in  \mathcal{E}$ ,since $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \varnothing }^{\prime }$ ,it must hold that $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$ . It thus follows that $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ .

我们首先证明$\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)  \subseteq  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$。考虑任意元组$\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)$。对于任意属性$X \in  \mathcal{L}$以及包含$X$的$\mathcal{G}$的任意交叉边$e$，由于$\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }$，必然有$\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \eta }^{\prime \prime }$。对于任意轻边$e = \{ X,Y\}  \in  \mathcal{E}$，由于$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \varnothing }^{\prime }$，必然有$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$。因此可得$\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$。

Next,we show $\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)  \subseteq  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)$ . Consider an arbitrary tuple $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$ . For any attribute $X \in  \mathcal{L}$ ,since $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime }$ ,it must hold that $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }$ for any cross edge $e$ of $\mathcal{G}$ containing $X$ . For any light edge $e = \{ X,Y\}  \in  \mathcal{E}$ ,since $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$ ,it must hold that $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \varnothing }^{\prime }$ . It thus follows that $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)$ .

接下来，我们证明$\operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)  \subseteq  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)$。考虑任意元组$\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime \prime }\right)$。对于任意属性$X \in  \mathcal{L}$，由于$\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime }$，对于包含$X$的$\mathcal{G}$的任意交叉边$e$，必然有$\mathbf{u}\left( X\right)  \in  {R}_{\{ X\}  \mid  \mathbf{\eta }\left\lbrack  {e\smallsetminus \{ X\} }\right\rbrack  }$。对于任意轻边$e = \{ X,Y\}  \in  \mathcal{E}$，由于$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \mathbf{\eta }}^{\prime \prime }$，必然有$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e \mid  \varnothing }^{\prime }$。因此可得$\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{\mathbf{\eta }}^{\prime }\right)$。

## D Proof of Lemma 10

## D 引理10的证明

Consider any ${R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime } \in  {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }$ . Observe that the content of ${R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime }$ does not depend on $\mathbf{\eta }\left( Y\right)$ for any $Y \in  \mathcal{H} \smallsetminus  \mathcal{F}$ . In other words,if we set ${\mathbf{\eta }}^{\prime } = \mathbf{\eta }\left\lbrack  \mathcal{F}\right\rbrack$ ,then ${R}_{\{ X\}  \mid  \mathbf{\eta }}^{n}$ is precisely the same as ${R}_{\{ X\}  \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }$ . Notice that ${\mathbf{\eta }}^{\prime }$ is a configuration of $\mathcal{F}$ that is extended by $\mathbf{\eta }$ (see the proof of Proposition 7 for the definition of extension). The lemma follows from the fact that a configuration ${\mathbf{\eta }}^{\prime }$ of $\mathcal{F}$ can be extended by $O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| \mathcal{F}\right| }\right)$ configurations $\mathbf{\eta }$ of $\mathcal{H}$ .

考虑任意的 ${R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime } \in  {\mathcal{Q}}_{\mathcal{I} \mid  \mathbf{\eta }}^{\prime \prime }$ 。观察可知，对于任意的 $Y \in  \mathcal{H} \smallsetminus  \mathcal{F}$ ，${R}_{\{ X\}  \mid  \mathbf{\eta }}^{\prime \prime }$ 的内容不依赖于 $\mathbf{\eta }\left( Y\right)$ 。换句话说，如果我们设 ${\mathbf{\eta }}^{\prime } = \mathbf{\eta }\left\lbrack  \mathcal{F}\right\rbrack$ ，那么 ${R}_{\{ X\}  \mid  \mathbf{\eta }}^{n}$ 与 ${R}_{\{ X\}  \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }$ 完全相同。注意，${\mathbf{\eta }}^{\prime }$ 是 $\mathcal{F}$ 的一个配置，它由 $\mathbf{\eta }$ 扩展而来（关于扩展的定义，请参见命题 7 的证明）。该引理源于这样一个事实：$\mathcal{F}$ 的一个配置 ${\mathbf{\eta }}^{\prime }$ 可以由 $\mathcal{H}$ 的 $O\left( {\lambda }^{\left| \mathcal{H}\right|  - \left| \mathcal{F}\right| }\right)$ 个配置 $\mathbf{\eta }$ 扩展得到。

## E Proof of Lemma 12

## E 引理 12 的证明

We will prove

我们将证明

$$
\mathop{\bigcup }\limits_{{\text{config. }{\mathbf{\eta }}^{\prime }\text{ of }\mathcal{F}}}\operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }\right)  \times  \left\{  {\mathbf{\eta }}^{\prime }\right\}   \subseteq  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) . \tag{7}
$$

from which the lemma follows.

由此可推出该引理。

Take a tuple $\mathbf{u}$ from the left hand side of (7),and set ${\mathbf{\eta }}^{\prime } = \mathbf{u}\left\lbrack  \mathcal{F}\right\rbrack$ . Based on the definition of ${\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }$ ,it is easy to verify that $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}$ for every cross edge $e \in  \mathcal{E}$ ,and hence, $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}^{ * }$ . Furthermore, $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\} }^{ * }$ for every $X \in  \mathcal{F}$ because $\mathbf{u}\left( X\right)  = {\mathbf{\eta }}^{\prime }\left( X\right)$ is a heavy value. Finally, obviously $\mathbf{u}\left( Y\right)  \in  {R}_{\{ Y\} }^{ * }$ for every $Y \in  {\mathcal{I}}_{0}$ . All these facts together ensure that $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ .

从 (7) 式的左边取一个元组 $\mathbf{u}$ ，并设 ${\mathbf{\eta }}^{\prime } = \mathbf{u}\left\lbrack  \mathcal{F}\right\rbrack$ 。根据 ${\mathcal{Q}}_{\mathcal{I} \mid  {\mathbf{\eta }}^{\prime }}^{\prime \prime }$ 的定义，很容易验证对于每一条交叉边 $e \in  \mathcal{E}$ ，都有 $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}$ ，因此，$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}^{ * }$ 。此外，对于每一个 $X \in  \mathcal{F}$ ，都有 $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\} }^{ * }$ ，因为 $\mathbf{u}\left( X\right)  = {\mathbf{\eta }}^{\prime }\left( X\right)$ 是一个重值。最后，显然对于每一个 $Y \in  {\mathcal{I}}_{0}$ ，都有 $\mathbf{u}\left( Y\right)  \in  {R}_{\{ Y\} }^{ * }$ 。所有这些事实共同保证了 $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ 。

## F Proof of Proposition 15

## F 命题 15 的证明

Consider first the case where $X$ and $Y$ are both border attributes. We have

首先考虑 $X$ 和 $Y$ 都是边界属性的情况。我们有

$$
{R}_{e \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{X \mid  \mathbf{\eta }}^{\prime \prime } \bowtie  {R}_{e \mid  \varnothing }^{\prime } \bowtie  {R}_{Y \mid  \mathbf{\eta }}^{\prime \prime }
$$

$$
 = \left( {{R}_{X \mid  \mathbf{\eta }}^{\prime \prime } \bowtie  {R}_{e \mid  \varnothing }^{\prime }}\right)  \bowtie  \left( {{R}_{e \mid  \varnothing }^{\prime } \bowtie  {R}_{Y \mid  \mathbf{\eta }}^{\prime \prime }}\right) 
$$

$$
 = {R}_{e \mid  \mathbf{\eta }}^{\# X} \bowtie  {R}_{e \mid  \mathbf{\eta }}^{\# Y}.
$$

If $X$ is a border attribute but $Y$ is not,then:

如果 $X$ 是边界属性，但 $Y$ 不是，那么：

$$
{R}_{e \mid  \mathbf{\eta }}^{\prime \prime } = {R}_{X \mid  \mathbf{\eta }}^{\prime \prime } \bowtie  {R}_{e \mid  \varnothing }^{\prime }
$$

$$
 = \left( {{R}_{X \mid  \mathbf{\eta }}^{\prime \prime } \boxtimes  {R}_{e \mid  \varnothing }^{\prime }}\right)  \boxtimes  {R}_{e \mid  \varnothing }^{\prime }
$$

$$
 = {R}_{e \mid  \mathbf{\eta }}^{\# X} \bowtie  {R}_{e \mid  \mathbf{\eta }}^{\# Y}.
$$

If neither $X$ nor $Y$ is a border attribute,the proposition is trivial.

如果 $X$ 和 $Y$ 都不是边界属性，那么该命题是显然的。