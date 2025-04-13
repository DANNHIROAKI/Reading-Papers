# A NEAR-OPTIMAL PARALLEL ALGORITHM FOR JOINING BINARY RELATIONS

# 一种用于连接二元关系的近最优并行算法

BAS KETSMAN, DAN SUCIU, AND YUFEI TAO

巴斯·凯茨曼（Bas Ketsman）、丹·苏丘（Dan Suciu）和陶宇飞（Yufei Tao）

Vrije Universiteit Brussel

布鲁塞尔自由大学（Vrije Universiteit Brussel）

e-mail address: bas.ketsman@vub.be

电子邮件地址：bas.ketsman@vub.be

University of Washington

华盛顿大学（University of Washington）

e-mail address: suciu@cs.washington.edu

电子邮件地址：suciu@cs.washington.edu

Chinese University of Hong Kong, Hong Kong

中国香港中文大学（Chinese University of Hong Kong, Hong Kong）

e-mail address: taoyf@cse.cuhk.edu.hk

电子邮件地址：taoyf@cse.cuhk.edu.hk

ABSTRACT. We present a constant-round algorithm in the massively parallel computation (MPC) model for evaluating a natural join where every input relation has two attributes. Our algorithm achieves a load of $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ where $m$ is the total size of the input relations, $p$ is the number of machines, $\rho$ is the join’s fractional edge covering number,and $O\left( \text{.}\right) h$ ides a polylogarithmic factor. The load matches a known lower bound up to a polylogarithmic factor. At the core of the proposed algorithm is a new theorem (which we name the isolated cartesian product theorem) that provides fresh insight into the problem's mathematical structure. Our result implies that the subgraph enumeration problem, where the goal is to report all the occurrences of a constant-sized subgraph pattern, can be settled optimally (up to a polylogarithmic factor) in the MPC model.

摘要。我们提出了一种在大规模并行计算（MPC）模型下的常数轮算法，用于评估每个输入关系都有两个属性的自然连接。我们的算法实现了$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$的负载，其中$m$是输入关系的总大小，$p$是机器的数量，$\rho$是连接的分数边覆盖数，$O\left( \text{.}\right) h$表示一个多对数因子。该负载在多对数因子范围内与已知的下界相匹配。所提出算法的核心是一个新定理（我们将其命名为孤立笛卡尔积定理），它为该问题的数学结构提供了全新的见解。我们的结果表明，子图枚举问题（目标是报告固定大小子图模式的所有出现情况）可以在MPC模型中以最优方式（在多对数因子范围内）解决。

## 1. INTRODUCTION

## 1. 引言

Understanding the hardness of joins has been a central topic in database theory. Traditional efforts have focused on discovering fast algorithms for processing joins in the random access machine (RAM) model (see $\left\lbrack  {1,5,{16} - {18},{21},{22}}\right\rbrack$ and the references therein). Nowadays, massively parallel systems such as Hadoop [8] and Spark [2] have become the mainstream architecture for analytical tasks on gigantic volumes of data. Direct adaptations of RAM algorithms, which are designed to reduce CPU time, rarely give satisfactory performance on that architecture. In systems like Hadoop and Spark, it is crucial to minimize communication across the participating machines because usually the overhead of message exchanging overwhelms the CPU calculation cost. This has motivated a line of research - which includes this work - that aims to understand the communication complexities of join problems.

理解连接操作的难度一直是数据库理论中的核心话题。传统的研究主要集中在随机访问机器（RAM）模型中寻找处理连接的快速算法（见$\left\lbrack  {1,5,{16} - {18},{21},{22}}\right\rbrack$及其参考文献）。如今，像Hadoop [8]和Spark [2]这样的大规模并行系统已成为处理海量数据分析任务的主流架构。为减少CPU时间而设计的RAM算法直接应用在这种架构上时，很少能取得令人满意的性能。在Hadoop和Spark等系统中，最小化参与机器之间的通信至关重要，因为通常消息交换的开销会超过CPU计算成本。这推动了一系列旨在理解连接问题通信复杂性的研究，本文的工作也是其中之一。

---

<!-- Footnote -->

Key words and phrases: Joins, Conjunctive Queries, Parallel Computing, Database Theory.

关键词和短语：连接、合取查询、并行计算、数据库理论。

The research of Bas Ketsman was partially supported by FWO-grant G062721N. The research of Dan Suciu was partially supported by projects NSF IIS 1907997 and NSF-BSF 2109922. The research of Yufei Tao was partially supported by GRF projects 142034/21 and 142078/20 from HKRGC, and an AIR project from the Alibaba group.

巴斯·凯茨曼（Bas Ketsman）的研究部分得到了弗拉芒研究基金会（FWO）资助项目G062721N的支持。丹·苏丘（Dan Suciu）的研究部分得到了美国国家科学基金会（NSF）项目IIS 1907997和NSF - 以色列科学基金会（BSF）项目2109922的支持。陶宇飞（Yufei Tao）的研究部分得到了香港研究资助局（HKRGC）的研资局研究基金（GRF）项目142034/21和142078/20以及阿里巴巴集团的一个AIR项目的支持。

<!-- Footnote -->

---

1.1. Problem Definition. We will first give a formal definition of the join operation studied in this paper and then elaborate on the computation model assumed.

1.1. 问题定义。我们首先给出本文所研究的连接操作的正式定义，然后详细阐述所假设的计算模型。

Joins. Let att be a finite set where each element is called an attribute, and dom be a countably infinite set where each element is called a value. A tuple over a set $U \subseteq$ att is a function $\mathbf{u} : U \rightarrow$ dom. Given a subset $V$ of $U$ ,define $\mathbf{u}\left\lbrack  V\right\rbrack$ as the tuple $\mathbf{v}$ over $V$ such that $\mathbf{v}\left( X\right)  = \mathbf{u}\left( X\right)$ for every $X \in  V$ . We say that $\mathbf{u}\left\lbrack  V\right\rbrack$ is the projection of $\mathbf{u}$ on $V$ .

连接。设att是一个有限集，其中每个元素称为一个属性，dom是一个可数无限集，其中每个元素称为一个值。一个基于集合$U \subseteq$⊆att的元组是一个函数$\mathbf{u} : U \rightarrow$:$U \subseteq$→dom。给定$U$的一个子集$V$，将$\mathbf{u}\left\lbrack  V\right\rbrack$定义为基于$V$的元组$\mathbf{v}$，使得对于每个$X \in  V$∈$V$，都有$\mathbf{v}\left( X\right)  = \mathbf{u}\left( X\right)$。我们称$\mathbf{u}\left\lbrack  V\right\rbrack$是$\mathbf{u}$在$V$上的投影。

A relation is a set $R$ of tuples over the same set $U$ of attributes. We say that the scheme of $R$ is $U$ ,and write this fact as $\operatorname{scheme}\left( R\right)  = U.R$ is unary or binary if $\left| {\operatorname{scheme}\left( R\right) }\right|  = 1$ or 2,respectively. A value $x \in$ dom appears in $R$ if there exist a tuple $\mathbf{u} \in  R$ and an attribute $X \in  U$ such that $\mathbf{u}\left( X\right)  = x$ ; we will also use the expression that $x$ is "a value on the attribute $X$ in $R$ ".

关系是在同一属性集$U$上的元组集合$R$。我们称$R$的模式为$U$，并将这一事实记为$\operatorname{scheme}\left( R\right)  = U.R$。如果$\left| {\operatorname{scheme}\left( R\right) }\right|  = 1$的值分别为1或2，则$R$为一元或二元关系。如果存在元组$\mathbf{u} \in  R$和属性$X \in  U$使得$\mathbf{u}\left( X\right)  = x$，则值$x \in$∈dom出现在$R$中；我们也会使用“$x$是$R$中属性$X$上的值”这样的表述。

A join query (sometimes abbreviated as a "join" or a "query") is a set $\mathcal{Q}$ of relations. Define attset $\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{R \in  \mathcal{O}}}$ scheme(R). The result of the query,denoted as $\operatorname{Join}\left( \mathcal{Q}\right)$ ,is the following relation over attset(Q)

连接查询（有时简称为“连接”或“查询”）是一个关系集合$\mathcal{Q}$。定义attset($\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{R \in  \mathcal{O}}}$) = scheme(R)。查询的结果，记为$\operatorname{Join}\left( \mathcal{Q}\right)$，是attset(Q)上的如下关系

$$
\left\{  {\text{tuple }\mathbf{u}\text{ over attset }\left( \mathcal{Q}\right) \mid \forall R \in  \mathcal{Q},\mathbf{u}\left\lbrack  {\text{ scheme }\left( R\right) }\right\rbrack   \in  R}\right\}  .
$$

Q is

Q是

- simple if no distinct $R,S \in  \mathcal{Q}$ satisfy scheme $\left( R\right)  =$ scheme(S);

- 简单的，如果没有不同的$R,S \in  \mathcal{Q}$满足scheme($\left( R\right)  =$) = scheme(S)；

- binary if every $R \in  \mathcal{Q}$ is binary.

- 二元的，如果每个$R \in  \mathcal{Q}$都是二元关系。

Our objective is to design algorithms for answering simple binary queries.

我们的目标是设计用于回答简单二元查询的算法。

The integer

整数

$$
m = \mathop{\sum }\limits_{{R \in  \mathcal{Q}}}\left| R\right|  \tag{1.1}
$$

is the input size of $\mathcal{Q}$ . Concentrating on data complexity,we will assume that both $\left| \mathcal{Q}\right|$ and $\left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$ are constants.

是$\mathcal{Q}$的输入规模。专注于数据复杂度，我们将假设$\left| \mathcal{Q}\right|$和$\left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$都是常数。

Computation Model. We will assume the massively parallel computation (MPC) model, which is a widely-accepted abstraction of today’s massively parallel systems. Denote by $p$ the number of machines. In the beginning, the input elements are evenly distributed across these machines. For a join query,this means that each machine stores $\Theta \left( {m/p}\right)$ tuples from the input relations (we consider that every value in $\mathbf{{dom}}$ can be encoded in a single word).

计算模型。我们将采用大规模并行计算（MPC）模型，它是当今大规模并行系统被广泛接受的抽象模型。用$p$表示机器的数量。一开始，输入元素均匀分布在这些机器上。对于连接查询，这意味着每台机器存储来自输入关系的$\Theta \left( {m/p}\right)$个元组（我们认为$\mathbf{{dom}}$中的每个值都可以编码为一个字）。

An algorithm is executed in rounds, each having two phases:

算法按轮次执行，每一轮有两个阶段：

- In the first phase, every machine performs computation on the data of its local storage. - In the second phase, the machines communicate by sending messages to each other.

- 在第一阶段，每台机器对其本地存储的数据进行计算。 - 在第二阶段，机器之间通过相互发送消息进行通信。

All the messages sent out in the second phase must be prepared in the first phase. This prevents a machine from, for example, sending information based on the data received during the second phase. Another round is launched only if the current round has not solved the problem. In our context,solving a join query means that,for every tuple $\mathbf{u}$ in the join result, at least one of the machines has $\mathbf{u}$ in the local storage; furthermore,no tuples outside the join result should be produced.

在第二阶段发送的所有消息必须在第一阶段准备好。这防止了机器，例如，根据在第二阶段接收到的数据发送信息。只有当当前轮次未能解决问题时，才会启动下一轮。在我们的上下文中，解决连接查询意味着，对于连接结果中的每个元组$\mathbf{u}$，至少有一台机器在其本地存储中有$\mathbf{u}$；此外，不应产生连接结果之外的元组。

The load of a round is the largest number of words received by a machine in this round, that is,if machine $i \in  \left\lbrack  {1,p}\right\rbrack$ receives ${x}_{i}$ words,the load is $\mathop{\max }\limits_{{i = 1}}^{p}{x}_{i}$ . The performance of an algorithm is measured by two metrics: (i) the number of rounds, and (ii) the load of the algorithm, defined as the total load of all rounds. CPU computation is for free. We will be interested only in algorithms finishing in a constant number of rounds. The load of such an algorithm is asymptotically the same as the maximum load of the individual rounds.

一轮的负载是该轮中一台机器接收到的最大字数，即，如果机器$i \in  \left\lbrack  {1,p}\right\rbrack$接收到${x}_{i}$个字，则负载为$\mathop{\max }\limits_{{i = 1}}^{p}{x}_{i}$。算法的性能由两个指标衡量：（i）轮数，以及（ii）算法的负载，定义为所有轮次的总负载。CPU计算是免费的。我们只关注在常数轮数内完成的算法。这种算法的负载渐近等于各轮的最大负载。

The number $p$ of machines is assumed to be significantly less than $m$ ,which in this paper means ${p}^{3} \leq  m$ . For a randomized algorithm,when we say that its load is at most $L$ , we mean that its load is bounded by $L$ with probability at least $1 - 1/{p}^{c}$ where $c$ can be set to an arbitrarily large constant. The notation $\widetilde{O}\left( \text{.}\right) {hidesafafactorthatispolylogarithmicto}$ $m$ and $p$ .

假定机器的数量 $p$ 显著小于 $m$，在本文中这意味着 ${p}^{3} \leq  m$。对于一个随机算法，当我们说其负载至多为 $L$ 时，我们指的是其负载以至少 $1 - 1/{p}^{c}$ 的概率被 $L$ 所界定，其中 $c$ 可以被设定为任意大的常数。符号 $\widetilde{O}\left( \text{.}\right) {hidesafafactorthatispolylogarithmicto}$ $m$ 和 $p$。

1.2. Previous Results. Early work on join processing in the MPC model aimed to design algorithms performing only one round. Afrati and Ullman [3] explained how to answer a query $\mathcal{Q}$ with load $O\left( {m/{p}^{1/\left| \mathcal{Q}\right| }}\right)$ . Later,by refining their prior work in [6],Koutris,Beame, and Suciu [13] described an algorithm that can guarantee a load of $\widetilde{O}\left( {m/{p}^{1/\psi }}\right)$ ,where $\psi$ is the query's fractional edge quasi-packing number. To follow our discussion in Section 1, the reader does not need the formal definition of $\psi$ (which will be given in Section 2); it suffices to understand that $\psi$ is a positive constant which can vary significantly depending on $\mathcal{Q}$ . In [13],the authors also proved that any one-round algorithm must incur a load of $\Omega \left( {m/{p}^{1/\psi }}\right)$ ,under certain assumptions on the statistics available to the algorithm.

1.2. 过往研究成果。早期在大规模并行计算（MPC）模型下进行连接处理的工作旨在设计仅执行一轮的算法。阿夫拉蒂（Afrati）和厄尔曼（Ullman）[3] 阐述了如何以负载 $O\left( {m/{p}^{1/\left| \mathcal{Q}\right| }}\right)$ 来回答查询 $\mathcal{Q}$。后来，通过改进他们在文献 [6] 中的前期工作，库特里斯（Koutris）、比姆（Beame）和苏丘（Suciu）[13] 描述了一种算法，该算法能够保证负载为 $\widetilde{O}\left( {m/{p}^{1/\psi }}\right)$，其中 $\psi$ 是查询的分数边拟填充数（fractional edge quasi - packing number）。为了跟上我们在第 1 节的讨论，读者无需了解 $\psi$ 的正式定义（该定义将在第 2 节给出）；只需理解 $\psi$ 是一个正常数，它会根据查询 $\mathcal{Q}$ 有显著变化即可。在文献 [13] 中，作者还证明了，在算法可获取的统计信息的某些假设下，任何一轮算法都必然会产生 $\Omega \left( {m/{p}^{1/\psi }}\right)$ 的负载。

Departing from the one-round restriction, subsequent research has focused on algorithms performing multiple, albeit still a constant number of, rounds. The community already knows [13] that any constant-round algorithm must incur a load of $\Omega \left( {m/{p}^{1/\rho }}\right)$ answering a query,where $\rho$ is the query’s fractional edge covering number. As far as Section 1 is concerned,the reader does not need to worry about the definition of $\rho$ (which will appear in Section 2); it suffices to remember two facts:

突破一轮的限制后，后续研究聚焦于执行多轮（尽管轮数仍然是常数）的算法。业界已经知道 [13]，任何常数轮算法在回答查询时都必然会产生 $\Omega \left( {m/{p}^{1/\rho }}\right)$ 的负载，其中 $\rho$ 是查询的分数边覆盖数（fractional edge covering number）。就第 1 节而言，读者无需担心 $\rho$ 的定义（该定义将在第 2 节出现）；只需记住两个事实：

- Like $\psi ,\rho$ is a positive constant which can vary significantly depending on the query $\mathcal{Q}$ .

- 与 $\psi ,\rho$ 类似，$\mathcal{Q}$ 是一个正常数，它会根据查询 $\mathcal{Q}$ 有显著变化。

- On the same $\mathcal{Q},\rho$ never exceeds $\psi$ ,but can be much smaller than $\psi$ (more details in Section 2).

- 对于相同的 $\mathcal{Q},\rho$，其值从不超过 $\psi$，但可能远小于 $\psi$（第 2 节有更多细节）。

The second bullet indicates that $m/{p}^{1/\rho }$ can be far less than $m/{p}^{1/\psi }$ ,suggesting that we may hope to significantly reduce the load by going beyond only one round. Matching the lower bound $\Omega \left( {m/{p}^{1/\rho }}\right)$ with a concrete algorithm has been shown possible for several special query classes, including star joins [3], cycle joins [13], clique joins [13], line joins [3, 13], Loomis-Whitney joins [13], etc. The simple binary join defined in Section 1.1 captures cycle, clique,and line joins as special cases. Guaranteeing a load of $O\left( {m/{p}^{1/\rho }}\right)$ for arbitrary simple binary queries is still open.

第二点表明 $m/{p}^{1/\rho }$ 可能远小于 $m/{p}^{1/\psi }$，这意味着我们有望通过执行多轮而非仅一轮来显著降低负载。对于包括星型连接 [3]、循环连接 [13]、团连接 [13]、线性连接 [3, 13]、卢米斯 - 惠特尼连接 [13] 等在内的几个特殊查询类，已证明可以用具体算法达到下界 $\Omega \left( {m/{p}^{1/\rho }}\right)$。第 1.1 节定义的简单二元连接将循环、团和线性连接作为特殊情况包含在内。对于任意简单二元查询保证 $O\left( {m/{p}^{1/\rho }}\right)$ 的负载仍然是一个未解决的问题。

1.3. Our Contributions. The paper's main algorithmic contribution is to settle any simple binary join $\mathcal{Q}$ under the MPC model with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ in a constant number rounds (Theorem 13). The load is optimal up to a polylogarithmic factor. Our algorithm owes to a new theorem - we name the isolated cartesian product theorem (Theorem 7; see also Theorem 10) — that reveals a non-trivial fact on the problem's mathematical structure.

1.3. 我们的贡献。本文主要的算法贡献是在大规模并行计算（MPC）模型下，以常数轮数和负载 $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ 解决任何简单二元连接 $\mathcal{Q}$（定理 13）。该负载在一个多对数因子范围内是最优的。我们的算法得益于一个新定理——我们将其命名为孤立笛卡尔积定理（定理 7；另见定理 10）——该定理揭示了该问题数学结构中的一个非平凡事实。

<!-- Media -->

<!-- figureText: Ok $\mathrm{K} = \mathrm{k}$ G O d’. O L O H (c) After deleting (d) After semi-join black verticess reduction (a) A join query (b) A residual query -->

<img src="https://cdn.noedgeai.com/0195ccbd-9d97-7aa3-8943-c43bdb45848f_3.jpg?x=320&y=316&w=1147&h=351&r=0"/>

Figure 1: Processing a join by constraining heavy values

图 1：通过约束重值来处理连接

<!-- Media -->

Overview of Our Techniques. Consider the join query $\mathcal{Q}$ illustrated by the graph in Figure 1a. An edge connecting vertices $X$ and $Y$ represents a relation ${R}_{\{ X,Y\} }$ with scheme $\{ X,Y\} .\mathcal{Q}$ contains all the 18 relations represented by the edges in Figure 1a; $\operatorname{attset}\left( \mathcal{Q}\right)  = \{ \mathrm{A},\mathrm{B},\ldots ,\mathrm{L}\}$ has a size of 12 .

我们的技术概述。考虑图1a中的图所示的连接查询$\mathcal{Q}$。连接顶点$X$和$Y$的边表示一个关系${R}_{\{ X,Y\} }$，其模式为$\{ X,Y\} .\mathcal{Q}$，包含图1a中边所代表的所有18个关系；$\operatorname{attset}\left( \mathcal{Q}\right)  = \{ \mathrm{A},\mathrm{B},\ldots ,\mathrm{L}\}$的大小为12。

Set $\lambda  = \Theta \left( {p}^{1/\left( {2\rho }\right) }\right)$ where $\rho$ is the fractional edge covering number of $\mathcal{Q}$ (Section 2). A value $x \in$ dom is heavy if at least $m/\lambda$ tuples in an input relation $R \in  \mathcal{Q}$ carry $x$ on the same attribute. The number of heavy values is $O\left( \lambda \right)$ . A value $x \in  \mathbf{{dom}}$ is light if $x$ appears in at least one relation $R \in  \mathcal{Q}$ but is not heavy. A tuple in the join result may take a heavy or light value on each of the 12 attributes $\mathrm{A},\ldots ,\mathrm{L}$ . As there are $O\left( \lambda \right)$ choices on each attribute (i.e.,either a light value or one of the $O\left( \lambda \right)$ heavy values),there are $t = O\left( {\lambda }^{12}\right)$ "choice combinations" from all attributes; we will refer to each combination as a configuration. Our plan is to partition the set of $p$ servers into $t$ subsets of sizes ${p}_{1},{p}_{2},\ldots ,{p}_{t}$ with $\mathop{\sum }\limits_{{i = 1}}^{t}{p}_{i} = p$ ,and then dedicate ${p}_{i}$ servers $\left( {1 \leq  i \leq  t}\right)$ to computing the result tuples of the $i$ -th configuration. This can be done in parallel for all $O\left( {\lambda }^{12}\right)$ configurations. The challenge is to compute the query on each configuration with a load $O\left( {m/{p}^{1/\rho }}\right)$ ,given that only ${p}_{i}$ (which can be far less than $p$ ) servers are available for that subtask.

设$\lambda  = \Theta \left( {p}^{1/\left( {2\rho }\right) }\right)$，其中$\rho$是$\mathcal{Q}$的分数边覆盖数（第2节）。如果输入关系$R \in  \mathcal{Q}$中至少有$m/\lambda$个元组在同一属性上携带$x$，则值$x \in$ dom为重值。重值的数量为$O\left( \lambda \right)$。如果$x$至少出现在一个关系$R \in  \mathcal{Q}$中但不是重值，则值$x \in  \mathbf{{dom}}$为轻值。连接结果中的一个元组可能在12个属性$\mathrm{A},\ldots ,\mathrm{L}$的每个属性上取重值或轻值。由于每个属性有$O\left( \lambda \right)$种选择（即，要么是轻值，要么是$O\left( \lambda \right)$个重值之一），所有属性共有$t = O\left( {\lambda }^{12}\right)$种“选择组合”；我们将每种组合称为一个配置。我们的计划是将$p$台服务器的集合划分为$t$个子集，其大小为${p}_{1},{p}_{2},\ldots ,{p}_{t}$，且满足$\mathop{\sum }\limits_{{i = 1}}^{t}{p}_{i} = p$，然后分配${p}_{i}$台服务器$\left( {1 \leq  i \leq  t}\right)$来计算第$i$个配置的结果元组。这可以对所有$O\left( {\lambda }^{12}\right)$个配置并行进行。挑战在于，给定只有${p}_{i}$台（可能远少于$p$台）服务器可用于该子任务的情况下，以负载$O\left( {m/{p}^{1/\rho }}\right)$计算每个配置上的查询。

Figure 1b illustrates one possible configuration where we constrain attributes D, E, F, and $\mathrm{K}$ respectively to heavy values $\mathrm{d},\mathrm{e},\mathrm{f}$ ,and $\mathrm{k}$ and the other attributes to light values. Accordingly,vertices $\mathrm{D},\mathrm{E},\mathrm{F}$ ,and $\mathrm{K}$ are colored black in the figure. The configuration gives rise to a residual query ${\mathcal{Q}}^{\prime }$ :

图1b展示了一种可能的配置，其中我们分别将属性D、E、F和$\mathrm{K}$约束为重值$\mathrm{d},\mathrm{e},\mathrm{f}$和$\mathrm{k}$，并将其他属性约束为轻值。因此，顶点$\mathrm{D},\mathrm{E},\mathrm{F}$和$\mathrm{K}$在图中被涂成黑色。该配置产生了一个剩余查询${\mathcal{Q}}^{\prime }$：

- For each edge $\{ X,Y\}$ with two white vertices, ${\mathcal{Q}}^{\prime }$ has a relation ${R}_{\{ X,Y\} }^{\prime }$ that contains only the tuples in ${R}_{\{ X,Y\} } \in  \mathcal{Q}$ using light values on both $X$ and $Y$ ;

- 对于每条连接两个白色顶点的边$\{ X,Y\}$，${\mathcal{Q}}^{\prime }$有一个关系${R}_{\{ X,Y\} }^{\prime }$，该关系仅包含${R}_{\{ X,Y\} } \in  \mathcal{Q}$中在$X$和$Y$上都使用轻值的元组；

- For each edge $\{ X,Y\}$ with a white vertex $X$ and a black vertex $Y,{\mathcal{Q}}^{\prime }$ has a relation ${R}_{\{ X,Y\} }^{\prime }$ that contains only the tuples in ${R}_{\{ X,Y\} } \in  \mathcal{Q}$ each using a light value on $X$ and the constrained heavy value on $Y$ ;

- 对于每条连接一个白色顶点$X$和一个黑色顶点$Y,{\mathcal{Q}}^{\prime }$的边，${R}_{\{ X,Y\} }^{\prime }$有一个关系，该关系仅包含${R}_{\{ X,Y\} } \in  \mathcal{Q}$中每个在$X$上使用轻值且在$Y$上使用约束重值的元组；

- For each edge $\{ X,Y\}$ with two black vertices, ${\mathcal{Q}}^{\prime }$ has a relation ${R}_{\{ X,Y\} }^{\prime }$ with only one tuple that takes the constrained heavy values on $X$ and $Y$ ,respectively.

- 对于每条具有两个黑色顶点的边 $\{ X,Y\}$，关系 ${\mathcal{Q}}^{\prime }$ 存在一个关系 ${R}_{\{ X,Y\} }^{\prime }$，该关系仅包含一个元组，该元组分别在 $X$ 和 $Y$ 上取受约束的重值。

For example,a tuple in ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ must use light values on both $\mathrm{A}$ and $\mathrm{B}$ ; a tuple in ${R}_{\{ \mathrm{D},\mathrm{G}\} }^{\prime }$ must use value $d$ on $D$ and a light value on $G;{R}_{\{ D,K\} }^{\prime }$ has only a single tuple with values $d$ and $k$ on $\mathrm{D}$ and $\mathrm{K}$ ,respectively. Finding all result tuples for $\mathcal{Q}$ under the designated configuration amounts to evaluating ${\mathcal{Q}}^{\prime }$ .

例如，${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ 中的一个元组必须在 $\mathrm{A}$ 和 $\mathrm{B}$ 上都使用轻值；${R}_{\{ \mathrm{D},\mathrm{G}\} }^{\prime }$ 中的一个元组必须在 $D$ 上使用值 $d$，并且在 $G;{R}_{\{ D,K\} }^{\prime }$ 上使用轻值，$G;{R}_{\{ D,K\} }^{\prime }$ 仅包含一个元组，该元组在 $\mathrm{D}$ 和 $\mathrm{K}$ 上的值分别为 $d$ 和 $k$。在指定配置下找到 $\mathcal{Q}$ 的所有结果元组相当于计算 ${\mathcal{Q}}^{\prime }$。

Since the black attributes have had their values fixed in the configuration, they can be deleted from the residual query,after which some relations in ${\mathcal{Q}}^{\prime }$ become unary or even disappear. Relation ${R}_{\{ \mathrm{A},\mathrm{D}\} }^{\prime } \in  {\mathcal{Q}}^{\prime }$ ,for example,can be regarded as a unary relation over $\{ \mathrm{A}\}$ where every tuple is "piggybacked" the value d on D. Let us denote this unary relation as ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ ,which is illustrated in Figure 1c with a dotted edge extending from $\mathrm{A}$ and carrying the label $\mathrm{d}$ . The deletion of $\mathrm{D},\mathrm{E},\mathrm{F}$ ,and $\mathrm{K}$ results in 13 unary relations (e.g.,two of them are over $\{ \mathrm{A}\}  : {R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ and ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ ). Attributes $\mathrm{G},\mathrm{H}$ ,and $\mathrm{L}$ become isolated because they are not connected to any other vertices by solid edges. Relations ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime }$ ,and ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime }$ remain binary,whereas ${R}_{\{ \mathrm{D},\mathrm{K}\} }^{\prime }$ has disappeared (more precisely,if ${R}_{\{ \mathrm{D},\mathrm{K}\} }$ does not contain a tuple taking values $\mathrm{d}$ and $\mathrm{k}$ on $\mathrm{D}$ and $\mathrm{K}$ respectively,then ${\mathcal{Q}}^{\prime }$ has an empty answer; otherwise, we proceed in the way explained next).

由于黑色属性的值在配置中已固定，因此可以从剩余查询中删除这些属性，之后 ${\mathcal{Q}}^{\prime }$ 中的一些关系会变为一元关系甚至消失。例如，关系 ${R}_{\{ \mathrm{A},\mathrm{D}\} }^{\prime } \in  {\mathcal{Q}}^{\prime }$ 可以被视为关于 $\{ \mathrm{A}\}$ 的一元关系，其中每个元组都在 D 上“搭载”值 d。我们将这个一元关系表示为 ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$，如图 1c 所示，从 $\mathrm{A}$ 延伸出一条虚线边并带有标签 $\mathrm{d}$。删除 $\mathrm{D},\mathrm{E},\mathrm{F}$ 和 $\mathrm{K}$ 会产生 13 个一元关系（例如，其中两个是关于 $\{ \mathrm{A}\}  : {R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ 和 ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ 的）。属性 $\mathrm{G},\mathrm{H}$ 和 $\mathrm{L}$ 变得孤立，因为它们没有通过实线边与任何其他顶点相连。关系 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime }$ 和 ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime }$ 仍然是二元关系，而 ${R}_{\{ \mathrm{D},\mathrm{K}\} }^{\prime }$ 已消失（更准确地说，如果 ${R}_{\{ \mathrm{D},\mathrm{K}\} }$ 不包含一个分别在 $\mathrm{D}$ 和 $\mathrm{K}$ 上取值 $\mathrm{d}$ 和 $\mathrm{k}$ 的元组，那么 ${\mathcal{Q}}^{\prime }$ 的答案为空；否则，我们按接下来解释的方式继续处理）。

Our algorithm solves the residual query ${\mathcal{Q}}^{\prime }$ of Figure 1c as follows:

我们的算法按如下方式解决图 1c 中的剩余查询 ${\mathcal{Q}}^{\prime }$：

(1) Perform a semi-join reduction. There are two steps. First,for every vertex $X$ in Figure 1c,intersect all the unary relations over $\{ X\}$ (if any) into a single list ${R}_{\{ X\} }^{\prime \prime }$ . For example,the two unary relations ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ and ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ of $\mathrm{A}$ are intersected to produce ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ ; only the values in the intersection can appear in the join result. Second,for every non-isolated attribute $X$ in Figure 1c,use ${R}_{\{ X\} }^{\prime \prime }$ to shrink each binary relation ${R}_{\{ X,Y\} }^{\prime }$ (for all relevant $Y$ ) to eliminate tuples whose $X$ -values are absent in ${R}_{\{ X\} }^{\prime \prime }$ . This reduces ${R}_{\{ X,Y\} }^{\prime }$ to a subset ${R}_{\{ X,Y\} }^{\prime \prime }$ . For example,every tuple in ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ uses an A-value from ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ and a B-value from ${R}_{\{ \mathrm{B}\} }^{\prime \prime }$ .

(1) 执行半连接约简。有两个步骤。首先，对于图1c中的每个顶点 $X$，将所有关于 $\{ X\}$ 的一元关系（如果有的话）相交成一个单一列表 ${R}_{\{ X\} }^{\prime \prime }$。例如，$\mathrm{A}$ 的两个一元关系 ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ 和 ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ 相交得到 ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$；只有交集中的值才能出现在连接结果中。其次，对于图1c中的每个非孤立属性 $X$，使用 ${R}_{\{ X\} }^{\prime \prime }$ 来缩减每个二元关系 ${R}_{\{ X,Y\} }^{\prime }$（对于所有相关的 $Y$），以消除那些 $X$ 值不在 ${R}_{\{ X\} }^{\prime \prime }$ 中的元组。这将 ${R}_{\{ X,Y\} }^{\prime }$ 缩减为一个子集 ${R}_{\{ X,Y\} }^{\prime \prime }$。例如，${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ 中的每个元组使用来自 ${R}_{\{ \mathrm{A}\} }^{\prime \prime }$ 的A值和来自 ${R}_{\{ \mathrm{B}\} }^{\prime \prime }$ 的B值。

(2) Compute a cartesian product. The residual query ${\mathcal{Q}}^{\prime }$ can now be further simplified into a join query ${\mathcal{Q}}^{\prime \prime }$ which includes (i) the relation ${R}_{\{ X\} }^{\prime \prime }$ for every isolated attribute $X$ ,and (ii) the relation ${R}_{\{ X,Y\} }^{\prime \prime }$ for every solid edge in Figure 1c. Figure 1d gives a neater view of ${\mathcal{Q}}^{\prime \prime }$ ; clearly, $\operatorname{Join}\left( {\mathcal{Q}}^{\prime \prime }\right)$ is the cartesian product of ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime },{R}_{\{ \mathrm{L}\} }^{\prime \prime },{R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }$ ,and the result of the "triangle join" $\left\{  {{R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime \prime }}\right\}$ .

(2) 计算笛卡尔积。剩余查询 ${\mathcal{Q}}^{\prime }$ 现在可以进一步简化为一个连接查询 ${\mathcal{Q}}^{\prime \prime }$，它包括 (i) 每个孤立属性 $X$ 的关系 ${R}_{\{ X\} }^{\prime \prime }$，以及 (ii) 图1c中每条实边的关系 ${R}_{\{ X,Y\} }^{\prime \prime }$。图1d给出了 ${\mathcal{Q}}^{\prime \prime }$ 的更清晰视图；显然，$\operatorname{Join}\left( {\mathcal{Q}}^{\prime \prime }\right)$ 是 ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime },{R}_{\{ \mathrm{L}\} }^{\prime \prime },{R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }$ 的笛卡尔积，也是“三角形连接” $\left\{  {{R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime \prime }}\right\}$ 的结果。

As mentioned earlier,we plan to use only a small subset of the $p$ servers to compute ${\mathcal{Q}}^{\prime }$ . It turns out that the load of our strategy depends heavily on the cartesian product of the unary relations ${R}_{\{ X\} }^{\prime \prime }$ (one for every isolated attribute $X$ ,i.e., ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ ,and ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$ in our example) in a configuration. Ideally, if the cartesian product of every configuration is small, we can prove a load of $O\left( {m/{p}^{1/\rho }}\right)$ easily. Unfortunately,this is not true: in the worst case, the cartesian products of various configurations can differ dramatically.

如前所述，我们计划仅使用 $p$ 服务器的一个小子集来计算 ${\mathcal{Q}}^{\prime }$。结果表明，我们策略的负载在很大程度上取决于配置中一元关系 ${R}_{\{ X\} }^{\prime \prime }$ 的笛卡尔积（每个孤立属性 $X$ 对应一个，即在我们的示例中为 ${R}_{\{ \mathrm{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime }$ 和 ${R}_{\{ \mathrm{L}\} }^{\prime \prime }$）。理想情况下，如果每个配置的笛卡尔积都很小，我们可以轻松证明负载为 $O\left( {m/{p}^{1/\rho }}\right)$。不幸的是，情况并非如此：在最坏的情况下，各种配置的笛卡尔积可能会有很大差异。

Our isolated cartesian product theorem (Theorem 7) shows that the cartesian product size is small when averaged over all the possible configurations. This property allows us to allocate a different number of machines to process each configuration in parallel while ensuring that the total number of machines required will not exceed $p$ . The theorem is of independent interest and may be useful for developing join algorithms under other computation models (e.g., the external memory model [4]; see Section 7).

我们的孤立笛卡尔积定理（定理7）表明，当对所有可能的配置进行平均时，笛卡尔积的大小很小。这一特性使我们能够为每个配置分配不同数量的机器进行并行处理，同时确保所需的机器总数不超过$p$。该定理具有独立的研究价值，可能有助于在其他计算模型（例如，外部内存模型[4]；见第7节）下开发连接算法。

1.4. An Application: Subgraph Enumeration. The joins studied in this paper bear close relevance to the subgraph enumeration problem, where the goal is to find all occurrences of a pattern subgraph ${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$ in a graph $G = \left( {V,E}\right)$ . This problem is NP-hard [7] if the size of ${G}^{\prime }$ is unconstrained,but is polynomial-time solvable when ${G}^{\prime }$ has only a constant number of vertices. In the MPC model,the edges of $G$ are evenly distributed onto the $p$ machines at the beginning, whereas an algorithm must produce every occurrence on at least one machine in the end. The following facts are folklore regarding a constant-size ${G}^{\prime }$ :

1.4. 应用：子图枚举。本文研究的连接与子图枚举问题密切相关，该问题的目标是找出图$G = \left( {V,E}\right)$中模式子图${G}^{\prime } = \left( {{V}^{\prime },{E}^{\prime }}\right)$的所有出现位置。如果${G}^{\prime }$的大小不受限制，那么这个问题是NP难问题[7]，但当${G}^{\prime }$的顶点数量为常数时，该问题可以在多项式时间内解决。在MPC模型中，$G$的边在开始时均匀分布到$p$台机器上，而算法最终必须在至少一台机器上输出每个出现位置。关于固定大小的${G}^{\prime }$，有以下常识性结论：

<!-- Media -->

<table><tr><td>symbol</td><td>meaning</td><td>definition</td></tr><tr><td>$p$</td><td>number of machines</td><td>Sec 1.1</td></tr><tr><td>Q</td><td>join query</td><td>Sec 1.1</td></tr><tr><td>$m$</td><td>input size of $\mathcal{Q}$</td><td>(1.1)</td></tr><tr><td>Join (Q)</td><td>result of $\mathcal{Q}$</td><td>Sec 1.1</td></tr><tr><td>attset(Q)</td><td>set of attributes in the relations of $\mathcal{Q}$</td><td>Sec 1.1</td></tr><tr><td>$\mathcal{G}\left( {\mathcal{V},\mathcal{E}}\right)$</td><td>hypergraph of $\mathcal{Q}$</td><td>Sec 2</td></tr><tr><td>$W$</td><td>fractional edge covering/packing of $\mathcal{G}$</td><td>Sec 2</td></tr><tr><td>$W\left( e\right)$</td><td>weight of an edge $e \in  \mathcal{E}$</td><td>Sec 2</td></tr><tr><td>$\rho \left( {\operatorname{or}\tau }\right)$</td><td>fractional edge covering (or packing) number of $\mathcal{G}$</td><td>Sec 2</td></tr><tr><td>${R}_{e}\left( {e \in  \mathcal{E}}\right)$</td><td>relation $R \in  \mathcal{Q}$ with $\operatorname{scheme}\left( R\right)  = e$</td><td>Sec 2</td></tr><tr><td>$\lambda$</td><td>heavy parameter</td><td>Sec 4</td></tr><tr><td>H</td><td>set of heavy attributes in attset(Q)</td><td>Sec 4</td></tr><tr><td>config(Q,H)</td><td>set of configurations of $\mathcal{H}$</td><td>Sec 4</td></tr><tr><td>$\eta$</td><td>configuration</td><td>Sec 4</td></tr><tr><td>${R}_{e}^{\prime }\left( \eta \right)$</td><td>residual relation of $e \in  \mathcal{E}$ under $\mathbf{\eta }$</td><td>Sec 4</td></tr><tr><td>${\mathcal{Q}}^{\prime }\left( \eta \right)$</td><td>residual query under $\mathbf{\eta }$</td><td>(4.1)</td></tr><tr><td>$k$</td><td>size of attset(Q)</td><td>Lemma 6</td></tr><tr><td>${m}_{\eta }$</td><td>input size of ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$</td><td>Lemma 6</td></tr><tr><td>$\mathcal{L}$</td><td>set of light attributes in attset(Q)</td><td>(5.2)</td></tr><tr><td>$\mathcal{I}$</td><td>set of isolated attributes in attset(Q)</td><td>(5.3)</td></tr><tr><td>${R}_{X}^{\prime \prime }\left( \eta \right)$</td><td>relation on attribute $X$ after semi-join reduction</td><td>(5.4)</td></tr><tr><td>${R}_{e}^{\prime \prime }\left( \eta \right)$</td><td>relation on $e \in  \mathcal{E}$ after semi-join reduction</td><td>Sec 5.2</td></tr><tr><td>${\mathcal{Q}}_{\text{isolated}}^{\prime \prime }\left( \mathbf{\eta }\right)$</td><td>query on the isolated attributes after semi-join reduction</td><td>(5.5)</td></tr><tr><td>${\mathcal{Q}}_{\text{light}}^{\prime \prime }\left( \mathbf{\eta }\right)$</td><td>query on the light edges after semi-join reduction</td><td>(5.6)</td></tr><tr><td>${\mathcal{Q}}^{\prime \prime }\left( \eta \right)$</td><td>reduced query under $\mathbf{\eta }$</td><td>(5.7)</td></tr><tr><td>${W}_{\mathcal{I}}$</td><td>total weight of all vertices in $\mathcal{I}$ under fractional edge packing $W$</td><td>(5.10)</td></tr><tr><td>$\mathcal{I}$</td><td>non-empty subset of $\mathcal{I}$</td><td>Sec 5.4</td></tr><tr><td>${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \eta \right)$</td><td>query on the isolated attributes in $\mathcal{J}$ after semi-join reduction</td><td>(5.14)</td></tr><tr><td>Wj</td><td>total weight of all vertices in $\mathcal{J}$ under fractional edge packing $W$</td><td>(5.15)</td></tr></table>

<table><tbody><tr><td>符号</td><td>含义</td><td>定义</td></tr><tr><td>$p$</td><td>机器数量</td><td>第1.1节</td></tr><tr><td>Q</td><td>连接查询</td><td>第1.1节</td></tr><tr><td>$m$</td><td>$\mathcal{Q}$的输入大小</td><td>(1.1)</td></tr><tr><td>连接 (Q)</td><td>$\mathcal{Q}$的结果</td><td>第1.1节</td></tr><tr><td>attset(Q)</td><td>$\mathcal{Q}$的关系中的属性集</td><td>第1.1节</td></tr><tr><td>$\mathcal{G}\left( {\mathcal{V},\mathcal{E}}\right)$</td><td>$\mathcal{Q}$的超图</td><td>第2节</td></tr><tr><td>$W$</td><td>$\mathcal{G}$的分数边覆盖/填充</td><td>第2节</td></tr><tr><td>$W\left( e\right)$</td><td>边 $e \in  \mathcal{E}$ 的权重</td><td>第2节</td></tr><tr><td>$\rho \left( {\operatorname{or}\tau }\right)$</td><td>$\mathcal{G}$的分数边覆盖（或填充）数</td><td>第2节</td></tr><tr><td>${R}_{e}\left( {e \in  \mathcal{E}}\right)$</td><td>具有 $\operatorname{scheme}\left( R\right)  = e$ 的关系 $R \in  \mathcal{Q}$</td><td>第2节</td></tr><tr><td>$\lambda$</td><td>重参数</td><td>第4节</td></tr><tr><td>H</td><td>attset(Q) 中的重属性集</td><td>第4节</td></tr><tr><td>config(Q,H)</td><td>$\mathcal{H}$的配置集</td><td>第4节</td></tr><tr><td>$\eta$</td><td>配置</td><td>第4节</td></tr><tr><td>${R}_{e}^{\prime }\left( \eta \right)$</td><td>$e \in  \mathcal{E}$ 在 $\mathbf{\eta }$ 下的剩余关系</td><td>第4节</td></tr><tr><td>${\mathcal{Q}}^{\prime }\left( \eta \right)$</td><td>$\mathbf{\eta }$ 下的剩余查询</td><td>(4.1)</td></tr><tr><td>$k$</td><td>attset(Q) 的大小</td><td>引理6</td></tr><tr><td>${m}_{\eta }$</td><td>${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$的输入大小</td><td>引理6</td></tr><tr><td>$\mathcal{L}$</td><td>attset(Q) 中的轻属性集</td><td>(5.2)</td></tr><tr><td>$\mathcal{I}$</td><td>attset(Q) 中的孤立属性集</td><td>(5.3)</td></tr><tr><td>${R}_{X}^{\prime \prime }\left( \eta \right)$</td><td>半连接约简后属性 $X$ 上的关系</td><td>(5.4)</td></tr><tr><td>${R}_{e}^{\prime \prime }\left( \eta \right)$</td><td>半连接约简后 $e \in  \mathcal{E}$ 上的关系</td><td>第5.2节</td></tr><tr><td>${\mathcal{Q}}_{\text{isolated}}^{\prime \prime }\left( \mathbf{\eta }\right)$</td><td>半连接约简后对孤立属性的查询</td><td>(5.5)</td></tr><tr><td>${\mathcal{Q}}_{\text{light}}^{\prime \prime }\left( \mathbf{\eta }\right)$</td><td>半连接约简后对轻边的查询</td><td>(5.6)</td></tr><tr><td>${\mathcal{Q}}^{\prime \prime }\left( \eta \right)$</td><td>$\mathbf{\eta }$ 下的约简查询</td><td>(5.7)</td></tr><tr><td>${W}_{\mathcal{I}}$</td><td>分数边填充 $W$ 下 $\mathcal{I}$ 中所有顶点的总权重</td><td>(5.10)</td></tr><tr><td>$\mathcal{I}$</td><td>$\mathcal{I}$ 的非空子集</td><td>第5.4节</td></tr><tr><td>${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \eta \right)$</td><td>半连接约简后 $\mathcal{J}$ 中对孤立属性的查询</td><td>(5.14)</td></tr><tr><td>Wj</td><td>分数边填充 $W$ 下 $\mathcal{J}$ 中所有顶点的总权重</td><td>(5.15)</td></tr></tbody></table>

Table 1: Frequently used notations

表1：常用符号

<!-- Media -->

- Every constant-round subgraph enumeration algorithm must incur a load of $\Omega \left( {\left| E\right| /{p}^{1/\rho }}\right) ,{}^{1}$ where $\rho$ is the fractional edge covering number (Section 2) of ${G}^{\prime }$ .

- 每个常数轮的子图枚举算法必须产生$\Omega \left( {\left| E\right| /{p}^{1/\rho }}\right) ,{}^{1}$的负载，其中$\rho$是${G}^{\prime }$的分数边覆盖数（第2节）。

- The subgraph enumeration problem can be converted to a simple binary join with input size $O\left( \left| E\right| \right)$ and the same fractional edge covering number $\rho$ .

- 子图枚举问题可以转换为一个输入大小为$O\left( \left| E\right| \right)$且具有相同分数边覆盖数$\rho$的简单二元连接问题。

Given a constant-size ${G}^{\prime }$ ,our join algorithm (Theorem 13) solves subgraph enumeration with load $\widetilde{O}\left( {\left| E\right| /{p}^{1/\rho }}\right)$ ,which is optimal up to a polylogarithmic factor.

给定一个常数大小的${G}^{\prime }$，我们的连接算法（定理13）以$\widetilde{O}\left( {\left| E\right| /{p}^{1/\rho }}\right)$的负载解决子图枚举问题，该负载在一个多对数因子范围内是最优的。

1.5. Remarks. This paper is an extension of [12] and [20]. Ketsman and Suciu [12] were the first to discover a constant-round algorithm to solve simple binary joins with an asymptotically optimal load. Tao [20] introduced a preliminary version of the isolated cartesian product theorem and applied it to simplify the algorithm of [12]. The current work features a more powerful version of the isolated cartesian product theorem (see the remark in Section 5.5). Table 1 lists the symbols that will be frequently used.

1.5 备注。本文是文献[12]和[20]的扩展。Ketsman和Suciu [12]首次发现了一种常数轮算法，用于以渐近最优负载解决简单二元连接问题。Tao [20]引入了孤立笛卡尔积定理的初步版本，并将其应用于简化文献[12]中的算法。当前工作采用了更强大版本的孤立笛卡尔积定理（见第5.5节的备注）。表1列出了将频繁使用的符号。

---

<!-- Footnote -->

${}^{1}$ Here,we consider $\left| E\right|  \geq  \left| V\right|$ because vertices with no edges can be discarded directly.

${}^{1}$ 这里，我们考虑$\left| E\right|  \geq  \left| V\right|$，因为没有边的顶点可以直接舍弃。

<!-- Footnote -->

---

## 2. HYPERGRAPHS AND THE AGM BOUND

## 2. 超图与AGM界

We define a hypergraph $\mathcal{G}$ as a pair(V,E)where:

我们将超图$\mathcal{G}$定义为一个对(V, E)，其中：

- $\mathcal{V}$ is a finite set,where each element is called a vertex;

- $\mathcal{V}$是一个有限集，其中每个元素称为一个顶点；

- $\mathcal{E}$ is a set of subsets of $\mathcal{V}$ ,where each subset is called a (hyper-)edge.

- $\mathcal{E}$是$\mathcal{V}$的子集的集合，其中每个子集称为一条（超）边。

An edge $e$ is unary or binary if $\left| e\right|  = 1$ or 2,respectively. $\mathcal{G}$ is binary if all its edges are binary.

如果$\left| e\right|  = 1$分别为1或2，则边$e$分别为一元边或二元边。如果$\mathcal{G}$的所有边都是二元边，则$\mathcal{G}$是二元的。

Given a vertex $X \in  \mathcal{V}$ and an edge $e \in  \mathcal{E}$ ,we say that $X$ and $e$ are incident to each other if $X \in  e$ . Two distinct vertices $X,Y \in  \mathcal{V}$ are adjacent if there is an $e \in  \mathcal{E}$ containing $X$ and $Y$ . All hypergraphs discussed in this paper have the property that every vertex is incident to at least one edge.

给定一个顶点$X \in  \mathcal{V}$和一条边$e \in  \mathcal{E}$，如果$X \in  e$，我们称$X$和$e$相互关联。如果存在一条包含$X$和$Y$的边$e \in  \mathcal{E}$，则两个不同的顶点$X,Y \in  \mathcal{V}$相邻。本文讨论的所有超图都具有每个顶点至少与一条边关联的性质。

Given a subset $\mathcal{U}$ of $\mathcal{V}$ ,we define the subgraph induced by $\mathcal{U}$ as $\left( {\mathcal{U},{\mathcal{E}}_{\mathcal{U}}}\right)$ where ${\mathcal{E}}_{\mathcal{U}} =$ $\{ \mathcal{U} \cap  e \mid  e \in  \mathcal{E}\}$ .

给定$\mathcal{V}$的一个子集$\mathcal{U}$，我们将由$\mathcal{U}$诱导的子图定义为$\left( {\mathcal{U},{\mathcal{E}}_{\mathcal{U}}}\right)$，其中${\mathcal{E}}_{\mathcal{U}} =$ $\{ \mathcal{U} \cap  e \mid  e \in  \mathcal{E}\}$。

Fractional Edge Coverings and Packings. Let $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ be a hypergraph and $W$ be a function mapping $\mathcal{E}$ to real values in $\left\lbrack  {0,1}\right\rbrack$ . We call $W\left( e\right)$ the weight of edge $e$ and $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ the total weight of $W$ . Given a vertex $X \in  \mathcal{V}$ ,we refer to $\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)$ (i.e., the sum of the weights of all the edges incident to $X$ ) as the weight of $X$ .

分数边覆盖与填充。设$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$是一个超图，$W$是一个将$\mathcal{E}$映射到$\left\lbrack  {0,1}\right\rbrack$中的实数值的函数。我们称$W\left( e\right)$为边$e$的权重，称$\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$为$W$的总权重。给定一个顶点$X \in  \mathcal{V}$，我们将$\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)$（即与$X$关联的所有边的权重之和）称为$X$的权重。

$W$ is a fractional edge covering of $\mathcal{G}$ if the weight of every vertex $X \in  \mathcal{V}$ is at least 1 . The fractional edge covering number of $\mathcal{G}$ - denoted as $\rho \left( \mathcal{G}\right)$ - equals the smallest total weight of all the fractional edge coverings. $W$ is a fractional edge packing if the weight of every vertex $X \in  \mathcal{V}$ is at most 1 . The fractional edge packing number of $\mathcal{G}$ - denoted as $\tau \left( \mathcal{G}\right)$ - equals the largest total weight of all the fractional edge packings. A fractional edge packing $W$ is tight if it is simultaneously also a fractional edge covering; likewise,a fractional edge covering $W$ is tight if it is simultaneously also a fractional edge packing. Note that in a tight fractional edge covering/packing, the weight of every vertex must be exactly 1 .

$W$ 是 $\mathcal{G}$ 的一个分数边覆盖（fractional edge covering），如果每个顶点 $X \in  \mathcal{V}$ 的权重至少为 1。$\mathcal{G}$ 的分数边覆盖数（用 $\rho \left( \mathcal{G}\right)$ 表示）等于所有分数边覆盖的最小总权重。$W$ 是一个分数边填充（fractional edge packing），如果每个顶点 $X \in  \mathcal{V}$ 的权重至多为 1。$\mathcal{G}$ 的分数边填充数（用 $\tau \left( \mathcal{G}\right)$ 表示）等于所有分数边填充的最大总权重。如果一个分数边填充 $W$ 同时也是一个分数边覆盖，那么它是紧的（tight）；同样地，如果一个分数边覆盖 $W$ 同时也是一个分数边填充，那么它是紧的。注意，在一个紧的分数边覆盖/填充中，每个顶点的权重必须恰好为 1。

Binary hypergraphs have several interesting properties:

二元超图（Binary hypergraphs）有几个有趣的性质：

Lemma 1 . If $\mathcal{G}$ is binary,then:

引理 1。如果 $\mathcal{G}$ 是二元的，那么：

- $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$ ; furthermore, $\rho \left( \mathcal{G}\right)  \geq  \tau \left( \mathcal{G}\right)$ ,where the equality holds if and only if $\mathcal{G}$ admits a tight fractional edge packing (a.k.a. tight fractional edge covering).

- $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$；此外，$\rho \left( \mathcal{G}\right)  \geq  \tau \left( \mathcal{G}\right)$，其中等式成立当且仅当 $\mathcal{G}$ 存在一个紧的分数边填充（也称为紧的分数边覆盖）。

- $\mathcal{G}$ admits a fractional edge packing $W$ of total weight $\tau \left( \mathcal{G}\right)$ such that

- $\mathcal{G}$ 存在一个总权重为 $\tau \left( \mathcal{G}\right)$ 的分数边填充 $W$，使得

(1) the weight of every vertex $X \in  \mathcal{V}$ is either 0 or 1 ;

(1) 每个顶点 $X \in  \mathcal{V}$ 的权重要么为 0 要么为 1；

(2) if $\mathcal{Z}$ is the set of vertices in $\mathcal{V}$ with weight0,then $\rho \left( \mathcal{G}\right)  - \tau \left( \mathcal{G}\right)  = \left| \mathcal{Z}\right|$ .

(2) 如果 $\mathcal{Z}$ 是 $\mathcal{V}$ 中权重为 0 的顶点集，那么 $\rho \left( \mathcal{G}\right)  - \tau \left( \mathcal{G}\right)  = \left| \mathcal{Z}\right|$。

Proof. The first bullet is proved in Theorem 2.2.7 of [19]. The fractional edge packing $W$ in Theorem 2.1.5 of [19] satisfies Property (1) of the second bullet. Regarding such a $W$ ,we

证明。第一个要点在 [19] 的定理 2.2.7 中已证明。[19] 的定理 2.1.5 中的分数边填充 $W$ 满足第二个要点的性质 (1)。对于这样的 $W$，我们

have

有

$$
\tau \left( \mathcal{G}\right)  = \text{ total weight of }W = \frac{1}{2}\mathop{\sum }\limits_{{X \in  \mathcal{V}}}\left( {\text{ weight of }X}\right)  = \left( {\left| \mathcal{V}\right|  - \left| \mathcal{Z}\right| }\right) /2.
$$

Plugging this into $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$ yields $\rho \left( \mathcal{G}\right)  = \left( {\left| \mathcal{V}\right|  + \left| \mathcal{Z}\right| }\right) /2$ . Hence,Property (2) follows.

将其代入 $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$ 得到 $\rho \left( \mathcal{G}\right)  = \left( {\left| \mathcal{V}\right|  + \left| \mathcal{Z}\right| }\right) /2$。因此，性质 (2) 成立。

Example. Suppose that $\mathcal{G}$ is the binary hypergraph in Figure 1a. It has a fractional edge covering number $\rho \left( \mathcal{G}\right)  = {6.5}$ ,as is achieved by the function ${W}_{1}$ that maps $\{ \mathrm{G},\mathrm{F}\} ,\{ \mathrm{D},\mathrm{K}\}$ , $\{ \mathrm{I},\mathrm{J}\} ,\{ \mathrm{E},\mathrm{H}\}$ ,and $\{ \mathrm{E},\mathrm{L}\}$ to $1,\{ \mathrm{A},\mathrm{B}\} ,\{ \mathrm{A},\mathrm{C}\}$ ,and $\{ \mathrm{B},\mathrm{C}\}$ to $1/2$ ,and the other edges to0. Its fractional edge packing number is $\tau \left( \mathcal{G}\right)  = {5.5}$ ,achieved by the function ${W}_{2}$ which is the same as ${W}_{1}$ except that ${W}_{2}$ maps $\{ \mathrm{E},\mathrm{L}\}$ to 0 . Note that ${W}_{2}$ satisfies both properties of the second bullet (here $\mathcal{Z} = \{ \mathrm{L}\}$ ).

示例。假设$\mathcal{G}$是图1a中的二元超图（binary hypergraph）。它有一个分数边覆盖数（fractional edge covering number）$\rho \left( \mathcal{G}\right)  = {6.5}$，这可以通过函数${W}_{1}$实现，该函数将$\{ \mathrm{G},\mathrm{F}\} ,\{ \mathrm{D},\mathrm{K}\}$、$\{ \mathrm{I},\mathrm{J}\} ,\{ \mathrm{E},\mathrm{H}\}$和$\{ \mathrm{E},\mathrm{L}\}$映射到$1,\{ \mathrm{A},\mathrm{B}\} ,\{ \mathrm{A},\mathrm{C}\}$，将$\{ \mathrm{B},\mathrm{C}\}$映射到$1/2$，并将其他边映射到0。它的分数边填充数（fractional edge packing number）是$\tau \left( \mathcal{G}\right)  = {5.5}$，由函数${W}_{2}$实现，该函数与${W}_{1}$相同，只是${W}_{2}$将$\{ \mathrm{E},\mathrm{L}\}$映射到0。注意，${W}_{2}$满足第二个要点的两个性质（这里是$\mathcal{Z} = \{ \mathrm{L}\}$）。

Hypergraph of a Join Query and the AGM Bound. Every join $\mathcal{Q}$ defines a hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ where $\mathcal{V} =$ attset(Q)and $\mathcal{E} = \{$ scheme $\left( R\right)  \mid  R \in  \mathcal{Q}\}$ . When $\mathcal{Q}$ is simple,for each edge $e \in  \mathcal{E}$ we denote by ${R}_{e}$ the input relation $R \in  \mathcal{Q}$ with $e =$ scheme(R). The following result is known as the ${AGM}$ bound:

连接查询（Join Query）的超图与AGM界。每个连接$\mathcal{Q}$定义了一个超图$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$，其中$\mathcal{V} =$ attset(Q)且$\mathcal{E} = \{$ scheme $\left( R\right)  \mid  R \in  \mathcal{Q}\}$。当$\mathcal{Q}$是简单连接时，对于每条边$e \in  \mathcal{E}$，我们用${R}_{e}$表示输入关系$R \in  \mathcal{Q}$，其中$e =$ scheme(R)。以下结果被称为${AGM}$界：

Lemma 2 [5]. Let $\mathcal{Q}$ be a simple binary join and $W$ be any fractional edge covering of the hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ defined by $\mathcal{Q}$ . Then, $\left| {\operatorname{Join}\left( \mathcal{Q}\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\right| }^{W\left( e\right) }$ .

引理2 [5]。设$\mathcal{Q}$是一个简单二元连接（simple binary join），$W$是由$\mathcal{Q}$定义的超图$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$的任意分数边覆盖。那么，$\left| {\operatorname{Join}\left( \mathcal{Q}\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\right| }^{W\left( e\right) }$。

The fractional edge covering number of $\mathcal{Q}$ equals $\rho \left( \mathcal{G}\right)$ and,similarly,the fractional edge packing number of $\mathcal{Q}$ equals $\tau \left( \mathcal{G}\right)$ .

$\mathcal{Q}$的分数边覆盖数等于$\rho \left( \mathcal{G}\right)$，类似地，$\mathcal{Q}$的分数边填充数等于$\tau \left( \mathcal{G}\right)$。

Remark on the Fractional Edge Quasi-Packing Number. Although the technical development in the subsequent sections is irrelevant to "fractional edge quasi-packing number", we provide a full definition of the concept here because it enables the reader to better distinguish our solution and the one-round algorithm of [13] (reviewed in Section 1.2) Consider a hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ . For each subset $\mathcal{U} \subseteq  \mathcal{V}$ ,let ${\mathcal{G}}_{\smallsetminus \mathcal{U}}$ be the graph obtained by removing $\mathcal{U}$ from all the edges of $\mathcal{E}$ ,or formally: ${\mathcal{G}}_{\smallsetminus \mathcal{U}} = \left( {\mathcal{V} \smallsetminus  \mathcal{U},{\mathcal{E}}_{\smallsetminus \mathcal{U}}}\right)$ where ${\mathcal{E}}_{\smallsetminus \mathcal{U}} = \{ e \smallsetminus  \mathcal{U} \mid  e \in$ $\mathcal{E}$ and $e \smallsetminus  \mathcal{U} \neq  \varnothing \}$ . The fractional edge quasi-packing number of $\mathcal{G}$ - denoted as $\psi \left( \mathcal{G}\right)$ — is

关于分数边准填充数的注记。尽管后续章节中的技术发展与“分数边准填充数”无关，但我们在此给出该概念的完整定义，因为这能让读者更好地区分我们的解决方案和文献[13]中的一轮算法（在1.2节中回顾）。考虑一个超图$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$。对于每个子集$\mathcal{U} \subseteq  \mathcal{V}$，令${\mathcal{G}}_{\smallsetminus \mathcal{U}}$为从$\mathcal{E}$的所有边中移除$\mathcal{U}$后得到的图，或者形式上表示为：${\mathcal{G}}_{\smallsetminus \mathcal{U}} = \left( {\mathcal{V} \smallsetminus  \mathcal{U},{\mathcal{E}}_{\smallsetminus \mathcal{U}}}\right)$，其中${\mathcal{E}}_{\smallsetminus \mathcal{U}} = \{ e \smallsetminus  \mathcal{U} \mid  e \in$ $\mathcal{E}$且$e \smallsetminus  \mathcal{U} \neq  \varnothing \}$。$\mathcal{G}$的分数边准填充数（记为$\psi \left( \mathcal{G}\right)$）是

$$
\psi \left( \mathcal{G}\right)  = \mathop{\max }\limits_{{\text{all }\mathcal{U} \subseteq  \mathcal{V}}}\tau \left( {\mathcal{G}}_{\smallsetminus \mathcal{U}}\right) 
$$

where $\tau \left( {\mathcal{G}}_{\smallsetminus \mathcal{U}}\right)$ is the fractional edge packing number of ${\mathcal{G}}_{\smallsetminus \mathcal{U}}$ .

其中$\tau \left( {\mathcal{G}}_{\smallsetminus \mathcal{U}}\right)$是${\mathcal{G}}_{\smallsetminus \mathcal{U}}$的分数边填充数。

In [13],Koutris,Beame,and Suciu proved that $\psi \left( \mathcal{G}\right)  \geq  \rho \left( \mathcal{G}\right)$ holds on any $\mathcal{G}$ (which need not be binary). In general, $\psi \left( \mathcal{G}\right)$ can be considerably higher than $\rho \left( \mathcal{G}\right)$ . In fact,this is true even on "regular" binary graphs, about which we mention two examples (both can be found in [13]):

在文献[13]中，Koutris、Beame和Suciu证明了$\psi \left( \mathcal{G}\right)  \geq  \rho \left( \mathcal{G}\right)$对任何$\mathcal{G}$（不一定是二元的）都成立。一般来说，$\psi \left( \mathcal{G}\right)$可能比$\rho \left( \mathcal{G}\right)$高得多。事实上，即使在“正则”二元图上也是如此，我们举两个例子（都可以在文献[13]中找到）：

- when $\mathcal{G}$ is a clique, $\psi \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|  - 1$ but $\rho \left( \mathcal{G}\right)$ is only $\left| \mathcal{V}\right| /2$ ;

- 当$\mathcal{G}$是一个完全图时，$\psi \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|  - 1$，但$\rho \left( \mathcal{G}\right)$仅为$\left| \mathcal{V}\right| /2$；

- when $\mathcal{G}$ is a cycle, $\psi \left( \mathcal{G}\right)  = \lceil 2\left( {\left| \mathcal{V}\right|  - 1}\right) /3\rceil$ and $\rho \left( \mathcal{G}\right)$ is again $\left| \mathcal{V}\right| /2$ .

- 当$\mathcal{G}$是一个环时，$\psi \left( \mathcal{G}\right)  = \lceil 2\left( {\left| \mathcal{V}\right|  - 1}\right) /3\rceil$，并且$\rho \left( \mathcal{G}\right)$再次为$\left| \mathcal{V}\right| /2$。

If $\mathcal{G}$ is the hypergraph defined by a query $\mathcal{Q},\psi \left( \mathcal{G}\right)$ is said to be the query’s fractional edge covering number. It is evident from the above discussion that,when $\mathcal{G}$ is a clique or a cycle,the load $\widetilde{O}\left( {m/{p}^{1/\rho \left( \mathcal{G}\right) }}\right)$ of our algorithm improves the load $\widetilde{O}\left( {m/{p}^{1/\psi \left( \mathcal{G}\right) }}\right)$ of [13] by a polynomial factor.

如果$\mathcal{G}$是由查询定义的超图，$\mathcal{Q},\psi \left( \mathcal{G}\right)$被称为该查询的分数边覆盖数。从上述讨论可以明显看出，当$\mathcal{G}$是完全图或环时，我们算法的负载$\widetilde{O}\left( {m/{p}^{1/\rho \left( \mathcal{G}\right) }}\right)$比文献[13]的负载$\widetilde{O}\left( {m/{p}^{1/\psi \left( \mathcal{G}\right) }}\right)$提高了一个多项式因子。

## 3. FUNDAMENTAL MPC ALGORITHMS

## 3. 基本的MPC算法

This subsection will discuss several building-block routines in the MPC model useful later.

本小节将讨论MPC模型中几个后续有用的基本例程。

Cartesian Products. Suppose that $R$ and $S$ are relations with disjoint schemes. Their cartesian product,denoted as $R \times  S$ ,is a relation over $\operatorname{scheme}\left( R\right)  \cup  \operatorname{scheme}\left( S\right)$ that consists of all the tuples $\mathbf{u}$ over $\operatorname{scheme}\left( R\right)  \cup  \operatorname{scheme}\left( S\right)$ such that $\mathbf{u}\left\lbrack  {\operatorname{scheme}\left( R\right) }\right\rbrack   \in  R$ and $\mathbf{u}\left\lbrack  {\text{scheme}\left( S\right) }\right\rbrack   \in  S$ .

笛卡尔积。假设 $R$ 和 $S$ 是具有不相交模式的关系。它们的笛卡尔积，记为 $R \times  S$ ，是一个基于 $\operatorname{scheme}\left( R\right)  \cup  \operatorname{scheme}\left( S\right)$ 的关系，该关系由所有基于 $\operatorname{scheme}\left( R\right)  \cup  \operatorname{scheme}\left( S\right)$ 的元组 $\mathbf{u}$ 组成，使得 $\mathbf{u}\left\lbrack  {\operatorname{scheme}\left( R\right) }\right\rbrack   \in  R$ 且 $\mathbf{u}\left\lbrack  {\text{scheme}\left( S\right) }\right\rbrack   \in  S$ 。

The lemma below gives a deterministic algorithm for computing the cartesian product:

下面的引理给出了一种计算笛卡尔积的确定性算法：

Lemma 3 . Let $\mathcal{Q}$ be a set of $t = O\left( 1\right)$ relations ${R}_{1},{R}_{2},\ldots ,{R}_{t}$ with disjoint schemes. The tuples in ${R}_{i}\left( {1 \leq  i \leq  t}\right)$ have been labeled with ids $1,2,\ldots ,\left| {R}_{i}\right|$ ,respectively. We can deterministically compute $\operatorname{Join}\left( \mathcal{Q}\right)  = {R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{t}$ in one round with load

引理 3。设 $\mathcal{Q}$ 是一组包含 $t = O\left( 1\right)$ 个关系 ${R}_{1},{R}_{2},\ldots ,{R}_{t}$ 的集合，这些关系具有不相交的模式。 ${R}_{i}\left( {1 \leq  i \leq  t}\right)$ 中的元组分别被标记为 $1,2,\ldots ,\left| {R}_{i}\right|$ 。我们可以在一轮中以负载确定性地计算 $\operatorname{Join}\left( \mathcal{Q}\right)  = {R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{t}$ 

$$
O\left( {\mathop{\max }\limits_{{\text{non-empty }{\mathcal{Q}}^{\prime } \subseteq  \mathcal{Q}}}\frac{{\left| \operatorname{Join}\left( {\mathcal{Q}}^{\prime }\right) \right| }^{\frac{1}{\left| {\mathcal{Q}}^{\prime }\right| }}}{{p}^{\frac{1}{\left| {\mathcal{Q}}^{\prime }\right| }}}}\right)  \tag{3.1}
$$

using $p$ machines. Alternatively,if we assume $\left| {R}_{1}\right|  \geq  \left| {R}_{2}\right|  \geq  \ldots  \geq  \left| {R}_{t}\right|$ ,then the load can be written as

使用 $p$ 台机器。或者，如果我们假设 $\left| {R}_{1}\right|  \geq  \left| {R}_{2}\right|  \geq  \ldots  \geq  \left| {R}_{t}\right|$ ，那么负载可以写成

$$
O\left( {\mathop{\max }\limits_{{i = 1}}^{t}\frac{{\left| \operatorname{Join}\left( \left\{  {R}_{1},{R}_{2},\ldots ,{R}_{i}\right\}  \right) \right| }^{\frac{1}{i}}}{{p}^{\frac{1}{i}}}}\right) . \tag{3.2}
$$

In (3.1) and (3.2),the constant factors in the big-O depend on $t$ .

在 (3.1) 和 (3.2) 中，大 O 表示法中的常数因子取决于 $t$ 。

Proof. For each $i \in  \left\lbrack  {1,t}\right\rbrack$ ,define ${\mathcal{Q}}_{i} = \left\{  {{R}_{1},\ldots ,{R}_{i}}\right\}$ and ${L}_{i} = {\left| \operatorname{Join}\left( {\mathcal{Q}}_{i}\right) \right| }^{\frac{1}{i}}/{p}^{\frac{1}{i}}$ . Let ${t}^{\prime }$ be the largest integer satisfying $\left| {R}_{i}\right|  \geq  {L}_{i}$ for all $i \in  \left\lbrack  {1,{t}^{\prime }}\right\rbrack  ;{t}^{\prime }$ definitely exists because $\left| {R}_{1}\right|  \geq  {L}_{1} =$ $\left| {R}_{1}\right| /p$ . Note that this means $\left| {R}_{t}\right|  \leq  \left| {R}_{t - 1}\right|  \leq  \ldots  \leq  \left| {R}_{{t}^{\prime } + 1}\right|  < {L}_{{t}^{\prime } + 1}$ if ${t}^{\prime } < t$ .

证明。对于每个 $i \in  \left\lbrack  {1,t}\right\rbrack$ ，定义 ${\mathcal{Q}}_{i} = \left\{  {{R}_{1},\ldots ,{R}_{i}}\right\}$ 和 ${L}_{i} = {\left| \operatorname{Join}\left( {\mathcal{Q}}_{i}\right) \right| }^{\frac{1}{i}}/{p}^{\frac{1}{i}}$ 。设 ${t}^{\prime }$ 是满足对于所有 $i \in  \left\lbrack  {1,{t}^{\prime }}\right\rbrack  ;{t}^{\prime }$ 都有 $\left| {R}_{i}\right|  \geq  {L}_{i}$ 的最大整数。 ${t}^{\prime }$ 肯定存在，因为 $\left| {R}_{1}\right|  \geq  {L}_{1} =$ $\left| {R}_{1}\right| /p$ 。注意，这意味着如果 ${t}^{\prime } < t$ ，则 $\left| {R}_{t}\right|  \leq  \left| {R}_{t - 1}\right|  \leq  \ldots  \leq  \left| {R}_{{t}^{\prime } + 1}\right|  < {L}_{{t}^{\prime } + 1}$ 。

Next,we will explain how to obtain $\operatorname{Join}\left( {\mathcal{Q}}_{{t}^{\prime }}\right)$ with load $O\left( {L}_{{t}^{\prime }}\right)$ . If ${t}^{\prime } < t$ ,this implies that $\operatorname{Join}\left( \mathcal{Q}\right)$ can be obtained with load $O\left( {{L}_{{t}^{\prime }} + {L}_{{t}^{\prime } + 1}}\right)$ because ${R}_{{t}^{\prime } + 1},\ldots ,{R}_{t}$ can be broadcast to all the machines with an extra load $O\left( {{L}_{{t}^{\prime } + 1} \cdot  \left( {t - {t}^{\prime }}\right) }\right)  = O\left( {L}_{{t}^{\prime } + 1}\right)$ .

接下来，我们将解释如何在负载为 $O\left( {L}_{{t}^{\prime }}\right)$ 的情况下获得 $\operatorname{Join}\left( {\mathcal{Q}}_{{t}^{\prime }}\right)$。如果 ${t}^{\prime } < t$，这意味着在负载为 $O\left( {{L}_{{t}^{\prime }} + {L}_{{t}^{\prime } + 1}}\right)$ 的情况下可以获得 $\operatorname{Join}\left( \mathcal{Q}\right)$，因为 ${R}_{{t}^{\prime } + 1},\ldots ,{R}_{t}$ 可以以额外负载 $O\left( {{L}_{{t}^{\prime } + 1} \cdot  \left( {t - {t}^{\prime }}\right) }\right)  = O\left( {L}_{{t}^{\prime } + 1}\right)$ 广播到所有机器。

Align the machines into a ${t}^{\prime }$ -dimensional ${p}_{1} \times  {p}_{2} \times  \ldots  \times  {p}_{{t}^{\prime }}$ grid where

将机器排列成一个 ${t}^{\prime }$ 维的 ${p}_{1} \times  {p}_{2} \times  \ldots  \times  {p}_{{t}^{\prime }}$ 网格，其中

$$
{p}_{i} = \left\lfloor  {\left| {R}_{i}\right| /{L}_{{t}^{\prime }}}\right\rfloor  
$$

for each $i \in  \left\lbrack  {1,{t}^{\prime }}\right\rbrack$ . This is possible because $\left| {R}_{i}\right|  \geq  \left| {R}_{{t}^{\prime }}\right|  \geq  {L}_{{t}^{\prime }}$ and $\mathop{\prod }\limits_{{i = 1}}^{{t}^{\prime }}\frac{\left| {R}_{i}\right| }{{L}_{{t}^{\prime }}} = \frac{\left| \operatorname{Join}\left( {\mathcal{Q}}_{{t}^{\prime }}\right) \right| }{{\left( {L}_{{t}^{\prime }}\right) }^{{t}^{\prime }}} = p$ . Each machine can be uniquely identified as a ${t}^{\prime }$ -dimensional point $\left( {{x}_{1},\ldots ,{x}_{{t}^{\prime }}}\right)$ in the grid where ${x}_{i} \in  \left\lbrack  {1,{p}_{i}}\right\rbrack$ for each $i \in  \left\lbrack  {1,{t}^{\prime }}\right\rbrack$ . For each ${R}_{i}$ ,we send its tuple with id $j \in  \left\lbrack  {1,\left| {R}_{i}\right| }\right\rbrack$ to all the machines whose coordinates on dimension $i$ are $\left( {j{\;\operatorname{mod}\;{p}_{i}}}\right)  + 1$ . Hence,a machine receives $O\left( {\left| {R}_{i}\right| /{p}_{i}}\right)  = O\left( {L}_{{t}^{\prime }}\right)$ tuples from ${R}_{i}$ ; and the overall load is $O\left( {{L}_{{t}^{\prime }} \cdot  {t}^{\prime }}\right)  = O\left( {L}_{{t}^{\prime }}\right)$ . For each combination of ${\mathbf{u}}_{1},{\mathbf{u}}_{2},\ldots ,{\mathbf{u}}_{{t}^{\prime }}$ where ${\mathbf{u}}_{i} \in  {R}_{i}$ ,some machine has received all of ${\mathbf{u}}_{1},\ldots ,{\mathbf{u}}_{{t}^{\prime }}$ . Therefore,the algorithm is able to produce the entire $\operatorname{Join}\left( {\mathcal{Q}}_{{t}^{\prime }}\right)$ .

对于每个 $i \in  \left\lbrack  {1,{t}^{\prime }}\right\rbrack$。这是可行的，因为 $\left| {R}_{i}\right|  \geq  \left| {R}_{{t}^{\prime }}\right|  \geq  {L}_{{t}^{\prime }}$ 和 $\mathop{\prod }\limits_{{i = 1}}^{{t}^{\prime }}\frac{\left| {R}_{i}\right| }{{L}_{{t}^{\prime }}} = \frac{\left| \operatorname{Join}\left( {\mathcal{Q}}_{{t}^{\prime }}\right) \right| }{{\left( {L}_{{t}^{\prime }}\right) }^{{t}^{\prime }}} = p$。每台机器可以唯一地标识为网格中的一个 ${t}^{\prime }$ 维点 $\left( {{x}_{1},\ldots ,{x}_{{t}^{\prime }}}\right)$，其中对于每个 $i \in  \left\lbrack  {1,{t}^{\prime }}\right\rbrack$ 有 ${x}_{i} \in  \left\lbrack  {1,{p}_{i}}\right\rbrack$。对于每个 ${R}_{i}$，我们将其 ID 为 $j \in  \left\lbrack  {1,\left| {R}_{i}\right| }\right\rbrack$ 的元组发送到在第 $i$ 维上坐标为 $\left( {j{\;\operatorname{mod}\;{p}_{i}}}\right)  + 1$ 的所有机器。因此，一台机器从 ${R}_{i}$ 接收 $O\left( {\left| {R}_{i}\right| /{p}_{i}}\right)  = O\left( {L}_{{t}^{\prime }}\right)$ 个元组；总负载为 $O\left( {{L}_{{t}^{\prime }} \cdot  {t}^{\prime }}\right)  = O\left( {L}_{{t}^{\prime }}\right)$。对于 ${\mathbf{u}}_{1},{\mathbf{u}}_{2},\ldots ,{\mathbf{u}}_{{t}^{\prime }}$ 的每种组合（其中 ${\mathbf{u}}_{i} \in  {R}_{i}$），某台机器已经接收到所有的 ${\mathbf{u}}_{1},\ldots ,{\mathbf{u}}_{{t}^{\prime }}$。因此，该算法能够生成整个 $\operatorname{Join}\left( {\mathcal{Q}}_{{t}^{\prime }}\right)$。

The load in (3.2) matches a lower bound stated in Section 4.1.5 of [14]. The algorithm in the above proof generalizes an algorithm in [10] for computing the cartesian product of $t = 2$ relations. The randomized hypercube algorithm of [6] incurs a load higher than (3.2) by a logarithmic factor and can fail with a small probability.

(3.2)中的负载与文献[14]第4.1.5节中所述的下界相匹配。上述证明中的算法推广了文献[10]中用于计算$t = 2$个关系的笛卡尔积的算法。文献[6]中的随机超立方体算法产生的负载比(3.2)高一个对数因子，并且有很小的概率会失败。

Composition by Cartesian Product. If we already know how to solve queries ${\mathcal{Q}}_{1}$ and ${\mathcal{Q}}_{2}$ separately,we can compute the cartesian product of their results efficiently:

通过笛卡尔积进行组合。如果我们已经知道如何分别求解查询${\mathcal{Q}}_{1}$和${\mathcal{Q}}_{2}$，我们可以高效地计算它们结果的笛卡尔积：

Lemma 4 . Let ${\mathcal{Q}}_{1}$ and ${\mathcal{Q}}_{2}$ be two join queries satisfying the condition attset $\left( {\mathcal{Q}}_{1}\right)  \cap$ attset $\left( {\mathcal{Q}}_{2}\right)  = \varnothing$ . Let $m$ be the total number of tuples in the input relations of ${\mathcal{Q}}_{1}$ and ${\mathcal{Q}}_{2}$ . Suppose that

引理4。设${\mathcal{Q}}_{1}$和${\mathcal{Q}}_{2}$是两个满足条件属性集$\left( {\mathcal{Q}}_{1}\right)  \cap$属性集$\left( {\mathcal{Q}}_{2}\right)  = \varnothing$的连接查询。设$m$是${\mathcal{Q}}_{1}$和${\mathcal{Q}}_{2}$的输入关系中的元组总数。假设

- with probability at least $1 - {\delta }_{1}$ ,we can compute in one round $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ with load $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$ using ${p}_{1}$ machines;

- 至少以概率$1 - {\delta }_{1}$，我们可以使用${p}_{1}$台机器在一轮内以负载$\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$计算出$\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$；

- with probability at least $1 - {\delta }_{2}$ ,we can compute in one round $\operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ with load $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$ using ${p}_{2}$ machines.

- 至少以概率$1 - {\delta }_{2}$，我们可以使用${p}_{2}$台机器在一轮内以负载$\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$计算出$\operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$。

Then,with probability at least $1 - {\delta }_{1} - {\delta }_{2}$ ,we can compute $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ in one round with load $\widetilde{O}\left( {\max \left\{  {m/{p}_{1}^{1/{t}_{1}},m/{p}_{2}^{1/{t}_{2}}}\right\}  }\right)$ using ${p}_{1}{p}_{2}$ machines.

那么，至少以概率$1 - {\delta }_{1} - {\delta }_{2}$，我们可以使用${p}_{1}{p}_{2}$台机器在一轮内以负载$\widetilde{O}\left( {\max \left\{  {m/{p}_{1}^{1/{t}_{1}},m/{p}_{2}^{1/{t}_{2}}}\right\}  }\right)$计算出$\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$。

Proof. Let ${\mathcal{A}}_{1}$ and ${\mathcal{A}}_{2}$ be the algorithm for ${\mathcal{Q}}_{1}$ and ${\mathcal{Q}}_{2}$ ,respectively. If a tuple $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ is produced by ${\mathcal{A}}_{1}$ on the $i$ -th $\left( {i \in  \left\lbrack  {1,{p}_{1}}\right\rbrack  }\right)$ machine,we call $\mathbf{u}$ an $i$ -tuple. Similarly,if a tuple $\mathbf{v} \in  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ is produced by ${\mathcal{A}}_{2}$ on the $j$ -th $\left( {j \in  \left\lbrack  {1,{p}_{2}}\right\rbrack  }\right)$ machine,we call $\mathbf{v}$ a $j$ -tuple.

证明。设${\mathcal{A}}_{1}$和${\mathcal{A}}_{2}$分别是用于${\mathcal{Q}}_{1}$和${\mathcal{Q}}_{2}$的算法。如果元组$\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$是由${\mathcal{A}}_{1}$在第$i$台$\left( {i \in  \left\lbrack  {1,{p}_{1}}\right\rbrack  }\right)$机器上产生的，我们称$\mathbf{u}$为$i$ - 元组。类似地，如果元组$\mathbf{v} \in  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$是由${\mathcal{A}}_{2}$在第$j$台$\left( {j \in  \left\lbrack  {1,{p}_{2}}\right\rbrack  }\right)$机器上产生的，我们称$\mathbf{v}$为$j$ - 元组。

Arrange the ${p}_{1}{p}_{2}$ machines into a matrix where each row has ${p}_{1}$ machines and each column has ${p}_{2}$ machines (note that the number of rows is ${p}_{2}$ while the number of columns is ${p}_{1}$ ). For each row,we run ${A}_{1}$ using the ${p}_{1}$ machines on that row to compute $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$ ; this creates ${p}_{2}$ instances of ${A}_{1}$ (one per row). If ${A}_{1}$ is randomized,we instruct all those instances to take the same random choices. ${}^{2}$ This ensures:

将 ${p}_{1}{p}_{2}$ 台机器排列成一个矩阵，其中每行有 ${p}_{1}$ 台机器，每列有 ${p}_{2}$ 台机器（注意行数为 ${p}_{2}$，列数为 ${p}_{1}$）。对于每一行，我们使用该行的 ${p}_{1}$ 台机器运行 ${A}_{1}$ 来计算 $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)$；这会创建 ${p}_{2}$ 个 ${A}_{1}$ 的实例（每行一个）。如果 ${A}_{1}$ 是随机化的，我们指示所有这些实例做出相同的随机选择。${}^{2}$ 这确保：

- with probability at least $1 - {\delta }_{1}$ ,all the instances succeed simultaneously;

- 至少以 $1 - {\delta }_{1}$ 的概率，所有实例同时成功；

- for each $i \in  \left\lbrack  {1,{p}_{1}}\right\rbrack$ ,all the machines at the $i$ -th column produce exactly the same set of $i$ -tuples.

- 对于每个 $i \in  \left\lbrack  {1,{p}_{1}}\right\rbrack$，第 $i$ 列的所有机器恰好产生相同的 $i$ 元组集合。

The load incurred is $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$ . Likewise,for each column,we run ${A}_{2}$ using the ${p}_{2}$ machines on that column to $\operatorname{compute}\operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ . With probability at least $1 - {\delta }_{2}$ ,for each $j \in  \left\lbrack  {1,{p}_{2}}\right\rbrack$ , all the machines at the $j$ -th row produce exactly the same set of $j$ -tuples. The load is $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$ .

产生的负载为 $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$。同样，对于每一列，我们使用该列的 ${p}_{2}$ 台机器运行 ${A}_{2}$ 来 $\operatorname{compute}\operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$。至少以 $1 - {\delta }_{2}$ 的概率，对于每个 $j \in  \left\lbrack  {1,{p}_{2}}\right\rbrack$，第 $j$ 行的所有机器恰好产生相同的 $j$ 元组集合。负载为 $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$。

Therefore,it holds with probability at least $1 - {\delta }_{1} - {\delta }_{2}$ that,for each pair(i,j),some machine has produced all the $i$ - and $j$ -tuples. Hence,every tuple of $\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ appears on a machine. The overall load is the larger between $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$ and $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$ .

因此，至少以 $1 - {\delta }_{1} - {\delta }_{2}$ 的概率成立，即对于每一对 (i, j)，某台机器产生了所有的 $i$ 元组和 $j$ 元组。因此，$\operatorname{Join}\left( {\mathcal{Q}}_{1}\right)  \times  \operatorname{Join}\left( {\mathcal{Q}}_{2}\right)$ 的每个元组都出现在一台机器上。总负载是 $\widetilde{O}\left( {m/{p}_{1}^{1/{t}_{1}}}\right)$ 和 $\widetilde{O}\left( {m/{p}_{2}^{1/{t}_{2}}}\right)$ 中的较大值。

Skew-Free Queries. It is possible to solve a join query $\mathcal{Q}$ on binary relations in a single round with a small load if no value appears too often. To explain,denote by $m$ the input size of $\mathcal{Q}$ ; set $k = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$ ,and list out the attributes in attset(Q)as ${X}_{1},\ldots ,{X}_{k}$ . For $i \in  \left\lbrack  {1,k}\right\rbrack$ ,let ${p}_{i}$ be a positive integer referred to as the share of ${X}_{i}$ . A relation $R \in  \mathcal{Q}$ with scheme $\left\{  {{X}_{i},{X}_{j}}\right\}$ is skew-free if every value $x \in$ dom fulfills both conditions below:

无倾斜查询。如果没有值出现得过于频繁，就有可能以较小的负载在一轮中解决二元关系上的连接查询 $\mathcal{Q}$。为了解释这一点，用 $m$ 表示 $\mathcal{Q}$ 的输入大小；设 $k = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$，并将 attset(Q) 中的属性列为 ${X}_{1},\ldots ,{X}_{k}$。对于 $i \in  \left\lbrack  {1,k}\right\rbrack$，设 ${p}_{i}$ 为一个正整数，称为 ${X}_{i}$ 的份额。一个具有模式 $\left\{  {{X}_{i},{X}_{j}}\right\}$ 的关系 $R \in  \mathcal{Q}$ 是无倾斜的，如果每个值 $x \in$ dom 都满足以下两个条件：

- $R$ has $O\left( {m/{p}_{i}}\right)$ tuples $\mathbf{u}$ with $\mathbf{u}\left( {X}_{i}\right)  = x$ ;

- $R$ 有 $O\left( {m/{p}_{i}}\right)$ 个元组 $\mathbf{u}$，其中 $\mathbf{u}\left( {X}_{i}\right)  = x$；

- $R$ has $O\left( {m/{p}_{j}}\right)$ tuples $\mathbf{u}$ with $\mathbf{u}\left( {X}_{j}\right)  = x$ .

- $R$ 有 $O\left( {m/{p}_{j}}\right)$ 个元组 $\mathbf{u}$，其中 $\mathbf{u}\left( {X}_{j}\right)  = x$。

Define share $\left( R\right)  = {p}_{i} \cdot  {p}_{j}$ . If every $R \in  \mathcal{Q}$ is skew-free, $\mathcal{Q}$ is skew-free. We know:

定义份额 $\left( R\right)  = {p}_{i} \cdot  {p}_{j}$。如果每个 $R \in  \mathcal{Q}$ 都是无偏斜的，那么 $\mathcal{Q}$ 也是无偏斜的。我们知道：

---

<!-- Footnote -->

${}^{2}$ The random choices of an algorithm can be modeled as a sequence of random bits. Once the sequence is fixed,a randomized algorithm becomes deterministic. An easy way to "instruct" all instances of ${A}_{1}$ to make the same random choices is to ask all the participating machines to pre-agree on the random-bit sequence. For example, one machine can generate all the random bits and send them to the other machines. Such communication happens before receiving $\mathcal{Q}$ and hence does not contribute to the query’s load. The above approach works for a single $\mathcal{Q}$ (which suffices for proving Lemma 4). There is a standard technique [15] to extend the approach to work for any number of queries. The main idea is to have the machines pre-agree on a sufficiently large number of random-bit sequences. Given a query, a machine randomly picks a specific random-bit sequence and broadcasts the sequence's id (note: only the id, not the sequence itself) to all machines. As shown in [15],such an id can be encoded in $\widetilde{O}\left( 1\right)$ words. Broadcasting can be done in constant rounds with load $O\left( {p}^{\epsilon }\right)$ for an arbitrarily small constant $\epsilon  > 0$ .

${}^{2}$ 算法的随机选择可以建模为一个随机比特序列。一旦该序列确定，随机算法就变成了确定性算法。一种“指示”所有 ${A}_{1}$ 实例做出相同随机选择的简单方法是要求所有参与的机器预先就随机比特序列达成一致。例如，一台机器可以生成所有随机比特并将它们发送给其他机器。这种通信发生在接收 $\mathcal{Q}$ 之前，因此不会增加查询的负载。上述方法适用于单个 $\mathcal{Q}$（这足以证明引理 4）。有一种标准技术 [15] 可以将该方法扩展到适用于任意数量的查询。主要思想是让机器预先就足够多的随机比特序列达成一致。给定一个查询，一台机器随机选择一个特定的随机比特序列，并将该序列的 ID（注意：只是 ID，而不是序列本身）广播给所有机器。如 [15] 所示，这样的 ID 可以用 $\widetilde{O}\left( 1\right)$ 个字进行编码。对于任意小的常数 $\epsilon  > 0$，广播可以在常数轮内以负载 $O\left( {p}^{\epsilon }\right)$ 完成。

<!-- Footnote -->

---

Lemma 5 [6]. With probability at least $1 - 1/{p}^{c}$ where $p = \mathop{\prod }\limits_{{i = 1}}^{k}{p}_{i}$ and $c \geq  1$ can be set to an arbitrarily large constant,a skew-free query $\mathcal{Q}$ with input size $m$ can be answered in one round with load $O\left( {m/\mathop{\min }\limits_{{R \in  \mathcal{Q}}}\operatorname{share}\left( R\right) }\right)$ using $p$ machines.

引理 5 [6]。在概率至少为 $1 - 1/{p}^{c}$ 的情况下，其中 $p = \mathop{\prod }\limits_{{i = 1}}^{k}{p}_{i}$ 和 $c \geq  1$ 可以设置为任意大的常数，一个输入大小为 $m$ 的无偏斜查询 $\mathcal{Q}$ 可以使用 $p$ 台机器在一轮内以负载 $O\left( {m/\mathop{\min }\limits_{{R \in  \mathcal{Q}}}\operatorname{share}\left( R\right) }\right)$ 得到回答。

### 4.A TAXONOMY OF THE JOIN RESULT

### 4. 连接结果的分类

Given a simple binary join $\mathcal{Q}$ ,we will present a method to partition $\operatorname{Join}\left( \mathcal{Q}\right)$ based on the value frequencies in the relations of $\mathcal{Q}$ . Denote by $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ the hypergraph defined by $\mathcal{Q}$ and by $m$ the input size of $\mathcal{Q}$ .

给定一个简单的二元连接 $\mathcal{Q}$，我们将提出一种基于 $\mathcal{Q}$ 中关系的值频率对 $\operatorname{Join}\left( \mathcal{Q}\right)$ 进行划分的方法。用 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ 表示由 $\mathcal{Q}$ 定义的超图，用 $m$ 表示 $\mathcal{Q}$ 的输入大小。

Heavy and Light Values. Fix an arbitrary integer $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$ . A value $x \in  \mathbf{{dom}}$ is

重值和轻值。固定一个任意整数 $\lambda  \in  \left\lbrack  {1,m}\right\rbrack$。一个值 $x \in  \mathbf{{dom}}$ 是

- heavy if $\left| \left\{  {\mathbf{u} \in  R \mid  \mathbf{u}\left( X\right)  = x}\right\}  \right|  \geq  m/\lambda$ for some relation $R \in  \mathcal{Q}$ and some attribute $X \in$ scheme(R);

- 重值，如果对于某个关系 $R \in  \mathcal{Q}$ 和某个属性 $X \in$ 方案(R)，有 $\left| \left\{  {\mathbf{u} \in  R \mid  \mathbf{u}\left( X\right)  = x}\right\}  \right|  \geq  m/\lambda$；

- light if $x$ is not heavy,but appears in at least one relation $R \in  \mathcal{Q}$ .

- 轻值，如果 $x$ 不是重值，但至少出现在一个关系 $R \in  \mathcal{Q}$ 中。

It is easy to see that each attribute has at most $\lambda$ heavy values. Hence,the total number of heavy values is at most $\lambda  \cdot  \left| {\text{attset}\left( \mathcal{Q}\right) }\right|  = O\left( \lambda \right)$ . We will refer to $\lambda$ as the heavy parameter.

很容易看出，每个属性最多有 $\lambda$ 个重值。因此，重值的总数最多为 $\lambda  \cdot  \left| {\text{attset}\left( \mathcal{Q}\right) }\right|  = O\left( \lambda \right)$。我们将 $\lambda$ 称为重参数。

Configurations. Let $\mathcal{H}$ be an arbitrary (possibly empty) subset of attset(Q). A configuration of $\mathcal{H}$ is a tuple $\mathbf{\eta }$ over $\mathcal{H}$ such that $\mathbf{\eta }\left( X\right)$ is heavy for every $X \in  \mathcal{H}$ . Let $\operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ be the set of all configurations of $\mathcal{H}$ . It is clear that $\left| {\operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }\right|  = O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)$ .

配置。设 $\mathcal{H}$ 为 attset(Q) 的任意（可能为空）子集。$\mathcal{H}$ 的一个配置是一个基于 $\mathcal{H}$ 的元组 $\mathbf{\eta }$，使得对于每个 $X \in  \mathcal{H}$，$\mathbf{\eta }\left( X\right)$ 都是重的。设 $\operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ 为 $\mathcal{H}$ 的所有配置的集合。显然有 $\left| {\operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }\right|  = O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)$。

Residual Relations/Queries. Consider an edge $e \in  \mathcal{E}$ ; define ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ . We say that $e$ is active on $\mathcal{H}$ if ${e}^{\prime } \neq  \varnothing$ ,i.e., $e$ has at least one attribute outside $\mathcal{H}$ . An active $e$ defines a residual relation under $\mathbf{\eta } -$ denoted as ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)  -$ which

剩余关系/查询。考虑一条边 $e \in  \mathcal{E}$；定义 ${e}^{\prime } = e \smallsetminus  \mathcal{H}$。若 ${e}^{\prime } \neq  \varnothing$，即 $e$ 至少有一个属性不在 $\mathcal{H}$ 中，我们称 $e$ 在 $\mathcal{H}$ 上是活跃的。一个活跃的 $e$ 在 $\mathbf{\eta } -$ 下定义了一个剩余关系，记为 ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)  -$，该关系

- is over ${e}^{\prime }$ and

- 基于 ${e}^{\prime }$ 且

- consists of every tuple $\mathbf{v}$ that is the projection (on ${e}^{\prime }$ ) of some tuple $\mathbf{w} \in  {R}_{e}$ "consistent" with $\mathbf{\eta }$ ,namely:

- 由每个元组 $\mathbf{v}$ 组成，该元组是某个与 $\mathbf{\eta }$ “一致”的元组 $\mathbf{w} \in  {R}_{e}$ 在 ${e}^{\prime }$ 上的投影，即：

$- \mathbf{w}\left( X\right)  = \mathbf{\eta }\left( X\right)$ for every $X \in  e \cap  \mathcal{H}$ ;

对于每个 $X \in  e \cap  \mathcal{H}$，有 $- \mathbf{w}\left( X\right)  = \mathbf{\eta }\left( X\right)$；

$- \mathbf{w}\left( Y\right)$ is light for every $Y \in  {e}^{\prime }$ ;

对于每个 $Y \in  {e}^{\prime }$，$- \mathbf{w}\left( Y\right)$ 是轻的；

$- v = w\left\lbrack  {e}^{\prime }\right\rbrack  .$

The residual query under $\mathbf{\eta }$ is

在 $\mathbf{\eta }$ 下的剩余查询是

$$
{\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{e}^{\prime }\left( \mathbf{\eta }\right)  \mid  e \in  \mathcal{E},e\text{ active on }\mathcal{H}}\right\}  . \tag{4.1}
$$

Note that if $\mathcal{H} = \operatorname{attset}\left( \mathcal{Q}\right) ,{\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ is empty.

注意，如果 $\mathcal{H} = \operatorname{attset}\left( \mathcal{Q}\right) ,{\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ 为空。

Example. Consider the query $\mathcal{Q}$ in Section 1.3 (hypergraph $\mathcal{G}$ in Figure 1a) and the configuration $\mathbf{\eta }$ of $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ where $\mathbf{\eta }\left\lbrack  \mathrm{D}\right\rbrack   = \mathrm{d},\mathbf{\eta }\left\lbrack  \mathrm{E}\right\rbrack   = \mathbf{e},\mathbf{\eta }\left\lbrack  \mathrm{F}\right\rbrack   = \mathbf{f}$ ,and $\mathbf{\eta }\left\lbrack  \mathrm{K}\right\rbrack   = \mathrm{k}$ . If $e$ is the edge $\{ \mathtt{A},\mathtt{D}\}$ ,then ${e}^{\prime } = \{ \mathtt{A}\}$ and ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ is the relation ${R}_{\{ \mathtt{A}\}  \mid  \mathtt{d}}^{\prime }$ mentioned in Section 1.3. If $e$ is the edge $\{ \mathrm{A},\mathrm{B}\}$ ,then ${e}^{\prime } = \{ \mathrm{A},\mathrm{B}\}$ and ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ is the relation ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ in Section 1.3. The residual query ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ is precisely the query ${\mathcal{Q}}^{\prime }$ in Section 1.3.

示例。考虑第1.3节中的查询$\mathcal{Q}$（图1a中的超图$\mathcal{G}$）以及$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$的配置$\mathbf{\eta }$，其中$\mathbf{\eta }\left\lbrack  \mathrm{D}\right\rbrack   = \mathrm{d},\mathbf{\eta }\left\lbrack  \mathrm{E}\right\rbrack   = \mathbf{e},\mathbf{\eta }\left\lbrack  \mathrm{F}\right\rbrack   = \mathbf{f}$，且$\mathbf{\eta }\left\lbrack  \mathrm{K}\right\rbrack   = \mathrm{k}$。如果$e$是边$\{ \mathtt{A},\mathtt{D}\}$，那么${e}^{\prime } = \{ \mathtt{A}\}$，并且${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$是第1.3节中提到的关系${R}_{\{ \mathtt{A}\}  \mid  \mathtt{d}}^{\prime }$。如果$e$是边$\{ \mathrm{A},\mathrm{B}\}$，那么${e}^{\prime } = \{ \mathrm{A},\mathrm{B}\}$，并且${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$是第1.3节中的关系${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$。剩余查询${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$恰好是第1.3节中的查询${\mathcal{Q}}^{\prime }$。

It is rudimentary to verify

验证这一点是基本的

$$
\operatorname{Join}\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{\mathcal{H}}\left( {\mathop{\bigcup }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\operatorname{Join}\left( {{\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right) }\right) \times \{ \mathbf{\eta }\} }\right) . \tag{4.2}
$$

Lemma 6 . Let $\mathcal{Q}$ be a simple binary join with input size $m$ and $\mathcal{H}$ be a subset of attset(Q). For each configuration $\eta  \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ ,denote by ${m}_{\eta }$ the total size of all the relations in ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ . We have:

引理6。设$\mathcal{Q}$是一个输入规模为$m$的简单二元连接，$\mathcal{H}$是attset(Q)的一个子集。对于每个配置$\eta  \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$，用${m}_{\eta }$表示${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$中所有关系的总规模。我们有：

$$
\mathop{\sum }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}{m}_{\mathbf{\eta }} \leq  m \cdot  {\lambda }^{k - 2}
$$

where $k = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$ .

其中$k = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$。

Proof. Let $e$ be an edge in $\mathcal{E}$ and fix an arbitrary tuple $\mathbf{u} \in  {R}_{e}$ . Tuple $\mathbf{u}$ contributes 1 to the term ${m}_{\mathbf{\eta }}$ only if $\mathbf{\eta }\left( X\right)  = \mathbf{u}\left( X\right)$ for every attribute $X \in  e \cap  \mathcal{H}$ . How many such configurations $\mathbf{\eta }$ can there be? As these configurations must have the same value on every attribute in $e \cap  \mathcal{H}$ ,they can differ only in the attributes of $\mathcal{H} \smallsetminus  e$ . Since each attribute has at most $\lambda$ heavy values,we conclude that the number of those configurations $\eta$ is at most ${\lambda }^{\left| \mathcal{H} \smallsetminus  e\right| }.\left| {\mathcal{H} \smallsetminus  e}\right|$ is at most $k - 2$ because $\left| \mathcal{H}\right|  \leq  k$ and $e$ has two attributes. The lemma thus follows.

证明。设$e$是$\mathcal{E}$中的一条边，并固定一个任意元组$\mathbf{u} \in  {R}_{e}$。仅当对于每个属性$X \in  e \cap  \mathcal{H}$都有$\mathbf{\eta }\left( X\right)  = \mathbf{u}\left( X\right)$时，元组$\mathbf{u}$才会对项${m}_{\mathbf{\eta }}$贡献1。可能存在多少个这样的配置$\mathbf{\eta }$呢？由于这些配置在$e \cap  \mathcal{H}$中的每个属性上必须具有相同的值，它们只能在$\mathcal{H} \smallsetminus  e$的属性上有所不同。由于每个属性最多有$\lambda$个重值，我们得出这些配置$\eta$的数量最多为${\lambda }^{\left| \mathcal{H} \smallsetminus  e\right| }.\left| {\mathcal{H} \smallsetminus  e}\right|$，最多为$k - 2$，因为$\left| \mathcal{H}\right|  \leq  k$且$e$有两个属性。因此，引理得证。

### 5.A JOIN COMPUTATION FRAMEWORK

### 5.A 连接计算框架

Answering a simple binary join $\mathcal{Q}$ amounts to producing the right-hand side of (4.2). Due to symmetry,it suffices to explain how to do so for an arbitrary subset $\mathcal{H} \subseteq  \operatorname{attset}\left( \mathcal{Q}\right)$ ,i.e., the computation of

回答一个简单的二元连接$\mathcal{Q}$相当于得出(4.2)式的右侧。由于对称性，只需解释如何对任意子集$\mathcal{H} \subseteq  \operatorname{attset}\left( \mathcal{Q}\right)$进行此操作，即计算

$$
\mathop{\bigcup }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\operatorname{Join}\left( {{\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right) }\right) . \tag{5.1}
$$

At a high level,our strategy (illustrated in Section 1.3) works as follows. Let $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ be the hypergraph defined by $\mathcal{Q}$ . We will remove the vertices in $\mathcal{H}$ from $\mathcal{G}$ ,which disconnects $\mathcal{G}$ into connected components (CCs). We divide the CCs into two groups: (i) the set of CCs each involving at least 2 vertices, and (ii) the set of all other CCs, namely those containing only 1 vertex. We will process the CCs in Group 1 together using Lemma 5, process the CCs in Group 2 together using Lemma 3, and then compute the cartesian product between Groups 1 and 2 using Lemma 4.

从高层次来看，我们的策略（在1.3节中说明）如下。设$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$为由$\mathcal{Q}$定义的超图。我们将从$\mathcal{G}$中移除$\mathcal{H}$中的顶点，这会将$\mathcal{G}$拆分为连通分量（CC）。我们将这些连通分量分为两组：（i）每个至少包含2个顶点的连通分量集合，以及（ii）所有其他连通分量的集合，即那些仅包含1个顶点的连通分量。我们将使用引理5一起处理第1组的连通分量，使用引理3一起处理第2组的连通分量，然后使用引理4计算第1组和第2组之间的笛卡尔积。

Sections 5.1 and 5.2 will formalize the strategy into a processing framework. Sections 5.3 and 5.4 will then establish two important properties of this framework, which are the key to its efficient implementation in Section 6.

5.1节和5.2节将把该策略形式化为一个处理框架。5.3节和5.4节将确立该框架的两个重要属性，这是在第6节中对其进行高效实现的关键。

5.1. Removing the Attributes in $\mathcal{H}$ . We will refer to each attribute in $\mathcal{H}$ as a heavy attribute. Define

5.1. 移除$\mathcal{H}$中的属性。我们将$\mathcal{H}$中的每个属性称为重属性。定义

$$
\mathcal{L} = \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H} \tag{5.2}
$$

and call each attribute in $\mathcal{L}$ a light attribute. An edge $e \in  \mathcal{E}$ is

并将$\mathcal{L}$中的每个属性称为轻属性。一条边$e \in  \mathcal{E}$是

- a light edge if $e$ contains two light attributes,or

- 如果$e$包含两个轻属性，则为轻边；或者

- a cross edge if $e$ contains a heavy attribute and a light attribute.

- 如果$e$包含一个重属性和一个轻属性，则为交叉边。

A light attribute $X \in  \mathcal{L}$ is a border attribute if it appears in at least one cross edge $e$ of $\mathcal{G}$ . Denote by ${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$ the subgraph of $\mathcal{G}$ induced by $\mathcal{L}$ . A vertex $X \in  \mathcal{L}$ is isolated if $\{ X\}$ is the only edge in ${\mathcal{E}}^{\prime }$ incident to $X$ . Define

如果一个轻属性$X \in  \mathcal{L}$出现在$\mathcal{G}$的至少一条交叉边$e$中，则它是一个边界属性。用${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$表示由$\mathcal{L}$诱导的$\mathcal{G}$的子图。如果$\{ X\}$是${\mathcal{E}}^{\prime }$中与$X$关联的唯一边，则顶点$X \in  \mathcal{L}$是孤立的。定义

$$
\mathcal{I} = \{ X \in  \mathcal{L} \mid  X\text{ is isolated }\} . \tag{5.3}
$$

<!-- Media -->

<!-- figureText: On -->

<img src="https://cdn.noedgeai.com/0195ccbd-9d97-7aa3-8943-c43bdb45848f_12.jpg?x=731&y=321&w=322&h=315&r=0"/>

Figure 2: Subgraph induced by $\mathcal{L}$

图2：由$\mathcal{L}$诱导的子图

<!-- Media -->

Example (cont.). Consider again the join query $\mathcal{Q}$ whose hypergraph $\mathcal{G}$ is shown in Figure 1a. Set $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ . Set $\mathcal{L}$ includes all the white vertices in Figure 1b. Edge $\{ \mathrm{A},\mathrm{B}\}$ is a light edge, $\{ \mathrm{A},\mathrm{D}\}$ is a cross edge,while $\{ \mathrm{D},\mathrm{K}\}$ is neither a light edge nor a cross edge. All the vertices in $\mathcal{L}$ except $\mathrm{J}$ are border vertices. Figure 2 shows the subgraph of $\mathcal{G}$ induced by $\mathcal{L}$ ,where a unary edge is represented by a box and a binary edge by a segment. The isolated vertices are $\mathrm{G},\mathrm{H}$ ,and $\mathrm{L}$ .

示例（续）。再次考虑连接查询$\mathcal{Q}$，其超图$\mathcal{G}$如图1a所示。设$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$。集合$\mathcal{L}$包括图1b中所有的白色顶点。边$\{ \mathrm{A},\mathrm{B}\}$是轻边，$\{ \mathrm{A},\mathrm{D}\}$是交叉边，而$\{ \mathrm{D},\mathrm{K}\}$既不是轻边也不是交叉边。除$\mathrm{J}$之外，$\mathcal{L}$中的所有顶点都是边界顶点。图2展示了由$\mathcal{L}$诱导的$\mathcal{G}$的子图，其中一元边用方框表示，二元边用线段表示。孤立顶点是$\mathrm{G},\mathrm{H}$和$\mathrm{L}$。

5.2. Semi-Join Reduction. Recall from Section 4 that every configuration $\mathbf{\eta }$ of $\mathcal{H}$ defines a residual query ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ . Next,we will simplify ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ into a join ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ with the same result.

5.2. 半连接约简。回顾第4节，$\mathcal{H}$的每个配置$\mathbf{\eta }$都定义了一个剩余查询${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$。接下来，我们将把${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$简化为一个具有相同结果的连接${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$。

Observe that the hypergraph defined by ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ is always ${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$ ,regardless of $\mathbf{\eta }$ . Consider a border attribute $X \in  \mathcal{L}$ and a cross edge $e$ of $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ incident to $X$ . As explained in Section 4,the input relation ${R}_{e} \in  \mathcal{Q}$ defines a unary residual relation ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)  \in  {\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ . Note that ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ has scheme $\{ X\}$ . We define:

观察到，由${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$定义的超图始终是${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$，与$\mathbf{\eta }$无关。考虑一个边界属性$X \in  \mathcal{L}$以及$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$中与$X$相关联的一条交叉边$e$。如第4节所述，输入关系${R}_{e} \in  \mathcal{Q}$定义了一个一元残差关系${R}_{e}^{\prime }\left( \mathbf{\eta }\right)  \in  {\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$。注意，${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$具有模式$\{ X\}$。我们定义：

$$
{R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)  = \mathop{\bigcap }\limits_{{\text{cross edge }e \in  \mathcal{E}\text{ s.t. }X \in  e}}{R}_{e}^{\prime }\left( \mathbf{\eta }\right) . \tag{5.4}
$$

Example (cont.). Let $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ ,and consider its configuration $\mathbf{\eta }$ with $\mathbf{\eta }\left( \mathrm{D}\right)  = \mathrm{d}$ , $\mathbf{\eta }\left( \mathrm{E}\right)  = \mathrm{e},\mathbf{\eta }\left( \mathrm{F}\right)  = \mathrm{f}$ ,and $\mathbf{\eta }\left( \mathrm{K}\right)  = \mathrm{k}$ . Set $X$ to the border attribute $\mathrm{A}$ . When $e$ is $\{ \mathrm{A},\mathrm{D}\}$ or $\{ \mathrm{A},\mathrm{E}\} ,{R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ is the relation ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$ or ${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ mentioned in Section 1.3,respectively. $\{ \mathrm{A},\mathrm{D}\}$ and $\{ \mathrm{A},\mathrm{E}\}$ are the only cross edges containing $\mathrm{A}$ . Hence, ${R}_{A}^{\prime \prime }\left( \mathbf{\eta }\right)  = {R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime } \cap  {R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$ ,which is the relation ${R}_{\{ A\} }^{\prime \prime }$ in Section 1.3.

示例（续）。设$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$，并考虑其配置$\mathbf{\eta }$，其中$\mathbf{\eta }\left( \mathrm{D}\right)  = \mathrm{d}$、$\mathbf{\eta }\left( \mathrm{E}\right)  = \mathrm{e},\mathbf{\eta }\left( \mathrm{F}\right)  = \mathrm{f}$和$\mathbf{\eta }\left( \mathrm{K}\right)  = \mathrm{k}$。将$X$设为边界属性$\mathrm{A}$。当$e$分别为$\{ \mathrm{A},\mathrm{D}\}$或$\{ \mathrm{A},\mathrm{E}\} ,{R}_{e}^{\prime }\left( \mathbf{\eta }\right)$分别为第1.3节中提到的关系${R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime }$或${R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$时。$\{ \mathrm{A},\mathrm{D}\}$和$\{ \mathrm{A},\mathrm{E}\}$是仅有的包含$\mathrm{A}$的交叉边。因此，${R}_{A}^{\prime \prime }\left( \mathbf{\eta }\right)  = {R}_{\{ \mathrm{A}\}  \mid  \mathrm{d}}^{\prime } \cap  {R}_{\{ \mathrm{A}\}  \mid  \mathrm{e}}^{\prime }$，即第1.3节中的关系${R}_{\{ A\} }^{\prime \prime }$。

Recall that every light edge $e = \{ X,Y\}$ in $\mathcal{G}$ defines a residual relation ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ with scheme $e$ . We define ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)$ as a relation over $e$ that contains every tuple $\mathbf{u} \in  {R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ satisfying:

回顾一下，$\mathcal{G}$ 中的每条轻边 $e = \{ X,Y\}$ 都定义了一个具有模式 $e$ 的残差关系 ${R}_{e}^{\prime }\left( \mathbf{\eta }\right)$。我们将 ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)$ 定义为一个基于 $e$ 的关系，它包含每个满足以下条件的元组 $\mathbf{u} \in  {R}_{e}^{\prime }\left( \mathbf{\eta }\right)$：

- (applicable only if $X$ is a border attribute) $\mathbf{u}\left( X\right)  \in  {R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$ ;

- （仅当 $X$ 是边界属性时适用）$\mathbf{u}\left( X\right)  \in  {R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$；

- (applicable only if $Y$ is a border attribute) $\mathbf{u}\left( Y\right)  \in  {R}_{Y}^{\prime \prime }\left( \mathbf{\eta }\right)$ .

- （仅当 $Y$ 是边界属性时适用）$\mathbf{u}\left( Y\right)  \in  {R}_{Y}^{\prime \prime }\left( \mathbf{\eta }\right)$。

Note that if neither $X$ nor $Y$ is a border attribute,then ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)  = {R}_{e}^{\prime }\left( \mathbf{\eta }\right)$ .

注意，如果 $X$ 和 $Y$ 都不是边界属性，那么 ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)  = {R}_{e}^{\prime }\left( \mathbf{\eta }\right)$。

Example (cont.). For the light edge $e = \{ \mathrm{A},\mathrm{B}\} ,{R}_{e}^{\prime }\left( \eta \right)$ is the relation ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ mentioned in Section 1.3. Because A and B are border attributes, ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)$ includes all the tuples in ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ that take an A-value from ${R}_{\mathrm{A}}^{\prime \prime }\left( \mathbf{\eta }\right)$ and a B-value from ${R}_{\mathrm{B}}^{\prime \prime }\left( \mathbf{\eta }\right)$ . This ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)$ is precisely the relation ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$ in Section 1.3.

示例（续）。对于轻边 $e = \{ \mathrm{A},\mathrm{B}\} ,{R}_{e}^{\prime }\left( \eta \right)$，它是第 1.3 节中提到的关系 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$。因为 A 和 B 是边界属性，${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)$ 包含 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime }$ 中所有从 ${R}_{\mathrm{A}}^{\prime \prime }\left( \mathbf{\eta }\right)$ 中获取 A 值且从 ${R}_{\mathrm{B}}^{\prime \prime }\left( \mathbf{\eta }\right)$ 中获取 B 值的元组。这个 ${R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)$ 恰好就是第 1.3 节中的关系 ${R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime }$。

Every vertex $X \in  \mathcal{I}$ must be a border attribute and,thus,must now be associated with ${R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$ . We can legally define:

每个顶点 $X \in  \mathcal{I}$ 必须是边界属性，因此，现在必须与 ${R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$ 相关联。我们可以合法地定义：

$$
{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)  \mid  X \in  \mathcal{I}}\right\}   \tag{5.5}
$$

$$
{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{e}^{\prime \prime }\left( \mathbf{\eta }\right)  \mid  \text{ light edge }e \in  \mathcal{E}}\right\}   \tag{5.6}
$$

$$
{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)  = {\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)  \cup  {\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) . \tag{5.7}
$$

Example (cont.). ${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{\{ \mathbf{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime },{R}_{\{ \mathrm{L}\} }^{\prime \prime }}\right\}$ and ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime \prime }}\right.$ ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }\}$ ,where all the relation names follow those in Section 1.3.

示例（续）。${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{\{ \mathbf{G}\} }^{\prime \prime },{R}_{\{ \mathrm{H}\} }^{\prime \prime },{R}_{\{ \mathrm{L}\} }^{\prime \prime }}\right\}$ 和 ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{\{ \mathrm{A},\mathrm{B}\} }^{\prime \prime },{R}_{\{ \mathrm{A},\mathrm{C}\} }^{\prime \prime },{R}_{\{ \mathrm{B},\mathrm{C}\} }^{\prime \prime }}\right.$ ${R}_{\{ \mathrm{I},\mathrm{J}\} }^{\prime \prime }\}$，其中所有关系名遵循第 1.3 节中的命名。

We will refer to the conversion from ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ to ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ as semi-join reduction and call ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ the reduced query under $\mathbf{\eta }$ . It is rudimentary to verify:

我们将从 ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ 到 ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ 的转换称为半连接约简，并将 ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ 称为在 $\mathbf{\eta }$ 下的约简查询。验证以下内容是很基础的：

$$
\operatorname{Join}\left( {{\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right) }\right)  = \operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  = \operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  \times  \operatorname{Join}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) . \tag{5.8}
$$

5.3. The Isolated Cartesian Product Theorem. As shown in (5.5), ${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)$ contains $\left| \mathcal{I}\right|$ unary relations,one for each isolated attribute in $\mathcal{I}$ . Hence, $\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ is the cartesian product of all those relations. The size of $\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ has a crucial impact on the efficiency of our join strategy because, as shown in Lemma 3, the load for computing a cartesian product depends on the cartesian product's size. To prove that our strategy is efficient, we want to argue that

5.3. 孤立笛卡尔积定理。如(5.5)所示，${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)$包含$\left| \mathcal{I}\right|$个一元关系，每个关系对应$\mathcal{I}$中的一个孤立属性。因此，$\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$是所有这些关系的笛卡尔积。$\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$的大小对我们的连接策略的效率有着至关重要的影响，因为正如引理3所示，计算笛卡尔积的负载取决于笛卡尔积的大小。为了证明我们的策略是高效的，我们需要论证

$$
\mathop{\sum }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \tag{5.9}
$$

is low,namely,the cartesian products of all the configurations $\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ have a small size overall.

是低的，即所有配置$\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$的笛卡尔积总体规模较小。

It is easy to place an upper bound of ${\lambda }^{\left| \mathcal{H}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }$ on (5.9). As each relation (trivially) has size at most $m$ ,we have $\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \leq  {m}^{\left| \mathcal{I}\right| }$ . Given that $\mathcal{H}$ has at most ${\lambda }^{\left| \mathcal{H}\right| }$ different configurations,(5.9) is at most ${\lambda }^{\left| \mathcal{H}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }$ . Unfortunately,the bound is not enough to establish the claimed performance of our MPC algorithm (to be presented in Section 6). For that purpose, we will need to prove a tighter upper bound on (5.9) - this is where the isolated cartesian product theorem (described next) comes in.

很容易为(5.9)设定一个上界${\lambda }^{\left| \mathcal{H}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }$。由于每个关系（显然）的大小至多为$m$，我们有$\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \leq  {m}^{\left| \mathcal{I}\right| }$。鉴于$\mathcal{H}$至多有${\lambda }^{\left| \mathcal{H}\right| }$种不同的配置，(5.9)至多为${\lambda }^{\left| \mathcal{H}\right| } \cdot  {m}^{\left| \mathcal{I}\right| }$。不幸的是，这个界不足以确立我们的MPC算法（将在第6节介绍）所宣称的性能。为此，我们需要为(5.9)证明一个更紧的上界——这就是孤立笛卡尔积定理（接下来介绍）发挥作用的地方。

Given an arbitrary fractional edge packing $W$ of the hypergraph $\mathcal{G}$ ,we define

给定超图$\mathcal{G}$的任意分数边填充$W$，我们定义

$$
{W}_{\mathcal{I}} = \mathop{\sum }\limits_{{Y \in  \mathcal{I}}}\text{ weight of }Y\text{ under }W. \tag{5.10}
$$

Recall that the weight of a vertex $Y$ under $W$ is the sum of $W\left( e\right)$ for all the edges $e \in  \mathcal{E}$ containing $Y$ .

回顾一下，在$W$下顶点$Y$的权重是包含$Y$的所有边$e \in  \mathcal{E}$的$W\left( e\right)$之和。

Theorem 7 (The isolated cartesian product theorem). Let $\mathcal{Q}$ be a simple binary query whose relations have a total size of $m$ . Denote by $\mathcal{G}$ the hypergraph defined by $\mathcal{Q}$ . Consider an arbitrary subset $\mathcal{H} \subseteq$ attset(Q),where attset(Q)is the set of attributes in the relations of $\mathcal{Q}$ . Let $\mathcal{I}$ be the set of isolated vertices defined in (5.3). Take an arbitrary fractional edge packing $W$ of $\mathcal{G}$ . It holds that

定理7（孤立笛卡尔积定理）。设$\mathcal{Q}$是一个简单的二元查询，其关系的总大小为$m$。用$\mathcal{G}$表示由$\mathcal{Q}$定义的超图。考虑任意子集$\mathcal{H} \subseteq$⊆attset(Q)，其中attset(Q)是$\mathcal{Q}$的关系中的属性集。设$\mathcal{I}$是(5.3)中定义的孤立顶点集。取$\mathcal{G}$的任意分数边填充$W$。则有

$$
\mathop{\sum }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \leq  {\lambda }^{\left| \mathcal{H}\right|  - {W}_{\mathcal{I}}} \cdot  {m}^{\left| \mathcal{I}\right| } \tag{5.11}
$$

where $\lambda$ is the heavy parameter (Section 4),config(Q,H)is the set of configurations of $\mathcal{H}$ (Section 4), ${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)$ is defined in (5.5),and ${W}_{\mathcal{I}}$ is defined in (5.10).

其中$\lambda$是重参数（第4节），config(Q,H)是$\mathcal{H}$的配置集（第4节），${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)$在(5.5)中定义，${W}_{\mathcal{I}}$在(5.10)中定义。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195ccbd-9d97-7aa3-8943-c43bdb45848f_14.jpg?x=718&y=321&w=345&h=315&r=0"/>

Figure 3: Illustration of ${\mathcal{Q}}^{ * }$

图3：${\mathcal{Q}}^{ * }$的图示

<!-- Media -->

Theorem 7 is in the strongest form when ${W}_{\mathcal{I}}$ is maximized. Later in Section 5.5,we will choose a specific $W$ that yields a bound sufficient for us to prove the efficiency claim on our join algorithm.

当${W}_{\mathcal{I}}$取最大值时，定理7的形式最强。在后面的5.5节中，我们将选择一个特定的$W$，它能给出一个足以证明我们的连接算法效率声明的界。

Proof of Theorem 7. We will construct a set ${\mathcal{Q}}^{ * }$ of relations such that $\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ has a result size at least the left-hand side of (5.11). Then, we will prove that the hypergraph of ${\mathcal{Q}}^{ * }$ has a fractional edge covering that (by the AGM bound; Lemma 2) implies an upper bound on $\left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$ matching the right-hand side of (5.11).

定理7的证明。我们将构建一个关系集合${\mathcal{Q}}^{ * }$，使得$\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$的结果大小至少为(5.11)式的左侧。然后，我们将证明${\mathcal{Q}}^{ * }$的超图具有一个分数边覆盖，根据算术 - 几何平均不等式（AGM bound；引理2），这意味着$\left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$的一个上界与(5.11)式的右侧相匹配。

Initially,set ${\mathcal{Q}}^{ * }$ to $\varnothing$ . For every cross edge $e \in  \mathcal{E}$ incident to a vertex in $\mathcal{I}$ ,add to ${\mathcal{Q}}^{ * }$ a relation ${R}_{e}^{ * } = {R}_{e}$ . For every $X \in  \mathcal{H}$ ,add a unary relation ${R}_{\{ X\} }^{ * }$ to ${\mathcal{Q}}^{ * }$ which consists of all the heavy values on $X$ ; note that ${R}_{\{ X\} }^{ * }$ has at most $\lambda$ tuples. Finally,for every $Y \in  \mathcal{I}$ ,add a unary relation ${R}_{\{ Y\} }^{ * }$ to ${\mathcal{Q}}^{ * }$ which contains all the heavy and light values on $Y$ .

初始时，将${\mathcal{Q}}^{ * }$设为$\varnothing$。对于每条与$\mathcal{I}$中的顶点相关联的交叉边$e \in  \mathcal{E}$，向${\mathcal{Q}}^{ * }$中添加一个关系${R}_{e}^{ * } = {R}_{e}$。对于每个$X \in  \mathcal{H}$，向${\mathcal{Q}}^{ * }$中添加一个一元关系${R}_{\{ X\} }^{ * }$，该关系由$X$上的所有重值组成；注意，${R}_{\{ X\} }^{ * }$最多有$\lambda$个元组。最后，对于每个$Y \in  \mathcal{I}$，向${\mathcal{Q}}^{ * }$中添加一个一元关系${R}_{\{ Y\} }^{ * }$，该关系包含$Y$上的所有重值和轻值。

Define ${\mathcal{G}}^{ * } = \left( {{\mathcal{V}}^{ * },{\mathcal{E}}^{ * }}\right)$ as the hypergraph defined by ${\mathcal{Q}}^{ * }$ . Note that ${\mathcal{V}}^{ * } = \mathcal{I} \cup  \mathcal{H}$ ,while ${\mathcal{E}}^{ * }$ consists of all the cross edges in $\mathcal{G}$ incident to a vertex in $\mathcal{I},\left| \mathcal{H}\right|$ unary edges $\{ X\}$ for every $X \in  \mathcal{H}$ ,and $\left| \mathcal{I}\right|$ unary edges $\{ Y\}$ for every $Y \in  \mathcal{I}$ .

将${\mathcal{G}}^{ * } = \left( {{\mathcal{V}}^{ * },{\mathcal{E}}^{ * }}\right)$定义为由${\mathcal{Q}}^{ * }$所定义的超图。注意，${\mathcal{V}}^{ * } = \mathcal{I} \cup  \mathcal{H}$，而${\mathcal{E}}^{ * }$由$\mathcal{G}$中与$\mathcal{I},\left| \mathcal{H}\right|$中的顶点相关联的所有交叉边、每个$X \in  \mathcal{H}$对应的一元边$\{ X\}$以及每个$Y \in  \mathcal{I}$对应的一元边$\{ Y\}$组成。

Example (cont.). Figure 3 shows the hypergraph of the ${\mathcal{Q}}^{ * }$ constructed. As before,a box and a segment represent a unary and a binary edge,respectively. Recall that $\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$ and $\mathcal{I} = \{ \mathrm{G},\mathrm{H},\mathrm{L}\}$ .

示例（续）。图3展示了所构建的${\mathcal{Q}}^{ * }$的超图。和之前一样，一个方框和一条线段分别表示一元边和二元边。回顾$\mathcal{H} = \{ \mathrm{D},\mathrm{E},\mathrm{F},\mathrm{K}\}$和$\mathcal{I} = \{ \mathrm{G},\mathrm{H},\mathrm{L}\}$。

Lemma 8 . $\mathop{\sum }\limits_{{{\mathbf{\eta }}^{\prime } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( {\mathbf{\eta }}^{\prime }\right) }\right) }\right|  \leq  \left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$ .

引理8。$\mathop{\sum }\limits_{{{\mathbf{\eta }}^{\prime } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( {\mathbf{\eta }}^{\prime }\right) }\right) }\right|  \leq  \left| {\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right|$。

Proof. We will prove

证明。我们将证明

$$
\mathop{\bigcup }\limits_{{{\mathbf{\eta }}^{\prime } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( {\mathbf{\eta }}^{\prime }\right) }\right)  \times  \left\{  {\mathbf{\eta }}^{\prime }\right\}   \subseteq  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) . \tag{5.12}
$$

from which the lemma follows.

由此可推出该引理。

Take a tuple $\mathbf{u}$ from the left-hand side of (5.12),and set ${\mathbf{\eta }}^{\prime } = \mathbf{u}\left\lbrack  \mathcal{H}\right\rbrack$ . Based on the definition of ${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( {\mathbf{\eta }}^{\prime }\right)$ ,it is easy to verify that $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}$ for every cross edge $e \in  \mathcal{E}$ incident a vertex in $\mathcal{I}$ ; hence, $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}^{ * }$ . Furthermore, $\mathbf{u}\left( X\right)  \in  {R}_{\{ X\} }^{ * }$ for every $X \in  \mathcal{H}$ because $\mathbf{u}\left( X\right)  = {\mathbf{\eta }}^{\prime }\left( X\right)$ is a heavy value. Finally,obviously $\mathbf{u}\left( Y\right)  \in  {R}_{\{ Y\} }^{ * }$ for every $Y \in  \mathcal{I}$ . All these facts together ensure that $\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ .

从(5.12)式的左侧取一个元组$\mathbf{u}$，并设${\mathbf{\eta }}^{\prime } = \mathbf{u}\left\lbrack  \mathcal{H}\right\rbrack$。根据${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( {\mathbf{\eta }}^{\prime }\right)$的定义，很容易验证对于每一条与$\mathcal{I}$中的顶点相关联的交叉边$e \in  \mathcal{E}$，都有$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}$；因此，$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {R}_{e}^{ * }$。此外，对于每一个$X \in  \mathcal{H}$，都有$\mathbf{u}\left( X\right)  \in  {R}_{\{ X\} }^{ * }$，因为$\mathbf{u}\left( X\right)  = {\mathbf{\eta }}^{\prime }\left( X\right)$是一个较大的值。最后，显然对于每一个$Y \in  \mathcal{I}$，都有$\mathbf{u}\left( Y\right)  \in  {R}_{\{ Y\} }^{ * }$。所有这些事实共同保证了$\mathbf{u} \in  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$。

Lemma 9 . ${\mathcal{G}}^{ * }$ admits a tight fractional edge covering ${W}^{ * }$ satisfying $\mathop{\sum }\limits_{{X \in  \mathcal{H}}}{W}^{ * }\left( {\{ X\} }\right)  =$ $\left| \mathcal{H}\right|  - {W}_{\mathcal{I}}$

引理9. ${\mathcal{G}}^{ * }$存在一个满足$\mathop{\sum }\limits_{{X \in  \mathcal{H}}}{W}^{ * }\left( {\{ X\} }\right)  =$ $\left| \mathcal{H}\right|  - {W}_{\mathcal{I}}$的紧分数边覆盖${W}^{ * }$。

Proof. We will construct a desired function ${W}^{ * }$ from the fractional edge packing $W$ in Theorem 7.

证明。我们将从定理7中的分数边填充$W$构造出所需的函数${W}^{ * }$。

For every cross edge $e \in  \mathcal{E}$ incident to a vertex in $\mathcal{I}$ ,set ${W}^{ * }\left( e\right)  = W\left( e\right)$ . Every edge in $\mathcal{E}$ incident to $Y \in  \mathcal{I}$ must be a cross edge. Hence, $\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)$ is precisely the weight of $Y$ under $W$ .

对于每一条与$\mathcal{I}$中的顶点相关联的交叉边$e \in  \mathcal{E}$，设${W}^{ * }\left( e\right)  = W\left( e\right)$。$\mathcal{E}$中与$Y \in  \mathcal{I}$相关联的每一条边都必须是交叉边。因此，$\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)$恰好是$Y$在$W$下的权重。

Next,we will ensure that each attribute $Y \in  \mathcal{I}$ has a weight 1 under ${W}^{ * }$ . Since $W$ is a fractional edge packing of $\mathcal{G}$ ,it must hold that $\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}W\left( e\right)  \leq  1$ . This permits us to assign the following weight to the unary edge $\{ Y\}$ :

接下来，我们要确保每个属性$Y \in  \mathcal{I}$在${W}^{ * }$下的权重为1。由于$W$是$\mathcal{G}$的一个分数边填充，那么必然有$\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}W\left( e\right)  \leq  1$。这使我们可以为一元边$\{ Y\}$分配以下权重：

$$
{W}^{ * }\left( {\{ Y\} }\right)  = 1 - \mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}W\left( e\right) .
$$

Finally,in a similar way,we make sure that each attribute $X \in  \mathcal{H}$ has a weight 1 under ${W}^{ * }$ by assigning:

最后，以类似的方式，通过分配以下值，我们确保每个属性$X \in  \mathcal{H}$在${W}^{ * }$下的权重为1：

$$
{W}^{ * }\left( {\{ X\} }\right)  = 1 - \mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right) .
$$

This finishes the design of ${W}^{ * }$ ,which is now a tight fractional edge covering of ${\mathcal{G}}^{ * }$ .

至此，${W}^{ * }$的设计完成，它现在是${\mathcal{G}}^{ * }$的一个紧分数边覆盖。

Clearly:

显然：

$$
\mathop{\sum }\limits_{{X \in  \mathcal{H}}}{W}^{ * }\left( {\{ X\} }\right)  = \left| \mathcal{H}\right|  - \mathop{\sum }\limits_{{X \in  \mathcal{H}}}\left( {\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right) }\right) . \tag{5.13}
$$

Every binary edge $e \in  {\mathcal{E}}^{ * }$ contains a vertex in $\mathcal{H}$ and a vertex in $\mathcal{I}$ . Therefore:

每一条二元边$e \in  {\mathcal{E}}^{ * }$都包含$\mathcal{H}$中的一个顶点和$\mathcal{I}$中的一个顶点。因此：

$$
\mathop{\sum }\limits_{{X \in  \mathcal{H}}}\left( {\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : X \in  e}}W\left( e\right) }\right)  = \mathop{\sum }\limits_{{Y \in  \mathcal{I}}}\left( {\mathop{\sum }\limits_{{\text{binary }e \in  {\mathcal{E}}^{ * } : Y \in  e}}W\left( e\right) }\right)  = {W}_{\mathcal{I}}.
$$

Putting together the above equation with (5.13) completes the proof.

将上述等式与(5.13)式结合起来，证明完毕。

The AGM bound in Lemma 2 tells us that

引理2中的算术 - 几何平均（AGM）界告诉我们

$$
\operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)  \leq  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * }}}{\left| {R}_{e}^{ * }\right| }^{{W}^{ * }\left( e\right) }
$$

$$
 = \left( {\mathop{\prod }\limits_{{X \in  \mathcal{H}}}{\left| {R}_{\{ X\} }^{ * }\right| }^{{W}^{ * }\left( {\{ X\} }\right) }}\right) \left( {\mathop{\prod }\limits_{{Y \in  \mathcal{I}}}\mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * } : Y \in  e}}{\left| {R}_{e}^{ * }\right| }^{{W}^{ * }\left( e\right) }}\right) 
$$

$$
 \leq  \left( {\mathop{\prod }\limits_{{X \in  \mathcal{H}}}{\lambda }^{{W}^{ * }\left( {\{ X\} }\right) }}\right) \left( {\mathop{\prod }\limits_{{Y \in  \mathcal{I}}}\mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * } : Y \in  e}}{m}^{{W}^{ * }\left( e\right) }}\right) 
$$

$$
\text{(applying}\left| {R}_{\{ X\} }^{ * }\right|  \leq  \lambda \text{and}\left| {R}_{e}^{ * }\right|  \leq  m\text{)}
$$

$$
 \leq  {\lambda }^{\left| \mathcal{H}\right|  - {W}_{\mathcal{I}}} \cdot  {m}^{\left| \mathcal{I}\right| }
$$

$$
\text{(by Lemma 9 and}\mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * } : Y \in  e}}{W}^{ * }\left( e\right)  = 1\text{for each}Y\text{due to tightness of}{W}^{ * }\text{)}
$$

which completes the proof of Theorem 7.

这就完成了定理7的证明。

5.4. A Subset Extension of Theorem 7. Remember that ${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)$ contains a relation ${R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$ (defined in (5.4)) for every attribute $X \in  \mathcal{I}$ . Given a non-empty subset $\mathcal{J} \subseteq  \mathcal{I}$ ,define

5.4. 定理7的子集扩展。请记住，${\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right)$ 对于每个属性 $X \in  \mathcal{I}$ 都包含一个关系 ${R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$（在(5.4)中定义）。给定一个非空子集 $\mathcal{J} \subseteq  \mathcal{I}$，定义

$$
{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right)  = \left\{  {{R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)  \mid  X \in  \mathcal{J}}\right\}   \tag{5.14}
$$

Note that $\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ is the cartesian product of the relations in ${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right)$ .

注意，$\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ 是 ${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right)$ 中各关系的笛卡尔积。

Take an arbitrary fractional edge packing $W$ of the hypergraph $\mathcal{G}$ . Define

取超图 $\mathcal{G}$ 的任意分数边填充 $W$。定义

$$
{W}_{\mathcal{J}} = \mathop{\sum }\limits_{{Y \in  \mathcal{J}}}\text{ weight of }Y\text{ under }W. \tag{5.15}
$$

We now present a general version of the isolated cartesian product theorem:

我们现在给出孤立笛卡尔积定理的一般形式：

Theorem 10 . Let $\mathcal{Q}$ be a simple binary query whose relations have a total size of $m$ . Denote by $\mathcal{G}$ the hypergraph defined by $\mathcal{Q}$ . Consider an arbitrary subset $\mathcal{H} \subseteq  \operatorname{attset}\left( \mathcal{Q}\right)$ , where attset(Q)is the set of attributes in the relations of $\mathcal{Q}$ . Let $\mathcal{I}$ be the set of isolated vertices defined in (5.3) and $\mathcal{J}$ be any non-empty subset of $\mathcal{I}$ . Take an arbitrary fractional edge packing $W$ of $\mathcal{G}$ . It holds that

定理10。设 $\mathcal{Q}$ 是一个简单二元查询，其关系的总大小为 $m$。用 $\mathcal{G}$ 表示由 $\mathcal{Q}$ 定义的超图。考虑任意子集 $\mathcal{H} \subseteq  \operatorname{attset}\left( \mathcal{Q}\right)$，其中 attset(Q) 是 $\mathcal{Q}$ 的关系中的属性集。设 $\mathcal{I}$ 是在(5.3)中定义的孤立顶点集，$\mathcal{J}$ 是 $\mathcal{I}$ 的任意非空子集。取 $\mathcal{G}$ 的任意分数边填充 $W$。则有

$$
\mathop{\sum }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \leq  {\lambda }^{\left| \mathcal{H}\right|  - {W}_{\mathcal{J}}} \cdot  {m}^{\left| \mathcal{J}\right| }. \tag{5.16}
$$

where $\lambda$ is the heavy parameter (see Section 4),config(Q,H)is the set of configurations of $\mathcal{H}$ (Section 4), ${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }$ is defined in (5.14),and ${W}_{\mathcal{J}}$ is defined in (5.15).

其中 $\lambda$ 是重参数（见第4节），config(Q,H) 是 $\mathcal{H}$ 的配置集（第4节），${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }$ 在(5.14)中定义，${W}_{\mathcal{J}}$ 在(5.15)中定义。

Proof. We will prove the theorem by reducing it to Theorem 7. Define $\overline{\mathcal{J}} = \mathcal{I} \smallsetminus  \mathcal{J}$ and

证明。我们将通过将该定理归约为定理7来证明它。定义 $\overline{\mathcal{J}} = \mathcal{I} \smallsetminus  \mathcal{J}$ 并且

$$
\widetilde{\mathcal{Q}} = \{ R \in  Q \mid  \text{ scheme }\left( R\right)  \cap  \overline{\mathcal{J}} = \varnothing \} .
$$

One can construct $\widetilde{\mathcal{Q}}$ alternatively as follows. First,discard from $\mathcal{Q}$ every relation whose scheme contains an attribute in $\overline{\mathcal{J}}$ . Then, $\widetilde{\mathcal{Q}}$ consists of the relations remaining in $\mathcal{Q}$ .

也可以按如下方式构造 $\widetilde{\mathcal{Q}}$。首先，从 $\mathcal{Q}$ 中舍弃其模式包含 $\overline{\mathcal{J}}$ 中某个属性的每个关系。然后，$\widetilde{\mathcal{Q}}$ 由 $\mathcal{Q}$ 中剩余的关系组成。

Denote by $\widetilde{\mathcal{G}} = \left( {\widetilde{\mathcal{V}},\widetilde{\mathcal{E}}}\right)$ the hypergraph defined by $\widetilde{\mathcal{Q}}$ . Set $\widetilde{\mathcal{H}} = \mathcal{H} \cap$ attset $\left( \widetilde{\mathcal{Q}}\right)$ and $\mathcal{L} = \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}.\mathcal{J}$ is precisely the set of isolated attributes decided by $\mathcal{Q}$ and $\mathcal{H}.{}^{3}$

用 $\widetilde{\mathcal{G}} = \left( {\widetilde{\mathcal{V}},\widetilde{\mathcal{E}}}\right)$ 表示由 $\widetilde{\mathcal{Q}}$ 定义的超图。设 $\widetilde{\mathcal{H}} = \mathcal{H} \cap$ attset $\left( \widetilde{\mathcal{Q}}\right)$，并且 $\mathcal{L} = \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}.\mathcal{J}$ 恰好是由 $\mathcal{Q}$ 和 $\mathcal{H}.{}^{3}$ 确定的孤立属性集

---

<!-- Footnote -->

${}^{3}$ Let $\widetilde{\mathcal{I}}$ be the set of isolated attributes after removing $\widetilde{\mathcal{H}}$ from $\widetilde{\mathcal{G}}$ . We want to prove $\mathcal{J} = \widetilde{\mathcal{I}}$ . It is easy to show $\mathcal{J} \subseteq  \widetilde{\mathcal{I}}$ . To prove $\widetilde{\mathcal{I}} \subseteq  \mathcal{J}$ ,suppose that there is an attribute $X$ such that $X \in  \widetilde{\mathcal{I}}$ but $X \notin  \mathcal{J}$ . As $X$ appears in $\widetilde{\mathcal{G}}$ ,we know $X \notin  \mathcal{I}$ . Hence, $\mathcal{G}$ must contain an edge $\{ X,Y\}$ with $Y \notin  \mathcal{H}$ . This means $Y \notin  \mathcal{I}$ ,

${}^{3}$ 设 $\widetilde{\mathcal{I}}$ 为从 $\widetilde{\mathcal{G}}$ 中移除 $\widetilde{\mathcal{H}}$ 后得到的孤立属性集。我们要证明 $\mathcal{J} = \widetilde{\mathcal{I}}$。很容易证明 $\mathcal{J} \subseteq  \widetilde{\mathcal{I}}$。为了证明 $\widetilde{\mathcal{I}} \subseteq  \mathcal{J}$，假设存在一个属性 $X$ 使得 $X \in  \widetilde{\mathcal{I}}$ 但 $X \notin  \mathcal{J}$。由于 $X$ 出现在 $\widetilde{\mathcal{G}}$ 中，我们知道 $X \notin  \mathcal{I}$。因此，$\mathcal{G}$ 必须包含一条边 $\{ X,Y\}$ 且 $Y \notin  \mathcal{H}$。这意味着 $Y \notin  \mathcal{I}$，

<!-- Footnote -->

---

Define a function $\widetilde{W} : \widetilde{\mathcal{E}} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ by setting $\widetilde{W}\left( e\right)  = W\left( e\right)$ for every $e \in  \widetilde{\mathcal{E}}.\widetilde{W}$ is a fractional edge packing of $\mathcal{G}$ . Because every edge $e \in  \mathcal{E}$ containing an attribute in $\mathcal{J}$ is preserved in $\widetilde{\mathcal{E}},{}^{4}$ we have ${W}_{\mathcal{J}} = {\widetilde{W}}_{\mathcal{J}}$ . Applying Theorem 7 to $\widetilde{\mathcal{Q}}$ gives:

通过为每个 $e \in  \widetilde{\mathcal{E}}.\widetilde{W}$ 设置 $\widetilde{W}\left( e\right)  = W\left( e\right)$ 来定义一个函数 $\widetilde{W} : \widetilde{\mathcal{E}} \rightarrow  \left\lbrack  {0,1}\right\rbrack$，它是 $\mathcal{G}$ 的一个分数边填充。因为包含 $\mathcal{J}$ 中某个属性的每条边 $e \in  \mathcal{E}$ 都在 $\widetilde{\mathcal{E}},{}^{4}$ 中得以保留，所以我们有 ${W}_{\mathcal{J}} = {\widetilde{W}}_{\mathcal{J}}$。对 $\widetilde{\mathcal{Q}}$ 应用定理 7 可得：

$$
\mathop{\sum }\limits_{{\widetilde{\mathbf{\eta }} \in  \operatorname{config}\left( {\widetilde{\mathcal{Q}},\widetilde{\mathcal{H}}}\right) }}\left| {\operatorname{Join}\left( {{\widetilde{\mathcal{Q}}}_{\text{ isolated }}^{\prime \prime }\left( \widetilde{\mathbf{\eta }}\right) }\right) }\right|  \leq  {\lambda }^{\left| \widetilde{\mathcal{H}}\right|  - {\widetilde{W}}_{\mathcal{T}}} \cdot  {m}^{\left| \mathcal{J}\right| } = {\lambda }^{\left| \widetilde{\mathcal{H}}\right|  - {W}_{\mathcal{T}}} \cdot  {m}^{\left| \mathcal{J}\right| }. \tag{5.17}
$$

It remains to show

还需证明

$$
\mathop{\sum }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \leq  {\lambda }^{\left| \mathcal{H}\right|  - \left| \widetilde{\mathcal{H}}\right| }\mathop{\sum }\limits_{{\widetilde{\mathbf{\eta }} \in  \operatorname{config}\left( {\widetilde{\mathcal{Q}},\widetilde{\mathcal{H}}}\right) }}\left| {\operatorname{Join}\left( {{\widetilde{\mathcal{Q}}}_{\text{isolated }}^{\prime \prime }\left( \widetilde{\mathbf{\eta }}\right) }\right) }\right|  \tag{5.18}
$$

after which Theorem 10 will follow from (5.17) and (5.18).

之后，定理 10 可由 (5.17) 和 (5.18) 推出。

For each $\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ ,we can find $\widetilde{\mathbf{\eta }} = \mathbf{\eta }\left\lbrack  \widetilde{\mathcal{H}}\right\rbrack   \in  \operatorname{config}\left( {\widetilde{Q},\widetilde{H}}\right)$ such that $\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  =$ $\operatorname{Join}\left( {{\widetilde{\mathcal{Q}}}_{\text{isolated }}^{\prime \prime }\left( \widetilde{\mathbf{\eta }}\right) }\right)$ . The correctness of (5.18) follows from the fact that at most ${\lambda }^{\left| \mathcal{H}\right|  - \left| \mathcal{H}\right| }$ configurations $\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ correspond to the same $\widetilde{\mathbf{\eta }}$ .

对于每个 $\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$，我们可以找到 $\widetilde{\mathbf{\eta }} = \mathbf{\eta }\left\lbrack  \widetilde{\mathcal{H}}\right\rbrack   \in  \operatorname{config}\left( {\widetilde{Q},\widetilde{H}}\right)$ 使得 $\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  =$ $\operatorname{Join}\left( {{\widetilde{\mathcal{Q}}}_{\text{isolated }}^{\prime \prime }\left( \widetilde{\mathbf{\eta }}\right) }\right)$。(5.18) 的正确性源于这样一个事实：至多 ${\lambda }^{\left| \mathcal{H}\right|  - \left| \mathcal{H}\right| }$ 个配置 $\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right)$ 对应于同一个 $\widetilde{\mathbf{\eta }}$。

5.5. A Weaker Result. One issue in applying Theorem 10 is that the quantity $\left| \mathcal{H}\right|  - {W}_{\mathcal{J}}$ is not directly related to the fractional edge covering number $\rho$ of $\mathcal{Q}$ . The next lemma gives a weaker result that addresses the issue to an extent sufficient for our purposes in Section 6:

5.5. 一个较弱的结果。应用定理 10 时的一个问题是，量 $\left| \mathcal{H}\right|  - {W}_{\mathcal{J}}$ 与 $\mathcal{Q}$ 的分数边覆盖数 $\rho$ 没有直接关系。接下来的引理给出了一个较弱的结果，在一定程度上解决了这个问题，足以满足我们在第 6 节中的目的：

Lemma 11 . Let $\mathcal{Q}$ be a simple binary query who relations have a total size of $m$ . Denote by $\mathcal{G}$ the hypergraph defined by $\mathcal{Q}$ . Consider an arbitrary subset $\mathcal{H} \subseteq$ attset(Q),where attset(Q) is the set of attributes in the relations of $\mathcal{Q}$ . Define $\mathcal{L} =$ attset $\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$ and $\mathcal{I}$ as the set of isolated vertices in $\mathcal{L}$ (see (5.3)). For any non-empty subset $\mathcal{J} \subseteq  \mathcal{I}$ ,it holds that

引理11。设$\mathcal{Q}$为一个简单二元查询，其关系的总大小为$m$。用$\mathcal{G}$表示由$\mathcal{Q}$定义的超图。考虑任意子集$\mathcal{H} \subseteq$⊆attset(Q)，其中attset(Q)是$\mathcal{Q}$的关系中的属性集。定义$\mathcal{L} =$=attset($\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$)且$\mathcal{I}$为$\mathcal{L}$中的孤立顶点集（见(5.3)）。对于任意非空子集$\mathcal{J} \subseteq  \mathcal{I}$，有

$$
\mathop{\sum }\limits_{{\mathbf{\eta } \in  \operatorname{config}\left( {\mathcal{Q},\mathcal{H}}\right) }}\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|  \leq  {\lambda }^{{2\rho } - \left| \mathcal{J}\right|  - \left| \mathcal{L}\right| } \cdot  {m}^{\left| \mathcal{J}\right| } \tag{5.19}
$$

where $\rho$ is the fractional edge covering number of $\mathcal{G},\lambda$ is the heavy parameter (Section 4), config(Q,H)is the set of configurations of $\mathcal{H}$ (Section 4),and ${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right)$ is defined in (5.14).

其中$\rho$是$\mathcal{G},\lambda$的分数边覆盖数，$\mathcal{G},\lambda$是重参数（第4节），config(Q,H)是$\mathcal{H}$的配置集（第4节），且${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right)$在(5.14)中定义。

Proof. Let $W$ be an arbitrary fractional edge packing of $\mathcal{G}$ satisfying the second bullet of Lemma 1. Specifically,the weight of $W$ is the fractional edge packing number $\tau$ of $\mathcal{G}$ ; and the weight of every vertex in $\mathcal{G}$ is either 0 or 1 . Denote by $Z$ the set of vertices in $\mathcal{G}$ whose weights under $W$ are 0 . Lemma 1 tells us $\tau  + \rho  = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$ and $\rho  - \tau  = \left| \mathcal{Z}\right|$ . Set ${\mathcal{J}}_{0} = \mathcal{J} \cap  Z$ and ${\mathcal{J}}_{1} = \mathcal{J} \smallsetminus  {\mathcal{J}}_{0}$ . Because ${\mathcal{J}}_{0} \subseteq  \mathcal{Z}$ ,we can derive:

证明。设$W$是$\mathcal{G}$的任意一个满足引理1第二个条件的分数边填充。具体来说，$W$的权重是$\mathcal{G}$的分数边填充数$\tau$；并且$\mathcal{G}$中每个顶点的权重要么为0要么为1。用$Z$表示$\mathcal{G}$中在$W$下权重为0的顶点集。引理1告诉我们$\tau  + \rho  = \left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|$和$\rho  - \tau  = \left| \mathcal{Z}\right|$。设${\mathcal{J}}_{0} = \mathcal{J} \cap  Z$和${\mathcal{J}}_{1} = \mathcal{J} \smallsetminus  {\mathcal{J}}_{0}$。因为${\mathcal{J}}_{0} \subseteq  \mathcal{Z}$，我们可以推导出：

$$
\tau  + \left| {\mathcal{J}}_{0}\right|  \leq  \rho  \Rightarrow  
$$

$$
\left| {\operatorname{attset}\left( \mathcal{Q}\right) }\right|  - \rho  + \left| {\mathcal{J}}_{0}\right|  \leq  \rho  \Rightarrow  
$$

$$
\left( {\left| \mathcal{H}\right|  + \left| \mathcal{L}\right| }\right)  + \left( {\left| \mathcal{J}\right|  - \left| {\mathcal{J}}_{1}\right| }\right)  \leq  {2\rho } \Rightarrow  
$$

$$
\left| \mathcal{H}\right|  - \left| {\mathcal{J}}_{1}\right|  \leq  {2\rho } - \left| \mathcal{J}\right|  - \left| \mathcal{L}\right| .
$$

Lemma 11 now follows from Theorem 10 due to $\left| {\mathcal{J}}_{1}\right|  = {W}_{\mathcal{J}}$ ,which holds because every vertex in ${\mathcal{J}}_{1}$ has weight 1 under $W$ .

由于$\left| {\mathcal{J}}_{1}\right|  = {W}_{\mathcal{J}}$，引理11现在可由定理10推出，这是因为${\mathcal{J}}_{1}$中的每个顶点在$W$下的权重都为1。

---

<!-- Footnote -->

because of which the edge $\{ X,Y\}$ is disjoint with $\overline{\mathcal{J}}$ and thus must belong to $\widetilde{\mathcal{G}}$ . But this contradicts the fact $X \in  \widetilde{\mathcal{I}}$ .

因此边$\{ X,Y\}$与$\overline{\mathcal{J}}$不相交，从而必定属于$\widetilde{\mathcal{G}}$。但这与事实$X \in  \widetilde{\mathcal{I}}$矛盾。

${}^{4}$ Suppose that there is an edge $e = \{ X,Y\}$ such that $X \in  \mathcal{J}$ and yet $e \notin  \widetilde{\mathcal{E}}$ . It means that $Y \in  \bar{\mathcal{J}} \subseteq  \mathcal{I}$ . But then $e$ is incident on two attributes in $\mathcal{I}$ ,which is impossible.

${}^{4}$假设存在一条边$e = \{ X,Y\}$使得$X \in  \mathcal{J}$，然而$e \notin  \widetilde{\mathcal{E}}$。这意味着$Y \in  \bar{\mathcal{J}} \subseteq  \mathcal{I}$。但这样$e$与$\mathcal{I}$中的两个属性相关联，这是不可能的。

<!-- Footnote -->

---

Remark. The above lemma was the "isolated cartesian product theorem" presented in the preliminary version [20] of this work. The new version (i.e., Theorem 10) is more powerful and better captures the mathematical structure underneath.

注记。上述引理是本文初稿[20]中提出的“孤立笛卡尔积定理”。新版本（即定理10）更强大，并且更好地捕捉了其背后的数学结构。

### 6.An MPC JOIN ALGORITHM

### 6. 一种MPC连接算法

This section will describe how to answer a simple binary join $\mathcal{Q}$ in the MPC model with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ .

本节将描述如何在MPC模型中以负载$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$回答一个简单二元连接$\mathcal{Q}$。

We define a statistical record as a tuple(R,X,x, cnt),where $R$ is a relation in $\mathcal{Q},X$ an attribute in scheme $\left( R\right) ,x$ a value in $\mathbf{{dom}}$ ,and cnt the number of tuples $\mathbf{u} \in  R$ with $\mathbf{u}\left( X\right)  = x$ . Specially, $\left( {R,\varnothing ,{nil},{cnt}}\right)$ is also a statistical record where ${cnt}$ gives the number of tuples in $R$ that use only light values. A histogram is defined as the set of statistical records for all possible $R,X$ ,and $x$ satisfying (i) cnt $= \Omega \left( {m/{p}^{1/\rho }}\right)$ or (ii) $X = \varnothing$ (and,hence $x =$ nil); note that there are only $O\left( {p}^{1/\rho }\right)$ such records. We assume that every machine has a local copy of the histogram. By resorting to standard MPC sorting algorithms [9,10], the assumption can be satisfied with a preprocessing that takes constant rounds and load $\widetilde{O}\left( {{p}^{1/\rho } + m/p}\right)$ .

我们将统计记录定义为一个元组 (R, X, x, cnt)，其中 $R$ 是 $\mathcal{Q},X$ 中的一个关系，$\left( R\right) ,x$ 模式中的一个属性，$\mathbf{{dom}}$ 中的一个值，cnt 是 $\mathbf{u} \in  R$ 中满足 $\mathbf{u}\left( X\right)  = x$ 的元组数量。特别地，$\left( {R,\varnothing ,{nil},{cnt}}\right)$ 也是一个统计记录，其中 ${cnt}$ 给出了 $R$ 中仅使用轻量级值的元组数量。直方图被定义为所有可能的 $R,X$ 和 $x$ 满足 (i) cnt $= \Omega \left( {m/{p}^{1/\rho }}\right)$ 或 (ii) $X = \varnothing$（因此 $x =$ 为空）的统计记录集合；请注意，这样的记录只有 $O\left( {p}^{1/\rho }\right)$ 条。我们假设每台机器都有直方图的本地副本。通过采用标准的多方计算（MPC）排序算法 [9,10]，可以通过一个需要常数轮次和负载 $\widetilde{O}\left( {{p}^{1/\rho } + m/p}\right)$ 的预处理来满足该假设。

Henceforth, we will fix the heavy parameter

此后，我们将固定重参数

$$
\lambda  = \Theta \left( {p}^{1/\left( {2\rho }\right) }\right) 
$$

and focus on explaining how to compute (5.1) for an arbitrary subset $\mathcal{H}$ of attset(Q). As attset(Q)has ${2}^{k} = O\left( 1\right)$ subsets (where $k =  \mid$ attset $\left( \mathcal{Q}\right)  \mid$ ),processing them all in parallel increases the load by only a constant factor and, as guaranteed by (4.2), discovers the entire $\operatorname{Join}\left( \mathcal{Q}\right)$ .

并专注于解释如何为 attset(Q) 的任意子集 $\mathcal{H}$ 计算 (5.1)。由于 attset(Q) 有 ${2}^{k} = O\left( 1\right)$ 个子集（其中 $k =  \mid$ 为 attset $\left( \mathcal{Q}\right)  \mid$），并行处理所有子集只会使负载增加一个常数因子，并且如 (4.2) 所保证的那样，可以发现整个 $\operatorname{Join}\left( \mathcal{Q}\right)$。

Our algorithm produces (5.1) in three steps:

我们的算法分三步计算 (5.1)：

(1) Generate the input relations of the residual query ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ of every configuration $\mathbf{\eta }$ of $\mathcal{H}$ (Section 5.1).

(1) 生成 $\mathcal{H}$ 的每个配置 $\mathbf{\eta }$ 的残差查询 ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ 的输入关系（第 5.1 节）。

(2) Generate the input relations of the reduced query ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ of every $\mathbf{\eta }$ (Section 5.2).

(2) 生成每个 $\mathbf{\eta }$ 的简化查询 ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ 的输入关系（第 5.2 节）。

(3) Evaluate ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ for every $\mathbf{\eta }$ .

(3) 为每个 $\mathbf{\eta }$ 计算 ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$。

The number of configurations of $\mathcal{H}$ is $O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)  = O\left( {\lambda }^{k}\right)  = O\left( {p}^{k/\left( {2\rho }\right) }\right)$ ,which is $O\left( p\right)$ because $\rho  \geq  k/2$ by the first bullet of Lemma 1. Next,we elaborate on the details of each step.

$\mathcal{H}$ 的配置数量为 $O\left( {\lambda }^{\left| \mathcal{H}\right| }\right)  = O\left( {\lambda }^{k}\right)  = O\left( {p}^{k/\left( {2\rho }\right) }\right)$，由于引理 1 的第一点可知 $\rho  \geq  k/2$，所以该数量为 $O\left( p\right)$。接下来，我们详细阐述每个步骤的细节。

Step 1. Lemma 6 tells us that the input relations of all the residual queries have at most $m \cdot  {\lambda }^{k - 2}$ tuples in total. We allocate ${p}_{\mathbf{\eta }}^{\prime } = \left\lceil  {p \cdot  \frac{{m}_{\mathbf{\eta }}}{\Theta \left( {m \cdot  {\lambda }^{k - 2}}\right) }}\right\rceil$ machines to store the relations of ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ ,making sure that $\mathop{\sum }\limits_{\mathbf{\eta }}{p}_{\mathbf{\eta }}^{\prime } \leq  p$ . Each machine keeps on average

步骤 1. 引理 6 告诉我们，所有残差查询的输入关系总共最多有 $m \cdot  {\lambda }^{k - 2}$ 个元组。我们分配 ${p}_{\mathbf{\eta }}^{\prime } = \left\lceil  {p \cdot  \frac{{m}_{\mathbf{\eta }}}{\Theta \left( {m \cdot  {\lambda }^{k - 2}}\right) }}\right\rceil$ 台机器来存储 ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ 的关系，确保 $\mathop{\sum }\limits_{\mathbf{\eta }}{p}_{\mathbf{\eta }}^{\prime } \leq  p$。每台机器平均保留

$$
O\left( {{m}_{\eta }/{p}_{\eta }^{\prime }}\right)  = O\left( {m \cdot  {\lambda }^{k - 2}/p}\right)  = O\left( {m/{p}^{1/\rho }}\right) 
$$

tuples,where the last equality used $\rho  \geq  k/2$ . Each machine $i \in  \left\lbrack  {1,p}\right\rbrack$ can use the histogram to calculate the input size ${m}_{\mathbf{\eta }}$ of ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ precisely for each $\mathbf{\eta }$ ; it can compute locally the id range of the ${m}_{\mathbf{\eta }}$ machines responsible for ${Q}^{\prime }\left( \mathbf{\eta }\right)$ . If a tuple $\mathbf{u}$ in the local storage of machine $i$ belongs to ${Q}^{\prime }\left( \mathbf{\eta }\right)$ ,the machine sends $\mathbf{u}$ to a random machine within that id range. Standard analysis shows that each of the ${m}_{\eta }$ machines receives asymptotically the same number of tuples of ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ (up to an $\widetilde{O}\left( 1\right)$ factor) with probability at least $1 - 1/{p}^{c}$ for an arbitrarily large constant $c$ . Hence,Step 1 can be done in a single round with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ with probability at least $1 - 1/{p}^{c}$ .

元组，其中最后一个等式使用了$\rho  \geq  k/2$。每台机器$i \in  \left\lbrack  {1,p}\right\rbrack$可以使用直方图为每个$\mathbf{\eta }$精确计算${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$的输入大小${m}_{\mathbf{\eta }}$；它可以在本地计算负责${Q}^{\prime }\left( \mathbf{\eta }\right)$的${m}_{\mathbf{\eta }}$台机器的id范围。如果机器$i$本地存储中的一个元组$\mathbf{u}$属于${Q}^{\prime }\left( \mathbf{\eta }\right)$，则该机器将$\mathbf{u}$发送到该id范围内的一台随机机器。标准分析表明，对于任意大的常数$c$，${m}_{\eta }$台机器中的每一台以至少$1 - 1/{p}^{c}$的概率渐近地接收到相同数量的${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$元组（误差在$\widetilde{O}\left( 1\right)$因子范围内）。因此，步骤1可以在一轮内以负载$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$完成，且概率至少为$1 - 1/{p}^{c}$。

Step 2. Now that all the input relations of each ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ have been stored on ${p}_{\mathbf{\eta }}^{\prime }$ machines,the semi-join reduction in Section 5.2 that converts ${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$ to ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ is a standard process that can be accomplished [10] with sorting in $O\left( 1\right)$ rounds entailing a load of $\widetilde{O}\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ ; see also [13] for a randomized algorithm that performs fewer rounds.

步骤2。现在，每个${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$的所有输入关系都已存储在${p}_{\mathbf{\eta }}^{\prime }$台机器上，第5.2节中将${\mathcal{Q}}^{\prime }\left( \mathbf{\eta }\right)$转换为${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$的半连接约简是一个标准过程，可以通过排序在$O\left( 1\right)$轮内完成，负载为$\widetilde{O}\left( {{m}_{\mathbf{\eta }}/{p}_{\mathbf{\eta }}^{\prime }}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$；另见[13]中执行轮数更少的随机算法。

Step 3. This step starts by letting each machine know about the value of $\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|$ for every $\mathbf{\eta }$ . For this purpose,each machine broadcasts to all other machines how many tuples it has in ${R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$ for every $X \in  \mathcal{I}$ and every $\mathbf{\eta }$ . Since there are $O\left( p\right)$ different $\mathbf{\eta },O\left( p\right)$ numbers are sent by each machine,such that the load of this round is $O\left( {p}^{2}\right)$ . From the numbers received,each machine can independently figure out the values of all $\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|$ .

步骤3。此步骤首先让每台机器了解每个$\mathbf{\eta }$对应的$\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|$的值。为此，每台机器向所有其他机器广播它在每个$X \in  \mathcal{I}$和每个$\mathbf{\eta }$对应的${R}_{X}^{\prime \prime }\left( \mathbf{\eta }\right)$中拥有的元组数量。由于每台机器会发送$O\left( p\right)$个不同的$\mathbf{\eta },O\left( p\right)$，因此这一轮的负载为$O\left( {p}^{2}\right)$。根据接收到的数字，每台机器可以独立地算出所有$\left| {\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right) }\right|$的值。

We allocate

我们分配

$$
{p}_{\mathbf{\eta }}^{\prime \prime } = \Theta \left( {{\lambda }^{\left| \mathcal{L}\right| } + p \cdot  \mathop{\sum }\limits_{{\text{non-empty }\mathcal{J} \subseteq  \mathcal{I}}}\frac{\left| \operatorname{Join}\left( {\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) \right) \right| }{{\lambda }^{{2\rho } - \left| \mathcal{J}\right|  - \left| \mathcal{L}\right| } \cdot  {m}^{\left| \mathcal{J}\right| }}}\right)  \tag{6.1}
$$

machines for computing ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ . Notice that

台机器来计算${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$。注意到

$$
\mathop{\sum }\limits_{\mathbf{\eta }}{p}_{\mathbf{\eta }}^{\prime \prime } = O\left( {\mathop{\sum }\limits_{\mathbf{\eta }}{\lambda }^{\left| \mathcal{L}\right| }}\right)  + O\left( {p \cdot  \mathop{\sum }\limits_{{\text{non-empty }\mathcal{J} \subseteq  \mathcal{I}}}\mathop{\sum }\limits_{\mathbf{\eta }}\frac{\left| \operatorname{Join}\left( {\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) \right) \right| }{{\lambda }^{{2\rho } - \left| \mathcal{J}\right|  - \left| \mathcal{L}\right| } \cdot  {m}^{\left| \mathcal{J}\right| }}}\right)  = O\left( p\right) 
$$

where the equality used Lemma 11,the fact that $\mathcal{I}$ has constant non-empty subsets,and that $\mathop{\sum }\limits_{n}{\lambda }^{\left| \mathcal{L}\right| } \leq  {\lambda }^{\left| \mathcal{H}\right| } \cdot  {\lambda }^{\left| \mathcal{L}\right| } = {\lambda }^{k} \leq  p$ . We can therefore adjust the constants in (6.1) to make sure that the total number of machines needed by all the configurations is at most $p$ .

其中等式使用了引理11，即$\mathcal{I}$具有恒定的非空子集这一事实，以及$\mathop{\sum }\limits_{n}{\lambda }^{\left| \mathcal{L}\right| } \leq  {\lambda }^{\left| \mathcal{H}\right| } \cdot  {\lambda }^{\left| \mathcal{L}\right| } = {\lambda }^{k} \leq  p$ 。因此，我们可以调整(6.1)中的常数，以确保所有配置所需的机器总数最多为$p$ 。

Lemma 12. ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$ can be answered in one round with load $O\left( {m/{p}^{1/\rho }}\right)$ using ${p}_{\mathbf{\eta }}^{\prime \prime }$ machines, subject to a failure probability of at most $1/{p}^{c}$ where $c$ can be set to an arbitrarily large constant.

引理12. ${\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right)$可以在一轮中使用${p}_{\mathbf{\eta }}^{\prime \prime }$台机器以负载$O\left( {m/{p}^{1/\rho }}\right)$进行解答，故障概率至多为$1/{p}^{c}$，其中$c$可以设为任意大的常数。

Proof. As shown in (5.8), $\operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ is the cartesian product of $\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ and $\operatorname{Join}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ . We deploy $\Theta \left( {{p}_{\mathbf{\eta }}^{\prime \prime }/{\lambda }^{\left| \mathcal{L}\right|  - \left| \mathcal{I}\right| }}\right)$ machines to compute $\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ in one round. By Lemma 3, the load is

证明。如(5.8)所示，$\operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$是$\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$和$\operatorname{Join}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$的笛卡尔积。我们部署$\Theta \left( {{p}_{\mathbf{\eta }}^{\prime \prime }/{\lambda }^{\left| \mathcal{L}\right|  - \left| \mathcal{I}\right| }}\right)$台机器在一轮中计算$\operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$。根据引理3，负载为

$$
\widetilde{O}\left( \frac{{\left| \operatorname{Join}\left( {\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) \right) \right| }^{1/\left| \mathcal{J}\right| }}{{\left( \frac{{p}_{\mathbf{\eta }}^{\prime \prime }}{\lambda \left| \mathcal{L}\right|  - \left| \mathcal{I}\right| }\right) }^{1/\left| \mathcal{J}\right| }}\right)  \tag{6.2}
$$

for some non-empty $\mathcal{J} \subseteq  \mathcal{I}$ . (6.1) guarantees that

对于某个非空的$\mathcal{J} \subseteq  \mathcal{I}$ 。(6.1)保证了

$$
{p}_{\mathbf{\eta }}^{\prime \prime } = \Omega \left( {p \cdot  \frac{\left| \operatorname{Join}\left( {\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathbf{\eta }\right) \right) \right| }{{\lambda }^{{2\rho } - \left| \mathcal{J}\right|  - \left| \mathcal{L}\right| } \cdot  m\left| \mathcal{J}\right| }}\right) 
$$

with which we can derive

由此我们可以推导出

$$
\left( {6.2}\right)  = \widetilde{O}\left( \frac{m \cdot  {\lambda }^{\frac{{2\rho } - \left| \mathcal{J}\right|  - \left| \mathcal{Z}\right| }{\left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{m \cdot  {\lambda }^{\frac{{2\rho } - 2\left| \mathcal{J}\right| }{\left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{m \cdot  {p}^{\frac{{2\rho } - 2\left| \mathcal{J}\right| }{{2\rho }\left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{m}{{p}^{1/\rho }}\right) .
$$

Regarding ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$ ,first note that attset $\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  = \mathcal{L} \smallsetminus  \mathcal{I}$ . If $\mathcal{L} \smallsetminus  \mathcal{I}$ is empty,no ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$ exists and $\operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  = \operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ . The subsequent discussion considers that $\mathcal{L} \smallsetminus  \mathcal{I}$ is not empty. As the input relations of ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$ contain only light values, ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$ is skew-free if a share of $\lambda$ is assigned to each attribute in $\mathcal{L} \smallsetminus  \mathcal{I}$ . By Lemma 5, $\operatorname{Join}\left( {{\mathcal{Q}}_{\text{liaht }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ can be computed in one round with load $\widetilde{O}\left( {m/{\lambda }^{2}}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ using $\Theta \left( {\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }\right)$ machines, subject to a certain failure probability $\delta$ . As ${\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| } \geq  \lambda$ which is a polynomial of $p$ ,Lemma 5 allows us to make sure $\delta  \leq  1/{p}^{c}$ for any constant $c$ .

关于${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$，首先注意到属性集为$\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  = \mathcal{L} \smallsetminus  \mathcal{I}$。如果$\mathcal{L} \smallsetminus  \mathcal{I}$为空，则不存在${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$且$\operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)  = \operatorname{Join}\left( {{\mathcal{Q}}_{\text{isolated }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$。后续讨论考虑$\mathcal{L} \smallsetminus  \mathcal{I}$不为空的情况。由于${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$的输入关系仅包含轻值，如果将$\lambda$的一部分分配给$\mathcal{L} \smallsetminus  \mathcal{I}$中的每个属性，则${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( \mathbf{\eta }\right)$是无偏斜的。根据引理5，在一定的失败概率$\delta$下，使用$\Theta \left( {\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }\right)$台机器可以在一轮内以负载$\widetilde{O}\left( {m/{\lambda }^{2}}\right)  = \widetilde{O}\left( {m/{p}^{1/\rho }}\right)$计算出$\operatorname{Join}\left( {{\mathcal{Q}}_{\text{liaht }}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$。由于${\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| } \geq  \lambda$是$p$的多项式，引理5使我们能够确保对于任何常数$c$都有$\delta  \leq  1/{p}^{c}$。

By combining the above discussion with Lemma 4,we conclude that $\operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$ can be computed in one round with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ using ${p}_{\mathbf{\eta }}^{\prime \prime }$ machines,subject to a failure probability at most $\delta  \leq  1/{p}^{c}$ .

将上述讨论与引理4相结合，我们得出结论：在最多$\delta  \leq  1/{p}^{c}$的失败概率下，使用${p}_{\mathbf{\eta }}^{\prime \prime }$台机器可以在一轮内以负载$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$计算出$\operatorname{Join}\left( {{\mathcal{Q}}^{\prime \prime }\left( \mathbf{\eta }\right) }\right)$。

Overall,the load of our algorithm is $\widetilde{O}\left( {{p}^{1/\rho } + {p}^{2} + m/{p}^{1/\rho }}\right)$ . This brings us to our second main result:

总体而言，我们算法的负载为$\widetilde{O}\left( {{p}^{1/\rho } + {p}^{2} + m/{p}^{1/\rho }}\right)$。这引出了我们的第二个主要结果：

Theorem 13 . Given a simple binary join query with input size $m \geq  {p}^{3}$ and a fractional edge covering number $\rho$ ,we can answer it in the MPC model using $p$ machines in constant rounds with load $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ ,subject to a failure probability of at most $1/{p}^{c}$ where $c$ can be set to an arbitrarily large constant.

定理13。给定一个输入大小为$m \geq  {p}^{3}$且分数边覆盖数为$\rho$的简单二元连接查询，在最多$1/{p}^{c}$的失败概率下，我们可以在MPC（大规模并行计算，Massively Parallel Computation）模型中使用$p$台机器在常数轮数内以负载$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$回答该查询，其中$c$可以设置为任意大的常数。

## 7. Concluding Remarks

## 7. 总结性评论

This paper has introduced an algorithm for computing a natural join over binary relations under the MPC model. Our algorithm performs a constant number of rounds and incurs a load of $\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$ where $m$ is the total size of the input relations, $p$ is the number of machines,and $\rho$ is the fractional edge covering number of the query. The load matches a known lower bound up to a polylogarithmic factor. Our techniques heavily rely on a new finding, which we refer to as the isolated cartesian product theorem, on the join problem's mathematical structure.

本文介绍了一种在大规模并行计算（MPC）模型下计算二元关系自然连接的算法。我们的算法执行固定轮数，负载为$\widetilde{O}\left( {m/{p}^{1/\rho }}\right)$，其中$m$是输入关系的总大小，$p$是机器数量，$\rho$是查询的分数边覆盖数。该负载在一个多对数因子范围内与已知的下界相匹配。我们的技术在很大程度上依赖于一个新的发现，我们称之为孤立笛卡尔积定理，它与连接问题的数学结构有关。

## We conclude the paper with two remarks:

## 我们以两条备注结束本文：

- The assumption ${p}^{3} \leq  m$ can be relaxed to $p \leq  {m}^{1 - \epsilon }$ for an arbitrarily small constant $\epsilon  > 0$ . Recall that our algorithm incurs a load of $\widetilde{O}\left( {{p}^{1/\rho } + {p}^{2} + m/{p}^{1/\rho }}\right)$ where the terms $\widetilde{O}\left( {p}^{1/\rho }\right)$ and $\widetilde{O}\left( {p}^{2}\right)$ are both due to the computation of statistics (in preprocessing and Step 2, respectively). In turn, these statistics are needed to allocate machines for subproblems. By using the machine-allocation techniques in [10], we can avoid most of the statistics communication and reduce the load to $\widetilde{O}\left( {{p}^{\epsilon } + m/{p}^{1/\rho }}\right)$ .

- 假设${p}^{3} \leq  m$可以放宽为$p \leq  {m}^{1 - \epsilon }$，其中$\epsilon  > 0$是任意小的常数。回想一下，我们的算法负载为$\widetilde{O}\left( {{p}^{1/\rho } + {p}^{2} + m/{p}^{1/\rho }}\right)$，其中项$\widetilde{O}\left( {p}^{1/\rho }\right)$和$\widetilde{O}\left( {p}^{2}\right)$分别是由于统计信息的计算（分别在预处理和步骤2中）。反过来，这些统计信息是为子问题分配机器所必需的。通过使用文献[10]中的机器分配技术，我们可以避免大部分统计信息的通信，并将负载降低到$\widetilde{O}\left( {{p}^{\epsilon } + m/{p}^{1/\rho }}\right)$。

- In the external memory (EM) model [4],we have a machine equipped with $M$ words of internal memory and an unbounded disk that has been formatted into blocks of size $B$ words. An I/O either reads a block of $B$ words from the disk to the memory,or overwrites a block with $B$ words in the memory. A join query $\mathcal{Q}$ is considered solved if every tuple $\mathbf{u} \in  \mathcal{Q}$ has been generated in memory at least once. The challenge is to design an algorithm to achieve the purpose with as few I/Os as possible. There exists a reduction [13] that can be used to convert an MPC algorithm to an EM counterpart. Applying the reduction on our algorithm gives an EM algorithm that solves $\mathcal{Q}$ with $\widetilde{O}\left( \frac{{m}^{\rho }}{B \cdot  {M}^{\rho  - 1}}\right)$ I/Os,provided that $M \geq  {m}^{c}$ for some positive constant $c < 1$ that depends on $\mathcal{Q}$ . The I/O complexity can be shown to be optimal up to a polylogarithmic factor using the lower-bound arguments in $\left\lbrack  {{11},{18}}\right\rbrack$ . We suspect that the constraint $M \geq  {m}^{c}$ can be removed by adapting the isolated cartesian product theorem to the EM model.

- 在外存（EM）模型[4]中，我们有一台配备$M$字内部内存的机器和一个无界磁盘，磁盘已被格式化为大小为$B$字的块。一次输入/输出（I/O）操作要么从磁盘读取一个$B$字的块到内存，要么用内存中的$B$字覆盖一个块。如果每个元组$\mathbf{u} \in  \mathcal{Q}$至少在内存中生成一次，则认为连接查询$\mathcal{Q}$已解决。挑战在于设计一种算法，以尽可能少的I/O操作实现这一目的。存在一种归约方法[13]，可用于将MPC算法转换为对应的EM算法。将该归约方法应用于我们的算法，可得到一个EM算法，该算法在满足$M \geq  {m}^{c}$（其中$c < 1$是依赖于$\mathcal{Q}$的某个正常数）的条件下，以$\widetilde{O}\left( \frac{{m}^{\rho }}{B \cdot  {M}^{\rho  - 1}}\right)$次I/O操作解决$\mathcal{Q}$。使用文献$\left\lbrack  {{11},{18}}\right\rbrack$中的下界论证可以证明，该I/O复杂度在一个多对数因子范围内是最优的。我们怀疑通过将孤立笛卡尔积定理应用于EM模型，可以去除约束$M \geq  {m}^{c}$。

## REFERENCES

## 参考文献

[1] Serge Abiteboul, Richard Hull, and Victor Vianu. Foundations of Databases. Addison-Wesley, 1995.

[2] Azza Abouzeid, Kamil Bajda-Pawlikowski, Daniel J. Abadi, Alexander Rasin, and Avi Silberschatz. Hadoopdb: An architectural hybrid of mapreduce and dbms technologies for analytical workloads. Proceedings of the VLDB Endowment (PVLDB), 2(1):922-933, 2009.

[3] Foto N. Afrati and Jeffrey D. Ullman. Optimizing multiway joins in a map-reduce environment. IEEE Transactions on Knowledge and Data Engineering (TKDE), 23(9):1282-1298, 2011.

[4] Alok Aggarwal and Jeffrey Scott Vitter. The input/output complexity of sorting and related problems. Communications of the ACM (CACM), 31(9):1116-1127, 1988.

[5] Albert Atserias, Martin Grohe, and Daniel Marx. Size bounds and query plans for relational joins. SIAM Journal on Computing, 42(4):1737-1767, 2013.

[6] Paul Beame, Paraschos Koutris, and Dan Suciu. Communication steps for parallel query processing. Journal of the ACM (JACM), 64(6):40:1-40:58, 2017.

[7] Stephen A. Cook. The complexity of theorem-proving procedures. In Proceedings of ACM Symposium on Theory of Computing (STOC), pages 151-158, 1971.

[8] Jeffrey Dean and Sanjay Ghemawat. Mapreduce: Simplified data processing on large clusters. In Proceedings of USENIX Symposium on Operating Systems Design and Implementation (OSDI), pages 137-150, 2004.

[9] Michael T. Goodrich. Communication-efficient parallel sorting. SIAM Journal of Computing, 29(2):416- 432, 1999.

[10] Xiao Hu, Ke Yi, and Yufei Tao. Output-optimal massively parallel algorithms for similarity joins. ACM Transactions on Database Systems (TODS), 44(2):6:1-6:36, 2019.

[11] Xiaocheng Hu, Miao Qiao, and Yufei Tao. I/O-efficient join dependency testing, loomis-whitney join, and triangle enumeration. Journal of Computer and System Sciences (JCSS), 82(8):1300-1315, 2016.

[12] Bas Ketsman and Dan Suciu. A worst-case optimal multi-round algorithm for parallel computation of conjunctive queries. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 417-428, 2017.

[13] Paraschos Koutris, Paul Beame, and Dan Suciu. Worst-case optimal algorithms for parallel query processing. In Proceedings of International Conference on Database Theory (ICDT), pages 8:1-8:18, 2016.

[14] Paraschos Koutris, Semih Salihoglu, and Dan Suciu. Algorithmic aspects of parallel data processing. Foundations and Trends in Databases, 8(4):239-370, 2018.

[15] Ilan Newman. Private vs. common random bits in communication complexity. Information Processing Letters (IPL), 39(2):67-71, 1991.

[16] Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. Worst-case optimal join algorithms. Journal of the ACM (JACM), 65(3):16:1-16:40, 2018.

[17] Hung Q. Ngo, Christopher Re, and Atri Rudra. Skew strikes back: new developments in the theory of join algorithms. SIGMOD Record, 42(4):5-16, 2013.

[18] Rasmus Pagh and Francesco Silvestri. The input/output complexity of triangle enumeration. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 224-233, 2014.

[19] Edward R. Scheinerman and Daniel H. Ullman. Fractional Graph Theory: A Rational Approach to the Theory of Graphs. Wiley, New York, 1997.

[20] Yufei Tao. A simple parallel algorithm for natural joins on binary relations. In Proceedings of International Conference on Database Theory (ICDT), pages 25:1-25:18, 2020.

[21] Todd L. Veldhuizen. Triejoin: A simple, worst-case optimal join algorithm. In Proceedings of International Conference on Database Theory (ICDT), pages 96-106, 2014.

[22] Mihalis Yannakakis. Algorithms for acyclic database schemes. In Proceedings of Very Large Data Bases (VLDB), pages 82-94, 1981.

Commons, 171 Second St, Suite 300, San Francisco, CA 94105, USA, or Eisenacher Strass

美国加利福尼亚州旧金山第二街171号300室，公共资源中心，或艾森纳赫大街