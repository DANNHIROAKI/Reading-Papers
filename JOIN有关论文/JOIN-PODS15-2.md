# Join Dependency Testing, Loomis-Whitney Join, and Triangle Enumeration

# 连接依赖测试、卢米斯 - 惠特尼连接与三角形枚举

Xiaocheng Hu Miao Qiao Yufei Tao

胡晓成 乔淼 陶宇飞

CUHK

Hong Kong

香港

## Abstract

## 摘要

In this paper, we revisit two fundamental problems in database theory. The first one is called join dependency (JD) testing, where we are given a relation $r$ and a JD,and need to determine whether the JD holds on $r$ . The second problem is called ${JD}$ existence testing, where we need to determine if there exists any non-trivial JD that holds on $r$ .

在本文中，我们重新探讨了数据库理论中的两个基本问题。第一个问题称为连接依赖（Join Dependency，JD）测试，给定一个关系 $r$ 和一个连接依赖，需要确定该连接依赖是否在 $r$ 上成立。第二个问题称为 ${JD}$ 存在性测试，需要确定是否存在任何非平凡的连接依赖在 $r$ 上成立。

We prove that JD testing is NP-hard even if the JD is defined only on binary relations (i.e.,each with only two attributes). Unless P $= \mathrm{{NP}}$ ,this result puts a negative answer to the question whether it is possible to efficiently test JDs defined exclusively on small (in terms of attribute number) relations. The question has been open since the classic NP-hard proof of Maier, Sagiv, and Yannakakis in JACM’81 which requires the JD to involve a relation of $\Omega \left( d\right)$ attributes,where $d$ is the number of attributes in $r$ .

我们证明了，即使连接依赖仅定义在二元关系（即每个关系只有两个属性）上，连接依赖测试也是 NP 难的。除非 P $= \mathrm{{NP}}$，否则这个结果对是否可以高效地测试仅定义在小（就属性数量而言）关系上的连接依赖这一问题给出了否定答案。自 Maier、Sagiv 和 Yannakakis 在 1981 年发表于《ACM 杂志》（JACM）的经典 NP 难证明以来，这个问题一直悬而未决，该证明要求连接依赖涉及一个具有 $\Omega \left( d\right)$ 个属性的关系，其中 $d$ 是 $r$ 中的属性数量。

For JD existence testing, the challenge is to minimize the computation cost because the problem is known to be solvable in polynomial time. We present a new algorithm for solving the problem I/O-efficiently in the external memory model. Our algorithm in fact settles the closely related Loomis-Whitney (LW) enumeration problem, and as a side product, achieves the optimal $\mathrm{I}/\mathrm{O}$ complexity for the triangle enumeration problem,improving a recent result of Pagh and Silvestri in PODS'14.

对于连接依赖存在性测试，挑战在于最小化计算成本，因为已知该问题可以在多项式时间内解决。我们提出了一种新的算法，用于在外部内存模型中以 I/O 高效的方式解决该问题。实际上，我们的算法解决了密切相关的卢米斯 - 惠特尼（Loomis - Whitney，LW）枚举问题，并且作为附带成果，实现了三角形枚举问题的最优 $\mathrm{I}/\mathrm{O}$ 复杂度，改进了 Pagh 和 Silvestri 在 2014 年发表于数据库系统原理研讨会（PODS）的最新结果。

## Categories and Subject Descriptors

## 类别与主题描述

F.2.2 [Analysis of algorithms and problem complexity]: Nonnumerical Algorithms and Problems; H.2.4 [Database Management]: Systems-Relational databases

F.2.2 [算法分析与问题复杂度]：非数值算法与问题；H.2.4 [数据库管理]：系统 - 关系数据库

## Keywords

## 关键词

Join Dependency; Loomis-Whitney Join; Triangle Enumeration

连接依赖；卢米斯 - 惠特尼连接；三角形枚举

## 1. INTRODUCTION

## 1. 引言

Given a relation $r$ of $d$ attributes,a key question in database theory is to ask if $r$ is decomposable,namely,whether $r$ can be projected onto a set $S$ of relations with less than $d$ attributes such that the natural join of those relations equals precisely $r$ . Intuitively,a yes answer to the question implies that $r$ contains a certain form of redundancy. Some of the redundancy may be removed by decomposing $r$ into the smaller (in terms of attribute number) relations in $S$ ,which can be joined together to restore $r$ whenever needed. A no answer, on the other hand, implies that the decomposition of $r$ based on $S$ will lose information,as far as natural join is concerned.

给定一个具有 $d$ 个属性的关系 $r$，数据库理论中的一个关键问题是询问 $r$ 是否可分解，即 $r$ 是否可以投影到一组属性数量少于 $d$ 的关系 $S$ 上，使得这些关系的自然连接恰好等于 $r$。直观地说，对这个问题的肯定回答意味着 $r$ 包含某种形式的冗余。通过将 $r$ 分解为 $S$ 中的较小（就属性数量而言）关系，可以去除一些冗余，这些关系可以在需要时连接起来恢复 $r$。另一方面，否定回答意味着就自然连接而言，基于 $S$ 对 $r$ 进行分解会丢失信息。

Join Dependency Testing. The above question (as well as its variants) has been extensively studied by resorting to the notion of join dependency (JD). To formalize the notion,let us refer to $d$ as the arity of $r$ . Denote by $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ the set of names of the $d$ attributes in $r.R$ is called the schema of $r$ . Sometimes we may denote $r$ as $r\left( R\right)$ or $r\left( {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right)$ to emphasize on its schema. Let $\left| r\right|$ represent the number of tuples in $r$ .

连接依赖测试。上述问题（及其变体）已通过连接依赖（JD）的概念进行了广泛研究。为了形式化这个概念，我们将 $d$ 称为 $r$ 的元数。用 $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 表示 $r.R$ 中 $d$ 个属性的名称集合，$R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 称为 $r$ 的模式。有时我们可以将 $r$ 表示为 $r\left( R\right)$ 或 $r\left( {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right)$ 以强调其模式。用 $\left| r\right|$ 表示 $r$ 中的元组数量。

A JD defined on $R$ is an expression of the form

定义在 $R$ 上的连接依赖是以下形式的表达式

$$
J =  \bowtie  \left\lbrack  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\rbrack  
$$

where (i) $m \geq  1$ ,(ii) each ${R}_{i}\left( {1 \leq  i \leq  m}\right)$ is a subset of $R$ that contains at least 2 attributes,and (iii) ${ \cup  }_{i = 1}^{m}{R}_{i} = R.J$ is non-trivial if none of ${R}_{1},\ldots ,{R}_{m}$ equals $R$ . The arity of $J$ is defined to be $\mathop{\max }\limits_{{i = 1}}^{m}\left| {R}_{i}\right|$ ,i.e.,the largest size of ${R}_{1},\ldots ,{R}_{m}$ . Clearly,the arity of a non-trivial $J$ is between 2 and $d - 1$ .

其中 (i) $m \geq  1$ ，(ii) 每个 ${R}_{i}\left( {1 \leq  i \leq  m}\right)$ 都是 $R$ 的一个子集，且该子集至少包含 2 个属性，并且 (iii) 如果 ${R}_{1},\ldots ,{R}_{m}$ 中没有一个等于 $R$ ，则 ${ \cup  }_{i = 1}^{m}{R}_{i} = R.J$ 是非平凡的。$J$ 的元数定义为 $\mathop{\max }\limits_{{i = 1}}^{m}\left| {R}_{i}\right|$ ，即 ${R}_{1},\ldots ,{R}_{m}$ 的最大规模。显然，一个非平凡的 $J$ 的元数介于 2 和 $d - 1$ 之间。

Relation $r$ is said to satisfy $J$ if

如果关系 $r$ 满足 $J$ ，则称其满足该条件

$$
r = {\pi }_{{R}_{1}}\left( r\right)  \bowtie  {\pi }_{{R}_{2}}\left( r\right)  \bowtie  \ldots  \bowtie  {\pi }_{{R}_{m}}\left( r\right) 
$$

where ${\pi }_{X}\left( r\right)$ denotes the projection of $r$ onto an attribute set $X$ , and $\bowtie$ represents natural join. We are ready to formally state the first two problems studied in this paper:

其中 ${\pi }_{X}\left( r\right)$ 表示 $r$ 在属性集 $X$ 上的投影，$\bowtie$ 表示自然连接。我们准备正式阐述本文研究的前两个问题：

Problem 1. [λ-JD Testing] Given a relation $r$ and a join dependency $J$ of arity at most $\lambda$ that is defined on the schema of $r$ ,we want to determine whether $r$ satisfies $J$ .

问题 1. [λ - 连接依赖测试] 给定一个关系 $r$ 和一个定义在 $r$ 的模式上且元数至多为 $\lambda$ 的连接依赖 $J$ ，我们要确定 $r$ 是否满足 $J$ 。

Problem 2. [JD Existence Testing] Given a relation $r$ ,we want to determine whether there is any non-trivial join dependency $J$ such that $r$ satisfies $J$ .

问题 2. [连接依赖存在性测试] 给定一个关系 $r$ ，我们要确定是否存在任何非平凡的连接依赖 $J$ ，使得 $r$ 满足 $J$ 。

Note the difference in the objectives of the above problems. Problem 1 aims to decide if $r$ can be decomposed according to a specific set $J$ of projections. On the other hand,Problem 2 aims to find out if there is any way to decompose $r$ at all.

注意上述问题目标的差异。问题 1 的目标是确定 $r$ 是否可以根据特定的投影集 $J$ 进行分解。另一方面，问题 2 的目标是找出是否存在任何分解 $r$ 的方法。

Computation Model. Our discussion on Problem 1 will concentrate on proving its NP-hardness. For this purpose, we will describe all our reductions in the standard RAM model.

计算模型。我们对问题 1 的讨论将集中在证明其 NP 难性上。为此，我们将在标准随机存取机（RAM）模型中描述所有的归约。

For Problem 2, which is known to be polynomial time solvable (as we will explain shortly), the main issue is to design fast algorithms. We will do so in the external memory (EM) model [2], which has become the de facto model for analyzing I/O-efficient algorithms. Under this model,a machine is equipped with $M$ words of memory, and an unbounded disk that has been formatted into blocks of $B$ words. It holds that $M \geq  {2B}$ . An $I/O$ operation exchanges a block of data between the disk and the memory. The cost of an algorithm is defined to be the number of I/Os performed. CPU calculation is for free.

对于问题 2，已知它可以在多项式时间内求解（我们将很快解释），主要问题是设计快速算法。我们将在外部存储器（EM）模型 [2] 中进行，该模型已成为分析 I/O 高效算法的事实上的模型。在该模型下，一台机器配备有 $M$ 个字的内存，以及一个无界磁盘，该磁盘已被格式化为 $B$ 个字的块。满足 $M \geq  {2B}$ 。一次 $I/O$ 操作在磁盘和内存之间交换一个数据块。算法的代价定义为执行的 I/O 次数。CPU 计算是免费的。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.PODS'15, May 31-June 4, 2015, Melbourne, Victoria, Australia. Copyright C 2015 ACM 978-1-4503-2757-2/15/05 ...\$15.00. Http://dx.doi.org/10.1145/2745754.2745768.

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不是为了盈利或商业利益而制作或分发，并且这些拷贝要带有此声明和第一页上的完整引用。必须尊重本作品中除美国计算机协会（ACM）之外的其他所有者的版权。允许进行带引用的摘要。否则，要进行复制、重新发布、发布到服务器或分发给列表，需要事先获得特定许可和/或支付费用。请向 permissions@acm.org 请求许可。PODS'15，2015 年 5 月 31 日 - 6 月 4 日，澳大利亚维多利亚州墨尔本。版权所有 © 2015 ACM 978 - 1 - 4503 - 2757 - 2/15/05 ... 15.00 美元。http://dx.doi.org/10.1145/2745754.2745768。

<!-- Footnote -->

---

To avoid rounding,we define ${\lg }_{x}y = \max \left\{  {1,{\log }_{x}y}\right\}$ ,and will describe all logarithms using ${\lg }_{x}y$ . In all cases,the value of an attribute is assumed to fit in a single word.

为避免舍入，我们定义 ${\lg }_{x}y = \max \left\{  {1,{\log }_{x}y}\right\}$ ，并将使用 ${\lg }_{x}y$ 来描述所有对数。在所有情况下，假设一个属性的值可以容纳在一个字中。

Loomis-Whitney Enumeration. As will be clear later, the JD existence-testing problem is closely related to the so-called Loomis-Whitney (LW) join. Let $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ be a set of $d$ attributes. For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,define ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ ,that is,removing ${A}_{i}$ from $R$ . Let ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ be $d$ relations such that ${r}_{i}\left( {1 \leq  i \leq  d}\right)$ has schema ${R}_{i}$ . Then,the natural join ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ is called an LW join. Note that the schema of the join result is $R$ .

卢米斯 - 惠特尼枚举法。正如后文将清晰展示的，JD存在性测试问题与所谓的卢米斯 - 惠特尼（Loomis - Whitney，LW）连接密切相关。设$R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$为一组包含$d$个属性的集合。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，定义${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$，即从$R$中移除${A}_{i}$。设${r}_{1},{r}_{2},\ldots ,{r}_{d}$为$d$个关系，使得${r}_{i}\left( {1 \leq  i \leq  d}\right)$的模式为${R}_{i}$。那么，自然连接${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$被称为LW连接。注意，连接结果的模式为$R$。

We will consider LW joins in the EM model, where traditionally a join must write out all the tuples in the result to the disk. However, the result size can be so huge that the number of I/Os for writing the result may (by far) overwhelm the cost of the join's rest execution. Furthermore, in some applications of LW joins (e.g., for solving Problem 2), it is not necessary to actually write the result tuples to the disk; instead, it suffices to witness each result tuple once in the memory.

我们将在EM模型中考虑LW连接，在该模型中，传统上连接操作必须将结果中的所有元组写入磁盘。然而，结果规模可能非常大，以至于写入结果的I/O次数可能（远远）超过连接操作其余执行部分的成本。此外，在LW连接的某些应用中（例如，用于解决问题2），实际上并不需要将结果元组写入磁盘；相反，只需在内存中见证每个结果元组一次即可。

Because of the above, we follow the approach of [14] by studying an enumerate version of the problem. Specifically, we are given a memory-resident routine emit $\left( \text{.}\right) {whichirequiresO}\left( 1\right)$ words to store. The parameter of the routine is a tuple $t$ of $d$ values $\left( {{a}_{1},\ldots ,{a}_{d}}\right)$ such that ${a}_{i}$ is in the domain of ${A}_{i}$ for each $i \in  \left\lbrack  {1,d}\right\rbrack$ . The routine simply sends out $t$ to an outbound socket with no I/O cost. Then, our problem can be formally stated as:

基于上述原因，我们采用文献[14]的方法，研究该问题的枚举版本。具体而言，我们有一个驻留在内存中的例程emit，它需要$\left( \text{.}\right) {whichirequiresO}\left( 1\right)$个字的存储空间。该例程的参数是一个包含$d$个值$\left( {{a}_{1},\ldots ,{a}_{d}}\right)$的元组$t$，使得对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，${a}_{i}$都在${A}_{i}$的域中。该例程只是将$t$发送到一个出站套接字，且无需I/O成本。那么，我们的问题可以正式表述为：

Problem 3. [LW Enumeration] Given relations ${r}_{1},\ldots ,{r}_{d}$ as defined earlier where $d \leq  M/2$ ,we want to invoke emit(t)once and exactly once for each tuple $t \in  {r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$ .

问题3. [LW枚举] 给定如前文所定义的关系${r}_{1},\ldots ,{r}_{d}$，其中$d \leq  M/2$，我们希望对每个元组$t \in  {r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$恰好调用一次emit(t)。

As a noteworthy remark, if an algorithm can solve the above problem in $x$ I/Os using $M - B$ words of memory,then it can also report the entire LW join result of $K$ tuples (i.e.,totally ${Kd}$ values) in $x + O\left( {{Kd}/B}\right)$ I/Os.

值得注意的是，如果一个算法能够使用$M - B$个字的内存，在$x$次I/O内解决上述问题，那么它也能够在$x + O\left( {{Kd}/B}\right)$次I/O内报告包含$K$个元组（即总共${Kd}$个值）的整个LW连接结果。

Triangle Enumeration. Besides being a stepping stone for Problem 2, LW enumeration has relevance to several other problems, among which the most prominent one is perhaps the triangle enumeration problem [14] due to its large variety of applications (see $\left\lbrack  {8,{14}}\right\rbrack$ and the references therein for an extensive summary).

三角形枚举。除了作为解决问题2的垫脚石之外，LW枚举还与其他几个问题相关，其中最突出的可能是三角形枚举问题[14]，因为它有各种各样的应用（有关详细总结，请参阅$\left\lbrack  {8,{14}}\right\rbrack$及其参考文献）。

Let $G = \left( {V,E}\right)$ be an undirected simple graph,where $V$ (or $E$ ) is the set of vertices (or edges, resp.). A triangle is defined as a clique of 3 vertices in $G$ . We are again given a memory-resident routine emit(.) that occupies $O\left( 1\right)$ words. This time,given a triangle $\Delta$ as its parameter,the routine sends out $\Delta$ to an outbound socket with no I/O cost (this implies that all the 3 edges of $\Delta$ must be in the memory at this moment). Then, the triangle enumeration problem can be formally stated as:

设$G = \left( {V,E}\right)$为一个无向简单图，其中$V$（或$E$）是顶点集（或边集）。三角形被定义为$G$中由3个顶点组成的团。我们再次有一个驻留在内存中的例程emit(.)，它占用$O\left( 1\right)$个字的存储空间。这次，给定一个三角形$\Delta$作为其参数，该例程将$\Delta$发送到一个出站套接字，且无需I/O成本（这意味着$\Delta$的所有3条边此时必须都在内存中）。那么，三角形枚举问题可以正式表述为：

Problem 4. [Triangle Enumeration] Given graph $G$ as defined earlier,we want to invoke emit $\left( \Delta \right)$ once and exactly once for each triangle $\Delta$ in $G$ .

问题4. [三角形枚举] 给定如前文所定义的图 $G$，我们希望针对 $G$ 中的每个三角形 $\Delta$ 恰好调用一次 emit $\left( \Delta \right)$。

Observe that this is merely a special instance of LW enumeration with $d = 3$ where ${r}_{1} = {r}_{2} = {r}_{3} = E$ (with some straightforward care to avoid emitting a triangle twice in no extra I/O cost).

注意，这仅仅是 $d = 3$ 下 LW 枚举的一个特殊实例，其中 ${r}_{1} = {r}_{2} = {r}_{3} = E$（通过一些简单的处理以避免在不产生额外 I/O 成本的情况下重复输出一个三角形）。

### 1.1 Previous Results

### 1.1 先前的研究成果

Join Dependency Testing. Beeri and Vardi [5] proved that $\lambda$ -JD testing (Problem 1) is NP-hard if $\lambda  = d - o\left( d\right)$ ; recall that $d$ is the number of attributes in the input relation $r$ . Maier,Sagiv,and Yannakakis [11] gave a stronger proof showing that $\lambda$ -JD testing is still NP-hard for $\lambda  = \Omega \left( d\right)$ (more specifically,roughly ${2d}/3$ ). In other words,(unless $\mathrm{P} = \mathrm{{NP}}$ ) no polynomial-time algorithm can exist to verify every $\mathrm{{JD}} \bowtie  \left\lbrack  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\rbrack$ on $r$ ,when one of ${R}_{1},\ldots ,{R}_{m}$ has $\Omega \left( d\right)$ attributes.

连接依赖测试。Beeri 和 Vardi [5] 证明，如果 $\lambda  = d - o\left( d\right)$，则 $\lambda$ -JD 测试（问题 1）是 NP 难的；回顾一下，$d$ 是输入关系 $r$ 中的属性数量。Maier、Sagiv 和 Yannakakis [11] 给出了一个更强的证明，表明对于 $\lambda  = \Omega \left( d\right)$（更具体地说，大约为 ${2d}/3$），$\lambda$ -JD 测试仍然是 NP 难的。换句话说，（除非 $\mathrm{P} = \mathrm{{NP}}$），当 ${R}_{1},\ldots ,{R}_{m}$ 中的一个具有 $\Omega \left( d\right)$ 个属性时，不存在多项式时间算法来验证 $r$ 上的每个 $\mathrm{{JD}} \bowtie  \left\lbrack  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\rbrack$。

However, the above result does not rule out the possibility of efficient testing when the JD has a small arity, namely, all of ${R}_{1},\ldots ,{R}_{m}$ have just a few attributes (e.g.,as few as just 2). Small-arity JDs are important because many relations in reality can eventually be losslessly decomposed into relations with small arities. By definition,for any ${\lambda }_{1} < {\lambda }_{2}$ ,the ${\lambda }_{1}$ -JD testing problem may only be easier than ${\lambda }_{2}$ -JD testing problem because an algorithm for the latter can be used to solve the former problem, but not the vice versa. The ultimate question, therefore, is whether 2-JD testing can be solved within polynomial time. Unfortunately, that the arity of $J$ being $\Omega \left( d\right)$ appears to be an inherent requirement in the reductions of $\left\lbrack  {5,{11}}\right\rbrack$ .

然而，上述结果并未排除在连接依赖（JD）的元数较小时进行高效测试的可能性，即所有的 ${R}_{1},\ldots ,{R}_{m}$ 只有少数几个属性（例如，少至 2 个）。小元数的 JD 很重要，因为现实中的许多关系最终都可以无损地分解为小元数的关系。根据定义，对于任何 ${\lambda }_{1} < {\lambda }_{2}$，${\lambda }_{1}$ -JD 测试问题可能比 ${\lambda }_{2}$ -JD 测试问题更容易，因为解决后者的算法可用于解决前者，但反之则不行。因此，最终的问题是 2 - JD 测试是否可以在多项式时间内解决。不幸的是，在 $\left\lbrack  {5,{11}}\right\rbrack$ 的归约中，$J$ 的元数为 $\Omega \left( d\right)$ 似乎是一个内在要求。

We note that a large body of beautiful theory has been developed on dependency inference, where the objective is to determine whether a target dependency can be inferred from a set $\sum$ of dependencies (see $\left\lbrack  {1,{10}}\right\rbrack$ for excellent guides into the literature). When the target dependency is a join dependency, the inference problem has been proven to be NP-hard in a variety of scenarios, most notably: (i) when $\sum$ contains one join dependency and a set of functional dependencies [5,11],(ii) when $\sum$ is a set of multi-valued dependencies [6],and (iii) when $\sum$ has one domain dependency and a set of functional dependencies [9]. The proofs of [5, 11] are essentially the same ones used to establish the NP-hardness of $\Omega \left( d\right)$ -JD testing,while those of $\left\lbrack  {6,9}\right\rbrack$ do not imply any conclusions on $\lambda$ -JD testing.

我们注意到，在依赖推理方面已经发展出了大量优美的理论，其目标是确定一个目标依赖是否可以从一组依赖 $\sum$ 中推导出来（有关该领域的优秀文献指南，请参阅 $\left\lbrack  {1,{10}}\right\rbrack$）。当目标依赖是一个连接依赖时，推理问题在多种场景下已被证明是 NP 难的，最显著的情况包括：（i）当 $\sum$ 包含一个连接依赖和一组函数依赖时 [5,11]；（ii）当 $\sum$ 是一组多值依赖时 [6]；（iii）当 $\sum$ 有一个域依赖和一组函数依赖时 [9]。[5, 11] 的证明本质上与用于证明 $\Omega \left( d\right)$ -JD 测试的 NP 难性的证明相同，而 $\left\lbrack  {6,9}\right\rbrack$ 的证明并未对 $\lambda$ -JD 测试得出任何结论。

JD Existence Testing and LW Join. There is an interesting connection between JD existence testing (Problem 2) and LW join. Let $r\left( R\right)$ be the input relation to Problem 2,where $R =$ $\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,define ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ , and ${r}_{i} = {\pi }_{{R}_{i}}\left( r\right)$ . Nicolas showed [13] that $r$ satisfies at least one non-trivial JD if and only if $r = {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ . In fact,since it is always true that $r \subseteq  {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ ,Problem 2 has an answer yes if and only if ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ returns exactly $\left| r\right|$ result tuples.

连接依赖（JD）存在性测试与轻量级（LW）连接。连接依赖存在性测试（问题2）和轻量级连接之间存在着有趣的联系。设$r\left( R\right)$为问题2的输入关系，其中$R =$ $\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，定义${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$和${r}_{i} = {\pi }_{{R}_{i}}\left( r\right)$。尼古拉斯（Nicolas）[13]证明了，当且仅当$r = {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$时，$r$满足至少一个非平凡的连接依赖。事实上，由于$r \subseteq  {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$总是成立，所以当且仅当${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$恰好返回$\left| r\right|$个结果元组时，问题2的答案为“是”。

Therefore, Problem 2 boils down to evaluating the result size of the LW join ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ . Atserias,Grohe,and Marx [4] showed that the result size can be as large as ${\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}}$ , where ${n}_{i} = \left| {r}_{i}\right|$ for each $i \in  \left\lbrack  {1,d}\right\rbrack$ . They also gave a RAM algorithm to compute the join result in $O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}}}\right.$ . $\left. {\mathop{\sum }\limits_{{i = 1}}^{n}{n}_{i}}\right)$ time. Since apparently ${n}_{i} \leq  n = \left| r\right| \left( {1 \leq  i \leq  d}\right)$ , it follows that their algorithm has running time $O\left( {{d}^{2} \cdot  {n}^{d/\left( {d - 1}\right) }}\right.$ . ${dn}) = O\left( {{d}^{3} \cdot  {n}^{2 + o\left( 1\right) }}\right)$ ,which in turn means that Problem 2 is solvable in polynomial time. Ngo et al. [12] designed a faster RAM algorithm to perform the LW join (hence, solving Problem 2) in $O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$ time.

因此，问题2可归结为计算轻量级连接${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$的结果规模。阿塞里亚斯（Atserias）、格罗赫（Grohe）和马克思（Marx）[4]证明了，结果规模可能高达${\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}}$，其中对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$有${n}_{i} = \left| {r}_{i}\right|$。他们还给出了一种随机存取机（RAM）算法，用于在$O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}}}\right.$ $\left. {\mathop{\sum }\limits_{{i = 1}}^{n}{n}_{i}}\right)$时间内计算连接结果。由于显然有${n}_{i} \leq  n = \left| r\right| \left( {1 \leq  i \leq  d}\right)$，因此他们的算法运行时间为$O\left( {{d}^{2} \cdot  {n}^{d/\left( {d - 1}\right) }}\right.$ ${dn}) = O\left( {{d}^{3} \cdot  {n}^{2 + o\left( 1\right) }}\right)$，这意味着问题2可以在多项式时间内解决。恩戈（Ngo）等人[12]设计了一种更快的随机存取机算法，用于在$O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$时间内执行轻量级连接（从而解决问题2）。

Problems 2 and 3 become much more challenging in external memory (EM). The algorithm of [12] (similarly, also the algorithm of [4]) is unaware of data blocking, relies heavily on hashing, and can entail up to $O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$ I/Os. When $d$ is small,this may be even worse than a naive generalized blocked-nested loop,whose I/O complexity for $d = O\left( 1\right)$ is $O\left( {{n}_{1}{n}_{2}\ldots {n}_{d}/\left( {{M}^{d - 1}B}\right) }\right)$ I/Os. Recall that $B$ and $M$ are the sizes of a disk block and memory, respectively.

问题2和问题3在外存（External Memory，EM）中变得更具挑战性。文献[12]中的算法（类似地，文献[4]中的算法也是如此）未考虑数据分块，严重依赖哈希，并且可能需要多达$O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$次I/O操作。当$d$较小时，这甚至可能比简单的广义分块嵌套循环算法更差，该算法对于$d = O\left( 1\right)$的I/O复杂度为$O\left( {{n}_{1}{n}_{2}\ldots {n}_{d}/\left( {{M}^{d - 1}B}\right) }\right)$次I/O操作。请记住，$B$和$M$分别是磁盘块和内存的大小。

Triangle Enumeration. Problem 4 has received a large amount of attention from the database and theory communities (see [8] for a survey). Recently, Pagh and Silvestri [14] solved the problem in EM with a randomized algorithm whose I/O cost is $O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$ expected,where $\left| E\right|$ is the number of edges in the input graph. They also presented a sophisticated de-randomization technique to convert their algorithm into a deterministic one that performs $O\left( {\frac{{\left| E\right| }^{1.5}}{\sqrt{M}B} \cdot  {\log }_{M/B}\frac{\left| E\right| }{B}}\right) \mathrm{I}/\mathrm{{Os}}$ . An I/O lower bound of $\Omega \left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$ has been independently developed in $\left\lbrack  {8,{14}}\right\rbrack$ on the witnessing class of algorithms.

三角形枚举。问题4受到了数据库和理论研究界的广泛关注（相关综述见文献[8]）。最近，帕格（Pagh）和西尔维斯特里（Silvestri）[14]使用一种随机算法在外存中解决了该问题，其I/O成本的期望值为$O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$，其中$\left| E\right|$是输入图中的边数。他们还提出了一种复杂的去随机化技术，将其算法转换为确定性算法，该确定性算法的I/O成本为$O\left( {\frac{{\left| E\right| }^{1.5}}{\sqrt{M}B} \cdot  {\log }_{M/B}\frac{\left| E\right| }{B}}\right) \mathrm{I}/\mathrm{{Os}}$。文献$\left\lbrack  {8,{14}}\right\rbrack$针对一类见证算法独立得出了$\Omega \left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$的I/O下界。

### 1.2 Our Results

### 1.2 我们的研究成果

Section 2 will establish our first main result:

第2节将阐述我们的第一个主要研究成果：

THEOREM 1. 2-JD testing is NP-hard.

定理1. 二元连接依赖（2-JD）测试是NP难问题。

The theorem officially puts a negative answer to the question whether a small-arity JD can be tested efficiently (remember that 2 is already the smallest possible arity). As a consequence, we know that Problem 2 is NP-hard for every value $\lambda  \in  \left\lbrack  {2,d - 1}\right\rbrack$ . Our proof is completely different from those of $\left\lbrack  {5,{11}}\right\rbrack$ ,and is based on a novel reduction from the Hamiltonian path problem.

该定理正式对小元数连接依赖能否被高效测试这一问题给出了否定答案（要知道2已经是可能的最小元数）。因此，我们知道对于任意值$\lambda  \in  \left\lbrack  {2,d - 1}\right\rbrack$，问题2都是NP难问题。我们的证明与文献$\left\lbrack  {5,{11}}\right\rbrack$中的证明完全不同，它基于从哈密顿路径问题进行的一种新颖归约。

Our second main result is an I/O-efficient algorithm for LW enumeration (Problem 3). Let ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ be the input relations; and set ${n}_{i} = \left| {r}_{i}\right|$ . In Section 3,we will prove:

我们的第二个主要研究成果是一种用于轻量级连接（LW）枚举（问题3）的I/O高效算法。设${r}_{1},{r}_{2},\ldots ,{r}_{d}$为输入关系；并令${n}_{i} = \left| {r}_{i}\right|$。在第3节中，我们将证明：

THEOREM 2. There is an EM algorithm that solves the LW enumeration problem with I/O complexity:

定理2. 存在一种外存算法，其I/O复杂度可解决轻量级连接枚举问题：

$$
O\left( {\operatorname{sort}\left\lbrack  {{d}^{3 + o\left( 1\right) }{\left( \frac{\mathop{\prod }\limits_{{i = 1}}^{d}{n}_{i}}{M}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right\rbrack  }\right) .
$$

where function sort(x)equals $\left( {x/B}\right) {\lg }_{M/B}\left( {x/B}\right)$ .

其中排序函数sort(x)等于$\left( {x/B}\right) {\lg }_{M/B}\left( {x/B}\right)$。

The main difficulty in obtaining the above theorem is that we cannot materialize the join result, because (as mentioned before) the result may have up to ${\left( {\Pi }_{i = 1}^{d}{n}_{i}\right) }^{1/\left( {d - 1}\right) }$ tuples such that writing them all to the disk may necessitate $\Omega \left( {\frac{d}{B}{\left( {\Pi }_{i = 1}^{d}{n}_{i}\right) }^{1/\left( {d - 1}\right) }}\right)$ I/Os. This is why the problem is more challenging in EM (than in RAM where it is affordable, in fact even compulsory, to list out the entire join result $\left\lbrack  {4,{12}}\right\rbrack  )$ . We overcome the challenge with a delicate piece of recursive machinery, and prove its efficiency through a non-trivial analysis.

得到上述定理的主要困难在于我们无法物化连接结果，因为（如前文所述）结果可能多达${\left( {\Pi }_{i = 1}^{d}{n}_{i}\right) }^{1/\left( {d - 1}\right) }$个元组，将它们全部写入磁盘可能需要$\Omega \left( {\frac{d}{B}{\left( {\Pi }_{i = 1}^{d}{n}_{i}\right) }^{1/\left( {d - 1}\right) }}\right)$次I/O操作。这就是为什么该问题在外存中比在随机存取存储器（RAM）中更具挑战性（在随机存取存储器中，列出整个连接结果$\left\lbrack  {4,{12}}\right\rbrack  )$不仅可行，实际上甚至是必要的）。我们通过一套精妙的递归机制克服了这一挑战，并通过非平凡的分析证明了其效率。

As our third main result, we prove in Section 4 an improved version of Theorem 2 for $d = 3$ :

作为我们的第三个主要结果，我们在第4节证明了针对$d = 3$的定理2的一个改进版本：

THEOREM 3. There is an EM algorithm that solves the LW enumeration problem of $d = 3$ with I/O complexity $O\left( {\frac{1}{B}\sqrt{\frac{{n}_{1}{n}_{2}{n}_{3}}{M}} + \operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$ .

定理3. 存在一种期望最大化（EM）算法，它能以输入/输出（I/O）复杂度$O\left( {\frac{1}{B}\sqrt{\frac{{n}_{1}{n}_{2}{n}_{3}}{M}} + \operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$解决$d = 3$的最小弱（LW）枚举问题。

By combining the above two theorems with the reduction from JD existence testing to LW enumeration described in Section 1.1, we obtain the first non-trivial algorithm for I/O-efficient JD existence testing (Problem 2):

通过将上述两个定理与1.1节中描述的从连接依赖（JD）存在性测试到最小弱（LW）枚举的归约相结合，我们得到了第一个用于高效输入/输出（I/O）的连接依赖（JD）存在性测试的非平凡算法（问题2）：

COROLLARY 1. Let $r\left( R\right)$ be the input relation to the JD existence testing problem,where $R = \left\{  {{A}_{1},\ldots ,{A}_{d}}\right\}$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,define ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ ,and ${n}_{i}$ as the number of tuples in ${\pi }_{{R}_{i}}\left( r\right)$ . Then:

推论1. 设$r\left( R\right)$为连接依赖（JD）存在性测试问题的输入关系，其中$R = \left\{  {{A}_{1},\ldots ,{A}_{d}}\right\}$。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，定义${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$，并将${n}_{i}$定义为${\pi }_{{R}_{i}}\left( r\right)$中的元组数量。那么：

- For $d > 3$ ,the problem can be solved with the I/O complexity in Theorem 2.

- 对于$d > 3$，该问题可以用定理2中的输入/输出（I/O）复杂度来解决。

- For $d = 3$ ,the I/O complexity can be improved to the one in Theorem 3.

- 对于$d = 3$，输入/输出（I/O）复杂度可以改进为定理3中的复杂度。

Finally,when ${n}_{1} = {n}_{2} = {n}_{3} = \left| E\right|$ ,Theorem 3 directly gives a new algorithm for triangle enumeration (Problem 4), noticing that $\operatorname{sort}\left( \left| E\right| \right)  = O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$ :

最后，当${n}_{1} = {n}_{2} = {n}_{3} = \left| E\right|$时，注意到$\operatorname{sort}\left( \left| E\right| \right)  = O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$，定理3直接给出了一个用于三角形枚举的新算法（问题4）：

COROLLARY 2. There is an algorithm that solves the triangle enumeration problem optimally in $O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right) I/{Os}$ .

推论2. 存在一种算法，它能以$O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right) I/{Os}$的复杂度最优地解决三角形枚举问题。

Our triangle enumeration algorithm is deterministic, and strictly improves that of [14] by a factor of $O\left( {{\lg }_{M/B}\left( {\left| E\right| /B}\right) }\right)$ . Furthermore, the algorithm belongs to the witnessing class [8], and is the first (deterministic algorithm) in this class achieving the optimal I/O complexity for all values of $M$ and $B$ .

我们的三角形枚举算法是确定性的，并且相对于文献[14]中的算法严格改进了$O\left( {{\lg }_{M/B}\left( {\left| E\right| /B}\right) }\right)$倍。此外，该算法属于见证类[8]，并且是该类中第一个（确定性算法）针对所有$M$和$B$的值都能达到最优输入/输出（I/O）复杂度的算法。

## 2. NP-HARDNESS OF 2-JD TESTING

## 2. 2 - 连接依赖（2 - JD）测试的NP难问题

This section will establish Theorem 1 with a reduction from the Hamiltonian path problem. Let $G = \left( {V,E}\right)$ be an undirected simple graph ${}^{1}$ with a vertex set $V$ and an edge set $E$ . Set $n = \left| V\right|$ and $m = \left| E\right|$ . Without loss of generality,assume that each vertex $v \in  V$ is uniquely identified by an integer id in $\left\lbrack  {1,n}\right\rbrack$ ,denoted as ${id}\left( v\right)$ . A path of length $\ell$ in $G$ is a sequence of $\ell$ vertices ${v}_{1},{v}_{2},\ldots ,{v}_{\ell }$ such that $E$ has an edge between ${v}_{i}$ and ${v}_{i + 1}$ for each $i \in  \left\lbrack  {1,\ell  - 1}\right\rbrack$ . The path is simple if no two vertices in the path are the same. A Hamiltonian path is a simple path in $G$ of length $n$ (such a path must pass each vertex in $V$ exactly once). Deciding whether $G$ has a Hamiltonian path is known to be NP-hard [7].

本节将通过从哈密顿路径问题进行归约来证明定理1。设$G = \left( {V,E}\right)$是一个无向简单图${}^{1}$，其顶点集为$V$，边集为$E$。设$n = \left| V\right|$和$m = \left| E\right|$。不失一般性，假设每个顶点$v \in  V$由$\left\lbrack  {1,n}\right\rbrack$中的一个整数ID唯一标识，记为${id}\left( v\right)$。$G$中长度为$\ell$的路径是一个由$\ell$个顶点组成的序列${v}_{1},{v}_{2},\ldots ,{v}_{\ell }$，使得对于每个$i \in  \left\lbrack  {1,\ell  - 1}\right\rbrack$，$E$中在${v}_{i}$和${v}_{i + 1}$之间存在一条边。如果路径中没有两个顶点相同，则该路径是简单路径。哈密顿路径是$G$中长度为$n$的简单路径（这样的路径必须恰好经过$V$中的每个顶点一次）。判定$G$是否存在哈密顿路径是已知的NP难问题[7]。

Let $R$ be a set of $n$ attributes: $\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{n}}\right\}$ . We will create $\left( \begin{array}{l} n \\  2 \end{array}\right)$ relations. Specifically,for each pair of $i,j$ such that $1 \leq  i <$ $j \leq  n$ ,we generate a relation ${r}_{i,j}$ with attributes ${A}_{i},{A}_{j}$ . The tuples in ${r}_{i,j}$ are determined as follows:

设 $R$ 为一组包含 $n$ 个属性的集合：$\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{n}}\right\}$。我们将创建 $\left( \begin{array}{l} n \\  2 \end{array}\right)$ 个关系。具体而言，对于每一对满足 $1 \leq  i <$ $j \leq  n$ 的 $i,j$，我们生成一个具有属性 ${A}_{i},{A}_{j}$ 的关系 ${r}_{i,j}$。${r}_{i,j}$ 中的元组确定方式如下：

- Case $j = i + 1$ : Initially, ${r}_{i,j}$ is empty. For each edge $E$ between vertices $u$ and $v$ ,we add two tuples to ${r}_{i,j}$ : $\left( {{id}\left( u\right) ,{id}\left( v\right) }\right)$ and $\left( {{id}\left( v\right) ,{id}\left( u\right) }\right)$ . In total, ${r}_{i,j}$ has ${2m}$ tuples.

- 情况 $j = i + 1$：初始时，${r}_{i,j}$ 为空。对于顶点 $u$ 和 $v$ 之间的每条边 $E$，我们向 ${r}_{i,j}$ 中添加两个元组：$\left( {{id}\left( u\right) ,{id}\left( v\right) }\right)$ 和 $\left( {{id}\left( v\right) ,{id}\left( u\right) }\right)$。总体而言，${r}_{i,j}$ 有 ${2m}$ 个元组。

- Case $j \geq  i + 2 : {r}_{i,j}$ contains $n\left( {n - 1}\right)$ tuples(x,y),for all possible integers $x,y$ such that $x \neq  y$ ,and $1 \leq  x,y \leq  n$ .

- 情况 $j \geq  i + 2 : {r}_{i,j}$ 包含 $n\left( {n - 1}\right)$ 个元组 (x, y)，其中 $x,y$ 为满足 $x \neq  y$ 且 $1 \leq  x,y \leq  n$ 的所有可能整数。

In general,the total number of tuples in the ${r}_{i,j}$ of all possible $i,j$ is $O\left( {{nm} + {n}^{4}}\right)  = O\left( {n}^{4}\right)$ .

一般来说，所有可能的 $i,j$ 的 ${r}_{i,j}$ 中的元组总数为 $O\left( {{nm} + {n}^{4}}\right)  = O\left( {n}^{4}\right)$。

Define:

定义：

CLIQUE $=$ the output of the natural join of all ${r}_{i,j}$

团（CLIQUE） $=$ 所有 ${r}_{i,j}$ 的自然连接的输出

$$
\left( {1 \leq  i < j \leq  n}\right) \text{.}
$$

For example,for $n = 3,$ CLIQUE $= {r}_{1,2} \bowtie  {r}_{1,3} \bowtie  {r}_{2,3}$ . In general,CLIQUE is a relation with schema $R$ .

例如，对于 $n = 3,$ 团（CLIQUE） $= {r}_{1,2} \bowtie  {r}_{1,3} \bowtie  {r}_{2,3}$。一般来说，团（CLIQUE）是一个具有模式 $R$ 的关系。

LEMMA 1. G has a Hamiltonian path if and only if CLIQUE is not empty.

引理 1. 图 G 有哈密顿路径当且仅当团（CLIQUE）不为空。

Proof. Direction If. Assuming that CLIQUE is not empty, next we show that $G$ has a Hamiltonian path. Let $\left( {{id}\left( {v}_{1}\right) ,{id}\left( {v}_{2}\right) ,\ldots }\right.$ , $\left. {{id}\left( {v}_{n}\right) }\right)$ be an arbitrary tuple in CLIQUE. It follows that:

证明。“如果”方向。假设团（CLIQUE）不为空，接下来我们证明 $G$ 有哈密顿路径。设 $\left( {{id}\left( {v}_{1}\right) ,{id}\left( {v}_{2}\right) ,\ldots }\right.$，$\left. {{id}\left( {v}_{n}\right) }\right)$ 为团（CLIQUE）中的任意一个元组。由此可得：

---

<!-- Footnote -->

${}^{1}$ Recall that a graph is simple if it has at most one edge between any two vertices.

${}^{1}$ 回顾一下，如果一个图在任意两个顶点之间最多只有一条边，那么这个图就是简单图。

<!-- Footnote -->

---

- For every $i \in  \left\lbrack  {1,n - 1}\right\rbrack  ,\left( {{id}\left( {v}_{i}\right) ,{id}\left( {v}_{i + 1}\right) }\right)$ is a tuple in ${r}_{i,i + 1}$ ,indicating that $E$ has an edge between ${v}_{i}$ and ${v}_{i + 1}$ .

- 对于每个 $i \in  \left\lbrack  {1,n - 1}\right\rbrack  ,\left( {{id}\left( {v}_{i}\right) ,{id}\left( {v}_{i + 1}\right) }\right)$ 是 ${r}_{i,i + 1}$ 中的一个元组，这表明 $E$ 在 ${v}_{i}$ 和 ${v}_{i + 1}$ 之间有一条边。

- For every $i,j$ such that $j \geq  i + 2,\left( {{id}\left( {v}_{i}\right) ,{id}\left( {v}_{j}\right) }\right)$ is a tuple in ${r}_{i,j}$ ,indicating that ${id}\left( {v}_{i}\right)  \neq  {id}\left( {v}_{j}\right)$ ,i.e., ${v}_{i} \neq  {v}_{j}$ .

- 对于每个满足 $j \geq  i + 2,\left( {{id}\left( {v}_{i}\right) ,{id}\left( {v}_{j}\right) }\right)$ 是 ${r}_{i,j}$ 中的一个元组的 $i,j$，这表明 ${id}\left( {v}_{i}\right)  \neq  {id}\left( {v}_{j}\right)$，即 ${v}_{i} \neq  {v}_{j}$。

We thus have found a Hamiltonian path ${v}_{1},{v}_{2},\ldots ,{v}_{n}$ in $G$ .

因此，我们在 $G$ 中找到了一条哈密顿路径 ${v}_{1},{v}_{2},\ldots ,{v}_{n}$。

Direction Only-If. Assuming that $G$ has a Hamiltonian path,next we show that CLIQUE is not empty. Let ${v}_{1},{v}_{2},\ldots ,{v}_{n}$ be any Hamiltonian path in $G$ . It is easy to verify that $\left( {{id}\left( {v}_{1}\right) ,{id}\left( {v}_{2}\right) }\right.$ , $\left. {\ldots ,{id}\left( {v}_{n}\right) }\right)$ must appear in CLIQUE.

仅当方向。假设$G$有一条哈密顿路径（Hamiltonian path），接下来我们证明团（CLIQUE）非空。设${v}_{1},{v}_{2},\ldots ,{v}_{n}$是$G$中的任意一条哈密顿路径。很容易验证$\left( {{id}\left( {v}_{1}\right) ,{id}\left( {v}_{2}\right) }\right.$、$\left. {\ldots ,{id}\left( {v}_{n}\right) }\right)$必定出现在团中。

For each pair of $i,j$ satisfying $1 \leq  i < j \leq  n$ ,define an attribute set ${R}_{i,j} = \left\{  {{A}_{i},{A}_{j}}\right\}$ . Denote by $J$ the JD that "corresponds to" CLIQUE, namely:

对于每一对满足$1 \leq  i < j \leq  n$的$i,j$，定义一个属性集${R}_{i,j} = \left\{  {{A}_{i},{A}_{j}}\right\}$。用$J$表示“对应于”团的连接依赖（JD，Join Dependency），即：

$$
J =  \bowtie  \left\lbrack  {{R}_{i,j},\forall i,j\text{ s.t. }1 \leq  i < j \leq  n}\right\rbrack  .
$$

For instance,for $n = 3,J =  \bowtie  \left\lbrack  {{R}_{1,2},{R}_{1,3},{R}_{2,3}}\right\rbrack$ . Note that $J$ has arity 2,and $R = { \cup  }_{i,j}{R}_{i,j}$ in general.

例如，对于$n = 3,J =  \bowtie  \left\lbrack  {{R}_{1,2},{R}_{1,3},{R}_{2,3}}\right\rbrack$。注意，$J$的元数为2，并且一般情况下为$R = { \cup  }_{i,j}{R}_{i,j}$。

Next,we will construct from $G$ a relation ${r}^{ * }$ of schema $R$ such that CLIQUE is empty if and only if ${r}^{ * }$ satisfies $J$ . The construction of ${r}^{ * }$ takes time polynomial to $n$ (and hence,also to $m$ because $m \leq  {n}^{2}$ ).

接下来，我们将从$G$构造一个模式为$R$的关系${r}^{ * }$，使得团为空当且仅当${r}^{ * }$满足$J$。${r}^{ * }$的构造时间是关于$n$的多项式时间（因此，由于$m \leq  {n}^{2}$，也是关于$m$的多项式时间）。

Initially, ${r}^{ * }$ is empty. For every tuple $t$ in every relation ${r}_{i,j}$ $\left( {1 \leq  i < j \leq  n}\right)$ ,we will insert a tuple ${t}^{\prime }$ into ${r}^{ * }$ . Recall that ${r}_{i,j}$ has schema $\left\{  {{A}_{i},{A}_{j}}\right\}$ . Suppose,without loss of generality,that $t = \left( {{a}_{i},{a}_{j}}\right)$ . Then, ${t}^{\prime }$ is determined as follows:

初始时，${r}^{ * }$为空。对于每个关系${r}_{i,j}$ $\left( {1 \leq  i < j \leq  n}\right)$中的每个元组$t$，我们将把一个元组${t}^{\prime }$插入到${r}^{ * }$中。回顾一下，${r}_{i,j}$的模式为$\left\{  {{A}_{i},{A}_{j}}\right\}$。不失一般性，假设$t = \left( {{a}_{i},{a}_{j}}\right)$。那么，${t}^{\prime }$的确定方式如下：

- ${t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}\left( {{t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack  }\right.$ is the value of $\left. {{t}^{\prime }\text{on attribute}{A}_{i}}\right)$

- ${t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}\left( {{t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack  }\right.$是$\left. {{t}^{\prime }\text{on attribute}{A}_{i}}\right)$的值

- ${t}^{\prime }\left\lbrack  {A}_{j}\right\rbrack   = {a}_{j}$

- For any $k \in  \left\lbrack  {1,n}\right\rbrack$ but $k \neq  i$ and $k \neq  j,{t}^{\prime }\left\lbrack  {A}_{k}\right\rbrack$ is set to a dummy value that appears only once in the whole ${r}^{ * }$ .

- 对于任意的$k \in  \left\lbrack  {1,n}\right\rbrack$，但$k \neq  i$且$k \neq  j,{t}^{\prime }\left\lbrack  {A}_{k}\right\rbrack$被设置为一个在整个${r}^{ * }$中仅出现一次的虚拟值。

Since (as mentioned before) there are $O\left( {n}^{4}\right)$ tuples in the ${r}_{i,j}$ of all $i,j$ ,we know that ${r}^{ * }$ has $O\left( {n}^{4}\right)$ tuples,and hence,can be built in $O\left( {n}^{5}\right)$ time.

由于（如前所述）所有$i,j$的${r}_{i,j}$中有$O\left( {n}^{4}\right)$个元组，我们知道${r}^{ * }$有$O\left( {n}^{4}\right)$个元组，因此可以在$O\left( {n}^{5}\right)$时间内构建。

LEMMA 2. CLIQUE is empty if and only if ${r}^{ * }$ satisfies $J$ .

引理2. 团为空当且仅当${r}^{ * }$满足$J$。

Proof. We first point out three facts:

证明。我们首先指出三个事实：

1. Every tuple in ${r}^{ * }$ has $n - 2$ dummy values.

1. ${r}^{ * }$中的每个元组都有$n - 2$个虚拟值。

2. Define ${r}_{i,j}^{ * } = {\pi }_{{A}_{i},{A}_{j}}\left( {r}^{ * }\right)$ for $i,j$ satisfying $1 \leq  i < j \leq  n$ . Clearly, ${r}_{i,j}^{ * }$ and ${r}_{i,j}$ share the same schema ${R}_{i,j}$ . It is easy to verify that ${r}_{i,j}$ is exactly the set of tuples in ${r}_{i,j}^{ * }$ that do not contain dummy values.

2. 为满足$1 \leq  i < j \leq  n$的$i,j$定义${r}_{i,j}^{ * } = {\pi }_{{A}_{i},{A}_{j}}\left( {r}^{ * }\right)$。显然，${r}_{i,j}^{ * }$和${r}_{i,j}$具有相同的模式${R}_{i,j}$。很容易验证，${r}_{i,j}$恰好是${r}_{i,j}^{ * }$中不包含虚拟值的元组集合。

3. Define:

3. 定义：

${\text{CLIQUE}}^{ * } =$ the output of the natural join of

${\text{CLIQUE}}^{ * } =$ 自然连接的输出

all ${r}_{i,j}^{ * }\left( {1 \leq  i < j \leq  n}\right)$ .

所有${r}_{i,j}^{ * }\left( {1 \leq  i < j \leq  n}\right)$。

Then, ${r}^{ * }$ satisfies $J$ if and only if ${r}^{ * } = {\text{CLIQUE}}^{ * }$ .

那么，当且仅当${r}^{ * } = {\text{CLIQUE}}^{ * }$时，${r}^{ * }$满足$J$。

Equipped with these facts, we now proceed to prove the lemma.

有了这些事实，我们现在开始证明引理。

Direction If. Assuming that ${r}^{ * }$ satisfies $J$ ,next we show that CLIQUE is empty. Suppose,on the contrary,that $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$ is a tuple in CLIQUE. Hence, $\left( {{a}_{i},{a}_{j}}\right)$ is a tuple in ${r}_{i,j}$ for any $i,j$ satisfying $1 \leq  i < j \leq  n$ . As neither ${a}_{i}$ nor ${a}_{j}$ is dummy, by Fact 2,we know that $\left( {{a}_{i},{a}_{j}}\right)$ belongs to ${r}_{i,j}^{ * }$ . It thus follows that $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$ is a tuple in CLIQUE ${}^{ * }$ . However,by Fact 1, $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$ cannot belong to ${r}^{ * }$ ,thus giving a contradiction against Fact 3.

“如果”方向。假设${r}^{ * }$满足$J$，接下来我们证明团（CLIQUE）为空。相反，假设$\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$是团中的一个元组。因此，对于任何满足$1 \leq  i < j \leq  n$的$i,j$，$\left( {{a}_{i},{a}_{j}}\right)$是${r}_{i,j}$中的一个元组。由于${a}_{i}$和${a}_{j}$都不是虚拟的，根据事实2，我们知道$\left( {{a}_{i},{a}_{j}}\right)$属于${r}_{i,j}^{ * }$。因此，$\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$是团${}^{ * }$中的一个元组。然而，根据事实1，$\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$不能属于${r}^{ * }$，这与事实3矛盾。

Direction Only-If. Assuming that CLIQUE is empty, next we show that ${r}^{ * }$ satisfies $J$ . Suppose,on the contrary,that ${r}^{ * }$ does not satisfy $J$ ,namely, ${r}^{ * } \neq  {\text{CLIQUE}}^{ * }$ (Fact 3). Let $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ be a tuple in CLIQUE ${}^{ * }$ but not in ${r}^{ * }$ . We distinguish two cases:

“仅当”方向。假设团为空，接下来我们证明${r}^{ * }$满足$J$。相反，假设${r}^{ * }$不满足$J$，即${r}^{ * } \neq  {\text{CLIQUE}}^{ * }$（事实3）。设$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$是团${}^{ * }$中但不在${r}^{ * }$中的一个元组。我们区分两种情况：

- Case 1: none of ${a}_{1}^{ * },\ldots ,{a}_{n}^{ * }$ is dummy. This means that,for any $i,j$ satisfying $1 \leq  i < j \leq  n,\left( {{a}_{i}^{ * },{a}_{j}^{ * }}\right)$ is a tuple in ${r}_{i,j}$ (Fact 2). Therefore, $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ must be a tuple in CLIQUE. contradicting the assumption that CLIQUE is empty.

- 情况1：${a}_{1}^{ * },\ldots ,{a}_{n}^{ * }$都不是虚拟的。这意味着，对于任何满足$1 \leq  i < j \leq  n,\left( {{a}_{i}^{ * },{a}_{j}^{ * }}\right)$的$i,j$，它是${r}_{i,j}$中的一个元组（事实2）。因此，$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$一定是团中的一个元组，这与团为空的假设矛盾。

- Case 2: ${a}_{k}^{ * }$ is dummy for at least one $k \in  \left\lbrack  {1,n}\right\rbrack$ . Since every dummy value appears exactly once in ${r}^{ * }$ ,we can identify a unique tuple ${t}^{ * }$ in ${r}^{ * }$ such that ${t}^{ * }\left\lbrack  {A}_{k}\right\rbrack   = {a}_{k}^{ * }$ . Next,we will show that ${t}^{ * }$ is precisely $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ ,thus contradicting the assumption that $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ is not in ${r}^{ * }$ ,which will then complete the proof.

- 情况2：对于至少一个 $k \in  \left\lbrack  {1,n}\right\rbrack$ 而言，${a}_{k}^{ * }$ 是哑元（dummy）。由于每个哑元值在 ${r}^{ * }$ 中恰好出现一次，我们可以在 ${r}^{ * }$ 中确定一个唯一的元组 ${t}^{ * }$，使得 ${t}^{ * }\left\lbrack  {A}_{k}\right\rbrack   = {a}_{k}^{ * }$ 成立。接下来，我们将证明 ${t}^{ * }$ 恰好就是 $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$，从而与 $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ 不在 ${r}^{ * }$ 中的假设相矛盾，进而完成证明。

Consider any $i$ such that $1 \leq  i < k$ . That $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ is in CLIQUE ${}^{ * }$ implies that $\left( {{a}_{i}^{ * },{a}_{k}^{ * }}\right)$ is in ${r}_{i,k}^{ * }$ . However, because in ${r}^{ * }$ the value ${a}_{k}^{ * }$ appears only in ${t}^{ * }$ ,it must hold that ${t}^{ * }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}^{ * }$ . By a similar argument,for any $j$ such that $k < j \leq  n$ ,we must have ${t}^{ * }\left\lbrack  {A}_{j}\right\rbrack   = {a}_{j}^{ * }$ . It thus follows that $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ is precisely ${t}^{ * }$ .

考虑任意满足 $1 \leq  i < k$ 的 $i$。$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ 在团（CLIQUE）${}^{ * }$ 中这一事实意味着 $\left( {{a}_{i}^{ * },{a}_{k}^{ * }}\right)$ 在 ${r}_{i,k}^{ * }$ 中。然而，由于在 ${r}^{ * }$ 中值 ${a}_{k}^{ * }$ 仅出现在 ${t}^{ * }$ 中，所以必然有 ${t}^{ * }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}^{ * }$ 成立。通过类似的论证，对于任意满足 $k < j \leq  n$ 的 $j$，我们必然有 ${t}^{ * }\left\lbrack  {A}_{j}\right\rbrack   = {a}_{j}^{ * }$ 成立。因此可以得出，$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ 恰好就是 ${t}^{ * }$。

From the above discussion, we know that any 2-JD testing algorithm can be used to check whether CLIQUE is empty (Lemma 2),and hence,can be used to check whether $G$ has a Hamiltonian path (Lemma 1). We thus conclude that 2-JD testing is NP-hard.

从上述讨论可知，我们知道任何2 - JD测试算法都可用于检查CLIQUE（团）是否为空（引理2），因此，也可用于检查$G$是否存在哈密顿路径（引理1）。由此我们得出结论：2 - JD测试是NP难问题。

### 3.LW ENUMERATION

### 3. 轻量级（LW）枚举

The discussion from the previous section has eliminated the hope of efficient JD testing no matter how small the JD arity is (unless $\mathrm{P} = \mathrm{{NP}}$ ). We therefore switch to the less stringent goal of JD existence testing (Problem 2). Based on the reduction described in Section 1.1, next we concentrate on LW enumeration as formulated in Problem 3, and will establish Theorem 2.

上一节的讨论消除了无论JD元数多小都能进行高效JD测试的希望（除非$\mathrm{P} = \mathrm{{NP}}$ ）。因此，我们转向要求没那么严格的JD存在性测试目标（问题2）。基于1.1节中描述的归约，接下来我们专注于问题3中阐述的轻量级（LW）枚举，并将证明定理2。

Let us recall a few basic definitions. We have a "global" set of attributes $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,let ${R}_{i} =$ $R \smallsetminus  \left\{  {A}_{i}\right\}$ . We are given relations ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ where ${r}_{i}\left( {1 \leq  i \leq  d}\right)$ has schema ${R}_{i}$ . The objective of LW enumeration is that,for every tuple $t$ in the result of ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ ,we should invoke emit(t)once and exactly once. We want to do so I/O-efficiently in the EM model,where $B$ and $M$ represent the sizes (in words) of a disk block and memory, respectively.

让我们回顾一些基本定义。我们有一个“全局”属性集$R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$ ，设${R}_{i} =$ $R \smallsetminus  \left\{  {A}_{i}\right\}$ 。给定关系${r}_{1},{r}_{2},\ldots ,{r}_{d}$ ，其中${r}_{i}\left( {1 \leq  i \leq  d}\right)$ 的模式为${R}_{i}$ 。轻量级（LW）枚举的目标是，对于${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 结果中的每个元组$t$ ，我们应该且仅应该调用一次emit(t)。我们希望在外部内存（EM）模型中以高效的I/O方式完成此操作，其中$B$ 和$M$ 分别表示磁盘块和内存的大小（以字为单位）。

For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,set ${n}_{i} = \left| {r}_{i}\right|$ ,and define $\operatorname{dom}\left( {A}_{i}\right)$ as the domain of attribute ${A}_{i}$ . Given a tuple $t$ and an attribute ${A}_{i}$ (in the schema of the relation containing $t$ ),we denote by $t\left\lbrack  {A}_{i}\right\rbrack$ the value of $t$ on ${A}_{i}$ . Furthermore,we assume that each of ${r}_{1},\ldots ,{r}_{d}$ is given in an array,but the $d$ arrays do not need to be consecutive.

对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$ ，设${n}_{i} = \left| {r}_{i}\right|$ ，并将$\operatorname{dom}\left( {A}_{i}\right)$ 定义为属性${A}_{i}$ 的域。给定一个元组$t$ 和一个属性${A}_{i}$ （在包含$t$ 的关系模式中），我们用$t\left\lbrack  {A}_{i}\right\rbrack$ 表示$t$ 在${A}_{i}$ 上的值。此外，我们假设每个${r}_{1},\ldots ,{r}_{d}$ 都以数组形式给出，但$d$ 个数组不需要是连续的。

### 3.1 Basic Algorithms

### 3.1 基本算法

Let us first deal with two scenarios under which LW enumeration is easier. The first situation arises when there is an ${n}_{i}$ (for some $i \in  \left\lbrack  {1,d}\right\rbrack$ ) satisfying ${n}_{i} = O\left( {M/d}\right)$ . In such a case,we call ${r}_{1} \bowtie$ ${r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ a small join.

让我们首先处理两种轻量级（LW）枚举较容易的场景。第一种情况是当存在一个${n}_{i}$ （对于某个$i \in  \left\lbrack  {1,d}\right\rbrack$ ）满足${n}_{i} = O\left( {M/d}\right)$ 时。在这种情况下，我们称${r}_{1} \bowtie$ ${r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 为小连接。

LEMMA 3. Given a small join, we can emit all its result tuples in $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) }\right)$ I/Os.

引理3. 给定一个小连接，我们可以在$O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) }\right)$ 次I/O操作内输出其所有结果元组。

Proof. See appendix.

证明：见附录。

The second scenario takes a bit more efforts to explain. In addition to ${r}_{1},\ldots ,{r}_{d}$ ,we accept two more input parameters:

第二种场景需要多花点力气来解释。除了${r}_{1},\ldots ,{r}_{d}$ 之外，我们还接受另外两个输入参数：

- an integer $H \in  \left\lbrack  {1,d}\right\rbrack$

- 一个整数$H \in  \left\lbrack  {1,d}\right\rbrack$

- a value $a \in  \operatorname{dom}\left( {A}_{H}\right)$ . It is required that $a$ should be the only value that appears in the ${A}_{H}$ attributes of ${r}_{1},\ldots ,{r}_{H - 1},{r}_{H + 1},\ldots ,{r}_{d}$ (recall that ${r}_{H}$ does not have $\left. {A}_{H}\right)$ . In such a case,we call ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ a point join.

- 一个值 $a \in  \operatorname{dom}\left( {A}_{H}\right)$ 。要求 $a$ 必须是 ${r}_{1},\ldots ,{r}_{H - 1},{r}_{H + 1},\ldots ,{r}_{d}$ 的 ${A}_{H}$ 属性中唯一出现的值（回想一下，${r}_{H}$ 没有 $\left. {A}_{H}\right)$ 。在这种情况下，我们称 ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 为点连接。

LEMMA 4. Given a point join, we can emit all its result tuples in $O\left( {d + \operatorname{sort}\left( {{d}^{2}{n}_{H} + d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \smallsetminus \{ H\} }}{n}_{i}}\right) }\right) I/{Os}$ .

引理4。给定一个点连接，我们可以在 $O\left( {d + \operatorname{sort}\left( {{d}^{2}{n}_{H} + d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \smallsetminus \{ H\} }}{n}_{i}}\right) }\right) I/{Os}$ 内输出其所有结果元组。

Proof. See appendix.

证明。见附录。

We will denote the algorithm in the above lemma as $\operatorname{PTJOIN}\left( {H,a,{r}_{1},{r}_{2},\ldots ,{r}_{d}}\right)$ .

我们将上述引理中的算法记为 $\operatorname{PTJOIN}\left( {H,a,{r}_{1},{r}_{2},\ldots ,{r}_{d}}\right)$ 。

### 3.2 The Full Algorithm

### 3.2 完整算法

This subsection presents an algorithm for solving the general LW enumeration problem. We will focus on ${n}_{1} > {2M}/d$ ; if ${n}_{1} \leq$ ${2M}/d$ ,simply apply Lemma 3 because this is a small-join scenario.

本小节提出一种用于解决一般LW枚举问题的算法。我们将关注 ${n}_{1} > {2M}/d$ ；如果 ${n}_{1} \leq$ ${2M}/d$ ，则直接应用引理3，因为这是一个小连接场景。

Define:

定义：

$$
U = {\left( \frac{\mathop{\prod }\limits_{{i = 1}}^{d}{n}_{i}}{M}\right) }^{\frac{1}{d - 1}} \tag{1}
$$

$$
{\tau }_{i} = \frac{{n}_{1}{n}_{2}\ldots {n}_{i}}{{\left( U \cdot  {d}^{\frac{1}{d - 1}}\right) }^{i - 1}}\text{ for each }i \in  \left\lbrack  {1,d}\right\rbrack  . \tag{2}
$$

Notice that ${\tau }_{1} = {n}_{1}$ and ${\tau }_{d} = M/d$ .

注意到 ${\tau }_{1} = {n}_{1}$ 和 ${\tau }_{d} = M/d$ 。

Our general algorithm is a recursive procedure $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,which has three requirements:

我们的通用算法是一个递归过程 $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ，它有三个要求：

- $h$ is an integer in $\left\lbrack  {1,d}\right\rbrack$ ;

- $h$ 是 $\left\lbrack  {1,d}\right\rbrack$ 中的一个整数；

- Each ${\rho }_{i}\left( {1 \leq  i \leq  d}\right)$ is a subset of the tuples in ${r}_{i}$ .

- 每个 ${\rho }_{i}\left( {1 \leq  i \leq  d}\right)$ 是 ${r}_{i}$ 中元组的一个子集。

- The size of ${\rho }_{1}$ satisfies:

- ${\rho }_{1}$ 的大小满足：

$$
\left| {\rho }_{1}\right|  \leq  {\tau }_{h} \tag{3}
$$

$\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ emits all result tuples in ${\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ . The original LW enumeration problem can be settled by calling $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ .

$\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 输出 ${\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ 中的所有结果元组。原始的LW枚举问题可以通过调用 $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 来解决。

#### 3.2.1 Case ${\tau }_{h} \leq  {2M}/d$

#### 3.2.1 情况 ${\tau }_{h} \leq  {2M}/d$

In this case,by the requirements of $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,it holds that $\left| {\rho }_{1}\right|  \leq  {\tau }_{h} = O\left( {M/d}\right)$ . Hence,we can directly apply the small-join algorithm in Lemma 3 to carry out the LW enumeration.

在这种情况下，根据 $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 的要求，有 $\left| {\rho }_{1}\right|  \leq  {\tau }_{h} = O\left( {M/d}\right)$ 。因此，我们可以直接应用引理3中的小连接算法来进行LW枚举。

#### 3.2.2 Case ${\tau }_{h} > {2M}/d$

#### 3.2.2 情况 ${\tau }_{h} > {2M}/d$

Denote by $H$ the smallest integer in $\left\lbrack  {h + 1,d}\right\rbrack$ such that ${\tau }_{H} <$ ${\tau }_{h}/2.H$ always exists because ${\tau }_{d} = M/d < {\tau }_{h}/2$ . Given a value $a \in  \operatorname{dom}\left( {A}_{H}\right)$ ,we define

用 $H$ 表示 $\left\lbrack  {h + 1,d}\right\rbrack$ 中满足 ${\tau }_{H} <$ 的最小整数。${\tau }_{h}/2.H$ 总是存在，因为 ${\tau }_{d} = M/d < {\tau }_{h}/2$。给定一个值 $a \in  \operatorname{dom}\left( {A}_{H}\right)$，我们定义

$$
\text{freq}\left( a\right)  = \text{number of tuples}t\text{in}{\rho }_{1}\text{with}t\left\lbrack  {A}_{H}\right\rbrack   = a\text{.}
$$

Now we introduce:

现在我们引入：

$$
\Phi  = \left\{  {a \in  \operatorname{dom}\left( {A}_{H}\right)  \mid  \operatorname{freq}\left( a\right)  > {\tau }_{H}/2}\right\}  . \tag{4}
$$

Let ${t}^{ * }$ be a result tuple of ${\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ . Conceptually, ${t}^{ * }$ is given a color: (i) red,if ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   \in  \Phi$ ,or (ii) blue,otherwise.

设 ${t}^{ * }$ 是 ${\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ 的一个结果元组。从概念上讲，${t}^{ * }$ 被赋予一种颜色：(i) 如果 ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   \in  \Phi$，则为红色；(ii) 否则为蓝色。

Our strategy is to emit red and blue tuples separately. Towards this purpose,for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we partition ${\rho }_{i}$ into:

我们的策略是分别输出红色和蓝色元组。为此，对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$，我们将 ${\rho }_{i}$ 划分为：

$$
{\rho }_{i}^{\text{red }} = \left\{  {\text{ tuple }t\text{ in }{\rho }_{i} \mid  t\left\lbrack  {A}_{H}\right\rbrack   \in  \Phi }\right\}  
$$

$$
{\rho }_{i}^{\text{blue }} = \left\{  {\text{ tuple }t\text{ in }{\rho }_{i} \mid  t\left\lbrack  {A}_{H}\right\rbrack   \notin  \Phi }\right\}  
$$

To emit red tuples,it suffices to consider ${\rho }_{1}^{\text{red }},\ldots ,{\rho }_{H - 1}^{\text{red }},{\rho }_{H}$ , ${\rho }_{H + 1}^{red},\ldots ,{\rho }_{d}^{red}$ . Likewise,to emit blue tuples,it suffices to consider ${\rho }_{1}^{\text{blue }},\ldots ,{\rho }_{H - 1}^{\text{blue }},{\rho }_{H},{\rho }_{H + 1}^{\text{blue }},\ldots ,{\rho }_{d}^{\text{blue }}$ . Next,we will elaborate on how to do so.

要输出红色元组，只需考虑 ${\rho }_{1}^{\text{red }},\ldots ,{\rho }_{H - 1}^{\text{red }},{\rho }_{H}$，${\rho }_{H + 1}^{red},\ldots ,{\rho }_{d}^{red}$。同样，要输出蓝色元组，只需考虑 ${\rho }_{1}^{\text{blue }},\ldots ,{\rho }_{H - 1}^{\text{blue }},{\rho }_{H},{\rho }_{H + 1}^{\text{blue }},\ldots ,{\rho }_{d}^{\text{blue }}$。接下来，我们将详细说明如何实现。

Remark. The set $\Phi$ ,as well as ${\rho }_{i}^{\text{red }}$ and ${\rho }_{i}^{\text{blue }}$ for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus$ $\{ H\}$ ,can be produced by sorting each ${\rho }_{i}$ on ${A}_{H}$ . More specifically, each element to be sorted is a tuple of $d - 1$ values where $d$ can be as large as $M/2$ . Using an EM string sorting algorithm of [3],all the sorting can be completed with $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \smallsetminus \{ H\} }}\left| {\rho }_{i}\right| }\right) }\right)$ I/Os in total.

注：集合 $\Phi$，以及对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus$ $\{ H\}$ 的 ${\rho }_{i}^{\text{red }}$ 和 ${\rho }_{i}^{\text{blue }}$，可以通过对每个 ${\rho }_{i}$ 按 ${A}_{H}$ 排序得到。更具体地说，每个待排序的元素是一个包含 $d - 1$ 个值的元组，其中 $d$ 最大可以达到 $M/2$。使用文献 [3] 中的一种外部内存（EM）字符串排序算法，所有排序总共可以用 $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \smallsetminus \{ H\} }}\left| {\rho }_{i}\right| }\right) }\right)$ 次输入/输出（I/O）操作完成。

Emitting Red Tuples. For every $a \in  \Phi$ ,we aim to emit the red tuples ${t}^{ * }$ with ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   = a$ separately. Define for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus$ $\{ H\}$ :

输出红色元组。对于每个 $a \in  \Phi$，我们的目标是分别输出满足 ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   = a$ 的红色元组 ${t}^{ * }$。对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus$ $\{ H\}$，定义：

$$
{\rho }_{i}^{\text{red }}\left\lbrack  a\right\rbrack   = \text{ set of tuples }t\text{ in }{\rho }_{i}^{\text{red }}\text{ with }t\left\lbrack  {A}_{H}\right\rbrack   = a.
$$

The tuples of ${\rho }_{i}^{\text{red }}\left\lbrack  a\right\rbrack$ are stored consecutively in the disk because we have sorted ${\rho }_{i}^{\text{red }}$ by ${A}_{H}$ earlier. All the red tuples ${t}^{ * }$ with ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   = a$ can be emitted by:

由于我们之前已经按 ${A}_{H}$ 对 ${\rho }_{i}^{\text{red }}$ 进行了排序，因此 ${\rho }_{i}^{\text{red }}\left\lbrack  a\right\rbrack$ 中的元组在磁盘上是连续存储的。所有满足 ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   = a$ 的红色元组 ${t}^{ * }$ 可以通过以下方式输出：

$\operatorname{PTJOIN}\left( {H,a,{\rho }_{1}^{\text{red }}\left\lbrack  a\right\rbrack  ,\ldots ,{\rho }_{H - 1}^{\text{red }}\left\lbrack  a\right\rbrack  ,{\rho }_{H},{\rho }_{H + 1}^{\text{red }}\left\lbrack  a\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{red }}\left\lbrack  a\right\rbrack  }\right)$ .

Emitting Blue Tuples. First,divide $\operatorname{dom}\left( {A}_{H}\right)$ into $q =$ $O\left( {1 + \left| {\rho }_{1}\right| /{\tau }_{H}}\right)$ disjoint intervals ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ with the following properties:

输出蓝色元组。首先，将 $\operatorname{dom}\left( {A}_{H}\right)$ 划分为 $q =$ $O\left( {1 + \left| {\rho }_{1}\right| /{\tau }_{H}}\right)$ 个不相交的区间 ${I}_{1},{I}_{2},\ldots ,{I}_{q}$，这些区间具有以下性质：

- ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ are in ascending order ${}^{2}$ .

- ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ 按升序排列 ${}^{2}$ 。

- For each $j \in  \left\lbrack  {1,q}\right\rbrack$ ,define:

- 对于每个 $j \in  \left\lbrack  {1,q}\right\rbrack$ ，定义：

${\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack   =$ set of tuples in ${\rho }_{1}^{\text{blue }}$ whose ${A}_{H}$ -values

${\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack   =$ 是 ${\rho }_{1}^{\text{blue }}$ 中那些 ${A}_{H}$ 值的元组集合

fall in ${I}_{j}$

落在 ${I}_{j}$ 范围内

If $j < q$ ,we require ${\tau }_{H}/2 \leq  \left| {{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right|  \leq  {\tau }_{H}$ . Regarding ${\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack$ ,we require $1 \leq  \left| {{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack  }\right|  \leq  {\tau }_{H}$ .

如果 $j < q$ ，我们要求 ${\tau }_{H}/2 \leq  \left| {{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right|  \leq  {\tau }_{H}$ 。关于 ${\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack$ ，我们要求 $1 \leq  \left| {{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack  }\right|  \leq  {\tau }_{H}$ 。

Because ${\rho }_{1}$ has been sorted by ${A}_{H}$ ,all the ${I}_{1},\ldots ,{I}_{q}$ and ${\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{1}\right\rbrack  ,\ldots ,{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack$ can all be obtained with one scan of ${\rho }_{1}$ .

因为 ${\rho }_{1}$ 已按 ${A}_{H}$ 排序，所以所有的 ${I}_{1},\ldots ,{I}_{q}$ 和 ${\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{1}\right\rbrack  ,\ldots ,{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack$ 都可以通过对 ${\rho }_{1}$ 进行一次扫描得到。

Next,for each $i \in  \left\lbrack  {2,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we produce for each $j \in  \left\lbrack  {1,q}\right\rbrack$ :

接下来，对于每个 $i \in  \left\lbrack  {2,d}\right\rbrack   \smallsetminus  \{ H\}$ ，我们为每个 $j \in  \left\lbrack  {1,q}\right\rbrack$ 生成：

${\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack   =$ set of tuples in ${\rho }_{i}^{\text{blue }}$ whose ${A}_{H}$ -values

${\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack   =$ 是 ${\rho }_{i}^{\text{blue }}$ 中那些 ${A}_{H}$ 值的元组集合

fall in ${I}_{j}$ .

落在 ${I}_{j}$ 范围内。

Because ${\rho }_{i}^{\text{blue }}$ has been sorted by ${A}_{H}$ ,all the ${\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{1}\right\rbrack  ,{\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{2}\right\rbrack$ , ..., ${\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack$ can be obtained by scanning synchronously ${\rho }_{i}^{\text{blue }}$ and $\left\{  {{I}_{1},\ldots ,{I}_{q}}\right\}$ once.

因为 ${\rho }_{i}^{\text{blue }}$ 已按 ${A}_{H}$ 排序，所以所有的 ${\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{1}\right\rbrack  ,{\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{2}\right\rbrack$ ，…， ${\rho }_{i}^{\text{blue }}\left\lbrack  {I}_{q}\right\rbrack$ 可以通过同步扫描 ${\rho }_{i}^{\text{blue }}$ 和 $\left\{  {{I}_{1},\ldots ,{I}_{q}}\right\}$ 一次得到。

Finally, to emit all the blue tuples, we simply recursively call our algorithm for each $j \in  \left\lbrack  {1,q}\right\rbrack$ :

最后，为了输出所有蓝色元组，我们只需对每个 $j \in  \left\lbrack  {1,q}\right\rbrack$ 递归调用我们的算法：

$\operatorname{JOIN}\left( {H,{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{H - 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,{\rho }_{H},{\rho }_{H + 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right)$ .

Note that the requirements for calling JOIN are fulfilled-in particular, $\left| {{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right|  \leq  {\tau }_{H}$ ,due to the way ${I}_{1},\ldots ,{I}_{q}$ were determined.

注意，调用 JOIN 的要求得到了满足 —— 特别是 $\left| {{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right|  \leq  {\tau }_{H}$ ，这是由于 ${I}_{1},\ldots ,{I}_{q}$ 的确定方式。

### 3.3 Analysis

### 3.3 分析

Define a sequence of integers as follows:

定义一个整数序列如下：

- ${h}_{1} = 1$ ;

- After ${h}_{i}$ has been defined $\left( {i \geq  1}\right)$ :

- 在 ${h}_{i}$ 被定义之后 $\left( {i \geq  1}\right)$ ：

- if ${\tau }_{{h}_{i}} > {2M}/d$ ,then define ${h}_{i + 1}$ as the smallest integer

- 如果 ${\tau }_{{h}_{i}} > {2M}/d$ ，则将 ${h}_{i + 1}$ 定义为最小的整数

in $\left\lbrack  {1 + {h}_{i},d}\right\rbrack$ satisfying ${\tau }_{{h}_{i + 1}} < {\tau }_{{h}_{i}}/2$ ;

在 $\left\lbrack  {1 + {h}_{i},d}\right\rbrack$ 中满足 ${\tau }_{{h}_{i + 1}} < {\tau }_{{h}_{i}}/2$ ;

- otherwise, ${h}_{i + 1}$ is undefined.

- 否则，${h}_{i + 1}$ 未定义。

Denote by $w$ the largest integer with ${h}_{w}$ defined.

用 $w$ 表示使 ${h}_{w}$ 有定义的最大整数。

Recall that our LW enumeration algorithm starts by calling the JOIN procedure with $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ ,which recursively makes

回顾一下，我们的LW枚举算法首先调用带有 $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的JOIN过程，该过程会递归地进行

$$
f\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  = \left\{  \begin{array}{ll} O\left( d\right) & \text{ if }\ell  = w \\  O\left( {d \cdot  {\mu }_{\ell }}\right)  + \mathop{\sum }\limits_{{j = 1}}^{q}f\left( {\ell  + 1,{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{{h}_{\ell  + 1} - 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,{\rho }_{{h}_{\ell  + 1}},{\rho }_{{h}_{\ell  + 1} + 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right) & \text{ if }\ell  < w \end{array}\right. 
$$

$$
g\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right) 
$$

$$
 = \left\{  \begin{array}{ll} d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| & \text{ if }\ell  = w \\  {d}^{2}{\mu }_{\ell }\left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right|  + \mathop{\sum }\limits_{{j = 1}}^{q}g\left( {\ell  + 1,{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{{h}_{\ell  + 1} - 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,{\rho }_{{h}_{\ell  + 1}},{\rho }_{{h}_{\ell  + 1} + 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right) & \text{ if }\ell  < w \end{array}\right. 
$$

---

<!-- Footnote -->

${}^{2}$ An interval $\left\lbrack  {x,y}\right\rbrack$ precedes another $\left\lbrack  {{x}^{\prime },{y}^{\prime }}\right\rbrack$ if $y < {x}^{\prime }$ .

${}^{2}$ 如果 $y < {x}^{\prime }$ ，则区间 $\left\lbrack  {x,y}\right\rbrack$ 先于另一个区间 $\left\lbrack  {{x}^{\prime },{y}^{\prime }}\right\rbrack$ 。

<!-- Footnote -->

---

Figure 1: Definitions of $f\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ and $g\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ subsequent calls to the same procedure. These calls form a tree $\mathcal{T}$ . Equipped with the sequence ${h}_{1},{h}_{2},\ldots ,{h}_{w}$ ,we can describe $\mathcal{T}$ in a more specific manner. Given a call $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,let us refer to the value of $h$ as the call’s axis. The initial call $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ has axis ${h}_{1} = 1$ . In general,an axis- ${h}_{i}\left( {i \in  \left\lbrack  {1,w - 1}\right\rbrack  }\right)$ call generates axis- ${h}_{i + 1}$ calls,and hence,parents those calls in $\mathcal{T}$ . Finally,all axis- ${h}_{w}$ calls are leaf nodes in $\mathcal{T}$ (recall that an axis- ${h}_{w}$ call simply invokes the small-join algorithm of Lemma 3). In other words, $\mathcal{T}$ has $w$ levels; and all the calls at level $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ have an identical axis ${h}_{\ell }$ .

图1：$f\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 和 $g\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 的定义 对同一过程的后续调用。这些调用形成一棵树 $\mathcal{T}$ 。借助序列 ${h}_{1},{h}_{2},\ldots ,{h}_{w}$ ，我们可以更具体地描述 $\mathcal{T}$ 。给定一个调用 $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ，我们将 $h$ 的值称为该调用的轴。初始调用 $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的轴为 ${h}_{1} = 1$ 。一般来说，轴为 ${h}_{i}\left( {i \in  \left\lbrack  {1,w - 1}\right\rbrack  }\right)$ 的调用会生成轴为 ${h}_{i + 1}$ 的调用，因此，在 $\mathcal{T}$ 中是这些调用的父节点。最后，所有轴为 ${h}_{w}$ 的调用都是 $\mathcal{T}$ 中的叶节点（回顾一下，轴为 ${h}_{w}$ 的调用只是调用引理3中的小连接算法）。换句话说，$\mathcal{T}$ 有 $w$ 层；并且第 $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ 层的所有调用都有相同的轴 ${h}_{\ell }$ 。

Given a level $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ ,define function $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ to be the number of I/Os performed by JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ . Our goal is to prove that $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ is as claimed in Theorem 2 .

给定一个层 $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ ，定义函数 $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 为JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 执行的I/O次数。我们的目标是证明 $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 如定理2中所声称的那样。

Case $\ell  = w$ . Lemma 3 immediately shows:

情况 $\ell  = w$ 。引理3立即表明：

$$
\operatorname{cost}\left( {w,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  = O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right) . \tag{5}
$$

Case $\ell  < w$ . Define for $\ell  \in  \left\lbrack  {1,w - 1}\right\rbrack$ :

情况 $\ell  < w$ 。对于 $\ell  \in  \left\lbrack  {1,w - 1}\right\rbrack$ 定义：

$$
{\mu }_{\ell } = 2{\tau }_{{h}_{\ell }}/{\tau }_{{h}_{\ell  + 1}}.
$$

Consider the set $\Phi$ defined in (4). Recall that for every $a \in  \Phi$ , freq $\left( a\right)  > {\tau }_{{h}_{\ell  + 1}}/2$ . Hence:

考虑 (4) 中定义的集合 $\Phi$ 。回顾一下，对于每个 $a \in  \Phi$ ，频率为 $\left( a\right)  > {\tau }_{{h}_{\ell  + 1}}/2$ 。因此：

$$
\left| \Phi \right|  < 2\left| {\rho }_{1}\right| /{\tau }_{{h}_{\ell  + 1}} \leq  2{\tau }_{{h}_{\ell }}/{\tau }_{{h}_{\ell  + 1}} = {\mu }_{\ell }.
$$

where the second inequality is due to (3).

其中第二个不等式是由于 (3) 。

For emitting red tuples, the cost is dominated by that of the point-join algorithm whose total I/O cost, by Lemma 4, is bounded by:

对于输出红色元组，成本主要由点连接算法的成本决定，根据引理4，该算法的总I/O成本上限为：

$$
O\left( {\mathop{\sum }\limits_{{a \in  \Phi }}\left( {d + \operatorname{sort}\left( {{d}^{2}\left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \left\{  {h}_{\ell  + 1}\right\}  }}\left| {{\rho }_{i}^{red}\left\lbrack  a\right\rbrack  }\right| }\right) }\right) }\right) 
$$

$$
 = O\left( {d\left| \Phi \right|  + \operatorname{sort}\left( {{d}^{2}\left| \Phi \right| \left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right) 
$$

$$
 = O\left( {d \cdot  {\mu }_{\ell } + \operatorname{sort}\left( {{d}^{2}{\mu }_{\ell }\left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right) . \tag{6}
$$

The cost of emitting blue tuples comes from recursion. Therefore, we can establish a recurrence:

输出蓝色元组的成本来自递归。因此，我们可以建立一个递推关系：

$$
\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right) 
$$

$$
 = \left( 6\right)  + \mathop{\sum }\limits_{{j = 1}}^{q}\operatorname{cost}\left( {\ell  + 1,{\rho }_{1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{{h}_{\ell  + 1} - 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,}\right. 
$$

$$
\left. {{\rho }_{{h}_{\ell  + 1}},{\rho }_{{h}_{\ell  + 1} + 1}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{blue }}\left\lbrack  {I}_{j}\right\rbrack  }\right) \text{.} \tag{7}
$$

Recall that $q$ is the number of disjoint intervals that $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ uses to divide $\operatorname{dom}\left( {A}_{\ell }\right)$ for blue tuple emission (see Section 3.2).

回顾一下，$q$ 是 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 用于对蓝色元组发射划分 $\operatorname{dom}\left( {A}_{\ell }\right)$ 的不相交区间的数量（见 3.2 节）。

The rest of the subsection is devoted to solving this non-conventional recurrence. Let functions $f\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ and $g\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ be as defined in Figure 1. The following proposition is fundamental:

本小节的其余部分致力于解决这个非常规递推问题。设函数 $f\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 和 $g\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 如图 1 所定义。以下命题是基础的：

Proposition 1. $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  = f\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  +$ $O\left( {\operatorname{sort}\left( {g\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right) }\right) }\right)$ .

命题 1. $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  = f\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  +$ $O\left( {\operatorname{sort}\left( {g\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right) }\right) }\right)$ 。

Proof. By the convexity of function $\operatorname{sort}\left( x\right)$ .

证明：根据函数 $\operatorname{sort}\left( x\right)$ 的凸性。

To prove Theorem 2, our target is to give an upper bound on $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  = f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  + O\left( {\operatorname{sort}\left( {g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right) }\right) }\right) .$

为了证明定理 2，我们的目标是给出 $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  = f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  + O\left( {\operatorname{sort}\left( {g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right) }\right) }\right) .$ 的一个上界

#### 3.3.1 Bounding $f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$

#### 3.3.1 界定 $f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$

Define ${m}_{\ell }$ as the total number of level- $\ell$ calls in $\mathcal{T}$ . Each level- $\ell$ call contributes $O\left( {d \cdot  {\mu }_{\ell }}\right)$ I/Os to $f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ (see Figure 1). ${}^{3}$ Hence:

将 ${m}_{\ell }$ 定义为 $\mathcal{T}$ 中第 $\ell$ 层调用的总数。每个第 $\ell$ 层调用会为 $f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 贡献 $O\left( {d \cdot  {\mu }_{\ell }}\right)$ 次输入/输出操作（见图 1）。${}^{3}$ 因此：

$$
f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  = \mathop{\sum }\limits_{{\ell  = 1}}^{w}O\left( {{m}_{\ell } \cdot  d \cdot  {\mu }_{\ell }}\right) . \tag{8}
$$

We say that a level- $\ell$ call $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ underflows if $\left| {\rho }_{1}\right|  < {\tau }_{{h}_{\ell }}/2$ ; otherwise,we say that it is ordinary. Consider all the calls $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ at level $\ell$ . The sets ${\rho }_{1}$ in the first parameters of those calls are disjoint. Hence, there can be at most $O\left( {{n}_{1}/{\tau }_{{h}_{\ell }}}\right)$ ordinary calls at level $\ell$ . Moreover,if $\ell  < w$ ,then a level- $\ell$ call creates at most one underflowing call at level $\ell  + 1$ . These facts indicate that,for each $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ :

我们称，如果 $\left| {\rho }_{1}\right|  < {\tau }_{{h}_{\ell }}/2$ ，则第 $\ell$ 层调用 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 发生下溢；否则，我们称其为普通调用。考虑第 $\ell$ 层的所有调用 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 。这些调用的第一个参数中的集合 ${\rho }_{1}$ 是不相交的。因此，第 $\ell$ 层最多可以有 $O\left( {{n}_{1}/{\tau }_{{h}_{\ell }}}\right)$ 个普通调用。此外，如果 $\ell  < w$ ，那么第 $\ell$ 层调用在第 $\ell  + 1$ 层最多创建一个发生下溢的调用。这些事实表明，对于每个 $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ ：

$$
{m}_{\ell } = O\left( {{m}_{\ell  - 1} + \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}}\right)  = O\left( {\mathop{\sum }\limits_{{i = 1}}^{\ell }\frac{{n}_{1}}{{\tau }_{{h}_{i}}}}\right)  = O\left( \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}\right) , \tag{9}
$$

where the second equality used ${m}_{1} = 1 = {n}_{1}/{\tau }_{{h}_{1}}$ ,and the last equality used the fact that ${\tau }_{{h}_{i}} > 2{\tau }_{{h}_{i + 1}}$ for every $i \in  \left\lbrack  {1,w - 1}\right\rbrack$ .

其中第二个等式使用了 ${m}_{1} = 1 = {n}_{1}/{\tau }_{{h}_{1}}$ ，最后一个等式使用了对于每个 $i \in  \left\lbrack  {1,w - 1}\right\rbrack$ 都有 ${\tau }_{{h}_{i}} > 2{\tau }_{{h}_{i + 1}}$ 这一事实。

Applying ${\tau }_{{h}_{w}} = M/d$ ,we get from (9):

应用 ${\tau }_{{h}_{w}} = M/d$ ，我们从 (9) 式可得：

$$
{m}_{w} = O\left( {d{n}_{1}/M}\right) .
$$

Moreover,for each $\ell  \in  \left\lbrack  {1,w - 1}\right\rbrack$ :

此外，对于每个 $\ell  \in  \left\lbrack  {1,w - 1}\right\rbrack$ ：

$$
{m}_{\ell }{\mu }_{\ell } = O\left( \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}\right) \frac{2{\tau }_{{h}_{\ell }}}{{\tau }_{{h}_{\ell  + 1}}} = O\left( \frac{{n}_{1}}{{\tau }_{{h}_{\ell  + 1}}}\right) .
$$

We can now derive from (8):

我们现在可以从 (8) 式推导出：

$$
f\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  = O\left( {\frac{{d}^{2}{n}_{1}}{M} + \mathop{\sum }\limits_{{\ell  = 1}}^{{w - 1}}\frac{d \cdot  {n}_{1}}{{\tau }_{{h}_{\ell  + 1}}}}\right) 
$$

$$
 = O\left( {\frac{{d}^{2}{n}_{1}}{M} + \frac{d{n}_{1}}{{\tau }_{{h}_{w}}}}\right) 
$$

$$
 = O\left( \frac{{d}^{2}{n}_{1}}{M}\right) . \tag{10}
$$

---

<!-- Footnote -->

${}^{3}$ Here define a boundary dummy ${\mu }_{w} = 1$ .

${}^{3}$ 这里定义一个边界虚拟变量 ${\mu }_{w} = 1$。

<!-- Footnote -->

---

#### 3.3.2 Bounding $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$

#### 3.3.2 界定 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$

Figure 1 shows that,in $\mathcal{T}$ ,each level- $\ell \left( {\ell  < w}\right)$ call $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ contributes ${d}^{2}{\mu }_{\ell }\left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right|$ to $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ . We can amortize the contribution onto the tuples in ${\rho }_{1},\ldots ,{\rho }_{d}$ ,such that:

图 1 表明，在 $\mathcal{T}$ 中，每个 $\ell \left( {\ell  < w}\right)$ 层调用 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 对 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的贡献为 ${d}^{2}{\mu }_{\ell }\left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right|$。我们可以将该贡献分摊到 ${\rho }_{1},\ldots ,{\rho }_{d}$ 中的元组上，使得：

- Each tuple in ${\rho }_{{h}_{\ell  + 1}}$ contributes ${d}^{2}{\mu }_{\ell }$ to $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ ;

- ${\rho }_{{h}_{\ell  + 1}}$ 中的每个元组对 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的贡献为 ${d}^{2}{\mu }_{\ell }$；

- Each tuple in any other relation ${\rho }_{i}\left( {i \neq  {h}_{\ell  + 1}}\right)$ contributes $d$ to $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ .

- 任何其他关系 ${\rho }_{i}\left( {i \neq  {h}_{\ell  + 1}}\right)$ 中的每个元组对 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的贡献为 $d$。

Similarly,for every level- $w$ call JOIN $\left( {{h}_{w},{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,each tuple in ${\rho }_{1},\ldots ,{\rho }_{d}$ contributes $d$ to $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ .

类似地，对于每个 $w$ 层调用 JOIN $\left( {{h}_{w},{\rho }_{1},\ldots ,{\rho }_{d}}\right)$，${\rho }_{1},\ldots ,{\rho }_{d}$ 中的每个元组对 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的贡献为 $d$。

Our strategy for bounding $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ is to sum up the largest possible contribution made by each individual tuple in the input relations ${r}_{1},\ldots ,{r}_{d}$ . For this purpose,given a value $i \in  \left\lbrack  {1,d}\right\rbrack$ ,we define ${L}_{i}$ as follows:

我们界定 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的策略是将输入关系 ${r}_{1},\ldots ,{r}_{d}$ 中每个单独元组可能做出的最大贡献相加。为此，给定一个值 $i \in  \left\lbrack  {1,d}\right\rbrack$，我们定义 ${L}_{i}$ 如下：

- ${L}_{1} = 0$ ;

- If $i \geq  2$ but no call in the entire $\mathcal{T}$ has axis $i$ ,then ${L}_{i} = 0$ ;

- 如果 $i \geq  2$ 但整个 $\mathcal{T}$ 中没有调用的轴为 $i$，则 ${L}_{i} = 0$；

- Otherwise,suppose that the level- $\ell$ calls of $T$ have axis ${h}_{\ell } =$ $i$ ; then we define ${L}_{i} = \ell  - 1$ .

- 否则，假设 $T$ 的 $\ell$ 层调用的轴为 ${h}_{\ell } =$ $i$；则我们定义 ${L}_{i} = \ell  - 1$。

Now,let us concentrate on a single tuple $t$ in an arbitrary input relation ${r}_{i}$ (for any $i \in  \left\lbrack  {1,d}\right\rbrack$ ). Consider a level- $\ell$ call $\left( {1 \leq  \ell  \leq  w}\right)$ $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ in $\mathcal{T}$ . We say that $t$ participates in the call if $t \in  {\rho }_{i}$ . If $t$ does not participate in the call,then $t$ contributes nothing to $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ . Otherwise,the contribution of $t$ depends on whether ${h}_{\ell  + 1}$ happens to be $i$ . As explained earlier,if ${h}_{\ell  + 1} = i$ , $t$ contributes ${d}^{2}{\mu }_{\ell }$ ,or else $t$ contributes $d$ .

现在，让我们关注任意输入关系 ${r}_{i}$ 中的单个元组 $t$（对于任意 $i \in  \left\lbrack  {1,d}\right\rbrack$）。考虑 $\mathcal{T}$ 中的一个 $\ell$ 层调用 $\left( {1 \leq  \ell  \leq  w}\right)$ $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$。如果 $t \in  {\rho }_{i}$，我们称 $t$ 参与了该调用。如果 $t$ 不参与该调用，则 $t$ 对 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 没有贡献。否则，$t$ 的贡献取决于 ${h}_{\ell  + 1}$ 是否恰好为 $i$。如前所述，如果 ${h}_{\ell  + 1} = i$，$t$ 的贡献为 ${d}^{2}{\mu }_{\ell }$，否则 $t$ 的贡献为 $d$。

Denote by ${\gamma }_{\ell }\left( t\right)$ the number of level- $\ell$ calls that $t$ participates in; specially,define ${\gamma }_{0}\left( t\right)  = 0$ . Then,the sequence ${L}_{1},{L}_{2},\ldots ,{L}_{d}$ defined earlier allows us to represent concisely the total contribution of $t$ as

用 ${\gamma }_{\ell }\left( t\right)$ 表示 $t$ 参与的 $\ell$ 级调用的数量；特别地，定义 ${\gamma }_{0}\left( t\right)  = 0$ 。那么，前面定义的序列 ${L}_{1},{L}_{2},\ldots ,{L}_{d}$ 使我们能够简洁地表示 $t$ 的总贡献为

$$
{\gamma }_{{L}_{i}}\left( t\right)  \cdot  {d}^{2}{\mu }_{{L}_{i}} + \mathop{\sum }\limits_{{\ell  \in  \left\lbrack  {1,w}\right\rbrack   \smallsetminus  {L}_{i}}}{\gamma }_{\ell }\left( t\right)  \cdot  d \tag{11}
$$

defining a boundary dummy value ${\mu }_{0} = 1$ .

定义一个边界虚拟值 ${\mu }_{0} = 1$ 。

LEMMA 5. If ${L}_{i} = 0$ ,then ${\gamma }_{\ell }\left( t\right)  \leq  1$ for all $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ . If ${L}_{i} \neq  0$ ,then

引理 5。如果 ${L}_{i} = 0$ ，那么对于所有的 $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ ，有 ${\gamma }_{\ell }\left( t\right)  \leq  1$ 。如果 ${L}_{i} \neq  0$ ，那么

$$
{\gamma }_{\ell }\left( t\right)  = \left\{  \begin{array}{ll} O\left( 1\right) & \text{ if }\ell  \in  \left\lbrack  {1,{L}_{i}}\right\rbrack  \\  O\left( {\mu }_{{L}_{i}}\right) & \text{ if }\ell  \in  \left\lbrack  {{L}_{i} + 1,w}\right\rbrack   \end{array}\right.  \tag{12}
$$

Proof. See appendix.

证明。见附录。

By applying the lemma to (11),we know that,in total, $t$ contributes to $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$

将该引理应用于 (11)，我们知道，总体而言，$t$ 对 $g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 有贡献

$$
O\left( {{d}^{2}{\mu }_{{L}_{i}} + w \cdot  {\mu }_{{L}_{i}} \cdot  d}\right)  = O\left( {{d}^{2}{\mu }_{{L}_{i}}}\right) .
$$

By summing up the contribution of all the tuples, we get:

通过对所有元组的贡献求和，我们得到：

$$
g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right) 
$$

$$
 = O\left( {\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \text{ s.t. }{L}_{i} \neq  0}}\mathop{\sum }\limits_{{t \in  {r}_{i}}}{d}^{2}{\mu }_{{L}_{i}} + \mathop{\sum }\limits_{{t \in  \left\lbrack  {1,d}\right\rbrack  \text{ s.t. }{L}_{i} = 0}}\mathop{\sum }\limits_{{t \in  {r}_{i}}}{d}^{2}}\right) 
$$

$$
 = O\left( {\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \text{ s.t. }{L}_{i} \neq  0}}{d}^{2}{\mu }_{{L}_{i}}{n}_{i} + \mathop{\sum }\limits_{{i = 1}}^{d}{d}^{2}{n}_{i}}\right) 
$$

$$
 = O\left( {\mathop{\sum }\limits_{{\ell  = 2}}^{w}{d}^{2}{\mu }_{\ell  - 1}{n}_{{h}_{\ell }} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) 
$$

where the last equality is due to the definition of ${L}_{i}$ .

其中最后一个等式是由于 ${L}_{i}$ 的定义。

It remains to bound ${\mu }_{\ell  - 1}{n}_{{h}_{\ell }}$ for each $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ . For this purpose, we prove:

接下来需要对每个 $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ 界定 ${\mu }_{\ell  - 1}{n}_{{h}_{\ell }}$ 。为此，我们证明：

LEMMA 6. ${\mu }_{\ell  - 1} = O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$ for each $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ .

引理 6。对于每个 $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ ，有 ${\mu }_{\ell  - 1} = O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$ 。

Proof. See appendix.

证明。见附录。

The lemma indicates that

引理表明

$$
g\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)  = O\left( {\mathop{\sum }\limits_{{\ell  = 2}}^{w}U{d}^{2 + \frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) 
$$

$$
 = O\left( {{d}^{3 + \frac{1}{d - 1}}U + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) .
$$

Combining the above equation with (1), (10), and Proposition 1, we now complete the whole proof of Theorem 2.

将上述方程与(1)、(10)以及命题1相结合，我们现在完成了定理2的整个证明。

### 4.A FASTER ALGORITHM FOR ARITY 3

### 4.A 三元的更快算法

The algorithm developed in the previous section solves the LW enumeration problem for any $d \leq  M/2$ . In this section,we focus on $d = 3$ ,and leverage intrinsic properties of this special instance to design a faster algorithm, which will establish Theorem 3 (and hence, also Corollaries 1 and 2). Specifically, the input consists of three relations: ${r}_{1}\left( {{A}_{2},{A}_{3}}\right) ,{r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ ,and ${r}_{3}\left( {{A}_{1},{A}_{2}}\right)$ ; and the goal is to emit all the tuples in the result of ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ .

上一节开发的算法解决了任意$d \leq  M/2$的LW枚举问题。在本节中，我们关注$d = 3$，并利用这个特殊实例的内在性质来设计一个更快的算法，该算法将证明定理3（因此，也能证明推论1和推论2）。具体来说，输入包括三个关系：${r}_{1}\left( {{A}_{2},{A}_{3}}\right) ,{r}_{2}\left( {{A}_{1},{A}_{3}}\right)$和${r}_{3}\left( {{A}_{1},{A}_{2}}\right)$；目标是输出${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$结果中的所有元组。

As before,for each $i \in  \left\lbrack  {1,3}\right\rbrack$ ,set ${n}_{i} = \left| {r}_{i}\right|$ ,and denote by $\operatorname{dom}\left( {A}_{i}\right)$ the domain of ${A}_{i}$ . Without loss of generality,we assume that ${n}_{1} \geq  {n}_{2} \geq  {n}_{3}$ .

和之前一样，对于每个$i \in  \left\lbrack  {1,3}\right\rbrack$，设${n}_{i} = \left| {r}_{i}\right|$，并用$\operatorname{dom}\left( {A}_{i}\right)$表示${A}_{i}$的定义域。不失一般性，我们假设${n}_{1} \geq  {n}_{2} \geq  {n}_{3}$。

### 4.1 Basic Algorithms

### 4.1 基本算法

## Let us start with:

## 让我们从以下内容开始：

LEMMA 7. If ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ and ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ have been sorted by ${A}_{3}$ ,the 3 -arity ${LW}$ enumeration problem can be solved in $O(1 +$ $\left. {\frac{\left( {{n}_{1} + {n}_{2}}\right) {n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$ .

引理7。如果${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$和${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$已按${A}_{3}$排序，那么三元${LW}$枚举问题可以在$O(1 +$ $\left. {\frac{\left( {{n}_{1} + {n}_{2}}\right) {n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$时间内解决。

Proof. If ${n}_{3} \leq  M$ ,we can achieve the purpose stated in the lemma using the small-join algorithm of Lemma 3 with straightforward modifications (e.g., apparently sorting is not required). When ${n}_{3} > M$ ,we simply chop ${r}_{3}$ into subsets of size $M$ ,and then repeat the above small-join algorithm $\left\lceil  {{n}_{3}/M}\right\rceil$ times.

证明。如果${n}_{3} \leq  M$，我们可以使用引理3的小连接算法进行直接修改（例如，显然不需要排序）来实现引理中所述的目的。当${n}_{3} > M$时，我们只需将${r}_{3}$分割成大小为$M$的子集，然后重复上述小连接算法$\left\lceil  {{n}_{3}/M}\right\rceil$次。

We call ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ an ${A}_{1}$ -point join if both conditions below are fulfilled:

如果满足以下两个条件，我们称${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$为${A}_{1}$点连接：

- all the ${A}_{1}$ values in ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ are the same;

- ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$中所有的${A}_{1}$值都相同；

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ and ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ are sorted by ${A}_{3}$ .

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$和${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$按${A}_{3}$排序。

LEMMA 8. Given an ${A}_{1}$ -point join,we can emit all its result tuples in $O\left( {1 + \frac{{n}_{1}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$ .

引理8。给定一个${A}_{1}$点连接，我们可以在$O\left( {1 + \frac{{n}_{1}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$时间内输出其所有结果元组。

Proof. We first obtain ${r}^{\prime }\left( {{A}_{1},{A}_{2},{A}_{3}}\right)  = {r}_{1} \bowtie  {r}_{2}$ ,and store all the tuples of ${r}^{\prime }$ into the disk. Since all the tuples in ${r}_{2}$ have the same ${A}_{1}$ -value,their ${A}_{3}$ -values must be distinct. Hence,each tuple in ${r}_{1}$ can be joined with at most one tuple in ${r}_{2}$ ,implying that $\left| {r}^{\prime }\right|  \leq  {n}_{1}$ . Utilizing the fact that ${r}_{1}$ and ${r}_{2}$ are both sorted on ${A}_{3},{r}^{\prime }$ can be produced by a synchronous scan over ${r}_{1}$ and ${r}_{2}$ in $O\left( {1 + \left( {{n}_{1} + {n}_{2}}\right) /B}\right)$ I/Os.

证明。我们首先得到 ${r}^{\prime }\left( {{A}_{1},{A}_{2},{A}_{3}}\right)  = {r}_{1} \bowtie  {r}_{2}$，并将 ${r}^{\prime }$ 的所有元组存储到磁盘中。由于 ${r}_{2}$ 中的所有元组具有相同的 ${A}_{1}$ 值，它们的 ${A}_{3}$ 值必定不同。因此，${r}_{1}$ 中的每个元组最多可以与 ${r}_{2}$ 中的一个元组进行连接，这意味着 $\left| {r}^{\prime }\right|  \leq  {n}_{1}$。利用 ${r}_{1}$ 和 ${r}_{2}$ 都按 ${A}_{3},{r}^{\prime }$ 排序这一事实，可以通过对 ${r}_{1}$ 和 ${r}_{2}$ 进行同步扫描，在 $O\left( {1 + \left( {{n}_{1} + {n}_{2}}\right) /B}\right)$ 次输入/输出（I/O）操作中生成 ${A}_{3},{r}^{\prime }$。

Then, we use the classic blocked nested loop (BNL) algorithm to perform the join ${r}^{\prime } \bowtie  {r}_{3}$ (which equals ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ ). The only difference is that, whenever BNL wants to write a block of $O\left( B\right)$ result tuples to the disk,we skip the write but simply emit those tuples. The BNL performs $O\left( {1 + \frac{\left| {r}^{\prime }\right| {n}_{3}}{MB} + \frac{{r}^{\prime } + {n}_{3}}{B}}\right)$ I/Os. The lemma thus follows.

然后，我们使用经典的块嵌套循环（BNL）算法来执行连接操作 ${r}^{\prime } \bowtie  {r}_{3}$（它等于 ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$）。唯一的区别是，每当 BNL 想要将 $O\left( B\right)$ 结果元组的一个块写入磁盘时，我们跳过写入操作，只是简单地输出这些元组。BNL 执行 $O\left( {1 + \frac{\left| {r}^{\prime }\right| {n}_{3}}{MB} + \frac{{r}^{\prime } + {n}_{3}}{B}}\right)$ 次输入/输出（I/O）操作。引理得证。

Symmetrically,we call ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ an ${A}_{2}$ -point join if

对称地，如果满足以下条件，我们称 ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ 为 ${A}_{2}$ 点连接

- all the ${A}_{2}$ values in ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ are the same.

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ 中的所有 ${A}_{2}$ 值都相同。

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ and ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ are sorted by ${A}_{3}$ .

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ 和 ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ 按 ${A}_{3}$ 排序。

LEMMA 9. Given an ${A}_{2}$ -point join,we can emit all its result tuples in $O\left( {1 + \frac{{n}_{2}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$ .

引理 9。给定一个 ${A}_{2}$ 点连接，我们可以在 $O\left( {1 + \frac{{n}_{2}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$ 中输出其所有结果元组。

Proof. Symmetric to Lemma 8.

证明。与引理 8 对称。

### 4.2 3-Arity LW Enumeration Algorithm

### 4.2 三元 LW 枚举算法

Next, we give our general algorithm for LW enumeration with $d = 3$ . We will focus on ${n}_{1} \geq  {n}_{2} \geq  {n}_{3} \geq  M$ ; otherwise,the algorithm in Lemma 7 already solves the problem in linear I/Os after sorting.

接下来，我们给出针对 $d = 3$ 的 LW 枚举通用算法。我们将关注 ${n}_{1} \geq  {n}_{2} \geq  {n}_{3} \geq  M$；否则，引理 7 中的算法在排序后已经可以在线性输入/输出（I/O）操作内解决该问题。

Set:

设置：

$$
{\theta }_{1} = \sqrt{\frac{{n}_{1}{n}_{3}M}{{n}_{2}}}\text{,and}{\theta }_{2} = \sqrt{\frac{{n}_{2}{n}_{3}M}{{n}_{1}}}\text{.} \tag{13}
$$

For values ${a}_{1} \in  \operatorname{dom}\left( {A}_{1}\right)$ and ${a}_{2} \in  \operatorname{dom}\left( {A}_{2}\right)$ ,define:

对于值 ${a}_{1} \in  \operatorname{dom}\left( {A}_{1}\right)$ 和 ${a}_{2} \in  \operatorname{dom}\left( {A}_{2}\right)$，定义：

freq $\left( {{a}_{1},{r}_{3}}\right)  =$ number of tuples $t$ in ${r}_{3}$ with $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$

频率 $\left( {{a}_{1},{r}_{3}}\right)  =$：${r}_{3}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ 的元组 $t$ 的数量

freq $\left( {{a}_{2},{r}_{3}}\right)  =$ number of tuples $t$ in ${r}_{3}$ with $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ .

频率 $\left( {{a}_{2},{r}_{3}}\right)  =$：${r}_{3}$ 中满足 $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ 的元组 $t$ 的数量。

Now we introduce:

现在我们引入：

$$
{\Phi }_{1} = \left\{  {{a}_{1} \in  \operatorname{dom}\left( {A}_{1}\right)  \mid  \operatorname{freq}\left( {{a}_{1},{r}_{3}}\right)  > {\theta }_{1}}\right\}  
$$

$$
{\Phi }_{2} = \left\{  {{a}_{2} \in  \operatorname{dom}\left( {A}_{2}\right)  \mid  \operatorname{freq}\left( {{a}_{2},{r}_{3}}\right)  > {\theta }_{2}}\right\}  .
$$

Let ${t}^{ * }$ be a result tuple of ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ . We can classify ${t}^{ * }$ into one of the following categories:

设${t}^{ * }$为${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$的一个结果元组。我们可以将${t}^{ * }$归为以下类别之一：

1. Red-red: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

1. 红 - 红：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$和${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

2. Red-blue: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$

2. 红 - 蓝：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$和${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$

3. Blue-red: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

3. 蓝 - 红：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$和${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

4. Blue-blue: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$ .

4. 蓝 - 蓝：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$和${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$。

We will emit each type of tuples separately, after a partitioning phase, as explained in the sequel.

如后文所述，在分区阶段之后，我们将分别输出每种类型的元组。

Partitioning ${r}_{3}$ . Define:

对${r}_{3}$进行分区。定义：

$$
{r}_{3}^{\text{red,red }} = \text{set of tuples}t\text{in}{r}_{3}\text{s.t.}t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}
$$

${r}_{3}^{\text{red,blue }} =$ set of tuples $t$ in ${r}_{3}$ s.t. $t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$

${r}_{3}^{\text{red,blue }} =$是${r}_{3}$中满足$t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$的元组$t$的集合

${r}_{3}^{\text{blue,red }} =$ set of tuples $t$ in ${r}_{3}$ s.t. $t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

${r}_{3}^{\text{blue,red }} =$是${r}_{3}$中满足$t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$的元组$t$的集合

${r}_{3}^{\text{blue,blue }} =$ set of tuples $t$ in ${r}_{3}$ s.t. $t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$

${r}_{3}^{\text{blue,blue }} =$是${r}_{3}$中满足$t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$的元组$t$的集合

$$
{r}_{3}^{\text{blue,} - } = {r}_{3}^{\text{blue,red }} \cup  {r}_{3}^{\text{blue,blue }}
$$

$$
{r}_{3}^{-,\text{ blue }} = {r}_{3}^{\text{red },\text{ blue }} \cup  {r}_{3}^{\text{blue },\text{ blue }}.
$$

Divide $\operatorname{dom}\left( {A}_{1}\right)$ into ${q}_{1} = O\left( {1 + {n}_{3}/{\theta }_{1}}\right)$ disjoint intervals ${I}_{1}^{1}$ , ${I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$ with the following properties:

将$\operatorname{dom}\left( {A}_{1}\right)$划分为${q}_{1} = O\left( {1 + {n}_{3}/{\theta }_{1}}\right)$个不相交的区间${I}_{1}^{1}$、${I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$，具有以下性质：

- ${I}_{1}^{1},{I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$ are in ascending order.

- ${I}_{1}^{1},{I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$按升序排列。

- For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack  ,{r}_{3}^{\text{blue,} - }$ has at most $2{\theta }_{1}$ tuples whose ${A}_{1}$ -values fall in ${I}_{j}^{1}$ .

- 对于每个$j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack  ,{r}_{3}^{\text{blue,} - }$，其${A}_{1}$值落在${I}_{j}^{1}$中的元组最多有$2{\theta }_{1}$个。

Similarly,we divide $\operatorname{dom}\left( {A}_{2}\right)$ into ${q}_{2} = O\left( {1 + {n}_{3}/{\theta }_{2}}\right)$ disjoint intervals ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ with the following properties:

类似地，我们将$\operatorname{dom}\left( {A}_{2}\right)$划分为${q}_{2} = O\left( {1 + {n}_{3}/{\theta }_{2}}\right)$个不相交的区间${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$，具有以下性质：

- ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ are in ascending order.

- ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$按升序排列。

- For each $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack  ,{r}_{3}^{-\text{,blue }}$ has at most $2{\theta }_{2}$ tuples whose ${A}_{2}$ -values fall in ${I}_{j}^{2}$ .

- 对于每个 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack  ,{r}_{3}^{-\text{,blue }}$，其 ${A}_{2}$ 值落在 ${I}_{j}^{2}$ 中的元组最多有 $2{\theta }_{2}$ 个。

We now define several partitions of ${r}_{3}$ :

我们现在定义 ${r}_{3}$ 的几个划分：

1. For each ${a}_{1} \in  {\Phi }_{1}$ and ${a}_{2} \in  {\Phi }_{2}$ :

1. 对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和 ${a}_{2} \in  {\Phi }_{2}$：

$$
{r}_{3}^{\text{red,red }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack   = \text{the (only) tuple}t\text{in}{r}_{3}^{\text{red,red }}\text{with}
$$

$$
t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}\text{and}t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}\text{.}
$$

2. For each ${a}_{1} \in  {\Phi }_{1}$ and $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ :

2. 对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$：

$$
{r}_{3}^{{red},\text{ blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack   = \text{set of tuples}t\text{in}{r}_{3}^{{red},\text{ blue }}\text{with}
$$

$$
t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}\text{and}t\left\lbrack  {A}_{2}\right\rbrack  \text{in}{I}_{j}^{2}\text{.}
$$

3. For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and ${a}_{2} \in  {\Phi }_{2}$ :

3. 对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和 ${a}_{2} \in  {\Phi }_{2}$：

$$
{r}_{3}^{\text{blue,red }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack   = \text{set of tuples}t\text{in}{r}_{3}^{\text{blue,red }}\text{with}
$$

$$
t\left\lbrack  {A}_{1}\right\rbrack  \text{in}{I}_{j}^{1}\text{and}t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}\text{.}
$$

4. For each ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and ${j}_{2} \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ :

4. 对于每个 ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和 ${j}_{2} \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$：

$$
{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack   = \text{set of tuples}t\text{in}{r}_{3}^{\text{blue,blue }}\text{with}
$$

$$
t\left\lbrack  {A}_{1}\right\rbrack  \text{in}{I}_{j}^{1}\text{and}t\left\lbrack  {A}_{2}\right\rbrack  \text{in}{I}_{j}^{2}\text{.}
$$

It is fundamental to produce all the above partitions with $O\left( {\operatorname{sort}\left( {n}_{3}\right) }\right)$ I/Os in total.

总共使用 $O\left( {\operatorname{sort}\left( {n}_{3}\right) }\right)$ 次输入/输出（I/O）操作来生成上述所有划分是至关重要的。

Partitioning ${r}_{1}$ and ${r}_{2}$ . Let:

对 ${r}_{1}$ 和 ${r}_{2}$ 进行划分。设：

$$
{r}_{1}^{\text{red }} = \text{ set of tuples }t\text{ in }{r}_{1}\text{ s.t. }t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}
$$

$$
{r}_{1}^{\text{blue }} = \text{ set of tuples }t\text{ in }{r}_{1}\text{ s.t. }t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}
$$

$$
{r}_{2}^{\text{red }} = \text{ set of tuples }t\text{ in }{r}_{2}\text{ s.t. }t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}
$$

$$
{r}_{2}^{\text{blue }} = \text{set of tuples}t\text{in}{r}_{2}\text{s.t.}t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}
$$

We now define several partitions of ${r}_{1}$ :

我们现在定义 ${r}_{1}$ 的几个划分：

1. For each ${a}_{2} \in  {\Phi }_{2}$ :

1. 对于每个 ${a}_{2} \in  {\Phi }_{2}$：

$$
{r}_{1}^{\text{red }}\left\lbrack  {a}_{2}\right\rbrack   = \text{ set of tuples }t\text{ in }{r}_{1}^{\text{red }}\text{ with }t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}.
$$

2. For each $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ :

2. 对于每个 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$：

$$
{r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack   = \text{ set of tuples }t\text{ in }{r}_{1}^{\text{blue }}\text{ with }t\left\lbrack  {A}_{2}\right\rbrack  \text{ in }{I}_{j}^{2}.
$$

Similarly,we define several partitions of ${r}_{2}$ :

类似地，我们定义 ${r}_{2}$ 的几个划分：

1. For each ${a}_{1} \in  {\Phi }_{1}$ :

1. 对于每个 ${a}_{1} \in  {\Phi }_{1}$：

$$
{r}_{2}^{red}\left\lbrack  {a}_{1}\right\rbrack   = \text{ set of tuples }t\text{ in }{r}_{2}^{red}\text{ with }t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}.
$$

2. For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ :

2. 对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$：

$$
{r}_{2}^{\text{blue }}\left\lbrack  {I}_{j}^{1}\right\rbrack   = \text{ set of tuples }t\text{ in }{r}_{2}^{\text{blue }}\text{ with }t\left\lbrack  {A}_{1}\right\rbrack  \text{ in }{I}_{j}^{1}.
$$

It is also fundamental to produce the above partitions using $O\left( {\operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$ I/Os in total. With the same cost,we make sure that all these partitions are sorted by ${A}_{3}$ .

同样，总共使用 $O\left( {\operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$ 次输入/输出（I/O）操作来生成上述划分也是至关重要的。以相同的成本，我们确保所有这些划分都按 ${A}_{3}$ 排序。

Emitting Red-Red Tuples. For each ${a}_{1} \in  {\Phi }_{1}$ and each ${a}_{2} \in$ ${\Phi }_{2}$ ,apply Lemma 7 to emit the result of ${r}_{1}^{\text{red }}\left\lbrack  {a}_{2}\right\rbrack   \bowtie  {r}_{2}^{\text{red }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie$ ${r}_{3}^{{red},{red}}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ .

输出红 - 红元组。对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和每个 ${a}_{2} \in$ ${\Phi }_{2}$，应用引理 7 来输出 ${r}_{1}^{\text{red }}\left\lbrack  {a}_{2}\right\rbrack   \bowtie  {r}_{2}^{\text{red }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie$ ${r}_{3}^{{red},{red}}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ 的结果。

Emitting Red-Blue Tuples. For each ${a}_{1} \in  {\Phi }_{1}$ and each $j \in$ $\left\lbrack  {1,{q}_{2}}\right\rbrack$ ,apply Lemma 8 to emit the result of the ${A}_{1}$ -point join ${r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{red }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie  {r}_{3}^{\text{red,blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  .$

发射红蓝元组。对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和每个 $j \in$ $\left\lbrack  {1,{q}_{2}}\right\rbrack$，应用引理 8 来发射 ${A}_{1}$ 点连接 ${r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{red }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie  {r}_{3}^{\text{red,blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  .$ 的结果

Emitting Blue-Red Tuples. For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and each ${a}_{2} \in$ ${\Phi }_{2}$ ,apply Lemma 9 to emit the result of the ${A}_{2}$ -point join ${r}_{1}^{red}\left\lbrack  {a}_{2}\right\rbrack$ $\bowtie  {r}_{2}^{\text{blue }}\left\lbrack  {I}_{j}^{1}\right\rbrack   \bowtie  {r}_{3}^{\text{blue,red }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack  .$

发射蓝红元组。对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和每个 ${a}_{2} \in$ ${\Phi }_{2}$，应用引理 9 来发射 ${A}_{2}$ 点连接 ${r}_{1}^{red}\left\lbrack  {a}_{2}\right\rbrack$ $\bowtie  {r}_{2}^{\text{blue }}\left\lbrack  {I}_{j}^{1}\right\rbrack   \bowtie  {r}_{3}^{\text{blue,red }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack  .$ 的结果

Emitting Blue-Blue Tuples. For each ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and each ${j}_{2} \in$ $\left\lbrack  {1,{q}_{2}}\right\rbrack$ ,apply Lemma 7 to emit the result of ${r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{blue }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack$ $\bowtie  {r}_{3}^{\text{blue },\text{ blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  .$

发射蓝蓝元组。对于每个 ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和每个 ${j}_{2} \in$ $\left\lbrack  {1,{q}_{2}}\right\rbrack$，应用引理 7 来发射 ${r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{blue }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack$ $\bowtie  {r}_{3}^{\text{blue },\text{ blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  .$ 的结果

### 4.3 Analysis

### 4.3 分析

We now analyze the algorithm of Section 4.2,assuming ${n}_{1} \geq$ ${n}_{2} \geq  {n}_{3} \geq  M$ . First,it should be clear that

我们现在分析 4.2 节的算法，假设 ${n}_{1} \geq$ ${n}_{2} \geq  {n}_{3} \geq  M$。首先，显然

$$
\left| {\Phi }_{1}\right|  \leq  \frac{{n}_{3}}{{\theta }_{1}} = \sqrt{\frac{{n}_{2}{n}_{3}}{{n}_{1}M}}
$$

$$
\left| {\Phi }_{2}\right|  \leq  \frac{{n}_{3}}{{\theta }_{2}} = \sqrt{\frac{{n}_{1}{n}_{3}}{{n}_{2}M}}
$$

$$
{q}_{1} = O\left( {1 + \frac{{n}_{3}}{{\theta }_{1}}}\right)  = O\left( {1 + \sqrt{\frac{{n}_{2}{n}_{3}}{{n}_{1}M}}}\right) 
$$

$$
{q}_{2} = O\left( {1 + \frac{{n}_{3}}{{\theta }_{2}}}\right)  = O\left( \sqrt{\frac{{n}_{1}{n}_{3}}{{n}_{2}M}}\right) .
$$

By Lemma 7, the cost of red-red emission is bounded by (remember that ${r}_{3}^{\text{red,red }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ has only 1 tuple):

根据引理 7，红红发射的成本有界于（记住 ${r}_{3}^{\text{red,red }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ 只有 1 个元组）：

$$
\mathop{\sum }\limits_{{{a}_{1},{a}_{2}}}O\left( {1 + \frac{\left| {{r}_{1}^{red}\left\lbrack  {a}_{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{red}\left\lbrack  {a}_{1}\right\rbrack  }\right| }{B}}\right) .
$$

$$
 = O\left( {\left| {\Phi }_{1}\right| \left| {\Phi }_{2}\right|  + \mathop{\sum }\limits_{{a}_{2}}\frac{\left| {{r}_{1}^{red}\left\lbrack  {a}_{2}\right\rbrack  }\right| \left| {\Phi }_{1}\right| }{B} + \mathop{\sum }\limits_{{a}_{1}}\frac{\left| {{r}_{2}^{red}\left\lbrack  {a}_{1}\right\rbrack  }\right| \left| {\Phi }_{2}\right| \left| \right| }{B}}\right) 
$$

$$
 = O\left( {\frac{{n}_{3}}{M} + \frac{{n}_{1}\left| {\Phi }_{1}\right| }{B} + \frac{{n}_{2}\left| {\Phi }_{2}\right| }{B}}\right)  = O\left( \frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}}\right) .
$$

By Lemma 8, the cost of red-blue emission is bounded by:

根据引理 8，红蓝发射的成本有界于：

$$
\mathop{\sum }\limits_{{{a}_{1},j}}O\left( {1 + \frac{\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| \left| {{r}_{3}^{\text{red,blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right| }{MB}}\right. 
$$

$$
\left. {+\frac{\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{red }}\left\lbrack  {a}_{1}\right\rbrack  }\right|  + \left| {{r}_{3}^{\text{red,blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right| }{B}}\right) \text{.}
$$

$$
 = O\left( {\left| {\Phi }_{1}\right| {q}_{2} + \mathop{\sum }\limits_{j}\frac{\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| \mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{red },\text{ blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right| }{MB}}\right. 
$$

$$
\left. {+\frac{\left| {\Phi }_{1}\right| \mathop{\sum }\limits_{j}\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| }{B} + \frac{{q}_{2}\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{2}^{\text{red }}\left\lbrack  {a}_{1}\right\rbrack  }\right| }{B} + \frac{{n}_{3}}{B}}\right) \text{.} \tag{14}
$$

Observe that $\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{red },\text{ blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right|$ is the total number of tuples in ${r}_{3}^{\text{red,blue }}$ whose ${A}_{2}$ -values fall in ${I}_{j}^{2}$ . By the way ${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ are constructed, we know:

观察可知，$\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{red },\text{ blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right|$ 是 ${r}_{3}^{\text{red,blue }}$ 中 ${A}_{2}$ 值落在 ${I}_{j}^{2}$ 内的元组总数。根据 ${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ 的构造方式，我们知道：

$$
\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{red,blue }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right|  \leq  2{\theta }_{2}.
$$

(14) is thus bounded by:

因此，(14) 有界于：

$$
O\left( {\frac{{n}_{3}}{M} + \mathop{\sum }\limits_{j}\frac{\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| {\theta }_{2}}{MB} + \frac{\left| {\Phi }_{1}\right| {n}_{1}}{B} + \frac{{q}_{2}{n}_{2}}{B} + \frac{{n}_{3}}{B}}\right) 
$$

$$
 = O\left( {\frac{{n}_{1}{\theta }_{2}}{MB} + \frac{\left| {\Phi }_{1}\right| {n}_{1}}{B} + \frac{{q}_{2}{n}_{2}}{B} + \frac{{n}_{3}}{B}}\right)  = O\left( \frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}}\right) .
$$

A similar argument shows that the cost of blue-red emission is bounded by $O\left( {\frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}} + \frac{{n}_{1}}{B}}\right)$ . Finally,by Lemma 7,the cost of blue-blue emission is bounded by:

类似的论证表明，蓝红发射的成本有界于 $O\left( {\frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}} + \frac{{n}_{1}}{B}}\right)$。最后，根据引理 7，蓝蓝发射的成本有界于：

$$
\mathop{\sum }\limits_{{{j}_{1},{j}_{2}}}O\left( {1 + \frac{\left( {\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{blue }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right| }\right) \left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right| }{MB}}\right. 
$$

$$
\left. {+\frac{\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{blue }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right|  + \left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right| }{B}}\right) \text{.} \tag{15}
$$

Let us analyze each term of (15) in turn. First:

让我们依次分析 (15) 中的每一项。首先：

$$
\mathop{\sum }\limits_{{{j}_{1},{j}_{2}}}\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| \left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right| 
$$

$$
 = \mathop{\sum }\limits_{{j}_{2}}\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| \mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  \tag{16}
$$

$\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|$ gives the number of tuples in ${r}_{3}^{\text{blue,blue }}$ whose ${A}_{2}$ -values fall in ${I}_{j}^{2}$ . By the way ${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ are constructed, we know:

$\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|$给出了${r}_{3}^{\text{blue,blue }}$中${A}_{2}$值落在${I}_{j}^{2}$内的元组数量。根据${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$的构造方式，我们可知：

$$
\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  \leq  2{\theta }_{2}.
$$

Therefore:

因此：

$$
\left( {16}\right)  = O\left( {{\theta }_{2}\mathop{\sum }\limits_{{j}_{2}}\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| }\right)  = O\left( {{n}_{1}{\theta }_{2}}\right) .
$$

Symmetrically, we have:

对称地，我们有：

$$
\mathop{\sum }\limits_{{{j}_{1},{j}_{2}}}\left| {{r}_{2}^{\text{blue }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right| \left| {{r}_{3}^{\text{blue,blue }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  = O\left( {{n}_{2}{\theta }_{1}}\right) .
$$

Thus, (15) is bounded by:

因此，(15)的上界为：

$$
O\left( {{q}_{1}{q}_{2} + \frac{{n}_{1}{\theta }_{2} + {n}_{2}{\theta }_{1}}{MB}}\right. 
$$

$$
 + \frac{{q}_{1}\mathop{\sum }\limits_{{j}_{2}}\left| {{r}_{1}^{\text{blue }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| }{B} + \frac{{q}_{2}\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{2}^{\text{blue }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right| }{B} + \frac{{n}_{3}}{B})
$$

$$
 = O\left( {{q}_{1}{q}_{2} + \frac{{n}_{1}{\theta }_{2} + {n}_{2}{\theta }_{1}}{MB} + \frac{{q}_{1}{n}_{1}}{B} + \frac{{q}_{2}{n}_{2}}{B} + \frac{{n}_{3}}{B}}\right) 
$$

$$
 = O\left( {\frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}} + \frac{{n}_{1}}{B}}\right) .
$$

As already mentioned in Section 4.2, the partitioning phase requires $O\left( {\operatorname{sort}\left( {\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) }\right)$ I/Os. We now complete the proof of Theorem 3.

正如4.2节所述，分区阶段需要$O\left( {\operatorname{sort}\left( {\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) }\right)$次输入/输出操作。现在我们完成定理3的证明。

## ACKNOWLEDGEMENTS

## 致谢

This work was supported in part by Grants GRF 4168/13 and GRF 142072/14 from HKRGC.

本研究部分得到了香港研究资助局（HKRGC）的GRF 4168/13和GRF 142072/14两项资助。

## 5. REFERENCES

## 5. 参考文献

[1] S. Abiteboul, R. Hull, and V. Vianu. Foundations of Databases. Addison-Wesley Publishing Company, 1995.

[2] A. Aggarwal and J. S. Vitter. The input/output complexity of sorting and related problems. CACM, 31(9):1116-1127, 1988.

[3] L. Arge, P. Ferragina, R. Grossi, and J. S. Vitter. On sorting strings in external memory (extended abstract). In STOC, pages 540-548, 1997.

[3] L. Arge、P. Ferragina、R. Grossi和J. S. Vitter。外部内存中字符串排序问题（扩展摘要）。收录于《ACM Symposium on Theory of Computing（STOC）》，第540 - 548页，1997年。

[4] A. Atserias, M. Grohe, and D. Marx. Size bounds and query plans for relational joins. SIAM J. of Comp., 42(4):1737-1767, 2013.

[4] A. Atserias、M. Grohe和D. Marx。关系连接的规模界限与查询计划。《工业与应用数学学会计算期刊（SIAM J. of Comp.）》，42(4):1737 - 1767，2013年。

[5] C. Beeri and M. Vardi. On the complexity of testing implications of data dependencies. Computer Science Report, Hebrew Univ, 1980.

[5] C. Beeri和M. Vardi。数据依赖蕴含关系测试的复杂度。计算机科学报告，希伯来大学，1980年。

[6] P. C. Fischer and D. Tsou. Whether a set of multivalued dependencies implies a join dependency is NP-hard. SIAM J. of Comp., 12(2):259-266, 1983.

[6] P. C. Fischer和D. Tsou。一组多值依赖是否蕴含一个连接依赖的问题是NP难问题。《工业与应用数学学会计算期刊（SIAM J. of Comp.）》，12(2):259 - 266，1983年。

[7] M. R. Garey and D. S. Johnson. Computers and Intractability: A Guide to the Theory of NP-Completeness. W. H. Freeman, 1979.

[7] M. R. Garey和D. S. Johnson。《计算机与难解性：NP完全性理论指南》。W. H. Freeman出版社，1979年。

[8] X. Hu, Y. Tao, and C.-W. Chung. I/O-efficient algorithms on triangle listing and counting. To appear in ACM TODS, 2014.

[8] X. Hu、Y. Tao和C.-W. Chung。三角形列举与计数的高效输入/输出算法。即将发表于《ACM Transactions on Database Systems（ACM TODS）》，2014年。

[9] P. C. Kanellakis. On the computational complexity of cardinality constraints in relational databases. IPL, 11(2):98-101, 1980.

[9] P. C. Kanellakis。关系数据库中基数约束的计算复杂度。《信息处理快报（IPL）》，11(2):98 - 101，1980年。

[10] D. Maier. The Theory of Relational Databases. Available Online at http://web.cecs.pdx.edu/ ~maier/TheoryBook/TRD.html, 1983.

[10] D. Maier。《关系数据库理论》。可在线访问：http://web.cecs.pdx.edu/ ~maier/TheoryBook/TRD.html，1983年。

[11] D. Maier, Y. Sagiv, and M. Yannakakis. On the complexity of testing implications of functional and join dependencies. JACM, 28(4):680-695, 1981.

[11] D. 迈尔（Maier）、Y. 萨吉夫（Sagiv）和 M. 亚纳卡基斯（Yannakakis）。关于函数依赖和连接依赖蕴含关系测试的复杂性。《美国计算机协会期刊》（JACM），28(4):680 - 695，1981 年。

[12] H. Q. Ngo, E. Porat, C. Ré, and A. Rudra. Worst-case optimal join algorithms: [extended abstract]. In PODS, pages 37-48, 2012.

[12] H. Q. 吴（Ngo）、E. 波拉特（Porat）、C. 雷（Ré）和 A. 鲁德拉（Rudra）。最坏情况最优连接算法：[扩展摘要]。收录于《数据库系统原理研讨会》（PODS），第 37 - 48 页，2012 年。

[13] J. Nicolas. Mutual dependencies and some results on undecomposable relations. In VLDB, pages 360-367, 1978.

[13] J. 尼古拉斯（Nicolas）。相互依赖关系及关于不可分解关系的一些结果。收录于《大型数据库会议》（VLDB），第 360 - 367 页，1978 年。

[14] R. Pagh and F. Silvestri. The input/output complexity of triangle enumeration. In PODS, pages 224-233, 2014.

[14] R. 帕格（Pagh）和 F. 西尔维斯特里（Silvestri）。三角形枚举的输入/输出复杂性。收录于《数据库系统原理研讨会》（PODS），第 224 - 233 页，2014 年。

## APPENDIX

## 附录

## Proof of Lemma 3

## 引理 3 的证明

Without loss of generality,suppose that ${r}_{1}$ has the smallest cardinality among all the input relations. Let us first assume that ${n}_{1} \leq  {cM}/d$ where $c$ is a sufficiently small constant so that ${r}_{1}$ can be kept in memory throughout the entire algorithm. With ${r}_{1}$ already in memory,we merge all the tuples of ${r}_{2},\ldots ,{r}_{d}$ into a set $L$ ,sorted by attribute ${A}_{1}$ . For each $a \in  \operatorname{dom}\left( {A}_{1}\right)$ ,let $L\left\lbrack  a\right\rbrack$ be the set of tuples in $L$ whose ${A}_{1}$ -values equal $a$ .

不失一般性，假设 ${r}_{1}$ 在所有输入关系中具有最小的基数。首先，我们假设 ${n}_{1} \leq  {cM}/d$，其中 $c$ 是一个足够小的常数，使得 ${r}_{1}$ 可以在整个算法过程中保存在内存中。由于 ${r}_{1}$ 已经在内存中，我们将 ${r}_{2},\ldots ,{r}_{d}$ 的所有元组合并到一个集合 $L$ 中，并按属性 ${A}_{1}$ 排序。对于每个 $a \in  \operatorname{dom}\left( {A}_{1}\right)$，令 $L\left\lbrack  a\right\rbrack$ 为 $L$ 中 ${A}_{1}$ 值等于 $a$ 的元组集合。

Next,for each $a \in  \operatorname{dom}\left( {A}_{1}\right)$ ,we use the procedure below to emit all the tuples ${t}^{ * }$ in the result of ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ such that ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   = a$ . First,initialize empty sets ${S}_{2},\ldots ,{S}_{d}$ in memory. Then,we process each tuple $t \in  L\left\lbrack  a\right\rbrack$ as follows. Suppose that $t$ originates from ${r}_{i}$ for some $i \in  \left\lbrack  {2,d}\right\rbrack$ . Check whether ${r}_{1}$ has a tuple ${t}^{\prime }$ satisfying

接下来，对于每个 $a \in  \operatorname{dom}\left( {A}_{1}\right)$，我们使用以下过程来输出 ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 结果中所有满足 ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   = a$ 的元组 ${t}^{ * }$。首先，在内存中初始化空集 ${S}_{2},\ldots ,{S}_{d}$。然后，我们按如下方式处理每个元组 $t \in  L\left\lbrack  a\right\rbrack$。假设 $t$ 来自某个 $i \in  \left\lbrack  {2,d}\right\rbrack$ 的 ${r}_{i}$。检查 ${r}_{1}$ 是否有一个元组 ${t}^{\prime }$ 满足

$$
{t}^{\prime }\left\lbrack  {A}_{j}\right\rbrack   = t\left\lbrack  {A}_{j}\right\rbrack  ,\;\forall j \in  \left\lbrack  {2,d}\right\rbrack   \smallsetminus  \{ i\} . \tag{17}
$$

If the answer is no, $t$ is discarded; otherwise,we add it to ${S}_{i}$ . Note that the checking happens in memory, and thus, entails no I/O. Having processed all the tuples of $L\left\lbrack  a\right\rbrack$ this way,we emit all the tuples in the result of ${r}_{1} \bowtie  {S}_{2} \bowtie  {S}_{3} \bowtie  \ldots  \bowtie  {S}_{d}$ (these are exactly the tuples in ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ whose ${A}_{1}$ -values equal $a$ ). The above tuple emission incurs no I/Os due to the following lemma.

如果答案是否定的，则丢弃 $t$；否则，将其添加到 ${S}_{i}$ 中。请注意，检查是在内存中进行的，因此不会产生输入/输出操作。以这种方式处理完 $L\left\lbrack  a\right\rbrack$ 的所有元组后，我们输出 ${r}_{1} \bowtie  {S}_{2} \bowtie  {S}_{3} \bowtie  \ldots  \bowtie  {S}_{d}$ 结果中的所有元组（这些元组恰好是 ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 中 ${A}_{1}$ 值等于 $a$ 的元组）。由于以下引理，上述元组输出不会产生输入/输出操作。

LEMMA 10. ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ fit in memory.

引理 10. ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ 可以放入内存中。

Proof. It is easy to show that $\left| {S}_{i}\right|  \leq  {n}_{1} \leq  {cM}/d$ for each $i \in  \left\lbrack  {2,d}\right\rbrack$ . A naive way to store ${S}_{i}$ takes $d\left| {S}_{i}\right|$ words,in which case we would need $\Omega \left( {dM}\right)$ words to store ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ ,exceeding the memory capacity $M$ .

证明。容易证明，对于每个$i \in  \left\lbrack  {2,d}\right\rbrack$，有$\left| {S}_{i}\right|  \leq  {n}_{1} \leq  {cM}/d$。一种简单的存储${S}_{i}$的方法需要$d\left| {S}_{i}\right|$个字，在这种情况下，我们需要$\Omega \left( {dM}\right)$个字来存储${r}_{1},{S}_{2},\ldots ,{S}_{d}$，这超出了内存容量$M$。

To remedy this issue,we store ${S}_{i}$ using only $\left| {S}_{i}\right|$ words as follows. Given a tuple $t \in  {S}_{i}$ ,we store a single integer that is the memory address ${}^{4}$ of the tuple ${t}^{\prime }$ in (17). This does not lose any information because we can recover $t$ by resorting to (17) and the fact that $t\left\lbrack  {A}_{1}\right\rbrack   = a$ .

为了解决这个问题，我们按如下方式仅用$\left| {S}_{i}\right|$个字来存储${S}_{i}$。给定一个元组$t \in  {S}_{i}$，我们存储一个整数，该整数是元组${t}^{\prime }$在(17)中的内存地址${}^{4}$。这不会丢失任何信息，因为我们可以借助(17)以及$t\left\lbrack  {A}_{1}\right\rbrack   = a$这一事实来恢复$t$。

Therefore, ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ can be represented in $O\left( {d \cdot  {n}_{1}}\right)$ words, which is smaller than $M$ when the constant $c$ is sufficiently small.

因此，${r}_{1},{S}_{2},\ldots ,{S}_{d}$可以用$O\left( {d \cdot  {n}_{1}}\right)$个字来表示，当常数$c$足够小时，$O\left( {d \cdot  {n}_{1}}\right)$小于$M$。

The overall cost of the algorithm is dominated by the cost of (i) merging ${r}_{2},\ldots ,{r}_{d}$ into $L$ ,which takes $O\left( {d + \left( {d/B}\right) \mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right)$ I/Os, and (ii) sorting $L$ ,which takes $O\left( {\operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right) }\right)$ I/Os,using a algorithm of [3] for string sorting in EM. Hence, the overall I/O complexity is as claimed in Theorem 2.

该算法的总体成本主要由以下两部分成本决定：(i) 将${r}_{2},\ldots ,{r}_{d}$合并到$L$中，这需要$O\left( {d + \left( {d/B}\right) \mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right)$次输入/输出操作；(ii) 对$L$进行排序，这需要$O\left( {\operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right) }\right)$次输入/输出操作，这里使用了文献[3]中用于外部内存（EM）字符串排序的算法。因此，总体输入/输出复杂度如定理2所述。

It remains to consider the case where ${n}_{1} > {cM}/d$ . We simply divide ${r}_{1}$ arbitrarily into $O\left( 1\right)$ subsets each with ${cM}/d$ tuples,and then apply the above algorithm to emit all the result tuples produced from each of the subsets.

还需要考虑${n}_{1} > {cM}/d$的情况。我们只需将${r}_{1}$任意划分为$O\left( 1\right)$个子集，每个子集包含${cM}/d$个元组，然后将上述算法应用于每个子集，以输出所有结果元组。

## Proof of Lemma 4

## 引理4的证明

For each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,define ${X}_{i} = {R}_{i} \cap  {R}_{H}$ (i.e., ${X}_{i}$ includes all the attributes in $R$ except ${A}_{i}$ and ${A}_{H}$ ).

对于每个$i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$，定义${X}_{i} = {R}_{i} \cap  {R}_{H}$（即${X}_{i}$包含$R$中除${A}_{i}$和${A}_{H}$之外的所有属性）。

In ascending order of $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we invoke the procedure below to process ${r}_{i}$ and ${r}_{H}$ ,which continuously removes some tuples from ${r}_{H}$ . First,sort ${r}_{i}$ and ${r}_{i}$ and ${r}_{H}$ by ${X}_{i}$ ,respectively. Then, synchronously scan ${r}_{i}$ and ${r}_{H}$ according to the sorted order. For each tuple $t$ in ${r}_{H}$ ,we check during the scan whether ${r}_{i}$ has a tuple ${t}^{\prime }$ that has the same values as $t$ on all the attributes in ${X}_{i}$ The sorted order ensures that if ${t}^{\prime }$ exists,then $t$ and ${t}^{\prime }$ must appear consecutively during the synchronous scan ${}^{5}$ . If ${t}^{\prime }$ exists, $t$ is kept in ${r}_{H}$ ; otherwise,we discard $t$ from ${r}_{H}$ ( $t$ cannot produce any tuple in ${r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$ ).

按照$i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$的升序，我们调用以下过程来处理${r}_{i}$和${r}_{H}$，该过程会不断从${r}_{H}$中移除一些元组。首先，分别按照${X}_{i}$对${r}_{i}$和${r}_{H}$进行排序。然后，根据排序顺序同步扫描${r}_{i}$和${r}_{H}$。对于${r}_{H}$中的每个元组$t$，在扫描过程中，我们检查${r}_{i}$中是否存在一个元组${t}^{\prime }$，该元组在${X}_{i}$中的所有属性上与$t$具有相同的值。排序顺序确保如果${t}^{\prime }$存在，那么$t$和${t}^{\prime }$在同步扫描${}^{5}$期间必定连续出现。如果${t}^{\prime }$存在，则将$t$保留在${r}_{H}$中；否则，我们从${r}_{H}$中丢弃$t$（$t$无法在${r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$中产生任何元组）。

After the above procedure has finished through all $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus$ $\{ H\}$ ,we know that every tuple $t$ remaining in ${r}_{H}$ must produce exactly one result tuple ${t}^{\prime }$ in ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ . Clearly, ${t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack   =$ $t\left\lbrack  {A}_{i}\right\rbrack$ for all $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,and (by definition of point join) ${t}^{\prime }\left\lbrack  {A}_{H}\right\rbrack   = a$ . Therefore,we can emit all such ${t}^{\prime }$ with one more scan of the (current) ${r}_{H}$ .

在上述过程对所有$i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus$ $\{ H\}$执行完毕后，我们知道${r}_{H}$中剩余的每个元组$t$必定会在${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$中恰好产生一个结果元组${t}^{\prime }$。显然，对于所有$i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$，有${t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack   =$ $t\left\lbrack  {A}_{i}\right\rbrack$，并且（根据点连接的定义）有${t}^{\prime }\left\lbrack  {A}_{H}\right\rbrack   = a$。因此，我们可以通过再一次扫描（当前的）${r}_{H}$来输出所有这样的${t}^{\prime }$。

The claimed I/O cost follows from the fact that ${r}_{H}$ is sorted $d - 1$ times in total,while ${r}_{i}$ is sorted once for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ .

所声称的I/O成本源于这样一个事实，即${r}_{H}$总共被排序$d - 1$次，而${r}_{i}$对于每个$i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$被排序一次。

## Proof of Lemma 5

## 引理5的证明

Let us first understand how $t$ is passed from a call to its descendants in $\mathcal{T}$ . Let $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ be a level- $\ell$ call that $t$ participates in. If ${h}_{\ell  + 1} \neq  i$ ,then $t$ participates in at most one of the call’s child nodes in $\mathcal{T}$ . Otherwise, $t$ may participate in all of the call’s child nodes in $\mathcal{T}$ .

让我们首先了解$t$是如何从$\mathcal{T}$中的一次调用传递给其后代的。设$\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$是$t$参与的一个第$\ell$层调用。如果${h}_{\ell  + 1} \neq  i$，那么$t$最多参与$\mathcal{T}$中该调用的一个子节点。否则，$t$可能参与$\mathcal{T}$中该调用的所有子节点。

We first consider the case ${L}_{i} = 0$ ,under which there are two possible scenarios: (i) $i = 1$ ,or (ii) $i$ is not the axis of any call in $\mathcal{T}$ . In neither case will we have a call $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ with ${h}_{\ell  + 1} = i$ . This implies that ${\gamma }_{\ell }\left( t\right)  \leq  1$ for all $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ .

我们首先考虑情况 ${L}_{i} = 0$，在此情况下有两种可能的情形：(i) $i = 1$，或者 (ii) $i$ 不是 $\mathcal{T}$ 中任何调用的轴。在这两种情形下，我们都不会有满足 ${h}_{\ell  + 1} = i$ 的调用 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$。这意味着对于所有 $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ 都有 ${\gamma }_{\ell }\left( t\right)  \leq  1$。

Now consider that ${L}_{i} \in  \left\lbrack  {1,w - 1}\right\rbrack$ . Let $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ be a level- $\ell$ call that $t$ participates in. If $\ell  \neq  {L}_{i}$ ,then the call passes $t$ to at most one of its child nodes. If $\ell  = {L}_{i}$ ,then by definition of ${L}_{i}$ ,we have $i = {h}_{1 + {L}_{i}}$ . In this scenario,the call may pass $t$ to all its $q$ child nodes where

现在考虑 ${L}_{i} \in  \left\lbrack  {1,w - 1}\right\rbrack$ 的情况。设 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 是 $t$ 参与的一个 $\ell$ 级调用。如果 $\ell  \neq  {L}_{i}$，那么该调用最多将 $t$ 传递给它的一个子节点。如果 $\ell  = {L}_{i}$，那么根据 ${L}_{i}$ 的定义，我们有 $i = {h}_{1 + {L}_{i}}$。在这种情形下，该调用可能会将 $t$ 传递给它的所有 $q$ 个子节点，其中

$$
q = O\left( {1 + \left| {\rho }_{1}\right| /{\tau }_{i}}\right) 
$$

$$
\text{(by}\left( 3\right) ) = O\left( {1 + {\tau }_{{h}_{{L}_{i}}}/{\tau }_{i}}\right) 
$$

$$
 = O\left( {1 + {\tau }_{{h}_{{L}_{i}}}/{\tau }_{{h}_{1 + {L}_{i}}}}\right) 
$$

$$
 = O\left( {\mu }_{{L}_{i}}\right) \text{.}
$$

---

<!-- Footnote -->

${}^{4}$ This address requires only ${\lg }_{2}{n}_{1}$ bits by storing an offset.

${}^{4}$ 通过存储一个偏移量，这个地址只需要 ${\lg }_{2}{n}_{1}$ 位。

${}^{5}$ Note that ${r}_{i}$ can have at most one tuple ${t}^{\prime }$ that has the same values as $t$ on all attributes in ${X}_{i}$ (recall that ${t}^{\prime }\left\lbrack  {A}_{H}\right\rbrack$ is fixed to $a$ by definition of point join).

${}^{5}$ 注意，${r}_{i}$ 最多可以有一个元组 ${t}^{\prime }$，它在 ${X}_{i}$ 中的所有属性上的值与 $t$ 相同（回想一下，根据点连接的定义，${t}^{\prime }\left\lbrack  {A}_{H}\right\rbrack$ 固定为 $a$）。

<!-- Footnote -->

---

This implies the equation of ${\gamma }_{\ell }\left( t\right)$ given in (12).

这意味着式 (12) 中给出的 ${\gamma }_{\ell }\left( t\right)$ 的等式成立。

## Proof of Lemma 6

## 引理 6 的证明

By the definition of ${\mu }_{\ell  - 1}$ ,it suffices to show that ${\tau }_{{h}_{\ell  - 1}}/{\tau }_{{h}_{\ell }} =$ $O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$ . (2) implies that

根据 ${\mu }_{\ell  - 1}$ 的定义，只需证明 ${\tau }_{{h}_{\ell  - 1}}/{\tau }_{{h}_{\ell }} =$ $O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$。(2) 意味着

$$
\frac{{\tau }_{{h}_{\ell  - 1}}}{{\tau }_{{h}_{\ell }}} = \frac{{\left( U{d}^{\frac{1}{d - 1}}\right) }^{{h}_{\ell } - {h}_{\ell  - 1}}}{\mathop{\prod }\limits_{{j = 1 + {h}_{\ell  - 1}}}^{{h}_{\ell }}{n}_{j}}. \tag{18}
$$

If ${h}_{\ell } = 1 + {h}_{\ell  - 1}$ ,then

如果 ${h}_{\ell } = 1 + {h}_{\ell  - 1}$，那么

$$
\left( {18}\right)  = \frac{U{d}^{\frac{1}{d - 1}}}{{n}_{{h}_{\ell }}}.
$$

For the case where ${h}_{\ell } > 1 + {h}_{\ell  - 1}$ ,the definition of ${h}_{\ell }$ indicates that

对于 ${h}_{\ell } > 1 + {h}_{\ell  - 1}$ 的情况，${h}_{\ell }$ 的定义表明

$$
\frac{{\tau }_{{h}_{\ell  - 1}}}{{\tau }_{{h}_{\ell } - 1}} = \frac{{\left( U{d}^{\frac{1}{d - 1}}\right) }^{{h}_{\ell } - 1 - {h}_{\ell  - 1}}}{\mathop{\prod }\limits_{{j = 1 + {h}_{\ell  - 1}}}^{{{h}_{\ell } - 1}}{n}_{j}} \leq  2;
$$

otherwise, ${h}_{\ell }$ would not be the smallest integer in $\left\lbrack  {1 + {h}_{\ell  - 1},d}\right\rbrack$ satisfying ${\tau }_{{h}_{\ell }} < {\tau }_{{h}_{\ell  - 1}}/2$ . Hence,

否则，${h}_{\ell }$ 就不会是 $\left\lbrack  {1 + {h}_{\ell  - 1},d}\right\rbrack$ 中满足 ${\tau }_{{h}_{\ell }} < {\tau }_{{h}_{\ell  - 1}}/2$ 的最小整数。因此，

$$
\left( {18}\right)  \leq  2 \cdot  \frac{U{d}^{\frac{1}{d - 1}}}{{n}_{{h}_{\ell }}},
$$

which completes the proof.

至此，证明完毕。