# I/O-Efficient Join Dependency Testing, Loomis-Whitney Join, and Triangle Enumeration*

# 高效输入输出的连接依赖测试、卢米斯 - 惠特尼连接与三角形枚举*

Xiaocheng Hu

胡晓成

Chinese University of Hong Kong

香港中文大学

xchu@cse.cuhk.edu.hk

Miao Qiao

乔苗

Massey University

梅西大学

m.qiao@massey.ac.nz

Yufei Tao

陶宇飞

University of Queensland

昆士兰大学

taoyf@itee.uq.edu.au

April 21, 2016

2016年4月21日

## Abstract

## 摘要

We revisit two fundamental problems in database theory. The join-dependency (JD) testing problem is to determine whether a given JD holds on a relation $r$ . We prove that the problem is NP-hard even if the JD involves only relations each of which has only two attributes. The JD-existence testing problem is to determine if there exists any non-trivial JD satisfied by $r$ . We present an I/O-efficient algorithm in the external memory model, which in fact settles the closely related Loomis-Whitney enumeration problem. As a side product, we solve the triangle enumeration problem with the optimal I/O-complexity, improving a recent result of Pagh and Silvestri in PODS'14.

我们重新探讨了数据库理论中的两个基本问题。连接依赖（Join Dependency，JD）测试问题是确定给定的连接依赖是否在关系 $r$ 上成立。我们证明，即使连接依赖仅涉及每个都只有两个属性的关系，该问题也是NP难的。连接依赖存在性测试问题是确定是否存在任何非平凡的连接依赖能被 $r$ 满足。我们在外存模型中提出了一种高效输入输出的算法，实际上该算法解决了密切相关的卢米斯 - 惠特尼枚举问题。作为附带成果，我们以最优的输入输出复杂度解决了三角形枚举问题，改进了帕格（Pagh）和西尔维斯特里（Silvestri）在2014年数据库系统原理研讨会（PODS'14）上的最新成果。

Keywords: Join Dependency, Loomis-Whitney Join, Triangle Enumeration, External Memory

关键词：连接依赖、卢米斯 - 惠特尼连接、三角形枚举、外存

---

<!-- Footnote -->

*A preliminary version of this article appeared in PODS'15.

*本文的初步版本发表于2015年数据库系统原理研讨会（PODS'15）。

<!-- Footnote -->

---

## 1 Introduction

## 1 引言

Given a relation $r$ of $d$ attributes,a key question in database theory is to ask if $r$ is decomposable, namely,whether $r$ can be projected onto a set $S$ of relations with less than $d$ attributes such that the natural join of those relations equals precisely $r$ . Intuitively,a yes answer to the question implies that $r$ contains a certain form of redundancy. Some of the redundancy may be removed by decomposing $r$ into the smaller (in terms of attribute number) relations in $S$ ,which can be joined together to restore $r$ whenever needed. A no answer,on the other hand,implies that the decomposition of $r$ based on $S$ will lose information,as far as natural join is concerned.

给定一个具有 $d$ 个属性的关系 $r$，数据库理论中的一个关键问题是询问 $r$ 是否可分解，即 $r$ 是否可以投影到一组属性少于 $d$ 个的关系 $S$ 上，使得这些关系的自然连接恰好等于 $r$。直观地说，对该问题的肯定回答意味着 $r$ 包含某种形式的冗余。通过将 $r$ 分解为 $S$ 中（就属性数量而言）更小的关系，可以去除一些冗余，这些小关系可以在需要时连接起来恢复 $r$。另一方面，否定回答意味着就自然连接而言，基于 $S$ 对 $r$ 进行分解会丢失信息。

Join Dependency Testing. The above question (as well as its variants) has been extensively studied by resorting to the notion of join dependency (JD). To formalize the notion, let us refer to $d$ as the arity of $r$ . Denote by $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ the set of names of the $d$ attributes in $r.R$ is called the schema of $r$ . Sometimes we may denote $r$ as $r\left( R\right)$ or $r\left( {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right)$ to emphasize on its schema. Let $\left| r\right|$ represent the number of tuples in $r$ .

连接依赖测试。上述问题（及其变体）已通过连接依赖（Join Dependency，JD）的概念得到了广泛研究。为了形式化这个概念，我们将 $d$ 称为 $r$ 的元数。用 $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 表示 $r.R$ 中 $d$ 个属性的名称集合，$R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 被称为 $r$ 的模式。有时我们可以将 $r$ 表示为 $r\left( R\right)$ 或 $r\left( {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right)$ 以强调其模式。用 $\left| r\right|$ 表示 $r$ 中的元组数量。

A JD defined on $R$ is an expression of the form

定义在 $R$ 上的连接依赖是如下形式的表达式

$$
J =  \bowtie  \left\lbrack  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\rbrack  
$$

where (i) $m \geq  1$ ,(ii) each ${R}_{i}\left( {1 \leq  i \leq  m}\right)$ is a subset of $R$ that contains at least 2 attributes,and (iii) ${ \cup  }_{i = 1}^{m}{R}_{i} = R.J$ is non-trivial if none of ${R}_{1},\ldots ,{R}_{m}$ equals $R$ . The arity of $J$ is defined to be $\mathop{\max }\limits_{{i = 1}}^{m}\left| {R}_{i}\right|$ ,i.e.,the largest size of ${R}_{1},\ldots ,{R}_{m}$ . Clearly,the arity of a non-trivial $J$ is between 2 and $d - 1$ .

其中 (i) $m \geq  1$ ，(ii) 每个 ${R}_{i}\left( {1 \leq  i \leq  m}\right)$ 都是 $R$ 的一个子集，且该子集至少包含 2 个属性，并且 (iii) 如果没有一个 ${R}_{1},\ldots ,{R}_{m}$ 等于 $R$ ，则 ${ \cup  }_{i = 1}^{m}{R}_{i} = R.J$ 是非平凡的。$J$ 的元数定义为 $\mathop{\max }\limits_{{i = 1}}^{m}\left| {R}_{i}\right|$ ，即 ${R}_{1},\ldots ,{R}_{m}$ 的最大规模。显然，一个非平凡的 $J$ 的元数介于 2 和 $d - 1$ 之间。

Relation $r$ is said to satisfy $J$ if

如果关系 $r$ 满足 $J$ ，则称其满足该条件

$$
r = {\pi }_{{R}_{1}}\left( r\right)  \bowtie  {\pi }_{{R}_{2}}\left( r\right)  \bowtie  \ldots  \bowtie  {\pi }_{{R}_{m}}\left( r\right) 
$$

where ${\pi }_{X}\left( r\right)$ denotes the projection of $r$ onto an attribute set $X$ ,and $\bowtie$ represents natural join. We are ready to formally state the first two problems studied in this article:

其中 ${\pi }_{X}\left( r\right)$ 表示 $r$ 在属性集 $X$ 上的投影，并且 $\bowtie$ 表示自然连接。我们准备正式陈述本文研究的前两个问题：

Problem 1. [λ-JD Testing] Given a relation $r$ and a join dependency $J$ of arity at most $\lambda$ that is defined on the schema of $r$ ,we want to determine whether $r$ satisfies $J$ .

问题 1. [λ - 连接依赖测试] 给定一个关系 $r$ 和一个定义在 $r$ 的模式上且元数至多为 $\lambda$ 的连接依赖 $J$ ，我们要确定 $r$ 是否满足 $J$ 。

Problem 2. [JD Existence Testing] Given a relation $r$ ,we want to determine whether there is any non-trivial join dependency $J$ such that $r$ satisfies $J$ .

问题 2. [连接依赖存在性测试] 给定一个关系 $r$ ，我们要确定是否存在任何非平凡的连接依赖 $J$ ，使得 $r$ 满足 $J$ 。

Note the difference in the objectives of the above problems. Problem 1 aims to decide if $r$ can be decomposed according to a specific set $J$ of projections. On the other hand,Problem 2 aims to find out if there is any way to decompose $r$ at all.

注意上述问题目标的差异。问题 1 的目标是确定 $r$ 是否可以根据特定的投影集 $J$ 进行分解。另一方面，问题 2 的目标是找出是否存在任何分解 $r$ 的方法。

Computation Model. Our discussion on Problem 1 will concentrate on proving its NP-hardness. For this purpose, we will describe all our reductions in the standard RAM model.

计算模型。我们对问题 1 的讨论将集中在证明其 NP 难性上。为此，我们将在标准随机存取机（RAM）模型中描述所有的归约。

For Problem 2, which is known to be polynomial time solvable (as we will explain shortly), the main issue is to design fast algorithms. We will do so in the external memory (EM) model [2], which has become the de facto model for analyzing I/O-efficient algorithms. Under this model, a machine is equipped with $M$ words of memory,and an unbounded disk that has been formatted into blocks of $B$ words. It holds that $M \geq  {2B}$ . An $I/O$ operation exchanges a block of data between the disk and the memory. The cost of an algorithm is defined to be the number of I/Os performed. CPU calculation is for free.

对于问题 2，已知它是多项式时间可解的（我们将很快解释），主要问题是设计快速算法。我们将在外部存储器（EM）模型 [2] 中进行，该模型已成为分析 I/O 高效算法的事实上的模型。在该模型下，一台机器配备有 $M$ 个字的内存，以及一个无界的磁盘，该磁盘已被格式化为 $B$ 个字的块。满足 $M \geq  {2B}$ 。一次 $I/O$ 操作在磁盘和内存之间交换一个数据块。算法的代价定义为执行的 I/O 次数。CPU 计算是免费的。

To avoid rounding,we define ${\lg }_{x}y = \max \left\{  {1,{\log }_{x}y}\right\}$ ,and will describe all logarithms using ${\lg }_{x}y$ . In all cases, the value of an attribute is assumed to fit in a single word.

为避免取整，我们定义 ${\lg }_{x}y = \max \left\{  {1,{\log }_{x}y}\right\}$ ，并将使用 ${\lg }_{x}y$ 来描述所有对数。在所有情况下，假设一个属性的值可以放入一个字中。

Loomis-Whitney Enumeration. As will be clear later, the JD existence-testing problem is closely related to the so-called $L$ comis-Whitney (LW) join (this term was coined in [12]). Let $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ be a set of $d$ attributes. For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,define ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ ,that is, removing ${A}_{i}$ from $R$ . Let ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ be $d$ relations such that ${r}_{i}\left( {1 \leq  i \leq  d}\right)$ has schema ${R}_{i}$ . Then, the natural join ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ is called an LW join. Note that the schema of the join result is $R$ .

卢米斯 - 惠特尼枚举。稍后会清楚，连接依赖存在性测试问题与所谓的 $L$ 卢米斯 - 惠特尼（LW）连接密切相关（该术语在 [12] 中被创造）。设 $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 是一个包含 $d$ 个属性的集合。对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack$ ，定义 ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ ，即从 $R$ 中移除 ${A}_{i}$ 。设 ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ 是 $d$ 个关系，使得 ${r}_{i}\left( {1 \leq  i \leq  d}\right)$ 的模式为 ${R}_{i}$ 。那么，自然连接 ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 被称为 LW 连接。注意，连接结果的模式是 $R$ 。

We will consider LW joins in the EM model, where traditionally a join must write out all the tuples in the result to the disk. However, the result size can be so huge that the number of I/Os for writing the result may (by far) overwhelm the cost of the join's rest execution. Furthermore, in some applications of LW joins (e.g., for solving Problem 2), it is not necessary to actually write the result tuples to the disk; instead, it suffices to witness each result tuple once in the memory.

我们将在外部内存（EM）模型中考虑轻量级（LW）连接，在该模型中，传统上连接操作必须将结果中的所有元组写入磁盘。然而，结果规模可能非常大，以至于写入结果的输入/输出（I/O）次数可能（远远）超过连接操作其余执行部分的成本。此外，在LW连接的某些应用中（例如，用于解决问题2），实际上并不需要将结果元组写入磁盘；相反，只需在内存中见证每个结果元组一次即可。

Because of the above, we follow the approach of [14] by studying an enumerate version of the problem. Specifically,we are given a memory-resident routine emit (.) which requires $O\left( 1\right)$ words to store. The parameter of the routine is a tuple $t$ of $d$ values $\left( {{a}_{1},\ldots ,{a}_{d}}\right)$ such that ${a}_{i}$ is in the domain of ${A}_{i}$ for each $i \in  \left\lbrack  {1,d}\right\rbrack$ . The routine simply sends out $t$ to an outbound socket with no I/O cost. Then, our problem can be formally stated as:

基于上述原因，我们采用文献[14]的方法来研究该问题的枚举版本。具体而言，我们有一个驻留在内存中的例程emit(.)，它需要$O\left( 1\right)$个字来存储。该例程的参数是一个包含$d$个值$\left( {{a}_{1},\ldots ,{a}_{d}}\right)$的元组$t$，使得对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，${a}_{i}$都在${A}_{i}$的定义域内。该例程只是将$t$发送到一个出站套接字，且无需I/O成本。那么，我们的问题可以正式表述为：

Problem 3. [LW Enumeration] Given relations ${r}_{1},\ldots ,{r}_{d}$ as defined earlier where $d \leq  M/2$ ,we want to invoke emit(t)once and exactly once for each tuple $t \in  {r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$ .

问题3. [LW枚举] 给定如前文所定义的关系${r}_{1},\ldots ,{r}_{d}$，其中$d \leq  M/2$，我们希望对于每个元组$t \in  {r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$，恰好调用一次emit(t)。

As a noteworthy remark,if an algorithm can solve the above problem in $x\mathrm{I}/\mathrm{{Os}}$ using $M - B$ words of memory,then it can also report the entire LW join result of $K$ tuples (i.e.,totally ${Kd}$ values) in $x + O\left( {{Kd}/B}\right) \mathrm{I}/\mathrm{{Os}}$ .

值得注意的是，如果一个算法能够在$x\mathrm{I}/\mathrm{{Os}}$内使用$M - B$个字的内存解决上述问题，那么它也能够在$x + O\left( {{Kd}/B}\right) \mathrm{I}/\mathrm{{Os}}$内报告$K$个元组（即总共${Kd}$个值）的整个LW连接结果。

Triangle Enumeration. Besides being a stepping stone for Problem 2, LW enumeration has relevance to several other problems, among which the most prominent one is perhaps the triangle enumeration problem [14] due to its large variety of applications (see $\left\lbrack  {8,{14}}\right\rbrack$ and the references therein for an extensive summary).

三角形枚举。除了作为解决问题2的垫脚石之外，LW枚举还与其他几个问题相关，其中最突出的可能是三角形枚举问题[14]，因为它有各种各样的应用（详见$\left\lbrack  {8,{14}}\right\rbrack$及其参考文献中的详细总结）。

Let $G = \left( {V,E}\right)$ be an undirected simple graph,where $V$ (or $E$ ) is the set of vertices (or edges, resp.). A triangle is defined as a clique of 3 vertices in $G$ . We are again given a memory-resident routine emit(.) that occupies $O\left( 1\right)$ words. This time,given a triangle $\Delta$ as its parameter,the routine sends out $\Delta$ to an outbound socket with no I/O cost (this implies that all the 3 edges of $\Delta$ must be in the memory at this moment). Then, the triangle enumeration problem can be formally stated as:

设$G = \left( {V,E}\right)$为一个无向简单图，其中$V$（或$E$）是顶点集（或边集）。三角形被定义为$G$中由3个顶点组成的团。我们再次有一个驻留在内存中的例程emit(.)，它占用$O\left( 1\right)$个字。这次，给定一个三角形$\Delta$作为其参数，该例程将$\Delta$发送到一个出站套接字，且无需I/O成本（这意味着此时$\Delta$的所有3条边都必须在内存中）。那么，三角形枚举问题可以正式表述为：

Problem 4. [Triangle Enumeration] Given graph $G$ as defined earlier,we want to invoke emit $\left( \Delta \right)$ once and exactly once for each triangle $\Delta$ in $G$ .

问题4. [三角形枚举] 给定如前文所定义的图$G$，我们希望对于$G$中的每个三角形$\Delta$，恰好调用一次emit $\left( \Delta \right)$。

Observe that this is merely a special instance of LW enumeration with $d = 3$ where ${r}_{1} = {r}_{2} =$ ${r}_{3} = E$ (specifically, $E$ is regarded as a relation with two columns,such that every edge(u,v)gives rise to two tuples(u,v)and(v,u)in the relation),with some straightforward care to avoid emitting a triangle twice in no extra $\mathrm{I}/\mathrm{O}$ cost.

注意，这只是LW枚举的一个特殊实例，其中$d = 3$，${r}_{1} = {r}_{2} =$ ${r}_{3} = E$（具体而言，$E$被视为一个有两列的关系，使得每条边(u, v)在该关系中产生两个元组(u, v)和(v, u)），只需进行一些简单的处理，就可以在不产生额外$\mathrm{I}/\mathrm{O}$成本的情况下避免重复输出一个三角形。

### 1.1 Previous Results

### 1.1 先前的研究成果

Join Dependency Testing. Beeri and Vardi [5] proved that $\lambda$ -JD testing (Problem 1) is NP-hard if $\lambda  = d - o\left( d\right)$ ; recall that $d$ is the number of attributes in the input relation $r$ . Maier,Sagiv, and Yannakakis [11] gave a stronger proof showing that $\lambda$ -JD testing is still NP-hard for $\lambda  = \Omega \left( d\right)$ (more specifically,roughly ${2d}/3$ ). In other words,(unless $\mathrm{P} = \mathrm{{NP}}$ ) no polynomial-time algorithm can exist to verify every $\mathrm{{JD}} \propto  \left\lbrack  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\rbrack$ on $r$ ,when one of ${R}_{1},\ldots ,{R}_{m}$ has $\Omega \left( d\right)$ attributes.

连接依赖测试。贝里（Beeri）和瓦尔迪（Vardi）[5]证明，如果$\lambda  = d - o\left( d\right)$，则$\lambda$ - 连接依赖测试（问题1）是NP难问题；回顾一下，$d$是输入关系$r$中的属性数量。迈尔（Maier）、萨吉夫（Sagiv）和亚纳卡基斯（Yannakakis）[11]给出了更有力的证明，表明对于$\lambda  = \Omega \left( d\right)$（更具体地说，大约为${2d}/3$），$\lambda$ - 连接依赖测试仍然是NP难问题。换句话说，（除非$\mathrm{P} = \mathrm{{NP}}$），当${R}_{1},\ldots ,{R}_{m}$中的一个具有$\Omega \left( d\right)$个属性时，不存在多项式时间算法来验证$r$上的每个$\mathrm{{JD}} \propto  \left\lbrack  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\rbrack$。

However, the above result does not rule out the possibility of efficient testing when the JD has a small arity,namely,all of ${R}_{1},\ldots ,{R}_{m}$ have just a few attributes (e.g.,as few as just 2). Small-arity JDs are important because many relations in reality can eventually be losslessly decomposed into relations with small arities. By definition,for any ${\lambda }_{1} < {\lambda }_{2}$ ,the ${\lambda }_{1}$ -JD testing problem may only be easier than ${\lambda }_{2}$ -JD testing problem because an algorithm for the latter can be used to solve the former problem, but not the vice versa. The ultimate question, therefore, is whether 2-JD testing can be solved within polynomial time. Unfortunately,the arity of $J$ being $\Omega \left( d\right)$ appears to be an inherent requirement in the reductions of $\left\lbrack  {5,{11}}\right\rbrack$ .

然而，上述结果并未排除在连接依赖的元数较小时进行高效测试的可能性，即所有的${R}_{1},\ldots ,{R}_{m}$都只有几个属性（例如，少至仅2个）。小元数连接依赖很重要，因为现实中的许多关系最终都可以无损地分解为小元数的关系。根据定义，对于任何${\lambda }_{1} < {\lambda }_{2}$，${\lambda }_{1}$ - 连接依赖测试问题可能比${\lambda }_{2}$ - 连接依赖测试问题更容易，因为解决后者的算法可用于解决前者问题，但反之则不行。因此，最终的问题是2 - 连接依赖测试是否可以在多项式时间内解决。不幸的是，$J$的元数为$\Omega \left( d\right)$似乎是$\left\lbrack  {5,{11}}\right\rbrack$归约中的一个内在要求。

We note that a large body of beautiful theory has been developed on dependency inference, where the objective is to determine whether a target dependency can be inferred from a set $\sum$ of dependencies (see $\left\lbrack  {1,{10}}\right\rbrack$ for excellent guides into the literature). When the target dependency is a join dependency, the inference problem has been proven to be NP-hard in a variety of scenarios, most notably: (i) when $\sum$ contains one join dependency and a set of functional dependencies [5,11], (ii) when $\sum$ is a set of multi-valued dependencies [6],and (iii) when $\sum$ has one domain dependency and a set of functional dependencies [9]. The proofs of $\left\lbrack  {5,{11}}\right\rbrack$ are essentially the same ones used to establish the NP-hardness of $\Omega \left( d\right)$ -JD testing,while those of $\left\lbrack  {6,9}\right\rbrack$ do not imply any conclusions on $\lambda$ -JD testing.

我们注意到，在依赖推理方面已经发展出了大量出色的理论，其目标是确定是否可以从一组依赖$\sum$中推断出目标依赖（有关文献的优秀指南，请参阅$\left\lbrack  {1,{10}}\right\rbrack$）。当目标依赖是连接依赖时，在多种场景下，推理问题已被证明是NP难问题，最显著的情况包括：（i）当$\sum$包含一个连接依赖和一组函数依赖时[5,11]；（ii）当$\sum$是一组多值依赖时[6]；（iii）当$\sum$有一个域依赖和一组函数依赖时[9]。$\left\lbrack  {5,{11}}\right\rbrack$的证明本质上与用于证明$\Omega \left( d\right)$ - 连接依赖测试的NP难性的证明相同，而$\left\lbrack  {6,9}\right\rbrack$的证明并未对$\lambda$ - 连接依赖测试得出任何结论。

JD Existence Testing and LW Join. There is an interesting connection between JD existence testing (Problem 2) and LW join. Let $r\left( R\right)$ be the input relation to Problem 2,where $R =$ $\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,define ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ ,and ${r}_{i} = {\pi }_{{R}_{i}}\left( r\right)$ . Nicolas showed [13] that $r$ satisfies at least one non-trivial JD if and only if $r = {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ . In fact,since it is always true that $r \subseteq  {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ ,Problem 2 has an answer yes if and only if ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ returns exactly $\left| r\right|$ result tuples.

连接依赖（JD）存在性测试与轻量级连接（LW Join）。连接依赖存在性测试（问题2）和轻量级连接之间存在着有趣的联系。设$r\left( R\right)$为问题2的输入关系，其中$R =$ $\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ 。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，定义${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$和${r}_{i} = {\pi }_{{R}_{i}}\left( r\right)$ 。尼古拉斯（Nicolas）在文献[13]中指出，当且仅当$r = {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$时，$r$满足至少一个非平凡的连接依赖。事实上，由于$r \subseteq  {r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$总是成立，所以当且仅当${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$恰好返回$\left| r\right|$个结果元组时，问题2的答案为“是”。

Therefore,Problem 2 boils down to evaluating the result size of the LW join ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ . Atserias,Grohe,and Marx [4] showed that the result size can be as large as ${\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}}$ ,where ${n}_{i} = \left| {r}_{i}\right|$ for each $i \in  \left\lbrack  {1,d}\right\rbrack$ . They also gave a RAM algorithm to compute the join result in $O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} \cdot  \mathop{\sum }\limits_{{i = 1}}^{n}{n}_{i}}\right)$ time. Since apparently ${n}_{i} \leq  n = \left| r\right| \left( {1 \leq  i \leq  d}\right)$ ,it follows that their algorithm has running time $O\left( {{d}^{2} \cdot  {n}^{d/\left( {d - 1}\right) } \cdot  {dn}}\right)  = O\left( {{d}^{3} \cdot  {n}^{2 + o\left( 1\right) }}\right)$ ,which in turn means that Problem 2 is solvable in polynomial time. Ngo et al. [12] designed a faster RAM algorithm to perform the LW join (hence,solving Problem 2) in $O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$ time. For constant $d$ ,Veldulzen [15] presented a simplified algorithm to achieve the same complexity.

因此，问题2可归结为计算轻量级连接${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$的结果规模。阿塞里亚斯（Atserias）、格罗赫（Grohe）和马克思（Marx）在文献[4]中表明，结果规模可能高达${\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}}$，其中对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$有${n}_{i} = \left| {r}_{i}\right|$ 。他们还给出了一个随机存取机（RAM）算法，可在$O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} \cdot  \mathop{\sum }\limits_{{i = 1}}^{n}{n}_{i}}\right)$时间内计算连接结果。显然${n}_{i} \leq  n = \left| r\right| \left( {1 \leq  i \leq  d}\right)$，因此他们的算法运行时间为$O\left( {{d}^{2} \cdot  {n}^{d/\left( {d - 1}\right) } \cdot  {dn}}\right)  = O\left( {{d}^{3} \cdot  {n}^{2 + o\left( 1\right) }}\right)$，这意味着问题2可在多项式时间内解决。恩戈（Ngo）等人在文献[12]中设计了一个更快的随机存取机算法，可在$O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$时间内执行轻量级连接（从而解决问题2）。对于常数$d$，费尔杜尔岑（Veldulzen）在文献[15]中提出了一个简化算法，可达到相同的复杂度。

Problems 2 and 3 become much more challenging in external memory (EM). All the algorithms of $\left\lbrack  {4,{12},{15}}\right\rbrack$ rely on the fact that dictionary search,i.e.,finding a specific element in a set,can be done efficiently in RAM (either in $O\left( 1\right)$ expected time or logarithmic worse-case time). In EM, however, spending even just one I/O per for dictionary search is excessively expensive for a problem like LW join because the number of such searches is huge. Because of this, when adapted to EM, the algorithms of $\left\lbrack  {4,{12},{15}}\right\rbrack$ do not offer competitive efficiency; for instance,that of $\left\lbrack  {12}\right\rbrack$ can entail up to $O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$ I/Os. When $d$ is small,this may be even worse than a naive generalized blocked-nested loop,whose I/O complexity for $d = O\left( 1\right)$ is $O\left( {{n}_{1}{n}_{2}\ldots {n}_{d}/\left( {{M}^{d - 1}B}\right) }\right)$ I/Os. Recall that $B$ and $M$ are the sizes of a disk block and memory,respectively.

问题2和问题3在外存（External Memory，EM）中变得更具挑战性。文献$\left\lbrack  {4,{12},{15}}\right\rbrack$中的所有算法都依赖于这样一个事实：字典查找，即在一个集合中查找特定元素，能在随机存取存储器（Random Access Memory，RAM）中高效完成（期望时间为$O\left( 1\right)$或最坏情况时间为对数级）。然而，在外存中，对于像LW连接这样的问题，即使每次字典查找仅花费一次输入/输出（Input/Output，I/O）操作也是极其昂贵的，因为此类查找的数量非常庞大。因此，当将文献$\left\lbrack  {4,{12},{15}}\right\rbrack$中的算法应用于外存时，它们的效率并不具有竞争力；例如，文献$\left\lbrack  {12}\right\rbrack$中的算法可能需要多达$O\left( {{d}^{2} \cdot  {\left( {n}_{1}{n}_{2}\ldots {n}_{d}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right)$次I/O操作。当$d$较小时，这甚至可能比简单的广义块嵌套循环算法更差，后者对于$d = O\left( 1\right)$的I/O复杂度为$O\left( {{n}_{1}{n}_{2}\ldots {n}_{d}/\left( {{M}^{d - 1}B}\right) }\right)$次I/O操作。请记住，$B$和$M$分别是磁盘块和内存的大小。

Triangle Enumeration. Problem 4 has received a large amount of attention from the database and theory communities (see [8] for a survey). Recently, Pagh and Silvestri [14] solved the problem in EM with a randomized algorithm whose I/O cost is $O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$ expected,where $\left| E\right|$ is the number of edges in the input graph. They also presented a sophisticated de-randomization technique to convert their algorithm into a deterministic one that performs $O\left( {\frac{{\left| E\right| }^{1.5}}{\sqrt{MB}} \cdot  {\log }_{M/B}\frac{\left| E\right| }{B}}\right)$ $\mathrm{I}/\mathrm{{Os}}$ . An $\mathrm{I}/\mathrm{O}$ lower bound of $\Omega \left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$ has been independently developed in [8,14] on the witnessing class of algorithms.

三角形枚举。问题4受到了数据库和理论研究界的广泛关注（相关综述见文献[8]）。最近，帕格（Pagh）和西尔维斯特里（Silvestri）[14]使用一种随机算法在外存中解决了该问题，其I/O成本的期望为$O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$，其中$\left| E\right|$是输入图中的边数。他们还提出了一种复杂的去随机化技术，将其算法转换为确定性算法，该确定性算法执行$O\left( {\frac{{\left| E\right| }^{1.5}}{\sqrt{MB}} \cdot  {\log }_{M/B}\frac{\left| E\right| }{B}}\right)$ $\mathrm{I}/\mathrm{{Os}}$ 。文献[8,14]独立地针对见证类算法得出了$\Omega \left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$的$\mathrm{I}/\mathrm{O}$下界。

### 1.2 Our Results

### 1.2 我们的研究成果

Our first main result is:

我们的第一个主要成果是：

Theorem 1. 2-JD testing is NP-hard.

定理1. 二元连接依赖（2-JD）测试是NP难问题。

The theorem officially puts a negative answer to the question whether a small-arity JD can be tested efficiently (remember that 2 is already the smallest possible arity). As a consequence, we know that Problem 2 is NP-hard for every value $\lambda  \in  \left\lbrack  {2,d - 1}\right\rbrack$ . Our proof is completely different from those of $\left\lbrack  {5,{11}}\right\rbrack$ ,and is based on a novel reduction from the Hamiltonian path problem.

该定理正式对小元数连接依赖（JD）能否被高效测试这一问题给出了否定答案（要知道2已经是可能的最小元数）。因此，我们知道对于任意值$\lambda  \in  \left\lbrack  {2,d - 1}\right\rbrack$，问题2都是NP难问题。我们的证明与文献$\left\lbrack  {5,{11}}\right\rbrack$中的证明完全不同，它基于从哈密顿路径问题进行的一种新颖归约。

Our second main result is an I/O-efficient algorithm for LW enumeration (Problem 3). Let ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ be the input relations; and set ${n}_{i} = \left| {r}_{i}\right|$ . We will prove:

我们的第二个主要成果是一种用于LW枚举（问题3）的I/O高效算法。设${r}_{1},{r}_{2},\ldots ,{r}_{d}$为输入关系；并设${n}_{i} = \left| {r}_{i}\right|$ 。我们将证明：

Theorem 2. There is an EM algorithm that solves the LW enumeration problem with I/O complexity:

定理2. 存在一种外存算法，该算法解决LW枚举问题的I/O复杂度为：

$$
O\left( {\operatorname{sort}\left\lbrack  {{d}^{3 + o\left( 1\right) }{\left( \frac{{\Pi }_{i = 1}^{d}{n}_{i}}{M}\right) }^{\frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right\rbrack  }\right) .
$$

where function sort $\left( x\right)  = \left( {x/B}\right) {\lg }_{M/B}\left( {x/B}\right)$ .

其中排序函数为$\left( x\right)  = \left( {x/B}\right) {\lg }_{M/B}\left( {x/B}\right)$ 。

The main obstacle we faced in performing LW enumeration I/O-efficiently is that, we can no longer rely on repetitive dictionary search, as is a key component of all the RAM algorithms $\left\lbrack  {4,{12},{15}}\right\rbrack$ . As mentioned in Section 1.1,while performing a dictionary search in $O\left( 1\right)$ time is good in RAM,it is prohibitively expensive to spend $O\left( 1\right) \mathrm{I}/\mathrm{{Os}}$ for the same purpose in EM. To overcome the obstacle, we abandoned dictionary search completely; in fact, our major contribution is a delicate piece of recursive machinery, which resorts to only sorting and scanning.

我们在高效执行LW枚举的输入/输出（I/O）操作时面临的主要障碍是，我们不能再依赖重复的字典搜索，而字典搜索是所有随机存取存储器（RAM）算法的关键组成部分 $\left\lbrack  {4,{12},{15}}\right\rbrack$。如1.1节所述，虽然在随机存取存储器中以$O\left( 1\right)$的时间复杂度执行字典搜索是可行的，但在外部存储器（EM）中为了相同目的花费$O\left( 1\right) \mathrm{I}/\mathrm{{Os}}$的时间复杂度是极其昂贵的。为了克服这一障碍，我们完全摒弃了字典搜索；事实上，我们的主要贡献是一套精妙的递归机制，该机制仅依赖排序和扫描操作。

As our third main result,we prove in Section 4 an improved version of Theorem 2 for $d = 3$ :

作为我们的第三个主要结果，我们在第4节证明了针对$d = 3$的定理2的改进版本：

Theorem 3. There is an EM algorithm that solves the LW enumeration problem of $d = 3$ with $I/O$ complexity $O\left( {\frac{1}{B}\sqrt{\frac{{n}_{1}{n}_{2}{n}_{3}}{M}} + \operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$ .

定理3. 存在一种外部存储器（EM）算法，该算法能以$I/O$复杂度$O\left( {\frac{1}{B}\sqrt{\frac{{n}_{1}{n}_{2}{n}_{3}}{M}} + \operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$解决$d = 3$的LW枚举问题。

By combining the above two theorems with the reduction from JD existence testing to LW enumeration described in Section 1.1, we obtain the first non-trivial algorithm for I/O-efficient JD existence testing (Problem 2):

通过将上述两个定理与1.1节中描述的从连接依赖（JD）存在性测试到LW枚举的归约相结合，我们得到了第一个非平凡的高效输入/输出（I/O）的连接依赖（JD）存在性测试算法（问题2）：

Corollary 1. Let $r\left( R\right)$ be the input relation to the JD existence testing problem,where $R =$ $\left\{  {{A}_{1},\ldots ,{A}_{d}}\right\}$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,define ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ ,and ${n}_{i}$ as the number of tuples in ${\pi }_{{R}_{i}}\left( r\right)$ . Then:

推论1. 设$r\left( R\right)$为连接依赖（JD）存在性测试问题的输入关系，其中$R =$ $\left\{  {{A}_{1},\ldots ,{A}_{d}}\right\}$。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，定义${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$，并将${n}_{i}$定义为${\pi }_{{R}_{i}}\left( r\right)$中的元组数量。那么：

- For $d > 3$ ,the problem can be solved with the $I/O$ complexity in Theorem 2.

- 对于$d > 3$，该问题可以用定理2中的$I/O$复杂度来解决。

- For $d = 3$ ,the $I/O$ complexity can be improved to the one in Theorem 3.

- 对于$d = 3$，$I/O$复杂度可以改进为定理3中的复杂度。

Finally,when ${n}_{1} = {n}_{2} = {n}_{3} = \left| E\right|$ ,Theorem 3 directly gives a new algorithm for triangle enumeration (Problem 4),noticing that $\operatorname{sort}\left( \left| E\right| \right)  = O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$ :

最后，当${n}_{1} = {n}_{2} = {n}_{3} = \left| E\right|$时，注意到$\operatorname{sort}\left( \left| E\right| \right)  = O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right)$，定理3直接给出了一种用于三角形枚举（问题4）的新算法：

Corollary 2. There is an algorithm that solves the triangle enumeration problem optimally in $O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right) I/{Os}$ .

推论2. 存在一种算法，能以$O\left( {{\left| E\right| }^{1.5}/\left( {\sqrt{M}B}\right) }\right) I/{Os}$的复杂度最优地解决三角形枚举问题。

Our triangle enumeration algorithm is deterministic, and strictly improves that of [14] by a factor of $O\left( {{\lg }_{M/B}\left( {\left| E\right| /B}\right) }\right)$ . Furthermore,the algorithm belongs to the witnessing class [8],and is the first (deterministic algorithm) in this class achieving the optimal I/O complexity for all values of $M$ and $B$ .

我们的三角形枚举算法是确定性的，并且在复杂度上比文献[14]中的算法严格提高了$O\left( {{\lg }_{M/B}\left( {\left| E\right| /B}\right) }\right)$倍。此外，该算法属于见证类[8]，并且是该类中第一个（确定性算法）针对所有$M$和$B$的值都能达到最优输入/输出（I/O）复杂度的算法。

## 2 NP-Hardness of 2-JD Testing

## 2 2 - 连接依赖（2 - JD）测试的NP难问题

This section will establish Theorem 1 with a reduction from the Hamiltonian path problem. Let $G = \left( {V,E}\right)$ be an undirected simple graph (a graph is simple if it has at most one edge between any two vertices). with a vertex set $V$ and an edge set $E$ . Set $n = \left| V\right|$ and $m = \left| E\right|$ . A path of length $\ell$ in $G$ is a sequence of $\ell$ vertices ${v}_{1},{v}_{2},\ldots ,{v}_{\ell }$ such that $E$ has an edge between ${v}_{i}$ and ${v}_{i + 1}$ for each $i \in  \left\lbrack  {1,\ell  - 1}\right\rbrack$ . The path is simple if no two vertices in the path are the same. A Hamiltonian path is a simple path in $G$ of length $n$ (such a path must pass each vertex in $V$ exactly once). Deciding whether $G$ has a Hamiltonian path is known to be NP-hard [7].

本节将通过从哈密顿路径问题（Hamiltonian path problem）进行归约来证明定理 1。设 $G = \left( {V,E}\right)$ 为一个无向简单图（如果任意两个顶点之间最多只有一条边，则该图为简单图），其顶点集为 $V$，边集为 $E$。设 $n = \left| V\right|$ 和 $m = \left| E\right|$。$G$ 中长度为 $\ell$ 的路径是一个由 $\ell$ 个顶点组成的序列 ${v}_{1},{v}_{2},\ldots ,{v}_{\ell }$，使得对于每个 $i \in  \left\lbrack  {1,\ell  - 1}\right\rbrack$，$E$ 中在 ${v}_{i}$ 和 ${v}_{i + 1}$ 之间都有一条边。如果路径中没有两个顶点是相同的，则该路径为简单路径。哈密顿路径是 $G$ 中长度为 $n$ 的简单路径（这样的路径必须恰好经过 $V$ 中的每个顶点一次）。判断 $G$ 是否存在哈密顿路径已知是 NP 难问题 [7]。

Let $R$ be a set of $n$ attributes: $\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{n}}\right\}$ . We will create $\left( \begin{array}{l} n \\  2 \end{array}\right)$ relations. Specifically,for each pair of $i,j$ such that $1 \leq  i < j \leq  n$ ,we generate a relation ${r}_{i,j}$ with attributes ${A}_{i},{A}_{j}$ . The tuples in ${r}_{i,j}$ are determined as follows:

设 $R$ 是一个包含 $n$ 个属性的集合：$\left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{n}}\right\}$。我们将创建 $\left( \begin{array}{l} n \\  2 \end{array}\right)$ 个关系。具体来说，对于每一对满足 $1 \leq  i < j \leq  n$ 的 $i,j$，我们生成一个具有属性 ${A}_{i},{A}_{j}$ 的关系 ${r}_{i,j}$。${r}_{i,j}$ 中的元组确定如下：

- Case $j = i + 1$ : Initially, ${r}_{i,j}$ is empty. For each edge $E$ between vertices $u$ and $v$ ,we add two tuples to ${r}_{i,j} : \left( {u,v}\right)$ and(v,u). In total, ${r}_{i,j}$ has ${2m}$ tuples.

- 情况 $j = i + 1$：最初，${r}_{i,j}$ 为空。对于顶点 $u$ 和 $v$ 之间的每条边 $E$，我们向 ${r}_{i,j} : \left( {u,v}\right)$ 中添加两个元组 (v,u) 和 (u,v)。总共，${r}_{i,j}$ 有 ${2m}$ 个元组。

- Case $j \geq  i + 2 : {r}_{i,j}$ contains $n\left( {n - 1}\right)$ tuples(x,y),for all possible integers $x,y$ such that $x \neq  y$ ,and $1 \leq  x,y \leq  n$ .

- 情况 $j \geq  i + 2 : {r}_{i,j}$ 包含 $n\left( {n - 1}\right)$ 个元组 (x,y)，其中 $x,y$ 为所有满足 $x \neq  y$ 和 $1 \leq  x,y \leq  n$ 的可能整数。

The total number of tuples in the ${r}_{i,j}$ of all possible $i,j$ is $O\left( {{nm} + {n}^{4}}\right)  = O\left( {n}^{4}\right)$ . Define:

所有可能的 $i,j$ 的 ${r}_{i,j}$ 中的元组总数为 $O\left( {{nm} + {n}^{4}}\right)  = O\left( {n}^{4}\right)$。定义：

CLIQUE = the output of the natural join of all ${r}_{i,j}\left( {1 \leq  i < j \leq  n}\right)$ .

CLIQUE = 所有 ${r}_{i,j}\left( {1 \leq  i < j \leq  n}\right)$ 的自然连接的输出。

As an example,for $n = 3$ ,CLIQUE $= {r}_{1,2} \bowtie  {r}_{1,3} \bowtie  {r}_{2,3}$ . In general,CLIQUE is a relation with schema $R$ . It should be easy to observe the fact below:

例如，对于 $n = 3$，CLIQUE $= {r}_{1,2} \bowtie  {r}_{1,3} \bowtie  {r}_{2,3}$。一般来说，CLIQUE 是一个具有模式 $R$ 的关系。应该很容易观察到以下事实：

Proposition 1. $G$ has a Hamiltonian path if and only if CLIQUE is not empty.

命题 1。$G$ 存在哈密顿路径当且仅当 CLIQUE 不为空。

For each pair of $i,j$ satisfying $1 \leq  i < j \leq  n$ ,define an attribute set ${R}_{i,j} = \left\{  {{A}_{i},{A}_{j}}\right\}$ . Denote by $J$ the JD that "corresponds to" CLIQUE,namely:

对于每一对满足 $1 \leq  i < j \leq  n$ 的 $i,j$，定义一个属性集 ${R}_{i,j} = \left\{  {{A}_{i},{A}_{j}}\right\}$。用 $J$ 表示与 CLIQUE “对应的” 连接依赖（Join Dependency，JD），即：

$$
J =  \bowtie  \left\lbrack  {{R}_{i,j},\forall i,j\text{ s.t. }1 \leq  i < j \leq  n}\right\rbrack  .
$$

For instance,for $n = 3,J =  \bowtie  \left\lbrack  {{R}_{1,2},{R}_{1,3},{R}_{2,3}}\right\rbrack$ . Note that $J$ has arity 2,and $R = { \cup  }_{i,j}{R}_{i,j}$ in general.

例如，对于 $n = 3,J =  \bowtie  \left\lbrack  {{R}_{1,2},{R}_{1,3},{R}_{2,3}}\right\rbrack$。注意 $J$ 的元数为 2，并且一般情况下 $R = { \cup  }_{i,j}{R}_{i,j}$。

Next,we will construct from $G$ a relation ${r}^{ * }$ of schema $R$ such that CLIQUE is empty if and only if ${r}^{ * }$ satisfies $J$ . Initially, ${r}^{ * }$ is empty. For every tuple $t$ in every relation ${r}_{i,j}\left( {1 \leq  i < j \leq  n}\right)$ , we will insert a tuple ${t}^{\prime }$ into ${r}^{ * }$ . Recall that ${r}_{i,j}$ has schema $\left\{  {{A}_{i},{A}_{j}}\right\}$ . Suppose,without loss of generality,that $t = \left( {{a}_{i},{a}_{j}}\right)$ . Then, ${t}^{\prime }$ is determined as follows:

接下来，我们将从$G$构造一个模式为$R$的关系${r}^{ * }$，使得团问题（CLIQUE）为空当且仅当${r}^{ * }$满足$J$。初始时，${r}^{ * }$为空。对于每个关系${r}_{i,j}\left( {1 \leq  i < j \leq  n}\right)$中的每个元组$t$，我们将向${r}^{ * }$中插入一个元组${t}^{\prime }$。回顾一下，${r}_{i,j}$的模式为$\left\{  {{A}_{i},{A}_{j}}\right\}$。不失一般性，假设$t = \left( {{a}_{i},{a}_{j}}\right)$。那么，${t}^{\prime }$的确定方式如下：

- ${t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}\left( {{t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack  }\right.$ is the value of $\left. {{t}^{\prime }\text{on attribute}{A}_{i}}\right)$

- ${t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}\left( {{t}^{\prime }\left\lbrack  {A}_{i}\right\rbrack  }\right.$是$\left. {{t}^{\prime }\text{on attribute}{A}_{i}}\right)$的值

- ${t}^{\prime }\left\lbrack  {A}_{j}\right\rbrack   = {a}_{j}$

- For any $k \in  \left\lbrack  {1,n}\right\rbrack$ but $k \neq  i$ and $k \neq  j,{t}^{\prime }\left\lbrack  {A}_{k}\right\rbrack$ is set to a dummy value that appears only once in the whole ${r}^{ * }$ .

- 对于任何$k \in  \left\lbrack  {1,n}\right\rbrack$，但$k \neq  i$和$k \neq  j,{t}^{\prime }\left\lbrack  {A}_{k}\right\rbrack$被设置为一个在整个${r}^{ * }$中仅出现一次的虚拟值。

Since there are $O\left( {n}^{4}\right)$ tuples in the ${r}_{i,j}$ of all $i,j$ ,we know that ${r}^{ * }$ has $O\left( {n}^{4}\right)$ tuples,and hence,can be built in $O\left( {n}^{5}\right)$ time.

由于所有$i,j$的${r}_{i,j}$中有$O\left( {n}^{4}\right)$个元组，我们知道${r}^{ * }$有$O\left( {n}^{4}\right)$个元组，因此可以在$O\left( {n}^{5}\right)$时间内构建。

Lemma 1. CLIQUE is empty if and only if ${r}^{ * }$ satisfies $J$ .

引理1. 团问题（CLIQUE）为空当且仅当${r}^{ * }$满足$J$。

Proof. We first point out three facts:

证明。我们首先指出三个事实：

1. Every tuple in ${r}^{ * }$ has $n - 2$ dummy values.

1. ${r}^{ * }$中的每个元组都有$n - 2$个虚拟值。

2. Define ${r}_{i,j}^{ * } = {\pi }_{{A}_{i},{A}_{j}}\left( {r}^{ * }\right)$ for $i,j$ satisfying $1 \leq  i < j \leq  n$ . Clearly, ${r}_{i,j}^{ * }$ and ${r}_{i,j}$ share the same schema ${R}_{i,j}$ . It is easy to verify that ${r}_{i,j}$ is exactly the set of tuples in ${r}_{i,j}^{ * }$ that do not contain dummy values.

2. 对于满足$1 \leq  i < j \leq  n$的$i,j$，定义${r}_{i,j}^{ * } = {\pi }_{{A}_{i},{A}_{j}}\left( {r}^{ * }\right)$。显然，${r}_{i,j}^{ * }$和${r}_{i,j}$具有相同的模式${R}_{i,j}$。很容易验证，${r}_{i,j}$恰好是${r}_{i,j}^{ * }$中不包含虚拟值的元组集合。

3. Define:

3. 定义：

${\text{CLIQUE}}^{ * } =$ the output of the natural join of all ${r}_{i,j}^{ * }\left( {1 \leq  i < j \leq  n}\right)$ .

${\text{CLIQUE}}^{ * } =$是所有${r}_{i,j}^{ * }\left( {1 \leq  i < j \leq  n}\right)$的自然连接的输出。

Then, ${r}^{ * }$ satisfies $J$ if and only if ${r}^{ * } = {\text{CLIQUE}}^{ * }$ .

那么，${r}^{ * }$满足$J$当且仅当${r}^{ * } = {\text{CLIQUE}}^{ * }$。

Equipped with these facts, we now proceed to prove the lemma.

有了这些事实，我们现在开始证明该引理。

For the "if" direction,assuming that ${r}^{ * }$ satisfies $J$ ,we need to show that CLIQUE is empty. Suppose,on the contrary,that $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$ is a tuple in CLIQUE. Hence, $\left( {{a}_{i},{a}_{j}}\right)$ is a tuple in ${r}_{i,j}$ for any $i,j$ satisfying $1 \leq  i < j \leq  n$ . As neither ${a}_{i}$ nor ${a}_{j}$ is dummy,by Fact 2,we know that $\left( {{a}_{i},{a}_{j}}\right)$ belongs to ${r}_{i,j}^{ * }$ . It thus follows that $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$ is a tuple in CLIQUE ${}^{ * }$ . However,by Fact $1,\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$ cannot belong to ${r}^{ * }$ ,thus giving a contradiction against Fact 3 .

对于“如果”方向，假设${r}^{ * }$满足$J$，我们需要证明团（CLIQUE）为空。相反，假设$\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$是团（CLIQUE）中的一个元组。因此，对于任何满足$1 \leq  i < j \leq  n$的$i,j$，$\left( {{a}_{i},{a}_{j}}\right)$是${r}_{i,j}$中的一个元组。由于${a}_{i}$和${a}_{j}$都不是虚拟值，根据事实2，我们知道$\left( {{a}_{i},{a}_{j}}\right)$属于${r}_{i,j}^{ * }$。由此可知，$\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$是团（CLIQUE ${}^{ * }$）中的一个元组。然而，根据事实$1,\left( {{a}_{1},{a}_{2},\ldots ,{a}_{n}}\right)$不能属于${r}^{ * }$，这与事实3矛盾。

For the "only-if" direction,assuming that CLIQUE is empty,we need to show that ${r}^{ * }$ satisfies $J$ . Suppose,on the contrary,that ${r}^{ * }$ does not satisfy $J$ ,namely, ${r}^{ * } \neq  {\text{CLIQUE }}^{ * }$ (Fact 3). Let $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ be a tuple in ${\mathrm{{CLIQUE}}}^{ * }$ but not in ${r}^{ * }$ . We distinguish two cases:

对于“仅当”方向，假设团（CLIQUE）为空，我们需要证明${r}^{ * }$满足$J$。相反，假设${r}^{ * }$不满足$J$，即${r}^{ * } \neq  {\text{CLIQUE }}^{ * }$（事实3）。设$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$是${\mathrm{{CLIQUE}}}^{ * }$中的一个元组，但不在${r}^{ * }$中。我们区分两种情况：

- Case 1: none of ${a}_{1}^{ * },\ldots ,{a}_{n}^{ * }$ is dummy. This means that,for any $i,j$ satisfying $1 \leq  i < j \leq  n$ , $\left( {{a}_{i}^{ * },{a}_{j}^{ * }}\right)$ is a tuple in ${r}_{i,j}$ (Fact 2). Therefore, $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ must be a tuple in CLIQUE, contradicting the assumption that CLIQUE is empty.

- 情况1：${a}_{1}^{ * },\ldots ,{a}_{n}^{ * }$中没有一个是虚拟值。这意味着，对于任何满足$1 \leq  i < j \leq  n$的$i,j$，$\left( {{a}_{i}^{ * },{a}_{j}^{ * }}\right)$是${r}_{i,j}$中的一个元组（事实2）。因此，$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$一定是团（CLIQUE）中的一个元组，这与团（CLIQUE）为空的假设相矛盾。

- Case 2: ${a}_{k}^{ * }$ is dummy for at least one $k \in  \left\lbrack  {1,n}\right\rbrack$ . Since every dummy value appears exactly once in ${r}^{ * }$ ,we can identify a unique tuple ${t}^{ * }$ in ${r}^{ * }$ such that ${t}^{ * }\left\lbrack  {A}_{k}\right\rbrack   = {a}_{k}^{ * }$ . Next,we will show that ${t}^{ * }$ is precisely $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ ,thus contradicting the assumption that $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ is not in ${r}^{ * }$ ,which will then complete the proof.

- 情况2：对于至少一个$k \in  \left\lbrack  {1,n}\right\rbrack$，${a}_{k}^{ * }$是虚拟值。由于每个虚拟值在${r}^{ * }$中恰好出现一次，我们可以在${r}^{ * }$中确定一个唯一的元组${t}^{ * }$，使得${t}^{ * }\left\lbrack  {A}_{k}\right\rbrack   = {a}_{k}^{ * }$。接下来，我们将证明${t}^{ * }$恰好是$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$，从而与$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$不在${r}^{ * }$中的假设相矛盾，这样就完成了证明。

Consider any $i$ such that $1 \leq  i < k$ . That $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ is in CLIQUE ${}^{ * }$ implies that $\left( {{a}_{i}^{ * },{a}_{k}^{ * }}\right)$ is in ${r}_{i,k}^{ * }$ . However,because in ${r}^{ * }$ the value ${a}_{k}^{ * }$ appears only in ${t}^{ * }$ ,it must hold that ${t}^{ * }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}^{ * }$ . By a similar argument,for any $j$ such that $k < j \leq  n$ ,we must have ${t}^{ * }\left\lbrack  {A}_{j}\right\rbrack   = {a}_{j}^{ * }$ . It thus follows that $\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$ is precisely ${t}^{ * }$ .

考虑任意满足$1 \leq  i < k$的$i$。$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$属于团问题（CLIQUE）${}^{ * }$意味着$\left( {{a}_{i}^{ * },{a}_{k}^{ * }}\right)$属于${r}_{i,k}^{ * }$。然而，因为在${r}^{ * }$中值${a}_{k}^{ * }$仅出现在${t}^{ * }$中，所以必然有${t}^{ * }\left\lbrack  {A}_{i}\right\rbrack   = {a}_{i}^{ * }$。通过类似的论证，对于任意满足$k < j \leq  n$的$j$，我们必然有${t}^{ * }\left\lbrack  {A}_{j}\right\rbrack   = {a}_{j}^{ * }$。因此，$\left( {{a}_{1}^{ * },{a}_{2}^{ * },\ldots ,{a}_{n}^{ * }}\right)$恰好就是${t}^{ * }$。

From the above discussion, we know that any 2-JD testing algorithm can be used to check whether CLIQUE is empty (Lemma 1),and hence,can be used to check whether $G$ has a Hamiltonian path (Lemma 1). We thus conclude that 2-JD testing is NP-hard.

从上述讨论可知，任何2 - 连接依赖（2 - JD）测试算法都可用于检查团问题（CLIQUE）是否为空（引理1），因此，也可用于检查$G$是否存在哈密顿路径（引理1）。由此我们得出结论：2 - 连接依赖测试是NP难问题。

## 3 LW Enumeration

## 3 轻量级（LW）枚举

The discussion from the previous section has eliminated the hope of efficient JD testing no matter how small the JD arity is (unless $\mathrm{P} = \mathrm{{NP}}$ ). We therefore switch to the less stringent goal of $\mathrm{{JD}}$ existence testing (Problem 2). Based on the reduction described in Section 1.1, next we concentrate on LW enumeration as formulated in Problem 3, and will establish Theorem 2.

上一节的讨论消除了无论连接依赖（JD）元数多小都能进行高效连接依赖测试的希望（除非$\mathrm{P} = \mathrm{{NP}}$）。因此，我们转向要求较低的$\mathrm{{JD}}$存在性测试目标（问题2）。基于1.1节中描述的归约，接下来我们专注于问题3中所阐述的轻量级（LW）枚举，并将证明定理2。

Let us recall a few basic definitions. We have a "global" set of attributes $R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,let ${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$ . We are given relations ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ where ${r}_{i}\left( {1 \leq  i \leq  d}\right)$ has schema ${R}_{i}$ . The objective of LW enumeration is that,for every tuple $t$ in the result of ${r}_{1} \bowtie  {r}_{2} \bowtie$ ... $\boxtimes  {r}_{d}$ ,we should invoke $\operatorname{emit}\left( t\right)$ once and exactly once. We want to do so I/O-efficiently in the EM model,where $B$ and $M$ represent the sizes (in words) of a disk block and memory,respectively.

让我们回顾几个基本定义。我们有一个“全局”属性集$R = \left\{  {{A}_{1},{A}_{2},\ldots ,{A}_{d}}\right\}$。对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，令${R}_{i} = R \smallsetminus  \left\{  {A}_{i}\right\}$。给定关系${r}_{1},{r}_{2},\ldots ,{r}_{d}$，其中${r}_{i}\left( {1 \leq  i \leq  d}\right)$的模式为${R}_{i}$。轻量级（LW）枚举的目标是，对于${r}_{1} \bowtie  {r}_{2} \bowtie$ ... $\boxtimes  {r}_{d}$结果中的每个元组$t$，我们应该且仅应该调用$\operatorname{emit}\left( t\right)$一次。我们希望在外部内存（EM）模型中以高效的输入/输出（I/O）方式完成此操作，其中$B$和$M$分别表示磁盘块和内存的大小（以字为单位）。

For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,set ${n}_{i} = \left| {r}_{i}\right|$ ,and define $\operatorname{dom}\left( {A}_{i}\right)$ as the domain of attribute ${A}_{i}$ . Given a tuple $t$ and an attribute ${A}_{i}$ (in the schema of the relation containing $t$ ),we denote by $t\left\lbrack  {A}_{i}\right\rbrack$ the value of $t$ on ${A}_{i}$ . Furthermore,we assume that each of ${r}_{1},\ldots ,{r}_{d}$ is given in an array,but the $d$ arrays do not need to be consecutive.

对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，设${n}_{i} = \left| {r}_{i}\right|$，并将$\operatorname{dom}\left( {A}_{i}\right)$定义为属性${A}_{i}$的域。给定一个元组$t$和一个属性${A}_{i}$（在包含$t$的关系模式中），我们用$t\left\lbrack  {A}_{i}\right\rbrack$表示$t$在${A}_{i}$上的值。此外，我们假设每个${r}_{1},\ldots ,{r}_{d}$都以数组形式给出，但$d$个数组不需要是连续的。

### 3.1 Basic Algorithms

### 3.1 基本算法

Let us first deal with two scenarios under which LW enumeration is easier.

让我们首先处理轻量级（LW）枚举较容易的两种情况。

#### 3.1.1 Small Join

#### 3.1.1 小连接

The first scenario arises when there is an ${n}_{i}$ (for some $i \in  \left\lbrack  {1,d}\right\rbrack$ ) satisfying ${n}_{i} = O\left( {M/d}\right)$ . In such a case,we call ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ a small join. Next,we prove:

第一种情况出现在存在一个${n}_{i}$（对于某个$i \in  \left\lbrack  {1,d}\right\rbrack$）满足${n}_{i} = O\left( {M/d}\right)$时。在这种情况下，我们称${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$为小并（small join）。接下来，我们证明：

Lemma 2. Given a small join,we can emit all its result tuples in $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) }\right) I/{Os}$ .

引理2. 给定一个小连接，我们可以在$O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) }\right) I/{Os}$内输出其所有结果元组。

Without loss of generality,suppose that ${r}_{1}$ has the smallest cardinality among all the input relations. Let us first assume that ${n}_{1} \leq  {cM}/d$ where $c$ is a sufficiently small constant so that ${r}_{1}$ can be kept in memory throughout the entire algorithm. With ${r}_{1}$ already in memory,we merge all the tuples of ${r}_{2},\ldots ,{r}_{d}$ into a set $L$ ,sorted by attribute ${A}_{1}$ . For each $a \in  \operatorname{dom}\left( {A}_{1}\right)$ ,let $L\left\lbrack  a\right\rbrack$ be the set of tuples in $L$ whose ${A}_{1}$ -values equal $a$ .

不失一般性，假设${r}_{1}$在所有输入关系中基数最小。首先，我们假设${n}_{1} \leq  {cM}/d$，其中$c$是一个足够小的常数，使得在整个算法过程中${r}_{1}$可以保存在内存中。由于${r}_{1}$已经在内存中，我们将${r}_{2},\ldots ,{r}_{d}$的所有元组合并到一个集合$L$中，并按属性${A}_{1}$排序。对于每个$a \in  \operatorname{dom}\left( {A}_{1}\right)$，令$L\left\lbrack  a\right\rbrack$为$L$中${A}_{1}$值等于$a$的元组集合。

Next,for each $a \in  \operatorname{dom}\left( {A}_{1}\right)$ ,we use the procedure below to emit all such tuples ${t}^{ * } \in  {r}_{1} \bowtie  {r}_{2} \bowtie$ $\ldots  \boxtimes  {r}_{d}$ that ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   = a$ . First,initialize empty sets ${S}_{2},\ldots ,{S}_{d}$ in memory. Then,process each tuple $t \in  L\left\lbrack  a\right\rbrack$ as follows. Suppose that $t$ originates from ${r}_{i}$ for some $i \in  \left\lbrack  {2,d}\right\rbrack$ . Check whether ${r}_{1}$ has a tuple ${t}^{\prime }$ satisfying

接下来，对于每个$a \in  \operatorname{dom}\left( {A}_{1}\right)$，我们使用以下过程输出所有满足${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   = a$的元组${t}^{ * } \in  {r}_{1} \bowtie  {r}_{2} \bowtie$ $\ldots  \boxtimes  {r}_{d}$。首先，在内存中初始化空集${S}_{2},\ldots ,{S}_{d}$。然后，按如下方式处理每个元组$t \in  L\left\lbrack  a\right\rbrack$。假设$t$来自某个$i \in  \left\lbrack  {2,d}\right\rbrack$对应的${r}_{i}$。检查${r}_{1}$是否有一个元组${t}^{\prime }$满足

$$
{t}^{\prime }\left\lbrack  {A}_{j}\right\rbrack   = t\left\lbrack  {A}_{j}\right\rbrack  ,\;\forall j \in  \left\lbrack  {2,d}\right\rbrack   \smallsetminus  \{ i\} . \tag{1}
$$

If the answer is no, $t$ is discarded; otherwise,we add it to ${S}_{i}$ . Note that the checking happens in memory and entails no I/O. Having processed all the tuples of $L\left\lbrack  a\right\rbrack$ this way,we emit all the tuples in the result of ${r}_{1} \bowtie  {S}_{2} \bowtie  {S}_{3} \bowtie  \ldots  \bowtie  {S}_{d}$ (these are exactly the tuples in ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ whose ${A}_{1}$ -values equal $a$ ). The above tuple emission incurs no I/Os due to the following lemma.

如果答案是否定的，则丢弃$t$；否则，将其添加到${S}_{i}$中。请注意，检查是在内存中进行的，不涉及输入/输出（I/O）操作。以这种方式处理完$L\left\lbrack  a\right\rbrack$的所有元组后，我们输出${r}_{1} \bowtie  {S}_{2} \bowtie  {S}_{3} \bowtie  \ldots  \bowtie  {S}_{d}$结果中的所有元组（这些元组恰好是${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$中${A}_{1}$值等于$a$的元组）。由于以下引理，上述元组输出不会产生I/O操作。

Lemma 3. ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ fit in memory.

引理3. ${r}_{1},{S}_{2},\ldots ,{S}_{d}$可以放入内存中。

Proof. It is easy to show that $\left| {S}_{i}\right|  \leq  {n}_{1} \leq  {cM}/d$ for each $i \in  \left\lbrack  {2,d}\right\rbrack$ . A naive way to store ${S}_{i}$ takes $d\left| {S}_{i}\right|$ words,in which case we would need $\Omega \left( {dM}\right)$ words to store ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ ,exceeding the memory capacity $M$ . To remedy this issue,we store ${S}_{i}$ using only $\left| {S}_{i}\right|$ words as follows. Given a tuple $t \in  {S}_{i}$ , we store a single integer that is the memory address of the tuple ${t}^{\prime }$ in (1),which requires only ${\lg }_{2}{n}_{1}$ bits by storing an offset. This does not lose any information because we can recover $t$ by resorting to (1) and the fact that $t\left\lbrack  {A}_{1}\right\rbrack   = a$ . Therefore, ${r}_{1},{S}_{2},\ldots ,{S}_{d}$ can be represented in $O\left( {d \cdot  {n}_{1}}\right)$ words, which is smaller than $M$ when the constant $c$ is sufficiently small.

证明。很容易证明，对于每个$i \in  \left\lbrack  {2,d}\right\rbrack$，有$\left| {S}_{i}\right|  \leq  {n}_{1} \leq  {cM}/d$。一种简单的存储${S}_{i}$的方法需要$d\left| {S}_{i}\right|$个字，在这种情况下，我们需要$\Omega \left( {dM}\right)$个字来存储${r}_{1},{S}_{2},\ldots ,{S}_{d}$，这超出了内存容量$M$。为了解决这个问题，我们按如下方式仅用$\left| {S}_{i}\right|$个字来存储${S}_{i}$。给定一个元组$t \in  {S}_{i}$，我们存储一个整数，它是元组${t}^{\prime }$在(1)中的内存地址，通过存储一个偏移量，这仅需要${\lg }_{2}{n}_{1}$位。这不会丢失任何信息，因为我们可以借助(1)以及$t\left\lbrack  {A}_{1}\right\rbrack   = a$这一事实来恢复$t$。因此，${r}_{1},{S}_{2},\ldots ,{S}_{d}$可以用$O\left( {d \cdot  {n}_{1}}\right)$个字来表示，当常数$c$足够小时，这个值小于$M$。

The overall cost of the algorithm is dominated by the cost of (i) merging ${r}_{2},\ldots ,{r}_{d}$ into $L$ ,which takes $O\left( {d + \left( {d/B}\right) \mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right)$ I/Os,and (ii) sorting $L$ ,which takes $O\left( {\operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right) }\right)$ I/Os,using an algorithm of [3] (see Theorem 1 of [3]) for string sorting in EM. Hence, the overall I/O complexity is as claimed in Lemma 2.

该算法的总体成本主要由以下两部分成本决定：(i) 将${r}_{2},\ldots ,{r}_{d}$合并到$L$中，这需要$O\left( {d + \left( {d/B}\right) \mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right)$次输入/输出操作；(ii) 对$L$进行排序，这需要$O\left( {\operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 2}}^{d}{n}_{i}}\right) }\right)$次输入/输出操作，这里使用了文献[3]中的算法（见文献[3]的定理1）在外部内存（EM）中进行字符串排序。因此，总体输入/输出复杂度如引理2所述。

For the case where ${n}_{1} > {cM}/d$ ,we simply divide ${r}_{1}$ arbitrarily into $O\left( 1\right)$ subsets each with ${cM}/d$ tuples,and then apply the above algorithm to emit all the result tuples produced from each of the subsets. This concludes the proof of Lemma 2.

对于${n}_{1} > {cM}/d$的情况，我们只需将${r}_{1}$任意划分为$O\left( 1\right)$个子集，每个子集包含${cM}/d$个元组，然后应用上述算法来输出从每个子集产生的所有结果元组。至此，引理2的证明结束。

#### 3.1.2 Point Join

#### 3.1.2 点连接

The second scenario where LW enumeration is relatively easy takes a bit more efforts to explain. In addition to ${r}_{1},\ldots ,{r}_{d}$ ,we accept two more input parameters:

第二种轻量级（LW）枚举相对容易的场景需要多花点力气来解释。除了${r}_{1},\ldots ,{r}_{d}$之外，我们还接受另外两个输入参数：

- an integer $H \in  \left\lbrack  {1,d}\right\rbrack$

- 一个整数$H \in  \left\lbrack  {1,d}\right\rbrack$

- a value $a \in  \operatorname{dom}\left( {A}_{H}\right)$ .

- 一个值$a \in  \operatorname{dom}\left( {A}_{H}\right)$。

It is required that $a$ should be the only value that appears in the ${A}_{H}$ attributes of ${r}_{1},\ldots ,{r}_{H - 1},{r}_{H + 1},\ldots ,{r}_{d}$ (recall that ${r}_{H}$ does not have $\left. {A}_{H}\right)$ . In such a case,we call ${r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$ a point join. Next, we prove:

要求$a$应该是在${r}_{1},\ldots ,{r}_{H - 1},{r}_{H + 1},\ldots ,{r}_{d}$的${A}_{H}$属性中出现的唯一值（回想一下，${r}_{H}$没有$\left. {A}_{H}\right)$）。在这种情况下，我们称${r}_{1} \boxtimes  {r}_{2} \boxtimes  \ldots  \boxtimes  {r}_{d}$为点连接。接下来，我们证明：

Lemma 4. Given a point join,we can emit all its result tuples in $O\left( {d + \operatorname{sort}\left( {{d}^{2}{n}_{H} + }\right. }\right.$ $\left. \left. {d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\} }}{n}_{i}}\right) \right) I/{Os}$ .

引理4。给定一个点连接，我们可以在$O\left( {d + \operatorname{sort}\left( {{d}^{2}{n}_{H} + }\right. }\right.$ $\left. \left. {d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\} }}{n}_{i}}\right) \right) I/{Os}$内输出其所有结果元组。

Proof. For each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,define ${X}_{i} = {R}_{i} \cap  {R}_{H}$ (i.e., ${X}_{i}$ includes all the attributes in $R$ except $\left. {{A}_{i}\text{ and }{A}_{H}}\right)$ .

证明。对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$，定义 ${X}_{i} = {R}_{i} \cap  {R}_{H}$（即，${X}_{i}$ 包含 $R$ 中除 $\left. {{A}_{i}\text{ and }{A}_{H}}\right)$ 之外的所有属性。

In ascending order of $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we invoke the procedure below to process ${r}_{i}$ and ${r}_{H}$ , which continuously removes some tuples from ${r}_{H}$ . First,sort ${r}_{i}$ and ${r}_{H}$ by ${X}_{i}$ ,respectively. Then, synchronously scan ${r}_{i}$ and ${r}_{H}$ according to the sorted order. For each tuple $t$ in ${r}_{H}$ ,we check during the scan whether ${r}_{i}$ has a tuple ${t}^{\prime }$ that has the same values as $t$ on all the attributes in ${X}_{i}$ -note that such ${t}^{\prime }$ (if exists) must be unique,due to the fact that $a$ is the only ${A}_{H}$ value in ${r}_{i}$ . The sorted order ensures that if ${t}^{\prime }$ exists,then $t$ and ${t}^{\prime }$ must appear consecutively during the synchronous scan. If ${t}^{\prime }$ exists, $t$ is kept in ${r}_{H}$ ; otherwise,we discard $t$ from ${r}_{H}$ ( $t$ cannot produce any tuple in ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ ).

按照 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ 的升序，我们调用以下过程来处理 ${r}_{i}$ 和 ${r}_{H}$，该过程会不断从 ${r}_{H}$ 中移除一些元组。首先，分别按照 ${X}_{i}$ 对 ${r}_{i}$ 和 ${r}_{H}$ 进行排序。然后，根据排序顺序同步扫描 ${r}_{i}$ 和 ${r}_{H}$。对于 ${r}_{H}$ 中的每个元组 $t$，我们在扫描过程中检查 ${r}_{i}$ 是否有一个元组 ${t}^{\prime }$，该元组在 ${X}_{i}$ 中的所有属性上与 $t$ 具有相同的值——注意，由于 $a$ 是 ${r}_{i}$ 中唯一的 ${A}_{H}$ 值，这样的 ${t}^{\prime }$（如果存在）必定是唯一的。排序顺序确保如果 ${t}^{\prime }$ 存在，那么 $t$ 和 ${t}^{\prime }$ 在同步扫描期间必定连续出现。如果 ${t}^{\prime }$ 存在，则将 $t$ 保留在 ${r}_{H}$ 中；否则，我们从 ${r}_{H}$ 中丢弃 $t$（$t$ 无法在 ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 中产生任何元组）。

After the above procedure has finished through all $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we know that every tuple $t$ remaining in ${r}_{H}$ must produce exactly one result tuple ${t}^{\prime \prime }$ in ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ where ${t}^{\prime \prime }\left\lbrack  {A}_{i}\right\rbrack   = t\left\lbrack  {A}_{i}\right\rbrack$ for all $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,and ${t}^{\prime }\left\lbrack  {A}_{H}\right\rbrack   = a$ . Therefore,we can emit all such ${t}^{\prime }$ with one more scan of the (current) ${r}_{H}$ .

在上述过程对所有 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ 执行完毕后，我们知道 ${r}_{H}$ 中剩余的每个元组 $t$ 必定会在 ${r}_{1} \bowtie  {r}_{2} \bowtie  \ldots  \bowtie  {r}_{d}$ 中恰好产生一个结果元组 ${t}^{\prime \prime }$，其中对于所有 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ 有 ${t}^{\prime \prime }\left\lbrack  {A}_{i}\right\rbrack   = t\left\lbrack  {A}_{i}\right\rbrack$，并且 ${t}^{\prime }\left\lbrack  {A}_{H}\right\rbrack   = a$。因此，我们可以通过对（当前的）${r}_{H}$ 再进行一次扫描来输出所有这样的 ${t}^{\prime }$。

The claimed I/O cost follows from the fact that ${r}_{H}$ is sorted $d - 1$ times in total,while ${r}_{i}$ is sorted once for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ .

所声称的 I/O 成本源于这样一个事实，即 ${r}_{H}$ 总共被排序 $d - 1$ 次，而 ${r}_{i}$ 对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ 被排序一次。

In what follows,we will denote the algorithm in the above lemma as PTJOIN $\left( {H,a,{r}_{1},{r}_{2},\ldots ,{r}_{d}}\right)$ .

在接下来的内容中，我们将上述引理中的算法记为 PTJOIN $\left( {H,a,{r}_{1},{r}_{2},\ldots ,{r}_{d}}\right)$。

### 3.2 The Full Algorithm

### 3.2 完整算法

This subsection presents an algorithm for solving the general LW enumeration problem. If ${n}_{1} \leq$ ${2M}/d$ ,we solve the problem directly by Lemma 2. The subsequent discussion focuses on ${n}_{1} > {2M}/d$ .

本小节提出一种用于解决一般 LW 枚举问题的算法。如果 ${n}_{1} \leq$ ${2M}/d$，我们直接根据引理 2 来解决该问题。后续讨论将聚焦于 ${n}_{1} > {2M}/d$。

Define:

定义：

$$
U = {\left( \frac{\mathop{\prod }\limits_{{i = 1}}^{d}{n}_{i}}{M}\right) }^{\frac{1}{d - 1}} \tag{2}
$$

$$
{\tau }_{i} = \frac{{n}_{1}{n}_{2}\ldots {n}_{i}}{{\left( U \cdot  {d}^{\frac{1}{d - 1}}\right) }^{i - 1}}\text{ for each }i \in  \left\lbrack  {1,d}\right\rbrack  . \tag{3}
$$

Notice that ${\tau }_{1} = {n}_{1}$ and ${\tau }_{d} = M/d$ .

注意到 ${\tau }_{1} = {n}_{1}$ 和 ${\tau }_{d} = M/d$。

Our algorithm is a recursive procedure $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,which has three requirements:

我们的算法是一个递归过程$\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$，它有三个要求：

- $h$ is an integer in $\left\lbrack  {1,d}\right\rbrack$ ;

- $h$ 是 $\left\lbrack  {1,d}\right\rbrack$ 中的一个整数；

- Each ${\rho }_{i}\left( {1 \leq  i \leq  d}\right)$ is a subset of the tuples in ${r}_{i}$ .

- 每个 ${\rho }_{i}\left( {1 \leq  i \leq  d}\right)$ 是 ${r}_{i}$ 中元组的一个子集。

- The size of ${\rho }_{1}$ satisfies:

- ${\rho }_{1}$ 的大小满足：

$$
\left| {\rho }_{1}\right|  \leq  {\tau }_{h} \tag{4}
$$

JOIN $\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ emits all result tuples in ${\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ . The LW enumeration problem can be settled by calling $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ .

连接 $\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 会输出 ${\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ 中的所有结果元组。轻量级（LW）枚举问题可以通过调用 $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 来解决。

If ${\tau }_{h} \leq  {2M}/d,\left| {\rho }_{1}\right|  \leq  {\tau }_{h} = O\left( {M/d}\right)$ ; JOIN $\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ simply runs the small-join algorithm of Lemma 2. Next,we focus on ${\tau }_{h} > {2M}/d$ . Define:

如果 ${\tau }_{h} \leq  {2M}/d,\left| {\rho }_{1}\right|  \leq  {\tau }_{h} = O\left( {M/d}\right)$；连接 $\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 只需运行引理 2 的小连接算法。接下来，我们关注 ${\tau }_{h} > {2M}/d$。定义：

$$
H = \min \left\{  {i \in  \left\lbrack  {h + 1,d}\right\rbrack   \mid  {\tau }_{i} < {\tau }_{h}/2}\right\}  . \tag{5}
$$

$H$ always exists because ${\tau }_{d} = M/d < {\tau }_{h}/2$ . Given a value $a \in  \operatorname{dom}\left( {A}_{H}\right)$ ,define

$H$ 总是存在，因为 ${\tau }_{d} = M/d < {\tau }_{h}/2$。给定一个值 $a \in  \operatorname{dom}\left( {A}_{H}\right)$，定义

$$
\operatorname{freq}\left( a\right)  = \left| \left\{  {t \in  {\rho }_{1} \mid  t\left\lbrack  {A}_{H}\right\rbrack   = a}\right\}  \right| .
$$

We collect all the frequent values $a$ into a set $\Phi$ :

我们将所有频繁值 $a$ 收集到一个集合 $\Phi$ 中：

$$
\Phi  = \left\{  {a \in  \operatorname{dom}\left( {A}_{H}\right)  \mid  \operatorname{freq}\left( a\right)  > {\tau }_{H}/2}\right\}  . \tag{6}
$$

For each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we partition ${\rho }_{i}$ into:

对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$，我们将 ${\rho }_{i}$ 划分为：

$$
{\rho }_{i}^{\text{heavy }} = \left\{  {t \in  {\rho }_{i} \mid  t\left\lbrack  {A}_{H}\right\rbrack   \in  \Phi }\right\}  
$$

$$
{\rho }_{i}^{\text{light }} = \left\{  {t \in  {\rho }_{i} \mid  t\left\lbrack  {A}_{H}\right\rbrack   \notin  \Phi }\right\}  
$$

It is rudimentary to produce $\Phi$ ,as well as ${\rho }_{i}^{\text{heavy }}$ and ${\rho }_{i}^{\text{light }}$ for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,by sorting on ${A}_{H}$ . Specifically,each element to be sorted is a tuple of $d - 1$ values where $d \leq  M/2$ (see the definition of Problem 3). Using an EM string sorting algorithm of [3], all the sorting can be completed with $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \smallsetminus \{ H\} }}\left| {\rho }_{i}\right| }\right) }\right)$ I/Os in total.

通过对 ${A}_{H}$ 进行排序来生成 $\Phi$ 以及每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ 对应的 ${\rho }_{i}^{\text{heavy }}$ 和 ${\rho }_{i}^{\text{light }}$ 是基本操作。具体来说，要排序的每个元素是一个包含 $d - 1$ 个值的元组，其中 $d \leq  M/2$（见问题 3 的定义）。使用文献[3]中的 EM 字符串排序算法，所有排序总共可以用 $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \smallsetminus \{ H\} }}\left| {\rho }_{i}\right| }\right) }\right)$ 次输入/输出（I/O）完成。

A result tuple ${t}^{ * } \in  {\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ is said to be heavy if ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   \in  \Phi$ ,or light otherwise. The set of heavy tuples is precisely

如果 ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   \in  \Phi$，则称结果元组 ${t}^{ * } \in  {\rho }_{1} \bowtie  \ldots  \bowtie  {\rho }_{d}$ 为重元组，否则为轻元组。重元组的集合恰好是

$$
{\rho }_{1}^{\text{heavy }} \bowtie  \ldots  \bowtie  {\rho }_{H - 1}^{\text{heavy }} \bowtie  {\rho }_{H} \bowtie  {\rho }_{H + 1}^{\text{heavy }} \bowtie  \ldots  \bowtie  {\rho }_{d}^{\text{heavy }},
$$

whereas the set of light tuples is

而轻元组的集合是

$$
{\rho }_{1}^{\text{light }} \bowtie  \ldots  \bowtie  {\rho }_{H - 1}^{\text{light }} \bowtie  {\rho }_{H} \bowtie  {\rho }_{H + 1}^{\text{light }} \bowtie  \ldots  \bowtie  {\rho }_{d}^{\text{light }}.
$$

We will emit heavy and light tuples separately, as explained next.

接下来将解释，我们将分别输出重元组和轻元组。

#### 3.2.1 Enumerating Heavy Tuples

#### 3.2.1 枚举重元组

For every $a \in  \Phi$ ,define for each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ :

对于每个 $a \in  \Phi$，为每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ 定义：

$$
{\rho }_{i}^{\text{heavy }}\left\lbrack  a\right\rbrack   = \left\{  {t \in  {\rho }_{i}^{\text{heavy }} \mid  t\left\lbrack  {A}_{H}\right\rbrack   = a}\right\}  .
$$

The tuples of ${\rho }_{i}^{\text{heavy }}\left\lbrack  a\right\rbrack$ are stored consecutively in the disk because we have sorted ${\rho }_{i}^{\text{heavy }}$ by ${A}_{H}$ earlier. Hence,all the heavy tuples ${t}^{ * }$ with ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   = a$ can be emitted by a point join:

由于我们之前已经按 ${A}_{H}$ 对 ${\rho }_{i}^{\text{heavy }}$ 进行了排序，所以 ${\rho }_{i}^{\text{heavy }}\left\lbrack  a\right\rbrack$ 中的元组在磁盘上是连续存储的。因此，所有满足 ${t}^{ * }\left\lbrack  {A}_{H}\right\rbrack   = a$ 的重元组 ${t}^{ * }$ 可以通过点连接输出：

$$
\operatorname{PTJOIN}\left( {H,a,{\rho }_{1}^{\text{heavy }}\left\lbrack  a\right\rbrack  ,\ldots ,{\rho }_{H - 1}^{\text{heavy }}\left\lbrack  a\right\rbrack  ,{\rho }_{H},{\rho }_{H + 1}^{\text{heavy }}\left\lbrack  a\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{heavy }}\left\lbrack  a\right\rbrack  }\right) \text{.}
$$

#### 3.2.2 Enumerating Light Tuples

#### 3.2.2 枚举轻元组

For each $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ,given an interval $I$ in $\operatorname{dom}\left( {A}_{H}\right)$ ,define:

对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \{ H\}$ ，给定 $\operatorname{dom}\left( {A}_{H}\right)$ 中的一个区间 $I$ ，定义：

$$
{\rho }_{i}^{\text{light }}\left\lbrack  I\right\rbrack   = \left\{  {t \in  {\rho }_{i}^{\text{light }} \mid  t\left\lbrack  {A}_{H}\right\rbrack   \in  I}\right\}  .
$$

With one scan of ${\rho }_{1}$ (which has been sorted by ${A}_{H}$ ),we can obtain a sequence of disjoint intervals ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ with the properties below:

通过对 ${\rho }_{1}$ 进行一次扫描（${\rho }_{1}$ 已按 ${A}_{H}$ 排序），我们可以得到一系列不相交的区间 ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ ，这些区间具有以下性质：

- $q = O\left( {1 + \left| {\rho }_{1}\right| /{\tau }_{H}}\right)$ ;

- ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ are in ascending order,and constitute a partition of $\operatorname{dom}\left( {A}_{H}\right)$ ;

- ${I}_{1},{I}_{2},\ldots ,{I}_{q}$ 按升序排列，并构成 $\operatorname{dom}\left( {A}_{H}\right)$ 的一个划分；

- The following balancing condition is fulfilled:

- 满足以下平衡条件：

$$
 - \left| {{\rho }_{1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  }\right|  \in  \left\lbrack  {{\tau }_{H}/2,{\tau }_{H}}\right\rbrack  \text{ for every }j \in  \left\lbrack  {1,q - 1}\right\rbrack  ;
$$

$$
 - \left| {{\rho }_{1}^{\text{light }}\left\lbrack  {I}_{q}\right\rbrack  }\right|  \in  \left\lbrack  {1,{\tau }_{H}}\right\rbrack  
$$

Next,for all $i \in  \left\lbrack  {2,d}\right\rbrack   \smallsetminus  \{ H\}$ ,we produce ${\rho }_{i}^{\text{light}}\left\lbrack  {I}_{1}\right\rbrack  ,{\rho }_{i}^{\text{light}}\left\lbrack  {I}_{2}\right\rbrack  ,\ldots ,{\rho }_{i}^{\text{light}}\left\lbrack  {I}_{q}\right\rbrack$ by sorting. To emit all the light tuples,we recursively invoke our algorithm for each $j \in  \left\lbrack  {1,q}\right\rbrack$ :

接下来，对于所有 $i \in  \left\lbrack  {2,d}\right\rbrack   \smallsetminus  \{ H\}$ ，我们通过排序生成 ${\rho }_{i}^{\text{light}}\left\lbrack  {I}_{1}\right\rbrack  ,{\rho }_{i}^{\text{light}}\left\lbrack  {I}_{2}\right\rbrack  ,\ldots ,{\rho }_{i}^{\text{light}}\left\lbrack  {I}_{q}\right\rbrack$ 。为了输出所有轻元组，我们针对每个 $j \in  \left\lbrack  {1,q}\right\rbrack$ 递归调用我们的算法：

$$
\operatorname{JOIN}\left( {H,{\rho }_{1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{H - 1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  ,{\rho }_{H},{\rho }_{H + 1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  }\right) \text{.}
$$

This completes the description of our LW enumeration algorithm.

至此，我们完成了对轻元组（LW）枚举算法的描述。

We remark that a crucial design of our recursive machinery is in how the parameter $h$ in JOIN $\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ increases. Specficailly,this parameter does not increase by 1 each time; rather, it will be set to the value of $H$ as given by (5),which is at least $h + 1$ but can be larger.

我们注意到，我们递归机制的一个关键设计在于连接操作（JOIN） $\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 中的参数 $h$ 是如何递增的。具体来说，该参数并非每次递增 1；相反，它将被设置为 (5) 式所给出的 $H$ 的值，该值至少为 $h + 1$ ，但可能更大。

### 3.3 A Recurrence on the I/O Cost

### 3.3 关于输入/输出（I/O）成本的递推关系

This and the next subsections will analyze the performance of our algorithm. Define a sequence of integers as follows:

本节和下一小节将分析我们算法的性能。定义一个整数序列如下：

- ${h}_{1} = 1$ ;

- Provided that ${h}_{i}\left( {i \geq  1}\right)$ has been defined:

- 假设 ${h}_{i}\left( {i \geq  1}\right)$ 已被定义：

$$
\text{- if}{\tau }_{{h}_{i}} > {2M}/d\text{,define}{h}_{i + 1} = \min \left\{  {j \in  \left\lbrack  {1 + {h}_{i},d}\right\rbrack   \mid  {\tau }_{j} < {\tau }_{{h}_{i}}/2}\right\}  \text{;}
$$

- otherwise, ${h}_{i + 1}$ is undefined. Denote by $w$ the largest integer $j$ such that ${h}_{j}$ is defined.

- 否则，${h}_{i + 1}$ 未定义。用 $w$ 表示使得 ${h}_{j}$ 有定义的最大整数 $j$ 。

Recall that our LW enumeration algorithm starts by calling the JOIN procedure with $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ ,which recursively makes subsequent calls to the same procedure. These calls form a tree $\mathcal{T}$ . Equipped with the sequence ${h}_{1},{h}_{2},\ldots ,{h}_{w}$ ,we can describe $\mathcal{T}$ in a more specific manner. Given a call $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,let us refer to the value of $h$ as the call’s axis. The initial call $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ has axis ${h}_{1} = 1$ . In general,an axis- ${h}_{i}\left( {i \in  \left\lbrack  {1,w - 1}\right\rbrack  }\right)$ call generates axis- ${h}_{i + 1}$ calls,and hence,parents those calls in $\mathcal{T}$ . Finally,all axis- ${h}_{w}$ calls are leaf nodes in $\mathcal{T}$ (recall that an axis- ${h}_{w}$ call simply invokes the small-join algorithm of Lemma 2). In other words, $\mathcal{T}$ has $w$ levels; and all the calls at level $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ have an identical axis ${h}_{\ell }$ .

回顾一下，我们的轻元组（LW）枚举算法从调用连接过程（JOIN）开始，初始参数为 $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ ，该过程会递归地对同一过程进行后续调用。这些调用形成了一棵树 $\mathcal{T}$ 。借助序列 ${h}_{1},{h}_{2},\ldots ,{h}_{w}$ ，我们可以更具体地描述 $\mathcal{T}$ 。给定一个调用 $\operatorname{JOIN}\left( {h,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ，我们将 $h$ 的值称为该调用的轴。初始调用 $\operatorname{JOIN}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ 的轴为 ${h}_{1} = 1$ 。一般来说，轴为 ${h}_{i}\left( {i \in  \left\lbrack  {1,w - 1}\right\rbrack  }\right)$ 的调用会生成轴为 ${h}_{i + 1}$ 的调用，因此在树 $\mathcal{T}$ 中是这些调用的父节点。最后，所有轴为 ${h}_{w}$ 的调用都是树 $\mathcal{T}$ 中的叶节点（回顾一下，轴为 ${h}_{w}$ 的调用只是调用引理 2 中的小连接算法）。换句话说，树 $\mathcal{T}$ 有 $w$ 层；并且第 $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ 层的所有调用都有相同的轴 ${h}_{\ell }$ 。

Given a level $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ ,define function $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ to be the number of $\mathrm{I}/\mathrm{{Os}}$ performed by $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ . We will work out a recurrence on $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,and then concentrate on solving it.

给定一个层级 $\ell  \in  \left\lbrack  {1,w}\right\rbrack$，定义函数 $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 为 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 执行的 $\mathrm{I}/\mathrm{{Os}}$ 的数量。我们将推导出关于 $\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 的递推关系，然后专注于求解它。

The Base Case of the Recurrence. If $\ell  = w$ ,Lemma 2 immediately shows

递推关系的基础情况。如果 $\ell  = w$，引理 2 立即表明

$$
\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)  = O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right) . \tag{7}
$$

The General Case. For $\ell  \in  \left\lbrack  {1,w - 1}\right\rbrack$ ,define:

一般情况。对于 $\ell  \in  \left\lbrack  {1,w - 1}\right\rbrack$，定义：

$$
{\mu }_{\ell } = \frac{2{\tau }_{{h}_{\ell }}}{{\tau }_{{h}_{\ell  + 1}}}. \tag{8}
$$

By the way ${h}_{\ell  + 1}$ is selected,it must hold that ${\mu }_{\ell } \geq  4$ . Furthermore,we can prove:

根据 ${h}_{\ell  + 1}$ 的选取方式，必然有 ${\mu }_{\ell } \geq  4$。此外，我们可以证明：

Lemma 5. ${\mu }_{\ell  - 1} = O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$ for each $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ .

引理 5。对于每个 $\ell  \in  \left\lbrack  {2,w}\right\rbrack$，有 ${\mu }_{\ell  - 1} = O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$。

Proof. It suffices to show that ${\tau }_{{h}_{\ell  - 1}}/{\tau }_{{h}_{\ell }} = O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$ . From (3),we get

证明。只需证明 ${\tau }_{{h}_{\ell  - 1}}/{\tau }_{{h}_{\ell }} = O\left( {U{d}^{\frac{1}{d - 1}}/{n}_{{h}_{\ell }}}\right)$。由 (3) 可得

$$
\frac{{\tau }_{{h}_{\ell  - 1}}}{{\tau }_{{h}_{\ell }}} = \frac{{\left( U{d}^{\frac{1}{d - 1}}\right) }^{{h}_{\ell } - {h}_{\ell  - 1}}}{\mathop{\prod }\limits_{{j = 1 + {h}_{\ell  - 1}}}^{{h}_{\ell }}{n}_{j}}. \tag{9}
$$

If ${h}_{\ell } = 1 + {h}_{\ell  - 1}$ ,then

如果 ${h}_{\ell } = 1 + {h}_{\ell  - 1}$，那么

$$
\left( 9\right)  = \frac{U{d}^{\frac{1}{d - 1}}}{{n}_{{h}_{\ell }}}.
$$

Otherwise $\left( {{h}_{\ell } > 1 + {h}_{\ell  - 1}}\right)$ ,the definition of ${h}_{\ell }$ indicates ${\tau }_{{h}_{\ell } - 1} \geq  {\tau }_{{h}_{\ell  - 1}}/2$ (otherwise, ${h}_{\ell }$ would not be the smallest integer $j \in  \left\lbrack  {1 + {h}_{\ell  - 1},d}\right\rbrack$ satisfying ${\tau }_{j} < {\tau }_{{h}_{\ell  - 1}}/2$ ),namely:

否则 $\left( {{h}_{\ell } > 1 + {h}_{\ell  - 1}}\right)$，${h}_{\ell }$ 的定义表明 ${\tau }_{{h}_{\ell } - 1} \geq  {\tau }_{{h}_{\ell  - 1}}/2$（否则，${h}_{\ell }$ 就不是满足 ${\tau }_{j} < {\tau }_{{h}_{\ell  - 1}}/2$ 的最小整数 $j \in  \left\lbrack  {1 + {h}_{\ell  - 1},d}\right\rbrack$），即：

$$
\frac{{\tau }_{{h}_{\ell  - 1}}}{{\tau }_{{h}_{\ell } - 1}} = \frac{{\left( U{d}^{\frac{1}{d - 1}}\right) }^{{h}_{\ell } - 1 - {h}_{\ell  - 1}}}{\mathop{\prod }\limits_{{j = 1 + {h}_{\ell  - 1}}}^{{{h}_{\ell } - 1}}{n}_{j}} \leq  2.
$$

Hence,

因此，

$$
\left( 9\right)  = \frac{{\left( U{d}^{\frac{1}{d - 1}}\right) }^{{h}_{\ell } - 1 - {h}_{\ell  - 1}}}{\mathop{\prod }\limits_{{j = 1 + {h}_{\ell  - 1}}}^{{{h}_{\ell } - 1}}{n}_{j}} \cdot  \frac{U{d}^{\frac{1}{d - 1}}}{{n}_{{h}_{\ell }}} \leq  2 \cdot  \frac{U{d}^{\frac{1}{d - 1}}}{{n}_{{h}_{\ell }}}.
$$

Consider the set $\Phi$ defined in (6). Recall that for every $a \in  \Phi$ ,freq $\left( a\right)  > {\tau }_{{h}_{\ell  + 1}}/2$ . Hence:

考虑 (6) 中定义的集合 $\Phi$。回想一下，对于每个 $a \in  \Phi$，频率为 $\left( a\right)  > {\tau }_{{h}_{\ell  + 1}}/2$。因此：

$$
\left| \Phi \right|  < \frac{\left| {\rho }_{1}\right| }{{\tau }_{{h}_{\ell  + 1}}/2} \leq  2\frac{{\tau }_{{h}_{\ell }}}{{\tau }_{{h}_{\ell  + 1}}} = {\mu }_{\ell }.
$$

where the second inequality is due to (4).

其中第二个不等式由 (4) 得出。

The number of $\mathrm{I}/\mathrm{{Os}}$ that $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ spends on emitting heavy tuples is dominated by that of the point joins performed, whose total I/O cost (by Lemma 4) is:

$\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 在输出重元组上花费的 $\mathrm{I}/\mathrm{{Os}}$ 的数量受执行的点连接数量的限制，根据引理 4，其总 I/O 成本为：

$$
O\left( {\mathop{\sum }\limits_{{a \in  \Phi }}\left( {d + \operatorname{sort}\left( {{d}^{2} \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack   \smallsetminus  \left\{  {h}_{\ell  + 1}\right\}  }}\left| {{\rho }_{i}^{\text{heavy }}\left\lbrack  a\right\rbrack  }\right| }\right) }\right) }\right) 
$$

$$
 = O\left( {d \cdot  \left| \Phi \right|  + \operatorname{sort}\left( {{d}^{2} \cdot  \left| \Phi \right|  \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right) 
$$

$$
 = O\left( {d \cdot  {\mu }_{\ell } + \operatorname{sort}\left( {{d}^{2} \cdot  {\mu }_{\ell } \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right) . \tag{10}
$$

The cost of emitting light tuples comes from recursion. Taking into account the fact that the sorting cost in Section 3.2.2 has been absorbed in (10), we have

输出轻元组的成本来自递归。考虑到 3.2.2 节中的排序成本已包含在 (10) 中，我们有

$$
\operatorname{cost}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right) 
$$

$$
 = \left( {10}\right)  + \mathop{\sum }\limits_{{j = 1}}^{q}\operatorname{cost}\left( {\ell  + 1,{\rho }_{1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{{h}_{\ell  + 1} - 1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  ,{\rho }_{{h}_{\ell  + 1}},{\rho }_{{h}_{\ell  + 1} + 1}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  ,\ldots ,{\rho }_{d}^{\text{light }}\left\lbrack  {I}_{j}\right\rbrack  }\right)  \tag{11}
$$

where $q$ is the number of disjoint intervals that $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ uses to divide $\operatorname{dom}\left( {A}_{\ell }\right)$ (see Section 3.2.2). By the balancing condition in Section 3.2.2, it must hold that

其中 $q$ 是 $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 用于划分 $\operatorname{dom}\left( {A}_{\ell }\right)$ 的不相交区间的数量（见 3.2.2 节）。根据 3.2.2 节中的平衡条件，必然有

$$
q = O\left( {1 + \left| {\rho }_{1}\right| /{\tau }_{{h}_{\ell  + 1}}}\right) 
$$

$$
\text{(by}\left( 4\right) ) = O\left( {1 + {\tau }_{{h}_{\ell }}/{\tau }_{{h}_{\ell  + 1}}}\right) 
$$

$$
 = O\left( {\mu }_{\ell }\right) \text{.} \tag{12}
$$

The rest of the section is devoted to analyzing the above non-conventional recurrence.

本节的其余部分致力于分析上述非常规递推关系。

### 3.4 Solving the Recurrence

### 3.4 求解递推关系

Our objective is to prove that $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ -which gives the number of I/Os of our LW enumeration algorithm - is as claimed in Theorem 2. We will do so by resorting to the recursion tree $\mathcal{T}$ (as is defined in Section 3.3). Specifically,for each node $\operatorname{JOIN}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ ,we associate it with cost

我们的目标是证明 $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$（它给出了我们的 LW 枚举算法的输入/输出（I/O）数量）如定理 2 所声称的那样。我们将通过借助递归树 $\mathcal{T}$（如第 3.3 节所定义）来实现这一目标。具体而言，对于每个节点 $\operatorname{JOIN}\left( {\ell ,{\rho }_{1},\ldots ,{\rho }_{d}}\right)$，我们为其关联成本

- $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right)$ ,if it is a leaf in $\mathcal{T}$ . We account for the two terms in different ways. First,the term $O\left( d\right)$ is charged as the factual cost on the leaf itself. On the other hand,on every tuple in ${\rho }_{i}$ (of all $i \in  \left\lbrack  {1,d}\right\rbrack$ ),we charge a nominal cost of $O\left( d\right)$ . In this way,the second term $O\left( {\operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right)$ equals $O\left( {\operatorname{sort}\left( x\right) }\right)$ ,where $x$ is the sum of the nominal costs of all the tuples in ${\rho }_{1},\ldots ,{\rho }_{d}$ .

- $O\left( {d + \operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right)$，如果它是 $\mathcal{T}$ 中的一个叶节点。我们以不同的方式处理这两项。首先，项 $O\left( d\right)$ 作为叶节点本身的实际成本进行计费。另一方面，对于 ${\rho }_{i}$（所有 $i \in  \left\lbrack  {1,d}\right\rbrack$ 的）中的每个元组，我们收取 $O\left( d\right)$ 的名义成本。通过这种方式，第二项 $O\left( {\operatorname{sort}\left( {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right)$ 等于 $O\left( {\operatorname{sort}\left( x\right) }\right)$，其中 $x$ 是 ${\rho }_{1},\ldots ,{\rho }_{d}$ 中所有元组的名义成本之和。

- $O\left( {d \cdot  {\mu }_{\ell } + \operatorname{sort}\left( {{d}^{2} \cdot  {\mu }_{\ell } \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right)$ ,if it is an internal node in $\mathcal{T}$ . Again,we account for the terms differently. The term $O\left( {d \cdot  {\mu }_{\ell }}\right)$ is charged as the factual cost of the node itself. On every tuple in ${\rho }_{i}$ for $i \in  \left\lbrack  {1,d}\right\rbrack$ ,we charge a nominal cost of $O\left( d\right)$ ,whereas on every tuple

- $O\left( {d \cdot  {\mu }_{\ell } + \operatorname{sort}\left( {{d}^{2} \cdot  {\mu }_{\ell } \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right) }\right)$，如果它是 $\mathcal{T}$ 中的一个内部节点。同样，我们以不同的方式处理这些项。项 $O\left( {d \cdot  {\mu }_{\ell }}\right)$ 作为节点本身的实际成本进行计费。对于 $i \in  \left\lbrack  {1,d}\right\rbrack$ 的 ${\rho }_{i}$ 中的每个元组，我们收取 $O\left( d\right)$ 的名义成本，而对于每个元组

in ${\rho }_{{h}_{\ell  + 1}}$ ,we charge an additional nominal cost of $O\left( {{d}^{2} \cdot  {\mu }_{\ell }}\right)$ . The term sort $\left( {{d}^{2} \cdot  {\mu }_{\ell } \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + }\right.$ $\left. {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right)$ equals $O\left( {\operatorname{sort}\left( {x}^{\prime }\right) }\right)$ ,where ${x}^{\prime }$ is the sum of the nominal costs of all the tuples in ${\rho }_{1},\ldots ,{\rho }_{d}$ incurred this way.

在 ${\rho }_{{h}_{\ell  + 1}}$ 中，我们额外收取 $O\left( {{d}^{2} \cdot  {\mu }_{\ell }}\right)$ 的名义成本。项 sort $\left( {{d}^{2} \cdot  {\mu }_{\ell } \cdot  \left| {\rho }_{{h}_{\ell  + 1}}\right|  + }\right.$ $\left. {d\mathop{\sum }\limits_{{i = 1}}^{d}\left| {\rho }_{i}\right| }\right)$ 等于 $O\left( {\operatorname{sort}\left( {x}^{\prime }\right) }\right)$，其中 ${x}^{\prime }$ 是以这种方式产生的 ${\rho }_{1},\ldots ,{\rho }_{d}$ 中所有元组的名义成本之和。

The above strategy allows us to bound $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$ by adding up two costs:

上述策略使我们能够通过将两种成本相加来界定 $\operatorname{cost}\left( {1,{r}_{1},\ldots ,{r}_{d}}\right)$：

- The sum of the factual costs of all nodes in $\mathcal{T}$ ;

- $\mathcal{T}$ 中所有节点的实际成本之和；

- sort(X),where $X$ is the sum of the total nominal costs charged on all the tuples in ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ across all levels of $\mathcal{T}$ . ${}^{1}$

- sort(X)，其中 $X$ 是在 $\mathcal{T}$ 的所有层级上对 ${r}_{1},{r}_{2},\ldots ,{r}_{d}$ 中所有元组收取的总名义成本之和。 ${}^{1}$

Next, we will concentrate on each bullet in turn.

接下来，我们将依次关注每个要点。

Bounding the Factual Costs. Let us now focus on Bullet 1. The key is to bound the number ${m}_{\ell }$ of nodes at each level $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ of $\mathcal{T}$ . We say that a level- $\ell$ call JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ underflows if $\left| {\rho }_{1}\right|  < {\tau }_{{h}_{\ell }}/2$ ; otherwise,it is ordinary. Consider all the calls $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ at level $\ell$ . The sets ${\rho }_{1}$ (i.e.,the first parameter) of those calls are disjoint. Hence,there can be at most $O\left( {{n}_{1}/{\tau }_{{h}_{\ell }}}\right)$ ordinary calls at level $\ell$ . Moreover,if $\ell  < w$ ,then a level- $\ell$ call creates at most one underflowing call at level $\ell  + 1$ . This discussion indicates that,for each $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ :

界定实际成本。现在让我们关注要点1。关键在于界定$\mathcal{T}$的每一层$\ell  \in  \left\lbrack  {1,w}\right\rbrack$上的节点数量${m}_{\ell }$。我们称，如果$\left| {\rho }_{1}\right|  < {\tau }_{{h}_{\ell }}/2$，则一个第$\ell$层调用JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$为下溢调用；否则，它为普通调用。考虑第$\ell$层的所有调用$\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$。这些调用的集合${\rho }_{1}$（即第一个参数）是不相交的。因此，第$\ell$层最多可以有$O\left( {{n}_{1}/{\tau }_{{h}_{\ell }}}\right)$个普通调用。此外，如果$\ell  < w$，那么一个第$\ell$层调用在第$\ell  + 1$层最多创建一个下溢调用。这一讨论表明，对于每个$\ell  \in  \left\lbrack  {2,w}\right\rbrack$：

$$
{m}_{\ell } = O\left( {{m}_{\ell  - 1} + \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}}\right)  = O\left( {\mathop{\sum }\limits_{{i = 1}}^{\ell }\frac{{n}_{1}}{{\tau }_{{h}_{i}}}}\right)  = O\left( \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}\right) , \tag{13}
$$

where the second equality used ${m}_{1} = 1 = {n}_{1}/{\tau }_{{h}_{1}}$ ,and the last equality used the fact that ${\tau }_{{h}_{i}} >$ $2{\tau }_{{h}_{i + 1}}$ for every $i \in  \left\lbrack  {1,w - 1}\right\rbrack$ .

其中第二个等式使用了${m}_{1} = 1 = {n}_{1}/{\tau }_{{h}_{1}}$，最后一个等式使用了对于每个$i \in  \left\lbrack  {1,w - 1}\right\rbrack$都有${\tau }_{{h}_{i}} >$ $2{\tau }_{{h}_{i + 1}}$这一事实。

Therefore,the sum of the factual costs of all the nodes in $\mathcal{T}$ is bounded by

因此，$\mathcal{T}$中所有节点的实际成本之和是有界的，其界限为

$$
O\left( {d \cdot  {m}_{w} + \mathop{\sum }\limits_{{\ell  = 1}}^{{w - 1}}d \cdot  {\mu }_{\ell } \cdot  {m}_{\ell }}\right)  = O\left( {\frac{d \cdot  {n}_{1}}{{\tau }_{{h}_{w}}} + \mathop{\sum }\limits_{{\ell  = 1}}^{{w - 1}}d \cdot  {\mu }_{\ell } \cdot  \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}}\right) 
$$

$$
\text{(by (8))} = O\left( {\frac{d \cdot  {n}_{1}}{{\tau }_{{h}_{w}}} + \mathop{\sum }\limits_{{\ell  = 1}}^{{w - 1}}d \cdot  \frac{{\tau }_{{\tau }_{{h}_{\ell }}}}{{\tau }_{{h}_{\ell  + 1}}} \cdot  \frac{{n}_{1}}{{\tau }_{{h}_{\ell }}}}\right) 
$$

$$
 = O\left( {\frac{d \cdot  {n}_{1}}{{\tau }_{{h}_{w}}} + \frac{d \cdot  {n}_{1}}{{\tau }_{{h}_{w}}}}\right) 
$$

$$
\left( {\text{ by }{\tau }_{{h}_{w}} = M/d}\right)  = O\left( \frac{{d}^{2}{n}_{1}}{M}\right) . \tag{14}
$$

Bounding $\mathbf{X}$ . Let us consider a single tuple $t$ in an arbitrary input relation ${r}_{i}$ (for any $i \in  \left\lbrack  {1,d}\right\rbrack$ ). We aim to bound the sum of all the nominal costs charged on $t$ ,across all the nodes of $\mathcal{T}$ . Consider a level- $\ell$ call $\left( {1 \leq  \ell  \leq  w}\right)$ JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ in $\mathcal{T}$ . We say that $t$ participates in the call if $t \in  {\rho }_{i}$ . Obviously, $t$ has a nominal cost on this node if and only if $t$ participates in it. For $i \in  \left\lbrack  {2,d}\right\rbrack$ ,define a value ${L}_{i} \in  \left\lbrack  {0,w}\right\rbrack$ as follows: - ${L}_{i} = \ell$ if $i$ is the axis of the calls at level- $\ell$ of $\mathcal{T}$ for some $\ell  \in  \left\lbrack  {2,w}\right\rbrack$ ,i.e., ${h}_{\ell } = i$ ; - ${L}_{i} = 0$ ,otherwise. We make the following observation on the participation of $t$ in different levels:

界定$\mathbf{X}$。让我们考虑任意输入关系${r}_{i}$（对于任意$i \in  \left\lbrack  {1,d}\right\rbrack$）中的单个元组$t$。我们的目标是界定在$\mathcal{T}$的所有节点上对$t$收取的所有名义成本之和。考虑$\mathcal{T}$中的一个第$\ell$层调用$\left( {1 \leq  \ell  \leq  w}\right)$ JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$。我们称，如果$t \in  {\rho }_{i}$，则$t$参与该调用。显然，当且仅当$t$参与该节点时，$t$在该节点上有名义成本。对于$i \in  \left\lbrack  {2,d}\right\rbrack$，定义一个值${L}_{i} \in  \left\lbrack  {0,w}\right\rbrack$如下： - 如果$i$是$\mathcal{T}$的第$\ell$层调用的轴（对于某个$\ell  \in  \left\lbrack  {2,w}\right\rbrack$），即${h}_{\ell } = i$，则${L}_{i} = \ell$； - 否则，${L}_{i} = 0$。我们对$t$在不同层的参与情况有如下观察：

---

<!-- Footnote -->

${}^{1}$ Notice that $\operatorname{sort}\left( x\right)  + \operatorname{sort}\left( {x}^{\prime }\right)  \leq  \operatorname{sort}\left( {x + {x}^{\prime }}\right)$ for any $x,{x}^{\prime } > 0$ .

${}^{1}$注意，对于任意$x,{x}^{\prime } > 0$，都有$\operatorname{sort}\left( x\right)  + \operatorname{sort}\left( {x}^{\prime }\right)  \leq  \operatorname{sort}\left( {x + {x}^{\prime }}\right)$。

<!-- Footnote -->

---

Lemma 6. If ${L}_{i} = 0$ ,then $t$ participates at most once at level $\ell$ for all $\ell  \in  \left\lbrack  {1,w}\right\rbrack$ . Otherwise, $t$ participates

引理6。如果${L}_{i} = 0$，那么对于所有$\ell  \in  \left\lbrack  {1,w}\right\rbrack$，$t$在第$\ell$层最多参与一次。否则，$t$参与

- at most once at level- $\ell$ for each $\ell  \in  \left\lbrack  {1,{L}_{i} - 1}\right\rbrack$ ;

- 对于每个 $\ell  \in  \left\lbrack  {1,{L}_{i} - 1}\right\rbrack$，在 $\ell$ 级别最多参与一次；

- $O\left( {\mu }_{{L}_{i} - 1}\right)$ times at level $\ell$ for each $\ell  \in  \left\lbrack  {{L}_{i},w}\right\rbrack$ .

- 对于每个 $\ell  \in  \left\lbrack  {{L}_{i},w}\right\rbrack$，在 $\ell$ 级别参与 $O\left( {\mu }_{{L}_{i} - 1}\right)$ 次。

Proof. The lemma follows from how $t$ is passed from a call to its descendants in $\mathcal{T}$ . Let JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ be a level- $\ell$ call that $t$ participates in. If ${h}_{\ell  + 1} \neq  i$ ,then $t$ participates in ${at}$ most one of the call’s child nodes in $\mathcal{T}$ . Otherwise $\left( {{h}_{\ell  + 1} = i}\right.$ ,and hence, ${L}_{i} = \ell  + 1$ by definition), $t$ may participate in all of the call’s child nodes in $\mathcal{T}$ . From (12),we know that $\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ has $q = O\left( {\mu }_{\ell }\right)  = O\left( {\mu }_{{L}_{i} - 1}\right)$ .

证明。该引理可由 $t$ 在 $\mathcal{T}$ 中从一次调用传递给其后代的方式得出。设 JOIN $\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 是 $t$ 参与的一个 $\ell$ 级调用。如果 ${h}_{\ell  + 1} \neq  i$，那么 $t$ 在 $\mathcal{T}$ 中最多参与该调用的一个子节点。否则 $\left( {{h}_{\ell  + 1} = i}\right.$（因此，根据定义 ${L}_{i} = \ell  + 1$），$t$ 可能参与 $\mathcal{T}$ 中该调用的所有子节点。由 (12) 可知，$\operatorname{JOIN}\left( {{h}_{\ell },{\rho }_{1},\ldots ,{\rho }_{d}}\right)$ 具有 $q = O\left( {\mu }_{\ell }\right)  = O\left( {\mu }_{{L}_{i} - 1}\right)$。

Hence,the total nominal cost of $t$ in the entire $\mathcal{T}$ equals:

因此，$t$ 在整个 $\mathcal{T}$ 中的总名义成本等于：

- $O\left( {d \cdot  w}\right)  = O\left( {d}^{2}\right)$ ,if ${L}_{i} = 0$ .

- $O\left( {d \cdot  w}\right)  = O\left( {d}^{2}\right)$，如果 ${L}_{i} = 0$。

- $O\left( {d \cdot  {L}_{i} + {d}^{2} \cdot  {\mu }_{{L}_{i} - 1} + d \cdot  {\mu }_{{L}_{i} - 1} \cdot  \left( {d - {L}_{i} + 1}\right) }\right)  = O\left( {{d}^{2} \cdot  {\mu }_{{L}_{i} - 1}}\right)$ ,otherwise. Specifically,the term $O\left( {d \cdot  {L}_{i}}\right)$ is due to the at most one participation of $t$ at each level from 1 to ${L}_{i} - 1$ ,whereas the term $O\left( {d \cdot  {\mu }_{{L}_{i} - 1} \cdot  \left( {d - {L}_{i} + 1}\right) }\right)$ is due to the $O\left( {\mu }_{{L}_{i} - 1}\right)$ participations at each level from ${L}_{i}$ to $d$ . The term $O\left( {{d}^{2} \cdot  {\mu }_{{L}_{i} - 1}}\right)$ is due to the additional cost charged on $t$ at level ${L}_{i} - 1$ .

- $O\left( {d \cdot  {L}_{i} + {d}^{2} \cdot  {\mu }_{{L}_{i} - 1} + d \cdot  {\mu }_{{L}_{i} - 1} \cdot  \left( {d - {L}_{i} + 1}\right) }\right)  = O\left( {{d}^{2} \cdot  {\mu }_{{L}_{i} - 1}}\right)$，否则。具体而言，项 $O\left( {d \cdot  {L}_{i}}\right)$ 是由于 $t$ 从 1 到 ${L}_{i} - 1$ 的每个级别最多参与一次，而项 $O\left( {d \cdot  {\mu }_{{L}_{i} - 1} \cdot  \left( {d - {L}_{i} + 1}\right) }\right)$ 是由于 $t$ 从 ${L}_{i}$ 到 $d$ 的每个级别参与 $O\left( {\mu }_{{L}_{i} - 1}\right)$ 次。项 $O\left( {{d}^{2} \cdot  {\mu }_{{L}_{i} - 1}}\right)$ 是由于在 ${L}_{i} - 1$ 级别对 $t$ 收取的额外成本。

Therefore, $X$ -the sum of the total nominal cost of all tuples-is bounded by

因此，$X$（所有元组的总名义成本之和）受限于

$$
O\left( {\mathop{\sum }\limits_{{i \in  \left\lbrack  {1,d}\right\rbrack  \text{ s.t. }{L}_{i} \neq  0}}{d}^{2}{\mu }_{{L}_{i}}{n}_{i} + \mathop{\sum }\limits_{{i = 1}}^{d}{d}^{2}{n}_{i}}\right)  = O\left( {\mathop{\sum }\limits_{{\ell  = 2}}^{w}{d}^{2}{\mu }_{\ell  - 1}{n}_{{h}_{\ell }} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) 
$$

$$
\text{(by Lemma 5)} = O\left( {\mathop{\sum }\limits_{{\ell  = 2}}^{w}U{d}^{2 + \frac{1}{d - 1}} + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) 
$$

$$
 = O\left( {{d}^{3 + \frac{1}{d - 1}}U + {d}^{2}\mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) .
$$

Summary. Combining the above equation with (2) and plugging in the definition of $U$ in (14),we now complete the whole proof of Theorem 2.

总结。将上述方程与 (2) 相结合，并代入 (14) 中 $U$ 的定义，我们现在完成了定理 2 的整个证明。

### 3.5 A Lower Bound Remark for Constant $d$

### 3.5 关于常数 $d$ 的下界注记

When the number $d$ of attributes is a constant,our LW enumeration algorithm guarantees I/O cost $O\left( {\operatorname{sort}\left( {{\left( \mathop{\prod }\limits_{{i = 1}}^{d}{n}_{i}/M\right) }^{\frac{1}{d - 1}} + \mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) }\right)$ . On the other hand,using an argument similar to those in $\left\lbrack  {8,{14}}\right\rbrack$ ,we can establish an $\mathrm{I}/\mathrm{O}$ lower bound of $\Omega \left( {{n}^{\frac{d}{d - 1}}/\left( {B{M}^{\frac{1}{d - 1}}}\right) }\right)$ in the scenario where ${n}_{1} = {n}_{2} =$ $\ldots  = {n}_{d} = n$ ,on the class of witnessing algorithms. Our algorithm,which is in this class,is thus asymptotically optimal up to a logarithmic factor. Next, we present a proof of the lower bound.

当属性数量 $d$ 为常数时，我们的LW枚举算法保证了I/O成本为 $O\left( {\operatorname{sort}\left( {{\left( \mathop{\prod }\limits_{{i = 1}}^{d}{n}_{i}/M\right) }^{\frac{1}{d - 1}} + \mathop{\sum }\limits_{{i = 1}}^{d}{n}_{i}}\right) }\right)$ 。另一方面，使用与 $\left\lbrack  {8,{14}}\right\rbrack$ 中类似的论证，我们可以在 ${n}_{1} = {n}_{2} =$ $\ldots  = {n}_{d} = n$ 的场景下，针对见证算法类建立一个 $\mathrm{I}/\mathrm{O}$ 下界 $\Omega \left( {{n}^{\frac{d}{d - 1}}/\left( {B{M}^{\frac{1}{d - 1}}}\right) }\right)$ 。我们的算法属于此类，因此在对数因子范围内是渐近最优的。接下来，我们给出该下界的证明。

We will refer to a tuple in ${r}_{1},{r}_{2},\ldots ,$ or ${r}_{d}$ as an input tuple. A witnessing algorithm is modeled as follows. A disk block has the capacity to store precisely $B$ input tuples. At the beginning,the memory is empty, while all the input tuples reside in the disk. At any moment, the algorithm is allowed to keep at most $M$ input tuples in the memory. At each step,it is permitted three operations:

我们将 ${r}_{1},{r}_{2},\ldots ,$ 或 ${r}_{d}$ 中的元组称为输入元组。见证算法的建模如下。一个磁盘块恰好能够存储 $B$ 个输入元组。一开始，内存为空，而所有输入元组都存于磁盘中。在任何时刻，算法最多允许在内存中保留 $M$ 个输入元组。在每一步，允许进行三种操作：

- Read $I/O$ : fetch $B$ input tuples from a disk block into the memory;

- 读取 $I/O$ ：从一个磁盘块中将 $B$ 个输入元组提取到内存中；

- Write $I/O$ : write $B$ input tuples in the memory to a disk block;

- 写入 $I/O$ ：将内存中的 $B$ 个输入元组写入一个磁盘块；

- Perform emit $\left( {t}^{ * }\right)$ for a result tuple ${t}^{ * } = {r}_{1} \bowtie  \ldots  \bowtie  {r}_{d}$ ,provided that the $d$ input tuples that produce ${t}^{ * }$ are all in the memory currently.

- 对于结果元组 ${t}^{ * } = {r}_{1} \bowtie  \ldots  \bowtie  {r}_{d}$ 执行输出 $\left( {t}^{ * }\right)$ ，前提是产生 ${t}^{ * }$ 的 $d$ 个输入元组当前都在内存中。

Let us consider a hard input to the LW enumeration problem where the size of ${r}_{1} \bowtie  \ldots  \bowtie  {r}_{d}$ equals ${n}^{d/\left( {d - 1}\right) }$ (such an input indeed exists [4]). Suppose that an algorithm solves the problem on this input using $H$ read I/Os (write I/Os are for free in our analysis). Chop the sequence of these read I/Os into epochs where each epoch is a subsequence of $M/B$ I/Os,except possibly the last one. As shown later,during each epoch,emit(.) can only be called $O\left( {M}^{d/\left( {d - 1}\right) }\right)$ times. This implies that

让我们考虑LW枚举问题的一个困难输入，其中${r}_{1} \bowtie  \ldots  \bowtie  {r}_{d}$的大小等于${n}^{d/\left( {d - 1}\right) }$（这样的输入确实存在[4]）。假设一个算法在这个输入上使用$H$次读I/O操作来解决该问题（在我们的分析中，写I/O操作是免费的）。将这些读I/O操作序列划分为若干个阶段，每个阶段是一个包含$M/B$次I/O操作的子序列，可能最后一个阶段除外。如后文所示，在每个阶段中，emit(.)函数最多只能被调用$O\left( {M}^{d/\left( {d - 1}\right) }\right)$次。这意味着

$$
\frac{H}{M/B} \cdot  O\left( {M}^{d/\left( {d - 1}\right) }\right)  \geq  {n}^{d/\left( {d - 1}\right) }
$$

which suggests $H = \Omega \left( {{n}^{\frac{d}{d - 1}}/\left( {B{M}^{\frac{1}{d - 1}}}\right) }\right)$ .

这表明$H = \Omega \left( {{n}^{\frac{d}{d - 1}}/\left( {B{M}^{\frac{1}{d - 1}}}\right) }\right)$ 。

It remains to explain why there can be only $O\left( {M}^{d/\left( {d - 1}\right) }\right)$ tuple emissions during an epoch. Define $S$ to be a set of input tuples defined as follows. $S$ includes (i) all the input tuples already in the memory at the beginning of the epoch,and (ii) all the input tuples in the (at most) $M/B$ blocks read in the epoch. Clearly $\left| S\right|  \leq  {2M}$ . Every tuple emitted during the epoch must be produced by the tuples in $S$ . Suppose that $S$ contains ${x}_{i}\left( {i \in  \left\lbrack  {1,d}\right\rbrack  }\right)$ tuples from ${r}_{i}$ . According to [4],these input tuples can produced at most ${\left( \mathop{\prod }\limits_{{i = 1}}^{d}{x}_{i}\right) }^{1/\left( {d - 1}\right) }$ tuples in ${r}_{1} \bowtie  \ldots  \bowtie  {r}_{d}$ ,which is at most ${\left( 2M/d\right) }^{d/\left( {d - 1}\right) } = O\left( {M}^{d/\left( {d - 1}\right) }\right)$ under the constraint $\mathop{\sum }\limits_{{i = 1}}^{d}{x}_{i} \leq  {2M}$ .

接下来需要解释为什么在一个阶段中最多只能有$O\left( {M}^{d/\left( {d - 1}\right) }\right)$次元组发射。定义$S$为一个输入元组集合，定义如下。$S$包含（i）在阶段开始时已经存在于内存中的所有输入元组，以及（ii）在该阶段中读取的（最多）$M/B$个数据块中的所有输入元组。显然$\left| S\right|  \leq  {2M}$ 。在该阶段中发射的每个元组都必须由$S$中的元组产生。假设$S$包含来自${r}_{i}$的${x}_{i}\left( {i \in  \left\lbrack  {1,d}\right\rbrack  }\right)$个元组。根据[4]，这些输入元组在${r}_{1} \bowtie  \ldots  \bowtie  {r}_{d}$中最多能产生${\left( \mathop{\prod }\limits_{{i = 1}}^{d}{x}_{i}\right) }^{1/\left( {d - 1}\right) }$个元组，在约束条件$\mathop{\sum }\limits_{{i = 1}}^{d}{x}_{i} \leq  {2M}$下，这个数量最多为${\left( 2M/d\right) }^{d/\left( {d - 1}\right) } = O\left( {M}^{d/\left( {d - 1}\right) }\right)$。

## 4 A Faster Algorithm for Arity 3

## 4 三元关系的更快算法

The algorithm developed in the previous section solves the LW enumeration problem for any $d \leq$ $M/2$ . In this section,we focus on $d = 3$ ,and leverage intrinsic properties of this special instance to design a faster algorithm, which will establish Theorem 3 (and hence, also Corollaries 1 and 2). Specifically,the input consists of three relations: ${r}_{1}\left( {{A}_{2},{A}_{3}}\right) ,{r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ ,and ${r}_{3}\left( {{A}_{1},{A}_{2}}\right)$ ; and the goal is to emit all the tuples in the result of ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ .

上一节开发的算法可以解决任意$d \leq$ $M/2$情况下的LW枚举问题。在本节中，我们将重点关注$d = 3$，并利用这个特殊实例的内在性质来设计一个更快的算法，该算法将证明定理3（从而也证明推论1和推论2）。具体来说，输入包含三个关系：${r}_{1}\left( {{A}_{2},{A}_{3}}\right) ,{r}_{2}\left( {{A}_{1},{A}_{3}}\right)$和${r}_{3}\left( {{A}_{1},{A}_{2}}\right)$；目标是输出${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$结果中的所有元组。

As before,for each $i \in  \left\lbrack  {1,3}\right\rbrack$ ,set ${n}_{i} = \left| {r}_{i}\right|$ ,and denote by $\operatorname{dom}\left( {A}_{i}\right)$ the domain of ${A}_{i}$ . Without loss of generality,we assume that ${n}_{1} \geq  {n}_{2} \geq  {n}_{3}$ .

和之前一样，对于每个$i \in  \left\lbrack  {1,3}\right\rbrack$，设${n}_{i} = \left| {r}_{i}\right|$，并用$\operatorname{dom}\left( {A}_{i}\right)$表示${A}_{i}$的定义域。不失一般性，我们假设${n}_{1} \geq  {n}_{2} \geq  {n}_{3}$ 。

### 4.1 Basic Algorithms

### 4.1 基本算法

Let us start with:

让我们从以下内容开始：

Lemma 7. If ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ and ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ have been sorted by ${A}_{3}$ ,the 3-arity ${LW}$ enumeration problem can be solved in $O\left( {1 + \frac{\left( {{n}_{1} + {n}_{2}}\right) {n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$ .

引理7. 如果${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$和${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$已按${A}_{3}$排序，则三元${LW}$枚举问题可在$O\left( {1 + \frac{\left( {{n}_{1} + {n}_{2}}\right) {n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) I/{Os}$时间内解决。

Proof. If ${n}_{3} \leq  M$ ,we can achieve the purpose stated in the lemma using the small-join algorithm of Lemma 2 with straightforward modifications (e.g., apparently sorting is not required). When ${n}_{3} > M$ ,we simply chop ${r}_{3}$ into subsets of size $M$ ,and then repeat the above small-join algorithm $\left\lceil  {{n}_{3}/M}\right\rceil$ times.

证明. 如果${n}_{3} \leq  M$，我们可以通过对引理2中的小连接算法进行直接修改（例如，显然不需要排序）来实现引理中所述的目的。当${n}_{3} > M$时，我们只需将${r}_{3}$分割成大小为$M$的子集，然后重复上述小连接算法$\left\lceil  {{n}_{3}/M}\right\rceil$次。

We call ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ an ${A}_{1}$ -point join if both conditions below are fulfilled:

如果满足以下两个条件，我们称${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$为${A}_{1}$点连接：

- all the ${A}_{1}$ values in ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ are the same;

- ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$中所有的${A}_{1}$值都相同；

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ and ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ are sorted by ${A}_{3}$ .

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$和${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$按${A}_{3}$排序。

Lemma 8. Given an ${A}_{1}$ -point join,we can emit all its result tuples in $O\left( {1 + \frac{{n}_{1}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right)$ $I/{Os}$ .

引理8. 给定一个${A}_{1}$点连接，我们可以在$O\left( {1 + \frac{{n}_{1}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right)$ $I/{Os}$时间内输出其所有结果元组。

Proof. We first obtain ${r}^{\prime }\left( {{A}_{1},{A}_{2},{A}_{3}}\right)  = {r}_{1} \bowtie  {r}_{2}$ ,and store all the tuples of ${r}^{\prime }$ into the disk. Since all the tuples in ${r}_{2}$ have the same ${A}_{1}$ -value,their ${A}_{3}$ -values must be distinct. Hence,each tuple in ${r}_{1}$ can be joined with at most one tuple in ${r}_{2}$ ,implying that $\left| {r}^{\prime }\right|  \leq  {n}_{1}$ . Utilizing the fact that ${r}_{1}$ and ${r}_{2}$ are both sorted on ${A}_{3},{r}^{\prime }$ can be produced by a synchronous scan over ${r}_{1}$ and ${r}_{2}$ in $O\left( {1 + \left( {{n}_{1} + {n}_{2}}\right) /B}\right) \mathrm{I}/\mathrm{{Os}}$ .

证明. 我们首先得到${r}^{\prime }\left( {{A}_{1},{A}_{2},{A}_{3}}\right)  = {r}_{1} \bowtie  {r}_{2}$，并将${r}^{\prime }$的所有元组存储到磁盘中。由于${r}_{2}$中的所有元组具有相同的${A}_{1}$值，它们的${A}_{3}$值必须是不同的。因此，${r}_{1}$中的每个元组最多可以与${r}_{2}$中的一个元组连接，这意味着$\left| {r}^{\prime }\right|  \leq  {n}_{1}$。利用${r}_{1}$和${r}_{2}$都按${A}_{3},{r}^{\prime }$排序这一事实，可以通过对${r}_{1}$和${r}_{2}$进行同步扫描在$O\left( {1 + \left( {{n}_{1} + {n}_{2}}\right) /B}\right) \mathrm{I}/\mathrm{{Os}}$时间内生成。

Then,we use the classic blocked nested loop (BNL) algorithm to perform the join ${r}^{\prime } \bowtie  {r}_{3}$ (which equals ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ ). The only difference is that,whenever BNL wants to write a block of $O\left( B\right)$ result tuples to the disk, we skip the write but simply emit those tuples. The BNL performs $O\left( {1 + \frac{\left| {r}^{\prime }\right| {n}_{3}}{MB} + \frac{{r}^{\prime } + {n}_{3}}{B}}\right)$ I/Os. The lemma thus follows.

然后，我们使用经典的块嵌套循环（BNL）算法来执行连接${r}^{\prime } \bowtie  {r}_{3}$（它等于${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$）。唯一的区别是，每当BNL想要将一块$O\left( B\right)$结果元组写入磁盘时，我们跳过写入操作，只是简单地输出这些元组。BNL执行$O\left( {1 + \frac{\left| {r}^{\prime }\right| {n}_{3}}{MB} + \frac{{r}^{\prime } + {n}_{3}}{B}}\right)$次I/O操作。因此，引理得证。

Symmetrically,we call ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ an ${A}_{2}$ -point join if

对称地，如果满足以下条件，我们称${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$为${A}_{2}$点连接

- all the ${A}_{2}$ values in ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ are the same.

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ 中的所有 ${A}_{2}$ 值都相同。

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ and ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ are sorted by ${A}_{3}$ .

- ${r}_{1}\left( {{A}_{2},{A}_{3}}\right)$ 和 ${r}_{2}\left( {{A}_{1},{A}_{3}}\right)$ 按 ${A}_{3}$ 排序。

Lemma 9. Given an ${A}_{2}$ -point join,we can emit all its result tuples in $O\left( {1 + \frac{{n}_{2}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right)$ $I/{Os}$ .

引理 9. 给定一个 ${A}_{2}$ 点连接，我们可以在 $O\left( {1 + \frac{{n}_{2}{n}_{3}}{MB} + \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right)$ $I/{Os}$ 中输出其所有结果元组。

Proof. Symmetric to Lemma 8.

证明：与引理 8 对称。

### 4.2 3-Arity LW Enumeration Algorithm

### 4.2 三元 LW 枚举算法

Next,we give our general algorithm for LW enumeration with $d = 3$ . We will focus on ${n}_{1} \geq  {n}_{2} \geq$ ${n}_{3} \geq  M$ ; if ${n}_{3} < M$ ,the algorithm in Lemma 7 already solves the problem in linear I/Os after sorting.

接下来，我们给出使用 $d = 3$ 进行 LW 枚举的通用算法。我们将关注 ${n}_{1} \geq  {n}_{2} \geq$ ${n}_{3} \geq  M$；如果 ${n}_{3} < M$，引理 7 中的算法在排序后已经可以在线性 I/O 内解决该问题。

Set:

设置：

$$
{\theta }_{1} = \sqrt{\frac{{n}_{1}{n}_{3}M}{{n}_{2}}}\text{,and}{\theta }_{2} = \sqrt{\frac{{n}_{2}{n}_{3}M}{{n}_{1}}}\text{.} \tag{15}
$$

For values ${a}_{1} \in  \operatorname{dom}\left( {A}_{1}\right)$ and ${a}_{2} \in  \operatorname{dom}\left( {A}_{2}\right)$ ,define:

对于值 ${a}_{1} \in  \operatorname{dom}\left( {A}_{1}\right)$ 和 ${a}_{2} \in  \operatorname{dom}\left( {A}_{2}\right)$，定义：

$$
\operatorname{freq}\left( {{a}_{1},{r}_{3}}\right)  = \left| \left\{  {t \in  {r}_{3} \mid  t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}}\right\}  \right| 
$$

$$
\operatorname{freq}\left( {{a}_{2},{r}_{3}}\right)  = \left| \left\{  {t \in  {r}_{3} \mid  t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}}\right\}  \right| .
$$

Now we introduce:

现在我们引入：

$$
{\Phi }_{1} = \left\{  {{a}_{1} \in  \operatorname{dom}\left( {A}_{1}\right)  \mid  \operatorname{freq}\left( {{a}_{1},{r}_{3}}\right)  > {\theta }_{1}}\right\}  
$$

$$
{\Phi }_{2} = \left\{  {{a}_{2} \in  \operatorname{dom}\left( {A}_{2}\right)  \mid  \operatorname{freq}\left( {{a}_{2},{r}_{3}}\right)  > {\theta }_{2}}\right\}  .
$$

Let ${t}^{ * }$ be a result tuple of ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ . We can classify ${t}^{ * }$ into one of the following categories:

设 ${t}^{ * }$ 是 ${r}_{1} \bowtie  {r}_{2} \bowtie  {r}_{3}$ 的一个结果元组。我们可以将 ${t}^{ * }$ 分类为以下类别之一：

- Heavy-heavy: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

- 重 - 重：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$ 和 ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

- Heavy-light: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$

- 重 - 轻：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}$ 和 ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$

- Light-heavy: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

- 轻 - 重：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$ 和 ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}$

- Light-light: ${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$ and ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$ .

- 轻 - 轻：${t}^{ * }\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}$ 和 ${t}^{ * }\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}$。

We will emit each type of tuples separately, after a partitioning phase, as explained in the sequel.

如后续所述，在分区阶段之后，我们将分别输出每种类型的元组。

Partitioning ${r}_{3}$ . Define:

对 ${r}_{3}$ 进行分区。定义：

$$
{r}_{3}^{\text{heavy,heavy }} = \left\{  {t \in  {r}_{3} \mid  t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}}\right\}  
$$

$$
{r}_{3}^{\text{heavy,light }} = \left\{  {t \in  {r}_{3} \mid  t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}}\right\}  
$$

$$
{r}_{3}^{\text{light,heavy }} = \left\{  {t \in  {r}_{3} \mid  t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}}\right\}  
$$

$$
{r}_{3}^{\text{light,light }} = \left\{  {t \in  {r}_{3} \mid  t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1},t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}}\right\}  
$$

$$
{r}_{3}^{\text{light,} - } = {r}_{3}^{\text{light,heavy }} \cup  {r}_{3}^{\text{light,light }}
$$

$$
{r}_{3}^{-\text{,light }} = {r}_{3}^{\text{heavy,light }} \cup  {r}_{3}^{\text{light,light }}\text{.}
$$

Divide $\operatorname{dom}\left( {A}_{1}\right)$ into ${q}_{1} = O\left( {1 + {n}_{3}/{\theta }_{1}}\right)$ disjoint intervals ${I}_{1}^{1},{I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$ with the following properties: (i) ${I}_{1}^{1},{I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$ are in ascending order,and (ii) for each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack  ,{r}_{3}^{\text{light,} - }$ has at most $2{\theta }_{1}$ tuples whose ${A}_{1}$ -values fall in ${I}_{j}^{1}$ . Similarly,we divide $\operatorname{dom}\left( {A}_{2}\right)$ into ${q}_{2} = O\left( {1 + {n}_{3}/{\theta }_{2}}\right)$ disjoint intervals ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ with the following properties: (i) ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ are in ascending order,and (ii) for each $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack  ,{r}_{3}^{-,\text{light }}$ has at most $2{\theta }_{2}$ tuples whose ${A}_{2}$ -values fall in ${I}_{j}^{2}$ .

将 $\operatorname{dom}\left( {A}_{1}\right)$ 划分为 ${q}_{1} = O\left( {1 + {n}_{3}/{\theta }_{1}}\right)$ 个不相交的区间 ${I}_{1}^{1},{I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$，这些区间具有以下性质：(i) ${I}_{1}^{1},{I}_{2}^{1},\ldots ,{I}_{{q}_{1}}^{1}$ 按升序排列；(ii) 对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack  ,{r}_{3}^{\text{light,} - }$，其 ${A}_{1}$ 值落在 ${I}_{j}^{1}$ 内的元组最多有 $2{\theta }_{1}$ 个。类似地，我们将 $\operatorname{dom}\left( {A}_{2}\right)$ 划分为 ${q}_{2} = O\left( {1 + {n}_{3}/{\theta }_{2}}\right)$ 个不相交的区间 ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$，这些区间具有以下性质：(i) ${I}_{1}^{2},{I}_{2}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ 按升序排列；(ii) 对于每个 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack  ,{r}_{3}^{-,\text{light }}$，其 ${A}_{2}$ 值落在 ${I}_{j}^{2}$ 内的元组最多有 $2{\theta }_{2}$ 个。

We now define several partitions of ${r}_{3}$ :

我们现在定义 ${r}_{3}$ 的几个划分：

- For each ${a}_{1} \in  {\Phi }_{1}$ and ${a}_{2} \in  {\Phi }_{2}$ ,let ${r}_{3}^{\text{heavy,heavy }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ be the (only) tuple $t$ in ${r}_{3}^{\text{heavy,heavy }}$ with $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ and $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ .

- 对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和 ${a}_{2} \in  {\Phi }_{2}$，设 ${r}_{3}^{\text{heavy,heavy }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ 为 ${r}_{3}^{\text{heavy,heavy }}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ 和 $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ 的（唯一）元组 $t$。

- For each ${a}_{1} \in  {\Phi }_{1}$ and $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ ,let ${r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack$ be the set of tuples $t$ in ${r}_{3}^{\text{heavy,light }}$ with $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ and $t\left\lbrack  {A}_{2}\right\rbrack$ in ${I}_{j}^{2}.$

- 对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$，设 ${r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack$ 为 ${r}_{3}^{\text{heavy,light }}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ 且 $t\left\lbrack  {A}_{2}\right\rbrack$ 属于 ${I}_{j}^{2}.$ 的元组 $t$ 的集合。

- For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and ${a}_{2} \in  {\Phi }_{2}$ ,let ${r}_{3}^{\text{light,heavy }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack$ be the set of tuples $t$ in ${r}_{3}^{\text{light,heavy }}$ with $t\left\lbrack  {A}_{1}\right\rbrack$ in ${I}_{j}^{1}$ and $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ .

- 对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和 ${a}_{2} \in  {\Phi }_{2}$，设 ${r}_{3}^{\text{light,heavy }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack$ 为 ${r}_{3}^{\text{light,heavy }}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack$ 属于 ${I}_{j}^{1}$ 且 $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ 的元组 $t$ 的集合。

- For each ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and ${j}_{2} \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ ,let ${r}_{3}^{{light},{light}}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack$ be the set of tuples $t$ in ${r}_{3}^{{light},{light}}$ with $t\left\lbrack  {A}_{1}\right\rbrack$ in ${I}_{j}^{1}$ and $t\left\lbrack  {A}_{2}\right\rbrack$ in ${I}_{j}^{2}$ .

- 对于每个 ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和 ${j}_{2} \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$，设 ${r}_{3}^{{light},{light}}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack$ 为 ${r}_{3}^{{light},{light}}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack$ 属于 ${I}_{j}^{1}$ 且 $t\left\lbrack  {A}_{2}\right\rbrack$ 属于 ${I}_{j}^{2}$ 的元组 $t$ 的集合。

It is rudimentary to produce all the above partitions with $O\left( {\operatorname{sort}\left( {n}_{3}\right) }\right) \mathrm{I}/\mathrm{{Os}}$ in total.

总共使用 $O\left( {\operatorname{sort}\left( {n}_{3}\right) }\right) \mathrm{I}/\mathrm{{Os}}$ 来生成上述所有分区是基本操作。

Partitioning ${r}_{1}$ and ${r}_{2}$ . Let:

对 ${r}_{1}$ 和 ${r}_{2}$ 进行分区。设：

$$
{r}_{1}^{\text{heavy }} = \text{set of tuples}t\text{in}{r}_{1}\text{s.t.}t\left\lbrack  {A}_{2}\right\rbrack   \in  {\Phi }_{2}
$$

$$
{r}_{1}^{\text{light }} = \text{ set of tuples }t\text{ in }{r}_{1}\text{ s.t. }t\left\lbrack  {A}_{2}\right\rbrack   \notin  {\Phi }_{2}
$$

$$
{r}_{2}^{\text{heavy }} = \text{ set of tuples }t\text{ in }{r}_{2}\text{ s.t. }t\left\lbrack  {A}_{1}\right\rbrack   \in  {\Phi }_{1}
$$

$$
{r}_{2}^{\text{light }} = \text{set of tuples}t\text{in}{r}_{2}\text{s.t.}t\left\lbrack  {A}_{1}\right\rbrack   \notin  {\Phi }_{1}
$$

We now define several partitions of ${r}_{1}$ :

我们现在定义 ${r}_{1}$ 的几个分区：

- For each ${a}_{2} \in  {\Phi }_{2}$ ,let ${r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack$ be the set of tuples $t$ in ${r}_{1}^{\text{heavy }}$ with $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ .

- 对于每个 ${a}_{2} \in  {\Phi }_{2}$，设 ${r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack$ 为 ${r}_{1}^{\text{heavy }}$ 中满足 $t\left\lbrack  {A}_{2}\right\rbrack   = {a}_{2}$ 的元组 $t$ 的集合。

- For each $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ ,let ${r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack$ be the set of tuples $t$ in ${r}_{1}^{\text{light }}$ with $t\left\lbrack  {A}_{2}\right\rbrack$ in ${I}_{j}^{2}$ .

- 对于每个 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$，设 ${r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack$ 为 ${r}_{1}^{\text{light }}$ 中满足 $t\left\lbrack  {A}_{2}\right\rbrack$ 属于 ${I}_{j}^{2}$ 的元组 $t$ 的集合。

Similarly,we define several partitions of ${r}_{2}$ :

类似地，我们定义 ${r}_{2}$ 的几个分区：

- For each ${a}_{1} \in  {\Phi }_{1}$ ,let ${r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack$ be the set of tuples $t$ in ${r}_{2}^{\text{heavy }}$ with $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ .

- 对于每个 ${a}_{1} \in  {\Phi }_{1}$，设 ${r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack$ 为 ${r}_{2}^{\text{heavy }}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack   = {a}_{1}$ 的元组 $t$ 的集合。

- For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ ,let ${r}_{2}^{\text{light }}\left\lbrack  {I}_{j}^{1}\right\rbrack$ be the set of tuples $t$ in ${r}_{2}^{\text{light }}$ with $t\left\lbrack  {A}_{1}\right\rbrack$ in ${I}_{j}^{1}$ .

- 对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$，设 ${r}_{2}^{\text{light }}\left\lbrack  {I}_{j}^{1}\right\rbrack$ 为 ${r}_{2}^{\text{light }}$ 中满足 $t\left\lbrack  {A}_{1}\right\rbrack$ 属于 ${I}_{j}^{1}$ 的元组 $t$ 的集合。

It is rudimentary to produce the above partitions using $O\left( {\operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$ I/Os in total. With the same cost,we make sure that all these partitions are sorted by ${A}_{3}$ .

总共使用 $O\left( {\operatorname{sort}\left( {{n}_{1} + {n}_{2} + {n}_{3}}\right) }\right)$ 次输入/输出（I/O）来生成上述分区是基本操作。以相同的成本，我们确保所有这些分区都按 ${A}_{3}$ 排序。

Enumerating Result Tuples. We emit each type of tuples as follows:

枚举结果元组。我们按如下方式输出每种类型的元组：

- Heavy-heavy: For each ${a}_{1} \in  {\Phi }_{1}$ and each ${a}_{2} \in  {\Phi }_{2}$ ,apply Lemma 7 to emit the result of ${r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack   \bowtie  {r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie  {r}_{3}^{\text{heavy,heavy }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack  .$

- 重 - 重：对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和每个 ${a}_{2} \in  {\Phi }_{2}$，应用引理 7 来输出 ${r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack   \bowtie  {r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie  {r}_{3}^{\text{heavy,heavy }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack  .$ 的结果

- Heavy-light: For each ${a}_{1} \in  {\Phi }_{1}$ and each $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ ,apply Lemma 8 to emit the result of the ${A}_{1}$ -point join ${r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie  {r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack$ .

- 重-轻：对于每个 ${a}_{1} \in  {\Phi }_{1}$ 和每个 $j \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$，应用引理 8 得出 ${A}_{1}$ 点连接 ${r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack   \bowtie  {r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack$ 的结果。

- Light-heavy: For each $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and each ${a}_{2} \in  {\Phi }_{2}$ ,apply Lemma 9 to emit the result of the ${A}_{2}$ -point join ${r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack   \bowtie  {r}_{2}^{\text{light }}\left\lbrack  {I}_{j}^{1}\right\rbrack   \bowtie  {r}_{3}^{\text{light,heavy }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack$ .

- 轻-重：对于每个 $j \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和每个 ${a}_{2} \in  {\Phi }_{2}$，应用引理 9 得出 ${A}_{2}$ 点连接 ${r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack   \bowtie  {r}_{2}^{\text{light }}\left\lbrack  {I}_{j}^{1}\right\rbrack   \bowtie  {r}_{3}^{\text{light,heavy }}\left\lbrack  {{I}_{j}^{1},{a}_{2}}\right\rbrack$ 的结果。

- Light-light: For each ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ and each ${j}_{2} \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$ ,apply Lemma 7 to emit the result of ${r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{light }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack   \bowtie  {r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  .$

- 轻-轻：对于每个 ${j}_{1} \in  \left\lbrack  {1,{q}_{1}}\right\rbrack$ 和每个 ${j}_{2} \in  \left\lbrack  {1,{q}_{2}}\right\rbrack$，应用引理 7 得出 ${r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack   \bowtie  {r}_{2}^{\text{light }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack   \bowtie  {r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  .$ 的结果

### 4.3 Analysis

### 4.3 分析

We now analyze the algorithm of Section 4.2,assuming ${n}_{1} \geq  {n}_{2} \geq  {n}_{3} \geq  M$ . First,it should be clear that

我们现在分析 4.2 节的算法，假设 ${n}_{1} \geq  {n}_{2} \geq  {n}_{3} \geq  M$。首先，显然

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

By Lemma 7,the cost of red-red emission is bounded by (remember that ${r}_{3}^{\text{heavy,heavy }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ has only 1 tuple):

根据引理 7，红 - 红发射的成本受限于（记住 ${r}_{3}^{\text{heavy,heavy }}\left\lbrack  {{a}_{1},{a}_{2}}\right\rbrack$ 只有 1 个元组）：

$$
\mathop{\sum }\limits_{{{a}_{1},{a}_{2}}}O\left( {1 + \frac{\left| {{r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack  }\right| }{B}}\right) .
$$

$$
 = O\left( {\left| {\Phi }_{1}\right| \left| {\Phi }_{2}\right|  + \mathop{\sum }\limits_{{a}_{2}}\frac{\left| {{r}_{1}^{\text{heavy }}\left\lbrack  {a}_{2}\right\rbrack  }\right| \left| {\Phi }_{1}\right| }{B} + \mathop{\sum }\limits_{{a}_{1}}\frac{\left| {{r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack  }\right| \left| {\Phi }_{2}\right| \left| \right| }{B}}\right) 
$$

$$
 = O\left( {\frac{{n}_{3}}{M} + \frac{{n}_{1}\left| {\Phi }_{1}\right| }{B} + \frac{{n}_{2}\left| {\Phi }_{2}\right| }{B}}\right)  = O\left( \frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}}\right) .
$$

By Lemma 8, the cost of red-blue emission is bounded by:

根据引理 8，红 - 蓝发射的成本受限于：

$$
\mathop{\sum }\limits_{{{a}_{1},j}}O\left( {1 + \frac{\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| \left| {{r}_{3}^{\text{ heavy },\text{ light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right| }{MB} + \frac{\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack  }\right|  + \left| {{r}_{3}^{\text{heavy }},\text{ light }\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right| }{B}}\right) .
$$

$$
 = O\left( {\left| {\Phi }_{1}\right| {q}_{2} + \mathop{\sum }\limits_{j}\frac{\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| \mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right| }{MB}}\right. 
$$

$$
\left. {+\frac{\left| {\Phi }_{1}\right| \mathop{\sum }\limits_{j}\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| }{B} + \frac{{q}_{2}\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{2}^{\text{heavy }}\left\lbrack  {a}_{1}\right\rbrack  }\right| }{B} + \frac{{n}_{3}}{B}}\right) \text{.} \tag{16}
$$

Observe that $\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right|$ is the total number of tuples in ${r}_{3}^{\text{heavy,light }}$ whose ${A}_{2}$ -values fall in ${I}_{j}^{2}$ . By the way ${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ are constructed,we know:

观察可知，$\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right|$ 是 ${r}_{3}^{\text{heavy,light }}$ 中 ${A}_{2}$ 值落在 ${I}_{j}^{2}$ 内的元组总数。根据 ${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ 的构造方式，我们知道：

$$
\mathop{\sum }\limits_{{a}_{1}}\left| {{r}_{3}^{\text{heavy,light }}\left\lbrack  {{a}_{1},{I}_{j}^{2}}\right\rbrack  }\right|  \leq  2{\theta }_{2}.
$$

(16) is thus bounded by:

因此，(16) 受限于：

$$
O\left( {\frac{{n}_{3}}{M} + \mathop{\sum }\limits_{j}\frac{\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{j}^{2}\right\rbrack  }\right| {\theta }_{2}}{MB} + \frac{\left| {\Phi }_{1}\right| {n}_{1}}{B} + \frac{{q}_{2}{n}_{2}}{B} + \frac{{n}_{3}}{B}}\right) 
$$

$$
 = O\left( {\frac{{n}_{1}{\theta }_{2}}{MB} + \frac{\left| {\Phi }_{1}\right| {n}_{1}}{B} + \frac{{q}_{2}{n}_{2}}{B} + \frac{{n}_{3}}{B}}\right)  = O\left( \frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}}\right) .
$$

A similar argument shows that the cost of blue-red emission is bounded by $O\left( {\frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}} + \frac{{n}_{1}}{B}}\right)$ . Finally, by Lemma 7, the cost of blue-blue emission is bounded by:

类似的论证表明，蓝 - 红发射的成本受限于 $O\left( {\frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}} + \frac{{n}_{1}}{B}}\right)$。最后，根据引理 7，蓝 - 蓝发射的成本受限于：

$$
\mathop{\sum }\limits_{{{j}_{1},{j}_{2}}}O\left( {1 + \frac{\left( {\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{light }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right| }\right) {r}_{3}^{\text{light },\text{ light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }{MB}}\right. 
$$

$$
\left. {+\frac{\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right|  + \left| {{r}_{2}^{\text{light }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right|  + \left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right| }{B}}\right) \text{.} \tag{17}
$$

Let us analyze each term of (17) in turn. First:

让我们依次分析 (17) 中的每一项。首先：

$$
\mathop{\sum }\limits_{{{j}_{1},{j}_{2}}}\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| \left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  = \mathop{\sum }\limits_{{j}_{2}}\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| \mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  \tag{18}
$$

$\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|$ gives the number of tuples in ${r}_{3}^{\text{light,light }}$ whose ${A}_{2}$ -values fall in ${I}_{j}^{2}$ . By the way ${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$ are constructed,we know:

$\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|$给出了${r}_{3}^{\text{light,light }}$中${A}_{2}$值落在${I}_{j}^{2}$中的元组数量。根据${I}_{1}^{2},\ldots ,{I}_{{q}_{2}}^{2}$的构造方式，我们知道：

$$
\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  \leq  2{\theta }_{2}.
$$

Therefore:

因此：

$$
\left( {18}\right)  = O\left( {{\theta }_{2}\mathop{\sum }\limits_{{j}_{2}}\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| }\right)  = O\left( {{n}_{1}{\theta }_{2}}\right) .
$$

Symmetrically, we have:

对称地，我们有：

$$
\mathop{\sum }\limits_{{{j}_{1},{j}_{2}}}\left| {{r}_{2}^{\text{light }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right| \left| {{r}_{3}^{\text{light,light }}\left\lbrack  {{I}_{{j}_{1}}^{1},{I}_{{j}_{2}}^{2}}\right\rbrack  }\right|  = O\left( {{n}_{2}{\theta }_{1}}\right) .
$$

Thus, (17) is bounded by:

因此，(17)的上界为：

$$
O\left( {{q}_{1}{q}_{2} + \frac{{n}_{1}{\theta }_{2} + {n}_{2}{\theta }_{1}}{MB} + \frac{{q}_{1}\mathop{\sum }\limits_{{j}_{2}}\left| {{r}_{1}^{\text{light }}\left\lbrack  {I}_{{j}_{2}}^{2}\right\rbrack  }\right| }{B} + \frac{{q}_{2}\mathop{\sum }\limits_{{j}_{1}}\left| {{r}_{2}^{\text{light }}\left\lbrack  {I}_{{j}_{1}}^{1}\right\rbrack  }\right| }{B} + \frac{{n}_{3}}{B}}\right) 
$$

$$
 = O\left( {{q}_{1}{q}_{2} + \frac{{n}_{1}{\theta }_{2} + {n}_{2}{\theta }_{1}}{MB} + \frac{{q}_{1}{n}_{1}}{B} + \frac{{q}_{2}{n}_{2}}{B} + \frac{{n}_{3}}{B}}\right) 
$$

$$
 = O\left( {\frac{\sqrt{{n}_{1}{n}_{2}{n}_{3}}}{B\sqrt{M}} + \frac{{n}_{1}}{B}}\right) .
$$

As already mentioned in Section 4.2,the partitioning phase requires $O\left( {\operatorname{sort}\left( {\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) }\right)$ I/Os. We now complete the proof of Theorem 3.

正如4.2节中已经提到的，分区阶段需要$O\left( {\operatorname{sort}\left( {\mathop{\sum }\limits_{{i = 1}}^{3}{n}_{i}}\right) }\right)$次输入/输出操作。现在我们完成定理3的证明。

## 5 Conclusions

## 5 结论

Checking whether a relation $r$ can be decomposed-as far as natural join is concerned-is extremely important in database systems, and is the key to normalization. This paper presents the first systematic study on I/O-efficient algorithms on this topic. Our results are three-fold. First, we proved that it is NP-hard to check whether $r$ satisfies a specific join dependency $J$ ,even if all the relation schemas in $J$ have only 2 attributes. Second,we presented an I/O-efficient algorithm for determining whether $r$ has redundancy at ${all}$ -namely,if there is any non-trivial $J$ satisfied by $r$ . Our algorithm in fact solves a type of joins known as Loomis-Whitney (LW) Joins. Third, by observing that the classic triangle enumeration problem is a special instance of LW-joins, we further enhanced our LW-join algorithm for this instance, and solved triangle enumeration with the optimal I/O cost under all problem parameters.

就自然连接而言，检查一个关系$r$是否可以分解在数据库系统中极为重要，并且是规范化的关键。本文首次对该主题的输入/输出高效算法进行了系统研究。我们的研究结果有三个方面。首先，我们证明了即使$J$中的所有关系模式都只有2个属性，检查$r$是否满足特定的连接依赖$J$也是NP难问题。其次，我们提出了一种输入/输出高效的算法，用于确定$r$在${all}$处是否存在冗余，即$r$是否满足任何非平凡的$J$。实际上，我们的算法解决了一种称为卢米斯 - 惠特尼（Loomis - Whitney，LW）连接的连接类型。第三，通过观察发现经典的三角形枚举问题是LW连接的一个特殊实例，我们针对该实例进一步改进了LW连接算法，并在所有问题参数下以最优的输入/输出成本解决了三角形枚举问题。

## References

## 参考文献

[1] Serge Abiteboul, Richard Hull, and Victor Vianu. Foundations of Databases. Addison-Wesley Publishing Company, 1995.

[2] Alok Aggarwal and Jeffrey Scott Vitter. The Input/Output Complexity of Sorting and Related Problems. Communications of the ACM (CACM), 31(9):1116-1127, 1988.

[3] Lars Arge, Paolo Ferragina, Roberto Grossi, and Jeffrey Scott Vitter. On Sorting Strings in External Memory (Extended Abstract). In Proceedings of ACM Symposium on Theory of Computing (STOC), pages 540-548, 1997.

[4] Albert Atserias, Martin Grohe, and Dániel Marx. Size Bounds and Query Plans for Relational Joins. SIAM Journal of Computing, 42(4):1737-1767, 2013.

[5] C Beeri and M Vardi. On the Complexity of Testing Implications of Data Dependencies. Computer Science Report, Hebrew Univ, 1980.

[6] Patrick C. Fischer and Don-Min Tsou. Whether a Set of Multivalued Dependencies Implies a Join Dependency is NP-Hard. SIAM Journal of Computing, 12(2):259-266, 1983.

[7] M. R. Garey and David S. Johnson. Computers and Intractability: A Guide to the Theory of ${NP}$ -Completeness. W. H. Freeman,1979.

[8] Xiaocheng Hu, Yufei Tao, and Chin-Wan Chung. I/O-Efficient Algorithms on Triangle Listing and Counting. ACM Transactions on Database Systems (TODS), 39(4):27:1-27:30, 2014.

[9] Paris C. Kanellakis. On the Computational Complexity of Cardinality Constraints in Relational Databases. Information Processing Letters (IPL), 11(2):98-101, 1980.

[10] David Maier. The Theory of Relational Databases. Computer Science Press, 1983.

[11] David Maier, Yehoshua Sagiv, and Mihalis Yannakakis. On the Complexity of Testing Implications of Functional and Join Dependencies. Journal of the ${ACM}\left( {JACM}\right) ,{28}\left( 4\right)  : {680} - {695}$ , 1981.

[12] Hung Q. Ngo, Ely Porat, Christopher Ré, and Atri Rudra. Worst-Case Optimal Join Algorithms: [Extended Abstract]. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 37-48, 2012.

[13] Jean-Marie Nicolas. Mutual Dependencies and Some Results on Undecomposable Relations. In Proceedings of Very Large Data Bases (VLDB), pages 360-367, 1978.

[14] Rasmus Pagh and Francesco Silvestri. The Input/Output Complexity of Triangle Enumeration. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 224-233, 2014.

[15] Todd L. Veldhuizen. Triejoin: A simple, worst-case optimal join algorithm. In Proceedings of International Conference on Database Theory (ICDT), pages 96-106, 2014.