# On Join Sampling and the Hardness of Combinatorial Output-Sensitive Join Algorithms

# 论连接采样与组合输出敏感连接算法的难度

Shiyuan Deng, Shangqi Lu, and Yufei Tao

邓世源、陆尚奇和陶宇飞

\{sydeng,sqlu,taoyf\}@cse.cuhk.edu.hk

\{sydeng,sqlu,taoyf\}@cse.cuhk.edu.hk

Chinese University of Hong Kong

香港中文大学

Hong Kong, China

中国香港

## Abstract

## 摘要

We present a dynamic index structure for join sampling. Built for an (equi-) join $Q$ - let IN be the total number of tuples in the input relations of $Q$ - the structure uses $\widetilde{O}\left( \mathrm{{IN}}\right)$ space,supports a tuple update of any relation in $\widetilde{O}\left( 1\right)$ time,and returns a uniform sample from the join result in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ time with high probability (w.h.p.),where OUT and ${\rho }^{ * }$ are the join’s output size and fractional edge covering number,respectively; notation $\widetilde{O}\left( \text{.}\right)$ hides a factor polylogarithmic to IN. We further show how our result justifies the $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ running time of existing worst-case optimal join algorithms (for full result reporting) even when OUT $\ll  {\mathrm{{IN}}}^{{\rho }^{ * }}$ . Specifically,unless the combinatorial $k$ -clique hypothesis is false,no combinatorial algorithms (i.e., algorithms not relying on fast matrix multiplication) can compute the join result in $O\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time w.h.p. even if $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ ,regardless of how small the constant $\epsilon  > 0$ is.

我们提出了一种用于连接采样的动态索引结构。该结构是为（等值）连接 $Q$ 构建的 —— 设 IN 为 $Q$ 输入关系中的元组总数 —— 该结构使用 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 的空间，支持在 $\widetilde{O}\left( 1\right)$ 时间内对任何关系进行元组更新，并以高概率（w.h.p.）在 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 时间内从连接结果中返回一个均匀样本，其中 OUT 和 ${\rho }^{ * }$ 分别是连接的输出大小和分数边覆盖数；符号 $\widetilde{O}\left( \text{.}\right)$ 隐藏了一个与 IN 呈多对数关系的因子。我们进一步展示了即使当 OUT $\ll  {\mathrm{{IN}}}^{{\rho }^{ * }}$ 时，我们的结果如何证明现有最坏情况最优连接算法（用于完整结果报告）的 $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 运行时间是合理的。具体而言，除非组合 $k$ -团假设不成立，否则即使 $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ ，无论常数 $\epsilon  > 0$ 有多小，任何组合算法（即不依赖快速矩阵乘法的算法）都无法以高概率在 $O\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间内计算出连接结果。

## CCS CONCEPTS

## 计算机与通信安全会议（CCS）概念

- Theory of computation $\rightarrow$ Database query processing and optimization (theory);

- 计算理论 $\rightarrow$ 数据库查询处理与优化（理论）；

## KEYWORDS

## 关键词

Join Algorithms, Sampling, Conjunctive Queries, Lower Bounds

连接算法、采样、合取查询、下界

## ACM Reference Format:

## ACM 引用格式：

Shiyuan Deng, Shangqi Lu, and Yufei Tao. 2023. On Join Sampling and the Hardness of Combinatorial Output-Sensitive Join Algorithms. In Proceedings of the 42nd ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems (PODS '23), June 18-23, 2023, Seattle, WA, USA. ACM, 13 pages. https://doi.org/10.1145/3584372.3588666

邓世源、陆尚奇和陶宇飞。2023 年。论连接采样与组合输出敏感连接算法的难度。收录于第 42 届 ACM SIGMOD - SIGACT - SIGAI 数据库系统原理研讨会（PODS '23）论文集，2023 年 6 月 18 - 23 日，美国华盛顿州西雅图。ACM，13 页。https://doi.org/10.1145/3584372.3588666

## 1 INTRODUCTION

## 1 引言

Joins, which combine the tuples across multiple tables based on equality conditions ${}^{1}$ ,are a fundamental operation in relational algebra and a main performance bottleneck in database systems. Research on joins has been a core field of database theory. Recent years have witnessed significant advances in this field. Particularly, in the realistic scenario where a join involves a constant number of attributes, the community has discovered join algorithms [6, 36, 42, 44- 47, 54] that can achieve the asymptotically optimal performance (sometimes up to a polylogarithmic factor) in the worst case.

连接操作基于相等条件 ${}^{1}$ 组合多个表中的元组，是关系代数中的基本操作，也是数据库系统中的主要性能瓶颈。连接操作的研究一直是数据库理论的核心领域。近年来，该领域取得了显著进展。特别是在连接涉及常量数量属性的现实场景中，学界已经发现了一些连接算法 [6, 36, 42, 44 - 47, 54] ，这些算法在最坏情况下可以实现渐近最优性能（有时相差一个多对数因子）。

Unfortunately, joins remain expensive even in the presence of worst-case optimal algorithms. The culprit is the output size: a join can produce $\Theta \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ tuples [8] where IN denotes the total number of tuples in the input relations,and ${\rho }^{ * }$ represents the join’s fractional edge covering number. We will defer the formal definition of ${\rho }^{ * }$ to Section 2, whereas, for our introductory discussion, it suffices to understand ${\rho }^{ * }$ as a constant at least 1 that is decided by the participating relations’ schemas. For instance, ${\rho }^{ * }$ equals 2 for a join between a relation with attributes $\mathrm{A},\mathrm{\;B}$ and another relation with attributes $\mathrm{B},\mathrm{C}$ ,suggesting that the join may output up to $\Theta \left( {\mathrm{{IN}}}^{2}\right)$ tuples. Even just listing the result necessitates $\Omega \left( {\mathrm{{IN}}}^{2}\right)$ time in the worst case (regardless of which join algorithm is deployed). The phenomenon has a severe impact on database systems because the value IN is gigantic in today’s big-data era,and the value ${\rho }^{ * }$ can be considerably higher for other joins.

不幸的是，即使存在最坏情况最优算法，连接操作的代价仍然很高。问题的根源在于输出规模：一次连接操作可能会产生 $\Theta \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 个元组 [8]，其中 IN 表示输入关系中的元组总数，${\rho }^{ * }$ 表示连接的分数边覆盖数。我们将在第 2 节给出 ${\rho }^{ * }$ 的正式定义，而在引言部分的讨论中，将 ${\rho }^{ * }$ 理解为一个至少为 1 的常数（由参与连接的关系模式决定）就足够了。例如，对于一个属性为 $\mathrm{A},\mathrm{\;B}$ 的关系和一个属性为 $\mathrm{B},\mathrm{C}$ 的关系进行连接，${\rho }^{ * }$ 等于 2，这意味着连接操作最多可能输出 $\Theta \left( {\mathrm{{IN}}}^{2}\right)$ 个元组。即使只是列出结果，在最坏情况下也需要 $\Omega \left( {\mathrm{{IN}}}^{2}\right)$ 的时间（无论采用哪种连接算法）。在当今的大数据时代，IN 的值非常大，而对于其他连接操作，${\rho }^{ * }$ 的值可能会更高，因此这种现象对数据库系统有严重影响。

Fortunately, many downstream tasks of the join operation do not require a complete result, but can benefit significantly from random samples. A classical example is "approximate aggregation" (which arises in OLAP frequently), whose objective is to estimate the result tuples' total value on a selected attribute (e.g., sales). It is well-known that an accurate estimate can be derived from a small number of tuples drawn uniformly at random from the join result. Another, more modern, example is "fair representative reporting" [50], whose objective is to return a few tuples diverse enough to adequately illustrate the join output's overall distribution. Random samples again serve the purpose very well.

幸运的是，连接操作的许多下游任务并不需要完整的结果，而是可以从随机样本中显著受益。一个经典的例子是“近似聚合”（在联机分析处理（OLAP）中经常出现），其目标是估计结果元组在选定属性（例如销售额）上的总值。众所周知，从连接结果中均匀随机抽取少量元组就可以得到准确的估计值。另一个更现代的例子是“公平代表性报告” [50]，其目标是返回几个足够多样化的元组，以充分说明连接输出的总体分布。随机样本再次很好地满足了这一目的。

Due to its profound importance, "join sampling" - the problem of extracting a join result tuple uniformly at random - has attracted considerable attention since its introduction by Chaudhuri et al. [18] in 1999 (a survey will appear in Section 2). The state of the art is an algorithm due to Chen and Yi [21] which,after an initial $\widetilde{O}\left( \mathrm{{IN}}\right)$ -time preprocessing, draws a sample tuple in time

由于其重要性，“连接采样”——即从连接结果中均匀随机抽取一个元组的问题——自 1999 年 Chaudhuri 等人 [18] 提出以来就受到了广泛关注（第 2 节将进行综述）。目前最先进的算法是 Chen 和 Yi [21] 提出的，在经过初始的 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 时间预处理后，该算法可以在一定时间内抽取一个样本元组

$$
\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } + 1}/\max \{ 1,\mathrm{{OUT}}\} }\right)  \tag{1}
$$

with high probability (or w.h.p. for short) - namely, with a probability at least $1 - 1/{\mathrm{{IN}}}^{c}$ for an arbitrarily large constant $c -$ where OUT is the join's output size (i.e., how many tuples in the join result), and the notation $\widetilde{O}\left( \text{.}\right) {hidesafafararatihmictoIN}.{Consider} -$ ing that computing the join result takes $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ time in the worst case, Chen and Yi's method produces a sample in shorter time when OUT $\gg$ IN. They called it an "intriguing open problem" to design an index structure that,after $\widetilde{O}\left( \mathrm{{IN}}\right)$ -time preprocessing,can be used to extract a sample in

具有高概率（简称 w.h.p.）——即对于任意大的常数 $c -$，概率至少为 $1 - 1/{\mathrm{{IN}}}^{c}$，其中 OUT 是连接的输出规模（即连接结果中的元组数量），符号 $\widetilde{O}\left( \text{.}\right) {hidesafafararatihmictoIN}.{Consider} -$ 表示在最坏情况下计算连接结果需要 $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 的时间。当 OUT $\gg$ IN 时，Chen 和 Yi 的方法可以在更短的时间内产生一个样本。他们将设计一种索引结构（在经过 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 时间的预处理后，可以用于在一定时间内抽取一个样本）称为“有趣的开放问题”

$$
\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)  \tag{2}
$$

---

<!-- Footnote -->

${}^{1}$ The joins discussed in this paper are more precisely known as equi-joins,as opposed to theta-joins that use non-equality conditions.

${}^{1}$ 本文讨论的连接操作更准确地称为等值连接，与使用非相等条件的 theta 连接相对。

<!-- Footnote -->

---

time w.h.p.,i.e.,reducing the bound in (1) by a factor of $O\left( \mathrm{{IN}}\right)$ . In [21], Chen and Yi managed to achieve the purpose for a special class of joins, but not for arbitrary joins.

在高概率下的时间，即把 (1) 中的界限降低 $O\left( \mathrm{{IN}}\right)$ 倍。在 [21] 中，Chen 和 Yi 设法为一类特殊的连接操作实现了这一目标，但并非针对任意连接操作。

Overview of Our Results and Techniques. Given an arbitrary join involving a constant number of attributes, we present an index structure fulfilling all the requirements below:

我们的结果和技术概述。对于涉及固定数量属性的任意连接操作，我们提出一种满足以下所有要求的索引结构：

- It occupies $\widetilde{O}\left( \mathrm{{IN}}\right)$ space and can be built in $\widetilde{O}\left( \mathrm{{IN}}\right)$ time.

- 它占用 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 的空间，并且可以在 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 的时间内构建。

- Its sampling time is bounded by (2) w.h.p. (without the value of OUT given). The random samples obtained by repeatedly applying our sampling algorithm are mutually independent.

- 其采样时间在高概率下受 (2) 限制（无需知道 OUT 的值）。通过反复应用我们的采样算法获得的随机样本相互独立。

- It is fully dynamic: inserting and deleting a tuple in any underlying relation takes $\widetilde{O}\left( 1\right)$ time.

- 它是完全动态的：在任何底层关系中插入和删除一个元组需要 $\widetilde{O}\left( 1\right)$ 的时间。

Our structure is different from all the previous solutions to join sampling. In particular, we take a perspective that can be viewed as the opposite of how Chen and Yi [21] approached the problem. They construct a random tuple by growing one attribute at a time, in a manner similar to Generic Join [47], which is a worst-case optimal join algorithm. Crucial to their strategy is the following subproblem: assuming that we have fixed the values ${x}_{1},\ldots ,{x}_{i}$ on attributes ${X}_{1},{X}_{2},\ldots ,{X}_{i}$ for some integer $i$ ,generate a (random) value ${x}_{i + 1}$ for the next attribute ${X}_{i + 1}$ according to a carefully crafted distribution ${}^{2}$ . The subproblem, however, is a major technical barrier, to which Chen and Yi’s solution incurs $\widetilde{O}\left( \mathrm{{IN}}\right)$ time,which is the main reason behind the gap between their complexity (1) and the desired complexity in (2).

我们的结构与以往所有用于连接采样的解决方案都不同。具体而言，我们采用的视角与陈和易 [21] 处理该问题的方式相反。他们每次增加一个属性来构建一个随机元组，其方式类似于通用连接算法 [47]，这是一种最坏情况下的最优连接算法。他们策略的关键在于以下子问题：假设对于某个整数 $i$，我们已经固定了属性 ${X}_{1},{X}_{2},\ldots ,{X}_{i}$ 上的值 ${x}_{1},\ldots ,{x}_{i}$，根据精心设计的分布 ${}^{2}$ 为下一个属性 ${X}_{i + 1}$ 生成一个（随机）值 ${x}_{i + 1}$。然而，这个子问题是一个主要的技术障碍，陈和易的解决方案需要 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 时间，这也是他们的复杂度 (1) 与期望复杂度 (2) 之间存在差距的主要原因。

Rather than attribute values (the finest granularity), our approach works on the attribute space (the coarsest granularity), which is the cartesian product of the domains of all the attributes participating in the join. We prove, in what we call the AGM split theorem, that it is always possible to divide the space into a constant number of subspaces such that

我们的方法不是处理属性值（最细粒度），而是处理属性空间（最粗粒度），属性空间是参与连接的所有属性的域的笛卡尔积。我们在所谓的 AGM 分割定理中证明，总是可以将该空间划分为恒定数量的子空间，使得

- the maximum number of join result tuples in each subspace — characterized by the so-called "AGM bound" [8] (to be introduced in Section 2) — is at most half of the AGM bound of the original space, and

- 每个子空间中连接结果元组的最大数量 —— 由所谓的 “AGM 界” [8]（将在第 2 节中介绍）来表征 —— 至多为原始空间 AGM 界的一半，并且

- the sum of the AGM bounds of the subspaces does not exceed the AGM bound of the original space

- 子空间的 AGM 界之和不超过原始空间的 AGM 界

unless the AGM bound of the original space is already small enough to allow the join to be evaluated in $\widetilde{O}\left( 1\right)$ time. Recursively performing the partitioning leads to a sampling algorithm faster (and simpler) than the state of the art [21]. As a high-level idea, space partitioning has been leveraged to design join algorithms before $\left\lbrack  {6,{27},{36},{42},{44}}\right\rbrack$ . However,the idea’s deployment in those scenarios was for purposes drastically different from ours. The relevant algorithms, as well as their analysis, in the aforementioned works also differ from ours considerably. Our technical development is new and, we believe, clean enough for teaching at a graduate level.

除非原始空间的 AGM 界已经足够小，允许在 $\widetilde{O}\left( 1\right)$ 时间内完成连接评估。递归地进行分区会得到一个比现有技术 [21] 更快（且更简单）的采样算法。从高层次的思路来看，空间分区此前已被用于设计连接算法 $\left\lbrack  {6,{27},{36},{42},{44}}\right\rbrack$。然而，该思路在那些场景中的应用目的与我们的截然不同。上述工作中的相关算法及其分析也与我们的有很大差异。我们的技术发展是全新的，并且我们认为它足够清晰，可以在研究生阶段进行教学。

We also study how much further improvement over (2) is still possible. In fact, perhaps the most challenging step is to pose the right question in this regard. At first glance, (2) appears obviously optimal because OUT,as mentioned,can reach $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ ,in which case (2) becomes $\widetilde{O}\left( 1\right)$ ,clearly the best achievable (we are not concerned with polylogarithmic factors in this work). Although correct, this argument tells us nothing in the realistic scenario where OUT $\ll  {\mathrm{{IN}}}^{{\rho }^{ * }}$ . We instead ask a more meaningful question:

我们还研究了在 (2) 的基础上还能有多大的进一步改进。事实上，也许最具挑战性的一步是在这方面提出正确的问题。乍一看，(2) 显然是最优的，因为如前所述，OUT 可以达到 $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$，在这种情况下 (2) 变为 $\widetilde{O}\left( 1\right)$，显然是可达到的最佳情况（在这项工作中我们不考虑多对数因子）。虽然这个论点是正确的，但在 OUT $\ll  {\mathrm{{IN}}}^{{\rho }^{ * }}$ 的现实场景中，它并没有告诉我们任何信息。我们转而提出一个更有意义的问题：

The join sampling question. Is there a constant $\epsilon$ satisfying $0 < \epsilon  < 1/2$ ,under which we can find a structure that,after an initial $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ -time preprocessing,extracts a uniformly random tuple from the join result in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)$ time w.h.p. when $1 \leq  \mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ ?

连接采样问题。是否存在一个满足 $0 < \epsilon  < 1/2$ 的常数 $\epsilon$，在这个常数下，我们可以找到一种结构，在经过初始的 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 时间预处理后，当 $1 \leq  \mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ 时，以高概率在 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)$ 时间内从连接结果中提取一个均匀随机元组？

Note that the structure does not need to guarantee anything for OUT $= 0$ or OUT $> {\mathrm{{IN}}}^{\epsilon }$ and has sampling time polynomially better than (2) for OUT $\in  \left\lbrack  {1,{\mathrm{{IN}}}^{\epsilon }}\right\rbrack$ .

请注意，对于 OUT $= 0$ 或 OUT $> {\mathrm{{IN}}}^{\epsilon }$，该结构不需要保证任何事情，并且对于 OUT $\in  \left\lbrack  {1,{\mathrm{{IN}}}^{\epsilon }}\right\rbrack$，其采样时间在多项式意义上优于 (2)。

We study the question within the class of combinatorial structures. These are structures whose preprocessing and sampling algorithms are combinatorial, namely, the algorithms do not rely on fast matrix multiplication (à la Strassen's). We prove that the answer to the question is "no" subject to the hypothesis below.

我们在组合结构类中研究这个问题。这些结构的预处理和采样算法是组合性的，即这些算法不依赖于快速矩阵乘法（如斯特拉森算法）。我们证明，在以下假设条件下，该问题的答案是“否”。

Combinatorial k-clique hypothesis. There does not exist any fixed constant $\epsilon  > 0$ under which a combinatorial algorithm can achieve the following for every constant $k \geq  3$ : it can detect with probability at least $1/3$ in $O\left( {n}^{k - \epsilon }\right)$ time whether an undirected graph of $n$ vertices has a $k$ -clique.

组合k - 团假设。不存在任何固定常数$\epsilon  > 0$，使得对于每个常数$k \geq  3$，组合算法能实现以下目标：它能在$O\left( {n}^{k - \epsilon }\right)$时间内以至少$1/3$的概率检测出一个具有$n$个顶点的无向图是否存在$k$ - 团。

The background behind the above hypothesis deserves some explanation. For $k$ being a (constant) multiple of $3,k$ -clique existence in an $n$ -vertex graph can be detected in $O\left( {n}^{{\omega k}/3}\right)$ time where $\omega$ is the matrix multiplication exponent [43] (see [28] for a more complex bound for arbitrary $k = O\left( 1\right)$ ). However,if only combinatorial algorithms are permitted,even beating the naive $O\left( {n}^{k}\right)$ -time approach is difficult: the fastest combinatorial algorithm [53] takes $O\left( {{n}^{k}/{\log }^{k - 1}n}\right)$ time for constant $k \geq  3$ . It is widely conjectured that no combinatorial algorithms can detect $k$ -cliques in $O\left( {n}^{k - \epsilon }\right)$ time for every constant $k \geq  3$ ; such a hypothesis has been applied to argue for computational hardness on a great variety of problems $\left\lbrack  {1,2,{10} - {13},{16},{17},{34},{35},{40},{41}}\right\rbrack$ . Our hypothesis (which requires success probability $1/3$ ) has been explicitly stated in [10,34].

上述假设背后的背景值得解释一下。当$k$是$3,k$的（常数）倍数时，在一个具有$n$个顶点的图中检测$3,k$ - 团是否存在可以在$O\left( {n}^{{\omega k}/3}\right)$时间内完成，其中$\omega$是矩阵乘法指数[43]（关于任意$k = O\left( 1\right)$的更复杂界限见[28]）。然而，如果只允许使用组合算法，即使要超越朴素的$O\left( {n}^{k}\right)$时间算法也很困难：对于常数$k \geq  3$，最快的组合算法[53]需要$O\left( {{n}^{k}/{\log }^{k - 1}n}\right)$时间。人们普遍猜测，对于每个常数$k \geq  3$，不存在能在$O\left( {n}^{k - \epsilon }\right)$时间内检测$k$ - 团的组合算法；这样的假设已被用于论证各种问题的计算难度$\left\lbrack  {1,2,{10} - {13},{16},{17},{34},{35},{40},{41}}\right\rbrack$。我们的假设（要求成功概率为$1/3$）已在文献[10,34]中明确提出。

In fact, if answering the join sampling question was the sole purpose, our argument (in Section 5) could be shortened. However, the argument discloses an inherent connection between join sampling and output-sensitive join computation. For a precise explanation, let us introduce the notion of " $\epsilon$ -output sensitivity":

事实上，如果回答连接采样问题是唯一目的，我们的论证（在第5节）可以简化。然而，该论证揭示了连接采样和输出敏感的连接计算之间的内在联系。为了精确解释，让我们引入“$\epsilon$ - 输出敏感性”的概念：

Let $\epsilon$ be a constant with $0 < \epsilon  < 1/2$ . An algorithm is $\epsilon$ -output sensitive if it can output all the tuples of the join result in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time w.h.p. whenever $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ .

设$\epsilon$是一个满足$0 < \epsilon  < 1/2$的常数。如果一个算法在$\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$时，能以高概率在$\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$时间内输出连接结果的所有元组，那么该算法是$\epsilon$ - 输出敏感的。

Note that the above is different from demanding an algorithm to run in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + \mathrm{{OUT}}}\right)$ time for all OUT values because the notion does not require the algorithm to guarantee anything when OUT $>$ ${\mathrm{{IN}}}^{\epsilon }$ . The significance of $\epsilon$ -output sensitive algorithms is that they convincingly enhance worst-case join algorithms. Although there has been research $\left\lbrack  {6,{36},{44},{49}}\right\rbrack$ on how to compute joins in time sensitive to OUT,none of the known algorithms is $\epsilon$ -output sensitive no matter how $\epsilon$ is chosen: those algorithms’ running time can still degenerate into $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ even when $\mathrm{{OUT}} = 0$ .

请注意，上述概念与要求算法对所有OUT值都在$\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + \mathrm{{OUT}}}\right)$时间内运行不同，因为该概念并不要求算法在OUT $>$ ${\mathrm{{IN}}}^{\epsilon }$时保证任何性能。$\epsilon$ - 输出敏感算法的重要性在于，它们能显著改进最坏情况下的连接算法。尽管已有关于如何以对OUT敏感的时间计算连接的研究$\left\lbrack  {6,{36},{44},{49}}\right\rbrack$，但无论如何选择$\epsilon$，已知的算法都不是$\epsilon$ - 输出敏感的：即使当$\mathrm{{OUT}} = 0$时，这些算法的运行时间仍可能退化为$O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$。

---

<!-- Footnote -->

${}^{2}$ More specifically,the probability of generating ${x}_{i + 1}$ is decided based on how many join result tuples would satisfy ${X}_{j} = {x}_{j}$ for all $j \in  \left\lbrack  {1,i + 1}\right\rbrack$ .

${}^{2}$更具体地说，生成${x}_{i + 1}$的概率是根据对于所有$j \in  \left\lbrack  {1,i + 1}\right\rbrack$有多少连接结果元组满足${X}_{j} = {x}_{j}$来确定的。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: breaking combinatorial $k$ -clique hypothesis a combinatorial join algorithm with time $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + \mathrm{{OUT}}}\right)$ finding a combinatorial $\epsilon$ -output sensi- tive algorithm for an arbitrary $\epsilon$ "Yes" for the join sam- pling question -->

<img src="https://cdn.noedgeai.com/0195cc9b-3c72-737a-b26c-ca2d0bc9b5c5_2.jpg?x=150&y=234&w=706&h=309&r=0"/>

## Figure 1: Reduction relationships (arrow means "implies")

## 图1：归约关系（箭头表示“蕴含”）

<!-- Media -->

We show that any combinatorial $\epsilon$ -output sensitive algorithm can be combined with our join sampling solution to detect whether Join(Q)is empty in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time w.h.p. regardless of the value OUT. However, such a detection algorithm (which is combinatorial) can determine - for all constants $k \geq  3 -$ in $\widetilde{O}\left( {n}^{k - {2\epsilon }}\right)$ time w.h.p. $k$ -clique existence in an $n$ -vertex graph,thus breaking the combinatorial $k$ -clique hypothesis. We further show that any combinatorial structure answering "yes" to the join sampling question with a constant $\epsilon$ satisfying $0 < \epsilon  < 1/2$ implies a combinatorial $\epsilon$ -output sensitive join algorithm. The above discussion yields the relationships in Figure 1, where an arrow from problem A to problem B means "A implies B" (i.e., B can be reduced to A).

我们证明，任何组合式 $\epsilon$ 输出敏感算法都可以与我们的连接采样解决方案相结合，以高概率（w.h.p.）在 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 时间内检测 Join(Q) 是否为空，而与 OUT 的值无关。然而，这样一种（组合式的）检测算法可以在 $\widetilde{O}\left( {n}^{k - {2\epsilon }}\right)$ 时间内（w.h.p.）为所有常数 $k \geq  3 -$ 确定一个 $n$ 顶点图中是否存在 $k$ 团（$k$-clique），从而打破组合式 $k$ 团假设。我们进一步证明，任何以满足 $0 < \epsilon  < 1/2$ 的常数 $\epsilon$ 对连接采样问题回答“是”的组合式结构都意味着存在一个组合式 $\epsilon$ 输出敏感连接算法。上述讨论得出了图 1 中的关系，其中从问题 A 到问题 B 的箭头表示“A 蕴含 B”（即，B 可以归约为 A）。

The findings in Figure 1 have another notable implication. The known worst-case optimal join algorithms (for result reporting) are all combinatorial. Previously, justification on the optimality of an $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ -time join algorithm relied heavily on $\mathrm{{OUT}} = \Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ . Our result suggests that term ${\mathrm{{IN}}}^{{\rho }^{ * }}$ is necessary (up to a sub-polynomial factor) even if OUT $\leq  {\mathrm{{IN}}}^{\epsilon }$ for any constant $\epsilon  > 0$ ,subject to the combinatorial $k$ -clique hypothesis. For example,if a combinatorial algorithm could compute $\mathcal{J}$ oin(Q)in $O\left( {\mathrm{{IN}}}^{{\rho }^{ * } - {0.001}}\right)$ time when OUT $\leq  {\mathrm{{IN}}}^{0.001}$ ,it would make a 0.001-output sensitive algorithm and thus break the hypothesis.

图 1 中的发现还有另一个值得注意的含义。已知的（用于结果报告的）最坏情况最优连接算法都是组合式的。此前，对一个 $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 时间连接算法最优性的论证在很大程度上依赖于 $\mathrm{{OUT}} = \Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$。我们的结果表明，即使对于任何常数 $\epsilon  > 0$ 有 OUT $\leq  {\mathrm{{IN}}}^{\epsilon }$，项 ${\mathrm{{IN}}}^{{\rho }^{ * }}$ 也是必要的（至多相差一个亚多项式因子），前提是组合式 $k$ 团假设成立。例如，如果一个组合式算法能在 OUT $\leq  {\mathrm{{IN}}}^{0.001}$ 时以 $O\left( {\mathrm{{IN}}}^{{\rho }^{ * } - {0.001}}\right)$ 时间计算 $\mathcal{J}$ oin(Q)，那么它将成为一个 0.001 输出敏感算法，从而打破该假设。

Our sampling structure also yields new algorithms on several related problems (e.g., join size estimation, subgraph sampling, randomly permuting the join result with a small delay, join union sampling, etc.). We will elaborate on the details in later sections.

我们的采样结构还为几个相关问题（例如，连接大小估计、子图采样、以小延迟随机排列连接结果、连接并集采样等）产生了新的算法。我们将在后续章节中详细阐述这些细节。

## 2 PRELIMINARIES

## 2 预备知识

Section 2.1 will formally define the join sampling problem. Then, Section 2.2 will introduce the AGM bound, which plays a crucial role in our techniques. Finally, Section 2.3 will present a survey of the existing join algorithms.

第 2.1 节将正式定义连接采样问题。然后，第 2.2 节将介绍 AGM 界，它在我们的技术中起着至关重要的作用。最后，第 2.3 节将对现有的连接算法进行综述。

### 2.1 The Problem of Join Sampling

### 2.1 连接采样问题

Denote by att a finite set whose elements are called attributes. Given a set $U \subseteq$ att,we define a tuple over $U$ as a function $\mathbf{u} : U \rightarrow  \mathbb{N}$ , where $\mathbb{N}$ is the set of integers. If $V$ is a subset of $U$ ,we define the projection of $u$ on $V -$ denoted as $\mathbf{u}\left\lbrack  V\right\rbrack   -$ to be the tuple $v$ over $V$ satisfying $\mathbf{u}\left( X\right)  = \mathbf{v}\left( X\right)$ for every attribute $X \in  V$ . A relation is a set $R$ of tuples over an identical set $U$ of attributes. We refer to $U$ as the schema of $R$ and represent this fact with $\operatorname{var}\left( R\right)  \mathrel{\text{:=}} U$ .

用 att 表示一个有限集，其元素称为属性。给定一个集合 $U \subseteq$ ⊆ att，我们将定义在 $U$ 上的一个元组为一个函数 $\mathbf{u} : U \rightarrow  \mathbb{N}$，其中 $\mathbb{N}$ 是整数集。如果 $V$ 是 $U$ 的一个子集，我们将 $u$ 在 $V -$ 上的投影（记为 $\mathbf{u}\left\lbrack  V\right\rbrack   -$）定义为定义在 $V$ 上的元组 $v$，使得对于每个属性 $X \in  V$ 都有 $\mathbf{u}\left( X\right)  = \mathbf{v}\left( X\right)$。一个关系是定义在同一属性集 $U$ 上的元组的集合 $R$。我们将 $U$ 称为 $R$ 的模式，并使用 $\operatorname{var}\left( R\right)  \mathrel{\text{:=}} U$ 来表示这一事实。

We define a join as a set $Q$ of relations with distinct schemas. Let $\operatorname{var}\left( \mathcal{Q}\right)  \mathrel{\text{:=}} \mathop{\bigcup }\limits_{{R \in  \mathcal{Q}}}\operatorname{var}\left( R\right)$ . The result of the join,denoted as $\mathcal{J}$ oin(Q), is a relation with schema $\operatorname{var}\left( \mathbf{Q}\right)$ given by

我们将连接（join）定义为具有不同模式的关系集合$Q$。设$\operatorname{var}\left( \mathcal{Q}\right)  \mathrel{\text{:=}} \mathop{\bigcup }\limits_{{R \in  \mathcal{Q}}}\operatorname{var}\left( R\right)$。连接的结果，记为$\mathcal{J}$ oin(Q)，是一个具有模式$\operatorname{var}\left( \mathbf{Q}\right)$的关系，其定义如下

$$
\mathcal{J}\text{oin}\left( \mathcal{Q}\right)  \mathrel{\text{:=}} \{ \text{tuple}\mathbf{u}\text{over}\operatorname{var}\left( \mathcal{Q}\right)  \mid  \forall R \in  \mathcal{Q} : \mathbf{u}\left\lbrack  {\operatorname{var}\left( R\right) }\right\rbrack   \in  R\} \text{.}
$$

A join sample of $\mathcal{Q}$ is a uniformly random tuple from $\operatorname{Join}\left( \mathcal{Q}\right)$ .

$\mathcal{Q}$的一个连接样本是从$\operatorname{Join}\left( \mathcal{Q}\right)$中均匀随机选取的一个元组。

The main problem we aim to solve is to design an index structure for $Q$ to support the two operations below:

我们旨在解决的主要问题是为$Q$设计一种索引结构，以支持以下两种操作：

- Extract a join sample of $Q$ . It is required that repeated extraction should produce mutually independent samples (i.e., every sample must be uniformly random even conditioned on all the samples already taken). In the case where $\mathcal{J}$ oin(Q)is empty, this operation should declare so with a special output.

- 提取$Q$的一个连接样本。要求重复提取应产生相互独立的样本（即，即使在已知所有已提取样本的条件下，每个样本也必须是均匀随机的）。当$\mathcal{J}$ oin(Q)为空时，此操作应通过特殊输出声明这一情况。

- Insert or delete a tuple in any relation of $Q$ ,collectively called an "update".

- 在$Q$的任何关系中插入或删除一个元组，统称为“更新”操作。

Focusing on data complexities,we assume that $Q$ has a constant number of relations,and the schema of every relation in $Q$ has a constant number of attributes. We introduce

聚焦于数据复杂度，我们假设$Q$包含固定数量的关系，并且$Q$中每个关系的模式都包含固定数量的属性。我们引入

$$
\mathrm{{IN}} \mathrel{\text{:=}} \mathop{\sum }\limits_{{R \in  Q}}\left| R\right| \text{,and OUT} \mathrel{\text{:=}} \left| {\mathcal{J}\operatorname{oin}\left( Q\right) }\right| 
$$

and refer to them as the input and output size of $Q$ ,respectively.

并分别将它们称为$Q$的输入和输出大小。

### 2.2 The AGM Bound

### 2.2 AGM界

A fundamental question in join processing is how many tuples there can be in the join result. The AGM bound, derived by Atserias et al. [8], answers the question from the graph theory perspective.

连接处理中的一个基本问题是连接结果中可以有多少个元组。由Atserias等人[8]推导得出的AGM界从图论的角度回答了这个问题。

To start with,given a join $\mathcal{Q}$ ,we define a hypergraph $\mathcal{G} \mathrel{\text{:=}} \left( {\mathcal{X},\mathcal{E}}\right)$ where $\mathcal{X} \mathrel{\text{:=}} \operatorname{var}\left( \mathcal{Q}\right)$ and $\mathcal{E} \mathrel{\text{:=}} \{ \operatorname{var}\left( R\right)  \mid  R \in  \mathcal{Q}\}$ . In other words, each vertex in $\mathcal{G}$ corresponds to an attribute involved in the join, and each (hyper) edge in $\mathcal{G}$ corresponds to the schema of an input relation of $Q$ . We will refer to $\mathcal{G}$ as the schema graph of $Q$ . For each edge $e \in  \mathcal{E}$ ,let ${R}_{e}$ be the (only) relation in $Q$ whose schema is $e$ .

首先，给定一个连接$\mathcal{Q}$，我们定义一个超图$\mathcal{G} \mathrel{\text{:=}} \left( {\mathcal{X},\mathcal{E}}\right)$，其中$\mathcal{X} \mathrel{\text{:=}} \operatorname{var}\left( \mathcal{Q}\right)$且$\mathcal{E} \mathrel{\text{:=}} \{ \operatorname{var}\left( R\right)  \mid  R \in  \mathcal{Q}\}$。换句话说，$\mathcal{G}$中的每个顶点对应于连接中涉及的一个属性，$\mathcal{G}$中的每条（超）边对应于$Q$的一个输入关系的模式。我们将$\mathcal{G}$称为$Q$的模式图。对于每条边$e \in  \mathcal{E}$，设${R}_{e}$为$Q$中模式为$e$的（唯一）关系。

A fractional edge covering of $\mathcal{G}$ is a function $W : \mathcal{E} \rightarrow  \mathbb{R}$ ,where $\mathbb{R}$ is the set of real values such that

$\mathcal{G}$的一个分数边覆盖是一个函数$W : \mathcal{E} \rightarrow  \mathbb{R}$，其中$\mathbb{R}$是实数集，使得

- for each $e \in  \mathcal{E},W\left( e\right)  \geq  0$ ;

- 对于每个$e \in  \mathcal{E},W\left( e\right)  \geq  0$；

- for each $X \in  \mathcal{X},\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)  \geq  1$ ,i.e.,the total weight of the edges covering vertex (a.k.a.,attribute) $X$ is at least 1 .

- 对于每个$X \in  \mathcal{X},\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}W\left( e\right)  \geq  1$，即覆盖顶点（也称为属性）$X$的边的总权重至少为1。

The AGM bound shows that any fractional edge covering of $\mathcal{G}$ yields an upper bound on the join result size OUT, as stated below:

AGM界表明，$\mathcal{G}$的任何分数边覆盖都能得出连接结果大小OUT的一个上界，如下所述：

LEMMA 1 (AGM BOUND [8]). Given any fractional edge covering $W$ of $\mathcal{G}$ ,we have $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|  \leq  {\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right)$ where

引理1（AGM界[8]）。给定$\mathcal{G}$的任何分数边覆盖$W$，我们有$\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|  \leq  {\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right)$，其中

$$
{\operatorname{AGM}}_{W}\left( \mathcal{Q}\right)  \mathrel{\text{:=}} \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\right| }^{W\left( e\right) }. \tag{3}
$$

We will refer to ${\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right)$ as the " $\mathrm{{AGM}}$ bound of $\mathcal{Q}$ " when $W$ is understood from the context.

当从上下文可以理解$W$时，我们将${\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right)$称为“$\mathcal{Q}$的$\mathrm{{AGM}}$界”。

Equation (3) expresses an upper bound of OUT using the concrete sizes of the relations in $Q$ . We often want to describe the upper bound more directly using the input size IN. For this purpose, we can apply the trivial fact $\left| {R}_{e}\right|  \leq  \mathrm{{IN}}$ (for all $e \in  \mathcal{E}$ ) to simplify Lemma 1 into OUT $\leq  {\operatorname{IN}}^{\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right) }$ . Note that the exponent $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ is exactly the total weight of all the edges in $\mathcal{E}$ assigned by $W$ . This motivates the concept of fractional edge covering number of $\mathcal{G}$ - denoted as ${\rho }^{ * }$ - which is the smallest $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ among all the fractional edge coverings $W$ . Hence,it always holds that $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{{\rho }^{ * }}$ .

方程 (3) 使用 $Q$ 中关系的具体大小表示了 OUT 的上界。我们通常希望更直接地使用输入大小 IN 来描述这个上界。为此，我们可以应用平凡事实 $\left| {R}_{e}\right|  \leq  \mathrm{{IN}}$（对于所有 $e \in  \mathcal{E}$）将引理 1 简化为 OUT $\leq  {\operatorname{IN}}^{\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right) }$。注意，指数 $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ 恰好是 $\mathcal{E}$ 中由 $W$ 分配的所有边的总权重。这引出了 $\mathcal{G}$ 的分数边覆盖数（fractional edge covering number）的概念——记为 ${\rho }^{ * }$——它是所有分数边覆盖 $W$ 中最小的 $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$。因此，始终有 $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{{\rho }^{ * }}$ 成立。

The AGM bound is tight: given any integer IN $\geq  1$ and hypergraph $\mathcal{G}$ ,we can always find a join $\mathcal{Q}$ ,which has input size IN and schema graph $\mathcal{G}$ ,such that the output size of $\mathcal{Q}$ reaches $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ [8].

算术 - 几何平均（AGM）界是紧的：给定任意整数 IN $\geq  1$ 和超图 $\mathcal{G}$，我们总能找到一个连接 $\mathcal{Q}$，其输入大小为 IN 且模式图为 $\mathcal{G}$，使得 $\mathcal{Q}$ 的输出大小达到 $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ [8]。

### 2.3 Algorithms on Join Processing

### 2.3 连接处理算法

Full Join Computation. The tightness of the AGM bound establishes $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ as a lower bound for the running time of any join algorithm because this amount of time is needed even just to output Join(Q)in the worst case. Ngo et al. [45] were the first to discover a worst-case optimal algorithm that can evaluate any join in $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ time. Since their invention,quite a number of algorithms with time complexity $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ have been subsequently developed $\left\lbrack  {6,{36},{42},{44},{46},{47},{54}}\right\rbrack$ .

全连接计算。AGM 界的紧性将 $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 确立为任何连接算法运行时间的下界，因为即使在最坏情况下仅输出 Join(Q) 也需要这么多时间。Ngo 等人 [45] 首次发现了一种最坏情况最优算法，该算法可以在 $O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 时间内计算任何连接。自他们发明该算法以来，随后又开发了相当多时间复杂度为 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 的算法 $\left\lbrack  {6,{36},{42},{44},{46},{47},{54}}\right\rbrack$。

In practice,a join’s output size OUT may be far less than ${\mathrm{{IN}}}^{{\rho }^{ * }}$ . This motivates the study of output-sensitive algorithms whose running time is sensitive to both IN and OUT. As a notable success, Yannakakis [56] presented an algorithm to process any "acyclic join" in $\widetilde{O}$ (IN + OUT) time. However,acyclic joins are a special type of joins on which stringent restrictions are imposed. Evaluating a join in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + \mathrm{{OUT}}}\right)$ time in the absence of those restrictions,even for an arbitrarily small constant $\epsilon  > 0$ ,is still open.

在实践中，连接的输出大小 OUT 可能远小于 ${\mathrm{{IN}}}^{{\rho }^{ * }}$。这激发了对输出敏感算法的研究，这类算法的运行时间对 IN 和 OUT 都敏感。作为一个显著的成功案例，Yannakakis [56] 提出了一种算法，可在 $\widetilde{O}$ (IN + OUT) 时间内处理任何“无环连接”。然而，无环连接是一种特殊类型的连接，对其施加了严格的限制。在没有这些限制的情况下，即使对于任意小的常数 $\epsilon  > 0$，要在 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + \mathrm{{OUT}}}\right)$ 时间内计算一个连接，仍然是一个未解决的问题。

Several non-trivial attempts have been made to tackle the open challenge. When executed on a general join, all the known output-sensitive algorithms $\left\lbrack  {6,{36},{44},{49}}\right\rbrack$ have running time of the form

已经有一些非平凡的尝试来解决这个未解决的挑战。当应用于一般连接时，所有已知的输出敏感算法 $\left\lbrack  {6,{36},{44},{49}}\right\rbrack$ 的运行时间形式为

$$
\widetilde{O}\left( {{\mathrm{{Cer}}}^{\text{width }} + \mathrm{{OUT}}}\right) 
$$

where width quantifies how much the schema graph $\mathcal{G}$ deviates from a tree (it can be defined using, e.g., tree width [24], query width [19], fractional hypertree width [31], etc.), and Cer is a value, called the "certificate size", measuring how difficult the input relations are for join processing in the instance-oriented sense (e.g., IN is a common value for Cer; see $\left\lbrack  {{36},{44}}\right\rbrack$ for other Cer definitions based on specialized "certificates"). Unfortunately, for all the algorithms in $\left\lbrack  {6,{36},{44},{49}}\right\rbrack$ ,the term Cer ${}^{\text{width }}$ always ends up being $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ at some "unfriendly" joins. Therefore, as far as general joins are concerned,no algorithms with $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + }\right.$ OUT) time have been found,no matter how small the constant $\epsilon$ is.

其中宽度量化了模式图 $\mathcal{G}$ 与树的偏离程度（可以使用例如树宽 [24]、查询宽度 [19]、分数超树宽 [31] 等来定义），而 Cer 是一个值，称为“证书大小”，用于衡量在面向实例的意义上输入关系进行连接处理的难度（例如，IN 是 Cer 的一个常见值；有关基于专门“证书”的其他 Cer 定义，请参见 $\left\lbrack  {{36},{44}}\right\rbrack$）。不幸的是，对于 $\left\lbrack  {6,{36},{44},{49}}\right\rbrack$ 中的所有算法，术语 Cer ${}^{\text{width }}$ 在某些“不友好”的连接中最终总是为 $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$。因此，就一般连接而言，无论常数 $\epsilon$ 有多小，都尚未找到具有 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon } + }\right.$ OUT) 时间复杂度的算法。

All the above algorithms are combinatorial. We are not aware of any non-combinatorial approaches for computing $\mathcal{J}$ oin(Q). There exist,however,algorithms $\left\lbrack  {7,{25},{26}}\right\rbrack$ that use matrix multiplication to evaluate conjunctive queries with projections. A join in our definition can be understood as a conjunctive query without projections, in which case the algorithms of $\left\lbrack  {7,{25},{26}}\right\rbrack$ do not promise faster running time than the combinatorial methods discussed earlier.

上述所有算法都是组合算法。我们不知道有任何非组合方法可用于计算 $\mathcal{J}$ oin(Q)。然而，存在一些算法 $\left\lbrack  {7,{25},{26}}\right\rbrack$ 使用矩阵乘法来评估带有投影的合取查询。在我们的定义中，连接可以理解为没有投影的合取查询，在这种情况下，$\left\lbrack  {7,{25},{26}}\right\rbrack$ 中的算法并不保证比前面讨论的组合方法运行时间更快。

Join Sampling. In 1999, Chaudhuri et al. [18] initialized the study on join sampling. They focused on joins with two relations, i.e., $Q = \left\{  {{R}_{1},{R}_{2}}\right\}$ ,and described a structure of $O\left( \mathrm{{IN}}\right)$ space that allows extracting a sample from $\mathcal{J}$ oin(Q)in constant time. They also proved a lower bound that, if no preprocessing is allowed, taking a sample demands $\Omega \left( \mathrm{{IN}}\right)$ time. Acharya et al. [3] considered joins with more than two relations, but their formal results apply only when the relations of $Q$ obey the so-called "star schema",namely,there is a "center relation" that has a foreign key to every other relation. Sampling is trivial on star-schema joins because it boils down to drawing a random tuple from a single relation (i.e., the center one).

连接采样。1999 年，乔杜里（Chaudhuri）等人 [18] 开启了对连接采样的研究。他们专注于两个关系的连接，即 $Q = \left\{  {{R}_{1},{R}_{2}}\right\}$，并描述了 $O\left( \mathrm{{IN}}\right)$ 空间的一种结构，该结构允许在常数时间内从 $\mathcal{J}$ oin(Q) 中提取一个样本。他们还证明了一个下界，即如果不允许进行预处理，抽取一个样本需要 $\Omega \left( \mathrm{{IN}}\right)$ 时间。阿查里雅（Acharya）等人 [3] 考虑了多于两个关系的连接，但他们的正式结果仅适用于 $Q$ 中的关系遵循所谓的“星型模式”的情况，即存在一个“中心关系”，它对其他每个关系都有一个外键。在星型模式连接上进行采样很简单，因为这归结为从单个关系（即中心关系）中随机抽取一个元组。

No theoretical progress had been documented on join sampling in the next 18 years following the work of $\left\lbrack  {3,{18}}\right\rbrack$ (in the meantime, the problem had received a huge amount of attention from the system community, as discussed later). Theory advancement resumed in 2018. For any acyclic join $Q$ ,Zhao et al. [58] presented an $O\left( \mathrm{{IN}}\right)$ -space structure that permits drawing a sample from $\operatorname{Join}\left( \mathcal{Q}\right)$ in constant time. By combining their structure with hypertree decompositions ${}^{3}$ ,one can obtain a structure for an arbitrary join $Q$ that has $O\left( \mathrm{{IN}}\right)$ space and $\widetilde{O}\left( {\mathrm{{IN}}}^{\text{fhtw }}\right)$ sampling time,where fhtw is the fractional hypertree width of $Q$ ; in the worst case,however,fhtw $= {\rho }^{ * }$ , causing the sampling time to degenerate into $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ . In 2020,Chen and $\mathrm{{Yi}}\left\lbrack  {21}\right\rbrack$ identified a class of joins under the name sequenceable joins,for which they obtained a (static) structure of $O\left( \mathrm{{IN}}\right)$ space that can sample from the join result in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ time w.h.p. ${}^{4}$ . For general (non-sequenceable) joins,their structure still works,but the sampling time deteriorates by a factor of $O\left( \mathrm{{IN}}\right)$ to $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } + 1}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ .

在$\left\lbrack  {3,{18}}\right\rbrack$的工作之后的18年里，连接采样（join sampling）方面没有记录到理论进展（同时，正如后文所述，该问题受到了系统领域社区的大量关注）。理论进展在2018年得以恢复。对于任何无环连接$Q$，Zhao等人[58]提出了一种$O\left( \mathrm{{IN}}\right)$空间结构，该结构允许在常数时间内从$\operatorname{Join}\left( \mathcal{Q}\right)$中抽取一个样本。通过将他们的结构与超树分解（hypertree decompositions）${}^{3}$相结合，可以为任意连接$Q$获得一种结构，该结构具有$O\left( \mathrm{{IN}}\right)$空间和$\widetilde{O}\left( {\mathrm{{IN}}}^{\text{fhtw }}\right)$采样时间，其中fhtw是$Q$的分数超树宽度（fractional hypertree width）；然而，在最坏的情况下，fhtw $= {\rho }^{ * }$，导致采样时间退化为$\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$。2020年，Chen和$\mathrm{{Yi}}\left\lbrack  {21}\right\rbrack$确定了一类名为可排序连接（sequenceable joins）的连接，为此他们获得了一种$O\left( \mathrm{{IN}}\right)$空间的（静态）结构，该结构可以以高概率（w.h.p.）${}^{4}$在$\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$时间内从连接结果中采样。对于一般的（不可排序的）连接，他们的结构仍然有效，但采样时间会恶化$O\left( \mathrm{{IN}}\right)$倍，变为$\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } + 1}/\max \{ 1,\mathrm{{OUT}}\} }\right)$。

Closely relevant to join sampling is the direct access (DA) problem on join computation. In that problem, there is a pre-agreed ordering on the tuples of $\oint \operatorname{join}\left( Q\right)$ such that,given an integer $k \in  \left\lbrack  {1,\mathrm{{OUT}}}\right\rbrack$ , a DA query returns the $k$ -th tuple in $\mathcal{J}$ oin(Q). If a structure can answer a DA query in ${T}_{\mathrm{{DA}}}$ time,we can use the query to draw a sample from $\mathcal{J}$ oin(Q)in $O\left( {T}_{\mathrm{{DA}}}\right)$ time with a random value $k \in  \left\lbrack  {1,\mathrm{{OUT}}}\right\rbrack$ , where the value OUT is available from preprocessing. When $Q$ is "free-connex",there is a structure of $\widetilde{O}\left( \mathrm{{IN}}\right)$ space answering a DA query in $\widetilde{O}\left( 1\right)$ time [14,15],which means that the structure also guarantees $\widetilde{O}\left( 1\right)$ sample time. However,because a free-connex $Q$ is necessarily acyclic, the sampling result is subsumed by that of [58].

与连接采样密切相关的是连接计算中的直接访问（direct access，DA）问题。在该问题中，$\oint \operatorname{join}\left( Q\right)$的元组有一个预先约定的顺序，使得给定一个整数$k \in  \left\lbrack  {1,\mathrm{{OUT}}}\right\rbrack$，一个DA查询会返回$\mathcal{J}$oin(Q)中的第$k$个元组。如果一个结构可以在${T}_{\mathrm{{DA}}}$时间内回答一个DA查询，我们可以使用该查询，通过一个随机值$k \in  \left\lbrack  {1,\mathrm{{OUT}}}\right\rbrack$在$O\left( {T}_{\mathrm{{DA}}}\right)$时间内从$\mathcal{J}$oin(Q)中抽取一个样本，其中OUT值可以通过预处理获得。当$Q$是“自由连通（free - connex）”时，存在一种$\widetilde{O}\left( \mathrm{{IN}}\right)$空间的结构可以在$\widetilde{O}\left( 1\right)$时间内回答一个DA查询[14,15]，这意味着该结构也保证了$\widetilde{O}\left( 1\right)$的采样时间。然而，由于自由连通的$Q$必然是无环的，其采样结果被[58]的结果所包含。

Finally, we note that join sampling has been studied extensively in system research (see $\left\lbrack  {5,{20},{33},{38},{39},{48},{51},{55},{57},{59}}\right\rbrack$ and the references therein), which has produced numerous empirically efficient solutions. In the worst case, however, those solutions all require $\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ time to draw one sample,regardless of the value of OUT. All the above join sampling solutions, theoretical or empirical, are combinatorial.

最后，我们注意到连接采样在系统研究中得到了广泛的研究（见$\left\lbrack  {5,{20},{33},{38},{39},{48},{51},{55},{57},{59}}\right\rbrack$及其参考文献），并产生了许多在经验上有效的解决方案。然而，在最坏的情况下，无论OUT的值是多少，这些解决方案都需要$\Omega \left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$时间来抽取一个样本。上述所有连接采样解决方案，无论是理论的还是经验的，都是组合性的。

## 3 THE AGM SPLIT THEOREM

## 3 算术 - 几何均值（AGM）分裂定理

This section will establish the AGM split theorem, which serves as the technical core of our sampling algorithm. The theorem provides a simple and intuitive way to split the "attribute space", with the guarantee that the upper bound on the join result size given by the AGM bound gets (at least) halved in each subspace after the split.

本节将建立AGM分割定理（AGM split theorem），该定理是我们采样算法的技术核心。该定理提供了一种简单直观的方法来分割“属性空间”，并保证分割后每个子空间中由AGM边界给出的连接结果大小的上界（至少）减半。

Recall that $\operatorname{var}\left( \mathcal{Q}\right)$ is the set of attributes involved in the join. Set $d \mathrel{\text{:=}} \left| {\operatorname{var}\left( \mathcal{Q}\right) }\right|$ . Let us impose an arbitrary ordering on the attributes of $\operatorname{var}\left( \mathcal{Q}\right)$ ,which can then be denoted as ${X}_{1},{X}_{2},\ldots ,{X}_{d}$ . This way, every tuple $\mathbf{u}$ in the join result Join(Q)can be interpreted as a point in ${\mathbb{N}}^{d}$ ,where $\mathbf{u}\left( {X}_{i}\right)$ is the point’s $i$ -th coordinate for each $i \in  \left\lbrack  {1,d}\right\rbrack$ . The attribute space is now formally defined to be ${\mathbb{N}}^{d}$ .

回顾一下，$\operatorname{var}\left( \mathcal{Q}\right)$ 是连接中涉及的属性集。设 $d \mathrel{\text{:=}} \left| {\operatorname{var}\left( \mathcal{Q}\right) }\right|$ 。让我们对 $\operatorname{var}\left( \mathcal{Q}\right)$ 中的属性施加任意顺序，然后可以将其表示为 ${X}_{1},{X}_{2},\ldots ,{X}_{d}$ 。这样，连接结果Join(Q)中的每个元组 $\mathbf{u}$ 都可以解释为 ${\mathbb{N}}^{d}$ 中的一个点，其中对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack$ ，$\mathbf{u}\left( {X}_{i}\right)$ 是该点的 $i$ 坐标。现在正式将属性空间定义为 ${\mathbb{N}}^{d}$ 。

Next, we introduce box-induced sub-join, a notion imperative in our subsequent discussion. Let $B$ be a box in the attribute space, namely, $B$ has the form $\left\lbrack  {{x}_{1},{y}_{1}}\right\rbrack   \times  \left\lbrack  {{x}_{2},{y}_{2}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {{x}_{d},{y}_{d}}\right\rbrack$ . For each $i \in  \left\lbrack  {1,d}\right\rbrack$ ,we use $B\left( {X}_{i}\right)$ to denote $\left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$ ,i.e.,the projection of $B$ on the $i$ -th attribute. On every relation $R \in  \mathcal{Q}$ ,the box $B$ induces a "subrelation" $R\left( B\right)$ ,which includes all the tuples of $R$ "falling" into $B$ . Care must be taken here because $R$ may not include all the attributes in $\operatorname{var}\left( \mathcal{Q}\right)$ . We say that a tuple $\mathbf{u} \in  R$ falls in $B$ if $B\left( X\right)$ covers $\mathbf{u}\left( X\right)$ for every attribute $X \in  \operatorname{var}\left( R\right) .R\left( B\right)$ can then be formalized as

接下来，我们引入盒诱导子连接（box-induced sub-join）的概念，这在我们后续的讨论中至关重要。设 $B$ 是属性空间中的一个盒，即 $B$ 具有 $\left\lbrack  {{x}_{1},{y}_{1}}\right\rbrack   \times  \left\lbrack  {{x}_{2},{y}_{2}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {{x}_{d},{y}_{d}}\right\rbrack$ 的形式。对于每个 $i \in  \left\lbrack  {1,d}\right\rbrack$ ，我们用 $B\left( {X}_{i}\right)$ 表示 $\left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$ ，即 $B$ 在第 $i$ 个属性上的投影。在每个关系 $R \in  \mathcal{Q}$ 上，盒 $B$ 诱导出一个“子关系” $R\left( B\right)$ ，它包含 $R$ 中所有“落入” $B$ 的元组。这里必须小心，因为 $R$ 可能不包含 $\operatorname{var}\left( \mathcal{Q}\right)$ 中的所有属性。我们说一个元组 $\mathbf{u} \in  R$ 落入 $B$ ，如果对于每个属性 $X \in  \operatorname{var}\left( R\right) .R\left( B\right)$ ，$B\left( X\right)$ 覆盖 $\mathbf{u}\left( X\right)$ ，那么 $X \in  \operatorname{var}\left( R\right) .R\left( B\right)$ 可以形式化为

$$
R\left( B\right)  \mathrel{\text{:=}} \{ \mathbf{u} \in  R \mid  \mathbf{u}\text{ falls in }B\} . \tag{4}
$$

---

<!-- Footnote -->

${}^{3}$ See Appendix A of [36] for an introduction to such decompositions.

${}^{3}$ 有关此类分解的介绍，请参阅 [36] 的附录A。

${}^{4}$ In [21],Chen and Yi claimed the sampling time as $O\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ in expectation,but under the assumption that an expression of the form ${x}^{y}$ (for a fractional $y$ ) can be evaluated in constant time. Removing the assumption incurs an $O\left( {\log \mathrm{{IN}}}\right)$ multiplicative factor. Moreover, it is standard to make their time complexity hold w.h.p. by paying yet another $O\left( {\log \mathrm{{IN}}}\right)$ multiplicative factor.

${}^{4}$ 在 [21] 中，Chen和Yi声称采样时间的期望值为 $O\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ ，但这是在形式为 ${x}^{y}$ （对于分数 $y$ ）的表达式可以在常数时间内计算的假设下。去掉这个假设会产生一个 $O\left( {\log \mathrm{{IN}}}\right)$ 的乘法因子。此外，通过再付出一个 $O\left( {\log \mathrm{{IN}}}\right)$ 的乘法因子，使他们的时间复杂度以高概率成立是很常见的做法。

<!-- Footnote -->

---

By putting together the $R\left( B\right)$ of all $R \in  Q$ ,we have the "sub-join induced by $B$ ",formally defined as

通过将所有 $R \in  Q$ 的 $R\left( B\right)$ 组合在一起，我们得到了“由 $B$ 诱导的子连接”，正式定义为

$$
Q\left( B\right)  \mathrel{\text{:=}} \{ R\left( B\right)  \mid  R \in  Q\} . \tag{5}
$$

Given a sub-join $\mathcal{Q}\left( B\right)$ and an attribute $X \in  \operatorname{var}\left( \mathcal{Q}\right)$ ,we will need to be concerned with the set of $X$ -values appearing in at least one relation of $\mathcal{Q}\left( B\right)$ . This can be formalized as

给定一个子连接 $\mathcal{Q}\left( B\right)$ 和一个属性 $X \in  \operatorname{var}\left( \mathcal{Q}\right)$，我们需要关注在 $\mathcal{Q}\left( B\right)$ 的至少一个关系中出现的 $X$ 值的集合。这可以形式化为

$$
\operatorname{actdom}\left( {X,B}\right)  \mathrel{\text{:=}} \{ x \in  \mathbb{N} \mid  \exists R\left( B\right)  \in  Q\left( B\right) ,\mathbf{u} \in  R\left( B\right)  : 
$$

$$
X \in  \operatorname{var}\left( R\right) ,\mathbf{u}\left( X\right)  = x\}  \tag{6}
$$

which will be referred to as the "active $X$ -domain induced by $B$ ".

这将被称为“由 $B$ 诱导的活跃 $X$ 域”。

We will reason about the AGM bounds on box-induced sub-joins under a fixed fractional edge covering $W$ . For that purpose,we "overload" the function ${\mathrm{{AGM}}}_{W}$ ,defined in (3),in a manner that will prove handy in our analysis. Let $\mathcal{G} \mathrel{\text{:=}} \left( {\mathcal{X},\mathcal{E}}\right)$ be the schema graph of $Q$ (defined in Section 2.2) and $W$ be an arbitrary fractional edge covering of $\mathcal{G}$ . Given a box $B$ ,we define

我们将在固定的分数边覆盖 $W$ 下推导盒诱导子连接的 AGM 边界。为此，我们以一种在分析中会很方便的方式“重载”在 (3) 中定义的函数 ${\mathrm{{AGM}}}_{W}$。设 $\mathcal{G} \mathrel{\text{:=}} \left( {\mathcal{X},\mathcal{E}}\right)$ 为 $Q$ 的模式图（在 2.2 节中定义），$W$ 为 $\mathcal{G}$ 的任意分数边覆盖。给定一个盒 $B$，我们定义

$$
{\operatorname{AGM}}_{W}\left( B\right)  \mathrel{\text{:=}} {\operatorname{AGM}}_{W}\left( {\mathcal{Q}\left( B\right) }\right)  = \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) }. \tag{7}
$$

With a slight abuse of notation,the "overloading" allows ${\mathrm{{AGM}}}_{W}$ to take a box as the parameter directly. No ambiguity can arise because, as we have seen,every box $B$ defines a sub-join $Q\left( B\right)$ . By Lemma 1, ${\operatorname{AGM}}_{W}\left( B\right)$ is an upper bound of the result size of $\mathcal{Q}\left( B\right)$ . It is worth mentioning that ${\mathrm{{AGM}}}_{W}\left( {\mathbb{N}}^{d}\right)$ - the parameter is set to the attribute space (the largest box) - equals $\operatorname{AGM}\left( \mathcal{Q}\right)$ ,the $\operatorname{AGM}$ bound of the original join $Q$ . We will refer to ${\mathrm{{AGM}}}_{W}\left( B\right)$ as the " $\mathrm{{AGM}}$ bound of $B$ ",when $W$ is clear from the context.

稍微滥用一下符号，这种“重载”允许 ${\mathrm{{AGM}}}_{W}$ 直接将盒作为参数。不会产生歧义，因为正如我们所见，每个盒 $B$ 都定义了一个子连接 $Q\left( B\right)$。根据引理 1，${\operatorname{AGM}}_{W}\left( B\right)$ 是 $\mathcal{Q}\left( B\right)$ 结果大小的上界。值得一提的是，当参数设置为属性空间（最大的盒）时，${\mathrm{{AGM}}}_{W}\left( {\mathbb{N}}^{d}\right)$ 等于 $\operatorname{AGM}\left( \mathcal{Q}\right)$，即原始连接 $Q$ 的 $\operatorname{AGM}$ 边界。当从上下文可以清楚知道 $W$ 时，我们将 ${\mathrm{{AGM}}}_{W}\left( B\right)$ 称为“$B$ 的 $\mathrm{{AGM}}$ 边界”。

Before unveiling the AGM split theorem, we need to clarify some "oracles" that provide efficient implementations of certain primitive operations. Specifically, two oracles will be useful:

在揭示 AGM 拆分定理之前，我们需要澄清一些“预言机”，它们提供了某些基本操作的高效实现。具体来说，有两个预言机会很有用：

- Count oracle: Given a relation $R \in  Q$ and a box $B$ ,the oracle returns the number of tuples of $R\left( B\right)$ in $\widetilde{O}\left( 1\right)$ time,where $R\left( B\right)$ is defined in (4).

- 计数预言机：给定一个关系 $R \in  Q$ 和一个盒 $B$，该预言机在 $\widetilde{O}\left( 1\right)$ 时间内返回 $R\left( B\right)$ 的元组数量，其中 $R\left( B\right)$ 在 (4) 中定义。

- Median oracle: Given an attribute $X \in  \operatorname{var}\left( \mathcal{Q}\right)$ and a box $B$ , the oracle returns the median value ${}^{5}$ of actdom(X,B)in $\widetilde{O}\left( 1\right)$ time,where $\operatorname{actdom}\left( {X,B}\right)$ is defined in (6).

- 中位数预言机：给定一个属性 $X \in  \operatorname{var}\left( \mathcal{Q}\right)$ 和一个盒 $B$，该预言机在 $\widetilde{O}\left( 1\right)$ 时间内返回 actdom(X,B) 的中位数 ${}^{5}$，其中 $\operatorname{actdom}\left( {X,B}\right)$ 在 (6) 中定义。

Both oracles can be implemented with rudimentary data structures, as will be discussed in later sections. Here, we will proceed by assuming that they have been made available at our disposal.

这两个预言机都可以用基本的数据结构实现，后续章节会讨论。在这里，我们将假设它们已经可供我们使用并继续进行。

THEOREM 2 (AGM SPLIT THEOREM). Fix an arbitrary fractional edge covering $W$ of $\mathcal{G}$ ,and assume the availability of count and median oracles. Given any box $B$ with ${\mathrm{{AGM}}}_{W}\left( B\right)  \geq  2$ , we can find in $\widetilde{O}\left( 1\right)$ time a set $C$ of at most ${2d} + 1$ (where $d \mathrel{\text{:=}} \left| {\operatorname{var}\left( \mathcal{Q}\right) }\right|$ ) boxes such that

定理 2（AGM 拆分定理）。固定 $\mathcal{G}$ 的任意分数边覆盖 $W$，并假设计数和中位数预言机可用。给定任意满足 ${\mathrm{{AGM}}}_{W}\left( B\right)  \geq  2$ 的盒 $B$，我们可以在 $\widetilde{O}\left( 1\right)$ 时间内找到一个最多包含 ${2d} + 1$ 个盒（其中 $d \mathrel{\text{:=}} \left| {\operatorname{var}\left( \mathcal{Q}\right) }\right|$）的集合 $C$，使得

(1) the boxes in $C$ are disjoint and have $B$ as their union;

(1) $C$ 中的盒互不相交，且它们的并集为 $B$；

(2) for each box ${B}^{\prime } \in  \mathcal{C},{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$ ;

(2) 对于每个盒子 ${B}^{\prime } \in  \mathcal{C},{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$ ;

(3) $\mathop{\sum }\limits_{{{B}^{\prime } \in  C}}{\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)  \leq  {\mathrm{{AGM}}}_{W}\left( B\right)$ .

The rest of the section serves as a proof of the theorem, which consists of two main parts. First, we will present a technical lemma to reveal an underlying mathematical relationship in AGM bounds that concerns splitting a box along one of the attributes. Then, we will utilize the lemma to design an efficient algorithm to produce the desired set $C$ of boxes.

本节的其余部分用作该定理的证明，它由两个主要部分组成。首先，我们将给出一个技术性引理，以揭示AGM边界（Adversarial Generalization Margin bounds，对抗泛化边界）中与沿某个属性分割盒子有关的潜在数学关系。然后，我们将利用该引理设计一个高效的算法来生成所需的盒子集合 $C$。

Let us start with the technical lemma. To facilitate explanation, it will be helpful to introduce a function replace(B,i,I),where $B$ is a box in the attribute space, $i$ is an integer between 1 and $d$ ,and $I$ is an interval of integers. The function yields the box

让我们从这个技术性引理开始。为便于解释，引入一个函数 replace(B,i,I) 会很有帮助，其中 $B$ 是属性空间中的一个盒子，$i$ 是介于 1 和 $d$ 之间的一个整数，$I$ 是一个整数区间。该函数会生成盒子

replace $\left( {B,i,I}\right)  \mathrel{\text{:=}} B\left( {X}_{1}\right)  \times  \ldots  \times  B\left( {X}_{i - 1}\right)  \times  I \times  B\left( {X}_{i + 1}\right)  \times  \ldots  \times  B\left( {X}_{d}\right)$ that is,replacing the projection of $B$ on attribute ${X}_{i}$ with $I$ ,while retaining the projections on the other attributes.

替换 $\left( {B,i,I}\right)  \mathrel{\text{:=}} B\left( {X}_{1}\right)  \times  \ldots  \times  B\left( {X}_{i - 1}\right)  \times  I \times  B\left( {X}_{i + 1}\right)  \times  \ldots  \times  B\left( {X}_{d}\right)$，即，将 $B$ 在属性 ${X}_{i}$ 上的投影替换为 $I$，同时保留在其他属性上的投影。

To state our lemma,let us fix a fractional edge covering $W$ of $\mathcal{G}$ and a box $B$ ,as we do in Theorem 2. In addition,fix an arbitrary attribute ${X}_{i}$ ,for some $i \in  \left\lbrack  {1,d}\right\rbrack$ . Suppose that we partition the (integer) interval $B\left( {X}_{i}\right)$ arbitrarily into $s$ disjoint integer intervals ${I}_{1},{I}_{2},\ldots ,{I}_{s}$ where $s$ can be any value at least 2 ( $s$ does not need to be a constant). For each $j \in  \left\lbrack  {1,s}\right\rbrack$ ,the interval ${I}_{j}$ defines a box

为了陈述我们的引理，让我们像在定理 2 中那样，固定 $\mathcal{G}$ 的一个分数边覆盖 $W$ 和一个盒子 $B$。此外，对于某个 $i \in  \left\lbrack  {1,d}\right\rbrack$，固定一个任意属性 ${X}_{i}$。假设我们将（整数）区间 $B\left( {X}_{i}\right)$ 任意划分为 $s$ 个不相交的整数区间 ${I}_{1},{I}_{2},\ldots ,{I}_{s}$，其中 $s$ 可以是至少为 2 的任意值（$s$ 不需要是一个常数）。对于每个 $j \in  \left\lbrack  {1,s}\right\rbrack$，区间 ${I}_{j}$ 定义了一个盒子

$$
{B}_{j} \mathrel{\text{:=}} \operatorname{replace}\left( {B,i,{I}_{j}}\right) . \tag{8}
$$

It is clear that ${B}_{1},{B}_{2},\ldots ,{B}_{s}$ are mutually disjoint and have $B$ as their union. Our technical lemma can now be presented as:

显然，${B}_{1},{B}_{2},\ldots ,{B}_{s}$ 是相互不相交的，并且它们的并集是 $B$。我们的技术引理现在可以表述为：

LEMMA 3. $\mathop{\sum }\limits_{{j = 1}}^{s}{\operatorname{AGM}}_{W}\left( {B}_{j}\right)  \leq  {\operatorname{AGM}}_{W}\left( B\right)$ .

引理 3. $\mathop{\sum }\limits_{{j = 1}}^{s}{\operatorname{AGM}}_{W}\left( {B}_{j}\right)  \leq  {\operatorname{AGM}}_{W}\left( B\right)$。

The above lemma is, in fact, Lemma 6 of [27] in disguise. Unfortunately, Lemma 6 of [27] was presented in a sophisticated context, because of which the reader would find it difficult to recognize the two lemmas' resemblance. We present a standalone proof in Appendix A for the sake of self-containment.

事实上，上述引理是文献 [27] 中引理 6 的另一种形式。不幸的是，文献 [27] 中的引理 6 是在一个复杂的上下文中给出的，因此读者很难识别出这两个引理的相似之处。为了内容的自包含性，我们在附录 A 中给出了一个独立的证明。

Equipped with the lemma, next we explain how to obtain the set $\mathcal{C}$ of boxes in Theorem 2 using $\widetilde{O}\left( 1\right)$ time. The following simple proposition will be useful throughout the paper.

有了这个引理，接下来我们解释如何在 $\widetilde{O}\left( 1\right)$ 时间内得到定理 2 中的盒子集合 $\mathcal{C}$。以下简单的命题在本文中会很有用。

Proposition 1. Given any box $B$ ,we can compute ${\operatorname{AGM}}_{W}\left( B\right)$ in $\widetilde{O}\left( 1\right)$ time.

命题 1. 给定任意盒子 $B$，我们可以在 $\widetilde{O}\left( 1\right)$ 时间内计算 ${\operatorname{AGM}}_{W}\left( B\right)$。

Proof. We first use the count oracle to obtain $\left| {{R}_{e}\left( B\right) }\right|$ in $\widetilde{O}\left( 1\right)$ time for each edge $e \in  \mathcal{E}$ (remember that $\mathcal{E}$ is the set of edges in the schema graph $\mathcal{G}$ ),and then feed these values into (7) to compute ${\mathrm{{AGM}}}_{W}\left( B\right) .{}^{6}$ The proposition holds because $\mathcal{E}$ has a constant number of edges.

证明。我们首先使用计数预言机在 $\widetilde{O}\left( 1\right)$ 时间内为每条边 $e \in  \mathcal{E}$ 获取 $\left| {{R}_{e}\left( B\right) }\right|$（记住 $\mathcal{E}$ 是模式图 $\mathcal{G}$ 中的边的集合），然后将这些值代入 (7) 来计算 ${\mathrm{{AGM}}}_{W}\left( B\right) .{}^{6}$。因为 $\mathcal{E}$ 中的边的数量是常数，所以该命题成立。

Figure 2 presents our algorithm for splitting a box $B$ into a set $\mathcal{C}$ of smaller boxes meeting the requirements of Theorem 2. The algorithm,split(i,B),admits two parameters: an integer $i \in  \left\lbrack  {1,d}\right\rbrack$ and a box $B = \left\lbrack  {{x}_{1},{y}_{1}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {{x}_{d},{y}_{d}}\right\rbrack$ . The box should satisfy the constraint that its projections on the first $i - 1$ attributes must be singleton intervals (a singleton interval $\left\lbrack  {x,y}\right\rbrack$ contains only one value,i.e., $x = y$ ). Note that the constraint does not prevent us from supplying an arbitrary box as $B$ ,as long as we set $i = 1$ in that case. The integer $i$ stands for the "split attribute". Specifically,given any value $z \in  \left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$ ,we can split $B$ into (at most) three boxes:

图2展示了我们将一个盒子 $B$ 分割成一组满足定理2要求的较小盒子 $\mathcal{C}$ 的算法。该算法split(i,B)有两个参数：一个整数 $i \in  \left\lbrack  {1,d}\right\rbrack$ 和一个盒子 $B = \left\lbrack  {{x}_{1},{y}_{1}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {{x}_{d},{y}_{d}}\right\rbrack$ 。该盒子应满足其在前 $i - 1$ 个属性上的投影必须是单元素区间（单元素区间 $\left\lbrack  {x,y}\right\rbrack$ 仅包含一个值，即 $x = y$ ）这一约束条件。请注意，该约束条件并不妨碍我们将任意盒子作为 $B$ ，只要在这种情况下我们设置 $i = 1$ 即可。整数 $i$ 表示“分割属性”。具体而言，给定任意值 $z \in  \left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$ ，我们可以将 $B$ 分割成（最多）三个盒子：

$$
{B}_{\text{left }} \mathrel{\text{:=}} \operatorname{replace}\left( {B,i,\left\lbrack  {{x}_{i},z - 1}\right\rbrack  }\right) 
$$

$$
{B}_{\text{mid }} \mathrel{\text{:=}} \operatorname{replace}\left( {B,i,\left\lbrack  {z,z}\right\rbrack  }\right) 
$$

$$
{B}_{\text{right }} \mathrel{\text{:=}} \text{ replace }\left( {B,i,\left\lbrack  {z + 1,{y}_{i}}\right\rbrack  }\right) .
$$

---

<!-- Footnote -->

${}^{6}$ Computing a quantity like ${\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) }$ requires a power operator - one that evaluates an expression like ${x}^{y}$ for a fractional $y$ - which is commonly assumed to take constant time in the join literature; e.g., see previous work $\left\lbrack  {{21},{27}}\right\rbrack$ . Strictly speaking the standard RAM model does not provide such an operator. One simple way to circumvent the issue is to use the observation that $W\left( e\right)$ has only $\left| \mathcal{E}\right|  = O\left( 1\right)$ different choices, i.e.,one for each $e \in  \left| \mathcal{E}\right|$ ,and $\left| {{R}_{e}\left( B\right) }\right|$ must be an integer from 0 to IN. Therefore, we can prepare,for each $e \in  \mathcal{E}$ ,the values ${1}^{W\left( e\right) },{2}^{W\left( e\right) },\ldots ,{\mathrm{{IN}}}^{W\left( e\right) }$ in preprocessing and store them all in $O\left( \mathrm{{IN}}\right)$ extra space. As a less straightforward,but more powerful, remedy,we can calculate ${\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) }$ up to an additive error of $1/{\mathrm{{IN}}}^{c}$ for some sufficiently large constant $c$ ,which is easy to achieve in $\widetilde{O}\left( 1\right)$ time. Such a precision level is sufficient for our algorithms in this paper to work. Henceforth, we will no longer dwell on the issue but will simply assume that the power operator takes $\widetilde{O}\left( 1\right)$ time. The reader may also refer to Section 6 of [32] for a relevant discussion.

${}^{6}$ 计算像 ${\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) }$ 这样的量需要一个幂运算符——一个能对分数 $y$ 计算形如 ${x}^{y}$ 表达式的值的运算符——在连接文献中通常假设该运算符的计算时间为常数；例如，参见先前的工作 $\left\lbrack  {{21},{27}}\right\rbrack$ 。严格来说，标准随机存取机（RAM）模型并不提供这样的运算符。一种简单的解决方法是利用这样的观察结果： $W\left( e\right)$ 只有 $\left| \mathcal{E}\right|  = O\left( 1\right)$ 种不同的选择，即每种 $e \in  \left| \mathcal{E}\right|$ 对应一种选择，并且 $\left| {{R}_{e}\left( B\right) }\right|$ 必须是一个从0到IN的整数。因此，我们可以在预处理阶段为每个 $e \in  \mathcal{E}$ 计算值 ${1}^{W\left( e\right) },{2}^{W\left( e\right) },\ldots ,{\mathrm{{IN}}}^{W\left( e\right) }$ ，并将它们全部存储在 $O\left( \mathrm{{IN}}\right)$ 的额外空间中。作为一种不太直接但更强大的解决办法，我们可以在某个足够大的常数 $c$ 的加法误差范围内计算 ${\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) }$ ，这在 $\widetilde{O}\left( 1\right)$ 时间内很容易实现。这样的精度水平足以使本文中的算法正常工作。此后，我们将不再详述这个问题，而是简单地假设幂运算符的计算时间为 $\widetilde{O}\left( 1\right)$ 。读者也可以参考文献[32]的第6节以获取相关讨论。

${}^{5}$ If a set $S$ has $n$ values,the median of $S$ is the $\lceil n/2\rceil$ -th smallest value in $S$ .

${}^{5}$ 如果一个集合 $S$ 有 $n$ 个值，那么 $S$ 的中位数是 $S$ 中第 $\lceil n/2\rceil$ 小的值。

<!-- Footnote -->

---

<!-- Media -->

---

algorithm split(i,B)

算法split(i,B)

/* assume $B = \left\lbrack  {{x}_{1},{y}_{1}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {{x}_{d},{y}_{d}}\right\rbrack$ ; it is required

/* 假设 $B = \left\lbrack  {{x}_{1},{y}_{1}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {{x}_{d},{y}_{d}}\right\rbrack$ ；这是必需的

that ${x}_{1} = {y}_{1},{x}_{2} = {y}_{2},\ldots$ ,and ${x}_{i - 1} = {y}_{i - 1} *$ /

即 ${x}_{1} = {y}_{1},{x}_{2} = {y}_{2},\ldots$ ，且 ${x}_{i - 1} = {y}_{i - 1} *$ /

1. $C \leftarrow  \varnothing$

2. $z \leftarrow$ the largest value in $\left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$ s.t. ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\mathrm{{AGM}}}_{W}\left( B\right)$

2. $z \leftarrow$是$\left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$中满足${\mathrm{{AGM}}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\mathrm{{AGM}}}_{W}\left( B\right)$的最大值

		where ${B}_{\text{left }} \leftarrow$ replace $\left( {B,i,\left\lbrack  {{x}_{i},z - 1}\right\rbrack  }\right)$

		其中${B}_{\text{left }} \leftarrow$替换$\left( {B,i,\left\lbrack  {{x}_{i},z - 1}\right\rbrack  }\right)$

3. if $z - 1 \geq  {x}_{i}$ then $C \leftarrow  C \cup  \left\{  {B}_{\text{left }}\right\}$

3. 如果$z - 1 \geq  {x}_{i}$，那么$C \leftarrow  C \cup  \left\{  {B}_{\text{left }}\right\}$

4. ${B}_{\text{mid }} \leftarrow  \operatorname{replace}\left( {B,i,\left\lbrack  {z,z}\right\rbrack  }\right)$

5. if $i = d$ then add ${B}_{\text{mid }}$ to $C$

5. 如果$i = d$，那么将${B}_{\text{mid }}$添加到$C$中

6. else $C \leftarrow  C \cup  \operatorname{split}\left( {i + 1,{B}_{\text{mid }}}\right)$

6. 否则$C \leftarrow  C \cup  \operatorname{split}\left( {i + 1,{B}_{\text{mid }}}\right)$

7. if $z + 1 \leq  {y}_{i}$ then $C \leftarrow  C \cup  \left\{  {B}_{\text{right }}\right\}$ where

7. 如果$z + 1 \leq  {y}_{i}$，那么$C \leftarrow  C \cup  \left\{  {B}_{\text{right }}\right\}$，其中

				${B}_{\text{right }} \leftarrow  \operatorname{replace}\left( {B,i,\left\lbrack  {z + 1,{y}_{i}}\right\rbrack  }\right)$

8. return $C$

8. 返回$C$

					## Figure 2: The split algorithm for Theorem 2

					## ## 图2：定理2的分割算法

---

<!-- Media -->

As a special case, ${B}_{\text{left }}$ or ${B}_{\text{right }}$ does not exist if $z = {x}_{i}$ or ${y}_{i}$ , respectively. We want to find the largest $z$ satisfying the condition ${\operatorname{AGM}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$ (Line 2); $z$ definitely exists because the condition is fulfilled by $z = {x}_{i}$ (in which case ${B}_{\text{left }}$ is empty and the condition is vacuously met). The boxes ${B}_{\text{left }}$ and ${B}_{\text{right }}$ ,if they exist,are added directly to $\mathcal{C}$ (Line 3 and 7). Regarding ${B}_{\text{mid }}$ , notice that it now has singleton projections on the first $i$ attributes. If $i = d$ ,then ${B}_{\text{mid }}$ has degenerated into a point in the attribute space and is also added to $C$ (Line 5). Otherwise,we recursively invoke split $\left( {i + 1,{B}_{\text{mid }}}\right)$ to split ${B}_{\text{mid }}$ into a set ${C}^{\prime }$ of boxes and union ${C}^{\prime }$ into $C$ (Line 6).

作为一种特殊情况，如果分别满足$z = {x}_{i}$或${y}_{i}$，则${B}_{\text{left }}$或${B}_{\text{right }}$不存在。我们想找到满足条件${\operatorname{AGM}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$的最大$z$（第2行）；$z$肯定存在，因为该条件可由$z = {x}_{i}$满足（在这种情况下，${B}_{\text{left }}$为空，条件自然满足）。如果${B}_{\text{left }}$和${B}_{\text{right }}$存在，则将它们直接添加到$\mathcal{C}$中（第3行和第7行）。关于${B}_{\text{mid }}$，注意到它现在在前$i$个属性上有单元素投影。如果$i = d$，那么${B}_{\text{mid }}$已退化为属性空间中的一个点，并且也被添加到$C$中（第5行）。否则，我们递归调用分割$\left( {i + 1,{B}_{\text{mid }}}\right)$将${B}_{\text{mid }}$分割成一组盒子${C}^{\prime }$，并将${C}^{\prime }$合并到$C$中（第6行）。

Next,we show that the set $\mathcal{C}$ produced by split(1,B)has the three properties in Theorem 2. Property 1 is obvious and omitted. Let us turn to Property 2. For every ${B}_{\text{left }}$ created by a recursive call split $\left( {i,{B}^{\prime }}\right)$ ,where ${B}^{\prime }$ is some box inside $B$ generated in the recursion,we have ${\operatorname{AGM}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$ . Hence,every ${B}_{\text{left }}$ added to $C$ must satisfy Property 2 .

接下来，我们证明由split(1,B)生成的集合$\mathcal{C}$具有定理2中的三个性质。性质1很明显，在此省略。让我们来看性质2。对于由递归调用split $\left( {i,{B}^{\prime }}\right)$创建的每个${B}_{\text{left }}$，其中${B}^{\prime }$是递归中生成的$B$内的某个盒子，我们有${\operatorname{AGM}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$。因此，添加到$C$中的每个${B}_{\text{left }}$都必须满足性质2。

Let us switch attention to the ${B}_{\text{right }}$ in an (arbitrary) recursive call split $\left( {i,{B}^{\prime }}\right)$ . By definition of the value $z$ at Line 2,it must hold that ${\operatorname{AGM}}_{W}\left( {{B}_{\text{left }} \cup  {B}_{\text{mid }}}\right)  \geq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)$ . Lemma 3,on the other hand, tells us that ${\mathrm{{AGM}}}_{W}\left( {{B}_{\text{left }} \cup  {B}_{\text{mid }}}\right)  + {\mathrm{{AGM}}}_{W}\left( {B}_{\text{right }}\right)  \leq  {\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)$ (apply the lemma with $s = 2,{B}_{1} = {B}_{\text{left }} \cup  {B}_{\text{mid }}$ ,and ${B}_{2} = {B}_{\text{right }}$ ). This gives ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{right }}\right)  \leq  \frac{1}{2}{\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)  \leq  \frac{1}{2}{\mathrm{{AGM}}}_{W}\left( B\right)$ . Hence, every ${B}_{\text{right }}$ added to $\mathcal{C}$ also satisfies Property 2.

让我们将注意力转向任意递归调用分割 $\left( {i,{B}^{\prime }}\right)$ 中的 ${B}_{\text{right }}$。根据第 2 行中值 $z$ 的定义，必定有 ${\operatorname{AGM}}_{W}\left( {{B}_{\text{left }} \cup  {B}_{\text{mid }}}\right)  \geq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)$ 成立。另一方面，引理 3 告诉我们 ${\mathrm{{AGM}}}_{W}\left( {{B}_{\text{left }} \cup  {B}_{\text{mid }}}\right)  + {\mathrm{{AGM}}}_{W}\left( {B}_{\text{right }}\right)  \leq  {\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)$（将该引理应用于 $s = 2,{B}_{1} = {B}_{\text{left }} \cup  {B}_{\text{mid }}$ 和 ${B}_{2} = {B}_{\text{right }}$）。由此可得 ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{right }}\right)  \leq  \frac{1}{2}{\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)  \leq  \frac{1}{2}{\mathrm{{AGM}}}_{W}\left( B\right)$。因此，添加到 $\mathcal{C}$ 中的每个 ${B}_{\text{right }}$ 也满足性质 2。

It remains to discuss ${B}_{\text{mid }}$ . In all the recursive calls to split, the only ${B}_{\text{mid }}$ added to $\mathcal{C}$ is the final one that has degenerated into a point (Line 5). For such a ${B}_{\text{mid }}$ ,we have ${\operatorname{AGM}}_{W}\left( {B}_{\text{mid }}\right)  \leq  1$ ,which is at most $\frac{1}{2}{\mathrm{{AGM}}}_{W}\left( B\right)$ because ${\mathrm{{AGM}}}_{W}\left( B\right)  \geq  2$ .

还需讨论 ${B}_{\text{mid }}$。在所有对分割的递归调用中，添加到 $\mathcal{C}$ 中的唯一 ${B}_{\text{mid }}$ 是最终退化为一个点的那个（第 5 行）。对于这样的 ${B}_{\text{mid }}$，我们有 ${\operatorname{AGM}}_{W}\left( {B}_{\text{mid }}\right)  \leq  1$，由于 ${\mathrm{{AGM}}}_{W}\left( B\right)  \geq  2$，它至多为 $\frac{1}{2}{\mathrm{{AGM}}}_{W}\left( B\right)$。

Lastly, Property 3 is a corollary of Lemma 3. At any recursive call split $\left( {i,{B}^{\prime }}\right)$ ,the lemma indicates ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{left }}\right)  + {\mathrm{{AGM}}}_{W}\left( {B}_{\text{mid }}\right)  +$ ${\operatorname{AGM}}_{W}\left( {B}_{\text{right }}\right)  \leq  {\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)$ . Now,Property 3 follows from a simple inductive argument on $i$ (for breaking ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{mid }}\right)$ into the sum of the AGM bounds of smaller boxes).

最后，性质 3 是引理 3 的一个推论。在任何递归调用分割 $\left( {i,{B}^{\prime }}\right)$ 时，该引理表明 ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{left }}\right)  + {\mathrm{{AGM}}}_{W}\left( {B}_{\text{mid }}\right)  +$ ${\operatorname{AGM}}_{W}\left( {B}_{\text{right }}\right)  \leq  {\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)$。现在，性质 3 可通过对 $i$ 进行简单的归纳论证得出（即将 ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{mid }}\right)$ 分解为较小盒子的算术 - 几何均值（AGM）边界之和）。

The set $\mathcal{C}$ returned by split(1,B)has a size no more than ${2d} + 1$ because we add at most two boxes to $\mathcal{C}$ for each $i \in  \left\lbrack  {1,d - 1}\right\rbrack$ and at most three for $i = d$ . Each call to split,excluding the recursive invocation at Line 6,runs in $\widetilde{O}\left( 1\right)$ time. In particular,Line 2 can be implemented in $\widetilde{O}\left( 1\right)$ time due to Proposition 1 and the fact that $z$ can be found with binary search in the active ${X}_{i}$ -domain induced by $B$ ,which necessitates only $O\left( {\log \mathrm{{IN}}}\right)$ calls to the median oracle. As the overall recursion depth is $d = O\left( 1\right)$ ,the total running time of split(1,B)is $\widetilde{O}\left( 1\right)$ . This completes the proof of Theorem 2.

由split(1,B)返回的集合$\mathcal{C}$的大小不超过${2d} + 1$，因为对于每个$i \in  \left\lbrack  {1,d - 1}\right\rbrack$，我们最多向$\mathcal{C}$中添加两个盒子，对于$i = d$最多添加三个。除了第6行的递归调用外，每次对split的调用都在$\widetilde{O}\left( 1\right)$时间内运行。特别地，由于命题1以及可以在由$B$诱导的活跃${X}_{i}$ - 域中通过二分查找找到$z$这一事实，第2行可以在$\widetilde{O}\left( 1\right)$时间内实现，这只需要对中位数预言机进行$O\left( {\log \mathrm{{IN}}}\right)$次调用。由于总的递归深度为$d = O\left( 1\right)$，split(1,B)的总运行时间为$\widetilde{O}\left( 1\right)$。这就完成了定理2的证明。

Remark. In Proposition 8 of [27], Deep and Koutris presented a splitting result in the so-called "lexicographical order". Under the lexicographical order,each tuple $\mathbf{u}$ in the attribute space is viewed as a string of $d$ characters $\mathbf{u}\left( {X}_{1}\right) \mathbf{u}\left( {X}_{2}\right) \ldots \mathbf{u}\left( {X}_{d}\right)$ ,and two tuples are compared by their string representations alphabetically. An interval $\left\lbrack  {{\mathbf{u}}_{1},{\mathbf{u}}_{2}}\right\rbrack$ under the order includes all the tuples $\mathbf{v}$ alphabetically between ${\mathbf{u}}_{1}$ and ${\mathbf{u}}_{2}$ . The goal in [27] is to divide $\left\lbrack  {{\mathbf{u}}_{1},{\mathbf{u}}_{2}}\right\rbrack$ into (i) two intervals whose "AGM bounds" (see [27] for what this means) are at most half of that of $\left\lbrack  {{\mathbf{u}}_{1},{\mathbf{u}}_{2}}\right\rbrack$ and (ii) a tuple. Their split algorithm, which differs from ours in Figure 2, also uses "boxes" somehow, but the "boxes" there are specially constrained, as opposed to being arbitrary boxes. Although related, the statements in our Theorem 2 and Proposition 8 of [27] present distinct findings, neither of which subsumes the other.

备注。在文献[27]的命题8中，迪普（Deep）和库特里斯（Koutris）提出了一个在所谓“字典序”下的分割结果。在字典序下，属性空间中的每个元组$\mathbf{u}$被视为一个由$d$个字符$\mathbf{u}\left( {X}_{1}\right) \mathbf{u}\left( {X}_{2}\right) \ldots \mathbf{u}\left( {X}_{d}\right)$组成的字符串，并且通过它们的字符串表示按字母顺序比较两个元组。在该顺序下的一个区间$\left\lbrack  {{\mathbf{u}}_{1},{\mathbf{u}}_{2}}\right\rbrack$包含按字母顺序介于${\mathbf{u}}_{1}$和${\mathbf{u}}_{2}$之间的所有元组$\mathbf{v}$。文献[27]的目标是将$\left\lbrack  {{\mathbf{u}}_{1},{\mathbf{u}}_{2}}\right\rbrack$划分为（i）两个“算术 - 几何平均（AGM）界”（其含义见文献[27]）至多为$\left\lbrack  {{\mathbf{u}}_{1},{\mathbf{u}}_{2}}\right\rbrack$的一半的区间，以及（ii）一个元组。他们的分割算法与图2中的我们的算法不同，它也以某种方式使用了“盒子”，但那里的“盒子”有特殊的约束，而不是任意的盒子。尽管相关，但我们的定理2和文献[27]的命题8呈现了不同的结果，彼此都不包含对方。

## 4 JOIN SAMPLING

## 4 连接采样

We are ready to solve the join sampling problem. To that end, Section 4.1 first introduces "the join box-tree", a conceptual hierarchy that paves the foundation of the proposed sampling algorithm, which is presented in Section 4.2. Our discussion in Sections 4.1 and 4.2 will assume the count and median oracles (defined in Section 3), whose implementation will be explained in Section 4.3.

我们准备好解决连接采样问题了。为此，4.1节首先介绍“连接盒子树”，这是一个概念性的层次结构，为所提出的采样算法奠定了基础，该算法将在4.2节中介绍。我们在4.1节和4.2节的讨论将假设使用计数预言机和中位数预言机（在第3节中定义），其实现将在4.3节中解释。

### 4.1 The Join Box-Tree

### 4.1 连接盒子树

Our AGM split theorem shows how to split an arbitrary box in the attribute space. Next, we will repeatedly utilize the theorem to partition the space into sufficiently small boxes with "trivial AGM bounds". This will produce a tree $\mathcal{T}$ that we name the join box-tree.

我们的算术 - 几何平均（AGM）分割定理展示了如何分割属性空间中的任意盒子。接下来，我们将反复利用该定理将空间划分为具有“平凡AGM界”的足够小的盒子。这将产生一棵树$\mathcal{T}$，我们将其命名为连接盒子树。

Fix an arbitrary fractional edge covering $W$ of the schema graph $\mathcal{G} \mathrel{\text{:=}} \left( {\mathcal{X},\mathcal{E}}\right)$ of the input join $Q$ . Define $\rho  \mathrel{\text{:=}} \mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ ,i.e.,the total weight of the edges in $\mathcal{E}$ under $W$ . The value $\rho$ is a constant that does not depend on IN.

固定输入连接$Q$的模式图$\mathcal{G} \mathrel{\text{:=}} \left( {\mathcal{X},\mathcal{E}}\right)$的任意分数边覆盖$W$。定义$\rho  \mathrel{\text{:=}} \mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$，即$\mathcal{E}$中的边在$W$下的总权重。值$\rho$是一个不依赖于输入（IN）的常数。

The join box-tree $\mathcal{T}$ is a tree dependent on $W$ . An internal node in $\mathcal{T}$ has at most ${2d} + 1$ child nodes,where $d \mathrel{\text{:=}} \left| {\operatorname{var}\left( \mathcal{Q}\right) }\right|$ . Every node, no matter leaf or internal, is associated with a box, and no two nodes are associated with the same box. For this reason, henceforth, if a node is associated with a box $B$ ,we will use $B$ to denote that node as well. Given a node $B$ ,we refer to ${\mathrm{{AGM}}}_{W}\left( B\right)$ ,defined in (7), as the node's AGM bound. Every internal node has an AGM bound at least 2, whereas every leaf node has an AGM bound less than 2.

连接盒树 $\mathcal{T}$ 是一棵依赖于 $W$ 的树。$\mathcal{T}$ 中的内部节点最多有 ${2d} + 1$ 个子节点，其中 $d \mathrel{\text{:=}} \left| {\operatorname{var}\left( \mathcal{Q}\right) }\right|$ 。每个节点，无论是叶节点还是内部节点，都与一个盒（box）相关联，并且没有两个节点与同一个盒相关联。因此，此后，如果一个节点与一个盒 $B$ 相关联，我们也将使用 $B$ 来表示该节点。给定一个节点 $B$ ，我们将 (7) 中定义的 ${\mathrm{{AGM}}}_{W}\left( B\right)$ 称为该节点的 AGM 界（AGM bound）。每个内部节点的 AGM 界至少为 2，而每个叶节点的 AGM 界小于 2。

Next,we formally define $\mathcal{T}$ in a top-down manner. The root of $\mathcal{T}$ is associated with the box ${\mathbb{N}}^{d}$ ,i.e.,the entire attribute space. Consider,in general,a node associated with box $B$ . If node $B$ has an AGM bound less than 2,we make it a leaf of $\mathcal{T}$ . Otherwise, $B$ is an internal node with child nodes created in two steps.

接下来，我们以自顶向下的方式正式定义 $\mathcal{T}$ 。$\mathcal{T}$ 的根节点与盒 ${\mathbb{N}}^{d}$ 相关联，即整个属性空间。一般来说，考虑一个与盒 $B$ 相关联的节点。如果节点 $B$ 的 AGM 界小于 2，我们将其设为 $\mathcal{T}$ 的叶节点。否则，$B$ 是一个内部节点，其子节点分两步创建。

(1) Apply Theorem 2 to split $B$ into a set $C$ of boxes.

(1) 应用定理 2 将 $B$ 分割成一个盒的集合 $C$ 。

(2) For each box ${B}^{\prime } \in  \mathcal{C}$ ,create a child node,associated with box ${B}^{\prime }$ ,of node $B$ . The number of child nodes of $B$ is $\left| C\right|$ .

(2) 对于每个盒 ${B}^{\prime } \in  \mathcal{C}$ ，创建节点 $B$ 的一个子节点，该子节点与盒 ${B}^{\prime }$ 相关联。$B$ 的子节点数量为 $\left| C\right|$ 。

We now proceed to explore the properties of the join box-tree, starting with:

我们现在开始探索连接盒树的性质，从以下内容开始：

Proposition 2. T has height $O\left( {\log \mathrm{{IN}}}\right)$ .

命题 2. T 的高度为 $O\left( {\log \mathrm{{IN}}}\right)$ 。

Proof. Every time we descend from a parent node to a child, the AGM bound decreases by at least a factor of two (Property 2 of Theorem 2). The root $B$ of $\mathcal{T}$ has an AGM bound that equals the AGM bound of $\mathcal{Q}$ (given by Lemma 1),which is no more than ${\mathrm{{IN}}}^{\rho }$ , where $\rho  = \mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ as mentioned earlier. As a node becomes a leaf as soon as its AGM bound drops below 2, we can descend only $O\left( {\log {\mathrm{{IN}}}^{\rho }}\right)  = O\left( {\log \mathrm{{IN}}}\right)$ levels.

证明。每次我们从父节点下降到子节点时，AGM 界至少降低一半（定理 2 的性质 2）。$\mathcal{T}$ 的根节点 $B$ 的 AGM 界等于 $\mathcal{Q}$ 的 AGM 界（由引理 1 给出），该界不超过 ${\mathrm{{IN}}}^{\rho }$ ，其中 $\rho  = \mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ 如前文所述。由于节点的 AGM 界一旦降至 2 以下就会成为叶节点，我们最多只能下降 $O\left( {\log {\mathrm{{IN}}}^{\rho }}\right)  = O\left( {\log \mathrm{{IN}}}\right)$ 层。

The following property focuses on the leaf nodes of $\mathcal{T}$ .

以下性质关注 $\mathcal{T}$ 的叶节点。

Proposition 3. The boxes of all the leaves of $\mathcal{T}$ are disjoint and have the attribute space ${\mathbb{N}}^{d}$ as the union.

命题 3. $\mathcal{T}$ 所有叶节点的盒是不相交的，并且它们的并集是属性空间 ${\mathbb{N}}^{d}$ 。

Proof. The root of $\mathcal{T}$ has the entire attribute space as its associated box. In general, the box of a parent node is partitioned by the boxes of the child nodes, and the boxes of the child nodes are always disjoint (Property 1 of Theorem 2). This proves the proposition.

证明。$\mathcal{T}$ 的根节点的关联盒是整个属性空间。一般来说，父节点的盒由子节点的盒划分，并且子节点的盒总是不相交的（定理 2 的性质 1）。这就证明了该命题。

The lemma below echoes our statement in Section 1 that a box with a "small-enough" AGM bound corresponds to a join that can be evaluated in near-constant time.

下面的引理呼应了我们在第 1 节中的陈述，即具有“足够小”AGM 界的盒对应于可以在近似常数时间内计算的连接。

LEMMA 4. Consider an arbitrary leaf of $\mathcal{T}$ . Let $B$ be the box associated with the leaf. The result of the join $Q\left( B\right)$ ,which contains at most one tuple,can be computed in $\widetilde{O}\left( 1\right)$ time,assuming the count and median oracles.

引理 4. 考虑 $\mathcal{T}$ 的任意一个叶节点。设 $B$ 是与该叶节点相关联的盒。连接 $Q\left( B\right)$ 的结果最多包含一个元组，假设存在计数和中位数预言机（count and median oracles），则该结果可以在 $\widetilde{O}\left( 1\right)$ 时间内计算。

Proof. First,compute ${\operatorname{AGM}}_{W}\left( B\right)$ in $\widetilde{O}\left( 1\right)$ time (Proposition 1). If ${\operatorname{AGM}}_{W}\left( B\right)  = 0$ ,we declare the join result $\mathcal{J}$ oin $\left( {\mathcal{Q}\left( B\right) }\right)$ empty. Otherwise, ${\mathrm{{AGM}}}_{W}\left( B\right)$ is at least 1 (because Equation (7),if not equal to 0,must be at least 1 ) but less than 2 (because $B$ is a leaf). It follows immediately that $\mathcal{J}$ oin $\left( {\mathcal{Q}\left( B\right) }\right)$ can have at most one tuple.

证明。首先，在$\widetilde{O}\left( 1\right)$时间内计算${\operatorname{AGM}}_{W}\left( B\right)$（命题1）。如果${\operatorname{AGM}}_{W}\left( B\right)  = 0$，我们声明连接结果$\mathcal{J}$连接$\left( {\mathcal{Q}\left( B\right) }\right)$为空。否则，${\mathrm{{AGM}}}_{W}\left( B\right)$至少为1（因为根据等式(7)，若不等于0，则必须至少为1）但小于2（因为$B$是一个叶子节点）。由此立即可以得出，$\mathcal{J}$连接$\left( {\mathcal{Q}\left( B\right) }\right)$最多只能有一个元组。

We can utilize algorithm split in Figure 2 to compute the result of $\mathcal{Q}\left( B\right)$ . For this purpose,run split(1,B)and collect the set $\mathcal{C}$ of boxes returned. We claim that every box ${B}^{\prime } \in  \mathcal{C}$ must satisfy ${\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)  = 0$ ,except possibly only one box ${B}^{\prime \prime }$ ; furthermore,if ${B}^{\prime \prime }$ exists,it must have degenerated into a point! We thus report the point if it is a join result tuple (this is equivalent to checking if $\left. {{\operatorname{AGM}}_{W}\left( {B}^{\prime \prime }\right)  = 1}\right)$ . The overall running time is $\widetilde{O}\left( 1\right)$ .

我们可以利用图2中的分割算法来计算$\mathcal{Q}\left( B\right)$的结果。为此，运行split(1, B)并收集返回的盒子集合$\mathcal{C}$。我们声称，每个盒子${B}^{\prime } \in  \mathcal{C}$都必须满足${\operatorname{AGM}}_{W}\left( {B}^{\prime }\right)  = 0$，可能只有一个盒子${B}^{\prime \prime }$除外；此外，如果${B}^{\prime \prime }$存在，它必定已经退化为一个点！因此，如果该点是一个连接结果元组，我们就报告该点（这等价于检查是否$\left. {{\operatorname{AGM}}_{W}\left( {B}^{\prime \prime }\right)  = 1}\right)$）。总的运行时间为$\widetilde{O}\left( 1\right)$。

It remains to prove our claim. Recall that whenever split adds ${B}_{\text{left }}$ to $C$ ,it ensures ${\operatorname{AGM}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$ ,which is less than 1. This means ${\mathrm{{AGM}}}_{W}\left( {B}_{\text{left }}\right)  = 0$ (as mentioned,Equation (7) is either 0 or at least 1). The same applies to every ${B}_{\text{right }}$ added to $C$ . Hence,if the aforementioned box ${B}^{\prime \prime }$ indeed exists,it must have been added to $C$ as ${B}_{\text{mid }}$ . However,during all the recursive calls to split,only one ${B}_{\text{mid }}$ is added to $\mathcal{C}$ (at Line 5 of Figure 2),and this ${B}_{\text{mid }}$ must have degenerated into a point. This explains why ${B}^{\prime \prime }$ is unique and must be a point.

现在需要证明我们的断言。回想一下，每当分割算法将${B}_{\text{left }}$添加到$C$时，它会确保${\operatorname{AGM}}_{W}\left( {B}_{\text{left }}\right)  \leq  \frac{1}{2}{\operatorname{AGM}}_{W}\left( B\right)$，而该值小于1。这意味着${\mathrm{{AGM}}}_{W}\left( {B}_{\text{left }}\right)  = 0$（如前所述，等式(7)要么为0，要么至少为1）。对于添加到$C$中的每个${B}_{\text{right }}$都适用同样的情况。因此，如果上述盒子${B}^{\prime \prime }$确实存在，它必定是作为${B}_{\text{mid }}$被添加到$C$中的。然而，在对分割算法的所有递归调用中，只有一个${B}_{\text{mid }}$被添加到$\mathcal{C}$中（在图2的第5行），并且这个${B}_{\text{mid }}$必定已经退化为一个点。这就解释了为什么${B}^{\prime \prime }$是唯一的，并且必定是一个点。

---

## algorithm sample ( $W$ )

## 算法 sample ( $W$ )

/* $W$ is a fractional edge covering of $Q *$ /

/* $W$ 是 $Q *$ 的一个分数边覆盖 */

	1. $B \leftarrow  {\mathbb{N}}^{d}$ /* the attribute space */

	1. $B \leftarrow  {\mathbb{N}}^{d}$ /* 属性空间 */

		while ${\mathrm{{AGM}}}_{W}\left( B\right)  \geq  2$ do

		while ${\mathrm{{AGM}}}_{W}\left( B\right)  \geq  2$ do

	3. apply Theorem 2 to split $B$ into a set $\mathcal{C}$ of boxes

	3. 应用定理2将$B$分割成一个盒子集合$\mathcal{C}$

		take a random box ${B}_{\text{child }}$ from $C$ such that

		从$C$中随机选取一个盒子${B}_{\text{child }}$，使得

				$\Pr \left\lbrack  {{B}_{\text{child }} = {B}^{\prime }}\right\rbrack   = \frac{{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right) }{{\operatorname{AGM}}_{W}\left( B\right) }$ for each ${B}^{\prime } \in  C$ ,and

				对于每个${B}^{\prime } \in  C$，有$\Pr \left\lbrack  {{B}_{\text{child }} = {B}^{\prime }}\right\rbrack   = \frac{{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right) }{{\operatorname{AGM}}_{W}\left( B\right) }$，并且

				$\Pr \left\lbrack  {{B}_{\text{child }} = \text{ nil }}\right\rbrack   = 1 - \mathop{\sum }\limits_{{{B}^{\prime } \in  \mathcal{C}}}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right) /{\operatorname{AGM}}_{W}\left( B\right)$

				if ${B}_{\text{child }} =$ nil then return "failure"

				if ${B}_{\text{child }} =$ 为空，则返回 "失败"

				$B \leftarrow  {B}_{\text{child }}$

		apply Lemma 4 to compute $\mathcal{J}$ oin $\left( {Q\left( B\right) }\right)$

		应用引理4来计算$\mathcal{J}$ ∘ $\left( {Q\left( B\right) }\right)$

		if $\operatorname{Join}\left( {Q\left( B\right) }\right)  = \varnothing$ then return "failure"

		如果$\operatorname{Join}\left( {Q\left( B\right) }\right)  = \varnothing$ ，则返回“失败”

		toss a coin with heads probability $1/{\mathrm{{AGM}}}_{W}\left( B\right)$

		抛掷一枚正面朝上概率为$1/{\mathrm{{AGM}}}_{W}\left( B\right)$ 的硬币

	0 . if the coin comes up heads then

	0. 如果硬币正面朝上，则

				return the (only) tuple in $\mathcal{J}$ oin $\left( {Q\left( B\right) }\right)$

				返回$\mathcal{J}$ ∘ $\left( {Q\left( B\right) }\right)$ 中的（唯一）元组

	1. return "failure"

	1. 返回“失败”

					Figure 3: The proposed sampling algorithm

					图3：所提出的采样算法

---

### 4.2 The Sampling Algorithm

### 4.2 采样算法

We emphasize that the join box-tree $\mathcal{T}$ is conceptual: its size is too large ${}^{7}$ such that we cannot afford to materialize it. To extract a sample from the join result Join(Q),our algorithm will generate - on the fly - a single root-to-leaf path of $\mathcal{T}$ in $\widetilde{O}\left( 1\right)$ time and then wipe off the path from memory immediately. The path generation may not always produce a sample, but it does so with probability ${\mathrm{{OUT}/{AGM}}}_{W}\left( \mathcal{Q}\right)$ ,where $\mathrm{{OUT}} = \left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ . Thus, ${\operatorname{AGM}}_{W}\left( \mathcal{Q}\right) /$ OUT repeats will get us a sample in expectation.

我们强调连接盒树$\mathcal{T}$ 是概念性的：其规模太大（${}^{7}$ ），以至于我们无法将其物化。为了从连接结果Join(Q)中提取一个样本，我们的算法将即时生成$\mathcal{T}$ 在$\widetilde{O}\left( 1\right)$ 时间内的一条从根到叶的单路径，然后立即从内存中清除该路径。路径生成并不总是能产生一个样本，但它以概率${\mathrm{{OUT}/{AGM}}}_{W}\left( \mathcal{Q}\right)$ 产生样本，其中$\mathrm{{OUT}} = \left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ 。因此，${\operatorname{AGM}}_{W}\left( \mathcal{Q}\right) /$ 次重复操作平均能得到一个样本。

Figure 3 presents the details of our sampling algorithm. We start from the root of $\mathcal{T}$ . In general,suppose that we are currently standing at a node with box $B$ . Let us first consider $B$ to be an internal node. We obtain the set $C$ of its child nodes using the AGM split theorem in $\widetilde{O}\left( 1\right)$ time,and then descend into a child ${B}_{\text{child }}$ randomly selected from $C$ with weighted sampling. Specifically,each ${B}^{\prime } \in  C$ is chosen with probability ${\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right) /{\mathrm{{AGM}}}_{W}\left( B\right)$ . By Property 3 of Theorem 2,we have $\mathop{\sum }\limits_{{{B}^{\prime } \in  C}}{\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)  \leq  {\mathrm{{AGM}}}_{W}\left( B\right)$ . Thus,with probability $1 - \mathop{\sum }\limits_{{{B}^{\prime } \in  C}}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right) /{\operatorname{AGM}}_{W}\left( B\right)$ ,no child is selected, in which case we declare "failure" and the algorithm terminates. Because $\mathcal{C}$ has size at most ${2d} + 1 = O\left( 1\right)$ ,the weighted sampling takes $\widetilde{O}\left( 1\right)$ time,which is the cost to compute the AGM bounds of all the boxes in $C$ (Proposition 1).

图3展示了我们采样算法的细节。我们从$\mathcal{T}$ 的根节点开始。一般来说，假设我们当前位于一个带有盒子$B$ 的节点。首先，让我们考虑$B$ 是一个内部节点的情况。我们使用$\widetilde{O}\left( 1\right)$ 时间内的AGM分裂定理获得其子节点集合$C$ ，然后通过加权采样从$C$ 中随机选择一个子节点${B}_{\text{child }}$ 并向下遍历。具体来说，每个${B}^{\prime } \in  C$ 被选中的概率为${\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right) /{\mathrm{{AGM}}}_{W}\left( B\right)$ 。根据定理2的性质3，我们有$\mathop{\sum }\limits_{{{B}^{\prime } \in  C}}{\mathrm{{AGM}}}_{W}\left( {B}^{\prime }\right)  \leq  {\mathrm{{AGM}}}_{W}\left( B\right)$ 。因此，以概率$1 - \mathop{\sum }\limits_{{{B}^{\prime } \in  C}}{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right) /{\operatorname{AGM}}_{W}\left( B\right)$ ，没有子节点被选中，在这种情况下我们宣布“失败”，算法终止。因为$\mathcal{C}$ 的规模至多为${2d} + 1 = O\left( 1\right)$ ，加权采样需要$\widetilde{O}\left( 1\right)$ 时间，这也是计算$C$ 中所有盒子的AGM边界的成本（命题1）。

Next,let us look at the scenario where $B$ is a leaf. We use Lemma 4 to compute the result of the sub-join $Q\left( B\right)$ in $\widetilde{O}\left( 1\right)$ time. If Join $\left( {\mathcal{Q}\left( B\right) }\right)  = \varnothing$ ,the algorithm terminates with "failure". Otherwise, Join $\left( {\mathcal{Q}\left( B\right) }\right)$ has only a single tuple $\mathbf{u}$ (Lemma 4). We return $\mathbf{u}$ (as the join sample of the original join $Q$ ) with probability $1/{\mathrm{{AGM}}}_{W}\left( B\right)$ , but still declare "failure" with probability $1 - 1/{\mathrm{{AGM}}}_{W}\left( B\right)$ .

接下来，让我们考虑 $B$ 为叶子节点的情况。我们使用引理 4 在 $\widetilde{O}\left( 1\right)$ 时间内计算子连接 $Q\left( B\right)$ 的结果。如果连接 $\left( {\mathcal{Q}\left( B\right) }\right)  = \varnothing$ ，则算法以“失败”终止。否则，连接 $\left( {\mathcal{Q}\left( B\right) }\right)$ 只有一个元组 $\mathbf{u}$（引理 4）。我们以概率 $1/{\mathrm{{AGM}}}_{W}\left( B\right)$ 返回 $\mathbf{u}$（作为原始连接 $Q$ 的连接样本），但仍以概率 $1 - 1/{\mathrm{{AGM}}}_{W}\left( B\right)$ 宣告“失败”。

It is clear that sample runs in $\widetilde{O}\left( 1\right)$ time (remember that $\mathcal{T}$ has height $\widetilde{O}\left( 1\right)$ ; see Proposition 2). Next,we prove that if it returns a tuple,then the tuple must have been taken from $\mathcal{J}$ oin(Q)uniformly at random. Proposition 3 guarantees that every tuple $\mathbf{u} \in  \mathcal{J}$ oin(Q), which can be regarded as a point in the attribute space, is covered by the box $B$ of exactly one leaf in $\mathcal{T}$ . Consider any $\mathbf{u} \in  \mathcal{J}$ oin(Q)and its covering leaf $B$ . Let $\Pi$ be the path from the root of $\mathcal{T}$ to the node

显然，采样操作在 $\widetilde{O}\left( 1\right)$ 时间内运行（记住 $\mathcal{T}$ 的高度为 $\widetilde{O}\left( 1\right)$；见命题 2）。接下来，我们证明如果它返回一个元组，那么该元组必定是从 $\mathcal{J}$ oin(Q) 中均匀随机选取的。命题 3 保证了每个元组 $\mathbf{u} \in  \mathcal{J}$ oin(Q)（可视为属性空间中的一个点）都被 $\mathcal{T}$ 中恰好一个叶子节点的框 $B$ 所覆盖。考虑任意 $\mathbf{u} \in  \mathcal{J}$ oin(Q) 及其覆盖叶子节点 $B$。设 $\Pi$ 为从 $\mathcal{T}$ 的根节点到该节点的路径

---

<!-- Footnote -->

${}^{7}$ The number of nodes in $\mathcal{T}$ is at least $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ .

${}^{7}$ $\mathcal{T}$ 中的节点数量至少为 $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$。

<!-- Footnote -->

---

$B$ . Algorithm sample outputs $\mathbf{u}$ with probability

$B$。算法 sample 以概率输出 $\mathbf{u}$

$$
\left( {\mathop{\prod }\limits_{{\text{non-root }{B}^{\prime } \in  \Pi }}\frac{{\operatorname{AGM}}_{W}\left( {B}^{\prime }\right) }{{\operatorname{AGM}}_{W}\left( {\operatorname{parent}\left( {B}^{\prime }\right) }\right) }}\right)  \cdot  \frac{1}{{\operatorname{AGM}}_{W}\left( B\right) } \tag{9}
$$

where parent $\left( {B}^{\prime }\right)$ represents the parent of node ${B}^{\prime }$ . It is easy to see that (9) evaluates to $1/{\mathrm{{AGM}}}_{W}\left( {\mathbb{N}}^{d}\right)$ ,where ${\mathrm{{AGM}}}_{W}\left( {\mathbb{N}}^{d}\right)$ is the AGM bound of the root of $\mathcal{T}$ and equals ${\operatorname{AGM}}_{W}\left( \mathcal{Q}\right)$ . In other words, $\mathbf{u}$ is sampled with probability $1/{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right)$ . As the probability is identical for all $\mathbf{u} \in  \mathcal{J}$ oin(Q),our algorithm returns a uniformly random tuple in $\mathcal{J}$ oin(Q),provided that it does not declare failure.

其中 parent $\left( {B}^{\prime }\right)$ 表示节点 ${B}^{\prime }$ 的父节点。容易看出，(9) 的计算结果为 $1/{\mathrm{{AGM}}}_{W}\left( {\mathbb{N}}^{d}\right)$，其中 ${\mathrm{{AGM}}}_{W}\left( {\mathbb{N}}^{d}\right)$ 是 $\mathcal{T}$ 根节点的 AGM（算术 - 几何平均）界，且等于 ${\operatorname{AGM}}_{W}\left( \mathcal{Q}\right)$。换句话说，$\mathbf{u}$ 以概率 $1/{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right)$ 被采样。由于对于所有 $\mathbf{u} \in  \mathcal{J}$ oin(Q) 该概率都相同，所以只要算法不宣告失败，它就会返回 $\mathcal{J}$ oin(Q) 中的一个均匀随机元组。

We can now calculate the probability that algorithm "succeeds" (i.e., returning a tuple) as

我们现在可以计算算法“成功”（即返回一个元组）的概率为

$$
\mathop{\sum }\limits_{{\mathbf{u} \in  \operatorname{Join}\left( \mathcal{Q}\right) }}\Pr \left\lbrack  {\mathbf{u}\text{ is returned }}\right\rbrack   = \frac{\text{ OUT }}{{\operatorname{AGM}}_{W}\left( \mathcal{Q}\right) }.
$$

A standard application of Chernoff bounds shows that we can get a sample w.h.p. by repeating the algorithm $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\mathrm{{OUT}}}\right)$ times with a total cost of $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\mathrm{{OUT}}}\right)$ .

切诺夫界（Chernoff bounds）的标准应用表明，通过将算法重复 $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\mathrm{{OUT}}}\right)$ 次，我们可以以高概率获得一个样本，总成本为 $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\mathrm{{OUT}}}\right)$。

A special case occurs when $\mathrm{{OUT}} = 0$ ,which would force us into infinite repeats. The issue can be easily dealt with by stopping after $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) }\right)$ repeats and reverting to a worst-case optimal join algorithm (e.g.,Generic Join [47]) to evaluate $Q$ in full (which will confirm $\mathrm{{OUT}} = 0$ ). The overall cost is $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) }\right)$ .

当 $\mathrm{{OUT}} = 0$ 时会出现一种特殊情况，这会迫使我们进行无限次重复。可以通过在 $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) }\right)$ 次重复后停止并恢复使用最坏情况下的最优连接算法（例如，通用连接算法 [47]）来完整计算 $Q$（这将确认 $\mathrm{{OUT}} = 0$），轻松解决该问题。总体成本为 $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) }\right)$。

### 4.3 Oracles

### 4.3 预言机

It is time to clarify the oracles. In fact, the count and median oracles only require solving problems that are nowadays considered rudimentary in the data-structure area of computer science. We can, for example, maintain a set of range trees [9, 23] to implement the count oracle and a set of binary search trees to implement the median oracle. All these trees occupy $\widetilde{O}\left( \mathrm{{IN}}\right)$ space,can be built in $\widetilde{O}\left( \mathrm{{IN}}\right)$ time,and can be modified in $\widetilde{O}\left( 1\right)$ time along with each update in the relations of $Q$ . Further details are available in Appendix B. We thus have established the following theorem.

现在是时候阐明预言机了。事实上，计数预言机和中位数预言机只需要解决如今在计算机科学的数据结构领域被认为是基础的问题。例如，我们可以维护一组范围树 [9, 23] 来实现计数预言机，并维护一组二叉搜索树来实现中位数预言机。所有这些树占用 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 空间，可以在 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 时间内构建，并且可以随着 $Q$ 中关系的每次更新在 $\widetilde{O}\left( 1\right)$ 时间内进行修改。更多详细信息见附录 B。因此，我们建立了以下定理。

THEOREM 5. Consider an arbitrary (natural) join $Q$ involving a constant number of attributes. Let $W$ be an arbitrary fractional edge covering of the schema graph of $Q$ . There is an index structure of $\widetilde{O}\left( \mathrm{{IN}}\right)$ space that can be used to extract, with high probability,a sample from the join result of $Q$ in $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ time,and supports an update in the relations of $\mathcal{Q}$ in $\widetilde{O}\left( 1\right)$ time,where IN and OUT,respectively, are the input and output sizes of $Q$ ,and ${\operatorname{AGM}}_{W}\left( Q\right)$ is the ${AGM}$ bound of $Q$ under $W$ given by Lemma 1. The structure’s update and sampling algorithms are combinatorial.

定理 5. 考虑涉及常量数量属性的任意（自然）连接 $Q$。设 $W$ 是 $Q$ 的模式图的任意分数边覆盖。存在一个占用 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 空间的索引结构，该结构可以大概率地在 $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\max \{ 1,\mathrm{{OUT}}\} }\right)$ 时间内从 $Q$ 的连接结果中提取一个样本，并支持在 $\widetilde{O}\left( 1\right)$ 时间内对 $\mathcal{Q}$ 中的关系进行更新，其中 IN 和 OUT 分别是 $Q$ 的输入和输出大小，${\operatorname{AGM}}_{W}\left( Q\right)$ 是引理 1 给出的 $Q$ 在 $W$ 下的 ${AGM}$ 界。该结构的更新和采样算法是组合式的。

By choosing an optimal fractional edge covering $W$ ,the sample time in the theorem is bounded by the complexity in (2).

通过选择最优的分数边覆盖 $W$，定理中的采样时间受 (2) 式中的复杂度限制。

## 5 HARDNESS OF JOIN SAMPLING AND OUTPUT-SENSITIVE JOIN ALGORITHMS

## 5 连接采样的难度和输出敏感的连接算法

This section will provide evidence that Theorem 5 can no longer be improved significantly for all joins. For this purpose, we will argue that no combinatorial structure can give a "yes" answer to the join sampling question unless the combinatorial $k$ -clique hypothesis is wrong. To do so, we will take a de-tour to bridge join sampling with output-sensitive join evaluation.

本节将提供证据表明，对于所有连接，定理 5 无法再显著改进。为此，我们将论证，除非组合 $k$ -团假设不成立，否则没有组合结构能够对连接采样问题给出“是”的答案。为此，我们将绕个弯，将连接采样与输出敏感的连接评估联系起来。

Let $\mathcal{A}$ be a combinatorial algorithm for computing $\mathcal{J}$ oin(Q). Recall from Section 1 that $\mathcal{A}$ is $\epsilon$ -output sensitive — where $0 < \epsilon  < 1/2$ — if it runs in time $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ as long as $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ . The lemma below is proved in Appendix C.

设 $\mathcal{A}$ 是用于计算 $\mathcal{J}$ oin(Q) 的组合算法。回顾第 1 节，如果 $\mathcal{A}$ 在 $\mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ 时能在 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间内运行，那么它是 $\epsilon$ -输出敏感的，其中 $0 < \epsilon  < 1/2$。下面的引理在附录 C 中证明。

LEMMA 6. If a combinatorial structure can answer "yes" to the join sampling question for a constant $\epsilon  \in  \left( {0,1/2}\right)$ ,there is a combinatorial $\epsilon$ -output sensitive algorithm for join computation.

引理 6. 如果一个组合结构能够对常量 $\epsilon  \in  \left( {0,1/2}\right)$ 的连接采样问题给出“是”的答案，那么存在一个用于连接计算的组合 $\epsilon$ -输出敏感算法。

An $\epsilon$ -output sensitive algorithm $\mathcal{A}$ does not need to be fast when OUT $> {\mathrm{{IN}}}^{\epsilon }$ . However,when OUT $> {\mathrm{{IN}}}^{\epsilon }$ ,the sampling algorithm in Theorem 5 runs in $\widetilde{O}\left( \frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}}\right)  = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time. In Appendix D,we combine the two algorithms to prove:

一种 $\epsilon$ -输出敏感算法 $\mathcal{A}$ 在输出（OUT） $> {\mathrm{{IN}}}^{\epsilon }$ 时不需要很快。然而，当输出（OUT） $> {\mathrm{{IN}}}^{\epsilon }$ 时，定理 5 中的采样算法的运行时间为 $\widetilde{O}\left( \frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}}\right)  = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$。在附录 D 中，我们将这两种算法结合起来证明：

LEMMA 7. Given a combinatorial $\epsilon$ -output sensitive algorithm, we can design a combinatorial algorithm to detect whether $\operatorname{Join}\left( \mathcal{Q}\right)$ is empty in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time w.h.p.,regardless of the value OUT.

引理 7. 给定一个组合的 $\epsilon$ -输出敏感算法，我们可以设计一个组合算法，以高概率（w.h.p.）在 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 时间内检测 $\operatorname{Join}\left( \mathcal{Q}\right)$ 是否为空，而无需考虑输出（OUT）的值。

As shown in Appendix F, however, the emptiness-detection algorithm in Lemma 7 breaks the combinatorial $k$ -clique hypothesis. In summary,finding a combinatorial $\epsilon$ -output sensitive algorithm is at least as hard as breaking the hypothesis. Lemma 6 indicates that it can be only harder to improve our result in Theorem 5 by a polynomial factor even when $\mathrm{{OUT}} \ll  {\mathrm{{IN}}}^{{\rho }^{ * }}$ .

然而，如附录 F 所示，引理 7 中的空集检测算法打破了组合的 $k$ -团假设（$k$ -clique hypothesis）。总之，找到一个组合的 $\epsilon$ -输出敏感算法至少和打破该假设一样困难。引理 6 表明，即使在 $\mathrm{{OUT}} \ll  {\mathrm{{IN}}}^{{\rho }^{ * }}$ 的情况下，要将我们定理 5 的结果提高一个多项式因子也只会更难。

## 6 APPLICATIONS

## 6 应用

Our join sampling structure can be utilized to tackle many problems, a partial list of which is presented below.

我们的连接采样结构可用于解决许多问题，下面列出了其中一部分。

- (Join size estimation) It is standard [21] to use join sampling to estimate $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ up to a relative error $\lambda$ . Our structure in Theorem 5 can be applied to produce an estimate in $\widetilde{O}\left( {\frac{1}{{\lambda }^{2}}\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\max \{ 1,\mathrm{{OUT}}\} }}\right)$ time w.h.p.,improving the state of the art in [21] by an $O\left( \mathrm{{IN}}\right)$ factor. We can also support each update in $\widetilde{O}\left( 1\right)$ time.

- （连接大小估计）使用连接采样将 $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ 估计到相对误差 $\lambda$ 是标准做法 [21]。我们定理 5 中的结构可以以高概率（w.h.p.）在 $\widetilde{O}\left( {\frac{1}{{\lambda }^{2}}\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\max \{ 1,\mathrm{{OUT}}\} }}\right)$ 时间内给出一个估计，比 [21] 中的现有技术提高了 $O\left( \mathrm{{IN}}\right)$ 倍。我们还可以在 $\widetilde{O}\left( 1\right)$ 时间内支持每次更新。

- (Subgraph sampling) Let $G \mathrel{\text{:=}} \left( {V,E}\right)$ be a simple undirected graph. Given a pattern graph $Q$ of a constant size,a query samples an occurrence ${}^{8}$ of $Q$ in $G$ uniformly at random. We obtain a structure of $\widetilde{O}\left( \left| E\right| \right)$ space that answers a query in $\widetilde{O}\left( {{\left| E\right| }^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OCC}}\} }\right)$ time w.h.p.,where ${\rho }^{ * }$ is the fractional edge covering number of $Q$ and OCC is the number of occurrences of $Q$ in $G$ . The structure supports an edge insertion and deletion in $\widetilde{O}\left( 1\right)$ time. Previously,a structure matching our guarantees was given in [29]. Our solution is drastically different and actually settles a problem we call "join sampling with predicates", which captures subgraph sampling as a (simple) special case. Details are available in Appendix E.

- （子图采样）设 $G \mathrel{\text{:=}} \left( {V,E}\right)$ 是一个简单无向图。给定一个固定大小的模式图 $Q$，一个查询会从 $G$ 中均匀随机地采样 $Q$ 的一个出现 ${}^{8}$。我们得到一个空间复杂度为 $\widetilde{O}\left( \left| E\right| \right)$ 的结构，它可以以高概率（w.h.p.）在 $\widetilde{O}\left( {{\left| E\right| }^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OCC}}\} }\right)$ 时间内回答一个查询，其中 ${\rho }^{ * }$ 是 $Q$ 的分数边覆盖数，OCC 是 $Q$ 在 $G$ 中出现的次数。该结构可以在 $\widetilde{O}\left( 1\right)$ 时间内支持边的插入和删除操作。此前，[29] 中给出了一个满足我们这些保证的结构。我们的解决方案有很大不同，实际上解决了一个我们称之为“带谓词的连接采样”的问题，子图采样是该问题的一个（简单）特例。详细内容见附录 E。

- (Joins with random enumeration) Carmeli et al. [15] proposed a variant of the join computation problem, where the objective is to design an algorithm that, after an initial pre-processing, can (i) produce a random permutation of the tuples in $\mathcal{J}$ oin(Q), and (ii) do so with a small delay $\Delta$ (the maximum time gap between the reporting of two consecutive tuples in the permutation). We obtain the first algorithm that, after an initial $\widetilde{O}$ (IN)-time preprocessing,produces the whole permutation in $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ time (i.e.,worst-case optimal up to a polyloga-rithmic factor) with a delay $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ w.h.p.. Details are available in Appendix G.

- （随机枚举连接）卡梅利（Carmeli）等人 [15] 提出了连接计算问题的一个变体，其目标是设计一种算法，在初始预处理之后，该算法能够 (i) 生成 $\mathcal{J}$ oin(Q) 中元组的随机排列，并且 (ii) 以较小的延迟 $\Delta$ 完成此操作（排列中两个连续元组报告之间的最大时间间隔）。我们得到了第一个算法，该算法在经过初始 $\widetilde{O}$ (IN) 时间的预处理后，以 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 时间生成整个排列（即，在多项式对数因子范围内达到最坏情况下的最优），且延迟为 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 的概率很高。详细信息见附录 G。

---

<!-- Footnote -->

${}^{8}$ An occurrence is a subgraph of $G$ isomorphic to $Q$ .

${}^{8}$ 一个出现是与 $Q$ 同构的 $G$ 的子图。

<!-- Footnote -->

---

- (Join Union Sampling) Let ${Q}_{1},{Q}_{2},\ldots ,{Q}_{k}$ be joins with $\operatorname{var}\left( {\mathcal{Q}}_{1}\right)  = \operatorname{var}\left( {\mathcal{Q}}_{2}\right)  = \ldots  = \operatorname{var}\left( {\mathcal{Q}}_{k}\right)$ ,where $k \geq  2$ is a constant. We want to sample from $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {Q}_{i}\right)$ uniformly at random. Let IN be the total number of tuples in the input relations of ${\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{k}$ ,OUT $\mathrel{\text{:=}} \left| {\mathop{\bigcup }\limits_{{i = 1}}^{k}\operatorname{Join}\left( {\mathcal{Q}}_{i}\right) }\right|$ ,and ${\rho }^{ * }$ be the maximum fractional edge covering number of the schema graphs of ${\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{k}$ . We obtain a structure of $\widetilde{O}\left( \mathrm{{IN}}\right)$ space that can extract a uniform sample in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ time w.h.p. and support an update in any input relation in $\widetilde{O}\left( 1\right)$ time. Details are available in Appendix H.

- （连接并采样）设 ${Q}_{1},{Q}_{2},\ldots ,{Q}_{k}$ 是与 $\operatorname{var}\left( {\mathcal{Q}}_{1}\right)  = \operatorname{var}\left( {\mathcal{Q}}_{2}\right)  = \ldots  = \operatorname{var}\left( {\mathcal{Q}}_{k}\right)$ 的连接，其中 $k \geq  2$ 是一个常数。我们希望从 $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {Q}_{i}\right)$ 中均匀随机采样。设 IN 是 ${\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{k}$ 的输入关系中的元组总数，OUT 为 $\mathrel{\text{:=}} \left| {\mathop{\bigcup }\limits_{{i = 1}}^{k}\operatorname{Join}\left( {\mathcal{Q}}_{i}\right) }\right|$，${\rho }^{ * }$ 是 ${\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{k}$ 的模式图的最大分数边覆盖数。我们得到一个 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 空间的结构，该结构可以以 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 时间（概率很高）提取一个均匀样本，并支持在 $\widetilde{O}\left( 1\right)$ 时间内对任何输入关系进行更新。详细信息见附录 H。

## 7 POST-ACCEPTANCE REMARKS

## 7 录用后备注

In another paper [37] accepted to PODS'23, Kim et al. also developed a join sampling algorithm achieving performance guarantees similar to ours. Their algorithm is elegant and approaches the problem from a perspective different from our work. In [37], Kim et al. further discussed (i) some scenarios where better running time was possible and (ii) how to estimate the join result size.

在另一篇被 PODS'23 录用的论文 [37] 中，金（Kim）等人也开发了一种连接采样算法，其性能保证与我们的类似。他们的算法很优雅，并且从与我们的工作不同的视角来处理这个问题。在 [37] 中，金等人进一步讨论了 (i) 一些可能实现更好运行时间的场景，以及 (ii) 如何估计连接结果的大小。

## ACKNOWLEDGEMENTS

## 致谢

This work was supported in part by GRF projects 14207820, 14203421, and 14222822 from HKRGC.

这项工作部分得到了香港研究资助局（HKRGC）的研资局项目（GRF）14207820、14203421和14222822的支持。

## REFERENCES

## 参考文献

[1] Amir Abboud, Arturs Backurs, and Virginia Vassilevska Williams. If the current clique algorithms are optimal, so is valiant's parser. In Proceedings of Annual IEEE Symposium on Foundations of Computer Science (FOCS), pages 98-117, 2015.

[2] Amir Abboud, Loukas Georgiadis, Giuseppe F. Italiano, Robert Krauthgamer, Nikos Parotsidis, Ohad Trabelsi, Przemyslaw Uznanski, and Daniel Wolleb-Graf. Faster algorithms for all-pairs bounded min-cuts. In Proceedings of International Colloquium on Automata, Languages and Programming (ICALP), pages 7:1-7:15, 2019.

[3] Swarup Acharya, Phillip B. Gibbons, Viswanath Poosala, and Sridhar Ramaswamy. Join synopses for approximate query answering. In Proceedings of ACM Management of Data (SIGMOD), pages 275-286, 1999.

[4] Pankaj K. Agarwal. Range searching. In Handbook of Discrete and Computational Geometry, 2nd Ed, pages 809-837. Chapman and Hall/CRC, 2004.

[5] Sameer Agarwal, Barzan Mozafari, Aurojit Panda, Henry Milner, Samuel Madden, and Ion Stoica. BlinkDB: queries with bounded errors and bounded response times on very large data. In Eurosys, pages 29-42, 2013.

[6] Kaleb Alway, Eric Blais, and Semih Salihoglu. Box covers and domain orderings for beyond worst-case join processing. In Proceedings of International Conference on Database Theory (ICDT), pages 3:1-3:23, 2021.

[7] Rasmus Resen Amossen and Rasmus Pagh. Faster join-projects and sparse matrix multiplications. In Ronald Fagin, editor, Proceedings of International Conference on Database Theory (ICDT), volume 361, pages 121-126, 2009.

[8] Albert Atserias, Martin Grohe, and Daniel Marx. Size bounds and query plans for relational joins. SIAM Journal on Computing, 42(4):1737-1767, 2013.

[9] Jon Louis Bentley. Decomposable searching problems. Information Processing Letters (IPL), 8(5):244-251, 1979.

[10] Thiago Bergamaschi, Monika Henzinger, Maximilian Probst Gutenberg, Virginia Vassilevska Williams, and Nicole Wein. New techniques and fine-grained hardness for dynamic near-additive spanners. In Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1836-1855, 2021.

[11] Karl Bringmann, Nick Fischer, and Marvin Kunnemann. A fine-grained analogue of schaefer’s theorem in P: dichotomy of exists $\mathrm{k}$ -forall-quantified first-order graph properties. In Computational Complexity Conference, pages 31:1-31:27, 2019.

[12] Karl Bringmann, Allan Gronlund, and Kasper Green Larsen. A dichotomy for regular expression membership testing. In Proceedings of Annual IEEE Symposium on Foundations of Computer Science (FOCS), pages 307-318, 2017.

[13] Karl Bringmann and Philip Wellnitz. Clique-based lower bounds for parsing tree-adjoining grammars. In Proceedings of Annual Symposium on Combinatorial Pattern Matching (CPM), pages 12:1-12:14, 2017.

[14] Nofar Carmeli, Nikolaos Tziavelis, Wolfgang Gatterbauer, Benny Kimelfeld, and Mirek Riedewald. Tractable orders for direct access to ranked answers of conjunctive queries. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 325-341, 2021.

[15] Nofar Carmeli, Shai Zeevi, Christoph Berkholz, Benny Kimelfeld, and Nicole Schweikardt. Answering (unions of) conjunctive queries using random access and random-order enumeration. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 393-409, 2020.

[16] Timothy M. Chan. A (slightly) faster algorithm for klee's measure problem. Comput. Geom., 43(3):243-250, 2010.

[17] Yi-Jun Chang. Hardness of RNA folding problem with four symbols. Theor: Comput. Sci., 757:11-26, 2019.

[18] Surajit Chaudhuri, Rajeev Motwani, and Vivek R. Narasayya. On random sampling over joins. In Proceedings of ACM Management of Data (SIGMOD), pages 263- 274, 1999.

[19] Chandra Chekuri and Anand Rajaraman. Conjunctive query containment revisited. Theoretical Computer Science, 239(2):211-229, 2000.

[20] Yu Chen and Ke Yi. Two-level sampling for join size estimation. In Proceedings of ACM Management of Data (SIGMOD), pages 759-774, 2017.

[21] Yu Chen and Ke Yi. Random sampling and size estimation over cyclic joins. In Proceedings of International Conference on Database Theory (ICDT), pages 7:1-7:18, 2020.

[22] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. Introduction to Algorithms, Second Edition. The MIT Press, 2001.

[23] Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars. Computational Geometry: Algorithms and Applications. Springer-Verlag, 3rd edition, 2008.

[24] Rina Dechter and Judea Pearl. Tree-clustering schemes for constraint-processing. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 150- 154, 1988.

[25] Shaleen Deep, Xiao Hu, and Paraschos Koutris. Enumeration algorithms for conjunctive queries with projection. In Proceedings of International Conference on Database Theory (ICDT), volume 186, pages 14:1-14:17.

[26] Shaleen Deep, Xiao Hu, and Paraschos Koutris. Fast join project query evaluation using matrix multiplication. In Proceedings of ACM Management of Data (SIGMOD), pages 1213-1223, 2020.

[27] Shaleen Deep and Paraschos Koutris. Compressed representations of conjunctive query results. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 307-322, 2018.

[28] Friedrich Eisenbrand and Fabrizio Grandoni. On the complexity of fixed parameter clique and dominating set. Theoretical Computer Science, 326(1-3):57-67, 2004.

[29] Hendrik Fichtenberger, Mingze Gao, and Pan Peng. Sampling arbitrary subgraphs exactly uniformly in sublinear time. In Proceedings of International Colloquium on Automata, Languages and Programming (ICALP), pages 45:1-45:13, 2020.

[30] Ehud Friedgut. Hypergraphs, entropy, and inequalities. Am. Math. Mon., 111(9):749-760, 2004.

[31] Georg Gottlob, Nicola Leone, and Francesco Scarcello. Robbers, marshals, and guards: game theoretic and logical characterizations of hypertree width. Journal of Computer and System Sciences (JCSS), 66(4):775-808, 2003.

[32] Etienne Grandjean and Louis Jachiet. Which arithmetic operations can be performed in constant time in the RAM model with addition? CoRR, abs/2206.13851, 2022.

[33] Peter J. Haas and Joseph M. Hellerstein. Ripple joins for online aggregation. In Proceedings of ACM Management of Data (SIGMOD), pages 287-298, 1999.

[34] Kathrin Hanauer, Monika Henzinger, and Qi Cheng Hua. Fully dynamic four-vertex subgraph counting. In Symposium on Algorithmic Foundations of Dynamic Networks (SAND), volume 221, pages 18:1-18:17, 2022.

[35] Ce Jin and Yinzhan Xu. Tight dynamic problem lower bounds from generalized BMM and onv. In Proceedings of ACM Symposium on Theory of Computing (STOC), 2022.

[36] Mahmoud Abo Khamis, Hung Q. Ngo, Christopher Re, and Atri Rudra. Joins via

geometric resolutions: Worst case and beyond. ACM Transactions on Database Systems (TODS), 41(4):22:1-22:45, 2016.

[37] Kyoungmin Kim, Jaehyun Ha, George Fletcher, and Wook-Shin Han. Guaranteeing the $\widetilde{O}$ (AGM/OUT) runtime for uniform sampling and size estimation over joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), 2023.

[38] Kyoungmin Kim, Hyeonji Kim, George Fletcher, and Wook-Shin Han. Combining sampling and synopses with worst-case optimal runtime and quality guarantees for graph pattern cardinality estimation. In Proceedings of ACM Management of Data (SIGMOD), pages 964-976, 2021.

[39] Feifei Li, Bin Wu, Ke Yi, and Zhuoyue Zhao. Wander join: Online aggregation via random walks. In Proceedings of ACM Management of Data (SIGMOD), pages 615-629, 2016.

[40] Jason Li. Faster minimum k-cut of a simple graph. In Proceedings of Annual IEEE Symposium on Foundations of Computer Science (FOCS), pages 1056-1077, 2019.

[41] Andrea Lincoln, Virginia Vassilevska Williams, and R. Ryan Williams. Tight hardness for shortest cycles and paths in sparse graphs. In Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1236-1252, 2018.

[42] Gonzalo Navarro, Juan L. Reutter, and Javiel Rojas-Ledesma. Optimal joins using compact data structures. In Proceedings of International Conference on Database Theory (ICDT), volume 155, pages 21:1-21:21, 2020.

[43] Jaroslav Nesetril and Svatopluk Poljak. On the complexity of the subgraph problem. Commentationes Mathematicae Universitatis Carolinae, 26(2):415-419, 1985.

[44] Hung Q. Ngo, Dung T. Nguyen, Christopher Re, and Atri Rudra. Beyond worst-case analysis for joins with minesweeper. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 234-245, 2014.

[45] Hung Q. Ngo, Ely Porat, Christopher Ré, and Atri Rudra. Worst-Case Optimal Join Algorithms: [Extended Abstract]. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 37-48, 2012.

[46] Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. Worst-case optimal join algorithms. Journal of the ACM (JACM), 65(3):16:1-16:40, 2018.

[47] Hung Q. Ngo, Christopher Re, and Atri Rudra. Skew strikes back: new developments in the theory of join algorithms. SIGMOD Rec., 42(4):5-16, 2013.

[48] Supriya Nirkhiwale, Alin Dobra, and Christopher M. Jermaine. A sampling algebra for aggregate estimation. Proceedings of the VLDB Endowment (PVLDB), 6(14):1798-1809, 2013.

[49] Dan Olteanu and Jakub Zavodny. Size bounds for factorised representations of query results. ACM Transactions on Database Systems (TODS), 40(1):2:1-2:44, 2015.

[50] Rodrygo L. T. Santos, Craig MacDonald, and Iadh Ounis. Search result diversification. Found. Trends Inf. Retr., 9(1):1-90, 2015.

[51] Ali Mohammadi Shanghooshabad, Meghdad Kurmanji, Qingzhi Ma, Michael Shekelyan, Mehrdad Almasi, and Peter Triantafillou. Pgmjoins: Random join sampling with graphical models. In Proceedings of ACM Management of Data (SIGMOD), pages 1610-1622, 2021.

[52] Yufei Tao and Ke Yi. Intersection joins under updates. Journal of Computer and System Sciences (JCSS), 124:41-64, 2022.

[53] Virginia Vassilevska. Efficient algorithms for clique problems. Information Processing Letters (IPL), 109(4):254-257, 2009.

[54] Todd L. Veldhuizen. Triejoin: A simple, worst-case optimal join algorithm. In Proceedings of International Conference on Database Theory (ICDT), pages 96-106, 2014.

[55] David Vengerov, Andre Cavalheiro Menck, Mohamed Zaït, and Sunil Chakkap-

pen. Join size estimation subject to filter conditions. Proceedings of the VLDB Endowment (PVLDB), 8(12):1530-1541, 2015.

[56] Mihalis Yannakakis. Algorithms for acyclic database schemes. In Very Large Data Bases, 7th International Conference, September 9-11, 1981, Cannes, France, Proceedings, pages 82-94, 1981.

[57] Feng Yu, Wen-Chi Hou, Cheng Luo, Dunren Che, and Mengxia Zhu. CS2: a new database synopsis for query estimation. In Proceedings of ACM Management of Data (SIGMOD), pages 469-480, 2013.

[58] Zhuoyue Zhao, Robert Christensen, Feifei Li, Xiao Hu, and Ke Yi. Random sampling over joins revisited. In Proceedings of ACM Management of Data (SIGMOD), pages 1525-1539, 2018.

[59] Zhuoyue Zhao, Feifei Li, and Yuxi Liu. Efficient join synopsis maintenance for data warehouse. In Proceedings of ACM Management of Data (SIGMOD), pages 2027-2042, 2020.

## APPENDIX

## 附录

## A PROOF OF LEMMA 3

## 引理3的证明

We first review Friedgut's inequality (sometimes called the generalized Höder’s inequality). Fix some integers $p$ and $q$ at least 1 . Let $\left\{  {{a}_{i,j} \mid  i \in  \left\lbrack  {1,p}\right\rbrack  ,j \in  \left\lbrack  {1,q}\right\rbrack  }\right\}$ be a set of non-negative real values. Also,let $\left\{  {{b}_{i} \mid  i \in  \left\lbrack  {1,q}\right\rbrack  }\right\}$ be another set of non-negative real values satisfying $\mathop{\sum }\limits_{{k = 1}}^{q}{b}_{k} \geq  1$ . Assuming ${0}^{0} = 0$ ,Friedgut’s inequality [30] states

我们首先回顾弗里德古特不等式（有时也称为广义赫尔德不等式）。固定一些至少为1的整数$p$和$q$。设$\left\{  {{a}_{i,j} \mid  i \in  \left\lbrack  {1,p}\right\rbrack  ,j \in  \left\lbrack  {1,q}\right\rbrack  }\right\}$是一组非负实数值。此外，设$\left\{  {{b}_{i} \mid  i \in  \left\lbrack  {1,q}\right\rbrack  }\right\}$是另一组满足$\mathop{\sum }\limits_{{k = 1}}^{q}{b}_{k} \geq  1$的非负实数值。假设${0}^{0} = 0$，弗里德古特不等式[30]表明

$$
\mathop{\sum }\limits_{{i = 1}}^{p}\mathop{\prod }\limits_{{j = 1}}^{q}{a}_{i,j}^{{b}_{j}} \leq  \mathop{\prod }\limits_{{j = 1}}^{q}{\left( \mathop{\sum }\limits_{{i = 1}}^{p}{a}_{i,j}\right) }^{{b}_{j}}. \tag{10}
$$

Returning to the context of Lemma 3, recall that we have already fixed an integer $i \in  \left\lbrack  {1,d}\right\rbrack$ : this is the " $i$ " used to create ${B}_{1},{B}_{2},\ldots ,{B}_{s}$ in (8). Define ${\mathcal{E}}_{i} = \left\{  {e \in  \mathcal{E} \mid  {X}_{i} \in  e}\right\}$ ,the set of edges in the schema graph of $Q$ that cover attribute ${X}_{i}$ . For each $j \in  \left\lbrack  {1,s}\right\rbrack$ ,we have

回到引理3的上下文，回顾我们已经固定了一个整数$i \in  \left\lbrack  {1,d}\right\rbrack$：这就是在(8)中用于创建${B}_{1},{B}_{2},\ldots ,{B}_{s}$的“$i$”。定义${\mathcal{E}}_{i} = \left\{  {e \in  \mathcal{E} \mid  {X}_{i} \in  e}\right\}$，即$Q$的模式图中覆盖属性${X}_{i}$的边的集合。对于每个$j \in  \left\lbrack  {1,s}\right\rbrack$，我们有

$$
{\operatorname{AGM}}_{W}\left( {B}_{j}\right)  = \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {R}_{e}\left( {B}_{j}\right) \right| }^{W\left( e\right) }\;\text{ (see (7)) }
$$

$$
 = \mathop{\prod }\limits_{{e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( {B}_{j}\right) \right| }^{W\left( e\right) } \cdot  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( {B}_{j}\right) \right| }^{W\left( e\right) }
$$

$$
 = \mathop{\prod }\limits_{{e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) } \cdot  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( {B}_{j}\right) \right| }^{W\left( e\right) }
$$

where the last equality used the fact ${R}_{e}\left( {B}_{j}\right)  = {R}_{e}\left( B\right)$ for $e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}$ , which holds because $B$ and ${B}_{j}$ have the same projections on all attributes other than ${X}_{i}$ ,but ${X}_{i} \notin  e$ . We can now derive

其中最后一个等式使用了对于$e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}$有${R}_{e}\left( {B}_{j}\right)  = {R}_{e}\left( B\right)$这一事实，这是因为$B$和${B}_{j}$在除${X}_{i}$之外的所有属性上的投影相同，但${X}_{i} \notin  e$。现在我们可以推导

$$
\mathop{\sum }\limits_{{j = 1}}^{s}{\operatorname{AGM}}_{W}\left( {B}_{j}\right) 
$$

$$
 = \mathop{\sum }\limits_{{j = 1}}^{s}\left( {\mathop{\prod }\limits_{{e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) } \cdot  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( {B}_{j}\right) \right| }^{W\left( e\right) }}\right) 
$$

$$
 = \mathop{\prod }\limits_{{e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) } \cdot  \mathop{\sum }\limits_{{j = 1}}^{s}\left( {\mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( {B}_{j}\right) \right| }^{W\left( e\right) }}\right) 
$$

$$
 \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) } \cdot  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{i}}}{\left( \mathop{\sum }\limits_{{j = 1}}^{s}\left| {R}_{e}\left( {B}_{j}\right) \right| \right) }^{W\left( e\right) }
$$

$$
\text{(applying (10),noticing that}\mathop{\sum }\limits_{{e \in  {\mathcal{E}}_{j}}}W\left( e\right)  \geq  1\text{)}
$$

$$
 = \mathop{\prod }\limits_{{e \in  \mathcal{E} \smallsetminus  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) } \cdot  \mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{i}}}{\left| {R}_{e}\left( B\right) \right| }^{W\left( e\right) }
$$

$$
\text{(because}\mathop{\bigcup }\limits_{{j = 1}}^{s}{B}_{j} = B\text{and}{B}_{1},\ldots ,{B}_{s}\text{are disjoint)}
$$

$$
 = {\operatorname{AGM}}_{W}\left( B\right) \text{.}
$$

## B ORACLE IMPLEMENTATION

## B 预言机实现

The count oracle essentially deals with a problem known as orthogonal range counting. In that problem,the input is a set $P$ of $n$ points in $d$ -dimensional space ${\mathbb{R}}^{d}$ for some constant $d$ . Given an axis-parallel rectangle $q$ in ${\mathbb{R}}^{d}$ ,a query returns $\left| {P \cap  q}\right|$ ,namely,the number of points in $P$ that are covered by $q$ . The goal is to store $P$ in a data structure to answer queries efficiently. We refer the reader to [4] for a survey on the known data structures solving this problem. Among them is the range tree $\left\lbrack  {9,{23}}\right\rbrack$ ,which consumes $O\left( {n{\log }^{d - 1}n}\right)$ space, answers a query in $O\left( {{\log }^{d}n}\right)$ time,and supports a point insertion and deletion in $P$ using $O\left( {{\log }^{d}n}\right)$ time. It serves as a count oracle meeting our requirements.

计数预言机本质上处理的是一个被称为正交范围计数的问题。在该问题中，输入是对于某个常数$d$，在$d$维空间${\mathbb{R}}^{d}$中的一组$n$个点$P$。给定${\mathbb{R}}^{d}$中的一个轴平行矩形$q$，一个查询返回$\left| {P \cap  q}\right|$，即$P$中被$q$覆盖的点的数量。目标是将$P$存储在一个数据结构中，以便高效地回答查询。我们建议读者参考[4]以了解解决此问题的已知数据结构的综述。其中包括范围树$\left\lbrack  {9,{23}}\right\rbrack$，它占用$O\left( {n{\log }^{d - 1}n}\right)$的空间，在$O\left( {{\log }^{d}n}\right)$时间内回答一个查询，并使用$O\left( {{\log }^{d}n}\right)$时间在$P$中支持点的插入和删除。它可以作为满足我们要求的计数预言机。

Before implementing the median oracle, let us look at an alternative problem first. The input is a set $S$ of $n$ real values. Given an interval $q$ in $\mathbb{R}$ and an integer $k \geq  1$ ,a query returns the $k$ -th smallest integer in $S \cap  q$ (or returns nothing in the special case where $k > \left| {S \cap  q}\right|$ ). The goal is to store $S$ in a data structure to answer queries efficiently. We can create a binary search tree (BST) on $S$ and keep,at each node $v$ in the tree,the number of descendant nodes of $v$ . The tree occupies $O\left( n\right)$ space,answers a query in $O\left( {\log n}\right)$ time (see Chapter 14 "Augmenting Data Structures" of [22]), and supports an insertion or deletion in $S$ using $O\left( {\log n}\right)$ time. The same structure can also find the size of $S \cap  q$ in $O\left( {\log n}\right)$ time (see the above chapter of [22] again). It thus follows that we can report the median value in $S \cap  q$ in $O\left( {\log n}\right)$ time.

在实现中位数预言机之前，让我们先看一个替代问题。输入是一个包含 $n$ 个实数值的集合 $S$。给定 $\mathbb{R}$ 中的一个区间 $q$ 和一个整数 $k \geq  1$，一次查询会返回 $S \cap  q$ 中第 $k$ 小的整数（在 $k > \left| {S \cap  q}\right|$ 的特殊情况下则不返回任何值）。目标是将 $S$ 存储在一个数据结构中，以便高效地回答查询。我们可以在 $S$ 上创建一棵二叉搜索树（BST），并在树中的每个节点 $v$ 处记录 $v$ 的后代节点数量。该树占用 $O\left( n\right)$ 的空间，回答一次查询的时间复杂度为 $O\left( {\log n}\right)$（见文献 [22] 的第 14 章“扩充数据结构”），并且使用 $O\left( {\log n}\right)$ 的时间支持在 $S$ 中进行插入或删除操作。同样的结构也能在 $O\left( {\log n}\right)$ 的时间内找出 $S \cap  q$ 的大小（再次参见文献 [22] 的上述章节）。因此，我们可以在 $O\left( {\log n}\right)$ 的时间内报告 $S \cap  q$ 的中位数。

To implement a median oracle,for every attribute $X \in  \operatorname{var}\left( Q\right)$ , we maintain the aforementioned (slightly-augmented) BST on the set $S$ of $X$ -values that appear in at least one relation in $Q$ . The structure occupies $\widetilde{O}\left( \mathrm{{IN}}\right)$ space and,given a box $B$ ,finds the median of actdom(X,B)— which is the median of the values of $S$ covered by the interval $B\left( X\right)$ - in $O\left( {\log \mathrm{{IN}}}\right)$ time. It is straightforward to maintain the tree in $O\left( {\log \mathrm{{IN}}}\right)$ time per update.

为了实现中位数预言机，对于每个属性 $X \in  \operatorname{var}\left( Q\right)$，我们在集合 $S$ 上维护上述（略有扩充的）二叉搜索树，该集合包含在 $Q$ 中至少一个关系里出现的 $X$ 值。该结构占用 $\widetilde{O}\left( \mathrm{{IN}}\right)$ 的空间，并且给定一个盒子 $B$，能在 $O\left( {\log \mathrm{{IN}}}\right)$ 的时间内找出 actdom(X,B) 的中位数，即由区间 $B\left( X\right)$ 覆盖的 $S$ 值的中位数。每次更新时，在 $O\left( {\log \mathrm{{IN}}}\right)$ 的时间内维护这棵树是很直接的。

## C PROOF OF LEMMA 6

## C 引理 6 的证明

Suppose that,given a join $Q$ ,we can build a combinatorial structure $\Upsilon$ in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time that can extract a uniform sample from Join(Q)in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)$ time when $\mathrm{{OUT}} \in  \left\lbrack  {1,{\mathrm{{IN}}}^{\epsilon }}\right\rbrack$ . Next,we will show how to use $\Upsilon$ to compute $\mathcal{J}$ oin(Q)in $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time when OUT $\leq  {\mathrm{{IN}}}^{\epsilon }$ ,thereby establishing the lemma.

假设给定一个连接 $Q$，我们可以在 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 的时间内构建一个组合结构 $\Upsilon$，当 $\mathrm{{OUT}} \in  \left\lbrack  {1,{\mathrm{{IN}}}^{\epsilon }}\right\rbrack$ 时，该结构能在 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)$ 的时间内从 Join(Q) 中提取一个均匀样本。接下来，我们将展示当 OUT $\leq  {\mathrm{{IN}}}^{\epsilon }$ 时，如何使用 $\Upsilon$ 在 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 的时间内计算 $\mathcal{J}$oin(Q)，从而证明该引理。

Let us first assume that we know, by magic, the value OUT. If OUT $= 0$ ,Join(Q)is empty and there is nothing to do. Otherwise,we deploy $\Upsilon$ to extract $s \mathrel{\text{:=}} c \cdot  \mathrm{{OUT}} \cdot  \ln \mathrm{{IN}}$ samples from $\operatorname{Join}\left( \mathcal{Q}\right)$ where $c$ is a sufficiently large constant. W.h.p.,every tuple $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ must have been sampled at least once. Indeed, the probability for $\mathbf{u}$ to have been missed by all those $s$ samples is ${\left( 1 - \frac{1}{\mathrm{{OUT}}}\right) }^{s}$ ,which is at most ${e}^{-s/\mathrm{{OUT}}} = 1/{\mathrm{{IN}}}^{c}$ . It thus holds with probability at least $1 - \mathrm{{OUT}}/{\mathrm{{IN}}}^{c} \geq  1 - {\mathrm{{IN}}}^{{\rho }^{ * }}/{\mathrm{{IN}}}^{c} = 1 - 1/{\mathrm{{IN}}}^{c - {\rho }^{ * }}$ that all the tuples in Join(Q)are sampled. The total running time is $O\left( {s \cdot  {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)  =$ $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ .

首先，让我们假设我们神奇地知道了值 OUT。如果 OUT $= 0$ ，连接结果 Join(Q) 为空，无需进行任何操作。否则，我们使用 $\Upsilon$ 从 $\operatorname{Join}\left( \mathcal{Q}\right)$ 中抽取 $s \mathrel{\text{:=}} c \cdot  \mathrm{{OUT}} \cdot  \ln \mathrm{{IN}}$ 个样本，其中 $c$ 是一个足够大的常数。在高概率（w.h.p.）情况下，每个元组 $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ 必定至少被抽取一次。实际上，$\mathbf{u}$ 被所有 $s$ 个样本都遗漏的概率为 ${\left( 1 - \frac{1}{\mathrm{{OUT}}}\right) }^{s}$ ，该概率至多为 ${e}^{-s/\mathrm{{OUT}}} = 1/{\mathrm{{IN}}}^{c}$ 。因此，至少以 $1 - \mathrm{{OUT}}/{\mathrm{{IN}}}^{c} \geq  1 - {\mathrm{{IN}}}^{{\rho }^{ * }}/{\mathrm{{IN}}}^{c} = 1 - 1/{\mathrm{{IN}}}^{c - {\rho }^{ * }}$ 的概率保证 Join(Q) 中的所有元组都被抽取到。总运行时间为 $O\left( {s \cdot  {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)  =$ $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 。

The rest of the proof explains how to remove the magic assumption. Instead of the exact OUT, we aim to obtain an over-estimate OÚT satisfying OUT $\leq$ OÜT $\leq  2$ OUT. By replacing OUT with $\widehat{\mathrm{{OUT}}}$ in the above,we can only decrease the algorithm’s failure probability, while keeping the execution time at the same order. In [21], Chen and Yi described a method for estimating OUT, but their method requires special knowledge of the sampling algorithm of ${\Upsilon }_{ \cdot  }^{9}$ Our method, presented below, works for any sampling algorithm.

证明的其余部分将解释如何去除这个神奇的假设。我们的目标不是获取精确的 OUT 值，而是得到一个高估的 OÚT 值，使其满足 OUT $\leq$ OÜT $\leq  2$ OUT。通过将上述过程中的 OUT 替换为 $\widehat{\mathrm{{OUT}}}$ ，我们只会降低算法的失败概率，同时保持执行时间的阶数不变。在文献 [21] 中，Chen 和 Yi 描述了一种估计 OUT 的方法，但他们的方法需要了解 ${\Upsilon }_{ \cdot  }^{9}$ 采样算法的特殊知识。下面介绍的我们的方法适用于任何采样算法。

We start by using the sample algorithm of $\Upsilon$ to find out if OUT $= 0$ . Recall that,if OUT $\geq  1$ ,the algorithm is required to return a sample in $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time w.h.p. (we can assume OUT $\leq  {\mathrm{{IN}}}^{\epsilon }$ because otherwise our $\epsilon$ -output sensitive algorithm does not need to guarantee anything). Motivated by this, we allow the algorithm to execute for $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time and then manually terminate it if it has not finished yet. If a sample has been returned,then obviously OUT $> 0$ ; otherwise,we declare $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)  = \varnothing$ .

我们首先使用 $\Upsilon$ 的采样算法来判断是否 OUT $= 0$ 。回顾一下，如果 OUT $\geq  1$ ，该算法需要在高概率（w.h.p.）情况下在 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间内返回一个样本（我们可以假设 OUT $\leq  {\mathrm{{IN}}}^{\epsilon }$ ，因为否则我们的 $\epsilon$ 输出敏感算法无需保证任何事情）。受此启发，我们允许该算法运行 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间，如果届时还未完成则手动终止它。如果已经返回了一个样本，那么显然 OUT $> 0$ ；否则，我们声明 $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)  = \varnothing$ 。

The subsequent discussion concentrates on the scenario of OUT $>$ 0 . We use $\Upsilon$ to extract samples continuously and,for each sample, check if it has been seen before (this can be done in $\widetilde{O}\left( 1\right)$ time by maintaining a dictionary-search structure, e.g., the BST, on the seen samples). The extraction stops as soon as $\Upsilon$ churns out $\Delta  \mathrel{\text{:=}} {c}^{\prime }\log \mathrm{{IN}}$ seen samples in a row,where ${c}^{\prime }$ is a sufficiently large constant. At this moment,count the number $t$ of distinct samples already obtained and finalize our estimate $\widehat{\mathrm{{OUT}}} \mathrel{\text{:=}} {2t}$ .

接下来的讨论集中在 OUT $>$ 0 的情况。我们使用 $\Upsilon$ 持续抽取样本，并且对于每个样本，检查它是否之前已经出现过（这可以通过在已出现的样本上维护一个字典搜索结构，例如二叉搜索树（BST），在 $\widetilde{O}\left( 1\right)$ 时间内完成）。一旦 $\Upsilon$ 连续输出 $\Delta  \mathrel{\text{:=}} {c}^{\prime }\log \mathrm{{IN}}$ 个已出现过的样本，抽取过程就停止，其中 ${c}^{\prime }$ 是一个足够大的常数。此时，统计已经获得的不同样本的数量 $t$ ，并确定我们的估计值 $\widehat{\mathrm{{OUT}}} \mathrel{\text{:=}} {2t}$ 。

It is easy to analyze the running time. Until termination, the algorithm finds a new tuple in $\mathfrak{J}\operatorname{oin}\left( \mathcal{Q}\right)$ after drawing at most $\Delta$ samples in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)$ time. As $\mathcal{J}$ oin(Q)has OUT tuples,the total execution time is no more than $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ .

分析运行时间很容易。直到终止，该算法在$\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }/\mathrm{{OUT}}}\right)$时间内最多抽取$\Delta$个样本后，能在$\mathfrak{J}\operatorname{oin}\left( \mathcal{Q}\right)$中找到一个新元组。由于$\mathcal{J}$ oin(Q)有OUT个元组，总执行时间不超过$\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$。

To complete the whole proof,we argue that $t \geq  \mathrm{{OUT}}/2$ w.h.p., which means OÜT $\in$ [OUT,2OUT] w.h.p.,as desired. Fix an arbitrary integer $\tau  \in  \lbrack 0,\mathrm{{OUT}}/2)$ . For the algorithm to terminate with $t = \tau ,\Upsilon$ needs to output $\Delta$ seen samples in a row when $\mathcal{J}$ oin(Q) still has at least OUT $- \tau  \geq$ OUT/2 tuples never sampled. As each sample is uniformly random,it has at least $1/2$ probability to hit an unseen tuple. The probability of fetching $\Delta$ seen samples continuously is at most ${\left( 1/2\right) }^{\Delta } = 1/{\mathrm{{IN}}}^{{c}^{\prime }}$ ,which is thus an upper bound for the algorithm to terminate with $t = \tau$ . Accounting for all the possible $\tau$ values,we can conclude that the algorithm finishes with a $t \in  \lbrack 0,\mathrm{{OUT}}/2)$ with probability $O\left( {\mathrm{{OUT}}/{\mathrm{{IN}}}^{{c}^{\prime }}}\right)  = O\left( {1/{\mathrm{{IN}}}^{{c}^{\prime } - {\rho }^{ * }}}\right)$ . Therefore, $t$ has probability $1 - O\left( {1/{\mathrm{{IN}}}^{{c}^{\prime } - {\rho }^{ * }}}\right)$ to be at least OUT $/2$ .

为完成整个证明，我们论证在高概率（w.h.p.）下$t \geq  \mathrm{{OUT}}/2$成立，这意味着在高概率下OÜT $\in$ [OUT, 2OUT]，符合要求。固定一个任意整数$\tau  \in  \lbrack 0,\mathrm{{OUT}}/2)$。要使算法以$t = \tau ,\Upsilon$终止，当$\mathcal{J}$ oin(Q)仍至少有OUT $- \tau  \geq$ OUT/2个元组从未被采样时，需要连续输出$\Delta$个已见过的样本。由于每个样本是均匀随机的，它至少有$1/2$的概率命中一个未见过的元组。连续获取$\Delta$个已见过的样本的概率至多为${\left( 1/2\right) }^{\Delta } = 1/{\mathrm{{IN}}}^{{c}^{\prime }}$，因此这是算法以$t = \tau$终止的一个上界。考虑所有可能的$\tau$值，我们可以得出结论：算法以$t \in  \lbrack 0,\mathrm{{OUT}}/2)$结束的概率为$O\left( {\mathrm{{OUT}}/{\mathrm{{IN}}}^{{c}^{\prime }}}\right)  = O\left( {1/{\mathrm{{IN}}}^{{c}^{\prime } - {\rho }^{ * }}}\right)$。因此，$t$至少为OUT $/2$的概率为$1 - O\left( {1/{\mathrm{{IN}}}^{{c}^{\prime } - {\rho }^{ * }}}\right)$。

## D PROOF OF LEMMA 7

## D 引理7的证明

As before,let $\mathcal{A}$ be the given combinatorial $\epsilon$ -output sensitive algorithm. Denote by ${\mathcal{A}}^{\prime }$ the sampling algorithm of our structure in Theorem 5. To detect whether $\mathcal{J}$ oin(Q)is empty,we first build our structure on $Q$ in $\widetilde{O}\left( \mathrm{{IN}}\right)$ time by inserting every tuple of the input relations one by one. Then,run $\mathcal{A}$ and ${\mathcal{A}}^{\prime }$ in an interleaving manner, that is,run a step (of constant time) of $\mathcal{A}$ ,followed by a step of ${\mathcal{A}}^{\prime }$ , another step of $\mathcal{A}$ ,then a step of ${\mathcal{A}}^{\prime }$ ,and so on. The interleaving process stops as soon as either algorithm finishes. At that moment, check whether $\mathcal{A}$ and ${\mathcal{A}}^{\prime }$ have found any tuple of $\mathcal{J}$ oin(Q). If so, obviously Join(Q)is not empty; otherwise,declare Join(Q)empty.

和之前一样，设$\mathcal{A}$是给定的组合式$\epsilon$ -输出敏感算法。用${\mathcal{A}}^{\prime }$表示定理5中我们所构建结构的采样算法。为了检测$\mathcal{J}$ oin(Q)是否为空，我们首先在$Q$上以$\widetilde{O}\left( \mathrm{{IN}}\right)$时间构建我们的结构，方法是将输入关系的每个元组逐个插入。然后，以交错的方式运行$\mathcal{A}$和${\mathcal{A}}^{\prime }$，即运行$\mathcal{A}$的一步（常数时间），接着运行${\mathcal{A}}^{\prime }$的一步，再运行$\mathcal{A}$的一步，然后运行${\mathcal{A}}^{\prime }$的一步，依此类推。一旦其中一个算法完成，交错过程就停止。在那一刻，检查$\mathcal{A}$和${\mathcal{A}}^{\prime }$是否找到了$\mathcal{J}$ oin(Q)的任何元组。如果找到了，显然Join(Q)不为空；否则，判定Join(Q)为空。

Let us represent the above emptiness-detection algorithm as ${\mathcal{A}}_{emp}$ . Next,we will prove that ${\mathcal{A}}_{\text{emp }}$ ,w.h.p.,correctly decides if $\operatorname{Join}\left( \mathcal{Q}\right)  =$ $\varnothing$ and runs in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time. Our analysis is through a case-by-case discussion on the value of OUT.

我们将上述空检测算法表示为${\mathcal{A}}_{emp}$。接下来，我们将证明在高概率（w.h.p.）下，${\mathcal{A}}_{\text{emp }}$能正确判定$\operatorname{Join}\left( \mathcal{Q}\right)  =$ $\varnothing$是否成立，并且运行时间为$\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$。我们的分析是通过对OUT的值进行逐例讨论来完成的。

- If OUT $= 0,{\mathcal{A}}_{\text{emp }}$ always returns "Ioin(Q)empty". Because $\mathcal{A}$ is $\epsilon$ -output sensitive,it terminates in $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time w.h.p.. The cost of ${\mathcal{A}}_{\text{emp }}$ is thus bounded by $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ w.h.p..

- 如果OUT $= 0,{\mathcal{A}}_{\text{emp }}$ 始终返回 “Ioin(Q)为空”。由于 $\mathcal{A}$ 是 $\epsilon$ 输出敏感的，它以高概率（w.h.p.）在 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间内终止。因此，${\mathcal{A}}_{\text{emp }}$ 的成本以高概率受限于 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$。

- Consider now $0 < \mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$ . Being $\epsilon$ -output sensitive, $\mathcal{A}$ must report the full $\operatorname{Join}\left( \mathcal{Q}\right)$ in $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time w.h.p.. Hence, ${\mathcal{A}}_{\text{emp }}$ finds a tuple of $\mathcal{J}$ oin(Q)in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time w.h.p.

- 现在考虑 $0 < \mathrm{{OUT}} \leq  {\mathrm{{IN}}}^{\epsilon }$。由于 $\epsilon$ 是输出敏感的，$\mathcal{A}$ 必须以高概率在 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间内报告完整的 $\operatorname{Join}\left( \mathcal{Q}\right)$。因此，${\mathcal{A}}_{\text{emp }}$ 以高概率在 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 时间内找到 $\mathcal{J}$ oin(Q) 的一个元组。

- The final case is $\mathrm{{OUT}} > {\mathrm{{IN}}}^{\epsilon }$ . By Theorem 5, ${\mathcal{A}}^{\prime }$ must return a join sample of $Q$ in $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ time w.h.p.. ${\mathcal{A}}_{emp}$ thus finds a tuple of $\mathcal{J}$ oin(Q)in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time w.h.p..

- 最后一种情况是 $\mathrm{{OUT}} > {\mathrm{{IN}}}^{\epsilon }$。根据定理5，${\mathcal{A}}^{\prime }$ 必须以高概率在 $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }\right)$ 时间内返回 $Q$ 的一个连接样本。因此，${\mathcal{A}}_{emp}$ 以高概率在 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 时间内找到 $\mathcal{J}$ oin(Q) 的一个元组。

## E SUBGRAPH SAMPLING

## E 子图采样

Before discussing subgraph sampling, we first extend Theorem 5 to a scenario we call join sampling with predicates, where sampling is performed on only a subset of the join result. Let $Q$ be a join defined in Section 2.1. Given a boolean predicate $\sigma$ ,define $\mathcal{J}$ oin $\left( {\sigma ,\mathcal{Q}}\right)  \mathrel{\text{:=}}$ $\{ \mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)  \mid  \mathbf{u}$ satisfies $\sigma \}$ ,i.e.,the subset of tuples in $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ passing the filtering condition $\sigma$ . A $\sigma$ -join sample of $Q$ is a tuple taken uniformly at random from $\mathcal{J}$ oin $\left( {\sigma ,Q}\right)$ . We want to create an index structure on $Q$ to allow fast extraction of $\sigma$ -join samples.

在讨论子图采样之前，我们首先将定理5扩展到一个我们称之为带谓词的连接采样的场景，在该场景中，采样仅在连接结果的一个子集上进行。设 $Q$ 是第2.1节中定义的一个连接。给定一个布尔谓词 $\sigma$，定义 $\mathcal{J}$ oin $\left( {\sigma ,\mathcal{Q}}\right)  \mathrel{\text{:=}}$ $\{ \mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)  \mid  \mathbf{u}$ 满足 $\sigma \}$，即 $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ 中通过过滤条件 $\sigma$ 的元组子集。$Q$ 的一个 $\sigma$ -连接样本是从 $\mathcal{J}$ oin $\left( {\sigma ,Q}\right)$ 中均匀随机选取的一个元组。我们希望在 $Q$ 上创建一个索引结构，以便快速提取 $\sigma$ -连接样本。

Interestingly, our structure in Theorem 5 can be deployed directly to draw a $\sigma$ -join sample,even if the predicate $\sigma$ is supplied at run time. For this purpose, simply apply the sample algorithm in Figure 3 . If sample declares "failure", we declare the same. Otherwise, suppose that sample returns a sample $\mathbf{u} \in  \mathcal{J}$ oin(Q),which we output only if $\mathbf{u}$ satisfies $\sigma$ . Otherwise ( $\mathbf{u}$ violates $\sigma$ ),we again declare "failure". The above algorithm (for drawing a $\sigma$ -join sample) will be referred to as $\sigma$ -sample henceforth.

有趣的是，即使谓词 $\sigma$ 是在运行时提供的，我们定理5中的结构也可以直接用于抽取一个 $\sigma$ -连接样本。为此，只需应用图3中的采样算法。如果采样声明 “失败”，我们也声明失败。否则，假设采样返回一个样本 $\mathbf{u} \in  \mathcal{J}$ oin(Q)，只有当 $\mathbf{u}$ 满足 $\sigma$ 时我们才输出它。否则（$\mathbf{u}$ 违反 $\sigma$），我们再次声明 “失败”。上述算法（用于抽取一个 $\sigma$ -连接样本）此后将被称为 $\sigma$ -采样。

---

<!-- Footnote -->

${}^{9}$ Specifically,Chen and Yi’s method assumes that the sampling algorithm of $\Upsilon$ works by repeatedly making trials, each of which either declares "failure" or produces a sample. The failure probability must be available for their method to work.

${}^{9}$ 具体来说，陈和易的方法假设 $\Upsilon$ 的采样算法通过反复进行试验来工作，每次试验要么声明 “失败”，要么产生一个样本。他们的方法要起作用必须知道失败概率。

<!-- Footnote -->

---

Recall that sample draws $\mathbf{u}$ from $\mathfrak{J}$ oin(Q)uniformly at random. Thus,every tuple in $\mathcal{J}$ oin $\left( {\sigma ,\mathcal{Q}}\right)$ ,which is a subset of $\mathcal{J}$ oin(Q),has the same chance to be taken as $\mathbf{u}$ . Hence, $\sigma$ -sample,if it succeeds (i.e.,outputting a tuple),returns a $\sigma$ -join sample of $Q$ . To analyze its success probability,let ${\mathrm{{OUT}}}_{\sigma } \mathrel{\text{:=}} \left| {\mathcal{J}\operatorname{oin}\left( {\sigma ,\mathcal{Q}}\right) }\right|$ . As shown in Section 4.2,sample outputs a tuple of $\mathcal{J}$ oin(Q)with probability $\frac{\text{OUT}}{{\text{AGM}}_{W}\left( Q\right) }$ ,which tells us that $\sigma$ -sample succeeds with probability $\frac{\mathrm{{OUT}}}{{\mathrm{{AGM}}}_{W}\left( Q\right) } \cdot  \frac{{\mathrm{{OUT}}}_{\sigma }}{\mathrm{{OUT}}} = \frac{{\mathrm{{OUT}}}_{\sigma }}{{\mathrm{{AGM}}}_{W}\left( Q\right) }$ . It thus follows that we can, by repeating the algorithm until a success,draw a $\sigma$ -join sample of $\mathcal{Q}$ in $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\max \left\{  {1,{\mathrm{{OUT}}}_{\sigma }}\right\}  }\right)$ time,which can be made $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \left\{  {1,{\mathrm{{OUT}}}_{\sigma }}\right\}  }\right)$ by choosing the fractional edge covering $W$ optimally,where ${\rho }^{ * }$ is the fractional edge covering number of the schema graph of $Q$ .

回顾一下，样本抽取 $\mathbf{u}$ 是从 $\mathfrak{J}$ （oin(Q)）中均匀随机抽取的。因此，$\mathcal{J}$ （oin $\left( {\sigma ,\mathcal{Q}}\right)$ ）中的每个元组（它是 $\mathcal{J}$ （oin(Q)）的一个子集）被选为 $\mathbf{u}$ 的机会是相同的。因此，$\sigma$ -样本，如果它成功（即输出一个元组），将返回 $Q$ 的一个 $\sigma$ -连接样本。为了分析其成功概率，设 ${\mathrm{{OUT}}}_{\sigma } \mathrel{\text{:=}} \left| {\mathcal{J}\operatorname{oin}\left( {\sigma ,\mathcal{Q}}\right) }\right|$ 。如第4.2节所示，样本以概率 $\frac{\text{OUT}}{{\text{AGM}}_{W}\left( Q\right) }$ 输出 $\mathcal{J}$ （oin(Q)）的一个元组，这告诉我们 $\sigma$ -样本以概率 $\frac{\mathrm{{OUT}}}{{\mathrm{{AGM}}}_{W}\left( Q\right) } \cdot  \frac{{\mathrm{{OUT}}}_{\sigma }}{\mathrm{{OUT}}} = \frac{{\mathrm{{OUT}}}_{\sigma }}{{\mathrm{{AGM}}}_{W}\left( Q\right) }$ 成功。因此，通过重复该算法直到成功，我们可以在 $\widetilde{O}\left( {{\mathrm{{AGM}}}_{W}\left( \mathcal{Q}\right) /\max \left\{  {1,{\mathrm{{OUT}}}_{\sigma }}\right\}  }\right)$ 时间内抽取 $\mathcal{Q}$ 的一个 $\sigma$ -连接样本，通过最优地选择分数边覆盖 $W$ ，可以使这个时间达到 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \left\{  {1,{\mathrm{{OUT}}}_{\sigma }}\right\}  }\right)$ ，其中 ${\rho }^{ * }$ 是 $Q$ 的模式图的分数边覆盖数。

We are ready to solve the subgraph sampling problem. Recall that the goal is to preprocess an undirected graph $G \mathrel{\text{:=}} \left( {V,E}\right)$ such that, given a constant-size pattern graph $Q$ ,we can uniformly sample an occurrence of $Q$ from $G$ . Let $\mathcal{X}$ be the set of vertices in $Q$ ,and $\mathcal{E}$ be the set of edges in $Q$ . It is worth reminding the reader that every edge in $G$ and $Q$ contains exactly two vertices.

我们准备好解决子图采样问题了。回顾一下，目标是对一个无向图 $G \mathrel{\text{:=}} \left( {V,E}\right)$ 进行预处理，使得在给定一个固定大小的模式图 $Q$ 的情况下，我们可以从 $G$ 中均匀地采样 $Q$ 的一个出现。设 $\mathcal{X}$ 是 $Q$ 中的顶点集，$\mathcal{E}$ 是 $Q$ 中的边集。值得提醒读者的是，$G$ 和 $Q$ 中的每条边都恰好包含两个顶点。

We create a join $Q$ with $\left| \mathcal{E}\right|$ relations as follows. For each edge $e = \{ X,Y\}$ in $Q$ (where $X$ and $Y$ are vertices in $Q$ ),create a relation ${R}_{e}$ with schema $\operatorname{var}\left( {R}_{e}\right)  = \{ X,Y\}$ . The size of ${R}_{e}$ is $2\left| E\right|$ ,i.e.,twice the number of edges in $G$ . Specifically,for every edge $\{ x,y\}$ in $G$ (where $x$ and $y$ are vertices in $G$ ),we create two tuples in ${R}_{e}$ : tuple ${\mathbf{u}}_{1}$ satisfying ${\mathbf{u}}_{1}\left( X\right)  = x$ and ${\mathbf{u}}_{1}\left( Y\right)  = y$ ,and tuple ${\mathbf{u}}_{2}$ satisfying ${\mathbf{u}}_{2}\left( X\right)  = y$ and ${\mathbf{u}}_{2}\left( Y\right)  = x$ . This completes the construction of $Q$ , which has input size $\operatorname{IN} = \left| \mathcal{E}\right|  \cdot  \left( {2\left| E\right| }\right)  = O\left( \left| E\right| \right)$ .

我们按如下方式创建一个连接 $Q$，它包含 $\left| \mathcal{E}\right|$ 个关系。对于 $Q$ 中的每条边 $e = \{ X,Y\}$（其中 $X$ 和 $Y$ 是 $Q$ 中的顶点），创建一个模式为 $\operatorname{var}\left( {R}_{e}\right)  = \{ X,Y\}$ 的关系 ${R}_{e}$。${R}_{e}$ 的大小为 $2\left| E\right|$，即 $G$ 中边的数量的两倍。具体来说，对于 $G$ 中的每条边 $\{ x,y\}$（其中 $x$ 和 $y$ 是 $G$ 中的顶点），我们在 ${R}_{e}$ 中创建两个元组：满足 ${\mathbf{u}}_{1}\left( X\right)  = x$ 和 ${\mathbf{u}}_{1}\left( Y\right)  = y$ 的元组 ${\mathbf{u}}_{1}$，以及满足 ${\mathbf{u}}_{2}\left( X\right)  = y$ 和 ${\mathbf{u}}_{2}\left( Y\right)  = x$ 的元组 ${\mathbf{u}}_{2}$。这样就完成了 $Q$ 的构建，其输入大小为 $\operatorname{IN} = \left| \mathcal{E}\right|  \cdot  \left( {2\left| E\right| }\right)  = O\left( \left| E\right| \right)$。

Every tuple $\mathbf{u}$ in the result $\mathcal{J}$ oin(Q)of $\mathcal{Q}$ can be thought of as mapping each edge $\{ X,Y\}$ in $Q$ to an edge $\{ \mathbf{u}\left( X\right) ,\mathbf{u}\left( Y\right) \}$ in $G$ . If the set of (mapped) edges $\{ \{ \mathbf{u}\left( X\right) ,\mathbf{u}\left( Y\right) \}  \mid  \{ X,Y\}  \in  \mathcal{E}\}$ induces an occurrence of $Q$ ,we say that $\mathbf{u}$ describes the occurrence.

$\mathcal{Q}$ 的结果 $\mathcal{J}$ oin(Q) 中的每个元组 $\mathbf{u}$ 都可以看作是将 $Q$ 中的每条边 $\{ X,Y\}$ 映射到 $G$ 中的一条边 $\{ \mathbf{u}\left( X\right) ,\mathbf{u}\left( Y\right) \}$。如果（映射后的）边集 $\{ \{ \mathbf{u}\left( X\right) ,\mathbf{u}\left( Y\right) \}  \mid  \{ X,Y\}  \in  \mathcal{E}\}$ 构成了 $Q$ 的一个实例，我们就说 $\mathbf{u}$ 描述了这个实例。

The following are two (folklore) facts about relationships between the tuples in $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ and the occurrences of $Q$ .

以下是关于 $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ 中的元组与 $Q$ 的实例之间关系的两个（常见）事实。

- Fact 1: Every occurrence of $Q$ is described by the same number $c$ of tuples in $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ ,where $c \geq  1$ is a constant. For example,consider $Q$ to be a triangle with vertices $X,Y$ ,and $Z$ . An occurrence of $Q$ ,which is a triangle with vertices $x,y$ ,and $z$ in $G$ ,is described by the tuple $\mathbf{u} \in  \mathcal{J}$ oin(Q)where $\mathbf{u}\left( X\right)  = x$ , $\mathbf{u}\left( Y\right)  = y$ ,and $\mathbf{u}\left( Z\right)  = z$ . It is easy to see that the triangle is described by six tuples of $\mathcal{J}$ oin(Q)in total,and six is exactly the number of automorhisms of $Q$ .

- 事实 1：$Q$ 的每个实例都由 $\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right)$ 中相同数量 $c$ 的元组描述，其中 $c \geq  1$ 是一个常数。例如，假设 $Q$ 是一个顶点为 $X,Y$、$Z$ 的三角形。$Q$ 的一个实例，即在 $G$ 中顶点为 $x,y$、$z$ 的三角形，由元组 $\mathbf{u} \in  \mathcal{J}$ oin(Q) 描述，其中 $\mathbf{u}\left( X\right)  = x$、$\mathbf{u}\left( Y\right)  = y$ 和 $\mathbf{u}\left( Z\right)  = z$。很容易看出，这个三角形总共由 $\mathcal{J}$ oin(Q) 中的六个元组描述，而六恰好是 $Q$ 的自同构数量。

- Fact 2: It is possible for $\operatorname{Join}\left( \mathcal{Q}\right)$ to contain tuples that do not describe any occurrence of $Q$ . For example,consider $Q$ to be a 4-cycle with vertices ${X}_{1},{X}_{2},{X}_{3}$ ,and ${X}_{4}$ . The tuple $\mathbf{u}$ with $\left( {\mathbf{u}\left( {X}_{1}\right) ,\mathbf{u}\left( {X}_{2}\right) ,\mathbf{u}\left( {X}_{3}\right) ,\mathbf{u}\left( {X}_{4}\right) }\right)  = \left( {x,y,x,y}\right)$ ,where $\{ x,y\}$ is an edge in $G$ ,belongs to $\mathcal{J}$ oin(Q)but does not describe any occurrence of $Q$ .

- 事实2：$\operatorname{Join}\left( \mathcal{Q}\right)$ 有可能包含未描述 $Q$ 任何出现情况的元组。例如，假设 $Q$ 是一个具有顶点 ${X}_{1},{X}_{2},{X}_{3}$ 和 ${X}_{4}$ 的4 - 环。元组 $\mathbf{u}$ 且 $\left( {\mathbf{u}\left( {X}_{1}\right) ,\mathbf{u}\left( {X}_{2}\right) ,\mathbf{u}\left( {X}_{3}\right) ,\mathbf{u}\left( {X}_{4}\right) }\right)  = \left( {x,y,x,y}\right)$，其中 $\{ x,y\}$ 是 $G$ 中的一条边，它属于 $\mathcal{J}$ oin(Q)，但并未描述 $Q$ 的任何出现情况。

To sample occurrences,we create a structure of Theorem 5 on $Q$ , which occupies $\widetilde{O}\left( \left| E\right| \right)$ space and can be easily maintained in $\widetilde{O}\left( 1\right)$ time per edge insertion and deletion. To draw a sample, we apply the $\sigma$ -sample algorithm by setting the predicate $\sigma$ to "tuple $\mathbf{u}$ should describe an occurrence of $Q$ ". The predicate can be evaluated in constant time. By Fact 1,the value of ${\mathrm{{OUT}}}_{\sigma }$ equals $c \cdot  \mathrm{{OCC}}$ ,where OCC is the number of occurrences of $Q$ in $G$ . Our earlier discussion indicates that the sample time is $\widetilde{O}\left( {{\left| E\right| }^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OCC}}\} }\right)$ .

为了对出现情况进行采样，我们在 $Q$ 上创建一个定理5的结构，该结构占用 $\widetilde{O}\left( \left| E\right| \right)$ 空间，并且在每次插入和删除边时可以在 $\widetilde{O}\left( 1\right)$ 时间内轻松维护。为了抽取一个样本，我们通过将谓词 $\sigma$ 设置为“元组 $\mathbf{u}$ 应描述 $Q$ 的一个出现情况”来应用 $\sigma$ - 采样算法。该谓词可以在常数时间内进行评估。根据事实1，${\mathrm{{OUT}}}_{\sigma }$ 的值等于 $c \cdot  \mathrm{{OCC}}$，其中OCC是 $Q$ 在 $G$ 中出现的次数。我们之前的讨论表明采样时间为 $\widetilde{O}\left( {{\left| E\right| }^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OCC}}\} }\right)$。

## F ( 6-OUTPUT SENSITIVITY BREAKS COMBINATORIAL $k$ -CLIQUE HYPOTHESIS

## F（6 - 输出敏感性打破组合 $k$ - 团假设

By Lemma 7,the existence of an $\epsilon$ -output sensitive algorithm implies a combinatorial algorithm ${\mathcal{A}}_{\text{emp }}$ that,given any join $Q$ ,can decide whether $\mathcal{J}$ oin $\left( \mathcal{Q}\right)  = \varnothing$ in $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ time w.h.p.. Next,we show how to break the combinatorial $k$ -clique hypothesis with ${\mathcal{A}}_{emp}$ for any constant $k \geq  3$ .

根据引理7，一个 $\epsilon$ - 输出敏感算法的存在意味着存在一个组合算法 ${\mathcal{A}}_{\text{emp }}$，给定任何连接 $Q$，该算法可以在 $\widetilde{O}\left( {\mathrm{{IN}} + {\mathrm{{IN}}}^{{\rho }^{ * } - \epsilon }}\right)$ 时间内以高概率决定是否 $\mathcal{J}$ oin $\left( \mathcal{Q}\right)  = \varnothing$。接下来，我们展示如何针对任何常数 $k \geq  3$，用 ${\mathcal{A}}_{emp}$ 打破组合 $k$ - 团假设。

Let $G \mathrel{\text{:=}} \left( {V,E}\right)$ be a simple undirected graph. In Appendix E,we presented a strategy to convert subgraph sampling to join sampling for any pattern graph $Q$ . Here,we apply the same ideas to construct a join $Q$ from $G$ ,setting $Q$ to $k$ -clique. As before,for each edge $e = \{ X,Y\}$ in $Q$ ,create a relation ${R}_{e}$ with schema $\operatorname{var}\left( {R}_{e}\right)  = \{ X,Y\}$ . For every edge $\{ x,y\}$ in $G$ ,relation ${R}_{e}$ contains two tuples: tuple ${\mathbf{u}}_{1}$ with ${\mathbf{u}}_{1}\left( X\right)  = x$ and ${\mathbf{u}}_{1}\left( Y\right)  = y$ ,and tuple ${\mathbf{u}}_{2}$ with ${\mathbf{u}}_{2}\left( X\right)  =$ $y$ and ${\mathbf{u}}_{2}\left( Y\right)  = x$ . Recall that the conversion has two properties, presented as Facts 1 and 2 in Appendix E. Crucially, Fact 2 can now be strengthened into the claim below (for $Q = k$ -clique):

设 $G \mathrel{\text{:=}} \left( {V,E}\right)$ 是一个简单无向图。在附录E中，我们提出了一种将子图采样转换为任意模式图 $Q$ 的连接采样的策略。在这里，我们应用相同的思路从 $G$ 构建一个连接 $Q$，将 $Q$ 设置为 $k$ - 团。和之前一样，对于 $Q$ 中的每条边 $e = \{ X,Y\}$，创建一个具有模式 $\operatorname{var}\left( {R}_{e}\right)  = \{ X,Y\}$ 的关系 ${R}_{e}$。对于 $G$ 中的每条边 $\{ x,y\}$，关系 ${R}_{e}$ 包含两个元组：元组 ${\mathbf{u}}_{1}$ 且 ${\mathbf{u}}_{1}\left( X\right)  = x$ 和 ${\mathbf{u}}_{1}\left( Y\right)  = y$，以及元组 ${\mathbf{u}}_{2}$ 且 ${\mathbf{u}}_{2}\left( X\right)  =$ $y$ 和 ${\mathbf{u}}_{2}\left( Y\right)  = x$。回想一下，这种转换有两个性质，在附录E中作为事实1和事实2给出。至关重要的是，现在可以将事实2强化为以下断言（针对 $Q = k$ - 团）：

Every tuple in $\mathcal{J}$ oin(Q)describes an occurrence of $Q$ in $G$ .

$\mathcal{J}$ oin(Q)中的每个元组都描述了$Q$在$G$中的一次出现。

To explain why,consider any tuple $\mathbf{u} \in  \mathcal{J}$ oin(Q). For each edge $e = \{ X,Y\}$ in $Q$ ,we know that $\{ \mathbf{u}\left( X\right) ,\mathbf{u}\left( Y\right) \}$ is in ${R}_{e}$ and,hence,is an edge in $G$ . It thus follows that $G$ has an edge between vertices $\mathbf{u}\left( X\right)$ and $\mathbf{u}\left( Y\right)$ for any two distinct vertices $X$ and $Y$ in $Q$ . Therefore, $\mathbf{u}$ describes a $k$ -clique occurrence in $G$ .

为了解释原因，考虑$\mathbf{u} \in  \mathcal{J}$ oin(Q)中的任意元组。对于$Q$中的每条边$e = \{ X,Y\}$，我们知道$\{ \mathbf{u}\left( X\right) ,\mathbf{u}\left( Y\right) \}$在${R}_{e}$中，因此，它是$G$中的一条边。由此可知，对于$Q$中任意两个不同的顶点$X$和$Y$，$G$在顶点$\mathbf{u}\left( X\right)$和$\mathbf{u}\left( Y\right)$之间都有一条边。因此，$\mathbf{u}$描述了$G$中一个$k$ - 团（$k$ - clique）的出现。

Combined with Fact 1,the above claim indicates that $G$ has a $k$ -clique if and only if $\operatorname{Join}\left( Q\right)$ is non-empty. Therefore,we can apply ${\mathcal{A}}_{\text{emp }}$ to detect the emptiness of $\mathcal{J}$ oin(Q)and then decide on the $k$ -clique presence in $G$ . To analyze running time,we note that the fractional edge covering number of $k$ -clique is $k/2$ ,as is also the fractional edge covering number ${\rho }^{ * }$ of the schema graph $Q$ of $Q$ . Therefore,the algorithm described earlier runs in $\widetilde{O}\left( {\left| E\right|  + {\left| E\right| }^{\frac{k}{2} - \epsilon }}\right)$ time. Applying the trivial fact $\left| E\right|  \leq  {\left| V\right| }^{2}$ ,we can relax the time complexity to $\widetilde{O}\left( {{\left| V\right| }^{2} + {\left| V\right| }^{k - {2\epsilon }}}\right)$ ,which is $\widetilde{O}\left( {\left| V\right| }^{k - {2\epsilon }}\right)$ because $k \geq  3$ and $\epsilon  < 1/2$ ,thus breaking the combinatorial $k$ -clique hypothesis.

结合事实1，上述断言表明，当且仅当$\operatorname{Join}\left( Q\right)$非空时，$G$有一个$k$ - 团（$k$ - clique）。因此，我们可以应用${\mathcal{A}}_{\text{emp }}$来检测$\mathcal{J}$ oin(Q)是否为空，然后确定$G$中是否存在$k$ - 团。为了分析运行时间，我们注意到$k$ - 团的分数边覆盖数是$k/2$，这也是$Q$的模式图$Q$的分数边覆盖数${\rho }^{ * }$。因此，前面描述的算法的运行时间为$\widetilde{O}\left( {\left| E\right|  + {\left| E\right| }^{\frac{k}{2} - \epsilon }}\right)$。应用平凡事实$\left| E\right|  \leq  {\left| V\right| }^{2}$，我们可以将时间复杂度放宽到$\widetilde{O}\left( {{\left| V\right| }^{2} + {\left| V\right| }^{k - {2\epsilon }}}\right)$，这是$\widetilde{O}\left( {\left| V\right| }^{k - {2\epsilon }}\right)$，因为$k \geq  3$和$\epsilon  < 1/2$，从而打破了组合$k$ - 团假设。

## G JOINS WITH RANDOM ENUMERATION

## 带随机枚举的G连接

Our solution to this problem combines ideas in Appendix C and a technique in [52] that adapts a delay-oblivious reporting algorithm for small-delay enumeration.

我们对这个问题的解决方案结合了附录C中的思想和文献[52]中的一种技术，该技术采用了一种无延迟报告算法来进行小延迟枚举。

First,create a structure of Theorem 5 on the given join $Q$ in $\widetilde{O}\left( \mathrm{{IN}}\right)$ time. Then,we find out whether $\mathrm{{OUT}} = 0$ . For this purpose, run the sampling algorithm of Theorem 5 - denoted as $\mathcal{A}$ henceforth - to see if it returns a sample. If so, $\mathcal{A}$ must have done so in $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ time w.h.p.; obviously, $\mathrm{{OUT}} > 0$ in this scenario. Otherwise,OUT $= 0$ w.h.p.,in which case the execution time of $\mathcal{A}$ is $\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ ,and we have already solved the join.

首先，在给定的连接$Q$上以$\widetilde{O}\left( \mathrm{{IN}}\right)$时间创建定理5的结构。然后，我们确定是否$\mathrm{{OUT}} = 0$。为此，运行定理5的采样算法（此后记为$\mathcal{A}$），看它是否返回一个样本。如果是这样，$\mathcal{A}$很可能（w.h.p.）在$\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$时间内完成；显然，在这种情况下$\mathrm{{OUT}} > 0$。否则，很可能（w.h.p.）输出OUT $= 0$，在这种情况下，$\mathcal{A}$的执行时间为$\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$，并且我们已经解决了该连接问题。

The subsequent discussion focuses on OUT $> 0$ . We carry out two steps in the same fashion as in Appendix C.

后续讨论集中在输出OUT $> 0$的情况。我们按照附录C中的相同方式执行两个步骤。

- The first step obtains an estimate $\widehat{\mathrm{{OUT}}} \in  \left\lbrack  {\mathrm{{OUT}},2\mathrm{{OUT}}}\right\rbrack$ and, in the meantime,reports at least OUT/2 tuples in $\mathcal{J}$ oin(Q). The algorithm is the same as described before: keep sampling with $\mathcal{A}$ until seeing $\Delta  \mathrel{\text{:=}} O\left( {\log \mathrm{{IN}}}\right)$ seen tuples in a row. The cost of the step is $\widetilde{O}\left( {\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \cdot  \mathrm{{OUT}} \cdot  \Delta }\right)  = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ .

- 第一步得到一个估计值 $\widehat{\mathrm{{OUT}}} \in  \left\lbrack  {\mathrm{{OUT}},2\mathrm{{OUT}}}\right\rbrack$，同时，在 $\mathcal{J}$ oin(Q) 中报告至少 OUT/2 个元组。该算法与之前描述的相同：使用 $\mathcal{A}$ 持续采样，直到连续看到 $\Delta  \mathrel{\text{:=}} O\left( {\log \mathrm{{IN}}}\right)$ 个已见元组。此步骤的成本为 $\widetilde{O}\left( {\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \cdot  \mathrm{{OUT}} \cdot  \Delta }\right)  = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$。

- The second step reports the remaining tuples of $\operatorname{Join}\left( \mathcal{Q}\right)$ that have not been found yet. As in Appendix C,we use $\mathcal{A}$ to extract $s \mathrel{\text{:=}} O\left( {\widehat{\mathrm{{OUT}}} \cdot  \ln \mathrm{{IN}}}\right)$ samples and output a sample only if it has never been reported before. The cost of the step is $\widetilde{O}\left( {\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \cdot  s}\right)  = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right) .$

- 第二步报告 $\operatorname{Join}\left( \mathcal{Q}\right)$ 中尚未找到的其余元组。如附录 C 中所述，我们使用 $\mathcal{A}$ 提取 $s \mathrel{\text{:=}} O\left( {\widehat{\mathrm{{OUT}}} \cdot  \ln \mathrm{{IN}}}\right)$ 个样本，并且仅当一个样本之前从未被报告过时才输出它。此步骤的成本为 $\widetilde{O}\left( {\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \cdot  s}\right)  = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right) .$

It is immediate from the analysis in Appendix C that the above two-step algorithm w.h.p. manages to output the entire $\mathcal{J}$ oin(Q)in a random permutation. However, it is not designed to achieve a small delay. In fact, Step 1 is fine: it enumerates at least OUT/2 result tuples with a delay $\widetilde{O}\left( {\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \cdot  \Delta }\right)  = \widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ . The trouble lies in Step 2 where we may fetch a large number of samples before hitting an unseen tuple.

从附录 C 的分析中可以立即看出，上述两步算法很可能（w.h.p.）能够以随机排列的方式输出整个 $\mathcal{J}$ oin(Q)。然而，它并非为实现小延迟而设计。实际上，第一步没问题：它以 $\widetilde{O}\left( {\frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \cdot  \Delta }\right)  = \widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ 的延迟枚举至少 OUT/2 个结果元组。问题在于第二步，在找到一个未见元组之前，我们可能会获取大量样本。

To eliminate the issue, we resort to a technique of Tao and Yi [52]. They defined a join reporting algorithm ${\mathcal{A}}^{\prime }$ to be $\alpha$ -aggressive if, after $t$ running time ${}^{10}$ for any integer $t \geq  1,{\mathcal{A}}^{\prime }$ must have discovered at least $\lfloor t/\alpha \rfloor$ distinct result tuples. They showed that any such ${\mathcal{A}}^{\prime }$ can be converted into an algorithm that reports all the result tuples with a delay $O\left( \alpha \right)$ . Furthermore,the converted algorithm outputs the result tuples in the same order as ${\mathcal{A}}^{\prime }$ does.

为消除这个问题，我们采用了 Tao 和 Yi [52] 的一种技术。他们将一个连接报告算法 ${\mathcal{A}}^{\prime }$ 定义为 $\alpha$ -激进的，如果在运行时间 $t$ 之后，对于任何整数 $t \geq  1,{\mathcal{A}}^{\prime }$ 必须已经发现至少 $\lfloor t/\alpha \rfloor$ 个不同的结果元组。他们表明，任何这样的 ${\mathcal{A}}^{\prime }$ 都可以转换为一个以 $O\left( \alpha \right)$ 的延迟报告所有结果元组的算法。此外，转换后的算法以与 ${\mathcal{A}}^{\prime }$ 相同的顺序输出结果元组。

We claim that the two-step algorithm explained earlier is $\alpha$ - aggressive with

我们声称，前面解释的两步算法是 $\alpha$ -激进的，其中

$$
\alpha  = \beta  \cdot  \Delta  \cdot  \frac{{\mathrm{{IN}}}^{{\rho }^{ * }}}{\mathrm{{OUT}}} \tag{11}
$$

where $\beta$ is a sufficiently large $\widetilde{O}\left( 1\right)$ factor. Thus,the method of [52] turns our algorithm into one that, w.h.p., outputs a random permutation of $\mathcal{J}$ oin(Q)with delay $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ .

其中 $\beta$ 是一个足够大的 $\widetilde{O}\left( 1\right)$ 因子。因此，[52] 中的方法将我们的算法转换为一个很可能（w.h.p.）以 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ 的延迟输出 $\mathcal{J}$ oin(Q) 的随机排列的算法。

To understand the claim,first note that Step 1 is $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ - aggressive because,as mentioned,it ensures a delay $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ . Consider any moment during Step 2; let $t$ be the running time from the start of Step 1 till that moment. As $t = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ ,we can raise $\beta$ to some $\widetilde{O}\left( 1\right)$ factor to make sure

为了理解这一论断，首先注意到步骤1是 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ -激进的，因为如前所述，它确保了一个延迟 $\widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\mathrm{{OUT}}}\right)$ 。考虑步骤2中的任意时刻；设 $t$ 为从步骤1开始到该时刻的运行时间。由于 $t = \widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ ，我们可以将 $\beta$ 提升到某个 $\widetilde{O}\left( 1\right)$ 因子以确保

$$
\lfloor t/\alpha \rfloor  \leq  \left\lfloor  {\widetilde{O}\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right) /\alpha }\right\rfloor   = \lfloor \mathrm{{OUT}} \cdot  \widetilde{O}\left( 1\right) /\beta \rfloor  \leq  \mathrm{{OUT}}/2
$$

with the value $\alpha$ calculated in (11). To argue that our algorithm is $\alpha$ -aggressive,it suffices to show that at least OUT/2 distinct result tuples must have been found at any moment during Step 2. This is true because Step 1 has output at least OUT/2 tuples.

结合在(11)中计算出的值 $\alpha$ 。要证明我们的算法是 $\alpha$ -激进的，只需证明在步骤2的任何时刻，至少已经找到了OUT/2个不同的结果元组。这是成立的，因为步骤1已经输出了至少OUT/2个元组。

## H JOIN UNION SAMPLING

## H连接并采样

As before,let ${Q}_{1},{Q}_{2},\ldots ,{Q}_{k}$ be the joins given (where constant $k \geq  2$ ),IN be the total input size of all these joins,and OUT : $=$ $\left| {\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {Q}_{i}\right) }\right|$ . For each $i \in  \left\lbrack  {1,k}\right\rbrack$ ,define

和之前一样，设 ${Q}_{1},{Q}_{2},\ldots ,{Q}_{k}$ 为给定的连接（其中常数为 $k \geq  2$ ），IN为所有这些连接的总输入大小，OUT : $=$ $\left| {\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {Q}_{i}\right) }\right|$ 。对于每个 $i \in  \left\lbrack  {1,k}\right\rbrack$ ，定义

- ${\mathcal{G}}_{i} \mathrel{\text{:=}} \left( {{\mathcal{X}}_{i},{\mathcal{E}}_{i}}\right)$ as the schema graph of ${\mathcal{Q}}_{i}$ ;

- ${\mathcal{G}}_{i} \mathrel{\text{:=}} \left( {{\mathcal{X}}_{i},{\mathcal{E}}_{i}}\right)$ 为 ${\mathcal{Q}}_{i}$ 的模式图；

- ${\rho }_{i}^{ * }$ as the fractional edge covering number of ${\mathcal{G}}_{i}$ ;

- ${\rho }_{i}^{ * }$ 为 ${\mathcal{G}}_{i}$ 的分数边覆盖数；

- ${W}_{i}$ as an optimal fractional edge covering of ${\mathcal{G}}_{i}$ ,namely,

- ${W}_{i}$ 为 ${\mathcal{G}}_{i}$ 的最优分数边覆盖，即

$$
{\rho }_{i}^{ * } = \mathop{\sum }\limits_{{e \in  {\mathcal{E}}_{i}}}{W}_{i}\left( e\right) .
$$

Therefore, ${\rho }^{ * } = \mathop{\max }\limits_{{i = 1}}^{k}{\rho }_{i}^{ * }$ . Introduce

因此， ${\rho }^{ * } = \mathop{\max }\limits_{{i = 1}}^{k}{\rho }_{i}^{ * }$ 。引入

$$
\text{ AGMSUM } \mathrel{\text{:=}} \mathop{\sum }\limits_{{i = 1}}^{k}{\operatorname{AGM}}_{{W}_{i}}\left( {Q}_{i}\right) .
$$

It is easy to see that $\mathrm{{AGMSUM}} = O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ .

很容易看出 $\mathrm{{AGMSUM}} = O\left( {\mathrm{{IN}}}^{{\rho }^{ * }}\right)$ 。

A tuple $u \in  \mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ oin $\left( {Q}_{i}\right)$ can be in the result of more than one join among ${Q}_{1},\ldots ,{Q}_{k}$ . For each $\mathbf{u}$ ,we define its owner as the ${Q}_{i}$ with the smallest $i \in  \left\lbrack  {1,k}\right\rbrack$ satisfying $\mathbf{u} \in  \mathcal{J}$ oin $\left( {Q}_{i}\right)$ . Once $\mathbf{u}$ is given, its owner can be easily determined in $O\left( k\right)  = O\left( 1\right)$ time.

一个元组 $u \in  \mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ 属于 $\left( {Q}_{i}\right)$ 可能存在于 ${Q}_{1},\ldots ,{Q}_{k}$ 中多个连接的结果里。对于每个 $\mathbf{u}$ ，我们将其所有者定义为满足 $\mathbf{u} \in  \mathcal{J}$ 属于 $\left( {Q}_{i}\right)$ 且 $i \in  \left\lbrack  {1,k}\right\rbrack$ 最小的 ${Q}_{i}$ 。一旦给定 $\mathbf{u}$ ，就可以在 $O\left( k\right)  = O\left( 1\right)$ 时间内轻松确定其所有者。

For each $i \in  \left\lbrack  {1,k}\right\rbrack$ ,we build a structure ${\Upsilon }_{i}$ of Theorem 5 on ${\mathcal{Q}}_{i}$ , under the fractional edge covering ${W}_{i}$ . Since $k$ is a constant,the space consumption of all the structures is $\widetilde{O}\left( \mathrm{{IN}}\right)$ ,and it is straightforward to handle an update in any input relation using $\widetilde{O}\left( 1\right)$ time.

对于每个 $i \in  \left\lbrack  {1,k}\right\rbrack$ ，我们在 ${\mathcal{Q}}_{i}$ 上基于分数边覆盖 ${W}_{i}$ 构建定理5的结构 ${\Upsilon }_{i}$ 。由于 $k$ 是一个常数，所有结构的空间消耗为 $\widetilde{O}\left( \mathrm{{IN}}\right)$ ，并且使用 $\widetilde{O}\left( 1\right)$ 时间处理任何输入关系的更新是很直接的。

The rest of the section will explain how to extract a sample from $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ oin $\left( {\mathcal{Q}}_{i}\right)$ . Our algorithm,named union-sample,combines our sample algorithm in Figure 3 with ideas from [15]. Algorithm union-sample has the properties below:

本节的其余部分将解释如何从 $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ 属于 $\left( {\mathcal{Q}}_{i}\right)$ 中提取一个样本。我们的算法，名为联合采样（union - sample），将图 3 中的采样算法与文献 [15] 中的思想相结合。联合采样算法具有以下性质：

- It finishes in $\widetilde{O}\left( 1\right)$ time;

- 它在 $\widetilde{O}\left( 1\right)$ 时间内完成；

- It declares "failure" with probability 1 - OUT/AGMSUM.

- 它以 1 - OUT/AGMSUM 的概率宣告“失败”。

- If not declaring failure,it outputs a tuple $\mathbf{u}$ from $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{i}\right)$ uniformly at random.

- 如果不宣告失败，它会从 $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{i}\right)$ 中均匀随机地输出一个元组 $\mathbf{u}$。

The above properties allow us to extract a sample from $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ oin $\left( {\mathcal{Q}}_{i}\right)$ w.h.p. in time $\widetilde{O}\left( {\mathrm{{AGMSUM}}/\max \{ 1,\mathrm{{OUT}}\} }\right)  = \widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ .

上述性质使我们能够以高概率（w.h.p.）在 $\widetilde{O}\left( {\mathrm{{AGMSUM}}/\max \{ 1,\mathrm{{OUT}}\} }\right)  = \widetilde{O}\left( {{\mathrm{{IN}}}^{{\rho }^{ * }}/\max \{ 1,\mathrm{{OUT}}\} }\right)$ 时间内从 $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ 属于 $\left( {\mathcal{Q}}_{i}\right)$ 中提取一个样本。

Next, we present the details of union-sample. It starts by generating a random integer $i \in  \left\lbrack  {1,k}\right\rbrack$ such that

接下来，我们介绍联合采样的详细步骤。它首先生成一个随机整数 $i \in  \left\lbrack  {1,k}\right\rbrack$，使得

$$
\Pr \left\lbrack  {i = j}\right\rbrack   = {\operatorname{AGM}}_{{W}_{j}}\left( {Q}_{j}\right) /\text{ AGMSUM }
$$

holds for each $j \in  \left\lbrack  {1,k}\right\rbrack$ . The generation takes $\widetilde{O}\left( 1\right)$ time,thanks to Proposition 1. Then,the algorithm instructs ${\Upsilon }_{i}$ to execute the sample algorithm (Figure 3) only once in $\widetilde{O}\left( 1\right)$ time. If sample declares "failure", union-sample does the same. Otherwise, sample has obtained a tuple $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( {Q}_{i}\right)$ . Now,check in constant time whether $\mathcal{J}$ oin $\left( {\mathcal{Q}}_{i}\right)$ is the owner of $\mathbf{u}$ . If so,union-sample outputs $\mathbf{u}$ as a sample of $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{i}\right)$ ; otherwise,it declares "failure".

对于每个 $j \in  \left\lbrack  {1,k}\right\rbrack$ 都成立。由于命题 1，生成该随机整数需要 $\widetilde{O}\left( 1\right)$ 时间。然后，该算法指示 ${\Upsilon }_{i}$ 在 $\widetilde{O}\left( 1\right)$ 时间内仅执行一次采样算法（图 3）。如果采样算法宣告“失败”，联合采样算法也宣告“失败”。否则，采样算法得到一个元组 $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( {Q}_{i}\right)$。现在，在常数时间内检查 $\mathcal{J}$ 属于 $\left( {\mathcal{Q}}_{i}\right)$ 是否是 $\mathbf{u}$ 的所有者。如果是，联合采样算法输出 $\mathbf{u}$ 作为 $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{i}\right)$ 的一个样本；否则，它宣告“失败”。

To analyze the algorithm,consider any tuple $\mathbf{u} \in  \mathop{\bigcup }\limits_{{i = 1}}^{k}\operatorname{Join}\left( {\mathcal{Q}}_{i}\right)$ . Assume,w.l.o.g.,that ${Q}_{{i}^{ * }}$ is the owner of $\mathbf{u}$ for some ${i}^{ * } \in  \left\lbrack  {1,k}\right\rbrack$ . As union-sample can return $\mathbf{u}$ only when the random variable $i$ selected in the beginning equals ${i}^{ * }$ ,the probability of outputting $\mathbf{u}$ equals

为了分析该算法，考虑任意元组 $\mathbf{u} \in  \mathop{\bigcup }\limits_{{i = 1}}^{k}\operatorname{Join}\left( {\mathcal{Q}}_{i}\right)$。不失一般性（w.l.o.g.），假设对于某个 ${i}^{ * } \in  \left\lbrack  {1,k}\right\rbrack$，${Q}_{{i}^{ * }}$ 是 $\mathbf{u}$ 的所有者。由于联合采样算法仅在开始时选择的随机变量 $i$ 等于 ${i}^{ * }$ 时才能返回 $\mathbf{u}$，输出 $\mathbf{u}$ 的概率等于

$\frac{{\operatorname{AGM}}_{{W}_{{i}^{ * }}}\left( {\mathcal{Q}}_{{i}^{ * }}\right) }{\text{ AGMSUM }} \cdot  \Pr \left\lbrack  {\text{the sample algorithm of }{\Upsilon }_{{i}^{ * }}\text{ samples }\mathbf{u}}\right\rbrack  .\;\left( {12}\right)$ As explained in Section 4.2, ${\Upsilon }_{{i}^{ * }}$ samples $\mathbf{u}$ with probability $\frac{1}{{\operatorname{AGM}}_{{W}_{{i}^{ * }}}\left( {Q}_{{i}^{ * }}\right) }$ . Therefore, (12) can be simplified into

$\frac{{\operatorname{AGM}}_{{W}_{{i}^{ * }}}\left( {\mathcal{Q}}_{{i}^{ * }}\right) }{\text{ AGMSUM }} \cdot  \Pr \left\lbrack  {\text{the sample algorithm of }{\Upsilon }_{{i}^{ * }}\text{ samples }\mathbf{u}}\right\rbrack  .\;\left( {12}\right)$ 如 4.2 节所述，${\Upsilon }_{{i}^{ * }}$ 以 $\frac{1}{{\operatorname{AGM}}_{{W}_{{i}^{ * }}}\left( {Q}_{{i}^{ * }}\right) }$ 的概率对 $\mathbf{u}$ 进行采样。因此，(12) 可以简化为

$$
\frac{{\operatorname{AGM}}_{{W}_{{i}^{ * }}}\left( {\mathcal{Q}}_{{i}^{ * }}\right) }{\text{ AGMSUM }} \cdot  \frac{1}{{\operatorname{AGM}}_{{W}_{{i}^{ * }}}\left( {\mathcal{Q}}_{{i}^{ * }}\right) } = \frac{1}{\text{ AGMSUM }}.
$$

We thus conclude that union-sample returns a uniformly random tuple of $\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$ oin $\left( {\mathcal{Q}}_{i}\right)$ with probability OUT/AGMSUM,and declares "failure" with probability 1 - OUT/AGMSUM.

因此，我们得出结论：联合采样（union - sample）以概率OUT/AGMSUM返回一个来自$\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$（其中$\mathop{\bigcup }\limits_{{i = 1}}^{k}\mathcal{J}$属于$\left( {\mathcal{Q}}_{i}\right)$）的均匀随机元组，并以概率1 - OUT/AGMSUM宣告“失败”。

---

<!-- Footnote -->

${}^{10}$ Recall that "running time" in the RAM model is defined as the number of atomic operations, each of which performs constant-time work such as comparison, register assignment, arithmetic computation, memory access, etc.

${}^{10}$ 回顾一下，随机存取机（RAM）模型中的“运行时间”定义为原子操作的数量，每个原子操作执行诸如比较、寄存器赋值、算术计算、内存访问等常量时间的工作。

<!-- Footnote -->

---