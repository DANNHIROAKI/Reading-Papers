# Output-Optimal Massively Parallel Algorithms for Similarity Joins

# 相似度连接的输出最优大规模并行算法

XIAO HU and KE YI, Hong Kong University of Science and Technology YUFEI TAO, The Chinese University of Hong Kong

胡潇、易柯，香港科技大学；陶宇飞，香港中文大学

Parallel join algorithms have received much attention in recent years due to the rapid development of massively parallel systems such as MapReduce and Spark. In the database theory community, most efforts have been focused on studying worst-case optimal algorithms. However, the worst-case optimality of these join algorithms relies on the hard instances having very large output sizes. In the case of a two-relation join, the hard instance is just a Cartesian product, with an output size that is quadratic in the input size.

近年来，由于MapReduce和Spark等大规模并行系统的快速发展，并行连接算法受到了广泛关注。在数据库理论界，大多数研究都集中在最坏情况最优算法上。然而，这些连接算法的最坏情况最优性依赖于具有非常大输出规模的困难实例。在双关系连接的情况下，困难实例就是笛卡尔积，其输出规模是输入规模的二次方。

In practice, however, the output size is usually much smaller. One recent parallel join algorithm by Beame et al. has achieved output-optimality (i.e., its cost is optimal in terms of both the input size and the output size), but their algorithm only works for a 2-relation equi-join and has some imperfections. In this article, we first improve their algorithm to true optimality. Then we design output-optimal algorithms for a large class of similarity joins. Finally, we present a lower bound, which essentially eliminates the possibility of having output-optimal algorithms for any join on more than two relations.

然而，在实际应用中，输出规模通常要小得多。Beame等人最近提出的一种并行连接算法实现了输出最优性（即，其成本在输入规模和输出规模方面都是最优的），但他们的算法仅适用于双关系等值连接，并且存在一些缺陷。在本文中，我们首先将他们的算法改进为真正的最优算法。然后，我们为一大类相似度连接设计了输出最优算法。最后，我们给出了一个下界，这实际上排除了对两个以上关系的任何连接存在输出最优算法的可能性。

CCS Concepts: - Theory of computation $\rightarrow$ Database query processing and optimization (theory);

CCS概念： - 计算理论 $\rightarrow$ 数据库查询处理与优化（理论）；

Additional Key Words and Phrases: Parallel computation, similarity joins, output-sensitive algorithms

其他关键词和短语：并行计算、相似度连接、输出敏感算法

## ACM Reference format:

## ACM引用格式：

Xiao Hu, Ke Yi, and Yufei Tao. 2019. Output-Optimal Massively Parallel Algorithms for Similarity Joins. ACM Trans. Database Syst. 44, 2, Article 6 (March 2019), 36 pages.

胡潇、易柯和陶宇飞。2019年。相似度连接的输出最优大规模并行算法。ACM数据库系统汇刊44卷，第2期，文章编号6（2019年3月），36页。

https://doi.org/10.1145/3311967

## 1 INTRODUCTION

## 1 引言

The similarity join problem is perhaps one of the most extensively studied problems in the database and data mining literature. Numerous variants exist, depending on the metric space and the distance function used. Let $\operatorname{dist}\left( {\cdot , \cdot  }\right)$ be a distance function. Given two point sets ${R}_{1}$ and ${R}_{2}$ and a threshold $r \geq  0$ ,the similarity join problem asks to find all pairs of points $x \in  {R}_{1},y \in  {R}_{2}$ ,such that $\operatorname{dist}\left( {x,y}\right)  \leq  r$ . In this article,we will be mostly interested in the ${\ell }_{1},{\ell }_{2}$ ,and ${\ell }_{\infty }$ distances,although some of our results (the one based on locality-sensitive hashing (LSH)) can be extended to other distance functions as well.

相似度连接问题可能是数据库和数据挖掘文献中研究最广泛的问题之一。根据所使用的度量空间和距离函数，存在许多变体。设 $\operatorname{dist}\left( {\cdot , \cdot  }\right)$ 为距离函数。给定两个点集 ${R}_{1}$ 和 ${R}_{2}$ 以及一个阈值 $r \geq  0$ ，相似度连接问题要求找出所有点对 $x \in  {R}_{1},y \in  {R}_{2}$ ，使得 $\operatorname{dist}\left( {x,y}\right)  \leq  r$ 。在本文中，我们主要关注 ${\ell }_{1},{\ell }_{2}$ 和 ${\ell }_{\infty }$ 距离，尽管我们的一些结果（基于局部敏感哈希（LSH）的结果）也可以扩展到其他距离函数。

---

<!-- Footnote -->

The first two authors were supported by Hong Kong RGC under grants 16200415, 16202317, and 16201318, and by grants from Microsoft and Alibaba. Y. Tao was partially supported by a direct grant (4055079) from The Chinese University of Hong Kong and by a Faculty Research Award from Google.

前两位作者得到了香港研究资助局（RGC）的资助（项目编号16200415、16202317和16201318），以及微软和阿里巴巴的资助。陶宇飞部分得到了香港中文大学的直接资助（项目编号4055079）和谷歌的教师研究奖的支持。

Authors' addresses: X. Hu and K. Yi, Department of Computer Science and Engineering, Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong, China; emails: \{xhuam, yike\}@cse.ust.hk; Y. Tao, Department of Computer Science and Engineering, The Chinese University of Hong Kong, Shatin, Hong Kong, China; email: taoyf@cse.cuhk.edu.hk.Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. © 2019 Association for Computing Machinery. 0362-5915/2019/03-ART6 \$15.00 https://doi.org/10.1145/3311967

作者地址：胡潇和易柯，香港科技大学计算机科学与工程系，中国香港清水湾；电子邮件：\{xhuam, yike\}@cse.ust.hk；陶宇飞，香港中文大学计算机科学与工程系，中国香港沙田；电子邮件：taoyf@cse.cuhk.edu.hk。允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是不得为盈利或商业利益制作或分发拷贝，并且拷贝必须带有此通知和第一页的完整引用。必须尊重本作品中除ACM之外的其他所有者的版权。允许进行带引用的摘要。否则，如需复制、重新发布、在服务器上发布或分发给列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。© 2019美国计算机协会。0362 - 5915/2019/03 - ART6 15.00美元https://doi.org/10.1145/3311967

<!-- Footnote -->

---

### 1.1 The Computation Model

### 1.1 计算模型

Driven by the rapid development of massively parallel systems such as MapReduce [14], Spark [33], and many other systems that adopt very similar architectures, there have also been resurrected interests in the theoretical computer science community to study algorithms in such massively parallel models. One popular model that has often been used to study join algorithms in particular is the massively parallel computation (MPC) model [1-3, 7, 8, 22-24].

受MapReduce [14]、Spark [33]等大规模并行系统以及许多采用非常相似架构的其他系统的快速发展的推动，理论计算机科学界也重新燃起了在这种大规模并行模型中研究算法的兴趣。一个特别常用于研究连接算法的流行模型是大规模并行计算（MPC）模型 [1 - 3, 7, 8, 22 - 24]。

In the MPC model,data is initially partitioned arbitrarily across $p$ servers that are connected by a complete network. Computation proceeds in rounds. In each round, each server first sends messages to other servers. After all messages have arrived at their destinations, the servers conduct some local computation in parallel and proceed to the next round. The complexity of the algorithm is measured first by the number of rounds,then the load,denoted as $L$ ,which is the maximum message size received by any server in any round. Initial efforts were mostly spent on understanding what can be done in a single round of computation $\left\lbrack  {2,7,8,{23},{24}}\right\rbrack$ ; however,recently, more interests have been given to multi-round (but still a constant) algorithms [1, 3, 22, 23], as new main memory based systems, such as Spark, tend to have much lower overhead per round than previous systems like Hadoop. Meanwhile, this puts more emphasis on minimizing the load, to ensure that the local memory at each server is never exceeded.

在多方计算（MPC）模型中，数据最初会任意划分到通过全连接网络相连的 $p$ 台服务器上。计算按轮次进行。在每一轮中，每台服务器首先向其他服务器发送消息。所有消息到达目的地后，服务器并行进行一些本地计算，然后进入下一轮。该算法的复杂度首先由轮数衡量，然后是负载（用 $L$ 表示），即任何一轮中任何服务器接收到的最大消息大小。最初的工作主要集中在理解一轮计算 $\left\lbrack  {2,7,8,{23},{24}}\right\rbrack$ 中可以完成什么任务；然而，最近，人们对多轮（但轮数仍为常数）算法 [1, 3, 22, 23] 更感兴趣，因为像 Spark 这样基于主内存的新系统，每一轮的开销往往比 Hadoop 等先前的系统低得多。同时，这更强调要尽量减少负载，以确保不会超出每台服务器的本地内存。

Note that the MPC model is essentially Valiant's bulk synchronous processing (BSP) model [32] without restricting the outgoing messages. Theoretically, this further simplifies the model. The practical justification is that the routing of outgoing messages overlaps with the computation phase, making it negligible. Furthermore, the incoming message size also indirectly determines the memory requirement and local computation time on each server in the following round, which is why this measure is referred to as "load."

请注意，MPC 模型本质上是瓦利安特（Valiant）的批量同步处理（BSP）模型 [32]，但不限制传出消息。从理论上讲，这进一步简化了模型。实际的理由是，传出消息的路由与计算阶段重叠，因此可以忽略不计。此外，传入消息的大小也间接决定了下一轮中每台服务器的内存需求和本地计算时间，这就是为什么这个指标被称为“负载”。

Let ${N}_{1}$ and ${N}_{2}$ be the sizes of the two relations to be joined,and let $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ . In this article, we will focus on the case when $1/p \leq  {N}_{1}/{N}_{2} \leq  p$ . This is because when ${N}_{1}/{N}_{2}$ falls outside this range, the trivial algorithm that broadcasts the smaller relation to all servers would be optimal, as shown in Section 3.2.

设 ${N}_{1}$ 和 ${N}_{2}$ 为要进行连接的两个关系的大小，并设 $\mathrm{{IN}} = {N}_{1} + {N}_{2}$。在本文中，我们将重点关注 $1/p \leq  {N}_{1}/{N}_{2} \leq  p$ 的情况。这是因为当 ${N}_{1}/{N}_{2}$ 超出这个范围时，将较小的关系广播到所有服务器的简单算法将是最优的，如第 3.2 节所示。

### 1.2 Previous Join Algorithms in the MPC Model

### 1.2 MPC 模型中的先前连接算法

All prior work on join algorithms in the MPC model has focused on equi-joins and has mostly been concerned with the worst case. Notably, the hypercube algorithm [2] computes the equi-join between two relations with load $L = \widetilde{O}\left( \sqrt{{N}_{1}{N}_{2}/p}\right) .{}^{1}$ This load is optimal in the worst case,as the output size can be as large as ${N}_{1}{N}_{2}$ ,when all tuples share the same join key and the join degenerates into a Cartesian product. Since each server can only produce $O\left( {L}^{2}\right)$ join results in a round ${}^{2}$ if the load is limited to $L$ ,all the $p$ servers can produce at most $O\left( {p{L}^{2}}\right)$ join results in a constant number of rounds. Thus,producing ${N}_{1}{N}_{2}$ results needs at least a load of $L = \Omega \left( \sqrt{{N}_{1}{N}_{2}/p}\right)$ . Note that this lower bound argument is assuming tuple-based join algorithms-that is, the tuples are atomic elements that must be processed and communicated in their entirety. They can be copied but cannot be broken up or manipulated with bit tricks. To produce a join result, all tuples (or their copies) that make up the join result must reside at the same server when the join result is output. However, the server does not have to do any further processing with the result, such as sending it to another server. The same model has also been used in other works [7, 8, 23].

MPC 模型中所有关于连接算法的先前工作都集中在等值连接上，并且大多关注最坏情况。值得注意的是，超立方体算法 [2] 计算两个关系之间的等值连接，负载为 $L = \widetilde{O}\left( \sqrt{{N}_{1}{N}_{2}/p}\right) .{}^{1}$。在最坏情况下，这个负载是最优的，因为当所有元组共享相同的连接键且连接退化为笛卡尔积时，输出大小可能高达 ${N}_{1}{N}_{2}$。由于如果负载限制为 $L$，每台服务器在一轮 ${}^{2}$ 中最多只能产生 $O\left( {L}^{2}\right)$ 个连接结果，那么所有 $p$ 台服务器在常数轮数内最多能产生 $O\left( {p{L}^{2}}\right)$ 个连接结果。因此，产生 ${N}_{1}{N}_{2}$ 个结果至少需要 $L = \Omega \left( \sqrt{{N}_{1}{N}_{2}/p}\right)$ 的负载。请注意，这个下界论证假设的是基于元组的连接算法，即元组是必须作为整体进行处理和通信的原子元素。它们可以被复制，但不能被拆分或使用位操作技巧进行处理。要产生一个连接结果，组成该连接结果的所有元组（或其副本）在输出连接结果时必须位于同一台服务器上。然而，服务器不必对结果进行任何进一步的处理，例如将其发送到另一台服务器。其他工作 [7, 8, 23] 也使用了相同的模型。

---

<!-- Footnote -->

${}^{1}$ The $\widetilde{O}$ notation suppresses polylogarithmic factors.

${}^{1}$ $\widetilde{O}$ 符号忽略了多项式对数因子。

${}^{2}$ Technically,this is true under the condition $L = \Omega \left( {\mathrm{{IN}}/p}\right)$ ,but as will be proved in this article,this condition indeed holds even just to decide whether the join result is empty.

${}^{2}$ 从技术上讲，这在条件 $L = \Omega \left( {\mathrm{{IN}}/p}\right)$ 下成立，但正如本文将证明的，即使只是为了确定连接结果是否为空，这个条件实际上也成立。

<!-- Footnote -->

---

However, on most realistic datasets, the join size is nowhere near the worst case. Suppose that the join size is OUT. Applying the same argument as earlier, one would hope to get a load of $\widetilde{O}\left( \sqrt{\mathrm{{OUT}}/p}\right)$ . Such a bound would be output-optimal. Of course,this is not entirely possible,as OUT can even be zero,so a more reasonable target would be $L = \widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ ,where $\mathrm{{IN}} =$ ${N}_{1} + {N}_{2}$ is the total input size. This is exactly the goal of this work,although in some cases we have not achieved this ideal input-dependent term $\widetilde{O}\left( {\mathrm{{IN}}/p}\right)$ exactly. Note that we are still doing worst-case analysis-that is, we do not make any assumptions on the input data and how it is distributed on the $p$ servers initially. We merely use OUT as an additional parameter to measure the complexity of the algorithm.

然而，在大多数实际数据集上，连接大小远未达到最坏情况。假设连接大小为 OUT。采用与之前相同的论证，我们希望负载为 $\widetilde{O}\left( \sqrt{\mathrm{{OUT}}/p}\right)$。这样的界限将是输出最优的。当然，这并非完全可行，因为 OUT 甚至可能为零，所以更合理的目标是 $L = \widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$，其中 $\mathrm{{IN}} =$ ${N}_{1} + {N}_{2}$ 是总输入大小。这正是本工作的目标，尽管在某些情况下我们并未精确实现这个理想的依赖输入的项 $\widetilde{O}\left( {\mathrm{{IN}}/p}\right)$。请注意，我们仍然在进行最坏情况分析，即我们不对输入数据以及它最初在 $p$ 台服务器上的分布情况做任何假设。我们只是将 OUT 作为一个额外的参数来衡量算法的复杂度。

There are some previous join algorithms that use both IN and OUT to measure the complexity. Afrati et al. [1] gave an algorithm with load $O\left( {{\mathrm{{IN}}}^{w}/\sqrt{p} + \mathrm{{OUT}}/\sqrt{p}}\right)$ ,where $w$ is the width of the join query, which is 1 for any acyclic query, including a 2-relation join. However, both terms $O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$ or $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ are far from optimal. Beame et al. [8] proposed a randomized algorithm with optimal load $\widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ ,but up to logarithmic factors,due to the use of random hashing. They also assume that each server knows the data statistics in advance.

之前有一些连接算法同时使用 IN 和 OUT 来衡量复杂度。阿夫拉蒂等人 [1] 提出了一种负载为 $O\left( {{\mathrm{{IN}}}^{w}/\sqrt{p} + \mathrm{{OUT}}/\sqrt{p}}\right)$ 的算法，其中 $w$ 是连接查询的宽度，对于任何无环查询（包括二元关系连接），其值为 1。然而，$O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$ 或 $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ 这两项都远非最优。比姆等人 [8] 提出了一种随机算法，其最优负载为 $\widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$，但由于使用了随机哈希，存在对数因子。他们还假设每个服务器事先知道数据统计信息。

Note that equi-join is a special case of similarity joins with $r = 0$ . There are previously no algorithms in the MPC model for similarity joins with $r > 0$ ,except computing the full Cartesian product of the two relations with load $O\left( \sqrt{{N}_{1}{N}_{2}/p}\right)$ ,which is not output-optimal.

请注意，等值连接是相似度连接在 $r = 0$ 情况下的特殊情况。在 MPC 模型中，之前没有针对 $r > 0$ 的相似度连接的算法，除了计算两个关系的全笛卡尔积，其负载为 $O\left( \sqrt{{N}_{1}{N}_{2}/p}\right)$，这并非输出最优的。

As a remark, there exists a general reduction [23] that converts MPC join algorithms into I/O-efficient counterparts under the enumerate version [29] of the external memory model [4], where each result tuple only needs to be seen in memory, as opposed to being reported in the disk. A nice application of the reduction has been demonstrated for the triangle enumeration problem, where an MPC algorithm [23] is shown to imply an EM algorithm matching the I/O lower bound of Pagh and Silvestri [29] up to a logarithmic factor.

值得一提的是，存在一种通用的归约方法 [23]，它可以将 MPC 连接算法转换为外部内存模型 [4] 的枚举版本 [29] 下的 I/O 高效算法，在该版本中，每个结果元组只需在内存中被查看，而无需在磁盘上报告。这种归约方法在三角形枚举问题上有一个很好的应用，其中一个 MPC 算法 [23] 被证明可以推导出一个 EM 算法，该算法在对数因子范围内与帕格和西尔维斯特里 [29] 的 I/O 下界相匹配。

### 1.3 Our Results

### 1.3 我们的成果

We start with an improved algorithm for computing the equi-join between two relations-for instance,a degenerated similarity join with $r = 0$ . We improve upon the algorithm of Beame et al. [8] in the following aspects. First, our algorithm does not assume any prior statistical information about the data, such as the heavy join values and their frequencies. Second, the load of our algorithm is exactly $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ tuples,without any extra logarithmic factors. Third,our algorithm is deterministic. The only price we pay is that the number of rounds increases from 1 to $O\left( 1\right)$ . This algorithm is described in Section 3.

我们首先提出一种改进的算法，用于计算两个关系之间的等值连接，例如，$r = 0$ 情况下的退化相似度连接。我们在以下几个方面对比姆等人 [8] 的算法进行了改进。首先，我们的算法不假设对数据有任何先验统计信息，例如重连接值及其频率。其次，我们算法的负载恰好为 $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ 个元组，没有任何额外的对数因子。第三，我们的算法是确定性的。我们唯一付出的代价是轮数从 1 增加到 $O\left( 1\right)$。该算法将在第 3 节中描述。

Although the $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ term is optimal by the preceding tuple-based argument,prior work did not show why the input-dependent term $O\left( {\mathrm{{IN}}/p}\right)$ is necessary. Note that if OUT is not a parameter, the worst-case input is always when the output size is maximized, i.e., a full Cartesian product for 2-relation joins or the AGM bound [6] for multi-way joins). In this case, the preceding simple tuple-based argument already leads to a lower bound higher than $\Omega \left( {\mathrm{{IN}}/p}\right)$ ,so this is not an issue. However, when the output size OUT becomes a parameter, these worst-case constructions do not work anymore,and it is not clear why $O\left( {\mathrm{{IN}}/p}\right)$ load is necessary. Indeed,if OUT = 1,then the preceding tuple-based argument yields a meaningless lower bound of $\Omega \left( {1/p}\right)$ . To complete the picture,we provide a lower bound showing that even if $\mathrm{{OUT}} = O\left( 1\right)$ ,computing the equi-join between two relations requires $\Omega \left( {\mathrm{{IN}}/p}\right)$ load,by resorting to strong results from communication complexity.

尽管通过前面基于元组的论证可知$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$项是最优的，但先前的工作并未说明为什么依赖输入的项$O\left( {\mathrm{{IN}}/p}\right)$是必要的。请注意，如果输出大小OUT不是一个参数，那么最坏情况的输入总是在输出大小达到最大时出现，即对于二元关系连接是全笛卡尔积，对于多路连接是AGM边界[6]。在这种情况下，前面简单的基于元组的论证已经得出了一个高于$\Omega \left( {\mathrm{{IN}}/p}\right)$的下界，所以这不是问题。然而，当输出大小OUT成为一个参数时，这些最坏情况的构造就不再适用了，而且不清楚为什么需要$O\left( {\mathrm{{IN}}/p}\right)$的负载。实际上，如果OUT = 1，那么前面基于元组的论证会得出一个无意义的下界$\Omega \left( {1/p}\right)$。为了完善这一情况，我们给出了一个下界，表明即使$\mathrm{{OUT}} = O\left( 1\right)$，通过借助通信复杂度的强大结果，计算两个关系之间的等值连接仍需要$\Omega \left( {\mathrm{{IN}}/p}\right)$的负载。

The main theoretical results in this article,however,are on similarity joins with $r > 0$ . Specifically, we obtain the following results under various distance metrics:

然而，本文的主要理论结果是关于具有$r > 0$的相似性连接。具体来说，我们在各种距离度量下得到了以下结果：

(1) For ${\ell }_{1}/{\ell }_{\infty }$ distance in constant dimensions,we give a deterministic algorithm with load

(1) 对于恒定维度下的${\ell }_{1}/{\ell }_{\infty }$距离，我们给出了一个负载为的确定性算法

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p} \cdot  {\log }^{O\left( 1\right) }p}\right) 
$$

-that is, the output-dependent term is optimal, whereas the input-dependent term is away from optimality by a polylogarithmic factor, which depends on the dimensionality.

也就是说，依赖输出的项是最优的，而依赖输入的项与最优值相差一个多项式对数因子，该因子取决于维度。

(2) For ${\ell }_{2}$ distance in $d$ dimensions,we give a randomized algorithm with load

(2) 对于$d$维度下的${\ell }_{2}$距离，我们给出了一个负载为的随机算法

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}} + {p}^{\frac{d}{{2d} - 1}}\log p}\right) .
$$

Again,the term $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ is output-optimal. The input-dependent term $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ is worse than the ${\ell }_{1}/{\ell }_{\infty }$ case due to the non-orthogonal nature of the ${\ell }_{2}$ metric,but it is always better than $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ ,which is the load for computing the full Cartesian product.

同样，项$O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$在输出方面是最优的。由于${\ell }_{2}$度量的非正交性质，依赖输入的项$O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$比${\ell }_{1}/{\ell }_{\infty }$的情况更差，但它总是优于$O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$，即计算全笛卡尔积的负载。

(3) In high dimensions, we provide an algorithm based on LSH with load

(3) 在高维情况下，我们提供了一个基于局部敏感哈希（LSH）的负载为的算法

$$
\widetilde{O}\left( {\sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}} + \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}} + \frac{\mathrm{{IN}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right) ,
$$

where $\mathrm{{OUT}}\left( {cr}\right)$ is the output size if the distance threshold is enlarged to ${cr}$ for some constant $c > 1$ ,and $0 < \rho  < 1$ is the quality measure of the hash function used,which depends only on $c$ and the distance function. Similarly,the term $O\left( {\mathrm{{IN}}/{p}^{1/\left( {1 + \rho }\right) }}\right)$ is always better than that for computing the Cartesian product, although output-optimality here is not only with respect to OUT but also OUT(cr), due to the approximation nature of LSH.

其中，如果距离阈值对于某个常数$c > 1$增大到${cr}$，则$\mathrm{{OUT}}\left( {cr}\right)$是输出大小，并且$0 < \rho  < 1$是所使用的哈希函数的质量度量，它仅取决于$c$和距离函数。类似地，项$O\left( {\mathrm{{IN}}/{p}^{1/\left( {1 + \rho }\right) }}\right)$总是优于计算笛卡尔积的情况，尽管由于LSH的近似性质，这里的输出最优性不仅相对于OUT，还相对于OUT(cr)。

All the algorithms run in $O\left( 1\right)$ rounds,under the mild assumption $\operatorname{IN} > {p}^{1 + \epsilon }$ ,where $\epsilon  > 0$ is any small constant. Note that the randomized output-optimal algorithm in Beame et al. [8] for equi-joins has an implicit assumption that $\operatorname{IN} \geq  {p}^{2}$ ,as there are $\Theta \left( p\right)$ heavy join values,so each server has load at least $\Omega \left( p\right)$ to store these values and their frequencies. We acknowledge that in practice, $\mathrm{{IN}} \geq  {p}^{2}$ is a very reasonable assumption. Our desire to relax this to $\mathrm{{IN}} > {p}^{1 + \epsilon }$ is more from a theoretical point of view, namely achieving the minimum requirement for solving these problem in $O\left( 1\right)$ rounds and optimal load. Indeed,Goodrich [16] has shown that if $\mathrm{{IN}} = {p}^{1 + o\left( 1\right) }$ , then even computing the "or" of IN bits requires $\omega \left( 1\right)$ rounds under load $O\left( {\mathrm{{IN}}/p}\right)$ .

在温和假设$\operatorname{IN} > {p}^{1 + \epsilon }$下，所有算法都在$O\left( 1\right)$轮内运行，其中$\epsilon  > 0$是任意小的常数。请注意，Beame等人[8]中用于等值连接的随机输出最优算法有一个隐含假设，即$\operatorname{IN} \geq  {p}^{2}$，因为有$\Theta \left( p\right)$个重连接值，所以每个服务器至少需要$\Omega \left( p\right)$的负载来存储这些值及其频率。我们承认，在实践中，$\mathrm{{IN}} \geq  {p}^{2}$是一个非常合理的假设。我们希望将此放宽到$\mathrm{{IN}} > {p}^{1 + \epsilon }$更多是从理论角度出发，即在$O\left( 1\right)$轮内解决这些问题并实现最优负载的最低要求。实际上，Goodrich[16]已经证明，如果$\mathrm{{IN}} = {p}^{1 + o\left( 1\right) }$，那么即使计算IN位的“或”运算在负载$O\left( {\mathrm{{IN}}/p}\right)$下也需要$\omega \left( 1\right)$轮。

Finally, we turn to multi-way joins. The only known multi-way equi-join algorithm in the MPC model that has a term related to OUT is the algorithm in Afrati et al. [1] mentioned in Section 1.2. However,that term is $O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$ ,which is almost quadratically larger than the output-optimal term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ that we achieved earlier. We show that,unfortunately,such an output-optimal term is not achievable for a simple multi-way equi-join, a 3-relation chain join ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)  \bowtie  {R}_{3}\left( {C,D}\right)$ . More precisely,in Section 7,we show that if any tuple-based algorithm computing this join has a load in the form of

最后，我们转向多路连接。在MPC模型中，唯一已知的与输出（OUT）相关的多路等值连接算法是第1.2节中提到的Afrati等人[1]的算法。然而，该算法中的相关项是$O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$，这比我们之前实现的输出最优项$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$几乎大了一个数量级。我们遗憾地发现，对于一个简单的多路等值连接，即三关系链连接${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)  \bowtie  {R}_{3}\left( {C,D}\right)$，这样的输出最优项是无法实现的。更准确地说，在第7节中，我们证明了如果任何基于元组的算法计算此连接的负载形式为

$$
L = O\left( {\frac{\mathrm{{IN}}}{{p}^{\alpha }} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right) ,
$$

for some constant $\alpha$ ,then we must have $\alpha  \leq  1/2$ ,provided $\operatorname{IN}{\log }^{2}\operatorname{IN} = \Omega \left( {p}^{3}\right)$ . However,the algorithm in Beame et al. [23] can already compute any 3-relation chain join with $\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ load. This means that it is meaningless to introduce the output-dependent term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ .

对于某个常数$\alpha$，那么在$\operatorname{IN}{\log }^{2}\operatorname{IN} = \Omega \left( {p}^{3}\right)$的条件下，我们必须有$\alpha  \leq  1/2$。然而，Beame等人[23]的算法已经可以以$\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right)$的负载计算任何三关系链连接。这意味着引入依赖于输出的项$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$是没有意义的。

The conference version of this article [19] is mostly concerned with theoretical optimality. In this extended article, we also look at the practical side of the problem. One of the most serious issues in implementing the theoretically optimal MPC algorithm in practice is the large, although still constant, number of rounds. In the current distributed data processing systems, each round incurs a substantial synchronization overhead, and the benefit of an asymptotic smaller load can be easily offset by the large system overhead. To more precisely model the cost of each synchronous round, we classify them into heavy rounds and light rounds (Section 8), where a round is light if its load is $\widetilde{O}\left( p\right)$ and heavy otherwise. Since $p \ll  L$ in practice,it is desirable to minimize the number of heavy rounds while being a bit more tolerant on the light rounds. Then, we design a practical version of our equi-join algorithm and the 1D similarity join algorithm so that both run in a single heavy round and a constant number of light rounds. We have implemented these algorithms in Spark and conducted experiments comparing with other techniques. For equi-join, we have also implemented the hash-based output-optimal algorithm of Beame et al. [8], which had not been implemented before. In Section 9, the experimental results suggest that the output-optimal equi-join algorithms can significantly outperform the vanilla join algorithm in Spark, which is based on the simple hash join. We have also conducted experiments for similarity joins on high-dimensional data. The experimental results show that our LSH-based algorithm yields the best performance among a number of alternatives.

本文会议版本[19]主要关注理论最优性。在这篇扩展文章中，我们还将探讨该问题的实际方面。在实际中实现理论最优的模型预测控制（MPC）算法时，最严重的问题之一是轮数虽然固定但数量较多。在当前的分布式数据处理系统中，每一轮都会产生大量的同步开销，而渐近更小负载带来的好处很容易被巨大的系统开销所抵消。为了更精确地对每一轮同步的成本进行建模，我们将其分为重轮次和轻轮次（第8节），如果一轮的负载为$\widetilde{O}\left( p\right)$，则为轻轮次，否则为重轮次。由于在实际中$p \ll  L$，因此希望在对轻轮次稍微宽容一些的同时，尽量减少重轮次的数量。然后，我们设计了等值连接算法和一维相似性连接算法的实用版本，使这两种算法都能在单个重轮次和固定数量的轻轮次内运行。我们已经在Spark中实现了这些算法，并与其他技术进行了实验比较。对于等值连接，我们还实现了Beame等人[8]提出的基于哈希的输出最优算法，该算法此前尚未被实现过。在第9节中，实验结果表明，输出最优的等值连接算法的性能明显优于Spark中基于简单哈希连接的普通连接算法。我们还对高维数据的相似性连接进行了实验。实验结果表明，我们基于局部敏感哈希（LSH）的算法在多种可选算法中表现最佳。

## 2 PRELIMINARIES

## 2 预备知识

In this section, we first define the similarity join problem under different distance metrics, and their reduction to various geometric containment problems, which will be focused upon in the rest of the article. Then we introduce several primitive operations in the MPC model, which will be used by our algorithms as building blocks.

在本节中，我们首先定义不同距离度量下的相似性连接问题，以及它们如何转化为各种几何包含问题，本文的其余部分将重点讨论这些问题。然后，我们介绍模型预测控制（MPC）模型中的几个基本操作，这些操作将作为我们算法的构建模块。

### 2.1 Similarity Joins

### 2.1 相似性连接

Similarity joins under $r = 0$ are equi-joins,where two points can be joined if and only if they are equal. For $r > 0$ ,a similarity join is better interpreted geometrically. We model input tuples as points in ${\mathbb{R}}^{d}$ ,and let ${R}_{1},{R}_{2} \subset  {\mathbb{R}}^{d}$ be two sets of points with $\left| {R}_{1}\right|  = {N}_{1},\left| {R}_{2}\right|  = {N}_{2}$ . For two points $x = \left( {{x}_{1},{x}_{2},\ldots ,{x}_{d}}\right)  \in  {R}_{1},y = \left( {{y}_{1},{y}_{2},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ ,their distance under the ${\ell }_{k}$ norm is

$r = 0$下的相似性连接是等值连接，即当且仅当两个点相等时，它们才能进行连接。对于$r > 0$，相似性连接从几何角度更容易解释。我们将输入元组建模为${\mathbb{R}}^{d}$中的点，并设${R}_{1},{R}_{2} \subset  {\mathbb{R}}^{d}$是两个点集，其中$\left| {R}_{1}\right|  = {N}_{1},\left| {R}_{2}\right|  = {N}_{2}$。对于两个点$x = \left( {{x}_{1},{x}_{2},\ldots ,{x}_{d}}\right)  \in  {R}_{1},y = \left( {{y}_{1},{y}_{2},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$，它们在${\ell }_{k}$范数下的距离为

$$
\parallel x - y{\parallel }_{k} = {\left( \mathop{\sum }\limits_{{i = 1}}^{d}{\left| {x}_{i} - {y}_{i}\right| }^{k}\right) }^{1/k}.
$$

The most commonly used ${\ell }_{k}$ norms are the ${\ell }_{1}$ -norm, ${\ell }_{2}$ -norm,and ${\ell }_{\infty }$ norm. Note that for $k = \infty$ , $\parallel x - y{\parallel }_{\infty } = \mathop{\max }\limits_{{i = 1,2,\ldots ,d}}\left| {{x}_{i} - {y}_{i}}\right| .$

最常用的${\ell }_{k}$范数是${\ell }_{1}$ - 范数、${\ell }_{2}$ - 范数和${\ell }_{\infty }$范数。注意，对于$k = \infty$，$\parallel x - y{\parallel }_{\infty } = \mathop{\max }\limits_{{i = 1,2,\ldots ,d}}\left| {{x}_{i} - {y}_{i}}\right| .$

2.1.1 Similarity Join Under ${\ell }_{1}/{\ell }_{\infty }$ Distance. It is well known that the ${\ell }_{1}$ metric in $d$ dimensions can be embedded into the ${\ell }_{\infty }$ metric in ${2}^{d - 1}$ dimensions via the following transformation. Note that for any point $\left( {{x}_{1},{x}_{2},\ldots ,{x}_{d}}\right)  \in  {\mathbb{R}}^{d}$ ,we have

2.1.1 ${\ell }_{1}/{\ell }_{\infty }$距离下的相似性连接。众所周知，$d$维的${\ell }_{1}$度量可以通过以下变换嵌入到${2}^{d - 1}$维的${\ell }_{\infty }$度量中。注意，对于任何点$\left( {{x}_{1},{x}_{2},\ldots ,{x}_{d}}\right)  \in  {\mathbb{R}}^{d}$，我们有

$$
\mathop{\sum }\limits_{{i = 1}}^{d}\left| {x}_{i}\right|  = \mathop{\max }\limits_{{\left( {{z}_{2},\ldots ,{z}_{d}}\right)  \in  \{  - 1,1{\} }^{d - 1}}}\left| {{x}_{1} + {z}_{2}{x}_{2} + \cdots  + {z}_{d}{x}_{d}}\right| .
$$

Thus two points $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {R}_{1}$ and $\left( {{y}_{1},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ join under ${\ell }_{1}$ distance if and only if

因此，两个点$\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {R}_{1}$和$\left( {{y}_{1},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$在${\ell }_{1}$距离下进行连接的充要条件是

$$
\mathop{\max }\limits_{{\left( {{z}_{2},\ldots ,{z}_{d}}\right)  \in  \{  - 1,1{\} }^{d - 1}}}\left| {\left( {{x}_{1} + {z}_{2}{x}_{2} + \cdots  + {z}_{d}{x}_{d}}\right)  - \left( {{y}_{1} + {z}_{2}{y}_{2} + \cdots  + {z}_{d}{y}_{d}}\right) }\right|  \leq  r.
$$

We map the point $\left( {{x}_{1},{x}_{2},\ldots ,{x}_{d}}\right)$ to a point in ${2}^{d - 1}$ dimensions,where each dimension has coordinate ${x}_{1} + {z}_{2}{x}_{2}\cdots  + {z}_{d}{x}_{d}$ corresponding to each $\left( {{z}_{2},\ldots ,{z}_{d}}\right)  \in  \{  - 1,1{\} }^{d - 1}$ . The similar transformation applies for point $\left( {{y}_{1},{y}_{2},\ldots ,{y}_{d}}\right)$ . These two $d$ -dimensional points join under ${\ell }_{1}$ distance if and only if the corresponding two ${2}^{d - 1}$ -dimensional points join under ${\ell }_{\infty }$ distance.

我们将点 $\left( {{x}_{1},{x}_{2},\ldots ,{x}_{d}}\right)$ 映射到 ${2}^{d - 1}$ 维空间中的一个点，其中每个维度的坐标 ${x}_{1} + {z}_{2}{x}_{2}\cdots  + {z}_{d}{x}_{d}$ 对应于每个 $\left( {{z}_{2},\ldots ,{z}_{d}}\right)  \in  \{  - 1,1{\} }^{d - 1}$。类似的变换适用于点 $\left( {{y}_{1},{y}_{2},\ldots ,{y}_{d}}\right)$。当且仅当对应的两个 ${2}^{d - 1}$ 维点在 ${\ell }_{\infty }$ 距离下相连时，这两个 $d$ 维点在 ${\ell }_{1}$ 距离下相连。

Next,we reduce the similarity join under ${\ell }_{\infty }$ distance to the rectangles-containing-points problem: given a set ${R}_{1}$ of ${N}_{1}$ points and a set ${R}_{2}$ of ${N}_{2}$ orthogonal rectangles,the goal is to return all pairs $\left( {x,y}\right)  \in  {R}_{1} \times  {R}_{2}$ such that $x \in  y$ . The reduction is quite straightforward. We map the point $\left( {{y}_{1},{y}_{2},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ to a $d$ -dimensional rectangle defined by the Cartesian product of intervals $\left\lbrack  {{y}_{i} - r,{y}_{i} + r}\right\rbrack$ for $i = 1,2,\ldots ,d$ . The points in ${R}_{1}$ remain unchanged. It should be obvious that two points $x \in  {R}_{1}$ and $y \in  {R}_{2}$ join under ${\ell }_{\infty }$ distance if and only if $x$ falls inside the rectangle corresponding to $y$ . Note that the reduction will only produce squares,but our solution to the rectangle-containing-points can handle general orthogonal rectangles.

接下来，我们将 ${\ell }_{\infty }$ 距离下的相似连接问题转化为矩形包含点问题：给定一个包含 ${N}_{1}$ 个点的集合 ${R}_{1}$ 和一个包含 ${N}_{2}$ 个正交矩形的集合 ${R}_{2}$，目标是返回所有满足 $x \in  y$ 的点对 $\left( {x,y}\right)  \in  {R}_{1} \times  {R}_{2}$。这种转化非常直接。我们将点 $\left( {{y}_{1},{y}_{2},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ 映射到一个由区间 $\left\lbrack  {{y}_{i} - r,{y}_{i} + r}\right\rbrack$（其中 $i = 1,2,\ldots ,d$）的笛卡尔积定义的 $d$ 维矩形。集合 ${R}_{1}$ 中的点保持不变。显然，当且仅当 $x$ 落在对应于 $y$ 的矩形内部时，两个点 $x \in  {R}_{1}$ 和 $y \in  {R}_{2}$ 在 ${\ell }_{\infty }$ 距离下相连。请注意，这种转化只会产生正方形，但我们对矩形包含点问题的解决方案可以处理一般的正交矩形。

2.1.2 Similarity Join Under ${\ell }_{2}$ Distance. We use the lifting transformation [13] to reduce the similarity join under ${\ell }_{2}$ distance to the halfspaces-containing-points problem in $d + 1$ dimensions. More precisely,we map each point $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {R}_{1}$ to a point in $d + 1$ dimensions as

2.1.2 ${\ell }_{2}$ 距离下的相似连接。我们使用提升变换 [13] 将 ${\ell }_{2}$ 距离下的相似连接问题转化为 $d + 1$ 维空间中的半空间包含点问题。更准确地说，我们将每个点 $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {R}_{1}$ 映射到 $d + 1$ 维空间中的一个点，如下所示

$$
\left( {{x}_{1},\ldots ,{x}_{d},{x}_{1}^{2} + \cdots  + {x}_{d}^{2}}\right) 
$$

and map a point $\left( {{y}_{1},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ to a halfspace in $d + 1$ dimensions $\left( {y}_{i}\right.$ ’s are the coefficients and ${z}_{i}$ ’s are the variables) as

并将点 $\left( {{y}_{1},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ 映射到 $d + 1$ 维空间中的一个半空间（$\left( {y}_{i}\right.$ 是系数，${z}_{i}$ 是变量），如下所示

$$
 - 2{y}_{1}{z}_{1} - \cdots  - 2{y}_{d}{z}_{d} + {z}_{d + 1} + {y}_{1}^{2} + \cdots  + {y}_{d}^{2} - {r}^{2} \geq  0.
$$

Observe that ${\left( {x}_{1} - {y}_{2}\right) }^{2} + \cdots  + {\left( {x}_{d} - {y}_{d}\right) }^{2} \leq  {r}^{2}$ can be rewritten as

观察可知，${\left( {x}_{1} - {y}_{2}\right) }^{2} + \cdots  + {\left( {x}_{d} - {y}_{d}\right) }^{2} \leq  {r}^{2}$ 可以重写为

$$
{x}_{1}^{2} + {y}_{1}^{2} + \cdots  + {x}_{d}^{2} + {y}_{d}^{2} - 2{x}_{1}{y}_{1} - \cdots  - 2{x}_{d}{y}_{d} - {r}^{2} \geq  0.
$$

Thus,two points $x \in  {R}_{1}$ and $y \in  {R}_{2}$ join in the original $d$ -dimensional space under ${\ell }_{2}$ distance if and only if the lifted $x$ falls inside the halfspace corresponding to $y$ in the $\left( {d + 1}\right)$ -dimensional space. Thus,it is sufficient to solve the halfspaces-containing-points problem: given a set of ${N}_{1}$ points and a set of ${N}_{2}$ halfspaces,report all the (point,halfspace) pairs such that the point is inside the halfspace.

因此，当且仅当提升后的 $x$ 落在 $\left( {d + 1}\right)$ 维空间中对应于 $y$ 的半空间内部时，原始 $d$ 维空间中的两个点 $x \in  {R}_{1}$ 和 $y \in  {R}_{2}$ 在 ${\ell }_{2}$ 距离下相连。因此，解决半空间包含点问题就足够了：给定一个包含 ${N}_{1}$ 个点的集合和一个包含 ${N}_{2}$ 个半空间的集合，报告所有点在半空间内部的（点，半空间）对。

### 2.2 MPC Primitives

### 2.2 多方计算原语

Assume that $\mathrm{{IN}} > {p}^{1 + \epsilon }$ ,where $\epsilon  > 0$ is any small constant. We introduce the following primitives in the MPC model,all of which can be computed with load $O\left( {\mathrm{{IN}}/p}\right)$ in $O\left( 1\right)$ rounds.

假设$\mathrm{{IN}} > {p}^{1 + \epsilon }$，其中$\epsilon  > 0$是任意小的常数。我们在多方计算（MPC）模型中引入以下原语，所有这些原语都可以在$O\left( 1\right)$轮内以负载$O\left( {\mathrm{{IN}}/p}\right)$进行计算。

2.2.1 Sorting. The sorting problem in the MPC model is defined as follows. Initially, IN elements are distributed arbitrarily on $p$ servers,which are labeled $1,2,\ldots ,p$ . The goal is to redistribute the elements so that each server has $\mathrm{{IN}}/p$ elements in the end,whereas any element at server $i$ is smaller than or equal to any element at server $j$ ,for any $i < j$ . By realizing that the MPC model is the same as the BSP model, we can directly invoke Goodrich's optimal BSP sorting algorithm [16]. His algorithm has load $L = \Theta \left( {\mathrm{{IN}}/p}\right)$ and runs in $O\left( {{\log }_{L}\mathrm{{IN}}}\right)  = O\left( {{\log }_{L}\left( {pL}\right) }\right)  = O\left( {{\log }_{L}p}\right)$ rounds. When IN $> {p}^{1 + \epsilon }$ ,this is $O\left( 1\right)$ rounds.

2.2.1 排序。多方计算（MPC）模型中的排序问题定义如下。最初，IN 个元素任意分布在$p$台服务器上，这些服务器标记为$1,2,\ldots ,p$。目标是重新分配这些元素，使得最终每台服务器有$\mathrm{{IN}}/p$个元素，并且对于任意$i < j$，服务器$i$上的任何元素都小于或等于服务器$j$上的任何元素。通过认识到多方计算（MPC）模型与批量同步并行（BSP）模型相同，我们可以直接调用古德里奇（Goodrich）的最优批量同步并行（BSP）排序算法[16]。他的算法负载为$L = \Theta \left( {\mathrm{{IN}}/p}\right)$，并在$O\left( {{\log }_{L}\mathrm{{IN}}}\right)  = O\left( {{\log }_{L}\left( {pL}\right) }\right)  = O\left( {{\log }_{L}p}\right)$轮内运行。当 IN $> {p}^{1 + \epsilon }$时，这是$O\left( 1\right)$轮。

2.2.2 Multi-Numbering. Suppose that each tuple carries a key,and that there are ${n}_{k}$ tuples for key $k$ . The goal of the multi-numbering problem is to,for each key $k$ ,assign consecutive numbers $1,2,\ldots ,{n}_{k}$ to the ${n}_{k}$ tuples with key $k$ ,respectively.

2.2.2 多重编号。假设每个元组都带有一个键，并且键$k$有${n}_{k}$个元组。多重编号问题的目标是，对于每个键$k$，分别为带有键$k$的${n}_{k}$个元组分配连续的编号$1,2,\ldots ,{n}_{k}$。

We solve this problem by reducing it to the all prefix-sums problem: given an array of elements $A\left\lbrack  1\right\rbrack  ,\ldots ,A\left\lbrack  \mathrm{{IN}}\right\rbrack$ ,compute $S\left\lbrack  i\right\rbrack   = A\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  i\right\rbrack$ for all $i = 1,\ldots ,\mathrm{{IN}}$ ,where $\oplus$ is any associative operator. Goodrich et al. [17] gave an algorithm in the BSP model for this problem that uses $O\left( {\mathrm{{IN}}/p}\right)$ load and $O\left( 1\right)$ rounds.

我们通过将这个问题归约为全前缀和问题来解决它：给定一个元素数组$A\left\lbrack  1\right\rbrack  ,\ldots ,A\left\lbrack  \mathrm{{IN}}\right\rbrack$，对所有$i = 1,\ldots ,\mathrm{{IN}}$计算$S\left\lbrack  i\right\rbrack   = A\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  i\right\rbrack$，其中$\oplus$是任意结合运算符。古德里奇（Goodrich）等人[17]在批量同步并行（BSP）模型中为这个问题给出了一个算法，该算法使用$O\left( {\mathrm{{IN}}/p}\right)$的负载和$O\left( 1\right)$轮。

To see how the multi-numbering problem reduces to the all prefix-sums problem, we first sort all tuples by their keys; ties are broken arbitrarily. The $i$ -th tuple in the sorted order will produce a pair (x,y),which will act as $A\left\lbrack  i\right\rbrack$ . For each tuple that is the first of its key in the sorted order,we produce the pair(0,1); otherwise,we produce(1,1). Note that we need another round of communication to determine whether each tuple is the first of its key, in case that its predecessor resides on another server. Then we define the operator $\oplus$ as $\left( {{x}_{1},{y}_{1}}\right)  \oplus  \left( {{x}_{2},{y}_{2}}\right)  = \left( {{x}_{1}{x}_{2},y}\right)$ ,where $y = {y}_{1} + {y}_{2}$ if ${x}_{2} = 1$ and otherwise $y = {y}_{2}$ .

为了了解多重编号问题如何归约为全前缀和问题，我们首先按元组的键对所有元组进行排序；若键相同则任意打破平局。排序顺序中的第$i$个元组将产生一个对 (x, y)，它将作为$A\left\lbrack  i\right\rbrack$。对于排序顺序中每个键的第一个元组，我们产生对 (0, 1)；否则，我们产生 (1, 1)。请注意，如果某个元组的前一个元组位于另一台服务器上，我们需要另一轮通信来确定每个元组是否是其键的第一个元组。然后我们将运算符$\oplus$定义为$\left( {{x}_{1},{y}_{1}}\right)  \oplus  \left( {{x}_{2},{y}_{2}}\right)  = \left( {{x}_{1}{x}_{2},y}\right)$，其中如果${x}_{2} = 1$则$y = {y}_{1} + {y}_{2}$，否则$y = {y}_{2}$。

Consider any $\left( {x,y}\right)  = A\left\lbrack  i\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  j\right\rbrack$ . Intuitively, $x = 0$ indicates that $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ contain at least one tuple that is the first of its key,whereas $y$ counts the number of tuples in $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ whose key is the same as that of $A\left\lbrack  j\right\rbrack$ . It is an easy exercise to check that $\oplus$ is associative,and after solving the all prefix-sums problem, $S\left\lbrack  i\right\rbrack$ is exactly the number of tuples in front of the $i$ -th tuple that has the same key (including the $i$ -th tuple itself),which solves the multi-numbering problem as desired.

考虑任意 $\left( {x,y}\right)  = A\left\lbrack  i\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  j\right\rbrack$ 。直观地说，$x = 0$ 表示 $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ 至少包含一个其键（key）为首个键的元组（tuple），而 $y$ 计算 $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ 中键与 $A\left\lbrack  j\right\rbrack$ 的键相同的元组数量。很容易验证 $\oplus$ 是可结合的，并且在解决所有前缀和问题后，$S\left\lbrack  i\right\rbrack$ 恰好是第 $i$ 个元组之前具有相同键的元组数量（包括第 $i$ 个元组本身），这就如愿解决了多编号问题。

2.2.3 Sum-by-Key. Suppose that each tuple is associated with a key and a weight. The goal of the sum-by-key problem is to compute, for each key, the total weight of all the tuples with the same key.

2.2.3 按键求和。假设每个元组都与一个键和一个权重相关联。按键求和问题的目标是为每个键计算所有具有相同键的元组的总权重。

This problem can be solved using essentially the same approach as for the multi-numbering problem. First,sort all the $N$ tuples by their keys. As earlier,each tuple will produce a pair(x,y). Now, $x$ still indicates whether this tuple is the first of its key,but we just set $y$ to be the weight associated with the tuple. After we have solved the all prefix-sums problem on these pairs, the last tuple of each key has the total weight for this key. Again, we need another round to identify the last tuple of each key by checking each tuple's successor.

这个问题本质上可以使用与多编号问题相同的方法来解决。首先，按键对所有 $N$ 元组进行排序。和之前一样，每个元组将产生一个对 (x, y)。现在，$x$ 仍然表示该元组是否是其键的第一个元组，但我们只需将 $y$ 设置为与该元组相关联的权重。在我们对这些对解决了所有前缀和问题之后，每个键的最后一个元组就拥有了该键的总权重。同样，我们需要再进行一轮，通过检查每个元组的后继元组来确定每个键的最后一个元组。

After the preceding algorithm finishes, for each key, exactly one tuple knows the total weight for the key (i.e., the last one in the sorted order). In some cases, we also need every tuple to know the total weight for the tuple's own key. To do so, we invoke the multi-numbering algorithm so that the last tuple of each key also knows the number of tuples with that key. From this number, we can compute exactly the range of servers that hold all the tuples with this key. Then we broadcast the total weight to these servers.

在上述算法完成后，对于每个键，恰好有一个元组知道该键的总权重（即按排序顺序的最后一个元组）。在某些情况下，我们还需要每个元组都知道其自身键的总权重。为此，我们调用多编号算法，使得每个键的最后一个元组也知道具有该键的元组数量。根据这个数量，我们可以准确计算出持有所有具有该键的元组的服务器范围。然后我们将总权重广播到这些服务器。

2.2.4 Multi-Search. The multi-search problem is defined as follows. Let $U$ be a universe with a total order,and let $S,Q \subseteq  U$ be two subsets of elements from $U$ . Let $\left| S\right|  = {N}_{1},\left| Q\right|  = {N}_{2}$ ,and $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ . The elements in $S$ are called keys,and the elements in $Q$ are called queries. The problem asks us to,for each query $q \in  Q$ ,find its predecessor in $S$ (i.e.,the largest key that is no larger than $q$ ).

2.2.4 多重搜索。多重搜索问题定义如下。设 $U$ 是一个具有全序关系的集合，设 $S,Q \subseteq  U$ 是来自 $U$ 的两个元素子集。设 $\left| S\right|  = {N}_{1},\left| Q\right|  = {N}_{2}$ ，且 $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ 。$S$ 中的元素称为键，$Q$ 中的元素称为查询。该问题要求我们为每个查询 $q \in  Q$ 找到其在 $S$ 中的前驱（即不大于 $q$ 的最大键）。

The multi-search algorithm given by Goodrich et al. [17] is randomized, with a small probability exceeding $O\left( {\mathrm{{IN}}/p}\right)$ load. In fact,this problem can also be solved using all prefix-sums,which results in a deterministic algorithm with load $O\left( {\mathrm{{IN}}/p}\right)$ . We first sort all the keys and queries together; in case of ties,we put the keys before the queries. Then for each key $k$ ,define its corresponding $A\left\lbrack  i\right\rbrack$ as itself; for each query,define its $A\left\lbrack  i\right\rbrack   =  - \infty$ ; define $\oplus   = \max$ . Then each query has its $S\left\lbrack  i\right\rbrack   =$ $\mathop{\max }\limits_{{j \leq  i}}A\left\lbrack  j\right\rbrack$ ,which is the largest key among those smaller than the query itself (i.e.,its predecessor in the keys).

古德里奇（Goodrich）等人 [17] 给出的多重搜索算法是随机化的，有很小的概率会超过 $O\left( {\mathrm{{IN}}/p}\right)$ 的负载。实际上，这个问题也可以使用所有前缀和来解决，从而得到一个负载为 $O\left( {\mathrm{{IN}}/p}\right)$ 的确定性算法。我们首先将所有的键和查询一起排序；如果出现平局，我们将键放在查询之前。然后对于每个键 $k$ ，将其对应的 $A\left\lbrack  i\right\rbrack$ 定义为其自身；对于每个查询，定义其 $A\left\lbrack  i\right\rbrack   =  - \infty$ ；定义 $\oplus   = \max$ 。然后每个查询都有其 $S\left\lbrack  i\right\rbrack   =$ $\mathop{\max }\limits_{{j \leq  i}}A\left\lbrack  j\right\rbrack$ ，它是那些小于查询本身的键中的最大键（即其在键中的前驱）。

2.2.5 Cartesian Product. The Cartesian product of two sets of size ${N}_{1}$ and ${N}_{2}$ ,respectively,can be reported using a degenerated version of the hypercube algorithm [2, 8], incurring a load of $O\left( {\left( {\sqrt{{N}_{1}{N}_{2}/p} + \operatorname{IN}/p}\right) \log p}\right)$ with probability $1 - 1/{p}^{\Omega \left( 1\right) }$ . The extra $\log$ factors are due to the use of hashing. We observe that if the elements in each set are numbered as $1,2,3,\ldots$ ,then we can achieve deterministic and perfect load balancing.

2.2.5 笛卡尔积。大小分别为 ${N}_{1}$ 和 ${N}_{2}$ 的两个集合的笛卡尔积，可以使用超立方体算法的简化版本 [2, 8] 来计算，产生负载为 $O\left( {\left( {\sqrt{{N}_{1}{N}_{2}/p} + \operatorname{IN}/p}\right) \log p}\right)$ 的概率为 $1 - 1/{p}^{\Omega \left( 1\right) }$。额外的 $\log$ 因子是由于使用了哈希。我们观察到，如果每个集合中的元素编号为 $1,2,3,\ldots$，那么我们可以实现确定性的完美负载均衡。

Without loss of generality,assume that ${N}_{1} \leq  {N}_{2}$ . As in the standard hypercube algorithm,we arrange the $p$ servers into a ${d}_{1} \times  {d}_{2}$ grid such that ${d}_{1}{d}_{2} = p$ . We first use multi-numbering to assign consecutive numbers to all tuples in ${R}_{1}$ and ${R}_{2}$ ,respectively. If an element in ${R}_{1}$ gets assigned a number $x$ ,then we send it to all servers in the $\left( {x{\;\operatorname{mod}\;{d}_{1}}}\right)$ -th row of the grid; for an element in ${R}_{2}$ ,we send it to all servers in the $\left( {x{\;\operatorname{mod}\;{d}_{2}}}\right)$ -th column of the grid. Each server then produces all pairs of elements received. By setting ${d}_{1} = \sqrt{\frac{p{N}_{1}}{{N}_{2}}},{d}_{2} = \sqrt{\frac{p{N}_{2}}{{N}_{1}}}$ ,the load is $O\left( {\sqrt{{N}_{1}{N}_{2}/p} + \mathrm{{IN}}/p}\right)$ .

不失一般性，假设 ${N}_{1} \leq  {N}_{2}$。与标准的超立方体算法一样，我们将 $p$ 个服务器排列成一个 ${d}_{1} \times  {d}_{2}$ 网格，使得 ${d}_{1}{d}_{2} = p$。我们首先使用多重编号分别为 ${R}_{1}$ 和 ${R}_{2}$ 中的所有元组分配连续的编号。如果 ${R}_{1}$ 中的一个元素被分配了编号 $x$，那么我们将其发送到网格的第 $\left( {x{\;\operatorname{mod}\;{d}_{1}}}\right)$ 行的所有服务器；对于 ${R}_{2}$ 中的一个元素，我们将其发送到网格的第 $\left( {x{\;\operatorname{mod}\;{d}_{2}}}\right)$ 列的所有服务器。然后每个服务器生成接收到的元素的所有对。通过设置 ${d}_{1} = \sqrt{\frac{p{N}_{1}}{{N}_{2}}},{d}_{2} = \sqrt{\frac{p{N}_{2}}{{N}_{1}}}$，负载为 $O\left( {\sqrt{{N}_{1}{N}_{2}/p} + \mathrm{{IN}}/p}\right)$。

2.2.6 Server Allocation. In many of our algorithms,we decompose the problem into up to $p$ subproblems and allocate the $p$ servers appropriately,with subproblem $j$ having $p\left( j\right)$ servers,where $\mathop{\sum }\limits_{j}p\left( j\right)  \leq  p$ . Thus,each subproblem needs to know which servers have been allocated to it. This is trivial if $\mathrm{{IN}} \geq  {p}^{2}$ ,as we can collect all the $p\left( j\right)$ ’s to one server,do a central allocation,and broadcast the allocation results to all servers,as is done in Beame et al. [8]. When we only have $\mathrm{{IN}} \geq  {p}^{1 + \epsilon }$ , some more work is needed to ensure $O\left( {\mathrm{{IN}}/p}\right)$ load.

2.2.6 服务器分配。在我们的许多算法中，我们将问题分解为最多 $p$ 个子问题，并适当地分配 $p$ 个服务器，子问题 $j$ 有 $p\left( j\right)$ 个服务器，其中 $\mathop{\sum }\limits_{j}p\left( j\right)  \leq  p$。因此，每个子问题需要知道哪些服务器已分配给它。如果 $\mathrm{{IN}} \geq  {p}^{2}$，这很简单，因为我们可以将所有的 $p\left( j\right)$ 收集到一个服务器，进行集中分配，并将分配结果广播到所有服务器，就像 Beame 等人 [8] 所做的那样。当我们只有 $\mathrm{{IN}} \geq  {p}^{1 + \epsilon }$ 时，需要做更多的工作来确保 $O\left( {\mathrm{{IN}}/p}\right)$ 负载。

More formally,in the server allocation problem,each tuple has a subproblem id $j$ ,which identifies the subproblem it belongs to (the $j$ ’s do not have to be consecutive),and $p\left( j\right)$ ,which is the number of servers allocated to subproblem $j$ . The goal is to attach to each tuple a range $\left\lbrack  {{p}_{1}\left( j\right) ,{p}_{2}\left( j\right) }\right\rbrack$ such that the ranges of different subproblems are disjoint and $\mathop{\max }\limits_{j}{p}_{2}\left( j\right)  \leq  p$ .

更正式地说，在服务器分配问题中，每个元组都有一个子问题 ID $j$，它标识该元组所属的子问题（$j$ 不必是连续的），以及 $p\left( j\right)$，它是分配给子问题 $j$ 的服务器数量。目标是为每个元组附加一个范围 $\left\lbrack  {{p}_{1}\left( j\right) ,{p}_{2}\left( j\right) }\right\rbrack$，使得不同子问题的范围不相交且 $\mathop{\max }\limits_{j}{p}_{2}\left( j\right)  \leq  p$。

We again resort to all prefix-sums. First sort all tuples by their subproblem id. For each tuple, define its corresponding $A\left\lbrack  i\right\rbrack   = p\left( j\right)$ if it is the first tuple of subproblem $j$ and 0 otherwise. After running all prefix-sums,for each tuple,we set its ${p}_{2}\left( j\right)  = S\left\lbrack  i\right\rbrack$ ,and ${p}_{1}\left( j\right)  = S\left\lbrack  i\right\rbrack   - p\left( j\right)  + 1$ .

我们再次使用所有前缀和。首先按子问题编号对所有元组进行排序。对于每个元组，如果它是子问题 $j$ 的第一个元组，则定义其对应的 $A\left\lbrack  i\right\rbrack   = p\left( j\right)$，否则定义为 0。在计算完所有前缀和后，对于每个元组，我们设置其 ${p}_{2}\left( j\right)  = S\left\lbrack  i\right\rbrack$ 和 ${p}_{1}\left( j\right)  = S\left\lbrack  i\right\rbrack   - p\left( j\right)  + 1$。

## 3 EQUI-JOIN

## 3 等值连接

We start by revisiting the equi-join (natural join) problem between two relations, ${R}_{1}\left( {A,B}\right)  \bowtie$ ${R}_{2}\left( {B,C}\right)$ . Let ${N}_{1}$ and ${N}_{2}$ be the sizes of ${R}_{1}$ and ${R}_{2}$ ,respectively; set $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ .

我们首先回顾两个关系 ${R}_{1}\left( {A,B}\right)  \bowtie$ ${R}_{2}\left( {B,C}\right)$ 之间的等值连接（自然连接）问题。设 ${N}_{1}$ 和 ${N}_{2}$ 分别为 ${R}_{1}$ 和 ${R}_{2}$ 的大小；令 $\mathrm{{IN}} = {N}_{1} + {N}_{2}$。

Beame et al. [8] classified the join values into being heavy and light. For a join value $v$ ,let ${R}_{i}\left( v\right)$ be the set of tuples in ${R}_{i}$ with join value $v$ . Then a join value $v$ is heavy if $\left| {{R}_{1}\left( v\right) }\right|  \geq  {N}_{1}/p$ or $\left| {{R}_{2}\left( v\right) }\right|  \geq  {N}_{2}/p$ ,and light otherwise. Then they gave an algorithm with load

比姆（Beame）等人 [8] 将连接值分为重值和轻值。对于一个连接值 $v$，设 ${R}_{i}\left( v\right)$ 为 ${R}_{i}$ 中连接值为 $v$ 的元组集合。若 $\left| {{R}_{1}\left( v\right) }\right|  \geq  {N}_{1}/p$ 或 $\left| {{R}_{2}\left( v\right) }\right|  \geq  {N}_{2}/p$，则连接值 $v$ 为重值，否则为轻值。然后他们给出了一个具有负载的算法

$$
\widetilde{\Theta }\left( {\sqrt{\frac{\mathop{\sum }\limits_{{\text{heavy }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right| }{p}} + \frac{\mathrm{{IN}}}{p}}\right) . \tag{1}
$$

We observe that this bound is asymptotically the same as $\widetilde{\Theta }\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ ,because

我们观察到这个边界在渐近意义上与 $\widetilde{\Theta }\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ 相同，因为

$$
\text{ OUT } = \mathop{\sum }\limits_{v}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right|  = \mathop{\sum }\limits_{{\text{heavy }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right|  + \mathop{\sum }\limits_{{\text{light }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right| ,
$$

so (1) is upper bounded by $\widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ . Meanwhile,it is also lower bounded by $\widetilde{\Omega }\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ . First,it is clearly in $\widetilde{\Omega }\left( {\mathrm{{IN}}/p}\right)$ . Second,it is also in $\widetilde{\Omega }\left( {\mathrm{{IN}}/p - {\mathrm{{IN}}}^{2}/{p}^{2}}\right)$ since

所以 (1) 的上界为 $\widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$。同时，它的下界为 $\widetilde{\Omega }\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$。首先，显然它属于 $\widetilde{\Omega }\left( {\mathrm{{IN}}/p}\right)$。其次，它也属于 $\widetilde{\Omega }\left( {\mathrm{{IN}}/p - {\mathrm{{IN}}}^{2}/{p}^{2}}\right)$，因为

$$
\mathop{\sum }\limits_{{\text{light }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right|  \leq  \frac{{N}_{1}{N}_{2}}{p} \leq  \frac{{\mathrm{{IN}}}^{2}}{p}.
$$

Thus, we have

因此，我们有

$$
\left( 1\right)  = \widetilde{\Omega }\left( {\sqrt{\frac{\mathrm{{OUT}} - {\mathrm{{IN}}}^{2}/p}{p}} + \frac{\mathrm{{IN}}}{p}}\right)  = \widetilde{\Omega }\left( \sqrt{\frac{\mathrm{{OUT}} - {\mathrm{{IN}}}^{2}/p}{p} + \frac{{\mathrm{{IN}}}^{2}}{{p}^{2}}}\right)  = \widetilde{\Omega }\left( \sqrt{\mathrm{{OUT}}/p}\right) .
$$

Therefore, their algorithm is output-optimal, but up to a logarithmic factor. Furthermore, their analysis relies on the uniform hashing assumption (i.e., the hash function distributes each distinct key to the servers uniformly and independently). It is not clear whether more realistic hash functions, such as universal hashing, could still work. They also assume that each server knows the entire set of heavy join values and their frequencies,namely all the $\left| {{R}_{i}\left( v\right) }\right|$ ’s that are larger than ${N}_{i}/p$ ,for $i = 1,2$ . In the following,we describe a deterministic algorithm that achieves the result in Theorem 3.1.

因此，他们的算法在输出上是最优的，但存在一个对数因子。此外，他们的分析依赖于均匀哈希假设（即哈希函数将每个不同的键均匀且独立地分配到各个服务器）。尚不清楚更现实的哈希函数，如通用哈希，是否仍然有效。他们还假设每个服务器都知道所有重连接值的集合及其频率，即对于 $i = 1,2$，所有大于 ${N}_{i}/p$ 的 $\left| {{R}_{i}\left( v\right) }\right|$。接下来，我们描述一个确定性算法，该算法能达到定理 3.1 中的结果。

### 3.1 The Algorithm

### 3.1 算法

Our algorithm can be seen as an MPC version of the classical sort-merge-join algorithm. The high-level procedure is given in Algorithm 1. It decomposes the equi-join into a set of Cartesian products,one for each distinct join value in the domain of $B$ ,and runs the hypercube algorithm to compute each Cartesian product in parallel. To achieve output-optimality, it needs to compute OUT first and allocate the appropriate number of servers to each subproblem. The details in each step are described in the following.

我们的算法可以看作是经典排序 - 合并 - 连接算法的 MPC（大规模并行计算）版本。高层过程如算法 1 所示。它将等值连接分解为一组笛卡尔积，每个笛卡尔积对应 $B$ 定义域中的一个不同连接值，并运行超立方体算法并行计算每个笛卡尔积。为了实现输出最优，它需要先计算 OUT，并为每个子问题分配适当数量的服务器。以下将详细描述每个步骤。

<!-- Media -->

ALGORITHM 1: EQUI-JOIN ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)$

算法 1：等值连接 ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)$

---

1 Collect data statistics and compute OUT;

1 收集数据统计信息并计算 OUT；

2 ${p}_{v} \leftarrow  \max \left\{  {\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right)  + {N}_{2}\left( v\right) }{\mathrm{{IN}}}}\right\rbrack  ,\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{\mathrm{{OUT}}}}\right\rbrack  }\right\}$ for each $v \in  \operatorname{dom}\left( B\right)$ that has tuples on at least 2

对于每个在至少2台服务器上有元组的$v \in  \operatorname{dom}\left( B\right)$，计算2 ${p}_{v} \leftarrow  \max \left\{  {\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right)  + {N}_{2}\left( v\right) }{\mathrm{{IN}}}}\right\rbrack  ,\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{\mathrm{{OUT}}}}\right\rbrack  }\right\}$

		servers;

		台服务器；

	for each such $v$ do in parallel

	对于每个这样的$v$并行执行

			Compute ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$ with ${p}_{v}$ servers;

					使用${p}_{v}$台服务器计算${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$；

---

<!-- Media -->

Step (1): Computing OUT. Consider each distinct value $v$ of the join attribute $B$ . Let ${R}_{i}\left( v\right)  =$ ${\sigma }_{B = v}{R}_{i}$ ,and let ${N}_{i}\left( v\right)  = \left| {{R}_{i}\left( v\right) }\right|$ . Note that $\mathrm{{OUT}} = \mathop{\sum }\limits_{v}{N}_{1}\left( v\right) {N}_{2}\left( v\right)$ . We first use the sum-by-key algorithm to compute all the ${N}_{i}\left( v\right)$ ’s (i.e.,each tuple in ${R}_{i}\left( v\right)$ is considered to have key $v$ and weight 1). Recall that after the sum-by-key algorithm,for each $v$ ,exactly one tuple in ${R}_{i}\left( v\right)$ knows ${N}_{i}\left( v\right)$ . We sort all such tuples by the key $v$ . Then we add up all the ${N}_{1}\left( v\right) {N}_{2}\left( v\right)$ ’s,which can also be done by sum-by-key (just that the key is the same for all tuples).

步骤(1)：计算OUT。考虑连接属性$B$的每个不同值$v$。设${R}_{i}\left( v\right)  =$ ${\sigma }_{B = v}{R}_{i}$，并设${N}_{i}\left( v\right)  = \left| {{R}_{i}\left( v\right) }\right|$。注意$\mathrm{{OUT}} = \mathop{\sum }\limits_{v}{N}_{1}\left( v\right) {N}_{2}\left( v\right)$。我们首先使用按键求和算法来计算所有的${N}_{i}\left( v\right)$（即，${R}_{i}\left( v\right)$中的每个元组被视为具有键$v$和权重1）。回想一下，在按键求和算法之后，对于每个$v$，${R}_{i}\left( v\right)$中恰好有一个元组知道${N}_{i}\left( v\right)$。我们按键$v$对所有这样的元组进行排序。然后我们将所有的${N}_{1}\left( v\right) {N}_{2}\left( v\right)$相加，这也可以通过按键求和来完成（只是所有元组的键相同）。

Step (2): Computing ${R}_{1} \bowtie  {R}_{2}$ . Next,we compute the join (i.e.,the Cartesian products ${R}_{1}\left( v\right)  \times$ ${R}_{2}\left( v\right)$ for all $v$ ). Sort all tuples in both ${R}_{1}$ and ${R}_{2}$ by the join attribute $B$ . Consider each distinct value $v$ in $B$ . If all tuples in ${R}_{1}\left( v\right)  \cup  {R}_{2}\left( v\right)$ land on the same server,their join results can be emitted directly, so we only need to deal with the case when they land on two or more servers. There are at most $p$ such $v$ ’s. For each such $v$ ,we allocate

步骤(2)：计算${R}_{1} \bowtie  {R}_{2}$。接下来，我们计算连接（即，对于所有的$v$，计算笛卡尔积${R}_{1}\left( v\right)  \times$ ${R}_{2}\left( v\right)$）。按连接属性$B$对${R}_{1}$和${R}_{2}$中的所有元组进行排序。考虑$B$中的每个不同值$v$。如果${R}_{1}\left( v\right)  \cup  {R}_{2}\left( v\right)$中的所有元组都落在同一台服务器上，它们的连接结果可以直接输出，因此我们只需要处理它们落在两台或更多台服务器上的情况。最多有$p$个这样的$v$。对于每个这样的$v$，我们分配

$$
{p}_{v} = \max \left\{  {\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right)  + {N}_{2}\left( v\right) }{\mathrm{{IN}}}}\right\rbrack  ,\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{\mathrm{{OUT}}}}\right\rbrack  }\right\}   \tag{2}
$$

servers and compute the Cartesian product ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$ . Note that we need a total of $O\left( p\right)$ servers; scaling down the initial $p$ can ensure that at most $p$ servers are needed. Here,we also need the server allocation primitive to allocate servers to these subproblems accordingly. Finally, to be able to use the deterministic version of the hypercube algorithm,the tuples in each ${R}_{i}\left( v\right)$ need to be assigned consecutive numbers, which can be achieved by running the multi-numbering algorithm,treating each distinct join value $v$ as a key. It can be easily verified that the load is $O\left( {\mathop{\max }\limits_{v}\left\{  {\sqrt{\frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{{p}_{v}}} + \frac{{N}_{1}\left( v\right) }{{p}_{v}} + \frac{{N}_{2}\left( v\right) }{{p}_{v}}}\right\}  }\right)  = O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right) .$

服务器并计算笛卡尔积 ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$ 。请注意，我们总共需要 $O\left( p\right)$ 台服务器；缩小初始的 $p$ 可以确保最多需要 $p$ 台服务器。在这里，我们还需要服务器分配原语来相应地为这些子问题分配服务器。最后，为了能够使用超立方体算法的确定性版本，每个 ${R}_{i}\left( v\right)$ 中的元组需要被分配连续的编号，这可以通过运行多编号算法来实现，将每个不同的连接值 $v$ 视为一个键。可以很容易地验证负载为 $O\left( {\mathop{\max }\limits_{v}\left\{  {\sqrt{\frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{{p}_{v}}} + \frac{{N}_{1}\left( v\right) }{{p}_{v}} + \frac{{N}_{2}\left( v\right) }{{p}_{v}}}\right\}  }\right)  = O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right) .$

THEOREM 3.1. There is a deterministic algorithm that computes the equi-join between two relations in $O\left( 1\right)$ rounds with load $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ . It does not assume any prior statistical information about the data.

定理3.1：存在一种确定性算法，它可以在 $O\left( 1\right)$ 轮内以负载 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ 计算两个关系之间的等值连接。它不假设关于数据的任何先验统计信息。

### 3.2 A Matching Lower Bound

### 3.2 匹配的下界

As argued in Section 1.2,the term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ is optimal for any tuple-based algorithm. In the following,we show an input-dependent lower bound of $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2},\mathrm{{IN}}/p}\right\}  }\right)$ in terms of the number of bits even when $\mathrm{{OUT}} = O\left( 1\right)$ . Note that when $1/p \leq  {N}_{1}/{N}_{2} \leq  p$ ,this lower bound becomes $\Omega \left( {\mathrm{{IN}}/p}\right)$ ,matching the upper bound in Theorem 3.1 up to a logarithmic factor. When ${N}_{1}/{N}_{2} < 1/p$ or ${N}_{1}/{N}_{2} > p$ ,the lower bound becomes $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2}}\right\}  }\right)$ ,which can be matched up to a logarithmic factor by the trivial algorithm that simply broadcasts the smaller relation to all servers.

正如在1.2节中所论证的，对于任何基于元组的算法，项 $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ 是最优的。在下面，我们展示了即使在 $\mathrm{{OUT}} = O\left( 1\right)$ 的情况下，关于比特数的一个依赖于输入的下界 $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2},\mathrm{{IN}}/p}\right\}  }\right)$ 。请注意，当 $1/p \leq  {N}_{1}/{N}_{2} \leq  p$ 时，这个下界变为 $\Omega \left( {\mathrm{{IN}}/p}\right)$ ，在对数因子范围内与定理3.1中的上界相匹配。当 ${N}_{1}/{N}_{2} < 1/p$ 或 ${N}_{1}/{N}_{2} > p$ 时，下界变为 $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2}}\right\}  }\right)$ ，通过简单地将较小的关系广播到所有服务器的平凡算法可以在对数因子范围内与之匹配。

THEOREM 3.2. Any randomized algorithm that computes the equi-join between two relations in $O\left( 1\right)$ rounds with a success probability more than $3/4$ must incur a load of at least $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2},\frac{\mathrm{{IN}}}{p}}\right\}  }\right)$ bits.

定理3.2：任何在 $O\left( 1\right)$ 轮内以超过 $3/4$ 的成功概率计算两个关系之间等值连接的随机算法，其负载必须至少为 $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2},\frac{\mathrm{{IN}}}{p}}\right\}  }\right)$ 比特。

Proof. We use a reduction from the lopsided set disjointness problem studied in communication complexity. Alice has $\leq  n$ elements and Bob has $\leq  m$ elements with $m > n$ ,both from a universe of size $m$ ,and the goal is to decide whether they have an element in common. It has been proved that in any multi-round communication protocol,either Alice has to send $\Omega \left( n\right)$ bits to Bob or Bob has to send $\Omega \left( m\right)$ bits to Alice [30]. This holds even for randomized algorithms with a success probability larger than $3/4$ . We also note that in the hard instances used in Pătraşcu [30],the intersection size of Alice's and Bob's sets is either 0 or 1 .

证明：我们采用通信复杂度中研究的不平衡集合不相交问题进行归约。爱丽丝有 $\leq  n$ 个元素，鲍勃有 $\leq  m$ 个元素，且 $m > n$ ，这些元素都来自一个大小为 $m$ 的全集，目标是确定他们是否有共同的元素。已经证明，在任何多轮通信协议中，要么爱丽丝必须向鲍勃发送 $\Omega \left( n\right)$ 比特，要么鲍勃必须向爱丽丝发送 $\Omega \left( m\right)$ 比特 [30] 。即使对于成功概率大于 $3/4$ 的随机算法，这也成立。我们还注意到，在Pătraşcu [30] 中使用的困难实例中，爱丽丝和鲍勃的集合的交集大小要么为0，要么为1。

The reduction works as follows. Assuming that ${N}_{1} \leq  {N}_{2}$ ,we will show in the following an $\Omega \left( {\min \left( {{N}_{1},\mathrm{{IN}}/p}\right) }\right)$ lower bound; symmetrically,if ${N}_{2} \leq  {N}_{1}$ ,we can show an $\Omega \left( {\min \left( {{N}_{2},\mathrm{{IN}}/p}\right) }\right)$ lower bound. Combining the two cases proves the theorem. Given a hard instance of lopsided set disjointness,we create ${R}_{1}$ with ${N}_{1} = n$ tuples,whose join values are the elements of Alice’s set; create ${R}_{2}$ with ${N}_{2} = m$ tuples,whose join values are the elements of Bob’s set. Then solving the join problem also determines whether the two sets intersect or not, whereas OUT can only be 1 or 0 .

归约过程如下。假设${N}_{1} \leq  {N}_{2}$ ，我们将在下面给出一个$\Omega \left( {\min \left( {{N}_{1},\mathrm{{IN}}/p}\right) }\right)$ 下界；对称地，如果${N}_{2} \leq  {N}_{1}$ ，我们可以给出一个$\Omega \left( {\min \left( {{N}_{2},\mathrm{{IN}}/p}\right) }\right)$ 下界。结合这两种情况即可证明该定理。给定一个不平衡集合不相交问题的困难实例，我们创建一个包含${N}_{1} = n$ 个元组的${R}_{1}$ ，其连接值是爱丽丝集合中的元素；创建一个包含${N}_{2} = m$ 个元组的${R}_{2}$ ，其连接值是鲍勃集合中的元素。那么，解决连接问题也能确定这两个集合是否相交，而输出（OUT）只能是 1 或 0。

Recall that in the MPC model,the adversary can allocate the input arbitrarily. We allocate ${R}_{1}$ and ${R}_{2}$ to the $p$ servers as follows.

回顾一下，在多方计算（MPC）模型中，对手可以任意分配输入。我们按如下方式将${R}_{1}$ 和${R}_{2}$ 分配给$p$ 个服务器。

If ${N}_{2} \leq  p \cdot  {N}_{1}$ ,we allocate Alice’s set to $\frac{p{N}_{2}}{\mathrm{{IN}}}$ servers and Bob’s set to $\frac{p{N}_{1}}{\mathrm{{IN}}}$ servers. Then Alice’s servers must send $\Omega \left( {N}_{1}\right)$ bits to Bob’s servers,which incurs a total load (across all rounds) of $\Omega \left( {\mathrm{{IN}}/p}\right)$ bits per server,or Bob’s servers must send $\Omega \left( {N}_{2}\right)$ bits to Alice’s servers,also incurring a total load of $\Omega \left( {\mathrm{{IN}}/p}\right)$ bits per server.

如果${N}_{2} \leq  p \cdot  {N}_{1}$ ，我们将爱丽丝的集合分配给$\frac{p{N}_{2}}{\mathrm{{IN}}}$ 个服务器，将鲍勃的集合分配给$\frac{p{N}_{1}}{\mathrm{{IN}}}$ 个服务器。那么，爱丽丝的服务器必须向鲍勃的服务器发送$\Omega \left( {N}_{1}\right)$ 位信息，这会导致每个服务器（在所有轮次中）的总负载为$\Omega \left( {\mathrm{{IN}}/p}\right)$ 位；或者，鲍勃的服务器必须向爱丽丝的服务器发送$\Omega \left( {N}_{2}\right)$ 位信息，同样会导致每个服务器的总负载为$\Omega \left( {\mathrm{{IN}}/p}\right)$ 位。

If ${N}_{2} > p \cdot  {N}_{1}$ ,then we allocate Bob’s set to one server and Alice’s set to the other $p - 1$ servers. Then Alice’s servers will send $\Omega \left( {N}_{1}\right)$ bits to Bob’s server or receive $\Omega \left( {N}_{2}\right)$ bits,so the load is $\Omega \left( {\min \left( {{N}_{1},{N}_{2}/p}\right) }\right)  = \Omega \left( {N}_{1}\right) .$

如果${N}_{2} > p \cdot  {N}_{1}$ ，那么我们将鲍勃的集合分配给一个服务器，将爱丽丝的集合分配给其他$p - 1$ 个服务器。那么，爱丽丝的服务器将向鲍勃的服务器发送$\Omega \left( {N}_{1}\right)$ 位信息或接收$\Omega \left( {N}_{2}\right)$ 位信息，因此负载为$\Omega \left( {\min \left( {{N}_{1},{N}_{2}/p}\right) }\right)  = \Omega \left( {N}_{1}\right) .$

## 4 SIMILARITY JOIN UNDER ${\ell }_{1}/{\ell }_{\infty }$

## 4 在${\ell }_{1}/{\ell }_{\infty }$下的相似连接

Recall that in Section 2.1, we reduced similarity joins to various geometric containment problems. In each geometric containment problem,we are given a set of points ${R}_{1}$ and a set of ranges ${R}_{2}$ ,and the goal is to find all (point, range) pairs such that the point is contained in the range.

回顾一下，在2.1节中，我们将相似连接简化为各种几何包含问题。在每个几何包含问题中，我们给定一组点${R}_{1}$和一组范围${R}_{2}$，目标是找出所有满足点包含在范围内的（点，范围）对。

All our algorithms for solving different containment problems are based on the following basic ideas. Each range is first decomposed into non-overlapping cells. We assign a join key for each cell, as well as for each point, and then invoke the equi-join algorithm to find all the (point, cell) pairs that share the same join key. We can easily assign join keys to ensure completeness (i.e., all (point, cell) pairs where the point is contained in the cell are assigned the same join key), and hence must be captured by the equi-join algorithm. In fact, a trivial method for completeness is just to assign the same join key to all points and all cells. However, this is essentially computing the full Cartesian product. To achieve output-optimality, we need to more carefully assign the join keys so that the equi-join algorithm does not report too many potential join results. For asymptotic optimality, the equi-join algorithm should return at most a constant times more than the actual join results (i.e., $O\left( \mathrm{{OUT}}\right)$ ) while guaranteeing completeness. The exact way in which the join keys are assigned are different depending on the particular distance metric under consideration. The ${\ell }_{1}/{\ell }_{\infty }$ metric (rectangles-containing-points) will be discussed in Section 4 and ${\ell }_{2}$ metric (halfspaces-containing-points) in Section 5.

我们解决不同包含问题的所有算法都基于以下基本思想。首先将每个范围分解为不重叠的单元格。我们为每个单元格以及每个点分配一个连接键，然后调用等值连接算法来找出所有共享相同连接键的（点，单元格）对。我们可以轻松地分配连接键以确保完整性（即，所有点包含在单元格中的（点，单元格）对都被分配相同的连接键），因此等值连接算法必定能捕获到这些对。实际上，一种确保完整性的简单方法是为所有点和所有单元格分配相同的连接键。然而，这本质上是在计算全笛卡尔积。为了实现输出最优性，我们需要更谨慎地分配连接键，以使等值连接算法不会报告过多的潜在连接结果。为了实现渐近最优性，等值连接算法在保证完整性的同时，返回的结果最多比实际连接结果多一个常数倍（即$O\left( \mathrm{{OUT}}\right)$）。连接键的确切分配方式取决于所考虑的特定距离度量。${\ell }_{1}/{\ell }_{\infty }$度量（矩形包含点）将在第4节讨论，${\ell }_{2}$度量（半空间包含点）将在第5节讨论。

<!-- Media -->

<!-- figureText: $b$ $b$ ${I}_{1}$ $b$ -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_10.jpg?x=506&y=258&w=552&h=310&r=0"/>

Fig. 1. Partially covered and fully covered slabs.

图1. 部分覆盖和完全覆盖的板片。

<!-- Media -->

In addition, knowing the value of OUT is crucial, which will be used to allocate servers for distinct join keys. Although for some problems OUT can be computed easily, for other problems computing OUT exactly would be difficult. In these cases, we compute an estimate of OUT that is accurate enough for asymptotic optimality.

此外，知道OUT的值至关重要，它将用于为不同的连接键分配服务器。虽然对于某些问题可以轻松计算出OUT，但对于其他问题，精确计算OUT会很困难。在这些情况下，我们计算一个对渐近最优性来说足够准确的OUT估计值。

In this section and the next, we will assume constant dimensions, which means that it is sufficient to solve the various geometric containment problems in constant dimensions by the reductions in Section 2.1. We deal with the high-dimensional case in Section 6.

在本节和下一节中，我们将假设维度为常数，这意味着通过2.1节中的简化方法，足以在常数维度下解决各种几何包含问题。我们将在第6节处理高维情况。

### 4.1 One Dimension

### 4.1 一维情况

We start by considering the 1D case-that is, the intervals-containing-points problem. We are given a set of ${N}_{1}$ points and a set of ${N}_{2}$ intervals. Set $\operatorname{IN} = {N}_{1} + {N}_{2}$ . The goal is to report all (point, interval) pairs such that the point is inside the interval. In the following, we describe how to solve this problem in $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ load.

我们首先考虑一维情况，即区间包含点的问题。我们给定一组${N}_{1}$个点和一组${N}_{2}$个区间。设$\operatorname{IN} = {N}_{1} + {N}_{2}$。目标是报告所有满足点在区间内的（点，区间）对。下面，我们描述如何在$O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$负载下解决这个问题。

Step (1): Computing OUT. As with the equi-join algorithm, we start by computing the value of OUT. First, we sort all the points and number them consecutively in the sorted order. Then, for each interval $I = \left\lbrack  {x,y}\right\rbrack$ ,we find the predecessor points of $x$ and $y$ (multi-search). Taking the difference of the numbers assigned to the two predecessors will give us the number of points inside I. Finally, we add up all of these counts to get OUT (special case of sum-by-key).

步骤（1）：计算OUT。与等值连接算法一样，我们首先计算OUT的值。首先，我们对所有点进行排序，并按排序顺序连续编号。然后，对于每个区间$I = \left\lbrack  {x,y}\right\rbrack$，我们找到$x$和$y$的前驱点（多重搜索）。取分配给这两个前驱点的编号之差，就可以得到区间I内的点数。最后，我们将所有这些计数相加得到OUT（按键求和的特殊情况）。

Step (2): Sorting into slabs. Setting $b = \sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p$ ,we will ensure that the load of the remaining steps is $O\left( b\right)$ . We sort all the points and divide them into slabs of size $b$ . There are at most $p$ slabs,which are labeled as $1,2,3,\ldots$ ,in sorted order. Consider each interval $I$ in ${R}_{2}$ . All the points inside $I$ can be classified into two cases: (1) points that fall in a slab partially covered by $I$ and (2) points that fall in a slab fully covered by $I$ . For example,in Figure 1,the join between ${I}_{1}$ and the points in the leftmost and the rightmost slab is considered under case (1),whereas the join between ${I}_{1}$ and the points in the two middle slabs is considered under case (2). Note that if an interval falls inside a slab completely, its join with the points in that slab is also considered under case (1),such as ${I}_{2}$ in Figure 1.

步骤 (2)：划分为条带。设 $b = \sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p$ ，我们将确保剩余步骤的负载为 $O\left( b\right)$ 。我们对所有点进行排序，并将它们划分为大小为 $b$ 的条带。最多有 $p$ 个条带，按排序顺序标记为 $1,2,3,\ldots$ 。考虑 ${R}_{2}$ 中的每个区间 $I$ 。$I$ 内的所有点可以分为两种情况：(1) 落在被 $I$ 部分覆盖的条带中的点；(2) 落在被 $I$ 完全覆盖的条带中的点。例如，在图 1 中，${I}_{1}$ 与最左侧和最右侧条带中的点的连接属于情况 (1)，而 ${I}_{1}$ 与中间两个条带中的点的连接属于情况 (2)。请注意，如果一个区间完全落在一个条带内，它与该条带中的点的连接也属于情况 (1)，如图 1 中的 ${I}_{2}$ 。

Step (3): Partially covered slabs. In this step, we deal with the partially covered slabs. For each interval endpoint, we find which slab it falls into (multi-search). Then, for each slab, we compute the number of endpoints falling inside (sum-by-key). Consider each slab $i$ . Suppose that it contains $P\left( i\right)$ endpoints. We allocate $\left\lceil  {p \cdot  \frac{P\left( i\right) }{{N}_{2}}}\right\rceil$ servers to compute the join between the $b$ points in the slab and the intervals with these $P\left( i\right)$ endpoints (note that we need $O\left( p\right)$ servers). We simply evenly allocate the $P\left( i\right)$ intervals to these servers (use multi-numbering to ensure balance) and broadcast all the $b$ points to them. The load is thus

步骤 (3)：部分覆盖的条带。在这一步中，我们处理部分覆盖的条带。对于每个区间端点，我们找出它落在哪个条带中（多重搜索）。然后，对于每个条带，我们计算落在其中的端点数量（按键求和）。考虑每个条带 $i$ 。假设它包含 $P\left( i\right)$ 个端点。我们分配 $\left\lceil  {p \cdot  \frac{P\left( i\right) }{{N}_{2}}}\right\rceil$ 台服务器来计算条带中的 $b$ 个点与具有这 $P\left( i\right)$ 个端点的区间的连接（注意我们需要 $O\left( p\right)$ 台服务器）。我们简单地将 $P\left( i\right)$ 个区间均匀分配给这些服务器（使用多重编号以确保平衡），并将所有 $b$ 个点广播给它们。因此负载为

$$
O\left( {b + \frac{P\left( i\right) }{{pP}\left( i\right) /{N}_{2}}}\right)  = O\left( b\right) .
$$

Step (4): Fully covered slabs. Let $F\left( i\right)$ be the number of intervals fully covering a slab $i$ . We can compute all the $F\left( i\right)$ ’s using the prefix-sums algorithm as follows. If an interval fully covers slabs $i,\ldots ,j$ ,we generate two (key,value) pairs(i,1)and $\left( {j + 1, - 1}\right)$ . For each slab $k$ ,we generate a (key, value) pair $\left( {k + {0.5},0}\right)$ . Then we sort all these pairs by key and compute the prefix-sums on the value. Consider the prefix-sum at key $k + {0.5}$ . Observe that an interval fully covering slabs $i,\ldots ,j$ contributes 1 to this prefix-sum if $i \leq  k \leq  j$ and 0 otherwise: if $i > k$ ,the keys of the two pairs generated by the interval are both after $k + {0.5}$ ; if $j < k$ ,the values of the two pairs cancel out. Therefore,the prefix-sum computed at key $k + {0.5}$ is exactly $F\left( i\right)$ .

步骤 (4)：完全覆盖的条带。设 $F\left( i\right)$ 为完全覆盖条带 $i$ 的区间数量。我们可以使用前缀和算法按如下方式计算所有的 $F\left( i\right)$ 。如果一个区间完全覆盖条带 $i,\ldots ,j$ ，我们生成两个 (键, 值) 对 (i, 1) 和 $\left( {j + 1, - 1}\right)$ 。对于每个条带 $k$ ，我们生成一个 (键, 值) 对 $\left( {k + {0.5},0}\right)$ 。然后我们按键对所有这些对进行排序，并对值计算前缀和。考虑键 $k + {0.5}$ 处的前缀和。观察到，如果 $i \leq  k \leq  j$ ，则完全覆盖条带 $i,\ldots ,j$ 的区间对这个前缀和贡献 1，否则贡献 0：如果 $i > k$ ，该区间生成的两个对的键都在 $k + {0.5}$ 之后；如果 $j < k$ ，这两个对的值相互抵消。因此，在键 $k + {0.5}$ 处计算的前缀和恰好是 $F\left( i\right)$ 。

Now,the full slabs can be dealt with using essentially the same algorithm. We allocate ${p}_{i} =$ $\left\lceil  {p \cdot  \frac{{bF}\left( i\right) }{\text{ OUT }}}\right\rceil$ servers to compute the join (full Cartesian product) of the $b$ points in slab $i$ and the $F\left( i\right)$ intervals fully covering the slab. Since $\mathop{\sum }\limits_{i}{bF}\left( i\right)  \leq  \mathrm{{OUT}}$ ,this requires at most $O\left( p\right)$ servers. We simply evenly allocate the $F\left( i\right)$ intervals to these servers and broadcast all the $b$ points to them.

现在，基本上可以使用相同的算法处理完全覆盖的条带。我们分配 ${p}_{i} =$ $\left\lceil  {p \cdot  \frac{{bF}\left( i\right) }{\text{ OUT }}}\right\rceil$ 台服务器来计算条带 $i$ 中的 $b$ 个点与完全覆盖该条带的 $F\left( i\right)$ 个区间的连接（完全笛卡尔积）。由于 $\mathop{\sum }\limits_{i}{bF}\left( i\right)  \leq  \mathrm{{OUT}}$ ，这最多需要 $O\left( p\right)$ 台服务器。我们简单地将 $F\left( i\right)$ 个区间均匀分配给这些服务器，并将所有 $b$ 个点广播给它们。

The load is thus

因此负载为

$$
O\left( {b + \frac{F\left( i\right) }{{pbF}\left( i\right) /\mathrm{{OUT}}}}\right)  = O\left( {b + \frac{\mathrm{{OUT}}}{pb}}\right)  = O\left( b\right) .
$$

THEOREM 4.1. There is a deterministic algorithm for the intervals-containing-points problem that runs in $O\left( 1\right)$ rounds with $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ load.

定理4.1. 对于区间包含点问题，存在一种确定性算法，该算法在$O\left( 1\right)$轮内运行，负载为$O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$。

### 4.2 Two and Higher Dimensions

### 4.2 二维及更高维度

Next, we consider the rectangle-containing-points problem in two dimensions. Here we are given a set of ${N}_{1}$ points in $2\mathrm{D}$ and a set of ${N}_{2}$ rectangles. Set $\operatorname{IN} = {N}_{1} + {N}_{2}$ . The goal is to report all (point, rectangle) pairs such that the point is inside the rectangle. The basic idea is to impose a canonical decomposition on the $x$ -axis to break down the problem into many instances of the 1D problem. Although canonical decomposition is a well-known technique, applying it in the MPC model has some technical challenges. For simplicity,we will assume that $p$ is a power of 2,which does not affect the asymptotic results.

接下来，我们考虑二维中的矩形包含点问题。这里给定了一个在$2\mathrm{D}$中的${N}_{1}$个点的集合和一个${N}_{2}$个矩形的集合。设$\operatorname{IN} = {N}_{1} + {N}_{2}$。目标是报告所有（点，矩形）对，使得点位于矩形内部。基本思路是对$x$轴进行规范分解，将问题分解为多个一维问题的实例。尽管规范分解是一种众所周知的技术，但在MPC（大规模并行计算，Massively Parallel Computation）模型中应用它存在一些技术挑战。为简单起见，我们假设$p$是2的幂，这不会影响渐近结果。

Step (1): Sorting into atomic slabs and canonical slabs. We sort all the $x$ -coordinates,including those of the points,as well as the left and right $x$ -coordinates of the rectangles. After the sorting, each server has $\operatorname{IN}/px$ -coordinates,which divides the whole $x$ -axis into $p$ slabs,which are labeled as $1,2,\ldots ,p$ in increasing order along the $x$ -axis. We call these slabs atomic slabs. We impose a binary tree over these $p$ atomic slabs in the standard fashion,where each node in the binary tree corresponds to a canonical slab (thus, an atomic slab is also a canonical slab), as shown in Figure 2. We decompose each rectangle into $O\left( {\log p}\right)$ disjoint pieces: a left piece and a right piece that partially cover an atomic slab,and at most $O\left( {\log p}\right)$ middle pieces,each spanning one canonical slab as large as possible. For example, ${\sigma }_{1}$ in Figure 2 is decomposed into four pieces: a left piece that falls inside slab 1, a right piece that falls inside slab 7, and 3 middle pieces that span canonical slabs $2,3 - 4$ ,and 5-6,respectively.

步骤(1)：排序为原子 slab（平板）和规范 slab。我们对所有的$x$坐标进行排序，包括点的坐标以及矩形的左右$x$坐标。排序后，每个服务器拥有$\operatorname{IN}/px$坐标，这些坐标将整个$x$轴划分为$p$个 slab，沿着$x$轴按递增顺序标记为$1,2,\ldots ,p$。我们称这些 slab 为原子 slab。我们以标准方式在这$p$个原子 slab 上构建一棵二叉树，其中二叉树中的每个节点对应一个规范 slab（因此，原子 slab 也是规范 slab），如图2所示。我们将每个矩形分解为$O\left( {\log p}\right)$个不相交的部分：一个左部分和一个右部分，它们部分覆盖一个原子 slab，以及最多$O\left( {\log p}\right)$个中间部分，每个中间部分尽可能大地跨越一个规范 slab。例如，图2中的${\sigma }_{1}$被分解为四个部分：一个落在 slab 1 内的左部分，一个落在 slab 7 内的右部分，以及分别跨越规范 slab $2,3 - 4$和 5 - 6 的3个中间部分。

All the join results between a point and a left/right piece can be found easily. Each server holds $O\left( {\mathrm{{IN}}/p}\right) x$ -coordinates,which correspond to at most $O\left( {\mathrm{{IN}}/p}\right)$ points and at most $O\left( {\mathrm{{IN}}/p}\right)$ left/right pieces whose left/right $x$ -coordinates are stored at this server. Thus,all of these join results can be found locally after the sorting.

点与左/右部分之间的所有连接结果都可以轻松找到。每个服务器持有$O\left( {\mathrm{{IN}}/p}\right) x$坐标，这些坐标对应最多$O\left( {\mathrm{{IN}}/p}\right)$个点和最多$O\left( {\mathrm{{IN}}/p}\right)$个左/右部分，其左右$x$坐标存储在该服务器上。因此，所有这些连接结果在排序后可以在本地找到。

<!-- Media -->

<!-- figureText: ${\sigma }_{1}$ 5 7 8 $5 - 6$ $7 - 8$ $5 - 8$ ${\sigma }_{2}$ 1 2 3 4 $1 - 2$ $3 - 4$ $1 - 4$ -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_12.jpg?x=424&y=256&w=711&h=406&r=0"/>

Fig. 2. Rectangles-joining-points. In this example,the atomic slabs are $\{ 1,2,3,4,5,6,7,8\}$ and the canonical slabs are $\{ 1,2,3,4,5,6,7,8,1 - 2,3 - 4,5 - 6,7 - 8,1 - 4,5 - 8,1 - 8\}$ .

图2. 矩形与点的连接。在这个例子中，原子 slab 是$\{ 1,2,3,4,5,6,7,8\}$，规范 slab 是$\{ 1,2,3,4,5,6,7,8,1 - 2,3 - 4,5 - 6,7 - 8,1 - 4,5 - 8,1 - 8\}$。

<!-- Media -->

Step (2): Middle pieces-Reduction to the 1D case. We are left with finding the join results between a point and a middle piece of a rectangle. Note that a middle piece of a rectangle is not just a canonical slab on the $x$ -axis,but an implicit rectangle induced by this canonical slab and the $y$ -projection of the original rectangle. The idea is to "collect," for each canonical slab $s$ ,all the middle pieces that span $s$ ,as well as all the points that fall inside $s$ . Because the $x$ -coordinates of all these points must fall in the $x$ -projection of these middle pieces (which is just the canonical slab), the problem will reduce to the 1D problem, where we need to find all the (point, middle piece) pairs where the $y$ -coordinate of the point falls inside the $y$ -projection of the middle piece. Then we can invoke the 1D algorithm to solve all of these 1D instances in parallel. This is exactly the classical technique of using a canonical decomposition to reduce a 2D problem to many 1D instances. However, in the MPC model, we face the following difficulties (which are trivial in a centralized model): (1) how to collect the inputs of each 1D problem? and (2) how to allocate the $p$ servers to these 1D instances so as to ensure a balanced load? We address each in turn.

步骤 (2)：中间部分——简化为一维情况。我们接下来要找出一个点和一个矩形的中间部分之间的连接结果。请注意，矩形的中间部分不仅仅是 $x$ 轴上的一个规范条带，而是由这个规范条带和原始矩形在 $y$ 上的投影所诱导出的一个隐式矩形。思路是，对于每个规范条带 $s$，“收集”所有跨越 $s$ 的中间部分，以及所有落在 $s$ 内的点。因为所有这些点的 $x$ 坐标必须落在这些中间部分的 $x$ 投影（即规范条带）内，所以问题将简化为一维问题，在这个问题中，我们需要找出所有（点，中间部分）对，其中点的 $y$ 坐标落在中间部分的 $y$ 投影内。然后我们可以调用一维算法并行解决所有这些一维实例。这正是使用规范分解将二维问题简化为多个一维实例的经典技术。然而，在大规模并行计算（MPC）模型中，我们面临以下困难（在集中式模型中这些困难微不足道）：(1) 如何收集每个一维问题的输入？(2) 如何将 $p$ 台服务器分配给这些一维实例，以确保负载均衡？我们将依次解决每个问题。

Step (2.1): Collecting inputs for each canonical slab. In the MPC model, it is not possible to actually collect all the inputs of a canonical slab to one server because there are too many (e.g., the largest canonical slab has all the points as its inputs). By "collect," we mean that for each canonical slab $s$ ,we will attach its id to all its inputs so that when a number of servers are allocated later to solve the 1D problem associated with $s$ ,its inputs can be sent to these servers,by just looking at the canonical slab id's attached to the points and middle pieces.

步骤 (2.1)：为每个规范条带收集输入。在大规模并行计算（MPC）模型中，实际上不可能将一个规范条带的所有输入收集到一台服务器上，因为输入太多了（例如，最大的规范条带将所有点作为其输入）。所谓“收集”，我们的意思是，对于每个规范条带 $s$，我们将其 ID 附加到其所有输入上，这样当稍后分配若干台服务器来解决与 $s$ 相关的一维问题时，只需查看附加在点和中间部分上的规范条带 ID，就可以将其输入发送到这些服务器。

For each point,it is easy to figure out the $O\left( {\log p}\right)$ canonical slabs it falls into,by simply looking at the atomic slab it belongs to. For each interval, we first find the two atomic slabs containing its two endpoints. Note that these two pieces of information are held by two different servers, so we need to bring them together by another sorting step (say, sorting by the id of the interval). Then from the two atomic slabs,we have enough information to break the interval into $O\left( {\log p}\right)$ middle pieces, each corresponding to a canonical slab.

对于每个点，通过简单查看它所属的原子条带，很容易找出它落入的 $O\left( {\log p}\right)$ 个规范条带。对于每个区间，我们首先找到包含其两个端点的两个原子条带。请注意，这两条信息由两台不同的服务器持有，因此我们需要通过另一个排序步骤（例如，按区间的 ID 排序）将它们汇总在一起。然后从这两个原子条带中，我们有足够的信息将该区间拆分为 $O\left( {\log p}\right)$ 个中间部分，每个中间部分对应一个规范条带。

Step (2.2): Computing IN(s) and OUT(s). Suppose that canonical slab $s$ has ${N}_{1}\left( s\right)$ points and ${N}_{2}\left( s\right)$ middle pieces. Its input size is thus $\operatorname{IN}\left( s\right)  = {N}_{1}\left( s\right)  + {N}_{2}\left( s\right)$ . Define $\operatorname{OUT}\left( s\right)$ as the output size of the 1D instance on $s$ . The allocation of servers will depend on $\operatorname{IN}\left( s\right)$ and $\operatorname{OUT}\left( s\right)$ . As each point belongs to $O\left( {\log p}\right)$ canonical slabs and each interval has $O\left( {\log p}\right)$ middle pieces,the total input size of all instances is $\mathop{\sum }\limits_{s}\mathrm{{IN}}\left( s\right)  = O\left( {\mathrm{{IN}} \cdot  \log p}\right)$ . Since each rectangle is broken up into disjoint middle pieces,we have $\mathop{\sum }\limits_{s}\operatorname{OUT}\left( s\right)  \leq  \operatorname{OUT}$ .

步骤 (2.2)：计算 IN(s) 和 OUT(s)。假设规范条带 $s$ 有 ${N}_{1}\left( s\right)$ 个点和 ${N}_{2}\left( s\right)$ 个中间部分。因此，其输入大小为 $\operatorname{IN}\left( s\right)  = {N}_{1}\left( s\right)  + {N}_{2}\left( s\right)$。将 $\operatorname{OUT}\left( s\right)$ 定义为 $s$ 上一维实例的输出大小。服务器的分配将取决于 $\operatorname{IN}\left( s\right)$ 和 $\operatorname{OUT}\left( s\right)$。由于每个点属于 $O\left( {\log p}\right)$ 个规范条带，每个区间有 $O\left( {\log p}\right)$ 个中间部分，所有实例的总输入大小为 $\mathop{\sum }\limits_{s}\mathrm{{IN}}\left( s\right)  = O\left( {\mathrm{{IN}} \cdot  \log p}\right)$。由于每个矩形被拆分为不相交的中间部分，我们有 $\mathop{\sum }\limits_{s}\operatorname{OUT}\left( s\right)  \leq  \operatorname{OUT}$。

All the IN(s)'s can be computed by sum-by-key, using the canonical slab as the key. To count the OUT(s)’s, we invoke an instance of step (1) of the 1D algorithm for each canonical slab s. However, the load of step (1) of the 1D algorithm depends on its input size. Thus, to ensure a uniform load, we allocate ${p}_{s} = \left\lceil  {p \cdot  \frac{\operatorname{IN}\left( s\right) }{\operatorname{IN}\log p}}\right\rceil$ servers for canonical slab $s$ (this uses $O\left( p\right)$ servers in total),with each server having load $O\left( {\mathrm{{IN}}\left( s\right) /{p}_{s}}\right)  = O\left( {\frac{\mathrm{{IN}}}{p}\log p}\right)$ .

所有的IN(s)都可以通过按键求和来计算，使用规范板（canonical slab）作为键。为了计算OUT(s)的数量，我们针对每个规范板s调用一维算法步骤（1）的一个实例。然而，一维算法步骤（1）的负载取决于其输入大小。因此，为了确保负载均匀，我们为规范板$s$分配${p}_{s} = \left\lceil  {p \cdot  \frac{\operatorname{IN}\left( s\right) }{\operatorname{IN}\log p}}\right\rceil$台服务器（总共使用$O\left( p\right)$台服务器），每台服务器的负载为$O\left( {\mathrm{{IN}}\left( s\right) /{p}_{s}}\right)  = O\left( {\frac{\mathrm{{IN}}}{p}\log p}\right)$。

Step (2.3): Solving 1D instances. Finally,we allocate servers to the $O\left( p\right)$ canonical slabs to solve the 1D problem associated to each. Since the load of the 1D algorithm depends on both the input and output size, we need to take both into account when allocating the servers. More precisely, we allocate

步骤（2.3）：求解一维实例。最后，我们为$O\left( p\right)$个规范板分配服务器，以求解与每个规范板相关的一维问题。由于一维算法的负载取决于输入和输出的大小，因此在分配服务器时，我们需要同时考虑这两个因素。更准确地说，我们分配

$$
{p}_{s} = \left\lbrack  {p \cdot  \frac{\operatorname{OUT}\left( s\right) }{\mathrm{{OUT}}} + p \cdot  \frac{\operatorname{IN}\left( s\right) }{\operatorname{IN}\log p}}\right\rbrack  
$$

servers for a canonical slab $s$ and invoke steps (2) and (3) of the 1D algorithm for all the canonical slabs in parallel. Plugging in the result of Theorem 4.1 yields the following result.

台服务器给规范板$s$，并并行地对所有规范板调用一维算法的步骤（2）和（3）。代入定理4.1的结果可得到以下结果。

THEOREM 4.2. There is a deterministic algorithm for the rectangles-containing-points problem in 2D that runs in $O\left( 1\right)$ rounds with $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}\log p}\right)$ load.

定理4.2：存在一种确定性算法，用于解决二维中的矩形包含点问题，该算法在$O\left( 1\right)$轮内运行，负载为$O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}\log p}\right)$。

The algorithm can be extended to higher dimensions using the same idea presented earlier. More precisely,we can reduce a $d$ -dimensional problem to many(d - 1)-dimensional instances. The algorithm is the same as the preceding 2D algorithm. We build a canonical decomposition on the first dimension,break up each rectangle into a left piece,a right piece,and $O\left( {\log p}\right)$ middle pieces. The join results on the left/right pieces can be found easily with $O\left( {\mathrm{{IN}}/p}\right)$ load,whereas we invoke $O\left( p\right)$ instances of the(d - 1)-dimensional algorithm to find the join results of the middle pieces. Since the total input size of the(d - 1)-dimensional instances is $O\left( {\mathrm{{IN}}\log p}\right)$ while the total output size remains the same,we incur an extra $O\left( {\log p}\right)$ factor in the input-dependent term every time the dimensionality is reduced by one. Therefore, we obtain the following.

可以使用前面介绍的相同思路将该算法扩展到更高维度。更准确地说，我们可以将一个$d$维问题简化为多个（d - 1）维实例。该算法与前面的二维算法相同。我们在第一个维度上进行规范分解，将每个矩形拆分为左部分、右部分和$O\left( {\log p}\right)$个中间部分。左/右部分的连接结果可以通过$O\left( {\mathrm{{IN}}/p}\right)$负载轻松找到，而我们调用（d - 1）维算法的$O\left( p\right)$个实例来找到中间部分的连接结果。由于（d - 1）维实例的总输入大小为$O\left( {\mathrm{{IN}}\log p}\right)$，而总输出大小保持不变，因此每次将维度降低一维时，我们在依赖输入的项中会产生一个额外的$O\left( {\log p}\right)$因子。因此，我们得到以下结果。

THEOREM 4.3. There is a deterministic algorithm for the rectangles-containing-points problem in $d$ dimensions that runs in $O\left( 1\right)$ rounds with $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}{\log }^{d - 1}p}\right)$ load.

定理4.3：存在一种确定性算法，用于解决$d$维中的矩形包含点问题，该算法在$O\left( 1\right)$轮内运行，负载为$O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}{\log }^{d - 1}p}\right)$。

## 5 SIMILARITY JOIN UNDER ${\ell }_{2}$

## 5 ${\ell }_{2}$下的相似性连接

As mentioned in Section 2.1, we will study the halfspaces-containing-points problem that generalizes similarity joins under ${\ell }_{2}$ distance. Compared to the ${\ell }_{1}/{\ell }_{\infty }$ case,a key challenge in this problem is that there is no easy way to compute OUT, due to the non-orthogonal nature of the problem. Knowing the value of OUT is crucial in the previous algorithms, which is used to determine the right slab size, which in turn decides the load.

如第2.1节所述，我们将研究半空间包含点问题，该问题推广了${\ell }_{2}$距离下的相似性连接。与${\ell }_{1}/{\ell }_{\infty }$的情况相比，这个问题的一个关键挑战在于，由于问题的非正交性质，没有简单的方法来计算OUT。在之前的算法中，知道OUT的值至关重要，它用于确定合适的板大小，而板大小又决定了负载。

Our way to get around this problem is based on the observation that the load is determined by the output-dependent term only when OUT is sufficiently large. But in this case, a constant-factor approximation of OUT suffices to guarantee the optimal load asymptotically, and random sampling can be used to estimate OUT. Random sampling will not be effective when OUT is small (it is known that to decide whether $\mathrm{{OUT}} = 1$ or 0 by sampling requires us to essentially sample the whole dataset), but in that case, the input-dependent term will dominate the load, and we do not need to know the value of OUT anyway.

我们解决这个问题的方法基于这样的观察：只有当OUT足够大时，负载才由依赖输出的项决定。但在这种情况下，OUT的常数因子近似值足以渐近地保证最优负载，并且可以使用随机抽样来估计OUT。当OUT较小时，随机抽样将无效（众所周知，通过抽样来确定是$\mathrm{{OUT}} = 1$还是0基本上需要对整个数据集进行抽样），但在这种情况下，依赖输入的项将主导负载，而且无论如何我们都不需要知道OUT的值。

### 5.1 Useful Tools From Computational Geometry

### 5.1 计算几何中的有用工具

5.1.1 Sampling With Threshold Approximation. We first mention the $\theta$ -thresholded approximation. A $\theta$ -thresholded approximation of $x$ is an estimate $\widehat{x}$ such that (1) if $x \geq  \theta$ ,then $\frac{x}{2} < \widehat{x} < {2x}$ ; (2) if $x < \theta$ ,then $\widehat{x} < {2\theta }$ . It captures our need of sampling when the output size is large enough. The following result first relates thresholded approximations with random sampling.

5.1.1 带阈值近似的采样。我们首先提及$\theta$ - 阈值近似。$x$的$\theta$ - 阈值近似是一个估计值$\widehat{x}$，使得（1）如果$x \geq  \theta$，那么$\frac{x}{2} < \widehat{x} < {2x}$；（2）如果$x < \theta$，那么$\widehat{x} < {2\theta }$。当输出规模足够大时，它满足了我们的采样需求。以下结果首先将阈值近似与随机采样联系起来。

THEOREM 5.1 ([18,25]). For any $q > 1$ ,let $S$ be a random sample with replacement from a set $P$ of $n$ points with $\left| S\right|  = O\left( {q\log \left( {q/\delta }\right) }\right)$ . Then with probability at least $1 - \delta ,n \cdot  \frac{\left| \Delta  \cap  S\right| }{\left| S\right| }$ is a $\frac{n}{q}$ -thresholded approximation of $\left| {\Delta  \cap  P}\right|$ for every simple ${x}^{3}\Delta$ .

定理 5.1（[18,25]）。对于任意的$q > 1$，设$S$是从包含$n$个点的集合$P$中有放回地抽取的一个随机样本，其中$\left| S\right|  = O\left( {q\log \left( {q/\delta }\right) }\right)$。那么，对于每个简单的${x}^{3}\Delta$，$1 - \delta ,n \cdot  \frac{\left| \Delta  \cap  S\right| }{\left| S\right| }$至少以$\frac{n}{q}$的概率是$\left| {\Delta  \cap  P}\right|$的$\frac{n}{q}$ - 阈值近似。

5.1.2 Partition Tree. We make use of the $b$ -partial partition tree of Chan [11]. A $b$ -partial partition tree on a set of points is a tree $T$ with constant fanout,where each leaf stores at most $b$ points, and each point is stored in exactly one leaf. Each node $v \in  T$ (both internal nodes and leaf nodes) stores a simplex $\Delta \left( v\right)$ ,which encloses all the points stored at the leaves below $v$ . For any $v$ ,the simplexes of its children do not overlap. In particular, this implies that all the leaf simplexes are disjoint. Chan [11] presented an algorithm to construct a $b$ -partial partition tree with the following properties in Theorem 5.2. Their construction of a partition tree is a recursive process of cutting the space into disjoint simplexes.

5.1.2 划分树。我们使用 Chan [11]提出的$b$ - 部分划分树。点集上的$b$ - 部分划分树是一棵扇出为常数的树$T$，其中每个叶子节点最多存储$b$个点，并且每个点恰好存储在一个叶子节点中。每个节点$v \in  T$（包括内部节点和叶子节点）都存储一个单纯形$\Delta \left( v\right)$，该单纯形包含存储在$v$下方叶子节点中的所有点。对于任意的$v$，其孩子节点的单纯形不重叠。特别地，这意味着所有叶子节点的单纯形是不相交的。Chan [11]在定理 5.2 中给出了一个构造具有以下性质的$b$ - 部分划分树的算法。他们构造划分树的过程是一个将空间递归切割成不相交单纯形的过程。

THEOREM 5.2 ([11]). Given $n$ points in ${\mathbb{R}}^{d}$ and a parameter $b < n/{\log }^{\omega \left( 1\right) }n$ ,we can build a $b$ -partial partition tree with $O\left( {n/b}\right)$ nodes each as a simplex such that any hyperplane intersects $O\left( {\left( n/b\right) }^{1 - 1/d}\right)$ simplexes of the tree.

定理 5.2（[11]）。给定${\mathbb{R}}^{d}$中的$n$个点和一个参数$b < n/{\log }^{\omega \left( 1\right) }n$，我们可以构建一个具有$O\left( {n/b}\right)$个节点（每个节点为一个单纯形）的$b$ - 部分划分树，使得任何超平面与树中的$O\left( {\left( n/b\right) }^{1 - 1/d}\right)$个单纯形相交。

In Chan's construction, all the leaf simplexes are disjoint and their union is the whole space. It only guarantees that each leaf simplex contains at most $b$ points but offers no lower bound, whereas we will need each leaf to have $\Theta \left( b\right)$ points. This can be easily achieved,however. Suppose that Chan’s $b$ -partial partition tree has at most $c \cdot  n/b$ leaves for some constant $c$ . A leaf simplex is big if it contains at least $b/{2c}$ points; otherwise,it is small. Observe that there must be at least $n/{2b}$ big leaf simplexes; otherwise,all the leaves together would contain less than $n/{2b} \cdot  b + {cn}/b \cdot  b/{2c} = n$ points. Then,we merge each big simplex with an equal number of small simplexes,and denote this union of simplexes as a cell so that each cell has $\Theta \left( b\right)$ points (Figure 3). Note that the small simplexes can be allocated to the big simplexes arbitrarily, and each cell is not necessarily connected. Since each cell consists of one big simplex and at most $\frac{{cn}/b}{n/{2b}} = {2c}$ small simplexes, which is a constant, we say that such a cell has constant description size. Note that on any cell with constant description size, any standard geometric operation (e.g., testing if a point falls inside the cell or if a hyperplane intersects the cell) takes $O\left( 1\right)$ time. The following corollary summarizes the properties we need from this modified $b$ -partial partition tree. In particular,we will not need the internal nodes of the tree.

在陈（Chan）的构造中，所有叶单纯形（leaf simplex）互不相交，且它们的并集是整个空间。它仅保证每个叶单纯形最多包含$b$个点，但没有给出下限，而我们需要每个叶包含$\Theta \left( b\right)$个点。不过，这很容易实现。假设陈的$b$ - 部分划分树（$b$ - partial partition tree）对于某个常数$c$最多有$c \cdot  n/b$个叶。如果一个叶单纯形至少包含$b/{2c}$个点，则称其为大单纯形；否则，称其为小单纯形。可以观察到，必定至少有$n/{2b}$个大的叶单纯形；否则，所有叶包含的点的总数将少于$n/{2b} \cdot  b + {cn}/b \cdot  b/{2c} = n$个。然后，我们将每个大单纯形与相同数量的小单纯形合并，并将这些单纯形的并集称为一个单元（cell），使得每个单元有$\Theta \left( b\right)$个点（图3）。注意，小单纯形可以任意分配给大单纯形，并且每个单元不一定是连通的。由于每个单元由一个大单纯形和最多$\frac{{cn}/b}{n/{2b}} = {2c}$个小单纯形组成，而$\frac{{cn}/b}{n/{2b}} = {2c}$是一个常数，我们称这样的单元具有常数描述大小（constant description size）。注意，对于任何具有常数描述大小的单元，任何标准的几何运算（例如，测试一个点是否落在单元内或一个超平面是否与单元相交）都需要$O\left( 1\right)$时间。以下推论总结了我们从这个修改后的$b$ - 部分划分树中所需的性质。特别地，我们不需要树的内部节点。

COROLLARY 5.3. Given $n$ points in ${\mathbb{R}}^{d}$ and a parameter $b < n/{\log }^{\omega \left( 1\right) }n$ ,we can find $O\left( {n/b}\right)$ disjoint cells such that (1) each cell has constant description size,(2) each cell contains $\Theta \left( b\right)$ points,and (3) any hyperplane intersects $O\left( {\left( n/b\right) }^{1 - 1/d}\right)$ cells.

推论5.3。给定$n$个位于${\mathbb{R}}^{d}$中的点和一个参数$b < n/{\log }^{\omega \left( 1\right) }n$，我们可以找到$O\left( {n/b}\right)$个不相交的单元，使得（1）每个单元具有常数描述大小，（2）每个单元包含$\Theta \left( b\right)$个点，并且（3）任何超平面与$O\left( {\left( n/b\right) }^{1 - 1/d}\right)$个单元相交。

### 5.2 The Algorithm

### 5.2 算法

Let $q$ be a parameter such that $1 < q < p$ ,where $p$ is the number of servers. The value of $q$ will be determined later.

设$q$为一个参数，使得$1 < q < p$，其中$p$是服务器的数量。$q$的值将在后面确定。

---

<!-- Footnote -->

${}^{3}$ Simplex generalizes the notion of triangle in 2D space (e.g.,in 3D space,a simplex is a tetrahedron). Although the precise definition of a $d$ -dimensional simplex is somewhat technical,for the following discussion,the reader can simply regard it as a polytope with a constant number of sides.

${}^{3}$单纯形（simplex）推广了二维空间中三角形的概念（例如，在三维空间中，单纯形是四面体）。尽管$d$维单纯形的精确定义有些技术性，但在下面的讨论中，读者可以简单地将其视为具有常数条边的多面体。

<!-- Footnote -->

---

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_15.jpg?x=507&y=262&w=547&h=413&r=0"/>

Fig. 3. The triangles are the leaf simplexes of the original partition tree, where each simplex contains at most four points but can contain as few as one point. Suppose thatsimplexes with at least three points are big simplexes and other simplexes are small simplexes. We merge each big simplex with an equal number of smaller simplexes, as shown by the dashed regions.

图3。三角形是原始划分树的叶单纯形，其中每个单纯形最多包含四个点，但也可以少至包含一个点。假设至少包含三个点的单纯形是大单纯形，其他单纯形是小单纯形。我们将每个大单纯形与相同数量的小单纯形合并，如虚线区域所示。

<!-- Media -->

Step (1): Constructing a partition tree. We ask all servers to randomly sample $\Theta \left( {q\log p}\right)$ points in total and send all the sampled points to one server. The server builds the modified $\Theta \left( {\log p}\right)$ - partial partition tree on the sampled points. By Theorem 5.1 and Corollary 5.3,we obtain $O\left( q\right)$ disjoint cells of constant description size. Note that the set of cells is a subdivision of the whole space,so it contains all points in ${R}_{1}$ . With probability $1 - 1/{p}^{\Omega \left( 1\right) }$ ,every cell contains $O\left( {{N}_{1}/q}\right)$ points of the original dataset. Any hyperplane always intersects $O\left( {q}^{1 - \frac{1}{d}}\right)$ cells due to the property of the partition tree. We broadcast these cells to all servers. This step incurs a load of $O\left( {q\log p}\right)$ .

步骤（1）：构建划分树。我们要求所有服务器总共随机采样$\Theta \left( {q\log p}\right)$个点，并将所有采样点发送到一台服务器。该服务器在采样点上构建修改后的$\Theta \left( {\log p}\right)$ - 部分划分树。根据定理5.1和推论5.3，我们得到$O\left( q\right)$个具有常数描述大小的不相交单元。注意，单元集合是整个空间的一个细分，因此它包含${R}_{1}$中的所有点。以概率$1 - 1/{p}^{\Omega \left( 1\right) }$，每个单元包含原始数据集中的$O\left( {{N}_{1}/q}\right)$个点。由于划分树的性质，任何超平面总是与$O\left( {q}^{1 - \frac{1}{d}}\right)$个单元相交。我们将这些单元广播给所有服务器。此步骤产生的负载为$O\left( {q\log p}\right)$。

Similar to the intervals-containing-points algorithm, we consider the following two cases for all points inside a halfspace: (1) those in cells partially covered by the halfspace and (2) those in cells fully covered by the halfspace.

与包含点的区间算法类似，我们针对半空间内的所有点考虑以下两种情况：(1) 位于被半空间部分覆盖的单元格中的点；(2) 位于被半空间完全覆盖的单元格中的点。

Step (2): Partially covered cells. For each halfspace,we find all the cells $\Delta$ intersected by its bounding halfplane. Note that there are $O\left( {q}^{1 - \frac{1}{d}}\right)$ such cells by Corollary 5.3. Note that this can be done locally,as we have broadcast all cells to all servers. Then for each cell $\Delta$ ,we compute the number of halfspaces whose bounding halfplane intersects $\Delta$ ,denoted as $P\left( \Delta \right)$ . This is a sum-by-key problem on a total of $\mathop{\sum }\limits_{\Delta }P\left( \Delta \right)  = O\left( {{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}\right)$ key-value pairs,and thus the load is

步骤 (2)：部分覆盖的单元格。对于每个半空间，我们找出其边界半平面与之相交的所有单元格 $\Delta$。根据推论 5.3 可知，这样的单元格有 $O\left( {q}^{1 - \frac{1}{d}}\right)$ 个。注意，由于我们已将所有单元格广播到所有服务器，因此可以在本地完成此操作。然后，对于每个单元格 $\Delta$，我们计算其边界半平面与 $\Delta$ 相交的半空间的数量，记为 $P\left( \Delta \right)$。这是一个关于总共 $\mathop{\sum }\limits_{\Delta }P\left( \Delta \right)  = O\left( {{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}\right)$ 个键值对的按键求和问题，因此负载为

$$
O\left( {\frac{{N}_{2}}{p} \cdot  {q}^{1 - \frac{1}{d}}}\right) \text{.} \tag{3}
$$

For each cell $\Delta$ ,we allocate ${p}_{\Delta } = \left\lceil  {p \cdot  \frac{P\left( \Delta \right) }{\mathop{\sum }\limits_{\Delta }P\left( \Delta \right) }}\right\rceil$ servers to compute the join between the $\Theta \left( \frac{{N}_{1}}{q}\right)$ points in the cell and these $P\left( \Delta \right)$ halfspaces partially covering $\Delta$ . The total number of servers needed is $O\left( p\right)$ . Invoking the hypercube algorithm to compute their Cartesian product incurs a load of

对于每个单元格 $\Delta$，我们分配 ${p}_{\Delta } = \left\lceil  {p \cdot  \frac{P\left( \Delta \right) }{\mathop{\sum }\limits_{\Delta }P\left( \Delta \right) }}\right\rceil$ 台服务器来计算该单元格中的 $\Theta \left( \frac{{N}_{1}}{q}\right)$ 个点与部分覆盖 $\Delta$ 的这 $P\left( \Delta \right)$ 个半空间之间的连接。所需服务器的总数为 $O\left( p\right)$。调用超立方体算法来计算它们的笛卡尔积会产生的负载为

$$
O\left( {\sqrt{\frac{\frac{{N}_{1}}{q} \cdot  P\left( \Delta \right) }{{p}_{\Delta }}} + \frac{\frac{{N}_{1}}{q} + P\left( \Delta \right) }{{p}_{\Delta }}}\right)  = O\left( {\sqrt{\frac{{N}_{1}{N}_{2}}{p{q}^{\frac{1}{d}}}} + \frac{{N}_{1}}{q} + \frac{{N}_{2}{q}^{1 - \frac{1}{d}}}{p}}\right) . \tag{4}
$$

Choosing $q = {p}^{\frac{d}{{2d} - 1}}$ balances the terms in (3) and (4),and the load becomes $O\left( \frac{\mathrm{{IN}}}{q}\right)  = O\left( \frac{\mathrm{{IN}}}{{p}^{d/\left( {{2d} - 1}\right) }}\right)$ .

选择 $q = {p}^{\frac{d}{{2d} - 1}}$ 可以平衡 (3) 和 (4) 中的项，负载变为 $O\left( \frac{\mathrm{{IN}}}{q}\right)  = O\left( \frac{\mathrm{{IN}}}{{p}^{d/\left( {{2d} - 1}\right) }}\right)$。

Step (3): Fully covered cells. In the intervals-containing-points algorithm, fully covered intervals are dealt with in a way similar to the partially covered intervals, as we can compute OUT exactly and set the right slab size. In this case,we may have used a cell size (i.e., ${N}_{1}/q$ ) that is too small in relation to OUT. This would result in too many replicated halfspaces to be distributed, exceeding the load target. Our strategy is thus to first estimate the join size for the fully covered cells (which is easier than computing OUT) and then rectify the mistake by restarting the whole algorithm with the right cell size, if needed.

步骤 (3)：完全覆盖的单元格。在包含点的区间算法中，完全覆盖的区间的处理方式与部分覆盖的区间类似，因为我们可以精确计算 OUT 并设置合适的平板大小。在这种情况下，相对于 OUT，我们使用的单元格大小（即 ${N}_{1}/q$）可能过小。这将导致需要分发过多的复制半空间，从而超过负载目标。因此，我们的策略是首先估计完全覆盖的单元格的连接大小（这比计算 OUT 更容易），然后如果需要，以合适的单元格大小重新启动整个算法来纠正错误。

Step (3.1): Join size estimation. For each cell $\Delta$ ,let $F\left( \Delta \right)$ be the number of halfspaces fully covering it,and let $K = \mathop{\sum }\limits_{\Delta }F\left( \Delta \right)$ . Since every point inside $\Delta$ joins with every halfspace fully covering $\Delta ,K \cdot  {N}_{1}/q$ is (a constant-factor approximation of) the remaining output size,and we will be able to estimate $K$ easily.

步骤 (3.1)：连接大小估计。对于每个单元格 $\Delta$，设 $F\left( \Delta \right)$ 为完全覆盖它的半空间的数量，并设 $K = \mathop{\sum }\limits_{\Delta }F\left( \Delta \right)$。由于 $\Delta$ 内的每个点都与完全覆盖它的每个半空间进行连接，$\Delta ,K \cdot  {N}_{1}/q$ 是（剩余输出大小的一个常数因子近似值），并且我们将能够轻松估计 $K$。

We first compute an $\left( \frac{{N}_{2}}{q}\right)$ -thresholded approximation of $F\left( \Delta \right)$ for each $\Delta$ ,denoted as $\widehat{F}\left( \Delta \right)$ . This can be done by sampling $O\left( {q\log p}\right)$ halfspaces and collecting them on one server. For each cell $\Delta$ ,we count the number of sampled halfspaces fully covering it and scale up appropriately. Consider any particular $\Delta$ . By the Chernoff inequality,with probability at least $1 - 1/{p}^{O\left( 1\right) }$ ,we get an $\left( \frac{{N}_{2}}{q}\right)$ -thresholded approximation of $F\left( \Delta \right)$ . Then applying a union bound on the $O\left( q\right)$ cells,we get an $\left( \frac{{\dot{N}}_{2}}{q}\right)$ -thresholded approximation for every $F\left( \Delta \right)$ with probability at least $1 - q/{p}^{O\left( 1\right) } = 1 - 1/{p}^{O\left( 1\right) }$ , as long as the hidden constant in the $O\left( {q\log p}\right)$ sample size is sufficiently large. We use these approximate $F\left( \Delta \right)$ ’s to compute $\widehat{K}$ (i.e., $\widehat{K} = \mathop{\sum }\limits_{\Delta }\widehat{F}\left( \Delta \right)$ ),which is then an ${N}_{2}$ -thresholded approximation of the true value of $K$ .

我们首先为每个$\Delta$计算$F\left( \Delta \right)$的$\left( \frac{{N}_{2}}{q}\right)$阈值近似值，记为$\widehat{F}\left( \Delta \right)$。这可以通过对$O\left( {q\log p}\right)$个半空间进行采样并将它们收集到一台服务器上来实现。对于每个单元格$\Delta$，我们计算完全覆盖它的采样半空间的数量，并进行适当的放大。考虑任意特定的$\Delta$。根据切尔诺夫不等式，至少以$1 - 1/{p}^{O\left( 1\right) }$的概率，我们可以得到$F\left( \Delta \right)$的$\left( \frac{{N}_{2}}{q}\right)$阈值近似值。然后对$O\left( q\right)$个单元格应用联合界，只要$O\left( {q\log p}\right)$样本量中的隐藏常数足够大，我们就可以至少以$1 - q/{p}^{O\left( 1\right) } = 1 - 1/{p}^{O\left( 1\right) }$的概率为每个$F\left( \Delta \right)$得到一个$\left( \frac{{\dot{N}}_{2}}{q}\right)$阈值近似值。我们使用这些近似的$F\left( \Delta \right)$来计算$\widehat{K}$（即$\widehat{K} = \mathop{\sum }\limits_{\Delta }\widehat{F}\left( \Delta \right)$），它是$K$真实值的${N}_{2}$阈值近似值。

Step (3.2): If $\widehat{K} < \frac{\mathbb{N} \cdot  p}{q}$ . As $\widehat{K}$ is an ${N}_{2}$ -thresholded approximation of $K,\widehat{K} < \frac{\mathbb{N} \cdot  p}{q}$ would imply that $K = O\left( \frac{\mathbb{N} \cdot  p}{q}\right)$ for $q = o\left( p\right)$ . In this case,we just break up each halfspace that fully covers $k$ cells into $k$ small pieces,which results in a total of $K$ pieces. Now every piece covers exactly one cell and thus joins with all the points in that cell. The problem now reduces to an equi-join on two relations of size ${N}_{1}$ and $K$ . Invoking the hypercube algorithm,the load is

步骤(3.2)：如果$\widehat{K} < \frac{\mathbb{N} \cdot  p}{q}$。由于$\widehat{K}$是$K,\widehat{K} < \frac{\mathbb{N} \cdot  p}{q}$的${N}_{2}$阈值近似值，这意味着对于$q = o\left( p\right)$有$K = O\left( \frac{\mathbb{N} \cdot  p}{q}\right)$。在这种情况下，我们只需将完全覆盖$k$个单元格的每个半空间拆分成$k$个小块，这样总共会得到$K$个小块。现在每个小块恰好覆盖一个单元格，因此会与该单元格中的所有点进行连接。现在问题简化为对大小分别为${N}_{1}$和$K$的两个关系进行等值连接。调用超立方体算法，负载为

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{K + {N}_{1}}{p}}\right)  = O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{q}}\right) .
$$

Step (3.3): If $\widehat{K} > \frac{\mathrm{{IN}} \cdot  p}{q}$ . In this case,we cannot afford to reduce the problem to an equi-join,as the halfspaces cover too many cells. This means we have used a cell size too small, and we need to restart the whole algorithm with a new ${q}^{\prime }$ . Note that if $\widehat{K} > \frac{\mathbb{N} \cdot  p}{q}$ ,then $\widehat{K}$ must be a constant-factor approximation of $K$ . Let $\overline{\mathrm{{OUT}}} = \widehat{K} \cdot  \frac{\mathrm{{IN}}}{q}$ ,and $\overline{\mathrm{{OUT}}}$ is also a constant-factor approximation of the remaining output size,and thus $\widehat{\mathrm{{OUT}}} = O\left( \mathrm{{OUT}}\right)$ . We set ${q}^{\prime } = \mathrm{{IN}}/\sqrt{\frac{\widehat{\mathrm{{OUT}}}}{p}}$ where ${q}^{\prime } < q$ .

步骤(3.3)：如果$\widehat{K} > \frac{\mathrm{{IN}} \cdot  p}{q}$。在这种情况下，我们无法将问题简化为等值连接，因为半空间覆盖的单元格太多。这意味着我们使用的单元格尺寸太小，我们需要用新的${q}^{\prime }$重新启动整个算法。注意，如果$\widehat{K} > \frac{\mathbb{N} \cdot  p}{q}$，那么$\widehat{K}$一定是$K$的常数因子近似值。设$\overline{\mathrm{{OUT}}} = \widehat{K} \cdot  \frac{\mathrm{{IN}}}{q}$，并且$\overline{\mathrm{{OUT}}}$也是剩余输出大小的常数因子近似值，因此$\widehat{\mathrm{{OUT}}} = O\left( \mathrm{{OUT}}\right)$。我们设置${q}^{\prime } = \mathrm{{IN}}/\sqrt{\frac{\widehat{\mathrm{{OUT}}}}{p}}$，其中${q}^{\prime } < q$。

In the re-execution of the algorithm,we further merge every $O\left( {q/{q}^{\prime }}\right)$ cells into a bigger cell containing $\Theta \left( {{N}_{1}/{q}^{\prime }}\right)$ points. Now,each newly merged cell has non-constant description complexity,but since there are only a total of $O\left( q\right)$ cells,the entire mapping from these cells to the newly merged cells can be broadcast to all servers. Each server can still identify, for each of its points, which newly merged cell contains it. With the new ${q}^{\prime }$ ,step (1) has load $O\left( {{q}^{\prime }\log p}\right)  = O\left( {q\log p}\right)$ , and step (2) has load $O\left( \frac{\mathrm{{IN}}}{{q}^{\prime }}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ . In the step (3.1),let ${F}^{\prime }\left( \Delta \right)$ be the number of halfspaces covering a newly merged cell $\Delta$ ,and let ${K}^{\prime } = \mathop{\sum }\limits_{\Delta }{F}^{\prime }\left( \Delta \right)$ . Observe that each newly merged cell consists of $\Theta \left( {q/{q}^{\prime }}\right)$ old cells. This means that we have ${K}^{\prime } = O\left( {K{q}^{\prime }/q}\right)$ ,as any halfspace fully covering one newly merged cell must cover $\Theta \left( {q/{q}^{\prime }}\right)$ old cells (but not vice versa). We argue that in the re-execution, ${\widehat{K}}^{\prime } = O\left( \frac{\mathrm{{IN}} \cdot  p}{{q}^{\prime }}\right)$ always holds by the following fact:

在算法的重新执行过程中，我们进一步将每 $O\left( {q/{q}^{\prime }}\right)$ 个单元格合并为一个包含 $\Theta \left( {{N}_{1}/{q}^{\prime }}\right)$ 个点的更大单元格。现在，每个新合并的单元格具有非常数的描述复杂度，但由于总共只有 $O\left( q\right)$ 个单元格，因此从这些单元格到新合并单元格的整个映射可以广播到所有服务器。对于其每个点，每个服务器仍然可以识别出哪个新合并的单元格包含该点。使用新的 ${q}^{\prime }$ ，步骤 (1) 的负载为 $O\left( {{q}^{\prime }\log p}\right)  = O\left( {q\log p}\right)$ ，步骤 (2) 的负载为 $O\left( \frac{\mathrm{{IN}}}{{q}^{\prime }}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ 。在步骤 (3.1) 中，设 ${F}^{\prime }\left( \Delta \right)$ 为覆盖新合并单元格 $\Delta$ 的半空间数量，并设 ${K}^{\prime } = \mathop{\sum }\limits_{\Delta }{F}^{\prime }\left( \Delta \right)$ 。观察可知，每个新合并的单元格由 $\Theta \left( {q/{q}^{\prime }}\right)$ 个旧单元格组成。这意味着我们有 ${K}^{\prime } = O\left( {K{q}^{\prime }/q}\right)$ ，因为任何完全覆盖一个新合并单元格的半空间必定覆盖 $\Theta \left( {q/{q}^{\prime }}\right)$ 个旧单元格（但反之不成立）。我们通过以下事实证明，在重新执行过程中， ${\widehat{K}}^{\prime } = O\left( \frac{\mathrm{{IN}} \cdot  p}{{q}^{\prime }}\right)$ 始终成立：

$$
{\widehat{K}}^{\prime } = O\left( {{K}^{\prime } + \mathrm{{IN}}}\right) \;\left( {\widehat{K}}^{\prime }\right. \text{is a}{N}_{2}\text{-thresholded approximation of}\left. {K}^{\prime }\right) 
$$

$$
 = O\left( {K \cdot  \frac{{q}^{\prime }}{q} + \mathrm{{IN}}}\right)  = O\left( {\frac{\mathrm{{IN}} \cdot  {pq}}{{\left( {q}^{\prime }\right) }^{2}} \cdot  \frac{{q}^{\prime }}{q} + \mathrm{{IN}}}\right)  = O\left( \frac{\mathrm{{IN}} \cdot  p}{{q}^{\prime }}\right) .
$$

Then it always reaches step (3.2),with load complexity $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{{q}^{\prime }}}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ . Therefore, the re-execution,if it takes place,must have load $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ . Combining with the load of the first execution, we obtain the following result.

然后它总是会进入步骤 (3.2)，负载复杂度为 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{{q}^{\prime }}}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ 。因此，如果进行重新执行，其负载必定为 $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ 。结合首次执行的负载，我们得到以下结果。

THEOREM 5.4. There is a randomized algorithm that solves the halfspaces-containing-points problem in $O\left( 1\right)$ rounds and load

定理 5.4。存在一种随机算法，该算法能在 $O\left( 1\right)$ 轮内解决包含点的半空间问题，且负载为

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}} + {p}^{\frac{d}{{2d} - 1}}\log p}\right) .
$$

The algorithm succeeds with probability at least $1 - 1/{p}^{O\left( 1\right) }$ .

该算法成功的概率至少为 $1 - 1/{p}^{O\left( 1\right) }$ 。

## 6 SIMILARITY JOIN IN HIGH DIMENSIONS

## 6 高维中的相似性连接

So far,we have assumed that the dimensionality $d$ is a constant. The load for both the ${\ell }_{1}/{\ell }_{\infty }$ algorithm and the ${\ell }_{2}$ algorithm hides constant factors that depend on $d$ exponentially in the big-Oh notation. For the ${\ell }_{2}$ algorithm,even for constant $d$ ,the term $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ approaches $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ as $d$ grows,which is the load for computing the full Cartesian product.

到目前为止，我们假设维度 $d$ 是一个常数。${\ell }_{1}/{\ell }_{\infty }$ 算法和 ${\ell }_{2}$ 算法的负载在大 O 表示法中都隐藏了依赖于 $d$ 的指数级常数因子。对于 ${\ell }_{2}$ 算法，即使 $d$ 为常数，当 $d$ 增大时，项 $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ 趋近于 $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ ，这是计算全笛卡尔积的负载。

In this section, we present an algorithm for high-dimensional similarity joins based on LSH, where $d$ is not considered a constant. The nice thing about the LSH-based algorithm is that its load is independent of $d$ (we still measure the load in terms of tuples; if measured in words,then there will be an extra factor of $d$ ). The downside is that its output-dependent term will not depend on OUT exactly; instead,it will depend on OUT(cr),which is the output size when the distance threshold of the similarity join is made $c$ times larger,for some constant $c > 1$ .

在本节中，我们提出一种基于局部敏感哈希（LSH）的高维相似性连接算法，其中 $d$ 不被视为常数。基于 LSH 的算法的优点在于其负载与 $d$ 无关（我们仍然以元组来衡量负载；如果以字来衡量，则会有一个额外的 $d$ 因子）。缺点是其与输出相关的项并不精确依赖于 OUT；相反，它将依赖于 OUT(cr)，即当相似性连接的距离阈值增大 $c$ 倍时的输出大小，其中 $c > 1$ 为某个常数。

### 6.1 Locality Sensitive Hashing

### 6.1 局部敏感哈希

LSH is known to be an approximate solution for nearest neighbor search, as it may return a neighbor whose distance is $c$ times larger than the true nearest neighbor. In the case of similarity joins, all answers returned are truly within a distance of $r$ (since this can be easily verified),but its cost will depend on $\operatorname{OUT}\left( {cr}\right)$ instead of $\operatorname{OUT}$ . It is also an approximate solution in the sense that it approximates the optimal cost. The same notion of approximation has also been used for LSH-based similarity joins in the external memory model [28].

局部敏感哈希（LSH）被认为是最近邻搜索的一种近似解决方案，因为它可能返回一个距离比真正最近邻大$c$倍的邻居。在相似连接的情况下，返回的所有答案确实都在距离$r$之内（因为这可以很容易地验证），但其成本将取决于$\operatorname{OUT}\left( {cr}\right)$而不是$\operatorname{OUT}$。从近似最优成本的意义上来说，它也是一种近似解决方案。在外部内存模型中，基于LSH的相似连接也使用了相同的近似概念[28]。

Let $\operatorname{dist}\left( {\cdot , \cdot  }\right)$ be a distance function. For $c > 1,{p}_{1} > {p}_{2}$ ,recall that a family $\mathcal{H}$ of hash functions is $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -sensitive,if for any uniformly chosen hash function $h \in  \mathcal{H}$ ,and any two tuples $x,y$ , we have (1) $\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \geq  {p}_{1}$ if $\operatorname{dist}\left( {x,y}\right)  \leq  r$ ; and (2) $\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \leq  {p}_{2}$ if $\operatorname{dist}\left( {x,y}\right)  \geq  {cr}$ . In addition,we require $\mathcal{H}$ to be monotone-that is,for a randomly chosen $h \in  \mathcal{H},\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack$ is a non-increasing function of $\operatorname{dist}\left( {x,y}\right)$ . This requirement is not in the standard definition of LSH, but the LSH constructions for most metric spaces satisfy this property, include Hamming [20], ${\ell }_{1}\left\lbrack  {12}\right\rbrack  ,{\ell }_{2}\left\lbrack  {5,{12}}\right\rbrack$ ,Jaccard [9],and so forth. Meanwhile,all of these LHS hash functions can be represented efficiently in $O\left( d\right)$ space. Since in this section each tuple is a point with $d$ coordinates and we measure the load in terms of tuples, we can consider each LSH function to have size the same as $O\left( 1\right)$ tuples.

设$\operatorname{dist}\left( {\cdot , \cdot  }\right)$为一个距离函数。对于$c > 1,{p}_{1} > {p}_{2}$，回顾一下，如果对于任何均匀选择的哈希函数$h \in  \mathcal{H}$以及任意两个元组$x,y$，我们有：(1) 若$\operatorname{dist}\left( {x,y}\right)  \leq  r$，则$\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \geq  {p}_{1}$；(2) 若$\operatorname{dist}\left( {x,y}\right)  \geq  {cr}$，则$\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \leq  {p}_{2}$，那么哈希函数族$\mathcal{H}$是$\left( {r,{cr},{p}_{1},{p}_{2}}\right)$敏感的。此外，我们要求$\mathcal{H}$是单调的，即对于随机选择的$h \in  \mathcal{H},\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack$是$\operatorname{dist}\left( {x,y}\right)$的非增函数。这个要求不在LSH的标准定义中，但大多数度量空间的LSH构造都满足这个性质，包括汉明距离（Hamming）[20]、${\ell }_{1}\left\lbrack  {12}\right\rbrack  ,{\ell }_{2}\left\lbrack  {5,{12}}\right\rbrack$、杰卡德距离（Jaccard）[9]等等。同时，所有这些LHS哈希函数都可以在$O\left( d\right)$空间中高效表示。由于在本节中每个元组是一个具有$d$个坐标的点，并且我们以元组来衡量负载，因此我们可以认为每个LSH函数的大小与$O\left( 1\right)$个元组相同。

The quality of a hash function family is measured by $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}} < 1$ ,which is bounded by a constant that depends only on $c$ ,but not the dimensionality,and $\rho  \approx  1/c$ for many common distance functions $\left\lbrack  {5,{12},{20}}\right\rbrack$ . In a standard hash family $\mathcal{H},{p}_{1}$ and ${p}_{2}$ are both constants,but by concatenating multiple hash functions independently chosen from $\mathcal{H}$ ,we can make ${p}_{1}$ and ${p}_{2}$ arbitrarily small,whereas $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$ is kept fixed,or equivalently, ${p}_{2} = {p}_{1}^{1/\rho }$ .

哈希函数族的质量由$\rho  = \frac{\log {p}_{1}}{\log {p}_{2}} < 1$衡量，它受一个仅取决于$c$而不取决于维度的常数限制，并且对于许多常见的距离函数$\left\lbrack  {5,{12},{20}}\right\rbrack$有$\rho  \approx  1/c$。在标准哈希族中，$\mathcal{H},{p}_{1}$和${p}_{2}$都是常数，但通过独立地从$\mathcal{H}$中选择多个哈希函数进行串联，我们可以使${p}_{1}$和${p}_{2}$任意小，而$\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$保持固定，或者等价地，${p}_{2} = {p}_{1}^{1/\rho }$。

### 6.2 The Algorithm

### 6.2 算法

In the description of our algorithm that follows,we leave ${p}_{1},{p}_{2}$ unspecified,which will be later determined in the analysis. The algorithm proceeds in the following three steps:

在接下来对我们算法的描述中，我们不指定${p}_{1},{p}_{2}$，它将在后续分析中确定。该算法按以下三个步骤进行：

(1) Choose $q = 3 \cdot  1/{p}_{1} \cdot  \ln \mathrm{{IN}}$ hash functions ${h}_{1},\ldots ,{h}_{q} \in  \mathcal{H}$ randomly and independently, and broadcast them to all servers.

(1) 随机且独立地选择$q = 3 \cdot  1/{p}_{1} \cdot  \ln \mathrm{{IN}}$个哈希函数${h}_{1},\ldots ,{h}_{q} \in  \mathcal{H}$，并将它们广播到所有服务器。

(2) For each tuple $x$ ,make $q$ copies,and attach the pair $\left( {i,{h}_{i}\left( x\right) }\right)$ to each of these copies,for $i = 1,\ldots ,q$ .

(2) 对于每个元组 $x$，制作 $q$ 份副本，并将对 $\left( {i,{h}_{i}\left( x\right) }\right)$ 附加到这些副本中的每一份上，其中 $i = 1,\ldots ,q$。

(3) Perform an equi-join on all the copies of tuples,treating the pair $\left( {i,{h}_{i}\left( x\right) }\right)$ as the join value (i.e.,two tuples $x,y$ join if ${h}_{i}\left( x\right)  = {h}_{i}\left( y\right)$ for some $i$ ). For two joined tuples $x,y$ ,output them if $\operatorname{dist}\left( {x,y}\right)  \leq  r$ .

(3) 对元组的所有副本执行等值连接，将对 $\left( {i,{h}_{i}\left( x\right) }\right)$ 视为连接值（即，如果对于某个 $i$ 有 ${h}_{i}\left( x\right)  = {h}_{i}\left( y\right)$，则两个元组 $x,y$ 进行连接）。对于两个连接的元组 $x,y$，如果 $\operatorname{dist}\left( {x,y}\right)  \leq  r$ 则输出它们。

THEOREM 6.1. Assume that there is a monotone $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -sensitive LSH family with $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$ . Then there is a randomized similarity join algorithm that runs in $O\left( 1\right)$ rounds and with expected load

定理 6.1. 假设存在一个单调的 $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -敏感局部敏感哈希（LSH，Locality-Sensitive Hashing）族，其中 $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$。那么存在一个随机相似度连接算法，该算法在 $O\left( 1\right)$ 轮内运行，且预期负载为

$$
\widetilde{O}\left( {\sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}} + \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}} + \frac{\mathrm{{IN}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right) .
$$

The algorithm reports all join results with probability at least $1 - 1/\mathrm{{IN}}$ .

该算法以至少 $1 - 1/\mathrm{{IN}}$ 的概率报告所有连接结果。

Proof. Correctness of the algorithm follows from standard LSH analysis: for any two tuples $x,y$ with $\operatorname{dist}\left( {x,y}\right)  \leq  r$ ,the probability that they join on one ${h}_{i}$ is at least ${p}_{1}$ . Across $q$ independently chosen hash functions, the probability that they do not collide on any one of hash function is at most ${\left( 1 - {p}_{1}\right) }^{3 \cdot  1/{p}_{1} \cdot  \ln \mathrm{{IN}}} \leq  {e}^{-3 \cdot  \ln \mathrm{{IN}}} \leq  1/{\mathrm{{IN}}}^{3}$ . As there are at most ${\mathrm{{IN}}}^{2}$ pairs of tuples with their distance smaller than $r$ ,the probability that any one of them is not reported is at most $1/\mathrm{{IN}}$ by the union bound. Then we have probability of at least $1 - 1/\mathrm{{IN}}$ to report all join results.

证明。该算法的正确性可通过标准的局部敏感哈希（LSH）分析得出：对于任意两个元组 $x,y$，若 $\operatorname{dist}\left( {x,y}\right)  \leq  r$，则它们在一个 ${h}_{i}$ 上进行连接的概率至少为 ${p}_{1}$。在 $q$ 个独立选择的哈希函数中，它们在任何一个哈希函数上都不发生碰撞的概率至多为 ${\left( 1 - {p}_{1}\right) }^{3 \cdot  1/{p}_{1} \cdot  \ln \mathrm{{IN}}} \leq  {e}^{-3 \cdot  \ln \mathrm{{IN}}} \leq  1/{\mathrm{{IN}}}^{3}$。由于距离小于 $r$ 的元组对至多有 ${\mathrm{{IN}}}^{2}$ 对，根据联合界，其中任何一对未被报告的概率至多为 $1/\mathrm{{IN}}$。那么我们至少有 $1 - 1/\mathrm{{IN}}$ 的概率报告所有连接结果。

In the following,we analyze the load. Step (1) has load $\widetilde{O}\left( {1/{p}_{1}}\right)$ ,and step (2) is local computation. Thus, we only need to analyze step (3).

接下来，我们分析负载。步骤 (1) 的负载为 $\widetilde{O}\left( {1/{p}_{1}}\right)$，步骤 (2) 是局部计算。因此，我们只需要分析步骤 (3)。

The total number of tuples generated in step (2) is $\widetilde{O}\left( {\mathrm{{IN}}/{p}_{1}}\right)$ ,which is the input size to the equi-join,denoted as ${\mathrm{{IN}}}_{LSH}$ . The output size,denoted as ${\mathrm{{OUT}}}_{LSH}$ ,has its expectation bounded by

步骤 (2) 中生成的元组总数为 $\widetilde{O}\left( {\mathrm{{IN}}/{p}_{1}}\right)$，这是等值连接的输入大小，记为 ${\mathrm{{IN}}}_{LSH}$。输出大小记为 ${\mathrm{{OUT}}}_{LSH}$，其期望值有界为

$$
E\left\lbrack  {\mathrm{{OUT}}}_{LSH}\right\rbrack   = \widetilde{O}\left( {\mathrm{{OUT}}/{p}_{1} + \mathrm{{OUT}}\left( {cr}\right)  + {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - 1/\rho }}\right) .
$$

The first term is for all pairs(x,y)such that $\operatorname{dist}\left( {x,y}\right)  \leq  r$ . They could join on every ${h}_{i}$ . The second term is for(x,y)’s such that $r < \operatorname{dist}\left( {x,y}\right)  \leq  {cr}$ . There are $\operatorname{OUT}\left( {cr}\right)$ such pairs,and each pair has probability at most ${p}_{1}$ to join on each ${h}_{i}$ ,so each pair joins exactly once in expectation. The last term is for all(x,y)’s such that $\operatorname{dist}\left( {x,y}\right)  > {cr}$ . There are ${\mathrm{{IN}}}^{2}$ such pairs,and each pair joins with probability at most ${p}_{2}$ on each ${h}_{i}$ ,so they contribute the term ${\mathrm{{IN}}}^{2}{p}_{2}/{p}_{1} = {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - \rho }$ in expectation.

第一项针对所有满足 $\operatorname{dist}\left( {x,y}\right)  \leq  r$ 的对 (x, y)。它们可以在每个 ${h}_{i}$ 上进行连接。第二项针对满足 $r < \operatorname{dist}\left( {x,y}\right)  \leq  {cr}$ 的 (x, y) 对。这样的对有 $\operatorname{OUT}\left( {cr}\right)$ 个，且每一对在每个 ${h}_{i}$ 上进行连接的概率至多为 ${p}_{1}$，因此每一对在期望意义下恰好连接一次。最后一项针对所有满足 $\operatorname{dist}\left( {x,y}\right)  > {cr}$ 的 (x, y) 对。这样的对有 ${\mathrm{{IN}}}^{2}$ 个，且每一对在每个 ${h}_{i}$ 上进行连接的概率至多为 ${p}_{2}$，因此它们在期望意义下贡献项 ${\mathrm{{IN}}}^{2}{p}_{2}/{p}_{1} = {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - \rho }$。

Plugging these into Theorem 3.1,and using Jensen’s inequality $E\left\lbrack  \sqrt{X}\right\rbrack   \leq  \sqrt{E\left\lbrack  X\right\rbrack  }$ ,the expected load can be bounded by (the $\widetilde{O}$ of)

将这些代入定理3.1，并使用詹森不等式$E\left\lbrack  \sqrt{X}\right\rbrack   \leq  \sqrt{E\left\lbrack  X\right\rbrack  }$，期望负载可以由（……的$\widetilde{O}$）界定

$$
E\left\lbrack  {\sqrt{\frac{{\mathrm{{OUT}}}_{LSH}}{p}} + \frac{{\mathrm{{IN}}}_{LSH}}{p}}\right\rbrack   \leq  \frac{\sqrt{E\left\lbrack  {\mathrm{{OUT}}}_{LSH}\right\rbrack  }}{\sqrt{p}} + \frac{{\mathrm{{IN}}}_{LSH}}{p} \leq  \frac{\sqrt{\mathrm{{OUT}}/{p}_{1} + \mathrm{{OUT}}\left( {cr}\right)  + {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - 1/p}}}{\sqrt{p}} + \frac{\mathrm{{IN}}}{p{p}_{1}}
$$

$$
 \leq  \sqrt{\frac{\mathrm{{OUT}}}{p{p}_{1}}} + \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}} + \mathrm{{IN}}\sqrt{\frac{1}{p{p}_{1}^{1 - 1/\rho }}} + \frac{\mathrm{{IN}}}{p{p}_{1}}.
$$

Setting ${p}_{1} = 1/{p}^{\frac{\rho }{1 + \rho }}$ balances the last two terms,and we obtain the claimed bound in the theorem.

令${p}_{1} = 1/{p}^{\frac{\rho }{1 + \rho }}$使最后两项达到平衡，我们就得到了定理中所声称的界。

<!-- Media -->

<!-- figureText: ${R}_{1}$ ${R}_{3}$ ${R}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_19.jpg?x=575&y=257&w=412&h=251&r=0"/>

Fig. 4. An instance of a 3-relation chain join.

图4. 一个三元关系链连接的实例。

<!-- Media -->

Remark 1. Note that since $0 < \rho  < 1$ ,the input-dependent term is always better than performing a full Cartesian product. The output-term $O\left( \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}}\right)$ is also the best we can achieve for any LSH-based algorithm, by the following intuitive argument: due to its approximation nature, LSH cannot tell whether the distance between two tuples is smaller than $r$ or slightly above $r$ . A worst-case scenario is all the OUT( ${cr}$ ) pairs of tuples have distance slightly above $r$ but none of them actually joins. Unfortunately, since the hash functions cannot distinguish the two cases, any LSH-based algorithm will have to check all the OUT(cr) pairs to make sure that it does not miss any true join results. Finally,the term $O\left( \sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right)$ is also worse than the bound $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ we achieved in earlier sections. This is perhaps the best one can hope for as well,if $O\left( 1\right)$ rounds are required: to capture all joining pairs, $1/{p}_{1}$ repetitions are necessary,and two very close tuples may join in all these repetitions,introducing the extra $1/{p}_{1}$ factor in the output size. If we want to perform all of them in parallel, there seems to be no way to eliminate the redundancy beforehand. Of course, this is just an intuitive argument, not a formal proof.

注1. 注意，由于$0 < \rho  < 1$，依赖输入的项总是比执行全笛卡尔积要好。通过以下直观的论证可知，输出项$O\left( \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}}\right)$也是我们对于任何基于局部敏感哈希（LSH）的算法所能达到的最优结果：由于LSH的近似性质，它无法判断两个元组之间的距离是小于$r$还是略大于$r$。最坏的情况是，所有OUT(${cr}$)元组对的距离都略大于$r$，但实际上它们都不进行连接。不幸的是，由于哈希函数无法区分这两种情况，任何基于LSH的算法都必须检查所有OUT(cr)对，以确保不会遗漏任何真正的连接结果。最后，项$O\left( \sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right)$也比我们在前面章节中得到的界$O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$要差。如果需要$O\left( 1\right)$轮，这可能也是人们所能期望的最好结果了：为了捕获所有连接对，需要$1/{p}_{1}$次重复，并且两个非常接近的元组可能在所有这些重复中都进行连接，从而在输出大小中引入额外的$1/{p}_{1}$因子。如果我们想并行执行所有这些操作，似乎没有办法事先消除冗余。当然，这只是一个直观的论证，并非正式的证明。

## 7 A LOWER BOUND ON 3-RELATION CHAIN JOIN

## 7 三元关系链连接的下界

In this section, we consider the possibility of designing output-optimal algorithms for multi-way joins. We present a negative result, showing that this is not possible for the 3-relation equi-join ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)  \bowtie  {R}_{3}\left( {C,D}\right)$ . This means that one cannot hope to achieve output-optimality for arbitrary multi-way joins. Precisely characterizing the class of joins for which output-optimal algorithms exist seems an interesting and challenging direction of future work.

在本节中，我们考虑为多路连接设计输出最优算法的可能性。我们给出了一个否定结果，表明对于三元关系等值连接${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)  \bowtie  {R}_{3}\left( {C,D}\right)$，这是不可能的。这意味着人们不能期望对于任意的多路连接都能实现输出最优。精确刻画存在输出最优算法的连接类似乎是未来工作中一个有趣且具有挑战性的方向。

The first question is how an output-optimal term would look like for a 3-relation join. Applying the tuple-based argument in Section 1.2,each server can potentially produce $O\left( {L}^{3}\right)$ join results in a single round,and hence $O\left( {p{L}^{3}}\right)$ results over all $p$ servers in a constant number of round. Thus, an $O\left( {\left( \mathrm{{OUT}}/p\right) }^{1/3}\right)$ term is definitely output-optimal.

第一个问题是，对于三元关系连接，输出最优项会是什么样子。应用1.2节中基于元组的论证，每个服务器在一轮中可能产生$O\left( {L}^{3}\right)$个连接结果，因此在常数轮数内，所有$p$个服务器总共会产生$O\left( {p{L}^{3}}\right)$个结果。因此，一个$O\left( {\left( \mathrm{{OUT}}/p\right) }^{1/3}\right)$项绝对是输出最优的。

However, consider the instance shown in Figure 4, where we use vertices to represent attribute values and edges for tuples. On such an instance, the 3-relation join degenerates into the Cartesian product of ${R}_{1}$ and ${R}_{3}$ . Each server can produce at most $O\left( {L}^{2}\right)$ pairs of tuples in one round,one from ${R}_{1}$ and one from ${R}_{3}$ ,so we must have $O\left( {p{L}^{2}}\right)  = \Omega \left( \mathrm{{OUT}}\right)$ ,or $L = \Omega \left( \sqrt{\mathrm{{OUT}}/p}\right)$ . This means that the best possible output-dependent term is still $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ . In the following,we show that this is not possible either, assuming any meaningful input-dependent term.

然而，考虑图4所示的实例，其中我们用顶点表示属性值，用边表示元组。在这样一个实例上，三元关系连接退化为${R}_{1}$和${R}_{3}$的笛卡尔积。每个服务器在一轮中最多可以生成$O\left( {L}^{2}\right)$对元组，一对来自${R}_{1}$，另一对来自${R}_{3}$，因此我们必须有$O\left( {p{L}^{2}}\right)  = \Omega \left( \mathrm{{OUT}}\right)$，或者$L = \Omega \left( \sqrt{\mathrm{{OUT}}/p}\right)$。这意味着，最佳的可能依赖于输出的项仍然是$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$。在接下来的内容中，我们将证明，假设存在任何有意义的依赖于输入的项，这也是不可能的。

THEOREM 7.1. For any tuple-based algorithm computing a 3-relation chain join, if its load is in the form of

定理7.1。对于任何计算三元关系链连接的基于元组的算法，如果其负载形式为

$$
L = O\left( {\frac{\mathrm{{IN}}}{{p}^{\alpha }} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right) ,
$$

for some constant $\alpha$ ,then we must have $\alpha  \leq  1/2$ ,provided $\mathrm{{IN}}/{\log }^{2}\mathrm{{IN}} > c{p}^{3}$ for some sufficiently large constant $c$ .

对于某个常数$\alpha$，那么我们必须有$\alpha  \leq  1/2$，前提是对于某个足够大的常数$c$有$\mathrm{{IN}}/{\log }^{2}\mathrm{{IN}} > c{p}^{3}$。

<!-- Media -->

<!-- figureText: ${R}_{1}$ ${R}_{2}$ ${R}_{3}$ $C$ $D$ $A$ $B$ -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_20.jpg?x=571&y=256&w=426&h=526&r=0"/>

Fig. 5. A randomly constructed hard instance.

图5. 一个随机构造的困难实例。

<!-- Media -->

Note that there is an algorithm for the 3-relation chain join with load $\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right) \left\lbrack  {23}\right\rbrack$ ,without any output-dependent term. This means that it is meaningless to introduce the output-dependent term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ .

请注意，存在一种用于三元关系链连接的算法，其负载为$\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right) \left\lbrack  {23}\right\rbrack$，且没有任何依赖于输出的项。这意味着引入依赖于输出的项$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$是没有意义的。

Proof. Suppose that there is an algorithm with a claimed load $L$ in the form stated previously. We will construct a hard instance on which we must have $\alpha  \leq  1/2$ . Our construction is probabilistic, and we will show that with high probability, the constructed instance satisfies our needs.

证明。假设存在一种算法，其声称的负载$L$具有前面所述的形式。我们将构造一个困难实例，在该实例上我们必须有$\alpha  \leq  1/2$。我们的构造是概率性的，并且我们将证明，以高概率，所构造的实例满足我们的需求。

The construction is illustrated in Figure 5. Let $N = \frac{\mathrm{{IN}}}{3}$ . More precisely,attribute $A$ and $D$ each have $N$ distinct values. Each distinct value of $A$ appears in one tuple in ${R}_{1}$ ,and each distinct value of $D$ appears in one tuple in ${R}_{3}$ . Attributes $B$ and $C$ each have $\frac{N}{\sqrt{L}}$ distinct values. Each distinct value of $B$ appears in $\sqrt{L}$ tuples in ${R}_{1}$ ,and each distinct value in $C$ appears in $\sqrt{L}$ tuples in ${R}_{3}$ . Each distinct value of $B$ and each distinct value of $C$ have a probability of $\frac{L}{N}$ to form a tuple in ${R}_{2}$ . Note that ${R}_{1}$ and ${R}_{3}$ are deterministic and always have $N$ tuples,whereas ${R}_{2}$ is probabilistic with $N$ tuples in expectation,so the expected input size is IN. The output size is expected to be ${NL}$ . By the Chernoff inequality, the probability that the input size or output size deviates from their expectations by more than a constant fraction is $\exp \left( {-\Omega \left( N\right) }\right)$ .

该构造如图5所示。设$N = \frac{\mathrm{{IN}}}{3}$。更准确地说，属性$A$和$D$各自有$N$个不同的值。$A$的每个不同值出现在${R}_{1}$中的一个元组中，$D$的每个不同值出现在${R}_{3}$中的一个元组中。属性$B$和$C$各自有$\frac{N}{\sqrt{L}}$个不同的值。$B$的每个不同值出现在${R}_{1}$中的$\sqrt{L}$个元组中，$C$的每个不同值出现在${R}_{3}$中的$\sqrt{L}$个元组中。$B$的每个不同值和$C$的每个不同值有$\frac{L}{N}$的概率在${R}_{2}$中形成一个元组。请注意，${R}_{1}$和${R}_{3}$是确定性的，并且始终有$N$个元组，而${R}_{2}$是概率性的，期望有$N$个元组，因此期望的输入大小为IN。期望的输出大小为${NL}$。根据切尔诺夫不等式，输入大小或输出大小偏离其期望值超过一个常数比例的概率为$\exp \left( {-\Omega \left( N\right) }\right)$。

We allow all servers to access ${R}_{2}$ for free and only charge the algorithm for receiving tuples from ${R}_{1}$ and ${R}_{3}$ . Furthermore,we allow a server in each round to retrieve $L$ tuples from each of ${R}_{1}$ and ${R}_{3}$ ,for a total of ${2L}$ tuples,which actually exceeds the load constraint by a factor of 2 . More precisely, we bound the maximum number of join results a server can produce in a round, if it only receives $L$ tuples from ${R}_{1}$ and $L$ tuples from ${R}_{3}$ . Then we multiply this number by $p$ ,which must be larger than OUT. Note that this is purely a counting argument; if the same join result is produced at two or more servers, it is counted multiple times.

我们允许所有服务器免费访问${R}_{2}$，仅对从${R}_{1}$和${R}_{3}$接收元组的算法收费。此外，我们允许每一轮中的服务器从${R}_{1}$和${R}_{3}$各自检索$L$个元组，总共${2L}$个元组，这实际上使负载约束超出了两倍。更准确地说，如果服务器仅从${R}_{1}$接收$L$个元组，从${R}_{3}$接收$L$个元组，我们会限制其在一轮中所能产生的连接结果的最大数量。然后我们将这个数量乘以$p$，该结果必须大于OUT。请注意，这纯粹是一个计数论证；如果相同的连接结果在两个或更多服务器上产生，则会被多次计数。

First,we argue that a server should load ${R}_{1}$ and ${R}_{3}$ in whole groups to maximize its output size. Here,a group in ${R}_{1}$ (respectively, ${R}_{2}$ ) means all tuples sharing the same value on $B$ (respectively, $C$ ). Suppose that two groups in ${R}_{1}$ ,say, ${g}_{1}$ and ${g}_{2}$ ,are not loaded in full by a server: ${x}_{1} < \sqrt{L}$ tuples of ${g}_{1}$ and ${x}_{2} < \sqrt{L}$ tuples of ${g}_{2}$ have been loaded. Suppose that they respectively join with ${y}_{1}$ and ${y}_{2}$ tuples in ${R}_{3}$ that are loaded by the server. Note that they will produce ${x}_{1}{y}_{1} + {x}_{2}{y}_{2}$ join results. Without loss of generality,assume that ${y}_{1} \geq  {y}_{2}$ . Now consider the alternative where the server loads ${x}_{1} + 1$ tuples of ${g}_{1}$ and ${x}_{2} - 1$ tuples of ${g}_{2}$ . Then this would produce $\left( {{x}_{1} + 1}\right) {y}_{1} + \left( {{x}_{2} - 1}\right) {y}_{2} =$ ${x}_{1}{y}_{1} + {x}_{2}{y}_{2} + {y}_{1} - {y}_{2} \geq  {x}_{1}{y}_{1} + {x}_{2}{y}_{2}$ tuples. This means that by moving one tuple from ${g}_{2}$ to ${g}_{1}$ ,the server can only get more join results (at least not less). We can move tuples from one group to another as long as there are two non-full groups. Eventually, we arrive at a configuration where all groups of ${R}_{1}$ are loaded by the server in full,without decreasing the output size. Next,we apply the same transformation to the groups of ${R}_{3}$ to make all its groups full as well.

首先，我们认为服务器应该以完整组的形式加载${R}_{1}$和${R}_{3}$，以最大化其输出规模。这里，${R}_{1}$（分别地，${R}_{2}$）中的一个组是指在$B$（分别地，$C$）上具有相同值的所有元组。假设${R}_{1}$中的两个组，例如${g}_{1}$和${g}_{2}$，没有被服务器完整加载：${g}_{1}$的${x}_{1} < \sqrt{L}$个元组和${g}_{2}$的${x}_{2} < \sqrt{L}$个元组已被加载。假设它们分别与服务器加载的${R}_{3}$中的${y}_{1}$和${y}_{2}$个元组进行连接。请注意，它们将产生${x}_{1}{y}_{1} + {x}_{2}{y}_{2}$个连接结果。不失一般性，假设${y}_{1} \geq  {y}_{2}$。现在考虑另一种情况，即服务器加载${g}_{1}$的${x}_{1} + 1$个元组和${g}_{2}$的${x}_{2} - 1$个元组。那么这将产生$\left( {{x}_{1} + 1}\right) {y}_{1} + \left( {{x}_{2} - 1}\right) {y}_{2} =$${x}_{1}{y}_{1} + {x}_{2}{y}_{2} + {y}_{1} - {y}_{2} \geq  {x}_{1}{y}_{1} + {x}_{2}{y}_{2}$个元组。这意味着通过将一个元组从${g}_{2}$移动到${g}_{1}$，服务器只能获得更多（至少不会更少）的连接结果。只要存在两个未完整加载的组，我们就可以将元组从一个组移动到另一个组。最终，我们会得到一种配置，其中服务器完整加载了${R}_{1}$的所有组，而不会减小输出规模。接下来，我们对${R}_{3}$的组应用相同的转换，使其所有组也都被完整加载。

The preceding argument implies that we can assume each server in each round loads $\sqrt{L}$ full groups from ${R}_{1}$ and $\sqrt{L}$ full groups from ${R}_{3}$ ,with ${2L}$ tuples in total. In the following,we show that on a random instance constructed as earlier, with high probability, not many pairs of groups can join,no matter which subset of $2\sqrt{L}$ groups are loaded. Consider any subset of $\sqrt{L}$ groups from ${R}_{1}$ and any subset of $\sqrt{L}$ groups from ${R}_{3}$ . There are $L$ possible pairs of groups,and each pair has probability $\frac{L}{N}$ to join,so we expect to see $\frac{{L}^{2}}{N}$ pairs to join. By Chernoff bound,the probability that more than $2\frac{{L}^{2}}{N}$ pairs join is at most $\exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right) }\right)$ . There are $O\left( {\left( \frac{N}{\sqrt{L}}\right) }^{2\sqrt{L}}\right)$ different choices of $\sqrt{L}$ groups from ${R}_{1}$ and $\sqrt{L}$ groups from ${R}_{3}$ . Thus,the probability that one of them yields more than $2\frac{{L}^{2}}{N}$ joining groups is at most

上述论证表明，我们可以假设每一轮中的每个服务器从${R}_{1}$加载$\sqrt{L}$个完整的组，并从${R}_{3}$加载$\sqrt{L}$个完整的组，总共包含${2L}$个元组。接下来，我们将证明，在如前文所述构建的随机实例中，无论加载$2\sqrt{L}$个组的哪个子集，大概率不会有很多组对能够合并。考虑从${R}_{1}$中选取的任意$\sqrt{L}$个组的子集，以及从${R}_{3}$中选取的任意$\sqrt{L}$个组的子集。存在$L$种可能的组对，并且每一组对有$\frac{L}{N}$的概率合并，因此我们预计会有$\frac{{L}^{2}}{N}$个组对合并。根据切尔诺夫界（Chernoff bound），合并的组对数量超过$2\frac{{L}^{2}}{N}$的概率至多为$\exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right) }\right)$。从${R}_{1}$中选取$\sqrt{L}$个组以及从${R}_{3}$中选取$\sqrt{L}$个组，共有$O\left( {\left( \frac{N}{\sqrt{L}}\right) }^{2\sqrt{L}}\right)$种不同的选择。因此，其中某一种选择产生超过$2\frac{{L}^{2}}{N}$个合并组的概率至多为

$$
O\left( {\left( \frac{N}{\sqrt{L}}\right) }^{2\sqrt{L}}\right)  \cdot  \exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right) }\right)  = \exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right)  + O\left( {\sqrt{L} \cdot  \log N}\right) }\right) .
$$

This probability is exponentially small if

如果满足以下条件，这个概率呈指数级小

$$
\frac{{L}^{2}}{N} > {c}_{1}\sqrt{L} \cdot  \log N
$$

for some sufficiently large constant ${c}_{1}$ . Rearranging,we get

对于某个足够大的常数${c}_{1}$。经过整理，我们得到

$$
N\log N < \frac{1}{{c}_{1}} \cdot  {L}^{\frac{3}{2}}
$$

By Theorem 3.2,we always have $L = \Omega \left( {N/p}\right)$ ,so this is true as long as

根据定理3.2，我们始终有$L = \Omega \left( {N/p}\right)$，因此只要满足以下条件，该式就成立

$$
N\log N < \frac{1}{{c}_{2}} \cdot  {\left( \frac{N}{p}\right) }^{\frac{3}{2}},
$$

for some sufficiently large constant ${c}_{2}$ ,or $N/{\log }^{2}N > {c}_{3} \cdot  {p}^{3}$ for some sufficiently large constant ${c}_{3}$ .

对于某个足够大的常数 ${c}_{2}$，或者对于某个足够大的常数 ${c}_{3}$ 有 $N/{\log }^{2}N > {c}_{3} \cdot  {p}^{3}$。

By a union bound, we conclude that with high probability, a randomly constructed instance has $\mathrm{{IN}} = \Theta \left( N\right) ,\mathrm{{OUT}} = \Theta \left( {NL}\right)$ ,and on this instance,no matter which groups are chosen,no more than $\frac{2{L}^{2}}{N}$ pairs of groups can join. Since each pair of joining groups produces $L$ results,the $p$ servers in total produce $O\left( \frac{{L}^{3}p}{N}\right)$ results in a constant number of rounds. Thus,we have

通过联合界，我们得出结论：在高概率情况下，一个随机构造的实例满足 $\mathrm{{IN}} = \Theta \left( N\right) ,\mathrm{{OUT}} = \Theta \left( {NL}\right)$，并且在这个实例上，无论选择哪些组，最多只有 $\frac{2{L}^{2}}{N}$ 对组可以合并。由于每对合并的组会产生 $L$ 个结果，因此 $p$ 台服务器在常数轮数内总共会产生 $O\left( \frac{{L}^{3}p}{N}\right)$ 个结果。因此，我们有

$$
\frac{{L}^{3}p}{N} = \Omega \left( {NL}\right) 
$$

in other words,

换句话说

$$
L = \Omega \left( \frac{N}{\sqrt{p}}\right) 
$$

Suppose that an algorithm has a load in the form as stated in the theorem,then with OUT $=$ $\Theta \left( {NL}\right)$ ,we have

假设一个算法的负载形式如定理所述，那么在 OUT $=$ $\Theta \left( {NL}\right)$ 的情况下，我们有

$$
\frac{N}{{p}^{\alpha }} + \sqrt{\frac{NL}{p}} = \Omega \left( \frac{N}{\sqrt{p}}\right) .
$$

If $\alpha  > 1/2$ ,we must have

如果 $\alpha  > 1/2$，我们必然有

$$
\sqrt{\frac{NL}{p}} = \Omega \left( \frac{N}{\sqrt{p}}\right) ,
$$

or $L = \Omega \left( N\right)$ ,which is an even higher lower bound. Thus,we must have $\alpha  \leq  1/2$ .

或者 $L = \Omega \left( N\right)$，这是一个更高的下界。因此，我们必然有 $\alpha  \leq  1/2$。

## 8 PRACTICAL SIMILARITY JOIN ALGORITHMS

## 8 实用的相似连接算法

The algorithms designed in previous sections have achieved output-optimality in theory, but at the expense of using a large, although still constant, number of rounds. In large-scale distributed systems, each round incurs a substantial synchronization overhead, and the benefit of an asymptotic smaller load can be easily offset by the large system overhead. In this section, we describe a practical version of the equi-join algorithm from Section 3 and the one-dimensional similarity join algorithm from Section 4.1. Specifically, the practical versions of the algorithms need just one heavy round and a constant number of light rounds,where a round is light if its load is $\widetilde{O}\left( p\right)$ and heavy otherwise. Theoretically speaking, however, a light round might be even heavier than a heavy round if $p > L$ ,where $L$ is the optimal load of the algorithm. Thus,the practical version of the algorithm is theoretically optimal only when $p < L$ ,but this is always the case in practice. In fact,in most massively parallel systems in practice, $p$ is usually on the order of hundreds,whereas $L$ is at least ${10}^{6}$ ,and thus the benefit of using multiple light rounds to avoid a heavy round will be obvious. Indeed, in our experiments, we observe that the total time spent in all the light rounds accounts for less than ${10}\%$ of the wall-clock time. This means the heavy round dominates the cost, and it is important to restrict the algorithms to using only one heavy round.

前面几节设计的算法在理论上实现了输出最优，但代价是使用了大量（尽管仍然是常数）的轮数。在大规模分布式系统中，每一轮都会产生大量的同步开销，并且渐近更小的负载带来的好处很容易被巨大的系统开销所抵消。在本节中，我们描述第 3 节的等值连接算法和第 4.1 节的一维相似连接算法的实用版本。具体来说，这些算法的实用版本只需要一轮重负载轮和常数轮轻负载轮，其中如果一轮的负载为 $\widetilde{O}\left( p\right)$ 则为轻负载轮，否则为重负载轮。然而，从理论上讲，如果 $p > L$，轻负载轮可能比重负载轮更重，其中 $L$ 是算法的最优负载。因此，该算法的实用版本仅在 $p < L$ 时在理论上是最优的，但在实践中总是满足这个条件。事实上，在大多数实际的大规模并行系统中，$p$ 通常是数百的数量级，而 $L$ 至少为 ${10}^{6}$，因此使用多轮轻负载轮来避免一轮重负载轮的好处将是显而易见的。实际上，在我们的实验中，我们观察到所有轻负载轮所花费的总时间占时钟时间的比例小于 ${10}\%$。这意味着重负载轮主导了成本，因此限制算法只使用一轮重负载轮是很重要的。

### 8.1 Primitive Operations

### 8.1 基本操作

We first show how to implement the primitive operations from Section 2.2 in $O\left( 1\right)$ light rounds and at most one heavy round.

我们首先展示如何在 $O\left( 1\right)$ 轮轻负载轮和最多一轮重负载轮内实现第 2.2 节中的基本操作。

Sorting. The theoretically optimal BSP sorting algorithm by Goodrich is very complicated and not practical. In practice, a sampling-based algorithm, such as TeraSort [27, 31], is often used instead, which operates in the following two steps:

排序。古德里奇（Goodrich）提出的理论最优的 BSP 排序算法非常复杂且不实用。在实践中，通常使用基于采样的算法，如 TeraSort [27, 31]，该算法分以下两步进行：

(1) Take a random sample of $O\left( {p\log p}\right)$ elements,and collect them to a master server. The master server sorts this sample,takes the $p - 1$ splitters,denoted as ${s}_{1},\ldots ,{s}_{p - 1}$ ,that split this sample evenly into $p$ partitions,and broadcasts these splitters to all the servers. This step can be implemented in two light rounds.

(1) 随机抽取 $O\left( {p\log p}\right)$ 个元素的样本，并将它们收集到一台主服务器上。主服务器对这个样本进行排序，选取 $p - 1$ 个分割点，记为 ${s}_{1},\ldots ,{s}_{p - 1}$，这些分割点将样本均匀地划分为 $p$ 个分区，并将这些分割点广播给所有服务器。这一步可以在两轮轻负载轮内实现。

(2) Upon receiving the $p - 1$ splitters,each server scans its own elements. For each element $x$ ,the server finds the two consecutive splitters ${s}_{i}$ and ${s}_{i + 1}$ (define ${s}_{0} =  - \infty ,{s}_{p} = \infty$ ) such that ${s}_{i} \leq  x < {s}_{i + 1}$ and sends $x$ to server $i$ . This step requires one heavy round. After all the shuffling, each server locally sorts all elements that it has received.

(2) 每台服务器收到 $p - 1$ 个分割点后，扫描自己的元素。对于每个元素 $x$，服务器找到两个连续的分割点 ${s}_{i}$ 和 ${s}_{i + 1}$（定义 ${s}_{0} =  - \infty ,{s}_{p} = \infty$），使得 ${s}_{i} \leq  x < {s}_{i + 1}$，并将 $x$ 发送到服务器 $i$。这一步需要一轮重负载轮。在所有数据洗牌完成后，每台服务器对其收到的所有元素进行本地排序。

This sorting algorithm will be used in our sort-merge-join-based equi-join algorithm. Thus, strictly speaking, the practical version of our algorithm is still randomized. However, as shown in Tao et al. [31],the number of elements received by any server is at most $O\left( {\mathrm{{IN}}/p}\right)$ with high probability $1 - 1/{p}^{\Omega \left( 1\right) }$ ,where the exponent depends on the hidden constant in the sample size. In our implementation,we choose a sample size of ${10p}\log p$ ,which makes this probability very close to 1 .

这种排序算法将用于我们基于排序合并连接的等值连接算法。因此，严格来说，我们算法的实际版本仍然是随机化的。然而，正如陶等人 [31] 所示，任何服务器接收到的元素数量在高概率 $1 - 1/{p}^{\Omega \left( 1\right) }$ 下至多为 $O\left( {\mathrm{{IN}}/p}\right)$，其中指数取决于样本大小中的隐藏常数。在我们的实现中，我们选择的样本大小为 ${10p}\log p$，这使得该概率非常接近 1。

All prefix-sums. Instead of the full BSP algorithm for all prefix-sums [17], we use the following simple variant, which can be done in two light rounds. Recall that in the all prefix-sums problem, a large array $A$ is stored on $p$ servers in a consecutive manner,and the goal is to compute $S\left\lbrack  j\right\rbrack   =$ $A\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  j\right\rbrack$ for every $j$ ,where $\oplus$ is an associative operator:

全前缀和。我们没有使用全前缀和的完整 BSP 算法 [17]，而是使用以下简单的变体，该变体可以在两轮轻量级通信中完成。回顾一下，在全前缀和问题中，一个大数组 $A$ 以连续的方式存储在 $p$ 台服务器上，目标是为每个 $j$ 计算 $S\left\lbrack  j\right\rbrack   =$ $A\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  j\right\rbrack$，其中 $\oplus$ 是一个结合运算符：

(1) Each server $i$ computes the partial sum on its local sub-array,denoted $B\left\lbrack  i\right\rbrack$ ,and sends it to a master server.

(1) 每台服务器 $i$ 计算其本地子数组上的部分和，记为 $B\left\lbrack  i\right\rbrack$，并将其发送到主服务器。

(2) The master server computes the prefix-sums on the partial sums elements received-that is, $C\left\lbrack  i\right\rbrack   = B\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  B\left\lbrack  {i - 1}\right\rbrack$ and sends it to server $i$ ,for $i = 2,\ldots ,p$ . Define $C\left\lbrack  1\right\rbrack   = 0$ . Then server $i$ computes the prefix-sums for its local sub-array sequentially starting from $C\left\lbrack  i\right\rbrack$ .

(2) 主服务器计算接收到的部分和元素的前缀和，即 $C\left\lbrack  i\right\rbrack   = B\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  B\left\lbrack  {i - 1}\right\rbrack$，并将其发送到服务器 $i$，其中 $i = 2,\ldots ,p$。定义 $C\left\lbrack  1\right\rbrack   = 0$。然后服务器 $i$ 从 $C\left\lbrack  i\right\rbrack$ 开始顺序计算其本地子数组的前缀和。

Multi-Numbering. The multi-numbering problem was solved in Section 2.2.2 by sorting all elements and then running all prefix-sums,requiring one heavy round and $O\left( 1\right)$ light rounds. We observe that if the number of distinct keys is $O\left( p\right)$ ,then we do not have to sort all tuples,thus avoiding the heavy round. To do so,we first ask each server $i$ to compute a partial-count $N\left( {i,v}\right)$ ,which is the number of elements with key $v$ at server $i$ . Then we sort these partial-counts by $v$ . Since there are only $O\left( {p}^{2}\right)$ partial-counts,the load in the "heavy round" of TeraSort is only $O\left( {{p}^{2}/p}\right)  = O\left( p\right)$ , making it a light round. Then we run the all prefix-sums algorithm on these $O\left( {p}^{2}\right)$ partial-counts, using the same definition of $\oplus$ as in Section 2.2.2. This will give us,for each distinct $v$ ,the prefix-sums $C\left( {i,v}\right)  = N\left( {1,v}\right)  + \cdots  + N\left( {1,i - 1}\right) ,i = 2,\ldots ,p$ . Finally,we send the $C\left( {i,v}\right)$ ’s to server $i$ for all $v$ ,which then assigns consecutive numbers to all tuples with key $v$ ,starting from $C\left( {i,v}\right)$ (define $C\left( {1,v}\right)  = 0)$ .

多重编号。多重编号问题在 2.2.2 节中通过对所有元素进行排序，然后运行全前缀和算法解决，这需要一轮重量级通信和 $O\left( 1\right)$ 轮轻量级通信。我们注意到，如果不同键的数量为 $O\left( p\right)$，那么我们不必对所有元组进行排序，从而避免了重量级通信轮。为此，我们首先要求每台服务器 $i$ 计算一个部分计数 $N\left( {i,v}\right)$，即服务器 $i$ 上键为 $v$ 的元素数量。然后我们按 $v$ 对这些部分计数进行排序。由于只有 $O\left( {p}^{2}\right)$ 个部分计数，TeraSort 中“重量级通信轮”的负载仅为 $O\left( {{p}^{2}/p}\right)  = O\left( p\right)$，使其成为一轮轻量级通信。然后我们对这 $O\left( {p}^{2}\right)$ 个部分计数运行全前缀和算法，使用与 2.2.2 节中相同的 $\oplus$ 定义。这将为每个不同的 $v$ 给出前缀和 $C\left( {i,v}\right)  = N\left( {1,v}\right)  + \cdots  + N\left( {1,i - 1}\right) ,i = 2,\ldots ,p$。最后，我们将 $C\left( {i,v}\right)$ 发送到服务器 $i$ 以处理所有 $v$，然后服务器 $i$ 从 $C\left( {i,v}\right)$ 开始为所有键为 $v$ 的元组分配连续编号（定义 $C\left( {1,v}\right)  = 0)$）。

Therefore,when the number of distinct keys is $O\left( p\right)$ ,multi-numbering can be solved with $O\left( 1\right)$ light rounds and no heavy round.

因此，当不同键的数量为 $O\left( p\right)$ 时，多重编号问题可以通过 $O\left( 1\right)$ 轮轻量级通信解决，无需重量级通信轮。

Sum-by-Key. Similar to multi-numbering,the sum-by-key problem can be also solved with $O\left( 1\right)$ light rounds and no heavy round when the number of distinct keys is $O\left( p\right)$ . We first ask each server $i$ to compute the partial-sum $S\left( {i,v}\right)$ of all elements with key $v$ ,for each distinct key $v$ . Then we sort these $O\left( {p}^{2}\right)$ partial-sums by $v$ and run all prefix-sums using the same definition of $\oplus$ as in Section 2.2.2.

按键求和。与多编号问题类似，当不同键的数量为 $O\left( p\right)$ 时，按键求和问题也可以通过 $O\left( 1\right)$ 轮轻量级通信回合解决，且无需重量级通信回合。我们首先要求每个服务器 $i$ 为每个不同的键 $v$ 计算键为 $v$ 的所有元素的部分和 $S\left( {i,v}\right)$。然后，我们按 $v$ 对这些 $O\left( {p}^{2}\right)$ 个部分和进行排序，并使用与 2.2.2 节中相同的 $\oplus$ 定义来计算所有前缀和。

### 8.2 A Practical Equi-Join Algorithm

### 8.2 一种实用的等值连接算法

The practical equi-join algorithm for computing ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)$ is described in the following. It follows the TeraSort framework but replaces certain steps appropriately. As in Section 3, define ${R}_{i}\left( v\right)  = {\sigma }_{B = v}{R}_{i}$ and ${N}_{i}\left( v\right)  = \left| {{R}_{i}\left( v\right) }\right|$ :

以下描述了用于计算 ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)$ 的实用等值连接算法。它遵循 TeraSort 框架，但对某些步骤进行了适当替换。与第 3 节一样，定义 ${R}_{i}\left( v\right)  = {\sigma }_{B = v}{R}_{i}$ 和 ${N}_{i}\left( v\right)  = \left| {{R}_{i}\left( v\right) }\right|$：

(1) The first step is the same as in TeraSort-that is,we take a sample of $O\left( {p\log p}\right)$ tuples from ${R}_{1} \cup  {R}_{2}$ and collect them to one server. The server sorts the sample on attribute $B$ , breaking ties using other attributes of the tuples (we assume that no two tuples are the same on all attributes). The server then finds the $p - 1$ splitters,denoted ${s}_{1},\ldots ,{s}_{p - 1}$ ,from the sample and broadcasts them to all servers. This step requires two light rounds. Let $C = {\pi }_{B}\left\{  {{s}_{1},\ldots ,{s}_{p - 1}}\right\}$ be the set of distinct $B$ values of the splitters.

(1) 第一步与 TeraSort 中的相同，即我们从 ${R}_{1} \cup  {R}_{2}$ 中抽取 $O\left( {p\log p}\right)$ 个元组作为样本，并将它们收集到一台服务器上。该服务器按属性 $B$ 对样本进行排序，若出现平局则使用元组的其他属性来打破平局（我们假设没有两个元组在所有属性上都相同）。然后，该服务器从样本中找出 $p - 1$ 个分割点，记为 ${s}_{1},\ldots ,{s}_{p - 1}$，并将它们广播到所有服务器。此步骤需要两轮轻量级通信回合。设 $C = {\pi }_{B}\left\{  {{s}_{1},\ldots ,{s}_{p - 1}}\right\}$ 为分割点的不同 $B$ 值的集合。

(2) Upon receiving the $p - 1$ splitters,each server scans its own tuples. For each tuple $t$ ,the server finds the two consecutive splitters,say ${s}_{i}$ ,and ${s}_{i + 1}$ ,such that ${s}_{i} \leq  x < {s}_{i + 1}$ . Ties are broken using the same tie breaker as in step (1). If $x.B = {s}_{i}.B$ or $x.B = {s}_{i + 1}.B,x$ is said to be a crossing tuple and non-crossing otherwise. One simplification we have made is not to compute OUT exactly, which would require a heavy round. Instead, we only compute the join size on the crossing tuples-that is,we compute ${\mathrm{{IN}}}_{c} = \mathop{\sum }\limits_{{v \in  C}}{N}_{1}\left( v\right)  + {N}_{2}\left( v\right) ,{\mathrm{{OUT}}}_{c} =$ $\mathop{\sum }\limits_{{v \in  C}}{N}_{1}\left( v\right) {N}_{2}\left( v\right)$ and use ${\mathrm{{IN}}}_{c},{\mathrm{{OUT}}}_{c}$ in place of IN,OUT,respectively,in formula (2) for allocating the servers. Note that ${\mathrm{{OUT}}}_{c}$ can be computed in $O\left( 1\right)$ light rounds by running the sum-by-key algorithm with the $\leq  p - 1$ distinct values in $\mathcal{C}$ as keys. We then run the multi-numbering algorithm to assign consecutive numbers to the tuples for each distinct value $v \in  \mathcal{C}$ ,which also takes $O\left( 1\right)$ light rounds as described earlier.

(2) 每个服务器收到 $p - 1$ 个分割点后，会扫描自己的元组。对于每个元组 $t$，服务器会找到两个连续的分割点，比如 ${s}_{i}$ 和 ${s}_{i + 1}$，使得 ${s}_{i} \leq  x < {s}_{i + 1}$。使用与步骤 (1) 中相同的平局打破规则来处理平局情况。如果 $x.B = {s}_{i}.B$ 或 $x.B = {s}_{i + 1}.B,x$，则称该元组为交叉元组，否则为非交叉元组。我们做的一个简化是不精确计算 OUT，因为这需要一轮重量级通信回合。相反，我们只计算交叉元组上的连接大小，即计算 ${\mathrm{{IN}}}_{c} = \mathop{\sum }\limits_{{v \in  C}}{N}_{1}\left( v\right)  + {N}_{2}\left( v\right) ,{\mathrm{{OUT}}}_{c} =$ $\mathop{\sum }\limits_{{v \in  C}}{N}_{1}\left( v\right) {N}_{2}\left( v\right)$，并在公式 (2) 中分别用 ${\mathrm{{IN}}}_{c},{\mathrm{{OUT}}}_{c}$ 代替 IN、OUT 来分配服务器。注意，通过以 $\mathcal{C}$ 中的 $\leq  p - 1$ 个不同值作为键运行按键求和算法，可以在 $O\left( 1\right)$ 轮轻量级通信回合内计算出 ${\mathrm{{OUT}}}_{c}$。然后，我们运行多编号算法为每个不同的值 $v \in  \mathcal{C}$ 的元组分配连续编号，如前所述，这也需要 $O\left( 1\right)$ 轮轻量级通信回合。

(3) With high probability, the number of non-crossing tuples between each pair of consecutive splitters ${s}_{i},{s}_{i + 1}$ is $O\left( {\mathrm{{IN}}/p}\right)$ ,so we send these tuples to server $i$ ,which will compute their join results. For each $v \in  C$ ,we allocate

(3) 大概率情况下，每对连续分割器 ${s}_{i},{s}_{i + 1}$ 之间的非交叉元组数量为 $O\left( {\mathrm{{IN}}/p}\right)$，因此我们将这些元组发送到服务器 $i$，该服务器将计算它们的连接结果。对于每个 $v \in  C$，我们分配

$$
{p}_{v} = \max \left\{  {\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right)  + {N}_{2}\left( v\right) }{{\mathrm{{IN}}}_{c}}}\right\rbrack  ,\left\lbrack  {p \cdot  \frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{{\mathrm{{OUT}}}_{c}}}\right\rbrack  }\right\}  
$$

servers and invoke the deterministic Cartesian product algorithm. Note that all crossing tuples for each distinct value $c \in  \mathcal{C}$ have been numbered as well. Thus,all the Cartesian products and the join of non-crossing tuples can be computed in parallel in one heavy round.

服务器并调用确定性笛卡尔积算法。请注意，每个不同值 $c \in  \mathcal{C}$ 的所有交叉元组也已编号。因此，所有笛卡尔积和非交叉元组的连接可以在一轮重轮次中并行计算。

THEOREM 8.1. The preceding equi-join algorithm runs in $O\left( 1\right)$ light rounds and one heavy round, where each light round has a load of $O\left( {p\log p}\right)$ ,whereas the heavy round has load $O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{O\mathrm{{UT}}}{p}}}\right)$ . These bounds hold with probability at least $1 - 1/{p}^{O\left( 1\right) }$ .

定理 8.1。上述等值连接算法在 $O\left( 1\right)$ 轮轻轮次和一轮重轮次中运行，其中每轮轻轮次的负载为 $O\left( {p\log p}\right)$，而重轮次的负载为 $O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{O\mathrm{{UT}}}{p}}}\right)$。这些界限至少以概率 $1 - 1/{p}^{O\left( 1\right) }$ 成立。

Proof. The $O\left( {p\log p}\right)$ load in the light rounds is due to the first step of collecting the sample to the master server; all other light rounds actually have load only $O\left( p\right)$ . For the heavy round, each server receives $O\left( \frac{\mathrm{{IN}}}{p}\right)$ non-crossing tuples and $O\left( {\frac{{\mathrm{{IN}}}_{c}}{p} + \sqrt{\frac{{\mathrm{{OUT}}}_{c}}{p}}}\right)  = O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right)$ crossing tuples.

证明。轻轮次中的 $O\left( {p\log p}\right)$ 负载是由于将样本收集到主服务器的第一步；实际上，所有其他轻轮次的负载仅为 $O\left( p\right)$。对于重轮次，每个服务器接收 $O\left( \frac{\mathrm{{IN}}}{p}\right)$ 个非交叉元组和 $O\left( {\frac{{\mathrm{{IN}}}_{c}}{p} + \sqrt{\frac{{\mathrm{{OUT}}}_{c}}{p}}}\right)  = O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right)$ 个交叉元组。

### 8.3 A Practical Intervals-Containing-Points Algorithm

### 8.3 一种实用的区间包含点算法

Recall that in the intervals-containing-points problem,we are given ${N}_{1}$ points and ${N}_{2}$ intervals, and the goal is to report all the (point, interval) pairs such that the interval contains the point. Note that the ${\ell }_{1}/{\ell }_{\infty }$ similarity join problem reduces to the special case of this problem where all intervals have length ${2r}$ :

回顾一下，在区间包含点问题中，我们给定了 ${N}_{1}$ 个点和 ${N}_{2}$ 个区间，目标是报告所有满足区间包含点的（点，区间）对。请注意，${\ell }_{1}/{\ell }_{\infty }$ 相似度连接问题可归结为此问题的特殊情况，即所有区间的长度均为 ${2r}$：

(1) The first step is the same as that in the equi-join algorithm-that is, we take a sample of $O\left( {p\log p}\right)$ points and collect them to one server. This server finds $p - 1$ splitters as $\left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{p - 1}}\right\}$ (define ${s}_{0} =  - \infty ,{s}_{p} = \infty$ ) and broadcasts them to all servers. Each consecutive pair of splitters defines a slab. Note that there are $p$ slabs labeled as $1,2,\ldots ,p$ (i.e., $\left( {{s}_{i - 1},{s}_{i}}\right)$ defines slab $i$ ). With high probability,the number of points falling into each slab is $O\left( {\mathrm{{IN}}/p}\right)$ . Same as before,this step requires two light rounds.

(1) 第一步与等值连接算法中的步骤相同，即我们抽取 $O\left( {p\log p}\right)$ 个点的样本并将它们收集到一台服务器。该服务器找到 $p - 1$ 个分割器，记为 $\left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{p - 1}}\right\}$（定义 ${s}_{0} =  - \infty ,{s}_{p} = \infty$），并将它们广播到所有服务器。每对连续的分割器定义一个平板。请注意，有 $p$ 个平板，标记为 $1,2,\ldots ,p$（即 $\left( {{s}_{i - 1},{s}_{i}}\right)$ 定义平板 $i$）。大概率情况下，落入每个平板的点数为 $O\left( {\mathrm{{IN}}/p}\right)$。和之前一样，此步骤需要两轮轻轮次。

(2) Upon receiving the $p - 1$ splitters,each server scans its own data. Similarly,we do not compute OUT exactly. Instead, we compute the join size of points and intervals fully covering slabs-that is, ${\mathrm{{OUT}}}_{f} = \mathop{\sum }\limits_{i}P\left( i\right)  \cdot  F\left( i\right)$ ,where $P\left( i\right)$ is the number of points in slab $i$ and $F\left( i\right)$ is the number of intervals fully covering slab $i$ . For each point,as well as each endpoint of intervals, the servers find which slab it falls into (i.e., a pair of consecutive splitters ${s}_{i},{s}_{i + 1}$ for $x$ such that ${s}_{i} \leq  x < {s}_{i + 1}$ ). We first run the sum-by-key algorithm to compute all $P\left( i\right)$ ’s,where each point $x$ with ${s}_{i - 1} \leq  x < {s}_{i}$ is considered to have key $i$ and weight 1 . An interval $\left\lbrack  {x,y}\right\rbrack$ with ${s}_{i - 1} \leq  x < {s}_{i}$ and ${s}_{j} \leq  y \leq  {s}_{j + 1}$ fully covers slabs $i,i + 1,\ldots ,j$ . We ask each server to compute $p$ partial-counts in term of(i,j)for $i = 1,2,\ldots ,p$ ,indicating that there are $j$ intervals fully covering slab $j$ in its local data. There would be ${p}^{2}$ such pairs in total. Then we just run the sum-by-key algorithm to compute all $F\left( i\right)$ ’s-that is,each pair(i,j)is considered to have key $i$ and weight $j$ . All $P\left( i\right)$ ’s and $F\left( i\right)$ ’s will be sent to the master server to compute ${\mathrm{{OUT}}}_{f}$ . Note that this step only requires $O\left( 1\right)$ light rounds.

(2) 收到 $p - 1$ 分割器后，每个服务器会扫描其自身的数据。同样地，我们并不精确计算 OUT。相反，我们计算完全覆盖平板（slab）的点和区间的连接大小，即 ${\mathrm{{OUT}}}_{f} = \mathop{\sum }\limits_{i}P\left( i\right)  \cdot  F\left( i\right)$，其中 $P\left( i\right)$ 是平板 $i$ 中的点数，$F\left( i\right)$ 是完全覆盖平板 $i$ 的区间数。对于每个点以及区间的每个端点，服务器会找出它所属的平板（即，对于 $x$，找到一对连续的分割器 ${s}_{i},{s}_{i + 1}$，使得 ${s}_{i} \leq  x < {s}_{i + 1}$）。我们首先运行按键求和算法来计算所有的 $P\left( i\right)$，其中每个满足 ${s}_{i - 1} \leq  x < {s}_{i}$ 的点 $x$ 被视为键为 $i$ 且权重为 1。一个满足 ${s}_{i - 1} \leq  x < {s}_{i}$ 和 ${s}_{j} \leq  y \leq  {s}_{j + 1}$ 的区间 $\left\lbrack  {x,y}\right\rbrack$ 完全覆盖平板 $i,i + 1,\ldots ,j$。我们要求每个服务器针对 $i = 1,2,\ldots ,p$ 计算以 (i,j) 为项的 $p$ 个部分计数，这表明在其本地数据中有 $j$ 个区间完全覆盖平板 $j$。总共会有 ${p}^{2}$ 个这样的对。然后我们运行按键求和算法来计算所有的 $F\left( i\right)$，即，每个对 (i,j) 被视为键为 $i$ 且权重为 $j$。所有的 $P\left( i\right)$ 和 $F\left( i\right)$ 都将被发送到主服务器以计算 ${\mathrm{{OUT}}}_{f}$。请注意，此步骤仅需要 $O\left( 1\right)$ 轮轻量级操作。

(3) When ${\mathrm{{OUT}}}_{f}$ is larger than ${\mathrm{{IN}}}^{2}/p$ ,we need to merge slabs into larger ones such that each slab has size ${b}^{\prime } = \max \left\{  {\frac{\mathrm{{IN}}}{p},\sqrt{\frac{{\mathrm{{OUT}}}_{f}}{p}}}\right\}$ . If this happens,the master server just broadcasts new splitters to all servers and redefines all slabs. For each slab $s$ ,we count the number of intervals partially and fully covering it,denoted as $F\left( s\right)$ and $G\left( s\right)$ ,respectively. All $F\left( s\right)$ ’s and $G\left( s\right)$ ’s can be computed similarly in $O\left( 1\right)$ light rounds using the sum-by-key algorithm. Note that $\mathop{\sum }\limits_{s}F\left( s\right)  \leq  2 \cdot  \mathrm{{IN}}$ and $\mathop{\sum }\limits_{s}{b}^{\prime }G\left( s\right)  \leq  {\mathrm{{OUT}}}_{f}$ . We then run the multi-numbering algorithm to assign consecutive numbers to the intervals covering $s$ for each slab $s$ ,which also takes $O\left( 1\right)$ light rounds as described earlier.

(3) 当 ${\mathrm{{OUT}}}_{f}$ 大于 ${\mathrm{{IN}}}^{2}/p$ 时，我们需要将平板合并成更大的平板，使得每个平板的大小为 ${b}^{\prime } = \max \left\{  {\frac{\mathrm{{IN}}}{p},\sqrt{\frac{{\mathrm{{OUT}}}_{f}}{p}}}\right\}$。如果发生这种情况，主服务器只需将新的分割器广播给所有服务器并重新定义所有平板。对于每个平板 $s$，我们分别统计部分覆盖和完全覆盖它的区间数量，分别记为 $F\left( s\right)$ 和 $G\left( s\right)$。所有的 $F\left( s\right)$ 和 $G\left( s\right)$ 可以使用按键求和算法在 $O\left( 1\right)$ 轮轻量级操作中类似地计算出来。请注意，$\mathop{\sum }\limits_{s}F\left( s\right)  \leq  2 \cdot  \mathrm{{IN}}$ 和 $\mathop{\sum }\limits_{s}{b}^{\prime }G\left( s\right)  \leq  {\mathrm{{OUT}}}_{f}$。然后我们运行多编号算法，为每个平板 $s$ 覆盖 $s$ 的区间分配连续的编号，如前所述，这也需要 $O\left( 1\right)$ 轮轻量级操作。

(4) For each slab $s$ ,we allocate ${p}_{s} = \left\lceil  \frac{F\left( s\right)  + G\left( s\right) }{{b}^{\prime }}\right\rceil$ servers and invoke the deterministic Cartesian product algorithm. Note that all intervals covering $s$ for each slab $s$ have been numbered as well. All the Cartesian products can be computed in parallel in one heavy round. This step only uses $O\left( p\right)$ servers since $\mathop{\sum }\limits_{s}\left\lceil  \frac{F\left( s\right)  + G\left( s\right) }{{b}^{\prime }}\right\rceil   \leq  p + \mathop{\sum }\limits_{s}\frac{F\left( s\right) }{{b}^{\prime }} + \mathop{\sum }\limits_{s}\frac{G\left( s\right) }{{b}^{\prime }} \leq  p + \frac{2 \cdot  \mathrm{{IN}}}{{b}^{\prime }} + \frac{{\mathrm{{OUT}}}_{f} \leq  }{{b}^{\prime 2}}$ 4p.

(4) 对于每个平板 $s$，我们分配 ${p}_{s} = \left\lceil  \frac{F\left( s\right)  + G\left( s\right) }{{b}^{\prime }}\right\rceil$ 台服务器并调用确定性笛卡尔积算法。请注意，每个平板 $s$ 覆盖 $s$ 的所有区间也已编号。所有笛卡尔积可以在一轮繁重的计算中并行计算。由于 $\mathop{\sum }\limits_{s}\left\lceil  \frac{F\left( s\right)  + G\left( s\right) }{{b}^{\prime }}\right\rceil   \leq  p + \mathop{\sum }\limits_{s}\frac{F\left( s\right) }{{b}^{\prime }} + \mathop{\sum }\limits_{s}\frac{G\left( s\right) }{{b}^{\prime }} \leq  p + \frac{2 \cdot  \mathrm{{IN}}}{{b}^{\prime }} + \frac{{\mathrm{{OUT}}}_{f} \leq  }{{b}^{\prime 2}}$ 4p，此步骤仅使用 $O\left( p\right)$ 台服务器。

THEOREM 8.2. The preceding intervals-containing-points algorithm runs in $O\left( 1\right)$ light rounds and one heavy round,where each light round has a load of $O\left( {p\log p}\right)$ ,whereas the heavy round has load $O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right)$ . These bounds hold with probability at least $1 - 1/{p}^{O\left( 1\right) }$ .

定理 8.2。前面的含点区间算法在 $O\left( 1\right)$ 轮轻量级计算和一轮重量级计算中运行，其中每轮轻量级计算的负载为 $O\left( {p\log p}\right)$，而重量级计算的负载为 $O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right)$。这些边界至少以 $1 - 1/{p}^{O\left( 1\right) }$ 的概率成立。

Proof. The $O\left( {p\log p}\right)$ load in the light rounds is due to the first step of collecting the sample to the master server; all other light rounds actually have load only $O\left( p\right)$ . For the heavy round,each server receives $O\left( {b}^{\prime }\right)  = O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{{\mathrm{{OUT}}}_{f}}{p}}}\right)  = O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right)$ points and intervals.

证明。轻量级计算轮次中的 $O\left( {p\log p}\right)$ 负载是由于将样本收集到主服务器的第一步；实际上，所有其他轻量级计算轮次的负载仅为 $O\left( p\right)$。对于重量级计算轮次，每台服务器接收 $O\left( {b}^{\prime }\right)  = O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{{\mathrm{{OUT}}}_{f}}{p}}}\right)  = O\left( {\frac{\mathrm{{IN}}}{p} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right)$ 个点和区间。

Unfortunately, we do not know if the rectangles-containing-points algorithm can be made to run in one heavy round and $O\left( 1\right)$ light rounds. Furthermore,the logarithmic factor will cause a quick degradation of the algorithm in higher dimensions. Similarly,the ${\ell }_{2}$ similarity join algorithm is unlikely to be competitive in practice, due to its complexity and the large hidden constants. Thus, these results are mostly of theoretical interest only.

不幸的是，我们不知道含点矩形算法是否可以在一轮重量级计算和 $O\left( 1\right)$ 轮轻量级计算中运行。此外，对数因子会导致该算法在高维情况下性能迅速下降。同样，由于 ${\ell }_{2}$ 相似性连接算法的复杂性和较大的隐藏常数，它在实际应用中不太可能具有竞争力。因此，这些结果主要仅具有理论意义。

### 8.4 Implementation Details

### 8.4 实现细节

We have implemented the algorithms in Spark [33]. The main abstraction Spark provides is a resilient distributed dataset (RDD) that enables efficient data reuse in a broad range of applications. In Spark, each RDD is partitioned into a number of partitions. An operation on an RDD is performed by launching multiple tasks that each run on a partition in parallel across a cluster of computing nodes. Thus, each partition/task can be naturally modeled as a "server" in the MPC model. In our implementation,the two input relations ${R}_{1}$ and ${R}_{2}$ are stored as two RDDs,respectively. Note that the number of partitions/tasks, also called the level of parallelism, does not have to be equal to the number of physical processors in the system; in fact, the official Spark documentation recommends setting the level of parallelism to be two to three times the number of CPU cores in the cluster to allow the system to perform some dynamic load balancing at runtime. Thus,we set $p$ to be three times the number of CPU cores in the cluster for steps (1) and (2) in both algorithms from Sections 8.2 and 8.3. Since the ${p}_{v}$ servers allocated to each join key $v \in  \mathcal{C}$ in step (3) of the equi-join case,and the ${p}_{s}$ servers allocated to each slab $s$ in step (4) of the intervals-containing-points case, should be arranged in a grid with dimensions specified as in Section 2.2.5, which have to be integers, some rounding has to be done, and the total number of allocated servers might range from one to two times of value $p$ . In this way,each of $p$ servers participates in one or two Cartesian product computations.

我们已在Spark [33]中实现了这些算法。Spark提供的主要抽象是弹性分布式数据集（RDD），它能在广泛的应用中实现高效的数据重用。在Spark中，每个RDD被划分为多个分区。对RDD的操作是通过启动多个任务来执行的，每个任务在计算节点集群中的一个分区上并行运行。因此，每个分区/任务可以自然地建模为多方计算（MPC）模型中的一个“服务器”。在我们的实现中，两个输入关系${R}_{1}$和${R}_{2}$分别存储为两个RDD。请注意，分区/任务的数量（也称为并行度）不必等于系统中物理处理器的数量；实际上，Spark官方文档建议将并行度设置为集群中CPU核心数量的两到三倍，以便系统在运行时进行一些动态负载均衡。因此，对于第8.2节和第8.3节中两种算法的步骤（1）和（2），我们将$p$设置为集群中CPU核心数量的三倍。由于在等值连接情况的步骤（3）中分配给每个连接键$v \in  \mathcal{C}$的${p}_{v}$个服务器，以及在区间包含点情况的步骤（4）中分配给每个平板$s$的${p}_{s}$个服务器，应按照第2.2.5节中指定的维度排列成一个网格，而这些维度必须是整数，因此需要进行一些取整操作，分配的服务器总数可能在值$p$的一到两倍之间。通过这种方式，$p$个服务器中的每一个都参与一到两次笛卡尔积计算。

The vanilla join operator provided by Spark performs a simple hash join. More precisely, all tuples with join value $v$ are sent to server $v{\;\operatorname{mod}\;p}$ ,which then computes the join of all tuples received. This requires a complicated and expensive shuffle operation that involves construction of hash tables, data serialization, and network I/O. We have implemented four algorithms: (1) the practical equi-join algorithm from Section 8.2 and the hash-based equi-join algorithm of Beame et al. [8], (2) the practical one-dimensional similarity join algorithm from Section 8.3, and (3) the LSH-based similarity join algorithm from Section 6.2. We did not implement our algorithms from scratch but leverage on Spark's own join operator.

Spark提供的普通连接运算符执行简单的哈希连接。更准确地说，所有连接值为$v$的元组都会被发送到服务器$v{\;\operatorname{mod}\;p}$，然后该服务器计算接收到的所有元组的连接。这需要一个复杂且开销大的洗牌操作，涉及哈希表的构建、数据序列化和网络I/O。我们实现了四种算法：（1）第8.2节中的实用等值连接算法和Beame等人[8]提出的基于哈希的等值连接算法；（2）第8.3节中的实用一维相似性连接算法；（3）第6.2节中的基于局部敏感哈希（LSH）的相似性连接算法。我们并非从头开始实现这些算法，而是利用了Spark自身的连接运算符。

In the practical equi-join algorithm, the idea is to "repackage" the join keys in a way such that the vanilla hash join will work just as how our algorithm would perform the join. More precisely, when our algorithm wants to send a tuple with join value $v$ to server $i$ ,we turn it into a tuple with key ${pv} + i$ . In the case of crossing tuples,which need to be sent to multiple servers,we use Spark's flatMap method to generate multiple tuples with different repackaged keys. It can be easily verified that the results from the vanilla hash join on the repackaged keys are exactly the same as those from the original join. Through this trick, we can implement our algorithm completely in the user space without modifying any existing code of Spark. The benefit of this approach is that our algorithm can automatically enjoy the improvements of any future updates to Spark's shuffle operation. Meanwhile, we point out that if one were to implement the algorithm inside the Spark core, we could have some (very) small savings by avoiding generating the intermediate repackaged tuples. The multi-numbering and sum-by-key algorithms described earlier use a constant number of light rounds,and the load in each round is $O\left( p\right)$ . Because $p$ is at most a few hundred in our experimental setting,we simply collect all the $O\left( {p}^{2}\right)$ partial-counts or partial-sums to the driver node, which then computes the prefix-sums and broadcasts them back to all workers using Spark's broadcast variables.

在实用等值连接算法中，其思路是以一种方式“重新打包”连接键，使得普通的哈希连接的执行效果与我们的算法执行连接的效果相同。更准确地说，当我们的算法想将连接值为$v$的元组发送到服务器$i$时，我们将其转换为键为${pv} + i$的元组。对于需要发送到多个服务器的交叉元组，我们使用Spark的flatMap方法生成多个具有不同重新打包键的元组。可以很容易地验证，对重新打包的键进行普通哈希连接的结果与原始连接的结果完全相同。通过这个技巧，我们可以在用户空间完全实现我们的算法，而无需修改Spark的任何现有代码。这种方法的好处是，我们的算法可以自动受益于Spark洗牌操作未来的任何更新改进。同时，我们指出，如果要在Spark核心内部实现该算法，我们可以通过避免生成中间重新打包的元组节省一些（非常）小的开销。前面描述的多编号和按键求和算法使用固定数量的轻量级轮次，并且每一轮的负载为$O\left( p\right)$。由于在我们的实验设置中$p$最多只有几百，我们只需将所有的$O\left( {p}^{2}\right)$个部分计数或部分和收集到驱动节点，然后驱动节点计算前缀和，并使用Spark的广播变量将它们广播回所有工作节点。

The hash-based equi-join algorithm has been implemented as a counterpart of the practical equi-join algorithm, with the following differences. First, instead of using crossing and non-crossing tuples, this algorithm uses information about the heavy hitters in the join values, where a join value is heavy if it appears more than IN/p times. The heavy hitters can be found by first finding the "local" heavy hitters on each server (i.e.,those that appear more than IN $/{p}^{2}$ times on one server) and then collecting the local heavy hitters to the driver node, which then finds the global heavy hitters. Second,instead of computing the destination server $i$ from the splitters,this algorithm uses a random hash function $h$ to assign light hitter $v$ to server $h\left( v\right)$ . Similarly,we generate a repackaged tuple with join key ${pv} + h\left( v\right)$ . Third,instead of assigning tuples for a heavy hitter deterministically to a row or a column in the grid,this algorithm uses a hash function on the $A$ and $C$ attributes of the two relations. Similarly,we use flatMap to generate multiple repackaged tuples for each such tuple.

基于哈希的等值连接算法已作为实用等值连接算法的对应算法实现，存在以下差异。首先，该算法不使用交叉和非交叉元组，而是使用连接值中频繁项（heavy hitters）的信息，若某个连接值出现次数超过 IN/p 次，则称其为频繁项。可以先在每个服务器上找出“局部”频繁项（即，在一台服务器上出现次数超过 IN $/{p}^{2}$ 次的项），然后将局部频繁项收集到驱动节点，再由驱动节点找出全局频繁项。其次，该算法不通过分割器计算目标服务器 $i$，而是使用随机哈希函数 $h$ 将非频繁项 $v$ 分配到服务器 $h\left( v\right)$。同样，我们生成一个带有连接键 ${pv} + h\left( v\right)$ 的重新打包元组。第三，该算法不将频繁项的元组确定性地分配到网格中的行或列，而是对两个关系的 $A$ 和 $C$ 属性使用哈希函数。同样，我们使用 flatMap 为每个这样的元组生成多个重新打包的元组。

The implementation of practical one-dimensional similarity join algorithm follows the same approach. The difference from the equi-join algorithms is that we need to assign join keys for points as well as intervals before invoking the Cartesian product algorithm. In our algorithm, the one-dimensional space is decomposed into a set of disjoint slabs in sorted order, whose ids are used as the join keys. Each slab induces one instance of Cartesian product. Note that each point participates in one Cartesian product, whereas each interval may participate in multiple Cartesian products. Implied by the algorithm from Section 2.2.5, servers allocated for each slab are arranged into one row such that points are broadcasted to the servers and interval are evenly distributed across the servers. Similarly, we use flatMap to generate multiple repackaged tuples for each point and interval. Note that for each pair of (point, interval) found by the Cartesian product, we report it if the point is indeed inside the interval.

实用的一维相似性连接算法的实现采用相同的方法。与等值连接算法的不同之处在于，在调用笛卡尔积算法之前，我们需要为点和区间分配连接键。在我们的算法中，一维空间被分解为一组按排序顺序排列的不相交的条带（slab），其 ID 用作连接键。每个条带会引发一次笛卡尔积实例。请注意，每个点参与一次笛卡尔积，而每个区间可能参与多次笛卡尔积。根据 2.2.5 节的算法，为每个条带分配的服务器排列成一行，以便将点广播到服务器，并将区间均匀分布在服务器之间。同样，我们使用 flatMap 为每个点和区间生成多个重新打包的元组。请注意，对于笛卡尔积找到的每一对（点，区间），如果点确实在区间内，我们就报告该结果。

Incorporating the LSH technique to the preceding equi-join algorithms, we are able to implement the LSH-based similarity join algorithm. The construction of the LSH family follows that in Gionis et al. [15], each hash function consists of multiple candidate hash keys, where each hash key is a set of bits of the input tuple, randomly sampled from the unary representation of the coordinates (real numbers are scaled up and rounded to integers). Given a similarity threshold, we first apply the LSH function to each tuple and use flatMap to generate multiple pairs for each candidate hash key. Then we use the hash key as the join key to perform a self-join by calling the equi-join algorithm as implemented earlier. For each pair of tuples in the join results, we compute their actual distance and report the result if it is indeed smaller than the similarity threshold. Thus, the algorithm will not return false positives but may miss a small number of true join results.

将局部敏感哈希（LSH）技术融入前面的等值连接算法，我们能够实现基于 LSH 的相似性连接算法。LSH 族的构造遵循 Gionis 等人 [15] 的方法，每个哈希函数由多个候选哈希键组成，其中每个哈希键是输入元组的一组位，这些位是从坐标的一元表示（实数被放大并四舍五入为整数）中随机采样得到的。给定一个相似性阈值，我们首先对每个元组应用 LSH 函数，并使用 flatMap 为每个候选哈希键生成多个对。然后，我们使用哈希键作为连接键，通过调用前面实现的等值连接算法进行自连接。对于连接结果中的每对元组，我们计算它们的实际距离，如果该距离确实小于相似性阈值，则报告该结果。因此，该算法不会返回误报，但可能会遗漏少量真正的连接结果。

## 9 EXPERIMENTS

## 9 实验

### 9.1 Experimental Setup

### 9.1 实验设置

All experiments have been performed on a Microsoft Azure HDInsight cluster running Spark 2.1. By default, the cluster is set up with six worker nodes each having eight CPU cores, whereas we vary the number of worker nodes from one to six in studying the scalability of algorithms. Each worker node has ${56}\mathrm{{GB}}$ of RAM,which is sufficient to keep all the data (raw and intermediate) so that even if all tuples are sent to one task, no data has to be swapped out to disk. Note that this is actually the ideal setting for the vanilla hash join algorithm, which may incur unbalanced loads when the data is highly skewed. If the worker nodes have smaller RAM, unbalanced loads can make the vanilla hash join algorithm even worse due to garbage collection and disk I/Os.

所有实验均在运行 Spark 2.1 的 Microsoft Azure HDInsight 集群上进行。默认情况下，集群设置为六个工作节点，每个节点有八个 CPU 核心，而在研究算法的可扩展性时，我们将工作节点的数量从一个变化到六个。每个工作节点有 ${56}\mathrm{{GB}}$ 的 RAM，这足以保存所有数据（原始数据和中间数据），因此即使所有元组都发送到一个任务，也无需将数据交换到磁盘。请注意，这实际上是普通哈希连接算法的理想设置，当数据高度倾斜时，该算法可能会导致负载不平衡。如果工作节点的 RAM 较小，由于垃圾回收和磁盘 I/O，负载不平衡会使普通哈希连接算法的性能更差。

We have evaluated three equi-join algorithms: the vanilla hash join algorithm provided by Spark, later referred to as Spark join; the hash-based output-optimal join algorithm of Beame et al. [8], referred to as hash join; and our sort-based output-optimal join algorithm, referred to as sort join.

我们评估了三种等值连接算法：Spark 提供的普通哈希连接算法，后文称为 Spark 连接；Beame 等人 [8] 提出的基于哈希的输出最优连接算法，称为哈希连接；以及我们基于排序的输出最优连接算法，称为排序连接。

For similarity join in one dimension, we evaluated three algorithms: the deterministic Cartesian product algorithm in Section 2.2.5, referred to as full join; the LSH-based similarity join algorithm, referred to as LSH join; and our intervals-containing-points algorithm, referred to as interval join. In higher dimensions, we evaluated the full join and the LSH join using different equi-join algorithms, referred to as LSH-Spark join, LSH-Hash join, and LSH-Sort join.

对于一维相似性连接，我们评估了三种算法：2.2.5 节中的确定性笛卡尔积算法，称为全连接；基于 LSH 的相似性连接算法，称为 LSH 连接；以及我们的包含点的区间算法，称为区间连接。在更高维度上，我们使用不同的等值连接算法评估了全连接和 LSH 连接，分别称为 LSH - Spark 连接、LSH - 哈希连接和 LSH - 排序连接。

As described earlier,we set $p$ to be three times the number of CPU cores,namely 144 for our cluster with 48 cores, when evaluating all algorithms.

如前所述，在评估所有算法时，我们将$p$设置为CPU核心数的三倍，即对于我们拥有48个核心的集群而言为144。

### 9.2 Datasets

### 9.2 数据集

We used both synthetic data and real data to test the performance of these algorithms.

我们使用合成数据和真实数据来测试这些算法的性能。

For equi-join algorithms, we generated two RDDs of (key, value) pairs and performed their equi-join on the key. The values are randomly generated strings of length 8 . The keys are generated according to some distribution. We tested two distributions. The first is the zipf distribution, where the frequency of the $i$ -th distinct key is proportional to ${i}^{-\alpha }$ . The parameter $\alpha  \geq  1$ controls the skewness of the distribution: larger $\alpha$ means larger skew. The second distribution is simply uniform, but we vary the number of distinct keys.

对于等值连接算法，我们生成了两个由（键，值）对组成的弹性分布式数据集（RDD），并对键进行等值连接。值是长度为8的随机生成字符串。键是根据某种分布生成的。我们测试了两种分布。第一种是齐普夫分布（Zipf distribution），其中第$i$个不同键的频率与${i}^{-\alpha }$成正比。参数$\alpha  \geq  1$控制分布的偏斜度：$\alpha$越大，偏斜度越大。第二种分布是简单的均匀分布，但我们改变不同键的数量。

For similarity join in one dimension, we generated two RDDs of point sets, one floating-point number for each point. The numbers are uniformally randomly generated from $\left\lbrack  {0,1}\right\rbrack$ . In fact,we also tested the Gaussian distribution, and the results are similar. In high dimensions, we use the same COREL dataset as in Gionis et al. [15], which is a set of 15,000 64-dimensional points. Each point corresponds to the histogram of one distinct color image taken from the COREL library. Note that we use the same LSH family, so all three equi-join algorithms have the same input size and give exactly the same results but only differ in running time.

对于一维相似性连接，我们生成了两个点集的弹性分布式数据集（RDD），每个点是一个浮点数。这些数字是从$\left\lbrack  {0,1}\right\rbrack$中均匀随机生成的。实际上，我们还测试了高斯分布，结果相似。在高维情况下，我们使用与吉奥尼斯（Gionis）等人[15]相同的COREL数据集，这是一组包含15000个64维点的集合。每个点对应于从COREL库中获取的一张不同彩色图像的直方图。请注意，我们使用相同的局部敏感哈希（LSH）族，因此所有三种等值连接算法具有相同的输入大小，并且给出完全相同的结果，只是运行时间不同。

<!-- Media -->

<!-- figureText: Spark Join IN $= {100},{000}$ 2.0 2.2 2.4 alpha 400 Hash Join Time(s) 200 100 1.0 1.2 1.4 1.6 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_28.jpg?x=147&y=256&w=628&h=493&r=0"/>

Fig. 6. Running times with different $\alpha$ .

图6. 不同$\alpha$下的运行时间。

<!-- figureText: alpha $= {1.1}$ Spark Join Hash Join Sort Join 3000 2500 2000 Time(s) 1500 1000 500 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_28.jpg?x=794&y=267&w=617&h=484&r=0"/>

Fig. 7. Running times with different input size.

图7. 不同输入大小下的运行时间。

<!-- figureText: 300 IN $= {500},{000}$ ,alpha $= {1.1}$ Hash Join Sort Join 45 Number of CPU cores 250 200 Time(s) 150 100 15 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_28.jpg?x=469&y=863&w=620&h=471&r=0"/>

Fig. 8. Running times with different numbers of CPU cores on the zipf distribution.

图8. 齐普夫分布下不同CPU核心数的运行时间。

<!-- Media -->

### 9.3 Experimental Results

### 9.3 实验结果

Results of equi-join on the zipf distribution. We have performed two sets of experiments, in which we vary $\alpha$ and data size,respectively. In the first set of experiments,the number of tuples in each relation is fixed at 100,000 and $\alpha$ increases from 1.0 to 2.5. From the results shown in Figure 6, we see an increasing gap in the running time between the Spark join and the two output-optimal algorithms. This is because the vanilla hash join algorithm used in Spark is bottlenecked by the heaviest key,as all tuples of the same key must be sent to the same server. As $\alpha$ increases, the skewness increases. When the data size is fixed,increasing $\alpha$ will increase the frequency of the heaviest key. However, the two output-optimal algorithms will allocate servers appropriately according to the frequencies, so they can much better balance the heavy keys with the light keys. Similarly,when we fix $\alpha$ and increase IN,the results in Figure 7 show a similar trend. This can also be explained by the same reason that increasing IN while keeping $\alpha$ fixed also increases the frequency of the heaviest key. We also tested the scalability of these algorithms by varying the number of worker nodes, and hence the total number of CPU cores in the cluster, and the result is shown in Figure 8. Both the two output-optimal algorithms will benefit from more CPU cores, whereas the Spark join almost stays the same, as all tuples of the heaviest key must be handled by the same CPU core, which is the bottleneck no matter how many workers are available.

齐普夫分布下等值连接的结果。我们进行了两组实验，分别改变$\alpha$和数据大小。在第一组实验中，每个关系中的元组数量固定为100000，$\alpha$从1.0增加到2.5。从图6所示的结果中，我们看到Spark连接和两种输出最优算法之间的运行时间差距在增大。这是因为Spark中使用的普通哈希连接算法受到最重键的瓶颈限制，因为相同键的所有元组必须发送到同一台服务器。随着$\alpha$的增加，偏斜度增大。当数据大小固定时，增加$\alpha$会增加最重键的频率。然而，两种输出最优算法会根据频率适当地分配服务器，因此它们能够更好地平衡重键和轻键。同样，当我们固定$\alpha$并增加IN时，图7中的结果显示出类似的趋势。这也可以用同样的原因来解释，即在固定$\alpha$的情况下增加IN也会增加最重键的频率。我们还通过改变工作节点的数量，从而改变集群中CPU核心的总数，来测试这些算法的可扩展性，结果如图8所示。两种输出最优算法都会从更多的CPU核心中受益，而Spark连接几乎保持不变，因为最重键的所有元组必须由同一个CPU核心处理，无论有多少工作节点可用，这都是瓶颈。

<!-- Media -->

<!-- figureText: 30 alpha $= {1.1},\mathrm{{IN}} = 1,{000},{000}$ Hash Join Sort Join 10 12 13 15 16 18 19 21 23 27 28 30 31 35 Time(s) 25 Number of tasks 20 15 5 0.3 0.4 0.5 0.6 0.8 2 5 9 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_29.jpg?x=148&y=253&w=1266&h=447&r=0"/>

Fig. 9. Running time distribution of tasks on the zipf distribution.

图9. 齐普夫分布下任务的运行时间分布。

<!-- Media -->

However, the difference between the two output-optimal algorithms is small, with sort join slightly better. Although, theoretically speaking, the randomized hash join algorithm of Beame et al. [23] is sub-optimal by a logarithmic factor, the actual difference is much smaller. There are two explanations for this phenomenon. First, the analysis in Beame et al. [23] is assuming the worst input. On the zipf distribution, the maximum load is determined by the hypercube algorithm on the heavy hitters. Since each grid of servers only has a small number of rows and columns while a large number of tuples are hashed to the rows and columns, a random allocation can already achieve good balance. Indeed, from the classical balls-in-bins analysis, we know that if sufficiently many balls are thrown into a small number of bins, then with high probability the bin sizes are within a constant factor of each other.

然而，两种输出最优算法之间的差异很小，排序连接略好一些。虽然从理论上讲，比姆（Beame）等人[23]的随机哈希连接算法在对数因子上是次优的，但实际差异要小得多。对于这种现象有两种解释。首先，比姆等人[23]的分析假设了最坏的输入。在齐普夫分布下，最大负载由针对频繁项的超立方体算法决定。由于服务器的每个网格只有少量的行和列，而大量的元组被哈希到这些行和列中，随机分配已经可以实现良好的平衡。实际上，从经典的球放入箱子分析中我们知道，如果将足够多的球放入少量的箱子中，那么箱子的大小很可能在彼此的常数因子范围内。

The second reason is more system related. To see how the two algorithms allocate work to the servers, we took a closer look at the running time distribution of the tasks, which is shown in Figure 9. We see that the sort join algorithm indeed yields a more concentrated distribution (i.e., more balanced tasks) than the hash join algorithm. Recall that we used a parallelism of 144 and the 144 tasks are dynamically allocated to the 48 CPU cores at runtime by Spark's scheduler, which uses sophisticated heuristics based on data locality for making scheduling decisions. This has the effect of "smoothing" things out since multiple small tasks can be allocated to the same CPU core. However, this dynamic load balancing incurs extra cost for data migration and increased overhead for the scheduler.

第二个原因与系统的关联性更强。为了了解这两种算法如何将工作分配给服务器，我们仔细研究了任务的运行时间分布，如图9所示。我们发现，排序连接算法（sort join algorithm）确实比哈希连接算法（hash join algorithm）产生了更集中的分布（即任务更加均衡）。请记住，我们使用的并行度为144，并且Spark调度器会在运行时将这144个任务动态分配到48个CPU核心上，该调度器会基于数据局部性使用复杂的启发式方法来做出调度决策。由于多个小任务可以分配到同一个CPU核心，这会产生“平滑”效果。然而，这种动态负载均衡会带来数据迁移的额外成本，并增加调度器的开销。

Since both the hash join algorithm and the sort join algorithm are randomized, we have also examined their stability over multiple repetitions. We repeated both algorithms multiple times with $\alpha  = {1.1}$ and $\mathrm{{IN}} = 1,{000},{000}$ . For the sort join algorithm,a different sample is drawn each time. For the hash join algorithm,we use a different hash function $h\left( x\right)  = {ax} + b$ ,where $a$ and $b$ are chosen randomly each time. Figure 10 shows the results of the running times of the algorithms on 80 repetitions. We see that both algorithms are quite stable. Note that due to system perturbation, even a completely deterministic algorithm will have some small variations from time to time.

由于哈希连接算法和排序连接算法都是随机化的，我们还研究了它们在多次重复运行中的稳定性。我们使用$\alpha  = {1.1}$和$\mathrm{{IN}} = 1,{000},{000}$对这两种算法进行了多次重复运行。对于排序连接算法，每次都会抽取不同的样本。对于哈希连接算法，我们使用不同的哈希函数$h\left( x\right)  = {ax} + b$，其中$a$和$b$每次都是随机选择的。图10展示了这两种算法在80次重复运行中的运行时间结果。我们发现这两种算法都相当稳定。请注意，由于系统扰动，即使是完全确定性的算法也会时不时出现一些小的波动。

Results of equi-join on uniform distribution. We note that the uniform distribution actually presents the best-case scenario for the vanilla Spark join, as the load is naturally balanced. Meanwhile,unless the number of distinct keys is much less than $p$ ,no key is heavy and the hypercube algorithm will not be needed. Therefore, the hash join algorithm essentially reduces to the vanilla Spark join. This can be observed from the results shown in Figures 11 and 12, where we vary the number of distinct keys and the input size, respectively. Since the hash join algorithm still needs to first collect data statistics, then it just realizes that all keys are light hitters. After that, it performs

均匀分布上等值连接的结果。我们注意到，对于普通的Spark连接而言，均匀分布实际上呈现出了最佳情况，因为负载自然是均衡的。同时，除非不同键的数量远小于$p$，否则没有键是重键，也就不需要超立方体算法。因此，哈希连接算法本质上退化为普通的Spark连接。这可以从图11和图12所示的结果中观察到，在这两个图中，我们分别改变了不同键的数量和输入大小。由于哈希连接算法仍然需要先收集数据统计信息，然后才会发现所有键都是轻量级键。之后，它会执行

<!-- Media -->

<!-- figureText: 95 alpha $= {1.1},\mathrm{{IN}} = 1,{000},{000}$ Hash Join Sort Join 40 50 60 80 Repeated times 90 Time(s) 10 20 30 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_30.jpg?x=149&y=255&w=1266&h=447&r=0"/>

Fig. 10. Stability of the two output-optimal algorithms on the zipf distribution.

图10. 两种输出最优算法在齐普夫分布上的稳定性。

<!-- figureText: 350 IN $= 3,{240},{000}$ Spark Join Hash Join Sort Join 800 1000 1200 Number of distinct keys 300 250 Time(s) 200 150 100 50 200 400 600 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_30.jpg?x=146&y=782&w=628&h=485&r=0"/>

keys.

键。

<!-- figureText: Number of distinct keys $= {144}$ and 600 3000000 4000000 Spark Join with 600 keys Hash Join with 600 keys 400 Sort Join with 600 keys Hash Join with 144 keys Sort Join with 144 keys 300 Time(s) 200 100 0 1000000 2000000 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_30.jpg?x=792&y=786&w=627&h=473&r=0"/>

<!-- Media -->

Fig. 11. Running time with different numbers of Fig. 12. Running time with different input size. exactly the same shuffle operation as Spark join, so we always see a small gap, proportional to IN, between the two algorithms.

图11. 不同键数量下的运行时间 图12. 不同输入大小下的运行时间 与Spark连接完全相同的混洗操作，因此我们总能看到这两种算法之间存在一个与IN成正比的小差距。

The comparison between the hash join and the sort join algorithm is more interesting. Note that the sort join algorithm achieves almost perfect load balance on the uniform distribution. Suppose that there are $n$ distinct keys,each with frequency $\mathrm{{IN}}/n$ . Then the number of non-crossing keys between every two consecutive splitters is almost always $n/p - 1$ ,as long as the splitters do not "drift away" by more than a distance of $\mathrm{{IN}}/n$ . There are exactly $p - 1$ crossing keys. But since each key's frequency is not high enough, the "grid" allocated to each crossing key degenerates to a $1 \times  1$ grid. Therefore,it is almost always the case that exactly $n/p$ distinct keys are assigned to each server.

哈希连接算法和排序连接算法之间的比较更有意思。请注意，排序连接算法在均匀分布上实现了几乎完美的负载均衡。假设存在$n$个不同的键，每个键的频率为$\mathrm{{IN}}/n$。那么，只要分割器的“漂移”距离不超过$\mathrm{{IN}}/n$，每两个连续分割器之间的非交叉键的数量几乎总是$n/p - 1$。恰好有$p - 1$个交叉键。但由于每个键的频率不够高，分配给每个交叉键的“网格”退化为一个$1 \times  1$网格。因此，几乎总是恰好有$n/p$个不同的键被分配到每个服务器。

However, the behavior of the hash join algorithm is characterized by the classical balls-in-bins problem [26],where we throw $n$ balls into $p$ bins randomly and study the number of balls landing in the largest bin. It is known that (1) if $n = \Theta \left( p\right)$ ,then with high probability the largest bin receives $\Theta \left( {\log p/\log \log p}\right)$ balls,and (2) when $n = \Omega \left( {p\log p}\right)$ ,then with high probability the largest bin receives $\Theta \left( {n/p}\right)$ balls. Thus,when $n$ is large,we expect the hash join algorithm to perform well, whereas for small $n$ ,it can be sub-optimal by an $\Theta \left( {\log p/\log \log p}\right)$ factor. This is can be verified in Figure 12. When there are 144 distinct keys, each server in the sort join algorithm is allocated with exactly $1\mathrm{{key}}$ ,wherease the largest server in the hash join gets approximately $3\mathrm{{keys}}$ . With the number of distinct keys increasing (e.g., 600 keys), the advantage of sort join will disappear since the hash join would achieve almost balanced load, whereas sort join always has extra overhead for server allocation. This can be further verified in Figure 13, where we plot the running time distribution of the 144 tasks of the two algorithms. Comparing Figures 13 and 9, we see that on the uniform dataset with 144 distinct keys, the tasks' running times are more unbalanced for the hash join algorithm, whereas the sort join algorithm produces tasks of almost equal running times.

然而，哈希连接算法的行为具有经典的球入箱问题 [26] 的特征，即在该问题中，我们将 $n$ 个球随机投入 $p$ 个箱子中，并研究落入最大箱子的球的数量。已知：(1) 如果 $n = \Theta \left( p\right)$ ，那么大概率最大的箱子会接收 $\Theta \left( {\log p/\log \log p}\right)$ 个球；(2) 当 $n = \Omega \left( {p\log p}\right)$ 时，那么大概率最大的箱子会接收 $\Theta \left( {n/p}\right)$ 个球。因此，当 $n$ 很大时，我们期望哈希连接算法表现良好，而当 $n$ 较小时，其性能可能会比最优情况差 $\Theta \left( {\log p/\log \log p}\right)$ 倍。这可以在图 12 中得到验证。当有 144 个不同的键时，排序连接算法中的每个服务器恰好分配到 $1\mathrm{{key}}$ ，而哈希连接中最大的服务器大约得到 $3\mathrm{{keys}}$ 。随着不同键的数量增加（例如 600 个键），排序连接的优势将消失，因为哈希连接将实现几乎均衡的负载，而排序连接总是在服务器分配上有额外的开销。这可以在图 13 中进一步验证，我们在图中绘制了两种算法的 144 个任务的运行时间分布。比较图 13 和图 9，我们可以看到，在具有 144 个不同键的均匀数据集上，哈希连接算法的任务运行时间更加不均衡，而排序连接算法产生的任务运行时间几乎相等。

<!-- Media -->

<!-- figureText: 120 Number of keys $= {144},\mathrm{{IN}} = 3,{240},{000}$ Hash Join Sort Join Time(s) 100 Number of tasks 80 60 40 20 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_31.jpg?x=145&y=254&w=1269&h=448&r=0"/>

Fig. 13. Running time distribution of tasks in the last stage on the uniform distribution.

图 13. 均匀分布下最后阶段任务的运行时间分布。

<!-- figureText: 450 Number of distinct keys $= {144},\mathrm{{IN}} = 3,{240},{000}$ Hash Join Sort Join 40 50 80 Repeated times 350 Time(s) 300 250 200 0 10 20 30 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_31.jpg?x=148&y=807&w=1266&h=447&r=0"/>

Fig. 14. Stability of two output-optimal algorithms on uniform distribution.

图 14. 均匀分布下两种输出最优算法的稳定性。

<!-- Media -->

As before, we also tested the stability of the two algorithms over repeated runs using different samples and random hash functions, using a uniform data set with 144 distinct keys. The results are shown in Figure 14. Compared to the results in Figure 10 on the zipf distribution, we see that the sort join algorithm is still very stable across different runs, but the hash join algorithm is much less stable. This is because on the zipf distribution, the largest load is determined by the heavy hitters, and a random allocation can deal with this case well because the hash function is applied on the other attributes of the tuples. However, on the uniform distribution, we hash on the join key and the maximum load is determined by the server receiving the most distinct join keys, which tends to be much less stable.

和之前一样，我们还使用具有 144 个不同键的均匀数据集，通过不同的样本和随机哈希函数对两种算法在多次运行中的稳定性进行了测试。结果如图 14 所示。与图 10 中关于齐普夫分布的结果相比，我们可以看到，排序连接算法在不同运行中仍然非常稳定，但哈希连接算法的稳定性则差得多。这是因为在齐普夫分布中，最大负载由高频项决定，并且随机分配可以很好地处理这种情况，因为哈希函数应用于元组的其他属性。然而，在均匀分布中，我们对连接键进行哈希处理，最大负载由接收最多不同连接键的服务器决定，这往往稳定性要差得多。

The scalability results of all algorithms are shown in Figure 15. As expected, all algorithms will benefit from more worker nodes because on the uniform distribution, load is always balanced.

所有算法的可扩展性结果如图 15 所示。正如预期的那样，所有算法都将从更多的工作节点中受益，因为在均匀分布下，负载总是均衡的。

Results of similarity joins in one dimension. We have performed two sets of experiments in which we vary the similarity threshold and the data size, respectively. In the first set of experiments,the number of points in each set is fixed at 100,000 and $r$ varies from 0.1 to 0.5,with the results shown in Figure 16. In the second set of experiments, we fix the similarity threshold at 0.01 and increase IN, and the results are shown in Figure 17.

一维相似性连接的结果。我们进行了两组实验，分别改变相似性阈值和数据大小。在第一组实验中，每个集合中的点数固定为 100,000， $r$ 从 0.1 变化到 0.5，结果如图 16 所示。在第二组实验中，我们将相似性阈值固定为 0.01 并增加 IN，结果如图 17 所示。

<!-- Media -->

<!-- figureText: 250 $\mathrm{{IN}} = 3,{240},{000}$ ,Number of distinct keys $= {1200}$ Spark Join Hash Join Sort Join 35 50 Number of CPU cores 225 200 175 Time(s) 150 125 100 10 25 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_32.jpg?x=472&y=254&w=620&h=471&r=0"/>

Fig. 15. Running times with different numbers of CPU cores on the uniform distribution.

图 15. 均匀分布下不同 CPU 核心数量的运行时间。

<!-- figureText: IN $= {100},{000}$ threshold $= {0.01}$ Interval Join Full Join LSH Join, $I = 1$ LSH Join, $1 = 2$ ${10}^{3}$ LSH Join, $1 = 3$ LSH Join, I = 4 Time(s) Fig. 17. Running times with different input sizes. 700 600 500 Interval Join Full join Time(s) 400 LSH Join, $I = 1$ LSH Join, $I = 2$ LSH Join, $1 = 3$ LSH Join, $I = 4$ 200 100 threshold r Fig. 16. Running times with different thresholds. -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_32.jpg?x=134&y=815&w=1285&h=524&r=0"/>

<!-- Media -->

From both sets of experiments, we see that, as expected, the full join has highest cost, whereas the interval join algorithm has the lowest cost. In both figures, the cost of the interval join grows, as its load has both an input-dependent term and an output-dependent term. Note that increasing $r$ will increase the output size. The cost of the full join algorithm is not affected by $r$ ,as it always enumerates all the ${\mathrm{{IN}}}^{2}$ pairs of tuples.

从这两组实验中，我们可以看到，正如预期的那样，全连接的成本最高，而区间连接算法的成本最低。在这两个图中，区间连接的成本会增加，因为其负载既有与输入相关的项，也有与输出相关的项。请注意，增加 $r$ 会增加输出大小。全连接算法的成本不受 $r$ 的影响，因为它总是枚举所有 ${\mathrm{{IN}}}^{2}$ 个元组对。

For the LSH join,we fixed the approximation ratio at 2 but varies $l$ ,the number of hash keys per hash function from 1 to 4 . Note that when fixing $l$ ,the input size of the equi-join is also fixed, thus the running time is not affected by $r$ . But a larger $r$ implies larger output size,the ratio of true results that can be recalled would be decreased,as shown in Table 1. Note that a larger $l$ implies larger input size of the equi-join and thus larger running times. Meanwhile,a larger $l$ increases the accuracy of the hash functions, resulting in a higher recall rate, as shown in Table 2. We see that the LSH join is outperformed by interval join, despite being an approximation algorithm. Essentially, LSH join is designed for the high-dimensional case. For one-dimensional similarity joins, a specialized algorithm like interval join is much more competitive.

对于局部敏感哈希连接（LSH join），我们将近似比率固定为2，但改变$l$，即每个哈希函数的哈希键数量，范围从1到4。请注意，当固定$l$时，等值连接的输入大小也会固定，因此运行时间不受$r$的影响。但更大的$r$意味着更大的输出大小，能够召回的真实结果的比例会降低，如表1所示。请注意，更大的$l$意味着等值连接的输入大小更大，因此运行时间更长。同时，更大的$l$会提高哈希函数的准确性，从而导致更高的召回率，如表2所示。我们发现，尽管局部敏感哈希连接是一种近似算法，但区间连接（interval join）的性能优于它。本质上，局部敏感哈希连接是为高维情况设计的。对于一维相似性连接，像区间连接这样的专用算法更具竞争力。

Results of similarity joins in high dimensions. In the last set of experiments, we perform self-similarity joins on the COREL dataset using Hamming distance. Given a distance threshold $r$ ,we first construct an(r,10r,0.9,0.1)-sensitive hash family such that any pair of points with distance smaller than $r$ has probability at least 0.9 to be hashed to the same bucket,whereas any pair of points with distance greater than ${10r}$ has probability no more than 0.1 to collide. In this set of experiments, we fix the number of candidate hash keys as 2 and vary the number bits per hash keys. The larger $r$ is,the less bits the hash keys will have,and more points will be hashed to the same bucket. The join size will also increase as well.

高维相似性连接的结果。在最后一组实验中，我们使用汉明距离（Hamming distance）对COREL数据集进行自相似性连接。给定一个距离阈值$r$，我们首先构建一个(r,10r,0.9,0.1) - 敏感哈希族，使得距离小于$r$的任意两点对被哈希到同一个桶中的概率至少为0.9，而距离大于${10r}$的任意两点对发生冲突的概率不超过0.1。在这组实验中，我们将候选哈希键的数量固定为2，并改变每个哈希键的位数。$r$越大，哈希键的位数就越少，更多的点将被哈希到同一个桶中。连接大小也会随之增加。

<!-- Media -->

Table 1. $k = {10},c = 2,l = 1$

表1. $k = {10},c = 2,l = 1$

<table><tr><td>$r$</td><td>0.01</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td></tr><tr><td>${p}_{1}$</td><td>0.9043</td><td>0.3487</td><td>0.1074</td><td>0.0282</td><td>0.0060</td></tr><tr><td>${p}_{2}$</td><td>0.8171</td><td>0.1074</td><td>0.0060</td><td>0.0001</td><td>1e-7</td></tr><tr><td>$\rho$</td><td>0.4975</td><td>0.4722</td><td>0.4368</td><td>0.3893</td><td>0.3174</td></tr><tr><td>Recall</td><td>0.9496</td><td>0.5863</td><td>0.3344</td><td>0.2548</td><td>0.2067</td></tr></table>

<table><tbody><tr><td>$r$</td><td>0.01</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td></tr><tr><td>${p}_{1}$</td><td>0.9043</td><td>0.3487</td><td>0.1074</td><td>0.0282</td><td>0.0060</td></tr><tr><td>${p}_{2}$</td><td>0.8171</td><td>0.1074</td><td>0.0060</td><td>0.0001</td><td>1e-7</td></tr><tr><td>$\rho$</td><td>0.4975</td><td>0.4722</td><td>0.4368</td><td>0.3893</td><td>0.3174</td></tr><tr><td>召回率（Recall）</td><td>0.9496</td><td>0.5863</td><td>0.3344</td><td>0.2548</td><td>0.2067</td></tr></tbody></table>

Table 2. $k = {10},c = 2,r = {0.01}$

表2. $k = {10},c = 2,r = {0.01}$

<table><tr><td>$l$</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>${p}_{1}$</td><td>0.9043</td><td>0.9909</td><td>0.9991</td><td>0.9999</td></tr><tr><td>${p}_{2}$</td><td>0.8171</td><td>0.9665</td><td>0.9939</td><td>0.9989</td></tr><tr><td>$\rho$</td><td>0.4975</td><td>0.2699</td><td>0.1424</td><td>0.0746</td></tr><tr><td>Recall</td><td>0.9496</td><td>0.9949</td><td>0.9968</td><td>0.9999</td></tr></table>

<table><tbody><tr><td>$l$</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>${p}_{1}$</td><td>0.9043</td><td>0.9909</td><td>0.9991</td><td>0.9999</td></tr><tr><td>${p}_{2}$</td><td>0.8171</td><td>0.9665</td><td>0.9939</td><td>0.9989</td></tr><tr><td>$\rho$</td><td>0.4975</td><td>0.2699</td><td>0.1424</td><td>0.0746</td></tr><tr><td>召回率（Recall）</td><td>0.9496</td><td>0.9949</td><td>0.9968</td><td>0.9999</td></tr></tbody></table>

<!-- figureText: 250 IN $= {15},{000}$ Full join LSH-Spark Join LSH-Sort Join 30000 40000 50000 threshold 200 Time(s) 150 100 10000 20000 -->

<img src="https://cdn.noedgeai.com/0195ccc8-6e83-7218-be51-018169e50069_33.jpg?x=469&y=566&w=621&h=476&r=0"/>

Fig. 18. Running time with different thresholds.

图18. 不同阈值下的运行时间。

<!-- Media -->

Figure 18 shows the running times of the three algorithms over different similarity thresholds. The overall trend is similar to that on the zipf distribution (Figure 6), which is as expected, as the COREL dataset has an uneven distribution. After hashed into buckets by an LSH function, the distribution of the buckets is also highly skewed.

图18展示了三种算法在不同相似度阈值下的运行时间。总体趋势与齐普夫分布（图6）上的趋势相似，这在预料之中，因为COREL数据集的分布不均匀。通过局部敏感哈希（LSH）函数哈希到桶中后，桶的分布也高度偏斜。

A small difference between Figures 18 and 6 is that when $r$ is very small,the running times of all algorithms also increase. This in fact is due to the length of the join keys being large for small $r$ ,which makes the tuples larger,although the amount of work in terms of tuple count should still decrease.

图18和图6的一个小差异在于，当$r$非常小时，所有算法的运行时间也会增加。实际上，这是因为对于较小的$r$，连接键的长度较大，这使得元组更大，尽管从元组数量来看工作量应该仍然会减少。

## 10 CONCLUDING REMARKS

## 10 结论

In this article, we have studied various similarity joins in the MPC model. The main difference between this and prior work is that we consider OUT, the output size of the join, as an additional parameter in characterizing the complexity of the algorithms. We first proposed a deterministic equi-join algorithm that improves the hypercube algorithm by a polylogarithmic factor and removes the assumption on knowing data statistics. Then we designed output-optimal algorithms for similarity joins under ${\ell }_{1}/{\ell }_{2}/{\ell }_{\infty }$ distances in constant dimensions. We also gave an approximation algorithm based on LSH for the high-dimensional case. Finally, we gave practical versions of some of these algorithms and performed an experimental evaluation.

在本文中，我们研究了大规模并行计算（MPC）模型中的各种相似度连接。本文与先前工作的主要区别在于，我们将连接的输出大小OUT作为表征算法复杂度的一个额外参数。我们首先提出了一种确定性等值连接算法，该算法将超立方体算法改进了一个多对数因子，并消除了对已知数据统计信息的假设。然后，我们为恒定维度下的${\ell }_{1}/{\ell }_{2}/{\ell }_{\infty }$距离相似度连接设计了输出最优算法。对于高维情况，我们还给出了一种基于局部敏感哈希（LSH）的近似算法。最后，我们给出了其中一些算法的实用版本并进行了实验评估。

This article is mainly concerned with 2-relation joins, but we also made an initial step towards multi-way joins by presenting a lower bound regarding output-optimality for the 3-relation chain join. However, it remains an open problem to more precisely characterize the class of multi-way joins that can be solved with output-optimal MPC algorithms.

本文主要关注二元关系连接，但我们也朝着多路连接迈出了第一步，给出了三元关系链连接的输出最优性下界。然而，更精确地表征可以用输出最优的MPC算法解决的多路连接类仍然是一个开放问题。

More broadly, using OUT as a parameter to measure the complexity falls under the realm of parameterized complexity or beyond-worst-case analysis in general. This type of analyses often yields more insights for problems where worst-case scenarios are rare in practice, such as joins. Although OUT is considered the most natural additional parameter to introduce, other possibilities exist, such as assuming that the data follows certain parameterized distributions, or the degree (i.e., maximum number of tuples a tuple can join) is bounded [10, 21], and so forth.

更广泛地说，使用OUT作为参数来衡量复杂度属于参数化复杂度或一般的超越最坏情况分析的范畴。对于实际中最坏情况很少出现的问题（如连接问题），这种类型的分析通常能提供更多的见解。尽管OUT被认为是最自然的额外参数，但也存在其他可能性，例如假设数据遵循某些参数化分布，或者度（即一个元组可以连接的最大元组数量）是有界的[10, 21]等等。

## REFERENCES

## 参考文献

[1] F. Afrati, M. Joglekar, C. Ré, S. Salihoglu, and J. D. Ullman. 2017. GYM: A multiround join algorithm in MapReduce. In Proceedings of the International Conference on Database Theory.

[2] F. N. Afrati and J. D. Ullman. 2011. Optimizing multiway joins in a map-reduce environment. IEEE Transactions on Knowledge and Data Engineering 23, 9 (2011), 1282-1298.

[3] P. K. Agarwal, K. Fox, K. Munagala, and A. Nath. 2016. Parallel algorithms for constructing range and nearest-neighbor searching data structures. In Proceedings of the ACM Symposium on Principles of Database Systems.

[4] A. Aggarwal and J. Vitter. 1988. The input/output complexity of sorting and related problems. Communications of the ACM 31, 9 (1988), 1116-1127.

[5] A. Andoni and P. Indyk. 2008. Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. Communications of the ACM 51, 1 (2008), 117.

[6] A. Atserias, M. Grohe, and D. Marx. 2013. Size bounds and query plans for relational joins. SIAM Journal on Computing 42, 4 (2013), 1737-1767.

[7] P. Beame, P. Koutris, and D. Suciu. 2013. Communication steps for parallel query processing. In Proceedings of the ACM Symposium on Principles of Database Systems.

[8] P. Beame, P. Koutris, and D. Suciu. 2014. Skew in parallel query processing. In Proceedings of the ACM Symposium on Principles of Database Systems.

[9] A. Z. Broder, S. C. Glassman, M. S. Manasse, and G. Zweig. 1997. Syntactic clustering of the web. Computer Networks 29, 8-13 (1997), 1157-1166.

[10] Y. Cao, W. Fan, T. Wo, and W. Yu. 2014. Bounded conjunctive queries. In Proceedings of the International Conference on Very Large Data Bases.

[11] T. M. Chan. 2012. Optimal partition trees. Discrete and Computational Geometry 47, 4 (2012), 661-690.

[12] M. Datar, N. Immorlica, P. Indyk, and V. S. Mirrokni. 2004. Locality sensitive hashing scheme based on p-stable distributions. In Proceedings of the Annual Symposium on Computational Geometry.

[13] M. De Berg, M. Van Kreveld, M. Overmars, and O. C. Schwarzkopf. 2000. Computational geometry. In Computational Geometry. Springer, 1-17.

[14] J. Dean and S. Ghemawat. 2004. MapReduce: Simplified data processing on large clusters. In Proceedings of the Symposium on Operating Systems Design and Implementation.

[15] A. Gionis, P. Indyk, and R. Motwani. 1999. Similarity search in high dimensions via hashing. In Proceedings of the International Conference on Very Large Data Bases.

[16] M. T. Goodrich. 1999. Communication-efficient parallel sorting. SIAM Journal on Computing 29, 2 (1999), 416-432.

[17] M. T. Goodrich, N. Sitchinava, and Q. Zhang. 2011. Sorting, searching and simulation in the MapReduce framework. In Proceedings of the International Symposium on Algorithms and Computation.

[18] S. Har-Peled and M. Sharir. 2011. Relative $\left( {\mathrm{p},\varepsilon }\right)$ -approximations in geometry. Discrete and Computational Geometry 45, 3 (2011), 462-496.

[19] X. Hu, Y. Tao, and K. Yi. 2017. Output-optimal parallel algorithms for similarity joins. In Proceedings of the ACM Symposium on Principles of Database Systems.

[20] P. Indyk and R. Motwani. 1998. Approximate nearest neighbors: Towards removing the curse of dimensionality. In Proceedings of the ACM Symposium on Theory of Computing.

[21] M. Joglekar and C. Ré. 2016. It's all a matter of degree: Using degree information to optimize multiway joins. In Proceedings of the International Conference on Database Theory.

[22] B. Ketsman and D. Suciu. 2017. A worst-case optimal multi-round algorithm for parallel computation of conjunctive queries. In Proceedings of the ACM Symposium on Principles of Database Systems.

[23] P. Koutris, P. Beame, and D. Suciu. 2016. Worst-case optimal algorithms for parallel query processing. In Proceedings of the International Conference on Database Theory.

[24] P. Koutris and D. Suciu. 2011. Parallel evaluation of conjunctive queries. In Proceedings of the ACM Symposium on Principles of Database Systems.

[25] Y. Li, P. M. Long, and A. Srinivasan. 2001. Improved bounds on the sample complexity of learning. Journal of Computer and System Sciences 62, 3 (2001), 516-527.

[26] M. Mitzenmacher and E. Upfal. 2005. Probability and Computing: Randomized Algorithms and Probabilistic Analysis. Cambridge University Press.

[27] O. O'Malley. 2008. Terabyte Sort on Apache Hadoop. Technical Report. Yahoo!

[28] R. Pagh, N. Pham, F. Silvestri, and M. Stöckel. 2015. I/O-efficient similarity join. In Proceedings of the European Symposium on Algorithms.

[29] R. Pagh and F. Silvestri. 2014. The input/output complexity of triangle enumeration. In Proceedings of the ACM Symposium on Principles of Database Systems.

[30] M. Pătraşcu. 2011. Unifying the landscape of cell-probe lower bounds. SIAM Journal on Computing 40, 3 (2011), 827- 847.

[31] Y. Tao, W. Lin, and X. Xiao. 2013. Minimal MapReduce algorithms. In Proceedings of the ACM SIGMOD International Conference on Management of Data.

[32] L. G. Valiant. 1990. A bridging model for parallel computation. Communications of the ACM 33, 8 (1990), 103-111.

[33] M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauley, M. J. Franklin, et al. 2012. Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing. In Proceedings of the USENIX Conference on Networked Systems Design and Implementation.