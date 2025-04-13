# Output-optimal Parallel Algorithms for Similarity Joins

# 相似度连接的输出最优并行算法

Xiao Hu Yufei Tao Ke Yi

胡晓 陶玉飞 柯毅

HKUST University of HKUST

香港科技大学（HKUST）

Queensland

昆士兰

## Abstract

## 摘要

Parallel join algorithms have received much attention in recent years, due to the rapid development of massively parallel systems such as MapReduce and Spark. In the database theory community, most efforts have been focused on studying worst-optimal algorithms. However, the worst-case optimality of these join algorithms relies on the hard instances having very large output sizes. In the case of a two-relation join, the hard instance is just a Cartesian product, with an output size that is quadratic in the input size.

近年来，由于MapReduce和Spark等大规模并行系统的快速发展，并行连接算法受到了广泛关注。在数据库理论界，大多数研究都集中在最坏情况最优算法上。然而，这些连接算法的最坏情况最优性依赖于具有非常大输出规模的困难实例。在双关系连接的情况下，困难实例就是笛卡尔积，其输出规模是输入规模的二次方。

In practice, however, the output size is usually much smaller. One recent parallel join algorithm by Beame et al. [8] has achieved output-optimality, i.e., its cost is optimal in terms of both the input size and the output size, but their algorithm only works for a 2-relation equi-join, and has some imperfections. In this paper, we first improve their algorithm to true optimality. Then we design output-optimal algorithms for a large class of similarity joins. Finally, we present a lower bound, which essentially eliminates the possibility of having output-optimal algorithms for any join on more than two relations.

然而，在实际应用中，输出规模通常要小得多。Beame等人[8]最近提出的一种并行连接算法实现了输出最优性，即其成本在输入规模和输出规模方面都是最优的，但他们的算法仅适用于双关系等值连接，并且存在一些缺陷。在本文中，我们首先将他们的算法改进为真正的最优算法。然后，我们为一大类相似度连接设计了输出最优算法。最后，我们给出了一个下界，这实际上排除了为任何多于两个关系的连接设计输出最优算法的可能性。

## Categories and Subject Descriptors

## 类别和主题描述

F.2.2 [Analysis of algorithms and problem complexity]: Nonnumerical Algorithms and Problems

F.2.2 [算法分析与问题复杂度]：非数值算法与问题

## Keywords

## 关键词

Parallel computation, similarity joins, output-sensitive algorithms

并行计算、相似度连接、输出敏感算法

## 1. INTRODUCTION

## 1. 引言

The similarity join problem is perhaps one of the most extensively studied problems in the database and data mining literature. Numerous variants exist, depending on the metric space and the distance function used. Let $\operatorname{dist}\left( {\cdot , \cdot  }\right)$ be a distance function. Given two point sets ${R}_{1}$ and ${R}_{2}$ and a threshold $r \geq  0$ ,the similarity join problem asks to find all pairs of points $x \in  {R}_{1},y \in  {R}_{2}$ ,such that $\operatorname{dist}\left( {x,y}\right)  \leq  r$ . In this paper,we will be mostly interested in the ${\ell }_{1},{\ell }_{2}$ ,and ${\ell }_{\infty }$ distances, although some of our results (the one based on LSH) can be extended to other distance functions as well.

相似度连接问题可能是数据库和数据挖掘文献中研究最广泛的问题之一。根据所使用的度量空间和距离函数，存在许多变体。设$\operatorname{dist}\left( {\cdot , \cdot  }\right)$为距离函数。给定两个点集${R}_{1}$和${R}_{2}$以及一个阈值$r \geq  0$，相似度连接问题要求找出所有满足$\operatorname{dist}\left( {x,y}\right)  \leq  r$的点对$x \in  {R}_{1},y \in  {R}_{2}$。在本文中，我们主要关注${\ell }_{1},{\ell }_{2}$和${\ell }_{\infty }$距离，尽管我们的一些结果（基于LSH的结果）也可以扩展到其他距离函数。

### 1.1 The computation model

### 1.1 计算模型

Driven by the rapid development of massively parallel systems such as MapReduce [14], Spark [29], and many other systems that adopt very similar architectures, there have also been resurrected interests in the theoretical computer science community to study algorithms in such massively parallel models. One popular model that has often been used to study join algorithms in particular, is the massively parallel computation model (MPC) [1, 2, 3, 7, 8, 20, 21, 22].

受MapReduce [14]、Spark [29]等大规模并行系统以及许多采用非常相似架构的其他系统的快速发展的推动，理论计算机科学界也重新燃起了在这种大规模并行模型中研究算法的兴趣。一个特别常用于研究连接算法的流行模型是大规模并行计算模型（MPC）[1, 2, 3, 7, 8, 20, 21, 22]。

In the MPC model, data is initially partitioned arbitrarily across $p$ servers that are connected by a complete network. Computation proceeds in rounds. In each round, each server first receives messages from other servers (sent in a previous round, if there is one), does some local computation, and then sends messages to other servers, which will be received by them at the beginning of the next round. The complexity of the algorithm is measured first by the number of rounds, then the load,denoted $L$ ,which is the maximum message size received by any server in any round. Initial efforts were mostly spent on understanding what can be done in a single round of computation $\left\lbrack  {2,7,8,{21},{22}}\right\rbrack$ ,but recently,more interests have been given to multi-round (but still a constant) algorithms $\left\lbrack  {1,3,{20},{21}}\right\rbrack$ ,since new main memory based systems, such as Spark, tend to have much lower overhead per round than previous systems like Hadoop. Meanwhile, this puts more emphasis on minimizing the load, to ensure that the local memory at each server is never exceeded.

在MPC模型中，数据最初被任意划分到通过完全网络连接的$p$台服务器上。计算按轮次进行。在每一轮中，每台服务器首先接收来自其他服务器的消息（如果有上一轮发送的消息），进行一些本地计算，然后向其他服务器发送消息，这些消息将在下一轮开始时被其他服务器接收。算法的复杂度首先由轮数衡量，然后是负载，用$L$表示，即任何服务器在任何一轮中接收到的最大消息大小。最初的研究主要集中在理解在一轮计算中可以完成什么$\left\lbrack  {2,7,8,{21},{22}}\right\rbrack$，但最近，人们对多轮（但仍然是常数轮）算法更感兴趣$\left\lbrack  {1,3,{20},{21}}\right\rbrack$，因为像Spark这样的基于新主内存的系统每一轮的开销往往比Hadoop等以前的系统低得多。同时，这更强调了最小化负载，以确保每台服务器的本地内存不会被超出。

One thing that we want to point out, which was never explicitly stated in the prior work on the MPC model, is that the MPC model is essentially the same as Valiant's bulk synchronous processing model (BSP) [28]. More precisely, it is the same as the CREW version of the BSP [15], where a server may broadcast a message to multiple servers. The incoming message size at each server is still limited in the CREW BSP model, as in the MPC model.

我们想要指出的一点是，在之前关于MPC模型（多方计算模型）的研究中从未明确提及，MPC模型本质上与瓦利安特（Valiant）的批量同步处理模型（BSP）[28]相同。更准确地说，它与BSP的CREW（并发读互斥写）版本[15]相同，在该版本中，一台服务器可以向多台服务器广播消息。与MPC模型一样，在CREW BSP模型中，每台服务器接收的消息大小仍然受限。

The MPC model, as well as the CREW BSP model, allows broadcasts. This is often justified by the fact that in some systems, broadcasts may indeed be more efficient than sending the message to each destination individually. Very recently, it has been shown [18] that any algorithm in the CREW BSP model can be simulated in the standard BSP model without using any broadcasts by just increasing the number of rounds and the load by a constant factor, provided that $\mathrm{{IN}} > {p}^{1 + \epsilon }$ ,where $\mathrm{{IN}}$ is the input size and $\epsilon  > 0$ is any small constant. This provides a stronger justification of using broadcasts.

MPC模型以及CREW BSP模型都允许进行广播。这通常是因为在某些系统中，广播确实可能比单独向每个目的地发送消息更高效。最近有研究[18]表明，只要满足$\mathrm{{IN}} > {p}^{1 + \epsilon }$（其中$\mathrm{{IN}}$是输入大小，$\epsilon  > 0$是任意小的常数），CREW BSP模型中的任何算法都可以在标准BSP模型中进行模拟，且无需使用任何广播，只需将轮数和负载增加一个常数因子。这为使用广播提供了更有力的依据。

---

<!-- Footnote -->

*Xiao Hu and Ke Yi are supported by HKRGC under grants GRF-16211614 and GRF-16200415.

*胡晓和易可获得了香港研究资助局（HKRGC）GRF - 16211614和GRF - 16200415项目的资助。

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和首页的完整引用信息。必须尊重本作品中除美国计算机协会（ACM）之外其他所有者的版权。允许进行带引用的摘要。否则，如需复制、重新发布、上传到服务器或分发给列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。

PODS'17, May 14-19, 2017, Chicago, IL, USA

2017年5月14 - 19日，美国伊利诺伊州芝加哥，第38届ACM SIGMOD - SIGACT - SIGAI数据库系统原理研讨会（PODS'17）

<!-- Footnote -->

---

### 1.2 Previous join algorithms in the MPC model

### 1.2 MPC模型中的先前连接算法

All prior work on join algorithms in the MPC model has focused on equi-joins, and has mostly been concerned with the worst case. Notably, the hypercube algorithm [2] computes the equi-join between two relations of size ${N}_{1}$ and ${N}_{2}$ , respectively,with load $L = \widetilde{O}{\left( \sqrt{{N}_{1}{N}_{2}/p}\right) }^{1}$ . Note that this is optimal in the worst case, as the output size can be as large as ${N}_{1}{N}_{2}$ ,when all tuples share the same join key and the join degenerates into a Cartesian product. Since each server can only produce $O\left( {L}^{2}\right)$ join results in a round ${}^{2}$ if the load is limited to $L$ ,all the $p$ servers can produce at most $O\left( {p{L}^{2}}\right)$ join results in a constant number of rounds. Thus, producing ${N}_{1}{N}_{2}$ results needs at least a load of $L = \Omega \left( \sqrt{{N}_{1}{N}_{2}/p}\right)$ . Note that this lower bound argument is assuming tuple-based join algorithms, i.e., the tuples are atomic elements that must be processed and communicated in their entirety. They can be copied but cannot be broken up or manipulated with bit tricks. To produce a join result, all tuples (or their copies) that make up the join result must reside at the same server when the join result is output. However, the server does not have to do any further processing with the result, such as sending it to another server. The same model has also been used in $\left\lbrack  {7,8,{21}}\right\rbrack$ .

之前所有关于MPC模型中连接算法的研究都集中在等值连接上，并且大多关注最坏情况。值得注意的是，超立方体算法[2]分别计算大小为${N}_{1}$和${N}_{2}$的两个关系之间的等值连接，负载为$L = \widetilde{O}{\left( \sqrt{{N}_{1}{N}_{2}/p}\right) }^{1}$。请注意，在最坏情况下这是最优的，因为当所有元组共享相同的连接键且连接退化为笛卡尔积时，输出大小可能高达${N}_{1}{N}_{2}$。由于如果负载限制为$L$，每台服务器在一轮${}^{2}$中最多只能产生$O\left( {L}^{2}\right)$个连接结果，因此所有$p$台服务器在常数轮数内最多只能产生$O\left( {p{L}^{2}}\right)$个连接结果。因此，产生${N}_{1}{N}_{2}$个结果至少需要$L = \Omega \left( \sqrt{{N}_{1}{N}_{2}/p}\right)$的负载。请注意，这个下界论证假设的是基于元组的连接算法，即元组是必须作为整体进行处理和通信的原子元素。它们可以被复制，但不能被拆分或使用位操作技巧进行处理。要产生一个连接结果，构成该连接结果的所有元组（或其副本）在输出连接结果时必须位于同一台服务器上。然而，服务器不必对结果进行任何进一步的处理，例如将其发送到另一台服务器。$\left\lbrack  {7,8,{21}}\right\rbrack$中也使用了相同的模型。

However, on most realistic data sets, the join size is nowhere near the worst case. Suppose the join size is OUT. Applying the same argument as above, one would hope to get a load of $\widetilde{O}\left( \sqrt{\mathrm{{OUT}}/p}\right)$ . Such a bound would be output-optimal. Of course, this is not entirely possible, as OUT can even be zero, so a more reasonable target would be $L = \widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ ,where $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ is the total input size. This is exactly the goal of this work, although in some cases, we have not achieved this ideal input-dependent term $\widetilde{O}\left( {\mathrm{{IN}}/p}\right)$ exactly. Note that we are still doing worst-case analysis, i.e., we do not make any assumptions on the input data and how it is distributed on the $p$ servers initially. We merely use OUT as an additional parameter to measure the complexity of the algorithm.

然而，在大多数实际数据集上，连接大小远未达到最坏情况。假设连接大小为 OUT。运用与上述相同的论证，我们希望负载为 $\widetilde{O}\left( \sqrt{\mathrm{{OUT}}/p}\right)$。这样的界限将是输出最优的。当然，这并非完全可行，因为 OUT 甚至可能为零，所以更合理的目标是 $L = \widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$，其中 $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ 是总输入大小。这正是本文的目标，尽管在某些情况下，我们并未精确实现这个理想的依赖输入的项 $\widetilde{O}\left( {\mathrm{{IN}}/p}\right)$。请注意，我们仍然在进行最坏情况分析，即我们不对输入数据以及它最初在 $p$ 台服务器上的分布情况做任何假设。我们仅仅将 OUT 作为一个额外的参数来衡量算法的复杂度。

There are some previous join algorithms that use both IN and OUT to measure the complexity. Afrati et al. [1] gave an algorithm with load $O\left( {{\mathrm{{IN}}}^{w}/\sqrt{p} + \mathrm{{OUT}}/\sqrt{p}}\right)$ ,where $w$ is the width of the join query,which is 1 for any acyclic query, including a two-relation join. However, both terms $O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$ or $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ are far from optimal.

之前有一些连接算法同时使用 IN 和 OUT 来衡量复杂度。Afrati 等人 [1] 提出了一种负载为 $O\left( {{\mathrm{{IN}}}^{w}/\sqrt{p} + \mathrm{{OUT}}/\sqrt{p}}\right)$ 的算法，其中 $w$ 是连接查询的宽度，对于任何无环查询（包括两关系连接），该宽度为 1。然而，$O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$ 或 $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ 这两项都远非最优。

Beame et al. [8] classified the join values into being heavy and light. For a join value $v$ ,let ${R}_{i}\left( v\right)$ be the set of tuples in ${R}_{i}$ with join value $v$ . Then a join value $v$ is heavy if $\left| {{R}_{1}\left( v\right) }\right|  \geq  {N}_{1}/p$ or $\left| {{R}_{2}\left( v\right) }\right|  \geq  {N}_{2}/p$ . Then they gave an algorithm with load

Beame 等人 [8] 将连接值分为重值和轻值。对于一个连接值 $v$，设 ${R}_{i}\left( v\right)$ 是 ${R}_{i}$ 中连接值为 $v$ 的元组集合。那么，如果 $\left| {{R}_{1}\left( v\right) }\right|  \geq  {N}_{1}/p$ 或 $\left| {{R}_{2}\left( v\right) }\right|  \geq  {N}_{2}/p$，则连接值 $v$ 为重值。然后他们提出了一种负载为

$$
\widetilde{\Theta }\left( {\sqrt{\frac{\mathop{\sum }\limits_{{\text{heavy }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right| }{p}} + \frac{\mathrm{{IN}}}{p}}\right) . \tag{1}
$$

In fact,this bound can be equivalently written as $\widetilde{\Theta }(\sqrt{\mathrm{{OUT}}/p} +$ IN/p). Note that

实际上，这个界限可以等价地写为 $\widetilde{\Theta }(\sqrt{\mathrm{{OUT}}/p} +$ IN/p)。注意

$$
\mathrm{{OUT}} = \mathop{\sum }\limits_{v}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right| 
$$

$$
 = \mathop{\sum }\limits_{{\text{heavy }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right|  + \mathop{\sum }\limits_{{\text{light }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right| ,
$$

so (1) is upper bounded by $\widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ . Meanwhile, it is also lower bounded by $\widetilde{\Omega }\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ : First,it is clearly in $\widetilde{\Omega }\left( {\mathrm{{IN}}/p}\right)$ . Second,it is also in $\widetilde{\Omega }\left( \sqrt{\mathrm{{OUT}}/p}\right)$ since

因此，(1) 的上界为 $\widetilde{O}\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$。同时，它的下界为 $\widetilde{\Omega }\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$：首先，它显然在 $\widetilde{\Omega }\left( {\mathrm{{IN}}/p}\right)$ 中。其次，它也在 $\widetilde{\Omega }\left( \sqrt{\mathrm{{OUT}}/p}\right)$ 中，因为

$$
\mathop{\sum }\limits_{{\text{light }v}}\left| {{R}_{1}\left( v\right) }\right|  \cdot  \left| {{R}_{2}\left( v\right) }\right|  \leq  \frac{{N}_{1}{N}_{2}}{p} \leq  \frac{{\mathrm{{IN}}}^{2}}{p},
$$

hence

因此

$$
\left( 1\right)  = \widetilde{\Omega }\left( {\sqrt{\frac{\mathrm{{OUT}} - {\mathrm{{IN}}}^{2}/p}{p}} + \frac{\mathrm{{IN}}}{p}}\right) 
$$

$$
 = \widetilde{\Omega }\left( \sqrt{\frac{\mathrm{{OUT}} - {\mathrm{{IN}}}^{2}/p}{p} + \frac{{\mathrm{{IN}}}^{2}}{{p}^{2}}}\right)  = \widetilde{\Omega }\left( \sqrt{\mathrm{{OUT}}/p}\right) .
$$

Therefore, their algorithm is output-optimal, but up to a polylog factor, due to the use of hashing (the hidden polylog factor is $O\left( {{\log }^{2}p}\right)$ ). Their analysis relies on the uniform hashing assumption, i.e., the hash function distributes each distinct key to the servers uniformly and independently. It is not clear whether more realistic hash functions, such as universal hashing, could still work. They also assume that each server knows the entire set of heavy join values and their frequencies,namely,all the $\left| {{R}_{i}\left( v\right) }\right|$ ’s that are larger than ${N}_{i}/p$ ,for $i = 1,2$ .

因此，他们的算法是输出最优的，但存在一个多对数因子，这是由于使用了哈希（隐藏的多对数因子为 $O\left( {{\log }^{2}p}\right)$）。他们的分析依赖于均匀哈希假设，即哈希函数将每个不同的键均匀且独立地分配到服务器上。尚不清楚更现实的哈希函数（如通用哈希）是否仍然有效。他们还假设每个服务器都知道所有重连接值及其频率，即所有大于 ${N}_{i}/p$ 的 $\left| {{R}_{i}\left( v\right) }\right|$，其中 $i = 1,2$。

Note that equi-join is a special case of similarity joins with $r = 0$ . There are previously no algorithms in the MPC model for similarity joins with $r > 0$ ,except computing the full Cartesian product of the two relations with load $O\left( \sqrt{{N}_{1}{N}_{2}/p}\right)$ ,which is not output-optimal.

请注意，等值连接是相似度连接在 $r = 0$ 时的特殊情况。之前在 MPC 模型中，对于 $r > 0$ 的相似度连接没有算法，除了计算两个关系的全笛卡尔积，其负载为 $O\left( \sqrt{{N}_{1}{N}_{2}/p}\right)$，这并非输出最优。

As a remark, there exists a general reduction [21] that converts MPC join algorithms into I/O-efficient counterparts under the enumerate version [26] of the external memory model [4], where each result tuple only needs to be seen in memory, as opposed to being reported in the disk. A nice application of the reduction has been demonstrated for the triangle enumeration problem, where an MPC algorithm [21] is shown to imply an EM algorithm matching the I/O lower bound of [26] up to a logarithmic factor.

作为一个备注，存在一种通用的归约方法 [21]，它可以在外部内存模型 [4] 的枚举版本 [26] 下将多方计算（MPC）连接算法转换为高效的输入/输出（I/O）对应算法，在该版本中，每个结果元组只需在内存中处理，而无需在磁盘上报告。这种归约方法在三角形枚举问题上有一个很好的应用，其中一个多方计算算法 [21] 被证明可以推导出一个外部内存（EM）算法，该算法在对数因子范围内与 [26] 的 I/O 下界相匹配。

### 1.3 Our results

### 1.3 我们的结果

We start with an improved algorithm for computing the equi-join between two relations, i.e., a degenerated similarity join with $r = 0$ . We improve upon the algorithm of Beame et al. [8] in the following aspects: (1) Our algorithm does not assume any prior statistical information about the data, such as the heavy join values and their frequencies. (2) The load of our algorithm is exactly $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ tuples, without any extra logarithmic factors. (3) Our algorithm is deterministic. The only price we pay is that the number of rounds increases from 1 to $O\left( 1\right)$ . This algorithm is described in Section 3.

我们从一个用于计算两个关系之间等值连接的改进算法开始，即一个退化的相似度连接，其中 $r = 0$ 。我们在以下方面对 Beame 等人 [8] 的算法进行了改进：（1）我们的算法不假设关于数据的任何先验统计信息，例如频繁连接值及其频率。（2）我们算法的负载恰好是 $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ 个元组，没有任何额外的对数因子。（3）我们的算法是确定性的。我们付出的唯一代价是轮数从 1 增加到 $O\left( 1\right)$ 。该算法将在第 3 节中描述。

While the $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ term is optimal by the tuple-based argument above, prior work did not show why the input-dependent term $O\left( {\mathrm{{IN}}/p}\right)$ is necessary. In fact,the load has often been written in form of $O\left( {\mathrm{{IN}}/{p}^{1 - \delta }}\right)$ for $\delta  \in  \left\lbrack  {0,1}\right\rbrack$ , implicitly assuming that $O\left( {\mathrm{{IN}}/p}\right)$ is the best load possible, i.e., every tuple has to be communicated at least once. Indeed, if OUT is not a parameter, the worst-case input is always when the output size is maximized, i.e., a full Cartesian product for two-relation joins, or the AGM bound [6] for multi-way joins. In this case, the simple tuple-based argument above already leads to a lower bound higher than $\Omega \left( {\mathrm{{IN}}/p}\right)$ ,so this is not an issue. However,when the output size is restrained to be a parameter OUT, these worst-case constructions do not work anymore; and it is not clear why $O\left( {\mathrm{{IN}}/p}\right)$ load is necessary. Indeed,if OUT $= 1$ ,then the tuple-based argument above yields a meaningless lower bound of $\Omega \left( {1/p}\right)$ . To complete the picture,we provide a lower bound showing that even if OUT $= O\left( 1\right)$ ,computing the equi-join between two relations requires $\Omega \left( {\mathrm{{IN}}/p}\right)$ load, by resorting to strong results from communication complexity.

虽然根据上述基于元组的论证，$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ 项是最优的，但先前的工作并未说明为什么依赖输入的项 $O\left( {\mathrm{{IN}}/p}\right)$ 是必要的。事实上，负载通常以 $O\left( {\mathrm{{IN}}/{p}^{1 - \delta }}\right)$ 的形式表示，其中 $\delta  \in  \left\lbrack  {0,1}\right\rbrack$ ，这隐含地假设 $O\left( {\mathrm{{IN}}/p}\right)$ 是可能的最佳负载，即每个元组至少需要通信一次。确实，如果输出大小 OUT 不是一个参数，最坏情况的输入总是输出大小最大的时候，即对于二元关系连接是全笛卡尔积，对于多路连接是 AGM 界 [6] 。在这种情况下，上述简单的基于元组的论证已经得出一个高于 $\Omega \left( {\mathrm{{IN}}/p}\right)$ 的下界，所以这不是问题。然而，当输出大小被限制为一个参数 OUT 时，这些最坏情况的构造就不再适用了；而且不清楚为什么 $O\left( {\mathrm{{IN}}/p}\right)$ 的负载是必要的。实际上，如果 OUT $= 1$ ，那么上述基于元组的论证会得出一个无意义的下界 $\Omega \left( {1/p}\right)$ 。为了完善这一情况，我们通过借助通信复杂度的强大结果，给出了一个下界，表明即使 OUT $= O\left( 1\right)$ ，计算两个关系之间的等值连接也需要 $\Omega \left( {\mathrm{{IN}}/p}\right)$ 的负载。

---

<!-- Footnote -->

${}^{1}$ The $\widetilde{O}$ notation suppresses polylogarithmic factors.

${}^{1}$ $\widetilde{O}$ 符号忽略了多项式对数因子。

${}^{2}$ Technically,this is true under the condition $L = \Omega \left( \frac{{N}_{1} + {N}_{2}}{p}\right)$ , but as will be proved in this paper, the condition indeed holds even just to decide whether the join result is empty.

${}^{2}$ 从技术上讲，这在条件 $L = \Omega \left( \frac{{N}_{1} + {N}_{2}}{p}\right)$ 下成立，但正如本文将证明的，即使只是为了判断连接结果是否为空，该条件实际上也成立。

<!-- Footnote -->

---

The main results in this paper, however, are on similarity joins with $r > 0$ . In this regard,We achieve the following results under various distance functions.

然而，本文的主要结果是关于 $r > 0$ 的相似度连接。在这方面，我们在各种距离函数下取得了以下结果。

1. For ${\ell }_{1}/{\ell }_{\infty }$ distance in constant dimensions,we give a deterministic algorithm with load

1. 对于恒定维度下的 ${\ell }_{1}/{\ell }_{\infty }$ 距离，我们给出一个负载为的确定性算法

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p} \cdot  {\log }^{O\left( 1\right) }p}\right) ,
$$

i.e., the output-dependent term is optimal, while the input-dependent term is away from optimal by a poly-logarithmic factor, which depends on the dimensionality.

即依赖输出的项是最优的，而依赖输入的项与最优值相差一个多项式对数因子，该因子取决于维度。

2. For ${\ell }_{2}$ distance in $d$ dimensions,we give a randomized algorithm with load

2. 对于 $d$ 维度下的 ${\ell }_{2}$ 距离，我们给出一个负载为的随机算法

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}} + {p}^{\frac{d}{{2d} - 1}}\log p}\right) .
$$

Again,the term $O\left( \sqrt{\frac{\text{ OUT }}{p}}\right)$ is output-optimal. The input-dependent term $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ is worse than the ${\ell }_{1}/{\ell }_{\infty }$ case,due to the non-orthogonal nature of the ${\ell }_{2}$ metric,but it is always better than $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ ,which is the load for computing the full Cartesian product.

同样，项 $O\left( \sqrt{\frac{\text{ OUT }}{p}}\right)$ 是输出最优的。依赖于输入的项 $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ 比 ${\ell }_{1}/{\ell }_{\infty }$ 的情况更差，这是由于 ${\ell }_{2}$ 度量的非正交性质所致，但它总是优于 $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$，$O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ 是计算完整笛卡尔积的负载。

3. In high dimensions, we provide an LSH-based algorithm with load

3. 在高维情况下，我们提供一种基于局部敏感哈希（LSH）的算法，其负载为

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}} + \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}} + \frac{\mathrm{{IN}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right) ,
$$

where $\operatorname{OUT}\left( {cr}\right)$ is the output size if the distance threshold is enlarged to ${cr}$ for some constant $c > 1$ ,and $0 < \rho  < 1$ is the quality measure of the hash function used,which depends only on $c$ and the distance function. Similarly,the term $O\left( {\mathrm{{IN}}/{p}^{1/\left( {1 + \rho }\right) }}\right)$ is always better than that for computing the Cartesian product, although output-optimality here is only with respect to $\operatorname{OUT}\left( {cr}\right)$ instead of OUT,due to the approximation nature of LSH.

其中，如果距离阈值对于某个常数 $c > 1$ 扩大到 ${cr}$，则 $\operatorname{OUT}\left( {cr}\right)$ 是输出大小，并且 $0 < \rho  < 1$ 是所使用的哈希函数的质量度量，它仅取决于 $c$ 和距离函数。类似地，项 $O\left( {\mathrm{{IN}}/{p}^{1/\left( {1 + \rho }\right) }}\right)$ 总是优于计算笛卡尔积的情况，尽管由于局部敏感哈希（LSH）的近似性质，这里的输出最优性仅相对于 $\operatorname{OUT}\left( {cr}\right)$ 而非输出（OUT）。

All the algorithms run in $O\left( 1\right)$ rounds,under the mild assumption $\operatorname{IN} > {p}^{1 + \epsilon }$ ,where $\epsilon  > 0$ is any small constant. Note that the randomized output-optimal algorithm in [8] for equi-joins has an implicit assumption that $\operatorname{IN} \geq  {p}^{2}$ ,since there are $\Theta \left( p\right)$ heavy join values,so each server has load at least $\Omega \left( p\right)$ to store these values and their frequencies. We acknowledge that in practice, $\operatorname{IN} \geq  {p}^{2}$ is a very reasonable assumption. Our desire to relax this to $\mathrm{{IN}} > {p}^{1 + \epsilon }$ is more from a theoretical point of view, namely, achieving the minimum requirement for solving these problem in $O\left( 1\right)$ rounds and optimal load. Indeed, Goodrich [15] has shown that, if IN $= {p}^{1 + o\left( 1\right) }$ ,then even computing the "or" of IN bits requires $\omega \left( 1\right)$ rounds under load $O\left( {\mathrm{{IN}}/p}\right)$ .

在温和假设 $\operatorname{IN} > {p}^{1 + \epsilon }$ 下，所有算法都在 $O\left( 1\right)$ 轮内运行，其中 $\epsilon  > 0$ 是任意小的常数。请注意，文献 [8] 中用于等值连接的随机输出最优算法有一个隐含假设，即 $\operatorname{IN} \geq  {p}^{2}$，因为有 $\Theta \left( p\right)$ 个重连接值，所以每个服务器至少需要 $\Omega \left( p\right)$ 的负载来存储这些值及其频率。我们承认，在实践中，$\operatorname{IN} \geq  {p}^{2}$ 是一个非常合理的假设。我们希望将此假设放宽到 $\mathrm{{IN}} > {p}^{1 + \epsilon }$ 更多是从理论角度出发，即实现以 $O\left( 1\right)$ 轮和最优负载解决这些问题的最低要求。实际上，古德里奇（Goodrich）[15] 已经表明，如果输入（IN） $= {p}^{1 + o\left( 1\right) }$，那么即使计算输入（IN）位的“或”运算在负载 $O\left( {\mathrm{{IN}}/p}\right)$ 下也需要 $\omega \left( 1\right)$ 轮。

Finally, we turn to multi-way joins. The only known multi-way equi-join algorithm in the MPC model that has a term related to OUT is the algorithm in [1] mentioned in Section 1.2. However,that term is $O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$ ,which is almost quadratically larger than the output-optimal term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ we have achieved above. We show that,unfortunately, such an output-optimal term is not achievable, even for the simplest multi-way equi-join, a 3-relation chain join ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)  \bowtie  {R}_{3}\left( {C,D}\right)$ . More precisely,in Section 7 we show that if any tuple-based algorithm computing this join has a load in the form of

最后，我们转向多路连接。在消息传递计算（MPC）模型中，唯一已知的与输出（OUT）相关的多路等值连接算法是第 1.2 节中提到的文献 [1] 中的算法。然而，该项是 $O\left( {\mathrm{{OUT}}/\sqrt{p}}\right)$，它几乎比我们上面实现的输出最优项 $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ 大二次方。我们遗憾地表明，即使对于最简单的多路等值连接，即三元关系链连接 ${R}_{1}\left( {A,B}\right)  \bowtie  {R}_{2}\left( {B,C}\right)  \bowtie  {R}_{3}\left( {C,D}\right)$，这样的输出最优项也无法实现。更准确地说，在第 7 节中我们表明，如果任何基于元组的算法计算此连接的负载形式为

$$
L = O\left( {\frac{\mathrm{{IN}}}{{p}^{\alpha }} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right) ,
$$

for some constant $\alpha$ ,then we must have $\alpha  \leq  1/2$ ,provided IN ${\log }^{2}\mathrm{{IN}} = \Omega \left( {p}^{3}\right)$ . On the other hand,the algorithm in [21] can already compute any 3-relation chain join with $\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ load. This means that it is meaningless to introduce the output-dependent term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ .

对于某个常数 $\alpha$，那么如果输入（IN） ${\log }^{2}\mathrm{{IN}} = \Omega \left( {p}^{3}\right)$，我们必须有 $\alpha  \leq  1/2$。另一方面，文献 [21] 中的算法已经可以以 $\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ 的负载计算任何三元关系链连接。这意味着引入依赖于输出的项 $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ 是没有意义的。

## 2. PRELIMINARIES

## 2. 预备知识

We first introduce the following primitives in the MPC model. We will assume $\operatorname{IN} > {p}^{1 + \epsilon }$ where $\epsilon  > 0$ is any small constant.

我们首先在多方计算（MPC）模型中引入以下基本概念。我们假设 $\operatorname{IN} > {p}^{1 + \epsilon }$，其中 $\epsilon  > 0$ 是任意小的常数。

### 2.1 Sorting

### 2.1 排序

The sorting problem in the MPC model is defined as follows. Initially,IN elements are distributed arbitrarily on $p$ servers,which are labeled $1,2,\ldots ,p$ . The goal is to redistribute the elements so that each server has $\mathrm{{IN}}/p$ elements in the end,while any element at server $i$ is smaller than or equal to any element at server $j$ ,for any $i < j$ . By realizing that the MPC model is the same as the BSP model, we can directly invoke Goodrich's optimal BSP sorting algorithm [15]. His algorithm has load $L = \Theta \left( {\mathrm{{IN}}/p}\right)$ and runs in $O\left( {{\log }_{L}\mathrm{{IN}}}\right)  = O\left( {{\log }_{L}\left( {pL}\right) }\right)  = O\left( {{\log }_{L}p}\right)$ rounds. When $\mathrm{{IN}} > {p}^{1 + \epsilon }$ ,this is $O\left( 1\right)$ rounds.

多方计算（MPC）模型中的排序问题定义如下。最初，N 个元素任意分布在 $p$ 台服务器上，这些服务器标记为 $1,2,\ldots ,p$。目标是重新分配这些元素，使得最终每台服务器有 $\mathrm{{IN}}/p$ 个元素，并且对于任意 $i < j$，服务器 $i$ 上的任何元素都小于或等于服务器 $j$ 上的任何元素。通过认识到多方计算（MPC）模型与块同步并行（BSP）模型相同，我们可以直接调用古德里奇（Goodrich）的最优块同步并行（BSP）排序算法 [15]。他的算法负载为 $L = \Theta \left( {\mathrm{{IN}}/p}\right)$，并在 $O\left( {{\log }_{L}\mathrm{{IN}}}\right)  = O\left( {{\log }_{L}\left( {pL}\right) }\right)  = O\left( {{\log }_{L}p}\right)$ 轮内运行。当 $\mathrm{{IN}} > {p}^{1 + \epsilon }$ 时，这是 $O\left( 1\right)$ 轮。

### 2.2 Multi-numbering

### 2.2 多重编号

Suppose each tuple has a key. The goal of the multi-numbering problem is, for each key, assign consecutive numbers $1,2,3,\ldots$ to all the tuples with the same key.

假设每个元组都有一个键。多重编号问题的目标是，对于每个键，为所有具有相同键的元组分配连续的编号 $1,2,3,\ldots$。

We solve this problem by reducing it to the all prefix-sums problem: Given an array of elements $A\left\lbrack  1\right\rbrack  ,\ldots ,A\left\lbrack  \mathrm{{IN}}\right\rbrack$ , compute $S\left\lbrack  i\right\rbrack   = A\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  i\right\rbrack$ for all $i = 1,\ldots ,\mathrm{{IN}}$ ,where $\oplus$ is any associative operator. Goodrich et al. [16] gave an algorithm in the BSP model for this problem that uses $O\left( {\mathrm{{IN}}/p}\right)$ load and $O\left( 1\right)$ rounds.

我们通过将此问题归约为全前缀和问题来解决它：给定一个元素数组 $A\left\lbrack  1\right\rbrack  ,\ldots ,A\left\lbrack  \mathrm{{IN}}\right\rbrack$，对所有 $i = 1,\ldots ,\mathrm{{IN}}$ 计算 $S\left\lbrack  i\right\rbrack   = A\left\lbrack  1\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  i\right\rbrack$，其中 $\oplus$ 是任意结合运算符。古德里奇（Goodrich）等人 [16] 给出了一个块同步并行（BSP）模型下解决此问题的算法，该算法使用 $O\left( {\mathrm{{IN}}/p}\right)$ 的负载和 $O\left( 1\right)$ 轮。

To see how the multi-numbering problem reduces to the all prefix-sums problem, we first sort all tuples by their keys; ties are broken arbitrarily. The $i$ -th tuple in the sorted order will produce a pair(x,y),which will act as $A\left\lbrack  i\right\rbrack$ . For each tuple that is the first of its key in the sorted order, we produce the pair(0,1); otherwise,we produce(1,1). Note that we need another round of communication to determine whether each tuple is the first of its key, in case that its predecessor resides on another server.

为了了解多重编号问题如何归约为全前缀和问题，我们首先按元组的键对所有元组进行排序；若键相同则任意打破平局。排序顺序中的第 $i$ 个元组将产生一个对 (x, y)，它将作为 $A\left\lbrack  i\right\rbrack$。对于排序顺序中每个键的第一个元组，我们产生对 (0, 1)；否则，我们产生 (1, 1)。请注意，如果某个元组的前一个元组位于另一台服务器上，我们需要另一轮通信来确定每个元组是否是其键的第一个元组。

Then we define the operator $\oplus$ as

然后我们将运算符 $\oplus$ 定义为

$$
\left( {{x}_{1},{y}_{1}}\right)  \oplus  \left( {{x}_{2},{y}_{2}}\right)  = \left( {{x}_{1}{x}_{2},y}\right) ,
$$

where

其中

$$
y = \left\{  \begin{array}{ll} {y}_{1} + {y}_{2}, & \text{ if }{x}_{2} = 1; \\  {y}_{2}, & \text{ if }{x}_{2} = 0. \end{array}\right. 
$$

Consider any $\left( {x,y}\right)  = A\left\lbrack  i\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  j\right\rbrack$ . Intuitively, $x = 0$ indicates that $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ contain at least one tuple that is the first of its key,while $y$ counts the number of tuples in $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ whose key is the same as that of $A\left\lbrack  j\right\rbrack$ . It is an easy exercise to check that $\oplus$ is associative,and after solving the all prefix-sums problem, $S\left\lbrack  i\right\rbrack$ is exactly the number of tuples in front of the $i$ -th tuple that has the same key (including the $i$ -th tuple itself),which solves the multi-numbering problem as desired.

考虑任意 $\left( {x,y}\right)  = A\left\lbrack  i\right\rbrack   \oplus  \cdots  \oplus  A\left\lbrack  j\right\rbrack$。直观地说，$x = 0$ 表示 $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ 中至少包含一个是其键的第一个元组，而 $y$ 计算 $A\left\lbrack  i\right\rbrack  ,\ldots ,A\left\lbrack  j\right\rbrack$ 中键与 $A\left\lbrack  j\right\rbrack$ 的键相同的元组的数量。很容易验证 $\oplus$ 是可结合的，并且在解决全前缀和问题后，$S\left\lbrack  i\right\rbrack$ 恰好是第 $i$ 个元组前面具有相同键的元组的数量（包括第 $i$ 个元组本身），这就如我们所愿地解决了多重编号问题。

### 2.3 Sum-by-key

### 2.3 按键求和

Suppose each tuple is associated with a key and a weight. The goal of the sum-by-key problem is to compute, for each key, the total weight of all the tuples with the same key.

假设每个元组都与一个键和一个权重相关联。按键求和问题的目标是，对于每个键，计算所有具有相同键的元组的总权重。

This problem can be solved using essentially the same approach as for the multi-numbering problem. First sort all the $N$ tuples by their keys. As above,each tuple will produce a pair(x,y). Now, $x$ still indicates whether this tuple is the first of its key,but we just set $y$ to be the weight associated with the tuple. After we have solved the all prefix-sums problem on these pairs, the last tuple of each key has the total weight for this key. Again, we need another round to identify the last tuple of each key, by checking each tuple's successor.

这个问题本质上可以使用与多编号问题相同的方法来解决。首先，根据键对所有 $N$ 元组进行排序。如上所述，每个元组将产生一个对 (x, y)。现在，$x$ 仍然表示该元组是否是其键对应的第一个元组，但我们只需将 $y$ 设置为与该元组相关联的权重。在我们解决了这些对上的所有前缀和问题之后，每个键对应的最后一个元组将包含该键的总权重。同样，我们需要再进行一轮操作，通过检查每个元组的后继元组来确定每个键对应的最后一个元组。

After the algorithm above finishes, for each key, exactly one tuple knows the total weight for the key, i.e., the last one in the sorted order. In some cases, we also need every tuple to know the total weight for the tuple's own key. To do so, we invoke the multi-numbering algorithm, so that the last tuple of each key also knows the number of tuples with that key. From this number, we can compute exactly the range of servers that hold all the tuples with this key. Then we broadcast the total weight to these servers.

在上述算法完成后，对于每个键，恰好有一个元组知道该键的总权重，即按排序顺序的最后一个元组。在某些情况下，我们还需要每个元组都知道其自身键的总权重。为此，我们调用多编号算法，使得每个键对应的最后一个元组也知道具有该键的元组数量。根据这个数量，我们可以准确计算出持有所有具有该键的元组的服务器范围。然后，我们将总权重广播到这些服务器。

### 2.4 Multi-search

### 2.4 多搜索

The multi-search problem is defined as follows. Given ${N}_{1}$ distinct keys and ${N}_{2}$ queries,where $\mathrm{{IN}} = {N}_{1} + {N}_{2}$ ,for each query, find its predecessor, i.e., the largest key that is no larger than the query. The multi-search algorithm given by Goodrich et al. [16] is randomized, with a small probability exceeding $O\left( {\mathrm{{IN}}/p}\right)$ load. In fact,this problem can also be solved using all prefix-sums, which results in a deterministic algorithm with load $O\left( {\mathrm{{IN}}/p}\right)$ : We first sort all the keys and queries together. Then for each key $k$ ,define its corresponding $A\left\lbrack  i\right\rbrack$ as itself; for each query,define its $A\left\lbrack  i\right\rbrack   =  - \infty$ ; define $\oplus   = \max$ . Then it should be obvious that $S\left\lbrack  i\right\rbrack$ is the predecessor of the corresponding query.

多搜索问题定义如下。给定 ${N}_{1}$ 个不同的键和 ${N}_{2}$ 个查询，其中 $\mathrm{{IN}} = {N}_{1} + {N}_{2}$，对于每个查询，找到其前驱，即不大于该查询的最大键。古德里奇（Goodrich）等人 [16] 提出的多搜索算法是随机化的，有很小的概率会超过 $O\left( {\mathrm{{IN}}/p}\right)$ 的负载。实际上，这个问题也可以使用所有前缀和来解决，从而得到一个负载为 $O\left( {\mathrm{{IN}}/p}\right)$ 的确定性算法：我们首先将所有键和查询一起排序。然后，对于每个键 $k$，将其对应的 $A\left\lbrack  i\right\rbrack$ 定义为其自身；对于每个查询，定义其 $A\left\lbrack  i\right\rbrack   =  - \infty$；定义 $\oplus   = \max$。那么显然，$S\left\lbrack  i\right\rbrack$ 就是相应查询的前驱。

### 2.5 Cartesian product

### 2.5 笛卡尔积

The hypercube algorithm $\left\lbrack  {2,8}\right\rbrack$ is a randomized algorithm that computes the Cartesian product of two sets ${R}_{1}$ and ${R}_{2}$ . Suppose the two sets have size ${N}_{1}$ and ${N}_{2}$ ,respectively. This algorithm has a load of $O\left( {\left( {\sqrt{{N}_{1}{N}_{2}/p} + \mathrm{{IN}}/p}\right) {\log }^{2}p}\right)$ with probability $1 - 1/{p}^{O\left( 1\right) }$ . The extra log factors are due to the use of hashing. We observe that if the elements in each set are numbered as $1,2,3,\ldots$ ,then we can achieve deterministic and perfect load balancing.

超立方体算法 $\left\lbrack  {2,8}\right\rbrack$ 是一种随机算法，用于计算两个集合 ${R}_{1}$ 和 ${R}_{2}$ 的笛卡尔积。假设这两个集合的大小分别为 ${N}_{1}$ 和 ${N}_{2}$。该算法有 $1 - 1/{p}^{O\left( 1\right) }$ 的概率负载为 $O\left( {\left( {\sqrt{{N}_{1}{N}_{2}/p} + \mathrm{{IN}}/p}\right) {\log }^{2}p}\right)$。额外的对数因子是由于使用了哈希。我们观察到，如果每个集合中的元素编号为 $1,2,3,\ldots$，那么我们可以实现确定性的完美负载均衡。

Without loss of generality,assume ${N}_{1} \leq  {N}_{2}$ . As in the standard hypercube algorithm,we arrange the $p$ servers into a ${d}_{1} \times  {d}_{2}$ grid such that ${d}_{1}{d}_{2} = p$ . If an element in ${R}_{1}$ gets assigned a number $x$ ,then we send it to all servers in the ( $x$ ${\;\operatorname{mod}\;{d}_{1}}$ )-th row of the grid; for an element in ${R}_{2}$ ,we send it to all servers in the $\left( {x{\;\operatorname{mod}\;{d}_{2}}}\right)$ -th column of the grid. Each server then produces all pairs of elements received. By setting (1) ${d}_{1} = \sqrt{\frac{p{N}_{1}}{{N}_{2}}},{d}_{2} = \sqrt{\frac{p{N}_{2}}{{N}_{1}}}$ ,if ${N}_{2} \leq  p{N}_{1}$ ; or (2) ${d}_{1} = 1,{d}_{2} = p$ ,if ${N}_{2} > p{N}_{1}$ ,the load is $O\left( {\sqrt{{N}_{1}{N}_{2}/p} + }\right.$ IN/p).

不失一般性，假设 ${N}_{1} \leq  {N}_{2}$ 。与标准超立方体算法一样，我们将 $p$ 个服务器排列成一个 ${d}_{1} \times  {d}_{2}$ 网格，使得 ${d}_{1}{d}_{2} = p$ 。如果 ${R}_{1}$ 中的一个元素被分配了一个编号 $x$ ，那么我们将其发送到网格的第 ( $x$ ${\;\operatorname{mod}\;{d}_{1}}$ ) 行的所有服务器；对于 ${R}_{2}$ 中的一个元素，我们将其发送到网格的第 $\left( {x{\;\operatorname{mod}\;{d}_{2}}}\right)$ 列的所有服务器。然后，每个服务器生成接收到的元素的所有对。通过设置 (1) 若 ${N}_{2} \leq  p{N}_{1}$ ，则 ${d}_{1} = \sqrt{\frac{p{N}_{1}}{{N}_{2}}},{d}_{2} = \sqrt{\frac{p{N}_{2}}{{N}_{1}}}$ ；或者 (2) 若 ${N}_{2} > p{N}_{1}$ ，则 ${d}_{1} = 1,{d}_{2} = p$ ，负载为 $O\left( {\sqrt{{N}_{1}{N}_{2}/p} + }\right.$ IN/p)。

### 2.6 Server allocation

### 2.6 服务器分配

In many of our algorithms, we decompose the problem into up to $p$ subproblems,and allocate the $p$ servers appropriately,with subproblem $j$ having $p\left( j\right)$ servers,where $\mathop{\sum }\limits_{j}p\left( j\right)  \leq  p$ . Thus,each subproblem needs to know which servers have been allocated to it. This is trivial if $\operatorname{IN} \geq  {p}^{2}$ , as we can collect all the $p\left( j\right)$ ’s to one server,do a central allocation, and broadcast the allocation results to all servers, as is done in [8]. When we only have $\mathrm{{IN}} \geq  {p}^{1 + \epsilon }$ ,some more work is needed to ensure $O\left( {\mathrm{{IN}}/p}\right)$ load.

在我们的许多算法中，我们将问题分解为最多 $p$ 个子问题，并适当地分配 $p$ 个服务器，子问题 $j$ 有 $p\left( j\right)$ 个服务器，其中 $\mathop{\sum }\limits_{j}p\left( j\right)  \leq  p$ 。因此，每个子问题需要知道哪些服务器已分配给它。如果 $\operatorname{IN} \geq  {p}^{2}$ ，这很简单，因为我们可以将所有的 $p\left( j\right)$ 收集到一个服务器，进行集中分配，并将分配结果广播到所有服务器，就像文献 [8] 中所做的那样。当我们只有 $\mathrm{{IN}} \geq  {p}^{1 + \epsilon }$ 时，需要做更多的工作来确保 $O\left( {\mathrm{{IN}}/p}\right)$ 负载。

More formally, in the server allocation problem, each tuple has a subproblem id $j$ ,which identifies the subproblem it belongs to (the $j$ ’s do not have to be consecutive),and $p\left( j\right)$ , which is the number of servers allocated to subproblem $j$ . The goal is to attach each tuple a range $\left\lbrack  {{p}_{1}\left( j\right) ,{p}_{2}\left( j\right) }\right\rbrack$ ,such that the ranges of different subproblems are disjoint and $\mathop{\max }\limits_{j}{p}_{2}\left( j\right)  \leq  p$ .

更正式地说，在服务器分配问题中，每个元组都有一个子问题 ID $j$ ，用于标识它所属的子问题（ $j$ 不必是连续的），以及 $p\left( j\right)$ ，即分配给子问题 $j$ 的服务器数量。目标是为每个元组附加一个范围 $\left\lbrack  {{p}_{1}\left( j\right) ,{p}_{2}\left( j\right) }\right\rbrack$ ，使得不同子问题的范围不相交且 $\mathop{\max }\limits_{j}{p}_{2}\left( j\right)  \leq  p$ 。

We again resort to all prefix-sums. First sort all tuples by their subproblem id. For each tuple, define its corresponding $A\left\lbrack  i\right\rbrack   = p\left( j\right)$ if it is the first tuple of subproblem $j$ ,and 0 otherwise. After running all prefix-sums, for each tuple, we set its ${p}_{2}\left( j\right)  = S\left\lbrack  i\right\rbrack$ ,and ${p}_{1}\left( j\right)  = S\left\lbrack  i\right\rbrack   - p\left( j\right)  + 1$ .

我们再次使用全前缀和。首先按子问题 ID 对所有元组进行排序。对于每个元组，如果它是子问题 $j$ 的第一个元组，则定义其对应的 $A\left\lbrack  i\right\rbrack   = p\left( j\right)$ ，否则定义为 0。在运行全前缀和之后，对于每个元组，我们设置其 ${p}_{2}\left( j\right)  = S\left\lbrack  i\right\rbrack$ ，以及 ${p}_{1}\left( j\right)  = S\left\lbrack  i\right\rbrack   - p\left( j\right)  + 1$ 。

## 3. EQUI-JOIN

## 3. 等值连接

We start by revisiting the equi-join problem between 2 relations, ${R}_{1} \boxtimes  {R}_{2}$ . Let ${N}_{1}$ and ${N}_{2}$ be the sizes of ${R}_{1}$ and ${R}_{2}$ ,respectively; set IN $= {N}_{1} + {N}_{2}$ . First,if ${N}_{1} >$ $p{N}_{2}$ or ${N}_{2} > p{N}_{1}$ ,the problem can be trivially solved by broadcasting the smaller relation to all servers, incurring a load of $O\left( {\min \left\{  {{N}_{1},{N}_{2}}\right\}  }\right)$ . Below,we assume ${N}_{1} \leq  {N}_{2} \leq$ $p{N}_{1}$ ,and describe an algorithm that achieves the following result.

我们首先回顾两个关系 ${R}_{1} \boxtimes  {R}_{2}$ 之间的等值连接问题。设 ${N}_{1}$ 和 ${N}_{2}$ 分别为 ${R}_{1}$ 和 ${R}_{2}$ 的大小；令 IN 为 $= {N}_{1} + {N}_{2}$。首先，如果 ${N}_{1} >$ $p{N}_{2}$ 或 ${N}_{2} > p{N}_{1}$，则可以通过将较小的关系广播到所有服务器来轻松解决该问题，负载为 $O\left( {\min \left\{  {{N}_{1},{N}_{2}}\right\}  }\right)$。下面，我们假设 ${N}_{1} \leq  {N}_{2} \leq$ $p{N}_{1}$，并描述一种能实现以下结果的算法。

THEOREM 1. There is a deterministic algorithm that computes the equi-join between 2 relations in $O\left( 1\right)$ rounds with load $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ . It does not assume any prior statistical information about the data.

定理 1. 存在一种确定性算法，它能在 $O\left( 1\right)$ 轮内以负载 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ 计算两个关系之间的等值连接。该算法不假设对数据有任何先验统计信息。

### 3.1 The algorithm

### 3.1 算法

Our algorithm can be seen as an MPC version of sort-merge-join.

我们的算法可以看作是排序 - 合并连接（sort - merge - join）的多方计算（MPC）版本。

## Step (1) Computing OUT

## 步骤 (1) 计算 OUT

Consider each distinct join value $v$ . Let ${R}_{i}\left( v\right)$ be the set of tuples in ${R}_{i}$ with join value $v$ ; let ${N}_{i}\left( v\right)  = \left| {{R}_{i}\left( v\right) }\right|$ . Note that $\mathrm{{OUT}} = \mathop{\sum }\limits_{v}{N}_{1}\left( v\right) {N}_{2}\left( v\right)$ . We first use the sum-by-key algorithm to compute all the ${N}_{i}\left( v\right)$ ’s,i.e.,each tuple in ${R}_{i}\left( v\right)$ is considered to have key $v$ and weight 1 . Recall that after the sum-by-key algorithm,for each $v$ ,exactly one tuple knows ${N}_{i}\left( v\right)$ . We sort all such tuples by the key $v$ . Then we add up all the ${N}_{1}\left( v\right) {N}_{2}\left( v\right)$ ’s,which can also be done by sum-by-key (just that the key is the same for all tuples).

考虑每个不同的连接值 $v$。设 ${R}_{i}\left( v\right)$ 是 ${R}_{i}$ 中连接值为 $v$ 的元组集合；令 ${N}_{i}\left( v\right)  = \left| {{R}_{i}\left( v\right) }\right|$。注意 $\mathrm{{OUT}} = \mathop{\sum }\limits_{v}{N}_{1}\left( v\right) {N}_{2}\left( v\right)$。我们首先使用按键求和算法来计算所有的 ${N}_{i}\left( v\right)$，即，${R}_{i}\left( v\right)$ 中的每个元组被视为键为 $v$ 且权重为 1。回想一下，在按键求和算法之后，对于每个 $v$，恰好有一个元组知道 ${N}_{i}\left( v\right)$。我们按键 $v$ 对所有这样的元组进行排序。然后我们将所有的 ${N}_{1}\left( v\right) {N}_{2}\left( v\right)$ 相加，这也可以通过按键求和来完成（只是所有元组的键相同）。

## Step (2) Computing ${R}_{1} \boxtimes  {R}_{2}$

## 步骤 (2) 计算 ${R}_{1} \boxtimes  {R}_{2}$

Next, we compute the join, i.e., the Cartesian products ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$ for all $v$ . Sort all tuples in both ${R}_{1}$ and ${R}_{2}$ by the join value. Consider each join value $v$ . If all tuples in ${R}_{1}\left( v\right)  \cup  {R}_{2}\left( v\right)$ land on the same server,their join results can be emitted directly, so we only need to deal with the case when they land on 2 or more servers. There are at most $p$ such $v$ ’s. For each such $v$ ,we allocate

接下来，我们计算连接，即，对于所有的 $v$ 计算笛卡尔积 ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$。按连接值对 ${R}_{1}$ 和 ${R}_{2}$ 中的所有元组进行排序。考虑每个连接值 $v$。如果 ${R}_{1}\left( v\right)  \cup  {R}_{2}\left( v\right)$ 中的所有元组都落在同一台服务器上，则可以直接输出它们的连接结果，因此我们只需要处理它们落在两台或更多台服务器上的情况。这样的 $v$ 最多有 $p$ 个。对于每个这样的 $v$，我们分配

$$
{p}_{v} = \left\lceil  {p \cdot  \frac{{N}_{1}\left( v\right) }{{N}_{1}} + p \cdot  \frac{{N}_{2}\left( v\right) }{{N}_{2}} + p \cdot  \frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{\mathrm{{OUT}}}}\right\rceil  
$$

servers and compute the Cartesian product ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$ . Note that we need a total of $O\left( p\right)$ servers; scaling down the initial $p$ can ensure that at most $p$ servers are needed. Here, we also need the server allocation primitive to allocate servers to these subproblems accordingly. Finally, to be able to use the deterministic version of the hypercube algorithm,the elements in each ${R}_{i}\left( v\right)$ need to be assigned consecutive numbers, which can be achieved by running the multi-numbering algorithm, treating each distinct join value $v$ as a key. It can be easily verified that the load is $O\left( {\sqrt{\frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{{p}_{v}}} + \frac{{N}_{1}\left( v\right) }{{p}_{v}} + \frac{{N}_{2}\left( v\right) }{{p}_{v}}}\right)  = O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{{N}_{1}}{p} + \frac{{N}_{2}}{p}}\right) .$

服务器并计算笛卡尔积 ${R}_{1}\left( v\right)  \times  {R}_{2}\left( v\right)$。请注意，我们总共需要 $O\left( p\right)$ 台服务器；缩小初始的 $p$ 可以确保最多需要 $p$ 台服务器。在这里，我们还需要服务器分配原语来相应地为这些子问题分配服务器。最后，为了能够使用超立方体算法的确定性版本，每个 ${R}_{i}\left( v\right)$ 中的元素需要被分配连续的编号，这可以通过运行多编号算法来实现，将每个不同的连接值 $v$ 视为一个键。可以很容易地验证负载为 $O\left( {\sqrt{\frac{{N}_{1}\left( v\right) {N}_{2}\left( v\right) }{{p}_{v}}} + \frac{{N}_{1}\left( v\right) }{{p}_{v}} + \frac{{N}_{2}\left( v\right) }{{p}_{v}}}\right)  = O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{{N}_{1}}{p} + \frac{{N}_{2}}{p}}\right) .$

### 3.2 A matching lower bound

### 3.2 匹配的下界

As argued in Section 1.2,the term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ is optimal for any tuple-based algorithm. Below we show that the term $O\left( {\mathrm{{IN}}/p}\right)$ is also necessary,even when $\mathrm{{OUT}} = O\left( 1\right)$ .

正如 1.2 节所论述的，对于任何基于元组的算法，项 $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ 是最优的。下面我们将证明，即使当 $\mathrm{{OUT}} = O\left( 1\right)$ 时，项 $O\left( {\mathrm{{IN}}/p}\right)$ 也是必要的。

THEOREM 2. Any randomized algorithm that computes the equi-join between 2 relations in $O\left( 1\right)$ rounds with a success probability more than $3/4$ must incur a load of at least $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2},\frac{\mathrm{{IN}}}{p}}\right\}  }\right)$ bits.

定理 2. 任何在 $O\left( 1\right)$ 轮内以超过 $3/4$ 的成功概率计算两个关系之间等值连接的随机算法，其负载至少为 $\Omega \left( {\min \left\{  {{N}_{1},{N}_{2},\frac{\mathrm{{IN}}}{p}}\right\}  }\right)$ 比特。

Proof. We use a reduction from the lopsided set disjointness problem studied in communication complexity: Alice has $\leq  n$ elements and Bob has $\leq  m$ elements with $m > n$ , both from a universe of size $m$ ,and the goal is to decide whether they have an element in common. It has been proved that in any multi-round communication protocol, either Alice has to send $\Omega \left( n\right)$ bits to Bob,or Bob has to send $\Omega \left( m\right)$ bits to Alice [27]. This holds even for randomized algorithms with a success probability larger than $3/4$ . We also note that in the hard instances used in [27], the intersection size of Alice's and Bob's sets is either 0 or 1 .

证明。我们采用通信复杂度中研究的不平衡集合不相交问题进行归约：爱丽丝有 $\leq  n$ 个元素，鲍勃有 $\leq  m$ 个元素，且 $m > n$，这些元素都来自大小为 $m$ 的全集，目标是判断他们是否有共同元素。已经证明，在任何多轮通信协议中，要么爱丽丝必须向鲍勃发送 $\Omega \left( n\right)$ 比特，要么鲍勃必须向爱丽丝发送 $\Omega \left( m\right)$ 比特 [27]。即使对于成功概率大于 $3/4$ 的随机算法，这一点也成立。我们还注意到，在文献 [27] 中使用的困难实例中，爱丽丝和鲍勃集合的交集大小要么为 0 要么为 1。

The reduction works as follows. Without loss of generality, we assume ${N}_{1} \leq  {N}_{2}$ . Given a hard instance of lopsided set disjointness,we create ${R}_{1}$ with ${N}_{1} = n$ tuples,whose join values are the elements of Alice’s set; create ${R}_{2}$ with ${N}_{2} = m$ tuples,whose join values are the elements of Bob’s set. Then solving the join problem also determines whether the two sets intersect or not, while OUT can only be 1 or 0 .

归约过程如下。不失一般性，我们假设 ${N}_{1} \leq  {N}_{2}$。给定一个不平衡集合不相交的困难实例，我们创建具有 ${N}_{1} = n$ 个元组的 ${R}_{1}$，其连接值是爱丽丝集合中的元素；创建具有 ${N}_{2} = m$ 个元组的 ${R}_{2}$，其连接值是鲍勃集合中的元素。那么解决连接问题也能确定这两个集合是否相交，而输出（OUT）只能是 1 或 0。

Recall that in the MPC model, the adversary can allocate the input arbitrarily. We allocate ${R}_{1}$ and ${R}_{2}$ to the $p$ servers as follows.

回想一下，在多方计算（MPC）模型中，对手可以任意分配输入。我们按如下方式将 ${R}_{1}$ 和 ${R}_{2}$ 分配到 $p$ 台服务器上。

If ${N}_{2} \leq  p \cdot  {N}_{1}$ ,we allocate Alice’s set to $\frac{p{N}_{2}}{\mathrm{{IN}}}$ servers and Bob’s set to $\frac{p{N}_{1}}{\mathrm{{IN}}}$ servers. Then Alice’s servers must send $\Omega \left( {N}_{1}\right)$ bits to Bob’s servers,which incurs a total load (across all rounds) of $\Omega \left( {\mathrm{{IN}}/p}\right)$ bits per server,or Bob’s servers must send $\Omega \left( {N}_{2}\right)$ bits to Alice’s servers,also incurring a total load of $\Omega \left( {\mathrm{{IN}}/p}\right)$ bits per server.

如果 ${N}_{2} \leq  p \cdot  {N}_{1}$ ，我们将爱丽丝（Alice）的集合分配给 $\frac{p{N}_{2}}{\mathrm{{IN}}}$ 台服务器，将鲍勃（Bob）的集合分配给 $\frac{p{N}_{1}}{\mathrm{{IN}}}$ 台服务器。那么爱丽丝的服务器必须向鲍勃的服务器发送 $\Omega \left( {N}_{1}\right)$ 比特的数据，这会导致每台服务器的总负载（在所有轮次中）为 $\Omega \left( {\mathrm{{IN}}/p}\right)$ 比特；或者鲍勃的服务器必须向爱丽丝的服务器发送 $\Omega \left( {N}_{2}\right)$ 比特的数据，同样会导致每台服务器的总负载为 $\Omega \left( {\mathrm{{IN}}/p}\right)$ 比特。

If ${N}_{2} > p \cdot  {N}_{1}$ ,then we allocate Bob’s set to one server,and Alice’s set to the other $p - 1$ servers. Then Alice’s servers will send $\Omega \left( {N}_{1}\right)$ bits to Bob’s server,or receive $\Omega \left( {N}_{2}\right)$ bits, so the load is $\Omega \left( {\min \left( {{N}_{1},{N}_{2}/p}\right) }\right)  = \Omega \left( {N}_{1}\right)$ .

如果 ${N}_{2} > p \cdot  {N}_{1}$ ，那么我们将鲍勃的集合分配给一台服务器，将爱丽丝的集合分配给其他 $p - 1$ 台服务器。那么爱丽丝的服务器将向鲍勃的服务器发送 $\Omega \left( {N}_{1}\right)$ 比特的数据，或者接收 $\Omega \left( {N}_{2}\right)$ 比特的数据，因此负载为 $\Omega \left( {\min \left( {{N}_{1},{N}_{2}/p}\right) }\right)  = \Omega \left( {N}_{1}\right)$ 。

## 4. SIMILARITY JOIN UNDER ${\ell }_{1}/{\ell }_{\infty }$

## 4. ${\ell }_{1}/{\ell }_{\infty }$ 下的相似连接

In this section,we study similarity joins under the ${\ell }_{1}$ or the ${\ell }_{\infty }$ metric. We will actually study a more general problem, namely the rectangles-containing-points problem. Here we are given a set ${R}_{1}$ of ${N}_{1}$ points and a set ${R}_{2}$ of ${N}_{2}$ orthogonal rectangles. The goal is to return all pairs $\left( {x,y}\right)  \in  {R}_{1} \times  {R}_{2}$ such that $x \in  y$ . Note that a similarity join with ${\ell }_{\infty }$ metric is equivalent to a rectangles-containing-points problem where each side of the rectangles has length ${2r}$ . A similarity join with ${\ell }_{1}$ metric in $d$ dimensions can be reduced to a similarity join with ${\ell }_{\infty }$ metric in ${2}^{d - 1}$ dimensions,by noticing that for any vector $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {\mathbb{R}}^{d}$ ,

在本节中，我们研究 ${\ell }_{1}$ 或 ${\ell }_{\infty }$ 度量下的相似连接问题。实际上，我们将研究一个更一般的问题，即矩形包含点问题。这里我们给定一个包含 ${N}_{1}$ 个点的集合 ${R}_{1}$ 和一个包含 ${N}_{2}$ 个正交矩形的集合 ${R}_{2}$ 。目标是返回所有满足 $x \in  y$ 的对 $\left( {x,y}\right)  \in  {R}_{1} \times  {R}_{2}$ 。注意，${\ell }_{\infty }$ 度量下的相似连接等价于一个矩形包含点问题，其中矩形的每条边的长度为 ${2r}$ 。通过注意到对于任何向量 $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {\mathbb{R}}^{d}$ ，$d$ 维的 ${\ell }_{1}$ 度量下的相似连接可以简化为 ${2}^{d - 1}$ 维的 ${\ell }_{\infty }$ 度量下的相似连接。

$$
\mathop{\sum }\limits_{{i = 1}}^{d}\left| {x}_{i}\right|  = \mathop{\max }\limits_{{\left( {{z}_{2},\ldots ,{z}_{d}}\right)  \in  \{  - 1,1{\} }^{d - 1}}}\left| {{x}_{1} + {z}_{2}{x}_{2} + \cdots  + {z}_{d}{x}_{d}}\right| .
$$

In this section and the next, we will assume constant dimensions,so that ${2}^{d - 1}$ is still a constant. We deal with the high-dimensional case in Section 6.

在本节和下一节中，我们将假设维度是常数，因此 ${2}^{d - 1}$ 仍然是一个常数。我们将在第6节中处理高维情况。

### 4.1 One dimension

### 4.1 一维情况

We start by considering the one-dimensional case, i.e., the intervals-containing-points problem. We are given a set of ${N}_{1}$ points and a set of ${N}_{2}$ intervals. Set $\operatorname{IN} = {N}_{1} + {N}_{2}$ . The goal to report all (point, interval) pairs such that the point is inside the interval. Below we describe how to solve this problem in $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ load. Note that as with the equi-join case,if ${N}_{1} > p \cdot  {N}_{2}$ or ${N}_{2} > p \cdot  {N}_{1}$ ,then the problem can be trivially and optimally solved with $O\left( {\min \left( {{N}_{1},{N}_{2}}\right) }\right)$ load.

我们首先考虑一维情况，即区间包含点问题。我们给定一个包含 ${N}_{1}$ 个点的集合和一个包含 ${N}_{2}$ 个区间的集合。设 $\operatorname{IN} = {N}_{1} + {N}_{2}$ 。目标是报告所有（点，区间）对，使得点在区间内。下面我们描述如何以 $O\left( {\sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p}\right)$ 的负载解决这个问题。注意，与等值连接情况一样，如果 ${N}_{1} > p \cdot  {N}_{2}$ 或 ${N}_{2} > p \cdot  {N}_{1}$ ，那么这个问题可以用 $O\left( {\min \left( {{N}_{1},{N}_{2}}\right) }\right)$ 的负载简单且最优地解决。

## Step (1) Computing OUT

## 步骤 (1) 计算 OUT

As with the equi-join algorithm, we start by computing the value of OUT. First, we sort all the points and number them consecutively in the sorted order. Then, for each interval $I = \left\lbrack  {x,y}\right\rbrack$ ,we find the predecessor points of $x$ and $y$ (multi-search). Taking the difference of the numbers assigned to the two predecessors will give us the number of points inside $I$ . Finally,we add up all these counts to get OUT (special case of sum-by-key).

与等值连接算法一样，我们首先计算OUT的值。首先，我们对所有点进行排序，并按排序顺序依次编号。然后，对于每个区间$I = \left\lbrack  {x,y}\right\rbrack$，我们找出$x$和$y$的前序点（多重搜索）。用分配给这两个前序点的编号相减，就能得到$I$内的点数。最后，我们将所有这些计数相加，得到OUT（按键求和的特殊情况）。

## Step (2) Partially covered slabs

## 步骤 (2) 部分覆盖的条带

By setting $b = \sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p$ ,we will ensure that the load of the remaining steps is $O\left( b\right)$ . We sort all the points and divide them into slabs of size $b$ . Note that there are at most $p$ slabs. Consider each interval $I$ in ${R}_{2}$ . All the points inside $I$ can be classified into two cases: (1) points that fall in a slab partially covered by $I$ ,and (2) points that fall in a slab fully covered by $I$ . For example,in Figure 1, the join between ${I}_{1}$ and the points in the leftmost and the rightmost slab is considered under case (1), while the join between ${I}_{1}$ and the points in the two middle slabs is considered under case (2). Note that if an interval falls inside a slab completely, its join with the points in that slab is also considered under case (1),such as ${I}_{2}$ in Figure 1.

通过设置 $b = \sqrt{\mathrm{{OUT}}/p} + \mathrm{{IN}}/p$ ，我们将确保剩余步骤的负载为 $O\left( b\right)$ 。我们对所有点进行排序，并将它们划分为大小为 $b$ 的条带。注意，最多有 $p$ 个条带。考虑 ${R}_{2}$ 中的每个区间 $I$ 。$I$ 内的所有点可以分为两种情况：(1) 落在被 $I$ 部分覆盖的条带中的点，以及 (2) 落在被 $I$ 完全覆盖的条带中的点。例如，在图 1 中，${I}_{1}$ 与最左侧和最右侧条带中的点的连接属于情况 (1) ，而 ${I}_{1}$ 与两个中间条带中的点的连接属于情况 (2) 。注意，如果一个区间完全落在一个条带内，它与该条带中的点的连接也属于情况 (1) ，如图 1 中的 ${I}_{2}$ 。

<!-- Media -->

<!-- figureText: $b$ ${I}_{1}$ $b$ -->

<img src="https://cdn.noedgeai.com/0195ccc1-025b-792b-90f9-77716d7b8c19_5.jpg?x=255&y=151&w=505&h=282&r=0"/>

Figure 1: Partially covered and fully covered slabs.

图 1：部分覆盖和完全覆盖的条带。

<!-- Media -->

In this step, we deal with the partially covered slabs. For each interval endpoint, we find which slab it falls into (multi-search). Then, for each slab, we compute the number of endpoints falling inside (sum-by-key). Consider each slab $i$ . Suppose it contains $P\left( i\right)$ endpoints. We allocate $\left\lceil  {p \cdot  \frac{P\left( i\right) }{{N}_{2}}}\right\rceil$ servers to compute the join between the $b$ points in the slab and the intervals with these $P\left( i\right)$ endpoints (note that we need $O\left( p\right)$ servers). We simply evenly allocate the $P\left( i\right)$ intervals to these servers (use multi-numbering to ensure balance),and broadcast all the $b$ points to them. The load is thus

在这一步中，我们处理部分覆盖的条带。对于每个区间端点，我们找出它落在哪个条带中（多重搜索）。然后，对于每个条带，我们计算落在其中的端点数量（按键求和）。考虑每个条带 $i$ 。假设它包含 $P\left( i\right)$ 个端点。我们分配 $\left\lceil  {p \cdot  \frac{P\left( i\right) }{{N}_{2}}}\right\rceil$ 台服务器来计算条带中的 $b$ 个点与具有这些 $P\left( i\right)$ 个端点的区间之间的连接（注意，我们需要 $O\left( p\right)$ 台服务器）。我们简单地将 $P\left( i\right)$ 个区间均匀分配给这些服务器（使用多重编号以确保平衡），并将所有 $b$ 个点广播给它们。因此负载为

$$
O\left( {b + \frac{P\left( i\right) }{{pP}\left( i\right) /{N}_{2}}}\right)  = O\left( b\right) .
$$

## Step (3) Fully covered slabs

## 步骤 (3) 完全覆盖的条带

Let $F\left( i\right)$ be the number of intervals fully covering a slab $i$ . We can compute all the $F\left( i\right)$ ’s using all prefix-sums algorithm,as follows. If an interval fully covers slabs $i,\ldots ,j$ , we generate two pairs(i,1)and $\left( {j + 1, - 1}\right)$ . For each slab $i$ , we generate a pair $\left( {i + {0.5},0}\right)$ . Then we sort all these(i,v) pairs by $i$ and compute the prefix-sums on the $v$ ’s.

设 $F\left( i\right)$ 为完全覆盖条带 $i$ 的区间数量。我们可以使用全前缀和算法计算所有的 $F\left( i\right)$ ，如下所示。如果一个区间完全覆盖条带 $i,\ldots ,j$ ，我们生成两对 (i,1) 和 $\left( {j + 1, - 1}\right)$ 。对于每个条带 $i$ ，我们生成一对 $\left( {i + {0.5},0}\right)$ 。然后我们按 $i$ 对所有这些 (i,v) 对进行排序，并对 $v$ 计算前缀和。

Now, the full slabs can be dealt with using essentially the same algorithm. We allocate ${p}_{i} = \left\lceil  {p \cdot  \frac{{bF}\left( i\right) }{\text{ OUT }}}\right\rceil$ servers to compute the join (full Cartesian product) of the $b$ points in slab $i$ and the $F\left( i\right)$ intervals fully covering the slab. Since $\mathop{\sum }\limits_{i}{bF}\left( i\right)  \leq$ OUT,this requires at most $O\left( p\right)$ servers. We simply evenly allocate the $F\left( i\right)$ intervals to these servers and broadcast all the $b$ points to them. The load is thus

现在，基本上可以使用相同的算法处理完全覆盖的条带。我们分配 ${p}_{i} = \left\lceil  {p \cdot  \frac{{bF}\left( i\right) }{\text{ OUT }}}\right\rceil$ 台服务器来计算条带 $i$ 中的 $b$ 个点与完全覆盖该条带的 $F\left( i\right)$ 个区间的连接（全笛卡尔积）。由于 $\mathop{\sum }\limits_{i}{bF}\left( i\right)  \leq$ 输出，这最多需要 $O\left( p\right)$ 台服务器。我们简单地将 $F\left( i\right)$ 个区间均匀分配给这些服务器，并将所有 $b$ 个点广播给它们。因此负载为

$$
O\left( {b + \frac{F\left( i\right) }{{pbF}\left( i\right) /\mathrm{{OUT}}}}\right)  = O\left( {b + \frac{\mathrm{{OUT}}}{pb}}\right)  = O\left( b\right) .
$$

THEOREM 3. There is a deterministic algorithm for the intervals-containing-points problem that runs in $O\left( 1\right)$ rounds with $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ load.

定理 3. 对于区间包含点问题，存在一种确定性算法，该算法在 $O\left( 1\right)$ 轮内运行，负载为 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}}\right)$ 。

### 4.2 Two and higher dimensions

### 4.2 二维及更高维度

Next, we generalize the algorithm above to two dimensions. Here we are given a set of ${N}_{1}$ points in $2\mathrm{D}$ and a set of ${N}_{2}$ rectangles. Set $\operatorname{IN} = {N}_{1} + {N}_{2}$ . The goal is to report all (point, rectangle) pairs such that the point is inside the rectangle.

接下来，我们将上述算法推广到二维情况。这里我们给定了 $2\mathrm{D}$ 中的一组 ${N}_{1}$ 个点和一组 ${N}_{2}$ 个矩形。设 $\operatorname{IN} = {N}_{1} + {N}_{2}$ 。目标是报告所有（点，矩形）对，使得点在矩形内部。

## Step (1) Computing OUT

## 步骤 (1) 计算输出

The first step is still to compute the output size OUT. We first sort all the $x$ -coordinates,including those of the points and those of the left and right sides of the rectangles. Then each server defines a vertical slab,containing $O\left( {\mathrm{{IN}}/p}\right)$ $x$ -coordinates. All joining (point,rectangle) pairs that land on the same server can be counted and output easily. For example,in Figure 2,the join results between ${\sigma }_{1}$ and all the points in slab 1 and slab 7 can be found by those two servers easily. This also includes rectangles that completely fall inside a slab,such as ${\sigma }_{2}$ .

第一步仍然是计算输出大小OUT。我们首先对所有的$x$坐标进行排序，包括点的坐标以及矩形左右两边的坐标。然后，每个服务器定义一个垂直条带，其中包含$O\left( {\mathrm{{IN}}/p}\right)$个$x$坐标。所有落在同一服务器上的（点，矩形）对都可以轻松计数并输出。例如，在图2中，${\sigma }_{1}$与条带1和条带7中所有点的连接结果可以由这两个服务器轻松找到。这也包括完全落在一个条带内的矩形，例如${\sigma }_{2}$。

<!-- Media -->

<!-- figureText: 2 3 5 6 8 -->

<img src="https://cdn.noedgeai.com/0195ccc1-025b-792b-90f9-77716d7b8c19_5.jpg?x=955&y=308&w=648&h=345&r=0"/>

Figure 2: Rectangles-joining-points.

图2：矩形与点的连接。

<!-- Media -->

So we are left to count the joining pairs such that the $x$ - projection of the rectangle fully spans the slab containing the point. We impose a binary hierarchy on the $p$ slabs, and decompose each rectangle into $O\left( {\log p}\right)$ canonical slabs. For example, ${\sigma }_{1}$ in Figure 2 fully spans slabs $2 - 6$ ,and it is decomposed into 3 canonical slabs: $2,3 - 4,5 - 6$ .

因此，我们接下来要计算这样的连接对：矩形在$x$上的投影完全跨越包含该点的条带。我们在$p$个条带上施加一个二叉层次结构，并将每个矩形分解为$O\left( {\log p}\right)$个规范条带。例如，图2中的${\sigma }_{1}$完全跨越条带$2 - 6$，它被分解为3个规范条带：$2,3 - 4,5 - 6$。

This results in a total of $O\left( {{N}_{2}\log p}\right)$ canonical rectangles, each of which corresponds to one canonical slab. For each canonical slab $s$ ,we count its output size $\operatorname{OUT}\left( s\right)$ ,using step (1) of the 1D algorithm. To run all these instances in parallel, we allocate the servers as follows. For a canonical slab $s$ that has ${N}_{2}\left( s\right)$ canonical rectangles and consists of $k\left( s\right)$ atomic slabs,we allocate ${p}_{s} = \left\lceil  {p \cdot  \frac{k\left( s\right) \mathrm{{IN}}/p + {N}_{2}\left( s\right) }{\mathrm{{IN}}\log p}}\right\rceil$ servers. As each atomic slab is covered by $O\left( {\log p}\right)$ canonical slabs,we have $\mathop{\sum }\limits_{s}k\left( s\right)  = O\left( {p\log p}\right)$ . This step uses $O\left( p\right)$ servers,with each server having load $O\left( \frac{k\left( s\right) \operatorname{IN}/p + {N}_{2}\left( s\right) }{{p}_{s}}\right)  =$ $O\left( {\frac{\mathrm{{IN}}}{p}\log p}\right)$ . To count the ${N}_{2}\left( s\right)$ ’s,we first count $F\left( i\right)$ ,the number of rectangles fully covering an atomic slab $i$ . This is the same as the $F\left( i\right)$ ’s in the 1D case,can be counted with $O\left( {\mathrm{{IN}}/p}\right)$ load. Then for each atomic slab $i$ ,we produce $O\left( {\log p}\right)$ pairs $\left( {s,F\left( i\right) }\right)$ ,one for each canonical slab $s$ that contains $i$ . This generates $O\left( {p\log p}\right)$ such pairs. Finally,we run sum-by-key on these pairs (using $s$ as the key) to compute ${N}_{2}\left( s\right)$ ,with load $O\left( {p\log p/p}\right)  = O\left( {\log p}\right)  = O\left( {\mathrm{{IN}}/p}\right)$ .

这总共产生了$O\left( {{N}_{2}\log p}\right)$个规范矩形，每个规范矩形对应一个规范条带。对于每个规范条带$s$，我们使用一维算法的步骤（1）来计算其输出大小$\operatorname{OUT}\left( s\right)$。为了并行运行所有这些实例，我们按如下方式分配服务器。对于一个具有${N}_{2}\left( s\right)$个规范矩形且由$k\left( s\right)$个原子条带组成的规范条带$s$，我们分配${p}_{s} = \left\lceil  {p \cdot  \frac{k\left( s\right) \mathrm{{IN}}/p + {N}_{2}\left( s\right) }{\mathrm{{IN}}\log p}}\right\rceil$个服务器。由于每个原子条带被$O\left( {\log p}\right)$个规范条带覆盖，我们有$\mathop{\sum }\limits_{s}k\left( s\right)  = O\left( {p\log p}\right)$。这一步使用$O\left( p\right)$个服务器，每个服务器的负载为$O\left( \frac{k\left( s\right) \operatorname{IN}/p + {N}_{2}\left( s\right) }{{p}_{s}}\right)  =$ $O\left( {\frac{\mathrm{{IN}}}{p}\log p}\right)$。为了计算${N}_{2}\left( s\right)$的值，我们首先计算$F\left( i\right)$，即完全覆盖一个原子条带$i$的矩形数量。这与一维情况下的$F\left( i\right)$相同，可以以$O\left( {\mathrm{{IN}}/p}\right)$的负载进行计算。然后，对于每个原子条带$i$，我们生成$O\left( {\log p}\right)$对$\left( {s,F\left( i\right) }\right)$，每个包含$i$的规范条带$s$对应一对。这会生成$O\left( {p\log p}\right)$这样的对。最后，我们对这些对（使用$s$作为键）执行按键求和操作来计算${N}_{2}\left( s\right)$，负载为$O\left( {p\log p/p}\right)  = O\left( {\log p}\right)  = O\left( {\mathrm{{IN}}/p}\right)$。

Note that this step always has load $O\left( {\frac{\mathrm{{IN}}}{p}\log p}\right)$ regardless of how large OUT is.

请注意，无论OUT有多大，这一步的负载始终为$O\left( {\frac{\mathrm{{IN}}}{p}\log p}\right)$。

## Step (2) Reduction to the 1D case

## 步骤（2）简化为一维情况

In this step,we compute,for each rectangle $\sigma$ ,all the result pairs produced by $\sigma$ and the points in the slabs fully spanned by $\sigma$ . We follow the same approach as in step (1), but will also need to take into account the output size of each canonical slab,i.e., $\operatorname{OUT}\left( s\right)$ ,when allocating the servers. More precisely,for a canonical slab $s$ that has ${N}_{2}\left( s\right)$ canonical rectangles and consists of $k\left( s\right)$ contiguous slabs,we allocate

在这一步中，我们为每个矩形 $\sigma$ 计算由 $\sigma$ 以及被 $\sigma$ 完全跨越的条带中的点所产生的所有结果对。我们采用与步骤 (1) 相同的方法，但在分配服务器时，还需要考虑每个规范条带的输出大小，即 $\operatorname{OUT}\left( s\right)$。更准确地说，对于一个具有 ${N}_{2}\left( s\right)$ 个规范矩形且由 $k\left( s\right)$ 个连续条带组成的规范条带 $s$，我们分配

$$
{p}_{s} = \left\lbrack  {p \cdot  \frac{\mathrm{{OUT}}\left( s\right) }{\mathrm{{OUT}}} + p \cdot  \frac{k\left( s\right) \mathrm{{IN}}/p + {N}_{2}\left( s\right) }{\mathrm{{IN}}\log p}}\right\rbrack  
$$

servers,and invoke step (2) and (3) of the 1D algorithm (since $\operatorname{OUT}\left( s\right)$ is already computed) for all the canonical slabs in parallel. Note that a point knows which canonical slabs it falls into,hence which 1D instances it should participate in, from just its own slab number. For each canonical slab $s$ ,denote the size of its derived instance as $\operatorname{IN}\left( s\right)  = k\left( s\right) \operatorname{IN}/p + {N}_{2}\left( s\right)$ . Plugging OUT $= \operatorname{OUT}\left( s\right) ,\operatorname{IN} =$ $\operatorname{IN}\left( s\right) ,p = {p}_{s}$ into Theorem 3 yields the following result.

服务器，并针对所有规范条带并行调用一维算法的步骤 (2) 和 (3)（因为 $\operatorname{OUT}\left( s\right)$ 已经计算得出）。请注意，一个点仅根据其自身的条带编号就知道它属于哪些规范条带，从而知道它应该参与哪些一维实例。对于每个规范条带 $s$，将其派生实例的大小记为 $\operatorname{IN}\left( s\right)  = k\left( s\right) \operatorname{IN}/p + {N}_{2}\left( s\right)$。将 OUT $= \operatorname{OUT}\left( s\right) ,\operatorname{IN} =$ $\operatorname{IN}\left( s\right) ,p = {p}_{s}$ 代入定理 3 可得到以下结果。

THEOREM 4. There is a deterministic algorithm for the rectangles-containing-points problem in 2D that runs in $O\left( 1\right)$ rounds with $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}\log p}\right)$ load.

定理 4. 对于二维中的矩形包含点问题，存在一种确定性算法，该算法在 $O\left( 1\right)$ 轮内运行，负载为 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}\log p}\right)$。

The algorithm can also be extended to higher dimensions using similar ideas,with an extra $O\left( {\log p}\right)$ factor for each dimension higher. We give the following result without proof.

使用类似的思路，该算法也可以扩展到更高维度，每增加一个维度会有一个额外的 $O\left( {\log p}\right)$ 因子。我们给出以下结果，但不进行证明。

THEOREM 5. There is a deterministic algorithm for the rectangles-containing-points problem in $d$ dimensions that runs in $O\left( 1\right)$ rounds with $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}{\log }^{d - 1}p}\right)$ load.

定理 5. 对于 $d$ 维中的矩形包含点问题，存在一种确定性算法，该算法在 $O\left( 1\right)$ 轮内运行，负载为 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{p}{\log }^{d - 1}p}\right)$。

## 5. SIMILARITY JOIN UNDER ${\ell }_{2}$

## 5. ${\ell }_{2}$ 下的相似性连接

In this section, we consider the similarity join between two point sets ${R}_{1}$ and ${R}_{2}$ under the ${\ell }_{2}$ distance in $d$ dimensions. We first use the lifting transformation [13] to convert the problem to the halfspaces-containing-points problem in $d + 1$ dimensions. Consider any two points $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {R}_{1}$ and $\left( {{y}_{1},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$ . The two points join under the ${\ell }_{2}$ distance if

在本节中，我们考虑在 $d$ 维空间中，两个点集 ${R}_{1}$ 和 ${R}_{2}$ 在 ${\ell }_{2}$ 距离下的相似性连接。我们首先使用提升变换 [13] 将该问题转换为 $d + 1$ 维空间中的半空间包含点问题。考虑任意两个点 $\left( {{x}_{1},\ldots ,{x}_{d}}\right)  \in  {R}_{1}$ 和 $\left( {{y}_{1},\ldots ,{y}_{d}}\right)  \in  {R}_{2}$。如果这两个点在 ${\ell }_{2}$ 距离下进行连接

$$
{\left( {x}_{1} - {y}_{2}\right) }^{2} + \cdots  + {\left( {x}_{d} - {y}_{d}\right) }^{2} \leq  {r}^{2},
$$

or

或者

$$
{x}_{1}^{2} + {y}_{1}^{2} + \cdots  + {x}_{d}^{2} + {y}_{d}^{2} - 2{x}_{1}{y}_{1} - \cdots  - 2{x}_{d}{y}_{d} - {r}^{2} \geq  0.
$$

We map the point $\left( {{x}_{1},\ldots ,{x}_{d}}\right)$ to a point in $d + 1$ dimensions: $\left( {{x}_{1},\ldots ,{x}_{d},{x}_{1}^{2} + \cdots  + {x}_{d}^{2}}\right)$ ,and the point $\left( {{y}_{1},\ldots ,{y}_{d}}\right)$ to a halfspace in $d + 1$ dimensions ( ${z}_{i}$ ’s are the parameters):

我们将点 $\left( {{x}_{1},\ldots ,{x}_{d}}\right)$ 映射到 $d + 1$ 维空间中的一个点：$\left( {{x}_{1},\ldots ,{x}_{d},{x}_{1}^{2} + \cdots  + {x}_{d}^{2}}\right)$，并将点 $\left( {{y}_{1},\ldots ,{y}_{d}}\right)$ 映射到 $d + 1$ 维空间中的一个半空间（${z}_{i}$ 为参数）：

$$
 - 2{y}_{1}{z}_{1} - \cdots  - 2{y}_{d}{z}_{d} + {z}_{d + 1} + {y}_{1}^{2} + \cdots  + {y}_{d}^{2} - {r}^{2} \geq  0.
$$

We see that the two $d$ -dimensional points join if and only if the corresponding $\left( {d + 1}\right)$ -dimensional halfspace contains the $\left( {d + 1}\right)$ -dimensional point.

我们发现，当且仅当对应的 $\left( {d + 1}\right)$ 维半空间包含 $\left( {d + 1}\right)$ 维点时，这两个 $d$ 维点才会进行连接。

Thus, in the following, we will study the halfspaces-containing-points problem. Given a set of ${N}_{1}$ points and a set of ${N}_{2}$ halfspaces in $d$ dimensions,report all the (point,halfspace) pairs such that the point is inside the halfspace.

因此，在接下来的内容中，我们将研究半空间包含点问题。给定 $d$ 维空间中的一组 ${N}_{1}$ 个点和一组 ${N}_{2}$ 个半空间，报告所有满足点位于半空间内的（点，半空间）对。

A key challenge in this problem,compared with the ${\ell }_{1}/{\ell }_{\infty }$ case, is that there is no easy way to compute OUT, due to the non-orthogonal nature of the problem. Knowing the value of OUT is crucial in the previous algorithms, which is used to determine the right slab size, which in turn decides the load.

与${\ell }_{1}/{\ell }_{\infty }$情况相比，这个问题的一个关键挑战在于，由于问题的非正交性质，没有简单的方法来计算OUT。在之前的算法中，知道OUT的值至关重要，它用于确定合适的板片大小，而板片大小又决定了负载。

Our way to get around this problem is based on the observation that the load is determined by the output-dependent term only when OUT is sufficiently large. But in this case, a constant-factor approximation of OUT suffices to guarantee the optimal load asymptotically, and random sampling can be used to estimate OUT. Random sampling will not be effective when OUT is small (it is known that to decide whether OUT $= 1$ or 0 by sampling requires us to essentially sample the whole data set), but in that case, the input-dependent term will dominate the load, and we do not need to know the value of OUT anyway.

我们解决这个问题的方法基于这样的观察：只有当OUT足够大时，负载才由与输出相关的项决定。但在这种情况下，OUT的常数因子近似足以渐近地保证最优负载，并且可以使用随机采样来估计OUT。当OUT较小时，随机采样将无效（已知通过采样来确定OUT是$= 1$还是0本质上需要对整个数据集进行采样），但在这种情况下，与输入相关的项将主导负载，而且无论如何我们都不需要知道OUT的值。

The following notion of a $\theta$ -thresholded approximation captures our needs, which will be used in the development of the algorithm.

以下关于$\theta$阈值近似的概念满足了我们的需求，它将在算法的开发中使用。

DEFINITION 1. A $\theta$ -thresholded approximation of $x$ is an estimate $\widehat{x}$ such that: (1) if $x \geq  \theta$ ,then $\frac{x}{2} < \widehat{x} < {2x}$ ; (2) if $x < \theta$ ,then $\widehat{x} < {2\theta }$ .

定义1. $x$的$\theta$阈值近似是一个估计值$\widehat{x}$，使得：(1) 如果$x \geq  \theta$，那么$\frac{x}{2} < \widehat{x} < {2x}$；(2) 如果$x < \theta$，那么$\widehat{x} < {2\theta }$。

### 5.1 Useful tools from computational geometry

### 5.1 计算几何中的有用工具

We will need the following tools from computational geometry. The first relates thresholded approximations with random sampling.

我们将需要计算几何中的以下工具。第一个工具将阈值近似与随机采样联系起来。

THEOREM 6 ([23,17]). For any $q > 1$ ,let $S$ be a random sample from a set $P$ of $n$ points with $\left| S\right|  = O\left( {q\log \left( {q/\delta }\right) }\right)$ . Then with probability at least $1 - \delta ,n \cdot  \frac{\left| \Delta  \cap  S\right| }{\left| S\right| }$ is a $\frac{n}{q}$ -thresholded approximation of $\left| {\Delta  \cap  P}\right|$ for every simplex $\Delta$ .

定理6 ([23,17])。对于任意$q > 1$，设$S$是从包含$n$个点的集合$P$中进行的随机采样，其中$\left| S\right|  = O\left( {q\log \left( {q/\delta }\right) }\right)$。那么，对于每个单纯形$\Delta$，$1 - \delta ,n \cdot  \frac{\left| \Delta  \cap  S\right| }{\left| S\right| }$至少以$\frac{n}{q}$的概率是$\left| {\Delta  \cap  P}\right|$的$\frac{n}{q}$阈值近似。

Next, we introduce the partition tree. In particular, we make use of the b-partial partition tree of Chan [11].

接下来，我们介绍划分树。特别地，我们使用Chan [11]提出的b - 部分划分树。

A $b$ -partial partition tree on a set of points is a tree $T$ with constant fanout,where each leaf stores at most $b$ points,and each point is stored in exactly one leaf. Each node $v \in  T$ (both internal nodes and leaf nodes) stores a cell $\Delta \left( v\right)$ ,which is a simplex that encloses all the points stored at the leaves below $v$ . For any $v$ ,the cells of its children do not overlap. In particular, this implies that all the leaf cells are disjoint. Chan [11] presented an algorithm to construct a $b$ -partial partition tree with the following properties.

点集上的$b$ - 部分划分树是一棵扇出为常数的树$T$，其中每个叶子节点最多存储$b$个点，并且每个点恰好存储在一个叶子节点中。每个节点$v \in  T$（包括内部节点和叶子节点）存储一个单元$\Delta \left( v\right)$，它是一个包含存储在$v$下方叶子节点处所有点的单纯形。对于任意$v$，其孩子节点的单元不重叠。特别地，这意味着所有叶子单元是不相交的。Chan [11]提出了一种算法来构造具有以下性质的$b$ - 部分划分树。

THEOREM 7 ([11]). Given $n$ points in ${\mathbb{R}}^{d}$ and a parameter $b < n/{\log }^{\omega \left( 1\right) }n$ ,we can build a $b$ -partial partition tree with $O\left( {n/b}\right)$ nodes,such that any hyperplane intersects $O\left( {\left( n/b\right) }^{1 - 1/d}\right)$ cells of the tree.

定理7 ([11])。给定${\mathbb{R}}^{d}$中的$n$个点和一个参数$b < n/{\log }^{\omega \left( 1\right) }n$，我们可以构建一个具有$O\left( {n/b}\right)$个节点的$b$ - 部分划分树，使得任何超平面与树的$O\left( {\left( n/b\right) }^{1 - 1/d}\right)$个单元相交。

Chan's construction only guarantees that each leaf cell contains at most $b$ points but offers no lower bound,while we will need each leaf cell to have $\Theta \left( b\right)$ points. This can be easily achieved,though. Suppose Chan’s $b$ -partial partition tree has at most $c \cdot  n/b$ leaves for some constant $c$ . A leaf cell is big if it contains at least $b/{2c}$ points; otherwise small. Observe that there must be at least $n/{2b}$ big leaf cells. Then, we simply combine $O\left( 1\right)$ small leaf cells with each of the big leaf cells. This will eliminate all small leaf cells, while each merged cell consists of a constant number of the original cells.

Chan的构造仅保证每个叶子单元最多包含$b$个点，但没有给出下界，而我们需要每个叶子单元有$\Theta \left( b\right)$个点。不过，这很容易实现。假设Chan的$b$ - 部分划分树对于某个常数$c$最多有$c \cdot  n/b$个叶子节点。如果一个叶子单元至少包含$b/{2c}$个点，则称其为大叶子单元；否则为小叶子单元。可以观察到，至少有$n/{2b}$个大叶子单元。然后，我们只需将$O\left( 1\right)$个小叶子单元与每个大叶子单元合并。这将消除所有小叶子单元，而每个合并后的单元由固定数量的原始单元组成。

### 5.2 The algorithm

### 5.2 算法

Let $q$ be a parameter such that $1 < q < p$ . The value of $q$ will be determined later.

设 $q$ 为一个参数，满足 $1 < q < p$ 。 $q$ 的值将在后面确定。

## Step (1) Constructing a partition tree

## 步骤 (1) 构建分区树

Randomly sample $\Theta \left( {q\log p}\right)$ points,and send all the sampled points to one server. The server builds a $\Theta \left( {\log p}\right)$ - partial partition tree on the sampled points. By Theorem 6 and 7,this tree has $O\left( q\right)$ nodes,and with probability $1 - 1/{p}^{O\left( 1\right) }$ ,every leaf cell contains $O\left( {{N}_{1}/q}\right)$ points in the original data set,and any hyperplane intersects $O\left( {q}^{1 - \frac{1}{d}}\right)$ cells. Then we perform the merging process described after Theorem 7. Since each merged cell consists of $O\left( 1\right)$ leaf cells, each merged cell has constant description complexity and still contains $O\left( {{N}_{1}/q}\right)$ points. Note that the merging process does not increase the total cell count and the number of cells intersected by any hyperplane.

随机采样 $\Theta \left( {q\log p}\right)$ 个点，并将所有采样点发送到一台服务器。服务器在采样点上构建一个 $\Theta \left( {\log p}\right)$ - 部分分区树。根据定理 6 和 7，这棵树有 $O\left( q\right)$ 个节点，并且以概率 $1 - 1/{p}^{O\left( 1\right) }$ ，每个叶单元格包含原始数据集中的 $O\left( {{N}_{1}/q}\right)$ 个点，并且任何超平面与 $O\left( {q}^{1 - \frac{1}{d}}\right)$ 个单元格相交。然后我们执行定理 7 之后描述的合并过程。由于每个合并后的单元格由 $O\left( 1\right)$ 个叶单元格组成，每个合并后的单元格具有恒定的描述复杂度，并且仍然包含 $O\left( {{N}_{1}/q}\right)$ 个点。请注意，合并过程不会增加单元格总数以及任何超平面相交的单元格数量。

We broadcast these merged cells to all servers. This step incurs a load of $O\left( {q\log p}\right)$ . Henceforth,a "cell" will refer to such a merged cell.

我们将这些合并后的单元格广播到所有服务器。此步骤产生的负载为 $O\left( {q\log p}\right)$ 。此后，“单元格”将指这样的合并单元格。

Similar to the intervals-containing-points algorithm, we consider the following two cases for all points inside a half-space: (1) those in cells partially covered by the halfspace, and (2) those in cells fully covered by the halfspace.

与包含点的区间算法类似，我们考虑半空间内所有点的以下两种情况：(1) 位于被半空间部分覆盖的单元格中的点，以及 (2) 位于被半空间完全覆盖的单元格中的点。

## Step (2) Partially covered cells

## 步骤 (2) 部分覆盖的单元格

For each halfspace,we find all the cells $\Delta$ such that its bounding halfplane intersects $\Delta$ . There are $O\left( {q}^{1 - \frac{1}{d}}\right)$ such intersecting cells,by Theorem 7. For each cell $\Delta$ ,we compute the number of halfspaces whose bounding halfplane intersects $\Delta$ ,denoted as $P\left( \Delta \right)$ . Note that $\mathop{\sum }\limits_{\Delta }P\left( \Delta \right)  =$ $O\left( {{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}\right)$ ,so this is a sum-by-key problem on a total of $O\left( {{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}\right)$ key-value pairs. The load is thus

对于每个半空间，我们找到所有满足其边界半平面与 $\Delta$ 相交的单元格 $\Delta$ 。根据定理 7，有 $O\left( {q}^{1 - \frac{1}{d}}\right)$ 个这样的相交单元格。对于每个单元格 $\Delta$ ，我们计算其边界半平面与 $\Delta$ 相交的半空间的数量，记为 $P\left( \Delta \right)$ 。请注意， $\mathop{\sum }\limits_{\Delta }P\left( \Delta \right)  =$ $O\left( {{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}\right)$ ，因此这是一个关于总共 $O\left( {{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}\right)$ 个键值对的按键求和问题。因此负载为

$$
O\left( {\frac{{N}_{2}}{p} \cdot  {q}^{1 - \frac{1}{d}}}\right) . \tag{2}
$$

For each cell $\Delta$ ,we allocate ${p}_{\Delta } = \left\lceil  {p \cdot  \frac{P\left( \Delta \right) }{{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}}\right\rceil$ servers to compute the join between the $\Theta \left( {{N}_{1}/q}\right)$ points in the cell and these $P\left( \Delta \right)$ halfspaces that partially cover $\Delta$ . Note that the total number of servers needed is $\mathop{\sum }\limits_{\Delta }{p}_{\Delta } = O\left( {q + p}\right)  = O\left( p\right)$ as there are $O\left( q\right)$ cells in total. Invoking the hypercube algorithm to compute their Cartesian product incurs a load of

对于每个单元格 $\Delta$ ，我们分配 ${p}_{\Delta } = \left\lceil  {p \cdot  \frac{P\left( \Delta \right) }{{N}_{2} \cdot  {q}^{1 - \frac{1}{d}}}}\right\rceil$ 台服务器来计算单元格中的 $\Theta \left( {{N}_{1}/q}\right)$ 个点与部分覆盖 $\Delta$ 的这些 $P\left( \Delta \right)$ 个半空间之间的连接。请注意，由于总共有 $O\left( q\right)$ 个单元格，因此所需的服务器总数为 $\mathop{\sum }\limits_{\Delta }{p}_{\Delta } = O\left( {q + p}\right)  = O\left( p\right)$ 。调用超立方体算法来计算它们的笛卡尔积会产生的负载为

$$
O\left( {\sqrt{\frac{\frac{{N}_{1}}{q} \cdot  P\left( \Delta \right) }{{p}_{\Delta }}} + \frac{\frac{{N}_{1}}{q} + P\left( \Delta \right) }{{p}_{\Delta }}}\right) 
$$

$$
 = O\left( {\sqrt{\frac{{N}_{1}{N}_{2}}{p{q}^{\frac{1}{d}}}} + \frac{{N}_{1}}{q} + \frac{{N}_{2}{q}^{1 - \frac{1}{d}}}{p}}\right) . \tag{3}
$$

Choosing $q = {p}^{\frac{d}{{2d} - 1}}$ balances the terms in (2) and (3), and the load becomes

选择 $q = {p}^{\frac{d}{{2d} - 1}}$ 可以平衡 (2) 和 (3) 中的项，并且负载变为

$$
O\left( \frac{\mathrm{{IN}}}{q}\right)  = O\left( \frac{\mathrm{{IN}}}{{p}^{d/\left( {{2d} - 1}\right) }}\right) .
$$

## Step (3): Fully covered cells

## 步骤 (3)：完全覆盖的单元格

In the intervals-containing-points algorithm, fully covered cells are dealt in a way similar to the partially covered cells, but that is because we can compute OUT and set the right slab size. In this algorithm, we may have used a cell size (i.e., $\mathrm{{IN}}/q$ ) that is too small in relation to OUT. This would result in too many join results to be produced for the fully covered cells, exceeding the load target. Our strategy is thus to first estimate the join size for the fully covered cells (which is easier than computing OUT), and then rectify the mistake by restarting the whole algorithm with the right cell size, if needed.

在包含点的区间算法中，完全覆盖的单元格的处理方式与部分覆盖的单元格类似，但这是因为我们可以计算 OUT 并设置合适的平板大小。在这个算法中，我们可能使用了相对于 OUT 来说太小的单元格大小（即 $\mathrm{{IN}}/q$ ）。这将导致为完全覆盖的单元格产生过多的连接结果，超过负载目标。因此，我们的策略是首先估计完全覆盖的单元格的连接大小（这比计算 OUT 更容易），然后如果需要，通过使用合适的单元格大小重新启动整个算法来纠正错误。

## Step (3.1): Join size estimation

## 步骤 (3.1)：连接大小估计

For each cell $\Delta$ ,let $F\left( \Delta \right)$ be the number of halfspaces fully covering it,and let $K = \mathop{\sum }\limits_{\bigtriangleup }F\left( \bigtriangleup \right)$ . Since every point inside $\Delta$ joins with every halfspace fully covering $\Delta ,K \cdot  {N}_{1}/q$ is (a constant-factor approximation of) the remaining output size,and we will be able to estimate $K$ easily.

对于每个单元格 $\Delta$，设 $F\left( \Delta \right)$ 为完全覆盖它的半空间数量，并设 $K = \mathop{\sum }\limits_{\bigtriangleup }F\left( \bigtriangleup \right)$。由于 $\Delta$ 内的每个点与完全覆盖 $\Delta ,K \cdot  {N}_{1}/q$ 的每个半空间相连，这是剩余输出大小的（一个常数因子近似），并且我们将能够轻松估计 $K$。

We first compute an $\left( \frac{{N}_{2}}{q}\right)$ -thresholded approximation of $F\left( \Delta \right)$ for each $\Delta$ . This can be done by sampling $O\left( {q\log p}\right)$ halfspaces and collecting them on one server. For each cell $\Delta$ ,we count the number of sampled halfspaces fully covering it, and scale up appropriately. Standard Chernoff type of analysis shows that with probability $1 - 1/{p}^{O\left( 1\right) }$ ,we get an $\left( \frac{{N}_{2}}{q}\right)$ -thresholded approximation for every $F\left( \Delta \right)$ . We use these approximate $F\left( \Delta \right)$ ’s to compute $\widehat{K}$ ,which is then an ${N}_{2}$ -thresholded approximation of the true value of $K$ .

我们首先为每个 $\Delta$ 计算 $F\left( \Delta \right)$ 的 $\left( \frac{{N}_{2}}{q}\right)$ -阈值近似。这可以通过采样 $O\left( {q\log p}\right)$ 个半空间并将它们收集到一台服务器上来完成。对于每个单元格 $\Delta$，我们计算完全覆盖它的采样半空间的数量，并进行适当的放大。标准的切尔诺夫类型分析表明，以概率 $1 - 1/{p}^{O\left( 1\right) }$，我们可以为每个 $F\left( \Delta \right)$ 得到一个 $\left( \frac{{N}_{2}}{q}\right)$ -阈值近似。我们使用这些近似的 $F\left( \Delta \right)$ 来计算 $\widehat{K}$，它是 $K$ 真实值的 ${N}_{2}$ -阈值近似。

Step (3.2): If $\widehat{K} < \frac{\mathrm{{IN}} \cdot  p}{q}$

步骤 (3.2)：如果 $\widehat{K} < \frac{\mathrm{{IN}} \cdot  p}{q}$

Since we have chosen $q = o\left( p\right)$ ,if $\widehat{K} < \frac{\mathrm{{IN}} \cdot  p}{q}$ and $\widehat{K}$ is an ${N}_{2}$ -thresholded approximation of $K$ ,then we must have $K = O\left( \frac{\mathrm{{IN}} \cdot  p}{q}\right)$ . In this case,we just break up each halfspace that fully covers $k$ cells into $k$ small pieces,which results in a total of $K$ pieces. Now every piece covers exactly one cell, thus joins with all the points in that cell. The problem now reduces to an equi-join on two relations of size ${N}_{1}$ and $K$ . Invoking the hypercube algorithm, the load is

由于我们选择了 $q = o\left( p\right)$，如果 $\widehat{K} < \frac{\mathrm{{IN}} \cdot  p}{q}$ 且 $\widehat{K}$ 是 $K$ 的 ${N}_{2}$ -阈值近似，那么我们必定有 $K = O\left( \frac{\mathrm{{IN}} \cdot  p}{q}\right)$。在这种情况下，我们只需将每个完全覆盖 $k$ 个单元格的半空间拆分成 $k$ 个小块，这将总共得到 $K$ 个小块。现在每个小块恰好覆盖一个单元格，因此与该单元格中的所有点相连。现在问题简化为对大小为 ${N}_{1}$ 和 $K$ 的两个关系进行等值连接。调用超立方体算法，负载为

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{K + {N}_{1}}{p}}\right)  = O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{q}}\right) .
$$

Step (3.3): If $\widehat{K} > \frac{\mathrm{{IN}} \cdot  p}{q}$

步骤 (3.3)：如果 $\widehat{K} > \frac{\mathrm{{IN}} \cdot  p}{q}$

In this case, we cannot afford to reduce the problem to an equi-join, since the halfspaces cover too many cells. This means we have used a cell size too small. Now, we restart the whole algorithm,but with a new ${q}^{\prime } = \sqrt{\frac{\mathrm{{IN}} \cdot  {pq}}{\widehat{K}}} < q$ . In the re-execution of the algorithm, we further merge every $O\left( {q/{q}^{\prime }}\right)$ cells into a bigger cell containing $\Theta \left( {{N}_{1}/{q}^{\prime }}\right)$ points. Now, each newly merged cell has non-constant description complexity,but since there are only a total of $O\left( q\right)$ cells,the entire mapping from these cells to the newly merged cells can be broadcast to all servers. Each server can still identify, for each of its points, which newly merged cell contains it.

在这种情况下，我们无法将问题简化为等值连接，因为半空间覆盖了太多单元格。这意味着我们使用的单元格尺寸太小。现在，我们重新启动整个算法，但使用一个新的 ${q}^{\prime } = \sqrt{\frac{\mathrm{{IN}} \cdot  {pq}}{\widehat{K}}} < q$。在算法的重新执行中，我们进一步将每 $O\left( {q/{q}^{\prime }}\right)$ 个单元格合并为一个包含 $\Theta \left( {{N}_{1}/{q}^{\prime }}\right)$ 个点的更大单元格。现在，每个新合并的单元格具有非常数的描述复杂度，但由于总共只有 $O\left( q\right)$ 个单元格，从这些单元格到新合并单元格的整个映射可以广播到所有服务器。每个服务器仍然可以为其每个点确定哪个新合并的单元格包含它。

Meanwhile,note that if $\widehat{K}$ is an ${N}_{2}$ -thresholded approximation and $\widehat{K} > \frac{\mathrm{{IN}} \cdot  p}{q}$ ,then $\widehat{K}$ must be a constant-factor approximation of $K$ and we have ${q}^{\prime } = \Theta \left( \sqrt{\frac{\mathrm{{IN}} \cdot  {pq}}{K}}\right)$ .

同时，请注意，如果 $\widehat{K}$ 是一个 ${N}_{2}$ -阈值近似且 $\widehat{K} > \frac{\mathrm{{IN}} \cdot  p}{q}$，那么 $\widehat{K}$ 必定是 $K$ 的一个常数因子近似，并且我们有 ${q}^{\prime } = \Theta \left( \sqrt{\frac{\mathrm{{IN}} \cdot  {pq}}{K}}\right)$。

With the new ${q}^{\prime }$ ,step (1) has load $O\left( {{q}^{\prime }\log p}\right)  = O\left( {q\log p}\right)$ ; step (2) has load

使用新的 ${q}^{\prime }$，步骤 (1) 的负载为 $O\left( {{q}^{\prime }\log p}\right)  = O\left( {q\log p}\right)$；步骤 (2) 的负载为

$$
O\left( \frac{\mathrm{{IN}}}{{q}^{\prime }}\right)  = O\left( \sqrt{\frac{\mathrm{{IN}} \cdot  K}{pq}}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right) ,
$$

where the second step uses the fact that $K{N}_{1}/q \leq  K \cdot  \mathrm{{IN}}/q$ is the output size for the fully covered cells in the first attempt of the algorithm, so must be no larger than OUT.

其中第二步利用了这样一个事实：$K{N}_{1}/q \leq  K \cdot  \mathrm{{IN}}/q$ 是算法首次尝试中完全覆盖的单元格的输出大小，因此它一定不大于 OUT。

In the re-execution of the algorithm,let ${F}^{\prime }\left( \Delta \right)$ be the number of halfspaces covering a newly merged cell $\Delta$ ,and let ${K}^{\prime } = \mathop{\sum }\limits_{\Delta }F\left( \Delta \right)$ . Observe that each newly merged cell consists of $\Theta \left( {q/{q}^{\prime }}\right)$ old cells. This means that we have ${K}^{\prime } =$ $O\left( {K{q}^{\prime }/q}\right)$ ,since any halfspace fully covering one newly merged cell must cover $\Theta \left( {q/{q}^{\prime }}\right)$ old cells (but not vice versa).

在算法的重新执行中，设 ${F}^{\prime }\left( \Delta \right)$ 为覆盖新合并单元格 $\Delta$ 的半空间数量，并设 ${K}^{\prime } = \mathop{\sum }\limits_{\Delta }F\left( \Delta \right)$。注意，每个新合并的单元格由 $\Theta \left( {q/{q}^{\prime }}\right)$ 个旧单元格组成。这意味着我们有 ${K}^{\prime } =$ $O\left( {K{q}^{\prime }/q}\right)$，因为任何完全覆盖一个新合并单元格的半空间必定覆盖 $\Theta \left( {q/{q}^{\prime }}\right)$ 个旧单元格（但反之不成立）。

We argue that in the re-execution, we will always have ${\widehat{K}}^{\prime } = O\left( \frac{\mathrm{{IN}} \cdot  p}{{q}^{\prime }}\right)$ ,thus always reaching step (3.2),whose load is $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{{q}^{\prime }}}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ . Indeed,we have

我们认为在重新执行过程中，我们始终会有 ${\widehat{K}}^{\prime } = O\left( \frac{\mathrm{{IN}} \cdot  p}{{q}^{\prime }}\right)$，因此总是会到达步骤 (3.2)，其负载为 $O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \frac{\mathrm{{IN}}}{{q}^{\prime }}}\right)  = O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$。实际上，我们有

$$
{\widehat{K}}^{\prime } = O\left( {{K}^{\prime } + \mathrm{{IN}}}\right) 
$$

$$
\left( {\widehat{K}}^{\prime }\right. \text{is a}{N}_{2}\text{-thresholded approximation of}\left. {K}^{\prime }\right) 
$$

$$
 = O\left( {K \cdot  \frac{{q}^{\prime }}{q} + \mathrm{{IN}}}\right) 
$$

$$
 = O\left( {\frac{\mathrm{{IN}} \cdot  {pq}}{{\left( {q}^{\prime }\right) }^{2}} \cdot  \frac{{q}^{\prime }}{q} + \mathrm{{IN}}}\right)  = O\left( \frac{\mathrm{{IN}} \cdot  p}{{q}^{\prime }}\right) .
$$

Therefore, the re-execution, if it takes place, must have load $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ . Combining with the load of the first execution, we obtain the following result.

因此，如果进行重新执行，其负载必定为 $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$。结合首次执行的负载，我们得到以下结果。

THEOREM 8. There is a randomized algorithm that solves the halfspaces-containing-points problem in $O\left( 1\right)$ rounds and load

定理 8. 存在一种随机算法，该算法能在 $O\left( 1\right)$ 轮内解决半空间包含点问题，且负载为

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{p}} + \mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}} + {p}^{\frac{d}{{2d} - 1}}\log p}\right) .
$$

The algorithm succeeds with probability at least $1 - 1/{p}^{O\left( 1\right) }$ .

该算法成功的概率至少为 $1 - 1/{p}^{O\left( 1\right) }$。

## 6. SIMILARITY JOIN IN HIGH DIMENSIONS

## 6. 高维中的相似性连接

So far we have assumed that the dimensionality $d$ is a constant. The load for both the ${\ell }_{1}/{\ell }_{\infty }$ algorithm and the ${\ell }_{2}$ algorithm hides constant factors that depend on $d$ exponentially in the big-Oh notation. For the ${\ell }_{2}$ algorithm,even for constant $d$ ,the term $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ approaches $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$ as $d$ grows,which is the load for computing the full Cartesian product.

到目前为止，我们假设维度 $d$ 是一个常数。${\ell }_{1}/{\ell }_{\infty }$ 算法和 ${\ell }_{2}$ 算法的负载在大 O 表示法中隐藏了依赖于 $d$ 的指数级常数因子。对于 ${\ell }_{2}$ 算法，即使 $d$ 为常数，当 $d$ 增大时，项 $O\left( {\mathrm{{IN}}/{p}^{\frac{d}{{2d} - 1}}}\right)$ 趋近于 $O\left( {\mathrm{{IN}}/\sqrt{p}}\right)$，这是计算全笛卡尔积的负载。

In this section, we present an algorithm for high-dimensional similarity joins based on locality sensitive hashing (LSH), where $d$ is not considered a constant. The nice thing about the LSH-based algorithm is that its load is independent of $d$ (we still measure the load in terms of tuples; if measured in words,then there will be an extra factor of $d$ ). The downside is that its output-dependent term will not depend on OUT exactly; instead,it will depend on $\operatorname{OUT}\left( {cr}\right)$ ,which is the output size when the distance threshold of the similarity join is made $c$ times larger,for some constant $c > 1$ . LSH is known to be an approximate solution for nearest neighbor search,as it may return a neighbor whose distance is $c$ times larger than the true nearest neighbor. In the case of similarity joins, all answers returned are truly within a distance of $r$ (since this can be easily verified),but its cost will depend on $\operatorname{OUT}\left( {cr}\right)$ instead of OUT. It is also an approximate solution, in the sense that it approximates the optimal cost. The same notion of approximation has also been used for LSH-based similarity joins in the external memory model [25].

在本节中，我们提出一种基于局部敏感哈希（Locality Sensitive Hashing，LSH）的高维相似性连接算法，其中 $d$ 不被视为常数。基于 LSH 的算法的优点在于其负载与 $d$ 无关（我们仍然以元组来衡量负载；如果以字来衡量，则会有一个额外的 $d$ 因子）。缺点是其依赖于输出的项并不精确依赖于 OUT；相反，它将依赖于 $\operatorname{OUT}\left( {cr}\right)$，即当相似性连接的距离阈值增大 $c$ 倍时的输出大小，其中 $c > 1$ 为某个常数。众所周知，LSH 是最近邻搜索的一种近似解决方案，因为它可能返回一个距离是真正最近邻 $c$ 倍的邻居。在相似性连接的情况下，返回的所有答案确实都在距离 $r$ 之内（因为这可以很容易地验证），但其成本将依赖于 $\operatorname{OUT}\left( {cr}\right)$ 而非 OUT。从近似最优成本的意义上来说，它也是一种近似解决方案。在外部内存模型中，基于 LSH 的相似性连接也使用了相同的近似概念 [25]。

Let $\operatorname{dist}\left( {\cdot , \cdot  }\right)$ be a distance function. For $c > 1,{p}_{1} > {p}_{2}$ , recall that a family $\mathcal{H}$ of hash functions is $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ - sensitive,if for any uniformly chosen hash function $h \in  \mathcal{H}$ , and any two tuples $x,y$ ,we have (1) $\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \geq  {p}_{1}$ if $\operatorname{dist}\left( {x,y}\right)  \leq  r$ ; and (2) $\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \leq  {p}_{2}$ if $\operatorname{dist}\left( {x,y}\right)  \geq$ ${cr}$ . In addition,we require $\mathcal{H}$ to be monotone,i.e.,for a randomly chosen $h \in  \mathcal{H},\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack$ is a non-increasing function of $\operatorname{dist}\left( {x,y}\right)$ . This requirement is not in the standard definition of LSH, but the LSH constructions for most metric spaces satisfy this property, include Hamming [19], ${\ell }_{1}\left\lbrack  {12}\right\rbrack  ,{\ell }_{2}\left\lbrack  {5,{12}}\right\rbrack$ ,Jaccard [9],etc.

设$\operatorname{dist}\left( {\cdot , \cdot  }\right)$为一个距离函数。对于$c > 1,{p}_{1} > {p}_{2}$，回顾一下，如果对于任意均匀选取的哈希函数$h \in  \mathcal{H}$以及任意两个元组$x,y$，满足以下条件，则哈希函数族$\mathcal{H}$是$\left( {r,{cr},{p}_{1},{p}_{2}}\right)$敏感的：(1) 若$\operatorname{dist}\left( {x,y}\right)  \leq  r$，则$\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \geq  {p}_{1}$；(2) 若$\operatorname{dist}\left( {x,y}\right)  \geq$${cr}$，则$\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack   \leq  {p}_{2}$。此外，我们要求$\mathcal{H}$是单调的，即对于随机选取的$h \in  \mathcal{H},\Pr \left\lbrack  {h\left( x\right)  = h\left( y\right) }\right\rbrack$，它是$\operatorname{dist}\left( {x,y}\right)$的非增函数。这一要求不在局部敏感哈希（LSH，Locality - Sensitive Hashing）的标准定义中，但大多数度量空间的LSH构造都满足这一性质，包括汉明（Hamming）[19]、${\ell }_{1}\left\lbrack  {12}\right\rbrack  ,{\ell }_{2}\left\lbrack  {5,{12}}\right\rbrack$、杰卡德（Jaccard）[9]等。

The quality of a hash function family is measured by $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}} < 1$ ,which is bounded by a constant that depends only on $c$ ,but not the dimensionality,and $\rho  \approx  1/c$ for many common distance functions $\left\lbrack  {{19},{12},5}\right\rbrack$ . In a standard hash family $\mathcal{H},{p}_{1}$ and ${p}_{2}$ are both constants,but by concatenating multiple hash functions independently chosen from $\mathcal{H}$ , we can make ${p}_{1}$ and ${p}_{2}$ arbitrarily small,while $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$ is kept fixed,or equivalently, ${p}_{2} = {p}_{1}^{1/\rho }$ .

哈希函数族的质量由$\rho  = \frac{\log {p}_{1}}{\log {p}_{2}} < 1$衡量，它受一个仅依赖于$c$而非维度的常数限制，并且对于许多常见的距离函数$\left\lbrack  {{19},{12},5}\right\rbrack$有$\rho  \approx  1/c$。在标准哈希族中，$\mathcal{H},{p}_{1}$和${p}_{2}$都是常数，但通过独立地从$\mathcal{H}$中选取多个哈希函数进行串联，我们可以使${p}_{1}$和${p}_{2}$任意小，同时保持$\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$固定，或者等价地，${p}_{2} = {p}_{1}^{1/\rho }$。

In the description of our algorithm below,we leave ${p}_{1},{p}_{2}$ unspecified, which will be later determined in the analysis.

在下面对我们算法的描述中，我们不指定${p}_{1},{p}_{2}$，它将在后续分析中确定。

The algorithm proceeds in the following 3 steps:

该算法按以下三个步骤进行：

(1) Choose $1/{p}_{1}$ hash functions ${h}_{1},\ldots ,{h}_{1/{p}_{1}} \in  \mathcal{H}$ randomly and independently, and broadcast them to all servers.

(1) 随机且独立地选取$1/{p}_{1}$个哈希函数${h}_{1},\ldots ,{h}_{1/{p}_{1}} \in  \mathcal{H}$，并将它们广播到所有服务器。

(2) For each tuple $x$ ,make $1/{p}_{1}$ copies,and attach the pair $\left( {i,{h}_{i}\left( x\right) }\right)$ to each of these copies,for $i = 1,\ldots ,1/{p}_{1}$ .

(2) 对于每个元组$x$，制作$1/{p}_{1}$份副本，并将对$\left( {i,{h}_{i}\left( x\right) }\right)$附加到这些副本中的每一份上，其中$i = 1,\ldots ,1/{p}_{1}$。

(3) Perform an equi-join on all the copies of tuples, treating the pair $\left( {i,{h}_{i}\left( x\right) }\right)$ as the join value,i.e.,two tuples $x,y$ join if ${h}_{i}\left( x\right)  = {h}_{i}\left( y\right)$ for some $i$ . For two joined tuples $x,y$ ,output them if $\operatorname{dist}\left( {x,y}\right)  \leq  r$ .

(3) 对所有元组副本执行等值连接，将对$\left( {i,{h}_{i}\left( x\right) }\right)$视为连接值，即如果对于某个$i$有${h}_{i}\left( x\right)  = {h}_{i}\left( y\right)$，则两个元组$x,y$进行连接。对于两个连接的元组$x,y$，若$\operatorname{dist}\left( {x,y}\right)  \leq  r$，则输出它们。

THEOREM 9. Assume there is a monotone $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ - sensitive LSH family with $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$ . Then there is a randomized similarity join algorithm that runs in $O\left( 1\right)$ rounds and with expected load

定理9. 假设存在一个单调的 $\left( {r,{cr},{p}_{1},{p}_{2}}\right)$ -敏感局部敏感哈希（LSH）族，其中 $\rho  = \frac{\log {p}_{1}}{\log {p}_{2}}$ 。那么存在一个随机相似度连接算法，该算法在 $O\left( 1\right)$ 轮内运行，且预期负载为

$$
O\left( {\sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}} + \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}} + \frac{\mathrm{{IN}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right) .
$$

The algorithm reports each join result with at least constant probability.

该算法以至少恒定的概率报告每个连接结果。

Proof. Correctness of the algorithm follows from standard LSH analysis: For any two tuples $x,y$ with $\operatorname{dist}\left( {x,y}\right)  \leq$ $r$ ,the probability that they join on one ${h}_{i}$ is at least ${p}_{1}$ . Across $1/{p}_{1}$ independently hash functions,we have constant probability that they join on at least one of them.

证明。该算法的正确性可通过标准的局部敏感哈希（LSH）分析得出：对于任意两个元组 $x,y$ ，若 $\operatorname{dist}\left( {x,y}\right)  \leq$ $r$ ，则它们在一个 ${h}_{i}$ 上进行连接的概率至少为 ${p}_{1}$ 。在 $1/{p}_{1}$ 个独立的哈希函数中，它们至少在其中一个哈希函数上进行连接的概率为常数。

Below we analyze the load. Step (1) has load $O\left( {1/{p}_{1}}\right)$ ; step (2) is local computation. So we only need to analyze step (3).

下面我们分析负载情况。步骤（1）的负载为 $O\left( {1/{p}_{1}}\right)$ ；步骤（2）是局部计算。因此，我们只需分析步骤（3）。

The total number of tuples generated in step (2) is $O\left( {N/{p}_{1}}\right)$ , which is the input size to the equi-join. The expected output size is at most

步骤（2）中生成的元组总数为 $O\left( {N/{p}_{1}}\right)$ ，这也是等值连接的输入规模。预期输出规模至多为

$$
\mathrm{{OUT}}/{p}_{1} + \mathrm{{OUT}}\left( {cr}\right)  + {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - 1/\rho }.
$$

The first term is for all pairs(x,y)such that $\operatorname{dist}\left( {x,y}\right)  \leq  r$ . They could join on every ${h}_{i}$ . The second term is for(x,y)’s such that $r < \operatorname{dist}\left( {x,y}\right)  \leq  {cr}$ . There are $\operatorname{OUT}\left( {cr}\right)$ such pairs, and each pair has probability at most ${p}_{1}$ to join on each ${h}_{i}$ ,so each pair joins exactly once in expectation. The last term is for all(x,y)’s such that $\operatorname{dist}\left( {x,y}\right)  > {cr}$ . There are ${N}^{2}$ such pairs,and each pair joins with probability at most ${p}_{2}$ on each ${h}_{i}$ ,so they contribute the term ${\mathrm{{IN}}}^{2}{p}_{2}/{p}_{1} = {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - \rho }$ in expectation.

第一项针对所有满足 $\operatorname{dist}\left( {x,y}\right)  \leq  r$ 的对 (x, y) 。它们可能在每个 ${h}_{i}$ 上进行连接。第二项针对满足 $r < \operatorname{dist}\left( {x,y}\right)  \leq  {cr}$ 的 (x, y) 对。这样的对有 $\operatorname{OUT}\left( {cr}\right)$ 个，且每对在每个 ${h}_{i}$ 上进行连接的概率至多为 ${p}_{1}$ ，因此每对在期望意义下恰好连接一次。最后一项针对所有满足 $\operatorname{dist}\left( {x,y}\right)  > {cr}$ 的 (x, y) 对。这样的对有 ${N}^{2}$ 个，且每对在每个 ${h}_{i}$ 上进行连接的概率至多为 ${p}_{2}$ ，因此它们在期望意义下贡献项 ${\mathrm{{IN}}}^{2}{p}_{2}/{p}_{1} = {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - \rho }$ 。

Plugging these into Theorem 1, and using Jensen's inequality $E\left\lbrack  \sqrt{X}\right\rbrack   \leq  \sqrt{E\left\lbrack  X\right\rbrack  }$ ,the expected load can be bounded by (the big-Oh of)

将这些代入定理1，并使用詹森不等式 $E\left\lbrack  \sqrt{X}\right\rbrack   \leq  \sqrt{E\left\lbrack  X\right\rbrack  }$ ，预期负载可以被（大O表示法）界定为

$$
\frac{\sqrt{\mathrm{{OUT}}/{p}_{1} + \mathrm{{OUT}}\left( {cr}\right)  + {\mathrm{{IN}}}^{2}/{p}_{1}^{1 - 1/\rho }}}{\sqrt{p}} + \frac{\mathrm{{IN}}}{p{p}_{1}}
$$

$$
 \leq  \sqrt{\frac{\mathrm{{OUT}}}{p{p}_{1}}} + \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}} + \mathrm{{IN}}\sqrt{\frac{1}{p{p}_{1}^{1 - 1/\rho }}} + \frac{\mathrm{{IN}}}{p{p}_{1}}.
$$

Setting ${p}_{1} = 1/{p}^{\frac{\rho }{1 + \rho }}$ balances the last two terms,and we obtain the claimed bound in the theorem.

令 ${p}_{1} = 1/{p}^{\frac{\rho }{1 + \rho }}$ 可平衡最后两项，我们便可得到定理中所声称的界。

Remark. Note that since $0 < \rho  < 1$ ,the input-dependent term is always better than performing a full Cartesian product. The output-term $O\left( \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}}\right)$ is also the best we can achieve for any LSH-based algorithm, by the following intuitive argument: Due to its approximation nature, LSH cannot tell whether the distance between two tuples are smaller than $r$ or slightly above $r$ . A worst-case scenario is all the OUT(cr)pairs of tuples have distance slightly above $r$ but none of them actually joins. Unfortunately,since the hash functions cannot distinguish the two cases, any LSH-based algorithm will have to check all the OUT(cr) pairs to make sure that it does not miss any true join results. Finally,the term $O\left( \sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right)$ is also worse than the bound $O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$ we achieved in earlier sections. This is perhaps the best one can hope for as well,if $O\left( 1\right)$ rounds are required: In order to capture all joining pairs, $1/{p}_{1}$ repetitions are necessary, and two very close tuples may join in all these repetitions,introducing the extra $1/{p}_{1}$ factor in the output size. If we want to perform all of them in parallel, there seems to be no way to eliminate the redundancy beforehand. Of course, this is just an intuitive argument, not a formal proof.

备注。请注意，由于$0 < \rho  < 1$，依赖于输入的项总是比执行完整的笛卡尔积要好。通过以下直观的论证可知，输出项$O\left( \sqrt{\frac{\mathrm{{OUT}}\left( {cr}\right) }{p}}\right)$也是我们对于任何基于局部敏感哈希（LSH，Locality-Sensitive Hashing）的算法所能达到的最佳结果：由于LSH的近似性质，它无法判断两个元组之间的距离是小于$r$还是略大于$r$。最坏的情况是，所有OUT(cr)元组对的距离都略大于$r$，但实际上它们都不进行连接。不幸的是，由于哈希函数无法区分这两种情况，任何基于LSH的算法都必须检查所有的OUT(cr)对，以确保不会遗漏任何真正的连接结果。最后，项$O\left( \sqrt{\frac{\mathrm{{OUT}}}{{p}^{1/\left( {1 + \rho }\right) }}}\right)$也比我们在前面章节中得到的边界$O\left( \sqrt{\frac{\mathrm{{OUT}}}{p}}\right)$要差。如果需要$O\left( 1\right)$轮，这可能也是人们所能期望的最佳结果了：为了捕获所有的连接对，需要$1/{p}_{1}$次重复，并且两个非常接近的元组可能在所有这些重复中都进行连接，从而在输出大小中引入额外的$1/{p}_{1}$因子。如果我们想并行执行所有这些操作，似乎没有办法事先消除冗余。当然，这只是一个直观的论证，并非正式的证明。

### 7.A LOWER BOUND ON 3-RELATION CHAIN JOIN

### 7. 三元关系链连接的下界

In this section, we consider the possibility of designing output-optimal algorithms for multi-way joins. We show that, unfortunately, this is not possible, even for the simplest multi-way join,a 3-relation equi-join, ${R}_{1}\left( {A,B}\right)  \boxtimes  {R}_{2}\left( {B,C}\right)  \boxtimes$ ${R}_{3}\left( {C,D}\right)$ .

在本节中，我们考虑为多路连接设计输出最优算法的可能性。不幸的是，我们发现即使对于最简单的多路连接，即三元关系等值连接${R}_{1}\left( {A,B}\right)  \boxtimes  {R}_{2}\left( {B,C}\right)  \boxtimes$ ${R}_{3}\left( {C,D}\right)$，这也是不可能的。

The first question is how an output-optimal term would look like for a 3-relation join. Applying the tuple-based argument in Section 1.2, each server can potentially produce $O\left( {L}^{3}\right)$ join results in a single round,hence $O\left( {p{L}^{3}}\right)$ results over all $p$ servers in a constant number of round. Thus,an $O\left( {\left( \mathrm{{OUT}}/p\right) }^{1/3}\right)$ term is definitely output-optimal.

第一个问题是，对于三元关系连接，输出最优项会是什么样子。应用1.2节中基于元组的论证，每个服务器在一轮中可能产生$O\left( {L}^{3}\right)$个连接结果，因此在常数轮数内，所有$p$个服务器总共会产生$O\left( {p{L}^{3}}\right)$个结果。因此，一个$O\left( {\left( \mathrm{{OUT}}/p\right) }^{1/3}\right)$项绝对是输出最优的。

<!-- Media -->

<!-- figureText: ${R}_{1}$ ${R}_{3}$ ${R}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195ccc1-025b-792b-90f9-77716d7b8c19_9.jpg?x=322&y=1013&w=374&h=234&r=0"/>

Figure 3: An instance of a 3-relation chain join.

图3：三元关系链连接的一个实例。

<!-- Media -->

However, consider the instance shown in Figure 3, where we use vertices to represent attribute values, and edges for tuples. On such an instance, the 3-relation join degenerates into the Cartesian product of ${R}_{1}$ and ${R}_{3}$ . Each server can produce at most $O\left( {L}^{2}\right)$ pairs of tuples in one round,one from ${R}_{1}$ and one from ${R}_{3}$ ,so we must have $O\left( {p{L}^{2}}\right)  = \Omega \left( \mathrm{{OUT}}\right)$ , or $L = \Omega \left( \sqrt{\mathrm{{OUT}}/p}\right)$ . This means that the best possible output-dependent term is still $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ . Below we show that this is not possible, either, assuming any meaningful input-dependent term.

然而，考虑图3所示的实例，我们用顶点表示属性值，用边表示元组。在这样的实例中，三元关系连接退化为${R}_{1}$和${R}_{3}$的笛卡尔积。每个服务器在一轮中最多可以产生$O\left( {L}^{2}\right)$对元组，一对来自${R}_{1}$，一对来自${R}_{3}$，因此我们必须有$O\left( {p{L}^{2}}\right)  = \Omega \left( \mathrm{{OUT}}\right)$，即$L = \Omega \left( \sqrt{\mathrm{{OUT}}/p}\right)$。这意味着最佳的可能依赖于输出的项仍然是$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$。下面我们将证明，假设存在任何有意义的依赖于输入的项，这也是不可能的。

THEOREM 10. For any tuple-based algorithm computing a 3-relation chain join, if its load is in the form of

定理10。对于任何计算三元关系链连接的基于元组的算法，如果其负载形式为

$$
L = O\left( {\frac{\mathrm{{IN}}}{{p}^{\alpha }} + \sqrt{\frac{\mathrm{{OUT}}}{p}}}\right) ,
$$

for some constant $\alpha$ ,then we must have $\alpha  \leq  1/2$ ,provided $\mathrm{{IN}}/{\log }^{2}\mathrm{{IN}} > c{p}^{3}$ for some sufficiently large constant $c$ .

对于某个常数$\alpha$，那么我们必须有$\alpha  \leq  1/2$，前提是对于某个足够大的常数$c$有$\mathrm{{IN}}/{\log }^{2}\mathrm{{IN}} > c{p}^{3}$。

Note that there is an algorithm for the 3-relation chain join with load $\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right) \left\lbrack  {21}\right\rbrack$ ,without any output-dependent term. This means that it is meaningless to introduce the output-dependent term $O\left( \sqrt{\mathrm{{OUT}}/p}\right)$ .

请注意，存在一种用于三关系链连接的算法，其负载为$\widetilde{O}\left( {\mathrm{{IN}}/\sqrt{p}}\right) \left\lbrack  {21}\right\rbrack$，且不包含任何与输出相关的项。这意味着引入与输出相关的项$O\left( \sqrt{\mathrm{{OUT}}/p}\right)$是没有意义的。

Proof. Suppose there is an algorithm with a claimed load $L$ in the form stated above. We will construct a hard instance on which we must have $\alpha  \leq  1/2$ . Our construction is probabilistic, and we will show that with high probability, the constructed instance satisfies our needs.

证明。假设存在一种算法，其声称的负载为上述形式的$L$。我们将构造一个困难实例，在该实例上我们必然有$\alpha  \leq  1/2$。我们的构造是概率性的，并且我们将证明，以高概率，所构造的实例满足我们的需求。

<!-- Media -->

<!-- figureText: ${R}_{1}$ ${R}_{2}$ ${R}_{3}$ $C$ $D$ $A$ $B$ -->

<img src="https://cdn.noedgeai.com/0195ccc1-025b-792b-90f9-77716d7b8c19_9.jpg?x=1092&y=271&w=382&h=475&r=0"/>

Figure 4: A randomly constructed hard instance.

图4：一个随机构造的困难实例。

<!-- Media -->

The construction is illustrated in Figure 4. More precisely, attributes $B$ and $C$ each have $\frac{N}{\sqrt{L}}$ distinct values. Each distinct value of $B$ appears in $\sqrt{L}$ tuples in ${R}_{1}$ ,and each distinct value in $C$ appears in $\sqrt{L}$ tuples in ${R}_{3}$ . Each distinct value of $B$ and each distinct value of $C$ have a probability of $\frac{L}{N}$ to form a tuple in ${R}_{2}$ . Note that ${R}_{1}$ and ${R}_{3}$ are deterministic and always have $N$ tuples,while ${R}_{2}$ is probabilistic with $N$ tuples in expectation,so $E\left\lbrack  \mathrm{{IN}}\right\rbrack   = {3N}$ . The output size is expected to be $E\left\lbrack  \mathrm{{OUT}}\right\rbrack   = {NL}$ . By the Chernoff inequality, the probability that IN or OUT deviates from their expectations by more than a constant fraction is $\exp \left( {-\Omega \left( N\right) }\right)$ .

该构造如图4所示。更确切地说，属性$B$和$C$各自有$\frac{N}{\sqrt{L}}$个不同的值。$B$的每个不同值在${R}_{1}$中出现在$\sqrt{L}$个元组中，$C$的每个不同值在${R}_{3}$中出现在$\sqrt{L}$个元组中。$B$的每个不同值和$C$的每个不同值有$\frac{L}{N}$的概率在${R}_{2}$中形成一个元组。请注意，${R}_{1}$和${R}_{3}$是确定性的，并且始终有$N$个元组，而${R}_{2}$是概率性的，期望有$N$个元组，因此$E\left\lbrack  \mathrm{{IN}}\right\rbrack   = {3N}$。输出大小预计为$E\left\lbrack  \mathrm{{OUT}}\right\rbrack   = {NL}$。根据切尔诺夫不等式，IN或OUT与其期望值的偏差超过一个常数比例的概率为$\exp \left( {-\Omega \left( N\right) }\right)$。

We allow all servers to access ${R}_{2}$ for free,and only charge the algorithm for receiving tuples from ${R}_{1}$ and ${R}_{3}$ . More precisely, we bound the maximum number of join results a server can produce in a round,if it only receives $L$ tuples from ${R}_{1}$ and $L$ tuples from ${R}_{3}$ . Then we multiply this number by $p$ ,which must be larger than OUT. Note that this is purely a counting argument; if the same join result is produced at two or more servers, it is counted multiple times.

我们允许所有服务器免费访问${R}_{2}$，并且仅对从${R}_{1}$和${R}_{3}$接收元组的算法收费。更准确地说，如果服务器在一轮中仅从${R}_{1}$接收$L$个元组，从${R}_{3}$接收$L$个元组，我们会限制该服务器在一轮中能够产生的连接结果的最大数量。然后我们将这个数量乘以$p$，这个结果必须大于OUT。请注意，这纯粹是一个计数论证；如果同一个连接结果在两个或更多服务器上产生，则会被多次计数。

First we argue that a server should load ${R}_{1}$ and ${R}_{3}$ in whole groups in order to maximize its output size. Here, a group means all tuples sharing the same value on $B$ (or $C$ ). Without loss of generality,suppose two groups in ${R}_{1}$ are not loaded in full by a server,say, ${g}_{1}$ and ${g}_{2}$ . If ${g}_{1}$ joins more tuples from ${R}_{3}$ that have been loaded by the same server than ${g}_{2}$ ,then we can always shift tuples from ${g}_{2}$ to ${g}_{1}$ so as to produce more (at least not less) join results.

首先，我们认为服务器应该以完整的组为单位加载${R}_{1}$和${R}_{3}$，以最大化其输出规模。在这里，一个组是指在$B$（或$C$）上具有相同值的所有元组。不失一般性，假设服务器没有完整加载${R}_{1}$中的两个组，比如${g}_{1}$和${g}_{2}$。如果${g}_{1}$与同一服务器已加载的${R}_{3}$中的元组进行连接的数量比${g}_{2}$多，那么我们总是可以将元组从${g}_{2}$转移到${g}_{1}$，从而产生更多（至少不少于）连接结果。

Thus,each server in each round loads $\sqrt{L}$ groups from ${R}_{1}$ and $\sqrt{L}$ groups from ${R}_{3}$ . Below we show that,on a random instance constructed as above, with high probability, not many pairs of groups can join, no matter which subset of $2\sqrt{L}$ groups are loaded. Consider any subset of $\sqrt{L}$ groups from ${R}_{1}$ and any subset of $\sqrt{L}$ groups from ${R}_{3}$ . There are $L$ possible pairs of groups,and each pair has probability $\frac{L}{N}$ to join,so we expect to see $\frac{{L}^{2}}{N}$ pairs to join. By Chernoff bound,the probability that more than $2\frac{{L}^{2}}{N}$ pairs join is at most $\exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right) }\right)$ . There are $O\left( {\left( N/\sqrt{L}\right) }^{2\sqrt{L}}\right)$ different choices of $\sqrt{L}$ groups from ${R}_{1}$ and $\sqrt{L}$ groups from ${R}_{3}$ .

因此，每一轮中每个服务器从${R}_{1}$加载$\sqrt{L}$个组，从${R}_{3}$加载$\sqrt{L}$个组。下面我们将证明，在如上构造的随机实例中，无论加载$2\sqrt{L}$个组的哪个子集，大概率不会有很多组对能够进行连接。考虑从${R}_{1}$中选取的任意$\sqrt{L}$个组的子集和从${R}_{3}$中选取的任意$\sqrt{L}$个组的子集。总共有$L$种可能的组对，并且每一组对进行连接的概率为$\frac{L}{N}$，因此我们预计会有$\frac{{L}^{2}}{N}$组对进行连接。根据切尔诺夫界（Chernoff bound），超过$2\frac{{L}^{2}}{N}$组对进行连接的概率至多为$\exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right) }\right)$。从${R}_{1}$中选取$\sqrt{L}$个组和从${R}_{3}$中选取$\sqrt{L}$个组，总共有$O\left( {\left( N/\sqrt{L}\right) }^{2\sqrt{L}}\right)$种不同的选择。

So,the probability that one of them yields more than $2\frac{{L}^{2}}{N}$ joining groups is at most

因此，其中一组产生超过$2\frac{{L}^{2}}{N}$个连接组的概率至多为

$$
O\left( {\left( \frac{N}{\sqrt{L}}\right) }^{2\sqrt{L}}\right)  \cdot  \exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right) }\right) 
$$

$$
 = \exp \left( {-\Omega \left( \frac{{L}^{2}}{N}\right)  - O\left( {\sqrt{L} \cdot  \log N}\right) }\right) .
$$

This probability is exponentially small if

如果

$$
\frac{{L}^{2}}{N} > {c}_{1}\sqrt{L} \cdot  \log N
$$

for some sufficiently large constant ${c}_{1}$ . Rearranging,we get

对于某个足够大的常数${c}_{1}$，这个概率呈指数级小。经过整理，我们得到

$$
N\log N < \frac{1}{{c}_{1}} \cdot  {L}^{\frac{3}{2}}.
$$

By Theorem 2,we always have $L = \Omega \left( {N/p}\right)$ ,so this is true as long as

根据定理2，我们始终有$L = \Omega \left( {N/p}\right)$，因此只要

$$
N\log N < \frac{1}{{c}_{2}} \cdot  {\left( \frac{N}{p}\right) }^{\frac{3}{2}},
$$

for some sufficiently large constant ${c}_{2}$ ,or $N/{\log }^{2}N > {c}_{2}{p}^{3}$ .

对于某个足够大的常数${c}_{2}$，或者$N/{\log }^{2}N > {c}_{2}{p}^{3}$，这个结论就成立。

By a union bound, we conclude that with high probability, a randomly constructed instance has $\mathrm{{IN}} = \Theta \left( N\right) ,\mathrm{{OUT}} =$ $\Theta \left( {NL}\right)$ ,and on this instance,no matter which groups are chosen,no more than $\frac{2{L}^{2}}{N}$ pairs of groups can join. Since each pair of joining groups produces $L$ results,the $p$ servers in total produce $O\left( \frac{{L}^{3}p}{N}\right)$ results in a constant number of rounds. So we have

通过联合界（union bound），我们得出结论：大概率情况下，随机构造的实例满足$\mathrm{{IN}} = \Theta \left( N\right) ,\mathrm{{OUT}} =$ $\Theta \left( {NL}\right)$，并且在这个实例上，无论选择哪些组，进行连接的组对数量都不会超过$\frac{2{L}^{2}}{N}$。由于每一组连接的组对会产生$L$个结果，因此$p$个服务器在固定轮数内总共产生$O\left( \frac{{L}^{3}p}{N}\right)$个结果。所以我们有

$$
\frac{{L}^{3}p}{N} = \Omega \left( {NL}\right) 
$$

i.e.,

即

$$
L = \Omega \left( \frac{N}{\sqrt{p}}\right) .
$$

Suppose an algorithm has a load in the form as stated in the theorem,then with $\mathrm{{OUT}} = \Theta \left( {NL}\right)$ ,we have

假设一个算法的负载形式如定理所述，那么当$\mathrm{{OUT}} = \Theta \left( {NL}\right)$时，我们有

$$
\frac{N}{{p}^{\alpha }} + \sqrt{\frac{NL}{p}} = \Omega \left( \frac{N}{\sqrt{p}}\right) .
$$

If $\alpha  > 1/2$ ,we must have

如果$\alpha  > 1/2$，我们必然有

$$
\sqrt{\frac{NL}{p}} = \Omega \left( \frac{N}{\sqrt{p}}\right) ,
$$

or $L = \Omega \left( N\right)$ ,which is an even higher lower bound. Thus we must have $\alpha  \leq  1/2$ .

或者$L = \Omega \left( N\right)$，这是一个更高的下界。因此我们必然有$\alpha  \leq  1/2$。

## 8. CONCLUDING REMARKS

## 8. 总结性评论

Our negative result has ruled out the possibility of having output-optimal algorithms for any join on 3 or more relations. However, there is still hope if we sacrifice the output optimality slightly. For example, what can be done if the output-dependent term is to be $\widetilde{O}\left( \sqrt{\mathrm{{OUT}}/{p}^{1 - \delta }}\right)$ for some small $\delta$ ?

我们的否定结果排除了对3个或更多关系进行任何连接操作时使用输出最优算法的可能性。然而，如果我们稍微牺牲一下输出最优性，仍然有希望。例如，如果输出相关项变为某个小的 $\delta$ 对应的 $\widetilde{O}\left( \sqrt{\mathrm{{OUT}}/{p}^{1 - \delta }}\right)$ ，可以采取什么措施呢？

More broadly, using OUT as a parameter to measure the complexity falls under the realm of parameterized complexity, or beyond-worst-case analysis in general. This type of analyses often yields more insights for problems where worst-case scenarios are rare in practice, such as joins. While OUT is considered the most natural additional parameter to introduce, other possibilities exist, such as assuming that the data follows certain parameterized distributions, or the degree (i.e., maximum number of tuples a tuple can join) is bounded $\left\lbrack  {{10},{24}}\right\rbrack$ ,etc.

更广泛地说，使用OUT作为参数来衡量复杂度属于参数化复杂度的范畴，或者一般来说属于超越最坏情况分析的范畴。对于那些在实际中最坏情况很少出现的问题（如连接操作），这类分析通常能提供更多的见解。虽然OUT被认为是最自然要引入的额外参数，但也存在其他可能性，例如假设数据遵循某些参数化分布，或者度（即一个元组可以连接的最大元组数）有界 $\left\lbrack  {{10},{24}}\right\rbrack$ 等。

## 9. REFERENCES

## 9. 参考文献

[1] F. Afrati, M. Joglekar, C. Ré, S. Salihoglu, and J. D. Ullman. Gym: A multiround join algorithm in mapreduce. arXiv preprint arXiv:1410.4156, 2014.

[2] F. N. Afrati and J. D. Ullman. Optimizing multiway joins in a map-reduce environment. IEEE Transactions on Knowledge and Data Engineering, 23(9):1282-1298, 2011.

[3] P. K. Agarwal, K. Fox, K. Munagala, and A. Nath. Parallel algorithms for constructing range and nearest-neighbor searching data structures. In Proc. ACM Symposium on Principles of Database Systems, 2016.

[4] A. Aggarwal and J. Vitter. The input/output complexity of sorting and related problems. Communications of the ${ACM},{31}\left( 9\right)  : {1116} - {1127},{1988}$ .

[5] A. Andoni and P. Indyk. Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. In Proc. IEEE Symposium on Foundations of Computer Science, 2006.

[6] A. Atserias, M. Grohe, and D. Marx. Size bounds and query plans for relational joins. In Proc. IEEE Symposium on Foundations of Computer Science, pages 739-748, 2008.

[7] P. Beame, P. Koutris, and D. Suciu. Communication steps for parallel query processing. In Proc. ACM Symposium on Principles of Database Systems, 2013.

[8] P. Beame, P. Koutris, and D. Suciu. Skew in parallel query processing. In Proc. ACM Symposium on Principles of Database Systems, 2014.

[9] A. Z. Broder, S. C. Glassman, M. S. Manasse, and G. Zweig. Syntactic clustering of the web. Computer Networks, 29(8-13):1157-1166, 1997.

[10] Y. Cao, W. Fan, T. Wo, and W. Yu. Bounded conjunctive queries. In Proc. International Conference on Very Large Data Bases, 2014.

[11] T. M. Chan. Optimal partition trees. Discrete & Computational Geometry, 47(4):661-690, 2012.

[12] M. Datar, N. Immorlica, P. Indyk, and V. S. Mirrokni. Locality sensitive hashing scheme based on p-stable distributions. In Proc. Annual Symposium on Computational Geometry, 2004.

[13] M. De Berg, M. Van Kreveld, M. Overmars, and O. C. Schwarzkopf. Computational geometry. In Computational geometry, pages 1-17. Springer, 2000.

[14] J. Dean and S. Ghemawat. MapReduce: Simplified data processing on large clusters. In Proc. Symposium on Operating Systems Design and Implementation, 2004.

[15] M. T. Goodrich. Communication-efficient parallel sorting. SIAM Journal on Computing, 29(2):416-432, 1999.

[16] M. T. Goodrich, N. Sitchinava, and Q. Zhang. Sorting, searching and simulation in the mapreduce framework. In Proc. International Symposium on Algorithms and Computation, 2011.

[17] S. Har-Peled and M. Sharir. Relative (p, $\varepsilon$ )-approximations in geometry. Discrete $\mathcal{C}$ Computational Geometry, 45(3):462-496, 2011.

[17] S. Har - Peled和M. Sharir。几何中的相对 (p, $\varepsilon$ ) - 近似。《离散 $\mathcal{C}$ 计算几何》，45(3):462 - 496，2011年。

[18] X. Hu, P. Koutris, and K. Yi. The relationships among coarse-grained parallel models. Technical report, HKUST, 2016.

[18] X. Hu、P. Koutris和K. Yi。粗粒度并行模型之间的关系。技术报告，香港科技大学，2016年。

[19] P. Indyk and R. Motwani. Approximate nearest neighbors: Towards removing the curse of dimensionality. In Proc. ACM Symposium on Theory of Computing, 1998.

[19] P. Indyk和R. Motwani。近似最近邻：消除维度灾难。收录于《ACM计算理论研讨会论文集》，1998年。

[20] B. Ketsman and D. Suciu. A worst-case optimal multi-round algorithm for parallel computation of conjunctive queries. In Proc. ACM Symposium on Principles of Database Systems, 2017.

[20] B. Ketsman和D. Suciu。用于并行计算合取查询的最坏情况最优多轮算法。收录于《ACM数据库系统原理研讨会论文集》，2017年。

[21] P. Koutris, P. Beame, and D. Suciu. Worst-case optimal algorithms for parallel query processing. In Proc. International Conference on Database Theory, 2016.

[21] P. Koutris、P. Beame和D. Suciu。用于并行查询处理的最坏情况最优算法。收录于《国际数据库理论会议论文集》，2016年。

[22] P. Koutris and D. Suciu. Parallel evaluation of conjunctive queries. In Proc. ACM Symposium on Principles of Database Systems, 2011.

[22] P. Koutris和D. Suciu。合取查询的并行评估。收录于《ACM数据库系统原理研讨会论文集》，2011年。

[23] Y. Li, P. M. Long, and A. Srinivasan. Improved bounds on the sample complexity of learning. Journal of Computer and System Sciences, 62(3):516-527, 2001.

[23] Y. Li、P. M. Long和A. Srinivasan。学习样本复杂度的改进边界。《计算机与系统科学杂志》，62(3):516 - 527，2001年。

[24] C. R. Manas Joglekar. It's all a matter of degree: Using degree information to optimize multiway joins. In Proc. International Conference on Database Theory, 2016.

[24] C. R. Manas Joglekar。一切都与度有关：利用度信息优化多路连接。收录于《国际数据库理论会议论文集》，2016年。

[25] R. Pagh, N. Pham, F. Silvestri, and M. Stöckel. I/O-efficient similarity join. In Proc. European Symposium on Algorithms, 2015.

[25] R. Pagh、N. Pham、F. Silvestri和M. Stöckel。I/O高效的相似性连接。收录于《欧洲算法研讨会论文集》，2015年。

[26] R. Pagh and F. Silvestri. The input/output complexity of triangle enumeration. In Proc. ACM Symposium on Principles of Database Systems, 2014.

[26] R. Pagh和F. Silvestri。三角形枚举的输入/输出复杂度。收录于《ACM数据库系统原理研讨会论文集》，2014年。

[27] M. Pătraşcu. Unifying the landscape of cell-probe lower bounds. SIAM Journal on Computing, 40(3), 2011.

[27] M. Pătraşcu。统一单元探测下界的格局。《SIAM计算杂志》，40(3)，2011年。

[28] L. G. Valiant. A bridging model for parallel computation. Communications of the ${ACM}$ , 33(8):103-111, 1990.

[28] L. G. Valiant。并行计算的桥接模型。《${ACM}$ 通讯》，33(8):103 - 111，1990年。

[29] M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauley, M. J. Franklin, S. Shenker, and I. Stoica. Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing. In Proc. USENIX conference on Networked Systems Design and Implementation, 2012.

[29] M. 扎哈里亚（M. Zaharia）、M. 乔杜里（M. Chowdhury）、T. 达斯（T. Das）、A. 戴夫（A. Dave）、J. 马（J. Ma）、M. 麦考利（M. McCauley）、M. J. 富兰克林（M. J. Franklin）、S. 申克（S. Shenker）和 I. 斯托伊卡（I. Stoica）。弹性分布式数据集：内存集群计算的容错抽象。见《USENIX 网络系统设计与实现会议论文集》，2012 年。