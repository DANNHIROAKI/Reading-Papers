# Two-Attribute Skew Free, Isolated CP Theorem, and Massively Parallel Joins

# 双属性无倾斜、孤立笛卡尔积定理与大规模并行连接

Miao Qiao

乔苗

University of Auckland

奥克兰大学

Auckland, New Zealand

新西兰，奥克兰

miao.qiao@auckland.ac.nz

Yufei Tao

陶宇飞

Chinese University of Hong Kong

香港中文大学

Hong Kong, China

中国，香港

taoyf@cse.cuhk.edu.hk

## Abstract

## 摘要

This paper presents an algorithm to process a multi-way join with load $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ under the MPC model,where $n$ is the number of tuples in the input relations, $\alpha$ the maximum arity of those relations, $p$ the number of machines,and $\phi$ a newly introduced parameter called the generalized vertex packing number. The algorithm owes to two new findings. The first is a two-attribute skew free technique to partition the join result for parallel computation. The second is an isolated cartesian product theorem, which provides fresh graph-theoretic insights on joins with $\alpha  \geq  3$ and generalizes an existing theorem on $\alpha  = 2$ .

本文提出了一种在大规模并行计算（MPC）模型下处理负载为 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 的多路连接的算法，其中 $n$ 是输入关系中的元组数量，$\alpha$ 是这些关系的最大元数，$p$ 是机器数量，$\phi$ 是一个新引入的参数，称为广义顶点填充数。该算法基于两项新发现。第一项是一种双属性无倾斜技术，用于对连接结果进行分区以实现并行计算。第二项是一个孤立笛卡尔积定理，它为具有 $\alpha  \geq  3$ 的连接提供了全新的图论见解，并推广了现有的关于 $\alpha  = 2$ 的定理。

## CCS CONCEPTS

## 计算机协会分类系统（CCS）概念

- Information systems $\rightarrow$ Join algorithms; - Theory of computation $\rightarrow$ Massively parallel algorithms.

- 信息系统 $\rightarrow$ 连接算法； - 计算理论 $\rightarrow$ 大规模并行算法。

## KEYWORDS

## 关键词

Joins; Conjunctive Queries; MPC Algorithms; Parallel Computing

连接；合取查询；大规模并行计算（MPC）算法；并行计算

## ACM Reference Format:

## 美国计算机协会（ACM）引用格式：

Miao Qiao and Yufei Tao. 2021. Two-Attribute Skew Free, Isolated CP Theorem, and Massively Parallel Joins. In Proceedings of the 40th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems (PODS '21), June 20-25, 2021, Virtual Event, China. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3452021.3458321

乔苗和陶宇飞。2021 年。双属性无倾斜、孤立笛卡尔积定理与大规模并行连接。见第 40 届美国计算机协会数据管理专业组（SIGMOD）-计算理论专业组（SIGACT）-人工智能专业组（SIGAI）数据库系统原理研讨会（PODS '21）论文集，2021 年 6 月 20 - 25 日，中国线上会议。美国计算机协会，美国纽约州纽约市，共 15 页。https://doi.org/10.1145/3452021.3458321

## 1 INTRODUCTION

## 1 引言

Massively-parallel computation systems such as Hadoop [7] and Spark [1] are designed to leverage hundreds or even thousands of machines to accomplish a computation task on data of a gigantic volume. Performance bottleneck these systems is communication rather than CPU computation. Understanding the communication complexities of fundamental database problems has become an active research area $\left\lbrack  {2,3,5,6,{10} - {12},{14},{15},{20}}\right\rbrack$ .

像Hadoop [7] 和Spark [1] 这样的大规模并行计算系统旨在利用数百甚至数千台机器来完成对海量数据的计算任务。这些系统的性能瓶颈在于通信而非CPU计算。理解基本数据库问题的通信复杂性已成为一个活跃的研究领域 $\left\lbrack  {2,3,5,6,{10} - {12},{14},{15},{20}}\right\rbrack$ 。

### 1.1 Problem Definition

### 1.1 问题定义

This paper studies parallel algorithms for processing natural joins.

本文研究处理自然连接的并行算法。

Joins. Denote by att a countably infinite set where each element is an attribute. We will assume a total order on att,and use $A \prec  B$ to represent the fact "attribute $A$ ranking before attribute $B$ ". Denote by dom an infinite set where each element is a value.

连接。用att表示一个可数无限集，其中每个元素都是一个属性。我们假设att上存在一个全序关系，并使用 $A \prec  B$ 来表示“属性 $A$ 排在属性 $B$ 之前”这一事实。用dom表示一个无限集，其中每个元素都是一个值。

A tuple over a set $\mathcal{U} \subseteq$ att is a function $\mathbf{u} : \mathcal{U} \rightarrow$ dom. Alternatively,we may represent a tuple as $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{\left| \mathcal{U}\right| }}\right)$ where ${a}_{i}$ is the output of $\mathbf{u}$ on the $i$ -th $\left( {1 \leq  i \leq  \left| \mathcal{U}\right| }\right)$ smallest attribute in $\mathcal{U}$ (according to <). Given a non-empty $\mathcal{V} \subseteq  \mathcal{U}$ ,define $\mathbf{u}\left\lbrack  \mathcal{V}\right\rbrack$ as the tuple $\mathbf{v}$ over $\mathcal{V}$ such that $\mathbf{v}\left( X\right)  = \mathbf{u}\left( X\right)$ for every $X \in  \mathcal{V}$ ; we say that $\mathbf{v}$ is a projection of $\mathbf{u}$ .

集合 $\mathcal{U} \subseteq$ att上的一个元组是一个函数 $\mathbf{u} : \mathcal{U} \rightarrow$ dom。或者，我们可以将一个元组表示为 $\left( {{a}_{1},{a}_{2},\ldots ,{a}_{\left| \mathcal{U}\right| }}\right)$ ，其中 ${a}_{i}$ 是 $\mathbf{u}$ 在 $\mathcal{U}$ 中第 $i$ 个 $\left( {1 \leq  i \leq  \left| \mathcal{U}\right| }\right)$ 最小属性（根据<）上的输出。给定一个非空的 $\mathcal{V} \subseteq  \mathcal{U}$ ，将 $\mathbf{u}\left\lbrack  \mathcal{V}\right\rbrack$ 定义为 $\mathcal{V}$ 上的元组 $\mathbf{v}$ ，使得对于每个 $X \in  \mathcal{V}$ 都有 $\mathbf{v}\left( X\right)  = \mathbf{u}\left( X\right)$ ；我们称 $\mathbf{v}$ 是 $\mathbf{u}$ 的一个投影。

A relation is a set $\mathcal{R}$ of tuples over the same set $\mathcal{U}$ of attributes. $\mathcal{U}$ is the scheme of $\mathcal{R}$ ,denoted as $\operatorname{scheme}\left( \mathcal{R}\right)  = \mathcal{U}$ . Define $\operatorname{arity}\left( \mathcal{R}\right)  =$ scheme $\left( \mathcal{R}\right)  \mid$ ,referred to as the arity of $\mathcal{R}.\mathcal{R}$ is unary if arity $\left( \mathcal{R}\right)  = 1$ , or binary if arity $\left( \mathcal{R}\right)  = 2$ .

一个关系是一个元组集合 $\mathcal{R}$ ，这些元组基于相同的属性集合 $\mathcal{U}$ 。 $\mathcal{U}$ 是 $\mathcal{R}$ 的模式，记为 $\operatorname{scheme}\left( \mathcal{R}\right)  = \mathcal{U}$ 。定义 $\operatorname{arity}\left( \mathcal{R}\right)  =$ 模式 $\left( \mathcal{R}\right)  \mid$ ，称为 $\mathcal{R}.\mathcal{R}$ 的元数。如果元数 $\left( \mathcal{R}\right)  = 1$ ，则 $\mathcal{R}.\mathcal{R}$ 是一元的；如果元数 $\left( \mathcal{R}\right)  = 2$ ，则 $\mathcal{R}.\mathcal{R}$ 是二元的。

A join query is defined as a set $\mathcal{Q}$ of relations. The query result Join( $\mathcal{Q}$ ) is the relation:

一个连接查询被定义为一个关系集合 $\mathcal{Q}$ 。查询结果Join( $\mathcal{Q}$ )是这样一个关系：

$$
\{ \text{tuple}\mathbf{u}\text{over attset}\left( \mathcal{Q}\right)  \mid  \forall \mathcal{R} \in  \mathcal{Q},\mathbf{u}\left\lbrack  {\text{scheme}\left( \mathcal{R}\right) }\right\rbrack   \in  \mathcal{R}\} 
$$

where attset $\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{\mathcal{R} \in  \mathcal{Q}}}$ scheme(R). If $\mathcal{Q}$ includes only two relations $\mathcal{R}$ and $\mathcal{S}$ ,we may also represent $\mathcal{J}$ oin(Q)as $\mathcal{R} \bowtie  \mathcal{S}$ . Define

其中attset $\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{{\mathcal{R} \in  \mathcal{Q}}}$ 模式(R)。如果 $\mathcal{Q}$ 仅包含两个关系 $\mathcal{R}$ 和 $\mathcal{S}$ ，我们也可以将 $\mathcal{J}$ oin(Q)表示为 $\mathcal{R} \bowtie  \mathcal{S}$ 。定义

$$
n = \mathop{\sum }\limits_{{\mathcal{R} \in  \mathcal{Q}}}\left| \mathcal{R}\right| 
$$

$$
k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|  \tag{1}
$$

$$
\alpha  = \mathop{\max }\limits_{{\mathcal{R} \in  \mathcal{Q}}}\operatorname{arity}\left( \mathcal{R}\right) \text{.} \tag{2}
$$

In particular, $n$ is the input size of2. We will treat $\left| \mathcal{Q}\right|$ and $k$ (hence, $\alpha$ ) as constants,and assume $\alpha  \geq  2$ (a query with $\alpha  = 1$ has been optimally solved; see Section 3).

特别地， $n$ 是2的输入规模。我们将把 $\left| \mathcal{Q}\right|$ 和 $k$ （因此， $\alpha$ ）视为常数，并假设 $\alpha  \geq  2$ （一个具有 $\alpha  = 1$ 的查询已得到最优解决；见第3节）。

Computation model. We will work under the massively parallel computation (MPC) model, which has become a standard model for studying parallel join algorithms nowadays [6, 8, 11, 12, 14, 20]. At the beginning,the input relations of $\mathcal{Q}$ are distributed onto $p$ machines,each of which stores $O\left( {n/p}\right)$ tuples. An algorithm can perform only a constant number of rounds, each with two phases:

计算模型。我们将在大规模并行计算（MPC）模型下开展工作，该模型如今已成为研究并行连接算法的标准模型 [6, 8, 11, 12, 14, 20]。一开始，$\mathcal{Q}$ 的输入关系被分布到 $p$ 台机器上，每台机器存储 $O\left( {n/p}\right)$ 个元组。算法只能执行固定轮数，每一轮包含两个阶段：

- Phase 1: Each machine performs local computation, and prepares the messages to be sent to other machines.

- 阶段 1：每台机器进行本地计算，并准备要发送给其他机器的消息。

- Phase 2: The machines exchange messages (which must be prepared in Phase 1).

- 阶段 2：机器之间交换消息（这些消息必须在阶段 1 中准备好）。

Each tuple in $\mathcal{J}$ oin(Q)must reside on at least one machine when the algorithm terminates. The load of a round is the maximum number of words received by a machine in that round. The load of the algorithm is the maximum load of all rounds. The main challenge in algorithm design is to minimize the load.

当算法终止时，$\mathcal{J}$ oin(Q) 中的每个元组必须至少驻留在一台机器上。一轮的负载是该轮中一台机器接收到的最大单词数。算法的负载是所有轮次的最大负载。算法设计的主要挑战是最小化负载。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.PODS '21, June 20-25, 2021, Virtual Event, China © 2021 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-8381-3/21/06...\$15.00 https://doi.org/10.1145/3452021.3458321

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业利益，并且拷贝必须带有此声明和第一页的完整引用。必须尊重非作者拥有的本作品组件的版权。允许进行带引用的摘要。否则，如需复制、重新发布、发布到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向 permissions@acm.org 请求许可。PODS '21，2021 年 6 月 20 - 25 日，虚拟活动，中国 © 2021 版权归所有者/作者所有。出版权授权给 ACM。ACM 国际标准书号 978 - 1 - 4503 - 8381 - 3/21/06... 15.00 美元 https://doi.org/10.1145/3452021.3458321

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td>load</td><td>source</td><td>remark</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\left| \mathcal{Q}\right| }}}\right)$</td><td>[3]</td><td>the hyper-cube (HC) algorithm</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{k}}}\right)$</td><td>[6]</td><td>the BinHC algorithm</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\psi }}}\right)$</td><td>[14]</td><td>the KBS algorithm; $\psi  =$ the edge quasi-packing number (Appendix H)</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\rho }}}\right)$</td><td>$\left\lbrack  {{12},{20}}\right\rbrack$</td><td>$\rho  =$ the fractional edge-covering number (Section 3.1); the algorithm is applicable only to $\alpha  = 2$</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\rho }}}\right)$</td><td>[8]</td><td>for acyclic queries</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{2}{\alpha \phi }}}\right)$</td><td>ours</td><td>$\phi  =$ the generalized vertex-packing number (Section 4); the algorithm subsumes [12,20] when $\alpha  = 2$</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{2}{{\alpha \phi } - \alpha  + 2}}}\right)$</td><td>ours</td><td>for $\alpha$ -uniform queries</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{2}{k - \alpha  + 2}}}\right)$</td><td>ours</td><td>for symmetric queries</td></tr></table>

<table><tbody><tr><td>加载</td><td>源</td><td>备注</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\left| \mathcal{Q}\right| }}}\right)$</td><td>[3]</td><td>超立方体（HC）算法</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{k}}}\right)$</td><td>[6]</td><td>BinHC算法</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\psi }}}\right)$</td><td>[14]</td><td>KBS算法；$\psi  =$ 边拟填充数（附录H）</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\rho }}}\right)$</td><td>$\left\lbrack  {{12},{20}}\right\rbrack$</td><td>$\rho  =$ 分数边覆盖数（第3.1节）；该算法仅适用于$\alpha  = 2$</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{1}{\rho }}}\right)$</td><td>[8]</td><td>对于无环查询</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{2}{\alpha \phi }}}\right)$</td><td>我们的</td><td>$\phi  =$ 广义顶点填充数（第4节）；当$\alpha  = 2$时，该算法包含文献[12,20]的内容</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{2}{{\alpha \phi } - \alpha  + 2}}}\right)$</td><td>我们的</td><td>对于$\alpha$ -一致查询</td></tr><tr><td>$\widetilde{O}\left( {n/{p}^{\frac{2}{k - \alpha  + 2}}}\right)$</td><td>我们的</td><td>对于对称查询</td></tr></tbody></table>

Table 1: Comparison of all the known generic algorithms

表1：所有已知通用算法的比较

<!-- Media -->

We assume $p \leq  \sqrt{n}$ . Unless otherwise stated,by "an algorithm having load at most $L$ ",we mean that its load is at most $L$ with probability at least $1 - 1/{p}^{c}$ where $c > 0$ can be set to an arbitrarily large constant. Each value in $\mathbf{{dom}}$ is assumed to fit in a word. Notation $\widetilde{O}\left( \text{.}\right) {hidesafafaraprolylogarithmictop}.{Givenaninteger}$ $x \geq  1,\left\lbrack  x\right\rbrack$ denotes the set $\{ 1,\ldots ,x\}$ .

我们假设$p \leq  \sqrt{n}$ 。除非另有说明，“一个算法的负载至多为$L$ ”指的是其负载至多为$L$ 的概率至少为$1 - 1/{p}^{c}$ ，其中$c > 0$ 可以设为任意大的常数。假设$\mathbf{{dom}}$ 中的每个值都能放入一个字中。符号$\widetilde{O}\left( \text{.}\right) {hidesafafaraprolylogarithmictop}.{Givenaninteger}$ $x \geq  1,\left\lbrack  x\right\rbrack$ 表示集合$\{ 1,\ldots ,x\}$ 。

### 1.2 Previous Work

### 1.2 相关工作

Afrati and Ullman [3] developed the hyper-cube (HC) algorithm that answers a query $\mathcal{Q}$ with load $O\left( {n/{p}^{1/\left| \mathcal{Q}\right| }}\right)$ deterministically. Extending HC with random binning ideas, Beame, Koutris, and Suciu [5] obtained the ${BinHC}$ algorithm with load $\widetilde{O}\left( {n/{p}^{1/k}}\right)$ .

阿夫拉蒂（Afrati）和厄尔曼（Ullman）[3] 开发了超立方体（HC）算法，该算法能确定性地以负载$O\left( {n/{p}^{1/\left| \mathcal{Q}\right| }}\right)$ 回答查询$\mathcal{Q}$ 。比姆（Beame）、库特里斯（Koutris）和苏丘（Suciu）[5] 通过将随机分箱思想扩展到HC算法，得到了负载为$\widetilde{O}\left( {n/{p}^{1/k}}\right)$ 的${BinHC}$ 算法。

Improving their earlier work [6], Koutris, Beame, and Suciu [14] gave an algorithm - called ${KBS}$ henceforth - with load $\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$ , where $\psi$ is the edge quasi-packing number of $\mathcal{Q}$ (Appendix H).

库特里斯（Koutris）、比姆（Beame）和苏丘（Suciu）[14] 在改进他们早期工作[6] 的基础上，给出了一个算法（此后称为${KBS}$ ），其负载为$\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$ ，其中$\psi$ 是$\mathcal{Q}$ 的边拟填充数（附录H）。

There has been work dedicated to queries involving only relations with arity at most 2 , due to their special importance in subgraph enumeration ${}^{1}$ . Ketsman and Suciu [12] were the first to solve such queries $\mathcal{Q}$ with load $\widetilde{O}\left( {n/{p}^{1/\rho }}\right)$ ,where $\rho$ is the fractional edge covering number of $\mathcal{Q}$ (Section 3.1). A simpler algorithm with the same load was presented by Tao [20].

由于涉及元数至多为2的关系的查询在子图枚举${}^{1}$ 中具有特殊重要性，因此有相关工作专门针对此类查询。凯茨曼（Ketsman）和苏丘（Suciu）[12] 首次以负载$\widetilde{O}\left( {n/{p}^{1/\rho }}\right)$ 解决了此类查询$\mathcal{Q}$ ，其中$\rho$ 是$\mathcal{Q}$ 的分数边覆盖数（3.1节）。陶（Tao）[20] 提出了一个具有相同负载的更简单算法。

Recently, Hu [8] presented an algorithm for answering any acyclic query ${}^{2}$ with load $\widetilde{O}\left( {n/{p}^{1/\rho }}\right)$ .

最近，胡（Hu）[8] 提出了一种以负载$\widetilde{O}\left( {n/{p}^{1/\rho }}\right)$ 回答任何无环查询${}^{2}$ 的算法。

The above algorithms are generic because they support either arbitrary queries $\left\lbrack  {3,{14}}\right\rbrack$ or queries in a broad class $\left\lbrack  {8,{12},{20}}\right\rbrack$ . There are algorithms designed for specific joins, e.g., star joins [3], cycle joins [14], clique joins [14], line joins [3, 14], Loomis-Whitney joins [14], etc.

上述算法是通用算法，因为它们支持任意查询$\left\lbrack  {3,{14}}\right\rbrack$ 或某一广泛类别的查询$\left\lbrack  {8,{12},{20}}\right\rbrack$ 。也有针对特定连接设计的算法，例如星型连接[3] 、循环连接[14] 、团连接[14] 、线性连接[3, 14] 、卢米斯 - 惠特尼连接[14] 等。

We refer the reader to $\left\lbrack  {2,{10},{11}}\right\rbrack$ for algorithms that can achieve a load sensitive to the output size $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ .

关于能够实现对输出规模$\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ 敏感的负载的算法，读者可参考$\left\lbrack  {2,{10},{11}}\right\rbrack$ 。

On the lower bound side, Atserias, Grohe, and Marx [4] showed that $\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$ can reach $\Omega \left( {n}^{\rho }\right)$ but is always bounded by $O\left( {n}^{\rho }\right)$ . This implies [14] that any algorithm must incur a load of $\Omega \left( {n/{p}^{1/\rho }}\right)$ in the worst case. Furthermore, $\mathrm{{Hu}}\left\lbrack  8\right\rbrack$ proved that $\Omega \left( {n/{p}^{1/\tau }}\right)$ is another worst-case lower bound,where $\tau$ is the query’s factional edge packing number (Section 3.1); she also described a class of queries whose $\tau$ values are strictly larger than $\rho$ . It is clear from the above discussion that the upper and lower bounds have matched for (i) queries with $\alpha  = 2$ and (ii) acyclic queries. Matching the lower bounds for an arbitrary query is still open.

在下界方面，阿塞里亚斯（Atserias）、格罗赫（Grohe）和马克思（Marx）[4]表明，$\left| {\mathcal{J}\operatorname{oin}\left( \mathcal{Q}\right) }\right|$可以达到$\Omega \left( {n}^{\rho }\right)$，但始终受限于$O\left( {n}^{\rho }\right)$。这意味着[14]在最坏情况下，任何算法都必须承受$\Omega \left( {n/{p}^{1/\rho }}\right)$的负载。此外，$\mathrm{{Hu}}\left\lbrack  8\right\rbrack$证明了$\Omega \left( {n/{p}^{1/\tau }}\right)$是另一个最坏情况下的下界，其中$\tau$是查询的分数边填充数（第3.1节）；她还描述了一类查询，其$\tau$值严格大于$\rho$。从上述讨论可以清楚地看出，对于（i）具有$\alpha  = 2$的查询和（ii）无环查询，上界和下界已经匹配。对于任意查询匹配下界仍然是一个未解决的问题。

The lower bounds mentioned earlier apply to algorithms that perform an arbitrary (constant) number of rounds. In [14], Koutris, Beame,and Suciu showed that $\Omega \left( {n/{p}^{1/\psi }}\right)$ is a lower bound for single-round algorithms, subject to certain constraints.

前面提到的下界适用于执行任意（常数）轮数的算法。在文献[14]中，库特里斯（Koutris）、比姆（Beame）和苏丘（Suciu）表明，在某些约束条件下，$\Omega \left( {n/{p}^{1/\psi }}\right)$是单轮算法的下界。

Regarding other computational models, the (natural join) problem has been optimally settled in RAM [16, 17, 21], while external-memory (EM; a.k.a. I/O-efficient) algorithms have been developed for specific joins (see $\left\lbrack  {9,{10},{18}}\right\rbrack$ and the references therein). There exists a reduction [14] for converting an MPC algorithm to work in the EM model. The reduction also applies to the algorithms developed in this paper.

关于其他计算模型，（自然连接）问题在随机存取存储器（RAM）中已得到最优解决[16, 17, 21]，而已为特定连接开发了外部存储器（EM；也称为I/O高效）算法（见$\left\lbrack  {9,{10},{18}}\right\rbrack$及其参考文献）。存在一种将大规模并行计算（MPC）算法转换为在外部存储器模型中工作的归约方法[14]。这种归约方法也适用于本文开发的算法。

### 1.3 Our Results

### 1.3 我们的结果

We will describe an algorithm (Theorem 8.2) that answers an arbitrary query $\mathcal{Q}$ with load

我们将描述一种算法（定理8.2），该算法以负载回答任意查询$\mathcal{Q}$

$$
\widetilde{O}\left( {n/{p}^{\frac{2}{\alpha \phi }}}\right)  \tag{3}
$$

where $\alpha  \geq  2$ is the maximum arity and $\phi$ is the generalized vertex packing number of $\mathcal{Q}$ that will be formally introduced in Section 4. For $\alpha  = 2,\phi$ can be proven equal to $\rho$ ; thus,our load matches the lower bound $\Omega \left( {n/{p}^{1/\rho }}\right)$ . For $\alpha  \geq  3$ ,the algorithm improves the previous work on certain (non-trivial) query classes.

其中$\alpha  \geq  2$是最大元数，$\phi$是$\mathcal{Q}$的广义顶点填充数，将在第4节正式引入。对于$\alpha  = 2,\phi$，可以证明其等于$\rho$；因此，我们的负载与下界$\Omega \left( {n/{p}^{1/\rho }}\right)$相匹配。对于$\alpha  \geq  3$，该算法在某些（非平凡）查询类上改进了先前的工作。

One notable class is the $k$ -choose- $\alpha$ join $\mathcal{Q}$ ,which has $\left( \begin{array}{l} k \\  \alpha  \end{array}\right)$ relations (recall that $k$ is the number of attributes),each having a scheme that is a unique combination of $\alpha$ attributes. Currently,the best solution to a $k$ -choose- $\alpha$ join with $\alpha  \in  {\left\lbrack  3,k - 2\right\rbrack  }^{3}$ is the KBS algorithm [14],which (as mentioned) requires a load of $\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$ , with the edge quasi-packing number $\psi$ at least $k - \alpha  + 1$ (Appendix H). On the other hand,2has a generalized vertex packing number $\phi  = k/\alpha$ (Section 4). The load of our algorithm is $\widetilde{O}\left( {n/{p}^{2/k}}\right)$ ,and is already better than $\widetilde{O}\left( {n/{p}^{1/\left( {k - \alpha  + 1}\right) }}\right)$ for $\alpha  < k/2 + 1$ .

一个值得注意的类是$k$选$\alpha$连接$\mathcal{Q}$，它有$\left( \begin{array}{l} k \\  \alpha  \end{array}\right)$个关系（回想一下，$k$是属性的数量），每个关系的模式是$\alpha$个属性的唯一组合。目前，对于具有$\alpha  \in  {\left\lbrack  3,k - 2\right\rbrack  }^{3}$的$k$选$\alpha$连接，最佳解决方案是KBS算法[14]，如前所述，该算法需要$\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$的负载，边准填充数$\psi$至少为$k - \alpha  + 1$（附录H）。另一方面，2具有广义顶点填充数$\phi  = k/\alpha$（第4节）。我们算法的负载是$\widetilde{O}\left( {n/{p}^{2/k}}\right)$，并且对于$\alpha  < k/2 + 1$已经优于$\widetilde{O}\left( {n/{p}^{1/\left( {k - \alpha  + 1}\right) }}\right)$。

---

<!-- Footnote -->

${}^{1}$ Namely,find all the occurrences of a subgraph pattern in a graph.

${}^{1}$即，找出图中某个子图模式的所有出现位置。

${}^{2}$ Specifically,alpha-acyclic queries,which generalize berge-acylic and $r$ -hierarchical queries.

${}^{2}$具体来说，α - 无环查询，它推广了贝尔热（Berge）无环和$r$ - 分层查询。

${}^{3}$ With $\alpha  = k - 1,\mathcal{2}$ is the Loomis-Whitney join and has been solved optimally. The case $\alpha  \leq  2$ is left out because,in general,all queries with $\alpha  \leq  2$ have been settled. The case $\alpha  = k$ is not interesting either because $\mathcal{D}$ has only one relation.

${}^{3}$ 对于 $\alpha  = k - 1,\mathcal{2}$ 是卢米斯 - 惠特尼连接（Loomis - Whitney join），并且已经得到了最优解。排除 $\alpha  \leq  2$ 的情况是因为，一般来说，所有包含 $\alpha  \leq  2$ 的查询都已经有了定论。$\alpha  = k$ 的情况也没有什么意义，因为 $\mathcal{D}$ 只有一个关系。

<!-- Footnote -->

---

We can in fact prove a stronger result. A more general class is the $\alpha$ -uniform join,where all the relations in a query2have arity $\alpha$ . For such $\mathcal{Q}$ ,the load of our algorithm can be further bounded as (Theorem 9.1)

实际上，我们可以证明一个更强的结果。更一般的一类是 $\alpha$ - 均匀连接（$\alpha$ -uniform join），其中查询 2 中的所有关系的元数都为 $\alpha$。对于这样的 $\mathcal{Q}$，我们算法的负载可以进一步界定为（定理 9.1）

$$
\widetilde{O}\left( {n/{p}^{\frac{2}{{\alpha \phi } - \alpha  + 2}}}\right) \text{.} \tag{4}
$$

Hence,our load for a $k$ -choose- $\alpha$ join is actually $\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$ , which strictly improves the KBS algorithm as long as $\alpha  < k$ .

因此，我们对于 $k$ 选 $\alpha$ 连接（$k$ -choose- $\alpha$ join）的负载实际上是 $\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$，只要 $\alpha  < k$ 成立，它就严格优于 KBS 算法。

Yet another notable class is the symmetric join,each being an $\alpha$ - uniform join $\mathcal{Q}$ with an additional constraint that each attribute in attset(Q)belongs to the same number of relations. The $k$ -choose- $\alpha$ join is a proper subset of the symmetric join. To see this, consider the cycle join [14] where $\alpha  = 2$ and2contains $k$ relations with schemes $\left\{  {{A}_{1},{A}_{2}}\right\}  ,\left\{  {{A}_{2},{A}_{3}}\right\}  ,\ldots ,\left\{  {{A}_{k - 1},{A}_{k}}\right\}  ,\left\{  {{A}_{k},{A}_{1}}\right\}$ ,respectively. A cycle join is symmetric but not a $k$ -choose-2 join if $k > 3$ .

另一类值得注意的是对称连接（symmetric join），每个对称连接都是一个 $\alpha$ - 均匀连接 $\mathcal{Q}$，并附加一个约束条件，即 attset(Q) 中的每个属性都属于相同数量的关系。$k$ 选 $\alpha$ 连接（$k$ -choose- $\alpha$ join）是对称连接的一个真子集。为了说明这一点，考虑循环连接 [14]，其中 $\alpha  = 2$ 且 2 分别包含 $k$ 个具有模式 $\left\{  {{A}_{1},{A}_{2}}\right\}  ,\left\{  {{A}_{2},{A}_{3}}\right\}  ,\ldots ,\left\{  {{A}_{k - 1},{A}_{k}}\right\}  ,\left\{  {{A}_{k},{A}_{1}}\right\}$ 的关系。如果 $k > 3$，那么循环连接是对称的，但不是 $k$ 选 2 连接。

The value $\phi$ for a symmetric join is always $k/\alpha$ (Section 4). By (4), our algorithm answers a symmetric query with load $\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$ . This has an interesting implication. In general, $\rho  \geq  k/2$ holds on all queries with $\alpha  \leq  2$ . Hence,a query on binary relations must incur a load of $\Omega \left( {n/{p}^{2/k}}\right)$ (see the lower bound discussion in Section 1.2). As $\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$ is $o\left( {n/{p}^{2/k}}\right)$ for $\alpha  \geq  3$ , every symmetric query with $\alpha  \geq  3$ is inherently easier than every query with $\alpha  \leq  2$ (and the same value of $k$ ). No existing algorithms can achieve such a separation.

对称连接的 $\phi$ 值始终为 $k/\alpha$（第 4 节）。根据 (4)，我们的算法以负载 $\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$ 回答对称查询。这有一个有趣的含义。一般来说，对于所有具有 $\alpha  \leq  2$ 的查询，$\rho  \geq  k/2$ 都成立。因此，对二元关系的查询必然会产生 $\Omega \left( {n/{p}^{2/k}}\right)$ 的负载（见第 1.2 节中的下界讨论）。由于对于 $\alpha  \geq  3$，$\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$ 为 $o\left( {n/{p}^{2/k}}\right)$，所以每个具有 $\alpha  \geq  3$ 的对称查询本质上都比每个具有 $\alpha  \leq  2$（且 $k$ 值相同）的查询更容易。现有的算法都无法实现这样的区分。

Table 1 presents a summary of our results in comparison with the existing ones.

表 1 展示了我们的结果与现有结果的对比总结。

A lower bound remark. As mentioned, our algorithm is optimal for $\alpha  = 2$ . For $\alpha  \geq  3$ ,the load of our algorithm in (3) cannot be significantly improved in the following sense: no algorithm can achieve a load of $o\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ in general. To explain,consider a class of queries2constructed as follows. Let ${A}_{1},\ldots ,{A}_{k/2}$ and ${B}_{1},\ldots ,{B}_{k/2}$ be $k \geq  6$ distinct attributes.2has $2 + k/2$ relations: (i) one with scheme $\left\{  {{A}_{1},\ldots ,{A}_{k/2}}\right\}$ and another with scheme $\left\{  {{B}_{1},\ldots ,{B}_{k/2}}\right\}$ ,and (ii) a relation with scheme $\left\{  {{A}_{i},{B}_{i}}\right\}$ for each $i \in  \left\lbrack  {k/2}\right\rbrack$ . For every such $\mathcal{Q},\alpha  = k/2$ and $\phi  = 2$ . As shown in [8],every algorithm requires a load of $\Omega \left( {n/{p}^{2/k}}\right)$ processing2. Notice that $\Omega \left( {n/{p}^{2/k}}\right)  =$ $\Omega \left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ ,i.e., $\Omega \left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ is also a lower bound on the load. Our algorithm is thus optimal on this class of queries.

一个下界说明。如前所述，我们的算法对于$\alpha  = 2$是最优的。对于$\alpha  \geq  3$，从以下意义上来说，我们算法在(3)中的负载无法显著改进：一般而言，没有算法能达到$o\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$的负载。为解释这一点，考虑一类按如下方式构造的查询2。设${A}_{1},\ldots ,{A}_{k/2}$和${B}_{1},\ldots ,{B}_{k/2}$是$k \geq  6$个不同的属性。2有$2 + k/2$个关系：(i) 一个具有模式$\left\{  {{A}_{1},\ldots ,{A}_{k/2}}\right\}$，另一个具有模式$\left\{  {{B}_{1},\ldots ,{B}_{k/2}}\right\}$，并且 (ii) 对于每个$i \in  \left\lbrack  {k/2}\right\rbrack$，有一个具有模式$\left\{  {{A}_{i},{B}_{i}}\right\}$的关系。对于每个这样的$\mathcal{Q},\alpha  = k/2$和$\phi  = 2$。如文献[8]所示，每个算法处理2都需要$\Omega \left( {n/{p}^{2/k}}\right)$的负载。注意到$\Omega \left( {n/{p}^{2/k}}\right)  =$ $\Omega \left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$，即$\Omega \left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$也是负载的一个下界。因此，我们的算法在这类查询上是最优的。

## 2 OVERVIEW OF OUR TECHNIQUES

## 2 我们的技术概述

We will first review two standard techniques and then discuss the new techniques developed in this paper.

我们将首先回顾两种标准技术，然后讨论本文开发的新技术。

Standard 1: Skew free. Consider an arbitrary relation $\mathcal{R} \in  \mathcal{Q}$ and any non-empty $\mathcal{V} \subseteq$ scheme(R). Given a tuple $\mathbf{v}$ over $\mathcal{V}$ ,define ${f}_{\mathcal{V}}\left( {\mathbf{v},\mathcal{R}}\right)$ as the number of tuples $\mathbf{u} \in  \mathcal{R}$ satisfying $\mathbf{v} = \mathbf{u}\left\lbrack  \mathcal{V}\right\rbrack$ ; we will refer to ${f}_{\mathcal{V}}\left( {\mathbf{v},\mathcal{R}}\right)$ as the $\mathcal{V}$ -frequency of $\mathbf{v}$ in $\mathcal{R}$ .

标准1：无倾斜。考虑任意关系$\mathcal{R} \in  \mathcal{Q}$和任何非空的$\mathcal{V} \subseteq$模式(R)。给定一个基于$\mathcal{V}$的元组$\mathbf{v}$，将${f}_{\mathcal{V}}\left( {\mathbf{v},\mathcal{R}}\right)$定义为满足$\mathbf{v} = \mathbf{u}\left\lbrack  \mathcal{V}\right\rbrack$的元组$\mathbf{u} \in  \mathcal{R}$的数量；我们将${f}_{\mathcal{V}}\left( {\mathbf{v},\mathcal{R}}\right)$称为$\mathbf{v}$在$\mathcal{R}$中的$\mathcal{V}$ - 频率。

Suppose that we assign a share of ${p}_{A} \geq  1$ to each attribute $A \in$ attset(Q)subject to

假设我们为每个属性$A \in$∈attset(Q)分配${p}_{A} \geq  1$的一部分，需满足

$$
\mathop{\prod }\limits_{{A \in  \operatorname{attset}\left( \mathcal{Q}\right) }}{p}_{A} \leq  p \tag{5}
$$

Relation $\mathcal{R} \in  \mathcal{Q}$ is skew free if

如果关系$\mathcal{R} \in  \mathcal{Q}$满足

$$
{f}_{\mathcal{V}}\left( {\mathbf{v},\mathcal{R}}\right)  \leq  \frac{n}{\mathop{\prod }\limits_{{A \in  \mathcal{V}}}{p}_{A}} \tag{6}
$$

holds for every non-empty $\mathcal{V} \subseteq$ scheme(R) and any tuple $\mathbf{v}$ over $\mathcal{V}.\mathcal{2}$ is skew free if all relations in2are skew free.

对于每个非空的$\mathcal{V} \subseteq$模式(R)和任何基于$\mathcal{V}.\mathcal{2}$的元组$\mathbf{v}$都成立，则该关系是无倾斜的。如果2中的所有关系都是无倾斜的，则2是无倾斜的。

Standard 2: BinHC and the heavy-light technique. Suppose that the share ${p}_{A}$ of every attribute $A \in$ attset(Q)has been decided. Beame, Koutris, and Suciu [6] proved that the BinHC algorithm (Section 1.2) solves a skew free query with load

标准2：BinHC和轻重技术。假设每个属性$A \in$∈attset(Q)的份额${p}_{A}$已经确定。Beame、Koutris和Suciu [6]证明了BinHC算法（第1.2节）能以负载解决无倾斜查询

$$
\widetilde{O}\left( {\mathop{\max }\limits_{{\mathcal{R} \in  \mathcal{Q}}}\frac{n}{\mathop{\prod }\limits_{{A \in  \operatorname{scheme}\left( \mathcal{R}\right) }}{p}_{A}}}\right) . \tag{7}
$$

When $\mathcal{Q}$ is not skew free,a common approach $\left\lbrack  {{12},{14},{20}}\right\rbrack$ is to resort to the following algorithmic paradigm. First, design a number of sub-queries. Then, choose the (attribute) shares appropriately to make every sub-query skew free and, thus, solvable by BinHC. What differentiates different algorithms is how to determine the sub-queries such that they

当$\mathcal{Q}$不是无倾斜时，一种常见的方法$\left\lbrack  {{12},{14},{20}}\right\rbrack$是采用以下算法范式。首先，设计多个子查询。然后，适当地选择（属性）份额，使每个子查询都是无倾斜的，从而可以由BinHC解决。不同算法的区别在于如何确定子查询，使得它们

- (Objective 1) together produce precisely Join(2), and

- （目标 1）共同精确生成连接结果 Join(2)，并且

- (Objective 2) incur a small load under BinHC.

- （目标 2）在 BinHC 下产生较小的负载。

Fulfilling these objectives is non-trivial. In [14], Koutris, Beame, and Suciu presented a method that we refer to as the heavy-light technique,as outlined below. Let $\lambda  > 0$ be some real value. Call a value $x \in  \mathbf{{dom}}$

实现这些目标并非易事。在文献[14]中，库特里斯（Koutris）、比姆（Beame）和苏丘（Suciu）提出了一种我们称之为重 - 轻技术的方法，概述如下。设 $\lambda  > 0$ 为某个实数值。称一个值 $x \in  \mathbf{{dom}}$

- heavy,if there exist a relation $\mathcal{R} \in  \mathcal{Q}$ and an attribute $A \in$ scheme( $\mathcal{R}$ ) such that $\mathcal{R}$ has at least $n/\lambda$ tuples $\mathbf{u}$ with $\mathbf{u}\left( A\right)  =$ $x$ ;

- 为重值，如果存在一个关系 $\mathcal{R} \in  \mathcal{Q}$ 和一个属性 $A \in$ 模式( $\mathcal{R}$ )，使得 $\mathcal{R}$ 至少有 $n/\lambda$ 个元组 $\mathbf{u}$ 满足 $\mathbf{u}\left( A\right)  =$ $x$ ；

- light, otherwise.

- 否则为轻值。

For every possible $\mathcal{U} \subseteq$ attset(Q),create a sub-query ${\mathcal{Q}}_{\mathcal{U}}$ to produce all the tuples $\mathbf{u} \in  \mathcal{J}$ oin(Q)such that $\mathbf{u}\left( A\right)$ is heavy if $A \in  \mathcal{U}$ or light otherwise. Clearly,there are ${2}^{k}$ sub-queries (see (1) for $k$ ); and the union of their results is $\mathcal{J}$ oin(Q). This achieves Objective 1.

对于每个可能的 $\mathcal{U} \subseteq$ 属于 attset(Q)，创建一个子查询 ${\mathcal{Q}}_{\mathcal{U}}$ 以生成连接结果 Join(Q) 中的所有元组 $\mathbf{u} \in  \mathcal{J}$，使得如果 $A \in  \mathcal{U}$ 成立则 $\mathbf{u}\left( A\right)$ 为重值，否则为轻值。显然，有 ${2}^{k}$ 个子查询（见(1)中关于 $k$ 的内容）；并且它们结果的并集就是连接结果 Join(Q)。这实现了目标 1。

How about Objective 2? Let us fix a $\mathcal{U} \subseteq$ attset(Q)and concentrate on an arbitrary relation $\mathcal{R} \in  \mathcal{Q}$ . As far as $\mathcal{Q}\mathcal{U}$ is concerned, we only need to consider those tuples $\mathbf{u} \in  \mathcal{R}$ such that,for each $A \in$ scheme $\left( \mathcal{R}\right) ,\mathbf{u}\left( A\right)$ is heavy if $A \in  \mathcal{U}$ and light otherwise; denote by ${\mathcal{R}}^{\prime }$ the set of such tuples. To apply BinHC on ${\mathcal{Q}}_{\mathcal{U}}$ ,we must set the shares to make ${\mathcal{R}}^{\prime }$ skew free. This,however,can be difficult because we do not have control over the $\mathcal{V}$ -frequency of a tuple for any subset $\mathcal{V} \subseteq$ scheme $\left( {\mathcal{R}}^{\prime }\right)$ with $\left| \mathcal{V}\right|  \geq  2$ .

那么目标 2 呢？让我们固定一个 $\mathcal{U} \subseteq$ 属于 attset(Q)，并关注任意一个关系 $\mathcal{R} \in  \mathcal{Q}$。就 $\mathcal{Q}\mathcal{U}$ 而言，我们只需要考虑那些元组 $\mathbf{u} \in  \mathcal{R}$，使得对于每个 $A \in$ 模式，如果 $A \in  \mathcal{U}$ 成立则 $\left( \mathcal{R}\right) ,\mathbf{u}\left( A\right)$ 为重值，否则为轻值；用 ${\mathcal{R}}^{\prime }$ 表示这样的元组集合。为了在 ${\mathcal{Q}}_{\mathcal{U}}$ 上应用 BinHC，我们必须设置份额以使 ${\mathcal{R}}^{\prime }$ 无倾斜。然而，这可能很困难，因为我们无法控制任何子集 $\mathcal{V} \subseteq$ 模式 $\left( {\mathcal{R}}^{\prime }\right)$ 且 $\left| \mathcal{V}\right|  \geq  2$ 时元组的 $\mathcal{V}$ - 频率。

Koutris, Beame, and Suciu [14] circumvented the issue by setting $\lambda  = p$ and fixing the share ${p}_{A}$ to 1 for each attribute $A \in  \mathcal{U}$ . Interestingly, ${\mathcal{R}}^{\prime }$ is guaranteed skew free regardless of the shares ${p}_{A}$ of $A \notin  \mathcal{U}$ . This follows from two observations: (i) trivially,every heavy value can appear in at most $\left| {\mathcal{R}}^{\prime }\right|  \leq  n$ tuples,and (ii) every light value can belong to $\widetilde{O}\left( {n/p}\right)  = \widetilde{O}\left( {n/\mathop{\prod }\limits_{{A \notin  \mathcal{U}}}{p}_{A}}\right)$ tuples,noticing that $\mathop{\prod }\limits_{{A \notin  \mathcal{U}}}{p}_{A} \leq  p$ . The KBS algorithm achieves load $\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$ by optimizing the shares ${p}_{A}$ of $A \notin  \mathcal{U}$ .

库特里斯（Koutris）、比姆（Beame）和苏丘（Suciu）[14] 通过设置 $\lambda  = p$ 并将每个属性 $A \in  \mathcal{U}$ 的份额 ${p}_{A}$ 固定为 1 来规避这个问题。有趣的是，无论 $A \notin  \mathcal{U}$ 的份额 ${p}_{A}$ 如何，都能保证 ${\mathcal{R}}^{\prime }$ 无倾斜。这基于两个观察结果：(i) 显然，每个重值最多可以出现在 $\left| {\mathcal{R}}^{\prime }\right|  \leq  n$ 个元组中，(ii) 每个轻值可以属于 $\widetilde{O}\left( {n/p}\right)  = \widetilde{O}\left( {n/\mathop{\prod }\limits_{{A \notin  \mathcal{U}}}{p}_{A}}\right)$ 个元组，注意到 $\mathop{\prod }\limits_{{A \notin  \mathcal{U}}}{p}_{A} \leq  p$。KBS 算法通过优化 $A \notin  \mathcal{U}$ 的份额 ${p}_{A}$ 实现了负载 $\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$。

The above approach fails to work for $\lambda  \ll  p$ ,whereas we often need $\lambda  = {p}^{c}$ for some $c < 1$ to improve upon $\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$ . To deal with the issue, Ketsman and Suciu [12] and Tao [20] refined the heavy-light technique, but their refinement heavily relies on the premise $\alpha  \leq  2$ .

上述方法对于$\lambda  \ll  p$不起作用，然而我们通常需要针对某些$c < 1$的$\lambda  = {p}^{c}$来改进$\widetilde{O}\left( {n/{p}^{1/\psi }}\right)$。为了解决这个问题，凯茨曼（Ketsman）和苏丘（Suciu）[12]以及陶（Tao）[20]对轻重技术进行了改进，但他们的改进在很大程度上依赖于前提$\alpha  \leq  2$。

<!-- Media -->

<!-- figureText: C O E (b) A residual query for the plan $\left( {\{ \mathrm{D}\} ,\{ \left( {\mathrm{G},\mathrm{H}}\right) \} }\right)$ O B D=d E O G=g H=h F (a) The original query $\left( {\rho  = \phi  = 5\text{and}\psi  = 9}\right)$ -->

<img src="https://cdn.noedgeai.com/0195ccc4-dfe1-76a1-808a-46238815680a_3.jpg?x=177&y=235&w=1433&h=294&r=0"/>

Figure 1: Illustration of the proposed techniques

图1：所提出技术的示意图

<!-- Media -->

New 1: Two-attribute skew free. For a relation $\mathcal{R} \in  \mathcal{Q}$ ,the skew free condition demands that $\mathcal{V}$ -frequencies be low for all nonempty $\mathcal{V} \subseteq$ scheme(R). The is a stringent requirement and limits the applicability of BinHC.

新特性1：无两属性倾斜。对于关系$\mathcal{R} \in  \mathcal{Q}$，无倾斜条件要求对于所有非空的$\mathcal{V} \subseteq$方案(R)，$\mathcal{V}$频率都要低。这是一个严格的要求，限制了BinHC的适用性。

Our first idea is to relax the requirement. We say that $\mathcal{R}$ is two-attribute skew free as long as (6) holds for every non-empty $\mathcal{V} \subseteq$ scheme(R)with $\left| \mathcal{V}\right|  \leq  2.\mathcal{2}$ is two-attribute skew free if all its relations are two-attribute skew free. As will be proved later (Lemma 3.5),a two-attribute skew free query $\mathcal{Q}$ can be answered by BinHC with load

我们的第一个想法是放宽这个要求。我们说，只要对于每个非空的$\mathcal{V} \subseteq$方案(R)，(6)式成立，那么$\mathcal{R}$就是无两属性倾斜的。如果一个数据库的所有关系都是无两属性倾斜的，那么该数据库就是无两属性倾斜的。正如后面将证明的（引理3.5），无两属性倾斜的查询$\mathcal{Q}$可以由BinHC以一定负载进行处理

$$
\widetilde{O}\left( {\mathop{\max }\limits_{{\mathcal{R} \in  \mathcal{Q}}}\left( {\mathop{\min }\limits_{{\mathcal{V} \subseteq  \text{ scheme }\left( \mathcal{R}\right) }}\frac{n}{\mathop{\prod }\limits_{{A \in  \mathcal{V}}}{p}_{A}}}\right) }\right) . \tag{8}
$$

Relaxation of the skew free constraint is a matter of tradeoff. On the one hand, the load in (8) is higher than (7); but on the other hand, we gain greater flexibility in assigning shares. Fortunately, we can compensate for the loss in (8) with an enhanced heavy-light technique that is made possible by the new skew-free definition, as outlined below.

放宽无倾斜约束是一个权衡问题。一方面，(8)式中的负载高于(7)式；但另一方面，我们在分配份额方面获得了更大的灵活性。幸运的是，我们可以通过下面概述的新的无倾斜定义所实现的增强型轻重技术来弥补(8)式中的损失。

New 2: Two-attribute heavy-light. Given a $\lambda  > 0$ ,define a value $x \in$ dom to be light/heavy in the same way as before. In addition, we say that a value pair $\left( {y,z}\right)  \in  \mathbf{{dom}} \times  \mathbf{{dom}}$ is:

新特性2：两属性轻重。给定一个$\lambda  > 0$，以与之前相同的方式定义值$x \in$ dom为轻/重。此外，我们说值对$\left( {y,z}\right)  \in  \mathbf{{dom}} \times  \mathbf{{dom}}$是：

- heavy,if there exist a relation $\mathcal{R} \in  \mathcal{Q}$ and two distinct attributes $Y,Z \in$ scheme(R)such that the $\{ Y,Z\}$ -frequency of tuple(y,z)in $\mathcal{R}$ is at least $n/{\lambda }^{2}$ ;

- 重的，如果存在一个关系$\mathcal{R} \in  \mathcal{Q}$和两个不同的属性$Y,Z \in$方案(R)，使得元组(y,z)在$\mathcal{R}$中的$\{ Y,Z\}$频率至少为$n/{\lambda }^{2}$；

- light, otherwise.

- 轻的，否则。

Define a plan as:

将一个计划定义为：

$$
\mathbf{P} = \left( {\left\{  {{X}_{1},\ldots ,{X}_{a}}\right\}  ,\left\{  {\left( {{Y}_{1},{Z}_{1}}\right) ,\ldots ,\left( {{Y}_{b},{Z}_{b}}\right) }\right\}  }\right)  \tag{9}
$$

where $a \geq  0,b \geq  0,{X}_{1},\ldots ,{X}_{a},{Y}_{1},\ldots ,{Y}_{b},{Z}_{1},\ldots ,{Z}_{b}$ are distinct attributes in attset(Q),and ${Y}_{j} \prec  {Z}_{j}$ for each $j \in  \left\lbrack  b\right\rbrack$ . Since $\mid$ attset $\left( \mathcal{Q}\right)  \mid$ is a constant,only $O\left( 1\right)$ plans exist.

其中$a \geq  0,b \geq  0,{X}_{1},\ldots ,{X}_{a},{Y}_{1},\ldots ,{Y}_{b},{Z}_{1},\ldots ,{Z}_{b}$是attset(Q)中的不同属性，并且对于每个$j \in  \left\lbrack  b\right\rbrack$有${Y}_{j} \prec  {Z}_{j}$。由于$\mid$ attset $\left( \mathcal{Q}\right)  \mid$是一个常数，因此只存在$O\left( 1\right)$个计划。

Let us concentrate on a plan $\mathbf{P}$ . Define $\mathcal{H} = \left\{  {{X}_{1},\ldots ,{X}_{a},{Y}_{1}}\right.$ , $\left. {\ldots ,{Y}_{b},{Z}_{1},\ldots ,{Z}_{b}}\right\}$ . We issue sub-queries to extract all the tuples $\mathbf{u} \in  \mathcal{J}$ oin( $\mathcal{Q}$ ) satisfying:

让我们专注于一个计划$\mathbf{P}$。定义$\mathcal{H} = \left\{  {{X}_{1},\ldots ,{X}_{a},{Y}_{1}}\right.$，$\left. {\ldots ,{Y}_{b},{Z}_{1},\ldots ,{Z}_{b}}\right\}$。我们发出子查询以提取所有满足以下条件的元组$\mathbf{u} \in  \mathcal{J}$ oin( $\mathcal{Q}$ )：

- $\mathbf{u}\left( {X}_{i}\right)$ is heavy for all $i \in  \left\lbrack  a\right\rbrack$ ;

- 对于所有$i \in  \left\lbrack  a\right\rbrack$，$\mathbf{u}\left( {X}_{i}\right)$是重的；

- $\mathbf{u}\left( A\right)$ is light for any attribute $A \notin  \left\{  {{X}_{1},\ldots ,{X}_{a}}\right\}$ ;

- 对于任何属性$A \notin  \left\{  {{X}_{1},\ldots ,{X}_{a}}\right\}$，$\mathbf{u}\left( A\right)$是轻的；

- $\left( {\mathbf{u}\left( {Y}_{i}\right) ,\mathbf{u}\left( {Z}_{i}\right) }\right)$ is heavy for all $i \in  \left\lbrack  b\right\rbrack$ .

- $\left( {\mathbf{u}\left( {Y}_{i}\right) ,\mathbf{u}\left( {Z}_{i}\right) }\right)$ 对所有 $i \in  \left\lbrack  b\right\rbrack$ 而言都是重的。

- $\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ is light for any distinct attributes $A,B \notin  \mathcal{H}$ .

- $\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ 对任何不同的属性 $A,B \notin  \mathcal{H}$ 而言都是轻的。

The union of all sub-queries' results is precisely Join(2) (Lemma 5.2).

所有子查询结果的并集恰好是连接操作（Join(2)）（引理5.2）。

For an illustration,Figure 1(a) shows a query $\mathcal{Q}$ with attset $\left( \mathcal{Q}\right)  =$ $\{ \mathrm{A},\mathrm{B},\ldots ,\mathrm{K}\}$ . Each segment represents a binary relation,e.g., $\{ \mathrm{A},\mathrm{G}\}$ . Each ellipse represents a relation of arity 3,e.g., $\{ A,B,C\} .\mathcal{2}$ has thirteen binary relations and three arity-3 relations.

为了说明这一点，图1(a)展示了一个带有属性集 $\left( \mathcal{Q}\right)  =$ $\{ \mathrm{A},\mathrm{B},\ldots ,\mathrm{K}\}$ 的查询 $\mathcal{Q}$。每个线段代表一个二元关系，例如 $\{ \mathrm{A},\mathrm{G}\}$。每个椭圆代表一个三元关系，例如 $\{ A,B,C\} .\mathcal{2}$ 有十三个二元关系和三个三元关系。

Consider the plan $\mathbf{P} = \left( {\{ \mathrm{D}\} ,\{ \left( {\mathrm{G},\mathrm{H}}\right) \} }\right)$ . Each sub-query ${\mathcal{Q}}^{\prime }$ issued for $P$ assigns (i) a heavy value - assumed d below - to D,and (ii) a heavy value pair-assumed $\left( {\mathrm{g},\mathrm{h}}\right)  -$ to $\left( {\mathrm{G},\mathrm{H}}\right) .{\mathcal{Q}}^{\prime }$ returns all and only the tuples $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( Q\right)$ such that:

考虑计划 $\mathbf{P} = \left( {\{ \mathrm{D}\} ,\{ \left( {\mathrm{G},\mathrm{H}}\right) \} }\right)$。为 $P$ 发出的每个子查询 ${\mathcal{Q}}^{\prime }$ 会（i）为D分配一个重值（以下假设为d），并且（ii）为 $\left( {\mathrm{G},\mathrm{H}}\right) .{\mathcal{Q}}^{\prime }$ 分配一个重值对（假设为 $\left( {\mathrm{g},\mathrm{h}}\right)  -$），返回所有且仅返回满足以下条件的元组 $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( Q\right)$：

- $u\left( \mathrm{D}\right)  = \mathrm{d}$ ;

- $\left( {\mathbf{u}\left( \mathrm{G}\right) ,\mathbf{u}\left( \mathrm{H}\right) }\right)  = \left( {\mathrm{g},\mathrm{h}}\right)$ ;

- $\mathbf{u}\left( A\right)$ is light for every attribute $A \in  \{ \mathrm{A},\mathrm{B},\mathrm{C},\mathrm{E},\mathrm{F},\mathrm{G},\mathrm{H},\mathrm{I},\mathrm{J},\mathrm{K}\}$ (this implies that both $\mathrm{g}$ and $\mathrm{h}$ are light);

- $\mathbf{u}\left( A\right)$ 对每个属性 $A \in  \{ \mathrm{A},\mathrm{B},\mathrm{C},\mathrm{E},\mathrm{F},\mathrm{G},\mathrm{H},\mathrm{I},\mathrm{J},\mathrm{K}\}$ 而言都是轻的（这意味着 $\mathrm{g}$ 和 $\mathrm{h}$ 都是轻的）；

- $\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ is light for any distinct attributes $A,B \in  \{ \mathrm{A},\mathrm{B}$ , $C,E,F,I,J,K\}$ .

- $\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ 对任何不同的属性 $A,B \in  \{ \mathrm{A},\mathrm{B}$、$C,E,F,I,J,K\}$ 而言都是轻的。

The relations in ${\mathcal{Q}}^{\prime }$ only need to contain the relevant tuples. For example,let ${\mathcal{R}}_{\{ \mathrm{G},\mathrm{J}\} }$ be the relation in $\mathcal{Q}$ with scheme $\{ \mathrm{G},\mathrm{J}\}$ . In ${\mathcal{Q}}^{\prime }$ ,the corresponding relation ${\mathcal{R}}_{\{ \mathrm{G},\mathrm{\;J}\} }^{\prime }$ includes only the tuples $\mathbf{v} \in  {\mathcal{R}}_{\{ \mathrm{G},\mathrm{J}\} }$ such that $\mathbf{v}\left( \mathrm{G}\right)  = \mathrm{g}$ and $\mathbf{v}\left( \mathrm{J}\right)$ is light. As another example,let ${\mathcal{R}}_{\{ \mathrm{A},\mathrm{B},\mathrm{C}\} }$ be the relation in $\mathcal{Q}$ with scheme $\{ \mathrm{A},\mathrm{B},\mathrm{C}\}$ . The corresponding relation ${\mathcal{R}}_{\{ \mathrm{A},\mathrm{B},\mathrm{C}\} }^{\prime }$ in ${\mathcal{Q}}^{\prime }$ includes only the tuples $\mathbf{v} \in  {\mathcal{R}}_{\{ \mathrm{A},\mathrm{B},\mathrm{C}\} }$ such that $\mathbf{v}\left( \mathrm{A}\right) ,\mathbf{v}\left( \mathrm{B}\right) ,\mathbf{v}\left( \mathrm{C}\right)$ are light,and so are the value pairs $\left( {\mathbf{v}\left( \mathrm{A}\right) ,\mathbf{v}\left( \mathrm{B}\right) }\right) ,\left( {\mathbf{v}\left( \mathrm{B}\right) ,\mathbf{v}\left( \mathrm{C}\right) }\right) ,\left( {\mathbf{v}\left( \mathrm{A}\right) ,\mathbf{v}\left( \mathrm{C}\right) }\right)$ .

${\mathcal{Q}}^{\prime }$中的关系仅需包含相关元组。例如，设${\mathcal{R}}_{\{ \mathrm{G},\mathrm{J}\} }$为$\mathcal{Q}$中具有模式$\{ \mathrm{G},\mathrm{J}\}$的关系。在${\mathcal{Q}}^{\prime }$中，对应的关系${\mathcal{R}}_{\{ \mathrm{G},\mathrm{\;J}\} }^{\prime }$仅包含满足$\mathbf{v}\left( \mathrm{G}\right)  = \mathrm{g}$且$\mathbf{v}\left( \mathrm{J}\right)$为轻量（light）的元组$\mathbf{v} \in  {\mathcal{R}}_{\{ \mathrm{G},\mathrm{J}\} }$。再举一个例子，设${\mathcal{R}}_{\{ \mathrm{A},\mathrm{B},\mathrm{C}\} }$为$\mathcal{Q}$中具有模式$\{ \mathrm{A},\mathrm{B},\mathrm{C}\}$的关系。${\mathcal{Q}}^{\prime }$中对应的关系${\mathcal{R}}_{\{ \mathrm{A},\mathrm{B},\mathrm{C}\} }^{\prime }$仅包含满足$\mathbf{v}\left( \mathrm{A}\right) ,\mathbf{v}\left( \mathrm{B}\right) ,\mathbf{v}\left( \mathrm{C}\right)$为轻量（light）且值对$\left( {\mathbf{v}\left( \mathrm{A}\right) ,\mathbf{v}\left( \mathrm{B}\right) }\right) ,\left( {\mathbf{v}\left( \mathrm{B}\right) ,\mathbf{v}\left( \mathrm{C}\right) }\right) ,\left( {\mathbf{v}\left( \mathrm{A}\right) ,\mathbf{v}\left( \mathrm{C}\right) }\right)$也为轻量（light）的元组$\mathbf{v} \in  {\mathcal{R}}_{\{ \mathrm{A},\mathrm{B},\mathrm{C}\} }$。

Since attributes D, G, and H have been fixed to specific values, they can be removed, giving rise to a residual query as is shown in Figure 1(b). The 3-arity relation with scheme $\{ \mathrm{C},\mathrm{D},\mathrm{E}\}$ ,for instance, now becomes a binary relation with scheme $\{ \mathrm{C},\mathrm{E}\}$ . Similarly,three isolated unary relations (on attributes $\mathrm{F},\mathrm{J}$ ,and $\mathrm{K}$ ) have been created. We resolve this residual query in three (conceptual) steps:

由于属性D、G和H已被固定为特定值，因此可以将它们移除，从而产生如图1(b)所示的剩余查询。例如，具有模式$\{ \mathrm{C},\mathrm{D},\mathrm{E}\}$的三元关系现在变为具有模式$\{ \mathrm{C},\mathrm{E}\}$的二元关系。类似地，已经创建了三个孤立的一元关系（关于属性$\mathrm{F},\mathrm{J}$和$\mathrm{K}$）。我们分三个（概念上的）步骤来解决这个剩余查询：

(1) (Non-unary join) compute ${\mathcal{J}}_{1}$ ,the join result of the non-unary relations with schemes $\{ A,B,C\} ,\{ C,E\}$ ,and $\{ E,I\}$ .

(1)（非一元连接）计算${\mathcal{J}}_{1}$，即具有模式$\{ A,B,C\} ,\{ C,E\}$和$\{ E,I\}$的非一元关系的连接结果。

(2) (Isolated CP) compute ${\mathcal{J}}_{2}$ ,the cartesian product (CP) of the three isolated unary relations.

(2)（孤立笛卡尔积）计算${\mathcal{J}}_{2}$，即三个孤立的一元关系的笛卡尔积（CP）。

(3) (Final CP) compute ${\mathcal{J}}_{1} \times  {\mathcal{J}}_{2}$ ,the result of the residual query.

(3)（最终笛卡尔积）计算${\mathcal{J}}_{1} \times  {\mathcal{J}}_{2}$，即剩余查询的结果。

Set $\lambda  = {p}^{1/\left( {\alpha \phi }\right) }(\phi$ is the generalized vertex-packing number of $\mathcal{Q}$ ; Section 4). By assigning a share $\lambda$ to each attribute in $\{ \mathrm{A},\mathrm{B},\mathrm{C},\mathrm{E},\mathrm{I}\}$ , we guarantee that every non-unary relation is two-attribute skew free. Hence,BinHC can be used to perform Step (1) with load $\widetilde{O}\left( {n/{\lambda }^{2}}\right)  =$ $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ ,by virtue of (8).

集合 $\lambda  = {p}^{1/\left( {\alpha \phi }\right) }(\phi$ 是 $\mathcal{Q}$ 的广义顶点填充数（详见第4节）。通过为 $\{ \mathrm{A},\mathrm{B},\mathrm{C},\mathrm{E},\mathrm{I}\}$ 中的每个属性分配一个份额 $\lambda$ ，我们确保每个非一元关系都是无二元属性偏斜的。因此，根据式(8)，可以使用BinHC以负载 $\widetilde{O}\left( {n/{\lambda }^{2}}\right)  =$ $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 执行步骤(1)。

Ensuring load $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ for Steps (2) and (3),however,requires new insight into the mathematical structure of the problem, as explained next.

然而，要确保步骤(2)和(3)的负载为 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ ，需要对该问题的数学结构有新的见解，接下来将进行解释。

New 3: A new isolated cartesian product theorem. The load of Steps (2) and (3) critically depends on $\left| {\mathcal{J}}_{2}\right|$ ,namely,the CP size of the isolated unary relations (e.g.,those on F, J, and K in Figure 1(b)). If $\left| {\mathcal{J}}_{2}\right|$ were small for every residual query of the plan $\mathbf{P}$ ,we could bound the load to be $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ easily. Unfortunately,this is not true: $\left| {\mathcal{J}}_{2}\right|$ can vary significantly for different residual queries.

新成果3：一个新的孤立笛卡尔积定理。步骤(2)和(3)的负载关键取决于 $\left| {\mathcal{J}}_{2}\right|$ ，即孤立一元关系的笛卡尔积大小（例如，图1(b)中关于F、J和K的那些关系）。如果对于计划 $\mathbf{P}$ 的每个剩余查询， $\left| {\mathcal{J}}_{2}\right|$ 都很小，我们可以轻松地将负载限制为 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 。不幸的是，情况并非如此： $\left| {\mathcal{J}}_{2}\right|$ 对于不同的剩余查询可能会有显著差异。

We will establish an isolated cartesian product theorem, showing that the average $\left| {\mathcal{J}}_{2}\right|$ of all residual queries dedicated to $P$ is sufficiently small. This permits global optimization to allocate more (resp. less) machines to those residual queries with larger (resp. smaller) $\left| {\mathcal{J}}_{2}\right|$ ,thereby guaranteeing a load of $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ for every residual query. The theorem generalizes an earlier result on binary relations in $\left\lbrack  {{13},{20}}\right\rbrack$ . Its establishment,which constitutes the most important technical contribution of this paper, owes heavily to the newly introduced $\phi$ . Our argument is considerably different from (and actually subsumes) the ones in $\left\lbrack  {{13},{20}}\right\rbrack$ ,and sheds new light on the join problem.

我们将建立一个孤立笛卡尔积定理，表明专门针对 $P$ 的所有剩余查询的平均 $\left| {\mathcal{J}}_{2}\right|$ 足够小。这使得全局优化可以为 $\left| {\mathcal{J}}_{2}\right|$ 较大（或较小）的剩余查询分配更多（或更少）的机器，从而确保每个剩余查询的负载为 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 。该定理推广了文献 $\left\lbrack  {{13},{20}}\right\rbrack$ 中关于二元关系的早期结果。该定理的建立是本文最重要的技术贡献，这在很大程度上归功于新引入的 $\phi$ 。我们的论证与文献 $\left\lbrack  {{13},{20}}\right\rbrack$ 中的论证有很大不同（实际上包含了它们），并为连接问题提供了新的思路。

## 3 PRELIMINARIES

## 3 预备知识

### 3.1 Hypergraphs & Edge Coverings/Packings

### 3.1 超图与边覆盖/填充

A hypergraph $\mathcal{G}$ is a pair(V,E)such that (i) $\mathcal{V}$ is a finite set where each element is a vertex,and (ii) $\mathcal{E}$ is a set of hyperedges - or just edges - each being a non-empty subset of $\mathcal{V}$ . For each $e \in  \mathcal{E}$ ,the size $\left| e\right|$ is the edge’s arity; and $e$ is unary if $\left| e\right|  = 1$ . A vertex in $\mathcal{V}$ is exposed if it belongs to no edges. Our discussion will concentrate on hypergraphs without exposed vertices.

超图 $\mathcal{G}$ 是一个二元组(V, E)，满足：(i) $\mathcal{V}$ 是一个有限集，其每个元素都是一个顶点；(ii) $\mathcal{E}$ 是一个超边（或简称为边）的集合，每条超边都是 $\mathcal{V}$ 的非空子集。对于每条 $e \in  \mathcal{E}$ ，其大小 $\left| e\right|$ 是该边的元数；如果 $\left| e\right|  = 1$ ，则 $e$ 是一元的。 $\mathcal{V}$ 中的一个顶点如果不属于任何边，则称其为暴露顶点。我们的讨论将集中在没有暴露顶点的超图上。

Let $\mathcal{W}$ be a function mapping $\mathcal{E}$ to values in $\left\lbrack  {0,1}\right\rbrack$ . We call $\mathcal{W}\left( e\right)$ the weight of $e \in  \mathcal{E}$ (under $\mathcal{W}$ ) and $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$ the weight of $\mathcal{W}$ . Given a vertex $X \in  \mathcal{V}$ ,we call $\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}\mathcal{W}\left( e\right)$ the weight of $X$ (under $\mathcal{W}$ ). $\mathcal{W}$ is a fractional edge covering of $\mathcal{G}$ if the weight of every vertex $X \in  \mathcal{V}$ is at least 1 . The fractional edge covering number of $\mathcal{G} -$ denoted as $\rho \left( \mathcal{G}\right)  -$ is the minimum weight of all fractional edge coverings of $\mathcal{G}$ . $\mathcal{W}$ is a fractional edge packing of $\mathcal{G}$ if the weight of every vertex $X \in  \mathcal{V}$ is at most 1 . The fractional edge packing number of $\mathcal{G} -$ denoted as $\tau \left( \mathcal{G}\right)  -$ is the maximum weight of all fractional edge packings of $\mathcal{G}$ .

设$\mathcal{W}$是一个将$\mathcal{E}$映射到$\left\lbrack  {0,1}\right\rbrack$中值的函数。我们称$\mathcal{W}\left( e\right)$为$e \in  \mathcal{E}$的权重（在$\mathcal{W}$下），称$\mathop{\sum }\limits_{{e \in  \mathcal{E}}}W\left( e\right)$为$\mathcal{W}$的权重。给定一个顶点$X \in  \mathcal{V}$，我们称$\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}\mathcal{W}\left( e\right)$为$X$的权重（在$\mathcal{W}$下）。如果每个顶点$X \in  \mathcal{V}$的权重至少为1，则$\mathcal{W}$是$\mathcal{G}$的一个分数边覆盖。$\mathcal{G} -$的分数边覆盖数（记为$\rho \left( \mathcal{G}\right)  -$）是$\mathcal{G}$的所有分数边覆盖的最小权重。如果每个顶点$X \in  \mathcal{V}$的权重至多为1，则$\mathcal{W}$是$\mathcal{G}$的一个分数边填充。$\mathcal{G} -$的分数边填充数（记为$\tau \left( \mathcal{G}\right)  -$）是$\mathcal{G}$的所有分数边填充的最大权重。

Example. The hypergraph $\mathcal{G}$ in Figure 1(a) has a fractional edge covering number $\rho \left( \mathcal{G}\right)  = 5$ ,achieved by the function $\mathcal{W}$ that maps $\{ \mathrm{D},\mathrm{K}\} ,\{ \mathrm{G},\mathrm{J}\} ,\{ \mathrm{I},\mathrm{E}\} ,\{ \mathrm{A},\mathrm{B},\mathrm{C}\}$ ,and $\{ \mathrm{F},\mathrm{G},\mathrm{H}\}$ to 1,and the other edges to 0 . $\mathcal{G}$ has a fractional edge packing number $\tau \left( \mathcal{G}\right)  = {4.5}$ ,achieved by the function $\mathcal{W}$ that maps $\{ \mathrm{D},\mathrm{H}\} ,\{ \mathrm{D},\mathrm{K}\}$ ,and $\{ \mathrm{K},\mathrm{H}\}$ to ${0.5},\{ \mathrm{E},\mathrm{I}\} ,\{ \mathrm{G},\mathrm{J}\}$ , and $\{ A,B,C\}$ to 1 ,and the other edges to 0 .

示例。图1(a)中的超图$\mathcal{G}$的分数边覆盖数为$\rho \left( \mathcal{G}\right)  = 5$，这可以通过函数$\mathcal{W}$实现，该函数将$\{ \mathrm{D},\mathrm{K}\} ,\{ \mathrm{G},\mathrm{J}\} ,\{ \mathrm{I},\mathrm{E}\} ,\{ \mathrm{A},\mathrm{B},\mathrm{C}\}$和$\{ \mathrm{F},\mathrm{G},\mathrm{H}\}$映射为1，将其他边映射为0。$\mathcal{G}$的分数边填充数为$\tau \left( \mathcal{G}\right)  = {4.5}$，这可以通过函数$\mathcal{W}$实现，该函数将$\{ \mathrm{D},\mathrm{H}\} ,\{ \mathrm{D},\mathrm{K}\}$和$\{ \mathrm{K},\mathrm{H}\}$映射为${0.5},\{ \mathrm{E},\mathrm{I}\} ,\{ \mathrm{G},\mathrm{J}\}$，将$\{ A,B,C\}$映射为1，将其他边映射为0。

LEMMA 3.1. If $\alpha  = \mathop{\max }\limits_{{e \in  \mathcal{E}}}\left| e\right|$ ,then $\alpha  \cdot  \rho \left( \mathcal{G}\right)  \geq  \left| \mathcal{V}\right|$ .

引理3.1。如果$\alpha  = \mathop{\max }\limits_{{e \in  \mathcal{E}}}\left| e\right|$，那么$\alpha  \cdot  \rho \left( \mathcal{G}\right)  \geq  \left| \mathcal{V}\right|$。

Proof. Let $\mathcal{W}$ be a fractional edge covering of $\mathcal{G}$ with the minimum weight. Thus, $\alpha  \cdot  \rho \left( \mathcal{G}\right)  = \alpha \mathop{\sum }\limits_{{e \in  \mathcal{E}}}\mathcal{W}\left( e\right)  \geq  \mathop{\sum }\limits_{{e \in  \mathcal{E}}}\left| e\right| \mathcal{W}\left( e\right)  =$ $\mathop{\sum }\limits_{{X \in  \left| \mathcal{V}\right| }}\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}\mathcal{W}\left( e\right)  \geq  \mathop{\sum }\limits_{{X \in  \left| \mathcal{V}\right| }}1 = \left| \mathcal{V}\right| .$

证明。设$\mathcal{W}$为$\mathcal{G}$的具有最小权重的分数边覆盖。因此，$\alpha  \cdot  \rho \left( \mathcal{G}\right)  = \alpha \mathop{\sum }\limits_{{e \in  \mathcal{E}}}\mathcal{W}\left( e\right)  \geq  \mathop{\sum }\limits_{{e \in  \mathcal{E}}}\left| e\right| \mathcal{W}\left( e\right)  =$ $\mathop{\sum }\limits_{{X \in  \left| \mathcal{V}\right| }}\mathop{\sum }\limits_{{e \in  \mathcal{E} : X \in  e}}\mathcal{W}\left( e\right)  \geq  \mathop{\sum }\limits_{{X \in  \left| \mathcal{V}\right| }}1 = \left| \mathcal{V}\right| .$

Given a subset $\mathcal{U}$ of $\mathcal{V}$ ,we define the subgraph induced by $\mathcal{U}$ as the hypergraph $\left( {\mathcal{U},{\mathcal{E}}^{\prime }}\right)$ where

给定$\mathcal{V}$的一个子集$\mathcal{U}$，我们将由$\mathcal{U}$诱导的子图定义为超图$\left( {\mathcal{U},{\mathcal{E}}^{\prime }}\right)$，其中

$$
{\mathcal{E}}^{\prime } = \{ \mathcal{U} \cap  e \mid  e \in  \mathcal{E} \land  \mathcal{U} \cap  e \neq  \varnothing \} .
$$

### 3.2 Query Hypergraph and AGM Bound

### 3.2 查询超图与AGM界

We say that a query $\mathcal{Q}$ is clean if no two relations in $\mathcal{Q}$ share the same scheme. A clean $\mathcal{Q}$ defines a hypergraph $\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$ where $\mathcal{E} = \{$ scheme $\left( \mathcal{R}\right)  \mid  \mathcal{R} \in  \mathcal{Q}\}$ . For each edge $e \in  \mathcal{E}$ ,let ${\mathcal{R}}_{e}$ represent the relation $\mathcal{R} \in  \mathcal{Q}$ with $e =$ scheme(R).

我们称一个查询$\mathcal{Q}$是干净的，如果$\mathcal{Q}$中没有两个关系共享相同的模式。一个干净的$\mathcal{Q}$定义了一个超图$\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$，其中$\mathcal{E} = \{$ 模式 $\left( \mathcal{R}\right)  \mid  \mathcal{R} \in  \mathcal{Q}\}$。对于每条边$e \in  \mathcal{E}$，设${\mathcal{R}}_{e}$表示具有$e =$模式(R)的关系$\mathcal{R} \in  \mathcal{Q}$。

LEMMA 3.2 ([4]). Let $\mathcal{W}$ be a fractional edge covering of the hy-pergraph $\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$ defined by a clean query $\mathcal{Q}$ . Then, $\left| {\operatorname{Join}\left( \mathcal{Q}\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {\mathcal{R}}_{e}\right| }^{\mathcal{W}\left( e\right) }.$

引理3.2 ([4])。设$\mathcal{W}$是由一个干净查询$\mathcal{Q}$定义的超图$\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$的分数边覆盖。那么，$\left| {\operatorname{Join}\left( \mathcal{Q}\right) }\right|  \leq  \mathop{\prod }\limits_{{e \in  \mathcal{E}}}{\left| {\mathcal{R}}_{e}\right| }^{\mathcal{W}\left( e\right) }.$

The above is commonly known as the AGM bound.

上述内容通常被称为AGM界。

### 3.3 MPC Building Blocks

### 3.3 MPC构建模块

Cartesian products (CP). Consider a query $\mathcal{Q} = \left\{  {{\mathcal{R}}_{1},{\mathcal{R}}_{2},\ldots ,{\mathcal{R}}_{t}}\right\}$ with $t \geq  2$ where the relations have mutually disjoint schemes. Thus, $\operatorname{Join}\left( \mathcal{Q}\right)  = {\mathcal{R}}_{1} \times  {\mathcal{R}}_{2} \times  \ldots  \times  {\mathcal{R}}_{t}$ ; we will use ${CP}\left( \mathcal{Q}\right)$ as an alternative representation of this cartesian product.

笛卡尔积(CP)。考虑一个查询$\mathcal{Q} = \left\{  {{\mathcal{R}}_{1},{\mathcal{R}}_{2},\ldots ,{\mathcal{R}}_{t}}\right\}$，其中$t \geq  2$，且这些关系具有互不相交的模式。因此，$\operatorname{Join}\left( \mathcal{Q}\right)  = {\mathcal{R}}_{1} \times  {\mathcal{R}}_{2} \times  \ldots  \times  {\mathcal{R}}_{t}$；我们将使用${CP}\left( \mathcal{Q}\right)$作为这个笛卡尔积的另一种表示。

LEMMA 3.3 ([13]). We can compute ${CP}\left( \mathcal{Q}\right)$ with load

引理3.3 ([13])。我们可以以负载计算${CP}\left( \mathcal{Q}\right)$

$$
O\left( {\mathop{\max }\limits_{{\text{non-empty }{\mathcal{Q}}^{\prime } \subseteq  \mathcal{Q}}}\frac{{\left| \operatorname{CP}\left( {\mathcal{Q}}^{\prime }\right) \right| }^{\frac{1}{\left| {\mathcal{Q}}^{\prime }\right| }}}{{p}^{\frac{1}{\left| {\mathcal{Q}}^{\prime }\right| }}}}\right)  \tag{10}
$$

using p machines.

使用p台机器。

LEMMA 3.4 $\left( \left\lbrack  {{12},{13}}\right\rbrack  \right)$ . ${\mathcal{Q}}_{1}$ and ${\mathcal{Q}}_{2}$ are queries whose input sizes are bounded by $N$ . If

引理3.4 $\left( \left\lbrack  {{12},{13}}\right\rbrack  \right)$。${\mathcal{Q}}_{1}$和${\mathcal{Q}}_{2}$是输入大小受$N$限制的查询。如果

- Join $\left( {\mathcal{Q}}_{1}\right)$ can be computed with load $\widetilde{O}\left( {N/{p}_{1}^{1/{t}_{1}}}\right)$ using ${p}_{1}$ machines and

- 可以使用 ${p}_{1}$ 台机器以负载 $\widetilde{O}\left( {N/{p}_{1}^{1/{t}_{1}}}\right)$ 来计算连接 $\left( {\mathcal{Q}}_{1}\right)$

- Join $\left( {\mathcal{Q}}_{2}\right)$ with load $\widetilde{O}\left( {N/{p}_{2}^{1/{t}_{2}}}\right)$ using ${p}_{2}$ machines

- 使用 ${p}_{2}$ 台机器以负载 $\widetilde{O}\left( {N/{p}_{2}^{1/{t}_{2}}}\right)$ 进行连接 $\left( {\mathcal{Q}}_{2}\right)$

then we can compute $\mathcal{J}$ oin $\left( {\mathcal{Q}}_{1}\right)  \times  \mathcal{J}$ oin $\left( {\mathcal{Q}}_{2}\right)$ using ${p}_{1}{p}_{2}$ machines with $\operatorname{load}\widetilde{O}\left( {N/\min \left\{  {{p}^{1/{t}_{1}},{p}^{1/{t}_{2}}}\right\}  }\right)$ .

然后我们可以使用 ${p}_{1}{p}_{2}$ 台机器以 $\operatorname{load}\widetilde{O}\left( {N/\min \left\{  {{p}^{1/{t}_{1}},{p}^{1/{t}_{2}}}\right\}  }\right)$ 计算 $\mathcal{J}$ 连接 $\left( {\mathcal{Q}}_{1}\right)  \times  \mathcal{J}$ 连接 $\left( {\mathcal{Q}}_{2}\right)$。

Skew-free queries. Suppose that we have assigned a share ${p}_{A}$ to every attribute $A \in$ attset(Q)subject to the condition in (5). In Appendix A, we prove:

无倾斜查询。假设我们已根据 (5) 中的条件为每个属性 $A \in$∈属性集(Q)分配了一个份额 ${p}_{A}$。在附录 A 中，我们证明：

LEMMA 3.5. When $\mathcal{Q}$ is two-attribute skew free (Section 2),the BinHC algorithm of [6] answers 2 with a load not exceeding (8), repeated here for the reader's convenience:

引理 3.5。当 $\mathcal{Q}$ 是无两属性倾斜时（第 2 节），文献 [6] 中的 BinHC 算法以不超过 (8) 的负载回答 Q2，为方便读者，此处重复如下：

$$
\widetilde{O}\left( {\mathop{\max }\limits_{{\mathcal{R} \in  \mathcal{Q}}}\left( {\mathop{\min }\limits_{{\mathcal{V} \subseteq  \text{ scheme }\left( \mathcal{R}\right) }}\frac{n}{\mathop{\prod }\limits_{{A \in  \mathcal{V}}}{p}_{A}}}\right) }\right) .
$$

## 4 GENERALIZED VERTEX PACKINGS

## 4 广义顶点覆盖

Let $\mathcal{F}$ be a function mapping $\mathcal{V}$ to the values in $( - \infty ,1\rbrack$ (note that the output of $\mathcal{F}$ can be negative). Define the weight of $\mathcal{F}$ as $\mathop{\sum }\limits_{{X \in  \mathcal{V}}}\mathcal{F}\left( X\right)$ . For each edge $e \in  \mathcal{E}$ ,we call $\mathop{\sum }\limits_{{X \in  e}}\mathcal{F}\left( X\right)$ the weight of $e$ (under $\mathcal{F}$ ).

设 $\mathcal{F}$ 是一个将 $\mathcal{V}$ 映射到 $( - \infty ,1\rbrack$ 中值的函数（注意 $\mathcal{F}$ 的输出可以为负）。将 $\mathcal{F}$ 的权重定义为 $\mathop{\sum }\limits_{{X \in  \mathcal{V}}}\mathcal{F}\left( X\right)$。对于每条边 $e \in  \mathcal{E}$，我们称 $\mathop{\sum }\limits_{{X \in  e}}\mathcal{F}\left( X\right)$ 为 $e$ 的权重（在 $\mathcal{F}$ 下）。

We say that $\mathcal{F}$ is a generalized vertex packing of $\mathcal{G}$ if the weight of every edge $e \in  \mathcal{E}$ is at most 1 . The generalized vertex-packing number of $\mathcal{G} -$ denoted as $\phi \left( \mathcal{G}\right)  -$ is the maximum weight of all the generalized vertex packings of $\mathcal{G}$ .

我们称 $\mathcal{F}$ 是 $\mathcal{G}$ 的一个广义顶点覆盖，如果每条边 $e \in  \mathcal{E}$ 的权重至多为 1。$\mathcal{G} -$ 的广义顶点覆盖数记为 $\phi \left( \mathcal{G}\right)  -$，它是 $\mathcal{G}$ 的所有广义顶点覆盖的最大权重。

Example. The hypergraph $\mathcal{G}$ in Figure 1(a) has a generalized vertex packing number $\phi \left( \mathcal{G}\right)  = 5$ ,as is given by the function $\mathcal{F}$ that maps $\mathrm{B}$ to $- 1,\mathrm{D},\mathrm{E},\mathrm{G}$ ,and $\mathrm{H}$ to 0,and the other vertices to 1 .

示例。图 1(a) 中的超图 $\mathcal{G}$ 具有广义顶点覆盖数 $\phi \left( \mathcal{G}\right)  = 5$，这由函数 $\mathcal{F}$ 给出，该函数将 $\mathrm{B}$ 映射到 $- 1,\mathrm{D},\mathrm{E},\mathrm{G}$，将 $\mathrm{H}$ 映射到 0，并将其他顶点映射到 1。

Next, we will prove several properties that will be useful to our analysis. Our discussion will revolve around a special linear program about $\mathcal{G}$ :

接下来，我们将证明几个对我们的分析有用的性质。我们的讨论将围绕一个关于$\mathcal{G}$的特殊线性规划展开：

<!-- Media -->

---

		The characterizing program of $\mathcal{G}$

		$\mathcal{G}$的特征规划

Variables: ${x}_{e}$ for each $e \in  \mathcal{E}$

变量：对于每个$e \in  \mathcal{E}$，有${x}_{e}$

Maximize: $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}{x}_{e}\left( {\left| e\right|  - 1}\right)$ subject to

最大化：$\mathop{\sum }\limits_{{e \in  \mathcal{E}}}{x}_{e}\left( {\left| e\right|  - 1}\right)$，约束条件为

	- for each $A \in  \mathcal{V},\mathop{\sum }\limits_{{e \in  \mathcal{E} : A \in  e}}{x}_{e} \leq  1$ ,and

	-  - 对于每个$A \in  \mathcal{V},\mathop{\sum }\limits_{{e \in  \mathcal{E} : A \in  e}}{x}_{e} \leq  1$，以及

	- for each $e \in  \mathcal{E},{x}_{e} \geq  0$ .

	-  - 对于每个$e \in  \mathcal{E},{x}_{e} \geq  0$。

---

The above program is always feasible (e.g.,setting ${x}_{e} = 0$ for all $e \in  \mathcal{E}$ ),and is always bounded (the first bullet implies ${x}_{e} \leq  1$ ). It thus has an optimal solution,which we denote as $\bar{\phi }\left( \mathcal{G}\right)$ .

上述规划总是可行的（例如，对所有$e \in  \mathcal{E}$设置${x}_{e} = 0$），并且总是有界的（第一个要点意味着${x}_{e} \leq  1$）。因此，它有一个最优解，我们将其记为$\bar{\phi }\left( \mathcal{G}\right)$。

<!-- Media -->

Example. For the hypergraph $\mathcal{G}$ in Figure 1(a),an optimal solution to the characterizing program sets ${x}_{e}$ to 1 for $e = \{ \mathrm{A},\mathrm{B},\mathrm{C}\} ,\{ \mathrm{F},\mathrm{G},\mathrm{H}\}$ , $\{ \mathrm{D},\mathrm{K}\}$ ,and $\{ \mathrm{E},\mathrm{I}\}$ ,and to 0 for the other edges. This assignment achieves the value $\bar{\phi }\left( \mathcal{G}\right)  = 6$ for the objective function.

示例。对于图1(a)中的超图$\mathcal{G}$，特征规划的一个最优解将${x}_{e}$在$e = \{ \mathrm{A},\mathrm{B},\mathrm{C}\} ,\{ \mathrm{F},\mathrm{G},\mathrm{H}\}$、$\{ \mathrm{D},\mathrm{K}\}$和$\{ \mathrm{E},\mathrm{I}\}$时设为1，在其他边上设为0。这个赋值使得目标函数的值达到$\bar{\phi }\left( \mathcal{G}\right)  = 6$。

Lemma 4.1. $\phi \left( \mathcal{G}\right)  + \bar{\phi }\left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$ .

引理4.1。$\phi \left( \mathcal{G}\right)  + \bar{\phi }\left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$。

Proof. Consider the dual of the characterizing program:

证明。考虑特征规划的对偶规划：

---

Variables: ${y}_{A}$ for each $A \in  \mathcal{V}$

变量：对于每个$A \in  \mathcal{V}$，有${y}_{A}$

Minimize: $\mathop{\sum }\limits_{{A \in  \mathcal{V}}}{y}_{A}$ subject to

最小化：$\mathop{\sum }\limits_{{A \in  \mathcal{V}}}{y}_{A}$，约束条件为

	- for each $e \in  \mathcal{E},\mathop{\sum }\limits_{{A \in  e}}{y}_{A} \geq  \left| e\right|  - 1$ ,and

	-  - 对于每个$e \in  \mathcal{E},\mathop{\sum }\limits_{{A \in  e}}{y}_{A} \geq  \left| e\right|  - 1$，以及

	- for each $A \in  \mathcal{V},{y}_{A} \geq  0$ .

	-  - 对于每个$A \in  \mathcal{V},{y}_{A} \geq  0$。

---

By the duality of linear programming (LP), the optimal value of the objective function in the dual program is identical to that in the characterizing program.

根据线性规划（LP）的对偶性，对偶规划中目标函数的最优值与特征规划中的相同。

Let $\mathcal{F}$ be a generalized vertex packing of $\mathcal{G}$ of the maximum weight. Define ${y}_{A} = 1 - \mathcal{F}\left( A\right)$ for each $A \in  \mathcal{V}$ . We will prove that the assignment $\left\{  {{y}_{A} \mid  A \in  \mathcal{V}}\right\}$ must be an optimal solution of the dual program,i.e., $\bar{\phi }\left( \mathcal{G}\right)  = \mathop{\sum }\limits_{A}{y}_{A}$ .

设$\mathcal{F}$是$\mathcal{G}$的具有最大权重的广义顶点覆盖。对于每个$A \in  \mathcal{V}$，定义${y}_{A} = 1 - \mathcal{F}\left( A\right)$。我们将证明赋值$\left\{  {{y}_{A} \mid  A \in  \mathcal{V}}\right\}$一定是对偶规划的一个最优解，即$\bar{\phi }\left( \mathcal{G}\right)  = \mathop{\sum }\limits_{A}{y}_{A}$。

For each $e \in  \mathcal{E},{\sum }_{A \in  e}{y}_{A} = {\sum }_{A \in  e}\left( {1 - \mathcal{F}\left( A\right) }\right)  = \left| e\right|  - \mathop{\sum }\limits_{{A \in  e}}\mathcal{F}\left( A\right)  \geq$ $\left| e\right|  - 1$ . This,together with the fact that ${y}_{A} \geq  0$ for all $A \in  \mathcal{V}$ , indicates that $\left\{  {{y}_{A} \mid  A \in  \mathcal{V}}\right\}$ is a feasible assignment (for the dual).

对于每个 $e \in  \mathcal{E},{\sum }_{A \in  e}{y}_{A} = {\sum }_{A \in  e}\left( {1 - \mathcal{F}\left( A\right) }\right)  = \left| e\right|  - \mathop{\sum }\limits_{{A \in  e}}\mathcal{F}\left( A\right)  \geq$ $\left| e\right|  - 1$ 。这一点，再加上对于所有 $A \in  \mathcal{V}$ 都有 ${y}_{A} \geq  0$ 这一事实，表明 $\left\{  {{y}_{A} \mid  A \in  \mathcal{V}}\right\}$ 是一个可行赋值（对于对偶问题而言）。

Suppose that there is another a feasible assignment $\left\{  {{y}_{A}^{\prime } \mid  }\right.$ $A \in  \mathcal{V}\}$ able to make the objective function even lower,namely, $\mathop{\sum }\limits_{A}{y}_{A}^{\prime } < \mathop{\sum }\limits_{A}{y}_{A}$ . Let us design a function ${\mathcal{F}}^{\prime }$ by setting ${\mathcal{F}}^{\prime }\left( A\right)  =$ $1 - {y}_{A}^{\prime }$ for each $A \in  \mathcal{V}$ . Clearly, ${\mathcal{F}}^{\prime }\left( A\right)  \leq  1$ for every $A \in  \mathcal{V}$ . Furthermore,for each $e \in  \mathcal{E},\mathop{\sum }\limits_{{A \in  e}}{\mathcal{F}}^{\prime }\left( A\right)  = {\sum }_{A \in  e}\left( {1 - {y}_{A}^{\prime }}\right)  =$ $\left| e\right|  - {\sum }_{A \in  e}{y}_{A}^{\prime } \leq  1$ . Therefore, ${\mathcal{F}}^{\prime }$ is a generalized vertex packing. However, $\mathop{\sum }\limits_{A}{\mathcal{F}}^{\prime }\left( A\right)  = \mathop{\sum }\limits_{A}\left( {1 - {y}_{A}^{\prime }}\right)  > \mathop{\sum }\limits_{A}\left( {1 - {y}_{A}}\right)  = \mathop{\sum }\limits_{A}\mathcal{F}\left( A\right)$ , which contradicts the definition of $\mathcal{F}$ .

假设存在另一个可行赋值 $\left\{  {{y}_{A}^{\prime } \mid  }\right.$ $A \in  \mathcal{V}\}$ 能够使目标函数值更低，即 $\mathop{\sum }\limits_{A}{y}_{A}^{\prime } < \mathop{\sum }\limits_{A}{y}_{A}$ 。我们通过为每个 $A \in  \mathcal{V}$ 设置 ${\mathcal{F}}^{\prime }\left( A\right)  =$ $1 - {y}_{A}^{\prime }$ 来定义一个函数 ${\mathcal{F}}^{\prime }$ 。显然，对于每个 $A \in  \mathcal{V}$ 都有 ${\mathcal{F}}^{\prime }\left( A\right)  \leq  1$ 。此外，对于每个 $e \in  \mathcal{E},\mathop{\sum }\limits_{{A \in  e}}{\mathcal{F}}^{\prime }\left( A\right)  = {\sum }_{A \in  e}\left( {1 - {y}_{A}^{\prime }}\right)  =$ $\left| e\right|  - {\sum }_{A \in  e}{y}_{A}^{\prime } \leq  1$ 。因此， ${\mathcal{F}}^{\prime }$ 是一个广义顶点覆盖。然而， $\mathop{\sum }\limits_{A}{\mathcal{F}}^{\prime }\left( A\right)  = \mathop{\sum }\limits_{A}\left( {1 - {y}_{A}^{\prime }}\right)  > \mathop{\sum }\limits_{A}\left( {1 - {y}_{A}}\right)  = \mathop{\sum }\limits_{A}\mathcal{F}\left( A\right)$ ，这与 $\mathcal{F}$ 的定义相矛盾。

It follows from the above discussion that $\bar{\phi }\left( \mathcal{G}\right)  = \mathop{\sum }\limits_{A}{y}_{A} =$ $\mathop{\sum }\limits_{A}\left( {1 - \mathcal{F}\left( A\right) }\right)  = \left| \mathcal{V}\right|  - \mathop{\sum }\limits_{A}\mathcal{F}\left( A\right)  = \left| \mathcal{V}\right|  - \phi \left( \mathcal{G}\right)$ ,which establishes the lemma.

由上述讨论可知 $\bar{\phi }\left( \mathcal{G}\right)  = \mathop{\sum }\limits_{A}{y}_{A} =$ $\mathop{\sum }\limits_{A}\left( {1 - \mathcal{F}\left( A\right) }\right)  = \left| \mathcal{V}\right|  - \mathop{\sum }\limits_{A}\mathcal{F}\left( A\right)  = \left| \mathcal{V}\right|  - \phi \left( \mathcal{G}\right)$ ，这就证明了该引理。

LEMMA 4.2. If every edge in $\mathcal{G}$ contains two vertices, $\phi \left( \mathcal{G}\right)  = \rho \left( \mathcal{G}\right)$ .

引理4.2。如果 $\mathcal{G}$ 中的每条边都包含两个顶点，则 $\phi \left( \mathcal{G}\right)  = \rho \left( \mathcal{G}\right)$ 。

Proof. When $\left| e\right|  = 2$ for every $e \in  \mathcal{E},\bar{\phi }\left( \mathcal{G}\right)$ is exactly the fractional edge packing number $\tau \left( \mathcal{G}\right)$ (Section 3.1). As proved in Theorem 2.2.7 of [19],such a $\mathcal{G}$ must have the property that $\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$ . The lemma thus follows from Lemma 4.1.

证明。当对于每个$e \in  \mathcal{E},\bar{\phi }\left( \mathcal{G}\right)$都有$\left| e\right|  = 2$时，其恰好是分数边填充数$\tau \left( \mathcal{G}\right)$（3.1节）。正如文献[19]的定理2.2.7所证明的，这样的$\mathcal{G}$必定具有$\rho \left( \mathcal{G}\right)  + \tau \left( \mathcal{G}\right)  = \left| \mathcal{V}\right|$的性质。因此，该引理可由引理4.1推出。

LEMMA 4.3. If $\mathcal{G}$ is the hypergraph defined by a symmetric query $\mathcal{Q},\phi \left( \mathcal{G}\right)  = k/\alpha$ ,where $\alpha$ and $k$ are given in (1) and (2),respectively.

引理4.3。若$\mathcal{G}$是由对称查询$\mathcal{Q},\phi \left( \mathcal{G}\right)  = k/\alpha$定义的超图，其中$\alpha$和$k$分别由式(1)和式(2)给出。

Proof. We will prove:

证明。我们将证明：

$$
\rho \left( \mathcal{G}\right)  + \bar{\phi }\left( \mathcal{G}\right)  \leq  k \tag{11}
$$

which will establish the lemma by the following argument. Assume without loss of generality that every vertex in $\mathcal{G}$ is covered by $c$ edges. The number of edges in $\mathcal{G}$ must be ${ck}/\alpha$ . Consider the function $\mathcal{W}$ that maps each edge to $1/c.\mathcal{W}$ is clearly a fractional edge covering with weight $k/\alpha$ . On the other hand,Lemma 3.1 tells us $\rho \left( \mathcal{G}\right)  \geq  k/\alpha$ ,which leads to $\rho \left( \mathcal{G}\right)  = k/\alpha$ . It follows from (11) that $\bar{\phi }\left( \mathcal{G}\right)  \leq  k\left( {1 - \frac{1}{\alpha }}\right)$ . On the other hand,consider the following assignment to the variables in the characterizing program: $\left\{  {{x}_{e} = }\right.$ $1/c \mid  e \in  \mathcal{E}\}$ . This assignment satisfies all constraints,and achieves $\mathop{\sum }\limits_{{e \in  \mathcal{E}}}{x}_{e}\left( {\left| e\right|  - 1}\right)  = \frac{ck}{\alpha }\frac{1}{c}\left( {\alpha  - 1}\right)  = k\left( {1 - \frac{1}{\alpha }}\right)$ . This implies $\bar{\phi }\left( \mathcal{G}\right)  =$ $k\left( {1 - \frac{1}{\alpha }}\right)$ ,which by Lemma 4.1 leads to $\phi \left( \mathcal{G}\right)  = k/\alpha$ .

通过以下论证可确立该引理。不失一般性，假设$\mathcal{G}$中的每个顶点都被$c$条边覆盖。$\mathcal{G}$中的边数必定为${ck}/\alpha$。考虑将每条边映射到$1/c.\mathcal{W}$的函数$\mathcal{W}$，显然这是一个权重为$k/\alpha$的分数边覆盖。另一方面，引理3.1告诉我们$\rho \left( \mathcal{G}\right)  \geq  k/\alpha$，由此可得$\rho \left( \mathcal{G}\right)  = k/\alpha$。由式(11)可知$\bar{\phi }\left( \mathcal{G}\right)  \leq  k\left( {1 - \frac{1}{\alpha }}\right)$。另一方面，考虑对特征化程序中的变量进行如下赋值：$\left\{  {{x}_{e} = }\right.$ $1/c \mid  e \in  \mathcal{E}\}$。该赋值满足所有约束条件，并达到$\mathop{\sum }\limits_{{e \in  \mathcal{E}}}{x}_{e}\left( {\left| e\right|  - 1}\right)  = \frac{ck}{\alpha }\frac{1}{c}\left( {\alpha  - 1}\right)  = k\left( {1 - \frac{1}{\alpha }}\right)$。这意味着$\bar{\phi }\left( \mathcal{G}\right)  =$ $k\left( {1 - \frac{1}{\alpha }}\right)$，根据引理4.1可推出$\phi \left( \mathcal{G}\right)  = k/\alpha$。

It remains to prove (11), which (by Lemma 4.1) is equivalent to showing $\rho \left( \mathcal{G}\right)  \leq  \phi \left( \mathcal{G}\right)$ . For this purpose,let us recall two standard concepts from the fractional graph theory [19]. Let ${\mathcal{F}}^{\prime }$ be a function that maps $\mathcal{V}$ to the set of real values in $\left\lbrack  {0,1}\right\rbrack  .{\mathcal{F}}^{\prime }$ is a fractional vertex packing of $\mathcal{G}$ if $\mathop{\sum }\limits_{{A \in  e}}{\mathcal{F}}^{\prime }\left( A\right)  \leq  1$ for every $e \in  \mathcal{E}$ . The fractional vertex-packing number of $\mathcal{G}$ is the maximum $\mathop{\sum }\limits_{{A \in  \mathcal{V}}}{\mathcal{F}}^{\prime }\left( A\right)$ of all the fractional vertex packings ${\mathcal{F}}^{\prime }$ . By the duality of LP,the fractional vertex-packing number is precisely $\rho \left( \mathcal{G}\right)$ . The inequality $\rho \left( \mathcal{G}\right)  \leq  \phi \left( \mathcal{G}\right)$ follows from the definition of $\phi \left( \mathcal{G}\right)$ (at the beginning of Section 4), and the fact that any fractional vertex packing is a generalized vertex packing.

接下来需要证明 (11)，根据引理 4.1，这等价于证明 $\rho \left( \mathcal{G}\right)  \leq  \phi \left( \mathcal{G}\right)$。为此，让我们回顾分数图论 [19] 中的两个标准概念。设 ${\mathcal{F}}^{\prime }$ 是一个将 $\mathcal{V}$ 映射到 $\left\lbrack  {0,1}\right\rbrack  .{\mathcal{F}}^{\prime }$ 中实数值集合的函数，如果对于每个 $e \in  \mathcal{E}$ 都有 $\mathop{\sum }\limits_{{A \in  e}}{\mathcal{F}}^{\prime }\left( A\right)  \leq  1$，则 ${\mathcal{F}}^{\prime }$ 是 $\mathcal{G}$ 的一个分数顶点填充（fractional vertex packing）。$\mathcal{G}$ 的分数顶点填充数是所有分数顶点填充 ${\mathcal{F}}^{\prime }$ 的最大值 $\mathop{\sum }\limits_{{A \in  \mathcal{V}}}{\mathcal{F}}^{\prime }\left( A\right)$。根据线性规划（LP）的对偶性，分数顶点填充数恰好是 $\rho \left( \mathcal{G}\right)$。不等式 $\rho \left( \mathcal{G}\right)  \leq  \phi \left( \mathcal{G}\right)$ 由 $\phi \left( \mathcal{G}\right)$ 的定义（在第 4 节开头）以及任何分数顶点填充都是广义顶点填充这一事实得出。

## 5 2-ATTRIBUTE HEAVY-LIGHT TAXONOMY

## 5 双属性重 - 轻分类法

In Sections 5-7,we consider that the input query $\mathcal{Q}$ is (i) clean (Section 3.2),and (ii) unary-free,namely,each relation in $\mathcal{Q}$ has arity at least 2 .

在第 5 - 7 节中，我们假设输入查询 $\mathcal{Q}$ 满足：(i) 无噪声（第 3.2 节）；(ii) 无一元关系，即 $\mathcal{Q}$ 中的每个关系的元数至少为 2。

Denote by $\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$ the hypergraph defined by $\mathcal{Q}$ (Section 3.2). Set $\lambda$ to an arbitrary positive real value. In the manner explained in Section 2,each value $x \in  \mathbf{{dom}}$ can be classified into light/heavy,and so can each value pair $\left( {x,y}\right)  \in  \mathbf{{dom}} \times  \mathbf{{dom}}$ .

用 $\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$ 表示由 $\mathcal{Q}$ 定义的超图（第 3.2 节）。将 $\lambda$ 设为任意正实数值。按照第 2 节中解释的方式，每个值 $x \in  \mathbf{{dom}}$ 可以被分类为轻或重，每个值对 $\left( {x,y}\right)  \in  \mathbf{{dom}} \times  \mathbf{{dom}}$ 也可以如此分类。

Configurations. Consider a plan $P$ as defined in (9). As before, set $\mathcal{H} = \left\{  {{X}_{1},\ldots ,{X}_{a},{Y}_{1},\ldots ,{Y}_{b},{Z}_{1},\ldots ,{Z}_{b}}\right\}$ for some $a,b \geq  0$ . Given a $\mathcal{U} \subseteq  \mathcal{H}$ and a tuple $\mathbf{u}$ over $\mathcal{U}$ ,we say that(U,u)is a $\mathcal{U}$ -configuration of $\mathbf{P}$ if

配置。考虑如 (9) 中所定义的计划 $P$。和之前一样，对于某个 $a,b \geq  0$，设 $\mathcal{H} = \left\{  {{X}_{1},\ldots ,{X}_{a},{Y}_{1},\ldots ,{Y}_{b},{Z}_{1},\ldots ,{Z}_{b}}\right\}$。给定一个 $\mathcal{U} \subseteq  \mathcal{H}$ 和一个在 $\mathcal{U}$ 上的元组 $\mathbf{u}$，如果满足以下条件，我们称 (U, u) 是 $\mathbf{P}$ 的一个 $\mathcal{U}$ - 配置：

- $\mathbf{u}\left( {X}_{i}\right)$ is heavy for every ${X}_{i} \in  \mathcal{U},i \in  \left\lbrack  a\right\rbrack$ ;

- 对于每个 ${X}_{i} \in  \mathcal{U},i \in  \left\lbrack  a\right\rbrack$，$\mathbf{u}\left( {X}_{i}\right)$ 是重的；

- $\mathbf{u}\left( {Y}_{j}\right)$ is light for every ${Y}_{j} \in  \mathcal{U},j \in  \left\lbrack  b\right\rbrack$ ;

- 对于每个 ${Y}_{j} \in  \mathcal{U},j \in  \left\lbrack  b\right\rbrack$，$\mathbf{u}\left( {Y}_{j}\right)$ 是轻的；

- $\mathbf{u}\left( {Z}_{j}\right)$ is light for every ${Z}_{j} \in  \mathcal{U},j \in  \left\lbrack  b\right\rbrack$ ;

- 对于每个 ${Z}_{j} \in  \mathcal{U},j \in  \left\lbrack  b\right\rbrack$，$\mathbf{u}\left( {Z}_{j}\right)$ 是轻的；

- $\left( {\mathbf{u}\left( {Y}_{j}\right) ,\mathbf{u}\left( {Z}_{j}\right) }\right)$ is heavy for every $\left( {{Y}_{j},{Z}_{j}}\right)  \in  \mathcal{U} \times  \mathcal{U},j \in  \left\lbrack  b\right\rbrack$ (if either ${Y}_{j} \notin  U$ or ${Z}_{j} \notin  U$ ,this condition does not apply to $j$ ). When $\mathcal{U} = \mathcal{H},\left( {\mathcal{U},\mathbf{u}}\right)$ is a full configuration of $P$ .

- 对于每个 $\left( {{Y}_{j},{Z}_{j}}\right)  \in  \mathcal{U} \times  \mathcal{U},j \in  \left\lbrack  b\right\rbrack$，$\left( {\mathbf{u}\left( {Y}_{j}\right) ,\mathbf{u}\left( {Z}_{j}\right) }\right)$ 是重的（如果 ${Y}_{j} \notin  U$ 或 ${Z}_{j} \notin  U$，则此条件不适用于 $j$）。当 $\mathcal{U} = \mathcal{H},\left( {\mathcal{U},\mathbf{u}}\right)$ 是 $P$ 的一个完整配置时。

Proposition 5.1. P has at most ${\lambda }^{\left| \mathcal{H}\right| }$ full configurations.

命题 5.1. P 至多有 ${\lambda }^{\left| \mathcal{H}\right| }$ 个完整配置。

Proof. There can be at most $\lambda$ values on each ${X}_{i}\left( {i \in  \left\lbrack  a\right\rbrack  }\right)$ ,and at most ${\lambda }^{2}$ value pairs on each $\left( {{Y}_{j},{Z}_{j}}\right) \left( {j \in  \left\lbrack  b\right\rbrack  }\right)$ . Hence,the number of full configurations is at most ${\lambda }^{a} \cdot  {\left( {\lambda }^{2}\right) }^{b} = {\lambda }^{\left| \mathcal{H}\right| }$ .

证明。每个 ${X}_{i}\left( {i \in  \left\lbrack  a\right\rbrack  }\right)$ 上至多有 $\lambda$ 个值，每个 $\left( {{Y}_{j},{Z}_{j}}\right) \left( {j \in  \left\lbrack  b\right\rbrack  }\right)$ 上至多有 ${\lambda }^{2}$ 个值对。因此，完整配置的数量至多为 ${\lambda }^{a} \cdot  {\left( {\lambda }^{2}\right) }^{b} = {\lambda }^{\left| \mathcal{H}\right| }$。

Residual queries. Given a full configuration(H,h)of plan $\mathbf{P}$ ,we will formulate a residual query ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ which returns exactly the set of tuples $\mathbf{u} \in  \mathcal{J}$ oin(Q)"consistent" with(H,h),meaning that:

剩余查询。给定计划 $\mathbf{P}$ 的一个完整配置 (H, h)，我们将构造一个剩余查询 ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$，它恰好返回与 (H, h) “一致” 的元组集合 $\mathbf{u} \in  \mathcal{J}$，意思是：

- $\mathbf{u}\left( A\right)  = \mathbf{h}\left( A\right)$ for each $A \in  \mathcal{H}$ ;

- 对于每个 $A \in  \mathcal{H}$，$\mathbf{u}\left( A\right)  = \mathbf{h}\left( A\right)$；

- $\mathbf{u}\left( A\right)$ is light for any attribute $A \notin  \mathcal{H}$ ; - $\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ is light for any distinct attributes $A,B \notin  \mathcal{H}$ .

- 对于任何属性 $A \notin  \mathcal{H}$，$\mathbf{u}\left( A\right)$ 是轻的； - 对于任何不同的属性 $A,B \notin  \mathcal{H}$，$\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ 是轻的。

Formally,an edge $e \in  \mathcal{E}$ is active on $P$ if $e$ has at least one attribute outside $\mathcal{H}$ . For an active edge $e$ ,let ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ be the residual relation of $e$ that fulfills all the conditions below:

形式上，如果 $e$ 至少有一个属性不在 $\mathcal{H}$ 中，则边 $e \in  \mathcal{E}$ 在 $P$ 上是活跃的。对于一个活跃边 $e$，令 ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 为满足以下所有条件的 $e$ 的剩余关系：

- ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is over ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ ;

- ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 基于 ${e}^{\prime } = e \smallsetminus  \mathcal{H}$；

- ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ collects the projection $\mathbf{v}\left\lbrack  {e}^{\prime }\right\rbrack$ of all the tuples $\mathbf{v} \in$

- ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 收集所有元组 $\mathbf{v} \in$ 的投影 $\mathbf{v}\left\lbrack  {e}^{\prime }\right\rbrack$

${\mathcal{R}}_{e}$ (see Section 3.2 for the definition of ${\mathcal{R}}_{e}$ ) satisfying

${\mathcal{R}}_{e}$（${\mathcal{R}}_{e}$ 的定义见第 3.2 节）满足

$$
 - \mathbf{v}\left( A\right)  = \mathbf{h}\left( A\right) \text{for each}A \in  e \cap  \mathcal{H}\text{;}
$$

$$
\text{-}v\left( A\right) \text{is light for each}A \in  {e}^{\prime }\text{;}
$$

- $\left( {\mathbf{v}\left( A\right) ,\mathbf{v}\left( B\right) }\right)$ is light for any distinct $A,B \in  {e}^{\prime }$ .

- 对于任何不同的 $A,B \in  {e}^{\prime }$，$\left( {\mathbf{v}\left( A\right) ,\mathbf{v}\left( B\right) }\right)$ 是轻的。

We can now formulate the residual query as:

我们现在可以将剩余查询表述为：

$$
{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)  = \left\{  {{\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)  \mid  e \in  \mathcal{E},e\text{ active on }P}\right\}  . \tag{12}
$$

Example. Consider the query $\mathcal{Q}$ in Figure 1(a) and the plan $P =$ $\left( {\{ \mathrm{D}\} ,\{ \left( {\mathrm{G},\mathrm{H}}\right) \} }\right)$ . Hence, $\mathcal{H} = \{ \mathrm{D},\mathrm{G},\mathrm{H}\}$ . Consider the full configuration (H,h)where $\mathbf{h} = \left( {\mathrm{d},\mathrm{g},\mathrm{h}}\right)$ .

示例。考虑图1(a)中的查询$\mathcal{Q}$以及计划$P =$ $\left( {\{ \mathrm{D}\} ,\{ \left( {\mathrm{G},\mathrm{H}}\right) \} }\right)$。因此，$\mathcal{H} = \{ \mathrm{D},\mathrm{G},\mathrm{H}\}$。考虑完整配置(H,h)，其中$\mathbf{h} = \left( {\mathrm{d},\mathrm{g},\mathrm{h}}\right)$。

All edges of the graph in Figure 1(a) are active except $\{ \mathrm{D},\mathrm{H}\}$ . Set $e = \{ \mathrm{G},\mathrm{J}\}$ ; thus ${e}^{\prime } = \{ \mathrm{J}\}$ . The residual relation ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ collects all tuples of the form $\left( {\mathrm{g},x}\right)  \in  {\mathcal{R}}_{e}$ where $x$ is light. For another example, set $e = \{ \mathrm{A},\mathrm{B},\mathrm{C}\}$ ; thus ${e}^{\prime } = e$ . The residual relation ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ collects all tuples $\left( {x,y,z}\right)  \in  {\mathcal{R}}_{e}$ such that the values $x,y,z$ and the value pairs $\left( {x,y}\right) ,\left( {x,z}\right) ,\left( {y,z}\right)$ are all light.

图1(a)中除$\{ \mathrm{D},\mathrm{H}\}$之外的图的所有边都是活跃的。设$e = \{ \mathrm{G},\mathrm{J}\}$；因此${e}^{\prime } = \{ \mathrm{J}\}$。残差关系${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$收集所有形式为$\left( {\mathrm{g},x}\right)  \in  {\mathcal{R}}_{e}$的元组，其中$x$是轻的。再举一个例子，设$e = \{ \mathrm{A},\mathrm{B},\mathrm{C}\}$；因此${e}^{\prime } = e$。残差关系${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$收集所有满足值$x,y,z$和值对$\left( {x,y}\right) ,\left( {x,z}\right) ,\left( {y,z}\right)$都是轻的元组$\left( {x,y,z}\right)  \in  {\mathcal{R}}_{e}$。

The next lemma, proved in Appendix B, shows that we can find the entire $\mathcal{J}$ oin(Q)by answering each residual query correctly:

附录B中证明的下一个引理表明，我们可以通过正确回答每个残差查询来找到整个$\mathcal{J}$连接(Q)：

LEMMA 5.2.

引理5.2。

$$
\text{Join}\left( \mathcal{Q}\right)  = \mathop{\bigcup }\limits_{P}\left( {\mathop{\bigcup }\limits_{{\text{full config. }\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }P}}{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) \times \{ \mathbf{h}\} }\right) \text{.} \tag{13}
$$

Completing a configuration. Consider a $\mathcal{U}$ -configuration of $\mathbf{P}$ : (U,u). We say that a full configuration(H,h)of $\mathbf{P}$ completes(U,u) if $\mathbf{u} = \mathbf{h}\left\lbrack  \mathcal{U}\right\rbrack$ . The following lemma states that(U,u)cannot be completed by too many full configurations:

完成一个配置。考虑$\mathbf{P}$的一个$\mathcal{U}$ - 配置：(U,u)。我们称$\mathbf{P}$的一个完整配置(H,h)完成了(U,u)，如果$\mathbf{u} = \mathbf{h}\left\lbrack  \mathcal{U}\right\rbrack$。以下引理指出，(U,u)不能由太多完整配置完成：

LEMMA 5.3.(U,u)can be completed by $O\left( {\lambda }^{\left| \mathcal{H} \smallsetminus  \mathcal{U}\right| }\right)$ full configurations of $P$ .

引理5.3。(U,u)可以由$P$的$O\left( {\lambda }^{\left| \mathcal{H} \smallsetminus  \mathcal{U}\right| }\right)$个完整配置完成。

Proof. We will design a tuple $\mathbf{h}$ over $\mathcal{H}$ to make the full configuration(H,h)complete(U,u). Clearly, $\mathbf{h}\left( A\right)  = \mathbf{u}\left( A\right)$ for every $A \in  \mathcal{U}$ . It remains to design $\mathbf{h}\left( A\right)$ for $A \in  \mathcal{H} \smallsetminus  \mathcal{U}$ .

证明。我们将在$\mathcal{H}$上设计一个元组$\mathbf{h}$，以使完整配置(H,h)完成(U,u)。显然，对于每个$A \in  \mathcal{U}$，有$\mathbf{h}\left( A\right)  = \mathbf{u}\left( A\right)$。还需要为$A \in  \mathcal{H} \smallsetminus  \mathcal{U}$设计$\mathbf{h}\left( A\right)$。

(1) For every $i \in  \left\lbrack  a\right\rbrack$ with ${X}_{i} \notin  \mathcal{U},\mathbf{h}\left( {X}_{i}\right)$ can be any heavy value.

(1) 对于每个满足${X}_{i} \notin  \mathcal{U},\mathbf{h}\left( {X}_{i}\right)$的$i \in  \left\lbrack  a\right\rbrack$，可以是任何重值。

(2) For every $j \in  \left\lbrack  b\right\rbrack$ with ${Y}_{j} \in  \mathcal{U}$ but ${Z}_{j} \notin  \mathcal{U},\mathbf{h}\left( {Z}_{j}\right)$ can be any light value that makes $\left( {\mathbf{u}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$ heavy.

(2) 对于每个满足${Y}_{j} \in  \mathcal{U}$但${Z}_{j} \notin  \mathcal{U},\mathbf{h}\left( {Z}_{j}\right)$的$j \in  \left\lbrack  b\right\rbrack$，可以是任何使$\left( {\mathbf{u}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$为重的轻值。

(3) For every $j \in  \left\lbrack  b\right\rbrack$ with ${Y}_{j} \notin  \mathcal{U}$ but ${Z}_{j} \in  \mathcal{U},\mathbf{h}\left( {Y}_{j}\right)$ can be any light value that makes $\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{u}\left( {Z}_{j}\right) }\right)$ heavy.

(3) 对于每一个满足${Y}_{j} \notin  \mathcal{U}$但${Z}_{j} \in  \mathcal{U},\mathbf{h}\left( {Y}_{j}\right)$可以是使$\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{u}\left( {Z}_{j}\right) }\right)$为“重”的任意“轻”值的$j \in  \left\lbrack  b\right\rbrack$。

(4) For every $j \in  \left\lbrack  b\right\rbrack$ with ${Y}_{j} \notin  \mathcal{U}$ and ${Z}_{j} \notin  \mathcal{U},\mathbf{h}\left( {Y}_{j}\right)$ and $\mathbf{h}\left( {Z}_{j}\right)$ can be any light values that make $\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$ heavy.

(4) 对于每一个满足${Y}_{j} \notin  \mathcal{U}$、${Z}_{j} \notin  \mathcal{U},\mathbf{h}\left( {Y}_{j}\right)$且$\mathbf{h}\left( {Z}_{j}\right)$可以是使$\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$为“重”的任意“轻”值的$j \in  \left\lbrack  b\right\rbrack$。

We claim:

我们断言：

- for (1),there are at most $\lambda$ ways to set $\mathbf{h}\left( {X}_{i}\right)$ ;

- 对于(1)，设置$\mathbf{h}\left( {X}_{i}\right)$最多有$\lambda$种方法；

- for (2),there are at most $\lambda  \cdot  \left| \mathcal{Q}\right|$ ways to set $\mathbf{h}\left( {Z}_{j}\right)$ ;

- 对于(2)，设置$\mathbf{h}\left( {Z}_{j}\right)$最多有$\lambda  \cdot  \left| \mathcal{Q}\right|$种方法；

- for (3),there are at most $\lambda  \cdot  \left| \mathcal{Q}\right|$ ways to set $\mathbf{h}\left( {Y}_{j}\right)$ ;

- 对于(3)，设置$\mathbf{h}\left( {Y}_{j}\right)$最多有$\lambda  \cdot  \left| \mathcal{Q}\right|$种方法；

- for (4),there are at most ${\lambda }^{2}$ ways to set $\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$ .

- 对于(4)，设置$\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$最多有${\lambda }^{2}$种方法。

The first and the last bullets are obvious because there are at most $\lambda$ heavy values and ${\lambda }^{2}$ heavy value pairs. As the second and third bullets are symmetric, we will prove only the former. Fix an arbitrary $j$ satisfying the condition in (2); denote by $z$ a possible value for $\mathbf{h}\left( {Z}_{j}\right)$ . As $\left( {\mathbf{u}\left( {Y}_{j}\right) ,z}\right)$ is heavy,at least $n/{\lambda }^{2}$ tuples - counting over all the relations in $\mathcal{Q} - \operatorname{carry}\mathbf{u}\left( {Y}_{j}\right)$ and $z$ (on attributes ${Y}_{j}$ and ${Z}_{j}$ ,resp.) simultaneously. If there are at least $\lambda  \cdot  \left| \mathcal{Q}\right|$ choices of $z$ ,the total number of tuples carrying $\mathbf{u}\left( {Y}_{j}\right)$ must be at least $\frac{n}{{\lambda }^{2}}\lambda \left| \mathcal{Q}\right|  = \left| \mathcal{Q}\right| n/\lambda$ . But this means that at least $n/\lambda$ such tuples fall in the same relation in $\mathcal{D}$ ,contradicting the fact that $\mathbf{u}\left( {Y}_{j}\right)$ is light.

第一条和最后一条结论很明显，因为最多有$\lambda$个“重”值和${\lambda }^{2}$个“重”值对。由于第二条和第三条结论是对称的，我们仅证明前者。固定一个满足(2)中条件的任意$j$；用$z$表示$\mathbf{h}\left( {Z}_{j}\right)$的一个可能值。由于$\left( {\mathbf{u}\left( {Y}_{j}\right) ,z}\right)$是“重”的，在$\mathcal{Q} - \operatorname{carry}\mathbf{u}\left( {Y}_{j}\right)$和$z$（分别关于属性${Y}_{j}$和${Z}_{j}$）中的所有关系上同时至少有$n/{\lambda }^{2}$个元组。如果$z$至少有$\lambda  \cdot  \left| \mathcal{Q}\right|$种选择，那么携带$\mathbf{u}\left( {Y}_{j}\right)$的元组总数至少为$\frac{n}{{\lambda }^{2}}\lambda \left| \mathcal{Q}\right|  = \left| \mathcal{Q}\right| n/\lambda$。但这意味着至少有$n/\lambda$个这样的元组落在$\mathcal{D}$中的同一个关系中，这与$\mathbf{u}\left( {Y}_{j}\right)$是“轻”的事实相矛盾。

Denote by ${c}_{1}$ the number of $i$ values satisfying (1),and by ${c}_{2},{c}_{3}$ , and ${c}_{4}$ the number of $j$ values satisfying (2),(3),and (4),respectively. The above discussion indicates that the number of distinct(H,h) meeting all the four conditions is (applying $\left| \mathcal{Q}\right|  = O\left( 1\right)$ )

用${c}_{1}$表示满足(1)的$i$值的数量，分别用${c}_{2},{c}_{3}$、${c}_{4}$表示满足(2)、(3)和(4)的$j$值的数量。上述讨论表明，满足所有四个条件的不同的(H, h)的数量为（应用$\left| \mathcal{Q}\right|  = O\left( 1\right)$）

$$
O\left( {\lambda }^{{c}_{1} + {c}_{2} + {c}_{3} + 2{c}_{4}}\right) 
$$

Notice that ${c}_{1} + {c}_{2} + {c}_{3} + 2{c}_{4}$ is at most the number of attributes in $\mathcal{H} \smallsetminus  \mathcal{U}$ . This completes the proof.

注意，${c}_{1} + {c}_{2} + {c}_{3} + 2{c}_{4}$最多是$\mathcal{H} \smallsetminus  \mathcal{U}$中属性的数量。至此，证明完成。

Total input size of the residual queries. Lemma 5.3 has an important corollary:

剩余查询的总输入大小。引理5.3有一个重要的推论：

COROLLARY 5.4. For any plan $P$ of a unary-free query $\mathcal{D}$ ,the residual queries (i.e.,one for each full configuration of $P$ ) together have a total input size $O\left( {n \cdot  {\lambda }^{k - 2}}\right)$ . If $\mathcal{Q}$ is $\alpha$ -uniform,the bound can be reduced to $O\left( {n \cdot  {\lambda }^{k - \alpha }}\right)$ .

推论5.4。对于无一元查询$\mathcal{D}$的任何计划$P$，剩余查询（即，$P$的每个完整配置对应一个）的总输入大小为$O\left( {n \cdot  {\lambda }^{k - 2}}\right)$。如果$\mathcal{Q}$是$\alpha$ - 均匀的，则该界限可以降低到$O\left( {n \cdot  {\lambda }^{k - \alpha }}\right)$。

Proof. Consider any edge $e \in  \mathcal{E}$ . A tuple $\mathbf{u} \in  {\mathcal{R}}_{e}$ participates in a residual query if and only if the full configuration producing the residual query completes the $e$ -configuration(e,u). By Lemma 5.3,(e,u)can be completed by $O\left( {\lambda }^{k - \left| e\right| }\right)$ full configurations. For a unary-free $\mathcal{Q},k - \left| e\right|  \leq  k - 2$ . Hence,the total input size of all residual queries is $O\left( {n \cdot  {\lambda }^{k - 2}}\right)$ . The claim on $\alpha$ -uniform queries follows from the fact that $k - \left| e\right|  = k - \alpha$ .

证明。考虑任意边$e \in  \mathcal{E}$。元组$\mathbf{u} \in  {\mathcal{R}}_{e}$参与剩余查询当且仅当产生该剩余查询的完整配置完成了$e$ - 配置(e,u)。根据引理5.3，(e,u)可以由$O\left( {\lambda }^{k - \left| e\right| }\right)$个完整配置完成。对于无一元的$\mathcal{Q},k - \left| e\right|  \leq  k - 2$。因此，所有剩余查询的总输入大小为$O\left( {n \cdot  {\lambda }^{k - 2}}\right)$。关于$\alpha$ - 均匀查询的结论源于$k - \left| e\right|  = k - \alpha$这一事实。

We close the section by pointing out that $\mathcal{Q}$ has only a constant number of plans. Therefore, all the residual queries on the right hand side of (13) can be larger than what is stated in Corollary 5.4 by a constant factor.

在本节结尾，我们指出$\mathcal{Q}$只有常数数量的计划。因此，(13)式右侧的所有剩余查询可能比推论5.4中所述的大一个常数因子。

## 6 SIMPLIFYING A RESIDUAL QUERY

## 6 简化剩余查询

This section will concentrate on an arbitrary full configuration (H,h)of $\mathbf{P}$ . We will explain how to simplify the residual query ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ in (12). As before,let $\mathcal{G} =$ (attset $\left( \mathcal{Q}\right) ,\mathcal{E}$ ) be the hyper-graph of2. Set $\mathcal{L} = \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$ .

本节将关注$\mathbf{P}$的任意完整配置(H,h)。我们将解释如何简化(12)中的剩余查询${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$。和之前一样，设$\mathcal{G} =$（属性集$\left( \mathcal{Q}\right) ,\mathcal{E}$）为2的超图。令$\mathcal{L} = \operatorname{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$。

The residual graph. Define ${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$ as the subgraph of $\mathcal{G}$ induced (Section 3.1) by $\mathcal{L}$ ; we will refer to ${\mathcal{G}}^{\prime }$ as the residual graph of $\mathcal{H}$ . Note that ${\mathcal{G}}^{\prime }$ depends only on $\mathcal{H}$ (not on $\mathbf{h}$ ).

剩余图。将${\mathcal{G}}^{\prime } = \left( {\mathcal{L},{\mathcal{E}}^{\prime }}\right)$定义为由$\mathcal{L}$诱导（第3.1节）的$\mathcal{G}$的子图；我们将${\mathcal{G}}^{\prime }$称为$\mathcal{H}$的剩余图。注意，${\mathcal{G}}^{\prime }$仅依赖于$\mathcal{H}$（不依赖于$\mathbf{h}$）。

A vertex (a.k.a. attribute) $A \in  \mathcal{L}$ is orphaned if it appears in a unary edge in ${\mathcal{E}}^{\prime }$ (that is, $\{ A\}  \in  {\mathcal{E}}^{\prime }$ ). Such a vertex is further said to be isolated if it appears in no non-unary edges in ${\mathcal{E}}^{\prime }$ . Denote by $\mathcal{I}$ the set of isolated vertices.

如果顶点（也称为属性）$A \in  \mathcal{L}$出现在${\mathcal{E}}^{\prime }$的一元边中（即$\{ A\}  \in  {\mathcal{E}}^{\prime }$），则称其为孤立顶点。如果这样的顶点不出现在${\mathcal{E}}^{\prime }$的任何非一元边中，则进一步称其为孤立点。用$\mathcal{I}$表示孤立顶点的集合。

Example. For the $\mathcal{G}$ in Figure 1(a),Figure 1(b) shows the residual graph ${\mathcal{G}}^{\prime }$ for $\mathcal{H} = \{ \mathrm{D},\mathrm{G},\mathrm{H}\}$ . The isolated set is $\mathcal{I} = \{ \mathrm{F},\mathrm{J},\mathrm{K}\}$ . Every other vertex in $\mathcal{L} = \{ \mathrm{A},\mathrm{B},\mathrm{C},\mathrm{E},\mathrm{F},\mathrm{I},\mathrm{J},\mathrm{K}\}$ is orphaned. For example, A is orphaned because the edge $\{ \mathrm{A},\mathrm{G}\}$ in $\mathcal{G}$ shrinks into a unary edge A in ${\mathcal{G}}^{\prime }$ ; however, $\mathrm{A}$ is not isolated because it also appears in the edge $\{ A,B,C\}$ of ${\mathcal{G}}^{\prime }$ .

示例。对于图1(a)中的$\mathcal{G}$，图1(b)展示了针对$\mathcal{H} = \{ \mathrm{D},\mathrm{G},\mathrm{H}\}$的残差图${\mathcal{G}}^{\prime }$。孤立集为$\mathcal{I} = \{ \mathrm{F},\mathrm{J},\mathrm{K}\}$。$\mathcal{L} = \{ \mathrm{A},\mathrm{B},\mathrm{C},\mathrm{E},\mathrm{F},\mathrm{I},\mathrm{J},\mathrm{K}\}$中的其他每个顶点都是孤立的。例如，A是孤立的，因为$\mathcal{G}$中的边$\{ \mathrm{A},\mathrm{G}\}$在${\mathcal{G}}^{\prime }$中收缩为一元边A；然而，$\mathrm{A}$不是孤立的，因为它也出现在${\mathcal{G}}^{\prime }$的边$\{ A,B,C\}$中。

Unary intersection on an orphaned attribute. Consider an orphaned attribute $A \in  \mathcal{L}$ . Given an edge $e$ in $\mathcal{G}$ ,we call $e$ an orphaning edge of $A$ if $e \smallsetminus  \mathcal{H} = \{ A\}$ . Define a unary relation for $A$ :

孤立属性上的一元交集。考虑一个孤立属性$A \in  \mathcal{L}$。给定$\mathcal{G}$中的一条边$e$，如果$e \smallsetminus  \mathcal{H} = \{ A\}$，我们称$e$为$A$的孤立边。为$A$定义一个一元关系：

$$
{\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  = \mathop{\bigcap }\limits_{{\text{orphaning edge }e\text{ of }A}}{\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)  \tag{14}
$$

where ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is the residual relation of $e$ (Section 5). Only the values in ${\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ can possibly contribute to the result of $\mathcal{Q}$ .

其中${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$是$e$的残差关系（第5节）。只有${\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$中的值可能对$\mathcal{Q}$的结果有贡献。

Example (cont.). The orphaned attribute $\mathrm{C}$ in Figure 1(b) has two orphaning edges: $\{ \mathrm{C},\mathrm{G}\}$ and $\{ \mathrm{C},\mathrm{H}\} .{\mathcal{R}}_{\mathrm{C}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is the intersection of the residual relations of those two edges. Equivalently, ${\mathcal{R}}_{\mathrm{C}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ contains all the values $x$ such that tuples(x,g)and(x,h)belong to relations ${\mathcal{R}}_{\{ \mathrm{C},\mathrm{G}\} }$ and ${\mathcal{R}}_{\{ \mathrm{C},\mathrm{H}\} }$ ,respectively. Similarly,the isolated vertex $\mathrm{K}$ has three orphaning edges: $\{ \mathrm{K},\mathrm{D}\} ,\{ \mathrm{K},\mathrm{G}\}$ ,and $\{ \mathrm{K},\mathrm{H}\} .{\mathcal{R}}_{\mathrm{K}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ contains all the values $x$ such that tuples $\left( {x,\mathrm{\;d}}\right) ,\left( {x,\mathrm{\;g}}\right)$ ,and(x,h) belong to relations ${\mathcal{R}}_{\{ \mathrm{K},\mathrm{G}\} },{\mathcal{R}}_{\{ \mathrm{K},\mathrm{G}\} }$ ,and ${\mathcal{R}}_{\{ \mathrm{K},\mathrm{H}\} }$ ,respectively.

示例（续）。图1(b)中的孤立属性$\mathrm{C}$有两条孤立边：$\{ \mathrm{C},\mathrm{G}\}$和$\{ \mathrm{C},\mathrm{H}\} .{\mathcal{R}}_{\mathrm{C}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$是这两条边的残差关系的交集。等价地，${\mathcal{R}}_{\mathrm{C}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$包含所有值$x$，使得元组(x,g)和(x,h)分别属于关系${\mathcal{R}}_{\{ \mathrm{C},\mathrm{G}\} }$和${\mathcal{R}}_{\{ \mathrm{C},\mathrm{H}\} }$。类似地，孤立顶点$\mathrm{K}$有三条孤立边：$\{ \mathrm{K},\mathrm{D}\} ,\{ \mathrm{K},\mathrm{G}\}$，并且$\{ \mathrm{K},\mathrm{H}\} .{\mathcal{R}}_{\mathrm{K}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$包含所有值$x$，使得元组$\left( {x,\mathrm{\;d}}\right) ,\left( {x,\mathrm{\;g}}\right)$和(x,h)分别属于关系${\mathcal{R}}_{\{ \mathrm{K},\mathrm{G}\} },{\mathcal{R}}_{\{ \mathrm{K},\mathrm{G}\} }$和${\mathcal{R}}_{\{ \mathrm{K},\mathrm{H}\} }$。

Semi-join reduction. Let $e$ be an edge in $\mathcal{G}$ such that ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ is not unary. Define the semi-join reduced relation of $e$ as:

半连接约简。设$e$是$\mathcal{G}$中的一条边，使得${e}^{\prime } = e \smallsetminus  \mathcal{H}$不是一元的。将$e$的半连接约简关系定义为：

$$
{\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  = \text{the join result of}{\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) \text{and all}{\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) 
$$

$$
\text{where}A \in  {e}^{\prime }\text{is an orphaned attribute.} \tag{15}
$$

To see the rationale behind,consider a tuple $\mathbf{u} \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ such that $\mathbf{u}\left( A\right)$ does not appear in ${\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ; clearly, $\mathbf{u}$ cannot contribute to Join(2). ${\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ contains what remains in ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ after eliminating all such tuples $\mathbf{u}$ .

为了理解其背后的原理，考虑一个元组$\mathbf{u} \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$，使得$\mathbf{u}\left( A\right)$不出现在${\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$中；显然，$\mathbf{u}$对连接操作（2）没有贡献。${\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$包含了在消除所有此类元组$\mathbf{u}$后${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$中剩余的部分。

Example (cont.). In Figure 1(b),consider $e = \{ \mathrm{C},\mathrm{D},\mathrm{E}\}$ ,and hence, ${e}^{\prime } = \{ \mathrm{C},\mathrm{E}\}$ . Recall that ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ includes all tuples(x,y)such that (x,d,y)in ${\mathcal{R}}_{e}.{\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ shrinks ${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ by keeping only the tuples(x,y)where $x \in  {\mathcal{R}}_{\mathrm{C}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ and $y \in  {\mathcal{R}}_{\mathrm{E}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ .

示例（续）。在图1（b）中，考虑$e = \{ \mathrm{C},\mathrm{D},\mathrm{E}\}$，因此，${e}^{\prime } = \{ \mathrm{C},\mathrm{E}\}$。回顾一下，${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$包含所有元组（x，y），使得（x，d，y）在${\mathcal{R}}_{e}.{\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$中，通过仅保留满足$x \in  {\mathcal{R}}_{\mathrm{C}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$和$y \in  {\mathcal{R}}_{\mathrm{E}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$的元组（x，y）来缩减${\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$。

The simplified residual query. We are now ready to formulate:

简化后的剩余查询。现在我们准备进行如下表述：

$$
{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  = \left\{  {{\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  \mid  e \smallsetminus  \mathcal{H}\text{ is non-unary }}\right\}   \tag{16}
$$

$$
{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  = \left\{  {{\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  \mid  A \in  \mathcal{L}\text{ is isolated }}\right\}   \tag{17}
$$

$$
{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  = {\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  \cup  {\mathcal{Q}}_{\mathcal{I}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) . \tag{18}
$$

Every relation in ${\mathcal{Q}}_{\mathcal{T}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is unary and is on an isolated vertex, whereas every relation in ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ has at least two attributes.

${\mathcal{Q}}_{\mathcal{T}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$中的每个关系都是一元关系，并且位于一个孤立的顶点上，而${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$中的每个关系至少有两个属性。

Example. In Figure 1(b), ${\mathcal{D}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ joins the semi-join reduced relations of $\{ \mathrm{A},\mathrm{B},\mathrm{C}\} ,\{ \mathrm{C},\mathrm{E}\}$ ,and $\{ \mathrm{E},\mathrm{I}\} .{\mathcal{Q}}_{T}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ computes the $\mathrm{{CP}}$ (cartesian product) of ${\mathcal{R}}_{\mathrm{F}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) ,{\mathcal{R}}_{\mathrm{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ,and ${\mathcal{R}}_{\mathrm{K}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . Finally, ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ returns the CP of the previous two queries’ results.

示例。在图1（b）中，${\mathcal{D}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$对$\{ \mathrm{A},\mathrm{B},\mathrm{C}\} ,\{ \mathrm{C},\mathrm{E}\}$的半连接约简关系进行连接操作，并且$\{ \mathrm{E},\mathrm{I}\} .{\mathcal{Q}}_{T}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$计算${\mathcal{R}}_{\mathrm{F}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) ,{\mathcal{R}}_{\mathrm{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$和${\mathcal{R}}_{\mathrm{K}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$的$\mathrm{{CP}}$（笛卡尔积）。最后，${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$返回前两个查询结果的笛卡尔积。

We prove in Appendix C:

我们在附录C中证明：

Proposition 6.1. ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ and ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ have the same result.

命题6.1。${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$和${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$具有相同的结果。

## 7 AN ISOLATED CP THEOREM

## 7 孤立笛卡尔积定理

In this section,we will focus on an arbitrary plan $P$ . Every full configuration(H,h)of $\mathbf{P}$ has a simplified residual query ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ that computes $\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{I}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . As shown in Lemma 3.3,the cost of ${CP}\left( {{\mathcal{Q}}_{\mathcal{T}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ depends on the cartesian product size of every non-empty subset of the relations in ${CP}\left( {{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . Next,we will prove that the sum of those cartesian product sizes is small.

在本节中，我们将关注任意计划 $P$。$\mathbf{P}$ 的每个完整配置 (H, h) 都有一个简化的残差查询 ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$，用于计算 $\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{I}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。如引理 3.3 所示，${CP}\left( {{\mathcal{Q}}_{\mathcal{T}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 的成本取决于 ${CP}\left( {{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 中关系的每个非空子集的笛卡尔积大小。接下来，我们将证明这些笛卡尔积大小的总和很小。

For every non-empty subset $\mathcal{J}$ of $\mathcal{I}$ ,define:

对于 $\mathcal{I}$ 的每个非空子集 $\mathcal{J}$，定义：

$$
{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  = \left\{  {{\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)  \mid  A \in  \mathcal{J}}\right\}  . \tag{19}
$$

where ${\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is given in (14). Although the size of ${CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ may vary significantly on different(H,h),we are able to bound the total size (lumping over all(H,h)):

其中 ${\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 在 (14) 中给出。尽管 ${CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 的大小在不同的 (H, h) 上可能会有显著差异，但我们能够对总大小（对所有 (H, h) 进行汇总）进行界定：

THEOREM 7.1 (ISOLATED CARTESIAN PRODUCT THEOREM). Let Q be a unary-free clean query. For each plan $P$ of $\mathcal{Q}$ and any non-empty $\mathcal{J} \subseteq  I$ ,it holds that

定理 7.1（孤立笛卡尔积定理）。设 Q 是一个无一元的干净查询。对于 $\mathcal{Q}$ 的每个计划 $P$ 和任何非空的 $\mathcal{J} \subseteq  I$，有

$$
\mathop{\sum }\limits_{\substack{\text{ full config. } \\  {\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }\mathbf{P}} }}\left| {\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right) }\right|  \leq  {\lambda }^{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right)  - \left| {\mathcal{L} \smallsetminus  \mathcal{J}}\right| } \cdot  {n}^{\left| \mathcal{J}\right| }
$$

where $\phi$ the generalized vertex packing number of $\mathcal{Q}$ (Section 4).

其中 $\phi$ 是 $\mathcal{Q}$ 的广义顶点覆盖数（第 4 节）。

Recall that $\mathcal{L} =$ attset $\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$ ,and $\alpha$ is the maximum arity defined in (2). The rest of the section serves as a proof of Theorem 7.1.

回顾一下，$\mathcal{L} =$ attset $\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}$，并且 $\alpha$ 是 (2) 中定义的最大元数。本节的其余部分用于证明定理 7.1。

### 7.1 Three Properties of the Isolated Vertices

### 7.1 孤立顶点的三个性质

Let $\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$ be the hypergraph defined by $\mathcal{Q}$ . The next lemma states several properties of $\mathcal{J}$ :

设 $\mathcal{G} = \left( {\text{attset}\left( \mathcal{Q}\right) ,\mathcal{E}}\right)$ 是由 $\mathcal{Q}$ 定义的超图。下一个引理陈述了 $\mathcal{J}$ 的几个性质：

LEMMA 7.2. If an edge $e \in  \mathcal{E}$ satisfies $e \cap  \mathcal{J} \neq  \varnothing$ ,then

引理 7.2。如果一条边 $e \in  \mathcal{E}$ 满足 $e \cap  \mathcal{J} \neq  \varnothing$，那么

(1) $\left| {e \cap  \mathcal{J}}\right|  = 1$ ,

(2) $e \subseteq  \left( {\mathcal{J} \cup  \mathcal{H}}\right)$ ,and

(2) $e \subseteq  \left( {\mathcal{J} \cup  \mathcal{H}}\right)$，并且

(3) $\left| e\right|  = \left| {e \cap  \mathcal{J}}\right|  + \left| {e \cap  \mathcal{H}}\right|  = \left| {e \cap  \mathcal{H}}\right|  + 1$ .

Proof. By definition of isolated vertex (Section 6),no edge $e \in  \mathcal{E}$ can contain two vertices in $\mathcal{J}$ ,which yields Property (1). To prove (2),let $A$ be the only attribute in $e \cap  \mathcal{J}$ . If (2) does not hold,then because of (1), $e$ must have another attribute $B \notin  \mathcal{H}$ which,however, contradicts the fact that $A$ is isolated. (3) follows from (1),(2) and the fact that $\mathcal{J}$ and $\mathcal{H}$ are disjoint.

证明。根据孤立顶点的定义（第 6 节），没有边 $e \in  \mathcal{E}$ 可以包含 $\mathcal{J}$ 中的两个顶点，这就得到了性质 (1)。为了证明 (2)，设 $A$ 是 $e \cap  \mathcal{J}$ 中唯一的属性。如果 (2) 不成立，那么由于 (1)，$e$ 必须有另一个属性 $B \notin  \mathcal{H}$，然而，这与 $A$ 是孤立的这一事实相矛盾。(3) 由 (1)、(2) 以及 $\mathcal{J}$ 和 $\mathcal{H}$ 不相交这一事实得出。

### 7.2 Utilizing the Characterizing Program

### 7.2 利用特征化程序

Henceforth,we will denote by $\left\{  {{x}_{e} \mid  e \in  \mathcal{E}}\right\}$ an optimal assignment for the characterizing program of $\mathcal{Q}$ (Section 4). Thus, $\bar{\phi }\left( \mathcal{Q}\right)  =$ $\mathop{\sum }\limits_{e}{x}_{e}\left( {\left| e\right|  - 1}\right)$ ; we will abbreviate $\bar{\phi }\left( \mathcal{Q}\right)$ as $\bar{\phi }$ .

此后，我们将用$\left\{  {{x}_{e} \mid  e \in  \mathcal{E}}\right\}$表示$\mathcal{Q}$的特征规划（第4节）的一个最优赋值。因此，$\bar{\phi }\left( \mathcal{Q}\right)  =$ $\mathop{\sum }\limits_{e}{x}_{e}\left( {\left| e\right|  - 1}\right)$；我们将$\bar{\phi }\left( \mathcal{Q}\right)$缩写为$\bar{\phi }$。

Define

定义

$$
{\mathcal{E}}^{ * } = \{ e \in  \mathcal{E} \mid  e \cap  \mathcal{J} \neq  \varnothing \} . \tag{20}
$$

We prove in Appendix D (recall that $k =  \mid$ attset $\left( \mathcal{Q}\right)  \mid$ ):

我们在附录D中证明（回顾$k =  \mid$ attset $\left( \mathcal{Q}\right)  \mid$）：

LEMMA 7.3.

引理7.3。

$$
k - \left| \mathcal{J}\right|  - \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}{x}_{e} \cdot  \left( {\left| e\right|  - 1}\right)  \leq  \alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) .
$$

In the next subsection, we will prove:

在下一小节中，我们将证明：

LEMMA 7.4.

引理7.4。

$$
\mathop{\sum }\limits_{\substack{\text{ full config. } \\  {\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }\mathbf{P}} }}\left| {\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right) }\right|  \leq  {\lambda }^{\left| \mathcal{H}\right|  - \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}{x}_{e}\left( {\left| e\right|  - 1}\right) } \cdot  {n}^{\left| \mathcal{J}\right| }.
$$

This will complete the whole proof of Theorem 7.1 because

这将完成定理7.1的整个证明，因为

$$
\left| \mathcal{H}\right|  - \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}\left| {x}_{e}\right|  \cdot  \left( {\left| e\right|  - 1}\right)  \leq  \left| \mathcal{H}\right|  - k + \left| \mathcal{J}\right|  + \alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) 
$$

$$
 = \alpha \left( {\phi  - \left| \mathcal{J}\right| }\right)  - \left| \mathcal{L}\right|  + \left| \mathcal{J}\right| 
$$

$$
 = \alpha \left( {\phi  - \left| \mathcal{J}\right| }\right)  - \left| {\mathcal{L} \smallsetminus  \mathcal{J}}\right| 
$$

where the first inequality applied Lemma 7.3.

其中第一个不等式应用了引理7.3。

### 7.3 Proof of Lemma 7.4

### 7.3 引理7.4的证明

Recall from (9) that plan $P$ specifies $a \geq  0$ attributes ${X}_{1},\ldots ,{X}_{a}$ and $b \geq  0$ attribute pairs $\left( {{Y}_{1},{Z}_{1}}\right) ,\ldots ,\left( {{Y}_{b},{Z}_{b}}\right)$ . We create $a + b$ special relations as follows:

回顾（9）可知，规划$P$指定了$a \geq  0$个属性${X}_{1},\ldots ,{X}_{a}$和$b \geq  0$个属性对$\left( {{Y}_{1},{Z}_{1}}\right) ,\ldots ,\left( {{Y}_{b},{Z}_{b}}\right)$。我们按如下方式创建$a + b$个特殊关系：

- for each $i \in  \left\lbrack  a\right\rbrack$ ,create a relation ${\mathcal{S}}_{i}$ with scheme $\left( {\mathcal{S}}_{i}\right)  = \left\{  {X}_{i}\right\}$ which contains all the values heavy on ${X}_{i}$ ;

- 对于每个$i \in  \left\lbrack  a\right\rbrack$，创建一个具有模式$\left( {\mathcal{S}}_{i}\right)  = \left\{  {X}_{i}\right\}$的关系${\mathcal{S}}_{i}$，该关系包含所有关于${X}_{i}$的重值；

- for each $j \in  \left\lbrack  b\right\rbrack$ ,create a relation ${\mathcal{D}}_{j}$ with scheme $\left( {\mathcal{D}}_{j}\right)  =$ $\left\{  {{Y}_{j},{Z}_{j}}\right\}$ which contains all the value pairs(y,z)such that

- 对于每个$j \in  \left\lbrack  b\right\rbrack$，创建一个具有模式$\left( {\mathcal{D}}_{j}\right)  =$ $\left\{  {{Y}_{j},{Z}_{j}}\right\}$的关系${\mathcal{D}}_{j}$，该关系包含所有满足以下条件的值对(y,z)：

-(y,z)is heavy on $\left\{  {{Y}_{j},{Z}_{j}}\right\}$ ;

-(y,z)关于$\left\{  {{Y}_{j},{Z}_{j}}\right\}$是重的；

$- y$ and $z$ are both light values.

$- y$和$z$都是轻值。

Clearly, $\left| {\mathcal{S}}_{i}\right|  \leq  \lambda$ and $\left| {\mathcal{D}}_{j}\right|  \leq  {\lambda }^{2}$ for all $i,j$ . Let ${\mathcal{Q}}_{\text{heavy }}$ be the query that includes the above $a + b$ relations. Note that attset $\left( {\mathcal{Q}}_{\text{heavy }}\right)  = \mathcal{H}$ . As the relations in ${\mathcal{D}}_{\text{heavy }}$ have disjoint schemes,the result of ${\mathcal{Q}}_{\text{heavy }}$ is ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)$ .

显然，对于所有$i,j$，有$\left| {\mathcal{S}}_{i}\right|  \leq  \lambda$和$\left| {\mathcal{D}}_{j}\right|  \leq  {\lambda }^{2}$。设${\mathcal{Q}}_{\text{heavy }}$是包含上述$a + b$个关系的查询。注意，attset $\left( {\mathcal{Q}}_{\text{heavy }}\right)  = \mathcal{H}$。由于${\mathcal{D}}_{\text{heavy }}$中的关系具有不相交的模式，所以${\mathcal{Q}}_{\text{heavy }}$的结果是${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)$。

Define:

定义：

$$
{\mathcal{Q}}^{ * } = \left\{  {{\mathcal{R}}_{e} \mid  e \in  {\mathcal{E}}^{ * }}\right\}  .
$$

where ${\mathcal{E}}^{ * }$ is defined in (20). Recall that ${\mathcal{R}}_{e}$ is the input relation in $\mathcal{Q}$ with scheme $e$ . By Property (2) in Lemma 7.2,attset $\left( {\mathcal{Q}}^{ * }\right)  \subseteq  \mathcal{J} \cup  \mathcal{H}$ . We prove in Appendix E:

其中 ${\mathcal{E}}^{ * }$ 在 (20) 中定义。回顾一下，${\mathcal{R}}_{e}$ 是 $\mathcal{Q}$ 中具有模式 $e$ 的输入关系。根据引理 7.2 中的性质 (2)，属性集 $\left( {\mathcal{Q}}^{ * }\right)  \subseteq  \mathcal{J} \cup  \mathcal{H}$。我们在附录 E 中证明：

Proposition 7.5.

命题 7.5。

$$
\mathop{\sum }\limits_{\substack{\text{ full config. } \\  {\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }\mathbf{P}} }}\left| {\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right) }\right|  \leq  \left| {\operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) }\right| .
$$

Our goal,therefore,is to prove that the size of ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ Join $\left( {\mathcal{Q}}^{ * }\right)$ is sufficiently small.

因此，我们的目标是证明 ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ 连接 $\left( {\mathcal{Q}}^{ * }\right)$ 的规模足够小。

Strategy overview. Towards the above purpose, we will construct a sequence of queries ${\mathcal{Q}}_{0},{\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{\ell }$ for some $\ell  \geq  0$ such that:

策略概述。为了实现上述目的，我们将为某些 $\ell  \geq  0$ 构造一个查询序列 ${\mathcal{Q}}_{0},{\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{\ell }$，使得：

- ${\mathcal{Q}}_{0} = {\mathcal{Q}}^{ * }$ ;

- attset $\left( {\mathcal{Q}}_{s}\right)  \subseteq  \mathcal{J} \cup  \mathcal{H}$ for each $s \in  \left\lbrack  {0,\ell }\right\rbrack$ ;

- 对于每个 $s \in  \left\lbrack  {0,\ell }\right\rbrack$，属性集 $\left( {\mathcal{Q}}_{s}\right)  \subseteq  \mathcal{J} \cup  \mathcal{H}$；

- $\operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{s}\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ for each $s \in  \left\lbrack  {0,\ell }\right\rbrack$

- 对于每个 $s \in  \left\lbrack  {0,\ell }\right\rbrack$，$\operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{s}\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$

Eventually,we will prove that $\left| {{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \mathcal{J}\text{oin}\left( {\mathcal{Q}}_{\ell }\right) }\right|$ is low enough. This, together with Proposition 7.5, will allow us to establish Lemma 7.4.

最终，我们将证明 $\left| {{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \mathcal{J}\text{oin}\left( {\mathcal{Q}}_{\ell }\right) }\right|$ 足够低。结合命题 7.5，这将使我们能够证明引理 7.4。

Constructing ${\mathcal{Q}}_{s + 1}$ for $s \geq  \mathbf{0}$ . Suppose that we have already obtained ${\mathcal{Q}}_{s}$ for some $s \geq  0$ . Let ${\mathcal{G}}_{s} = \left( {\operatorname{attset}\left( {\mathcal{Q}}_{s}\right) ,{\mathcal{E}}_{s}}\right)$ be the hyper-graph defined by ${\mathcal{Q}}_{s}$ . For each edge $e \in  {\mathcal{E}}_{s}$ ,denote by ${\mathcal{R}}_{e,s}$ the relation in ${\mathcal{Q}}_{s}$ whose scheme is $e$ .

为 $s \geq  \mathbf{0}$ 构造 ${\mathcal{Q}}_{s + 1}$。假设我们已经为某些 $s \geq  0$ 得到了 ${\mathcal{Q}}_{s}$。令 ${\mathcal{G}}_{s} = \left( {\operatorname{attset}\left( {\mathcal{Q}}_{s}\right) ,{\mathcal{E}}_{s}}\right)$ 为由 ${\mathcal{Q}}_{s}$ 定义的超图。对于每条边 $e \in  {\mathcal{E}}_{s}$，用 ${\mathcal{R}}_{e,s}$ 表示 ${\mathcal{Q}}_{s}$ 中模式为 $e$ 的关系。

We assume that each edge $e \in  {\mathcal{E}}_{s}$ is assigned a real value ${x}_{e,s} \geq  0$ such that $\left\{  {{x}_{e,s} \mid  e \in  {\mathcal{E}}_{s}}\right\}$ is a feasible assignment of the characterizing program of ${\mathcal{G}}_{s}$ (Section 4). At $s = 0$ ,this assumption is fulfilled by simply setting ${x}_{e,0} = {x}_{e}$ for each $e \in  {\mathcal{E}}_{s}$ . Recall (Section 7.2) that $\left\{  {{x}_{e} \mid  e \in  \mathcal{E}}\right\}$ is a feasible assignment for the characterizing program of $\mathcal{G}$ .

我们假设每条边 $e \in  {\mathcal{E}}_{s}$ 都被赋予一个实数值 ${x}_{e,s} \geq  0$，使得 $\left\{  {{x}_{e,s} \mid  e \in  {\mathcal{E}}_{s}}\right\}$ 是 ${\mathcal{G}}_{s}$ 的特征规划（第 4 节）的一个可行赋值。在 $s = 0$ 处，通过简单地为每条 $e \in  {\mathcal{E}}_{s}$ 设置 ${x}_{e,0} = {x}_{e}$ 即可满足这一假设。回顾（第 7.2 节），$\left\{  {{x}_{e} \mid  e \in  \mathcal{E}}\right\}$ 是 $\mathcal{G}$ 的特征规划的一个可行赋值。

For each attribute $A \in  \mathcal{J} \cup  \mathcal{H}$ ,define:

对于每个属性 $A \in  \mathcal{J} \cup  \mathcal{H}$，定义：

$$
{\mathcal{F}}_{s}\left( A\right)  = \left\{  \begin{array}{ll} \mathop{\sum }\limits_{{e \in  {\mathcal{E}}_{s} : A \in  e}}{x}_{e,s} & \text{ if }A \in  \operatorname{attset}\left( {\mathcal{Q}}_{s}\right) \\  0 & \text{ otherwise } \end{array}\right.  \tag{21}
$$

Note that ${\mathcal{F}}_{s}\left( A\right)  \leq  1$ for every $A$ . We build ${\mathcal{Q}}_{s + 1}$ only if there exists some $j \in  \left\lbrack  b\right\rbrack$ such that ${\mathcal{F}}_{s}\left( {Y}_{j}\right)  \neq  {\mathcal{F}}_{s}\left( {Z}_{j}\right)$ ; otherwise, ${\mathcal{Q}}_{s}$ is the last query constructed (i.e., $\ell  = s$ ). We refer to $j$ as the triggering index of ${\mathcal{Q}}_{s + 1}$ . Without loss of generality,our discussion below assumes ${\mathcal{F}}_{s}\left( {Y}_{j}\right)  > {\mathcal{F}}_{s}\left( {Z}_{j}\right)$ . The opposite case is symmetric and can be handled analogously.

注意，对于每个$A$，都有${\mathcal{F}}_{s}\left( A\right)  \leq  1$。仅当存在某个$j \in  \left\lbrack  b\right\rbrack$使得${\mathcal{F}}_{s}\left( {Y}_{j}\right)  \neq  {\mathcal{F}}_{s}\left( {Z}_{j}\right)$时，我们才构建${\mathcal{Q}}_{s + 1}$；否则，${\mathcal{Q}}_{s}$是构造的最后一个查询（即$\ell  = s$）。我们将$j$称为${\mathcal{Q}}_{s + 1}$的触发索引。不失一般性，下面的讨论假设${\mathcal{F}}_{s}\left( {Y}_{j}\right)  > {\mathcal{F}}_{s}\left( {Z}_{j}\right)$。相反的情况是对称的，可以类似地处理。

By the fact ${\mathcal{F}}_{s}\left( {Y}_{j}\right)  > {\mathcal{F}}_{s}\left( {Z}_{j}\right)$ ,there must exist a triggering edge ${e}^{ * } \in  {\mathcal{E}}_{s}$ which includes ${Y}_{j}$ but not ${Z}_{j}$ . Based on ${e}^{ * }$ ,we will produce a feasible assignment $\left\{  {{x}_{e,s + 1} \mid  e \in  {\mathcal{E}}_{s + 1}}\right\}$ for the characterizing program of the hypergraph ${\mathcal{G}}_{s + 1} = \left( {\operatorname{attset}\left( {\mathcal{Q}}_{s + 1}\right) ,{\mathcal{E}}_{s + 1}}\right)$ defined by ${\mathcal{Q}}_{s + 1}$ . After that,the construction process is iteratively invoked.

根据事实${\mathcal{F}}_{s}\left( {Y}_{j}\right)  > {\mathcal{F}}_{s}\left( {Z}_{j}\right)$，必定存在一条触发边${e}^{ * } \in  {\mathcal{E}}_{s}$，它包含${Y}_{j}$但不包含${Z}_{j}$。基于${e}^{ * }$，我们将为由${\mathcal{Q}}_{s + 1}$定义的超图${\mathcal{G}}_{s + 1} = \left( {\operatorname{attset}\left( {\mathcal{Q}}_{s + 1}\right) ,{\mathcal{E}}_{s + 1}}\right)$的特征规划生成一个可行赋值$\left\{  {{x}_{e,s + 1} \mid  e \in  {\mathcal{E}}_{s + 1}}\right\}$。之后，迭代调用构造过程。

Define:

定义：

$$
{e}^{ + } = {e}^{ * } \cup  \left\{  {Z}_{j}\right\}   \tag{22}
$$

$$
{\mathcal{R}}^{ + } = \left\{  \begin{array}{ll} {\mathcal{R}}_{{e}^{ * },s} \bowtie  {\mathcal{D}}_{j} \bowtie  {\mathcal{R}}_{{e}^{ + },s} & \text{ if }{e}^{ + } \in  {\mathcal{E}}_{s} \\  {\mathcal{R}}_{{e}^{ * },s} \bowtie  {\mathcal{D}}_{j} & \text{ otherwise } \end{array}\right.  \tag{23}
$$

Note that scheme $\left( {\mathcal{R}}^{ + }\right)  = {e}^{ + }$ . Set

注意方案$\left( {\mathcal{R}}^{ + }\right)  = {e}^{ + }$。设置

$$
{\Delta }_{s} = \min \left\{  {{x}_{{e}^{ * },s},{\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right) }\right\}  .
$$

Depending on ${\Delta }_{s}$ ,we generate ${\mathcal{Q}}_{s + 1}$ differently:

根据${\Delta }_{s}$，我们以不同的方式生成${\mathcal{Q}}_{s + 1}$：

- Case ${\Delta }_{s} < {x}_{{e}^{ * },s}$ :

- 情况${\Delta }_{s} < {x}_{{e}^{ * },s}$：

- If ${e}^{ + } \in  {\mathcal{E}}_{s}$ ,then ${\mathcal{Q}}_{s + 1} = \left( {{\mathcal{Q}}_{s} \smallsetminus  \left\{  {\mathcal{R}}_{{e}^{ + },s}\right\}  }\right)  \cup  \left\{  {\mathcal{R}}^{ + }\right\}$ ;

- 如果${e}^{ + } \in  {\mathcal{E}}_{s}$，那么${\mathcal{Q}}_{s + 1} = \left( {{\mathcal{Q}}_{s} \smallsetminus  \left\{  {\mathcal{R}}_{{e}^{ + },s}\right\}  }\right)  \cup  \left\{  {\mathcal{R}}^{ + }\right\}$；

- otherwise, ${\mathcal{Q}}_{s + 1} = {\mathcal{Q}}_{s} \cup  \left\{  {\mathcal{R}}^{ + }\right\}$ .

- 否则，${\mathcal{Q}}_{s + 1} = {\mathcal{Q}}_{s} \cup  \left\{  {\mathcal{R}}^{ + }\right\}$。

- Case ${\Delta }_{s} = {x}_{{e}^{ * },s}$ :

- 情况${\Delta }_{s} = {x}_{{e}^{ * },s}$：

- If ${e}^{ + } \in  {\mathcal{E}}_{s}$ ,then ${\mathcal{Q}}_{s + 1} = \left( {{\mathcal{Q}}_{s} \smallsetminus  \left\{  {{\mathcal{R}}_{{e}^{ * },s},{\mathcal{R}}_{{e}^{ + },s}}\right\}  }\right)  \cup  \left\{  {\mathcal{R}}^{ + }\right\}$ ;

- 如果${e}^{ + } \in  {\mathcal{E}}_{s}$，那么${\mathcal{Q}}_{s + 1} = \left( {{\mathcal{Q}}_{s} \smallsetminus  \left\{  {{\mathcal{R}}_{{e}^{ * },s},{\mathcal{R}}_{{e}^{ + },s}}\right\}  }\right)  \cup  \left\{  {\mathcal{R}}^{ + }\right\}$；

- otherwise, ${\mathcal{Q}}_{s + 1} = \left( {{\mathcal{Q}}_{s} \smallsetminus  \left\{  {\mathcal{R}}_{{e}^{ * },s}\right\}  }\right)  \cup  \left\{  {\mathcal{R}}^{ + }\right\}$ .

- 否则，${\mathcal{Q}}_{s + 1} = \left( {{\mathcal{Q}}_{s} \smallsetminus  \left\{  {\mathcal{R}}_{{e}^{ * },s}\right\}  }\right)  \cup  \left\{  {\mathcal{R}}^{ + }\right\}$。

Remember that we also need to produce a feasible assignment $\left\{  {{x}_{{e}^{ * },s + 1} \mid  {e}^{ * } \in  {\mathcal{G}}_{s + 1}}\right\}$ for the characterizing program of ${\mathcal{G}}_{s + 1}$ . This is done as follows:

请记住，我们还需要为${\mathcal{G}}_{s + 1}$的特征规划生成一个可行的赋值$\left\{  {{x}_{{e}^{ * },s + 1} \mid  {e}^{ * } \in  {\mathcal{G}}_{s + 1}}\right\}$。具体步骤如下：

- First,for every edge $e$ in ${\mathcal{G}}_{s}$ other than ${e}^{ * }$ and ${e}^{ + }$ ,retain its assigned value,namely, ${x}_{e,s + 1} = {x}_{e,s}$ .

- 首先，对于${\mathcal{G}}_{s}$中除${e}^{ * }$和${e}^{ + }$之外的每条边$e$，保留其已分配的值，即${x}_{e,s + 1} = {x}_{e,s}$。

- Then,we proceed differently depending on ${\Delta }_{s}$ :

- 然后，我们根据${\Delta }_{s}$的不同情况进行不同处理：

- Case ${\Delta }_{S} < {x}_{{e}^{ * },s}$ :

- 情况${\Delta }_{S} < {x}_{{e}^{ * },s}$：

* Set ${x}_{{e}^{ * },s + 1} = {x}_{{e}^{ * },s} - {\Delta }_{s}$ .

* 设置${x}_{{e}^{ * },s + 1} = {x}_{{e}^{ * },s} - {\Delta }_{s}$。

* Set ${x}_{{e}^{ + },s + 1}$ to ${\Delta }_{s} + {x}_{{e}^{ + },s}$ if ${e}^{ + } \in  {\mathcal{E}}_{s}$ ,or to ${\Delta }_{s}$ otherwise.

* 如果${e}^{ + } \in  {\mathcal{E}}_{s}$，则将${x}_{{e}^{ + },s + 1}$设置为${\Delta }_{s} + {x}_{{e}^{ + },s}$；否则，将其设置为${\Delta }_{s}$。

- Case ${\Delta }_{S} = {x}_{{e}^{ * },S}$ :

- 情况${\Delta }_{S} = {x}_{{e}^{ * },S}$：

* Set ${x}_{{e}^{ + },s + 1}$ to ${\Delta }_{s} + {x}_{{e}^{ + },s}$ if ${e}^{ + } \in  {\mathcal{E}}_{s}$ ,or to ${\Delta }_{s}$ otherwise.

* 如果${e}^{ + } \in  {\mathcal{E}}_{s}$，则将${x}_{{e}^{ + },s + 1}$设置为${\Delta }_{s} + {x}_{{e}^{ + },s}$；否则，将其设置为${\Delta }_{s}$。

We now prove that the above generation fulfills our purposes:

现在我们证明上述生成过程满足我们的目的：

LEMMA 7.6. Both statements below are true:

引理7.6。以下两个陈述均为真：

- $\left\{  {{x}_{e,s + 1} \mid  e \in  {\mathcal{G}}_{s + 1}}\right\}$ is a feasible assignment for the characterizing program of ${\mathcal{G}}_{s + 1}$ .

- $\left\{  {{x}_{e,s + 1} \mid  e \in  {\mathcal{G}}_{s + 1}}\right\}$是${\mathcal{G}}_{s + 1}$的特征规划的一个可行赋值。

- $\operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{s + 1}\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right)$ .

Proof. For the first statement,we need to prove that ${\mathcal{F}}_{s + 1}\left( A\right)  \leq$ 1 for every $A \in  \mathcal{J} \cup  \mathcal{H}$ ,where ${\mathcal{F}}_{s + 1}\left( A\right)$ is as defined in (21). ${}^{4}$ Our design makes sure ${\mathcal{F}}_{s + 1}\left( A\right)  = {\mathcal{F}}_{s}\left( A\right)$ for every $A \in  \mathcal{J} \cup  \mathcal{H}$ with $A \neq  {Z}_{j}$ ; thus, ${\mathcal{F}}_{s + 1}\left( A\right)  \leq  1$ follows from ${\mathcal{F}}_{s}\left( A\right)  \leq  1$ . Regarding ${Z}_{j}$ ,our design guarantees ${\mathcal{F}}_{s + 1}\left( {Z}_{j}\right)  = {\mathcal{F}}_{s}\left( {Z}_{j}\right)  + {\Delta }_{s}$ . By definition ${\Delta }_{s} \leq  {\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right)$ . Therefore, ${\mathcal{F}}_{s + 1}\left( {Z}_{j}\right)  \leq  {\mathcal{F}}_{s}\left( {Y}_{j}\right)  \leq  1$ .

证明。对于第一个陈述，我们需要证明对于每个$A \in  \mathcal{J} \cup  \mathcal{H}$，都有${\mathcal{F}}_{s + 1}\left( A\right)  \leq$ ≤ 1，其中${\mathcal{F}}_{s + 1}\left( A\right)$如(21)中所定义。${}^{4}$ 我们的设计确保对于每个满足$A \neq  {Z}_{j}$的$A \in  \mathcal{J} \cup  \mathcal{H}$，都有${\mathcal{F}}_{s + 1}\left( A\right)  = {\mathcal{F}}_{s}\left( A\right)$；因此，由${\mathcal{F}}_{s}\left( A\right)  \leq  1$可推出${\mathcal{F}}_{s + 1}\left( A\right)  \leq  1$。关于${Z}_{j}$，我们的设计保证了${\mathcal{F}}_{s + 1}\left( {Z}_{j}\right)  = {\mathcal{F}}_{s}\left( {Z}_{j}\right)  + {\Delta }_{s}$。根据定义${\Delta }_{s} \leq  {\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right)$。因此，${\mathcal{F}}_{s + 1}\left( {Z}_{j}\right)  \leq  {\mathcal{F}}_{s}\left( {Y}_{j}\right)  \leq  1$。

For the second statement, it suffices to show

对于第二个陈述，只需证明

$$
{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{s + 1}\right)  = {CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{s}\right) . \tag{24}
$$

Consider first the case where ${e}^{ + } \in  {\mathcal{E}}_{s}$ . Recall that ${\mathcal{D}}_{j} \in  {\mathcal{Q}}_{\text{heavy }}$ . Thus, ${\mathcal{R}}_{{e}^{ * },s} \bowtie  {\mathcal{R}}_{{e}^{ + },s} \bowtie  {CP}\left( {\mathcal{D}}_{\text{heavy }}\right)$ equals ${\mathcal{R}}_{{e}^{ * },s} \bowtie  {\mathcal{D}}_{j} \bowtie  {\mathcal{R}}_{{e}^{ + },s} \bowtie$ ${CP}\left( {\mathcal{D}}_{\text{heavy }}\right)$ ,which is simply ${\mathcal{R}}^{ + } \bowtie  {CP}\left( {\mathcal{D}}_{\text{heavy }}\right)$ . This proves the correctness of (24). The case where ${e}^{ + } \notin  {\mathcal{E}}_{s}$ is similar.

首先考虑${e}^{ + } \in  {\mathcal{E}}_{s}$的情况。回顾${\mathcal{D}}_{j} \in  {\mathcal{Q}}_{\text{heavy }}$。因此，${\mathcal{R}}_{{e}^{ * },s} \bowtie  {\mathcal{R}}_{{e}^{ + },s} \bowtie  {CP}\left( {\mathcal{D}}_{\text{heavy }}\right)$等于${\mathcal{R}}_{{e}^{ * },s} \bowtie  {\mathcal{D}}_{j} \bowtie  {\mathcal{R}}_{{e}^{ + },s} \bowtie$ ${CP}\left( {\mathcal{D}}_{\text{heavy }}\right)$，即${\mathcal{R}}^{ + } \bowtie  {CP}\left( {\mathcal{D}}_{\text{heavy }}\right)$。这证明了(24)的正确性。${e}^{ + } \notin  {\mathcal{E}}_{s}$的情况类似。

---

<!-- Footnote -->

${}^{4}$ Obviously,by replacing $s$ with $s + 1$ .

${}^{4}$ 显然，用$s + 1$替换$s$即可。

<!-- Footnote -->

---

The next lemma shows that the above inductive generative process will terminate eventually:

接下来的引理表明，上述归纳生成过程最终会终止：

LEMMA 7.7. The sequence of queries ${\mathcal{Q}}_{0},{\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{\ell }$ is finite.

引理7.7。查询序列${\mathcal{Q}}_{0},{\mathcal{Q}}_{1},\ldots ,{\mathcal{Q}}_{\ell }$是有限的。

Proof. Recall that the generation proceeds differently in two cases depending on ${\Delta }_{s}$ . It suffices to show that each case can occur only a finite number of times.

证明。回顾一下，根据${\Delta }_{s}$的不同，生成过程在两种情况下有所不同。只需证明每种情况只能发生有限次即可。

A new query is generated only if we can find $j \in  \left\lbrack  b\right\rbrack$ such that ${\mathcal{F}}_{s}\left( {Y}_{j}\right)  \neq  {\mathcal{F}}_{s}\left( {Z}_{j}\right)$ . Every time the case ${\Delta }_{s} < {x}_{{e}^{ * },s}$ happens, ${\Delta }_{s} = {\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right)$ . Therefore, ${\mathcal{F}}_{s + 1}\left( {Y}_{j}\right)  = {\mathcal{F}}_{s + 1}\left( {Z}_{j}\right)$ ,by the reasoning in the proof of Lemma 7.6. This means that ${\mathcal{F}}_{{s}^{\prime }}$ will remain the same as ${\mathcal{F}}_{{s}^{\prime }}\left( {Z}_{j}\right)$ for all ${s}^{\prime } \geq  s + 1$ . Therefore,this case can occur at most $b$ times.

仅当我们能找到满足${\mathcal{F}}_{s}\left( {Y}_{j}\right)  \neq  {\mathcal{F}}_{s}\left( {Z}_{j}\right)$的$j \in  \left\lbrack  b\right\rbrack$时，才会生成一个新的查询。每次出现${\Delta }_{s} < {x}_{{e}^{ * },s}$的情况时，${\Delta }_{s} = {\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right)$。因此，根据引理7.6的证明中的推理，${\mathcal{F}}_{s + 1}\left( {Y}_{j}\right)  = {\mathcal{F}}_{s + 1}\left( {Z}_{j}\right)$。这意味着对于所有的${s}^{\prime } \geq  s + 1$，${\mathcal{F}}_{{s}^{\prime }}$将与${\mathcal{F}}_{{s}^{\prime }}\left( {Z}_{j}\right)$保持相同。因此，这种情况最多发生$b$次。

For each relation $\mathcal{R} \in  {\mathcal{Q}}_{s}$ ,we define its imbalance count as the number of $j \in  \left\lbrack  b\right\rbrack$ values satisfying the condition that scheme(R) contains exactly one attribute in $\left\{  {{Y}_{j},{Z}_{j}}\right\}$ . The imbalance count of ${\mathcal{Q}}_{s}$ is the sum of imbalanced counts of all the relations therein.

对于每个关系$\mathcal{R} \in  {\mathcal{Q}}_{s}$，我们将其不平衡计数定义为满足方案(R)在$\left\{  {{Y}_{j},{Z}_{j}}\right\}$中恰好包含一个属性这一条件的$j \in  \left\lbrack  b\right\rbrack$值的数量。${\mathcal{Q}}_{s}$的不平衡计数是其中所有关系的不平衡计数之和。

- After an occurrence of the case ${\Delta }_{s} < {x}_{{e}^{ * },s}$ ,the imbalance count of ${\mathcal{Q}}_{s + 1}$ can increase by at most $b$ compared to that of ${\mathcal{Q}}_{s}$ ,due to the inclusion of ${\mathcal{R}}^{ + }$ .

- 在出现${\Delta }_{s} < {x}_{{e}^{ * },s}$的情况后，由于包含了${\mathcal{R}}^{ + }$，${\mathcal{Q}}_{s + 1}$的不平衡计数与${\mathcal{Q}}_{s}$相比最多增加$b$。

- After an occurrence of the case ${\Delta }_{s} = {x}_{{e}^{ * },s}$ ,the imbalance count of ${\mathcal{Q}}_{s + 1}$ must decrease by 1 compared to that of ${\mathcal{Q}}_{s}$ , due to the eviction of ${\mathcal{R}}_{{e}^{ * },s}$ .

- 在出现${\Delta }_{s} = {x}_{{e}^{ * },s}$的情况后，由于剔除了${\mathcal{R}}_{{e}^{ * },s}$，${\mathcal{Q}}_{s + 1}$的不平衡计数与${\mathcal{Q}}_{s}$相比必须减少1。

Given that the ${\Delta }_{s} < {x}_{{e}^{ * },s}$ case happens at most $b$ times,we conclude that ${\Delta }_{s} = {x}_{{e}^{ * },s}$ can occur at most $b\left( {\left| \mathcal{Q}\right|  + b}\right)$ times.

鉴于${\Delta }_{s} < {x}_{{e}^{ * },s}$情况最多发生$b$次，我们得出${\Delta }_{s} = {x}_{{e}^{ * },s}$最多可发生$b\left( {\left| \mathcal{Q}\right|  + b}\right)$次。

LEMMA 7.8. In the last query ${\mathcal{Q}}_{\ell }$ ,it must hold for each $j \in  \left\lbrack  b\right\rbrack$ that

引理7.8。在最后一次查询${\mathcal{Q}}_{\ell }$中，对于每个$j \in  \left\lbrack  b\right\rbrack$，必定有如下情况成立

$$
{\mathcal{F}}_{\ell }\left( {Y}_{j}\right)  = {\mathcal{F}}_{\ell }\left( {Z}_{j}\right)  = \max \left\{  {{\mathcal{F}}_{0}\left( {Y}_{j}\right) ,{\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right\}  .
$$

For any $A \notin  \left\{  {{Y}_{1},{Z}_{1},\ldots ,{Y}_{b},{Z}_{b}}\right\}$ ,it holds that ${\mathcal{F}}_{0}\left( A\right)  = {\mathcal{F}}_{1}\left( A\right)  =$ $\ldots  = {\mathcal{F}}_{\ell }\left( A\right)$ .

对于任意$A \notin  \left\{  {{Y}_{1},{Z}_{1},\ldots ,{Y}_{b},{Z}_{b}}\right\}$，有${\mathcal{F}}_{0}\left( A\right)  = {\mathcal{F}}_{1}\left( A\right)  =$ $\ldots  = {\mathcal{F}}_{\ell }\left( A\right)$成立。

Proof. The generative process must continue if ${\mathcal{F}}_{\ell }\left( {X}_{j}\right)  \neq$ ${\mathcal{F}}_{\ell }\left( {Y}_{j}\right)$ ; this proves ${\mathcal{F}}_{\ell }\left( {Y}_{j}\right)  = {\mathcal{F}}_{\ell }\left( {Z}_{j}\right)$ . Assume,without loss of generality,that ${\mathcal{F}}_{0}\left( {Y}_{j}\right)  > {\mathcal{F}}_{0}\left( {Z}_{j}\right)$ . By the argument in the proof of Lemma 7.6,we know ${\mathcal{F}}_{0}\left( {Y}_{j}\right)  = {\mathcal{F}}_{1}\left( {Y}_{j}\right)  = \ldots  = {\mathcal{F}}_{\ell }\left( {Y}_{j}\right)$ . This proves the correctness of the first sentence of the lemma.

证明。如果${\mathcal{F}}_{\ell }\left( {X}_{j}\right)  \neq$ ${\mathcal{F}}_{\ell }\left( {Y}_{j}\right)$，生成过程必须继续；这证明了${\mathcal{F}}_{\ell }\left( {Y}_{j}\right)  = {\mathcal{F}}_{\ell }\left( {Z}_{j}\right)$。不失一般性，假设${\mathcal{F}}_{0}\left( {Y}_{j}\right)  > {\mathcal{F}}_{0}\left( {Z}_{j}\right)$。根据引理7.6证明中的论证，我们知道${\mathcal{F}}_{0}\left( {Y}_{j}\right)  = {\mathcal{F}}_{1}\left( {Y}_{j}\right)  = \ldots  = {\mathcal{F}}_{\ell }\left( {Y}_{j}\right)$。这证明了引理第一句话的正确性。

The second sentence follows directly from the way we design ${x}_{e,s}$ .

第二句话直接由我们对${x}_{e,s}$的设计方式得出。

Bounding the size of ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ Join $\left( {\mathcal{Q}}_{\ell }\right)$ . For each $s \in  \left\lbrack  {0,\ell }\right\rbrack$ , define:

界定${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$连接$\left( {\mathcal{Q}}_{\ell }\right)$的规模。对于每个$s \in  \left\lbrack  {0,\ell }\right\rbrack$，定义：

$$
{\mathcal{B}}_{s} = \mathop{\prod }\limits_{{e \in  {\mathcal{E}}_{s}}}{\left| {\mathcal{R}}_{e,s}\right| }^{{x}_{e,s}}
$$

$$
\Delta  = \mathop{\sum }\limits_{{j \in  \left\lbrack  b\right\rbrack  }}\left| {{\mathcal{F}}_{0}\left( {Y}_{j}\right)  - {\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right| .
$$

LEMMA 7.9.

引理7.9。

$$
{\mathcal{B}}_{\ell } \leq  {\mathcal{B}}_{0} \cdot  {\lambda }^{\Delta }.
$$

Proof. Let us start by observing:

证明。让我们先观察：

$$
\Delta  = \mathop{\sum }\limits_{{s \in  \left\lbrack  {0,\ell  - 1}\right\rbrack  }}{\Delta }_{s} \tag{25}
$$

To explain why,consider an arbitrary $s \in  \left\lbrack  {0,\ell  - 1}\right\rbrack$ . If $j$ is the triggering index of ${\mathcal{Q}}_{s + 1},\left| {{\mathcal{F}}_{s + 1}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s + 1}\left( {Z}_{j}\right) }\right|$ must decrease by ${\Delta }_{s}$ compared to $\left| {{\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right) }\right|$ . The correctness of (25) then follows from Lemma 7.8.

为了解释原因，考虑任意的$s \in  \left\lbrack  {0,\ell  - 1}\right\rbrack$。如果$j$是${\mathcal{Q}}_{s + 1},\left| {{\mathcal{F}}_{s + 1}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s + 1}\left( {Z}_{j}\right) }\right|$的触发索引，那么与$\left| {{\mathcal{F}}_{s}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s}\left( {Z}_{j}\right) }\right|$相比，${\mathcal{Q}}_{s + 1},\left| {{\mathcal{F}}_{s + 1}\left( {Y}_{j}\right)  - {\mathcal{F}}_{s + 1}\left( {Z}_{j}\right) }\right|$必定减少${\Delta }_{s}$。那么(25)的正确性可由引理7.8得出。

Equipped with (25), to prove the lemma it suffices to show:

有了(25)，要证明该引理，只需证明：

$$
{\mathcal{B}}_{s + 1} \leq  {\mathcal{B}}_{s} \cdot  {\lambda }^{{\Delta }_{s}} \tag{26}
$$

holds for every $s \in  \left\lbrack  {0,\ell  - 1}\right\rbrack$ . For this purpose,let us scrutinize once again the two cases that happen in generating ${\mathcal{Q}}_{s + 1}$ . As before,let $j$ be the triggering index and ${e}^{ * }$ be the triggering edge. We will first consider the scenario where ${e}^{ + } \notin  {\mathcal{E}}_{s}$ .

对于每个$s \in  \left\lbrack  {0,\ell  - 1}\right\rbrack$都成立。为此，让我们再次仔细研究生成${\mathcal{Q}}_{s + 1}$时出现的两种情况。和之前一样，设$j$为触发索引，${e}^{ * }$为触发边。我们首先考虑${e}^{ + } \notin  {\mathcal{E}}_{s}$的情况。

- Case ${\Delta }_{s} < {x}_{{e}^{ * },s}$ : We have

- 情况${\Delta }_{s} < {x}_{{e}^{ * },s}$：我们有

$$
{\mathcal{B}}_{s + 1} = {\mathcal{B}}_{s} \cdot  \frac{{\left| {\mathcal{R}}_{{e}^{ * },s}\right| }^{{x}_{{e}^{ * },s} - {\Delta }_{s}} \cdot  {\left| {\mathcal{R}}^{ + }\right| }^{{\Delta }_{s}}}{{\left| {\mathcal{R}}_{{e}^{ * },s}\right| }^{{x}_{{e}^{ * },s}}} \tag{27}
$$

where ${e}^{ + }$ and ${\mathcal{R}}^{ + }$ are defined in (22) and (23),respectively. In general, it holds that

其中${e}^{ + }$和${\mathcal{R}}^{ + }$分别在(22)和(23)中定义。一般来说，有如下情况成立

$$
\left| {\mathcal{R}}^{ + }\right|  \leq  \left| {\mathcal{R}}_{{e}^{ * },s}\right|  \cdot  \lambda  \tag{28}
$$

To understand why,first notice that less than $\lambda$ tuples in ${\mathcal{D}}_{j}$ can share the same value on ${Y}_{j}.{}^{5}$ This,in turn,indicates that a tuple ${\mathcal{R}}_{{e}^{ * },s}$ can join with less than $\lambda$ tuples in ${\mathcal{D}}_{j}$ ,which yields (28). Putting together (27) and (28) validates (26).

为了理解原因，首先注意到，在${\mathcal{D}}_{j}$中，少于$\lambda$个元组可以在${Y}_{j}.{}^{5}$上共享相同的值。这反过来表明，一个元组${\mathcal{R}}_{{e}^{ * },s}$可以与${\mathcal{D}}_{j}$中少于$\lambda$个元组进行连接，由此得到(28)式。将(27)式和(28)式结合起来，就验证了(26)式。

- Case ${\Delta }_{s} = {x}_{{e}^{ * },s}$ :

- 情况${\Delta }_{s} = {x}_{{e}^{ * },s}$：

$$
{\mathcal{B}}_{s + 1} = {\mathcal{B}}_{s} \cdot  \frac{{\left| {\mathcal{R}}^{ + }\right| }^{{\Delta }_{s}}}{{\left| {\mathcal{R}}_{{e}^{ * },s}\right| }^{{\Delta }_{s}}}
$$

which together with (28) validates (26).

这与(28)式一起验证了(26)式。

The scenario where ${e}^{ + } \in  {\mathcal{E}}_{s}$ is similar:

${e}^{ + } \in  {\mathcal{E}}_{s}$的情况与之类似：

- Case ${\Delta }_{s} < {x}_{{e}^{ * },s}$ :

- 情况${\Delta }_{s} < {x}_{{e}^{ * },s}$：

$$
{\mathcal{B}}_{s + 1} = {\mathcal{B}}_{s} \cdot  \frac{{\left| {\mathcal{R}}_{{e}^{ * },s}\right| }^{{x}_{{e}^{ * },s} - {\Delta }_{s}} \cdot  {\left| {\mathcal{R}}^{ + }\right| }^{{\Delta }_{s} + {x}_{{e}^{ + },s}}}{{\left| {\mathcal{R}}_{{e}^{ * },s}\right| }^{{x}_{{e}^{ * },s}} \cdot  {\left| {\mathcal{R}}_{{e}^{ + },s}\right| }^{{x}_{{e}^{ + },s}}} \tag{29}
$$

- Case ${\Delta }_{s} = {x}_{{e}^{ * },s}$ :

- 情况${\Delta }_{s} = {x}_{{e}^{ * },s}$：

$$
{\mathcal{B}}_{s + 1} = {\mathcal{B}}_{s} \cdot  \frac{{\left| {\mathcal{R}}^{ + }\right| }^{{\Delta }_{s} + {x}_{{e}^{ + },s}}}{{\left| {\mathcal{R}}_{{e}^{ * },s}\right| }^{{\Delta }_{s}} \cdot  {\left| {\mathcal{R}}_{{e}^{ + },s}\right| }^{{x}_{{e}^{ + },s}}} \tag{30}
$$

Both (29) and (30) give (26),using (28) and the fact that $\left| {\mathcal{R}}^{ + }\right|  \leq$ $\left| {\mathcal{R}}_{{e}^{ + },s}\right|$ . The completes the proof of Lemma 7.9.

利用(28)式以及$\left| {\mathcal{R}}^{ + }\right|  \leq$ $\left| {\mathcal{R}}_{{e}^{ + },s}\right|$这一事实，(29)式和(30)式都能推出(26)式。至此，引理7.9的证明完成。

We are ready to bound the size of ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{\ell }\right)$ . Towards this purpose,for each attribute $A \in  \mathcal{J}$ ,let us create a "domain" relation ${u}_{A}$ which includes all the $A$ -values that appear in the input relations of2. Clearly, $\left| {\mathcal{U}}_{A}\right|  \leq  n$ . Now,define a clean query:

我们准备对${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{\ell }\right)$的大小进行界定。为此，对于每个属性$A \in  \mathcal{J}$，我们创建一个“域”关系${u}_{A}$，它包含出现在2的输入关系中的所有$A$值。显然，$\left| {\mathcal{U}}_{A}\right|  \leq  n$。现在，定义一个简洁查询：

$$
{\mathcal{2}}_{\text{final }} = {\mathcal{2}}_{\text{heavy }} \cup  {\mathcal{2}}_{\ell } \cup  \left( {\mathop{\bigcup }\limits_{{A \in  \mathcal{J}}}\left\{  {\mathcal{U}}_{A}\right\}  }\right) .
$$

We prove in Appendix F:

我们在附录F中证明：

Proposition 7.10. Join $\left( {\mathcal{Q}}_{\text{final }}\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{\ell }\right)$ .

命题7.10. 连接$\left( {\mathcal{Q}}_{\text{final }}\right)  = \operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}_{\ell }\right)$。

---

<!-- Footnote -->

${}^{5}$ Recall that each tuple(y,z)in ${\mathcal{D}}_{j}$ must appear in at least $n/{\lambda }^{2}$ tuples of the input relations of2. Hence,if $y$ can pair up with at least $\lambda$ distinct $z,y$ appears in at least $\frac{n}{{\lambda }^{2}}\lambda  = n/\lambda$ tuples. This contradicts the fact that $y$ is a light value.

${}^{5}$ 回顾一下，${\mathcal{D}}_{j}$中的每个元组(y,z)必须出现在2的输入关系的至少$n/{\lambda }^{2}$个元组中。因此，如果$y$可以与至少$\lambda$个不同的$z,y$配对，那么它至少出现在$\frac{n}{{\lambda }^{2}}\lambda  = n/\lambda$个元组中。这与$y$是一个轻值这一事实相矛盾。

<!-- Footnote -->

---

Hence,instead of the size of ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ Join $\left( {\mathcal{Q}}_{\ell }\right)$ ,we can focus on bounding $\left| {\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|$ :

因此，我们可以不考虑${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$连接$\left( {\mathcal{Q}}_{\ell }\right)$的大小，而是专注于界定$\left| {\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|$：

LEMMA 7.11.

引理7.11。

$$
\left| {\operatorname{Join}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|  \leq  {n}^{\left| \mathcal{J}\right| } \cdot  {\lambda }^{\left| \mathcal{H}\right|  - \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}{x}_{e}\left( {\left| e\right|  - 1}\right) }. \tag{31}
$$

Proof. Let ${\mathcal{G}}_{\text{final }}$ be the hypergraph defined by ${\mathcal{D}}_{\text{final }}$ . We will construct a fractional edge covering $\mathcal{W}$ of ${\mathcal{G}}_{\text{final }}$ ,which by the AGM bound in Lemma 3.2 yields an upper bound on $\left| {\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|$ which will turn out to be the right hand side of (31).

证明。设${\mathcal{G}}_{\text{final }}$是由${\mathcal{D}}_{\text{final }}$定义的超图。我们将构造${\mathcal{G}}_{\text{final }}$的一个分数边覆盖$\mathcal{W}$，根据引理3.2中的AGM界，这将给出$\left| {\mathcal{J}\operatorname{oin}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|$的一个上界，结果将是(31)式的右边。

The construction of $\mathcal{W}$ is based on the feasible assignment $\left\{  {{x}_{e,\ell } \mid  }\right.$ $\left. {e \in  {\mathcal{E}}_{\ell }}\right\}$ of the characterizing program of ${\mathcal{G}}_{\ell }$ :

$\mathcal{W}$的构造基于${\mathcal{G}}_{\ell }$的特征规划的可行赋值$\left\{  {{x}_{e,\ell } \mid  }\right.$ $\left. {e \in  {\mathcal{E}}_{\ell }}\right\}$：

- For each edge $e$ of ${\mathcal{G}}_{\text{final }}$ ,we set $\mathcal{W}\left( e\right)  = {x}_{e,\ell }$ .

- 对于${\mathcal{G}}_{\text{final }}$的每条边$e$，我们设$\mathcal{W}\left( e\right)  = {x}_{e,\ell }$。

- For the unary edge $\{ A\}$ where $A \in  \mathcal{J}$ ,set $\mathcal{W}\left( {\{ A\} }\right)  = 1 -$ ${\mathcal{F}}_{\ell }\left( A\right)$ . By the feasibility of $\left\{  {{x}_{e,\ell } \mid  e \in  {\mathcal{E}}_{\ell }}\right\}  ,{\mathcal{F}}_{\ell }\left( A\right)  \leq  1$ ; and hence, $\mathcal{W}\left( {\{ A\} }\right)  \geq  0$ .

- 对于一元边$\{ A\}$，其中$A \in  \mathcal{J}$，设$\mathcal{W}\left( {\{ A\} }\right)  = 1 -$ ${\mathcal{F}}_{\ell }\left( A\right)$。由$\left\{  {{x}_{e,\ell } \mid  e \in  {\mathcal{E}}_{\ell }}\right\}  ,{\mathcal{F}}_{\ell }\left( A\right)  \leq  1$的可行性可知；因此，$\mathcal{W}\left( {\{ A\} }\right)  \geq  0$。

- For the unary edge $\left\{  {X}_{i}\right\}$ where $i \in  \left\lbrack  a\right\rbrack$ ,set $\mathcal{W}\left( \left\{  {X}_{i}\right\}  \right)  =$ $1 - {\mathcal{F}}_{\ell }\left( {X}_{i}\right)$ .

- 对于一元边$\left\{  {X}_{i}\right\}$，其中$i \in  \left\lbrack  a\right\rbrack$，设$\mathcal{W}\left( \left\{  {X}_{i}\right\}  \right)  =$ $1 - {\mathcal{F}}_{\ell }\left( {X}_{i}\right)$。

- For the binary edge $\left\{  {{Y}_{j},{Z}_{j}}\right\}$ where $j \in  \left\lbrack  b\right\rbrack$ ,set $\mathcal{W}\left( \left\{  {{Y}_{j},{Z}_{j}}\right\}  \right)  =$ $1 - {\mathcal{F}}_{\ell }\left( {Y}_{j}\right)$ ,which must be equal to $1 - {\mathcal{F}}_{\ell }\left( {Z}_{j}\right)$ by Lemma 7.8.

- 对于二元边$\left\{  {{Y}_{j},{Z}_{j}}\right\}$，其中$j \in  \left\lbrack  b\right\rbrack$，设$\mathcal{W}\left( \left\{  {{Y}_{j},{Z}_{j}}\right\}  \right)  =$ $1 - {\mathcal{F}}_{\ell }\left( {Y}_{j}\right)$，根据引理7.8，它必定等于$1 - {\mathcal{F}}_{\ell }\left( {Z}_{j}\right)$。

The weight of $A$ under $\mathcal{W}$ equals exactly 1 for every $A \in$ attset $\left( {\mathcal{Q}}_{\text{final }}\right)  = \mathcal{J} \cup  \mathcal{H}$ ,namely, $\mathcal{W}$ is an edge covering for ${\mathcal{G}}_{\text{final }}$ .

对于每个$A \in$属性集$\left( {\mathcal{Q}}_{\text{final }}\right)  = \mathcal{J} \cup  \mathcal{H}$，$A$在$\mathcal{W}$下的权重恰好等于1，即$\mathcal{W}$是${\mathcal{G}}_{\text{final }}$的一个边覆盖。

By Lemma 3.2, $\left| {\operatorname{Join}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|$ is bounded by

根据引理3.2，$\left| {\operatorname{Join}\left( {\mathcal{Q}}_{\text{final }}\right) }\right|$有界于

$$
\mathop{\prod }\limits_{{A \in  \mathcal{J}}}{n}^{1 - {\mathcal{F}}_{l}\left( A\right) }\mathop{\prod }\limits_{{e \in  {\mathcal{G}}_{\ell }}}{\left| {\mathcal{R}}_{e,\ell }\right| }^{{x}_{e,\ell }}\mathop{\prod }\limits_{{i = 1}}^{a}{\left| {\mathcal{S}}_{i}\right| }^{1 - {\mathcal{F}}_{l}\left( {X}_{i}\right) }\mathop{\prod }\limits_{{j = 1}}^{b}{\left| {\mathcal{D}}_{j}\right| }^{1 - {\mathcal{F}}_{l}\left( {Y}_{j}\right) }
$$

$$
 = {n}^{\left| \mathcal{J}\right|  - \mathop{\sum }\limits_{{A \in  \mathcal{J}}}{\mathcal{F}}_{l}\left( A\right) } \cdot  {\mathcal{B}}_{\ell } \cdot  {\lambda }^{a - \mathop{\sum }\limits_{{i = 1}}^{a}{\mathcal{F}}_{\ell }\left( {X}_{i}\right) } \cdot  {\lambda }^{2\mathop{\sum }\limits_{{j = 1}}^{b}\left( {1 - {\mathcal{F}}_{\ell }\left( {Y}_{j}\right) }\right) }. \tag{32}
$$

Utilizing Lemmas 7.8 and 7.9, we can derive:

利用引理7.8和7.9，我们可以推导出：

$$
{\mathcal{B}}_{\ell } \cdot  {\lambda }^{2}\mathop{\sum }\limits_{{j = 1}}^{b}\left( {1 - {\mathcal{F}}_{\ell }\left( {Y}_{j}\right) }\right) 
$$

$$
 \leq  {\mathcal{B}}_{0} \cdot  {\lambda }^{\Delta } \cdot  {\lambda }^{\mathop{\sum }\limits_{{j = 1}}^{b}2 - 2\max \left\{  {{\mathcal{F}}_{0}\left( {Y}_{j}\right) ,{\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right\}  }
$$

$$
 = {\mathcal{B}}_{0} \cdot  {\lambda }^{\mathop{\sum }\limits_{{j = 1}}^{b}2 - 2\max \left\{  {{\mathcal{F}}_{0}\left( {Y}_{j}\right) ,{\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right\}   + \left| {{\mathcal{F}}_{0}\left( {Y}_{j}\right)  - {\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right| }
$$

$$
 = {\mathcal{B}}_{0} \cdot  {\lambda }^{\mathop{\sum }\limits_{{j = 1}}^{b}\left( {1 - {\mathcal{F}}_{0}\left( {Y}_{j}\right)  + \left( {1 - {\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right) }\right. }
$$

$$
 \leq  \left( {\mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * }}}{n}^{{x}_{e}}}\right)  \cdot  {\lambda }^{{2b} - \mathop{\sum }\limits_{{j = 1}}^{b}\left( {{\mathcal{F}}_{0}\left( {Y}_{j}\right)  + {\mathcal{F}}_{0}\left( {Z}_{j}\right) }\right) }.
$$

Plugging the above into (32) and applying ${\mathcal{F}}_{\ell }\left( {X}_{i}\right)  = {\mathcal{F}}_{0}\left( {X}_{i}\right)$ for each $i \in  \left\lbrack  a\right\rbrack$ (Lemma 7.8) yields:

将上述内容代入(32)式，并对每个$i \in  \left\lbrack  a\right\rbrack$应用${\mathcal{F}}_{\ell }\left( {X}_{i}\right)  = {\mathcal{F}}_{0}\left( {X}_{i}\right)$（引理7.8），可得：

$$
\left( {32}\right)  \leq  {n}^{\left| \mathcal{J}\right|  - \mathop{\sum }\limits_{{A \in  \mathcal{J}}}{\mathcal{F}}_{l}\left( A\right) } \cdot  \left( {\mathop{\prod }\limits_{{e \in  {\mathcal{E}}^{ * }}}{n}^{{x}_{e}}}\right)  \cdot  {\lambda }^{\left| \mathcal{H}\right|  - \mathop{\sum }\limits_{{A \in  \mathcal{H}}}{\mathcal{F}}_{0}\left( A\right) }. \tag{33}
$$

By Property (1) of Lemma 7.2,every $e \in  {\mathcal{E}}^{ * }$ covers exactly one attribute in $\mathcal{J}$ . Thus:

根据引理7.2的性质(1)，每个$e \in  {\mathcal{E}}^{ * }$恰好覆盖$\mathcal{J}$中的一个属性。因此：

$$
\mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}{x}_{e} = \mathop{\sum }\limits_{{A \in  \mathcal{J}}}{\mathcal{F}}_{0}\left( A\right)  = \mathop{\sum }\limits_{{A \in  \mathcal{J}}}{\mathcal{F}}_{\ell }\left( A\right) 
$$

where the last equality used Lemma 7.8. Furthermore:

其中最后一个等式使用了引理7.8。此外：

$$
\mathop{\sum }\limits_{{A \in  \mathcal{H}}}{\mathcal{F}}_{0}\left( A\right)  \geq  \mathop{\sum }\limits_{{A \in  \mathcal{H}}}\mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * } : A \in  e}}{x}_{e} = \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}{x}_{e}\left( {\left| e\right|  - 1}\right) 
$$

where the last equality used Property (3) of Lemma 7.2. Therefore, from (33), we get

其中最后一个等式使用了引理7.2的性质(3)。因此，由(33)式，我们得到

$$
\text{(32)} \leq  {n}^{\left| \mathcal{J}\right| } \cdot  {\lambda }^{\left| \mathcal{H}\right|  - \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}{x}_{e}\left( {\left| e\right|  - 1}\right) }
$$

which completes the proof.

至此，证明完毕。

By combining Lemmas 7.11, 7.6 and Propositions 7.10 and 7.5, we conclude the proof of Lemma 7.2.

通过结合引理7.11、7.6以及命题7.10和7.5，我们完成了引理7.2的证明。

## 8 AN MPC JOIN ALGORITHM

## 8 一种多方计算连接算法

Next, we will describe our MPC algorithm for answering an arbitrary query2. It suffices to consider that2is clean (Section 3.1) because otherwise $\mathcal{Q}$ can be converted to a clean query with the same result in load $\widetilde{O}\left( {n/p}\right) \left\lbrack  {14}\right\rbrack$ .

接下来，我们将描述用于回答任意查询$q_2$的多方计算算法。考虑$q_2$是干净的情况（第3.1节）就足够了，因为否则可以在负载$O(\log n)$下将$q_2$转换为具有相同结果的干净查询。

Specifically,we will fix an arbitrary plan $\mathbf{P}$ of $\mathcal{Q}$ ,and explain how to compute

具体来说，我们将固定$q_2$的任意一个计划$\pi$，并解释如何计算

${\mathcal{Q}}^{\prime }\left( {\mathcal{H},h}\right)$

full config.(H,h)of $\mathbf{P}$

$\pi$的完整配置$(H,h)$

where the equality is due to Proposition 6.1. By taking the union of the above for every $P$ ,we obtain the final result Foin(Q) (Lemma 5.2). We will prove an identical upper bound on the load for all $\mathbf{P}$ . Since $\mathcal{Q}$ has $O\left( 1\right)$ plans,processing all of them concurrently can increase the load only by a constant factor.

其中等式成立是由于命题6.1。通过对每个$\pi$取上述结果的并集，我们得到最终结果$Foin(Q)$（引理5.2）。我们将证明对于所有$q_2$的计划，负载有相同的上界。由于$q_2$有$O(\log n)$个计划，同时处理所有计划只会使负载增加一个常数因子。

We will first discuss the scenario where $\mathcal{Q}$ is unary-free. Extending the algorithm to allow unary relations is easy, as will be shown in Appendix G.

我们首先讨论$q_2$不含一元关系的情况。将该算法扩展以允许一元关系很容易，如附录G所示。

From now, we will fix

从现在起，我们固定

$$
\lambda  = {p}^{1/\left( {\alpha \phi }\right) }. \tag{34}
$$

In general,it holds that ${}^{6}$

一般来说，有$\tau(q_2)\leq \tau^*(q_2)$

$$
k \leq  {\alpha \rho } \leq  {\alpha \phi } \tag{35}
$$

where $\rho$ is the fractional edge covering number of2. By Proposition 5.1,the number of full configurations of $\mathbf{P}$ is at most

其中$\tau^*(q_2)$是$q_2$的分数边覆盖数。根据命题5.1，$\pi$的完整配置的数量至多为

$$
{\lambda }^{\left| \mathcal{H}\right| } \leq  {\lambda }^{k} = {p}^{k/\left( {\alpha \phi }\right) } \leq  p.
$$

Without loss of generality,we assume that,given a tuple $\mathbf{u}$ in any input relation of $\mathcal{Q}$ ,the machine can identify all heavy values and value-pairs contained in $\mathbf{u}$ . This can be achieved with the techniques of [11] which essentially sort the input relations a constant number of times,incurring an extra load of $\widetilde{O}\left( {n/p}\right)$ .

不失一般性，我们假设，给定$q_2$的任何输入关系中的一个元组$t$，机器可以识别$t$中包含的所有重值和值对。这可以通过文献[11]中的技术实现，该技术本质上对输入关系进行常数次数的排序，产生额外的负载$O(\log n)$。

Step 1: Generating the input relations of the residual queries. Recall (from Section 5) that a residual query ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is defined for every full configuration(H,h). Denote by ${n}_{\mathcal{H},\mathbf{h}}$ the input size of ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . All the ${n}_{\mathcal{H},\mathbf{h}}$ values can be obtained with sorting and broadcast to all machines in load $\widetilde{O}\left( {p + n/p}\right)  = \widetilde{O}\left( {n/p}\right)$ .

步骤1：生成残差查询的输入关系。回顾（从第5节）可知，对于每个完整配置$(H,h)$都定义了一个残差查询$q_{(H,h)}$。用$N_{(H,h)}$表示$q_{(H,h)}$的输入大小。所有$N_{(H,h)}$的值可以通过排序得到，并以负载$O(\log n)$广播到所有机器。

We allocate

我们分配

$$
{p}_{\mathcal{H},\mathbf{h}}^{\prime } = p \cdot  \frac{{n}_{\mathcal{H},\mathbf{h}}}{\Theta \left( {n \cdot  {\lambda }^{k - 2}}\right) }
$$

machines to store the relations of ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . By Corollary 5.4,the total number of machines needed is at most $p$ . The number of tuples received by each machine is bounded by

用于存储${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$关系的机器。根据推论5.4，所需机器的总数最多为$p$。每台机器接收的元组数量受限于

$$
O\left( \frac{{n}_{\mathcal{H},\mathbf{h}}}{{p}_{\mathcal{H},\mathbf{h}}^{\prime }}\right)  = O\left( \frac{n \cdot  {\lambda }^{k - 2}}{p}\right)  = O\left( \frac{n \cdot  {p}^{\frac{k - 2}{\alpha \phi }}}{p}\right)  = O\left( \frac{n}{{p}^{\frac{2}{\alpha \phi }}}\right) 
$$

where the last equality used (35).

其中最后一个等式使用了(35)。

---

<!-- Footnote -->

${}^{6}$ The first inequality is by Lemma 3.1,and the second is due to fact $\rho  \leq  \phi$ (shown in the proof of Lemma 4.3).

${}^{6}$第一个不等式由引理3.1得出，第二个不等式则基于事实$\rho  \leq  \phi$（在引理4.3的证明中已展示）。

<!-- Footnote -->

---

Step 2: Simplifying the residual queries. In this step, for each ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ,its ${p}_{\mathcal{H},\mathbf{h}}^{\prime }$ assigned machines work together to produce the simplified residual query ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ as given in (18). This requires only set intersection (for (14)) and semi-join reduction (for (15)),both of which can be performed with load $O\left( {{n}_{\mathcal{H},\mathbf{h}}/{p}_{\mathcal{H},\mathbf{h}}^{\prime }}\right)  =$ $O\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ [14]. After this,the size of ${CP}\left( {{\mathcal{Q}}_{T}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ is known for each(H,h); those at most $p\mathrm{{CP}}$ sizes are broadcast to all $p$ machines with load $O\left( p\right)  = O\left( {n/p}\right)$ .

步骤2：简化剩余查询。在这一步中，对于每个${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$，其分配的${p}_{\mathcal{H},\mathbf{h}}^{\prime }$台机器共同协作，以生成如(18)中所给出的简化剩余查询${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$。这仅需要集合交集运算（针对(14)）和半连接约简（针对(15)），这两种操作都可以在负载为$O\left( {{n}_{\mathcal{H},\mathbf{h}}/{p}_{\mathcal{H},\mathbf{h}}^{\prime }}\right)  =$ $O\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$的情况下执行[14]。在此之后，对于每个(H,h)，${CP}\left( {{\mathcal{Q}}_{T}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$的大小是已知的；那些最多为$p\mathrm{{CP}}$的大小会以负载$O\left( p\right)  = O\left( {n/p}\right)$广播到所有$p$台机器。

Step 3: Computing the simplified residual queries. For each (H,h),we allocate

步骤3：计算简化后的剩余查询。对于每个(H,h)，我们分配

$$
{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime } = \Theta \left( {{\lambda }^{\left| \mathcal{L}\right| } + p\mathop{\sum }\limits_{{\text{non-empty }\mathcal{J} \subseteq  I}}\frac{\left| \operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathcal{H},\mathbf{h}\right) \right) \right| }{\Theta \left( {{\lambda }^{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) } - \left| {\mathcal{L} \smallsetminus  \mathcal{J}}\right|  \cdot  {n}^{\left| \mathcal{J}\right| }}\right) }}\right)  \tag{36}
$$

machines to process ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ,where ${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ is defined in (19). By Theorem 7.1 and the fact $\mathop{\sum }\limits_{\left( \mathcal{H},\mathbf{h}\right) }{\lambda }^{\left| \mathcal{L}\right| } \leq  {\lambda }^{\left| \mathcal{H}\right|  + \left| \mathcal{L}\right| } \leq  {\lambda }^{k} \leq  p$ , we can adjust the hidden constants to make sure $\mathop{\sum }\limits_{\left( \mathcal{H},\mathbf{h}\right) }{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime } \leq  p$ .

机器来处理${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$，其中${\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$在(19)中定义。根据定理7.1和事实$\mathop{\sum }\limits_{\left( \mathcal{H},\mathbf{h}\right) }{\lambda }^{\left| \mathcal{L}\right| } \leq  {\lambda }^{\left| \mathcal{H}\right|  + \left| \mathcal{L}\right| } \leq  {\lambda }^{k} \leq  p$，我们可以调整隐藏常数以确保$\mathop{\sum }\limits_{\left( \mathcal{H},\mathbf{h}\right) }{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime } \leq  p$。

LEMMA 8.1. ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ can be answered with load $O\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ using ${p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }$ machines.

引理8.1。可以使用${p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }$台机器以负载$O\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$回答${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$。

Proof. Join $\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ is the cartesian product of ${CP}\left( {{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ and $\oint \operatorname{oin}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . If $\mathcal{I} \neq  \varnothing$ ,use ${p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$ machines to compute $\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{I}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . By Lemma 3.3,the load is

证明。连接 $\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 是 ${CP}\left( {{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 和 $\oint \operatorname{oin}\left( {{\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 的笛卡尔积。如果 $\mathcal{I} \neq  \varnothing$，使用 ${p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$ 台机器来计算 $\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{I}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。根据引理3.3，负载为

$$
\widetilde{O}\left( \frac{\left| {\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right) }\right| {}^{1/\left| \mathcal{J}\right| }}{{\left( \frac{{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }}{{\lambda }^{\left| \mathcal{L}\right| ,\left| \mathcal{I}\right| }}\right) }^{1/\left| \mathcal{J}\right| }}\right)  \tag{37}
$$

for some non-empty $\mathcal{J} \subseteq  \mathcal{I}$ . (36) guarantees that

对于某些非空的 $\mathcal{J} \subseteq  \mathcal{I}$。(36) 保证了

$$
{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime } = \Omega \left( {p \cdot  \frac{\left| \operatorname{CP}\left( {\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( \mathcal{H},\mathbf{h}\right) \right) \right| }{\Theta \left( {{\lambda }^{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) } - \left| {\mathcal{L} \smallsetminus  \mathcal{J}}\right|  \cdot  {n}^{\left| \mathcal{J}\right| }}\right) }}\right) 
$$

with which we can derive

由此我们可以推导出

$$
\text{(37)} = \widetilde{O}\left( \frac{n \cdot  {\lambda }^{\frac{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right)  - \left| I\right|  + \left| \mathcal{J}\right| }{\left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{n \cdot  {p}^{\frac{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) }{{\alpha \phi }\left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{n}{{p}^{1/\phi }}\right) 
$$

which is $O\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ because $\alpha  \geq  2$ . Regarding ${\mathcal{D}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ,all of its relations are two-attribute skew free if a share of $\lambda$ is assigned to each attribute in $\mathcal{L} \smallsetminus  I$ . By Lemma 3.5, ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ can be solved with load $\widetilde{O}\left( {n/{\lambda }^{2}}\right)  = \widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ using ${\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$ machines.

这是 $O\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$，因为 $\alpha  \geq  2$。对于 ${\mathcal{D}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$，如果将 $\lambda$ 的一部分分配给 $\mathcal{L} \smallsetminus  I$ 中的每个属性，那么它的所有关系都是无两属性倾斜的。根据引理3.5，使用 ${\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$ 台机器可以以负载 $\widetilde{O}\left( {n/{\lambda }^{2}}\right)  = \widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 解决 ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$。

By combining the above with Lemma 3.4, we conclude that Join $\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ can be computed with load $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ using $\left( {{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }}\right)  \cdot  {\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| } = {p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }$ machines.

将上述内容与引理3.4相结合，我们得出结论：使用 $\left( {{p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| }}\right)  \cdot  {\lambda }^{\left| \mathcal{L} \smallsetminus  \mathcal{I}\right| } = {p}_{\mathcal{H},\mathbf{h}}^{\prime \prime }$ 台机器可以以负载 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 计算连接 $\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。

Combining the above with Appendix G for handling queries with unary relations, we have established:

将上述内容与附录G中处理一元关系查询的方法相结合，我们得到：

THEOREM 8.2. There exists an MPC algorithm that answers a query $\mathcal{Q}$ with load $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ ,where $n$ is the input size of $\mathcal{Q},p$ is the number of machines, $\alpha$ is given in (2),and $\phi$ is the generalized vertex-covering number of 2 .

定理8.2。存在一种MPC（大规模并行计算，Massively Parallel Computation）算法，该算法可以以负载 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 回答查询 $\mathcal{Q}$，其中 $n$ 是 $\mathcal{Q},p$ 的输入大小，$\alpha$ 是机器数量，$\alpha$ 在(2)中给出，$\phi$ 是2的广义顶点覆盖数。

When $\alpha  = 2$ ,the fractional edge covering number $\rho$ of2equals $\phi$ (Lemma 4.2); therefore,our algorithm achieves the optimal load $\widetilde{O}\left( {n/{p}^{1/\rho }}\right)$ .

当 $\alpha  = 2$ 时，2的分数边覆盖数 $\rho$ 等于 $\phi$（引理4.2）；因此，我们的算法实现了最优负载 $\widetilde{O}\left( {n/{p}^{1/\rho }}\right)$。

## 9 UNIFORM QUERIES

## 9 均匀查询

We can strengthen Theorem 8.2 when $\mathcal{Q}$ is $\alpha$ -uniform:

当 $\mathcal{Q}$ 是 $\alpha$ -均匀时，我们可以强化定理8.2：

THEOREM 9.1. There exists an MPC algorithm that answers an $\alpha$ -uniform query with $\alpha  \geq  2$ using load $\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$ ,where the meanings of $n,p,\alpha$ ,and $\phi$ are the same as in Theorem 8.2.

定理9.1. 存在一种多方计算（MPC）算法，该算法使用负载$\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$来回答一个$\alpha$ - 均匀查询，其中$n,p,\alpha$和$\phi$的含义与定理8.2中的相同。

We will prove the theorem by adapting the algorithm in the previous section. The first change is to set $\lambda$ higher:

我们将通过调整上一节中的算法来证明该定理。第一个改变是将$\lambda$设置得更高：

$$
\lambda  = {p}^{\frac{1}{{\alpha \phi } - \alpha  + 2}}. \tag{38}
$$

In Step 1, we set

在步骤1中，我们设置

$$
{p}_{\mathcal{H},\mathbf{h}}^{\prime } = p \cdot  \frac{{n}_{\mathcal{H},\mathbf{h}}}{\Theta \left( {n \cdot  {\lambda }^{k - \alpha }}\right) }
$$

Corollary 5.4 ensures that still at most $p$ machines are necessary. The load becomes:

推论5.4确保仍然最多需要$p$台机器。负载变为：

$$
O\left( \frac{{n}_{\mathcal{H},\mathbf{h}}}{{p}_{\mathcal{H},\mathbf{h}}^{\prime }}\right)  = O\left( \frac{n \cdot  {\lambda }^{k - \alpha }}{p}\right)  \tag{39}
$$

Proposition 9.2. (39) is bounded by $\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$ .

命题9.2. (39)式有界于$\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$。

Proof. With the $\lambda$ in (38),we only need to prove:

证明. 对于(38)式中的$\lambda$，我们只需要证明：

$$
\frac{k - \alpha  + 2}{{\alpha \phi } - \alpha  + 2} \leq  1
$$

which is true because of (35).

由于(35)式，这是成立的。

Step 2 requires no changes and entails a load of $O\left( {\frac{{n}_{\mathcal{H},h}}{{p}_{\mathcal{H},h}^{\prime }} + \frac{n}{p}}\right)$ , which is bounded by (39). Step 3 also proceeds in the same manner as in Section 8, but Lemma 8.1 can be improved to:

步骤2无需更改，并且产生的负载为$O\left( {\frac{{n}_{\mathcal{H},h}}{{p}_{\mathcal{H},h}^{\prime }} + \frac{n}{p}}\right)$，该负载有界于(39)式。步骤3也与第8节中的方式相同，但引理8.1可以改进为：

LEMMA 9.3. We can answer ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ in load $\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$ using ${p}_{\mathcal{H},h}^{\prime \prime }$ machines,where ${p}_{\mathcal{H},h}^{\prime \prime }$ is given in (36).

引理9.3. 我们可以使用${p}_{\mathcal{H},h}^{\prime \prime }$台机器，以负载$\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$回答${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$，其中${p}_{\mathcal{H},h}^{\prime \prime }$在(36)式中给出。

Proof. If $\left| \mathcal{I}\right|  > 0$ ,use ${p}_{\mathcal{H},h}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$ machines to compute ${CP}\left( {{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . The load is now

证明. 如果$\left| \mathcal{I}\right|  > 0$，使用${p}_{\mathcal{H},h}^{\prime \prime }/{\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$台机器来计算${CP}\left( {{\mathcal{Q}}_{I}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。现在的负载是

$$
\widetilde{O}\left( \frac{n \cdot  {\lambda }^{\frac{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) }{\left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{n \cdot  {p}^{\frac{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) }{\left( {{\alpha \phi } - \alpha  + 2}\right)  \cdot  \left| \mathcal{J}\right| }}}{{p}^{1/\left| \mathcal{J}\right| }}\right)  = \widetilde{O}\left( \frac{n}{{p}^{\frac{1}{\left| \mathcal{J}\right| }} - \frac{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) }{\left( {{\alpha \phi } - \alpha  + 2}\right)  \cdot  \left| \mathcal{J}\right| }}\right) 
$$

for some non-empty $\mathcal{J} \subseteq  I$ . To bound the above with $\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$ ,it suffices to show:

对于某个非空的$\mathcal{J} \subseteq  I$。为了用$\widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$来界定上述式子，只需证明：

$$
\frac{1}{\left| \mathcal{J}\right| } - \frac{\alpha \left( {\phi  - \left| \mathcal{J}\right| }\right) }{\left( {{\alpha \phi } - \alpha  + 2}\right)  \cdot  \left| \mathcal{J}\right| } \geq  \frac{2}{{\alpha \phi } - \alpha  + 2}
$$

$$
 \Leftrightarrow  {\alpha \phi } - \alpha  + 2 - \alpha \left( {\phi  - \left| \mathcal{J}\right| }\right)  \geq  2\left| \mathcal{J}\right| 
$$

$$
 \Leftrightarrow  \left( {\left| \mathcal{J}\right|  - 1}\right) \left( {\alpha  - 2}\right)  \geq  0
$$

which is true.

这是成立的。

The rest of the proof proceeds as in Lemma 8.1. Given ${\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$ machines, ${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ incurs load $\widetilde{O}\left( {n/{\lambda }^{2}}\right)  = \widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$ . Lemma 9.3 then follows from an application of Lemma 3.4.

证明的其余部分与引理8.1相同。给定${\lambda }^{\left| \mathcal{L} \smallsetminus  I\right| }$台机器，${\mathcal{Q}}_{\text{light }}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$产生的负载为$\widetilde{O}\left( {n/{\lambda }^{2}}\right)  = \widetilde{O}\left( {n/{p}^{2/\left( {{\alpha \phi } - \alpha  + 2}\right) }}\right)$。然后，通过应用引理3.4可以得到引理9.3。

The proof of Theorem 9.1 is now complete. By combining the theorem with Lemma 4.3, we have:

定理9.1的证明至此完成。将该定理与引理4.3相结合，我们得到：

COROLLARY 9.4. There exists an MPC algorithm that answers a symmetric query $\mathcal{Q}$ using load $\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$ ,where the meanings of $n,p,\alpha  \geq  2$ are the same as in Theorem 9.1,and $k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|$ .

推论9.4. 存在一种多方计算（MPC）算法，该算法使用负载$\widetilde{O}\left( {n/{p}^{2/\left( {k - \alpha  + 2}\right) }}\right)$来回答一个对称查询$\mathcal{Q}$，其中$n,p,\alpha  \geq  2$的含义与定理9.1中的相同，并且$k = \left| {\text{attset}\left( \mathcal{Q}\right) }\right|$。

## ACKNOWLEDGMENTS

## 致谢

Miao Qiao was patially supported by Marsden Fund UOA1732 from Royal Society of New Zealand, and the Catalyst: Strategic Fund NZ-Singapore Data Science Research Programme UOAX2001, from the Ministry of Business Innovation and Employment, New Zealand. Yufei Tao was partially supported by GRF grant 14207820 from HKRGC and a research grant from Alibaba Group.

缪乔部分得到了新西兰皇家学会的马斯登基金（Marsden Fund）UOA1732资助，以及新西兰商业、创新与就业部的催化剂战略基金（Catalyst: Strategic Fund）新-新数据科学研究计划UOAX2001资助。陶宇飞部分得到了香港研究资助局（HKRGC）的研资局研究基金（GRF）拨款14207820以及阿里巴巴集团的一项研究资助。

## REFERENCES

## 参考文献

[1] Azza Abouzeid, Kamil Bajda-Pawlikowski, Daniel J. Abadi, Alexander Rasin, and Avi Silberschatz. 2009. HadoopDB: An Architectural Hybrid of MapReduce

[1] 阿扎·阿布泽德（Azza Abouzeid）、卡米尔·巴伊达 - 帕夫利科夫斯基（Kamil Bajda - Pawlikowski）、丹尼尔·J·阿巴迪（Daniel J. Abadi）、亚历山大·拉辛（Alexander Rasin）和阿维·西尔伯沙茨（Avi Silberschatz）。2009年。HadoopDB：MapReduce的架构混合体

and DBMS Technologies for Analytical Workloads. Proceedings of the VLDB Endowment (PVLDB) 2, 1 (2009), 922-933.

[2] Foto N. Afrati, Manas R. Joglekar, Christopher Ré, Semih Salihoglu, and Jeffrey D. Ullman. 2017. GYM: A Multiround Distributed Join Algorithm. In Proceedings of International Conference on Database Theory (ICDT). 4:1-4:18.

[3] Foto N. Afrati and Jeffrey D. Ullman. 2011. Optimizing Multiway Joins in a Map-Reduce Environment. IEEE Transactions on Knowledge and Data Engineering (TKDE) 23, 9 (2011), 1282-1298.

[4] Albert Atserias, Martin Grohe, and Daniel Marx. 2013. Size Bounds and Query Plans for Relational Joins. SIAM Journal of Computing 42, 4 (2013), 1737-1767.

[5] Paul Beame, Paraschos Koutris, and Dan Suciu. 2014. Skew in parallel query processing. In Proceedings of ACM Symposium on Principles of Database Systems (PODS). 212-223.

[6] Paul Beame, Paraschos Koutris, and Dan Suciu. 2017. Communication Steps for Parallel Query Processing. Journal of the ACM (JACM) 64, 6 (2017), 40:1-40:58.

[7] Jeffrey Dean and Sanjay Ghemawat. 2004. MapReduce: Simplified Data Processing on Large Clusters. In Proceedings of USENIX Symposium on Operating Systems Design and Implementation (OSDI). 137-150.

[8] Xiao Hu. 2021. Cover or Pack: New Upper and Lower Bounds for Massively Parallel Joins. Accepted to appear in PODS (2021).

[9] Xiaocheng Hu, Miao Qiao, and Yufei Tao. 2016. I/O-efficient join dependency

testing, Loomis-Whitney join, and triangle enumeration. Journal of Computer and System Sciences (JCSS) 82, 8 (2016), 1300-1315.

[10] Xiao Hu and Ke Yi. 2019. Instance and Output Optimal Parallel Algorithms for Acyclic Joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS). 450-463.

[11] Xiao Hu, Ke Yi, and Yufei Tao. 2019. Output-Optimal Massively Parallel Algorithms for Similarity Joins. ACM Transactions on Database Systems (TODS) 44, 2 (2019), 6:1-6:36.

[12] Bas Ketsman and Dan Suciu. 2017. A Worst-Case Optimal Multi-Round Algorithm for Parallel Computation of Conjunctive Queries. In Proceedings of ACM Symposium on Principles of Database Systems (PODS). 417-428.

[13] Bas Ketsman, Dan Suciu, and Yufei Tao. 2020. A Near-Optimal Parallel Algorithm for Joining Binary Relations. CoRR abs/2011.14482 (2020).

[14] Paraschos Koutris, Paul Beame, and Dan Suciu. 2016. Worst-Case Optimal Algorithms for Parallel Query Processing. In Proceedings of International Conference on Database Theory (ICDT). 8:1-8:18.

[15] Paraschos Koutris and Dan Suciu. 2011. Parallel evaluation of conjunctive queries. In Proceedings of ACM Symposium on Principles of Database Systems (PODS). 223- 234.

[16] Hung Q. Ngo, Ely Porat, Christopher Re, and Atri Rudra. 2018. Worst-case Optimal Join Algorithms. Journal of the ACM (JACM) 65, 3 (2018), 16:1-16:40.

[17] Hung Q. Ngo, Christopher Re, and Atri Rudra. 2013. Skew strikes back: new developments in the theory of join algorithms. SIGMOD Rec. 42, 4 (2013), 5-16.

[18] Rasmus Pagh and Francesco Silvestri. 2014. The input/output complexity of triangle enumeration. In Proceedings of ACM Symposium on Principles of Database Systems (PODS). 224-233.

[19] Edward R. Scheinerman and Daniel H. Ullman. 1997. Fractional Graph Theory: A Rational Approach to the Theory of Graphs. Wiley, New York.

[20] Yufei Tao. 2020. A Simple Parallel Algorithm for Natural Joins on Binary Relations. In Proceedings of International Conference on Database Theory (ICDT). 25:1-25:18.

[21] Todd L. Veldhuizen. 2014. Triejoin: A Simple, Worst-Case Optimal Join Algorithm. In Proceedings of International Conference on Database Theory (ICDT). 96-106.

## APPENDIX

## 附录

## A PROOF OF LEMMA 3.5

## A 引理3.5的证明

Denote by actdom the set of values that appear in the relations of2. Let $\mathcal{R}$ be an arbitrary relation in2. Assume,without loss of generality,that scheme $\left( \mathcal{R}\right)  = \left\{  {{A}_{1},\ldots ,{A}_{r}}\right\}$ where $r = \operatorname{arity}\left( \mathcal{R}\right)$ . Define ${p}_{i}$ as the assigned share of ${A}_{i}$ ,for each $i \in  \left\lbrack  r\right\rbrack$ . Choose independent and perfectly random hash functions ${h}_{1},\ldots ,{h}_{r}$ such that ${h}_{i}$ maps actdom to $\left\lbrack  {p}_{i}\right\rbrack$ . Allocate each tuple $\mathbf{u} \in  \mathcal{R}$ to the bin $\left( {{h}_{1}\left( {a}_{1}\right) ,\ldots ,{h}_{r}\left( {a}_{r}\right) }\right)$ where $\left( {{a}_{1},\ldots ,{a}_{r}}\right)  = \left( {\mathbf{u}\left( {A}_{1}\right) ,\ldots ,\mathbf{u}\left( {A}_{r}\right) }\right)$ .

用actdom表示出现在关系集合2中的值的集合。设$\mathcal{R}$是集合2中的任意一个关系。不失一般性，假设模式为$\left( \mathcal{R}\right)  = \left\{  {{A}_{1},\ldots ,{A}_{r}}\right\}$，其中$r = \operatorname{arity}\left( \mathcal{R}\right)$。对于每个$i \in  \left\lbrack  r\right\rbrack$，定义${p}_{i}$为${A}_{i}$的分配份额。选择独立且完全随机的哈希函数${h}_{1},\ldots ,{h}_{r}$，使得${h}_{i}$将actdom映射到$\left\lbrack  {p}_{i}\right\rbrack$。将每个元组$\mathbf{u} \in  \mathcal{R}$分配到桶$\left( {{h}_{1}\left( {a}_{1}\right) ,\ldots ,{h}_{r}\left( {a}_{r}\right) }\right)$中，其中$\left( {{a}_{1},\ldots ,{a}_{r}}\right)  = \left( {\mathbf{u}\left( {A}_{1}\right) ,\ldots ,\mathbf{u}\left( {A}_{r}\right) }\right)$。

LEMMA A.1 (THEOREM 3.2 OF [6]). If $\mathcal{R}$ is skew free,the probability that every bin is allocated $\widetilde{O}\left( {n/\mathop{\prod }\limits_{{i = 1}}^{r}{p}_{i}}\right)$ tuples of $\mathcal{R}$ is at least $1 - 1/{p}^{c}$ where the constant $c$ can be arbitrarily large.

引理A.1（文献[6]的定理3.2）。如果$\mathcal{R}$是无倾斜的，那么每个桶分配到$\mathcal{R}$的$\widetilde{O}\left( {n/\mathop{\prod }\limits_{{i = 1}}^{r}{p}_{i}}\right)$个元组的概率至少为$1 - 1/{p}^{c}$，其中常数$c$可以任意大。

Now,consider $r \geq  2$ and that $\mathcal{R}$ is not skew free,but:

现在，考虑$r \geq  2$且$\mathcal{R}$是有倾斜的，但：

- $\mathcal{R}$ has at most $n/{p}_{1}$ tuples that agree on ${A}_{1}$ ;

- $\mathcal{R}$在${A}_{1}$上相同的元组最多有$n/{p}_{1}$个；

- $\mathcal{R}$ has at most $n/{p}_{2}$ tuples that agree on ${A}_{2}$ ;

- $\mathcal{R}$在${A}_{2}$上相同的元组最多有$n/{p}_{2}$个；

- $\mathcal{R}$ has at most $n/\left( {{p}_{1}{p}_{2}}\right)$ tuples that agree on ${A}_{1}$ and ${A}_{2}$ simultaneously.

- $\mathcal{R}$同时在${A}_{1}$和${A}_{2}$上相同的元组最多有$n/\left( {{p}_{1}{p}_{2}}\right)$个。

LEMMA A.2. Subject to the above conditions, the probability that every bin is allocated $\widetilde{O}\left( {n/\left( {{p}_{1}{p}_{2}}\right) }\right)$ tuples of $\mathcal{R}$ is at least $1 - 1/{p}^{c}$ where the constant $c$ can be arbitrarily large.

引理A.2。在上述条件下，每个桶分配到$\mathcal{R}$的$\widetilde{O}\left( {n/\left( {{p}_{1}{p}_{2}}\right) }\right)$个元组的概率至少为$1 - 1/{p}^{c}$，其中常数$c$可以任意大。

Proof. Define ${p}_{1}^{\prime } = {p}_{1},{p}_{2}^{\prime } = {p}_{2}$ ,and ${p}_{i}^{\prime } = 1$ for all $i \in  \left\lbrack  {3,r}\right\rbrack$ . Let us re-assign attribute ${A}_{i}$ a share of ${p}_{i}^{\prime }$ for each $i \in  \left\lbrack  r\right\rbrack$ . The stated conditions indicate that $\mathcal{R}$ is skew free under the shares ${p}_{1}^{\prime },\ldots ,{p}_{r}^{\prime }$ . Allocate each tuple $\left( {{a}_{1},\ldots ,{a}_{r}}\right)$ in $\mathcal{R}$ to the two-attribute $\operatorname{bin}\left( {{h}_{1}\left( {a}_{1}\right) ,{h}_{2}\left( {a}_{2}\right) }\right)$ . By Lemma A.1,with probability at least $1 - 1/{p}^{c}$ , each two-attribute bin is allocated $\widetilde{O}\left( {n/\left( {{p}_{1}{p}_{2}}\right) }\right)$ tuples.

证明。定义 ${p}_{1}^{\prime } = {p}_{1},{p}_{2}^{\prime } = {p}_{2}$，并且对于所有 $i \in  \left\lbrack  {3,r}\right\rbrack$ 定义 ${p}_{i}^{\prime } = 1$。对于每个 $i \in  \left\lbrack  r\right\rbrack$，我们为属性 ${A}_{i}$ 重新分配 ${p}_{i}^{\prime }$ 的份额。所述条件表明，在份额 ${p}_{1}^{\prime },\ldots ,{p}_{r}^{\prime }$ 下，$\mathcal{R}$ 是无偏斜的。将 $\mathcal{R}$ 中的每个元组 $\left( {{a}_{1},\ldots ,{a}_{r}}\right)$ 分配到二属性 $\operatorname{bin}\left( {{h}_{1}\left( {a}_{1}\right) ,{h}_{2}\left( {a}_{2}\right) }\right)$ 中。根据引理 A.1，至少以 $1 - 1/{p}^{c}$ 的概率，每个二属性桶分配到 $\widetilde{O}\left( {n/\left( {{p}_{1}{p}_{2}}\right) }\right)$ 个元组。

Lemma A. 2 then follows from the fact that each $\operatorname{bin}\left( {{x}_{1},{x}_{2},\ldots ,{x}_{r}}\right)$ is allocated a subset of the tuples allocated to the two-attribute bin $\left( {{x}_{1},{x}_{2}}\right)$ .

引理 A.2 可由以下事实推出：每个 $\operatorname{bin}\left( {{x}_{1},{x}_{2},\ldots ,{x}_{r}}\right)$ 分配到的元组是分配到二属性桶 $\left( {{x}_{1},{x}_{2}}\right)$ 的元组的一个子集。

Lemma A. 2 implies:

引理 A.2 意味着：

Corollary A.3. If $\mathcal{R}$ has arity $r \geq  2$ and is two-attribute free,the probability that every bin is allocated

推论 A.3。如果 $\mathcal{R}$ 的元数为 $r \geq  2$ 且是二属性无关的，那么每个桶分配到

$$
\widetilde{O}\left( {\mathop{\min }\limits_{\substack{{i,j \in  \left\lbrack  r\right\rbrack  } \\  {i \neq  j} }}\frac{n}{{p}_{i}{p}_{j}}}\right) 
$$

tuples of $\mathcal{R}$ is at least $1 - 1/{p}^{c}$ where the constant $c$ can be arbitrarily large.

$\mathcal{R}$ 的元组的概率至少为 $1 - 1/{p}^{c}$，其中常数 $c$ 可以任意大。

The BinHC algorithm answers $\mathcal{Q}$ as follows. For every $A \in$ attset(Q),it chooses an independent and perfectly random hash function ${h}_{A}$ that maps actdom to $\left\lbrack  {p}_{A}\right\rbrack$ . A bucket is a function $\mathbf{b}$ : attset $\left( \mathcal{Q}\right)  \rightarrow  \left\lbrack  p\right\rbrack$ subject to the constraint that $\mathbf{b}\left( A\right)  \in  \left\lbrack  {p}_{A}\right\rbrack$ for each $A \in$ attset(Q). Due to (5),the number of distinct buckets is at most $p$ . Assigning a machine to each possible bucket,BinHC solves $\mathcal{Q}$ in two steps:

BinHC 算法按如下方式处理 $\mathcal{Q}$。对于每个 $A \in$ 属于 attset(Q)，它选择一个独立且完全随机的哈希函数 ${h}_{A}$，该函数将 actdom 映射到 $\left\lbrack  {p}_{A}\right\rbrack$。一个桶是一个函数 $\mathbf{b}$：attset $\left( \mathcal{Q}\right)  \rightarrow  \left\lbrack  p\right\rbrack$，需满足对于每个 $A \in$ 属于 attset(Q) 有 $\mathbf{b}\left( A\right)  \in  \left\lbrack  {p}_{A}\right\rbrack$ 这一约束条件。由于 (5)，不同桶的数量至多为 $p$。为每个可能的桶分配一台机器，BinHC 分两步解决 $\mathcal{Q}$：

(1) For every relation $\mathcal{R} \in  \mathcal{Q}$ ,send each tuple $\mathbf{u} \in  \mathcal{R}$ to every machine responsible for a bucket $\mathbf{b}$ satisfying the condition that $\mathbf{b}\left( A\right)  = {h}_{A}\left( {\mathbf{u}\left( A\right) }\right)$ for all $A \in$ scheme(R).

(1) 对于每个关系 $\mathcal{R} \in  \mathcal{Q}$，将每个元组 $\mathbf{u} \in  \mathcal{R}$ 发送到负责满足对于所有 $A \in$ 属于 scheme(R) 有 $\mathbf{b}\left( A\right)  = {h}_{A}\left( {\mathbf{u}\left( A\right) }\right)$ 这一条件的桶 $\mathbf{b}$ 的每台机器。

(2) Each machine generates the maximum subset of $\mathcal{J}$ oin(Q) that can be produced from the data received.

(2) 每台机器从接收到的数据中生成 $\mathcal{J}$ oin(Q) 的最大子集。

By Corollary A. 3 (for non-unary relations) and Lemma A.1 (for unary relations), the load is at most (8).

根据推论 A.3（针对非一元关系）和引理 A.1（针对一元关系），负载至多为 (8)。

## B PROOF OF LEMMA 5.2

## B 引理 5.2 的证明

It is obvious that the right hand side of (13) is a subset of the left hand side. Next,we will prove the opposite: any tuple $\mathbf{u} \in  \mathcal{J}$ oin(Q) must be produced by the right hand side.

显然，(13) 的右侧是左侧的一个子集。接下来，我们将证明相反的情况：任何元组 $\mathbf{u} \in  \mathcal{J}$ 属于 oin(Q) 必定由右侧生成。

Given $\mathbf{u}$ ,we construct a plan $\mathbf{P}$ and its corresponding $\mathcal{H}$ as follows:

给定 $\mathbf{u}$，我们按如下方式构造一个计划 $\mathbf{P}$ 及其对应的 $\mathcal{H}$：

${S}_{1} = \varnothing ,{S}_{2} = \varnothing$

$S =$ attset(Q)

$S =$ 属于 attset(Q)

while $\exists X \in  S$ such that $\mathbf{u}\left( X\right)$ is heavy do

当 $\exists X \in  S$ 满足 $\mathbf{u}\left( X\right)$ 为重度（heavy）时执行

add $X$ to ${S}_{1}$ ,and remove $X$ from $S$

将 $X$ 添加到 ${S}_{1}$ 中，并从 $S$ 中移除 $X$

while $\exists$ distinct $Y,Z \in  S$ s.t. $\left( {\mathbf{u}\left( Y\right) ,\mathbf{u}\left( Z\right) }\right)$ is heavy do

当 $\exists$ 与 $Y,Z \in  S$ 不同且满足 $\left( {\mathbf{u}\left( Y\right) ,\mathbf{u}\left( Z\right) }\right)$ 为重度（heavy）时执行

add(Y,Z)to ${S}_{2}$ (assuming $Y \prec  Z$ ),and remove $Y,Z$ from $S$ return $\mathbf{P} = \left( {{S}_{1},{S}_{2}}\right)$ and $\mathcal{H} =$ scheme $\left( \mathcal{Q}\right)  \smallsetminus  S$

将(Y,Z)添加到 ${S}_{2}$ 中（假设 $Y \prec  Z$ ），并从 $S$ 中移除 $Y,Z$ 返回 $\mathbf{P} = \left( {{S}_{1},{S}_{2}}\right)$ 和 $\mathcal{H} =$ 方案 $\left( \mathcal{Q}\right)  \smallsetminus  S$

Set $\mathbf{h} = \mathbf{u}\left\lbrack  \mathcal{H}\right\rbrack$ . We will show that $\mathbf{u}\left\lbrack  {\text{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}}\right\rbrack   \in$ Join $\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ ,which will complete the proof.

设 $\mathbf{h} = \mathbf{u}\left\lbrack  \mathcal{H}\right\rbrack$ 。我们将证明 $\mathbf{u}\left\lbrack  {\text{attset}\left( \mathcal{Q}\right)  \smallsetminus  \mathcal{H}}\right\rbrack   \in$ 与 $\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ 连接，这将完成证明。

Consider an arbitrary edge $e \in  \mathcal{E}$ active on $\mathbf{P}$ . As $\mathbf{u} \in  \operatorname{Join}\left( \mathcal{Q}\right)$ , we know $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}$ . As before,set ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ . To prove $\mathbf{u}\lbrack$ attset $\left( \mathcal{Q}\right)  \smallsetminus$ $\mathcal{H}\rbrack  \in  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ ,it suffices to show that $\mathbf{u}\left\lbrack  {e}^{\prime }\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . In turn,to prove $\mathbf{u}\left\lbrack  {e}^{\prime }\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ,we must establish two facts:

考虑在 $\mathbf{P}$ 上活跃的任意边 $e \in  \mathcal{E}$ 。由于 $\mathbf{u} \in  \operatorname{Join}\left( \mathcal{Q}\right)$ ，我们知道 $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}$ 。如前所述，设 ${e}^{\prime } = e \smallsetminus  \mathcal{H}$ 。为了证明 $\mathbf{u}\lbrack$ 附着集 $\left( \mathcal{Q}\right)  \smallsetminus$ $\mathcal{H}\rbrack  \in  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ ，只需证明 $\mathbf{u}\left\lbrack  {e}^{\prime }\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 。反过来，为了证明 $\mathbf{u}\left\lbrack  {e}^{\prime }\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ，我们必须确立两个事实：

- for any attribute $A \in  {e}^{\prime },\mathbf{u}\left( A\right)$ is light;

- 对于任何属性 $A \in  {e}^{\prime },\mathbf{u}\left( A\right)$ 为轻度（light）；

- for any distinct attributes $A,B \in  {e}^{\prime },\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ is light.

- 对于任何不同的属性 $A,B \in  {e}^{\prime },\left( {\mathbf{u}\left( A\right) ,\mathbf{u}\left( B\right) }\right)$ 为轻度（light）。

The first bullet holds because otherwise $A$ would have been added to ${S}_{1}$ at Line 4 . The second bullet also holds because otherwise (A,B)(assuming $A \prec  B$ ) would have been added to ${S}_{2}$ at Line 6 .

第一个要点成立，因为否则 $A$ 会在第4行被添加到 ${S}_{1}$ 中。第二个要点也成立，因为否则 (A,B)（假设 $A \prec  B$ ）会在第6行被添加到 ${S}_{2}$ 中。

## C PROOF OF PROPOSITION 6.1

## C 命题6.1的证明

Let $\mathbf{u}$ be a tuple output by ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . We will prove $\mathbf{u} \in$ Join $\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . Consider an arbitrary orphaned vertex $A$ . For each orphaning edge $e$ of $A$ ,we must have $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ; therefore, $\mathbf{u}\left( A\right)  \in  {\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . This further implies that,for every edge $e$ of $\mathcal{G}$ such that $e \smallsetminus  \mathcal{H}$ is non-unary, $\mathbf{u}\left\lbrack  {e \smallsetminus  \mathcal{H}}\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . It thus follows that $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ .

设 $\mathbf{u}$ 是由 ${\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 输出的一个元组。我们将证明 $\mathbf{u} \in$ 连接 $\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。考虑任意一个孤立顶点 $A$。对于 $A$ 的每条孤立边 $e$，我们必定有 $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$；因此，$\mathbf{u}\left( A\right)  \in  {\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$。这进一步意味着，对于 $\mathcal{G}$ 的每条满足 $e \smallsetminus  \mathcal{H}$ 是非一元的边 $e$，有 $\mathbf{u}\left\lbrack  {e \smallsetminus  \mathcal{H}}\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$。由此可得 $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。

Conversely,let $\mathbf{u}$ be a tuple output by ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . We will prove $\mathbf{u} \in$ Join $\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ . This means that,for every orphaned vertex $A$ ,we must have $\mathbf{u}\left( A\right)  \in  {\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . Thus,for each orphaning edge $e$ of $A$ ,it holds that $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . Consider an arbitrary edge $e$ of $\mathcal{G}$ such that $e \smallsetminus  \mathcal{H}$ is non-unary. Clearly, $\mathbf{u}\left\lbrack  {e \smallsetminus  \mathcal{H}}\right\rbrack$ appears in ${\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ ,which ensures $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ . It thus follows that $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$ .

反之，设 $\mathbf{u}$ 是由 ${\mathcal{Q}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 输出的一个元组。我们将证明 $\mathbf{u} \in$ 连接 $\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。这意味着，对于每个孤立顶点 $A$，我们必定有 $\mathbf{u}\left( A\right)  \in  {\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$。因此，对于 $A$ 的每条孤立边 $e$，有 $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 成立。考虑 $\mathcal{G}$ 的任意一条满足 $e \smallsetminus  \mathcal{H}$ 是非一元的边 $e$。显然，$\mathbf{u}\left\lbrack  {e \smallsetminus  \mathcal{H}}\right\rbrack$ 出现在 ${\mathcal{R}}_{e}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ 中，这保证了 $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$。由此可得 $\mathbf{u} \in  \mathcal{J}\operatorname{oin}\left( {{\mathcal{Q}}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)$。

## D PROOF OF LEMMA 7.3

## D 引理 7.3 的证明

Consider an edge $e \in  \mathcal{E}$ . As $\left| e\right|  \leq  \alpha$ ,we have

考虑一条边 $e \in  \mathcal{E}$。由于 $\left| e\right|  \leq  \alpha$，我们有

$$
1 \geq  \frac{\left| e\right|  - 1}{\left| e\right| }\frac{\alpha }{\alpha  - 1}. \tag{40}
$$

As $\left\{  {{x}_{e} \mid  e \in  \mathcal{E}}\right\}$ is a feasible assignment for the characterizing program, $\mathop{\sum }\limits_{{e \in  \mathcal{E} : A \in  e}}{x}_{e} \leq  1$ holds for every $A \in  \mathcal{V}$ . Hence:

由于 $\left\{  {{x}_{e} \mid  e \in  \mathcal{E}}\right\}$ 是特征化程序的一个可行赋值，所以对于每个 $A \in  \mathcal{V}$，$\mathop{\sum }\limits_{{e \in  \mathcal{E} : A \in  e}}{x}_{e} \leq  1$ 都成立。因此：

$$
k - \left| \mathcal{J}\right|  = \mathop{\sum }\limits_{{A \notin  \mathcal{J}}}1
$$

$$
 \geq  \mathop{\sum }\limits_{{A \notin  \mathcal{J}}}\mathop{\sum }\limits_{{e \in  \mathcal{E} : A \in  e}}{x}_{e} = \mathop{\sum }\limits_{{e \in  \mathcal{E}}}\left| {e \smallsetminus  \mathcal{J}}\right|  \cdot  {x}_{e}
$$

$$
 = \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} = \varnothing }}\left| e\right|  \cdot  {x}_{e} + \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} \neq  \varnothing }}\left( {\left| e\right|  - 1}\right)  \cdot  {x}_{e}
$$

where the last equality used Property (1) of Lemma 7.2. With the above, we can derive

其中最后一个等式使用了引理 7.2 的性质 (1)。由此，我们可以推导出

$$
k - \left| \mathcal{J}\right|  + \frac{1}{\alpha  - 1}\mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} \neq  \varnothing }}\left( {\left| e\right|  - 1}\right)  \cdot  {x}_{e}
$$

$$
 \geq  \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} = \varnothing }}{x}_{e}\left| e\right|  + \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} \neq  \varnothing }}\left( {\left| e\right|  - 1}\right) {x}_{e}\left( {1 + \frac{1}{\alpha  - 1}}\right) 
$$

$$
 \geq  \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} = \varnothing }}{x}_{e}\left| e\right|  \cdot  \frac{\left| e\right|  - 1}{\left| e\right| }\frac{\alpha }{\alpha  - 1} + \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} \neq  \varnothing }}{x}_{e}\left( {\left| e\right|  - 1}\right) \frac{\alpha }{\alpha  - 1}
$$

$$
\text{(applied (40))}
$$

$$
 = \mathop{\sum }\limits_{{e \in  \mathcal{E}}}{x}_{e}\left( {\left| e\right|  - 1}\right)  \cdot  \frac{\alpha }{\alpha  - 1} = \bar{\phi } \cdot  \frac{\alpha }{\alpha  - 1}.
$$

Multiplying both sides by $\alpha  - 1$ ,we have:

两边同时乘以 $\alpha  - 1$，我们得到：

$$
\left( {\alpha  - 1}\right) \left( {k - \left| \mathcal{J}\right| }\right)  + \mathop{\sum }\limits_{{e \in  \mathcal{E} : e \cap  \mathcal{J} \neq  \varnothing }}\left( {\left| e\right|  - 1}\right)  \cdot  {x}_{e} \geq  \bar{\phi }\alpha 
$$

$$
 = \left( {k - \phi }\right) \alpha 
$$

where the last equality used Lemma 4.1. Therefore:

其中最后一个等式使用了引理 4.1。因此：

$$
{k\alpha } - k - \left| \mathcal{J}\right| \alpha  + \left| \mathcal{J}\right|  + \mathop{\sum }\limits_{{e \in  {\mathcal{E}}^{ * }}}\left( {\left| e\right|  - 1}\right) {x}_{e} \geq  {k\alpha } - {\phi \alpha }.
$$

Re-arranging the terms proves the lemma.

重新排列各项即可证明该引理。

## E PROOF OF PROPOSITION 7.5

## E 命题 7.5 的证明

Clearly:

显然：

$$
\mathop{\sum }\limits_{\substack{\text{ full config. } \\  {\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }\mathbf{P}} }}\left| {{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right) }\right|  = \mathop{\sum }\limits_{\substack{\text{ full config. } \\  {\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }\mathbf{P}} }}\left| {{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right) \times \{ \mathbf{h}\} }\right| .
$$

Thus, we only need to prove:

因此，我们只需要证明：

$$
{\bigcup }_{\begin{matrix} \text{ full config. } \\  {\left( {\mathcal{H},\mathbf{h}}\right) \text{ of }\mathbf{P}} \end{matrix}}\operatorname{CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \{ \mathbf{h}\}  \subseteq  \operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \operatorname{Join}\left( {\mathcal{Q}}^{ * }\right) . \tag{41}
$$

Fix any(H,h)of $\mathbf{P}$ . Let $\mathbf{u}$ be an arbitrary tuple in ${CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \{ \mathbf{h}\}$ . We will show that $\mathbf{u} \in  {CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ Join $\left( {\mathcal{Q}}^{ * }\right)$ ,which will complete the proof.

固定$\mathbf{P}$中的任意(H, h)。设$\mathbf{u}$是${CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \{ \mathbf{h}\}$中的任意元组。我们将证明$\mathbf{u} \in  {CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ 连接 $\left( {\mathcal{Q}}^{ * }\right)$，这将完成证明。

Consider attribute ${X}_{i}$ for any $i \in  \left\lbrack  a\right\rbrack$ . By definition of $\mathbf{h},\mathbf{u}\left( {X}_{i}\right)  =$ $\mathbf{h}\left( {X}_{i}\right)$ must be heavy. Hence, $\mathbf{u}\left( {X}_{i}\right)  \in  {S}_{i}$ . Likewise,consider attributes ${Y}_{j}$ and ${Z}_{j}$ for any $j \in  \left\lbrack  b\right\rbrack$ . By definition of $\mathbf{h},\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$ must be heavy while both $\mathbf{h}\left( {Y}_{j}\right)$ and $\mathbf{h}\left( {Z}_{j}\right)$ ) must be light. Therefore, $\left( {\mathbf{u}\left( {Y}_{j}\right) ,\mathbf{u}\left( {Z}_{j}\right) }\right)  = \left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)  \in  {\mathcal{D}}_{j}.$

考虑任意$i \in  \left\lbrack  a\right\rbrack$的属性${X}_{i}$。根据$\mathbf{h},\mathbf{u}\left( {X}_{i}\right)  =$的定义，$\mathbf{h}\left( {X}_{i}\right)$必须是重的。因此，$\mathbf{u}\left( {X}_{i}\right)  \in  {S}_{i}$。同样地，考虑任意$j \in  \left\lbrack  b\right\rbrack$的属性${Y}_{j}$和${Z}_{j}$。根据$\mathbf{h},\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$的定义，$\mathbf{h},\left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)$必须是重的，而$\mathbf{h}\left( {Y}_{j}\right)$和$\mathbf{h}\left( {Z}_{j}\right)$都必须是轻的。因此，$\left( {\mathbf{u}\left( {Y}_{j}\right) ,\mathbf{u}\left( {Z}_{j}\right) }\right)  = \left( {\mathbf{h}\left( {Y}_{j}\right) ,\mathbf{h}\left( {Z}_{j}\right) }\right)  \in  {\mathcal{D}}_{j}.$

Consider an arbitrary edge $e \in  {\mathcal{E}}^{ * }$ . Let $A$ be the (only) isolated vertex in $e$ (Property (1) of Lemma 7.2). Note that $e \smallsetminus  \{ A\}  \subseteq  \mathcal{H}$ (Property (2) of Lemma 7.2),and that $e$ is an orphaning edge of $A$ (Section 6). The fact $\mathbf{u} \in  {CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \{ \mathbf{h}\}$ tells us $\mathbf{u}\left( A\right)  \in  {\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$ (Section 6). By (15),this indicates $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$ (Section 5),and hence, $\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}$ .

考虑任意一条边$e \in  {\mathcal{E}}^{ * }$。设$A$是$e$中（唯一的）孤立顶点（引理7.2的性质(1)）。注意到$e \smallsetminus  \{ A\}  \subseteq  \mathcal{H}$（引理7.2的性质(2)），并且$e$是$A$的一条孤立边（第6节）。事实$\mathbf{u} \in  {CP}\left( {{\mathcal{Q}}_{\mathcal{J}}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right) }\right)  \times  \{ \mathbf{h}\}$告诉我们$\mathbf{u}\left( A\right)  \in  {\mathcal{R}}_{A}^{\prime \prime }\left( {\mathcal{H},\mathbf{h}}\right)$（第6节）。根据(15)，这表明$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}^{\prime }\left( {\mathcal{H},\mathbf{h}}\right)$（第5节），因此，$\mathbf{u}\left\lbrack  e\right\rbrack   \in  {\mathcal{R}}_{e}$。

We have shown that $\mathbf{u}\left\lbrack  {\text{scheme}\left( \mathcal{R}\right) }\right\rbrack   \in  \mathcal{R}$ for every relation $\mathcal{R}$ involved on the right hand side of (41). It thus follows that $\mathbf{u} \in$ $\operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ Foin $\left( {\mathcal{Q}}^{ * }\right)$ .

我们已经证明了对于(41)右侧涉及的每个关系$\mathcal{R}$，都有$\mathbf{u}\left\lbrack  {\text{scheme}\left( \mathcal{R}\right) }\right\rbrack   \in  \mathcal{R}$。因此，可得$\mathbf{u} \in$ $\operatorname{CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ 连接 $\left( {\mathcal{Q}}^{ * }\right)$。

## F PROOF OF PROPOSITION 7.10

## F 命题7.10的证明

It is obvious that $\oint \operatorname{oin}\left( {\mathcal{Q}}_{\text{final }}\right)  \subseteq  {CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \oint \operatorname{oin}\left( {\mathcal{Q}}_{\ell }\right)$ . Next, we will prove that the opposite is also true. Let $\mathbf{u}$ be a tuple in ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ foin $\left( {\mathcal{Q}}_{\ell }\right)$ . For each $A \in  \mathcal{I},\mathbf{u}\left( A\right)$ must appear in some input relation of $\mathcal{Q}$ ,which means $\mathbf{u}\left( A\right)  \in  {\mathcal{U}}_{A}$ . It thus follows that $\mathbf{u}\left( A\right)  \in  \mathcal{J}$ oin $\left( {\mathcal{Q}}_{\text{final }}\right)$ .

显然，$\oint \operatorname{oin}\left( {\mathcal{Q}}_{\text{final }}\right)  \subseteq  {CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie  \oint \operatorname{oin}\left( {\mathcal{Q}}_{\ell }\right)$ 。接下来，我们将证明相反的情况同样成立。设 $\mathbf{u}$ 是 ${CP}\left( {\mathcal{Q}}_{\text{heavy }}\right)  \bowtie$ 中属于 $\left( {\mathcal{Q}}_{\ell }\right)$ 的一个元组。对于每个 $A \in  \mathcal{I},\mathbf{u}\left( A\right)$ ，它必定出现在 $\mathcal{Q}$ 的某个输入关系中，这意味着 $\mathbf{u}\left( A\right)  \in  {\mathcal{U}}_{A}$ 。因此，$\mathbf{u}\left( A\right)  \in  \mathcal{J}$ 属于 $\left( {\mathcal{Q}}_{\text{final }}\right)$ 。

## G QUERIES WITH UNARY RELATIONS

## 具有一元关系的G查询

A unary relation $\mathcal{R} \in  \mathcal{Q}$ can be of two types:

一元关系 $\mathcal{R} \in  \mathcal{Q}$ 可以分为两种类型：

- non-isolated: $\mathcal{Q}$ contains another relation ${\mathcal{R}}^{\prime }$ such that scheme(R) $\subseteq$ scheme(R’);

- 非孤立的：$\mathcal{Q}$ 包含另一个关系 ${\mathcal{R}}^{\prime }$ ，使得模式(R) $\subseteq$ 模式(R’)；

- isolated: no such ${\mathcal{R}}^{\prime }$ exists.

- 孤立的：不存在这样的 ${\mathcal{R}}^{\prime }$ 。

As shown in $\left\lbrack  {{11},{14}}\right\rbrack$ ,all the non-isolated unary relations can be eliminated with load $\widetilde{O}\left( {n/p}\right)$ . Henceforth,we will assume that all unary relations are isolated.

如 $\left\lbrack  {{11},{14}}\right\rbrack$ 所示，所有非孤立的一元关系都可以通过负载 $\widetilde{O}\left( {n/p}\right)$ 消除。此后，我们将假设所有一元关系都是孤立的。

Let $g$ be the number of (isolated) unary relations which are denoted as ${\mathcal{R}}_{1},\ldots ,{\mathcal{R}}_{g}$ ,respectively. Define ${\mathcal{Q}}^{\prime } = \mathcal{Q} \smallsetminus  \left\{  {{\mathcal{R}}_{1},\ldots ,{\mathcal{R}}_{g}}\right\}$ ; ${\mathcal{Q}}^{\prime }$ is a query without isolated relations. We have:

设 $g$ 分别表示（孤立的）一元关系的数量，这些关系分别记为 ${\mathcal{R}}_{1},\ldots ,{\mathcal{R}}_{g}$ 。定义 ${\mathcal{Q}}^{\prime } = \mathcal{Q} \smallsetminus  \left\{  {{\mathcal{R}}_{1},\ldots ,{\mathcal{R}}_{g}}\right\}$ ；${\mathcal{Q}}^{\prime }$ 是一个没有孤立关系的查询。我们有：

$$
\operatorname{Join}\left( \mathcal{Q}\right)  = \operatorname{Join}\left( {\mathcal{Q}}^{\prime }\right)  \times  \left( {{\mathcal{R}}_{1} \times  \ldots  \times  {\mathcal{R}}_{g}}\right) .
$$

Let $\phi$ and ${\phi }^{\prime }$ be the generalized vertex packing numbers of $\mathcal{Q}$ and ${\mathcal{Q}}^{\prime }$ ,respectively. It is easy to verify by definition that

设 $\phi$ 和 ${\phi }^{\prime }$ 分别是 $\mathcal{Q}$ 和 ${\mathcal{Q}}^{\prime }$ 的广义顶点覆盖数。根据定义很容易验证

$$
\phi  = {\phi }^{\prime } + g.
$$

Given ${p}_{1}$ machines,our algorithm in Section 8 computes Join $\left( {\mathcal{Q}}^{\prime }\right)$ with load $\widetilde{O}\left( {n/{p}_{1}^{2/\left( {\alpha {\phi }^{\prime }}\right) }}\right)$ . By Lemma 3.3, ${\mathcal{R}}_{1} \times  \ldots  \times  {\mathcal{R}}_{g}$ can be computed with load $O\left( {n/{p}_{2}^{1/g}}\right)$ using ${p}_{2}$ machines. Setting

给定 ${p}_{1}$ 台机器，我们在第8节中的算法以负载 $\widetilde{O}\left( {n/{p}_{1}^{2/\left( {\alpha {\phi }^{\prime }}\right) }}\right)$ 计算连接 $\left( {\mathcal{Q}}^{\prime }\right)$ 。根据引理3.3，使用 ${p}_{2}$ 台机器可以以负载 $O\left( {n/{p}_{2}^{1/g}}\right)$ 计算 ${\mathcal{R}}_{1} \times  \ldots  \times  {\mathcal{R}}_{g}$ 。设置

$$
{p}_{1} = {p}^{{\phi }^{\prime }/\phi }
$$

$$
{p}_{2} = {p}^{g/\phi }
$$

we can apply Lemma 3.4 to obtain $\mathcal{J}$ oin(Q)with load

我们可以应用引理3.4，以负载

$$
\widetilde{O}\left( \frac{n}{\min \left\{  {{p}^{\frac{2}{\alpha \phi }},{p}^{\frac{1}{\phi }}}\right\}  }\right) 
$$

using ${p}_{1}{p}_{2} = p$ machines. The above is bounded by $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ because $\alpha  \geq  2$ .

使用 ${p}_{1}{p}_{2} = p$ 台机器得到 ${p}_{1}{p}_{2} = p$ 属于(Q)。由于 $\alpha  \geq  2$ ，上述负载受限于 $\widetilde{O}\left( {n/{p}^{2/\left( {\alpha \phi }\right) }}\right)$ 。

## H EDGE QUASI-PACKING NUMBER

## H 边拟覆盖数

Consider a hypergraph $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ without exposed vertices (Section 3.1). Given a subset $\mathcal{U}$ of $\mathcal{V}$ ,denote by ${\mathcal{G}}_{ - }\mathcal{U}$ the graph obtained by removing $\mathcal{U}$ from $\mathcal{G}$ ,or formally: ${\mathcal{G}}_{-\mathcal{U}} = \left( {\mathcal{V} \smallsetminus  \mathcal{U},{\mathcal{E}}_{-\mathcal{U}}}\right)$ where ${\mathcal{E}}_{ - }\mathcal{U} = \{ e \smallsetminus  \mathcal{U} \mid  e \in  \mathcal{E}$ and $e \smallsetminus  \mathcal{U} \neq  \varnothing \}$ . The edge quasi-packing number $\psi \left( \mathcal{G}\right)$ of $\mathcal{G}$ equals

考虑一个没有暴露顶点的超图 $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$（3.1 节）。给定 $\mathcal{V}$ 的一个子集 $\mathcal{U}$，用 ${\mathcal{G}}_{ - }\mathcal{U}$ 表示从 $\mathcal{G}$ 中移除 $\mathcal{U}$ 后得到的图，或者形式上表示为：${\mathcal{G}}_{-\mathcal{U}} = \left( {\mathcal{V} \smallsetminus  \mathcal{U},{\mathcal{E}}_{-\mathcal{U}}}\right)$，其中 ${\mathcal{E}}_{ - }\mathcal{U} = \{ e \smallsetminus  \mathcal{U} \mid  e \in  \mathcal{E}$ 且 $e \smallsetminus  \mathcal{U} \neq  \varnothing \}$。$\mathcal{G}$ 的边准填充数 $\psi \left( \mathcal{G}\right)$ 等于

$$
\mathop{\max }\limits_{{\mathcal{U} \subseteq  \mathcal{V}}}\tau \left( {{\mathcal{G}}_{ - }\mathcal{U}}\right) 
$$

where $\tau \left( {\mathcal{G}}_{-\mathcal{U}}\right)$ is the fractional edge-packing number of ${\mathcal{G}}_{-\mathcal{U}}$ (Section 3.1).

其中 $\tau \left( {\mathcal{G}}_{-\mathcal{U}}\right)$ 是 ${\mathcal{G}}_{-\mathcal{U}}$ 的分数边填充数（3.1 节）。

Example. Let $\mathcal{G}$ be the graph in Figure 1(a). Consider a set $\mathcal{U}$ which includes all attributes except $\mathrm{D},\mathrm{G}$ ,and $\mathrm{H}.{\mathcal{G}}_{\mathcal{U}}$ contains 8 unary edges: $\{ A\} ,\{ B\} ,\{ C\} ,\{ E\} ,\{ F\} ,\{ I\} ,\{ J\}$ ,and $\{ K\}$ . Consider the fractional edge packing $\mathcal{W}$ that maps these edges to 1,and the other edges of ${\mathcal{G}}_{\mathcal{U}}$ to 0 . W has a weight of 8 . We can therefore conclude that $\psi \left( \mathcal{G}\right)  \geq  8$ .

示例。设 $\mathcal{G}$ 为图 1(a) 中的图。考虑一个集合 $\mathcal{U}$，它包含除 $\mathrm{D},\mathrm{G}$ 之外的所有属性，并且 $\mathrm{H}.{\mathcal{G}}_{\mathcal{U}}$ 包含 8 条一元边：$\{ A\} ,\{ B\} ,\{ C\} ,\{ E\} ,\{ F\} ,\{ I\} ,\{ J\}$ 以及 $\{ K\}$。考虑分数边填充 $\mathcal{W}$，它将这些边映射为 1，而将 ${\mathcal{G}}_{\mathcal{U}}$ 的其他边映射为 0。W 的权重为 8。因此，我们可以得出结论 $\psi \left( \mathcal{G}\right)  \geq  8$。

We now echo the claim in Section 1.3 that if $\mathcal{Q}$ is a $k$ -choose- $\alpha$ join,then $\psi \left( \mathcal{Q}\right)  \geq  k - \alpha  + 1$ . Let ${A}_{1},\ldots ,{A}_{k}$ be the attributes in attset(Q),and $\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$ be the hypergraph of $\mathcal{Q}$ . Consider a set $\mathcal{U} = \left\{  {{A}_{1},\ldots ,{A}_{k - \alpha  + 1}}\right\}$ . For each $i \in  \left\lbrack  {k - \alpha  + 1}\right\rbrack  ,{\mathcal{G}}_{\mathcal{U}}$ contains a unary edge $\left\{  {A}_{i}\right\}$ (shrunk from the edge $\left\{  {{A}_{i},{A}_{k - \alpha  + 2},{A}_{k - \alpha  + 3},\ldots ,{A}_{k}}\right\}$ in $\mathcal{G}$ ). Thus, ${\mathcal{G}}_{\mathcal{U}}$ admits a fractional edge packing $\mathcal{W}$ that maps only the $k - \alpha  + 1$ edges $\left\{  {A}_{1}\right\}  ,\left\{  {A}_{2}\right\}  ,\ldots ,\left\{  {A}_{k - \alpha  + 1}\right\}$ to 1,and the other edges to 0 . It thus follows that $\psi \left( \mathcal{Q}\right)  \geq  \tau \left( {\mathcal{G}}_{\mathcal{U}}\right)  = k - \alpha  + 1$ .

我们现在呼应1.3节中的论断，即如果$\mathcal{Q}$是一个$k$选$\alpha$连接，那么$\psi \left( \mathcal{Q}\right)  \geq  k - \alpha  + 1$。设${A}_{1},\ldots ,{A}_{k}$为attset(Q)中的属性，$\mathcal{G} = \left( {\mathcal{V},\mathcal{E}}\right)$为$\mathcal{Q}$的超图。考虑一个集合$\mathcal{U} = \left\{  {{A}_{1},\ldots ,{A}_{k - \alpha  + 1}}\right\}$。对于每个$i \in  \left\lbrack  {k - \alpha  + 1}\right\rbrack  ,{\mathcal{G}}_{\mathcal{U}}$都包含一条一元边$\left\{  {A}_{i}\right\}$（由$\mathcal{G}$中的边$\left\{  {{A}_{i},{A}_{k - \alpha  + 2},{A}_{k - \alpha  + 3},\ldots ,{A}_{k}}\right\}$收缩而来）。因此，${\mathcal{G}}_{\mathcal{U}}$存在一个分数边填充$\mathcal{W}$，它仅将$k - \alpha  + 1$条边$\left\{  {A}_{1}\right\}  ,\left\{  {A}_{2}\right\}  ,\ldots ,\left\{  {A}_{k - \alpha  + 1}\right\}$映射为1，而将其他边映射为0。由此可得$\psi \left( \mathcal{Q}\right)  \geq  \tau \left( {\mathcal{G}}_{\mathcal{U}}\right)  = k - \alpha  + 1$。