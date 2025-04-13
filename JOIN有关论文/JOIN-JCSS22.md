# Intersection Joins under Updates

# 更新情况下的交集连接

Yufei Tao

陶宇飞

CUHK

taoyf@cse.cuhk.edu.hk

$\mathrm{{Ke}}\mathrm{{Yi}}$

HKUST

yike@cse.ust.hk

September 28, 2021

2021年9月28日

## Abstract

## 摘要

In an intersection join,we are given $t$ sets ${R}_{1},\ldots ,{R}_{t}$ of axis-parallel rectangles in ${\mathbb{R}}^{d}$ ,where $d \geq  1$ and $t \geq  2$ are constants,and a join topology which is a connected undirected graph $G$ on vertices $1,\ldots ,t$ . The result consists of tuples $\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{t}$ where ${r}_{i} \cap  {r}_{j} \neq  \varnothing$ for all $i,j$ connected in $G$ . A structure is feasible if it stores $\widetilde{O}\left( n\right)$ words,supports an update in $\widetilde{O}\left( 1\right)$ amortized time,and can enumerate the join result with an $\widetilde{O}\left( 1\right)$ delay,where $n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right|$ and $\widetilde{O}\left( \text{.}\right) {hidesapolylognfactor}.{Weprovideadichotomyastowhenfeasiblestructuresexist} : {they}$ do when $t = 2$ or $d = 1$ ; subject to the OMv-conjecture,they do not exist when $t \geq  3$ and $d \geq  2$ , regardless of the join topology.

在交集连接中，给定$t$组在${\mathbb{R}}^{d}$中的轴平行矩形集合${R}_{1},\ldots ,{R}_{t}$，其中$d \geq  1$和$t \geq  2$为常数，以及一个连接拓扑，它是顶点为$1,\ldots ,t$的连通无向图$G$。结果由元组$\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{t}$组成，其中对于$G$中所有相连的$i,j$，都有${r}_{i} \cap  {r}_{j} \neq  \varnothing$。如果一个结构存储$\widetilde{O}\left( n\right)$个字，支持在$\widetilde{O}\left( 1\right)$的平摊时间内进行一次更新，并且能够以$\widetilde{O}\left( 1\right)$的延迟枚举连接结果，则该结构是可行的，其中当$t = 2$或$d = 1$时，$n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right|$和$\widetilde{O}\left( \text{.}\right) {hidesapolylognfactor}.{Weprovideadichotomyastowhenfeasiblestructuresexist} : {they}$成立；在OMv猜想的条件下，当$t \geq  3$和$d \geq  2$时，无论连接拓扑如何，这样的结构都不存在。

## Accepted by Journal of Computer and System Sciences (JCSS).

## 已被《计算机与系统科学杂志》（Journal of Computer and System Sciences，JCSS）录用。

Keywords: Intersection Joins, Enumeration, Dynamic Updates, Data Structures, OMv-Conjecture

关键词：交集连接、枚举、动态更新、数据结构、OMv猜想

## 1 Introduction

## 1 引言

Let ${R}_{1},{R}_{2},\ldots ,{R}_{t}$ be $t$ sets of $d$ -dimensional rectangles (note: all the rectangles in this paper are axis-parallel). An intersection join is defined by a join topology, which is a connected undirected graph $G$ on vertices $\{ 1,2,\ldots ,t\}$ . The join result is the set of tuples

设${R}_{1},{R}_{2},\ldots ,{R}_{t}$为$t$组$d$维矩形集合（注意：本文中所有矩形均为轴平行的）。交集连接由一个连接拓扑定义，它是顶点为$\{ 1,2,\ldots ,t\}$的连通无向图$G$。连接结果是元组的集合

$$
\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{t}
$$

where ${r}_{i} \cap  {r}_{j} \neq  \varnothing$ for all $i,j$ such that $G$ has an edge between $i$ and $j$ . Figure 1 shows a join topology with $t = 3$ ,for which the join result consists of all $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  {R}_{1} \times  {R}_{2} \times  {R}_{3}$ satisfying:

其中对于所有满足$G$在$i$和$j$之间有一条边的$i,j$，都有${r}_{i} \cap  {r}_{j} \neq  \varnothing$。图1展示了一个$t = 3$的连接拓扑，其连接结果由所有满足以下条件的$\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  {R}_{1} \times  {R}_{2} \times  {R}_{3}$组成：

$$
\left( {{r}_{1} \cap  {r}_{2} \neq  \varnothing }\right)  \land  \left( {{r}_{2} \cap  {r}_{3} \neq  \varnothing }\right) \text{.} \tag{1}
$$

Set $n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right|$ . We will concentrate on data complexity by restricting $t$ and $d$ to constants. Ideally, we want to maintain a feasible data structure with all of the following guarantees:

设$n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right|$。我们将通过将$t$和$d$限制为常数来关注数据复杂度。理想情况下，我们希望维护一个具有以下所有保证的可行数据结构：

- It stores $\widetilde{O}\left( n\right)$ words,where $\widetilde{O}\left( \text{.}\right)$ hides a polylog $n$ factor.

- 它存储$\widetilde{O}\left( n\right)$个字，其中$\widetilde{O}\left( \text{.}\right)$隐藏了一个关于$n$的多项式对数因子。

- It can be updated in $\widetilde{O}\left( 1\right)$ amortized time,when a rectangle is inserted into or deleted from any ${R}_{i}\left( {1 \leq  i \leq  t}\right)$ .

- 当一个矩形被插入到任何${R}_{i}\left( {1 \leq  i \leq  t}\right)$中或从其中删除时，它可以在$\widetilde{O}\left( 1\right)$的平摊时间内进行更新。

- It can be used to enumerate the join result with a delay of $\Delta  = \widetilde{O}\left( 1\right)$ ,that is:

- 它可以用于以$\Delta  = \widetilde{O}\left( 1\right)$的延迟枚举连接结果，即：

- within $\Delta$ time,the algorithm must either report the first result tuple or terminate (when the join result is empty);

- 在$\Delta$时间内，算法必须要么报告第一个结果元组，要么终止（当连接结果为空时）；

- after reporting a result tuple,within another $\Delta$ time,the algorithm must either report a new result tuple or terminate (when the entire result has been reported).

- 在报告一个结果元组后，在另外的$\Delta$时间内，算法必须要么报告一个新的结果元组，要么终止（当整个结果都已报告时）。

Notice that,if the join result has $k$ tuples,a feasible structure finds all of them in $\widetilde{O}\left( {1 + k}\right)$ time.

请注意，如果连接结果有 $k$ 个元组，那么一个可行的结构可以在 $\widetilde{O}\left( {1 + k}\right)$ 时间内找到所有这些元组。

<!-- Media -->

<!-- figureText: 3 1 -->

<img src="https://cdn.noedgeai.com/0195ccbc-b4ca-70cc-a106-44321046dfa5_1.jpg?x=791&y=1271&w=202&h=171&r=0"/>

Figure 1: A join topology for $t = 3$

图 1：$t = 3$ 的连接拓扑结构

<!-- Media -->

### 1.1 What Intersection Joins can Do

### 1.1 交集连接的作用

The most important non-natural joins are arguably those with a conjunction of predicates, each of which is defined with the " $\leq$ " or " $\geq$ " operator. Many of these joins can be modeled as intersection joins. For example,

可以说，最重要的非自然连接是那些带有谓词合取的连接，其中每个谓词都是用 “ $\leq$ ” 或 “ $\geq$ ” 运算符定义的。许多这样的连接可以建模成交集连接。例如，

$$
\left( {x,y,z}\right)  :  - {T}_{1}\left( x\right) ,{T}_{2}\left( {y,z}\right) ,y \leq  x \leq  z
$$

is an intersection join between two sets of 1D intervals: ${R}_{1} = \left\{  {\left\lbrack  {x,x}\right\rbrack   \mid  x \in  {T}_{1}}\right\}$ and ${R}_{2} = \left\{  \left\lbrack  {y,z}\right\rbrack  \right.$ $\left. {\left( {y,z}\right)  \in  {T}_{2} \land  y \leq  z}\right\}$ . As another example,

是两组一维区间 ${R}_{1} = \left\{  {\left\lbrack  {x,x}\right\rbrack   \mid  x \in  {T}_{1}}\right\}$ 和 ${R}_{2} = \left\{  \left\lbrack  {y,z}\right\rbrack  \right.$ $\left. {\left( {y,z}\right)  \in  {T}_{2} \land  y \leq  z}\right\}$ 之间的交集连接。再举一个例子，

$$
\left( {w,x,y,z}\right)  :  - {T}_{1}\left( {w,x}\right) ,{T}_{2}\left( {y,z}\right) ,w \leq  y,x \geq  z
$$

is an intersection join between two sets of $2\mathrm{D}$ rectangles: ${R}_{1} = \left\{  {\left\lbrack  {w,w}\right\rbrack   \times  \left\lbrack  {x,x}\right\rbrack   \mid  \left( {w,x}\right)  \in  {T}_{1}}\right\}$ and ${R}_{2} = \left\{  {\left( {-\infty ,y\rbrack \times \lbrack z,\infty }\right)  \mid  \left( {y,z}\right)  \in  {T}_{2}}\right\}  .$

是两组 $2\mathrm{D}$ 矩形 ${R}_{1} = \left\{  {\left\lbrack  {w,w}\right\rbrack   \times  \left\lbrack  {x,x}\right\rbrack   \mid  \left( {w,x}\right)  \in  {T}_{1}}\right\}$ 和 ${R}_{2} = \left\{  {\left( {-\infty ,y\rbrack \times \lbrack z,\infty }\right)  \mid  \left( {y,z}\right)  \in  {T}_{2}}\right\}  .$ 之间的交集连接

Outside relational databases, intersection joins find major applications as well:

在关系数据库之外，交集连接也有重要的应用：

- For $d = 1$ ,they are frequent in temporal databases,where a record is associated with a time interval indicating the record's validity period. An intersection join is what is needed to find tuples whose validity periods overlap in a designated manner, and is a crucial operation in many scenarios $\left\lbrack  {8,9,{11},{18},{27}}\right\rbrack$ .

- 对于 $d = 1$ 而言，它们在时态数据库中很常见，在时态数据库中，一条记录与一个表示该记录有效时间段的时间间隔相关联。交集连接是用于以指定方式查找有效时间段重叠的元组所必需的操作，并且在许多场景 $\left\lbrack  {8,9,{11},{18},{27}}\right\rbrack$ 中是一项关键操作。

- For $d \geq  2$ ,intersection joins are known under the name spatial join. This is a core operation in spatial databases, where each object is associated with a rectangle (typically, the minimum bounding rectangle of a geometric entity such as a line segment, a polygon, a circle, etc.). A spatial join is the key to extracting overlay information from different sets of objects and has received considerable attention (e.g., $\left\lbrack  {5,{22},{25},{26},{28},{32}}\right\rbrack$ ).

- 对于 $d \geq  2$ 而言，交集连接以空间连接的名称为人所知。这是空间数据库中的核心操作，在空间数据库中，每个对象都与一个矩形（通常是诸如线段、多边形、圆形等几何实体的最小边界矩形）相关联。空间连接是从不同对象集合中提取重叠信息的关键，并且受到了相当多的关注（例如，$\left\lbrack  {5,{22},{25},{26},{28},{32}}\right\rbrack$）。

### 1.2 Related Work

### 1.2 相关工作

"One-Shot" Intersection Joins. In the offline version of our problem, we want to compute the result of an intersection join on ${R}_{1},\ldots ,{R}_{t}$ . The computation is done only once,namely,there are no updates to worry about. Surprisingly, even this problem has not been well understood, except when the join topology is a tree. Willard [29] showed that the problem can be solved in $\widetilde{O}\left( {n + k}\right)$ time for any tree topology,where $k$ is the number of result tuples. Whether the offline version can be settled using $\widetilde{O}\left( {n + k}\right)$ time for an arbitrary topology is still open,even in 1D space.

“一次性” 交集连接。在我们问题的离线版本中，我们希望计算 ${R}_{1},\ldots ,{R}_{t}$ 上的交集连接结果。该计算只进行一次，即无需担心更新问题。令人惊讶的是，即使是这个问题也没有得到很好的理解，除非连接拓扑结构是树状的。威拉德（Willard）[29] 表明，对于任何树状拓扑结构，该问题可以在 $\widetilde{O}\left( {n + k}\right)$ 时间内解决，其中 $k$ 是结果元组的数量。即使在一维空间中，离线版本是否可以在任意拓扑结构下使用 $\widetilde{O}\left( {n + k}\right)$ 时间解决仍然是一个悬而未决的问题。

Even just for tree topologies, it would be tempting to adapt Willard's algorithm [29] to the dynamic scenario (i.e., our problem). His algorithm in essence processes a tree topology using the "leaf-to-root" semi-join idea that Yannakakis [30] introduced for processing an acyclic natural join. A straightforward adaptation,however,entails either a large update cost of $\widetilde{O}\left( n\right)$ ,or an uninteresting delay of $\Delta  = \widetilde{O}\left( n\right)$ in result enumeration. It is not clear how to improve this without introducing new ideas.

即使仅对于树状拓扑结构，也会很想将威拉德（Willard）的算法 [29] 应用到动态场景（即我们的问题）中。他的算法本质上是使用扬纳卡基斯（Yannakakis）[30] 为处理无环自然连接而引入的 “从叶到根” 半连接思想来处理树状拓扑结构。然而，直接应用该算法要么会导致 $\widetilde{O}\left( n\right)$ 的高更新成本，要么会导致结果枚举出现 $\Delta  = \widetilde{O}\left( n\right)$ 的无趣延迟。目前尚不清楚如何在不引入新思想的情况下改进这一点。

View Maintenance. Our problem can be regarded as a variant of incremental view maintenance. Define a view $W$ as the set of $t$ -tuples in the join result. We want to "maintain" $W$ incrementally with $\widetilde{O}\left( n\right)$ space and $\widetilde{O}\left( 1\right)$ time per update. As mentioned,the maintenance is done by storing ${R}_{1},\ldots ,{R}_{t}$ in a feasible structure,such that $W$ can be extracted from the structure in $\widetilde{O}\left( \left| W\right| \right)$ time whenever needed.

视图维护。我们的问题可以被视为增量视图维护的一种变体。将视图 $W$ 定义为连接结果中 $t$ 元组的集合。我们希望以 $\widetilde{O}\left( n\right)$ 的空间和每次更新 $\widetilde{O}\left( 1\right)$ 的时间增量地“维护” $W$。如前所述，维护是通过将 ${R}_{1},\ldots ,{R}_{t}$ 存储在一个可行的结构中来完成的，这样每当需要时，就可以在 $\widetilde{O}\left( \left| W\right| \right)$ 的时间内从该结构中提取出 $W$。

There are two main challenges in achieving the above goal. First, a join result may have a size up to $O\left( {n}^{t}\right)$ ,which rules out the possibility of materializing the view if the space must be kept $\widetilde{O}\left( n\right)$ . Second,a single update may change a significant portion of the join result. This makes result materialization an infeasible approach even if the join result has a size of $\widetilde{O}\left( n\right)$ .

要实现上述目标存在两个主要挑战。首先，连接结果的大小可能高达 $O\left( {n}^{t}\right)$，如果必须将空间保持在 $\widetilde{O}\left( n\right)$，那么物化视图的可能性就被排除了。其次，一次更新可能会改变连接结果的很大一部分。这使得即使连接结果的大小为 $\widetilde{O}\left( n\right)$，结果物化也是一种不可行的方法。

Overcoming these challenges is non-trivial even for natural joins (a.k.a., conjunctive queries) . Note that the concept of "feasible structure" is readily extendable to a natural join ${R}_{1} \bowtie  \ldots  \bowtie  {R}_{t}$ (in fact, this definition was explicit in [3]). Finding such structures for natural joins was studied as early as in the 80 's of the last century, but with success limited to binary joins [7,12]. As a breakthrough, Berkholz et al. [3] proved that,for natural joins on $3 \leq  t = O\left( 1\right)$ relations,feasible structures exist if and only if the join is $q$ -hierarchical,subject to the OMv-conjecture [13] (see also [14] for similar upper bound results). We refer the reader to [3] for the definition of $q$ -hierarchical (this notion will not be needed in our discussion), but state the OMv-conjecture here:

即使对于自然连接（也称为合取查询），克服这些挑战也并非易事。请注意，“可行结构”的概念很容易扩展到自然连接 ${R}_{1} \bowtie  \ldots  \bowtie  {R}_{t}$（事实上，这个定义在文献 [3] 中是明确的）。早在上个世纪 80 年代就开始研究为自然连接寻找这样的结构，但成功仅限于二元连接 [7,12]。作为一项突破，Berkholz 等人 [3] 证明，对于 $3 \leq  t = O\left( 1\right)$ 个关系上的自然连接，当且仅当该连接是 $q$ 分层的（subject to the OMv - 猜想 [13]）时，才存在可行结构（另见 [14] 中的类似上界结果）。我们请读者参考 [3] 了解 $q$ 分层的定义（在我们的讨论中不需要这个概念），但在此陈述 OMv 猜想：

OMv-conjecture [13]. Consider the following online boolean matrix-vector multiplication (OMv) problem. An algorithm is given an $n \times  n$ boolean matrix $\mathbf{M}$ ,and is allowed to preprocess $\mathbf{M}$ arbitrarily in poly(n)time. Then,the algorithm is given a stream of $n \times  1$ boolean vectors ${\mathbf{v}}_{\mathbf{1}},{\mathbf{v}}_{\mathbf{2}},\ldots ,{\mathbf{v}}_{\mathbf{n}}$ ,and is required to compute $\mathbf{M}{\mathbf{v}}_{\mathbf{i}}$ for each $i$ (the additions and multiplications on the elements of the matrices are performed in the boolean semi ring). In particular,vector ${\mathbf{v}}_{i + 1}\left( {i \geq  1}\right)$ is fed to the algorithm only after it has correctly output $\mathbf{M}{\mathbf{v}}_{\mathbf{i}}$ . The cost is the total time the algorithm takes in computing the $n$ multiplications. The OMv-conjecture states that, no algorithms can solve the problem successfully with probability at least $2/3$ in $O\left( {n}^{3 - \epsilon }\right)$ time,for any constant $\epsilon  > 0$ .

OMv 猜想 [13]。考虑以下在线布尔矩阵 - 向量乘法（OMv）问题。给一个算法一个 $n \times  n$ 的布尔矩阵 $\mathbf{M}$，并允许该算法在多项式时间 poly(n) 内对 $\mathbf{M}$ 进行任意预处理。然后，给该算法一个包含 $n \times  1$ 个布尔向量 ${\mathbf{v}}_{\mathbf{1}},{\mathbf{v}}_{\mathbf{2}},\ldots ,{\mathbf{v}}_{\mathbf{n}}$ 的流，并要求该算法为每个 $i$ 计算 $\mathbf{M}{\mathbf{v}}_{\mathbf{i}}$（矩阵元素上的加法和乘法在布尔半环中进行）。特别地，只有在算法正确输出 $\mathbf{M}{\mathbf{v}}_{\mathbf{i}}$ 之后，才会将向量 ${\mathbf{v}}_{i + 1}\left( {i \geq  1}\right)$ 输入给该算法。成本是该算法计算 $n$ 次乘法所花费的总时间。OMv 猜想指出，对于任何常数 $\epsilon  > 0$，没有算法能够在 $O\left( {n}^{3 - \epsilon }\right)$ 的时间内以至少 $2/3$ 的概率成功解决该问题。

Partially inspired by the above, recent years have witnessed efforts (see the representative works $\left\lbrack  {{16},{17}}\right\rbrack  )$ studying non-feasible structures on natural joins that can provide a good tradeoff between update efficiency and enumeration delay.

部分受上述内容的启发，近年来人们做出了一些努力（见代表性工作 $\left\lbrack  {{16},{17}}\right\rbrack  )$），研究自然连接上的非可行结构，这些结构可以在更新效率和枚举延迟之间提供良好的权衡。

None of the above works considered non-natural joins. To fill the void, Idris et al. [15] studied how to maintain data structures that can answer conjunctive queries with inequality predicates, and support efficient updates on the participating relations. Like our work, the structure should allow the query result (i.e.,a join result) to be enumerated with an $\widetilde{O}\left( 1\right)$ delay,but unlike our work, the structure is permitted to spend $\widetilde{O}\left( n\right)$ time to support each update,where $n$ is the size of the database (hence, the structure is not feasible).

上述所有工作均未考虑非自然连接。为填补这一空白，伊德里斯（Idris）等人 [15] 研究了如何维护数据结构，使其能够回答带有不等式谓词的连接查询，并支持对参与关系进行高效更新。与我们的工作类似，该结构应允许以 $\widetilde{O}\left( 1\right)$ 的延迟枚举查询结果（即连接结果），但与我们的工作不同的是，该结构允许花费 $\widetilde{O}\left( n\right)$ 的时间来支持每次更新，其中 $n$ 是数据库的大小（因此，该结构不可行）。

It is worth mentioning that the form of maintenance discussed above is different from another (perhaps more traditional) branch of incremental view maintenance, which aims at computing the delta result changes of a join caused by an update, i.e., find (i) all the new result tuples created by an insertion, and (ii) the existing result tuples removed by a deletion. Indeed, many works in the literature have explicitly focused on this branch; e.g.,see $\left\lbrack  {3,4,{14},{19} - {21},{31}}\right\rbrack$ and the references therein. In fact, feasible structures can be deployed to support the above style of maintenance as well, using a reduction which we explain in Appendix B.

值得一提的是，上述讨论的维护形式与增量视图维护的另一个（可能更传统的）分支不同，该分支旨在计算由更新引起的连接的增量结果变化，即找出 (i) 由插入操作创建的所有新结果元组，以及 (ii) 由删除操作移除的现有结果元组。实际上，文献中的许多工作都明确关注这一分支；例如，参见 $\left\lbrack  {3,4,{14},{19} - {21},{31}}\right\rbrack$ 及其参考文献。事实上，通过附录 B 中解释的一种归约方法，可行的结构也可用于支持上述维护方式。

### 1.3 Contributions

### 1.3 贡献

This paper provides a complete dichotomy on when an intersection join admits a feasible structure. Next, we provide an overview of our results and the proposed techniques.

本文对交集连接何时允许存在可行结构给出了完整的二分法。接下来，我们将概述我们的研究结果和所提出的技术。

$t = 2$ (Binary Joins). It is a good idea to start with the most fundamental: in our context,a binary intersection join in 1D space $\left( {t = 2,d = 1}\right)$ . For such joins,we can prove:

$t = 2$（二元连接）。从最基础的情况入手是个不错的选择：在我们的研究背景下，一维空间 $\left( {t = 2,d = 1}\right)$ 中的二元交集连接。对于此类连接，我们可以证明：

Theorem 1. For an intersection join with $d = 1$ and $t = 2$ ,there is a structure of $O\left( n\right)$ space that can be updated in $O\left( {\log n}\right)$ amortized time,and can be used to enumerate the join result with a constant delay.

定理 1。对于具有 $d = 1$ 和 $t = 2$ 的交集连接，存在一个占用 $O\left( n\right)$ 空间的结构，该结构可以在 $O\left( {\log n}\right)$ 的平摊时间内完成更新，并且可以用于以恒定延迟枚举连接结果。

The structure is asymptotically optimal in the comparison model of computation (see Appendix A for a proof), and contains just the right amount of sophistication for demonstrating two new ideas that are also applied in some other structures of the paper:

在比较计算模型中，该结构是渐近最优的（证明见附录 A），并且其复杂度恰到好处，足以展示两个新思想，这两个思想也应用于本文的其他一些结构中：

- Productive list: One issue in designing a feasible structure is how to enumerate a join result of an exceedingly small size $k$ . As the enumeration can take only $\widetilde{O}\left( {1 + k}\right)$ time,when $k \ll  n$ ,we cannot afford to read the whole input. We remedy the issue by marking certain nodes of the structure as "productive": these nodes tell us where to look to start reporting result tuples immediately. If $k = 0$ ,no productive nodes exist,permitting us to finish in constant time.

- 有效列表：设计可行结构时的一个问题是如何枚举规模极小 $k$ 的连接结果。由于枚举仅能花费 $\widetilde{O}\left( {1 + k}\right)$ 的时间，当 $k \ll  n$ 时，我们无法读取整个输入。我们通过将结构中的某些节点标记为“有效”来解决这个问题：这些节点告诉我们从哪里开始立即报告结果元组。如果 $k = 0$，则不存在有效节点，这样我们可以在恒定时间内完成操作。

- Buffering: Often times, it would be easier to come up with an intersection join algorithm that runs in $\widetilde{O}\left( {1 + k}\right)$ time,but harder to guarantee an $\widetilde{O}\left( 1\right)$ delay. If we could always turn such an algorithm into one with an $\widetilde{O}\left( 1\right)$ delay,designing feasible structures would become considerably easier. We propose a buffering technique to make this possible, provided that the algorithm is "aggressive" in reporting. Intuitively, such an algorithm would output most of the result during an early stage, and then possibly remain "quiet" for a long time before reporting the rest.

- 缓冲：很多时候，设计一个运行时间为 $\widetilde{O}\left( {1 + k}\right)$ 的交集连接算法相对容易，但要保证 $\widetilde{O}\left( 1\right)$ 的延迟则较为困难。如果我们总能将这样的算法转换为具有 $\widetilde{O}\left( 1\right)$ 延迟的算法，那么设计可行结构将变得相当容易。我们提出了一种缓冲技术来实现这一点，前提是该算法在报告结果时是“积极的”。直观地说，这样的算法会在早期输出大部分结果，然后可能会在很长一段时间内“安静”，之后再报告其余结果。

Our 1D structure can be extended to higher dimensionalities:

我们的一维结构可以扩展到更高维度：

Theorem 2. For any intersection join with $d = O\left( 1\right)$ and $t = 2$ ,there is a structure of $\widetilde{O}\left( n\right)$ space that can be updated in $\widetilde{O}\left( 1\right)$ amortized time,and can be used to enumerate the join result with an $\widetilde{O}\left( 1\right)$ delay.

定理 2。对于任何具有 $d = O\left( 1\right)$ 和 $t = 2$ 的交集连接，存在一个占用 $\widetilde{O}\left( n\right)$ 空间的结构，该结构可以在 $\widetilde{O}\left( 1\right)$ 的平摊时间内完成更新，并且可以用于以 $\widetilde{O}\left( 1\right)$ 的延迟枚举连接结果。

$t \geq  3$ and $d \geq  2$ . For intersection joins on $t \geq  3$ sets of rectangles,we are able to show that no feasible structures are likely to exist when the dimensionality is at least 2 :

$t \geq  3$ 和 $d \geq  2$。对于 $t \geq  3$ 矩形集合上的交集连接，我们能够证明，当维度至少为 2 时，可能不存在可行的结构：

Theorem 3. Unless the OMv-conjecture [13] fails,for any intersection join with $t \geq  3$ and $d \geq  2$ (regardless of the join topology), no structure can offer the following guarantees simultaneously: for some constant $0 < \epsilon  < {0.5}$ ,it (i) can be updated in $O\left( {n}^{{0.5} - \epsilon }\right)$ time,and (ii) can be used to enumerate the join result with a delay of $O\left( {n}^{{0.5} - \epsilon }\right)$ .

定理3. 除非OMv猜想[13]不成立，对于任何具有$t \geq  3$和$d \geq  2$的交集连接（无论连接拓扑如何），没有一种结构能够同时提供以下保证：对于某个常数$0 < \epsilon  < {0.5}$，(i) 它可以在$O\left( {n}^{{0.5} - \epsilon }\right)$时间内进行更新，并且 (ii) 可以用于以$O\left( {n}^{{0.5} - \epsilon }\right)$的延迟枚举连接结果。

Our proof is based on a reduction from a negative result established in [3] about the natural join ${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$ where the three relations have schemas ${T}_{1}\left( X\right) ,{T}_{2}\left( {X,Y}\right) ,{T}_{3}\left( Y\right)$ . The fact that this particular natural join "seals the fate" of all intersection joins of $t \geq  3$ and $d \geq  2$ is mildly surprising.

我们的证明基于[3]中针对自然连接${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$所建立的一个否定结果的归约，其中三个关系具有模式${T}_{1}\left( X\right) ,{T}_{2}\left( {X,Y}\right) ,{T}_{3}\left( Y\right)$。这个特定的自然连接“决定了”所有$t \geq  3$和$d \geq  2$的交集连接的“命运”，这有点令人惊讶。

By applying Theorem 3 to tree topologies, one can see that our problem is inherently more difficult than its offline version (see Section 1.2) in the following sense. First, recall that Willard [29] gave an offline algorithm that runs in $\widetilde{O}\left( {n + k}\right)$ time for any tree topology and any constant dimensionality $d$ . On the other hand,a feasible structure immediately provides an offline algorithm: one can simply perform $n$ insertions and then enumerate the join result. Thus,Theorem 3 points out that no such structures can offer an offline algorithm that matches Willard's solution in running time,when $t \geq  3$ and $d \geq  2$ ,even if the topology is a tree.

通过将定理3应用于树拓扑，可以看出我们的问题在以下意义上本质上比其离线版本（见第1.2节）更难。首先，回顾一下，威拉德（Willard）[29]给出了一种离线算法，对于任何树拓扑和任何常数维度$d$，该算法的运行时间为$\widetilde{O}\left( {n + k}\right)$。另一方面，一个可行的结构立即提供了一种离线算法：可以简单地执行$n$次插入操作，然后枚举连接结果。因此，定理3指出，当$t \geq  3$和$d \geq  2$时，即使拓扑是树，也没有这样的结构能够提供一种在运行时间上与威拉德的解决方案相匹配的离线算法。

1D Joins with $t \geq  3$ Sets. This last landscape turns out to be the most challenging (perhaps the most exciting). As explained in the preceding paragraph, if a feasible structure exists for any join topology when $d = 1$ ,then the offline version can be settled in $\widetilde{O}\left( {n + k}\right)$ time in 1D space for all topologies (not just trees as in [29]) - but whether that is achievable is still open.

具有$t \geq  3$个集合的一维连接。事实证明，最后这种情况是最具挑战性的（也许也是最令人兴奋的）。如前一段所述，如果当$d = 1$时对于任何连接拓扑都存在一个可行的结构，那么一维空间中所有拓扑（不仅仅是[29]中的树拓扑）的离线版本可以在$\widetilde{O}\left( {n + k}\right)$时间内解决——但这是否可以实现仍然未知。

We managed to overcome the challenge:

我们成功克服了这一挑战：

Theorem 4. For any 1D intersection join with constant $t \geq  3$ ,there is a structure of $\widetilde{O}\left( n\right)$ space that can be updated in $\widetilde{O}\left( 1\right)$ amortized time,and can be used to enumerate the join result with an $\widetilde{O}\left( 1\right)$ delay.

定理4. 对于任何具有常数$t \geq  3$的一维交集连接，存在一个占用$\widetilde{O}\left( n\right)$空间的结构，该结构可以在$\widetilde{O}\left( 1\right)$的均摊时间内进行更新，并且可以用于以$\widetilde{O}\left( 1\right)$的延迟枚举连接结果。

The theorem thus solves the offline intersection join problem in 1D space for arbitrary topologies on any constant number $t$ of interval sets. In particular,we guarantee not only a total output time of $\widetilde{O}\left( {1 + k}\right)$ ,but also an $\widetilde{O}\left( 1\right)$ delay as well,provided that $\widetilde{O}\left( n\right)$ preprocessing time is allowed.

因此，该定理解决了一维空间中任意数量（常数$t$）的区间集在任意拓扑下的离线交集连接问题。特别地，只要允许$\widetilde{O}\left( n\right)$的预处理时间，我们不仅保证总输出时间为$\widetilde{O}\left( {1 + k}\right)$，还保证$\widetilde{O}\left( 1\right)$的延迟。

Our structure incorporates a series of new ideas many of which are too detailed for the high-level discussion here, but two particular techniques are notable:

我们的结构融入了一系列新思想，其中许多思想过于细节化，不适合在这里进行高层次的讨论，但有两种特定的技术值得注意：

- Lexicographic ordering: We conceptually order all the result tuples $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ by concatenating the left endpoints of ${r}_{1},\ldots ,{r}_{t}$ ,and comparing the concatenated sequences lexicographically. The core of our structure is to find the first result tuple by this ordering. As the structure supports fast updates, we can find the entire result efficiently by repeatedly finding the "first" result tuple, after deleting certain tuples appropriately. The deleted tuples are eventually added back into the structure at the end of the join. This turns out to be a powerful method, and plays an indispensable role in our proof of Theorem 4.

- 字典序排序：我们在概念上通过连接${r}_{1},\ldots ,{r}_{t}$的左端点，并按字典序比较连接后的序列，对所有结果元组$\left( {{r}_{1},\ldots ,{r}_{t}}\right)$进行排序。我们结构的核心是通过这种排序找到第一个结果元组。由于该结构支持快速更新，我们可以在适当地删除某些元组后，通过反复找到“第一个”结果元组来高效地找到整个结果。被删除的元组最终会在连接结束时重新添加回结构中。事实证明，这是一种强大的方法，在我们对定理4的证明中起着不可或缺的作用。

- Recursive topology partitioning: The second technique we devised is a recursive mechanism for processing a 1D intersection join. The mechanism removes certain vertices from the join topology $G$ ,and breaks the remaining parts of $G$ into maximally connected subgraphs. Each subgraph gives rise to a smaller join to be handled by recursion. The correctness of the mechanism is based crucially on numerous characteristics of the problem in 1D space.

- 递归拓扑划分：我们设计的第二种技术是一种用于处理一维交集连接的递归机制。该机制从连接拓扑 $G$ 中移除某些顶点，并将 $G$ 的其余部分拆分为最大连通子图。每个子图会产生一个更小的连接，通过递归进行处理。该机制的正确性关键取决于一维空间中该问题的众多特性。

## 2 Preliminaries

## 2 预备知识

Throughout the paper,we consider that the input sets ${R}_{1},\ldots ,{R}_{t}$ are in "general position". To state this assumption formally,take a rectangle $r \in  \mathop{\bigcup }\limits_{i}{R}_{i}$ . If the projection of $r$ onto dimension $j \in  \left\lbrack  {1,d}\right\rbrack$ is an interval $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ ,we say that $r$ defines the coordinates ${x}_{1}$ and ${x}_{2}$ on dimension $j$ . The general position assumption says that every coordinate of any dimension is defined by at most one rectangle in $\mathop{\bigcup }\limits_{i}{R}_{i}$ . The assumption allows us to focus on explaining the new ideas behind our techniques. Removing the assumption can be done with standard tie-breaking techniques (see, e.g., [6]), and does not affect any of our claims.

在整篇论文中，我们假设输入集 ${R}_{1},\ldots ,{R}_{t}$ 处于“一般位置”。为了正式表述这一假设，取一个矩形 $r \in  \mathop{\bigcup }\limits_{i}{R}_{i}$ 。如果 $r$ 在维度 $j \in  \left\lbrack  {1,d}\right\rbrack$ 上的投影是一个区间 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ ，我们称 $r$ 在维度 $j$ 上定义了坐标 ${x}_{1}$ 和 ${x}_{2}$ 。一般位置假设表明，任何维度的每个坐标最多由 $\mathop{\bigcup }\limits_{i}{R}_{i}$ 中的一个矩形定义。该假设使我们能够专注于解释我们技术背后的新思想。可以使用标准的平局决胜技术（例如，参见 [6]）来去除该假设，并且这不会影响我们的任何结论。

### 2.1 Binary Search Tree (BST)

### 2.1 二叉搜索树（BST）

Even though the BST is a rudimentary structure, it can be described in multiple ways, whose differences are usually subtle, but can cause ambiguity when one needs to design new structures by augmenting BSTs. Next, we clarify the BST assumed in this paper and take the opportunity to define some relevant concepts and notations.

尽管二叉搜索树是一种基础结构，但它可以用多种方式描述，这些方式的差异通常很细微，但在需要通过扩展二叉搜索树来设计新结构时可能会导致歧义。接下来，我们明确本文所假设的二叉搜索树，并借此机会定义一些相关的概念和符号。

Let $\mathcal{S}$ be a set of $n$ values in $\mathbb{R}$ . A BST $\mathcal{T}$ on $\mathcal{S}$ is a binary tree with the following properties:

设 $\mathcal{S}$ 是 $\mathbb{R}$ 中的一组 $n$ 个值。 $\mathcal{S}$ 上的二叉搜索树 $\mathcal{T}$ 是一棵具有以下属性的二叉树：

- Each leaf stores a distinct element of $\mathcal{S}$ ,and conversely,every element of $\mathcal{S}$ is stored at a leaf.

- 每个叶子节点存储 $\mathcal{S}$ 中的一个不同元素，反之， $\mathcal{S}$ 中的每个元素都存储在一个叶子节点中。

- Every internal node $u$ has two child nodes. It stores a value - called the search key of $u$ and denoted as $\operatorname{key}\left( u\right)$ - which equals the smallest element of its right subtree.

- 每个内部节点 $u$ 有两个子节点。它存储一个值 —— 称为 $u$ 的搜索键，记为 $\operatorname{key}\left( u\right)$ —— 该值等于其右子树中的最小元素。

- For each internal node $u$ ,all the elements stored in its left (or right) subtree must be less than (or at least,resp.) $\operatorname{key}\left( u\right)$ .

- 对于每个内部节点 $u$ ，其左（或右）子树中存储的所有元素必须小于（或至少等于，分别地） $\operatorname{key}\left( u\right)$ 。

We conceptually associate each node $u$ of $\mathcal{T}$ with a slab $\sigma \left( u\right)$ ,which is a semi-open interval defined as follows. If $u$ is a leaf storing an element $p \in  \mathcal{S}$ ,then $\sigma \left( u\right)  = \left\lbrack  {p,{p}^{\prime }}\right)$ where ${p}^{\prime }$ is the successor of $p$ in $\mathcal{S}$ ; in the special case where $p$ is already the largest element in $\mathcal{S},\sigma \left( u\right)  = \lbrack p,\infty )$ . If $u$ is an internal node, $\sigma \left( u\right)$ is the union of the slabs of its child nodes.

我们在概念上将 $\mathcal{T}$ 的每个节点 $u$ 与一个板片 $\sigma \left( u\right)$ 关联起来，板片是一个半开区间，定义如下。如果 $u$ 是一个存储元素 $p \in  \mathcal{S}$ 的叶子节点，那么 $\sigma \left( u\right)  = \left\lbrack  {p,{p}^{\prime }}\right)$ ，其中 ${p}^{\prime }$ 是 $p$ 在 $\mathcal{S}$ 中的后继；在特殊情况下， $p$ 已经是 $\mathcal{S},\sigma \left( u\right)  = \lbrack p,\infty )$ 中的最大元素。如果 $u$ 是一个内部节点， $\sigma \left( u\right)$ 是其子节点的板片的并集。

For each node $u$ of $\mathcal{T}$ ,we use ${\mathcal{T}}_{u}$ to represent the subtree rooted at $u$ ,and define its subtree size — denoted as $\left| {\mathcal{T}}_{u}\right|$ — as the number of leaf nodes in ${\mathcal{T}}_{u}$ .

对于 $\mathcal{T}$ 的每个节点 $u$ ，我们使用 ${\mathcal{T}}_{u}$ 表示以 $u$ 为根的子树，并将其子树大小（记为 $\left| {\mathcal{T}}_{u}\right|$ ）定义为 ${\mathcal{T}}_{u}$ 中的叶子节点数量。

### 2.2 The Interval Tree

### 2.2 区间树

Let $\mathcal{R}$ be a set of intervals in $\mathbb{R}$ . Next,we describe what is an interval tree $\left\lbrack  {{10},{23}}\right\rbrack$ on $\mathcal{R}$ .

设 $\mathcal{R}$ 是 $\mathbb{R}$ 中的一组区间。接下来，我们描述什么是 $\mathcal{R}$ 上的区间树 $\left\lbrack  {{10},{23}}\right\rbrack$ 。

Let $\mathcal{T}$ be a BST on the set of endpoints of the intervals in $\mathcal{R}$ . Associate each node $u$ in $\mathcal{T}$ with a stabbing set - which we denote as $\operatorname{stab}\left( u\right)$ - including all and only intervals $r \in  \mathcal{R}$ with the property that $u$ is the highest node in $\mathcal{T}$ whose search key is covered by $r$ . At $u,{stab}\left( u\right)$ is stored in two separate lists: one sorted by the left endpoints of the intervals therein, and the other by their right endpoints. This completes the definition of the interval tree.

设$\mathcal{T}$是关于区间集合$\mathcal{R}$端点的一棵二叉搜索树（BST）。将$\mathcal{T}$中的每个节点$u$与一个穿刺集（stabbing set）关联起来——我们将其表示为$\operatorname{stab}\left( u\right)$——该集合包含且仅包含所有满足以下性质的区间$r \in  \mathcal{R}$：$u$是$\mathcal{T}$中搜索键被$r$覆盖的最高节点。在$u,{stab}\left( u\right)$处，区间存储在两个单独的列表中：一个按其中区间的左端点排序，另一个按其右端点排序。至此，区间树的定义完成。

<!-- Media -->

<!-- figureText: ${u}_{15}$ ${u}_{14}$ ${u}_{11}$ ${u}_{12}$ ${u}_{5}$ ${u}_{6}$ ${u}_{7}$ $u{}_{8}$ 。 16 ${u}_{13}$ ${u}_{9}$ ${u}_{10}$ ${u}_{3}$ 6 -->

<img src="https://cdn.noedgeai.com/0195ccbc-b4ca-70cc-a106-44321046dfa5_6.jpg?x=549&y=197&w=687&h=567&r=0"/>

Figure 2: A BST on the endpoints of 8 intervals

图2：8个区间端点的二叉搜索树（BST）

<!-- Media -->

Example. Suppose that $\mathcal{R} = \{ \left\lbrack  {1,2}\right\rbrack  ,\left\lbrack  {3,7}\right\rbrack  ,\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  ,\left\lbrack  {{10},{14}}\right\rbrack  ,\left\lbrack  {{13},{16}}\right\rbrack  \}$ . Figure 2 shows a BST on the endpoints of the 8 intervals.

示例。假设$\mathcal{R} = \{ \left\lbrack  {1,2}\right\rbrack  ,\left\lbrack  {3,7}\right\rbrack  ,\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  ,\left\lbrack  {{10},{14}}\right\rbrack  ,\left\lbrack  {{13},{16}}\right\rbrack  \}$。图2展示了一个基于8个区间端点的二叉搜索树（BST）。

Consider the root ${u}_{15}$ . It has a search key $\operatorname{key}\left( {u}_{15}\right)  = 9$ and a stabbing set $\operatorname{stab}\left( {u}_{15}\right)  = \{ \left\lbrack  {4,{12}}\right\rbrack$ , $\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  \}$ . Similarly,one can verify that $\operatorname{stab}\left( {u}_{1}\right)  = \{ \left\lbrack  {1,2}\right\rbrack  \} ,\operatorname{stab}\left( {u}_{13}\right)  = \{ \left\lbrack  {3,7}\right\rbrack  \}$ ,and $\operatorname{stab}\left( {u}_{14}\right)  = \{ \left\lbrack  {{10},{14}}\right\rbrack  ,\left\lbrack  {{13},{16}}\right\rbrack  \} .$

考虑根节点${u}_{15}$。它有一个搜索键$\operatorname{key}\left( {u}_{15}\right)  = 9$和一个穿刺集（stabbing set）$\operatorname{stab}\left( {u}_{15}\right)  = \{ \left\lbrack  {4,{12}}\right\rbrack$，$\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  \}$。类似地，可以验证$\operatorname{stab}\left( {u}_{1}\right)  = \{ \left\lbrack  {1,2}\right\rbrack  \} ,\operatorname{stab}\left( {u}_{13}\right)  = \{ \left\lbrack  {3,7}\right\rbrack  \}$和$\operatorname{stab}\left( {u}_{14}\right)  = \{ \left\lbrack  {{10},{14}}\right\rbrack  ,\left\lbrack  {{13},{16}}\right\rbrack  \} .$

We will use ${\operatorname{stab}}^{ < }\left( u\right)$ to represent the set of intervals stored in the stabbing sets that are in the left subtree of $u$ . Define ${\operatorname{stab}}^{ > }\left( u\right)$ analogously with respect to the right subtree of $u$ . The following facts are rudimentary:

我们将使用${\operatorname{stab}}^{ < }\left( u\right)$来表示存储在$u$左子树中穿刺集里的区间集合。类似地，相对于$u$的右子树定义${\operatorname{stab}}^{ > }\left( u\right)$。以下事实是基本的：

- Every interval in $\mathcal{R}$ belongs to exactly one stabbing set.

- $\mathcal{R}$中的每个区间恰好属于一个穿刺集（stabbing set）。

- For each node $u,\left| {{\operatorname{stab}}^{ < }\left( u\right) }\right|  + \left| {\operatorname{stab}\left( u\right) }\right|  + \left| {{\operatorname{stab}}^{ > }\left( u\right) }\right|$ can never exceed $\left| {\mathcal{T}\left( u\right) }\right|$ because the endpoints of each interval in ${\operatorname{stab}}^{ < }\left( u\right)  \cup  s\operatorname{tab}\left( u\right)  \cup  {\operatorname{stab}}^{ > }\left( u\right)$ must be stored at nodes in the subtree of $u$ .

- 对于每个节点$u,\left| {{\operatorname{stab}}^{ < }\left( u\right) }\right|  + \left| {\operatorname{stab}\left( u\right) }\right|  + \left| {{\operatorname{stab}}^{ > }\left( u\right) }\right|$，其数量永远不会超过$\left| {\mathcal{T}\left( u\right) }\right|$，因为${\operatorname{stab}}^{ < }\left( u\right)  \cup  s\operatorname{tab}\left( u\right)  \cup  {\operatorname{stab}}^{ > }\left( u\right)$中每个区间的端点必须存储在$u$的子树节点中。

### 2.3 Weight-Balancing Lemmas for Updates

### 2.3 更新的权重平衡引理

#### 2.3.1 Weight-Balancing on the Interval Tree

#### 2.3.1 区间树的权重平衡

The interval tree serves as the base of several structures proposed in this paper. As we will see, our structures will associate each node $u$ in an interval tree $\mathcal{T}$ with an additional secondary structure, denoted as ${\Gamma }_{u}$ . We want to avoid a full-blown description on how to update the resulting interval tree (i.e., augmented with all the secondary structures). There are two reasons. First, the challenges are to figure out how data should be organized in ${\Gamma }_{u}$ ,as opposed to how to update ${\Gamma }_{u}$ . Second, unfolding all the details of updating would force us to ramble on many standard techniques related to weight balancing. The reader would find it rather tedious and unrewarding to plow through all that technical content.

区间树（Interval tree）是本文提出的几种结构的基础。正如我们将看到的，我们的结构会将区间树 $\mathcal{T}$ 中的每个节点 $u$ 与一个额外的二级结构相关联，记为 ${\Gamma }_{u}$。我们不想详细描述如何更新最终的区间树（即，添加了所有二级结构的区间树）。有两个原因。首先，挑战在于弄清楚数据应如何在 ${\Gamma }_{u}$ 中组织，而不是如何更新 ${\Gamma }_{u}$。其次，详细展开更新的所有细节会迫使我们赘述许多与权重平衡相关的标准技术。读者会觉得钻研所有这些技术内容相当乏味且没有收获。

Fortunately, we are able to find a "middle ground" to avoid most of the details, and yet allow the reader to verify the correctness of our algorithms in a (much) lighter way. This is achieved by extracting the key properties that ${\Gamma }_{u}$ needs to have,for the overall augmented interval tree to have the desired update efficiency.

幸运的是，我们能够找到一个“折中点”来避免大部分细节，同时让读者以（更）轻松的方式验证我们算法的正确性。这是通过提取 ${\Gamma }_{u}$ 为使整个增强型区间树具有所需的更新效率而需要具备的关键属性来实现的。

To explain,again let $\mathcal{R}$ be the underlying set of intervals on which the interval tree $\mathcal{T}$ is built, and set $n$ to the number of nodes in $\mathcal{T}$ . Fix any node $u$ in $\mathcal{T}$ . If $u$ is an internal node with child nodes ${v}_{1},{v}_{2}$ ,we will assume that ${\Gamma }_{{v}_{1}}$ and ${\Gamma }_{{v}_{2}}$ are both ready. We prescribe four properties ${P1} - {P4}$ that need to be satisfied by ${\Gamma }_{u}$ :

为了解释这一点，再次设 $\mathcal{R}$ 为构建区间树 $\mathcal{T}$ 所基于的区间集合，并设 $n$ 为 $\mathcal{T}$ 中的节点数量。固定 $\mathcal{T}$ 中的任意节点 $u$。如果 $u$ 是一个具有子节点 ${v}_{1},{v}_{2}$ 的内部节点，我们将假设 ${\Gamma }_{{v}_{1}}$ 和 ${\Gamma }_{{v}_{2}}$ 都已就绪。我们规定 ${\Gamma }_{u}$ 需要满足四个属性 ${P1} - {P4}$：

- $\mathbf{{P1}}$ : When an interval $r = \left\lbrack  {x,y}\right\rbrack   \in  \mathcal{R}$ with $y < \operatorname{key}\left( u\right)$ is inserted or deleted in ${\operatorname{stab}}^{ < }\left( u\right)$ ,we can update ${\Gamma }_{u}$ in ${f}_{1}\left( n\right)$ amortized time,for some function ${f}_{1}$ .

- $\mathbf{{P1}}$：当一个满足 $y < \operatorname{key}\left( u\right)$ 的区间 $r = \left\lbrack  {x,y}\right\rbrack   \in  \mathcal{R}$ 在 ${\operatorname{stab}}^{ < }\left( u\right)$ 中插入或删除时，对于某个函数 ${f}_{1}$，我们可以在 ${f}_{1}\left( n\right)$ 的均摊时间内更新 ${\Gamma }_{u}$。

- $\mathbf{{P2}}$ : Given an interval $r \in  \mathcal{R}$ with $x \leq  \operatorname{key}\left( u\right)  \leq  y$ is inserted or deleted in $\operatorname{stab}\left( u\right)$ ,we can update ${\Gamma }_{u}$ in ${f}_{2}\left( n\right)$ amortized time,for some function ${f}_{2}$ .

- $\mathbf{{P2}}$：当一个满足 $x \leq  \operatorname{key}\left( u\right)  \leq  y$ 的区间 $r \in  \mathcal{R}$ 在 $\operatorname{stab}\left( u\right)$ 中插入或删除时，对于某个函数 ${f}_{2}$，我们可以在 ${f}_{2}\left( n\right)$ 的均摊时间内更新 ${\Gamma }_{u}$。

- $\mathbf{{P3}}$ : Given an interval $r = \left\lbrack  {x,y}\right\rbrack   \in  \mathcal{R}$ with $\operatorname{key}\left( u\right)  < x$ is inserted or deleted in ${\operatorname{stab}}^{ > }\left( u\right)$ ,we can update ${\Gamma }_{u}$ in in ${f}_{3}\left( n\right)$ amortized time,for some function ${f}_{3}$ .

- $\mathbf{{P3}}$：当一个满足 $\operatorname{key}\left( u\right)  < x$ 的区间 $r = \left\lbrack  {x,y}\right\rbrack   \in  \mathcal{R}$ 在 ${\operatorname{stab}}^{ > }\left( u\right)$ 中插入或删除时，对于某个函数 ${f}_{3}$，我们可以在 ${f}_{3}\left( n\right)$ 的均摊时间内更新 ${\Gamma }_{u}$。

- $\mathbf{{P4}} : {\Gamma }_{u}$ can be constructed in ${f}_{4}\left( \left| {\mathcal{T}}_{u}\right| \right)$ time under the condition that,the intervals in $\operatorname{stab}\left( u\right)$ have been sorted in two separate lists: one by left endpoint, and the other by right endpoint.

- 在 $\operatorname{stab}\left( u\right)$ 中的区间已分别按左端点和右端点排序成两个列表的条件下，$\mathbf{{P4}} : {\Gamma }_{u}$ 可以在 ${f}_{4}\left( \left| {\mathcal{T}}_{u}\right| \right)$ 时间内构建。

When ${\Gamma }_{u}$ meets the above requirements,we have following guarantee:

当 ${\Gamma }_{u}$ 满足上述要求时，我们有以下保证：

Lemma 5. The augmented interval tree $\mathcal{T}$ can be updated in

引理 5。当在 $\mathcal{T}$ 中插入或删除一个区间时，增强型区间树 $\mathcal{T}$ 可以在

$$
O\left( {\log n \cdot  \left( {1 + {f}_{1}\left( n\right)  + {f}_{3}\left( n\right) }\right)  + {f}_{2}\left( n\right)  + \frac{\log n \cdot  {f}_{4}\left( n\right) }{n}}\right) 
$$

amortized time when an interval is inserted or deleted in $\mathcal{R}$ .

均摊时间内更新。

Proof. This is a corollary of the results in [2] (see also [24]).

证明。这是文献 [2] 中结果的一个推论（另见文献 [24]）。

#### 2.3.2 Weight-Balancing on the BST

#### 2.3.2 二叉搜索树（BST）上的权重平衡

Next,we mention another result similar to Lemma 5 that is pertinent to BSTs. Let $\mathcal{T}$ be a BST on a set $\mathcal{S}$ of $n$ values in $\mathbb{R}$ . Suppose that we associate each node $u$ of $\mathcal{T}$ with a secondary structure ${\Gamma }_{u}$ having the following guarantees:

接下来，我们提及另一个与引理5类似且与二叉搜索树（BST）相关的结果。设$\mathcal{T}$是集合$\mathcal{S}$上的一棵二叉搜索树，该集合包含$n$个取值于$\mathbb{R}$的值。假设我们将$\mathcal{T}$的每个节点$u$与一个具有以下特性的二级结构${\Gamma }_{u}$相关联：

- When an element is inserted/deleted in the subtree ${\mathcal{T}}_{u}$ of $u,{\Gamma }_{u}$ can be updated in $\widetilde{O}\left( 1\right)$ amortized time.

- 当在$u,{\Gamma }_{u}$的子树${\mathcal{T}}_{u}$中插入/删除一个元素时，${\mathcal{T}}_{u}$可以在分摊时间$\widetilde{O}\left( 1\right)$内完成更新。

- ${\Gamma }_{u}$ can be reconstructed in $\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$ time.

- ${\Gamma }_{u}$可以在时间$\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$内重构。

Then, we have:

那么，我们有：

Lemma 6. $\mathcal{T}$ can updated in $\widetilde{O}\left( 1\right)$ amortized time per insertion and deletion in $\mathcal{S}$ .

引理6. 在$\mathcal{S}$中进行每次插入和删除操作时，$\mathcal{T}$可以在分摊时间$\widetilde{O}\left( 1\right)$内完成更新。

Proof. This is a corollary of the results in [2] (see also [24]).

证明：这是文献[2]中结果的一个推论（另见文献[24]）。

### 2.4 A Result from Computational Geometry

### 2.4 计算几何中的一个结果

Let $R$ be a set of $n$ rectangles in ${\mathbb{R}}^{d}$ for some constant dimensionality $d$ . Each rectangle in $R$ is associated with a weight drawn from some ordered domain. Given a query rectangle $q$ ,a range min query returns the rectangle in $R$ with the smallest weight,among all the rectangles in $R$ that intersect with $q$ . The result below is well-known:

设$R$是${\mathbb{R}}^{d}$中$n$个矩形构成的集合，其中${\mathbb{R}}^{d}$是某个固定维度$d$的空间。$R$中的每个矩形都关联着一个取自某个有序域的权重。给定一个查询矩形$q$，范围最小值查询会返回$R$中与$q$相交的所有矩形中权重最小的矩形。以下结果是众所周知的：

Lemma 7. We can store $R$ in a structure of $\widetilde{O}\left( n\right)$ space that answers any range min query in $\widetilde{O}\left( 1\right)$ time. The structure can be updated in $\widetilde{O}\left( 1\right)$ amortized time per insertion or deletion in $R$ .

引理7. 我们可以将$R$存储在一个占用$\widetilde{O}\left( n\right)$空间的结构中，该结构能在$\widetilde{O}\left( 1\right)$时间内回答任意区间最小值查询。该结构在$R$中每次插入或删除操作的均摊更新时间为$\widetilde{O}\left( 1\right)$。

Proof. Achievable in many ways; see, for example, [1].

证明. 有多种实现方式；例如，参见文献[1]。

## 3 Multi-Way Joins with $\geq  2$ Dimensions

## 3 具有$\geq  2$维的多路连接

Let us start with our negative result. In this section,we will concentrate on intersection joins on $t \geq  3$ sets ${R}_{1},\ldots ,{R}_{t}$ of rectangles in ${\mathbb{R}}^{d}$ where $d \geq  2$ . We will show that,subject to the OMV-conjecture, no feasible structures can exist.

让我们从负面结果开始。在本节中，我们将关注${\mathbb{R}}^{d}$中$t \geq  3$个矩形集合${R}_{1},\ldots ,{R}_{t}$的交集连接，其中$d \geq  2$。我们将证明，在OMV猜想成立的前提下，不存在可行的结构。

A Natural-Join Result of [3]. Consider three relations with schema ${T}_{1}\left( X\right) ,{T}_{2}\left( {X,Y}\right)$ ,and ${T}_{3}\left( Y\right)$ (each relation is under the set semantics,i.e.,no duplicate tuples). We refer to ${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$ as a wedge natural join.

文献[3]中的自然连接结果。考虑具有模式${T}_{1}\left( X\right) ,{T}_{2}\left( {X,Y}\right)$和${T}_{3}\left( Y\right)$的三个关系（每个关系采用集合语义，即无重复元组）。我们将${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$称为楔形自然连接。

Suppose that the values of attributes $X$ and $Y$ are integers in $\left\lbrack  {1,D}\right\rbrack$ ,i.e.,the domain size of each attribute is $D$ . We want to incrementally maintain a feasible structure,that is,one that supports an update in $\widetilde{O}\left( 1\right)$ time,and can be used to enumerate the result of ${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$ with a small delay $\Delta$ .

假设属性$X$和$Y$的值是$\left\lbrack  {1,D}\right\rbrack$中的整数，即每个属性的域大小为$D$。我们希望增量式地维护一个可行的结构，即一个能在$\widetilde{O}\left( 1\right)$时间内支持更新，并且可用于以较小延迟$\Delta$枚举${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$结果的结构。

Lemma 8. ([3]) For wedge natural joins, subject to the OMv-conjecture [13], no structure can offer the following guarantees simultaneously: for some constant $0 < \epsilon  < 1$ ,it (i) can be updated in $O\left( {D}^{1 - \epsilon }\right)$ time,and (ii) admits the join result to be enumerated with a delay of $O\left( {D}^{1 - \epsilon }\right)$ .

引理8. （文献[3]）对于楔形自然连接，在OMv猜想[13]成立的前提下，不存在能同时提供以下保证的结构：对于某个常数$0 < \epsilon  < 1$，(i) 它能在$O\left( {D}^{1 - \epsilon }\right)$时间内更新，并且(ii) 能以$O\left( {D}^{1 - \epsilon }\right)$的延迟枚举连接结果。

The above lemma holds regardless of the space of the structure.

上述引理的成立与结构的空间大小无关。

Hardness of the "Wedge" Topology. Consider again the topology shown in Figure 1. We will refer to the join with this topology as a wedge intersection join. Recall that it describes an intersection join on ${R}_{1},{R}_{2}$ ,and ${R}_{3}$ that returns all $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  {R}_{1} \times  {R}_{2} \times  {R}_{3}$ satisfying the conditions in (1). We will show that the existence of any feasible structures on such joins in 2D space will defy Lemma 8.

“楔形”拓扑的难度。再次考虑图1所示的拓扑。我们将具有此拓扑的连接称为楔形交集连接。回顾一下，它描述了${R}_{1},{R}_{2}$和${R}_{3}$的交集连接，该连接返回所有满足(1)中条件的$\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  {R}_{1} \times  {R}_{2} \times  {R}_{3}$。我们将证明，二维空间中此类连接存在任何可行结构都将与引理8矛盾。

Given the sets ${T}_{1}\left( X\right) ,{T}_{2}\left( {X,Y}\right)$ ,and ${T}_{3}\left( Y\right)$ participating in the wedge natural join,we construct three sets of rectangles in two-dimensional space as follows:

给定参与楔形自然连接的集合${T}_{1}\left( X\right) ,{T}_{2}\left( {X,Y}\right)$和${T}_{3}\left( Y\right)$，我们在二维空间中构造三组矩形如下：

$$
{R}_{1} = \left\{  {x \times  \left( {-\infty ,\infty }\right)  \mid  x \in  {T}_{1}}\right\}  
$$

$$
{R}_{2} = \left\{  {\left( {x,y}\right)  \mid  \left( {x,y}\right)  \in  {T}_{2}}\right\}  
$$

$$
{R}_{3} = \left\{  {\left( {-\infty ,\infty }\right)  \times  y \mid  y \in  {T}_{3}}\right\}  .
$$

Note that the rectangles in all three sets are degenerated: each rectangle in ${R}_{1}$ (or ${R}_{3}$ ) is a line perpendicular to dimension 1 (or 2,resp.),whereas each rectangle in ${R}_{2}$ is a point. Our construction guarantees:

注意，所有三组中的矩形都是退化的：${R}_{1}$（或${R}_{3}$）中的每个矩形都是垂直于维度1（或2）的直线，而${R}_{2}$中的每个矩形都是一个点。我们的构造保证：

Proposition 1. Tuples $x \in  {T}_{1},\left( {x,y}\right)  \in  {T}_{2}$ ,and $y \in  {T}_{3}$ make a pair(x,y)in the result of ${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$ if and only if $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$ is in the result of the wedge intersection join on ${R}_{1},{R}_{2}$ , and ${R}_{3}$ ,where ${r}_{1}$ is the rectangle $x \times  \left( {-\infty ,\infty }\right)$ in ${R}_{1},{r}_{2}$ is the rectangle(x,y)in ${R}_{2}$ ,and ${r}_{3}$ the rectangle $\left( {-\infty ,\infty }\right)  \times  y$ in ${R}_{3}$ .

命题1. 元组 $x \in  {T}_{1},\left( {x,y}\right)  \in  {T}_{2}$ 和 $y \in  {T}_{3}$ 在 ${T}_{1} \bowtie  {T}_{2} \bowtie  {T}_{3}$ 的结果中构成一个对 (x, y)，当且仅当 $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$ 在 ${R}_{1},{R}_{2}$ 和 ${R}_{3}$ 的楔形交集连接结果中，其中 ${r}_{1}$ 是 ${R}_{1},{r}_{2}$ 中的矩形 $x \times  \left( {-\infty ,\infty }\right)$，${R}_{1},{r}_{2}$ 是 ${R}_{2}$ 中的矩形 (x, y)，并且 ${r}_{3}$ 是 ${R}_{3}$ 中的矩形 $\left( {-\infty ,\infty }\right)  \times  y$。

Suppose that,we are given a wedge-intersection-join structure which can be updated in $U\left( n\right)$ time,and be used to enumerate the join result in $\Delta \left( n\right)$ time,where $n = \left| {R}_{1}\right|  + \left| {R}_{2}\right|  + \left| {R}_{3}\right|$ . By our construction,the sizes of ${R}_{1}$ and ${R}_{2}$ are at most $D$ ,while that of ${R}_{2}$ is at most ${D}^{2}$ . It thus follows from Lemma 8 that,subject to the OMv-conjecture, $U\left( {O\left( {D}^{2}\right) }\right)  = O\left( {D}^{1 - \epsilon }\right)$ and $\Delta \left( {O\left( {D}^{2}\right) }\right)  = O\left( {D}^{1 - \epsilon }\right)$ cannot hold at the same time. Combining this with $n \leq  {D}^{2}$ shows:

假设我们有一个楔形交集连接结构，它可以在 $U\left( n\right)$ 时间内更新，并且可以在 $\Delta \left( n\right)$ 时间内枚举连接结果，其中 $n = \left| {R}_{1}\right|  + \left| {R}_{2}\right|  + \left| {R}_{3}\right|$。根据我们的构造，${R}_{1}$ 和 ${R}_{2}$ 的大小至多为 $D$，而 ${R}_{2}$ 的大小至多为 ${D}^{2}$。因此，由引理8可知，在 OMv 猜想的条件下，$U\left( {O\left( {D}^{2}\right) }\right)  = O\left( {D}^{1 - \epsilon }\right)$ 和 $\Delta \left( {O\left( {D}^{2}\right) }\right)  = O\left( {D}^{1 - \epsilon }\right)$ 不能同时成立。将此与 $n \leq  {D}^{2}$ 相结合可得：

Lemma 9. For wedge intersection joins, subject to the OMv-conjecture [13], no structure can offer the following guarantees simultaneously: for some constant $0 < \epsilon  < 1$ ,it (i) can be updated in $O\left( {n}^{{0.5} - \epsilon }\right)$ time,and (ii) admits the join result to be enumerated with a delay of $O\left( {n}^{{0.5} - \epsilon }\right)$ .

引理9. 对于楔形交集连接，在 OMv 猜想 [13] 的条件下，不存在一种结构能够同时提供以下保证：对于某个常数 $0 < \epsilon  < 1$，(i) 它可以在 $O\left( {n}^{{0.5} - \epsilon }\right)$ 时间内更新，并且 (ii) 允许以 $O\left( {n}^{{0.5} - \epsilon }\right)$ 的延迟枚举连接结果。

Before proceeding, let us point out a property of our construction. Consider the triangle topology that has an edge between $i,j$ for all $1 \leq  i < j \leq  3$ . On the constructed ${R}_{1},{R}_{2},{R}_{3}$ ,a tuple $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$ is in the join result under the wedge topology if and only if it is in the joint result under the triangle topology. This is a crucial property that we rely on to extend the hardness result to arbitrary intersection joins with $t \geq  3$ and $d \geq  2$ ,as shown next.

在继续之前，让我们指出我们构造的一个性质。考虑对于所有 $1 \leq  i < j \leq  3$，在 $i,j$ 之间都有一条边的三角形拓扑。在构造的 ${R}_{1},{R}_{2},{R}_{3}$ 上，一个元组 $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$ 在楔形拓扑下的连接结果中，当且仅当它在三角形拓扑下的连接结果中。这是一个关键性质，我们依靠它将硬度结果扩展到具有 $t \geq  3$ 和 $d \geq  2$ 的任意交集连接，如下所示。

Hardness of $t \geq  3$ and $d \geq  2$ . Consider an arbitrary intersection join with topology $G$ with $t \geq  3$ vertices. Since $G$ is connected,it must have at least one vertex with two edges. Without loss of generality,assume that $G$ contains an edge between 1 and 2,and an edge between 2 and 3 (otherwise, rename the input sets).

$t \geq  3$ 和 $d \geq  2$ 的硬度。考虑一个具有拓扑 $G$ 且有 $t \geq  3$ 个顶点的任意交集连接。由于 $G$ 是连通的，它必须至少有一个顶点有两条边。不失一般性，假设 $G$ 包含 1 和 2 之间的一条边，以及 2 和 3 之间的一条边（否则，对输入集重新命名）。

Suppose that we are given a feasible structure for $G$ in two-dimensional space. We can use the structure to maintain the result of the $2\mathrm{D}$ wedge intersection join on ${R}_{1},{R}_{2},{R}_{3}$ that we constructed earlier. For this purpose,we add $t - 3$ dummy input sets ${R}_{4},{R}_{5},\ldots ,{R}_{t}$ ,each of which contains only a single rectangle,which is simply ${\mathbb{R}}^{2}$ (i.e.,the whole data space). The dummy input sets are never updated. It is clear that $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$ is in the result of the wedge intersection join if and only if

假设我们在二维空间中得到了一个关于$G$的可行结构。我们可以利用该结构来维护我们之前构建的关于${R}_{1},{R}_{2},{R}_{3}$的$2\mathrm{D}$楔形交集连接的结果。为此，我们添加$t - 3$个虚拟输入集${R}_{4},{R}_{5},\ldots ,{R}_{t}$，每个虚拟输入集仅包含一个矩形，即${\mathbb{R}}^{2}$（也就是整个数据空间）。这些虚拟输入集永远不会被更新。显然，当且仅当

$$
\left( {{r}_{1},{r}_{2},{r}_{3},\underset{t - 3}{\underbrace{{\mathbb{R}}^{2},\ldots ,{\mathbb{R}}^{2}}}}\right) 
$$

is in the result of the (constructed) $t$ -way intersection join with topology $G$ . Note that this is true no matter whether $G$ has an edge between 2 and 3,due to the property mentioned below Lemma 9 .

在具有拓扑结构$G$的（已构建的）$t$路交集连接的结果中时，[latex6]才在楔形交集连接的结果中。请注意，由于引理9下面提到的性质，无论$G$在2和3之间是否有边，这都是成立的。

Finally,the absence of efficient structures when $t \geq  3$ and $d = 2$ implies the same when $t \geq  3$ and $d > 2$ (by adding dummy dimensions). We now conclude the proof of Theorem 3.

最后，当$t \geq  3$和$d = 2$时缺乏高效结构意味着当$t \geq  3$和$d > 2$时（通过添加虚拟维度）同样缺乏高效结构。现在我们完成定理3的证明。

## 4 Binary Joins

## 4 二元连接

Having explained what cannot be done, in the rest of the paper we will focus on what can be done. We will prove the existence of feasible structures in all the other scenarios,namely,either $t = 2$ (binary joins) or $d = 1$ (1D joins).

在解释了哪些事情无法做到之后，在本文的其余部分，我们将专注于哪些事情可以做到。我们将证明在所有其他场景中存在可行结构，即要么是$t = 2$（二元连接），要么是$d = 1$（一维连接）。

This section will focus on $t = 2$ . First,Siest,Section 4.1 will present the structure of Theorem 1, which as mentioned solves 1D binary joins optimally in the comparison model. Then, Section 4.2 will extend our solutions to constant dimensionalities $d \geq  2$ .

本节将专注于$t = 2$。首先，在4.1节中，我们将介绍定理1的结构，如前所述，该结构在比较模型中能最优地解决一维二元连接问题。然后，4.2节将把我们的解决方案扩展到常数维度$d \geq  2$。

### 4.1 An Optimal 1D Structure

### 4.1 最优的一维结构

#### 4.1.1 Interval-Point Joins

#### 4.1.1 区间 - 点连接

In the 1D version, ${R}_{1}$ and ${R}_{2}$ are two sets of intervals in $\mathbb{R}$ . We will first deal with a special instance of the problem where every interval of ${R}_{2}$ degenerates into a real value,i.e.,a point. For clarity,let us denote ${R}_{1}$ simply as $R$ ,and use $P$ to represent the set of values in ${R}_{2}$ . The join result can now be re-defined in a simpler manner: it reports all $\left( {r,p}\right)  \in  R \times  P$ satisfying $p \in  r$ . We will refer to this as the interval-point join.

在一维版本中，${R}_{1}$和${R}_{2}$是$\mathbb{R}$中的两个区间集合。我们将首先处理该问题的一个特殊情况，即${R}_{2}$中的每个区间都退化为一个实数值，也就是一个点。为了清晰起见，我们将${R}_{1}$简单地表示为$R$，并使用$P$来表示${R}_{2}$中的值的集合。现在可以以更简单的方式重新定义连接结果：它报告所有满足$p \in  r$的$\left( {r,p}\right)  \in  R \times  P$。我们将此称为区间 - 点连接。

Structure. Build an interval tree $\mathcal{T}$ on $R \cup  P$ ,regarding each point in $P$ as a degenerated interval. As defined in Section 2.2,each node $u$ of $\mathcal{T}$ is associated with a stabbing set $\operatorname{stab}\left( u\right)$ . Denote by $R\left( u\right)$ the set of intervals in $\operatorname{stab}\left( u\right)$ from $R$ . Sort $R\left( u\right)$ in two separate lists: the first by left endpoint and the other by right endpoint.

结构。在$R \cup  P$上构建一个区间树$\mathcal{T}$，将$P$中的每个点视为一个退化的区间。如2.2节所定义，$\mathcal{T}$的每个节点$u$都与一个穿刺集$\operatorname{stab}\left( u\right)$相关联。用$R\left( u\right)$表示$\operatorname{stab}\left( u\right)$中来自$R$的区间集合。将$R\left( u\right)$分别按左端点和右端点排序成两个列表。

At each internal node $u$ ,keep:

在每个内部节点$u$处，保存：

- A left pilot value,which is the largest point of $P$ stored at a leaf in the left subtree of $u$ ;

- 一个左引导值，它是存储在$u$的左子树的叶子节点中的$P$的最大点；

<!-- Media -->

<!-- figureText: ${u}_{15}$ ${u}_{14}$ ${u}_{11}$ ${u}_{12}$ ${u}_{5}$ ${u}_{6}$ ${u}_{7}$ $u\mathrm{s}$ 。 10 16 ${u}_{13}$ ${u}_{9}$ ${u}_{10}$ ${u}_{3}$ 6 -->

<img src="https://cdn.noedgeai.com/0195ccbc-b4ca-70cc-a106-44321046dfa5_10.jpg?x=546&y=197&w=691&h=567&r=0"/>

Figure 3: Illustration of the interval-point-join structure

图3：区间 - 点连接结构的图示

<!-- Media -->

- A right pilot value,which is the smallest point of $P$ stored at a leaf in the right subtree of $u$ .

- 一个右引导值，它是存储在$u$的右子树的叶子节点中的$P$的最小点。

We say that an internal node $u$ is productive if $R\left( u\right)$ has at least one interval $r$ that covers at least one point in $P$ . Whether $u$ is productive can be decided in constant time as follows:

我们称一个内部节点$u$是有产出的，如果$R\left( u\right)$中至少有一个区间$r$覆盖了$P$中的至少一个点。可以按如下方式在常数时间内判断$u$是否有产出：

- Find the interval $\left\lbrack  {x,y}\right\rbrack   \in  R\left( u\right)$ with the smallest left endpoint $x$ . Decide $u$ as productive,if $x$ is no greater than the left pilot value of $u$ .

- 找到左端点$x$最小的区间$\left\lbrack  {x,y}\right\rbrack   \in  R\left( u\right)$。如果$x$不大于$u$的左引导值，则判定$u$是有产出的。

- Otherwise,find the interval $\left\lbrack  {x,y}\right\rbrack   \in  R\left( u\right)$ with the largest right endpoint $y$ . Decide $u$ as productive,if $y$ is no less than the right pilot value of $u$ .

- 否则，找到右端点 $y$ 最大的区间 $\left\lbrack  {x,y}\right\rbrack   \in  R\left( u\right)$。如果 $y$ 不小于 $u$ 的右引导值，则判定 $u$ 为有效节点。

- Otherwise,decide $u$ as non-productive.

- 否则，判定 $u$ 为无效节点。

We link up all the productive nodes with a doubly linked list $\mathcal{L}$ (ordering does not matter),referred to as the productive list.

我们用一个双向链表 $\mathcal{L}$ 将所有有效节点连接起来（顺序无关紧要），这个链表称为有效列表。

Finally,we keep $P$ in a sorted list ${\sum }_{P}$ (managed by a BST). Recall that each value $p \in  P$ is stored at a leaf $u$ in $\mathcal{T}$ . We keep a cross pointer from $u$ to the position of $p$ in ${\sum }_{P}$ . Remember that $p$ may also be stored as a pilot value at several internal nodes ${u}^{\prime }$ in $\mathcal{T}$ as well. We also keep a cross pointer from each such ${u}^{\prime }$ to the position of $p$ in ${\sum }_{P}$ .

最后，我们将 $P$ 保存在一个有序列表 ${\sum }_{P}$ 中（由二叉搜索树管理）。回想一下，每个值 $p \in  P$ 都存储在 $\mathcal{T}$ 中的一个叶子节点 $u$ 处。我们维护一个交叉指针，从 $u$ 指向 $p$ 在 ${\sum }_{P}$ 中的位置。请记住，$p$ 也可能作为引导值存储在 $\mathcal{T}$ 中的几个内部节点 ${u}^{\prime }$ 处。我们还维护一个交叉指针，从每个这样的 ${u}^{\prime }$ 指向 $p$ 在 ${\sum }_{P}$ 中的位置。

The overall space consumption of our structure is clearly $O\left( n\right)$ .

我们的结构的总体空间消耗显然是 $O\left( n\right)$。

Example: Figure 3 shows a BST created on $R = \{ \left\lbrack  {3,7}\right\rbrack  ,\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  \}$ ,and $P = \{ 1,2$ , ${10},{13},{14},{16}\}$ .

示例：图 3 展示了在 $R = \{ \left\lbrack  {3,7}\right\rbrack  ,\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  \}$、$P = \{ 1,2$ 和 ${10},{13},{14},{16}\}$ 上创建的一个二叉搜索树。

For the root ${u}_{15},R\left( {u}_{15}\right)  = \{ \left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  \}$ . It has a left pilot value 2,and a right pilot value 10. It is productive because $\left\lbrack  {4,{12}}\right\rbrack   \in  R\left( {u}_{15}\right)$ covers a value in $P$ . On the other hand, node ${u}_{13}$ - with $\operatorname{key}\left( {u}_{13}\right)  = 5,R\left( {u}_{13}\right)  = \{ \left\lbrack  {3,7}\right\rbrack  \}$ ,left pilot value 2,and no right pilot value - is non-productive.

对于根节点 ${u}_{15},R\left( {u}_{15}\right)  = \{ \left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack  \}$，它的左引导值为 2，右引导值为 10。它是有效节点，因为 $\left\lbrack  {4,{12}}\right\rbrack   \in  R\left( {u}_{15}\right)$ 覆盖了 $P$ 中的一个值。另一方面，节点 ${u}_{13}$（其 $\operatorname{key}\left( {u}_{13}\right)  = 5,R\left( {u}_{13}\right)  = \{ \left\lbrack  {3,7}\right\rbrack  \}$，左引导值为 2，没有右引导值）是无效节点。

Point 10 is stored as the right pilot value of ${u}_{15},{u}_{5}$ ,and as the left pilot value of ${u}_{14},{u}_{11}$ . Hence, each of ${u}_{15},{u}_{14},{u}_{11}$ ,and ${u}_{5}$ keeps a cross pointer to the position of 10 in ${\sum }_{P}$ (the sorted list of $P)$ .

点 10 作为 ${u}_{15},{u}_{5}$ 的右引导值和 ${u}_{14},{u}_{11}$ 的左引导值存储。因此，${u}_{15},{u}_{14},{u}_{11}$ 和 ${u}_{5}$ 中的每一个都维护一个交叉指针，指向 10 在 ${\sum }_{P}$（$P)$ 的有序列表）中的位置。

Join Result Enumeration. If $\mathcal{L}$ is empty,we finish immediately,declaring that the join result is empty. The time in this case is constant.

连接结果枚举。如果 $\mathcal{L}$ 为空，我们立即结束，并声明连接结果为空。这种情况下的时间复杂度是常数级的。

<!-- Media -->

---

enumerate(u)

枚举(u)

1. ${p}_{\text{left }} \leftarrow$ the left pilot value at $u$

1. ${p}_{\text{left }} \leftarrow$ 是 $u$ 处的左引导值

2. ${\sum }_{\text{left }} \leftarrow$ the list that sorts $R\left( u\right)$ in ascending order of

2. ${\sum }_{\text{left }} \leftarrow$ 是按以下顺序对 $R\left( u\right)$ 进行升序排序的列表

		left endpoint

		左端点

	$r \leftarrow$ the first interval in ${\sum }_{\text{left }}$

	$r \leftarrow$ 是 ${\sum }_{\text{left }}$ 中的第一个区间

		repeat

		重复

5. if ${p}_{\text{left }} \in  r$ then

5. 如果 ${p}_{\text{left }} \in  r$ 则

6. report all such result pairs(r,p)that are produced by $r$

6. 报告由$r$生成的所有此类结果对(r, p)

					and a point $p$ in the left subtree of $u$

					 并且是$u$左子树中的一个点$p$

					/* use ${\sum }_{P}$ for this purpose; see main texts */

					 /* 为此目的使用${\sum }_{P}$；参见正文 */

					$r \leftarrow$ the next interval in ${\sum }_{\text{left }}$

					 $r \leftarrow$ ${\sum }_{\text{left }}$中的下一个区间

8. else break

8. 否则跳出

		until $r =$ null,i.e., ${\sum }_{\text{left }}$ has been exhausted

		 直到$r =$为空，即${\sum }_{\text{left }}$已遍历完

	10. ${p}_{\text{right }} \leftarrow$ the right pilot value at $u$

	 10. ${p}_{\text{right }} \leftarrow$ $u$处的右引导值

11. ${\sum }_{\text{right }} \leftarrow$ the list that sorts $R\left( u\right)$ in descending order of

11. ${\sum }_{\text{right }} \leftarrow$ 按$R\left( u\right)$的

		right endpoint

		 右端点降序排序的列表

12. $r \leftarrow$ the first interval in ${\sum }_{\text{right }}$

12. $r \leftarrow$ ${\sum }_{\text{right }}$中的第一个区间

13. repeat

13. 重复

			if ${p}_{\text{right }} \in  r$ then

			 如果${p}_{\text{right }} \in  r$则

15. report all such result pairs(r,p)that are produced by $r$

15. 报告由$r$生成的所有此类结果对(r, p)

					and a point $p$ in the right subtree of $u$

					 并且是$u$右子树中的一个点$p$

					/* use ${\sum }_{P}$ for this purpose; see main texts */

					 /* 为此目的使用${\sum }_{P}$；参见正文 */

16. $r \leftarrow$ the next interval in ${\sum }_{\text{right }}$

16. $r \leftarrow$ ${\sum }_{\text{right }}$中的下一个区间

			else return

			否则返回

18. until $r =$ null

18. 直到 $r =$ 为空

---

<!-- Media -->

## Figure 4: Enumerating the result pairs at a productive node

## 图4：枚举有效节点处的结果对

Otherwise,for each productive node $u \in  \mathcal{L}$ ,we use the algorithm in Figure 4 to report all result pairs(r,p)satisfying $r \in  R\left( u\right)$ . The algorithm does so with a constant delay (as will be explained shortly) and guarantee outputting at least one pair (by definition of productive node). Because all the productive nodes have been explicitly stored in $\mathcal{L}$ ,we can run the algorithm on every node in $\mathcal{L}$ to report the entire query result with a constant delay. No result pair(r,p)can be missed because the interval $r$ must reside in the $R\left( u\right)$ of exactly one productive node $u$ .

否则，对于每个有效节点 $u \in  \mathcal{L}$，我们使用图4中的算法来报告所有满足 $r \in  R\left( u\right)$ 的结果对 (r, p)。该算法以恒定延迟完成此操作（稍后将进行解释），并保证至少输出一对（根据有效节点的定义）。因为所有有效节点都已显式存储在 $\mathcal{L}$ 中，所以我们可以对 $\mathcal{L}$ 中的每个节点运行该算法，以恒定延迟报告整个查询结果。不会遗漏任何结果对 (r, p)，因为区间 $r$ 必定恰好位于一个有效节点 $u$ 的 $R\left( u\right)$ 中。

The algorithm of Figure 4 has two parts: (i) Lines 2-9,which find such(r,p)where $p$ is in the left subtree of $u$ ,and (ii) Lines 10-18,which find such(r,p)where $p$ is in the right subtree of $u$ . Due to symmetry, we will discuss only the first part.

图4中的算法有两部分：(i) 第2 - 9行，用于查找 $p$ 在 $u$ 的左子树中的 (r, p)；(ii) 第10 - 18行，用于查找 $p$ 在 $u$ 的右子树中的 (r, p)。由于对称性，我们仅讨论第一部分。

After obtaining at Line 1 the left pilot value ${p}_{\text{left }}$ at $u$ ,the algorithm (Lines 2-9) processes the intervals of $R\left( u\right)$ in ascending order of left endpoint. To explain how,let $r \in  R\left( u\right)$ be the interval being processed. If $r$ does not cover ${p}_{\text{left }}$ ,we are sure that,for any ${r}^{\prime } \in  R\left( u\right)$ that has not been processed yet (that is,the left endpoint of ${r}^{\prime }$ is greater than that of $r$ ), ${r}^{\prime }$ cannot make a result pair with ${p}_{\text{left }}$ and - due to the definition of ${p}_{\text{left }} -$ cannot make a result pair with any point in the left subtree of $u$ . In this case,we move on to the second part of the algorithm starting at Line 10 .

在第1行获得 $u$ 处的左引导值 ${p}_{\text{left }}$ 后，算法（第2 - 9行）按左端点升序处理 $R\left( u\right)$ 的区间。为了解释具体过程，设 $r \in  R\left( u\right)$ 为正在处理的区间。如果 $r$ 不覆盖 ${p}_{\text{left }}$，我们可以确定，对于任何尚未处理的 ${r}^{\prime } \in  R\left( u\right)$（即 ${r}^{\prime }$ 的左端点大于 $r$ 的左端点），${r}^{\prime }$ 不能与 ${p}_{\text{left }}$ 构成结果对，并且根据 ${p}_{\text{left }} -$ 的定义，它也不能与 $u$ 的左子树中的任何点构成结果对。在这种情况下，我们转到从第10行开始的算法的第二部分。

If,on the other hand, $r$ does cover ${p}_{\text{left }}$ ,we (at Line 6) find all those points $p$ that (i) are in the left subtree of $u$ and (ii) are covered by $r$ . Every such $p$ makes a result pair with $r$ . A constant delay can be ensured by resorting to the sorted list ${\sum }_{p}$ . First,use a cross pointer stored at $u$ to find the position of ${p}_{\text{left }}$ in ${\sum }_{p}$ . Then,scan ${\sum }_{p}$ from ${p}_{\text{left }}$ in descending order until seeing the first point that falls outside $r$ .

另一方面，如果 $r$ 确实覆盖 ${p}_{\text{left }}$，我们（在第6行）找到所有满足以下条件的点 $p$：(i) 位于 $u$ 的左子树中；(ii) 被 $r$ 覆盖。每个这样的 $p$ 都与 $r$ 构成一个结果对。通过借助已排序列表 ${\sum }_{p}$ 可以确保恒定延迟。首先，使用存储在 $u$ 处的交叉指针在 ${\sum }_{p}$ 中找到 ${p}_{\text{left }}$ 的位置。然后，从 ${p}_{\text{left }}$ 开始按降序扫描 ${\sum }_{p}$，直到看到第一个落在 $r$ 之外的点。

Example. Let us illustrate the algorithm using Figure 3. Node ${u}_{15}$ is the only productive node. At Line $1,{p}_{\text{left }} = 2$ . Then,we scan $R\left( {u}_{15}\right)$ in this order: $\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack$ ,starting at Line 3 with $r = \left\lbrack  {4,{12}}\right\rbrack$ . At Line 4,we find that $r$ does not cover ${p}_{\text{left }}$ . The scan is therefore aborted; and the execution jumps to Line 10 .

示例。让我们使用图3来说明该算法。节点 ${u}_{15}$ 是唯一的有效节点。在第 $1,{p}_{\text{left }} = 2$ 行。然后，我们按此顺序扫描 $R\left( {u}_{15}\right)$：$\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {5,9}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {8,{15}}\right\rbrack$，从第3行开始处理 $r = \left\lbrack  {4,{12}}\right\rbrack$。在第4行，我们发现 $r$ 不覆盖 ${p}_{\text{left }}$。因此扫描中止；执行跳转到第10行。

After setting ${p}_{\text{right }} = {10}$ at Line 10,we scan $R\left( {u}_{15}\right)$ in this order: $\left\lbrack  {8,{15}}\right\rbrack  ,\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {6,9}\right\rbrack$ , starting with $r = \left\lbrack  {8,{15}}\right\rbrack$ (Line 12). After seeing at Line 14 that $r$ contains ${p}_{\text{right }}$ ,we extract all the points $p \in  P$ such that $p \geq  {p}_{\text{right }}$ and $p$ is covered by $r$ . There are 3 such points: 10,13,14. They can be found by scanning ${\sum }_{P}$ in ascending order from ${10} = {p}_{\text{right }}$ .

在第10行设置${p}_{\text{right }} = {10}$之后，我们按此顺序扫描$R\left( {u}_{15}\right)$：$\left\lbrack  {8,{15}}\right\rbrack  ,\left\lbrack  {4,{12}}\right\rbrack  ,\left\lbrack  {6,{11}}\right\rbrack  ,\left\lbrack  {6,9}\right\rbrack$，从$r = \left\lbrack  {8,{15}}\right\rbrack$开始（第12行）。在第14行看到$r$包含${p}_{\text{right }}$之后，我们提取所有满足$p \geq  {p}_{\text{right }}$且$p$被$r$覆盖的点$p \in  P$。有3个这样的点：10、13、14。可以通过从${10} = {p}_{\text{right }}$开始按升序扫描${\sum }_{P}$来找到它们。

Next, $r$ is moved to the next interval $\left\lbrack  {4,{12}}\right\rbrack$ in $R\left( {u}_{15}\right)$ ,and processed in the same fashion. The rest of the execution is omitted.

接下来，将$r$移动到$R\left( {u}_{15}\right)$中的下一个区间$\left\lbrack  {4,{12}}\right\rbrack$，并以相同的方式进行处理。其余的执行过程省略。

Update. Thanks to Lemma 5, it becomes much easier to explain why our structure can be updated in $O\left( {\log n}\right)$ amortized time per insertion and deletion.

更新。多亏引理5，解释为什么我们的结构可以在每次插入和删除操作的均摊时间$O\left( {\log n}\right)$内进行更新变得容易得多。

Given a node $u$ of $\mathcal{T}$ ,regard the following together as its secondary structure ${\Gamma }_{u}$ :

给定$\mathcal{T}$的一个节点$u$，将以下内容共同视为其二级结构${\Gamma }_{u}$：

- The two sorted lists of $R\left( u\right)$ ,i.e.,one sorted by left endpoint,and the other by right endpoint;

- $R\left( u\right)$的两个排序列表，即一个按左端点排序，另一个按右端点排序；

- (Only if $u$ is an internal node) its left and right pilot values,and the cross pointers associated with those values;

- （仅当$u$是内部节点时）其左右引导值，以及与这些值关联的交叉指针；

- (Only if $u$ is a leaf node and stores a point $p \in  P$ ) the cross pointer associated with $p$ .

- （仅当$u$是叶节点并存储一个点$p \in  P$时）与$p$关联的交叉指针。

For an internal node $u$ ,its pilot values (and the cross pointers) can be obtained from its child nodes ${v}_{1},{v}_{2}$ in $O\left( 1\right)$ time,assuming that ${\Gamma }_{{v}_{1}},{\Gamma }_{{v}_{2}}$ are both ready. Thus,with respect to the properties ${P1} - {P4}$ prescribed in Section 2.3.1,it is straightforward to achieve: ${f}_{1}\left( n\right)  = {f}_{3}\left( n\right)  = O\left( 1\right)$ , ${f}_{2}\left( n\right)  = O\left( {\log n}\right)$ ,and ${f}_{4}\left( \left| {\mathcal{T}}_{u}\right| \right)  = O\left( \left| {\mathcal{T}}_{u}\right| \right)$ . By Lemma 5, $\mathcal{T}$ (augmented with ${\Gamma }_{u}$ ) can be maintained in $O\left( {\log n}\right)$ amortized time per update.

对于一个内部节点$u$，假设${\Gamma }_{{v}_{1}},{\Gamma }_{{v}_{2}}$都已就绪，其引导值（和交叉指针）可以在时间$O\left( 1\right)$内从其子节点${v}_{1},{v}_{2}$获得。因此，关于第2.3.1节规定的属性${P1} - {P4}$，很容易实现：${f}_{1}\left( n\right)  = {f}_{3}\left( n\right)  = O\left( 1\right)$、${f}_{2}\left( n\right)  = O\left( {\log n}\right)$和${f}_{4}\left( \left| {\mathcal{T}}_{u}\right| \right)  = O\left( \left| {\mathcal{T}}_{u}\right| \right)$。根据引理5，（用${\Gamma }_{u}$扩充后的）$\mathcal{T}$可以在每次更新的均摊时间$O\left( {\log n}\right)$内维护。

It remains to explain how to modify the productive list $\mathcal{L}$ . This can be "piggybacked" on the updates on $\mathcal{T}$ . In general,whenever the secondary structure ${\Gamma }_{u}$ of a node $u$ is affected by an update, one can spend $O\left( 1\right)$ time to determine the current productive status of $u$ ,and insert/delete $u$ in $\mathcal{L}$ (remember that the ordering in $\mathcal{L}$ does not matter). Therefore,the maintenance of $\mathcal{L}$ cannot be more expensive than maintaining $\mathcal{T}$ .

还需要解释如何修改有效列表$\mathcal{L}$。这可以“搭载”在对$\mathcal{T}$的更新上。一般来说，只要节点$u$的二级结构${\Gamma }_{u}$受到更新的影响，就可以花费时间$O\left( 1\right)$来确定$u$的当前有效状态，并在$\mathcal{L}$中插入/删除$u$（记住$\mathcal{L}$中的顺序无关紧要）。因此，维护$\mathcal{L}$的成本不会高于维护$\mathcal{T}$。

#### 4.1.2 Intersection Joins

#### 4.1.2 交集连接

We now return to the intersection join problem with $d = 1$ and $t = 2$ where the inputs are two sets ${R}_{1}$ and ${R}_{2}$ of intervals.

我们现在回到关于$d = 1$和$t = 2$的交集连接问题，其中输入是两个区间集合${R}_{1}$和${R}_{2}$。

For two intervals ${r}_{1}$ and ${r}_{2}$ ,if ${r}_{1} \cap  {r}_{2} \neq  \varnothing$ ,then either ${r}_{1}$ covers at least an endpoint of ${r}_{2}$ , or ${r}_{2}$ covers at least an endpoint of ${r}_{1}$ . This suggests that the problem can be reduced to four interval-point joins. Specifically,the first (or second) interval-point join sets $R$ to ${R}_{1}$ and $P$ to the set of left (or right,resp.) endpoints of the intervals in ${R}_{2}$ ,while the other two interval-point joins are defined analogously by reversing the roles of ${R}_{1},{R}_{2}$ . Each pair $\left( {{r}_{1},{r}_{2}}\right)$ in the result of the original intersection join is output by at least one interval-point join.

对于两个区间${r}_{1}$和${r}_{2}$，如果${r}_{1} \cap  {r}_{2} \neq  \varnothing$，那么要么${r}_{1}$覆盖${r}_{2}$的至少一个端点，要么${r}_{2}$覆盖${r}_{1}$的至少一个端点。这表明该问题可以简化为四个区间 - 点连接。具体来说，第一个（或第二个）区间 - 点连接将$R$设为${R}_{1}$，并将$P$设为${R}_{2}$中区间的左（或右）端点集合，而另外两个区间 - 点连接通过交换${R}_{1},{R}_{2}$的角色类似地定义。原始交集连接结果中的每一对$\left( {{r}_{1},{r}_{2}}\right)$至少由一个区间 - 点连接输出。

We, therefore, maintain four structures of Section 4.1.1, one for each interval-point join. The space consumption and the update cost apparently remain as $O\left( n\right)$ and $O\left( {\log n}\right)$ amortized,respectively.

因此，我们维护第4.1.1节中的四种结构，每种结构对应一个区间 - 点连接。显然，空间消耗和更新成本分别保持为摊还的$O\left( n\right)$和$O\left( {\log n}\right)$。

To obtain the result of the intersection join, we enumerate the result of each of the four interval-point structures in tandem. However, two issues arise:

为了获得交集连接的结果，我们依次枚举四个区间 - 点结构中每个结构的结果。然而，出现了两个问题：

- How to avoid reporting the same pair twice?

- 如何避免重复报告同一对？

- How to ensure a constant delay? Note that even though enumerating the result of each interval-point join guarantees a constant delay, it does not directly imply a constant delay on the intersection join. The reason is that result pairs from an interval-point join may have already been found by an earlier interval-point join. In the worst case, the result pairs of an interval-point join may have all been found, thus forcing its enumeration algorithm to incur a long delay without reporting any new pairs.

- 如何确保恒定延迟？请注意，即使枚举每个区间 - 点连接的结果能保证恒定延迟，但这并不直接意味着交集连接也有恒定延迟。原因是一个区间 - 点连接的结果对可能已经被更早的区间 - 点连接找到。在最坏的情况下，一个区间 - 点连接的结果对可能都已被找到，从而迫使它的枚举算法在不报告任何新对的情况下产生很长的延迟。

For the first issue, it suffices to adopt a consistent policy regarding which interval-point join should report a pair $\left( {{r}_{1},{r}_{2}}\right)$ . For example,suppose that ${r}_{1}$ covers both endpoints of ${r}_{2}$ . We may follow the policy that in this case only the first interval-point join (i.e., $R = {R}_{1}$ and $P$ includes the left endpoints of the intervals in ${R}_{2}$ ) should report it. The pair is simply ignored when discovered by another interval-point join.

对于第一个问题，采用一种一致的策略来确定哪个区间 - 点连接应该报告一对$\left( {{r}_{1},{r}_{2}}\right)$就足够了。例如，假设${r}_{1}$覆盖${r}_{2}$的两个端点。我们可以遵循这样的策略：在这种情况下，只有第一个区间 - 点连接（即$R = {R}_{1}$且$P$包含${R}_{2}$中区间的左端点）应该报告它。当另一个区间 - 点连接发现该对时，直接忽略它。

To resolve the second issue, we introduce a buffering technique. We actually aim at achieving a more general purpose, which has been briefly described in Section 1.3. Formally, suppose that we are given an algorithm $A$ that does not guarantee a short delay in enumerating the join result,but has the following $\alpha$ -aggressive property:

为了解决第二个问题，我们引入一种缓冲技术。实际上，我们的目标是实现一个更通用的目的，这在第1.3节中已简要描述。形式上，假设我们有一个算法$A$，它在枚举连接结果时不能保证短延迟，但具有以下$\alpha$ - 激进性质：

For any integer $x \geq  1$ ,after running for an $x$ amount of time, $A$ definitely has found $\lfloor x/\alpha \rfloor$ distinct result tuples.

对于任何整数$x \geq  1$，在运行$x$的时间后，$A$肯定已经找到了$\lfloor x/\alpha \rfloor$个不同的结果元组。

We remind the reader that, in the RAM model, the running time is defined as the number of atomic operations (i.e., operations each taking one unit of time, e.g., addition, multiplication, comparison, accessing a memory word,etc.) performed. The above property essentially says that $A$ must have found $\lfloor x/\alpha \rfloor$ result tuples after $x$ atomic operations,for all $x \geq  1$ .

我们提醒读者，在随机存取机（RAM）模型中，运行时间定义为执行的原子操作（即每个操作需要一个时间单位，例如加法、乘法、比较、访问一个内存字等）的数量。上述性质本质上表明，对于所有的$x \geq  1$，在$x$次原子操作后，$A$必须已经找到了$\lfloor x/\alpha \rfloor$个结果元组。

Our buffering technique ensures:

我们的缓冲技术确保：

Lemma 10. Given an $\alpha$ -aggressive algorithm $A$ for a join,we can design an algorithm with a delay of at most $\alpha$ .

引理10。给定一个用于连接的$\alpha$ - 激进算法$A$，我们可以设计一个延迟至多为$\alpha$的算法。

Proof. We run $A$ with a buffer,which includes all the result pairs that have been found,but not yet reported. Divide the overall execution $A$ into epochs,each consisting of $\alpha$ atomic operations. We keep counting the number of atomic operations performed, and report a pair from the buffer at the end of each epoch.

证明。我们使用一个缓冲区运行$A$，该缓冲区包含所有已找到但尚未报告的结果对。将整个执行过程$A$划分为多个时期，每个时期由$\alpha$次原子操作组成。我们持续计算执行的原子操作数量，并在每个时期结束时从缓冲区报告一对。

To prove that the above strategy works, we need to show that the buffer is never empty at the end of each epoch. Consider the end of the $i$ -th epoch $\left( {i \geq  1}\right)$ . By $\alpha$ -aggressiveness, $A$ must have found at least $\alpha  \cdot  i/\alpha  = i$ distinct result pairs. This completes the proof.

为了证明上述策略有效，我们需要证明在每个时期结束时缓冲区永远不会为空。考虑第$i$个时期$\left( {i \geq  1}\right)$的结束。根据$\alpha$ - 激进性，$A$必须已经找到了至少$\alpha  \cdot  i/\alpha  = i$个不同的结果对。证明完毕。

Let us now go back to our algorithm that runs the four interval-point joins sequentially. As mentioned,each interval-point join ensures a delay $\Delta  = O\left( 1\right)$ ; and the four interval-point joins perform in total at most $c\left( {1 + k}\right)$ atomic operations,for some constant $c \geq  \Delta$ . Next,we argue that this algorithm is ${8c}$ -aggressive. This,together with Lemma 10,will complete the proof of Theorem 1.

现在让我们回到依次执行四次区间 - 点连接的算法。如前所述，每次区间 - 点连接确保延迟为 $\Delta  = O\left( 1\right)$；并且对于某个常数 $c \geq  \Delta$，四次区间 - 点连接总共最多执行 $c\left( {1 + k}\right)$ 次原子操作。接下来，我们证明该算法是 ${8c}$ - 激进的。结合引理 10，这将完成定理 1 的证明。

Suppose that there exists an integer $x \geq  1$ such that,after $x$ atomic operations,the algorithm has found less than $\lfloor x/\left( {8c}\right) \rfloor$ result pairs. As each pair can be reported at most 4 times,strictly less than $x/\left( {2c}\right)$ result pairs - counting duplicate ones - have been found. Thus,the delay before at least one pair must be strictly larger than $\frac{x}{x/{2c}} = {2c}$ . However,since all interval-point joins ensure a delay at most $\Delta$ ,our in-tandem algorithm should find a (new or duplicated) pair with a delay at most ${2\Delta } \leq  {2c}$ ,thus creating a contradiction.

假设存在一个整数 $x \geq  1$，使得在执行 $x$ 次原子操作后，该算法找到的结果对少于 $\lfloor x/\left( {8c}\right) \rfloor$ 个。由于每个结果对最多被报告 4 次，那么严格少于 $x/\left( {2c}\right)$ 个结果对（包括重复的）已被找到。因此，至少有一对结果的延迟必须严格大于 $\frac{x}{x/{2c}} = {2c}$。然而，由于所有区间 - 点连接确保的延迟最多为 $\Delta$，我们的串联算法应该以最多 ${2\Delta } \leq  {2c}$ 的延迟找到一个（新的或重复的）结果对，从而产生矛盾。

### 4.2 A Multi-Dimensional Structure

### 4.2 多维结构

Next,we discuss intersection joins on $t = 2$ sets ${R}_{1},{R}_{2}$ of rectangles in ${\mathbb{R}}^{d}$ with a constant dimensionality $d$ .

接下来，我们讨论在具有恒定维度 $d$ 的 ${\mathbb{R}}^{d}$ 中，$t = 2$ 个矩形集合 ${R}_{1},{R}_{2}$ 的交集连接。

#### 4.2.1 Dominance Joins

#### 4.2.1 支配连接

We say that a rectangle $r$ is $d$ -sided if it has the form $\left( {-\infty ,{x}_{1}}\right\rbrack   \times  \left( {-\infty ,{x}_{2}}\right\rbrack   \times  \ldots  \times  \left( {-\infty ,{x}_{d}}\right\rbrack$ . This section focuses on a special instance of the problem - referred to as dominance join - where all the rectangles in ${R}_{1}$ are $d$ -sided,and all the rectangles in ${R}_{2}$ degenerate into points. For convenience, we rename ${R}_{1}$ as $R$ ,and ${R}_{2}$ as $P$ ; the join result contains all $\left( {r,p}\right)  \in  R \times  P$ where $r$ contains $p$ .

我们称一个矩形 $r$ 是 $d$ 边的，如果它具有 $\left( {-\infty ,{x}_{1}}\right\rbrack   \times  \left( {-\infty ,{x}_{2}}\right\rbrack   \times  \ldots  \times  \left( {-\infty ,{x}_{d}}\right\rbrack$ 的形式。本节重点讨论该问题的一个特殊实例——称为支配连接，其中 ${R}_{1}$ 中的所有矩形都是 $d$ 边的，并且 ${R}_{2}$ 中的所有矩形都退化为点。为方便起见，我们将 ${R}_{1}$ 重命名为 $R$，将 ${R}_{2}$ 重命名为 $P$；连接结果包含所有满足 $r$ 包含 $p$ 的 $\left( {r,p}\right)  \in  R \times  P$。

Set $n = \left| R\right|  + \left| P\right|$ . We will design a feasible structure with a recursive approach. The base case is $d = 1$ and has been resolved in Theorem 1 (particularly,Section 4.1.1). Assuming the availability of a feasible structure for(d - 1)-dimensional dominance joins,next we will describe how to achieve the purpose in $d$ -dimensional space.

设 $n = \left| R\right|  + \left| P\right|$。我们将使用递归方法设计一个可行的结构。基础情况是 $d = 1$，并且已在定理 1（特别是第 4.1.1 节）中解决。假设对于 (d - 1) 维支配连接存在一个可行的结构，接下来我们将描述如何在 $d$ 维空间中实现这一目标。

Structure. We will refer to dimension 1 the $x$ -dimension. Accordingly,the $x$ -range of a rectangle $r$ is the projection of $r$ on the first dimension.

结构。我们将第 1 维称为 $x$ 维。相应地，矩形 $r$ 的 $x$ 范围是 $r$ 在第一维上的投影。

Create a BST $\mathcal{T}$ on the set of values that includes (i) the endpoints of the x-ranges of the rectangles in $R$ ,and (ii) the x-coordinates of the points in $P$ . We assign each rectangle of $R$ and each point of $P$ to $O\left( {\log n}\right)$ nodes in $\mathcal{T}$ as follows:

在包含以下值的集合上创建一个二叉搜索树（BST）$\mathcal{T}$：(i) $R$ 中矩形的 x 范围的端点，以及 (ii) $P$ 中点的 x 坐标。我们按如下方式将 $R$ 中的每个矩形和 $P$ 中的每个点分配给 $\mathcal{T}$ 中的 $O\left( {\log n}\right)$ 个节点：

- For a rectangle $r \in  R$ ,let $( - \infty ,x\rbrack$ be its x-range. Descend the path $\Pi$ from the root of $\mathcal{T}$ to the leaf storing ${x}_{1}$ . Every time we go into the right child at some node $u$ on $\Pi$ ,we assign $r$ to the left child of $u$ .

- 对于矩形 $r \in  R$，设 $( - \infty ,x\rbrack$ 是其 x 范围。从 $\mathcal{T}$ 的根节点沿着路径 $\Pi$ 下降到存储 ${x}_{1}$ 的叶子节点。每次我们在 $\Pi$ 上的某个节点 $u$ 进入其右子节点时，我们将 $r$ 分配给 $u$ 的左子节点。

- A point $p \in  P$ is assigned to all the proper ancestors of the leaf storing the x-coordinate of $p$ .

- 点 $p \in  P$ 被分配给存储 $p$ 的 x 坐标的叶子节点的所有适当祖先节点。

Denote by ${R}_{u} \subseteq  R$ the set of rectangles assigned to a node $u$ of $\mathcal{T}$ ,and by ${P}_{u} \subseteq  P$ the set of points assigned to $u$ . The projection of each $r \in  {R}_{u}$ (or $p \in  {P}_{u}$ ) onto dimensions 2,3,..., $d$ defines a(d - 1)-dimensional rectangle (or point,resp.). Clearly,for any $r \in  {R}_{u}$ and any $p \in  {P}_{u}$ ,the $\mathrm{x}$ -range of $r$ covers the $\mathrm{x}$ -coordinate of $p$ . Hence, $r$ contains $p$ if and only if the(d - 1)-dimensional rectangle defined by $r$ contains the(d - 1)-dimensional point defined by $p$ . We denote by ${R}_{u}^{\prime }$ the set of(d - 1)-dimensional rectangles obtained from ${R}_{u}$ ,and by ${P}_{u}^{\prime }$ the set of(d - 1)-dimensional rectangles obtained from ${P}_{u}$ .

用 ${R}_{u} \subseteq  R$ 表示分配给 $\mathcal{T}$ 的节点 $u$ 的矩形集合，用 ${P}_{u} \subseteq  P$ 表示分配给 $u$ 的点集合。每个 $r \in  {R}_{u}$（或 $p \in  {P}_{u}$）在第 2、3、...、$d$ 维上的投影定义了一个 (d - 1) 维的矩形（或点）。显然，对于任意的 $r \in  {R}_{u}$ 和任意的 $p \in  {P}_{u}$，$r$ 的 $\mathrm{x}$ 范围覆盖了 $p$ 的 $\mathrm{x}$ 坐标。因此，$r$ 包含 $p$ 当且仅当由 $r$ 定义的 (d - 1) 维矩形包含由 $p$ 定义的 (d - 1) 维点。我们用 ${R}_{u}^{\prime }$ 表示从 ${R}_{u}$ 得到的 (d - 1) 维矩形集合，用 ${P}_{u}^{\prime }$ 表示从 ${P}_{u}$ 得到的 (d - 1) 维矩形集合。

Motivated by this,we associate $u$ with a secondary structure ${\Gamma }_{u}$ ,which is a(d - 1)-dimensional dominance-join structure on ${R}_{u}^{\prime }$ and ${P}_{u}^{\prime }$ . Node $u$ is productive if the(d - 1)-dimensional join on ${R}_{u}^{\prime }$ and ${P}_{u}^{\prime }$ returns a non-empty result. Because ${\Gamma }_{u}$ is a feasible structure,whether $u$ is productive can be decided in $\widetilde{O}\left( 1\right)$ time. All the productive nodes are collected into a productive list $\mathcal{L}$ .

受此启发，我们将 $u$ 与一个二级结构 ${\Gamma }_{u}$ 关联起来，该二级结构是关于 ${R}_{u}^{\prime }$ 和 ${P}_{u}^{\prime }$ 的 (d - 1) 维支配连接结构。如果关于 ${R}_{u}^{\prime }$ 和 ${P}_{u}^{\prime }$ 的 (d - 1) 维连接返回非空结果，则节点 $u$ 是有产出的。由于 ${\Gamma }_{u}$ 是一个可行的结构，因此可以在 $\widetilde{O}\left( 1\right)$ 时间内确定 $u$ 是否有产出。所有有产出的节点都被收集到一个有产出节点列表 $\mathcal{L}$ 中。

The size of ${\Gamma }_{u}$ is $\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$ by the inductive assumption. Therefore,our structure uses $\widetilde{O}\left( n\right)$ space overall.

根据归纳假设，${\Gamma }_{u}$ 的大小为 $\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$。因此，我们的结构总体上使用 $\widetilde{O}\left( n\right)$ 的空间。

Reporting the Join Result. For each node $u$ in $\mathcal{L}$ ,use ${\Gamma }_{u}$ to report the result of the(d - 1)- dimensional join on ${R}_{u}^{\prime }$ and ${P}_{u}^{\prime }$ with an $\widetilde{O}\left( 1\right)$ delay.

报告连接结果。对于 $\mathcal{L}$ 中的每个节点 $u$，使用 ${\Gamma }_{u}$ 以 $\widetilde{O}\left( 1\right)$ 的延迟报告关于 ${R}_{u}^{\prime }$ 和 ${P}_{u}^{\prime }$ 的 (d - 1) 维连接结果。

Observe that any(r,p)in the result of the original $d$ -dimensional join is reported by exactly one (d - 1)-dimensional join. Specifically,suppose that $( - \infty ,x\rbrack$ is the x-range of $r$ . Let $\Pi$ be the path in $\mathcal{T}$ from the root to the leaf of $x$ ,and $z$ be the leaf in $\mathcal{T}$ storing the x-coordinate of $p$ . Set node $u$ to the lowest ancestor of $z$ on $\Pi$ . Then,the pair(r,p)is reported at the left child of $u$ .

观察可知，原始 $d$ 维连接结果中的任何 (r, p) 都恰好由一个 (d - 1) 维连接报告。具体来说，假设 $( - \infty ,x\rbrack$ 是 $r$ 的 x 范围。设 $\Pi$ 是 $\mathcal{T}$ 中从根节点到 $x$ 的叶子节点的路径，$z$ 是 $\mathcal{T}$ 中存储 $p$ 的 x 坐标的叶子节点。将节点 $u$ 设置为 $\Pi$ 上 $z$ 的最低祖先节点。那么，(r, p) 这一对将在 $u$ 的左子节点处被报告。

We therefore achieve an $\widetilde{O}\left( 1\right)$ delay overall.

因此，我们总体上实现了 $\widetilde{O}\left( 1\right)$ 的延迟。

Update. For each node $u$ of $\mathcal{T}$ ,whenever a rectangle (or point) inserted/deleted from ${R}_{u}$ (or ${P}_{u}$ ),we inserted/deleted it in the(d - 1)-dimensional structure ${\Gamma }_{u}$ ,which takes $\widetilde{O}\left( 1\right)$ time by the inductive assumption. Furthermore, ${\Gamma }_{u}$ can be reconstructed by simply re-inserting all the rectangles in ${R}_{u}^{\prime }$ and ${P}_{u}^{\prime }$ ,which by the inductive assumption takes $\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$ time. It immediately follows from Lemma 6 that our structure can be updated in $\widetilde{O}\left( 1\right)$ time (both conditions in Section 2.3.2 have been satisfied).

更新。对于$\mathcal{T}$的每个节点$u$，每当从${R}_{u}$（或${P}_{u}$）中插入/删除一个矩形（或点）时，我们会在(d - 1)维结构${\Gamma }_{u}$中插入/删除它，根据归纳假设，这需要$\widetilde{O}\left( 1\right)$的时间。此外，${\Gamma }_{u}$可以通过简单地重新插入${R}_{u}^{\prime }$和${P}_{u}^{\prime }$中的所有矩形来重建，根据归纳假设，这需要$\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$的时间。由引理6可知，我们的结构可以在$\widetilde{O}\left( 1\right)$的时间内更新（第2.3.2节中的两个条件均已满足）。

We now have officially established the claim that any dominance join of a fixed dimensionality $d$ admits a feasible structure.

我们现在正式证明了这样一个命题：任何固定维度$d$的支配连接都存在一个可行的结构。

#### 4.2.2 Intersection Joins

#### 4.2.2 交集连接

We now attend to the $d$ -dimensional intersection join between two sets ${R}_{1}$ and ${R}_{2}$ of rectangles. It turns out that,as shown below,such a join can be converted to ${4}^{d} = O\left( 1\right)$ dominance joins,each of which has dimensionality at most ${3d} = O\left( 1\right)$ .

我们现在来处理两个矩形集合${R}_{1}$和${R}_{2}$之间的$d$维交集连接。结果表明，如下所示，这样的连接可以转换为${4}^{d} = O\left( 1\right)$个支配连接，每个支配连接的维度至多为${3d} = O\left( 1\right)$。

Consider two intersecting rectangles $r \in  {R}_{1}$ and ${r}^{\prime } \in  {R}_{2}$ . Fix a dimensionality $i \in  \left\lbrack  {1,d}\right\rbrack$ . Let $\left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$ and $\left\lbrack  {{x}_{i}^{\prime },{y}_{i}^{\prime }}\right\rbrack$ be the projections of $r$ and ${r}^{\prime }$ on this dimension,respectively. If we look at the permutation that sorts the four coordinates in ascending order, there are 4 possible permutations:

考虑两个相交的矩形$r \in  {R}_{1}$和${r}^{\prime } \in  {R}_{2}$。固定一个维度$i \in  \left\lbrack  {1,d}\right\rbrack$。设$\left\lbrack  {{x}_{i},{y}_{i}}\right\rbrack$和$\left\lbrack  {{x}_{i}^{\prime },{y}_{i}^{\prime }}\right\rbrack$分别是$r$和${r}^{\prime }$在这个维度上的投影。如果我们考虑将四个坐标按升序排序的排列，有4种可能的排列：

- ${x}_{i},{x}_{i}^{\prime },{y}_{i},{y}_{i}^{\prime }$

- ${x}_{i},{x}_{i}^{\prime },{y}_{i}^{\prime },{y}_{i}$

- ${x}_{i}^{\prime },{x}_{i},{y}_{i},{y}_{i}^{\prime }$

- ${x}_{i}^{\prime },{x}_{i},{y}_{i}^{\prime },{y}_{i}$ .

We can enforce each of the above permutations using the conjunction of at most 3 conditions of the form " $a \in  ( - \infty ,b\rbrack$ ". Specifically,the permutation ${x}_{i},{x}_{i}^{\prime },{y}_{i},{y}_{i}^{\prime }$ is enforced by:

我们可以使用至多3个形如“$a \in  ( - \infty ,b\rbrack$”的条件的合取来强制实现上述每种排列。具体来说，排列${x}_{i},{x}_{i}^{\prime },{y}_{i},{y}_{i}^{\prime }$由以下条件强制实现：

$$
{x}_{i} \in  \left( {-\infty ,{x}_{i}^{\prime }}\right\rbrack   \land   - {y}_{i} \in  \left( {-\infty , - {x}_{i}^{\prime }}\right\rbrack   \land  {y}_{i} \in  \left( {-\infty ,{y}_{i}^{\prime }}\right\rbrack  .
$$

The above is equivalent to requiring that the 3D point $\left( {{x}_{i}, - {y}_{i},{y}_{i}}\right)$ be covered by the 3-sided rectangle $\left( {-\infty ,{x}_{i}^{\prime }}\right\rbrack   \times  \left( {-\infty , - {x}_{i}^{\prime }}\right\rbrack   \times  \left( {-\infty ,{y}_{i}^{\prime }}\right\rbrack$ . Likewise,the permutation ${x}_{i},{x}_{i}^{\prime },{y}_{i}^{\prime },{y}_{i}$ is enforced by

上述条件等价于要求三维点$\left( {{x}_{i}, - {y}_{i},{y}_{i}}\right)$被三边矩形$\left( {-\infty ,{x}_{i}^{\prime }}\right\rbrack   \times  \left( {-\infty , - {x}_{i}^{\prime }}\right\rbrack   \times  \left( {-\infty ,{y}_{i}^{\prime }}\right\rbrack$覆盖。同样，排列${x}_{i},{x}_{i}^{\prime },{y}_{i}^{\prime },{y}_{i}$由以下条件强制实现

$$
{x}_{i} \in  \left( {-\infty ,{x}_{i}^{\prime }}\right\rbrack   \land   - {y}_{i} \in  \left( {-\infty , - {y}_{i}^{\prime }}\right\rbrack  .
$$

which is equivalent to requiring that the $2\mathrm{D}$ point $\left( {{x}_{i}, - {y}_{i}}\right)$ be covered by the 2-sided rectangle $\left( {-\infty ,{x}_{i}}\right\rbrack   \times  \left( {-\infty , - {y}_{i}^{\prime }}\right\rbrack$ . The other two permutations can also be enforced in a symmetric manner.

这等价于要求$2\mathrm{D}$点$\left( {{x}_{i}, - {y}_{i}}\right)$被两边矩形$\left( {-\infty ,{x}_{i}}\right\rbrack   \times  \left( {-\infty , - {y}_{i}^{\prime }}\right\rbrack$覆盖。另外两种排列也可以以对称的方式强制实现。

If one chooses a permutation independently for every dimension,the number of choices is ${4}^{d}$ . This is precisely the number of different ways that $r$ can intersect with ${r}^{\prime }$ . Let us refer to each of them as a configuration.

如果为每个维度独立选择一种排列，选择的数量为${4}^{d}$。这恰好是$r$与${r}^{\prime }$相交的不同方式的数量。我们将它们中的每一种称为一种配置。

It is clear from the above discussion that, we can create a dominance-join structure for each of the ${4}^{d}$ configurations. For each configuration,we convert a rectangle ${r}_{1} \in  {R}_{1}$ into a point $p$ by creating 3 or 2 new dimensions for every original dimension. This creates a point of dimensionality ${d}^{\prime } \leq  {3d}$ . Accordingly,a rectangle ${r}_{2} \in  {R}_{2}$ is converted to a ${d}^{\prime }$ -sided rectangle $r$ ,such that ${r}_{1}$ intersects with ${r}_{2}$ under that configuration if and only if $r$ covers $p$ .

从上述讨论中可以清楚地看到，我们可以为每个 ${4}^{d}$ 配置创建一个支配连接结构。对于每个配置，我们通过为每个原始维度创建 3 个或 2 个新维度，将一个矩形 ${r}_{1} \in  {R}_{1}$ 转换为一个点 $p$。这会创建一个维度为 ${d}^{\prime } \leq  {3d}$ 的点。相应地，一个矩形 ${r}_{2} \in  {R}_{2}$ 被转换为一个 ${d}^{\prime }$ 边形矩形 $r$，使得在该配置下 ${r}_{1}$ 与 ${r}_{2}$ 相交，当且仅当 $r$ 覆盖 $p$。

Since each result pair $\left( {{r}_{1},{r}_{2}}\right)  \in  {R}_{1} \times  {R}_{2}$ is reported by only one dominance join,we have obtained a feasible structure for the intersection join between ${R}_{1}$ and ${R}_{2}$ . This completes the proof of Theorem 2.

由于每个结果对 $\left( {{r}_{1},{r}_{2}}\right)  \in  {R}_{1} \times  {R}_{2}$ 仅由一个支配连接报告，我们已经为 ${R}_{1}$ 和 ${R}_{2}$ 之间的交集连接获得了一个可行的结构。这就完成了定理 2 的证明。

## 5 One-Dimensional Multi-Way Joins

## 5 一维多路连接

We now proceed to discuss 1D joins (i.e., $d$ fixed to 1) on a constant number $t$ of input sets ${R}_{1},\ldots ,{R}_{t}$ . We will prove Theorem 4 by presenting a feasible structure for any join topology $G$

现在我们继续讨论在固定数量 $t$ 的输入集合 ${R}_{1},\ldots ,{R}_{t}$ 上的一维连接（即，$d$ 固定为 1）。我们将通过为任何连接拓扑 $G$ 呈现一个可行的结构来证明定理 4。

### 5.1 Min/Max Intersection Joins

### 5.1 最小/最大交集连接

Next, we introduce the "min" and "max" versions of intersection joins whose purposes will be clear in the next subsection.

接下来，我们介绍交集连接的“最小”和“最大”版本，其用途将在下一小节中明确。

First,let us impose two total orders on ${R}_{1} \times  \ldots  \times  {R}_{t}$ . Let $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ and $\left( {{r}_{1}^{\prime },\ldots ,{r}_{t}^{\prime }}\right)$ be two distinct tuples from the cartesian product. Identify the first $i \in  \left\lbrack  {1,t}\right\rbrack$ such that ${r}_{i} \neq  {r}_{i}^{\prime }$ . Then,we say:

首先，让我们在 ${R}_{1} \times  \ldots  \times  {R}_{t}$ 上施加两个全序。设 $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ 和 $\left( {{r}_{1}^{\prime },\ldots ,{r}_{t}^{\prime }}\right)$ 是来自笛卡尔积的两个不同元组。找出第一个 $i \in  \left\lbrack  {1,t}\right\rbrack$ 使得 ${r}_{i} \neq  {r}_{i}^{\prime }$。然后，我们说：

- $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ is left-smaller (or left-larger) than $\left( {{r}_{1}^{\prime },\ldots ,{r}_{t}^{\prime }}\right)$ if the left endpoint of ${r}_{i}$ is smaller (or larger) than that of ${r}_{i}^{\prime }$ ;

- 如果 ${r}_{i}$ 的左端点小于（或大于）${r}_{i}^{\prime }$ 的左端点，则 $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ 比 $\left( {{r}_{1}^{\prime },\ldots ,{r}_{t}^{\prime }}\right)$ 左更小（或左更大）；

- $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ is right-smaller (or right-larger) than $\left( {{r}_{1}^{\prime },\ldots ,{r}_{t}^{\prime }}\right)$ if the right endpoint of ${r}_{i}$ is smaller (or larger) than that of ${r}_{i}^{\prime }$ .

- 如果 ${r}_{i}$ 的右端点小于（或大于）${r}_{i}^{\prime }$ 的右端点，则 $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ 比 $\left( {{r}_{1}^{\prime },\ldots ,{r}_{t}^{\prime }}\right)$ 右更小（或右更大）。

Note that the above are always well-defined because of the general position assumption stated in Section 2 (specifically, ${r}_{i}$ and ${r}_{i}^{\prime }$ must differ in both left endpoint and right endpoint).

请注意，由于第 2 节中陈述的一般位置假设（具体来说，${r}_{i}$ 和 ${r}_{i}^{\prime }$ 的左端点和右端点必须不同），上述定义总是明确的。

Let $J$ represent the set of tuples returned by the intersection join on ${R}_{1},\ldots ,{R}_{t}$ under the topology $G$ . We define:

设 $J$ 表示在拓扑 $G$ 下对 ${R}_{1},\ldots ,{R}_{t}$ 进行交集连接所返回的元组集合。我们定义：

- Min-IJ Query: return the left-smallest tuple in $J$ ;

- 最小交集连接查询：返回 $J$ 中左最小的元组；

- Max-IJ Query: return the right-largest tuple in $J$ .

- 最大交集连接查询：返回 $J$ 中右最大的元组。

The two queries can be supported efficiently:

这两个查询可以被高效地支持：

Lemma 11. There exists a structure that consumes $\widetilde{O}\left( n\right)$ space,supports an update (i.e.,insertion/deletion in any of ${R}_{1},\ldots ,{R}_{t}$ ) in $\widetilde{O}\left( 1\right)$ amortized time,and answers any min-/max-IJ query in $\widetilde{O}\left( 1\right)$ time.

引理 11。存在一种结构，它占用 $\widetilde{O}\left( n\right)$ 的空间，支持在 $\widetilde{O}\left( 1\right)$ 的均摊时间内进行一次更新（即，在任何 ${R}_{1},\ldots ,{R}_{t}$ 中进行插入/删除操作），并在 $\widetilde{O}\left( 1\right)$ 时间内回答任何最小/最大交集连接查询。

We will refer to the structure of the above lemma as an ${IJ}$ -heap on $\left( {{R}_{1},\ldots ,{R}_{t}}\right)$ under $G$ . The proof of the lemma is deferred to Section 6.

我们将上述引理的结构称为在$G$下关于$\left( {{R}_{1},\ldots ,{R}_{t}}\right)$的${IJ}$ -堆。该引理的证明推迟到第6节。

### 5.2 Reduction to min-IJ Queries

### 5.2 简化为最小IJ查询

This subsection serves as a proof for:

本小节用于证明以下内容：

Lemma 12. Given an IJ-heap on $\left( {{R}_{1},\ldots ,{R}_{t}}\right)$ under $G$ ,we can report all the result tuples in the intersection join with an $\widetilde{O}\left( 1\right)$ delay.

引理12. 给定在$G$下关于$\left( {{R}_{1},\ldots ,{R}_{t}}\right)$的IJ -堆，我们可以以$\widetilde{O}\left( 1\right)$的延迟报告交集连接中的所有结果元组。

Combining the above with Lemma 11 gives a feasible structure needed for Theorem 4.

将上述内容与引理11相结合，可得到定理4所需的可行结构。

Proof of Lemma 12. We answer an IJ query by calling the algorithm in Figure 5 as

引理12的证明。我们通过调用图5中的算法来回答IJ查询，

$$
\operatorname{IJ}\left( {0,\varnothing }\right) 
$$

which performs recursive calls at Line 8. The proposition below establishes the correctness of our algorithm:

该算法在第8行进行递归调用。下面的命题证明了我们算法的正确性：

<!-- Media -->

---

$\mathbf{{IJ}}\left( {\lambda ,\left\{  {{\rho }_{1},\ldots ,{\rho }_{\lambda }}\right\}  }\right)$

/* requirements: if $\lambda  \geq  1$ then

/* 要求：如果$\lambda  \geq  1$ 则

${C1} : {\rho }_{i} \in  {R}_{i}$ for each $i \in  \left\lbrack  {1,d}\right\rbrack$ .

对于每个$i \in  \left\lbrack  {1,d}\right\rbrack$，有${C1} : {\rho }_{i} \in  {R}_{i}$。

$\mathbf{{C2} : }{\rho }_{1},\ldots ,{\rho }_{\lambda }$ produce at least one result tuple.

$\mathbf{{C2} : }{\rho }_{1},\ldots ,{\rho }_{\lambda }$ 产生至少一个结果元组。

${C3}$ : The minimum result tuple from the current ${R}_{1},\ldots ,{R}_{t}$ (whose content may shrink and grow

${C3}$：当前${R}_{1},\ldots ,{R}_{t}$（其内容在算法执行期间可能会缩小和增大）中的最小结果元组

during the algorithm’s execution) is a tuple $\left( {{r}_{1}^{ * },\ldots ,{r}_{t}^{ * }}\right)$ satisfying ${r}_{i}^{ * } = {\rho }_{i}$ for all $i \in  \left\lbrack  {1,\lambda }\right\rbrack$ .

是一个满足对于所有$i \in  \left\lbrack  {1,\lambda }\right\rbrack$都有${r}_{i}^{ * } = {\rho }_{i}$的元组$\left( {{r}_{1}^{ * },\ldots ,{r}_{t}^{ * }}\right)$。

output: all result tuples $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ satisfying ${r}_{i} = {\rho }_{i}$ for all $i \in  \left\lbrack  {1,\lambda }\right\rbrack   *$ /

输出：所有满足对于所有$i \in  \left\lbrack  {1,\lambda }\right\rbrack   *$都有${r}_{i} = {\rho }_{i}$的结果元组$\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ */

	. if $\lambda  = t$ then output $\left( {{\rho }_{1},\ldots ,{\rho }_{\lambda }}\right)$ and return

	 . 如果$\lambda  = t$ 则输出$\left( {{\rho }_{1},\ldots ,{\rho }_{\lambda }}\right)$ 并返回

		${S}_{\text{del }} = \varnothing$

		repeat

		 重复

4. ${\rho }_{\lambda  + 1} \leftarrow$ the interval in ${R}_{\lambda  + 1}$ with the smallest left endpoint s.t. ${\rho }_{1},\ldots ,{\rho }_{\lambda },{\rho }_{\lambda  + 1}$ produce

4. ${\rho }_{\lambda  + 1} \leftarrow$ 是${R}_{\lambda  + 1}$ 中左端点最小的区间，使得${\rho }_{1},\ldots ,{\rho }_{\lambda },{\rho }_{\lambda  + 1}$ 产生

				at least one result tuple

				至少一个结果元组

				/* this requires a min-IJ query; see Proposition 3 */

				/* 这需要一个最小交集连接（min - IJ）查询；参见命题3 */

				if ${\rho }_{\lambda  + 1} =$ null then

				如果 ${\rho }_{\lambda  + 1} =$ 为空，则

6. insert all the tuples of ${S}_{\text{del }}$ back into ${R}_{\lambda  + 1}$

6. 将 ${S}_{\text{del }}$ 的所有元组重新插入 ${R}_{\lambda  + 1}$ 中

						return

						返回

				$\operatorname{IJ}\left( {\lambda  + 1,\left\{  {{\rho }_{1},\ldots ,{\rho }_{\lambda },{\rho }_{\lambda  + 1}}\right\}  }\right)$

				delete ${\rho }_{\lambda  + 1}$ from ${R}_{\lambda  + 1}$

				从 ${R}_{\lambda  + 1}$ 中删除 ${\rho }_{\lambda  + 1}$

				add ${\rho }_{\lambda  + 1}$ to ${S}_{\text{del }}$

				将 ${\rho }_{\lambda  + 1}$ 添加到 ${S}_{\text{del }}$ 中

---

## Figure 5: Reduction from intersection joins to min-IJ

## 图5：从交集连接到最小交集连接（min - IJ）的归约

<!-- Media -->

Proposition 2. ${C1},{C2}$ ,and ${C3}$ in Figure 5 are fulfilled by each recursive call to IJ during the execution of ${IJ}\left( {0,\varnothing }\right)$ . Furthermore,Every result tuple is output exactly once.

命题2. 在执行 ${IJ}\left( {0,\varnothing }\right)$ 期间，对交集连接（IJ）的每次递归调用都满足图5中的 ${C1},{C2}$ 和 ${C3}$。此外，每个结果元组都恰好输出一次。

Proof. See Appendix C.

证明：参见附录C。

We now use the supplied IJ-heap to implement Line 4:

我们现在使用提供的交集连接堆（IJ - heap）来实现第4行：

Proposition 3. Line 4 takes $\widetilde{O}\left( 1\right)$ time.

命题3. 第4行需要 $\widetilde{O}\left( 1\right)$ 时间。

Proof. Use the IJ-heap to perform a min-IJ query. If the query returns nothing,set ${\rho }_{\lambda  + 1}$ to null. Otherwise,suppose that it returns $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ . Check whether ${r}_{i} = {\rho }_{i}$ for all $i \in  \left\lbrack  {1,\lambda }\right\rbrack$ . If so,set ${\rho }_{\lambda  + 1}$ to ${r}_{\lambda  + 1}$ ; otherwise,set ${\rho }_{\lambda  + 1}$ to null. Requirement ${C3}$ ensures the correctness of the above strategy.

证明：使用交集连接堆（IJ - heap）执行一个最小交集连接（min - IJ）查询。如果查询没有返回任何结果，则将 ${\rho }_{\lambda  + 1}$ 设置为 null。否则，假设它返回 $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$。检查对于所有 $i \in  \left\lbrack  {1,\lambda }\right\rbrack$ 是否满足 ${r}_{i} = {\rho }_{i}$。如果是，则将 ${\rho }_{\lambda  + 1}$ 设置为 ${r}_{\lambda  + 1}$；否则，将 ${\rho }_{\lambda  + 1}$ 设置为 null。条件 ${C3}$ 确保了上述策略的正确性。

The following fact is crucial for proving that our algorithm has a short delay:

以下事实对于证明我们的算法具有较短延迟至关重要：

Proposition 4. At any moment of our algorithm,if Line 1 has output $x$ tuples,at most $t \cdot  x$ deletions have been performed at Line 9 (counting the deletions made at all levels of the recursion).

命题4. 在我们算法的任何时刻，如果第1行已经输出了 $x$ 个元组，那么在第9行最多执行了 $t \cdot  x$ 次删除操作（统计递归所有层级上的删除操作）。

Proof. An interval (in any of ${R}_{1},\ldots ,{R}_{t}$ ) is deleted after it has produced at least a result tuple. Each result tuple output at Line 1 can trigger at most $t$ deletions at Line 9 . The proposition thus follows.

证明：（${R}_{1},\ldots ,{R}_{t}$ 中的）一个区间在产生至少一个结果元组后被删除。第1行输出的每个结果元组最多可以触发第9行的 $t$ 次删除操作。因此，该命题成立。

We complete the proof of Lemma 12 by combining the following with Lemma 10:

我们通过将以下内容与引理10相结合来完成引理12的证明：

Proposition 5. Our algorithm is $\widetilde{O}\left( 1\right)$ -aggressive. Proof. Using the supplied IJ-heap,every insertion and deletion into any ${R}_{i}\left( {i \in  \left\lbrack  {1,t}\right\rbrack  }\right)$ takes $\widetilde{O}\left( 1\right)$ amortized time.

命题5. 我们的算法是$\widetilde{O}\left( 1\right)$ -激进的。证明. 使用所提供的IJ堆，对任何${R}_{i}\left( {i \in  \left\lbrack  {1,t}\right\rbrack  }\right)$的每次插入和删除操作的均摊时间为$\widetilde{O}\left( 1\right)$。

Consider any moment during the execution of our algorithm. Let ${n}_{del}$ be the total number of deletions that have been made at Line 9 so far (counting all levels of recursion). This implies that the total number of insertions at Line 6 is at most ${n}_{del}$ . It follows that the running time thus far is $\widetilde{O}\left( {n}_{del}\right)$ .

考虑我们算法执行过程中的任意时刻。设${n}_{del}$为到目前为止在第9行执行的删除操作的总数（统计所有递归层级）。这意味着在第6行执行的插入操作的总数最多为${n}_{del}$。由此可知，到目前为止的运行时间为$\widetilde{O}\left( {n}_{del}\right)$。

The $\widetilde{O}\left( 1\right)$ -aggressiveness then follows from Proposition 4,which indicates that we must have reported at least ${n}_{del}/t$ result tuples.

那么，$\widetilde{O}\left( 1\right)$ -激进性可由命题4推出，该命题表明我们必须至少报告了${n}_{del}/t$个结果元组。

## 6 The IJ-Heap

## 6 IJ堆

This section is dedicated to proving Lemma 11. We actually aim to support a more general form of min-/max-IJ queries. Remember that we have a constant number $t$ of interval sets ${R}_{1},\ldots ,{R}_{t}$ ,and a join topology $G$ . A min-/max-IJ query is now given $t$ pairs of values

本节致力于证明引理11。实际上，我们的目标是支持更一般形式的最小/最大IJ查询。请记住，我们有常数数量$t$的区间集${R}_{1},\ldots ,{R}_{t}$，以及一个连接拓扑$G$。现在，对于$t$给出了$t$对值的最小/最大IJ查询

$$
\left( {{a}_{i},{b}_{i}}\right) 
$$

for $i \in  \left\lbrack  {1,t}\right\rbrack$ . Let $J$ be the set of tuples $\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{t}$ satisfying all of the following:

对于$i \in  \left\lbrack  {1,t}\right\rbrack$。设$J$为满足以下所有条件的元组$\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{t}$的集合：

- $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ is in the result of the intersection join under topology $G$ ;

- $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$在拓扑$G$下的交集连接结果中；

- for each $i \in  \left\lbrack  {1,t}\right\rbrack  ,{r}_{i}$ intersects with both $\left( {-\infty ,{a}_{i}}\right\rbrack$ and $\left\lbrack  {{b}_{i},\infty }\right)$ . Note that,if ${b}_{i} \leq  {a}_{i}$ ,then this condition means that ${r}_{i}$ must intersect with $\left\lbrack  {{b}_{i},{a}_{i}}\right\rbrack$ .

- 对于每个$i \in  \left\lbrack  {1,t}\right\rbrack  ,{r}_{i}$，它与$\left( {-\infty ,{a}_{i}}\right\rbrack$和$\left\lbrack  {{b}_{i},\infty }\right)$都相交。请注意，如果${b}_{i} \leq  {a}_{i}$，那么这个条件意味着${r}_{i}$必须与$\left\lbrack  {{b}_{i},{a}_{i}}\right\rbrack$相交。

Then,the min-/max-IJ query should return the left-smallest/right-largest tuple in $J$ . Clearly,by setting ${a}_{i} = \infty$ and ${b}_{i} =  - \infty$ for all $i \in  \left\lbrack  {1,t}\right\rbrack$ ,a min-/max-IJ query degenerates into the version defined in Section 5.1.

那么，最小/最大IJ查询应该返回$J$中最左/最右的元组。显然，通过对所有$i \in  \left\lbrack  {1,t}\right\rbrack$设置${a}_{i} = \infty$和${b}_{i} =  - \infty$，最小/最大IJ查询退化为第5.1节中定义的版本。

The IJ-heap we aim to design should use of $\widetilde{O}\left( n\right)$ space $\left( {n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right| }\right)$ ,can be updated in $\widetilde{O}\left( 1\right)$ amortized time (per insertion/deletion in any ${R}_{i},1 \leq  i \leq  t$ ),and answer any (re-defined) $\min  - /$ max-IJ query in $\widetilde{O}\left( 1\right)$ time.

我们旨在设计的IJ堆应使用$\widetilde{O}\left( n\right)$空间$\left( {n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right| }\right)$，可以在$\widetilde{O}\left( 1\right)$的均摊时间内进行更新（对任何${R}_{i},1 \leq  i \leq  t$的每次插入/删除操作），并能在$\widetilde{O}\left( 1\right)$时间内回答任何（重新定义的）$\min  - /$最大IJ查询。

### 6.1 Notations

### 6.1 符号表示

Our strategy is to break $G$ into smaller subgraphs and handle the "sub-joins" represented by those subgraphs recursively. Some extra concepts and notations are needed to reason about those subjoins effectively.

我们的策略是将$G$分解为更小的子图，并递归地处理由这些子图表示的“子连接”。需要一些额外的概念和符号来有效地推理这些子连接。

Recall that $G$ has the vertex set $\{ 1,2,\ldots ,t\}$ ,which we will call the universe and represent as $U$ . Let $V$ be any non-empty subset of $U$ . A vector $\mathbf{v}$ is said to be defined in $V$ if:

回想一下，$G$的顶点集为$\{ 1,2,\ldots ,t\}$，我们将其称为全集并表示为$U$。设$V$为$U$的任意非空子集。如果向量$\mathbf{v}$满足以下条件，则称其在$V$中定义：

- $v$ has length $\left| V\right|$ ;

- $v$ 的长度为 $\left| V\right|$；

- for each $i \in  V,v$ has a distinct component which we denote as $\mathbf{v}\left\lbrack  i\right\rbrack$ ;

- 对于每个 $i \in  V,v$ 都有一个不同的分量，我们将其表示为 $\mathbf{v}\left\lbrack  i\right\rbrack$；

- $v$ lists its components $v\left\lbrack  i\right\rbrack  \left( {i \in  V}\right)$ in ascending order of $i$ .

- $v$ 按 $i$ 的升序列出其分量 $v\left\lbrack  i\right\rbrack  \left( {i \in  V}\right)$。

Henceforth,we will write $\mathbf{v}$ as ${\mathbf{v}}_{V}$ to indicate explicitly the set $V$ . The only exception arises when $V = U$ ,in which case $V$ is omitted but implicitly understood.

此后，我们将把 $\mathbf{v}$ 写成 ${\mathbf{v}}_{V}$ 以明确表示集合 $V$。唯一的例外情况是当 $V = U$ 时，此时 $V$ 被省略但隐含理解。

Consider two non-empty subsets $V,{V}^{\prime }$ of $U$ such that ${V}^{\prime } \subset  V$ . Given a vector ${\mathbf{v}}_{V}$ defined in $V$ , its projection in ${V}^{\prime }$ is the vector ${\mathbf{v}}_{{V}^{\prime }}^{\prime }$ where ${\mathbf{v}}_{{V}^{\prime }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{v}}_{V}\left\lbrack  i\right\rbrack$ for each $i \in  {V}^{\prime }$ .

考虑 $U$ 的两个非空子集 $V,{V}^{\prime }$，使得 ${V}^{\prime } \subset  V$。给定在 $V$ 中定义的向量 ${\mathbf{v}}_{V}$，它在 ${V}^{\prime }$ 中的投影是向量 ${\mathbf{v}}_{{V}^{\prime }}^{\prime }$，其中对于每个 $i \in  {V}^{\prime }$ 有 ${\mathbf{v}}_{{V}^{\prime }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{v}}_{V}\left\lbrack  i\right\rbrack$。

Given a non-empty subset $V \subseteq  U$ ,we refer to a vector ${\mathbf{R}}_{V}$ as an instance vector in $V$ if

给定一个非空子集 $V \subseteq  U$，如果

$$
{\mathbf{R}}_{V}\left\lbrack  i\right\rbrack   \subseteq  {R}_{i}
$$

for each $i \in  V$ . Define

对于每个 $i \in  V$。定义

$$
 \times  \left( {\mathbf{R}}_{V}\right)  = {\mathbf{R}}_{V}\left\lbrack  {i}_{1}\right\rbrack   \times  {\mathbf{R}}_{V}\left\lbrack  {i}_{2}\right\rbrack   \times  \ldots  \times  {\mathbf{R}}_{V}\left\lbrack  {i}_{\left| V\right| }\right\rbrack  
$$

where ${i}_{1},{i}_{2},\ldots ,{i}_{\left| V\right| }$ list out the integers in $V$ in ascending order. We will reserve $\mathcal{R}$ to denote the special instance vector $\left( {{R}_{1},\ldots ,{R}_{t}}\right)$ . An instance vector,in general,gives the interval sets that participate in a join.

其中 ${i}_{1},{i}_{2},\ldots ,{i}_{\left| V\right| }$ 按升序列出 $V$ 中的整数。我们将保留 $\mathcal{R}$ 来表示特殊实例向量 $\left( {{R}_{1},\ldots ,{R}_{t}}\right)$。一般来说，实例向量给出参与连接的区间集。

A vector ${\mathbf{r}}_{V}$ is said to be a data vector in $V$ if

如果

$$
{\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   \in  {R}_{i}
$$

for each $i \in  V$ . Note that a data vector differs from an instance vector in that,each component of the former is a rectangle while each component of the latter is a set of rectangles.

对于每个$i \in  V$。请注意，数据向量与实例向量的不同之处在于，前者的每个分量是一个矩形，而后者的每个分量是一组矩形。

We use ${G}_{V}$ to represent the subgraph of $G$ induced by the vertices in $V$ . Given an instance vector ${\mathbf{R}}_{V}$ ,define

我们使用${G}_{V}$来表示由$V$中的顶点所诱导出的$G$的子图。给定一个实例向量${\mathbf{R}}_{V}$，定义

$$
J\left( {{G}_{V},{\mathbf{R}}_{V}}\right)  = \left\{  {{\mathbf{r}}_{V} \in   \times  \left( {\mathbf{R}}_{V}\right)  \mid  {\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   \cap  {\mathbf{r}}_{V}\left\lbrack  j\right\rbrack   \neq  \varnothing \text{ for any distinct }i,j \in  V\text{ adjacent in }{G}_{V}}\right\}  .
$$

Note that $J\left( {{G}_{V},{\mathbf{R}}_{V}}\right)$ is the result of the intersection join defined by ${G}_{V}$ on the interval sets $\left\{  {{\mathbf{R}}_{V}\left\lbrack  i\right\rbrack   \mid  i \in  V}\right\}$ . In particular, $J\left( {G,\mathcal{R}}\right)$ is the result of the (full) intersection join on ${R}_{1},\ldots ,{R}_{t}$ and $G$ .

请注意，$J\left( {{G}_{V},{\mathbf{R}}_{V}}\right)$是由${G}_{V}$在区间集$\left\{  {{\mathbf{R}}_{V}\left\lbrack  i\right\rbrack   \mid  i \in  V}\right\}$上定义的交集连接的结果。特别地，$J\left( {G,\mathcal{R}}\right)$是在${R}_{1},\ldots ,{R}_{t}$和$G$上进行（完全）交集连接的结果。

Next,we impose two total orders on $\times  \left( {\mathbf{R}}_{V}\right)$ ,in a way consistent with the total orders defined in Section 5.1 on $\times  \left( \mathcal{R}\right)$ . Take any distinct elements ${\mathbf{r}}_{V},{\mathbf{r}}_{V}^{\prime }$ from $\times  \left( {\mathbf{R}}_{V}\right)$ . Let $i$ be the smallest integer in $V$ such that ${\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   \neq  {\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$ . Then,we say:

接下来，我们以与5.1节中在$\times  \left( \mathcal{R}\right)$上定义的全序一致的方式，在$\times  \left( {\mathbf{R}}_{V}\right)$上施加两个全序。从$\times  \left( {\mathbf{R}}_{V}\right)$中选取任意不同的元素${\mathbf{r}}_{V},{\mathbf{r}}_{V}^{\prime }$。设$i$是$V$中使得${\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   \neq  {\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$成立的最小整数。那么，我们说：

- ${\mathbf{r}}_{V}$ is left-smaller (or left-larger) than ${\mathbf{r}}_{V}^{\prime }$ if the left endpoint of ${\mathbf{r}}_{V}\left\lbrack  i\right\rbrack$ is smaller (or larger) than that of ${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$ ;

- 如果${\mathbf{r}}_{V}\left\lbrack  i\right\rbrack$的左端点小于（或大于）${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$的左端点，则${\mathbf{r}}_{V}$在左方小于（或大于）${\mathbf{r}}_{V}^{\prime }$；

- ${\mathbf{r}}_{V}$ is right-smaller (or right-larger) than ${\mathbf{r}}_{V}^{\prime }$ if the right endpoint of ${\mathbf{r}}_{V}\left\lbrack  i\right\rbrack$ is smaller (or larger) than that of ${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$ ;

- 如果${\mathbf{r}}_{V}\left\lbrack  i\right\rbrack$的右端点小于（或大于）${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$的右端点，则${\mathbf{r}}_{V}$在右方小于（或大于）${\mathbf{r}}_{V}^{\prime }$；

A vector ${\mathbf{q}}_{V}$ is said to be a constraint vector in $V$ if ${\mathbf{q}}_{V}\left\lbrack  i\right\rbrack$ is a pair

如果${\mathbf{q}}_{V}\left\lbrack  i\right\rbrack$是一个二元组，则向量${\mathbf{q}}_{V}$被称为$V$中的约束向量

$$
\left( {{\mathbf{q}}_{V}\left\lbrack  i\right\rbrack   \cdot  a,{\mathbf{q}}_{V}\left\lbrack  i\right\rbrack   \cdot  b}\right) 
$$

Define

定义

$$
J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)  = \left\{  {{\mathbf{r}}_{V} \in  J\left( {{G}_{V},{\mathbf{R}}_{V}}\right)  \mid  }\right. \text{for all}i \in  V
$$

$$
\left. {{\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   \cap  \left( {-\infty ,q\left\lbrack  i\right\rbrack  .a}\right\rbrack   \neq  \varnothing \text{ and }{\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   \cap  \lbrack q\left\lbrack  i\right\rbrack  .b,\infty ) \neq  \varnothing }\right\}  
$$

Given a constraint vector ${\mathbf{q}}_{V}$ ,a min-IJ query (or a max-IJ query) on ${\mathbf{R}}_{V}$ under ${G}_{V}$ returns the left-smallest (right-largest) element in $J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ .

给定一个约束向量${\mathbf{q}}_{V}$，在${G}_{V}$条件下对${\mathbf{R}}_{V}$进行的最小交集连接查询（或最大交集连接查询）返回$J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$中左方最小（右方最大）的元素。

Finally,it is worth pointing out that,all the above definitions apply to $t = 1$ as well.

最后，值得指出的是，上述所有定义同样适用于$t = 1$。

### 6.2 The Endpoint Property

### 6.2 端点性质

Let $q$ be a constraint vector. Set:

设$q$为一个约束向量。设定：

$$
a = \mathop{\max }\limits_{{i = 1}}^{t}\mathbf{q}\left\lbrack  i\right\rbrack   \cdot  a \tag{2}
$$

$$
b = \mathop{\min }\limits_{{i = 1}}^{t}\mathbf{q}\left\lbrack  i\right\rbrack   \cdot  b \tag{3}
$$

Consider an arbitrary data vector $\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$ in $J\left( {G,\mathbf{\mathcal{R}},\mathbf{q}}\right)$ . Define:

考虑$J\left( {G,\mathbf{\mathcal{R}},\mathbf{q}}\right)$中任意一个数据向量$\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$。定义：

$$
 \sqcup  \left( \mathbf{r}\right)  = {r}_{1} \cup  \ldots  \cup  {r}_{t}. \tag{4}
$$

Since $G$ is connected, $\sqcup  \left( \mathbf{r}\right)$ must be a consecutive interval. Specifically,if $x$ (or $y$ ,resp.) is the smallest (or largest,resp.) left (or right,resp.) endpoint of ${r}_{1},\ldots ,{r}_{t}$ ,then $\sqcup  \left( \mathbf{r}\right)  = \left\lbrack  {x,y}\right\rbrack$ . We must have:

由于 $G$ 是连通的，$\sqcup  \left( \mathbf{r}\right)$ 必定是一个连续区间。具体而言，如果 $x$（或 $y$）分别是 ${r}_{1},\ldots ,{r}_{t}$ 的最小（或最大）左（或右）端点，那么 $\sqcup  \left( \mathbf{r}\right)  = \left\lbrack  {x,y}\right\rbrack$。我们必有：

Lemma 13 (Endpoint Property). If $b \leq  a$ ,then $\sqcup  \left( \mathbf{r}\right)$ must have a non-empty intersection with $\left\lbrack  {b,a}\right\rbrack$ . Otherwise, $\sqcup  \left( \mathbf{r}\right)$ must contain both $a$ and $b$ .

引理 13（端点性质）。若 $b \leq  a$，则 $\sqcup  \left( \mathbf{r}\right)$ 必定与 $\left\lbrack  {b,a}\right\rbrack$ 有非空交集。否则，$\sqcup  \left( \mathbf{r}\right)$ 必定同时包含 $a$ 和 $b$。

Proof. We will prove that $\sqcup  \left( \mathbf{r}\right)$ must intersect with both of $\left( {-\infty ,a\rbrack \text{and}\lbrack b,\infty }\right)$ . Then,the lemma will follow.

证明。我们将证明 $\sqcup  \left( \mathbf{r}\right)$ 必定与 $\left( {-\infty ,a\rbrack \text{and}\lbrack b,\infty }\right)$ 两者都相交。这样，引理即可得证。

Assume that $\sqcup  \left( \mathbf{r}\right)$ is disjoint with $( - \infty ,a\rbrack$ ,that is, $\sqcup  \left( \mathbf{r}\right)$ is entirely to the right of $a$ . Suppose that $a = \mathbf{q}\left\lbrack  i\right\rbrack  .a$ ,for some $i \in  \left\lbrack  {1,d}\right\rbrack$ . Thus, $\sqcup  \left( \mathbf{r}\right)$ is disjoint with $( - \infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack$ . This contradicts the fact that at least one of ${r}_{1},\ldots ,{r}_{t}$ must intersect with $( - \infty ,\mathbf{q}\left\lbrack  i\right\rbrack$ .a].

假设 $\sqcup  \left( \mathbf{r}\right)$ 与 $( - \infty ,a\rbrack$ 不相交，即 $\sqcup  \left( \mathbf{r}\right)$ 完全位于 $a$ 的右侧。假设对于某个 $i \in  \left\lbrack  {1,d}\right\rbrack$ 有 $a = \mathbf{q}\left\lbrack  i\right\rbrack  .a$。因此，$\sqcup  \left( \mathbf{r}\right)$ 与 $( - \infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack$ 不相交。这与 ${r}_{1},\ldots ,{r}_{t}$ 中至少有一个必定与 $( - \infty ,\mathbf{q}\left\lbrack  i\right\rbrack$ 相交这一事实相矛盾。

A symmetric argument shows that $\sqcup  \left( \mathbf{r}\right)$ must also intersect with $\lbrack b,\infty )$ .

通过对称的论证可知，$\sqcup  \left( \mathbf{r}\right)$ 也必定与 $\lbrack b,\infty )$ 相交。

### 6.3 Structure Overview

### 6.3 结构概述

Given $\left( {{R}_{1},\ldots ,{R}_{t}}\right)$ and $G$ ,we build an IJ-heap using a recursive approach,which works by induction on $t$ .

给定 $\left( {{R}_{1},\ldots ,{R}_{t}}\right)$ 和 $G$，我们使用递归方法构建一个 IJ - 堆，该方法通过对 $t$ 进行归纳来实现。

Base: $t = 1$ . Recall that the definition of min-/max-IJ queries have been extended to $t = 1$ in Section 6.1. In this case,only one interval set ${R}_{1}$ exists. Given a pair of real values(a,b),a min- (max-,resp.) IJ query returns the interval $r$ with the smallest left (or largest right,resp.) endpoint, among all the intervals of ${R}_{1}$ that intersect with both $\left( {-\infty ,a}\right\rbrack$ and $\lbrack b,\infty )$ .

基础情况：$t = 1$。回顾一下，在 6.1 节中，最小/最大 IJ 查询的定义已扩展到 $t = 1$。在这种情况下，仅存在一个区间集 ${R}_{1}$。给定一对实数值 (a, b)，最小（或最大）IJ 查询会在与 $\left( {-\infty ,a}\right\rbrack$ 和 $\lbrack b,\infty )$ 都相交的 ${R}_{1}$ 的所有区间中，返回左端点最小（或右端点最大）的区间 $r$。

The query can be regarded as a 2D "range min" query described in Section 2.4. For this purpose, convert $r$ into a 2D rectangle $r \times  r$ . Clearly, $r$ intersects with both $\left( {-\infty ,a\rbrack \text{and}\lbrack b,\infty }\right)$ if and only if $r \times  r$ intersects with the $2\mathrm{D}$ rectangle $\left( {-\infty ,a\rbrack \times \lbrack b,\infty }\right)$ . Thus,the structure of Lemma 7 adequately serves our purposes.

该查询可被视为 2.4 节中描述的二维“范围最小值”查询。为此，将 $r$ 转换为二维矩形 $r \times  r$。显然，$r$ 与 $\left( {-\infty ,a\rbrack \text{and}\lbrack b,\infty }\right)$ 都相交当且仅当 $r \times  r$ 与 $2\mathrm{D}$ 矩形 $\left( {-\infty ,a\rbrack \times \lbrack b,\infty }\right)$ 相交。因此，引理 7 的结构足以满足我们的需求。

Inductive: $t \geq  2$ . Assume that we already know how to obtain an IJ-heap when $G$ has at most $t - 1$ vertices. Next,we will design an IJ-heap for any $G$ with $t$ vertices.

归纳情况：$t \geq  2$。假设我们已经知道当 $G$ 最多有 $t - 1$ 个顶点时如何获得一个 IJ - 堆。接下来，我们将为任何具有 $t$ 个顶点的 $G$ 设计一个 IJ - 堆。

Build an interval tree $\mathcal{T}$ on ${R}_{1} \cup  \ldots  \cup  {R}_{t}$ . Denote by $\mathcal{S}$ the set of endpoints of all the intervals in ${R}_{1} \cup  \ldots  \cup  {R}_{t}$ . Note that $\mathcal{S}$ is also the set of keys stored in the leaves of $\mathcal{T}$ .

在${R}_{1} \cup  \ldots  \cup  {R}_{t}$上构建一个区间树$\mathcal{T}$。用$\mathcal{S}$表示${R}_{1} \cup  \ldots  \cup  {R}_{t}$中所有区间的端点集合。注意，$\mathcal{S}$也是存储在$\mathcal{T}$叶子节点中的键的集合。

Consider a min-IJ query with a constraint vector $\mathbf{q}$ . Define ${\Pi }_{1}$ (or ${\Pi }_{2}$ ) be the path in $\mathcal{T}$ from the root to the leaf storing the successor of $a$ (or predecessor of $b$ ) in $\mathcal{S}$ ,where $a$ and $b$ are given in (2) and (3),respectively. Next,we introduce a taxonomy of the tuples in $J\left( {G,\mathcal{R},\mathbf{q}}\right)$ . The taxonomy will naturally lead to our strategy of answering the query.

考虑一个带有约束向量$\mathbf{q}$的最小区间连接（min - IJ）查询。定义${\Pi }_{1}$（或${\Pi }_{2}$）为$\mathcal{T}$中从根节点到存储$\mathcal{S}$中$a$的后继（或$b$的前驱）的叶子节点的路径，其中$a$和$b$分别在(2)和(3)中给出。接下来，我们引入$J\left( {G,\mathcal{R},\mathbf{q}}\right)$中元组的分类。这种分类将自然地引导我们回答查询的策略。

We say that a data vector $\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$ hinges on a node $u$ in $\mathcal{T}$ if

我们称一个数据向量$\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$依赖于$\mathcal{T}$中的节点$u$，如果

- at least one of ${r}_{1},\ldots ,{r}_{t}$ belongs to $\operatorname{stab}\left( u\right)$ (i.e.,the stabbing set of $u$ ; see Section 2.1);

- ${r}_{1},\ldots ,{r}_{t}$中至少有一个属于$\operatorname{stab}\left( u\right)$（即$u$的刺穿集；见2.1节）；

- none of ${r}_{1},\ldots ,{r}_{t}$ belongs to the stabbing set of any proper ancestor of $u$ .

- ${r}_{1},\ldots ,{r}_{t}$中没有一个属于$u$任何真祖先节点的刺穿集。

As each interval appears in exactly one stabbing set, $\mathbf{r}$ must hinge on exactly one node. If $\mathbf{r}$ hinges on $u$ ,then the interval $\sqcup  \left( \mathbf{r}\right)$ as defined in (4) must be covered by $\sigma \left( u\right)$ (i.e.,the slab of $u$ ; see Section 2.1).

由于每个区间恰好出现在一个刺穿集中，$\mathbf{r}$必定恰好依赖于一个节点。如果$\mathbf{r}$依赖于$u$，那么(4)中定义的区间$\sqcup  \left( \mathbf{r}\right)$必定被$\sigma \left( u\right)$覆盖（即$u$的平板；见2.1节）。

By the endpoint property in Lemma 13,a data vector $\mathbf{r} \in  J\left( {G,\mathcal{R},\mathbf{q}}\right)$ must belong to one of the following categories:

根据引理13中的端点性质，一个数据向量$\mathbf{r} \in  J\left( {G,\mathcal{R},\mathbf{q}}\right)$必定属于以下类别之一：

- Category 1: $r$ hinges on a node on ${\Pi }_{1}$ or ${\Pi }_{2}$ .

- 类别1：$r$依赖于${\Pi }_{1}$或${\Pi }_{2}$上的一个节点。

- Category 2: (Applicable only if $b \leq  a$ ) $r$ hinges on a node $u$ whose slab $\sigma \left( u\right)$ is covered by $\left\lbrack  {b,a}\right\rbrack$ .

- 类别2：（仅当$b \leq  a$时适用）$r$依赖于一个节点$u$，其平板$\sigma \left( u\right)$被$\left\lbrack  {b,a}\right\rbrack$覆盖。

We will find the left-smallest data vectors from Categories 1 and 2, respectively. Then, the fina answer to the min-IJ query is the left-smaller one between the two fetched data vectors.

我们将分别从类别1和类别2中找到最左侧的数据向量。然后，最小区间连接查询的最终答案是这两个获取的数据向量中最左侧的那个。

In Section 6.4,we will describe a secondary structure associated with $u$ ,which is crucial to retrieving the data vectors of Categories 1 and 2. The algorithms for retrieving those categories will be presented in Section 6.5.

在6.4节中，我们将描述与$u$相关的二级结构，这对于检索类别1和类别2的数据向量至关重要。检索这些类别的算法将在6.5节中给出。

### 6.4 The Combination Structure

### 6.4 组合结构

To motivate the problem to be tackled in this subsection,let us fix a node $u$ in the interval tree $\mathcal{T}$ and consider any data vector $\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$ that hinges on $u$ . By definition,there must be at least one integer $i \in  \left\lbrack  {1,t}\right\rbrack$ such that ${r}_{i}$ appears in $\operatorname{stab}\left( u\right)$ . Consider any other integer $j \in  \left\lbrack  {1,t}\right\rbrack$ that is different from $i$ . Where can ${r}_{j}$ appear in the interval tree $\mathcal{T}$ ? There are three possibilities: in the stabbing set of (i) $u$ itself,(ii) a node in the left subtree of $u$ ,or (iii) a node in the right subtree of $u$ .

为了引出本节要解决的问题，让我们固定区间树 $\mathcal{T}$ 中的一个节点 $u$，并考虑依赖于 $u$ 的任意数据向量 $\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$。根据定义，必定存在至少一个整数 $i \in  \left\lbrack  {1,t}\right\rbrack$，使得 ${r}_{i}$ 出现在 $\operatorname{stab}\left( u\right)$ 中。考虑与 $i$ 不同的任意其他整数 $j \in  \left\lbrack  {1,t}\right\rbrack$。${r}_{j}$ 可能出现在区间树 $\mathcal{T}$ 的什么位置呢？有三种可能性：(i) $u$ 自身的穿刺集（stabbing set）中；(ii) $u$ 左子树的某个节点的穿刺集中；或者 (iii) $u$ 右子树的某个节点的穿刺集中。

We can divide all the data vectors hinging on $u$ into disjoint "groups" as follows. For each $i \in  \left\lbrack  {1,t}\right\rbrack$ ,the ${r}_{i}$ component of such a data vector $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ can independently take any of the aforementioned three possibilities,which gives ${3}^{t}$ "possibility combinations". Imagine placing two data vectors in the same group if their possibility combinations are identical. At first glance, this yields ${3}^{t}$ groups,but ${2}^{t}$ groups must be empty and,hence,useless. Specifically,a group is useless if there does not exist any $i \in  \left\lbrack  {1,t}\right\rbrack$ such that ${r}_{i}$ takes the possibility of appearing in $\operatorname{stab}\left( u\right)$ . In a useless group,each ${r}_{i}$ has only two possibilities; hence,the number of useless groups is ${2}^{t}$ . We thus conclude that the number of useful groups is ${3}^{t} - {2}^{t}$ ,which is a constant.

我们可以按如下方式将所有依赖于 $u$ 的数据向量划分为不相交的“组”。对于每个 $i \in  \left\lbrack  {1,t}\right\rbrack$，这样一个数据向量 $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ 的 ${r}_{i}$ 分量可以独立地取上述三种可能性中的任意一种，这就产生了 ${3}^{t}$ 种“可能性组合”。设想如果两个数据向量的可能性组合相同，就将它们放在同一组中。乍一看，这会产生 ${3}^{t}$ 个组，但 ${2}^{t}$ 个组必定为空，因此是无用的。具体来说，如果不存在任何 $i \in  \left\lbrack  {1,t}\right\rbrack$ 使得 ${r}_{i}$ 取出现在 $\operatorname{stab}\left( u\right)$ 中的可能性，那么这个组就是无用的。在一个无用的组中，每个 ${r}_{i}$ 只有两种可能性；因此，无用组的数量是 ${2}^{t}$。于是我们得出，有用组的数量是 ${3}^{t} - {2}^{t}$，这是一个常数。

Recall that our goal in Section 6.3 is to identify the left-smallest data vector in $J\left( {G,\mathcal{R},\mathbf{q}}\right)$ ,and every data vector in $J\left( {G,\mathbf{R},\mathbf{q}}\right)$ hinges on a node $u$ of Category 1 or 2 . Suppose that,for every $u$ , we can fetch the left-smallest data vector of $J\left( {G,\mathbf{R},\mathbf{q}}\right)$ in every useful group of $u$ . Then,the overall left-smallest data vector in $J\left( {G,\mathcal{R},\mathbf{q}}\right)$ is simply the left-smallest from all the data vectors fetched earlier. The challenge is to design a secondary structure for every $u$ that allows fast retrieval of the aforementioned data vector from each of its useful groups. The structure must support efficient updates as well.

回顾一下，我们在 6.3 节的目标是找出 $J\left( {G,\mathcal{R},\mathbf{q}}\right)$ 中最左的数据向量，并且 $J\left( {G,\mathbf{R},\mathbf{q}}\right)$ 中的每个数据向量都依赖于类别 1 或 2 的节点 $u$。假设对于每个 $u$，我们都能从 $u$ 的每个有用组中取出 $J\left( {G,\mathbf{R},\mathbf{q}}\right)$ 的最左数据向量。那么，$J\left( {G,\mathcal{R},\mathbf{q}}\right)$ 中整体的最左数据向量就是之前取出的所有数据向量中最左的那个。挑战在于为每个 $u$ 设计一个二级结构，以便能从其每个有用组中快速检索到上述数据向量。该结构还必须支持高效的更新操作。

Next,we formalize the above strategy. Fix an arbitrary internal $u$ in $\mathcal{T}$ . For each $i \in  \left\lbrack  {1,t}\right\rbrack$ , define:

接下来，我们将上述策略形式化。固定 $\mathcal{T}$ 中任意一个内部节点 $u$。对于每个 $i \in  \left\lbrack  {1,t}\right\rbrack$，定义：

- ${\operatorname{stab}}_{i}^{ < }\left( u\right)$ : the set of intervals from ${R}_{i}$ in the stabbing sets of the nodes in the left subtree of $u$ ;

- ${\operatorname{stab}}_{i}^{ < }\left( u\right)$：${R}_{i}$ 中位于 $u$ 左子树节点的穿刺集内的区间集合；

- ${\operatorname{stab}}_{i}^{ = }\left( u\right)$ : the set of intervals from ${R}_{i}$ in the stabbing set of $u$ ;

- ${\operatorname{stab}}_{i}^{ = }\left( u\right)$：${R}_{i}$ 中位于 $u$ 的穿刺集内的区间集合；

- ${\operatorname{stab}}_{i}^{ > }\left( u\right)$ : the set of intervals from ${R}_{i}$ in the stabbing sets of the nodes in the right subtree of $u$ . We define a combination of $u -$ denoted as $\mathcal{C} -$ as the cartesian product

- ${\operatorname{stab}}_{i}^{ > }\left( u\right)$：${R}_{i}$ 中位于 $u$ 右子树节点的穿刺集内的区间集合。我们将 $u -$ 的一种组合记为 $\mathcal{C} -$，定义为笛卡尔积

$$
{\operatorname{stab}}_{1}^{?}\left( u\right)  \times  {\operatorname{stab}}_{2}^{?}\left( u\right)  \times  \ldots  \times  {\operatorname{stab}}_{t}^{?}\left( u\right) 
$$

where each of the $t$ question marks "?" can independently take "<","=",or ">",subject to the constraint that at least one of those symbols must take "=". The number of combinations is ${3}^{t} - {2}^{t} = O\left( 1\right)$ .

其中每个$t$问号“?”可以独立地取“<”、“=”或“>”，但需满足至少有一个符号必须取“=”的约束条件。组合的数量为${3}^{t} - {2}^{t} = O\left( 1\right)$。

Phrased differently,a combination $\mathcal{C}$ of $u$ is determined by three disjoint sets ${V}^{ < },{V}^{ = }$ ,and ${V}^{ > }$ whose union equals the universe $U = \{ 1,\ldots ,t\}$ . Construct a vector ${\mathbf{R}}^{\mathbb{C}}$ where,for each $i \in  U,{\mathbf{R}}^{\mathbb{C}}\left\lbrack  i\right\rbrack$ equals

换一种说法，$u$的一个组合$\mathcal{C}$由三个不相交的集合${V}^{ < },{V}^{ = }$和${V}^{ > }$确定，它们的并集等于全集$U = \{ 1,\ldots ,t\}$。构造一个向量${\mathbf{R}}^{\mathbb{C}}$，其中，对于每个$i \in  U,{\mathbf{R}}^{\mathbb{C}}\left\lbrack  i\right\rbrack$等于

- ${\operatorname{stab}}_{i}^{ < }\left( u\right)$ if $i \in  {V}^{ < }$

- 如果$i \in  {V}^{ < }$，则为${\operatorname{stab}}_{i}^{ < }\left( u\right)$

- ${\operatorname{stab}}_{i}^{ = }\left( u\right)$ if $i \in  {V}^{ = }$

- 如果$i \in  {V}^{ = }$，则为${\operatorname{stab}}_{i}^{ = }\left( u\right)$

- ${\operatorname{stab}}_{i}^{ > }\left( u\right)$ if $i \in  {V}^{ > }$ .

- 如果$i \in  {V}^{ > }$，则为${\operatorname{stab}}_{i}^{ > }\left( u\right)$。

Thus,the combination is simply $\times  \left( {\mathbf{R}}^{\mathcal{C}}\right)$ . Remember that $\left| {V}^{ = }\right|  \geq  1$ .

因此，该组合就是$\times  \left( {\mathbf{R}}^{\mathcal{C}}\right)$。记住$\left| {V}^{ = }\right|  \geq  1$。

The rest of the subsection serves as a proof of:

本小节的其余部分用作以下内容的证明：

Lemma 14. For each combination $\mathcal{C}$ of $u$ ,we build a structure of $\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$ space to meet both requirements below:

引理14。对于$u$的每个组合$\mathcal{C}$，我们构建一个$\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$空间的结构以满足以下两个要求：

- Any min-/max-IJ query on ${\mathbf{R}}^{\mathcal{C}}$ under the topology $G$ can be answered in $\widetilde{O}\left( 1\right)$ time. Specifically, for any constraint vector $\mathbf{q}$ ,we can return in $\widetilde{O}\left( 1\right)$ time the left-smallest tuple in $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ .

- 在拓扑结构$G$下，对${\mathbf{R}}^{\mathcal{C}}$的任何最小/最大IJ查询都可以在$\widetilde{O}\left( 1\right)$时间内得到答案。具体来说，对于任何约束向量$\mathbf{q}$，我们可以在$\widetilde{O}\left( 1\right)$时间内返回$J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$中最左边的元组。

- Given an insertion/deletion in ${\mathbf{R}}^{\mathbb{C}}\left\lbrack  i\right\rbrack$ for any $i \in  \left\lbrack  {1,t}\right\rbrack$ ,we can update the structure in $\widetilde{O}\left( 1\right)$ amortized time.

- 对于任何$i \in  \left\lbrack  {1,t}\right\rbrack$，给定${\mathbf{R}}^{\mathbb{C}}\left\lbrack  i\right\rbrack$中的一次插入/删除操作，我们可以在$\widetilde{O}\left( 1\right)$的均摊时间内更新该结构。

We refer to the structure of the above lemma as the combination structure of $\mathcal{C}$ .

我们将上述引理的结构称为$\mathcal{C}$的组合结构。

Structure. Does $G$ have an edge between a vertex $i \in  {V}^{ < }$ and a vertex $j \in  {V}^{ > }$ ? If so,we answer any min-IJ query on ${\mathbf{R}}^{\mathbb{C}}$ under $G$ by returning nothing at all. To see why,notice that no intervals in ${\operatorname{stab}}_{i}^{ < }\left( u\right)$ can intersect with any intervals in ${\operatorname{stab}}_{j}^{ > }\left( u\right)$ . Hence, $J\left( {G,{\mathbf{R}}^{\complement }}\right)  = \varnothing$ ; and accordingly, $J\left( {G,{\mathbf{R}}^{\mathcal{C}},\mathbf{q}}\right)  \subseteq  J\left( {G,{\mathbf{R}}^{\mathcal{C}}}\right)  = \varnothing$ regardless of $\mathbf{q}$ .

结构。在顶点$i \in  {V}^{ < }$和顶点$j \in  {V}^{ > }$之间，$G$是否有一条边？如果有，我们通过不返回任何内容来回答在$G$下对${\mathbf{R}}^{\mathbb{C}}$的任何最小IJ查询。要明白为什么，请注意${\operatorname{stab}}_{i}^{ < }\left( u\right)$中的任何区间都不能与${\operatorname{stab}}_{j}^{ > }\left( u\right)$中的任何区间相交。因此，$J\left( {G,{\mathbf{R}}^{\complement }}\right)  = \varnothing$；相应地，无论$\mathbf{q}$如何，都有$J\left( {G,{\mathbf{R}}^{\mathcal{C}},\mathbf{q}}\right)  \subseteq  J\left( {G,{\mathbf{R}}^{\mathcal{C}}}\right)  = \varnothing$。

Next,we focus on the situation where $G$ has no edges between ${V}^{ < }$ and ${V}^{ > }$ . Consider the subgraph ${G}^{ <  > }$ of $G$ that is induced by the vertices in $U \smallsetminus  {V}^{ = }$ ; in other words, ${G}^{ <  > }$ is obtained by removing the vertices in ${V}^{ = }$ from $G$ .

接下来，我们关注$G$在${V}^{ < }$和${V}^{ > }$之间没有边的情况。考虑$G$中由$U \smallsetminus  {V}^{ = }$中的顶点所诱导的子图${G}^{ <  > }$；换句话说，${G}^{ <  > }$是通过从$G$中移除${V}^{ = }$中的顶点得到的。

Compute the connected components (CC) of ${G}^{ <  > }$ . Every CC must be a subset of either ${V}^{ < }$ or ${V}^{ > }$ . Let ${h}_{1}$ be the number of CCs that are subsets of ${V}^{ < }$ ; and if ${h}_{1} > 0$ ,denote the CCs as

计算${G}^{ <  > }$的连通分量（CC）。每个连通分量必定是${V}^{ < }$或${V}^{ > }$的子集。设${h}_{1}$为是${V}^{ < }$子集的连通分量的数量；并且如果${h}_{1} > 0$，将这些连通分量表示为

$$
{V}_{1}^{ < },\ldots ,{V}_{{h}_{1}}^{ < }
$$

note that their union is ${V}^{ < }$ . Likewise,let ${h}_{2}$ be the number of CCs that are subsets of ${V}^{ > }$ ; and if ${h}_{2} > 0$ ,represent them as

注意它们的并集是${V}^{ < }$。同样地，设${h}_{2}$为是${V}^{ > }$子集的连通分量的数量；并且如果${h}_{2} > 0$，将它们表示为

$$
{V}_{1}^{ > },\ldots ,{V}_{{h}_{2}}^{ > }
$$

note that their union is ${V}^{ > }$ .

注意它们的并集是${V}^{ > }$。

The combination structure of $\mathcal{C}$ has three parts:

$\mathcal{C}$的组合结构有三个部分：

- (if ${h}_{1} > 0$ ) for each $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ ,build an IJ-heap on the instance vector ${\mathbf{R}}_{{V}_{i}^{ < }}$ under the topology ${G}_{{V}_{j}^{ < }}$ ,where

- （如果${h}_{1} > 0$）对于每个$j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$，在拓扑结构${G}_{{V}_{j}^{ < }}$下的实例向量${\mathbf{R}}_{{V}_{i}^{ < }}$上构建一个IJ堆，其中

$$
{\mathbf{R}}_{{V}_{j}^{ < }} = \text{the projection of}{\mathbf{R}}^{\mathcal{C}}\text{in}{V}_{j}^{ < }\text{.}
$$

Recall that (as defined in Section 6.1) ${G}_{{V}_{j}^{ < }}$ is the subgraph of $G$ induced by ${V}_{j}^{ < }$ . Since $\left| {V}_{j}^{ < }\right|  \leq  t - 1$ ,we already know how to build the IJ-heap on ${G}_{{V}_{j}^{ < }}$ by the inductive assumption in Section 6.3.

回顾（如6.1节所定义）${G}_{{V}_{j}^{ < }}$是由${V}_{j}^{ < }$所诱导的$G$的子图。由于$\left| {V}_{j}^{ < }\right|  \leq  t - 1$，根据6.3节的归纳假设，我们已经知道如何在${G}_{{V}_{j}^{ < }}$上构建IJ堆。

- (if ${h}_{2} > 0$ ) for each $j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$ ,build an IJ-heap on the instance vector ${\mathbf{R}}_{{V}_{j}^{ > }}$ under the topology ${G}_{{V}_{j}^{ > }}$ ,where for each $i \in  {V}_{j}^{ > }$ :

- （如果${h}_{2} > 0$）对于每个$j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$，在拓扑结构${G}_{{V}_{j}^{ > }}$下的实例向量${\mathbf{R}}_{{V}_{j}^{ > }}$上构建一个IJ堆，其中对于每个$i \in  {V}_{j}^{ > }$：

$$
{\mathbf{R}}_{{V}_{j}^{ > }} = \text{the projection of}{\mathbf{R}}^{\complement }\text{in}{V}_{j}^{ > }\text{.}
$$

- for each $i \in  {V}^{ = }$ ,build a structure on ${\operatorname{stab}}_{i}^{ = }\left( u\right)$ to support:

- 对于每个$i \in  {V}^{ = }$，在${\operatorname{stab}}_{i}^{ = }\left( u\right)$上构建一个结构以支持：

Given real values ${\lambda }_{1},{\lambda }_{2}$ satisfying ${\lambda }_{1} \leq  \operatorname{key}\left( u\right)  \leq  {\lambda }_{2}$ and arbitrary real values $a,b$ ,this operation finds the interval with the smallest left endpoint,among all the intervals $r$ in ${\operatorname{stab}}_{i}^{ = }\left( u\right)$ such that $r$ (i) covers the entire interval $\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$ ,and (ii) $r$ intersects with $\left( {-\infty ,a\rbrack \text{and}\lbrack b,\infty }\right)$ .

给定满足${\lambda }_{1} \leq  \operatorname{key}\left( u\right)  \leq  {\lambda }_{2}$的实数值${\lambda }_{1},{\lambda }_{2}$和任意实数值$a,b$，此操作会在${\operatorname{stab}}_{i}^{ = }\left( u\right)$中的所有区间$r$中找到左端点最小的区间，使得（i）$r$覆盖整个区间$\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$，并且（ii）$r$与$\left( {-\infty ,a\rbrack \text{and}\lbrack b,\infty }\right)$相交。

Note that this operation can be supported by a 4D range-min structure of Lemma 7. To see why,let us convert $r = \left\lbrack  {x,y}\right\rbrack$ to a 4D rectangle $\lbrack x,\infty ) \times  ( - \infty ,y\rbrack  \times  r \times  r$ . For any ${\lambda }_{1},{\lambda }_{2},a$ ,and $b$ ,we know: $r$ satisfies the two conditions aforementioned,if and only if $\lbrack x,\infty ) \times  \left( {-\infty ,y}\right\rbrack   \times  r \times  r$ intersects with the $4\mathrm{D}$ rectangle $\left( {-\infty ,{\lambda }_{1}\rbrack  \times  \left\lbrack  {{\lambda }_{2},\infty }\right\rbrack   \times  \left( {-\infty ,a}\right) \times \lbrack b,\infty }\right)$ .

注意，此操作可由引理7中的四维范围最小值结构支持。为了理解原因，让我们将$r = \left\lbrack  {x,y}\right\rbrack$转换为一个四维矩形$\lbrack x,\infty ) \times  ( - \infty ,y\rbrack  \times  r \times  r$。对于任意的${\lambda }_{1},{\lambda }_{2},a$和$b$，我们知道：$r$满足上述两个条件，当且仅当$\lbrack x,\infty ) \times  \left( {-\infty ,y}\right\rbrack   \times  r \times  r$与$4\mathrm{D}$矩形$\left( {-\infty ,{\lambda }_{1}\rbrack  \times  \left\lbrack  {{\lambda }_{2},\infty }\right\rbrack   \times  \left( {-\infty ,a}\right) \times \lbrack b,\infty }\right)$相交。

By the inductive assumption in Section 6.3,the combination structure uses $\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$ space overall, and can be updated in $\widetilde{O}\left( 1\right)$ amortized time per insertion/deletion in ${\mathbf{R}}^{\mathrm{C}}\left\lbrack  i\right\rbrack$ ,for any $i \in  \left\lbrack  {1,t}\right\rbrack$ .

根据6.3节中的归纳假设，组合结构总体上使用$\widetilde{O}\left( \left| {\mathcal{T}}_{u}\right| \right)$的空间，并且对于任意的$i \in  \left\lbrack  {1,t}\right\rbrack$，在${\mathbf{R}}^{\mathrm{C}}\left\lbrack  i\right\rbrack$中每次插入/删除操作的均摊时间为$\widetilde{O}\left( 1\right)$。

Query. We consider only min-IJ queries on ${\mathbf{R}}^{@}$ under $G$ because max-IJ queries are symmetric. Our algorithm answers a min-IJ query with constraint vector $\mathbf{q}$ in five steps.

查询。由于最大IJ查询是对称的，我们仅考虑在$G$条件下对${\mathbf{R}}^{@}$的最小IJ查询。我们的算法分五步回答带有约束向量$\mathbf{q}$的最小IJ查询。

Step 1: Skip this step if ${h}_{1} = 0$ . Otherwise,for each $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ ,construct a constraint vector:

步骤1：如果${h}_{1} = 0$，则跳过此步骤。否则，对于每个$j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$，构造一个约束向量：

$$
{\mathbf{q}}_{{V}_{j}^{ < }} = \text{the projection of}\mathbf{q}\text{in}{V}_{j}^{ < }\text{.}
$$

Then,perform a max-IJ query with this vector on ${\mathbf{R}}_{{V}_{j}^{ < }}$ under ${G}_{{V}_{j}^{ < }}$ (an IJ-heap has been built for this purpose). Denote the data vector retrieved as ${\mathbf{r}}_{{V}_{i}}^{\max }$ ; if the vector is null,we terminate and return nothing.

然后，在${G}_{{V}_{j}^{ < }}$条件下使用此向量对${\mathbf{R}}_{{V}_{j}^{ < }}$执行最大IJ查询（为此已构建了一个IJ堆）。将检索到的数据向量记为${\mathbf{r}}_{{V}_{i}}^{\max }$；如果该向量为空，则终止并返回空值。

Construct a data vector ${\mathbf{r}}_{{V}^{ < }}^{\max }$ by setting for each $i \in  {V}^{ < }$

通过为每个$i \in  {V}^{ < }$设置来构造一个数据向量${\mathbf{r}}_{{V}^{ < }}^{\max }$

$$
{\mathbf{r}}_{{V}^{ < }}^{\max }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}_{j}^{ < }}^{\max }\left\lbrack  i\right\rbrack  
$$

where $j$ is the only integer in $\left\lbrack  {1,{h}_{1}}\right\rbrack$ such that $i \in  {V}_{j}^{ < }$ .

其中$j$是$\left\lbrack  {1,{h}_{1}}\right\rbrack$中唯一满足$i \in  {V}_{j}^{ < }$的整数。

Step 2: Skip this step if ${h}_{2} = 0$ . Otherwise,for each $j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$ ,we will issue a min-IJ query recursively on the subjoin defined by ${G}_{{V}_{j}^{ > }}$ . Construct a constraint vector:

步骤2：如果${h}_{2} = 0$，则跳过此步骤。否则，对于每个$j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$，我们将对由${G}_{{V}_{j}^{ > }}$定义的子连接递归地发出最小IJ查询。构造一个约束向量：

$$
{\mathbf{q}}_{{V}_{j}^{ > }} = \text{the projection of}\mathbf{q}\text{in}{V}_{j}^{ > }\text{.}
$$

Perform a min-IJ query with this vector on ${\mathbf{R}}_{{V}_{j}^{ > }}$ under ${G}_{{V}_{j}^{ > }}$ . Denote the data vector retrieved as ${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }$ ; if the vector is null,we terminate and return nothing.

在${G}_{{V}_{j}^{ > }}$条件下使用此向量对${\mathbf{R}}_{{V}_{j}^{ > }}$执行最小IJ查询。将检索到的数据向量记为${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }$；如果该向量为空，则终止并返回空值。

Construct a data vector ${\mathbf{r}}_{{V}^{ > }}^{\min }$ by setting for each $i \in  {V}^{ > }$

通过为每个$i \in  {V}^{ > }$设置来构造一个数据向量${\mathbf{r}}_{{V}^{ > }}^{\min }$

$$
{\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  i\right\rbrack  
$$

where $j$ is the only integer in $\left\lbrack  {1,{h}_{2}}\right\rbrack$ such that $i \in  {V}_{j}^{ > }$ .

其中$j$是$\left\lbrack  {1,{h}_{2}}\right\rbrack$中唯一满足$i \in  {V}_{j}^{ > }$的整数。

Step 3: For each $i \in  {V}^{ = }$ ,we will retrieve the interval ${r}_{i}^{\min }$ with smallest left endpoint,from all the intervals in ${\operatorname{stab}}_{i}^{ = }\left( u\right)$ that (i) intersect with $\left( {-\infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack \text{and}\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty }\right)$ ,and (ii) contain a range $\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$ ,where ${\lambda }_{1}$ and ${\lambda }_{2}$ are decided in the following manner.

步骤3：对于每个$i \in  {V}^{ = }$，我们将从${\operatorname{stab}}_{i}^{ = }\left( u\right)$中的所有区间里检索左端点最小的区间${r}_{i}^{\min }$，这些区间需满足：(i) 与$\left( {-\infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack \text{and}\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty }\right)$相交；(ii) 包含一个范围$\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$，其中${\lambda }_{1}$和${\lambda }_{2}$按以下方式确定。

Denote by ${N}^{ < }\left( i\right)$ the set of vertices in ${V}^{ < }$ that are adjacent to $i$ in $G$ . If ${N}^{ < }\left( i\right)$ is empty,then set ${\lambda }_{1} = \operatorname{key}\left( u\right)$ . Otherwise,set ${\lambda }_{1}$ to the smallest right endpoint of the intervals in the following set:

用${N}^{ < }\left( i\right)$表示在$G$中与$i$相邻的${V}^{ < }$中的顶点集。如果${N}^{ < }\left( i\right)$为空，则令${\lambda }_{1} = \operatorname{key}\left( u\right)$。否则，将${\lambda }_{1}$设为以下集合中区间的最小右端点：

$$
\left\{  {{\mathbf{r}}_{{V}^{ < }}^{\max }\left\lbrack  {i}^{\prime }\right\rbrack   \mid  {i}^{\prime } \in  {N}^{ < }\left( i\right) }\right\}  . \tag{5}
$$

Conversely,denote by ${N}^{ > }\left( i\right)$ the set of vertices in ${V}^{ > }$ that are adjacent to $i$ in $G$ . If ${N}^{ > }\left( i\right)$ is empty,then set ${\lambda }_{2} = \operatorname{key}\left( u\right)$ . Otherwise,set ${\lambda }_{2}$ to the largest left endpoint of the intervals in the following set:

相反，用${N}^{ > }\left( i\right)$表示在$G$中与$i$相邻的${V}^{ > }$中的顶点集。如果${N}^{ > }\left( i\right)$为空，则令${\lambda }_{2} = \operatorname{key}\left( u\right)$。否则，将${\lambda }_{2}$设为以下集合中区间的最大左端点：

$$
\left\{  {{\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack   \mid  {i}^{\prime } \in  {N}^{ > }\left( i\right) }\right\}  . \tag{6}
$$

Now that ${\lambda }_{1},{\lambda }_{2}$ are ready,we use the $4\mathrm{D}$ range min structure (of the combination structure) to find ${r}_{i}^{\min }$ . If ${r}_{i}^{\min }$ does not exist,we terminate and return nothing. Otherwise,proceed to the next step.

现在${\lambda }_{1},{\lambda }_{2}$已准备好，我们使用（组合结构的）$4\mathrm{D}$范围最小值结构来查找${r}_{i}^{\min }$。如果${r}_{i}^{\min }$不存在，我们终止操作并返回空。否则，进入下一步。

Step 4: For each $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ ,we will issue yet another min-IJ query on ${\mathbf{R}}_{{V}_{j}^{ < }}$ . First,construct a constrain vector:

步骤4：对于每个$j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$，我们将在${\mathbf{R}}_{{V}_{j}^{ < }}$上再进行一次最小IJ查询。首先，构建一个约束向量：

$$
{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime } = \text{the projection of}\mathbf{q}\text{on}{V}_{j}^{ < }\text{.}
$$

Consider each $i \in  {V}_{i}^{ < }$ in turn. Denote by ${N}^{ = }\left( i\right)$ the set of vertices in ${V}^{ = }$ that are adjacent to $i$ in $G$ . If ${N}^{ = }\left( i\right)  = \varnothing$ ,then the current ${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }\left\lbrack  i\right\rbrack$ is finalized. Otherwise,we update ${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }\left\lbrack  i\right\rbrack$ . $b$ to the maximum between $\mathbf{q}\left\lbrack  i\right\rbrack  .b$ and ${\lambda }_{3}$ ,where ${\lambda }_{3}$ is the largest left endpoint of the intervals in the following set:

依次考虑每个$i \in  {V}_{i}^{ < }$。用${N}^{ = }\left( i\right)$表示在$G$中与$i$相邻的${V}^{ = }$中的顶点集。如果${N}^{ = }\left( i\right)  = \varnothing$，则当前的${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }\left\lbrack  i\right\rbrack$确定。否则，我们将${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }\left\lbrack  i\right\rbrack$.$b$更新为$\mathbf{q}\left\lbrack  i\right\rbrack  .b$和${\lambda }_{3}$中的最大值，其中${\lambda }_{3}$是以下集合中区间的最大左端点：

$$
\left\{  {{r}_{{i}^{\prime }}^{\min } \mid  {i}^{\prime } \in  {N}^{ = }\left( i\right) }\right\}  . \tag{7}
$$

Now,perform a min-IJ query on ${\mathbf{R}}_{{V}_{j}^{ < }}$ under ${G}_{{V}_{j}^{ < }}$ with ${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }$ . Denote the data vector retrieved as ${\mathbf{r}}_{{V}_{j}^{ < }}^{\min }$ ; if the vector is null,we terminate and return nothing.

现在，在${G}_{{V}_{j}^{ < }}$的约束下，对${\mathbf{R}}_{{V}_{j}^{ < }}$使用${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }$进行最小IJ查询。将检索到的数据向量记为${\mathbf{r}}_{{V}_{j}^{ < }}^{\min }$；如果该向量为空，我们终止操作并返回空。

Step 5: Return a data vector $\mathbf{\rho }$ where for each $i \in  \left\lbrack  {1,t}\right\rbrack$ :

步骤5：返回一个数据向量$\mathbf{\rho }$，其中对于每个$i \in  \left\lbrack  {1,t}\right\rbrack$：

- $\mathbf{\rho }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack$ if $i \in  {V}^{ < }$ ;

- $\mathbf{\rho }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack$ 如果 $i \in  {V}^{ < }$；

- $\mathbf{\rho }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$ if $i \in  {V}^{ = }$ ;

- $\mathbf{\rho }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$ 如果 $i \in  {V}^{ = }$；

- $\mathbf{\rho }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack$ if $i \in  {V}^{ > }$ .

- $\mathbf{\rho }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack$ 如果 $i \in  {V}^{ > }$。

Each of the above steps finishes in $\widetilde{O}\left( 1\right)$ time,by the inductive assumption in Section 6.3. The overall query time is therefore $\widetilde{O}\left( 1\right)$ . Deferring the correctness proof of the query algorithm to Appendix D, we have now established Lemma 14.

根据6.3节中的归纳假设，上述每个步骤都能在 $\widetilde{O}\left( 1\right)$ 时间内完成。因此，总的查询时间为 $\widetilde{O}\left( 1\right)$。将查询算法的正确性证明推迟到附录D，我们现在已经证明了引理14。

### 6.5 Structures for Categories 1 and 2

### 6.5 类别1和类别2的结构

For every node $u$ in the interval tree $\mathcal{T}$ ,we build the structure of Lemma 14 on each combination of $u$ . All these structures occupy $\widetilde{O}\left( n\right)$ space in total.

对于区间树 $\mathcal{T}$ 中的每个节点 $u$，我们在 $u$ 的每个组合上构建引理14的结构。所有这些结构总共占用 $\widetilde{O}\left( n\right)$ 的空间。

We now resume our discussion in Section 6.3 and explain how to find the left-smallest data vector $\mathbf{r}$ from Categories 1 and 2.

现在我们继续6.3节的讨论，解释如何从类别1和类别2中找到最左侧的数据向量 $\mathbf{r}$。

Category 1. Let $u$ be a node on ${\Pi }_{1}$ or ${\Pi }_{2}$ . A data vector that hinges on $u$ must belong to one combination of $u$ . We issue a min-IJ query with the constraint vector $\mathbf{q}$ on all the ${3}^{t} - {2}^{t} = O\left( 1\right)$ combination structures,and find the left-smallest data vector $\mathbf{r}$ from the data vectors fetched by those queries. It is guaranteed that $\mathbf{r}$ must be the left-smallest among all the data vectors in $J\left( {G,\mathcal{R},\mathbf{q}}\right)$ that hinge on $u$ . By Lemma 14,this takes $\widetilde{O}\left( 1\right)$ time.

类别1。设 $u$ 是 ${\Pi }_{1}$ 或 ${\Pi }_{2}$ 上的一个节点。依赖于 $u$ 的数据向量必须属于 $u$ 的某一个组合。我们对所有 ${3}^{t} - {2}^{t} = O\left( 1\right)$ 组合结构发出带有约束向量 $\mathbf{q}$ 的最小IJ查询，并从这些查询获取的数据向量中找到最左侧的数据向量 $\mathbf{r}$。可以保证 $\mathbf{r}$ 一定是 $J\left( {G,\mathcal{R},\mathbf{q}}\right)$ 中所有依赖于 $u$ 的数据向量中最左侧的。根据引理14，这需要 $\widetilde{O}\left( 1\right)$ 时间。

As ${\Pi }_{1}$ and ${\Pi }_{2}$ have $\widetilde{O}\left( 1\right)$ nodes only,the left-smallest data vector of Category 1 can be found in $\widetilde{O}\left( 1\right)$ time in total.

由于 ${\Pi }_{1}$ 和 ${\Pi }_{2}$ 只有 $\widetilde{O}\left( 1\right)$ 个节点，因此总共可以在 $\widetilde{O}\left( 1\right)$ 时间内找到类别1的最左侧数据向量。

Category 2. This category applies only if $b \leq  a$ . Any node $u$ that needs to be considered must be have its slab $\sigma \left( u\right)$ covered by $\left\lbrack  {b,a}\right\rbrack$ . We pre-compute at each node $u$ the "best answer" that $u$ has to offer. The pre-computed information is organized in such a way that, when a min-IJ query comes, "the best of the best" answers from the relevant nodes can be retrieved efficiently.

类别2。只有当 $b \leq  a$ 时，此类别才适用。任何需要考虑的节点 $u$ 必须使其平板 $\sigma \left( u\right)$ 被 $\left\lbrack  {b,a}\right\rbrack$ 覆盖。我们在每个节点 $u$ 上预先计算 $u$ 能提供的“最佳答案”。预先计算的信息以这样的方式组织：当一个最小IJ查询到来时，可以有效地检索相关节点的“最佳中的最佳”答案。

To implement the above idea,let us define the local minimum of $u$ ,as the left-smallest among all data vectors $\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$ in $J\left( {G,\mathcal{R}}\right)$ that hinge on $u$ . The local minimum must belong to one of the combinations of $u$ and can be found in $\widetilde{O}\left( 1\right)$ time as follows:

为了实现上述想法，让我们定义 $u$ 的局部最小值，即 $J\left( {G,\mathcal{R}}\right)$ 中所有依赖于 $u$ 的数据向量 $\mathbf{r} = \left( {{r}_{1},\ldots ,{r}_{t}}\right)$ 中最左侧的。局部最小值必须属于 $u$ 的某一个组合，并且可以按如下方式在 $\widetilde{O}\left( 1\right)$ 时间内找到：

- Create a dummy constraint vector ${\mathbf{q}}^{\prime }$ where ${\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack  .a = \infty$ and $\mathbf{q}\left\lbrack  i\right\rbrack  .b =  - \infty$ for all $i \in  \left\lbrack  {1,t}\right\rbrack$ .

- 创建一个虚拟约束向量 ${\mathbf{q}}^{\prime }$，其中对于所有 $i \in  \left\lbrack  {1,t}\right\rbrack$ 有 ${\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack  .a = \infty$ 和 $\mathbf{q}\left\lbrack  i\right\rbrack  .b =  - \infty$。

- Issue a min-IJ query on each of the ${3}^{t} - {2}^{t}$ combination structures of $u$ using ${\mathbf{q}}^{\prime }$ .

- 使用 ${\mathbf{q}}^{\prime }$ 对 $u$ 的每个 ${3}^{t} - {2}^{t}$ 组合结构发出最小 IJ 查询。

- Set the local minimum to the left-smallest of all the data vectors fetched by the queries at the previous step.

- 将局部最小值设置为上一步查询获取的所有数据向量中最左侧的最小值。

We now use a 2D range-min structure of Lemma 7 — denoted as ${\mathcal{T}}^{\prime } -$ to manage the slabs of all the nodes in $\mathcal{T}$ . Specifically,given a node $u$ with slab $\sigma \left( u\right)  = \left\lbrack  {x,y}\right\rbrack$ ,we create a (degenerated) 2D rectangle $\left\lbrack  {x,x}\right\rbrack   \times  \left\lbrack  {y,y}\right\rbrack$ ,treating the local minimum of $u$ as the rectangle’s "weight". A weight $\mathbf{r}$ is smaller than another ${\mathbf{r}}^{\prime }$ if $\mathbf{r}$ is left-smaller than ${\mathbf{r}}^{\prime }$ . By Lemma 7, ${\mathcal{T}}^{\prime }$ requires only $\widetilde{O}\left( n\right)$ space.

我们现在使用引理 7 中的二维范围最小值结构（表示为 ${\mathcal{T}}^{\prime } -$）来管理 $\mathcal{T}$ 中所有节点的条带。具体来说，给定一个带有条带 $\sigma \left( u\right)  = \left\lbrack  {x,y}\right\rbrack$ 的节点 $u$，我们创建一个（退化的）二维矩形 $\left\lbrack  {x,x}\right\rbrack   \times  \left\lbrack  {y,y}\right\rbrack$，将 $u$ 的局部最小值视为矩形的“权重”。如果权重 $\mathbf{r}$ 在最左侧小于另一个权重 ${\mathbf{r}}^{\prime }$，则 $\mathbf{r}$ 小于 ${\mathbf{r}}^{\prime }$。根据引理 7，${\mathcal{T}}^{\prime }$ 仅需要 $\widetilde{O}\left( n\right)$ 的空间。

This completes the description of our IJ-heap,whose space consumption is $\widetilde{O}\left( n\right)$ overall. Given a min-IJ query with constraint vector $\mathbf{q}$ (on $\mathcal{R}$ under $G$ ),we find the left-smallest data vector of Category 2 as follows. First,compute the values of $a$ and $b$ using (2) and (3). If $a < b$ ,ignore this category. Otherwise,perform a 2D range-min query on ${\mathcal{T}}^{\prime }$ using the rectangle $\left\lbrack  {b,\infty )\times ( - \infty ,a}\right\rbrack$ . The result of the range-min query is the left-smallest local minimum of the nodes whose slabs $\sigma \left( u\right)  = \left\lbrack  {x,y}\right\rbrack$ are covered by $\left\lbrack  {b,a}\right\rbrack$ ,and is what we look for in Category 2 .

至此，我们完成了对 IJ 堆的描述，其总体空间消耗为 $\widetilde{O}\left( n\right)$。给定一个带有约束向量 $\mathbf{q}$（在 $\mathcal{R}$ 下，受 $G$ 约束）的最小 IJ 查询，我们按如下方式找到类别 2 中最左侧的最小数据向量。首先，使用 (2) 和 (3) 计算 $a$ 和 $b$ 的值。如果 $a < b$，则忽略此类别。否则，使用矩形 $\left\lbrack  {b,\infty )\times ( - \infty ,a}\right\rbrack$ 对 ${\mathcal{T}}^{\prime }$ 执行二维范围最小值查询。范围最小值查询的结果是其条带 $\sigma \left( u\right)  = \left\lbrack  {x,y}\right\rbrack$ 被 $\left\lbrack  {b,a}\right\rbrack$ 覆盖的节点的最左侧局部最小值，这就是我们在类别 2 中要查找的内容。

Therefore,we now conclude that a min-IJ query on $\mathcal{R}$ under $G$ can be answered in $\widetilde{O}\left( 1\right)$ time.

因此，我们现在得出结论，在 $\mathcal{R}$ 上受 $G$ 约束的最小 IJ 查询可以在 $\widetilde{O}\left( 1\right)$ 时间内得到解答。

### 6.6 Update

### 6.6 更新

The update algorithm is fairly straightforward, utilizing the result in Lemma 5.

更新算法相当直接，利用了引理 5 中的结果。

For any node $u$ in the interval tree $\mathcal{T}$ ,its secondary structure ${\Gamma }_{u}$ involves only $O\left( 1\right)$ combination structures of Lemma 14. Whenever an interval is insert/deleted in any of ${\operatorname{stab}}^{ < }\left( u\right) ,\operatorname{stab}\left( u\right)$ ,or ${\operatorname{stab}}^{ > }\left( u\right)$ ,we update the ${\Gamma }_{u}$ using Lemma 14 in $\widetilde{O}\left( 1\right)$ time. This means that all the functions ${f}_{1}\left( n\right) ,{f}_{2}\left( n\right) ,{f}_{3}\left( n\right)$ ,and ${f}_{4}\left( n\right)$ in Lemma 5 are $\widetilde{O}\left( 1\right)$ . Thus,Lemma 5 tells us that $\mathcal{T}$ can be updated in $\widetilde{O}\left( 1\right)$ amortized time per insertion/deletion in any of ${R}_{1},\ldots ,{R}_{t}$ .

对于区间树 $\mathcal{T}$ 中的任何节点 $u$，其二级结构 ${\Gamma }_{u}$ 仅涉及引理 14 中的 $O\left( 1\right)$ 组合结构。每当在 ${\operatorname{stab}}^{ < }\left( u\right) ,\operatorname{stab}\left( u\right)$ 或 ${\operatorname{stab}}^{ > }\left( u\right)$ 中的任何一个中插入/删除一个区间时，我们使用引理 14 在 $\widetilde{O}\left( 1\right)$ 时间内更新 ${\Gamma }_{u}$。这意味着引理 5 中的所有函数 ${f}_{1}\left( n\right) ,{f}_{2}\left( n\right) ,{f}_{3}\left( n\right)$ 和 ${f}_{4}\left( n\right)$ 的时间复杂度都是 $\widetilde{O}\left( 1\right)$。因此，引理 5 告诉我们，在 ${R}_{1},\ldots ,{R}_{t}$ 中的任何一个进行插入/删除操作时，$\mathcal{T}$ 可以在每次操作的平摊时间 $\widetilde{O}\left( 1\right)$ 内完成更新。

Finally,the cost of updating ${\mathcal{T}}^{\prime }$ can be piggybacked on the cost of updating $\mathcal{T}$ . Specifically,for every node $u$ in ${\mathcal{T}}^{\prime }$ that is affected by an insertion/deletion in $\mathcal{T}$ ,we re-compute its local minimum in $\widetilde{O}\left( 1\right)$ time in the way described in Section 6.5,and update ${\mathcal{T}}^{\prime }$ accordingly in $\widetilde{O}\left( 1\right)$ amortized time.

最后，更新${\mathcal{T}}^{\prime }$的成本可以“搭便车”到更新$\mathcal{T}$的成本中。具体而言，对于${\mathcal{T}}^{\prime }$中受$\mathcal{T}$的插入/删除操作影响的每个节点$u$，我们按照6.5节中描述的方式在$\widetilde{O}\left( 1\right)$时间内重新计算其在$\widetilde{O}\left( 1\right)$中的局部最小值，并在$\widetilde{O}\left( 1\right)$的均摊时间内相应地更新${\mathcal{T}}^{\prime }$。

The overall update time of our IJ-heap is therefore $\widetilde{O}\left( 1\right)$ amortized. This completes the proof of Lemma 11 and, hence, also the proof of Theorem 4.

因此，我们的IJ堆的总体更新时间为$\widetilde{O}\left( 1\right)$均摊时间。这就完成了引理11的证明，进而也完成了定理4的证明。

## 7 Conclusions

## 7 结论

Given (i) $t$ sets of $d$ -dimensional rectangles ${R}_{1},{R}_{2},\ldots ,{R}_{t}$ and (ii) a connected undirected graph $G$ (called a topology graph) on vertices $\{ 1,2,\ldots ,t\}$ ,an intersection join returns all $\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in$ ${R}_{1} \times  \ldots  \times  {R}_{t}$ satisfying the condition that ${r}_{i} \cap  {r}_{j} \neq  \varnothing$ for all $i,j$ such that $G$ has an edge between vertices $i$ and $j$ . The paper investigates the question "when do feasible structures exist for intersection joins?",where a feasible structure needs to use $\widetilde{O}\left( n\right)$ space (note: $n = \mathop{\sum }\limits_{{i = 1}}^{t}\left| {R}_{i}\right|$ ),supports an insertion/deletion in $\widetilde{O}\left( 1\right)$ amortized time,and permits the join result to be reported with an $\widetilde{O}\left( 1\right)$ -time delay. Subject to the OMv-conjecture,we have answered the question by showing that a feasible structure exists if and only if $t = 2$ (binary joins,regardless of $d$ ) or $d = 1$ (1D joins, regardless of $t$ ).

给定 (i) $t$组$d$维矩形${R}_{1},{R}_{2},\ldots ,{R}_{t}$，以及 (ii) 顶点为$\{ 1,2,\ldots ,t\}$的连通无向图$G$（称为拓扑图），交集连接会返回所有满足以下条件的$\left( {{r}_{1},\ldots ,{r}_{t}}\right)  \in$${R}_{1} \times  \ldots  \times  {R}_{t}$：对于所有使得$G$在顶点$i$和$j$之间存在边的$i,j$，都有${r}_{i} \cap  {r}_{j} \neq  \varnothing$。本文研究了“交集连接何时存在可行结构？”这一问题，其中可行结构需要使用$\widetilde{O}\left( n\right)$的空间（注意：$n = \mathop{\sum }\limits_{{i = 1}}^{t}\left| {R}_{i}\right|$），支持在$\widetilde{O}\left( 1\right)$的均摊时间内进行插入/删除操作，并允许以$\widetilde{O}\left( 1\right)$的时间延迟报告连接结果。在OMv猜想的条件下，我们通过证明当且仅当$t = 2$（二元连接，与$d$无关）或$d = 1$（一维连接，与$t$无关）时存在可行结构，回答了该问题。

A natural (and promising) direction for future research is to study how to relax the feasibility requirements to support intersection joins under updates in the scenarios where feasible structures do not exist,namely, $t \geq  3$ and $d \geq  2$ (multidimensional multiway joins). An interesting question is: if the update time has to be $\widetilde{O}\left( 1\right)$ amortized,what is the smallest delay (in join result reporting) achievable? An equally interesting question lies in the other extreme: if the delay has to be $\widetilde{O}\left( 1\right)$ , what is the fastest amortized update time possible? Settling these questions would provide helpful hints towards resolving the ultimate puzzle: what is the precise tradeoff between the amortized update cost and the delay? We suspect that joins with different topology graphs $G$ may exhibit various tradeoffs.

未来研究的一个自然（且有前景）的方向是研究如何放宽可行性要求，以在不存在可行结构的场景（即$t \geq  3$和$d \geq  2$，多维多路连接）下支持更新时的交集连接。一个有趣的问题是：如果更新时间必须为$\widetilde{O}\left( 1\right)$均摊时间，那么报告连接结果时可实现的最小延迟是多少？另一个同样有趣的问题则处于另一个极端：如果延迟必须为$\widetilde{O}\left( 1\right)$，那么最快的均摊更新时间是多少？解决这些问题将为解决最终难题提供有用的线索：均摊更新成本和延迟之间的精确权衡是什么？我们推测，具有不同拓扑图$G$的连接可能会呈现出不同的权衡关系。

## Acknowledgements

## 致谢

This work was supported in part by GRF projects 14207820, 16201318, 16201819, and 16205420 from HKRGC.

这项工作部分得到了香港研究资助局（HKRGC）的研资局项目14207820、16201318、16201819和16205420的支持。

## Appendix

## 附录

## A Optimality of Theorem 1

## A 定理1的最优性

It suffices to show that $\Omega \left( {\log n}\right)$ time is needed to handle an update.

只需证明处理一次更新需要$\Omega \left( {\log n}\right)$的时间。

We achieve the purpose via a reduction from predecessor search. In that problem, we are given a set $P$ of real values,and want to answer the following queries efficiently: given an arbitrary real value $q$ ,find its predecessor in $P$ ,namely,the largest value in $P$ that is smaller than or equal to $q$ . Under the comparison model,this query requires $\Omega \left( {\log n}\right)$ time to solve,regardless of the preprocessing on $P$ .

我们通过从前驱搜索问题进行归约来实现这一目的。在前驱搜索问题中，给定一组实数值$P$，我们希望高效地回答以下查询：给定任意实数值$q$，找出它在$P$中的前驱，即$P$中小于或等于$q$的最大数值。在比较模型下，无论对$P$进行何种预处理，解决此查询都需要$\Omega \left( {\log n}\right)$的时间。

Suppose that for intersection joins with $d = 1$ and $t = 2$ ,we are given a structure that ensures constant delay enumeration,and can be updated in $U\left( n\right)$ time. Then,we can deploy the structure to answer predecessor search in $O\left( {U\left( n\right) }\right)$ time as explained below.

假设对于涉及$d = 1$和$t = 2$的交集连接，我们有一个能确保常量延迟枚举且可在$U\left( n\right)$时间内更新的结构。那么，我们可以如下面所述，利用该结构在$O\left( {U\left( n\right) }\right)$时间内回答前驱搜索问题。

In preprocessing,convert $P$ into an interval set

在预处理阶段，将$P$转换为一个区间集合

$$
{R}_{1} = \{ \left\lbrack  {\operatorname{pre}\left( x\right) ,x}\right\rbrack   \mid  x \in  P\} 
$$

where $\operatorname{pre}\left( x\right)$ is the value in $P$ immediately preceding $x$ (if $x$ is the minimum in $P$ ,then $\operatorname{pre}\left( x\right)  =  - \infty$ ). Create an intersection join structure $T$ on ${R}_{1}$ and an empty ${R}_{2}$ . Given a predecessor query $q$ on $P$ ,insert a degenerated interval $\left\lbrack  {q,q}\right\rbrack$ into ${R}_{2}$ ,and then use $T$ to enumerate the join result. Note that the join result contains only one tuple $\left\lbrack  {\operatorname{pre}\left( q\right) ,\operatorname{suc}\left( q\right) }\right\rbrack$ if $q \notin  P$ ,where $\operatorname{suc}\left( q\right)$ is the value in $P$ immediately succeeding $q$ (or $\infty$ if no such value exists). If $q \in  P$ ,then the join result contains 2 tuples: $\left\lbrack  {\operatorname{pre}\left( q\right) ,q}\right\rbrack$ and $\left\lbrack  {q,\operatorname{suc}\left( q\right) }\right\rbrack$ . In either case,the predecessor of $q$ can be found in $O\left( 1\right)$ time.

其中$\operatorname{pre}\left( x\right)$是$P$中紧接在$x$之前的值（如果$x$是$P$中的最小值，那么$\operatorname{pre}\left( x\right)  =  - \infty$）。在${R}_{1}$和一个空的${R}_{2}$上创建一个交集连接结构$T$。给定一个关于$P$的前驱查询$q$，将一个退化区间$\left\lbrack  {q,q}\right\rbrack$插入到${R}_{2}$中，然后使用$T$来枚举连接结果。注意，如果$q \notin  P$，连接结果仅包含一个元组$\left\lbrack  {\operatorname{pre}\left( q\right) ,\operatorname{suc}\left( q\right) }\right\rbrack$，其中$\operatorname{suc}\left( q\right)$是$P$中紧接在$q$之后的值（如果不存在这样的值，则为$\infty$）。如果$q \in  P$，那么连接结果包含2个元组：$\left\lbrack  {\operatorname{pre}\left( q\right) ,q}\right\rbrack$和$\left\lbrack  {q,\operatorname{suc}\left( q\right) }\right\rbrack$。在这两种情况下，都可以在$O\left( 1\right)$时间内找到$q$的前驱。

The query time is $O\left( {U\left( n\right) }\right)$ ,which implies $U\left( n\right)  = \Omega \left( {\log n}\right)$ .

查询时间为$O\left( {U\left( n\right) }\right)$，这意味着$U\left( n\right)  = \Omega \left( {\log n}\right)$。

## B “Traditional” Incremental View Maintenance

## B “传统”增量视图维护

As mentioned in Section 1.2, a more conventional form of maintenance aims to report the delta changes in the join result. In our intersection-join context where ${R}_{1},\ldots ,{R}_{t}$ are the input sets of intervals, the objectives are two fold:

如1.2节所述，一种更传统的维护形式旨在报告连接结果中的增量变化。在我们的交集连接场景中，其中${R}_{1},\ldots ,{R}_{t}$是输入的区间集合，目标有两个方面：

1. When an interval $r$ is inserted into ${R}_{i}$ (for some $i \in  \left\lbrack  {1,t}\right\rbrack$ ),we must enumerate all the new result tuples of the intersection join (i.e.,those involving $r$ ) with an $\widetilde{O}\left( 1\right)$ delay;

1. 当一个区间$r$被插入到${R}_{i}$中（对于某个$i \in  \left\lbrack  {1,t}\right\rbrack$）时，我们必须以$\widetilde{O}\left( 1\right)$的延迟枚举交集连接的所有新结果元组（即，那些涉及$r$的元组）；

2. When an interval $r$ is deleted ${R}_{i}$ (for some $i \in  \left\lbrack  {1,t}\right\rbrack$ ),we must enumerate all the disappearing result tuples of the intersection join (i.e.,those involving $r$ ) with an $\widetilde{O}\left( 1\right)$ delay.

2. 当一个区间$r$从${R}_{i}$中被删除（对于某个$i \in  \left\lbrack  {1,t}\right\rbrack$）时，我们必须以$\widetilde{O}\left( 1\right)$的延迟枚举交集连接的所有消失的结果元组（即，那些涉及$r$的元组）。

The feasible structures formulated in this paper can be combined with a leave-one-out approach to achieve the above objectives. Specifically,we maintain $t$ feasible structures,such that the $i$ -th $\left( {i \in  \left\lbrack  {1,t}\right\rbrack  }\right)$ one is built on:

本文提出的可行结构可以与留一法相结合以实现上述目标。具体来说，我们维护$t$个可行结构，使得第$i$个$\left( {i \in  \left\lbrack  {1,t}\right\rbrack  }\right)$结构基于以下内容构建：

$$
{R}_{1},\ldots ,{R}_{i - 1},\varnothing ,{R}_{i + 1},\ldots ,{R}_{t}\text{.}
$$

Note that the $i$ -th input set is deliberately set to empty. We will refer to the $i$ -th feasible structure as "structure $i$ ".

注意，第$i$个输入集被特意设置为空。我们将第$i$个可行结构称为“结构$i$”。

To insert/delete an interval $r \in  {R}_{i}$ ,we first insert/delete $r$ in structures $1,\ldots ,i - 1,i + 1,\ldots ,t$ . To fulfill objective $\left( 1\right) /\left( 2\right)$ ,insert $r$ into structure $i$ ,thereby turning the $i$ -th input set of this structure from $\varnothing$ to $\{ r\}$ . Now,use structure $i$ to enumerate its "join result",namely,the result of the intersection join on

要插入/删除一个区间 $r \in  {R}_{i}$ ，我们首先在结构 $1,\ldots ,i - 1,i + 1,\ldots ,t$ 中插入/删除 $r$ 。为了实现目标 $\left( 1\right) /\left( 2\right)$ ，将 $r$ 插入到结构 $i$ 中，从而使该结构的第 $i$ 个输入集从 $\varnothing$ 变为 $\{ r\}$ 。现在，使用结构 $i$ 来枚举其“连接结果”，即对……进行交集连接的结果

$$
{R}_{1},\ldots ,{R}_{i - 1},\{ r\} ,{R}_{i + 1},\ldots ,{R}_{t}\text{.}
$$

The result is exactly what is needed to achieve objective (1)/(2). After this is done,remove $r$ from the $i$ -th feasible structure.

该结果正是实现目标 (1)/(2) 所需要的。完成此操作后，从第 $i$ 个可行结构中移除 $r$ 。

Apart from enumerating the delta result changes,the update cost is $\widetilde{O}\left( 1\right)$ . The space consumption is $\widetilde{O}\left( n\right)$ where $n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right|$ .

除了枚举增量结果的变化外，更新成本为 $\widetilde{O}\left( 1\right)$ 。空间消耗为 $\widetilde{O}\left( n\right)$ ，其中 $n = \mathop{\sum }\limits_{i}\left| {R}_{i}\right|$ 。

## C Proof of Proposition 2

## C 命题 2 的证明

### C.1 Assumptions Never Violated

### C.1 假设永不被违反

This is obvious about ${C1}$ and ${C2}$ (in particular, ${C2}$ is ensured by the if-condition at Line 5). Next, we focus on ${C3}$ .

对于 ${C1}$ 和 ${C2}$ 这是显而易见的（特别是， ${C2}$ 由第 5 行的条件语句保证）。接下来，我们关注 ${C3}$ 。

First observe that, because of Line 6, we definitely the following clean return property:

首先观察到，由于第 6 行，我们肯定有以下干净返回属性：

When $\operatorname{IJ}\left( {\lambda ,\ldots }\right)$ finishes, ${R}_{\lambda  + 1},\ldots ,{R}_{t}$ have been restored to their original content (i.e.,same as before the query started).

当 $\operatorname{IJ}\left( {\lambda ,\ldots }\right)$ 结束时， ${R}_{\lambda  + 1},\ldots ,{R}_{t}$ 已恢复到其原始内容（即，与查询开始前相同）。

Next we prove that ${C3}$ always holds by induction on $\lambda$ . In general,when the IJ algorithm is invoked with parameters $\left( {\lambda ,\ldots }\right)$ ,we say that a level- $\lambda$ call has been made.

接下来，我们通过对 $\lambda$ 进行归纳来证明 ${C3}$ 始终成立。一般来说，当使用参数 $\left( {\lambda ,\ldots }\right)$ 调用 IJ 算法时，我们称进行了一次第 $\lambda$ 层调用。

Base case: $\mathbf{\lambda } = \mathbf{0}.{C3}$ obviously holds for the first recursive call made by $\operatorname{IJ}\left( {0,\varnothing }\right)$ (at Line 8).

基本情况： $\mathbf{\lambda } = \mathbf{0}.{C3}$ 显然对于 $\operatorname{IJ}\left( {0,\varnothing }\right)$ 进行的第一次递归调用（在第 8 行）成立。

Consider two consecutive recursive calls made by $\operatorname{IJ}\left( {0,\varnothing }\right)$ : let the first be $\operatorname{IJ}\left( {1,\left\{  {\rho }_{1}\right\}  }\right)$ ,and the second be $\operatorname{IJ}\left( {1,\left\{  {\rho }_{1}^{\prime }\right\}  }\right) .{R}_{2},\ldots ,{R}_{t}$ are the same for both calls,due to the clean return property. Regarding ${R}_{1}$ ,the difference is that,for the second call, ${R}_{1}$ contains one less interval: ${\rho }_{1}$ (which has been deleted at Line 9 after the first call finished). Assuming that ${C3}$ holds for the first call, next we show that it must also hold for the second.

考虑 $\operatorname{IJ}\left( {0,\varnothing }\right)$ 进行的两次连续递归调用：设第一次为 $\operatorname{IJ}\left( {1,\left\{  {\rho }_{1}\right\}  }\right)$ ，第二次为 $\operatorname{IJ}\left( {1,\left\{  {\rho }_{1}^{\prime }\right\}  }\right) .{R}_{2},\ldots ,{R}_{t}$ 。由于干净返回属性，两次调用的……是相同的。关于 ${R}_{1}$ ，不同之处在于，对于第二次调用， ${R}_{1}$ 少包含一个区间： ${\rho }_{1}$ （在第一次调用结束后，该区间已在第 9 行被删除）。假设 ${C3}$ 对于第一次调用成立，接下来我们证明它对于第二次调用也一定成立。

Suppose that this is not true. Thus,for the second call,there exists an interval ${\rho }_{1}^{\prime \prime } \in  {R}_{1}$ such that ${\rho }_{1}^{\prime \prime }$ produces a result tuple and has a smaller left endpoint than ${\rho }_{1}^{\prime }$ . But this contradicts how ${\rho }_{1}^{\prime }$ was obtained at Line 4 right before the second call.

假设这不是真的。因此，对于第二次调用，存在一个区间 ${\rho }_{1}^{\prime \prime } \in  {R}_{1}$ ，使得 ${\rho }_{1}^{\prime \prime }$ 产生一个结果元组，并且其左端点比 ${\rho }_{1}^{\prime }$ 的左端点小。但这与第二次调用前在第 4 行获取 ${\rho }_{1}^{\prime }$ 的方式相矛盾。

Inductive case $\lambda  = i + 1$ . Assume that ${C3}$ holds for all the level- $\lambda$ calls with $\lambda  \leq  i$ . We will prove that the same is true for $\lambda  = i + 1$ .

归纳情况 $\lambda  = i + 1$ 。假设对于所有第 $\lambda$ 层且 $\lambda  \leq  i$ 的调用， ${C3}$ 成立。我们将证明对于 $\lambda  = i + 1$ 同样成立。

For this purpose,fix an arbitrary level- $i$ call $\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$ . It suffices to show that all the calls it makes at Line 8 have ${C3}$ fulfilled.

为此，固定一个任意的第 $i$ 层调用 $\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$ 。只需证明它在第 8 行进行的所有调用都满足 ${C3}$ 即可。

By the inductive assumption,at the beginning of $\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right) ,{\rho }_{1},\ldots ,{\rho }_{i}$ produce the minimum result tuple from the current ${R}_{1},\ldots ,{R}_{t}$ . Thus, ${C3}$ holds for the first call made by $\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$ .

根据归纳假设，在$\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right) ,{\rho }_{1},\ldots ,{\rho }_{i}$开始时，从当前的${R}_{1},\ldots ,{R}_{t}$中生成最小结果元组。因此，${C3}$对于$\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$进行的第一次调用成立。

Consider two consecutive recursive calls made by $\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$ : let the first be $\operatorname{IJ}(i +$ $\left. {1,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i},{\rho }_{i + 1}}\right\}  }\right)$ ,and the second be $\operatorname{IJ}\left( {i + 1,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i},{\rho }_{i + 1}^{\prime }}\right\}  }\right)$ . ${R}_{i + 2},\ldots ,{R}_{t}$ are the same for both calls,due to the clean return property. ${R}_{1},\ldots ,{R}_{i}$ are also the same because they are not modified within $\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$ . Regarding ${R}_{i}$ ,the difference is that,for the second call, ${R}_{i}$ contains one less interval: ${\rho }_{i + 1}$ (which has been deleted at Line 9 after the first call finished). Assuming that ${C3}$ holds for the first call,next we show that it must also hold for the second.

考虑$\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$进行的两次连续递归调用：设第一次调用为$\operatorname{IJ}(i +$ $\left. {1,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i},{\rho }_{i + 1}}\right\}  }\right)$，第二次调用为$\operatorname{IJ}\left( {i + 1,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i},{\rho }_{i + 1}^{\prime }}\right\}  }\right)$。由于干净返回属性，两次调用的${R}_{i + 2},\ldots ,{R}_{t}$相同。${R}_{1},\ldots ,{R}_{i}$也相同，因为它们在$\operatorname{IJ}\left( {i,\left\{  {{\rho }_{1},\ldots ,{\rho }_{i}}\right\}  }\right)$内未被修改。关于${R}_{i}$，不同之处在于，对于第二次调用，${R}_{i}$少包含一个区间：${\rho }_{i + 1}$（该区间在第一次调用结束后的第9行被删除）。假设${C3}$对于第一次调用成立，接下来我们证明它对于第二次调用也一定成立。

Suppose that this is not true. Thus,for the second call,there exists an interval ${\rho }_{i + 1}^{\prime \prime } \in  {R}_{i + 1}$ such that ${\rho }_{i + 1}^{\prime \prime }$ produces a result tuple with ${\rho }_{1},\ldots ,{\rho }_{i}$ ,and has a smaller left endpoint than ${\rho }_{i + 1}^{\prime }$ . But this contradicts how ${\rho }_{i + 1}^{\prime }$ was obtained at Line 4 right before the second call.

假设这不是真的。因此，对于第二次调用，存在一个区间${\rho }_{i + 1}^{\prime \prime } \in  {R}_{i + 1}$，使得${\rho }_{i + 1}^{\prime \prime }$生成一个带有${\rho }_{1},\ldots ,{\rho }_{i}$的结果元组，并且其左端点比${\rho }_{i + 1}^{\prime }$小。但这与第二次调用前第4行获取${\rho }_{i + 1}^{\prime }$的方式相矛盾。

### C.2 Correctness of the Output

### C.2 输出的正确性

That no result tuple is reported twice follows from the fact that the tuple $\left( {{\rho }_{1},\ldots ,{\rho }_{t}}\right)$ output at Line 1 becomes monotonically left-larger (see the definition of "left-larger" in Section 5.1).

没有结果元组被报告两次，这是因为在第1行输出的元组$\left( {{\rho }_{1},\ldots ,{\rho }_{t}}\right)$在左侧单调增大（见5.1节中“左侧更大”的定义）。

To prove that no result tuple is missed, suppose that the algorithm fails to output some result tuple $\left( {{r}_{1},\ldots ,{r}_{t}}\right)$ . Let $i \geq  0$ be the largest integer such that a call to IJ was made with the parameters $\left( {i,\left\{  {{r}_{1},\ldots ,{r}_{i}}\right\}  }\right)$ . Hence, $i < t$ .

为了证明没有遗漏任何结果元组，假设算法未能输出某个结果元组$\left( {{r}_{1},\ldots ,{r}_{t}}\right)$。设$i \geq  0$是最大的整数，使得使用参数$\left( {i,\left\{  {{r}_{1},\ldots ,{r}_{i}}\right\}  }\right)$对IJ进行了一次调用。因此，$i < t$。

By the clean return property stated in the proof of Proposition 2, we know that when $\operatorname{IJ}\left( {i,\left\{  {{r}_{1},\ldots ,{r}_{i}}\right\}  }\right)$ started, ${R}_{i + 1},\ldots ,{R}_{t}$ had been fully restored,i.e.,no tuples were missing in ${R}_{i + 1},\ldots ,{R}_{t}$ . This immediately implies that Line 8 must have made a recursive call $\mathrm{{IJ}}(i +$ $\left. {1,\left\{  {{r}_{1},\ldots ,{r}_{i},{r}_{i + 1}}\right\}  }\right)$ ,giving a contradiction.

根据命题2证明中所述的干净返回性质，我们知道当$\operatorname{IJ}\left( {i,\left\{  {{r}_{1},\ldots ,{r}_{i}}\right\}  }\right)$开始时，${R}_{i + 1},\ldots ,{R}_{t}$已完全恢复，即${R}_{i + 1},\ldots ,{R}_{t}$中没有缺失的元组。这立即意味着第8行必定进行了递归调用$\mathrm{{IJ}}(i +$ $\left. {1,\left\{  {{r}_{1},\ldots ,{r}_{i},{r}_{i + 1}}\right\}  }\right)$，从而产生矛盾。

## D Correctness Proof of the Query Algorithm in Section 6.4

## D 6.4节查询算法的正确性证明

We will first prove in Section D. 1 an important property before establishing the query algorithm's correctness in Section D.2.

我们将首先在D.1节证明一个重要性质，然后在D.2节证明查询算法的正确性。

### D.1 The Local-Extreme Property

### D.1 局部极值性质

Consider any non-empty $V \subseteq  U$ ,and any instance vector ${\mathbf{R}}_{V}$ in $V$ . Given a constraint vector ${\mathbf{q}}_{V}$ in $V$ ,define for each $i \in  V$ :

考虑任意非空的$V \subseteq  U$，以及$V$中的任意实例向量${\mathbf{R}}_{V}$。给定$V$中的一个约束向量${\mathbf{q}}_{V}$，为每个$i \in  V$定义：

$$
{J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)  = \left\{  {r\mid \exists {\mathbf{r}}_{V} \in  J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right) \text{ s.t. }{\mathbf{r}}_{V}\left\lbrack  i\right\rbrack   = r}\right\}  .
$$

One can regard ${J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ as the "projection" of $J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ on $i$ .

可以将${J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$视为$J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$在$i$上的“投影”。

Let ${r}_{i}^{\min }$ (or ${r}_{i}^{\max }$ ) be the interval in ${J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ with the smallest left (or largest right, resp.) endpoint. We have:

设${r}_{i}^{\min }$（或${r}_{i}^{\max }$）是${J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$中左端点最小（或右端点最大）的区间。我们有：

Lemma 15 (Local-Extreme Property). Let ${\mathbf{\rho }}_{V}^{\min }$ and ${\mathbf{\rho }}_{V}^{\max }$ be the left-smallest and right-largest elements of $J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ ,respectively. Then,for every $i \in  V$ ,it must hold that

引理15（局部极值性质）。设${\mathbf{\rho }}_{V}^{\min }$和${\mathbf{\rho }}_{V}^{\max }$分别是$J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$中左端点最小和右端点最大的元素。那么，对于每个$i \in  V$，必定有

$$
{\rho }_{V}^{\min }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }
$$

$$
{\mathbf{\rho }}_{V}^{\max }\left\lbrack  i\right\rbrack   = {r}_{i}^{\max }.
$$

Proof. We will prove only the part of the lemma about ${\rho }_{V}^{\min }$ ,because a symmetric argument will then apply to ${\rho }_{V}^{\max }$ .

证明。我们仅证明引理中关于${\rho }_{V}^{\min }$的部分，因为对称的论证同样适用于${\rho }_{V}^{\max }$。

Construct a vector ${\mathbf{r}}_{V}^{\min }$ with ${\mathbf{r}}_{V}^{\min }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$ for every $i \in  V$ . It suffices to show that ${\mathbf{r}}_{V}^{\min }$ is in $J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ . Suppose that this is not true. Thus,there exist distinct $i,j$ in $V$ such that (i) ${G}_{V}$ has an edge between $i$ and $j$ ,(ii) but ${r}_{i}^{\min }$ is disjoint with ${r}_{j}^{\min }$ . Without loss of generality,assume that ${r}_{i}^{\min }$ is to the left of ${r}_{j}^{\min }$ .

构造一个向量${\mathbf{r}}_{V}^{\min }$，使得对于每个$i \in  V$都有${\mathbf{r}}_{V}^{\min }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$。只需证明${\mathbf{r}}_{V}^{\min }$在$J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$中。假设这不是真的。因此，存在$V$中不同的$i,j$，使得（i）${G}_{V}$在$i$和$j$之间有一条边，（ii）但${r}_{i}^{\min }$与${r}_{j}^{\min }$不相交。不失一般性，假设${r}_{i}^{\min }$在${r}_{j}^{\min }$的左侧。

By the fact that ${r}_{i}^{\min } \in  {J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right) ,J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ has a data vector ${\mathbf{r}}_{V}^{\prime }$ with ${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$ The fact ${\mathbf{r}}_{V}^{\prime } \in  J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ means that ${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{r}}_{V}^{\prime }\left\lbrack  j\right\rbrack$ . Thus, ${\mathbf{r}}_{V}^{\prime }\left\lbrack  j\right\rbrack$ must have a smaller left endpoint than ${r}_{j}^{\min }$ ,contradicting the definition of ${r}_{j}^{\min }$ .

根据${r}_{i}^{\min } \in  {J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right) ,J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$有一个数据向量${\mathbf{r}}_{V}^{\prime }$且${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$这一事实，${\mathbf{r}}_{V}^{\prime } \in  J\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$这一事实意味着${\mathbf{r}}_{V}^{\prime }\left\lbrack  i\right\rbrack$与${\mathbf{r}}_{V}^{\prime }\left\lbrack  j\right\rbrack$相交。因此，${\mathbf{r}}_{V}^{\prime }\left\lbrack  j\right\rbrack$的左端点必须小于${r}_{j}^{\min }$的左端点，这与${r}_{j}^{\min }$的定义相矛盾。

The lemma suggests a direction for answering a min-IJ query. We can "detach" the query by looking at the "projection" ${J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$ on each $i \in  V$ individually. Once we have found ${r}_{i}^{\min }$ for each $i$ ,putting them together gives the answer to the min-IJ query. A symmetric strategy works for max-IJ queries.

该引理为回答最小IJ查询（min - IJ query）指明了一个方向。我们可以通过分别查看每个$i \in  V$上的“投影”${J}_{i}\left( {{G}_{V},{\mathbf{R}}_{V},{\mathbf{q}}_{V}}\right)$来“拆解”查询。一旦我们为每个$i$找到了${r}_{i}^{\min }$，将它们组合起来就得到了最小IJ查询的答案。对称策略适用于最大IJ查询（max - IJ query）。

### D.2 Correctness Proof

### D.2 正确性证明

We are now ready to establish the correctness of the query algorithm in Section 6.4. Remember that,given a constraint vector $\mathbf{q}$ ,a min-IJ query finds the left-smallest element in $J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)$ . We will denote that element as ${\mathbf{\rho }}^{\min }$ ; note that when $J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)$ is empty, ${\mathbf{\rho }}^{\min }$ is null. We need to prove that (i) if ${\mathbf{\rho }}^{\min }$ is null,our algorithm returns empty,and (ii) otherwise,the data vector $\mathbf{\rho }$ we return must be ${\mathbf{\rho }}^{\min }$ .

我们现在准备证明6.4节中查询算法的正确性。请记住，给定一个约束向量$\mathbf{q}$，最小IJ查询会在$J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)$中找到最靠左的元素。我们将该元素记为${\mathbf{\rho }}^{\min }$；注意，当$J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)$为空时，${\mathbf{\rho }}^{\min }$为空。我们需要证明：(i) 如果${\mathbf{\rho }}^{\min }$为空，我们的算法返回空；(ii) 否则，我们返回的数据向量$\mathbf{\rho }$必须是${\mathbf{\rho }}^{\min }$。

In this proof,given an interval $r$ ,we will use $r. \vdash$ to denote its left endpoint,and $r.\neg$ to denote its right endpoint. Note that both $r. \vdash$ and $r.\neg$ are real values. Also,remember that $u$ is the node whose combination structure is being searched.

在这个证明中，给定一个区间$r$，我们将用$r. \vdash$表示其左端点，用$r.\neg$表示其右端点。注意，$r. \vdash$和$r.\neg$都是实数值。此外，请记住，$u$是正在搜索其组合结构的节点。

Proposition 6. $J\left( {G,{\mathbf{R}}^{\mathfrak{C}},\mathbf{q}}\right)  = \varnothing$ when either of the following happens:

命题6. 当以下任何一种情况发生时，$J\left( {G,{\mathbf{R}}^{\mathfrak{C}},\mathbf{q}}\right)  = \varnothing$：

- ${\mathbf{r}}_{{V}_{j}^{ < }}^{\max }$ (computed in Step 1) is null for any $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ ;

- 对于任何$j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$，（在步骤1中计算得到的）${\mathbf{r}}_{{V}_{j}^{ < }}^{\max }$为空；

- ${r}_{{V}_{j}^{ > }}^{\min }$ (computed in Step 2) is null for any $j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$ .

- 对于任何$j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$，（在步骤2中计算得到的）${r}_{{V}_{j}^{ > }}^{\min }$为空。

Proof. We will prove only the first bullet, because a similar argument works for the second. Note that ${\mathbf{r}}_{{V}_{j}^{ < }}^{\max }$ being null implies $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)  = \varnothing$ . If $J\left( {G,{\mathbf{R}}^{@},\mathbf{q}}\right)$ is not empty,let $\mathbf{\rho }$ be an arbitrary data vector therein. It is easy to verify that the projection of $\mathbf{\rho }$ in ${V}_{j}^{ < }$ is in $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)$ , contradicting $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)  = \varnothing$ .

证明。我们仅证明第一个要点，因为第二个要点可通过类似的论证得出。注意，${\mathbf{r}}_{{V}_{j}^{ < }}^{\max }$ 为空意味着 $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)  = \varnothing$。如果 $J\left( {G,{\mathbf{R}}^{@},\mathbf{q}}\right)$ 非空，设 $\mathbf{\rho }$ 为其中任意一个数据向量。容易验证，$\mathbf{\rho }$ 在 ${V}_{j}^{ < }$ 中的投影在 $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)$ 中，这与 $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)  = \varnothing$ 矛盾。

Proposition 7. If $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ is not empty,then for every $i \in  {V}^{ > }$ ,it must hold that

命题 7。如果 $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ 非空，那么对于每个 $i \in  {V}^{ > }$，必有

$$
{\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{V}^{\min }\left\lbrack  i\right\rbrack  
$$

Proof. Suppose that this is not true. Construct a different data vector ${\mathbf{\rho }}^{\prime }$ as follows:

证明。假设这不是真的。按如下方式构造一个不同的数据向量 ${\mathbf{\rho }}^{\prime }$：

- for every $i \in  {V}^{ < } \cup  {V}^{ = }$ ,set ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ ;

- 对于每个 $i \in  {V}^{ < } \cup  {V}^{ = }$，令 ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$；

- for every $i \in  {V}^{ > }$ ,set ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack$ .

- 对于每个 $i \in  {V}^{ > }$，令 ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack$。

For any $j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$ ,the projection of ${\mathbf{\rho }}^{\min }$ in ${V}_{j}^{ > }$ is in $J\left( {{G}_{{V}_{j}^{ > }},{\mathbf{R}}_{{V}_{j}^{ > }},{\mathbf{q}}_{{V}_{j}^{ > }}}\right)$ . By applying the local-extreme property of Lemma 15 on ${G}_{{V}_{i}^{ > }}$ ,we know ${\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash$ for every $i \in  {V}^{ > }$ . Hence, ${\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash$ for every $i \in  {V}^{ > }$ . This suggests that ${\mathbf{\rho }}^{\prime }$ must be left-smaller than ${\mathbf{\rho }}^{\min }$ .

对于任意 $j \in  \left\lbrack  {1,{h}_{2}}\right\rbrack$，${\mathbf{\rho }}^{\min }$ 在 ${V}_{j}^{ > }$ 中的投影在 $J\left( {{G}_{{V}_{j}^{ > }},{\mathbf{R}}_{{V}_{j}^{ > }},{\mathbf{q}}_{{V}_{j}^{ > }}}\right)$ 中。通过对 ${G}_{{V}_{i}^{ > }}$ 应用引理 15 的局部极值性质，我们知道对于每个 $i \in  {V}^{ > }$ 有 ${\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash$。因此，对于每个 $i \in  {V}^{ > }$ 有 ${\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash$。这表明 ${\mathbf{\rho }}^{\prime }$ 必定在左侧小于 ${\mathbf{\rho }}^{\min }$。

Next we will show that ${\mathbf{\rho }}^{\prime }$ is in $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ ,thus contradicting the role of ${\mathbf{\rho }}^{\min }$ . It suffices to prove: for any $i \in  {V}^{ > }$ and ${i}^{\prime } \in  {V}^{ = }$ such that $G$ has an edge between $i,{i}^{\prime }$ ,we must have: ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ . Clearly, ${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ . As ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ covers $\operatorname{key}\left( u\right)$ but ${\mathbf{\rho }}^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash   > \operatorname{key}\left( u\right)$ ,we know that ${\mathbf{\rho }}^{\text{min }}\left\lbrack  {i}^{\prime }\right\rbrack$ must cover ${\mathbf{\rho }}^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash$ and,thus,also ${\mathbf{r}}_{V > }^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash$ (using the fact $\operatorname{key}\left( u\right)  < {\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash  )$ . We therefore conclude that ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ .

接下来我们将证明 ${\mathbf{\rho }}^{\prime }$ 属于 $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$，从而与 ${\mathbf{\rho }}^{\min }$ 的作用相矛盾。只需证明：对于任意满足 $G$ 在 $i,{i}^{\prime }$ 之间有一条边的 $i \in  {V}^{ > }$ 和 ${i}^{\prime } \in  {V}^{ = }$，我们必定有：${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ 与 ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ 相交。显然，${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack$ 与 ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ 相交。由于 ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ 覆盖 $\operatorname{key}\left( u\right)$ 但 ${\mathbf{\rho }}^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash   > \operatorname{key}\left( u\right)$，我们知道 ${\mathbf{\rho }}^{\text{min }}\left\lbrack  {i}^{\prime }\right\rbrack$ 必定覆盖 ${\mathbf{\rho }}^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash$，因此也覆盖 ${\mathbf{r}}_{V > }^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash$（利用事实 $\operatorname{key}\left( u\right)  < {\mathbf{r}}_{{V}^{ > }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash  )$）。因此，我们得出结论：${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ 与 ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ 相交。

Proposition 8. If ${r}_{i}^{\min }$ (computed in Step 3) is null for any $i \in  {V}^{ = }$ ,then $J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)  = \varnothing$ .

命题 8. 如果对于任意 $i \in  {V}^{ = }$，${r}_{i}^{\min }$（在步骤 3 中计算得出）为空，则 $J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)  = \varnothing$。

Proof. Suppose that $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)  \neq  \varnothing$ ; thus, ${\mathbf{\rho }}^{\min }$ is not null. By definition, ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ intersects with $\left( {-\infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack \text{and}\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty }\right)$ . We will prove that ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ contains the interval $\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$ obtained in Step 3 for $i$ ,thus contradicting the fact that ${r}_{i}^{\min }$ is null.

证明. 假设 $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)  \neq  \varnothing$；因此，${\mathbf{\rho }}^{\min }$ 不为空。根据定义，${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 与 $\left( {-\infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack \text{and}\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty }\right)$ 相交。我们将证明 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 包含在步骤 3 中为 $i$ 得到的区间 $\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$，从而与 ${r}_{i}^{\min }$ 为空这一事实相矛盾。

By definition, ${\lambda }_{1} \leq  \operatorname{key}\left( u\right)  \leq  {\lambda }_{2}$ . Hence,it suffices to prove that ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ contains both $\left\lbrack  {{\lambda }_{1},\operatorname{key}\left( u\right) }\right\rbrack$ and $\left\lbrack  {\operatorname{key}\left( u\right) ,{\lambda }_{2}}\right\rbrack$ . We will prove this only for $\left\lbrack  {{\lambda }_{1},\operatorname{key}\left( u\right) }\right\rbrack$ because a symmetric argument proves the same for $\left\lbrack  {\operatorname{key}\left( u\right) ,{\lambda }_{2}}\right\rbrack$ .

根据定义，${\lambda }_{1} \leq  \operatorname{key}\left( u\right)  \leq  {\lambda }_{2}$ 。因此，只需证明 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 同时包含 $\left\lbrack  {{\lambda }_{1},\operatorname{key}\left( u\right) }\right\rbrack$ 和 $\left\lbrack  {\operatorname{key}\left( u\right) ,{\lambda }_{2}}\right\rbrack$ 即可。我们仅针对 $\left\lbrack  {{\lambda }_{1},\operatorname{key}\left( u\right) }\right\rbrack$ 进行证明，因为通过对称的论证可以对 $\left\lbrack  {\operatorname{key}\left( u\right) ,{\lambda }_{2}}\right\rbrack$ 得出相同的结论。

If ${N}^{ < }\left( i\right)$ is empty,then ${\lambda }_{1} = \operatorname{key}\left( u\right)$ ,in which case ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ obviously covers $\left\lbrack  {{\lambda }_{1},\operatorname{key}\left( u\right) }\right\rbrack$ . Next, we focus on the scenario where ${N}^{ < }\left( i\right)$ is not empty.

如果 ${N}^{ < }\left( i\right)$ 为空集，那么 ${\lambda }_{1} = \operatorname{key}\left( u\right)$ ，在这种情况下，${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 显然覆盖 $\left\lbrack  {{\lambda }_{1},\operatorname{key}\left( u\right) }\right\rbrack$ 。接下来，我们关注 ${N}^{ < }\left( i\right)$ 不为空集的情况。

For every ${i}^{\prime } \in  {N}^{ < }\left( i\right) ,{\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ lies to the left of $\operatorname{key}\left( u\right)$ . Therefore, ${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack$ must cover ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ . This indicates that ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ must cover the smallest right endpoint ${\lambda }_{1}^{\prime }$ of the intervals in the following set:

对于每个 ${i}^{\prime } \in  {N}^{ < }\left( i\right) ,{\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ 都位于 $\operatorname{key}\left( u\right)$ 的左侧。因此，${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack$ 必须覆盖 ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ 。这表明 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 必须覆盖以下集合中区间的最小右端点 ${\lambda }_{1}^{\prime }$ ：

$$
\left\{  {{\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack   \mid  {i}^{\prime } \in  {N}^{ < }\left( i\right) }\right\}  . \tag{8}
$$

It remains to show that ${\lambda }_{1}^{\prime } \leq  {\lambda }_{1}$ . By comparing (8) to (5),one can see that we only need to show that ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv   \leq  {\mathbf{r}}_{{V}^{ < }}^{\max }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv$ ,for each ${i}^{\prime } \in  {N}^{ < }\left( i\right)$ .

还需证明 ${\lambda }_{1}^{\prime } \leq  {\lambda }_{1}$ 。通过比较 (8) 和 (5)，可以看出我们只需证明对于每个 ${i}^{\prime } \in  {N}^{ < }\left( i\right)$ ，都有 ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv   \leq  {\mathbf{r}}_{{V}^{ < }}^{\max }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv$ 。

Fix an arbitrary ${i}^{\prime } \in  {N}^{ < }\left( i\right)$ . Identify the only $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ satisfying ${i}^{\prime } \in  {V}_{j}^{ < }$ . Let ${\rho }_{{V}_{j}^{ < }}^{min}$ be the projection of ${\mathbf{\rho }}^{\min }$ in ${V}_{j}^{ < }$ . Since ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min } \in  J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)$ ,applying the local-extreme property in Lemma 15 on ${G}_{{V}_{j}^{ < }}$ shows that ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack   \cdot   \dashv   \leq  {\mathbf{r}}_{{V}_{j}^{ < }}^{max}\left\lbrack  {i}^{\prime }\right\rbrack   \cdot   \dashv   = {\mathbf{r}}_{{V}^{ < }}^{max}\left\lbrack  {i}^{\prime }\right\rbrack   \cdot   \dashv$ .

固定任意的 ${i}^{\prime } \in  {N}^{ < }\left( i\right)$。确定唯一满足 ${i}^{\prime } \in  {V}_{j}^{ < }$ 的 $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$。设 ${\rho }_{{V}_{j}^{ < }}^{min}$ 为 ${\mathbf{\rho }}^{\min }$ 在 ${V}_{j}^{ < }$ 中的投影。由于 ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min } \in  J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}}\right)$，对 ${G}_{{V}_{j}^{ < }}$ 应用引理 15 中的局部极值性质表明 ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack   \cdot   \dashv   \leq  {\mathbf{r}}_{{V}_{j}^{ < }}^{max}\left\lbrack  {i}^{\prime }\right\rbrack   \cdot   \dashv   = {\mathbf{r}}_{{V}^{ < }}^{max}\left\lbrack  {i}^{\prime }\right\rbrack   \cdot   \dashv$。

Proposition 9. If $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ is not empty,then for every $i \in  {V}^{ = }$ ,it must hold that

命题 9. 如果 $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ 非空，那么对于每个 $i \in  {V}^{ = }$，必有

$$
{\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }.
$$

Proof. Suppose that this is not true. Fix an $i \in  {V}^{ = }$ such that ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack   \neq  {r}_{i}^{\min }$ . In Proposition 8,we have proved that ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ contains the interval $\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$ obtained in Step 3 for $i$ . Furthermore, ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ needs to intersect with $\left( {-\infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack \text{and}\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty }\right)$ . By how ${r}_{i}^{\min }$ is computed,it must hold that ${r}_{i}^{\min } \cdot   \vdash   < {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash$ .

证明. 假设这不是真的。固定一个 $i \in  {V}^{ = }$ 使得 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack   \neq  {r}_{i}^{\min }$。在命题 8 中，我们已经证明了 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 包含在步骤 3 中为 $i$ 得到的区间 $\left\lbrack  {{\lambda }_{1},{\lambda }_{2}}\right\rbrack$。此外，${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 需要与 $\left( {-\infty ,\mathbf{q}\left\lbrack  i\right\rbrack  .a\rbrack \text{and}\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty }\right)$ 相交。根据 ${r}_{i}^{\min }$ 的计算方式，必有 ${r}_{i}^{\min } \cdot   \vdash   < {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash$。

Construct a data vector ${\mathbf{\rho }}^{\prime }$ as follows:

按如下方式构造一个数据向量 ${\mathbf{\rho }}^{\prime }$：

- for every ${i}^{\prime } \neq  i$ ,set ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ ;

- 对于每个 ${i}^{\prime } \neq  i$，设 ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$；

- set ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$ .

- 设 ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {r}_{i}^{\min }$。

We will prove that ${\mathbf{\rho }}^{\prime }$ is in $J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)$ which,given the fact that ${\mathbf{\rho }}^{\prime }$ is left-smaller than ${\mathbf{\rho }}^{\min }$ , contradicts the role of ${\mathbf{\rho }}^{\min }$ .

我们将证明 ${\mathbf{\rho }}^{\prime }$ 在 $J\left( {G,{\mathbf{R}}^{\complement },\mathbf{q}}\right)$ 中，鉴于 ${\mathbf{\rho }}^{\prime }$ 在左侧小于 ${\mathbf{\rho }}^{\min }$ 这一事实，这与 ${\mathbf{\rho }}^{\min }$ 的作用相矛盾。

It suffices to show that:

只需证明：

- for any ${i}^{\prime } \in  {V}^{ < }$ that is adjacent to $i$ in $G,{\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ ;

- 对于在 $G,{\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ 中与 $i$ 相邻的任意 ${i}^{\prime } \in  {V}^{ < }$，其与 ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ 相交；

- for any ${i}^{\prime } \in  {V}^{ > }$ that is adjacent to $i$ in $G,{\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ .

- 对于在 $G,{\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ 中与 $i$ 相邻的任意 ${i}^{\prime } \in  {V}^{ > }$，其与 ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ 相交。

To prove the first bullet,first note that ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ intersects with ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ . Since ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ contains $\operatorname{key}\left( u\right)$ but ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv   < \operatorname{key}\left( u\right)$ ,we know ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv$ . This leads to ${r}_{i}^{\min }. \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv$ , indicating that ${r}_{i}^{\min }$ intersects with ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ (recall that ${r}_{i}^{\min }$ covers $\operatorname{key}\left( u\right)$ ). We therefore conclude that ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ .

为证明第一点，首先注意到 ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ 与 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 相交。由于 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ 包含 $\operatorname{key}\left( u\right)$ 但不包含 ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv   < \operatorname{key}\left( u\right)$，我们可知 ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv$。这导致 ${r}_{i}^{\min }. \vdash   \leq  {\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \dashv$，表明 ${r}_{i}^{\min }$ 与 ${\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ 相交（回想 ${r}_{i}^{\min }$ 覆盖 $\operatorname{key}\left( u\right)$）。因此，我们得出 ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ 与 ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ 相交。

To prove the second bullet,let $j$ be the only integer in ${V}^{ > }$ such that ${i}^{\prime } \in  {V}_{j}^{ > }$ . Consider the ${\mathbf{r}}_{{V}_{j}^{ > }}^{min}$ obtained in Step 2. Proposition 7 guarantees that ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack   = {\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ . Since ${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ belongs to the set in (6),the value ${\lambda }_{2}$ obtained at Step 3 for $i$ must satisfy

为证明第二点，设 $j$ 是 ${V}^{ > }$ 中唯一满足 ${i}^{\prime } \in  {V}_{j}^{ > }$ 的整数。考虑在步骤 2 中得到的 ${\mathbf{r}}_{{V}_{j}^{ > }}^{min}$。命题 7 保证 ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack   = {\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$。由于 ${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ 属于 (6) 中的集合，在步骤 3 中为 $i$ 得到的值 ${\lambda }_{2}$ 必须满足

$$
{\lambda }_{2} \geq  {\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \vdash   > \operatorname{key}\left( u\right) .
$$

By how ${r}_{i}^{\min }$ is computed, ${r}_{i}^{\min }$ must cover $\operatorname{key}\left( u\right)$ and ${\lambda }_{2}$ . Hence, ${r}_{i}^{\min }$ covers ${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ . $\vdash$ as well, implying that ${r}_{i}^{\min }$ intersects with ${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$ . We thus conclude that ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ .

根据${r}_{i}^{\min }$的计算方式，${r}_{i}^{\min }$必须覆盖$\operatorname{key}\left( u\right)$和${\lambda }_{2}$。因此，${r}_{i}^{\min }$也覆盖${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$和$\vdash$，这意味着${r}_{i}^{\min }$与${\mathbf{r}}_{{V}_{j}^{ > }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack$相交。由此我们得出结论，${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$与${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$相交。

Proposition 10. If ${\mathbf{r}}_{{V}_{j}^{ < }}^{\min }$ (computed in Step 4) is null for any $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ ,then $J\left( {G,{\mathbf{R}}^{@},\mathbf{q}}\right)  = \varnothing$ .

命题10. 如果对于任意$j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$，${\mathbf{r}}_{{V}_{j}^{ < }}^{\min }$（在步骤4中计算得出）为空，则$J\left( {G,{\mathbf{R}}^{@},\mathbf{q}}\right)  = \varnothing$。

Proof. That ${\mathbf{r}}_{{V}_{j}^{ < }}^{\min }$ is null implies $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)  = \varnothing$ . Suppose that $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)  \neq  \varnothing$ ; thus, ${\mathbf{\rho }}^{\min }$ is not null. Let ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min }$ the projection of ${\mathbf{\rho }}^{\min }$ in ${V}_{j}^{ < }$ . We will show that ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min } \in  J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)$ , thus contradicting $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)  = \varnothing$ .

证明。${\mathbf{r}}_{{V}_{j}^{ < }}^{\min }$为空意味着$J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)  = \varnothing$。假设$J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)  \neq  \varnothing$；因此，${\mathbf{\rho }}^{\min }$不为空。设${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min }$为${\mathbf{\rho }}^{\min }$在${V}_{j}^{ < }$上的投影。我们将证明${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min } \in  J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)$，从而与$J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)  = \varnothing$矛盾。

It suffices to show that,for each $i \in  {V}_{j}^{ < },{\rho }_{{V}_{j}^{ < }}^{min}\left\lbrack  i\right\rbrack$ intersects with $\left\lbrack  {{q}_{{V}_{j}^{ < }}^{\prime } \cdot  b,\infty }\right)$ . This is obvious if ${N}^{ = }\left( i\right)  = \varnothing$ because (i) in this case ${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }$ is the projection of $\mathbf{q}$ in ${V}_{j}^{ < }$ ,and (ii) by definition ${\mathbf{\rho }}^{\text{min }}\left\lbrack  i\right\rbrack$ must intersect with $\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty )$ .

只需证明对于每个$i \in  {V}_{j}^{ < },{\rho }_{{V}_{j}^{ < }}^{min}\left\lbrack  i\right\rbrack$都与$\left\lbrack  {{q}_{{V}_{j}^{ < }}^{\prime } \cdot  b,\infty }\right)$相交即可。如果${N}^{ = }\left( i\right)  = \varnothing$，这是显然的，因为（i）在这种情况下，${\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }$是$\mathbf{q}$在${V}_{j}^{ < }$上的投影，并且（ii）根据定义，${\mathbf{\rho }}^{\text{min }}\left\lbrack  i\right\rbrack$必须与$\lbrack \mathbf{q}\left\lbrack  i\right\rbrack  .b,\infty )$相交。

Consider now ${N}^{ = }\left( i\right)  \neq  \varnothing$ . For each ${i}^{\prime } \in  {N}^{ = }\left( i\right)$ ,we know from Proposition 9 that ${\rho }^{\min }\left\lbrack  {i}^{\prime }\right\rbrack   =$ ${r}_{{i}^{\prime }}^{min}$ . As ${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack  .\forall  < {key}\left( u\right)$ but ${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$ covers ${key}\left( u\right)$ ,we know that ${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack$ must intersect with $\left\lbrack  {{\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \vdash  ,\infty )}\right.$ . As the above holds for every ${i}^{\prime } \in  {N}^{ = }\left( i\right)$ ,we assert that ${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ must intersect with $\left\lbrack  {{\lambda }_{3},\infty }\right)$ (recall how ${\lambda }_{3}$ is derived from (7)). Therefore, ${\rho }_{{V}_{j}^{ < }}^{min}\left\lbrack  i\right\rbrack$ must intersect with $\left\lbrack  {{q}_{{V}_{j}^{ < }}^{\prime } \cdot  b,\infty }\right)$ , meaning that ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min }$ is in $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)$ .

现在考虑${N}^{ = }\left( i\right)  \neq  \varnothing$。对于每个${i}^{\prime } \in  {N}^{ = }\left( i\right)$，根据命题9我们知道${\rho }^{\min }\left\lbrack  {i}^{\prime }\right\rbrack   =$ ${r}_{{i}^{\prime }}^{min}$。由于${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack  .\forall  < {key}\left( u\right)$但${\mathbf{\rho }}^{min}\left\lbrack  {i}^{\prime }\right\rbrack$覆盖${key}\left( u\right)$，我们知道${\mathbf{\rho }}^{min}\left\lbrack  i\right\rbrack$必定与$\left\lbrack  {{\mathbf{\rho }}^{\min }\left\lbrack  {i}^{\prime }\right\rbrack  . \vdash  ,\infty )}\right.$相交。由于上述情况对每个${i}^{\prime } \in  {N}^{ = }\left( i\right)$都成立，我们断言${\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$必定与$\left\lbrack  {{\lambda }_{3},\infty }\right)$相交（回顾${\lambda }_{3}$是如何从(7)推导出来的）。因此，${\rho }_{{V}_{j}^{ < }}^{min}\left\lbrack  i\right\rbrack$必定与$\left\lbrack  {{q}_{{V}_{j}^{ < }}^{\prime } \cdot  b,\infty }\right)$相交，这意味着${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\min }$在$J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)$中。

Proposition 11. If $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ is not empty,then for every $i \in  {V}^{ < }$ ,it must hold that

命题11. 如果$J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$非空，则对于任意$i \in  {V}^{ < }$，必有

$$
{\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack  
$$

Proof. For each $j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$ ,We have proved in Proposition 10 that the projection ${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\text{min }}$ of ${\mathbf{\rho }}^{\text{min }}$ in ${V}_{j}^{ < }$ must belong to $J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)$ . The local-extreme property of Lemma 15 tells us that that

证明. 对于每个$j \in  \left\lbrack  {1,{h}_{1}}\right\rbrack$，我们在命题10中已证明，${\mathbf{\rho }}^{\text{min }}$在${V}_{j}^{ < }$上的投影${\mathbf{\rho }}_{{V}_{j}^{ < }}^{\text{min }}$必定属于$J\left( {{G}_{{V}_{j}^{ < }},{\mathbf{R}}_{{V}_{j}^{ < }},{\mathbf{q}}_{{V}_{j}^{ < }}^{\prime }}\right)$。引理15的局部极值性质告诉我们

$$
{\mathbf{r}}_{{V}^{ < }}^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash   \leq  {\mathbf{\rho }}_{{V}_{j}^{ < }}^{\text{min }}\left\lbrack  i\right\rbrack  . \vdash  
$$

holds for every $i \in  {V}_{j}^{ < }$ .

对任意$i \in  {V}_{j}^{ < }$都成立。

Suppose that the proposition does not hold. Construct a data vector ${\mathbf{\rho }}^{\prime }$ as follows:

假设该命题不成立。按如下方式构造一个数据向量${\mathbf{\rho }}^{\prime }$：

- for every $i \in  {V}^{ = } \cup  {V}^{ > }$ ,set ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$ ;

- 对于任意$i \in  {V}^{ = } \cup  {V}^{ > }$，令${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{\rho }}^{\min }\left\lbrack  i\right\rbrack$；

- for every $i \in  {V}^{ < }$ ,set ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack$ .

- 对于任意$i \in  {V}^{ < }$，令${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack$。

Thus, ${\mathbf{\rho }}^{\prime }$ is left-smaller than ${\mathbf{\rho }}^{\min }$ . Next,we will show that ${\mathbf{\rho }}^{\prime }$ is in $J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$ ,thus contradicting the role of ${\mathbf{\rho }}^{\min }$ .

因此，${\mathbf{\rho }}^{\prime }$在左侧小于${\mathbf{\rho }}^{\min }$。接下来，我们将证明${\mathbf{\rho }}^{\prime }$属于$J\left( {G,{\mathbf{R}}^{\mathbb{C}},\mathbf{q}}\right)$，从而与${\mathbf{\rho }}^{\min }$的作用相矛盾。

It suffices to show that ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ ,for any $i \in  {V}^{ < }$ and ${i}^{\prime } \in  {V}^{ = }\left( i\right)$ . Fix $j$ to be the unique integer in $\left\lbrack  {1,{h}_{1}}\right\rbrack$ such that $i \in  {V}_{j}^{ < }$ . Since ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ is in the set of (7),the ${\lambda }_{3}$ computed in Step 4 for $i$ is at least ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ . $\vdash$ . By how ${\mathbf{r}}_{{V}_{j}^{ < }}^{min}$ is computed, ${\mathbf{r}}_{{V}_{j}^{ < }}^{min}\left\lbrack  i\right\rbrack$ must intersect with $\left\lbrack  {{\lambda }_{3},\infty }\right)$ and,hence, must also intersect with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ (here,we used the fact that ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ covers $\operatorname{key}\left( u\right)$ ). We thus conclude that ${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}_{j}^{ < }}^{\min }\left\lbrack  i\right\rbrack$ intersects with ${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$ .

只需证明对于任意的$i \in  {V}^{ < }$和${i}^{\prime } \in  {V}^{ = }\left( i\right)$，${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack$与${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$相交即可。固定$j$为$\left\lbrack  {1,{h}_{1}}\right\rbrack$中唯一的整数，使得$i \in  {V}_{j}^{ < }$成立。由于${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$在集合(7)中，步骤4中为$i$计算得到的${\lambda }_{3}$至少为${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$。$\vdash$。根据${\mathbf{r}}_{{V}_{j}^{ < }}^{min}$的计算方式，${\mathbf{r}}_{{V}_{j}^{ < }}^{min}\left\lbrack  i\right\rbrack$必定与$\left\lbrack  {{\lambda }_{3},\infty }\right)$相交，因此也必定与${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$相交（这里，我们利用了${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$覆盖$\operatorname{key}\left( u\right)$这一事实）。因此，我们得出结论：${\mathbf{\rho }}^{\prime }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}^{ < }}^{\min }\left\lbrack  i\right\rbrack   = {\mathbf{r}}_{{V}_{j}^{ < }}^{\min }\left\lbrack  i\right\rbrack$与${\mathbf{\rho }}^{\prime }\left\lbrack  {i}^{\prime }\right\rbrack$相交。

The correctness of our algorithm follows from all the above propositions, and the fact that the vector $\mathbf{\rho }$ constructed in Step 5 is in $J\left( {G,{\mathbf{R}}^{C},\mathbf{q}}\right)$ ,implying that $J\left( {G,{\mathbf{R}}^{C},\mathbf{q}}\right)$ is not empty.

我们算法的正确性由上述所有命题以及步骤5中构造的向量$\mathbf{\rho }$属于$J\left( {G,{\mathbf{R}}^{C},\mathbf{q}}\right)$这一事实得出，这意味着$J\left( {G,{\mathbf{R}}^{C},\mathbf{q}}\right)$非空。

## References

## 参考文献

[1] Pankaj K. Agarwal, Lars Arge, Haim Kaplan, Eyal Molad, Robert Endre Tarjan, and Ke Yi. An optimal dynamic data structure for stabbing-semigroup queries. SIAM J. of Comp., 41(1):104-127, 2012.

[2] Lars Arge and Jeffrey Scott Vitter. Optimal external memory interval management. SIAM J. of Comp., 32(6):1488-1508, 2003.

[3] Christoph Berkholz, Jens Keppeler, and Nicole Schweikardt. Answering conjunctive queries under updates. In PODS, pages 303-318, 2017.

[4] Christoph Berkholz, Jens Keppeler, and Nicole Schweikardt. Answering FO+MOD queries under updates on bounded degree databases. In ICDT, pages 8:1-8:18, 2017.

[5] Thomas Brinkhoff, Hans-Peter Kriegel, and Bernhard Seeger. Efficient processing of spatial joins using R-trees. In SIGMOD, pages 237-246, 1993.

[6] Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars. Computational Geometry: Algorithms and Applications. Springer-Verlag, 3rd edition, 2008.

[7] Bipin C. Desai. Performance of a composite attribute and join index. IEEE Trans. Software Eng., 15(2):142-152, 1989.

[8] David J. DeWitt, Jeffrey F. Naughton, and Donovan A. Schneider. An evaluation of non-equijoin algorithms. In VLDB, pages 443-452, 1991.

[9] Anton Dignös, Michael H. Böhlen, and Johann Gamper. Overlap interval partition join. In SIGMOD, pages 1459-1470, 2014.

[10] Herbert Edelsbrunner. Dynamic data structures for orthogonal intersection queries. Report F59, Inst. Informationsverarb., Tech. Univ. Graz, 1980.

[11] Jost Enderle, Matthias Hampel, and Thomas Seidl. Joining interval data in relational databases. In ${SIGMOD}$ ,pages ${683} - {694},{2004}$ .

[12] Pankaj Goyal, Hon Fung Li, Eric Regener, and Fereidoon Sadri. Scheduling of page fetches in join operations using Bc-trees. In ICDE, pages 304-310, 1988.

[13] Monika Henzinger, Sebastian Krinninger, Danupon Nanongkai, and Thatchaphol Saranurak. Unifying and strengthening hardness for dynamic problems via the online matrix-vector multiplication conjecture. In ${STOC}$ ,pages ${21} - {30},{2015}$ .

[14] Muhammad Idris, Martín Ugarte, and Stijn Vansummeren. The dynamic yannakakis algorithm: Compact and efficient query processing under updates. In SIGMOD, pages 1259-1274. ACM, 2017.

[15] Muhammad Idris, Martín Ugarte, Stijn Vansummeren, Hannes Voigt, and Wolfgang Lehner. Conjunctive queries with inequalities under updates. PVLDB, 11(7):733-745, 2018.

[16] Ahmet Kara, Hung Q. Ngo, Milos Nikolic, Dan Olteanu, and Haozhe Zhang. Maintaining triangle queries under updates. TODS, 45(3):11:1-11:46, 2020.

[17] Ahmet Kara, Milos Nikolic, Dan Olteanu, and Haozhe Zhang. Trade-offs in static and dynamic evaluation of hierarchical queries. In Dan Suciu, Yufei Tao, and Zhewei Wei, editors, PODS, pages 375-392, 2020.

[18] Zuhair Khayyat, William Lucia, Meghna Singh, Mourad Ouzzani, Paolo Papotti, Jorge-Arnulfo Quiané-Ruiz, Nan Tang, and Panos Kalnis. Fast and scalable inequality joins. VLDB J., 26(1):125-150, 2017.

[19] Christoph Koch. Incremental query evaluation in a ring of databases. In PODS, pages 87-98, 2010.

[20] Christoph Koch, Daniel Lupei, and Val Tannen. Incremental view maintenance for collection programming. In PODS, pages 75-90, 2016.

[21] Katja Losemann and Wim Martens. MSO queries on trees: enumerating answers under updates. In Joint Meeting of the Annual Conference on Computer Science Logic (CSL) and the Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), CSL-LICS, pages 67:1-67:10, 2014.

[22] Nikos Mamoulis and Dimitris Papadias. Multiway spatial joins. TODS, 26(4):424-475, 2001.

[23] Edward M. McCreight. Efficient algorithms for enumerating intersecting intervals and rectangles. Report CSL-80-9, Xerox Palo Alto Res. Center, 1980.

[24] Jurg Nievergelt and Edward M. Reingold. Binary search trees of bounded balance. SIAM J. of Comp., 2(1):33-43, 1973.

[25] Dimitris Papadias, Nikos Mamoulis, and Yannis Theodoridis. Processing and optimization of multiway spatial joins using r-trees. In PODS, pages 44-55, 1999.

[26] Jignesh M. Patel and David J. DeWitt. Partition based spatial-merge join. In SIGMOD, pages 259-270, 1996.

[27] Danila Piatov, Sven Helmer, and Anton Dignös. An interval join optimized for modern hardware. In ${ICDE}$ ,pages ${1098} - {1109},{2016}$ .

[28] Farhan Tauheed, Thomas Heinis, and Anastasia Ailamaki. THERMAL-JOIN: A scalable spatial join for dynamic workloads. In SIGMOD, pages 939-950. ACM, 2015.

[29] Dan E. Willard. An algorithm for handling many relational calculus queries efficiently. JCSS, 65(2):295-331, 2002.

[30] Mihalis Yannakakis. Algorithms for acyclic database schemes. In Very Large Data Bases, 7th International Conference, September 9-11, 1981, Cannes, France, Proceedings, pages 82-94, 1981.

[31] Thomas Zeume and Thomas Schwentick. Dynamic conjunctive queries. JCSS, 88:3-26, 2017.

[32] Rui Zhang, Jianzhong Qi, Dan Lin, Wei Wang, and Raymond Chi-Wing Wong. A highly optimized algorithm for continuous intersection join queries over moving objects. VLDB J., 21(4):561-586, 2012.