# Nearest Neighbor Queries *

# 最近邻查询*

Nick Roussopoulos Stephen Kelley Frédéric Vincent

尼克·鲁索普洛斯 斯蒂芬·凯利 弗雷德里克·文森特

Department of Computer Science

计算机科学系

University of Maryland

马里兰大学

College Park, MD 20742

马里兰州大学公园市，邮编20742

## Abstract

## 摘要

A frequently encountered type of query in Geographic Information Systems is to find the $\mathbf{k}$ nearest neighbor objects to a given point in space. Processing such queries requires substantially different search algorithms than those for location or range queries. In this paper we present an efficient branch-and-bound R-tree traversal algorithm to find the nearest neighbor object to a point, and then generalize it to finding the $k$ nearest neighbors. We also discuss metrics for an optimistic and a pessimistic search ordering strategy as well as for pruning. Finally, we present the results of several experiments obtained using the implementation of our algorithm and examine the behavior of the metrics and the scalability of the algorithm.

地理信息系统（Geographic Information Systems，GIS）中经常遇到的一种查询类型是，查找空间中给定一点的 $\mathbf{k}$ 个最近邻对象。处理此类查询所需的搜索算法与处理位置查询或范围查询的算法有很大不同。在本文中，我们提出了一种高效的分支限界R树遍历算法，用于查找给定点的最近邻对象，然后将其推广到查找 $k$ 个最近邻对象。我们还讨论了乐观和悲观搜索排序策略以及剪枝的度量方法。最后，我们展示了使用我们实现的算法进行的几个实验结果，并研究了这些度量方法的特性以及算法的可扩展性。

## 1 INTRODUCTION

## 1 引言

The efficient implementation of Nearest Neighbor (NN) queries is of a particular interest in Geographic Information Systems (GIS). For example, a user may point to a specific location or an object on the screen, and request the system to find the five nearest objects to it in the database. Another situation where NN query is useful is when the user is not familiar with the layout of the spatial objects. In the case of an astrophysics database, finding the nearest star to a given point in the sky could involve multiple unsuccessful searches with varying window sizes if we were to use a more traditional $2\mathrm{D}$ range query. Another even more complex query that could be handled by an NN technique is to find the four nearest stars which are at least ten light-years away.

最近邻（Nearest Neighbor，NN）查询的高效实现是地理信息系统（GIS）中特别受关注的问题。例如，用户可能会在屏幕上指向一个特定位置或对象，并要求系统在数据库中查找与之最近的五个对象。另一种需要使用NN查询的情况是，用户不熟悉空间对象的布局。在天体物理数据库中，如果使用更传统的 $2\mathrm{D}$ 范围查询，查找天空中给定点的最近恒星可能需要多次使用不同窗口大小进行不成功的搜索。另一个更复杂的可以用NN技术处理的查询是，查找至少相距十光年的四颗最近恒星。

The versatility of $k$ nearest neighbors search increases substantially if we consider all variations of it, such as the $k$ furthest neighbors,or when it is combined with

如果考虑 $k$ 最近邻搜索的所有变体，例如 $k$ 最远邻搜索，或者将其与

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the ACM copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Association of Computing Machinery. To copy otherwise, or to republish, requires

允许免费复制本材料的全部或部分内容，但前提是复制的目的不是为了直接的商业利益，并且要保留ACM版权声明、出版物标题及其日期，并注明复制获得了美国计算机协会的许可。否则，复制或重新发布需要

© 1995 ACM 0-89791-731-6/95/0005..\$3.50 other spatial queries such as find the $\mathrm{{kNN}}$ to the East of a location, or even spatial joins with NN join predicate, such as find the three closest restaurants for each of two different movie theaters.

© 1995 ACM 0 - 89791 - 731 - 6/95/0005..\$3.50 其他空间查询，例如查找位于某个位置以东的 $\mathrm{{kNN}}$ 个对象，甚至是带有NN连接谓词的空间连接，例如为两家不同的电影院分别查找三家最近的餐厅。

Efficient processing of NN queries requires spatial data structures which capitalize on the proximity of the objects to focus the search of potential neighbors only. There is a wide variety of spatial access methods [Same89]. However, very few have been used for NN. In [Same90], heuristics are provided to find objects in quadtrees. The exact k-NN problem, is also posed for hierarchical spatial data structures such as the PM quadtree. The proposed solution is a top-down recursive algorithm which first goes down the quadtree, exploring the subtree that contains the query point, in order to get a first estimate of the NN location. Then it backtracks and explores remaining subtrees which potentially contain NN until no subtree needs be visited. In [FBF77] a NN algorithm for $k$ - $d$ -trees was proposed which was later refined in [Spro91].

高效处理NN查询需要利用对象的邻近性的空间数据结构，以便仅聚焦于潜在邻居的搜索。有各种各样的空间访问方法 [Same89]。然而，很少有方法用于NN查询。在 [Same90] 中，提供了在四叉树中查找对象的启发式方法。精确的k - NN问题也针对诸如PM四叉树之类的分层空间数据结构提出。所提出的解决方案是一种自顶向下的递归算法，该算法首先遍历四叉树，探索包含查询点的子树，以便对NN的位置进行初步估计。然后回溯并探索可能包含NN的其余子树，直到无需再访问任何子树为止。在 [FBF77] 中，提出了一种用于 $k$ - $d$ 树的NN算法，该算法后来在 [Spro91] 中得到了改进。

R-trees [Gutt84], Packed R-trees [Rous85], [Kame94], R-tree variations [SRF87], [Beck90] have been primarily used for overlap/containment range queries and spatial join queries [BKS93] based on overlap/containment. In this paper, we provide an efficient branch-and-bound search algorithm for processing exact k-NN queries for the R-trees, introduce several metrics for ordering and pruning the search tree, and perform several experiments on synthetic and real-world data to demonstrate the performance and scalability of our approach. To the best of our knowledge, neither NN algorithms have been developed for R-trees, nor similar metrics for NN search. We would also like to point out that, although the algorithm and these metrics are in the context of R-trees, they are directly applicable to all other spatial data structures.

R树 [Gutt84]、紧凑R树 [Rous85]、[Kame94]、R树变体 [SRF87]、[Beck90] 主要用于基于重叠/包含关系的重叠/包含范围查询和空间连接查询 [BKS93]。在本文中，我们为R树处理精确的k - NN查询提供了一种高效的分支限界搜索算法，引入了几种用于对搜索树进行排序和剪枝的度量方法，并在合成数据和真实世界数据上进行了几个实验，以证明我们方法的性能和可扩展性。据我们所知，尚未为R树开发过NN算法，也没有类似的用于NN搜索的度量方法。我们还想指出的是，尽管该算法和这些度量方法是在R树的背景下提出的，但它们可直接应用于所有其他空间数据结构。

Section 2 of the paper contains the theoretical foundation for the nearest neighbor search. Section 3 describes the algorithm and the metrics for ordering the search and pruning during it. Section 4 has the experiments with the implementation of the algorithm. The conclusion is in section 5 .

本文第2节包含最近邻搜索的理论基础。第3节描述了算法以及在搜索过程中对搜索进行排序和剪枝的度量方法。第4节是算法实现的实验。结论在第5节。

---

<!-- Footnote -->

*This research was sponsored partially by the National Science Foundation under grant BIR 9318183, by ARPA under contract 003195 Ve1001D, and by NASA/USRA under contract 5555-09.

*本研究部分由美国国家科学基金会（National Science Foundation）资助（资助号：BIR 9318183），由美国高级研究计划局（ARPA）资助（合同号：003195 Ve1001D），以及由美国国家航空航天局/美国大学空间研究协会（NASA/USRA）资助（合同号：5555 - 09）。

<!-- Footnote -->

---

## 2 NEAREST NEIGHBOR SEARCH USING R-TREES

## 2 使用R树进行最近邻搜索

R-trees were proposed as a natural extension of B-trees in higher than one dimensions [Gutt84]. They combine most of the nice features of both B-trees and quadtrees. Like B-trees, they remain balanced, while they maintain the flexibility of dynamically adjusting their grouping to deal with either dead-space or dense areas, like the quadtrees do. The decomposition used in R-trees is dynamic, driven by the spatial data objects. And with appropriate split algorithms, if a region of an n-dimensional space includes dead-space, no entry in the R-tree will be introduced.

R树是作为一维以上B树的自然扩展而提出的 [Gutt84]。它们结合了B树和四叉树的大部分优点。与B树一样，它们保持平衡，同时又像四叉树一样，能够动态调整分组，以处理空白区域或密集区域。R树中使用的分解是动态的，由空间数据对象驱动。通过适当的分裂算法，如果n维空间的某个区域包含空白区域，则R树中不会引入该区域的条目。

Leaf nodes of the R-tree contain entries of the form (RECT,oid) where oid is an object-identifier and is used as a a pointer to a data object and ${RECT}$ is an n-dimensional Minimal Bounding Rectangle (MBR) which bounds the corresponding object. For example, in a 2-dimensional space,an entry ${RECT}$ will be of the form $\left( {{x}_{\text{low }},{x}_{\text{high }},{y}_{\text{low }},{y}_{\text{high }}}\right)$ which represents the coordinates of the lower-left and upper-right corner of the rectangle. The possibly composite spatial objects stored at the leaf level are considered atomic, and are not further decomposed into their spatial primitives, i.e. quadrants, triangles, trapezoids, line segments, or pixels. Non-leaf R-tree nodes contain entries of the form (RECT,p)where $p$ is a pointer to a successor node in the next level of the R-tree,and ${RECT}$ is a minimal rectangle which bounds all the entries in the descendent node.

R树的叶子节点包含形式为(RECT,oid)的条目，其中oid是对象标识符，用作指向数据对象的指针，${RECT}$是一个n维最小边界矩形（MBR），用于界定相应的对象。例如，在二维空间中，条目${RECT}$的形式为$\left( {{x}_{\text{low }},{x}_{\text{high }},{y}_{\text{low }},{y}_{\text{high }}}\right)$，表示矩形左下角和右上角的坐标。存储在叶子层的可能复合的空间对象被视为原子对象，不会进一步分解为其空间基元，即象限、三角形、梯形、线段或像素。非叶子R树节点包含形式为(RECT,p)的条目，其中$p$是指向R树下一层后继节点的指针，${RECT}$是一个最小矩形，用于界定后代节点中的所有条目。

The term branching factor (or fan-out) can be used to specify the maximum number of entries that a node can have; each node of an R-tree with branching factor fifty, for example, points to a maximum of fifty descendents or leaf objects. To illustrate the way an R-tree is defined on some space, Figure 1 shows a collection of rectangles and Figure 2 the corresponding tree. Performance of an R-tree search is measured by the number of disk accesses (reads) necessary to find (or not find) the desired object(s) in the database. So, the R-tree branching factor is chosen such that the size of a node is equal to (or a multiple of) the size of a disk block or file system page.

术语分支因子（或扇出）可用于指定节点可以拥有的最大条目数；例如，分支因子为50的R树的每个节点最多指向50个后代节点或叶子对象。为了说明如何在某个空间上定义R树，图1展示了一组矩形，图2展示了相应的树。R树搜索的性能通过在数据库中查找（或未找到）所需对象所需的磁盘访问（读取）次数来衡量。因此，选择R树的分支因子，使得节点的大小等于磁盘块或文件系统页面的大小（或其倍数）。

### 2.1 Metrics for Nearest Neighbor Search

### 2.1 最近邻搜索的度量标准

Given a query point $\mathrm{P}$ and an object $\mathrm{O}$ enclosed in its MBR, we provide two metrics for ordering the NN search. The first one is based on the minimum distance (MINDIST) of the object O from P. The second metric is based on the minimum of the maximum possible distances (MINMAXDIST) from $\mathrm{P}$ to a face (or vertex) of the MBR containing O. MINDIST and MINMAXDIST offer a lower and an upper bound on the actual distance of $\mathrm{O}$ from $\mathrm{P}$ respectively. These bounds are used by the nearest neighbor algorithm to order and efficiently prune the paths of the search space in an R-tree.

给定一个查询点$\mathrm{P}$和一个包含在其MBR中的对象$\mathrm{O}$，我们提供两种度量标准来对最近邻搜索进行排序。第一种基于对象O到P的最小距离（MINDIST）。第二种度量标准基于从$\mathrm{P}$到包含O的MBR的一个面（或顶点）的最大可能距离的最小值（MINMAXDIST）。MINDIST和MINMAXDIST分别为$\mathrm{O}$到$\mathrm{P}$的实际距离提供了下界和上界。最近邻算法使用这些边界对R树搜索空间的路径进行排序并有效剪枝。

<!-- Media -->

<!-- figureText: $\mathbf{F}$ K G D $\mathrm{N}$ -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_1.jpg?x=1060&y=301&w=479&h=488&r=0"/>

Figure 1: Collection of Rectangles

图1：矩形集合

<!-- figureText: C J K L M $\mathrm{N}$ D $\mathrm{H}$ $\mathbf{I}$ -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_1.jpg?x=959&y=1404&w=691&h=263&r=0"/>

Figure 2: R-tree Construction

图2：R树构建

<!-- Media -->

Definition 1 $A$ rectangle $R$ in Euclidean space $E\left( n\right)$ of dimension $n$ ,will be defined by the two endpoints $S$ and $T$ of its major diagonal:

定义1 在维度为$n$的欧几里得空间$E\left( n\right)$中的矩形$R$，将由其主对角线的两个端点$S$和$T$定义：

$$
R = \left( {S,T}\right) 
$$

$$
\text{where}S = \left\lbrack  {{s}_{1},{s}_{2},\ldots ,{s}_{n}}\right\rbrack  \text{and}T = \left\lbrack  {{t}_{1},{t}_{2},\ldots ,{t}_{n}}\right\rbrack  
$$

$$
\text{and}{s}_{i} \leq  {t}_{i}\text{for}1 \leq  i \leq  n\text{.}
$$

Minimum Distance (MINDIST) The first metric we introduce is a variation of the classic Euclidean distance applied to a point and a rectangle. If the point is inside the rectangle, the distance between them is zero. If the point is outside the rectangle, we use the square of the Euclidean distance between the point and the nearest edge of the rectangle. We use the square of the Euclidean distance because it involves fewer and less costly computations. To avoid confusion, whenever we refer to distance in this paper, we will in practice be using the square of the distance and the construction of our metrics will reflect this.

最小距离（MINDIST） 我们引入的第一个度量标准是经典欧几里得距离应用于点和矩形的一种变体。如果点在矩形内部，则它们之间的距离为零。如果点在矩形外部，我们使用点与矩形最近边之间的欧几里得距离的平方。我们使用欧几里得距离的平方是因为它涉及的计算更少且成本更低。为避免混淆，在本文中，每当我们提到距离时，实际上我们将使用距离的平方，并且我们的度量标准的构建将反映这一点。

Definition 2 The distance of a point $P$ in $E\left( n\right)$ from a rectangle $R$ in the same space,denoted $\operatorname{MINDIST}\left( {P,R}\right)$ , is:

定义2 空间$E\left( n\right)$中的点$P$到同一空间中的矩形$R$的距离，记为$\operatorname{MINDIST}\left( {P,R}\right)$，为：

$$
\operatorname{MINDIST}\left( {P,R}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}{\left| {p}_{i} - {r}_{i}\right| }^{2}
$$

where

其中

$$
{r}_{i} = \left\{  \begin{array}{ll} {s}_{i} & \text{ if }{p}_{i} < {s}_{i} \\  {t}_{i} & \text{ if }{p}_{i} > {t}_{i} \\  {p}_{i} & \text{ otherwise } \end{array}\right. 
$$

Lemma 1 The distance of definition 2 is equal to the square of the minimal Euclidean distance from $P$ to any point on the perimeter of $R$ .

引理1 定义2中的距离等于从$P$到$R$周长上任意点的最小欧几里得距离的平方。

Proof: If $\mathrm{P}$ is inside $\mathrm{R}$ ,then MINDIST $= 0$ which is less than or equal to the distance of $P$ from any point on the perimeter of $R$ . If $P$ is on the perimeter,again MINDIST $= 0$ and so is equal to the square of minimal Euclidean distance of $P$ from its closest point on the perimeter, namely itself.

证明：如果$\mathrm{P}$在$\mathrm{R}$内部，那么最小距离$= 0$小于或等于$P$到$R$边界上任意一点的距离。如果$P$在边界上，同样最小距离为$= 0$，且等于$P$到其在边界上最近点（即其自身）的最小欧几里得距离的平方。

If $\mathrm{P}$ is outside $\mathrm{R}$ and $j$ coordinates, $j = 1,2,\ldots ,n - 1$ of P satisfy ${s}_{j} \leq  {p}_{j} \leq  {t}_{j}$ ,then MINDIST measures the square of the length of a perpendicular segment from $\mathrm{P}$ to an edge,for $j = 1$ or to a plane for $j = 2$ ,or a hyperface for $j \geq  3$ . If none of the ${p}_{j}$ coordinates fall between $\left( {{s}_{i},{t}_{i}}\right)$ ,then MINDIST is the square of the distance to the closest vertex of $\mathrm{R}$ by the way of selecting ${r}_{i}$ .

如果$\mathrm{P}$在$\mathrm{R}$外部，且$j$坐标，点P的$j = 1,2,\ldots ,n - 1$满足${s}_{j} \leq  {p}_{j} \leq  {t}_{j}$，那么最小距离衡量的是从$\mathrm{P}$到一条边（对于$j = 1$）、一个平面（对于$j = 2$）或一个超面（对于$j \geq  3$）的垂线段长度的平方。如果没有一个${p}_{j}$坐标落在$\left( {{s}_{i},{t}_{i}}\right)$之间，那么最小距离是通过选择${r}_{i}$到$\mathrm{R}$最近顶点的距离的平方。

Notice that computing MINDIST requires only linear in the number of dimensions, $O\left( n\right)$ ,operations.

注意，计算最小距离仅需要与维度数量$O\left( n\right)$呈线性关系的操作。

Definition 3 The minimum distance of a point $P$ from a spatial object o,denoted by $\parallel \left( {P,o}\right) \parallel$ ,is:

定义3：点$P$到空间对象o的最小距离，用$\parallel \left( {P,o}\right) \parallel$表示，为：

$$
\parallel \left( {P,o}\right) \parallel  = \min \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\left| {p}_{i} - {x}_{i}\right| }^{2},}\right. 
$$

$$
\forall X = \left\lbrack  {{x}_{1},\ldots ,{x}_{n}}\right\rbrack   \in  O).
$$

Theorem 1 Given a point $P$ and an ${MBRR}$ enclosing a set of objects $O = \left\{  {{o}_{i},1 \leq  i \leq  m}\right\}$ ,the following is true:

定理1：给定一个点$P$和一个包含一组对象$O = \left\{  {{o}_{i},1 \leq  i \leq  m}\right\}$的${MBRR}$，以下结论成立：

## $\forall o \in  O,\operatorname{MINDIST}\left( {P,R}\right)  \leq  \parallel \left( {P,o}\right) \parallel$

Proof: If $\mathrm{P}$ is inside $\mathrm{R}$ ,then MINDIST $= 0$ which is less than the distance of any object within $\mathrm{R}$ including one that may be touching $P$ . If $P$ is outside $R$ ,then according to lemma $1,\forall X$ on the perimeter of $\mathrm{R}$ , $\operatorname{MINDIST}\left( {P,R}\right)  \leq  \parallel \left( {P,X}\right) \parallel$

证明：如果$\mathrm{P}$在$\mathrm{R}$内部，那么最小距离$= 0$小于$\mathrm{R}$内任何对象（包括可能与$P$接触的对象）的距离。如果$P$在$R$外部，那么根据$\mathrm{R}$边界上的引理$1,\forall X$，$\operatorname{MINDIST}\left( {P,R}\right)  \leq  \parallel \left( {P,X}\right) \parallel$

MINDIST is used to determine the closest object to $\mathrm{P}$ from all those enclosed in $\mathrm{R}$ . The equality in the above theorem holds when an object of $R$ touches the circle with center $\mathrm{P}$ and radius the square root of MINDIST.

最小距离用于从$\mathrm{R}$所包含的所有对象中确定离$\mathrm{P}$最近的对象。当$R$中的一个对象与以$\mathrm{P}$为圆心、以最小距离的平方根为半径的圆相切时，上述定理中的等式成立。

When searching an R-tree for the nearest neighbor to a query point $P$ ,at each visited node of the R-tree, one must decide which MBR to search first. MINDIST offers a first approximation of the NN distance to every MBR of the node and, therefore, can be used to direct the search.

在R树中搜索查询点$P$的最近邻时，在R树的每个访问节点处，必须决定首先搜索哪个最小边界矩形（MBR）。最小距离为节点的每个最小边界矩形提供了最近邻距离的初步近似值，因此可用于指导搜索。

In general, deciding which MBR to visit first in order to minimize the total number of visited nodes is not that straightforward. In fact, in many cases, due to dead space inside the MBRs, the NN might be much further than MINDIST, and visiting first the MBR with the smallest MINDIST may result in false-drops, i.e. visits to unnecessary nodes. For this reason, we introduce a second metric MINMAXDIST. But first, the following lemma is necessary.

一般来说，为了最小化访问节点的总数而决定首先访问哪个最小边界矩形并非那么简单。实际上，在许多情况下，由于最小边界矩形内部存在空白空间，最近邻可能比最小距离远得多，并且首先访问最小距离最小的最小边界矩形可能会导致误判，即访问不必要的节点。出于这个原因，我们引入第二个度量指标——最小最大距离（MINMAXDIST）。但首先，需要以下引理。

Lemma 2 The MBR Face Property: Every face (i.e. edge in dimension 2, rectangle in dimension 3 and 'hyperface' in higher dimensions) of any MBR (at any level of the R-tree) contains at least one point of some spatial object in the DB. (See figures 3 and 4).

引理2：最小边界矩形面属性：任何最小边界矩形（在R树的任何层级）的每个面（即二维中的边、三维中的矩形以及更高维度中的“超面”）都包含数据库中某个空间对象的至少一个点。（见图3和图4）

Proof: At the leaf level in the R-tree (object level), assume by contradiction that one face of the enclosing MBR does not touch the enclosed object. Then, there exists a smaller rectangle that encloses the object which contradicts the definition of the Minimum Bounding Rectangle. For the non-leaf levels, we use an induction on the level in the tree of the MBR. Assume any level $k \geq  0$ MBR has the MBR face property,and consider an MBR at level $k + 1$ . By the definition of an MBR,each face of that MBR touches an MBR of lower level, and therefore, with a leaf object by applying the inductive hypothesis.

证明：在R树的叶子节点层（对象层），通过反证法假设包围最小边界矩形（MBR）的一个面不与被包围的对象接触。那么，存在一个更小的矩形可以包围该对象，这与最小边界矩形的定义相矛盾。对于非叶子节点层，我们对MBR所在树的层级进行归纳。假设任意层级$k \geq  0$的MBR具有MBR面属性，并考虑层级为$k + 1$的一个MBR。根据MBR的定义，该MBR的每个面都与一个更低层级的MBR接触，因此，通过应用归纳假设，也与一个叶子对象接触。

<!-- Media -->

<!-- figureText: MBR Level -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_3.jpg?x=162&y=201&w=716&h=670&r=0"/>

Each edge of the MBR at level i is in contact with a graphic object of the DB. (The same property applies for the MBRs at level $i + 1$ )

层级i的MBR的每条边都与数据库（DB）中的一个图形对象接触。（相同的属性适用于层级$i + 1$的MBR）

Figure 3: MBR Face Property in 2-Space

图3：二维空间中的MBR面属性

<!-- figureText: Enclosed MBRs or Objects Enclosing MBR -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_3.jpg?x=166&y=1290&w=724&h=561&r=0"/>

Figure 4: MBR Face Property in 3-Space

图4：三维空间中的MBR面属性

<!-- Media -->

Minimax Distance (MINMAXDIST) In order to avoid visiting unnecessary MBRs, we should have an upper bound of the NN distance to any object inside an MBR. This will allow us to prune MBRs that have MINDIST higher than this upper bound. The following distance construction (called MINMAXDIST) is being introduced to compute the minimum value of all the maximum distances between the query point and points on the each of the $n$ axes respectively. The MINMAXDIST guarantees there is an object within the MBR at a distance less than or equal to MINMAXDIST.

最小最大距离（MINMAXDIST） 为了避免访问不必要的MBR，我们应该有一个到MBR内任何对象的最近邻（NN）距离的上界。这将使我们能够修剪那些最小距离（MINDIST）高于此上界的MBR。下面引入的距离构造（称为MINMAXDIST）用于分别计算查询点与每个$n$轴上的点之间所有最大距离的最小值。MINMAXDIST保证MBR内存在一个对象，其距离小于或等于MINMAXDIST。

Definition 4 Given a point $P$ in $E\left( n\right)$ and an ${MBR}$ $R = \left( {S,T}\right)$ of the same dimensionality,we define MINMAXDIST(P,R) as:

定义4 给定$E\left( n\right)$中的一个点$P$和一个相同维度的${MBR}$ $R = \left( {S,T}\right)$，我们将MINMAXDIST(P,R)定义为：

$$
\operatorname{MINMAXDIST}\left( {P,R}\right)  = 
$$

$$
\mathop{\min }\limits_{{1 \leq  k \leq  n}}\left( {{\left| {p}_{k} - r{m}_{k}\right| }^{2} + \mathop{\sum }\limits_{\substack{{i \neq  k} \\  {1 \leq  i \leq  n} }}{\left| {p}_{i} - r{M}_{i}\right| }^{2}}\right) 
$$

where:

其中：

$$
r{m}_{k} = \left\{  {\begin{array}{ll} {s}_{k} & \text{ if }{p}_{k} \leq  \frac{\left( {s}_{k} + {t}_{k}\right) }{2}; \\  {t}_{k} & \text{ otherwise. } \end{array}\text{ and }}\right. 
$$

$$
r{M}_{i} = \left\{  \begin{array}{ll} {s}_{i} & \text{ if }{p}_{i} \geq  \frac{\left( {s}_{i} + {t}_{i}\right) }{2} \\  {t}_{i} & \text{ otherwise } \end{array}\right. 
$$

This construction can be described as follows: For each $k$ select the hyperplane ${H}_{k} = r{m}_{k}$ which contains the closer of the two faces of the MBR orthogonal to the ${k}^{th}$ space axis. (One of these faces has ${H}_{k} = {S}_{k}$ and the other has $\left. {{H}_{k} = {T}_{k}}\right)$ . The point ${V}_{k} = \left( {r{M}_{1},r{M}_{2},\ldots ,r{M}_{k - 1},r{m}_{k},r{M}_{k + 1},\ldots ,r{M}_{n}}\right)$ ,is the farthest vertex from $P$ on this face. MINMAXDIST then, is the minimum of the squares of the distance to each of these points.

这种构造可以描述如下：对于每个$k$，选择包含与${k}^{th}$空间轴正交的MBR的两个面中较近的那个面的超平面${H}_{k} = r{m}_{k}$。（其中一个面的坐标为${H}_{k} = {S}_{k}$，另一个面的坐标为$\left. {{H}_{k} = {T}_{k}}\right)$。点${V}_{k} = \left( {r{M}_{1},r{M}_{2},\ldots ,r{M}_{k - 1},r{m}_{k},r{M}_{k + 1},\ldots ,r{M}_{n}}\right)$是该面上距离$P$最远的顶点。然后，MINMAXDIST是到这些点的距离的平方的最小值。

Notice that this expression can be efficiently implemented,in $O\left( n\right)$ operations by first computing $S =$ $\mathop{\sum }\limits_{{1 < i < n}}{\left| {p}_{i} - r{M}_{i}\right| }^{2}$ ,(the distance from $P$ to the furthest vertex on the MBR), then iteratively selecting the minimum of $S - {\left| {p}_{k} - r{M}_{k}\right| }^{2} + {\left| {p}_{k} - r{m}_{k}\right| }^{2}$ for $1 \leq  k \leq  n$ .

注意，这个表达式可以通过以下方式高效实现：首先计算$S =$ $\mathop{\sum }\limits_{{1 < i < n}}{\left| {p}_{i} - r{M}_{i}\right| }^{2}$（从$P$到MBR上最远顶点的距离），然后迭代选择$1 \leq  k \leq  n$时$S - {\left| {p}_{k} - r{M}_{k}\right| }^{2} + {\left| {p}_{k} - r{m}_{k}\right| }^{2}$的最小值，总共需要$O\left( n\right)$次操作。

Theorem 2 Given a point $P$ and an MBR $R$ enclosing a set of objects $O = \left\{  {{o}_{i},1 \leq  i \leq  m}\right\}$ ,the following property holds: $\exists o \in  O,\parallel \left( {P,o}\right) \parallel  \leq  \operatorname{MINMAXDIST}\left( {P,R}\right)$ .

定理2 给定一个点$P$和一个包围一组对象$O = \left\{  {{o}_{i},1 \leq  i \leq  m}\right\}$的MBR $R$，以下属性成立：$\exists o \in  O,\parallel \left( {P,o}\right) \parallel  \leq  \operatorname{MINMAXDIST}\left( {P,R}\right)$。

Proof: Because of lemma 2, we know that each MBR's face is touching at least one object within the MBR. Since the definition of MINMAXDIST produces an estimate of the NN distance to an object touching its MBR at the extremity of one of its faces, this guarantees that MINMAXDIST is greater than or equal to the NN distance. On the other hand, a point object located exactly at the vertex of the MBR at distance MINMAXDIST would contradict the proposition that one could find a smaller distance than MINMAXDIST as an upper bound of the NN distance.

证明：由于引理2，我们知道每个MBR的面至少与MBR内的一个对象接触。因为MINMAXDIST的定义产生了到一个在其某个面的端点处与MBR接触的对象的最近邻距离的估计，这保证了MINMAXDIST大于或等于最近邻距离。另一方面，一个恰好位于距离为MINMAXDIST的MBR顶点处的点对象将与可以找到一个小于MINMAXDIST的距离作为最近邻距离的上界这一命题相矛盾。

Theorem 2 says that MINMAXDIST is the minimum distance that guarantees the presence of an object $O$ in $R$ whose distance from $P$ is within this distance. A value larger or equal to MINMAXDIST would always 'catch' some object inside an MBR, but a smaller distance could 'miss' some object.

定理2表明，最小最大距离（MINMAXDIST）是一个最小距离，它确保在$R$中存在一个对象$O$，该对象与$P$的距离在这个距离之内。一个大于或等于最小最大距离（MINMAXDIST）的值总能“捕获”最小边界矩形（MBR）内的某个对象，但更小的距离可能会“错过”某些对象。

Figures 5 and 6 illustrate MINDIST and MIN-MAXDIST in a 2-Space and 3-Space respectively.

图5和图6分别展示了二维空间和三维空间中的最小距离（MINDIST）和最小最大距离（MINMAXDIST）。

<!-- Media -->

<!-- figureText: MBR MINDIST -0 Query Point MINDIST _____ MINMAXDIST MINMAXDIST MBR MINDIST MINDIST MBR -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_4.jpg?x=168&y=830&w=717&h=490&r=0"/>

Figure 5: MINDIST and MINMAXDIST in 2-Space

图5：二维空间中的最小距离（MINDIST）和最小最大距离（MINMAXDIST）

<!-- figureText: MINMAXDIST (P, R) MINDIST (P, R) Query Point: P Rectangle: $R$ Moreoverserate _____。 -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_4.jpg?x=171&y=1481&w=715&h=443&r=0"/>

Figure 6: MINDIST and MINMAXDIST 3-Space

图6：三维空间中的最小距离（MINDIST）和最小最大距离（MINMAXDIST）

<!-- Media -->

## 3 Nearest Neighbor Algorithm for R-trees

## 3 R树的最近邻算法

In this section we present a branch-and-bound R-tree traversal algorithm to find the k-NN objects to a given query point. We first discuss the merits of using the MINDIST and MINMAXDIST metrics to order and prune the search tree, then we present the algorithm for finding 1-NN and finally, generalize the algorithm for finding the $\mathrm{k} - \mathrm{{NN}}$ .

在本节中，我们提出一种分支限界的R树遍历算法，用于查找给定查询点的k个最近邻对象。我们首先讨论使用最小距离（MINDIST）和最小最大距离（MINMAXDIST）度量来对搜索树进行排序和剪枝的优点，然后给出查找1个最近邻的算法，最后将该算法推广到查找$\mathrm{k} - \mathrm{{NN}}$的情况。

### 3.1 MINDIST and MINMAXDIST for Ordering and Pruning the Search

### 3.1 用于搜索排序和剪枝的最小距离（MINDIST）和最小最大距离（MINMAXDIST）

Branch-and-bound algorithms have been studied and used extensively in the areas of artificial intelligence and operations research [HS78]. If the ordering and pruning heuristics are chosen well, they can significantly reduce the number of nodes visited in a large search space.

分支限界算法在人工智能和运筹学领域得到了广泛的研究和应用[HS78]。如果排序和剪枝启发式方法选择得当，它们可以显著减少在大型搜索空间中访问的节点数量。

Search Ordering: The heuristics we use in our algorithm and in the following experiments are based on orderings of the MINDIST and MINMAXDIST metrics. The MINDIST ordering is the optimistic choice, while the MINMAXDIST metric is the pessimistic (though not worst case) one. Since MINDIST estimates the distance from the query point to any enclosed MBR or data object as the minimum distance from the point to the MBR itself, it is the most optimistic choice possible. Due to the properties of MBRs and the construction of it, MINMAXDIST produces the most pessimistic ordering that need ever be considered.

搜索排序：我们在算法和后续实验中使用的启发式方法基于最小距离（MINDIST）和最小最大距离（MINMAXDIST）度量的排序。最小距离（MINDIST）排序是一种乐观的选择，而最小最大距离（MINMAXDIST）度量则是一种悲观的选择（尽管不是最坏情况）。由于最小距离（MINDIST）将查询点到任何包含的最小边界矩形（MBR）或数据对象的距离估计为该点到最小边界矩形（MBR）本身的最小距离，因此它是最乐观的选择。由于最小边界矩形（MBR）的性质及其构造方式，最小最大距离（MINMAXDIST）产生的排序是最悲观的，也是需要考虑的排序。

In applying a depth first traversal to find the NN to a query point in an $\mathrm{R}$ -tree,the optimal MBR visit ordering depends not only on the distance from the query point to each of the MBRs along the path(s) from the root to the leaf node(s), but also on the size and layout of the MBRs (or in the leaf node case, objects) within each MBR. In particular, one can construct examples in which the MINDIST metric ordering produces tree traversals that are more costly (in terms of nodes visited) than the MINMAXDIST metric.

在对$\mathrm{R}$树进行深度优先遍历来查找查询点的最近邻时，最优的最小边界矩形（MBR）访问顺序不仅取决于查询点到从根节点到叶节点路径上每个最小边界矩形（MBR）的距离，还取决于每个最小边界矩形（MBR）内最小边界矩形（MBR）的大小和布局（或者在叶节点的情况下，对象的大小和布局）。特别是，可以构造出这样的例子，其中最小距离（MINDIST）度量排序产生的树遍历（就访问的节点而言）比最小最大距离（MINMAXDIST）度量的成本更高。

This is shown in figure 7, where the MINDIST metric ordering will lead the search to MBR1 which would require of opening up M11 and M12. If on the other hand, MINMAXDIST metric ordering was used, visiting MBR2 would result in an smaller estimate of the actual distance to the NN (which will be found to be in M21) which will then eliminate the need to examine M11 and M12. The MINDIST ordering optimistically assumes that the NN to $P$ in MBR $M$ is going to be close to $\operatorname{MINDIST}\left( {M,P}\right)$ ,which is not always the case. Similarly, counterexamples could be constructed for any predefined ordering.

如图7所示，最小距离（MINDIST）度量排序会引导搜索到最小边界矩形1（MBR1），这需要打开M11和M12。另一方面，如果使用最小最大距离（MINMAXDIST）度量排序，访问最小边界矩形2（MBR2）将得到到最近邻实际距离的较小估计值（最终会发现最近邻在M21中），这样就无需检查M11和M12。最小距离（MINDIST）排序乐观地假设最小边界矩形$M$中到$P$的最近邻会靠近$\operatorname{MINDIST}\left( {M,P}\right)$，但实际情况并非总是如此。同样，可以为任何预定义的排序构造反例。

As we stated above, the MINDIST metric produces most optimistic ordering, but that is not always the best choice. Many other orderings are possible by choosing metrics which compute the distance from the query point to faces or vertices of the MBR which are further away. The importance of MINMAXDIST(P,M) is that it computes the smallest distance between point $P$ and MBR $M$ that guarantees the finding of an object in $M$ at a Euclidean distance less than or equal to $\operatorname{MINMAXDIST}\left( {P,M}\right)$ .

正如我们上面所述，最小距离（MINDIST）度量产生最乐观的排序，但这并不总是最佳选择。通过选择计算查询点到最小边界矩形（MBR）更远面或顶点距离的度量，还可以有许多其他排序方式。最小最大距离（MINMAXDIST(P,M)）的重要性在于，它计算点$P$和最小边界矩形$M$之间的最小距离，该距离确保能找到$M$中一个欧几里得距离小于或等于$\operatorname{MINMAXDIST}\left( {P,M}\right)$的对象。

<!-- Media -->

<!-- figureText: -NN 1 Query Point M21 MBR2 The NN is somewhere in there. 1. MINDIST ordering: if we visit MBR1 first, we have to visit M11, M12, MBR1 M12 MBR2 and M21 before finding the NN. 2. MINMAXDIST ordering: If we visit MBR2 first, and then M21, when we eventually visit MBR1, we can prune M11 and M12. -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_5.jpg?x=162&y=104&w=702&h=460&r=0"/>

Figure 7: MINDIST is not always the better ordering

图7：最小距离（MINDIST）并不总是更好的排序方式

<!-- Media -->

Search Pruning: We utilize the two theorems we developed to formulate the following three strategies to prune MBRs during the search:

搜索剪枝：我们利用所推导的两个定理，制定以下三种在搜索过程中对最小边界矩形（MBR）进行剪枝的策略：

1. an MBR M with MINDIST(P,M) greater than the MINMAXDIST(P,M') of another MBR M' is discarded because it cannot contain the NN (theorems 1 and 2). We use this in downward pruning.

1. 若一个最小边界矩形M的最小距离（MINDIST(P,M)）大于另一个最小边界矩形M'的最小最大距离（MINMAXDIST(P,M')），则丢弃该最小边界矩形M，因为它不可能包含最近邻（定理1和定理2）。我们在向下剪枝中使用此策略。

2. an actual distance from $\mathrm{P}$ to a given object $\mathrm{O}$ which is greater than the MINMAXDIST(P,M)for an MBR M can be discarded (actually replaced by it as an estimate of the NN distance) because M contains an object $O$ ’ which is nearer to $P$ (theorem 2). This is also used in downward pruning.

2. 若从$\mathrm{P}$到给定对象$\mathrm{O}$的实际距离大于某个最小边界矩形M的最小最大距离（MINMAXDIST(P,M)），则可以丢弃该实际距离（实际上用它作为最近邻距离的估计值来替代），因为M中包含一个更接近$P$的对象$O$'（定理2）。这也用于向下剪枝。

3. every MBR M with MINDIST(P,M) greater than the actual distance from $\mathrm{P}$ to a given object $\mathrm{O}$ is discarded because it cannot enclose an object nearer than $\mathrm{O}$ (theorem 1). We use this in upward pruning.

3. 所有最小距离（MINDIST）函数值 MINDIST(P, M) 大于从 $\mathrm{P}$ 到给定对象 $\mathrm{O}$ 的实际距离的最小边界矩形（MBR）M 都会被舍弃，因为它不可能包含比 $\mathrm{O}$ 更近的对象（定理 1）。我们在向上剪枝中使用这一规则。

Although we specify only the use of MINMAXDIST in downward pruning, in practice, there are situations where it is better to apply MINDIST (and in fact strategy 3) instead. For example, when there is no dead space (or at least very little) in the nodes of the R-tree, MINDIST is a much better estimate of $\parallel \left( {P,N}\right) \parallel$ ,the actual distance to the NN than is MINMAXDIST, at all levels in the tree. So, it will prune more candidate MBRs than will MINMAXDIST.

虽然我们仅指定了在向下剪枝中使用最大最小距离（MINMAXDIST），但实际上，在某些情况下，应用最小距离（MINDIST）（实际上是策略 3）会更好。例如，当 R 树的节点中没有空白区域（或者至少非常少）时，在树的所有层级上，最小距离（MINDIST）比最大最小距离（MINMAXDIST）更能准确估计 $\parallel \left( {P,N}\right) \parallel$ 到最近邻（NN）的实际距离。因此，与最大最小距离（MINMAXDIST）相比，它能剪枝更多的候选最小边界矩形（MBR）。

### 3.2 Nearest Neighbor Search Algorithm

### 3.2 最近邻搜索算法

The algorithm presented here implements an ordered depth first traversal. It begins with the R-tree root node and proceeds down the tree. Originally, our guess for the nearest neighbor distance (call it Nearest) is infinity. During the descending phase, at each newly visited non-leaf node, the algorithm computes the ordering metric bounds (e.g. MINDIST, Definition 2) for all its MBRs and sorts them (associated with their corresponding node) into an Active Branch List (ABL). We then apply pruning strategies 1 and 2 to the ABL to remove unnecessary branches. The algorithm iterates on this ABL until the ABL is empty: For each iteration, the algorithm selects the next branch in the list and applies itself recursively to the node corresponding to the MBR of this branch. At a leaf node (DB objects level), the algorithm calls a type specific distance function for each object and selects the smaller distance between current value of Nearest and each computed value and updates Nearest appropriately. At the return from the recursion, we take this new estimate of the NN and apply pruning strategy 3 to remove all branches with $\operatorname{MINDIST}\left( {P,M}\right)  >$ Nearest for all MBRs M in the ABL.

这里提出的算法实现了一种有序的深度优先遍历。它从 R 树的根节点开始，然后向下遍历树。最初，我们对最近邻距离的猜测（称之为 Nearest）设为无穷大。在向下遍历阶段，在每个新访问的非叶节点处，算法会计算该节点所有最小边界矩形（MBR）的排序度量边界（例如，最小距离（MINDIST），定义 2），并将它们（与对应的节点关联）排序到一个活动分支列表（ABL）中。然后，我们对活动分支列表（ABL）应用剪枝策略 1 和 2，以移除不必要的分支。算法会在这个活动分支列表（ABL）上迭代，直到列表为空：对于每次迭代，算法会选择列表中的下一个分支，并递归地对该分支的最小边界矩形（MBR）对应的节点应用该算法。在叶节点（数据库对象层），算法会为每个对象调用特定类型的距离函数，并选择当前 Nearest 值和每个计算值中的较小距离，并相应地更新 Nearest。在递归返回时，我们采用这个新的最近邻估计值，并应用剪枝策略 3，从活动分支列表（ABL）中移除所有最小边界矩形（MBR）的 $\operatorname{MINDIST}\left( {P,M}\right)  >$ 大于 Nearest 的分支。

See Figure 8 for the pseudo-code description of the algorithm.

算法的伪代码描述见图 8。

### 3.3 Generalization: Finding the $k$ Nearest Neighbors

### 3.3 推广：查找 $k$ 个最近邻

The algorithm presented above can be easily generalized to answer queries of the type: Find The k Nearest Neighbors to a given Query Point,where $\mathrm{k}$ is greater than zero.

上述算法可以很容易地推广到回答以下类型的查询：查找给定查询点的 k 个最近邻，其中 $\mathrm{k}$ 大于零。

The only differences are:

唯一的区别是：

- A sorted buffer of at most $k$ current nearest neighbors is needed.

- 需要一个最多包含 $k$ 个当前最近邻的排序缓冲区。

- The MBRs pruning is done according to the distance of the furthest nearest neighbor in this buffer.

- 最小边界矩形（MBR）的剪枝是根据该缓冲区中最远的最近邻的距离进行的。

The next section provides experimental results using both MINDIST and MINMAXDIST.

下一节提供了同时使用最小距离（MINDIST）和最大最小距离（MINMAXDIST）的实验结果。

## 4 Experimental Results

## 4 实验结果

We implemented our k-NN search algorithm and designed and carried out our experiments in order to demonstrate the capability and usefulness of our NN search approach as applied to GIS type of queries. We examined the behavior of our algorithm as the number of neighbors increased, the cardinality of the data set size grew, and how the MINDIST and MINMAXDIST metrics affected performance.

我们实现了 k - 最近邻（k - NN）搜索算法，并设计和进行了实验，以证明我们的最近邻（NN）搜索方法应用于地理信息系统（GIS）类型查询时的能力和实用性。我们研究了随着邻居数量的增加、数据集规模的增大，我们的算法的行为，以及最小距离（MINDIST）和最大最小距离（MINMAXDIST）度量对性能的影响。

We performed our experiments on both publically available real-world spatial data sets and synthetic data sets. The real-world data sets included segment based

我们在公开可用的真实世界空间数据集和合成数据集上进行了实验。真实世界数据集包括基于线段的

<!-- Media -->

---

RECURSIVE PROCEDURE

递归过程

nearestNeighborSearch (Node, Point, Nearest)

最近邻搜索（节点，点，最近邻距离）

NODE // Current NODE

节点 // 当前节点

POINT Point // Search POINT

点 点 // 搜索点

NEARESTN Nearest // Nearest Neighbor

最近邻 最近的 // 最近邻

		//Local Variables

		// 局部变量

		NODE newNode

		节点 新节点

		BRANCHARRAY branchList

		分支数组 分支列表

		integer dist, last, i

		整数 距离, 最后一个, i

		// At leaf level - compute distance to actual objects

		// 在叶节点级别 - 计算到实际对象的距离

		If Node.type = LEAF

		如果 节点类型 = 叶节点

		Then

		那么

			For $\mathrm{i} \mathrel{\text{:=}} 1$ to Node.count

			从 $\mathrm{i} \mathrel{\text{:=}} 1$ 到 节点计数

					dist $\mathrel{\text{:=}}$ objectDIST(Point,Node.branchi.rect)

					距离 $\mathrel{\text{:=}}$ 对象距离(点, 节点.分支i.矩形)

					If (dist < Nearest.dist)

					如果 (距离 < 最近邻.距离)

						Nearest.dist $\mathrel{\text{:=}}$ dist

						最近邻.距离 $\mathrel{\text{:=}}$ 距离

						Nearest.rect $\mathrel{\text{:=}}$ Node.branchi.rect

						最近矩形 $\mathrel{\text{:=}}$ 节点分支矩形

		// Non-leaf level - order, prune and visit nodes

		// 非叶子节点层 - 排序、剪枝并访问节点

		Else

		否则

			// Generate Active Branch List

			// 生成活动分支列表

			genBranchList(Point, Node, branchList)

			生成分支列表(点, 节点, 分支列表)

			// Sort ABL based on ordering metric values

			// 根据排序度量值对活动分支列表进行排序

			sortBranchList(branchList)

			对分支列表排序(分支列表)

			// Perform Downward Pruning

			// 执行向下剪枝

			// (may discard all branches)

			// (可能会丢弃所有分支)

			last $=$ pruneBranchList(Node,Point,Nearest,

			最后 $=$ 剪枝分支列表(节点,点,最近节点,

						branchList)

						分支列表)

			// Iterate through the Active Branch List

			// 遍历活动分支列表

			For $\mathrm{i} \mathrel{\text{:=}} 1$ to last

			从 $\mathrm{i} \mathrel{\text{:=}} 1$ 到最后

					newNode $\mathrel{\text{:=}}$ Node.branchList,

					新节点 $\mathrel{\text{:=}}$ 节点.分支列表,

					// Recursively visit child nodes

					// 递归访问子节点

					nearestNeighborSearch(newNode, Point,

					最近邻搜索(新节点, 点,

						Nearest)

						最近邻)

					// Perform Upward Pruning

					// 执行向上剪枝

					last $\mathrel{\text{:=}}$ pruneBranchList(Node,Point,Nearest,

					最后 $\mathrel{\text{:=}}$ 修剪分支列表(节点, 点, 最近邻,

						branchList)

						分支列表)

	Figure 8: Nearest Neighbor Search Pseudo-Code

	图 8：最近邻搜索伪代码

---

<!-- Media -->

TIGER data files for the city of Long Beach, CA and Montgomery County, MD, and observation records from the International Ultraviolet Explorer (I.U.E) satellite from N.A.S.A. Our examples will be from the Long Beach data,which consists of 55,000 street segments stored as pairs of latitude and longitude coordinates. For the synthetic data experiments, we generated test data files of $1\mathrm{\;K},2\mathrm{\;K},4\mathrm{\;K},8\mathrm{\;K},{16}\mathrm{\;K},{32}\mathrm{\;K},{64}\mathrm{\;K},{128}\mathrm{\;K}$ and ${256}\mathrm{\;K}$ points (stored as rectangles) in a grid of $8\mathrm{\;K}$ by $8\mathrm{\;K}$ . The points were unique and randomly generated using a different seed value for each data set. We then generated 100 equally spaced query points in the $8\mathrm{\;K}$ by 8K space.

加利福尼亚州长滩市和马里兰州蒙哥马利县的 TIGER 数据文件，以及美国国家航空航天局（NASA）国际紫外探测器（I.U.E）卫星的观测记录。我们的示例将使用长滩的数据，该数据包含 55000 个街道段，存储为经纬度坐标对。对于合成数据实验，我们在 $8\mathrm{\;K}$×$8\mathrm{\;K}$ 的网格中生成了包含 $1\mathrm{\;K},2\mathrm{\;K},4\mathrm{\;K},8\mathrm{\;K},{16}\mathrm{\;K},{32}\mathrm{\;K},{64}\mathrm{\;K},{128}\mathrm{\;K}$ 和 ${256}\mathrm{\;K}$ 个点（存储为矩形）的测试数据文件。这些点是唯一的，并且每个数据集使用不同的种子值随机生成。然后，我们在 $8\mathrm{\;K}$×8K 的空间中生成了 100 个等间距的查询点。

We built the R-tree indexes by first presorting the data files using a Hilbert [Jaga90] number generating function, and then applying a modified version of [Rous85] R-tree packing technique according to the suggestion of [Kame94]. The branching factor of both the terminal and non-terminal nodes was set to be approximately 50 in all indexes.

我们通过以下方式构建 R 树索引：首先使用希尔伯特 [Jaga90] 数生成函数对数据文件进行预排序，然后根据 [Kame94] 的建议应用 [Rous85] R 树填充技术的改进版本。在所有索引中，终端节点和非终端节点的分支因子均设置为约 50。

In Figure 9 we see the average of 100 queries for each of several different numbers of nearest neighbors for both the MINDIST and MINMAXDIST ordering metrics applied to the Long Beach data. We generated three uniform sets of querys of 100 points each based on regions of the Long Beach, CA data set. The first set was from a sparse (few or no segments at all) region of the city, the second was from a dense (large number of segments per unit area), and the third was a uniform sample from the MBR of the whole city map. We then executed a series of nearest neighbor queries for each of the query points in each of these regions and plotted the average number of nodes accessed against the number of nearest neighbors.

在图 9 中，我们展示了将 MINDIST 和 MINMAXDIST 排序指标应用于长滩数据时，针对不同数量的最近邻进行 100 次查询的平均值。我们基于加利福尼亚州长滩数据集的不同区域生成了三组均匀的查询集，每组包含 100 个点。第一组来自城市的稀疏区域（几乎没有街道段），第二组来自密集区域（单位面积内街道段数量众多），第三组是整个城市地图最小边界矩形（MBR）内的均匀样本。然后，我们针对每个区域中的每个查询点执行了一系列最近邻查询，并绘制了访问的节点平均数量与最近邻数量的关系图。

<!-- Media -->

<!-- figureText: Long Beach, CA Map MINMAX Dense MINMAX Map MINMAX Sparse MNDIST Map MINDIST Sparse MINDIST DENSE No. of Neighbors 80.00 100.00 120.00 Pages Accessed 20.00 18.00 16.00 14.00 12.00 10.00 8.00 0.00 20.00 60.00 -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_6.jpg?x=939&y=1304&w=705&h=353&r=0"/>

Figure 9: MINDIST and MINMAXDIST Metric Comparison

图 9：MINDIST 和 MINMAXDIST 指标比较

<!-- Media -->

For this experiment we see that graphs of the MIN-MAXDIST ordered searches were similar in shape to the graphs of the MINDIST ordered searches but the number of pages accessed was consistently, approximately 20% higher. MINMAXDIST performed the worst in dense regions of the various data sets which was not surprising. It turned out that in all the experiments we performed comparing the two metrics, the results were similar to this one. Since this occurred with both real world and (pseudo) randomly distributed data, we surmise that for spatially sorted and packed R-trees, MINDIST is the ordering metric of choice. So, for the sake of clarity and simplicity, the rest of the figures will show the results of the MINDIST metric only.

在这个实验中，我们发现 MIN - MAXDIST 排序搜索的图形形状与 MINDIST 排序搜索的图形形状相似，但访问的页面数量始终大约高出 20%。MINMAXDIST 在各种数据集的密集区域表现最差，这并不令人意外。结果表明，在我们比较这两种指标的所有实验中，结果都与此类似。由于在真实世界数据和（伪）随机分布数据中都出现了这种情况，我们推测对于空间排序和填充的 R 树，MINDIST 是首选的排序指标。因此，为了清晰和简洁起见，其余的图将仅展示 MINDIST 指标的结果。

Figure 10 shows the results of an experiment using synthetic data. We ran and averaged the results from the 100 equally spaced query points for 25 different values of $k\mathrm{{NN}}$ (ranging from 1 to 121) on the $1\mathrm{\;K},4\mathrm{\;K}$ , ${16}\mathrm{\;K},{64}\mathrm{\;K}$ and ${256}\mathrm{\;K}$ data sets. We graphed the results as the (average) number of pages (nodes) accessed versus the number of nearest neighbors searched for.

图 10 展示了使用合成数据进行实验的结果。我们对 $1\mathrm{\;K},4\mathrm{\;K}$、${16}\mathrm{\;K},{64}\mathrm{\;K}$ 和 ${256}\mathrm{\;K}$ 数据集上 25 个不同的 $k\mathrm{{NN}}$ 值（范围从 1 到 121），对 100 个等间距的查询点进行了运行并取平均结果。我们将结果绘制成访问的页面（节点）平均数量与搜索的最近邻数量的关系图。

<!-- Media -->

<!-- figureText: Nearest Neighbor Scalability - Synthetic Data 256K-Points 64K-Points 16K-Points 4. KPoints T. R. Foints No. of Neighbors 80.00 100.00 120.00 Pages Accessed 16.00 14.00 12.00 10.00 8.00 6.00 0.00 20.00 40.00 60.00 -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_7.jpg?x=173&y=626&w=713&h=354&r=0"/>

Figure 10: Synthetic Data Experiment 1 Results

图 10：合成数据实验 1 结果

<!-- Media -->

From the experimental behavior, we can make two observations. First, as the number of nearest neighbors increased the number of pages accessed grew in a linear ratio with a (small) fractional constant of proportionality. Since all the synthetic data sets were created with the same node cardinality (approximately 50), the degree of similarity of the curves strongly suggests that this is the dominant term in the components of the constant of proportionality (at least in spatially ordered data sets). Second, as the data set size grew, the average number of page accesses grew sublinearly and clustered in three distinct groupings (producing a banded pattern) as the number of neighbors increased. The fact that the $4\mathrm{\;K},{16}\mathrm{\;K}$ and ${64}\mathrm{\;K}$ curves were so close to each other gave us the insight to run the next experiment.

从实验结果中，我们可以得出两点观察结论。首先，随着最近邻数量的增加，访问的页面数量呈线性增长，比例常数为（较小的）分数。由于所有合成数据集的节点基数相同（约为 50），曲线的相似程度强烈表明，这是比例常数组成部分中的主导项（至少在空间有序的数据集中是这样）。其次，随着数据集规模的增大，平均页面访问数量呈亚线性增长，并且随着邻居数量的增加，形成了三个不同的聚类（呈现出带状模式）。$4\mathrm{\;K},{16}\mathrm{\;K}$ 和 ${64}\mathrm{\;K}$ 曲线非常接近这一事实，促使我们进行下一个实验。

In the experiment of Figure 11, we examined the correlation between the increase in the size of the data set with the number of nodes accessed for a limited number of nearest neighbor queries. We plotted the number of pages accessed against the logarithm (base 2) of the data set size in terms of kilobytes (so $\left. {{\log }_{2}\left( {256K}\right)  = 8}\right)$ for each of 1,16,32,64 and 128 nearest neighbor queries. We noticed in this graph the curves appeared to be piecewise linear step functions. We then examined the height of the R-trees and observed that the steps appear at the points where the height of the tree increases. The 1K R-tree index has a depth of 1 , the $2\mathrm{\;K}$ index has a depth of 2,the $4\mathrm{\;K},8\mathrm{\;K},{16}\mathrm{\;K},{32}\mathrm{\;K}$ and ${64}\mathrm{\;K}$ data sets have a depth of 3,and the ${128}\mathrm{\;K}$ and the ${256}\mathrm{\;K}$ data sets have a depth of 4 .

在图11的实验中，我们研究了数据集大小的增加与有限数量的最近邻查询所访问的节点数量之间的相关性。我们绘制了所访问的页面数量与以千字节为单位的数据集大小的对数（以2为底）的关系图（因此对于1、16、32、64和128个最近邻查询中的每一个都有$\left. {{\log }_{2}\left( {256K}\right)  = 8}\right)$）。我们在这个图中注意到，这些曲线似乎是分段线性阶跃函数。然后，我们检查了R树的高度，并观察到阶跃出现在树的高度增加的点上。1K的R树索引深度为1，$2\mathrm{\;K}$索引深度为2，$4\mathrm{\;K},8\mathrm{\;K},{16}\mathrm{\;K},{32}\mathrm{\;K}$和${64}\mathrm{\;K}$数据集的深度为3，${128}\mathrm{\;K}$和${256}\mathrm{\;K}$数据集的深度为4。

<!-- Media -->

<!-- figureText: Database Size Scalability - Synthetic Data 128-Neighbors 64-Neighbors 32-Neighbons 16-Neighbors I-Neighbor $\log \left( 2\right)$ (Size in KB) Pages Accessed 20.00 18.00 16.00 12.00 10.00 8.00 -->

<img src="https://cdn.noedgeai.com/0195c903-e6c4-7687-a96c-6bf6064926e5_7.jpg?x=941&y=107&w=719&h=364&r=0"/>

Figure 11: Synthetic Data Experiment 2 Results

图11：合成数据实验2结果

<!-- Media -->

This is an important observation because it shows that the algorithm behaves well for ordered data sets and the cost of NN increases linearly with the height of the tree.

这是一个重要的观察结果，因为它表明该算法对于有序数据集表现良好，并且最近邻查询（NN）的成本随树的高度线性增加。

## 5 CONCLUSIONS

## 5 结论

In this paper, we developed a branch-and-bound R-tree traversal algorithm which finds the $\mathrm{{kN}}$ earest Neighbors of a given query point. We also introduced two metrics that can be used to guide an ordered depth first spatial search. The first metric, MINDIST, produces the most optimistic ordering possible, whereas the second, MINMAXDIST, produces the most pessimistic ordering that ever need be considered. Although our experiments have shown that MINDIST ordering was more effective in the case where the data were spatially sorted, other orderings or more sophisticated metrics using a combination of them are possible and might prove useful in the case where the R-tree was either not constructed as well or subject to (many) updates. Nonetheless, these two metrics were shown to be valuable tools in effectively directing and pruning the Nearest Neighbor search.

在本文中，我们开发了一种分支限界R树遍历算法，用于查找给定查询点的$\mathrm{{kN}}$个最近邻。我们还引入了两个可用于指导有序深度优先空间搜索的度量指标。第一个度量指标MINDIST（最小距离）产生尽可能最乐观的排序，而第二个度量指标MINMAXDIST（最小最大距离）产生需要考虑的最悲观的排序。尽管我们的实验表明，在数据进行了空间排序的情况下，MINDIST排序更有效，但其他排序或使用它们组合的更复杂的度量指标也是可行的，并且在R树构建不佳或需要进行（多次）更新的情况下可能会证明是有用的。尽管如此，这两个度量指标被证明是有效指导和剪枝最近邻搜索的有价值工具。

We implemented and thoroughly tested our k-NN algorithm. The experiments on both real and synthetic data sets showed that the algorithm scales up well with respect to both the number of NN requested and with size of the data sets. Further research on NN queries in our group will focus on defining and analyzing other metrics and how to characterize the behavior of our algorithm in dynamic as well as static database environments.

我们实现并彻底测试了我们的k近邻（k - NN）算法。对真实和合成数据集的实验表明，该算法在请求的最近邻数量和数据集大小方面都具有良好的扩展性。我们团队关于最近邻查询的进一步研究将集中在定义和分析其他度量指标，以及如何描述我们的算法在动态和静态数据库环境中的行为。

## 6 ACKNOWLEDGEMENTS

## 6 致谢

We would like to thank Christos Faloutsos for his insightful comments and suggestions.

我们要感谢克里斯托斯·法劳索斯（Christos Faloutsos）提出的深刻见解和建议。

## References

## 参考文献

[Beck90] Beckmann, N., H.-P. Kriegel, R. Schneider and B. Seeger, "The R*-tree: an efficient and robust access method for points and rectangles," ACM SIGMOD, pp 322-331, May 1990.

[BKS93] Brinkhoff, T., Kriegel, H.P., and Seeger, B., "Efficient Processing of Spatial Joins Using R-trees," Proc. ACM SIGMOD, May 1993, pp. 237- 246.

[FBF77] Friedman, J.H., Bentley, J.L., and Finkel, R.A., "An algorithm for finding the best matches in logarithmic expected time," ACM Trans. Math. Software, 3, September 1977, pp. 209-226.

[Gutt84] Guttman, A., "R-trees,: A Dynamic Index Structure for Spatial Searching," Proc. ACM SIG-MOD, pp. 47-57, June 1984.

[HS78] Horowitz, E., Sahni, S., "Fundamentals of Computer Algorithms," Computer Science Press, 1978, pp. 370-421.

[Jaga90] Jagadish, H.V., "Linear Clustering of Objects with Multiple Attributes," Proc. ACM SIGMOD, May 1990, pp. 332-342.

[Kame94] Kamel, I. and Faloutsos, C., "Hilbert R-Tree: an Improved R-Tree Using Fractals," Proc. of VLDB, 1994, pp. 500-509.

[Rous85] Roussopoulos, N. and D. Leifker, "Direct Spatial Search on Pictorial Databases Using Packed R-trees," Proc. ACM SIGMOD, May 1985.

[Same89] Samet, H., "The Design & Analysis Of Spatial Data Structures," Addison-Wesley, 1989.

[Same90] Samet, H., "Applications Of Spatial Data Structures, Computer Graphics, Image Processing and GIS," Addison-Wesley, 1990.

[SRF87] Sellis T., Roussopoulos, N., and Faloutsos, C., "The R+-tree: A Dynamic Index for Multidimensional Objects," Proc. 13th International Conference on Very Large Data Bases, 1987, pp. 507-518.

[Spro91] Sproull, R.F., "Refinements to Nearest-Neighbor Searching in k-Dimensional Trees," Al-gorithmica, 6, 1987, pp. 579-589.