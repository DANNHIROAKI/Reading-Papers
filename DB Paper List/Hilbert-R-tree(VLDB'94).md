# Hilbert R-tree: An Improved R-tree Using Fractals

# 希尔伯特R树（Hilbert R-tree）：一种使用分形的改进型R树

Ibrahim Kamel

易卜拉欣·卡迈勒（Ibrahim Kamel）

Department of Computer Science

计算机科学系

University of Maryland

马里兰大学

College Park, MD 20742

马里兰州大学公园市，邮编20742

kamel@cs.umd.edu

Christos Faloutsos ${}^{ * }$

克里斯托斯·法劳索斯（Christos Faloutsos） ${}^{ * }$

Department of Computer Science and

计算机科学系与

Institute for Systems Research (ISR)

系统研究学院（ISR）

University of Maryland

马里兰大学

College Park, MD 20742

马里兰州大学公园市，邮编20742

christos@cs.umd.edu

## Abstract

## 摘要

We propose a new R-tree structure that outperforms all the older ones. The heart of the idea is to facilitate the deferred splitting approach in R-trees. This is done by proposing an ordering on the R-tree nodes. This ordering has to be 'good', in the sense that it should group 'similar' data rectangles together, to minimize the area and perimeter of the resulting minimum bounding rectangles (MBRs).

我们提出了一种新的R树结构，其性能优于所有旧的R树结构。该想法的核心是促进R树中的延迟分裂方法。这是通过对R树节点进行排序来实现的。这种排序必须是“良好的”，即它应该将“相似的”数据矩形分组在一起，以最小化所得最小边界矩形（MBR）的面积和周长。

Following [KF93] we have chosen the so-called '2D-c' method, which sorts rectangles according to the Hilbert value of the center of the rectangles. Given the ordering, every node has a well-defined set of sibling nodes; thus, we can use deferred splitting. By adjusting the split policy, the Hilbert R-tree can achieve as high utilization as desired. To the contrary, the ${R}^{ * }$ -tree has no control over the space utilization, typically achieving up to ${70}\%$ . We designed the manipulation algorithms in detail, and we did a full implementation of the

遵循[KF93]，我们选择了所谓的“二维c”方法，该方法根据矩形中心的希尔伯特值对矩形进行排序。给定排序后，每个节点都有一组明确的兄弟节点；因此，我们可以使用延迟分裂。通过调整分裂策略，希尔伯特R树可以实现所需的高利用率。相反，${R}^{ * }$树无法控制空间利用率，通常最多只能达到${70}\%$。我们详细设计了操作算法，并对

Hilbert R-tree. Our experiments show that the '2-to-3' split policy provides a compromise between the insertion complexity and the search cost,giving up to ${28}\%$ savings over the ${R}^{ * }$ - tree [BKSS90] on real data.

希尔伯特R树进行了全面实现。我们的实验表明，“2到3”分裂策略在插入复杂度和搜索成本之间提供了折衷，在真实数据上比${R}^{ * }$树[BKSS90]节省了多达${28}\%$。

## 1 Introduction

## 1 引言

One of the requirements for the database management systems (DBMSs) of the near future is the ability to handle spatial data [SSU91]. Spatial data arise in many applications, including: Cartography [Whi81]; Computer-Aided Design (CAD) [OHM+84] [Gut84a]; computer vision and robotics [BB82]; traditional databases,where a record with $k$ attributes corresponds to a point in a $k$ -d space; temporal databases, where time can be considered as one more dimension [KS91]; scientific databases with spatial-temporal data, such as the ones in the 'Grand Challenge' applications [Gra92], etc.

未来数据库管理系统（DBMS）的要求之一是能够处理空间数据[SSU91]。空间数据出现在许多应用中，包括：制图学[Whi81]；计算机辅助设计（CAD）[OHM+84][Gut84a]；计算机视觉和机器人技术[BB82]；传统数据库，其中具有$k$个属性的记录对应于$k$维空间中的一个点；时态数据库，其中时间可以被视为另一个维度[KS91]；具有时空数据的科学数据库，例如“重大挑战”应用中的数据库[Gra92]等。

In the above applications, one of the most typical queries is the range query: Given a rectangle, retrieve all the elements that intersect it. A special case of the range query is the point query or stabbing query, where the query rectangle degenerates to a point.

在上述应用中，最典型的查询之一是范围查询：给定一个矩形，检索所有与之相交的元素。范围查询的一种特殊情况是点查询或刺探查询，此时查询矩形退化为一个点。

We focus on the R-tree [Gut84b] family of methods, which contains some of the most efficient methods that support range queries. The advantage of our method (and the rest of the R-tree-based methods) over the methods that use linear quad-trees and z-ordering is that R-trees treat the data objects as a whole, while quad-tree based methods typically divide objects into quad-tree blocks, increasing the number of items to be stored.

我们专注于R树 [Gut84b] 系列的方法，其中包含一些支持范围查询的最有效方法。与使用线性四叉树和z序的方法相比，我们的方法（以及其他基于R树的方法）的优势在于，R树将数据对象视为一个整体，而基于四叉树的方法通常会将对象划分为四叉树块，从而增加了需要存储的项的数量。

The most successful variant of R-trees seems to be the ${R}^{ * }$ -tree [BKSS90]. One of its main contributions is the idea of 'forced-reinsert' by deleting some rectangles from the overflowing node, and reinserting them.

R树最成功的变体似乎是${R}^{ * }$树 [BKSS90]。它的主要贡献之一是“强制重新插入”的思想，即从溢出节点中删除一些矩形，然后重新插入它们。

---

<!-- Footnote -->

*This research was partially funded by the Institute for Systems Research (ISR), by the National Science Foundation under Grants IRI-9205273 and IRI-8958546 (PYI), with matching funds from EMPRESS Software Inc. and Thinking Machines Inc.

*本研究部分由系统研究学院（ISR）资助，由美国国家科学基金会根据资助编号IRI - 9205273和IRI - 8958546（PYI）提供资金，同时得到了EMPRESS软件公司和思维机器公司的配套资金支持。

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the VLDB copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Very Large Data Base Endowment. To copy otherwise, or to republish, requires a fee and/or special permission from the Endowment.

允许免费复制本材料的全部或部分内容，前提是复制的目的不是为了直接的商业利益，要显示VLDB版权声明、出版物的标题及其日期，并注明复制是经超大型数据库基金会许可的。否则，若要进行复制或重新发布，则需要向基金会支付费用和/或获得特别许可。

Proceedings of the 20th VLDB Conference

第20届VLDB会议论文集

Santiago, Chile, 1994

智利，圣地亚哥，1994年

<!-- Footnote -->

---

The main idea in the present paper is to impose an ordering on the data rectangles. The consequences are important: using this ordering, each R-tree node has a well defined set of siblings; thus, we can use the algorithms for deferred splitting. By adjusting the split policy (2-to-3 or 3-to-4 etc) we can drive the utilization as close to ${100}\%$ as desirable. Notice that the ${R}^{ * }$ -tree does not have control over the utilization, typically achieving an average of $\approx  {70}\%$ .

本文的主要思想是对数据矩形施加一种排序。其影响很重要：利用这种排序，每个R树节点都有一组明确的兄弟节点；因此，我们可以使用延迟分裂算法。通过调整分裂策略（2对3或3对4等），我们可以使利用率尽可能接近${100}\%$。请注意，${R}^{ * }$树无法控制利用率，通常平均利用率为$\approx  {70}\%$。

The only requirement for the ordering is that it has to be 'good', that is, it should lead to small R-tree nodes.

这种排序的唯一要求是它必须是“良好的”，即它应该能使R树节点较小。

The paper is organized as follows. Section 2 gives a brief description of the R-tree and its variants. Section 3 describes the Hilbert R-tree. Section 4 presents our experimental results that compare the Hilbert R-tree with other R-tree variants. Section 5 gives the conclusions and directions for future research.

本文的组织如下。第2节简要介绍R树及其变体。第3节描述希尔伯特R树。第4节展示我们将希尔伯特R树与其他R树变体进行比较的实验结果。第5节给出结论和未来研究的方向。

## 2 Survey

## 2 综述

Several spatial access methods have been proposed. A recent survey can be found in [Sam89]. These methods fall in the following broad classes: methods that transform rectangles into points in a higher dimensionality space [HN83, Fre87]; methods that use linear quadtrees [Gar82] [AS91] or,equivalently,the $z$ - ordering [Ore86] or other space filling curves [FR89] [Jag90b]; and finally, methods based on trees (R-tree [Gut84b], k-d-trees [Ben75], k-d-B-trees [Rob81], hB-trees [LS90], cell-trees [Gun89] e.t.c.)

已经提出了几种空间访问方法。最近的综述可参见 [Sam89]。这些方法大致可分为以下几类：将矩形转换为更高维空间中的点的方法 [HN83, Fre87]；使用线性四叉树 [Gar82] [AS91] 或等效的$z$序 [Ore86] 或其他空间填充曲线 [FR89] [Jag90b] 的方法；最后，基于树的方法（R树 [Gut84b]、k - d树 [Ben75]、k - d - B树 [Rob81]、hB树 [LS90]、单元树 [Gun89] 等）

One of the most promising approaches in the last class is the R-tree [Gut84b]: Compared to the transformation methods, R-trees work on the native space, which has lower dimensionality; compared to the linear quadtrees, the R-trees do not need to divide the spatial objects into (several) pieces (quadtree blocks). The R-tree is an extension of the B-tree for multidimensional objects. A geometric object is represented by its minimum bounding rectangle (MBR): Non-leaf nodes contain entries of the form(R,ptr)where ${ptr}$ is a pointer to a child node in the R-tree; $\mathrm{R}$ is the MBR that covers all rectangles in the child node. Leaf nodes contain entries of the form (obj-id, R) where obj-id is a pointer to the object description,and $\mathrm{R}$ is the MBR of the object. The main innovation in the R-tree is that father nodes are allowed to overlap. This way, the R-tree can guarantee at least ${50}\%$ space utilization and remain balanced.

最后一类中最有前景的方法之一是R树 [Gut84b]：与转换方法相比，R树在较低维的原生空间中工作；与线性四叉树相比，R树不需要将空间对象划分为（多个）块（四叉树块）。R树是B树对多维对象的扩展。一个几何对象由其最小边界矩形（MBR）表示：非叶节点包含形式为(R,ptr)的条目，其中${ptr}$是指向R树中一个子节点的指针；$\mathrm{R}$是覆盖子节点中所有矩形的MBR。叶节点包含形式为(obj - id, R)的条目，其中obj - id是指向对象描述的指针，$\mathrm{R}$是对象的MBR。R树的主要创新之处在于允许父节点重叠。这样，R树可以保证至少${50}\%$的空间利用率并保持平衡。

Guttman proposed three splitting algorithms, the linear split, the quadratic split and the exponential split. Their names come from their complexity; among the three, the quadratic split algorithm is the one that achieves the best trade-off between splitting time and search performance.

古特曼提出了三种分裂算法，即线性分裂、二次分裂和指数分裂。它们的名称来源于其复杂度；在这三种算法中，二次分裂算法在分裂时间和搜索性能之间实现了最佳平衡。

Subsequent work on R-trees includes the work by Greene [Gre89],Roussopoulos and Leifker [RL85], ${R}^{ + }$ - tree by Sellis et al. [SRF87], R-trees using Minimum Bounding Ploygons [Jag90a], Kamel and Falout-sos [KF93] and the ${R}^{ * }$ -tree [BKSS90] of Beckmann et al. , which seems to have better performance than Guttman R-tree "quadratic split". The main idea in the ${R}^{ * }$ -tree is the concept of forced re-insert. When a node overflows, some of its children are carefully chosen; they are deleted and re-inserted, usually resulting in a R-tree with better structure.

后续关于R树的研究包括格林（Greene）[Gre89]、鲁索普洛斯（Roussopoulos）和利夫克（Leifker）[RL85]的工作，塞利斯（Sellis）等人提出的${R}^{ + }$树[SRF87]，使用最小边界多边形的R树[Jag90a]，卡迈勒（Kamel）和法洛索斯（Faloutsos）[KF93]的研究，以及贝克曼（Beckmann）等人提出的${R}^{ * }$树[BKSS90]，它的性能似乎比古特曼R树的“二次分裂”更好。${R}^{ * }$树的主要思想是强制重新插入的概念。当一个节点溢出时，会仔细选择它的一些子节点；将它们删除并重新插入，通常会得到一个结构更好的R树。

## 3 Hilbert R-trees

## 3 希尔伯特R树

In this section we introduce the Hilbert R-tree and discuss algorithms for searching, insertion, deletion, and overflow handling. The performance of the R-trees depends on how good is the algorithm that cluster the data rectangles to a node. We propose to use space filling curves (or fractals), and specifically, the Hilbert curve to impose a linear ordering on the data rectangles.

在本节中，我们将介绍希尔伯特R树，并讨论搜索、插入、删除和处理溢出的算法。R树的性能取决于将数据矩形聚类到一个节点的算法的优劣。我们建议使用空间填充曲线（或分形），特别是希尔伯特曲线，对数据矩形施加线性排序。

A space filling curve visits all the points in a $k$ - dimensional grid exactly once and never crosses itself. The Z-order (or Morton key order, or bit-interleaving, or Peano curve), the Hilbert curve, and the Gray-code curve [Fal88] are examples of space filling curves. In [FR89], it was shown experimentally that the Hilbert curve achieves the best clustering among the three above methods.

空间填充曲线恰好访问$k$维网格中的所有点一次，并且从不自相交。Z序（或莫顿键序、或位交织、或皮亚诺曲线）、希尔伯特曲线和格雷码曲线[Fal88]都是空间填充曲线的例子。在[FR89]中，实验表明希尔伯特曲线在上述三种方法中实现了最佳的聚类效果。

Next we provide a brief introduction to the Hilbert curve: The basic Hilbert curve on a $2 \times  2$ grid,denoted by ${H}_{1}$ ,is shown in Figure 1. To derive a curve of order $i$ ,each vertex of the basic curve is replaced by the curve of order $i - 1$ ,which may be appropriately rotated and/or reflected. Figure 1 also shows the Hilbert curves of order 2 and 3 . When the order of the curve tends to infinity, the resulting curve is a fractal, with a fractal dimension of 2 [Man77]. The Hilbert curve can be generalized for higher dimensionalities. Algorithms to draw the two-dimensional curve of a given order, can be found in [Gri86], [Jag90b]. An algorithm for higher dimensionalities is in [Bia69].

接下来，我们简要介绍一下希尔伯特曲线：$2 \times  2$网格上的基本希尔伯特曲线，用${H}_{1}$表示，如图1所示。为了推导出$i$阶曲线，基本曲线的每个顶点都被$i - 1$阶曲线替换，该曲线可能会进行适当的旋转和/或反射。图1还展示了2阶和3阶的希尔伯特曲线。当曲线的阶数趋于无穷大时，得到的曲线是一个分形，分形维数为2[Man77]。希尔伯特曲线可以推广到更高维度。绘制给定阶数的二维曲线的算法可以在[Gri86]、[Jag90b]中找到。用于更高维度的算法在[Bia69]中。

The path of a space filling curve imposes a linear ordering on the grid points. Figure 1 shows one such ordering for a $4 \times  4$ grid (see curve ${H}_{2}$ ). For example the point(0,0)on the ${H}_{2}$ curve has a Hilbert value of0,while the point(1,1)has a Hilbert value of 2 . The Hilbert value of a rectangle needs to be defined. Following the experiments in [KF93], a good choice is the following:

空间填充曲线的路径对网格点施加了线性排序。图1展示了$4 \times  4$网格的一种这样的排序（见曲线${H}_{2}$）。例如，${H}_{2}$曲线上的点(0,0)的希尔伯特值为0，而点(1,1)的希尔伯特值为2。需要定义矩形的希尔伯特值。根据[KF93]中的实验，一个不错的选择如下：

Definition 1 : The Hilbert value of a rectangle is defined as the Hilbert value of its center.

定义1：矩形的希尔伯特值定义为其中心的希尔伯特值。

<!-- Media -->

<!-- figureText: 13 11 12 15 ${\mathrm{H}}_{3}$ 14 ${\mathrm{H}}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_2.jpg?x=540&y=152&w=757&h=352&r=0"/>

Figure 1: Hilbert Curves of order 1, 2 and 3

图1：1阶、2阶和3阶的希尔伯特曲线

<!-- Media -->

After this preliminary material, we are in a position now to describe the proposed methods.

在介绍了这些预备知识之后，我们现在可以描述所提出的方法了。

### 3.1 Description

### 3.1 描述

The main idea is to create a tree structure that can

主要思想是创建一种树结构，它可以

- behave like an $\mathrm{R}$ -tree on search.

- 在搜索时表现得像$\mathrm{R}$树。

- support deferred splitting on insertion, using the Hilbert value of the inserted data rectangle as the primary key.

- 在插入时支持延迟分裂，使用插入的数据矩形的希尔伯特值作为主键。

These goals can be achieved as follows: for every node $n$ of our tree,we store (a) its MBR,and (b) the Largest Hilbert Value (LHV) of the data rectangles that belong to the subtree with root $n$ .

这些目标可以通过以下方式实现：对于我们树中的每个节点$n$，我们存储（a）它的最小边界矩形（MBR），以及（b）以$n$为根的子树中数据矩形的最大希尔伯特值（LHV）。

Specifically, the Hilbert R-tree has the following structure. A leaf node contains at most ${C}_{l}$ entries each of the form

具体来说，希尔伯特R树具有以下结构。一个叶节点最多包含${C}_{l}$个条目，每个条目的形式为

(R,objid)

(R,对象ID)

where ${C}_{l}$ is the capacity of the leaf, $R$ is the MBR of the real object $\left( {{x}_{\text{low }},{x}_{\text{high }},{y}_{\text{low }},{y}_{\text{high }}}\right)$ and obj-id is a pointer to the object description record. The main difference with $\mathrm{R}$ - and ${\mathrm{R}}^{ * }$ -trees is that nonleaf nodes also contain information about the LHVs. Thus, a non-leaf node in the Hilbert R-tree contains at most ${C}_{n}$ entries of the form

其中 ${C}_{l}$ 是叶子节点的容量，$R$ 是真实对象 $\left( {{x}_{\text{low }},{x}_{\text{high }},{y}_{\text{low }},{y}_{\text{high }}}\right)$ 的最小边界矩形（MBR），obj - id 是指向对象描述记录的指针。与 $\mathrm{R}$ 树和 ${\mathrm{R}}^{ * }$ 树的主要区别在于，非叶子节点还包含关于最大希尔伯特值（LHV）的信息。因此，希尔伯特 R 树中的非叶子节点最多包含 ${C}_{n}$ 个以下形式的条目

$$
\left( {R,\text{ ptr,LHV }}\right) 
$$

where ${C}_{n}$ is the capacity of a non-leaf node, $R$ is the MBR that encloses all the children of that node, ptr is a pointer to the child node,and ${LHV}$ is the largest Hilbert value among the data rectangles enclosed by $R$ . Notice that we never calculate or use the Hilbert values of the MBRs. Figure 2 illustrates some rectangles, organized in a Hilbert R-tree. The Hilbert values of the centers are the numbers by the ’ $x$ ’ symbols (shown only for the parent node ’II’). The LHV’s are in [brackets]. Figure 3 shows how is the tree of Figure 2 stored on the disk; the contents of the parent node 'II' are shown in more detail. Every data rectangle in node ’I’ has Hilbert value $\leq  {33}$ ; everything in node ’II’ has Hilbert value greater than 33 and $\leq  {107}$ etc.

其中 ${C}_{n}$ 是非叶子节点的容量，$R$ 是包含该节点所有子节点的最小边界矩形（MBR），ptr 是指向子节点的指针，${LHV}$ 是 $R$ 所包含的数据矩形中的最大希尔伯特值。请注意，我们从不计算或使用最小边界矩形（MBR）的希尔伯特值。图 2 展示了一些组织在希尔伯特 R 树中的矩形。中心的希尔伯特值是 “ $x$ ” 符号旁边的数字（仅为父节点 “II” 显示）。最大希尔伯特值（LHV）用 [方括号] 表示。图 3 展示了图 2 中的树是如何存储在磁盘上的；父节点 “II” 的内容展示得更详细。节点 “I” 中的每个数据矩形的希尔伯特值为 $\leq  {33}$ ；节点 “II” 中的所有内容的希尔伯特值都大于 33 且 $\leq  {107}$ 等等。

Before we continue, we list some definitions. A plain $R$ -tree splits a node on overflow,turning 1 node to 2 . We call this policy a 1-to-2 splitting policy. We propose to defer the split, waiting until they turn 2 nodes into 3 . We refer to it as the 2-to-3 splitting policy. In general, we can have an s-to-(s+1) splitting policy; we refer to $s$ as the order of the splitting policy. To implement the order- $s$ splitting policy,the overflowing node tries to push some of its entries to one of its $s - 1$ siblings; if all of them are full,then we have an s-to- $\left( {s + 1}\right)$ split. We refer to these $s - 1$ siblings as the cooperating siblings of a given node.

在继续之前，我们列出一些定义。普通的 $R$ 树在节点溢出时进行分裂，将 1 个节点变为 2 个。我们将这种策略称为 1 对 2 分裂策略。我们提议推迟分裂，直到 2 个节点变为 3 个。我们将其称为 2 对 3 分裂策略。一般来说，我们可以有 s 对 (s + 1) 分裂策略；我们将 $s$ 称为分裂策略的阶。为了实现阶为 $s$ 的分裂策略，溢出的节点会尝试将其一些条目推送到它的 $s - 1$ 个兄弟节点之一；如果所有兄弟节点都已满，那么我们进行 s 对 $\left( {s + 1}\right)$ 分裂。我们将这些 $s - 1$ 个兄弟节点称为给定节点的协作兄弟节点。

Next, we will describe in detail the algorithms for searching, insertion, and overflow handling.

接下来，我们将详细描述搜索、插入和处理溢出的算法。

### 3.2 Searching

### 3.2 搜索

The searching algorithm is similar to the one used in other R-tree variants. Starting from the root it descends the tree examining all nodes that intersect the query rectangle. At the leaf level it reports all entries that intersect the query window $w$ as qualified data items.

搜索算法与其他 R 树变体中使用的算法类似。从根节点开始，它遍历树，检查所有与查询矩形相交的节点。在叶子节点级别，它将所有与查询窗口 $w$ 相交的条目报告为合格的数据项。

Algorithm Search(node Root,rect $w$ ):

算法 Search（节点 Root，矩形 $w$ ）：

S1. Search nonleaf nodes:

S1. 搜索非叶子节点：

invoke Search for every entry whose MBR intersects the query window $w$ .

对每个最小边界矩形（MBR）与查询窗口 $w$ 相交的条目调用 Search。

S2. Search leaf nodes:

S2. 搜索叶子节点：

Report all the entries that intersect the query window $w$ as candidate.

将所有与查询窗口 $w$ 相交的条目报告为候选条目。

### 3.3 Insertion

### 3.3 插入

To insert a new rectangle $r$ in the Hilbert R-tree,the Hilbert value $h$ of the center of the new rectangle is used as a key. In each level we choose the node with minimum LHV among the siblings. When a leaf node is reached the rectangle $r$ is inserted in its correct order according to $h$ . After a new rectangle is inserted in a leaf node $N$ ,Adjust Tree is called to fix the MBR and LHV values in upper level nodes.

要在希尔伯特 R 树中插入一个新的矩形 $r$ ，新矩形中心的希尔伯特值 $h$ 被用作键。在每一层，我们选择兄弟节点中最大希尔伯特值（LHV）最小的节点。当到达叶子节点时，矩形 $r$ 根据 $h$ 按正确顺序插入。在叶子节点 $N$ 中插入一个新矩形后，调用调整树（Adjust Tree）来修正上层节点中的最小边界矩形（MBR）和最大希尔伯特值（LHV）。

<!-- Media -->

<!-- figureText: (30,75) II (61.75) (68,75) (56)50) III (80,40) [107] [206] ${}^{x}$ [98] [92] (35,65) (60) [107] (36,40) (20,38) (45,35) [33] (50,10) (3,5 (0,0) -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_3.jpg?x=381&y=155&w=1024&h=762&r=0"/>

Figure 2: Data rectangles organized in a Hilbert R-tree

图 2：组织在希尔伯特 R 树中的数据矩形

<!-- figureText: LHV XL YL XH YH LHY XL YL XH YH LHV XL YL XH 55 75 206 50 10 LHV</i07 四 $\angle {HV} <  = {206}^{ \circ  }$ XH WH XL XH YH XL YL XH YH YL 81 78 33 8 35 40 107 20 38 ${LHV} <  = {33}$ H YH XL YL XH XL XH YH XL 20 30 78 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_3.jpg?x=376&y=1030&w=1061&h=339&r=0"/>

Figure 3: The file structure for the previous Hilbert R-tree

图 3：前一个希尔伯特 R 树的文件结构

<!-- Media -->

Algorithm Insert(node Root,rect $r$ ):

算法 Insert（节点 Root，矩形 $r$ ）：

/* inserts a new rectangle $r$ in the Hilbert R-tree.

/* 在希尔伯特R树中插入一个新的矩形 $r$。

$h$ is the Hilbert value of the rectangle. */

$h$ 是该矩形的希尔伯特值。 */

I1. Find the appropriate leaf node:

I1. 找到合适的叶节点：

Invoke ChooseLeaf(r,h)to select a leaf node $L$ in which to place $r$ .

调用ChooseLeaf(r,h) 选择一个叶节点 $L$ 来放置 $r$。

12. Insert $r$ in a leaf node $L$ :

12. 在叶节点 $L$ 中插入 $r$：

if $L$ has an empty slot,insert $r$ in $L$ in the appropriate place according to the Hilbert order and return.

如果 $L$ 有空闲槽位，按照希尔伯特顺序将 $r$ 插入到 $L$ 的合适位置并返回。

if $L$ is full,invoke HandleOverflow(L,r),

如果 $L$ 已满，调用HandleOverflow(L,r)，

which will return new leaf if split was inevitable.

如果不可避免地发生分裂，它将返回新的叶节点。

I3. Propagate changes upward:

I3. 向上传播更改：

form a set $\mathcal{S}$ that contains $L$ ,its cooperating siblings and the new leaf (if any).

形成一个集合 $\mathcal{S}$，其中包含 $L$、其协作兄弟节点和新的叶节点（如果有）。

invoke AdjustTree( $\mathcal{S}$ )

调用AdjustTree( $\mathcal{S}$ )

I4. Grow tree taller:

I4. 增加树的高度：

if node split propagation caused the root to split, create a new root whose children are the two resulting nodes.

如果节点分裂传播导致根节点分裂，则创建一个新的根节点，其孩子节点为分裂产生的两个节点。

Algorithm ChooseLeaf(rect $r$ ,int $h$ ): /* Returns the leaf node in which to place a new rectangle $r. * /$

算法ChooseLeaf(矩形 $r$，整数 $h$ )：/* 返回用于放置新矩形 $r. * /$ 的叶节点

C1. Initialize:

C1. 初始化：

Set $N$ to be the root node. C2. Leaf check:

将 $N$ 设置为根节点。C2. 叶节点检查：

if $N$ is a leaf,return $N$ .

如果 $N$ 是叶子节点，则返回 $N$。

C3. Choose subtree:

C3. 选择子树：

if $N$ is a non-leaf node,choose the entry (R, ptr, LHV) with the minimum LHV value greater than $h$ .

如果 $N$ 是非叶子节点，则选择 LHV（最低希尔伯特值，Lowest Hilbert Value）值大于 $h$ 且最小的条目 (R, ptr, LHV)。

C4. Descend until a leaf is reached: set $N$ to the node pointed by ptr and repeat from C2.

C4. 向下遍历直到到达叶子节点：将 $N$ 设置为 ptr 所指向的节点，并从 C2 开始重复。

Algorithm AdjustTree(set $\mathcal{S}$ ):

算法 AdjustTree(设置 $\mathcal{S}$ )：

/* $\mathcal{S}$ is a set of nodes that contains the node being updated, its cooperating siblings (if overflow has occurred) and newly created node ${NN}$ (if split has occurred). The routine ascends from leaf level towards the root, adjusting MBR and LHV of nodes that coverthe nodes in $\mathcal{S}$ siblings. It propagates splits (if any). */

/* $\mathcal{S}$ 是一个节点集合，包含正在更新的节点、其协作兄弟节点（如果发生溢出）和新创建的节点 ${NN}$（如果发生分裂）。该例程从叶子层向根节点上升，调整覆盖 $\mathcal{S}$ 兄弟节点的节点的最小边界矩形（Minimum Bounding Rectangle，MBR）和 LHV。它会传播分裂（如果有的话）。 */

A1. if reached root level stop.

A1. 如果到达根节点层，则停止。

A2. Propagate node split upward let ${N}_{p}$ be the parent node of $N$ .

A2. 向上传播节点分裂 设 ${N}_{p}$ 为 $N$ 的父节点。

if $N$ has been split,let ${NN}$ be the new node. insert ${NN}$ in ${N}_{p}$ in the correct order according to its Hilbert value if there is room. Otherwise, invoke HandleOverflow $\left( {{N}_{p},\mathbf{{MBR}}\left( {NN}\right) }\right)$ . if ${N}_{p}$ is split,let ${PP}$ be the new node.

如果 $N$ 已分裂，设 ${NN}$ 为新节点。如果有空间，根据其希尔伯特值按正确顺序将 ${NN}$ 插入 ${N}_{p}$ 中。否则，调用 HandleOverflow $\left( {{N}_{p},\mathbf{{MBR}}\left( {NN}\right) }\right)$。如果 ${N}_{p}$ 分裂，设 ${PP}$ 为新节点。

A3. adjust the MBR's and LHV's in the parent level: let $\mathcal{P}$ be the set of parent nodes for the nodes in $\mathcal{S}$ .

A3. 调整父节点层的 MBR 和 LHV：设 $\mathcal{P}$ 为 $\mathcal{S}$ 中节点的父节点集合。

Adjust the corresponding MBR's and LHV's appropriately of the nodes in $\mathcal{P}$ .

适当地调整 $\mathcal{P}$ 中节点的相应 MBR 和 LHV。

A4. Move up to next level:

A4. 移动到下一层：

Let $\mathcal{S}$ become the set of parent nodes $\mathcal{P}$ ,with ${NN} = {PP}$ ,if ${N}_{p}$ was split. repeat from A1.

如果 ${N}_{p}$ 分裂，让 $\mathcal{S}$ 变为 $\mathcal{P}$ 的父节点集合，包含 ${NN} = {PP}$。从 A1 开始重复。

### 3.4 Deletion

### 3.4 删除操作

In Hilbert R-tree we do NOT need to re-insert orphaned nodes, whenever a father node underflows. Instead, we borrow keys from the siblings or we merge an underflowing node with its siblings. We are able to do so, because the nodes have a clear ordering (Largest Hilbert Value LHV); in contrast, in R-trees there is no such concept of sibling node. Notice that, for deletion, we need $s$ cooperating siblings while for insertion we need $s - 1$ .

在希尔伯特 R 树（Hilbert R-tree）中，每当父节点下溢时，我们不需要重新插入孤立节点。相反，我们从兄弟节点借用键，或者将下溢节点与其兄弟节点合并。我们能够这样做，是因为节点有明确的顺序（最大希尔伯特值，Largest Hilbert Value，LHV）；相比之下，在 R 树中没有兄弟节点的这种概念。请注意，对于删除操作，我们需要 $s$ 个协作兄弟节点，而对于插入操作，我们需要 $s - 1$ 个。

Algorithm Delete(r):

算法 Delete(r)：

D1. Find the host leaf:

D1. 查找宿主叶节点：

Perform an exact match search to find the leaf node $L$ that contain $r$ .

执行精确匹配搜索，以找到包含 $r$ 的叶节点 $L$。

D2. Delete $r$ :

D2. 删除 $r$ ：

Remove $r$ from node $L$ .

从节点 $L$ 中移除 $r$ 。

D3. if $L$ underflows

D3. 如果 $L$ 发生下溢

borrow some entries from $s$ cooperating siblings.

从 $s$ 的协作兄弟节点中借用一些条目。

if all the siblings are ready to underflow, merge $s + 1$ to $s$ nodes, adjust the resulting nodes.

如果所有兄弟节点都即将发生下溢，将 $s + 1$ 合并到 $s$ 节点，并调整结果节点。

D4. adjust MBR and LHV in parent levels: form a set $\mathcal{S}$ that contains $L$ and its cooperating siblings (if underflow has occurred).

D4. 调整父层级中的最小边界矩形（MBR）和最低希尔伯特值（LHV）：形成一个集合 $\mathcal{S}$ ，其中包含 $L$ 及其协作兄弟节点（如果发生了下溢）。

invoke AdjustTree( $\mathcal{S}$ ).

调用 AdjustTree( $\mathcal{S}$ )。

### 3.5 Overflow handling

### 3.5 溢出处理

The overflow handling algorithm in the Hilbert R-tree treats the overflowing nodes either by moving some of the entries to one of the $s - 1$ cooperating siblings or splitting $s$ nodes to $s + 1$ nodes.

希尔伯特R树（Hilbert R-tree）中的溢出处理算法通过将一些条目移动到 $s - 1$ 个协作兄弟节点之一，或者将 $s$ 个节点拆分为 $s + 1$ 个节点来处理溢出节点。

Algorithm HandleOverflow(node $N$ ,rect $r$ ): /* return the new node if a split occurred. */

算法 HandleOverflow(节点 $N$ ，矩形 $r$ )：/* 如果发生拆分，则返回新节点。 */

H1. let $\mathcal{E}$ be a set that contains all the entries from $N$ and its $s - 1$ cooperating siblings.

H1. 令 $\mathcal{E}$ 为一个集合，包含来自 $N$ 及其 $s - 1$ 个协作兄弟节点的所有条目。

H2. add $r$ to $\mathcal{E}$ .

H2. 将 $r$ 添加到 $\mathcal{E}$ 中。

H3. if at least one of the $s - 1$ cooperating siblings is not full,distribute $\mathcal{E}$ evenly among the $s$ nodes according to the Hilbert value.

H3. 如果 $s - 1$ 个协作兄弟节点中至少有一个未满，则根据希尔伯特值将 $\mathcal{E}$ 均匀分配到 $s$ 个节点中。

H4. if all the $s$ cooperating siblings are full, create a new node ${NN}$ and distribute $\mathcal{E}$ evenly among the $s + 1$ nodes according to the Hilbert value. return ${NN}$ .

H4. 如果 $s$ 个协作兄弟节点都已满，则创建一个新节点 ${NN}$ ，并根据希尔伯特值将 $\mathcal{E}$ 均匀分配到 $s + 1$ 个节点中。返回 ${NN}$ 。

## 4 Experimental results

## 4 实验结果

To assess the merit of our proposed Hilbert R-tree, we implemented it and ran experiments on a two dimensional space. The method was implemented in C, under UNIX. We compared our methods against the quadratic-split $\mathrm{R}$ -tree,and the ${R}^{ * }$ -tree. Since the CPU time required to process the node is negligible, we based our comparison on the number of nodes (=pages) retrieved by range queries.

为了评估我们提出的希尔伯特R树（Hilbert R-tree）的优点，我们实现了它并在二维空间中进行了实验。该方法在UNIX系统下用C语言实现。我们将我们的方法与二次拆分 $\mathrm{R}$ 树和 ${R}^{ * }$ 树进行了比较。由于处理节点所需的CPU时间可以忽略不计，我们的比较基于范围查询检索到的节点（=页面）数量。

Without loss of generality, the address space was normalized to the unit square. There are several factors that affect the search time; we studied the following ones:

不失一般性，将地址空间归一化为单位正方形。有几个因素会影响搜索时间；我们研究了以下因素：

Data items: points and/or rectangles and/or line segments (represented by their MBR)

数据项：点和/或矩形和/或线段（由其最小边界矩形（MBR）表示）

File size: ranged from 10,000 - 100,000 records

文件大小：范围从10,000 - 100,000条记录

Query area ${Q}_{\text{area }} = {q}_{x} \times  {q}_{y}$ : ranged from 0 - 0.3 of the area of the address space

查询区域 ${Q}_{\text{area }} = {q}_{x} \times  {q}_{y}$ ：范围从地址空间面积的0 - 0.3

Another important factor,which is derived from $N$ and the average area $a$ of the data rectangles,is the ’data density’ $d$ (or ’cover quotient’) of the data rectangles. This is the sum of the areas of the data rectangles in the unit square, or equivalently, the average number of rectangles that cover a randomly selected point. Mathematically: $d = N \times  a$ . For the selected values of $N$ and $a$ ,the data density ranges from 0.25 - 2.0 .

另一个重要因素是数据矩形的“数据密度” $d$ （或“覆盖商”），它由 $N$ 和数据矩形的平均面积 $a$ 推导得出。这是单位正方形中数据矩形面积之和，或者等效地，是覆盖随机选择点的矩形的平均数量。数学表达式为： $d = N \times  a$ 。对于所选的 $N$ 和 $a$ 值，数据密度范围从0.25 - 2.0。

To compare the performance of our proposed structures we used 5 data files that contained different types of data: points, rectangles, lines, or mixed. Specifically, we used:

为了比较我们提出的结构的性能，我们使用了5个包含不同类型数据的数据文件：点、矩形、线或混合数据。具体来说，我们使用了：

A) Real Data: we used real data from the TIGER system of the U.S. Bureau of Census. An important observation is that the data in the TIGER datasets follow a highly skewed distribution.

A) 真实数据：我们使用了美国人口普查局TIGER系统的真实数据。一个重要的观察结果是，TIGER数据集中的数据遵循高度偏斜的分布。

'MGCounty' : This file consists of 39717 line segments, representing the roads of Montgomery county in Maryland. Using the minimum bounding rectangles of the segments, we obtained 39717 rectangles, with data density $d = {0.35}$ . We refer to this dataset as the 'MGCounty' dataset.

“MGCounty”：该文件包含39717条线段，代表马里兰州蒙哥马利县的道路。使用这些线段的最小边界矩形，我们得到了39717个矩形，数据密度为 $d = {0.35}$ 。我们将这个数据集称为“MGCounty”数据集。

'LBeach' : It consists of 53145 line segments, representing the roads of Long Beach, California. The data density of the MBRs that cover these line segments is $d = {0.15}$ . We refer to this dataset as the ’ $L$ Beach’ dataset.

“LBeach”：它包含53145条线段，代表加利福尼亚州长滩的道路。覆盖这些线段的最小边界矩形的数据密度为 $d = {0.15}$ 。我们将这个数据集称为“ $L$ Beach”数据集。

B) Synthetic Data: The reason for using synthetic data is that we can control the parameters (data density, number of rectangles, ratio of points to rectangles etc.).

B) 合成数据：使用合成数据的原因是我们可以控制参数（数据密度、矩形数量、点与矩形的比例等）。

'Points' : This file contains 75,000 uniformly distributed points.

“Points”：该文件包含75,000个均匀分布的点。

'Rects' : This file contains 100,000 rectangles, no points. The centers of the rectangles are uniformly distributed in the unit square. The data density is $d = {1.0}$

“Rects”：该文件包含100,000个矩形，没有点。矩形的中心在单位正方形中均匀分布。数据密度为 $d = {1.0}$

'Mix' : This file contains a mix of points and rectangles; specifically 50,000 points and 10,000 rectangles; the data density is $d =$ 0.029 .

“Mix”：该文件包含点和矩形的混合数据；具体来说是50,000个点和10,000个矩形；数据密度为 $d =$ 0.029。

The query rectangles were squares with side ${q}_{s}$ ; their centers were uniformly distributed in the unit square. For each experiment, 200 randomly generated queries were asked and the results were averaged. The standard deviation was very small and is not even plotted in our graphs. The page size used is $1\mathrm{{KB}}$ .

查询矩形是边长为 ${q}_{s}$ 的正方形；它们的中心在单位正方形中均匀分布。对于每个实验，进行了200次随机生成的查询，并对结果进行了平均。标准差非常小，在我们的图表中甚至没有绘制出来。使用的页面大小为 $1\mathrm{{KB}}$ 。

We compare the Hilbert R-tree against the original R-tree ( quadratic split) and the ${R}^{ * }$ -tree. Next we present experiments that (a) compare our method against other R-tree variants (b) show the effect of the different split policies on the performance of the proposed method and (c) evaluate the insertion cost.

我们将希尔伯特R树与原始R树（二次分裂）和 ${R}^{ * }$ -树进行比较。接下来，我们展示的实验（a）将我们的方法与其他R树变体进行比较（b）展示不同分裂策略对所提出方法性能的影响（c）评估插入成本。

<!-- Media -->

<!-- figureText: 550.00 50k points and 10k rectangles; 2-to-3 split policy Elbert R-1999 Quest 10-3 150.00 200.00 250.00 \$00.00 450.00 400.00 350.00 250.00 200.00 150.00 100.00 50.00 0.00 0.00 50.00 100.00 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_5.jpg?x=900&y=395&w=658&h=673&r=0"/>

Figure 4: Points and Rectangles ('Mix' Dataset); Disk Accesses vs. Query Area

图4：点和矩形（“Mix”数据集）；磁盘访问次数与查询区域的关系

<!-- figureText: 100k rectangles; 2-to-3 split policy Hubertless Question 150.00 200.00 250.00 Pages Touched 850.00 800.00 750.00 700.00 650.00 S50.00 500.00 400.00 300.00 250.00 200.00 150.00 50,00 0.00 0.00 50.00 100.00 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_5.jpg?x=900&y=1224&w=652&h=674&r=0"/>

Figure 5: Rectangles Only ('Rects' dataset); Disk Accesses vs. Query Area

图5：仅含矩形（“矩形”数据集）；磁盘访问次数与查询区域的关系

<!-- figureText: 75k points; 2-to-3 split policy Hubert R-tree Resistant Question 3 150.00 200.00 250.00 600.00 550.00 500.00 450.00 400.00 350.00 300.00 250.00 200.00 150.00 100.00 50.00 0.00 50.00 100.00 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_6.jpg?x=226&y=153&w=650&h=668&r=0"/>

Figure 6: Points Only ('Points' dataset); Disk Accesses vs. Query Area

图6：仅含点（“点”数据集）；磁盘访问次数与查询区域的关系

<!-- Media -->

### 4.1 Comparison of the Hilbert R-tree vs. other R-tree variants

### 4.1 希尔伯特R树与其他R树变体的比较

In this section we show the performance superiority of our Hilbert R-tree over the ${R}^{ * }$ -tree,which is the most successful variant of the R-tree. We present experiments with all five datasets, namely: 'Mix', 'Rects', 'Points', 'MGCounty', and 'LBeach' (see Figures 4 - 6, respectively). In all these experiments, we used the '2-to-3' split policy for the Hilbert R-tree.

在本节中，我们展示了我们的希尔伯特R树相对于${R}^{ * }$树的性能优势，${R}^{ * }$树是R树最成功的变体。我们对所有五个数据集进行了实验，即：“混合”“矩形”“点”“蒙哥马利县（MGCounty）”和“长滩（LBeach）”（分别见图4 - 6）。在所有这些实验中，我们对希尔伯特R树采用了“2到3”的分裂策略。

In all the experiment the Hilbert R-tree is the clear winner,achieving up to ${28}\%$ savings in response time over the next best contender (the ${R}^{ * }$ -tree). This maximum gain is achieved for the 'MGCounty' dataset (Figure 7). It is interesting to notice that the performance gap is larger for the real data, whose main difference from the synthetic one is that it is skewed, as opposed to uniform. Thus, we can conjecture that the skeweness of the data favors the Hilbert R-tree.

在所有实验中，希尔伯特R树明显胜出，与次优竞争者（${R}^{ * }$树）相比，响应时间最多可节省${28}\%$。这种最大增益在“蒙哥马利县（MGCounty）”数据集上实现（图7）。有趣的是，对于真实数据，性能差距更大，真实数据与合成数据的主要区别在于它是倾斜的，而非均匀的。因此，我们可以推测，数据的倾斜性有利于希尔伯特R树。

Figure 4 also plots the results for the quadratic-split R-tree, which, as expected, is outperformed by the ${R}^{ * }$ -tree. In the rest of the figures,we omit the quadratic-split R-tree, because it was consistently outperformed by ${R}^{ * }$ -tree.

图4还绘制了二次分裂R树的结果，正如预期的那样，它的性能不如${R}^{ * }$树。在其余的图中，我们省略了二次分裂R树，因为它的性能始终不如${R}^{ * }$树。

### 4.2 The effect of the split policy on the per- formance

### 4.2 分裂策略对性能的影响

Figure 9 shows the response time as a function of the query size for the 1-to-2, 2-to-3, 3-to-4 and 4-to- 5 split policies. The corresponding space utilization was ${65.5}\% ,{82.2}\% ,{89.1}\%$ and ${92.3}\%$ respectively. For comparison, we also plot the response times of the ${R}^{ * }$ - tree. As expected,the response time for the range queries improves with the average node utilization. However, there seems to be a point of diminishing returns as $s$ increases. For this reason,we recommend the ’2-to-3’ splitting policy $\left( {s = 2}\right)$ ,which strikes a balance between insertion speed (which deteriorates with $s)$ and search speed,which improves with $s$ .

图9显示了1到2、2到3、3到4和4到5分裂策略下响应时间随查询大小的变化情况。相应的空间利用率分别为${65.5}\% ,{82.2}\% ,{89.1}\%$和${92.3}\%$。为了进行比较，我们还绘制了${R}^{ * }$树的响应时间。正如预期的那样，范围查询的响应时间随着平均节点利用率的提高而改善。然而，随着$s$的增加，似乎存在收益递减的点。出于这个原因，我们推荐“2到3”分裂策略$\left( {s = 2}\right)$，它在插入速度（随着$s)$而下降）和搜索速度（随着$s$而提高）之间取得了平衡。

<!-- Media -->

<!-- figureText: Montgomery County: 39717 line segements; 2-to-3 split policy HilbertR-toss 400.00 500.00 360.00 340.00 320.00 300.00 260.00 240.00 220.00 200.00 180.00 160.00 140.00 120.00 100.00 80.00 60.00 40.00 20.00 0.00 100.00 200.00 300.00 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_6.jpg?x=961&y=148&w=650&h=674&r=0"/>

Figure 7: Montgomery County Dataset; Disk Accesses vs. Query Area

图7：蒙哥马利县数据集；磁盘访问次数与查询区域的关系

<!-- Media -->

### 4.3 Insertion cost

### 4.3 插入成本

The higher space utilization in the Hilbert R-tree comes at the expense of higher insertion cost. As we employ higher split policy the number of cooperating siblings need to be inspected at overflow increases. We see that '2-to-3' policy is a good compromise between the performance and the insertion cost. In this section we compare the insertion cost of the Hilbert R-tree ‘2-to-3’ split with the insertion cost in the ${R}^{ * } -$ tree. Also, show the effect of the split policy on the insertion cost. The cost is measured by the number of disk accesses per insertion.

希尔伯特R树较高的空间利用率是以较高的插入成本为代价的。随着我们采用更高的分裂策略，在溢出时需要检查的协作兄弟节点数量会增加。我们发现“2到3”策略是性能和插入成本之间的一个很好的折衷方案。在本节中，我们比较了希尔伯特R树“2到3”分裂的插入成本与${R}^{ * } -$树的插入成本。此外，还展示了分裂策略对插入成本的影响。成本通过每次插入的磁盘访问次数来衡量。

Table 4.3 shows the insertion cost of the Hilbert R-tree and the ${R}^{ * }$ - tree for the five different datasets. The main observation here is that there is no clear winner in the insertion cost.

表4.3显示了希尔伯特R树和${R}^{ * }$树在五个不同数据集上的插入成本。这里的主要观察结果是，在插入成本方面没有明显的赢家。

Table 4.3 shows the effect of increasing the split policy in the Hilbert R-tree on the insertion cost for ${MGCounty}$ dataset. As expected,the insertion cost

表4.3显示了在${MGCounty}$数据集上，希尔伯特R树增加分裂策略对插入成本的影响。正如预期的那样，插入成本

<!-- Media -->

<!-- figureText: Long Beach: 53145 line segements; 2-to-3 split policy Bulleen R-ton Res Quest 10-3 150.00 200.00 250.00 800.00 700.00 650.00 600.00 500.00 400.00 300.00 250.00 200.00 100.00 0.00 0.00 50.00 100.00 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_7.jpg?x=183&y=156&w=647&h=655&r=0"/>

Figure 8: Long Beach Dataset; Disk Accesses vs. Query Area

图8：长滩数据集；磁盘访问次数与查询区域的关系

Table 1: Comparison Between Insertion Cost in Hilbert R-tree "2-to-3' Split and ${R}^{ * }$ -tree; Disk Accesses per Insertion

表1：希尔伯特R树“2到3”分裂与${R}^{ * }$树插入成本的比较；每次插入的磁盘访问次数

<table><tr><td rowspan="2">dataset</td><td colspan="2">(disk accesses</td></tr><tr><td>Hilbert R-tree (2-to-3 split)</td><td>${R}^{ * } -$ tree</td></tr><tr><td>MGCounty</td><td>3.55</td><td>3.10</td></tr><tr><td>LBeach</td><td>3.56</td><td>4.01</td></tr><tr><td>Points</td><td>3.66</td><td>4.06</td></tr><tr><td>Rects</td><td>3.95</td><td>4.07</td></tr><tr><td>Mix</td><td>3.47</td><td>3.39</td></tr></table>

<table><tbody><tr><td rowspan="2">数据集</td><td colspan="2">（磁盘访问</td></tr><tr><td>希尔伯特R树（2到3分裂）</td><td>${R}^{ * } -$ 树</td></tr><tr><td>MG县</td><td>3.55</td><td>3.10</td></tr><tr><td>L海滩</td><td>3.56</td><td>4.01</td></tr><tr><td>点</td><td>3.66</td><td>4.06</td></tr><tr><td>矩形</td><td>3.95</td><td>4.07</td></tr><tr><td>混合</td><td>3.47</td><td>3.39</td></tr></tbody></table>

<!-- Media -->

increases with the order $s$ of the split policy.

随着分割策略的阶数 $s$ 的增加而增加。

## 5 Conclusions

## 5 结论

In this paper we designed and implemented a superior $\mathrm{R}$ -tree variant,which outperforms all the previous $\mathrm{R}$ - tree methods. The major idea is to introduce a 'good' ordering among rectangles. By simply defining an ordering, the R-tree structure is amenable to deferred splitting, which can make the utilization approach the ${100}\%$ mark as closely as we want. Better packing results in a shallower tree and a higher fanout. If the ordering happens to be 'good', that is, to group similar rectangles together, then the R-tree will in addition have nodes with small MBRs, and eventually, fast response times.

在本文中，我们设计并实现了一种更优的 $\mathrm{R}$ -树变体，它的性能优于以往所有的 $\mathrm{R}$ -树方法。主要思路是在矩形之间引入一种“良好”的排序。通过简单地定义一种排序，R -树结构适合进行延迟分割，这可以使利用率尽可能接近 ${100}\%$ 标记。更好的填充效果会使树的深度更浅，扇出更高。如果这种排序恰好是“良好”的，即能将相似的矩形分组在一起，那么 R -树的节点还将具有较小的最小边界矩形（MBR），最终实现更快的响应时间。

Based on this idea, we designed in detail and implemented the Hilbert R-tree, a dynamic tree structure that is capable of handling insertions and deletions. Experiments on real and synthetic data showed that the proposed Hilbert R-tree with the '2-to-3' splitting policy consistently outperforms all the R-tree methods,with up to ${28}\%$ savings over the best competitor (the ${R}^{ * }$ -tree).

基于这一思路，我们详细设计并实现了希尔伯特 R -树（Hilbert R -tree），这是一种能够处理插入和删除操作的动态树结构。对真实数据和合成数据的实验表明，采用“2 到 3”分割策略的希尔伯特 R -树始终优于所有 R -树方法，与最佳竞争对手（${R}^{ * }$ -树）相比，最多可节省 ${28}\%$ 。

<!-- Media -->

Montgomery County: 39717 line segements; different split policies

蒙哥马利县（Montgomery County）：39717 条线段；不同的分割策略

<!-- figureText: 7-4 300.00 400.00 500.00 Pages Tenched 360.00 320.00 300.00 280.00 260.00 240.00 220.00 200.00 180.00 160.00 140.00 120.00 100.00 80.06 60.00 40.00 20.00 0.00 0.00 100.00 200.00 -->

<img src="https://cdn.noedgeai.com/0195c91a-8dea-7cc4-9d3d-d760172328bb_7.jpg?x=910&y=166&w=646&h=646&r=0"/>

Figure 9: The Effect of The Split Policy; Disk Accesses vs. Query Area

图 9：分割策略的影响；磁盘访问次数与查询区域的关系

Table 2: The Effect of The Split Policy on The Insertion Cost; MGCounty Dataset

表 2：分割策略对插入成本的影响；蒙哥马利县（MGCounty）数据集

<table><tr><td>split policy</td><td>(disk accesses)/insertion</td></tr><tr><td>$1 - \operatorname{to} - 2$</td><td>3.23</td></tr><tr><td>2-to-3</td><td>3.55</td></tr><tr><td>3-to-4</td><td>4.09</td></tr><tr><td>$4 - t = 5$</td><td>4.72</td></tr></table>

<table><tbody><tr><td>拆分策略</td><td>（磁盘访问次数）/插入操作</td></tr><tr><td>$1 - \operatorname{to} - 2$</td><td>3.23</td></tr><tr><td>2到3</td><td>3.55</td></tr><tr><td>3到4</td><td>4.09</td></tr><tr><td>$4 - t = 5$</td><td>4.72</td></tr></tbody></table>

<!-- Media -->

Future work could focus on the analysis of Hilbert R-trees, providing analytical formulas that predict the response time as a function of the characteristics of the data rectangles (count, data density etc).

未来的工作可以聚焦于希尔伯特R树（Hilbert R-trees）的分析，提供能够根据数据矩形的特征（数量、数据密度等）预测响应时间的解析公式。

## References

## 参考文献

[AS91] Walid G. Aref and Hanan Samet. Optimization strategies for spatial query processing. Proc. of VLDB (Very Large Data Bases), pages 81-90, September 1991. [BB82] D. Ballard and C. Brown. Computer Vision. Prentice Hall, 1982.

[Ben75] J.L. Bentley. Multidimensional binary search trees used for associative searching. ${CACM},{18}\left( 9\right)  : {509} - {517}$ ,September 1975.

[Ben75] J.L. 本特利（Bentley）。用于关联搜索的多维二叉搜索树。${CACM},{18}\left( 9\right)  : {509} - {517}$，1975年9月。

[Bia69] T. Bially. Space-filling curves: Their generation and their application to bandwidth reduction. IEEE Trans. on Information Theory, IT-15(6):658-664, November 1969.

[Bia69] T. 比亚利（Bially）。填充空间曲线：其生成及其在降低带宽方面的应用。《电气与电子工程师协会信息论汇刊》（IEEE Trans. on Information Theory），IT - 15(6):658 - 664，1969年11月。

[BKSS90] N. Beckmann, H.-P. Kriegel, R. Schneider, and B. Seeger. The r*-tree: an efficient and robust access method for points and rectangles. ACM SIGMOD, pages 322- 331, May 1990.

[BKSS90] N. 贝克曼（Beckmann）、H.-P. 克里格尔（Kriegel）、R. 施耐德（Schneider）和B. 西格（Seeger）。R*树（r*-tree）：一种高效且稳健的点和矩形访问方法。美国计算机协会管理数据专业组会议（ACM SIGMOD），第322 - 331页，1990年5月。

[Fal88] C. Faloutsos. Gray codes for partial match and range queries. IEEE Trans. on Software Engineering, 14(10):1381-1393, October 1988. early version available as UMIACS-TR-87-4, also CS-TR-1796.

[Fal88] C. 法劳托斯（Faloutsos）。用于部分匹配和范围查询的格雷码。《电气与电子工程师协会软件工程汇刊》（IEEE Trans. on Software Engineering），14(10):1381 - 1393，1988年10月。早期版本可参考UMIACS - TR - 87 - 4，也可参考CS - TR - 1796。

[FR89] C. Faloutsos and S. Roseman. Fractals for secondary key retrieval. Eighth ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS), pages 247-252, March 1989. also available as UMIACS-TR-89-47 and CS-TR-2242.

[FR89] C. 法劳托斯（Faloutsos）和S. 罗斯曼（Roseman）。用于二级键检索的分形。第八届美国计算机协会数据库系统原理研讨会（PODS），第247 - 252页，1989年3月。也可参考UMIACS - TR - 89 - 47和CS - TR - 2242。

[Fre87] Michael Freeston. The bang file: a new kind of grid file. Proc. of ACM SIGMOD, pages 260-269, May 1987.

[Fre87] 迈克尔·弗里斯顿（Michael Freeston）。邦文件（bang file）：一种新型网格文件。美国计算机协会管理数据专业组会议（ACM SIGMOD）论文集，第260 - 269页，1987年5月。

[Gar82] I. Gargantini. An effective way to represent quadtrees. Comm. of ${ACM}\left( {CACM}\right)$ , 25(12):905-910, December 1982.

[Gar82] I. 加尔甘蒂尼（Gargantini）。一种有效的四叉树表示方法。《${ACM}\left( {CACM}\right)$通讯》（Comm. of ${ACM}\left( {CACM}\right)$），25(12):905 - 910，1982年12月。

[Gra92] Grand challenges: High performance computing and communications, 1992. The FY 1992 U.S. Research and Development Program.

[Gra92] 重大挑战：高性能计算与通信，1992年。1992财年美国研发计划。

[Gre89] D. Greene. An implementation and performance analysis of spatial data access methods. Proc. of Data Engineering, pages 606-615, 1989.

[Gre89] D. 格林（Greene）。空间数据访问方法的实现与性能分析。数据工程会议论文集，第606 - 615页，1989年。

[Gri86] J.G. Griffiths. An algorithm for displaying a class of space-filling curves. Software-Practice and Experience, 16(5):403-411, May 1986.

[Gri86] J.G. 格里菲思（Griffiths）。一种显示一类填充空间曲线的算法。《软件实践与经验》（Software - Practice and Experience），16(5):403 - 411，1986年5月。

[Gun89] O. Gunther. The cell tree: an index for geometric data. Proc. Data Engineering, 1989.

[Gun89] O. 冈瑟（Gunther）。单元树（cell tree）：一种几何数据索引。数据工程会议论文集，1989年。

[Gut84a] A. Guttman. New Features for Relational Database Systems to Support CAD Appli-

[Gut84a] A. 古特曼（Guttman）。支持计算机辅助设计应用的关系数据库系统新特性

cations. PhD thesis, University of California, Berkeley, June 1984.

[Gut84b] A. Guttman. R-trees: a dynamic index structure for spatial searching. Proc. ACM SIGMOD, pages 47-57, June 1984.

[HN83] K. Hinrichs and J. Nievergelt. The grid file: a data structure to support proximity queries on spatial objects. Proc. of the WG'83 (Intern. Workshop on Graph Theoretic Concepts in Computer Science), pages ${100} - {113},{1983}$ .

[Jag90a] H. V. Jagadish. Spatial search with polyhedra. Proc. Sixth IEEE Int'l Conf. on Data Engineering, February 1990.

[Jag90b] H.V. Jagadish. Linear clustering of objects with multiple attributes. ACM SIGMOD Conf., pages 332-342, May 1990.

[KF93] I. Kamel and C. Faloutsos. On packing r-trees. In Proc. 2nd International Conference on Information and Knowledge Management(CIKM-93), pages 490- 499, Arlington, VA, November 1993.

[KS91] Curtis P. Kolovson and Michael Stone-braker. Segment indexes: Dynamic indexing techniques for multi-dimensional interval data. Proc. ACM SIGMOD, pages 138-147, May 1991.

[LS90] David B. Lomet and Betty Salzberg. The hb-tree: a multiattribute indexing method with good guaranteed performance. ${ACM}$ ${TODS},{15}\left( 4\right)  : {625} - {658}$ ,December 1990.

[Man77] B. Mandelbrot. Fractal Geometry of Nature. W.H. Freeman, New York, 1977.

$\left\lbrack  {{\mathrm{{OHM}}}^{ + }{84}}\right\rbrack  \mathrm{J}.\mathrm{K}$ . Ousterhout,G. T. Hamachi,R. N. Mayo, W. S. Scott, and G. S. Taylor. Magic: a vlsi layout system. In 21st Design Automation Conference, pages 152- 159, Alburquerque, NM, June 1984.

[Ore86] J. Orenstein. Spatial query processing in an object-oriented database system. Proc. ACM SIGMOD, pages 326-336, May 1986.

[RL85] N. Roussopoulos and D. Leifker. Direct spatial search on pictorial databases using packed r-trees. Proc. ACM SIGMOD, May 1985.

[Rob81] J.T. Robinson. The k-d-b-tree: a search structure for large multidimensional dynamic indexes. Proc. ACM SIGMOD, pages ${10} - {18},{1981}$ .

[Rob81] J.T. 罗宾逊（Robinson）。k - d - b树（k - d - b - tree）：一种用于大型多维动态索引的搜索结构。美国计算机协会管理数据专业组会议（ACM SIGMOD）论文集，第${10} - {18},{1981}$页。

[Sam89] H. Samet. The Design and Analysis of Spatial Data Structures. Addison-Wesley, 1989.

[Sam89] H. 萨梅特（Samet）。《空间数据结构的设计与分析》（The Design and Analysis of Spatial Data Structures）。艾迪生 - 韦斯利出版社（Addison - Wesley），1989年。

[SRF87] T. Sellis, N. Roussopoulos, and C. Falout-sos. The r+tree: a dynamic index for multi-dimensional objects. In Proc. 13th International Conference on VLDB, pages 507-518, England,, September 1987. also available as SRC-TR-87-32, UMIACS-TR- 87-3, CS-TR-1795.

[SRF87] T. 塞利斯（T. Sellis）、N. 鲁索普洛斯（N. Roussopoulos）和 C. 法洛索斯（C. Falout-sos）。R+树（r+tree）：一种用于多维对象的动态索引。收录于《第13届国际超大型数据库会议论文集》，第507 - 518页，英国，1987年9月。也可作为SRC - TR - 87 - 32、UMIACS - TR - 87 - 3、CS - TR - 1795获取。

[SSU91] Avi Silberschatz, Michael Stonebraker, and Jeff Ullman. Database systems: Achievements and opportunities. Comm. of ${ACM}\left( {CACM}\right) ,{34}\left( {10}\right)  : {110} - {120}$ ,October 1991.

[SSU91] 阿维·西尔伯沙茨（Avi Silberschatz）、迈克尔·斯通布雷克（Michael Stonebraker）和杰夫·厄尔曼（Jeff Ullman）。数据库系统：成就与机遇。《${ACM}\left( {CACM}\right) ,{34}\left( {10}\right)  : {110} - {120}$通讯》，1991年10月。

[Whi81] M. White. N-Trees: Large Ordered Indexes for Multi-Dimensional Space. Application Mathematics Research Staff, Statistical Research Division, U.S. Bureau of the Census, December 1981.

[Whi81] M. 怀特（M. White）。N树（N - Trees）：用于多维空间的大型有序索引。美国人口普查局统计研究部应用数学研究组，1981年12月。