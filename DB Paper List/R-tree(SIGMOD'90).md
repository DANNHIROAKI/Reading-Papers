# The ${\mathrm{R}}^{ * }$ -tree: An Efficient and Robust Access Method for Points and Rectangles ${}^{ + }$

# R*树（R*-tree）：一种高效且稳健的点和矩形访问方法 ${}^{ + }$

Norbert Beckmann, Hans-Peter Kriegel

诺伯特·贝克曼（Norbert Beckmann），汉斯 - 彼得·克里格尔（Hans - Peter Kriegel）

Ralf Schneider, Bernhard Seeger

拉尔夫·施奈德（Ralf Schneider），伯恩哈德·西格（Bernhard Seeger）

Praktische Informatik, Universitaet Bremen, D-2800 Bremen 33, West Germany

实用信息学系，德国不来梅大学，邮编：D - 2800，不来梅33区，西德

## Abstract

## 摘要

The R-tree, one of the most popular access methods for rectangles, is based on the heuristic optimization of the area of the enclosing rectangle in each inner node By running numerous experiments in a standardized testbed under highly varying data, queries and operations, we were able to design the R*-tree which incorporates a combined optimization of area, margin and overlap of each enclosing rectangle in the directory Using our standardized testbed in an exhaustive performance comparison,it turned out that the ${R}^{ * }$ -tree clearly outperforms the existing R-tree variants Guttman's linear and quadratic R-tree and Greene's variant of the R-tree This superiority of the R*-tree holds for different types of queries and operations, such as map overlay, for both rectangles and multidimensional points in all experiments From a practical point of view the R*-tree is very attractive because of the following two reasons 1 it efficiently supports point and spatial data at the same time and 2 its implementation cost is only slightly higher than that of other R-trees

R树（R - tree）是最流行的矩形访问方法之一，它基于对每个内部节点中包围矩形面积的启发式优化。通过在标准化测试平台上对高度可变的数据、查询和操作进行大量实验，我们设计出了R*树（R* - tree），它结合了对目录中每个包围矩形的面积、周长和重叠部分的优化。使用我们的标准化测试平台进行全面的性能比较后发现，R*树明显优于现有的R树变体，如古特曼（Guttman）的线性和二次R树以及格林（Greene）的R树变体。在所有实验中，对于不同类型的查询和操作（如地图叠加），无论是矩形还是多维点，R*树都表现出优越性。从实际应用的角度来看，R*树非常有吸引力，原因有二：其一，它能同时高效支持点数据和空间数据；其二，其实现成本仅略高于其他R树。

## 1.Introduction

## 1. 引言

In this paper we will consider spatial access methods (SAMs) which are based on the approximation of a complex spatial object by the minimum bounding rectangle with the sides of the rectangle parallel to the axes of the data space

在本文中，我们将探讨基于用最小边界矩形（矩形的边与数据空间的坐标轴平行）来近似复杂空间对象的空间访问方法（SAMs）。

The most important property of this simple approximation is that a complex object is represented by a limited number of bytes Although a lot of information is lost, minimum bounding rectangles of spatial objects preserve the most essential geometric properties of the object, i e the location of the object and the extension of the object in each axis

这种简单近似方法最重要的特性是，一个复杂对象可以用有限数量的字节来表示。尽管会丢失大量信息，但空间对象的最小边界矩形保留了对象最基本的几何属性，即对象的位置以及对象在每个坐标轴上的延伸范围。

In [SK 88] we showed that known SAMs organizing (minimum bounding) rectangles are based on an underlying point access method (PAM) using one of the following three techniques clipping, transformation and overlapping regions

在文献[SK 88]中，我们表明已知的组织（最小边界）矩形的空间访问方法是基于底层的点访问方法（PAM），采用以下三种技术之一：裁剪、变换和重叠区域。

The most popular SAM for storing rectangles is the R-tree [Gut 84] Following our classification, the R-tree is based on the PAM ${\mathrm{B}}^{ + }$ -tree [Knu 73] using the technique over-lapping regions Thus the R-tree can be easily implemented which considerably contributes to its popularity

最流行的存储矩形的空间访问方法是R树[Gut 84]。根据我们的分类，R树基于点访问方法R*树[Knu 73]，采用重叠区域技术。因此，R树易于实现，这在很大程度上促成了它的流行。

The R-tree is based on a heuristic optimization The optimization criterion which it persues, is to minimize the area of each enclosing rectangle in the inner nodes This criterion is taken for granted and not shown to be the best possible Questions arise such as Why not minimize the margin or the overlap of such minimum bounding rectangles Why not optimize storage utilization? Why not optimize all of these criteria at the same time? Could these criteria interact in a negative way? Only an engineering approach will help to find the best possible combination of optimization criteria

R树基于启发式优化。它所追求的优化准则是最小化内部节点中每个包围矩形的面积。这个准则被认为是理所当然的，但并没有被证明是最优的。由此产生了一些问题，例如：为什么不最小化这种最小边界矩形的周长或重叠部分？为什么不优化存储利用率？为什么不同时优化所有这些准则？这些准则会不会产生负面的相互作用？只有通过工程方法才能找到最优的优化准则组合。

Necessary condition for such an engineering approach is the availability of a standardized testbed which allows us to run large volumes of experiments with highly varying data, queries and operations We have implemented such a standardized testbed and used it for performance comparisons particularly of point access methods [KSSS 89]

这种工程方法的必要条件是要有一个标准化的测试平台，它能让我们对高度可变的数据、查询和操作进行大量实验。我们已经实现了这样一个标准化测试平台，并将其用于性能比较，特别是点访问方法的性能比较[KSSS 89]。

As the result of our research we designed a new R-tree variant, the R*-tree, which outperforms the known R-tree variants under all experiments For many realistic profiles of data and operations the gain in performance is quite considerable Additionally to the usual point query, rectangle intersection and rectangle enclosure query, we have analyzed our new R*-tree for the map overlay operation, also called spatial join, which is one of the most important operations in geographic and environmental database systems

作为我们研究的成果，我们设计了一种新的R树变体——R*树，它在所有实验中都优于已知的R树变体。对于许多实际的数据和操作场景，性能提升相当显著。除了常见的点查询、矩形相交查询和矩形包含查询外，我们还分析了新的R*树在地图叠加操作（也称为空间连接，这是地理和环境数据库系统中最重要的操作之一）中的表现。

---

<!-- Footnote -->

- This work was supported by grant no $\operatorname{Kr}{670}/4 - 3$ from the Deutsche Forschungsgemeinschaft (German Research Society) and by the Ministry of Environmental and Urban Planning of Bremen notice is given that copying is by permission of the Association for Computing

- 这项工作得到了德国研究协会（Deutsche Forschungsgemeinschaft）编号为$\operatorname{Kr}{670}/4 - 3$的资助，以及不来梅环境与城市规划部的支持。特此声明，复制需获得计算机协会的许可。

<!-- Footnote -->

---

This paper is organized as follows In section 2, we introduce the principles of R-trees including their optimization criteria In section 3 we present the existing R-tree variants of Guttman and Greene Section 4 describes in detail the design our new R*-tree The results of the comparisons of the ${R}^{ * }$ -tree with the other $R$ -tree variants are reported in section 5 Section 6 concludes the paper

本文的组织结构如下：第2节介绍R树的原理，包括其优化准则；第3节介绍古特曼和格林现有的R树变体；第4节详细描述我们新的R*树的设计；第5节报告R*树与其他R树变体的比较结果；第6节对本文进行总结。

## 2. Principles of R-trees and possible optimization criteria

## 2. R树原理及可能的优化准则

An $\mathrm{R}$ -tree is a ${\mathrm{B}}^{ + }$ -tree like structure which stores multidimensional rectangles as complete objects without clipping them or transforming them to higher dimensional points before

一棵 $\mathrm{R}$ 树是一种类似 ${\mathrm{B}}^{ + }$ 树的结构，它将多维矩形作为完整对象进行存储，而无需对其进行裁剪或将其转换为更高维的点

A non-leaf node contains entries of the form $({cp}$ , Rectangle) where ${cp}$ is the address of a child node in the R-tree and Rectangle is the minimum bounding rectangle of all rectangles which are entries in that child node A leaf node contains entries of the form(Otd, Rectangle)where Old refers to a record in the database, describing a spatial object and Rectangle is the enclosing rectangle of that spatial object Leaf nodes containing entries of the form (dataobject, Rectangle) are also possible This will not affect the basic structure of the R-tree In the following we will not consider such leaf nodes

非叶节点包含形式为 $({cp}$ ，矩形) 的条目，其中 ${cp}$ 是 R 树中一个子节点的地址，矩形是该子节点中所有矩形条目的最小边界矩形。叶节点包含形式为(旧记录, 矩形)的条目，其中旧记录指数据库中描述一个空间对象的记录，矩形是该空间对象的包围矩形。包含形式为(数据对象, 矩形)条目的叶节点也是可能的。这不会影响 R 树的基本结构。在下面的内容中，我们将不考虑此类叶节点

Let $M$ be the maximum number of entries that will fit in one node and let $m$ be a parameter specifying the minimum number of entries in a node $\left( {2 \leq  \mathrm{m} \leq  \mathrm{M}/2}\right)$ An R-tree satisfies the following properties

设 $M$ 为一个节点所能容纳的最大条目数，设 $m$ 为指定节点中最小条目数的参数 $\left( {2 \leq  \mathrm{m} \leq  \mathrm{M}/2}\right)$ 。一棵 R 树满足以下属性

- The root has at least two children unless it is a leaf

- 除非根节点是叶节点，否则它至少有两个子节点

- Every non-leaf node has between $m$ and $M$ children unless It is the root

- 除非是非根节点，否则每个非叶节点有介于 $m$ 和 $M$ 之间的子节点

- Every leaf node contains between $\mathrm{m}$ and $\mathrm{M}$ entries unless It is the root

- 除非是根节点，否则每个叶节点包含介于 $\mathrm{m}$ 和 $\mathrm{M}$ 之间的条目

- All leaves appear on the same level

- 所有叶节点位于同一层

An R-tree (R*-tree) is completely dynamic, insertions and deletions can be intermixed with queries and no periodic global reorganization is required Obviously, the structure must allow overlapping directory rectangles Thus it cannot guarantee that only one search path is required for an exact match query For further information we refer to [Gut84] We will show in this paper that the overlapping-regions-technique does not imply bad average retrieval performance Here and in the following, we use the term directory rectangle, which is geometrically the minimum bounding rectangle of the underlying rectangles

一棵 R 树（R* 树）是完全动态的，插入和删除操作可以与查询操作混合进行，并且不需要定期进行全局重组。显然，该结构必须允许目录矩形重叠。因此，它不能保证精确匹配查询只需要一条搜索路径。更多信息请参考 [Gut84]。我们将在本文中表明，重叠区域技术并不意味着平均检索性能不佳。在这里和下面的内容中，我们使用术语“目录矩形”，它在几何上是底层矩形的最小边界矩形

The main problem in R-trees is the following For an arbitrary set of rectangles, dynamically build up bounding boxes from subsets of between $m$ and $M$ rectangles,in a way that arbitrary retrieval operations with query rectangles of arbitrary size are supported efficiently The known parameters of good retrieval performance affect each other in a very complex way, such that it is impossible to optimize one of them without influencing other parameters which may cause a deterioration of the overall performance Moreover, since the data rectangles may have very different size and shape and the directory rectangles grow and shrink dynamically, the success of methods which will optimize one parameter is very uncertain Thus a heuristic approach is applied, which is based on many different experiments carried out in a systematic framework

R 树的主要问题如下。对于任意一组矩形，以一种能有效支持使用任意大小查询矩形进行任意检索操作的方式，从 $m$ 到 $M$ 个矩形的子集中动态构建边界框。已知的良好检索性能参数之间以非常复杂的方式相互影响，以至于在不影响其他可能导致整体性能下降的参数的情况下优化其中一个参数是不可能的。此外，由于数据矩形的大小和形状可能差异很大，并且目录矩形会动态增长和收缩，因此优化一个参数的方法的成功与否非常不确定。因此，采用了一种基于在系统框架下进行的许多不同实验的启发式方法

In this section some of the parameters which are essential for the retrieval performance are considered Furthermore, interdependencies between different parameters and optimization criteria are analyzed

在本节中，将考虑一些对检索性能至关重要的参数。此外，还将分析不同参数之间的相互依赖关系和优化准则

(O1) The area covered by a directory rectangle should be minimized, i e the area covered by the bounding rectangle but not covered by the enclosed rectangles, the dead space, should be minimized This will improve performance since decisions which paths have to be traversed, can be taken on higher levels

(O1) 应最小化目录矩形所覆盖的面积，即边界矩形所覆盖但被包围矩形未覆盖的面积，也就是死空间，应最小化。这将提高性能，因为可以在更高层次上决定需要遍历哪些路径

(O2) The overlap between directory rectangles should be minimized This also decreases the number of paths to be traversed

(O2) 应最小化目录矩形之间的重叠。这也会减少需要遍历的路径数量

(O3) The margin of a directory rectangle should be minimized Here the margin is the sum of the lengths of the edges of a rectangle Assuming fixed area, the object with the smallest margin is the square Thus minimizing the margin instead of the area, the directory rectangles will be shaped more quadratic Essentially queries with large quadratic query rectangles will profit from this optimization More important, minimization of the margin will basically improve the structure Since quadratic objects can be packed easier, the bounding boxes of a level will build smaller directory rectangles in the level above Thus clustering rectangles into bounding boxes with only little variance of the lengths of the edges will reduce the area of directory rectangles

(O3) 应最小化目录矩形的周长。这里的周长是矩形各边长度之和。假设面积固定，周长最小的对象是正方形。因此，与最小化面积相比，最小化周长会使目录矩形更接近正方形。本质上，使用大的正方形查询矩形进行的查询将从这种优化中受益。更重要的是，最小化周长将从根本上改善结构。由于正方形对象更容易打包，一层的边界框将在其上一层构建更小的目录矩形。因此，将矩形聚类到边长差异很小的边界框中，将减少目录矩形的面积

(04) Storage utilization should be optimized Higher storage utilization will generally reduce the query cost as the height of the tree will be kept low Evidently, query types with large query rectangles are influenced more since the concentration of rectangles in several nodes will have a stronger effect if the number of found keys is high

(O4) 应优化存储利用率。更高的存储利用率通常会降低查询成本，因为树的高度将保持较低。显然，使用大查询矩形的查询类型受到的影响更大，因为如果找到的键的数量较多，多个节点中矩形的集中程度将产生更强的效果

Keeping the area and overlap of a directory rectangle small, requires more freedom in the number of rectangles stored in one node Thus minimizing these parameters will be paid with lower storage utilization. Moreover, when applying (O1) or (O2) more freedom in choosing the shape is necessary Thus rectangles will be less quadratic With (O1) the overlap between directory rectangles may be affected in a positive way since the covering of the data space is reduced As for every geometric optimization, minimizing the margins will also lead to reduced storage utilization However, since more quadratic directory rectangles support packing better, it will be easier to maintain high storage utilization Obviously, the performance for queries with sufficiently large query rectangles will be affected more by the storage utilization than by the parameters of (O1)-(O3)

要使目录矩形的面积和重叠部分较小，就需要在一个节点中存储的矩形数量上有更大的灵活性。因此，将这些参数最小化会导致存储利用率降低。此外，在应用（O1）或（O2）时，需要在选择形状上有更大的灵活性，这样矩形就不太接近正方形。使用（O1）时，由于数据空间的覆盖范围减小，目录矩形之间的重叠可能会得到积极影响。与所有几何优化一样，最小化边界也会导致存储利用率降低。然而，由于更接近正方形的目录矩形更有利于填充，因此更容易保持较高的存储利用率。显然，对于查询矩形足够大的查询，其性能受存储利用率的影响要比受（O1） - （O3）参数的影响更大。

## 3. R-tree Variants

## 3. R树变体

The R-tree is a dynamic structure Thus all approaches of optimizing the retrieval performance have to be applied during the insertion of a new data rectangle The insertion algorithm calls two more algorithms in which the crucial decisions for good retrieval performance are made The first is the algorithm ChooseSubtree Beginning in the root, descending to a leaf, it finds on every level the most suitable subtree to accomodate the new entry The second is the algorithm Split It is called, if ChooseSubtree ends in a node filled with the maximum number of entries M Split should distribute $M + 1$ rectangles into two nodes in the most appropriate manner

R树是一种动态结构。因此，所有优化检索性能的方法都必须在插入新的数据矩形时应用。插入算法会调用另外两个算法，在这两个算法中会做出对良好检索性能至关重要的决策。第一个是ChooseSubtree（选择子树）算法，它从根节点开始，向下遍历到叶子节点，在每一层找到最适合容纳新条目的子树。第二个是Split（分裂）算法，如果ChooseSubtree算法最终到达一个已填满最大条目数M的节点，就会调用该算法。Split算法应该以最合适的方式将$M + 1$个矩形分配到两个节点中。

In the following, the ChooseSubtree- and Split-algorithms, suggested in available R-tree variants are analyzed and discussed We will first consider the original R-tree as proposed by Guttman in [Gut 84]

接下来，将对现有R树变体中提出的ChooseSubtree和Split算法进行分析和讨论。我们首先考虑Guttman在[Gut 84]中提出的原始R树。

Algorithm ChooseSubtree

ChooseSubtree算法

CS1 Set $\mathrm{N}$ to be the root

CS1：将$\mathrm{N}$设为根节点

CS2 If $\mathrm{N}$ is a leaf,

CS2：如果$\mathrm{N}$是叶子节点

return $\mathrm{N}$

返回$\mathrm{N}$

else

否则

Choose the entry in $\mathrm{N}$ whose rectangle needs least area enlargement to include the new data Resolve ties by choosing the entry with the rectangle of smallest area

选择$\mathrm{N}$中的条目，其矩形包含新数据所需的面积扩展最小。若出现平局，则选择矩形面积最小的条目。

end

结束

CS3 Set $\mathrm{N}$ to be the childnode pointed to by the

CS3：将$\mathrm{N}$设为所选条目子指针指向的子节点，并从CS2开始重复执行

childpointer of the chosen entry an repeat from CS2

Obviously, the method of optimization is to minimize the area covered by a directory rectangle, i e (O1) This may also reduce the overlap and the cpu cost will be relatively low

显然，优化方法是最小化目录矩形所覆盖的面积，即（O1）。这也可能减少重叠，并且CPU成本相对较低。

Guttman discusses split-algorithms with exponential, quadratic and linear cost with respect to the number of entries of a node All of them are designed to minimize the area, covered by the two rectangles resulting from the split The exponential split finds the area with the global minimum, but the cpu cost is too high The others try to find approximations In his experiments, Guttman obtains nearly the same retrieval performance for the linear as for the quadratic version We implemented the R-tree in both variants However in our tests with different distributions, different overlap, variable numbers of data-entries and different combinations of $M$ and $m$ ,the quadratic $R$ -tree yielded much better performance than the linear version (see also section 5) Thus we will only discuss the quadratic algorithm in detail

Guttman讨论了与节点条目数量相关的具有指数、二次和线性成本的分裂算法。所有这些算法都旨在最小化分裂产生的两个矩形所覆盖的面积。指数分裂算法能找到全局最小面积，但CPU成本太高。其他算法则试图找到近似值。在他的实验中，Guttman发现线性版本和二次版本的检索性能几乎相同。我们实现了这两种变体的R树。然而，在我们使用不同分布、不同重叠、不同数量的数据条目以及$M$和$m$的不同组合进行的测试中，二次$R$树的性能比线性版本要好得多（另见第5节）。因此，我们将只详细讨论二次算法。

Algorithm QuadraticSplit

QuadraticSplit算法

[Divide a set of $M + 1$ entries into two groups]

[将一组 $M + 1$ 个条目划分为两组]

QS1 Invoke PickSeeds to choose two entries to be the first entries of the groups

QS1 调用 PickSeeds 算法选择两个条目作为两组的首个条目

QS2 Repeat

QS2 重复

DistributeEntry until all entries are distributed or one of the two groups has $M - m + 1$ entries

DistributeEntry 算法，直到所有条目都被分配完毕，或者两组中的某一组达到 $M - m + 1$ 个条目

QS3 If entries remain, assign them to the other group such that it has the minimum number $m$

QS3 如果还有剩余条目，将它们分配到另一组，使得该组的条目数量达到最小值 $m$

## Algorithm PickSeeds

## 算法 PickSeeds

PS1 For each pair of entries E1 and E2, compose a rectangle $\mathrm{R}$ including $\mathrm{E}1$ rectangle and $\mathrm{E}2$ rectangle Calculate $\mathrm{d} = \operatorname{area}\left( \mathrm{R}\right)  - \operatorname{area}\left( {\mathrm{E}1\text{rectangle}}\right)  -$ area(E2 rectangle)

PS1 对于每一对条目 E1 和 E2，组成一个包含 $\mathrm{E}1$ 矩形和 $\mathrm{E}2$ 矩形的矩形 $\mathrm{R}$，计算 $\mathrm{d} = \operatorname{area}\left( \mathrm{R}\right)  - \operatorname{area}\left( {\mathrm{E}1\text{rectangle}}\right)  -$（E2 矩形的面积）

PS2 Choose the pair with the largest d

PS2 选择 d 值最大的那一对

## Algorithm DistributeEntry

## 算法 DistributeEntry

DE1 Invoke PickNext to choose the next entry to be assigned

DE1 调用 PickNext 算法选择下一个要分配的条目

DE2 Add it to the group whose covering rectangle will have to be enlarged least to accommodate it Resolve ties by adding the entry to the group with the smallest area, then to the one with the fewer entries, then to enther

DE2 将其添加到覆盖矩形扩展最少就能容纳它的组中。若出现平局，则将该条目添加到面积最小的组，若仍平局则添加到条目数量最少的组，若还是平局则任选一组

## Algorithm PickNext

## 算法 PickNext

PN1 For each entry $E$ not yet in a group,calculate ${d}_{1} =$ the area increase required in the covering rectangle of Group 1 to include E Rectangle Calculate ${d}_{2}$ analogously for Group 2

PN1 对于尚未分组的每个条目 $E$，计算 ${d}_{1} =$（将 E 矩形纳入组 1 的覆盖矩形所需的面积增加量）。类似地，为组 2 计算 ${d}_{2}$

PN2 Choose the entry with the maximum difference between ${d}_{1}$ and ${d}_{2}$

PN2 选择 ${d}_{1}$ 和 ${d}_{2}$ 差值最大的条目

The algorithm PickSeeds finds the two rectangles which would waste the largest area put in one group In this sense the two rectangles are the most distant ones It is important to mention that the seeds will tend to be small too, if the rectangles to be distributed are of very different size (and) or the overlap between them is high The algorithm DistributeEntry assigns the remaining entries by the criterion of minimum area PickNext chooses the entry with the best area-goodness-value in every situation

PickSeeds 算法会找出如果放在一组会浪费最大面积的两个矩形。从这个意义上说，这两个矩形是距离最远的。需要注意的是，如果要分配的矩形大小差异很大，或者它们之间的重叠度很高，那么种子矩形往往也会很小。DistributeEntry 算法根据最小面积准则分配剩余条目。PickNext 算法在每种情况下都会选择面积优度值最佳的条目

If this algorithm starts with small seeds, problems may occur If in d-1 of the d axes a far away rectangle has nearly the same coordinates as one of the seeds, it will be distributed first Indeed, the area and the area enlargement of the created needle-like bounding rectangle will be very small, but the distance is very large This may initiate a very bad split Moreover, the algorithm tends to prefer the bounding rectangle, created from the first assignment of a rectangle to one seed Since it was enlarged, it will be larger than others Thus it needs less area enlargement to include the next entry, it will be enlarged again, and so on Another problem is, that if one group has reached the maximum number of entries $\mathrm{M} - \mathrm{m} + 1$ ,all remaining entries are assigned to the other group without considering geometric properties Figure 1 (see section 4 3) gives an example showing all these problems The result is either a split with much overlap (fig 1c) or a split with uneven distribution of the entries reducing the storage utilization (fig 1b)

如果该算法以小种子矩形开始，可能会出现问题。如果在 d 个坐标轴中的 d - 1 个坐标轴上，一个远处的矩形与其中一个种子矩形的坐标几乎相同，那么它会首先被分配。实际上，所创建的针状边界矩形的面积和面积扩展会非常小，但距离却很大。这可能会导致非常糟糕的划分。此外，该算法倾向于优先选择由一个矩形首次分配给一个种子矩形所创建的边界矩形。由于它已经被扩展过，所以会比其他矩形大。因此，它在纳入下一个条目时所需的面积扩展更小，会再次被扩展，依此类推。另一个问题是，如果一组达到了最大条目数 $\mathrm{M} - \mathrm{m} + 1$，所有剩余条目都会被分配到另一组，而不考虑几何属性。图 1（见第 4.3 节）给出了一个展示所有这些问题的示例。结果要么是划分后重叠很多（图 1c），要么是条目分布不均匀，降低了存储利用率（图 1b）

We tested the quadratic split of our R-tree implementation varying the minimum number of entries $\mathrm{m} = {20}\% ,{30}\%$ , ${35}\% ,{40}\%$ and ${45}\%$ relatively to $M$ and obtained the best retrieval performance with $\mathrm{m}$ set to ${40}\%$

我们测试了R树（R-tree）实现的二次分裂，改变了相对于$M$的最小条目数$\mathrm{m} = {20}\% ,{30}\%$、${35}\% ,{40}\%$和${45}\%$，并在将$\mathrm{m}$设置为${40}\%$时获得了最佳的检索性能

On the occasion of comparing the R-tree with other structures storing rectangles, Greene proposed the following alternative split-algorithm [Gre 89] To determine the appropriate path to insert a new entry she uses Guttman's original ChooseSubtree-algorithm

在将R树（R-tree）与其他存储矩形的结构进行比较时，格林（Greene）提出了以下替代分裂算法[Gre 89]。为了确定插入新条目的合适路径，她使用了古特曼（Guttman）的原始选择子树算法

Algorithm Greene's-Split

格林分裂算法（Greene's-Split）

[Divide a set of $M + 1$ entries into two groups]

[将一组$M + 1$个条目划分为两组]

GS1 Invoke ChooseAxis to determine the axis perpendicular to which the split is to be performed

GS1：调用选择轴算法（ChooseAxis）来确定进行分裂所垂直的轴

GS2 Invoke Distribute

GS2：调用分配算法（Distribute）

Algorithm ChooseAxis

选择轴算法（ChooseAxis）

CAI Invoke PickSeeds (see p 5) to find the two most distant rectangles of the current node

CA1：调用选择种子算法（PickSeeds）（见第5页）来找到当前节点中距离最远的两个矩形

CA2 For each axis record the separation of the two seeds

CA2：记录每个轴上两个种子的分隔情况

CA3 Normalize the separations by dividing them by the length of the nodes enclosing rectangle along the appropriate axis

CA3：通过将分隔值除以节点包围矩形在相应轴上的长度来对分隔值进行归一化处理

CA4 Return the axis with the greatest normalized separation

CA4：返回归一化分隔值最大的轴

Algorithm Distribute

分配算法（Distribute）

D1 Sort the entries by the low value of their rectangles along the chosen axis

D1：根据所选轴上矩形的最小值对条目进行排序

D2 Assign the first $\left( {M + 1}\right)$ div 2 entries to one group,the last $\left( {M + 1}\right)$ div 2 entries to the other

D2：将前$\left( {M + 1}\right)$整除2个条目分配到一组，后$\left( {M + 1}\right)$整除2个条目分配到另一组

D3 If $M + 1$ is odd,then assign the remaining entry to the group whose enclosing rectangle will be increased least by its addition

D3：如果$M + 1$是奇数，则将剩余的条目分配到添加该条目后包围矩形增加最小的组

Almost the only geometric criterion used in Greene's split algorithm is the choice of the split axis Although choosing a suitable split axis is important, our investigations show that more geometric optimization criteria have to be applied to considerably improve the retrieval performance of the R-tree In spite of a well clustering, in some situations Greene's split method cannot find the "right" axis and thus a very bad split may result Figure 2b (see p 12) depicts such a situation

格林分裂算法（Greene's split algorithm）中几乎唯一使用的几何准则是分裂轴的选择。尽管选择合适的分裂轴很重要，但我们的研究表明，必须应用更多的几何优化准则才能显著提高R树（R-tree）的检索性能。尽管聚类效果良好，但在某些情况下，格林分裂方法无法找到“正确”的轴，从而可能导致非常糟糕的分裂。图2b（见第12页）描述了这样一种情况

### 4.The R*-tree

### 4. R*树

### 4.1 Algorithm ChooseSubtree

### 4.1 选择子树算法

To solve the problem of choosing an appropriate insertion path, previous R-tree versions take only the area parameter into consideration In our investigations, we tested the parameters area, margin and overlap in different combinations, where the overlap of an entry is defined as follows

为解决选择合适插入路径的问题，之前的R树版本仅考虑面积参数。在我们的研究中，我们测试了面积、周长和重叠这几个参数的不同组合，其中条目的重叠定义如下

Let ${E}_{1},{E}_{p}$ be the entries in the current node Then

设${E}_{1},{E}_{p}$为当前节点中的条目。那么

$$
\operatorname{overlap}\left( {E}_{k}\right)  = \mathop{\sum }\limits_{{i = 1,i \neq  k}}^{p}\operatorname{area}\left( {{E}_{k}\text{Rectangle} \cap  {E}_{r}\text{Rectangle}}\right) ,1 \leq  k \leq  p
$$

The version with the best retrieval performance is described in the following algorithm

具有最佳检索性能的版本在以下算法中描述

Algorithm ChooseSubtree

选择子树算法

CS1 Set $\mathrm{N}$ to be the root

CS1：将$\mathrm{N}$设为根节点

CS2 If $N$ is a leaf,

CS2：如果$N$是叶子节点，

return $\mathrm{N}$

返回$\mathrm{N}$

else

否则

if the childpointers in $\mathrm{N}$ point to leaves [determine the minimum overlap cost],

如果$\mathrm{N}$中的子指针指向叶子节点（确定最小重叠成本），

choose the entry in $N$ whose rectangle needs least

选择$N$中其矩形包含新数据矩形所需重叠扩展最小的条目

overlap enlargement to include the new data

通过选择其矩形所需面积扩展最小的条目来解决平局情况，

rectangle Resolve ties by choosing the entry

然后

whose rectangle needs least area enlargement,

then

the entry with the rectangle of smallest area if the childpointers in $\mathrm{N}$ do not point to leaves

如果$\mathrm{N}$中的子指针不指向叶子节点，则选择具有最小面积矩形的条目

[determine the minimum area cost],

[确定最小面积成本]

choose the entry in $\mathrm{N}$ whose rectangle needs least area enlargement to include the new data rectangle Resolve ties by choosing the entry with the rectangle of smallest area end CS3 Set $N$ to be the childnode pointed to by the childpointer of the chosen entry and repeat from CS2

选择$\mathrm{N}$中其矩形需要最小面积扩展以包含新数据矩形的条目。若出现平局，则选择矩形面积最小的条目。结束CS3。将$N$设为所选条目的子指针所指向的子节点，并从CS2开始重复操作。

For choosing the best non-leaf node, alternative methods did not outperform Guttman's original algorithm For the leaf nodes, minimizing the overlap performed slightly better

对于选择最佳非叶节点，替代方法的表现并未优于古特曼（Guttman）的原始算法。对于叶节点，最小化重叠的方法表现略好。

In this version, the cpu cost of determining the overlap is quadratic in the number of entries, because for each entry the overlap with all other entries of the node has to be calculated However, for large node sizes we can reduce the number of entries for which the calculation has to be done, since for very distant rectangles the probability to yield the minimum overlap is very small Thus, in order to reduce the cpu cost, this part of the algorithm might be modified as follows

在这个版本中，确定重叠的CPU成本与条目数量呈二次关系，因为对于每个条目，都必须计算其与节点中所有其他条目的重叠。然而，对于大节点大小，我们可以减少需要进行计算的条目数量，因为对于距离非常远的矩形，产生最小重叠的概率非常小。因此，为了降低CPU成本，算法的这部分可以按如下方式修改。

[determine the nearly minimum overlap cost] Sort the rectangles in $N$ in increasing order of their area enlargement needed to include the new data rectangle

[确定近似最小重叠成本] 按包含新数据矩形所需的面积扩展从小到大的顺序对$N$中的矩形进行排序。

Let $A$ be the group of the first $p$ entries From the entries in A, considering all entries in $\mathrm{N}$ ,choose the entry whose rectangle needs least overlap enlargement Resolve ties as described above

设$A$为前$p$个条目的组。从A中的条目出发，考虑$\mathrm{N}$中的所有条目，选择其矩形需要最小重叠扩展的条目。按上述方法解决平局情况。

For two dimensions we found that with $p$ set to 32 there is nearly no reduction of retrieval performance to state For more than two dimensions further tests have to be done Nevertheless the cpu cost remains higher than the original version of ChooseSubtree, but the number of disc accesses is reduced for the exact match query preceding each insertion and is reduced for the ChooseSubtree algorithm itself

对于二维情况，我们发现当$p$设为32时，检索性能几乎没有下降。对于超过二维的情况，还需要进行进一步的测试。尽管如此，CPU成本仍然高于原始版本的选择子树算法，但对于每次插入前的精确匹配查询，磁盘访问次数减少了，并且选择子树算法本身的磁盘访问次数也减少了。

The tests showed that the ChooseSubtree optimization improves the retrieval performance particulary in the following situation Queries with small query rectangles on datafiles with non-uniformly distributed small rectangles or points

测试表明，选择子树优化在以下情况下特别能提高检索性能：在具有非均匀分布的小矩形或点的数据文件上进行小查询矩形的查询。

In the other cases the performance of Guttman's algorithm was similar to this one Thus principally an improvement of robustness can be stated

在其他情况下，古特曼（Guttman）算法的性能与此算法相似。因此，原则上可以说鲁棒性得到了提高。

## 4 2 Split of the R*-tree

## 4 2 R*树的分裂

The R*-tree uses the following method to find good splits Along each axis, the entries are first sorted by the lower value, then sorted by the upper value of their rectangles For each sort $M - {2m} + 2$ distributions of the $M + 1$ entries into two groups are determined,where the $k$ -th distribution ( $k =$ $1,,\left( {\mathrm{M} - 2\mathrm{m} + 2}\right) )$ is described as follows The first group contains the first $\left( {m - 1}\right)  + k$ entries,the second group contains the remaining entries

R*树使用以下方法来找到良好的分裂。沿着每个轴，首先按矩形的下限值对条目进行排序，然后按矩形的上限值进行排序。对于每次排序，确定$M + 1$个条目分成两组的$M - {2m} + 2$种分布，其中第$k$种分布（$k =$ $1,,\left( {\mathrm{M} - 2\mathrm{m} + 2}\right) )$）描述如下：第一组包含前$\left( {m - 1}\right)  + k$个条目，第二组包含其余条目。

For each distribution goodness values are determined Depending on these goodness values the final distribution of the entries is determined Three different goodness values and different approaches of using them in different combinations are tested experimentally

对于每种分布，确定其优度值。根据这些优度值，确定条目的最终分布。通过实验测试了三种不同的优度值以及以不同组合使用它们的不同方法。

(1) area-value area[bb(first group)] +

(1) 面积值 area[bb(第一组)] +

area[bb(second group)]

area[bb(第二组)]

(11) margin-value margin[bb(first group)] +

(11) 边界值 margin[bb(第一组)] +

margin[bb(second group)]

margin[bb(第二组)]

(111) overlap-value area[bb(first group) $\cap$

(111) 重叠值区域[第一组的边界框(bb(first group)) $\cap$

bb(second group)]

第二组的边界框(bb(second group))]

Here bb denotes the bounding box of a set of rectangles

这里，bb 表示一组矩形的边界框

Possible methods of processing are to determine

可能的处理方法是确定

- the minimum over one axis or one sort

- 一个轴或一种排序上的最小值

- the minimum of the sum of the goodness values over one axis or one sort

- 一个轴或一种排序上的优度值之和的最小值

- the overall minimum

- 全局最小值

The obtained values may be applied to determine a split axis or the final distribution (on a chosen split axis) The best overall performance resulted from the following algorithm

所获得的值可用于确定分割轴或最终分布（在选定的分割轴上）。以下算法可实现最佳的整体性能

Algorithm Split

分割算法

S1 Invoke ChooseSplitAxis to determine the axis,

S1 调用选择分割轴函数（ChooseSplitAxis）来确定轴

perpendicular to which the split is performed

分割操作垂直于该轴进行

S2 Invoke ChooseSplitIndex to determine the best

S2 调用选择分割索引函数（ChooseSplitIndex）来确定最佳

distribution into two groups along that axis

沿该轴分为两组的分布

S3 Distribute the entries into two groups

S3 将条目分配到两组中

## Algorithm ChooseSplitAxis

## 选择分割轴算法

CSA1 For each axis

CSA1 对于每个轴

Sort the entries by the lower then by the upper value of their rectangles and determine all distributions as described above Compute S, the sum of all margin-values of the different distributions

按照矩形的下限值对条目进行排序，然后再按上限值排序，并按照上述描述确定所有分布。计算S，即不同分布的所有边距值之和

end

结束

CSA2 Choose the axis with the minimum $S$ as split axis

CSA2：选择$S$值最小的轴作为分割轴

Algorithm ChooseSplitIndex

算法：选择分割索引

CSI1 Along the chosen split axis, choose the distribution with the minimum overlap-value Resolve ties by choosing the distribution with minimum area-value

CSI1：沿着选定的分割轴，选择重叠值最小的分布。若出现平局，则选择面积值最小的分布

The split algorithm is tested with $\mathrm{m} = {20}\% ,{30}\% ,{40}\%$ and 45% of the maximum number of entries M As ex-periments with several values of $M$ have shown, $m = {40}\%$ yields the best performance Additionally,we varied $m$ over the life cycle of one and the same R*-tree in order to correlate the storage utilization with geometric paremeters However, even the following method did result in worse retrieval performance Compute a split using ${m}_{1} = {30}\%$ of $M$ ,then compute a split using ${m}_{2} = {40}\%$ If split $\left( {m}_{2}\right)$ yields overlap and split $\left( {m}_{1}\right)$ does not,take split $\left( {m}_{1}\right)$ ,otherwise take split $\left( {\mathrm{m}}_{2}\right)$

使用$\mathrm{m} = {20}\% ,{30}\% ,{40}\%$和最大条目数M的45%对分割算法进行测试。对$M$的多个值进行的实验表明，$m = {40}\%$的性能最佳。此外，我们在同一棵R*树的生命周期内改变$m$，以便将存储利用率与几何参数相关联。然而，即使采用以下方法，检索性能仍然较差。使用$M$的${m}_{1} = {30}\%$计算一次分割，然后使用${m}_{2} = {40}\%$计算一次分割。如果分割$\left( {m}_{2}\right)$产生重叠，而分割$\left( {m}_{1}\right)$不产生重叠，则选择分割$\left( {m}_{1}\right)$；否则，选择分割$\left( {\mathrm{m}}_{2}\right)$

Concerning the cost of the split algorithm of the R*-tree we will mention the following facts For each axis (dimension) the entries have to be sorted two times which requires $\mathrm{O}\left( {\mathrm{M}\log \left( \mathrm{M}\right) }\right)$ time As an experimental cost analysis has shown, this needs about half of the cost of the split The remaining split cost is spent as follows For each axis the margin of $2 * \left( {2 * \left( {M - {2m} + 2}\right) }\right)$ rectangles and the overlap of 2*(M-2m+2) distributions have to be calculated

关于R*树分割算法的成本，我们将提及以下事实。对于每个轴（维度），条目需要排序两次，这需要$\mathrm{O}\left( {\mathrm{M}\log \left( \mathrm{M}\right) }\right)$的时间。实验成本分析表明，这大约占分割成本的一半。其余的分割成本花费如下：对于每个轴，需要计算$2 * \left( {2 * \left( {M - {2m} + 2}\right) }\right)$个矩形的边距和2*(M - 2m + 2)种分布的重叠

## 4 3 Forced Reinsert

## 4 3 强制重新插入

Both, R-tree and R*-tree are nondeterministic in allocating the entries onto the nodes i e different sequences of insertions will build up different trees For this reason the R-tree suffers from its old entries Data rectangles inserted during the early growth of the structure may have introduced directory rectangles, which are not suitable to guarantee a good retrieval performance in the current situation A very local reorganization of the directory rectangles is performend during a split But this is rather poor and therefore it is desirable to have a more powerful and less local instrument to reorganize the structure

R树和R*树在将条目分配到节点时都是非确定性的，即不同的插入顺序会构建出不同的树。因此，R树会受到旧条目的影响。在结构早期增长阶段插入的数据矩形可能引入了不适合在当前情况下保证良好检索性能的目录矩形。在分割过程中会对目录矩形进行非常局部的重组，但这相当有限，因此需要一种更强大、更全局的工具来重组结构

The discussed problem would be maintained or even worsened, if underfilled nodes, resulting from deletion of records would be merged under the old parent Thus the known approach of treating underfilled nodes in an R-tree is to delete the node and to reinsert the orphaned entries in the corresponding level [Gut 84] This way the ChooseSubtree algorithm has a new chance of distributing entries into different nodes

如果将因记录删除而产生的未填满节点在旧父节点下合并，那么上述讨论的问题将持续存在甚至恶化。因此，处理R树中未填满节点的已知方法是删除该节点，并将孤立的条目重新插入到相应的层级中[Gut 84]。这样，选择子树算法就有了将条目分配到不同节点的新机会

Since it was to be expected, that the deletion and reinsertion of old data rectangles would improve the retrieval performance, we made the following simple experiment with the linear R-tree Insert 20000 uniformly distributed rectangles Delete the first 10000 rectangles and insert them again The result was a performance improvement of ${20}\%$ up to ${50}\% \left( 1\right)$ depending on the types of the queries Therefore to delete randomly half of the data and then to insert it again seems to be a very simple way of tuning existing R-tree datafiles But this is a static situation, and for nearly static datafiles the pack algorithm [RL 85] is a more sophisticated approach

由于可以预期，删除并重新插入旧的数据矩形会提高检索性能，我们对线性R树进行了以下简单实验。插入20000个均匀分布的矩形，删除前10000个矩形，然后再重新插入。结果表明，根据查询类型的不同，性能提升了${20}\%$至${50}\% \left( 1\right)$。因此，随机删除一半的数据，然后再重新插入，似乎是调整现有R树数据文件的一种非常简单的方法。但这是一种静态情况，对于几乎静态的数据文件，打包算法[RL 85]是一种更复杂的方法

To achieve dynamic reorganizations, the R*-tree forces entries to be reinserted during the insertion routine The following algorithm is based on the ability of the insert routine to insert entries on every level of the tree as already required by the deletion algorithm [Gut 84] Except for the overflow treatment, it is the same as described originally by Guttman and therefore it is only sketched here

为了实现动态重组，R*树在插入例程中强制重新插入条目。以下算法基于插入例程在树的每个层级插入条目的能力，这也是删除算法[Gut 84]所要求的。除了溢出处理外，它与Guttman最初描述的算法相同，因此这里仅作简要概述

## Algorithm InsertData

## 算法：插入数据

ID1 Invoke Insert starting with the leaf level as a parameter, to insert a new data rectangle

ID1：以叶节点层级作为参数调用插入操作，以插入一个新的数据矩形

## Algorithm Insert

## 算法：插入

I1 Invoke ChooseSubtree, with the level as a parameter, to find an appropriate node $N$ ,in which to place the new entry $E$

I1：以层级作为参数调用选择子树操作，以找到一个合适的节点$N$，将新条目$E$放入其中

I2 If $\mathrm{N}$ has less than $\mathrm{M}$ entries,accommodate $\mathrm{E}$ in $\mathrm{N}$ If $\mathrm{N}$ has $\mathrm{M}$ entries,invoke OverflowTreatment with the level of $N$ as a parameter [for reinsertion or split]

I2 如果$\mathrm{N}$的条目少于$\mathrm{M}$个，将$\mathrm{E}$放入$\mathrm{N}$中。如果$\mathrm{N}$有$\mathrm{M}$个条目，则以$N$的层级作为参数调用溢出处理（用于重新插入或拆分）

I3 If OverflowTreatment was called and a split was performed, propagate OverflowTreatment upwards if necessary

I3 如果调用了溢出处理并进行了拆分，则在必要时向上传播溢出处理

If OverflowTreatment caused a split of the root, create a new root

如果溢出处理导致根节点拆分，则创建一个新的根节点

I4 Adjust all covering rectangles in the insertion path such that they are minimum bounding boxes enclosing their children rectangles

I4 调整插入路径中的所有覆盖矩形，使其成为包含其子矩形的最小边界框

## Algorithm OverflowTreatment

## 算法：溢出处理

OT1 If the level is not the root level and this is the first call of OverflowTreatment in the given level during the insertion of one data rectangle, then invoke Relnsert

OT1 如果当前层级不是根层级，并且这是在插入一个数据矩形期间在给定层级上首次调用溢出处理，则调用重新插入

else

否则

invoke Split

调用拆分

end

结束

Algorithm ReInsert

算法：重新插入

RI1 For all $\mathrm{M} + 1$ entries of a node $\mathrm{N}$ ,compute the distance between the centers of their rectangles and the center of the bounding rectangle of $N$

RI1 对于节点$\mathrm{N}$的所有$\mathrm{M} + 1$个条目，计算它们矩形的中心与$N$的边界矩形中心之间的距离

RI2 Sort the entries in decreasing order of their distances computed in RI1

RI2 按照在RI1中计算的距离降序对条目进行排序

RI3 Remove the first $p$ entries from $N$ and adjust the bounding rectangle of $\mathrm{N}$

RI3 从$N$中移除前$p$个条目，并调整$\mathrm{N}$的边界矩形

RI4 In the sort, defined in RI2, starting with the maximum distance (= far reinsert) or minimum distance (= close reinsert), invoke Insert to reinsert the entries

RI4 在RI2定义的排序中，从最大距离（= 远重新插入）或最小距离（= 近重新插入）开始，调用插入操作以重新插入这些条目

If a new data rectangle is inserted, each first overflow treatment on each level will be a reinsertion of $p$ entries This may cause a split in the node which caused the overflow if all entries are reinserted in the same location Otherwise splits may occur in one or more other nodes, but in many situations splits are completely prevented The parameter $p$ can be varied independently for leaf nodes and non-leaf nodes as part of performance tuning, and different values were tested experimentally The experiments have shown that $\mathrm{p} = {30}\%$ of $\mathrm{M}$ for leaf nodes as well as for non-leaf nodes yields the best performance Furthermore, for all data files and query files close reinsert outperforms far reinsert Close reinsert prefers the node which included the entries before, and this is intended, because its enclosing rectangle was reduced in size Thus this node has lower probability to be selected by ChooseSubtree again

如果插入了一个新的数据矩形，每个层级上的首次溢出处理将是对$p$个条目的重新插入。如果所有条目都重新插入到同一位置，这可能会导致导致溢出的节点发生拆分。否则，拆分可能会在一个或多个其他节点中发生，但在许多情况下，拆分可以完全避免。作为性能调优的一部分，可以为叶节点和非叶节点独立地改变参数$p$，并通过实验测试了不同的值。实验表明，对于叶节点和非叶节点，$\mathrm{M}$的$\mathrm{p} = {30}\%$能产生最佳性能。此外，对于所有数据文件和查询文件，近重新插入的性能优于远重新插入。近重新插入更倾向于之前包含这些条目的节点，这是有意为之的，因为其包围矩形的大小已经减小。因此，该节点再次被选择子树算法选中的概率较低

## Summarizing, we can say

## 总结来说，我们可以说

- Forced reinsert changes entries between neighboring nodes and thus decreases the overlap

- 强制重新插入会改变相邻节点之间的条目，从而减少重叠

- As a side effect, storage utilization is improved

- 作为附带效果，存储利用率得到提高

- Due to more restructuring, less splits occur

- 由于进行了更多的结构调整，分裂情况会减少

- Since the outer rectangles of a node are reinserted, the shape of the directory rectangles will be more quadratic As discussed before, this is a desirable property

- 由于节点的外部矩形被重新插入，目录矩形的形状将更接近正方形。如前所述，这是一个理想的特性

Obviously, the cpu cost will be higher now since the insertion routine is called more often This is alleviated, because less splits have to be performed The experiments show that the average number of disc accesses for insertions increases only about $4\%$ (and remains the lowest of all R-tree variants), if Forced Reinsert is applied to the R*-tree This is particularly due to the structure improving properties of the insertion algorithm

显然，由于插入例程被更频繁地调用，现在的CPU成本会更高。不过，由于需要执行的分裂操作减少，这种情况得到了缓解。实验表明，如果将强制重新插入应用于R*树，插入操作的平均磁盘访问次数仅增加约$4\%$（并且仍然是所有R树变体中最少的）。这尤其得益于插入算法对结构的改进特性

<!-- Media -->

<!-- figureText: Figure 1b: Figure 1c: 0 Figure 1d Figure 1e Greene s split Spik of the ${R}^{ * }$ -tree, $m = {40}\%$ Figure 2b. Greene s split where Figure 2c: Split of the ${R}^{ * }$ tree where the spintaxis is vertical Spill of the quadratic Split of the quadratic $R$ -tree, $R$ -tree, $m = {30}\%$ $\mathrm{m} = {40}\%$ the splintaxis is horizontal -->

<img src="https://cdn.noedgeai.com/0195c90a-7932-7243-9d4f-573b5620f0fe_5.jpg?x=935&y=731&w=718&h=1259&r=0"/>

<!-- Media -->

## 5. Performance Comparison 5 1 Experimental Setup and Results of the Experiments

## 5. 性能比较 5.1 实验设置和实验结果

We ran the performance comparison on SUN workstations under UNIX using Modula-2 implementaions of the different R-tree variants and our R*-tree Analogously to our performance comparison of PAM's and SAM's in [KSSS 89] we keep the last accessed path of the trees in main memory If orphaned entries occur from insertions or deletions, they are stored in main memory additionally to the path

我们在运行UNIX系统的SUN工作站上进行了性能比较，使用不同R树变体和我们的R*树的Modula - 2实现。与我们在[KSSS 89]中对PAM和SAM的性能比较类似，我们将树的最后访问路径保存在主内存中。如果插入或删除操作产生了孤立条目，它们将除路径之外额外存储在主内存中

In order to keep the performance comparison manageable, we have chosen the page size for data and directory pages to be 1024 bytes which is at the lower end of realistic page sizes Using smaller page sizes, we obtain similar performance results as for much larger file sizes From the chosen page size the maximum number of entries in directory pages is 56 According to our standardized testbed we have restricted the maximum number of entries in a data page to 50

为了使性能比较易于管理，我们选择数据页和目录页的页面大小为1024字节，这处于实际页面大小的下限。使用较小的页面大小，我们获得的性能结果与使用大得多的文件大小时相似。根据所选的页面大小，目录页中的最大条目数为56。根据我们的标准化测试平台，我们将数据页中的最大条目数限制为50

As candidates of our performance comparison we selected the R-tree with quadratic split algorithm (abbre- viation qua Gut), Greene's variant of the R-tree (Greene) and our R*-tree where the parameters of the different structures are set to the best values as described in the previous sections Additionally, we tested the most popular R-tree implementation, the variant with the linear split algorithm (In Gut) The popularity of the linear R-tree is due to the statement in the original paper [Gut84] that no essential performance gain resulted from the quadratic version vs the linear version For the linear R-tree we found $\mathrm{m} = {20}\%$ (of M) to be the variant with the best performance

作为性能比较的候选对象，我们选择了采用二次分裂算法的R树（缩写为qua Gut）、Greene的R树变体（Greene）以及我们的R*树，其中不同结构的参数设置为前几节中描述的最佳值。此外，我们还测试了最流行的R树实现，即采用线性分裂算法的变体（In Gut）。线性R树的流行是因为原始论文[Gut84]中指出，二次版本与线性版本相比没有显著的性能提升。对于线性R树，我们发现$\mathrm{m} = {20}\%$（M的）是性能最佳的变体

To compare the performance of the four structures we selected six data files containing about 100,000 2- dimensional rectangle Each rectangle is assumed to be in the unit cube $\lbrack 0,1{)}^{2}$ In the following each data file is described by the distribution of the centers of the rectangles and by the tripel $\left( {n,{\mu }_{\text{area }},n{v}_{\text{area }}}\right)$ Here $n$ denotes the number of rectangles, ${\mu }_{\text{area }}$ is the mean value of the area of a rectangle and ${\mathrm{{nv}}}_{\text{area }} = {\sigma }_{\text{area }}/{\mu }_{\text{area }}$ is the normalized variance where ${\sigma }_{\text{area }}$ denotes the variance of the areas of the rectangles Obviously,the parameter ${\mathrm{{nv}}}_{\text{area }}$ increases independently of the distribution the more the areas of the rectangles differ from the mean value and the average overlap is simply obtained by ${n}^{ * }{\mu }_{\text{area }}$

为了比较这四种结构的性能，我们选择了六个包含约100,000个二维矩形的数据文件。每个矩形都假设位于单位立方体$\lbrack 0,1{)}^{2}$中。在下面，每个数据文件由矩形中心的分布和三元组$\left( {n,{\mu }_{\text{area }},n{v}_{\text{area }}}\right)$描述。这里$n$表示矩形的数量，${\mu }_{\text{area }}$是矩形面积的平均值，${\mathrm{{nv}}}_{\text{area }} = {\sigma }_{\text{area }}/{\mu }_{\text{area }}$是归一化方差，其中${\sigma }_{\text{area }}$表示矩形面积的方差。显然，参数${\mathrm{{nv}}}_{\text{area }}$独立于分布而增加，矩形面积与平均值的差异越大，平均重叠简单地由${n}^{ * }{\mu }_{\text{area }}$获得

(F1) "Uniform"

(F1) “均匀分布”

The centers of the rectangles follow a 2-dimensional independent uniform distribution

矩形的中心遵循二维独立均匀分布

$$
\left( {\mathrm{n} = {100},{000},{\mu }_{\text{area }} = {0001},{\mathrm{{nv}}}_{\text{area }} = {9505}}\right) 
$$

(F2) "Cluster"

(F2) “聚类分布”

The centers follow a distribution with 640 clusters, each cluster contains about 1600 objects

中心遵循具有640个聚类的分布，每个聚类包含约1600个对象

$\left( {\mathrm{n} = {99},{968},{\mu }_{\text{area }} = {00002},{\mathrm{{nv}}}_{\text{area }} = {1538}}\right)$

(F3) "Parcel"

(F3) “地块分布”

First we decompose the unit square into100,000 disjoint rectangles Then we expand the area of each rectangle by the factor 25

首先，我们将单位正方形分解为100,000个不相交的矩形。然后，我们将每个矩形的面积扩大25倍

$\left( {\mathrm{n} = {100},{000},{\mu }_{\text{area }} = {00002504},{\mathrm{{nv}}}_{\text{area }} = {303458}}\right)$

(F4) "Real-data"

(F4) “真实数据”

These rectangles are the minimum bounding rectangles of elevation lines from real cartography data

这些矩形是来自真实地图数据的等高线的最小边界矩形

$\left( {\mathrm{n} = {120},{576},{\mu }_{\text{area }} = {0000926},{\mathrm{{nv}}}_{\text{area }} = {1504}}\right)$

(F5) "Gaussian"

(F5) “高斯分布”

The centers follow a 2-dimensional independent Gaussian distribution

中心点遵循二维独立高斯分布

$\left( {\mathrm{n} = {100},{000},{\mu }_{\text{area }} = {00008},{\mathrm{{nv}}}_{\text{area }} = {89875}}\right)$

(F6) "Mixed-Uniform"

(F6) “混合均匀分布”

The centers of the rectangles follow a 2-dimensional independent uniform distribution

矩形的中心点遵循二维独立均匀分布

First we take 99,000 small rectangles with

首先，我们取99,000个小矩形，其[latex0] 

${\mu }_{\text{area }} = {0000101}$ Then we add 1,000 large rectangles with ${\mu }_{\text{area }} = {001}$ Finally these two data files are merged to one

${\mu }_{\text{area }} = {0000101}$ 然后我们添加1,000个大矩形，其${\mu }_{\text{area }} = {001}$ 最后将这两个数据文件合并为一个

$$
\left( {\mathrm{n} = {100},{000},{\mu }_{\text{area }} = {00002},{\mathrm{{nv}}}_{\text{area }} = {6778}}\right) 
$$

For each of the files (F1) - (F6) we generated queries of the following three types

对于每个文件(F1) - (F6)，我们生成了以下三种类型的查询

- rectangle intersection query Given a rectangle $\mathrm{S}$ ,find all rectangles $R$ in the file with $R \cap  S \neq  \varnothing$

- 矩形相交查询 给定一个矩形$\mathrm{S}$ ，找出文件中所有满足$R \cap  S \neq  \varnothing$ 的矩形$R$ 

- point query Given a point $P$ ,find all rectangles $R$ in the file with $P \in  R$

- 点查询 给定一个点$P$ ，找出文件中所有满足$P \in  R$ 的矩形$R$ 

- rectangle enclosure query Given a rectangle $\mathrm{S}$ ,find all rectangles $R$ in the file with $R \supseteq  S$

- 矩形包含查询 给定一个矩形$\mathrm{S}$ ，找出文件中所有满足$R \supseteq  S$ 的矩形$R$ 

For each of these files (F1) - (F6) we performed 400 rectangle intersection queries where the ratio of the $x$ - extension to the y-extension uniformly varies from 0 25 to 2 25 and the centers of the query rectangles themselves are uniformly distributed in the unit cube In the following, we consider four query files (Q1) - (Q4) of 100 rectangle intersection queries each The area of the query rectangles of each query file (Q1) - (Q4) varies from 1%, 0 1%, 0 01% to 0 001% relatively to the area of the data space For the rectangle enclosure query we consider two query files (Q5) and (Q6) where the corresponding rectangles are the same as in the query files (Q3) and (Q4), respectively Additionally, we analyzed a query file (Q7) of 1,000 point queries where the query points are uniformly distributed

对于这些文件(F1) - (F6)中的每一个，我们执行了400次矩形相交查询，其中$x$ 方向的扩展与y方向扩展的比率从0.25均匀变化到2.25，并且查询矩形的中心点本身在单位立方体中均匀分布。接下来，我们考虑四个查询文件(Q1) - (Q4)，每个文件包含100次矩形相交查询。每个查询文件(Q1) - (Q4)的查询矩形面积相对于数据空间面积分别为1%、0.1%、0.01%和0.001%。对于矩形包含查询，我们考虑两个查询文件(Q5)和(Q6)，其中相应的矩形分别与查询文件(Q3)和(Q4)中的矩形相同。此外，我们分析了一个包含1,000次点查询的查询文件(Q7)，其中查询点均匀分布

For each query file (Q1) - (Q7) we measured the average number of disc accesses per query In the performance comparison we use the R*-tree as a measuring stick for the other access methods, i e we standardize the number of page accesses for the queries of the R*-tree to ${100}\%$ Thus we can observe the performance of the R-tree variants relative to the ${100}\%$ performance of the ${\mathrm{R}}^{ * }$ -tree

对于每个查询文件(Q1) - (Q7)，我们测量了每次查询的平均磁盘访问次数。在性能比较中，我们使用R*树作为其他访问方法的衡量标准，即我们将R*树查询的页面访问次数标准化为${100}\%$ 。这样我们就可以观察到R树变体相对于${\mathrm{R}}^{ * }$ 树的${100}\%$ 性能的表现

To analyze the performance for building up the different R-tree variants we measured the parameters insert and stor Here insert denotes the average number of disc accesses per insertion and stor denotes the storage utilization after completely building up the files In the following table we present the results of our experiments depending on the different distributions (data files) For the R*-tree we also depict "# accesses", the average number of disk accesses per query Additionally to the conventional queries like point query, intersection query and enclosure query we have considered the operation spatial join usually used in applications like map overlay We have defined the spatial join over two rectangle files as the set of all pairs of rectangles where the one rectangle from file ${}_{1}$ intersects the other rectangle from ${\mathrm{{file}}}_{2}$

为了分析构建不同R树变体的性能，我们测量了参数insert和stor。这里insert表示每次插入的平均磁盘访问次数，stor表示文件完全构建后的存储利用率。在下表中，我们展示了根据不同分布(数据文件)进行实验的结果。对于R*树，我们还描绘了“#访问次数”，即每次查询的平均磁盘访问次数。除了像点查询、相交查询和包含查询这样的常规查询外，我们还考虑了通常用于地图叠加等应用程序的空间连接操作。我们将两个矩形文件上的空间连接定义为所有矩形对的集合，其中来自文件${}_{1}$ 的一个矩形与来自${\mathrm{{file}}}_{2}$ 的另一个矩形相交

<!-- Media -->

Uniform

均匀分布

<table><tr><td rowspan="7">In $\mathbf{{Gut}}$ qua. Gut Greene ${\mathrm{R}}^{ * }$ -tree #accesses</td><td rowspan="2">point</td><td colspan="4">intersection</td><td colspan="2" rowspan="2">enclosure 0.0010.01</td><td rowspan="2">stor</td><td rowspan="2">insert</td></tr><tr><td>10.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>225.8</td><td>2126</td><td>207 7</td><td>1830</td><td>144.5</td><td>224.7</td><td>248 1</td><td>64.1</td><td>743</td></tr><tr><td>124.8</td><td>121.9</td><td>124.4</td><td>124.1</td><td>114.2</td><td>1167</td><td>121.9</td><td>69 \$</td><td>4.27</td></tr><tr><td>140 0</td><td>1361</td><td>1354</td><td>1301</td><td>1151</td><td>132.8</td><td>153.8</td><td>70.3</td><td>4.67</td></tr><tr><td>100 0</td><td>100 \$</td><td>100 0</td><td>100 0</td><td>1999</td><td>100 *</td><td>100 *</td><td>75.8</td><td>4.42</td></tr><tr><td>5.26</td><td>6 04</td><td>7 63</td><td>13.29</td><td>5342</td><td>4.85</td><td>3 66</td><td colspan="2"/></tr></table>

<table><tbody><tr><td rowspan="7">在$\mathbf{{Gut}}$四叉树（qua. Gut Greene ${\mathrm{R}}^{ * }$ -tree）中 #访问次数</td><td rowspan="2">点</td><td colspan="4">交集</td><td colspan="2" rowspan="2">包围 0.001 0.01</td><td rowspan="2">存储</td><td rowspan="2">插入</td></tr><tr><td>10.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>225.8</td><td>2126</td><td>207 7</td><td>1830</td><td>144.5</td><td>224.7</td><td>248 1</td><td>64.1</td><td>743</td></tr><tr><td>124.8</td><td>121.9</td><td>124.4</td><td>124.1</td><td>114.2</td><td>1167</td><td>121.9</td><td>69 \$</td><td>4.27</td></tr><tr><td>140 0</td><td>1361</td><td>1354</td><td>1301</td><td>1151</td><td>132.8</td><td>153.8</td><td>70.3</td><td>4.67</td></tr><tr><td>100 0</td><td>100 \$</td><td>100 0</td><td>100 0</td><td>1999</td><td>100 *</td><td>100 *</td><td>75.8</td><td>4.42</td></tr><tr><td>5.26</td><td>6 04</td><td>7 63</td><td>13.29</td><td>5342</td><td>4.85</td><td>3 66</td><td colspan="2"></td></tr></tbody></table>

Cluster

簇（Cluster）

<table><tr><td rowspan="2">STUNKE</td><td rowspan="2">point</td><td colspan="4">intersection</td><td colspan="2" rowspan="2">enclosure 9.0010.01</td><td rowspan="2">stor</td><td rowspan="2">insert</td></tr><tr><td colspan="4">10.0010.010.11.0</td></tr><tr><td rowspan="5">lin Gut qua Gut Greene ${R}^{ * }$ -tree #accesses</td><td>250.9</td><td>231 ●</td><td>2197</td><td>176 6</td><td>136.9</td><td>247.8</td><td>249 4</td><td>617</td><td>6 13</td></tr><tr><td>166 1</td><td>152.7</td><td>160 7</td><td>1391</td><td>120.4</td><td>155.4</td><td>182.9</td><td>66.9</td><td>4.97</td></tr><tr><td>1599</td><td>151.8</td><td>152.2</td><td>144.3</td><td>116.9</td><td>151 6</td><td>153.2</td><td>69.2</td><td>4.32</td></tr><tr><td>100 0</td><td>100 0</td><td>100 \$</td><td>100 ●</td><td>100 \$</td><td>100 0</td><td>100 0</td><td>72.2</td><td>3 77</td></tr><tr><td>200</td><td>2.26</td><td>295</td><td>713</td><td>36 0</td><td>1.86</td><td>1 58</td><td colspan="2"/></tr></table>

<table><tbody><tr><td rowspan="2">斯滕克（STUNKE）</td><td rowspan="2">点</td><td colspan="4">交点；交集</td><td colspan="2" rowspan="2">围栏 9.0010.01</td><td rowspan="2">斯托尔（stor）</td><td rowspan="2">插入</td></tr><tr><td colspan="4">10.0010.010.11.0</td></tr><tr><td rowspan="5">林·古特（Lin Gut）作为古特·格林（Gut Greene） ${R}^{ * }$ -树 #访问次数</td><td>250.9</td><td>231 ●</td><td>2197</td><td>176 6</td><td>136.9</td><td>247.8</td><td>249 4</td><td>617</td><td>6 13</td></tr><tr><td>166 1</td><td>152.7</td><td>160 7</td><td>1391</td><td>120.4</td><td>155.4</td><td>182.9</td><td>66.9</td><td>4.97</td></tr><tr><td>1599</td><td>151.8</td><td>152.2</td><td>144.3</td><td>116.9</td><td>151 6</td><td>153.2</td><td>69.2</td><td>4.32</td></tr><tr><td>100 0</td><td>100 0</td><td>100 \$</td><td>100 ●</td><td>100 \$</td><td>100 0</td><td>100 0</td><td>72.2</td><td>3 77</td></tr><tr><td>200</td><td>2.26</td><td>295</td><td>713</td><td>36 0</td><td>1.86</td><td>1 58</td><td colspan="2"></td></tr></tbody></table>

Parcel

包裹

<table><tr><td rowspan="2">Parcel</td><td rowspan="2">point</td><td colspan="4">intersection</td><td colspan="2">enclosure</td><td rowspan="2">stor</td><td rowspan="2">insert</td></tr><tr><td>10 001</td><td>001</td><td>01</td><td>10</td><td>0 001</td><td>1001</td></tr><tr><td>In $\mathbf{{Gut}}$</td><td>264 1</td><td>265 0</td><td>258 6</td><td>214.3</td><td>1779</td><td>269 4</td><td>281.0</td><td>60.2</td><td>2307</td></tr><tr><td>qua Gut</td><td>129.5</td><td>132.3</td><td>129.9</td><td>126 1</td><td>1221</td><td>1310</td><td>125 6</td><td>67 0</td><td>13.30</td></tr><tr><td>Greene</td><td>199.8</td><td>196.2</td><td>2069</td><td>1841</td><td>156.5</td><td>195.8</td><td>207.5</td><td>68.9</td><td>16.02</td></tr><tr><td>${\mathrm{R}}^{ * }$ -tree</td><td>1000</td><td>100.0</td><td>1000</td><td>1000</td><td>1000</td><td>100 0</td><td>1000</td><td>72.5</td><td>1073</td></tr><tr><td>#accesses</td><td>5 67</td><td>6.26</td><td>7.36</td><td>13.29</td><td>3676</td><td>542</td><td>4.96</td><td colspan="2"/></tr></table>

<table><tbody><tr><td rowspan="2">包裹</td><td rowspan="2">点</td><td colspan="4">交点；交集</td><td colspan="2">围栏；围场；附件</td><td rowspan="2">存储（推测，原词可能有误，正确形式或许是“store”）</td><td rowspan="2">插入</td></tr><tr><td>10 001</td><td>001</td><td>01</td><td>10</td><td>0 001</td><td>1001</td></tr><tr><td>在 $\mathbf{{Gut}}$ 中</td><td>264 1</td><td>265 0</td><td>258 6</td><td>214.3</td><td>1779</td><td>269 4</td><td>281.0</td><td>60.2</td><td>2307</td></tr><tr><td>作为肠道（推测，“qua”有“作为”之意，“Gut”有“肠道”等意思，需结合具体语境）</td><td>129.5</td><td>132.3</td><td>129.9</td><td>126 1</td><td>1221</td><td>1310</td><td>125 6</td><td>67 0</td><td>13.30</td></tr><tr><td>格林（人名）</td><td>199.8</td><td>196.2</td><td>2069</td><td>1841</td><td>156.5</td><td>195.8</td><td>207.5</td><td>68.9</td><td>16.02</td></tr><tr><td>${\mathrm{R}}^{ * }$ -树</td><td>1000</td><td>100.0</td><td>1000</td><td>1000</td><td>1000</td><td>100 0</td><td>1000</td><td>72.5</td><td>1073</td></tr><tr><td>#访问次数</td><td>5 67</td><td>6.26</td><td>7.36</td><td>13.29</td><td>3676</td><td>542</td><td>4.96</td><td colspan="2"></td></tr></tbody></table>

Real Data

真实数据

<table><tr><td rowspan="2"/><td rowspan="2">point</td><td colspan="4">intersection</td><td colspan="2">enclosure</td><td rowspan="2">stor</td><td rowspan="2">insert</td></tr><tr><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td><td>0.001</td><td>10.01</td></tr><tr><td>lin Gut</td><td>245 6</td><td>246 7</td><td>220.8</td><td>1816</td><td>143.8</td><td>2681</td><td>284.1</td><td>62.9</td><td>7.30</td></tr><tr><td>qua Gut</td><td>147.3</td><td>1531</td><td>143.3</td><td>132.5</td><td>116 4</td><td>158.8</td><td>160 1</td><td>681</td><td>5 08</td></tr><tr><td>Greene</td><td>147.8</td><td>144 0</td><td>146.5</td><td>130.2</td><td>115.9</td><td>155 1</td><td>169.8</td><td>69 6</td><td>505</td></tr><tr><td>${R}^{ * }$ -tree</td><td>1000</td><td>1000</td><td>100.0</td><td>100 0</td><td>100.0</td><td>1000</td><td>100 0</td><td>70.5</td><td>4.22</td></tr><tr><td>#accesses</td><td>4 78</td><td>5.29</td><td>7.35</td><td>14 65</td><td>60.84</td><td>4.08</td><td>308</td><td colspan="2"/></tr></table>

<table><tbody><tr><td rowspan="2"></td><td rowspan="2">点</td><td colspan="4">交点；交集</td><td colspan="2">外壳；围合；封入物</td><td rowspan="2">存储（推测，原词可能有误，正确可能是“store”）</td><td rowspan="2">插入</td></tr><tr><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td><td>0.001</td><td>10.01</td></tr><tr><td>线性古特（“lin Gut”可能是特定术语，“Gut”可能是人名音译）</td><td>245 6</td><td>246 7</td><td>220.8</td><td>1816</td><td>143.8</td><td>2681</td><td>284.1</td><td>62.9</td><td>7.30</td></tr><tr><td>二次古特（“qua Gut”可能是特定术语，“Gut”可能是人名音译）</td><td>147.3</td><td>1531</td><td>143.3</td><td>132.5</td><td>116 4</td><td>158.8</td><td>160 1</td><td>681</td><td>5 08</td></tr><tr><td>格林（人名）</td><td>147.8</td><td>144 0</td><td>146.5</td><td>130.2</td><td>115.9</td><td>155 1</td><td>169.8</td><td>69 6</td><td>505</td></tr><tr><td>${R}^{ * }$ -树</td><td>1000</td><td>1000</td><td>100.0</td><td>100 0</td><td>100.0</td><td>1000</td><td>100 0</td><td>70.5</td><td>4.22</td></tr><tr><td>#访问次数</td><td>4 78</td><td>5.29</td><td>7.35</td><td>14 65</td><td>60.84</td><td>4.08</td><td>308</td><td colspan="2"></td></tr></tbody></table>

Gaussian

高斯（Gaussian）

<table><tr><td rowspan="2"/><td rowspan="2">point</td><td colspan="4">intersection</td><td colspan="2">enclosure</td><td rowspan="2">stor</td><td rowspan="2">insert</td></tr><tr><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td><td>0.001</td><td>10.01</td></tr><tr><td>lin Gut</td><td>1711</td><td>165 6</td><td>168 1</td><td>150 1</td><td>143.8</td><td>1711</td><td>180.2</td><td>63.8</td><td>19 12</td></tr><tr><td>qua Gut</td><td>116.2</td><td>1080</td><td>1160</td><td>117 6</td><td>119.2</td><td>106 4</td><td>106.8</td><td>68.8</td><td>14.0</td></tr><tr><td>Greene</td><td>123.2</td><td>1187</td><td>131.2</td><td>122.9</td><td>114.2</td><td>1207</td><td>130 6</td><td>69.9</td><td>11 41</td></tr><tr><td>${R}^{ * }$ -tree</td><td>100.0</td><td>100.0</td><td>100 0</td><td>100.0</td><td>1000</td><td>100.0</td><td>1000</td><td>73.8</td><td>9 15</td></tr><tr><td>#accesses</td><td>4.83</td><td>5.87</td><td>7 69</td><td>10.88</td><td>46 19</td><td>4.39</td><td>3.24</td><td colspan="2"/></tr></table>

<table><tbody><tr><td rowspan="2"></td><td rowspan="2">点</td><td colspan="4">交点；交集</td><td colspan="2">外壳；围合（enclosure）</td><td rowspan="2">存储（推测，stor可能拼写有误，原词可能是store）</td><td rowspan="2">插入</td></tr><tr><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td><td>0.001</td><td>10.01</td></tr><tr><td>线性古特（lin Gut，可能是特定术语，因信息不足只能直译）</td><td>1711</td><td>165 6</td><td>168 1</td><td>150 1</td><td>143.8</td><td>1711</td><td>180.2</td><td>63.8</td><td>19 12</td></tr><tr><td>二次古特（qua Gut，可能是特定术语，因信息不足只能直译）</td><td>116.2</td><td>1080</td><td>1160</td><td>117 6</td><td>119.2</td><td>106 4</td><td>106.8</td><td>68.8</td><td>14.0</td></tr><tr><td>格林（Greene，人名）</td><td>123.2</td><td>1187</td><td>131.2</td><td>122.9</td><td>114.2</td><td>1207</td><td>130 6</td><td>69.9</td><td>11 41</td></tr><tr><td>${R}^{ * }$ -树</td><td>100.0</td><td>100.0</td><td>100 0</td><td>100.0</td><td>1000</td><td>100.0</td><td>1000</td><td>73.8</td><td>9 15</td></tr><tr><td>#访问次数</td><td>4.83</td><td>5.87</td><td>7 69</td><td>10.88</td><td>46 19</td><td>4.39</td><td>3.24</td><td colspan="2"></td></tr></tbody></table>

Mixed Uniform

混合均匀分布（Mixed Uniform）

<table><tr><td rowspan="2"/><td rowspan="2">point</td><td colspan="4">intersection</td><td colspan="2" rowspan="2">enclosure 10.0010.01</td><td rowspan="2">stor</td><td rowspan="2">Insert</td></tr><tr><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>In $\mathbf{{Gut}}$</td><td>354.1</td><td>332.5</td><td>3117</td><td>233 1</td><td>165.9</td><td>3581</td><td>401 6</td><td>63.4</td><td>1270</td></tr><tr><td>qua Gut</td><td>127 6</td><td>126.3</td><td>122.7</td><td>1190</td><td>1130</td><td>119 6</td><td>124.7</td><td>682</td><td>494</td></tr><tr><td>Greene</td><td>1214</td><td>1167</td><td>1160</td><td>114.5</td><td>109.3</td><td>1140</td><td>116.3</td><td>70 1</td><td>4.58</td></tr><tr><td>${\mathbf{R}}^{ * }$ -tree</td><td>100.0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>731</td><td>4 46</td></tr><tr><td>#accesses</td><td>4.87</td><td>5.51</td><td>7.27</td><td>1376</td><td>52 06</td><td>4.44</td><td>3 69</td><td colspan="2"/></tr></table>

<table><tbody><tr><td rowspan="2"></td><td rowspan="2">点</td><td colspan="4">交点；交集</td><td colspan="2" rowspan="2">封闭区域 10.0010.01</td><td rowspan="2">存储（推测，原词可能有误，正确形式或许是“store”）</td><td rowspan="2">插入</td></tr><tr><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>在 $\mathbf{{Gut}}$</td><td>354.1</td><td>332.5</td><td>3117</td><td>233 1</td><td>165.9</td><td>3581</td><td>401 6</td><td>63.4</td><td>1270</td></tr><tr><td>由于古特（推测，“qua”有“由于”之意，“Gut”为人名）</td><td>127 6</td><td>126.3</td><td>122.7</td><td>1190</td><td>1130</td><td>119 6</td><td>124.7</td><td>682</td><td>494</td></tr><tr><td>格林（人名）</td><td>1214</td><td>1167</td><td>1160</td><td>114.5</td><td>109.3</td><td>1140</td><td>116.3</td><td>70 1</td><td>4.58</td></tr><tr><td>${\mathbf{R}}^{ * }$ -树</td><td>100.0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>100 0</td><td>731</td><td>4 46</td></tr><tr><td>#访问次数</td><td>4.87</td><td>5.51</td><td>7.27</td><td>1376</td><td>52 06</td><td>4.44</td><td>3 69</td><td colspan="2"></td></tr></tbody></table>

<!-- Media -->

For the spatial join operation we performed the following

对于空间连接操作，我们执行了以下操作

<!-- Media -->

<table><tr><td>experiments (SJ1) ${\mathrm{{fle}}}_{1}$</td><td>"Parcel"-distribution with 1000</td></tr><tr><td>${\mathrm{{flle}}}_{2}$</td><td>rectangles randomly selected from file (F3) data file (F4)</td></tr><tr><td>(SJ2) ${\mathrm{{fle}}}_{1}$</td><td>"Parcel"-distribution with 7500 rectangles</td></tr><tr><td>${\mathrm{f}}_{1}{\mathrm{{le}}}_{2}$</td><td>randomly selected from data file (F3) 7,536 rectangles generated from elevation lines $\left( {\mathrm{n} = 7,{536},{\mu }_{\text{area }} = {00148},{\mathrm{{nv}}}_{\text{area }} = {15}}\right)$</td></tr><tr><td>(SJ3) ${\mathrm{{fle}}}_{1}$</td><td>"Parcel"-distribution with 20,000 rectangles</td></tr><tr><td>${\mathrm{{file}}}_{2}$</td><td>randomly selected from data file (F3) ${\mathrm{{flle}}}_{1}$</td></tr></table>

<table><tbody><tr><td>实验 (SJ1) ${\mathrm{{fle}}}_{1}$</td><td>从1000个中进行“地块”分布</td></tr><tr><td>${\mathrm{{flle}}}_{2}$</td><td>从文件 (F3) 数据文件 (F4) 中随机选取的矩形</td></tr><tr><td>(SJ2) ${\mathrm{{fle}}}_{1}$</td><td>7500个矩形的“地块”分布</td></tr><tr><td>${\mathrm{f}}_{1}{\mathrm{{le}}}_{2}$</td><td>从数据文件 (F3) 中随机选取，由等高线生成的7536个矩形 $\left( {\mathrm{n} = 7,{536},{\mu }_{\text{area }} = {00148},{\mathrm{{nv}}}_{\text{area }} = {15}}\right)$</td></tr><tr><td>(SJ3) ${\mathrm{{fle}}}_{1}$</td><td>20000个矩形的“地块”分布</td></tr><tr><td>${\mathrm{{file}}}_{2}$</td><td>从数据文件 (F3) 中随机选取 ${\mathrm{{flle}}}_{1}$</td></tr></tbody></table>

<!-- Media -->

For these experiments we measured the number of disc accesses per operation The normalized results are presented in the following table

在这些实验中，我们测量了每次操作的磁盘访问次数。归一化后的结果如下表所示

<!-- Media -->

Spatial Join

空间连接

<table><tr><td rowspan="5">lin.Gut qua.Gut Greene R -tree</td><td>(SJ 1)</td><td>(SJ 2)</td><td>(SJ 3)</td></tr><tr><td>296 6</td><td>229.2</td><td>257 8</td></tr><tr><td>1424</td><td>1547</td><td>144 8</td></tr><tr><td>187.1</td><td>166.3</td><td>160 4</td></tr><tr><td>100.0</td><td>1000</td><td>100.0</td></tr></table>

<table><tbody><tr><td rowspan="5">林氏肠道（lin.Gut）、夸氏肠道（qua.Gut）格林R树</td><td>(SJ 1)</td><td>(SJ 2)</td><td>(SJ 3)</td></tr><tr><td>296 6</td><td>229.2</td><td>257 8</td></tr><tr><td>1424</td><td>1547</td><td>144 8</td></tr><tr><td>187.1</td><td>166.3</td><td>160 4</td></tr><tr><td>100.0</td><td>1000</td><td>100.0</td></tr></tbody></table>

<!-- Media -->

### 5.2 Interpretation of the Results

### 5.2 结果解读

In table 1 for the parameters stor and insert we computed the unweighted average over all six distributions (data files) The parameter spatial join denotes the average over the three spatial join operations (SJ1) - (SJ3) For the average query performance we present the parameter query average which is averaged over all seven query files for each distribution and then averaged over all six distributions The loss of information in the parameter query average is even less in table 2 where the parameter is displayed separately for each data file (F1) - (F6) as an average over all seven query files and in table 3 where the parameter query average is depicted separately for each query (Q1) - (Q7) as an average over all six data files

在表1中，对于参数stor和insert，我们计算了所有六种分布（数据文件）的未加权平均值。参数空间连接（spatial join）表示三种空间连接操作（SJ1） - （SJ3）的平均值。对于平均查询性能，我们给出了参数查询平均值（query average），该值是先对每种分布的所有七个查询文件求平均值，然后再对所有六种分布求平均值。在表2中，参数查询平均值的信息损失更小，该表中该参数针对每个数据文件（F1） - （F6）分别显示，作为所有七个查询文件的平均值；在表3中，参数查询平均值针对每个查询（Q1） - （Q7）分别展示，作为所有六个数据文件的平均值。

<!-- Media -->

<table><tr><td rowspan="5">lin Gut qua.Gut Greene RT-tree</td><td>query average</td><td>spatial Join</td><td>stor</td><td>insert</td></tr><tr><td>2275</td><td>261 2</td><td>627</td><td>1263</td></tr><tr><td>1300</td><td>1473</td><td>681</td><td>7.76</td></tr><tr><td>142 3</td><td>171 3</td><td>697</td><td>7 67</td></tr><tr><td>1000</td><td>1000</td><td>730</td><td>6 13</td></tr></table>

<table><tbody><tr><td rowspan="5">林·古特·夸。古特·格林 RT 树（lin Gut qua.Gut Greene RT-tree）</td><td>查询平均值</td><td>空间连接</td><td>存储</td><td>插入</td></tr><tr><td>2275</td><td>261 2</td><td>627</td><td>1263</td></tr><tr><td>1300</td><td>1473</td><td>681</td><td>7.76</td></tr><tr><td>142 3</td><td>171 3</td><td>697</td><td>7 67</td></tr><tr><td>1000</td><td>1000</td><td>730</td><td>6 13</td></tr></tbody></table>

Table 1 unweighted average over all distributions

表1 所有分布的未加权平均值

<table><tr><td rowspan="5">In Gut qua Gut Greene ${R}^{\prime }$ -tree</td><td>gaussian</td><td>cluster</td><td>mix uni</td><td>parcel</td><td>real data</td><td>uniform</td></tr><tr><td>164 3</td><td>2160</td><td>3081</td><td>247.2</td><td>227.2</td><td>206 6</td></tr><tr><td>1129</td><td>1539</td><td>1218</td><td>1281</td><td>144.5</td><td>121</td></tr><tr><td>1231</td><td>1471</td><td>1155</td><td>1924</td><td>144.2</td><td>1348</td></tr><tr><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr></table>

<table><tbody><tr><td rowspan="5">在古特（Gut）意义下的格林（Greene）${R}^{\prime }$ - 树</td><td>高斯（Gaussian）</td><td>聚类</td><td>混合均匀</td><td>包裹；地块</td><td>真实数据</td><td>均匀的</td></tr><tr><td>164 3</td><td>2160</td><td>3081</td><td>247.2</td><td>227.2</td><td>206 6</td></tr><tr><td>1129</td><td>1539</td><td>1218</td><td>1281</td><td>144.5</td><td>121</td></tr><tr><td>1231</td><td>1471</td><td>1155</td><td>1924</td><td>144.2</td><td>1348</td></tr><tr><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr></tbody></table>

Table 2 unweighted average over all seven types of queries depending on the distribution

表2 根据分布情况对所有七种查询类型的未加权平均值

<table><tr><td rowspan="5">lin. Gut qua. Gut Greene ${R}^{ * }$ -tree</td><td>point</td><td colspan="4">intersection 0.0010.010.11.0</td><td colspan="2">enclosure 0.001LO, 01</td><td>stor</td><td>insert</td></tr><tr><td>251.9</td><td>2A2.2</td><td>231 1</td><td>189.8</td><td>1521</td><td>256.5</td><td>2741</td><td>62.7</td><td>12.63</td></tr><tr><td>135.3</td><td>132.4</td><td>132.8</td><td>126.4</td><td>117 6</td><td>131.3</td><td>137.0</td><td>68.1</td><td>7.76</td></tr><tr><td>1487</td><td>143.9</td><td>1480</td><td>137.7</td><td>121.3</td><td>145.0</td><td>155.2</td><td>69</td><td>767</td></tr><tr><td>1000</td><td>100 0</td><td>100.0</td><td>100 0</td><td>100 0</td><td>1000</td><td>100.0</td><td>73.0</td><td>6.13</td></tr></table>

<table><tbody><tr><td rowspan="5">林（Lin）. 古特（Gut）. 夸（Qua）. 古特（Gut）. 格林（Greene） ${R}^{ * }$ -树</td><td>点</td><td colspan="4">交集 0.001 0.01 0.1 1.0</td><td colspan="2">封闭区域 0.001 LO, 01</td><td>存储</td><td>插入</td></tr><tr><td>251.9</td><td>2A2.2</td><td>231 1</td><td>189.8</td><td>1521</td><td>256.5</td><td>2741</td><td>62.7</td><td>12.63</td></tr><tr><td>135.3</td><td>132.4</td><td>132.8</td><td>126.4</td><td>117 6</td><td>131.3</td><td>137.0</td><td>68.1</td><td>7.76</td></tr><tr><td>1487</td><td>143.9</td><td>1480</td><td>137.7</td><td>121.3</td><td>145.0</td><td>155.2</td><td>69</td><td>767</td></tr><tr><td>1000</td><td>100 0</td><td>100.0</td><td>100 0</td><td>100 0</td><td>1000</td><td>100.0</td><td>73.0</td><td>6.13</td></tr></tbody></table>

Table 3 unweighted average over all six distributions depending on the query type

表3：根据查询类型对所有六种分布的未加权平均值

<!-- Media -->

First of all, the R*-tree clearly outperforms the R-tree variants in all experiments Moreover the most popular variant, the linear R-tree, performs essentially worse than all other R-trees The following remarks emphasize the superiority of the R*-tree in comparison to the R-trees

首先，在所有实验中，R*树（R*-tree）明显优于R树（R-tree）的各种变体。此外，最流行的变体——线性R树（linear R-tree）的性能实际上比所有其他R树都差。以下说明强调了R*树相对于R树的优越性。

- The ${R}^{ * }$ -tree is the most robust method which is underligned by the fact that for every query file and every data file less disk acesses are required than by any other variants To say it in other words, there is no experiment where the R*-tree is not the winner

- R*树（${R}^{ * }$ -tree）是最稳健的方法，这一点可以通过以下事实得到强调：对于每个查询文件和每个数据文件，它所需的磁盘访问次数都比任何其他变体少。换句话说，没有一个实验中R*树不是赢家。

- The gain in efficiency of the R*-tree for smaller query rectangles is higher than for larger query rectangles, because storage utilization gets more important for larger query rectangles This emphasizes the goodness of the order preservation of the R*-tree (i e rectangles close to each other are more likely stored together in one page)

- 对于较小的查询矩形，R*树的效率提升比对于较大的查询矩形更高，因为对于较大的查询矩形，存储利用率变得更加重要。这强调了R*树顺序保留的优势（即彼此靠近的矩形更有可能存储在同一页面中）。

- The maximum performance gain of the R*-tree taken over all query and data files is in comparison to the linear R-tree about ${400}\% (1\mathrm{e}$ it takes four times as long as the R*-tree 11), to Greene's R-tree about 200% and to the quadratic $\mathrm{R}$ -tree ${180}\%$

- 与线性R树相比，R*树在所有查询和数据文件上的最大性能提升约为${400}\% (1\mathrm{e}$，线性R树所需的时间是R*树的四倍（见图11）；与格林（Greene）的R树相比约为200%；与二次$\mathrm{R}$ -树相比为${180}\%$。

- As expected, the R*-tree has the best storage utilization

- 正如预期的那样，R*树具有最佳的存储利用率。

- Surprisingly in spite of using the concept of Forced Reinsert, the average insertion cost is not increased, but essentially decreased regarding the R-tree variants

- 令人惊讶的是，尽管使用了强制重新插入（Forced Reinsert）的概念，但与R树的各种变体相比，平均插入成本并未增加，实际上还降低了。

- The average performance gain for the spatial join operation is higher than for the other queries The quadratic R-tree, Greene's R-tree and the linear R-tree require ${147}\% ,{171}\%$ and ${261}\%$ of the disc accesses of the R*-tree, respectively, averaged over all spatial join operations

- 空间连接操作的平均性能提升比其他查询更高。二次R树、格林的R树和线性R树在所有空间连接操作上的平均磁盘访问次数分别是R*树的${147}\% ,{171}\%$和${261}\%$。

### 5.3 The R*-tree: an efficient point access method

### 5.3 R*树：一种高效的点访问方法

An important requirement for a spatial access method is to handle both spatial objects and point objects efficiently Points can be considered as degenerated rectangles and in most applications rectangles are very small relatively to the data space If a SAM is also an efficient PAM, this would underlign the robustness of the SAM Moreover, in many applications it is desirable to support additionally to the bounding rectangle of an object at least an atomar key with one access method

空间访问方法的一个重要要求是能够高效地处理空间对象和点对象。点可以被视为退化的矩形，并且在大多数应用中，矩形相对于数据空间非常小。如果一种空间访问方法（SAM）同时也是一种高效的点访问方法（PAM），这将强调该空间访问方法的稳健性。此外，在许多应用中，除了对象的边界矩形之外，还希望至少用一种访问方法支持一个原子键。

Therefore we ran the different $R$ -tree variants and our ${R}^{ * }$ - tree against a benchmark proposed and used for point access methods The reader interested in the details of this benchmark is referred to [KSSS 89] In this paper, let us mention that the benchmark incorporates seven data files of highly correlated 2-dimensional points Each data file contains about100,000records For each data file we considered five query files each of them containing 20 queries The first query files contain range queries specified by square shaped rectangles of size ${01}\% ,1\%$ and ${10}\%$ relatively to the data space The other two query files contain partial match queries where in the one only the $\mathrm{x}$ - value and in the other only the y-value is specified, respectively

因此，我们针对一个为点访问方法提出并使用的基准测试运行了不同的$R$ -树变体和我们的${R}^{ * }$ -树。对该基准测试细节感兴趣的读者可参考[KSSS 89]。在本文中，我们提及该基准测试包含七个高度相关的二维点的数据文件。每个数据文件包含约100,000条记录。对于每个数据文件，我们考虑了五个查询文件，每个查询文件包含20个查询。第一个查询文件包含由相对于数据空间大小为${01}\% ,1\%$和${10}\%$的方形矩形指定的范围查询。另外两个查询文件包含部分匹配查询，其中一个仅指定$\mathrm{x}$ -值，另一个仅指定y值。

Similar to the previous section, we measured the storage utilization (stor), the average insertion cost (insert) and the average query cost averaged over all query and data files The results are presented in table 4 where we included the 2-level grid file ([NHS84], [Hin85]), a very popular point access method

与上一节类似，我们测量了存储利用率（stor）、平均插入成本（insert）以及在所有查询和数据文件上的平均查询成本。结果显示在表4中，我们还纳入了二级网格文件（[NHS84]，[Hin85]），这是一种非常流行的点访问方法。

<!-- Media -->

<table><tr><td rowspan="6">lin.Gut qua.Gut Greene GRID R*-tree</td><td>query average</td><td>stor</td><td>insert</td></tr><tr><td>233.1</td><td>64.1</td><td>7 34</td></tr><tr><td>175.9</td><td>67.8</td><td>4.51</td></tr><tr><td>237.8</td><td>69.0</td><td>5.20</td></tr><tr><td>127.6</td><td>58.3</td><td>2.56</td></tr><tr><td>100 0</td><td>70.9</td><td>3.36</td></tr></table>

<table><tbody><tr><td rowspan="6">林氏肠道（lin.Gut）、夸氏肠道（qua.Gut）、格林网格R*树（Greene GRID R*-tree）</td><td>查询平均值</td><td>存储</td><td>插入</td></tr><tr><td>233.1</td><td>64.1</td><td>7 34</td></tr><tr><td>175.9</td><td>67.8</td><td>4.51</td></tr><tr><td>237.8</td><td>69.0</td><td>5.20</td></tr><tr><td>127.6</td><td>58.3</td><td>2.56</td></tr><tr><td>100 0</td><td>70.9</td><td>3.36</td></tr></tbody></table>

Table 4: unweighted average over all seven distributions

表4：所有七种分布的未加权平均值

<!-- Media -->

We were positively surprised by our results The performance gain of the R*-tree over the R-tree variants is considerably higher for points than for rectangles In particular Greene's R-tree is very inefficient for point data It requires even more accesses than the linear R-tree and 138% more than the R*-tree,whereas the quadratic R-tree requires ${75}\%$ more disc accesses than the R*-tree Nevertheless, we had expected that PAMs like the 2-level grid file would perform better than the R*-tree However in the over all average the 2-level grid file performs essentially worse than the R*-tree for point data An advantage of the grid file is the low average insertion cost In that sense it might be more suitable in an insertion-intensive application Let us mention that the complexity of the algorithms of the R*- trees is rather low in comparison to highly tuned PAMs

我们对实验结果感到惊喜。对于点数据，R*树相对于R树变体的性能提升明显高于矩形数据。特别是格林（Greene）的R树在处理点数据时效率极低。它甚至比线性R树需要更多的访问次数，比R*树多138%，而二次R树比R*树需要${75}\%$更多的磁盘访问次数。然而，我们原本期望像两级网格文件这样的分区访问方法（PAMs）能比R*树表现更好。但总体而言，对于点数据，两级网格文件的性能实际上比R*树差。网格文件的一个优点是平均插入成本低。从这个意义上说，它可能更适合插入密集型应用。需要指出的是，与高度优化的分区访问方法相比，R*树算法的复杂度相当低。

## 6 Conclusions

## 6 结论

The experimental comparison pointed out that the R*-tree proposed in this paper can efficiently be used as an access method in database systems organizing both, multidimensional points and spatial data As demonstrated in an extensive performance comparison with rectangle data, the R*-tree clearly outperforms Greene ’s R-tree, the quadratic R-tree and the popular linear R-tree in all experiments Moreover, for point data the gain in performance of the R*- tree over the other variants is increased Additionally, the R*-tree performs essentially better than the 2-level grid file for point data

实验对比表明，本文提出的R*树可以有效地用作数据库系统中组织多维点数据和空间数据的访问方法。在与矩形数据进行的广泛性能对比中，R*树在所有实验中明显优于格林（Greene）的R树、二次R树和流行的线性R树。此外，对于点数据，R*树相对于其他变体的性能提升更为显著。而且，对于点数据，R*树的性能实际上比两级网格文件更好。

The new concepts incorporated in the R*-tree are based on the reduction of the area, margin and overlap of the directory rectangles Since all three values are reduced, the R*-tree is very robust against ugly data distributions Furthermore, due to the fact of the concept of Forced Reinsert, splits can be prevented, the structure is reorganized dynamically and storage utilization is higher than for other R-tree variants The average insertion cost of the R*-tree is lower than for the well known R-trees Although the R*-tree outperforms its competitors, the cost for the implementation of the R*-tree is only slightly higher than for the other R-trees

R*树中引入的新概念基于减少目录矩形的面积、周长和重叠。由于这三个值都有所减少，R*树对不良数据分布具有很强的鲁棒性。此外，由于采用了强制重新插入的概念，可以避免分裂，动态重组结构，并且存储利用率比其他R树变体更高。R*树的平均插入成本低于著名的R树。虽然R*树的性能优于其竞争对手，但其实现成本仅比其他R树略高。

In our future work, the we will investigate whether the fan out can be increased by prefixes or by using the grid approximation as proposed in [SK 90] Moreover, we are generalizing the R*-tree to handle polygons efficiently

在未来的工作中，我们将研究是否可以通过前缀或如[SK 90]中提出的使用网格近似来增加扇出。此外，我们正在将R*树进行推广，以有效地处理多边形数据。

## References:

## 参考文献：

[Gre 89] D Greene 'An Implementation and Performance Analysis of Spatial Data Access Methods', Proc 5th I n t Conf on Data Engineering, 606-615, 1989

[Gre 89] D·格林（D Greene）《空间数据访问方法的实现与性能分析》，第五届国际数据工程会议论文集，606 - 615页，1989年

[Gut 84] A Guttman 'R-trees a dynamic index structure for spatial searching', Proc ACM SIGMOD Int

[Gut 84] A·古特曼（A Guttman）《R树：一种用于空间搜索的动态索引结构》，ACM SIGMOD国际

Conf on Management of Data, 47-57, 1984

数据管理会议论文集，47 - 57页，1984年

[Hin 85] K Hinrichs 'The grid file system implementation and case studies for applications', Dissertation No 7734, Eidgenössische Technische Hochschule (ETH), Zuerich, 1985

[Hin 85] K·欣里克斯（K Hinrichs）《网格文件系统的实现及应用案例研究》，博士论文第7734号，苏黎世联邦理工学院（Eidgenössische Technische Hochschule，ETH），1985年

[Knu 73] D Knuth 'The art of computer programming', Vol 3 sorting and searching, Addison-Wesley Publ Co , Reading, Mass, 1973

[KSSS 89] H P Kriegel, M Schiwietz, R Schneider, B Seeger 'Performance comparison of point and spatial access methods', Proc Symp on the Design and Implementation of Large Spatial Databases', Santa Barbara, 1989, Lecture Notes in Computer Science

[NHS 84] J Nievergelt, H Hinterberger, K C Sevcik The grid file an adaptable, symmetric multikey file structure', ACM Trans on Database Systems, Vol 9, 1, 38- 71, 1984

[RL 85] N Roussopoulos, D Leifker 'Direct spatial search on pictorial databases using packed R-trees', Proc ACM SIGMOD Int Conf on Managment of Data, 17-31, 1985

[SK 88] B Seeger, H P Kriegel 'Design and implementation of spatial access methods', Proc 14th Int Conf on Very Large Databases, 360-371, 1988

[SK 90] B Seeger, H P Kriegel 'The design and implementation of the buddy tree', Computer Science Technical Report 3/90, University of Bremen, submitted for publication, 1990