# Distance Browsing in Spatial Databases

# 空间数据库中的距离浏览

GÍSLI R. HJALTASON and HANAN SAMET

吉斯利·R·哈尔塔松（GÍSLI R. HJALTASON）和哈南·萨梅特（HANAN SAMET）

University of Maryland

马里兰大学

We compare two different techniques for browsing through a collection of spatial objects stored in an $\mathrm{R}$ -tree spatial data structure on the basis of their distances from an arbitrary spatial query object. The conventional approach is one that makes use of a $k$ -nearest neighbor algorithm where $k$ is known prior to the invocation of the algorithm. Thus if $m > k$ neighbors are needed,the $k$ -nearest neighbor algorithm has to be reinvoked for $m$ neighbors,thereby possibly performing some redundant computations. The second approach is incremental in the sense that having obtained the $k$ nearest neighbors,the $k + {1}^{st}$ neighbor can be obtained without having to calculate the $k + 1$ nearest neighbors from scratch. The incremental approach is useful when processing complex queries where one of the conditions involves spatial proximity (e.g., the nearest city to Chicago with population greater than a million), in which case a query engine can make use of a pipelined strategy. We present a general incremental nearest neighbor algorithm that is applicable to a large class of hierarchical spatial data structures. This algorithm is adapted to the R-tree and its performance is compared to an existing $k$ -nearest neighbor algorithm for R-trees [Roussopoulos et al. 1995]. Experiments show that the incremental nearest neighbor algorithm significantly outperforms the $k$ -nearest neighbor algorithm for distance browsing queries in a spatial database that uses the R-tree as a spatial index. Moreover, the incremental nearest neighbor algorithm usually outperforms the $k$ -nearest neighbor algorithm when applied to the $k$ -nearest neighbor problem for the R-tree, although the improvement is not nearly as large as for distance browsing queries. In fact, we prove informally that at any step in its execution the incremental nearest neighbor algorithm is optimal with respect to the spatial data structure that is employed. Furthermore, based on some simplifying assumptions, we prove that in two dimensions the number of distance computations and leaf nodes accesses made by the algorithm for finding $k$ neighbors is $O\left( {k + k}\right)$ .

我们基于空间对象与任意空间查询对象之间的距离，比较了两种不同的技术，用于浏览存储在$\mathrm{R}$树空间数据结构中的一组空间对象。传统方法是使用$k$最近邻算法，其中$k$在调用该算法之前是已知的。因此，如果需要$m > k$个邻居，则必须为$m$个邻居重新调用$k$最近邻算法，从而可能会进行一些冗余计算。第二种方法是增量式的，即获得$k$个最近邻后，可以在不必从头计算$k + 1$个最近邻的情况下获得第$k + {1}^{st}$个邻居。当处理复杂查询时，增量式方法很有用，其中一个条件涉及空间邻近性（例如，距离芝加哥最近且人口超过一百万的城市），在这种情况下，查询引擎可以采用流水线策略。我们提出了一种通用的增量式最近邻算法，该算法适用于一大类分层空间数据结构。该算法适用于R树，并将其性能与现有的R树$k$最近邻算法[Roussopoulos等人，1995年]进行了比较。实验表明，对于使用R树作为空间索引的空间数据库中的距离浏览查询，增量式最近邻算法的性能明显优于$k$最近邻算法。此外，当将增量式最近邻算法应用于R树的$k$最近邻问题时，它通常也优于$k$最近邻算法，尽管改进幅度远不如距离浏览查询那么大。实际上，我们非正式地证明了，在执行的任何步骤中，增量式最近邻算法相对于所采用的空间数据结构都是最优的。此外，基于一些简化假设，我们证明了在二维情况下，该算法为查找$k$个邻居所进行的距离计算和叶节点访问次数为$O\left( {k + k}\right)$。

Categories and Subject Descriptors: H.2.8 [Database Management]: Database applications; Spatial databases and GIS; E. 1 [Data]: Data Structures; Trees

分类和主题描述符：H.2.8 [数据库管理]：数据库应用；空间数据库和地理信息系统（GIS）；E. 1 [数据]：数据结构；树

General Terms: Algorithms, Performance

通用术语：算法、性能

Additional Key Words and Phrases: Distance browsing, hierarchical spatial data structures, nearest neighbors, R-trees, ranking

其他关键词和短语：距离浏览、分层空间数据结构、最近邻、R树、排序

---

<!-- Footnote -->

This work was supported in part by the National Science Foundation under grants IRI-9712715 and ASC-9318183, and the Department of Energy under contract DEFG0295ER25237.

这项工作部分得到了美国国家科学基金会（National Science Foundation）的资助（资助编号：IRI - 9712715和ASC - 9318183），以及美国能源部（Department of Energy）的合同支持（合同编号：DEFG0295ER25237）。

Authors' address: Computer Science Department, University of Maryland, College Park, MD 20742.

作者地址：马里兰大学计算机科学系，学院公园，马里兰州20742。

Permission to make digital/hard copy of part or all of this work for personal or classroom use is granted without fee provided that the copies are not made or distributed for profit or commercial advantage, the copyright notice, the title of the publication, and its date appear, and notice is given that copying is by permission of the ACM, Inc. To copy otherwise, to republish, to post on servers, or to redistribute to lists, requires prior specific permission and/or a fee.

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，必须保留版权声明、出版物标题和日期，并注明复制需获得美国计算机协会（ACM）的许可。否则，如需复制、重新发布、上传到服务器或分发给列表，需要事先获得特定许可和/或支付费用。

© 1999 ACM 0362-5915/99/0600-0265 \$5.00

© 1999美国计算机协会 0362 - 5915/99/0600 - 0265 5.00美元

<!-- Footnote -->

---

## 1. INTRODUCTION

## 1. 引言

In this paper we focus on the issue of obtaining data objects in their order of distance from a given query object (termed ranking). This issue is of primary interest in a spatial database, although it is also useful in other database applications, including multimedia indexing [Korn et al. 1996], CAD, and molecular biology [Kriegel et al. 1997]. The desired ranking may be full or partial (e.g.,only the first $k$ objects). This problem can also be posed in a conventional database system. For example, given a table of individuals containing a weight attribute, we can ask "who has a weight closest to $w$ lbs.?" or "rank the individuals by how much their weight differs from $w$ lbs." If no index exists on the weight attribute,a scan of all tuples must be performed to answer the first query. However, if an appropriate index structure is used, more efficient methods can be employed. For example,using a ${\mathrm{B}}^{ + }$ -tree,the query can be answered by a single descent to a leaf,for a cost of $O\left( {\log n}\right)$ for $n$ tuples. The correct answer is found either in that leaf or an adjacent one. To rank all the individuals, the search would proceed in two directions along the leaves of the ${\mathrm{B}}^{ + }$ -tree,with a constant cost for each tuple. The index can be used for any such query, regardless of the reference weight $w$ .

在本文中，我们关注的是按照数据对象与给定查询对象的距离顺序获取数据对象的问题（称为排序）。这个问题在空间数据库中是首要关注的问题，不过它在其他数据库应用中也很有用，包括多媒体索引 [Korn 等人，1996 年]、计算机辅助设计（CAD）和分子生物学 [Kriegel 等人，1997 年]。所需的排序可以是全排序或部分排序（例如，仅前 $k$ 个对象）。这个问题也可以在传统数据库系统中提出。例如，给定一个包含体重属性的人员表，我们可以问“谁的体重最接近 $w$ 磅？”或者“根据人员体重与 $w$ 磅的差值对他们进行排序”。如果体重属性上没有索引，必须扫描所有元组才能回答第一个查询。然而，如果使用合适的索引结构，就可以采用更高效的方法。例如，使用 ${\mathrm{B}}^{ + }$ -树，通过一次下降到叶子节点就可以回答查询，对于 $n$ 个元组，成本为 $O\left( {\log n}\right)$。正确答案可以在该叶子节点或相邻节点中找到。要对所有人员进行排序，搜索将沿着 ${\mathrm{B}}^{ + }$ -树的叶子节点向两个方向进行，每个元组的成本是固定的。无论参考体重 $w$ 是多少，该索引都可用于任何此类查询。

For multidimensional data, things are not so simple. Consider, for example, a set of points in two dimensions representing cities. Queries analogous to the previous ones are "what city is closest to point $p$ ?" and "rank the cities by their distances from point $p$ ." In a database context,we wish to know what kind of index structures will aid in processing these queries. For a fixed reference point $p$ and distance metric,we might build a one-dimensional index on the distances of the cities from the point $p$ . This would provide an efficient execution time for this particular point (i.e., for $p$ ),but it would be useless for any other point or distance metric. Thus,we have to rebuild the index, which is a costly process if we need to do it for each query. Contrast this to the one-dimensional case, where there is generally only one choice of metric. Furthermore, for a given reference point, any other point can have only two positions in relation to it, larger or smaller. It is not possible to define such a simple relationship in the multidimensional case.

对于多维数据，情况就没那么简单了。例如，考虑一组二维点来表示城市。与前面类似的查询有“哪个城市离点 $p$ 最近？”以及“根据城市与点 $p$ 的距离对城市进行排序”。在数据库环境中，我们想知道什么样的索引结构有助于处理这些查询。对于固定的参考点 $p$ 和距离度量，我们可以在城市到点 $p$ 的距离上构建一维索引。这将为这个特定点（即 $p$）提供高效的执行时间，但对于任何其他点或距离度量则毫无用处。因此，我们必须重建索引，如果每次查询都需要这样做，这是一个成本很高的过程。这与一维情况形成对比，在一维情况下通常只有一种度量选择。此外，对于给定的参考点，任何其他点相对于它只有两种位置关系，即更大或更小。在多维情况下，不可能定义这样简单的关系。

As another example, suppose we want to find the nearest city to Chicago with more than a million inhabitants. There are several ways to proceed. An intuitive solution is to guess some area range around Chicago and check the populations of the cities in that range. If we find a city with the requisite population, we must make sure that there are no other cities that are closer and that meet the population condition. This approach is rather inefficient, as we have to guess the size of the area to be searched. The problem with guessing is that we may choose too small a region or too large a region. If the size is too small, the area may not contain any cities satisfying the population criterion, in which case we need to expand the region being searched. If the size is too large, we may be examining many cities needlessly.

再举一个例子，假设我们想找到离芝加哥最近且人口超过一百万的城市。有几种方法可以进行。一种直观的解决方案是猜测芝加哥周围的某个区域范围，并检查该范围内城市的人口。如果我们找到一个符合人口要求的城市，我们必须确保没有其他更近且满足人口条件的城市。这种方法效率相当低，因为我们必须猜测要搜索的区域大小。猜测的问题在于我们可能选择的区域太小或太大。如果区域太小，该区域可能不包含任何满足人口标准的城市，在这种情况下，我们需要扩大搜索区域。如果区域太大，我们可能会不必要地检查许多城市。

A radical solution is to sort all the cities by their distances from Chicago. This is not very practical because we need to resort them each time we pose a similar query with respect to another city. Moreover, sorting requires a considerable amount of extra work, especially when usually all that is needed to obtain the desired result is to inspect the first few nearest neighbors.

一种极端的解决方案是根据城市与芝加哥的距离对所有城市进行排序。这不是很实际，因为每次我们针对另一个城市提出类似查询时，都需要重新排序。此外，排序需要大量的额外工作，尤其是当通常只需要检查前几个最近邻就能获得所需结果时。

A less radical solution is to retrieve the closest $k$ cities and determine if any of them satisfy the population criterion. The problem here lies in determining the value of $k$ . As in the area range solution,we may choose too small or too large a value of $k$ . If $k$ is too small,failure to find a city satisfying the population criterion means that we have to restart the search with a value larger than $k$ ,say $m$ . The drawback is that such a search forces us to expend work in finding the $k$ nearest neighbors (which we already did once before) as part of the cost of finding the $m > k$ nearest neighbors. On the other hand,if $k$ is too large,we waste work in calculating neighbors whose populations we will never check.

一种不那么极端的解决方案是检索最近的 $k$ 个城市，并确定其中是否有任何城市满足人口标准。这里的问题在于确定 $k$ 的值。与区域范围解决方案一样，我们可能选择的 $k$ 值太小或太大。如果 $k$ 太小，未能找到满足人口标准的城市意味着我们必须以大于 $k$ 的值（例如 $m$）重新开始搜索。缺点是这样的搜索迫使我们在寻找 $k$ 个最近邻（我们之前已经做过一次）上花费精力，这是寻找 $m > k$ 个最近邻成本的一部分。另一方面，如果 $k$ 太大，我们会在计算那些我们永远不会检查其人口的邻居上浪费精力。

A logical way to overcome the drawbacks of the second and third solutions is to obtain the neighbors incrementally (i.e., one-by-one) as they are needed. In essence, we are browsing through the database on the basis of distance, and we use the term distance browsing to describe this operation. The result is an incremental ranking of the cities by distance, where we cease the search as soon as the secondary population condition is satisfied. The idea is that we want only a small, but unknown, number of neighbors. The incremental solution finds application in a much more general setting than our specialized query example. In particular, this includes queries that require the application of the "nearest" predicate to a subset $s$ of the attributes of a relation (or object class) $r$ . This class of queries is part of a more restricted, but very common, class that imposes an additional condition $c$ ,which usually involves attributes other than $s$ . This means that the "nearest" condition serves as a primary condition, while condition $c$ serves as a secondary condition. Using an incremental solution enables such a query to be processed in a pipelined fashion.

克服第二种和第三种解决方案缺点的一种合理方法是在需要时逐步（即逐个）获取邻居。本质上，我们是基于距离在数据库中进行浏览，我们使用“距离浏览”这一术语来描述此操作。其结果是按距离对城市进行逐步排序，一旦满足次要的人口条件，我们就停止搜索。我们的想法是只需要数量未知但较少的邻居。逐步解决方案的应用场景比我们的特定查询示例要广泛得多。具体而言，这包括需要对关系（或对象类） $r$ 的属性子集 $s$ 应用“最近”谓词的查询。这类查询属于更受限但非常常见的一类，它会施加一个额外条件 $c$ ，该条件通常涉及除 $s$ 之外的属性。这意味着“最近”条件作为主要条件，而条件 $c$ 作为次要条件。使用逐步解决方案可以以流水线方式处理此类查询。

Of course, in the worst case, we have to examine all (or most) of the neighbors even when using an incremental approach. This may occur if few objects satisfy the secondary condition (e.g., if none of the cities have the requisite population). In this case, it may actually be better to first select on the basis of the secondary condition (the population criterion in our example) before considering the "spatially nearest" condition, especially if an index exists that can be used to compute the secondary condition. Using a $k$ -nearest neighbor algorithm may also be preferable,provided it is more efficient than the incremental algorithm for large values of $k$ . It only makes sense to choose this solution if we know in advance how many neighbors are needed (i.e.,the value of $k$ ),but this value can be estimated based on the selectivity of the secondary condition. These issues demonstrate the need for a query engine to make estimates using selectivity factors (e.g., Aref and Samet [1993]; Muralikrishna and DeWitt [1988]; Selinger et al. [1979]) involving the numbers of values that are expected to satisfy various parts of the query and the computational costs of the applicable algorithms.

当然，在最坏的情况下，即使使用逐步方法，我们也必须检查所有（或大部分）邻居。如果很少有对象满足次要条件（例如，如果没有一个城市具有所需的人口），就可能会出现这种情况。在这种情况下，实际上最好先根据次要条件（在我们的示例中是人口标准）进行选择，然后再考虑“空间最近”条件，特别是如果存在可用于计算次要条件的索引时。如果对于较大的 $k$ 值， $k$ -最近邻算法比逐步算法更高效，那么使用该算法可能也是更可取的。只有在我们事先知道需要多少个邻居（即 $k$ 的值）时，选择此解决方案才有意义，但这个值可以根据次要条件的选择性进行估计。这些问题表明查询引擎需要使用选择性因子进行估计（例如，Aref 和 Samet [1993]；Muralikrishna 和 DeWitt [1988]；Selinger 等人 [1979]），这些选择性因子涉及预计满足查询各个部分的数值数量以及适用算法的计算成本。

In this paper we compare the incremental and $k$ -nearest neighbor approaches for browsing through a collection of spatial objects stored in an R-tree spatial data structure on the basis of their distances from an arbitrary spatial query object. In the process, we present a general incremental nearest neighbor algorithm applicable to a large class of hierarchical spatial data structures, and show how to adapt this algorithm to the R-tree. Its performance is compared to an existing $k$ -nearest neighbor algorithm for R-trees [Roussopoulos et al. 1995]. In addition, we demonstrate that the $k$ -nearest neighbor algorithm of Roussopoulos et al. [1995] can be transformed into a special case of our R-tree adaptation of the general incremental nearest neighbor algorithm. The transformation process also reveals that the R-tree incremental nearest neighbor algorithm achieves more pruning than the R-tree $k$ -nearest neighbor algorithm. Moreover, our R-tree adaptation leads to a considerably more efficient (and conceptually different) algorithm because the presence of object bounding rectangles in the tree enables their use as pruning devices to reduce disk $\mathrm{I}/\mathrm{O}$ for accessing the spatial descriptions of objects (stored external to the tree). Experiments show that the incremental nearest neighbor algorithm significantly outperforms the $k$ -nearest neighbor algorithm for distance browsing queries in a spatial database that uses the R-tree as a spatial index. Moreover, the incremental nearest neighbor algorithm usually outperforms the $k$ -nearest neighbor algorithm when applied to the $k$ -nearest neighbor problem for the R-tree, although the improvement is not nearly as large as for distance browsing queries.

在本文中，我们比较了基于空间对象与任意空间查询对象的距离，对存储在 R 树空间数据结构中的一组空间对象进行浏览时的逐步方法和 $k$ -最近邻方法。在此过程中，我们提出了一种适用于一大类分层空间数据结构的通用逐步最近邻算法，并展示了如何将该算法应用于 R 树。我们将其性能与现有的 R 树 $k$ -最近邻算法 [Roussopoulos 等人 1995] 进行了比较。此外，我们证明了 Roussopoulos 等人 [1995] 的 $k$ -最近邻算法可以转化为我们对通用逐步最近邻算法进行 R 树适配后的一个特殊情况。转换过程还表明，R 树逐步最近邻算法比 R 树 $k$ -最近邻算法能实现更多的剪枝。此外，我们对 R 树的适配产生了一种效率更高（且概念上不同）的算法，因为树中存在的对象边界矩形可作为剪枝工具，以减少访问对象空间描述（存储在树外部）的磁盘 $\mathrm{I}/\mathrm{O}$ 操作。实验表明，在使用 R 树作为空间索引的空间数据库中，对于距离浏览查询，逐步最近邻算法的性能明显优于 $k$ -最近邻算法。此外，当将逐步最近邻算法应用于 R 树的 $k$ -最近邻问题时，它通常也优于 $k$ -最近邻算法，尽管改进幅度远不如距离浏览查询那么大。

The rest of this paper is organized as follows. Section 2 discusses algorithms related to nearest neighbor queries. Section 3 reviews the structure of R-trees. Section 4 describes the incremental nearest neighbor algorithm as well as its adaptation to the R-tree. Section 5 introduces the $k$ -nearest neighbor algorithm. Section 6 presents the results of an empirical study comparing the incremental nearest neighbor algorithm with the $k$ -nearest neighbor algorithm. Section 7 discusses issues that arise in high-dimensional spaces. Conclusions are drawn in Section 8.

本文的其余部分组织如下。第 2 节讨论与最近邻查询相关的算法。第 3 节回顾 R 树的结构。第 4 节描述逐步最近邻算法及其对 R 树的适配。第 5 节介绍 $k$ -最近邻算法。第 6 节展示了将逐步最近邻算法与 $k$ -最近邻算法进行比较的实证研究结果。第 7 节讨论在高维空间中出现的问题。第 8 节得出结论。

## 2. RELATED WORK

## 2. 相关工作

Numerous algorithms exist for answering nearest neighbor and $k$ -nearest neighbor queries, motivated by the importance of these queries in fields including geographical information systems (GIS), pattern recognition, document retrieval, and learning theory. Almost all of these algorithms, many of them from the field of computational geometry, are for points in a $d$ -dimensional vector space [Broder 1990; Eastman and Zemankova 1982; Friedman et al. 1977; Fukunaga and Narendra 1975; Kamgar-Parsi and Kanal 1985; Roussopoulos et al. 1995; Sproull 1991], but some allow for arbitrary spatial objects [Henrich 1994; Hoel and Samet 1991], although most are still limited to a point as the query object. In many applications, a rough answer suffices, so that algorithms have been developed that return an approximate result [Arya et al. 1994; Bern 1993; White and Jain 1996a], thereby saving time in computing it. Many of the above algorithms require specialized search structures [Arya et al. 1994; Bern 1993; Eastman and Zemankova 1982; Fukunaga and Narendra 1975; Kamgar-Parsi and Kanal 1985], but some employ commonly used spatial data structures. For example, algorithms exist for the k-d tree [Broder 1990; Friedman et al. 1977; Murphy and Selkow 1986; Sproull 1991], quadtree-related structures [Hjaltason and Samet 1995; Hoel and Samet 1991], the R-tree [Roussopou-los et al. 1995; White and Jain 1996a], the LSD-tree [Henrich 1994], and others. In addition, many of the algorithms can be applied to other spatial data structures.

存在众多用于回答最近邻和 $k$ -最近邻查询的算法，这些查询在地理信息系统（GIS）、模式识别、文档检索和学习理论等领域具有重要意义，从而推动了这些算法的发展。几乎所有这些算法（其中许多来自计算几何领域）都是针对 $d$ 维向量空间中的点的[布罗德（Broder）1990 年；伊斯特曼（Eastman）和泽曼科娃（Zemankova）1982 年；弗里德曼（Friedman）等人 1977 年；福永（Fukunaga）和纳伦德拉（Narendra）1975 年；卡姆加尔 - 帕西（Kamgar - Parsi）和卡纳尔（Kanal）1985 年；鲁索普洛斯（Roussopoulos）等人 1995 年；斯普劳尔（Sproull）1991 年]，但有些算法允许处理任意空间对象[亨里奇（Henrich）1994 年；霍尔（Hoel）和萨梅特（Samet）1991 年]，不过大多数算法仍将查询对象限制为点。在许多应用中，粗略的答案就足够了，因此已经开发出了返回近似结果的算法[阿亚（Arya）等人 1994 年；伯恩（Bern）1993 年；怀特（White）和贾因（Jain）1996a 年]，从而节省了计算时间。上述许多算法需要专门的搜索结构[阿亚（Arya）等人 1994 年；伯恩（Bern）1993 年；伊斯特曼（Eastman）和泽曼科娃（Zemankova）1982 年；福永（Fukunaga）和纳伦德拉（Narendra）1975 年；卡姆加尔 - 帕西（Kamgar - Parsi）和卡纳尔（Kanal）1985 年]，但有些算法采用常用的空间数据结构。例如，存在适用于 k - d 树的算法[布罗德（Broder）1990 年；弗里德曼（Friedman）等人 1977 年；墨菲（Murphy）和塞尔科夫（Selkow）1986 年；斯普劳尔（Sproull）1991 年]、与四叉树相关的结构[哈尔塔松（Hjaltason）和萨梅特（Samet）1995 年；霍尔（Hoel）和萨梅特（Samet）1991 年]、R - 树[鲁索普洛斯（Roussopoulos）等人 1995 年；怀特（White）和贾因（Jain）1996a 年]、LSD - 树[亨里奇（Henrich）1994 年]等。此外，许多算法可以应用于其他空间数据结构。

To our knowledge, only three incremental solutions to the nearest neighbor problem exist in the literature [Broder 1990; Henrich 1994; Hjaltason and Samet 1995]. All these algorithms employ priority queues (see Section 4). The Broder [1990] algorithm was developed for the k-d tree [Bentley 1975]. It is differs considerably from the other two algorithms, in that the Broder [1990] algorithm stores only the data objects in the priority queue, and uses a stack to keep track of the subtrees of the spatial data structure that have yet to be completely processed. This makes it necessary to use an elaborate mechanism to avoid processing the contents of a node more than once. The Henrich [1994] algorithm was developed for the LSD-tree [Hen-rich et al. 1989]. It is very similar to our method (in Hjaltason and Samet [1995]), and was published at about the same time. The principal difference between Henrich [1994] and our method is that the LSD-tree algorithm uses two priority queues, one for the data objects and another for the nodes of the spatial data structure. This makes their algorithm somewhat more complicated than ours, while, according to our experiments, the use of two priority queues does not offer any performance benefits. Our algorithm [Hjaltason and Samet 1995] was initially developed for the PMR quadtree [Nelson and Samet 1986] although its presentation was general. In this paper we expand considerably on our initial solution by showing how it can be adapted to the R-tree [Guttman 1984], as well as comparing it to a solution that makes use of an existing $k$ -nearest neighbor algorithm [Roussopoulos et al. 1995]. In addition,we show how this $k$ -nearest neighbor algorithm [Roussopoulos et al. 1995] can be transformed into a special case of our R-tree adaptation of the general incremental nearest neighbor algorithm. As a byproduct of the transformation process, the $k$ -nearest neighbor algorithm is considerably simplified.

据我们所知，文献中仅存在三种针对最近邻问题的增量式解决方案[布罗德（Broder）1990 年；亨里奇（Henrich）1994 年；哈尔塔松（Hjaltason）和萨梅特（Samet）1995 年]。所有这些算法都采用优先队列（见第 4 节）。布罗德（Broder）[1990 年]的算法是为 k - d 树[本特利（Bentley）1975 年]开发的。它与其他两种算法有很大不同，因为布罗德（Broder）[1990 年]的算法仅在优先队列中存储数据对象，并使用栈来跟踪尚未完全处理的空间数据结构的子树。这就需要使用一种复杂的机制来避免多次处理节点的内容。亨里奇（Henrich）[1994 年]的算法是为 LSD - 树[亨里奇（Henrich）等人 1989 年]开发的。它与我们的方法（见哈尔塔松（Hjaltason）和萨梅特（Samet）[1995 年]）非常相似，并且大约在同一时间发表。亨里奇（Henrich）[1994 年]的方法与我们的方法的主要区别在于，LSD - 树算法使用两个优先队列，一个用于数据对象，另一个用于空间数据结构的节点。这使得他们的算法比我们的算法稍微复杂一些，而根据我们的实验，使用两个优先队列并没有带来任何性能上的优势。我们的算法[哈尔塔松（Hjaltason）和萨梅特（Samet）1995 年]最初是为 PMR 四叉树[纳尔逊（Nelson）和萨梅特（Samet）1986 年]开发的，尽管其表述是通用的。在本文中，我们通过展示如何将其应用于 R - 树[古特曼（Guttman）1984 年]，并将其与一种使用现有 $k$ -最近邻算法[鲁索普洛斯（Roussopoulos）等人 1995 年]的解决方案进行比较，对我们最初的解决方案进行了大幅扩展。此外，我们还展示了如何将这种 $k$ -最近邻算法[鲁索普洛斯（Roussopoulos）等人 1995 年]转换为我们对通用增量最近邻算法进行 R - 树适配的一个特殊情况。作为转换过程的一个副产品， $k$ -最近邻算法得到了显著简化。

The term distance scan [Becker and Güting 1992] has also been used for what we term distance browsing. Becker and Güting [1992] introduce the concept of a distance scan and motivate its use, similarly to the procedure in Section 1; that is, in the context of finding the closest object to a query point where additional conditions may be imposed on the object. In addition, their paper provides optimization rules for mapping a "closest" operator into a "distance scan" operation in an example GIS query language.

“距离扫描”（distance scan）[贝克尔（Becker）和居廷（Güting）1992 年]这一术语也用于我们所说的“距离浏览”。贝克尔（Becker）和居廷（Güting）[1992 年]引入了距离扫描的概念并阐述了其用途，这与第 1 节中的过程类似；也就是说，在寻找与查询点最接近的对象的背景下，可能会对该对象施加额外的条件。此外，他们的论文为在一个示例地理信息系统查询语言中将“最近”运算符映射为“距离扫描”操作提供了优化规则。

All the algorithms mentioned thus far assume that the objects exist in a $d$ -dimensional Euclidean space,so that distances are defined between every two objects in a data set as well as between an object and any point in the space. Another class of nearest neighbor algorithms operates on more general objects, in what is commonly called the metric space model. The only restriction on the objects is that they reside in some metric space, i.e., a distance metric is defined between any two objects. However, in this general case, it is not possible to produce new objects in the metric space, e.g., to aggregate or divide two objects (in a Euclidean space, bounding rectangles are often used for this purpose). Various methods exist for indexing objects in the metric space model as well as for computing proximity queries [Brin 1995; Burkhard and Keller 1973; Ciaccia et al. 1997; Uhlmann 1991; Wang and Shasha 1990]. These methods can only make use of the properties of distance metrics (non-negativity, symmetry, and the triangle inequality), and operate without any knowledge of how objects are represented or how the distances between objects are computed. Such a general approach is usually slower than methods based on spatial properties of objects, but must be used for objects for which such properties do not exist (e.g., images, chemical data, time series, etc.). This approach has also been advocated for high-dimensional vector spaces. It may often be possible to map general objects into geometric space, thereby reaping the benefit of more efficient search methods. Most such mapping approaches are domain-specific [Hafner et al. 1995; Korn et al. 1996], but general approaches have also been proposed [Faloutsos and Lin 1995].

到目前为止提到的所有算法都假设对象存在于一个$d$维欧几里得空间中，这样就可以定义数据集中每两个对象之间以及对象与空间中任意点之间的距离。另一类最近邻算法适用于更一般的对象，这通常被称为度量空间模型。对这些对象的唯一限制是它们存在于某个度量空间中，即任意两个对象之间都定义了距离度量。然而，在这种一般情况下，无法在度量空间中生成新的对象，例如，无法聚合或分割两个对象（在欧几里得空间中，通常使用边界矩形来实现这一目的）。在度量空间模型中，存在各种用于对对象进行索引以及计算邻近查询的方法[布林1995年；伯克哈德和凯勒1973年；恰西亚等人1997年；乌尔尔曼1991年；王和沙沙1990年]。这些方法只能利用距离度量的性质（非负性、对称性和三角不等式），并且在不了解对象如何表示或对象之间的距离如何计算的情况下进行操作。这种通用方法通常比基于对象空间属性的方法慢，但必须用于那些不存在此类属性的对象（例如，图像、化学数据、时间序列等）。这种方法也被推荐用于高维向量空间。通常可以将一般对象映射到几何空间中，从而受益于更高效的搜索方法。大多数此类映射方法是特定领域的[哈夫纳等人1995年；科恩等人1996年]，但也有人提出了通用方法[法洛托斯和林1995年]。

## 3. R-TREES

## 3. R树

The R-tree (e.g., Figure 1) [Guttman 1984] is an object hierarchy in the form of a balanced structure inspired by the ${\mathrm{B}}^{ + }$ -tree [Comer 1979]. Each R-tree node contains an array of (key, pointer) entries where key is a hyper-rectangle that minimally bounds the data objects in the subtree pointed at by pointer. In an R-tree leaf node, the pointer is an object identifier (e.g., a tuple ID in a relational system), while in a nonleaf node it is a pointer to a child node on the next lower level. The maximum number of entries in each node is termed its node capacity or fan-out, and may be different for leaf and nonleaf nodes. The node capacity is usually chosen such that a node fills up one disk page (or a small number of them). It should be clear that the R-tree can be used to index a space of arbitrary dimension and arbitrary spatial objects rather than just points.

R树（例如，图1）[古特曼1984年]是一种对象层次结构，其形式为受${\mathrm{B}}^{ + }$树启发的平衡结构[科默1979年]。每个R树节点包含一个（键，指针）条目数组，其中键是一个超矩形，它最小限度地包围了指针所指向的子树中的数据对象。在R树的叶节点中，指针是一个对象标识符（例如，关系系统中的元组ID），而在非叶节点中，它是指向下一层子节点的指针。每个节点中的最大条目数称为其节点容量或扇出，叶节点和非叶节点的节点容量可能不同。通常选择节点容量，使得一个节点填满一个磁盘页面（或少量磁盘页面）。显然，R树可用于对任意维度和任意空间对象的空间进行索引，而不仅仅是点。

As described above, an R-tree leaf node contains a minimal bounding rectangle and an object identifier for each object in the node, i.e., geometric descriptions of the objects are stored external to the R-tree itself. Another possibility is to store the actual object, or its geometric description, in the leaf instead of its bounding rectangle. This is usually useful only if the object representation is relatively small (e.g., similar in size to a bounding rectangle) and is fixed in length. If all the data about the object (i.e., all its relevant attributes) are stored in the leaf nodes, the object identifiers need not be stored. The disadvantage of this approach is that objects will not have fixed addresses, as some objects must be moved each time an R-tree node is split.

如上所述，R树叶节点包含每个对象的最小边界矩形和对象标识符，即对象的几何描述存储在R树本身之外。另一种可能性是将实际对象或其几何描述存储在叶节点中，而不是其边界矩形。通常只有当对象表示相对较小（例如，大小与边界矩形相似）且长度固定时，这种方法才有用。如果关于对象的所有数据（即其所有相关属性）都存储在叶节点中，则无需存储对象标识符。这种方法的缺点是对象没有固定的地址，因为每次分割R树节点时，都必须移动一些对象。

<!-- Media -->

<!-- figureText: R1 R3 R0 R1 R2 R4 : d 9 R5 : C R6 : e f (b) h R6 R2 R4 R3 : a b (a) -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_6.jpg?x=228&y=223&w=1183&h=452&r=0"/>

Fig. 1. An R-tree index for a set of nine line segments. (a) Spatial rendering of the line segments and bounding rectangles; (b) a tree access structure for (a). In the interest of clarity, the bounding rectangles for the individual line segments are omitted from (a) .

图1. 一组九条线段的R树索引。(a) 线段和边界矩形的空间渲染；(b) (a)的树访问结构。为了清晰起见，(a)中省略了各个线段的边界矩形。

<!-- Media -->

Several variations of R-trees have been devised, differing in the way nodes are split or combined during insertion or deletion. In our experiments we make use of a variant called the ${\mathrm{R}}^{ * }$ -tree [Beckmann et al. 1990]. It differs from the conventional R-tree in employing more sophisticated insertion and node-splitting algorithms that attempt to minimize a combination of overlap between bounding rectangles and their total area. In addition,when ${\mathrm{R}}^{ * }$ -tree node $p$ overflows,instead of immediately splitting $p$ , the R-tree insertion algorithm first tries to see if some of the entries in $p$ could possibly fit better in another node. This is achieved by reinserting a fixed fraction of the entries in $p$ ,thus increasing the construction time for the index, but usually resulting in less node overlap, and therefore in improved query response time.

已经设计出了几种R树的变体，它们在插入或删除过程中节点的分割或合并方式上有所不同。在我们的实验中，我们使用了一种称为${\mathrm{R}}^{ * }$树的变体[贝克曼等人1990年]。它与传统R树的不同之处在于采用了更复杂的插入和节点分割算法，这些算法试图最小化边界矩形之间的重叠及其总面积的组合。此外，当${\mathrm{R}}^{ * }$树节点$p$溢出时，R树插入算法不会立即分割$p$，而是首先尝试查看$p$中的某些条目是否可能更适合放入另一个节点。这是通过重新插入$p$中固定比例的条目来实现的，从而增加了索引的构建时间，但通常会减少节点重叠，因此提高了查询响应时间。

## 4. INCREMENTAL NEAREST NEIGHBOR ALGORITHM

## 4. 增量最近邻算法

Most algorithms that traverse tree structures in a top-down manner use some form of depth-first or breadth-first tree traversal. Finding a leaf node containing a query object $q$ in a spatial index can be done in a depth-first manner by recursively descending the tree structure. With this method, the recursion stack keeps track of what nodes have yet to be visited. Having reached a leaf, we need to be able to extend this technique to find the nearest object, as the leaf may not actually contain the nearest neighbor. The problem here is that we have to unwind the recursion to find the nearest object. Moreover, if we want to find the second nearest object, the solution becomes even tougher. With breadth-first traversal, the nodes of the tree are visited level by level, and a queue is used to keep track of nodes that have yet to be visited. However, with this technique, a lot of work has to be done before reaching a leaf node containing $q$ . To resolve the problems with depth-first and breadth-first traversal, the incremental nearest neighbor algorithm employs what may be termed best-first traversal. When deciding what node to traverse next, it picks the node with the least distance in the set of all nodes that have yet to be visited. So that instead of using a stack or a plain queue to keep track of the nodes to be visited, we use a priority queue where the distance from the query object is used as a key. The key feature of our solution is that the objects as well as the nodes are stored in the priority queue.

大多数以自上而下的方式遍历树结构的算法都采用某种形式的深度优先或广度优先树遍历。在空间索引中查找包含查询对象 $q$ 的叶节点可以通过递归下降树结构以深度优先的方式完成。使用这种方法时，递归栈会记录哪些节点尚未被访问。到达叶节点后，我们需要能够扩展这种技术以找到最近的对象，因为该叶节点实际上可能并不包含最近邻。这里的问题是，我们必须展开递归才能找到最近的对象。此外，如果我们想找到第二近的对象，解决方案会变得更加棘手。采用广度优先遍历时，树的节点会逐层访问，并使用一个队列来记录尚未访问的节点。然而，使用这种技术，在到达包含 $q$ 的叶节点之前需要做大量的工作。为了解决深度优先和广度优先遍历的问题，增量最近邻算法采用了所谓的最佳优先遍历。在决定接下来遍历哪个节点时，它会从所有尚未访问的节点集合中选择距离最小的节点。因此，我们不使用栈或普通队列来记录要访问的节点，而是使用一个优先队列，其中将与查询对象的距离用作键。我们解决方案的关键特征是，对象和节点都存储在优先队列中。

This section is organized as follows: In Section 4.1 we specify what conditions must hold for our incremental nearest neighbor algorithm to be applicable (e.g., conditions on the index, spatial object types, distance functions, etc.). In Section 4.2 we present the general incremental nearest neighbor algorithm in detail. In Section 4.3 we discuss ways to exploit the particular nature of the R-tree spatial index, while in Section 4.4 we give an example of the execution of the algorithm on a simple R-tree structure. Several variants of the algorithm are described in Section 4.5. In Section 4.6 we present some analytical results for the algorithm, while in Section 4.7 we prove its correctness. Finally, in Section 4.8 we show how to deal with a large priority queue.

本节的组织如下：在 4.1 节中，我们指定了增量最近邻算法适用必须满足的条件（例如，对索引、空间对象类型、距离函数等的条件）。在 4.2 节中，我们详细介绍通用的增量最近邻算法。在 4.3 节中，我们讨论利用 R 树空间索引特性的方法，而在 4.4 节中，我们给出一个在简单 R 树结构上执行该算法的示例。4.5 节描述了该算法的几种变体。在 4.6 节中，我们给出该算法的一些分析结果，而在 4.7 节中，我们证明其正确性。最后，在 4.8 节中，我们展示如何处理大型优先队列。

### 4.1 Introduction

### 4.1 引言

Our incremental nearest neighbor algorithm can be applied to virtually any hierarchical spatial data structure. In fact, it is generally applicable to any data structure based on hierarchical containment/partitioning (e.g., see Aoki [1998]). In our description, we assume a tree structure (although our method is applicable to more general structures), where each tree node represents some regions of space and where objects (or pointers to them in an external table) are stored in the leaf nodes whose regions intersect the objects. In the remainder of this section, we do not make a distinction between a node and the region that it represents; the meaning should be clear from the context. A basic requirement for the method to be applicable is that the region covered by a node must be completely contained within the region(s) of the parent node(s). ${}^{1}$ Examples of structures that satisfy this requirement include quadtrees [Samet 1990], R-trees [Guttman 1984], ${\mathrm{R}}^{ + }$ -trees [Sellis et al. 1987],LSD-trees [Henrich et al. 1989],and k-d-B-trees [Robinson 1981]. In all these examples, the node region is rectangular, but this is not a requirement. Our algorithm handles the possibility of an object being represented in more than one leaf node, as in the PMR quadtree [Nelson and Samet 1986] and ${\mathrm{R}}^{ + }$ -tree [Sellis et al. 1987]. Although we assume in our exposition that each node has only one parent and that only leaf nodes store objects, the algorithm could easily be adapted to handle other cases (such as the hB-tree [Lomet and Salzberg 1989] and the cell tree with oversize shelves [Günther and Noltemeier 1991]).

我们的增量最近邻算法实际上可以应用于任何分层空间数据结构。事实上，它通常适用于任何基于分层包含/划分的数据结构（例如，参见 Aoki [1998]）。在我们的描述中，我们假设是树结构（尽管我们的方法适用于更通用的结构），其中每个树节点代表空间的某个区域，并且对象（或外部表中指向它们的指针）存储在其区域与对象相交的叶节点中。在本节的其余部分，我们不区分节点和它所代表的区域；其含义应从上下文中明确。该方法适用的一个基本要求是，节点所覆盖的区域必须完全包含在其父节点的区域内。 ${}^{1}$ 满足此要求的结构示例包括四叉树 [Samet 1990]、R 树 [Guttman 1984]、 ${\mathrm{R}}^{ + }$ 树 [Sellis 等人 1987]、LSD 树 [Henrich 等人 1989] 和 k-d-B 树 [Robinson 1981]。在所有这些示例中，节点区域是矩形的，但这不是必需的。我们的算法处理一个对象可能在多个叶节点中表示的情况，如 PMR 四叉树 [Nelson 和 Samet 1986] 和 ${\mathrm{R}}^{ + }$ 树 [Sellis 等人 1987]。尽管我们在阐述中假设每个节点只有一个父节点，并且只有叶节点存储对象，但该算法可以很容易地适应处理其他情况（如 hB 树 [Lomet 和 Salzberg 1989] 和带有超大货架的单元树 [Günther 和 Noltemeier 1991]）。

---

<!-- Footnote -->

${}^{1}$ For structures in which each node can have more than one parent (e.g.,the hB-tree [Lomet and Salzberg 1989] or Partition Fieldtree [Frank and Barrera 1989]), the node region must be fully contained in the union of the regions of the parent nodes.

${}^{1}$ 对于每个节点可以有多个父节点的结构（例如，hB 树 [Lomet 和 Salzberg 1989] 或分区场树 [Frank 和 Barrera 1989]），节点区域必须完全包含在父节点区域的并集中。

<!-- Footnote -->

---

Observe that the data objects as well as the query objects can be of arbitrary type (e.g., points, rectangles, polygons, etc.). The only requirement is that consistent distance functions ${d}_{o}$ and ${d}_{n}$ be used for calculating the distance from the query object $q$ to data objects and to nodes,to ensure that each object is encountered in at least one node that is no farther from the query object than the object itself; otherwise, the strictly nondecreasing distances of elements retrieved from the queue cannot be guaranteed. Consistency can be defined formally as follows: (In the definition, we do not make any assumptions about the nature of the index hierarchy.)

请注意，数据对象以及查询对象可以是任意类型（例如，点、矩形、多边形等）。唯一的要求是使用一致的距离函数 ${d}_{o}$ 和 ${d}_{n}$ 来计算查询对象 $q$ 到数据对象以及到节点的距离，以确保每个对象至少在一个节点中被访问到，且该节点到查询对象的距离不大于对象本身到查询对象的距离；否则，无法保证从队列中检索出的元素的距离严格非递减。一致性可以正式定义如下：（在定义中，我们不对索引层次结构的性质做任何假设。）

Definition 1. Let $d$ be the combination of functions ${d}_{o}$ and ${d}_{n}$ ,and let $e \sqsubseteq  N$ denote the fact that item $e$ is contained in exactly the set of nodes $N$ (i.e.,if $e$ is an object, $N$ is the set of leaf nodes referencing the object,and if $e$ is a node, $N$ is its set of parent nodes). ${}^{2}$ The functions ${d}_{o}$ and ${d}_{n}$ are consistent iff for any query object $q$ and any object or node $e$ in the hierarchical data structure there exists $n$ in $N$ ,where $e \sqsubseteq  N$ ,such that $d\left( {q,n}\right)  \leq  d\left( {q,e}\right) .$

定义 1。设 $d$ 是函数 ${d}_{o}$ 和 ${d}_{n}$ 的组合，并且设 $e \sqsubseteq  N$ 表示项 $e$ 恰好包含在节点集合 $N$ 中这一事实（即，如果 $e$ 是一个对象，$N$ 是引用该对象的叶节点集合；如果 $e$ 是一个节点，$N$ 是其父节点集合）。${}^{2}$ 函数 ${d}_{o}$ 和 ${d}_{n}$ 是一致的，当且仅当对于分层数据结构中的任何查询对象 $q$ 以及任何对象或节点 $e$，存在 $N$ 中的 $n$（其中 $e \sqsubseteq  N$ 成立），使得 $d\left( {q,n}\right)  \leq  d\left( {q,e}\right) .$

This definition is strictly tied to the hierarchy defined by the data structure. However, since this hierarchy is influenced by properties of the node regions and data objects, we can usually recast the definition in terms of these properties. For example, in spatial data structures the containment of objects in leaf nodes and child nodes in parent nodes is based on spatial containment; thus the $\sqsubseteq$ in the definition also denotes spatial containment. In other words, $e \sqsubseteq  N$ means that the union of the node regions for the nodes in $N$ completely encloses the region covered by the object or node $e$ . Informally,our definition of consistency means that if $p$ is the point in $e$ (or,more accurately,in the region that corresponds to it) closest to $q$ ,then $p$ must also be contained in the region covered by some node in $N$ . Note that since we assume spatial indexes that form a tree hierarchy (i.e., each nonroot node has exactly one parent), in the case of nodes the definition above simplifies to the following condition: if ${n}^{\prime }$ is a child node of node $n$ ,then ${d}_{n}\left( {q,n}\right)  \leq  {d}_{n}\left( {q,{n}^{\prime }}\right)$ .

这个定义与数据结构所定义的层次结构紧密相关。然而，由于这种层次结构受到节点区域和数据对象属性的影响，我们通常可以根据这些属性重新表述该定义。例如，在空间数据结构中，对象在叶节点中的包含关系以及子节点在父节点中的包含关系是基于空间包含的；因此，定义中的 $\sqsubseteq$ 也表示空间包含。换句话说，$e \sqsubseteq  N$ 意味着 $N$ 中节点的节点区域的并集完全包围了对象或节点 $e$ 所覆盖的区域。通俗地说，我们对一致性的定义意味着，如果 $p$ 是 $e$ 中（或者更准确地说，是与之对应的区域中）最接近 $q$ 的点，那么 $p$ 也必须包含在 $N$ 中某个节点所覆盖的区域内。请注意，由于我们假设空间索引形成树状层次结构（即，每个非根节点恰好有一个父节点），对于节点的情况，上述定义简化为以下条件：如果 ${n}^{\prime }$ 是节点 $n$ 的子节点，那么 ${d}_{n}\left( {q,n}\right)  \leq  {d}_{n}\left( {q,{n}^{\prime }}\right)$。

An easy way to ensure consistency is to base both functions on the same metric ${d}_{p}\left( {{p}_{1},{p}_{2}}\right)$ for points; common choices of metrics include the Euclidean,Manhattan,and Chessboard metrics. We then define $d\left( {q,e}\right)  \mathrel{\text{:=}}$ $\mathop{\min }\limits_{{{p}_{1} \in  q,{p}_{2} \in  e}}{d}_{p}\left( {{p}_{1},{p}_{2}}\right)$ ,where $e$ is either a spatial object or a node region. It is important to note that this is not the only way to define consistent distance functions. When $d$ is defined based on a metric ${d}_{p}$ ,its consistency is guaranteed by the properties of ${d}_{p}$ ,specifically,nonnegativity and the triangle inequality. The nonnegativity property states, among other things, that ${d}_{p}\left( {p,p}\right)  = 0$ ,and the triangle inequality states that ${d}_{p}\left( {{p}_{1},{p}_{3}}\right)  \leq$ ${d}_{p}\left( {{p}_{1},{p}_{2}}\right)  + {d}_{p}\left( {{p}_{2},{p}_{3}}\right)$ . Since $e$ is spatially contained in $N,e$ and $N$ have points in common, so their distance is zero. Thus, according to the triangle inequality, $d\left( {q,e}\right)  \leq  d\left( {q,N}\right)  + d\left( {N,e}\right)  = d\left( {q,N}\right)$ ,using a broad definition of $d$ (to allow $d\left( {N,e}\right)$ ,which equals 0 ). Note that if the distance functions are defined in this way, the distance from a query object to a node that intersects it is zero (i.e., it is not equal to the distance to the boundary of the node region).

确保一致性的一种简单方法是让两个函数都基于相同的点度量${d}_{p}\left( {{p}_{1},{p}_{2}}\right)$；常见的度量选择包括欧几里得（Euclidean）、曼哈顿（Manhattan）和棋盘（Chessboard）度量。然后我们定义$d\left( {q,e}\right)  \mathrel{\text{:=}}$ $\mathop{\min }\limits_{{{p}_{1} \in  q,{p}_{2} \in  e}}{d}_{p}\left( {{p}_{1},{p}_{2}}\right)$，其中$e$可以是一个空间对象或一个节点区域。需要注意的是，这并非定义一致距离函数的唯一方法。当$d$基于度量${d}_{p}$定义时，其一致性由${d}_{p}$的性质保证，具体来说，就是非负性和三角不等式。非负性性质表明，在其他条件中，${d}_{p}\left( {p,p}\right)  = 0$，而三角不等式表明${d}_{p}\left( {{p}_{1},{p}_{3}}\right)  \leq$ ${d}_{p}\left( {{p}_{1},{p}_{2}}\right)  + {d}_{p}\left( {{p}_{2},{p}_{3}}\right)$。由于$e$在空间上包含于$N,e$且$N$有共同的点，所以它们的距离为零。因此，根据三角不等式，$d\left( {q,e}\right)  \leq  d\left( {q,N}\right)  + d\left( {N,e}\right)  = d\left( {q,N}\right)$，这里使用了$d$的广义定义（允许$d\left( {N,e}\right)$，其等于0）。请注意，如果以这种方式定义距离函数，查询对象到与其相交的节点的距离为零（即，它不等于到节点区域边界的距离）。

---

<!-- Footnote -->

${}^{2}$ In most spatial data structures,each node has only one parent node; the hB-tree is an exception.

${}^{2}$ 在大多数空间数据结构中，每个节点只有一个父节点；hB树是个例外。

<!-- Footnote -->

---

The incremental nearest neighbor algorithm works in any number of dimensions, although the examples we give are restricted to two dimensions. Also, the query object need not be in the space of the dataset.

增量最近邻算法适用于任意维度，尽管我们给出的示例仅限于二维。此外，查询对象不必位于数据集的空间内。

### 4.2 Algorithm Description

### 4.2 算法描述

We first consider a regular recursive top-down traversal of the index to locate a leaf node containing the query object. Note that there may be more than one such node. The traversal is initiated with the root node of the spatial index (i.e., the node spanning the whole index space) as the second argument.

我们首先考虑对索引进行常规的递归自顶向下遍历，以定位包含查询对象的叶节点。请注意，可能存在多个这样的节点。遍历以空间索引的根节点（即，跨越整个索引空间的节点）作为第二个参数启动。

---

FINDLEAF(QueryObject, Node)

查找叶节点(查询对象, 节点)

if QueryObject is in node Node then

如果查询对象在节点中，则

	if Node is a leaf node then

	如果该节点是叶节点，则

		Report leaf node Node

		报告叶节点

	else

	否则

		for each Child of node Node do

		对该节点的每个子节点执行

			FINDLEAF(QueryObject, Child)

			查找叶节点(查询对象, 子节点)

		enddo

		结束循环

	endif

	结束条件判断

endif

结束条件判断

---

The first task is to extend the algorithm to find the object nearest to the query object. In particular, once a leaf node containing QueryObject has been found in line 3 , we could start by examining the objects contained in that node. However, the object closest to the query object might reside in another node. Finding that node may in fact require unwinding the recursion to the top and descending again, deeper into the tree. Furthermore, once that node has been found, it does not aid in finding the next nearest object.

第一项任务是扩展该算法以找到离查询对象最近的对象。具体而言，一旦在第3行找到包含查询对象的叶节点，我们可以从检查该节点中包含的对象开始。然而，离查询对象最近的对象可能位于另一个节点中。实际上，找到该节点可能需要将递归回溯到顶部，然后再次深入树中。此外，一旦找到该节点，它对找到下一个最近的对象并无帮助。

To resolve this dilemma, we replace the recursion stack of the regular top-down traversal with a priority queue. In addition to using the priority queue for nodes, objects are also put on the queue as leaf nodes are processed. The key used to order the elements on the queue is distance from the query object. In order to distinguish between two elements at equal distances from the query object, we adopt the convention that nodes are ordered before objects, while objects are ordered according to some arbitrary (but unique) rule. This secondary ordering makes it possible to avoid reporting an object more than once, which is necessary when using a disjoint decomposition, e.g., a PMR quadtree [Nelson and Samet 1986] or an ${\mathrm{R}}^{ + }$ -tree [Sellis et al. 1987],in which nonpoint objects may be associated with more than one node.

为解决这一困境，我们用优先队列取代了常规自顶向下遍历的递归栈。除了将节点放入优先队列，在处理叶节点时，对象也会被放入队列。用于对队列中元素排序的键是与查询对象的距离。为了区分与查询对象距离相等的两个元素，我们采用以下约定：节点排在对象之前，而对象则根据某种任意（但唯一）的规则排序。这种二次排序使得避免多次报告同一个对象成为可能，这在使用不相交分解时是必要的，例如，PMR四叉树（PMR quadtree）[Nelson和Samet 1986]或${\mathrm{R}}^{ + }$ -树[Sellis等人 1987]，在这些结构中，非点对象可能与多个节点相关联。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_10.jpg?x=559&y=220&w=514&h=512&r=0"/>

Fig. 2. The circle around query object $q$ depicts the search region after reporting $o$ as next nearest object. For simplicity, the leaf nodes are represented by a grid; in most spatial indexes, the shapes of the leaf nodes are more irregular than in a grid. Only the shaded leaf nodes are accessed by the incremental nearest neighbor algorithm. The region with darker shading is where we find the objects in the priority queue.

图2. 查询对象$q$周围的圆圈描绘了将$o$报告为下一个最近对象后的搜索区域。为简单起见，叶节点用网格表示；在大多数空间索引中，叶节点的形状比网格更不规则。增量最近邻算法仅访问阴影叶节点。颜色较深的阴影区域是我们在优先队列中找到对象的地方。

<!-- Media -->

A node is not examined until it reaches the head of the queue. At this time, all nodes and objects closer to the query object have been examined. Initially, the node spanning the whole index space is the sole element in the priority queue. At subsequent steps, the element at the head of the queue (i.e., the closest element not yet examined) is retrieved, and this is repeated until the queue has been emptied. Informally, we can visualize the progress of the algorithm for a query object $q$ as follows,when $q$ is a point (see Figure 2). We start by locating the leaf node(s) containing $q$ . Next,imagine a circle centered at $q$ being expanded from a starting radius of 0 ; we call this circle the search region. Each time the circle hits the boundary of a node region, the contents of that node are put on the queue, and each time the circle hits an object, we have found the object next nearest to $q$ . Note that when the circle hits a node or an object,we are guaranteed that the node or object is already in the priority queue, since the node that contains it must already have been hit (this is guaranteed by the consistency condition).

在节点到达队列头部之前不会对其进行检查。此时，所有比该节点更接近查询对象的节点和对象都已被检查过。最初，覆盖整个索引空间的节点是优先队列中的唯一元素。在后续步骤中，取出队列头部的元素（即尚未检查的最近元素），并重复此操作，直到队列为空。通俗地说，当$q$是一个点时，我们可以将查询对象$q$的算法执行过程可视化如下（见图2）。我们首先定位包含$q$的叶节点。接下来，想象一个以$q$为中心、从半径为0开始扩展的圆；我们称这个圆为搜索区域。每次圆触及节点区域的边界时，该节点的内容会被放入队列，每次圆碰到一个对象时，我们就找到了距离$q$下一个最近的对象。请注意，当圆碰到一个节点或一个对象时，我们可以保证该节点或对象已经在优先队列中，因为包含它的节点肯定已经被圆碰到过（这由一致性条件保证）。

Figure 3 presents the algorithm. Lines 1-2 initialize the queue. Notice that it is not really necessary to provide the correct distance when enqueue-ing the root node, since it will always be dequeued first. In line 9, the next closest object is reported. At that point, some other routine (such as a query engine) can take control, possibly resuming the algorithm at a later time to get the next closest object, or alternately terminating it if no more objects are desired.

图3展示了该算法。第1 - 2行初始化队列。请注意，在将根节点入队时，提供正确的距离并非真正必要，因为它总是会首先出队。在第9行，报告下一个最近的对象。此时，其他某个例程（如查询引擎）可以接管控制权，可能在稍后恢复算法以获取下一个最近的对象，或者如果不需要更多对象，则终止算法。

<!-- Media -->

---

INCNEAREST( Query Object, SpatialIndex)

增量最近邻算法（查询对象，空间索引）

			Queue $\leftarrow$ NEWPRIORITYQUEUE(   )

			 队列$\leftarrow$ 新建优先队列( )

			Enqueue(Queue, SpatialIndex.RootNode, 0)

			 入队(队列，空间索引.根节点，0)

		while not ISEMPTY (Queue) do

		 当队列不为空时执行以下操作

				Element $\leftarrow$ DEQUEUE(Queue)

				 元素$\leftarrow$ 出队(队列)

				if Element is a spatial object then

				 如果元素是一个空间对象

						while Element $=$ FIRST(Queue) do

						 当元素$=$ 是队列的第一个元素时执行以下操作

							DELETEFIRST (Queue)

							 删除队列的第一个元素

						enddo

						 结束循环

						Report Element

						 报告该元素

				elseif Element is a leaf node then

				 否则如果元素是一个叶节点

						for each Object in leaf node Element do

						 对于叶节点元素中的每个对象执行以下操作

							if DIST $\left( \text{Query Object,Object}\right)  \geq$ DIST(Query Object,Element)then

							如果距离 $\left( \text{Query Object,Object}\right)  \geq$ DIST(查询对象, 元素)，则

									Enqueue( Queue, Object, DIST( Query Object, Object) $)$

									将(队列, 对象, DIST(查询对象, 对象) $)$ 入队

							endif

							结束条件判断

						enddo

						结束循环

				else /* Element is a non-leaf node */

				否则 /* 元素是一个非叶子节点 */

						for each Child node of node Element in SpatialIndex do

						对于空间索引中元素节点的每个子节点，执行以下操作

							Enqueue( Queue, Child, DIST( QueryObject, Child))

							将(队列, 子节点, DIST(查询对象, 子节点))入队

						enddo

						结束循环

				endif

				结束条件判断

			enddo

			结束循环

						Fig. 3. Incremental nearest neighbor algorithm.

						图3. 增量最近邻算法。

---

<!-- Media -->

Recall that for some types of spatial indexes, a spatial object may span several nodes. In such a case, the algorithm must guard against objects being reported more than once [Aref and Samet 1992]. The test (i.e., the if statement) in line 12 ensures that objects that have already been reported are not put on the queue again. (Note that this test is not needed in the case when Element is a nonleaf node, as it holds implicitly by the assumption that child nodes are fully contained in their parent nodes.) For this to work properly, nodes must be retrieved from the queue before spatial objects at the same distance. Otherwise, an object may be retrieved from the queue before a node $n$ containing it,that is,at the same distance from the query object (this means that the object was contained in another node that was already dequeued). Then, when the object is encountered again in node $n$ ,there is no way of knowing that it was already reported. The loop in lines 6-8 eliminates duplicate instances of an object from the queue. By inducing an ordering on objects that are at the same distance from the query object, all of the instances of an object will be clustered at the front of the queue when the first instance reaches the front.

回顾一下，对于某些类型的空间索引，一个空间对象可能跨越多个节点。在这种情况下，算法必须防止对象被多次报告 [阿雷夫和萨梅特1992年]。第12行的测试（即if语句）确保已经报告过的对象不会再次被放入队列。（注意，当元素是非叶子节点时，不需要进行此测试，因为根据子节点完全包含在其父节点中的假设，这是隐含成立的。）为了使该算法正常工作，必须在距离相同的空间对象之前从队列中检索节点。否则，一个对象可能会在包含它的节点 $n$ 之前从队列中被检索出来，也就是说，该对象与查询对象的距离相同（这意味着该对象包含在另一个已经出队的节点中）。然后，当在节点 $n$ 中再次遇到该对象时，就无法知道它已经被报告过了。第6 - 8行的循环从队列中消除对象的重复实例。通过对与查询对象距离相同的对象进行排序，当一个对象的第一个实例到达队列前端时，该对象的所有实例将聚集在队列的前端。

We explicitly check for duplicates in this manner because for many priority queue implementations (e.g., binary heap), it is not efficient to detect duplicates among the queue elements, as these implementations only maintain a partial among the elements. A possible alternative is to use a priority queue implementation that maintains a total order among all the queue elements (e.g., a balanced binary tree), and thus is able to detect duplicates efficiently.

我们以这种方式显式检查重复项，因为对于许多优先队列实现（例如，二叉堆），检测队列元素中的重复项效率不高，因为这些实现只维护元素之间的部分顺序。一种可能的替代方法是使用一种优先队列实现，该实现维护所有队列元素之间的全序（例如，平衡二叉树），从而能够有效地检测重复项。

### 4.3 Adapting to R-trees

### 4.3 适配R树

In this section we demonstrate how to adapt the general incremental algorithm, presented above, to R-trees by exploiting some of their unique properties. If the spatial objects are stored external to the R-tree, such that leaf nodes contain only bounding rectangles for objects, then this adaptation leads to a considerably more efficient (and conceptually different) incremental algorithm. This enables the bounding rectangles to be used as pruning devices, thereby reducing the disk I/O needed to access the spatial descriptions of the objects. In addition, R-trees store each object just once, making it unnecessary to worry about reporting an object more than once. This also removes the need to enforce the secondary ordering on the priority queue used by the general algorithm (see Section 4.2).

在本节中，我们将展示如何利用R树的一些独特属性，将上述通用增量算法适配到R树上。如果空间对象存储在R树外部，使得叶子节点仅包含对象的边界矩形，那么这种适配会产生一个效率更高（并且概念上不同）的增量算法。这使得边界矩形可以用作剪枝工具，从而减少访问对象空间描述所需的磁盘I/O。此外，R树只存储每个对象一次，因此无需担心对象被多次报告。这也消除了对通用算法使用的优先队列强制执行二级排序的需要（见4.2节）。

The inputs to the R-tree incremental nearest neighbor algorithm are a query object $q$ and an R-tree $R$ containing a set of spatial data objects. As with the general incremental nearest neighbor algorithm, the data objects as well as the query object may be of any dimension and of arbitrary type (e.g., points, rectangles, polygons, etc.), as long as consistent distance functions are used for calculating the distance from $q$ to data objects and bounding rectangles. In the case of an $\mathrm{R}$ -tree,this means that if $e$ is a data object or a rectangle completely contained in rectangle $r$ ,then $d\left( {q,r}\right)  \leq$ $d\left( {q,e}\right)$ .

R树增量最近邻算法的输入是一个查询对象 $q$ 和一个包含一组空间数据对象的R树 $R$。与通用增量最近邻算法一样，数据对象和查询对象可以是任意维度和任意类型（例如，点、矩形、多边形等），只要使用一致的距离函数来计算从 $q$ 到数据对象和边界矩形的距离。对于 $\mathrm{R}$ 树，这意味着如果 $e$ 是一个数据对象或完全包含在矩形 $r$ 中的矩形，那么 $d\left( {q,r}\right)  \leq$ $d\left( {q,e}\right)$。

The general algorithm can be used virtually unchanged if object geometry is stored in the R-tree leaf nodes, the only changes being the ones already described. If the spatial objects are stored external to the R-tree, the primary difference from the general algorithm is in the use of the bounding rectangles stored in the leaf nodes. To exploit that information, a third type of queue element is introduced: an object bounding rectangle. The distance of an object bounding rectangle is never greater than the distance of the object, provided the distance functions used are consistent. Informally, the modifications to the algorithm are as follows: When an R-tree leaf is being processed in the main loop of the algorithm, instead of computing the real distances of the objects, the distances of their bounding boxes are computed and inserted into the queue. Only when an object's bounding box is retrieved from the queue is the actual distance computed. If the object is closer to the query object than the next element on the priority queue, it can be reported as the next nearest neighbor. Otherwise, the object is inserted into the queue with its real distance.

如果对象的几何信息存储在R树（R-tree）的叶子节点中，那么通用算法几乎可以原封不动地使用，唯一的改动就是前面已经描述过的那些。如果空间对象存储在R树之外，与通用算法的主要区别在于对存储在叶子节点中的边界矩形的使用。为了利用这些信息，引入了第三种类型的队列元素：对象边界矩形。只要所使用的距离函数是一致的，对象边界矩形的距离就绝不会大于对象本身的距离。简单来说，对该算法的修改如下：在算法的主循环中处理R树叶子节点时，不计算对象的实际距离，而是计算它们边界框的距离并将其插入队列。只有当从队列中取出对象的边界框时，才计算其实际距离。如果该对象比优先队列中的下一个元素更接近查询对象，则可以将其报告为下一个最近邻。否则，将该对象以其实际距离插入队列。

Figure 4 shows our algorithm. In lines 1-2, the queue is initialized. In line 9,the next closest object is reported. In line 7,an object $p$ is enqueued with its real distance as the key after it has been determined that there are elements on the queue with a key less than the real distance from $p$ to the query object $q$ . If there are no such elements, $p$ is reported as the next nearest object. Line 13 enqueues an object bounding rectangle; brackets around Object signal that it is not the object itself but is instead the bounding rectangle along with a pointer to the corresponding object. The general incremental nearest neighbor algorithm had an extra test at this point to guard against reporting duplicates, but that is not needed here.

图4展示了我们的算法。在第1 - 2行，对队列进行初始化。在第9行，报告下一个最近的对象。在第7行，当确定队列中存在键值小于对象$p$到查询对象$q$的实际距离的元素后，将对象$p$以其实际距离作为键值加入队列。如果不存在这样的元素，则将$p$报告为下一个最近的对象。第13行将一个对象边界矩形加入队列；“Object”周围的方括号表示这不是对象本身，而是边界矩形以及指向相应对象的指针。通用的增量最近邻算法在这一点上有一个额外的测试，以防止重复报告，但这里不需要。

<!-- Media -->

---

				INCNEAREST( Query Object, R-tree)

				增量最近邻算法(查询对象, R树)

								Queue $\leftarrow$ NEWPRIORITYQUEUE(   )

								队列 $\leftarrow$ 新建优先队列(   )

								Enqueue(Queue,R-tree.RootNode,0)

								入队(队列, R树的根节点, 0)

								while not ISEMPTY( Queue) do

								当队列不为空时执行

										Element $\leftarrow$ DEQUEUE(Queue)

										元素 $\leftarrow$ 出队(队列)

									if Element is an object or its bounding rectangle then

									如果元素是一个对象或其边界矩形

												if Element is the bounding rectangle of Object and not ISEMPTY (Queue)

												如果元素（Element）是对象（Object）的边界矩形，并且队列（Queue）不为空

														and DIST( Query Object, Object) > FIRST( Queue ). Key then

														并且查询对象（Query Object）与对象（Object）之间的距离（DIST）大于队列（Queue）的第一个元素的键值，则

														Enqueue (Queue, Object, DIST( Query Object, Object) $)$

														将对象（Object）及其与查询对象（Query Object）的距离（DIST）$)$入队到队列（Queue）中

												else

												否则

														Report Element (or if bounding rectangle, the associated object)

														将元素（Element）（或者如果是边界矩形，则是关联的对象）报告为

														as the next nearest object

														下一个最近的对象

												endif

												结束条件判断

										elseif Element is a leaf node then

										否则，如果元素（Element）是叶节点，则

												for each entry (Object, Rect) in leaf node Element do

												对于叶节点元素（Element）中的每个条目（对象（Object），矩形（Rect））执行以下操作

													Enqueue( Queue, [Object], DIST( QueryObject, Rect) )

													将对象（Object）及其与查询对象（QueryObject）的矩形（Rect）的距离（DIST）入队到队列（Queue）中

												enddo

												结束循环

										else /* Element is a non-leaf node */

										否则 /* 元素（Element）是非叶节点 */

												for each entry (Node, Rect) in node Element do

												对于节点元素（Element）中的每个条目（节点（Node），矩形（Rect））执行以下操作

														Enqueue(Queue, Node, Dist( QueryObject, Rect) $)$

														将节点（Node）及其与查询对象（QueryObject）的矩形（Rect）的距离（Dist）$)$入队到队列（Queue）中

												enddo

												结束循环

										endif

										结束条件判断

							enddo

							结束循环

Fig. 4. Incremental nearest neighbor algorithm for an R-tree where spatial objects are stored

图4. 用于存储空间对象的R树的增量最近邻算法

---

external to the R-tree.

存储在R树外部。

<!-- Media -->

The R-tree variant given above can be used for any spatial data structure method that separates the storage of bounding rectangles and the actual geometric descriptions of objects. For complex objects, for example polygons, one can even conceive of several levels of refinement, e.g., with the use of orthogonal polygons [Esperança and Samet 1997].

上述R树变体可用于任何将边界矩形存储与对象实际几何描述分开的空间数据结构方法。对于复杂对象，例如多边形，甚至可以设想进行多级细化，例如，使用正交多边形[Esperança和萨梅特（Samet）1997年]。

### 4.4 Example

### 4.4 示例

As an example, suppose that we want to find the three nearest neighbors to query point $q$ in the R-tree given in Figure 1,where the spatial objects are line segments stored external to the R-tree. Below, we show the steps of the algorithm and the contents of the priority queue. The algorithm must compute the distances between $\mathrm{q}$ and the line segments and bounding rectangles. These distances are given in Table I ( ${BR}$ means bounding rectangle). They are based on an arbitrary coordinate system and are approximate. When depicting the contents of the priority queue, the line segments and bounding rectangles are listed with their distances, in increasing order of distance, with ties broken using alphabetical ordering. Bounding rectangles of objects are denoted by the corresponding object

作为一个示例，假设我们要在图1所示的R树中找到查询点$q$的三个最近邻，其中空间对象是存储在R树外部的线段。下面，我们展示算法的步骤和优先队列的内容。该算法必须计算$\mathrm{q}$与线段和边界矩形之间的距离。这些距离在表I中给出（${BR}$表示边界矩形）。它们基于任意坐标系，并且是近似值。在描述优先队列的内容时，线段和边界矩形按距离从小到大列出，距离相同时按字母顺序排序。对象的边界矩形用相应的对象表示

<!-- Media -->

Table 1. Distances of Line segments and Bounding Rectangles from the Query Point $q$ in the R-Tree of Figure 1.

表1. 图1的R树中线段和边界矩形到查询点$q$的距离。

<table><tr><td>Seg.</td><td>Dist.</td><td>BRDist.</td></tr><tr><td>a</td><td>17</td><td>13</td></tr><tr><td>b</td><td>48</td><td>27</td></tr><tr><td>C</td><td>57</td><td>53</td></tr><tr><td>d</td><td>59</td><td>30</td></tr><tr><td>e</td><td>48</td><td>45</td></tr><tr><td>f</td><td>86</td><td>74</td></tr><tr><td>9</td><td>81</td><td>74</td></tr><tr><td>h</td><td>17</td><td>17</td></tr><tr><td>i</td><td>21</td><td>0</td></tr></table>

<table><tbody><tr><td>段（Seg.）</td><td>距离（Dist.）</td><td>分支距离（BRDist.）</td></tr><tr><td>a</td><td>17</td><td>13</td></tr><tr><td>b</td><td>48</td><td>27</td></tr><tr><td>C</td><td>57</td><td>53</td></tr><tr><td>d</td><td>59</td><td>30</td></tr><tr><td>e</td><td>48</td><td>45</td></tr><tr><td>f</td><td>86</td><td>74</td></tr><tr><td>9</td><td>81</td><td>74</td></tr><tr><td>h</td><td>17</td><td>17</td></tr><tr><td>i</td><td>21</td><td>0</td></tr></tbody></table>

<table><tr><td>BR</td><td>Dist.</td></tr><tr><td>R0</td><td>0</td></tr><tr><td>R1</td><td>0</td></tr><tr><td>R2</td><td>0</td></tr><tr><td>R3</td><td>13</td></tr><tr><td>R4</td><td>11</td></tr><tr><td>R5</td><td>0</td></tr><tr><td>R6</td><td>44</td></tr></table>

<table><tbody><tr><td>巴西（Brazil）</td><td>距离（Distance）</td></tr><tr><td>R0</td><td>0</td></tr><tr><td>R1</td><td>0</td></tr><tr><td>R2</td><td>0</td></tr><tr><td>R3</td><td>13</td></tr><tr><td>R4</td><td>11</td></tr><tr><td>R5</td><td>0</td></tr><tr><td>R6</td><td>44</td></tr></tbody></table>

<!-- Media -->

names embedded in brackets (e.g., $\left\lbrack  \mathrm{\;h}\right\rbrack$ ). The algorithm starts by enqueue-

嵌入括号中的名称（例如，$\left\lbrack  \mathrm{\;h}\right\rbrack$ ）。该算法首先进行入队操作

ing R0, after which it executes the following steps:

(1) Dequeue R0,enqueue R1 and R2. Queue: $\{ \left( {{R1},0}\right) ,\left( {{R2},0}\right) \}$ .

(2) Dequeue R1, enqueue R3 and R4. Queue: \{(R2, 0), (R4, 11), (R3, 13)\}.

(3) Dequeue R2, enqueue R5 and R6. Queue: \{(R5, 0), (R4, 11), (R3, 13), $\left( {\mathrm{R}6,{44}}\right) \}$ .

(4) Dequeue R5, enqueue [c] and [i] (i.e., the bounding rectangles of c and i). Queue: $\{ \left( {\left\lbrack  i\right\rbrack  ,0}\right) ,\left( {\mathrm{R}4,{11}}\right) ,\left( {\mathrm{R}3,{13}}\right) ,\left( {\mathrm{R}6,{44}}\right) ,\left( {\left\lbrack  \mathrm{c}\right\rbrack  ,{53}}\right) \}$ .

(5) Dequeue [ i ]. The distance of i is 21 , which is larger than the distance of R4, so enqueue i. Queue: \{(R4, 11), (R3, 13), (i, 21), (R6, 44), $\left( {\left\lbrack  c\right\rbrack  ,{53}}\right) \}$ .

(6) Dequeue R4, and enqueue [d], [g], and [h]. Queue: \{(R3, 13), $\left( {\left\lbrack  \mathrm{\;h}\right\rbrack  ,{17}}\right) ,\left( {\mathrm{i},{21}}\right) ,\left( {\left\lbrack  \mathrm{d}\right\rbrack  ,{30}}\right) ,\left( {\mathrm{R}6,{44}}\right) ,\left( {\left\lbrack  \mathrm{c}\right\rbrack  ,{53}}\right) ,\left( {\left\lbrack  \mathrm{g}\right\rbrack  ,{74}}\right) \} .$

(7) Dequeue R3, enqueue [a] and [b]. Queue: \{([a], 13), ([h], 17), $\left( {i,{21}}\right) ,\left( {\left\lbrack  b\right\rbrack  ,{27}}\right) ,\left( {\left\lbrack  d\right\rbrack  ,{30}}\right) ,\left( {{R6},{44}}\right) ,\left( {\left\lbrack  c\right\rbrack  ,{53}}\right) ,\left( {\left\lbrack  g\right\rbrack  ,{74}}\right) \} .$

(8) Dequeue [a]. The distance of a is 17, which is not larger than the distance of $\left\lbrack  h\right\rbrack$ ,so a is reported as nearest neighbor. Queue: $\{ \left( {\left\lbrack  h\right\rbrack  ,{17}}\right) ,\left( {i,{21}}\right) ,\left( {\left\lbrack  b\right\rbrack  ,{27}}\right) ,\left( {\left\lbrack  d\right\rbrack  ,{30}}\right) ,\left( {{R6},{44}}\right) ,\left( {\left\lbrack  c\right\rbrack  ,{53}}\right) ,\left( {\left\lbrack  g\right\rbrack  ,{74}}\right) \} .$

(9) Dequeue $\left\lbrack  \mathrm{h}\right\rbrack$ . The distance of $\mathrm{h}$ is 17,which is not larger than the distance of $i$ ,so $h$ is reported as second nearest neighbor. Queue: $\{ \left( {i,{21}}\right) ,\left( {\left\lbrack  b\right\rbrack  ,{27}}\right) ,\left( {\left\lbrack  d\right\rbrack  ,{30}}\right) ,\left( {{R6},{44}}\right) ,\left( {\left\lbrack  c\right\rbrack  ,{53}}\right) ,\left( {\left\lbrack  g\right\rbrack  ,{74}}\right) \} .$

(10) Dequeue i and report it as third nearest neighbor.

Observe that node $\mathrm{R}6$ is left on the priority queue at the end of the execution. This corresponds to the $k$ -nearest neighbor algorithm not being invoked on that node (see Section 5.2). For larger examples, the incremental algorithm will generally achieve more pruning than the $k$ -nearest neighbor algorithm, but never less.

Also note that once the nearest neighbor was found, the second and third nearest neighbors were obtained with very little additional work. This is often the case with the incremental nearest neighbor algorithm, regardless of the underlying spatial index. In other words, once the nearest neighbor is found, the next few nearest neighbors can be retrieved with virtually no additional work.

还要注意，一旦找到最近邻，获取第二近邻和第三近邻所需的额外工作量非常少。对于增量最近邻算法来说，通常都是这种情况，无论底层的空间索引是什么。换句话说，一旦找到最近邻，接下来的几个最近邻几乎可以在不增加额外工作量的情况下被检索出来。

### 4.5 Variants

### 4.5 变体

With relatively minor modifications, the incremental nearest neighbor algorithm can be used to find the farthest object from the query object. In this case, the queue elements are sorted in decreasing order of their distances. This is not enough, though, since objects or nodes contained in a node $n$ are generally at larger distances from the query object $q$ than $n$ is. This means that elements are enqueued with larger keys than the node they are contained in, which breaks the condition that elements are dequeued in decreasing order of distance. Instead, the key used for a node $n$ on the queue must be an upper bound on the distance from $q$ to an object in the subtree at $n$ ,e.g., ${d}_{\max }\left( {q,n}\right)  = \mathop{\max }\limits_{{p \in  n}}{d}_{p}\left( {q,p}\right)$ . The function implementing ${d}_{\max }$ must satisfy a consistency condition similar to that defined above for ${d}_{n}$ ; the only difference is that for ${d}_{\max }$ ,we replace $\leq$ in the condition by $\geq$ .

通过相对较小的修改，增量最近邻算法可用于查找与查询对象最远的对象。在这种情况下，队列元素按其距离的降序排序。不过，这还不够，因为节点 $n$ 中包含的对象或节点通常比 $n$ 本身与查询对象 $q$ 的距离更远。这意味着元素入队时的键值比它们所在的节点更大，这就破坏了元素按距离降序出队的条件。相反，队列中节点 $n$ 使用的键必须是从 $q$ 到 $n$ 子树中对象的距离的上限，例如 ${d}_{\max }\left( {q,n}\right)  = \mathop{\max }\limits_{{p \in  n}}{d}_{p}\left( {q,p}\right)$ 。实现 ${d}_{\max }$ 的函数必须满足类似于上面为 ${d}_{n}$ 定义的一致性条件；唯一的区别是，对于 ${d}_{\max }$ ，我们将条件中的 $\leq$ 替换为 $\geq$ 。

Another extension to the algorithm is to allow a minimum and a maximum to be imposed on the distances of objects that are reported. However, in order to effectively utilize a minimum, the distance function ${d}_{\max }$ defined above is needed. A node $n$ is then put on the queue only if ${d}_{\max }\left( {q,n}\right)$ is greater or equal to the minimum desired distance. Notice that, in this case, the algorithm performs a spatial selection operation in addition to the ranking.

该算法的另一个扩展是允许对所报告对象的距离施加最小值和最大值限制。然而，为了有效利用最小值，需要上面定义的距离函数 ${d}_{\max }$ 。只有当 ${d}_{\max }\left( {q,n}\right)$ 大于或等于所需的最小距离时，节点 $n$ 才会被放入队列。请注意，在这种情况下，该算法除了进行排序之外，还执行了空间选择操作。

Figure 5 gives a version of the algorithm with these two extensions added. The arguments ${Min}$ and ${Max}$ specify the minimum and maximum desired distances, and DoFarthest is a Boolean variable that is true when the farthest object is desired. In the latter case, negative distances are used as keys for the priority queue, so that elements get sorted in decreasing order of distance. The condition KeySign $\left( {d - e}\right)  \geq  0$ in line 19 of Figure 5 encompasses the conditions $d \geq  e$ and $d \leq  e$ ,to show when DoFarthest is false or true, respectively. In line 16, the key of the leaf node is assigned to $e$ . This is the minimum or maximum distance of the node,depending on the value of DoFarthest. The reason for multiplying the key by KeySign in line 16 is to cancel out the effect of multiplying the value of $d$ by KeySign in line 33, which makes it negative when looking for the farthest objects.

图 5 给出了添加了这两个扩展的算法版本。参数 ${Min}$ 和 ${Max}$ 指定了所需的最小和最大距离，DoFarthest 是一个布尔变量，当需要查找最远对象时为真。在后一种情况下，负距离被用作优先队列的键，以便元素按距离降序排序。图 5 第 19 行的条件 KeySign $\left( {d - e}\right)  \geq  0$ 包含了条件 $d \geq  e$ 和 $d \leq  e$ ，分别表示 DoFarthest 为假或真的情况。在第 16 行，叶节点的键被赋值给 $e$ 。这是该节点的最小或最大距离，具体取决于 DoFarthest 的值。在第 16 行将键乘以 KeySign 的原因是为了抵消在第 33 行将 $d$ 的值乘以 KeySign 的影响，当查找最远对象时，这会使该值变为负数。

A powerful way of extending the incremental nearest neighbor algorithm is to combine it with other spatial queries and/or restrictions on the objects or nodes. As an example, the algorithm can be combined with a range query by checking each object and node against the range prior to inserting it onto the priority queue, and rejecting those that do not fall in the range.

扩展增量最近邻算法的一种有效方法是将其与其他空间查询和/或对对象或节点的限制相结合。例如，可以通过在将每个对象和节点插入优先队列之前检查其是否在指定范围内，并拒绝那些不在范围内的对象和节点，从而将该算法与范围查询相结合。

<!-- Media -->

---

JCNEAREST( Query Object, SpatialIndex, Min, Max, DoFarthest)

JCNEAREST( 查询对象, 空间索引, 最小值, 最大值, 是否查找最远对象)

		Queue $\leftarrow$ NEWPRIORITYQUEUE(   )

		队列 $\leftarrow$ 新建优先队列(   )

		Enqueue( Queue, SpatialIndex.RootNode, 0)

		将( 队列, 空间索引.根节点, 0) 入队

		if DoFarthest then

		如果需要查找最远对象

				KeySign $\leftarrow   - 1$

				键符号 $\leftarrow   - 1$

		else

		否则

				KeySign $\leftarrow  1$

				键符号 $\leftarrow  1$

		endif

		结束条件判断

		while not ISEMPTY (Queue) do

		当队列不为空时执行以下操作

				Element $\leftarrow$ DEQUEUE(Queue)

				元素 $\leftarrow$ 出队(队列)

				if Element is a spatial object then

				如果元素是一个空间对象，则

						while Element $= \operatorname{FIRST}\left( \text{ Queue }\right)$ do

						当元素 $= \operatorname{FIRST}\left( \text{ Queue }\right)$ 时

							DELETEFIRST( Queue)

							删除首个元素(队列)

						enddo

						结束循环

						Report Element

						报告元素

				elseif Element is a leaf node then

				否则，如果元素是一个叶节点，则

						$e \leftarrow$ Element. Key*KeySign

						$e \leftarrow$ 元素.键 * 键符号

						for each Object in leaf node Element do

						对于叶节点元素中的每个对象，执行以下操作

								$d \leftarrow$ DIST(QueryObject,Object)

								$d \leftarrow$ 距离(查询对象,对象)

								if $d \geq$ Min and $d \leq$ Max and KeySign $* \left( {d - e}\right)  \geq  0$ then

								如果 $d \geq$ 小于最小值且 $d \leq$ 大于最大值且键符号 $* \left( {d - e}\right)  \geq  0$，则

										ENQUEUE( Queue, Object, KeySign * d )

										入队(队列, 对象, 键符号 * d)

								endif

								结束条件判断

						enddo

						结束循环

				else /* Element is a non-leaf node */

				否则 /* 元素是一个非叶节点 */

						for each Child node of node Element in SpatialIndex do

						对于空间索引中节点元素的每个子节点，执行以下操作

								${d}_{\min } \leftarrow$ MINDIST(QueryObject,Child)

								${d}_{\min } \leftarrow$ 最小距离(查询对象,子对象)

								${d}_{\max } \leftarrow  \operatorname{MaxDist}\left( \text{ QueryObject,Child }\right)$

								if ${d}_{\max } \geq$ Min and ${d}_{\min } \leq$ Max then

								如果 ${d}_{\max } \geq$ 为最小值且 ${d}_{\min } \leq$ 为最大值，则

										if DoFarthest then

										如果执行最远查询，则

												$d \leftarrow  {d}_{\max }$

										else

										否则

												$d \leftarrow  {d}_{\min }$

										endif

										结束条件判断

										Enqueue( Queue, Child, KeySign * d )

										将(队列, 子对象, 键符号 * d)入队

								endif

								结束条件判断

						enddo

						结束循环

				endif

				结束条件判断

		enddo

		结束循环

		Fig. 5. Enhanced incremental nearest neighbor algorithm.

		图 5. 增强型增量最近邻算法。

---

<!-- Media -->

Many such combined queries can be obtained by manipulating the distance functions so that they return special values for objects and nodes that should be rejected.

通过操作距离函数，使得它们为应被排除的对象和节点返回特殊值，就可以得到许多这样的组合查询。

The incremental nearest neighbor algorithm can clearly be used to solve the traditional $k$ -nearest neighbor problem,i.e.,given $k$ and a query object $q$ find the $k$ nearest neighbors of $q$ . This is done by simply retrieving $k$ neighbors with the algorithm and terminating once they have all been determined.

显然，增量最近邻算法可用于解决传统的 $k$ -最近邻问题，即给定 $k$ 和一个查询对象 $q$ ，找出 $q$ 的 $k$ 个最近邻。这可以通过使用该算法简单地检索 $k$ 个邻居，并在确定所有邻居后终止来实现。

### 4.6 Analysis

### 4.6 分析

Performing a comprehensive theoretical analysis of the incremental nearest neighbor algorithm is complicated, especially for high-dimensional spaces. Prior work in this area is limited to the case where both the data objects and the query object are points [Berchtold et al. 1997; Henrich 1994]. A number of simplifying assumptions were made, e.g., that the data objects are uniformly distributed in the data space. In this section, we discuss some of the issues involved, and sketch a rudimentary analysis for two-dimensional points, based on the one in [Henrich 1994].

对增量最近邻算法进行全面的理论分析是复杂的，特别是在高维空间中。该领域的先前工作仅限于数据对象和查询对象均为点的情况[Berchtold 等人，1997；Henrich，1994]。做出了许多简化假设，例如，假设数据对象在数据空间中均匀分布。在本节中，我们讨论其中涉及的一些问题，并基于[Henrich 1994]中的方法，对二维点进行初步分析。

We wish to analyze the situation after finding the $k$ nearest neighbors. Let $o$ be the ${k}^{th}$ nearest neighbor of the query object $q$ ,and let $r$ be the distance of $o$ from $q$ . The region within distance $r$ from $q$ is called the search region. Since we assume that $q$ is a point,the search region is a circle (or a hypersphere in higher dimensions) with radius $r$ . Figure 2 depicts this scenario. Observe that all objects inside the search region have already been reported by the algorithm (as the next nearest object), while all nodes intersecting the search region have been examined and their contents put on the priority queue. A further insight can be obtained about the contents of the priority queue by noting that if $n$ is a node that is completely inside the search region, all nodes and objects in the subtree rooted at $n$ have already been taken off the queue. Thus all elements on the priority queue are contained in nodes intersecting the boundary of the search region (the dark shaded region in Figure 2).

我们希望分析找到 $k$ 个最近邻后的情况。设 $o$ 是查询对象 $q$ 的第 ${k}^{th}$ 个最近邻，设 $r$ 是 $o$ 到 $q$ 的距离。距离 $q$ 为 $r$ 的区域称为搜索区域。由于我们假设 $q$ 是一个点，所以搜索区域是一个半径为 $r$ 的圆（在更高维度中是超球体）。图 2 描绘了这种情况。观察可知，搜索区域内的所有对象都已由算法报告（作为下一个最近对象），而与搜索区域相交的所有节点都已被检查，并且其内容已放入优先队列中。通过注意到如果 $n$ 是一个完全位于搜索区域内的节点，那么以 $n$ 为根的子树中的所有节点和对象都已从队列中移除，我们可以进一步了解优先队列的内容。因此，优先队列中的所有元素都包含在与搜索区域边界相交的节点中（图 2 中的深色阴影区域）。

Before proceeding any further, we point out that the algorithm does not access any nodes or objects that lie entirely outside the search region (i.e., that are farther from $q$ than $o$ is). This follows directly from the queue order and consistency conditions. In particular, the elements are retrieved from the priority queue in order of distance, and the consistency conditions guarantee that we never insert elements into the queue with smaller distances than that of the element last dequeued. Conversely, any algorithm that uses a spatial index must visit all the nodes that intersect the search region; otherwise, it may miss some objects that are closer to the query object than $o$ . Thus we have established that the algorithm visits the minimal number of nodes necessary for finding the ${k}^{th}$ nearest neighbor. This can be characterized by saying that the algorithm is optimal with respect to the structure of the spatial index. However, this does not mean that the algorithm is optimal with respect to the nearest neighbor problem; how close the algorithm comes to being optimal in this respect depends on the spatial index.

在进一步讨论之前，我们指出该算法不会访问完全位于搜索区域之外的任何节点或对象（即，比$o$距离$q$更远的节点或对象）。这直接源于队列顺序和一致性条件。特别是，元素按照距离顺序从优先队列中取出，并且一致性条件保证我们永远不会将距离小于最后出队元素的元素插入队列。相反，任何使用空间索引的算法都必须访问与搜索区域相交的所有节点；否则，它可能会错过一些比$o$更接近查询对象的对象。因此，我们已经确定该算法访问了找到${k}^{th}$最近邻所需的最少节点数。可以这样描述：该算法相对于空间索引的结构是最优的。然而，这并不意味着该算法相对于最近邻问题是最优的；该算法在这方面接近最优的程度取决于空间索引。

Generally, two steps are needed to derive performance measures for the incremental nearest neighbor algorithm. First, the expected area of the search region is determined. Then, based on the expected area of the search region and an assumed distribution of the locations and sizes of the leaf nodes, we can derive such measures as the expected number of leaf nodes accessed by the algorithm (i.e., intersected by the search region) or the expected number of objects in the priority queue. Henrich [1994] describes one such approach, which uses a number of simplifying assumptions. In particular,it assumes $N$ uniformly distributed data points in the two-dimensional interval $\left\lbrack  {0,1}\right\rbrack   \times  \left\lbrack  {0,1}\right\rbrack$ ,the leaf nodes are assumed to form a grid at the lowest level of the spatial index with average occupancy of $c$ points, and the search region is assumed to be completely contained in the data space. Since we assume uniformly distributed points, the expected area of the search region is $k/N$ and the expected area of the leaf node regions is $c/N$ . The area of a circle of radius $r$ is $\pi {r}^{2}$ ,so for the search region we have $\pi {r}^{2} = k/N$ ,which means that its radius is $r = \sqrt{k/\left( {\pi N}\right) }$ . The leaf node regions are squares,so their side length is $s = \sqrt{c/N}$ . Henrich [1994] points out that the number of leaf node regions intersected by the boundary of the search region is the same as that intersected by the boundary of its circumscribed square. Each of the four sides of the circumscribed square intersects $\lfloor {2r}/s\rfloor  \leq  {2r}/s$ leaf node regions. Since each two adjacent sides intersect the same leaf node region at a corner of the square, the expected number of leaf node regions intersected by the search region is bounded by

通常，需要两个步骤来推导增量最近邻算法的性能指标。首先，确定搜索区域的预期面积。然后，基于搜索区域的预期面积以及对叶节点位置和大小的假设分布，我们可以推导出诸如算法访问的叶节点的预期数量（即，与搜索区域相交的叶节点数量）或优先队列中对象的预期数量等指标。Henrich [1994]描述了一种这样的方法，该方法使用了一些简化假设。特别是，它假设在二维区间$\left\lbrack  {0,1}\right\rbrack   \times  \left\lbrack  {0,1}\right\rbrack$中存在$N$个均匀分布的数据点，假设叶节点在空间索引的最低层形成一个网格，平均每个网格包含$c$个点，并且假设搜索区域完全包含在数据空间中。由于我们假设点是均匀分布的，搜索区域的预期面积是$k/N$，叶节点区域的预期面积是$c/N$。半径为$r$的圆的面积是$\pi {r}^{2}$，因此对于搜索区域，我们有$\pi {r}^{2} = k/N$，这意味着其半径是$r = \sqrt{k/\left( {\pi N}\right) }$。叶节点区域是正方形，因此它们的边长是$s = \sqrt{c/N}$。Henrich [1994]指出，搜索区域边界相交的叶节点区域数量与该区域外接正方形边界相交的叶节点区域数量相同。外接正方形的四条边每条都与$\lfloor {2r}/s\rfloor  \leq  {2r}/s$个叶节点区域相交。由于每两条相邻的边在正方形的一个角处相交于同一个叶节点区域，搜索区域相交的叶节点区域的预期数量受限于

$$
4\left( {{2r}/s - 1}\right)  = 4\left( {\frac{2\sqrt{k/\left( {\pi N}\right) }}{\sqrt{c/N}} - 1}\right)  = 4\left( {2\sqrt{\frac{k}{\pi c}}}\right)  - 1.
$$

It is reasonable to assume that,on the average,half of the $c$ points in these leaf nodes are inside the search region, while half are outside. Thus the expected number of points remaining in the priority queue (the points in the dark shaded region in Figure 2) is at most

合理的假设是，平均而言，这些叶节点中的$c$个点有一半在搜索区域内，而另一半在搜索区域外。因此，优先队列中剩余的点的预期数量（图2中深色阴影区域中的点）最多为

$$
\frac{c}{2}4\left( {2\sqrt{\frac{k}{\pi c}} - 1}\right)  = {2c}\left( {2\sqrt{\frac{k}{\pi c}} - 1}\right)  = \frac{4}{\sqrt{\pi }}\sqrt{ck} - {2c} \approx  {2.26}\sqrt{ck} - {2c}.
$$

The number of points inside the search region (the light shaded region in Figure 2) is $k$ . Thus the expected number of points in leaf nodes intersected by the search region is at most $k + {2.26}\sqrt{ck} - {2c}$ . Since each leaf node contains $c$ points,the expected number of leaf nodes that were accessed to get these points is bounded by $k/c + {2.26}\sqrt{k/c} - 2$ .

搜索区域内的点的数量（图2中浅色阴影区域中的点）是$k$。因此，与搜索区域相交的叶节点中的点的预期数量最多为$k + {2.26}\sqrt{ck} - {2c}$。由于每个叶节点包含$c$个点，为获取这些点而访问的叶节点的预期数量受限于$k/c + {2.26}\sqrt{k/c} - 2$。

To summarize,the expected number of leaf node accesses is $O\left( {k + \sqrt{k}}\right)$ and the expected number of objects in the priority queue is $O\left( \sqrt{k}\right)$ . Intuitively, the "extra work" done by the algorithm comes from the boundary of the search region. Roughly speaking,the $k$ term in the expected number of leaf node accesses accounts for the leaf nodes completely inside the search region,while the $\sqrt{k}$ term accounts for the leaf nodes intersected by the boundary of the search region. The points on the priority queue lie outside the search region (since otherwise they would have been taken off the queue), but inside leaf nodes intersected by the boundary of the search region. If the average leaf node occupancy and average node fan-out are fairly high (say 50 or more), the number of leaf node accesses dominates the number of nonleaf node accesses, and the number of objects on the priority queue greatly exceeds the number of nodes on the queue. Thus we can approximate the total number of node accesses and total number of priority queue elements by the number of leaf node accesses and the number of objects on the priority queue. However, the traversal from the root of the spatial index to a leaf node containing the query object adds an $O\left( {\log N}\right)$ term to both of these measures.

综上所述，叶节点访问的期望次数为$O\left( {k + \sqrt{k}}\right)$，优先队列中对象的期望数量为$O\left( \sqrt{k}\right)$。直观地说，该算法所做的“额外工作”来自搜索区域的边界。大致而言，叶节点访问期望次数中的$k$项表示完全位于搜索区域内的叶节点，而$\sqrt{k}$项表示与搜索区域边界相交的叶节点。优先队列中的点位于搜索区域之外（因为否则它们会从队列中移除），但位于与搜索区域边界相交的叶节点内。如果平均叶节点占用率和平均节点扇出率相当高（例如50或更高），则叶节点访问次数将主导非叶节点访问次数，并且优先队列中的对象数量将大大超过队列中的节点数量。因此，我们可以用叶节点访问次数和优先队列中的对象数量来近似节点访问的总次数和优先队列元素的总数。然而，从空间索引的根节点遍历到包含查询对象的叶节点会在这两个度量中都增加一个$O\left( {\log N}\right)$项。

If the spatial index is disk-based, the cost of disk accesses is likely to dominate the cost of priority queue operations. However, if the spatial index is memory-based, the priority queue operations are the single largest cost factor for the algorithm. In typical priority queue implementations (e.g., binary heap), the cost of each insertion and deletion operation is $O\left( {\log m}\right)$ where $m$ is the size of the priority queue. The number of objects inserted into the priority queue is $O\left( {k + \sqrt{k}}\right)$ ,each for a cost of $O\left( {\log \sqrt{k}}\right)$ (since the expected size is bounded by $O\left( {\sqrt{}k}\right)$ ),for a total cost of $O(k +$ $\sqrt{k}) \cdot  O\left( {\log \sqrt{k}}\right)  = O\left( {k\log k}\right)$ (again,if we take the nonleaf nodes into account, the formulas become more complicated).

如果空间索引基于磁盘，那么磁盘访问成本可能会主导优先队列操作的成本。然而，如果空间索引基于内存，那么优先队列操作是该算法的最大成本因素。在典型的优先队列实现（例如，二叉堆）中，每次插入和删除操作的成本为$O\left( {\log m}\right)$，其中$m$是优先队列的大小。插入优先队列的对象数量为$O\left( {k + \sqrt{k}}\right)$，每次插入的成本为$O\left( {\log \sqrt{k}}\right)$（因为期望大小受$O\left( {\sqrt{}k}\right)$限制），总成本为$O(k +$$\sqrt{k}) \cdot  O\left( {\log \sqrt{k}}\right)  = O\left( {k\log k}\right)$（同样，如果考虑非叶节点，公式会变得更复杂）。

The analysis we have outlined is based on assumptions that generally do not hold in practice. In particular, the data is rarely uniformly distributed and the search region often extends beyond the data space. Nevertheless, our analysis allows fairly close predictions of actual behavior for two-dimensional point data, even when these assumptions do not hold. For higher dimensions, the situation is somewhat more complicated. A detailed analysis in that context is presented in Berchtold et al. [1997].

我们所概述的分析是基于一些在实际中通常不成立的假设。特别是，数据很少是均匀分布的，并且搜索区域通常会超出数据空间。尽管如此，即使这些假设不成立，我们的分析也能对二维点数据的实际行为做出相当准确的预测。对于更高维度，情况会稍微复杂一些。Berchtold等人（1997年）对该情况进行了详细分析。

### 4.7 Correctness

### 4.7 正确性

Let us turn to the correctness of the algorithm in Figure 3. We ignore for the moment the issue of reporting an object more than once. Given a data object $o$ ,define its ancestor set,denoted $A\left( o\right)$ ,to include $o$ itself,leaf nodes $n$ that contain $o$ for which ${d}_{o}\left( {q,o}\right)  \geq  {d}_{n}\left( {q,n}\right)$ (at least one such node is guaranteed to exist by the consistency of the distance functions), and all ancestors ${n}^{\prime }$ of $n$ . Applied recursively,the consistency property ensures that ${d}_{o}\left( {q,o}\right)  \geq  {d}_{n}\left( {q,{n}^{\prime }}\right)$ . The elements in $A\left( o\right)$ can be interpreted as representing the object $o$ . The following theorem guarantees that an unreported object always has a representative on the queue. This directly implies that every object will eventually be reported, since only bounded numbers of objects and nodes are ever put on the queue.

让我们来讨论图3中算法的正确性。我们暂时忽略多次报告同一对象的问题。给定一个数据对象$o$，定义其祖先集，记为$A\left( o\right)$，包括$o$本身、包含$o$且满足${d}_{o}\left( {q,o}\right)  \geq  {d}_{n}\left( {q,n}\right)$的叶节点$n$（根据距离函数的一致性，至少保证存在一个这样的节点），以及$n$的所有祖先${n}^{\prime }$。递归应用一致性属性可确保${d}_{o}\left( {q,o}\right)  \geq  {d}_{n}\left( {q,{n}^{\prime }}\right)$。$A\left( o\right)$中的元素可以解释为表示对象$o$。以下定理保证未报告的对象在队列中始终有一个代表。这直接意味着每个对象最终都会被报告，因为放入队列的对象和节点数量是有限的。

THEOREM 1. Let $R$ be the set of objects already reported,and $Q$ the set of elements on the queue. The following is an invariant for the outer while-loop of INCNEAREST: For each object o in SpatialIndex,we have $A\left( o\right)  \cap  (Q$ $\cup  R) \neq  \varnothing$ (i.e.,at least one element in $A\left( o\right)$ is in $Q$ or in $R$ ).

定理1。设$R$为已报告的对象集合，$Q$为队列中的元素集合。以下是INCNEAREST外部while循环的一个不变式：对于空间索引中的每个对象o，有$A\left( o\right)  \cap  (Q$$\cup  R) \neq  \varnothing$（即，$A\left( o\right)$中至少有一个元素在$Q$或$R$中）。

Proof. We prove the theorem for an arbitrary object $o$ by induction. Since we choose $o$ arbitrarily,the proof holds for all objects. The induction is on the number of loop executions. If we can show that the invariant holds before the first execution and that no loop execution falsifies it (i.e., it does not hold after the execution of the loop, assuming that it held before the execution), then we have shown that the invariant always holds. Clearly, it holds initially, as the only element on the queue is the root node of SpatialIndex,and the root is an ancestor of all nodes,and thus is in $A\left( o\right)$ for $o$ .

证明。我们通过归纳法为任意对象 $o$ 证明该定理。由于我们任意选择 $o$，所以该证明适用于所有对象。归纳是基于循环执行的次数。如果我们能证明不变式在第一次执行之前成立，并且没有循环执行会使其失效（即，假设在循环执行之前不变式成立，在循环执行之后它仍然成立），那么我们就证明了不变式始终成立。显然，它最初是成立的，因为队列中唯一的元素是空间索引（SpatialIndex）的根节点，而根节点是所有节点的祖先，因此对于 $o$ 来说它在 $A\left( o\right)$ 中。

Now assume that the invariant holds at the beginning of an execution of the while-loop. We show that it also holds at the end of it. If $o \in  R$ (i.e., $o$ has been reported),the invariant trivially holds,as $o$ will not be affected during loop execution. Otherwise, by the assumption that the invariant holds,there exists some $a \in  A\left( o\right)$ such that $a \in  Q$ . The invariant is unaffected if the next element to be dequeued is not $a$ ,so let us assume that $a$ will be dequeued next.

现在假设在 while 循环执行开始时不变式成立。我们证明在循环结束时它仍然成立。如果 $o \in  R$（即，$o$ 已被报告），那么不变式显然成立，因为 $o$ 在循环执行期间不会受到影响。否则，根据不变式成立的假设，存在某个 $a \in  A\left( o\right)$ 使得 $a \in  Q$。如果下一个要出队的元素不是 $a$，则不变式不受影响，因此让我们假设下一个要出队的元素是 $a$。

If $a = o$ ,then $o$ is subsequently reported,thereby moving from $Q$ to $R$ , and the invariant is maintained. If $a$ is a node,we consider the case of a leaf and nonleaf node separately:

如果 $a = o$，那么随后会报告 $o$，从而从 $Q$ 移动到 $R$，并且不变式得以维持。如果 $a$ 是一个节点，我们分别考虑叶节点和非叶节点的情况：

(1) If $a$ is a leaf node,the for-loop at line 11 enqueues all objects with a distance from $q$ of at least ${d}_{n}\left( {q,a}\right)$ (i.e.,at least DIST (QueryObject, Element)). Since $o$ is stored in $a$ (recall that $a \in  A\left( o\right)$ ),and since ${d}_{o}\left( {q,o}\right)  \geq  {d}_{n}\left( {q,a}\right)$ ,by construction of $A\left( o\right) ,o$ is indeed put on the queue.

(1) 如果 $a$ 是一个叶节点，第 11 行的 for 循环将所有与 $q$ 的距离至少为 ${d}_{n}\left( {q,a}\right)$（即，至少为 DIST (查询对象, 元素)）的对象入队。由于 $o$ 存储在 $a$ 中（回想 $a \in  A\left( o\right)$），并且由于 ${d}_{o}\left( {q,o}\right)  \geq  {d}_{n}\left( {q,a}\right)$，根据 $A\left( o\right) ,o$ 的构造，它确实会被放入队列中。

(2) If $a$ is a nonleaf node,then all its child nodes are enqueued. Since $a$ is in $A\left( o\right)$ ,i.e., $a$ is an ancestor of a leaf node $n$ that contains $o$ ,at least one of the child nodes of $a$ is in $A\left( o\right)$ ,maintaining the invariant.

(2) 如果 $a$ 是一个非叶节点，那么它的所有子节点都会入队。由于 $a$ 在 $A\left( o\right)$ 中，即，$a$ 是包含 $o$ 的叶节点 $n$ 的祖先，$a$ 的至少一个子节点在 $A\left( o\right)$ 中，从而维持了不变式。

So we see that for both leaf and nonleaf nodes, at least one of the enqueued elements is in $A\left( o\right)$ . Thus the invariant is maintained for object o. Since $o$ was chosen arbitrarily,we have shown that the invariant holds for all objects.

因此我们看到，对于叶节点和非叶节点，入队的元素中至少有一个在 $A\left( o\right)$ 中。因此，对象 o 的不变式得以维持。由于 $o$ 是任意选择的，我们已经证明了该不变式对所有对象都成立。

As mentioned, the theorem guarantees that an unreported object always has a representative on the queue. Since elements are retrieved from the queue in order of distance and all elements in $A\left( o\right)$ are no farther from the query point than $o$ ,at some point $o$ will be put on the queue and eventually reported. Also,when $o$ is reported,it is indeed the next closest object to $q$ . If not,then there exists an unreported object ${o}^{\prime }$ closer to $q$ . However,since all representatives of ${o}^{\prime }$ are also closer to $q$ than $o$ is,at least one of them will be dequeued before $o$ ,contradicting the assumption that $o$ was most recently dequeued.

如前所述，该定理保证未报告的对象在队列中始终有一个代表。由于元素是按照距离顺序从队列中取出的，并且 $A\left( o\right)$ 中的所有元素与查询点的距离都不超过 $o$，在某个时刻 $o$ 会被放入队列中并最终被报告。此外，当 $o$ 被报告时，它确实是距离 $q$ 最近的下一个对象。如果不是这样，那么存在一个未报告的对象 ${o}^{\prime }$ 比 $o$ 更接近 $q$。然而，由于 ${o}^{\prime }$ 的所有代表也比 $o$ 更接近 $q$，它们中至少有一个会在 $o$ 之前出队，这与 $o$ 是最近出队的假设相矛盾。

The correctness of the duplicate removal (lines 6-8 in Figure 3) follows directly from the ordering imposed on the priority queue. Thus the only way an object can be reported more than once is if it is inserted again into the queue after it is reported. However, this is avoided by the test in line 12 , and the fact that nodes are always processed before objects at the same distance from the query object.

去重（图 3 中的第 6 - 8 行）的正确性直接源于优先队列所施加的排序。因此，一个对象被报告多次的唯一方式是在它被报告后再次插入队列。然而，第 12 行的测试以及与查询对象距离相同的节点总是在对象之前处理这一事实避免了这种情况。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_21.jpg?x=520&y=221&w=593&h=602&r=0"/>

Fig. 6. An example of an R-tree of points with node capacity of 8 , showing a worst case for nearest neighbor search.

图 6. 一个节点容量为 8 的点 R 树示例，展示了最近邻搜索的最坏情况。

<!-- Media -->

### 4.8 Priority Queue

### 4.8 优先队列

The cost of priority queue operations plays a role in the performance of the incremental nearest neighbor algorithm. The larger the queue size, the more costly each operation becomes. If the queue gets too large to fit in memory, its contents must be stored in a disk-based structure instead of memory, making each operation even more costly. An example of the worst case of the queue size for the R-tree incremental nearest neighbor algorithm arises when all leaf nodes are within distance $d$ from the query object $q$ ,while all data objects are farther away from $q$ than $d$ . This is shown in Figure 6, where the query object as well as the data objects are points. In this case, all leaf nodes must be processed by the incremental algorithm, and all data objects must be inserted into the priority queue before the nearest neighbor can be determined. Note that any nearest neighbor algorithm that uses this R-tree has to visit all the leaf nodes, since the nearest neighbor is farther away from the query object than all the leaf nodes, and there is no other way to make sure that we have seen the nearest neighbor. Furthermore, note that a worst case like that depicted in Figure 6 is highly unlikely to arise in practice, since it depends on a particular configuration of both the data objects and the query object.

优先队列操作的成本在增量最近邻算法的性能中起着重要作用。队列规模越大，每次操作的成本就越高。如果队列变得太大而无法装入内存，其内容必须存储在基于磁盘的结构中，而不是内存中，这使得每次操作的成本更高。对于R树增量最近邻算法，队列规模最坏情况的一个例子是，当所有叶节点与查询对象 $q$ 的距离都在 $d$ 以内，而所有数据对象与 $q$ 的距离都大于 $d$ 时。如图6所示，其中查询对象和数据对象均为点。在这种情况下，增量算法必须处理所有叶节点，并且在确定最近邻之前，所有数据对象都必须插入到优先队列中。请注意，任何使用此R树的最近邻算法都必须访问所有叶节点，因为最近邻与查询对象的距离比所有叶节点都远，而且没有其他方法可以确保我们已经找到了最近邻。此外，请注意，如图6所示的最坏情况在实际中极不可能出现，因为它取决于数据对象和查询对象的特定配置。

As pointed out in Section 4.6, the objects on the priority queue are contained in leaf nodes intersected by the boundary of the search region. For two-dimensional uniformly distributed data points, we mentioned that the expected number of points in the priority queue when finding the $k$ nearest neighbors is $O\left( \sqrt{k}\right)$ . Even if $k$ is as large as several hundred million (of course,the data set has to be even larger than $k$ ),the size of the priority queue can still be manageably kept in memory. However, more complex objects than points and very skewed data distributions may cause larger proportions of the objects to be inserted into the priority queue. Moreover, as the number of dimensions grows, the size of the priority queue as a function of $k$ tends to get larger (see Section 7). So we must be prepared to deal with a very large priority queue.

正如4.6节所指出的，优先队列中的对象包含在与搜索区域边界相交的叶节点中。对于二维均匀分布的数据点，我们提到在查找 $k$ 个最近邻时，优先队列中预期的点数为 $O\left( \sqrt{k}\right)$ 。即使 $k$ 大到数亿（当然，数据集必须比 $k$ 更大），优先队列的规模仍然可以在内存中进行管理。然而，比点更复杂的对象和非常不均匀的数据分布可能会导致更大比例的对象被插入到优先队列中。此外，随着维数的增加，优先队列的规模作为 $k$ 的函数往往会变大（见第7节）。因此，我们必须做好处理非常大的优先队列的准备。

In cases where the priority queue exceeds the size of available memory it must be stored in whole or in part in a disk-resident structure. One possibility is to use a B-tree structure to store the entire contents of the priority queue. With proper buffer management, we should be able to arrange that the B-tree nodes that store elements with smaller distances (which are dequeued early) will be kept in memory. However, we believe that when the priority queue actually fits in memory, using B-trees is considerably slower than using fast heap-based approaches [Fredman et al. 1986], since the B-tree must expend more work on maintaining the queue elements in fully sorted order. In contrast, heap methods impose a much looser structure on the elements. A hybrid scheme for storing the priority queue, where a portion of the priority queue is kept in memory and a portion is kept on disk, seems more appropriate.

在优先队列超过可用内存大小的情况下，它必须全部或部分存储在基于磁盘的结构中。一种可能性是使用B树结构来存储优先队列的全部内容。通过适当的缓冲区管理，我们应该能够安排将存储距离较小元素（这些元素会较早出队）的B树节点保留在内存中。然而，我们认为，当优先队列实际上可以装入内存时，使用B树比使用基于快速堆的方法要慢得多[Fredman等人，1986]，因为B树必须花费更多的工作来将队列元素保持在完全排序的顺序。相比之下，堆方法对元素施加的结构要宽松得多。一种混合方案，即优先队列的一部分保留在内存中，一部分保留在磁盘上，似乎更为合适。

A simple way to implement a hybrid memory/disk-based priority queue is to partition the queue elements based on distance. We outline how this can be done below. The contents of the priority queue are split into three tiers. The first tier is kept in a memory-based heap structure, while the second and third tiers are kept in a disk file (the difference is that a little more structure is imposed on the contents of the second tier). Let ${D}_{0},{D}_{1},{D}_{2}$ ,

实现基于内存/磁盘的混合优先队列的一种简单方法是根据距离对队列元素进行分区。下面我们概述如何实现这一点。优先队列的内容被分为三层。第一层保存在基于内存的堆结构中，而第二层和第三层保存在磁盘文件中（不同之处在于第二层的内容施加了更多的结构）。设 ${D}_{0},{D}_{1},{D}_{2}$ ，

$\ldots ,{D}_{m}$ be some monotonically increasing sequence,where ${D}_{0} = 0$ and ${D}_{m}$ is an upper bound on the largest possible distance from the query object $q$ to a data object (e.g.,the distance from $q$ to the farthest corner of the data space). We use the sequence to define ranges of distance, and associate different ranges with the various tiers. When a new element with a distance of $r$ from the query object is inserted into the priority queue,that element gets added to the tier whose associated distance range matches $r$ . Initially,tier 1 is associated with the distance range $\left\lbrack  {{D}_{0},{D}_{1}}\right)$ ,i.e.,queue elements in this range are stored in the memory-based heap structure; tier 2 with the range $\left\lbrack  {{D}_{1},{D}_{p}}\right)$ ; and tier 3 with the range $\left\lbrack  {{D}_{p + 1},{D}_{m}}\right)$ . The contents of tier 2 are divided into $p$ ranges, $\left\lbrack  {{D}_{1},{D}_{2}}\right) ,\left\lbrack  {{D}_{2},{D}_{3}}\right) ,\ldots$ , $\left\lbrack  {{D}_{p},{D}_{p + 1}}\right)$ . The value of $p$ depends on how many ranges it is cost-effective to maintain,but it can be as high as $m$ . When tier 1 is exhausted,we move the elements in distance range $\left\lbrack  {{D}_{1},{D}_{2}}\right)$ from tier 2 to tier 1 and associate tier 1 with that distance range. The next time tier 1 is exhausted, we move elements in distance range $\left\lbrack  {{D}_{2},{D}_{3}}\right)$ into tier 1,and so on. If this happens often enough, we will eventually exhaust tier 2 . When this happens, we scan the entire contents of tier 3 and rebuild tiers 1 and 2 with new ranges. Note that moving elements from tier 3 to tier 2 only when tier 2 is exhausted, rather than each time tier 1 is exhausted, reduces the number of scans of tier 3 , which may contain a large number of elements.

$\ldots ,{D}_{m}$ 为某个单调递增序列，其中 ${D}_{0} = 0$ 且 ${D}_{m}$ 是查询对象 $q$ 到数据对象的最大可能距离的上限（例如，$q$ 到数据空间最远角落的距离）。我们使用该序列来定义距离范围，并将不同的范围与各个层级相关联。当一个与查询对象的距离为 $r$ 的新元素插入到优先队列中时，该元素会被添加到其关联距离范围与 $r$ 匹配的层级。最初，层级 1 与距离范围 $\left\lbrack  {{D}_{0},{D}_{1}}\right)$ 相关联，即此范围内的队列元素存储在基于内存的堆结构中；层级 2 与范围 $\left\lbrack  {{D}_{1},{D}_{p}}\right)$ 相关联；层级 3 与范围 $\left\lbrack  {{D}_{p + 1},{D}_{m}}\right)$ 相关联。层级 2 的内容被划分为 $p$ 个范围，$\left\lbrack  {{D}_{1},{D}_{2}}\right) ,\left\lbrack  {{D}_{2},{D}_{3}}\right) ,\ldots$、$\left\lbrack  {{D}_{p},{D}_{p + 1}}\right)$。$p$ 的值取决于维护多少个范围具有成本效益，但它可以高达 $m$。当层级 1 耗尽时，我们将距离范围 $\left\lbrack  {{D}_{1},{D}_{2}}\right)$ 内的元素从层级 2 移动到层级 1，并将层级 1 与该距离范围相关联。下次层级 1 耗尽时，我们将距离范围 $\left\lbrack  {{D}_{2},{D}_{3}}\right)$ 内的元素移动到层级 1，依此类推。如果这种情况频繁发生，最终我们将耗尽层级 2。当这种情况发生时，我们扫描层级 3 的全部内容，并使用新的范围重建层级 1 和层级 2。请注意，仅在层级 2 耗尽时才将元素从层级 3 移动到层级 2，而不是每次层级 1 耗尽时都移动，这样可以减少对可能包含大量元素的层级 3 的扫描次数。

In general, when the distance of the elements at the head of the priority queue is in the range $\left\lbrack  {{D}_{i},{D}_{i + 1}}\right)$ for some $i = 0,\ldots ,m$ ,i.e.,all neighbors with distances less than ${D}_{i}$ from $q$ have already been reported; tier 1 is associated with the range $\left\lbrack  {{D}_{i},{D}_{i + 1}}\right)$ ; tier 2 with the range $\left\lbrack  {{D}_{i + 1},{D}_{i + s + 1}}\right)$ ; and tier 3 with the range $\left\lbrack  {{D}_{i + s + 2},{D}_{m}}\right)$ ,where $s = p - \left( {i{\;\operatorname{mod}\;p}}\right)$ . We keep the elements in tier 2 in a set of linked lists,one for each interval $\left\lbrack  {D}_{j}\right.$ , ${D}_{j + 1}$ ) where $j = i + 1,\ldots ,i + s$ . In order to save on disk I/Os,we can associate a buffer with each of these linked lists and group elements into pages of fixed size. An alternative to using linked lists within the same file is to use a separate file for each range. Also, rather than associating range $\left\lbrack  {{D}_{i + 1},{D}_{i + s + 1}}\right)$ with tier 2,we can associate with it the entire range $\left\lbrack  {D}_{i + 1}\right.$ , ${D}_{i + p + 1}$ ),so that newly inserted elements in that range get inserted into tier 2 rather than tier 3 . However, we still do not want to scan tier 3 each time we exhaust tier 1 , so tier 3 will also contain elements in the range $\left\lbrack  {{D}_{i + s + 1},{D}_{i + p + 1}}\right)$ . These elements get moved into tier 2 when tier 3 gets scanned next,which happens when $i{\;\operatorname{mod}\;p} = 0$ .

一般来说，当优先队列头部元素的距离处于某个$i = 0,\ldots ,m$对应的范围$\left\lbrack  {{D}_{i},{D}_{i + 1}}\right)$内时，即所有与$q$的距离小于${D}_{i}$的邻居都已被报告；第一层（tier 1）对应范围$\left\lbrack  {{D}_{i},{D}_{i + 1}}\right)$；第二层（tier 2）对应范围$\left\lbrack  {{D}_{i + 1},{D}_{i + s + 1}}\right)$；第三层（tier 3）对应范围$\left\lbrack  {{D}_{i + s + 2},{D}_{m}}\right)$，其中$s = p - \left( {i{\;\operatorname{mod}\;p}}\right)$。我们将第二层的元素存储在一组链表中，每个区间$\left\lbrack  {D}_{j}\right.$，${D}_{j + 1}$）（其中$j = i + 1,\ldots ,i + s$）对应一个链表。为了节省磁盘输入/输出（I/O），我们可以为每个链表关联一个缓冲区，并将元素分组到固定大小的页面中。在同一文件中使用链表的一种替代方法是为每个范围使用单独的文件。此外，我们可以将整个范围$\left\lbrack  {D}_{i + 1}\right.$，${D}_{i + p + 1}$）与第二层关联，而不是仅将范围$\left\lbrack  {{D}_{i + 1},{D}_{i + s + 1}}\right)$与第二层关联，这样该范围内新插入的元素将插入到第二层而不是第三层。然而，我们仍然不希望每次耗尽第一层时都扫描第三层，因此第三层也将包含范围$\left\lbrack  {{D}_{i + s + 1},{D}_{i + p + 1}}\right)$内的元素。当第三层下次被扫描时（即$i{\;\operatorname{mod}\;p} = 0$时），这些元素将被移动到第二层。

A variation of this technique is to use an additional tier, between tier 1 and tier 2, in which elements are stored in an unsorted list in memory. The idea is that because we limit the size of the memory-based heap, the insertion and deletion operations on it are less expensive. Keeping the new tier 2 in memory, but outside the heap, makes it inexpensive to add elements to it (i.e., this does not require disk I/Os). Moreover, if only a small number of neighbors is requested, the elements in tier 2 will never need to be placed on the heap.

这种技术的一种变体是在第一层和第二层之间使用一个额外的层，其中元素以未排序的列表形式存储在内存中。其思路是，由于我们限制了基于内存的堆的大小，因此对其进行插入和删除操作的成本较低。将新的第二层保留在内存中，但在堆之外，使得向其中添加元素的成本较低（即不需要磁盘输入/输出）。此外，如果只请求少量邻居，第二层中的元素将永远不需要放置在堆上。

The remaining question is how to choose the sequence ${D}_{0},{D}_{1},{D}_{2},\ldots$ , ${D}_{m}$ . A naive way is to simply guess some distance threshold ${D}_{T}$ ,and then set ${D}_{i} = i \cdot  {D}_{T}$ . Alternatively,we can assume some data distribution and use it to derive an appropriate sequence. For example, recall from Section 4 that under the assumptions made there, the expected number of leaf nodes intersected by the boundary of a search region of radius $r$ is bounded by $4\left( {{2r}/s - 1}\right)$ ,where $s = \sqrt{c/N}$ is the expected side length of each leaf node region. Again,assuming that half of the points in these nodes (i.e., $c/2$ ) are outside the search region, the expected number of points on the priority queue is at most $\left( {c/{24}}\right) \left( {{2r}/s - 1}\right)  = {2c}\left( {{2r}/s - 1}\right)$ . Assuming that we have space in memory for $M$ priority queue elements means that ${D}_{i}$ must satisfy the equation $i \cdot  M = {2c}\left( {2{D}_{i}/s - 1}\right)$ ,so that

剩下的问题是如何选择序列${D}_{0},{D}_{1},{D}_{2},\ldots$，${D}_{m}$。一种简单的方法是简单地猜测某个距离阈值${D}_{T}$，然后设置${D}_{i} = i \cdot  {D}_{T}$。或者，我们可以假设某种数据分布，并使用它来推导合适的序列。例如，回顾第4节，在那里所做的假设下，半径为$r$的搜索区域边界所相交的叶节点的期望数量受$4\left( {{2r}/s - 1}\right)$限制，其中$s = \sqrt{c/N}$是每个叶节点区域的期望边长。同样，假设这些节点中一半的点（即$c/2$）在搜索区域之外，优先队列上的点的期望数量最多为$\left( {c/{24}}\right) \left( {{2r}/s - 1}\right)  = {2c}\left( {{2r}/s - 1}\right)$。假设我们在内存中有存储$M$个优先队列元素的空间，这意味着${D}_{i}$必须满足方程$i \cdot  M = {2c}\left( {2{D}_{i}/s - 1}\right)$，从而

$$
{D}_{i} = \frac{i \cdot  M}{2c} + 1/2 \cdot  s.
$$

Of course, this derivation is based on assumptions that do not generally hold in practice. Nevertheless, for two-dimensional points, it should work fairly well in practice. Moreover, it gives an indication of how to obtain such a sequence for other ways of analyzing the size of the priority queue.

当然，这种推导是基于一些在实际中通常不成立的假设。然而，对于二维点，它在实际应用中应该效果相当不错。此外，它还为如何通过其他方式分析优先队列的规模来获得这样一个序列提供了思路。

## 5. $k$ -NEAREST NEIGHBOR SEARCH IN R-TREES

## 5. $k$ -R树中的最近邻搜索

An alternative approach to nearest neighbor search in R-trees was proposed by Roussopoulos et al. [1995]. This approach is applicable when finding the $k$ nearest neighbors where $k$ is fixed in advance. This is in contrast to the incremental nearest neighbor algorithm,where $k$ does not have to be fixed in advance. The key idea of the $k$ -nearest neighbor algorithm is to maintain a global list of the candidate $k$ nearest neighbors as the R-tree is traversed in a depth-first manner. As we will see, the fact that the $k$ -nearest neighbor algorithm employs a pure depth-first traversal means that at any step the algorithm can only make local decisions about which node to visit (i.e., the next node to visit must be a child node of the current node), whereas our incremental nearest neighbor algorithm makes global decisions based on the contents of the priority queue (i.e., it can choose among the child nodes of all nodes that have already been visited).

Roussopoulos等人（1995年）提出了一种在R树中进行最近邻搜索的替代方法。当要查找预先固定的$k$个最近邻时，这种方法适用。这与增量最近邻算法不同，在增量最近邻算法中，$k$不必预先固定。$k$ -最近邻算法的关键思想是，在以深度优先的方式遍历R树时，维护一个候选$k$个最近邻的全局列表。正如我们将看到的，$k$ -最近邻算法采用纯深度优先遍历这一事实意味着，在任何步骤中，该算法只能对要访问的节点做出局部决策（即，下一个要访问的节点必须是当前节点的子节点），而我们的增量最近邻算法则根据优先队列的内容做出全局决策（即，它可以在所有已访问节点的子节点中进行选择）。

In this section we first describe a somewhat simplified version of the $k$ -nearest neighbor algorithm of Roussopoulos et al. [1995] and show an example of its execution. Next, we prove that our simplified version is in fact equivalent to the algorithm proposed by Roussopoulos et al. [1995]: Both versions visit the same nodes in the R-tree. Finally, we show how the $k$ -nearest neighbor algorithm can be transformed in a sequence of steps into an incremental algorithm.

在本节中，我们首先描述Roussopoulos等人（1995年）提出的$k$ -最近邻算法的一个简化版本，并展示其执行示例。接下来，我们证明我们的简化版本实际上等同于Roussopoulos等人（1995年）提出的算法：两个版本都会访问R树中的相同节点。最后，我们展示如何将$k$ -最近邻算法通过一系列步骤转换为增量算法。

### 5.1 Algorithm Description

### 5.1 算法描述

In the $k$ -nearest neighbor algorithm [Roussopoulos et al. 1995],the R-tree is traversed in a depth-first manner. The complications mentioned in Section 4 in performing nearest neighbor search with a depth-first traversal are overcome by maintaining a list of the candidate $k$ nearest neighbors. In particular, once we reach a leaf node containing the query object, we insert the contents of that node into the candidate list, and unwind the recursive traversal of the tree. Once the candidate list contains $k$ members,the largest distance of any of its members from the query object can be used to prune the search.

在$k$ -最近邻算法（Roussopoulos等人，1995年）中，以深度优先的方式遍历R树。通过维护一个候选$k$个最近邻的列表，克服了第4节中提到的在深度优先遍历中进行最近邻搜索时出现的复杂情况。具体来说，一旦我们到达包含查询对象的叶节点，就将该节点的内容插入到候选列表中，并展开树的递归遍历。一旦候选列表包含$k$个成员，就可以使用其任何成员到查询对象的最大距离来修剪搜索范围。

Figure 7 shows the $k$ -nearest neighbor algorithm,where NearestList denotes the list of the $k$ candidate nearest neighbors and NearestList.Max-Dist denotes the largest distance from the query object of any of the members of NearestList; if NearestList contains fewer than $k$ members,this distance is taken to be $\infty$ . When an object is inserted into NearestList in line 4 of KNEARESTTRAVERSAL, an existing member is replaced if the list already contains $k$ members. In particular,we replace the member that is farthest from the query object (i.e., the one at distance NearestList.Max-Dist). Before inserting an object into NearestList, we first make sure that its distance from the query object is smaller than NearestList.MaxDist (line 3 of KNEARESTTRAVERSAL). Note that NearestList.MaxDist decreases monotonically as more objects are inserted into the list, since we always replace objects with objects closer to the query object.

图7展示了$k$ -最近邻算法，其中NearestList表示$k$个候选最近邻的列表，NearestList.Max - Dist表示NearestList中任何成员到查询对象的最大距离；如果NearestList包含的成员少于$k$个，则该距离取为$\infty$。当在KNEARESTTRAVERSAL的第4行将一个对象插入到NearestList中时，如果列表已经包含$k$个成员，则会替换一个现有成员。具体来说，我们替换距离查询对象最远的成员（即，距离为NearestList.Max - Dist的那个成员）。在将一个对象插入到NearestList之前，我们首先要确保它到查询对象的距离小于NearestList.MaxDist（KNEARESTTRAVERSAL的第3行）。请注意，随着更多对象插入到列表中，NearestList.MaxDist会单调减小，因为我们总是用更接近查询对象的对象替换现有对象。

<!-- Media -->

---

	NEAREST( $k$ ,QueryObject,SpatialIndex)

	NEAREST( $k$ ,查询对象,空间索引)

			NearestList $\leftarrow$ NEWLIST(k)

			NearestList $\leftarrow$ NEWLIST(k)

			KNEARESTTRAVERSAL(NearestList, k, Query Object, SpatialIndex.RootNode)

			KNEARESTTRAVERSAL(NearestList, k, 查询对象, 空间索引.根节点)

			return NearestList

			返回NearestList

(NEARESTTRAVERSAL(NearestList, k, QueryObject, Node)

(NEARESTTRAVERSAL(NearestList, k, 查询对象, 节点)

			if Node is a leaf node then

			如果节点是叶节点

					for each Object in Node do

					对于节点中的每个对象

							if DIST(QueryObject, Object) < NearestList.MaxDist then

							如果DIST(查询对象, 对象) < NearestList.MaxDist

										INSERT(NearestList, DIST( Query Object, Object), Object)

										INSERT(NearestList, DIST( 查询对象, 对象), 对象)

							endif

							结束条件判断

					enddo

					结束循环

			else

			否则

					ActiveBranchList $\leftarrow$ entries in Node

					活动分支列表 $\leftarrow$ 节点中的条目

					SORTBRANCHLIST( QueryObject, ActiveBranchList)

					对分支列表排序（查询对象，活动分支列表）

					for each Child node in ActiveBranchList do

					对活动分支列表中的每个子节点执行以下操作

							if DIST(QueryObject, Child) < NearestList.MaxDist then

							如果查询对象与子节点的距离（DIST）小于最近列表的最大距离（NearestList.MaxDist），则

										KNEARESTTRAVERSAL(NearestList, k, QueryObject, Child)

										执行K近邻遍历（最近列表，k值，查询对象，子节点）

							else

							否则

										exit loop

										退出循环

							endif

							结束条件判断

					enddo

					结束循环

			endif

			结束条件判断

---

Fig. 7. $k$ -nearest neighbor algorithm.

图7. $k$ -最近邻算法。

<!-- Media -->

In KNEARESTTRAVERSAL, if Node is a nonleaf node, its child nodes are visited in order of distance from the query object. This is done by building the list ActiveBranchList of the entries in Node and sorting it by distance from the query object (see Section 5.3 for different ways of defining this order). Next, we iterate through the list (in the sorted order) and recursively invoke KNEARESTTRAVERSAL on the child nodes. Once the distance of Child from the query object is larger than NearestList.MaxDist, we ignore Child and the rest of the entries in ActiveBranchList. We do this because no object in the subtree of Child (or the remaining entries in ActiveBranch-List) will be inserted into NearestList.

在K近邻遍历（KNEARESTTRAVERSAL）中，如果节点是非叶子节点，则按与查询对象的距离顺序访问其子节点。这通过构建节点条目的活动分支列表（ActiveBranchList）并按与查询对象的距离对其进行排序来实现（有关定义此顺序的不同方法，请参阅第5.3节）。接下来，我们按排序顺序遍历列表，并对子节点递归调用K近邻遍历。一旦子节点与查询对象的距离大于最近列表的最大距离（NearestList.MaxDist），我们就忽略该子节点以及活动分支列表中的其余条目。我们这样做是因为子节点子树中的任何对象（或活动分支列表中的其余条目）都不会插入到最近列表中。

The difference between the $k$ -nearest neighbor algorithm in Figure 7 and the original presentation of Roussopoulos et al. [1995] is in the treatment of ActiveBranchList. We use only one pruning strategy to eliminate entries from consideration, by comparing their distances to NearestList.MaxDist, while Roussopoulos et al. [1995] identify two other pruning strategies. However, in Section 5.3 we show that the other two pruning strategies do not in fact allow any more pruning than the one we use.

图7中的$k$ -最近邻算法与Roussopoulos等人[1995]的原始表述之间的区别在于对活动分支列表（ActiveBranchList）的处理。我们仅使用一种剪枝策略，通过将它们与最近列表的最大距离（NearestList.MaxDist）进行比较来排除条目，而Roussopoulos等人[1995]确定了另外两种剪枝策略。然而，在第5.3节中，我们表明另外两种剪枝策略实际上并不比我们使用的策略允许更多的剪枝。

If the objects are stored outside the R-tree (i.e., the R-tree leaf nodes contain bounding rectangles and object references), a minor optimization can be made in line 4 of KNEARESTTRAVERSAL. We first compute the distance from the query object to the bounding rectangle. Only if this distance is less than NearestList.MaxDist, do we compute the real distance from Object to the query object. Otherwise, Object is not accessed, thereby potentially saving a disk I/O, as in this scenario the objects are stored outside the R-tree. Recall that $d\left( {q,r}\right)  \leq  d\left( {q,o}\right)$ if $r$ is a bounding rectangle of the object $o$ ,i.e.,the distance of $o$ from $q$ is never less than the distance of $r$ from $q$ .

如果对象存储在R树之外（即，R树叶节点包含边界矩形和对象引用），则可以在K近邻遍历（KNEARESTTRAVERSAL）的第4行进行一个小的优化。我们首先计算查询对象到边界矩形的距离。只有当此距离小于最近列表的最大距离（NearestList.MaxDist）时，我们才计算对象到查询对象的实际距离。否则，不访问该对象，从而可能节省磁盘I/O，因为在这种情况下对象存储在R树之外。回想一下，如果$r$是对象$o$的边界矩形，即$o$到$q$的距离永远不会小于$r$到$q$的距离。

In Roussopoulos et al. [1995] it is suggested that a sorted buffer be used to store NearestList. However,we found that for large values of $k$ ,the manipulation of NearestList started to become a major factor in the execution time of the algorithm. So we replaced the sorted buffer with a simple priority queue structure, sorted in decreasing order of distance, thereby making it easy to replace the farthest object.

在Roussopoulos等人[1995]中，建议使用排序缓冲区来存储最近列表（NearestList）。然而，我们发现对于较大的$k$值，最近列表的操作开始成为算法执行时间的一个主要因素。因此，我们用一个简单的优先队列结构代替了排序缓冲区，该结构按距离降序排序，从而便于替换最远的对象。

### 5.2 Example

### 5.2 示例

As an example of the algorithm, we describe its use in finding the three nearest neighbors to query point $q$ in the R-tree given in Figure 1. We show the algorithm steps and the contents of the ActiveBranchLists and of NearestList. The example makes use of the distances between $\mathrm{q}$ and the line segments and bounding rectangles given in Table I. An invocation with node $x$ is denoted $k$ -NN(x). We start by applying it to the root of the R-tree, R0. Next, we describe the subsequent invocations of the algorithm. Each of the line segment elements in NearestList is listed along with its distance from q. In our specification of NearestList, we also list the maximum distance for pruning (i.e., NearestList.MaxDist). Initially, NearestList is empty and the maximum distance is $\infty$ .

作为该算法的一个示例，我们描述其在图1所示的R树中查找查询点$q$的三个最近邻的应用。我们展示算法步骤以及活动分支列表（ActiveBranchLists）和最近列表（NearestList）的内容。该示例使用了表I中给出的$\mathrm{q}$与线段和边界矩形之间的距离。对节点$x$的调用表示为$k$ -NN(x)。我们首先将其应用于R树的根节点R0。接下来，我们描述算法的后续调用。最近列表中的每个线段元素都与其到查询点q的距离一起列出。在我们对最近列表的说明中，我们还列出了用于剪枝的最大距离（即，最近列表的最大距离NearestList.MaxDist）。最初，最近列表为空，最大距离为$\infty$。

(1) $k$ -NN(R0): ActiveBranchList for R0 is (R1, R2).

(1) $k$ -NN(R0)：R0的活动分支列表为(R1, R2)。

(a) $k$ -NN(R1): ActiveBranchList for R1 is (R4, R3).

(a) $k$ -NN(R1)：R1的活动分支列表为(R4, R3)。

(i) $k$ -NN(R4): insert d,g,h on NearestList: $\{ \left( {\mathrm{h},{17}}\right) ,\left( {\mathrm{d},{59}}\right) ,\left( {\mathrm{g},{81}}\right)$ : 81\}.

(i) $k$ -NN(R4)：在最近邻列表中插入d、g、h：$\{ \left( {\mathrm{h},{17}}\right) ,\left( {\mathrm{d},{59}}\right) ,\left( {\mathrm{g},{81}}\right)$ : 81}。

(ii) $k - \mathrm{{NN}}\left( {\mathrm{R}3}\right)$ : insert a,b in NearestList (replacing d,g): $\{ \left( {\mathrm{h},{17}}\right)$ , (a,17),(b,48): 48\}.

(ii) $k - \mathrm{{NN}}\left( {\mathrm{R}3}\right)$ ：在最近邻列表中插入a、b（替换d、g）：$\{ \left( {\mathrm{h},{17}}\right)$ , (a,17),(b,48): 48}。

(b) $k$ -NN(R2): ActiveBranchList for R2 is (R5, R6).

(b) $k$ -NN(R2)：R2的活动分支列表为(R5, R6)。

(i) $k - \mathrm{{NN}}\left( {\mathrm{R}5}\right)$ : i replaces b,but $\mathrm{c}$ is too distant: $\{ \left( {\mathrm{h},{17}}\right) ,\left( {\mathrm{a},{17}}\right)$ , (i, 21): 21\}.

(i) $k - \mathrm{{NN}}\left( {\mathrm{R}5}\right)$ ：i替换b，但$\mathrm{c}$ 距离太远：$\{ \left( {\mathrm{h},{17}}\right) ,\left( {\mathrm{a},{17}}\right)$ , (i, 21): 21}。

(ii) $k - \mathrm{{NN}}\left( {\mathrm{R}6}\right)$ : this invocation does not occur,as the distance of R6 from q is ≥ 21 .

(ii) $k - \mathrm{{NN}}\left( {\mathrm{R}6}\right)$ ：由于R6到q的距离≥21，此次调用不会发生。

The final contents of NearestList is $\{ \left( {h,{17}}\right) ,\left( {a,{17}}\right) ,\left( {i,{21}}\right) \}$ ,which is returned as the list of the three nearest neighbors of $\mathrm{q}$ .

最近邻列表的最终内容为$\{ \left( {h,{17}}\right) ,\left( {a,{17}}\right) ,\left( {i,{21}}\right) \}$，该列表作为$\mathrm{q}$ 的三个最近邻返回。

<!-- Media -->

<!-- figureText: a a r r O (b) 0 q (a) -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_27.jpg?x=470&y=231&w=689&h=387&r=0"/>

Fig. 8. An example of MinDIST (solid line) and MinMAXDIST (broken line) for a bounding rectangle $r$ . The distance of the object $o$ from $q$ is bounded from below by $\operatorname{MINDIST}\left( {q,r}\right)$ and from above by MinMaxDist(q,r). Notice that in (b) point $b$ is closer to $q$ than point $a$ ,while this is not the case in (a).

图8. 边界矩形$r$ 的最小距离（实线）和最小最大距离（虚线）示例。对象$o$ 到$q$ 的距离下界为$\operatorname{MINDIST}\left( {q,r}\right)$ ，上界为MinMaxDist(q,r)。注意，在(b)中，点$b$ 比点$a$ 更靠近$q$ ，而在(a)中并非如此。

<!-- Media -->

### 5.3 Node Ordering and Metrics

### 5.3 节点排序与度量

The ordering used to sort the elements in ActiveBranchList in Figure 7 can be based on various metrics for measuring the distances between QueryObject and the elements' bounding rectangles. Two such metrics are considered by Roussopoulos et al. [1995], MINDIST and MINMAXDIST. For bounding rectangle $r$ of node $n,\operatorname{MINDIST}\left( {q,r}\right)$ is the minimum possible distance from $q$ to an object in the subtree rooted at $n$ ,while $\operatorname{MINMAXDIST}\left( {q,r}\right)$ is the maximum distance from $q$ at which an object in the subtree rooted at $n$ is guaranteed to be found (i.e., it is the minimum of the maximum distances at which an object can be found). MINDIST and MINMAXDIST are calculated by using the geometry (i.e.,position and size) of the bounding rectangle $r$ of node $n$ and do not require examining the actual contents of $n$ . A more precise definition is given as follows. MINDIST(q,r)is the distance from $q$ to the closest point on the boundary of $r$ (not necessarily a corner),while $\operatorname{MINMAXDIST}\left( {q,r}\right)$ is the distance from $q$ to the closest corner of $r$ that is "adjacent" to the corner farthest from $q$ . Figure 8 shows two examples of the calculation of MINDIST and MinMaxDist, which are shown with a solid and a broken line, respectively. Notice that for the bounding rectangle in Figure 8a,the distance from $q$ to $a$ is less than the distance from $q$ to $b$ ,thereby accounting for the value of MINMAXDIST being equal to the former rather than the latter, while the opposite is true for Figure $8\mathrm{\;b}$ . In some sense,the two orderings represent the optimistic (MINDIST) and the pessimistic (MINMAXDIST) choice. To see this, observe that if ${r}_{1}$ and ${r}_{2}$ are minimum bounding rectangles in order of increasing value of MinDIST (i.e.,MINDIST $\left( {q,{r}_{1}}\right)  \leq  \operatorname{MINDIST}\left( {q,{r}_{2}}\right)$ ),then,at best, ${r}_{1}$ contains an object ${o}_{1}$ at a distance close to its MINDIST value,such that $\operatorname{DIST}\left( {q,{o}_{1}}\right)  \leq  \operatorname{MINDIST}\left( {q,{r}_{2}}\right)$ ; but this need not hold,as ${r}_{2}$ may contain an object closer to $q$ . If ${r}_{1}$ and ${r}_{2}$ are in order of increasing MinMAxDIST value,on the other hand,then in the worst case,the object in ${r}_{1}$ nearest to $q$ is at distance MinMaxDist $\left( {q,{r}_{1}}\right)$ ,which is no larger than MinMaxDist $\left( {q,{r}_{2}}\right)$ .

图7中用于对活动分支列表（ActiveBranchList）中的元素进行排序的顺序可以基于各种用于衡量查询对象（QueryObject）与元素边界矩形之间距离的度量标准。鲁索普洛斯（Roussopoulos）等人[1995]考虑了两种这样的度量标准，即最小距离（MINDIST）和最小最大距离（MINMAXDIST）。对于节点$n,\operatorname{MINDIST}\left( {q,r}\right)$的边界矩形$r$，$q$到以$n$为根的子树中的对象的最小可能距离为$q$到该子树中对象的最小距离，而$\operatorname{MINMAXDIST}\left( {q,r}\right)$是在以$n$为根的子树中保证能找到对象的$q$的最大距离（即，它是能找到对象的最大距离中的最小值）。最小距离（MINDIST）和最小最大距离（MINMAXDIST）是通过使用节点$n$的边界矩形$r$的几何信息（即位置和大小）来计算的，不需要检查$n$的实际内容。更精确的定义如下。最小距离（MINDIST(q,r)）是$q$到$r$边界上最近点（不一定是角点）的距离，而$\operatorname{MINMAXDIST}\left( {q,r}\right)$是$q$到$r$中与离$q$最远的角点“相邻”的最近角点的距离。图8展示了最小距离（MINDIST）和最小最大距离（MinMaxDist）计算的两个示例，分别用实线和虚线表示。请注意，对于图8a中的边界矩形，$q$到$a$的距离小于$q$到$b$的距离，因此最小最大距离（MINMAXDIST）的值等于前者而非后者，而对于图$8\mathrm{\;b}$则相反。从某种意义上说，这两种排序方式分别代表了乐观（最小距离，MINDIST）和悲观（最小最大距离，MINMAXDIST）的选择。为了说明这一点，观察可知，如果${r}_{1}$和${r}_{2}$是按照最小距离（MinDIST）值递增顺序排列的最小边界矩形（即，最小距离$\left( {q,{r}_{1}}\right)  \leq  \operatorname{MINDIST}\left( {q,{r}_{2}}\right)$），那么，在最好的情况下，${r}_{1}$包含一个距离接近其最小距离（MINDIST）值的对象${o}_{1}$，使得$\operatorname{DIST}\left( {q,{o}_{1}}\right)  \leq  \operatorname{MINDIST}\left( {q,{r}_{2}}\right)$；但这不一定成立，因为${r}_{2}$可能包含一个更接近$q$的对象。另一方面，如果${r}_{1}$和${r}_{2}$是按照最小最大距离（MinMAxDIST）值递增顺序排列的，那么在最坏的情况下，${r}_{1}$中离$q$最近的对象的距离为最小最大距离$\left( {q,{r}_{1}}\right)$，该距离不大于最小最大距离$\left( {q,{r}_{2}}\right)$。

Experiments reported by Roussopoulos et al. [1995] showed that ordering ActiveBranchList using MINDIST consistently performed better than MIN-MAXDIST. This was confirmed in our experiments, although we do not include that result in Section 6, which describes our experimental findings. We suspect that this indicates that the optimism inherent in MINDIST usually gives a better estimate of the distance of the nearest object than the pessimism inherent in MinMAxDIST, so that MinDIST order will in general lead to the nearest object(s) being found earlier in the Active-BranchList. So in this paper we assume that ActiveBranchList is ordered using MINDIST. In fact, the algorithm in Figure 7 depends on this, as we discuss at the end of Section 5.4.

鲁索普洛斯（Roussopoulos）等人[1995]报告的实验表明，使用最小距离（MINDIST）对活动分支列表（ActiveBranchList）进行排序的效果始终优于最小最大距离（MIN - MAXDIST）。我们的实验也证实了这一点，尽管我们在描述实验结果的第6节中没有包含该结果。我们怀疑这表明，最小距离（MINDIST）固有的乐观性通常比最小最大距离（MinMAxDIST）固有的悲观性能更好地估计最近对象的距离，因此最小距离（MinDIST）排序通常会使活动分支列表（Active - BranchList）中更早地找到最近的对象。因此，在本文中，我们假设使用最小距离（MINDIST）对活动分支列表（ActiveBranchList）进行排序。实际上，图7中的算法依赖于此，正如我们在第5.4节末尾所讨论的那样。

The metrics have other uses, regardless of which one is used for ordering ActiveBranchList. Since MINDIST represents the minimum distance at which an object can be found in a bounding rectangle $r$ ,it provides a means of pruning nodes from the search, given that a bound on the maximum distance is available. On the other hand,for any bounding rectangle $r$ , MINMAXDIST(q,r)is an upper bound on the distance of the object $o$ nearest $q$ . It should be clear that MinMAXDIST by itself does not help in pruning the search,as objects closer to $q$ can be found in elements of $n$ at positions with higher MinMAxDIST values. Moreover, since it only bounds the distance at which the closest element can be found, this property is of limited value because it is only useful when we seek the nearest neighbor (i.e., $k = 1$ ).

无论使用哪一种指标来对活动分支列表（ActiveBranchList）进行排序，这些指标都有其他用途。由于最小距离（MINDIST）表示在边界矩形 $r$ 中可以找到对象的最小距离，因此在已知最大距离上限的情况下，它提供了一种从搜索中剪枝节点的方法。另一方面，对于任何边界矩形 $r$，最小最大距离（MINMAXDIST(q,r)）是对象 $o$ 到 $q$ 的距离的上限。显然，仅靠最小最大距离（MinMAXDIST）本身并不能帮助剪枝搜索，因为在 $n$ 中具有更高最小最大距离（MinMAxDIST）值的位置的元素中可以找到更接近 $q$ 的对象。此外，由于它仅限制了可以找到最接近元素的距离，因此该属性的价值有限，因为它仅在我们寻找最近邻（即 $k = 1$）时才有用。

### 5.4 Pruning Strategies

### 5.4 剪枝策略

As already mentioned, as the entries are processed, the Roussopoulos et al. [1995] algorithm employs a set of three pruning strategies to prune entries from ActiveBranchList. Two classes of pruning strategies are identified in Roussopoulos et al. [1995]: downward pruning and upward pruning. In downward pruning, entries on ActiveBranchList are eliminated prior to processing the nodes (i.e., before entering the for-loop in line 10 of KNEARESTTRAVERSAL in Figure 7). In upward pruning, entries on Active-BranchList are eliminated after processing each node (i.e., after returning from the recursive call to KNEARESTTRAVERSAL in line 10 in Figure 7). Of the three pruning strategies discussed in Roussopoulos et al. [1995], two are said to be applicable to downward pruning and one to upward pruning. Below, we discuss these strategies in turn, and show that one of them is sufficient when used in a combination of upward and downward pruning. ${}^{3}$

如前所述，在处理条目时，鲁索普洛斯（Roussopoulos）等人 [1995] 的算法采用了一组三种剪枝策略，从活动分支列表（ActiveBranchList）中剪去条目。鲁索普洛斯等人 [1995] 确定了两类剪枝策略：向下剪枝和向上剪枝。在向下剪枝中，在处理节点之前（即，在图 7 中最近邻遍历（KNEARESTTRAVERSAL）的第 10 行进入 for 循环之前），消除活动分支列表（ActiveBranchList）上的条目。在向上剪枝中，在处理每个节点之后（即，在图 7 中第 10 行对最近邻遍历（KNEARESTTRAVERSAL）的递归调用返回之后），消除活动分支列表（ActiveBranchList）上的条目。在鲁索普洛斯等人 [1995] 讨论的三种剪枝策略中，有两种适用于向下剪枝，一种适用于向上剪枝。下面，我们依次讨论这些策略，并表明在结合使用向上和向下剪枝时，其中一种策略就足够了。${}^{3}$

Strategy 1 is used in downward pruning. It allows pruning an entry from ActiveBranchList whose bounding rectangle ${r}_{1}$ is such that $\operatorname{MINDIST}\left( {q,{r}_{1}}\right)$ $> \operatorname{MINMAXDIST}\left( {q,{r}_{2}}\right)$ ,where ${r}_{2}$ is some other bounding rectangle in Active-BranchList. However, as already pointed out, using MINMAXDIST for pruning is of limited value,as it is only useful when $k = 1$ .

策略 1 用于向下剪枝。它允许从活动分支列表（ActiveBranchList）中剪去其边界矩形 ${r}_{1}$ 满足 $\operatorname{MINDIST}\left( {q,{r}_{1}}\right)$ $> \operatorname{MINMAXDIST}\left( {q,{r}_{2}}\right)$ 的条目，其中 ${r}_{2}$ 是活动分支列表（ActiveBranchList）中的某个其他边界矩形。然而，如前所述，使用最小最大距离（MINMAXDIST）进行剪枝的价值有限，因为它仅在 $k = 1$ 时才有用。

---

<!-- Footnote -->

${}^{3}$ It may appear that we only use this pruning strategy for upward pruning in line 11 of KNEARESTTRAVERSAL in Figure 7. However, since the condition is checked before the recursive call to KNEARESTTRAVERSAL, the if statement actually does both upward and downward pruning.

${}^{3}$ 看起来我们仅在图 7 中最近邻遍历（KNEARESTTRAVERSAL）的第 11 行将此剪枝策略用于向上剪枝。然而，由于在对最近邻遍历（KNEARESTTRAVERSAL）进行递归调用之前检查了条件，因此 if 语句实际上同时进行了向上和向下剪枝。

<!-- Footnote -->

---

Strategy 2 prunes an object $o$ when $\operatorname{DIST}\left( {q,o}\right)  > \operatorname{MINMAXDIST}\left( {q,r}\right)$ , where $r$ is some bounding rectangle. Again,this strategy is only applicable to $k = 1$ . In Roussopoulos et al. [1995],it is claimed that this strategy is useful in downward pruning, but its inclusion is somewhat puzzling, since it does not help in pruning nodes from the search. It is possible that the authors intended strategy 2 to be used to prune objects in leaf nodes. However, this does not appear to be particularly fruitful, since it still requires that objects be accessed and their distances from $q$ be calculated. Another possible explanation for the inclusion of this strategy is that it can be used to discard the nearest object found in a subtree $s$ in ActiveBranch-List after $s$ is processed. However,the purpose of this is not clear,since a better candidate will replace this object later on, anyway.

当 $\operatorname{DIST}\left( {q,o}\right)  > \operatorname{MINMAXDIST}\left( {q,r}\right)$ 时，策略 2 剪去对象 $o$，其中 $r$ 是某个边界矩形。同样，此策略仅适用于 $k = 1$。在鲁索普洛斯等人 [1995] 中，声称此策略在向下剪枝中有用，但将其包含在内有点令人费解，因为它对从搜索中剪去节点没有帮助。有可能作者打算使用策略 2 来剪去叶节点中的对象。然而，这似乎并不是特别有效，因为仍然需要访问对象并计算它们与 $q$ 的距离。包含此策略的另一个可能的解释是，它可以用于在处理活动分支列表（ActiveBranchList）中的子树 $s$ 之后丢弃在该子树中找到的最近对象。然而，这样做的目的并不明确，因为无论如何，稍后会有更好的候选对象替换此对象。

Strategy 3 prunes any node from ActiveBranchList whose bounding rectangle $r$ is such that MinDist $\left( {q,r}\right)  >$ NearestList.MaxDist. It is applicable for any value of $k$ and in both downward and upward pruning. Note that although strategy 3 is not explicitly labeled as a downward pruning strategy in Roussopoulos et al. [1995], its use in downward pruning is noted. In particular, before entering the for-loop in line 10 of KNEAREST-TRAVERSAL in Figure 7, we can eliminate entries in ActiveBranchList with distances larger than NearestList.MaxDist (no pruning will occur though, unless NearestList contains at least $k$ entries).

策略 3 从活动分支列表（ActiveBranchList）中剪去其边界矩形 $r$ 满足最小距离（MinDist） $\left( {q,r}\right)  >$ 最近邻列表最大距离（NearestList.MaxDist）的任何节点。它适用于 $k$ 的任何值，并且适用于向下和向上剪枝。请注意，尽管在鲁索普洛斯等人 [1995] 中，策略 3 没有明确标记为向下剪枝策略，但提到了它在向下剪枝中的使用。具体而言，在进入图 7 中最近邻遍历（KNEAREST - TRAVERSAL）的第 10 行的 for 循环之前，我们可以消除活动分支列表（ActiveBranchList）中距离大于最近邻列表最大距离（NearestList.MaxDist）的条目（不过，除非最近邻列表（NearestList）至少包含 $k$ 个条目，否则不会进行剪枝）。

Recalling that strategy 1 is only applicable when $k = 1$ ,it can be shown that even in this case applying strategy 3 in upward pruning eliminates at least as many bounding rectangles as applying strategy 1 in downward pruning. To see this,let $r$ be the bounding rectangle in ActiveBranchList with the smallest MinMAxDIST value. Using strategy 1, we can prune any entry in ActiveBranchList with bounding rectangle ${r}^{\prime }$ such that MINDIST $\left( {q,{r}^{\prime }}\right)  > \operatorname{MINMAXDIST}\left( {q,r}\right)$ . However,strategy 1 will not prune $r$ or any entry in ActiveBranchList preceding it, regardless of the ordering. If ActiveBranchList is ordered based on MINMAXDIST, this clearly holds, since $\operatorname{MINDIST}\left( {q,r}\right)  \leq  \operatorname{MINMAXDIST}\left( {q,r}\right)$ . If ActiveBranchList is ordered based on MINDIST,the nodes preceding $r$ have MINDIST values smaller than $r$ ,so their MinDIST values must also be smaller than MinMAXDIST(q,r). Now let us see what entries can be pruned from ActiveBranchList by strategy 3 after processing the node corresponding to $r$ . In particular,at that point, $\operatorname{DIST}\left( {q,o}\right)  \leq  \operatorname{MINMAXDIST}\left( {q,r}\right)$ where $o$ is the candidate nearest object; this follows directly from the definition of MINMAXDIST. Therefore, when strategy 3 (based on $\operatorname{DIST}\left( {q,o}\right)$ ) is now applied to ActiveBranchList,it will prune at least as many entries as strategy 1 (based on MINMAXDIST(q,r)).

回顾策略1仅在$k = 1$ 时适用，由此可以证明，即使在这种情况下，在向上剪枝中应用策略3所消除的边界矩形数量至少与在向下剪枝中应用策略1所消除的数量相同。为了说明这一点，设$r$ 为活动分支列表（ActiveBranchList）中最小最大距离（MinMAxDIST）值最小的边界矩形。使用策略1，我们可以剪去活动分支列表中边界矩形为${r}^{\prime }$ 且最小距离（MINDIST）满足$\left( {q,{r}^{\prime }}\right)  > \operatorname{MINMAXDIST}\left( {q,r}\right)$ 的任何条目。然而，无论排序如何，策略1都不会剪去$r$ 或活动分支列表中在它之前的任何条目。如果活动分支列表是根据最小最大距离（MINMAXDIST）排序的，这显然成立，因为$\operatorname{MINDIST}\left( {q,r}\right)  \leq  \operatorname{MINMAXDIST}\left( {q,r}\right)$ 。如果活动分支列表是根据最小距离（MINDIST）排序的，那么在$r$ 之前的节点的最小距离（MINDIST）值小于$r$ 的最小距离值，因此它们的最小距离（MinDIST）值也一定小于最小最大距离（MinMAXDIST(q,r)）。现在让我们看看在处理与$r$ 对应的节点之后，策略3可以从活动分支列表中剪去哪些条目。特别地，在那个时刻，$\operatorname{DIST}\left( {q,o}\right)  \leq  \operatorname{MINMAXDIST}\left( {q,r}\right)$ ，其中$o$ 是候选最近对象；这直接源于最小最大距离（MINMAXDIST）的定义。因此，当现在将基于$\operatorname{DIST}\left( {q,o}\right)$ 的策略3应用于活动分支列表时，它至少会像基于最小最大距离（MINMAXDIST(q,r)）的策略1那样剪去同样多的条目。

The fact that we have eliminated strategies 1 and 2, and are interested in finding more than $k$ neighbors,implies that MINMAXDIST is not necessary for pruning, as it is not involved in strategy 3 . Thus, assuming that MINMAXDIST is not used for node ordering, the CPU cost of the algorithm is reduced, since we do not have to compute the MinMAXDIST value of each bounding rectangle; this is especially important because MinMAxDIST is more expensive to compute than MINDIST. We also observe that there is really no need to distinguish between downward and upward pruning, in the sense that there is no need to explicitly remove items from Active-BranchList. Instead, we just test each element on ActiveBranchList when its turn comes. If ActiveBranchList is ordered according to MINDIST, then once we prune one element, we can terminate all computation at this level, as all remaining elements have larger MINDIST values. This is exactly what we do in the if statement in line 11 of KNEARESTTRAVERSAL in Figure 7.

我们已经排除了策略1和策略2，并且有兴趣寻找多于$k$ 个邻居，这意味着最小最大距离（MINMAXDIST）对于剪枝并非必要，因为它不涉及策略3。因此，假设不使用最小最大距离（MINMAXDIST）进行节点排序，算法的CPU成本会降低，因为我们不必计算每个边界矩形的最小最大距离（MinMAXDIST）值；这一点尤为重要，因为计算最小最大距离（MinMAxDIST）比计算最小距离（MINDIST）的成本更高。我们还注意到，实际上没有必要区分向下剪枝和向上剪枝，也就是说，没有必要从活动分支列表（Active-BranchList）中显式地移除项目。相反，当轮到活动分支列表中的每个元素时，我们只需对其进行测试。如果活动分支列表是根据最小距离（MINDIST）排序的，那么一旦我们剪去一个元素，就可以终止该层的所有计算，因为所有剩余元素的最小距离（MINDIST）值都更大。这正是我们在图7中K近邻遍历（KNEARESTTRAVERSAL）的第11行的if语句中所做的。

### 5.5 Transformation

### 5.5 转换

In this section we show how the $k$ -nearest neighbor algorithm can be transformed into an incremental algorithm, and that the result is identical to our R-tree incremental algorithm. This discussion reveals the main difference between the two algorithms, namely that the control structure of the $k$ -nearest neighbor algorithm is fragmented among the nodes on the path from the root to the current node (as specified in the ActiveBranchList of each invocation of the algorithm), while the incremental nearest neighbor algorithm employs a unified control structure embodied in its priority queue.

在本节中，我们将展示如何将$k$ -近邻算法转换为增量算法，并且证明转换结果与我们的R树增量算法相同。这一讨论揭示了这两种算法的主要区别，即$k$ -近邻算法的控制结构分散在从根节点到当前节点的路径上的各个节点中（如算法每次调用的活动分支列表（ActiveBranchList）中所指定的），而增量近邻算法采用了一种统一的控制结构，体现在其优先队列中。

Recall that the R-tree $k$ -nearest neighbor algorithm traverses the R-tree in a depth-first manner. It keeps track of the state of the traversal (i.e., which nodes or bounding rectangles it has yet to process) by use of an ActiveBranchList for each level (note that at most one node is active at each level at any given time). In addition, in its original formulation (i.e., assuming a sorted buffer implementation) it keeps track of the distances from the query object of the data objects that it has seen by using NearestList sorted in increasing order of distance from the query object. Output of the $k$ nearest neighbors only occurs at the end of the traversal, since the R-tree is being traversed in its entirety (subject to the pruning of nodes in ActiveBranchList).

回顾一下，R树$k$ -近邻算法以深度优先的方式遍历R树。它通过为每一层使用一个活动分支列表（ActiveBranchList）来跟踪遍历的状态（即，它尚未处理的节点或边界矩形）（注意，在任何给定时间，每一层最多只有一个节点是活动的）。此外，在其原始形式中（即，假设采用排序缓冲区实现），它通过使用按与查询对象的距离递增顺序排序的最近列表（NearestList）来跟踪它所见过的数据对象与查询对象之间的距离。只有在遍历结束时才会输出$k$ 个最近邻，因为R树是被完整遍历的（受活动分支列表（ActiveBranchList）中节点剪枝的影响）。

If we want to transform the R-tree $k$ -nearest neighbor algorithm into an incremental algorithm, we also need to keep track of the nodes in the R-tree that have been seen (i.e., inserted into an ActiveBranchList) but not processed. These are the elements of the various instances of ActiveBranch-List; let $B$ denote their union. We assume that elements are removed from NearestList as they are processed. With the aid of $B$ ,it is now possible to tell if the first element $o$ in NearestList should be reported as the next nearest neighbor to $q$ . In particular,this is the case if $o$ is closer to $q$ than the closest node in $B$ because objects not yet encountered are in subtrees of nodes in $B$ . Without the global knowledge embodied by $B$ ,it is not possible to report even the nearest neighbor until we have unwound the recursive traversal of the algorithm up to the root node of the R-tree, because before then we do not know what is in the other subtrees of the root.

如果我们想将R树$k$最近邻算法转换为增量算法，我们还需要跟踪R树中已被访问（即插入到活动分支列表中）但尚未处理的节点。这些是活动分支列表（ActiveBranchList）不同实例中的元素；让$B$表示它们的并集。我们假设元素在被处理时会从最近列表（NearestList）中移除。借助$B$，现在可以判断最近列表中的第一个元素$o$是否应被报告为$q$的下一个最近邻。特别地，如果$o$比$B$中最近的节点更接近$q$，那么情况就是如此，因为尚未遇到的对象位于$B$中节点的子树中。如果没有$B$所体现的全局信息，在我们将算法的递归遍历展开到R树的根节点之前，甚至无法报告最近邻，因为在此之前我们不知道根节点的其他子树中有什么。

The $k$ -nearest neighbor algorithm can be modified to maintain this global unprocessed node list $B$ ,thereby enabling it to report nearest neighbors incrementally. This process can be made more efficient by keeping $B$ in sorted order based on distance from $q$ . However,this still leaves open the question of how to efficiently add and remove nodes from $B$ .

可以对$k$最近邻算法进行修改，以维护这个全局未处理节点列表$B$，从而使其能够增量式地报告最近邻。通过根据与$q$的距离对$B$进行排序，可以提高这个过程的效率。然而，这仍然留下了一个问题，即如何有效地从$B$中添加和移除节点。

Having made this modification, we can go even further and change the control structure. In particular, instead of keeping to the strict depth-first traversal,the list $B$ can be used to guide the traversal,i.e.,the node in $B$ closest to $q$ is taken as the next node to process. As a node is processed,it is deleted from $B$ ,and as a nonleaf node is processed,all its entries are added to $B$ . Note also that,as described above, $B$ is sorted in MINDIST order. It could be ordered by MinMAXDIST, but such an ordering has the disadvantage that the node in $B$ nearest to $q$ would not be immediately accessible. Furthermore, we observe that the penalty for choosing to process a wrong node is far less than the penalty for doing so in the $k$ -nearest algorithm, since all that has happened is the inspection of the node's entries, rather than the traversal of its entire subtree (subject to pruning, of course).

进行了这一修改后，我们可以更进一步，改变控制结构。特别地，不必严格遵循深度优先遍历，列表$B$可以用于引导遍历，即，将$B$中最接近$q$的节点作为下一个要处理的节点。当一个节点被处理时，它会从$B$中删除，当一个非叶节点被处理时，它的所有条目都会被添加到$B$中。还要注意，如上所述，$B$是按最小距离（MINDIST）顺序排序的。它也可以按最小最大距离（MinMAXDIST）排序，但这种排序的缺点是，$B$中最接近$q$的节点不能立即访问。此外，我们观察到，选择处理错误节点的代价远小于$k$最近邻算法中的代价，因为所发生的只是检查该节点的条目，而不是遍历其整个子树（当然，要进行剪枝）。

Note that with this transformation it is now possible to allow an unbounded $k$ as the last element in NearestList,i.e.,the one farthest from $q$ ,no longer plays a role. Of course,this also means that NearestList is no longer bounded, except by the total number of objects in the R-tree.

注意，通过这种转换，现在可以允许最近列表中的最后一个元素（即离$q$最远的元素）为无界的$k$，即它不再起作用。当然，这也意味着最近列表不再有界，除非受到R树中对象总数的限制。

The entire process can be performed most easily by merging $B$ and NearestList into one list called CombinedNearestList. By ordering Com-binedNearestList in increasing order of distance, we are able to preserve the role of the previous contents of ActiveBranchList, in that nodes that would have been pruned will be at greater distances in the CombinedNear-estList than the ${k}^{\text{th }}$ nearest object. Thus,they and their subtrees will not be traversed when outputting the $k$ nearest neighbors. Observe that the transformed algorithm only makes use of the MINDIST distance metric, thereby rendering moot the issue of whether or not to use the MinMAxDIST [Roussopoulos et al. 1995] metric. Also, the transformed algorithm will in general achieve more pruning of nodes than the original $k$ -nearest neighbor algorithm.

通过将$B$和最近列表合并为一个名为组合最近列表（CombinedNearestList）的列表，可以最轻松地执行整个过程。通过按距离递增的顺序对组合最近列表进行排序，我们能够保留活动分支列表先前内容的作用，因为在组合最近列表中，原本会被剪枝的节点与${k}^{\text{th }}$最近对象的距离会更远。因此，在输出$k$最近邻时，不会遍历这些节点及其子树。请注意，转换后的算法仅使用最小距离（MINDIST）距离度量，从而使是否使用最小最大距离（MinMAXDIST）[Roussopoulos等人，1995]度量的问题变得无关紧要。此外，一般来说，转换后的算法比原始的$k$最近邻算法能实现更多的节点剪枝。

We conclude our discussion of the $k$ -nearest neighbor algorithm by pointing out that the transformation yields an algorithm equivalent to the incremental algorithm presented earlier when CombinedNearestList is organized with a priority queue.

在结束对$k$最近邻算法的讨论时，我们指出，当组合最近列表使用优先队列进行组织时，这种转换产生的算法等同于前面介绍的增量算法。

## 6. EXPERIMENTAL RESULTS

## 6. 实验结果

In order to evaluate the R-tree incremental nearest neighbor algorithm in Figure 4 (denoted INN), we compared it to the result of using the R-tree $k$ -nearest neighbor algorithm of Roussopoulos et al. [1995] (denoted $k$ -NN) for distance browsing (Section 6.1). We also measured the incremental cost of using INN,i.e.,the cost of obtaining the $k + {1}^{st}$ neighbor once we have already obtained the ${k}^{th}$ neighbor (Section 6.2). By varying the number of objects that are browsed, we were able to see the true advantage of our method of computing the nearest neighbors incrementally, rather than committing ourselves to a predetermined number of nearest neighbors, as would be the case if we used the $k$ -nearest neighbor algorithm. (Remember that we do not know in advance how many objects will be browsed before finding the desired object.) Finally,we compare INN with $k$ -NN for computing the result of a $k$ -nearest neighbor query (Section 6.3). These studies were performed for small numbers of neighbors (i.e., fewer than 25), as this is the most common situation in which distance browsing is useful. Nevertheless, we also treat the case of a large number of neighbors in Section 6.3.

为了评估图4中的R树增量最近邻算法（记为INN），我们将其与使用Roussopoulos等人[1995]的R树$k$ -最近邻算法（记为$k$ -NN）进行距离浏览的结果进行了比较（6.1节）。我们还测量了使用INN的增量成本，即，在已经获得第$i$个邻居的情况下获取第$i + 1$个邻居的成本（6.2节）。通过改变浏览的对象数量，我们能够看到增量计算最近邻的方法的真正优势，而不是像使用$k$ -最近邻算法那样预先确定最近邻的数量。（请记住，在找到所需对象之前，我们事先并不知道会浏览多少个对象。）最后，我们将INN与$k$ -NN在计算$k$ -最近邻查询结果方面进行了比较（6.3节）。这些研究是针对少量邻居（即少于25个）进行的，因为这是距离浏览最有用的常见情况。尽管如此，我们在6.3节中也处理了大量邻居的情况。

In the experiments mentioned above, we measured the execution time, the disk I/O behavior, and the number of distance computations for two representative maps. In order to discern whether the size of the maps was a factor, we performed experiments in which the size was varied (Section 6.4). In addition, for an extreme case, we experimented with a very large data set (Section 6.5). Finally, in Section 6.6 we report the maximum size of the priority queue for the experiments in Sections 6.3 and 6.4.

在上述实验中，我们测量了两个代表性地图的执行时间、磁盘I/O行为和距离计算次数。为了辨别地图大小是否是一个因素，我们进行了改变地图大小的实验（6.4节）。此外，对于极端情况，我们对一个非常大的数据集进行了实验（6.5节）。最后，在6.6节中，我们报告了6.3节和6.4节实验中优先队列的最大大小。

The data sets used in the experiments consisted of line segments, both real-world data and randomly generated data. The real-world data consisted of four data sets from the TIGER/Line File [Bureau of the Census 1989] (see Figure 9):

实验中使用的数据集由线段组成，包括真实世界数据和随机生成的数据。真实世界数据由来自TIGER/Line文件[人口普查局1989]的四个数据集组成（见图9）：

(1) Howard County: 17,421 line segments.

(1) 霍华德县：17421条线段。

(2) Water in the Washington DC metro area: 37,495 line segments.

(2) 华盛顿特区都会区的水域：37495条线段。

(3) Prince George's County: 59,551 line segments.

(3) 乔治王子县：59551条线段。

(4) Roads in the Washington DC metro area: 200,482 line segments.

(4) 华盛顿特区都会区的道路：200482条线段。

The randomly generated line segment maps were constructed by generating random infinite lines in a manner independent of translation and scaling of the coordinate system [Lindenbaum and Samet 1995]. These lines were clipped to the map area to obtain line segments and then subdivided further at their intersection points with other line segments so that at the end, line segments meet only at endpoints. Note that the random maps do not necessarily model real-world maps perfectly. In particular, by their construction, random maps cover an entire square area, whereas this is not the case for most real maps (e.g., TIGER/Line File county maps). Furthermore, the random maps tend to be rather uniform, while real maps tend to have dense clusters of small line segments mixed with more sparsely covered areas. Nevertheless, these randomly generated maps do capture some important features of real maps (e.g., there is a low probability of more than four line segments meeting at a point), and they enabled us to run the experiments on a wide range of map sizes for maps with similar characteristics.

随机生成的线段地图是通过以与坐标系的平移和缩放无关的方式生成随机无限线来构建的[Lindenbaum和Samet 1995]。这些线被裁剪到地图区域以获得线段，然后在它们与其他线段的交点处进一步细分，以便最终线段仅在端点处相交。请注意，随机地图不一定能完美地模拟真实世界的地图。特别是，通过其构建方式，随机地图覆盖了整个正方形区域，而大多数真实地图并非如此（例如，TIGER/Line文件中的县地图）。此外，随机地图往往相当均匀，而真实地图往往有小线段的密集簇与覆盖更稀疏的区域混合。尽管如此，这些随机生成的地图确实捕捉到了真实地图的一些重要特征（例如，超过四条线段在一点相交的概率很低），并且它们使我们能够在具有相似特征的地图上对各种地图大小进行实验。

<!-- Media -->

<!-- figureText: (a) (b) (d) (c) -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_33.jpg?x=260&y=248&w=1144&h=1084&r=0"/>

Fig. 9. The four real-world data sets from the TIGER/Line File: (a) Howard, (b) Water, (c) PG, and (d) Roads.

图9. 来自TIGER/Line文件的四个真实世界数据集：(a) 霍华德，(b) 水域，(c) 乔治王子县，(d) 道路。

<!-- Media -->

Our experiments differ from those in Roussopoulos et al. [1995], which used a Hilbert-packed R-tree [Kamel and Faloutsos 1993; Roussopoulos and Leifker 1985],whereas we used an ${\mathrm{R}}^{ * }$ -tree. The Hilbert-packed R-tree is a static structure, constructed by applying a Peano-Hilbert space ordering (e.g., Samet [1990]) to spatial objects on the basis of their centroids. The leaf nodes of the R-tree are then built by filling them with the objects, and the nonleaf nodes are built on top, with bounding rectangles computed for the nodes. Notice that the conventional R-tree node splitting rules were not applied in the construction of the Hilbert-packed R-tree, since each node is filled to capacity by the Hilbert-packed R-tree construction algorithm. Because we are interested in dynamic environments, we chose to use the ${\mathrm{R}}^{ * }$ -tree rather than the Hilbert-packed R-tree for our experiments,except where noted.

我们的实验与Roussopoulos等人[1995]的实验不同，他们使用了希尔伯特填充R树[Kamel和Faloutsos 1993；Roussopoulos和Leifker 1985]，而我们使用了$R^*$ -树。希尔伯特填充R树是一种静态结构，它通过基于空间对象的质心对其应用皮亚诺 - 希尔伯特空间排序（例如，Samet [1990]）来构建。然后通过用对象填充R树的叶节点来构建它们，而非叶节点则在其上构建，并为节点计算边界矩形。请注意，在构建希尔伯特填充R树时没有应用传统的R树节点分裂规则，因为希尔伯特填充R树的构建算法会将每个节点填充到容量上限。因为我们对动态环境感兴趣，所以除特别说明外，我们在实验中选择使用$R^*$ -树而不是希尔伯特填充R树。

Most of the data sets we used were small enough to fit in the main memory of many modern computers (except in the experiments reported in Section 6.5). Nevertheless, we used a disk-based R-tree structure and employed buffers to store a limited number of recently used R-tree nodes (128). Thus we believe that our results will scale well to large data sets. The fact that we employ buffered $\mathrm{I}/\mathrm{O}$ ,with the added possibility of a requested disk block being in a disk cache or in operating system buffers, complicates the comparison between the two algorithms. There are two extremes: for each I/O, the requested disk block is found in memory, or every $\mathrm{I}/\mathrm{O}$ leads to disk activity. Given a query for a fixed number of neighbors, the incremental nearest neighbor (INN) algorithm shows less improvement over the $k$ -nearest neighbor algorithm(k - NN)in the former case (i.e., if the entire data sets resides in memory), and may even be slower, as seen for the small random data sets. This is mainly due to the overhead incurred by priority queue operations. However, for the other extreme, the INN algorithm would be even more advantageous than we found,as it always requests fewer R-tree nodes and objects than the $k$ -NN algorithm.

我们使用的大多数数据集都足够小，可以存入许多现代计算机的主内存中（第6.5节报告的实验除外）。尽管如此，我们还是使用了基于磁盘的R树结构，并采用缓冲区来存储有限数量的最近使用过的R树节点（128个）。因此，我们相信我们的结果能够很好地扩展到大型数据集。我们采用了带缓冲的$\mathrm{I}/\mathrm{O}$，并且请求的磁盘块有可能存在于磁盘缓存或操作系统缓冲区中，这使得两种算法之间的比较变得复杂。存在两种极端情况：对于每次I/O操作，请求的磁盘块都能在内存中找到；或者每次$\mathrm{I}/\mathrm{O}$操作都会导致磁盘活动。对于固定数量的最近邻查询，在前者情况下（即如果整个数据集都驻留在内存中），增量最近邻（INN）算法相对于$k$ -最近邻算法（k - NN）的改进较小，甚至可能更慢，从小型随机数据集的情况可以看出这一点。这主要是由于优先队列操作产生的开销所致。然而，对于另一种极端情况，INN算法会比我们所发现的更具优势，因为它总是比$k$ - NN算法请求更少的R树节点和对象。

For each experiment, we ran multiple queries on the same data set for the same number of neighbors. This was done so that more than one query point could be tested, as well as to make sure that the timing results were meaningful (given the timing granularity of the system we used). Since our R-tree implementation utilizes buffered I/O, a query may access disk blocks that were already loaded into the buffer by earlier queries in the same sequence. We feel that this was a reasonable choice to make, since the buffers were small compared to the data size, and clearing them prior to each query would have affected the timing results. Also, in a real world scenario, it is likely that a user will execute more than one query for a given map.

对于每个实验，我们在同一个数据集上针对相同数量的最近邻运行多个查询。这样做是为了能够测试多个查询点，同时确保计时结果有意义（考虑到我们所使用系统的计时粒度）。由于我们的R树实现采用了带缓冲的I/O，一个查询可能会访问已经被同一序列中较早的查询加载到缓冲区中的磁盘块。我们认为这是一个合理的选择，因为与数据大小相比，缓冲区较小，并且在每次查询之前清空缓冲区会影响计时结果。此外，在现实场景中，用户很可能会针对给定的地图执行多个查询。

We use three measures for comparing the algorithms: execution time, R-tree node I/O (frequently referred to as disk I/O [Beckmann et al. 1990; Kamel and Faloutsos 1994]), and object distance calculations. The R-tree node I/O is reported as the number of accesses, and may not correspond to actual disk I/O if nodes can be found in database or system buffers. However, we found that the number of accesses predicts the relative performance of actual disk I/O reasonably well. Furthermore, any saving due to buffering will show up in reduced execution time. Thus we used the disk I/O characterization.

我们使用三种指标来比较这些算法：执行时间、R树节点I/O（通常称为磁盘I/O [贝克曼等人，1990年；卡梅尔和法洛托斯，1994年]）和对象距离计算。R树节点I/O以访问次数来报告，如果节点可以在数据库或系统缓冲区中找到，那么它可能与实际的磁盘I/O不对应。然而，我们发现访问次数能够很好地预测实际磁盘I/O的相对性能。此外，由于缓冲而节省的任何开销都会体现在执行时间的减少上。因此，我们采用了磁盘I/O特征描述。

In all the experiments that we conducted, the maps were embedded in a ${16}\mathrm{\;K}$ by ${16}\mathrm{\;K}$ grid,and the capacity of each $\mathrm{R}$ -tree node was 50 . In order to simplify the analysis of the execution time results, we chose to store the actual line segments in the R-tree leaf nodes instead of just their bounding boxes. Organization of external object storage also has a large effect on the performance, and thus introduces an extra variable into the comparison of the two algorithms. Query points were uniformly distributed over the space covered by the map data, and the distance functions used to measure the distances of lines and bounding rectangles from the query points were based on the squared Euclidean metric (to avoid computing square roots). The experiments were run sufficiently often to obtain consistent results with a different query point each time. Execution times are reported in milliseconds per query; they include the CPU time consumed by the algorithm and its system calls. We used a SPARCstation 5 Model 70 rated at 60 SPECint92 and 47 SPECfp92, and a GNU C++ compiler set for maximum optimization (-03).

在我们进行的所有实验中，地图都嵌入在一个${16}\mathrm{\;K}$×${16}\mathrm{\;K}$的网格中，每个$\mathrm{R}$ -树节点的容量为50。为了简化对执行时间结果的分析，我们选择将实际的线段存储在R树叶节点中，而不仅仅是它们的边界框。外部对象存储的组织方式也对性能有很大影响，因此在比较这两种算法时引入了一个额外的变量。查询点在地图数据所覆盖的空间上均匀分布，用于测量线段和边界矩形与查询点之间距离的距离函数基于欧几里得平方度量（以避免计算平方根）。实验进行了足够多次，以每次使用不同的查询点获得一致的结果。执行时间以每次查询的毫秒数报告；它们包括算法及其系统调用所消耗的CPU时间。我们使用了一台SPARCstation 5 Model 70，其SPECint92评分为60，SPECfp92评分为47，并使用了一个设置为最大优化（-03）的GNU C++编译器。

<!-- Media -->

<!-- figureText: 100 INN (PG) INN (R64K) -+--- 3 k-NN, k=1,2,... (PG) k-NN, k=5,10,... (PG) 15 20 25 Number of nearest neighbors Execution time (ms, log scale) 10 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_35.jpg?x=226&y=229&w=539&h=437&r=0"/>

Fig. 10. Cumulative execution time for distance browsing.

图10. 距离浏览的累积执行时间。

<!-- figureText: R-tree node disk I/Os (log scale) 100 INN (PG) INN (R64K) J. K-NN, k=1,2,… (PG) - 曰 -- k-NN, k=5,10,... (PG) 15 20 25 Number of nearest neighbors 10 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_35.jpg?x=831&y=227&w=553&h=440&r=0"/>

Fig. 11. Cumulative R-tree node disk I/O for distance browsing.

图11. 距离浏览的累积R树节点磁盘I/O。

<!-- Media -->

### 6.1 Cumulative Cost of Distance Browsing

### 6.1 距离浏览的累积成本

In this section we focus on the distance browsing query when we do not know in advance how many neighbors will be needed before the query terminates. In this case,we need to reapply the $k$ -nearest neighbor algorithm as the value of $k$ changes. In contrast,in the case of the incremental nearest neighbor algorithm, we need to reinvoke the algorithm to obtain just one neighbor (i.e., the next nearest one). For these experiments we used the map of Prince George’s County (denoted ${PG}$ in the figures) as well as a randomly generated line map of a similar size, containing 64,000 lines (denoted ${R64K}$ ). We included the random line map to see if the performance was affected by some unknown characteristics of the PG map.

在本节中，我们关注距离浏览查询，即我们事先不知道在查询终止之前需要多少个最近邻的情况。在这种情况下，随着$k$值的变化，我们需要重新应用$k$ -最近邻算法。相比之下，对于增量最近邻算法，我们只需要重新调用该算法来获取一个最近邻（即下一个最近的邻居）。在这些实验中，我们使用了乔治王子县的地图（在图中表示为${PG}$）以及一个类似大小的随机生成的线图，其中包含64,000条线（表示为${R64K}$）。我们纳入随机线图是为了查看性能是否受到乔治王子县地图某些未知特征的影响。

Figures 10 through 12 show each measure's cumulative cost for distance browsing through the database by finding the neighbors incrementally. There are a number of ways of using a $k$ -nearest neighbor algorithm to perform distance browsing. In our tests (shown in the figures) we use two such methods: (1) Execute $k$ -NN each time we need a new neighbor. (2) Invoke $k$ -NN for every five neighbors. Thus,for example,in case (2) the cost of computing the ${11}^{\text{th }}$ through ${14}^{\text{th }}$ neighbors is the same as the cost of computing the ${15}^{th}$ neighbor (which requires invoking the $k$ -NN algorithm for $k = 5,{10}$ ,and 15). From the figures,it is clear that using the incremental nearest neighbor (INN) algorithm for distance browsing significantly outperforms simulating incremental access with the $k$ -NN algorithm. In fact, the difference quickly becomes an order of magnitude. The figures use a logarithmic scale for the $y$ -axis in order to bring out relative scale. Since the differences were so great, in order to simplify the presentation,we include results only for the $k$ -NN algorithm for the PG map because the results for the random data were similar.

图10至图12展示了通过逐步查找邻居来进行数据库距离浏览时，各项指标的累计成本。有多种方法可以使用$k$近邻算法进行距离浏览。在我们的测试中（如图所示），我们使用了两种这样的方法：（1）每次需要新的邻居时都执行$k$ - NN算法。（2）每查找五个邻居调用一次$k$ - NN算法。因此，例如，在情况（2）中，计算第${11}^{\text{th }}$到第${14}^{\text{th }}$个邻居的成本与计算第${15}^{th}$个邻居的成本相同（这需要对$k = 5,{10}$和15调用$k$ - NN算法）。从图中可以明显看出，使用增量近邻（INN）算法进行距离浏览的性能明显优于使用$k$ - NN算法模拟增量访问。实际上，这种差异很快就达到了一个数量级。为了突出相对比例，图中$y$轴使用了对数刻度。由于差异非常大，为了简化展示，我们仅包含了PG地图的$k$ - NN算法的结果，因为随机数据的结果与之相似。

<!-- Media -->

<!-- figureText: Dbject distance calculations (log) 1000 INN (PG) INN (R64K) $\mathrm{k} - \mathrm{{NN}},\mathrm{k} = 1,2,\ldots ,\left( \mathrm{{PG}}\right) \}  - \pi  -$ k-NN, k=5,10,... (PG) 15 20 25 Number of nearest neighbors 100 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_36.jpg?x=226&y=227&w=549&h=439&r=0"/>

Fig. 12. Cumulative object distance calculations for distance browsing.

图12. 距离浏览的累积对象距离计算。

<!-- figureText: 500 100 1000 Number of nearest neighbors (logscale) Execution time relative to INN (%) Prune ( 450 Restart ([4]) Prune (50) 400 Restart (50) 350 300 250 200 150 100 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_36.jpg?x=835&y=225&w=556&h=443&r=0"/>

Fig. 13. Execution time of $k$ -NN relative to that of INN when used for distance browsing when the $k$ -NN approach is made as good as possible.

图13. 当$k$ - NN方法尽可能优化时，用于距离浏览时$k$ - NN相对于INN的执行时间。

<!-- Media -->

The method we used above for choosing the value of $k$ when performing distance browsing with the $k$ -NN algorithm is not the best that we can do for larger values of $k$ . For example,it would be better to multiply $k$ by 2 each time the algorithm must be reinvoked. In addition,the $k$ -NN algorithm can be adapted to make it more suitable for distance browsing. In particular,after finding the $m$ nearest neighbors and determining that we must find the ${m}^{\prime } > m$ nearest neighbors,we can use the distance of the ${m}^{th}$ nearest neighbor as a minimum distance when the $k$ -NN algorithm is reinvoked with $k = {m}^{\prime }$ (actually, $k$ is set to ${m}^{\prime } - m$ ,since the $m$ nearest neighbors are excluded from the search). This minimum distance can be used to prune the search in much the same way as described in Section 4.5, using minimum distance in the INN algorithm. Some complications arise if other objects have the same distance from $q$ as the ${m}^{th}$ nearest neighbor. The best way to resolve this is to return all neighbors with that distance, which means that sometimes we obtain more neighbors than we requested. In Figure 13, we compare the execution time when using such an adapted $k$ -NN algorithm (labelled "Prune") for distance browsing to the execution time when using the INN algorithm. We also show the result for the unmodified algorithm, where we must restart the search from scratch when the $k$ -NN algorithm must be reinvoked (labelled "Restart"). We only show the results for the real-world data set (PG), as they were almost identical when using the random data set.

我们在使用$k$ - NN算法进行距离浏览时选择$k$值的上述方法，对于较大的$k$值来说并非最佳。例如，每次必须重新调用算法时，将$k$乘以2会更好。此外，可以对$k$ - NN算法进行调整，使其更适合距离浏览。具体来说，在找到第$m$个最近邻并确定需要找到第${m}^{\prime } > m$个最近邻后，当以$k = {m}^{\prime }$重新调用$k$ - NN算法时（实际上，$k$被设置为${m}^{\prime } - m$，因为第$m$个最近邻被排除在搜索之外），我们可以将第${m}^{th}$个最近邻的距离用作最小距离。这种最小距离可以像第4.5节中描述的那样，在INN算法中使用最小距离的方式来修剪搜索范围。如果其他对象与$q$的距离和第${m}^{th}$个最近邻相同，就会出现一些复杂情况。解决这个问题的最佳方法是返回所有具有该距离的邻居，这意味着有时我们获得的邻居数量会比请求的多。在图13中，我们比较了使用这种调整后的$k$ - NN算法（标记为“修剪”）进行距离浏览时的执行时间与使用INN算法时的执行时间。我们还展示了未修改算法的结果，即当必须重新调用$k$ - NN算法时，我们必须从头开始重新搜索（标记为“重启”）。我们仅展示了真实数据集（PG）的结果，因为使用随机数据集时结果几乎相同。

We use two different starting values for $k$ in Figure 13,namely 5 and 50 (shown in parentheses). Each time the $k$ -NN algorithm is reinvoked, $k$ is doubled. The figure shows that if the $k$ -NN algorithm has to be reinvoked at least once, it usually takes more than twice (and up to nearly five times) as long as the INN algorithm. Using the "Prune" variant of the $k$ -NN algorithm does not pay off unless a rather large number of neighbors is needed (over 100 or 200 in these experiments). The reason why this variant takes longer for a smaller number of neighbors is that not enough nodes get pruned to offset the cost of more node distance computations (for each node we must compute two distances, a minimum and a maximum, instead of just the minimum). Another observation is that the $k$ -NN approach is highly sensitive to the initial value of $k$ . Which initial value is better depends on how many neighbors we need (which we do not know in advance in distance browsing). The spikes on the curves occur where the $k$ -NN algorithm is reinvoked an additional time for higher values of $k$ ,and between a spike and the next low point,no more neighbors are computed. ${}^{4}$ The reason the slope of the curve decreases after each spike is that in the range from a spike to the next low point,the cost of the $k$ -NN approach remains constant (since no more neighbors are computed), while the cost of the INN approach increases gradually because we must compute additional neighbors. Note that the absolute low point on the two curves corresponds to the case where the number of neighbors needed happens to be equal to the initial value of $k$ ( 5 and 50,respectively). For those values of $k$ ,the $k$ -NN algorithm is not much slower than the INN algorithm (about 25% slower for $k = 5$ and ${14}\%$ slower for $k = {50}$ ).

在图13中，我们对$k$使用了两个不同的起始值，即5和50（括号内所示）。每次重新调用$k$ -近邻（NN）算法时，$k$的值都会加倍。该图显示，如果$k$ -NN算法至少需要重新调用一次，那么它通常比增量近邻（INN）算法耗时多两倍以上（最多接近五倍）。除非需要相当多的近邻（在这些实验中超过100或200个），否则使用$k$ -NN算法的“剪枝（Prune）”变体并不划算。对于较少数量的近邻，该变体耗时更长的原因是，被剪枝的节点数量不足以抵消更多节点距离计算的成本（对于每个节点，我们必须计算两个距离，即最小值和最大值，而不仅仅是最小值）。另一个观察结果是，$k$ -NN方法对$k$的初始值高度敏感。哪个初始值更好取决于我们需要多少近邻（在距离浏览中我们事先并不知道）。曲线上的尖峰出现在为更高的$k$值再次调用$k$ -NN算法的位置，并且在一个尖峰和下一个低点之间，不再计算更多的近邻。${}^{4}$每次尖峰后曲线斜率下降的原因是，在从一个尖峰到下一个低点的范围内，$k$ -NN方法的成本保持不变（因为不再计算更多近邻），而INN方法的成本逐渐增加，因为我们必须计算额外的近邻。请注意，两条曲线上的绝对低点对应于所需近邻数量恰好等于$k$的初始值（分别为5和50）的情况。对于这些$k$值，$k$ -NN算法并不比INN算法慢很多（对于$k = 5$慢约25%，对于$k = {50}$慢${14}\%$）。

### 6.2 Incremental Cost of Distance Browsing

### 6.2 距离浏览的增量成本

The results of the experiments conducted in Section 6.1 show the total cost of distance browsing after retrieving the ${k}^{th}$ neighbor. Using INN to implement each browsing step requires us to examine just one neighbor, regardless of how many browsing steps we already executed. In contrast, use of $k$ -NN for distance browsing requires us to examine $k + 1$ neighbors when $k$ browsing steps have already been executed. In this section we compare the two algorithms in terms of the cost of each browsing step (i.e., the incremental cost). This is shown in Figures 14 through 16. For the INN algorithm, the incremental cost can be seen to fluctuate somewhat, but it is always at least one order of magnitude less than the cost of the $k$ -NN algorithm once the first neighbor is obtained. Although not shown here, we found that this holds for all values of $k$ . Again,we use a logarithmic scale for the $y$ -axis,so that the fluctuation in the cost of the incremental algorithm can be seen more clearly.

6.1节中进行的实验结果显示了检索到${k}^{th}$个近邻后距离浏览的总成本。使用INN实现每个浏览步骤时，无论我们已经执行了多少个浏览步骤，我们只需要检查一个近邻。相比之下，使用$k$ -NN进行距离浏览时，当已经执行了$k$个浏览步骤，我们需要检查$k + 1$个近邻。在本节中，我们从每个浏览步骤的成本（即增量成本）方面比较这两种算法。这在图14至图16中展示。对于INN算法，可以看到增量成本有一定的波动，但一旦获得第一个近邻，它总是比$k$ -NN算法的成本至少小一个数量级。虽然这里没有展示，但我们发现这对于所有的$k$值都成立。同样，我们对$y$轴使用对数刻度，以便更清楚地看到增量算法成本的波动。

---

<!-- Footnote -->

${}^{4}$ There should be a spike at 5 neighbors for "Prune (5)," but it occurs at 6 neighbors instead. The reason is that, occasionally, when requesting the nearest five neighbors, the sixth nearest neighbor has the same distance as the fifth one,so the $k$ -NN algorithm does not need to be reinvoked when we want to obtain the sixth neighbor (the same is true for the second spike at 10 neighbors).

${}^{4}$ “剪枝（5）”在5个近邻处应该有一个尖峰，但实际上它出现在6个近邻处。原因是，偶尔在请求最近的5个近邻时，第6个最近邻与第5个最近邻的距离相同，因此当我们想要获取第6个近邻时，不需要重新调用$k$ -NN算法（在10个近邻处的第二个尖峰也是如此）。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Execution time (ms, log scale) 1 INN (PG) k-NN (PG) _____. INN (R64K) k-NN (R64K) 15 20 25 Number of nearest neighbors 0.1 0.01 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_38.jpg?x=226&y=229&w=551&h=438&r=0"/>

Fig. 14. Incremental execution times for distance browsing.

图14. 距离浏览的增量执行时间。

<!-- figureText: R-tree node disk I/Os (log scale) 10 INN (PG) k-NN (PG) INN (R64K) - 曰 -- k-NN (R64K) -> 15 20 25 Number of nearest neighbors 0.1 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_38.jpg?x=832&y=230&w=551&h=437&r=0"/>

Fig. 15. Incremental R-tree node disk I/O for distance browsing.

图15. 距离浏览的增量R树节点磁盘输入/输出。

<!-- Media -->

We evaluated the incremental execution time for up to 1000 neighbors in the PG map. Interestingly, we found that the incremental execution time clusters around an average of about ${.04}\mathrm{\;{ms}}$ after the first 100 neighbors or so. This is in agreement with the results that we will discuss in Section 6.3, where we find that the average execution time per neighbor is around .04 ms when retrieving a few thousand neighbors or more in the PG and R64K maps. Thus we see that for a given map, the incremental execution time is remarkably close to constant after a small fraction of the objects have been retrieved (for the PG map this is around 100 neighbors or less than 0.2% of the map size).

我们评估了PG地图中最多1000个近邻的增量执行时间。有趣的是，我们发现大约在前100个近邻之后，增量执行时间聚集在平均约${.04}\mathrm{\;{ms}}$左右。这与我们将在6.3节中讨论的结果一致，在那里我们发现，在PG和R64K地图中检索几千个或更多近邻时，每个近邻的平均执行时间约为0.04毫秒。因此我们看到，对于给定的地图，在检索了一小部分对象之后（对于PG地图，这大约是100个近邻，即小于地图大小的0.2%），增量执行时间非常接近常数。

For the R-tree node disk I/Os (Figure 15), the incremental algorithm (INN) was at least an order of magnitude better than $k$ -NN after the first neighbor has been found. INN appears to be decreasing (i.e., between . 1 and .2 after 25 neighbors), but levels off after a few hundred neighbors have been found. (The graph is not a step function because the number of node accesses is averaged over many queries.)

对于R树节点的磁盘输入/输出（图15），在找到第一个邻居后，增量算法（INN）至少比$k$ -最近邻算法好一个数量级。INN似乎在下降（即，在找到25个邻居后，介于0.1和0.2之间），但在找到几百个邻居后趋于平稳。（该图不是阶跃函数，因为节点访问次数是对多个查询求平均值得到的。）

For the object distance calculations (Figure 16), the incremental algorithm (INN) was at least an order of magnitude better than $k$ -NN after the first few neighbors had been found. The improvement approaches two orders of magnitude when 25 neighbors have been found, and continues in this manner for larger values of $k$ (not shown here). The average number of distance calculations performed for each incremental invocation is seen to be decreasing. This continues as more neighbors are retrieved and is below 1.2 after 300 neighbors. Thus INN quickly reaches a stage of accessing only about one object per reported neighbor.

对于对象距离计算（图16），在找到前几个邻居后，增量算法（INN）至少比$k$ -最近邻算法好一个数量级。当找到25个邻居时，这种改进接近两个数量级，并且对于更大的$k$值（此处未显示），这种情况会持续。可以看到，每次增量调用执行的平均距离计算次数在减少。随着检索到更多的邻居，这种情况会持续，在找到300个邻居后，该值低于1.2。因此，INN很快就会达到每个报告的邻居仅访问约一个对象的阶段。

## ${6.3k}$ -Nearest Neighbor Queries

## ${6.3k}$ -最近邻查询

We now consider what the cost would be if we used the incremental nearest neighbor algorithm to solve the $k$ -nearest neighbor problem. In other words, instead of browsing the database on the basis of distance, obtaining one neighbor at a time,we address the related problem of finding all $k$ neighbors at once, as we would do if we knew in advance how many neighbors we need. It is interesting to see if a performance penalty is incurred in solving this classical problem by using our incremental algorithm,rather than using approaches such as the $k$ -NN algorithm which obtain all $k$ neighbors at once. We ran a sequence of tests in the same manner as those reported in Sections 6.1 and 6.2; the results are shown in Figures 17 through 19. From these figures we observe that using the INN algorithm leads to no sacrifice of performance. In fact, the incremental algorithm outperforms the $k$ -nearest neighbor algorithm for the two maps for all values of $k$ .

现在我们考虑，如果使用增量最近邻算法来解决$k$ -最近邻问题，成本会是多少。换句话说，我们不是基于距离浏览数据库，一次获取一个邻居，而是解决一次性找到所有$k$个邻居的相关问题，就像我们事先知道需要多少个邻居时会做的那样。有趣的是，看看使用我们的增量算法来解决这个经典问题，而不是使用像$k$ -最近邻算法那样一次性获取所有$k$个邻居的方法，是否会产生性能损失。我们以与6.1节和6.2节中报告的相同方式进行了一系列测试；结果如图17至图19所示。从这些图中我们观察到，使用INN算法不会牺牲性能。事实上，对于两个地图，在所有$k$值的情况下，增量算法的性能都优于$k$ -最近邻算法。

<!-- Media -->

<!-- figureText: Object distance calculations (lo 00 INN (PG) k-NN (PG) INN (R64K) k-NN (R64K) 15 20 25 Number of nearest neighbors 10 1 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_39.jpg?x=239&y=230&w=539&h=437&r=0"/>

Fig. 16. Incremental object distance calculations for distance browsing.

图16. 用于距离浏览的增量对象距离计算。

<!-- figureText: Execution time (milliseconds) 5 INN (PG) -><---- 15 20 25 Number of nearest neighbors k-NN (PG) INN (R64K) 4.5 k-NN (R64K) 4 3.5 3 2.5 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_39.jpg?x=847&y=229&w=536&h=437&r=0"/>

Fig. 17. Execution time for $k$ -nearest neighbor query.

图17. $k$ -最近邻查询的执行时间。

<!-- Media -->

In addition to the experiments mentioned above,we ran $k$ -nearest neighbor queries for values of $k$ from 1 up to the size of the data set. The results of these experiments are reported in Figures 20 through 22, where the cost measures are divided by the number $k$ of nearest neighbors,so that we are reporting the cost per neighbor. For the incremental nearest neighbor algorithm, this value is close to the average incremental cost for all but the smallest values of $k$ (for small $k$ ,the cost of retrieving the first neighbor dominates the cost). Dividing the cost measures by $k$ makes it possible to distinguish the cost measures for large values of $k$ ,which is difficult otherwise. In Figures 20-22,the $y$ axis uses a logarithmic scale.

除了上述实验外，我们还对从1到数据集大小的$k$值进行了$k$ -最近邻查询。这些实验的结果如图20至图22所示，其中成本度量除以最近邻的数量$k$，这样我们报告的是每个邻居的成本。对于增量最近邻算法，除了最小的$k$值（对于小的$k$，检索第一个邻居的成本占主导地位）之外，该值接近平均增量成本。将成本度量除以$k$使得能够区分大$k$值的成本度量，否则这是很困难的。在图20 - 22中，$y$轴使用对数刻度。

For the execution time (Figure 17), we see that the two algorithms have similar growth patterns,with $k$ -NN somewhat slower than INN (about 11-14% for PG and 4-10% for R64K). While the improvement of INN over $k$ -NN is modest for values of $k$ up to 25,Figure 20 reveals that the difference widens as $k$ grows larger,up to ${75}\%$ for $\mathrm{{PG}}$ and ${87}\%$ for $\mathrm{{R64K}}$ (for $k = {2}^{15} = {32},{768}$ ). Even for values of $k$ as small as several hundred, the improvement of INN over $k$ -NN is ${20} - {30}\%$ . Note how the performance of INN for the two maps is very similar,whereas the performance of $k$ -NN is worse for the PG map than for the R64K map. This observation holds for the other two cost measures as well, suggesting that INN is much less sensitive than $k$ -NN to the distribution of data objects.

对于执行时间（图17），我们看到这两种算法具有相似的增长模式，$k$ -最近邻算法比INN稍慢（PG地图约慢11 - 14%，R64K地图约慢4 - 10%）。虽然在$k$值达到25之前，INN相对于$k$ -最近邻算法的改进并不显著，但图20显示，随着$k$值的增大，差异会扩大，对于$\mathrm{{PG}}$达到${75}\%$，对于$\mathrm{{R64K}}$达到${87}\%$（对于$k = {2}^{15} = {32},{768}$）。即使对于几百这样小的$k$值，INN相对于$k$ -最近邻算法的改进也达到${20} - {30}\%$。注意，INN在两个地图上的性能非常相似，而$k$ -最近邻算法在PG地图上的性能比在R64K地图上更差。这一观察结果对于其他两个成本度量也成立，这表明INN对数据对象分布的敏感性远低于$k$ -最近邻算法。

<!-- Media -->

<!-- figureText: 13 INN (PG) k-NN (PG) INN (R64K) k-NN (R64K) 15 20 25 Number of nearest neighbors R-tree node disk I/Os 12 11 10 9 8 7 6 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_40.jpg?x=225&y=227&w=553&h=440&r=0"/>

Fig. 18. R-tree node disk I/O for $k$ -nearest neighbor query.

图18. $k$ -最近邻查询的R树节点磁盘输入/输出。

<!-- figureText: Dbject distance calculations (170) INN (PG) k-NN (PG) INN (R64K) k-NN (R64K) 15 20 25 Number of nearest neighbors 120 100 80 60 40 5 10 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_40.jpg?x=848&y=230&w=536&h=437&r=0"/>

Fig. 19. Object distance calculations for $k$ -nearest neighbor query.

图19. $k$ -最近邻查询的对象距离计算。

<!-- Media -->

For very large values of $k$ ,we may ask whether it is not better to simply calculate distances for the entire database and then sort on the distance. If all the objects are ranked with the INN algorithm (or the $k$ -NN algorithm), we must also compute the distances for all the objects in the database. The question then reduces to whether the overhead of the INN algorithm (for computing distances of nodes and manipulating the priority queue) exceeds the cost of sorting all the distance values once they are computed. Interestingly, we found that for the PG map, using the INN algorithm to rank all the objects was faster than computing all the distances and sorting them, whereas the $k$ -NN algorithm was a little slower than the sorting approach. Of course, this result cannot be generalized, as it depends on numerous factors, such as size of the data set, the spatial index being used, and whether the spatial objects are stored directly in the leaf nodes of the $\mathrm{R}$ -tree or in an external object table.

对于$k$的非常大的值，我们可能会问，是否直接计算整个数据库的距离，然后按距离排序会更好。如果使用INN算法（或$k$ - 最近邻（NN）算法）对所有对象进行排序，我们还必须计算数据库中所有对象的距离。那么问题就归结为INN算法的开销（用于计算节点的距离和操作优先队列）是否超过了一旦计算出所有距离值后对其进行排序的成本。有趣的是，我们发现对于PG地图，使用INN算法对所有对象进行排序比计算所有距离并对其进行排序更快，而$k$ - NN算法比排序方法稍慢。当然，这个结果不能一概而论，因为它取决于许多因素，例如数据集的大小、所使用的空间索引，以及空间对象是直接存储在$\mathrm{R}$ - 树的叶节点中还是存储在外部对象表中。

For the R-tree node disk I/Os (Figure 18) we see that INN is always better than $k$ -NN,while the rate of growth is similar for both,and appears to be linear in $k$ for low values of $k$ . In fact,we found that this same pattern held for all values of $k$ ,as we see in Figure 21. The figures show that for each value of $k$ ,INN achieves more pruning of the input tree than $k$ -NN. This partially explains its better execution time performance. For values of $k$ ranging between ${2}^{6}$ and ${2}^{15}$ ,INN accesses ${20} - {53}\%$ fewer nodes for PG and ${12} - {35}\%$ for $\mathrm{R}{64}\mathrm{\;K}$ ,with the largest difference at $k - {2}^{9}$ for both maps.

对于R - 树节点磁盘输入/输出（图18），我们看到INN总是比$k$ - NN更好，而两者的增长率相似，并且在$k$值较低时，似乎与$k$呈线性关系。实际上，正如我们在图21中看到的，我们发现对于$k$的所有值都遵循相同的模式。这些图显示，对于$k$的每个值，INN比$k$ - NN对输入树进行了更多的剪枝。这部分解释了它更好的执行时间性能。当$k$的值在${2}^{6}$和${2}^{15}$之间时，对于PG地图，INN访问的节点数少${20} - {53}\%$个，对于$\mathrm{R}{64}\mathrm{\;K}$地图少${12} - {35}\%$个，在两种地图中，最大差异出现在$k - {2}^{9}$处。

For the object distance calculations (Figure 19), we see that the INN algorithm again outperforms the $k$ -NN algorithm. Figure 22 shows that this holds for all values of $k$ ,except when ranking all the map objects (in which case the number of distance calculations equals the number of map objects in both cases, as no pruning of objects or nodes is possible). The shapes of the curves in Figure 22 can be seen to be very similar to those in Figure 21. This is not surprising when we realize that the number of distance calculations is proportional to the number of R-tree leaf nodes that are accessed, and the leaf nodes in an R-tree greatly outnumber the nonleaf nodes.

对于对象距离计算（图19），我们看到INN算法再次优于$k$ - NN算法。图22显示，除了对所有地图对象进行排序的情况（在这种情况下，两种算法的距离计算次数都等于地图对象的数量，因为无法对对象或节点进行剪枝），对于$k$的所有值都是如此。可以看出图22中的曲线形状与图21中的非常相似。当我们意识到距离计算的次数与访问的R - 树叶节点的数量成正比，并且R - 树中的叶节点数量远远超过非叶节点时，这并不奇怪。

<!-- Media -->

<!-- figureText: Execution time (ms, logscale) 0.1 INN (PG) k-NN (PG) -+-- INN (R64K) - 曰 -- k-NN (R64K) 8 10 12 16 log2(Number of nearest neighbors) 0 2 4 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_41.jpg?x=231&y=229&w=536&h=439&r=0"/>

Fig. 20. Execution time per neighbor for $k$ -nearest neighbor query.

图20. $k$ - 最近邻查询每个邻居的执行时间。

<!-- figureText: R-tree node disk I/Os (logscale) 0.1 INN (PG) k-NN (PG) _____. INN (R64K) - 曰 -- k-NN (R64K) ->----- 8 10 12 14 16 log2(Number of nearest neighbors) 0 2 6 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_41.jpg?x=846&y=227&w=534&h=442&r=0"/>

Fig. 21. R-tree node disk I/O per neighbor for $k$ -nearest neighbor query.

图21. $k$ - 最近邻查询每个邻居的R - 树节点磁盘输入/输出。

<!-- Media -->

Figure 23 shows the fraction of total execution time attributed to disk I/O operations in the above experiments. We compute this by recording the node accesses performed during the execution of the algorithms and measuring the time needed to do nothing but access those nodes. The figure shows that the fraction of time spent by the INN algorithm in doing I/O is relatively constant, but starts to decrease for a large number of neighbors. In contrast,the fraction of time spent by the $k$ -NN algorithm in doing I/O has a much larger variation, initially increasing rapidly, and decreasing significantly as the number of neighbors needed increases. In fact, eventually the fraction of time spent in doing I/O by the $k$ -NN algorithm is considerably less than spent by the INN algorithm as the number of neighbors increases; thus the INN algorithm becomes more efficient from a CPU cost perspective. (This may be due, in part, to the fact that for a large number of neighbors, the priority queue for the INN algorithm is considerably smaller than the NearestList maintained by the $k$ -NN algorithm,as discussed in Section 6.6 and seen in Figure 31.)

图23显示了在上述实验中磁盘输入/输出操作占总执行时间的比例。我们通过记录算法执行期间执行的节点访问，并测量仅访问这些节点所需的时间来计算这个比例。该图显示，INN算法用于输入/输出的时间比例相对恒定，但对于大量邻居时开始下降。相比之下，$k$ - NN算法用于输入/输出的时间比例变化要大得多，最初迅速增加，并且随着所需邻居数量的增加而显著减少。实际上，最终随着邻居数量的增加，$k$ - NN算法用于输入/输出的时间比例比INN算法少得多；因此从CPU成本的角度来看，INN算法变得更高效。（这可能部分是因为，对于大量邻居，如第6.6节所讨论并在图31中所示，INN算法的优先队列比$k$ - NN算法维护的最近列表要小得多。）

### 6.4 Results for Varying Data Size

### 6.4 不同数据大小的结果

In the previous sections we investigated the performance of the two algorithms by varying the number of neighbors for both distance browsing and computing the $k$ nearest neighbors for similarly sized data sets. It is important that the performance of the algorithms remain reasonable, even when the size of the data set is increased. To verify that this is indeed the case,we tested the performance of INN and $k$ -NN on both random and real-world map data. Our experiments showed the same relationships for the two algorithms between the cumulative and incremental costs of distance browsing,as well as computing the $k$ nearest neighbors,that we found in the experiments reported in Sections 6.1-6.3 (provided the maps are nontrivial in size). In particular, they confirmed the superiority of INN over $k$ -NN. In the interest of saving space,we do not show these results here.

在前面的章节中，我们通过改变距离浏览和计算$k$近邻时的邻居数量，研究了这两种算法在规模相近的数据集上的性能。即使数据集规模增大，算法的性能仍保持合理，这一点很重要。为了验证实际情况确实如此，我们在随机地图数据和真实世界地图数据上测试了INN（增量近邻搜索算法）和$k$ -NN（$k$近邻算法）的性能。我们的实验表明，在距离浏览的累积成本和增量成本以及计算$k$近邻方面，这两种算法之间的关系与第6.1 - 6.3节报告的实验结果相同（前提是地图规模并非微不足道）。特别是，这些实验证实了INN相对于$k$ -NN的优越性。为了节省篇幅，我们在此不展示这些结果。

<!-- Media -->

<!-- figureText: Dbject distance calculations (logscale 10 INN (PG) k-NN (PG) INN (R64K) - 巴 - - k-NN (R64K) 8 10 12 14 16 log2(Number of nearest neighbors) 0 4 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_42.jpg?x=229&y=220&w=537&h=448&r=0"/>

Fig. 22. Object distance calculations per neighbor for $k$ -nearest neighbor query.

图22. $k$近邻查询中每个邻居的对象距离计算。

<!-- figureText: 40 日龄 -日 - - 8 10 12 14 16 log2(Number of nearest neighbors O cost relative to total cost (%) 35 30 25 20 INN (PG) 15 k-NN (PG) INN (R64K) 10 k-NN (R64K) 5 0 0 6 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_42.jpg?x=847&y=223&w=536&h=446&r=0"/>

Fig. 23. Fraction of total execution time taken by disk I/Os in computing $k$ -nearest neighbor query.

图23. 计算$k$近邻查询时磁盘I/O占用的总执行时间比例。

<!-- Media -->

In the rest of this section we focus on the relative behavior of the algorithms when finding the nearest neighbor (i.e., $k = 1$ ). This operation is important as it is the first step in distance browsing, and as we saw in Section 6.2, its execution time dominates the cost of distance browsing for small values of $k$ .

在本节的其余部分，我们将重点关注算法在寻找最近邻（即 $k = 1$ ）时的相对性能。这一操作非常重要，因为它是距离浏览的第一步，而且正如我们在6.2节中所见，对于较小的 $k$ 值，其执行时间主导了距离浏览的成本。

Figures 24 through 26 show the performance of the two algorithms when finding the nearest neighbor. The $x$ -axis in the figure is ${\log }_{2}N$ ,where $N$ is the number of line segments. The real-world maps appear in the same order in which they were described (from left to right: Howard County, Water, Prince George's County, and Roads). The random maps that we tested contained 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, and 256000 line segments.

图24至图26展示了两种算法在寻找最近邻时的性能。图中的 $x$ 轴为 ${\log }_{2}N$ ，其中 $N$ 是线段的数量。真实世界地图按照描述的顺序出现（从左到右依次为：霍华德县、水域、乔治王子县和道路）。我们测试的随机地图包含1000、2000、4000、8000、16000、32000、64000、128000和256000条线段。

For the execution time (Figure 24), we see that the INN algorithm is faster for most of the maps; $k$ -NN took from 10-19% more time for the real-world maps and up to ${14}\%$ more time for the randomly generated maps. The exceptions are the three smallest randomly generated maps. This can be explained partly by the fact that these maps were small enough to fit in the R-tree node buffer, and partly by the fact that their small sizes gave less room for improvement (see Figures 25 and 26). Even so, for larger values of $k$ ,INN became better than $k$ -NN for these data sets. For all the randomly generated maps, which have similar characteristics, the rate of growth of the execution time can be seen to be nearly identical for the two algorithms. In fact, the rate of growth appears to be very nearly logarithmic in the number of line segments (recall that the $x$ -axis uses a log scale). The execution times for the real-world maps correlate remarkably well with the execution times for the random maps of comparable size.

对于执行时间（图24），我们发现INN算法在大多数地图上更快； $k$ -NN算法在真实世界地图上多花费10 - 19%的时间，在随机生成的地图上最多多花费 ${14}\%$ 的时间。例外情况是三个最小的随机生成地图。部分原因是这些地图足够小，可以放入R树节点缓冲区，部分原因是它们的规模较小，改进空间有限（见图25和图26）。即便如此，对于较大的 $k$ 值，INN算法在这些数据集上的表现优于 $k$ -NN算法。对于所有具有相似特征的随机生成地图，两种算法的执行时间增长率几乎相同。实际上，执行时间的增长率似乎与线段数量近乎呈对数关系（回想一下， $x$ 轴使用对数刻度）。真实世界地图的执行时间与规模相当的随机地图的执行时间相关性非常好。

For the R-tree node disk I/Os (Figure 25), we find the same relative behavior of the algorithms,with INN always better than $k$ -NN,while the rate of growth is similar for both. The rate of growth appears to be logarithmic in the number of line segments. This compares with the results reported by Roussopoulos et al. [1995] for $k$ -NN,where it was observed that the number of R-tree node accesses grew linearly with the height of the tree. Our experiments are not in exact agreement with that observation, but asymptotically the two observations are equivalent, since in R-trees the height of the tree grows logarithmically with the number of objects.

对于R树节点磁盘输入/输出（图25），我们发现算法的相对性能相同，INN算法始终优于 $k$ -NN算法，而两者的增长率相似。增长率似乎与线段数量呈对数关系。这与Roussopoulos等人 [1995] 报告的 $k$ -NN算法的结果相比，他们观察到R树节点的访问次数与树的高度呈线性增长。我们的实验结果与该观察结果并不完全一致，但从渐近意义上讲，这两种观察结果是等价的，因为在R树中，树的高度与对象数量呈对数增长。

<!-- Media -->

<!-- figureText: 4 14 15 16 17 18 log2(Number of line segments) Execution time (milliseconds) INN (Real) 3.5 k-NN (Real) INN (Random) 3 k-NN (Random) 2.5 2 1.5 1 0.5 10 11 12 13 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_43.jpg?x=226&y=223&w=538&h=445&r=0"/>

Fig. 24. Execution time for finding one neighbor.

图24. 寻找一个邻居的执行时间。

<!-- figureText: 8 INN (Real) 14 15 16 17 18 log2(Number of line segments) k-NN (Real) R-tree node disk I/Os INN (Random) 7 k-NN (Random) 6 10 11 12 13 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_43.jpg?x=861&y=227&w=522&h=442&r=0"/>

Fig. 25. R-tree node disk I/O for finding one neighbor.

图25. 寻找一个邻居时的R树节点磁盘输入/输出。

<!-- figureText: 60 14 15 16 log2(Number of line segments) Object distance calculations INN (Real) 55 k-NN (Real) INN (Random) 50 k-NN (Random) 45 40 35 30 10 11 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_43.jpg?x=231&y=788&w=539&h=446&r=0"/>

Fig. 26. Object distance calculations for finding one neighbor.

图26. 寻找一个邻居时的对象距离计算。

<!-- figureText: Execution time (ms, logscale) 100000 INN 4 6 log(Number of nearest neighbors) 10000 1000 100 1 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_43.jpg?x=805&y=795&w=572&h=437&r=0"/>

Fig. 27. Execution time for a large data set.

图27. 大数据集的执行时间。

<!-- Media -->

For the object distance calculations (Figure 26), INN again performs better than $k$ -NN.

对于对象距离计算（图26），INN算法的表现再次优于 $k$ -NN算法。

### 6.5 Results for Large Data Sets

### 6.5 大数据集的结果

Admittedly, the data sets we used in the experiments above were moderate in size. For the largest data set, the spatial index occupies approximately 9 MB of disk space, which is small enough to fit into the main memory of most modern computers. Even so, in our experiments, we only used a small amount of main memory for buffers (128 nodes), and the size of the priority queue remained small compared to the data size $({100}\mathrm{{KB}}$ in the worst case for the experiments in Section 6.3, or about 3% of the size of the map files). Thus we believe that our results will also hold for larger data sets, i.e., data sets much larger than the size of main memory.

诚然，我们在上述实验中使用的数据集规模适中。对于最大的数据集，空间索引占用约9MB的磁盘空间，这足够小，可以放入大多数现代计算机的主内存中。即便如此，在我们的实验中，我们仅使用了少量的主内存作为缓冲区（128个节点），并且优先队列的大小与数据大小相比仍然较小（在6.3节的实验中，最坏情况下为 $({100}\mathrm{{KB}}$ ，约为地图文件大小的3%）。因此，我们相信我们的结果也适用于更大的数据集，即远大于主内存大小的数据集。

In order to verify this claim, we conducted an experiment with a randomly generated data set of 8 million lines. Because it was prohibitively slow to build an ${\mathrm{R}}^{ * }$ -tree for such a large data set,we built a Hilbert-packed R-tree [Kamel and Faloutsos 1993], which occupied almost 300 MB. We used the same level of fan-out (50) and the same amount of buffering (128 nodes) as in our previous experiments (though it might have been better to use larger fan-out and buffer sizes for such a large data set). Incidentally, we found that both algorithms performed more poorly with a Hilbert-packed R-tree than with an ${\mathrm{R}}^{ * }$ -tree for the same data set. This appears to be due to the greater amount of node overlap in the Hilbert-packed R-tree. The incremental nearest neighbor algorithm proved to be much less sensitive to the level of node overlap, due to its superior pruning of the R-tree nodes.

为了验证这一说法，我们使用一个随机生成的800万行数据集进行了实验。由于为如此大的数据集构建一棵${\mathrm{R}}^{ * }$ -树速度极慢，我们构建了一棵希尔伯特填充R树（Hilbert-packed R-tree）[Kamel和Faloutsos 1993]，它占用了近300 MB的空间。我们使用了与之前实验相同的扇出级别（50）和相同的缓冲区大小（128个节点）（尽管对于如此大的数据集，使用更大的扇出和缓冲区大小可能会更好）。顺便说一下，我们发现对于同一数据集，两种算法在希尔伯特填充R树上的性能都比在${\mathrm{R}}^{ * }$ -树上差。这似乎是由于希尔伯特填充R树中节点重叠的数量更多。由于增量最近邻算法对R树节点的剪枝效果更好，它被证明对节点重叠程度的敏感度要低得多。

Figures 27-29 show the results of our experiments on this large map, which consisted of $k$ -nearest neighbor queries for values of $k$ from 1 through the size of the data set ( 8 million). Unfortunately, we were not able to run the $k$ -NN algorithm for $k = 8$ million,as there was not enough memory to hold the neighbor list for 8 million neighbors. This is in contrast to the INN algorithm, where the priority queue contained at most about 83,000 elements,or about $1\%$ of the number of neighbors. The speedup in execution time for INN over $k$ -NN ranged from 1.8 to 5.8. $k$ -NN accessed from 1.8 to 5.3 times as many nodes and performed up to 6 times as many distance calculations as INN.

图27 - 29展示了我们在这个大型地图上的实验结果，该实验包括对$k$从1到数据集大小（800万）的$k$ -最近邻查询。不幸的是，我们无法对$k = 8$ 万运行$k$ -NN算法，因为没有足够的内存来保存800万个邻居的邻居列表。这与INN算法形成对比，在INN算法中，优先队列最多包含约83,000个元素，约为邻居数量的$1\%$ 。INN相对于$k$ -NN的执行时间加速比在1.8到5.8之间。$k$ -NN访问的节点数量是INN的1.8到5.3倍，执行的距离计算次数最多是INN的6倍。

### 6.6 Priority Queue Size

### 6.6 优先队列大小

In Section 4.8 we showed that in the worst case, all the data objects must be inserted into the priority queue when using the incremental nearest neighbor algorithm. In our experiments, however, we found that the priority queue remained modest in size. The size of the priority queue affects the performance of queue operations during the algorithm's execution. Also, a very large queue requires a disk-based implementation, thereby slowing the algorithm. However, in most applications the maximum queue size remains relatively modest, which permits using a memory-based data structure for the queue. For example, consider Figure 30, which shows the maximum size of the queue when computing the nearest neighbor (i.e., $k = 1$ ) using the same data sets as in Section 6.4. Notice that for the worst case situation, described above, in this first step of distance browsing for the given query object, all objects must be inserted into the queue before determining the nearest neighbor. From the figure it is evident that the maximum queue size grows remarkably slowly as the number of line segments increases. The results for the random maps suggest that this growth is logarithmic in the number of line segments.

在第4.8节中，我们表明在最坏的情况下，使用增量最近邻算法时，所有数据对象都必须插入到优先队列中。然而，在我们的实验中，我们发现优先队列的大小保持适中。优先队列的大小会影响算法执行期间队列操作的性能。此外，一个非常大的队列需要基于磁盘的实现，从而减慢算法速度。然而，在大多数应用中，最大队列大小仍然相对适中，这允许对队列使用基于内存的数据结构。例如，考虑图30，它展示了在使用与第6.4节相同的数据集计算最近邻（即$k = 1$ ）时队列的最大大小。请注意，对于上述最坏情况，在给定查询对象的距离浏览的第一步中，在确定最近邻之前，所有对象都必须插入到队列中。从图中可以明显看出，随着线段数量的增加，最大队列大小的增长非常缓慢。随机地图的结果表明，这种增长与线段数量呈对数关系。

<!-- Media -->

<!-- figureText: R-tree node disk I/Os (logscale) 100000 k-NN INN 3 4 5 6 7 log(Number of nearest neighbors) 10000 1000 100 0 1 2 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_45.jpg?x=226&y=225&w=554&h=429&r=0"/>

Fig. 28. Node disk I/Os for a large data set.

图28. 大型数据集的节点磁盘输入/输出。

<!-- figureText: bject distance calculations (logscale k-NN INN 3 4 6 7 log(Number of nearest neighbors) 1e+06 100000 10000 1000 0 1 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_45.jpg?x=821&y=222&w=556&h=433&r=0"/>

Fig. 29. Object distance calculations for a large data set.

图29. 大型数据集的对象距离计算。

<!-- Media -->

Figure 31 shows the maximum size of the priority queue when using the incremental nearest neighbor algorithm after $k$ distance browsing operations for the maps in Section 6.1 ( $k$ ranged from 1 up to the size of the map). In the figure,the $y$ -axis is logarithmic. We see that the maximum queue size $M$ grows extremely slowly. Note also that $M$ is relatively small (less than 5% in the worse case), in comparison with the sum of the number of data objects and R-tree nodes for the two comparably sized maps, which is $M$ ’s theoretical maximum. When $k$ reaches a value of ${2}^{10} \approx  {1000}$ ,the priority queue needed by the incremental nearest neighbor algorithm is smaller than the priority queue needed to store the sorted buffer for the $k$ -NN algorithm. A similar picture emerged for the large map used in Section 6.5, where the size of the priority queue was an even smaller fraction of the map size (1% in the worst case).

图31展示了在对第6.1节中的地图进行$k$ 次距离浏览操作后，使用增量最近邻算法时优先队列的最大大小（$k$ 的范围从1到地图的大小）。在图中，$y$ 轴是对数轴。我们看到最大队列大小$M$ 增长极其缓慢。还需注意的是，与两个大小相当的地图的数据对象数量和R树节点数量之和（这是$M$ 的理论最大值）相比，$M$ 相对较小（在最坏情况下小于5%）。当$k$ 达到${2}^{10} \approx  {1000}$ 时，增量最近邻算法所需的优先队列比$k$ -NN算法存储排序缓冲区所需的优先队列小。在第6.5节使用的大型地图中也出现了类似的情况，其中优先队列的大小在地图大小中所占的比例更小（最坏情况下为1%）。

## 7. HIGH-DIMENSIONAL SPACE

## 7. 高维空间

As already pointed out, the incremental nearest neighbor algorithm is independent of the dimensionality of the data objects, and is equally applicable to data embedded in low-dimensional and high-dimensional spaces. Unfortunately, it is difficult to effectively index high-dimensional data, and nearest neighbor search also becomes more costly. In this section we address some of the issues that arise. Because it is hard to reason about arbitrary data distributions, some of the conclusions we draw are based on uniformly-distributed data.

正如已经指出的，增量最近邻算法与数据对象的维度无关，同样适用于嵌入低维和高维空间的数据。不幸的是，对高维数据进行有效索引很困难，最近邻搜索的成本也会更高。在本节中，我们将讨论一些出现的问题。由于很难对任意数据分布进行推理，我们得出的一些结论是基于均匀分布的数据。

High-dimensional data arises in a number of current applications, including multimedia databases, data warehouses, and information retrieval. Such data is usually limited to points, but more general objects also arise [Berchtold et al. 1997]. As an example of an application that leads to high-dimensional data, color histograms have been used in image databases to allow searching for images with a specific color or with a combination of colors similar to some query image. The colors in an image are described by $d$ -dimensional vectors,in which each element encodes the intensity of a particular range of colors (e.g., by using RGB values). To compare the closeness of the sets of colors in two images, a complex distance function is used, involving matrix multiplication. Using that distance function, we can use a nearest neighbor search on an image database to find the image closest in color to some query image. The number of dimensions, $d$ ,for color histograms is typically 64,100,or 256. In other applications, the number of dimensions can be even higher (as much as several thousand).

高维数据出现在许多当前应用中，包括多媒体数据库、数据仓库和信息检索。此类数据通常局限于点，但也会出现更通用的对象[Berchtold等人，1997]。作为一个产生高维数据的应用示例，颜色直方图已被用于图像数据库，以允许搜索具有特定颜色或与某个查询图像颜色组合相似的图像。图像中的颜色由$d$维向量描述，其中每个元素编码特定颜色范围的强度（例如，使用RGB值）。为了比较两幅图像中颜色集的接近程度，使用了一个复杂的距离函数，涉及矩阵乘法。使用该距离函数，我们可以在图像数据库上进行最近邻搜索，以找到颜色与某个查询图像最接近的图像。颜色直方图的维数$d$通常为64、100或256。在其他应用中，维数可能更高（多达数千）。

<!-- Media -->

<!-- figureText: 500 14 15 16 17 18 log2(Number of line segments) Real 450 Random Priority queue items 400 350 300 250 200 150 10 11 12 13 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_46.jpg?x=226&y=222&w=551&h=446&r=0"/>

Fig. 30. Maximum queue size for finding the nearest neighbor (i.e., $k = 1$ ).

图30. 查找最近邻（即$k = 1$）时的最大队列大小。

<!-- figureText: Priority queue items (logscale) INN (PG) 8 10 12 14 16 log2(Number of nearest neighbors 10000 INN (R64K) k-NN 1000 100 10 1 0 2 6 -->

<img src="https://cdn.noedgeai.com/0195c918-72e5-7b1b-ae72-97b2d298d382_46.jpg?x=819&y=229&w=564&h=440&r=0"/>

Fig. 31. Maximum queue size for a wide range of $k$ .

图31. 大范围$k$下的最大队列大小。

<!-- Media -->

Most spatial indexing structures do not work very well for high dimensions. The R-tree, for example, has been found to degenerate for dimensions higher than 7 or so [Berchtold et al. 1996]. Specifically, even for range queries with small query windows, so many index pages must be read that reading them is more expensive than sequentially scanning the data. Several indexing structures have been proposed to address this issue; for example,the X-tree [Berchtold et al. 1996] and ${\mathrm{{LSD}}}^{\mathrm{h}}$ -tree [Henrich 1998], based on the R-tree and LSD-tree, respectively. However, even these often do not provide much speedup compared to sequential scan for dimensions above 20 or so. An approach often taken to speed up access to point data of very high dimension is to map the points into a space of lower dimension [Faloutsos and Lin 1995; Kanth et al. 1998], in which case we can use the incremental nearest neighbor algorithm on the lower-dimensional space. In order to guarantee the accuracy of the result, the output of the algorithm can be filtered based on the distances of the corresponding higher-dimensional points [Seidl and Kriegel 1998]. Another approach is to abandon the goal of indexing the data points based on space occupancy and instead use properties of the distance metric (see the discussion of the metric space model in Section 2). If a hierararchical index method based on distance (e.g., Brin [1995]; Ciaccia et al. [1997]; and Uhlmann [1991]) is employed, our algorithm is still applicable. In fact,the $k$ -nearest neighbor algorithm presented in Ciaccia et al. [1997] is similar to our algorithm, in that it uses a priority queue for nodes to guide the traversal of the index.

大多数空间索引结构在高维情况下效果不佳。例如，研究发现R树在维度高于7左右时会退化[Berchtold等人，1996]。具体而言，即使对于查询窗口较小的范围查询，也必须读取大量索引页，以至于读取这些索引页的成本比顺序扫描数据还要高。为了解决这个问题，已经提出了几种索引结构；例如，分别基于R树和LSD树的X树[Berchtold等人，1996]和${\mathrm{{LSD}}}^{\mathrm{h}}$树[Henrich，1998]。然而，对于20维以上的情况，与顺序扫描相比，即使是这些结构通常也不会带来太多的加速。一种常用于加速访问超高维点数据的方法是将这些点映射到低维空间[Faloutsos和Lin，1995；Kanth等人，1998]，在这种情况下，我们可以在低维空间上使用增量最近邻算法。为了保证结果的准确性，可以根据相应高维点的距离对算法的输出进行过滤[Seidl和Kriegel，1998]。另一种方法是放弃基于空间占用对数据点进行索引的目标，转而使用距离度量的属性（见第2节对度量空间模型的讨论）。如果采用基于距离的分层索引方法（例如，Brin[1995]；Ciaccia等人[1997]；以及Uhlmann[1991]），我们的算法仍然适用。事实上，Ciaccia等人[1997]提出的$k$ -最近邻算法与我们的算法类似，因为它使用节点的优先队列来引导索引的遍历。

If we use the Euclidean distance metric, the nearest neighbor search region (Section 4.6) is spherical. On the other hand, the node regions for most types of spatial index structures are hyperrectangular in shape, making nearest neighbor search more expensive as more points are accessed than is necessary. To see why this is true, consider that in two dimensions the areas of a square and a circle,both with radius $r$ ,are $4{r}^{2}$ and $\pi {r}^{2}$ ,respectively. Thus the ratio of the area of the circle to the area of the square is $\pi /4 \approx  {79}\%$ . In three dimensions the ratio of the volume of a sphere to the volume of a cube is about ${52}\%$ ,and in four dimensions the corresponding ratio for a hypersphere and hypercube is ${10}\%$ . In general, the ratio between the volume of a hypersphere and its circumscribed hypercube decreases exponentially with the number of dimensions. Intuitively, the reason for this is that the number of "corners" of the hypercube grows exponentially with dimension. This effect has a direct consequence for nearest neighbor search using the Euclidean distance metric. To see why, let us assume that we have uniformly distributed data points inside a hypercube of radius $r$ and a search region of radius $r$ centered inside the hypercube; the hypercube represents the smallest bounding box of the set of hyperrectangular leaf node regions that intersect the search region. Then the proportion of the data points inside the search region decreases exponentially with the number of dimensions; e.g., for four dimensions, about ${10}\%$ only are inside the search region. The large number of data points inside the hypercube but outside the search region represent wasted effort for a nearest neighbor search. In order to alleviate this effect, spatial index structures that use hyperspheres as node regions [White and Jain 1996b] have been proposed for use in nearest neighbor applications for higher dimensions. However, since this can lead to a much higher level of overlap between nodes than using hyperrectangles, a compromise is to use shapes formed by intersections of hyperspheres and hyperrectangles [Katayama and Satoh 1997], essentially smoothing out the corners of the hyperrectangles.

如果我们使用欧几里得距离度量，最近邻搜索区域（4.6节）是球形的。另一方面，大多数类型的空间索引结构的节点区域是超矩形的，这使得最近邻搜索成本更高，因为会访问比必要数量更多的点。为了理解为什么会这样，考虑在二维空间中，半径均为$r$的正方形和圆形的面积分别为$4{r}^{2}$和$\pi {r}^{2}$。因此，圆形面积与正方形面积之比为$\pi /4 \approx  {79}\%$。在三维空间中，球体体积与立方体体积之比约为${52}\%$，在四维空间中，超球体与超立方体的相应比例为${10}\%$。一般来说，超球体体积与其外接超立方体体积之比随维度数呈指数下降。直观地说，原因在于超立方体的“角”的数量随维度呈指数增长。这种效应对于使用欧几里得距离度量的最近邻搜索有直接影响。为了理解原因，让我们假设在半径为$r$的超立方体内有均匀分布的数据点，并且以超立方体中心为圆心、半径为$r$的搜索区域；该超立方体代表与搜索区域相交的一组超矩形叶节点区域的最小边界框。那么，搜索区域内的数据点比例随维度数呈指数下降；例如，在四维空间中，只有约${10}\%$的数据点在搜索区域内。超立方体内但在搜索区域外的大量数据点意味着最近邻搜索做了无用功。为了减轻这种影响，有人提出在高维最近邻应用中使用以超球体作为节点区域的空间索引结构[White和Jain 1996b]。然而，由于这可能导致节点之间的重叠程度比使用超矩形时高得多，一种折中的方法是使用超球体和超矩形相交形成的形状[Katayama和Satoh 1997]，本质上是将超矩形的角平滑化。

In Section 4.6 we pointed out that the objects on the priority queue are contained in the leaf nodes intersected by the boundary of the search region (and similarly for the nodes on the priority queue). As the number of dimensions grows, the ratio of the number leaf nodes intersected by the boundary of the search region to the number of leaf nodes intersected by the interior of the search region tends to grow. Thus the size of the priority queue also tends to grow with the number of dimensions. For uniformly distributed points spread evenly among the leaf nodes, where each leaf node covers about the same amount of space, it can be shown that this ratio grows exponentially with the number of dimensions. This is true even if both the search region and leaf node regions are hypercubes (i.e., if we use the Chessboard metric ${L}_{\infty }$ ). Of course,this is only of major significance when the number of desired neighbors is large, since the volume of the search region depends on the number of neighbors.

在4.6节中，我们指出优先队列中的对象包含在与搜索区域边界相交的叶节点中（优先队列中的节点情况类似）。随着维度数的增加，与搜索区域边界相交的叶节点数量与与搜索区域内部相交的叶节点数量之比趋于增大。因此，优先队列的大小也趋于随维度数增加。对于均匀分布在叶节点中的点，每个叶节点覆盖的空间大致相同，可以证明这个比例随维度数呈指数增长。即使搜索区域和叶节点区域都是超立方体（即如果我们使用棋盘距离度量${L}_{\infty }$），情况也是如此。当然，只有当所需邻居数量较大时，这才具有重要意义，因为搜索区域的体积取决于邻居数量。

Some of the problems arising from operating in high-dimensional spaces can be alleviated by relaxing the requirement that the nearest neighbors be computed exactly. Our goal is to report neighbors as quickly as possible. In the incremental nearest neighbor algorithm,when an object $o$ is slightly farther from the query object $q$ than a node $n$ ,the algorithm must process $n$ before reporting $o$ . As we have seen,in a high-dimensional space this may cause a lot of extra work. Instead,we can report $o$ as the next nearest neighbor if its distance from $q$ is not "much" larger than that of $n$ . In particular,suppose $o$ is the object on the priority queue closest to $q$ ,and $n$ is the node on the queue closest to $q$ . We propose reporting $o$ as the next (approximate) nearest neighbor if ${d}_{o}\left( {q,o}\right)  \leq  \left( {1 + \epsilon }\right) {d}_{n}\left( {q,n}\right)$ ,where $\epsilon$ is some nonnegative constant. This leads to a definition of approximate nearest neighbor that conforms to that in Arya et al. [1994]: if $r$ is the distance of the ${k}^{th}$ nearest neighbor,then the distances of the objects returned by an approximate $k$ -nearest neighbor search must be no larger than $\left( {1 + \epsilon }\right) r$ . Obviously,for $\epsilon  = 0$ we get the exact result,and the larger $\epsilon$ is,the less exact the result. The only change required to the incremental nearest neighbor algorithm to make it approximate in this sense is in the key used for nodes on the priority queue. Specifically,for a node $n$ we use $\left( {1 + \epsilon }\right) {d}_{n}\left( {q,n}\right)$ as a key,instead of ${d}_{n}\left( {q,n}\right)$ . In Arya et al. [1994], ${}^{5}$ it was found that a significant reduction in node accesses results from finding the $k$ approximate nearest neighbors,as opposed to the $k$ exact nearest neighbors. Moreover, with relatively high probability, the result is the same in the exact and approximate cases. For example, for approximate nearest neighbor search in 16 dimensions using $\epsilon  = 3$ (meaning that a ${300}\%$ relative error in distance is allowed),it was found [Arya et al. 1994] that speedup in execution time was on the order of 10 to 50 over exact nearest neighbor search,while the average relative error was only 10% and the true nearest neighbor was found almost half the time.

通过放宽必须精确计算最近邻的要求，可以缓解在高维空间中操作所产生的一些问题。我们的目标是尽快报告邻居。在增量最近邻算法中，当对象$o$与查询对象$q$的距离比节点$n$稍远时，算法必须先处理$n$才能报告$o$。正如我们所见，在高维空间中，这可能会导致大量额外的工作。相反，如果$o$与$q$的距离并不比$n$与$q$的距离“大很多”，我们可以将$o$报告为下一个最近邻。具体来说，假设$o$是优先队列中最接近$q$的对象，$n$是队列中最接近$q$的节点。如果${d}_{o}\left( {q,o}\right)  \leq  \left( {1 + \epsilon }\right) {d}_{n}\left( {q,n}\right)$，我们建议将$o$报告为下一个（近似）最近邻，其中$\epsilon$是某个非负常数。这引出了近似最近邻的定义，该定义与Arya等人[1994]中的定义一致：如果$r$是第${k}^{th}$个最近邻的距离，那么近似$k$ -最近邻搜索返回的对象的距离必须不大于$\left( {1 + \epsilon }\right) r$。显然，当$\epsilon  = 0$时，我们得到精确结果，并且$\epsilon$越大，结果越不精确。要使增量最近邻算法在这种意义上具有近似性，唯一需要改变的是优先队列中节点使用的键。具体来说，对于节点$n$，我们使用$\left( {1 + \epsilon }\right) {d}_{n}\left( {q,n}\right)$作为键，而不是${d}_{n}\left( {q,n}\right)$。在Arya等人[1994]的研究中，${}^{5}$发现，与寻找$k$个精确最近邻相比，寻找$k$个近似最近邻可以显著减少节点访问次数。此外，在相对较高的概率下，精确情况和近似情况的结果是相同的。例如，对于使用$\epsilon  = 3$（意味着允许${300}\%$的距离相对误差）的16维近似最近邻搜索，[Arya等人1994]发现，与精确最近邻搜索相比，执行时间的加速比约为10到50倍，而平均相对误差仅为10%，并且几乎有一半的时间能找到真正的最近邻。

## 8. CONCLUDING REMARKS

## 8. 结论

We presented a detailed comparison of two approaches to browsing spatial objects in an $\mathrm{R}$ -tree,on the basis of their distances from an arbitrary spatial query object. It was shown that an incremental algorithm (INN) significantly outperforms (in execution time, R-tree node disk I/O, and object distance calculations) a solution based on a $k$ -nearest neighbor algorithm(k - NN). This is true even when the $k - \mathrm{{NN}}$ approach was optimized for this application by carefully choosing the increments for $k$ and using previous search results for pruning when the $k$ -NN algorithm must be reinvoked. The incremental approach was also found to have superior performance when applied to the problem of computing the $k$ nearest neighbors of a given query object. Our experiments confirm that the INN algorithm achieves a higher level of pruning than the $k$ -NN algorithm. This is important because it reduces the amount of R-tree node disk I/O as well as the number of distance calculations, which, when combined, account for a major portion of execution time. Moreover, as the data sets became larger, the superiority of the INN algorithm became more pronounced

我们基于空间对象与任意空间查询对象的距离，对在$\mathrm{R}$ -树中浏览空间对象的两种方法进行了详细比较。结果表明，增量算法（INN）在执行时间、R -树节点磁盘I/O和对象距离计算方面明显优于基于$k$ -最近邻算法（k - NN）的解决方案。即使通过仔细选择$k$的增量，并在必须重新调用$k$ - NN算法时使用先前的搜索结果进行剪枝，对$k - \mathrm{{NN}}$方法进行了针对此应用的优化，情况依然如此。在计算给定查询对象的$k$个最近邻的问题上，增量方法也表现出更优的性能。我们的实验证实，INN算法比$k$ - NN算法实现了更高程度的剪枝。这很重要，因为它减少了R -树节点磁盘I/O的数量以及距离计算的次数，而这两者加起来占了执行时间的很大一部分。此外，随着数据集的增大，INN算法的优势更加明显。

---

<!-- Footnote -->

${}^{5}$ The algorithm described in Arya et al. [1994] is not incremental,but it accesses the same set of nodes as the incremental nearest neighbor algorithm modified as described above

${}^{5}$ Arya等人[1994]中描述的算法不是增量式的，但它访问的节点集与上述修改后的增量最近邻算法访问的节点集相同。

<!-- Footnote -->

---

The experimental results were in reasonably close agreement with our rudimentary anaysis of the INN algorithm, which predicts that the number of node accesses is $O\left( {k + k + \log N}\right)$ ,where $k$ is the number of neighbors and $N$ the size of the data set. The superior performance of our algorithm in the experimental study was perhaps not surprising, as we prove informally that at any step in its execution the incremental nearest neighbor algorithm is optimal with respect to the spatial data structure employed. From a practical standpoint, this means that a minimum number of nodes is visited in order to report each object. In other words,upon reporting the ${k}^{\text{th }}$ neighbor ${o}_{k}$ of the query object $q$ ,the algorithm has only accessed nodes that lie within a distance of $d\left( {q,{o}_{k}}\right)$ of $q$ . Our adaptation of the algorithm to the R-tree has the added benefit that a minimum number of objects is accessed, i.e., only objects whose minimum bounding rectangles lie within a distance of $d\left( {q,{o}_{k}}\right)$ of $q$ .

实验结果与我们对增量最近邻（INN）算法的初步分析相当吻合，该分析预测节点访问次数为$O\left( {k + k + \log N}\right)$，其中$k$是邻居数量，$N$是数据集大小。在实验研究中，我们的算法表现出色或许并不令人意外，因为我们非形式化地证明了，在其执行的任何步骤中，增量最近邻算法相对于所采用的空间数据结构而言是最优的。从实际角度来看，这意味着为了报告每个对象，访问的节点数量达到最少。换句话说，在报告查询对象$q$的第${k}^{\text{th }}$个邻居${o}_{k}$时，该算法仅访问了与$q$距离在$d\left( {q,{o}_{k}}\right)$以内的节点。我们将该算法应用于R树，还有一个额外的好处，即访问的对象数量达到最少，也就是说，仅访问其最小边界矩形与$q$的距离在$d\left( {q,{o}_{k}}\right)$以内的对象。

In the experiments in Section 6, we used an R-tree variant in which the spatial objects were stored directly in the leaf nodes of the R-tree. This is not always practical, especially for complex and variable-size objects such as polygons. The other alternative is to store the objects in an external file, in which case the leaf nodes store the bounding boxes of the spatial objects and pointers to the objects. We performed additional experiments where the maps used in Section 6 were stored in such an R-tree; we used the INN variant given in Figure $4.{}^{6}$ These experiments revealed an even larger advantage for the incremental nearest neighbor algorithm over the $k$ -nearest neighbor algorithm (typically over 50%). This is primarily because the INN algorithm accessed many fewer data objects (for calculating their distances from the query object) than the $k$ -NN algorithm. The $k$ -NN algorithm typically accessed 4-6 times as many objects as the INN algorithm for low values of $k$ ,and up to twice as many for values of $k$ as high as $5\%$ of the map size. Reducing the number of object accesses and object distance calculations when using the incremental algorithm has an even greater effect in reducing the execution time for more complex spatial objects (e.g., polygons).

在第6节的实验中，我们使用了一种R树变体，其中空间对象直接存储在R树的叶节点中。这并不总是可行的，特别是对于多边形等复杂且大小可变的对象。另一种选择是将对象存储在外部文件中，在这种情况下，叶节点存储空间对象的边界框以及指向这些对象的指针。我们进行了额外的实验，将第6节中使用的地图存储在这样的R树中；我们使用了图$4.{}^{6}$中给出的INN变体。这些实验表明，增量最近邻算法相对于$k$ -最近邻算法具有更大的优势（通常超过50%）。这主要是因为与$k$ -NN算法相比，INN算法访问的数据对象（用于计算它们与查询对象的距离）要少得多。对于较小的$k$值，$k$ -NN算法通常访问的对象数量是INN算法的4 - 6倍，当$k$的值高达地图大小的$5\%$时，访问的对象数量最多可达INN算法的两倍。在使用增量算法时，减少对象访问次数和对象距离计算次数，对于更复杂的空间对象（例如多边形），在减少执行时间方面会产生更大的影响。

In a worst-case scenario, all the leaf nodes in the spatial data structure must be accessed (see Figure 6 and the discussion in Section 4.8). In contrast to the incremental algorithm in Figure 3, the variant presented in Figure 4 for the R-tree implementation, where the spatial objects are stored external to the R-tree, alleviates the worst case described above by making use of bounding rectangles in leaf nodes, thereby enabling it to avoid accessing many data objects from disk. ${}^{7}$ In particular,in the original version of the algorithm, the spatial index was not assumed to have bounding rectangles, which meant that, for this worst case, all data objects had to be accessed from disk in order to measure their distances from the query object. The use of bounding rectangles stored in the tree leads to a considerably more efficient (and conceptually different) incremental algorithm for R-trees, in that bounding boxes can be used as pruning devices to reduce disk I/O for accessing spatial descriptions of objects.

在最坏的情况下，必须访问空间数据结构中的所有叶节点（见图6和第4.8节的讨论）。与图3中的增量算法不同，图4中给出的用于R树实现的变体（其中空间对象存储在R树外部），通过利用叶节点中的边界矩形，缓解了上述最坏情况，从而使其能够避免从磁盘访问许多数据对象。${}^{7}$特别是，在该算法的原始版本中，假设空间索引没有边界矩形，这意味着在这种最坏情况下，为了测量所有数据对象与查询对象的距离，必须从磁盘访问所有数据对象。使用存储在树中的边界矩形，为R树带来了一种效率更高（且概念上不同）的增量算法，因为边界框可以用作剪枝工具，以减少访问对象空间描述的磁盘I/O。

---

<!-- Footnote -->

${}^{6}$ We decided to report only those results of experiments where the spatial objects are stored in the leaf nodes rather than external to the R-tree. This was done, in part, because the organization of the external object storage has a large effect on the performance, and thus introduces an extra variable into the comparison of the algorithms.

${}^{6}$我们决定只报告那些空间对象存储在叶节点中而不是存储在R树外部的实验结果。这样做的部分原因是，外部对象存储的组织方式对性能有很大影响，因此会在算法比较中引入一个额外的变量。

<!-- Footnote -->

---

Future work involves comparing the behavior of the incremental nearest neighbor algorithm on different spatial data structures such as PMR quadtrees,R-trees,and ${\mathrm{R}}^{ + }$ -trees,as well as adapting the algorithm to other classes of index structures, such as distance-based indexes [Brin 1995; Ciaccia et al. 1997; Uhlmann 1991]. We also wish to further investigate the use of the algorithm with very large data sets and in high-dimensional spaces, where the priority queue may have to be stored on disk.

未来的工作包括比较增量最近邻算法在不同空间数据结构（如PMR四叉树、R树和${\mathrm{R}}^{ + }$ -树）上的行为，以及将该算法应用于其他类型的索引结构，如基于距离的索引[Brin 1995; Ciaccia等人1997; Uhlmann 1991]。我们还希望进一步研究该算法在非常大的数据集和高维空间中的应用，在这些情况下，优先队列可能必须存储在磁盘上。

## REFERENCES

## 参考文献

Aoki, P. M. 1998. Generalizing "search" in generalized search trees. In Proceedings of the 14th International Conference on Data Engineering (Orlando, FL, Feb.). IEEE Computer Society Press, Los Alamitos, CA, 380-389.

AREF, W. G. AND SAMET, H. 1992. Uniquely reporting spatial objects: Yet another operation for comparing spatial data structures. In Proceedings of the 5th Symposium on Spatial Data Handling (Charleston, SC, Aug.). 178-189.

AREF, W. G. AND SAMET, H. 1993. Estimating selectivity factors of spatial operations. In Optimization in Databases - Proceedings of the 5th International Workshop on Foundations of Models and Languages for Data and Objects (Aigen, Austria, Sept.). 31-40.

ARyA, S., Mount, D. M., NETANYAHU, N. S., SILVERMAN, R., AND Wu, A. Y. 1998. An optimal algorithm for approximate nearest neighbor searching fixed dimensions. J. ACM 45, 6, 891-923.

BECKER, L. AND GUTING, R. H. 1992. Rule-based optimization and query processing in an extensible geometric database system. ACM Trans. Database Syst. 17, 2 (June 1992), 247-303.

BECKMANN, N., KRIEGEL, H.-P., SCHNEIDER, R., AND SEEGER, B. 1990. The R*-tree: An efficient and robust access method for points and rectangles. In Proceedings of the ${1990}\mathrm{{ACM}}$ SIGMOD International Conference on Management of Data (SIGMOD '90, Atlantic City, NJ, May 23-25, 1990), H. Garcia-Molina, Ed. ACM Press, New York, NY, 322-331.

BENTLEY, J. L. 1975. Multidimensional binary search trees used for associative searching. Commun. ACM 18, 9 (Sept.), 509-517.

BERCHTOLD, S., BôHM, C., KEIM, D. A., AND KRIEGEL, H.-P. 1997. A cost model for nearest neighbor search in high-dimensional data space. In Proceedings of the 16th ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS '97, Tucson, AZ, May 12-14, 1997), A. Mendelzon and Z. M. Özsoyoglu, Eds. ACM Press, New York, NY, 78-86.

---

<!-- Footnote -->

${}^{7}$ Recall that we decided to report only those experiments in which the spatial objects are stored in the leaf nodes rather than external to the R-tree.

${}^{7}$回想一下，我们决定只报告那些空间对象存储在叶节点中而不是存储在R树外部的实验。

<!-- Footnote -->

---

BERCHTOLD, S., KEIM, D. A., AND KRIEGEL, H.-P. 1996. The X-tree: An index structure for high-dimensional data. In Proceedings of the 22nd International Conference on Very Large Data Bases (VLDB '96, Mumbai, India, Sept.). 28-39.

BERN, M. 1993. Approximate closest-point queries in high dimensions. Inf. Process. Lett. 45, 2 (Feb. 26, 1993), 95-99.

BRIN, S. 1995. Near neighbor search in large metric space. In Proceedings of the 21st International Conference on Very Large Data Bases (VLDB '95, Zurich, Sept.). 574-584.

Broder, A. J. 1990. Strategies for efficient incremental nearest neighbor search. Pattern Recogn. 23, 1/2 (Jan. 1990), 171-178.

BURKHARD, W. A. AND KELLER, R. 1973. Some approaches to best-match file searching. Commun. ACM 16, 4 (Apr.), 230-236.

CIACCIA, P., PATELLA, M., AND ZEZULA, P. 1997. M-tree: An efficient access method for similarity search in metric spaces. In Proceedings of the 23rd International Conference on Very Large Data Bases (VLDB '97, Athens, Greece, Aug.). 426-435.

Comer, D. 1979. The ubiquitous B-tree. ACM Comput. Surv. 11, 2 (June), 121-137.

EASTMAN, C. M. AND ZEMANKOVA, M. 1982. Partially specified nearest neighbor searches using k-d-trees. Inf. Process. Lett. 15, 2 (Sept.), 53-56.

ESPERANÇA, C. AND SAMET, H. 1997. Orthogonal polygons as bounding structures in filter-refine query processing strategies. In Proceedings of the Fifth International Symposium on Advances in Spatial Databases (SSD'97, Berlin, July), M. Scholl and A. Voisard, Eds. Springer-Verlag, New York, 197-220.

FALOUTSOS, C. AND LIN, K. 1995. FastMap: A fast algorithm for indexing, data-mining and visualization of traditional and multimedia datasets. In Proceedings of the ACM SIGMOD Conference on Management of Data (San Jose, CA, May). ACM Press, New York, NY, 163-174.

FRANK, A. U. AND BARRERA, R. 1989. The Fieldtree: a data structure for geographic information systems. In Proceedings of the First Symposium on Design and Implementation of Large Spatial Databases (SSD'89, Santa Barbara, CA, July), A. Buchmann, O. Günther, T. R. Smith, and Y. F. Wang, Eds. Springer-Verlag, New York, 29-44.

FREDMAN, M. L., SEDGEWICK, R., SLEATOR, D. D., AND TARJAN, R. E. 1986. The pairing heap: a new form of self-adjusting heap. Algorithmica 1, 1 (Jan. 1986), 111-129.

FRIEDMAN, J. H., BENTLEY, J. L., AND FINKEL,, R. A. 1977. An algorithm for finding best matches in logarithmic expected time. ACM Trans. Math. Softw. 3, 3 (Sept.), 209-226.

Fukunaga, K. AND NarenDRA, P. M. 1975. A branch and bound algorithm for computing. IEEE Trans. Comput. 24, 7 (July), 750-753.

GUNTHER, O. AND NOLTEMEIER, H. 1991. Spatial database indices for large extended objects. In Proceedings of the Seventh International Conference on Data Engineering (Kobe, Japan). IEEE Computer Society Press, Los Alamitos, CA, 520-526.

GUTTMAN, A. 1984. R-trees: A dynamic index structure for spatial searching. In Proceedings of the ACM SIGMOD Annual Meeting on Management of Data (SIGMOD '84, Boston, MA, June18-21). ACM, New York, NY, 47-57.

HAFNER, J., SAWHNEY, H., AND EQUITZ, W. E. AL. 1995. Efficient color histogram indexing for quadratic form distance functions. IEEE Trans. Pattern Anal. Mach. Intell. 17, 7 (July), 729-736.

HENRICH, A. 1994. A distance-scan algorithm for spatial access structures. In Proceedings of the Second ACM Workshop on Geographic Information Systems (Gaithersburg, MD, Dec.). ACM Press, New York, NY, 136-143.

HENRICH,A. 1998. The ${\mathrm{{LSD}}}^{h}$ -tree: an access structure for feature vectors. In Proceedings of the 14th International Conference on Data Engineering (Orlando, FL, Feb.). IEEE Computer Society Press, Los Alamitos, CA, 362-369.

HENRICH, A., Six, H.-W., AND WIDMAYER, P. 1989. The LSD tree: spatial access to multidimensional and non-point objects. In Proceedings of the 15th International Conference on Very Large Data Bases (VLDB '89, Amsterdam, The Netherlands, Aug 22-25), R. P. van de Riet, Ed. Morgan Kaufmann Publishers Inc., San Francisco, CA, 45-53.

HJALTASON, G. R. AND SAMET, H. 1995. Ranking in spatial databases. In Proceedings of the Fourth International Symposium on Advances in Spatial Databases (SSD'95, Portland, ME, Aug.), M. J. Egenhofer and J. R. Herring, Eds. Springer-Verlag, New York, 83-95.

HOEL, E. G. AND SAMET, H. 1991. Efficient processing of spatial queries in line segment databases. In Proceedings of the 2nd symposium on Advances in Spatial Databases (SSD '91, Zurich, Switzerland, Aug. 28-30, 1991), O. Günther and H.-J. Schek, Eds. Springer Lecture Notes in Computer Science, vol. 525. Springer-Verlag, New York, NY, 237-256.

Kamel, I. AND FALOUTSOS, C. 1993. On packing R-trees. In Proceedings of the Second International Conference on Information and Knowledge Management (CIKM '93, Washington, DC, Nov. 1-5 1993), B. Bhargava, T. Finin, and Y. Yesha, Eds. ACM Press, New York, NY, 490-499.

KAMEL, I. AND FALOUTSOS, C. 1994. Hilbert R-tree: An improved R-tree using fractals. In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB'94, Santiago, Chile, Sept.). VLDB Endowment, Berkeley, CA, 500-509.

KAMGAR-PARSI, B. AND KANAL, L. N. 1985. An improved branch and bound algorithm for computing $k$ -nearest neighbors. Pattern Recogn. Lett. 3,1 (Jan.).

KANTH, K. V. R., AGRAWAL, D., AND SINGH, A. 1998. Dimensionality reduction for similarity searching in dynamic databases. In Proceedings of ACM SIGMOD International Conference on Management of Data (SIGMOD '98, Seattle, WA, June 1-4, 1998), L. Haas, P. Drew, A. Tiwary, and M. Franklin, Eds. ACM Press, New York, NY, 237-248.

KATAYAMA, N. AND SATOH, S. 1997. The SR-tree: an index structure for high-dimensional nearest neighbor queries. In Proceedings of the International ACM Conference on Management of Data (SIGMOD '97, May). ACM, New York, NY, 369-380.

Korn, F., SIDIROPOULOS, N., FALOUTSOS, C., SIEGEL, E., AND PROTOPAPAS, Z. 1996. Fast nearest neighbor search in medical image databases. In Proceedings of the 22nd International Conference on Very Large Data Bases (VLDB '96, Mumbai, India, Sept.). 215-226.

KRIEGEL, H.-P., Schmidt, T., AND SEIDL, T. 1997. 3-D similarity search by shape approximation. In Proceedings of the Fifth International Symposium on Advances in Spatial Databases (SSD'97, Berlin, July), M. Scholl and A. Voisard, Eds. Springer-Verlag, New York, 11-28.

LINDENBAUM, M. AND SAMET, H. 1995. A probabilistic analysis of trie-based sorting of large collections of line segments. TR-3455. University of Maryland at College Park, College Park, MD.

LOMET, D. AND SALZBERG, B. 1989. A robust multi-attribute search structure. In Proceedings of the Fifth IEEE International Conference on Data Engineering (Los Angeles, CA, Feb. 1989). 296-304.

MurALIKRISHNA, M. AND DEWITT, D. J. 1988. Equi-depth multidimensional histograms. In Proceedings of the Conference on Management of Data (SIGMOD '88, Chicago, IL, June 1-3, 1988), H. Boral and P.-A. Larson, Eds. ACM Press, New York, NY, 28-36.

MurPHY, O J AND SELKOW, S M 1986. The efficiency of using k-d trees for finding nearest neighbors in discrete space. Inf. Process. Lett. 23, 4 (Nov. 8, 1986), 215-218.

NELSON, R. C AND SAMET, H. 1986. A consistent hierarchical representation for vector data. SIGGRAPH Comput. Graph. 20, 4 (Aug. 1986), 197-206.

BUREAU OF THE CENSUS, 1989. Tiger/Line precensus files. Bureau of the Census, Washington DC.

ROBINSON, J. T. 1981. The k-d-b-tree: A search structure for large multidimensional dynamic indexes. In Proceedings of the ACM SIGMOD 1981 International Conference on Management of Data (Ann Arbor, MI, Apr. 29-May 1). ACM Press, New York, NY, 10-18.

Roussopoulos, N., Kelley, S., AND VINCENT, F. 1995. Nearest neighbor queries. In Proceedings of the 1995 ACM SIGMOD International Conference on Management of Data (SIGMOD '95, San Jose, CA, May 23-25), M. Carey and D. Schneider, Eds. ACM Press, New York, NY, 71-79.

ROUSSOPOULOS, N. AND LEIFKER, D. 1985. Direct spatial search on pictorial databases using packed R-trees. In Proceedings of the ACM SIGMOD Conference on Management of Data (SIGMOD, Austin, TX, May). ACM Press, New York, NY, 17-31.

SAMET, H. 1990. The Design and Analysis of Spatial Data Structures. Addison-Wesley Series in Computer Science. Addison-Wesley Longman Publ. Co., Inc., Reading, MA.

SEIDL, T. AND KRIEGEL, H.-P. 1998. Optimal multi-step k-nearest neighbor search. In Proceedings of ACM SIGMOD International Conference on Management of Data (SIGMOD '98, Seattle, WA, June 1-4, 1998), L. Haas, P. Drew, A. Tiwary, and M. Franklin, Eds. ACM Press, New York, NY, 154-165.

SELINGER, P. G., ASTRAHAN, M. M., LORIE, R. A., AND PRICE, T. G. 1979. Access path selection in a relational database management system. In Proceedings of ACM SIGMOD International Conference on Management of Data (SIGMOD '79, Boston, MA, May 30-June 1). ACM Press, New York, NY, 23-34.

SELLIS,T.,RoussopouLos,N.,AND FALOUTSOS,C. 1987. The ${\mathrm{R}}^{ + }$ -tree: A dynamic index for multi-dimensional objects. In Proceedings of the 13th International Conference on Very Large Data Bases (Brighton, UK, Sept.). 71-79.

SProulL, R. F. 1991. Refinements to nearest-neighbor searching in k-dimensional trees. Algorithmica 6, 4, 579-589.

UHLMANN, J. K. 1991. Satisfying general proximity/similarity queries with metric trees. Inf. Process. Lett. 40, 4 (Nov.), 175-179.

WANG, T. L. AND SHASHA, D. 1990. Query processing for distance metrics. In Proceedings of the 16th VLDB Conference on Very Large Data Bases (VLDB, Brisbane, Australia). VLDB Endowment, Berkeley, CA, 602-613.

WHITE, D. A. AND JAIN, R. 1996. Algorithms and strategies for similarity retrieval. Tech. Rep. VCL-96-101. University of California at San Diego, La Jolla, CA.

WHITE, D. A. AND JAIN, R. 1996. Similarity indexing with the SS-tree. In Proceedings of the 12th IEEE International Conference on Data Engineering (New Orleans, LA). IEEE Press, Piscataway, NJ, 516-523.