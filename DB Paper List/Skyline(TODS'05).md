# Progressive Skyline Computation in Database Systems

# 数据库系统中的渐进式天际线计算

DIMITRIS PAPADIAS

迪米特里斯·帕帕迪亚斯（DIMITRIS PAPADIAS）

Hong Kong University of Science and Technology

香港科技大学

YUFEI TAO

City University of Hong Kong

香港城市大学

GREG FU

JP Morgan Chase

摩根大通银行

and

和

BERNHARD SEEGER

伯恩哈德·西格（BERNHARD SEEGER）

Philipps University

菲利普斯大学

The skyline of a $d$ -dimensional dataset contains the points that are not dominated by any other point on all dimensions. Skyline computation has recently received considerable attention in the database community, especially for progressive methods that can quickly return the initial results without reading the entire database. All the existing algorithms, however, have some serious shortcomings which limit their applicability in practice. In this article we develop branch-and-bound skyline (BBS), an algorithm based on nearest-neighbor search, which is I/O optimal, that is, it performs a single access only to those nodes that may contain skyline points. BBS is simple to implement and supports all types of progressive processing (e.g., user preferences, arbitrary dimensionality, etc). Furthermore, we propose several interesting variations of skyline computation, and show how BBS can be applied for their efficient processing.

一个 $d$ 维数据集的天际线（skyline）包含那些在所有维度上都不被其他任何点所支配的点。最近，天际线计算在数据库领域受到了相当多的关注，特别是对于那些无需读取整个数据库就能快速返回初始结果的渐进式方法。然而，所有现有的算法都存在一些严重的缺点，这限制了它们在实际中的应用。在本文中，我们开发了基于最近邻搜索的分支限界天际线算法（branch-and-bound skyline，BBS），该算法具有最优的输入/输出（I/O）性能，即它仅对那些可能包含天际线点的节点进行单次访问。BBS 易于实现，并且支持所有类型的渐进式处理（例如，用户偏好、任意维度等）。此外，我们提出了几种有趣的天际线计算变体，并展示了如何应用 BBS 对其进行高效处理。

Categories and Subject Descriptors: H.2 [Database Management]; H.3.3 [Information Storage and Retrieval]: Information Search and Retrieval

分类与主题描述符：H.2 [数据库管理]；H.3.3 [信息存储与检索]：信息搜索与检索

General Terms: Algorithms, Experimentation

通用术语：算法、实验

Additional Key Words and Phrases: Skyline query, branch-and-bound algorithms, multidimensional access methods

其他关键词和短语：天际线查询、分支限界算法、多维访问方法

This research was supported by the grants HKUST 6180/03E and CityU 1163/04E from Hong Kong RGC and Se ${553}/3 - 1$ from DFG.

本研究得到了香港研究资助局（RGC）的 HKUST 6180/03E 和 CityU 1163/04E 资助，以及德国研究基金会（DFG）的 Se ${553}/3 - 1$ 资助。

Authors' addresses: D. Papadias, Department of Computer Science, Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong; email: dimitris@cs.ust.hk; Y. Tao, Department of Computer Science, City University of Hong Kong, Tat Chee Avenue, Hong Kong; email: taoyf@cs.cityu.edu.hk; G. Fu, JP Morgan Chase, 277 Park Avenue, New York, NY 10172-0002; email: gregory.c.fu@jpmchase.com; B. Seeger, Department of Mathematics and Computer Science, Philipps University, Hans-Meerwein-Strasse, Marburg, Germany 35032; email: seeger@mathematik.uni-marburg.de.

作者地址：D. 帕帕迪亚斯，香港科技大学计算机科学系，香港清水湾；电子邮件：dimitris@cs.ust.hk；Y. 陶，香港城市大学计算机科学系，香港达之路；电子邮件：taoyf@cs.cityu.edu.hk；G. 傅，摩根大通银行，纽约公园大道 277 号，纽约州 10172 - 0002；电子邮件：gregory.c.fu@jpmchase.com；B. 西格，菲利普斯大学数学与计算机科学系，德国马尔堡汉斯 - 米尔魏因大街，邮编 35032；电子邮件：seeger@mathematik.uni - marburg.de。

Permission to make digital/hard copy of part or all of this work for personal or classroom use is granted without fee provided that the copies are not made or distributed for profit or commercial advantage, the copyright notice, the title of the publication, and its date appear, and notice is given that copying is by permission of ACM, Inc. To copy otherwise, to republish, to post on servers, or to redistribute to lists requires prior specific permission and/or a fee. © 2005 ACM 0362-5915/05/0300-0041 \$5.00

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，必须保留版权声明、出版物标题及其日期，并注明复制获得了美国计算机协会（ACM）的许可。否则，如需复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。© 2005 ACM 0362 - 5915/05/0300 - 0041 5 美元

<!-- Media -->

<!-- figureText: price distance ${y}_{10}$ 9 9 8 3 $m$ -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_1.jpg?x=608&y=319&w=484&h=389&r=0"/>

Fig. 1. Example dataset and skyline.

图 1. 示例数据集和天际线。

<!-- Media -->

## 1. INTRODUCTION

## 1. 引言

The skyline operator is important for several applications involving multicrite-ria decision making. Given a set of objects ${p}_{1},{p}_{2},\ldots ,{p}_{N}$ ,the operator returns all objects ${p}_{i}$ such that ${p}_{i}$ is not dominated by another object ${p}_{j}$ . Using the common example in the literature, assume in Figure 1 that we have a set of hotels and for each hotel we store its distance from the beach ( $x$ axis) and its price ( $y$ axis). The most interesting hotels are $a,i$ ,and $k$ ,for which there is no point that is better in both dimensions. Borzsonyi et al. [2001] proposed an SQL syntax for the skyline operator, according to which the above query would be expressed as: [Select ${}^{ * }$ ,From Hotels,Skyline of Price min,Distance min],where min indicates that the price and the distance attributes should be minimized. The syntax can also capture different conditions (such as max), joins, group-by, and so on.

天际线运算符（skyline operator）对于涉及多标准决策的多个应用程序非常重要。给定一组对象 ${p}_{1},{p}_{2},\ldots ,{p}_{N}$ ，该运算符返回所有对象 ${p}_{i}$ ，使得 ${p}_{i}$ 不被另一个对象 ${p}_{j}$ 所支配。使用文献中常见的示例，假设在图 1 中我们有一组酒店，并且对于每个酒店，我们存储其与海滩的距离（ $x$ 轴）及其价格（ $y$ 轴）。最值得关注的酒店是 $a,i$ 和 $k$ ，对于这些酒店，不存在在两个维度上都更优的点。Borzsonyi 等人 [2001] 为天际线运算符提出了一种 SQL 语法，根据该语法，上述查询可以表示为：[Select ${}^{ * }$ ,From Hotels,Skyline of Price min,Distance min]，其中 min 表示价格和距离属性应被最小化。该语法还可以涵盖不同的条件（如 max）、连接、分组等。

For simplicity, we assume that skylines are computed with respect to ${min}$ conditions on all dimensions; however, all methods discussed can be applied with any combination of conditions. Using the min condition,a point ${p}_{i}$ dominates ${}^{1}$ another point ${p}_{j}$ if and only if the coordinate of ${p}_{i}$ on any axis is not larger than the corresponding coordinate of ${p}_{j}$ . Informally,this implies that ${p}_{i}$ is preferable to ${p}_{j}$ according to any preference (scoring) function which is monotone on all attributes. For instance,hotel $a$ in Figure 1 is better than hotels $b$ and $e$ since it is closer to the beach and cheaper (independently of the relative importance of the distance and price attributes). Furthermore,for every point $p$ in the skyline there exists a monotone function $f$ such that $p$ minimizes $f$ [Borzsonyi et al. 2001].

为了简单起见，我们假设天际线是根据所有维度上的 ${min}$ 条件计算的；然而，所讨论的所有方法都可以应用于任何条件组合。使用最小条件，当且仅当点 ${p}_{i}$ 在任何轴上的坐标不大于点 ${p}_{j}$ 的相应坐标时，点 ${p}_{i}$ 支配点 ${p}_{j}$ 。通俗地说，这意味着根据在所有属性上单调的任何偏好（评分）函数，${p}_{i}$ 比 ${p}_{j}$ 更受青睐。例如，图 1 中的酒店 $a$ 比酒店 $b$ 和 $e$ 更好，因为它离海滩更近且价格更便宜（与距离和价格属性的相对重要性无关）。此外，对于天际线中的每个点 $p$ ，都存在一个单调函数 $f$ ，使得 $p$ 使 $f$ 最小化 [Borzsonyi 等人 2001]。

Skylines are related to several other well-known problems, including convex hulls, top-K queries, and nearest-neighbor search. In particular, the convex hull contains the subset of skyline points that may be optimal only for linear preference functions (as opposed to any monotone function). Böhm and Kriegel [2001] proposed an algorithm for convex hulls, which applies branch-and-bound search on datasets indexed by R-trees. In addition, several main-memory algorithms have been proposed for the case that the whole dataset fits in memory [Preparata and Shamos 1985].

天际线与其他几个著名的问题相关，包括凸包（convex hulls）、前 K 个查询（top - K queries）和最近邻搜索（nearest - neighbor search）。特别是，凸包包含的天际线点子集可能仅对于线性偏好函数是最优的（与任何单调函数相反）。Böhm 和 Kriegel [2001] 提出了一种用于凸包的算法，该算法对由 R 树索引的数据集应用分支限界搜索。此外，对于整个数据集可以放入内存的情况，已经提出了几种主存算法 [Preparata 和 Shamos 1985]。

---

<!-- Footnote -->

${}^{1}$ According to this definition,two or more points with the same coordinates can be part of the skyline.

${}^{1}$ 根据这个定义，具有相同坐标的两个或多个点可以是天际线的一部分。

<!-- Footnote -->

---

Top- $K$ (or ranked) queries retrieve the best $K$ objects that minimize a specific preference function. As an example,given the preference function $f\left( {x,y}\right)  =$ $x + y$ ,the top-3 query,for the dataset in Figure 1,retrieves $< i,5 > , < h,7 >$ , $< m,8 >$ (in this order),where the number with each point indicates its score. The difference from skyline queries is that the output changes according to the input function and the retrieved points are not guaranteed to be part of the skyline ( $h$ and $m$ are dominated by $i$ ). Database techniques for top- $K$ queries include Prefer [Hristidis et al. 2001] and Onion [Chang et al. 2000], which are based on prematerialization and convex hulls, respectively. Several methods have been proposed for combining the results of multiple top- $K$ queries [Fagin et al. 2001; Natsev et al. 2001].

前 $K$ 个（或排名）查询检索使特定偏好函数最小化的最佳 $K$ 个对象。例如，给定偏好函数 $f\left( {x,y}\right)  =$ $x + y$ ，对于图 1 中的数据集，前 3 个查询检索 $< i,5 > , < h,7 >$ 、 $< m,8 >$ （按此顺序），其中每个点旁边的数字表示其得分。与天际线查询的区别在于，输出会根据输入函数而变化，并且检索到的点不一定是天际线的一部分（ $h$ 和 $m$ 被 $i$ 支配）。用于前 $K$ 个查询的数据库技术包括 Prefer [Hristidis 等人 2001] 和 Onion [Chang 等人 2000]，它们分别基于预物化和凸包。已经提出了几种方法来组合多个前 $K$ 个查询的结果 [Fagin 等人 2001；Natsev 等人 2001]。

Nearest-neighbor queries specify a query point $q$ and output the objects closest to $q$ ,in increasing order of their distance. Existing database algorithms assume that the objects are indexed by an $\mathrm{R}$ -tree (or some other data-partitioning method) and apply branch-and-bound search. In particular, the depth-first algorithm of Roussopoulos et al. [1995] starts from the root of the R-tree and recursively visits the entry closest to the query point. Entries, which are farther than the nearest neighbor already found, are pruned. The best-first algorithm of Henrich [1994] and Hjaltason and Samet [1999] inserts the entries of the visited nodes in a heap, and follows the one closest to the query point. The relation between skyline queries and nearest-neighbor search has been exploited by previous skyline algorithms and will be discussed in Section 2.

最近邻查询指定一个查询点 $q$，并按距离递增的顺序输出最接近 $q$ 的对象。现有的数据库算法假设对象由 $\mathrm{R}$ 树（或其他某种数据分区方法）进行索引，并采用分支限界搜索。特别是，Roussopoulos 等人 [1995] 提出的深度优先算法从 R 树的根节点开始，递归访问最接近查询点的条目。那些比已找到的最近邻更远的条目将被剪枝。Henrich [1994] 以及 Hjaltason 和 Samet [1999] 提出的最佳优先算法将已访问节点的条目插入一个堆中，并沿着最接近查询点的条目继续搜索。先前的天际线算法已经利用了天际线查询和最近邻搜索之间的关系，这将在第 2 节中进行讨论。

Skylines, and other directly related problems such as multiobjective optimization [Steuer 1986], maximum vectors [Kung et al. 1975; Matousek 1991], and the contour problem [McLain 1974], have been extensively studied and numerous algorithms have been proposed for main-memory processing. To the best of our knowledge, however, the first work addressing skylines in the context of databases was Borzsonyi et al. [2001], which develops algorithms based on block nested loops, divide-and-conquer, and index scanning. An improved version of block nested loops is presented in Chomicki et al. [2003]. Tan et al. [2001] proposed progressive (or on-line) algorithms that can output skyline points without having to scan the entire data input. Kossmann et al. [2002] presented an algorithm,called ${NN}$ due to its reliance on nearest-neighbor search,which applies the divide-and-conquer framework on datasets indexed by R-trees. The experimental evaluation of Kossmann et al. [2002] showed that NN outperforms previous algorithms in terms of overall performance and general applicability independently of the dataset characteristics, while it supports on-line processing efficiently.

天际线以及其他直接相关的问题，如多目标优化 [Steuer 1986]、最大向量 [Kung 等人 1975；Matousek 1991] 和轮廓问题 [McLain 1974]，已经得到了广泛的研究，并且已经提出了许多用于主存处理的算法。然而，据我们所知，第一篇在数据库环境中研究天际线的工作是 Borzsonyi 等人 [2001] 的研究，该研究开发了基于块嵌套循环、分治法和索引扫描的算法。Chomicki 等人 [2003] 提出了块嵌套循环的改进版本。Tan 等人 [2001] 提出了渐进式（或在线）算法，该算法无需扫描整个数据输入即可输出天际线点。Kossmann 等人 [2002] 提出了一种算法，由于该算法依赖于最近邻搜索，因此称为 ${NN}$，它对由 R 树索引的数据集应用分治法框架。Kossmann 等人 [2002] 的实验评估表明，在整体性能和一般适用性方面，NN 算法优于先前的算法，且不受数据集特征的影响，同时它还能有效地支持在线处理。

Despite its advantages, NN has also some serious shortcomings such as need for duplicate elimination, multiple node visits, and large space requirements. Motivated by this fact, we propose a progressive algorithm called branch and bound skyline (BBS), which, like NN, is based on nearest-neighbor search on multidimensional access methods, but (unlike NN) is optimal in terms of node accesses. We experimentally and analytically show that BBS outperforms NN (usually by orders of magnitude) for all problem instances, while incurring less space overhead. In addition to its efficiency, the proposed algorithm is simple and easily extendible to several practical variations of skyline queries.

尽管 NN 算法有其优点，但它也存在一些严重的缺点，如需要消除重复项、多次访问节点以及需要大量的空间。基于这一事实，我们提出了一种称为分支限界天际线（BBS）的渐进式算法，该算法与 NN 算法类似，也是基于对多维访问方法的最近邻搜索，但（与 NN 算法不同）在节点访问方面是最优的。我们通过实验和分析表明，对于所有问题实例，BBS 算法通常比 NN 算法性能优越几个数量级，同时产生的空间开销更小。除了效率高之外，所提出的算法还很简单，并且易于扩展到天际线查询的几种实际变体。

<!-- Media -->

<!-- figureText: ${y}_{10}$ ${}^{s}2$ s4 $m$ 10 7- ${s}_{1}$ $h$ 。 s 3 -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_3.jpg?x=608&y=319&w=484&h=434&r=0"/>

Fig. 2. Divide-and-conquer.

图 2. 分治法。

<!-- Media -->

The rest of the article is organized as follows: Section 2 reviews previous secondary-memory algorithms for skyline computation, discussing their advantages and limitations. Section 3 introduces BBS, proves its optimality, and analyzes its performance and space consumption. Section 4 proposes alternative skyline queries and illustrates their processing using BBS. Section 5 introduces the concept of approximate skylines, and Section 6 experimentally evaluates BBS, comparing it against NN under a variety of settings. Finally, Section 7 concludes the article and describes directions for future work.

本文的其余部分组织如下：第 2 节回顾了先前用于天际线计算的二级存储算法，讨论了它们的优点和局限性。第 3 节介绍了 BBS 算法，证明了其最优性，并分析了其性能和空间消耗。第 4 节提出了替代的天际线查询，并说明了如何使用 BBS 算法处理这些查询。第 5 节介绍了近似天际线的概念，第 6 节通过实验评估了 BBS 算法，并在各种设置下将其与 NN 算法进行了比较。最后，第 7 节总结了本文并描述了未来的研究方向。

## 2. RELATED WORK

## 2. 相关工作

This section surveys existing secondary-memory algorithms for computing skylines, namely: (1) divide-and-conquer, (2) block nested loop, (3) sort first skyline, (4) bitmap, (5) index, and (6) nearest neighbor. Specifically, (1) and (2) were proposed in Borzsonyi et al. [2001], (3) in Chomicki et al. [2003], (4) and (5) in Tan et al. [2001], and (6) in Kossmann et al. [2002]. We do not consider the sorted list scan,and the $B$ -tree algorithms of Borzsonyi et al. [2001] due to their limited applicability (only for two dimensions) and poor performance, respectively.

本节概述了现有的用于计算天际线的二级存储算法，即：（1）分治法，（2）块嵌套循环法，（3）先排序天际线法，（4）位图法，（5）索引法，以及（6）最近邻法。具体而言，（1）和（2）由 Borzsonyi 等人 [2001] 提出，（3）由 Chomicki 等人 [2003] 提出，（4）和（5）由 Tan 等人 [2001] 提出，（6）由 Kossmann 等人 [2002] 提出。我们不考虑 Borzsonyi 等人 [2001] 提出的排序列表扫描算法和 $B$ 树算法，因为它们分别存在适用性有限（仅适用于二维）和性能较差的问题。

### 2.1 Divide-and-Conquer

### 2.1 分治法

The divide-and-conquer (D&C) approach divides the dataset into several partitions so that each partition fits in memory. Then, the partial skyline of the points in every partition is computed using a main-memory algorithm (e.g., Matousek [1991]), and the final skyline is obtained by merging the partial ones. Figure 2 shows an example using the dataset of Figure 1. The data space is divided into four partitions ${s}_{1},{s}_{2},{s}_{3},{s}_{4}$ ,with partial skylines $\{ a,c,g\} ,\{ d\} ,\{ i\}$ , $\{ m,k\}$ ,respectively. In order to obtain the final skyline,we need to remove those points that are dominated by some point in other partitions. Obviously all points in the skyline of ${s}_{3}$ must appear in the final skyline,while those in ${s}_{2}$ are discarded immediately because they are dominated by any point in ${s}_{3}$ (in fact ${s}_{2}$ needs to be considered only if ${s}_{3}$ is empty). Each skyline point in ${s}_{1}$ is compared only with points in ${s}_{3}$ ,because no point in ${s}_{2}$ or ${s}_{4}$ can dominate those in ${s}_{1}$ . In this example,points $c,g$ are removed because they are dominated by $i$ . Similarly,the skyline of ${s}_{4}$ is also compared with points in ${s}_{3}$ ,which results in the removal of $m$ . Finally,the algorithm terminates with the remaining points $\{ a,i,k\}$ . D&C is efficient only for small datasets (e.g.,if the entire dataset fits in memory then the algorithm requires only one application of a main-memory skyline algorithm). For large datasets, the partitioning process requires reading and writing the entire dataset at least once, thus incurring significant I/O cost. Further, this approach is not suitable for on-line processing because it cannot report any skyline until the partitioning phase completes.

分治法（Divide-and-Conquer，D&C）将数据集划分为多个分区，以使每个分区都能装入内存。然后，使用主存算法（例如，Matousek [1991]）计算每个分区中数据点的部分天际线（partial skyline），并通过合并这些部分天际线得到最终的天际线。图2展示了一个使用图1数据集的示例。数据空间被划分为四个分区${s}_{1},{s}_{2},{s}_{3},{s}_{4}$，其部分天际线分别为$\{ a,c,g\} ,\{ d\} ,\{ i\}$、$\{ m,k\}$。为了得到最终的天际线，我们需要移除那些被其他分区中某些点支配（dominated）的点。显然，${s}_{3}$的天际线中的所有点都必须出现在最终的天际线中，而${s}_{2}$中的点会立即被丢弃，因为它们被${s}_{3}$中的任何点所支配（实际上，只有当${s}_{3}$为空时才需要考虑${s}_{2}$）。${s}_{1}$中的每个天际线点仅与${s}_{3}$中的点进行比较，因为${s}_{2}$或${s}_{4}$中的任何点都无法支配${s}_{1}$中的点。在这个示例中，点$c,g$被移除，因为它们被$i$所支配。类似地，${s}_{4}$的天际线也与${s}_{3}$中的点进行比较，结果移除了$m$。最后，算法以剩余的点$\{ a,i,k\}$结束。分治法仅对小数据集有效（例如，如果整个数据集都能装入内存，那么该算法只需要应用一次主存天际线算法）。对于大数据集，分区过程至少需要对整个数据集进行一次读写操作，从而产生显著的I/O成本。此外，这种方法不适合在线处理，因为在分区阶段完成之前，它无法报告任何天际线。

### 2.2 Block Nested Loop and Sort First Skyline

### 2.2 块嵌套循环与先排序天际线算法

A straightforward approach to compute the skyline is to compare each point $p$ with every other point,and report $p$ as part of the skyline if it is not dominated. Block nested loop (BNL) builds on this concept by scanning the data file and keeping a list of candidate skyline points in main memory. At the beginning, the list contains the first data point,while for each subsequent point $p$ ,there are three cases: (i) if $p$ is dominated by any point in the list,it is discarded as it is not part of the skyline; (ii) if $p$ dominates any point in the list,it is inserted, and all points in the list dominated by $p$ are dropped; and (iii) if $p$ is neither dominated by, nor dominates, any point in the list, it is simply inserted without dropping any point.

计算天际线的一种直接方法是将每个点$p$与其他所有点进行比较，如果$p$未被支配，则将其作为天际线的一部分进行报告。块嵌套循环（Block Nested Loop，BNL）基于这一概念，通过扫描数据文件并在主存中维护一个候选天际线点列表。一开始，列表中包含第一个数据点，对于后续的每个点$p$，有三种情况：（i）如果$p$被列表中的任何点支配，则将其丢弃，因为它不是天际线的一部分；（ii）如果$p$支配列表中的任何点，则将其插入列表，并删除列表中被$p$支配的所有点；（iii）如果$p$既不被列表中的任何点支配，也不支配列表中的任何点，则直接插入该点，而不删除任何点。

The list is self-organizing because every point found dominating other points is moved to the top. This reduces the number of comparisons as points that dominate multiple other points are likely to be checked first. A problem of BNL is that the list may become larger than the main memory. When this happens, all points falling in the third case (cases (i) and (ii) do not increase the list size) are added to a temporary file. This fact necessitates multiple passes of BNL. In particular, after the algorithm finishes scanning the data file, only points that were inserted in the list before the creation of the temporary file are guaranteed to be in the skyline and are output. The remaining points must be compared against the ones in the temporary file. Thus, BNL has to be executed again, this time using the temporary (instead of the data) file as input.

该列表是自组织的，因为每个被发现支配其他点的点都会被移到列表顶部。由于支配多个其他点的点很可能会首先被检查，因此这减少了比较次数。块嵌套循环算法的一个问题是，列表的大小可能会超过主存容量。当这种情况发生时，所有属于第三种情况的点（情况（i）和（ii）不会增加列表大小）会被添加到一个临时文件中。这就使得块嵌套循环算法需要进行多次遍历。具体来说，在算法完成对数据文件的扫描后，只有在创建临时文件之前插入到列表中的点才能保证在天际线中并被输出。其余的点必须与临时文件中的点进行比较。因此，块嵌套循环算法必须再次执行，这次使用临时文件（而不是数据文件）作为输入。

The advantage of BNL is its wide applicability, since it can be used for any dimensionality without indexing or sorting the data file. Its main problems are the reliance on main memory (a small memory may lead to numerous iterations) and its inadequacy for progressive processing (it has to read the entire data file before it returns the first skyline point). The sort first skyline (SFS) variation of BNL alleviates these problems by first sorting the entire dataset according to a (monotone) preference function. Candidate points are inserted into the list in ascending order of their scores, because points with lower scores are likely to dominate a large number of points, thus rendering the pruning more effective. SFS exhibits progressive behavior because the presorting ensures that a point $p$ dominating another ${p}^{\prime }$ must be visited before ${p}^{\prime }$ ; hence we can immediately output the points inserted to the list as skyline points. Nevertheless, SFS has to scan the entire data file to return a complete skyline, because even a skyline point may have a very large score and thus appear at the end of the sorted list (e.g.,in Figure 1,point $a$ has the third largest score for the preference function $0 \cdot$ distance $+ 1 \cdot$ price). Another problem of SFS (and BNL) is that the order in which the skyline points are reported is fixed (and decided by the sort order), while as discussed in Section 2.6, a progressive skyline algorithm should be able to report points according to user-specified scoring functions.

块嵌套循环（BNL，Block Nested Loop）算法的优势在于其广泛的适用性，因为它可以用于任何维度的数据，而无需对数据文件进行索引或排序。其主要问题是依赖主内存（内存较小可能导致多次迭代），并且不适合渐进式处理（在返回第一个天际点之前，它必须读取整个数据文件）。BNL的先排序天际线（SFS，Sort First Skyline）变体通过首先根据一个（单调）偏好函数对整个数据集进行排序来缓解这些问题。候选点按照其得分的升序插入到列表中，因为得分较低的点更有可能支配大量的点，从而使剪枝更加有效。SFS具有渐进式特性，因为预排序确保了支配另一个点 ${p}^{\prime }$ 的点 $p$ 必须在 ${p}^{\prime }$ 之前被访问；因此，我们可以立即将插入到列表中的点作为天际点输出。然而，SFS必须扫描整个数据文件才能返回完整的天际线，因为即使是一个天际点也可能具有非常大的得分，从而出现在排序列表的末尾（例如，在图1中，点 $a$ 对于偏好函数 $0 \cdot$ 距离 $+ 1 \cdot$ 价格的得分是第三大）。SFS（和BNL）的另一个问题是，天际点的报告顺序是固定的（由排序顺序决定），而正如第2.6节所讨论的，渐进式天际线算法应该能够根据用户指定的评分函数报告点。

<!-- Media -->

Table I. The Bitmap Approach

表I. 位图方法

<table><tr><td>id</td><td>Coordinate</td><td>Bitmap Representation</td></tr><tr><td>$a$</td><td>(1,9)</td><td>(1111111111, 1100000000)</td></tr><tr><td>$b$</td><td>(2,10)</td><td>(1111111110, 1000000000)</td></tr><tr><td>$C$</td><td>(4, 8)</td><td>(1111111000, 1110000000)</td></tr><tr><td>$d$</td><td>(6,7)</td><td>$\left( {\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 1 & 0 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  ,\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 0 & 0 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  }\right)$</td></tr><tr><td>$e$</td><td>(9,10)</td><td>(1100000000, 1000000000)</td></tr><tr><td>$f$</td><td>(7, 5)</td><td>$\left( {\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 0 & 0 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  ,\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 1 & 1 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  }\right)$</td></tr><tr><td>$g$</td><td>(5, 6)</td><td>(1111110000, 1111100000)</td></tr><tr><td>$h$</td><td>(4,3)</td><td>(1111111000, 1111111100)</td></tr><tr><td>$i$</td><td>(3,2)</td><td>(111111100, 1111111110)</td></tr><tr><td>$k$</td><td>(9,1)</td><td>(1100000000, 11111111111)</td></tr><tr><td>$l$</td><td>(10,4)</td><td>(1000000000, 1111111000)</td></tr><tr><td>$m$</td><td>(6, 2)</td><td>(1111100000, 1111111110)</td></tr><tr><td>$n$</td><td>(8, 3)</td><td>(1110000000, 111111100)</td></tr></table>

<table><tbody><tr><td>编号</td><td>坐标</td><td>位图表示法</td></tr><tr><td>$a$</td><td>(1,9)</td><td>(1111111111, 1100000000)</td></tr><tr><td>$b$</td><td>(2,10)</td><td>(1111111110, 1000000000)</td></tr><tr><td>$C$</td><td>(4, 8)</td><td>(1111111000, 1110000000)</td></tr><tr><td>$d$</td><td>(6,7)</td><td>$\left( {\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 1 & 0 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  ,\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 0 & 0 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  }\right)$</td></tr><tr><td>$e$</td><td>(9,10)</td><td>(1100000000, 1000000000)</td></tr><tr><td>$f$</td><td>(7, 5)</td><td>$\left( {\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 0 & 0 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  ,\left\lbrack  \begin{matrix} 1 & 1 & 1 \\  1 & 1 & 1 \\  0 & 0 & 0 \\  0 & 0 &  \end{matrix}\right\rbrack  }\right)$</td></tr><tr><td>$g$</td><td>(5, 6)</td><td>(1111110000, 1111100000)</td></tr><tr><td>$h$</td><td>(4,3)</td><td>(1111111000, 1111111100)</td></tr><tr><td>$i$</td><td>(3,2)</td><td>(111111100, 1111111110)</td></tr><tr><td>$k$</td><td>(9,1)</td><td>(1100000000, 11111111111)</td></tr><tr><td>$l$</td><td>(10,4)</td><td>(1000000000, 1111111000)</td></tr><tr><td>$m$</td><td>(6, 2)</td><td>(1111100000, 1111111110)</td></tr><tr><td>$n$</td><td>(8, 3)</td><td>(1110000000, 111111100)</td></tr></tbody></table>

<!-- Media -->

### 2.3 Bitmap

### 2.3 位图

This technique encodes in bitmaps all the information needed to decide whether a point is in the skyline. Toward this,a data point $p = \left( {{p}_{1},{p}_{2},\ldots ,{p}_{d}}\right)$ ,where $d$ is the number of dimensions,is mapped to an $m$ -bit vector,where $m$ is the total number of distinct values over all dimensions. Let ${k}_{i}$ be the total number of distinct values on the $i$ th dimension (i.e., $m = \mathop{\sum }\limits_{{i = 1 \sim  d}}{k}_{i}$ ). In Figure 1,for example,there are ${k}_{1} = {k}_{2} = {10}$ distinct values on the $x,y$ dimensions and $m = {20}$ . Assume that ${p}_{i}$ is the ${j}_{i}$ th smallest number on the $i$ th axis; then it is represented by ${k}_{i}$ bits,where the leftmost $\left( {{k}_{i} - {j}_{i} + 1}\right)$ bits are 1,and the remaining ones 0 . Table I shows the bitmaps for points in Figure 1. Since point $a$ has the smallest value (1) on the $x$ axis,all bits of ${a}_{1}$ are 1 . Similarly,since ${a}_{2}\left( { = 9}\right)$ is the ninth smallest on the $y$ axis,the first ${10} - 9 + 1 = 2$ bits of its representation are 1 , while the remaining ones are 0 .

这种技术在位图中对判断一个点是否属于天际线（skyline）所需的所有信息进行编码。为此，一个数据点 $p = \left( {{p}_{1},{p}_{2},\ldots ,{p}_{d}}\right)$（其中 $d$ 是维度数）被映射到一个 $m$ 位向量，其中 $m$ 是所有维度上不同值的总数。设 ${k}_{i}$ 是第 $i$ 维上不同值的总数（即 $m = \mathop{\sum }\limits_{{i = 1 \sim  d}}{k}_{i}$）。例如，在图 1 中，$x,y$ 维上有 ${k}_{1} = {k}_{2} = {10}$ 个不同的值，且 $m = {20}$。假设 ${p}_{i}$ 是第 $i$ 轴上第 ${j}_{i}$ 小的数；那么它由 ${k}_{i}$ 位表示，其中最左边的 $\left( {{k}_{i} - {j}_{i} + 1}\right)$ 位为 1，其余位为 0。表 I 展示了图 1 中各点的位图。由于点 $a$ 在 $x$ 轴上的值最小（为 1），所以 ${a}_{1}$ 的所有位都为 1。类似地，由于 ${a}_{2}\left( { = 9}\right)$ 是 $y$ 轴上第 9 小的数，其表示的前 ${10} - 9 + 1 = 2$ 位为 1，其余位为 0。

Consider that we want to decide whether a point,for example, $c$ with bitmap representation (1111111000, 1110000000), belongs to the skyline. The rightmost bits equal to 1,are the fourth and the eighth,on dimensions $x$ and $y$ , respectively. The algorithm creates two bit-strings, ${c}_{X} = {1110000110000}$ and ${c}_{Y} = {0011011111111}$ ,by juxtaposing the corresponding bits (i.e.,the fourth and eighth) of every point. In Table I, these bit-strings (shown in bold) contain 13 bits (one from each object,starting from $a$ and ending with $n$ ). The 1 s in the result of ${c}_{X}\& {c}_{Y} = {0010000110000}$ indicate the points that dominate $c$ ,that is, $c,h$ ,and $i$ . Obviously,if there is more than a single 1,the considered point is not in the skyline. ${}^{2}$ The same operations are repeated for every point in the dataset to obtain the entire skyline.

假设我们要判断一个点，例如位图表示为 (1111111000, 1110000000) 的点 $c$ 是否属于天际线。最右边等于 1 的位分别是第 $x$ 维和第 $y$ 维上的第 4 位和第 8 位。该算法通过并列每个点的相应位（即第 4 位和第 8 位）创建两个位串 ${c}_{X} = {1110000110000}$ 和 ${c}_{Y} = {0011011111111}$。在表 I 中，这些位串（用粗体显示）包含 13 位（每个对象一位，从 $a$ 开始到 $n$ 结束）。${c}_{X}\& {c}_{Y} = {0010000110000}$ 的结果中的 1 表示支配 $c$ 的点，即 $c,h$ 和 $i$。显然，如果结果中 1 的数量不止一个，那么所考虑的点就不属于天际线。${}^{2}$ 对数据集中的每个点重复相同的操作以获得整个天际线。

<!-- Media -->

Table II. The Index Approach

表 II. 索引方法

<table><tr><td colspan="2">List 1</td><td colspan="2">List 2</td></tr><tr><td>$a\left( {1,9}\right)$</td><td>minC = 1</td><td>$k\left( {9,1}\right)$</td><td>min $C = 1$</td></tr><tr><td>$b\left( {2,{10}}\right)$</td><td>minC = 2</td><td>$i\left( {3,2}\right) ,m\left( {6,2}\right)$</td><td>min $C = 2$</td></tr><tr><td>$c\left( {4,8}\right)$</td><td>minC = 4</td><td>$h\left( {4,3}\right) ,n\left( {8,3}\right)$</td><td>min $C = 3$</td></tr><tr><td>$g\left( {5,6}\right)$</td><td>minC = 5</td><td>$l\left( {{10},4}\right)$</td><td>minC = 4</td></tr><tr><td>$d\left( {6,7}\right)$</td><td>minC = 6</td><td>$f\left( {7,5}\right)$</td><td>minC = 5</td></tr><tr><td>$e\left( {9,{10}}\right)$</td><td>minC = 9</td><td/><td/></tr></table>

<table><tbody><tr><td colspan="2">列表1</td><td colspan="2">列表2</td></tr><tr><td>$a\left( {1,9}\right)$</td><td>最小C值 = 1</td><td>$k\left( {9,1}\right)$</td><td>最小 $C = 1$</td></tr><tr><td>$b\left( {2,{10}}\right)$</td><td>最小C值 = 2</td><td>$i\left( {3,2}\right) ,m\left( {6,2}\right)$</td><td>最小 $C = 2$</td></tr><tr><td>$c\left( {4,8}\right)$</td><td>最小C值 = 4</td><td>$h\left( {4,3}\right) ,n\left( {8,3}\right)$</td><td>最小 $C = 3$</td></tr><tr><td>$g\left( {5,6}\right)$</td><td>最小C值 = 5</td><td>$l\left( {{10},4}\right)$</td><td>最小C值 = 4</td></tr><tr><td>$d\left( {6,7}\right)$</td><td>最小C值 = 6</td><td>$f\left( {7,5}\right)$</td><td>最小C值 = 5</td></tr><tr><td>$e\left( {9,{10}}\right)$</td><td>最小C值 = 9</td><td></td><td></td></tr></tbody></table>

<!-- Media -->

The efficiency of bitmap relies on the speed of bit-wise operations. The approach can quickly return the first few skyline points according to their insertion order (e.g., alphabetical order in Table I), but, as with BNL and SFS, it cannot adapt to different user preferences. Furthermore, the computation of the entire skyline is expensive because, for each point inspected, it must retrieve the bitmaps of all points in order to obtain the juxtapositions. Also the space consumption may be prohibitive, if the number of distinct values is large. Finally, the technique is not suitable for dynamic datasets where insertions may alter the rankings of attribute values.

位图的效率依赖于按位操作的速度。该方法可以根据点的插入顺序（例如，表I中的字母顺序）快速返回前几个天际线点，但与BNL和SFS一样，它无法适应不同的用户偏好。此外，计算整个天际线的成本很高，因为对于每个被检查的点，都必须检索所有点的位图以获得并列关系。而且，如果不同值的数量很大，空间消耗可能会过高。最后，该技术不适用于动态数据集，因为在动态数据集中插入操作可能会改变属性值的排名。

### 2.4 Index

### 2.4 索引

The index approach organizes a set of $d$ -dimensional points into $d$ lists such that a point $p = \left( {{p}_{1},{p}_{2},\ldots ,{p}_{d}}\right)$ is assigned to the $i$ th list $\left( {1 \leq  i \leq  d}\right)$ ,if and only if its coordinate ${p}_{i}$ on the $i$ th axis is the minimum among all dimensions,or formally, ${p}_{i} \leq  {p}_{j}$ for all $j \neq  i$ . Table II shows the lists for the dataset of Figure 1 . Points in each list are sorted in ascending order of their minimum coordinate (min $C$ ,for short) and indexed by a B-tree. A batch in the $i$ th list consists of points that have the same $i$ th coordinate (i.e., $\min C$ ). In Table II,every point of list 1 constitutes an individual batch because all $x$ coordinates are different. Points in list 2 are divided into five batches $\{ k\} ,\{ i,m\} ,\{ h,n\} ,\{ l\}$ ,and $\{ f\}$ .

索引方法将一组$d$维点组织成$d$个列表，使得一个点$p = \left( {{p}_{1},{p}_{2},\ldots ,{p}_{d}}\right)$被分配到第$i$个列表$\left( {1 \leq  i \leq  d}\right)$中，当且仅当它在第$i$个轴上的坐标${p}_{i}$是所有维度中的最小值，或者形式上，对于所有$j \neq  i$都满足${p}_{i} \leq  {p}_{j}$。表II展示了图1数据集的列表。每个列表中的点按其最小坐标（简称为min $C$）升序排序，并通过B树进行索引。第$i$个列表中的一个批次由具有相同第$i$个坐标（即$\min C$）的点组成。在表II中，列表1中的每个点都构成一个单独的批次，因为所有$x$坐标都不同。列表2中的点被分为五个批次$\{ k\} ,\{ i,m\} ,\{ h,n\} ,\{ l\}$和$\{ f\}$。

Initially, the algorithm loads the first batch of each list, and handles the one with the minimum $\min C$ . In Table II,the first batches $\{ a\} ,\{ k\}$ have identical $\min C = 1$ ,in which case the algorithm handles the batch from list 1 . Processing a batch involves (i) computing the skyline inside the batch, and (ii) among the computed points, it adds the ones not dominated by any of the already-found skyline points into the skyline list. Continuing the example,since batch $\{ a\}$ contains a single point and no skyline point is found so far, $a$ is added to the skyline list. The next batch $\{ b\}$ in list 1 has min $C = 2$ ; thus,the algorithm handles batch $\{ k\}$ from list 2 . Since $k$ is not dominated by $a$ ,it is inserted in the skyline. Similarly,the next batch handled is $\{ b\}$ from list 1,where $b$ is dominated by point $a$ (already in the skyline). The algorithm proceeds with batch $\{ i,m\}$ ,computes the skyline inside the batch that contains a single point $i$ (i.e., $i$ dominates $m$ ),and adds $i$ to the skyline. At this step,the algorithm does not need to proceed further,because both coordinates of $i$ are smaller than or equal to the $\min C$ (i.e.,4,3) of the next batches (i.e., $\{ c\} ,\{ h,n\}$ ) of lists 1 and 2. This means that all the remaining points (in both lists) are dominated by $i$ , and the algorithm terminates with $\{ a,i,k\}$ .

最初，算法加载每个列表的第一个批次，并处理具有最小$\min C$的批次。在表II中，第一个批次$\{ a\} ,\{ k\}$具有相同的$\min C = 1$，在这种情况下，算法处理列表1中的批次。处理一个批次包括：（i）计算批次内的天际线；（ii）在计算出的点中，将未被任何已找到的天际线点支配的点添加到天际线列表中。继续这个例子，由于批次$\{ a\}$只包含一个点，并且到目前为止还没有找到天际线点，所以$a$被添加到天际线列表中。列表1中的下一个批次$\{ b\}$的min $C = 2$；因此，算法处理列表2中的批次$\{ k\}$。由于$k$未被$a$支配，所以它被插入到天际线中。类似地，接下来处理的批次是列表1中的$\{ b\}$，其中$b$被点$a$（已在天际线中）支配。算法继续处理批次$\{ i,m\}$，计算该批次内的天际线，该批次只包含一个点$i$（即$i$支配$m$），并将$i$添加到天际线中。在这一步，算法不需要进一步处理，因为$i$的两个坐标都小于或等于列表1和列表2中下一批次（即$\{ c\} ,\{ h,n\}$）的$\min C$（即4,3）。这意味着（两个列表中）所有剩余的点都被$i$支配，算法以$\{ a,i,k\}$结束。

---

<!-- Footnote -->

${}^{2}$ The result of "&" will contain several 1s if multiple skyline points coincide. This case can be handled with an additional "or" operation [Tan et al. 2001].

${}^{2}$ 如果多个天际线点重合，“与”运算的结果将包含多个1。这种情况可以通过额外的“或”运算来处理[Tan等人，2001]。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 3 ${f}_{\text{O}}$ 89 7- 6- $h$ Og 6 o 10 (b) Discovery of point $a$ 4 $h$ 3 . Og d (a) Discovery of point $i$ -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_7.jpg?x=404&y=319&w=893&h=388&r=0"/>

Fig. 3. Example of NN.

图3. 最近邻示例。

<!-- Media -->

Although this technique can quickly return skyline points at the top of the lists, the order in which the skyline points are returned is fixed, not supporting user-defined preferences. Furthermore, as indicated in Kossmann et al. [2002], the lists computed for $d$ dimensions cannot be used to retrieve the skyline on any subset of the dimensions because the list that an element belongs to may change according the subset of selected dimensions. In general, for supporting queries on arbitrary dimensions, an exponential number of lists must be precomputed.

虽然这种技术可以快速返回列表顶部的天际线点，但返回天际线点的顺序是固定的，不支持用户定义的偏好。此外，正如Kossmann等人[2002]所指出的，为$d$个维度计算的列表不能用于检索任何维度子集上的天际线，因为元素所属的列表可能会根据所选维度的子集而改变。一般来说，为了支持对任意维度的查询，必须预先计算指数数量的列表。

### 2.5 Nearest Neighbor

### 2.5 最近邻

NN uses the results of nearest-neighbor search to partition the data universe recursively. As an example, consider the application of the algorithm to the dataset of Figure 1, which is indexed by an R-tree [Guttman 1984; Sellis et al. 1987; Beckmann et al. 1990]. NN performs a nearest-neighbor query (using an existing algorithm such as one of the proposed by Roussopoulos et al. [1995], or Hjaltason and Samet [1999] on the R-tree, to find the point with the minimum distance (mindist) from the beginning of the axes (point $o$ ). Without loss of generality, ${}^{3}$ we assume that distances are computed according to the ${\mathrm{L}}_{1}$ norm, that is,the mindist of a point $p$ from the beginning of the axes equals the sum of the coordinates of $p$ . It can be shown that the first nearest neighbor (point $i$ with mindist 5 ) is part of the skyline. On the other hand,all the points in the dominance region of $i$ (shaded area in Figure 3(a)) can be pruned from further consideration. The remaining space is split in two partitions based on the coordinates $\left( {{i}_{x},{i}_{y}}\right)$ of point $i$ : (i) $\left\lbrack  {0,{i}_{x}}\right) \left\lbrack  {0,\infty }\right)$ and (ii) $\lbrack 0,\infty )\left\lbrack  {0,{i}_{y}}\right)$ . In Figure 3(a), the first partition contains subdivisions 1 and 3, while the second one contains subdivisions 1 and 2 .

NN算法利用最近邻搜索的结果对数据空间进行递归划分。例如，考虑将该算法应用于图1所示的数据集，该数据集由R树进行索引（[古特曼1984年；塞利斯等人1987年；贝克曼等人1990年]）。NN算法在R树上执行最近邻查询（使用现有的算法，如鲁索普洛斯等人[1995年]或哈尔塔松和萨梅特[1999年]提出的算法之一），以找到与坐标轴起点（点$o$）距离最小（最小距离）的点。不失一般性，${}^{3}$我们假设距离是根据${\mathrm{L}}_{1}$范数计算的，即点$p$与坐标轴起点的最小距离等于$p$各坐标之和。可以证明，第一个最近邻（点$i$，最小距离为5）是天际线的一部分。另一方面，$i$的支配区域（图3(a)中的阴影区域）内的所有点都可以不再考虑。根据点$i$的坐标$\left( {{i}_{x},{i}_{y}}\right)$，将剩余空间划分为两个分区：(i) $\left\lbrack  {0,{i}_{x}}\right) \left\lbrack  {0,\infty }\right)$和(ii) $\lbrack 0,\infty )\left\lbrack  {0,{i}_{y}}\right)$。在图3(a)中，第一个分区包含子分区1和3，而第二个分区包含子分区1和2。

The partitions resulting after the discovery of a skyline point are inserted in a to-do list. While the to-do list is not empty, NN removes one of the partitions from the list and recursively repeats the same process. For instance,point $a$ is the nearest neighbor in partition $\left\lbrack  {0,{i}_{x}}\right) \lbrack 0,\infty )$ ,which causes the insertion of partitions $\left\lbrack  {0,{a}_{x}}\right) \lbrack 0,\infty )$ (subdivisions 5 and 7 in Figure 3(b)) and $\left\lbrack  {0,{i}_{x}}\right) \left\lbrack  {0,{a}_{y}}\right)$ (subdivisions 5 and 6 in Figure 3(b)) in the to-do list. If a partition is empty, it is not subdivided further. In general,if $d$ is the dimensionality of the data-space, a new skyline point causes $d$ recursive applications of NN. In particular,each coordinate of the discovered point splits the corresponding axis, introducing a new search region towards the origin of the axis.

发现一个天际线点后得到的分区会被插入到待办列表中。只要待办列表不为空，NN算法就会从列表中移除一个分区，并递归地重复相同的过程。例如，点$a$是分区$\left\lbrack  {0,{i}_{x}}\right) \lbrack 0,\infty )$中的最近邻，这会导致将分区$\left\lbrack  {0,{a}_{x}}\right) \lbrack 0,\infty )$（图3(b)中的子分区5和7）和$\left\lbrack  {0,{i}_{x}}\right) \left\lbrack  {0,{a}_{y}}\right)$（图3(b)中的子分区5和6）插入到待办列表中。如果一个分区为空，则不再对其进行进一步划分。一般来说，如果$d$是数据空间的维数，一个新的天际线点会导致NN算法进行$d$次递归应用。具体而言，所发现点的每个坐标都会分割相应的坐标轴，从而引入一个朝向坐标轴原点的新搜索区域。

---

<!-- Footnote -->

${}^{3}\mathrm{{NN}}$ (and BBS) can be applied with any monotone function; the skyline points are the same,but the order in which they are discovered may be different.

${}^{3}\mathrm{{NN}}$（以及BBS算法）可以与任何单调函数一起使用；天际线点是相同的，但发现它们的顺序可能不同。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: $\left( {{n}_{x},{n}_{y},{n}_{z}}\right)$ (b) First query $\left\lbrack  {0,{n}_{x}}\right) \lbrack 0,\infty )\lbrack 0,\infty )$ (d) Third query $\lbrack 0,\infty )\lbrack 0,\infty )\left\lbrack  {0,{n}_{z}}\right)$ axis z axis $y$ axis $x$ (a) First skyline point (c) Second query $\lbrack 0,\infty )\left\lbrack  {0,{n}_{y}}\right) \lbrack 0,\infty )$ -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_8.jpg?x=438&y=317&w=856&h=502&r=0"/>

Fig. 4. NN partitioning for three-dimensions.

图4. 三维空间的NN划分。

<!-- Media -->

Figure 4(a) shows a three-dimensional (3D) example,where point $n$ with coordinates $\left( {{n}_{x},{n}_{y},{n}_{z}}\right)$ is the first nearest neighbor (i.e.,skyline point). The NN algorithm will be recursively called for the partitions (i) $\left\lbrack  {0,{n}_{x}}\right) \lbrack 0,\infty )\lbrack 0,\infty )$ (Figure 4(b)),(ii) $\lbrack 0,\infty )\left\lbrack  {0,{n}_{y}}\right) \left\lbrack  {0,\infty }\right)$ (Figure 4(c)) and (iii) $\lbrack 0,\infty )\lbrack 0,\infty )\left\lbrack  {0,{n}_{z}}\right)$ (Figure 4(d)). Among the eight space subdivisions shown in Figure 4, the eighth one will not be searched by any query since it is dominated by point $n$ . Each of the remaining subdivisions, however, will be searched by two queries, for example, a skyline point in subdivision 2 will be discovered by both the second and third queries.

图4(a)展示了一个三维（3D）示例，其中坐标为$\left( {{n}_{x},{n}_{y},{n}_{z}}\right)$的点$n$是第一个最近邻（即天际线点）。NN算法将对以下分区进行递归调用：(i) $\left\lbrack  {0,{n}_{x}}\right) \lbrack 0,\infty )\lbrack 0,\infty )$（图4(b)）、(ii) $\lbrack 0,\infty )\left\lbrack  {0,{n}_{y}}\right) \left\lbrack  {0,\infty }\right)$（图4(c)）和(iii) $\lbrack 0,\infty )\lbrack 0,\infty )\left\lbrack  {0,{n}_{z}}\right)$（图4(d)）。在图4所示的八个空间子分区中，第八个子分区不会被任何查询搜索，因为它被点$n$所支配。然而，其余的每个子分区都会被两个查询搜索，例如，子分区2中的一个天际线点将由第二个和第三个查询共同发现。

In general,for $d > 2$ ,the overlapping of the partitions necessitates duplicate elimination. Kossmann et al. [2002] proposed the following elimination methods:

一般来说，对于 $d > 2$，分区的重叠需要进行重复消除。科斯曼（Kossmann）等人 [2002] 提出了以下消除方法：

-Laisser-faire: A main memory hash table stores the skyline points found so far. When a point $p$ is discovered,it is probed and,if it already exists in the hash table, $p$ is discarded; otherwise, $p$ is inserted into the hash table. The technique is straightforward and incurs minimum CPU overhead, but results in very high I/O cost since large parts of the space will be accessed by multiple queries.

-放任法（Laisser - faire）：一个主内存哈希表存储到目前为止找到的天际点。当发现一个点 $p$ 时，会对其进行探查，如果它已经存在于哈希表中，则丢弃 $p$；否则，将 $p$ 插入到哈希表中。这种技术很直接，产生的 CPU 开销最小，但会导致非常高的 I/O 成本，因为空间的大部分区域会被多个查询访问。

-Propagate: When a point $p$ is found,all the partitions in the to-do list that contain $p$ are removed and repartitioned according to $p$ . The new partitions are inserted into the to-do list. Although propagate does not discover the same skyline point twice, it incurs high CPU cost because the to-do list is scanned every time a skyline point is discovered.

-传播法（Propagate）：当找到一个点 $p$ 时，待办列表中包含 $p$ 的所有分区都会被移除，并根据 $p$ 重新分区。新的分区会被插入到待办列表中。虽然传播法不会两次发现相同的天际点，但它会产生很高的 CPU 成本，因为每次发现一个天际点时都要扫描待办列表。

-Merge: The main idea is to merge partitions in to-do, thus reducing the number of queries that have to be performed. Partitions that are contained in other ones can be eliminated in the process. Like propagate, merge also incurs high CPU cost since it is expensive to find good candidates for merging. —Fine-grained partitioning: The original NN algorithm generates $d$ partitions after a skyline point is found. An alternative approach is to generate ${2}^{d}$ nonoverlapping subdivisions. In Figure 4, for instance, the discovery of point $n$ will lead to six new queries (i.e., ${2}^{3} - 2$ since subdivisions 1 and 8 cannot contain any skyline points). Although fine-grained partitioning avoids duplicates, it generates the more complex problem of false hits, that is, it is possible that points in one subdivision (e.g., subdivision 4) are dominated by points in another (e.g., subdivision 2) and should be eliminated.

-合并法（Merge）：主要思想是合并待办列表中的分区，从而减少必须执行的查询数量。在这个过程中，可以消除包含在其他分区中的分区。与传播法一样，合并法也会产生很高的 CPU 成本，因为找到合适的合并候选分区代价很高。—细粒度分区法（Fine - grained partitioning）：原始的最近邻（NN）算法在找到一个天际点后会生成 $d$ 个分区。另一种方法是生成 ${2}^{d}$ 个不重叠的子分区。例如，在图 4 中，点 $n$ 的发现将导致六个新的查询（即 ${2}^{3} - 2$，因为子分区 1 和 8 不可能包含任何天际点）。虽然细粒度分区法避免了重复，但它会产生更复杂的误判问题，也就是说，一个子分区（例如，子分区 4）中的点可能会被另一个子分区（例如，子分区 2）中的点所支配，应该被消除。

According to the experimental evaluation of Kossmann et al. [2002], the performance of laisser-faire and merge was unacceptable, while fine-grained partitioning was not implemented due to the false hits problem. Propagate was significantly more efficient, but the best results were achieved by a hybrid method combining propagate and laisser-faire.

根据科斯曼（Kossmann）等人 [2002] 的实验评估，放任法和合并法的性能不可接受，而细粒度分区法由于误判问题没有实现。传播法的效率明显更高，但最好的结果是通过结合传播法和放任法的混合方法实现的。

### 2.6 Discussion About the Existing Algorithms

### 2.6 关于现有算法的讨论

We summarize this section with a comparison of the existing methods, based on the experiments of Tan et al. [2001], Kossmann et al. [2002], and Chomicki et al. [2003]. Tan et al. [2001] examined BNL, D&C, bitmap, and index, and suggested that index is the fastest algorithm for producing the entire skyline under all settings. D&C and bitmap are not favored by correlated datasets (where the skyline is small) as the overhead of partition-merging and bitmap-loading, respectively, does not pay-off. BNL performs well for small skylines, but its cost increases fast with the skyline size (e.g., for anticorrelated datasets, high dimensionality, etc.) due to the large number of iterations that must be performed. Tan et al. [2001] also showed that index has the best performance in returning skyline points progressively, followed by bitmap. The experiments of Chomicki et al. [2003] demonstrated that SFS is in most cases faster than BNL without, however, comparing it with other algorithms. According to the evaluation of Kossmann et al. [2002], NN returns the entire skyline more quickly than index (hence also more quickly than BNL, D&C, and bitmap) for up to four dimensions, and their difference increases (sometimes to orders of magnitudes) with the skyline size. Although index can produce the first few skyline points in shorter time, these points are not representative of the whole skyline (as they are good on only one axis while having large coordinates on the others).

我们根据谭（Tan）等人 [2001]、科斯曼（Kossmann）等人 [2002] 和乔米基（Chomicki）等人 [2003] 的实验，通过比较现有方法来总结本节内容。谭（Tan）等人 [2001] 研究了块嵌套循环（BNL）、分治法（D&C）、位图法（bitmap）和索引法（index），并指出在所有设置下，索引法是生成整个天际线最快的算法。对于相关数据集（天际线较小），分治法和位图法不受青睐，因为分区合并和位图加载的开销分别无法得到回报。块嵌套循环法对于小天际线表现良好，但由于必须执行大量迭代，其成本会随着天际线大小的增加而快速上升（例如，对于反相关数据集、高维数据等）。谭（Tan）等人 [2001] 还表明，索引法在逐步返回天际点方面性能最佳，其次是位图法。乔米基（Chomicki）等人 [2003] 的实验表明，在大多数情况下，顺序过滤扫描法（SFS）比块嵌套循环法快，但没有将其与其他算法进行比较。根据科斯曼（Kossmann）等人 [2002] 的评估，对于最多四维的数据，最近邻法（NN）比索引法更快地返回整个天际线（因此也比块嵌套循环法、分治法和位图法更快），并且它们之间的差异会随着天际线大小的增加而增大（有时达到数量级）。虽然索引法可以在更短的时间内生成前几个天际点，但这些点不能代表整个天际线（因为它们只在一个轴上表现良好，而在其他轴上坐标较大）。

Kossmann et al. [2002] also suggested a set of criteria (adopted from Heller-stein et al. [1999]) for evaluating the behavior and applicability of progressive skyline algorithms:

科斯曼（Kossmann）等人 [2002] 还提出了一组标准（取自赫勒斯坦（Heller - stein）等人 [1999]），用于评估渐进式天际线算法的行为和适用性：

(i) Progressiveness: the first results should be reported to the user almost instantly and the output size should gradually increase.

(i) 渐进性：应几乎立即将第一批结果报告给用户，并且输出大小应逐渐增加。

(ii) Absence of false misses: given enough time, the algorithm should generate the entire skyline.

(ii) 无漏判：在足够的时间内，算法应生成整个天际线。

(iii) Absence of false hits: the algorithm should not discover temporary skyline points that will be later replaced.

(iii) 无误判：算法不应发现后续会被替换的临时天际点。

(iv) Fairness: the algorithm should not favor points that are particularly good in one dimension.

(iv) 公平性：算法不应偏爱在某一个维度上特别好的点。

(v) Incorporation of preferences: the users should be able to determine the order according to which skyline points are reported.

(v) 偏好纳入：用户应该能够确定报告天际点的顺序。

(vi) Universality: the algorithm should be applicable to any dataset distribution and dimensionality, using some standard index structure.

(vi) 通用性：使用一些标准的索引结构，算法应适用于任何数据集分布和维度。

All the methods satisfy criterion (ii), as they deal with exact (as opposed to approximate) skyline computation. Criteria (i) and (iii) are violated by D&C and BNL since they require at least a scan of the data file before reporting skyline points and they both insert points (in partial skylines or the self-organizing list) that are later removed. Furthermore, SFS and bitmap need to read the entire file before termination, while index and NN can terminate as soon as all skyline points are discovered. Criteria (iv) and (vi) are violated by index because it outputs the points according to their minimum coordinates in some dimension and cannot handle skylines in some subset of the original dimensionality. All algorithms, except NN, defy criterion (v); NN can incorporate preferences by simply changing the distance definition according to the input scoring function.

所有方法都满足准则（ii），因为它们处理的是精确（而非近似）的天际线计算。分治法（D&C）和块嵌套循环法（BNL）违反了准则（i）和（iii），因为它们在报告天际线点之前至少需要扫描一次数据文件，并且它们都会插入一些点（在部分天际线或自组织列表中），而这些点随后会被移除。此外，顺序过滤扫描法（SFS）和位图法（bitmap）在结束之前需要读取整个文件，而索引法（index）和最近邻法（NN）一旦发现所有天际线点就可以终止。索引法违反了准则（iv）和（vi），因为它根据点在某些维度上的最小坐标输出点，并且无法处理原始维度的某些子集中的天际线。除了最近邻法（NN）之外，所有算法都违反了准则（v）；最近邻法（NN）可以通过根据输入的评分函数简单地改变距离定义来纳入偏好。

Finally, note that progressive behavior requires some form of preprocessing, that is, index creation (index, NN), sorting (SFS), or bitmap creation (bitmap). This preprocessing is a one-time effort since it can be used by all subsequent queries provided that the corresponding structure is updateable in the presence of record insertions and deletions. The maintenance of the sorted list in SFS can be performed by building a B+-tree on top of the list. The insertion of a record in index simply adds the record in the list that corresponds to its minimum coordinate; similarly, deletion removes the record from the list. NN can also be updated incrementally as it is based on a fully dynamic structure (i.e., the R-tree). On the other hand, bitmap is aimed at static datasets because a record insertion/deletion may alter the bitmap representation of numerous (in the worst case, of all) records.

最后，请注意，渐进式处理行为需要某种形式的预处理，即创建索引（索引法、最近邻法）、排序（顺序过滤扫描法）或创建位图（位图法）。这种预处理是一次性的工作，因为只要相应的结构在记录插入和删除时是可更新的，它就可以被所有后续查询使用。顺序过滤扫描法（SFS）中排序列表的维护可以通过在列表之上构建一个B + 树来完成。在索引法（index）中插入一条记录只需将该记录添加到与其最小坐标对应的列表中；类似地，删除操作则从列表中移除该记录。最近邻法（NN）也可以进行增量更新，因为它基于一个完全动态的结构（即R树）。另一方面，位图法（bitmap）适用于静态数据集，因为插入或删除一条记录可能会改变许多（在最坏情况下，是所有）记录的位图表示。

## 3. BRANCH-AND-BOUND SKYLINE ALGORITHM

## 3. 分支限界天际线算法

Despite its general applicability and performance advantages compared to existing skyline algorithms, NN has some serious shortcomings, which are described in Section 3.1. Then Section 3.2 proposes the BBS algorithm and proves its correctness. Section 3.3 analyzes the performance of BBS and illustrates its $\mathrm{I}/\mathrm{O}$ optimality. Finally,Section 3.4 discusses the incremental maintenance of skylines in the presence of database updates.

尽管与现有的天际线算法相比，最近邻法（NN）具有广泛的适用性和性能优势，但它也有一些严重的缺点，这些缺点将在3.1节中描述。然后，3.2节提出了分支限界天际线算法（BBS）并证明其正确性。3.3节分析了BBS算法的性能并说明了其$\mathrm{I}/\mathrm{O}$最优性。最后，3.4节讨论了在数据库更新时天际线的增量维护问题。

### 3.1 Motivation

### 3.1 动机

A recursive call of the NN algorithm terminates when the corresponding nearest-neighbor query does not retrieve any point within the corresponding space. Lets call such a query empty, to distinguish it from nonempty queries that return results,each spawning $d$ new recursive applications of the algorithm (where $d$ is the dimensionality of the data space). Figure 5 shows a query processing tree, where empty queries are illustrated as transparent cycles. For the second level of recursion, for instance, the second query does not return any results, in which case the recursion will not proceed further. Some of the nonempty queries may be redundant, meaning that they return skyline points already found by previous queries. Let $s$ be the number of skyline points in the result, $e$ the number of empty queries, ${ne}$ the number of nonempty ones,and $r$ the number of redundant queries. Since every nonempty query either retrieves a skyline point,or is redundant,we have ${ne} = s + r$ . Furthermore, the number of empty queries in Figure 5 equals the number of leaf nodes in the recursion tree,that is, $e = {ne} \cdot  \left( {d - 1}\right)  + 1$ . By combining the two equations,we get $e = \left( {s + r}\right)  \cdot  \left( {d - 1}\right)  + 1$ . Each query must traverse a whole path from the root to the leaf level of the R-tree before it terminates; therefore,its I/O cost is at least $h$ node accesses,where $h$ is the height of the tree. Summarizing the above observations, the total number of accesses for NN is: $N{A}_{NN} \geq  \left( {e + s + r}\right)  \cdot  h = \left( {s + r}\right)  \cdot  h \cdot  d + h > s \cdot  h \cdot  d$ . The value $s \cdot  h \cdot  d$ is a rather optimistic lower bound since,for $d > 2$ ,the number $r$ of redundant queries may be very high (depending on the duplicate elimination method used), and queries normally incur more than $h$ node accesses.

当最近邻法（NN）算法的递归调用所对应的最近邻查询在相应空间内未检索到任何点时，该递归调用终止。我们将这样的查询称为空查询，以区别于返回结果的非空查询，每个非空查询都会产生$d$个新的算法递归应用（其中$d$是数据空间的维度）。图5展示了一个查询处理树，其中空查询用透明圆圈表示。例如，在第二层递归中，第二个查询没有返回任何结果，在这种情况下，递归将不再继续。一些非空查询可能是冗余的，这意味着它们返回的天际线点已经被之前的查询找到。设$s$为结果中的天际线点数量，$e$为空查询数量，${ne}$为非空查询数量，$r$为冗余查询数量。由于每个非空查询要么检索到一个天际线点，要么是冗余的，因此有${ne} = s + r$。此外，图5中的空查询数量等于递归树中的叶节点数量，即$e = {ne} \cdot  \left( {d - 1}\right)  + 1$。将这两个方程结合起来，我们得到$e = \left( {s + r}\right)  \cdot  \left( {d - 1}\right)  + 1$。每个查询在终止之前必须遍历R树从根节点到叶节点的整个路径；因此，其I/O成本至少为$h$次节点访问，其中$h$是树的高度。总结上述观察结果，最近邻法（NN）的总访问次数为：$N{A}_{NN} \geq  \left( {e + s + r}\right)  \cdot  h = \left( {s + r}\right)  \cdot  h \cdot  d + h > s \cdot  h \cdot  d$。值$s \cdot  h \cdot  d$是一个相当乐观的下界，因为对于$d > 2$，冗余查询的数量$r$可能非常高（取决于所使用的去重方法），并且查询通常会产生超过$h$次的节点访问。

<!-- Media -->

<!-- figureText: INN -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_11.jpg?x=527&y=317&w=652&h=262&r=0"/>

Fig. 5. Recursion tree.

图5. 递归树。

<!-- Media -->

Another problem of NN concerns the to-do list size, which can exceed that of the dataset for as low as three dimensions, even without considering redundant queries. Assume,for instance,a 3D uniform dataset (cardinality $N$ ) and a skyline query with the preference function $f\left( {x,y,z}\right)  = x$ . The first skyline point $n\left( {{n}_{x},{n}_{y},{n}_{z}}\right)$ has the smallest $x$ coordinate among all data points,and adds partitions ${P}_{x} = \left\lbrack  {0,{n}_{x}}\right) \lbrack 0,\infty )\lbrack 0,\infty ),{P}_{y} = \lbrack 0,\infty )\left\lbrack  {0,{n}_{y}}\right) \lbrack 0,\infty ),{P}_{z} = \lbrack 0,\infty )$ $\lbrack 0,\infty )\left\lbrack  {0,{n}_{z}}\right)$ in the to-do list. Note that the NN query in ${P}_{x}$ is empty because there is no other point whose $x$ coordinate is below ${n}_{x}$ . On the other hand,the expected volume of ${P}_{y}\left( {P}_{z}\right)$ is $1/2$ (assuming unit axis length on all dimensions), because the nearest neighbor is decided solely on $x$ coordinates,and hence ${n}_{y}$ $\left( {n}_{z}\right)$ distributes uniformly in $\left\lbrack  {0,1}\right\rbrack$ . Following the same reasoning,a NN in ${P}_{y}$ finds the second skyline point that introduces three new partitions such that one partition leads to an empty query, while the volumes of the other two are $1/4.{P}_{z}$ is handled similarly,after which the to-do list contains four partitions with volumes $1/4$ ,and 2 empty partitions. In general,after the $i$ th level of recursion,the to-do list contains ${2}^{i}$ partitions with volume $1/{2}^{i}$ ,and ${2}^{i - 1}$ empty partitions. The algorithm terminates when $1/{2}^{i} < 1/N$ (i.e., $i > \log N$ ) so that all partitions in the to-do list are empty. Assuming that the empty queries are performed at the end, the size of the to-do list can be obtained by summing the number $e$ of empty queries at each recursion level $i$ :

神经网络（NN）的另一个问题涉及待办事项列表的大小，即使不考虑冗余查询，在低至三维的情况下，该列表的大小也可能超过数据集的大小。例如，假设一个三维均匀数据集（基数为 $N$ ）和一个具有偏好函数 $f\left( {x,y,z}\right)  = x$ 的天际线查询。第一个天际线点 $n\left( {{n}_{x},{n}_{y},{n}_{z}}\right)$ 在所有数据点中具有最小的 $x$ 坐标，并在待办事项列表中添加了分区 ${P}_{x} = \left\lbrack  {0,{n}_{x}}\right) \lbrack 0,\infty )\lbrack 0,\infty ),{P}_{y} = \lbrack 0,\infty )\left\lbrack  {0,{n}_{y}}\right) \lbrack 0,\infty ),{P}_{z} = \lbrack 0,\infty )$ $\lbrack 0,\infty )\left\lbrack  {0,{n}_{z}}\right)$ 。请注意，${P}_{x}$ 中的最近邻（NN）查询为空，因为没有其他点的 $x$ 坐标低于 ${n}_{x}$ 。另一方面，${P}_{y}\left( {P}_{z}\right)$ 的预期体积为 $1/2$ （假设所有维度上的单位轴长度），因为最近邻仅由 $x$ 坐标决定，因此 ${n}_{y}$ $\left( {n}_{z}\right)$ 在 $\left\lbrack  {0,1}\right\rbrack$ 中均匀分布。按照同样的推理，${P}_{y}$ 中的最近邻查询找到第二个天际线点，该点引入了三个新分区，使得一个分区导致空查询，而另外两个分区的体积为 $1/4.{P}_{z}$ 。处理方式类似，之后待办事项列表包含四个体积为 $1/4$ 的分区和两个空分区。一般来说，在第 $i$ 层递归之后，待办事项列表包含 ${2}^{i}$ 个体积为 $1/{2}^{i}$ 的分区和 ${2}^{i - 1}$ 个空分区。当 $1/{2}^{i} < 1/N$ （即 $i > \log N$ ）时，算法终止，使得待办事项列表中的所有分区都为空。假设空查询在最后执行，待办事项列表的大小可以通过对每个递归级别 $i$ 上空查询的数量 $e$ 求和得到：

$$
\mathop{\sum }\limits_{{i = 1}}^{{\log N}}{2}^{i - 1} = N - 1
$$

<!-- Media -->

<!-- figureText: 9 - ${N}_{6}$ ${e}_{7}$ $n$ N3 N4 ${N}_{5}$ ${N}_{2}$ ${e}_{2}$ 3 - ${N}_{7}$ -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_12.jpg?x=354&y=321&w=1019&h=302&r=0"/>

Fig. 6. R-tree example.

图 6. R 树示例。

<!-- Media -->

The implication of the above equation is that,even in $3\mathrm{D},\mathrm{{NN}}$ may behave like a main-memory algorithm (since the to-do list, which resides in memory, is the same order of size as the input dataset). Using the same reasoning, for arbitrary dimensionality $d > 2,e = \Theta \left( {\left( d - 1\right) }^{\log N}\right)$ ,that is,the to-do list may become orders of magnitude larger than the dataset, which seriously limits the applicability of NN. In fact, as shown in Section 6, the algorithm does not terminate in the majority of experiments involving four and five dimensions.

上述方程的含义是，即使在 $3\mathrm{D},\mathrm{{NN}}$ 中，该算法的行为也可能类似于主存算法（因为驻留在内存中的待办事项列表的大小与输入数据集的大小处于同一数量级）。使用相同的推理，对于任意维度 $d > 2,e = \Theta \left( {\left( d - 1\right) }^{\log N}\right)$ ，即待办事项列表的大小可能比数据集大几个数量级，这严重限制了最近邻（NN）算法的适用性。事实上，如第 6 节所示，在大多数涉及四维和五维的实验中，该算法不会终止。

### 3.2 Description of BBS

### 3.2 最佳优先天际线算法（BBS）的描述

Like NN, BBS is also based on nearest-neighbor search. Although both algorithms can be used with any data-partitioning method, in this article we use R-trees due to their simplicity and popularity. The same concepts can be applied with other multidimensional access methods for high-dimensional spaces, where the performance of R-trees is known to deteriorate. Furthermore, as claimed in Kossmann et al. [2002], most applications involve up to five dimensions, for which R-trees are still efficient. For the following discussion, we use the set of $2\mathrm{D}$ data points of Figure 1,organized in the $\mathrm{R}$ -tree of Figure 6 with node capacity $= 3$ . An intermediate entry ${e}_{i}$ corresponds to the minimum bounding rectangle (MBR) of a node ${N}_{i}$ at the lower level,while a leaf entry corresponds to a data point. Distances are computed according to ${\mathrm{L}}_{1}$ norm,that is, the mindist of a point equals the sum of its coordinates and the mindist of a MBR (i.e., intermediate entry) equals the mindist of its lower-left corner point.

与最近邻算法（NN）一样，最佳优先搜索算法（BBS）同样基于最近邻搜索。尽管这两种算法可以与任何数据分区方法结合使用，但在本文中，由于R树的简单性和普及性，我们使用R树。相同的概念也可以应用于高维空间的其他多维访问方法，在这些方法中，已知R树的性能会下降。此外，正如科斯曼等人（Kossmann et al. [2002]）所声称的，大多数应用涉及的维度最多为五个，对于这些维度，R树仍然是高效的。在接下来的讨论中，我们使用图1中的$2\mathrm{D}$个数据点集，这些数据点组织在图6的$\mathrm{R}$ - 树中，节点容量为$= 3$。中间条目${e}_{i}$对应于较低层节点${N}_{i}$的最小边界矩形（MBR），而叶条目对应于一个数据点。距离是根据${\mathrm{L}}_{1}$范数计算的，即一个点的最小距离（mindist）等于其坐标之和，而一个MBR（即中间条目）的最小距离等于其左下角点的最小距离。

BBS, similar to the previous algorithms for nearest neighbors [Roussopoulos et al. 1995; Hjaltason and Samet 1999] and convex hulls [Böhm and Kriegel 2001], adopts the branch-and-bound paradigm. Specifically, it starts from the root node of the R-tree and inserts all its entries $\left( {{e}_{6},{e}_{7}}\right)$ in a heap sorted according to their mindist. Then,the entry with the minimum mindist $\left( {e}_{7}\right)$ is "expanded". This expansion removes the entry $\left( {e}_{7}\right)$ from the heap and inserts its children $\left( {{e}_{3},{e}_{4},{e}_{5}}\right)$ . The next expanded entry is again the one with the minimum mindist $\left( {e}_{3}\right)$ ,in which the first nearest neighbor(i)is found. This point (i) belongs to the skyline,and is inserted to the list $S$ of skyline points.

与之前的最近邻算法[鲁索普洛斯等人（Roussopoulos et al. 1995）；雅尔塔松和萨梅特（Hjaltason and Samet 1999）]和凸包算法[博姆和克里格尔（Böhm and Kriegel 2001）]类似，最佳优先搜索算法（BBS）采用分支限界范式。具体来说，它从R树的根节点开始，将其所有条目$\left( {{e}_{6},{e}_{7}}\right)$插入到一个根据最小距离（mindist）排序的堆中。然后，最小距离最小的条目$\left( {e}_{7}\right)$被“展开”。这种展开操作将条目$\left( {e}_{7}\right)$从堆中移除，并插入其孩子节点$\left( {{e}_{3},{e}_{4},{e}_{5}}\right)$。下一个被展开的条目同样是最小距离最小的条目$\left( {e}_{3}\right)$，在其中找到了第一个最近邻点(i)。这个点(i)属于天际线，并被插入到天际线点列表$S$中。

<!-- Media -->

Table III. Heap Contents

表III. 堆的内容

<table><tr><td>Action</td><td>Heap Contents</td><td>$S$</td></tr><tr><td>Access root</td><td>$< {e}_{7},4 >  < {e}_{6},6 >$</td><td>0</td></tr><tr><td>Expand ${e}_{7}$</td><td>< ${e}_{3,}5$ >< ${e}_{6,}6$ >< ${e}_{5,}8$ >< ${e}_{4,}{10}$ ></td><td>0</td></tr><tr><td>Expand ${e}_{3}$</td><td><1, 5>< ${e}_{6,}6 >  < h,7 >  < {e}_{5,}8 >  < {e}_{4,}{10} >  < g,{11} >$</td><td>$\{ i\}$</td></tr><tr><td>Expand ${e}_{6}$</td><td>$\mathrm{\mathit{ < h,7 >  < {e}_{5},8 >  < {e}_{1},9 >  < {e}_{4},{10} >  < g,{11} > }}$</td><td>$\{ i\}$</td></tr><tr><td>Expand ${e}_{1}$</td><td><a, 10><e4. 10><g, 11><b, 12><c, 12></td><td>$\{ i,\alpha \}$</td></tr><tr><td>Expand ${e}_{4}$</td><td><k, 10> <g, 11>< b, 12><c, 12>< l, 14></td><td>$\{ i,a,k\}$</td></tr></table>

<table><tbody><tr><td>操作</td><td>堆内容</td><td>$S$</td></tr><tr><td>访问根节点</td><td>$< {e}_{7},4 >  < {e}_{6},6 >$</td><td>0</td></tr><tr><td>展开 ${e}_{7}$</td><td>< ${e}_{3,}5$ >< ${e}_{6,}6$ >< ${e}_{5,}8$ >< ${e}_{4,}{10}$ ></td><td>0</td></tr><tr><td>展开 ${e}_{3}$</td><td><1, 5>< ${e}_{6,}6 >  < h,7 >  < {e}_{5,}8 >  < {e}_{4,}{10} >  < g,{11} >$</td><td>$\{ i\}$</td></tr><tr><td>展开 ${e}_{6}$</td><td>$\mathrm{\mathit{ < h,7 >  < {e}_{5},8 >  < {e}_{1},9 >  < {e}_{4},{10} >  < g,{11} > }}$</td><td>$\{ i\}$</td></tr><tr><td>展开 ${e}_{1}$</td><td><a, 10=""><e4. 10=""><g, 11=""><b, 12=""><c, 12=""></c,></b,></g,></e4.></a,></td><td>$\{ i,\alpha \}$</td></tr><tr><td>展开 ${e}_{4}$</td><td><k, 10=""> <g, 11="">< b, 12><c, 12="">< l, 14></c,></g,></k,></td><td>$\{ i,a,k\}$</td></tr></tbody></table>

Algorithm BBS (R-tree $R$ )

算法BBS（R树 $R$ ）

---

	$S = \varnothing //$ list of skyline points

	$S = \varnothing //$ 天际点列表

		insert all entries of the root $R$ in the heap

		将根节点 $R$ 的所有条目插入堆中

		while heap not empty

			 当堆不为空时

		remove top entry $e$

			 移除堆顶条目 $e$

		if $e$ is dominated by some point in $S$ discard $e$

			 如果 $e$ 被 $S$ 中的某个点支配，则丢弃 $e$

				else // $e$ is not dominated

					 否则 // $e$ 未被支配

						if $e$ is an intermediate entry

							 如果 $e$ 是中间条目

									for each child ${e}_{i}$ of $e$

										 对于 $e$ 的每个子节点 ${e}_{i}$

9. if ${e}_{i}$ is not dominated by some point in $S$ insert ${e}_{i}$ into heap

9. 如果 ${e}_{i}$ 未被 $S$ 中的某个点支配，则将 ${e}_{i}$ 插入堆中

10. else // $e$ is a data point

10. 否则 // $e$ 是一个数据点

11. insert ${e}_{i}$ into $S$

11. 将 ${e}_{i}$ 插入 $S$ 中

		. end while

			 结束循环

	End BBS

	结束BBS算法

---

Fig. 7. BBS algorithm.

图7. BBS算法。

<!-- Media -->

Notice that up to this step BBS behaves like the best-first nearest-neighbor algorithm of Hjaltason and Samet [1999]. The next entry to be expanded is ${e}_{6}$ . Although the nearest-neighbor algorithm would now terminate since the mindist (6) of ${e}_{6}$ is greater than the distance (5) of the nearest neighbor (i) already found,BBS will proceed because node ${N}_{6}$ may contain skyline points (e.g.,a). Among the children of ${e}_{6}$ ,however,only the ones that are not dominated by some point in $S$ are inserted into the heap. In this case, ${e}_{2}$ is pruned because it is dominated by point $i$ . The next entry considered(h)is also pruned as it also is dominated by point $i$ . The algorithm proceeds in the same manner until the heap becomes empty. Table III shows the ids and the mindist of the entries inserted in the heap (skyline points are bold).

注意，到这一步为止，BBS算法的行为与Hjaltason和Samet [1999] 提出的最佳优先最近邻算法类似。下一个要扩展的条目是 ${e}_{6}$ 。尽管最近邻算法此时会终止，因为 ${e}_{6}$ 的最小距离（6）大于已找到的最近邻（i）的距离（5），但BBS算法会继续执行，因为节点 ${N}_{6}$ 可能包含天际点（例如，a）。然而，在 ${e}_{6}$ 的子节点中，只有那些未被 $S$ 中的某个点支配的节点才会被插入堆中。在这种情况下， ${e}_{2}$ 被剪枝，因为它被点 $i$ 支配。接下来考虑的条目（h）也被剪枝，因为它同样被点 $i$ 支配。算法以相同的方式继续执行，直到堆为空。表III显示了插入堆中的条目的id和最小距离（天际点用粗体表示）。

The pseudocode for BBS is shown in Figure 7. Notice that an entry is checked for dominance twice: before it is inserted in the heap and before it is expanded. The second check is necessary because an entry (e.g., ${e}_{5}$ ) in the heap may become dominated by some skyline point discovered after its insertion (therefore, the entry does not need to be visited).

BBS的伪代码如图7所示。注意，一个条目会被检查两次是否被支配：一次是在它被插入堆之前，另一次是在它被扩展之前。第二次检查是必要的，因为堆中的一个条目（例如，${e}_{5}$ ）可能会被插入后发现的某个天际点所支配（因此，该条目无需被访问）。

Next we prove the correctness for BBS.

接下来我们证明BBS的正确性。

LEMMA 1. BBS visits (leaf and intermediate) entries of an $R$ -tree in ascending order of their distance to the origin of the axis.

引理1. BBS按照$R$ -树的（叶节点和中间节点）条目到坐标轴原点的距离升序访问这些条目。

<!-- Media -->

<!-- figureText: 8 - edge of the lower left point of $e$ $k$ 10 7 - 5 - 4 - 3. 2 7 -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_14.jpg?x=603&y=319&w=518&h=356&r=0"/>

Fig. 8. Entries of the main-memory R-tree.

图8. 主存R - 树的条目。

<!-- Media -->

Proof. The proof is straightforward since the algorithm always visits entries according to their mindist order preserved by the heap.

证明。证明很直接，因为该算法总是按照堆中保存的最小距离顺序访问条目。

LEMMA 2. Any data point added to $S$ during the execution of the algorithm is guaranteed to be a final skyline point.

引理2. 算法执行期间添加到$S$ 中的任何数据点都保证是最终的天际点。

Proof. Assume,on the contrary,that point ${p}_{j}$ was added into $S$ ,but it is not a final skyline point. Then ${p}_{j}$ must be dominated by a (final) skyline point,say, ${p}_{i}$ ,whose coordinate on any axis is not larger than the corresponding coordinate of ${p}_{j}$ ,and at least one coordinate is smaller (since ${p}_{i}$ and ${p}_{j}$ are different points). This in turn means that $\operatorname{mindist}\left( {p}_{i}\right)  < \operatorname{mindist}\left( {p}_{j}\right)$ . By Lemma 1, ${p}_{i}$ must be visited before ${p}_{j}$ . In other words,at the time ${p}_{j}$ is processed, ${p}_{i}$ must have already appeared in the skyline list,and hence ${p}_{j}$ should be pruned,which contradicts the fact that ${p}_{j}$ was added in the list.

证明。相反地，假设点${p}_{j}$ 被添加到$S$ 中，但它不是最终的天际点。那么${p}_{j}$ 必定被一个（最终的）天际点（设为${p}_{i}$ ）所支配，该天际点在任何轴上的坐标都不大于${p}_{j}$ 的相应坐标，并且至少有一个坐标更小（因为${p}_{i}$ 和${p}_{j}$ 是不同的点）。这反过来意味着$\operatorname{mindist}\left( {p}_{i}\right)  < \operatorname{mindist}\left( {p}_{j}\right)$ 。根据引理1，${p}_{i}$ 必定在${p}_{j}$ 之前被访问。换句话说，在处理${p}_{j}$ 时，${p}_{i}$ 必定已经出现在天际线列表中，因此${p}_{j}$ 应该被剪枝，这与${p}_{j}$ 被添加到列表中的事实相矛盾。

LEMMA 3. Every data point will be examined, unless one of its ancestor nodes has been pruned.

引理3. 每个数据点都会被检查，除非它的某个祖先节点已被剪枝。

Proof. The proof is obvious since all entries that are not pruned by an existing skyline point are inserted into the heap and examined.

证明。证明很明显，因为所有未被现有天际点剪枝的条目都会被插入堆中并进行检查。

Lemmas 2 and 3 guarantee that, if BBS is allowed to execute until its termination, it will correctly return all skyline points, without reporting any false hits. An important issue regards the dominance checking, which can be expensive if the skyline contains numerous points. In order to speed up this process we insert the skyline points found in a main-memory R-tree. Continuing the example of Figure 6,for instance,only points $i,a,k$ will be inserted (in this order) to the main-memory R-tree. Checking for dominance can now be performed in a way similar to traditional window queries. An entry (i.e., node MBR or data point) is dominated by a skyline point $p$ ,if its lower left point falls inside the dominance region of $p$ ,that is,the rectangle defined by $p$ and the edge of the universe. Figure 8 shows the dominance regions for points $i$ , $a,k$ and two entries; $e$ is dominated by $i$ and $k$ ,while ${e}^{\prime }$ is not dominated by any point (therefore is should be expanded). Note that, in general, most dominance regions will cover a large part of the data space, in which case there will be significant overlap between the intermediate nodes of the main-memory R-tree. Unlike traditional window queries that must retrieve all results, this is not a problem here because we only need to retrieve a single dominance region in order to determine that the entry is dominated (by at least one skyline point).

引理2和引理3保证，如果允许BBS执行到终止，它将正确返回所有天际点，且不会报告任何误报。一个重要的问题是支配检查，如果天际线包含大量点，该检查可能会很昂贵。为了加速这个过程，我们将找到的天际点插入到主存R - 树中。例如，继续图6的示例，只有点$i,a,k$ 会（按此顺序）被插入到主存R - 树中。现在可以以类似于传统窗口查询的方式进行支配检查。如果一个条目（即节点最小边界矩形或数据点）的左下角点落在天际点$p$ 的支配区域内，也就是由$p$ 和整个空间边界所定义的矩形内，那么该条目就被$p$ 所支配。图8显示了点$i$ 、$a,k$ 的支配区域以及两个条目；$e$ 被$i$ 和$k$ 所支配，而${e}^{\prime }$ 未被任何点支配（因此应该被扩展）。请注意，一般来说，大多数支配区域会覆盖数据空间的很大一部分，在这种情况下，主存R - 树的中间节点之间会有显著的重叠。与必须检索所有结果的传统窗口查询不同，这里这不是问题，因为我们只需要检索一个支配区域就可以确定该条目被（至少一个天际点）支配。

To conclude this section, we informally evaluate BBS with respect to the criteria of Hellerstein et al. [1999] and Kossmann et al. [2002], presented in Section 2.6. BBS satisfies property (i) as it returns skyline points instantly in ascending order of their distance to the origin, without having to visit a large part of the R-tree. Lemma 3 ensures property (ii), since every data point is examined unless some of its ancestors is dominated (in which case the point is dominated too). Lemma 2 guarantees property (iii). Property (iv) is also fulfilled because BBS outputs points according to their mindist, which takes into account all dimensions. Regarding user preferences (v), as we discuss in Section 4.1, the user can specify the order of skyline points to be returned by appropriate preference functions. Furthermore, BBS also satisfies property (vi) since it does not require any specialized indexing structure, but (like NN) it can be applied with R-trees or any other data-partitioning method. Furthermore, the same index can be used for any subset of the $d$ dimensions that may be relevant to different users.

为结束本节内容，我们依据第2.6节中介绍的赫勒斯坦（Hellerstein）等人[1999]和科斯曼（Kossmann）等人[2002]提出的标准，对BBS算法进行非形式化评估。BBS算法满足属性（i），因为它能立即按天际点（skyline points）到原点距离的升序返回这些点，而无需访问R树的大部分节点。引理3确保了属性（ii），因为除非某个数据点的祖先节点被支配（在这种情况下，该数据点也会被支配），否则会检查每个数据点。引理2保证了属性（iii）。属性（iv）也能满足，因为BBS算法根据天际点的最小距离（mindist）输出点，该距离考虑了所有维度。关于用户偏好（v），正如我们在第4.1节中讨论的，用户可以通过适当的偏好函数指定要返回的天际点的顺序。此外，BBS算法还满足属性（vi），因为它不需要任何专门的索引结构，但（与最近邻搜索算法（NN）一样）它可以与R树或任何其他数据分区方法一起使用。此外，对于可能与不同用户相关的$d$个维度的任何子集，都可以使用相同的索引。

### 3.3 Analysis of BBS

### 3.3 BBS算法分析

In this section, we first prove that BBS is I/O optimal, meaning that (i) it visits only the nodes that may contain skyline points, and (ii) it does not access the same node twice. Then we provide a theoretical comparison with NN in terms of the number of node accesses and memory consumption (i.e., the heap versus the to-do list sizes). Central to the analysis of BBS is the concept of the skyline search region (SSR), that is, the part of the data space that is not dominated by any skyline point. Consider for instance the running example (with skyline points $i,a,k$ ). The ${SSR}$ is the shaded area in Figure 8 defined by the skyline and the two axes. We start with the following observation.

在本节中，我们首先证明BBS算法在输入/输出（I/O）方面是最优的，即（i）它只访问可能包含天际点的节点，（ii）它不会两次访问同一个节点。然后，我们从节点访问次数和内存消耗（即堆与待办事项列表的大小）方面，对BBS算法和最近邻搜索算法（NN）进行理论比较。BBS算法分析的核心是天际线搜索区域（skyline search region，SSR）的概念，即数据空间中未被任何天际点支配的部分。例如，考虑正在运行的示例（天际点为$i,a,k$）。${SSR}$是图8中由天际线和两个坐标轴定义的阴影区域。我们从以下观察开始。

LEMMA 4. Any skyline algorithm based on $R$ -trees must access all the nodes whose MBRs intersect the SSR.

引理4. 任何基于$R$树的天际线算法都必须访问其最小边界矩形（MBR）与天际线搜索区域（SSR）相交的所有节点。

For instance,although entry ${e}^{\prime }$ in Figure 8 does not contain any skyline points, this cannot be determined unless the child node of ${e}^{\prime }$ is visited.

例如，尽管图8中的条目${e}^{\prime }$不包含任何天际点，但在访问${e}^{\prime }$的子节点之前，无法确定这一点。

LEMMA 5. If an entry e does not intersect the SSR, then there is a skyline point $p$ whose distance from the origin of the axes is smaller than the mindist of $e$ .

引理5. 如果一个条目e与天际线搜索区域（SSR）不相交，那么存在一个天际点$p$，其到坐标轴原点的距离小于$e$的最小距离（mindist）。

Proof. Since $e$ does not intersect the SSR,it must be dominated by at least one skyline point $p$ ,meaning that $p$ dominates the lower-left corner of $e$ . This implies that the distance of $p$ to the origin is smaller than the mindist of $e$ .

证明：由于$e$与天际线搜索区域（SSR）不相交，它必须被至少一个天际点$p$支配，这意味着$p$支配$e$的左下角。这意味着$p$到原点的距离小于$e$的最小距离（mindist）。

THEOREM 6. The number of node accesses performed by BBS is optimal.

定理6. BBS算法执行的节点访问次数是最优的。

Proof. First we prove that BBS only accesses nodes that may contain skyline points. Assume, to the contrary, that the algorithm also visits an entry (let it be $e$ in Figure 8) that does not intersect the SSR. Clearly, $e$ should not be accessed because it cannot contain skyline points. Consider a skyline point that dominates $e$ (e.g., $k$ ). Then,by Lemma 5,the distance of $k$ to the origin is smaller than the mindist of $e$ . According to Lemma 1,BBS visits the entries of the R-tree in ascending order of their mindist to the origin. Hence, $k$ must be processed before $e$ ,meaning that $e$ will be pruned by $k$ ,which contradicts the fact that $e$ is visited.

证明：首先，我们证明BBS算法只访问可能包含天际点的节点。相反，假设该算法还访问了一个与天际线搜索区域（SSR）不相交的条目（设为图8中的$e$）。显然，不应访问$e$，因为它不可能包含天际点。考虑一个支配$e$的天际点（例如，$k$）。然后，根据引理5，$k$到原点的距离小于$e$的最小距离（mindist）。根据引理1，BBS算法按R树条目到原点的最小距离（mindist）的升序访问这些条目。因此，$k$必须在$e$之前处理，这意味着$e$将被$k$修剪，这与访问$e$的事实相矛盾。

In order to complete the proof, we need to show that an entry is not visited multiple times. This is straightforward because entries are inserted into the heap (and expanded) at most once, according to their mindist.

为了完成证明，我们需要证明一个条目不会被多次访问。这很直接，因为根据条目到原点的最小距离（mindist），它们最多被插入到堆中（并展开）一次。

Assuming that each leaf node visited contains exactly one skyline point, the number $N{A}_{BBS}$ of node accesses performed by BBS is at most $s \cdot  h$ (where $s$ is the number of skyline points,and $h$ the height of the R-tree). This bound corresponds to a rather pessimistic case, where BBS has to access a complete path for each skyline point. Many skyline points, however, may be found in the same leaf nodes, or in the same branch of a nonleaf node (e.g., the root of the tree!), so that these nodes only need to be accessed once (our experiments show that in most cases the number of node accesses at each level of the tree is much smaller than $s$ ). Therefore,BBS is at least $d\left( { = s \cdot  h \cdot  d/s \cdot  h}\right)$ times faster than NN (as explained in Section 3.1,the $\operatorname{cost}N{A}_{NN}$ of $\mathrm{{NN}}$ is at least $s \cdot  h \cdot  d$ ). In practice, for $d > 2$ ,the speedup is much larger than $d$ (several orders of magnitude) as $N{A}_{NN} = s \cdot  h \cdot  d$ does not take into account the number $r$ of redundant queries.

假设每个被访问的叶节点恰好包含一个天际点（skyline point），那么BBS执行的节点访问次数$N{A}_{BBS}$最多为$s \cdot  h$（其中$s$是天际点的数量，$h$是R树的高度）。这个界限对应于一个相当悲观的情况，即BBS必须为每个天际点访问一条完整的路径。然而，许多天际点可能位于相同的叶节点中，或者位于非叶节点的同一分支中（例如，树的根节点！），因此这些节点只需要访问一次（我们的实验表明，在大多数情况下，树的每一层的节点访问次数远小于$s$）。因此，BBS至少比NN快$d\left( { = s \cdot  h \cdot  d/s \cdot  h}\right)$倍（如3.1节所述，$\mathrm{{NN}}$的$\operatorname{cost}N{A}_{NN}$至少为$s \cdot  h \cdot  d$）。实际上，对于$d > 2$，加速比远大于$d$（几个数量级），因为$N{A}_{NN} = s \cdot  h \cdot  d$没有考虑冗余查询的数量$r$。

Regarding the memory overhead,the number of entries ${n}_{\text{heap }}$ in the heap of BBS is at most $\left( {f - 1}\right)  \cdot  N{A}_{BBS}$ . This is a pessimistic upper bound,because it assumes that a node expansion removes from the heap the expanded entry and inserts all its $f$ children (in practice,most children will be dominated by some discovered skyline point and pruned). Since for independent dimensions the expected number of skyline points is $s = \Theta \left( {{\left( \ln N\right) }^{d - 1}/\left( {d - 1}\right) !}\right)$ (Buchta [1989]), ${n}_{\text{heap }} \leq  \left( {f - 1}\right)  \cdot  N{A}_{BBS} \approx  \left( {f - 1}\right)  \cdot  h \cdot  s \approx  \left( {f - 1}\right)  \cdot  h \cdot  {\left( \ln N\right) }^{d - 1}/\left( {d - 1}\right) !$ . For $d \geq  3$ and typical values of $N$ and $f$ (e.g., $N = {10}^{5}$ and $f \approx  {100}$ ),the heap size is much smaller than the corresponding to-do list size, which as discussed in Section 3.1 can be in the order of ${\left( d - 1\right) }^{\log N}$ . Furthermore,a heap entry stores $d + 2$ numbers (i.e.,entry id,mindist,and the coordinates of the lower-left corner),as opposed to ${2d}$ numbers for to-do list entries (i.e., $d$ -dimensional ranges).

关于内存开销，BBS堆中的条目数量${n}_{\text{heap }}$最多为$\left( {f - 1}\right)  \cdot  N{A}_{BBS}$。这是一个悲观的上界，因为它假设节点扩展会从堆中移除被扩展的条目，并插入其所有的$f$个子节点（实际上，大多数子节点会被某些已发现的天际点所支配并被修剪掉）。由于对于独立维度，天际点的期望数量为$s = \Theta \left( {{\left( \ln N\right) }^{d - 1}/\left( {d - 1}\right) !}\right)$（Buchta [1989]），所以${n}_{\text{heap }} \leq  \left( {f - 1}\right)  \cdot  N{A}_{BBS} \approx  \left( {f - 1}\right)  \cdot  h \cdot  s \approx  \left( {f - 1}\right)  \cdot  h \cdot  {\left( \ln N\right) }^{d - 1}/\left( {d - 1}\right) !$。对于$d \geq  3$以及$N$和$f$的典型值（例如，$N = {10}^{5}$和$f \approx  {100}$），堆的大小远小于相应的待办列表大小，如3.1节所述，待办列表大小可能达到${\left( d - 1\right) }^{\log N}$的数量级。此外，堆条目存储$d + 2$个数字（即条目ID、最小距离和左下角的坐标），而待办列表条目存储${2d}$个数字（即$d$维范围）。

In summary, the main-memory requirement of BBS is at the same order as the size of the skyline, since both the heap and the main-memory R-tree sizes are at this order. This is a reasonable assumption because (i) skylines are normally small and (ii) previous algorithms, such as index, are based on the same principle. Nevertheless, the size of the heap can be further reduced. Consider that in Figure 9 intermediate node $e$ is visited first and its children (e.g., ${e}_{1}$ ) are inserted into the heap. When ${e}^{\prime }$ is visited afterward ( $e$ and ${e}^{\prime }$ have the same mindist), ${e}_{1}^{\prime }$ can be immediately pruned,because there must exist at least a (not yet discovered) point in the bottom edge of ${e}_{1}$ that dominates ${e}_{1}^{\prime }$ . A similar situation happens if node ${e}^{\prime }$ is accessed first. In this case ${e}_{1}^{\prime }$ is inserted into the heap,but it is removed (before its expansion) when ${e}_{1}$ is added. BBS can easily incorporate this mechanism by checking the contents of the heap before the insertion of an entry $e$ : (i) all entries dominated by $e$ are removed; (ii) if $e$ is dominated by some entry,it is not inserted. We chose not to implement this optimization because it induces some CPU overhead without affecting the number of node accesses,which is optimal (in the above example ${e}_{1}^{\prime }$ would be pruned during its expansion since by that time ${e}_{1}$ will have been visited).

综上所述，BBS（最佳优先天空线搜索算法，Best-First Skyline Search）的主存需求与天际线（skyline）的大小处于同一数量级，因为堆（heap）和主存R树（R-tree）的大小都处于这一数量级。这是一个合理的假设，原因如下：（i）天际线通常较小；（ii）之前的算法（如索引算法）也是基于相同的原理。不过，堆的大小可以进一步减小。考虑图9，中间节点$e$首先被访问，其孩子节点（例如${e}_{1}$）被插入堆中。之后访问${e}^{\prime }$时（$e$和${e}^{\prime }$具有相同的最小距离），${e}_{1}^{\prime }$可以立即被剪枝，因为在${e}_{1}$的底边必定存在至少一个（尚未发现的）点支配${e}_{1}^{\prime }$。如果先访问节点${e}^{\prime }$，也会出现类似的情况。在这种情况下，${e}_{1}^{\prime }$被插入堆中，但当${e}_{1}$被添加时，${e}_{1}^{\prime }$（在其扩展之前）会被移除。BBS可以通过在插入条目$e$之前检查堆的内容轻松整合这一机制：（i）移除所有被$e$支配的条目；（ii）如果$e$被某个条目支配，则不插入。我们选择不实现这一优化，因为它会带来一些CPU开销，而不影响节点访问次数，节点访问次数已经是最优的（在上述示例中，${e}_{1}^{\prime }$在其扩展期间会被剪枝，因为到那时${e}_{1}$已经被访问过）。

<!-- Media -->

<!-- figureText: 8 6 8 10 $e$ 6- 5 - -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_17.jpg?x=643&y=320&w=416&h=377&r=0"/>

Fig. 9. Reducing the size of the heap.

图9. 减小堆的大小。

<!-- Media -->

### 3.4 Incremental Maintenance of the Skyline

### 3.4 天际线的增量维护

The skyline may change due to subsequent updates (i.e., insertions and deletions) to the database, and hence should be incrementally maintained to avoid recomputation. Given a new point $p$ (e.g.,a hotel added to the database),our incremental maintenance algorithm first performs a dominance check on the main-memory R-tree. If $p$ is dominated (by an existing skyline point),it is simply discarded (i.e., it does not affect the skyline); otherwise, BBS performs a window query (on the main-memory R-tree),using the dominance region of $p$ , to retrieve the skyline points that will become obsolete (i.e., those dominated by $p)$ . This query may not retrieve anything (e.g.,Figure 10(a)),in which case the number of skyline points increases by one. Figure 10(b) shows another case, where the dominance region of $p$ covers two points $i,k$ ,which are removed (from the main-memory R-tree). The final skyline consists of only points $a,p$ .

由于对数据库的后续更新（即插入和删除操作），天际线可能会发生变化，因此应该进行增量维护以避免重新计算。给定一个新点$p$（例如，数据库中新增的一家酒店），我们的增量维护算法首先在主存R树上进行支配检查。如果$p$被（现有的天际线点）支配，则直接丢弃它（即它不影响天际线）；否则，BBS使用$p$的支配区域在主存R树上执行窗口查询，以检索将变得过时的天际线点（即那些被$p)$支配的点）。此查询可能检索不到任何内容（例如，图10(a)），在这种情况下，天际线点的数量增加一个。图10(b)展示了另一种情况，其中$p$的支配区域覆盖了两个点$i,k$，这两个点被（从主存R树中）移除。最终的天际线仅由点$a,p$组成。

Handling deletions is more complex. First, if the point removed is not in the skyline (which can be easily checked by the main-memory R-tree using the point's coordinates), no further processing is necessary. Otherwise, part of the skyline must be reconstructed. To illustrate this,assume that point $i$ in Figure 11(a) is deleted. For incremental maintenance, we need to compute the skyline with respect only to the points in the constrained (shaded) area, which is the region exclusively dominated by $i$ (i.e.,not including areas dominated by other skyline points). This is because points (e.g., $e,l$ ) outside the shaded area cannot appear in the new skyline, as they are dominated by at least one other point (i.e., $a$ or $k$ ). As shown in Figure 11(b),the skyline within the exclusive dominance region of $i$ contains two points $h$ and $m$ ,which substitute $i$ in the final skyline (of the whole dataset). In Section 4.1, we discuss skyline computation in a constrained region of the data space.

处理删除操作更为复杂。首先，如果被删除的点不在天际线中（这可以通过主存R树使用该点的坐标轻松检查），则无需进行进一步处理。否则，必须重建部分天际线。为了说明这一点，假设图11(a)中的点$i$被删除。对于增量维护，我们只需要相对于受限（阴影）区域内的点计算天际线，该区域是仅由$i$支配的区域（即不包括其他天际线点支配的区域）。这是因为阴影区域外的点（例如$e,l$）不能出现在新的天际线中，因为它们至少被另一个点（即$a$或$k$）支配。如图11(b)所示，$i$的专属支配区域内的天际线包含两个点$h$和$m$，它们在（整个数据集的）最终天际线中替代了$i$。在4.1节中，我们将讨论数据空间受限区域内的天际线计算。

<!-- Media -->

<!-- figureText: the dominance ${y}_{{10}\text{-}}$ the original skyline the dominance region of $p$ (b) Skyline cardinality decreases 8 region of $p$ 7 6 - the original 4 - skyline 3 10 (a) Skyline cardinality increases -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_18.jpg?x=446&y=318&w=830&h=393&r=0"/>

Fig. 10. Incremental skyline maintenance for insertion.

图10. 插入操作的天际线增量维护。

<!-- figureText: 9 9 8 6 $h$ $m$ (b) The skyline after removing $i$ exclusive dominance - O region of point $i$ ${\mathrm{O}}_{d}$ 6 deleted (a) Finding a skyline within a region -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_18.jpg?x=441&y=819&w=843&h=385&r=0"/>

Fig. 11. Incremental skyline maintenance for deletion.

图11. 删除操作的天际线增量维护。

<!-- Media -->

Except for the above case of deletion, incremental skyline maintenance involves only main-memory operations. Given that the skyline points constitute only a small fraction of the database, the probability of deleting a skyline point is expected to be very low. In extreme cases (e.g., bulk updates, large number of skyline points) where insertions/deletions frequently affect the skyline, we may adopt the following "lazy" strategy to minimize the number of disk accesses: after deleting a skyline point $p$ ,we do not compute the constrained skyline immediately,but add $p$ to a buffer. For each subsequent insertion,if $p$ is dominated by a new point ${p}^{\prime }$ ,we remove it from the buffer because all the points potentially replacing $p$ would become obsolete anyway as they are dominated by ${p}^{\prime }$ (the insertion of ${p}^{\prime }$ may also render other skyline points obsolete). When there are no more updates or a user issues a skyline query, we perform a single constrained skyline search, setting the constraint region to the union of the exclusive dominance regions of the remaining points in the buffer, which is emptied afterward.

除了上述删除情况外，增量天际线维护仅涉及主内存操作。鉴于天际线点仅占数据库的一小部分，删除天际线点的概率预计非常低。在极端情况下（例如批量更新、大量天际线点），插入/删除操作频繁影响天际线时，我们可以采用以下“惰性”策略来最小化磁盘访问次数：删除一个天际线点 $p$ 后，我们不会立即计算受限天际线，而是将 $p$ 添加到一个缓冲区。对于后续的每次插入，如果 $p$ 被一个新点 ${p}^{\prime }$ 支配，我们将其从缓冲区中移除，因为所有可能替换 $p$ 的点无论如何都会变得过时，因为它们被 ${p}^{\prime }$ 支配（ ${p}^{\prime }$ 的插入也可能使其他天际线点过时）。当没有更多更新或用户发出天际线查询时，我们执行一次受限天际线搜索，将约束区域设置为缓冲区中剩余点的排他支配区域的并集，之后清空缓冲区。

<!-- Media -->

<!-- figureText: 9 ${N}_{l}$ ${N}_{2}$ ${N}_{6}$ ① $d$ ${N}_{7}$ IV. 10 8 - 7 6 * 5 * 4 - 3 . ${N}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_19.jpg?x=642&y=320&w=416&h=357&r=0"/>

Fig. 12. Constrained query example.

图12. 受限查询示例。

<!-- Media -->

## 4. VARIATIONS OF SKYLINE QUERIES

## 4. 天际线查询的变体

In this section we propose novel variations of skyline search, and illustrate how BBS can be applied for their processing. In particular, Section 4.1 discusses constrained skylines, Section 4.2 ranked skylines, Section 4.3 group-by skylines,Section 4.4 dynamic skylines,Section 4.5 enumerating and $K$ -dominating queries, and Section 4.6 skybands.

在本节中，我们提出了新颖的天际线搜索变体，并说明如何将BBS算法应用于它们的处理。具体而言，4.1节讨论受限天际线，4.2节讨论排序天际线，4.3节讨论分组天际线，4.4节讨论动态天际线，4.5节讨论枚举和 $K$ -支配查询，4.6节讨论天际带。

### 4.1 Constrained Skyline

### 4.1 受限天际线

Given a set of constraints, a constrained skyline query returns the most interesting points in the data space defined by the constraints. Typically, each constraint is expressed as a range along a dimension and the conjunction of all constraints forms a hyperrectangle (referred to as the constraint region) in the $d$ -dimensional attribute space. Consider the hotel example,where a user is interested only in hotels whose prices ( $y$ axis) are in the range $\left\lbrack  {4,7}\right\rbrack$ . The skyline in this case contains points $g,f$ ,and $l$ (Figure 12),as they are the most interesting hotels in the specified price range. Note that $d$ (which also satisfies the constraints) is not included as it is dominated by $g$ . The constrained query can be expressed using the syntax of Borzsonyi et al. [2001] and the where clause: Select ${}^{ * }$ ,From Hotels,Where Price $\in  \left\lbrack  {4,7}\right\rbrack$ ,Skyline of Price min,Distance min. In addition, constrained queries are useful for incremental maintenance of the skyline in the presence of deletions (as discussed in Section 3.4).

给定一组约束条件，受限天际线查询返回由这些约束条件定义的数据空间中最有趣的点。通常，每个约束条件表示为一个维度上的范围，所有约束条件的合取在 $d$ 维属性空间中形成一个超矩形（称为约束区域）。以酒店示例为例，用户只对价格（ $y$ 轴）在 $\left\lbrack  {4,7}\right\rbrack$ 范围内的酒店感兴趣。在这种情况下，天际线包含点 $g,f$ 和 $l$ （图12），因为它们是指定价格范围内最有趣的酒店。请注意， $d$ （它也满足约束条件）不包括在内，因为它被 $g$ 支配。受限查询可以使用Borzsonyi等人 [2001] 的语法和where子句来表示：Select ${}^{ * }$ ,From Hotels,Where Price $\in  \left\lbrack  {4,7}\right\rbrack$ ,Skyline of Price min,Distance min。此外，受限查询对于存在删除操作时天际线的增量维护很有用（如3.4节所述）。

BBS can easily process such queries. The only difference with respect to the original algorithm is that entries not intersecting the constraint region are pruned (i.e., not inserted in the heap). Table IV shows the contents of the heap during the processing of the query in Figure 12. The same concept can also be applied when the constraint region is not a (hyper-) rectangle, but an arbitrary area in the data space.

BBS算法可以轻松处理此类查询。与原始算法的唯一区别是，不与约束区域相交的条目会被修剪（即不插入堆中）。表IV显示了处理图12中的查询时堆的内容。当约束区域不是（超）矩形，而是数据空间中的任意区域时，相同的概念也可以应用。

The NN algorithm can also support constrained skylines with a similar modification. In particular,the first nearest neighbor (e.g., $g$ ) is retrieved in the constraint region using constrained nearest-neighbor search [Ferhatosman-oglu et al. 2001]. Then, each space subdivision is the intersection of the original subdivision (area to be searched by NN for the unconstrained query) and the constraint region. The index method can benefit from the constraints, by starting with the batches at the beginning of the constraint ranges (instead of the top of the lists). Bitmap can avoid loading the juxtapositions (see Section 2.3) for points that do not satisfy the query constraints, and D&C may discard, during the partitioning step, points that do not belong to the constraint region. For BNL and SFS, the only difference with respect to regular skyline retrieval is that only points in the constraint region are inserted in the self-organizing list.

NN算法也可以通过类似的修改来支持受限天际线。具体而言，使用受限最近邻搜索 [Ferhatosman-oglu等人2001] 在约束区域中检索第一个最近邻（例如 $g$ ）。然后，每个空间细分是原始细分（无约束查询时NN要搜索的区域）与约束区域的交集。索引方法可以从约束条件中受益，从约束范围开始处的批次（而不是列表顶部）开始。位图可以避免加载不满足查询约束条件的点的并置（见2.3节），并且分治法（D&C）在分区步骤中可以丢弃不属于约束区域的点。对于BNL和SFS算法，与常规天际线检索的唯一区别是，只有约束区域内的点才会插入自组织列表中。

<!-- Media -->

Table IV. Heap Contents for Constrained Query

表IV. 受限查询的堆内容

<table><tr><td>Action</td><td>Heap Contents</td><td>$S$</td></tr><tr><td>Access root</td><td>$\left\langle  {{e}_{7},4}\right\rangle   < \left\langle  {{e}_{6},6}\right\rangle$</td><td>0</td></tr><tr><td>Expand ${e}_{7}$</td><td>$\left\langle  {{e}_{3},5}\right\rangle   > \left\langle  {{e}_{6},6}\right\rangle   > \left\langle  {{e}_{4},{10}}\right\rangle$</td><td>0</td></tr><tr><td>Expand ${e}_{3}$</td><td>${\epsilon }_{6},6 > {\epsilon }_{4},{10} > \epsilon  < g,{11} >$</td><td>0</td></tr><tr><td>Expand ${e}_{6}$</td><td>${e}_{4},{10} > {4g},{11} > {e}_{2},{11} >$</td><td>0</td></tr><tr><td>Expand ${e}_{4}$</td><td>< $g,$ 11>< ${e}_{\text{2}},$ 11><l,14></td><td>$\{ g\}$</td></tr><tr><td>Expand ${e}_{2}$</td><td><f, 12><d, 13><l, 14></td><td>$\{ g,f,l\}$</td></tr></table>

<table><tbody><tr><td>操作</td><td>堆内容</td><td>$S$</td></tr><tr><td>访问根节点</td><td>$\left\langle  {{e}_{7},4}\right\rangle   < \left\langle  {{e}_{6},6}\right\rangle$</td><td>0</td></tr><tr><td>展开 ${e}_{7}$</td><td>$\left\langle  {{e}_{3},5}\right\rangle   > \left\langle  {{e}_{6},6}\right\rangle   > \left\langle  {{e}_{4},{10}}\right\rangle$</td><td>0</td></tr><tr><td>展开 ${e}_{3}$</td><td>${\epsilon }_{6},6 > {\epsilon }_{4},{10} > \epsilon  < g,{11} >$</td><td>0</td></tr><tr><td>展开 ${e}_{6}$</td><td>${e}_{4},{10} > {4g},{11} > {e}_{2},{11} >$</td><td>0</td></tr><tr><td>展开 ${e}_{4}$</td><td>< $g,$ 11>< ${e}_{\text{2}},$ 11><l,14></l,14></td><td>$\{ g\}$</td></tr><tr><td>展开 ${e}_{2}$</td><td><f, 12=""><d, 13=""><l, 14=""></l,></d,></f,></td><td>$\{ g,f,l\}$</td></tr></tbody></table>

<!-- Media -->

### 4.2 Ranked Skyline

### 4.2 排序天际线

Given a set of points in the $d$ -dimensional space ${\left\lbrack  0,1\right\rbrack  }^{d}$ ,a ranked (top- $K$ ) skyline query (i) specifies a parameter $K$ ,and a preference function $f$ which is monotone on each attribute,(ii) and returns the $K$ skyline points $p$ that have the minimum score according to the input function. Consider the running example,where $K = 2$ and the preference function is $f\left( {x,y}\right)  = x + 3{y}^{2}$ . The output skyline points should be $< k,{12} > , < i,{15} >$ in this order (the number with each point indicates its score). Such ranked skyline queries can be expressed using the syntax of Borzsonyi et al. [2001] combined with the order by and stop after clauses: Select ${}^{ * }$ ,From Hotels,Skyline of Price ${min}$ ,Distance ${min}$ ,order by Price $+ 3 \cdot  \operatorname{sqr}$ (Distance),stop after 2 .

给定 $d$ 维空间 ${\left\lbrack  0,1\right\rbrack  }^{d}$ 中的一组点，一个排序（前 $K$ 个）天际线查询 (i) 指定一个参数 $K$ ，以及一个在每个属性上单调的偏好函数 $f$ ；(ii) 并根据输入函数返回得分最小的 $K$ 个天际线点 $p$ 。考虑运行示例，其中 $K = 2$ ，偏好函数为 $f\left( {x,y}\right)  = x + 3{y}^{2}$ 。输出的天际线点应按此顺序排列 $< k,{12} > , < i,{15} >$ （每个点旁边的数字表示其得分）。此类排序天际线查询可以使用博尔佐尼等人 [2001] 的语法，结合 order by 和 stop after 子句来表示：Select ${}^{ * }$ ,From Hotels,Skyline of Price ${min}$ ,Distance ${min}$ ,order by Price $+ 3 \cdot  \operatorname{sqr}$ (Distance),stop after 2 。

BBS can easily handle such queries by modifying the mindist definition to reflect the preference function (i.e.,the mindist of a point with coordinates $x$ and $y$ equals $x + 3{y}^{2}$ ). The mindist of an intermediate entry equals the score of its lower-left point. Furthermore,the algorithm terminates after exactly $K$ points have been reported. Due to the monotonicity of $f$ ,it is easy to prove that the output points are indeed skyline points. The only change with respect to the original algorithm is the order of entries visited, which does not affect the correctness or optimality of BBS because in any case an entry will be considered after all entries that dominate it.

通过修改最小距离（mindist）的定义以反映偏好函数（即，坐标为 $x$ 和 $y$ 的点的最小距离等于 $x + 3{y}^{2}$ ），最佳优先搜索（BBS）算法可以轻松处理此类查询。中间条目的最小距离等于其左下角点的得分。此外，该算法在准确报告 $K$ 个点后终止。由于 $f$ 的单调性，很容易证明输出的点确实是天际线点。与原始算法唯一的区别是访问条目的顺序，这并不影响 BBS 的正确性或最优性，因为在任何情况下，一个条目都会在所有支配它的条目之后被考虑。

None of the other algorithms can answer this query efficiently. Specifically, BNL, D&C, bitmap, and index (as well as SFS if the scoring function is different from the sorting one) require first retrieving the entire skyline, sorting the skyline points by their scores,and then outputting the best $K$ ones. On the other hand,although NN can be used with all monotone functions, its application to ranked skyline may incur almost the same cost as that of a complete skyline. This is because, due to its divide-and-conquer nature, it is difficult to establish the termination criterion. If,for instance, $K = 2,\mathrm{{NN}}$ must perform $d$ queries after the first nearest neighbor (skyline point) is found, compare their results, and return the one with the minimum score. The situation is more complicated when $K$ is large where the output of numerous queries must be compared.

其他算法都无法高效地回答此查询。具体而言，块嵌套循环（BNL）、分治法（D&C）、位图法和索引法（以及如果评分函数与排序函数不同的顺序过滤扫描法（SFS））都需要先检索整个天际线，按得分对天际线点进行排序，然后输出得分最高的 $K$ 个点。另一方面，虽然最近邻（NN）算法可用于所有单调函数，但将其应用于排序天际线可能会产生与完整天际线查询几乎相同的成本。这是因为，由于其分治性质，很难确定终止准则。例如，如果 $K = 2,\mathrm{{NN}}$ 必须在找到第一个最近邻（天际线点）后执行 $d$ 次查询，比较它们的结果，并返回得分最小的那个。当 $K$ 很大时，情况会更复杂，因为必须比较大量查询的输出。

### 4.3 Group-By Skyline

### 4.3 分组天际线

Assume that for each hotel, in addition to the price and distance, we also store its class (i.e., 1-star, 2-star, ... , 5-star). Instead of a single skyline covering all three attributes, a user may wish to find the individual skyline in each class. Conceptually, this is equivalent to grouping the hotels by their classes, and then computing the skyline for each group; that is, the number of skylines equals the cardinality of the group-by attribute domain. Using the syntax of Borzsonyi et al. [2001], the query can be expressed as Select *, From Hotels, Skyline of Price $\min$ ,Distance $\min$ ,Class diff (i.e.,the group-by attribute is specified by the keyword diff).

假设对于每家酒店，除了价格和距离之外，我们还存储其星级（即，一星、二星……五星）。用户可能希望找到每个星级的单独天际线，而不是涵盖所有三个属性的单个天际线。从概念上讲，这相当于按星级对酒店进行分组，然后为每个组计算天际线；也就是说，天际线的数量等于分组属性域的基数。使用博尔佐尼等人 [2001] 的语法，该查询可以表示为 Select *, From Hotels, Skyline of Price $\min$ ,Distance $\min$ ,Class diff（即，分组属性由关键字 diff 指定）。

One straightforward way to support group-by skylines is to create a separate $\mathrm{R}$ -tree for the hotels in the same class,and then invoke BBS in each tree. Separating one attribute (i.e., class) from the others, however, would compromise the performance of queries involving all the attributes. ${}^{4}$ In the following, we present a variation of BBS which operates on a single R-tree that indexes all the attributes. For the above example, the algorithm (i) stores the skyline points already found for each class in a separate main-memory $2\mathrm{D}$ R-tree and (ii) maintains a single heap containing all the visited entries. The difference is that the sorting key is computed based only on price and distance (i.e., excluding the group-by attribute). Whenever a data point is retrieved, we perform the dominance check at the corresponding main-memory R-tree (i.e., for its class), and insert it into the tree only if it is not dominated by any existing point.

支持分组天际线的一种直接方法是为同一星级的酒店创建单独的 $\mathrm{R}$ -树，然后在每棵树中调用最佳优先搜索（BBS）算法。然而，将一个属性（即星级）与其他属性分开会影响涉及所有属性的查询性能。${}^{4}$ 下面，我们介绍一种 BBS 的变体，它在对所有属性进行索引的单个 R - 树中操作。对于上述示例，该算法 (i) 将每个星级已找到的天际线点存储在单独的主存 $2\mathrm{D}$ R - 树中；(ii) 维护一个包含所有已访问条目的单一堆。不同之处在于，排序键仅基于价格和距离计算（即，不包括分组属性）。每当检索到一个数据点时，我们在相应的主存 R - 树（即，针对其星级）中进行支配检查，并且仅当该点不被任何现有点支配时才将其插入树中。

On the other hand the dominance check for each intermediate entry $e$ (performed before its insertion into the heap, and during its expansion) is more complicated,because $e$ is likely to contain hotels of several classes (we can identify the potential classes included in $e$ by its projection on the corresponding axis). First, its MBR (i.e., a 3D box) is projected onto the price-distance plane and the lower-left corner $c$ is obtained. We need to visit $e$ ,only if $c$ is not dominated in some main-memory R-tree corresponding to a class covered by $e$ . Consider, for instance,that the projection of $e$ on the class dimension is $\left\lbrack  {2,4}\right\rbrack$ (i.e., $e$ may contain only hotels with 2,3,and 4 stars). If the lower-left point of $e$ (on the price-distance plane) is dominated in all three classes, $e$ cannot contribute any skyline point. When the number of distinct values of the group-by attribute is large, the skylines may not fit in memory. In this case, we can perform the algorithm in several passes, each pass covering a number of continuous values. The processing cost will be higher as some nodes (e.g., the root) may be visited several times.

另一方面，对每个中间条目 $e$ 进行的支配检查（在将其插入堆之前以及在扩展过程中执行）更为复杂，因为 $e$ 可能包含多个等级的酒店（我们可以通过其在相应轴上的投影来识别 $e$ 中包含的潜在等级）。首先，将其最小边界矩形（MBR，即一个三维盒子）投影到价格 - 距离平面上，得到左下角点 $c$。仅当 $c$ 在与 $e$ 所涵盖的某个等级相对应的某个主存 R - 树中不被支配时，我们才需要访问 $e$。例如，假设 $e$ 在等级维度上的投影是 $\left\lbrack  {2,4}\right\rbrack$（即 $e$ 可能仅包含二星级、三星级和四星级酒店）。如果 $e$ 的左下角点（在价格 - 距离平面上）在所有这三个等级中都被支配，那么 $e$ 就不能贡献任何天际线点。当分组属性的不同值的数量很大时，天际线可能无法全部存入内存。在这种情况下，我们可以分多次执行该算法，每次处理一定数量的连续值。由于某些节点（例如根节点）可能会被多次访问，处理成本将会更高。

It is not clear how to extend NN, D&C, index, or bitmap for group-by skylines beyond the naïve approach, that is, invoke the algorithms for every value of the group-by attribute (e.g., each time focusing on points belonging to a specific group), which, however, would lead to high processing cost. BNL and SFS can be applied in this case by maintaining separate temporary skylines for each class value (similar to the main memory R-trees of BBS).

目前尚不清楚如何在简单方法之外，将最近邻（NN）、分治法（D&C）、索引或位图方法扩展应用于分组天际线查询。简单方法是针对分组属性的每个值调用相应算法（例如，每次聚焦于属于特定组的点），但这会导致较高的处理成本。在这种情况下，可以通过为每个等级值维护单独的临时天际线（类似于块嵌套循环（BBS）算法中的主存 R - 树）来应用块嵌套循环（BNL）和排序过滤扫描（SFS）算法。

---

<!-- Footnote -->

${}^{4}$ A 3D skyline in this case should maximize the value of the class (e.g.,given two hotels with the same price and distance, the one with more stars is preferable).

${}^{4}$ 在这种情况下，三维天际线应使等级值最大化（例如，给定两家价格和距离相同的酒店，星级更高的那家更可取）。

<!-- Footnote -->

---

### 4.4 Dynamic Skyline

### 4.4 动态天际线

Assume a database containing points in a $d$ -dimensional space with axes ${d}_{1},{d}_{2},\ldots ,{d}_{d}$ . A dynamic skyline query specifies $m$ dimension functions ${f}_{1}$ , ${f}_{2},\ldots ,{f}_{m}$ such that each function ${f}_{i}\left( {1 \leq  i \leq  m}\right)$ takes as parameters the coordinates of the data points along a subset of the $d$ axes. The goal is to return the skyline in the new data space with dimensions defined by ${f}_{1},{f}_{2},\ldots ,{f}_{m}$ . Consider, for instance, a database that stores the following information for each hotel: (i) its $x$ and (ii) $y$ coordinates,and (iii) its price (i.e.,the database contains three dimensions). Then,a user specifies his/her current location $\left( {{u}_{x},{u}_{y}}\right)$ ,and requests the most interesting hotels, where preference must take into consideration the hotels' proximity to the user (in terms of Euclidean distance) and the price. Each point $p$ with coordinates $\left( {{p}_{x},{p}_{y},{p}_{z}}\right)$ in the original 3D space is transformed to a point ${p}^{\prime }$ in the $2\mathrm{D}$ space with coordinates $\left( {{f}_{1}\left( {{p}_{x},{p}_{y}}\right) ,{f}_{2}\left( {p}_{z}\right) }\right)$ , where the dimension functions ${f}_{1}$ and ${f}_{2}$ are defined as

假设一个数据库包含 $d$ 维空间中的点，其坐标轴为 ${d}_{1},{d}_{2},\ldots ,{d}_{d}$。动态天际线查询指定了 $m$ 个维度函数 ${f}_{1}$、${f}_{2},\ldots ,{f}_{m}$，使得每个函数 ${f}_{i}\left( {1 \leq  i \leq  m}\right)$ 以数据点在 $d$ 个坐标轴的一个子集上的坐标作为参数。目标是返回由 ${f}_{1},{f}_{2},\ldots ,{f}_{m}$ 定义的新数据空间中的天际线。例如，考虑一个数据库，它为每家酒店存储以下信息：（i）其 $x$ 坐标，（ii）$y$ 坐标，以及（iii）其价格（即该数据库包含三个维度）。然后，用户指定其当前位置 $\left( {{u}_{x},{u}_{y}}\right)$，并请求最感兴趣的酒店，其中偏好必须考虑酒店与用户的接近程度（以欧几里得距离衡量）和价格。原始三维空间中坐标为 $\left( {{p}_{x},{p}_{y},{p}_{z}}\right)$ 的每个点 $p$ 被转换为 $2\mathrm{D}$ 空间中坐标为 $\left( {{f}_{1}\left( {{p}_{x},{p}_{y}}\right) ,{f}_{2}\left( {p}_{z}\right) }\right)$ 的点 ${p}^{\prime }$，其中维度函数 ${f}_{1}$ 和 ${f}_{2}$ 定义如下

$$
{f}_{1}\left( {{p}_{x},{p}_{y}}\right)  = \sqrt{{\left( {p}_{x} - {u}_{x}\right) }^{2} + {\left( {p}_{y} - {u}_{y}\right) }^{2}},\;\text{ and }\;{f}_{2}\left( {p}_{z}\right)  = {p}_{z}.
$$

The terms original and dynamic space refer to the original $d$ -dimensional data space and the space with computed dimensions (from ${f}_{1},{f}_{2},\ldots ,{f}_{m}$ ),respectively. Correspondingly, we refer to the coordinates of a point in the original space as original coordinates, while to those of the point in the dynamic space as dynamic coordinates.

术语“原始空间”和“动态空间”分别指原始的 $d$ 维数据空间和具有计算维度（根据 ${f}_{1},{f}_{2},\ldots ,{f}_{m}$ 计算得出）的空间。相应地，我们将点在原始空间中的坐标称为原始坐标，而将该点在动态空间中的坐标称为动态坐标。

BBS is applicable to dynamic skylines by expanding entries in the heap according to their mindist in the dynamic space (which is computed on-the-fly when the entry is considered for the first time). In particular, the mindist of a leaf entry (data point) $e$ with original coordinates $\left( {{e}_{x},{e}_{y},{e}_{z}}\right)$ ,equals $\sqrt{{\left( {e}_{x} - {u}_{x}\right) }^{2} + {\left( {e}_{y} - {u}_{y}\right) }^{2}} + {e}_{z}$ . The mindist of an intermediate entry $e$ whose MBR has ranges $\left\lbrack  {{e}_{x0},{e}_{x1}}\right\rbrack  \left\lbrack  {{e}_{y0},{e}_{y1}}\right\rbrack  \left\lbrack  {{e}_{z0},{e}_{z1}}\right\rbrack$ is computed as mindist $\left( \left\lbrack  {{e}_{x0},{e}_{x1}}\right\rbrack  \right.$ $\left. {\left\lbrack  {{e}_{y0},{e}_{y1}}\right\rbrack  ,\left( {{u}_{x},{u}_{y}}\right) }\right)  + {e}_{z0}$ ,where the first term equals the mindist between point $\left( {{u}_{x},{u}_{y}}\right)$ to the $2\mathrm{D}$ rectangle $\left\lbrack  {{e}_{x0},{e}_{x1}}\right\rbrack  \left\lbrack  {{e}_{y0},{e}_{y1}}\right\rbrack$ . Furthermore,notice that the concept of dynamic skylines can be employed in conjunction with ranked and constraint queries (i.e.,find the top five hotels within $1\mathrm{\;{km}}$ ,given that the price is twice as important as the distance). BBS can process such queries by appropriate modification of the mindist definition (the $z$ coordinate is multiplied by 2) and by constraining the search region $\left( {{f}_{1}\left( {x,y}\right)  \leq  1\mathrm{\;{km}}}\right)$ .

通过根据堆中条目在动态空间中的最小距离（该距离在首次考虑该条目时即时计算）扩展堆中的条目，BBS算法适用于动态天际线。具体而言，具有原始坐标 $\left( {{e}_{x},{e}_{y},{e}_{z}}\right)$ 的叶节点条目（数据点） $e$ 的最小距离等于 $\sqrt{{\left( {e}_{x} - {u}_{x}\right) }^{2} + {\left( {e}_{y} - {u}_{y}\right) }^{2}} + {e}_{z}$ 。中间条目 $e$ 的最小边界矩形（MBR）范围为 $\left\lbrack  {{e}_{x0},{e}_{x1}}\right\rbrack  \left\lbrack  {{e}_{y0},{e}_{y1}}\right\rbrack  \left\lbrack  {{e}_{z0},{e}_{z1}}\right\rbrack$ ，其最小距离计算为 mindist $\left( \left\lbrack  {{e}_{x0},{e}_{x1}}\right\rbrack  \right.$ $\left. {\left\lbrack  {{e}_{y0},{e}_{y1}}\right\rbrack  ,\left( {{u}_{x},{u}_{y}}\right) }\right)  + {e}_{z0}$ ，其中第一项等于点 $\left( {{u}_{x},{u}_{y}}\right)$ 到 $2\mathrm{D}$ 矩形 $\left\lbrack  {{e}_{x0},{e}_{x1}}\right\rbrack  \left\lbrack  {{e}_{y0},{e}_{y1}}\right\rbrack$ 之间的最小距离。此外，请注意，动态天际线的概念可以与排名查询和约束查询结合使用（例如，在 $1\mathrm{\;{km}}$ 范围内找出前五名酒店，假设价格的重要性是距离的两倍）。BBS可以通过适当修改最小距离的定义（将 $z$ 坐标乘以2）并约束搜索区域 $\left( {{f}_{1}\left( {x,y}\right)  \leq  1\mathrm{\;{km}}}\right)$ 来处理此类查询。

Regarding the applicability of the previous methods, BNL still applies because it evaluates every point, whose dynamic coordinates can be computed on-the-fly. The optimizations, of SFS, however, are now useless since the order of points in the dynamic space may be different from that in the original space. $\mathrm{D}\& \mathrm{C}$ and $\mathrm{{NN}}$ can also be modified for dynamic queries with the transformations described above, suffering, however, from the same problems as the original algorithms. Bitmap and index are not applicable because these methods rely on pre-computation, which provides little help when the dimensions are defined dynamically.

关于之前方法的适用性，BNL算法仍然适用，因为它会评估每个点，而这些点的动态坐标可以即时计算。然而，SFS算法的优化现在变得无用，因为点在动态空间中的顺序可能与原始空间中的顺序不同。 $\mathrm{D}\& \mathrm{C}$ 和 $\mathrm{{NN}}$ 也可以通过上述变换进行修改以处理动态查询，但会遇到与原始算法相同的问题。位图和索引方法不适用，因为这些方法依赖于预计算，而当维度是动态定义时，预计算的帮助不大。

### 4.5 Enumerating and $K$ -Dominating Queries

### 4.5 枚举查询和 $K$ -支配查询

Enumerating queries return,for each skyline point $p$ ,the number of points dominated by $p$ . This information provides some measure of "goodness" for the skyline points. In the running example,for instance,hotel $i$ may be more interesting than the other skyline points since it dominates nine hotels as opposed to two for hotels $a$ and $k$ . Let’s call $\operatorname{num}\left( p\right)$ the number of points dominated by point $p$ . A straightforward approach to process such queries involves two steps: (i) first compute the skyline and (ii) for each skyline point $p$ apply a query window in the data R-tree and count the number of points $\operatorname{num}\left( p\right)$ falling inside the dominance region of $p$ . Notice that since all (except for the skyline) points are dominated,all the nodes of the R-tree will be accessed by some query. Furthermore, due to the large size of the dominance regions, numerous R-tree nodes will be accessed by several window queries. In order to avoid multiple node visits, we apply the inverse procedure, that is, we scan the data file and for each point we perform a query in the main-memory R-tree to find the dominance regions that contain it. The corresponding counters $\operatorname{num}\left( p\right)$ of the skyline points are then increased accordingly.

枚举查询会为每个天际线点 $p$ 返回被 $p$ 支配的点的数量。此信息为天际线点提供了某种“优劣”衡量标准。例如，在当前示例中，酒店 $i$ 可能比其他天际线点更有吸引力，因为它支配了九家酒店，而酒店 $a$ 和 $k$ 仅支配了两家。我们将被点 $p$ 支配的点的数量称为 $\operatorname{num}\left( p\right)$ 。处理此类查询的一种直接方法包括两个步骤：（i）首先计算天际线；（ii）对于每个天际线点 $p$ ，在数据R树中应用查询窗口，并统计落在 $p$ 支配区域内的点的数量 $\operatorname{num}\left( p\right)$ 。请注意，由于除天际线点之外的所有点都被支配，R树的所有节点都会被某些查询访问。此外，由于支配区域较大，多个窗口查询会访问大量的R树节点。为了避免多次访问节点，我们采用反向过程，即扫描数据文件，并为每个点在主内存R树中执行查询，以找到包含该点的支配区域。然后相应地增加天际线点的计数器 $\operatorname{num}\left( p\right)$ 。

An interesting variation of the problem is the $K$ -dominating query,which retrieves the $K$ points that dominate the largest number of other points. Strictly speaking, this is not a skyline query, since the result does not necessarily contain skyline points. If $K = 3$ ,for instance,the output should include hotels $i,h$ ,and $m$ ,with $\operatorname{num}\left( i\right)  = 9,\operatorname{num}\left( h\right)  = 7$ ,and $\operatorname{num}\left( m\right)  = 5$ . In order to obtain the result, we first perform an enumerating query that returns the skyline points and the number of points that they dominate. This information for the first $K = 3$ points is inserted into a list sorted according to num(p),that is,list $=$ $< i,9 > , < a,2 > , < k,2 >$ . The first element of the list (point $i$ ) is the first result of the 3-dominating query. Any other point potentially in the result should be in the (exclusive) dominance region of $i$ ,but not in the dominance region of $a$ ,or $k$ (i.e.,in the shaded area of Figure 13(a)); otherwise,it would dominate fewer points than $a$ ,or $k$ . In order to retrieve the candidate points,we perform a local skyline query ${S}^{\prime }$ in this region (i.e.,a constrained query),after removing $i$ from $S$ and reporting it to the user. ${S}^{\prime }$ contains points $h$ and $m$ . The new skyline ${S}_{1} = \left( {S-\{ i\} }\right)  \cup  {S}^{\prime }$ is shown in Figure 13(b).

该问题的一个有趣变体是 $K$ -支配查询，它会检索支配其他点数量最多的 $K$ 个点。严格来说，这并非天际线查询，因为查询结果不一定包含天际线点。例如，如果 $K = 3$ ，那么输出应包含酒店 $i,h$ 和 $m$ ，其中 $\operatorname{num}\left( i\right)  = 9,\operatorname{num}\left( h\right)  = 7$ 和 $\operatorname{num}\left( m\right)  = 5$ 。为了得到结果，我们首先执行一个枚举查询，返回天际线点以及它们所支配的点的数量。前 $K = 3$ 个点的这些信息会被插入到一个根据 num(p) 排序的列表中，即列表 $=$ $< i,9 > , < a,2 > , < k,2 >$ 。列表的第一个元素（点 $i$ ）是 3 - 支配查询的第一个结果。结果中其他可能的点应位于 $i$ 的（排他）支配区域内，但不在 $a$ 或 $k$ 的支配区域内（即图 13(a) 中的阴影区域）；否则，它所支配的点会比 $a$ 或 $k$ 少。为了检索候选点，在从 $S$ 中移除 $i$ 并将其报告给用户后，我们在该区域执行局部天际线查询 ${S}^{\prime }$ （即约束查询）。 ${S}^{\prime }$ 包含点 $h$ 和 $m$ 。新的天际线 ${S}_{1} = \left( {S-\{ i\} }\right)  \cup  {S}^{\prime }$ 如图 13(b) 所示。

Since $h$ and $m$ do not dominate each other,they may each dominate at most seven points (i.e., $\operatorname{num}\left( i\right)  - 2$ ),meaning that they are candidates for the 3-dominating query. In order to find the actual number of points dominated, we perform a window query in the data R-tree using the dominance regions of $h$ and $m$ as query windows. After this step, $\langle h,7\rangle$ and $\langle m,5\rangle$ replace the previous candidates $< a,2 > , < k,2 >$ in the list. Point $h$ is the second result of the 3-dominating query and is output to the user. Then, the process is repeated for the points that belong to the dominance region of $h$ ,but not in the dominance regions of other points in ${S}_{1}$ (i.e.,shaded area in Figure 13(c)). The new skyline ${S}_{2} = \left( {{S}_{1}-\{ h\} }\right)  \cup  \{ c,g\}$ is shown in Figure 13(d). Points $c$ and $g$ may dominate at most five points each (i.e., $\operatorname{num}\left( h\right)  - 2$ ),meaning that they cannot outnumber $m$ . Hence,the query terminates with $< i,9 >  < h,7 >  < m,5 >$ as the final result. In general, the algorithm can be thought of as skyline "peeling," since it computes local skylines at the points that have the largest dominance.

由于 $h$ 和 $m$ 彼此不支配，它们各自最多可支配七个点（即 $\operatorname{num}\left( i\right)  - 2$ ），这意味着它们是 3 - 支配查询的候选点。为了确定实际支配的点的数量，我们使用 $h$ 和 $m$ 的支配区域作为查询窗口，在数据 R - 树中执行窗口查询。此步骤之后， $\langle h,7\rangle$ 和 $\langle m,5\rangle$ 会替换列表中之前的候选点 $< a,2 > , < k,2 >$ 。点 $h$ 是 3 - 支配查询的第二个结果，并输出给用户。然后，对属于 $h$ 的支配区域但不在 ${S}_{1}$ 中其他点的支配区域内的点重复该过程（即图 13(c) 中的阴影区域）。新的天际线 ${S}_{2} = \left( {{S}_{1}-\{ h\} }\right)  \cup  \{ c,g\}$ 如图 13(d) 所示。点 $c$ 和 $g$ 各自最多可支配五个点（即 $\operatorname{num}\left( h\right)  - 2$ ），这意味着它们无法超过 $m$ 。因此，查询以 $< i,9 >  < h,7 >  < m,5 >$ 作为最终结果终止。一般来说，该算法可以被视为天际线“剥离”，因为它在具有最大支配性的点处计算局部天际线。

<!-- Media -->

<!-- figureText: 9 $e$ O ${y}_{{10} - }$ 9 8 * 7 6 - $h$ 3 2 * $m$ (b) Skyline ${S}_{l}$ after removal of $i$ 8 。 d) Skyline ${S}_{2}$ after removal of $h$ 7* 6 - 10 (a) Search region for the 2nd point 9 8 . 6 5 . 4 - -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_24.jpg?x=456&y=316&w=810&h=782&r=0"/>

Fig. 13. Example of 3-dominating query.

图 13. 3 - 支配查询示例。

<!-- Media -->

Figure 14 shows the pseudocode for $K$ -dominating queries. It is worth pointing out that the exclusive dominance region of a skyline point for $d > 2$ is not necessarily a hyperrectangle (e.g., in 3D space it may correspond to an "L-shaped" polyhedron derived by removing a cube from another cube). In this case, the constraint region can be represented as a union of hyperrect-angles (constrained BBS is still applicable). Furthermore, since we only care about the number of points in the dominance regions (as opposed to their ids), the performance of window queries can be improved by using aggregate R-trees [Papadias et al. 2001] (or any other multidimensional aggregate index).

图 14 展示了 $K$ - 支配查询的伪代码。值得指出的是，对于 $d > 2$ ，天际线点的排他支配区域不一定是超矩形（例如，在三维空间中，它可能对应于一个通过从另一个立方体中移除一个立方体而得到的“L 形”多面体）。在这种情况下，约束区域可以表示为超矩形的并集（约束 BBS 仍然适用）。此外，由于我们只关心支配区域中点的数量（而不是它们的标识），可以通过使用聚合 R - 树 [Papadias 等人，2001]（或任何其他多维聚合索引）来提高窗口查询的性能。

All existing algorithms can be employed for enumerating queries, since the only difference with respect to regular skylines is the second step (i.e., counting the number of points dominated by each skyline point). Actually, the bitmap approach can avoid scanning the actual dataset, because information about $\operatorname{num}\left( p\right)$ for each point $p$ can be obtained directly by appropriate juxtapositions of the bitmaps. $K$ -dominating queries require an effective mechanism for skyline "peeling," that is, discovery of skyline points in the exclusive dominance region of the last point removed from the skyline. Since this requires the application of a constrained query, all algorithms are applicable (as discussed in Section 4.1).

所有现有的算法都可用于枚举查询，因为与常规天际线（skyline）的唯一区别在于第二步（即，计算每个天际线点所支配的点数）。实际上，位图方法可以避免扫描实际数据集，因为每个点 $p$ 的 $\operatorname{num}\left( p\right)$ 信息可以通过位图的适当并置直接获得。 $K$ -支配查询需要一种有效的天际线“剥离”机制，即，在从天际线中移除的最后一个点的排他支配区域内发现天际线点。由于这需要应用约束查询，因此所有算法都适用（如第4.1节所述）。

<!-- Media -->

---

Algorithm $K$ -dominating_BBS (R-tree $R$ ,int $K$ )

算法 $K$ -支配_BBS（R树 $R$ ，整数 $K$ ）

1. compute skyline $S$ using BBS

1. 使用BBS算法计算天际线 $S$ 

2. for each point in $S$ compute the number of dominated points

2. 对于 $S$ 中的每个点，计算其支配的点数

3. insert the top- $K$ points of $S$ in list sorted on $\operatorname{num}\left( p\right)$

3. 将 $S$ 中按 $\operatorname{num}\left( p\right)$ 排序的前 $K$ 个点插入列表

	. counter=0

	. 计数器 = 0

	while counter $< K$

	当计数器 $< K$

			$p =$ remove first entry of list

					$p =$ 移除列表的第一个条目

			output $p$

					输出 $p$

			${S}^{\prime } =$ set of local skyline points in the dominance region of $p$

					${S}^{\prime } =$ $p$ 支配区域内的局部天际线点集

			if $\left( {\operatorname{num}\left( p\right)  - \left| {S}^{\prime }\right| }\right)  > \operatorname{num}$ (last element of list) $//{S}^{\prime }$ may contain candidate points

					如果 $\left( {\operatorname{num}\left( p\right)  - \left| {S}^{\prime }\right| }\right)  > \operatorname{num}$ （列表的最后一个元素） $//{S}^{\prime }$ 可能包含候选点

						for each point ${p}^{\prime }$ in ${S}^{\prime }$

									对于 ${S}^{\prime }$ 中的每个点 ${p}^{\prime }$ 

							find $\operatorname{num}\left( {p}^{\prime }\right) //$ perform a window query in data $R$ -tree

											查找 $\operatorname{num}\left( {p}^{\prime }\right) //$ 在数据 $R$ -树中执行窗口查询

							if $\operatorname{num}\left( {p}^{\prime }\right)  > \operatorname{num}$ (last element of list)

											如果 $\operatorname{num}\left( {p}^{\prime }\right)  > \operatorname{num}$ （列表的最后一个元素）

									update list // remove last element and insert ${p}^{\prime }$

														更新列表 // 移除最后一个元素并插入 ${p}^{\prime }$

			counter=counter+1;

					计数器 = 计数器 + 1;

	end while

	结束循环

	and $K$ -dominating_BBS

	以及 $K$ -支配的BBS算法

---

Fig. 14. $K$ -dominating_BBS algorithm.

图14. $K$ -支配的BBS算法。

<!-- figureText: ${N}_{l}$ ${N}_{2}$ ${N}_{6}$ ${N}_{7}$ $N$ : 10 7- 5 - 4 - $h$ ${N}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_25.jpg?x=643&y=885&w=416&h=354&r=0"/>

Fig. 15. Example of 2-skyband query.

图15. 2-天际带查询示例。

<!-- Media -->

### 4.6 Skyband Query

### 4.6 天际带查询

Similar to $K$ nearest-neighbor queries (that return the $K$ NNs of a point),a $K$ -skyband query reports the set of points which are dominated by at most $K$ points. Conceptually, $K$ represents the thickness of the skyline; the case $K = 0$ corresponds to a conventional skyline. Figure 15 illustrates the result of a 2- skyband query containing hotels $\{ a,b,c,\mathrm{\;g},h,i,k,m\}$ ,each dominated by at most two other hotels.

与 $K$ 最近邻查询（返回一个点的 $K$ 个最近邻）类似，$K$ -天际带查询报告最多被 $K$ 个点支配的点集。从概念上讲，$K$ 表示天际线的厚度；$K = 0$ 的情况对应于传统的天际线。图15展示了一个包含酒店 $\{ a,b,c,\mathrm{\;g},h,i,k,m\}$ 的2-天际带查询结果，每个酒店最多被另外两家酒店支配。

A naïve approach to check if a point $p$ with coordinates $\left( {{p}_{1},{p}_{2},\ldots ,{p}_{d}}\right)$ is in the skyband would be to perform a window query in the R-tree and count the number of points inside the range $\left\lbrack  {0,{p}_{1}}\right) \left\lbrack  {0,{p}_{2}}\right) \ldots \left\lbrack  {0,{p}_{d}}\right)$ . If this number is smaller than or equal to $K$ ,then $p$ belongs to the skyband. Obviously,the approach is very inefficient, since the number of window queries equals the cardinality of the dataset. On the other hand, BBS provides an efficient way for processing skyband queries. The only difference with respect to conventional skylines is that an entry is pruned only if it is dominated by more than $K$ discovered skyline points. Table V shows the contents of the heap during the processing of the query in Figure 15. Note that the skyband points are reported in ascending order of their scores, therefore maintaining the progressiveness of the results. BNL and SFS can support $K$ -skyband queries with similar modifications (i.e.,insert a point in the list if it is dominated by no more than $K$ other points). None of the other algorithms is applicable, at least in an obvious way.

一种简单的方法来检查坐标为 $\left( {{p}_{1},{p}_{2},\ldots ,{p}_{d}}\right)$ 的点 $p$ 是否在天际带中，是在R树中执行窗口查询并统计范围 $\left\lbrack  {0,{p}_{1}}\right) \left\lbrack  {0,{p}_{2}}\right) \ldots \left\lbrack  {0,{p}_{d}}\right)$ 内的点数。如果这个数量小于或等于 $K$，那么 $p$ 属于天际带。显然，这种方法效率非常低，因为窗口查询的数量等于数据集的基数。另一方面，BBS（最佳优先搜索，Best-First Search）为处理天际带查询提供了一种高效的方法。与传统天际线的唯一区别是，只有当一个条目被超过 $K$ 个已发现的天际线点支配时，才会被修剪。表V显示了在处理图15中的查询时堆的内容。请注意，天际带点按其得分升序报告，因此保持了结果的渐进性。BNL（块嵌套循环，Block Nested Loop）和SFS（排序过滤扫描，Sorted Filtering Scan）可以通过类似的修改来支持 $K$ -天际带查询（即，如果一个点最多被 $K$ 个其他点支配，则将其插入列表中）。其他算法至少在明显的方式下都不适用。

<!-- Media -->

Table V. Heap Contents of 2-Skyband Query

表V. 2-天际带查询的堆内容

<table><tr><td>Action</td><td>Heap Contents</td><td>$S$</td></tr><tr><td>Access root</td><td>$< {e}_{7},4 >  < {e}_{6},6 >$</td><td>0</td></tr><tr><td>Expand ${e}_{7}$</td><td>$< {e}_{3},5 >  < {e}_{6},6 >  < {e}_{5},8 >  < {e}_{4},{10} >$</td><td>0</td></tr><tr><td>Expand ${e}_{3}$</td><td><i, 5>< ${e}_{6}$ , 6><h, 7>< ${e}_{5}$ , 8>< ${e}_{4}$ , 10><g, 11></td><td>$\{ i\}$</td></tr><tr><td>Expand ${e}_{6}$</td><td>< $h$ , 7>< ${e}_{5}$ , 8>< ${e}_{1}$ , 9>< ${e}_{4}$ , 10>< ${e}_{2}$ , 11><g, 11></td><td>$\{ i,h\}$</td></tr><tr><td>Expand ${e}_{5}$</td><td><m, 8>< ${e}_{1},9$ >< ${e}_{4},{10}$ ><n,11><e ${}_{2},{11}$ ><g,11></td><td>$\{ i,h,m\}$</td></tr><tr><td>Expand ${e}_{1}$</td><td>< $a$ , 10>< ${e}_{4}$ , 10>< $n$ , 11>< ${e}_{2}$ , 11><g, 11><b, 12><c, 12></td><td>$\{ i,h,m,a\}$</td></tr><tr><td>Expand ${e}_{4}$</td><td><k, 10><n, 11><e ${}_{2}$ , 11><g, 11><b, 12><c, 12><l, 14></td><td>$\{ i,h,m,a,k,g,b,c\}$</td></tr></table>

<table><tbody><tr><td>操作</td><td>堆内容</td><td>$S$</td></tr><tr><td>访问根节点</td><td>$< {e}_{7},4 >  < {e}_{6},6 >$</td><td>0</td></tr><tr><td>展开 ${e}_{7}$</td><td>$< {e}_{3},5 >  < {e}_{6},6 >  < {e}_{5},8 >  < {e}_{4},{10} >$</td><td>0</td></tr><tr><td>展开 ${e}_{3}$</td><td><i, 5="">< ${e}_{6}$ , 6><h, 7="">< ${e}_{5}$ , 8>< ${e}_{4}$ , 10><g, 11=""></g,></h,></i,></td><td>$\{ i\}$</td></tr><tr><td>展开 ${e}_{6}$</td><td>< $h$ , 7>< ${e}_{5}$ , 8>< ${e}_{1}$ , 9>< ${e}_{4}$ , 10>< ${e}_{2}$ , 11><g, 11=""></g,></td><td>$\{ i,h\}$</td></tr><tr><td>展开 ${e}_{5}$</td><td><m, 8="">< ${e}_{1},9$ >< ${e}_{4},{10}$ ><n,11><e $="" {}_{2},{11}=""$=""><g,11></g,11></e></n,11></m,></td><td>$\{ i,h,m\}$</td></tr><tr><td>展开 ${e}_{1}$</td><td>< $a$ , 10>< ${e}_{4}$ , 10>< $n$ , 11>< ${e}_{2}$ , 11><g, 11=""><b, 12=""><c, 12=""></c,></b,></g,></td><td>$\{ i,h,m,a\}$</td></tr><tr><td>展开 ${e}_{4}$</td><td><k, 10=""><n, 11=""><e $="" {}_{2}=""$="" ,="" 11=""><g, 11=""><b, 12=""><c, 12=""><l, 14=""></l,></c,></b,></g,></e></n,></k,></td><td>$\{ i,h,m,a,k,g,b,c\}$</td></tr></tbody></table>

Table VI. Applicability Comparison

表六. 适用性比较

<table><tr><td/><td>D&C</td><td>BNL</td><td>SFS</td><td>Bitmap</td><td>Index</td><td>NN</td><td>BBS</td></tr><tr><td>Constrained</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Ranked</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>Yes</td></tr><tr><td>Group-by</td><td>${No}$</td><td>Yes</td><td>Yes</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>Yes</td></tr><tr><td>Dynamic</td><td>Yes</td><td>Yes</td><td>Yes</td><td>${No}$</td><td>${No}$</td><td>Yes</td><td>Yes</td></tr><tr><td>$K$ -dominating</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>$K$ -skyband</td><td>${No}$</td><td>Yes</td><td>Yes</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>Yes</td></tr></table>

<table><tbody><tr><td></td><td>分治法（Divide and Conquer）</td><td>块嵌套循环连接算法（Block Nested Loops）</td><td>顺序过滤算法（Sequential Filtering Search）</td><td>位图（Bitmap）</td><td>索引（Index）</td><td>最近邻（Nearest Neighbor）</td><td>电子布告栏系统（Bulletin Board System）</td></tr><tr><td>受限的（Constrained）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td></tr><tr><td>排序的（Ranked）</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>是（Yes）</td></tr><tr><td>分组（Group-by）</td><td>${No}$</td><td>是（Yes）</td><td>是（Yes）</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>是（Yes）</td></tr><tr><td>动态的（Dynamic）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>${No}$</td><td>${No}$</td><td>是（Yes）</td><td>是（Yes）</td></tr><tr><td>$K$ -支配（$K$ -dominating）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td></tr><tr><td>$K$ -天际带（$K$ -skyband）</td><td>${No}$</td><td>是（Yes）</td><td>是（Yes）</td><td>${No}$</td><td>${No}$</td><td>${No}$</td><td>是（Yes）</td></tr></tbody></table>

<!-- Media -->

### 4.7 Summary

### 4.7 总结

Finally, we close this section with Table VI, which summarizes the applicability of the existing algorithms for each skyline variation. A "no" means that the technique is inapplicable, inefficient (e.g., it must perform a postprocessing step on the basic algorithm), or its extension is nontrivial. Even if an algorithm (e.g., BNL) is applicable for a query type (group-by skylines), it does not necessarily imply that it is progressive (the criteria of Section 2.6 also apply to the new skyline queries). Clearly, BBS has the widest applicability since it can process all query types effectively.

最后，我们用表 VI 结束本节内容，该表总结了现有算法对每种天际线变体的适用性。“否”表示该技术不适用、效率低下（例如，它必须对基本算法执行后处理步骤），或者其扩展并非易事。即使某个算法（例如 BNL）适用于某种查询类型（分组天际线），也不一定意味着它是渐进式的（第 2.6 节的标准同样适用于新的天际线查询）。显然，BBS 的适用性最广，因为它可以有效地处理所有查询类型。

## 5. APPROXIMATE SKYLINES

## 5. 近似天际线

In this section we introduce approximate skylines, which can be used to provide immediate feedback to the users (i) without any node accesses (using a histogram on the dataset), or (ii) progressively, after the root visit of BBS. The problem for computing approximate skylines is that, even for uniform data, we cannot probabilistically estimate the shape of the skyline based only on the dataset cardinality $N$ . In fact,it is difficult to predict the actual number of skyline points (as opposed to their order of magnitude [Buchta 1989]). To illustrate this, Figures 16(a) and 16(b) show two datasets that differ in the position of a single point, but have different skyline cardinalities (1 and 4, respectively). Thus,instead of obtaining the actual shape,we target a hypothetical point $p$ such that its $x$ and $y$ coordinates are the minimum among all the expected coordinates in the dataset. We then define the approximate skyline using the two line segments enclosing the dominance region of $p$ . As shown in Figure 16(c), this approximation can be thought of as a "low-resolution" skyline.

在本节中，我们将介绍近似天际线，它可用于向用户提供即时反馈：(i) 无需进行任何节点访问（使用数据集上的直方图）；(ii) 在 BBS 访问根节点后逐步提供。计算近似天际线的问题在于，即使对于均匀数据，我们也无法仅基于数据集基数 $N$ 从概率上估计天际线的形状。实际上，很难预测天际线点的实际数量（与它们的数量级相反 [Buchta 1989]）。为了说明这一点，图 16(a) 和 16(b) 展示了两个数据集，它们仅在一个点的位置上有所不同，但天际线基数不同（分别为 1 和 4）。因此，我们不追求获得实际形状，而是以一个假设点 $p$ 为目标，使得其 $x$ 和 $y$ 坐标是数据集中所有预期坐标中的最小值。然后，我们使用包围 $p$ 支配区域的两条线段来定义近似天际线。如图 16(c) 所示，这种近似可以被视为“低分辨率”的天际线。

<!-- Media -->

<!-- figureText: 。 7- 。 (b) Actual skyline with four points all points fall in here 。 (a) Actual skyline with one point $\lambda$ (c) Approximate skyline -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_27.jpg?x=448&y=317&w=808&h=745&r=0"/>

Fig. 16. Skylines of uniform data.

图 16. 均匀数据的天际线。

<!-- Media -->

Next we compute the expected coordinates of $p$ . First,for uniform distribution,it is reasonable to assume that $p$ falls on the diagonal of the data space (because the data characteristics above and below the diagonal are similar). Assuming, for simplicity, that the data space has unit length on each axis, we denote the coordinates of $p$ as $\left( {\lambda ,\lambda }\right)$ with $0 \leq  \lambda  \leq  1$ . To derive the expected value for $\lambda$ ,we need the probability $\mathrm{P}\{ \lambda  \leq  \xi \}$ that $\lambda$ is no larger than a specific value $\xi$ . To calculate this,note that $\lambda  > \xi$ implies that all the points fall in the dominance region of $\left( {\xi ,\xi }\right)$ (i.e.,a square with length $1 - \xi$ ). For uniform data,a point has probability ${\left( 1 - \xi \right) }^{2}$ to fall in this region,and thus $\mathrm{P}\{ \lambda  > \xi \}$ (i.e.,the probability that all points are in this region) equals ${\left\lbrack  {\left( 1 - \xi \right) }^{2}\right\rbrack  }^{N}$ . So, $\mathrm{P}$ $\{ \lambda  \leq  \xi \}  = 1 - {\left( 1 - \xi \right) }^{2N}$ ,and the expected value of $\lambda$ is given by

接下来，我们计算 $p$ 的预期坐标。首先，对于均匀分布，合理假设 $p$ 落在数据空间的对角线上（因为对角线上下的数据特征相似）。为简单起见，假设数据空间在每个轴上的长度为单位长度，我们将 $p$ 的坐标表示为 $\left( {\lambda ,\lambda }\right)$，其中 $0 \leq  \lambda  \leq  1$。为了推导 $\lambda$ 的期望值，我们需要 $\lambda$ 不大于特定值 $\xi$ 的概率 $\mathrm{P}\{ \lambda  \leq  \xi \}$。为了计算这个概率，注意到 $\lambda  > \xi$ 意味着所有点都落在 $\left( {\xi ,\xi }\right)$ 的支配区域内（即一个边长为 $1 - \xi$ 的正方形）。对于均匀数据，一个点落在该区域的概率为 ${\left( 1 - \xi \right) }^{2}$，因此 $\mathrm{P}\{ \lambda  > \xi \}$（即所有点都在该区域的概率）等于 ${\left\lbrack  {\left( 1 - \xi \right) }^{2}\right\rbrack  }^{N}$。所以，$\mathrm{P}$ $\{ \lambda  \leq  \xi \}  = 1 - {\left( 1 - \xi \right) }^{2N}$，并且 $\lambda$ 的期望值由下式给出

$$
E\left( \lambda \right)  = {\int }_{0}^{1}\xi  \cdot  \frac{\mathrm{{dP}}\left( {\lambda  \leq  \xi }\right) }{\mathrm{d}\xi }\mathrm{d}\xi  = {2N}{\int }_{0}^{1}\xi  \cdot  {\left( 1 - \xi \right) }^{{2N} - 1}\mathrm{\;d}\xi . \tag{5.1}
$$

Solving this integral, we have

求解这个积分，我们得到

$$
E\left( \lambda \right)  = 1/\left( {{2N} + 1}\right) . \tag{5.2}
$$

Following similar derivations for $d$ -dimensional spaces,we obtain $E\left( \lambda \right)  =$ $1/\left( {d \cdot  N + 1}\right)$ . If the dimensions of the data space have different lengths,then the expected coordinate of the hypothetical skyline point on dimension $i$ equals $A{L}_{i}/\left( {d \cdot  N + 1}\right)$ ,where $A{L}_{i}$ is the length of the axis. Based on the above analysis, we can obtain the approximate skyline for arbitrary data distribution using a multidimensional histogram [Muralikrishna and DeWitt 1988; Acharya et al. 1999], which typically partitions the data space into a set of buckets and stores for each bucket the number (called density) of points in it. Figure 17(a) shows the extents of 6 buckets $\left( {{b}_{1},\ldots ,{b}_{6}}\right)$ and their densities,for the dataset of Figure 1 . Treating each bucket as a uniform data space, we compute the hypothetical skyline point based on its density. Then the approximate skyline of the original dataset is the skyline of all the hypothetical points, as shown in Figure 17(b). Since the number of hypothetical points is small (at most the number of buckets), the approximate skyline can be computed using existing main-memory algorithms (e.g., Kung et al. [1975]; Matousek [1991]). Due to the fact that histograms are widely used for selectivity estimation and query optimization, the extraction of approximate skylines does not incur additional requirements and does not involve I/O cost.

通过对$d$维空间进行类似推导，我们得到$E\left( \lambda \right)  =$ $1/\left( {d \cdot  N + 1}\right)$。如果数据空间的各维度长度不同，那么假设的天际点在维度$i$上的期望坐标等于$A{L}_{i}/\left( {d \cdot  N + 1}\right)$，其中$A{L}_{i}$是该轴的长度。基于上述分析，我们可以使用多维直方图[Muralikrishna和DeWitt 1988年；Acharya等人1999年]为任意数据分布获取近似天际线，该直方图通常将数据空间划分为一组桶，并为每个桶存储其中的点数（称为密度）。图17(a)展示了图1数据集的6个桶$\left( {{b}_{1},\ldots ,{b}_{6}}\right)$的范围及其密度。将每个桶视为均匀的数据空间，我们根据其密度计算假设的天际点。然后，原始数据集的近似天际线就是所有假设点的天际线，如图17(b)所示。由于假设点的数量较少（最多为桶的数量），可以使用现有的主存算法（例如，Kung等人[1975年]；Matousek [1991年]）来计算近似天际线。由于直方图广泛用于选择性估计和查询优化，提取近似天际线不会产生额外要求，也不涉及I/O成本。

<!-- Media -->

<!-- figureText: ${y}_{IO}$ ${y}_{{10} - }$ approximate skyline 。 5 4. (b) Approximate skyline 9 3 ${b}_{4}$ 7- 0 ${b}_{5}$ ${b}_{6}$ 10 (a) Bucket information -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_28.jpg?x=503&y=321&w=721&h=356&r=0"/>

Fig. 17. Obtaining the approximate skyline for nonuniform data.

图17. 为非均匀数据获取近似天际线。

<!-- Media -->

Approximate skylines using histograms can provide some information about the actual skyline in environments (e.g., data streams, on-line processing systems) where only limited statistics of the data distribution (instead of individual data) can be maintained; thus, obtaining the exact skyline is impossible. When the actual data are available, the concept of approximate skyline, combined with BBS, enables the "drill-down" exploration of the actual one. Consider, for instance, that we want to estimate the skyline (in the absence of histograms) by performing a single node access. In this case, BBS retrieves the data R-tree root and computes by Equation (5.2), for every entry MBR, a hypothetical skyline point (i) assuming that the distribution in each MBR is almost uniform (a reasonable assumption for R-trees [Theodoridis et al. 2000]), and (ii) using the average node capacity and the tree level to estimate the number of points in the MBR. The skyline of the hypothetical points constitutes a rough estimation of the actual skyline. Figure 18(a) shows the approximate skyline after visiting the root entry as well as the real skyline (dashed line). The approximation error corresponds to the difference of the ${SSR}$ s of the two skylines, that is, the area that is dominated by exactly one skyline (shaded region in Figure 18(a)).

在只能维护数据分布的有限统计信息（而非单个数据）的环境（例如，数据流、在线处理系统）中，使用直方图的近似天际线可以提供有关实际天际线的一些信息；因此，无法获取精确的天际线。当有实际数据可用时，近似天际线的概念与BBS算法相结合，可以对实际天际线进行“深入”探索。例如，假设我们想通过单次节点访问来估计天际线（在没有直方图的情况下）。在这种情况下，BBS算法检索数据R树的根节点，并通过公式(5.2)为每个条目最小边界矩形（MBR）计算一个假设的天际点：(i) 假设每个MBR内的分布几乎是均匀的（这对于R树来说是一个合理的假设[Theodoridis等人2000年]），以及(ii) 使用平均节点容量和树的层级来估计MBR中的点数。假设点的天际线构成了对实际天际线的粗略估计。图18(a)展示了访问根条目后的近似天际线以及实际天际线（虚线）。近似误差对应于两条天际线的${SSR}$值之差，即恰好被一条天际线所支配的区域（图18(a)中的阴影区域）。

<!-- Media -->

<!-- figureText: ${y}_{10}$ 9 ${N}_{6}$ 4 ${N}_{3}$ 10 (b) After retrieval of ${N}_{i}$ ${N}_{6}$ 。 (c) After retrieval of ${N}_{3}$ ${N}_{6}$ 6- ${N}_{7}$ (a) After root retrieval 9 8 . 6 5 4 * 3 -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_29.jpg?x=483&y=319&w=742&h=774&r=0"/>

Fig. 18. Approximate skylines as a function of node accesses.

图18. 近似天际线与节点访问次数的关系。

<!-- Media -->

The approximate version of BBS maintains, in addition to the actual skyline $S$ ,a set ${HS}$ consisting of points in the approximate skyline. ${HS}$ is used just for reporting the current skyline approximation and not to guide the search (the order of node visits remains the same as the original algorithm). For each intermediate entry found,if its hypothetical point $p$ is not dominated by any point in ${HS}$ ,it is added into the approximate skyline and all the points dominated by $p$ are removed from ${HS}$ . Leaf entries correspond to actual data points and are also inserted in ${HS}$ (provided that they are not dominated). When an entry is deheaped, we remove the corresponding (hypothetical or actual) point from ${HS}$ . If a data point is added to $S$ ,it is also inserted in ${HS}$ . The approximate skyline is progressively refined as more nodes are visited, for example, when the second node ${N}_{7}$ is deheaped,the hypothetical point of ${N}_{7}$ is replaced with those of its children and the new ${HS}$ is computed as shown in Figure 18(b). Similarly,the expansion of ${N}_{3}$ will lead to the approximate skyline of Figure 18(c). At the termination of approximate BBS, the estimated skyline coincides with the actual one. To show this, assume, on the contrary, that at the termination of the algorithm there still exists a hypothetical/actual point $p$ in ${HS}$ , which does not belong to $S$ . It follows that $p$ is not dominated by the actual skyline. In this case, the corresponding (intermediate or leaf) entry producing $p$ should be processed,contradicting the fact that the algorithm terminates.

除了实际天际线 $S$ 之外，近似版的 BBS（分支限界搜索，Branch and Bound Search）还维护了一个由近似天际线中的点组成的集合 ${HS}$。${HS}$ 仅用于报告当前的天际线近似值，而不用于指导搜索（节点访问顺序与原始算法保持一致）。对于找到的每个中间条目，如果其假设点 $p$ 未被 ${HS}$ 中的任何点所支配，则将其添加到近似天际线中，并从 ${HS}$ 中移除所有被 $p$ 支配的点。叶条目对应于实际数据点，并且（前提是它们未被支配）也会插入到 ${HS}$ 中。当一个条目从堆中移除时，我们会从 ${HS}$ 中移除相应的（假设或实际）点。如果一个数据点被添加到 $S$ 中，它也会被插入到 ${HS}$ 中。随着更多节点被访问，近似天际线会逐步细化，例如，当第二个节点 ${N}_{7}$ 从堆中移除时，${N}_{7}$ 的假设点会被其子女节点的假设点所取代，并如图 18(b) 所示计算新的 ${HS}$。类似地，${N}_{3}$ 的扩展将得到图 18(c) 中的近似天际线。在近似 BBS 算法终止时，估计的天际线与实际天际线重合。为了证明这一点，假设相反情况，即在算法终止时，${HS}$ 中仍然存在一个不属于 $S$ 的假设/实际点 $p$。由此可知，$p$ 未被实际天际线所支配。在这种情况下，产生 $p$ 的相应（中间或叶）条目应该被处理，这与算法终止的事实相矛盾。

Note that for computing the hypothetical point of each MBR we use Equation (5.2) because it (i) is simple and efficient (in terms of computation cost), (ii) provides a uniform treatment of approximate skylines (i.e., the same as in the case of histograms), and (iii) has high accuracy (as shown in Section 6.8). Nevertheless, we may derive an alternative approximation based on the fact that each MBR boundary contains a data point. Assuming a uniform distribution on the MBR projections and that no point is minimum on two different dimensions,this approximation leads to $d$ hypothetical points per MBR such that the expected position of each point is $1/\left( {\left( {d - 1}\right)  \cdot  N + 1}\right)$ . Figure 19(a) shows the approximate skyline in this case after the first two node visits (root and ${N}_{7}$ ). Alternatively, BBS can output an envelope enclosing the actual skyline, where the lower bound refers to the skyline obtained from the lower-left vertices of the MBRs and the upper bound refers to the skyline obtained from the upper-right vertices. Figure 19(b) illustrates the corresponding envelope (shaded region) after the first two node visits. The volume of the envelope is an upper bound for the actual approximation error, which shrinks as more nodes are accessed. The concepts of skyline approximation or envelope permit the immediate visualization of information about the skyline, enhancing the progressive behavior of BBS. In addition, approximate BBS can be easily modified for processing the query variations of Section 4 since the only difference is the maintenance of the hypothetical points in ${HS}$ for the entries encountered by the original algorithm. The computation of hypothetical points depends on the skyline variation, for example, for constrained skylines the points are computed by taking into account only the node area inside the constraint region. On the other hand, the application of these concepts to NN is not possible (at least in an obvious way), because of the duplicate elimination problem and the multiple accesses to the same node(s).

请注意，为了计算每个最小边界矩形（MBR，Minimum Bounding Rectangle）的假设点，我们使用方程 (5.2)，因为它 (i) 简单高效（就计算成本而言），(ii) 对近似天际线提供了统一的处理方式（即与直方图的情况相同），并且 (iii) 具有较高的准确性（如第 6.8 节所示）。尽管如此，我们可以基于每个 MBR 边界包含一个数据点这一事实推导出另一种近似方法。假设 MBR 投影上是均匀分布的，并且没有点在两个不同维度上都是最小值，这种近似方法会为每个 MBR 产生 $d$ 个假设点，使得每个点的预期位置为 $1/\left( {\left( {d - 1}\right)  \cdot  N + 1}\right)$。图 19(a) 展示了在访问前两个节点（根节点和 ${N}_{7}$）后这种情况下的近似天际线。或者，BBS 可以输出一个包围实际天际线的包络，其中下界指的是从 MBR 的左下角顶点得到的天际线，上界指的是从 MBR 的右上角顶点得到的天际线。图 19(b) 展示了在访问前两个节点后相应的包络（阴影区域）。包络的体积是实际近似误差的上界，随着更多节点被访问，该上界会缩小。天际线近似或包络的概念允许立即可视化有关天际线的信息，增强了 BBS 的渐进式特性。此外，近似 BBS 可以很容易地修改以处理第 4 节中的查询变体，因为唯一的区别是为原始算法遇到的条目维护 ${HS}$ 中的假设点。假设点的计算取决于天际线变体，例如，对于受限天际线，仅考虑约束区域内的节点区域来计算这些点。另一方面，由于重复消除问题和对同一节点的多次访问，这些概念无法（至少以明显的方式）应用于最近邻（NN，Nearest Neighbor）查询。

<!-- Media -->

<!-- figureText: ${y}_{\text{104}}$ ${y}_{{10} - }$ skyline envelope upper skyline ${N}_{6}$ ${N}_{3}$ ${N}_{5}$ lower skylin (b) Skyline envelope 9 approximate skyline 8 - using 2 boundary points per ${MB}$ . ${N}_{6}$ ${N}_{3}$ (a) Approximation using $d\left( { = 2}\right)$ points per MBR -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_30.jpg?x=386&y=319&w=953&h=376&r=0"/>

Fig. 19. Alternative approximations after visiting root and ${N}_{7}$ .

图 19. 访问根节点和 ${N}_{7}$ 后的替代近似方法。

<!-- Media -->

## 6. EXPERIMENTAL EVALUATION

## 6. 实验评估

In this section we verify the effectiveness of BBS by comparing it against NN which, according to the evaluation of Kossmann et al. [2002], is the most efficient existing algorithm and exhibits progressive behavior. Our implementation of NN combined laisser-faire and propagate because, as discussed in Section 2.5, it gives the best results. Specifically,only the first ${20}\%$ of the to-do list was searched for duplicates using propagate and the rest of the duplicates were handled with laisser-faire. Following the common methodology in the literature, we employed independent (uniform) and anticorrelated ${}^{5}$ datasets (generated in the same way as described in Borzsonyi et al. [2001]) with dimensionality $d$ in the range $\left\lbrack  {2,5}\right\rbrack$ and cardinality $N$ in the range $\left\lbrack  {{100}\mathrm{K},{10}\mathrm{M}}\right\rbrack$ . The length of each axis was 10,000. Datasets were indexed by R*-trees [Beckmann et al. 1990] with a page size of $4\mathrm{\;{kB}}$ ,resulting in node capacities between ${204}\left( {d = 2}\right)$ and ${94}\left( {d = 5}\right)$ . For all experiments we measured the cost in terms of node accesses since the diagrams for CPU-time are very similar (see Papadias et al. [2003]).

在本节中，我们通过将BBS（批量分支剪枝算法，Batch Branch-and-Bound Skyline）与NN（最近邻算法，Nearest Neighbor）进行比较来验证BBS的有效性。根据科斯曼（Kossmann）等人 [2002] 的评估，NN是现有最有效的算法，并且具有渐进式特性。我们实现的NN算法结合了放任（laisser-faire）和传播（propagate）策略，因为正如2.5节所讨论的，这种结合能得到最佳结果。具体而言，仅对待办列表中的前 ${20}\%$ 项使用传播策略来查找重复项，其余重复项则采用放任策略处理。遵循文献中的通用方法，我们采用了独立（均匀）和反相关 ${}^{5}$ 数据集（生成方式与博尔佐尼（Borzsonyi）等人 [2001] 所述相同），其维度 $d$ 范围为 $\left\lbrack  {2,5}\right\rbrack$，基数 $N$ 范围为 $\left\lbrack  {{100}\mathrm{K},{10}\mathrm{M}}\right\rbrack$。每个轴的长度为10,000。数据集使用R*树 [贝克曼（Beckmann）等人1990] 进行索引，页面大小为 $4\mathrm{\;{kB}}$，节点容量在 ${204}\left( {d = 2}\right)$ 到 ${94}\left( {d = 5}\right)$ 之间。在所有实验中，我们以节点访问次数来衡量成本，因为CPU时间的图表非常相似（见帕帕迪亚斯（Papadias）等人 [2003]）。

<!-- Media -->

<!-- figureText: 1e+7 [ node accesses BBS node accesses 1e+7 1e+6 1e+5 1e+4 1e+3 $1\mathrm{e} + 2$ $1\mathrm{e} + 1$ dimensionality $1\mathrm{e} + 0$ 3 4 (b) Anticorrelated 1e+6 1e+5 le+4 le+3 1e+2 le+1 dimensionality $1\mathrm{e} + 0$ 3 (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_31.jpg?x=340&y=317&w=1022&h=333&r=0"/>

Fig. 20. Node accesses vs. dimensionality $d\left( {N = 1\mathrm{M}}\right)$ .

图20. 节点访问次数与维度 $d\left( {N = 1\mathrm{M}}\right)$ 的关系。

<!-- Media -->

Sections 6.1 and 6.2 study the effects of dimensionality and cardinality for conventional skyline queries, whereas Section 6.3 compares the progressive behavior of the algorithms. Sections 6.4, 6.5, 6.6, and 6.7 evaluate constrained, group-by skyline, $K$ -dominating skyline, and $K$ -skyband queries, respectively. Finally, Section 6.8 focuses on approximate skylines. Ranked queries are not included because NN is inapplicable, while the performance of BBS is the same as in the experiments for progressive behavior. Similarly, the cost of dynamic skylines is the same as that of conventional skylines in selected dimension projections and omitted from the evaluation.

6.1节和6.2节研究了传统天际线查询中维度和基数的影响，而6.3节比较了各算法的渐进式特性。6.4节、6.5节、6.6节和6.7节分别评估了约束天际线查询、分组天际线查询、 $K$ -支配天际线查询和 $K$ -天际带查询。最后，6.8节重点讨论近似天际线。排名查询未包含在内，因为NN算法不适用，而BBS的性能与渐进式特性实验中的性能相同。同样，动态天际线的成本与选定维度投影中的传统天际线成本相同，因此在评估中省略。

### 6.1 The Effect of Dimensionality

### 6.1 维度的影响

In order to study the effect of dimensionality, we used the datasets with cardinality $N = 1\mathrm{M}$ and varied $d$ between 2 and 5 . Figure 20 shows the number of node accesses as a function of dimensionality, for independent and anticorre-lated datasets. NN could not terminate successfully for $d > 4$ in case of independent,and for $d > 3$ in case of anticorrelated,datasets due to the prohibitive size of the to-do list (to be discussed shortly). BBS clearly outperformed NN and the difference increased fast with dimensionality. The degradation of NN was caused mainly by the growth of the number of partitions (i.e., each skyline point spawned $d$ partitions),as well as the number of duplicates. The degradation of BBS was due to the growth of the skyline and the poor performance of R-trees in high dimensions. Note that these factors also influenced NN, but their effect was small compared to the inherent deficiencies of the algorithm.

为了研究维度的影响，我们使用了基数为 $N = 1\mathrm{M}$ 的数据集，并将 $d$ 在2到5之间变化。图20显示了独立和反相关数据集的节点访问次数随维度的变化情况。对于独立数据集，当 $d > 4$ 时，以及对于反相关数据集，当 $d > 3$ 时，由于待办列表的规模过大（稍后将进行讨论），NN算法无法成功终止。BBS明显优于NN，并且随着维度的增加，两者的差距迅速扩大。NN性能下降的主要原因是分区数量的增加（即每个天际线点会产生 $d$ 个分区）以及重复项数量的增加。BBS性能下降是由于天际线的增长以及R树在高维情况下的性能不佳。请注意，这些因素也会影响NN，但与该算法的固有缺陷相比，其影响较小。

---

<!-- Footnote -->

${}^{5}$ For anticorrelated distribution,the dimensions are linearly correlated such that,if ${p}_{i}$ is smaller than ${p}_{j}$ on one axis,then ${p}_{i}$ is likely to be larger on at least one other dimension (e.g.,hotels near the beach are typically more expensive). An anticorrelated dataset has fractal dimensionality close to 1 (i.e., points lie near the antidiagonal of the space).

${}^{5}$ 对于反相关分布，各维度呈线性相关，即如果 ${p}_{i}$ 在一个轴上小于 ${p}_{j}$，那么 ${p}_{i}$ 很可能在至少另一个维度上更大（例如，海滩附近的酒店通常更贵）。反相关数据集的分形维度接近1（即点位于空间的反对角线附近）。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: to-do list heap dataset 1e+5 size (Kbytes) $1\mathrm{e} + 4$ ${1e} + 3$ ${1e} + 2$ 1e+1 $1\mathrm{e} + 0$ dimensionality 1e- (b) Anticorrelated 1e+5 size (Kbytes) 1e+4 1e+3 ${1e} + 2$ 1e+1 $1\mathrm{e} + 0$ 1e-1l dimensionality 1e-2 (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_32.jpg?x=351&y=319&w=1022&h=331&r=0"/>

Fig. 21. Heap and to-do list sizes versus dimensionality $d\left( {N = 1\mathrm{M}}\right)$ .

图21. 堆和待办列表大小与维度 $d\left( {N = 1\mathrm{M}}\right)$ 的关系。

<!-- Media -->

Figure 21 shows the maximum sizes (in kbytes) of the heap, the to-do list, and the dataset,as a function of dimensionality. For $d = 2$ ,the to-do list was smaller than the heap, and both were negligible compared to the size of the dataset. For $d = 3$ ,however,the to-do list surpassed the heap (for independent data) and the dataset (for anticorrelated data). Clearly, the maximum size of the to-do list exceeded the main-memory of most existing systems for $d \geq  4$ (anticorrelated data),which explains the missing numbers about NN in the diagrams for high dimensions. Notice that Kossmann et al. [2002] reported the cost of NN for returning up to the first 500 skyline points using anticorrelated data in five dimensions. NN can return a number of skyline points (but not the complete skyline), because the to-do list does not reach its maximum size until a sufficient number of skyline points have been found (and a large number of partitions have been added). This issue is discussed further in Section 6.3, where we study the sizes of the heap and to-do lists as a function of the points returned.

图21展示了堆、待办列表和数据集的最大大小（以千字节为单位）与维度的函数关系。对于$d = 2$，待办列表比堆小，并且与数据集的大小相比，两者都可以忽略不计。然而，对于$d = 3$，待办列表的大小超过了堆（对于独立数据）和数据集（对于负相关数据）。显然，对于$d \geq  4$（负相关数据），待办列表的最大大小超过了大多数现有系统的主内存，这就解释了高维图中关于最近邻（NN）的缺失数据。请注意，科斯曼等人（Kossmann et al.）[2002]报告了在五维空间中使用负相关数据返回前500个天际点时最近邻（NN）的成本。最近邻（NN）可以返回一定数量的天际点（但不是完整的天际线），因为在找到足够数量的天际点（并添加大量分区）之前，待办列表不会达到其最大大小。这个问题将在6.3节进一步讨论，在该节中，我们将研究堆和待办列表的大小与返回点数的函数关系。

### 6.2 The Effect of Cardinality

### 6.2 基数的影响

Figure 22 shows the number of node accesses versus the cardinality for 3D datasets. Although the effect of cardinality was not as important as that of dimensionality, in all cases BBS was several orders of magnitude faster than NN. For anticorrelated data,NN did not terminate successfully for $N \geq  5\mathrm{M}$ , again due to the prohibitive size of the to-do list. Some irregularities in the diagrams (a small dataset may be more expensive than a larger one) are due to the positions of the skyline points and the order in which they were discovered. If, for instance, the first nearest neighbor is very close to the origin of the axes, both BBS and NN will prune a large part of their respective search spaces.

图22展示了三维数据集的节点访问次数与基数的关系。尽管基数的影响不如维度的影响重要，但在所有情况下，最佳优先搜索（BBS）都比最近邻（NN）快几个数量级。对于负相关数据，由于待办列表的大小过大，最近邻（NN）在$N \geq  5\mathrm{M}$时未能成功终止。图中的一些不规则情况（小数据集的处理成本可能比大数据集更高）是由于天际点的位置以及发现它们的顺序造成的。例如，如果第一个最近邻非常接近坐标轴的原点，最佳优先搜索（BBS）和最近邻（NN）都会修剪掉各自搜索空间的很大一部分。

### 6.3 Progressive Behavior

### 6.3 渐进式行为

Next we compare the speed of the algorithms in returning skyline points incrementally. Figure 23 shows the node accesses of BBS and NN as a function of the points returned for datasets with $N = 1\mathrm{M}$ and $d = 3$ (the number of points in the final skyline was 119 and 977, for independent and anticorrelated datasets, respectively). Both algorithms return the first point with the same cost (since they both apply nearest neighbor search to locate it). Then, BBS starts to gradually outperform $\mathrm{{NN}}$ and the difference increases with the number of points returned.

接下来，我们比较算法逐步返回天际点的速度。图23展示了最佳优先搜索（BBS）和最近邻（NN）的节点访问次数与返回点数的函数关系，数据集分别为$N = 1\mathrm{M}$和$d = 3$（对于独立数据集和负相关数据集，最终天际线中的点数分别为119和977）。两种算法返回第一个点的成本相同（因为它们都使用最近邻搜索来定位该点）。然后，最佳优先搜索（BBS）开始逐渐优于$\mathrm{{NN}}$，并且随着返回点数的增加，差距逐渐增大。

<!-- Media -->

<!-- figureText: NN $1\mathrm{e} + {10}$ node accesses ${1e} + 8$ 1e+6 1e+4 ${1e} + 2$ $1\mathrm{e} + 0$ 1M 2M 5M 10M (b) Anticorrelated node accesses 1e+4 $1\mathrm{e} + 3$ 1e+2 1e+1 cardinality 1e+0 100k 500k 2M (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_33.jpg?x=338&y=319&w=1024&h=323&r=0"/>

Fig. 22. Node accesses versus cardinality $N\left( {d = 3}\right)$ .

图22. 节点访问次数与基数$N\left( {d = 3}\right)$的关系。

<!-- figureText: NN BBS $1\mathrm{e} + 8$ node accesses 1e+6 le+4 $1\mathrm{e} + 2$ 200 400 600 800 977 number of reported point (b) Anticorrelated 1e+5 node accesses $1\mathrm{e} + 4$ $1\mathrm{e} + 3$ ${1e} + 2$ $1\mathrm{e} + 1$ 20 40 60 80 100 119 number of reported points (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_33.jpg?x=338&y=711&w=1025&h=333&r=0"/>

Fig. 23. Node accesses versus number of points reported $\left( {N = 1\mathrm{M},d = 3}\right)$ .

图23. 节点访问次数与报告点数$\left( {N = 1\mathrm{M},d = 3}\right)$的关系。

<!-- Media -->

To evaluate the quality of the results, Figure 24 shows the distribution of the first 50 skyline points (out of 977) returned by each algorithm for the anticor-related dataset with $N = 1\mathrm{M}$ and $d = 3$ . The initial skyline points of BBS are evenly distributed in the whole skyline, since they were discovered in the order of their mindist (which was independent of the algorithm). On the other hand, NN produced points concentrated in the middle of the data universe because the partitioned regions, created by new skyline points, were inserted at the end of the to-do list, and thus nearby points were subsequently discovered.

为了评估结果的质量，图24展示了对于具有$N = 1\mathrm{M}$和$d = 3$的负相关数据集，每种算法返回的前50个天际点（共977个）的分布情况。最佳优先搜索（BBS）的初始天际点在整个天际线中均匀分布，因为它们是按照最小距离的顺序被发现的（这与算法无关）。另一方面，最近邻（NN）产生的点集中在数据空间的中间，因为由新天际点创建的分区区域被插入到待办列表的末尾，因此随后会发现附近的点。

Figure 25 compares the sizes of the heap and to-do lists as a function of the points returned. The heap reaches its maximum size at the beginning of BBS, whereas the to-do list reaches it toward the end of NN. This happens because before BBS discovered the first skyline point, it inserted all the entries of the visited nodes in the heap (since no entry can be pruned by existing skyline points). The more skyline points were discovered, the more heap entries were pruned, until the heap eventually became empty. On the other hand, the to-do list size is dominated by empty queries, which occurred toward the late phases of NN when the space subdivisions became too small to contain any points. Thus, NN could still be used to return a number of skyline points (but not the complete skyline) even for relatively high dimensionality.

图25比较了堆和待办列表的大小与返回点数的函数关系。在最佳优先搜索（BBS）开始时，堆达到其最大大小，而在最近邻（NN）接近结束时，待办列表达到其最大大小。这是因为在最佳优先搜索（BBS）发现第一个天际点之前，它将访问节点的所有条目插入到堆中（因为现有天际点无法修剪任何条目）。发现的天际点越多，堆中的条目被修剪得越多，直到堆最终为空。另一方面，待办列表的大小主要由空查询决定，这些空查询在最近邻（NN）的后期阶段出现，此时空间细分变得太小，无法包含任何点。因此，即使对于相对较高的维度，最近邻（NN）仍然可以用于返回一定数量的天际点（但不是完整的天际线）。

<!-- Media -->

<!-- figureText: 10000 10000 8000 6000 4000 2000 10000 8000 400C 2000 2000 6000 10000 (b) NN 8000 6000 4000 10000 2000 6000 4000 2000 2000 4000 8000 10000 (a) BBS -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_34.jpg?x=446&y=315&w=835&h=400&r=0"/>

Fig. 24. Distribution of the first 50 skyline points (anticorrelated, $N = 1\mathrm{M},d = 3$ ).

图24. 前50个天际点的分布（负相关，$N = 1\mathrm{M},d = 3$）。

<!-- figureText: to-do list heap - size (Kbytes) $1\mathrm{e} + 2$ ${1e} + 1$ $1\mathrm{e} + 0$ 200 400 600 800 977 number of reported points (b) Anticorrelated size (Kbytes) 0 20 40 60 80 89 number of reported points (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_34.jpg?x=355&y=792&w=1018&h=338&r=0"/>

Fig. 25. Sizes of the heap and to-do list versus number of points reported $\left( {N = 1\mathrm{M},d = 3}\right)$ .

图25. 堆和待办列表的大小与报告点数$\left( {N = 1\mathrm{M},d = 3}\right)$的关系。

<!-- Media -->

### 6.4 Constrained Skyline

### 6.4 约束天际线

Having confirmed the efficiency of BBS for conventional skyline retrieval, we present a comparison between BBS and NN on constrained skylines. Figure 26 shows the node accesses of BBS and NN as a function of the constraint region volume $\left( {N = 1\mathrm{M},d = 3}\right)$ ,which is measured as a percentage of the volume of the data universe. The locations of constraint regions were uniformly generated and the results were computed by taking the average of 50 queries. Again BBS was several orders of magnitude faster than NN.

在确认了BBS（批量分支剪枝搜索，Batch Branch-and-Bound Search）在传统天际线检索中的效率后，我们对BBS和NN（最近邻，Nearest Neighbor）在约束天际线方面进行了比较。图26展示了BBS和NN的节点访问次数与约束区域体积 $\left( {N = 1\mathrm{M},d = 3}\right)$ 的关系，约束区域体积以数据空间总体积的百分比来衡量。约束区域的位置是均匀生成的，结果是通过对50次查询取平均值计算得出的。同样，BBS比NN快了几个数量级。

The counterintuitive observation here is that constraint regions covering more than $8\%$ of the data space are usually more expensive than regular skylines. Figure 27(a) verifies the observation by illustrating the node accesses of BBS on independent data, when the volume of the constraint region ranges between 98% and 100% (i.e., regular skyline). Even a range very close to 100% is much more expensive than a conventional skyline. Similar results hold for NN (see Figure 27(b)) and anticorrelated data.

这里有一个违反直觉的现象，即覆盖超过 $8\%$ 数据空间的约束区域通常比常规天际线的计算成本更高。图27(a)通过展示在独立数据上BBS的节点访问次数，验证了这一现象，此时约束区域的体积在98%到100%之间（即常规天际线）。即使是非常接近100%的范围，其计算成本也比传统天际线高得多。对于NN（见图27(b)）和负相关数据，也有类似的结果。

To explain this,consider Figure 28(a),which shows a skyline $S$ in a constraint region. The nodes that must be visited intersect the constrained skyline search region (shaded area) defined by $S$ and the constraint region. In this example, all four nodes ${e}_{1},{e}_{2},{e}_{3},{e}_{4}$ may contain skyline points and should be accessed. On the other hand,if $S$ were a conventional skyline,as in Figure 28(b),nodes ${e}_{2},{e}_{3}$ ,and ${e}_{4}$ could not exist because they should contain at least a point that dominates $S$ . In general,the only data points of the conventional ${SSR}$ (shaded area in Figure 28(b)) lie on the skyline, implying that, for any node MBR, at most one of its vertices can be inside the SSR. For constrained skylines there is no such restriction and the number of nodes intersecting the constrained ${SSR}$ can be arbitrarily large.

为了解释这一点，请参考图28(a)，它展示了约束区域内的一条天际线 $S$ 。必须访问的节点与由 $S$ 和约束区域定义的约束天际线搜索区域（阴影区域）相交。在这个例子中，所有四个节点 ${e}_{1},{e}_{2},{e}_{3},{e}_{4}$ 都可能包含天际线点，因此应该被访问。另一方面，如果 $S$ 是传统天际线，如图28(b)所示，节点 ${e}_{2},{e}_{3}$ 和 ${e}_{4}$ 不可能存在，因为它们应该至少包含一个支配 $S$ 的点。一般来说，传统 ${SSR}$ 的数据点（图28(b)中的阴影区域）仅位于天际线上，这意味着对于任何节点的最小边界矩形（MBR），其顶点最多只有一个可以在天际线搜索区域（SSR）内。对于约束天际线，没有这样的限制，与约束 ${SSR}$ 相交的节点数量可以任意大。

<!-- Media -->

<!-- figureText: $1\mathrm{e} + 5$ node accesses BBS 1e+7 -node accesses 1e+6 1e+5 1e+4 $1\mathrm{e} + 3$ ${1e} + 2$ $1\mathrm{e} + 1$ constraint region (%) $1\mathrm{e} + 0$ 16 32 64 (b) Anticorrelated 1e+4 1e+3 $1\mathrm{e} + 2$ ${1e} + 1$ constraint region (%) 1e+0 8 16 64 (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_35.jpg?x=339&y=317&w=1025&h=333&r=0"/>

Fig. 26. Node accesses versus volume of constraint region $\left( {N = 1\mathrm{M},d = 3}\right)$ .

图26. 节点访问次数与约束区域体积 $\left( {N = 1\mathrm{M},d = 3}\right)$ 的关系。

<!-- figureText: 1400 C node accesses 70000 - node accesses 60000 50000 40000 30000 20000 10000 constraint region (%) 98 98.5 99 99.5 100 (b) NN 1200 1000 800 600 400 200 constraint region (%) 98 98.5 99 99.5 100 (a) BBS -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_35.jpg?x=369&y=721&w=966&h=338&r=0"/>

Fig. 27. Node accesses versus volume of constraint region ${98} - {100}\%$ (independent, $N = 1\mathrm{M},d = 3$ ).

图27. 节点访问次数与约束区域体积 ${98} - {100}\%$ 的关系（独立数据， $N = 1\mathrm{M},d = 3$ ）。

<!-- Media -->

It is important to note that the constrained queries issued when a skyline point is removed during incremental maintenance (see Section 3.4) are always cheaper than computing the entire skyline from scratch. Consider, for instance, that the partial skyline of Figure 28(a) is computed for the exclusive dominance area of a deleted skyline point $p$ on the lower-left corner of the constraint region. In this case nodes such as ${e}_{2},{e}_{3},{e}_{4}$ cannot exist because otherwise they would have to contain skyline points, contradicting the fact that the constraint region corresponds to the exclusive dominance area of $p$ .

需要注意的是，在增量维护过程中移除一个天际线点时发出的约束查询（见3.4节），其成本总是比从头计算整个天际线要低。例如，考虑图28(a)中的部分天际线是为约束区域左下角一个已删除的天际线点 $p$ 的独占支配区域计算的。在这种情况下，像 ${e}_{2},{e}_{3},{e}_{4}$ 这样的节点不可能存在，因为否则它们就必须包含天际线点，这与约束区域对应于 $p$ 的独占支配区域这一事实相矛盾。

### 6.5 Group-By Skyline

### 6.5 分组天际线

Next we consider group-by skyline retrieval, including only BBS because, as discussed in Section 4, NN is inapplicable in this case. Toward this, we generate datasets (with cardinality $1\mathrm{M}$ ) in a $3\mathrm{D}$ space that involves two numerical dimensions and one categorical axis. In particular,the number ${c}_{\text{num }}$ of categories is a parameter ranging from 2 to ${64}\left( {c}_{\text{num }}\right.$ is also the number of 2D skylines returned by a group-by skyline query). Every data point has equal probability to fall in each category, and, for all the points in the same category, their distribution (on the two numerical axes) is either independent or anticorrelated. Figure 29 demonstrates the number of node accesses as a function of ${c}_{num}$ . The cost of BBS increases with ${c}_{\text{num }}$ because the total number of skyline points (in all 2D skylines) and the probability that a node may contain qualifying points in some category (and therefore it should be expanded) is proportional to the size of the categorical domain.

接下来，我们考虑分组天际线检索，只使用BBS，因为正如第4节所讨论的，NN在这种情况下不适用。为此，我们在一个 $3\mathrm{D}$ 空间中生成数据集（基数为 $1\mathrm{M}$ ），该空间包含两个数值维度和一个分类轴。具体来说，类别数量 ${c}_{\text{num }}$ 是一个从2到 ${64}\left( {c}_{\text{num }}\right.$ 的参数（ ${64}\left( {c}_{\text{num }}\right.$ 也是分组天际线查询返回的二维天际线的数量）。每个数据点落入每个类别的概率相等，并且对于同一类别中的所有点，它们在两个数值轴上的分布要么是独立的，要么是负相关的。图29展示了节点访问次数与 ${c}_{num}$ 的关系。BBS的成本随着 ${c}_{\text{num }}$ 的增加而增加，因为天际线点的总数（在所有二维天际线中）以及一个节点可能包含某个类别中合格点的概率（因此需要展开该节点）与分类域的大小成正比。

<!-- Media -->

<!-- figureText: constraint region Skyline S annot exist SSR (b) Conventional skyline Skyline S ${p}_{ - }\left( \text{deleted}\right)$ constrained SSR (a) Constrained skyline -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_36.jpg?x=438&y=321&w=852&h=298&r=0"/>

Fig. 28. Nodes potentially intersecting the ${SSR}$ .

图28. 可能与 ${SSR}$ 相交的节点。

<!-- figureText: 300 number of node accesses 1000 number of node accesses 800 600 400 200 0 8 16 32 number of groups (b) Anticorrelated 250 200 150 100 50 0 2 8 16 32 number of groups (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_36.jpg?x=420&y=682&w=884&h=288&r=0"/>

Fig. 29. BBS node accesses versus cardinality of categorical axis ${c}_{\text{num }}\left( {N = 1\mathrm{M},d = 3}\right)$ .

图29. BBS节点访问次数与分类轴基数 ${c}_{\text{num }}\left( {N = 1\mathrm{M},d = 3}\right)$ 的关系。

<!-- Media -->

## ${6.6K}$ -Dominating Skyline

## ${6.6K}$ -支配天际线

This section measures the performance of NN and BBS on $K$ -dominating queries. Recall that each $K$ -dominating query involves an enumerating query (i.e., a file scan), which retrieves the number of points dominated by each skyline point. The $K$ skyline points with the largest counts are found and the top-1 is immediately reported. Whenever an object is reported, a constrained skyline is executed to find potential candidates in its exclusive dominance region (see Figure 13). For each such candidate, the number of dominated points is retrieved using a window query on the data R-tree. After this process, the object with the largest count is reported (i.e., the second best object), another constrained query is performed, and so on. Therefore, the total number of constrained queries is $K - 1$ ,and each such query may trigger multiple window queries. Figure 30 demonstrates the cost of BBS and NN as a function of $K$ . The overhead of the enumerating and (multiple) window queries dominates the total cost, and consequently BBS and NN have a very similar performance.

本节衡量了最近邻（NN）和最佳优先搜索（BBS）在 $K$ -支配查询上的性能。回顾一下，每个 $K$ -支配查询都涉及一个枚举查询（即文件扫描），该查询会检索每个天际点所支配的点数。找到计数最大的 $K$ 个天际点，并立即报告排名第一的点。每当报告一个对象时，都会执行一个受限天际线操作，以在其独占支配区域中找到潜在候选对象（见图13）。对于每个这样的候选对象，使用数据R树的窗口查询来检索其支配的点数。此过程完成后，报告计数最大的对象（即次优对象），然后执行另一个受限查询，依此类推。因此，受限查询的总数为 $K - 1$ ，并且每个这样的查询可能会触发多个窗口查询。图30展示了BBS和NN的成本与 $K$ 的函数关系。枚举查询和（多个）窗口查询的开销主导了总成本，因此BBS和NN的性能非常相似。

Interestingly, the overhead of the anticorrelated data is lower (than the independent distribution) because each skyline point dominates fewer points (therefore, the number of window queries is smaller). The high cost of $K$ -dominating queries (compared to other skyline variations) is due to the complexity of the problem itself (and not the proposed algorithm). In particular, a $K$ -dominating query is similar to a semijoin and could be processed accordingly. For instance a nested-loops algorithm would (i) count, for each data point, the number of dominated points by scanning the entire database, (ii) sort all the points in descending order of the counts,and (iii) report the $K$ points with the highest counts. Since in our case the database occupies more than $6\mathrm{K}$ nodes,this algorithm would need to access ${36}\mathrm{E} + 6$ nodes (for any $K$ ),which is significantly higher than the costs in Figure 30 (especially for low $K$ ).

有趣的是，反相关数据的开销较低（比独立分布情况），因为每个天际点支配的点数较少（因此，窗口查询的数量也较少）。 $K$ -支配查询的高成本（与其他天际线变体相比）是由于问题本身的复杂性（而不是所提出的算法）。特别是， $K$ -支配查询类似于半连接，可以相应地进行处理。例如，嵌套循环算法会（i）通过扫描整个数据库，为每个数据点计算其支配的点数，（ii）按计数的降序对所有点进行排序，（iii）报告计数最高的 $K$ 个点。由于在我们的案例中，数据库占用了超过 $6\mathrm{K}$ 个节点，因此该算法需要访问 ${36}\mathrm{E} + 6$ 个节点（对于任何 $K$ ），这明显高于图30中的成本（特别是对于较小的 $K$ ）。

<!-- Media -->

<!-- figureText: 8E+6 number of node accesses ${3.5}\mathrm{E} + 6$ number of node accesses 3E+6 ${2.5}\mathrm{E} + 6$ 2E+6 1.5E+6 1E+6 ${0.5}\mathrm{E} + 6$ 16 32 64 128 $K$ -dominating (b) Anticorrelated 6E+6 4E+6 2E+6 0 16 32 64 128 $K$ -dominating (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_37.jpg?x=368&y=315&w=963&h=346&r=0"/>

Fig. 30. NN and BBS node accesses versus number of objects to be reported for $K$ -dominating queries $\left( {N = 1\mathrm{M},d = 2}\right)$ .

图30. 对于 $K$ -支配查询 $\left( {N = 1\mathrm{M},d = 2}\right)$ ，NN和BBS节点访问次数与要报告的对象数量的关系。

<!-- figureText: 0 - number of node accesses ${900}_{ T }$ number of node accesses 700 600 500 400 300 200 100 0 0 1 2 3 4 $K$ -skyband (b) Anticorrelated 300 250 200 150 100 50 0 6 7 8 9 K-skyband (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_37.jpg?x=365&y=766&w=973&h=319&r=0"/>

Fig. 31. BBS node accesses versus "thickness" of the skyline for $K$ -skyband queries $(N = 1\mathrm{M}$ , $d = 3$ ).

图31. 对于 $K$ -天际带查询 $(N = 1\mathrm{M}$ ， $d = 3$ ），BBS节点访问次数与天际线“厚度”的关系。

<!-- Media -->

### 6.7 $K$ -Skyband

### 6.7 $K$ -天际带

Next,we evaluate the performance of BBS on $K$ -skyband queries (NN is inapplicable). Figure 31 shows the node accesses as a function of $K$ ranging from 0 (conventional skyline) to 9 . As expected,the performance degrades as $K$ increases because a node can be pruned only if it is dominated by more than $K$ discovered skyline points,which becomes more difficult for higher $K$ . Furthermore, the number of skyband points is significantly larger for anticorrelated data,for example,for $K = 9$ ,the number is 788 (6778) in the independent (anticorrelated) case, which explains the higher costs in Figure 31(b).

接下来，我们评估BBS在 $K$ -天际带查询上的性能（NN不适用）。图31展示了节点访问次数与 $K$ 的函数关系， $K$ 的范围从0（传统天际线）到9。正如预期的那样，随着 $K$ 的增加，性能会下降，因为只有当一个节点被超过 $K$ 个已发现的天际点支配时，该节点才能被剪枝，而对于较大的 $K$ ，这变得更加困难。此外，反相关数据的天际带点数明显更多，例如，对于 $K = 9$ ，在独立（反相关）情况下，点数为788（6778），这解释了图31（b）中较高的成本。

<!-- Media -->

<!-- figureText: 0.009% approximation error 20% approximation error 18% 16% 14% 12% 10% 8% 6% 2% 0% 100200 400 600 800 1000 number of buckets (b) Anticorrelated 0.008% 0.007% 0.006% 0.005% 0.004% 0.003% 0.002% 0.001% 0.000% 100 200 400 600 800 1000 number of buckets (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_38.jpg?x=382&y=317&w=957&h=385&r=0"/>

Fig. 32. Approximation error versus number of minskew buckets $\left( {N = 1\mathrm{M},d = 3}\right)$ .

图32. 近似误差与最小偏斜桶数量 $\left( {N = 1\mathrm{M},d = 3}\right)$ 的关系。

<!-- Media -->

### 6.8 Approximate Skylines

### 6.8 近似天际线

This section evaluates the quality of the approximate skyline using a hypothetical point per bucket or visited node (as shown in the examples of Figures 17 and 18, respectively). Given an estimated and an actual skyline, the approximation error corresponds to their SSR difference (see Section 5). In order to measure this error, we used a numerical approach: (i) we first generated a large number $\alpha$ of points $\left( {\alpha  = {10}^{4}}\right)$ uniformly distributed in the data space,and (ii) counted the number $\beta$ of points that are dominated by exactly one skyline. The error equals $\beta /\alpha$ ,which approximates the volume of the ${SSR}$ difference divided by the volume of the entire data space. We did not use a relative error (e.g., volume of the ${SSR}$ difference divided by the volume of the actual ${SSR}$ ) because such a definition is sensitive to the position of the actual skyline (i.e., a skyline near the origin of the axes would lead to higher error even if the ${SSR}$ difference remains constant).

本节使用每个桶或访问节点的假设点（分别如图17和图18的示例所示）评估近似天际线的质量。给定一个估计天际线和一个实际天际线，近似误差对应于它们的SSR差异（见第5节）。为了测量这个误差，我们采用了一种数值方法：(i) 我们首先在数据空间中均匀生成大量的点 $\alpha$ $\left( {\alpha  = {10}^{4}}\right)$；(ii) 统计恰好被一个天际线所支配的点的数量 $\beta$。误差等于 $\beta /\alpha$，它近似于 ${SSR}$ 差异的体积除以整个数据空间的体积。我们没有使用相对误差（例如，${SSR}$ 差异的体积除以实际 ${SSR}$ 的体积），因为这样的定义对实际天际线的位置很敏感（即，即使 ${SSR}$ 差异保持不变，靠近坐标轴原点的天际线也会导致更高的误差）。

In the first experiment, we built a minskew [Acharya et al. 1999] histogram on the 3D datasets by varying the number of buckets from 100 to 1000 , resulting in main-memory consumption in the range of $3\mathrm{\;K}$ bytes(100)to ${30}\mathrm{\;K}$ bytes $({1000}$ buckets). Figure 32 illustrates the error as a function of the bucket number. For independent distribution, the error is very small (less than 0.01%) even with the smallest number of buckets because the rough "shape" of the skyline for a uniform dataset can be accurately predicted using Equation (5.2). On the other hand, anticorrelated data were skewed and required a large number of buckets for achieving high accuracy.

在第一个实验中，我们通过将桶的数量从100变化到1000，在三维数据集上构建了一个最小偏斜（minskew，[Acharya等人，1999]）直方图，导致主内存消耗范围从 $3\mathrm{\;K}$ 字节（100个桶）到 ${30}\mathrm{\;K}$ 字节（$({1000}$ 个桶）。图32展示了误差随桶数量的变化情况。对于独立分布，即使桶的数量最少，误差也非常小（小于0.01%），因为使用方程(5.2)可以准确预测均匀数据集天际线的大致“形状”。另一方面，反相关数据是有偏的，需要大量的桶才能达到高精度。

Figure 33 evaluates the quality of the approximation as a function of node accesses (without using a histogram). As discussed in Section 5, the first rough estimate of the skyline is produced when BBS visits the root entry and then the approximation is refined as more nodes are accessed. For independent data, extremely accurate approximation (with error 0.01%) can be obtained immediately after retrieving the root, a phenomenon similar to that in Figure 32(a). For anti-correlated data, the error is initially large (around 15% after the root visit), but decreases considerably with only a few additional node accesses. Particularly, the error is less than 3% after visiting 30 nodes, and close to zero with around 100 accesses (i.e., the estimated skyline is almost identical to the actual

图33评估了近似质量随节点访问次数的变化情况（不使用直方图）。如第5节所述，当BBS访问根条目时会产生天际线的第一个粗略估计，然后随着访问更多节点，近似结果会得到细化。对于独立数据，在检索根节点后立即可以获得极其准确的近似结果（误差为0.01%），这一现象与图32(a)类似。对于反相关数据，误差最初很大（访问根节点后约为15%），但只需额外访问几个节点，误差就会显著降低。特别是，访问30个节点后误差小于3%，访问约100个节点时误差接近零（即，估计的天际线几乎与实际天际线相同

<!-- Media -->

<!-- figureText: 0.007% approximation error a approximation error 14% 12% 10% 8% 6% 380 accesses required 4% for actual skyline 2% 0% 1 100 200 300 400 number of node accesses (b) Anticorrelated 0.006% 0.005% 0.004% 0.003% 95 accesses required 0.002% for actual skyline 0.001% 0.000% 20 40 60 80 100 number of node accesses (a) Independent -->

<img src="https://cdn.noedgeai.com/0195c90b-ff08-7bf7-b456-a316eac8fc7a_39.jpg?x=365&y=315&w=971&h=385&r=0"/>

Fig. 33. BBS approximation error versus number of node accesses $\left( {N = 1\mathrm{M},d = 3}\right)$ .

图33. BBS近似误差与节点访问次数 $\left( {N = 1\mathrm{M},d = 3}\right)$ 的关系。

<!-- Media -->

one with about ${25}\%$ of the node accesses required for the discovery of the actual skyline).

即使用发现实际天际线所需节点访问次数的约 ${25}\%$ 就能达到这种效果）。

## 7. CONCLUSION

## 7. 结论

The importance of skyline computation in database systems increases with the number of emerging applications requiring efficient processing of preference queries and the amount of available data. Consider, for instance, a bank information system monitoring the attribute values of stock records and answering queries from multiple users. Assuming that the user scoring functions are monotonic, the top-1 result of all queries is always a part of the skyline. Similarly,the top- $K$ result is always a part of the $K$ -skyband. Thus,the system could maintain only the skyline (or $K$ -skyband) and avoid searching a potentially very large number of records. However, all existing database algorithms for skyline computation have several deficiencies, which severely limit their applicability. BNL and D&C are not progressive. Bitmap is applicable only for datasets with small attribute domains and cannot efficiently handle updates. Index cannot be used for skyline queries on a subset of the dimensions. SFS, like all above algorithms, does not support user-defined preferences. Although NN was presented as a solution to these problems, it introduces new ones, namely, poor performance and prohibitive space requirements for more than three dimensions. This article proposes BBS, a novel algorithm that overcomes all these shortcomings since (i) it is efficient for both progressive and complete skyline computation, independently of the data characteristics (dimensionality, distribution), (ii) it can easily handle user preferences and process numerous alternative skyline queries (e.g., ranked, constrained, approximate skylines), (iii) it does not require any precomputation (besides building the R-tree), (iv) it can be used for any subset of the dimensions, and (v) it has limited main-memory requirements.

随着越来越多的应用需要高效处理偏好查询，以及可用数据量的增加，数据库系统中天际线计算的重要性也日益凸显。例如，考虑一个银行信息系统，它监控股票记录的属性值并回答多个用户的查询。假设用户评分函数是单调的，所有查询的top - 1结果始终是天际线的一部分。类似地，top - $K$ 结果始终是 $K$ - 天际带的一部分。因此，系统可以只维护天际线（或 $K$ - 天际带），避免搜索可能非常多的记录。然而，所有现有的天际线计算数据库算法都存在一些缺陷，这严重限制了它们的适用性。BNL和分治法（D&C）不是渐进式的。位图法仅适用于属性域较小的数据集，并且不能有效地处理更新。索引不能用于对部分维度进行天际线查询。与上述所有算法一样，顺序过滤扫描法（SFS）不支持用户定义的偏好。虽然最近邻法（NN）被提出作为解决这些问题的方案，但它又引入了新的问题，即对于三维以上的数据，性能较差且空间需求过高。本文提出了一种新颖的算法——最佳优先搜索法（BBS），它克服了所有这些缺点，因为(i) 无论数据特征（维度、分布）如何，它对于渐进式和完整的天际线计算都很高效；(ii) 它可以轻松处理用户偏好并处理众多替代天际线查询（例如，排序、约束、近似天际线）；(iii) 除了构建R树之外，它不需要任何预计算；(iv) 它可以用于任何维度子集；(v) 它对主内存的需求有限。

Although in this implementation of BBS we used R-trees in order to perform a direct comparison with NN, the same concepts are applicable to any data-partitioning access method. In the future, we plan to investigate alternatives (e.g., X-trees [Berchtold et al. 1996], and A-trees [Sakurai et al. 2000]) for high-dimensional spaces, where R-trees are inefficient). Another possible solution for high dimensionality would include (i) converting the data points to subspaces with lower dimensionalities, (ii) computing the skyline in each subspace, and (iii) merging the partial skylines. Finally, a topic worth studying concerns skyline retrieval in other application domains. For instance, Balke et al. [2004] studied skyline computation for Web information systems considering that the records are partitioned in several lists, each residing at a distributed server. The tuples in every list are sorted in ascending order of a scoring function, which is monotonic on all attributes. Their processing method uses the main concept of the threshold algorithm [Fagin et al. 2001] to compute the entire skyline by reading the minimum number of records in each list. Another interesting direction concerns skylines in temporal databases [Salzberg and Tsotras 1999] that retain historical information. In this case, a query could ask for the most interesting objects at a past timestamp or interval.

尽管在本次BBS（批量边界搜索）实现中，我们使用了R树（R-tree）以便与最近邻搜索（NN）进行直接比较，但相同的概念适用于任何数据分区访问方法。未来，我们计划研究适用于高维空间的替代方案（例如，X树[Berchtold等人，1996年]和A树[Sakurai等人，2000年]，在高维空间中R树效率较低）。针对高维度问题，另一种可能的解决方案包括：（i）将数据点转换为低维子空间；（ii）计算每个子空间中的天际线；（iii）合并部分天际线。最后，一个值得研究的主题是其他应用领域中的天际线检索。例如，Balke等人[2004年]研究了Web信息系统的天际线计算，他们假设记录被划分到多个列表中，每个列表位于一个分布式服务器上。每个列表中的元组按评分函数的升序排列，该评分函数在所有属性上都是单调的。他们的处理方法采用了阈值算法[Fagin等人，2001年]的主要概念，通过读取每个列表中最少数量的记录来计算整个天际线。另一个有趣的研究方向涉及保留历史信息的时态数据库中的天际线[Salzberg和Tsotras，1999年]。在这种情况下，查询可能会要求获取过去某个时间戳或时间间隔内最有趣的对象。

## REFERENCES

## 参考文献

Acharya, S., Poosala, V., and Ramaswamy, S. 1999. Selectivity estimation in spatial databases. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; Philadelphia, PA, June 1-3). 13-24.

BALKE, W., GUNZER, U., AND ZHENG, J. 2004. Efficient distributed skylining for Web information systems. In Proceedings of the International Conference on Extending Database Technology (EDBT; Heraklio, Greece, Mar. 14-18). 256-273.

Beckhann, N., Kriegel, H., Schneider, R., AND Seeger, B. 1990. The R*-tree: An efficient and robust access method for points and rectangles. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; Atlantic City, NJ, May 23-25). 322-331.

BERCHTOLD, S., KEIM, D., AND KRIEGEL, H. 1996. The X-tree: An index structure for high-dimensional data. In Proceedings of the Very Large Data Bases Conference (VLDB; Mumbai, India, Sep. 3-6). 28-39.

BöHM, C. AND KRIEGEL, H. 2001. Determining the convex hull in large multidimensional databases. In Proceedings of the International Conference on Data Warehousing and Knowledge Discovery (DaWaK; Munich, Germany, Sep. 5-7). 294-306.

Borzsonyl, S., Kosshann, D., and Stocker, K. 2001. The skyline operator. In Proceedings of the IEEE International Conference on Data Engineering (ICDE; Heidelberg, Germany, Apr. 2-6). ${421} - {430}$ .

BUCHTA, C. 1989. On the average number of maxima in a set of vectors. Inform. Process. Lett., ${33},2,{63} - {65}$ .

Chang, Y., Bergman, L., Castelli, V., Li, C., Lo, M., and Smith, J. 2000. The Onion technique: Indexing for linear optimization queries. In Proceedings of the ACM Conference on the Management of data (SIGMOD; Dallas, TX, May 16-18). 391-402.

Chomicki, J., Gobfrey, P., Gryz, J., AND Liang, D. 2003. Skyline with pre-sorting. In Proceedings of the IEEE International Conference on Data Engineering (ICDE; Bangalore, India, Mar. 5-8). 717-719.

FAGIN, R., LOTEM, A., AND NAOR, M. 2001. Optimal aggregation algorithms for middleware. In Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS; Santa Barbara, CA, May 21-23). 102-113.

FERHATOSMANOGLU, H., STANOI, I., AGRAWAL, D., AND ABBADI, A. 2001. Constrained nearest neighbor queries. In Proceedings of the International Symposium on Spatial and Temporal Databases (SSTD; Redondo Beach, CA, July 12-15). 257-278.

GUTTMAN, A. 1984. R-trees: A dynamic index structure for spatial searching. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; Boston, MA, June 18-21). 47- 57.

Hellerstein, J., Anvur, R., Chou, A., Hidber, C., Olston, C., Rahan, V., Roth, T., Ano Haas, P. 1999. Interactive data analysis: The control project. IEEE Comput. 32, 8, 51- 59.

Henrich, A. 1994. A distance scan algorithm for spatial access structures. In Proceedings of the ACM Workshop on Geographic Information Systems (ACM GIS; Gaithersburg, MD, Dec.). 136-143.

HJALTASON, G. AND SAMET, H. 1999. Distance browsing in spatial databases. ACM Trans. Database Syst. 24, 2, 265-318.

HRISTIDIS, V., KOUDAS, N., AND PAPAKONSTANTINOU, Y. 2001. PREFER: A system for the efficient execution of multi-parametric ranked queries. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; May 21-24). 259-270.

Kosshann, D., Rahssk, F., and Rost, S. 2002. Shooting stars in the sky: An online algorithm for skyline queries. In Proceedings of the Very Large Data Bases Conference (VLDB; Hong Kong, China, Aug. 20-23). 275-286.

Kung, H., Luccio, F., and Preparatata, F. 1975. On finding the maxima of a set of vectors. J. Assoc. Comput. Mach., 22, 4, 469-476.

Matousek,J. 1991. Computing dominances in ${\mathrm{E}}^{n}$ . Inform. Process. Lett. 38,5,277-278.

Mclain, D. 1974. Drawing contours from arbitrary data points. Comput. J. 17, 4, 318-324.

Muralikrishna, M. and Dewitt, D. 1988. Equi-depth histograms for estimating selectivity factors for multi-dimensional queries. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; Chicago, IL, June 1-3). 28-36.

Natsev, A., Chang, Y., Smitth, J., Li., C., and Virter. J. 2001. Supporting incremental join queries on ranked inputs. In Proceedings of the Very Large Data Bases Conference (VLDB; Rome, Italy, Sep. 11-14). 281-290.

Papaniss, D., Tao, Y., Fu, G., AND Seeger, B. 2003. An optimal and progressive algorithm for skyline queries. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; San Diego, CA, June 9-12). 443-454.

Papadias, D., Kalnis, P., Zhang, J., AND Tao, Y. 2001. Efficient OLAP operations in spatial data warehouses. In Proceedings of International Symposium on Spatial and Temporal Databases (SSTD; Redondo Beach, CA, July 12-15). 443-459.

PREPARATA, F. AND SHAMOS, M. 1985. Computational Geometry-An Introduction. Springer, Berlin, Germany.

Roussopoulos, N., Kelly, S., AND VINCENT, F. 1995. Nearest neighbor queries. In Proceedings of the ACM Conference on the Management of Data (SIGMOD; San Jose, CA, May 22-25). 71-79.

Sakurai, Y., Yoshikawa, M., Uemura, S., and Kojima, H. 2000. The A-tree: An index structure for high-dimensional spaces using relative approximation. In Proceedings of the Very Large Data Bases Conference (VLDB; Cairo, Egypt, Sep. 10-14). 516-526.

SALZBERG, B. AND TSOTRAS, V. 1999. A comparison of access methods for temporal data. ACM Comput. Surv. 31, 2, 158-221.

SELLIS, T., RoussopouLos, N., AND FALOUTSOS, C. 1987. The R+-tree: A dynamic index for multidimensional objects. In Proceedings of the Very Large Data Bases Conference (VLDB; Brighton, England, Sep. 1-4). 507-518.

Steuer, R. 1986. Multiple Criteria Optimization. Wiley, New York, NY.

TAN, K., ENG, P., AND OOI, B. 2001. Efficient progressive skyline computation. In Proceedings of the Very Large Data Bases Conference (VLDB; Rome, Italy, Sep. 11-14). 301-310.

Theoporidis, Y., Stefanakis, E., and Sellis, T. 2000. Efficient cost models for spatial queries using R-trees. IEEE Trans. Knowl. Data Eng. 12, 1, 19-32.