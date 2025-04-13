# An asymptotically optimal multiversion B-tree

# 一种渐近最优的多版本B树

Bruno Becker ${}^{2}$ ,Stephan Gschwind ${}^{2}$ ,Thomas Ohler ${}^{2}$ ,Bernhard Seeger ${}^{3}$ ,Peter Widmayer ${}^{1}$

布鲁诺·贝克尔 ${}^{2}$ 、斯蒂芬·格施温德 ${}^{2}$ 、托马斯·奥勒 ${}^{2}$ 、伯恩哈德·塞格 ${}^{3}$ 、彼得·维德迈尔 ${}^{1}$

${}^{1}$ Institut für Theoretische Informatik,ETH Zentrum,CH-8092 Zürich,Switzerland

${}^{1}$ 瑞士苏黎世联邦理工学院理论计算机科学研究所，CH - 8092 苏黎世

Tel. ++41-1-63-27400, Fax ++41-1-63-21172, email: widmayer@inf.ethz.ch

电话：++41 - 1 - 63 - 27400，传真：++41 - 1 - 63 - 21172，电子邮件：widmayer@inf.ethz.ch

${}^{2}$ isys software gmbh,Ensisheimer Str. 2a,D-79110 Freiburg,Germany

${}^{2}$ 德国弗莱堡isys软件有限公司，恩西舍姆大街2a号，D - 79110 弗莱堡

${}^{3}$ Philipps-Universität Marburg,Fachbereich Mathematik,Fachgebiet Informatik,Hans-Meerwein-Strasse,D-35032 Marburg,Germany

${}^{3}$ 德国马尔堡菲利普斯大学，数学系，计算机科学专业，汉斯 - 米尔魏因大街，D - 35032 马尔堡

Abstract. In a variety of applications, we need to keep track of the development of a data set over time. For maintaining and querying these multiversion data efficiently, external storage structures are an absolute necessity. We propose a multiversion B-tree that supports insertions and deletions of data items at the current version and range queries and exact match queries for any version, current or past. Our multiversion B-tree is asymptotically optimal in the sense that the time and space bounds are asymptotically the same as those of the (single-version) B-tree in the worst case. The technique we present for transforming a (single-version) B-tree into a multiversion B-tree is quite general: it applies to a number of hierarchical external access structures with certain properties directly, and it can be modified for others.

摘要：在各种应用中，我们需要跟踪数据集随时间的发展情况。为了高效地维护和查询这些多版本数据，外部存储结构是绝对必要的。我们提出了一种多版本B树，它支持在当前版本插入和删除数据项，以及对当前或过去的任何版本进行范围查询和精确匹配查询。我们的多版本B树在渐近意义上是最优的，即在最坏情况下，其时间和空间界限与（单版本）B树的渐近相同。我们提出的将（单版本）B树转换为多版本B树的技术非常通用：它可以直接应用于具有某些属性的许多分层外部访问结构，并且可以针对其他结构进行修改。

Key words: Information systems - Physical design - Access methods - Versioned data

关键词：信息系统 - 物理设计 - 访问方法 - 版本化数据

## 1 Introduction

## 1 引言

The importance of not only maintaining data in their latest version, but also keeping track of their development over time, has been widely recognized (Tansel et al. 1993). Version data in engineering databases (Katz 1990) and time-oriented data (Clifford and Ariav 1986) are two prime examples for situations in which the concepts of versions and time are visible to the user. In multiversion concurrency control (Barghouti and Kaiser 1991; Bernstein et al. 1987), these concepts are transparent to the user, but they are used by the system (e.g. the scheduler) for concurrency control and recovery purposes. In this paper, we are concerned with access structures that support version-based operations on external storage efficiently. We follow the convention of Bernstein et al. (1987) and Driscoll et al. (1989) in that each update to the data creates a new version; note that this differs from the terminology in engineering databases, where an explicit operation exists for creating versions, and versions of design objects are equipped with semantic properties and mechanisms, such as inheritance or change propagation. Our choice of creating a new version after each update turns out not to be restrictive, in the sense that the data-structuring method we propose can be easily adapted to create versions only on request, without loss of efficiency.

不仅维护数据的最新版本，还跟踪其随时间的发展情况，其重要性已得到广泛认可（坦塞尔等人，1993年）。工程数据库中的版本数据（卡茨，1990年）和面向时间的数据（克利福德和阿里亚夫，1986年）是用户能够看到版本和时间概念的两种典型情况。在多版本并发控制中（巴尔古蒂和凯泽，1991年；伯恩斯坦等人，1987年），这些概念对用户是透明的，但系统（如调度器）会将其用于并发控制和恢复目的。在本文中，我们关注的是能够高效支持外部存储上基于版本操作的访问结构。我们遵循伯恩斯坦等人（1987年）和德里斯科尔等人（1989年）的约定，即每次对数据的更新都会创建一个新版本；请注意，这与工程数据库中的术语不同，在工程数据库中，存在用于创建版本的显式操作，并且设计对象的版本具有语义属性和机制，如继承或变更传播。我们选择在每次更新后创建新版本并不具有局限性，因为我们提出的数据结构方法可以很容易地进行调整，仅在需要时创建版本，而不会损失效率。

We are interested in asymptotically worst-case efficient access structures for external storage that support at least insertions, deletions, exact-match queries (associative search) - the dictionary operations (Sedgewick 1988; Mehlhorn and Tsakalidis 1990; Gonnet and Baeza-Yates 1991) - and range queries in addition to application-specific operations such as purging of old enough versions in concurrency control. That is, we aim at a theoretical understanding of the fundamentals of multiversion access to data, with little attention to constant factors [studies with this flavor have attracted interest in other areas, too (Kanellakis et al. 1993, Vitter 1991)]. We limit our discussion to the situation in which a change can only be applied to the current version, whereas queries can be performed on any version, current or past. Some authors call this a management problem for partially persistent data; we call an access structure that supports the required operations efficiently a multiversion structure.

我们对外部存储的渐近最坏情况高效访问结构感兴趣，这些结构至少支持插入、删除、精确匹配查询（关联搜索）——字典操作（塞奇威克，1988年；梅尔霍恩和查卡利迪斯，1990年；贡内特和贝萨 - 耶茨，1991年）——以及范围查询，此外还支持特定于应用的操作，如并发控制中清除足够旧的版本。也就是说，我们旨在从理论上理解数据多版本访问的基本原理，而很少关注常数因子[具有这种特点的研究在其他领域也引起了关注（卡内拉基斯等人，1993年，维特，1991年）]。我们将讨论限制在这样一种情况，即更改只能应用于当前版本，而查询可以在当前或过去的任何版本上执行。一些作者将此称为部分持久数据的管理问题；我们将能够高效支持所需操作的访问结构称为多版本结构。

The problem in designing a multiversion access structure lies in the fact that data are on external storage. For main memory, there is a recipe for designing a multiversion structure, given a single-version structure. More precisely, any single-version main memory data structure in a very general class, based on pointers from record to record, can be transformed into a multiversion structure, with no change in the amortized asymptotic worst-case time and space costs, by applying a general technique (Driscoll et al. 1989). For the special case of balanced binary search trees, this efficiency is achieved even in the worst case per operation - clearly a perfect result.

设计多版本访问结构的问题在于数据存储在外部存储设备上。对于主存，在给定单版本结构的情况下，有一个设计多版本结构的方法。更准确地说，基于记录到记录的指针的非常通用类别的任何单版本主存数据结构，都可以通过应用一种通用技术（德里斯科尔等人，1989年）转换为多版本结构，而不会改变摊销后的渐近最坏情况时间和空间成本。对于平衡二叉搜索树的特殊情况，甚至在每次操作的最坏情况下也能实现这种效率——显然是一个完美的结果。

Given quite a general recipe for transforming single-version main memory data structures into multiversion structures, it is an obvious temptation to apply that recipe accordingly to external access structures. This can be done by simply viewing a block in the external structure as a record in the main memory structure. At first glance, this models block access operations well; unfortunately, it does not model storage space appropriately, in that the size of a block is not taken into consideration. That is, a block is viewed to store a constant number of data items, and the constant is of no concern. Even worse, the direct application of the recipe consumes one block of storage space for each data item. However, no external data structure can ever be satisfactory unless it stores significantly more than one data item in a block on average; balanced structures, such as the B-tree variants, actually require to store in each block at least some constant fraction of the number of items the block can hold (the latter being called the block capacity $b$ ). As a consequence, the space efficiency of this approach is clearly unacceptable, and this also entails an unacceptable time complexity.

鉴于有一个将单版本主存数据结构转换为多版本结构的通用方法，很自然地会想将该方法应用于外部访问结构。这可以通过简单地将外部结构中的一个块视为主存结构中的一条记录来实现。乍一看，这种方法能很好地模拟块访问操作；不幸的是，它不能恰当地模拟存储空间，因为没有考虑块的大小。也就是说，一个块被视为存储固定数量的数据项，而这个固定数量并不受关注。更糟糕的是，直接应用该方法会为每个数据项消耗一个块的存储空间。然而，除非一个外部数据结构平均每个块存储的数据项显著多于一个，否则它永远不会令人满意；像B树变体这样的平衡结构，实际上要求每个块至少存储该块所能容纳数据项数量的某个固定比例（后者称为块容量 $b$ ）。因此，这种方法的空间效率显然是不可接受的，这也导致了不可接受的时间复杂度。

---

<!-- Footnote -->

Correspondence to: P. Widmayer

通信作者：P. Widmayer

<!-- Footnote -->

---

It is the contribution of this paper ${}^{1}$ to propose a technique for transforming single-version external access structures into multiversion structures, at the cost of a constant factor in time and space requirements,where the block capacity $b$ is not considered to be a constant. That is, the asymptotic bounds for the worst case remain the same as for the corresponding single-version structure, but the involved constants change. We call such a multiversion structure asymptotically optimal, because the asymptotic worst-case bounds certainly cannot decrease by adding multiversion capabilities to a data structure. Our result holds for a certain class of hierarchical external access structures. It is worth noting that this class contains the B-tree and its variants, not only because the B-tree is an ubiquitous external data structure, but also because an asymptotically optimal multiversion B-tree has not been obtained so far, despite the considerable interest this problem has received in the literature. Since we are interested primarily in the asymptotic efficiency, we will discuss the involved constants only later in the paper. Multiversion structures with excellent asymptotic worst-case bounds for insert and exact-match operations (but not for delete) and for related problems have been obtained previously; we will discuss them in some detail later in the paper.

本文 ${}^{1}$ 的贡献在于提出一种将单版本外部访问结构转换为多版本结构的技术，该技术在时间和空间需求上仅增加一个常数因子，且块容量 $b$ 不被视为常数。也就是说，最坏情况下的渐近边界与相应的单版本结构保持相同，但涉及的常数会发生变化。我们称这样的多版本结构为渐近最优的，因为通过为数据结构添加多版本功能，最坏情况下的渐近边界肯定不会降低。我们的结果适用于某一类分层外部访问结构。值得注意的是，这类结构包含B树及其变体，这不仅是因为B树是一种普遍存在的外部数据结构，还因为到目前为止，尽管文献中对这个问题有相当大的兴趣，但尚未得到渐近最优的多版本B树。由于我们主要关注渐近效率，因此将在论文后面再讨论涉及的常数。此前已经获得了在插入和精确匹配操作（但不包括删除操作）以及相关问题上具有出色渐近最坏情况边界的多版本结构；我们将在论文后面详细讨论它们。

For the sake of concreteness, we base the presentation of our technique in this paper on B-trees; it is implicit how to apply our technique to other hierarchical structures. Each data item stored in the tree consists of a key and an information part; access to data items is by key only, and the keys are supposed to be taken from some linearly ordered set. Let us restrict our presentation to the following operations:

为了具体说明，本文将我们的技术基于B树进行介绍；如何将我们的技术应用于其他分层结构是不言而喻的。树中存储的每个数据项由一个键和一个信息部分组成；对数据项的访问仅通过键进行，并且假设这些键取自某个线性有序集。让我们将介绍限制在以下操作上：

- Insert (key,info): insert a record with given key and info component into the current version; this operation creates a new version.

- 插入（键，信息）：将具有给定键和信息组件的记录插入到当前版本中；此操作会创建一个新版本。

- Delete (key): delete the (unique) record with given key from the current version; this operation creates a new version.

- 删除（键）：从当前版本中删除具有给定键的（唯一）记录；此操作会创建一个新版本。

- Exact-match query (key,version): return the (unique) record with given key in the given version; this operation does not create a new version.

- 精确匹配查询（键，版本）：返回给定版本中具有给定键的（唯一）记录；此操作不会创建新版本。

- Range query (lowkey,highkey,version): return all records whose key lies between the given lowkey and the given highkey in the given version; this operation does not create a new version.

- 范围查询（低键，高键，版本）：返回给定版本中键位于给定低键和给定高键之间的所有记录；此操作不会创建新版本。

Before briefly reviewing the previous approaches of designing a B-tree that supports these operations efficiently, let us state the strongest efficiency requirements that a multiversion B-tree can be expected to satisfy. To this end, consider a sequence of $N$ update operations (insert or delete),applied to the initially empty structure,and let ${m}_{i}$ be the number of data items present after the $i$ -th update (we say,in version $i$ ), $0 \leq  i \leq  N$ . Then a multiversion B-tree with the following properties holding for each $i$ (all bounds are for the worst case) is the best we can expect:

在简要回顾之前设计能有效支持这些操作的B树的方法之前，让我们先说明一个多版本B树有望满足的最强效率要求。为此，考虑对初始为空的结构应用 $N$ 个更新操作（插入或删除）的序列，并设 ${m}_{i}$ 为第 $i$ 次更新后（我们称在版本 $i$ 中）存在的数据项数量， $0 \leq  i \leq  N$ 。那么，对于每个 $i$ 都具有以下属性（所有边界都是针对最坏情况）的多版本B树是我们所能期望的最佳情况：

- For the first $i$ versions,altogether the tree requires $O\left( {i/b}\right)$ blocks of storage space.

- 在前 $i$ 个版本中，树总共需要 $O\left( {i/b}\right)$ 个块的存储空间。

- The $\left( {i + 1}\right)$ -th update (insertion or deletion) accesses and modifies $O\left( {{\log }_{b}{m}_{i}}\right)$ blocks.

- 第 $\left( {i + 1}\right)$ 次更新（插入或删除）访问并修改 $O\left( {{\log }_{b}{m}_{i}}\right)$ 个块。

- An exact-match query in version $i$ accesses $O\left( {{\log }_{b}{m}_{i}}\right)$ blocks.

- 在版本 $i$ 中的精确匹配查询访问 $O\left( {{\log }_{b}{m}_{i}}\right)$ 个块。

- A range query in version $i$ that returns $r$ records accesses $O\left( {{\log }_{b}{m}_{i} + r/b}\right)$ blocks.

- 在版本 $i$ 中返回 $r$ 条记录的范围查询访问 $O\left( {{\log }_{b}{m}_{i} + r/b}\right)$ 个块。

The reason why these are lower bounds is the following. For a query to any version $i$ ,the required efficiency is the same as if the data present in version $i$ were maintained separately in their own B-tree. For insertions and deletions on the current version, the required efficiency is the same as for a (single-version) B-tree maintaining the data set valid for the current version. In other words, a better multiversion B-tree would immediately yield a better B-tree.

这些是下界的原因如下。对于对任何版本$i$的查询，所需的效率与将版本$i$中存在的数据单独维护在它们自己的B树中时相同。对于当前版本的插入和删除操作，所需的效率与维护当前版本有效数据集的（单版本）B树相同。换句话说，一个更好的多版本B树将立即产生一个更好的B树。

This paper presents a multiversion B-tree structure satisfying these efficiency requirements, under the assumption that in a query, access to the root of the requested B-tree has only constant cost [we could even tolerate a cost of $O\left( {{\log }_{b}{m}_{i}}\right)$ ,to be asymptotically precise]. We have thus separated the concerns of, first, identifying the requested version, and, second, querying the requested version (that is, the root of the appropriate B-tree). This separation of concerns makes sense because in an application of a multiversion structure, access to the requested version may be supported from the context, such as in concurrency control. For instance, the block address of the requested root block may directly be known (possibly from previous accesses) or only a constant number of versions might be relevant for queries, such that the root block can be accessed in time $O\left( 1\right)$ . This assumption has been made in other papers (Driscoll et al. 1989; Lanka and Mays 1991), allowing the investigation to concentrate on querying within a version. In this paper, we follow this view and try to take advantage of a possibly direct version access for querying a version. We therefore concern ourselves with ways to identify the requested version only later, with little emphasis, since any of a number of search techniques can be applied for this purpose. Note that if we do not separate these issues, but instead assume that the root of the requested B-tree needs to be identified through a search operation, $\Omega \left( {{\log }_{b}N}\right)$ instead of $\Omega \left( {{\log }_{b}{m}_{i}}\right)$ is a lower bound on the run-time of a query, since one item out of as many as $N$ items needs to be found.

本文提出了一种满足这些效率要求的多版本B树结构，假设在查询中，访问所请求B树的根节点仅需常数成本[为了渐近精确，我们甚至可以容忍$O\left( {{\log }_{b}{m}_{i}}\right)$的成本]。因此，我们将问题分为两部分，首先是识别所请求的版本，其次是查询所请求的版本（即适当B树的根节点）。这种关注点的分离是有意义的，因为在多版本结构的应用中，对所请求版本的访问可以从上下文得到支持，例如在并发控制中。例如，所请求的根块的块地址可能直接已知（可能来自先前的访问），或者对于查询而言，可能只有常数数量的版本是相关的，这样就可以在时间$O\left( 1\right)$内访问根块。其他论文（Driscoll等人，1989年；Lanka和Mays，1991年）也做了这个假设，以便将研究集中在一个版本内的查询上。在本文中，我们遵循这一观点，并尝试利用可能的直接版本访问来查询一个版本。因此，我们稍后才关注识别所请求版本的方法，并且不太强调这一点，因为可以为此目的应用多种搜索技术。请注意，如果我们不分离这些问题，而是假设需要通过搜索操作来识别所请求B树的根节点，那么查询的运行时间下界是$\Omega \left( {{\log }_{b}N}\right)$而不是$\Omega \left( {{\log }_{b}{m}_{i}}\right)$，因为需要从多达$N$个项中找到一个项。

In building multiversion structures, there is a general tradeoff between storage space, update time and query time. For instance, building an extra copy of the structure at each update is extremely slow for updates and extremely costly in space, but extremely fast for queries. Near the other extreme, Kolovson and Stonebraker (1989) view versions (time) as an extra dimension and store one-dimensional version intervals in two-dimensional space in an R-tree. As a consequence of using an R-tree, they can also maintain one-dimensional key intervals (and not only single keys). This gives good storage space efficiency, but query efficiency need not be as good, because the R-tree gives no guarantee on selectivity. That is,even if access to version $i$ is taken care of in the context, the time to answer a query on version $i$ does not depend on the number of items in that version only, but instead on the total number of all updates. We will discuss other multiversion B-trees suggested in the literature in Sect. 5; none of them achieves asymptotically optimal performance in time and space.

在构建多版本结构时，存储空间、更新时间和查询时间之间存在一般的权衡。例如，在每次更新时构建结构的额外副本，对于更新操作来说极其缓慢，并且在空间上极其昂贵，但对于查询来说极其快速。在另一个极端附近，Kolovson和Stonebraker（1989年）将版本（时间）视为一个额外的维度，并将一维的版本区间存储在二维空间的R树中。由于使用了R树，他们还可以维护一维的键区间（而不仅仅是单个键）。这提供了良好的存储空间效率，但查询效率可能并不理想，因为R树对选择性没有保证。也就是说，即使在上下文中处理了对版本$i$的访问，回答关于版本$i$的查询的时间不仅取决于该版本中的项数，还取决于所有更新的总数。我们将在第5节讨论文献中提出的其他多版本B树；它们都没有在时间和空间上实现渐近最优性能。

---

<!-- Footnote -->

${}^{1}$ A preliminary version of this paper has been published (Becker et al. 1993).

${}^{1}$本文的一个初步版本已发表（Becker等人，1993年）。

<!-- Footnote -->

---

In Sect. 2, we present an optimal multiversion B-tree. Our description suggests a rather general method for transforming hierarchical external data structures into optimal multiversion structures, provided that operations proceed in a certain way along paths between the root and the leaves. But even if the external single version data structure does not precisely follow the operation pattern we request (as in the case of R-trees, for instance), we conjecture that the basic ideas carry over to an extent that makes a corresponding multiversion structure competitive and useful. Section 3 provides an efficiency analysis of our multiversion B-tree, and Sect. 4 adds some thoughts around the main result. Section 5 puts the obtained result into perspective, by comparing it with previous work, and Sect. 6 concludes the paper.

在第2节中，我们提出了一种最优的多版本B树。我们的描述提出了一种相当通用的方法，用于将分层外部数据结构转换为最优的多版本结构，前提是操作沿着根节点和叶子节点之间的路径以某种方式进行。但是，即使外部单版本数据结构并不完全遵循我们要求的操作模式（例如在R树的情况下），我们推测基本思想在一定程度上仍然适用，使得相应的多版本结构具有竞争力且有用。第3节对我们的多版本B树进行了效率分析，第4节围绕主要结果添加了一些思考。第5节通过将所得结果与先前的工作进行比较，对其进行了全面审视，第6节对本文进行了总结。

## 2 An optimal multiversion B-tree

## 2 一种最优的多版本B树

We present our technique to transform single-version external access structures into multiversion structures using the example of the leaf-oriented B-tree.

我们以面向叶子的B树为例，介绍将单版本外部访问结构转换为多版本结构的技术。

### 2.1 The basic idea

### 2.1 基本思想

To achieve the desired behavior, we associate insertion and deletion versions with items, since items of different lifespans need to be stored in the same block. Let $<$ key, in version,del version,info > denote a data item,stored in a leaf, with a key that is unique for any given version, an associated information, and a lifespan from its insertion version in version to its deletion version del version. Similarly, an entry in an inner node of the tree is denoted by $<$ router,in version,del version,reference $>$ ; the router, together with the in version and del version information on the referenced subtree, guides the search for a data item. For example, the B-tree uses a separator key and the R-tree uses a rectangle as a router.

为实现预期行为，我们将插入版本和删除版本与数据项关联起来，因为不同生命周期的数据项需要存储在同一个块中。设 $<$ <键，插入版本，删除版本，信息> 表示一个存储在叶子节点的数据项，其键在任何给定版本中都是唯一的，包含关联信息，并且其生命周期从插入版本 “插入版本” 到删除版本 “删除版本”。类似地，树的内部节点中的一个条目用 $<$ <路由，插入版本，删除版本，引用 $>$ > 表示；路由与被引用子树的插入版本和删除版本信息一起，指导对数据项的搜索。例如，B 树使用分隔键，而 R 树使用矩形作为路由。

From a bird's eye view, the multiversion B-tree is a directed acyclic graph of B-tree nodes that results from certain incremental changes to an initial B-tree. In particular, the multiversion B-tree embeds a number of B-trees; it has a number of B-tree root nodes that partition the versions from the first to the current one in such a way that each B-tree root stands for an interval of versions. A query for a given version can then be answered by entering the multiversion B-tree at the corresponding root.

从宏观角度看，多版本 B 树是一个由 B 树节点组成的有向无环图，它是对初始 B 树进行某些增量更改的结果。具体而言，多版本 B 树嵌入了多个 B 树；它有多个 B 树根节点，这些根节点将从第一个版本到当前版本的所有版本进行划分，使得每个 B 树根节点代表一个版本区间。然后，可以通过在相应的根节点处进入多版本 B 树来回答针对给定版本的查询。

Each update (insert or delete operation) creates a new version; the $i$ -th update creates version $i$ . An entry is said to be of version $i$ if its lifespan contains $i$ . A block is said to be live if it has not been copied, and dead otherwise. In a live block,deletion version $*$ for an entry denotes that the entry has not yet been deleted at present; in a dead block, it indicates that the entry has not been deleted before the block died. For each version $i$ and each block $A$ except the roots of versions, we require that the number of entries of version $i$ in block $A$ is either zero or at least $d$ ,where $b = k \cdot  d$ for block capacity $b$ and some constant $k$ (assume for simplicity that $b,k,d$ are all integers and $b$ is the same for directory and data blocks); we call this the weak version condition.

每次更新（插入或删除操作）都会创建一个新版本；第 $i$ 次更新创建版本 $i$。如果一个条目的生命周期包含 $i$，则称该条目为版本 $i$ 的条目。如果一个块尚未被复制，则称其为活动块，否则为非活动块。在活动块中，条目的删除版本 $*$ 表示该条目目前尚未被删除；在非活动块中，它表示该条目在块变为非活动状态之前尚未被删除。对于每个版本 $i$ 和除版本根节点之外的每个块 $A$，我们要求块 $A$ 中版本 $i$ 的条目数量要么为零，要么至少为 $d$，其中 $b = k \cdot  d$ 是块容量 $b$ 和某个常数 $k$ 的函数（为简单起见，假设 $b,k,d$ 都是整数，并且目录块和数据块的 $b$ 相同）；我们将此称为弱版本条件。

Operations that do not entail structural changes are performed in the straightforward way that can be inferred from the single-version structure by taking the lifespan of entries into account. That is, an entry inserted by update operation $i$ into a block carries a lifespan of $\lbrack i, * )$ at the time of insertion; deletion of an entry by update operation $i$ from a block changes its del version from $*$ to $i$ .

不涉及结构更改的操作以一种直接的方式执行，这种方式可以通过考虑条目的生命周期从单版本结构中推断出来。也就是说，通过更新操作 $i$ 插入到块中的条目在插入时具有 $\lbrack i, * )$ 的生命周期；通过更新操作 $i$ 从块中删除一个条目会将其删除版本从 $*$ 更改为 $i$。

Structural changes are triggered in two ways. First, a block overflow occurs as the result of an insertion of an entry into a block that already contains $b$ entries. A block underflow, as in B-trees, for example, cannot occur, since entries are never removed from blocks. However, the weak version condition may be violated in a non-root block as a result of a deletion; such a weak version underflow occurs if an entry is deleted in a block with exactly $d$ current entries. Moreover, we say that a weak version underflow occurs in the root of the present version if there is only one live entry (except for the pathological case in which the tree contains only one record in the present version).

结构更改通过两种方式触发。首先，当一个条目插入到已经包含 $b$ 个条目的块中时，会发生块溢出。例如，与 B 树不同，块下溢不会发生，因为条目永远不会从块中移除。然而，由于删除操作，非根块中的弱版本条件可能会被违反；如果在一个恰好有 $d$ 个当前条目的块中删除一个条目，则会发生这种弱版本下溢。此外，我们称如果当前版本的根节点中只有一个活动条目（除了树在当前版本中只包含一条记录这种特殊情况），则当前版本的根节点发生了弱版本下溢。

The structural modification after a block overflow copies the block and removes all but the current entries from the copy. We call this operation a version split; it is comparable to a time split at the current time in Lomet and Salzberg (1989); equivalently, it may be compared to the node-copying operation of Driscoll et al. (1989). In general, a copy produced by this version split may be an almost full block. In that case, a few subsequent insertions would again trigger a version split,resulting in a space cost of $\Theta \left( 1\right)$ block per insertion. To avoid this and the similar phenomenon of an almost empty block, we request that immediately after a version split,at least $\varepsilon  \cdot  d + 1$ insert operations or delete operations are necessary to arrive at the next block overflow or version underflow in that block,for some constant $\varepsilon$ to be defined more precisely in the next section (assume for simplicity that $\varepsilon  \cdot  d$ is integer). As a consequence,the number of current entries after a version split must be in the range from $\left( {1 + \varepsilon }\right)  \cdot  d$ to $\left( {k - \varepsilon }\right)  \cdot  d$ ; we call this the strong version condition. If a version split leads to less than $\left( {1 + \varepsilon }\right)  \cdot  d$ entries - we say: a strong version underflow occurs - a merge is attempted with a copy of a sibling block containing only its current entries. If necessary, this merge must be followed by a version-independent split according to the key values of the items in the block - a key split. Similarly, if a version split leads to more than $\left( {k - \varepsilon }\right)  \cdot  d$ entries in a block - we say: a strong version overflow occurs - a key split is performed.

块溢出后的结构修改会复制该块，并从副本中移除除当前条目之外的所有内容。我们将此操作称为版本分割；这类似于洛梅特（Lomet）和萨尔茨伯格（Salzberg）（1989 年）中在当前时间进行的时间分割；等效地，它可以与德里斯科尔（Driscoll）等人（1989 年）的节点复制操作相比较。一般来说，由这种版本分割产生的副本可能是一个几乎满的块。在这种情况下，随后的几次插入操作会再次触发版本分割，导致每次插入的空间成本为$\Theta \left( 1\right)$个块。为避免这种情况以及几乎为空的块的类似现象，我们要求在版本分割之后，至少需要进行$\varepsilon  \cdot  d + 1$次插入操作或删除操作，才能使该块发生下一次块溢出或版本下溢，其中常数$\varepsilon$将在下一节中更精确地定义（为简单起见，假设$\varepsilon  \cdot  d$为整数）。因此，版本分割后当前条目的数量必须在$\left( {1 + \varepsilon }\right)  \cdot  d$到$\left( {k - \varepsilon }\right)  \cdot  d$的范围内；我们将此称为强版本条件。如果版本分割导致条目数量少于$\left( {1 + \varepsilon }\right)  \cdot  d$个——我们称之为：发生强版本下溢——则尝试与仅包含其当前条目的兄弟块的副本进行合并。如有必要，此合并之后必须根据块中项的键值进行与版本无关的分割——键分割。类似地，如果版本分割导致块中的条目数量超过$\left( {k - \varepsilon }\right)  \cdot  d$个——我们称之为：发生强版本溢出——则执行键分割。

### 2.2 An example

### 2.2 示例

To illustrate the basic ideas described above, let us discuss the following example of a multiversion B-tree that organizes records with an integer key. The initial situation (i.e. first version) of our multiversion B-tree is given in Fig. 1a.

为了说明上述基本思想，让我们讨论以下组织具有整数键记录的多版本 B 树的示例。我们的多版本 B 树的初始情况（即第一个版本）如图 1a 所示。

<!-- Media -->

<!-- figureText: <10,1,*,A> 2nd version 3rd version B <10,1,*> <45,1,*> <15,1,*> <55,1,*> <25,1,*> <65,1,3> <30,1*> <70,1,*> <35,1,*> <75,1,*> <40,2,*> <80,1,*> (b) <45,1,*,B> B <10,1,*> <45,1,*> <15,1,*> <55,1,*> <25,1,*> <65,1,*> <70,1,*> <35,1,*> <75.1.*> <80,1,*> (a) -->

<img src="https://cdn.noedgeai.com/0195c902-d752-7cdd-b0a3-cd71d4e4465a_3.jpg?x=113&y=74&w=722&h=453&r=0"/>

Fig. 1. Development of the multiversion B-tree up to the third version

图 1. 多版本 B 树发展到第三个版本的情况

<!-- figureText: R <10,1,8,A> <45,1,*,B> <5,8,*,A*> A A* <10,1,*> <10,1,*> <45,1,*> <15,1,5> <40,2,*> <55,1,*> <25,1,7> <5,8,*> <65,1,3> <30,1,6> <70,1,*> <35,1,4> <75,1,*> <80,1,*> (b) <10,1,*,A> <45,1,*,B> B <10,1,*> <45,1,*> <15,1,5> <55.1.*> <25,1,7> <65,1,3> <30,1,6> <70,1,*> <35,1,4> <75,1,*> <80,1,*> (a) -->

<img src="https://cdn.noedgeai.com/0195c902-d752-7cdd-b0a3-cd71d4e4465a_3.jpg?x=113&y=586&w=736&h=431&r=0"/>

Fig. 2. a The seventh version of the multiversion B-tree; b the multiversion B-tree after version split of block A

图 2. a 多版本 B 树的第七个版本；b 块 A 进行版本分割后的多版本 B 树

<!-- Media -->

For the sake of simplicity of our example, we assume that already 11 data records are in the first version. The multiversion B-tree consists of three blocks: a root $R$ and two leaves $A$ and $B$ . The parameters of the multiversion B-tree are set up in the following way: $b = 6,d = 2$ ,and $\varepsilon  =$ 0.5 . Hence, after a structural change, a new block contains at least three and at most five current entries.

为了简化我们的示例，我们假设第一个版本中已经有 11 条数据记录。多版本 B 树由三个块组成：一个根节点$R$和两个叶子节点$A$和$B$。多版本 B 树的参数设置如下：$b = 6,d = 2$，且$\varepsilon  =$ = 0.5。因此，在结构更改后，一个新块至少包含三个且最多包含五个当前条目。

The second version is created by the operation insert(40), adding a new entry to block $A$ . In Figure 1b,for the second and the third version, the result of the corresponding update operation is shown by depicting the block which has been modified. The next operation delete(65) creates the third version. As shown in Fig. 1b, for the deletion of a record, the deletion version of the corresponding entry is set to the current version,overwriting the $*$ marker.

第二个版本是通过插入操作 insert(40) 创建的，该操作向块$A$添加了一个新条目。在图 1b 中，对于第二个和第三个版本，相应更新操作的结果通过描绘已修改的块来展示。下一个删除操作 delete(65) 创建了第三个版本。如图 1b 所示，对于记录的删除，相应条目的删除版本被设置为当前版本，覆盖了$*$标记。

To be able to illustrate different underflow and overflow situations, let us assume further updates - delete(35), delete(15), delete(30) and delete(25) - resulting in the seventh version of the multiversion B-tree (Fig.2a).

为了能够说明不同的下溢和溢出情况，让我们假设进行进一步的更新操作——delete(35)、delete(15)、delete(30) 和 delete(25)——从而得到多版本 B 树的第七个版本（图 2a）。

Now, let us consider two different cases for creating the eighth version of the multiversion B-tree, illustrating the various types of structural changes.

现在，让我们考虑创建多版本 B 树第八个版本的两种不同情况，以说明各种类型的结构更改。

In the first case, we consider the operation insert(5) to create the eighth version of the multiversion B-tree. This results in a block overflow of block $A$ that is eliminated by performing a version split on that block. All current entries of block $A$ are now copied into a new live block ${A}^{ * }$ . Because block ${A}^{ * }$ fulfills the strong version condition,no further restructuring is needed. Eventually,the parent block $R$ is updated accordingly (Fig. 2b).

在第一种情况下，我们考虑使用插入操作 insert(5) 来创建多版本 B 树的第八个版本。这导致块$A$发生块溢出，通过对该块执行版本分割来消除溢出。块$A$的所有当前条目现在都被复制到一个新的活动块${A}^{ * }$中。由于块${A}^{ * }$满足强版本条件，因此不需要进一步的结构调整。最终，父块$R$会相应地更新（图 2b）。

<!-- Media -->

<!-- figureText: A <10,1,8,A> <45,1,8,B> <10,8,*,C> <70,8,*,D> C D <10,1,*> <70,1,*> <45.1,*> <75.1.*> <55,1,*> <80,1,*> <10,1,*> <45,1,*> <15,1,5> <55,1,*> <25,1,7> <65,1,3> <30,1,6> <70,1,*> <40,2,8> <80,1,*> -->

<img src="https://cdn.noedgeai.com/0195c902-d752-7cdd-b0a3-cd71d4e4465a_3.jpg?x=888&y=78&w=524&h=426&r=0"/>

Fig. 3. Structural changes after weak version underflow of block $A$

图 3. 块$A$发生弱版本下溢后的结构更改

<!-- figureText: R1 R1 R2 <10,1,8,A> <10,11,*,E> <45,1,8,B> <70,18,*,G> <10,8,11,C> <70,8,15,D> <10,11,*,E> <70,15,18,F> <10,1,8,A> <45,1,8,B> <10,8,11,C> <70,8,15,D> <10,11,*,E> <70,15,18,F> <70,18,*,G> -->

<img src="https://cdn.noedgeai.com/0195c902-d752-7cdd-b0a3-cd71d4e4465a_3.jpg?x=887&y=585&w=625&h=240&r=0"/>

Fig. 4. Creation of two roots ${R1},{R2}$ by version split of root block ${R1}$

图4. 通过根块${R1}$的版本拆分创建两个根${R1},{R2}$

<!-- Media -->

In the second case, the eighth version is created by operation delete(40), which leads to a weak version underflow, i.e. the number of current entries in block $A$ is less than $d$ $\left( { = 2}\right)$ . Then,a version split is performed on block $A$ ,copying the current entries of block $A$ into a new block ${A}^{ * }$ . Now a strong version underflow occurs in ${A}^{ * }$ ,which is treated by merging this block with a block resulting from version split of a sibling block. In our example, $B$ is found to be a sibling. Accordingly,by version split a temporary block ${B}^{ * }$ is created from $B$ and blocks ${A}^{ * }$ and ${B}^{ * }$ are merged. As in our example, a block resulting from a merge can violate the strong version condition. To treat the strong version overflow,a key split is performed,creating two new blocks $C$ and $D$ . Because a key split is always balanced for a B-tree, blocks $C$ and $D$ fulfill the strong version condition. Eventually,the parent block $R$ has to be updated by overwriting the $*$ of the entries which refer to block $A$ and $B$ and inserting two new current entries,referring to blocks $C$ and $D$ (Fig. 3). Now,blocks $A$ and $B$ are dead and blocks $C$ and $D$ are live.

在第二种情况下，第八个版本通过操作delete(40)创建，这导致了弱版本下溢，即块$A$中的当前条目数量少于$d$ $\left( { = 2}\right)$。然后，对块$A$执行版本拆分，将块$A$的当前条目复制到一个新块${A}^{ * }$中。现在，${A}^{ * }$中出现了强版本下溢，通过将该块与兄弟块版本拆分产生的块合并来处理。在我们的示例中，发现$B$是一个兄弟块。因此，通过版本拆分从$B$创建一个临时块${B}^{ * }$，并将块${A}^{ * }$和${B}^{ * }$合并。正如我们的示例所示，合并产生的块可能违反强版本条件。为了处理强版本溢出，执行键拆分，创建两个新块$C$和$D$。因为对于B树来说，键拆分总是平衡的，所以块$C$和$D$满足强版本条件。最终，父块$R$必须通过覆盖引用块$A$和$B$的条目的$*$并插入两个新的当前条目（引用块$C$和$D$）来更新（图3）。现在，块$A$和$B$已失效，块$C$和$D$处于活动状态。

Now let us consider an exact match query in the multiversion B-tree of Fig. 3. A record with key 25 is requested in version 5. First, the root of version 5 is accessed; in our example this is block $R$ . We consider only the entries in the root that belong to version 5 . Among these entries we choose the one whose separator key is the greatest key lower than the search key 25 and follow the corresponding reference to the next block. In our example, the search is directed to block $A$ . Eventually,the desired entry $< {25},1,7 >$ is found in block $A$ .

现在让我们考虑图3的多版本B树中的精确匹配查询。在版本5中请求键为25的记录。首先，访问版本5的根；在我们的示例中，这是块$R$。我们只考虑根中属于版本5的条目。在这些条目中，我们选择分隔键是小于搜索键25的最大键的条目，并跟随相应的引用到下一个块。在我们的示例中，搜索指向块$A$。最终，在块$A$中找到所需的条目$< {25},1,7 >$。

As mentioned before, our multiversion B-tree is not a tree, but a directed acyclic graph. In general, several root blocks may exist. This and the effect of structural changes in root blocks is illustrated in Figs. 4-6.

如前所述，我们的多版本B树不是树，而是有向无环图。一般来说，可能存在多个根块。根块中的结构变化的影响如图4 - 6所示。

<!-- Media -->

<!-- figureText: R1 R2 <10,25,*,R3> <40,25,*,R4> R3 R4 <10,18,*,A> <40,21,*,D> <25,18,*,B> <55,25,*,G> <30,21,*,C> <70,14,*,F> <10,18,*,A> <25,18,*,B> <30,21,*,C> <40,21,*,D> <55,14,25,E> <70,14,*,F> <55,25,*,G> -->

<img src="https://cdn.noedgeai.com/0195c902-d752-7cdd-b0a3-cd71d4e4465a_4.jpg?x=115&y=92&w=647&h=423&r=0"/>

Fig. 5. Key split after strong version overflow of root block ${R1}$

图5. 根块${R1}$强版本溢出后的键拆分

<!-- figureText: R2 R5 <10,32,*,I> <40,21,*,D> <55,25,*,G> <70,14,*,F> <10,25,32,R3> <40,25,32,R4> R3 R4 <10,18,29,A> <40,21,*,D> <25,18,29,B> <55,25,*,G> <70,14,*,F> <10,29,32,H> <10,32.*,I> -->

<img src="https://cdn.noedgeai.com/0195c902-d752-7cdd-b0a3-cd71d4e4465a_4.jpg?x=117&y=586&w=697&h=418&r=0"/>

Fig. 6. Weak version underflow of root block ${R2}$

图6. 根块${R2}$的弱版本下溢

<!-- Media -->

Figure 4 shows an overfull root block ${R1}$ and the two new roots ${R1},{R2}$ resulting from version split of block ${R1}$ . Block ${R2}$ is the root of the current version,version 18, whereas block ${R1}$ is the root of versions 1-17. References to roots ${R1}$ and ${R2}$ can be stored in an appropriate data structure, supporting access to the root blocks over versions.

图4显示了一个满溢的根块${R1}$以及由块${R1}$的版本拆分产生的两个新根${R1},{R2}$。块${R2}$是当前版本（版本18）的根，而块${R1}$是版本1 - 17的根。对根${R1}$和${R2}$的引用可以存储在适当的数据结构中，以支持跨版本访问根块。

Figure 5 illustrates the case that after the version split a strong version overflow occurs and a key split becomes necessary. In this case,a new root block(R2)is allocated, which stores entries referring to the two blocks ${R3}$ and ${R4}$ resulting from key split of the copy of root ${R1}$ . By that,the height of the subtree valid for the current version, version 25, has grown.

图5说明了在版本拆分后发生强版本溢出并且需要进行键拆分的情况。在这种情况下，分配一个新的根块（R2），它存储引用由根${R1}$的副本进行键拆分产生的两个块${R3}$和${R4}$的条目。这样，当前版本（版本25）有效的子树的高度增加了。

Figure 6 shows the shrinking of a subtree. By several data block merges,the number of current entries in ${R3}$ has shrunk, a weak version underflow occurred. To handle this underflow,block copies of ${R3}$ and ${R4}$ are created and merged into a block ${R5}$ . Since this causes a weak version underflow of block ${R2},{R5}$ becomes the new root block valid for the current version.

图6显示了子树的收缩。通过多次数据块合并，${R3}$中的当前条目数量减少，发生了弱版本下溢。为了处理这种下溢，创建${R3}$和${R4}$的块副本并合并到一个块${R5}$中。由于这导致块${R2},{R5}$的弱版本下溢，${R2},{R5}$成为当前版本有效的新根块。

### 2.3 The multiversion operations in detail

### 2.3 详细的多版本操作

To make these restructuring operations more precise, let us now present the main points in a semi-formal algorithmic notation. In order to present the main idea without obstructing irrelevant details, we assume that an exact-match query in the single-version structure returns a block in which the searched item is stored if it is present in the structure. For the same reason, we ignore the treatment of the end of the recursion in our operations, when a change propagates up to the root of the tree.

为了更精确地描述这些重构操作，让我们现在用半形式化的算法表示法来阐述要点。为了在不被无关细节干扰的情况下呈现主要思想，我们假设单版本结构中的精确匹配查询会返回一个块（block），如果被搜索的项存在于该结构中，那么它就存储在这个块里。出于同样的原因，当一个更改向上传播到树的根节点时，我们在操作中忽略对递归结束情况的处理。

To insert a data item, we proceed as follows:

要插入一个数据项，我们按以下步骤进行：

insert key $k$ ,current version $i$ ,information info :

插入键 $k$ ，当前版本 $i$ ，信息 info ：

\{assume $k$ is not yet present $\}$

{假设 $k$ 尚未出现在 $\}$ 中

exact-match query for $k$ in version $i$ leads to block $A$ ; blockinsert $< k,i, *$ ,info > into $A$ .

在版本 $i$ 中对 $k$ 进行精确匹配查询，会定位到块 $A$ ；将 < $< k,i, *$ , info > 插入到块 $A$ 中。

Here, blockinsert is defined as follows:

这里，块插入（blockinsert）定义如下：

blockinsert entry $e$ into block $A$ :

将条目 $e$ 插入到块 $A$ 中：

enter $e$ into $A$ ;

将 $e$ 插入到 $A$ 中；

\{this may momentarily lead to a block overflow in $A$ ,

{从概念上讲，这可能会暂时导致 $A$ 中的块溢出，

conceptually; such an overflow is eliminated immediately\}

但这种溢出会立即被消除}

if block overflow of $A$ then

如果 $A$ 发生块溢出，那么

version split: copy current entries of $A$ into a new block $B$ ;

版本分裂（version split）：将 $A$ 的当前条目复制到一个新块 $B$ 中；

blockinsert entry referencing $B$ into father of $A$ ;

将引用 $B$ 的条目插入到 $A$ 的父节点中；

if strong version underflow of $B$ then merge $B$

如果 $B$ 发生强版本下溢（strong version underflow），则合并 $B$

elsif strong version overflow of $B$ then treat strong version overflow of $B$ .

否则，如果 $B$ 发生强版本上溢（strong version overflow），则处理 $B$ 的强版本上溢。

Note that after a version split, the deletion version stored in the father entry referring to the dead block must be adjusted to represent the version of the version split, in order to guide subsequent searches correctly.

请注意，在版本分裂之后，存储在指向已删除块的父条目里的删除版本必须调整为表示版本分裂的版本，以便正确引导后续搜索。

Merging a block makes use of the fact that a suitable sibling can always be found in the access structure:

合并一个块利用了这样一个事实：在访问结构中总能找到合适的兄弟块：

merge block $B$ :

合并块 $B$ ：

identify a sibling $D$ of $B$ to be merged;

确定要与 $B$ 合并的兄弟块 $D$ ；

version split: copy current entries of $D$ into a new

版本拆分：将 $D$ 的当前条目复制到一个新的

block $E$ ; unite $B$ and $E$ into $B$ and discard $E$ ;

块 $E$ 中；将 $B$ 和 $E$ 合并为 $B$ 并丢弃 $E$ ；

if strong version overflow of $B$ then

如果 $B$ 出现强版本溢出，那么

treat strong version overflow of $B$

处理 $B$ 的强版本溢出

\{no weak version underflow possible in father of $B$ \} else

{ $B$ 的父块不可能出现弱版本下溢 } 否则

adapt router to $B$ in father of $B$ ;

调整 $B$ 的父块中的路由器以适应 $B$ ；

check weak version underflow of father of $B$ .

检查 $B$ 的父块是否存在弱版本下溢。

Essentially, a strong version overflow is treated by a key split of the entries according to their key or router values:

本质上，强版本溢出是通过根据条目键值或路由器值对条目进行键拆分来处理的：

treat strong version overflow of block $A$ :

处理块 $A$ 的强版本溢出：

key split: distribute entries of $A$ evenly among $A$ and $B$ ; adapt router to $A$ in father of $A$ ;

键拆分：将 $A$ 的条目均匀分配到 $A$ 和 $B$ 中；调整 $A$ 的父块中的路由器以适应 $A$ ；

blockinsert entry referencing $B$ into father of $A$ .

将引用 $B$ 的条目插入到 $A$ 的父块中。

A weak version underflow leads to a version split and a merge:

弱版本下溢会导致版本拆分和合并：

check weak version underflow of block $A$ :

检查块 $A$ 是否存在弱版本下溢：

if weak version underflow of $A$ then

如果$A$的弱版本下溢，则

version split: copy current entries of $A$ into a new block $B$ ;

版本拆分：将$A$的当前条目复制到一个新块$B$中；

blockinsert entry referencing $B$ into father of $A$ ; merge $B$ .

块插入：将引用$B$的条目插入到$A$的父块中；合并$B$。

This completes the description of the insertion of an item into a block. To delete an item, we proceed as follows:

至此完成了将一个项插入到块中的描述。要删除一个项，我们按以下步骤进行：

delete key $k$ ,current version $i\{$ assume $k$ is present $\}$ : exact match query for $k$ in version $i$ leads to block $A$ ; blockdelete $k,i$ from $A$ .

删除键$k$，当前版本$i\{$ 假设$k$存在$\}$：在版本$i$中对$k$进行精确匹配查询，结果指向块$A$；从$A$中块删除$k,i$。

blockdelete key $k$ ,version $i$ from block $A$ :

从块 $A$ 中删除键 $k$，版本 $i$：

change entry $< k,{i}^{\prime }, *$ ,info > into $< k,{i}^{\prime },i$ ,info > in $A$ ;

将 $A$ 中的条目 $< k,{i}^{\prime }, *$，信息 > 更改为 $< k,{i}^{\prime },i$，信息 >；

check weak version underflow of $A$ .

检查 $A$ 的弱版本下溢情况。

This completes the more detailed presentation of update operations. Let us repeat that the multiversion structure defined in this way is not a tree, but a directed acyclic graph. In general, more than one root block may exist. Since the number of root blocks to be expected is very small, maintaining these blocks is not a major data organization problem; see Sect. 4 for a suggestion.

至此，我们完成了对更新操作更详细的介绍。让我们再次强调，以这种方式定义的多版本结构不是树，而是有向无环图。一般来说，可能存在多个根块。由于预期的根块数量非常少，维护这些块并不是一个主要的数据组织问题；具体建议见第4节。

In the next section, we show in an analysis that the basic operations actually do lead to the desired behavior.

在下一节中，我们将通过分析表明，基本操作实际上确实能实现预期的行为。

## 3 Efficiency analysis

## 3 效率分析

Recall that a block is live if it was not copied up to the current version,dead otherwise. $N$ is the number of update operations performed on the data structure from the beginning up to the current version, ${m}_{i}$ is the number of data items present in version $i$ .

回顾一下，如果一个块在当前版本之前未被复制，则该块是活跃的；否则，该块是不活跃的。$N$ 是从开始到当前版本对数据结构执行的更新操作的数量，${m}_{i}$ 是版本 $i$ 中存在的数据项的数量。

What are the restrictions for the choice of $k$ and $\varepsilon$ ? First, after a key split, the resulting blocks must fulfill the strong version condition. Before a key split on a block $A$ is performed, $A$ contains at least $\left( {k - \varepsilon }\right)  \cdot  d + 1$ entries. After the key split operation that distributes the entries of $A$ among two blocks,both blocks must contain at least $\left( {1 + \varepsilon }\right)  \cdot  d$ entries. Therefore, the following inequality must hold:

$k$ 和 $\varepsilon$ 的选择有哪些限制呢？首先，在进行键分裂后，生成的块必须满足强版本条件。在对块 $A$ 进行键分裂之前，$A$ 至少包含 $\left( {k - \varepsilon }\right)  \cdot  d + 1$ 个条目。在将 $A$ 的条目分配到两个块的键分裂操作之后，两个块都必须至少包含 $\left( {1 + \varepsilon }\right)  \cdot  d$ 个条目。因此，必须满足以下不等式：

$$
\left( {k - \varepsilon }\right)  \cdot  d + 1 \geq  \frac{1}{\alpha } \cdot  \left( {1 + \varepsilon }\right)  \cdot  d \tag{1}
$$

$$
\text{or,equivalently,}\widehat{k} \geq  \frac{1}{\alpha } + \left( {1 + \frac{1}{\alpha }}\right)  \cdot  \varepsilon  - \frac{1}{d}
$$

Here, $\alpha$ depends on the underlying access structure. It denotes the constant fraction of data entries that are guaranteed to be in a new node. For example, $\alpha  = {0.5}$ is fulfilled for B-trees,i.e. inequality 1 is equivalent to $k \geq  2 + 3 \cdot  \varepsilon  - \frac{1}{d}$ .

这里，$\alpha$ 取决于底层的访问结构。它表示保证存在于新节点中的数据条目的恒定比例。例如，对于B树，$\alpha  = {0.5}$ 成立，即不等式1等价于 $k \geq  2 + 3 \cdot  \varepsilon  - \frac{1}{d}$。

Second, no strong version underflow is allowed for a block $A$ resulting from a merge operation. Before a merge operation is performed,together there are at least $2 \cdot  d -$ 1 current entries in the blocks which have to be merged. Therefore we have:

其次，对于合并操作产生的块 $A$，不允许出现强版本下溢。在执行合并操作之前，需要合并的块中总共至少有 $2 \cdot  d -$ + 1 个当前条目。因此，我们有：

$$
2 \cdot  d - 1 \geq  \left( {1 + \varepsilon }\right)  \cdot  d \tag{2}
$$

or,equivalently, $\varepsilon  \leq  1 - \frac{1}{d}$

或者，等价地，$\varepsilon  \leq  1 - \frac{1}{d}$

### 3.1 Run-time analysis

### 3.1 运行时分析

As introduced before, for our multiversion B-tree we have separated the concerns of identifying the root block of the requested version and querying the requested version. For the following analysis we assume that, supported from the application context, the appropriate root block is given.

如前所述，对于我们的多版本B树，我们将识别请求版本的根块和查询请求版本这两个问题分开处理。在接下来的分析中，我们假设在应用上下文的支持下，已经给出了合适的根块。

Recall that our multiversion structures are based on leaf-oriented balanced-access structures. The data blocks are on level 0, the directory blocks are on level 1,2,... Then the number of block accesses for searching a data item $x$ in version $i$ is at most $\left\lceil  {{\log }_{d}{m}_{i}}\right\rceil$ ,because each directory block on the path from the root of version $i$ to the leaf where $x$ is stored has at least $d$ references of $i$ . Given direct access to the root of the version in question, we conclude:

回顾一下，我们的多版本结构基于面向叶子的平衡访问结构。数据块位于第0层，目录块位于第1、2……层。那么，在版本 $i$ 中搜索数据项 $x$ 时，块访问的数量最多为 $\left\lceil  {{\log }_{d}{m}_{i}}\right\rceil$，因为从版本 $i$ 的根到存储 $x$ 的叶子的路径上的每个目录块至少有 $d$ 个版本 $i$ 的引用。假设可以直接访问所讨论版本的根，我们得出以下结论：

Theorem 1 The number of block accesses for searching a data item in version $i$ is $\left\lceil  {{\log }_{d}{m}_{i}}\right\rceil$ in the worst case.

定理1 在最坏情况下，在版本 $i$ 中搜索数据项的块访问数量为 $\left\lceil  {{\log }_{d}{m}_{i}}\right\rceil$。

The arguments above can be extended to range queries that are answered by traversing the corresponding umbrella-like part of a subtree of the tree for the queried version:

上述论点可以扩展到范围查询，范围查询通过遍历所查询版本的树的子树的相应伞状部分来回答：

Theorem 2 The number of block accesses for answering a range query in version $i$ that returns $r$ data items is $O\left( {\left\lceil  {{\log }_{d}{m}_{i}}\right\rceil   + r/d}\right)$ in the worst case.

定理2 在版本$i$中回答一个范围查询（返回$r$个数据项）时，最坏情况下的块访问次数为$O\left( {\left\lceil  {{\log }_{d}{m}_{i}}\right\rceil   + r/d}\right)$。

The $\left( {i + 1}\right)$ -th update operation first performs an exact match query in version $i$ and then modifies at least one data block $A$ . If $A$ violates the weak version condition,up to three other data blocks have to be created or modified. In this case, the parent of $A$ - say ${A}^{\prime }$ - has to be modified. Again,this can lead to a violation of the weak version condition of ${A}^{\prime }$ . In the worst case, this situation occurs on each directory level up to the root of version $i$ . On each directory level,at most five directory blocks have to be accessed, modified or created. Therefore we have:

第$\left( {i + 1}\right)$次更新操作首先在版本$i$中执行一次精确匹配查询，然后修改至少一个数据块$A$。如果$A$违反了弱版本条件，则最多需要创建或修改另外三个数据块。在这种情况下，$A$的父节点（设为${A}^{\prime }$）必须被修改。同样，这可能会导致${A}^{\prime }$的弱版本条件被违反。在最坏情况下，这种情况会在版本$i$的根节点之前的每个目录级别上发生。在每个目录级别上，最多需要访问、修改或创建五个目录块。因此，我们有：

Theorem 3 The number of block accesses and modifications for the $\left( {i + 1}\right)$ -th update operation is $5 \cdot  \left\lceil  {{\log }_{d}{m}_{i}}\right\rceil$ in the worst case.

定理3 第$\left( {i + 1}\right)$次更新操作在最坏情况下的块访问和修改次数为$5 \cdot  \left\lceil  {{\log }_{d}{m}_{i}}\right\rceil$。

### 3.2 Space analysis

### 3.2 空间分析

We analyze the worst-case space utilization over the sequence of the $N$ update operations. The crucial factor in the analysis is the fact that a version split, if necessary followed by a merge or a key split, leads to new blocks which fulfill the strong version condition. Therefore we need a certain number of update operations on these blocks before the next underflow or overflow situation on these blocks can occur. To be more precise, we consider the utilization of data blocks and of directory blocks separately.

我们分析在$N$次更新操作序列中的最坏情况空间利用率。分析中的关键因素是，如果必要的话，版本分裂（随后可能进行合并或键分裂）会产生满足强版本条件的新块。因此，在这些块上发生下一次下溢或上溢情况之前，我们需要对这些块进行一定数量的更新操作。更准确地说，我们分别考虑数据块和目录块的利用率。

For data blocks, one update operation can lead to at most one overflow or underflow situation. We distinguish four types of situations:

对于数据块，一次更新操作最多可能导致一次上溢或下溢情况。我们区分四种情况：

- Version split only: One block $A$ becomes dead and one new live block $B$ is created. $A$ was the first data block in the data structure or has fulfilled initially - after its creation - the strong version condition. If it becomes overfull,at least $\varepsilon  \cdot  d + 1$ operations must have taken place on $A$ since its creation. So the amortized space cost for each of these operations is at most $\begin{matrix} k \cdot  d \\  \varepsilon  \cdot  d + 1 \end{matrix}$ .

- 仅版本分裂：一个块$A$变为无效，创建一个新的有效块$B$。$A$是数据结构中的第一个数据块，或者在创建后最初满足强版本条件。如果它变得过满，自创建以来，$A$上至少必须进行了$\varepsilon  \cdot  d + 1$次操作。因此，这些操作中每次操作的摊还空间成本最多为$\begin{matrix} k \cdot  d \\  \varepsilon  \cdot  d + 1 \end{matrix}$。

- Version split and key split: One block $A$ becomes dead and two new live blocks ${B1}$ and ${B2}$ are created. Again, at least $\varepsilon  \cdot  d + 1$ operations must have taken place on $A$ and therefore the amortized space cost for each of these operations is at most ${}_{\varepsilon  \cdot  d + 1}^{2 \cdot  k \cdot  d}$ .

- 版本分裂和键分裂：一个块$A$变为无效，创建两个新的有效块${B1}$和${B2}$。同样，$A$上至少必须进行了$\varepsilon  \cdot  d + 1$次操作，因此这些操作中每次操作的摊还空间成本最多为${}_{\varepsilon  \cdot  d + 1}^{2 \cdot  k \cdot  d}$。

- Version split and merge without key split: Two blocks ${A1}$ and ${A2}$ become dead and one new live block $B$ is created. On ${A1}$ or ${A2}$ at least $\varepsilon  \cdot  d + 1$ operations must have taken place. Thus, the amortized space cost for each of these operations is at most $\begin{matrix} k \cdot  d \\  \varepsilon  \cdot  d + 1 \end{matrix}$ .

- 无键分裂的版本分裂和合并：两个块${A1}$和${A2}$变为无效，创建一个新的有效块$B$。在${A1}$或${A2}$上至少必须进行了$\varepsilon  \cdot  d + 1$次操作。因此，这些操作中每次操作的摊还空间成本最多为$\begin{matrix} k \cdot  d \\  \varepsilon  \cdot  d + 1 \end{matrix}$。

- Version split and merge with key split: Two blocks ${A1}$ and ${A2}$ become dead and two new live blocks ${B1}$ and ${B2}$ are created. Again,on ${A1}$ or ${A2}$ at least $\varepsilon  \cdot  d + 1$ operations must have taken place. The amortized space cost for each of these operations is at most ${}_{\varepsilon  \cdot  d + 1}^{2 \cdot  k \cdot  d}$ .

- 有键分裂的版本分裂和合并：两个块${A1}$和${A2}$变为无效，创建两个新的有效块${B1}$和${B2}$。同样，在${A1}$或${A2}$上至少必须进行了$\varepsilon  \cdot  d + 1$次操作。这些操作中每次操作的摊还空间成本最多为${}_{\varepsilon  \cdot  d + 1}^{2 \cdot  k \cdot  d}$。

In all cases the amortized data block space cost per update operation ${S}_{\text{dat }}$ is at most

在所有情况下，每次更新操作${S}_{\text{dat }}$的摊还数据块空间成本最多为

$$
\begin{array}{l} 2 \cdot  k \cdot  d \\  \varepsilon  \cdot  d + 1 \end{array} < \begin{matrix} 2 \cdot  k \\  \varepsilon  \end{matrix} = O\left( 1\right)  \tag{3}
$$

For directory blocks, one update operation can lead to at most one block overflow or version underflow situation on each directory level up to the directory level of the root in the current version. Let $L$ denote the maximum level that occurs during the $N$ operations. To look precisely at the different underflow and overflow situations, we distinguish between directory blocks that are roots during their lifetime and inner blocks.

对于目录块，一次更新操作在当前版本中，最多会在每个目录层级（直至根目录层级）上导致一个块溢出或版本下溢情况。令 $L$ 表示在 $N$ 次操作期间出现的最大层级。为了精确研究不同的下溢和溢出情况，我们区分在其生命周期内作为根的目录块和内部块。

Let ${A}^{l}$ denote an inner directory block of level $l$ . We call a reference in ${A}^{l}$ dead,if it is a reference to a dead block, live otherwise. The following situations can cause a weak version underflow or a block overflow of ${A}^{l}$ :

令 ${A}^{l}$ 表示层级为 $l$ 的内部目录块。如果 ${A}^{l}$ 中的一个引用指向一个已死亡的块，则称该引用为死引用，否则为活引用。以下情况可能导致 ${A}^{l}$ 出现弱版本下溢或块溢出：

- One reference in ${A}^{l}$ becomes dead and one new reference has to be inserted into ${A}^{l}$ . This can cause a block overflow with the creation of two new directory blocks.

- ${A}^{l}$ 中的一个引用变为死引用，并且必须向 ${A}^{l}$ 中插入一个新引用。这可能会在创建两个新目录块时导致块溢出。

- One reference in ${A}^{l}$ becomes dead and two new references have to be inserted into ${A}^{l}$ . This can cause a block overflow with the creation of two new directory blocks.

- ${A}^{l}$ 中的一个引用变为死引用，并且必须向 ${A}^{l}$ 中插入两个新引用。这可能会在创建两个新目录块时导致块溢出。

- Two references in ${A}^{l}$ become dead and one new reference has to be inserted into ${A}^{l}$ . This can cause a weak version underflow or a block overflow. In the case of a weak version underflow,a sibling of ${A}^{l}$ also becomes dead, and up to two new directory blocks are created.

- ${A}^{l}$ 中的两个引用变为死引用，并且必须向 ${A}^{l}$ 中插入一个新引用。这可能会导致弱版本下溢或块溢出。在出现弱版本下溢的情况下，${A}^{l}$ 的一个兄弟块也会变为死引用，并且最多会创建两个新目录块。

- Two references in ${A}^{l}$ become dead and two new references have to be inserted into ${A}^{l}$ . This can cause a block overflow with the creation of two new directory blocks.

- ${A}^{l}$ 中的两个引用变为死引用，并且必须向 ${A}^{l}$ 中插入两个新引用。这可能会在创建两个新目录块时导致块溢出。

Note that if a directory block is the root of the data structure in version $i$ ,a weak version underflow does not lead to a new copy of the block. A block overflow of a root block is treated in the same manner as a block overflow of an inner block.

请注意，如果一个目录块是版本 $i$ 中数据结构的根，弱版本下溢不会导致该块的新副本产生。根块的块溢出处理方式与内部块的块溢出处理方式相同。

We explain the amortized space cost per operation for the first case. The extension to the other cases and the root blocks is straightforward and yields the same result. ${A}^{l}$ is the only live parent for the live blocks referenced from ${A}^{l}$ and has initially fulfilled the strong version condition. Therefore, in the subtree of ${A}^{l}$ on level $l - 1$ at least $\varepsilon  \cdot  d + 1$ new blocks have been created between the creation of ${A}^{l}$ and the block overflow of ${A}^{l}$ . Hence,at least $\left( {\varepsilon  \cdot  d + 1}\right)  \cdot  k \cdot  d$ space was used. Let us assume that the amortized space cost per update on level $l - 1$ is at most ${C}^{l - 1}$ . Then it follows that at least ${C}^{\left( {\varepsilon  \cdot  d + 1}\right)  \cdot  k \cdot  d}$ operations have taken place in the subtree of ${A}^{l}$ between the creation of ${A}^{l}$ and its block overflow. The space cost for the version split of ${A}^{l}$ and the subsequent key split is $2 \cdot  k \cdot  d$ . Therefore,the amortized space cost per update on level $l$ is at most

我们解释第一种情况每次操作的平摊空间成本。扩展到其他情况和根块是直接的，并且会得到相同的结果。${A}^{l}$ 是从 ${A}^{l}$ 引用的活动块的唯一活动父块，并且最初满足强版本条件。因此，在 ${A}^{l}$ 的层级为 $l - 1$ 的子树中，在 ${A}^{l}$ 创建和 ${A}^{l}$ 块溢出之间至少创建了 $\varepsilon  \cdot  d + 1$ 个新块。因此，至少使用了 $\left( {\varepsilon  \cdot  d + 1}\right)  \cdot  k \cdot  d$ 的空间。假设层级 $l - 1$ 上每次更新的平摊空间成本最多为 ${C}^{l - 1}$。那么可以得出，在 ${A}^{l}$ 创建和其块溢出之间，${A}^{l}$ 的子树中至少进行了 ${C}^{\left( {\varepsilon  \cdot  d + 1}\right)  \cdot  k \cdot  d}$ 次操作。${A}^{l}$ 的版本分裂和后续键分裂的空间成本为 $2 \cdot  k \cdot  d$。因此，层级 $l$ 上每次更新的平摊空间成本最多为

$$
{C}^{l} < 2 \cdot  k \cdot  d \cdot  \begin{matrix} {C}^{l - 1} \\  \left( {\varepsilon  \cdot  d + 1}\right)  \cdot  k \cdot  d < \frac{2}{\varepsilon  \cdot  d} \cdot  {C}^{l - 1} \end{matrix} \tag{4}
$$

for $1 \leq  l \leq  L$ . With ${C}^{0} \mathrel{\text{:=}} {S}_{\text{data }}$ ,i.e. ${C}^{0} = \frac{2 \cdot  k}{\varepsilon }$ (from inequality 3 ), we can rewrite inequality 4 :

对于 $1 \leq  l \leq  L$。使用 ${C}^{0} \mathrel{\text{:=}} {S}_{\text{data }}$，即 ${C}^{0} = \frac{2 \cdot  k}{\varepsilon }$（由不等式 3 得出），我们可以重写不等式 4：

$$
{C}^{l} < {\left( \begin{matrix} 2 \\  \varepsilon  \cdot  d \end{matrix}\right) }^{l} \cdot  {C}^{0} = {\left( \begin{matrix} 2 \\  \varepsilon  \cdot  d \end{matrix}\right) }^{l} \cdot  \begin{matrix} 2 \cdot  k \\  \varepsilon  \end{matrix} \tag{5}
$$

for $1 \leq  l \leq  L$ .

对于 $1 \leq  l \leq  L$。

Therefore, the total amortized directory block space cost per operation ${S}_{\text{dir }}$ is at most:

因此，每次操作 ${S}_{\text{dir }}$ 的总平摊目录块空间成本最多为：

$$
{S}_{\text{dir }} < \mathop{\sum }\limits_{{l = 1}}^{L}{C}^{l} = \frac{2 \cdot  k}{\varepsilon } \cdot  \mathop{\sum }\limits_{{l = 1}}^{L}{\left( \begin{matrix} 2 \\  \varepsilon  \cdot  d \end{matrix}\right) }^{l} \tag{6}
$$

For $d > \frac{2}{\varepsilon }$ ,which can easily be satisfied in all practically relevant circumstances, we get:

对于 $d > \frac{2}{\varepsilon }$，在所有实际相关的情况下都很容易满足，我们得到：

$$
{S}_{\text{dir }} < \frac{2 \cdot  k}{\varepsilon } \cdot  \mathop{\sum }\limits_{{l = 1}}^{\infty }{\left( \begin{matrix} 2 \\  \varepsilon  \cdot  d \end{matrix}\right) }^{l} = O\left( 1\right)  \tag{7}
$$

In summary, from inequalities 3 and 7 we can conclude:

综上所述，从不等式 3 和 7 我们可以得出结论：

Theorem 4 The worst-case amortized space cost per update operation $S = {S}_{\text{dat }} + {S}_{\text{dir }}$ is $O\left( 1\right)$ if $d \geq  \frac{2}{\varepsilon }$ .

定理4 如果$d \geq  \frac{2}{\varepsilon }$ ，则每次更新操作$S = {S}_{\text{dat }} + {S}_{\text{dir }}$ 的最坏情况均摊空间成本为$O\left( 1\right)$ 。

## In total, we get:

## 总体而言，我们得到：

Theorem 5 The multiversion B-tree constructed in the described way from the single-version B-tree is asymptotically optimal in the worst case in time and space for all considered operations.

定理5 以所述方式从单版本B树构建的多版本B树，对于所有考虑的操作，在最坏情况下的时间和空间复杂度上是渐近最优的。

The analysis shows that for a given block capacity $b$ it is useful for the time complexity to choose $d$ large and $k$ small. To guarantee good space utilization it is useful to choose $\varepsilon$ maximum,that is equal to $1 - \frac{1}{d}$ ,and $k$ as small as possible without violating inequality 1 . Choosing $\varepsilon  = 1 - \frac{1}{d}$ gives bounds for the strong version condition of $2 \cdot  d - 1$ and $\left( {k - 1}\right)  \cdot  d + 1$ . For instance,for block capacity $b = {25}$ we get $k = 5,d = 5$ ,and $\varepsilon  = {0.8}$ . In the worst case,this implies that we have ${11.5}\left( {{}^{2 \cdot  k} - 1}\right)$ redundant records for each key on average. Because this is quite a high number, we implemented the multiversion B-tree and ran a number of experiments with the above parameters and $N = {100000}$ update operations. It turned out that in all experiments, we had between 1.31 and 1.70 redundant records for each key on average. Hence, our worst-case bounds are extremely pessimistic and do not imply high constant costs on average.

分析表明，对于给定的块容量$b$ ，为了时间复杂度考虑，选择较大的$d$ 和较小的$k$ 是有益的。为了保证良好的空间利用率，选择最大的$\varepsilon$ （即等于$1 - \frac{1}{d}$ ），并在不违反不等式1的前提下尽可能选择较小的$k$ 是有益的。选择$\varepsilon  = 1 - \frac{1}{d}$ 为强版本条件给出了$2 \cdot  d - 1$ 和$\left( {k - 1}\right)  \cdot  d + 1$ 的边界。例如，对于块容量$b = {25}$ ，我们得到$k = 5,d = 5$ 和$\varepsilon  = {0.8}$ 。在最坏情况下，这意味着平均每个键有${11.5}\left( {{}^{2 \cdot  k} - 1}\right)$ 条冗余记录。由于这个数字相当大，我们实现了多版本B树，并使用上述参数和$N = {100000}$ 次更新操作进行了一系列实验。结果表明，在所有实验中，平均每个键有1.31到1.70条冗余记录。因此，我们的最坏情况边界非常悲观，并不意味着平均有很高的常数成本。

## 4 Thoughts around the main result

## 4 关于主要结果的思考

In the following, we present some of the thoughts around the main result that may be interesting or important in practice. First, we discuss the organization of the access to the requested B-tree root; this also solves the problem of time-oriented access, where query points in time differ from version creation times, and of maintaining user-defined versions. Second, we show how to efficiently remove the oldest versions, in order to save storage space. Our thoughts are intended to demonstrate the high potential of adapting the multiversion B-tree to different settings and different requirements. The given list of modifications and extensions is not meant to be exhaustive; additions to this list should be performed as needed.

接下来，我们将介绍一些围绕主要结果的思考，这些思考在实践中可能很有趣或很重要。首先，我们讨论对请求的B树根节点的访问组织；这也解决了面向时间的访问问题（查询时间点与版本创建时间不同）以及维护用户定义版本的问题。其次，我们展示如何有效地删除最旧的版本，以节省存储空间。我们的思考旨在证明将多版本B树应用于不同场景和不同需求的巨大潜力。给出的修改和扩展列表并非详尽无遗；应根据需要对该列表进行补充。

### 4.1 Access to the requested version

### 4.1 对请求版本的访问

Our presentation of the multiversion B-tree so far assumes that access to the root of a version is taken care of in the context of the application. If this is not the case, a search structure may be used to guide the access. As an example, a B-tree maintaining the version intervals of the multiversion B-tree root nodes in its leaves serves this purpose. Even in its most direct application, this access structure to the roots of the multiversion B-tree (we call it root*) allows access to a root as well as insertion of a new root into root* in time $O\left( {{\log }_{b}p}\right)$ ,where $p$ is the number of roots being maintained. The space efficiency of such a B-tree is obviously $O\left( p\right)$ . Since $p$ is less than $\lceil N/d\rceil$ ,the storage cost of root* is $O\left( {N/b}\right)$ and the search for a key in a multiversion query can be realized in time $O\left( {{\log }_{b}N}\right)$ in total,including the search for the appropriate version.

到目前为止，我们对多版本B树的介绍假设对某个版本根节点的访问是在应用程序的上下文中处理的。如果不是这种情况，可以使用一种搜索结构来引导访问。例如，一个在其叶子节点中维护多版本B树根节点版本区间的B树就可以实现这一目的。即使在其最直接的应用中，这种对多版本B树根节点的访问结构（我们称之为root*）也允许在时间$O\left( {{\log }_{b}p}\right)$ 内访问根节点以及将新的根节点插入到root*中，其中$p$ 是所维护的根节点数量。显然，这样一个B树的空间效率为$O\left( p\right)$ 。由于$p$ 小于$\lceil N/d\rceil$ ，root*的存储成本为$O\left( {N/b}\right)$ ，并且在多版本查询中搜索一个键总共可以在时间$O\left( {{\log }_{b}N}\right)$ 内实现，包括搜索合适的版本。

In most cases, we expect that the number of roots is much less than $\lceil N/d\rceil$ . Consider,for example,the situation when the current version data set has been created by a sequence of insertions only, beginning at an empty structure. Then, the left path of the current B-tree contains all the roots of the multiversion B-tree. Therefore, the number of roots is only $O\left( {{\log }_{b}N}\right)$ which is considerably less than the worst-case results of the general case.

在大多数情况下，我们预计根节点的数量远小于$\lceil N/d\rceil$ 。例如，考虑当前版本数据集是从一个空结构开始仅通过一系列插入操作创建的情况。那么，当前B树的左路径包含了多版本B树的所有根节点。因此，根节点的数量仅为$O\left( {{\log }_{b}N}\right)$ ，这远小于一般情况下的最坏情况结果。

Furthermore, root* can be used to support time-oriented queries. If our setup changes from versions to time, such that each key has an insertion time stamp and a deletion time stamp, root* supports queries for any point in time (not necessarily coinciding with some insertion or deletion time) in the standard B-tree fashion.

此外，root* 可用于支持面向时间的查询。如果我们的设置从版本改为时间，使得每个键都有插入时间戳和删除时间戳，那么 root* 就能以标准 B 树的方式支持对任意时间点（不一定与某些插入或删除时间重合）的查询。

Moreover, root* can be tuned to achieve even higher performance by observing that a new multiversion root can only be added at the high end of the current version or time spectrum. Therefore, a split of a node of root* can be made totally unbalanced: the node of the lower key range is full, whereas the node of the higher key range contains just one key, namely the new one. As a consequence, all nodes in root* are full, except those on the rightmost path. This straightforward approach is somewhat reminiscent of the append-only tree (Segev and Gunadhi 1989), where an entry pointer to the rightmost node for each level of the tree is maintained in addition, in order to favor queries to the recent past. Then, access to the records of the current (and recent past) version can be organized more efficiently, leading to a path length of $O\left( {{\log }_{b}{m}_{N}}\right)$ . Therefore,the worst-case time bound for range queries to the current version for the MVBT tree is $O\left( {{\log }_{b}{m}_{N} + r/b}\right)$ . An update costs time $O\left( {{\log }_{b}N}\right)$ in the worst case,because a change may propagate up to the root of the root ${}^{ * }$ B-tree. Amortized over a sequence of updates, however, the worst-case cost of a single update is only $O\left( {{\log }_{b}{m}_{N}}\right)$ ,for the following reasons: First, in root*, the entry pointing to the current root is found in $O\left( 1\right)$ . Second,the record to perform the update in the current B-tree is found in $O\left( {{\log }_{b}{m}_{N}}\right)$ . Third,the remaining effort to perform the update has only constant amortized cost (Huddleston and Mehlhorn 1982). Overall, this proves our statement.

此外，通过观察发现新的多版本根节点只能添加到当前版本或时间范围的高端，root* 可以进行调优以实现更高的性能。因此，root* 节点的分裂可以完全不平衡：键范围较低的节点是满的，而键范围较高的节点只包含一个键，即新键。因此，除了最右侧路径上的节点外，root* 中的所有节点都是满的。这种直接的方法有点让人想起仅追加树（Segev 和 Gunadhi，1989 年），在该树中，为了便于对近期过去的查询，还会为树的每一层维护一个指向最右侧节点的条目指针。这样，对当前（和近期过去）版本记录的访问可以更高效地组织，从而使路径长度为 $O\left( {{\log }_{b}{m}_{N}}\right)$。因此，MVBT 树对当前版本进行范围查询的最坏情况时间界限是 $O\left( {{\log }_{b}{m}_{N} + r/b}\right)$。最坏情况下，一次更新操作需要 $O\left( {{\log }_{b}N}\right)$ 的时间，因为一次更改可能会向上传播到根 ${}^{ * }$ B 树的根节点。然而，在一系列更新操作中进行平摊后，单次更新的最坏情况成本仅为 $O\left( {{\log }_{b}{m}_{N}}\right)$，原因如下：首先，在 root* 中，指向当前根节点的条目可以在 $O\left( 1\right)$ 中找到。其次，在当前 B 树中执行更新操作的记录可以在 $O\left( {{\log }_{b}{m}_{N}}\right)$ 中找到。第三，执行更新操作的其余工作的平摊成本仅为常数（Huddleston 和 Mehlhorn，1982 年）。总体而言，这证明了我们的观点。

Other access structures may be plugged in to serve as root*. For instance, if a high locality of reference to nearby versions is required, a finger search tree may be the method of choice (Huddleston and Mehlhorn 1982). To summarize, root* has the potential to be tuned to the particular application.

其他访问结构也可以作为 root* 使用。例如，如果需要对相邻版本有较高的引用局部性，那么手指搜索树可能是首选方法（Huddleston 和 Mehlhorn，1982 年）。总之，root* 有潜力针对特定应用进行调优。

### 4.2 Purging old versions

### 4.2 清除旧版本

The operation of removing the oldest versions from disk, the so-called purge operation, is very important in multiversion access structures, because maintaining all versions forever may be too costly in terms of storage space. Under the assumption that old versions are accessed significantly less frequently than newer ones, the amount of secondary storage can be reduced substantially by moving old versions to tertiary storage (e.g. optical disks) or, whenever the application permits, by simply deleting them (e.g. in multiversion concurrency control).

从磁盘中移除最旧版本的操作，即所谓的清除操作，在多版本访问结构中非常重要，因为永久保留所有版本在存储空间方面的成本可能过高。假设旧版本的访问频率明显低于新版本，那么可以通过将旧版本迁移到三级存储（如光盘），或者在应用程序允许的情况下直接删除它们（如在多版本并发控制中），来大幅减少二级存储的使用量。

The deletion of versions older than a specified version $i$ can be supported easily in the multiversion B-tree. A straightforward approach would be to search for all blocks which have been split by a version split in a version less than or equal to $i$ . This search starts at the root blocks valid for version $i$ and older. Performing a depth-first search, all blocks fulfilling the above condition can immediately be deallocated. The disadvantage of this approach is that it may access many blocks for a few that can be purged. A more efficient approach accesses only blocks that must be purged: An additional data structure is used to keep track for each node of the most recent (i.e. newest) version for which this node is relevant in a query. Since this version is just the version before the version in which the node dies, this defines a linear order on the nodes; a simple first-in-first-out queue will therefore suffice to perform all operations efficiently. Whenever a node dies, a corresponding entry is added to the tail of the queue. Whenever the oldest versions before some version $i$ are to be deleted,triggered by the user or by some other mechanism such as concurrency control, the corresponding head entries of the queue are removed, as well as the corresponding multiversion B-tree nodes.

在多版本 B 树中，可以轻松支持删除早于指定版本 $i$ 的版本。一种直接的方法是搜索所有在小于或等于 $i$ 的版本中因版本分裂而分裂的块。此搜索从对版本 $i$ 及更早版本有效的根块开始。通过进行深度优先搜索，可以立即释放所有满足上述条件的块。这种方法的缺点是，为了清除少数几个块，可能会访问很多块。一种更高效的方法只访问必须清除的块：使用一个额外的数据结构来跟踪每个节点在查询中相关的最新（即最新的）版本。由于这个版本正好是节点失效版本之前的版本，这就定义了节点的线性顺序；因此，一个简单的先进先出队列就足以高效地执行所有操作。每当一个节点失效时，会在队列的尾部添加一个相应的条目。每当用户或其他机制（如并发控制）触发要删除某个版本 $i$ 之前的最旧版本时，会移除队列的相应头部条目以及相应的多版本 B 树节点。

Note that the removal of a node from a multiversion B-tree may leave the tree in an inconsistent state: there may be pointers in the tree that point to the node that is no longer present. Nevertheless, this inconsistency is not harmful, as long as no search for a deleted version (older than version $i$ ) initiates: A search may encounter a dangling pointer but will never follow it.

请注意，从多版本B树中移除一个节点可能会使树处于不一致的状态：树中可能存在指向已不存在节点的指针。不过，只要不发起对已删除版本（早于版本$i$ ）的搜索，这种不一致就不会造成危害：搜索可能会遇到悬空指针，但绝不会顺着它继续查找。

## 5 Related work

## 5 相关工作

A number of investigations on how to maintain multiversion data (historical data, time-dependent data) on external storage have been presented in the literature. Often, the goal of these investigations has been somewhat different from our goal in designing the multiversion B-tree. Nevertheless, some previous proposals pursue almost the same objective as we do, and others have been influential in setting the stage. To put our work into its proper perspective, we present a synopsis of relevant previous work in this section.

文献中已经提出了许多关于如何在外部存储中维护多版本数据（历史数据、与时间相关的数据）的研究。通常，这些研究的目标与我们设计多版本B树的目标有所不同。不过，一些先前的提议与我们追求的目标几乎相同，而另一些则为我们的工作奠定了基础。为了正确看待我们的工作，我们在本节中对相关的先前工作进行了概述。

Kolovson and Stonebraker (1989) discussed the problem of maintaining multiversion data using two external storage media, magnetic disk for recent data and WORM optical disk for historical versions. They proposed two approaches, both using the R-tree index (Guttman 1984), to organize data records: according to their key values in one, according to their lifespans in the other dimension. The approaches differ by the techniques of moving data and index blocks from magnetic disk to WORM disk, also called vacuuming. In the first approach, vacuuming is triggered in the following way. If the size of the index on magnetic disk reaches a given threshold, a vacuuming process moves a given fraction of the oldest (i.e. dead) data blocks to WORM disk and - recursively up the tree - those directory blocks that refer only to blocks already stored on WORM disk. The second approach maintains two R-trees, one completely on magnetic disk, the other with the upper levels on magnetic and all levels below on WORM disk. Again, if the size of the R-tree completely stored on magnetic disk reaches a threshold, all its blocks except the root are moved to WORM disk. Then, references to the blocks below the root level, now stored on WORM disk, are inserted into the corresponding level of the R-tree on magnetic disk. Updates are only performed on the R-tree that completely resides on magnetic disk, while queries may affect both R-trees. Both approaches presented by Kolov-son and Stonebraker (1989) support the same operations as the multiversion B-tree (MVBT). Additionally, queries over version intervals can be answered.

科洛夫森（Kolovson）和斯通布雷克（Stonebraker）（1989年）讨论了使用两种外部存储介质来维护多版本数据的问题，其中，近期数据存储在磁盘上，历史版本存储在一次写入多次读取（WORM）光盘上。他们提出了两种方法，均使用R树索引（古特曼（Guttman），1984年）来组织数据记录：一种方法是根据数据记录的键值进行组织，另一种方法是根据数据记录的生命周期进行组织。这两种方法的区别在于将数据和索引块从磁盘移动到WORM光盘的技术，这种技术也称为清理。在第一种方法中，清理操作按以下方式触发。如果磁盘上的索引大小达到给定阈值，清理过程会将一定比例的最旧（即已失效）的数据块移动到WORM光盘，并递归地将那些仅引用已存储在WORM光盘上的块的目录块也移动过去。第二种方法维护两棵R树，一棵完全存储在磁盘上，另一棵的上层存储在磁盘上，其余层存储在WORM光盘上。同样，如果完全存储在磁盘上的R树的大小达到阈值，除根节点外的所有块都会被移动到WORM光盘。然后，将指向现在存储在WORM光盘上的根节点以下块的引用插入到磁盘上R树的相应层中。更新操作仅在完全驻留在磁盘上的R树上执行，而查询操作可能会影响两棵R树。科洛夫森和斯通布雷克（1989年）提出的两种方法支持与多版本B树（MVBT）相同的操作。此外，还可以回答跨版本区间的查询。

In both approaches,the height of the R-trees is $\Theta \left( {{\log }_{b}N}\right)$ ; remember that $N$ is the total number of updates to the tree, and $b$ is the maximum number of entries in a tree node. Therefore,each insertion needs time $\Theta \left( {{\log }_{b}N}\right)$ ; this compares with amortized time $\Theta \left( {{\log }_{b}{m}_{N}}\right)$ in the MVBT,since access to the newest version is always immediate. Deletion must be implemented as modification of the corresponding R-tree entry. For that, the affected entry has to be searched in the tree before modification. Because of overlapping regions in the R-tree, the search for a record may necessitate a traversal of the whole index tree in the worst case. Therefore, deletion can be extremely expensive in the worst case; this compares with worst-case time $O\left( {{\log }_{b}{m}_{N}}\right)$ in the MVBT. The same arguments show that exact-match queries and range queries on a given version may access $\Theta \left( {N/b}\right)$ blocks in the worst case. This compares with a worst-case time for exact-match queries and range queries of $\Theta \left( {{\log }_{b}N}\right)$ and $\Theta \left( {{\log }_{b}N + r/b}\right)$ ,respectively. Note,however,that the goals of these approaches have been somewhat different from our goal of building a multiversion B-tree.

在这两种方法中，R树的高度为$\Theta \left( {{\log }_{b}N}\right)$ ；请记住，$N$ 是对树进行的更新操作的总数，$b$ 是树节点中的最大条目数。因此，每次插入操作需要$\Theta \left( {{\log }_{b}N}\right)$ 的时间；而在多版本B树（MVBT）中，插入操作的摊还时间为$\Theta \left( {{\log }_{b}{m}_{N}}\right)$ ，因为对最新版本的访问总是即时的。删除操作必须实现为对相应R树条目的修改。为此，在修改之前必须在树中搜索受影响的条目。由于R树中存在重叠区域，在最坏的情况下，搜索一条记录可能需要遍历整个索引树。因此，在最坏的情况下，删除操作可能会非常耗时；而在多版本B树（MVBT）中，删除操作的最坏情况时间复杂度为$O\left( {{\log }_{b}{m}_{N}}\right)$ 。同样的道理表明，在给定版本上进行精确匹配查询和范围查询时，在最坏的情况下可能会访问$\Theta \left( {N/b}\right)$ 个块。相比之下，在多版本B树（MVBT）中，精确匹配查询和范围查询的最坏情况时间复杂度分别为$\Theta \left( {{\log }_{b}N}\right)$ 和$\Theta \left( {{\log }_{b}N + r/b}\right)$ 。不过请注意，这些方法的目标与我们构建多版本B树的目标有所不同。

Because no data are replicated, the space efficiency of Kolovson and Stonebraker's approaches is perfect. However, especially for sets of records with lifespans of non-uniformly distributed lengths, Kolovson and Stonebraker observed a decreasing efficiency for the R-tree.

由于没有数据被复制，科洛夫森和斯通布雷克的方法的空间效率是完美的。然而，特别是对于生命周期长度分布不均匀的记录集，科洛夫森和斯通布雷克发现R树的效率会降低。

In order to achieve better query and update performance for such data distributions, Kolovson and Stonebraker have proposed segment R-trees (SR-trees; Kolovson and Stone-braker 1991), a hybrid of segment trees (Bentley 1977) and R-trees (Guttman 1984). Skeleton SR-trees operate with a preset data space partitioning, based on an assumption about the data distribution. In the performance evaluation presented by Kolovson and Stonebraker (1991), the SR-trees never outperformed R-trees in the non-skeleton variant. However, skeleton SR-trees have better performance than skeleton R-trees for non-uniformly distributed interval lengths and query regions of very high or very low aspect ratio. The approach of (skeleton)SR-trees suffers from the same major inefficiencies as using R-trees to store multiversion data. There is no good worst-case guarantee for deletions, exact-match queries, and range queries.

为了针对此类数据分布实现更好的查询和更新性能，科洛夫森（Kolovson）和斯通布雷克（Stonebraker）提出了分段R树（SR树；Kolovson和Stone - braker，1991年），它是分段树（本特利（Bentley），1977年）和R树（古特曼（Guttman），1984年）的混合体。骨架SR树基于对数据分布的假设，采用预设的数据空间划分方式进行操作。在科洛夫森和斯通布雷克（1991年）进行的性能评估中，非骨架变体的SR树性能从未超过R树。然而，对于非均匀分布的区间长度以及高纵横比或低纵横比的查询区域，骨架SR树的性能优于骨架R树。（骨架）SR树的方法与使用R树存储多版本数据存在同样的主要低效问题。对于删除操作、精确匹配查询和范围查询，没有良好的最坏情况保证。

Elmasri et al. (1990, 1991) proposed the time index for maintaining historical data. The time index supports all the operations of our setting, plus range queries over versions. In the time index,data records are organized in a ${B}^{ + }$ -tree according to versions (time). For each version, a bucket is maintained for storing all data records (or references to it) valid for that version. Elmasri et al. proposed several modifications of the basic approach to reduce the high redundancy resulting from this data organization. However, assuming that each update creates a new version, the space efficiency of all those variants may be as bad as $\Theta \left( {{N}^{2}/b}\right)$ in the worst case. An insertion of a record in the time index may create a new bucket containing all records for the new version. In the worst case,this operation requires $\Theta \left( {N/b}\right)$ time. Moreover, the time index does not support range queries efficiently: range query efficiency may be as bad as $\Theta \left( {{\log }_{b}N + N/b}\right)$ in the worst case.

埃尔马斯里（Elmasri）等人（1990年、1991年）提出了用于维护历史数据的时间索引。时间索引支持我们设定的所有操作，还支持对版本进行范围查询。在时间索引中，数据记录根据版本（时间）组织成${B}^{ + }$树。对于每个版本，都会维护一个桶来存储该版本有效的所有数据记录（或其引用）。埃尔马斯里等人提出了对基本方法的几种改进，以减少这种数据组织导致的高冗余。然而，假设每次更新都会创建一个新版本，在最坏情况下，所有这些变体的空间效率可能低至$\Theta \left( {{N}^{2}/b}\right)$。在时间索引中插入一条记录可能会创建一个新桶，其中包含新版本的所有记录。在最坏情况下，此操作需要$\Theta \left( {N/b}\right)$时间。此外，时间索引不能有效地支持范围查询：在最坏情况下，范围查询效率可能低至$\Theta \left( {{\log }_{b}N + N/b}\right)$。

The Write-once B-tree (WOBT), proposed by Easton (1986),is a variation of the ${B}^{ + }$ -tree; it is completely stored on a WORM medium, e.g. an optical disk. Because of the write-once characteristic, all versions of data are kept forever. If version numbers are assigned to the index and data records, multiversion queries can be answered in a straightforward way. To treat an overflow of a data or an index block in the WOBT, first a version split must be performed, because the overflow block itself cannot be rewritten. Afterwards, if the current entries occupy more than a given fraction of the new block (e.g. two-thirds), a key split is performed on the block before writing it to external memory. So far, the WOBT split policy is comparable to the one of the MVBT. One major difference is the treatment of a root split: if a root is split in the WOBT, a new root block is allocated that initially contains three references. One reference is pointing to the old root block, whereas the other references are pointing to the blocks obtained from splitting the old root. Thus, a WOBT has one root, and all the paths from the root to a data block have the same length $\Theta \left( {{\log }_{b}N}\right)$ . In contrast, if a root is split in the MVBT, the reference to the new root is inserted into root*, the data structure organizing the root blocks.

伊斯顿（Easton）在1986年提出的一次写入B树（WOBT）是${B}^{ + }$树的一种变体；它完全存储在一次写入多次读取（WORM）介质上，例如光盘。由于一次写入的特性，数据的所有版本都会永久保存。如果为索引和数据记录分配版本号，就可以直接回答多版本查询。为了处理WOBT中数据块或索引块的溢出问题，首先必须执行版本分割，因为溢出块本身不能被重写。之后，如果当前条目占用新块的比例超过给定值（例如三分之二），则在将该块写入外部存储器之前对其进行键分割。到目前为止，WOBT的分割策略与多版本B树（MVBT）的分割策略相当。一个主要区别在于对根节点分割的处理：如果在WOBT中分割根节点，会分配一个新的根块，该块最初包含三个引用。一个引用指向旧的根块，而其他引用指向从分割旧根块得到的块。因此，WOBT有一个根节点，并且从根节点到数据块的所有路径长度都相同，为$\Theta \left( {{\log }_{b}N}\right)$。相比之下，如果在MVBT中分割根节点，对新根节点的引用会插入到root*中，root*是组织根块的数据结构。

Under the pessimistic assumption that the computation of the root of an arbitrary non-current version requires $\Theta \left( {{\log }_{b}N}\right)$ time,the MVBT is still more time-efficient than the WOBT for updates and queries on the current version. Recall that the root of the MVBT valid for the current version - and for some recent non-current versions - can be accessed in time $O\left( 1\right)$ by maintaining a direct reference to this block. Therefore, queries to these versions and updates to the current version are more efficient than $O\left( {{\log }_{b}N + r/b}\right)$ and $O\left( {{\log }_{b}N}\right)$ ,the respective bounds in the WOBT. Moreover, the WOBT is restricted to insertions and modifications of the non-key part of records, while the MVBT supports both insertions and deletions.

在悲观假设下，即计算任意非当前版本的根节点需要$\Theta \left( {{\log }_{b}N}\right)$时间，对于当前版本的更新和查询，多版本B树（MVBT）仍然比一次写入B树（WOBT）更具时间效率。请记住，通过维护对当前版本（以及一些近期非当前版本）有效的MVBT根节点块的直接引用，可以在$O\left( 1\right)$时间内访问该根节点。因此，对这些版本的查询和对当前版本的更新比WOBT中的相应界限$O\left( {{\log }_{b}N + r/b}\right)$和$O\left( {{\log }_{b}N}\right)$更高效。此外，WOBT仅限于对记录的非键部分进行插入和修改，而MVBT支持插入和删除操作。

In order to reduce storage costs and to improve performance of queries on the current version, Lomet and Salzberg (1989) proposed a variant of the WOBT, the time-split B-tree (TSBT). The TSBT spans over magnetic and WORM disk. All live blocks are stored on magnetic disk, while a dead block migrates to WORM disk during a version split. Lomet and Salzberg distinguish split policies for index blocks from those for data blocks.

为了降低存储成本并提高对当前版本查询的性能，洛梅特（Lomet）和萨尔茨伯格（Salzberg）（1989年）提出了写一次读多次B树（WOBT）的一种变体——时间分割B树（TSBT）。TSBT跨越了磁盘和一次写入多次读取（WORM）磁盘。所有活动块都存储在磁盘上，而在版本分割期间，死亡块会迁移到WORM磁盘。洛梅特和萨尔茨伯格区分了索引块和数据块的分割策略。

For splitting data blocks, the following two basic types of splits can be performed in the TSBT. First, in contrast to the WOBT, the version (time) used for a version split (time split) of a data block is not restricted to the current version, but can be chosen arbitrarily. Second, a key split can be performed on a data block instead of a version (time) split. Lomet and Salzberg(1989,1990)discussed the effects of different data block split policies, with emphasis on space cost. The space cost is given as the sum of storage cost on magnetic and WORM disk. For data block split, the following three split policies were proposed:

对于分割数据块，在TSBT中可以执行以下两种基本类型的分割。首先，与WOBT不同，用于数据块版本分割（时间分割）的版本（时间）不限于当前版本，而是可以任意选择。其次，可以对数据块执行键分割而不是版本（时间）分割。洛梅特和萨尔茨伯格（1989年，1990年）讨论了不同数据块分割策略的效果，重点关注空间成本。空间成本是磁盘和WORM磁盘上存储成本的总和。对于数据块分割，提出了以下三种分割策略：

- The WOBT policy is the split policy as used in the WOBT.

- WOBT策略是WOBT中使用的分割策略。

- The time-of-last-update policy performs a version split with the version of the last update. This reduces the number of entries to be kept in the dead block after a version split, and therefore the storage space needed on WORM disk. The number of entries in the live block remains unchanged in comparison to a version split of the WOBT. As for the WOBT, a key split will be performed immediately after a version split, if the current entries occupy at least a given fraction of the new block (e.g. two-thirds).

- 最后更新时间策略使用最后一次更新的版本进行版本分割。这减少了版本分割后死亡块中需要保留的条目数量，从而减少了WORM磁盘上所需的存储空间。与WOBT的版本分割相比，活动块中的条目数量保持不变。对于WOBT，如果当前条目至少占据新块的给定比例（例如三分之二），则在版本分割后将立即执行键分割。

- The isolated-key-split policy performs a key split if at least a given fraction of the entries (e.g. two-thirds) of the overfull node belongs to the current version. Otherwise, a version split with the current version is performed. In comparison with the two split policies described above, this split policy reduces redundancy and therefore storage space: a version split is not performed if it would be immediately followed by a key split. The disadvantage of this policy is that by a key split the dead entries of the block are spread over two blocks. This decreases the performance for range queries to non-current versions. Consequently, in contrast to the MVBT it is not guaranteed that a block contains for each version either no entries or $\Theta \left( b\right)$ entries. Then,a range query in the worst case requires $\Theta \left( {N/b}\right)$ blocks,independent of the size of the response set and independent of the number of records in the corresponding version. In comparison with the TSBT, the MVBT requires more storage space [but it is still $O\left( N\right)$ ] to cluster versions appropriately such that range queries can be answered with $O\left( {{\log }_{b}N + r/b}\right)$ disk accesses.

- 孤立键分割策略，如果过满节点中至少给定比例（例如三分之二）的条目属于当前版本，则执行键分割。否则，使用当前版本进行版本分割。与上述两种分割策略相比，这种分割策略减少了冗余，从而减少了存储空间：如果版本分割后会立即进行键分割，则不执行版本分割。这种策略的缺点是，通过键分割，块的死亡条目会分散到两个块中。这会降低对非当前版本范围查询的性能。因此，与多版本B树（MVBT）不同，不能保证一个块对于每个版本要么没有条目，要么有$\Theta \left( b\right)$个条目。那么，在最坏的情况下，范围查询需要$\Theta \left( {N/b}\right)$个块，与响应集的大小无关，也与相应版本中的记录数量无关。与TSBT相比，MVBT需要更多的存储空间[但仍然可以$O\left( N\right)$]来适当地聚类版本，以便可以通过$O\left( {{\log }_{b}N + r/b}\right)$次磁盘访问来回答范围查询。

For index blocks, split policies cannot be the same as for data blocks. The problem of using data block split policies on index nodes is the following: a dead index block may still contain references to live blocks on the next lower level of the tree. If such a live block becomes dead (i.e. it migrates to optical disk), the corresponding references have to be updated in the parent nodes. However, this would require that the dead blocks are stored on a write-many storage medium.

对于索引块，分割策略不能与数据块相同。在索引节点上使用数据块分割策略的问题如下：一个死亡索引块可能仍然包含对树中下一级活动块的引用。如果这样的活动块变为死亡块（即它迁移到光盘），则必须在父节点中更新相应的引用。然而，这需要将死亡块存储在可多次写入的存储介质上。

In their first paper on the TSBT, Lomet and Salzberg (1989) discuss the effects of using version and key splits for index block splitting. An index block split policy based on key splits avoids redundancy, but leads to an index which gives no selectivity according to versions. Moreover, an index block may contain entries which cannot be separated by a key split. Therefore, for the simulations presented in Lomet and Salzberg (1990), the authors applied another policy for index block splitting. A version split is performed using the insertion version of the oldest index entry that is still valid for the current version. Then, a dead block contains only non-current index entries and therefore it can be written onto WORM disk. In addition to the redundancy that this entails in index blocks, the main problem of this split policy is that such a split version may not exist. In this case, a key split is possible. This separates not only current entries (as desired), but also dead ones. As a consequence, the TSBT does not have a lower bound on the number of entries for a version in an index block.

在他们关于TSBT的第一篇论文中，洛梅特和萨尔茨伯格（1989年）讨论了使用版本分割和键分割进行索引块分割的效果。基于键分割的索引块分割策略避免了冗余，但导致索引根据版本没有选择性。此外，一个索引块可能包含无法通过键分割分离的条目。因此，在洛梅特和萨尔茨伯格（1990年）提出的模拟中，作者应用了另一种索引块分割策略。使用仍然对当前版本有效的最旧索引条目的插入版本进行版本分割。然后，一个死亡块只包含非当前索引条目，因此可以将其写入WORM磁盘。除了这在索引块中带来的冗余之外，这种分割策略的主要问题是这样的分割版本可能不存在。在这种情况下，可以进行键分割。这不仅分离了当前条目（如预期的那样），还分离了死亡条目。因此，TSBT在索引块中一个版本的条目数量上没有下限。

Lanka and Mays (1991) presented three approaches for fully persistent ${B}^{ + }$ -trees. Full persistence means that changes can be applied to any version, current or past, creating a new version. Because this concept of multiple versions of data is more general than ours, all three approaches also can be used to maintain our type of multiversion data (partially persistent data). Like the MVBT, and in contrast to the WOBT and the TSBT, all the proposed techniques support insertions and deletions.

兰卡（Lanka）和梅斯（Mays）（1991年）提出了三种用于完全持久化${B}^{ + }$树的方法。完全持久化意味着可以对任何版本（当前版本或过去版本）进行更改，从而创建一个新版本。由于这种多版本数据的概念比我们的更通用，因此这三种方法也可用于维护我们这种类型的多版本数据（部分持久化数据）。与多版本B树（MVBT）一样，与写时复制B树（WOBT）和时间戳B树（TSBT）不同，所有提出的技术都支持插入和删除操作。

The first approach, the fat node method, is based on the idea that each node, index node or leaf with data items is fat enough to store all versions of all its entries. Lanka and Mays proposed implementing such a fat node as a set of blocks, one block per version, and a version block, containing references to each of the blocks. Although query and update efficiency for any given version $i$ is $O\left( {{\log }_{b}{m}_{i}}\right)$ [based on the assumption that the root block for version $i$ can be accessed in time $O\left( 1\right)$ ],this obviously leads to storage cost of $\Theta \left( 1\right)$ blocks per update. Moreover,it is doubtful whether one physical block is sufficient to implement a version block, as assumed in the paper.

第一种方法，即胖节点法，其基于这样的理念：每个节点（索引节点或包含数据项的叶子节点）都足够“胖”，能够存储其所有条目的所有版本。兰卡和梅斯提议将这样的胖节点实现为一组块，每个版本对应一个块，再加上一个版本块，其中包含对每个块的引用。尽管对于任何给定版本$i$的查询和更新效率为$O\left( {{\log }_{b}{m}_{i}}\right)$（基于版本$i$的根块可以在时间$O\left( 1\right)$内访问的假设），但这显然会导致每次更新的存储成本为$\Theta \left( 1\right)$个块。此外，正如论文中所假设的，一个物理块是否足以实现一个版本块，这是值得怀疑的。

The fat field method is an improvement on the fat node method, storing entries of different versions in the same block. To describe which versions a ${B}^{ + }$ -tree entry belongs to, each entry is extended by a field representing its insertion version and the set of its deletion versions. Applying the fat field method to our multiversion data, the structure of an entry is equal to that of a MVBT entry, because only one deletion version can occur. Also comparable to the MVBT, the fat field method guarantees for each block and each version in the block that a number of entries proportional to the block capacity (namely ${50}\%$ ) is stored in that block. If for any version less than half of the entries belong to that version, a version split and a merge is performed. The split policy is a version split, followed by a key split if the block is still overfull. In contrast to the MVBT, for the fat field method a block may be full after split or merge. That means that after a constant number of updates, the next split or merge may be triggered, leading to a worst-case storage cost of $\Theta \left( 1\right)$ blocks per update. As for the fat node method,the query performance analysis for the fat field method is based on the assumption that each version block fits into one physical block. This assumption is not realistic for organizing a high number of versions in the structure.

胖字段法是对胖节点法的一种改进，它将不同版本的条目存储在同一个块中。为了描述一个${B}^{ + }$树条目属于哪些版本，每个条目都扩展了一个字段，用于表示其插入版本和删除版本集合。将胖字段法应用于我们的多版本数据时，条目的结构与多版本B树（MVBT）条目的结构相同，因为只会出现一个删除版本。同样与多版本B树（MVBT）类似，胖字段法保证对于每个块以及块中的每个版本，与块容量成比例（即${50}\%$）的条目数量存储在该块中。如果对于任何版本，属于该版本的条目少于一半，则执行版本拆分和合并操作。拆分策略是先进行版本拆分，如果块仍然过满，则再进行键拆分。与多版本B树（MVBT）不同，对于胖字段法，块在拆分或合并后可能是满的。这意味着在进行一定数量的更新后，可能会触发下一次拆分或合并，导致每次更新的最坏情况存储成本为$\Theta \left( 1\right)$个块。与胖节点法一样，胖字段法的查询性能分析基于每个版本块都能放入一个物理块的假设。对于在结构中组织大量版本的情况，这个假设并不现实。

The third approach proposed by Lanka and Mays (1991) is the pure version block method. In this technique,a ${B}^{ + }$ - tree index is built over the key values of the data items. This technique does not give any selectivity according to versions.

兰卡和梅斯（1991年）提出的第三种方法是纯版本块法。在这种技术中，会在数据项的键值上构建一个${B}^{ + }$树索引。这种技术不会根据版本提供任何选择性。

As a result, we conclude that the approaches for multiversion B-trees proposed in the literature have their merits in exposing many interesting ideas and achieving good performance in one or the other aspect. Nevertheless, none of them achieves asymptotic worst-case optimality both in the time for all operations and in space. Therefore, we feel the MVBT to be a worthwhile addition to the list of multiversion external B-trees.

因此，我们得出结论，文献中提出的多版本B树方法在揭示许多有趣的想法以及在某一个或其他方面取得良好性能方面有其优点。然而，它们都没有在所有操作的时间和空间上实现渐近最坏情况的最优性。因此，我们认为多版本B树（MVBT）是多版本外部B树列表中值得添加的一种方法。

## 6 Conclusion

## 6 结论

In this paper, we have presented a technique to transform certain single-version hierarchical external storage access structures into multiversion structures. We have shown that our technique delivers multiversion capabilities with no change in asymptotic worst-case performance for B-trees, if we assume that the root block for a requested version is given from the application context. Otherwise, a search structure for the appropriate root block can be tuned to the particular requirements. The properties of B-trees that we have used include the following characteristics of access structures:

在本文中，我们提出了一种将某些单版本分层外部存储访问结构转换为多版本结构的技术。我们已经证明，如果我们假设从应用程序上下文中可以获取请求版本的根块，那么我们的技术在不改变B树渐近最坏情况性能的前提下提供了多版本功能。否则，可以根据特定需求调整合适根块的搜索结构。我们所使用的B树的属性包括访问结构的以下特征：

1. The access structure is a rooted tree of external storage blocks.

1. 访问结构是一个由外部存储块组成的有根树。

2. Data items are stored in the leaves (data blocks) of the tree; the inner nodes (directory blocks) store routing information.

2. 数据项存储在树的叶子节点（数据块）中；内部节点（目录块）存储路由信息。

3. The tree is balanced; typically, all leaves are on the same level.

3. 树是平衡的；通常，所有叶子节点都在同一层。

4. The tree can be restructured by splitting blocks or by merging blocks with siblings along a path between the root and a leaf.

4. 可以通过拆分块或沿着根节点和叶子节点之间的路径将块与其兄弟块合并来重构树。

5. A block split can be balanced; that is, each of the two resulting blocks is guaranteed to contain at least a constant fraction $\alpha ,0 < \alpha  \leq  {0.5}$ ,of the entries.

5. 块拆分可以是平衡的；也就是说，保证两个结果块中的每一个都至少包含条目的一个恒定比例$\alpha ,0 < \alpha  \leq  {0.5}$。

Single-version access structures satisfying these requirements are therefore the prime candidates for carrying over and applying our technique. Examples of such access structures other than the B-tree include the cell-tree (Günther and Bilmes 1991), the BANG file (Freeston 1987), and the R-tree family (Guttman 1984, Greene 1989, Beckmann et al. 1990), whenever reinsertion of data items can be replaced by block merge without loss of geometric clustering. Note that the data items are not limited to one-dimensional points.

因此，满足这些要求的单版本访问结构是应用我们技术的首选。除了B树之外，此类访问结构的示例还包括单元树（Cell-tree，Günther和Bilmes，1991年）、BANG文件（BANG file，Freeston，1987年）以及R树族（R-tree family，Guttman，1984年；Greene，1989年；Beckmann等人，1990年），前提是数据项的重新插入可以通过块合并来替代，而不会损失几何聚类性。请注意，数据项不限于一维点。

We conjecture that our technique may be useful also for access structures that do not satisfy all of our requirements, such as hierarchical grid files. In that case, the performance guarantees derived for the MVBT do not carry over without change. This is clearly due to the fact that these performance guarantees do not hold for the single-version structure in the first place. However, we do not know in sufficient generality how the performance of an arbitrary external access structure changes if it is transformed into a multiversion structure along the lines of our technique.

我们推测，我们的技术对于那些不完全满足我们所有要求的访问结构（如分层网格文件）可能也有用。在这种情况下，为多版本B树（MVBT）推导的性能保证不能直接适用。这显然是因为这些性能保证首先就不适用于单版本结构。然而，我们并不清楚，如果将任意外部访问结构按照我们的技术转换为多版本结构，其性能会如何变化。

Acknowledgements. We want to thank an anonymous referee for an extraordinary effort and thorough discussion that led to a great improvement in the presentation of the paper. This work was partially supported by grants ESPRIT 6881 of the European Community and Wi810/2-5 of the Deutsche Forschungsgemeinschaft DFG.

致谢。我们要感谢一位匿名审稿人付出的非凡努力和进行的深入讨论，这些使得本文的呈现有了很大的改进。这项工作部分得到了欧洲共同体ESPRIT 6881项目和德国研究基金会（Deutsche Forschungsgemeinschaft DFG）Wi810/2 - 5项目的资助。

## References

## 参考文献

Barghouti NS, Kaiser GE (1991) Concurrency control in advanced database applications. ACM Comput Surv 23:269-317

Becker B, Gschwind S, Ohler T, Seeger B, Widmayer P (1993) On optimal multiversion access structures. In: 3rd International Symposium on Large Spatial Databases. (Lecture Notes in Computer Science, vol 692) Springer, Berlin Heidelberg New York, pp 123-141

Beckmann N,Kriegel HP,Schneider R,Seeger B (1990) The R*-tree: an efficient and robust access method for points and rectangles. ACM SIGMOD International Conference on Management of Data 19:322- 331

Bentley JL (1977) Algorithms for Klee's rectangle problems. Computer Science Department, Carnegie-Mellon University, Pittsburg, Pa

Bernstein PA, Hadzilacos V, Goodman N (1987) Concurrency control and recovery in database systems. Addison Wesley, Reading, Mass

Clifford J, Ariav G (1986) Temporal data management: models and systems. In: Ariav G, Clifford J (eds) New directions for database systems. Ablex, Norwood, NJ, pp 168-186

Driscoll JR, Sarnak N, Sleator DD, Tarjan RE (1989) Making data structures persistent. J Comput Syst Sci 38:86-124

Easton M (1986) Key-sequence data sets on indelible storage. IBM J Res Dev 30:230-241

Elmasri R, Wuu G, Kim Y-J (1990) The time index: an access structure for temporal data. 16th International Conference on Very Large Data Bases, pp 1-12

Elmasri R, Wuu G, Kim Y-J (1991) Efficient implementation techniques for the time index. Seventh IEEE International Conference on Data Engineering, pp 102-111

Freeston MW (1987) The BANG-file: a new kind of grid file. ACM SIGMOD International Conference on Management of Data 16:260- 269

Gonnet GH, Baeza-Yates R (1991) Handbook of algorithms and data structures: in PASCAL and C. Addison-Wesley, Reading, Mass

Greene, D (1989) An implementation and performance analysis of spatial access methods. Fifth IEEE International Conference on Data Engineering, pp 606-615

Günther O, Bilmes J (1991) Tree-based access methods for spatial databases: implementation and performance evaluation. IEEE Trans Knowl Data Eng 3:342-356

Guttman A (1984) R-trees: a dynamic index structure for spatial searching. ACM SIGMOD International Conference on Management of Data 12:47-57

Huddleston S, Mehlhorn K (1982) A new data structure for representing sorted lists. Acta Inform 17:157-184

Kanellakis PC, Ramaswamy S, Vengroff DE, Vitter JS (1993) Indexing for data models with constraints and classes. ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems 12:233-243

Katz RH (1990) Towards a unified framework for version modeling in engineering databases. ACM Comput Surv 22:375-408

Kolovson C, Stonebraker M (1989) Indexing techniques for historical databases. Fifth IEEE International Conference on Data Engineering, pp 127-137

Kolovson C, Stonebraker M (1991) Segment indexes: dynamic indexing techniques for multi-dimensional interval data. ACM SIGMOD International Conference on Management of Data 20:138-147

Lanka S,Mays E (1991) Fully persistent ${B}^{ + }$ -trees. ACM SIGMOD International Conference on Management of Data 20:426-435

Lomet D, Salzberg B (1989) Access methods for multiversion data. ACM SIGMOD International Conference on Management of Data 18:315- 324

Lomet D, Salzberg B (1990) The performance of a multiversion access method. ACM SIGMOD International Conference on Management of Data 19:353-363

Mehlhorn K, Tsakalidis A (1990) Data structures. In: Leeuwen J van (ed) Handbook of theoretical computer science, vol A: Algorithms and complexity. Elsevier, Amsterdam, pp 301-341

Sedgewick R (1988) Algorithms. Addison-Wesley, Reading, Mass

Segev A, Gunadhi H (1989) Event-join optimization in temporal relational databases. 15th International Conference on Very Large Data Bases, pp 205-215

Tansel, AU, Clifford J, Gadia S, Jajodia S, Segev A, Snodgrass R (1993) Temporal databases - theory, design, implementation. Benjamin/Cummings, Redwood City, Calif

Vitter JS, (1991) Efficient memory access in large-scale computation. In: 8th Annual Symposium on Theoretical Aspects of Computer Science. (Lecture Notes in Computer Science, vol 480) Springer, Berlin Heidelberg New York, pp 26-41