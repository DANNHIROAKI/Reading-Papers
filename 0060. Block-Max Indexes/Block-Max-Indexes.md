# Faster Top-k Document Retrieval Using Block-Max Indexes

# 使用块最大索引实现更快的前 k 个文档检索

Shuai Ding

丁帅

Polytechnic Institute of NYU

纽约大学理工学院（Polytechnic Institute of NYU）

Brooklyn, New York, USA

美国纽约布鲁克林

sding@cis.poly.edu

Torsten Suel

托尔斯滕·苏埃尔（Torsten Suel）

Polytechnic Institute of NYU

纽约大学理工学院（Polytechnic Institute of NYU）

Brooklyn, New York, USA

美国纽约布鲁克林

suel@poly.edu

## ABSTRACT

## 摘要

Large search engines process thousands of queries per second over billions of documents, making query processing a major performance bottleneck. An important class of optimization techniques called early termination achieves faster query processing by avoiding the scoring of documents that are unlikely to be in the top results. We study new algorithms for early termination that outperform previous methods. In particular, we focus on safe techniques for disjunctive queries, which return the same result as an exhaustive evaluation over the disjunction of the query terms. The current state-of-the-art methods for this case, the WAND algorithm by Broder et al. [11] and the approach of Strohman and Croft [30], achieve great benefits but still leave a large performance gap between disjunctive and (even non-early terminated) conjunctive queries.

大型搜索引擎每秒要对数十亿个文档处理数千个查询，这使得查询处理成为主要的性能瓶颈。一类重要的优化技术——提前终止（early termination），通过避免对不太可能出现在顶级结果中的文档进行评分，实现了更快的查询处理。我们研究了用于提前终止的新算法，这些算法的性能优于以往的方法。具体而言，我们专注于用于析取查询的安全技术，该技术返回的结果与对查询词的析取进行穷举评估的结果相同。针对这种情况，目前最先进的方法是布罗德（Broder）等人 [11] 提出的 WAND 算法以及斯特罗曼（Strohman）和克罗夫特（Croft） [30] 提出的方法，这些方法带来了很大的益处，但在析取查询和（甚至未提前终止的）合取查询之间仍存在较大的性能差距。

We propose a new set of algorithms by introducing a simple augmented inverted index structure called a block-max index. Essentially, this is a structure that stores the maximum impact score for each block of a compressed inverted list in uncompressed form, thus enabling us to skip large parts of the lists. We show how to integrate this structure into the WAND approach, leading to considerable performance gains. We then describe extensions to a layered index organization, and to indexes with reassigned document IDs, that achieve additional gains that narrow the gap between disjunctive and conjunctive top- $k$ query processing.

我们通过引入一种名为块最大索引（block-max index）的简单增强倒排索引结构，提出了一组新的算法。本质上，这是一种以未压缩形式存储压缩倒排列表中每个块的最大影响得分的结构，从而使我们能够跳过列表的大部分内容。我们展示了如何将这种结构集成到宽阈跳跃（WAND，Wide Area Network Data）方法中，从而显著提高性能。然后，我们描述了对分层索引组织以及重新分配文档ID的索引的扩展，这些扩展实现了额外的性能提升，缩小了析取和合取前$k$查询处理之间的差距。

## Categories and Subject Descriptors

## 类别和主题描述符

H.3.3 [INFORMATION STORAGE AND RETRIEVAL]: Information Search and Retrieval.

H.3.3 [信息存储与检索（INFORMATION STORAGE AND RETRIEVAL）]：信息搜索与检索。

## General Terms

## 通用术语

Algorithms, Performance.

算法、性能。

## Keywords

## 关键词

IR query processing, top-k query processing, early termination, inverted index.

信息检索（IR）查询处理、前k个查询处理、提前终止、倒排索引。

## 1. INTRODUCTION

## 1. 引言

Due to the rapid growth of the web, more and more people are relying on search engines to locate useful information. As a result, an increasing share of the world's computational resources is spent on search-related tasks. Current large-scale search engines have to be able to answer hundreds of millions of queries per day on tens of billions of web pages. Thus, highly optimized methods are needed to efficiently process all these queries.

由于网络的快速发展，越来越多的人依靠搜索引擎来查找有用信息。因此，全球越来越多的计算资源被用于与搜索相关的任务。当前的大规模搜索引擎必须能够每天对数以千亿计的网页处理数亿个查询。因此，需要高度优化的方法来高效处理所有这些查询。

One major bottleneck in query processing is the length of the inverted list index structures (described in the next section), which can easily grow to hundreds of MBs or even GBs for common terms (roughly linear in the size of the data set). Given that search engines need to answer user queries within fractions of a second, naively traversing this basic index structure, which could take hundreds of milliseconds or more for common terms, is not acceptable.

查询处理的一个主要瓶颈是倒排列表索引结构的长度（下一节将详细介绍），对于常见词汇，其长度很容易增长到数百兆字节甚至数吉字节（大致与数据集的大小呈线性关系）。鉴于搜索引擎需要在不到一秒的时间内响应用户查询，简单地遍历这种基本索引结构（对于常见词汇可能需要数百毫秒甚至更长时间）是不可接受的。

This basic problem has long been recognized by researchers, and has motivated a lot of work on optimization techniques including distributed computation [29, 5, 24], index compression [37], caching [7, 22], and early termination [31, 12] (also called pruning or optimized top-k processing). In this paper we focus on early termination,which in a nutshell means returning the best $k = {10}$ or 100 results without an exhaustive traversal of the relevant index structures. In particular, we propose to augment the index by adding additional information. Essentially, we add to each compressed block in the inverted lists one value, the maximum impact score. While this is a simple idea, we are not aware of any previous work that stores such information for better query processing. We call this modified index structure a Block-Max Index. We also propose a new set of algorithms based on the WAND approach [11] for safe early termination (where exactly the same results as in the naive baseline are returned) using our block-max index structure. One interesting property of our algorithms is that they performs document-at-a-time (DAAT) index traversal, based on either document-sorted or impact-layered index structures.

研究人员早就认识到了这个基本问题，并因此推动了许多关于优化技术的研究工作，这些技术包括分布式计算 [29, 5, 24]、索引压缩 [37]、缓存 [7, 22] 和提前终止 [31, 12]（也称为剪枝或优化的前 k 项处理）。在本文中，我们专注于提前终止，简而言之，就是在不详尽遍历相关索引结构的情况下返回最佳的 $k = {10}$ 个或 100 个结果。具体而言，我们提议通过添加额外信息来扩充索引。本质上，我们为倒排列表中的每个压缩块添加一个值，即最大影响得分。虽然这是一个简单的想法，但我们并未发现之前有任何工作为了更好地处理查询而存储此类信息。我们将这种修改后的索引结构称为块最大索引（Block-Max Index）。我们还基于 WAND 方法 [11] 提出了一组新的算法，用于利用我们的块最大索引结构进行安全的提前终止（即返回与朴素基线方法完全相同的结果）。我们的算法有一个有趣的特性，即它们基于按文档排序或按影响分层的索引结构，进行逐文档（DAAT）的索引遍历。

## 2. BACKGROUND

## 2. 背景

In this section, we provide background on inverted index structures, query processing, and early termination.

在本节中，我们将介绍倒排索引结构、查询处理和提前终止的相关背景知识。

### 2.1 Inverted Indexes and Index Compression

### 2.1 倒排索引与索引压缩

Current search engines perform query processing based on an inverted index, which is a simple and efficient data structure that allows us to find documents that contain a particular term [37]. Given a collection of $N$ documents,we assume that each document is identified by a unique document ${ID}$ (docID) between 0 and $N - 1$ . An inverted index consists of many inverted lists, where each inverted list ${L}_{w}$ is a list of postings describing all places where term $w$ occurs in the collection. More precisely,each posting contains the docID of a document that contains the term $w$ ,the number of occurrences of $w$ in the document (called frequency),and sometimes the exact locations of these occurrences in the document (called positions), plus maybe other context such as font size etc. Postings in an inverted list are typically sorted by docID, or sometimes by some other measure (described later). Thus, in the case where we store docIDs and frequencies,each posting is of the form $\left( {{d}_{i},{f}_{i}}\right)$ . We focus on this case in this paper, but all our techniques also apply to cases where positions, context information, or precomputed quantized impact scores are stored.

当前的搜索引擎基于倒排索引（inverted index）进行查询处理，倒排索引是一种简单高效的数据结构，它使我们能够找到包含特定术语的文档 [37]。给定一个包含 $N$ 个文档的集合，我们假设每个文档由一个介于 0 到 $N - 1$ 之间的唯一文档 ${ID}$（文档编号，docID）标识。倒排索引由许多倒排列表组成，其中每个倒排列表 ${L}_{w}$ 是一个包含所有描述术语 $w$ 在集合中出现位置的记录列表。更准确地说，每个记录包含一个包含术语 $w$ 的文档的 docID、术语 $w$ 在该文档中的出现次数（称为词频，frequency），有时还包含这些出现位置在文档中的精确位置（称为位置信息，positions），此外可能还有其他上下文信息，如字体大小等。倒排列表中的记录通常按 docID 排序，有时也按其他度量标准（稍后描述）排序。因此，在我们存储 docID 和词频的情况下，每个记录的形式为 $\left( {{d}_{i},{f}_{i}}\right)$。本文主要关注这种情况，但我们所有的技术也适用于存储位置信息、上下文信息或预先计算的量化影响分数的情况。

The inverted lists of common query terms may consist of many millions or even billions of postings. To allow faster access to lists on disk, and limit the memory needed, search engines use sophisticated compression techniques that significantly reduce the size of each inverted list [37]. Compression is crucial for search engine performance $\left\lbrack  {{14},{36}}\right\rbrack$ ,and there are many compression techniques in the literature; see $\left\lbrack  {{36},{35},{27}}\right\rbrack$ . In this paper we use the New-PFD compression method, which was shown to perform well in [35], but our ideas also apply to other compression techniques.

常见查询词的倒排表可能包含数百万甚至数十亿个倒排列表项。为了能更快地访问磁盘上的列表，并限制所需的内存，搜索引擎采用了复杂的压缩技术，这些技术能显著减小每个倒排表的大小[37]。压缩对于搜索引擎的性能至关重要$\left\lbrack  {{14},{36}}\right\rbrack$，文献中也有许多压缩技术；参见$\left\lbrack  {{36},{35},{27}}\right\rbrack$。在本文中，我们使用了New - PFD压缩方法，该方法在文献[35]中表现良好，但我们的思路同样适用于其他压缩技术。

As the lists for common terms could be very long, we want to be able to skip most parts of the lists during query processing. To do so, inverted lists are often split into blocks of, say, 64 or 128 do-cIDs, such that each block can be decompressed separately. To do so, we have an extra table, which stores for each block the uncompressed maximum (or minimum) docID and the block size in this table. The size of this extra table is small compared to the size of the inverted index. Thus, 64 or 128 postings are grouped together as a block where we store 64 or 128 compressed docIDs, followed by the corresponding compressed frequencies.

由于常用术语的列表可能非常长，我们希望在查询处理过程中能够跳过列表的大部分内容。为此，倒排表通常会被分割成若干块，例如每块包含64或128个文档ID（doc - ID），这样每个块就可以单独解压缩。为此，我们有一个额外的表，该表为每个块存储未压缩的最大（或最小）文档ID以及块的大小。与倒排索引的大小相比，这个额外表的大小较小。因此，64或128个记录会被分组为一个块，在这个块中我们存储64或128个压缩后的文档ID，随后是相应的压缩频率。

### 2.2 Query Processing

### 2.2 查询处理

Given the inverted index structure mentioned above, the most basic form of query processing is called Boolean query processing. A query (apple AND orange) OR pear for all documents containing both words apple and orange, or the word pear, can be implemented by first intersecting the docIDs in the inverted lists for apple and orange, and then merging the result with the inverted list for pear.

鉴于上述倒排索引结构，最基本的查询处理形式称为布尔查询处理。对于查询 (apple AND orange) OR pear，即查找所有同时包含单词“apple”和“orange”，或者包含单词“pear”的文档，可以通过以下方式实现：首先对“apple”和“orange”的倒排表中的文档ID进行交集操作，然后将结果与“pear”的倒排表进行合并。

Search engines use ranked query processing, where a ranking function is used to compute a score for each document passing a simple Boolean filter,and then the $\mathrm{k}$ top-scoring documents are finally returned. This ranking function should be efficiently computable from the information in the inverted lists (i.e., the frequencies and maybe positions) plus a limited amount of other statistics stored outside the inverted index (e.g., document lengths or global scores such as Pagerank). Many classes of functions similar to BM25 or Cosine have been studied; see [6] for more details.

搜索引擎采用排序查询处理，其中使用排序函数为通过简单布尔过滤器的每个文档计算得分，然后最终返回得分最高的 $\mathrm{k}$ 个文档。该排序函数应能根据倒排列表中的信息（即词频，可能还有位置信息）以及存储在倒排索引之外的少量其他统计信息（例如文档长度或诸如网页排名（Pagerank）之类的全局得分）高效计算得出。已经对许多类似于 BM25 或余弦相似度的函数类别进行了研究；更多详细信息请参阅 [6]。

Current web search engines use ranking functions based on hundreds of features. Such functions are quite complicated and fairly little has been published about how to efficiently execute them on large collections. One "folklore" approach separates ranking into two phases. In the first phase, a simple and fast ranking function such as BM25 is used to get, say, the top 100 or 1000 documents. Then in the second phase a more involved ranking function with hundreds of features is applied to the top documents returned from the first phase. As the second phase only examines a small number of top candidates, a significant amount of the computation time is still spent on the first phase. In this paper we focus on executing such a simple first-phase function, say BM25, a problem that has been extensively studied in the literature.

当前的网络搜索引擎使用基于数百个特征的排序函数。这类函数相当复杂，而且关于如何在大型文档集合上高效执行这些函数的公开资料非常少。一种“经验性”方法将排序分为两个阶段。在第一阶段，使用诸如BM25（最佳匹配25算法）这样简单快速的排序函数来获取，例如，前100或1000篇文档。然后在第二阶段，将包含数百个特征、更为复杂的排序函数应用于第一阶段返回的顶级文档。由于第二阶段仅考察少量的顶级候选文档，大量的计算时间仍花费在第一阶段。在本文中，我们专注于执行这样一个简单的第一阶段函数，例如BM25，这是一个在文献中已被广泛研究的问题。

Recall that ranked query processing consists of a Boolean filter followed by scoring and ranking the documents that pass this filter. The most commonly used Boolean filters are conjunctive (AND) and disjunctive (OR). In general, disjunctive queries have traditionally been used in the IR community while web search engines have often tried to employ conjunctive queries as much as possible. One reason is that disjunctive queries tend to be significantly (by about an order of magnitude for exhaustive query processing) more expensive than conjunctive queries, as they have to evaluate many more documents.

回顾一下，排序查询处理包括一个布尔过滤器，然后对通过该过滤器的文档进行评分和排序。最常用的布尔过滤器是合取（AND）和析取（OR）。一般来说，信息检索（IR）领域传统上使用析取查询，而网络搜索引擎则经常尽可能多地采用合取查询。其中一个原因是，析取查询的成本往往比合取查询高得多（对于穷举查询处理而言，大约高出一个数量级），因为它们必须评估更多的文档。

To traverse the index structure, there are two basic techniques, Document-At-A-Time (DAAT) and Term-At-A-Time (TAAT) [32]: For conjunctive queries, DAAT is often preferred, while many optimized approaches for disjunctive queries use TAAT.

遍历索引结构有两种基本技术，即按文档处理（Document-At-A-Time，DAAT）和按词项处理（Term-At-A-Time，TAAT） [32]：对于合取查询，通常首选DAAT，而许多针对析取查询的优化方法则使用TAAT。

### 2.3 Early Termination Algorithms

### 2.3 提前终止算法

As discussed before, one bottleneck in query processing is the length of the inverted lists. Early termination is one important technique that addresses this problem. We say that a query processing algorithm is exhaustive if it fully evaluates all documents that satisfy the Boolean filter condition. Any non-exhaustive algorithm is considered to use early termination (ET). There are four ways in which early termination often happens:

如前所述，查询处理中的一个瓶颈是倒排列表的长度。提前终止是解决这一问题的一项重要技术。如果一个查询处理算法对所有满足布尔过滤条件的文档进行了全面评估，我们就称其为穷举算法。任何非穷举算法都被认为使用了提前终止（ET）技术。提前终止通常有以下四种情况：

- Stop early: In this case, the postings are usually arranged such that the most promising documents appear early. Then we stop the traversal of the index as soon as we (may) have the top- $k$ results. Well-known examples are the TA,FA,and NRA algorithms of Fagin [21]; see [8] for a highly optimized implementation of some of these algorithms.

- 提前停止：在这种情况下，通常会对倒排列表进行排序，使最有希望的文档排在前面。这样，一旦（可能）得到了前 $k$ 个结果，我们就停止对索引的遍历。著名的例子有法金（Fagin）[21]提出的TA、FA和NRA算法；关于其中一些算法的高度优化实现，请参阅文献[8]。

- Skip within lists: When the postings in each list are sorted by docIDs, the promising documents are spread out throughout the inverted lists, and thus the standard intuition for "stop early" does not apply. There are few published works on early termination techniques under this scenario. An exception is the WAND algorithm in [11], which uses a smart pointer movement technique to skip many documents that would be evaluated by an exhaustive algorithm. More details are provided further below.

- 列表内跳过：当每个列表中的倒排列表按文档ID排序时，有希望的文档会分散在整个倒排列表中，因此“提前停止”的常规思路并不适用。在这种情况下，关于提前终止技术的已发表研究较少。文献[11]中的WAND算法是一个例外，它使用了一种智能指针移动技术，跳过了许多穷举算法会评估的文档。下面将提供更多细节。

- Omit lists: One or more lists for the query terms are completely ignored, if they do not affect the final results by much.

- 省略列表：如果查询词的一个或多个列表对最终结果影响不大，则会被完全忽略。

- Score only partially: We partially evaluate a document by computing only some term scores, or by computing approximate scores. When we find that the document cannot be in the top results, we stop evaluation; an example is [33].

- 部分计分：我们通过仅计算部分词项得分或计算近似得分来部分评估文档。当发现该文档不可能进入排名靠前的结果时，我们会停止评估；文献[33]给出了一个示例。

Note that our definition of ET is very general and includes other techniques such as static pruning $\left\lbrack  {{10},{19}}\right\rbrack$ and tiering $\left\lbrack  {16}\right\rbrack$ . In this paper we focus on safe early termination [30], which means we want exactly the same results as in the naive baseline, i.e., the same set of documents in the same order with the same scores. We will ignore other techniques, which try to return search results that are somehow similar, or of similar quality. Also, we focus on memory-based indexes,as for example considered in $\left\lbrack  {{30},{17}}\right\rbrack$ ,or at least on the case where disk is not the main bottleneck.

请注意，我们对提前终止（Early Termination，ET）的定义非常宽泛，涵盖了其他技术，如静态剪枝（static pruning $\left\lbrack  {{10},{19}}\right\rbrack$）和分层（tiering $\left\lbrack  {16}\right\rbrack$）。在本文中，我们专注于安全提前终止（safe early termination [30]），这意味着我们希望得到与朴素基线方法完全相同的结果，即相同的文档集合，以相同的顺序和相同的得分呈现。我们将忽略其他试图返回某种程度上相似或质量相近搜索结果的技术。此外，我们专注于基于内存的索引，例如文献$\left\lbrack  {{30},{17}}\right\rbrack$中所考虑的，或者至少关注磁盘不是主要瓶颈的情况。

### 2.4 Index Organizations

### 2.4 索引组织

Many existing techniques for early termination from the DB and IR communities are based on the idea of reorganizing the inverted index such that the most promising documents appear early in the inverted lists. This can be done by either reordering the postings in each list, or partitioning the index into several layers or tiers. In particular, we can distinguish among the following widely used index organizations:

数据库（DB）和信息检索（IR）领域现有的许多提前终止技术都基于这样一种理念：对倒排索引进行重组，使最有希望的文档在倒排列表中尽早出现。这可以通过对每个列表中的倒排记录重新排序，或者将索引划分为多个层来实现。具体而言，我们可以区分以下广泛使用的索引组织方式：

- Document-Sorted Indexes: This is the standard approach for basic exhaustive query processing, where the postings in each inverted list are sorted by document ID.

- 文档排序索引：这是基本的穷举查询处理的标准方法，其中每个倒排列表中的倒排记录按文档ID排序。

- Impact-Sorted Indexes: Postings in each list are sorted by their impact, that is, their contribution to the score of a document. Postings with the same impact are sorted by document ID. Note that this assumes that the ranking function is decomposable (i.e., a sum or other simple combination of per-term scores), which is true for Cosine, BM25, and many other functions in the literature.

- 影响排序索引：每个列表中的倒排记录按其影响进行排序，即它们对文档得分的贡献。具有相同影响的倒排记录按文档ID排序。请注意，这假设排序函数是可分解的（即，是每个词项得分的总和或其他简单组合），这对于余弦相似度、BM25以及文献中的许多其他函数都是成立的。

- Impact-Layered Indexes: We partition the postings in each list into a number of layers,such that all postings in layer $i$ have a higher impact than those in layer $i + 1$ ,and then sort the postings in each layer by docID.

- 影响分层索引：我们将每个列表中的倒排记录划分为多个层，使得层 $i$ 中的所有倒排记录的影响都高于层 $i + 1$ 中的倒排记录，然后按文档ID对每个层中的倒排记录进行排序。

Impact-sorted and impact-layered indexes are very popular index organizations for early termination techniques, as they place the most promising postings close to the start of the lists $\lbrack {25},{21},3$ , $4,8,{30},{23},{32}\rbrack$ . A problem with impact-sorted indexes is that compression could suffer as docID gaps in the inverted lists may be very large. In this case, an impact-layered index that uses a small number of appropriately chosen layers may provide a better alternative. However, impact-sorted indexes are useful when the number of distinct impact scores is small, or frequencies are used as proxies for impacts.

按影响排序和按影响分层的索引是早期终止技术中非常流行的索引组织方式，因为它们将最有希望的倒排列表项置于列表开头附近 $\lbrack {25},{21},3$ ， $4,8,{30},{23},{32}\rbrack$ 。按影响排序的索引存在一个问题，即压缩效果可能不佳，因为倒排列表中的文档ID间隔可能非常大。在这种情况下，使用少量适当选择的层的按影响分层索引可能是更好的选择。然而，当不同影响得分的数量较少，或者使用词频作为影响的代理时，按影响排序的索引很有用。

In contrast, document-sorted indexes tend to be less studied for early termination techniques, and only few algorithms use them (e.g., [11]).

相比之下，按文档排序的索引在早期终止技术方面的研究较少，只有少数算法会使用它们（例如 [11]）。

### 2.5 Index Traversal Techniques

### 2.5 索引遍历技术

For index traversal, the two most commonly used techniques are:

对于索引遍历，最常用的两种技术是：

- Document-At-A-Time (DAAT) : In DAAT query processing, each list has a pointer that points to a "current" posting in the list. All the pointers move forward in parallel as the query is being processed.

- 按文档逐个处理（Document-At-A-Time，DAAT）：在DAAT查询处理中，每个列表都有一个指针，指向列表中的“当前”倒排列表项。在处理查询时，所有指针并行向前移动。

- Term-At-A-Time (TAAT): In TAAT query processing, we first access one term, or one layer from one term, and then move to the next term, or the next layer from the same term or a different term. We use a temporary data structure to keep track of currently active top- $k$ candidates.

- 逐词处理（Term-At-A-Time，TAAT）：在TAAT查询处理中，我们首先处理一个词项，或一个词项的一层，然后再处理下一个词项，或同一个词项或不同词项的下一层。我们使用一个临时数据结构来跟踪当前活跃的前 $k$ 候选对象。

Note that TAAT requires additional data structures to store promising candidates seen in some but not all of the lists; this is one of the main differences to DAAT. In this paper we use TAAT to refer to any techniques that use nontrivial data structures to keep track of promising candidates (beyond the simple heap structure used for the current top- $k$ results),and thus this includes the original Term-At-A-Time technique [13] as well as Score-At-A-Time in [3].

请注意，TAAT需要额外的数据结构来存储在部分而非所有列表中出现的有潜力的候选对象；这是它与逐文档处理（DAAT）的主要区别之一。在本文中，我们使用TAAT来指代任何使用非平凡数据结构（超出用于当前前 $k$ 结果的简单堆结构）来跟踪有潜力的候选对象的技术，因此这包括原始的逐词处理技术 [13] 以及文献 [3] 中的逐得分处理（Score-At-A-Time）技术。

For conjunctive and exhaustive query execution, DAAT is very fast and considered state of the art (at least for queries with a moderate number of queries terms), whereas TAAT-type methods are often bottlenecked by the nontrivial data structures. However, for disjunctive queries it is hard to integrate early termination and exploit layered indexes with DAAT. Thus, for this case most early termination algorithms in the literature are based on TAAT that use impact-sorted or layered indexes. In this paper we challenge this assumption and suggest DAAT algorithms may actually do better for early termination even in the case of disjunctive queries. We note that DAAT does have a significant advantage by not having any expensive temporary data structures.

对于合取和穷举查询执行，按文档求与（DAAT，Document-at-a-Time）方法非常快，被认为是最先进的技术（至少对于查询词数量适中的查询而言），而按词求与（TAAT，Term-at-a-Time）类型的方法往往会受到非平凡数据结构的瓶颈限制。然而，对于析取查询，使用DAAT方法很难实现提前终止并利用分层索引。因此，在这种情况下，文献中大多数提前终止算法都基于使用影响排序或分层索引的TAAT方法。在本文中，我们对这一假设提出质疑，并认为即使在析取查询的情况下，DAAT算法实际上在提前终止方面可能表现得更好。我们注意到，DAAT方法的一个显著优势是无需使用任何昂贵的临时数据结构。

### 2.6 Two State-of-the-Art Techniques

### 2.6 两项最先进的技术

For disjunctive queries, the fastest existing safe early termination techniques appear to be the approach by Strohman and Croft (SC) in [30] (based on earlier work in $\left\lbrack  {3,2}\right\rbrack$ ),and the WAND approach by Broder et al [11]. Both approaches are safe in that they return exactly the same top- $k$ results as the baseline in the same order and with the same score (actually, the original SC algorithm may not return the same scores, but is easily extended to do so).

对于析取查询，现有的最快的安全早期终止技术似乎是斯特罗曼（Strohman）和克罗夫特（Croft）（SC）在文献[30]中提出的方法（基于文献$\left\lbrack  {3,2}\right\rbrack$中的早期工作），以及布罗德（Broder）等人在文献[11]中提出的WAND方法。这两种方法都是安全的，因为它们返回的前$k$个结果与基线方法完全相同，顺序相同且得分相同（实际上，原始的SC算法可能不会返回相同的得分，但可以很容易地进行扩展以实现这一点）。

The approach by Strohman and Croft (SC) uses impact-sorted indexes and assumes a ranking function proposed in [3], where all impacts have one of 8 distinct values; however, it can be extended to ranking functions such as BM25 using an impact-layered index organization. The approach then uses a variation of Term-At-A-Time (TAAT) query processing where we first access the higher layers of the lists and then move to the lower layers with smaller impact scores; when we are guaranteed to have a set containing the top-k results,we will switch from OR mode to AND mode,by only searching for the docIDs already stored in the structure among the remaining layers, to find the final correct results. Figure 1 shows one example of the index layout for SC, with equal-size layers. Note that the original SC applies a different ranking technique, so the size of each layer varies. We explore this and find that it does not make a big difference whether we use variable-size or equal-size layers; we will explain this further below. In SC, a sorted array is used to keep track of candidate documents that have been seen in some but not all term lists. After processing each layer, an extra phase is used to "filter out" candidates that can not make into the top- $k$ . Thus,this approach requires temporary data structures (arrays) to keep track of promising candidates.

斯特罗曼（Strohman）和克罗夫特（Croft）（SC）的方法使用按影响排序的索引，并采用了文献[3]中提出的排名函数，其中所有影响值有8种不同取值之一；不过，它可以通过使用分层影响索引结构扩展到诸如BM25之类的排名函数。该方法随后采用了逐词（Term-At-A-Time，TAAT）查询处理的一种变体，即我们首先访问列表的高层，然后再转向影响得分较小的低层；当我们确定已经得到包含前k个结果的集合时，我们将从“或”（OR）模式切换到“与”（AND）模式，即仅在剩余层中搜索已存储在该结构中的文档ID，以找到最终的正确结果。图1展示了SC的索引布局示例，各层大小相等。请注意，原始的SC方法采用了不同的排名技术，因此每层的大小各不相同。我们对此进行了研究，发现使用可变大小层还是等大小层并没有太大区别；我们将在下面进一步解释这一点。在SC方法中，使用一个排序数组来跟踪那些在部分但并非所有词项列表中出现过的候选文档。处理完每一层后，会有一个额外的阶段用于“过滤掉”无法进入前$k$的候选文档。因此，这种方法需要临时数据结构（数组）来跟踪有潜力的候选文档。

The WAND approach, on the other hand, uses a standard document-sorted index, and can thus employ a Document-At-A-Time (DAAT) approach that does not require additional temporary data structures (apart from the small top- $k$ heap used by all methods). The downside is that the promising documents are spread out throughout the inverted lists, and thus the standard intuition for early termination does not apply. Instead, WAND uses an ingenious pointer movement strategy based on pivoting that allows it to skip many documents that would be evaluated by an exhaustive algorithm. More precisely, in DAAT query processing each list has a pointer that points to a "current" posting in the list, and that moves forward as the query is being processed. Thus any posting to the left of the pointers have already been processed. Throughout the algorithm, WAND keeps the terms sorted in increasing order by their current docIDs. Assume that at some point during a query "dog, cat, kangaroo, monkey", the current docIDs are 609, 273, 9007, and 4866, respectively, as shown in Figure 2, where the lists are arranged from top to bottom according to these docIDs. Suppose also that we know the maximum impact score for each list, and that a total document score of at least 6.8 (the threshold) is needed in order to make it into the current top- $k$ results. We now sum up the maximum scores of the lists from top to bottom until we reach a score no smaller than 6.8. In Figure 2, this happens at the third list from the top $\left( {{2.3} + {1.8} + {3.3} > {6.8}}\right)$ . We can now claim that the smallest docID that can make it into the top- $k$ is 4866 . Thus,we can move the top two pointers forward to the first postings in their lists with docIDs at least 4866, enabling skipping in these lists. If docID 4866 appears in both the first two lists then we evaluate this docID. Otherwise, we sort the lists according to the current docIDs and pivot again. Thus, WAND achieves early termination by enabling skips over postings that cannot make into the top results. For the threshold value, we use the lowest score in the heap that contains the top- $k$ results found thusfar. Note that WAND also stores one maximum impact score for each list, which could be kept in the term dictionary of the index.

另一方面，WAND（弱AND）方法使用标准的按文档排序的索引，因此可以采用一次处理一个文档（DAAT）的方法，该方法不需要额外的临时数据结构（除了所有方法都使用的小型前$k$堆）。其缺点是有潜力的文档分散在整个倒排列表中，因此标准的提前终止直觉并不适用。相反，WAND使用了一种基于枢轴的巧妙指针移动策略，使其能够跳过许多会被穷举算法评估的文档。更准确地说，在DAAT查询处理中，每个列表都有一个指针，指向列表中的“当前”记录项，并且该指针会在查询处理过程中向前移动。因此，指针左侧的任何记录项都已经被处理过。在整个算法过程中，WAND会根据当前文档ID（docID）按升序对词项进行排序。假设在查询“狗、猫、袋鼠、猴子”的某个时刻，当前的文档ID分别为609、273、9007和4866，如图2所示，其中列表根据这些文档ID从上到下排列。假设我们还知道每个列表的最大影响得分，并且为了进入当前的前$k$结果，文档的总得分至少需要达到6.8（阈值）。我们现在从上到下对列表的最大得分进行求和，直到达到一个不小于6.8的得分。在图2中，这发生在从上往下数的第三个列表$\left( {{2.3} + {1.8} + {3.3} > {6.8}}\right)$处。我们现在可以声称，能够进入前$k$的最小文档ID是4866。因此，我们可以将前两个指针向前移动到其列表中文档ID至少为4866的第一个记录项，从而实现这些列表中的跳过操作。如果文档ID 4866同时出现在前两个列表中，那么我们就评估这个文档ID。否则，我们根据当前文档ID对列表进行排序，然后再次进行枢轴操作。因此，WAND通过跳过那些无法进入前几名结果的记录项来实现提前终止。对于阈值，我们使用到目前为止找到的前$k$结果所在堆中的最低得分。请注意，WAND还为每个列表存储一个最大影响得分，该得分可以保存在索引的词项字典中。

These two methods are different in interesting ways - index organization and index traversal choices: WAND uses a document-sorted index and DAAT as index traversal technique, whereas SC uses an impact-ordered or layered index and TAAT, using an additional data structure to keep the candidates. For disjunctive queries SC seems to outperform WAND, although the numbers for SC that we report are not as fast as those in [30], due to our use of BM25 as the ranking function and due to us keeping stopwords and removing 1-term queries. WAND is of interest to us as it will form the basis of our improved approaches.

这两种方法在索引组织和索引遍历选择方面存在有趣的差异：WAND（加权和动态修剪，Weighted AND）方法使用按文档排序的索引，并采用文档逐个处理（DAAT，Document-At-A-Time）作为索引遍历技术；而SC（得分组合，Score Combination）方法使用按影响排序或分层的索引，并采用词项逐个处理（TAAT，Term-At-A-Time），同时使用额外的数据结构来保存候选结果。对于析取查询，SC似乎比WAND表现更优。不过，我们报告的SC的相关数据不如文献[30]中的快，这是因为我们使用了BM25作为排序函数，并且保留了停用词并移除了单词查询。WAND对我们来说很有意义，因为它将成为我们改进方法的基础。

<!-- Media -->

<!-- figureText: Switch from OR Max $= {3.3}$ Max $= {0.9}$ Max $= {10.5}$ Max $= {7.9}$ Layer 2 Layer 3 Max $= {30.1}$ Max $= {10.2}$ apple Max $= {24.8}$ Max $= {13.0}$ pear Layer 0 Layer 1 OR phase -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_3.jpg?x=187&y=396&w=637&h=290&r=0"/>

Figure 1: The index layout for SC. The layers are processed in descending order by their maximum impact scores. Inside each layer, postings are sorted by docIDs.

图1：SC的索引布局。各层按其最大影响得分降序处理。在每层内部，倒排列表按文档ID排序。

<!-- figureText: current threshold $= {6.8}$ $\max  = {2.3}$ max = 1.8 max = 3.3 max = 4.3 cat dog monkey kangaroo -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_3.jpg?x=184&y=859&w=642&h=328&r=0"/>

Figure 2: A scenario during the processing of a 4-term query, where the current pointers point to docIDs 273, 609, 4866, and 9007. WAND selects the third list as a pivot, and moves earlier pointers to docID 4866. Then all lists are sorted again according to their current docIDs.

图2：处理一个包含4个词项的查询时的场景，当前指针分别指向文档ID 273、609、4866和9007。WAND选择第三个列表作为枢轴，并将较早的指针移动到文档ID 4866。然后，所有列表再根据其当前文档ID重新排序。

<!-- Media -->

### 3.OUR CONTRIBUTION

### 3. 我们的贡献

In this paper, we propose new early termination algorithms by building on the WAND approach [11]. Recall that WAND stores the maximum impact for each inverted list. Our initial insight is that skipping in WAND is limited because it uses the maximum impact scores over the entire lists, which can be much larger than average. Recall that we have an extra table to store information allowing us to skip blocks. We propose to augment the index structure by also storing in this table the maximum impact value for each block. We call such an index a Block-Max Index. In this way, we get a piece-wise upper-bound approximation of the impact scores in the lists, as shown in Figure 3. This approximation hides the detailed scores, shown in Figure 3 for one block in the kangaroo list, which can only be obtained by decompressing the block. This idea is very simple and easy to implement. As we will see later, this gives many optimization opportunities and leads to large performance gains.

在本文中，我们基于宽阈跳跃（WAND，Wide-bounded AND）方法[11]提出了新的提前终止算法。回顾一下，宽阈跳跃方法会存储每个倒排列表的最大影响值。我们最初的见解是，宽阈跳跃方法中的跳过操作存在局限性，因为它使用的是整个列表的最大影响得分，而这些得分可能远高于平均值。我们有一个额外的表来存储信息，以便我们能够跳过数据块。我们提议通过在这个表中同时存储每个数据块的最大影响值来扩充索引结构。我们将这样的索引称为块最大索引（Block-Max Index）。通过这种方式，我们可以得到列表中影响得分的分段上界近似值，如图3所示。这种近似隐藏了详细得分，图3展示了袋鼠列表中一个数据块的详细得分，这些得分只能通过解压缩该数据块来获得。这个想法非常简单且易于实现。正如我们稍后将看到的，这提供了许多优化机会，并带来了显著的性能提升。

After this slight change to the index structure, resulting in only a small increase in index size, we have to adapt the WAND algorithm to work with it. One obvious idea is to just use the local maximum value for the current block, instead of the global one, in the pivoting phase. Unfortunately, this does not guarantee correctness. To see this, let us look at the example in Figure 4. Looking only at the max scores for the blocks containing the current pointers (i.e., score

在对索引结构进行这一微小更改后（这仅导致索引大小略有增加），我们必须调整WAND算法以使其能与之配合使用。一个显而易见的想法是，在枢轴阶段使用当前块的局部最大值，而非全局最大值。遗憾的是，这并不能保证正确性。为了说明这一点，让我们来看图4中的示例。仅查看包含当前指针的块的最大得分（即得分

<!-- Media -->

<!-- figureText: docID space -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_3.jpg?x=999&y=154&w=559&h=354&r=0"/>

Figure 3: Three inverted lists where lists are piecewise upper-bounded by the maximum scores in each block. As shown for one block in the bottom list, inside each block we have various values, including many (implied) zero values, that can be retrieved by decompressing the block.

图3：三个倒排列表，其中每个列表在每个块中都由最大得分分段上界约束。如底部列表中的一个块所示，在每个块内部，我们有各种值，包括许多（隐含的）零值，这些值可以通过对块进行解压缩来获取。

<!-- figureText: cat current threshold $= {6.8}$ Block max = 4.3 Block max = 1.8 dog Block max $= {3.3}\vdots  \downarrow$ monkey kangaroo -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_3.jpg?x=1007&y=697&w=550&h=337&r=0"/>

Figure 4: An example showing why directly using block max scores does not work.

图4：一个示例，展示了为什么直接使用块的最大得分行不通。

<!-- Media -->

2.3 for the first list, and so on), we cannot conclude that 4866 is the smallest docID that can make into it the top- $k$ ,because it is possible that the next block after docID 273 in the first list (but with docIDs smaller than 4866) has a much higher maximum impact score. Thus, directly applying the local max value does not work. We will describe how to modify the algorithm in later sections.

对于第一个列表是2.3，依此类推），我们不能得出4866是能进入前$k$的最小文档编号（docID），因为有可能第一个列表中文档编号273之后的下一个块（但文档编号小于4866）具有更高的最大影响分数。因此，直接应用局部最大值是行不通的。我们将在后续章节中描述如何修改该算法。

Overall, we make the following main contributions in this paper:

总体而言，本文做出了以下主要贡献：

1. We propose a modified index structure, the Block-Max Index, which only slightly increases the index size. We then study improved techniques for safe early termination based on this index structure and the WAND approach in [11].

1. 我们提出了一种改进的索引结构，即块最大索引（Block - Max Index），它仅略微增加了索引大小。然后，我们基于此索引结构和文献[11]中的WAND方法研究了用于安全提前终止的改进技术。

2. We show how to extend our techniques to layered indexes, reordered indexes, and conjunctive query processing.

2. 我们展示了如何将我们的技术扩展到分层索引、重新排序的索引和联合查询处理。

3. We evaluate our techniques on the TREC GOV2 collection of 25.2 million documents, and demonstrate considerable improvements compared to the state-of-the-art techniques.

3. 我们在包含2520万篇文档的TREC GOV2数据集上评估了我们的技术，并证明与现有技术相比有显著改进。

4. We discuss some interesting open questions resulting from our work.

4. 我们讨论了由我们的工作引发的一些有趣的开放性问题。

## 4. RELATED WORK

## 4. 相关工作

Previous work on early termination (ET) techniques can be divided into two fairly disjoint sets of literature. In the IR community, researchers have studied ET techniques for the fast evaluation of vector space queries since the 1980s; some early work appears in $\left\lbrack  {{13},{32},{33}}\right\rbrack$ . There have been a large number of papers in recent years. Most relevant to us, several recent papers have focused on how to use impact-sorted indexes $\left\lbrack  {2,3,{30}}\right\rbrack$ for early termination, resulting in highly efficient methods for disjunctive queries.

先前关于提前终止（ET）技术的研究可大致分为两类互不相关的文献。在信息检索（IR）领域，自20世纪80年代以来，研究人员就开始研究用于快速评估向量空间查询的提前终止技术；一些早期研究成果见文献$\left\lbrack  {{13},{32},{33}}\right\rbrack$。近年来相关论文数量众多。与我们的研究最为相关的是，近期有几篇论文聚焦于如何利用影响排序索引$\left\lbrack  {2,3,{30}}\right\rbrack$实现提前终止，从而为析取查询提供了高效的方法。

There has also been a lot of work on early termination in the database community; see [21] for a survey and [20] for a formal analysis. Stated in IR terms, the algorithms also assume that postings in the inverted lists are sorted by their contributions and accessed in sorted order. However, the application scenarios are somewhat different, and many (but not all) of the algorithms assume that once a document is found in one inverted list, we can efficiently evaluate it by performing lookups into the other inverted lists. Such random lookups are highly undesirable in most IR scenarios.

数据库领域也有很多关于提前终止（early termination）的研究；相关综述可参考[21]，形式化分析可参考[20]。用信息检索（IR）术语来说，这些算法同样假设倒排表中的记录按其贡献排序，并按排序顺序访问。然而，应用场景有所不同，而且许多（并非全部）算法假设，一旦在一个倒排表中找到某篇文档，就可以通过在其他倒排表中进行查找来高效地对其进行评估。在大多数信息检索场景中，这种随机查找是非常不可取的。

Early termination techniques also differ in terms of their assumptions about result quality. We can distinguish between safe (or reliable) early termination techniques that return exactly the same top- $k$ results as the baseline [30,20],techniques that return mostly the same results, and those that just return results of equivalent quality, as determined by suitable IR measures. We focus on safe early termination of disjunctive queries, where the most relevant previous techniques are the approach of Strohman and Croft [30] and the WAND approach of Broder et al. in [11].

提前终止技术在对结果质量的假设方面也存在差异。我们可以区分出安全（或可靠）的提前终止技术，这些技术返回的前$k$个结果与基线方法完全相同[30,20]；还有返回的结果大多相同的技术；以及那些根据合适的信息检索指标确定返回等效质量结果的技术。我们专注于析取查询的安全提前终止，其中最相关的先前技术是斯特罗曼（Strohman）和克罗夫特（Croft）的方法[30]，以及布罗德（Broder）等人在文献[11]中提出的WAND方法。

Two other recent ideas in IR query processing are also relevant to our work. First, a number of recent papers show how to decrease inverted index size and query processing costs by optimizing the assignment of docIDs to the documents in the collection [26, 9]. Intuitively, if we assign consecutive docIDs to very similar pages, for example by sorting pages by URL [28] or clustering by textual similarity, we obtain runs of small docIDs gaps that allow better index compression with suitable techniques. Moreover, as shown in [35], reordering significantly increases the speed of conjunctive queries. Our work here shows that, somewhat surprising to us, reordering can help even more for disjunctive queries.

信息检索（IR）查询处理领域近期的另外两个思路也与我们的工作相关。首先，近期有若干论文展示了如何通过优化文档集合中文档的文档编号（docID）分配来减小倒排索引的大小并降低查询处理成本 [26, 9]。直观地说，如果我们将连续的文档编号分配给非常相似的页面，例如通过按统一资源定位符（URL）对页面进行排序 [28] 或按文本相似度进行聚类，我们就能得到一系列文档编号差距较小的区间，从而可以使用合适的技术对索引进行更好的压缩。此外，如文献 [35] 所示，重新排序能显著提高合取查询的速度。我们在此的研究表明，有些出乎我们意料的是，重新排序对析取查询的帮助更大。

The second relevant idea is that of two-level indexes proposed in [1]. The idea is to cluster documents by similarity and then in the first level only index the clusters (i.e., whether a cluster contains a term), while the second level says which documents in the cluster actually contain the term. Thus the first level basically approximates the overall index, similar to the way in which we use the maximum impact scores in each block to approximate the distribution of impact scores in a list.

第二个相关思路是文献 [1] 中提出的两级索引。其思路是按相似度对文档进行聚类，然后在第一级仅对聚类进行索引（即，一个聚类是否包含某个词项），而第二级则指出聚类中哪些文档实际包含该词项。因此，第一级基本上是对整个索引的近似，这与我们使用每个块中的最大影响得分来近似列表中影响得分分布的方式类似。

Finally, we note that very recently, and independent of our work, Kaushik et al [15] have proposed an index structure very similar to the Block-Max Index in this paper, which also stores maximum impact information for blocks. Their algorithm for disjunctive queries first performs preprocessing to split blocks into intervals with aligned boundaries and to discard intervals that cannot contain any top results. Then a version of the maxScore technique [32] is applied to the remaining intervals. While their and our algorithms are different, they both achieve significant performance improvements based on similar underlying ideas.

最后，我们注意到，就在最近，与我们的工作无关，考希克等人 [15] 提出了一种与本文中的块最大索引（Block-Max Index）非常相似的索引结构，该结构同样存储了块的最大影响信息。他们针对析取查询的算法首先进行预处理，将块分割成边界对齐的区间，并舍弃那些不可能包含任何顶级结果的区间。然后，将最大得分（maxScore）技术 [32] 的一个版本应用于剩余的区间。虽然他们的算法和我们的算法有所不同，但基于相似的底层思想，两者都实现了显著的性能提升。

## 5. BLOCK-MAX WAND ALGORITHM

## 5. 块最大魔杖（BLOCK-MAX WAND）算法

In this section we give our basic algorithm, Block-Max WAND (BMW),which is an extension of WAND to our Block-Max Index.

在本节中，我们将介绍我们的基本算法——块最大魔杖（Block-Max WAND，BMW）算法，它是魔杖（WAND）算法在我们的块最大索引上的扩展。

### 5.1 The Basic Idea

### 5.1 基本思想

As described in Section 2, naively using the maximum impact score for each block in the "pivoting" phase will not work, and thus we need to add some additional ideas. In the traditional DAAT query processing,one core function is called $\operatorname{Next}\left( {d,\operatorname{list}\left( i\right) }\right)$ or $\operatorname{Next}$ - ${GEQ}\left( {d,\operatorname{list}\left( i\right) }\right) \left\lbrack  {11}\right\rbrack$ ; this function receives a docID $d$ and an inverted list ${list}\left( i\right)$ as inputs and returns the first docID after the current docID in list(i) that is equal to or greater than $d$ . The call to this particular function usually involves a decompression of one block in list(i). We call this a deep pointer movement due to the reason that it usually involves a block decompression. As we have the max score for each block, we design another function called NextShallow $\left( {d,\operatorname{list}\left( i\right) }\right)$ which only moves the current pointer to the corresponding block without decompression (using d and information about the block boundaries in the table). We call this a shallow pointer movement. We use two main ideas in our modified algorithm: (i) we use the global maximum scores to determine a candidate pivot, as in WAND, but then use the block maximum scores to check if the candidate pivot is a real pivot, and (ii) we use shallow instead of deep pointer movements whenever possible.

如第2节所述，在“枢轴”阶段简单地为每个块使用最大影响得分是行不通的，因此我们需要添加一些额外的思路。在传统的DAAT（按文档编号交替遍历，Distributed and Asynchronous Access to Tables）查询处理中，有一个核心函数称为$\operatorname{Next}\left( {d,\operatorname{list}\left( i\right) }\right)$或$\operatorname{Next}$ - ${GEQ}\left( {d,\operatorname{list}\left( i\right) }\right) \left\lbrack  {11}\right\rbrack$；该函数接收一个文档ID $d$和一个倒排列表${list}\left( i\right)$作为输入，并返回列表(i)中当前文档ID之后第一个等于或大于$d$的文档ID。调用这个特定的函数通常涉及对列表(i)中一个块的解压缩。由于它通常涉及块解压缩，我们将此称为深度指针移动。由于我们有每个块的最大得分，我们设计了另一个名为NextShallow $\left( {d,\operatorname{list}\left( i\right) }\right)$的函数，该函数仅将当前指针移动到相应的块，而无需解压缩（使用d和表中关于块边界的信息）。我们将此称为浅指针移动。我们在改进的算法中使用了两个主要思路：(i) 与WAND（弱AND，Weak AND）算法一样，我们使用全局最大得分来确定候选枢轴，然后使用块最大得分来检查候选枢轴是否为真正的枢轴；(ii) 只要有可能，我们就使用浅指针移动而非深度指针移动。

### 5.2 The Algorithm

### 5.2 算法

The detailed algorithm is shown in Algorithm 1, and we refer to it as Block-Max WAND (BMW). Note that in BMW we still also keep the maximum score for the whole list as in WAND. based on local max values.

详细算法如算法1所示，我们将其称为块最大WAND（Block - Max WAND，BMW）。请注意，在BMW中，我们仍然像WAND算法一样，基于局部最大值保留整个列表的最大得分。

<!-- Media -->

Initialize(   );

初始化(   );

---

repeat

	/* sort the lists by current docIDs */

	Sort(lists);

	/* same "pivoting" as in WAND using the max

	impact for the whole lists, use p to denote

	the pivot */

	$p = \operatorname{Pivoting}\left( {\text{ lists },\theta }\right)$ ;

	$d =$ lists $\left\lbrack  p\right\rbrack   \rightarrow$ curDoc $;$

	if $\left( {d =  = {MAXDOC}}\right)$ then

		break;

	end

	for $i = 0\ldots p + 1$ do

		NextShallow(d, list(i));

	end

	${flag} = {CheckBlockMax}\left( {\theta ,p}\right)$ ;

	if (flag == true) then

		if ( lists $\left\lbrack  0\right\rbrack   \rightarrow$ curDoc == $d$ ) then

				EvaluatePartial(d, p);

				Move all pointers from lists[0] to lists[p] by calling

				Next(list, $d + 1$ )

		end

		else

				Choose one list from the lists before lists[p] with the

				largest IDF,move it by calling Next(list, $d + 1$ )

		end

	end

	else

		${d}^{\prime } =$ GetNewCandidate(   );

		Choose one list from the lists before and including lists[p]

		with the largest IDF,move it by calling Next(list, ${d}^{\prime }$ )

	end

until Stop;

---

Algorithm 1: Block-Max WAND for disjunctive query processing

算法1：用于析取查询处理的块最大WAND算法

<!-- Media -->

As shown in Algorithm 1, the main difference compared with WAND is that before we evaluate one docID, we will first move shallow pointers to check if we indeed have to evaluate this docID or not, based on the maximum scores for blocks. By doing this we filter out most of the candidates and achieve much faster query processing. Also, when the check fails, we can skip further forward using GetNewCandidate(   ), as described later in detail.

如算法1所示，与WAND算法相比，主要区别在于，在评估一个文档ID（docID）之前，我们会首先移动浅层指针，根据块的最大得分来检查是否确实需要评估这个文档ID。通过这样做，我们过滤掉了大部分候选对象，实现了更快的查询处理。此外，当检查失败时，我们可以使用GetNewCandidate(   )函数进一步向前跳过，具体细节将在后面描述。

The two functions used in Algorithm 1, NextShallow(   ) and Check-BlockMax(   ), are listed in Algorithm 2 and Algorithm 3. They are fairly obvious from the context. In EvaluatePartial(   ), we evaluate the document by summing up the scores from list[0] until list[pivot+1]. As soon as we find that the document can not make it into the top results, we stop the evaluation.

算法1中使用的两个函数NextShallow( )和Check - BlockMax( )，分别列于算法2和算法3中。从上下文来看，它们的含义相当明显。在EvaluatePartial( )函数中，我们通过累加从list[0]到list[pivot + 1]的得分来评估文档。一旦发现该文档无法进入排名靠前的结果，我们就停止评估。

Another important improvement happens when CheckBlockMax(   ) returns false,which means the current document $d$ can not make it into the top results. Instead of picking one list (usually the one with the largest IDF) and moving it forward to at least $d + 1$ ,we will use the ${d}^{\prime }$ returned by GetNewCandidate(   ). The reason is that since the current document was ruled out based on the block maxima, we should skip at least beyond the end of one of the current blocks. This idea behind GetNewCandidate(   ) is shown in Figure 5. Assume docID 266 is the pivot; when it fails the CheckBlockMax(   ) check,instead of moving one of the first three lists to ${266} + 1$ ,we will move it to ${d}^{\prime } = \min \left( {{d1},{d2},{d3},{d4}}\right)$ where ${d1},{d2},{d3}$ are the block boundaries plus one of the first three lists,and ${d4}$ is the current docID in the forth list (equal to 1807 in this case). By doing this,skipping is greatly improved compared to using $d + 1$ ,while still guaranteeing a safe result. The proof should be obvious.

当CheckBlockMax( )返回false时，会出现另一个重要的改进，这意味着当前文档$d$无法进入顶级结果。我们不会选择一个列表（通常是逆文档频率（IDF）最大的列表）并将其向前移动至少$d + 1$，而是会使用GetNewCandidate( )返回的${d}^{\prime }$。原因是，由于当前文档基于块最大值被排除，我们至少应该跳过当前某个块的末尾。GetNewCandidate( )背后的思路如图5所示。假设文档ID 266是枢轴；当它未通过CheckBlockMax( )检查时，我们不会将前三个列表中的一个移动到${266} + 1$，而是会将其移动到${d}^{\prime } = \min \left( {{d1},{d2},{d3},{d4}}\right)$，其中${d1},{d2},{d3}$是块边界加上前三个列表中的一个，${d4}$是第四个列表中的当前文档ID（在这种情况下等于1807）。通过这样做，与使用$d + 1$相比，跳过操作得到了极大的改进，同时仍能保证结果的安全性。证明过程应该很明显。

<!-- Media -->

<!-- figureText: monkey kangaroo -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_5.jpg?x=184&y=156&w=642&h=314&r=0"/>

Figure 5: An example showing how GetNewCandidate(   ) works. Assume 266 is the pivot and it fails to make it into the top results. In this case,we enable better skipping by choosing $\min \left( {{d1},{d2},{d3},{d4}}\right)$ as the next possible candidate,instead of ${266} + 1$

图5：展示GetNewCandidate( )工作原理的示例。假设266是枢轴（pivot），且它未能进入顶级结果。在这种情况下，我们通过选择$\min \left( {{d1},{d2},{d3},{d4}}\right)$作为下一个可能的候选对象来实现更好的跳过操作，而非选择${266} + 1$

<!-- Media -->

while did $>$ list- $>$ blockboundary[current_block] do current_block $+  +$ ; end

当$>$列表 - $>$块边界[current_block]成立时，执行current_block $+  +$；结束

Algorithm 2: NextShallow(list, did)

算法2：NextShallow（列表，文档ID）

<!-- Media -->

---

maxposs $= {0.0f}$ ;

for $i = 0\ldots$ pivot +1 do

		maxposs $+  =$ list $\left\lbrack  i\right\rbrack   -  >$ blockmax $\left\lbrack  \text{current_block}\right\rbrack$ ;

end

	( maxposs > threshold ) then

		return true;

end

else

		return false;

end

---

Algorithm 3: CheckBlockMax(threshold, pivot)

算法3：CheckBlockMax（阈值，枢轴）

<!-- Media -->

## 6. EXPERIMENTS

## 6. 实验

In this section, we provide a first set of experimental results.

在本节中，我们给出第一组实验结果。

### 6.1 Experimental Setup

### 6.1 实验设置

We evaluate our methods on the TREC GOV2 collection. The GOV2 collection consists of 25.2 million web pages crawled from the gov Internet domain. The uncompressed size of these web pages is ${426}\mathrm{{GB}}$ . We compress the inverted index using the New-PFD version of PForDelta as described in [35], with 64 docIDs and frequencies in each block.(We also tried other block sizes, but this one gave the best results.) The compressed index consumes 8759MB and the extra information for the Max-Block Index (the maximum score for each block) adds about ${400}\mathrm{{MB}}$ (using 32 bits for each score though this could be reduced).

我们在TREC GOV2数据集（TREC GOV2 collection）上评估我们的方法。GOV2数据集包含从.gov互联网域名抓取的2520万个网页。这些网页的未压缩大小为${426}\mathrm{{GB}}$。我们使用如文献[35]中所述的PForDelta的New - PFD版本来压缩倒排索引，每个块中有64个文档ID（docIDs）和频率。（我们也尝试了其他块大小，但这个大小给出了最佳结果。）压缩后的索引占用8759MB，最大块索引（Max - Block Index）的额外信息（每个块的最大得分）大约增加了${400}\mathrm{{MB}}$（每个得分使用32位，不过这个位数可以减少）。

We randomly picked 1000 queries from the TREC 2006 Efficiency queries, and 1000 queries from the TREC 2005 Efficiency queries as our testing sets. The average numbers of postings per query are ${4.67}\mathrm{M}$ and ${6.07}\mathrm{M}$ using 2006 and 2005 set,respectively. We use BM25 as our ranking function. In all our runs, we load the inverted index completely into main memory. Unless stated otherwise, we return top-10 results. Runs are performed on a single core of a ${2.27}\mathrm{{GHz}}$ Intel(R) Xeon CPU. All the codes are available by contacting the first author.

我们从TREC 2006效率查询集（TREC 2006 Efficiency queries）中随机选取了1000个查询，从TREC 2005效率查询集（TREC 2005 Efficiency queries）中随机选取了1000个查询作为测试集。使用2006年和2005年的数据集时，每个查询的平均倒排列表项数量分别为${4.67}\mathrm{M}$和${6.07}\mathrm{M}$。我们使用BM25作为排序函数。在所有实验中，我们将倒排索引完全加载到主内存中。除非另有说明，我们返回前10个结果。实验在${2.27}\mathrm{{GHz}}$英特尔（Intel）至强（Xeon）CPU的单核上进行。所有代码可通过联系第一作者获取。

<!-- Media -->

<!-- figureText: average processing time for SC with different $\#$ of layers -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_5.jpg?x=998&y=152&w=559&h=275&r=0"/>

Figure 6: Processing time in ms for different number of layers in SC.

图6：SC中不同层数的处理时间（毫秒）。

<!-- Media -->

### 6.2 Results

### 6.2 结果

In this section we compare our algorithm BMW with exhaustive OR (using DAAT), WAND, and SC on disjunctive queries. We measure the performance by three criteria - time (average time per query in ms), decoded integers per query and evaluated docIDs (do-cIDs that are completely scored against all query terms) per query. These criteria are also used in previous work $\left\lbrack  {{30},{11}}\right\rbrack$ .

在本节中，我们将我们的算法BMW与析取查询上的穷举OR（使用DAAT）、WAND和SC算法进行比较。我们通过三个标准来衡量性能——时间（每个查询的平均时间，单位为毫秒）、每个查询解码的整数数量以及每个查询评估的文档ID（针对所有查询词完全评分的文档ID）。这些标准也在之前的工作$\left\lbrack  {{30},{11}}\right\rbrack$中使用过。

We reimplemented the SC algorithm based on the code available at http://repo.or.cz/w/galago.git, with BM25 as the ranking function. The original SC has four phases-OR, AND, Refine, and Ignore. In this paper we focus on getting safe results; thus we do not have an Ignore phase in our implementation (as we need the exact scores). Most of the time is spend in the ${OR}$ phase,so this does not change the query processing time by much.

我们基于http://repo.or.cz/w/galago.git上可用的代码重新实现了SC算法，使用BM25作为排序函数。原始的SC算法有四个阶段——OR、AND、细化和忽略。在本文中，我们专注于获得可靠的结果；因此，在我们的实现中没有忽略阶段（因为我们需要精确的分数）。大部分时间花在${OR}$阶段，所以这不会对查询处理时间产生太大影响。

We partitioned each list into 8 equal-sized layers. Note that in the original SC algorithm in [30], a different ranking function was used that has only 8 distinct impact scores, with each layer dealing with one score, and thus the layers were of different sizes. We also tried different combinations of variable-sized layers We found that equal-sized layers usually did at least as well as the other heuristics, though in principle there is still room for better approaches. Our explanation is that in SC, there is a "switching point", where we are guaranteed to have encountered the docIDs of all the corrects top- $k$ results in at least one list. At this point, we can switch from OR mode (TAAT) to AND mode before starting the next layer. Most of the computation time in SC is spend in OR mode, and things become much faster afterwards. While this "switching point" depends on how we partition the lists into layers, the switching point cannot happen before the point where the threshold condition is satisfied in the well-known TA algorithm of Fagin. The optimal partitioning would make a cut right after the threshold condition is satisfied for a query, but since this depends on the query one good choice is to spread out cuts evenly over the lists so that the next cut is never far away.

我们将每个列表划分为8个大小相等的层。请注意，在文献[30]中的原始SC算法中，使用了一种不同的排序函数，该函数只有8个不同的影响得分，每个层处理一个得分，因此各层的大小不同。我们还尝试了不同大小层的不同组合。我们发现，大小相等的层通常至少和其他启发式方法表现得一样好，不过原则上仍有改进的空间。我们的解释是，在SC算法中存在一个“切换点”，在该点上，我们可以确保在至少一个列表中已经遇到了所有正确的前$k$个结果的文档ID。此时，我们可以在开始下一层之前从OR模式（逐项处理，TAAT）切换到AND模式。SC算法中的大部分计算时间都花在OR模式上，之后处理速度会快很多。虽然这个“切换点”取决于我们如何将列表划分为层，但在著名的法金（Fagin）TA算法中满足阈值条件的点之前，切换点不会出现。最优的划分方式是在查询满足阈值条件后立即进行分割，但由于这取决于查询，一个不错的选择是在列表上均匀地分布分割点，这样下一个分割点就不会太远。

We also experimented with different numbers of layers. Figure 6 shows SC performance for different numbers of layers for equal-size layers, justifying the use of 8 layers.

我们还对不同的层数进行了实验。图6展示了等大小层在不同层数下的SC性能，证明使用8层是合理的。

Table 1 shows the query processing time using different algorithms. All runs in this paper exclude single-term queries, and no stopwords were removed (changing these assumptions would result in even better times). We observe that for TREC 2006, the effect of removing single-term queries is small, while for TREC 2005 the difference is significant as there are many such queries in the log. (But note that most single-term queries are resolved by result caching in real search engines.)

表1展示了使用不同算法的查询处理时间。本文中的所有运行都排除了单词查询，并且没有去除停用词（改变这些假设会使时间更短）。我们观察到，对于TREC 2006，去除单词查询的影响较小，而对于TREC 2005，差异显著，因为日志中有许多这样的查询。（但请注意，在实际搜索引擎中，大多数单词查询通过结果缓存来解决。）

<!-- Media -->

<table><tr><td colspan="7">TREC 2006</td></tr><tr><td/><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>exhaustive OR</td><td>225.7</td><td>60</td><td>159.2</td><td>261.4</td><td>376</td><td>646.4</td></tr><tr><td>WAND</td><td>77.6</td><td>23.0</td><td>42.5</td><td>89.9</td><td>141.2</td><td>251.6</td></tr><tr><td>SC</td><td>64.3</td><td>12.2</td><td>36.7</td><td>75.6</td><td>117.2</td><td>226.3</td></tr><tr><td>BMW</td><td>27.9</td><td>4.07</td><td>11.52</td><td>33.6</td><td>54.5</td><td>114.2</td></tr><tr><td>exhaustive AND</td><td>11.4</td><td>10.3</td><td>10.8</td><td>14.0</td><td>15.4</td><td>15.2</td></tr></table>

<table><tbody><tr><td colspan="7">2006年文本检索会议（TREC 2006）</td></tr><tr><td></td><td>平均值（avg）</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>穷举或（exhaustive OR）</td><td>225.7</td><td>60</td><td>159.2</td><td>261.4</td><td>376</td><td>646.4</td></tr><tr><td>加权与或网络（WAND）</td><td>77.6</td><td>23.0</td><td>42.5</td><td>89.9</td><td>141.2</td><td>251.6</td></tr><tr><td>得分组合（SC）</td><td>64.3</td><td>12.2</td><td>36.7</td><td>75.6</td><td>117.2</td><td>226.3</td></tr><tr><td>宝马（BMW）</td><td>27.9</td><td>4.07</td><td>11.52</td><td>33.6</td><td>54.5</td><td>114.2</td></tr><tr><td>穷举与</td><td>11.4</td><td>10.3</td><td>10.8</td><td>14.0</td><td>15.4</td><td>15.2</td></tr></tbody></table>

<table><tr><td colspan="7">TREC 2005</td></tr><tr><td/><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>exhaustive OR</td><td>369.3</td><td>62.1</td><td>238.9</td><td>515.2</td><td>778.3</td><td>1501.4</td></tr><tr><td>WAND</td><td>64.4</td><td>23.5</td><td>43.7</td><td>73.4</td><td>98.9</td><td>265.9</td></tr><tr><td>SC</td><td>63.5</td><td>14.2</td><td>37.5</td><td>119.7</td><td>172.9</td><td>316.9</td></tr><tr><td>BMW</td><td>21.2</td><td>3.5</td><td>12.7</td><td>25.2</td><td>39</td><td>104</td></tr><tr><td>exhaustive AND</td><td>6.86</td><td>6.4</td><td>7.3</td><td>9.2</td><td>4.7</td><td>5.9</td></tr></table>

<table><tbody><tr><td colspan="7">2005年文本检索会议（TREC 2005）</td></tr><tr><td></td><td>平均值（avg）</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>穷举或（exhaustive OR）</td><td>369.3</td><td>62.1</td><td>238.9</td><td>515.2</td><td>778.3</td><td>1501.4</td></tr><tr><td>加权和（WAND）</td><td>64.4</td><td>23.5</td><td>43.7</td><td>73.4</td><td>98.9</td><td>265.9</td></tr><tr><td>得分组合（SC）</td><td>63.5</td><td>14.2</td><td>37.5</td><td>119.7</td><td>172.9</td><td>316.9</td></tr><tr><td>宝马（BMW）</td><td>21.2</td><td>3.5</td><td>12.7</td><td>25.2</td><td>39</td><td>104</td></tr><tr><td>穷举与</td><td>6.86</td><td>6.4</td><td>7.3</td><td>9.2</td><td>4.7</td><td>5.9</td></tr></tbody></table>

Table 1: Average query processing time in ms for different numbers of query terms, using different algorithms on the TREC 2006 and 2005 query logs. Exhaustive OR, WAND, SC, and BMW are for disjunctive queries, while Exhaustive AND is for conjunctive queries.

表1：针对不同数量的查询词，在TREC 2006和2005查询日志上使用不同算法时的平均查询处理时间（毫秒）。穷举OR（Exhaustive OR）、WAND、SC和BMW用于析取查询，而穷举AND（Exhaustive AND）用于合取查询。

<!-- Media -->

From Table 1 we can see that our ${BMW}$ algorithm improves query processing performance. This is mainly due to the superiority of the DAAT index traversal over TAAT and the large amount of skipping. Also, our implementation for SC is not as fast as the numbers reported in [30], especially on TREC 2005 query log. This is because we remove single-term queries but do not remove stop-words as in [30], and also due to the use of BM25 as our ranking function. Still, SC outperforms basic WAND, as also reported in previous work. Overall, our basic ${BMW}$ algorithm achieves much faster query processing but is still much slower than exhaustive ${AND}$ using standard DAAT.

从表1中我们可以看出，我们的${BMW}$算法提高了查询处理性能。这主要是由于按文档编号遍历（DAAT）索引遍历优于按词项编号遍历（TAAT），并且有大量的跳跃操作。此外，我们实现的SC算法不如文献[30]中报告的速度快，特别是在TREC 2005查询日志上。这是因为我们移除了单词查询，但没有像文献[30]那样移除停用词，并且我们使用了BM25作为排名函数。不过，正如之前的工作所报道的那样，SC算法的性能仍然优于基本的WAND算法。总体而言，我们的基本${BMW}$算法实现了更快的查询处理速度，但仍比使用标准DAAT的穷举${AND}$算法慢得多。

Table 2 shows the other two criteria for different methods on the TREC 2006 query log (we omit the 2005 data due to space constraints). We also include the number of deep pointer movements and shallow pointer movements. As these measures are only meaningful for DAAT index traversal, we ignore the numbers for SC which uses TAAT traversal. Note that we did not include numbers for SC for evaluated docIDs, mainly because SC adopts TAAT-like query processing and the definition of evaluated docIDs will be misleading. Also, in BMW each partial evaluation is counted as an evaluated docID, no matter whether it stops early or not.

表2展示了在TREC 2006查询日志上不同方法的另外两个标准（由于篇幅限制，我们省略了2005年的数据）。我们还列出了深层指针移动次数和浅层指针移动次数。由于这些指标仅对DAAT索引遍历有意义，因此我们忽略了使用TAAT遍历的SC方法的相关数值。请注意，我们没有列出SC方法的评估文档ID（evaluated docIDs）的数值，主要是因为SC采用类似TAAT的查询处理方式，评估文档ID的定义可能会产生误导。此外，在BMW方法中，无论部分评估是否提前停止，每次部分评估都计为一个评估文档ID。

From Table 2, we see that all techniques improve greatly over exhaustive OR. WAND only evaluates ${4.6}\%$ of the docIDs compared to exhaustive OR, which approximately matches the numbers in [11]. BMW evaluates even fewer docIDs. This means that BMW should perform even better when we have a more expensive scoring function than BM25, such as the one mentioned in [11]. Another interesting point is that SC decodes less integers compared with the fastest method, BMW, which means that assigning promising docIDs to the first layers does help a lot. However, SC uses an additional data structure to temporarily store candidates, and this is the main drawback for SC and many other TAAT-based techniques.

从表2中我们可以看到，所有技术相较于穷举或（exhaustive OR）都有了显著改进。与穷举或相比，WAND（宽优先剪枝算法，WAND）仅评估了${4.6}\%$个文档ID，这与文献[11]中的数据大致相符。BMW（一种算法，BMW）评估的文档ID更少。这意味着当我们使用比BM25更复杂的评分函数（如文献[11]中提到的那种）时，BMW的表现会更好。另一个有趣的点是，与最快的方法BMW相比，SC（一种算法，SC）解码的整数更少，这意味着将有潜力的文档ID分配到第一层确实有很大帮助。然而，SC使用了一个额外的数据结构来临时存储候选对象，这是SC和许多其他基于TAAT（逐项累加，TAAT）的技术的主要缺点。

### 6.3 Document ID Reassignment

### 6.3 文档ID重新分配

In this section we show results after document ID reassignment. The idea for document ID reassignment is to assign docIDs to documents so that similar pages have close IDs. This idea is extensively explored in $\left\lbrack  {{28},{35},{18}}\right\rbrack$ and it is shown that after the reassignment, both compressed index size and query processing speed under exhaustive AND are significantly improved.

在本节中，我们展示文档ID重新分配后的结果。文档ID重新分配的思路是为文档分配文档ID，使得相似的页面具有相近的ID。这一思路在文献$\left\lbrack  {{28},{35},{18}}\right\rbrack$中得到了广泛探讨，结果表明，重新分配后，压缩索引大小和穷举与（exhaustive AND）下的查询处理速度都有显著提升。

One natural question is whether reassignment can also help processing speeds for disjunctive query processing. For exhaustive $\mathrm{{OR}}$ ,the intuition is that the improvement should be tiny because we fully evaluate the documents in all the lists anyway, no matter how the docIDs are assigned. For SC, the improvement should also be modest because in SC we will assign the postings such that the postings with higher impact scores appear earlier in the list. However, for WAND and BMW, reassignment of docIDs might give some benefits, as the distribution of impact values within each block should become more even, helping both WAND and BMW.

一个自然的问题是，重新分配是否也有助于提高析取查询处理的速度。对于穷举法（exhaustive），直觉上这种改进应该很小，因为无论文档ID（docIDs）如何分配，我们都会对所有列表中的文档进行全面评估。对于得分累积法（SC，Score Cumulation），改进也应该是适度的，因为在得分累积法中，我们会对倒排列表进行分配，使得影响得分较高的倒排列表项在列表中更早出现。然而，对于阈值算法（WAND，Weak AND）和块最大阈值算法（BMW，Block-Max WAND），重新分配文档ID可能会带来一些好处，因为每个块内影响值的分布应该会变得更加均匀，这对阈值算法和块最大阈值算法都有帮助。

<!-- Media -->

<table><tr><td/><td>evaluated docs</td><td>decoded ints</td><td>dpm</td><td>spm</td></tr><tr><td>exhaustive OR</td><td>3815676</td><td>9356032</td><td>15.9M</td><td>-</td></tr><tr><td>WAND</td><td>178391</td><td>6274432</td><td>1.18M</td><td>-</td></tr><tr><td>SC</td><td>-</td><td>965248</td><td>-</td><td>-</td></tr><tr><td>BMW</td><td>21921</td><td>2642752</td><td>0.42M</td><td>0.76M</td></tr><tr><td>exhaustive AND</td><td>20026</td><td>1939584</td><td>0.25M</td><td>-</td></tr></table>

<table><tbody><tr><td></td><td>已评估文档</td><td>已解码整数</td><td>文档处理模块（Document Processing Module）</td><td>搜索处理模块（Search Processing Module）</td></tr><tr><td>穷举或运算</td><td>3815676</td><td>9356032</td><td>15.9M</td><td>-</td></tr><tr><td>弱AND算法（Weak AND）</td><td>178391</td><td>6274432</td><td>1.18M</td><td>-</td></tr><tr><td>SC</td><td>-</td><td>965248</td><td>-</td><td>-</td></tr><tr><td>宝马（BMW）</td><td>21921</td><td>2642752</td><td>0.42M</td><td>0.76M</td></tr><tr><td>穷举与</td><td>20026</td><td>1939584</td><td>0.25M</td><td>-</td></tr></tbody></table>

Table 2: The average number of evaluated docIDs, decoded integers, deep pointer movements (dpm), and shallow pointer movements (spm) for different methods on the TREC 2006 query log. Exhaustive OR, WAND, SC, and BMW are for disjunctive queries, while Exhaustive AND is for conjunctive queries.

表2：在TREC 2006查询日志上，不同方法的评估文档ID（docID）、解码整数、深度指针移动（dpm）和浅层指针移动（spm）的平均数量。穷举或（Exhaustive OR）、宽阈与门（WAND）、得分组合（SC）和块最大加权（BMW）用于析取查询，而穷举与（Exhaustive AND）用于合取查询。

<table><tr><td colspan="7">TREC 2006</td></tr><tr><td/><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>exhaustive OR</td><td>210.6</td><td>55.3</td><td>156.6</td><td>245.9</td><td>354.2</td><td>583.9</td></tr><tr><td>WAND</td><td>50.1</td><td>17.2</td><td>28.7</td><td>57.5</td><td>94.9</td><td>168.9</td></tr><tr><td>SC</td><td>69.3</td><td>14.1</td><td>40</td><td>80.9</td><td>126.7</td><td>239.9</td></tr><tr><td>BMW</td><td>8.89</td><td>1.4</td><td>3.6</td><td>10.2</td><td>16.9</td><td>37.8</td></tr><tr><td>exhaustive AND</td><td>6.56</td><td>5.5</td><td>5.3</td><td>7.1</td><td>10.8</td><td>8.4</td></tr><tr><td colspan="7">TREC 2005</td></tr><tr><td/><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>exhaustive OR</td><td>349.7</td><td>56.4</td><td>226</td><td>495.8</td><td>743.1</td><td>1411.9</td></tr><tr><td>WAND</td><td>42.4</td><td>18.1</td><td>29.4</td><td>47.4</td><td>64.5</td><td>163.3</td></tr><tr><td>SC</td><td>76.4</td><td>12.6</td><td>52.9</td><td>112.2</td><td>162.8</td><td>288.9</td></tr><tr><td>BMW</td><td>7.2</td><td>1.3</td><td>3.7</td><td>8.9</td><td>13.5</td><td>35.8</td></tr><tr><td>exhaustive AND</td><td>4.5</td><td>4.3</td><td>4.7</td><td>5.9</td><td>2.7</td><td>4.1</td></tr></table>

<table><tbody><tr><td colspan="7">2006年文本检索会议（TREC 2006）</td></tr><tr><td></td><td>平均值（avg）</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>穷举或（exhaustive OR）</td><td>210.6</td><td>55.3</td><td>156.6</td><td>245.9</td><td>354.2</td><td>583.9</td></tr><tr><td>加权与或网络（WAND）</td><td>50.1</td><td>17.2</td><td>28.7</td><td>57.5</td><td>94.9</td><td>168.9</td></tr><tr><td>得分组合（SC）</td><td>69.3</td><td>14.1</td><td>40</td><td>80.9</td><td>126.7</td><td>239.9</td></tr><tr><td>宝马（BMW）</td><td>8.89</td><td>1.4</td><td>3.6</td><td>10.2</td><td>16.9</td><td>37.8</td></tr><tr><td>穷举与</td><td>6.56</td><td>5.5</td><td>5.3</td><td>7.1</td><td>10.8</td><td>8.4</td></tr><tr><td colspan="7">文本检索会议2005年会议（TREC 2005）</td></tr><tr><td></td><td>平均值（avg）</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>穷举或（exhaustive OR）</td><td>349.7</td><td>56.4</td><td>226</td><td>495.8</td><td>743.1</td><td>1411.9</td></tr><tr><td>加权与或网络（WAND）</td><td>42.4</td><td>18.1</td><td>29.4</td><td>47.4</td><td>64.5</td><td>163.3</td></tr><tr><td>得分组合（SC）</td><td>76.4</td><td>12.6</td><td>52.9</td><td>112.2</td><td>162.8</td><td>288.9</td></tr><tr><td>宝马（BMW）</td><td>7.2</td><td>1.3</td><td>3.7</td><td>8.9</td><td>13.5</td><td>35.8</td></tr><tr><td>穷举与</td><td>4.5</td><td>4.3</td><td>4.7</td><td>5.9</td><td>2.7</td><td>4.1</td></tr></tbody></table>

Table 3: Average query processing times in ms for different numbers of query terms after docID reassignment, on the TREC 2006 and 2005 query logs.

表3：在TREC 2006和2005查询日志上，文档ID重新分配后，不同数量查询词的平均查询处理时间（毫秒）。

<!-- Media -->

Table 3 shows the query processing times for the different techniques after docID reassignment. In particular, we assign docIDs according to the alphabetical ordering used in [28, 35]. From the table, we see that query processing performance is greatly improved for WAND and especially for BMW. In fact, the gaps between disjunctive and conjunctive queries are significantly narrowed. This means that reassignment succeeds in making the scores inside the blocks block much smoother, thus improving skipping.

表3展示了文档ID重新分配后不同技术的查询处理时间。具体而言，我们根据文献[28, 35]中使用的字母顺序分配文档ID。从表中可以看出，WAND（加权和动态修剪，Weighted AND）尤其是BMW（块最大加权和，Block-Max WAND）的查询处理性能有了显著提升。实际上，析取查询和合取查询之间的差距明显缩小。这意味着重新分配成功地使块内的得分更加平滑，从而提高了跳跃效率。

Query processing performance is also slightly improved for exhaustive OR. This is mainly because reassignment reduces the compressed size of the inverted index, thus reducing the cost of main memory accesses; see also [14] for more discussion of this issue.

穷举或（exhaustive OR）的查询处理性能也略有提升。这主要是因为重新分配减小了倒排索引的压缩大小，从而降低了主存访问成本；关于这个问题的更多讨论可参见文献[14]。

Table 4 shows the corresponding results for evaluated docIDs and decoded integers on the TREC 2006 query log. We observe similar trends as in the query processing time, with significant reductions for WAND and BMW.

表4展示了在TREC 2006查询日志上评估的文档ID和解码整数的相应结果。我们观察到与查询处理时间类似的趋势，WAND和BMW有显著的减少。

## 7. EXTENSIONS

## 7. 扩展

In this section we give some extensions to our BMW algorithm.

在本节中，我们对我们的BMW算法进行一些扩展。

### 7.1 A Layered Version of BMW

### 7.1 宝马（BMW）的分层版本

As shown in previous section, BMW achieved the best query processing performance for disjunctive queries, and significantly narrowed the performance gap between disjunctive and conjunctive queries. The main advantage for BMW seems to be that it uses DAAT index traversal, and thus does not have to use an expensive data structure to keep track of promising candidate documents.

如前一节所示，宝马（BMW）在析取查询方面实现了最佳的查询处理性能，并显著缩小了析取查询与合取查询之间的性能差距。宝马（BMW）的主要优势似乎在于它采用了按需文档访问（DAAT）索引遍历方式，因此无需使用昂贵的数据结构来跟踪有潜力的候选文档。

<!-- Media -->

<table><tr><td/><td>evaluated docs</td><td>decoded ints</td><td>dpm</td><td>spm</td></tr><tr><td>exhaustive OR</td><td>3815676</td><td>9356032</td><td>15.9M</td><td>-</td></tr><tr><td>WAND</td><td>221926</td><td>3472704</td><td>0.74M</td><td>-</td></tr><tr><td>SC</td><td>-</td><td>715776</td><td>-</td><td>-</td></tr><tr><td>BMW</td><td>9308</td><td>1181760</td><td>0.126M</td><td>0.22M</td></tr><tr><td>exhaustive AND</td><td>20026</td><td>951744</td><td>0.10M</td><td>-</td></tr></table>

<table><tbody><tr><td></td><td>已评估文档</td><td>已解码整数</td><td>每百万次展示成本（Dots Per Minute，这里推测可能是专业领域特定含义，需结合具体场景确定）</td><td>每百万次展示收入（Spend Per Mille，这里推测可能是专业领域特定含义，需结合具体场景确定）</td></tr><tr><td>穷举或运算</td><td>3815676</td><td>9356032</td><td>15.9M</td><td>-</td></tr><tr><td>宽优先与或网络（WAND，Wide Area Network Directory，这里推测可能是专业领域特定含义，需结合具体场景确定）</td><td>221926</td><td>3472704</td><td>0.74M</td><td>-</td></tr><tr><td>SC</td><td>-</td><td>715776</td><td>-</td><td>-</td></tr><tr><td>宝马（BMW）</td><td>9308</td><td>1181760</td><td>0.126M</td><td>0.22M</td></tr><tr><td>穷举与</td><td>20026</td><td>951744</td><td>0.10M</td><td>-</td></tr></tbody></table>

Table 4: The average number of evaluated docIDs, average decoded integers, deep pointer movements (dpm), and shallow pointer movements (spm) for different methods, after docID reassignment on the TREC 2006 query log.

表4：在TREC 2006查询日志上重新分配文档ID（docID）后，不同方法的评估文档ID平均数量、平均解码整数数量、深度指针移动次数（dpm）和浅度指针移动次数（spm）。

<!-- Media -->

On the other hand, SC achieves pretty good early termination performance (note from the previous section that SC actually decodes fewer integers than the other algorithms), by using an impact-layered index and assigning the most promising documents to the first layers. This means that the intuition for SC, putting top-scoring documents early in the lists to stop early, does have a lot of merit. A natural question is if we can combine this idea of a layered index with our BMW algorithm and its DAAT traversal mechanism. We now show how to do this. Our basic algorithm is very simple: For each inverted list,we split it into $\mathrm{N}$ layers. Then we treat each layer just as a separate term. In this case we directly apply the BMW algorithm on the impact-layered index.

另一方面，SC（选择性剪枝，Selective Cutoff）通过使用影响分层索引并将最有希望的文档分配到第一层，实现了相当不错的提前终止性能（从上一节可知，SC实际解码的整数比其他算法少）。这意味着SC的直觉——将得分最高的文档排在列表前面以尽早停止——确实有很多优点。一个自然的问题是，我们能否将这种分层索引的思想与我们的BMW（块最大权重，Block-Max WAND）算法及其按文档逐个处理（DAAT，Document-at-a-Time）遍历机制相结合。现在我们来展示如何做到这一点。我们的基本算法非常简单：对于每个倒排列表，我们将其拆分为$\mathrm{N}$层。然后我们将每一层视为一个单独的词项。在这种情况下，我们直接将BMW算法应用于影响分层索引。

The intuition behind this idea is that after we pick out the top-scoring documents from each list, the scores for the remaining do-cIDs are much smoother. So when we store the maximum impact score for each block, it is less likely that this score will be much larger than the others in the block. It's not difficult to understand that such spiky values are bad for BMW: If two two spikes are in two separate blocks, we probably have to decode both blocks, but if they are in one block we may only need to decode and access that one block. have to decode the two blocks. We call the new algorithm $N$ -layer ${BMW}$ where $\mathrm{N}$ is the number of layers. The disadvantage of doing this is that we will have a larger number of terms for each query. To minimize this disadvantage, we only split each list into 2 layers, a fancy layer and a normal layer, and each list is split only when it has more than $\alpha$ postings. We put the top-scoring $\beta$ postings in each list into the fancy layer. After some experiments,we set $\alpha  = {50K}$ and $\beta  = 2\%$ ,which seems to work well (getting the best possible parameters is left for future work).

这个想法背后的直觉是，在我们从每个列表中挑选出得分最高的文档后，其余文档ID（do - cIDs）的得分会平滑得多。因此，当我们存储每个块的最大影响得分时，这个得分比块中其他得分大很多的可能性就会降低。不难理解，这种尖峰状的值对BMW算法不利：如果两个尖峰分别位于两个不同的块中，我们可能需要对这两个块进行解码；但如果它们位于一个块中，我们可能只需要对该块进行解码和访问。我们将这种新算法称为$N$层${BMW}$算法，其中$\mathrm{N}$表示层数。这样做的缺点是，对于每个查询，我们会有更多的词项。为了尽量减少这一缺点，我们只将每个列表分成两层，即高级层和普通层，并且只有当列表中的倒排列表项超过$\alpha$个时才进行拆分。我们将每个列表中得分最高的$\beta$个倒排列表项放入高级层。经过一些实验，我们将$\alpha  = {50K}$和$\beta  = 2\%$作为参数，这似乎效果不错（确定最佳参数留待未来研究）。

Thus, in our 2-layer ${BMW}$ , we just treat the layers from one list as separate lists, and the different layers from the same list do not know the existence of each other. We also design a version where the layers from the same list know the existence of the others. The observation is that one document ID can only exist in one, if any, of the layers from the same list. In this case, in the "pivoting" phase of the BMW algorithm, we can do better by choosing the maximum score of the layers from the same list (instead of the sum of the scores). We experimented with this idea and found that it moderately decreases pointer movements and decoded integers, but not the actual running time, due to the overhead of tracking layers belonging to the same term during pivoting. For space reason we omit the results of these experiments, though future work may lead to more practical variants based on this idea.

因此，在我们的两层${BMW}$中，我们仅将来自一个列表的各层视为独立的列表，并且同一列表中的不同层彼此不知道对方的存在。我们还设计了一个版本，其中同一列表中的各层知道其他层的存在。观察发现，一个文档ID如果存在于同一列表的各层中，也只能存在于其中一层。在这种情况下，在BMW算法的“枢轴”阶段，我们可以通过选择同一列表各层的最大得分（而不是得分总和）来取得更好的效果。我们对这一想法进行了实验，发现它适度减少了指针移动和已解码整数的数量，但由于在枢轴操作期间跟踪属于同一术语的各层存在开销，实际运行时间并未减少。由于篇幅原因，我们省略了这些实验的结果，不过未来的工作可能会基于这一想法开发出更实用的变体。

Table 5 shows the query processing performance for 2-layer ${BMW}$ without and with docID reassignment. As we can see, 2-layer ${BMW}$ obtains improved running times over basic BMW. We also show the query processing time for queries with different numbers of terms in Table 6.

表5展示了两层${BMW}$在有无文档ID重新分配情况下的查询处理性能。正如我们所见，两层${BMW}$的运行时间比基本的BMW算法有所改善。我们还在表6中展示了不同词项数量的查询的处理时间。

<!-- Media -->

<table><tr><td>before reassignment</td><td>time (ms)</td><td>evaluated docs</td><td>decoded ints</td></tr><tr><td>BMW</td><td>27.9</td><td>21921</td><td>2642752</td></tr><tr><td>2-layer BMW</td><td>22.9</td><td>7435</td><td>1731264</td></tr><tr><td>after reassignment</td><td>time (ms)</td><td>evaluated docs</td><td>decoded ints</td></tr><tr><td>BMW</td><td>8.89</td><td>9308</td><td>1181760</td></tr><tr><td>2-layer BMW</td><td>7.4</td><td>4196</td><td>790464</td></tr></table>

<table><tbody><tr><td>重新分配之前</td><td>时间（毫秒）</td><td>已评估文档</td><td>已解码整数</td></tr><tr><td>宝马（BMW）</td><td>27.9</td><td>21921</td><td>2642752</td></tr><tr><td>双层宝马（2-layer BMW）</td><td>22.9</td><td>7435</td><td>1731264</td></tr><tr><td>重新分配后</td><td>时间（毫秒）</td><td>已评估文档</td><td>已解码整数</td></tr><tr><td>宝马（BMW）</td><td>8.89</td><td>9308</td><td>1181760</td></tr><tr><td>双层宝马（2-layer BMW）</td><td>7.4</td><td>4196</td><td>790464</td></tr></tbody></table>

Table 5: Query processing performance after combining layered index and BMW, before and after docID reassignment, on the TREC 2006 query log. All numbers are averaged per query.

表5：在TREC 2006查询日志上，结合分层索引和BMW（块最大小波，Block-Max Wavelet）后，文档ID（docID）重新分配前后的查询处理性能。所有数值均为每个查询的平均值。

<table><tr><td>before reassignment</td><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>BMW</td><td>27.9</td><td>4.07</td><td>11.52</td><td>33.6</td><td>54.5</td><td>114.2</td></tr><tr><td>2-layer BMW</td><td>22.9</td><td>2.9</td><td>10</td><td>30.8</td><td>46.3</td><td>98.2</td></tr><tr><td>after reassignment</td><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>BMW</td><td>8.89</td><td>1.4</td><td>3.6</td><td>10.2</td><td>16.9</td><td>37.8</td></tr><tr><td>2-layer BMW</td><td>7.4</td><td>1.1</td><td>3.3</td><td>8.5</td><td>15.0</td><td>31.4</td></tr></table>

<table><tbody><tr><td>重新分配之前</td><td>平均值</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>宝马（BMW）</td><td>27.9</td><td>4.07</td><td>11.52</td><td>33.6</td><td>54.5</td><td>114.2</td></tr><tr><td>双层宝马（2-layer BMW）</td><td>22.9</td><td>2.9</td><td>10</td><td>30.8</td><td>46.3</td><td>98.2</td></tr><tr><td>重新分配之后</td><td>平均值</td><td>2</td><td>3</td><td>4</td><td>5</td><td>$> 5$</td></tr><tr><td>宝马（BMW）</td><td>8.89</td><td>1.4</td><td>3.6</td><td>10.2</td><td>16.9</td><td>37.8</td></tr><tr><td>双层宝马（2-layer BMW）</td><td>7.4</td><td>1.1</td><td>3.3</td><td>8.5</td><td>15.0</td><td>31.4</td></tr></tbody></table>

Table 6: Average query processing time in ms for different number of query terms for BMW and 2-layer BMW, before and after docID reassignment, on the TREC 2006 query log.

表6：在TREC 2006查询日志上，对于宝马（BMW）和两层宝马（2 - layer BMW），在文档ID重新分配前后，不同查询词数量下的平均查询处理时间（毫秒）。

<!-- Media -->

### 7.2 Increasing Top-k

### 7.2 增加前k个结果

As mentioned, one commonly used ranking technique in current web search engines is based on a two-phase approach, where we first get the, say, top-1000 documents according to a simple ranking function such as BM25, and then compute the exact score according to a more complicated ranking function only for these 1000 . An example appears in [34] for a ranking function that uses position information in addition to BM25. Of course, in these scenarios we usually need much larger values of $k$ than $k = {10}$ ,e.g.,a few hundred or few thousand results.

如前所述，当前网络搜索引擎中常用的一种排序技术基于两阶段方法，即首先根据简单的排序函数（如BM25）获取前1000个文档，然后仅针对这1000个文档根据更复杂的排序函数计算精确得分。文献[34]中给出了一个除BM25外还使用位置信息的排序函数示例。当然，在这些场景中，我们通常需要比$k = {10}$大得多的$k$值，例如几百或几千个结果。

In Figure 7 we show the performance as we increase $k$ ,with reassigned docIDs. We find that the performance for the naive ${ex}$ - haustive OR algorithm is quite stable as it decodes and evaluates all the postings anyway. For other algorithms, the query processing time increases. For SC, we observe a huge performance degradation. This is mainly because the performance of the temporal data structure degrades more and more as we store more candidates. We also see that ${BMW}$ and 2-layer ${BMW}$ perform quite well even when $k$ is equal to 1000,and that the increase in time is fairly moderate.

在图7中，我们展示了在重新分配文档ID（docIDs）的情况下，随着$k$值的增加所呈现的性能表现。我们发现，朴素的${ex}$ - 穷举或（OR）算法的性能相当稳定，因为无论如何它都会对所有倒排列表项进行解码和评估。对于其他算法，查询处理时间会增加。对于SC算法，我们观察到其性能出现了大幅下降。这主要是因为随着我们存储的候选项越来越多，时态数据结构的性能会越来越差。我们还发现，即使当$k$等于1000时，${BMW}$和两层${BMW}$算法的表现也相当不错，并且处理时间的增加幅度相当适中。

<!-- Media -->

<!-- figureText: 200 top k -WAND -BMW — 2 layer BMW -->

<img src="https://cdn.noedgeai.com/01957b28-344b-73e7-8dd0-fa52f1bc6e02_7.jpg?x=959&y=1541&w=644&h=449&r=0"/>

Figure 7: Query processing times for different techniques with docID reassignment, on the TREC 2006 query log. The X-axis is the value for $k$ ,while the Y-axis is the average processing time.

图7：在TREC 2006查询日志上，不同技术在重新分配文档ID（docID）时的查询处理时间。X轴表示$k$的值，Y轴表示平均处理时间。

<!-- Media -->

### 7.3 Block-Max AND

### 7.3 块最大与（Block - Max AND）

One advantage of BMW is that the idea can also be applied to conjunctive query processing using standard DAAT index traversal. For conjunctive query processing, standard AND starts from the shortest list, and then tries to find the corresponding docID in the longer lists. It is a natural extension to integrate the Block-Max Index structure and add shallow pointers in DAAT for conjunctive query processing. The improved DAAT algorithm is shown in Algorithm 4; it is called Block-Max AND (BMA).

宝马（BMW）的一个优势在于，该理念还可应用于使用标准文档按序存取（DAAT）索引遍历的合取查询处理。对于合取查询处理，标准的“与”操作从最短的列表开始，然后尝试在较长的列表中查找对应的文档编号（docID）。在合取查询处理中，将块最大索引（Block - Max Index）结构集成到DAAT中并添加浅层指针是一种自然的扩展。改进后的DAAT算法如算法4所示；它被称为块最大“与”（Block - Max AND，BMA）。

<!-- Media -->

---

Sort the lists from shortest to longest;

d = 0 ;

repeat

	$\mathrm{d} = \operatorname{NextGEQ}\left( {\operatorname{list}\left\lbrack  0\right\rbrack  ,\mathrm{d}}\right)$ ;

	for $i = 1\ldots n$ do

		NextShallow(d, list(i));

	end

	${flag} = {CheckBlockMax}\left( {\theta ,n}\right)$ ;

	if ${flag} =  =$ true then

		search $d$ in the rest lists,if found evaluate,otherwise

		$d = d + 1$

	end

	else

		d = GetNewCandidate(   );

		continue;

	end

until Stop;

---

Algorithm 4: Block-Max AND for conjunctive queries with $\mathrm{n}$ terms.

算法4：针对具有$\mathrm{n}$个词项的合取查询的块最大“与”算法。

<!-- Media -->

As we see we only have to slightly adapt the standard AND algorithm to get the BMA algorithm. In particular, we use three routines from BMW - NextShallow(   ), CheckBlockMax(   ) and GetNewCan-didate(   ). The processing cost for BMA is shown in Table 7. We observe that the BMA works better for queries with smaller numbers of terms (otherwise the shallow pointer movements will become expensive). So we also propose one hybrid algorithm: Apply BMA when the number of terms in the query is less than $T$ ; otherwise use exhaustive AND. We use $T = 4$ in this paper.

正如我们所见，我们只需对标准的“与”（AND）算法稍作调整，就能得到块最大算法（BMA，Block-Max Algorithm）。具体而言，我们使用了来自块最大跳跃（BMW，Block-Max With Jumps）算法的三个例程：NextShallow( )、CheckBlockMax( )和GetNewCandidate( )。块最大算法的处理成本如表7所示。我们发现，块最大算法在处理词项数量较少的查询时效果更好（否则浅层指针移动的成本会很高）。因此，我们还提出了一种混合算法：当查询中的词项数量少于$T$时，应用块最大算法；否则，使用穷举“与”（exhaustive AND）算法。本文使用了$T = 4$。

Table 8 shows the performance using BMA and the hybrid BMA, before and after docID reassignment. We can see significant improvements over an exhaustive AND. Note that ${BMW}$ and ${BMA}$ algorithms use the same index structure; thus the Block-Max Index can support both types of queries.

表8展示了在文档标识符（docID）重新分配前后，使用块最大算法和混合块最大算法的性能。我们可以看到，与穷举“与”算法相比，性能有了显著提升。请注意，${BMW}$和${BMA}$算法使用相同的索引结构；因此，块最大索引（Block-Max Index）可以支持这两种类型的查询。

## 8. OPEN QUESTIONS

## 8. 开放性问题

Our results in this paper raise several interesting open questions.

本文的研究结果引发了几个有趣的开放性问题。

A Cleaner Algorithm and Analysis: While the described methods already achieve large benefits, we are not yet convinced that we have really found the optimal algorithm. It would also be interesting to provide some analysis, say of the optimal number of pointer movements and document evaluations under our approach.

一种更优的算法与分析：虽然所描述的方法已经带来了显著的益处，但我们仍不确定是否真的找到了最优算法。对我们的方法下指针移动的最优次数和文档评估情况进行分析也会很有意思。

Other Applications of Block-Max Indexes: The basic idea behind our augmented index structure could also be applied to other scenarios. For example, it would be natural to try to integrate local maximum scores into the two-level index structure in [1].

块最大值索引的其他应用：我们增强型索引结构背后的基本思想也可应用于其他场景。例如，尝试将局部最大得分整合到文献[1]中的两级索引结构中是很自然的想法。

<!-- Media -->

<table><tr><td>before reassignment</td><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>exhaustive AND</td><td>11.4</td><td>10.8</td><td>10</td><td>12.5</td><td>11.57</td><td>9.94</td></tr><tr><td>BMA</td><td>9.89</td><td>3.96</td><td>7.92</td><td>13.2</td><td>14.08</td><td>14.63</td></tr><tr><td>after reassignment</td><td>avg</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>exhaustive AND</td><td>6.56</td><td>6.93</td><td>6.11</td><td>7.06</td><td>6.84</td><td>5.57</td></tr><tr><td>BMA</td><td>5.12</td><td>1.69</td><td>4.02</td><td>7.03</td><td>7.33</td><td>9.06</td></tr></table>

<table><tbody><tr><td>重新分配之前</td><td>平均值</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>穷举与</td><td>11.4</td><td>10.8</td><td>10</td><td>12.5</td><td>11.57</td><td>9.94</td></tr><tr><td>贝叶斯模型平均法（Bayesian Model Averaging，BMA）</td><td>9.89</td><td>3.96</td><td>7.92</td><td>13.2</td><td>14.08</td><td>14.63</td></tr><tr><td>重新分配之后</td><td>平均值</td><td>2</td><td>3</td><td>4</td><td>5</td><td>> 5</td></tr><tr><td>穷举与</td><td>6.56</td><td>6.93</td><td>6.11</td><td>7.06</td><td>6.84</td><td>5.57</td></tr><tr><td>贝叶斯模型平均法（Bayesian Model Averaging，BMA）</td><td>5.12</td><td>1.69</td><td>4.02</td><td>7.03</td><td>7.33</td><td>9.06</td></tr></tbody></table>

Table 7: Average query processing times in ms for different numbers of query terms, using exhaustive AND and BMA, before and after do-cID reassignment, on TREC 2006.

表7：在TREC 2006数据集上，使用穷举与（exhaustive AND）和布尔多值算法（BMA），在文档ID（do - cID）重新分配前后，针对不同数量的查询词的平均查询处理时间（毫秒）。

<table><tr><td>before reassignment</td><td>time</td><td>evaluated docs</td><td>decoded ints</td></tr><tr><td>exhaustive AND</td><td>11.4</td><td>20026</td><td>1939584</td></tr><tr><td>BMA</td><td>9.89</td><td>5725</td><td>1460992</td></tr><tr><td>Hybrid</td><td>9.4</td><td>6594</td><td>1568704</td></tr><tr><td>after reassignment</td><td>time</td><td>evaluated docs</td><td>decoded ints</td></tr><tr><td>exhaustive AND</td><td>6.56</td><td>20026</td><td>951744</td></tr><tr><td>BMA</td><td>5.12</td><td>3108</td><td>607680</td></tr><tr><td>Hybrid</td><td>4.53</td><td>3673</td><td>641344</td></tr></table>

<table><tbody><tr><td>重新分配之前</td><td>时间</td><td>已评估的文档</td><td>已解码的整数</td></tr><tr><td>穷举与运算</td><td>11.4</td><td>20026</td><td>1939584</td></tr><tr><td>块匹配算法（BMA）</td><td>9.89</td><td>5725</td><td>1460992</td></tr><tr><td>混合（Hybrid）</td><td>9.4</td><td>6594</td><td>1568704</td></tr><tr><td>重新分配后</td><td>时间</td><td>已评估的文档</td><td>已解码的整数</td></tr><tr><td>穷举与运算</td><td>6.56</td><td>20026</td><td>951744</td></tr><tr><td>块匹配算法（BMA）</td><td>5.12</td><td>3108</td><td>607680</td></tr><tr><td>混合（Hybrid）</td><td>4.53</td><td>3673</td><td>641344</td></tr></tbody></table>

Table 8: Average query processing times in ms, numbers of evaluated docIDs per query and average decoded integers per query for conjunctive queries, before and after docID reassignment, on TREC 2006.

表8：在TREC 2006数据集上，合取查询在文档ID重新分配前后的平均查询处理时间（毫秒）、每个查询评估的文档ID数量以及每个查询平均解码的整数数量。

<table><tr><td/><td>no reassignment</td><td>with reassignment</td></tr><tr><td>BMW</td><td>27.9</td><td>8.89</td></tr><tr><td>Clairvoyant BMW</td><td>23.0</td><td>7.2</td></tr></table>

<table><tbody><tr><td></td><td>不重新分配</td><td>重新分配</td></tr><tr><td>宝马（BMW）</td><td>27.9</td><td>8.89</td></tr><tr><td>透视眼宝马（Clairvoyant BMW）</td><td>23.0</td><td>7.2</td></tr></tbody></table>

Table 9: Average query processing times in ms for BMW versus Clairvoyant BMW, for top-10 results, on the Trec 2006 query log.

表9：在Trec 2006查询日志上，针对前10个结果，宝马（BMW）与先知宝马（Clairvoyant BMW）的平均查询处理时间（毫秒）。

<!-- Media -->

Estimating Top-k Thresholds: Our methods could be further improved if we somehow had a good a-priori estimate of what score is needed to make it into the top $k$ results. Currently,we start with a threshold of zero, and then update the value as results are discovered. Thus, the algorithm starts slow and then speeds up. This motivates the following algorithmic problem that also has other applications, for example in distributed IR: Given an inverted index, a query,and a number $k$ ,how do we quickly estimate the score of the $k$ -th best result (possibly using some small auxiliary structures),or conversely,given a threshold $t$ how do we estimate the number of results with score higher than $t$ .

估计前k个阈值：如果我们能以某种方式先验地很好地估计进入前 $k$ 个结果所需的分数，我们的方法可以进一步改进。目前，我们从阈值为零开始，然后在发现结果时更新该值。因此，该算法开始时较慢，然后会加速。这引出了以下算法问题，该问题还有其他应用，例如在分布式信息检索（IR）中：给定一个倒排索引、一个查询和一个数字 $k$ ，我们如何快速估计第 $k$ 个最佳结果的分数（可能使用一些小型辅助结构），或者相反，给定一个阈值 $t$ ，我们如何估计分数高于 $t$ 的结果数量。

For motivation, we show in Table 9 that a clairvoyant algorithm that knows the score of the $k$ -th best result would get a circa ${20}\%$ reduction in query processing costs.

为了说明动机，我们在表9中展示了一个先知算法（clairvoyant algorithm），该算法知道第 $k$ 个最佳结果的分数，其查询处理成本大约会降低 ${20}\%$ 。

Query Processing with Score Approximations: Another interesting more general question is how to best approximate the impact scores in inverted lists, and how to best use such approximations during query processing. Consider the scenario in Figure 3, where we have a long, sparse, array of impact values that is upper-bounded by some block-wise approximation. What is the best approximation for a given array of values? Does it make sense to have a multi-level structure (similar to wavelet trees) that provides upper bounds for progressively smaller block sizes as we descend to lower levels? Are there statistical measures other than the maximum impact, say the skew of the values, that are useful?

使用得分近似值进行查询处理：另一个更具普遍性的有趣问题是，如何最佳地近似倒排列表中的影响得分，以及在查询处理过程中如何最佳地使用这些近似值。考虑图3中的场景，我们有一个长而稀疏的影响值数组，该数组由某种逐块近似值界定上限。对于给定的值数组，最佳的近似方法是什么？是否有必要构建一个多级结构（类似于小波树），随着层级降低，为逐渐变小的块大小提供上限？除了最大影响值之外，是否还有其他统计指标（例如值的偏度）是有用的？

## 9. CONCLUSION

## 9. 结论

In this paper, we have described and evaluated improved safe early termination for disjunctive queries. This was achieved by an augmented index structure called a Block-Max Index, which stores maximum impacts for blocks of postings. We then showed how to integrate this structure into the WAND approach. Finally, we extended it to the impact-layered indexes, indexes with reassigned do-cIDs, and conjunctive queries, leading to additional improvements. Our results also lead to a number of interesting opportunities for future research, as discussed in Section 8.

在本文中，我们描述并评估了用于析取查询的改进型安全早期终止方法。这是通过一种名为块最大索引（Block - Max Index）的增强索引结构实现的，该结构存储了 postings 块的最大影响值。然后，我们展示了如何将此结构集成到 WAND 方法中。最后，我们将其扩展到影响分层索引、重新分配文档 ID 的索引以及合取查询，从而带来了更多改进。如第8节所述，我们的研究结果也为未来的研究提供了许多有趣的机会。

## Acknowledgments

## 致谢

This research was supported by NSF Grant IIS-0803605, "Efficient and Effective Search Services over Archival Webs", and by a grant from Google.

本研究得到了美国国家科学基金会（NSF）资助项目IIS - 0803605“存档网络上高效的搜索服务”以及谷歌（Google）的一项资助。

## 10. REFERENCES

## 10. 参考文献

[1] Ismail Sengor Altingovde, Engin Demir, Fazli Can, and Ozgur Ulusoy. Incremental cluster-based retrieval using compressed cluster-skipping inverted files. ${ACM}$ Transactions on Information Systems, 26(3):1-36, 2008.

[1] 伊斯梅尔·森戈尔·阿尔廷戈夫德（Ismail Sengor Altingovde）、恩金·德米尔（Engin Demir）、法兹利·灿（Fazli Can）和厄兹居尔·于勒索伊（Ozgur Ulusoy）。使用压缩簇跳跃倒排文件的增量式基于簇的检索。《信息系统汇刊》（Transactions on Information Systems），26(3):1 - 36，2008年。

[2] V. Anh and A. Moffat. Simplified similarity scoring using term ranks. In Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 226-233, 2005.

[2] V. 安（V. Anh）和A. 莫法特（A. Moffat）。使用词项排名简化相似度评分。见《第28届ACM国际信息检索研究与发展大会论文集》，第226 - 233页，2005年。

[3] V. Anh and A. Moffat. Pruned query evaluation using pre-computed impacts. In Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 372-379, 2006.

[3] V. 安（V. Anh）和A. 莫法特（A. Moffat）。使用预计算影响值的剪枝查询评估。见《第29届ACM国际信息检索研究与发展大会论文集》，第372 - 379页，2006年。

[4] Vo Ngoc Anh, Owen de Kretser, and Alistair Moffat. Vector-space ranking with effective early termination. In Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2001.

[4] 阮玉英（Vo Ngoc Anh）、欧文·德·克雷泽（Owen de Kretser）和阿利斯泰尔·莫法特（Alistair Moffat）。具有有效早期终止的向量空间排序。见《第24届ACM国际信息检索研究与发展年会论文集》，2001年。

[5] C. Badue, R. Baeza-Yates, B. Ribeiro-Neto, and N. Ziviani. Distributed query processing using partitioned inverted files. In Proceedings of the 9th String Processing and Information Retrieval Symposium, 2002.

[5] C. 巴杜埃（C. Badue）、R. 贝萨 - 耶茨（R. Baeza - Yates）、B. 里贝罗 - 内托（B. Ribeiro - Neto）和N. 齐维亚尼（N. Ziviani）。使用分区倒排文件的分布式查询处理。见《第9届字符串处理与信息检索研讨会论文集》，2002年。

[6] R. Baeza-Yates and B. Ribeiro-Neto. Modern Information Retrieval. Addision Wesley, 1999.

[6] R. 贝萨 - 耶茨（R. Baeza - Yates）和B. 里贝罗 - 内托（B. Ribeiro - Neto）。《现代信息检索》。艾迪生·韦斯利出版社，1999年。

[7] Ricardo Baeza-Yates, Aristides Gionis, Flavio Junqueira, Vanessa Murdock, Vassilis Plachouras, and Fabrizio Silvestri. The impact of caching on search engines. In Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2007.

[7] 里卡多·贝萨 - 耶茨（Ricardo Baeza - Yates）、阿里斯蒂德斯·吉奥尼斯（Aristides Gionis）、弗拉维奥·容凯拉（Flavio Junqueira）、凡妮莎·默多克（Vanessa Murdock）、瓦西里斯·普拉胡拉斯（Vassilis Plachouras）和法布里齐奥·西尔维斯特里（Fabrizio Silvestri）。缓存对搜索引擎的影响。见《第30届ACM国际信息检索研究与发展年会论文集》，2007年。

[8] Holger Bast, Debapriyo Majumdar, Ralf Schenkel, Martin Theobald, and Gerhard Weikum. IO-Top-K: Index-access optimized top-k query processing. In Proceedings of the 32th International Conference on Very Large Data Bases, 2006.

[8] 霍尔格·巴斯（Holger Bast）、德巴普里约·马宗达（Debapriyo Majumdar）、拉尔夫·申克尔（Ralf Schenkel）、马丁·特奥巴尔德（Martin Theobald）和格哈德·魏库姆（Gerhard Weikum）。IO-Top-K：索引访问优化的前k查询处理。见《第32届国际超大型数据库会议论文集》，2006年。

[9] Roi Blanco and Álvaro Barreiro. TSP and cluster-based solutions to the reassignment of document identifiers. Information Retrieval, 9(4):499-517, 2006.

[9] 罗伊·布兰科（Roi Blanco）和阿尔瓦罗·巴雷罗（Álvaro Barreiro）。基于旅行商问题（TSP）和聚类的文档标识符重新分配解决方案。《信息检索》，9(4):499 - 517，2006年。

[10] Roi Blanco and Alvaro Barreiro. Probabilistic static pruning of inverted files. ACM Transactions on Information Systems, 28(1), January 2010.

[10] 罗伊·布兰科（Roi Blanco）和阿尔瓦罗·巴雷罗（Alvaro Barreiro）。倒排文件的概率静态剪枝。《ACM信息系统汇刊》，28(1)，2010年1月。

[11] Andrei Z. Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. Efficient query evaluation using a two-level retrieval process. In Proceedings of the 12th ACM Conference on Information and Knowledge Management, 2003.

[11] 安德烈·Z·布罗德（Andrei Z. Broder）、大卫·卡梅尔（David Carmel）、迈克尔·赫斯科维奇（Michael Herscovici）、阿亚·索弗（Aya Soffer）和杰森·齐恩（Jason Zien）。使用两级检索过程进行高效查询评估。见《第12届ACM信息与知识管理会议论文集》，2003年。

[12] N. Bruno, L. Gravano, and A. Marian. Evaluating top-k queries over web-accessible databases. In Proceedings of the 18th Annual International Conference on Data Engineering, 2002.

[12] N. 布鲁诺（N. Bruno）、L. 格拉瓦诺（L. Gravano）和A. 玛丽安（A. Marian）。对可通过网络访问的数据库评估前k查询。见《第18届年度国际数据工程会议论文集》，2002年。

[13] C. Buckley and A. F. Lewit. Optimization of inverted vector searches. In Proceedings of the 8th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 1985.

[13] C. 巴克利（C. Buckley）和 A. F. 莱维特（A. F. Lewit）。倒排向量搜索的优化。见《第 8 届年度国际计算机协会信息检索研究与发展会议论文集》，1985 年。

[14] Stefan Buttcher and Charles L. A. Clarke. Index compression is good, especially for random access. In Proceedings of the 16th ACM Conference on Information and Knowledge Management, 2007.

[14] 斯特凡·布彻（Stefan Buttcher）和查尔斯·L. A. 克拉克（Charles L. A. Clarke）。索引压缩有益，尤其对于随机访问。见《第 16 届计算机协会信息与知识管理会议论文集》，2007 年。

[15] Kaushik Chakrabarti, Surajit Chaudhuri, and Venkatesh Ganti. Interval-based pruning for top-k processing over compressed lists. In Proceedings of the 27th IEEE International Conference on Data Engineering (ICDE), 2011.

[15] 考希克·查克拉巴蒂（Kaushik Chakrabarti）、苏拉吉特·乔杜里（Surajit Chaudhuri）和文卡特什·甘蒂（Venkatesh Ganti）。基于区间的压缩列表前 k 处理剪枝。见《第 27 届电气与电子工程师协会国际数据工程会议（ICDE）论文集》，2011 年。

[16] J. Cho and A. Ntoulas. Pruning policies for two-tiered inverted index with correctness guarantee. In Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2007.

[16] J. 赵（J. Cho）和 A. 恩图拉斯（A. Ntoulas）。保证正确性的两层倒排索引剪枝策略。见《第 30 届年度国际计算机协会信息检索研究与发展会议论文集》，2007 年。

[17] Jeffrey Dean. Challenges in building large-scale information retrieval systems. In Proceedings of the Second ACM International Conference on Web Search and Data Mining, 2009.

[17] 杰弗里·迪恩（Jeffrey Dean）。构建大规模信息检索系统的挑战。见《第二届计算机协会国际网络搜索与数据挖掘会议论文集》，2009 年。

[18] S. Ding, J. Attenberg, and T. Suel. Scalable techniques for document identifier assignment in inverted indexes. In Proceedings of the 19th International Conference on World Wide Web, 2010.

[18] 丁（S. Ding）、阿滕伯格（J. Attenberg）和苏埃尔（T. Suel）。倒排索引中文档标识符分配的可扩展技术。见《第19届万维网国际会议论文集》，2010年。

[19] R. Fagin, D. Carmel, D. Cohen, E. Farchi, M. Herscovici, Y. Maarek, and A. Soffer. Static index pruning for information retrieval systems. In Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2001.

[19] 法金（R. Fagin）、卡梅尔（D. Carmel）、科恩（D. Cohen）、法尔基（E. Farchi）、赫斯科维西（M. Herscovici）、马雷克（Y. Maarek）和索弗（A. Soffer）。信息检索系统的静态索引剪枝。见《第24届ACM信息检索研究与发展年度国际会议论文集》，2001年。

[20] R. Fagin, A. Lotem, and M. Naor. Optimal aggregation algorithms for middleware. In Proceedings of the ACM Symp. on Principles of Database Systems, 2001.

[20] 法金（R. Fagin）、洛特姆（A. Lotem）和纳尔（M. Naor）。中间件的最优聚合算法。见《ACM数据库系统原理研讨会论文集》，2001年。

[21] Ronald Fagin. Combining fuzzy information: an overview. SIGMOD Record, 31:2002, 2002.

[21] 罗纳德·法金（Ronald Fagin）。模糊信息的组合：概述。《SIGMOD记录》，31:2002，2002年。

[22] R. Lempel and S. Moran. Optimizing result prefetching in web search engines with segmented indices. In Proceedings of the 28th International Conference on Very Large Data Bases, 2002.

[22] R. 伦佩尔（R. Lempel）和 S. 莫兰（S. Moran）。利用分段索引优化网络搜索引擎中的结果预取。见《第28届大型数据库国际会议论文集》，2002年。

[23] X. Long and T. Suel. Optimized query execution in large search engines with global page ordering. In Proceedings of the 29th International Conference on Very Large Data Bases, 2003.

[23] X. 朗（X. Long）和 T. 苏埃尔（T. Suel）。在具有全局页面排序的大型搜索引擎中优化查询执行。见《第29届大型数据库国际会议论文集》，2003年。

[24] S. Melnik, S. Raghavan, B. Yang, and H. Garcia-Molina. Building a distributed full-text index for the web. In Proceedings of the 10th International Conference on World Wide Web, 2000.

[24] S. 梅尔尼克（S. Melnik）、S. 拉加万（S. Raghavan）、B. 杨（B. Yang）和 H. 加西亚 - 莫利纳（H. Garcia - Molina）。为网络构建分布式全文索引。见《第10届万维网国际会议论文集》，2000年。

[25] Michael Persin, Justin Zobel, and Ron Sacks-davis. Filtered document retrieval with frequency-sorted indexes. Journal of the American Society for Information Science, 47:749-764, 1996.

[25] 迈克尔·佩尔辛（Michael Persin）、贾斯汀·佐贝尔（Justin Zobel）和罗恩·萨克斯 - 戴维斯（Ron Sacks - davis）。使用频率排序索引进行过滤文档检索。《美国信息科学学会期刊》，47:749 - 764，1996年。

[26] W. Shieh, T. Chen, J. Shann, and C. Chung. Inverted file compression through document identifier reassignment. Information Processing and Management, 39(1):117-131, 2003.

[26] W. 谢（W. Shieh）、T. 陈（T. Chen）、J. 尚（J. Shann）和 C. 钟（C. Chung）。通过文档标识符重新分配进行倒排文件压缩。《信息处理与管理》，39(1):117 - 131，2003年。

[27] F. Silvestri and R. Venturini. Vsencoding: efficient coding and fast decoding of integer lists via dynamic programming. In Proceedings of the 19th ACM Conference on Information and Knowledge Management, 2010.

[27] F. 西尔维斯特里（F. Silvestri）和 R. 文图里尼（R. Venturini）。Vs 编码：通过动态规划实现整数列表的高效编码和快速解码。收录于《第 19 届 ACM 信息与知识管理会议论文集》，2010 年。

[28] Fabrizio Silvestri. Sorting out the document identifier assignment problem. In Proceedings of 29th European Conference on IR Research, pages 101-112, 2007.

[28] 法布里齐奥·西尔维斯特里（Fabrizio Silvestri）。解决文档标识符分配问题。收录于《第 29 届欧洲信息检索研究会议论文集》，第 101 - 112 页，2007 年。

[29] Fabrizio Silvestri and Domenico LaForenza. Query-driven document partitioning and collection selection. In Proceedings of the First International Conference on Scalable Information Systems, 2006.

[29] 法布里齐奥·西尔维斯特里（Fabrizio Silvestri）和多梅尼科·拉福伦扎（Domenico LaForenza）。查询驱动的文档分区与集合选择。收录于《第一届可扩展信息系统国际会议论文集》，2006 年。

[30] T. Strohman and W. Bruce Croft. Efficient document retrieval in main memory. In Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2007.

[30] T. 斯特罗曼（T. Strohman）和 W. 布鲁斯·克罗夫特（W. Bruce Croft）。主存中的高效文档检索。收录于《第 30 届 ACM SIGIR 国际信息检索研究与发展年会论文集》，2007 年。

[31] Trevor Strohman, Howard Turtle, and Bruce W. Croft. Optimization strategies for complex queries. In Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2005.

[31] 特雷弗·斯特罗曼（Trevor Strohman）、霍华德·特特尔（Howard Turtle）和布鲁斯·W. 克罗夫特（Bruce W. Croft）。复杂查询的优化策略。收录于《第 28 届 ACM SIGIR 国际信息检索研究与发展年会论文集》，2005 年。

[32] H. Turtle and J. Flood. Query evaluation: strategies and optimizations. Information Processing and Management, 31(6):831-850, November 1995.

[32] H. 特特尔（H. Turtle）和 J. 弗勒德（J. Flood）。查询评估：策略与优化。《信息处理与管理》，31(6):831 - 850，1995 年 11 月。

[33] HW. Wong and D. Lee. Implementations of partial document ranking using inverted files. Information Processing and Management, 29(5):647-669, 1993.

[33] H.W. 黄（H.W. Wong）和 D. 李（D. Lee）。使用倒排文件实现部分文档排名。《信息处理与管理》，29(5):647 - 669，1993 年。

[34] Hao Yan, Shuai Ding, and Torsten Suel. Compressing term positions in web indexes. In Proceedings of the 32th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2009.

[34] 闫浩（Hao Yan）、丁帅（Shuai Ding）和托尔斯滕·苏埃尔（Torsten Suel）。压缩网络索引中的词项位置。收录于《第 32 届 ACM 信息检索研究与发展国际年会论文集》，2009 年。

[35] Hao Yan, Shuai Ding, and Torsten Suel. Inverted index compression and query processing with optimized document ordering. In Proceedings of the 18th International Conference on World Wide Web, 2009.

[35] 闫浩（Hao Yan）、丁帅（Shuai Ding）和托尔斯滕·苏埃尔（Torsten Suel）。采用优化文档排序的倒排索引压缩与查询处理。收录于《第 18 届万维网国际会议论文集》，2009 年。

[36] J. Zhang, X. Long, and T. Suel. Performance of compressed inverted list caching in search engines. In Proceedings of the 17th International Conference on World Wide Web, 2008.

[36] 张（J. Zhang）、龙（X. Long）和苏埃尔（T. Suel）。搜索引擎中压缩倒排列表缓存的性能。收录于《第 17 届万维网国际会议论文集》，2008 年。

[37] J. Zobel and A. Moffat. Inverted files for text search engines. ACM Computing Surveys, 38(2), 2006.

[37] J. 佐贝尔（J. Zobel）和 A. 莫法特（A. Moffat）。文本搜索引擎的倒排文件。《ACM 计算调查》，38(2)，2006 年。