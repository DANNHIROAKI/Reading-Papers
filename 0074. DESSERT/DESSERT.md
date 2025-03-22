# DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries

# DESSERT：一种用于向量集查询的向量集搜索高效算法

Joshua Engels

约书亚·恩格尔斯

ThirdAI

ThirdAI公司

josh.adam.engels@gmail.com

Benjamin Coleman

本杰明·科尔曼

ThirdAI

ThirdAI公司

benjamin.ray.coleman@gmail.com

Vihan Lakshman

维汉·拉克什曼

ThirdAI

ThirdAI公司

vihan@thirdai.com

Anshumali Shrivastava

安舒马利·什里瓦斯塔瓦

ThirdAI, Rice University

ThirdAI公司，莱斯大学

anshu@thirdai.com, anshumali@rice.edu

anshu@thirdai.com，anshumali@rice.edu

## Abstract

## 摘要

We study the problem of vector set search with vector set queries. This task is analogous to traditional near-neighbor search, with the exception that both the query and each element in the collection are sets of vectors. We identify this problem as a core subroutine for semantic search applications and find that existing solutions are unacceptably slow. Towards this end, we present a new approximate search algorithm, DESSERT (DESSERT Effeciently Searches Sets of Embeddings via Retrieval Tables). DESSERT is a general tool with strong theoretical guarantees and excellent empirical performance. When we integrate DESSERT into ColBERT, a state-of-the-art semantic search model, we find a 2-5x speedup on the MS MARCO and LoTTE retrieval benchmarks with minimal loss in recall, underscoring the effectiveness and practical applicability of our proposal.

我们研究了使用向量集查询进行向量集搜索的问题。这项任务与传统的近邻搜索类似，不同之处在于查询和集合中的每个元素都是向量集。我们将此问题确定为语义搜索应用的核心子程序，并发现现有的解决方案速度慢得令人无法接受。为此，我们提出了一种新的近似搜索算法，DESSERT（DESSERT通过检索表高效搜索嵌入集）。DESSERT是一种通用工具，具有强大的理论保证和出色的实证性能。当我们将DESSERT集成到最先进的语义搜索模型ColBERT中时，我们发现在MS MARCO和LoTTE检索基准测试中速度提高了2 - 5倍，同时召回率损失极小，这凸显了我们所提方案的有效性和实际适用性。

## 1 Introduction

## 1 引言

Similarity search is a fundamental driver of performance for many high-profile machine learning applications. Examples include web search [16], product recommendation [33], image search [21], de-duplication of web indexes [29] and friend recommendation for social media networks [39]. In this paper,we study a variation on the traditional vector search problem where the dataset $D$ consists of a collection of vector sets $D = \left\{  {{S}_{1},\ldots {S}_{N}}\right\}$ and the query $Q$ is also a vector set. We call this problem vector set search with vector set queries because both the collection elements and the query are sets of vectors. Unlike traditional vector search, this problem currently lacks a satisfactory solution.

相似性搜索是许多备受瞩目的机器学习应用性能的基本驱动力。示例包括网络搜索[16]、产品推荐[33]、图像搜索[21]、网络索引去重[29]和社交媒体网络的好友推荐[39]。在本文中，我们研究了传统向量搜索问题的一个变体，其中数据集$D$由一组向量集$D = \left\{  {{S}_{1},\ldots {S}_{N}}\right\}$组成，查询$Q$也是一个向量集。我们将此问题称为使用向量集查询的向量集搜索，因为集合元素和查询都是向量集。与传统的向量搜索不同，这个问题目前缺乏令人满意的解决方案。

Furthermore, efficiently solving the vector set search problem has immediate practical implications. Most notably, the popular ColBERT model, a state-of-the-art neural architecture for semantic search over documents [23], achieves breakthrough performance on retrieval tasks by representing each query and document as a set of BERT token embeddings. ColBERT's current implementation of vector set search over these document sets, while superior to brute force, is prohibitively slow for real-time inference applications like e-commerce that enforce strict search latencies under 20-30 milliseconds $\left\lbrack  {5,{34}}\right\rbrack$ . Thus,a more efficient algorithm for searching over sets of vectors would have significant implications in making state-of-the-art semantic search methods feasible to deploy in large-scale production settings, particularly on cost-effective CPU hardware.

此外，高效解决向量集搜索问题具有直接的实际意义。最值得注意的是，流行的ColBERT模型是一种用于文档语义搜索的最先进的神经架构[23]，它通过将每个查询和文档表示为一组BERT词元嵌入来在检索任务上取得突破性性能。ColBERT目前对这些文档集进行向量集搜索的实现虽然优于暴力搜索，但对于像电子商务这样要求严格搜索延迟在20 - 30毫秒以下的实时推理应用来说速度慢得令人难以接受$\left\lbrack  {5,{34}}\right\rbrack$。因此，一种更高效的向量集搜索算法将对使最先进的语义搜索方法能够在大规模生产环境中部署具有重大意义，特别是在经济高效的CPU硬件上。

Given ColBERT's success in using vector sets to represent documents more accurately, and the prevailing focus on traditional single-vector near-neighbor search in the literature $\lbrack 1,{12},{14},{18}$ , ${19},{28},{41}\rbrack$ ,we believe that the potential for searching over sets of representations remains largely untapped. An efficient algorithmic solution to this problem could enable new applications in domains where multi-vector representations are more suitable. To that end, we propose DESSERT, a novel randomized algorithm for efficient set vector search with vector set queries. We also provide a general theoretical framework for analyzing DESSERT and evaluate its performance on standard passage ranking benchmarks,achieving a $2 - 5\mathrm{x}$ speedup over an optimized ColBERT implementation on several passage retrieval tasks.

鉴于ColBERT在使用向量集更准确地表示文档方面取得的成功，以及文献中普遍关注传统的单向量近邻搜索$\lbrack 1,{12},{14},{18}$，${19},{28},{41}\rbrack$，我们认为对表示集进行搜索的潜力在很大程度上尚未得到挖掘。针对这个问题的高效算法解决方案可以在多向量表示更合适的领域中实现新的应用。为此，我们提出了DESSERT，一种用于使用向量集查询进行高效集向量搜索的新型随机算法。我们还提供了一个用于分析DESSERT的通用理论框架，并在标准段落排名基准测试中评估其性能，在多个段落检索任务上比优化后的ColBERT实现提高了$2 - 5\mathrm{x}$倍的速度。

### 1.1 Problem Statement

### 1.1 问题陈述

More formally, we consider the following problem statement.

更正式地说，我们考虑以下问题陈述。

Definition 1.1. Given a collection of $N$ vector sets $D = \left\{  {{S}_{1},\ldots {S}_{N}}\right\}$ ,a query set $Q$ ,a failure probability $\delta  \geq  0$ ,and a set-to-set relevance score function $F\left( {Q,S}\right)$ ,the Vector Set Search Problem is the task of returning ${S}^{ * }$ with probability at least $1 - \delta$ :

定义1.1。给定一组$N$个向量集$D = \left\{  {{S}_{1},\ldots {S}_{N}}\right\}$、一个查询集$Q$、一个失败概率$\delta  \geq  0$以及一个集对集相关性得分函数$F\left( {Q,S}\right)$，向量集搜索问题是指以至少$1 - \delta$的概率返回${S}^{ * }$的任务：

$$
{S}^{ \star  } = \mathop{\operatorname{argmax}}\limits_{{i \in  \{ 1,\ldots N\} }}F\left( {Q,{S}_{i}}\right) 
$$

Here,each set ${S}_{i} = \left\{  {{x}_{1},\ldots {x}_{{m}_{i}}}\right\}$ contains ${m}_{i}$ vectors with each ${x}_{j} \in  {\mathbb{R}}^{d}$ ,and similarly $Q =$ $\left\{  {{q}_{1},\ldots {q}_{{m}_{q}}}\right\}$ contains ${m}_{q}$ vectors with each ${q}_{j} \in  {\mathbb{R}}^{d}$ .

这里，每个集合${S}_{i} = \left\{  {{x}_{1},\ldots {x}_{{m}_{i}}}\right\}$包含${m}_{i}$个向量，每个向量为${x}_{j} \in  {\mathbb{R}}^{d}$，类似地，$Q =$ $\left\{  {{q}_{1},\ldots {q}_{{m}_{q}}}\right\}$包含${m}_{q}$个向量，每个向量为${q}_{j} \in  {\mathbb{R}}^{d}$。

We further restrict our consideration to structured forms of $F\left( {Q,S}\right)$ ,where the relevance score consists of two "set aggregation" or "variadic" functions. The inner aggregation $\sigma$ operates on the pairwise similarities between a single vector from the query set and each vector from the target set. Because there are $\left| S\right|$ elements in $S$ over which to perform the aggregation, $\sigma$ takes $\left| S\right|$ arguments. The outer aggregation $A$ operates over the $\left| Q\right|$ scores obtained by applying $A$ to each query vector $q \in  Q$ . Thus,we have that

我们进一步将考虑范围限制在$F\left( {Q,S}\right)$的结构化形式上，其中相关性得分由两个“集合聚合”或“可变参数”函数组成。内部聚合$\sigma$对查询集中的单个向量与目标集中的每个向量之间的成对相似度进行操作。因为要在$S$中的$\left| S\right|$个元素上进行聚合，所以$\sigma$接受$\left| S\right|$个参数。外部聚合$A$对通过将$A$应用于每个查询向量$q \in  Q$而获得的$\left| Q\right|$个得分进行操作。因此，我们有

$$
F\left( {Q,S}\right)  = A\left( \left\{  {{\text{ Inner }}_{q,S} : q \in  Q}\right\}  \right) 
$$

$$
{\text{Inner}}_{q,S} = \sigma \left( {\{ \operatorname{sim}\left( {q,x}\right)  : x \in  S\} }\right) 
$$

Here, sim is a vector similarity function. Because the inner aggregation is often a maximum or other non-linearity,we use $\sigma \left( \cdot \right)$ to denote it,and similarly since the outer aggregation is often a linear function we denote it with $A\left( \cdot \right)$ . These structured forms for $F = A \circ  \sigma$ are a good measure of set similarity when they are monotonically non-decreasing with respect to the similarity between any pair of vectors from $Q$ and $S$ .

这里，sim是一个向量相似度函数。由于内部聚合通常是最大值或其他非线性函数，我们用$\sigma \left( \cdot \right)$来表示它，类似地，由于外部聚合通常是线性函数，我们用$A\left( \cdot \right)$来表示它。当$F = A \circ  \sigma$的这些结构化形式相对于$Q$和$S$中任意一对向量之间的相似度单调非减时，它们是集合相似度的一个良好度量。

### 1.2 Why is near-neighbor search insufficient?

### 1.2 为什么近邻搜索不够？

It may at first seem that we could solve the Vector Set Search Problem by placing all of the individual vectors into a near-neighbor index, along with metadata indicating the set to which they belonged. One could then then identify high-scoring sets by finding near neighbors to each $q \in  Q$ and returning their corresponding sets.

起初，我们似乎可以通过将所有单个向量放入一个近邻索引中，并附带指示它们所属集合的元数据，来解决向量集搜索问题。然后，人们可以通过找到每个$q \in  Q$的近邻并返回它们对应的集合来识别高分集合。

There are two problems with this approach. The first problem is that a single high-similarity interaction between $q \in  Q$ and $x \in  S$ does not imply that $F\left( {Q,S}\right)$ will be large. For a concrete example,suppose that we are dealing with sets of word embeddings and that $Q$ is a phrase where one of the items is "keyword." With a standard near-neighbor index, $Q$ will match (with 100% similarity) any set $S$ that also contains "keyword," regardless of whether the other words in $S$ bear any relevance to the other words in $Q$ . The second problem is that the search must be conducted over all individual vectors, leading to a search problem that is potentially very large. For example, if our sets are documents consisting of roughly a thousand words and we wish to search over a million documents, we now have to solve a billion-scale similarity search problem.

这种方法有两个问题。第一个问题是，$q \in  Q$和$x \in  S$之间的单个高相似度交互并不意味着$F\left( {Q,S}\right)$会很大。举一个具体的例子，假设我们处理的是词嵌入集合，并且$Q$是一个短语，其中一个项是“关键词”。使用标准的近邻索引，$Q$将与任何也包含“关键词”的集合$S$匹配（相似度为100%），而不管$S$中的其他词与$Q$中的其他词是否有任何相关性。第二个问题是，搜索必须在所有单个向量上进行，这导致搜索问题可能非常大。例如，如果我们的集合是由大约一千个单词组成的文档，并且我们希望在一百万个文档上进行搜索，那么我们现在必须解决一个十亿规模的相似度搜索问题。

Contributions: In this work, we formulate and carefully study the set of vector search problem with the goal of developing a more scalable algorithm capable of tackling large-scale semantic retrieval problems involving sets of embeddings. Specifically, our research contributions can be summarized as follows:

贡献：在这项工作中，我们提出并仔细研究了向量集搜索问题，目标是开发一种更具可扩展性的算法，能够处理涉及嵌入集合的大规模语义检索问题。具体来说，我们的研究贡献可以总结如下：

1. We develop the first non-trivial algorithm, DESSERT, for the vector set search problem that scales to large collections $\left( {n > {10}^{6}}\right)$ of sets with $m > 3$ items.

1. 我们为向量集搜索问题开发了第一个非平凡的算法DESSERT，该算法可以扩展到包含$m > 3$个项的大型集合$\left( {n > {10}^{6}}\right)$。

2. We formalize the vector set search problem in a rigorous theoretical framework, and we provide strong guarantees for a common (and difficult) instantiation of the problem.

2. 我们在严格的理论框架中对向量集搜索问题进行了形式化，并为该问题的一个常见（且困难）实例提供了强有力的保证。

3. We provide an open-source $\mathrm{C} +  +$ implementation of our proposed algorithm that has been deployed in a real-world production setting ${}^{1}$ . Our implementation scales to hundreds of millions of vectors and is $3 - {5x}$ faster than existing approximate set of vector search techniques. We also describe the implementation details and tricks we discovered to achieve these speedups and provide empirical latency and recall results on passage retrieval tasks.

3. 我们提供了所提出算法的开源 $\mathrm{C} +  +$ 实现，该实现已部署在实际生产环境中 ${}^{1}$。我们的实现可扩展到数亿个向量，并且比现有的近似向量搜索技术集 $3 - {5x}$ 更快。我们还描述了为实现这些加速而发现的实现细节和技巧，并提供了段落检索任务的经验延迟和召回率结果。

## 2 Related Work

## 2 相关工作

Near-Neighbor Search: Near-neighbor search has received heightened interest in recent years with the advent of vector-based representation learning. In particular, considerable research has gone into developing more efficient approximate near-neighbor (ANN) search methods that trade off an exact solution for sublinear query times. A number of ANN algorithms have been proposed, including those based on locality-sensitive hashing $\left\lbrack  {1,{41}}\right\rbrack$ ,quantization and space partition methods $\left\lbrack  {{12},{14},{19}}\right\rbrack$ ,and graph-based methods $\left\lbrack  {{18},{28}}\right\rbrack$ . Among these classes of techniques,our proposed DESSERT framework aligns most closely with the locality-sensitive hashing paradigm. However, nearly all of the well-known and effective ANN methods focus on searching over individual vectors; our work studies the search problem for sets of entities. This modification changes the nature of the problem considerably, particularly with regards to the choice of similarity metrics between entities.

近邻搜索：近年来，随着基于向量的表示学习的出现，近邻搜索受到了更多关注。特别是，大量研究致力于开发更高效的近似近邻（ANN）搜索方法，这些方法以牺牲精确解为代价来换取亚线性查询时间。已经提出了许多 ANN 算法，包括基于局部敏感哈希 $\left\lbrack  {1,{41}}\right\rbrack$、量化和空间划分方法 $\left\lbrack  {{12},{14},{19}}\right\rbrack$ 以及基于图的方法 $\left\lbrack  {{18},{28}}\right\rbrack$。在这些技术类别中，我们提出的 DESSERT 框架与局部敏感哈希范式最为契合。然而，几乎所有知名且有效的 ANN 方法都专注于对单个向量进行搜索；我们的工作研究的是实体集合的搜索问题。这种修改极大地改变了问题的性质，特别是在实体之间相似性度量的选择方面。

Vector Set Search: The general problem of vector set search has been relatively understudied in the literature. A recent work on database lineage tracking [25] addresses this precise problem, but with severe limitations. The proposed approximate algorithm designs a concatenation scheme for the vectors in a given set, and then performs approximate search over these concatenated vectors. The biggest drawback to this method is scalability, as the size of the concatenated vectors scales quadratically with the size of the vector set. This leads to increased query latency as well as substantial memory overhead; in fact, we are unable to apply the method to the datasets in this paper without terabytes of RAM. In this work, we demonstrate that DESSERT can scale to thousands of items per set with a linear increase (and a slight logarithmic overhead) in query time, which, to our knowledge, has not been previously demonstrated in the literature.

向量集搜索：向量集搜索的一般问题在文献中相对研究较少。最近一项关于数据库谱系跟踪的工作 [25] 解决了这个确切的问题，但存在严重的局限性。所提出的近似算法为给定集合中的向量设计了一种拼接方案，然后对这些拼接后的向量进行近似搜索。这种方法的最大缺点是可扩展性，因为拼接后向量的大小与向量集的大小呈二次方关系。这导致查询延迟增加以及大量的内存开销；事实上，如果没有数 TB 的随机存取存储器（RAM），我们无法将该方法应用于本文中的数据集。在这项工作中，我们证明了 DESSERT 可以扩展到每个集合包含数千个项目，查询时间呈线性增加（并有轻微的对数开销），据我们所知，这在文献中尚未得到证明。

Document Retrieval: In the problem of document retrieval, we receive queries and must return the relevant documents from a preindexed corpus. Early document retrieval methods treated each documents as bags of words and had at their core an inverted index [30]. More recent methods embed each document into a single representative vector, embed the query into the same space, and performed ANN search on those vectors. These semantic methods achieve far greater accuracies than their lexical predecessors, but require similarity search instead of inverted index lookups [15, 26, 33].

文档检索：在文档检索问题中，我们接收查询并必须从预索引的语料库中返回相关文档。早期的文档检索方法将每个文档视为词袋，其核心是倒排索引 [30]。最近的方法将每个文档嵌入到一个代表性向量中，将查询嵌入到相同的空间中，并对这些向量进行 ANN 搜索。这些语义方法比其基于词法的前辈们获得了更高的准确率，但需要进行相似性搜索而不是倒排索引查找 [15, 26, 33]。

ColBERT and PLAID: ColBERT [23] is a recent state of the art algorithm for document retrieval that takes a subtly different approach. Instead of generating a single vector per document, ColBERT generates a set of vectors for each document, approximately one vector per word. To rank a query, ColBERT also embeds the query into a set of vectors, filters the indexed sets, and then performs a brute force sum of max similarities operation between the query set and each of the document sets. ColBERT’s passage ranking system is an instantiation of our framework,where $\operatorname{sim}\left( {q,x}\right)$ is the cosine similarity between vectors, $\sigma$ is the max operation,and $A$ is the sum operation.

ColBERT 和 PLAID：ColBERT [23] 是最近用于文档检索的最先进算法，采用了一种略有不同的方法。ColBERT 不是为每个文档生成一个单一的向量，而是为每个文档生成一组向量，大约每个单词对应一个向量。为了对查询进行排序，ColBERT 还将查询嵌入到一组向量中，过滤索引集，然后在查询集和每个文档集之间执行最大相似度的暴力求和操作。ColBERT 的段落排序系统是我们框架的一个实例，其中 $\operatorname{sim}\left( {q,x}\right)$ 是向量之间的余弦相似度，$\sigma$ 是最大值操作，$A$ 是求和操作。

In a similar spirit to our work, PLAID [37] is a recently optimized form of ColBERT that includes more efficient filtering techniques and faster quantization based set similarity kernels. However, we note that these techniques are heuristics that do not come with theoretical guarantees and do not immediately generalize to other notions of vector similarity, which is a key property of the theoretical framework behind DESSERT.

与我们的工作类似，PLAID [37] 是 ColBERT 的一种最近优化形式，它包括更高效的过滤技术和基于量化的更快的集合相似度核。然而，我们注意到这些技术是启发式方法，没有理论保证，并且不能立即推广到其他向量相似度概念，而这是 DESSERT 背后理论框架的一个关键特性。

## 3 Algorithm

## 3 算法

At a high level,a DESSERT index $\mathcal{D}$ compresses the collection of target sets into a form that makes set to set similarity operations efficient to calculate. This is done by replacing each set ${S}_{i}$ with a sketch $\mathcal{D}\left\lbrack  i\right\rbrack$ that contains the LSH values of each ${x}_{j} \in  {S}_{i}$ . At query time,we compare the corresponding LSH values of the query set $Q$ with the hashes in each $\mathcal{D}\left\lbrack  i\right\rbrack$ to approximate the pairwise similarity matrix between $Q$ and $S$ (Figure 1). This matrix is used as the input for the aggregation functions $A$ and $\sigma$ to rank the target sets and return an estimate of ${S}^{ * }$ .

从高层次来看，DESSERT索引$\mathcal{D}$将目标集合压缩成一种形式，使得集合间的相似度运算能够高效计算。这是通过用一个包含每个${x}_{j} \in  {S}_{i}$的局部敏感哈希（LSH）值的草图$\mathcal{D}\left\lbrack  i\right\rbrack$来替换每个集合${S}_{i}$实现的。在查询时，我们将查询集合$Q$对应的LSH值与每个$\mathcal{D}\left\lbrack  i\right\rbrack$中的哈希值进行比较，以近似计算$Q$和$S$之间的成对相似度矩阵（图1）。该矩阵作为聚合函数$A$和$\sigma$的输入，用于对目标集合进行排序并返回${S}^{ * }$的估计值。

---

<!-- Footnote -->

${}^{1}$ https://github.com/ThirdAIResearch/Dessert

${}^{1}$ https://github.com/ThirdAIResearch/Dessert

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Indexing Querying Query Set Set Relevance -->

<img src="https://cdn.noedgeai.com/0195a865-affe-7ef5-9800-93ec0736f0e2_3.jpg?x=433&y=206&w=941&h=423&r=0"/>

Figure 1: The DESSERT indexing and querying algorithms. During indexing (left), we represent each target set as a set of hash values ( $L$ hashes for each element). To query the index (right),we approximate the similarity between each target and query element by averaging the number of hash collisions. These similarities are used to approximate the set relevance score for each target set.

图1：DESSERT索引和查询算法。在索引阶段（左图），我们将每个目标集合表示为一组哈希值（每个元素对应$L$个哈希值）。在查询索引时（右图），我们通过对哈希冲突的数量求平均值来近似计算每个目标元素和查询元素之间的相似度。这些相似度用于近似计算每个目标集合的集合相关性得分。

<!-- Media -->

We assume the existence of a locality-sensitive hashing (LSH) family $\mathcal{H} \subset  \left( {{\mathbb{R}}^{d} \rightarrow  \mathbb{Z}}\right)$ such that for all LSH functions $h \in  \mathcal{H},p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}\left( {x,y}\right)$ . LSH functions with this property exist for cosine similarity (signed random projections) [8],Euclidean similarity $(p$ -stable projections) [11], and Jaccard similarity (minhash or simhash) [6]. LSH is a well-developed theoretical framework with a wide variety of results and extensions $\left\lbrack  {3,4,{20},{40}}\right\rbrack$ . See Appendix C for a deeper overview.

我们假设存在一个局部敏感哈希（LSH）族$\mathcal{H} \subset  \left( {{\mathbb{R}}^{d} \rightarrow  \mathbb{Z}}\right)$，使得对于所有的LSH函数$h \in  \mathcal{H},p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}\left( {x,y}\right)$都成立。具有这种性质的LSH函数在余弦相似度（有符号随机投影）[8]、欧几里得相似度（$(p$ - 稳定投影）[11]和杰卡德相似度（最小哈希或相似哈希）[6]中都存在。LSH是一个发展成熟的理论框架，有各种各样的研究成果和扩展$\left\lbrack  {3,4,{20},{40}}\right\rbrack$。更深入的概述请参见附录C。

Algorithm 1 describes how to construct a DESSERT index $\mathcal{D}$ . We first take $L$ LSH functions ${f}_{t}$ for $t \in  \left\lbrack  {1,L}\right\rbrack  ,{f}_{t} \in  \mathcal{H}$ . We next loop over each ${S}_{i}$ to construct $\mathcal{D}\left\lbrack  i\right\rbrack$ . For a given ${S}_{i}$ ,we arbitrarily assign an identifier $j$ to each vector $x \in  {S}_{i},j \in  \left\lbrack  {1,\left| {S}_{i}\right| }\right\rbrack$ . We next partition the set $\left\lbrack  {1,{m}_{i}}\right\rbrack$ using each hash function ${h}_{t}$ ,such that for a partition ${p}_{t}$ ,indices ${j}_{1}$ and ${j}_{2}$ are in the same set in the partition iff $h\left( {S}_{{j}_{1}}\right)  = h\left( {S}_{{j}_{2}}\right)$ . We represent the results of these partitions in a universal hash table indexed by hash function id and hash function value,such that $\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h} = \left\{  {j \mid  {x}_{j} \in  {S}_{i} \land  {f}_{t}\left( {x}_{j}\right)  = h}\right\}$ .

算法1描述了如何构建DESSERT索引$\mathcal{D}$。我们首先为$t \in  \left\lbrack  {1,L}\right\rbrack  ,{f}_{t} \in  \mathcal{H}$选取$L$个LSH函数${f}_{t}$。接着，我们遍历每个${S}_{i}$来构建$\mathcal{D}\left\lbrack  i\right\rbrack$。对于给定的${S}_{i}$，我们为每个向量$x \in  {S}_{i},j \in  \left\lbrack  {1,\left| {S}_{i}\right| }\right\rbrack$任意分配一个标识符$j$。然后，我们使用每个哈希函数${h}_{t}$对集合$\left\lbrack  {1,{m}_{i}}\right\rbrack$进行划分，使得对于一个划分${p}_{t}$，当且仅当$h\left( {S}_{{j}_{1}}\right)  = h\left( {S}_{{j}_{2}}\right)$时，索引${j}_{1}$和${j}_{2}$在该划分的同一个集合中。我们在一个以哈希函数ID和哈希函数值为索引的通用哈希表中表示这些划分的结果，即$\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h} = \left\{  {j \mid  {x}_{j} \in  {S}_{i} \land  {f}_{t}\left( {x}_{j}\right)  = h}\right\}$。

Algorithm 2 describes how to query a DESSERT index $\mathcal{D}$ . At a high level,we query each sketch ${\mathcal{D}}_{i}$ to get an estimate of $F\left( {Q,{S}_{i}}\right)$ ,score ${i}_{i}$ ,and then take the argmax over the estimates to get an estimate of ${\operatorname{argmax}}_{i \in  \{ 1,\ldots N\} }F\left( {Q,{S}_{i}}\right)$ . To get these estimates,we first compute the hashes ${h}_{t,q}$ for each query $q$ and LSH function ${f}_{t}$ . Then,to get an estimate ${\operatorname{score}}_{i}$ for a set ${S}_{i}$ ,we loop over the hashes ${h}_{t,q}$ for each query vector $q$ and count how often each index $j$ appears in $\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,{h}_{t,q}}$ . After we finish this step,we have a count for each $j$ that represents how many times ${h}_{t}\left( q\right)  = {h}_{t}\left( {x}_{j}\right)$ . Equivalently, since $p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}\left( {x,y}\right)$ ,if we divide by $L$ we have an estimate for $\operatorname{sim}\left( {{x}_{j},q}\right)$ . We then apply $\sigma$ to these estimates and save the result in a variable ${\operatorname{agg}}_{q}$ to build up the inputs to $A$ ,and then apply $A$ to get our final estimate for $F\left( {Q,{S}_{i}}\right)$ ,which we store in ${\operatorname{score}}_{i}$ .

算法2描述了如何查询DESSERT索引$\mathcal{D}$。从高层次来看，我们查询每个草图${\mathcal{D}}_{i}$以获得$F\left( {Q,{S}_{i}}\right)$的估计值、得分${i}_{i}$，然后对这些估计值取最大值索引以获得${\operatorname{argmax}}_{i \in  \{ 1,\ldots N\} }F\left( {Q,{S}_{i}}\right)$的估计值。为了得到这些估计值，我们首先为每个查询$q$和LSH函数${f}_{t}$计算哈希值${h}_{t,q}$。然后，为了得到集合${S}_{i}$的估计值${\operatorname{score}}_{i}$，我们遍历每个查询向量$q$的哈希值${h}_{t,q}$，并统计每个索引$j$在$\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,{h}_{t,q}}$中出现的频率。完成这一步后，我们得到每个$j$的计数，该计数表示${h}_{t}\left( q\right)  = {h}_{t}\left( {x}_{j}\right)$出现的次数。等价地，由于$p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}\left( {x,y}\right)$，如果我们将其除以$L$，就可以得到$\operatorname{sim}\left( {{x}_{j},q}\right)$的估计值。然后，我们对这些估计值应用$\sigma$，并将结果保存在变量${\operatorname{agg}}_{q}$中，以构建$A$的输入，接着应用$A$得到$F\left( {Q,{S}_{i}}\right)$的最终估计值，并将其存储在${\operatorname{score}}_{i}$中。

## 4 Theory

## 4 理论

In this section, we analyze DESSERT's query runtime and provide probabilistic bounds on the correctness of its search results. We begin by finding the hyperparameter values and conditions that are necessary for DESSERT to return the top-ranked set with high probability. Then, we use these results to prove bounds on the query time. In the interest of space, we defer proofs to the Appendix.

在本节中，我们分析DESSERT的查询运行时间，并为其搜索结果的正确性提供概率界限。我们首先找出DESSERT以高概率返回排名最高的集合所需的超参数值和条件。然后，我们利用这些结果来证明查询时间的界限。为了节省篇幅，我们将证明过程放在附录中。

Notation: For the sake of simplicity of presentation, we suppose that all target sets have the same number of elements $m$ ,i.e. $\left| {S}_{i}\right|  = m$ . If this is not the case,one may replace ${m}_{i}$ with ${m}_{\max }$ in our analysis. We will use the boldface vector $\mathbf{s}\left( {q,{S}_{i}}\right)  \in  {\mathbb{R}}^{\left| {S}_{i}\right| }$ to refer to the set of pairwise similarity calculations $\left\{  {\operatorname{sim}\left( {q,{x}_{1}}\right) ,\ldots ,\operatorname{sim}\left( {q,{x}_{{m}_{i}}}\right) }\right\}$ between a query vector and the elements of ${S}_{i}$ ,and we will drop the subscript $\left( {q,{S}_{i}}\right)$ when the context is clear. See Table 1 for a complete notation reference.

符号说明：为了表述简单，我们假设所有目标集合具有相同数量的元素$m$，即$\left| {S}_{i}\right|  = m$。如果情况并非如此，在我们的分析中可以用${m}_{\max }$代替${m}_{i}$。我们将使用粗体向量$\mathbf{s}\left( {q,{S}_{i}}\right)  \in  {\mathbb{R}}^{\left| {S}_{i}\right| }$来表示查询向量与${S}_{i}$的元素之间的成对相似度计算集合$\left\{  {\operatorname{sim}\left( {q,{x}_{1}}\right) ,\ldots ,\operatorname{sim}\left( {q,{x}_{{m}_{i}}}\right) }\right\}$，并且在上下文明确时，我们将省略下标$\left( {q,{S}_{i}}\right)$。完整的符号参考见表1。

<!-- Media -->

Table 1: Notation table with examples from the document search application.

表1：包含文档搜索应用示例的符号表。

<table><tr><td>Notation</td><td>Definition</td><td>Intuition (Document Search)</td></tr><tr><td>$D$</td><td>Set of target vector sets</td><td>Collection of documents</td></tr><tr><td>$N$</td><td>Cardinality $\left| D\right|$</td><td>Number of documents</td></tr><tr><td>$\mathcal{D}$</td><td>DESSERT index of $D$</td><td>Search index data structure</td></tr><tr><td>${S}_{i}$</td><td>Target vector set $i$</td><td>$i$ th document</td></tr><tr><td>Q</td><td>Query vector set</td><td>Multi-word query (e.g., a question)</td></tr><tr><td>${S}^{ * }$</td><td>See Definition 1.1</td><td>The most relevant document to $Q$</td></tr><tr><td>${x}_{j} \in  {S}_{i}$</td><td>$j$ th vector in target set ${S}_{i}$</td><td>Embedding from document $i$</td></tr><tr><td>${q}_{j} \in  Q$</td><td>$j$ th vector in query set $Q$</td><td>Embedding from a query</td></tr><tr><td>$d$</td><td>${s}_{j},{x}_{j} \in  {\mathbb{R}}^{d}$</td><td>Embedding dimension</td></tr><tr><td>${m}_{i},m$</td><td>Cardinality $\left| {S}_{i}\right| ,{m}_{i} = m$</td><td>Number of embeddings in $i$ th document</td></tr><tr><td>$F\left( {Q,{S}_{i}}\right)$</td><td>$Q$ and ${S}_{i}$ relevance score</td><td>Measures query-document similarity</td></tr><tr><td>score ${}_{i}$</td><td>Estimate of $F\left( {Q,{S}_{i}}\right)$</td><td>Approximation of relevance score</td></tr><tr><td>$\mathcal{D}\left\lbrack  i\right\rbrack$</td><td>Sketch of $i$ th target set</td><td>Estimates relevance score for ${S}_{i}$ and any $Q$</td></tr><tr><td>$\operatorname{sim}\left( {a,b}\right)$</td><td>$a$ and $b$ vector similarity</td><td>Embedding similarity</td></tr><tr><td>$A,\sigma$</td><td>See Section 1.1</td><td>Components of relevance score</td></tr><tr><td>$L$</td><td>Number of hashes</td><td>Larger $L$ increases accuracy and latency</td></tr><tr><td>${f}_{i}$</td><td>$i$ th LSH function</td><td>Often maps nearby points to the same value</td></tr><tr><td>$\mathbf{s}\left( {q,{S}_{i}}\right) ,\mathbf{s}$</td><td>$\operatorname{sim}\left( {q,{x}_{j}}\right)$ for ${x}_{j} \in  {S}_{i}$</td><td>Query embedding similarities with ${S}_{i}$</td></tr></table>

<table><tbody><tr><td>符号表示</td><td>定义</td><td>直观理解（文档搜索）</td></tr><tr><td>$D$</td><td>目标向量集的集合</td><td>文档集合</td></tr><tr><td>$N$</td><td>基数 $\left| D\right|$</td><td>文档数量</td></tr><tr><td>$\mathcal{D}$</td><td>$D$ 的DESSERT索引</td><td>搜索索引数据结构</td></tr><tr><td>${S}_{i}$</td><td>目标向量集 $i$</td><td>第 $i$ 个文档</td></tr><tr><td>Q</td><td>查询向量集</td><td>多词查询（例如，一个问题）</td></tr><tr><td>${S}^{ * }$</td><td>参见定义1.1</td><td>与 $Q$ 最相关的文档</td></tr><tr><td>${x}_{j} \in  {S}_{i}$</td><td>目标集 ${S}_{i}$ 中的第 $j$ 个向量</td><td>来自文档 $i$ 的嵌入向量</td></tr><tr><td>${q}_{j} \in  Q$</td><td>查询集 $Q$ 中的第 $j$ 个向量</td><td>来自查询的嵌入向量</td></tr><tr><td>$d$</td><td>${s}_{j},{x}_{j} \in  {\mathbb{R}}^{d}$</td><td>嵌入维度</td></tr><tr><td>${m}_{i},m$</td><td>基数 $\left| {S}_{i}\right| ,{m}_{i} = m$</td><td>第 $i$ 个文档中的嵌入向量数量</td></tr><tr><td>$F\left( {Q,{S}_{i}}\right)$</td><td>$Q$ 和 ${S}_{i}$ 的相关性得分</td><td>衡量查询 - 文档相似度</td></tr><tr><td>得分 ${}_{i}$</td><td>$F\left( {Q,{S}_{i}}\right)$ 的估计值</td><td>相关性得分的近似值</td></tr><tr><td>$\mathcal{D}\left\lbrack  i\right\rbrack$</td><td>第 $i$ 个目标集的草图</td><td>估计 ${S}_{i}$ 和任意 $Q$ 的相关性得分</td></tr><tr><td>$\operatorname{sim}\left( {a,b}\right)$</td><td>$a$ 和 $b$ 的向量相似度</td><td>嵌入向量相似度</td></tr><tr><td>$A,\sigma$</td><td>参见第1.1节</td><td>相关性得分的组成部分</td></tr><tr><td>$L$</td><td>哈希函数数量</td><td>更大的 $L$ 会提高准确性和延迟</td></tr><tr><td>${f}_{i}$</td><td>第 $i$ 个局部敏感哈希（LSH）函数</td><td>通常将附近的点映射到相同的值</td></tr><tr><td>$\mathbf{s}\left( {q,{S}_{i}}\right) ,\mathbf{s}$</td><td>$\operatorname{sim}\left( {q,{x}_{j}}\right)$ 对应 ${x}_{j} \in  {S}_{i}$</td><td>查询嵌入向量与 ${S}_{i}$ 的相似度</td></tr></tbody></table>

Algorithm 1 Building a DESSERT idex

算法1：构建DESSERT索引

---

Input: $N$ sets ${S}_{i},\left| {S}_{i}\right|  = {m}_{i}$

Output: A DESSERT index $\mathcal{D}$

$\mathcal{D} =$ an array of $N$ hash tables,

each indexed by $x \in  {\mathbb{Z}}^{2}$

for $i = 1$ to $N$ do

	for ${x}_{j}$ in ${S}_{i}$ do

		for $t = 1$ to $L$ do

			$h = {f}_{t}\left( {x}_{j}\right)$

			$\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h} = \mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h} \cup  \{ j\}$

Return $\mathcal{D}$

---

Algorithm 2 Querying a DESSERT Index

算法2：查询DESSERT索引

---

1: Input: DESSERT index $\mathcal{D}$ ,query set $Q,\left| Q\right|  = {m}_{q}$ .

	Output: Estimate of ${\operatorname{argmax}}_{i \in  \{ 1\ldots N\} }A \circ  \sigma \left( {Q,{S}_{i}}\right)$

	for $q$ in $Q$ do

		${h}_{1,q},\ldots ,{h}_{L,q} = {f}_{1}\left( q\right) ,\ldots ,{f}_{L}\left( q\right)$

	for $i = 1$ to $N$ do

		for $q$ in $Q$ do

				$\widehat{\mathbf{s}} = 0$

				for $t = 1$ to $L$ do

					for $j$ in $\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,{h}_{t,q}}$ do

						${\widehat{\mathbf{s}}}_{j} = {\widehat{\mathbf{s}}}_{j} + 1$

			$\widehat{\mathbf{s}} = \widehat{\mathbf{s}}/\mathbf{L}$

				${ag}{g}_{q} = \sigma \left( \widehat{\mathbf{s}}\right)$

		${\text{score}}_{i} = A\left( \left\{  {{ag}{g}_{q} \mid  q \in  Q}\right\}  \right)$

	Return argmax ${}_{i \in  \{ 1\ldots N\} }{\text{score}}_{i}$

---

<!-- Media -->

### 4.1 Inner Aggregation

### 4.1 内部聚合

We begin by introducing a condition on the $\sigma$ component of the relevance score that allows us to prove useful statements about the retrieval process.

我们首先引入一个关于相关性得分的$\sigma$分量的条件，该条件使我们能够证明关于检索过程的有用陈述。

Definition 4.1. A function $\sigma \left( \mathbf{x}\right)  : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}$ is $\left( {\alpha ,\beta }\right)$ -maximal on $U \subset  {\mathbb{R}}^{m}$ if for $0 < \beta  \leq  1 \leq  \alpha$ , $\forall x \in  U :$

定义4.1。若对于$0 < \beta  \leq  1 \leq  \alpha$，$\forall x \in  U :$，则函数$\sigma \left( \mathbf{x}\right)  : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}$在$U \subset  {\mathbb{R}}^{m}$上是$\left( {\alpha ,\beta }\right)$ - 极大的。

$$
\beta \max \mathbf{x} \leq  \sigma \left( \mathbf{x}\right)  \leq  \alpha \max \mathbf{x}
$$

The function $\sigma \left( \mathbf{x}\right)  = \max \mathbf{x}$ is a trivial example of an $\left( {\alpha ,\beta }\right)$ -maximal function on ${\mathbb{R}}^{m}$ ,with $\beta  = \alpha  = 1$ . However,we can show that other functions also satisfy this definition:

函数$\sigma \left( \mathbf{x}\right)  = \max \mathbf{x}$是在${\mathbb{R}}^{m}$上的$\left( {\alpha ,\beta }\right)$ - 极大函数的一个平凡示例，其中$\beta  = \alpha  = 1$。然而，我们可以证明其他函数也满足这个定义：

Lemma 4.1.1. If $\varphi \left( x\right)  : \mathbb{R} \rightarrow  \mathbb{R}$ is $\left( {\alpha ,\beta }\right)$ -maximal on an interval $I$ ,then the following function $\sigma \left( x\right)  : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}$ is $\left( {\alpha ,\frac{\beta }{m}}\right)$ -maximal on $U = {I}^{m}$ :

引理4.1.1。若$\varphi \left( x\right)  : \mathbb{R} \rightarrow  \mathbb{R}$在区间$I$上是$\left( {\alpha ,\beta }\right)$ - 极大的，则以下函数$\sigma \left( x\right)  : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}$在$U = {I}^{m}$上是$\left( {\alpha ,\frac{\beta }{m}}\right)$ - 极大的：

$$
\sigma \left( \mathbf{x}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\varphi \left( {x}_{i}\right) 
$$

Note that in $\mathbb{R}$ ,the $\left( {\alpha ,\beta }\right)$ -maximal condition is equivalent to lower and upper bounds by linear functions ${\beta x}$ and ${\alpha x}$ respectively,so many natural functions satisfy Lemma 4.1.1. We are particularly interested in the case $I = \left\lbrack  {0,1}\right\rbrack$ ,and note that possible such $\varphi$ include $\varphi \left( x\right)  = x$ with $\beta  = \alpha  = 1$ , the exponential function $\varphi \left( x\right)  = {e}^{x} - 1$ with $\beta  = 1,\alpha  = e - 1$ ,and the debiased sigmoid function $\varphi \left( x\right)  = \frac{1}{1 + {e}^{-x}} - \frac{1}{2}$ with $\beta  \approx  {0.23},\alpha  = {0.25}$ . Our analysis of DESSERT holds when the $\sigma$ component of the relevance score is an $\left( {\alpha ,\beta }\right)$ maximal function.

注意，在$\mathbb{R}$中，$\left( {\alpha ,\beta }\right)$ - 极大条件分别等价于由线性函数${\beta x}$和${\alpha x}$给出的下界和上界，因此许多自然函数都满足引理4.1.1。我们特别关注$I = \left\lbrack  {0,1}\right\rbrack$的情况，并注意到可能的此类$\varphi$包括带有$\beta  = \alpha  = 1$的$\varphi \left( x\right)  = x$、带有$\beta  = 1,\alpha  = e - 1$的指数函数$\varphi \left( x\right)  = {e}^{x} - 1$以及带有$\beta  \approx  {0.23},\alpha  = {0.25}$的去偏Sigmoid函数$\varphi \left( x\right)  = \frac{1}{1 + {e}^{-x}} - \frac{1}{2}$。当相关性得分的$\sigma$分量是一个$\left( {\alpha ,\beta }\right)$ - 极大函数时，我们对DESSERT的分析成立。

In line 12 of Algorithm 2,we estimate $\sigma \left( \mathbf{s}\right)$ by applying $\sigma$ to a vector of normalized counts $\widehat{\mathbf{s}}$ . In Lemma 4.1.2,we bound the probability that a low-similarity set (one for which $\sigma \left( \mathbf{s}\right)$ is low) scores well enough to outrank a high-similarity set. In Lemma 4.1.3, we bound the probability that a high-similarity set scores poorly enough to be outranked by other sets. Note that the failure rate in both lemmas decays exponentially with the number of hash tables $L$ .

在算法2的第12行，我们通过将$\sigma$应用于归一化计数向量$\widehat{\mathbf{s}}$来估计$\sigma \left( \mathbf{s}\right)$。在引理4.1.2中，我们界定了低相似度集合（即$\sigma \left( \mathbf{s}\right)$较低的集合）得分足够高以超过高相似度集合的概率。在引理4.1.3中，我们界定了高相似度集合得分足够低以被其他集合超过的概率。注意，两个引理中的失败率都随着哈希表数量$L$呈指数衰减。

Lemma 4.1.2. Assume $\sigma$ is $\left( {\alpha ,\beta }\right)$ -maximal. Let $0 < {s}_{\max } < 1$ be the maximum similarity between a query vector and the vectors in the target set and let $\widehat{\mathbf{s}}$ be the set of estimated similarity scores. Given a threshold $\alpha {s}_{\max } < \tau  < \alpha$ ,we write $\Delta  = \tau  - \alpha {s}_{\max }$ ,and we have

引理4.1.2。假设$\sigma$是$\left( {\alpha ,\beta }\right)$ - 极大的。设$0 < {s}_{\max } < 1$是查询向量与目标集中向量之间的最大相似度，设$\widehat{\mathbf{s}}$是估计的相似度得分集合。给定一个阈值$\alpha {s}_{\max } < \tau  < \alpha$，我们记$\Delta  = \tau  - \alpha {s}_{\max }$，并且我们有

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \alpha {s}_{\max } + \Delta }\right\rbrack   \leq  m{\gamma }^{L}
$$

for $\gamma  = {\left( \frac{{s}_{\max }\left( {\alpha  - \tau }\right) }{\tau \left( {1 - {s}_{\max }}\right) }\right) }^{\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{\max }}\right) }{\alpha  - \tau }\right)  \in  \left( {{s}_{\max },1}\right)$ . Furthermore,this expression for $\gamma$ is increasing in ${s}_{\max }$ and decreasing in $\tau$ ,and $\gamma$ has one sided limits $\mathop{\lim }\limits_{{\tau  \searrow  \alpha {s}_{\max }}}\gamma  = 1$ and $\mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}\gamma  = {s}_{\max }$ . Lemma 4.1.3. With the same assumptions as Lemma 4.1.2 and given $\Delta  > 0$ ,we have:

对于 $\gamma  = {\left( \frac{{s}_{\max }\left( {\alpha  - \tau }\right) }{\tau \left( {1 - {s}_{\max }}\right) }\right) }^{\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{\max }}\right) }{\alpha  - \tau }\right)  \in  \left( {{s}_{\max },1}\right)$ 。此外，$\gamma$ 的这个表达式在 ${s}_{\max }$ 上递增，在 $\tau$ 上递减，并且 $\gamma$ 有单侧极限 $\mathop{\lim }\limits_{{\tau  \searrow  \alpha {s}_{\max }}}\gamma  = 1$ 和 $\mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}\gamma  = {s}_{\max }$ 。引理 4.1.3。在与引理 4.1.2 相同的假设下，给定 $\Delta  > 0$ ，我们有：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \leq  \beta {s}_{\max } - \Delta }\right\rbrack   \leq  2{e}^{-{2L}{\Delta }^{2}/{\beta }^{2}}
$$

### 4.2 Outer Aggregation

### 4.2 外部聚合

Our goal in this section is to use the bounds established previously to prove that our algorithm correctly ranks sets according to $F\left( {Q,S}\right)$ . To do this,we must find conditions under which the algorithm successfully identifies ${S}^{ \star  }$ based on the approximate $F\left( {Q,S}\right)$ scores.

本节的目标是利用先前建立的边界来证明我们的算法能够根据 $F\left( {Q,S}\right)$ 正确地对集合进行排序。为此，我们必须找到算法能够基于近似的 $F\left( {Q,S}\right)$ 得分成功识别 ${S}^{ \star  }$ 的条件。

Recall that $F\left( {Q,S}\right)$ consists of two aggregations: the inner aggregation $\sigma$ (analyzed in Section 4.1) and the outer aggregation $A$ . We consider normalized linear functions for $A$ ,where we are given a set of weights $0 \leq  w \leq  1$ and we rank the target set according to a weighted linear combination of $\sigma$

回顾一下，$F\left( {Q,S}\right)$ 由两个聚合组成：内部聚合 $\sigma$（在 4.1 节中分析）和外部聚合 $A$ 。我们考虑 $A$ 的归一化线性函数，其中给定一组权重 $0 \leq  w \leq  1$ ，我们根据 $\sigma$ 的加权线性组合对目标集合进行排序

scores.

得分。

$$
F\left( {Q,S}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{j = 1}}^{m}{w}_{j}\sigma \left( {\widehat{\mathbf{s}}\left( {{q}_{j},S}\right) }\right) 
$$

With this instantiation of the vector set search problem, we will proceed in Theorem 4.2 to identify a choice of the number of hash tables $L$ that allows us to provide a probabilistic guarantee that the algorithm's query operation succeeds. We will then use this parameter selection to bound the runtime of the query operation in Theorem 4.3.

通过向量集搜索问题的这种实例化，我们将在定理 4.2 中确定哈希表数量 $L$ 的一个选择，该选择能让我们为算法的查询操作成功提供概率保证。然后，我们将使用这个参数选择在定理 4.3 中对查询操作的运行时间进行界定。

Theorem 4.2. Let ${S}^{ \star  }$ be the set with the maximum $F\left( {Q,S}\right)$ and let ${S}_{i}$ be any other set. Let ${B}^{ \star  }$ and ${B}_{i}$ be the following sums (which are lower and upper bounds for $F\left( {Q,{S}^{ \star  }}\right)$ and $F\left( {Q,{S}_{i}}\right)$ , respectively)

定理 4.2。设 ${S}^{ \star  }$ 是具有最大 $F\left( {Q,S}\right)$ 的集合，设 ${S}_{i}$ 是任何其他集合。设 ${B}^{ \star  }$ 和 ${B}_{i}$ 是以下和（分别是 $F\left( {Q,{S}^{ \star  }}\right)$ 和 $F\left( {Q,{S}_{i}}\right)$ 的下界和上界）

$$
{B}^{ \star  } = \frac{\beta }{{m}_{q}}\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}{s}_{\max }\left( {{q}_{j},{S}^{ \star  }}\right) \;{B}_{i} = \frac{\alpha }{{m}_{q}}\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}{s}_{\max }\left( {{q}_{j},{S}_{i}}\right) 
$$

Here, ${s}_{\max }\left( {q,S}\right)$ is the maximum similarity between a query vector $q$ and any element of the target set $S$ . Let ${B}^{\prime }$ be the maximum value of ${B}_{i}$ over any set ${S}_{i} \neq  S$ . Let $\Delta$ be the following value (proportional to the difference between the lower and upper bounds)

这里，${s}_{\max }\left( {q,S}\right)$ 是查询向量 $q$ 与目标集合 $S$ 中任何元素之间的最大相似度。设 ${B}^{\prime }$ 是 ${B}_{i}$ 在任何集合 ${S}_{i} \neq  S$ 上的最大值。设 $\Delta$ 是以下值（与下界和上界之间的差值成比例）

$$
\Delta  = \left( {{B}^{ \star  } - {B}^{\prime }}\right) /3
$$

If $\Delta  > 0$ ,a DESSERT structure with the following value ${}^{2}$ of $L$ solves the search problem from Definition 1.1 with probability $1 - \delta$ .

如果 $\Delta  > 0$ ，一个具有 $L$ 的以下值 ${}^{2}$ 的 DESSERT 结构以概率 $1 - \delta$ 解决定义 1.1 中的搜索问题。

$$
L = O\left( {\log \left( \frac{N{m}_{q}m}{\delta }\right) }\right) 
$$

---

<!-- Footnote -->

${}^{2}L$ additionally depends on the data-dependent parameter $\Delta$ ,which we elide in the asymptotic bound; see the proof in the appendix for the full expression for $L$ .

${}^{2}L$ 还依赖于与数据相关的参数 $\Delta$ ，我们在渐近界中省略了它；完整的 $L$ 表达式见附录中的证明。

<!-- Footnote -->

---

### 4.3 Runtime Analysis

### 4.3 运行时间分析

Theorem 4.3. Suppose that each hash function call runs in time $O\left( d\right)$ and that $\left| {\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h}}\right|  < T\forall i,t,h$ for some positive threshold $T$ ,which we treat as a data-dependent constant in our analysis. Then, using the assumptions and value of $L$ from Theorem 4.2,Algorithm 2 solves the Vector Set Search Problem in query time

定理 4.3。假设每个哈希函数调用的运行时间为 $O\left( d\right)$ ，并且对于某个正阈值 $T$ 有 $\left| {\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h}}\right|  < T\forall i,t,h$ ，在我们的分析中，我们将其视为与数据相关的常数。然后，利用定理 4.2 中的假设和 $L$ 的值，算法 2 在查询时间内解决向量集搜索问题

$$
O\left( {{m}_{q}\log \left( {N{m}_{q}m/\delta }\right) d + {m}_{q}N\log \left( {N{m}_{q}m/\delta }\right) }\right) 
$$

This bound is an improvement over a brute force search of $O\left( {{m}_{q}{mNd}}\right)$ when $m$ or $d$ is large. The above theorem relies upon the choice of $L$ that we derived in Theorem 4.2.

当$m$或$d$很大时，这个边界比暴力搜索$O\left( {{m}_{q}{mNd}}\right)$有所改进。上述定理依赖于我们在定理4.2中推导得出的$L$的选择。

## 5 Implementation Details

## 5 实现细节

Filtering: We find that for large $N$ it is useful to have an initial lossy filtering step that can cheaply reduce the total number of sets we consider with a low false-negative rate. We use an inverted index on the documents for this filtering step.

过滤：我们发现，对于较大的$N$，进行一个初始的有损过滤步骤是有用的，该步骤可以以较低的假阴性率廉价地减少我们考虑的集合总数。我们在文档上使用倒排索引进行此过滤步骤。

To build the inverted index,we first perform $k$ -means clustering on a representative sample of individual item vectors at the start of indexing. The inverted index we will build is a map from centroid ids to document ids. As we add each set ${S}_{i}$ to $\mathcal{D}$ in Algorithm 1,we also add it into the inverted index: we find the closest centroid to each vector $x \in  {S}_{i}$ ,and then we add the document id $i$ to all of the buckets in the inverted index corresponding to those centroids.

为了构建倒排索引，我们在索引开始时首先对单个项目向量的代表性样本执行$k$ -均值聚类。我们要构建的倒排索引是一个从质心ID到文档ID的映射。当我们在算法1中将每个集合${S}_{i}$添加到$\mathcal{D}$时，我们也将其添加到倒排索引中：我们找到每个向量$x \in  {S}_{i}$最近的质心，然后将文档ID $i$添加到倒排索引中与这些质心对应的所有桶中。

This method is similar to PLAID, the recent optimized ColBERT implementation [37], but our query process is much simpler. During querying, we query the inverted index buckets corresponding to the closest filter_probe centroids to each query vector. We aggregate the buckets to get a count for each document id,and then only rank the filter_k documents with DESSERT that have the highest count.

这种方法与PLAID（最近优化的ColBERT实现 [37]）类似，但我们的查询过程要简单得多。在查询期间，我们查询与每个查询向量最近的filter_probe质心对应的倒排索引桶。我们聚合这些桶以获得每个文档ID的计数，然后仅使用DESSERT对计数最高的filter_k个文档进行排序。

Space Optimized Sketches: DESSERT has two features that constrain the underlying hash table implementation: (1) every document is represented by a hash table, so the tables must be low memory, and (2) each query performs many table lookups, so the lookup operation must be fast. If (1) is not met, then we cannot fit the index into memory. If (2) is not met, then the similarity approximation for the inner aggregation step will be far too slow. Initially, we tried a naive implementation of the table, backed by a std::vector, std::map, or std::unordered_map. In each case, the resulting structure did not meet our criteria, so we developed TinyTable, a compact hash table that optimizes memory usage while preserving fast access times. TinyTables sacrifice $O\left( 1\right)$ update-access (which DESSERT does not require) for a considerable improvement to (1) and (2).

空间优化草图：DESSERT有两个限制底层哈希表实现的特性：（1）每个文档由一个哈希表表示，因此这些表必须占用低内存；（2）每个查询要执行多次表查找，因此查找操作必须快速。如果不满足（1），那么我们无法将索引放入内存。如果不满足（2），那么内部聚合步骤的相似度近似将太慢。最初，我们尝试了由std::vector、std::map或std::unordered_map支持的朴素表实现。在每种情况下，得到的结构都不符合我们的标准，因此我们开发了TinyTable，这是一种紧凑的哈希表，它在保持快速访问时间的同时优化了内存使用。TinyTable牺牲了$O\left( 1\right)$更新访问（DESSERT不需要此功能），以显著改善（1）和（2）。

A TinyTable replaces the universal hash table in Algorithm 1, so it must provide a way to map pairs of (hash value,hash table id) to lists of vector ids. At a high level,a TinyTable is composed of $L$ inverted indices from LSH values to vector ids. Bucket $b$ of table $i$ consists of vectors ${x}_{j}$ such that ${h}_{i}\left( {x}_{j}\right)  = b$ . During a query,we simply need to go to the $L$ buckets that correspond to the query vector’s $L$ lsh values to find the ids of ${S}_{i}$ ’s colliding vectors. This design solves (1),the fast lookup requirement, because we can immediately go to the relevant bucket once we have a query's hash value. However, there is a large overhead in storing a resizable vector in every bucket. Even an empty bucket will use $3 * 8 = {24}$ bytes. This adds up: let $r$ be the hash range of the LSH functions (the number of buckets in the inverted index for each of the $L$ tables). If $N = {1M},L = {64}$ ,and $r = {128}$ , we will use $N \cdot  L \cdot  r \cdot  {24} = {196}$ gigabytes even when all of the buckets are empty.

TinyTable取代了算法1中的通用哈希表，因此它必须提供一种将（哈希值，哈希表ID）对映射到向量ID列表的方法。从高层次来看，TinyTable由$L$个从LSH值到向量ID的倒排索引组成。表$i$的桶$b$由满足${h}_{i}\left( {x}_{j}\right)  = b$的向量${x}_{j}$组成。在查询期间，我们只需访问与查询向量的$L$个LSH值对应的$L$个桶，即可找到与${S}_{i}$发生冲突的向量的ID。这种设计解决了（1）快速查找的要求，因为一旦我们有了查询的哈希值，就可以立即访问相关的桶。然而，在每个桶中存储一个可调整大小的向量会有很大的开销。即使是空桶也会使用$3 * 8 = {24}$字节。这会累积起来：设$r$为LSH函数的哈希范围（$L$个表中每个表的倒排索引中的桶数）。如果$N = {1M},L = {64}$且$r = {128}$，即使所有桶都是空的，我们也将使用$N \cdot  L \cdot  r \cdot  {24} = {196}$吉字节。

Thus,a TinyTable has more optimizations that make it space efficient. Each of the $L$ hash table repetitions in a TinyTable are conceptually split into two parts: a list of offsets and a list of vector ids. The vector ids are the concatenated contents of the buckets of the table with no space in between (thus,they are always some permutation of 0 through $m - 1$ ). The offset list describes where one bucket ends and the next begins: the $i$ th entry in the offset list is the (inclusive) index of the start of the $i$ th hash bucket within the vector id list,and the $i + 1$ th entry is the (exclusive) end of the ith hash bucket (if a bucket is empty,indices $\left\lbrack  i\right\rbrack   =$ indices $\left\lbrack  {i + 1}\right\rbrack$ ). To save more bytes,we can further concatenate the $L$ offset lists together and the $L$ vector id lists together,since their lengths are always $r$ and ${m}_{i}$ respectively. Finally,we note that if $m \leq  {256}$ ,we can store all of the the offsets and ids can be safely be stored as single byte integers. Using the same hypothetical numbers as before, a filled TinyTable with $m = {100}$ will take up just $N\left( {{24} + L\left( {m + r + 1}\right) }\right)  = {14.7}\mathrm{{GB}}$ .

因此，TinyTable（微型表）有更多优化措施，使其在空间利用上更高效。TinyTable 中的 $L$ 个哈希表副本在概念上被分为两部分：一个偏移量列表和一个向量 ID 列表。向量 ID 是表中各个桶的内容拼接而成，中间没有间隔（因此，它们始终是从 0 到 $m - 1$ 的某种排列）。偏移量列表描述了一个桶的结束位置和下一个桶的开始位置：偏移量列表中的第 $i$ 项是向量 ID 列表中第 $i$ 个哈希桶起始位置的（包含）索引，第 $i + 1$ 项是第 i 个哈希桶的（不包含）结束位置（如果一个桶为空，则索引为 $\left\lbrack  i\right\rbrack   =$ 到 $\left\lbrack  {i + 1}\right\rbrack$）。为了节省更多字节，我们可以进一步将 $L$ 个偏移量列表和 $L$ 个向量 ID 列表分别拼接在一起，因为它们的长度始终分别为 $r$ 和 ${m}_{i}$。最后，我们注意到，如果 $m \leq  {256}$，我们可以安全地将所有偏移量和 ID 存储为单字节整数。使用与之前相同的假设数字，一个填充了 $m = {100}$ 的 TinyTable 仅占用 $N\left( {{24} + L\left( {m + r + 1}\right) }\right)  = {14.7}\mathrm{{GB}}$ 的空间。

The Concatenation Trick: In our theory,we assumed LSH functions such that $p\left( {h\left( x\right)  = h\left( y\right) }\right)  =$ $\operatorname{sim}\left( {x,y}\right)$ . However,for practical problems such functions lead to overfull buckets; for example, GLOVE has an average vector cosine similarity of around 0.3 , which would mean each bucket in the LSH table would contain a third of the set. The standard trick to get around this problem is to concatenate $C$ hashes for each of the $L$ tables together such that $p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}{\left( x,y\right) }^{C}$ . Rewriting, we have that

拼接技巧：在我们的理论中，我们假设局部敏感哈希（LSH）函数满足 $p\left( {h\left( x\right)  = h\left( y\right) }\right)  =$ $\operatorname{sim}\left( {x,y}\right)$。然而，对于实际问题，这样的函数会导致桶过度填充；例如，GLOVE（全局向量词表示）的向量平均余弦相似度约为 0.3，这意味着 LSH 表中的每个桶将包含集合的三分之一。解决这个问题的标准技巧是将 $L$ 个表中每个表的 $C$ 个哈希值拼接在一起，使得 $p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}{\left( x,y\right) }^{C}$。重写后，我们得到

$$
\operatorname{sim}\left( {x,y}\right)  = \exp \left( \frac{\ln \left\lbrack  {p\left( {h\left( x\right)  = h\left( y\right) }\right) }\right\rbrack  }{C}\right)  \tag{1}
$$

During a query,we count the number of collisions across the $L$ tables and divide by $L$ to get $\widehat{p}\left( {h\left( x\right)  = h\left( y\right) }\right)$ on line 11 of Algorithm 2. We now additionally pass $\operatorname{count}/L$ into Equation 1 to get an accurate similarity estimate to pass into $\sigma$ on line 12 . Furthermore,evaluating Equation 1 for every collision probability estimate is slow in practice. There are only $L + 1$ possible values for the count $/L$ ,so we precompute the mapping in a lookup table.

在查询过程中，我们统计 $L$ 个表中的碰撞次数，并将其除以 $L$，以在算法 2 的第 11 行得到 $\widehat{p}\left( {h\left( x\right)  = h\left( y\right) }\right)$。现在，我们额外将 $\operatorname{count}/L$ 代入方程 1，以获得准确的相似度估计值，并在第 12 行将其代入 $\sigma$。此外，在实践中，为每个碰撞概率估计值计算方程 1 的速度很慢。计数 $/L$ 只有 $L + 1$ 种可能的值，因此我们在查找表中预先计算映射关系。

## 6 Experiments

## 6 实验

Datasets: We tested DESSERT on both synthetic data and real-world problems. We first examined a series of synthetic datasets to measure DESSERT's speedup over a reasonable CPU brute force algorithm (using the PyTorch library [35] for matrix multiplications). For this experiment, we leave out the prefiltering optimization described in Section 5 to better show how DESSERT performs on its own. Following the authors of [25], our synthetic dataset consists of random groups of Glove [36] vectors; we vary the set size $m$ and keep the total number of sets $N = {1000}$ .

数据集：我们在合成数据和实际问题上对 DESSERT 进行了测试。我们首先研究了一系列合成数据集，以衡量 DESSERT 相对于合理的 CPU 暴力算法（使用 PyTorch 库 [35] 进行矩阵乘法）的加速效果。在这个实验中，我们省略了第 5 节中描述的预过滤优化，以便更好地展示 DESSERT 自身的性能。遵循文献 [25] 的作者的做法，我们的合成数据集由随机的 Glove（全局向量词表示）[36] 向量组组成；我们改变集合大小 $m$，并保持集合总数为 $N = {1000}$。

We next experimented with the MS MARCO passage ranking dataset (Creative Commons License) [32], $N \approx  {8.8M}$ . The task for MS MARCO is to retrieve passages from the corpus relevant to a query. We used ColBERT to map the words from each passage and query to sets of embedding vectors suitable for DESSERT [37]. Following [37], we use the development set for our experiments, which contains 6980 queries.

接下来，我们对 MS MARCO 段落排名数据集（知识共享许可协议）[32] 进行了实验，$N \approx  {8.8M}$。MS MARCO 的任务是从语料库中检索与查询相关的段落。我们使用 ColBERT 将每个段落和查询中的单词映射到适合 DESSERT 的嵌入向量集 [37]。遵循文献 [37] 的做法，我们使用开发集进行实验，该开发集包含 6980 个查询。

Finally, we computed the full resource-accuracy tradeoff for ten of the LoTTE out-of-domain benchmark datasets, introduced by ColBERTv2 [38]. We excluded the pooled dataset, which is simply the individual datasets merged.

最后，我们计算了 ColBERTv2 [38] 引入的十个 LoTTE 域外基准数据集的完整资源 - 准确性权衡。我们排除了合并数据集，它只是各个数据集的简单合并。

Experiment Setup: We ran our experiments on an Intel(R) Xeon(R) CPU E5-2680 v3 machine with 252 GB of RAM. We restricted all experiments to 4 cores ( 8 threads). We ran each experiment with the chosen hyperparameters and reported overall average recall and average query latency. For all experiments we used the average of max similarities scoring function.

实验设置：我们在一台配备252GB内存的英特尔（Intel）至强（Xeon）CPU E5 - 2680 v3机器上运行实验。我们将所有实验限制在4个核心（8个线程）上进行。我们使用选定的超参数运行每个实验，并报告总体平均召回率和平均查询延迟。对于所有实验，我们使用最大相似度平均评分函数。

### 6.1 Synthetic Data

### 6.1 合成数据

The goal of our synthetic data experiment was to examine DESSERT's speedup over brute force vector set scoring. Thus, we generated synthetic data where both DESSERT and the brute force implementation achieved perfect recall so we could compare the two methods solely on query time.

我们的合成数据实验的目标是检验DESSERT相对于暴力向量集评分的加速效果。因此，我们生成了合成数据，在这些数据上DESSERT和暴力实现方法都能实现完美召回，这样我们就可以仅在查询时间上对这两种方法进行比较。

<!-- Media -->

<!-- figureText: Query Time v. Set Size on Synthetic Glove Data DESSERT — Pytorch Brute Force Combined -->

<img src="https://cdn.noedgeai.com/0195a865-affe-7ef5-9800-93ec0736f0e2_7.jpg?x=956&y=1583&w=431&h=342&r=0"/>

Figure 2: Query time for DESSERT vs. brute force on 1000 random sets of $m$ glove vectors with the $y$ -axis as a log scale. Lower is better.

图2：DESSERT与暴力方法在1000个随机的$m$ glove向量集上的查询时间，$y$轴为对数刻度。数值越低越好。

<!-- Media -->

The two optimized brute force implementations we tried both used PyTorch, and differed only in whether they computed the score between the query set and each document set individually ("Individual") or between the query set and all document sets at once using PyTorch's highly performant reduce and reshape operations ("Combined").

我们尝试的两种优化暴力实现方法都使用了PyTorch，它们的区别仅在于是否分别计算查询集与每个文档集之间的分数（“单独计算”），还是使用PyTorch高性能的归约和重塑操作一次性计算查询集与所有文档集之间的分数（“组合计算”）。

In each synthetic experiment, we inserted 1000 documents of size $m$ for $m \in$ $\left\lbrack  {2,4,8,{16},\ldots ,{1024}}\right\rbrack$ into DESSERT and the brute force index. The queries in each experiment were simply the documents with added noise. The DESSERT hyperparameters we chose were $L = 8$ and $C = {\log }_{2}\left( m\right)  + 1$ . The results of our experiment,which show the relative speedup of using DESSERT at different values of $m$ ,are in Figure 2. We observe that DESSERT achieves a consistent 10-50x speedup over the optimized Pytorch brute force method and that the speedup increases with larger $m$ (we could not run experiments with even larger $m$ because the PyTorch runs did not finish within the time allotted).

在每个合成实验中，我们将1000个大小为$m$的文档插入到DESSERT和暴力索引中。每个实验中的查询只是添加了噪声的文档。我们选择的DESSERT超参数是$L = 8$和$C = {\log }_{2}\left( m\right)  + 1$。实验结果（展示了在不同$m$值下使用DESSERT的相对加速效果）如图2所示。我们观察到，与优化后的PyTorch暴力方法相比，DESSERT始终能实现10 - 50倍的加速，并且加速效果随着$m$的增大而提高（我们无法对更大的$m$进行实验，因为PyTorch运行在规定时间内无法完成）。

### 6.2 Passage Retrieval

### 6.2 段落检索

Passage retrieval refers to the task of identifying and returning the most relevant passages from a large corpus of documents in response to a search query. In these experiments, we compared DESSERT to PLAID, ColBERT's heavily-optimzed state-of-the-art late interaction search algorithm, on the MS MARCO and LoTTE passage retrieval tasks.

段落检索是指根据搜索查询从大量文档语料库中识别并返回最相关段落的任务。在这些实验中，我们在MS MARCO和LoTTE段落检索任务上，将DESSERT与PLAID（ColBERT经过高度优化的最先进的后期交互搜索算法）进行了比较。

We found that the best ColBERT hyperparameters were the same as reported in the PLAID paper, and we successfully replicated their results. Although PLAID offers a way to trade off time for accuracy, this tradeoff only increases accuracy at the sake of time, and even then only by a fraction of a percent. Thus, our results represent points on the recall vs time Pareto frontier that PLAID cannot reach.

我们发现，最佳的ColBERT超参数与PLAID论文中报告的相同，并且我们成功复现了他们的结果。虽然PLAID提供了一种在时间和准确性之间进行权衡的方法，但这种权衡只是以牺牲时间为代价来提高准确性，而且即使如此，提高的幅度也只有百分之几。因此，我们的结果代表了召回率与时间的帕累托前沿上PLAID无法达到的点。

MS MARCO Results For MS MARCO, we performed a grid search over DESSERT parameters $C = \{ 4,5,6,7\} ,L = \{ {16},{32},{64}\}$ ,filter_probe $= \{ 1,2,4,8\}$ ,and filter_ $k =$ $\{ {1000},{2048},{4096},{8192},{16384}\}$ . We reran the best configurations to obtain the results in Table 2. We report two types of results: methods tuned to return $k = {10}$ results and methods tuned to return $k = {1000}$ results. For each,we report DESSERT results from a low latency and a high latency part of the Pareto frontier. For $k = {1000}$ we use the standard $R@{1000}$ metric,the average recall of the top 1 passage in the first 1000 returned passages. This metric is meaningful because retrieval pipelines frequently rerank candidates after an initial retrieval stage. For $k = {10}$ we use the standard ${MRR}@{10}$ metric,the average mean reciprocal rank of the top 1 passage in the first 10 returned passages. Overall, DESSERT is 2-5x faster than PLAID with only a few percent loss in recall.

MS MARCO结果 对于MS MARCO，我们对DESSERT参数$C = \{ 4,5,6,7\} ,L = \{ {16},{32},{64}\}$、filter_probe $= \{ 1,2,4,8\}$和filter_ $k =$ $\{ {1000},{2048},{4096},{8192},{16384}\}$进行了网格搜索。我们重新运行最佳配置以获得表2中的结果。我们报告了两种类型的结果：调整为返回$k = {10}$个结果的方法和调整为返回$k = {1000}$个结果的方法。对于每种情况，我们报告了帕累托前沿上低延迟和高延迟部分的DESSERT结果。对于$k = {1000}$，我们使用标准的$R@{1000}$指标，即前1000个返回段落中排名第一的段落的平均召回率。这个指标很有意义，因为检索管道通常会在初始检索阶段后对候选段落进行重新排序。对于$k = {10}$，我们使用标准的${MRR}@{10}$指标，即前10个返回段落中排名第一的段落的平均倒数排名。总体而言，DESSERT比PLAID快2 - 5倍，而召回率仅损失几个百分点。

<!-- Media -->

<table><tr><td>Method</td><td>Latency (ms)</td><td>${MRR}@{10}$</td><td>Method</td><td>Latency (ms)</td><td>$R@{1000}$</td></tr><tr><td>DESSERT</td><td>9.5</td><td>${35.7} \pm  {1.14}$</td><td>DESSERT</td><td>22.7</td><td>${95.1} \pm  {0.49}$</td></tr><tr><td>DESSERT</td><td>15.5</td><td>37.2 ± 1.14</td><td>DESSERT</td><td>32.3</td><td>${96.0} \pm  {0.45}$</td></tr><tr><td>PLAID</td><td>45.1</td><td>${39.2} \pm  {1.15}$</td><td>PLAID</td><td>100</td><td>${97.5} \pm  {0.36}$</td></tr></table>

<table><tbody><tr><td>方法</td><td>延迟（毫秒）</td><td>${MRR}@{10}$</td><td>方法</td><td>延迟（毫秒）</td><td>$R@{1000}$</td></tr><tr><td>甜点（DESSERT）</td><td>9.5</td><td>${35.7} \pm  {1.14}$</td><td>甜点（DESSERT）</td><td>22.7</td><td>${95.1} \pm  {0.49}$</td></tr><tr><td>甜点（DESSERT）</td><td>15.5</td><td>37.2 ± 1.14</td><td>甜点（DESSERT）</td><td>32.3</td><td>${96.0} \pm  {0.45}$</td></tr><tr><td>格子图案（PLAID）</td><td>45.1</td><td>${39.2} \pm  {1.15}$</td><td>格子图案（PLAID）</td><td>100</td><td>${97.5} \pm  {0.36}$</td></tr></tbody></table>

Table 2: MS MARCO passage retrieval,with methods optimized for $\mathrm{k} = {10}$ (left) and $\mathrm{k} = {1000}$ (right). Intervals denote 95% confidence intervals for average latency and recall.

表2：MS MARCO段落检索，方法分别针对$\mathrm{k} = {10}$（左）和$\mathrm{k} = {1000}$（右）进行了优化。区间表示平均延迟和召回率的95%置信区间。

<!-- Media -->

LoTTE Results For LoTTE,we performed a grid search over $C = \{ 4,6,8\} ,L = \{ {32},{64},{128}\}$ , filter_probe $= \{ 1,2,4\}$ ,and filter_ $k = \{ {1000},{2048},{4096},{8192}\}$ . In Figure 3,we plot the full Pareto tradeoff for DESSERT on the 10 LoTTE datasets (each of the 5 categories has a "forum" and "search" split) over these hyperparameters, as well as the single lowest-latency point achievable by PLAID. For all test datasets, DESSERT provides a Pareto frontier that allows a tradeoff between recall and latency. For both Lifestyle test splits, both Technology test splits, and the Recreation and Science test-search splits,DESSERT achieves a 2-5x speedup with minimal loss in accuracy. On Technology, DESSERT even exceeds the accuracy of PLAID at half of PLAID's latency.

LoTTE结果 对于LoTTE，我们对$C = \{ 4,6,8\} ,L = \{ {32},{64},{128}\}$、filter_probe $= \{ 1,2,4\}$和filter_ $k = \{ {1000},{2048},{4096},{8192}\}$进行了网格搜索。在图3中，我们绘制了DESSERT在10个LoTTE数据集（5个类别中每个类别都有“论坛”和“搜索”两个子集）上关于这些超参数的完整帕累托权衡图，以及PLAID可达到的单个最低延迟点。对于所有测试数据集，DESSERT提供了一个帕累托前沿，允许在召回率和延迟之间进行权衡。对于生活方式的两个测试子集、技术的两个测试子集以及娱乐和科学的测试 - 搜索子集，DESSERT实现了2 - 5倍的加速，同时精度损失最小。在技术领域，DESSERT甚至在PLAID一半的延迟下超过了PLAID的精度。

## 7 Discussion

## 7 讨论

We observe a substantial speedup when we integrate DESSERT into ColBERT, even when compared against the highly-optimized PLAID implementation. While the use of our algorithm incurs a slight recall penalty - as is the case with most algorithms that use randomization to achieve acceleration - Table 2 and Figure 3 shows that we are Pareto-optimal when compared with baseline approaches.

我们发现，即使与高度优化的PLAID实现相比，将DESSERT集成到ColBERT中也能实现显著的加速。虽然使用我们的算法会导致轻微的召回率损失（大多数使用随机化来实现加速的算法都是如此），但表2和图3表明，与基线方法相比，我们的算法是帕累托最优的。

We are not aware of any algorithm other than DESSERT that is capable of latencies in this range for set-to-set similarity search. While systems such as PLAID are tunable, we were unable to get them to operate in this range. For this reason, DESSERT is likely the only set-to-set similarity search algorithm that can be run in real-time production environments with strict latency constraints. We also ran a single-vector search baseline using ScaNN, the leading approximate kNN index [14]. ScaNN yielded 0.77 Recall@1000, substantially below the state of the art. This result reinforces our discussion in Section 1.2 on why single-vector search is insufficient.

据我们所知，除了DESSERT之外，没有其他算法能够在集合到集合的相似性搜索中达到这样的延迟范围。虽然像PLAID这样的系统是可调节的，但我们无法让它们在这个范围内运行。因此，DESSERT可能是唯一一种可以在具有严格延迟约束的实时生产环境中运行的集合到集合的相似性搜索算法。我们还使用领先的近似k近邻索引ScaNN [14]运行了单向量搜索基线。ScaNN在1000召回率下为0.77，远低于当前的先进水平。这一结果强化了我们在1.2节中关于单向量搜索为何不足的讨论。

<!-- Media -->

<!-- figureText: lifestyle writing Mean Latency (ms) Mean Latency (ms) -->

<img src="https://cdn.noedgeai.com/0195a865-affe-7ef5-9800-93ec0736f0e2_9.jpg?x=427&y=251&w=984&h=664&r=0"/>

Figure 3: Full Pareto frontier of DESSERT on the LoTTE datasets. The PLAID baseline shows the lowest-latency result attainable by PLAID (with a FAISS-IVF base index and centroid pre-filtering).

图3：DESSERT在LoTTE数据集上的完整帕累托前沿。PLAID基线显示了PLAID可达到的最低延迟结果（使用FAISS - IVF基础索引和质心预过滤）。

<!-- Media -->

Broader Impacts and Limitations: Ranking and retrieval are important steps in language modeling applications, some of which have recently come under increased scrutiny. However, our algorithm is unlikely to have negative broader effects, as it mainly enables faster, more cost-effective search over larger vector collections and does not contribute to the problematic capabilities of the aforementioned language models. Due to computational limitations, we conduct our experiments on a relatively small set of benchmarks; a larger-scale evaluation would strengthen our argument. Finally, we assume sufficiently high relevance scores and large gaps in our theoretical analysis to identify the correct results. These hardness assumptions are standard for LSH.

更广泛的影响和局限性：排序和检索是语言建模应用中的重要步骤，其中一些最近受到了更多的审查。然而，我们的算法不太可能产生更广泛的负面影响，因为它主要是实现对更大向量集合的更快、更具成本效益的搜索，并且不会导致上述语言模型的问题能力。由于计算限制，我们在相对较小的一组基准上进行实验；更大规模的评估将加强我们的论点。最后，我们在理论分析中假设相关性得分足够高且差距足够大，以识别正确的结果。这些难度假设对于局部敏感哈希（LSH）来说是标准的。

## 8 Conclusion

## 8 结论

In this paper, we consider the problem of vector set search with vector set queries, a task understudied in the existing literature. We present a formal definition of the problem and provide a motivating application in semantic search, where a more efficient algorithm would provide considerable immediate impact in accelerating late interaction search methods. To address the large latencies inherent in existing vector search methods, we propose a novel randomized algorithm called DESSERT that achieves significant speedups over baseline techniques. We also analyze DESSERT theoretically and, under natural assumptions, prove rigorous guarantees on the algorithm's failure probability and runtime. Finally,we provide an open-source and highly performant $\mathrm{C} +  +$ implementation of our proposed DESSERT algorithm that achieves 2-5x speedup over ColBERT-PLAID on the MS MARCO and LoTTE retrieval benchmarks. We also note that a general-purpose algorithmic framework for vector set search with vector set queries could have impact in a number of other applications, such as image similarity search [42], market basket analysis [22], and graph neural networks [43], where it might be more natural to model entities via sets of vectors as opposed to restricting representations to a single embedding. We believe that DESSERT could provide a viable algorithmic engine for enabling such applications and we hope to study these potential use cases in the future.

在本文中，我们考虑了使用向量集查询进行向量集搜索的问题，这是现有文献中研究不足的一个任务。我们给出了该问题的正式定义，并在语义搜索中提供了一个有启发性的应用，在语义搜索中，更高效的算法将对加速后期交互搜索方法产生相当大的直接影响。为了解决现有向量搜索方法固有的大延迟问题，我们提出了一种名为DESSERT的新型随机算法，该算法比基线技术实现了显著的加速。我们还对DESSERT进行了理论分析，并在自然假设下，对算法的失败概率和运行时间给出了严格的保证。最后，我们提供了我们提出的DESSERT算法的开源且高性能的$\mathrm{C} +  +$实现，该实现在MS MARCO和LoTTE检索基准上比ColBERT - PLAID实现了2 - 5倍的加速。我们还注意到，用于使用向量集查询进行向量集搜索的通用算法框架可能会在许多其他应用中产生影响，如图像相似性搜索[42]、购物篮分析[22]和图神经网络[43]，在这些应用中，通过向量集对实体进行建模可能比将表示限制为单个嵌入更自然。我们相信DESSERT可以为实现此类应用提供一个可行的算法引擎，我们希望在未来研究这些潜在的用例。

## 9 Acknowledgments

## 9 致谢

This work was completed while the authors were working at ThirdAI. We do not have any external funding sources to acknowledge.

这项工作是作者在ThirdAI工作期间完成的。我们没有需要致谢的外部资金来源。

## References

## 参考文献

[1] Alexandr Andoni and Piotr Indyk. Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. Communications of the ACM, 51(1):117-122, 2008.

[1] 亚历山大·安多尼（Alexandr Andoni）和彼得·因迪克（Piotr Indyk）。高维近似最近邻的近似最优哈希算法。《美国计算机协会通讯》（Communications of the ACM），51(1):117 - 122，2008年。

[2] Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, and Ludwig Schmidt. Practical and optimal lsh for angular distance. Advances in neural information processing systems, 28, 2015.

[2] 亚历山大·安多尼（Alexandr Andoni）、彼得·因迪克（Piotr Indyk）、蒂杰斯·拉尔霍文（Thijs Laarhoven）、伊利亚·拉曾施泰因（Ilya Razenshteyn）和路德维希·施密特（Ludwig Schmidt）。用于角度距离的实用且最优的局部敏感哈希（LSH）。《神经信息处理系统进展》（Advances in neural information processing systems），28，2015年。

[3] Alexandr Andoni, Piotr Indyk, Huy L Nguyen, and Ilya Razenshteyn. Beyond locality-sensitive hashing. In Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms, pages 1018-1028. SIAM, 2014.

[3] 亚历山大·安多尼（Alexandr Andoni）、彼得·因迪克（Piotr Indyk）、胡伊·L·阮（Huy L Nguyen）和伊利亚·拉曾施泰因（Ilya Razenshteyn）。超越局部敏感哈希。《第二十五届年度ACM - SIAM离散算法研讨会论文集》（Proceedings of the twenty - fifth annual ACM - SIAM symposium on Discrete algorithms），第1018 - 1028页。工业与应用数学学会（SIAM），2014年。

[4] Alexandr Andoni and Ilya Razenshteyn. Optimal data-dependent hashing for approximate near neighbors. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing, pages 793-801, 2015.

[4] 亚历山大·安多尼（Alexandr Andoni）和伊利亚·拉曾施泰因（Ilya Razenshteyn）。用于近似近邻的最优数据相关哈希。《第四十七届年度ACM计算理论研讨会论文集》（Proceedings of the forty - seventh annual ACM symposium on Theory of computing），第793 - 801页，2015年。

[5] Ioannis Arapakis, Souneil Park, and Martin Pielot. Impact of response latency on user behaviour in mobile web search. In Proceedings of the 2021 Conference on Human Information Interaction and Retrieval, pages 279-283, 2021.

[5] 约安尼斯·阿拉帕基斯（Ioannis Arapakis）、苏尼尔·朴（Souneil Park）和马丁·皮洛特（Martin Pielot）。响应延迟对移动网页搜索中用户行为的影响。《2021年人类信息交互与检索会议论文集》（Proceedings of the 2021 Conference on Human Information Interaction and Retrieval），第279 - 283页，2021年。

[6] A. Broder. On the resemblance and containment of documents. In Proceedings of the Compression and Complexity of Sequences 1997, SEQUENCES '97, page 21, USA, 1997. IEEE Computer Society.

[6] A. 布罗德（A. Broder）。关于文档的相似度和包含度。《1997年序列压缩与复杂度会议论文集》（Proceedings of the Compression and Complexity of Sequences 1997, SEQUENCES '97），第21页，美国，1997年。电气和电子工程师协会计算机学会（IEEE Computer Society）。

[7] J Lawrence Carter and Mark N Wegman. Universal classes of hash functions. In Proceedings of the ninth annual ACM symposium on Theory of computing, pages 106-112, 1977.

[7] J·劳伦斯·卡特（J Lawrence Carter）和马克·N·韦格曼（Mark N Wegman）。通用哈希函数类。《第九届年度ACM计算理论研讨会论文集》（Proceedings of the ninth annual ACM symposium on Theory of computing），第106 - 112页，1977年。

[8] Moses S. Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, STOC '02, page 380-388, New York, NY, USA, 2002. Association for Computing Machinery.

[8] 摩西·S·查里卡尔（Moses S. Charikar）。基于舍入算法的相似度估计技术。《第三十四届年度ACM计算理论研讨会论文集》（Proceedings of the Thiry - Fourth Annual ACM Symposium on Theory of Computing, STOC '02），第380 - 388页，美国纽约州纽约市，2002年。美国计算机协会（Association for Computing Machinery）。

[9] Benjamin Coleman, Richard Baraniuk, and Anshumali Shrivastava. Sub-linear memory sketches for near neighbor search on streaming data. In International Conference on Machine Learning, pages 2089-2099. PMLR, 2020.

[9] 本杰明·科尔曼（Benjamin Coleman）、理查德·巴拉纽克（Richard Baraniuk）和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。用于流数据近邻搜索的亚线性内存草图。《国际机器学习会议论文集》（International Conference on Machine Learning），第2089 - 2099页。机器学习研究会议录（PMLR），2020年。

[10] Benjamin Coleman and Anshumali Shrivastava. Sub-linear race sketches for approximate kernel density estimation on streaming data. In Proceedings of The Web Conference 2020, pages 1739-1749, 2020.

[10] 本杰明·科尔曼（Benjamin Coleman）和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。用于流数据近似核密度估计的亚线性竞赛草图。《2020年网络会议论文集》（Proceedings of The Web Conference 2020），第1739 - 1749页，2020年。

[11] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S. Mirrokni. Locality-sensitive hashing scheme based on p-stable distributions. In Proceedings of the Twentieth Annual Symposium on Computational Geometry, SCG '04, page 253-262, New York, NY, USA, 2004. Association for Computing Machinery.

[11] 马尤尔·达塔尔（Mayur Datar）、妮可·伊莫利卡（Nicole Immorlica）、彼得·因迪克（Piotr Indyk）和瓦哈布·S·米罗克尼（Vahab S. Mirrokni）。基于p - 稳定分布的局部敏感哈希方案。《第二十届年度计算几何研讨会论文集》（Proceedings of the Twentieth Annual Symposium on Computational Geometry, SCG '04），第253 - 262页，美国纽约州纽约市，2004年。美国计算机协会（Association for Computing Machinery）。

[12] Yihe Dong, Piotr Indyk, Ilya Razenshteyn, and Tal Wagner. Learning space partitions for nearest neighbor search. arXiv preprint arXiv:1901.08544, 2019.

[12] 董一禾（Yihe Dong）、彼得·因迪克（Piotr Indyk）、伊利亚·拉曾施泰因（Ilya Razenshteyn）和塔尔·瓦格纳（Tal Wagner）。学习用于最近邻搜索的空间划分。预印本arXiv:1901.08544，2019年。

[13] Joshua Engels, Benjamin Coleman, and Anshumali Shrivastava. Practical near neighbor search via group testing. Advances in Neural Information Processing Systems, 34:9950-9962, 2021.

[13] 约书亚·恩格尔斯（Joshua Engels）、本杰明·科尔曼（Benjamin Coleman）和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。通过分组测试实现实用的近邻搜索。《神经信息处理系统进展》（Advances in Neural Information Processing Systems），34:9950 - 9962，2021年。

[14] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pages 3887-3896. PMLR, 2020.

[14] 郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿全（Quan Geng）、大卫·辛查（David Simcha）、费利克斯·陈（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。使用各向异性向量量化加速大规模推理。《国际机器学习会议论文集》（International Conference on Machine Learning），第3887 - 3896页。机器学习研究会议录（PMLR），2020年。

[15] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. Embedding-based retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2553-2561, 2020.

[15] 黄瑞婷（Jui - Ting Huang）、阿什什·夏尔马（Ashish Sharma）、孙淑英（Shuying Sun）、夏莉（Li Xia）、张大卫（David Zhang）、菲利普·普罗宁（Philip Pronin）、贾纳尼·帕德马纳班（Janani Padmanabhan）、朱塞佩·奥塔维亚诺（Giuseppe Ottaviano）和杨林军（Linjun Yang）。基于嵌入的Facebook搜索检索。《第26届ACM SIGKDD国际知识发现与数据挖掘会议论文集》（Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining），第2553 - 2561页，2020年。

[16] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pages 2333-2338, 2013.

[16] 黄伯森（Po - Sen Huang）、何晓东（Xiaodong He）、高剑锋（Jianfeng Gao）、邓力（Li Deng）、亚历克斯·阿塞罗（Alex Acero）和拉里·赫克（Larry Heck）。使用点击数据学习用于网页搜索的深度结构化语义模型。《第22届ACM国际信息与知识管理会议论文集》（Proceedings of the 22nd ACM international conference on Information & Knowledge Management），第2333 - 2338页，2013年。

[17] Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: Towards removing the curse of dimensionality. In Proceedings of the Thirtieth Annual ACM Symposium on Theory of Computing, STOC '98, page 604-613, New York, NY, USA, 1998. Association for Computing Machinery.

[17] 皮奥特·因迪克（Piotr Indyk）和拉杰夫·莫特瓦尼（Rajeev Motwani）。近似最近邻：消除维度灾难的探索。收录于第三十届美国计算机协会计算理论年会论文集（Proceedings of the Thirtieth Annual ACM Symposium on Theory of Computing, STOC '98），第604 - 613页，美国纽约，1998年。美国计算机协会。

[18] Masajiro Iwasaki and Daisuke Miyazaki. Optimization of indexing based on k-nearest neighbor graph for proximity search in high-dimensional data. arXiv preprint arXiv:1810.07355, 2018.

[18] 岩崎正次郎（Masajiro Iwasaki）和宫崎大辅（Daisuke Miyazaki）。基于k近邻图的高维数据邻近搜索索引优化。预印本arXiv:1810.07355，2018年。

[19] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128, 2010.

[19] 埃尔韦·热古（Herve Jegou）、马蒂亚斯·杜泽（Matthijs Douze）和科迪莉亚·施密德（Cordelia Schmid）。用于最近邻搜索的乘积量化。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE transactions on pattern analysis and machine intelligence），33(1):117 - 128，2010年。

[20] Jianqiu Ji, Jianmin Li, Shuicheng Yan, Bo Zhang, and Qi Tian. Super-bit locality-sensitive hashing. Advances in neural information processing systems, 25, 2012.

[20] 季建秋、李建民、颜水成、张波和齐天。超位局部敏感哈希。《神经信息处理系统进展》（Advances in neural information processing systems），25，2012年。

[21] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535-547, 2019.

[21] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。基于GPU的十亿级相似度搜索。《电气与电子工程师协会大数据汇刊》（IEEE Transactions on Big Data），7(3):535 - 547，2019年。

[22] Manpreet Kaur and Shivani Kang. Market basket analysis: Identify the changing trends of market data using association rule mining. Procedia computer science, 85:78-85, 2016.

[22] 曼普里特·考尔（Manpreet Kaur）和希瓦尼·康（Shivani Kang）。购物篮分析：使用关联规则挖掘识别市场数据的变化趋势。《计算机科学进展》（Procedia computer science），85:78 - 85，2016年。

[23] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextual-ized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39-48, 2020.

[23] 奥马尔·哈塔布（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。COLBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。收录于第43届美国计算机协会信息检索研究与发展会议论文集（Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval），第39 - 48页，2020年。

[24] Runze Lei, Pinghui Wang, Rundong Li, Peng Jia, Junzhou Zhao, Xiaohong Guan, and Chao Deng. Fast rotation kernel density estimation over data streams. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 892-902, 2021.

[24] 雷润泽、王品辉、李润东、贾鹏、赵俊洲、关晓红和邓超。数据流上的快速旋转核密度估计。收录于第27届美国计算机协会知识发现与数据挖掘会议论文集（Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining），第892 - 902页，2021年。

[25] Michael Leybovich and Oded Shmueli. Efficient approximate search for sets of vectors. arXiv preprint arXiv:2107.06817, 2021.

[25] 迈克尔·莱博维奇（Michael Leybovich）和奥代德·什穆埃利（Oded Shmueli）。向量集的高效近似搜索。预印本arXiv:2107.06817，2021年。

[26] Sen Li, Fuyu Lv, Taiwei Jin, Guli Lin, Keping Yang, Xiaoyi Zeng, Xiao-Ming Wu, and Qianli Ma. Embedding-based product retrieval in taobao search. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 3181-3189, 2021.

[26] 李森、吕富宇、金泰伟、林谷丽、杨克平、曾晓艺、吴晓明和马千里。淘宝搜索中基于嵌入的商品检索。收录于第27届美国计算机协会知识发现与数据挖掘会议论文集（Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining），第3181 - 3189页，2021年。

[27] Chen Luo and Anshumali Shrivastava. Arrays of (locality-sensitive) count estimators (ace) anomaly detection on the edge. In Proceedings of the 2018 World Wide Web Conference, pages 1439-1448, 2018.

[27] 罗晨和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。（局部敏感）计数估计器数组（ACE）边缘异常检测。收录于2018年万维网会议论文集（Proceedings of the 2018 World Wide Web Conference），第1439 - 1448页，2018年。

[28] Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4):824-836, 2018.

[28] 尤·A·马尔科夫（Yu A Malkov）和德米特里·A·亚舒宁（Dmitry A Yashunin）。使用层次可导航小世界图进行高效鲁棒的近似最近邻搜索。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE transactions on pattern analysis and machine intelligence），42(4):824 - 836，2018年。

[29] Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma. Detecting near-duplicates for web crawling. In Proceedings of the 16th international conference on World Wide Web, pages 141-150, 2007.

[29] 古尔米特·辛格·曼库（Gurmeet Singh Manku）、阿尔温德·贾因（Arvind Jain）和阿尼什·达斯·萨尔马（Anish Das Sarma）。网页爬取中的近似重复检测。收录于第16届万维网国际会议论文集（Proceedings of the 16th international conference on World Wide Web），第141 - 150页，2007年。

[30] Christopher D Manning, Prabhakar Raghavan, and Hinrich Schütze. Introduction to information retrieval. Cambridge university press, 2008.

[30] 克里斯托弗·D·曼宁（Christopher D Manning）、普拉巴卡尔·拉加万（Prabhakar Raghavan）和欣里希·舒策（Hinrich Schütze）。《信息检索导论》（Introduction to information retrieval）。剑桥大学出版社，2008年。

[31] Nicholas Meisburger and Anshumali Shrivastava. Distributed tera-scale similarity search with mpi: Provably efficient similarity search over billions without a single distance computation. arXiv preprint arXiv:2008.03260, 2020.

[31] 尼古拉斯·迈斯伯格（Nicholas Meisburger）和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。基于MPI的分布式万亿级相似度搜索：无需单次距离计算的数十亿级高效相似度搜索。预印本arXiv:2008.03260，2020年。

[32] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. Ms marco: A human generated machine reading comprehension dataset. In Workshop on Cognitive Computing at NIPS, 2016.

[32] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞、高剑锋、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李邓。MS MARCO：一个人工生成的机器阅读理解数据集。收录于神经信息处理系统大会认知计算研讨会（Workshop on Cognitive Computing at NIPS），2016年。

[33] Priyanka Nigam, Yiwei Song, Vijai Mohan, Vihan Lakshman, Weitian Ding, Ankit Shingavi, Choon Hui Teo, Hao Gu, and Bing Yin. Semantic product search. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2876-2885, 2019.

[33] 普里扬卡·尼加姆（Priyanka Nigam）、宋依薇（Yiwei Song）、维贾伊·莫汉（Vijai Mohan）、维汉·拉克什曼（Vihan Lakshman）、丁伟天（Weitian Ding）、安基特·辛加维（Ankit Shingavi）、朱恩·许·特奥（Choon Hui Teo）、顾浩（Hao Gu）和尹冰（Bing Yin）。语义产品搜索。见《第25届ACM SIGKDD国际知识发现与数据挖掘会议论文集》，第2876 - 2885页，2019年。

[34] Steve Olenski. Why brands are fighting over milliseconds, Nov 2016.

[34] 史蒂夫·奥伦斯基（Steve Olenski）。品牌为何为毫秒而战，2016年11月。

[35] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[35] 亚当·帕兹克（Adam Paszke）、山姆·格罗斯（Sam Gross）、弗朗西斯科·马萨（Francisco Massa）、亚当·勒雷尔（Adam Lerer）、詹姆斯·布拉德伯里（James Bradbury）、格雷戈里·查南（Gregory Chanan）、特雷弗·基林（Trevor Killeen）、林泽明（Zeming Lin）、娜塔莉亚·吉梅尔申（Natalia Gimelshein）、卢卡·安蒂加（Luca Antiga）等。PyTorch：一种命令式风格的高性能深度学习库。《神经信息处理系统进展》，第32卷，2019年。

[36] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532-1543, 2014.

[36] 杰弗里·彭宁顿（Jeffrey Pennington）、理查德·索舍尔（Richard Socher）和克里斯托弗·D·曼宁（Christopher D. Manning）。GloVe：用于词表示的全局向量。见《自然语言处理经验方法会议（EMNLP）》，第1532 - 1543页，2014年。

[37] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. Plaid: An efficient engine for late interaction retrieval. arXiv preprint arXiv:2205.09707, 2022.

[37] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。Plaid：一种用于后期交互检索的高效引擎。预印本arXiv:2205.09707，2022年。

[38] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488, 2021.

[38] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。ColBERTv2：通过轻量级后期交互实现高效检索。预印本arXiv:2112.01488，2021年。

[39] Aneesh Sharma, C Seshadhri, and Ashish Goel. When hashes met wedges: A distributed algorithm for finding high similarity vectors. In Proceedings of the 26th International Conference on World Wide Web, pages 431-440, 2017.

[39] 阿尼什·夏尔马（Aneesh Sharma）、C·塞沙德里（C Seshadhri）和阿希什·戈尔（Ashish Goel）。当哈希遇到楔形：一种用于查找高相似向量的分布式算法。见《第26届万维网国际会议论文集》，第431 - 440页，2017年。

[40] Anshumali Shrivastava and Ping Li. Improved asymmetric locality sensitive hashing (alsh) for maximum inner product search (mips). In Proceedings of the Thirty-First Conference on Uncertainty in Artificial Intelligence, pages 812-821, 2015.

[40] 安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）和平·李（Ping Li）。用于最大内积搜索（MIPS）的改进非对称局部敏感哈希（ALSH）。见《第三十一届人工智能不确定性会议论文集》，第812 - 821页，2015年。

[41] Yiqiu Wang, Anshumali Shrivastava, Jonathan Wang, and Junghee Ryu. Flash: Randomized algorithms accelerated over cpu-gpu for ultra-high dimensional similarity search. arXiv preprint arXiv:1709.01190, 2017.

[41] 王艺秋（Yiqiu Wang）、安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）、乔纳森·王（Jonathan Wang）和柳正姬（Junghee Ryu）。Flash：在CPU - GPU上加速的用于超高维相似性搜索的随机算法。预印本arXiv:1709.01190，2017年。

[42] Jun Yang, Yu-Gang Jiang, Alexander G Hauptmann, and Chong-Wah Ngo. Evaluating bag-of-visual-words representations in scene classification. In Proceedings of the international workshop on Workshop on multimedia information retrieval, pages 197-206, 2007.

[42] 杨军（Jun Yang）、蒋玉刚（Yu - Gang Jiang）、亚历山大·G·豪普特曼（Alexander G Hauptmann）和吴崇华（Chong - Wah Ngo）。评估场景分类中的视觉词袋表示。见《多媒体信息检索国际研讨会论文集》，第197 - 206页，2007年。

[43] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. CoRR, abs/1806.01973, 2018.

[43] 应睿（Rex Ying）、何瑞宁（Ruining He）、陈开封（Kaifeng Chen）、庞·埃克松巴查伊（Pong Eksombatchai）、威廉·L·汉密尔顿（William L. Hamilton）和尤雷·莱斯科维奇（Jure Leskovec）。用于网络规模推荐系统的图卷积神经网络。CoRR，abs/1806.01973，2018年。

## A Proofs of Main Results

## 主要结果的证明

Lemma 4.1.1. If $\varphi \left( x\right)  : \mathbb{R} \rightarrow  \mathbb{R}$ is $\left( {\alpha ,\beta }\right)$ -maximal on an interval I,then the following function $\sigma \left( x\right)  : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}$ is $\left( {\alpha ,\frac{\beta }{m}}\right)$ -maximal on $U = {I}^{m}$ :

引理4.1.1。如果$\varphi \left( x\right)  : \mathbb{R} \rightarrow  \mathbb{R}$在区间I上是$\left( {\alpha ,\beta }\right)$ - 极大的，那么以下函数$\sigma \left( x\right)  : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}$在$U = {I}^{m}$上是$\left( {\alpha ,\frac{\beta }{m}}\right)$ - 极大的：

$$
\sigma \left( \mathbf{x}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\varphi \left( {x}_{i}\right) 
$$

Proof. Take some $\mathbf{x} \in  D$ (so each ${x}_{i} \in  I$ ). Since in $\mathbb{R},\max \left( x\right)  = x$ ,we have from the definition of $\left( {\alpha ,\beta }\right)$ -maximal that

证明。取一些$\mathbf{x} \in  D$（因此每个${x}_{i} \in  I$）。由于在$\mathbb{R},\max \left( x\right)  = x$中，根据$\left( {\alpha ,\beta }\right)$ - 极大的定义，我们有

$$
{\beta x} \leq  x \leq  {\alpha x}
$$

For the upper bound, we have

对于上界，我们有

$$
\sigma \left( \mathbf{x}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\varphi \left( {x}_{i}\right)  \leq  \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\alpha {x}_{i} = \frac{\alpha }{m}\mathop{\sum }\limits_{{i = 1}}^{m}{x}_{i} \leq  \frac{\alpha }{m}\left( {m\max \left( \mathbf{x}\right) }\right)  = \alpha \max \left( \mathbf{x}\right) 
$$

where the second inequality follows by the properties of the max function.

其中第二个不等式由最大值函数的性质得出。

For the lower bound, we have that

对于下界，我们有

$$
\sigma \left( \mathbf{x}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\varphi \left( {x}_{i}\right)  \geq  \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\beta {x}_{i} = \frac{\beta }{m}\mathop{\sum }\limits_{{i = 1}}^{m}{x}_{i} \geq  \frac{\beta }{m}\max \mathbf{x}
$$

where the the second inequality again follows by the properties of the max function.

其中第二个不等式同样由最大值函数的性质得出。

Lemma 4.1.2. Assume $\sigma$ is $\left( {\alpha ,\beta }\right)$ -maximal. Let $0 < {s}_{\max } < 1$ be the maximum similarity between a query vector and the vectors in the target set and let $\widehat{\mathbf{s}}$ be the set of estimated similarity scores. Given a threshold $\alpha {s}_{\max } < \tau  < \alpha$ ,we write $\Delta  = \tau  - \alpha {s}_{\max }$ ,and we have

引理4.1.2。假设 $\sigma$ 是 $\left( {\alpha ,\beta }\right)$ -极大的。设 $0 < {s}_{\max } < 1$ 为查询向量与目标集中向量之间的最大相似度，设 $\widehat{\mathbf{s}}$ 为估计相似度得分的集合。给定一个阈值 $\alpha {s}_{\max } < \tau  < \alpha$ ，我们记 $\Delta  = \tau  - \alpha {s}_{\max }$ ，并且我们有

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \alpha {s}_{\max } + \Delta }\right\rbrack   \leq  m{\gamma }^{L}
$$

for $\gamma  = {\left( \frac{{s}_{\max }\left( {\alpha  - \tau }\right) }{\tau \left( {1 - {s}_{\max }}\right) }\right) }^{\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{\max }}\right) }{\alpha  - \tau }\right)  \in  \left( {{s}_{\max },1}\right)$ . Furthermore,this expression for $\gamma$ is increasing in ${s}_{\max }$ and decreasing in $\tau$ ,and $\gamma$ has one sided limits $\mathop{\lim }\limits_{{\tau  \searrow  \alpha {s}_{\max }}}\gamma  = 1$ and $\mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}\gamma  = {s}_{\max }$ .

对于 $\gamma  = {\left( \frac{{s}_{\max }\left( {\alpha  - \tau }\right) }{\tau \left( {1 - {s}_{\max }}\right) }\right) }^{\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{\max }}\right) }{\alpha  - \tau }\right)  \in  \left( {{s}_{\max },1}\right)$ 。此外， $\gamma$ 的这个表达式关于 ${s}_{\max }$ 单调递增，关于 $\tau$ 单调递减，并且 $\gamma$ 有单侧极限 $\mathop{\lim }\limits_{{\tau  \searrow  \alpha {s}_{\max }}}\gamma  = 1$ 和 $\mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}\gamma  = {s}_{\max }$ 。

Proof. We first apply a generic Chernoff bound to $\sigma \left( \widehat{\mathbf{s}}\right)$ ,which gives us the following bounds for any

证明。我们首先对 $\sigma \left( \widehat{\mathbf{s}}\right)$ 应用一个通用的切尔诺夫界（Chernoff bound），这为我们给出了对于任意……的以下界

$t > 0$ :

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \tau }\right\rbrack   = \Pr \left\lbrack  {{e}^{{t\sigma }\left( \widehat{\mathbf{s}}\right) } \geq  {e}^{t\tau }}\right\rbrack   \leq  \frac{\mathbb{E}\left\lbrack  {e}^{{t\sigma }\left( \widehat{\mathbf{s}}\right) }\right\rbrack  }{{e}^{t\tau }}
$$

We now proceed by continuing to bound the numerator. Because $\sigma$ is $\left( {\alpha ,\beta }\right)$ -maximal,we can bound $\sigma \left( \widehat{\mathbf{s}}\right)$ with $\alpha \max \widehat{\mathbf{s}}$ . We can further bound $\max \widehat{\mathbf{s}}$ by bounding the maximum with the sum and the sum with $m$ times the maximal element. We are now left with the formula for the moment generating function for ${\widehat{s}}_{\max } \cdot  {\widehat{s}}_{\max } \sim$ scaled binomial ${L}^{-1}\mathcal{B}\left( {{s}_{\max },L}\right)$ ,so we can directly substitute the binomial moment generating function into the expression:

现在我们继续对分子进行界定。因为 $\sigma$ 是 $\left( {\alpha ,\beta }\right)$ -极大的，我们可以用 $\alpha \max \widehat{\mathbf{s}}$ 来界定 $\sigma \left( \widehat{\mathbf{s}}\right)$ 。我们可以通过用和来界定最大值，并用 $m$ 乘以最大元素来界定和，从而进一步界定 $\max \widehat{\mathbf{s}}$ 。现在我们剩下 ${\widehat{s}}_{\max } \cdot  {\widehat{s}}_{\max } \sim$ 缩放二项式 ${L}^{-1}\mathcal{B}\left( {{s}_{\max },L}\right)$ 的矩生成函数的公式，所以我们可以直接将二项式矩生成函数代入表达式：

$$
\mathbb{E}\left\lbrack  {e}^{{t\sigma }\left( \widehat{\mathbf{s}}\right) }\right\rbrack   \leq  \mathbb{E}\left\lbrack  {e}^{{t\alpha }\mathop{\max }\limits_{j}{\widehat{s}}_{j}}\right\rbrack   \leq  \mathop{\sum }\limits_{{j = 1}}^{{m}_{i}}\mathbb{E}\left\lbrack  {e}^{{t\alpha }{\widehat{s}}_{j}}\right\rbrack   \leq  m\mathbb{E}\left\lbrack  {e}^{{t\alpha }{\widehat{s}}_{\max }}\right\rbrack  
$$

$$
 = m{\left( 1 - {s}_{\max } + {s}_{\max }{e}^{\frac{\alpha t}{L}}\right) }^{L}
$$

Combining these two equations yields the following bound:

将这两个方程结合起来得到以下界：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \tau }\right\rbrack   \leq  m{e}^{-{t\tau }}{\left( 1 - {s}_{\max } + {s}_{\max }{e}^{\frac{\alpha t}{L}}\right) }^{L}
$$

We wish to select a value of $t$ to minimize the upper bound. By setting the derivative of the upper bound to zero,and imposing $0 < \tau  < \alpha ,\alpha  \geq  1$ ,and $0 < {s}_{\max } < 1$ ,we find that

我们希望选择一个 $t$ 的值来最小化上界。通过令上界的导数为零，并施加 $0 < \tau  < \alpha ,\alpha  \geq  1$ 和 $0 < {s}_{\max } < 1$ 的条件，我们发现

$$
{t}^{ \star  } = \frac{L}{\alpha }\ln \left( \frac{\tau \left( {1 - {s}_{\max }}\right) }{{s}_{\max }\left( {\alpha  - \tau }\right) }\right) 
$$

This is greater than zero when the numerator inside the $\ln$ is greater than the denominator,or equivalently when $\tau  > {s}_{\max }\alpha$ . Thus the valid range for $\tau$ is $\left( {{s}_{\max }\alpha ,\alpha }\right)$ (and similarly the valid range for ${s}_{\max }$ is $\left( {0,\tau /\alpha }\right)$ ). These bounds have a natural interpretation: to be meaningful,the threshold must be between the expected value and the maximum value for $\alpha$ times a $p = {s}_{\max }$ binomial. Substituting $t = {t}^{ \star  }$ into our upper bound,we obtain:

当 $\ln$ 内的分子大于分母时，即等价于 $\tau  > {s}_{\max }\alpha$ 时，这个值大于零。因此 $\tau$ 的有效范围是 $\left( {{s}_{\max }\alpha ,\alpha }\right)$ （类似地， ${s}_{\max }$ 的有效范围是 $\left( {0,\tau /\alpha }\right)$ ）。这些界有一个自然的解释：为了有意义，阈值必须在 $\alpha$ 乘以一个 $p = {s}_{\max }$ 二项式的期望值和最大值之间。将 $t = {t}^{ \star  }$ 代入我们的上界，我们得到：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \tau }\right\rbrack   \leq  m{\left( {\left( \frac{\tau \left( {1 - {s}_{\max }}\right) }{{s}_{\max }\left( {\alpha  - \tau }\right) }\right) }^{-\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{\max }}\right) }{\alpha  - \tau }\right) \right) }^{L}
$$

Thus we have that

因此我们有

$$
\gamma  = {\left( \frac{{s}_{\max }\left( {\alpha  - \tau }\right) }{\tau \left( {1 - {s}_{\max }}\right) }\right) }^{\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{\max }}\right) }{\alpha  - \tau }\right) 
$$

We will now prove our claims about $\gamma$ viewed as a function of ${s}_{\max } \in  \left( {0,\tau /\alpha }\right)$ and $\tau  \in  \left( {{s}_{\max }\alpha ,\alpha }\right)$ . We will first examine the limits of $\gamma$ with respect to $\tau$ at the ends of its range. Since $\gamma$ is continuous, we can find one of the limits by direct substitution:

现在我们将证明关于把 $\gamma$ 看作 ${s}_{\max } \in  \left( {0,\tau /\alpha }\right)$ 和 $\tau  \in  \left( {{s}_{\max }\alpha ,\alpha }\right)$ 的函数的断言。我们首先考察 $\gamma$ 在 $\tau$ 取值范围端点处的极限。由于 $\gamma$ 是连续的，我们可以通过直接代入来求出其中一个极限：

$$
\mathop{\lim }\limits_{{\tau  \searrow  {s}_{\max }\alpha }}\gamma  = \mathop{\lim }\limits_{{{s}_{\max } \nearrow  \tau /\alpha }}\gamma  = {\left( \frac{\tau /\alpha \left( {\alpha  - \tau }\right) }{\tau \left( {1 - \tau /\alpha }\right) }\right) }^{\frac{\tau }{\alpha }}\left( \frac{\alpha \left( {1 - \tau /\alpha }\right) }{\alpha  - \tau }\right)  = {1}^{\frac{\tau }{\alpha }} * 1 = 1
$$

The second limit is harder; we merge $\gamma$ into one exponent and then simplify:

第二个极限更难；我们将 $\gamma$ 合并为一个指数，然后进行化简：

$$
\mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}\gamma  = \mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}{\left( \frac{{s}_{\max }{\left( \alpha  - \tau \right) }^{1 - \alpha }{}^{\tau }{\alpha }^{\alpha /\tau }}{\tau {\left( 1 - {s}_{\max }\right) }^{1 - \alpha /\tau }}\right) }^{\frac{\tau }{\alpha }}
$$

$$
 = \mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}\frac{{s}_{\max }{\left( \alpha  - \tau \right) }^{1 - {\alpha \tau }}{\alpha }^{\alpha /\tau }}{\tau {\left( 1 - {s}_{\max }\right) }^{1 - \alpha /\tau }}
$$

$$
 = \mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}{s}_{\max }{\left( \alpha  - \tau \right) }^{1 - \alpha /\tau } = \mathop{\lim }\limits_{{\tau  \nearrow  \alpha }}{\left( {s}_{\max }{\left( \alpha  - \tau \right) }^{\alpha  - \tau }\right) }^{-1/\tau }
$$

$$
 = {s}_{\max }{\left( \mathop{\lim }\limits_{{\alpha  - \tau  \searrow  0}}\left( {\left( \alpha  - \tau \right) }^{\alpha  - \tau }\right) \right) }^{-1/\alpha }
$$

$$
 = {s}_{\max }{\left( 1\right) }^{-1/\alpha } = {s}_{\max }
$$

where we use the fact that $\mathop{\lim }\limits_{{x \searrow  0}}{x}^{x} = {e}^{\mathop{\lim }\limits_{{x \searrow  0}}x\ln \left( x\right) } = 1$ (we can see that $\mathop{\lim }\limits_{{x \rightarrow  0}}x\ln \left( x\right)  =$ $\mathop{\lim }\limits_{{x \rightarrow  0}}\ln \left( x\right) /\left( {1/x}\right)  = 0$ with L’Hopital’s rule). We next find the partial derivatives of $\gamma$ :

在这里我们利用了$\mathop{\lim }\limits_{{x \searrow  0}}{x}^{x} = {e}^{\mathop{\lim }\limits_{{x \searrow  0}}x\ln \left( x\right) } = 1$这一事实（我们可以用洛必达法则（L’Hopital’s rule）看出$\mathop{\lim }\limits_{{x \rightarrow  0}}x\ln \left( x\right)  =$ $\mathop{\lim }\limits_{{x \rightarrow  0}}\ln \left( x\right) /\left( {1/x}\right)  = 0$）。接下来我们求$\gamma$的偏导数：

$$
\frac{\delta \gamma }{\delta {s}_{\max }} = \frac{\left( {\tau  - \alpha {s}_{\max }}\right) {\left( \frac{\alpha {s}_{\max } - {s}_{\max }\tau }{\tau  - {s}_{\max }\tau }\right) }^{\frac{\tau }{\alpha }}}{{s}_{\max }\left( {\alpha  - \tau }\right) }\;\frac{\delta \gamma }{\delta \tau } = \frac{\left( {{s}_{\max } - 1}\right) {\left( \frac{\alpha {s}_{\max } - {s}_{\max }\tau }{\tau  - {s}_{\max }\tau }\right) }^{\frac{\tau }{\alpha }}\ln \left( \frac{\tau  - {s}_{\max }\tau }{\alpha {s}_{\max }\tau }\right) }{\alpha  - \tau }
$$

We are interested in the signs of these partial derivatives. First examining $\frac{\delta \gamma }{\delta {s}_{\max }},\tau  > \alpha {s}_{\max } \Rightarrow$ $\tau  - \alpha {s}_{\max } > 0$ . Similarly, $\alpha  > \tau  \Rightarrow  \alpha  - \tau  > 0$ and ${s}_{\max }\left( {\alpha  - \tau }\right)  = {s}_{\max }\alpha  - {s}_{\max }\tau  > 0$ . Finally, ${s}_{\max } < 1 \Rightarrow  \tau \left( {1 - {s}_{\max }}\right)  = \tau  - \tau {s}_{\max } > 0$ . Thus every term is positive and the entire fraction is positive. Next examining $\frac{\delta \gamma }{\delta \tau }$ ,by similar logic $\alpha  - \tau  > 0$ and $\tau  - {s}_{\max }\tau  > 0$ and $\alpha {s}_{\max } - {s}_{\max }\tau  > 0$ . For the $\ln$ ,since $\tau  > \alpha {s}_{\max },\tau  - {s}_{\max }\tau  > \alpha {s}_{\max } - {s}_{\max }\tau$ ,so the numerator is greater than the denominator and the $\ln$ is positive. Finally,since ${s}_{\max } < 1,{s}_{\max } - 1 < 0$ ,and thus the entire fraction has a single negative term in the product, so it is negative.

我们关注这些偏导数的符号。首先考察$\frac{\delta \gamma }{\delta {s}_{\max }},\tau  > \alpha {s}_{\max } \Rightarrow$ $\tau  - \alpha {s}_{\max } > 0$。类似地，$\alpha  > \tau  \Rightarrow  \alpha  - \tau  > 0$和${s}_{\max }\left( {\alpha  - \tau }\right)  = {s}_{\max }\alpha  - {s}_{\max }\tau  > 0$。最后，${s}_{\max } < 1 \Rightarrow  \tau \left( {1 - {s}_{\max }}\right)  = \tau  - \tau {s}_{\max } > 0$。因此每一项都是正的，整个分式也是正的。接下来考察$\frac{\delta \gamma }{\delta \tau }$，通过类似的逻辑，$\alpha  - \tau  > 0$、$\tau  - {s}_{\max }\tau  > 0$和$\alpha {s}_{\max } - {s}_{\max }\tau  > 0$。对于$\ln$，由于$\tau  > \alpha {s}_{\max },\tau  - {s}_{\max }\tau  > \alpha {s}_{\max } - {s}_{\max }\tau$，所以分子大于分母，并且$\ln$是正的。最后，由于${s}_{\max } < 1,{s}_{\max } - 1 < 0$，因此整个分式在乘积中有一项为负，所以它是负的。

This completes our lemma: $\gamma$ is a strictly decreasing function of $\tau$ and a strictly increasing function of ${s}_{\max }$ . Since $\tau$ is decreasing and has a leftward limit of 1 and a rightward limit of ${s}_{\max }$ ,all values for $\gamma$ are in $\left( {{s}_{\max },1}\right)$ .

这就完成了我们的引理：$\gamma$是$\tau$的严格递减函数，是${s}_{\max }$的严格递增函数。由于$\tau$是递减的，并且左极限为1，右极限为${s}_{\max }$，所以$\gamma$的所有值都在$\left( {{s}_{\max },1}\right)$内。

First,we will make a substitution. We note that $\gamma$ is a strictly decreasing function on this interval of $\tau$ with range $\left( {{s}_{\max },1}\right)$ . To see this,we will first make the following change of variabls:

首先，我们将进行一次代换。我们注意到$\gamma$在$\tau$的这个区间上是一个严格递减函数，其值域为$\left( {{s}_{\max },1}\right)$。为了说明这一点，我们首先进行如下变量替换：

$$
\tau  = \frac{\alpha \left( {k + s}\right) }{k + 1}
$$

for $k \in  \left( {0,\infty }\right)$ . This parameterizes $\tau  \in  \left( {{s}_{\max }\alpha ,\alpha }\right)$ as a weighted sum of ${s}_{\max }\alpha$ and $\alpha$ . Plugging in and simplifying, we have that

对于$k \in  \left( {0,\infty }\right)$。这将$\tau  \in  \left( {{s}_{\max }\alpha ,\alpha }\right)$参数化为${s}_{\max }\alpha$和$\alpha$的加权和。代入并化简后，我们得到

$$
\gamma  = {\left( \frac{{s}_{\max }}{k + {s}_{\max }}\right) }^{\frac{k + {s}_{\max }}{k + 1}}\left( {k + 1}\right) 
$$

This is a continous function over $k \in  \left( {0,\infty }\right)$ and ${s}_{\max } \in  \left( \right)$

这是一个在$k \in  \left( {0,\infty }\right)$和${s}_{\max } \in  \left( \right)$上的连续函数

Lemma 4.1.3. With the same assumptions as Lemma 4.1.2 and given $\Delta  > 0$ ,we have:

引理4.1.3。在与引理4.1.2相同的假设下，并且给定$\Delta  > 0$，我们有：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \leq  \beta {s}_{\max } - \Delta }\right\rbrack   \leq  2{e}^{-{2L}{\Delta }^{2}/{\beta }^{2}}
$$

Proof. We will prove this lemma with a chain of inequalities,starting with $\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \leq  \beta {s}_{\max } - \Delta }\right\rbrack$ :

证明。我们将通过一系列不等式来证明这个引理，从$\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \leq  \beta {s}_{\max } - \Delta }\right\rbrack$开始：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \leq  \beta {s}_{\max } - \Delta }\right\rbrack   \leq  \Pr \left\lbrack  {\beta \max \widehat{\mathbf{s}} \leq  \beta {s}_{\max } - \Delta }\right\rbrack  
$$

$$
 \leq  \Pr \left\lbrack  {\beta {s}_{\max } \leq  \beta {s}_{\max } - \Delta }\right\rbrack  
$$

$$
 = \Pr \left\lbrack  {\beta {s}_{\max } - \beta {s}_{\max } \geq  \Delta }\right\rbrack  
$$

$$
 \leq  \Pr \left\lbrack  {\left| {\beta {s}_{\max } - \beta {s}_{\max }}\right|  \geq  \Delta }\right\rbrack   = \Pr \left\lbrack  {\left| {\beta {s}_{\max } - \beta {s}_{\max }}\right|  \geq  \Delta }\right\rbrack  
$$

$$
 \leq  2{e}^{-{2L}{\Delta }^{2}/{\beta }^{2}}
$$

The explanations for each step are as follows:

每一步的解释如下：

1. Because $\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \beta \max \mathbf{s}$ ,we can replace $\sigma \left( \widehat{\mathbf{s}}\right)$ with $\beta \max \mathbf{s}$ and the probability will be strictly larger.

1. 因为$\sigma \left( \widehat{\mathbf{s}}\right)  \geq  \beta \max \mathbf{s}$，我们可以用$\beta \max \mathbf{s}$替换$\sigma \left( \widehat{\mathbf{s}}\right)$，并且概率会严格增大。

2. By the definition of the max operator,each individual ${\widehat{s}}_{i} \leq  \max \widehat{\mathbf{s}}$ ,and in particular this is true for $\widehat{{s}_{\max }}$ (the estimated similarity for the ground-truth maximum similarity vector). Thus,we have $\beta {s}_{\max } \leq  \beta \max \widehat{\mathbf{s}}$ ,so we can again apply a replacement to get a further upper bound.

2. 根据最大值运算符的定义，每个单独的${\widehat{s}}_{i} \leq  \max \widehat{\mathbf{s}}$，特别是对于$\widehat{{s}_{\max }}$（真实最大相似度向量的估计相似度）也是如此。因此，我们有$\beta {s}_{\max } \leq  \beta \max \widehat{\mathbf{s}}$，所以我们可以再次进行替换以得到一个更大的上界。

3. Rearranging.

3. 重新排列。

4. Because $\Pr \left\lbrack  {\left| {a - b}\right|  \geq  c}\right\rbrack   = \Pr \left\lbrack  {a - b \geq  c}\right\rbrack   + \Pr \left\lbrack  {b - a \geq  c}\right\rbrack$

4. 因为$\Pr \left\lbrack  {\left| {a - b}\right|  \geq  c}\right\rbrack   = \Pr \left\lbrack  {a - b \geq  c}\right\rbrack   + \Pr \left\lbrack  {b - a \geq  c}\right\rbrack$

5. ${\widehat{s}}_{\max }$ is the sum of $L$ Bernoulli trials with success probability ${s}_{\max }$ and scaled by $\beta /L$ . Thus,we can directly apply the Hoeffding ineugliaty with $L$ trials with success probability $\frac{\beta {s}_{\max }}{L}$ .

5. ${\widehat{s}}_{\max }$是$L$次伯努利试验（Bernoulli trials）的和，成功概率为${s}_{\max }$，并按$\beta /L$进行缩放。因此，我们可以直接对成功概率为$\frac{\beta {s}_{\max }}{L}$的$L$次试验应用霍夫丁不等式（Hoeffding inequality）。

Theorem 4.2. Let ${S}^{ \star  }$ be the set with the maximum $F\left( {Q,S}\right)$ and let ${S}_{i}$ be any other set. Let ${B}^{ \star  }$ and ${B}_{i}$ be the following sums (which are lower and upper bounds for $F\left( {Q,{S}^{ \star  }}\right)$ and $F\left( {Q,{S}_{i}}\right)$ , respectively)

定理4.2。设${S}^{ \star  }$是具有最大$F\left( {Q,S}\right)$的集合，设${S}_{i}$是任何其他集合。设${B}^{ \star  }$和${B}_{i}$是以下和（分别是$F\left( {Q,{S}^{ \star  }}\right)$和$F\left( {Q,{S}_{i}}\right)$的下界和上界）

$$
{B}^{ \star  } = \frac{\beta }{{m}_{q}}\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}{s}_{\max }\left( {{q}_{j},{S}^{ \star  }}\right) \;{B}_{i} = \frac{\alpha }{{m}_{q}}\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}{s}_{\max }\left( {{q}_{j},{S}_{i}}\right) 
$$

Here, ${s}_{\max }\left( {q,S}\right)$ is the maximum similarity between a query vector $q$ and any element of the target set $S$ . Let ${B}^{\prime }$ be the maximum value of ${B}_{i}$ over any set ${S}_{i} \neq  S$ . Let $\Delta$ be the following value (proportional to the difference between the lower and upper bounds)

这里，${s}_{\max }\left( {q,S}\right)$是查询向量$q$与目标集合$S$中任何元素之间的最大相似度。设${B}^{\prime }$是${B}_{i}$在任何集合${S}_{i} \neq  S$上的最大值。设$\Delta$是以下值（与下界和上界之间的差值成比例）

$$
\Delta  = \left( {{B}^{ \star  } - {B}^{\prime }}\right) /3
$$

If $\Delta  > 0$ ,a DESSERT structure with the following value ${}^{3}$ of $L$ solves the search problem from Definition 1.1 with probability $1 - \delta$ .

如果$\Delta  > 0$，一个具有以下$L$值${}^{3}$的DESSERT结构以概率$1 - \delta$解决定义1.1中的搜索问题。

$$
L = O\left( {\log \left( \frac{N{m}_{q}m}{\delta }\right) }\right) 
$$

Proof. For set ${S}^{ \star  }$ to have the highest estimated score $\widehat{F}\left( {Q,{S}^{ \star  }}\right)$ ,we need all other sets to have lower scores. Our overall proof strategy will find a minimum $L$ that upper bounds the probability that each inner aggregation of a set $S \neq  {S}^{ * }$ is greater than $\Delta  + \alpha {s}_{j,\max }^{\prime }$ and a minimum $L$ that lower bounds the probability that the inner aggregation of ${S}^{ * }$ is less $\beta {s}_{j,\max }^{ * } - \Delta$ . Finally,we will show that an $L$ that is a maximum of these two values solves the search problem.

证明。为了使集合${S}^{ \star  }$具有最高的估计得分$\widehat{F}\left( {Q,{S}^{ \star  }}\right)$，我们需要所有其他集合的得分更低。我们的总体证明策略将找到一个最小的$L$，它是集合$S \neq  {S}^{ * }$的每个内部聚合大于$\Delta  + \alpha {s}_{j,\max }^{\prime }$的概率的上界，以及一个最小的$L$，它是${S}^{ * }$的内部聚合小于$\beta {s}_{j,\max }^{ * } - \Delta$的概率的下界。最后，我们将证明这两个值中的最大值$L$可以解决搜索问题。

Upper Bound: We start with the upper bound on ${S}_{i} \neq  {S}^{ * }$ : we have from Lemma 4.1.2 that

上界：我们从${S}_{i} \neq  {S}^{ * }$的上界开始：根据引理4.1.2，我们有

$$
\Pr \left\lbrack  {\sigma \left( {\widehat{{\mathbf{s}}_{\mathbf{i}}},{q}_{j}}\right)  \geq  \alpha {s}_{i,\max } + \Delta }\right\rbrack   \leq  m{\gamma }_{i}^{L}
$$

with

其中

$$
{\gamma }_{i} = {\left( \frac{\left( {\Delta  + \alpha {s}_{i,\max }}\right) \left( {1 - {s}_{i,\max }}\right) }{{s}_{i,\max }\left( {\alpha  - \left( {\Delta  + \alpha {s}_{i,\max }}\right) }\right) }\right) }^{-\frac{\Delta  + \alpha {s}_{i,\max }}{\alpha }}\left( \frac{\alpha \left( {1 - {s}_{i,\max }}\right) }{\alpha  - \left( {\Delta  + \alpha {s}_{i,\max }}\right) }\right) 
$$

and ${\gamma }_{i} \in  \left( {0,1}\right)$ . To make our analysis simpler,we are interested in the maximum ${\gamma }_{\max }$ of all these ${\gamma }_{i}$ as a function of $\Delta$ ,since then all of these bounds will hold with the same $\gamma$ ,making it easy to solve for L. Since $\mathop{\lim }\limits_{{{s}_{\max } \searrow  0}}\gamma  = 0$ and $\mathop{\lim }\limits_{{{s}_{\max } \nearrow  1 - \Delta }} = 1 - \Delta /\alpha$ ,there must be some ${\gamma }_{\max } \in  \left( {1 - \Delta /\alpha ,1}\right)$ that maximizes this expression over any ${s}_{\max }$ . This exact maximum is hard to find analytically,but we are guaranteed that it is less than 1 by Lemma 4.1.2. We will use the term ${\gamma }_{\max }$ in our analysis, since it is data dependent and guaranteed to be in the range(0,1). We also numerically plot some values of ${\gamma }_{\max }$ here with $\alpha  = 1$ to give some intuition for what the function looks like over different $\Delta$ ; we note that it is decreasing in $\Delta$ and approximates a linear function for $\Delta  >  > 0$ .

并且 ${\gamma }_{i} \in  \left( {0,1}\right)$ 。为了简化我们的分析，我们关注所有这些 ${\gamma }_{i}$ 关于 $\Delta$ 的函数的最大 ${\gamma }_{\max }$ ，因为这样所有这些边界条件都将以相同的 $\gamma$ 成立，从而便于求解 L。由于 $\mathop{\lim }\limits_{{{s}_{\max } \searrow  0}}\gamma  = 0$ 和 $\mathop{\lim }\limits_{{{s}_{\max } \nearrow  1 - \Delta }} = 1 - \Delta /\alpha$ ，在任何 ${s}_{\max }$ 上必定存在某个 ${\gamma }_{\max } \in  \left( {1 - \Delta /\alpha ,1}\right)$ 能使该表达式达到最大值。这个精确的最大值很难通过解析方法求得，但根据引理 4.1.2 我们可以确定它小于 1。在我们的分析中，我们将使用术语 ${\gamma }_{\max }$ ，因为它依赖于数据且必定在区间(0,1)内。我们还在此处用 $\alpha  = 1$ 对 ${\gamma }_{\max }$ 的一些值进行了数值绘图，以便直观了解该函数在不同 $\Delta$ 下的情况；我们注意到它在 $\Delta$ 上是递减的，并且在 $\Delta  >  > 0$ 时近似为线性函数。

$$
\Pr \left\lbrack  {\sigma \left( {{\widehat{\mathbf{s}}}_{\mathbf{i}},{q}_{j}}\right)  \geq  \alpha {s}_{i,\max } + \Delta }\right\rbrack   \leq  m{\gamma }_{i}^{L} \leq  m{\gamma }_{\max }^{L} \leq  m{\left( {\gamma }_{\max }\right) }^{\log \frac{2\left( {N - 1}\right) {m}_{q}m}{\delta }\log {\left( \frac{1}{{\gamma }_{\max }}\right) }^{-1}} = \frac{\delta }{2\left( {N - 1}\right) {m}_{q}}
$$

<!-- Media -->

<!-- figureText: 0.0 Numerical Solutions for ${\gamma }_{\max }$ with $\alpha  = 1$ -->

<img src="https://cdn.noedgeai.com/0195a865-affe-7ef5-9800-93ec0736f0e2_16.jpg?x=341&y=1171&w=739&h=585&r=0"/>

To hold with the union bound over all $N - 1$ target sets and all ${m}_{q}$ query vectors with probability $\frac{\delta }{2}$ ,we want the probability that our bound holds on a single set and query vector to be less than $\frac{\delta }{2\left( {N - 1}\right) {m}_{q}}$ . We find that this is true with $L \geq  \log \frac{2\left( {N - 1}\right) {m}_{q}m}{\delta }\log {\left( \frac{1}{{\gamma }_{\max }}\right) }^{-1}$ for any ${q}_{j}$ and ${S}_{i}$ :

为了使所有 $N - 1$ 个目标集和所有 ${m}_{q}$ 个查询向量在联合边界条件下以概率 $\frac{\delta }{2}$ 成立，我们希望边界条件在单个集合和查询向量上成立的概率小于 $\frac{\delta }{2\left( {N - 1}\right) {m}_{q}}$ 。我们发现对于任何 ${q}_{j}$ 和 ${S}_{i}$ ，当 $L \geq  \log \frac{2\left( {N - 1}\right) {m}_{q}m}{\delta }\log {\left( \frac{1}{{\gamma }_{\max }}\right) }^{-1}$ 时这是成立的：

<!-- Media -->

---

<!-- Footnote -->

${}^{3}L$ additionally depends on the data-dependent parameter $\Delta$ ,which we elide in the asymptotic bound; see the proof in the appendix for the full expression for $L$ .

${}^{3}L$ 还依赖于与数据相关的参数 $\Delta$ ，在渐近边界中我们省略了该参数；完整的 $L$ 表达式见附录中的证明。

<!-- Footnote -->

---

Lower Bound We next examine the lower bound on ${S}^{ * }$ : we have from Lemma 4.1.3 that

下界 接下来我们研究 ${S}^{ * }$ 的下界：根据引理 4.1.3 我们有

$$
\Pr \left\lbrack  {\sigma \left( {{\widehat{\mathbf{s}}}^{ * },{q}_{j}}\right)  \leq  \beta {s}_{*,\max } - \Delta }\right\rbrack   \leq  2{e}^{-{2L}{\Delta }^{2}/{\beta }^{2}}
$$

To hold with the union bound over all ${m}_{q}$ query vectors with probability $\frac{\delta }{2}$ ,we want the probability that our bound holds on a single set and query vector to be less than $\frac{\delta }{2{m}_{q}}$ . We find that this is true with $L \geq  \frac{\log \left( \frac{4{m}_{q}}{\delta }\right) {\beta }^{2}}{2{\Delta }^{2}}$ for any ${q}_{j}$ :

为了以概率 $\frac{\delta }{2}$ 对所有 ${m}_{q}$ 个查询向量使用联合界，我们希望我们的界在单个集合和查询向量上成立的概率小于 $\frac{\delta }{2{m}_{q}}$。我们发现，对于任何 ${q}_{j}$，当 $L \geq  \frac{\log \left( \frac{4{m}_{q}}{\delta }\right) {\beta }^{2}}{2{\Delta }^{2}}$ 时这是成立的：

$$
\Pr \left\lbrack  {\sigma \left( {{\widehat{\mathbf{s}}}^{ * },{q}_{j}}\right)  \leq  \beta {s}_{*,\max } - \Delta }\right\rbrack   \leq  2{e}^{-{2L}{\Delta }^{2}/{\beta }^{2}} \leq  2{e}^{-2\frac{\log \left( \frac{4{m}_{q}}{\delta }\right) {\beta }^{2}}{2{\Delta }^{2}}{\Delta }^{2}/{\beta }^{2}} = \frac{\delta }{2{m}_{q}}
$$

## Putting it Together

## 综合起来

Let

设

$$
L = \max \left( {\frac{\log \frac{2\left( {N - 1}\right) {m}_{q}m}{\delta }}{\log \left( \frac{1}{{\gamma }_{\max }}\right) },\frac{\log \left( \frac{4{m}_{q}}{\delta }\right) {\beta }^{2}}{2{\Delta }^{2}}}\right) 
$$

Then the upper and lower bounds we derived in the last two sections both apply. Let1be the random variable that is 1 when the $m * {m}_{q} * \left( {N - 1}\right)$ upper bounds and the ${m}_{q}$ lower bounds hold and that is 0 otherwise. Consider all sets ${S}_{i} \neq  {S}^{ * }$ . Then the probability we solve the Vector Set Search Problem from Definition 1.1 is equal to the probability that all ${\forall }_{i},\left( {\widehat{F}\left( {Q,{S}^{ * }}\right)  - \widehat{F}\left( {Q,{S}_{i}}\right)  > 0}\right)$ . We now lower bound this probability:

那么我们在前两节中推导的上界和下界都适用。设 $m * {m}_{q} * \left( {N - 1}\right)$ 个上界和 ${m}_{q}$ 个下界成立时随机变量 ${S}_{i} \neq  {S}^{ * }$ 为 1，否则为 0。考虑所有集合 ${S}_{i} \neq  {S}^{ * }$。那么我们解决定义 1.1 中的向量集搜索问题的概率等于所有 ${\forall }_{i},\left( {\widehat{F}\left( {Q,{S}^{ * }}\right)  - \widehat{F}\left( {Q,{S}_{i}}\right)  > 0}\right)$ 成立的概率。我们现在对这个概率进行下界估计：

$$
\Pr \left( {{\forall }_{i}\left( {\widehat{F}\left( {Q,{S}^{ * }}\right)  - \widehat{F}\left( {Q,{S}_{i}}\right)  > 0}\right) }\right) 
$$

$$
 = \Pr \left( {{\forall }_{i}\left( {\frac{1}{{m}_{q}}\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}\sigma \left( {{\widehat{\mathbf{s}}}^{ * },{q}_{j}}\right)  - \frac{1}{{m}_{q}}\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}\sigma \left( {\widehat{{\mathbf{s}}_{i}},{q}_{j}}\right)  > 0}\right) }\right) 
$$

Definition of $\widehat{F}$

$$
 = \Pr \left( {{\forall }_{i}\left( {\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}\left( {\sigma \left( {{\widehat{\mathbf{s}}}^{ * },{q}_{j}}\right)  - \sigma \left( {\widehat{{\mathbf{s}}_{i}},{q}_{j}}\right) }\right)  > 0}\right) }\right) 
$$

$$
 = \Pr \left( {{\forall }_{i}\left( {\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}\left( {\sigma \left( {{\widehat{\mathbf{s}}}^{ * },{q}_{j}}\right)  - \sigma \left( {{\widehat{\mathbf{s}}}_{i},{q}_{j}}\right) }\right)  > 0 \mid  \mathbb{1} = 1}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) \;\Pr \left( A\right)  \geq  \Pr \left( {A \land  B}\right) 
$$

$$
 \geq  \Pr \left( {{\forall }_{i}\left( {\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}\left( {\beta {s}_{*,\max } - \Delta  - \left( {\alpha {s}_{i,\max } + \Delta }\right) }\right)  > 0}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) \;\text{ Bounds hold on }\mathbb{1} = 1
$$

$$
 = \Pr \left( {{\forall }_{i}\left( {\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}\left( {\beta {s}_{*,\max } - \alpha {s}_{i,\max }}\right)  > {2\Delta }\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

$$
 = \Pr \left( {{\forall }_{i}\left( {{m}_{q}\left( {{B}^{ * } - {B}_{i}}\right)  > {2\Delta }\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

Definition of ${B}^{ * },{B}_{i}$

$$
 \geq  \Pr \left( {{\forall }_{i}\left( {{m}_{q}\left( {{B}^{ * } - {B}^{\prime }}\right)  > {2\Delta }\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

Definition of ${B}^{\prime }$

$$
 \geq  \Pr \left( {{\forall }_{i}\left( {3{m}_{q}\Delta  > {2\Delta }\mathop{\sum }\limits_{{j = 1}}^{{m}_{q}}{w}_{j}}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

Definition of $\Delta$

$$
 \geq  \Pr \left( {{\forall }_{i}\left( {3{m}_{q}\Delta  > 2{m}_{q}\Delta }\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

${w}_{j} \leq  1$

$$
 = 1 * \left( {\mathbb{1} = 1}\right) 
$$

$\Delta  > 0$

$$
 = 1 - \left( {\mathbb{1} = 0}\right) 
$$

$$
 \geq  1 - \left( {m * {m}_{q} * \left( {N - 1}\right)  * \frac{\delta }{2\left( {N - 1}\right) {m}_{q}} + \frac{\delta }{2{m}_{q}} * {m}_{q}}\right)  = 1 - \delta 
$$

Union bound

and thus DESSERT solves the Vector Set Search Problem with this choice of $L$ . Finally,we can now examine the expression for $L$ to determine its asymptotic behavior. Dropping the positive data dependent constants $\frac{1}{{\gamma }_{\max }},\frac{1}{2{\Delta }^{2}}$ ,and ${\beta }^{2}$ ,the left term in the max for $L$ is $O\left( {\log \left( \frac{N{m}_{q}m}{\delta }\right) }\right)$ and the right term in the max is $O\left( {\log \left( \frac{{m}_{q}}{\delta }\right) }\right)$ ,and thus $L = O\left( {\log \left( \frac{N{m}_{q}m}{\delta }\right) }\right)$ .

因此，DESSERT（动态高效稀疏集检索，Dynamic Efficient Sparse Set Retrieval）使用这个 $L$ 的选择解决了向量集搜索问题。最后，我们现在可以检查 $L$ 的表达式以确定其渐近行为。去掉与数据相关的正常数 $\frac{1}{{\gamma }_{\max }},\frac{1}{2{\Delta }^{2}}$ 和 ${\beta }^{2}$，$L$ 取最大值时左边的项是 $O\left( {\log \left( \frac{N{m}_{q}m}{\delta }\right) }\right)$，右边的项是 $O\left( {\log \left( \frac{{m}_{q}}{\delta }\right) }\right)$，因此 $L = O\left( {\log \left( \frac{N{m}_{q}m}{\delta }\right) }\right)$。

Theorem 4.3. Suppose that each hash function call runs in time $O\left( d\right)$ and that $\left| {\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h}}\right|  < T\forall i,t,h$ for some positive threshold $T$ ,which we treat as a data-dependent constant in our analysis. Then, using the assumptions and value of $L$ from Theorem 4.2,Algorithm 2 solves the Vector Set Search Problem in query time

定理 4.3。假设每个哈希函数调用的运行时间为 $O\left( d\right)$，并且对于某个正阈值 $T$ 有 $\left| {\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h}}\right|  < T\forall i,t,h$，在我们的分析中，我们将其视为与数据相关的常数。那么，使用定理 4.2 中的假设和 $L$ 的值，算法 2 在查询时间内解决了向量集搜索问题

$$
O\left( {{m}_{q}\log \left( {N{m}_{q}m/\delta }\right) d + {m}_{q}N\log \left( {N{m}_{q}m/\delta }\right) }\right) 
$$

Proof. If we suppose that each call to the hash function ${f}_{t}$ is $O\left( d\right)$ ,the runtime of the algorithm is

证明。如果我们假设对哈希函数 ${f}_{t}$ 的每次调用是 $O\left( d\right)$，则该算法的运行时间为

$$
O\left( {{nLd} + \mathop{\sum }\limits_{{i = 0}}^{{n - 1}}\mathop{\sum }\limits_{{k = 0}}^{{N - 1}}\mathop{\sum }\limits_{{t = 0}}^{{L - 1}}\left| {M}_{k,t,{f}_{t}\left( {q}_{j}\right) }\right| }\right) 
$$

To bound this quantity,we use the sparsity assumption we made in the theorem: no set ${S}_{i}$ contains too many elements that are very similar to a single query vector ${q}_{j}$ . Formally,we require that

为了对这个量进行界定，我们使用定理中所做的稀疏性假设：没有集合 ${S}_{i}$ 包含太多与单个查询向量 ${q}_{j}$ 非常相似的元素。形式上，我们要求

$$
\left| {\mathcal{D}{\left\lbrack  i\right\rbrack  }_{t,h}}\right|  < T\;\forall i,t,h
$$

for some positive threshold $T$ . With this assumption,the runtime of Algorithm 2 is

对于某个正阈值 $T$。在这个假设下，算法 2 的运行时间为

$$
O\left( {{m}_{q}{Ld} + {m}_{q}{NLT}}\right) 
$$

Plugging in the $L$ we found in our previous theorem,and treating $T$ as data dependent constant,we have that the runtime of Algorithm 2 is

代入我们在前一个定理中找到的 $L$，并将 $T$ 视为与数据相关的常数，我们得到算法 2 的运行时间为

$$
O\left( {{m}_{q}\log \left( {N{m}_{q}m/\delta }\right) d + {m}_{q}N\log \left( {N{m}_{q}m/\delta }\right) }\right) 
$$

which completes the proof.

这就完成了证明。

## B Hyperparameter Settings

## B 超参数设置

Settings for DESSERT corresponding to the first row in the left part of Table 2, where DESSERT was optimized for returning 10 documents in a low latency part of the Pareto frontier:

对应于表 2 左半部分第一行的 DESSERT 设置，其中 DESSERT 针对在帕累托前沿的低延迟部分返回 10 个文档进行了优化：

---

hashes_per_table (C) = 7

	num_tables (L) = 32

	filter_k = 4096

	filter_probe = 1

---

Settings for DESSERT corresponding to the second row in the left part of Table 2, where DESSERT was optimized for returning 10 documents in a high latency part of the Pareto frontier:

对应于表 2 左半部分第二行的 DESSERT 设置，其中 DESSERT 针对在帕累托前沿的高延迟部分返回 10 个文档进行了优化：

---

hashes_per_table (C) = 7

num_tables (L) = 64

	filter_k = 4096

	filter_probe = 2

---

Settings for DESSERT corresponding to the first row in the right part of Table 2, where DESSERT was optimized for returning 1000 documents in a low latency part of the Pareto frontier:

对应于表 2 右半部分第一行的 DESSERT 设置，其中 DESSERT 针对在帕累托前沿的低延迟部分返回 1000 个文档进行了优化：

---

hashes_per_table $\left( C\right)  = 6$

num_tables (L) = 32

filter_k = 8192

filter_probe = 4

---

Settings for DESSERT corresponding to the second row in the right part of Table 2, where DESSERT was optimized for returning 1000 documents in a high latency part of the Pareto frontier:

对应于表 2 右半部分第二行的 DESSERT 设置，其中 DESSERT 针对在帕累托前沿的高延迟部分返回 1000 个文档进行了优化：

---

hashes_per_table (C) = 7

num_tables (L) = 32

	filter_k = 16384

	filter_probe = 4

---

Intuitively, these parameter settings make sense: increase the initial filtering size and the number of total hashes for higher accuracy, and increase the initial filtering size for returning more documents (1000 vs. 10).

直观地说，这些参数设置是合理的：为了提高准确性，增加初始过滤大小和总哈希数；为了返回更多文档，增加初始过滤大小（1000 与 10 对比）。

## C Background on Locality-Sensitive Hashing and Inverted Indices for Similarity Search

## C 用于相似性搜索的局部敏感哈希和倒排索引背景

Here, we offer a refresher on using locality-sensitive hashing for similarity search with a basic inverted index structure.

在这里，我们简要回顾一下如何使用局部敏感哈希结合基本的倒排索引结构进行相似性搜索。

Consider a set of distinct vectors $X = \left\{  {{x}_{1},\ldots ,{x}_{N}}\right\}$ where each ${x}_{i} \in  {\mathbb{R}}^{d}$ . A hash function $h$ with a range $m$ maps each ${x}_{i}$ to an integer in the range $\left\lbrack  {1,m}\right\rbrack$ . Two vectors ${x}_{i}$ and ${x}_{j}$ are said to "collide" when $h\left( {x}_{i}\right)  = h\left( {x}_{j}\right)$ .

考虑一组不同的向量 $X = \left\{  {{x}_{1},\ldots ,{x}_{N}}\right\}$，其中每个 ${x}_{i} \in  {\mathbb{R}}^{d}$ 。一个哈希函数 $h$，其值域为 $m$，将每个 ${x}_{i}$ 映射到范围 $\left\lbrack  {1,m}\right\rbrack$ 内的一个整数。当 $h\left( {x}_{i}\right)  = h\left( {x}_{j}\right)$ 时，称两个向量 ${x}_{i}$ 和 ${x}_{j}$ “发生碰撞”。

As a warmup,we will first consider the case of a hash function $h$ drawn from a set of universal hash functions $H$ . Under such a function,if $i \neq  j,p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \frac{1}{m}$ ; such families exist in practice [7]. We can build an inverted index using this hash function by mapping each hash value in $\left\lbrack  {1,m}\right\rbrack$ to the set of vectors ${X}_{v}$ that have this hash value. Then,given a new vector $y$ ,we can query the inverted index with $v = h\left( y\right)$ . We can see that $y \in  X$ iff $y \in  {X}_{v}$ . Such an index is in a sense solving a search problem, if we only care about finding exact duplicates of our search query. Additionally, we can solve the nearest neighbor problem with this index in time $O\left( N\right)$ ,by going to every bucket and checking the distance of a query against every vector in the bucket.

作为热身，我们首先考虑从一组通用哈希函数 $H$ 中选取的哈希函数 $h$ 的情况。在这样的函数下，如果 $i \neq  j,p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \frac{1}{m}$ ；实际上存在这样的函数族 [7]。我们可以使用这个哈希函数构建一个倒排索引，将 $\left\lbrack  {1,m}\right\rbrack$ 中的每个哈希值映射到具有该哈希值的向量集合 ${X}_{v}$。然后，给定一个新向量 $y$，我们可以用 $v = h\left( y\right)$ 查询倒排索引。我们可以看到，当且仅当 $y \in  {X}_{v}$ 时，$y \in  X$ 成立。从某种意义上说，如果我们只关心找到搜索查询的精确副本，这样的索引就在解决一个搜索问题。此外，我们可以在时间 $O\left( N\right)$ 内使用这个索引解决最近邻问题，方法是遍历每个桶并检查查询与桶中每个向量的距离。

Now,in a similar way as in the universal case,let $h$ to be drawn from a family of locality-sensitive hash functions $H$ . At a high level,instead of mapping vectors uniformly to $\left\lbrack  {1,m}\right\rbrack  ,h$ maps vectors that are close together to the same hash value more often. Formally, if we define a "close" threshold ${r}_{1}$ ,a "far" threshold ${r}_{2}$ ,a "close" probability ${p}_{1}$ ,and a "far" probability ${p}_{2}$ ,with ${p}_{1} > {p}_{2}$ and ${r}_{1} < {r}_{2}$ , then we say $H$ is $\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$ -sensitive if

现在，与通用情况类似，让 $h$ 从一族局部敏感哈希函数 $H$ 中选取。从高层次来看，$\left\lbrack  {1,m}\right\rbrack  ,h$ 不是将向量均匀地映射，而是更频繁地将彼此接近的向量映射到相同的哈希值。形式上，如果我们定义一个“接近”阈值 ${r}_{1}$、一个“远离”阈值 ${r}_{2}$、一个“接近”概率 ${p}_{1}$ 和一个“远离”概率 ${p}_{2}$，且 ${p}_{1} > {p}_{2}$ 和 ${r}_{1} < {r}_{2}$，那么当满足以下条件时，我们称 $H$ 是 $\left( {{r}_{1},{r}_{2},{p}_{1},{p}_{2}}\right)$ -敏感的：

$$
d\left( {x,y}\right)  < {r}_{1} \Rightarrow  \Pr \left( {h\left( x\right)  = h\left( y\right) }\right)  > {p}_{1}
$$

$$
d\left( {x,y}\right)  > {r}_{2} \Rightarrow  \Pr \left( {h\left( x\right)  = h\left( y\right) }\right)  < {p}_{2}
$$

where $d$ is a distance metric. See [17] for the origin of locality-sensitive hashing and this definition. Intuitively,if we build an inverted index using $h$ in the same way as before,it now seems we have a strategy to solve the (approximate) nearest neighbor problem more efficiently: given a query $q$ ,only search for nearest neighbors in the bucket $h\left( q\right)$ ,since each of these points $x$ likely has $d\left( {q,x}\right)  < {r}_{1}$ . However, this strategy has a problem: with our definition, even a close neighbor might not be a collision with probability $\left( {1 - {p}_{1}}\right)$ . Thus,we can repeat our inverted index $L$ times with different ${h}_{i}$ drawn independently from $H$ ,such that our probability of not finding a close neighbor in any bucket is ${\left( 1 - {p}_{1}\right) }^{L}$ . FALCONN [2] is an LSH inverted index algorithm that uses this basic idea,along with concatenation and probing tricks, to achieve an asymptotically optimal (and data-dependent sub-linear) runtime; see the paper and associated code repository for more details.

其中 $d$ 是一个距离度量。有关局部敏感哈希的起源和这个定义，请参阅 [17]。直观地说，如果我们像之前一样使用 $h$ 构建一个倒排索引，现在我们似乎有了一种更有效地解决（近似）最近邻问题的策略：给定一个查询 $q$，只在桶 $h\left( q\right)$ 中搜索最近邻，因为这些点 $x$ 中的每一个可能都满足 $d\left( {q,x}\right)  < {r}_{1}$。然而，这个策略有一个问题：根据我们的定义，即使是接近的邻居也可能以概率 $\left( {1 - {p}_{1}}\right)$ 不发生碰撞。因此，我们可以使用从 $H$ 中独立选取的不同 ${h}_{i}$ 重复构建我们的倒排索引 $L$ 次，使得我们在任何桶中都找不到接近邻居的概率为 ${\left( 1 - {p}_{1}\right) }^{L}$。FALCONN [2] 是一种局部敏感哈希倒排索引算法，它利用了这个基本思想，以及连接和探测技巧，以实现渐近最优（且与数据相关的次线性）运行时间；更多细节请参阅论文和相关代码库。

One final note is that in practice, most LSH families satisfy a much stronger condition than the above. Consider a similarity function $\operatorname{sim} \in  \left\lbrack  {0,1}\right\rbrack$ ,where $s\left( {x,y}\right)  = 1 \Rightarrow  x = y$ . As $x$ and $y$ get more dissimilar (e.g. their distance increases according to some distance metric), $s\left( {x,y}\right)$ decreases. For most LSH families, there exists an explicit similarity function that their collision probability satisfies, such that $p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}\left( {x,y}\right)$ . Such LSH families exist for most common similarity functions,including cosine similarity (signed random projections) [8],Euclidean similarity $(p$ -stable projections) [11], and Jaccard similarity (minhash or simhash) [6]. Following [9, 10, 13, 24, 27, 31], in our work, we use LSH families with this explicit similarity description to provide tight analyses and strong guarantees for similarity-search algorithms.

最后需要说明的是，在实践中，大多数局部敏感哈希（LSH）族满足的条件比上述条件要强得多。考虑一个相似度函数$\operatorname{sim} \in  \left\lbrack  {0,1}\right\rbrack$，其中$s\left( {x,y}\right)  = 1 \Rightarrow  x = y$。随着$x$和$y$的差异增大（例如，根据某种距离度量，它们之间的距离增加），$s\left( {x,y}\right)$会减小。对于大多数LSH族，存在一个明确的相似度函数，其碰撞概率满足该函数，即$p\left( {h\left( x\right)  = h\left( y\right) }\right)  = \operatorname{sim}\left( {x,y}\right)$。对于大多数常见的相似度函数，都存在这样的LSH族，包括余弦相似度（带符号随机投影）[8]、欧几里得相似度（$(p$ - 稳定投影）[11]和杰卡德相似度（最小哈希或相似哈希）[6]。遵循文献[9, 10, 13, 24, 27, 31]，在我们的工作中，我们使用具有这种明确相似度描述的LSH族，为相似度搜索算法提供精确的分析和有力的保证。