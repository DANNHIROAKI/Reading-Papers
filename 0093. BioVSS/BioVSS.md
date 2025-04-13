# Approximate Vector Set Search: A Bio-Inspired Approach for High-Dimensional Spaces

# 近似向量集搜索：一种高维空间的生物启发式方法

Yiqi ${\mathrm{{Li}}}^{ \dagger  }$ ,Sheng Wang ${}^{\dagger  * }$ ,Zhiyu Chen ${}^{ \ddagger  }$ ,Shangfeng Chen ${}^{ \dagger  }$ ,and Zhiyong Peng ${}^{\dagger  \ddagger   * }$

易奇 ${\mathrm{{Li}}}^{ \dagger  }$ ，王盛 ${}^{\dagger  * }$ ，陈志宇 ${}^{ \ddagger  }$ ，陈上峰 ${}^{ \dagger  }$ ，彭智勇 ${}^{\dagger  \ddagger   * }$

${}^{ \dagger  }$ School of Computer Science,Wuhan University

${}^{ \dagger  }$ 武汉大学计算机学院

*Amazon.com, Inc. $\;{}^{§}$ Big Data Institute,Wuhan University

*亚马逊公司 $\;{}^{§}$ 武汉大学大数据研究院

[bruceprofession, swangcs, brucechen, peng]@whu.edu.cn, zhiyuche@amazon.com

[bruceprofession, swangcs, brucechen, peng]@whu.edu.cn, zhiyuche@amazon.com

Abstract-Vector set search, an underexplored similarity search paradigm, aims to find vector sets similar to a query set. This search paradigm leverages the inherent structural alignment between sets and real-world entities to model more fine-grained and consistent relationships for diverse applications. This task, however, faces more severe efficiency challenges than traditional single-vector search due to the combinatorial explosion of pairings in set-to-set comparisons. In this work, we aim to address the efficiency challenges posed by the combinatorial explosion in vector set search, as well as the curse of dimensionality inherited from single-vector search. To tackle these challenges, we present an efficient algorithm for vector set search, BioVSS (Bio-inspired Vector Set Search). BioVSS simulates the fly olfactory circuit to quantize vectors into sparse binary codes and then designs an index based on the set membership property of the Bloom filter. The quantization and indexing strategy enables BioVSS to efficiently perform vector set search by pruning the search space. Experimental results demonstrate over 50 times speedup compared to linear scanning on million-scale datasets while maintaining a high recall rate of up to ${98.9}\%$ ,making it an efficient solution for vector set search.

摘要——向量集搜索是一种尚未充分探索的相似性搜索范式，旨在找到与查询集相似的向量集。这种搜索范式利用集合与现实世界实体之间的固有结构对齐，为各种应用建模更细粒度和一致的关系。然而，由于集合间比较中配对的组合爆炸，这项任务比传统的单向量搜索面临更严峻的效率挑战。在这项工作中，我们旨在解决向量集搜索中组合爆炸带来的效率挑战，以及从单向量搜索继承而来的维度灾难问题。为应对这些挑战，我们提出了一种用于向量集搜索的高效算法BioVSS（生物启发式向量集搜索）。BioVSS模拟果蝇嗅觉回路，将向量量化为稀疏二进制代码，然后基于布隆过滤器的集合成员属性设计索引。量化和索引策略使BioVSS能够通过修剪搜索空间高效地执行向量集搜索。实验结果表明，与在百万级数据集上的线性扫描相比，速度提高了50倍以上，同时保持了高达 ${98.9}\%$ 的高召回率，使其成为向量集搜索的一种高效解决方案。

## I. INTRODUCTION

## I. 引言

Vector search is a fundamental computational problem in various domains such as information retrieval [18], recommender systems [31], and computer vision [21]. The majority of vector search methods, tailored for querying by a single vector, address the curse of dimensionality [19] and the sparsity issue in high-dimensional vector space [50]. In this paper, we study a novel and challenging problem called Vector Set Search. Given a query vector set $\mathbf{Q}$ ,the objective is to find the top- $k$ most relevant vector sets from a vector set database $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ . This problem transcends the boundaries of single-vector search [12] by incorporating matching and aggregation between vectors within each set. This introduces additional computational complexity, for which there are currently no effective solutions.

向量搜索是信息检索 [18]、推荐系统 [31] 和计算机视觉 [21] 等各个领域中的一个基本计算问题。大多数向量搜索方法是为单向量查询量身定制的，旨在解决维度灾难 [19] 和高维向量空间中的稀疏性问题 [50]。在本文中，我们研究了一个名为向量集搜索的新颖且具有挑战性的问题。给定一个查询向量集 $\mathbf{Q}$ ，目标是从向量集数据库 $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ 中找到前 $k$ 个最相关的向量集。这个问题超越了单向量搜索的边界 [12]，因为它包含了每个集合内向量之间的匹配和聚合。这引入了额外的计算复杂性，目前尚无有效的解决方案。

Applications. Vector set search has many applications due to its structural alignment with real-world entities, for example:

应用。由于向量集搜索与现实世界实体的结构对齐，它有许多应用，例如：

- Academic Entity Discovery. Academic entity discovery facilitates effective exploration of the rapidly growing literature. By representing academic entities as vector sets, such as a paper's profile [48] consisting of vectors representing its citations, or an author's profile [38] as a set of vectors representing their publications, vector set search enables effective identification of relevant research and tracking of academic trends.

- 学术实体发现。学术实体发现有助于有效探索快速增长的文献。通过将学术实体表示为向量集，例如一篇论文的概况 [48] 由表示其引用的向量组成，或者一位作者的概况 [38] 作为表示其出版物的向量集，向量集搜索能够有效识别相关研究并跟踪学术趋势。

<!-- Media -->

<!-- figureText: Scholar Profiles Vector Set Database Vec. ID Vector 0.2 0.3 0.7 2 0.2 0.2 0.1 ... 0.2 0.2 0.5 ... 2 0.2 0.1 0.9 3 0.2 0.5 0.7 Vector Set Search Compute Distance Pairwise Distance Aggregation 3 Vec 4.3 CI 3.0 ① Vectorizing SETID ① ② Query Set Vector Set (Set ID:②) Database Vec. - Get Top-k Scholars -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_0.jpg?x=916&y=563&w=737&h=496&r=0"/>

Figure 1: An Exemplar Data Flow of Vector Set Search

图1：向量集搜索的示例数据流

<!-- Media -->

- Multi-Modal Query Answering. Multi-modal query answering systems leverage vector set search to enhance accuracy. As demonstrated in [43], this approach encodes multi-modal data (e.g., images, text, and audio) into vector sets using a multi-modal encoder [26]. Vector set search then enables accurate queries across multi-modal representations, improving the overall performance of multi-modal query answering systems. - Multi-Vector Ranking for Recommender. Multi-vector ranking enhances recommendation systems by representing users with vector sets. This approach, as demonstrated in [30], represents users as vector sets to capture various aspects of user profiles, including browsing history, search queries, click records, etc. The ranking process compares these vector sets, enabling flexible matching between diverse user interests.

- 多模态查询回答。多模态查询回答系统利用向量集搜索来提高准确性。如 [43] 所示，这种方法使用多模态编码器 [26] 将多模态数据（如图像、文本和音频）编码为向量集。然后，向量集搜索能够跨多模态表示进行准确查询，提高多模态查询回答系统的整体性能。 - 推荐系统的多向量排序。多向量排序通过用向量集表示用户来增强推荐系统。如 [30] 所示，这种方法将用户表示为向量集，以捕捉用户概况的各个方面，包括浏览历史、搜索查询、点击记录等。排序过程比较这些向量集，实现不同用户兴趣之间的灵活匹配。

Let us delve into scholar search, an exemplar of vector set search that serves as a key focus in our experimental study.

让我们深入研究学者搜索，这是向量集搜索的一个示例，也是我们实验研究的重点。

Example 1: Figure 1 illustrates an exemplar data flow of vector set search applied to scholar search [38], preceded by vector set database preparation. The vector set database construction involves representing scholar profiles as vector sets derived from their publications [9], creating the database of scholars. The vector set search then operates on this database, comprising two key stages: distance computation and neighbor identification. The distance computation stage involves calculating the pairwise distances (such as Euclidean distance) between the query vector set and another set in the database, followed by aggregating (such as combining max and min) into set-to-set distances. The neighbor identification phase then determines the top-k nearest scholars based on these set distances for the query. This process enables the discovery of similar scholars for subsequent research activities. Motivation. The example above reveals a key insight: the structure of vector sets aligns well with real-world entities, enabling effective applications. Recent advancements in embedding models [9] that generate high-dimensional, semantically rich vectors have further expanded the applicability. However, the computational demands of high-dimensional vectors and the combinatorial explosion of pairwise comparisons [3] complicate the development of efficient search.

示例1：图1展示了应用于学者搜索的向量集搜索的一个示例数据流[38]，该过程之前需要进行向量集数据库的准备工作。向量集数据库的构建包括将学者简介表示为从他们的出版物中提取的向量集[9]，从而创建学者数据库。然后，向量集搜索在这个数据库上进行，包括两个关键阶段：距离计算和邻居识别。距离计算阶段涉及计算查询向量集与数据库中另一个集合之间的成对距离（如欧几里得距离），然后将这些距离聚合（如结合最大值和最小值）为集合到集合的距离。邻居识别阶段则根据这些集合距离为查询确定前k个最近的学者。这个过程能够发现相似的学者，以便进行后续的研究活动。动机。上述示例揭示了一个关键见解：向量集的结构与现实世界的实体非常契合，从而能够实现有效的应用。最近在生成高维、语义丰富向量的嵌入模型[9]方面的进展进一步扩大了其适用性。然而，高维向量的计算需求以及成对比较的组合爆炸[3]使得高效搜索的开发变得复杂。

---

<!-- Footnote -->

*Sheng Wang and Zhiyong Peng are the corresponding authors.

*盛王和彭志勇是通讯作者。

${}^{ \ddagger  }$ Work does not relate to position at Amazon.

${}^{ \ddagger  }$ 该工作与在亚马逊的职位无关。

<!-- Footnote -->

---

Recent relevant studies such as DESSERT [11] utilize hash tables to accelerate specific non-metric distance computation. However, the fundamental requirements for vector set similarity assessment - precise differentiation capability and consistent measurement across comparisons - demand careful consideration of distance metrics (detailed analysis in Appendix VII-A). The Hausdorff [16], [17] distance naturally aligns with these requirements, providing mathematically rigorous set-to-set similarity measurement while maintaining essential symmetric properties for reliable similarity assessment.

最近的相关研究，如DESSERT[11]，利用哈希表来加速特定的非度量距离计算。然而，向量集相似性评估的基本要求——精确的区分能力和跨比较的一致测量——需要仔细考虑距离度量（附录VII - A中有详细分析）。豪斯多夫[16]、[17]距离自然地符合这些要求，它提供了数学上严格的集合到集合的相似性测量，同时保持了可靠相似性评估所需的基本对称属性。

Motivated by these fundamental requirements for robust set comparison, we recognize that metric distances are crucial for consistent evaluations. The Hausdorff distance, a well-studied metric in set theory [16], [17], offers native support for set distance measurement. However, it requires complex pairwise distance calculations, making it computationally intensive. This computational burden underscores the urgent need for more efficient vector set search methods.

受这些对稳健集合比较的基本要求的启发，我们认识到度量距离对于一致评估至关重要。豪斯多夫距离是集合论中经过充分研究的一种度量[16]、[17]，它为集合距离测量提供了天然支持。然而，它需要进行复杂的成对距离计算，这使得计算量很大。这种计算负担凸显了对更高效的向量集搜索方法的迫切需求。

Challenges. To provide efficient vector set search in high-dimensional spaces using Hausdorff distance, several key challenges arise. The first challenge is the curse of dimensionality [19]. Recent methods address this challenge through hash table construction [11] and the transformation of vector set to single vector [23]. However, both methods are limited to specific nonmetric distances, constraining their applicability in Hausdorff distance. The second challenge is the Aggregation complexity. Existing Hausdorff distance search algorithms [44], [1], [32] leverage geometric information to decrease the number of aggregation operations. However, geometric information loses effectiveness in high dimensions.

挑战。要使用豪斯多夫距离在高维空间中提供高效的向量集搜索，会出现几个关键挑战。第一个挑战是维度诅咒[19]。最近的方法通过构建哈希表[11]和将向量集转换为单个向量[23]来应对这一挑战。然而，这两种方法都仅限于特定的非度量距离，限制了它们在豪斯多夫距离中的适用性。第二个挑战是聚合复杂性。现有的豪斯多夫距离搜索算法[44]、[1]、[32]利用几何信息来减少聚合操作的数量。然而，几何信息在高维空间中会失去有效性。

Contributions. To overcome the above two challenges, we propose an algorithm to support an approximate top- $k$ vector set search. Specifically, our work draws inspiration from a locality-sensitive hashing [8], [37] inspired by the olfactory circuit [27]. We simulate the olfactory circuit to quantize vectors into sparse binary codes, addressing the curse of dimensionality. Based on this, an index is designed utilizing the set membership property of Bloom filter, which reduces the search space and decreases the number of aggregation operations. Overall, our contributions are summarized as follows:

贡献。为了克服上述两个挑战，我们提出了一种算法来支持近似的前$k$向量集搜索。具体来说，我们的工作受到了受嗅觉回路启发的局部敏感哈希[8]、[37]的启发[27]。我们模拟嗅觉回路将向量量化为稀疏二进制代码，以应对维度诅咒。在此基础上，利用布隆过滤器的集合成员属性设计了一个索引，该索引减少了搜索空间并减少了聚合操作的数量。总体而言，我们的贡献总结如下：

- We define the first approximate vector set search problem in high-dimensional spaces using Hausdorff distance, which is a native set metric distance (see Section III).

- 我们定义了在高维空间中使用豪斯多夫距离（一种天然的集合度量距离）的第一个近似向量集搜索问题（见第三节）。

- We propose BioVSS, which uses locality-sensitive property to accelerate vector set search, and provide a comprehensive theoretical analysis with proofs validating the correctness of the proposed method (see Section IV).

- 我们提出了BioVSS，它利用局部敏感属性来加速向量集搜索，并提供了全面的理论分析和证明，验证了所提出方法的正确性（见第四节）。

- We present an enhanced version of BioVSS called BioVSS++, which employs a dual-layer cascaded filter composed of inverted index and vector set sketches to reduce unnecessary scans (see Section V).

- 我们提出了BioVSS的增强版本BioVSS++，它采用了由倒排索引和向量集草图组成的双层级联过滤器来减少不必要的扫描（见第五节）。

- We conduct extensive experiments showing our method achieves over 50 times speedup compared to linear scanning on million-scale datasets while maintaining a recall of up to 98.9%, validating its efficiency (see Section VI).

- 我们进行了广泛的实验，结果表明，与在百万级数据集上的线性扫描相比，我们的方法实现了超过50倍的加速，同时保持了高达98.9%的召回率，验证了其效率（见第六节）。

## II. RELATED WORK

## 二、相关工作

Single-Vector Search. Single-vector search [49] is increasingly becoming a unified paradigm with the development of embedding models. However, efficiency remains a critical challenge for practical application. Various approximate search algorithms [25] have been developed to perform efficient single-vector search. These algorithms fall into three main categories: locality-sensitive hashing methods [37], [37], [47], [51], graph-based methods [28], and space partition methods [24], [15]. Locality-sensitive hashing like FlyHash [37] maps similar vectors to the same hash buckets. Graph-based approaches such as HNSW [28] construct navigable graphs for efficient search. Space partitioning methods, such as IVFFLAT [49], [10], IndexIVFPQ [14], [20], [10] and IVFScalarQuantizer [10], utilize k-means [46] clustering to partition the vector space and build inverted file indices for fast search. However, existing approaches focus solely on single-vector rather than set-based queries.

单向量搜索。随着嵌入模型的发展，单向量搜索 [49] 正日益成为一种统一的范式。然而，效率仍然是实际应用中的一个关键挑战。人们已经开发出各种近似搜索算法 [25] 来进行高效的单向量搜索。这些算法主要分为三类：局部敏感哈希方法 [37]、[37]、[47]、[51]，基于图的方法 [28]，以及空间划分方法 [24]、[15]。像 FlyHash [37] 这样的局部敏感哈希方法将相似的向量映射到相同的哈希桶中。像 HNSW [28] 这样的基于图的方法构建可导航图以进行高效搜索。空间划分方法，如 IVFFLAT [49]、[10]，IndexIVFPQ [14]、[20]、[10] 和 IVFScalarQuantizer [10]，利用 k - 均值 [46] 聚类来划分向量空间并构建倒排文件索引以实现快速搜索。然而，现有的方法仅专注于单向量查询，而非基于集合的查询。

Vector Set Search. The problem of vector set search remains relatively unexplored and has not received sufficient attention in the research community. Early research focused on low-dimensional vector space, treating vector set search as a set matching problem often solved with the Kuhn-Munkres algorithm [22]. In geospatial research, trajectories were typically represented as point collections and measured with the Hausdorff distance [2], [44], [45]. Researchers developed various optimizations, such as incorporating road network information [36], using branch-and-bound techniques [32], and employing estimated Hausdorff distances for accelerated computations [1]. However, these approaches, tailored for low-dimensional spaces, do not extend well to high-dimensional spaces. Recent advancements in embedding models enable high-dimensional vectors to carry rich semantic information. By transforming vector sets into single vectors [23], existing high-dimensional single-vector approximate search algorithms can be leveraged to speed up the computational process. However, this transformation relies on a specific distance metric proposed by the work. While the method proposed in [11] is an efficient algorithm that accelerates the vector set search process through hash table construction, it has limitations in terms of measures. Specifically, this approach lacks theoretical support for the Hausdorff distance metric, particularly when the max function is used as the outer aggregation strategy. Moreover, these methods lack effective support for Hausdorff distance. To address this gap, we focus on high-dimensional vector set search using Hausdorff distance.

向量集搜索。向量集搜索问题相对未被充分探索，在研究界也未得到足够的关注。早期的研究集中在低维向量空间，将向量集搜索视为一个集合匹配问题，通常使用库恩 - 蒙克斯算法 [22] 来解决。在地理空间研究中，轨迹通常被表示为点集，并使用豪斯多夫距离 [2]、[44]、[45] 进行度量。研究人员开发了各种优化方法，如结合道路网络信息 [36]、使用分支限界技术 [32] 以及采用估计的豪斯多夫距离来加速计算 [1]。然而，这些为低维空间量身定制的方法在高维空间中效果不佳。嵌入模型的最新进展使高维向量能够携带丰富的语义信息。通过将向量集转换为单向量 [23]，可以利用现有的高维单向量近似搜索算法来加速计算过程。然而，这种转换依赖于该工作提出的特定距离度量。虽然文献 [11] 中提出的方法是一种通过构建哈希表来加速向量集搜索过程的高效算法，但它在度量方面存在局限性。具体而言，这种方法缺乏对豪斯多夫距离度量的理论支持，尤其是当使用最大值函数作为外部聚合策略时。此外，这些方法缺乏对豪斯多夫距离的有效支持。为了弥补这一差距，我们专注于使用豪斯多夫距离进行高维向量集搜索。

Bio-Inspired Hashing & Bloom Filter. Olfactory systems in various species show remarkable precision in identifying odorants [6]. For instance, Mice (rodent animals) analyze neural responses in the olfactory bulb to discriminate odors [35]. Similarly, Zebrafish (teleost animals) detect water-dissolved amino acids for olfactory signaling [33]. Furthermore, the olfaction of fly (insect animals) research is the most in-depth and has led to the development of a sparse locality-sensitive hashing function [8], [37] through computational simulations. Fly-inspired hashing [8], a novel locality-sensitive hashing function for single vector search, draws inspiration from the fly olfactory circuit. In the olfactory circuit, only a few neurons respond to specific odor molecules, creating sparse activation patterns that enhance odor recognition efficiency. BioHash [37] improves accuracy by learning the intrinsic patterns of data to set the connection strengths of projection neurons in the fly olfactory circuit [42]. Similarly, a Bloom filter maps set elements into a sparse array using hash functions, enabling efficient set storage. The binary Bloom filter [42] includes a binary array, where each position is either 0 or 1 . Additionally, the count Bloom filter [5] offers finer granularity through counting. This work leverages the locality-sensitive property of the fly olfactory circuit and the set-storing capabilities of Bloom filters. Meanwhile, we exploit their structural similarities to construct an efficient index.

受生物启发的哈希与布隆过滤器。各种物种的嗅觉系统在识别气味物质方面表现出显著的精度 [6]。例如，老鼠（啮齿动物）通过分析嗅球中的神经反应来区分气味 [35]。同样，斑马鱼（硬骨鱼类动物）检测水中溶解的氨基酸以进行嗅觉信号传递 [33]。此外，对苍蝇（昆虫类动物）嗅觉的研究最为深入，并通过计算模拟开发出了一种稀疏局部敏感哈希函数 [8]、[37]。受苍蝇启发的哈希 [8] 是一种用于单向量搜索的新型局部敏感哈希函数，它的灵感来自于苍蝇的嗅觉回路。在嗅觉回路中，只有少数神经元对特定的气味分子做出反应，形成稀疏的激活模式，从而提高了气味识别效率。BioHash [37] 通过学习数据的内在模式来设置苍蝇嗅觉回路中投射神经元的连接强度，从而提高了准确性 [42]。同样，布隆过滤器使用哈希函数将集合元素映射到一个稀疏数组中，实现了高效的集合存储。二进制布隆过滤器 [42] 包含一个二进制数组，其中每个位置要么是 0 要么是 1。此外，计数布隆过滤器 [5] 通过计数提供了更精细的粒度。这项工作利用了苍蝇嗅觉回路的局部敏感特性和布隆过滤器的集合存储能力。同时，我们利用它们的结构相似性来构建一个高效的索引。

## III. DEFINITIONS & PRELIMINARIES

## 三、定义与预备知识

Definition 1 (Vector). A vector $\mathbf{v} = \left( {{v}_{1},{v}_{2},\ldots ,{v}_{d}}\right)$ is a tuple of real numbers,in which ${v}_{i} \in  \mathbb{R}$ and $d \in  {\mathbb{N}}^{ + }$ represents the dimensionality of the vector.

定义 1（向量）。向量 $\mathbf{v} = \left( {{v}_{1},{v}_{2},\ldots ,{v}_{d}}\right)$ 是一个实数元组，其中 ${v}_{i} \in  \mathbb{R}$ 且 $d \in  {\mathbb{N}}^{ + }$ 表示向量的维度。

Definition 2 (Vector Set). A vector set $\mathbf{V} = \left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2},\ldots ,{\mathbf{v}}_{m}}\right\}$ contains a set of vectors,where $m \in  {\mathbb{N}}^{ + }$ is number of vectors.

定义 2（向量集）。向量集 $\mathbf{V} = \left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2},\ldots ,{\mathbf{v}}_{m}}\right\}$ 包含一组向量，其中 $m \in  {\mathbb{N}}^{ + }$ 是向量的数量。

Definition 3 (Vector Set Database). A vector set database $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ is a collection of vector sets,where each ${\mathbf{V}}_{i}$ is a vector set and $n \in  {\mathbb{N}}^{ + }$ is the number of vector sets.

定义3（向量集数据库）。向量集数据库 $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ 是向量集的集合，其中每个 ${\mathbf{V}}_{i}$ 是一个向量集， $n \in  {\mathbb{N}}^{ + }$ 是向量集的数量。

Definition 4 (Hausdorff Distance). Given two vector sets $\mathbf{Q}$ and $\mathbf{V}$ ,the Hausdorff distance from $\mathbf{Q}$ to $\mathbf{V}$ is defined as:

定义4（豪斯多夫距离）。给定两个向量集 $\mathbf{Q}$ 和 $\mathbf{V}$ ，从 $\mathbf{Q}$ 到 $\mathbf{V}$ 的豪斯多夫距离定义为：

$\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)  = \max \left( {\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\min }\limits_{{\mathbf{v} \in  \mathbf{V}}}\operatorname{dist}\left( {\mathbf{q},\mathbf{v}}\right) ,\mathop{\max }\limits_{{\mathbf{v} \in  \mathbf{V}}}\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\operatorname{dist}\left( {\mathbf{v},\mathbf{q}}\right) }\right) ,$

where dist $\left( {\mathbf{q},\mathbf{v}}\right)  = \parallel \mathbf{q} - \mathbf{v}{\parallel }_{2}$ is the Euclidean distance between vectors $\mathbf{q} \in  {\mathbb{R}}^{d}$ and $\mathbf{v} \in  {\mathbb{R}}^{d}$ .

其中 dist $\left( {\mathbf{q},\mathbf{v}}\right)  = \parallel \mathbf{q} - \mathbf{v}{\parallel }_{2}$ 是向量 $\mathbf{q} \in  {\mathbb{R}}^{d}$ 和 $\mathbf{v} \in  {\mathbb{R}}^{d}$ 之间的欧几里得距离。

To elucidate the computational process of Hausdorff distance, let us consider a concrete example.

为了阐明豪斯多夫距离的计算过程，让我们考虑一个具体的例子。

Example 2: The computation process of Hausdorff distance is illustrated in Figure 2 Given two finite vector sets $\mathbf{Q} =$ $\left\{  {{\mathbf{q}}_{1},{\mathbf{q}}_{2},{\mathbf{q}}_{3}}\right\}$ and $\mathbf{V} = \left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2}}\right\}$ in a high-dimensional spaces, the Hausdorff distance Haus(Q,V)is calculated through the following steps: 1) calculate the maximum of the minimum distances from each vector in $\mathbf{V}$ to all vectors in $\mathbf{Q}$ ,denoted (1) $\mathbf{V} \rightarrow  \mathbf{Q} : \max \left( {1,2}\right)  = 2$ ; (2) $\mathbf{Q} \rightarrow  \mathbf{V} : \max \left( {1,3,2}\right)  = 3$ ; (3) $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)  : \max \left( {2,3}\right)  = 3$ . as $\mathop{\max }\limits_{{\mathbf{v} \in  \mathbf{V}}}\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}d\left( {\mathbf{v},\mathbf{q}}\right)  = 2$ ; 2) calculate the maximum of the minimum distances from each vector in $\mathbf{Q}$ to all vectors in $\mathbf{V}$ ,expressed as $\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\min }\limits_{{\mathbf{v} \in  \mathbf{V}}}d\left( {\mathbf{q},\mathbf{v}}\right)  = 3$ ; 3) determine the Hausdorff distance by max aggregating the results from steps 1 and 2 to obtain $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)  = 3$ .

示例2：图2展示了豪斯多夫距离的计算过程。给定高维空间中的两个有限向量集 $\mathbf{Q} =$ $\left\{  {{\mathbf{q}}_{1},{\mathbf{q}}_{2},{\mathbf{q}}_{3}}\right\}$ 和 $\mathbf{V} = \left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2}}\right\}$ ，豪斯多夫距离 Haus(Q,V) 通过以下步骤计算：1) 计算 $\mathbf{V}$ 中的每个向量到 $\mathbf{Q}$ 中所有向量的最小距离的最大值，记为 (1) $\mathbf{V} \rightarrow  \mathbf{Q} : \max \left( {1,2}\right)  = 2$ ；(2) $\mathbf{Q} \rightarrow  \mathbf{V} : \max \left( {1,3,2}\right)  = 3$ ；(3) $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)  : \max \left( {2,3}\right)  = 3$ ，表示为 $\mathop{\max }\limits_{{\mathbf{v} \in  \mathbf{V}}}\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}d\left( {\mathbf{v},\mathbf{q}}\right)  = 2$ ；2) 计算 $\mathbf{Q}$ 中的每个向量到 $\mathbf{V}$ 中所有向量的最小距离的最大值，表示为 $\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\min }\limits_{{\mathbf{v} \in  \mathbf{V}}}d\left( {\mathbf{q},\mathbf{v}}\right)  = 3$ ；3) 通过对步骤1和步骤2的结果进行最大聚合来确定豪斯多夫距离，得到 $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)  = 3$ 。

<!-- Media -->

<!-- figureText: Distance Matrix Vector ${\mathrm{q}}_{2}$ ${\mathbf{q}}_{3}$ ${\mathbf{q}}_{1}$ ${\mathbf{q}}_{2}$ ${\mathbf{q}}_{3}$ min ${\mathbf{V}}_{\mathbf{1}}$ 1.0 3.1 4.3 ${\mathbf{V}}_{2}$ 4.2 3.0 2.0 乙 min 1 3 2 Set $\mathbf{Q}$ ${q}_{1}$ Vector Set $\mathbf{V}$ ${\mathbf{V}}_{\mathbf{1}}$ ${\mathbf{V}}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_2.jpg?x=929&y=225&w=707&h=209&r=0"/>

Figure 2: The Calculation Process of Hausdorff Distance

图2：豪斯多夫距离的计算过程

<!-- Media -->

Hausdorff distance calculation involves pairwise distance computations between vectors from two sets, followed by aggregation as described in Definition 4. The computational complexity for the Hausdorff distance is $O\left( {{m}^{2} \cdot  d}\right)$ ,where $m$ is the number of vectors per set and $d$ is the vector dimensionality. This quadratic dependence on $m$ makes exact computation prohibitive for large-scale datasets, especially as set sizes and dimensionality $d$ increase.

豪斯多夫距离的计算涉及计算两个集合中向量之间的两两距离，然后按照定义4进行聚合。豪斯多夫距离的计算复杂度为 $O\left( {{m}^{2} \cdot  d}\right)$ ，其中 $m$ 是每个集合中的向量数量， $d$ 是向量的维度。这种对 $m$ 的二次依赖使得对于大规模数据集进行精确计算变得不可行，尤其是当集合大小和维度 $d$ 增加时。

To reduce computational overhead, extensive single-vector search works have validated the efficacy of approximation techniques such as quantization [8], [37], [47] and index pruning [24], [15], [28], [14], [20], [10]. Inspired by these, we introduce the approximate top- $k$ vector set search problem.

为了减少计算开销，大量的单向量搜索工作已经验证了诸如量化 [8]、[37]、[47] 和索引剪枝 [24]、[15]、[28]、[14]、[20]、[10] 等近似技术的有效性。受这些启发，我们引入了近似前 $k$ 向量集搜索问题。

Definition 5 (Approximate Top- $k$ Vector Set Search). Given a vector set database $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ ,a query vector set $\mathbf{Q}$ and a Hausdorff distance function $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ,the approximate top- $k$ vector set search is the task of returning $\mathbf{R}$ with probability at least $1 - \delta$ :

定义 5（近似前 $k$ 向量集搜索）。给定一个向量集数据库 $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$、一个查询向量集 $\mathbf{Q}$ 和一个豪斯多夫距离函数 $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$，近似前 $k$ 向量集搜索的任务是至少以概率 $1 - \delta$ 返回 $\mathbf{R}$：

$$
\mathbf{R} = \left\{  {{\mathbf{V}}_{1}^{ \star  },{\mathbf{V}}_{2}^{ \star  },\ldots ,{\mathbf{V}}_{k}^{ \star  }}\right\}   = {\operatorname{argmin}}_{{\mathbf{V}}_{i}^{ \star  } \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},{\mathbf{V}}_{i}^{ \star  }}\right) ,
$$

where ${\operatorname{argmin}}_{{\mathbf{V}}_{i} \in  \mathbf{D}}^{k}$ selects $k$ sets ${\mathbf{V}}_{i}^{ \star  }$ minimizing Haus $\left( {\mathbf{Q},{\mathbf{V}}_{i}^{ \star  }}\right)$ and ${\mathbf{V}}_{k}^{ \star  }$ has the $k$ -th smallest distance from $\mathbf{Q}$ . The failure probability is denoted by $\delta  \in  \left\lbrack  {0,1}\right\rbrack$ .

其中 ${\operatorname{argmin}}_{{\mathbf{V}}_{i} \in  \mathbf{D}}^{k}$ 选择使豪斯 $\left( {\mathbf{Q},{\mathbf{V}}_{i}^{ \star  }}\right)$ 最小化的 $k$ 个集合 ${\mathbf{V}}_{i}^{ \star  }$，并且 ${\mathbf{V}}_{k}^{ \star  }$ 与 $\mathbf{Q}$ 的距离是第 $k$ 小的。失败概率用 $\delta  \in  \left\lbrack  {0,1}\right\rbrack$ 表示。

Approximate top- $k$ vector set search trades off speed and accuracy, permitting a small error in exchange for substantially improved search efficiency.

近似前 $k$ 向量集搜索在速度和准确性之间进行权衡，允许存在小的误差以换取显著提高的搜索效率。

## IV. Proposed BioVSS

## 四、提出的 BioVSS

We propose BioVSS, an efficient search algorithm designed to address the approximate top- $k$ vector set search problem. Drawing inspiration from the fly olfactory circuit, BioVSS leverages the locality-sensitive property of the olfactory circuit to enhance search efficiency. Section IV-B establishes theoretical foundations for locality-sensitive Hausdorff distance.

我们提出了 BioVSS，这是一种旨在解决近似前 $k$ 向量集搜索问题的高效搜索算法。受果蝇嗅觉回路的启发，BioVSS 利用嗅觉回路的局部敏感特性来提高搜索效率。第四节 B 部分为局部敏感豪斯多夫距离奠定了理论基础。

## A. Overview Algorithm

## A. 算法概述

BioVSS algorithm comprises two principal components: 1) the binary hash encoding for vector sets, and 2) the search execution in BioVSS.

BioVSS 算法主要由两个部分组成：1) 向量集的二进制哈希编码，以及 2) BioVSS 中的搜索执行。

<!-- Media -->

TABLE I: Summary of Major Notations

表一：主要符号总结

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>D</td><td>Vector Set Database</td></tr><tr><td>${\mathbf{D}}^{\mathbf{H}}$</td><td>Sparse Binary Codes of $\mathbf{D}$</td></tr><tr><td>T, Q</td><td>Vector Set of Target and Query</td></tr><tr><td>$n,m,{m}_{q}$</td><td>Cardinality of $\mathbf{D},\mathbf{T}$ ,and $\mathbf{Q}$</td></tr><tr><td>${L}_{wta} = L$</td><td>Number of Hash Functions / Winner-Takes-All</td></tr><tr><td>H</td><td>Hash Function (see Definition 7)</td></tr><tr><td>$k$</td><td>The Number of Results Returned</td></tr><tr><td>$\sigma \left( \cdot \right)$</td><td>Min-Max Similarity Function (see Lemma 1)</td></tr><tr><td>${S}_{ij}^{\alpha },{S}_{ij}^{\beta },\mathbf{S}$</td><td>(i,j)-th Entry of the Similarity Matrix $\mathbf{S}$</td></tr><tr><td>${s}_{\max },{s}_{\min }$</td><td>Max. and Min. Real Similarities btw. Vector Sets</td></tr><tr><td>$\widehat{\mathbf{S}},{\widehat{s}}_{\max },{\widehat{s}}_{\min }$</td><td>Estimated Values of $\mathbf{S},{s}_{\max }$ ,and ${s}_{\min }$</td></tr><tr><td>$\delta$</td><td>Vector Set Search Failure Probability</td></tr><tr><td>${B}_{\alpha }^{ \star  },{B}_{\beta },{B}^{\prime },{B}^{ \star  }$</td><td>Similarity Bounds of Vector Sets (see Theorem 4)</td></tr></table>

<table><tbody><tr><td>符号表示</td><td>描述</td></tr><tr><td>D</td><td>向量集数据库</td></tr><tr><td>${\mathbf{D}}^{\mathbf{H}}$</td><td>$\mathbf{D}$的稀疏二进制码</td></tr><tr><td>T, Q</td><td>目标和查询的向量集</td></tr><tr><td>$n,m,{m}_{q}$</td><td>$\mathbf{D},\mathbf{T}$和$\mathbf{Q}$的基数</td></tr><tr><td>${L}_{wta} = L$</td><td>哈希函数的数量/胜者全得</td></tr><tr><td>H</td><td>哈希函数（见定义7）</td></tr><tr><td>$k$</td><td>返回结果的数量</td></tr><tr><td>$\sigma \left( \cdot \right)$</td><td>最小 - 最大相似度函数（见引理1）</td></tr><tr><td>${S}_{ij}^{\alpha },{S}_{ij}^{\beta },\mathbf{S}$</td><td>相似度矩阵$\mathbf{S}$的第(i,j)个元素</td></tr><tr><td>${s}_{\max },{s}_{\min }$</td><td>向量集之间的最大和最小实际相似度</td></tr><tr><td>$\widehat{\mathbf{S}},{\widehat{s}}_{\max },{\widehat{s}}_{\min }$</td><td>$\mathbf{S},{s}_{\max }$和${s}_{\min }$的估计值</td></tr><tr><td>$\delta$</td><td>向量集搜索失败概率</td></tr><tr><td>${B}_{\alpha }^{ \star  },{B}_{\beta },{B}^{\prime },{B}^{ \star  }$</td><td>向量集的相似度边界（见定理4）</td></tr></tbody></table>

<!-- Media -->

1) Binary Hash Encoding for Vector Set: the rationale behind binary hash encoding is rooted in exploiting the inherent similarity-preserving property of hashing to mitigate computational complexity and enhance search efficiency.

1) 向量集的二进制哈希编码：二进制哈希编码背后的基本原理在于利用哈希固有的保相似性特性，以降低计算复杂度并提高搜索效率。

First, we define the unified paradigm of LSH shared by various LSH [8], [37], [47], [51] and proposed by [19].

首先，我们定义由[19]提出的、各种局部敏感哈希（LSH）[8]、[37]、[47]、[51]所共享的统一范式。

Definition 6 (Locality-Sensitive Hashing Function [19]). An LSH hash function $h : {\mathbb{R}}^{d} \rightarrow  \mathbb{R}$ is called a similarity-preserving hash for vectors $\mathbf{a},\mathbf{b} \in  {\mathbb{R}}^{d}$ if

定义6（局部敏感哈希函数 [19]）。如果满足以下条件，则LSH哈希函数 $h : {\mathbb{R}}^{d} \rightarrow  \mathbb{R}$ 被称为向量 $\mathbf{a},\mathbf{b} \in  {\mathbb{R}}^{d}$ 的保相似性哈希函数：

$$
\mathbb{P}\left( {h\left( \mathbf{a}\right)  = h\left( \mathbf{b}\right) }\right)  = \operatorname{sim}\left( {\mathbf{a},\mathbf{b}}\right) ,\forall \mathbf{a},\mathbf{b} \in  {\mathbb{R}}^{d},
$$

where $\operatorname{sim}\left( {\mathbf{a},\mathbf{b}}\right)  \in  \left\lbrack  {0,1}\right\rbrack$ is the similarity between vectors $\mathbf{q} \in$ ${\mathbb{R}}^{d}$ and $\mathbf{v} \in  {\mathbb{R}}^{d}$ .

其中 $\operatorname{sim}\left( {\mathbf{a},\mathbf{b}}\right)  \in  \left\lbrack  {0,1}\right\rbrack$ 是向量 $\mathbf{q} \in$ ${\mathbb{R}}^{d}$ 和 $\mathbf{v} \in  {\mathbb{R}}^{d}$ 之间的相似度。

<!-- Media -->

<!-- figureText: Short Binary Code Sparse Binary Code Fly-Based Hash Classical Hash Vector -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_3.jpg?x=142&y=1052&w=741&h=196&r=0"/>

Figure 3: Classical Hashing vs. Fly-Based Hashing

图3：经典哈希与基于果蝇的哈希

<!-- Media -->

Building on this unified paradigm, we adopt a fly-based hashing approach for our framework. As illustrated in Figure 3 this approach fundamentally differs from classical hashing methods in its dimensional transformation strategy. Classical hashing techniques typically perform dimensionality reduction to generate dense hash codes. In contrast, fly-inspired hashing emulates the neural projection patterns in Drosophila's olfactory circuit [8]. This fly-based approach performs expansion by projections, resulting in sparse binary codes.

基于这一统一范式，我们为我们的框架采用了基于果蝇的哈希方法。如图3所示，这种方法在维度转换策略上与经典哈希方法有根本区别。经典哈希技术通常进行降维以生成密集的哈希码。相比之下，受果蝇启发的哈希模拟了果蝇嗅觉回路中的神经投射模式 [8]。这种基于果蝇的方法通过投影进行扩展，从而产生稀疏的二进制代码。

The choice of fly-based hashing is motivated by two key factors. First, its sparse binary coding aligns naturally with the set-based structure of Bloom filters, which is essential for our framework’s effectiveness (see Section V). This structural compatibility makes it more suitable than traditional distance-preserving hash functions. Second, among fly-based methods, we specifically adopt BioHash [37] over the original FlyHash [8]. Empirical studies have shown that BioHash achieves approximately twice the similarity preservation performance of FlyHash [37].

选择基于果蝇的哈希方法主要受两个关键因素驱动。首先，其稀疏二进制编码与布隆过滤器的基于集合的结构自然契合，这对我们框架的有效性至关重要（见第五节）。这种结构兼容性使其比传统的保距离哈希函数更合适。其次，在基于果蝇的方法中，我们特别选择了BioHash [37] 而非原始的FlyHash [8]。实证研究表明，BioHash的保相似性性能约为FlyHash的两倍 [37]。

Definition 7 (FlyHash [8] & BioHash [37]). A FlyHash or BioHash is a function $\mathcal{H} : {\mathbb{R}}^{d} \rightarrow  \{ 0,1{\} }^{b}$ that maps a vector $\mathbf{v} \in  {\mathbb{R}}^{d}$ to a sparse binary vector $\mathbf{h} = \mathcal{H}\left( \mathbf{v}\right)  \in  \{ 0,1{\} }^{b}$ . The hash code is generated as $\mathbf{h} = {WTA}\left( {W\mathbf{v}}\right)$ ,where $W \in  {\mathbb{R}}^{b \times  d}$ is a random projection matrix, and WTA is the Winner-Take-All (WTA) operation that sets the ${L}_{wta}$ largest elements to 1 and the rest to 0 . The resulting binary vector $\mathbf{h}$ has ${L}_{wta}$ non-zero elements $\left( {\parallel \mathbf{h}{\parallel }_{1} = {L}_{wta} \ll  b}\right)$ ,where ${L}_{wta}$ also represents the number of hash functions composing $\mathcal{H}$ defined in Lemma 6

定义7（FlyHash [8] 和BioHash [37]）。FlyHash或BioHash是一个将向量 $\mathbf{v} \in  {\mathbb{R}}^{d}$ 映射到稀疏二进制向量 $\mathbf{h} = \mathcal{H}\left( \mathbf{v}\right)  \in  \{ 0,1{\} }^{b}$ 的函数 $\mathcal{H} : {\mathbb{R}}^{d} \rightarrow  \{ 0,1{\} }^{b}$。哈希码生成方式为 $\mathbf{h} = {WTA}\left( {W\mathbf{v}}\right)$，其中 $W \in  {\mathbb{R}}^{b \times  d}$ 是一个随机投影矩阵，WTA是赢家通吃（WTA）操作，它将 ${L}_{wta}$ 个最大元素置为1，其余元素置为0。得到的二进制向量 $\mathbf{h}$ 有 ${L}_{wta}$ 个非零元素 $\left( {\parallel \mathbf{h}{\parallel }_{1} = {L}_{wta} \ll  b}\right)$，其中 ${L}_{wta}$ 也表示引理6中定义的组成 $\mathcal{H}$ 的哈希函数的数量。

<!-- Media -->

<!-- figureText: Matrix Winner- BioHash 0.4 (0.4) 0.2 0 0 0.9 (1.8) 0 1.5(1.1) 0 0.7)⑥.9 0 0 1.4 (0.4) 0 Vector Set Sparse Binary Codes of Vector Set (b) The Hashing Process of Vector Set Projection Take-Al 0.4 (0.9) 0.9 Vector Expanding Sparse Vector Binary Code (a) The Process of BioHash & FlyHash -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_3.jpg?x=916&y=134&w=740&h=532&r=0"/>

Figure 4: The Hashing Process of BioVSS

图4：BioVSS的哈希过程

<!-- Media -->

As illustrated in Figure 4(a), FlyHash [8] generates LSH codes by mimicking the olfactory neural circuit of the fly. Vectors are projected into higher-dimensional spaces using a matrix that simulates projection neurons, enhancing representational capacity. Winner-take-all generates sparse binary codes by selecting the top responding neurons. This has been shown to exhibit locality-sensitive property [8] (Definition 6).

如图4(a)所示，FlyHash [8] 通过模拟果蝇的嗅觉神经回路来生成LSH代码。使用模拟投射神经元的矩阵将向量投影到更高维空间，增强表示能力。赢家通吃通过选择响应最强的神经元来生成稀疏二进制代码。已证明这种方法具有局部敏感特性 [8]（定义6）。

Section IV-B establishes the theoretical foundations of this method. It demonstrates that the hash codes effectively support the approximate top- $k$ vector set search. As illustrated in Figure 4(b), the algorithm quantizes the vector sets into binary hash codes. Next, we introduce the hashing process.

第四节B部分奠定了该方法的理论基础。它表明哈希码有效地支持近似前 $k$ 向量集搜索。如图4(b)所示，该算法将向量集量化为二进制哈希码。接下来，我们介绍哈希过程。

Algorithm 1 outlines the process of generating sparse binary codes for a database of vector sets. The input comprises a database $\mathbf{D}$ of $n$ vector sets ${\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ and a parameter ${L}_{wta}$ for the winner-takes-all operation,while the output ${\mathbf{D}}^{\mathbf{H}}$ is a collection of corresponding sparse binary codes. The algorithm iterates through each vector set ${\mathbf{V}}_{j}$ (line 2),applying the BioHash (see Definition 7) to each vector $\mathbf{v}$ (lines 5-6). This involves a matrix projection $W\mathbf{v}$ followed by the WTA operation,which sets the ${L}_{wta}$ largest elements to 1 and the rest to 0,resulting in a binary code $\mathbf{h}$ with exactly ${L}_{wta}$ nonzero elements. These sparse binary codes are aggregated per vector set (line 7) and compiled into ${\mathbf{D}}^{\mathbf{H}}$ (line 8).

算法1概述了为向量集数据库生成稀疏二进制代码的过程。输入包括一个包含$n$个向量集${\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$的数据库$\mathbf{D}$，以及一个用于胜者全得（winner-takes-all，WTA）操作的参数${L}_{wta}$，而输出${\mathbf{D}}^{\mathbf{H}}$是一组对应的稀疏二进制代码。该算法遍历每个向量集${\mathbf{V}}_{j}$（第2行），对每个向量$\mathbf{v}$应用生物哈希（BioHash，见定义7）（第5 - 6行）。这包括一个矩阵投影$W\mathbf{v}$，随后进行WTA操作，该操作将${L}_{wta}$个最大的元素置为1，其余元素置为0，从而得到一个恰好有${L}_{wta}$个非零元素的二进制代码$\mathbf{h}$。这些稀疏二进制代码按向量集进行聚合（第7行），并汇总到${\mathbf{D}}^{\mathbf{H}}$中（第8行）。

<!-- Media -->

Algorithm 1: Gen_Binary_Codes $\left( {\mathbf{D},{L}_{wta}}\right)$

算法1：Gen_Binary_Codes $\left( {\mathbf{D},{L}_{wta}}\right)$

---

Input: $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ : vector set database, ${L}_{wta}$ : # of WTA.

输入：$\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$：向量集数据库，${L}_{wta}$：WTA的数量。

Output: ${\mathbf{D}}^{\mathbf{H}}$ : sparse binary codes.

输出：${\mathbf{D}}^{\mathbf{H}}$：稀疏二进制代码。

Initialization: ${\mathbf{D}}^{\mathbf{H}} = \varnothing$

初始化：${\mathbf{D}}^{\mathbf{H}} = \varnothing$

for $j = 1$ to $n$ do

从$j = 1$到$n$进行循环

	${\mathbf{H}}_{j} = \varnothing ;$

	for each $\mathbf{v} \in  {\mathbf{V}}_{j}$ do

	对每个$\mathbf{v} \in  {\mathbf{V}}_{j}$进行循环

		${\mathbf{h}}_{p} = W\mathbf{v}$ ; // Matrix projection

		${\mathbf{h}}_{p} = W\mathbf{v}$；// 矩阵投影

		$\mathbf{h} = {WTA}\left( {{\mathbf{h}}_{p},{L}_{wta}}\right) ;\;//$ Winner-takes-all

		$\mathbf{h} = {WTA}\left( {{\mathbf{h}}_{p},{L}_{wta}}\right) ;\;//$ 胜者全得

		${\mathbf{H}}_{j} = {\mathbf{H}}_{j} \cup  \{ \mathbf{h}\}$ ;

	${\mathbf{D}}^{\mathbf{H}} = {\mathbf{D}}^{\mathbf{H}} \cup  \left\{  {\mathbf{H}}_{j}\right\}  ;$

return ${\mathbf{D}}^{\mathrm{H}}$ ;

返回 ${\mathbf{D}}^{\mathrm{H}}$；

---

Algorithm 2: BioVSS_Topk_Search $\left( {\mathbf{Q},k,\mathbf{D},{\mathbf{D}}^{\mathbf{H}},{L}_{wta},c}\right)$

算法2：BioVSS_Topk_Search $\left( {\mathbf{Q},k,\mathbf{D},{\mathbf{D}}^{\mathbf{H}},{L}_{wta},c}\right)$

---

Input: $\mathbf{Q}$ : query vector set, $k : \#$ of top sets, $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ :

输入：$\mathbf{Q}$：查询向量集，$k : \#$：前k个集合，$\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$：

			vector set database, ${\mathbf{D}}^{\mathbf{H}}$ : sparse binary codes, ${L}_{wta}$ :

					向量集数据库，${\mathbf{D}}^{\mathbf{H}}$：稀疏二进制代码，${L}_{wta}$：

			#of WTA, $c$ : the size of candidate set.

			#		WTA的数量，$c$：候选集的大小。

Output: $\mathbf{R}$ : top- $k$ vector sets.

输出：$\mathbf{R}$：前$k$个向量集。

Initialization: ${\mathbf{Q}}^{\mathbf{H}} = \varnothing ,C = \varnothing$ ;

初始化：${\mathbf{Q}}^{\mathbf{H}} = \varnothing ,C = \varnothing$；

for each $\mathbf{q} \in  \mathbf{Q}$ do

对每个 $\mathbf{q} \in  \mathbf{Q}$ 执行

		${\mathbf{h}}_{p} = W\mathbf{q}$ ; // Matrix projection

		${\mathbf{h}}_{p} = W\mathbf{q}$ ; // 矩阵投影

		${\mathbf{h}}_{q} = {WTA}\left( {{\mathbf{h}}_{p},{L}_{wta}}\right) ;\;//$ Winner-takes-all

		${\mathbf{h}}_{q} = {WTA}\left( {{\mathbf{h}}_{p},{L}_{wta}}\right) ;\;//$ 胜者全得

		${\mathbf{Q}}^{\mathbf{H}} = {\mathbf{Q}}^{\mathbf{H}} \cup  \left\{  {\mathbf{h}}_{q}\right\}$

// Select candidates

// 选择候选对象

for $j = 1$ to $n$ do

从 $j = 1$ 到 $n$ 执行

		${d}_{H} = {\operatorname{Haus}}^{H}\left( {{\mathbf{Q}}^{\mathbf{H}},{\mathbf{H}}_{j}}\right)$ ;

		$C = C \cup  \left\{  \left( {{\mathbf{V}}_{j},{d}_{H}}\right) \right\}$

$\mathcal{F} = \left\{  {\left( {{\mathbf{V}}_{i},{d}_{H}}\right)  \in  \mathcal{C} \mid  {d}_{H} \leq  {d}_{H}^{\left( c\right) }}\right\}$ ,where ${d}_{H}^{\left( c\right) }$ is the $c$ -th

$\mathcal{F} = \left\{  {\left( {{\mathbf{V}}_{i},{d}_{H}}\right)  \in  \mathcal{C} \mid  {d}_{H} \leq  {d}_{H}^{\left( c\right) }}\right\}$ ，其中 ${d}_{H}^{\left( c\right) }$ 是 $c$ 小的

	smallest ${d}_{H}$ in $C$ ;

	${d}_{H}$ 在 $C$ 中；

// Select top- $k$ results

// 选择前 $k$ 个结果

$\mathcal{D} = \varnothing ;$

for each $\left( {{\mathbf{V}}_{i},{d}_{H}}\right)  \in  \mathcal{F}$ do

对每个 $\left( {{\mathbf{V}}_{i},{d}_{H}}\right)  \in  \mathcal{F}$ 执行

		${d}_{i} = \operatorname{Haus}\left( {\mathbf{Q},{\mathbf{V}}_{i}}\right)$ ;

		$\mathcal{D} = \mathcal{D} \cup  \left\{  \left( {{\mathbf{V}}_{i},{d}_{i}}\right) \right\}$

$\mathbf{R} = \left\{  {\left( {{\mathbf{V}}_{i},{d}_{i}}\right)  \in  \mathcal{D} \mid  {d}_{i} \leq  {d}_{i}^{\left( k\right) }}\right\}$ ,where ${d}_{i}^{\left( k\right) }$ is the $k$ -th

$\mathbf{R} = \left\{  {\left( {{\mathbf{V}}_{i},{d}_{i}}\right)  \in  \mathcal{D} \mid  {d}_{i} \leq  {d}_{i}^{\left( k\right) }}\right\}$ ，其中 ${d}_{i}^{\left( k\right) }$ 是 $k$ 小的

	smallest ${d}_{i}$ in $\mathcal{D}$ ;

	${d}_{i}$ 在 $\mathcal{D}$ 中；

return $\mathbf{R}$ ;

返回 $\mathbf{R}$ ；

---

<!-- Media -->

2) Search Execution in BioVSS: binary codes offer computational advantages by enabling fast bitwise operations, which are inherently supported by modern CPU architectures [29]. This leads to significant speedups compared to traditional floating-point computations.

2) BioVSS 中的搜索执行：二进制代码通过支持快速位运算提供了计算优势，现代 CPU 架构本身就支持这些运算 [29]。与传统的浮点计算相比，这显著提高了速度。

Algorithm 2 outlines BioVSS top- $k$ search process in vector set databases. The input includes a query set $\mathbf{Q}$ ,the number of desired results $k$ ,the database $\mathbf{D}$ ,pre-computed sparse binary codes ${\mathbf{D}}^{\mathbf{H}}$ ,and the WTA parameter ${L}_{wta}$ . The output $\mathbf{R}$ contains the top- $k$ most similar vector sets. The algorithm begins by generating sparse binary codes for the query set $\mathbf{Q}$ (lines 2-5). This creates the binary representation of the query set ${\mathbf{Q}}^{\mathbf{H}}$ . To leverage CPU-friendly bit operations, we compute the Hamming-based Hausdorff distance ${\text{Haus}}^{H}$ (substituting dist with hamming(q,v)[34] in Definition 4) between ${\mathbf{Q}}^{\mathbf{H}}$ and each set of binary codes ${\mathbf{H}}_{j}$ in ${\mathbf{D}}^{\mathbf{H}}$ (lines 6-8). This step efficiently identifies a set of candidate vector sets $\mathcal{F}$ based on their binary code similarities (line 9). The final stage refines these candidates by computing the actual Hausdorff distance (see Definition 4) between $\mathbf{Q}$ and each candidate vector set ${\mathbf{V}}_{i}$ in the original space (lines 10-13). The algorithm then selects the top- $k$ results based on these distances (line 14) and then returns $\mathbf{R}$ (line 15).

算法 2 概述了 BioVSS 在向量集数据库中的前 $k$ 搜索过程。输入包括查询集 $\mathbf{Q}$、所需结果的数量 $k$、数据库 $\mathbf{D}$、预计算的稀疏二进制代码 ${\mathbf{D}}^{\mathbf{H}}$ 以及胜者全得（Winner-takes-all，WTA）参数 ${L}_{wta}$。输出 $\mathbf{R}$ 包含最相似的前 $k$ 个向量集。该算法首先为查询集 $\mathbf{Q}$ 生成稀疏二进制代码（第 2 - 5 行）。这创建了查询集 ${\mathbf{Q}}^{\mathbf{H}}$ 的二进制表示。为了利用对 CPU 友好的位运算，我们计算 ${\mathbf{Q}}^{\mathbf{H}}$ 与 ${\mathbf{D}}^{\mathbf{H}}$ 中每组二进制代码 ${\mathbf{H}}_{j}$ 之间基于汉明距离的豪斯多夫距离 ${\text{Haus}}^{H}$（在定义 4 中用 hamming(q,v)[34] 替换 dist）（第 6 - 8 行）。这一步根据二进制代码的相似性有效地识别出一组候选向量集 $\mathcal{F}$（第 9 行）。最后阶段通过计算 $\mathbf{Q}$ 与原始空间中每个候选向量集 ${\mathbf{V}}_{i}$ 之间的实际豪斯多夫距离（见定义 4）来优化这些候选对象（第 10 - 13 行）。然后，该算法根据这些距离选择前 $k$ 个结果（第 14 行），并返回 $\mathbf{R}$（第 15 行）。

## B. Theoretical Analysis of Algorithm Correctness

## B. 算法正确性的理论分析

This section presents a theoretical analysis of the probabilistic guarantees for result correctness in BioVSS. We define a function constraining similarity measure bounds, followed by upper and lower tail probability bounds for similarity comparisons. Through several lemmas, we construct our theoretical framework, culminating in Theorem 4, which establishes the relationship between the error rate $\delta$ and $L = {L}_{wta}$ .

本节对 BioVSS 中结果正确性的概率保证进行理论分析。我们定义一个约束相似度度量界限的函数，接着给出相似度比较的上尾和下尾概率界限。通过几个引理，我们构建了理论框架，最终得出定理 4，该定理建立了错误率 $\delta$ 和 $L = {L}_{wta}$ 之间的关系。

Assumptions. Our framework assumes all vectors undergo ${L2}$ normalization. This allows us to define similarity between vectors $\mathbf{q}$ and $\mathbf{v}$ as their inner product: $\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)  = {\mathbf{q}}^{T}\mathbf{v}$ . We denote the query vector set and another vector set as $\mathbf{Q}$ and $\mathbf{V}$ , respectively. To facilitate proof,we define ${\operatorname{Sim}}_{\text{Haus }}\left( {\mathbf{Q},\mathbf{V}}\right)  =$ $\min \left( {\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\max }\limits_{{\mathbf{v} \in  \mathbf{V}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) ,\mathop{\min }\limits_{{\mathbf{v} \in  \mathbf{V}}}\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) .$

假设。我们的框架假设所有向量都经过${L2}$归一化处理。这使我们能够将向量$\mathbf{q}$和$\mathbf{v}$之间的相似度定义为它们的内积：$\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)  = {\mathbf{q}}^{T}\mathbf{v}$ 。我们分别将查询向量集和另一个向量集表示为$\mathbf{Q}$和$\mathbf{V}$ 。为便于证明，我们定义${\operatorname{Sim}}_{\text{Haus }}\left( {\mathbf{Q},\mathbf{V}}\right)  =$ $\min \left( {\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\max }\limits_{{\mathbf{v} \in  \mathbf{V}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) ,\mathop{\min }\limits_{{\mathbf{v} \in  \mathbf{V}}}\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) .$

The similarity metric is equivalent to the Hausdorff distance for ${L2}$ -normalized vectors.

对于${L2}$归一化向量，该相似度度量等同于豪斯多夫距离（Hausdorff distance）。

1) Bounds on Min-Max Similarity Scores: we begin by introducing upper and lower bounds for a function $\sigma$ ,which forms the foundation for our subsequent proofs.

1) 最小 - 最大相似度得分的边界：我们首先引入函数$\sigma$的上下界，这是我们后续证明的基础。

Lemma 1. Consider a matrix $\mathbf{S} = \left\lbrack  {s}_{ij}\right\rbrack   \in  {\mathbb{R}}^{{m}_{q} \times  m}$ containing similarity scores between a query vector set $\mathbf{Q}$ and a target vector set $\mathbf{T}$ ,where $\left| \mathbf{Q}\right|  = {m}_{q}$ and $\left| \mathbf{T}\right|  = m$ . Define a function $\sigma  : {\mathbb{R}}^{{m}_{q} \times  m} \rightarrow  \mathbb{R}$ as $\sigma \left( \mathbf{S}\right)  = \min \left( {\mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij},\mathop{\min }\limits_{j}\mathop{\max }\limits_{i}{s}_{ij}}\right)$ Then, $\sigma \left( \mathbf{S}\right)$ satisfies the following bounds: $\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq$ $\sigma \left( \mathbf{S}\right)  \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ .

引理1。考虑一个矩阵$\mathbf{S} = \left\lbrack  {s}_{ij}\right\rbrack   \in  {\mathbb{R}}^{{m}_{q} \times  m}$，它包含查询向量集$\mathbf{Q}$和目标向量集$\mathbf{T}$之间的相似度得分，其中$\left| \mathbf{Q}\right|  = {m}_{q}$且$\left| \mathbf{T}\right|  = m$ 。将函数$\sigma  : {\mathbb{R}}^{{m}_{q} \times  m} \rightarrow  \mathbb{R}$定义为$\sigma \left( \mathbf{S}\right)  = \min \left( {\mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij},\mathop{\min }\limits_{j}\mathop{\max }\limits_{i}{s}_{ij}}\right)$ 。那么，$\sigma \left( \mathbf{S}\right)$满足以下边界条件：$\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq$ $\sigma \left( \mathbf{S}\right)  \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ 。

Proof. Let $a = \mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij}$ and $b = \mathop{\min }\limits_{j}\mathop{\max }\limits_{i}{s}_{ij}$ . Then $\sigma \left( \mathbf{S}\right)  = \min \left( {a,b}\right)$ .

证明。设$a = \mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij}$且$b = \mathop{\min }\limits_{j}\mathop{\max }\limits_{i}{s}_{ij}$ 。则$\sigma \left( \mathbf{S}\right)  = \min \left( {a,b}\right)$ 。

For the lower bound: $\forall i,j : {s}_{ij} \leq  \mathop{\max }\limits_{j}{s}_{ij} \Rightarrow  \mathop{\min }\limits_{{i,j}}{s}_{ij} \leq$ $\mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij} = a$ . Similarly, $\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq  b$ . Therefore, $\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq  \min \left( {a,b}\right)  = \sigma \left( \mathbf{S}\right)$ .

对于下界：$\forall i,j : {s}_{ij} \leq  \mathop{\max }\limits_{j}{s}_{ij} \Rightarrow  \mathop{\min }\limits_{{i,j}}{s}_{ij} \leq$ $\mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij} = a$ 。同理，$\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq  b$ 。因此，$\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq  \min \left( {a,b}\right)  = \sigma \left( \mathbf{S}\right)$ 。

For the upper bound: $\forall i : \mathop{\max }\limits_{j}{s}_{ij} \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij} \Rightarrow$ $a = \mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij} \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ . Similarly, $b \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ . Therefore, $\sigma \left( \mathbf{S}\right)  = \min \left( {a,b}\right)  \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ .

对于上界：$\forall i : \mathop{\max }\limits_{j}{s}_{ij} \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij} \Rightarrow$ $a = \mathop{\min }\limits_{i}\mathop{\max }\limits_{j}{s}_{ij} \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ 。类似地，$b \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ 。因此，$\sigma \left( \mathbf{S}\right)  = \min \left( {a,b}\right)  \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ 。

Thus,we have $\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq  \sigma \left( \mathbf{S}\right)  \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ .

因此，我们有 $\mathop{\min }\limits_{{i,j}}{s}_{ij} \leq  \sigma \left( \mathbf{S}\right)  \leq  \mathop{\max }\limits_{{i,j}}{s}_{ij}$ 。

2) Upper Tail Probability Bound: we now establish the upper tail probability bound for our relevance score function, which forms the foundation for our subsequent proofs.

2) 上尾概率界：我们现在为我们的相关性得分函数建立上尾概率界，这为我们后续的证明奠定了基础。

Lemma 2. Consider a similarity matrix $\mathbf{S} \in  {\mathbb{R}}^{{m}_{q} \times  m}$ between a query vector set and a target vector set. Define the maximum similarity score in $\mathbf{S}$ as ${s}_{\max } = \mathop{\max }\limits_{{i,j}}{s}_{ij}$ . Given an estimated similarity matrix $\widehat{\mathbf{S}}$ of $\mathbf{S}$ and a threshold ${\tau }_{1} \in  \left( {{s}_{\max },1}\right)$ ,we define ${\Delta }_{1} = {\tau }_{1} - {s}_{\max }$ . Then,the following inequality holds:

引理2。考虑查询向量集和目标向量集之间的相似度矩阵 $\mathbf{S} \in  {\mathbb{R}}^{{m}_{q} \times  m}$ 。将 $\mathbf{S}$ 中的最大相似度得分定义为 ${s}_{\max } = \mathop{\max }\limits_{{i,j}}{s}_{ij}$ 。给定 $\mathbf{S}$ 的估计相似度矩阵 $\widehat{\mathbf{S}}$ 和一个阈值 ${\tau }_{1} \in  \left( {{s}_{\max },1}\right)$ ，我们定义 ${\Delta }_{1} = {\tau }_{1} - {s}_{\max }$ 。那么，以下不等式成立：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \geq  {s}_{\max } + {\Delta }_{1}}\right\rbrack   \leq  {m}_{q}m{\gamma }^{L},
$$

for $\gamma  = {\left( \frac{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }\right) }^{{\tau }_{1}}\left( \frac{1 - {s}_{\max }}{1 - {\tau }_{1}}\right)$ . Here, $\sigma \left( \cdot \right)$ is the operator defined in Lemma 1 and $L \in  {\mathbb{Z}}^{ + }$ represents the number of hash functions.

对于 $\gamma  = {\left( \frac{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }\right) }^{{\tau }_{1}}\left( \frac{1 - {s}_{\max }}{1 - {\tau }_{1}}\right)$ 。这里，$\sigma \left( \cdot \right)$ 是引理1中定义的算子，$L \in  {\mathbb{Z}}^{ + }$ 表示哈希函数的数量。

Proof. Applying a generic Chernoff bound to $\sigma \left( \widehat{\mathbf{S}}\right)$ yields the following bounds for any $t > 0$ :

证明。对 $\sigma \left( \widehat{\mathbf{S}}\right)$ 应用通用的切尔诺夫界（Chernoff bound），对于任何 $t > 0$ 可得到以下界：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \geq  {\tau }_{1}}\right\rbrack   = \Pr \left\lbrack  {{e}^{{t\sigma }\left( \widehat{\mathbf{S}}\right) } \geq  {e}^{t{\tau }_{1}}}\right\rbrack   \leq  \frac{\mathbb{E}\left\lbrack  {e}^{{t\sigma }\left( \widehat{\mathbf{S}}\right) }\right\rbrack  }{{e}^{t{\tau }_{1}}}.
$$

Given an LSH family (see Definition 7), the estimated maximum similarity ${\widehat{s}}_{\max }$ follows a scaled binomial distribution with parameters ${s}_{\max }$ and ${L}^{-1}$ ,i.e., ${\widehat{s}}_{\max } \sim  {L}^{-1}\mathcal{B}\left( {{s}_{\max },L}\right)$ . Substituting the binomial moment generating function into the expression and using Lemma 1, we can bound the numerator:

给定一个局部敏感哈希（LSH）族（见定义7），估计的最大相似度 ${\widehat{s}}_{\max }$ 服从参数为 ${s}_{\max }$ 和 ${L}^{-1}$ 的缩放二项分布，即 ${\widehat{s}}_{\max } \sim  {L}^{-1}\mathcal{B}\left( {{s}_{\max },L}\right)$ 。将二项矩生成函数代入表达式并使用引理1，我们可以对分子进行界定：

$$
\mathbb{E}\left\lbrack  {e}^{{t\sigma }\left( \widehat{\mathbf{S}}\right) }\right\rbrack   \leq  \mathbb{E}\left\lbrack  {e}^{t\mathop{\max }\limits_{{1 \leq  i \leq  {m}_{q},1 \leq  j \leq  m}}{\widehat{S}}_{ij}}\right\rbrack  
$$

$$
 \leq  {m}_{q}m\mathbb{E}\left\lbrack  {e}^{t{\widehat{s}}_{\max }}\right\rbrack   = {m}_{q}m{\left( 1 - {s}_{\max } + {s}_{\max }{e}^{\frac{t}{L}}\right) }^{L}.
$$

Combining the Chernoff bound and the numerator bound yields: $\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \geq  {\tau }_{1}}\right\rbrack   \leq  {m}_{q}m{e}^{-t{\tau }_{1}}{\left( 1 - {s}_{\max } + {s}_{\max }{e}^{\frac{t}{L}}\right) }^{L}$ .

结合切尔诺夫界和分子界可得：$\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \geq  {\tau }_{1}}\right\rbrack   \leq  {m}_{q}m{e}^{-t{\tau }_{1}}{\left( 1 - {s}_{\max } + {s}_{\max }{e}^{\frac{t}{L}}\right) }^{L}$ 。

To find the tightest upper tail probability bound, we can determine the infimum by setting the derivative of the bound for $t$ equal to zero. Then,we get: ${t}^{ \star  } = L\ln \left( \frac{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }\right)$ .

为了找到最紧的上尾概率界，我们可以通过将 $t$ 的界的导数设为零来确定下确界。然后，我们得到：${t}^{ \star  } = L\ln \left( \frac{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }\right)$ 。

Since ${\tau }_{1} \in  \left( {{s}_{\max },1}\right)$ ,the numerator of the fraction inside the logarithm is greater than the denominator, ensuring that ${t}^{ \star  } > 0$ . This result allows us to obtain the tightest bound.

由于${\tau }_{1} \in  \left( {{s}_{\max },1}\right)$，对数内分数的分子大于分母，从而确保${t}^{ \star  } > 0$。这一结果使我们能够得到最紧的界。

Substituting $t = {t}^{ \star  }$ into the bound,we obtain:

将$t = {t}^{ \star  }$代入该界，我们得到：

$\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \geq  {\tau }_{1}}\right\rbrack   \leq  {m}_{q}m{\left( {\left( \frac{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }\right) }^{-{\tau }_{1}}\left( \frac{1 - {s}_{\max }}{1 - {\tau }_{1}}\right) \right) }^{L}.$

Thus we have: $\gamma  = {\left( \frac{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }\right) }^{{\tau }_{1}}\left( \frac{1 - {s}_{\max }}{1 - {\tau }_{1}}\right)$ . $\square$

因此，我们有：$\gamma  = {\left( \frac{{s}_{\max }\left( {1 - {\tau }_{1}}\right) }{{\tau }_{1}\left( {1 - {s}_{\max }}\right) }\right) }^{{\tau }_{1}}\left( \frac{1 - {s}_{\max }}{1 - {\tau }_{1}}\right)$。$\square$

3) Lower Tail Probability Bound: we now establish the lower tail probability bound for our relevance score function, which forms the foundation for our subsequent proofs.

3) 下尾概率界：我们现在为相关性得分函数建立下尾概率界，这是后续证明的基础。

Lemma 3. Consider a similarity matrix $\mathbf{S} \in  {\mathbb{R}}^{{m}_{q} \times  m}$ between a query vector set and a target vector set. Define the minimum similarity score in $\mathbf{S}$ as ${s}_{\min } = \mathop{\min }\limits_{{i,j}}{s}_{ij}$ . Given an estimated similarity matrix $\widehat{\mathbf{S}}$ of $\mathbf{S}$ and a threshold ${\tau }_{2} \in  \left( {0,{s}_{\min }}\right)$ ,we define ${\Delta }_{2} = {s}_{\min } - {\tau }_{2}$ . Then,the following inequality holds:

引理3。考虑查询向量集和目标向量集之间的相似度矩阵$\mathbf{S} \in  {\mathbb{R}}^{{m}_{q} \times  m}$。将$\mathbf{S}$中的最小相似度得分定义为${s}_{\min } = \mathop{\min }\limits_{{i,j}}{s}_{ij}$。给定$\mathbf{S}$的估计相似度矩阵$\widehat{\mathbf{S}}$和阈值${\tau }_{2} \in  \left( {0,{s}_{\min }}\right)$，我们定义${\Delta }_{2} = {s}_{\min } - {\tau }_{2}$。那么，以下不等式成立：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{s}}\right)  \leq  {s}_{\min } - {\Delta }_{2}}\right\rbrack   \leq  {m}_{q}m{\xi }^{L},
$$

for $\gamma  = {\left( \frac{{s}_{\min }\left( {1 - {\tau }_{2}}\right) }{{\tau }_{2}\left( {1 - {s}_{\min }}\right) }\right) }^{{\tau }_{2}}\left( \frac{1 - {s}_{\min }}{1 - {\tau }_{2}}\right)$ . Here, $\sigma \left( \cdot \right)$ is the operator defined in Lemma 1 and $L \in  {\mathbb{Z}}^{ + }$ represents the number of hash functions.

对于$\gamma  = {\left( \frac{{s}_{\min }\left( {1 - {\tau }_{2}}\right) }{{\tau }_{2}\left( {1 - {s}_{\min }}\right) }\right) }^{{\tau }_{2}}\left( \frac{1 - {s}_{\min }}{1 - {\tau }_{2}}\right)$。这里，$\sigma \left( \cdot \right)$是引理1中定义的算子，$L \in  {\mathbb{Z}}^{ + }$表示哈希函数的数量。

Proof. Applying a generic Chernoff bound to $\sigma \left( \widehat{\mathbf{S}}\right)$ yields the following inequality for any $t < 0$ :

证明。对$\sigma \left( \widehat{\mathbf{S}}\right)$应用通用的切尔诺夫界（Chernoff bound），对于任意$t < 0$，可得到以下不等式：

$$
\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \leq  {\tau }_{2}}\right\rbrack   = \Pr \left\lbrack  {{e}^{{t\sigma }\left( \widehat{\mathbf{S}}\right) } \geq  {e}^{t{\tau }_{2}}}\right\rbrack   \leq  \frac{\mathbb{E}\left\lbrack  {e}^{{t\sigma }\left( \widehat{\mathbf{S}}\right) }\right\rbrack  }{{e}^{t{\tau }_{2}}}.
$$

Given an LSH family (see Definition 7), the estimated minimum similarity ${\widehat{s}}_{\min }$ follows a scaled binomial distribution with parameters ${s}_{\min }$ and ${L}^{-1}$ ,i.e., ${\widehat{s}}_{\min } \sim  {L}^{-1}\mathcal{B}\left( {{s}_{\min },L}\right)$ . Substituting the binomial moment generating function into the expression and using Lemma 1, we can bound the numerator:

给定一个局部敏感哈希（LSH）族（见定义7），估计的最小相似度${\widehat{s}}_{\min }$服从参数为${s}_{\min }$和${L}^{-1}$的缩放二项分布，即${\widehat{s}}_{\min } \sim  {L}^{-1}\mathcal{B}\left( {{s}_{\min },L}\right)$。将二项矩生成函数代入表达式并使用引理1，我们可以对分子进行界定：

$$
\mathbb{E}\left\lbrack  {e}^{{t\sigma }\left( \widehat{\mathbf{S}}\right) }\right\rbrack   \leq  \mathbb{E}\left\lbrack  {e}^{t\mathop{\min }\limits_{{1 \leq  i \leq  {m}_{q},1 \leq  j \leq  m}}{\widehat{S}}_{ij}}\right\rbrack  
$$

$$
 \leq  {m}_{q}m\mathbb{E}\left\lbrack  {e}^{t{\widehat{s}}_{\min }}\right\rbrack   = {m}_{q}m{\left( 1 - {s}_{\min } + {s}_{\min }{e}^{\frac{t}{L}}\right) }^{L}.
$$

Combining the Chernoff bound and the numerator bound yields: $\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \leq  {\tau }_{2}}\right\rbrack   \leq  {m}_{q}m{e}^{-t{\tau }_{2}}{\left( 1 - {s}_{\min } + {s}_{\min }{e}^{\frac{t}{L}}\right) }^{L}$ .

结合切尔诺夫界和分子界，我们得到：$\Pr \left\lbrack  {\sigma \left( \widehat{\mathbf{S}}\right)  \leq  {\tau }_{2}}\right\rbrack   \leq  {m}_{q}m{e}^{-t{\tau }_{2}}{\left( 1 - {s}_{\min } + {s}_{\min }{e}^{\frac{t}{L}}\right) }^{L}$。

To find the tightest upper bound, we can determine the infimum by setting the derivative of the upper bound with respect to $t$ equal to zero. Then,we obtain: ${t}^{ \star  } = L\ln \left( \frac{{\tau }_{2}\left( {1 - {s}_{\min }}\right) }{{s}_{\min }\left( {1 - {\tau }_{2}}\right) }\right)$ .

为了找到最紧的上界，我们可以通过令上界关于$t$的导数等于零来确定下确界。然后，我们得到：${t}^{ \star  } = L\ln \left( \frac{{\tau }_{2}\left( {1 - {s}_{\min }}\right) }{{s}_{\min }\left( {1 - {\tau }_{2}}\right) }\right)$。

Since ${\tau }_{2} \in  \left( {0,{s}_{\min }}\right)$ ,the numerator of the fraction inside the logarithm is greater than the denominator,ensuring that ${t}^{ \star  } < 0$ . This result allows us to obtain the tightest upper bound.

由于${\tau }_{2} \in  \left( {0,{s}_{\min }}\right)$，对数内分数的分子大于分母，从而确保${t}^{ \star  } < 0$。这一结果使我们能够得到最紧的上界。

Similar to Lemma 2,we get: $\xi  = {\left( \frac{{s}_{\min }\left( {1 - {\tau }_{2}}\right) }{{\tau }_{2}\left( {1 - {s}_{\min }}\right) }\right) }^{{\tau }_{2}}\left( \frac{1 - {s}_{\min }}{1 - {\tau }_{2}}\right)$ .

与引理2类似，我们得到：$\xi  = {\left( \frac{{s}_{\min }\left( {1 - {\tau }_{2}}\right) }{{\tau }_{2}\left( {1 - {s}_{\min }}\right) }\right) }^{{\tau }_{2}}\left( \frac{1 - {s}_{\min }}{1 - {\tau }_{2}}\right)$ 。

4) Probabilistic Correctness for Search Results: this section presents a theorem establishing the probabilistic guarantees to the approximate top- $k$ vector set search problem.

4) 搜索结果的概率正确性：本节给出一个定理，为近似前 $k$ 向量集搜索问题建立概率保证。

Theorem 4. Let ${\mathbf{V}}_{\alpha }^{ \star  }$ be one of the top- $k$ vector sets that minimize the Hausdorff distance $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ,i.e., ${\mathbf{V}}_{\alpha }^{ \star  } \in$ ${\operatorname{argmin}}_{\mathbf{V} \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ,and ${\mathbf{V}}_{\beta }$ be any other vector set in the database D. By applying Lemma 1, we obtain the following upper and lower bounds for Haus $\left( {\overline{\mathbf{Q}},{\mathbf{V}}_{\alpha }^{ \star  }}\right)$ and Haus $\left( {\mathbf{Q},{\mathbf{V}}_{\beta }}\right)$ :

定理4。设 ${\mathbf{V}}_{\alpha }^{ \star  }$ 是使豪斯多夫距离 $\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ 最小的前 $k$ 向量集之一，即 ${\mathbf{V}}_{\alpha }^{ \star  } \in$ ${\operatorname{argmin}}_{\mathbf{V} \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ，且 ${\mathbf{V}}_{\beta }$ 是数据库D中的任何其他向量集。通过应用引理1，我们得到豪斯 $\left( {\overline{\mathbf{Q}},{\mathbf{V}}_{\alpha }^{ \star  }}\right)$ 和豪斯 $\left( {\mathbf{Q},{\mathbf{V}}_{\beta }}\right)$ 的以下上下界：

${B}_{\alpha }^{ \star  } = 1 - \mathop{\min }\limits_{{i,j}}{S}_{ij}^{\alpha } = 1 - {s}_{\alpha ,\min },\;{B}_{\beta } = 1 - \mathop{\max }\limits_{{i,j}}{S}_{ij}^{\beta } = 1 - {s}_{\beta ,\max },$

where ${S}_{ij}^{\alpha }$ and ${S}_{ij}^{\beta }$ denote the element-wise similarity matrices between each vector in the query set $\mathbf{Q}$ and each vector in ${\mathbf{V}}_{\alpha }^{ \star  }$ and ${\mathbf{V}}_{\beta }$ .

其中 ${S}_{ij}^{\alpha }$ 和 ${S}_{ij}^{\beta }$ 分别表示查询集 $\mathbf{Q}$ 中的每个向量与 ${\mathbf{V}}_{\alpha }^{ \star  }$ 和 ${\mathbf{V}}_{\beta }$ 中的每个向量之间的逐元素相似性矩阵。

Let ${B}^{\prime } = \mathop{\min }\limits_{\beta }{B}_{\beta }$ be the minimum value of ${B}_{\beta }$ over any vector set ${\mathbf{V}}_{\beta }$ not in the top- $k$ results,and let ${B}^{ \star  } = \mathop{\max }\limits_{\alpha }{B}_{\alpha }^{ \star  }$ be the maximum value of ${B}_{\alpha }^{ \star  }$ over any top- $k$ vector set ${\mathbf{V}}_{\alpha }^{ \star  }$ . Define ${\Delta }_{1} + {\Delta }_{2}$ as follows:

设 ${B}^{\prime } = \mathop{\min }\limits_{\beta }{B}_{\beta }$ 是不在前 $k$ 结果中的任何向量集 ${\mathbf{V}}_{\beta }$ 上 ${B}_{\beta }$ 的最小值，设 ${B}^{ \star  } = \mathop{\max }\limits_{\alpha }{B}_{\alpha }^{ \star  }$ 是前 $k$ 向量集 ${\mathbf{V}}_{\alpha }^{ \star  }$ 上 ${B}_{\alpha }^{ \star  }$ 的最大值。定义 ${\Delta }_{1} + {\Delta }_{2}$ 如下：

$$
{B}_{\beta } - {B}_{\alpha }^{ \star  } = {s}_{\alpha ,\min } - {s}_{\beta ,\max } \geq  {B}^{\prime } - {B}^{ \star  } = 2\left( {{\Delta }_{1} + {\Delta }_{2}}\right) .
$$

If ${\Delta }_{1} > 0$ and ${\Delta }_{2} > 0$ ,a hash structure with $L =$ $O\left( {\log \left( \frac{n{m}_{q}m}{\delta }\right) }\right)$ solves the approximate top- $k$ vector set search problem (Definition 5) with probability at least $1 - \delta$ ,where $n = \left| \mathbf{D}\right| ,{m}_{q} = \left| \mathbf{Q}\right| ,\bar{m} = \left| {\mathbf{V}}_{i}\right|$ (Assume $m$ is invariant across all ${\mathbf{V}}_{i}$ ),and $\delta$ is the failure probability.

如果 ${\Delta }_{1} > 0$ 且 ${\Delta }_{2} > 0$ ，一个具有 $L =$ $O\left( {\log \left( \frac{n{m}_{q}m}{\delta }\right) }\right)$ 的哈希结构以至少 $1 - \delta$ 的概率解决近似前 $k$ 向量集搜索问题（定义5），其中 $n = \left| \mathbf{D}\right| ,{m}_{q} = \left| \mathbf{Q}\right| ,\bar{m} = \left| {\mathbf{V}}_{i}\right|$ （假设 $m$ 在所有 ${\mathbf{V}}_{i}$ 上是不变的），且 $\delta$ 是失败概率。

Proof. We aim to determine the value of $L$ that satisfies both tail bounds and combine them to obtain the final $L$ .

证明。我们的目标是确定同时满足两个尾部界限的 $L$ 的值，并将它们结合起来得到最终的 $L$ 。

Upper Tail Probability Bound. For ${\mathbf{V}}_{\beta } \notin$ ${\operatorname{argmin}}_{\mathbf{V} \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ,Lemma 2 yields:

上尾部概率界限。对于 ${\mathbf{V}}_{\beta } \notin$ ${\operatorname{argmin}}_{\mathbf{V} \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ，引理2给出：

$$
\Pr \left\lbrack  {\sigma \left( {\widehat{\mathbf{S}}}_{\beta }\right)  \geq  {s}_{\beta ,\max } + {\Delta }_{1}}\right\rbrack   \leq  {m}_{q}m{\gamma }_{\beta }^{L},
$$

where ${\gamma }_{\beta } = {\left( \frac{{s}_{\beta ,\max }\left( {1 - {\tau }_{1}}\right) }{{\tau }_{1}\left( {1 - {s}_{\beta ,\max }}\right) }\right) }^{{\tau }_{1}}\left( \frac{1 - {s}_{\beta ,\max }}{1 - {\tau }_{1}}\right)$ .

其中 ${\gamma }_{\beta } = {\left( \frac{{s}_{\beta ,\max }\left( {1 - {\tau }_{1}}\right) }{{\tau }_{1}\left( {1 - {s}_{\beta ,\max }}\right) }\right) }^{{\tau }_{1}}\left( \frac{1 - {s}_{\beta ,\max }}{1 - {\tau }_{1}}\right)$ 。

To simplify the analysis,we consider ${\gamma }_{\max } = \mathop{\max }\limits_{{\mathbf{V}}_{\beta }}{\gamma }_{\beta }$ , allowing us to express all bounds using the same ${\gamma }_{\max }$ . To ensure that the probability of the bound holding for all $n - k$ non-top- $k$ sets is $\frac{\delta }{2}$ ,we choose $L$ such that $L \geq$ $\log \frac{2\left( {n - k}\right) {m}_{q}m}{\delta }{\left( \log \frac{1}{{\gamma }_{\max }}\right) }^{-1}$ . Consequently,

为简化分析，我们考虑 ${\gamma }_{\max } = \mathop{\max }\limits_{{\mathbf{V}}_{\beta }}{\gamma }_{\beta }$ ，这样我们就可以用相同的 ${\gamma }_{\max }$ 来表示所有边界。为确保该边界对所有 $n - k$ 个非前 $k$ 集合都成立的概率为 $\frac{\delta }{2}$ ，我们选择 $L$ 使得 $L \geq$ $\log \frac{2\left( {n - k}\right) {m}_{q}m}{\delta }{\left( \log \frac{1}{{\gamma }_{\max }}\right) }^{-1}$ 。因此，

$$
\Pr \left\lbrack  {\sigma \left( {\widehat{\mathbf{S}}}_{\beta }\right)  \geq  {s}_{\beta ,\max } + {\Delta }_{1}}\right\rbrack   \leq  {m}_{q}m{\gamma }_{\beta }^{L} \leq  {m}_{q}m{\gamma }_{\max }^{L}
$$

$$
 \leq  {m}_{q}m{\gamma }_{\max }^{2\left( {n - k}\right) {m}_{q}m}{\left( \log \frac{1}{{\gamma }_{\max }}\right) }^{-1}
$$

$$
 = \frac{\delta }{2\left( {n - k}\right) }\text{.}
$$

Lower Tail Probability Bound. For ${\mathbf{V}}_{\alpha }^{ \star  } \in$ ${\operatorname{argmin}}_{\mathbf{V} \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ,Lemma 3 yields:

下尾概率边界。对于 ${\mathbf{V}}_{\alpha }^{ \star  } \in$ ${\operatorname{argmin}}_{\mathbf{V} \in  \mathbf{D}}^{k}\operatorname{Haus}\left( {\mathbf{Q},\mathbf{V}}\right)$ ，引理3得出：

$$
\Pr \left\lbrack  {\sigma \left( {\widehat{\mathbf{S}}}_{\alpha }\right)  \leq  {s}_{\alpha ,\min } - {\Delta }_{2}}\right\rbrack   \leq  {m}_{q}m{\xi }_{\alpha }^{L},
$$

where ${\xi }_{\alpha } = {\left( \frac{{s}_{\alpha ,\min }\left( {1 - {\tau }_{2}}\right) }{{\tau }_{2}\left( {1 - {s}_{\alpha ,\min }}\right) }\right) }^{{\tau }_{2}}\left( \frac{1 - {s}_{\alpha ,\min }}{1 - {\tau }_{2}}\right)$ .

其中 ${\xi }_{\alpha } = {\left( \frac{{s}_{\alpha ,\min }\left( {1 - {\tau }_{2}}\right) }{{\tau }_{2}\left( {1 - {s}_{\alpha ,\min }}\right) }\right) }^{{\tau }_{2}}\left( \frac{1 - {s}_{\alpha ,\min }}{1 - {\tau }_{2}}\right)$ 。

Following a similar approach as for the upper bound, we ensure that the probability of the bound holding for all top- $k$ sets is $\frac{\delta }{2}$ by selecting $L$ such that $L \geq  \log \frac{{2k}{m}_{q}m}{\delta }{\left( \log \frac{1}{{\xi }_{\max }}\right) }^{-1}$ , where ${\xi }_{\max } = \mathop{\max }\limits_{\alpha }{\xi }_{\alpha }$ . As a result,

采用与上界类似的方法，我们通过选择 $L$ 使得 $L \geq  \log \frac{{2k}{m}_{q}m}{\delta }{\left( \log \frac{1}{{\xi }_{\max }}\right) }^{-1}$ （其中 ${\xi }_{\max } = \mathop{\max }\limits_{\alpha }{\xi }_{\alpha }$ ），确保该边界对所有前 $k$ 集合都成立的概率为 $\frac{\delta }{2}$ 。结果是，

$$
\Pr \left\lbrack  {\sigma \left( {\widehat{\mathbf{S}}}_{\alpha }\right)  \leq  {s}_{\alpha ,\min } - {\Delta }_{2}}\right\rbrack   \leq  {m}_{q}m{\xi }_{\alpha }^{L} \leq  {m}_{q}m{\xi }_{\max }^{L}
$$

$$
 \leq  {m}_{q}m{\xi }_{\max }^{\log \frac{{2k}{m}_{q}m}{\delta }{\left( \log \frac{1}{{\xi }_{\max }}\right) }^{-1}}
$$

$$
 = \frac{\delta }{2k}\text{.}
$$

Combining the Bounds. Let $L =$ $\max \left( {\frac{\log \frac{2\left( {n - k}\right) {m}_{q}m}{\delta }}{\log \left( \frac{1}{{\gamma }_{\max }}\right) },\frac{\log \frac{{2k}{m}_{q}m}{\delta }}{\log \left( \frac{1}{{\xi }_{\max }}\right) }}\right)$ ,we calculate the final $L$ by combining the upper and lower tail probability bounds. Let 1 be an indicator random variable that equals 1 when all the upper and lower tail probability bounds are satisfied, and 0 otherwise. The probability of solving the approximate top- $k$ vector set problem is equivalent to the probability that:

合并边界。设 $L =$ $\max \left( {\frac{\log \frac{2\left( {n - k}\right) {m}_{q}m}{\delta }}{\log \left( \frac{1}{{\gamma }_{\max }}\right) },\frac{\log \frac{{2k}{m}_{q}m}{\delta }}{\log \left( \frac{1}{{\xi }_{\max }}\right) }}\right)$ ，我们通过合并上尾和下尾概率边界来计算最终的 $L$ 。设1为一个指示随机变量，当所有上尾和下尾概率边界都满足时其值为1，否则为0。求解近似前 $k$ 向量集问题的概率等同于以下情况的概率：

$$
\Pr \left( {{\forall }_{\alpha ,\beta }\left( {\widehat{H}\left( {\mathbf{Q},{\mathbf{V}}_{\alpha }^{ \star  }}\right)  - \widehat{H}\left( {\mathbf{Q},{\mathbf{V}}_{\beta }}\right)  < 0}\right) }\right) 
$$

$$
 = \Pr \left( {{\forall }_{\alpha ,\beta }\left( {\min \left( {\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\max }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{\alpha }^{ \star  }}}\left( {\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) ,\mathop{\min }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{\alpha }^{ \star  }}}\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\left( {\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) }\right) }\right. }\right. 
$$

$$
\left. \left. {-\min \left( {\mathop{\min }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\max }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{\beta }}}\left( {\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) ,\mathop{\min }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{\beta }}}\mathop{\max }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\left( {\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) }\right)  > 0}\right) \right) 
$$

$$
 = \Pr \left( {{\forall }_{\alpha ,\beta }\left( {\min \left( {\mathop{\min }\limits_{{1 \leq  i \leq  {m}_{q}}}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{S}_{ij}^{\alpha },\mathop{\min }\limits_{{1 \leq  j \leq  m}}\mathop{\max }\limits_{{1 \leq  i \leq  {m}_{q}}}{S}_{ij}^{\alpha }}\right) }\right) }\right. 
$$

$$
\left. \left. {-\min \left( {\mathop{\min }\limits_{{1 \leq  i \leq  {m}_{q}}}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{S}_{ij}^{\beta },\mathop{\min }\limits_{{1 \leq  j \leq  m}}\mathop{\max }\limits_{{1 \leq  i \leq  {m}_{q}}}{S}_{ij}^{\beta }}\right)  > 0}\right) \right) 
$$

$$
 \geq  \Pr \left( {\forall \alpha ,\beta \left( {\min \left( {\mathop{\min }\limits_{{1 \leq  i \leq  {m}_{q}}}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{S}_{ij}^{\alpha },\mathop{\min }\limits_{{1 \leq  j \leq  m}}\mathop{\max }\limits_{{1 \leq  i \leq  {m}_{q}}}{S}_{ij}^{\alpha }}\right) }\right) }\right. 
$$

$$
\left. {\left. {-\min \left( {\mathop{\min }\limits_{{1 \leq  i \leq  {m}_{q}}}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{S}_{ij}^{\beta },\mathop{\min }\limits_{{1 \leq  j \leq  m}}\mathop{\max }\limits_{{1 \leq  i \leq  {m}_{q}}}{S}_{ij}^{\beta }}\right)  > 0 \mid  \mathbb{1} = 1}\right) \Pr \left( {\mathbb{1} = 1}\right) }\right) 
$$

$$
 \geq  \Pr \left( {{\forall }_{\alpha ,\beta }\left( {\min {S}_{ij}^{\alpha } - {\Delta }_{2} - \left( {\max {S}_{ij}^{\beta } + {\Delta }_{1}}\right)  > 0 \mid  \mathbb{1} = 1}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

$$
 = \Pr \left( {{\forall }_{\alpha ,\beta }\left( {\min {S}_{ij}^{\alpha } - \max {S}_{ij}^{\beta } > {\Delta }_{1} + {\Delta }_{2} \mid  \mathbb{1} = 1}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

$$
 = \Pr \left( {{\forall }_{\alpha ,\beta }\left( {{B}_{\beta } - {B}_{\alpha }^{ \star  } > {\Delta }_{1} + {\Delta }_{2} \mid  \mathbb{1} = 1}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

$$
 \geq  \Pr \left( {{\forall }_{\alpha ,\beta }\left( {{B}^{\prime } - {B}^{ \star  } > {\Delta }_{1} + {\Delta }_{2} \mid  \mathbb{1} = 1}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

$$
 \geq  \Pr \left( {{\forall }_{\alpha ,\beta }\left( {2\left( {{\Delta }_{1} + {\Delta }_{2}}\right)  > {\Delta }_{1} + {\Delta }_{2} \mid  \mathbb{1} = 1}\right) }\right) \Pr \left( {\mathbb{1} = 1}\right) 
$$

$$
 = 1 * \Pr \left( {\mathbb{1} = 1}\right)  \geq  1 - \left( {\left( {n - k}\right)  * \frac{\delta }{2\left( {n - k}\right) } + \frac{\delta }{2k} * k}\right) 
$$

$$
 = 1 - \delta \text{.}
$$

Therefore,with this choice of $L$ ,the algorithm effectively solves the approximate top- $k$ vector set search problem. By eliminating the data-dependent correlation terms ${\gamma }_{\max }$ and ${\xi }_{\max }$ ,and considering the practical scenario where $n \gg  k$ , then $L = O\left( {\log \left( \frac{n{m}_{q}m}{\delta }\right) }\right)$ .

因此，通过这样选择 $L$ ，该算法能有效解决近似前 $k$ 向量集搜索问题。通过消除与数据相关的相关项 ${\gamma }_{\max }$ 和 ${\xi }_{\max }$ ，并考虑 $n \gg  k$ 的实际场景，那么 $L = O\left( {\log \left( \frac{n{m}_{q}m}{\delta }\right) }\right)$ 。

## C. Performance Discussion

## C. 性能讨论

The time complexity of BioVSS is $O\left( {n{m}^{2}L/w}\right)$ ,where $n$ is the number of vector sets, $m$ is the number of vectors per set, $L$ is the binary vector length,and $w$ is the machine word size (typically 32 or 64 bits). The factor $L/w$ represents the number of machine words needed to store each binary vector,allowing for parallel processing of $w$ bits. This results in improved efficiency compared to real-number operations.

BioVSS的时间复杂度为$O\left( {n{m}^{2}L/w}\right)$，其中$n$是向量集的数量，$m$是每个集合中的向量数量，$L$是二进制向量的长度，$w$是机器字长（通常为32位或64位）。因子$L/w$表示存储每个二进制向量所需的机器字数量，允许对$w$位进行并行处理。与实数运算相比，这提高了效率。

<!-- Media -->

<!-- figureText: Insert into Blooms Build a Dual-Layer Cascade Filter & Query Vector Set Database Inverted Index Key : Bloom Location $\rightarrow  \left( {1,6}\right)  \rightarrow  \left( {4,5}\right)$ Insert into Blooms $\rightarrow  \left( {3,7}\right)$ CBFs of Vector Sets Set ID1 DOOOOOOO Set ID2 DODOOOOO $\rightarrow  \left( {2,9}\right)  \rightarrow  \left( {1,5}\right)$ (Set ID, Count Value) BBFs of Vector Sets Sketches of Sets Set ID1 DOOOOOOO Set ID1 0101000 Set ID2 DOOOOOOO Get Filtering By CBF & BBF Inverted Index Refinement Filtering By Sketchs 0 0 1 1 0 0 0 3 1 0 0 0 0 0 Query Sparse Binary CBF BBF Input Query Codes of Set CBF: Count Bloom Filter Get Top-k BBF: Binary Bloom Filter -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_6.jpg?x=911&y=138&w=749&h=563&r=0"/>

Figure 5: Filter Construction and Query Execution in BioVSS++

图5：BioVSS++中的过滤器构建和查询执行

<!-- Media -->

Despite this optimization, BioVSS still requires an exhaustive scan. To address this limitation, we propose enhancing BioVSS with a filter. Next, we will detail this method.

尽管进行了这种优化，BioVSS仍然需要进行穷举扫描。为了解决这一局限性，我们提议为BioVSS添加一个过滤器。接下来，我们将详细介绍这种方法。

## V. ENHANCING BIOVSS VIA CASCADE FILTER

## 五、通过级联过滤器增强BioVSS

We propose BioVSS++, an enhanced method that addresses the linear scan limitation of BioVSS through a Bio-Inspired Dual-Layer Cascade Filter (BioFilter). The fundamental principle of BioVSS++ exploits the structural similarity between BioHash and Bloom filters, both employing sparse structure for vector encoding and set storage. Next, we detail the filter construction and the search execution.

我们提出了BioVSS++，这是一种增强方法，通过生物启发的双层级联过滤器（BioFilter）解决了BioVSS的线性扫描局限性。BioVSS++的基本原理利用了BioHash和布隆过滤器（Bloom filters）之间的结构相似性，两者都采用稀疏结构进行向量编码和集合存储。接下来，我们将详细介绍过滤器的构建和搜索执行过程。

## A. Filter Construction

## A. 过滤器构建

The bio-inspired dual-layer cascade filter forms the basis for efficient search execution in BioVSS++. BioFilter consists of two key components: an inverted index based on count Bloom filters and vector set sketches based on binary Bloom filters. The inverted index leverages the count Bloom filter to construct inverted lists, each storing vector sets based on their count values. The vector set sketches employ a binary Bloom filter for second filtering via Hamming distance, enabling efficient scanning. BioFilter's dual-layer approach significantly enhances search efficiency by reducing the candidate set through successive refinement stages.

生物启发的双层级联过滤器是BioVSS++中高效搜索执行的基础。BioFilter由两个关键组件组成：基于计数布隆过滤器（count Bloom filters）的倒排索引和基于二进制布隆过滤器（binary Bloom filters）的向量集草图。倒排索引利用计数布隆过滤器构建倒排列表，每个列表根据计数值存储向量集。向量集草图使用二进制布隆过滤器通过汉明距离进行二次过滤，实现高效扫描。BioFilter的双层方法通过连续的细化阶段减少候选集，显著提高了搜索效率。

1) Inverted Index Based on Count Bloom Filter: the inverted index serves as the first layer of BioFilter. It accelerates search execution by constructing an inverted index of vector sets to reduce candidate items.

1) 基于计数布隆过滤器的倒排索引：倒排索引是BioFilter的第一层。它通过构建向量集的倒排索引来减少候选项，从而加速搜索执行。

To mitigate the linear scanning limitation of Algorithm 1, we leverage the count Bloom filters to manage vector sets, which form the basis for constructing the inverted index.

为了缓解算法1的线性扫描局限性，我们利用计数布隆过滤器来管理向量集，这是构建倒排索引的基础。

Definition 8 (Count Bloom Filter [5]). Let $\mathrm{V} =$ $\left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2},\ldots ,{\mathbf{v}}_{m}}\right\}$ be a vector set. A count Bloom filter $\mathbf{C} =$ $\left( {{c}_{1},{c}_{2},\ldots ,{c}_{b}}\right)$ for $\mathbf{V}$ is an array of $b$ counters,where each counter ${c}_{i}$ represents the sum of the $i$ -th bits of the sparse

定义8（计数布隆过滤器 [5]）。设$\mathrm{V} =$ $\left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2},\ldots ,{\mathbf{v}}_{m}}\right\}$为一个向量集。$\mathbf{V}$的计数布隆过滤器$\mathbf{C} =$ $\left( {{c}_{1},{c}_{2},\ldots ,{c}_{b}}\right)$是一个包含$b$个计数器的数组，其中每个计数器${c}_{i}$表示稀疏向量的第$i$位的和

<!-- Media -->

Algorithm 3: Gen_Count_Bloom_Filter $\left( {\mathbf{D},{L}_{wta},b}\right)$

算法3：Gen_Count_Bloom_Filter $\left( {\mathbf{D},{L}_{wta},b}\right)$

---

Input: $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ : vector set database, ${L}_{wta} : \#$ of WTA, $b$ :

输入：$\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ ：向量集数据库，${L}_{wta} : \#$ 为全赢者通吃（WTA）算法的参数，$b$ ：

		the length of bloom filters

		布隆过滤器（Bloom filter）的长度

Output: $C$ : count Bloom filters

输出：$C$ ：计数布隆过滤器（Count Bloom filter）

// Generate binary codes: Algorithm 1

// 生成二进制码：算法1

${\mathbf{D}}^{\mathbf{H}} =$ Gen_Binary_Codes $\left( {\mathbf{D},{L}_{wta}}\right)$ ;

${\mathbf{D}}^{\mathbf{H}} =$ 生成二进制码 $\left( {\mathbf{D},{L}_{wta}}\right)$ ；

// Construct Count Bloom Filters

// 构建计数布隆过滤器

$C = {\left\{  {\mathbf{C}}^{\left( j\right) }\right\}  }_{j = 1}^{n}$ ,where ${\mathbf{C}}^{\left( j\right) } \in  {\mathbb{N}}^{b}$ ;

$C = {\left\{  {\mathbf{C}}^{\left( j\right) }\right\}  }_{j = 1}^{n}$ ，其中 ${\mathbf{C}}^{\left( j\right) } \in  {\mathbb{N}}^{b}$ ；

for $j = 1$ to $n$ do

对于从 $j = 1$ 到 $n$ 执行

	${\mathbf{C}}^{\left( j\right) } = \mathop{\sum }\limits_{{\mathbf{h} \in  {\mathbf{H}}_{j}}}\mathbf{h}$ ,where ${\mathbf{H}}_{j} \in  {\mathbf{D}}^{\mathbf{H}}$ ;

	${\mathbf{C}}^{\left( j\right) } = \mathop{\sum }\limits_{{\mathbf{h} \in  {\mathbf{H}}_{j}}}\mathbf{h}$ ，其中 ${\mathbf{H}}_{j} \in  {\mathbf{D}}^{\mathbf{H}}$ ；

return $C$ ;

返回 $C$ ；

---

<!-- Media -->

binary codes of vectors in $\mathbf{V}$ . Formally, ${c}_{i} = \mathop{\sum }\limits_{{j = 1}}^{m}\mathcal{H}{\left( {\mathbf{v}}_{j}\right) }_{i}$ , where $\mathcal{H}{\left( {\mathbf{v}}_{j}\right) }_{i}$ is the $i$ -th bit of the binary code of ${\mathbf{v}}_{j}$ .

$\mathbf{V}$ 中向量的二进制码。形式上，${c}_{i} = \mathop{\sum }\limits_{{j = 1}}^{m}\mathcal{H}{\left( {\mathbf{v}}_{j}\right) }_{i}$ ，其中 $\mathcal{H}{\left( {\mathbf{v}}_{j}\right) }_{i}$ 是 ${\mathbf{v}}_{j}$ 的二进制码的第 $i$ 位。

The count Bloom filter encodes each binary code's frequency across all vectors in the set. Next, we define the inverted index, which is built upon the count Bloom filters.

计数布隆过滤器对集合中所有向量的每个二进制码的频率进行编码。接下来，我们定义基于计数布隆过滤器构建的倒排索引。

Definition 9 (Inverted Index Based on Count Bloom Filter). Let $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ be a database of vector sets. $A$ inverted index based on count bloom filter $\mathbf{I}$ is a data structure that maps each bit position to a sorted list of tuples. For each position $i \in  \{ 1,2,\ldots ,b\} ,\mathbf{I}\left\lbrack  i\right\rbrack$ contains a list of tuples $\left( {j,{c}_{i}^{\left( j\right) }}\right)$ ,where $j$ is the index of the vector set ${\mathbf{V}}_{j} \in  \mathbf{D}$ ,and ${c}_{i}^{\left( j\right) }$ is the $i$ -th count value in the count Bloom filter for ${\mathbf{V}}_{j}$ . The list is sorted in descending order based on ${c}_{i}^{\left( j\right) }$ values.

定义9（基于计数布隆过滤器的倒排索引）。设 $\mathbf{D} = \left\{  {{\mathbf{V}}_{1},{\mathbf{V}}_{2},\ldots ,{\mathbf{V}}_{n}}\right\}$ 为向量集数据库。基于计数布隆过滤器 $\mathbf{I}$ 的 $A$ 倒排索引是一种将每个位位置映射到元组排序列表的数据结构。对于每个位置 $i \in  \{ 1,2,\ldots ,b\} ,\mathbf{I}\left\lbrack  i\right\rbrack$ 包含一个元组列表 $\left( {j,{c}_{i}^{\left( j\right) }}\right)$ ，其中 $j$ 是向量集 ${\mathbf{V}}_{j} \in  \mathbf{D}$ 的索引，并且 ${c}_{i}^{\left( j\right) }$ 是 ${\mathbf{V}}_{j}$ 的计数布隆过滤器中的第 $i$ 个计数值。该列表根据 ${c}_{i}^{\left( j\right) }$ 的值按降序排序。

<!-- Media -->

Algorithm 4: Build_Inverted_Index $\left( {\mathbf{D},{L}_{wta},b}\right)$

算法4：构建倒排索引 $\left( {\mathbf{D},{L}_{wta},b}\right)$

---

Input: $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ : vector set database, ${L}_{wta} : \#$ of WTA, $b$ :

输入：$\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ ：向量集数据库，${L}_{wta} : \#$ 为全赢者通吃（WTA）算法的参数，$b$ ：

		the length of bloom filters

		布隆过滤器（Bloom filter）的长度

Output: I: inverted index

输出：I：倒排索引（inverted index）

// Generate Count Bloom Filters: Algorithm

// 生成计数布隆过滤器：算法

	3

$C =$ Gen_Count_Bloom_Filter $\left( {\mathbf{D},{L}_{wta},b}\right)$ ;

$C =$ 生成计数布隆过滤器 $\left( {\mathbf{D},{L}_{wta},b}\right)$ ;

// Build Inverted Index

// 构建倒排索引

Initialize $\mathbf{I} = {\left\{  {\mathbf{I}}_{i}\right\}  }_{i = 1}^{b}$ ,where ${\mathbf{I}}_{i} = \varnothing$ for $i = 1,\ldots ,b$ ;

初始化 $\mathbf{I} = {\left\{  {\mathbf{I}}_{i}\right\}  }_{i = 1}^{b}$，其中 ${\mathbf{I}}_{i} = \varnothing$ 对于 $i = 1,\ldots ,b$ ;

for $j = 1$ to $n$ do

对于从 $j = 1$ 到 $n$ 执行

	for $i = 1$ to $b$ do

	  对于从 $i = 1$ 到 $b$ 执行

		${c}_{i}^{\left( j\right) } = {\mathbf{C}}^{\left( j\right) }\left\lbrack  i\right\rbrack$

		${\mathbf{I}}_{i} = {\mathbf{I}}_{i} \cup  \left\{  \left( {j,{c}_{i}^{\left( j\right) }}\right) \right\}$

for $i = 1$ to $b$ do

对于从 $i = 1$ 到 $b$ 执行

	Sort ${\mathbf{I}}_{i}$ in descending order by ${c}_{i}^{\left( j\right) }$ ;

	  按 ${c}_{i}^{\left( j\right) }$ 对 ${\mathbf{I}}_{i}$ 进行降序排序;

return I;

返回 I;

---

<!-- Media -->

The inverted index facilitates efficient vector set search. As shown in Figure 5, each vector set is inserted into count Bloom filters. Subsequently, inverted lists are constructed and sorted in descending order based on the count values at various positions in these filters. This approach capitalizes on the principle that higher count values indicate a greater likelihood of collisions at the same positions for similar vector sets.

倒排索引（inverted index）有助于高效的向量集搜索。如图 5 所示，每个向量集都被插入到计数布隆过滤器中。随后，构建倒排列表，并根据这些过滤器中各个位置的计数值进行降序排序。这种方法利用了计数值越高表明相似向量集在相同位置发生冲突的可能性越大这一原理。

Algorithm 3 describes the process of generating count Bloom filters for all vector sets. The input consists of a database $\mathbf{D}$ of $n$ vector sets,the winner-takes-all parameter ${L}_{wta}$ ,and the length $b$ of the bloom filters. The algorithm begins by generating sparse binary codes for the database using the Gen_Binary_Codes function from Algorithm 1

算法 3 描述了为所有向量集生成计数布隆过滤器的过程。输入包括一个包含 $n$ 个向量集的数据库 $\mathbf{D}$、胜者全得参数 ${L}_{wta}$ 以及布隆过滤器的长度 $b$。该算法首先使用算法 1 中的 Gen_Binary_Codes 函数为数据库生成稀疏二进制代码

<!-- Media -->

Algorithm 5: Build_Set_Sketches $\left( {\mathbf{D},{L}_{wta},b}\right)$

算法 5：构建集合草图 $\left( {\mathbf{D},{L}_{wta},b}\right)$

---

Input: $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ : vector set database, ${L}_{wta} : \#$ of WTA, $b$ :

输入：$\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$：向量集数据库，胜者全得（WTA）的 ${L}_{wta} : \#$，$b$：

		the length of bloom filters

		  布隆过滤器的长度

Output: $\mathcal{S} = {\left\{  {\mathbf{S}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ : set sketches

输出：$\mathcal{S} = {\left\{  {\mathbf{S}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$：集合草图

// Generate Binary Codes: Algorithm 1

// 生成二进制代码：算法1

${\mathbf{D}}^{\mathbf{H}} =$ Gen_Binary_Codes $\left( {\mathbf{D},{L}_{wta}}\right)$ ;

${\mathbf{D}}^{\mathbf{H}} =$ 生成二进制代码 $\left( {\mathbf{D},{L}_{wta}}\right)$ ;

// Construct Binary Bloom Filters

// 构建二进制布隆过滤器

$\mathcal{B} = {\left\{  {\mathbf{B}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ ,where ${\mathbf{B}}^{\left( i\right) } \in  \{ 0,1{\} }^{b}$ ;

$\mathcal{B} = {\left\{  {\mathbf{B}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ ，其中 ${\mathbf{B}}^{\left( i\right) } \in  \{ 0,1{\} }^{b}$ ;

for $i = 1$ to $n$ do

从 $i = 1$ 到 $n$ 执行

	${\mathbf{B}}^{\left( i\right) } = \mathop{\bigvee }\limits_{{\mathbf{h} \in  {\mathbf{H}}_{i}}}\mathbf{h}$ ,where ${\mathbf{H}}_{i} \in  {\mathbf{D}}^{\mathbf{H}}$ ; $\;//\; \vee$ is ${OR}$

	${\mathbf{B}}^{\left( i\right) } = \mathop{\bigvee }\limits_{{\mathbf{h} \in  {\mathbf{H}}_{i}}}\mathbf{h}$ ，其中 ${\mathbf{H}}_{i} \in  {\mathbf{D}}^{\mathbf{H}}$ ; $\;//\; \vee$ 是 ${OR}$

// Build Set Sketches

// 构建集合草图

$\mathcal{S} = {\left\{  {\mathbf{S}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ ,where ${\mathbf{S}}^{\left( i\right) } = {\mathbf{B}}^{\left( i\right) }$ ;

$\mathcal{S} = {\left\{  {\mathbf{S}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ ，其中 ${\mathbf{S}}^{\left( i\right) } = {\mathbf{B}}^{\left( i\right) }$ ;

return $\mathcal{S}$ ;

返回 $\mathcal{S}$ ;

---

<!-- Media -->

(line 1). It then constructs count Bloom filters $\mathcal{C}$ for each vector set. Each count Bloom filter ${\mathbf{C}}^{\left( j\right) }$ is created by summing the binary codes of all vectors in the corresponding set ${\mathbf{H}}_{j}$ (line 2-4) and then returns count Bloom filters (line 5).

(第1行)。然后，它为每个向量集构建计数布隆过滤器 $\mathcal{C}$。每个计数布隆过滤器 ${\mathbf{C}}^{\left( j\right) }$ 是通过对相应集合 ${\mathbf{H}}_{j}$ 中所有向量的二进制代码求和来创建的(第2 - 4行)，然后返回计数布隆过滤器(第5行)。

Algorithm 4 describes the process of building an inverted index. The input is the same as algorithm 3 . The output is an inverted index $\mathbf{I}$ . First, count Bloom filters for all data are generated through Algorithm 3 (line 1). For each position $i$ in the count Bloom filters, the algorithm creates an inverted list ${\mathbf{I}}_{i}$ containing pairs of set indices and their corresponding count values (lines 2-6). Finally, each inverted list is sorted in descending order based on the count values (lines 7-8). This inverted index structure effectively reduces the search space.

算法4描述了构建倒排索引的过程。输入与算法3相同。输出是一个倒排索引 $\mathbf{I}$。首先，通过算法3为所有数据生成计数布隆过滤器(第1行)。对于计数布隆过滤器中的每个位置 $i$，该算法创建一个倒排列表 ${\mathbf{I}}_{i}$，其中包含集合索引对及其对应的计数值(第2 - 6行)。最后，每个倒排列表根据计数值按降序排序(第7 - 8行)。这种倒排索引结构有效地减少了搜索空间。

2) Vector Set Sketches Based on Binary Bloom Filter: vector set sketches form the second layer of BioFilter, providing a single binary representation of each vector set. These sketches are based on binary Bloom filters, constructed by applying a bitwise OR operation to the binary codes of all vectors within each set.

2) 基于二进制布隆过滤器的向量集草图：向量集草图构成了BioFilter的第二层，为每个向量集提供单一的二进制表示。这些草图基于二进制布隆过滤器，通过对每个集合内所有向量的二进制代码执行按位或运算来构建。

The binary nature of these sketches enables efficient similarity estimation through Hamming distance, leveraging fast XOR and popcount operations available in modern CPUs [29]. The sketches reduce computational complexity by avoiding the expensive aggregation operations in Hausdorff distance.

这些草图的二进制特性通过汉明距离实现了高效的相似度估计，利用了现代CPU中可用的快速异或和位计数操作[29]。草图通过避免豪斯多夫距离中昂贵的聚合操作降低了计算复杂度。

To better understand the operation of the building set sketches, we first present the binary Bloom filter:

为了更好地理解构建集合草图的操作，我们首先介绍二进制布隆过滤器：

Definition 10 (Binary Bloom Filter [5]). Let $\mathrm{V} =$ $\left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2},\ldots ,{\mathbf{v}}_{m}}\right\}$ be a set of vectors. The binary Bloom filter $\mathbf{B}$ for $\mathbf{V}$ is a binary array of length $b$ ,obtained by performing a bitwise ${OR}$ operation on the binary codes all vectors in $\mathbf{V}$ . Formally, $\mathbf{B} = \mathcal{H}\left( {\mathbf{v}}_{1}\right)  \vee  \mathcal{H}\left( {\mathbf{v}}_{2}\right)  \vee  \ldots  \vee  \mathcal{H}\left( {\mathbf{v}}_{m}\right)$ ,where $\mathcal{H}\left( {\mathbf{v}}_{i}\right)$ is the binary vector of ${\mathbf{v}}_{i}$ and $\vee$ denotes the bitwise OR operation.

定义10(二进制布隆过滤器 [5])。设 $\mathrm{V} =$ $\left\{  {{\mathbf{v}}_{1},{\mathbf{v}}_{2},\ldots ,{\mathbf{v}}_{m}}\right\}$ 为一个向量集。$\mathbf{V}$ 的二进制布隆过滤器 $\mathbf{B}$ 是一个长度为 $b$ 的二进制数组，通过对 $\mathbf{V}$ 中所有向量的二进制代码执行按位 ${OR}$ 操作获得。形式上，$\mathbf{B} = \mathcal{H}\left( {\mathbf{v}}_{1}\right)  \vee  \mathcal{H}\left( {\mathbf{v}}_{2}\right)  \vee  \ldots  \vee  \mathcal{H}\left( {\mathbf{v}}_{m}\right)$ ，其中 $\mathcal{H}\left( {\mathbf{v}}_{i}\right)$ 是 ${\mathbf{v}}_{i}$ 的二进制向量，$\vee$ 表示按位或运算。

As shown in Figure 5, each vector set is inserted into binary Bloom filters. Subsequently, the binary Bloom filter of each vector set is transformed into a vector set sketch.

如图5所示，每个向量集都被插入到二进制布隆过滤器中。随后，每个向量集的二进制布隆过滤器被转换为一个向量集草图。

Algorithm 5 outlines the process of building set sketches in vector set databases. The input consists of a database $\mathbf{D}$ of $n$ vector sets,the winner-takes-all parameter ${L}_{wta}$ ,and the length $b$ of the binary code. The output is a collection of set sketches $\mathcal{S}$ that provide compact representations of the vector sets. The algorithm begins by generating sparse binary codes for the database using the Gen_Binary_Codes function from Algorithm 1 (line 1). It then constructs binary Bloom filters $\mathcal{B}$ for each vector set (lines 2-4). Each binary Bloom filter ${\mathbf{B}}^{\left( i\right) }$ is created by applying a bitwise OR operation ( $\vee$ ) to all binary codes of vectors in the corresponding set ${\mathbf{H}}_{i}$ . Finally,the algorithm builds the set sketches $\mathcal{S}$ (line 5) by directly using the binary Bloom filters as the sketches and then returns $\mathcal{S}$ .

算法5概述了在向量集数据库中构建集合草图（set sketches）的过程。输入包括一个包含 $n$ 个向量集的数据库 $\mathbf{D}$、胜者全得（winner-takes-all）参数 ${L}_{wta}$ 以及二进制代码的长度 $b$。输出是一组集合草图 $\mathcal{S}$，它们为向量集提供了紧凑的表示。该算法首先使用算法1中的Gen_Binary_Codes函数为数据库生成稀疏二进制代码（第1行）。然后，为每个向量集构建二进制布隆过滤器（binary Bloom filters） $\mathcal{B}$（第2 - 4行）。每个二进制布隆过滤器 ${\mathbf{B}}^{\left( i\right) }$ 是通过对相应集合 ${\mathbf{H}}_{i}$ 中所有向量的二进制代码进行按位或运算（ $\vee$ ）而创建的。最后，算法通过直接使用二进制布隆过滤器作为草图来构建集合草图 $\mathcal{S}$（第5行），然后返回 $\mathcal{S}$。

<!-- Media -->

Algorithm 6: BioVSS++_Topk_Search $\left( {\mathbf{Q},k,\mathbf{D},\mathbf{I},\mathcal{S},\Theta }\right)$

算法6：BioVSS++_Topk_Search $\left( {\mathbf{Q},k,\mathbf{D},\mathbf{I},\mathcal{S},\Theta }\right)$

---

	Input: $\mathbf{Q}$ : query vector set, $k : \#$ of results, $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ :

	输入： $\mathbf{Q}$ ：查询向量集， $k : \#$ 个结果， $\mathbf{D} = {\left\{  {\mathbf{V}}_{i}\right\}  }_{i = 1}^{n}$ ：

						vector set database, $\mathbf{I}$ : inverted index, $\mathcal{S} = {\left\{  {\mathbf{S}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ :

						向量集数据库， $\mathbf{I}$ ：倒排索引， $\mathcal{S} = {\left\{  {\mathbf{S}}^{\left( i\right) }\right\}  }_{i = 1}^{n}$ ：

						set sketches, $\Theta  = \{ A$ : access number of lists, $M$ :

						集合草图， $\Theta  = \{ A$ ：列表访问次数， $M$ ：

						minimum count, $c$ : the size of candidate set, $T$ :

						最小计数， $c$ ：候选集的大小， $T$ ：

						$\#$ of candidates, ${L}_{wta} : \#$ of WTA, $b$ :

						 $\#$ 个候选， ${L}_{wta} : \#$ 个胜者全得（WTA）， $b$ ：

						the length of bloom filters $\}$

						布隆过滤器（bloom filters）的长度 $\}$

	Output: $\mathcal{R}$ : top- $k$ vector sets

	输出： $\mathcal{R}$ ：前 $k$ 个向量集

	// Get Query Count Bloom Filter and

	// 获取查询计数布隆过滤器（Count Bloom Filter）和

				Sketch: Algorithm 3 and 5

				草图：算法3和5

	${\mathbf{C}}_{O} =$ Gen_Count_Bloom_Filter $\left( \overline{\mathbf{Q}}\right)$ ;

	${\mathbf{C}}_{O} =$ Gen_Count_Bloom_Filter $\left( \overline{\mathbf{Q}}\right)$ ;

$2{\mathbf{S}}_{O} =$ Build_Set_Sketch $\left( {\mathbf{Q},{L}_{wta},b}\right)$ ;

$2{\mathbf{S}}_{O} =$ Build_Set_Sketch $\left( {\mathbf{Q},{L}_{wta},b}\right)$ ;

	// Filtering by Inverted Index

	// 通过倒排索引进行过滤

	$\pi  =$ ArgsortDescending $\left( {\mathbf{C}}_{O}\right)$ ;

	$\pi  =$ ArgsortDescending $\left( {\mathbf{C}}_{O}\right)$ ;

	$\mathcal{P} = \{ \pi \left( i\right)  : i \in  \left\lbrack  {1,A}\right\rbrack  \}$

	${\mathcal{F}}_{1} = \varnothing ;$

	for $p \in  \mathcal{P}$ do

	for $p \in  \mathcal{P}$ do

				for $\left( {i,{c}_{p}^{\left( i\right) }}\right)  \in  \mathbf{I}\left\lbrack  p\right\rbrack$ do

				对于 $\left( {i,{c}_{p}^{\left( i\right) }}\right)  \in  \mathbf{I}\left\lbrack  p\right\rbrack$ 执行

						if ${c}_{p}^{\left( i\right) } \geq  M$ then

						如果 ${c}_{p}^{\left( i\right) } \geq  M$ 成立，则

								${\mathcal{F}}_{1} = {\mathcal{F}}_{1} \cup  \{ i\}$

	// Filtering by Sketches

	// 通过草图进行过滤

	Initialize $\mathcal{G} \leftarrow$ empty max-heap with capacity $c$ ;

	初始化容量为 $c$ 的空最大堆 $\mathcal{G} \leftarrow$；

	for $i \in  {\mathcal{F}}_{1}$ do

	对于 $i \in  {\mathcal{F}}_{1}$ 执行

				$d = \operatorname{Hamming}\left( {{\mathbf{S}}_{Q},{\mathbf{S}}^{\left( i\right) }}\right)$ ;

				if $\left| \mathcal{G}\right|  < T$ then

				如果 $\left| \mathcal{G}\right|  < T$ 成立，则

						$\mathcal{G}$ .push $\left( \left( {d,i}\right) \right)$ ;

						$\mathcal{G}$ 压入 $\left( \left( {d,i}\right) \right)$；

				else if $d < \mathcal{G}$ .top(   ) [0] then

				否则，如果 $d < \mathcal{G}$ 的顶部元素 [0]，则

						$\mathcal{G} \cdot  \operatorname{pop}\left( \right)$ ;

						$\mathcal{G}$ .push $\left( \left( {d,i}\right) \right)$ ;

						$\mathcal{G}$ 压入 $\left( \left( {d,i}\right) \right)$；

	${\mathcal{F}}_{2} = \{ i \mid  \left( {\_ ,i}\right)  \in  \mathcal{G}\}$

	// Select top- $k$ results

	// 选择前 $k$ 个结果

	$\mathcal{D} = \varnothing$ ;

	for each $\left( {i,{d}_{H}}\right)  \in  \mathcal{F}$ do

	对于每个 $\left( {i,{d}_{H}}\right)  \in  \mathcal{F}$ 执行

				${d}_{i} = \operatorname{Haus}\left( {\mathbf{Q},{\mathbf{V}}_{i}}\right)$ ;

				$\mathcal{D} = \mathcal{D} \cup  \left\{  \left( {{\mathbf{V}}_{i},{d}_{i}}\right) \right\}$

	$\mathbf{R} = \left\{  {\left( {{\mathbf{V}}_{i},{d}_{i}}\right)  \in  \mathcal{D} \mid  {d}_{i} \leq  {d}_{i}^{\left( k\right) }}\right\}$ ,where ${d}_{i}^{\left( k\right) }$ is the $k$ -th

	$\mathbf{R} = \left\{  {\left( {{\mathbf{V}}_{i},{d}_{i}}\right)  \in  \mathcal{D} \mid  {d}_{i} \leq  {d}_{i}^{\left( k\right) }}\right\}$，其中 ${d}_{i}^{\left( k\right) }$ 是第 $k$ 个

		smallest ${d}_{i}$ in $\mathcal{D}$ ;

		$\mathcal{D}$ 中最小的 ${d}_{i}$；

	return $\mathbf{R}$ ;

	返回 $\mathbf{R}$；

---

<!-- Media -->

## B. Search Execution in BioVSS++

## B. BioVSS++ 中的搜索执行

This section introduces the search execution process in BioVSS++. Building upon the inverted index and set sketches, BioFilter can quickly filter unrelated vector sets.

本节介绍 BioVSS++ 中的搜索执行过程。基于倒排索引和集合草图，BioFilter 可以快速过滤不相关的向量集。

The strategy of our search execution lies in a dual-layer filtering mechanism. In the first layer, we employ the inverted index to quickly eliminate a large portion of dissimilar vector sets. In the second layer, we use the vector set sketches for further refinement. This overcomes the limitations of linear scanning in Algorithm 1, reducing the computational overhead.

我们的搜索执行策略基于双层过滤机制。在第一层，我们使用倒排索引（inverted index）快速排除大部分不相似的向量集。在第二层，我们使用向量集草图进行进一步筛选。这克服了算法1中线性扫描的局限性，降低了计算开销。

Algorithm 6 presents BioVSS++ top- $k$ query process, which employs a two-stage filter BioFilter for efficient

算法6展示了BioVSS++的前 $k$ 查询过程，该过程采用了两阶段过滤器BioFilter以实现高效性

<!-- Media -->

TABLE II: Summary of Datasets

表二：数据集总结

<table><tr><td>Dataset</td><td>#of Vectors</td><td>#of Vector Sets</td><td>Dim.</td><td>of Vectors/Set</td></tr><tr><td>CS</td><td>5,553,031</td><td>1,192,792</td><td>384</td><td>$\left\lbrack  {2,{362}}\right\rbrack$</td></tr><tr><td>Medicine</td><td>15,053,338</td><td>2,693,842</td><td>384</td><td>$\left\lbrack  {2,{1923}}\right\rbrack$</td></tr><tr><td>Picture</td><td>2,513,970</td><td>982,730</td><td>512</td><td>$\left\lbrack  {2,9}\right\rbrack$</td></tr></table>

<table><tbody><tr><td>数据集</td><td>向量数量</td><td>向量集数量</td><td>维度</td><td>每个向量集的向量数量</td></tr><tr><td>计算机科学（Computer Science）</td><td>5,553,031</td><td>1,192,792</td><td>384</td><td>$\left\lbrack  {2,{362}}\right\rbrack$</td></tr><tr><td>医学</td><td>15,053,338</td><td>2,693,842</td><td>384</td><td>$\left\lbrack  {2,{1923}}\right\rbrack$</td></tr><tr><td>图片</td><td>2,513,970</td><td>982,730</td><td>512</td><td>$\left\lbrack  {2,9}\right\rbrack$</td></tr></tbody></table>

<!-- Media -->

vector set search. The input includes a query set $\mathbf{Q}$ ,the number of desired results $k$ ,the database $\mathbf{D}$ ,a pre-computed inverted index $\mathbf{I}$ ,set sketches $\mathcal{S}$ ,and parameters $\Theta$ . The output $\mathbf{R}$ contains the top- $k$ most similar vector sets. The algorithm begins by generating a count Bloom filter ${\mathbf{C}}_{Q}$ and a set sketch ${\mathbf{S}}_{Q}$ for the query set (lines 1-2). BioFilter then applies two filtering stages: 1) inverted index filtering (lines 3-9): This stage uses the query's count Bloom filter to identify potential candidates. It selects the top- $A$ positions with the highest counts in ${\mathbf{C}}_{Q}$ and retrieves vector sets from the inverted index $\mathbf{I}$ that have counts above a threshold $M$ at these positions. This produces an initial candidate set ${\mathcal{F}}_{1}$ . 2) sketch-based filtering (lines 10-18): This stage refines the candidates using set sketches. It computes the Hamming distance between the query sketch ${\mathbf{S}}_{Q}$ and each candidate’s sketch ${\mathbf{S}}^{\left( i\right) }$ ,maintaining a max-heap of the $T$ closest candidates. This results in a further refined candidate set ${\mathcal{F}}_{2}$ . Finally,the algorithm computes the actual Hausdorff distance between $\mathbf{Q}$ and each remaining candidate in ${\mathcal{F}}_{2}$ (lines 19-22),selecting the top- $k$ results based on these distances (line 23) and then return it (line 24).

向量集搜索。输入包括查询集 $\mathbf{Q}$、期望结果数量 $k$、数据库 $\mathbf{D}$、预计算的倒排索引 $\mathbf{I}$、集合草图 $\mathcal{S}$ 和参数 $\Theta$。输出 $\mathbf{R}$ 包含前 $k$ 个最相似的向量集。该算法首先为查询集生成一个计数布隆过滤器 ${\mathbf{C}}_{Q}$ 和一个集合草图 ${\mathbf{S}}_{Q}$（第 1 - 2 行）。BioFilter 然后应用两个过滤阶段：1) 倒排索引过滤（第 3 - 9 行）：此阶段使用查询的计数布隆过滤器来识别潜在候选集。它选择 ${\mathbf{C}}_{Q}$ 中计数最高的前 $A$ 个位置，并从倒排索引 $\mathbf{I}$ 中检索在这些位置上计数高于阈值 $M$ 的向量集。这会产生一个初始候选集 ${\mathcal{F}}_{1}$。2) 基于草图的过滤（第 10 - 18 行）：此阶段使用集合草图来优化候选集。它计算查询草图 ${\mathbf{S}}_{Q}$ 和每个候选草图 ${\mathbf{S}}^{\left( i\right) }$ 之间的汉明距离，并维护一个包含 $T$ 个最接近候选集的最大堆。这会得到一个进一步优化的候选集 ${\mathcal{F}}_{2}$。最后，该算法计算 $\mathbf{Q}$ 和 ${\mathcal{F}}_{2}$ 中每个剩余候选集之间的实际豪斯多夫距离（第 19 - 22 行），根据这些距离选择前 $k$ 个结果（第 23 行），然后返回这些结果（第 24 行）。

## C. Discussion on Metrics Extensibility

## C. 指标可扩展性讨论

BioVSS++ algorithm exhibits potential compatibility with diverse set-based distance metrics beyond the Hausdorff distance, owing to the decoupling of the filter structure from the specific distance metric employed. Examples of potentially compatible distance metrics include: 1) set distances based on maximum or minimum point-pair distances [41]; 2) mean aggregate distance [13]; 3) Hausdorff distance variants [7], etc.

由于过滤结构与所采用的特定距离指标解耦，BioVSS++ 算法显示出与豪斯多夫距离之外的各种基于集合的距离指标潜在的兼容性。潜在兼容的距离指标示例包括：1) 基于最大或最小点对距离的集合距离 [41]；2) 平均聚合距离 [13]；3) 豪斯多夫距离变体 [7] 等。

Specifically, the low coupling between distance metrics and the filter is evident in the construction process. Whether building inverted indexes or hash codes, the specific distance metric is not involved. This allows BioVSS++ to be potentially extended to other metrics. Further exploration can be found in Appendix VII-B4 and VII-C

具体而言，距离指标和过滤器之间的低耦合在构建过程中很明显。无论是构建倒排索引还是哈希码，都不涉及特定的距离指标。这使得 BioVSS++ 有可能扩展到其他指标。更多探索可在附录 VII - B4 和 VII - C 中找到。

## VI. EXPERIMENTS

## VI. 实验

## A. Settings

## A. 设置

1) Datasets: the text datasets for this work are sourced from the Microsoft academic graph [39]. We extracted two datasets from the fields of computer science and medicine, including those with at least two first-author papers. The texts from these papers were converted into vectors using the embedding model all-MiniLM-L6-v2 ${}^{1}$ (commonly used). Additionally, we constructed an image dataset using ResNet18 1 for feature extraction. Datasets' details are shown as follows:

1) 数据集：本研究的文本数据集来自微软学术图谱 [39]。我们从计算机科学和医学领域提取了两个数据集，包括那些至少有两篇第一作者论文的数据集。使用嵌入模型 all - MiniLM - L6 - v2 ${}^{1}$（常用模型）将这些论文的文本转换为向量。此外，我们使用 ResNet18 1 进行特征提取构建了一个图像数据集。数据集的详细信息如下：

- Computer Science Literature (CS): It contains 1,192,792 vector sets in the field of computer science,with 5,553,031 vectors. Each set comprises 2 to 362 vectors (Table II).

- 计算机科学文献（CS）：它包含计算机科学领域的 1,192,792 个向量集，共有 5,553,031 个向量。每个集合包含 2 到 362 个向量（表 II）。

---

<!-- Footnote -->

https://www.sbert.net & https://huggingface.co

https://www.sbert.net & https://huggingface.co

<!-- Footnote -->

---

- Medicine Literature (Medicine): It contains 2,693,842 vector sets in the field of Medicine,with 15,053,338 vectors. Each set comprises 2 to 1923 vectors (Table II).

- 医学文献（医学）：它包含医学领域的 2,693,842 个向量集，共有 15,053,338 个向量。每个集合包含 2 到 1923 个向量（表 II）。

- Product Pictures (Picture): It contains 982,730 vector sets sourced from the AliProduct dataset [40], with 2,513,970 vectors. Each set represents 2 to 9 images of the same product, covering 50,000 different products (Table II).

- 产品图片（图片）：它包含来自 AliProduct 数据集 [40] 的 982,730 个向量集，共有 2,513,970 个向量。每个集合代表同一产品的 2 到 9 张图片，涵盖 50,000 种不同产品（表 II）。

2) Baselines: we compare our proposed method against several indexing and quantization techniques. Specifically, we employ methods from the Faiss library [10] developed by Facebook as comparative baselines. The methods evaluated include: 1) IVFFLAT [49], [10], using an inverted file index with flat vectors for efficient high-dimensional data processing; 2) IndexIVFPQ [14], [20], [10], combining inverted file index with product quantization for vector compression and improved query speed; 3) IVFScalarQuantizer [10], employing scalar quantization within the inverted file index to optimize speed-accuracy trade-off; 4) BioVSS++, our proposed method utilizing a bio-inspired dual-layer cascade filter to enhance efficiency through effective pruning.

2) 基线：我们将所提出的方法与几种索引和量化技术进行比较。具体而言，我们采用Facebook开发的Faiss库[10]中的方法作为对比基线。评估的方法包括：1) IVFFLAT [49]、[10]，使用带有扁平向量的倒排文件索引进行高效的高维数据处理；2) IndexIVFPQ [14]、[20]、[10]，将倒排文件索引与乘积量化相结合以进行向量压缩并提高查询速度；3) IVFScalarQuantizer [10]，在倒排文件索引中采用标量量化来优化速度 - 准确率的权衡；4) BioVSS++，我们提出的方法利用受生物启发的双层级联过滤器通过有效剪枝来提高效率。

Due to the absence of efficient direct vector set search methods using Hausdorff distance in high-dimensional spaces, all methods rely on centroid vectors to construct indices.

由于在高维空间中缺乏使用豪斯多夫距离（Hausdorff distance）的高效直接向量集搜索方法，所有方法都依赖质心向量来构建索引。

3) Evaluation Metric: to evaluate the performance of vector set search algorithms, we employ the recall rate at different top- $k$ values as the evaluation metric.

3) 评估指标：为了评估向量集搜索算法的性能，我们采用不同top - $k$值下的召回率作为评估指标。

The recall rate at top- $k$ is defined as follows: Recall $@k =$ $\frac{\left| {R}_{k}\left( \mathbf{Q}\right)  \cap  {G}_{k}\left( \mathbf{Q}\right) \right| }{\left| {G}_{k}\left( \mathbf{Q}\right) \right| }$ ,where $\mathbf{Q}$ denotes a query set, ${R}_{k}\left( \mathbf{Q}\right)$ represents algorithm’s the top- $k$ retrieved results for query $\mathbf{Q}$ ,and ${G}_{k}\left( \mathbf{Q}\right)$ denotes the ground-truth (accurate calculations by Definition 4). We perform 500 queries and report the average recall rate.

top - $k$的召回率定义如下：召回率 $@k =$ $\frac{\left| {R}_{k}\left( \mathbf{Q}\right)  \cap  {G}_{k}\left( \mathbf{Q}\right) \right| }{\left| {G}_{k}\left( \mathbf{Q}\right) \right| }$ ，其中 $\mathbf{Q}$ 表示一个查询集，${R}_{k}\left( \mathbf{Q}\right)$ 表示算法针对查询 $\mathbf{Q}$ 的top - $k$ 检索结果，${G}_{k}\left( \mathbf{Q}\right)$ 表示真实值（根据定义4进行的精确计算）。我们进行500次查询并报告平均召回率。

4) Implementation: Our experiments were conducted on a computing platform with Intel Xeon Platinum 8352V and 512 GB of memory. The core components of our method ${}^{2}$ were implemented in $\mathrm{C} +  +$ and interfaced with Python.

4) 实现：我们的实验在配备英特尔至强铂金8352V处理器和512GB内存的计算平台上进行。我们方法的核心组件 ${}^{2}$ 用 $\mathrm{C} +  +$ 实现并与Python进行接口对接。

5) Default Parameters: In our experiments, we focus on the following parameters: 1) the size of Bloom filter $\{ \underline{1024},{2048}\}$ ; 2) the number of winner-takes-all $\{ {16},{32},{48},\underline{64}\}$ ; 3) the list number of inverted index accessed $\{ 1,2,\underline{3}\} ;4)$ Minimum count value of inverted index $\{ \underline{1},2\} ;5)$ The size of candidate set $\{ {20k},{30k},{40k},{50k}\}$ ; and 6) The number of results returned $\{ 3,5,{10},{15},{20},{25},{30}\}$ . Underlined values denote default parameters in our controlled experiments.

5) 默认参数：在我们的实验中，我们关注以下参数：1) 布隆过滤器（Bloom filter）的大小 $\{ \underline{1024},{2048}\}$ ；2) 胜者全得（winner - takes - all）的数量 $\{ {16},{32},{48},\underline{64}\}$ ；3) 访问的倒排索引列表数量 $\{ 1,2,\underline{3}\} ;4)$ 倒排索引的最小计数值 $\{ \underline{1},2\} ;5)$ 候选集的大小 $\{ {20k},{30k},{40k},{50k}\}$ ；以及6) 返回结果的数量 $\{ 3,5,{10},{15},{20},{25},{30}\}$ 。带下划线的值表示我们对照实验中的默认参数。

## B. Storage and Construction Efficiency of Filter Structures

## B. 过滤器结构的存储和构建效率

BioFilter comprises two sparse data structures: the count Bloom filter and the binary Bloom filter. Both filters exhibit sparsity, necessitating optimized storage strategies.

BioFilter由两种稀疏数据结构组成：计数布隆过滤器（count Bloom filter）和二进制布隆过滤器（binary Bloom filter）。这两种过滤器都具有稀疏性，因此需要优化的存储策略。

The storage optimization leverages two established sparse formats: Coordinate (COO) [4] and Compressed Sparse Row (CSR) [4]. COO format manages dynamic updates through (row, column, value) tuples. CSR format achieves superior compression by maintaining row pointers and column indices, which is particularly effective for static index structures.

存储优化利用了两种已有的稀疏格式：坐标格式（Coordinate, COO）[4]和压缩稀疏行格式（Compressed Sparse Row, CSR）[4]。COO格式通过（行，列，值）元组管理动态更新。CSR格式通过维护行指针和列索引实现了更好的压缩，这对于静态索引结构特别有效。

<!-- Media -->

TABLE III: Filter Storage Comparison on CS Dataset

表III：CS数据集上的过滤器存储比较

<table><tr><td rowspan="2">Bloom</td><td rowspan="2">$L$</td><td colspan="3">Count Bloom Space (GB)</td><td colspan="3">Binary Bloom Space (GB)</td></tr><tr><td>Dense</td><td>COO</td><td>CSR</td><td>Dense</td><td>COO</td><td>CSR</td></tr><tr><td rowspan="4">1024</td><td>16</td><td rowspan="4">9.1</td><td>1.09</td><td>0.55</td><td rowspan="4">1.14</td><td>0.36</td><td>0.19</td></tr><tr><td>32</td><td>2.02</td><td>1.01</td><td>0.67</td><td>0.34</td></tr><tr><td>48</td><td>2.84</td><td>1.43</td><td>0.95</td><td>0.48</td></tr><tr><td>64</td><td>3.59</td><td>1.8</td><td>1.2</td><td>0.6</td></tr><tr><td rowspan="4">2048</td><td>16</td><td rowspan="4">18.2</td><td>1.18</td><td>0.59</td><td rowspan="4">2.28</td><td>0.39</td><td>0.2</td></tr><tr><td>32</td><td>2.2</td><td>1.1</td><td>0.73</td><td>0.37</td></tr><tr><td>48</td><td>3.13</td><td>1.57</td><td>1.04</td><td>0.53</td></tr><tr><td>64</td><td>4</td><td>2</td><td>1.33</td><td>0.67</td></tr></table>

<table><tbody><tr><td rowspan="2">布隆（Bloom）</td><td rowspan="2">$L$</td><td colspan="3">计数布隆空间（GB）（Count Bloom Space (GB)）</td><td colspan="3">二进制布隆空间（GB）（Binary Bloom Space (GB)）</td></tr><tr><td>密集（Dense）</td><td>坐标格式（COO）</td><td>压缩稀疏行格式（CSR）</td><td>密集（Dense）</td><td>坐标格式（COO）</td><td>压缩稀疏行格式（CSR）</td></tr><tr><td rowspan="4">1024</td><td>16</td><td rowspan="4">9.1</td><td>1.09</td><td>0.55</td><td rowspan="4">1.14</td><td>0.36</td><td>0.19</td></tr><tr><td>32</td><td>2.02</td><td>1.01</td><td>0.67</td><td>0.34</td></tr><tr><td>48</td><td>2.84</td><td>1.43</td><td>0.95</td><td>0.48</td></tr><tr><td>64</td><td>3.59</td><td>1.8</td><td>1.2</td><td>0.6</td></tr><tr><td rowspan="4">2048</td><td>16</td><td rowspan="4">18.2</td><td>1.18</td><td>0.59</td><td rowspan="4">2.28</td><td>0.39</td><td>0.2</td></tr><tr><td>32</td><td>2.2</td><td>1.1</td><td>0.73</td><td>0.37</td></tr><tr><td>48</td><td>3.13</td><td>1.57</td><td>1.04</td><td>0.53</td></tr><tr><td>64</td><td>4</td><td>2</td><td>1.33</td><td>0.67</td></tr></tbody></table>

<!-- Media -->

Table III demonstrates the results on CS. With a 1024-size Bloom filter, CSR reduces the storage overhead from 9.1GB to ${0.55}\mathrm{{GB}}$ for the count Bloom filter,achieving a ${94}\%$ reduction ratio. Similar patterns emerge across different parameter settings, with CSR consistently outperforming COO in storage efficiency. Results for Medicine and Picture exhibit analogous characteristics and are detailed in the Appendix VII-B3 Table IV illustrates the time overhead of different stages. While BioHash training constitutes the primary computational cost at 1504s, the construction of count and binary Bloom filters

表三展示了在CS数据集上的实验结果。使用大小为1024的布隆过滤器（Bloom filter）时，CSR（Compressed Sparse Row）将计数布隆过滤器（count Bloom filter）的存储开销从9.1GB降低至${0.55}\mathrm{{GB}}$，实现了${94}\%$的降低率。在不同的参数设置下都出现了类似的模式，CSR在存储效率方面始终优于COO（Coordinate Format）。医学（Medicine）和图片（Picture）数据集的结果也呈现出类似的特征，具体细节见附录VII - B3。表四展示了不同阶段的时间开销。虽然BioHash训练的计算成本最高，达到1504秒，但计数布隆过滤器和二进制布隆过滤器的构建

<!-- Media -->

demonstrates efficiency,requiring only ${24s}$ and ${22s}$ . TABLE IV: Filter Processing Time on CS Dataset

显示出较高的效率，分别仅需${24s}$和${22s}$。表四：CS数据集上的过滤器处理时间

<table><tr><td>Processing Stage</td><td>BioHash Training</td><td>BioHash Hashing</td><td>Count Bloom</td><td>Single Bloom</td></tr><tr><td>Time</td><td>1504s</td><td>14s</td><td>24s</td><td>22s</td></tr></table>

<table><tbody><tr><td>处理阶段</td><td>生物哈希训练</td><td>生物哈希处理</td><td>计数布隆过滤器（Count Bloom）</td><td>单布隆过滤器（Single Bloom）</td></tr><tr><td>时间</td><td>1504s</td><td>14s</td><td>24s</td><td>22s</td></tr></tbody></table>

<!-- Media -->

## C. Performance Analysis and Parameter Experiments

## C. 性能分析与参数实验

In this section, we first conduct preliminary experiments to evaluate the performance of BioVSS and BioVSS++, demonstrating their superiority over brute-force search. We then concentrate analysis on BioVSS++, an enhanced version that incorporates BioFilter for improved efficiency. Finally, we perform extensive parameter tuning studies on BioVSS++ to assess its performance and optimize its configuration.

在本节中，我们首先进行初步实验，以评估BioVSS和BioVSS++的性能，证明它们相对于暴力搜索的优越性。然后，我们将分析重点放在BioVSS++上，这是一个结合了BioFilter以提高效率的增强版本。最后，我们对BioVSS++进行广泛的参数调优研究，以评估其性能并优化其配置。

1) Comparison with Brute-Force Search: We first compare the performance of BioVSS and BioVSS++ against brute-force search in terms of execution time and recall rate on CS, Medicine, and Picture datasets, with a candidate set size of ${20k}$ . As shown in Tables V, VI and VII, both BioVSS terms of speed, while maintaining high recall rates. On CS dataset, BioVSS++ achieves a remarkable 46-fold speedup compared to brute-force search, reducing the total execution time from 9.16 seconds to just 0.20 seconds, while still maintaining high top-3 and top-5 recall rates of ${97.9}\%$ and ${96.2}\%$ . The performance gain is even more pronounced on Medicine dataset, where BioVSS++ demonstrates a 78 times speedup, reducing the execution time from 16.32 seconds to a mere 0.24 seconds. On Picture dataset, BioVSS++ achieves a 44-fold speedup, reducing execution time from 8.75 seconds to 0.20 seconds,with recall rates of ${97.8}\%$ for top-3 and ${96.1}\%$ for top-5. This substantial improvement can be attributed to the filtering mechanism employed by BioVSS++, which effectively addresses the limitation of global scanning present in BioVSS.

1) 与暴力搜索的比较：我们首先在计算机科学（CS）、医学和图片数据集上，以候选集大小为${20k}$，从执行时间和召回率方面比较BioVSS和BioVSS++与暴力搜索的性能。如表V、表VI和表VII所示，BioVSS和BioVSS++在速度方面都显著优于暴力搜索，同时保持了较高的召回率。在计算机科学数据集上，与暴力搜索相比，BioVSS++实现了惊人的46倍加速，将总执行时间从9.16秒减少到仅0.20秒，同时仍保持了${97.9}\%$和${96.2}\%$的高前3和前5召回率。在医学数据集上，性能提升更为明显，BioVSS++实现了78倍的加速，将执行时间从16.32秒减少到仅0.24秒。在图片数据集上，BioVSS++实现了44倍的加速，将执行时间从8.75秒减少到0.20秒，前3召回率为${97.8}\%$，前5召回率为${96.1}\%$。这种显著的改进可归因于BioVSS++采用的过滤机制，该机制有效解决了BioVSS中存在的全局扫描的局限性。

<!-- Media -->

and BioVSS++ significantly outperform brute-force search in TABLE V: Speedup vs. Linear Scan on CS Dataset

并且BioVSS和BioVSS++在表V：计算机科学数据集上的加速比与线性扫描对比中显著优于暴力搜索

<table><tr><td>$\mathbf{{Method}}$</td><td>Total Time (s)</td><td>Speedup</td><td>Top-3 Recall</td><td>Top-5 Recall</td></tr><tr><td>Brute</td><td>9.16</td><td>1x</td><td>100%</td><td>100%</td></tr><tr><td>BioVSS</td><td>0.73</td><td>12x</td><td>97.5%</td><td>97.2%</td></tr><tr><td>BioVSS++</td><td>0.20</td><td>46x</td><td>97.9%</td><td>96.2%</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Method}}$</td><td>总时间（秒）</td><td>加速比</td><td>前3召回率</td><td>前5召回率</td></tr><tr><td>暴力法</td><td>9.16</td><td>1x</td><td>100%</td><td>100%</td></tr><tr><td>生物向量空间搜索（BioVSS）</td><td>0.73</td><td>12x</td><td>97.5%</td><td>97.2%</td></tr><tr><td>生物向量空间搜索++（BioVSS++）</td><td>0.20</td><td>46x</td><td>97.9%</td><td>96.2%</td></tr></tbody></table>

TABLE VI: Speedup vs. Linear Scan on Medicine Dataset

表六：医学数据集上的加速比与线性扫描对比

<table><tr><td>$\mathbf{{Method}}$</td><td>Total Time (s)</td><td>Speedup</td><td>Top-3 Recall</td><td>Top-5 Recall</td></tr><tr><td>Brute</td><td>16.32</td><td>1x</td><td>100%</td><td>100%</td></tr><tr><td>BioVSS</td><td>2.13</td><td>8x</td><td>96.5%</td><td>95.8%</td></tr><tr><td>BioVSS++</td><td>0.24</td><td>78x</td><td>93.8%</td><td>92.3%</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Method}}$</td><td>总时间（秒）</td><td>加速比</td><td>前3召回率</td><td>前5召回率</td></tr><tr><td>暴力法</td><td>16.32</td><td>1x</td><td>100%</td><td>100%</td></tr><tr><td>生物向量空间搜索（BioVSS）</td><td>2.13</td><td>8x</td><td>96.5%</td><td>95.8%</td></tr><tr><td>生物向量空间搜索++（BioVSS++）</td><td>0.24</td><td>78x</td><td>93.8%</td><td>92.3%</td></tr></tbody></table>

TABLE VII: Speedup vs. Linear Scan on Picture Dataset

表七：图片数据集上的加速比与线性扫描对比

<table><tr><td>Method</td><td>Total Time (s)</td><td>Speedup</td><td>Top-3 Recall</td><td>Top-5 Recall</td></tr><tr><td>Brute</td><td>8.75</td><td>1x</td><td>100%</td><td>100%</td></tr><tr><td>BioVSS</td><td>0.49</td><td>17x</td><td>100%</td><td>99.9%</td></tr><tr><td>BioVSS++</td><td>0.20</td><td>44x</td><td>97.8%</td><td>96.1%</td></tr></table>

<table><tbody><tr><td>方法</td><td>总时间（秒）</td><td>加速比</td><td>前3召回率</td><td>前5召回率</td></tr><tr><td>暴力法</td><td>8.75</td><td>1x</td><td>100%</td><td>100%</td></tr><tr><td>生物向量搜索系统（BioVSS）</td><td>0.49</td><td>17x</td><td>100%</td><td>99.9%</td></tr><tr><td>生物向量搜索系统升级版（BioVSS++）</td><td>0.20</td><td>44x</td><td>97.8%</td><td>96.1%</td></tr></tbody></table>

<!-- Media -->

---

<!-- Footnote -->

${}^{2}$ https://github.com/whu-totemdb/biovss

${}^{2}$ https://github.com/whu-totemdb/biovss

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (a) CS Top-3 Search (b) CS Top-5 Search (c) Medicine Top-3 Search (d) Medicine Top-5 Search (e) Picture Top-3 Search (f) Picture Top-5 Search Recall Rate (%) he Number of WT The Number of WTA (d) Medicine Top-5 Search (e) Picture Top-3 Search (f) Picture Top-5 Search Recall Rate (%) 鸡食品加载机电动 The Number of WTA The Number of WTA The Number of WTA Recall Rate (%) Recall Rate (%) The Number of WT/ The Number of WTA The Number of WTI (a) CS Top-3 Search (b) CS Top-5 Search (c) Medicine Top-3 Search Recall Rate (%) Recall Rate (%) The Number of WTA The Number of WTA The Number of WTA Figure 7: Recall 100 (a) CS Top-3 Search (b) CS Top-5 Search Recall Rate (%) $= {1024}$ Recall Rate (%) 98 97 20K 30K 40K 50K 20K 30K 40K 50K The Size of Candidate Set The Size of Candidate Set 98 (c) Medicine Top-3 Search (d) Medicine Top-5 Search Recall Rate (%) $= {1024}$ Recall Rate (%) 92.5 94 20K 30K 40K 50K 20K 30K 40K 50K The Size of Candidate Set The Size of Candidate Set 98.0 (e) Picture Top-3 Search 97 (f) Picture Top-5 Search Recall Rate (%) 97.0 Recall Rate (%) 96.0 93 30K 50K The Size of Candidate Set The Size of Candidate Set Figure 8: Impact of Bloom Filter Size on Recall 0.07 (a) The 1024-Size of Bloom Filter 0.07 (b) The 2048-Size of Bloom Filter Time (seconds) 0.06 0.03 0.02 0.02 0.01 0.01 0.00 0.00 value of the inverted index. Number of Winners-Take-All. The number of winner-takes- all (see Definition 7) in BioVSS++ corresponds to the number of hash functions. As shown in Figures 6 and 7, increasing this parameter from 16 to 48 led to significant performance improvements across CS, Medicine, and Picture datasets for Bloom filter sizes of 1024 and 2048. On CS dataset, with a 1024-size Bloom filter and ${20}\mathrm{k}$ candidates,recall improved by ${5.7}\%$ when increasing the parameter from 16 to 48 . However, the performance gain plateaued between 48 and 64, with only a 0.4% improvement in the same scenario. Similarly, Medicine and Picture datasets exhibit the same trend when increasing the parameter. This trend was consistent across different datasets and Bloom filter sizes. This improvement can be attributed to the enhanced distance-preserving property of the hash code as the number of winner-takes-all increases. Hence, the default number of winner-takes-all is set to 64. Size of the Bloom Filter. The size of the Bloom filter directly influences the length of set sketches and the list number in the inverted index for BioVSS++. As illustrated in Figure 8, our experiments explore filter sizes of 1024 and 2048, revealing that a Bloom filter with a size of 1024 achieves optimal -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_10.jpg?x=133&y=134&w=1456&h=1235&r=0"/>

Figure 9: Filtering Times by the Size of Bloom Filter

图9：布隆过滤器（Bloom Filter）大小对应的过滤时间

<!-- Media -->

2) Parameter Study of BioVSS++: To optimize the performance of BioVSS++, we conducted a comprehensive parameter study. We examined several key parameters: the number of winner-takes-all, the size of the Bloom filter, the list number of inverted index accessed, and the minimum count recall rates across all candidate numbers. On CS dataset, the 1024-size filter configuration yields a 98.9% recall rate with a candidate set of ${50k}$ ,while maintaining a robust ${98}\%$ recall even with a reduced set of ${20k}$ candidates. Similarly,for Medicine and Picture datasets, the 1024-size filter achieves optimal recall rates. The effectiveness of this filter size can be attributed to its capacity to capture discriminative features without over-emphasizing local characteristics, thereby providing a good trade-off between specificity and generalization in this process. As demonstrated in Figure 9, The various sizes of the Bloom filters exhibit low latency below 70 milliseconds. Hence, the default size of the Bloom filter is set to 1024.

2) BioVSS++的参数研究：为了优化BioVSS++的性能，我们进行了全面的参数研究。我们研究了几个关键参数：胜者全得（winner-takes-all）的数量、布隆过滤器（Bloom Filter）的大小、访问的倒排索引列表数量，以及所有候选数量下的最小计数召回率。在计算机科学（CS）数据集上，1024大小的过滤器配置在候选集为${50k}$时可实现98.9%的召回率，即使候选集减少到${20k}$，仍能保持稳健的${98}\%$召回率。同样，对于医学和图片数据集，1024大小的过滤器实现了最佳召回率。这种过滤器大小的有效性可归因于其能够捕捉有区分性的特征，而不过分强调局部特征，从而在这个过程中在特异性和泛化性之间取得了良好的平衡。如图9所示，各种大小的布隆过滤器的延迟均低于70毫秒。因此，布隆过滤器的默认大小设置为1024。

List Number of Inverted Index Accessed. The list number of inverted index accessed determines the search range in BioVSS, with larger values leading to broader searches. As shown in Table VIII, increasing this parameter from 1 to 3 significantly improves recall rates while modestly increasing processing time. For CS dataset with a 1024-size Bloom filter, the top-3 recall improves from 92.9% to 98.9%, and the top-5 recall from 91.1% to 98.2%, with processing time increasing from 0.008s to 0.013s. Similar improvements are observed with a 2048-size Bloom filter. Medicine and Picture datasets show the same trends, with slightly lower recall rates but similar time increases. Notably, the recall rate gain diminishes when moving from 2 to 3 accesses. Hence, the default list number of inverted index accessed is set to 3 .

访问的倒排索引列表数量。访问的倒排索引列表数量决定了BioVSS中的搜索范围，值越大，搜索范围越广。如表VIII所示，将此参数从1增加到3可显著提高召回率，同时适度增加处理时间。对于使用1024大小布隆过滤器的计算机科学（CS）数据集，前3召回率从92.9%提高到98.9%，前5召回率从91.1%提高到98.2%，处理时间从0.008秒增加到0.013秒。使用2048大小布隆过滤器时也观察到了类似的改进。医学和图片数据集显示出相同的趋势，召回率略低，但时间增加情况相似。值得注意的是，从访问2次增加到访问3次时，召回率的提升幅度减小。因此，访问的倒排索引列表的默认数量设置为3。

<!-- Media -->

<!-- figureText: BioVSS++ IVEFLAT IndexIVFPQ IVFScalarQuantize: 100 (d) Medicine Top-5 Recall 100 (e) Picture Top-3 Recal 100 (f) Picture Top-5 Recall Recall Rate (%) Recall Rate (%) 80 Recall Rate (%) -8.2 0.3 0.4 0.5 0.6 0.2 0.3 0.4 0.5 0.2 0.3 0.4 Time (s) Time (s) Time (s) 100 (a) CS Top-3 Recall 100 (b) CS Top-5 Recall 100 (c) Medicine Top-3 Recall 95 Recall Rate (%) 80 75 Recall Rate (%) 90 85 0.2 0.3 0.4 0.5 65 0.3 0.4 0.5 0.5 0.6 Time (s) Time (s) Time (s) -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_11.jpg?x=134&y=142&w=1525&h=280&r=0"/>

Figure 10: Recall Rate Comparison for Different Methods

图10：不同方法的召回率比较

TABLE VIII: List Access Number for Top-3 (T-3) and Top-5 (T-5)

表VIII：前3（T - 3）和前5（T - 5）的列表访问数量

<table><tr><td rowspan="3">Dataset</td><td colspan="3">1 Access</td><td colspan="3">2 Access</td><td colspan="3">3 Access</td></tr><tr><td colspan="2">Recall(%)</td><td rowspan="2">Time (s)</td><td colspan="2">Recall(%)</td><td rowspan="2">Time (s)</td><td>Recall</td><td>(%)</td><td rowspan="2">Time (s)</td></tr><tr><td>T-3</td><td>T-5</td><td>T-3</td><td>T-5</td><td>T-3</td><td>T-5</td></tr><tr><td>CS (1024)</td><td>92.9</td><td>91.1</td><td>.008</td><td>98.0</td><td>97.1</td><td>.012</td><td>98.9</td><td>98.2</td><td>.013</td></tr><tr><td>CS (2048)</td><td>91.9</td><td>89.4</td><td>.008</td><td>97.6</td><td>96.4</td><td>.011</td><td>98.4</td><td>97.3</td><td>.011</td></tr><tr><td>Medicine (1024)</td><td>92.8</td><td>90.8</td><td>.014</td><td>96.9</td><td>95.6</td><td>.018</td><td>97.4</td><td>96.0</td><td>.029</td></tr><tr><td>Medicine (2048)</td><td>90.3</td><td>86.8</td><td>.014</td><td>94.3</td><td>92.2</td><td>.019</td><td>95.5</td><td>93.6</td><td>.028</td></tr><tr><td>Picture (1024)</td><td>85.1</td><td>76.3</td><td>.017</td><td>94.8</td><td>91.2</td><td>.024</td><td>97.9</td><td>96.6</td><td>.030</td></tr><tr><td>Picture (2048)</td><td>81.5</td><td>73.6</td><td>.013</td><td>93.1</td><td>88.0</td><td>.018</td><td>97.0</td><td>94.0</td><td>.023</td></tr></table>

<table><tbody><tr><td rowspan="3">数据集</td><td colspan="3">1次访问</td><td colspan="3">2次访问</td><td colspan="3">3次访问</td></tr><tr><td colspan="2">召回率(%)</td><td rowspan="2">时间 (秒)</td><td colspan="2">召回率(%)</td><td rowspan="2">时间 (秒)</td><td>召回率</td><td>(%)</td><td rowspan="2">时间 (秒)</td></tr><tr><td>T-3</td><td>T-5</td><td>T-3</td><td>T-5</td><td>T-3</td><td>T-5</td></tr><tr><td>CS (1024)</td><td>92.9</td><td>91.1</td><td>.008</td><td>98.0</td><td>97.1</td><td>.012</td><td>98.9</td><td>98.2</td><td>.013</td></tr><tr><td>CS (2048)</td><td>91.9</td><td>89.4</td><td>.008</td><td>97.6</td><td>96.4</td><td>.011</td><td>98.4</td><td>97.3</td><td>.011</td></tr><tr><td>医学 (1024)</td><td>92.8</td><td>90.8</td><td>.014</td><td>96.9</td><td>95.6</td><td>.018</td><td>97.4</td><td>96.0</td><td>.029</td></tr><tr><td>医学 (2048)</td><td>90.3</td><td>86.8</td><td>.014</td><td>94.3</td><td>92.2</td><td>.019</td><td>95.5</td><td>93.6</td><td>.028</td></tr><tr><td>图片 (1024)</td><td>85.1</td><td>76.3</td><td>.017</td><td>94.8</td><td>91.2</td><td>.024</td><td>97.9</td><td>96.6</td><td>.030</td></tr><tr><td>图片 (2048)</td><td>81.5</td><td>73.6</td><td>.013</td><td>93.1</td><td>88.0</td><td>.018</td><td>97.0</td><td>94.0</td><td>.023</td></tr></tbody></table>

TABLE IX: Minimum Count Value with Recall Rate and Time

表九：具有召回率和时间的最小计数值

<table><tr><td rowspan="2">$\mathbf{{Dataset}}$</td><td colspan="3">Minimum Count $= 1$</td><td colspan="3">Minimum Count $= 2$</td></tr><tr><td>Top-3</td><td>Top-5</td><td>Total Time</td><td>Top-3</td><td>Top-5</td><td>Total Time</td></tr><tr><td>CS (1024)</td><td>98.9%</td><td>98.2%</td><td>0.42s</td><td>96.5%</td><td>95.1%</td><td>0.45s</td></tr><tr><td>CS (2048)</td><td>98.4%</td><td>97.3%</td><td>0.45s</td><td>95.1%</td><td>94.0%</td><td>0.44s</td></tr><tr><td>Medicine (1024)</td><td>97.4%</td><td>96.0%</td><td>0.51s</td><td>93.3%</td><td>91.8%</td><td>0.50s</td></tr><tr><td>Medicine (2048)</td><td>95.5%</td><td>93.6%</td><td>0.48s</td><td>89.7%</td><td>87.0%</td><td>0.47s</td></tr><tr><td>Picture (1024)</td><td>97.9%</td><td>96.6%</td><td>0.42s</td><td>92.0%</td><td>87.3%</td><td>0.45s</td></tr><tr><td>Picture (2048)</td><td>97.0%</td><td>94.0%</td><td>0.43s</td><td>87.9%</td><td>81.6%</td><td>0.43s</td></tr></table>

<table><tbody><tr><td rowspan="2">$\mathbf{{Dataset}}$</td><td colspan="3">最小计数 $= 1$</td><td colspan="3">最小计数 $= 2$</td></tr><tr><td>前3名</td><td>前5名</td><td>总时间</td><td>前3名</td><td>前5名</td><td>总时间</td></tr><tr><td>计算机科学（1024）</td><td>98.9%</td><td>98.2%</td><td>0.42s</td><td>96.5%</td><td>95.1%</td><td>0.45s</td></tr><tr><td>计算机科学（2048）</td><td>98.4%</td><td>97.3%</td><td>0.45s</td><td>95.1%</td><td>94.0%</td><td>0.44s</td></tr><tr><td>医学（1024）</td><td>97.4%</td><td>96.0%</td><td>0.51s</td><td>93.3%</td><td>91.8%</td><td>0.50s</td></tr><tr><td>医学（2048）</td><td>95.5%</td><td>93.6%</td><td>0.48s</td><td>89.7%</td><td>87.0%</td><td>0.47s</td></tr><tr><td>图片（1024）</td><td>97.9%</td><td>96.6%</td><td>0.42s</td><td>92.0%</td><td>87.3%</td><td>0.45s</td></tr><tr><td>图片（2048）</td><td>97.0%</td><td>94.0%</td><td>0.43s</td><td>87.9%</td><td>81.6%</td><td>0.43s</td></tr></tbody></table>

<!-- Media -->

Minimum Count Value of Inverted Index. The minimum count value of the inverted index determines the threshold for accessing items in the inverted index lists. As illustrated in Figure IX, this parameter significantly impacts the trade-off between recall. Setting the value to 0 requires traversing all items. A value of 1 leverages the hash codes' sparsity, eliminating most irrelevant items while maintaining high recall. Increasing to 2 further reduces candidates but at the cost of the recall rate. For CS dataset with a 1024-size Bloom filter, the top-3 recall decreases from 98.9% at count 1 to 96.5% at count 2. Top-5 recall shows a similar trend, declining from 98.2% to 95.7%. Medicine and Picture datasets exhibit similar trends, with comparable drops in top-3 and top-5 recall rates. Thus, we set the default minimum count to 1 .

倒排索引的最小计数阈值。倒排索引的最小计数阈值决定了访问倒排索引列表中条目的阈值。如图IX所示，该参数显著影响召回率的权衡。将该值设置为0需要遍历所有条目。设置为1则利用哈希码的稀疏性，在保持高召回率的同时排除了大多数不相关的条目。将其增加到2会进一步减少候选条目，但会降低召回率。对于使用1024大小布隆过滤器的计算机科学（CS）数据集，前3召回率从计数为1时的98.9%降至计数为2时的96.5%。前5召回率也呈现类似趋势，从98.2%降至95.7%。医学和图像数据集也呈现类似趋势，前3和前5召回率有类似的下降。因此，我们将默认的最小计数阈值设置为1。

## D. Comparison Experiment

## D. 对比实验

To evaluate BioVSS++ method, we conducted comparative experiments against baseline algorithms on CS, Medicine, and Picture datasets. We analyzed query time and recall rate. The following presents the results and discussion.

为了评估BioVSS++方法，我们在计算机科学（CS）、医学和图像数据集上针对基线算法进行了对比实验。我们分析了查询时间和召回率。以下是实验结果和讨论。

Figures 10 (a) and (b) present the experimental results on CS dataset. By comparing the trends of the time-recall curves, it is evident that our proposed BioVSS++ algorithm outperforms the other three baseline algorithms. To control the query time, we adjust the size of the candidate set. The experimental results demonstrate that the recall rates of all algorithms increase with the growth of query time. Specifically, On BioVSS++ achieves a recall rate of ${98.9}\%$ in just 0.2 seconds for the top- 3 scenario,and the recall rate further reaches ${98.2}\%$ at 0.47 seconds. Even in the top-5 case, the recall rate of BioVSS++ remains above ${90}\%$ . It is worth noting that the recall rates of IVFScalarQuantizer and IVFFLAT exhibit a significant upward trend as the query time increases, while the upward trend of IndexIVFPQ is relatively less pronounced. We speculate that this may be due to the limitations of the product quantization encoding scheme. Product quantization encoding compresses data by dividing high-dimensional vectors into multiple subvectors and quantizing each subvector. However, this encoding approach may lead to information loss, thereby affecting the room for improvement in recall rate.

图10（a）和（b）展示了在计算机科学（CS）数据集上的实验结果。通过比较时间 - 召回率曲线的趋势，可以明显看出我们提出的BioVSS++算法优于其他三种基线算法。为了控制查询时间，我们调整了候选集的大小。实验结果表明，所有算法的召回率都随着查询时间的增加而提高。具体而言，BioVSS++在0.2秒内的前3场景中实现了${98.9}\%$的召回率，在0.47秒时召回率进一步达到${98.2}\%$。即使在前5的情况下，BioVSS++的召回率仍保持在${90}\%$以上。值得注意的是，随着查询时间的增加，IVFScalarQuantizer和IVFFLAT的召回率呈现出显著的上升趋势，而IndexIVFPQ的上升趋势相对不那么明显。我们推测这可能是由于乘积量化编码方案的局限性所致。乘积量化编码通过将高维向量划分为多个子向量并对每个子向量进行量化来压缩数据。然而，这种编码方法可能会导致信息丢失，从而影响召回率的提升空间。

Figures 10(c) and (d) showcase the query efficiency on Medicine dataset. Due to the larger scale, the query time increases accordingly. The recall rate of the IVFScalarQuantizer algorithm demonstrates a more significant upward trend as the query time increases. This indicates that for large-scale datasets, the IVFScalarQuantizer algorithm may be more sensitive to the candidate set size. Concurrently, this also implicitly reflects that our BioVSS++ algorithm maintains good query performance even with smaller candidate set sizes. Figures 10 (e) and (f) showcase the query efficiency on Picture dataset, exhibiting similar trends to CS dataset.

图10（c）和（d）展示了在医学数据集上的查询效率。由于数据集规模较大，查询时间相应增加。随着查询时间的增加，IVFScalarQuantizer算法的召回率呈现出更显著的上升趋势。这表明对于大规模数据集，IVFScalarQuantizer算法可能对候选集大小更为敏感。同时，这也暗示了我们的BioVSS++算法即使在候选集较小的情况下也能保持良好的查询性能。图10（e）和（f）展示了在图像数据集上的查询效率，其趋势与计算机科学（CS）数据集类似。

## VII. CONCLUSIONS

## VII. 结论

In this paper, we investigated the relatively unexplored problem of vector set search. We introduced BioVSS and BioVSS++ for efficient vector set search using the Hausdorff distance. We provide a theoretical analysis of algorithm correctness. Our dual-layer filter effectively prunes irrelevant items, reducing computational overhead. Experiments show that the proposed method achieves over a 50 times speedup compared to traditional linear scanning methods on million-scale datasets,with a recall rate of up to ${98.9}\%$ . Future work will extend our framework to support more distance metrics.

在本文中，我们研究了相对未被充分探索的向量集搜索问题。我们引入了BioVSS和BioVSS++，用于使用豪斯多夫距离（Hausdorff distance）进行高效的向量集搜索。我们对算法的正确性进行了理论分析。我们的双层过滤器有效地修剪了不相关的条目，减少了计算开销。实验表明，在百万级数据集上，与传统的线性扫描方法相比，所提出的方法实现了超过50倍的加速，召回率高达${98.9}\%$。未来的工作将扩展我们的框架以支持更多的距离度量。

## ACKNOWLEDGEMENT

## 致谢

This work was supported by the National Key R&D Program of China (2023YFB4503600), the National Natural Science Foundation of China (62202338, 62372337), and the Key R&D Program of Hubei Province (2023BAB081)

本工作得到了国家重点研发计划（2023YFB4503600）、国家自然科学基金（62202338，62372337）和湖北省重点研发计划（2023BAB081）的资助。

[1] M. D. Adelfio, S. Nutanong, and H. Samet. Similarity search on a large collection of point sets. In SIGSPATIAL, pages 132-141, 2011.

[2] M. J. Atallah. A linear time algorithm for the hausdorff distance between convex polygons. Information Processing Letters, 17(4):207-209, 1983.

[3] M. Aumüller and M. Ceccarello. Solving k-closest pairs in high-dimensional data. In SISAP, volume 14289, pages 200-214, 2023.

[4] N. Bell and M. Garland. Implementing sparse matrix-vector multiplication on throughput-oriented processors. In SC. ACM, 2009.

[5] F. Bonomi, M. Mitzenmacher, R. Panigrahy, S. Singh, and G. Varghese. An improved construction for counting bloom filters. In ESA, volume 4168, pages 684-695, 2006.

[6] L. Buck and R. Axel. A novel multigene family may encode odorant receptors: a molecular basis for odor recognition. Cell, 65(1):175-187, 1991.

[7] A. Conci and C. S. Kubrusly. Distance between sets-a survey. Advances in Mathematical Sciences and Applications, 26(1):1-18, 2018.

[8] S. Dasgupta, C. F. Stevens, and S. Navlakha. A neural algorithm for a fundamental computing problem. Science, 358(6364):793-796, 2017.

[9] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL- ${HLT}$ ,pages 4171-4186,2019.

[10] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou. The faiss library. CoRR, abs/2401.08281, 2024.

[11] J. Engels, B. Coleman, V. Lakshman, and A. Shrivastava. DESSERT: an efficient algorithm for vector set search with vector set queries. In NeurIPS, 2023.

[12] Y. Fu, C. Chen, X. Chen, W. Wong, and B. He. Optimizing the number of clusters for billion-scale quantization-based nearest neighbor search. IEEE Transactions on Knowledge and Data Engineering, (01):1-14, 2024.

[13] O. Fujita. Metrics based on average distance between sets. Japan Journal of Industrial and Applied Mathematics, 30:1-19, 2013.

[14] T. Ge, K. He, Q. Ke, and J. Sun. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence, 36(4):744- 755, 2014.

[15] G. Gupta, T. Medini, A. Shrivastava, and A. J. Smola. BLISS: A billion scale index using iterative re-partitioning. In SIGKDD, pages 486-495, 2022.

[16] F. Hausdorff. Grundziige der mengenlehre, volume 7. von Veit, 1914.

[17] J. Henrikson. Completeness and total boundedness of the hausdorff metric. MIT Undergraduate Journal of Mathematics, 1(69-80):10, 1999.

[18] P.-S. Huang, X. He, J. Gao, L. Deng, A. Acero, and L. Heck. Learning deep structured semantic models for web search using clickthrough data. In ${CIKM}$ ,pages ${2333} - {2338},{2013}$ .

[19] P. Indyk and R. Motwani. Approximate nearest neighbors: towards removing the curse of dimensionality. In STOC, pages 604-613, 1998.

[20] H. Jegou, M. Douze, and C. Schmid. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128, 2010.

[21] J. Johnson, M. Douze, and H. Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535-547, 2019.

[22] H. W. Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83-97, 1955.

[23] M. Leybovich and O. Shmueli. Efficient approximate search for sets of lineage vectors. In TaPP, pages 5:1-5:8. ACM, 2022.

[24] W. Li, C. Feng, D. Lian, Y. Xie, H. Liu, Y. Ge, and E. Chen. Learning balanced tree indexes for large-scale vector retrieval. In ${KDD}$ ,pages 1353-1362, 2023.

[25] W. Li, Y. Zhang, Y. Sun, W. Wang, M. Li, W. Zhang, and X. Lin. Approximate nearest neighbor search on high dimensional data - experiments, analyses, and improvement. IEEE Transactions on Knowledge and Data Engineering, 32(8):1475-1488, 2020.

[26] V. W. Liang, Y. Zhang, Y. Kwon, S. Yeung, and J. Y. Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. NeurIPS, 35:17612-17625, 2022.

[27] S. X. Luo, R. Axel, and L. Abbott. Generating sparse and selective third-order responses in the olfactory system of the fly. Proceedings of the National Academy of Sciences, 107(23):10713-10718, 2010.

[28] Y. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence,

[28] Y. A. Malkov和D. A. Yashunin。使用分层可导航小世界图进行高效且鲁棒的近似最近邻搜索。《IEEE模式分析与机器智能汇刊》

42(4):824-836, 2020.

[29] W. Muta, N. Kurz, and D. Lemire. Faster population counts using avx2 instructions. The Computer Journal, 61(1):111-120, 2018.

[30] S. Mysore, M. Jasim, A. Mccallum, and H. Zamani. Editable user profiles for controllable text recommendations. In SIGIR, page 993-1003, 2023.

[31] P. Nigam, Y. Song, V. Mohan, V. Lakshman, W. Ding, A. Shingavi, C. H. Teo, H. Gu, and B. Yin. Semantic product search. In SIGKDD, pages 2876-2885, 2019.

[32] S. Nutanong, E. H. Jacox, and H. Samet. An incremental hausdorff distance calculation algorithm. Proceedings of the VLDB Endowment, 4(8):506-517, 2011.

[33] U. Pace, E. Hanski, Y. Salomon, and D. Lancet. Odorant-sensitive adeny-late cyclase may mediate olfactory reception. Nature, 316(6025):255- 258, 1985.

[34] J. Qin, Y. Wang, C. Xiao, W. Wang, X. Lin, and Y. Ishikawa. Gph: Similarity search in hamming space. In ${ICDE}$ ,pages ${29} - {40},{2018}$ .

[35] L. D. Rhein and R. H. Cagan. Biochemical studies of olfaction: isolation, characterization, and odorant binding activity of cilia from rainbow trout olfactory rosettes. Proceedings of the National Academy of Sciences, 77(8):4412-4416, 1980.

[36] G. Roh, J. Roh, S. Hwang, and B. Yi. Supporting pattern-matching queries over trajectories on road networks. IEEE Transactions on Knowledge and Data Engineering, 23(11):1753-1758, 2011.

[37] C. Ryali, J. Hopfield, L. Grinberg, and D. Krotov. Bio-inspired hashing for unsupervised similarity search. In ICML, volume 119, pages 8295- 8306, 2020.

[38] B. Schäfermeier, G. Stumme, and T. Hanika. Mapping research trajectories. CoRR, abs/2204.11859, 2022.

[39] A. Sinha, Z. Shen, Y. Song, H. Ma, D. Eide, B.-J. Hsu, and K. Wang. An overview of microsoft academic service (mas) and applications. In ${WWW}$ ,pages ${243} - {246},{2015}$ .

[40] L. Song, P. Pan, K. Zhao, H. Yang, Y. Chen, Y. Zhang, Y. Xu, and R. Jin. Large-scale training system for 100-million classification at alibaba. In SIGKDD, pages 2909-2930. ACM, 2020.

[41] G. T. Toussaint. An optimal algorithm for computing the minimum vertex distance between two crossing convex polygons. Computing, 32(4):357-364, 1984.

[42] L. B. Vosshall, A. M. Wong, and R. Axel. An olfactory sensory map in the fly brain. Cell, 102(2):147-159, 2000.

[43] M. Wang, H. Wu, X. Ke, Y. Gao, X. Xu, and L. Chen. An interactive multi-modal query answering system with retrieval-augmented large language models. CoRR, abs/2407.04217, 2024.

[44] S. Wang, Z. Bao, J. S. Culpepper, and G. Cong. A survey on trajectory data management, analytics, and learning. ACM Computing Surveys, 54(2):39:1-39:36, 2022.

[45] S. Wang, Z. Bao, J. S. Culpepper, Z. Xie, Q. Liu, and X. Qin. Torch: A search engine for trajectory data. In SIGIR, pages 535-544. ACM, 2018.

[46] S. Wang, Y. Sun, and Z. Bao. On the efficiency of k-means clustering: Evaluation, optimization, and algorithm selection. Proceedings of the VLDB Endowment, 14(2):163-175, 2020.

[47] T. Wei, R. Alkhoury Maroun, Q. Guo, and B. Webb. Devfly: Bio-inspired development of binary connections for locality preserving sparse codes. In NeurIPS, volume 35, pages 2320-2332, 2022.

[48] D. Yin, W. L. Tam, M. Ding, and J. Tang. Mrt: Tracing the evolution of scientific publications. IEEE Transactions on Knowledge and Data Engineering, 35(1):711-724, 2021.

[49] Q. Zhang, S. Xu, Q. Chen, G. Sui, J. Xie, Z. Cai, Y. Chen, Y. He, Y. Yang, F. Yang, M. Yang, and L. Zhou. VBASE: Unifying online vector similarity search and relational queries via relaxed monotonicity. In ${OSDI}$ ,pages ${377} - {395},{2023}$ .

[50] X. Zhao, Y. Tian, K. Huang, B. Zheng, and X. Zhou. Towards efficient index construction and approximate nearest neighbor search in high-dimensional spaces. Proceedings of the VLDB Endowment, 16(8):1979- 1991, 2023.

[51] B. Zheng, Z. Xi, L. Weng, N. Hung, H. Liu, and C. Jensen. Pm-lsh: A fast and accurate lsh framework for high-dimensional approximate nn search. Proceedings of the VLDB Endowment, 13(5):643-655, 2020.

Appendix

附录

A. Suitability Analysis of Hausdorff Distance

A. 豪斯多夫距离的适用性分析

<!-- Media -->

<!-- figureText: Distinguishable(Q to A, B): True Distinguishable(Q to A, B): False Distinguishable(Q to A, B): True $\operatorname{dist}\left( {\mathrm{Q},\mathrm{A}}\right)  \neq  \operatorname{dist}\left( {\mathrm{Q},\mathrm{B}}\right)$ Min Distance MeanMin Distance Bidirectional consistency: True Bidirectional consistency: False Set A Set $\mathrm{B}$ $\operatorname{dist}\left( {\mathrm{Q},\mathrm{A}}\right)  \neq  \operatorname{dist}\left( {\mathrm{Q},\mathrm{B}}\right)$ $\operatorname{dist}\left( {\mathrm{Q},\mathrm{A}}\right)  = \operatorname{dist}\left( {\mathrm{Q},\mathrm{B}}\right)$ Hausdorff Distance Bidirectional consistency: True Set Q -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_13.jpg?x=142&y=234&w=738&h=342&r=0"/>

Figure 11: Comparative Analysis of Different Distance Measures

图11：不同距离度量的对比分析

<!-- Media -->

To demonstrate the advantages of the Hausdorff distance in vector set comparison, a comparative analysis of three distance measures is conducted: minimum distance, mean minimum distance, and Hausdorff distance. The focus is on illustrating the superior precision and symmetry of the Hausdorff distance.

为了证明豪斯多夫距离在向量集比较中的优势，我们对三种距离度量进行了对比分析：最小距离、平均最小距离和豪斯多夫距离。重点是说明豪斯多夫距离在精度和对称性方面的优越性。

Three vector sets $\mathbf{Q},\mathbf{A}$ ,and $\mathbf{B}$ are used for the precision analysis, each containing two vectors. For the symmetry analysis,two vector sets $\mathbf{Q}$ and $\mathbf{A}$ are employed,where $\mathbf{Q}$ contains two vectors and $\mathbf{A}$ contains three vectors. For visualization purposes, all vectors are represented as points in a two-dimensional space. This representation allows for a clear illustration of the distance measures' properties.

三个向量集$\mathbf{Q},\mathbf{A}$和$\mathbf{B}$用于精度分析，每个向量集包含两个向量。对于对称性分析，使用了两个向量集$\mathbf{Q}$和$\mathbf{A}$，其中$\mathbf{Q}$包含两个向量，$\mathbf{A}$包含三个向量。为了便于可视化，所有向量都表示为二维空间中的点。这种表示方式可以清晰地展示距离度量的性质。

The distance measures are defined as follows:

距离度量定义如下：

1) Minimum (Min) distance:

1) 最小（Min）距离：

$$
{d}_{\min }\left( {\mathbf{A},\mathbf{B}}\right)  = \mathop{\min }\limits_{{\mathbf{a} \in  \mathbf{A},\mathbf{b} \in  \mathbf{B}}}d\left( {\mathbf{a},\mathbf{b}}\right) ,
$$

2) Mean Minimum (MeanMin) distance:

2) 平均最小（MeanMin）距离：

$$
{d}_{\text{mean-min }}\left( {\mathbf{A},\mathbf{B}}\right)  = \frac{1}{\left| \mathbf{A}\right| }\mathop{\sum }\limits_{{\mathbf{a} \in  \mathbf{A}}}\mathop{\min }\limits_{{\mathbf{b} \in  \mathbf{B}}}d\left( {\mathbf{a},\mathbf{b}}\right) ,
$$

3) Hausdorff distance: As defined in Definition 4,

3) 豪斯多夫距离（Hausdorff distance）：如定义4中所定义，

Here, $d\left( {\cdot , \cdot  }\right)$ denotes the Euclidean distance,and $\left| \mathbf{A}\right|$ denotes the cardinality of vector set $\mathbf{A}$ .

这里，$d\left( {\cdot , \cdot  }\right)$表示欧几里得距离（Euclidean distance），$\left| \mathbf{A}\right|$表示向量集$\mathbf{A}$的基数。

To evaluate precision,the distances between $\mathbf{Q}$ and vector sets $\mathbf{A}$ and $\mathbf{B}$ are analyzed,as shown in the first row of Figure 11. The specific distance matrices are presented below:

为了评估精度，分析了$\mathbf{Q}$与向量集$\mathbf{A}$和$\mathbf{B}$之间的距离，如图11的第一行所示。具体的距离矩阵如下：

<!-- Media -->

<table><tr><td>QtoA</td><td>${\mathrm{Q}}_{1}$</td><td>${\mathrm{Q}}_{2}$</td></tr><tr><td>${\mathbf{A}}_{1}$</td><td>1</td><td>5</td></tr><tr><td>${\mathbf{A}}_{2}$</td><td>6</td><td>3</td></tr></table>

<table><tbody><tr><td>问答（QtoA）</td><td>${\mathrm{Q}}_{1}$</td><td>${\mathrm{Q}}_{2}$</td></tr><tr><td>${\mathbf{A}}_{1}$</td><td>1</td><td>5</td></tr><tr><td>${\mathbf{A}}_{2}$</td><td>6</td><td>3</td></tr></tbody></table>

<table><tr><td>QtoB</td><td>${\mathbf{Q}}_{1}$</td><td>${\mathrm{Q}}_{2}$</td></tr><tr><td>${\mathbf{B}}_{1}$</td><td>1</td><td>5</td></tr><tr><td>${\mathbf{B}}_{2}$</td><td>4</td><td>1</td></tr></table>

<table><tbody><tr><td>查询到边界（Query to Boundary）</td><td>${\mathbf{Q}}_{1}$</td><td>${\mathrm{Q}}_{2}$</td></tr><tr><td>${\mathbf{B}}_{1}$</td><td>1</td><td>5</td></tr><tr><td>${\mathbf{B}}_{2}$</td><td>4</td><td>1</td></tr></tbody></table>

<!-- Media -->

The analysis yields the following results:

分析得出以下结果：

1) ${d}_{\min }\left( {\mathbf{Q},\mathbf{A}}\right)  = {d}_{\min }\left( {\mathbf{Q},\mathbf{B}}\right)  = 1$

2) ${d}_{\text{mean-min }}\left( {\mathbf{Q},\mathbf{A}}\right)  = 2,{d}_{\text{mean-min }}\left( {\mathbf{Q},\mathbf{B}}\right)  = 1$

3) ${d}_{H}\left( {\mathbf{Q},\mathbf{A}}\right)  = 3,{d}_{H}\left( {\mathbf{Q},\mathbf{B}}\right)  = 2$

These results clearly demonstrate the superior precision of the Hausdorff distance. The minimum distance fails to differentiate between $\mathbf{Q}$ ’s relationships with $\mathbf{A}$ and $\mathbf{B}$ . The mean minimum distance and Hausdorff distance clearly distinguish both the similarities and differences between the vector sets. While both ${d}_{\text{mean } - \text{ min }}$ and ${d}_{H}$ show discriminative power,we further examine their symmetry properties.

这些结果清楚地表明了豪斯多夫距离（Hausdorff distance）具有更高的精度。最小距离无法区分 $\mathbf{Q}$ 与 $\mathbf{A}$ 和 $\mathbf{B}$ 之间的关系。平均最小距离和豪斯多夫距离能够清晰地区分向量集之间的相似性和差异性。虽然 ${d}_{\text{mean } - \text{ min }}$ 和 ${d}_{H}$ 都具有区分能力，但我们进一步研究它们的对称性。

To examine symmetry,a case with two vector sets, $\mathbf{Q}$ and $\mathbf{A}$ ,where $\mathbf{Q}$ contains two vectors and $\mathbf{A}$ contains three vectors, is analyzed. The distance matrix is:

为了研究对称性，我们分析了一个包含两个向量集 $\mathbf{Q}$ 和 $\mathbf{A}$ 的案例，其中 $\mathbf{Q}$ 包含两个向量，$\mathbf{A}$ 包含三个向量。距离矩阵如下：

<!-- Media -->

<table><tr><td>QtoA</td><td>${\mathbf{Q}}_{1}$</td><td>${\mathrm{Q}}_{2}$</td></tr><tr><td>${\mathbf{A}}_{1}$</td><td>1</td><td>4</td></tr><tr><td>${\mathbf{A}}_{2}$</td><td>4</td><td>1</td></tr><tr><td>${\mathbf{A}}_{3}$</td><td>7</td><td>3</td></tr></table>

<table><tbody><tr><td>问答（问题到答案，Question to Answer）</td><td>${\mathbf{Q}}_{1}$</td><td>${\mathrm{Q}}_{2}$</td></tr><tr><td>${\mathbf{A}}_{1}$</td><td>1</td><td>4</td></tr><tr><td>${\mathbf{A}}_{2}$</td><td>4</td><td>1</td></tr><tr><td>${\mathbf{A}}_{3}$</td><td>7</td><td>3</td></tr></tbody></table>

<!-- Media -->

The analysis yields the following results:

分析得出以下结果：

1) ${d}_{\min }\left( {\mathbf{Q},\mathbf{A}}\right)  = {d}_{\min }\left( {\mathbf{A},\mathbf{Q}}\right)  = 1$

2) ${d}_{\text{mean-min }}\left( {\mathbf{Q},\mathbf{A}}\right)  = 1,{d}_{\text{mean-min }}\left( {\mathbf{A},\mathbf{Q}}\right)  = {1.67}$

3) ${d}_{H}\left( {\mathbf{Q},\mathbf{A}}\right)  = {d}_{H}\left( {\mathbf{A},\mathbf{Q}}\right)  = 3$

These results highlight the perfect symmetry of the Hausdorff distance. The minimum distance also exhibits symmetry. However, the mean minimum distance produces different results depending on the direction of comparison. The Hausdorff distance maintains consistency regardless of the order of vector set comparison, ensuring reliable and consistent similarity assessments.

这些结果凸显了豪斯多夫距离（Hausdorff distance）的完美对称性。最小距离也呈现出对称性。然而，平均最小距离会根据比较方向产生不同的结果。无论向量集比较的顺序如何，豪斯多夫距离都能保持一致性，确保了可靠且一致的相似性评估。

The examples demonstrate the superiority of the Hausdorff distance in vector set comparisons. Its advantages are twofold. First, it provides enhanced precision in set differentiation. Second, it maintains consistent symmetry regardless of comparison direction. These properties are not present in other common measures. The Hausdorff distance effectively captures fine-grained differences while ensuring mathematical consistency. This balance makes it particularly suitable for complex vector set comparisons. As a result, it serves as a versatile and reliable metric across various applications.

这些示例展示了豪斯多夫距离在向量集比较中的优越性。它的优势有两方面。首先，它在集合区分方面提供了更高的精度。其次，无论比较方向如何，它都能保持一致的对称性。其他常见度量方法不具备这些特性。豪斯多夫距离在确保数学一致性的同时，能有效捕捉到细微的差异。这种平衡使其特别适用于复杂的向量集比较。因此，它在各种应用中都是一种通用且可靠的度量方法。

<!-- Media -->

B. Supplementary Experiments

B. 补充实验

TABLE X: Impact of Different Embedding Methods

表十：不同嵌入方法的影响

<table><tr><td>$\mathbf{{Dataset}}$</td><td>$\mathbf{{Embedding}}$ Model</td><td>$\mathbf{{Embedding}}$ Dimension</td><td>Recall (%) Top-3</td><td>Top-5</td><td>Time (s)</td></tr><tr><td>CS (1024)</td><td>MiniLM</td><td>384</td><td>98.9</td><td>98.2</td><td>0.42</td></tr><tr><td>CS (1024)</td><td>DistilUse</td><td>512</td><td>92.3</td><td>90.8</td><td>0.46</td></tr><tr><td>Picture (1024)</td><td>ResNet18</td><td>512</td><td>97.9</td><td>96.6</td><td>0.42</td></tr><tr><td>CS (2048)</td><td>MiniLM</td><td>384</td><td>98.4</td><td>97.3</td><td>0.43</td></tr><tr><td>CS (2048)</td><td>DistilUse</td><td>512</td><td>85.2</td><td>81.0</td><td>0.45</td></tr><tr><td>Picture (2048)</td><td>ResNet18</td><td>512</td><td>97.0</td><td>94.0</td><td>0.43</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Dataset}}$</td><td>$\mathbf{{Embedding}}$ 模型</td><td>$\mathbf{{Embedding}}$ 维度</td><td>召回率（%） 前3</td><td>前5</td><td>时间（秒）</td></tr><tr><td>CS（1024）</td><td>迷你语言模型（MiniLM）</td><td>384</td><td>98.9</td><td>98.2</td><td>0.42</td></tr><tr><td>CS（1024）</td><td>蒸馏通用句子编码器（DistilUse）</td><td>512</td><td>92.3</td><td>90.8</td><td>0.46</td></tr><tr><td>图片（1024）</td><td>残差网络18（ResNet18）</td><td>512</td><td>97.9</td><td>96.6</td><td>0.42</td></tr><tr><td>CS（2048）</td><td>迷你语言模型（MiniLM）</td><td>384</td><td>98.4</td><td>97.3</td><td>0.43</td></tr><tr><td>CS（2048）</td><td>蒸馏通用句子编码器（DistilUse）</td><td>512</td><td>85.2</td><td>81.0</td><td>0.45</td></tr><tr><td>图片（2048）</td><td>残差网络18（ResNet18）</td><td>512</td><td>97.0</td><td>94.0</td><td>0.43</td></tr></tbody></table>

<!-- Media -->

This section presents additional experiments and analyses further to validate the performance and versatility of BioVSS. We explore five key aspects: the impact of embedding models on vector representations,the effect of top-k on result quality, storage analysis on Medicine and Picture datasets, exploration of alternative distance metrics, and performance analysis via BioHash iteration.

本节将展示额外的实验并进行进一步分析，以验证生物向量语义搜索（BioVSS）的性能和通用性。我们将探讨五个关键方面：嵌入模型对向量表示的影响、前 k 个结果对结果质量的影响、医学和图片数据集的存储分析、替代距离度量的探索，以及通过生物哈希（BioHash）迭代进行的性能分析。

1) Impact of Embedding Models on Algorithm Performance: to assess the robustness of BioVSS across various vector representations, we conducted experiments using different embedding models. These models inherently produce vectors of varying dimensions, allowing us to evaluate the method's performance across different vector lengths.

1) 嵌入模型对算法性能的影响：为了评估生物向量语义搜索（BioVSS）在各种向量表示下的鲁棒性，我们使用不同的嵌入模型进行了实验。这些模型本质上会生成不同维度的向量，使我们能够评估该方法在不同向量长度下的性能。

The experiments were designed to evaluate two critical aspects of embedding model impact. First, the performance consistency was tested using different embedding models within the same modality but with varying vector dimensions.

实验旨在评估嵌入模型影响的两个关键方面。首先，使用同一模态但向量维度不同的不同嵌入模型测试了性能一致性。

<!-- Media -->

TABLE XI: Recall Rate for Top- $k$ Across Different Datasets

表十一：不同数据集上的前 $k$ 召回率

<table><tr><td>$\mathbf{{Method}}$</td><td>Top-3</td><td>Top-5</td><td>Top-10</td><td>Top-15</td><td>Top-20</td><td>Top-25</td><td>Top-30</td></tr><tr><td colspan="8">CS Dataset</td></tr><tr><td>BioVSS</td><td>98.6%</td><td>98.7%</td><td>98.5%</td><td>98.5%</td><td>98.4%</td><td>98.2%</td><td>98.1%</td></tr><tr><td>BioVSS++</td><td>98.9%</td><td>98.2%</td><td>97.5%</td><td>97.2%</td><td>96.9%</td><td>96.7%</td><td>96.4%</td></tr><tr><td colspan="8">MedicineDataset</td></tr><tr><td>BioVSS</td><td>99.1%</td><td>98.8%</td><td>97.8%</td><td>97.5%</td><td>97.2%</td><td>97.0%</td><td>96.9%</td></tr><tr><td>BioVSS++</td><td>97.4%</td><td>96.0%</td><td>94.8%</td><td>93.9%</td><td>93.2%</td><td>92.7%</td><td>92.3%</td></tr><tr><td colspan="8">PictureDataset</td></tr><tr><td>BioVSS</td><td>100%</td><td>99.9%</td><td>99.8%</td><td>99.7%</td><td>99.6%</td><td>99.6%</td><td>99.5%</td></tr><tr><td>BioVSS++</td><td>99.9%</td><td>99.9%</td><td>90.0%</td><td>80.0%</td><td>85.0%</td><td>80.0%</td><td>73.3%</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Method}}$</td><td>前3名</td><td>前5名</td><td>前10名</td><td>前15名</td><td>前20名</td><td>前25名</td><td>前30名</td></tr><tr><td colspan="8">计算机科学数据集（CS Dataset）</td></tr><tr><td>生物视觉语义搜索（BioVSS）</td><td>98.6%</td><td>98.7%</td><td>98.5%</td><td>98.5%</td><td>98.4%</td><td>98.2%</td><td>98.1%</td></tr><tr><td>生物视觉语义搜索增强版（BioVSS++）</td><td>98.9%</td><td>98.2%</td><td>97.5%</td><td>97.2%</td><td>96.9%</td><td>96.7%</td><td>96.4%</td></tr><tr><td colspan="8">医学数据集（MedicineDataset）</td></tr><tr><td>生物视觉语义搜索（BioVSS）</td><td>99.1%</td><td>98.8%</td><td>97.8%</td><td>97.5%</td><td>97.2%</td><td>97.0%</td><td>96.9%</td></tr><tr><td>生物视觉语义搜索增强版（BioVSS++）</td><td>97.4%</td><td>96.0%</td><td>94.8%</td><td>93.9%</td><td>93.2%</td><td>92.7%</td><td>92.3%</td></tr><tr><td colspan="8">图片数据集（PictureDataset）</td></tr><tr><td>生物视觉语义搜索（BioVSS）</td><td>100%</td><td>99.9%</td><td>99.8%</td><td>99.7%</td><td>99.6%</td><td>99.6%</td><td>99.5%</td></tr><tr><td>生物视觉语义搜索增强版（BioVSS++）</td><td>99.9%</td><td>99.9%</td><td>90.0%</td><td>80.0%</td><td>85.0%</td><td>80.0%</td><td>73.3%</td></tr></tbody></table>

<!-- Media -->

Second, the method's versatility was examined across different modalities while maintaining consistent vector dimensions. CS and Picture datasets represent text and image modalities respectively, enabling these comparative analyses.

其次，在保持向量维度一致的情况下，跨不同模态检验了该方法的通用性。计算机科学（CS）和图片数据集分别代表文本和图像模态，从而能够进行这些比较分析。

To validate the impact of different vector dimensions within the same modality, two text embedding models were applied to CS dataset. All-MiniLM-L6-v2 ${}^{\square }$ and distiluse-base-multilingual-cased-v2 from Hugging Face generate 384- dimensional and 512-dimensional vectors. Table X elucidates the efficacy and robustness of BioVSS across diverse embedding dimensionalities. For CS dataset with a Bloom filter cardinality of 1024, both the 384-dimensional MiniLM and 512-dimensional DistilUse models exhibit exceptional recall performance. MiniLM achieves Top-3 and Top-5 recall rates of 98.9% and 98.2%, while DistilUse demonstrates 92.3% and 90.8%. Significantly, the computational latencies are nearly equivalent irrespective of vector dimensionality, corroborating the dimension-agnostic nature of the search algorithm. The algorithm maintains similarly high recall rates when increasing the Bloom filter size to 2048. Notably, the search process exhibits dimension-invariant computational complexity, ensuring efficient performance across varying dimensional scales.

为了验证同一模态内不同向量维度的影响，将两种文本嵌入模型应用于计算机科学（CS）数据集。来自Hugging Face的All - MiniLM - L6 - v2 ${}^{\square }$和distiluse - base - multilingual - cased - v2分别生成384维向量和512维向量。表X阐明了BioVSS在不同嵌入维度下的有效性和鲁棒性。对于布隆过滤器基数为1024的计算机科学（CS）数据集，384维的MiniLM模型和512维的DistilUse模型均表现出卓越的召回性能。MiniLM的前3和前5召回率分别达到98.9%和98.2%，而DistilUse的前3和前5召回率分别为92.3%和90.8%。值得注意的是，无论向量维度如何，计算延迟几乎相同，这证实了搜索算法与维度无关的特性。当布隆过滤器大小增加到2048时，该算法仍保持相似的高召回率。值得一提的是，搜索过程表现出与维度无关的计算复杂度，确保了在不同维度尺度上的高效性能。

To validate the impact of different modalities with the same vector dimension, embedding models generating 512- dimensional vectors were applied to CS and Picture datasets. Specifically, with a Bloom filter cardinality of 1024, both CS and Picture datasets demonstrate high recall rates ( ${90}\%$ for Top-5), with comparable computational latencies (0.4s+). This performance is maintained when scaling to a size of 2048, evidencing the algorithm's robustness across data modalities.

为了验证相同向量维度下不同模态的影响，将生成512维向量的嵌入模型应用于计算机科学（CS）和图片数据集。具体而言，当布隆过滤器基数为1024时，计算机科学（CS）和图片数据集均显示出较高的召回率（前5召回率为${90}\%$），且计算延迟相当（0.4秒以上）。当规模扩大到2048时，这种性能依然保持，证明了该算法在不同数据模态下的鲁棒性。

These experiments validate that BioVSS's performance remains stable across varying embedding dimensions and data modalities.

这些实验验证了BioVSS的性能在不同的嵌入维度和数据模态下保持稳定。

2) Impact of Top-k on Result Quality: to evaluate the impact of the top- $k$ parameter on result quality,we conducted experiments using default parameters for both BioVSS and BioVSS++ on CS, Medicine, and Picture datasets. Table XI presents the recall rates for various top- $k$ values ranging from 3 to 30. On CS dataset, BioVSS demonstrates consistent performance, maintaining a high recall rate above 98% across all top- $k$ values. BioVSS++ shows slightly higher recall for top-3 (98.9%) but experiences a gradual decrease as $k$ increases, reaching 96.4% for top-30. Medicine dataset reveals a similar trend, with BioVSS maintaining high recall rates (99.1% for top-3,decreasing slightly to ${96.9}\%$ for top-30), while BioVSS++ shows a decline (from 97.5% for top-3 to ${92.3}\%$ for top-30). Picture dataset exhibits a similar trend. The slight performance degradation observed in BioVSS++ for larger $k$ values suggests a trade-off between efficiency and recall, which may be attributed to its more aggressive filtering mechanism. Notably, both methods maintain high recall rates (above ${92}\%$ ) even for larger $k$ values,demonstrating their effectiveness in retrieving relevant results across various retrieval scenarios. The filtering mechanism of BioVSS++ has brought significant improvements in query efficiency. Smaller top- $k$ values are typically more valuable and often yield the most relevant results in many applications. Consequently, we will focus on BioVSS++ in our subsequent experiments.

2) 前k值对结果质量的影响：为了评估前$k$参数对结果质量的影响，我们在计算机科学（CS）、医学和图片数据集上使用BioVSS和BioVSS++的默认参数进行了实验。表XI展示了前$k$值从3到30的各种情况下的召回率。在计算机科学（CS）数据集上，BioVSS表现出稳定的性能，在所有前$k$值下均保持98%以上的高召回率。BioVSS++在前3召回率上略高（98.9%），但随着$k$的增加逐渐下降，在前30时降至96.4%。医学数据集呈现出类似的趋势，BioVSS保持较高的召回率（前3为99.1%，前30时略降至${96.9}\%$），而BioVSS++则有所下降（从前3的97.5%降至前30的${92.3}\%$）。图片数据集也呈现出类似的趋势。BioVSS++在较大$k$值下观察到的轻微性能下降表明在效率和召回率之间存在权衡，这可能归因于其更激进的过滤机制。值得注意的是，即使对于较大的$k$值，两种方法均保持较高的召回率（高于${92}\%$），证明了它们在各种检索场景中检索相关结果的有效性。BioVSS++的过滤机制在查询效率方面带来了显著的改进。较小的前$k$值通常更有价值，并且在许多应用中往往能产生最相关的结果。因此，我们将在后续实验中重点关注BioVSS++。

3) Storage Analysis on Medicine and Picture Datasets: Tables XII and XIII present the storage efficiency analysis on Medicine and Picture datasets. These results corroborate the findings from CS, demonstrating consistent storage reduction patterns across different data domains.

3) 医学和图片数据集的存储分析：表XII和表XIII展示了医学和图片数据集的存储效率分析。这些结果证实了计算机科学（CS）数据集的研究结果，表明在不同数据领域中存在一致的存储减少模式。

<!-- Media -->

TABLE XII: Filter Storage Comparison on Medicine Dataset

表XII：医学数据集的过滤器存储比较

<table><tr><td rowspan="2">Bloom</td><td rowspan="2">$L$</td><td colspan="3">Count Bloom Space (GB)</td><td colspan="3">Binary Bloom Space (GB)</td></tr><tr><td>Dense</td><td>COO</td><td>CSR</td><td>Dense</td><td>COO</td><td>CSR</td></tr><tr><td rowspan="4">1024</td><td>16</td><td rowspan="4">7.5</td><td>0.57</td><td>0.29</td><td rowspan="4">0.94</td><td>0.19</td><td>0.1</td></tr><tr><td>32</td><td>1.13</td><td>0.57</td><td>0.38</td><td>0.19</td></tr><tr><td>48</td><td>1.67</td><td>0.84</td><td>0.56</td><td>0.28</td></tr><tr><td>64</td><td>2.18</td><td>1.1</td><td>0.73</td><td>0.37</td></tr><tr><td rowspan="4">2048</td><td>16</td><td rowspan="4">15</td><td>0.61</td><td>0.31</td><td rowspan="4">1.87</td><td>0.2</td><td>0.11</td></tr><tr><td>32</td><td>1.18</td><td>0.6</td><td>0.39</td><td>0.2</td></tr><tr><td>48</td><td>1.73</td><td>0.87</td><td>0.58</td><td>0.29</td></tr><tr><td>64</td><td>2.29</td><td>1.15</td><td>0.76</td><td>0.38</td></tr></table>

<table><tbody><tr><td rowspan="2">布隆（Bloom）</td><td rowspan="2">$L$</td><td colspan="3">计数布隆空间（GB）（Count Bloom Space (GB)）</td><td colspan="3">二进制布隆空间（GB）（Binary Bloom Space (GB)）</td></tr><tr><td>密集（Dense）</td><td>坐标格式（COO）</td><td>压缩稀疏行格式（CSR）</td><td>密集（Dense）</td><td>坐标格式（COO）</td><td>压缩稀疏行格式（CSR）</td></tr><tr><td rowspan="4">1024</td><td>16</td><td rowspan="4">7.5</td><td>0.57</td><td>0.29</td><td rowspan="4">0.94</td><td>0.19</td><td>0.1</td></tr><tr><td>32</td><td>1.13</td><td>0.57</td><td>0.38</td><td>0.19</td></tr><tr><td>48</td><td>1.67</td><td>0.84</td><td>0.56</td><td>0.28</td></tr><tr><td>64</td><td>2.18</td><td>1.1</td><td>0.73</td><td>0.37</td></tr><tr><td rowspan="4">2048</td><td>16</td><td rowspan="4">15</td><td>0.61</td><td>0.31</td><td rowspan="4">1.87</td><td>0.2</td><td>0.11</td></tr><tr><td>32</td><td>1.18</td><td>0.6</td><td>0.39</td><td>0.2</td></tr><tr><td>48</td><td>1.73</td><td>0.87</td><td>0.58</td><td>0.29</td></tr><tr><td>64</td><td>2.29</td><td>1.15</td><td>0.76</td><td>0.38</td></tr></tbody></table>

TABLE XIII: Filter Storage Comparison on Picture Dataset

表十三：图片数据集上的过滤器存储比较

<table><tr><td rowspan="2">Bloom</td><td rowspan="2">$L$</td><td colspan="3">Count Bloom Space (GB)</td><td colspan="3">Binary Bloom Space (GB)</td></tr><tr><td>Dense</td><td>COO</td><td>CSR</td><td>Dense</td><td>COO</td><td>CSR</td></tr><tr><td rowspan="4">1024</td><td>16</td><td rowspan="4">20.55</td><td>2.79</td><td>1.41</td><td rowspan="4">2.57</td><td>0.93</td><td>0.48</td></tr><tr><td>32</td><td>5.14</td><td>2.58</td><td>1.71</td><td>0.87</td></tr><tr><td>48</td><td>7.26</td><td>3.64</td><td>2.42</td><td>1.22</td></tr><tr><td>64</td><td>9.22</td><td>4.62</td><td>3.07</td><td>1.55</td></tr><tr><td rowspan="4">2048</td><td>16</td><td rowspan="4">41.1</td><td>3.01</td><td>1.51</td><td rowspan="4">5.14</td><td>1</td><td>0.51</td></tr><tr><td>32</td><td>5.6</td><td>2.81</td><td>1.87</td><td>0.94</td></tr><tr><td>48</td><td>8.02</td><td>4.02</td><td>2.67</td><td>1.35</td></tr><tr><td>64</td><td>10.71</td><td>5.37</td><td>3.57</td><td>1.8</td></tr></table>

<table><tbody><tr><td rowspan="2">布隆（Bloom）</td><td rowspan="2">$L$</td><td colspan="3">布隆计数空间（GB）（Count Bloom Space (GB)）</td><td colspan="3">二进制布隆空间（GB）（Binary Bloom Space (GB)）</td></tr><tr><td>密集（Dense）</td><td>坐标格式（COO）</td><td>压缩稀疏行格式（CSR）</td><td>密集（Dense）</td><td>坐标格式（COO）</td><td>压缩稀疏行格式（CSR）</td></tr><tr><td rowspan="4">1024</td><td>16</td><td rowspan="4">20.55</td><td>2.79</td><td>1.41</td><td rowspan="4">2.57</td><td>0.93</td><td>0.48</td></tr><tr><td>32</td><td>5.14</td><td>2.58</td><td>1.71</td><td>0.87</td></tr><tr><td>48</td><td>7.26</td><td>3.64</td><td>2.42</td><td>1.22</td></tr><tr><td>64</td><td>9.22</td><td>4.62</td><td>3.07</td><td>1.55</td></tr><tr><td rowspan="4">2048</td><td>16</td><td rowspan="4">41.1</td><td>3.01</td><td>1.51</td><td rowspan="4">5.14</td><td>1</td><td>0.51</td></tr><tr><td>32</td><td>5.6</td><td>2.81</td><td>1.87</td><td>0.94</td></tr><tr><td>48</td><td>8.02</td><td>4.02</td><td>2.67</td><td>1.35</td></tr><tr><td>64</td><td>10.71</td><td>5.37</td><td>3.57</td><td>1.8</td></tr></tbody></table>

<!-- Media -->

4) Exploration of Alternative Distance Metrics: while the main text demonstrates the effectiveness of BioVSS++ using Hausdorff distance, the framework's applicability extends to other set-based distance metrics. To further validate this extensibility, additional experiments were conducted using alternative distance measures.

4) 替代距离度量的探索：虽然正文展示了BioVSS++使用豪斯多夫距离（Hausdorff distance）的有效性，但该框架的适用性可扩展到其他基于集合的距离度量。为了进一步验证这种可扩展性，使用替代距离度量进行了额外的实验。

Our experiments compared BioVSS++ against DESSERT ${}^{3}$ under various parameter configurations (Table XIV). Using the MeanMin distance metric, BioVSS++ achieved a Top-3 recall of ${59.0}\%$ with a query time of 0.46 seconds. These results demonstrate reasonable efficiency beyond the Hausdorff setting. The performance disparity between Hausdorff and MeanMin distances arises from their aggregation mechanisms. Hausdorff utilizes a three-level structure (min- max- max), whereas MeanMin employs a two-level aggregation (min-mean). Such three-level aggregation creates more distinctive distance distributions. Consequently, similar sets exhibit higher collision probabilities, while dissimilar sets are more effectively separated.

我们的实验在各种参数配置下（表十四）将BioVSS++与DESSERT ${}^{3}$进行了比较。使用MeanMin距离度量，BioVSS++在查询时间为0.46秒的情况下实现了${59.0}\%$的前3召回率。这些结果表明，除了豪斯多夫设置之外，该方法也具有合理的效率。豪斯多夫距离和MeanMin距离之间的性能差异源于它们的聚合机制。豪斯多夫距离采用三级结构（最小 - 最大 - 最大），而MeanMin采用两级聚合（最小 - 均值）。这种三级聚合产生了更独特的距离分布。因此，相似集合的碰撞概率更高，而不相似集合则能更有效地分离。

---

<!-- Footnote -->

${}^{3}$ https://github.com/ThirdAIResearch/Dessert.Without IVF indexing.

${}^{3}$ https://github.com/ThirdAIResearch/Dessert.不使用IVF索引。

<!-- Footnote -->

---

<!-- Media -->

TABLE XIV: Performance Comparison of MeanMin

表十四：MeanMin的性能比较

<table><tr><td>Method</td><td>Top-3</td><td>Top-5</td><td>Time</td></tr><tr><td>DESSERT (tables=32, hashes_per_t=6)</td><td>45.9%</td><td>35.8%</td><td>0.21s</td></tr><tr><td>DESSERT (tables=32, hashes_per_t=12)</td><td>40.3%</td><td>28.8%</td><td>0.56s</td></tr><tr><td>DESSERT (tables=24, hashes_per_t=6)</td><td>42.3%</td><td>32.6%</td><td>0.18s</td></tr><tr><td>DESSERT (num_t=24, hashes_per_t=12)</td><td>38.7%</td><td>27.6%</td><td>0.36s</td></tr><tr><td>BioVSS++ (default_parameter)</td><td>59.0%</td><td>51.6%</td><td>0.46s</td></tr></table>

<table><tbody><tr><td>方法</td><td>前3名</td><td>前5名</td><td>时间</td></tr><tr><td>DESSERT（表数量=32，每个表的哈希数=6）</td><td>45.9%</td><td>35.8%</td><td>0.21s</td></tr><tr><td>DESSERT（表数量=32，每个表的哈希数=12）</td><td>40.3%</td><td>28.8%</td><td>0.56s</td></tr><tr><td>DESSERT（表数量=24，每个表的哈希数=6）</td><td>42.3%</td><td>32.6%</td><td>0.18s</td></tr><tr><td>DESSERT（表数量=24，每个表的哈希数=12）</td><td>38.7%</td><td>27.6%</td><td>0.36s</td></tr><tr><td>BioVSS++（默认参数）</td><td>59.0%</td><td>51.6%</td><td>0.46s</td></tr></tbody></table>

<!-- figureText: CS (Bloom Size = 1024) CS (Bloom Size = 2048) Medicine (Bloom Size = 2048) 1000 600 400 200 Picture (Bloom Size = 2048) 20000 15000 10000 5000 Parameter Update Magnitude 1200 1000 800 600 400 200 Medicine (Bloom Size = 1024) Parameter Update Magnitude 1000 800 400 Batch Picture (Bloom Size = 1024) 30000 150 Batch -->

<img src="https://cdn.noedgeai.com/0195dc0e-f36c-73d3-88fd-0fd3c6b1565d_15.jpg?x=143&y=413&w=760&h=703&r=0"/>

Figure 12: Parameter Update Magnitude Across Batches in BioHash

图12：BioHash中各批次的参数更新幅度

<!-- Media -->

The extensibility to different metrics stems from BioVSS++'s decoupled design. The filter structure operates independently of the specific distance metric. This architectural choice enables metric flexibility while maintaining the core filtering mechanisms.

对不同度量标准的可扩展性源于BioVSS++的解耦设计。过滤器结构独立于特定的距离度量标准运行。这种架构选择在保持核心过滤机制的同时，实现了度量标准的灵活性。

5) Performance Analysis via BioHash Iteration: BioHash algorithm serves as a crucial component in our framework, with its reliability directly impacting system performance. While BioHash has been validated in previous research [37], a rigorous examination of its behavior within our specific application domain is necessary.

5) 通过BioHash迭代进行性能分析：BioHash算法是我们框架中的关键组件，其可靠性直接影响系统性能。虽然BioHash在先前的研究[37]中已得到验证，但有必要对其在我们特定应用领域中的行为进行严格检查。

The iteration count parameter plays a vital role in BioHash's computational process, specifically in determining learning efficacy. At its core, BioHash implements a normalized gradient descent optimization approach. The magnitude of parameter updates functions as a key metric for quantifying learning dynamics. This magnitude is defined as:

迭代次数参数在BioHash的计算过程中起着至关重要的作用，特别是在确定学习效率方面。从本质上讲，BioHash实现了一种归一化梯度下降优化方法。参数更新的幅度是量化学习动态的关键指标。该幅度定义为：

$$
{M}_{t} = \mathop{\max }\limits_{{i,j}}\left| {\Delta {W}_{ij}^{t}}\right| 
$$

where ${M}_{t}$ represents the update magnitude at iteration $t$ ,and $\Delta {W}_{ij}^{t}$ denotes the weight change for the synaptic connection between neurons $i$ and $j$ . This metric effectively captures the most substantial parametric modifications occurring within the network during each training iteration (batch_size $= {10k}$ ).

其中${M}_{t}$表示迭代$t$时的更新幅度，$\Delta {W}_{ij}^{t}$表示神经元$i$和$j$之间突触连接的权重变化。该指标有效地捕捉了每次训练迭代（批量大小$= {10k}$）期间网络中发生的最显著的参数修改。

Figure 12 illustrates the parameter update magnitude dynamics across three distinct datasets (CS, Medicine, and Picture) under varying Bloom filter sizes (1024 and 2048 bits). The experimental results reveal several significant patterns:

图12展示了在不同的布隆过滤器大小（1024位和2048位）下，三个不同数据集（计算机科学、医学和图片）的参数更新幅度动态。实验结果揭示了几个显著的模式：

- Convergence Behavior: Across all configurations, the parameter update magnitude exhibits a consistent decay pattern. The parameter updates show high magnitudes during initial batches, followed by a rapid decrease and eventual stabilization. This pattern indicates BioHash algorithm's robust convergence properties regardless of the domain-specific data characteristics.

- 收敛行为：在所有配置中，参数更新幅度呈现出一致的衰减模式。在初始批次中，参数更新幅度较大，随后迅速下降并最终趋于稳定。这种模式表明，无论特定领域的数据特征如何，BioHash算法都具有强大的收敛特性。

- Filter Size Consistency: The comparison between 1024 and 2048 Bloom filter configurations demonstrates remarkable consistency in convergence patterns. This observation suggests that BioHash maintains stable performance characteristics independent of filter size, validating the robustness of our parameter update mechanism across different capacity settings.

- 过滤器大小一致性：1024位和2048位布隆过滤器配置之间的比较显示出收敛模式的显著一致性。这一观察结果表明，BioHash的性能特征不受过滤器大小的影响，验证了我们的参数更新机制在不同容量设置下的鲁棒性。

- Cross-Domain Consistency: The similar convergence patterns observed across CS, Medicine, and Picture datasets validate the algorithm's domain-agnostic nature. Despite the inherent differences in data distributions, BioHash iteration mechanism maintains consistent performance characteristics.

- 跨领域一致性：在计算机科学、医学和图片数据集上观察到的相似收敛模式验证了该算法的领域无关性。尽管数据分布存在固有差异，但BioHash迭代机制保持了一致的性能特征。

Through extensive empirical analysis across diverse application scenarios, we validated BioHash's effectiveness as a core component of our system. Our experiments demonstrated consistent and reliable convergence behavior under various environmental configurations. The results confirmed that BioHash achieves stable parameter updates well before completing the full training process.

通过对各种应用场景进行广泛的实证分析，我们验证了BioHash作为我们系统核心组件的有效性。我们的实验表明，在各种环境配置下，BioHash具有一致且可靠的收敛行为。结果证实，BioHash在完成整个训练过程之前就实现了稳定的参数更新。

6) Query Time Analysis: experiments evaluated query efficiency across candidate sets ranging from ${20}\mathrm{k}$ to ${50}\mathrm{k}$ . As illustrated in Table XV, query time scales approximately linearly with candidate count. With the configuration of WTA=64 and Bloom filter size $= {1024}$ ,the method achieves consistent and efficient query times (0.44s-0.51s for ${50}\mathrm{k}$ candidates) across all datasets. While reducing WTA hash count to 16 decreases query time by up to ${15}\%$ ,such reduction compromises recall performance as shown in Figures 6 and 7. Additionally, doubling the Bloom filter size to2048offers minimal efficiency gains ( $\leq  {0.03s}$ improvement),making the added memory overhead unjustifiable. These observations support selecting 64 and 1024 as the optimal configurations balancing efficiency and effectiveness.

6) 查询时间分析：实验评估了候选集范围从${20}\mathrm{k}$到${50}\mathrm{k}$的查询效率。如表XV所示，查询时间与候选数量大致呈线性关系。在WTA = 64和布隆过滤器大小为$= {1024}$的配置下，该方法在所有数据集上实现了一致且高效的查询时间（对于${50}\mathrm{k}$个候选，查询时间为0.44秒 - 0.51秒）。虽然将WTA哈希计数减少到16可使查询时间最多减少${15}\%$，但如图6和图7所示，这种减少会影响召回性能。此外，将布隆过滤器大小加倍至2048位仅带来极小的效率提升（$\leq  {0.03s}$的改进），使得增加的内存开销不合理。这些观察结果支持选择64和1024作为平衡效率和有效性的最佳配置。

## C. Theoretical Analysis of Dual-Layer Filtering Mechanism

## C. 双层过滤机制的理论分析

BioFilter framework demonstrates potential compatibility with diverse set-based distance metrics. This versatility stems from its filter structure being independent of specific distance measurements. This dual-layer approach achieves efficiency through progressive refinement: the count Bloom filter-based inverted index rapidly reduces the search space, while the binary Bloom filter-based sketches enable similarity assessment of the remaining candidates. To establish the theoretical foundation, we introduce the concept of set connectivity and analyze its relationship with filter collision patterns.

BioFilter框架显示出与各种基于集合的距离度量标准的潜在兼容性。这种多功能性源于其过滤器结构独立于特定的距离测量。这种双层方法通过逐步细化实现了效率：基于计数布隆过滤器的倒排索引迅速缩小了搜索空间，而基于二进制布隆过滤器的草图能够对剩余候选进行相似性评估。为了建立理论基础，我们引入了集合连通性的概念，并分析了其与过滤器冲突模式的关系。

<!-- Media -->

TABLE XV: Query Time (s) with Different Bloom Filter and WTA for Top-3&5

表XV：不同布隆过滤器和WTA设置下前3和前5的查询时间（秒）

<table><tr><td rowspan="2">Dataset</td><td colspan="4">Bloom $= {1024}$ ,WTA=64</td><td colspan="4">Bloom $= {1024}$ ,WTA=48</td><td colspan="4">Bloom $= {1024}$ ,WTA=32</td><td colspan="4">Bloom $= {1024}$ ,WTA=16</td></tr><tr><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td></tr><tr><td>CS</td><td>0.44</td><td>0.35</td><td>0.29</td><td>0.21</td><td>0.46</td><td>0.48</td><td>0.38</td><td>0.29</td><td>0.44</td><td>0.46</td><td>0.31</td><td>0.28</td><td>0.43</td><td>0.45</td><td>0.37</td><td>0.27</td></tr><tr><td>Medicine</td><td>0.51</td><td>0.40</td><td>0.34</td><td>0.24</td><td>0.49</td><td>0.41</td><td>0.32</td><td>0.25</td><td>0.47</td><td>0.39</td><td>0.28</td><td>0.22</td><td>0.45</td><td>0.36</td><td>0.28</td><td>0.19</td></tr><tr><td>Picture</td><td>0.44</td><td>0.36</td><td>0.28</td><td>0.20</td><td>0.45</td><td>0.36</td><td>0.29</td><td>0.20</td><td>0.45</td><td>0.39</td><td>0.30</td><td>0.21</td><td>0.45</td><td>0.38</td><td>0.29</td><td>0.21</td></tr><tr><td rowspan="2">Dataset</td><td colspan="4">Bloom $= {2048}$ ,WTA=64</td><td colspan="4">Bloom $= {2048}$ ,WTA=48</td><td colspan="4">Bloom $= {2048}$ ,WTA=32</td><td colspan="4">Bloom $= {2048}$ ,WTA=16</td></tr><tr><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td></tr><tr><td>CS</td><td>0.45</td><td>0.34</td><td>0.28</td><td>0.20</td><td>0.44</td><td>0.33</td><td>0.27</td><td>0.20</td><td>0.43</td><td>0.33</td><td>0.27</td><td>0.20</td><td>0.39</td><td>0.32</td><td>0.26</td><td>0.19</td></tr><tr><td>Medicine</td><td>0.48</td><td>0.40</td><td>0.31</td><td>0.22</td><td>0.47</td><td>0.39</td><td>0.29</td><td>0.22</td><td>0.46</td><td>0.38</td><td>0.29</td><td>0.21</td><td>0.43</td><td>0.37</td><td>0.28</td><td>0.19</td></tr><tr><td>Picture</td><td>0.43</td><td>0.35</td><td>0.27</td><td>0.20</td><td>0.46</td><td>0.37</td><td>0.29</td><td>0.20</td><td>0.43</td><td>0.37</td><td>0.28</td><td>0.20</td><td>0.43</td><td>0.37</td><td>0.28</td><td>0.20</td></tr></table>

<table><tbody><tr><td rowspan="2">数据集</td><td colspan="4">布隆过滤器 $= {1024}$ ，赢家通吃（WTA）=64</td><td colspan="4">布隆过滤器 $= {1024}$ ，赢家通吃（WTA）=48</td><td colspan="4">布隆过滤器 $= {1024}$ ，赢家通吃（WTA）=32</td><td colspan="4">布隆过滤器 $= {1024}$ ，赢家通吃（WTA）=16</td></tr><tr><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td></tr><tr><td>计算机科学（CS）</td><td>0.44</td><td>0.35</td><td>0.29</td><td>0.21</td><td>0.46</td><td>0.48</td><td>0.38</td><td>0.29</td><td>0.44</td><td>0.46</td><td>0.31</td><td>0.28</td><td>0.43</td><td>0.45</td><td>0.37</td><td>0.27</td></tr><tr><td>医学</td><td>0.51</td><td>0.40</td><td>0.34</td><td>0.24</td><td>0.49</td><td>0.41</td><td>0.32</td><td>0.25</td><td>0.47</td><td>0.39</td><td>0.28</td><td>0.22</td><td>0.45</td><td>0.36</td><td>0.28</td><td>0.19</td></tr><tr><td>图片</td><td>0.44</td><td>0.36</td><td>0.28</td><td>0.20</td><td>0.45</td><td>0.36</td><td>0.29</td><td>0.20</td><td>0.45</td><td>0.39</td><td>0.30</td><td>0.21</td><td>0.45</td><td>0.38</td><td>0.29</td><td>0.21</td></tr><tr><td rowspan="2">数据集</td><td colspan="4">布隆过滤器 $= {2048}$ ，赢家通吃（WTA）=64</td><td colspan="4">布隆过滤器 $= {2048}$ ，赢家通吃（WTA）=48</td><td colspan="4">布隆过滤器 $= {2048}$ ，赢家通吃（WTA）=32</td><td colspan="4">布隆过滤器 $= {2048}$ ，赢家通吃（WTA）=16</td></tr><tr><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td><td>50k</td><td>40k</td><td>30k</td><td>20k</td></tr><tr><td>计算机科学（CS）</td><td>0.45</td><td>0.34</td><td>0.28</td><td>0.20</td><td>0.44</td><td>0.33</td><td>0.27</td><td>0.20</td><td>0.43</td><td>0.33</td><td>0.27</td><td>0.20</td><td>0.39</td><td>0.32</td><td>0.26</td><td>0.19</td></tr><tr><td>医学</td><td>0.48</td><td>0.40</td><td>0.31</td><td>0.22</td><td>0.47</td><td>0.39</td><td>0.29</td><td>0.22</td><td>0.46</td><td>0.38</td><td>0.29</td><td>0.21</td><td>0.43</td><td>0.37</td><td>0.28</td><td>0.19</td></tr><tr><td>图片</td><td>0.43</td><td>0.35</td><td>0.27</td><td>0.20</td><td>0.46</td><td>0.37</td><td>0.29</td><td>0.20</td><td>0.43</td><td>0.37</td><td>0.28</td><td>0.20</td><td>0.43</td><td>0.37</td><td>0.28</td><td>0.20</td></tr></tbody></table>

<!-- Media -->

Definition 11 (Set Connectivity). For two vector sets $\mathbf{Q}$ and $\mathbf{V}$ ,their set connectivity is defined as:

定义11（集合连通性）。对于两个向量集 $\mathbf{Q}$ 和 $\mathbf{V}$，它们的集合连通性定义如下：

$$
\operatorname{Conn}\left( {\mathbf{Q},\mathbf{V}}\right)  = \mathop{\sum }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\sum }\limits_{{\mathbf{v} \in  \mathbf{V}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) ,
$$

where $\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)$ represents a pairwise similarity.

其中 $\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)$ 表示成对相似度。

A fundamental insight of our framework is that the effectiveness of both Bloom filters stems from their ability to capture set relationships through hash collision positions. Specifically, when vector sets share similar elements, their hash functions map to overlapping positions, manifesting as either accumulated counts in the count Bloom filter or shared bit patterns in the binary Bloom filter. This position-based collision mechanism forms the theoretical foundation for both filtering layers. The following theorem establishes that such collision patterns in both filter types correlate with set connectivity, thereby validating the effectiveness of our dual-layer approach:

我们框架的一个基本见解是，两种布隆过滤器（Bloom filters）的有效性源于它们通过哈希冲突位置捕捉集合关系的能力。具体而言，当向量集共享相似元素时，它们的哈希函数会映射到重叠位置，表现为计数布隆过滤器（count Bloom filter）中的累积计数，或二进制布隆过滤器（binary Bloom filter）中的共享位模式。这种基于位置的冲突机制构成了两个过滤层的理论基础。以下定理表明，两种过滤器类型中的此类冲突模式与集合连通性相关，从而验证了我们双层方法的有效性：

Theorem 5 (Collision-Similarity Relationship). For a query vector set $\mathbf{Q}$ and two vector sets ${\mathbf{V}}_{\mathbf{1}}$ and ${\mathbf{V}}_{\mathbf{2}}$ ,where $\mathbf{Q}{ \cap  }_{h}$ $\mathbf{V}$ denotes hash collisions between elements from $\mathbf{Q}$ and $\mathbf{V}$ in either count Bloom filter or binary Bloom filter. If their collision probability satisfies:

定理5（冲突 - 相似度关系）。对于一个查询向量集 $\mathbf{Q}$ 和两个向量集 ${\mathbf{V}}_{\mathbf{1}}$ 与 ${\mathbf{V}}_{\mathbf{2}}$，其中 $\mathbf{Q}{ \cap  }_{h}$ $\mathbf{V}$ 表示在计数布隆过滤器或二进制布隆过滤器中，$\mathbf{Q}$ 和 $\mathbf{V}$ 中的元素之间的哈希冲突。如果它们的冲突概率满足：

$$
P\left( {\mathbf{Q}{ \cap  }_{h}{\mathbf{V}}_{1} \neq  \varnothing }\right)  \geq  P\left( {\mathbf{Q}{ \cap  }_{h}{\mathbf{V}}_{2} \neq  \varnothing }\right) ,
$$

Then:

那么：

$$
\operatorname{Conn}\left( {\mathbf{Q},{\mathbf{V}}_{1}}\right)  \gtrsim  \operatorname{Conn}\left( {\mathbf{Q},{\mathbf{V}}_{2}}\right) ,
$$

where $\gtrsim$ denotes that the left-hand side is approximately greater than the right-hand side.

其中 $\gtrsim$ 表示左边近似大于右边。

Proof. By the definition of collision probability:

证明。根据冲突概率的定义：

$$
P\left( {\mathbf{Q}{ \cap  }_{h}{\mathbf{V}}_{1} \neq  \varnothing }\right)  = 1 - \mathop{\prod }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\prod }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{1}}}\left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) ,
$$

$$
P\left( {\mathbf{Q}{ \cap  }_{h}{\mathbf{V}}_{2} \neq  \varnothing }\right)  = 1 - \mathop{\prod }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\prod }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{2}}}\left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) .
$$

The given condition implies:

给定条件意味着：

$$
1 - \mathop{\prod }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\prod }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{1}}}\left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right)  \geq  1 - \mathop{\prod }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\prod }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{2}}}\left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) .
$$

Therefore:

因此：

$$
\mathop{\prod }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\prod }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{1}}}\left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right)  \leq  \mathop{\prod }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\prod }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{2}}}\left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) .
$$

Taking the logarithm of both sides:

对两边取对数：

$$
\mathop{\sum }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\sum }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{1}}}\log \left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right)  \leq  \mathop{\sum }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\sum }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{2}}}\log \left( {1 - \operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) }\right) .
$$

Using the Taylor series expansion of $\log \left( {1 - x}\right)$ at $x = 0$ :

使用 $\log \left( {1 - x}\right)$ 在 $x = 0$ 处的泰勒级数展开：

$$
\log \left( {1 - x}\right)  =  - x - \frac{{x}^{2}}{2} - \frac{{x}^{3}}{3} - \ldots ,\;x \in  \left\lbrack  {0,1}\right\rbrack  
$$

Approximating this for small $x$ :

对于小的 $x$ 进行近似：

$$
\log \left( {1 - x}\right)  \approx   - x + O\left( {x}^{2}\right) .
$$

Since $\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)  \in  \left\lbrack  {0,1}\right\rbrack$ is a small value,we can approximate the following relationship:

由于 $\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)  \in  \left\lbrack  {0,1}\right\rbrack$ 是一个小值，我们可以近似得到以下关系：

$$
\mathop{\sum }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\sum }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{1}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right)  \gtrsim  \mathop{\sum }\limits_{{\mathbf{q} \in  \mathbf{Q}}}\mathop{\sum }\limits_{{\mathbf{v} \in  {\mathbf{V}}_{2}}}\operatorname{sim}\left( {\mathbf{q},\mathbf{v}}\right) 
$$

This establishes:

由此可得：

## $\operatorname{Conn}\left( {\mathbf{Q},{\mathbf{V}}_{1}}\right)  \gtrsim  \operatorname{Conn}\left( {\mathbf{Q},{\mathbf{V}}_{2}}\right)$

This theoretical analysis demonstrates that higher collision probability in our filters correlates with stronger set connectivity, providing a foundation for the effectiveness of BioFilter. The relationship between hash collisions and set connectivity validates that our dual-layer filtering mechanism effectively preserves and identifies meaningful set relationships during the search process, while the metric-independent nature of this correlation supports the framework's potential adaptability to various set distance measures.

这种理论分析表明，我们的过滤器中较高的冲突概率与较强的集合连通性相关，为BioFilter的有效性提供了基础。哈希冲突与集合连通性之间的关系验证了我们的双层过滤机制在搜索过程中能有效保留和识别有意义的集合关系，而这种相关性与度量无关的特性支持了该框架对各种集合距离度量的潜在适应性。