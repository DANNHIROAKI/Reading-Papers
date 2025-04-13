# RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search

# RaBitQ：为近似最近邻搜索对高维向量进行量化并具有理论误差界

JIANYANG GAO, Nanyang Technological University, Singapore CHENG LONG*, Nanyang Technological University, Singapore

高建阳，新加坡南洋理工大学；龙成*，新加坡南洋理工大学

Searching for approximate nearest neighbors (ANN) in the high-dimensional Euclidean space is a pivotal problem. Recently, with the help of fast SIMD-based implementations, Product Quantization (PQ) and its variants can often efficiently and accurately estimate the distances between the vectors and have achieved great success in the in-memory ANN search. Despite their empirical success, we note that these methods do not have a theoretical error bound and are observed to fail disastrously on some real-world datasets. Motivated by this,we propose a new randomized quantization method named RaBitQ,which quantizes $D$ -dimensional vectors into $D$ -bit strings. RaBitQ guarantees a sharp theoretical error bound and provides good empirical accuracy at the same time. In addition, we introduce efficient implementations of RaBitQ, supporting to estimate the distances with bitwise operations or SIMD-based operations. Extensive experiments on real-world datasets confirm that (1) our method outperforms PQ and its variants in terms of accuracy-efficiency trade-off by a clear margin and (2) its empirical performance is well-aligned with our theoretical analysis.

在高维欧几里得空间中搜索近似最近邻（ANN）是一个关键问题。最近，借助基于快速单指令多数据（SIMD）的实现，乘积量化（PQ）及其变体通常能够高效且准确地估计向量之间的距离，并在内存中近似最近邻搜索方面取得了巨大成功。尽管这些方法在实践中取得了成功，但我们注意到它们没有理论误差界，并且在一些真实世界的数据集上表现不佳。受此启发，我们提出了一种新的随机量化方法，名为RaBitQ，它将$D$维向量量化为$D$位字符串。RaBitQ保证了严格的理论误差界，同时在实践中也具有良好的准确性。此外，我们还介绍了RaBitQ的高效实现，支持通过按位运算或基于SIMD的运算来估计距离。在真实世界数据集上的大量实验证实：（1）我们的方法在准确性 - 效率权衡方面明显优于PQ及其变体；（2）其实际性能与我们的理论分析高度一致。

CCS Concepts: - Theory of computation $\rightarrow$ Data structures and algorithms for data management; Random projections and metric embeddings; - Information systems $\rightarrow$ Information retrieval query processing.

计算机协会（ACM）概念： - 计算理论 $\rightarrow$ 数据管理的数据结构和算法；随机投影和度量嵌入； - 信息系统 $\rightarrow$ 信息检索查询处理。

Additional Key Words and Phrases: Approximate Nearest Neighbor Search, Johnson-Lindenstrauss Transformation, Quantization

其他关键词和短语：近似最近邻搜索、约翰逊 - 林登斯特劳斯变换、量化

## ACM Reference Format:

## ACM引用格式：

Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 2, 3 (SIGMOD), Article 167 (June 2024), 27 pages. https://doi.org/10.1145/3654970

高建阳和龙成。2024年。RaBitQ：为近似最近邻搜索对高维向量进行量化并具有理论误差界。《ACM数据管理会议论文集》2，3（SIGMOD），文章编号167（2024年6月），27页。https://doi.org/10.1145/3654970

## 1 INTRODUCTION

## 1 引言

Searching for the nearest neighbor (NN) in the high-dimensional Euclidean space is pivotal for various applications such as information retrieval [60], data mining [16], and recommendations [76]. However, the curse of dimensionality [39, 89] makes exact NN queries on extensive vector databases practically infeasible due to their long response time. To strike a balance between time and accuracy, researchers often explore its relaxed counterpart, known as approximate nearest neighbor (ANN) search $\left\lbrack  {{18},{34},{37},{45},{65},{70}}\right\rbrack$ .

在高维欧几里得空间中搜索最近邻（NN）对于信息检索[60]、数据挖掘[16]和推荐系统[76]等各种应用至关重要。然而，由于维度诅咒[39, 89]，在大规模向量数据库上进行精确的最近邻查询由于响应时间过长而在实际中不可行。为了在时间和准确性之间取得平衡，研究人员通常会探索其松弛版本，即近似最近邻（ANN）搜索$\left\lbrack  {{18},{34},{37},{45},{65},{70}}\right\rbrack$。

Product Quantization (PQ) and its variants are a family of popular methods for ANN [8, 34, 36, ${45},{66} - {68},{82},{84},{94}\rbrack$ . These methods target to efficiently estimate the distances between the data vectors and query vectors during the query phase to shortlist a list of candidates, which would

乘积量化（PQ）及其变体是一类流行的近似最近邻搜索方法[8, 34, 36, ${45},{66} - {68},{82},{84},{94}\rbrack$。这些方法旨在在查询阶段高效地估计数据向量和查询向量之间的距离，以筛选出候选列表，

Authors' addresses: Jianyang Gao, jianyang.gao@ntu.edu.sg, Nanyang Technological University, Singapore; Cheng Long, c.long@ntu.edu.sg, Nanyang Technological University, Singapore.

作者地址：高建阳，jianyang.gao@ntu.edu.sg，新加坡南洋理工大学；龙成，c.long@ntu.edu.sg，新加坡南洋理工大学。

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike Interna-

本作品遵循知识共享署名 - 非商业性使用 - 相同方式共享国际许可协议。

© 2024 Copyright held by the owner/author(s).

© 2024 版权归所有者/作者所有。

https://doi.org/10.1145/3654970 then be re-ranked based on exact distances for finding the NN. Specifically, during the index phase, they (1) construct a quantization codebook and (2) find for each data vector the nearest vector in the codebook as its quantized vector. The quantized vector is represented and stored as a short quantization code (e.g., the ID of the quantized data vector in the codebook). During the query phase, they (1) pre-compute the (squared) distances ${}^{1}$ between the query and the vectors in the codebook when a query comes and (2) for a data vector, they adopt the distances between the query vector and its quantized data vector (which can be computed by looking up the pre-computed values) as the estimated distances. Recently, with the help of the fast SIMD-based implementation [4, 5], PQ has achieved great success in the in-memory ANN search [37, 43, 48]. In particular, on many real-world datasets, the method can efficiently estimate the distances with high accuracy.

https://doi.org/10.1145/3654970 然后根据精确距离对这些候选进行重新排序以找到最近邻。具体来说，在索引阶段，它们（1）构建一个量化码本，（2）为每个数据向量在码本中找到最接近的向量作为其量化向量。量化向量以短量化码（例如，码本中量化数据向量的ID）的形式表示和存储。在查询阶段，当查询到来时，它们（1）预先计算查询向量与码本中向量之间的（平方）距离${}^{1}$，（2）对于一个数据向量，它们采用查询向量与其量化数据向量之间的距离（可以通过查找预先计算的值来计算）作为估计距离。最近，借助基于快速SIMD的实现[4, 5]，PQ在内存中近似最近邻搜索方面取得了巨大成功[37, 43, 48]。特别是在许多真实世界的数据集上，该方法能够高效且准确地估计距离。

---

<!-- Footnote -->

*Cheng Long is the corresponding author.

*龙成是通讯作者。

<!-- Footnote -->

---

Despite their empirical success on many real-world datasets, to the best of our knowledge, none of PQ and its variants [8, 34, 36, 45, 66-68, 82, 84, 94] provide theoretical error bounds on the estimated distances. This is because they lose guarantees in both (1) the codebook construction component and (2) the distance estimation component. Codebook Construction: They construct the codebook often via approximately solving an optimization problem for a heuristic objective function, e.g., PQ conducts KMeans clustering on the sub-segments of the data vectors and uses the set of the products of cluster centroids as the codebook. However, due to their heuristic nature, it is often difficult to analyze their results theoretically (e.g., no theoretical results have been achieved on the distance between a data vector and its nearest vector in the codebook (i.e., its quantized vector)). Distance Estimation: They estimate the distance between a data vector and a query vector with that between the quantized data vector and the query vector, i.e., they simply treat the quantized data vector as the data vector for computing the distance. While this looks intuitive, it does not come with a theoretical error bound on the approximation. The lack of a theoretical error bound indicates that these methods may unpredictably fail anytime, moderately or severely, when they are deployed in real-world systems to handle new datasets and queries which they have not been tested on. In fact, such failure has been observed on public real-world datasets which are widely adopted to benchmark ANN search. For example, on the dataset MSong, PQ (with the fast SIMD-based implementation $\left\lbrack  {4,5}\right\rbrack$ ) incurs more than ${50}\%$ of average relative error on the estimated distances between the query and data vectors, which causes disastrous recall of ANN search (e.g., it has no more than ${60}\%$ recall even with re-ranking applied,as shown in Section 5.2.3).

尽管量化乘积（PQ）及其变体[8, 34, 36, 45, 66 - 68, 82, 84, 94]在许多现实世界的数据集上取得了实证成功，但据我们所知，它们都没有为估计距离提供理论误差界。这是因为它们在（1）码本构建组件和（2）距离估计组件这两方面都缺乏理论保证。码本构建：它们通常通过近似求解启发式目标函数的优化问题来构建码本，例如，PQ对数据向量的子段进行K均值（KMeans）聚类，并使用聚类质心的乘积集合作为码本。然而，由于其启发式的本质，通常很难从理论上分析其结果（例如，尚未在数据向量与其在码本中最近向量（即其量化向量）之间的距离上取得理论结果）。距离估计：它们用量化后的数据向量与查询向量之间的距离来估计数据向量与查询向量之间的距离，即，它们简单地将量化后的数据向量视为数据向量来计算距离。虽然这看起来很直观，但它并没有为这种近似提供理论误差界。缺乏理论误差界表明，当这些方法被部署到现实世界的系统中，用于处理它们未曾测试过的新数据集和查询时，可能会在任何时候出现不可预测的失败，失败程度或轻或重。事实上，在被广泛用于近似最近邻（ANN）搜索基准测试的公共现实世界数据集上已经观察到了这种失败。例如，在MSong数据集上，PQ（采用基于单指令多数据（SIMD）的快速实现$\left\lbrack  {4,5}\right\rbrack$）在查询向量和数据向量之间的估计距离上产生了超过${50}\%$的平均相对误差，这导致了ANN搜索的召回率极低（例如，即使应用了重排序，其召回率也不超过${60}\%$，如5.2.3节所示）。

<!-- Media -->

Table 1. Comparison between RaBitQ and PQ and its variants. More $\star$ ’s indicates better efficiency.

表1. RaBitQ与PQ及其变体的比较。更多的$\star$表示效率更高。

<table><tr><td/><td>RaBitQ (new)</td><td>PQ and its variants</td></tr><tr><td>Codebook</td><td>Randomly transformed bi-valued vectors.</td><td>Cartesian product of sub-codebooks.</td></tr><tr><td>Quantization Code</td><td>A bit string.</td><td>A sequence of 4-bit/8-bit unsigned integers.</td></tr><tr><td>Distance Estimator</td><td>Unbiased and provides a sharp error bound.</td><td>Biased and provide no error bound.</td></tr><tr><td>Implementation (single)</td><td>Bitwise operations. ★★</td><td>Looking up tables in RAM. ★</td></tr><tr><td>Implementation (batch)</td><td>Fast SIMD-based operations. $\bigstar \bigstar \bigstar$</td><td>Fast SIMD-based operations. ★★★</td></tr></table>

<table><tbody><tr><td></td><td>兔量子编码（RaBitQ，新版）</td><td>乘积量化（PQ）及其变体</td></tr><tr><td>码本</td><td>随机变换的双值向量。</td><td>子码本的笛卡尔积。</td></tr><tr><td>量化码</td><td>一个位串。</td><td>一个4位/8位无符号整数序列。</td></tr><tr><td>距离估计器</td><td>无偏且能给出精确的误差界。</td><td>有偏且不提供误差界。</td></tr><tr><td>实现方式（单样本）</td><td>按位运算。★★</td><td>在随机存取存储器（RAM）中查找表。★</td></tr><tr><td>实现方式（批量）</td><td>基于单指令多数据（SIMD）的快速运算。$\bigstar \bigstar \bigstar$</td><td>基于单指令多数据（SIMD）的快速运算。★★★</td></tr></tbody></table>

<!-- Media -->

In this paper, we propose a new quantization method, which provides unbiased estimation on the distances and achieves a sharp ${}^{2}$ theoretical error bound. The new method achieves this with careful and integrated design in both the codebook construction and distance estimation components. Codebook Construction: It first normalizes the data vectors in order to align them on the unit hypersphere in the $D$ -dimensional space. It then constructs the codebook by (1) constructing a set of ${2}^{D}$ bi-valued vectors whose coordinates are $- 1/\sqrt{D}$ or $+ 1/\sqrt{D}$ (i.e.,the set consists of the vertices of a hypercube, which evenly spread on the unit hypersphere) and (2) randomly rotating the bi-valued vectors by multiplying each with a random orthogonal matrix ${}^{3}$ (i.e.,it performs a type of Johnson-Lindenstrauss Transformation [49], JLT in short). For each data vector, its nearest vector from the codebook is taken as the quantized vector. Since each quantized vector is a rotated $D$ -dimensional bi-valued vector,we represent its quantization code as a bit string of length $D$ , where 0 and 1 indicate the two distinct values. The rationale of the codebook construction is that it has a clear geometric interpretation (i.e., the vectors in the codebook are a set of randomly rotated vectors on the unit hypersphere) such that it is possible to analyze the geometric relationship among the data vectors, their quantized vectors and the query vectors explicitly. Distance Estimation: We carefully design an estimator of the distance between a data vector and a query vector by leveraging the aforementioned geometric relationship. We prove that the estimator is unbiased and has a sharp probabilistic error bound with the help of plentiful theoretical tools about the JLT [28, 52, 81]. This is in contrast to PQ and its variants, which simply treat the quantized vector as the data vector for estimating the distances,which is biased and provides no theoretical error bound. We call the new quantization method, which uses randomly transformed bi-valued vectors for quantizing data vectors, RaBitQ. Compared with PQ and its variants, RaBitQ has its superiority not only in providing error bounds in theory, but also in estimating the distances with smaller empirical errors even with shorter quantization codes by roughly a half (as verified on all the tested datasets shown in Section 5.2.1).

在本文中，我们提出了一种新的量化方法，该方法能对距离进行无偏估计，并实现了一个严格的${}^{2}$理论误差界。这种新方法通过在码本构建和距离估计组件中进行精心且综合的设计来实现这一目标。码本构建：首先对数据向量进行归一化处理，使其在$D$维空间的单位超球面上对齐。然后通过以下方式构建码本：(1) 构建一组${2}^{D}$个双值向量，其坐标为$- 1/\sqrt{D}$或$+ 1/\sqrt{D}$（即该集合由超立方体的顶点组成，这些顶点均匀分布在单位超球面上）；(2) 通过将每个双值向量与一个随机正交矩阵${}^{3}$相乘来随机旋转这些双值向量（即执行一种约翰逊 - 林登斯特劳斯变换 [49]，简称 JLT）。对于每个数据向量，从码本中选取与其最接近的向量作为量化向量。由于每个量化向量是一个旋转后的$D$维双值向量，我们将其量化码表示为长度为$D$的比特串，其中 0 和 1 表示两个不同的值。码本构建的基本原理是它具有清晰的几何解释（即码本中的向量是单位超球面上的一组随机旋转向量），这样就可以明确分析数据向量、其量化向量和查询向量之间的几何关系。距离估计：我们利用上述几何关系精心设计了一个数据向量与查询向量之间距离的估计器。借助大量关于 JLT 的理论工具 [28, 52, 81]，我们证明了该估计器是无偏的，并且具有严格的概率误差界。这与乘积量化（PQ）及其变体形成对比，后者简单地将量化向量视为数据向量来估计距离，这是有偏的，并且没有理论误差界。我们将这种使用随机变换的双值向量对数据向量进行量化的新量化方法称为 RaBitQ。与 PQ 及其变体相比，RaBitQ 的优势不仅在于在理论上提供误差界，而且即使使用大约短一半的量化码，在估计距离时的经验误差也更小（如第 5.2.1 节所示的所有测试数据集所验证）。

---

<!-- Footnote -->

${}^{1}$ By distances,we refer to the squared distances without further specification.

${}^{1}$ 除非另有说明，这里的距离指的是平方距离。

${}^{2}$ The error bound is sharp in the sense that it achieves the asymptotic optimality shown in [3]. Detailed discussions can be found in Section 3.2.2.

${}^{2}$ 误差界是严格的，因为它达到了文献 [3] 中所示的渐近最优性。详细讨论可在第 3.2.2 节中找到。

<!-- Footnote -->

---

We further introduce two efficient implementations for computing the value of RaBitQ’s distance estimator, namely one for a single data vector and the other for a batch of data vectors. For the former, our implementation is based on simple bitwise operations - recall that our quantization codes are bit strings. Our implementation is on average $3\mathrm{x}$ faster than the original implementation of PQ which relies on looking up tables in RAM while reaching the same accuracy (as shown in Section 5.2.1). Note that for a single data vector,the SIMD-based implementation of PQ [4,5] is not feasible as it requires to pack the quantization codes in a batch and reorganize their layout carefully. For the latter,the same strategy of the fast SIMD-based implementation [4, 5] can be adopted seamlessly, and thus it achieves similar efficiency as existing SIMD-based implementation of PQ does when similar length quantization codes are used - in this case, our method would provide more accurate estimated distances as explained earlier. Table 1 provides some comparison between RaBitQ and PQ and its variants.

我们进一步介绍了两种计算 RaBitQ 距离估计器值的高效实现方法，一种用于单个数据向量，另一种用于一批数据向量。对于前者，我们的实现基于简单的按位运算——请记住我们的量化码是比特串。我们的实现平均比依赖于在随机存取存储器（RAM）中查找表的 PQ 原始实现快$3\mathrm{x}$，同时达到相同的精度（如第 5.2.1 节所示）。请注意，对于单个数据向量，基于单指令多数据（SIMD）的 PQ 实现 [4,5] 不可行，因为它需要批量打包量化码并仔细重新组织其布局。对于后者，可以无缝采用基于 SIMD 的快速实现 [4, 5] 的相同策略，因此当使用相似长度的量化码时，它能达到与现有的基于 SIMD 的 PQ 实现类似的效率——在这种情况下，如前所述，我们的方法将提供更准确的估计距离。表 1 对 RaBitQ 与 PQ 及其变体进行了一些比较。

We summarize our major contributions as follows.

我们将主要贡献总结如下。

(1) We propose a new quantization method, namely RaBitQ. (1) It constructs the codebook via randomly transforming bi-valued vectors. (2) It designs an unbiased distance estimator with a sharp probabilistic error bound.

(1) 我们提出了一种新的量化方法，即 RaBitQ。(1) 它通过随机变换双值向量来构建码本。(2) 它设计了一个具有严格概率误差界的无偏距离估计器。

(2) We introduce efficient implementations of computing the distance estimator for RaBitQ. Our implementation is more efficient than its counterpart of PQ and its variants when estimating the distance for a single data vector and is comparably fast when estimating the distances for a batch of data vectors with quantization codes of similar lengths.

(2) 我们介绍了计算 RaBitQ 距离估计器的高效实现方法。在估计单个数据向量的距离时，我们的实现比 PQ 及其变体的对应实现更高效；在使用相似长度的量化码估计一批数据向量的距离时，速度相当。

(3) We conduct extensive experiments on real-world datasets, which show that (1) RaBitQ provides more accurate estimated distances than PQ (and its variants) even when the former uses shorter codes than the latter by roughly a half (which implies the accuracy gap would be further larger when both methods use codes of similar lengths); (2) RaBitQ works stably well on all datasets tested including some on which PQ (and its variants) fail (which is well aligned with the theoretical results); (3) RaBitQ is superior over PQ (and its variants) in terms of time-accuracy trade-offs for in-memory ANN by a clear margin on all datasets tested; and (4) RaBitQ has its empirical performance well-aligned with the theoretical analysis.

(3) 我们在真实世界的数据集上进行了广泛的实验，结果表明：（1）即使RaBitQ使用的编码长度比PQ（及其变体）大约短一半，前者提供的估计距离仍比后者更准确（这意味着当两种方法使用相似长度的编码时，精度差距会进一步扩大）；（2）RaBitQ在所有测试数据集上都能稳定良好地工作，包括一些PQ（及其变体）失效的数据集（这与理论结果高度一致）；（3）在所有测试数据集上，RaBitQ在内存中近似最近邻搜索（ANN）的时间 - 精度权衡方面明显优于PQ（及其变体）；（4）RaBitQ的实证性能与理论分析高度一致。

---

<!-- Footnote -->

${}^{3}$ We note that we do not explicitly materialize the codebook,but maintain it conceptually,as existing quantization methods such as PQ do.

${}^{3}$ 我们注意到，我们不会显式地实现码本，而是像现有量化方法（如PQ）那样在概念上维护它。

<!-- Footnote -->

---

The remainder of the paper is organized as follows. Section 2 introduces the ANN search and PQ and its variants. Section 3 presents our RaBitQ method. Section 4 illustrates the application of RaBitQ to the in-memory ANN search. Section 5 provides extensive experimental studies on real-world datasets. Section 6 discusses related work. Section 7 presents the conclusion and discussion.

本文的其余部分组织如下。第2节介绍近似最近邻搜索（ANN）、乘积量化（PQ）及其变体。第3节介绍我们的RaBitQ方法。第4节说明RaBitQ在内存中近似最近邻搜索中的应用。第5节提供在真实世界数据集上的广泛实验研究。第6节讨论相关工作。第7节给出结论和讨论。

## 2 ANN QUERY AND QUANTIZATION

## 2 近似最近邻查询与量化

ANN Query. Suppose that we have a database of $N$ data vectors in the $D$ -dimensional Euclidean space. The approximate nearest neighbor (ANN) search query is to retrieve the nearest vector from the database for a given query vector $\mathbf{q}$ . The question is usually extended to the query of retrieving the $K$ nearest neighbors. For the ease of narrative,we assume that $K = 1$ in our algorithm description,while all of the proposed techniques can be easily adapted to a general $K$ . We focus on the in-memory ANN, which assumes that all the raw data vectors and indexes can be hosted in the main memory $\left\lbrack  {4 - 6,{30},{32},{58},{65}}\right\rbrack$ .

近似最近邻查询。假设我们在$D$维欧几里得空间中有一个包含$N$个数据向量的数据库。近似最近邻（ANN）搜索查询是为给定的查询向量$\mathbf{q}$从数据库中检索最近的向量。这个问题通常会扩展到检索$K$个最近邻的查询。为了叙述方便，在我们的算法描述中假设$K = 1$，而所有提出的技术都可以很容易地适应一般的$K$。我们专注于内存中的近似最近邻搜索，它假设所有原始数据向量和索引都可以存储在主内存$\left\lbrack  {4 - 6,{30},{32},{58},{65}}\right\rbrack$中。

Product Quantization. Product Quantization (PQ) and its variants are a family of popular methods for ANN [8, 34, 36, 45, 66-68, 82, 84, 94] (for the discussion on a broader range of quantization methods, see Section 6). For a query vector and a data vector, these methods target to efficiently estimate their distance based on some pre-computed short quantization codes. Specifically, for PQ, it splits the $D$ -dimensional vectors into $M$ sub-segments (each sub-segment has $D/M$ dimensions). For each sub-segment,it performs KMeans clustering on the $D/M$ -dimensional vectors to obtain ${2}^{k}$ clusters and then takes the centroids of the clusters as a sub-codebook where $k$ is a tunable parameter which controls the size of the sub-codebook $(k = 8$ by default). The codebook of PQ is then formed by the Cartesian product of the sub-codebooks of the sub-segments and thus has the size of ${\left( {2}^{k}\right) }^{M}$ . Correspondingly each quantization code can be represented as an $M$ -sized sequence of $k$ -bit unsigned integers. During the query phase,asymmetric distance computation is adopted to estimate the distance [45]. In particular,it pre-processes $M$ look-up-tables (LUTs) for each sub-codebook when a query comes. The $i$ th LUT contains ${2}^{k}$ numbers which represent the squared distances between the vectors in the $i$ th sub-codebook and $i$ th sub-segment of the query vector. For a given quantization code,by looking up and accumulating the values in the LUTs for $M$ times, PQ can compute an estimated distance.

乘积量化。乘积量化（PQ）及其变体是一类流行的近似最近邻搜索方法[8, 34, 36, 45, 66 - 68, 82, 84, 94]（关于更广泛的量化方法的讨论，请参见第6节）。对于一个查询向量和一个数据向量，这些方法旨在基于一些预先计算的短量化编码来高效地估计它们之间的距离。具体来说，对于PQ，它将$D$维向量分割成$M$个子段（每个子段有$D/M$维）。对于每个子段，它对$D/M$维向量执行K均值聚类以获得${2}^{k}$个簇，然后将这些簇的质心作为一个子码本，其中$k$是一个可调参数，默认控制子码本$(k = 8$的大小。PQ的码本由子段的子码本的笛卡尔积形成，因此大小为${\left( {2}^{k}\right) }^{M}$。相应地，每个量化编码可以表示为一个由$k$位无符号整数组成的$M$大小的序列。在查询阶段，采用非对称距离计算来估计距离[45]。特别是，当一个查询到来时，它会为每个子码本预处理$M$个查找表（LUT）。第$i$个查找表包含${2}^{k}$个数字，这些数字表示第$i$个子码本中的向量与查询向量的第$i$个子段之间的平方距离。对于给定的量化编码，通过在查找表中查找并累加$M$次值，PQ可以计算出一个估计距离。

Recently, $\left\lbrack  {4,5}\right\rbrack$ propose a SIMD-based fast implementation for PQ (PQ Fast Scan,PQx4fs in short). They speed up the look-up and accumulation operations significantly, making PQ an important component in many popular libraries for in-memory ANN search such as Faiss from Meta [48], ScaNN from Google [37] and NGT-QG from Yahoo Japan [43]. At its core, unlike the original implementation of PQ which relies on looking up the LUTs in RAM, $\left\lbrack  {4,5}\right\rbrack$ propose to host the LUTs in SIMD registers and look up the LUTs with the SIMD shuffle instructions. To achieve so, the method makes several modifications on PQ. First, in order to fit the LUTs into the AVX2 256-bit registers,it modifies the original setting of $k = 8$ to $k = 4$ so that in each LUT,there are only ${2}^{4}$ floating-point numbers. It further quantizes the numbers in the LUT to be 8-bit unsigned integers so that one LUT takes the space of only ${128}\left( {{2}^{4} \times  8}\right)$ bits. Thus,one AVX2 256-bit register is able to host two LUTs. Second, in order to look up the LUTs efficiently, the method packs every 32 quantization codes in a batch and reorganizes their layout. In this case, a series of operations can estimate the distances for 32 data vectors all at once. Without further specification, by PQ, we refer to PQx4fs by default because without the fast SIMD-based implementation, the efficiency of PQ is much less competitive in the in-memory ANN search [4, 5] (see Section 5.2.1).

最近，$\left\lbrack  {4,5}\right\rbrack$提出了一种基于单指令多数据（SIMD）的乘积量化（PQ）快速实现方法（PQ快速扫描，简称PQx4fs）。他们显著加速了查找和累加操作，使PQ成为许多流行的内存中近似最近邻（ANN）搜索库的重要组成部分，如Meta的Faiss [48]、谷歌的ScaNN [37]和雅虎日本的NGT - QG [43]。其核心在于，与依赖在随机存取存储器（RAM）中查找查找表（LUT）的原始PQ实现不同，$\left\lbrack  {4,5}\right\rbrack$提议将查找表存于SIMD寄存器中，并使用SIMD混洗指令来查找查找表。为实现这一点，该方法对PQ进行了几处修改。首先，为了将查找表放入AVX2 256位寄存器中，它将$k = 8$的原始设置修改为$k = 4$，这样每个查找表中就只有${2}^{4}$个浮点数。它进一步将查找表中的数字量化为8位无符号整数，使得一个查找表仅占用${128}\left( {{2}^{4} \times  8}\right)$位的空间。因此，一个AVX2 256位寄存器能够容纳两个查找表。其次，为了高效地查找查找表，该方法将每32个量化码打包成一批，并重新组织它们的布局。在这种情况下，一系列操作可以一次性估计32个数据向量的距离。除非另有说明，默认情况下，我们所说的PQ指的是PQx4fs，因为如果没有基于SIMD的快速实现，PQ在内存中ANN搜索的效率将大大降低竞争力 [4, 5]（见5.2.1节）。

<!-- Media -->

Table 2. Notations.

表2. 符号说明。

<table><tr><td>Notation</td><td>Definition</td></tr><tr><td>${\mathbf{o}}_{r},{\mathbf{q}}_{r}$</td><td>The raw data and query vectors.</td></tr><tr><td>o, q</td><td>The normalized data and query vectors.</td></tr><tr><td>$C,{C}_{\text{rand }}$</td><td>The quantization codebook, its randomized version.</td></tr><tr><td>$P$</td><td>A random orthogonal transformation matrix.</td></tr><tr><td>$\bar{\mathbf{x}}$</td><td>The code in $C$ s.t. $P\overline{\mathbf{x}}$ is the quantized vector of $\mathbf{o}$ .</td></tr><tr><td>$\overline{\mathbf{O}}$</td><td>The quantized vector of $\mathbf{o}$ in ${C}_{\text{rand }}$ ,i.e., $\overline{\mathbf{o}} = P\overline{\mathbf{x}}$ .</td></tr><tr><td>${\overline{\mathbf{x}}}_{b}$</td><td>The quantization code of $\mathbf{o}$ as a $D$ -bit string.</td></tr><tr><td>${\mathrm{q}}^{\prime }$</td><td>The inversely transformed query vector,i.e., ${P}^{-1}\mathbf{q}$ .</td></tr><tr><td>$\bar{\mathrm{q}}$</td><td>The quantized query vector of ${\mathbf{q}}^{\prime }$ .</td></tr><tr><td>${\overline{\mathbf{q}}}_{u}$</td><td>The unsigned integer representation of $\overline{\mathbf{q}}$ .</td></tr></table>

<table><tbody><tr><td>符号表示</td><td>定义</td></tr><tr><td>${\mathbf{o}}_{r},{\mathbf{q}}_{r}$</td><td>原始数据和查询向量。</td></tr><tr><td>o, q</td><td>归一化后的数据和查询向量。</td></tr><tr><td>$C,{C}_{\text{rand }}$</td><td>量化码本及其随机化版本。</td></tr><tr><td>$P$</td><td>一个随机正交变换矩阵。</td></tr><tr><td>$\bar{\mathbf{x}}$</td><td>$C$ 中的代码，使得 $P\overline{\mathbf{x}}$ 是 $\mathbf{o}$ 的量化向量。</td></tr><tr><td>$\overline{\mathbf{O}}$</td><td>$\mathbf{o}$ 在 ${C}_{\text{rand }}$ 中的量化向量，即 $\overline{\mathbf{o}} = P\overline{\mathbf{x}}$ 。</td></tr><tr><td>${\overline{\mathbf{x}}}_{b}$</td><td>$\mathbf{o}$ 的量化码，以 $D$ 位字符串表示。</td></tr><tr><td>${\mathrm{q}}^{\prime }$</td><td>经过逆变换后的查询向量，即 ${P}^{-1}\mathbf{q}$ 。</td></tr><tr><td>$\bar{\mathrm{q}}$</td><td>${\mathbf{q}}^{\prime }$ 的量化查询向量。</td></tr><tr><td>${\overline{\mathbf{q}}}_{u}$</td><td>$\overline{\mathbf{q}}$ 的无符号整数表示。</td></tr></tbody></table>

<!-- Media -->

Nevertheless, none of PQ and its variants provide a theoretical error bound on the errors of the estimated distances $\left\lbrack  {8,{34},{36},{45},{66} - {68},{82},{84},{94}}\right\rbrack$ ,as explained in Section 1. Indeed,we find that the accuracy of PQ can be disastrous (see Section 5.2.3), e.g., on the dataset MSong, PQ cannot achieve $\geq  {60}\%$ recall even with re-ranking applied. We note that Locality Sensitive Hashing (LSH) is a family of methods which promise rigorous theoretical guarantee [18, 38, 39, 77-79]. However, as is widely reported $\left\lbrack  {6,{58},{84}}\right\rbrack$ ,these methods can hardly produce competitive empirical performance. Furthermore,their guarantees are on the accuracy of $c$ -approximate NN query. In particular,LSH guarantees to return a data vector whose distance from the query is at most $\left( {1 + c}\right)$ times of a fixed radius $r$ with high probability (if there exists a data vector whose distance from the query is within the radius $r$ ). Due to the relaxation factor $c$ ,there can be many that satisfy the statement. The guarantee of returning any of them does not help to produce high recall for ANN search. In contrast, a guarantee on the distance estimation can help to decide whether a data vector should be re-ranked for achieving high recall (see Section 4).

然而，如第1节所述，乘积量化（PQ，Product Quantization）及其变体均未为估计距离$\left\lbrack  {8,{34},{36},{45},{66} - {68},{82},{84},{94}}\right\rbrack$的误差提供理论误差界。实际上，我们发现PQ的准确性可能非常糟糕（见第5.2.3节），例如，在MSong数据集上，即使应用了重排序，PQ也无法达到$\geq  {60}\%$的召回率。我们注意到局部敏感哈希（LSH，Locality Sensitive Hashing）是一类能提供严格理论保证的方法[18, 38, 39, 77 - 79]。然而，正如广泛报道的那样$\left\lbrack  {6,{58},{84}}\right\rbrack$，这些方法在实际应用中很难产生有竞争力的性能。此外，它们的保证是针对$c$ -近似最近邻（NN，Nearest Neighbor）查询的准确性。具体而言，LSH保证以高概率返回一个数据向量，该向量与查询向量的距离至多为固定半径$r$的$\left( {1 + c}\right)$倍（如果存在一个与查询向量的距离在半径$r$内的数据向量）。由于存在松弛因子$c$，可能有很多向量满足该条件。保证返回其中任何一个向量并不能帮助在近似最近邻（ANN，Approximate Nearest Neighbor）搜索中实现高召回率。相比之下，对距离估计的保证有助于决定是否对某个数据向量进行重排序以实现高召回率（见第4节）。

## 3 THE RABITQ METHOD

## 3 RaBitQ方法

In this section, we present the details of RaBitQ. In Section 3.1, we present the index phase of RaBitQ, which normalizes the data vectors (Section 3.1.1), constructs a codebook (Section 3.1.2) and computes the quantized vectors of data vectors (Section 3.1.3). In Section 3.2, we introduce the distance estimator of RaBitQ, which is unbiased and provides a rigorous theoretical error bound. In Section 3.3, we illustrate how to efficiently compute the value of the estimator. In Section 3.4, we summarize the RaBitQ method. Table 2 lists the frequently used notations and their definitions.

在本节中，我们将详细介绍RaBitQ。在第3.1节中，我们将介绍RaBitQ的索引阶段，该阶段对数据向量进行归一化处理（第3.1.1节）、构建码本（第3.1.2节）并计算数据向量的量化向量（第3.1.3节）。在第3.2节中，我们将介绍RaBitQ的距离估计器，它是无偏的，并提供了严格的理论误差界。在第3.3节中，我们将说明如何高效地计算估计器的值。在第3.4节中，我们将总结RaBitQ方法。表2列出了常用的符号及其定义。

### 3.1 Quantizing the Data Vectors with RaBitQ

### 3.1 使用RaBitQ对数据向量进行量化

3.1.1 Converting the Raw Vectors into Unit Vectors via Normalization. We note that directly constructing a codebook for the raw data vectors is challenging for achieving the theoretical error bound because the Euclidean space is unbounded and the raw data vectors may appear anywhere in the infinitely large space. To deal with this issue, a natural idea is to normalize the raw vectors into unit vectors. Specifically,let $\mathbf{c}$ be the centroid of the raw data vectors. We normalize the raw data vectors ${\mathbf{o}}_{r}$ to be $\mathbf{o} \mathrel{\text{:=}} \frac{{\mathbf{o}}_{r} - \mathbf{c}}{\begin{Vmatrix}{\mathbf{o}}_{r} - \mathbf{c}\end{Vmatrix}}$ . Similarly,we normalize the raw query vector ${\mathbf{q}}_{r}$ (when it comes in the query phase) to be $\mathbf{q} \mathrel{\text{:=}} \frac{{\mathbf{q}}_{r} - \mathbf{c}}{\begin{Vmatrix}{\mathbf{q}}_{r} - \mathbf{c}\end{Vmatrix}}$ . The following expressions bridge the distance between the raw vectors (i.e., our target) and the inner product of the normalized vectors.

3.1.1 通过归一化将原始向量转换为单位向量。我们注意到，直接为原始数据向量构建码本以实现理论误差界具有挑战性，因为欧几里得空间是无界的，原始数据向量可能出现在无限大空间的任何位置。为了解决这个问题，一个自然的想法是将原始向量归一化为单位向量。具体来说，设$\mathbf{c}$为原始数据向量的质心。我们将原始数据向量${\mathbf{o}}_{r}$归一化为$\mathbf{o} \mathrel{\text{:=}} \frac{{\mathbf{o}}_{r} - \mathbf{c}}{\begin{Vmatrix}{\mathbf{o}}_{r} - \mathbf{c}\end{Vmatrix}}$。类似地，我们将原始查询向量${\mathbf{q}}_{r}$（在查询阶段到来时）归一化为$\mathbf{q} \mathrel{\text{:=}} \frac{{\mathbf{q}}_{r} - \mathbf{c}}{\begin{Vmatrix}{\mathbf{q}}_{r} - \mathbf{c}\end{Vmatrix}}$。以下表达式建立了原始向量之间的距离（即我们的目标）与归一化向量的内积之间的联系。

$$
{\begin{Vmatrix}{\mathbf{o}}_{r} - {\mathbf{q}}_{r}\end{Vmatrix}}^{2} = {\begin{Vmatrix}\left( {\mathbf{o}}_{r} - \mathbf{c}\right)  - \left( {\mathbf{q}}_{r} - \mathbf{c}\right) \end{Vmatrix}}^{2} \tag{1}
$$

$$
 = {\begin{Vmatrix}{\mathbf{o}}_{r} - \mathbf{c}\end{Vmatrix}}^{2} + {\begin{Vmatrix}{\mathbf{q}}_{r} - \mathbf{c}\end{Vmatrix}}^{2} - 2 \cdot  \begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix} \cdot  \begin{Vmatrix}{{\mathbf{q}}_{r} - \mathbf{c}}\end{Vmatrix} \cdot  \langle \mathbf{q},\mathbf{o}\rangle  \tag{2}
$$

We note that $\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$ is the distance from the data vector to the centroid,which can be precomputed during the index phase. $\begin{Vmatrix}{{\mathbf{q}}_{r} - \mathbf{c}}\end{Vmatrix}$ is the distance from the query vector to the centroid. It can be computed during the query phase and its cost can be shared by all the data vectors. Thus, based on Equation (2),the question of computing ${\begin{Vmatrix}{\mathbf{o}}_{r} - {\mathbf{q}}_{r}\end{Vmatrix}}^{2}$ is reduced to that of computing the inner product of two unit vectors $\langle \mathbf{q},\mathbf{o}\rangle$ . We note that in practice we can cluster the data vectors first (e.g., via KMeans clustering) and perform the normalization for data vectors within a cluster individually based on the centroid of the cluster. When considering the data vectors within a cluster, we normalize the query vector based on the corresponding centroid. In this way, the normalized data vectors are expected to spread evenly on the unit hypersphere, removing the skewness of the data (if any) to some extent. For the sake of convenience, in the following parts without further clarification, by the data and query vector, we refer to their corresponding unit vectors. With this conversion, we next focus on estimating the inner product of the unit vectors, i.e., $\langle \mathbf{q},\mathbf{o}\rangle$ .

我们注意到$\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$是数据向量到质心的距离，该距离可以在索引阶段预先计算。$\begin{Vmatrix}{{\mathbf{q}}_{r} - \mathbf{c}}\end{Vmatrix}$是查询向量到质心的距离。它可以在查询阶段计算，并且其计算成本可以由所有数据向量分摊。因此，根据方程(2)，计算${\begin{Vmatrix}{\mathbf{o}}_{r} - {\mathbf{q}}_{r}\end{Vmatrix}}^{2}$的问题就简化为计算两个单位向量$\langle \mathbf{q},\mathbf{o}\rangle$的内积问题。我们注意到，在实践中，我们可以先对数据向量进行聚类（例如，通过K均值聚类），然后基于聚类的质心分别对聚类内的数据向量进行归一化处理。当考虑某个聚类内的数据向量时，我们基于相应的质心对查询向量进行归一化处理。通过这种方式，归一化后的数据向量有望在单位超球面上均匀分布，在一定程度上消除数据的偏态（如果存在的话）。为了方便起见，在接下来未作进一步说明的部分中，提到数据向量和查询向量时，我们指的是它们对应的单位向量。通过这种转换，接下来我们将重点估计单位向量的内积，即$\langle \mathbf{q},\mathbf{o}\rangle$。

3.1.2 Constructing the Codebook. As mentioned in Section 3.1.1, the data vectors are supposed, to some extent, to be evenly spreading on the unit hypersphere due to the normalization. By intuition, our codebook should also spread evenly on the unit hypersphere. To this end, a natural construction of the codebook is given as follows.

3.1.2 码本的构建。如3.1.1节所述，由于归一化处理，数据向量在一定程度上应该均匀分布在单位超球面上。凭直觉，我们的码本也应该均匀分布在单位超球面上。为此，下面给出一种自然的码本构建方法。

$$
C \mathrel{\text{:=}} {\left\{  +\frac{1}{\sqrt{D}}, - \frac{1}{\sqrt{D}}\right\}  }^{D} \tag{3}
$$

It is easy to verify that the vectors in $\mathcal{C}$ are unit vectors and the codebook has the size of $\left| \mathcal{C}\right|  = {2}^{D}$ .

很容易验证，$\mathcal{C}$中的向量是单位向量，并且码本的大小为$\left| \mathcal{C}\right|  = {2}^{D}$。

However, such construction may favor some certain vectors and perform poorly for others. For example,for the data vector $\left( {1/\sqrt{D},\ldots ,1/\sqrt{D}}\right)$ ,its quantized data vector (which corresponds to the vector in $C$ closest from the data vector) is $\left( {1/\sqrt{D},\ldots ,1/\sqrt{D}}\right)$ ,and its squared distance to the quantized data vector is 0 . In contrast,for the vector $\left( {1,0,\ldots ,0}\right)$ ,its quantized data vector is also $\left( {1/\sqrt{D},\ldots ,1/\sqrt{D}}\right)$ ,and its squared distance to the quantized data vector equals to $2 - 2/\sqrt{D}$ . To deal with this issue,we inject the codebook some randomness. Specifically,let $P$ be a random orthogonal matrix. We propose to apply the transformation $P$ to the codebook (which is one type of the Johnson-Lindenstrauss Transformation [49]). Our final codebook is given as follows.

然而，这样的构建方式可能会对某些特定向量有利，而对其他向量表现不佳。例如，对于数据向量$\left( {1/\sqrt{D},\ldots ,1/\sqrt{D}}\right)$，其量化后的数据向量（即$C$中与该数据向量距离最近的向量）是$\left( {1/\sqrt{D},\ldots ,1/\sqrt{D}}\right)$，并且它到量化后的数据向量的平方距离为0。相比之下，对于向量$\left( {1,0,\ldots ,0}\right)$，其量化后的数据向量也是$\left( {1/\sqrt{D},\ldots ,1/\sqrt{D}}\right)$，并且它到量化后的数据向量的平方距离等于$2 - 2/\sqrt{D}$。为了解决这个问题，我们给码本注入一些随机性。具体来说，设$P$是一个随机正交矩阵。我们建议对码本应用变换$P$（这是一种约翰逊 - 林登施特劳斯变换[49]）。我们最终的码本如下所示。

$$
{C}_{\text{rand }} \mathrel{\text{:=}} \{ P\mathbf{x} \mid  \mathbf{x} \in  C\}  \tag{4}
$$

Geometrically,the transformation simply rotates the codebook because the matrix $P$ is orthogonal, and thus,the vectors in ${C}_{\text{rand }}$ are still unit vectors. Moreover,the rotation is uniformly sampled from "all the possible rotations" of the space. Thus,for a unit vector in the codebook $C$ ,it has equal probability to be rotated to anywhere on the unit hypersphere. This step thus removes the preference of the deterministic codebook $\mathcal{C}$ on specific vectors.

从几何角度看，由于矩阵$P$是正交的，该变换只是对码本进行旋转，因此，${C}_{\text{rand }}$中的向量仍然是单位向量。此外，该旋转是从空间的“所有可能旋转”中均匀采样得到的。因此，对于码本$C$中的一个单位向量，它有相等的概率被旋转到单位超球面上的任何位置。因此，这一步消除了确定性码本$\mathcal{C}$对特定向量的偏好。

We note that to construct the codebook ${\mathcal{C}}_{\text{rand }}$ ,we only need to sample a random transformation matrix $P$ . To store the codebook ${C}_{\text{rand }}$ ,we only need to physically store the sampled $P$ but not all the transformed vectors. The codebook constructed by this operation is much simpler than its counterpart in PQ and its variants which rely on approximately solving an optimization problem.

我们注意到，要构建码本${\mathcal{C}}_{\text{rand }}$，我们只需要采样一个随机变换矩阵$P$。要存储码本${C}_{\text{rand }}$，我们只需要实际存储采样得到的$P$，而不需要存储所有经过变换的向量。通过这种操作构建的码本比乘积量化（PQ）及其变体中的码本简单得多，后者依赖于近似求解一个优化问题。

3.1.3 Computing the Quantized Codes of Data Vectors. With the constructed codebook, the next step is to find the nearest vector from ${\mathcal{C}}_{\text{rand }}$ for each data vector as its quantized vector. For a unit vector $\mathbf{o}$ ,to find its nearest vector,it is equivalent to find the one which has the largest inner product with it. Let $P\overline{\mathbf{x}} \in  {\mathcal{C}}_{\text{rand }}$ be the quantized data vector (where $\overline{\mathbf{x}} \in  \mathcal{C}$ ). The following equations

3.1.3 计算数据向量的量化码。有了构建好的码本，下一步是为每个数据向量从 ${\mathcal{C}}_{\text{rand }}$ 中找到与其最接近的向量作为其量化向量。对于单位向量 $\mathbf{o}$，要找到与其最接近的向量，等价于找到与它具有最大内积的向量。设 $P\overline{\mathbf{x}} \in  {\mathcal{C}}_{\text{rand }}$ 为量化后的数据向量（其中 $\overline{\mathbf{x}} \in  \mathcal{C}$）。以下方程

illustrate the idea rigorously.

严格地阐述了这一思想。

$$
\overline{\mathbf{x}} = \underset{\mathbf{x} \in  \mathcal{C}}{\arg \min }\parallel \mathbf{o} - P\mathbf{x}{\parallel }^{2} \tag{5}
$$

$$
 = \underset{\mathbf{x} \in  \mathcal{C}}{\arg \min }\left( {\parallel \mathbf{o}{\parallel }^{2} + \parallel P\mathbf{x}{\parallel }^{2} - 2\langle \mathbf{o},P\mathbf{x}\rangle }\right)  \tag{6}
$$

$$
 = \underset{\mathbf{x} \in  \mathcal{C}}{\arg \min }\left( {2 - 2\langle \mathbf{o},P\mathbf{x}\rangle }\right)  = \underset{\mathbf{x} \in  \mathcal{C}}{\arg \max }\langle \mathbf{o},P\mathbf{x}\rangle  \tag{7}
$$

Equation (5) is based on the definition of the quantized data vector. Equation (6) is due to elementary linear algebra operations. Equation (7) is because $P\mathbf{x}$ and $\mathbf{o}$ are unit vectors. However,by Equation (7), it is costly to find the quantized data vector by physically transforming the huge codebook and finding the nearest vector via enumeration. We note that the inner product is invariant to orthogonal transformation (i.e., rotation). Thus, instead of transforming the huge codebook, we inversely transform the data vector $\mathbf{o}$ . The following expressions formally present the idea.

方程 (5) 基于量化数据向量的定义。方程 (6) 源于基本的线性代数运算。方程 (7) 是因为 $P\mathbf{x}$ 和 $\mathbf{o}$ 是单位向量。然而，根据方程 (7)，通过物理变换庞大的码本并通过枚举找到最接近的向量来寻找量化数据向量的成本很高。我们注意到内积在正交变换（即旋转）下是不变的。因此，我们不直接变换庞大的码本，而是对数据向量 $\mathbf{o}$ 进行逆变换。以下表达式正式呈现了这一思想。

$$
\langle \mathbf{o},P\mathbf{x}\rangle  = \left\langle  {{P}^{-1}\mathbf{o},{P}^{-1}P\mathbf{x}}\right\rangle   = \left\langle  {{P}^{-1}\mathbf{o},\mathbf{x}}\right\rangle   \tag{8}
$$

Recall that the entries of $\mathbf{x} \in  \mathcal{C}$ are $\pm  1/\sqrt{D}$ . To maximize the inner product,we only need to pick the $\overline{\mathbf{x}} \in  \mathcal{C}$ whose signs of the entries match those of ${P}^{-1}\mathbf{o}$ . Then $P\overline{\mathbf{x}}$ is the quantized data vector.

回顾一下，$\mathbf{x} \in  \mathcal{C}$ 的元素是 $\pm  1/\sqrt{D}$。为了使内积最大化，我们只需要选择其元素符号与 ${P}^{-1}\mathbf{o}$ 的元素符号相匹配的 $\overline{\mathbf{x}} \in  \mathcal{C}$。那么 $P\overline{\mathbf{x}}$ 就是量化后的数据向量。

In summary,to find the nearest vector of a data vector $\mathbf{o}$ from ${C}_{\text{rand }}$ ,we can inversely transform o with ${P}^{-1}$ and store the signs of its entries as a $D$ -bit string ${\overline{\mathbf{x}}}_{b} \in  \{ 0,1{\} }^{D}$ . We call the stored binary string ${\bar{\mathbf{x}}}_{b}$ as the quantization code,which can be used to re-construct the quantized vector $\bar{\mathbf{x}}$ . Let ${\mathbf{1}}_{D}$ be the $D$ -dimensional vector which has all its entries being ones. The relationship between ${\overline{\mathbf{x}}}_{b}$ and $\overline{\mathbf{x}}$ is given as $\overline{\mathbf{x}} = \left( {2{\overline{\mathbf{x}}}_{b} - {\mathbf{1}}_{D}}\right) /\sqrt{D}$ ,i.e.,when the $i$ th coordinate ${\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   = 1$ ,we have $\overline{\mathbf{x}}\left\lbrack  i\right\rbrack   = 1/\sqrt{D}$ and when ${\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   = 0$ ,we have $\overline{\mathbf{x}}\left\lbrack  i\right\rbrack   =  - 1/\sqrt{D}$ . For the sake of convenience,we denote the quantized data vector as $\overline{\mathbf{o}} \mathrel{\text{:=}} P\overline{\mathbf{x}}$ .

综上所述，要从 ${C}_{\text{rand }}$ 中找到数据向量 $\mathbf{o}$ 的最接近向量，我们可以用 ${P}^{-1}$ 对 o 进行逆变换，并将其元素的符号存储为一个 $D$ 位的字符串 ${\overline{\mathbf{x}}}_{b} \in  \{ 0,1{\} }^{D}$。我们将存储的二进制字符串 ${\bar{\mathbf{x}}}_{b}$ 称为量化码，它可用于重构量化向量 $\bar{\mathbf{x}}$。设 ${\mathbf{1}}_{D}$ 为所有元素都为 1 的 $D$ 维向量。${\overline{\mathbf{x}}}_{b}$ 和 $\overline{\mathbf{x}}$ 之间的关系由 $\overline{\mathbf{x}} = \left( {2{\overline{\mathbf{x}}}_{b} - {\mathbf{1}}_{D}}\right) /\sqrt{D}$ 给出，即当第 $i$ 个坐标 ${\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   = 1$ 时，我们有 $\overline{\mathbf{x}}\left\lbrack  i\right\rbrack   = 1/\sqrt{D}$；当 ${\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   = 0$ 时，我们有 $\overline{\mathbf{x}}\left\lbrack  i\right\rbrack   =  - 1/\sqrt{D}$。为了方便起见，我们将量化后的数据向量表示为 $\overline{\mathbf{o}} \mathrel{\text{:=}} P\overline{\mathbf{x}}$。

Till now, we have finished the pre-processing in the index phase. We note that the time cost in the index phase is not a bottleneck for our method, which is the same as in the cases of PQ and OPQ (a popular variant of PQ) [34]. For example, on the dataset GIST with one million 960-dimensional vectors, with 32 threads on CPU, our method, PQ and OPQ take 117s, 105s and 291s respectively. The space complexity of the methods is not a bottleneck for the in-memory ANN either, because the space consumption is largely due to the space for storing the raw vectors. As a comparison, each raw vector takes ${32D}$ bits (i.e., $D$ floating-point numbers). Our method by default has $D$ bits for a quantization code. PQ and OPQ by default have ${2D}$ bits for a quantization code (i.e., $M = D/2$ ) according to $\left\lbrack  {{25},{37}}\right\rbrack$ ,which is significantly smaller than the space for storing the raw vectors.

到目前为止，我们已经完成了索引阶段的预处理。我们注意到，索引阶段的时间成本并不是我们方法的瓶颈，这与乘积量化（PQ）和最优乘积量化（OPQ，PQ的一种流行变体）的情况相同 [34]。例如，在包含一百万个960维向量的GIST数据集上，使用CPU的32个线程时，我们的方法、PQ和OPQ分别需要117秒、105秒和291秒。这些方法的空间复杂度也不是内存中近似最近邻搜索（ANN）的瓶颈，因为空间消耗主要是用于存储原始向量。作为对比，每个原始向量占用 ${32D}$ 位（即 $D$ 个浮点数）。我们的方法默认情况下，一个量化码占用 $D$ 位。根据 $\left\lbrack  {{25},{37}}\right\rbrack$，PQ和OPQ默认情况下，一个量化码占用 ${2D}$ 位（即 $M = D/2$ ），这明显小于存储原始向量所需的空间。

### 3.2 Constructing an Unbiased Estimator

### 3.2 构建无偏估计器

Recall that the problem of computing ${\begin{Vmatrix}{\mathbf{o}}_{r} - {\mathbf{q}}_{r}\end{Vmatrix}}^{2}$ can be reduced to that of computing the inner product of two unit vectors $\langle \mathbf{o},\mathbf{q}\rangle$ . In this section,we introduce an unbiased estimator for $\langle \mathbf{o},\mathbf{q}\rangle$ . Unlike PQ and its variants which simply treat the quantized data vector as the data vector for estimating the distances without theoretical error bounds, we first explicitly derive the relationship between $\langle \mathbf{o},\mathbf{q}\rangle$ and $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ in Section 3.2.1. We then construct an unbiased estimator for $\langle \mathbf{o},\mathbf{q}\rangle$ based on the derived relationships and present its rigorous error bound in Section 3.2.2.

回顾一下，计算 ${\begin{Vmatrix}{\mathbf{o}}_{r} - {\mathbf{q}}_{r}\end{Vmatrix}}^{2}$ 的问题可以简化为计算两个单位向量 $\langle \mathbf{o},\mathbf{q}\rangle$ 的内积问题。在本节中，我们将为 $\langle \mathbf{o},\mathbf{q}\rangle$ 引入一个无偏估计器。与PQ及其变体不同，它们只是简单地将量化后的数据向量当作数据向量来估计距离，而没有理论误差界。我们首先在3.2.1节中明确推导 $\langle \mathbf{o},\mathbf{q}\rangle$ 和 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 之间的关系。然后，在3.2.2节中，基于推导出的关系为 $\langle \mathbf{o},\mathbf{q}\rangle$ 构建一个无偏估计器，并给出其严格的误差界。

3.2.1 Analyzing the Explicit Relationship between $\langle \mathbf{o},\mathbf{q}\rangle$ and $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ . We note that the relationship between $\langle \mathbf{o},\mathbf{q}\rangle$ and $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ depends only on the projection of $\overline{\mathbf{o}}$ on the two-dimensional subspace spanned by $\mathbf{o}$ and $\mathbf{q}$ ,which is illustrated on the left panel of Figure 1. For the component of $\overline{\mathbf{o}}$ which is perpendicular to the subspace,it has no effect on the inner product $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ . The following lemma presents the specific result. The proof can be found in the technical report [33].

3.2.1 分析 $\langle \mathbf{o},\mathbf{q}\rangle$ 和 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 之间的显式关系。我们注意到，$\langle \mathbf{o},\mathbf{q}\rangle$ 和 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 之间的关系仅取决于 $\overline{\mathbf{o}}$ 在由 $\mathbf{o}$ 和 $\mathbf{q}$ 张成的二维子空间上的投影，如图1左图所示。对于 $\overline{\mathbf{o}}$ 中与该子空间垂直的分量，它对 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 内积没有影响。下面的引理给出了具体结果。证明可在技术报告 [33] 中找到。

<!-- Media -->

<!-- figureText: 0 1.2 0.8 0.4 0.0 -0.8 -0.8 -0.4 0.0 0.4 0.8 1.2 e q -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_7.jpg?x=324&y=269&w=1067&h=426&r=0"/>

Fig. 1. Geometric Relationship among the Vectors.

图1. 向量间的几何关系。

<!-- Media -->

LEMMA 3.1 (GEOMETRIC RELATIONSHIP). Let $\mathbf{o},\mathbf{q}$ and $\overline{\mathbf{o}}$ be any three unit vectors. When $\mathbf{o}$ and $\mathbf{q}$ are collinear (i.e., $\mathbf{o} = \mathbf{q}$ or $\mathbf{o} =  - \mathbf{q}$ ),we have

引理3.1（几何关系）。设 $\mathbf{o},\mathbf{q}$ 和 $\overline{\mathbf{o}}$ 为任意三个单位向量。当 $\mathbf{o}$ 和 $\mathbf{q}$ 共线（即 $\mathbf{o} = \mathbf{q}$ 或 $\mathbf{o} =  - \mathbf{q}$ ）时，我们有

$$
\langle \overline{\mathbf{o}},\mathbf{q}\rangle  = \langle \overline{\mathbf{o}},\mathbf{o}\rangle  \cdot  \langle \mathbf{o},\mathbf{q}\rangle  \tag{9}
$$

When $\mathbf{o}$ and $\mathbf{q}$ are non-collinear,we have

当 $\mathbf{o}$ 和 $\mathbf{q}$ 不共线时，我们有

$$
\langle \overline{\mathbf{o}},\mathbf{q}\rangle  = \langle \overline{\mathbf{o}},\mathbf{o}\rangle  \cdot  \langle \mathbf{o},\mathbf{q}\rangle  + \left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle   \cdot  \sqrt{1-\langle \mathbf{o},\mathbf{q}{\rangle }^{2}} \tag{10}
$$

where ${\mathbf{e}}_{1}$ is $\mathbf{q} - \langle \mathbf{q},\mathbf{o}\rangle \mathbf{o}$ with its norm normalized to be 1,i.e., ${\mathbf{e}}_{1} \mathrel{\text{:=}} \frac{\mathbf{q}-\langle \mathbf{q},\mathbf{o}\rangle \mathbf{o}}{\begin{Vmatrix}\mathbf{q}-\langle \mathbf{q},\mathbf{o}\rangle \mathbf{o}\end{Vmatrix}}$ . We note that $\mathbf{o} \bot  {\mathbf{e}}_{1}$ (since $\left. {\left\langle  {\mathbf{o},{\mathbf{e}}_{1}}\right\rangle   = 0}\right)$ and $\begin{Vmatrix}{\mathbf{e}}_{1}\end{Vmatrix} = 1$ .

其中 ${\mathbf{e}}_{1}$ 是范数归一化为 1 的 $\mathbf{q} - \langle \mathbf{q},\mathbf{o}\rangle \mathbf{o}$，即 ${\mathbf{e}}_{1} \mathrel{\text{:=}} \frac{\mathbf{q}-\langle \mathbf{q},\mathbf{o}\rangle \mathbf{o}}{\begin{Vmatrix}\mathbf{q}-\langle \mathbf{q},\mathbf{o}\rangle \mathbf{o}\end{Vmatrix}}$。我们注意到 $\mathbf{o} \bot  {\mathbf{e}}_{1}$（因为 $\left. {\left\langle  {\mathbf{o},{\mathbf{e}}_{1}}\right\rangle   = 0}\right)$ 且 $\begin{Vmatrix}{\mathbf{e}}_{1}\end{Vmatrix} = 1$）。

Recall that we target to estimate $\langle \mathbf{o},\mathbf{q}\rangle$ . If we exactly know the values of all the variables other than $\langle \mathbf{o},\mathbf{q}\rangle$ ,we can compute the exact value of $\langle \mathbf{o},\mathbf{q}\rangle$ by solving Equations (9) and (10). In particular, in Equations (9) and (10), $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ is the inner product between the quantized data vector and the data vector. Its value can be pre-computed in the index phase. $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ is the inner product between the quantized data vector and the query vector. Its value can be efficiently computed in the query phase (we will specify in Section 3.3 how it can be efficiently computed). Thus,when $\mathbf{o}$ and $\mathbf{q}$ are collinear,we can compute the value of $\langle \mathbf{o},\mathbf{q}\rangle$ exactly by solving Equation (9),i.e., $\langle \mathbf{o},\mathbf{q}\rangle  = \frac{\langle \mathbf{o},\mathbf{q}\rangle }{\langle \mathbf{o},\mathbf{o})}$ .

回顾一下，我们的目标是估计 $\langle \mathbf{o},\mathbf{q}\rangle$。如果我们确切知道除 $\langle \mathbf{o},\mathbf{q}\rangle$ 之外所有变量的值，我们可以通过求解方程 (9) 和 (10) 来计算 $\langle \mathbf{o},\mathbf{q}\rangle$ 的精确值。具体来说，在方程 (9) 和 (10) 中，$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ 是量化数据向量与数据向量的内积。其值可以在索引阶段预先计算。$\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 是量化数据向量与查询向量的内积。其值可以在查询阶段高效计算（我们将在 3.3 节中详细说明如何高效计算）。因此，当 $\mathbf{o}$ 和 $\mathbf{q}$ 共线时，我们可以通过求解方程 (9) 精确计算 $\langle \mathbf{o},\mathbf{q}\rangle$ 的值，即 $\langle \mathbf{o},\mathbf{q}\rangle  = \frac{\langle \mathbf{o},\mathbf{q}\rangle }{\langle \mathbf{o},\mathbf{o})}$。

When $\mathbf{o}$ and $\mathbf{q}$ are non-collinear (which is a more common case),in order to exactly solve the Equation (10),we need to know the value of $\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ . However,as ${\mathbf{e}}_{1}$ depends on both $\mathbf{o}$ and $\mathbf{q}$ (which can be seen by its definition), $\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ can be neither pre-computed in the index phase (because it depends on $\mathbf{q}$ ) nor computed efficiently in the query phase without accessing $\mathbf{o}$ .

当 $\mathbf{o}$ 和 $\mathbf{q}$ 不共线时（这是更常见的情况），为了精确求解方程 (10)，我们需要知道 $\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ 的值。然而，由于 ${\mathbf{e}}_{1}$ 同时依赖于 $\mathbf{o}$ 和 $\mathbf{q}$（从其定义可以看出），$\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ 既不能在索引阶段预先计算（因为它依赖于 $\mathbf{q}$），也不能在不访问 $\mathbf{o}$ 的情况下在查询阶段高效计算。

We notice that although we cannot efficiently compute the exact value of ${\left\langle  \bar{o},{\mathbf{e}}_{1}\right\rangle  }^{4}$ ,given the random nature of $\overline{\mathbf{o}}$ ,we explicitly know its distribution. Specifically,recall that we have sampled a random orthogonal matrix $P$ ,applied it to the codebook $\mathcal{C}$ and generated a randomized codebook ${C}_{\text{rand }}$ . 0 is a vector picked from the randomized codebook ${C}_{\text{rand }}$ and thus,it is a random vector. $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ and $\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ correspond to the projection of the random vector $\overline{\mathbf{o}}$ onto two fixed directions (i.e., the directions are $\mathbf{o}$ and ${\mathbf{e}}_{1}$ ,where $\mathbf{o} \bot  {\mathbf{e}}_{1}$ ). Thus,they are mutually correlated random variables.

我们注意到，尽管我们不能高效地计算 ${\left\langle  \bar{o},{\mathbf{e}}_{1}\right\rangle  }^{4}$ 的精确值，但鉴于 $\overline{\mathbf{o}}$ 的随机性，我们明确知道其分布。具体来说，回顾一下，我们采样了一个随机正交矩阵 $P$，将其应用于码本 $\mathcal{C}$ 并生成了一个随机化码本 ${C}_{\text{rand }}$。${\left\langle  \bar{o},{\mathbf{e}}_{1}\right\rangle  }^{4}$ 是从随机化码本 ${C}_{\text{rand }}$ 中选取的一个向量，因此，它是一个随机向量。$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ 和 $\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ 对应于随机向量 $\overline{\mathbf{o}}$ 在两个固定方向上的投影（即方向分别为 $\mathbf{o}$ 和 ${\mathbf{e}}_{1}$，其中 $\mathbf{o} \bot  {\mathbf{e}}_{1}$）。因此，它们是相互关联的随机变量。

We rigorously analyze the distributions of the random variables. The core conclusions of the analysis are briefly summarized as follows while the detailed presentation and proof are left in the technical report [33] due to the page limit. Specifically,our analysis indicates that when $D$ ranges from ${10}^{2}$ to ${10}^{6}$ ,it is always true that $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ has the expectation ${}^{5}$ of around 0.8 and $\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ has the expectation of exactly 0 . It further indicates that, with high probability, these random variables would not deviate from their expectation by $\Omega \left( {1/\sqrt{D}}\right)$ . This conclusion quantitatively presents the extent that the random variables concentrate around their expected values, which will be used later for analyzing the error bound of our estimator. To empirically verify our analysis, we repeatedly and independently sample the random orthogonal matrices $P{10}^{5}$ times for a pair of fixed $\mathbf{o},\mathbf{q}$ in the 128-dimensional space. The right panel of Figure 1 visualizes the projection of $\overline{\mathbf{o}}$ on the 2-dimensional space spanned by $\mathbf{o},\mathbf{q}$ with the red point cloud (each point represents the projection of an $\overline{\mathbf{o}}$ based on a sampled random matrix $P$ ). In particular, $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ (the $\mathrm{x}$ -axis) is shown to be concentrated around ${0.8}.\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ (the $y$ -axis) is concentrated and symmetrically distributed around 0 , which verifies our theoretical analysis perfectly.

我们严格分析了随机变量的分布。分析的核心结论简要总结如下，由于篇幅限制，详细的表述和证明留于技术报告[33]中。具体而言，我们的分析表明，当$D$在${10}^{2}$到${10}^{6}$的范围内时，$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$的期望${}^{5}$始终约为0.8，而$\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$的期望恰好为0。进一步表明，这些随机变量以高概率不会偏离其期望$\Omega \left( {1/\sqrt{D}}\right)$。这一结论定量地呈现了随机变量在其期望值附近的集中程度，后续将用于分析我们估计量的误差界。为了通过实验验证我们的分析，对于128维空间中的一对固定的$\mathbf{o},\mathbf{q}$，我们独立重复采样随机正交矩阵$P{10}^{5}$次。图1的右半部分用红色点云可视化了$\overline{\mathbf{o}}$在由$\mathbf{o},\mathbf{q}$张成的二维空间上的投影（每个点代表基于采样的随机矩阵$P$得到的$\overline{\mathbf{o}}$的投影）。特别地，$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$（$\mathrm{x}$轴）显示集中在${0.8}.\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$附近，$y$（$y$轴）集中且对称分布在0附近，这完美地验证了我们的理论分析。

---

<!-- Footnote -->

${}^{4}$ In particular,when we say "computing the value" of a random variable,it refers to computing its observed value based on a certain sampled $P$ .

${}^{4}$特别地，当我们说计算一个随机变量的“值”时，指的是基于某个采样的$P$计算其观测值。

<!-- Footnote -->

---

3.2.2 Constructing an Unbiased Estimator for $\langle \mathbf{o},\mathbf{q}\rangle$ . Based on our analysis on Equation (9),for the case that $\mathbf{o},\mathbf{q}$ are collinear, $\langle \mathbf{o},\mathbf{q}\rangle$ can be explicitly solved by $\langle \mathbf{o},\mathbf{q}\rangle  = \frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }$ . Thus,it is natural to conjecture that for the case that $\mathbf{o},\mathbf{q}$ are non-collinear, $\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }$ should also be a good estimator for $\langle \mathbf{o},\mathbf{q}\rangle$ . We thus deduce from it as follows.

3.2.2 为$\langle \mathbf{o},\mathbf{q}\rangle$构造无偏估计量。基于我们对方程(9)的分析，对于$\mathbf{o},\mathbf{q}$共线的情况，可以通过$\langle \mathbf{o},\mathbf{q}\rangle  = \frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }$显式求解$\langle \mathbf{o},\mathbf{q}\rangle$。因此，很自然地可以推测，对于$\mathbf{o},\mathbf{q}$不共线的情况，$\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }$也应该是$\langle \mathbf{o},\mathbf{q}\rangle$的一个良好估计量。我们由此推导如下。

$$
\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle } = \frac{\langle \overline{\mathbf{o}},\mathbf{o}\rangle \cdot \langle \mathbf{o},\mathbf{q}\rangle  + \left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle   \cdot  \sqrt{1-\langle \mathbf{o},\mathbf{q}{\rangle }^{2}}}{\langle \overline{\mathbf{o}},\mathbf{o}\rangle } \tag{11}
$$

$$
 = \langle \mathbf{o},\mathbf{q}\rangle  + \sqrt{1-\langle \mathbf{o},\mathbf{q}{\rangle }^{2}} \cdot  \frac{\left\langle  \overline{\mathbf{o}},{\mathbf{e}}_{1}\right\rangle  }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle } \tag{12}
$$

where Equation (11) is by Equation (10) and Equation (12) simplifies Equation (11). We note that the last term in Equation (12) can be viewed as the error term of the estimator. Recall that based on our analysis in Section 3.2.1, $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ is concentrated around ${0.8}.\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$ has the expectation of 0 and is concentrated. It implies that the error term has 0 expectation and will not deviate largely from 0 due to the concentration. The following theorem presents the specific results. The rigorous proof can be found in the technical report [33].

其中方程(11)由方程(10)得出，方程(12)简化了方程(11)。我们注意到，方程(12)中的最后一项可以看作是估计量的误差项。回顾我们在3.2.1节的分析，$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$集中在${0.8}.\left\langle  {\overline{\mathbf{o}},{\mathbf{e}}_{1}}\right\rangle$附近，其期望为0且是集中的。这意味着误差项的期望为0，并且由于集中性不会大幅偏离0。以下定理给出了具体结果。严格的证明可以在技术报告[33]中找到。

THEOREM 3.2 (ESTIMATOR). The unbiasedness is given as

定理3.2（估计量）。无偏性表示为

$$
\mathbb{E}\left\lbrack  \frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }\right\rbrack   = \langle \mathbf{o},\mathbf{q}\rangle  \tag{13}
$$

The error bound of the estimator is given as

估计量的误差界表示为

$$
\mathbb{P}\left\{  {\left| {\frac{\langle \widehat{\mathbf{o}},\mathbf{q}\rangle }{\langle \widehat{\mathbf{o}},\mathbf{o}\rangle }-\langle \mathbf{o},\mathbf{q}\rangle }\right|  > \sqrt{\frac{1-\langle \widehat{\mathbf{o}},\mathbf{o}{\rangle }^{2}}{\langle \widehat{\mathbf{o}},\mathbf{o}{\rangle }^{2}}} \cdot  \frac{{\epsilon }_{0}}{\sqrt{D - 1}}}\right\}   \leq  2{e}^{-{c}_{0}{\epsilon }_{0}^{2}} \tag{14}
$$

where ${\epsilon }_{0}$ is a parameter which controls the failure probability. ${c}_{0}$ is a constant factor. The error bound can be concisely presented as

其中${\epsilon }_{0}$是一个控制失败概率的参数。${c}_{0}$是一个常数因子。误差界可以简洁地表示为

$$
\left| {\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }-\langle \mathbf{o},\mathbf{q}\rangle }\right|  = O\left( \frac{1}{\sqrt{D}}\right) \text{with high probability} \tag{15}
$$

Due to Equations (2) and (13),the unbiased estimator of $\langle \mathbf{o},\mathbf{q}\rangle$ can further induce an unbiased estimator of the squared distance between the raw data and query vectors. We provide empirical verification on the unbiasedness in Section 5.2.6. Besides, we would like to highlight that based on similar analysis,an alternative estimator $\langle \mathbf{o},\mathbf{q}\rangle  \approx  \langle \overline{\mathbf{o}},\mathbf{q}\rangle$ ,i.e.,by simply treating the quantized data vector as the data vector as PQ does, can be easily proved to be biased.

根据方程(2)和(13)，$\langle \mathbf{o},\mathbf{q}\rangle$的无偏估计量可以进一步推导出原始数据与查询向量之间平方距离的无偏估计量。我们在5.2.6节对无偏性进行了实证验证。此外，我们想强调的是，基于类似的分析，可以很容易地证明另一个估计量$\langle \mathbf{o},\mathbf{q}\rangle  \approx  \langle \overline{\mathbf{o}},\mathbf{q}\rangle$（即像乘积量化（PQ）那样简单地将量化数据向量视为数据向量）是有偏的。

---

<!-- Footnote -->

${}^{5}$ The exact expected value is $\mathbb{E}\left\lbrack  {\langle \overline{\mathbf{o}},\mathbf{o}\rangle }\right\rbrack   = \sqrt{\frac{D}{\pi }}\frac{{2\Gamma }\left( \frac{D}{2}\right) }{\left( {D - 1}\right) \Gamma \left( \frac{D - 1}{2}\right) }$ ,where $\Gamma \left( \cdot \right)$ is the Gamma function. The expected value ranges from 0.798 to 0.800 for $D \in  \left\lbrack  {{10}^{2},{10}^{6}}\right\rbrack$ .

${}^{5}$ 精确的期望值是$\mathbb{E}\left\lbrack  {\langle \overline{\mathbf{o}},\mathbf{o}\rangle }\right\rbrack   = \sqrt{\frac{D}{\pi }}\frac{{2\Gamma }\left( \frac{D}{2}\right) }{\left( {D - 1}\right) \Gamma \left( \frac{D - 1}{2}\right) }$，其中$\Gamma \left( \cdot \right)$是伽马函数（Gamma function）。对于$D \in  \left\lbrack  {{10}^{2},{10}^{6}}\right\rbrack$，期望值的范围是0.798到0.800。

<!-- Footnote -->

---

Equation (14) presents the error bound of our estimator. In particular,it presents a $1 - 2\exp \left( {-{c}_{0}{\epsilon }_{0}^{2}}\right)$ confidence interval

方程(14)给出了我们估计量的误差界。具体来说，它给出了一个$1 - 2\exp \left( {-{c}_{0}{\epsilon }_{0}^{2}}\right)$置信区间

$$
\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle } \pm  \sqrt{\frac{1-\langle \overline{\mathbf{o}},\mathbf{o}{\rangle }^{2}}{\langle \overline{\mathbf{o}},\mathbf{o}{\rangle }^{2}}} \cdot  \frac{{\epsilon }_{0}}{\sqrt{D - 1}} \tag{16}
$$

We note that the failure probability (i.e., the probability that the confidence interval does not cover the true value of $\left\langle  {\mathbf{o},\mathbf{q}}\right\rangle$ is $2\exp \left( {-{c}_{0}{\epsilon }_{0}^{2}}\right)$ . It decays in a quadratic-exponential trend wrt ${\epsilon }_{0}$ ,which is extremely fast. The length of the confidence interval grows linearly wrt ${\epsilon }_{0}$ . Thus, ${\epsilon }_{0} = \Theta \left( \sqrt{\log \left( {1/\delta }\right) }\right)$ corresponds to a failure probability of at most $\delta$ ,which indicates that a short confidence interval can correspond to a high confidence level. In practice, ${\epsilon }_{0}$ is fixed to be 1.9 in pursuit of nearly perfect confidence (see Section 5.2.4 for the empirical verification study). Recall that $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ is concentrated around 0.8 . Based on the values of ${\epsilon }_{0}$ and $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$ ,the error bound can be further concisely presented as Equation (15),i.e.,it guarantees an error bound of $O\left( {1/\sqrt{D}}\right)$ . According to a recent theoretical study [3],for $D$ -dimensional vectors,with a short code of $D$ bits,it is impossible in theory for a method to provide a bound which is tighter than $O\left( {1/\sqrt{D}}\right)$ (the failure probability is viewed as a constant). Thus, Equation (15) indicates that RaBitQ's error bound is sharp, i.e., it achieves the asymptotic optimality. The error bound will be later used in ANN search to determine whether a data vector should be re-ranked (see Section 4).

我们注意到，失败概率（即置信区间不包含$\left\langle  {\mathbf{o},\mathbf{q}}\right\rangle$真实值的概率）是$2\exp \left( {-{c}_{0}{\epsilon }_{0}^{2}}\right)$。它相对于${\epsilon }_{0}$呈二次指数衰减趋势，这是非常快的。置信区间的长度相对于${\epsilon }_{0}$呈线性增长。因此，${\epsilon }_{0} = \Theta \left( \sqrt{\log \left( {1/\delta }\right) }\right)$对应的失败概率至多为$\delta$，这表明较短的置信区间可以对应较高的置信水平。在实践中，为了追求近乎完美的置信度，${\epsilon }_{0}$固定为1.9（实证验证研究见5.2.4节）。回顾一下，$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$集中在0.8左右。根据${\epsilon }_{0}$和$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$的值，误差界可以进一步简洁地表示为方程(15)，即它保证了一个$O\left( {1/\sqrt{D}}\right)$的误差界。根据最近的一项理论研究[3]，对于$D$维向量，使用$D$位的短码，理论上一种方法不可能提供比$O\left( {1/\sqrt{D}}\right)$更紧的界（失败概率视为常数）。因此，方程(15)表明RaBitQ的误差界是精确的，即它达到了渐近最优性。误差界稍后将在近似最近邻（ANN）搜索中用于确定是否应该对数据向量进行重新排序（见第4节）。

Furthermore, we note that RaBitQ provides an error bound in an additive form [3] (i.e., absolute error). When the data vectors are well normalized (recall that in Section 3.1.1 we normalize the data vectors), the bound can be pushed forward to a multiplicative form [41] (i.e., relative error). We leave the detailed discussion in the technical report [33] because it is based on an assumption that the data vectors are well normalized. Note that all other theoretical results introduced in this paper do not rely on any assumptions on the data, i.e., the additive bound holds regardless of the data distribution. In the present work, we adopt a simple and natural method of normalization (i.e., with the centroids of IVF as will be introduced in Section 4) to instantiate our scheme of quantization, while we have yet to extensively explore the normalization step itself. We shall leave it as future work to rigorously study the normalization problem.

此外，我们注意到RaBitQ（随机比特量化）以加法形式 [3] 提供了一个误差界（即绝对误差）。当数据向量被很好地归一化时（回顾一下，在3.1.1节中我们对数据向量进行了归一化），该误差界可以推进到乘法形式 [41]（即相对误差）。我们将详细讨论放在技术报告 [33] 中，因为这是基于数据向量被很好地归一化这一假设。请注意，本文介绍的所有其他理论结果都不依赖于对数据的任何假设，即无论数据分布如何，加法误差界都成立。在当前工作中，我们采用一种简单自然的归一化方法（即使用第4节将介绍的倒排文件（IVF）质心）来实例化我们的量化方案，而我们尚未对归一化步骤本身进行广泛探索。我们将把严格研究归一化问题留作未来的工作。

### 3.3 Computing the Estimator Efficiently

### 3.3 高效计算估计量

Recall that $\frac{\langle \widetilde{\mathbf{o}},\mathbf{q}\rangle }{\langle \widetilde{\mathbf{o}},\mathbf{o}\rangle }$ is the estimator. Since $\langle \widetilde{\mathbf{o}},\mathbf{o}\rangle$ has been pre-computed during the index phase,the remaining question is to compute the value of $\langle \overline{0},q\rangle$ efficiently. For the sake of convenience,we denote ${P}^{-1}\mathbf{q}$ as ${\mathbf{q}}^{\prime }$ . Like what we do in Section 3.1.3,in order to compute $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ ,we can compute $\left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle$ ,which can be verified as follows.

回顾一下，$\frac{\langle \widetilde{\mathbf{o}},\mathbf{q}\rangle }{\langle \widetilde{\mathbf{o}},\mathbf{o}\rangle }$ 是估计量。由于 $\langle \widetilde{\mathbf{o}},\mathbf{o}\rangle$ 已在索引阶段预先计算，剩下的问题是高效计算 $\langle \overline{0},q\rangle$ 的值。为方便起见，我们将 ${P}^{-1}\mathbf{q}$ 记为 ${\mathbf{q}}^{\prime }$。与我们在3.1.3节中所做的一样，为了计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$，我们可以计算 $\left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle$，这可以验证如下。

$$
\langle \overline{\mathbf{o}},\mathbf{q}\rangle  = \langle P\overline{\mathbf{x}},\mathbf{q}\rangle  = \left\langle  {{P}^{-1}P\overline{\mathbf{x}},{P}^{-1}\mathbf{q}}\right\rangle   = \left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle   \tag{17}
$$

3.3.1 Quantizing the Transformed Query Vector. Recall that $\overline{\mathbf{x}}$ is a bi-valued vector whose entries are $\pm  1/\sqrt{D}$ . It is represented and stored as a binary quantization code ${\bar{\mathbf{x}}}_{b}$ as is discussed in Section 3.1.3. ${\mathbf{q}}^{\prime }$ is a real-valued vector,whose entries are conventionally represented by floating-point numbers (floats in short). We note that in our method,representing the entries of ${\mathbf{q}}^{\prime }$ with floats is an overkill. Specifically,recall that our method adopts $\frac{\langle \delta ,q\rangle }{\langle \delta ,0\rangle }$ as an estimator of $\langle o,q\rangle$ . Even if we obtain a perfectly accurate result in the computation of $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ ,our estimation of $\langle \mathbf{o},\mathbf{q}\rangle$ is still approximate. Thus,instead of exactly computing $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ ,we aim to guarantee that the error produced in the computation of $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ is much smaller than the error of the estimator itself.

3.3.1 量化变换后的查询向量。回顾一下，$\overline{\mathbf{x}}$ 是一个二值向量，其元素为 $\pm  1/\sqrt{D}$。如3.1.3节所讨论的，它被表示并存储为二进制量化码 ${\bar{\mathbf{x}}}_{b}$。${\mathbf{q}}^{\prime }$ 是一个实值向量，其元素通常用浮点数（简称float）表示。我们注意到，在我们的方法中，用浮点数表示 ${\mathbf{q}}^{\prime }$ 的元素有些过度了。具体来说，回顾一下，我们的方法采用 $\frac{\langle \delta ,q\rangle }{\langle \delta ,0\rangle }$ 作为 $\langle o,q\rangle$ 的估计量。即使我们在计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 时得到了完全准确的结果，我们对 $\langle \mathbf{o},\mathbf{q}\rangle$ 的估计仍然是近似的。因此，我们的目标不是精确计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$，而是确保在计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 时产生的误差远小于估计量本身的误差。

Specifically,we apply uniform scalar quantization on the entries of ${\mathbf{q}}^{\prime }$ and represent them as ${B}_{q}$ -bit unsigned integers. We denote the $i$ th entry of the vector ${\mathbf{q}}^{\prime }$ as ${\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack$ . Let ${v}_{l} \mathrel{\text{:=}} \mathop{\min }\limits_{{1 \leq  i \leq  D}}{\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack  ,{v}_{r} \mathrel{\text{:=}}$ $\mathop{\max }\limits_{{1 \leq  i \leq  D}}{\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack$ and $\Delta  \mathrel{\text{:=}} \left( {{v}_{r} - {v}_{l}}\right) /\left( {{2}^{{B}_{q}} - 1}\right)$ . The uniform scalar quantization uniformly splits the range of the values $\left\lbrack  {{v}_{l},{v}_{r}}\right\rbrack$ into ${2}^{{B}_{q}} - 1$ segments,where each segment has the length of $\Delta$ . Then for a value $v = {v}_{l} + m \cdot  \Delta  + t,m = 0,1,\ldots ,{2}^{{B}_{q}} - 1,t \in  \lbrack 0,\Delta )$ ,the method quantizes it by rounding it up to its nearest boundary of the segments (i.e., ${v}_{l} + m \cdot  \Delta$ or ${v}_{l} + \left( {m + 1}\right)  \cdot  \Delta$ ) and representing it with the corresponding ${B}_{q}$ -bit unsigned integer (i.e., $m$ or $m + 1$ ). Let $\overline{\mathbf{q}}$ be the vector whose entries are equal to the quantized values of the entries of ${\mathbf{q}}^{\prime }$ (we term it as the quantized query vector) and ${\overline{\mathbf{q}}}_{u}$ be its ${B}_{q}$ -bit unsigned integer representation,where $\overline{\mathbf{q}} = \Delta  \cdot  {\overline{\mathbf{q}}}_{u} + {v}_{l} \cdot  {\mathbf{1}}_{D}$ (recall that ${\mathbf{1}}_{D}$ is the $D$ -dimensional vector with all its entries as ones). Then,we can compute $\langle \bar{\mathbf{x}},\bar{\mathbf{q}}\rangle$ as an approximation of $\left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle$ .

具体而言，我们对${\mathbf{q}}^{\prime }$的元素应用均匀标量量化，并将它们表示为${B}_{q}$位无符号整数。我们将向量${\mathbf{q}}^{\prime }$的第$i$个元素表示为${\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack$。设${v}_{l} \mathrel{\text{:=}} \mathop{\min }\limits_{{1 \leq  i \leq  D}}{\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack  ,{v}_{r} \mathrel{\text{:=}}$ $\mathop{\max }\limits_{{1 \leq  i \leq  D}}{\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack$和$\Delta  \mathrel{\text{:=}} \left( {{v}_{r} - {v}_{l}}\right) /\left( {{2}^{{B}_{q}} - 1}\right)$。均匀标量量化将值$\left\lbrack  {{v}_{l},{v}_{r}}\right\rbrack$的范围均匀地划分为${2}^{{B}_{q}} - 1$个区间，其中每个区间的长度为$\Delta$。然后，对于一个值$v = {v}_{l} + m \cdot  \Delta  + t,m = 0,1,\ldots ,{2}^{{B}_{q}} - 1,t \in  \lbrack 0,\Delta )$，该方法通过将其向上舍入到最接近的区间边界（即${v}_{l} + m \cdot  \Delta$或${v}_{l} + \left( {m + 1}\right)  \cdot  \Delta$）并使用相应的${B}_{q}$位无符号整数（即$m$或$m + 1$）来表示它进行量化。设$\overline{\mathbf{q}}$为其元素等于${\mathbf{q}}^{\prime }$元素的量化值的向量（我们将其称为量化查询向量），${\overline{\mathbf{q}}}_{u}$为其${B}_{q}$位无符号整数表示，其中$\overline{\mathbf{q}} = \Delta  \cdot  {\overline{\mathbf{q}}}_{u} + {v}_{l} \cdot  {\mathbf{1}}_{D}$（回想一下，${\mathbf{1}}_{D}$是所有元素都为1的$D$维向量）。然后，我们可以计算$\langle \bar{\mathbf{x}},\bar{\mathbf{q}}\rangle$作为$\left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle$的近似值。

Furthermore, to retain the theoretical guarantee, we adopt the trick of randomizing the uniform scalar quantization [3, 92]. Specifically, unlike the conventional method which rounds up a value to its nearest boundary of the segments, the randomized method rounds it to its left or right boundary randomly. The rationale is that for a value $v = {v}_{l} + m \cdot  \Delta  + t,m = 0,1,\ldots ,{2}^{{B}_{q}} - 1,t \in  \lbrack 0,\Delta )$ ,when it is rounded to ${v}_{l} + m \cdot  \Delta$ ,it will cause an error of under-estimation $- t < 0$ . When it is rounded to ${v}_{l} + \left( {m + 1}\right)  \cdot  \Delta$ ,it will cause an error of over-estimation $\Delta  - t > 0$ . If we assign $1 - t/\Delta$ probability to the former event and $t/\Delta$ probability to the latter event,the expected error would be 0,which makes the computation unbiased. We note that this operation can be easily achieved by letting

此外，为了保留理论保证，我们采用了对均匀标量量化进行随机化的技巧[3, 92]。具体来说，与将值向上舍入到最接近的区间边界的传统方法不同，随机化方法将其随机舍入到左边界或右边界。其原理是，对于一个值$v = {v}_{l} + m \cdot  \Delta  + t,m = 0,1,\ldots ,{2}^{{B}_{q}} - 1,t \in  \lbrack 0,\Delta )$，当它被舍入到${v}_{l} + m \cdot  \Delta$时，会导致低估误差$- t < 0$。当它被舍入到${v}_{l} + \left( {m + 1}\right)  \cdot  \Delta$时，会导致高估误差$\Delta  - t > 0$。如果我们为前一个事件分配$1 - t/\Delta$的概率，为后一个事件分配$t/\Delta$的概率，那么预期误差将为0，这使得计算是无偏的。我们注意到，通过让

$$
{\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack   \mathrel{\text{:=}} \left\lfloor  {\frac{{\mathbf{q}}^{\prime }\left\lbrack  i\right\rbrack   - {v}_{l}}{\Delta } + {u}_{i}}\right\rfloor   \tag{18}
$$

where ${u}_{i}$ is sampled from the uniform distribution on $\left\lbrack  {0,1}\right\rbrack$ . Moreover,based on the randomized method,we can analyze the minimum ${B}_{q}$ needed for making the error introduced by the uniform scalar quantization negligible. The result is presented with the following theorem. The detailed proof can be found in the technical report [33].

其中 ${u}_{i}$ 是从 $\left\lbrack  {0,1}\right\rbrack$ 上的均匀分布中采样得到的。此外，基于随机化方法，我们可以分析使均匀标量量化引入的误差可忽略不计所需的最小 ${B}_{q}$。结果由以下定理给出。详细证明可在技术报告 [33] 中找到。

THEOREM 3.3. ${B}_{q} = \Theta \left( {\log \log D}\right)$ suffices to guarantee that $\left| {\left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle   - \langle \overline{\mathbf{x}},\overline{\mathbf{q}}\rangle }\right|  = O\left( {1/\sqrt{D}}\right)$ with high probability.

定理 3.3。${B}_{q} = \Theta \left( {\log \log D}\right)$ 足以以高概率保证 $\left| {\left\langle  {\overline{\mathbf{x}},{\mathbf{q}}^{\prime }}\right\rangle   - \langle \overline{\mathbf{x}},\overline{\mathbf{q}}\rangle }\right|  = O\left( {1/\sqrt{D}}\right)$。

Recall that the estimator has the error of $O\left( {1/\sqrt{D}}\right)$ (see Section 3.2.2). The above theorem shows that setting ${B}_{q} = \Theta \left( {\log \log D}\right)$ suffices to guarantee that the error introduced by the uniform scalar quantization is at the same order as the error introduced by estimator. Because the error decreases exponentially wrt ${B}_{q}$ ,increasing ${B}_{q}$ by a small constant (i.e., ${B}_{q}$ is still at the order of $\Theta \left( {\log \log D}\right)$ ) guarantees that the error is much smaller than that of the estimator. We provide the empirical verification study for ${B}_{q}$ in Section 5.2.5. The result shows that when ${B}_{q} = 4$ ,the error introduced by the uniform scalar quantization would be negligible.

回顾一下，估计器的误差为 $O\left( {1/\sqrt{D}}\right)$（见 3.2.2 节）。上述定理表明，设置 ${B}_{q} = \Theta \left( {\log \log D}\right)$ 足以保证均匀标量量化引入的误差与估计器引入的误差处于同一数量级。由于误差相对于 ${B}_{q}$ 呈指数下降，将 ${B}_{q}$ 增加一个小常数（即 ${B}_{q}$ 仍处于 $\Theta \left( {\log \log D}\right)$ 的数量级）可确保该误差远小于估计器的误差。我们在 5.2.5 节中对 ${B}_{q}$ 进行了实证验证研究。结果表明，当 ${B}_{q} = 4$ 时，均匀标量量化引入的误差可以忽略不计。

3.3.2 Computing $\langle \bar{\mathbf{x}},\bar{\mathbf{q}}\rangle$ Efficiently. We next present how to compute $\langle \bar{\mathbf{x}},\bar{\mathbf{q}}\rangle$ efficiently. We first express $\langle \overline{\mathbf{x}},\overline{\mathbf{q}}\rangle$ with ${\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}$ as follows.

3.3.2 高效计算 $\langle \bar{\mathbf{x}},\bar{\mathbf{q}}\rangle$。接下来，我们介绍如何高效计算 $\langle \bar{\mathbf{x}},\bar{\mathbf{q}}\rangle$。首先，我们用 ${\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}$ 表示 $\langle \overline{\mathbf{x}},\overline{\mathbf{q}}\rangle$ 如下。

$$
\langle \overline{\mathbf{x}},\overline{\mathbf{q}}\rangle  = \left\langle  {\frac{2{\overline{\mathbf{x}}}_{b} - {\mathbf{1}}_{D}}{\sqrt{D}},\Delta  \cdot  {\overline{\mathbf{q}}}_{u} + {v}_{l} \cdot  {\mathbf{1}}_{D}}\right\rangle   \tag{19}
$$

$$
 = \frac{2\Delta }{\sqrt{D}}\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle   + \frac{2{v}_{l}}{\sqrt{D}}\mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   - \frac{\Delta }{\sqrt{D}}\mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack   - \sqrt{D} \cdot  {v}_{l} \tag{20}
$$

Note that the factors $\Delta$ and ${v}_{l}$ are known when we quantize the query vector. $\mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack$ corresponds to the number of 1 ’s in the bit string ${\overline{\mathbf{x}}}_{b}$ ,which can be pre-computed during the index phase. $\mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack$ depends only on the query vector. Its cost of computation can be shared by all the data vectors. The remaining task is to compute $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ where the coordinates of ${\overline{\mathbf{x}}}_{b}$ are 0 or 1 and those of ${\overline{\mathbf{q}}}_{u}$ are unsigned ${B}_{q}$ -bit integers.

注意，当我们对查询向量进行量化时，因子 $\Delta$ 和 ${v}_{l}$ 是已知的。$\mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack$ 对应于位串 ${\overline{\mathbf{x}}}_{b}$ 中 1 的数量，这可以在索引阶段预先计算。$\mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack$ 仅取决于查询向量。其计算成本可以由所有数据向量共享。剩下的任务是计算 $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$，其中 ${\overline{\mathbf{x}}}_{b}$ 的坐标为 0 或 1，${\overline{\mathbf{q}}}_{u}$ 的坐标为无符号 ${B}_{q}$ 位整数。

We provide two versions of fast computation for $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ . The first version targets the case of a single quantization code, as the original implementation of PQ [45] does. The second version targets the case of a packed batch of quantization codes, as the fast SIMD-based implementation of PQ [4, 5] does. We note that in general, both our method and PQ have higher throughput in the second case than that in the first case, i.e., they estimate the distances for more quantization codes within certain time. We note that the second case requires the quantization codes to be packed in a batch, which is feasible in some certain scenarios only.

我们为 $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ 提供了两种快速计算版本。第一个版本针对单个量化码的情况，就像乘积量化（PQ）[45] 的原始实现那样。第二个版本针对一批打包的量化码的情况，就像基于单指令多数据（SIMD）的 PQ 快速实现 [4, 5] 那样。我们注意到，一般来说，在第二种情况下，我们的方法和 PQ 的吞吐量都比第一种情况高，即在一定时间内为更多的量化码估计距离。我们注意到，第二种情况要求量化码以批处理方式打包，这仅在某些特定场景下可行。

<!-- Media -->

<!-- figureText: ${\bar{\mathbf{q}}}_{u}\left\lbrack  1\right\rbrack$ ${\bar{\mathbf{q}}}_{u}\left\lbrack  D\right\rbrack$ ._____ ${\overline{\mathbf{x}}}_{b}$ ${\mathbf{q}}_{u}$ -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_11.jpg?x=329&y=639&w=908&h=362&r=0"/>

Fig. 2. Bitwise Decomposition of ${\overline{\mathbf{q}}}_{u}$ .

图 2. ${\overline{\mathbf{q}}}_{u}$ 的按位分解。

<!-- Media -->

For the first case where the estimation of the distance is for a query vector and a single quantization code,we note that an unsigned ${B}_{q}$ -bit integer can be decomposed into ${B}_{q}$ binary values as shown in Figure 2. The left panel represents the naive computation of $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ . The right panel represents the proposed bitwise computation of $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ . Let ${\overline{\mathbf{q}}}_{u}^{\left( j\right) }\left\lbrack  i\right\rbrack   \in  \{ 0,1\}$ be the $j$ th bit of ${\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack$ where $0 \leq  j < {B}_{q}$ . The following expression specifies the idea.

对于第一种情况，即对查询向量和单个量化码进行距离估计时，我们注意到一个无符号 ${B}_{q}$ 位整数可以分解为 ${B}_{q}$ 个二进制值，如图2所示。左图表示 $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ 的朴素计算方法。右图表示所提出的 $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ 的按位计算方法。设 ${\overline{\mathbf{q}}}_{u}^{\left( j\right) }\left\lbrack  i\right\rbrack   \in  \{ 0,1\}$ 为 ${\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack$ 的第 $j$ 位，其中 $0 \leq  j < {B}_{q}$ 。以下表达式阐述了该思路。

$$
\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle   = \mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   \cdot  {\overline{\mathbf{q}}}_{u}\left\lbrack  i\right\rbrack   = \mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   \cdot  \mathop{\sum }\limits_{{j = 0}}^{{{B}_{q} - 1}}{\overline{\mathbf{q}}}_{u}^{\left( j\right) }\left\lbrack  i\right\rbrack   \cdot  {2}^{j} \tag{21}
$$

$$
 = \mathop{\sum }\limits_{{j = 0}}^{{{B}_{q} - 1}}{2}^{j} \cdot  \mathop{\sum }\limits_{{i = 1}}^{D}{\overline{\mathbf{x}}}_{b}\left\lbrack  i\right\rbrack   \cdot  {\overline{\mathbf{q}}}_{u}^{\left( j\right) }\left\lbrack  i\right\rbrack   = \mathop{\sum }\limits_{{j = 0}}^{{{B}_{q} - 1}}{2}^{j} \cdot  \left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}^{\left( j\right) }}\right\rangle   \tag{22}
$$

Equation (22) shows that $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ can be expressed as a weighted sum of the inner product of the binary vectors,i.e., $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}^{\left( j\right) }}\right\rangle$ for $0 \leq  j < {B}_{q}$ . In particular,we note that the inner product of binary vectors can be efficiently achieved by bitwise operations, i.e., bitwise-and and popcount (a.k.a.,bitcount). Thus,the computation of $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ is finally reduced to ${B}_{q}$ bitwise-and and popcount operations on $D$ -bit strings,which are well supported by virtually all platforms. As a comparison, we note that, as is comprehensively studied in [4], PQ relies on looking up LUTs in RAM, which cannot be implemented efficiently. Based on our experiments in Section 5.2.1, on average our method runs $3\mathrm{x}$ faster than PQ and OPQ (a popular variant of PQ [34]) while reaching the same accuracy.

公式(22)表明，$\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ 可以表示为二进制向量内积的加权和，即对于 $0 \leq  j < {B}_{q}$ 有 $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}^{\left( j\right) }}\right\rangle$ 。特别地，我们注意到二进制向量的内积可以通过按位运算（即按位与和位计数（也称为比特计数））高效实现。因此，$\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 的计算最终简化为对 $D$ 位字符串进行 ${B}_{q}$ 次按位与和位计数运算，几乎所有平台都很好地支持这些运算。作为对比，我们注意到，正如文献[4]中全面研究的那样，乘积量化（PQ，Product Quantization）依赖于在随机存取存储器（RAM，Random Access Memory）中查找查找表（LUT，Lookup Table），这无法高效实现。根据我们在5.2.1节中的实验，平均而言，我们的方法比PQ和OPQ（PQ的一种流行变体 [34]）快 $3\mathrm{x}$ 倍，同时达到相同的精度。

For the second case where the estimation of the distance is for a query vector and a packed batch of quantization codes, we note that our method can seamlessly adopt the same fast SIMD-based implementation $\left\lbrack  {4,5}\right\rbrack$ as $\mathrm{{PQ}}$ does. In particular,for a $D$ -bit string,we split it into $D/4$ sub-segments where each sub-segment has 4 bits. We then pre-process $D/4$ LUTs where each LUT has ${2}^{4}$ unsigned integers corresponding to the inner products between a sub-segment of ${\overline{\mathbf{q}}}_{u}$ and the ${2}^{4}$ possible binary strings of a 4-bit sub-segment. For a quantization code of a data vector, we can compute $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$ by looking up and accumulating the values in the LUTs for $D/4$ times. We note that the computation is reduced to exactly the form of PQ and thus can adopt the fast SIMD-based implementation seamlessly. Recall that our method has the quantization codes of $D$ bits by default while PQ and OPQ have the codes of ${2D}$ bits by default. Therefore,our method has better efficiency than PQ and OPQ for computing approximate distances based on quantized vectors. Furthermore, as is shown in Section 5.2.1, in the default setting, our method also achieves consistently better accuracy than PQ and OPQ despite that our method uses a shorter quantization code (i.e., $D$ v.s. ${2D})$ .

对于第二种情况，即对查询向量和一批打包的量化码进行距离估计时，我们注意到我们的方法可以无缝采用与 $\mathrm{{PQ}}$ 相同的基于单指令多数据（SIMD，Single Instruction Multiple Data）的快速实现方法 $\left\lbrack  {4,5}\right\rbrack$。特别地，对于一个 $D$ 位字符串，我们将其拆分为 $D/4$ 个子段，每个子段有4位。然后，我们预处理 $D/4$ 个查找表，每个查找表有 ${2}^{4}$ 个无符号整数，这些整数对应于 ${\overline{\mathbf{q}}}_{u}$ 的一个子段与4位子段的 ${2}^{4}$ 种可能的二进制字符串之间的内积。对于数据向量的量化码，我们可以通过在查找表中查找并累加 $D/4$ 次值来计算 $\left\langle  {{\overline{\mathbf{x}}}_{b},{\overline{\mathbf{q}}}_{u}}\right\rangle$。我们注意到，该计算最终简化为PQ的形式，因此可以无缝采用基于SIMD的快速实现方法。回想一下，我们的方法默认使用 $D$ 位的量化码，而PQ和OPQ默认使用 ${2D}$ 位的码。因此，在基于量化向量计算近似距离方面，我们的方法比PQ和OPQ更高效。此外，如5.2.1节所示，在默认设置下，尽管我们的方法使用更短的量化码（即 $D$ 位对比 ${2D})$ 位），但它的精度也始终优于PQ和OPQ。

### 3.4 Summary of RaBitQ

### 3.4 RaBitQ总结

We summarize the RaBitQ algorithm in Algorithm 1 (the index phase) and Algorithm 2 (the query phase). In the index phase, it takes a set of raw data vectors as inputs. It normalizes the set of vectors based on Section 3.1.1 (line 1), constructs the RaBitQ codebook by sampling a random orthogonal matrix $P$ based on Section 3.1.2 (line 2) and computes the quantization codes ${\overline{\mathbf{x}}}_{b}$ based on Section 3.1.3 (line 3). In the query phase, it takes a raw query vector, a set of IDs of the data vectors and the pre-processed variables about the RaBitQ method as inputs. It first inversely transforms, normalizes and quantizes the raw query vector (line 1-2). We note that the time cost of these steps can be shared by all the data vectors. Then for each input ID of the data vectors, it efficiently computes the value of $\underset{\left\langle  \mathbf{0},\mathbf{o}\right\rangle  }{\left\langle  \mathbf{0},\mathbf{q}\right\rangle  }$ based on Section 3.3.2,adopts it as an unbiased estimation of $\langle \mathbf{o},\mathbf{q}\rangle$ based on Section 3.2 and further computes an estimated distance between the raw query and the raw data vectors based on Section 3.1.1 (line 3-5).

我们在算法1（索引阶段）和算法2（查询阶段）中总结了RaBitQ算法。在索引阶段，该算法将一组原始数据向量作为输入。它根据3.1.1节对向量集进行归一化处理（第1行），根据3.1.2节通过对随机正交矩阵$P$进行采样来构建RaBitQ码本（第2行），并根据3.1.3节计算量化码${\overline{\mathbf{x}}}_{b}$（第3行）。在查询阶段，它将一个原始查询向量、一组数据向量的ID以及关于RaBitQ方法的预处理变量作为输入。它首先对原始查询向量进行逆变换、归一化和量化（第1 - 2行）。我们注意到，这些步骤的时间成本可以由所有数据向量共享。然后，对于数据向量的每个输入ID，它根据3.3.2节高效地计算$\underset{\left\langle  \mathbf{0},\mathbf{o}\right\rangle  }{\left\langle  \mathbf{0},\mathbf{q}\right\rangle  }$的值，根据3.2节将其用作$\langle \mathbf{o},\mathbf{q}\rangle$的无偏估计，并根据3.1.1节进一步计算原始查询向量与原始数据向量之间的估计距离（第3 - 5行）。

<!-- Media -->

Algorithm 1: RaBitQ (Index Phase)

算法1：RaBitQ（索引阶段）

---

	Input : A set of raw data vectors

	输入：一组原始数据向量

	Output:The sampled matrix $P$ ; the quantization code ${\overline{\mathbf{x}}}_{b}$ ; the pre-computed results of

	输出：采样矩阵$P$；量化码${\overline{\mathbf{x}}}_{b}$；$P$和${\overline{\mathbf{x}}}_{b}$的预计算结果

					$\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$ and $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$

					$\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$和$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$

	Normalize the set of vectors (Section 3.1.1)

	对向量集进行归一化处理（3.1.1节）

2 Sample a random orthogonal matrix $P$ to construct the codebook ${C}_{\text{rand }}$ (Section 3.1.2)

2. 采样一个随机正交矩阵$P$以构建码本${C}_{\text{rand }}$（3.1.2节）

	Compute the quantization codes ${\overline{\mathbf{x}}}_{b}$ (Section 3.1.3)

	计算量化码${\overline{\mathbf{x}}}_{b}$（3.1.3节）

	Pre-compute the values of $\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$ and $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$

	预计算$\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$和$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$的值

---

Algorithm 2: RaBitQ (Query Phase)

算法2：RaBitQ（查询阶段）

---

	Input : A raw query vector ${\mathbf{q}}_{r}$ ; the sampled matrix $P$ ; a set of IDs of the data vectors,their

	输入：一个原始查询向量${\mathbf{q}}_{r}$；采样矩阵$P$；一组数据向量的ID、它们的量化码${\mathbf{q}}_{r}$以及$P$和[latex2]的结果

						quantization codes ${\overline{\mathbf{x}}}_{b}$ and the results of $\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$ and $\langle \overline{\mathbf{o}},\mathbf{o}\rangle$

						量化码${\overline{\mathbf{x}}}_{b}$以及$\begin{Vmatrix}{{\mathbf{o}}_{r} - \mathbf{c}}\end{Vmatrix}$和$\langle \overline{\mathbf{o}},\mathbf{o}\rangle$的结果

	Output: A set of approximate distances between the raw query and the raw data vectors

	输出：一组原始查询向量与原始数据向量之间的近似距离

1 Normalize and inversely transform the raw query vector and obtain ${\mathbf{q}}^{\prime }$

1. 对原始查询向量进行归一化和逆变换，得到${\mathbf{q}}^{\prime }$

	Quantize ${\mathbf{q}}^{\prime }$ into $\overline{\mathbf{q}}$ (Section 3.3.1)

	将${\mathbf{q}}^{\prime }$量化为$\overline{\mathbf{q}}$（3.3.1节）

	foreach input ID of the data vectors do

	对数据向量的每个输入ID执行以下操作

			Compute the value of the estimator $\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }$ as an approximation of $\langle \mathbf{o},\mathbf{q}\rangle$ (Section 3.3.2)

			计算估计量$\frac{\langle \overline{\mathbf{o}},\mathbf{q}\rangle }{\langle \overline{\mathbf{o}},\mathbf{o}\rangle }$的值，作为$\langle \mathbf{o},\mathbf{q}\rangle$的近似值（3.3.2节）

			Compute an estimated distance between the raw query and the data vector based on

			基于以下内容计算原始查询与数据向量之间的估计距离

				Equation (2)

				公式(2)

---

<!-- Media -->

## 4 RABITQ FOR IN-MEMORY ANN SEARCH

## 4 用于内存中近似最近邻搜索的RaBitQ方法

Next we present the application of our method to the in-memory ANN search. We note that the popular quantization method PQx4fs has been used in combination with the inverted-file-based indexes such as IVF [45] or the graph-based indexes such as NGT-QG [43] for in-memory ANN search. The combination of a quantization method with IVF can be easily done without much efforts. For example, we can use the quantization method to estimate the distances between the data vectors in the clusters that are probed, which decide those vectors to be re-ranked based on exact distances. In this case, batches of data vectors can be formed and the SIMD-based fast implementation (i.e., PQx4fs) can be adopted. Nevertheless, the combination of a quantization method with graph-based methods such as NGT-QG would require much more efforts in order to make the combined method work competitively in the in-memory setting, which would be of independent interest. This is because in graph-based methods, the vectors to be searched are decided one after one based on the greedy search process in the run-time, and it is not easy to form batches of them so that SIMD-based fast implementation can be adopted. Therefore, we apply our method in combination with IVF index [45] in this paper. We leave it as future work to apply our quantization method in graph-based methods.

接下来，我们介绍我们的方法在内存中近似最近邻（ANN）搜索中的应用。我们注意到，流行的量化方法PQx4fs已与基于倒排文件的索引（如IVF [45]）或基于图的索引（如NGT - QG [43]）结合用于内存中ANN搜索。量化方法与IVF的结合可以轻松完成，无需太多努力。例如，我们可以使用量化方法来估计被探查的聚类中数据向量之间的距离，这些距离决定了哪些向量将基于精确距离进行重新排序。在这种情况下，可以形成批量的数据向量，并采用基于单指令多数据（SIMD）的快速实现（即PQx4fs）。然而，量化方法与基于图的方法（如NGT - QG）的结合需要付出更多努力，才能使组合后的方法在内存环境中具有竞争力，这本身就很值得研究。这是因为在基于图的方法中，要搜索的向量是在运行时基于贪心搜索过程逐个确定的，并且不容易形成批量向量以采用基于SIMD的快速实现。因此，在本文中，我们将我们的方法与IVF索引[45]结合使用。将我们的量化方法应用于基于图的方法留作未来的工作。

We present the workflow of the RaBitQ method with the IVF index as follows. During the index phase, for a set of raw data vectors, the IVF algorithm first clusters them with the KMeans algorithm, builds a bucket for each cluster and assigns the vectors to their corresponding buckets. Our method then normalizes the raw data vectors based on the centroids of their corresponding clusters and feeds the set of the normalized vectors to the subsequent steps of our RaBitQ method. During the query phase,for a raw query vector,the algorithm selects the first ${N}_{\text{probe }}$ clusters whose centroids are nearest to the query. Then for each selected cluster, the algorithm retrieves all the quantization codes and estimates their distances based on the quantization codes, which decide the vectors to be re-ranked based on exact distances.

我们将RaBitQ方法与IVF索引结合的工作流程介绍如下。在索引阶段，对于一组原始数据向量，IVF算法首先使用K均值算法对它们进行聚类，为每个聚类创建一个桶，并将向量分配到相应的桶中。然后，我们的方法根据其对应聚类的质心对原始数据向量进行归一化处理，并将归一化向量集输入到我们的RaBitQ方法的后续步骤中。在查询阶段，对于一个原始查询向量，算法选择质心离查询最近的前${N}_{\text{probe }}$个聚类。然后，对于每个选定的聚类，算法检索所有量化码，并根据量化码估计它们的距离，这些距离决定了哪些向量将基于精确距离进行重新排序。

As for re-ranking [84], PQ and its variants set a fixed hyper-parameter which decides the number of data vectors to be re-ranked (i.e., they re-rank the vectors with the smallest estimated distances). Specifically, they retrieve their raw data vectors, compute the exact distances and find out the final NN. In particular, the tuning of the hyper-parameter is empirical and often hard as it can vary largely across different datasets (see Section 5.2.3). In contrast, recall that in our method, there is a sharp error bound as discussed in Section 3.2 (note that the error bound is rigorous and always holds regardless of the data distribution). Thus, we decide the data vectors to be re-ranked based on the error bound without tuning hyper-parameters. Specifically, if a data vector has its lower bound of the distance greater than the exact distance of the currently searched nearest neighbor, then we drop it. Otherwise, we compute its exact distance for re-ranking. Due to the theoretical error bound, the re-ranking strategy has the guarantee of correctly sending the true NN from the probed clusters to re-ranking with high probability. The empirical verification can be found in Section 5.2.4. We emphasize that the idea of re-ranking based on a bound is not new. There are many studies from the database community that adopt a similar strategy $\left\lbrack  {{24},{27},{75},{88},{89},{93}}\right\rbrack$ for improving the robustness of similarity search for various data types. We note that beyond the idea of re-ranking based on an error bound, RaBitQ provides rigorous theoretical analysis on the tightness of the bounds and achieves the asymptotic optimality as we have discussed in Section 3.2.2.

关于重新排序[84]，乘积量化（PQ）及其变体设置了一个固定的超参数，该超参数决定了要重新排序的数据向量的数量（即，它们对估计距离最小的向量进行重新排序）。具体来说，它们检索其原始数据向量，计算精确距离并找出最终的最近邻（NN）。特别是，超参数的调整是经验性的，而且通常很困难，因为它在不同的数据集上可能会有很大的变化（见5.2.3节）。相比之下，回想一下，在我们的方法中，如3.2节所述，存在一个严格的误差界（注意，该误差界是严格的，并且无论数据分布如何都始终成立）。因此，我们根据误差界来决定要重新排序的数据向量，而无需调整超参数。具体来说，如果一个数据向量的距离下界大于当前搜索到的最近邻的精确距离，那么我们就舍弃它。否则，我们计算其精确距离以进行重新排序。由于有理论误差界，这种重新排序策略有很高的概率保证能将被探查聚类中的真正最近邻正确地送去重新排序。实证验证可在5.2.4节中找到。我们强调，基于界进行重新排序的想法并不新鲜。数据库领域有许多研究采用了类似的策略$\left\lbrack  {{24},{27},{75},{88},{89},{93}}\right\rbrack$来提高各种数据类型的相似性搜索的鲁棒性。我们注意到，除了基于误差界进行重新排序的想法之外，RaBitQ还对界的紧性进行了严格的理论分析，并实现了如3.2.2节所讨论的渐近最优性。

Moreover, it is worth noting that re-ranking is a vital step for pushing forward RaBitQ's rigorous error bounds on the distances to the robustness of ANN search. In particular, when the ANN search requires higher accuracy than what RaBitQ can guarantee (e.g., when the true distances from the query to two different data vectors are extremely close to each other), then the estimated distance produced by RaBitQ would be less effective to rank them correctly. Re-ranking, in this case, is necessary for achieving high recall. Note that it is inherently difficult for any methods of distance estimation when the distances are extremely close to each other.

此外，值得注意的是，重排序是推进RaBitQ在近似最近邻（ANN）搜索鲁棒性距离上严格误差界的关键步骤。特别是当ANN搜索所需的精度高于RaBitQ所能保证的精度时（例如，当查询点到两个不同数据向量的真实距离非常接近时），RaBitQ产生的估计距离在正确排序这些向量时效果会较差。在这种情况下，重排序对于实现高召回率是必要的。需要注意的是，当距离非常接近时，任何距离估计方法本质上都很难准确估计。

## 5 EXPERIMENTS

## 5 实验

### 5.1 Experimental Setup

### 5.1 实验设置

Our experiments involve three folds. First, we compare our method with the conventional quantization methods in terms of the time-accuracy trade-off of distance estimation and time cost of index phase (with results shown in Section 5.2.1 and Section 5.2.2). Second, we compare the methods when applied for in-memory ANN (with results shown in Section 5.2.3). For ANN, we target to retrieve the 100 nearest neighbors for each query,i.e., $K = {100}$ ,by following [30]. Third,we empirically verify our theoretical analysis (with results shown in Section 5.2.4 to 5.2.6). Finally, we note that RaBitQ is a method with rigorous theoretical guarantee. Its components are an integral whole and together explain its asymptotically optimal performance. The ablation of any component would cause the loss of the theoretical guarantee (i.e., the method becomes heuristic and the performance is no more theoretically predictable) and further disables the error-bound-based re-ranking (Section 4). Despite this, we include empirical ablation studies in the technical report [33].

我们的实验包括三个方面。首先，我们在距离估计的时间 - 精度权衡和索引阶段的时间成本方面，将我们的方法与传统量化方法进行比较（结果见5.2.1节和5.2.2节）。其次，我们比较这些方法在内存中ANN搜索中的应用情况（结果见5.2.3节）。对于ANN搜索，按照文献[30]的方法，我们的目标是为每个查询检索100个最近邻，即 $K = {100}$。第三，我们通过实验验证我们的理论分析（结果见5.2.4节至5.2.6节）。最后，我们要指出的是，RaBitQ是一种具有严格理论保证的方法。其各个组件是一个整体，共同解释了其渐近最优性能。任何组件的缺失都会导致理论保证的丧失（即该方法变成启发式方法，其性能在理论上不再可预测），并进一步使基于误差界的重排序（第4节）无法进行。尽管如此，我们在技术报告[33]中包含了实验性的消融研究。

Datasets. We use six public real-world datasets with varying sizes and dimensionalities, whose details can be found in Table 3. These datasets have been widely used to benchmark ANN algorithms [6, 58, 62, 67]. In particular, it has been reported that on the datasets SIFT, DEEP and GIST, PQx4fs and OPQx4fs have good empirical performance [5]. We note that all these public datasets provide both data and query vectors.

数据集。我们使用了六个不同规模和维度的公开真实世界数据集，其详细信息见表3。这些数据集已被广泛用于对ANN算法进行基准测试[6, 58, 62, 67]。特别是，有报告指出，在SIFT、DEEP和GIST数据集上，PQx4fs和OPQx4fs具有良好的实验性能[5]。我们注意到，所有这些公开数据集都提供了数据向量和查询向量。

<!-- Media -->

Table 3. Dataset Statistics

表3. 数据集统计信息

<table><tr><td>Dataset</td><td>Size</td><td>$D$</td><td>Query Size</td><td>Data Type</td></tr><tr><td>Msong</td><td>992,272</td><td>420</td><td>200</td><td>Audio</td></tr><tr><td>SIFT</td><td>1,000,000</td><td>128</td><td>10,000</td><td>Image</td></tr><tr><td>DEEP</td><td>1,000,000</td><td>256</td><td>1,000</td><td>Image</td></tr><tr><td>Word2Vec</td><td>1,000,000</td><td>300</td><td>1,000</td><td>Text</td></tr><tr><td>GIST</td><td>1,000,000</td><td>960</td><td>1,000</td><td>Image</td></tr><tr><td>Image</td><td>2,340,373</td><td>150</td><td>200</td><td>Image</td></tr></table>

<table><tbody><tr><td>数据集</td><td>大小</td><td>$D$</td><td>查询大小</td><td>数据类型</td></tr><tr><td>百万歌曲数据集（Msong）</td><td>992,272</td><td>420</td><td>200</td><td>音频</td></tr><tr><td>尺度不变特征变换（SIFT）</td><td>1,000,000</td><td>128</td><td>10,000</td><td>图像</td></tr><tr><td>深度（DEEP）</td><td>1,000,000</td><td>256</td><td>1,000</td><td>图像</td></tr><tr><td>词向量（Word2Vec）</td><td>1,000,000</td><td>300</td><td>1,000</td><td>文本</td></tr><tr><td>吉斯特特征（GIST）</td><td>1,000,000</td><td>960</td><td>1,000</td><td>图像</td></tr><tr><td>图像</td><td>2,340,373</td><td>150</td><td>200</td><td>图像</td></tr></tbody></table>

<!-- Media -->

Algorithms. First, for estimating the distances between data vectors and query vectors, we consider three baselines, PQ [45], OPQ [34] and LSQ [66, 67]. In particular, (1) PQ and (2) OPQ are the most popular methods among the quantization methods $\left\lbrack  {{34},{45}}\right\rbrack$ . They are widely deployed in industry $\left\lbrack  {{48},{69},{83},{91}}\right\rbrack$ . The popularity of PQ and OPQ indicates that they have been empirically evaluated to the widest extent and are expected to have the best known stability. Thus, we adopt PQ and OPQ as the primary baseline methods representing the quantization methods which have no theoretical error bounds. There is another line of the quantization methods named the additive quantization $\left\lbrack  {8,{66},{67},{94}}\right\rbrack$ . Compared with PQ,these methods target extreme accuracy at the cost of much higher time for optimizing the codebook and mapping the data vectors into quantization codes in the index phase. (3) LSQ [66, 67] is the state-of-the-art method of this line. Thus, we adopt LSQ as the baseline method representing the quantization methods which pursue extreme performance in the query phase. The baseline methods are taken from the 1.7.4 release version of the open-source library Faiss [48], which is well-optimized with the SIMD instructions of AVX2. Second, for ANN, we compare our method with the most competitive baseline method OPQ according to the results in Section 5.2.1. For both our method and OPQ, we combine them with the IVF index as specified in Section 4. We also include the comparison with HNSW [65] as a reference. It is one of the state-of-the-art graph-based methods as is benchmarked in [6, 85] and is also widely adopted in industry $\left\lbrack  {{48},{69},{83},{91}}\right\rbrack$ . The implementation is taken from the hnswlib [65] optimized with the SIMD instructions of AVX2. We note that a recent quantization method ScaNN [37] proposes a new objective function for constructing the quantization codebook of PQ and claims better empirical performance. However,as has been reported ${}^{6}$ ,its superior performance is mainly due to the fast SIMD-based implementation [5]. The advantage vanishes when PQ is implemented with the same technique. Thus, we exclude it from the comparison. Furthermore, we exclude the comparison with the LSH methods because it has been reported that the quantization methods outperform these methods empirically by orders of magnitudes in efficiency when reaching the same recall [58]. The latest advances in LSH have not changed this trend [79]. Thus, comparable performance with the quantization methods indicates significant improvement over the LSH methods.

算法。首先，在估计数据向量与查询向量之间的距离时，我们考虑了三个基线方法，分别是乘积量化（Product Quantization，PQ） [45]、优化乘积量化（Optimized Product Quantization，OPQ） [34] 和最小二乘量化（Least Squares Quantization，LSQ） [66, 67]。具体而言，(1) PQ 和 (2) OPQ 是量化方法 $\left\lbrack  {{34},{45}}\right\rbrack$ 中最常用的方法。它们在工业界得到了广泛应用 $\left\lbrack  {{48},{69},{83},{91}}\right\rbrack$。PQ 和 OPQ 的广泛使用表明，它们在经验上得到了最广泛的评估，并且有望具有已知的最佳稳定性。因此，我们采用 PQ 和 OPQ 作为主要的基线方法，代表那些没有理论误差界的量化方法。还有一类量化方法称为加法量化 $\left\lbrack  {8,{66},{67},{94}}\right\rbrack$。与 PQ 相比，这些方法以在索引阶段优化码本和将数据向量映射为量化码时花费更高的时间为代价，追求极高的精度。(3) LSQ [66, 67] 是这类方法中的最先进方法。因此，我们采用 LSQ 作为基线方法，代表那些在查询阶段追求极致性能的量化方法。这些基线方法取自开源库 Faiss [48] 的 1.7.4 版本，该库使用 AVX2 的单指令多数据（Single Instruction Multiple Data，SIMD）指令进行了良好的优化。其次，对于近似最近邻搜索（Approximate Nearest Neighbor，ANN），根据 5.2.1 节的结果，我们将我们的方法与最具竞争力的基线方法 OPQ 进行了比较。对于我们的方法和 OPQ，我们都按照第 4 节的规定将它们与倒排文件（Inverted File，IVF）索引相结合。我们还将与分层可导航小世界图（Hierarchical Navigable Small World，HNSW） [65] 的比较作为参考。它是基于图的最先进方法之一，在 [6, 85] 中进行了基准测试，并且在工业界也得到了广泛应用 $\left\lbrack  {{48},{69},{83},{91}}\right\rbrack$。其实现取自使用 AVX2 的 SIMD 指令优化的 hnswlib [65]。我们注意到，最近的一种量化方法 ScaNN [37] 提出了一种新的目标函数来构建 PQ 的量化码本，并声称具有更好的经验性能。然而，正如已有报道 ${}^{6}$ 所示，其优越的性能主要归功于基于快速 SIMD 的实现 [5]。当 PQ 使用相同的技术实现时，这种优势就会消失。因此，我们将其排除在比较之外。此外，我们排除了与局部敏感哈希（Locality Sensitive Hashing，LSH）方法的比较，因为已有报道称，在达到相同召回率时，量化方法在效率上比这些方法高出几个数量级 [58]。LSH 的最新进展并未改变这一趋势 [79]。因此，与量化方法相当的性能意味着相对于 LSH 方法有显著的改进。

Performance Metrics. First, for estimating the distances between data vectors and query vectors, we use two metrics to measure the accuracy and one metric to measure the efficiency. In particular, we measure the accuracy with (1) the average relative error and (2) the maximum relative error on the estimated squared distances. The former measures the general quality of the estimated distances while the latter measures the robustness of the estimated distances. We measure the efficiency with the time for distance estimation per vector. We note that due to the effects of cache, the efficiency depends on the order of estimating distances for the vectors. To simulate the order when the methods are used in practice, we build the IVF index for all methods and estimate the distances in the order that the IVF index probes the clusters. We measure the end-to-end time of estimating distances for all the quantization codes in a dataset and divide it by the size of the dataset. We take the pre-processing time in the query phase (e.g., the time for normalizing, transforming and quantizing the query vector for our method) into account, thus making the comparisons fair. We also measure the time costs of the methods in the index phase. Second, for ANN, we adopt recall and average distance ratio for measuring the accuracy of ANN search. Specifically, recall is the ratio between the number of retrieved true nearest neighbors over $K$ . Average distance ratio is the average of the distance ratios of the returned $K$ data vectors wrt the ground truth nearest neighbors. These metrics are widely adopted to measure the accuracy of ANN algorithms [6, 31, 38, 58, 77]. We adopt query per second (QPS), i.e., the number of queries a method can handle in a second, to measure the efficiency. It is widely adopted to measure the efficiency of ANN algorithms [6, 58, 85]. Following [6, 58, 85], the query time is evaluated in a single thread and the search is conducted for each query individually (instead of queries in a batch). All the metrics are measured on every single query and averaged over the whole query set.

性能指标。首先，为了估计数据向量和查询向量之间的距离，我们使用两种指标来衡量准确性，使用一种指标来衡量效率。具体而言，我们用以下两种方式衡量准确性：（1）估计的平方距离的平均相对误差；（2）估计的平方距离的最大相对误差。前者衡量估计距离的总体质量，而后者衡量估计距离的鲁棒性。我们用每个向量的距离估计时间来衡量效率。我们注意到，由于缓存的影响，效率取决于向量距离估计的顺序。为了模拟实际使用这些方法时的顺序，我们为所有方法构建IVF（倒排文件，Inverted File）索引，并按照IVF索引探查聚类的顺序来估计距离。我们测量对数据集中所有量化码进行距离估计的端到端时间，并将其除以数据集的大小。我们考虑了查询阶段的预处理时间（例如，我们的方法中对查询向量进行归一化、变换和量化的时间），从而使比较更加公平。我们还测量了索引阶段这些方法的时间成本。其次，对于近似最近邻搜索（ANN，Approximate Nearest Neighbor），我们采用召回率和平均距离比来衡量ANN搜索的准确性。具体来说，召回率是检索到的真实最近邻数量与$K$的比值。平均距离比是返回的$K$个数据向量相对于真实最近邻的距离比的平均值。这些指标被广泛用于衡量ANN算法的准确性 [6, 31, 38, 58, 77]。我们采用每秒查询次数（QPS），即一种方法每秒可以处理的查询数量，来衡量效率。它被广泛用于衡量ANN算法的效率 [6, 58, 85]。遵循文献 [6, 58, 85] 的做法，查询时间在单线程中进行评估，并且对每个查询单独进行搜索（而不是批量查询）。所有指标都针对每个单独的查询进行测量，并在整个查询集上取平均值。

Parameter Setting. As is suggested by Faiss [25], the number of clusters for IVF is set to be 4,096 as the datasets are at the million-scale. For our method,there are two parameters,i.e., ${\epsilon }_{0}$ and ${B}_{q}$ . The theoretical analysis in Section 3.2.2 and Section 3.3.1 has provided clear suggestions that ${\epsilon }_{0} = \Theta \left( \sqrt{\log \left( {1/\delta }\right) }\right)$ and ${B}_{q} = \Theta \left( {\log \log D}\right)$ ,where $\delta$ is the failure probability. In practice,the parameters are fixed to be ${\epsilon }_{0} = {1.9}$ and ${B}_{q} = 4$ across all the datasets. The empirical parameter study can be found in Section 5.2.4 and Section 5.2.5. As for the length of the quantization code, it equals to $D$ by definition,but it can also be varied by padding the raw vectors with 0 ’s before generating the quantization codes ${}^{7}$ . More padded 0 ’s indicate longer quantization codes and higher accuracy due to Theorem 3.2 (recall that the error is bounded by $O\left( {1/\sqrt{D}}\right)$ ). By default,the length of the quantization code is set to be the smallest multiple of 64 which is no smaller than $D$ (it is equal to or slightly larger than $D$ ) in order to make it possible to store the bit string with a sequence of 64-bit unsigned integers. For the conventional quantization methods (including PQ, OPQ and LSQ),there are two parameters,namely the number of sub-segments of quantization codes $M$ and the number of candidates for re-ranking which should be tuned empirically. Following the default parameter setting [25,37],we set the number of partitions to be $M = D/2$ . We note that it cannot be further increased as $D$ should be divisible by $M$ for PQ and OPQ. The number of candidates for re-ranking is varied among 500, 1,000 and 2,500 . The experimental results in Section 5.2.3 show that none of the parameters work consistently well across different datasets. For HNSW, we follow its original paper [65] by setting the number of maximum out-degree of each vertex in the graph as 32 (corresponding to ${M}_{HNSW} = {16}$ ) and a parameter which controls the construction of the graph named efConstruction as 500 .

参数设置。正如Faiss [25]所建议的，由于数据集规模达到百万级别，倒排文件（IVF）的聚类数量设置为4096。对于我们的方法，有两个参数，即${\epsilon }_{0}$和${B}_{q}$。3.2.2节和3.3.1节的理论分析明确建议${\epsilon }_{0} = \Theta \left( \sqrt{\log \left( {1/\delta }\right) }\right)$和${B}_{q} = \Theta \left( {\log \log D}\right)$，其中$\delta$是失败概率。在实践中，所有数据集的参数固定为${\epsilon }_{0} = {1.9}$和${B}_{q} = 4$。实证参数研究可在5.2.4节和5.2.5节中找到。至于量化码的长度，根据定义它等于$D$，但也可以在生成量化码${}^{7}$之前用0填充原始向量来改变其长度。根据定理3.2，填充更多的0意味着量化码更长，精度更高（回想一下，误差由$O\left( {1/\sqrt{D}}\right)$界定）。默认情况下，量化码的长度设置为不小于$D$的64的最小倍数（它等于或略大于$D$），以便能够用一系列64位无符号整数存储位串。对于传统的量化方法（包括乘积量化（PQ）、正交乘积量化（OPQ）和最小二乘量化（LSQ）），有两个参数，即量化码的子段数量$M$和需要通过实证调整的重排序候选数量。遵循默认参数设置[25,37]，我们将分区数量设置为$M = D/2$。我们注意到，由于对于PQ和OPQ，$D$应该能被$M$整除，所以该参数不能再增加。重排序候选数量在500、1000和2500之间变化。5.2.3节的实验结果表明，没有一个参数在不同数据集上都能始终表现良好。对于分层可导航小世界图（HNSW），我们遵循其原始论文[65]，将图中每个顶点的最大出度设置为32（对应于${M}_{HNSW} = {16}$），并将控制图构建的参数efConstruction设置为500。

---

<!-- Footnote -->

${}^{6}$ https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors

${}^{6}$ https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors

${}^{7}$ We emphasize that the padded dimensions will not be retained after the generation,and thus,will not affect the space and time costs related to the raw vectors.

${}^{7}$ 我们强调，填充的维度在生成后不会保留，因此不会影响与原始向量相关的空间和时间成本。

<!-- Footnote -->

---

The C++ source codes are compiled by g++ 9.4.0 with -0 fast -march=core-avx2 under Ubuntu 20.04LTS. The Python source codes are run on Python 3.8. All experiments are run on a machine with AMD Threadripper PRO 3955WX 3.9GHz processor (with Zen2 microarchitecture which supports the SIMD instructions till AVX2) and 64GB RAM. The code and datasets are available at https://github.com/gaoj0017/RaBitQ.

C++源代码在Ubuntu 20.04 LTS系统下使用g++ 9.4.0编译器，编译选项为 -0 fast -march=core-avx2。Python源代码在Python 3.8环境下运行。所有实验均在配备AMD Threadripper PRO 3955WX 3.9GHz处理器（采用支持直至AVX2的单指令多数据（SIMD）指令的Zen2微架构）和64GB内存的机器上进行。代码和数据集可在https://github.com/gaoj0017/RaBitQ获取。

<!-- Media -->

<!-- figureText: OPQx4fs-batch - 타 OPQx8-single PQx4fs-batch PQ×8-single LSQx4fs-batch RaBitQ-batch RaBitQ-single Average Relative Error (%) GIST Maximum Relative Error (%) GIST 100 75 50 0 100 150 Average Time / Vector (ns) 15 10 0 100 150 Average Time / Vector (ns) Average Relative Error (%) SIFT Maximum Relative Error (%) SIFT 100 60 20 5 10 20 12 10 20 Average Time / Vector (ns) Average Time / Vector (ns) Average Relative Error (%) Word2Vec Maximum Relative Error (%) Word2Vec out of range $\left( { > {200}\% }\right)$ 100 10 20 30 40 50 10 20 30 40 50 Average Time / Vector (ns) Average Time / Vector (ns) Average Relative Error (%) Image Maximum Relative Error (%) Image 150 100 10 15 12.5 10.0 7.5 5.0 2.5 10 15 Average Time / Vector (ns) Average Time / Vector (ns) Average Relative Error (%) MSong Maximum Relative Error (%) MSong 200 out of range $\left( { > {200}\% }\right)$ 100 50 0 20 40 100 out of range (>100%) 80 ${\Delta x} =  = 0,{\Delta x} =  - 2$ 20 40 0 Average Time / Vector (ns) Average Time / Vector (ns) Average Relative Error (%) DEEP Maximum Relative Error (%) DEEP 80 60 20 40 20 20 40 Average Time / Vector (ns) Average Time / Vector (ns) -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_16.jpg?x=148&y=848&w=1266&h=721&r=0"/>

Fig. 3. Time-Accuracy Trade-Off for Distance Approximation. For baseline methods, (1) "x4fs-batch" means that the SIMD-based fast implementation is adopted (where 4 bits encode a quantized code and approximate distances for a batch of 32 data vectors are computed each time), and (2) "x8-single" means that 8 bits encode a quantized code and the approximate distance of one data vector is computed each time. In addition, the results of LSQx8-single are omitted since it,with the implementation from Faiss, has the time cost significantly larger than others.

图3. 距离近似的时间 - 精度权衡。对于基线方法，（1）“x4fs - batch”表示采用基于单指令多数据（SIMD）的快速实现（其中4位编码一个量化码，每次计算一批32个数据向量的近似距离），（2）“x8 - single”表示8位编码一个量化码，每次计算一个数据向量的近似距离。此外，由于最小二乘量化8位单向量（LSQx8 - single）使用Faiss的实现时，其时间成本明显高于其他方法，因此省略其结果。

<!-- Media -->

### 5.2 Experimental Results

### 5.2 实验结果

5.2.1 Time-Accuracy Trade-Off per Vector for Distance Estimation. We estimate the distance between a data vector (from the set of data vectors) and a query vector (from the set of query vectors) with different quantization methods including PQ, OPQ, LSQ and our RaBitQ method. We plot the "average relative error"-"time per vector" curve (left panels, bottom-left is better) and the "maximum relative error"-"time per vector" curve (right panels, bottom-left is better) by varying the length of the quantization codes in Figure 3. In particular, for our method, to plot the curve, we vary the length by padding different number of0’s in the vectors when generating the quantization codes. For PQ,OPQ and LSQ,we vary the length by setting different $M$ (note that $D$ must be divisible by $M$ for $\mathrm{{PQ}}$ and $\mathrm{{OPQ}}$ ).

5.2.1 距离估计中每个向量的时间 - 精度权衡。我们使用包括乘积量化（PQ，Product Quantization）、正交乘积量化（OPQ，Orthogonal Product Quantization）、局部标量量化（LSQ，Local Scalar Quantization）和我们的RaBitQ方法在内的不同量化方法，来估计数据向量（来自数据向量集合）与查询向量（来自查询向量集合）之间的距离。在图3中，我们通过改变量化码的长度，绘制了“平均相对误差”-“每个向量的时间”曲线（左图，左下方为优）和“最大相对误差”-“每个向量的时间”曲线（右图，左下方为优）。特别地，对于我们的方法，为了绘制曲线，我们在生成量化码时通过在向量中填充不同数量的0来改变长度。对于PQ、OPQ和LSQ，我们通过设置不同的$M$来改变长度（注意，对于$\mathrm{{PQ}}$和$\mathrm{{OPQ}}$，$D$必须能被$M$整除）。

Based on the results in Figure 3, we have the following observations. (1) LSQ has much less stable performance than PQ and OPQ. Except for the dataset SIFT and DEEP, LSQx4fs has its accuracy worse than PQx4fs and OPQx4fs. (2) Comparing the solid curves, we find that under the default setting of the number of bits (which corresponds to the last point in the red and orange solid curves and the first point in the green solid curve), our method shows consistently better accuracy than PQ and OPQ while having comparable efficiency on all the tested datasets. We emphasize that in the default setting, the length of the quantization code of our method is only around a half of those of PQ and OPQ (i.e., $D$ v.s. 2D). (3) Comparing the dashed curves,we find that our method has significantly better efficiency than PQ and OPQ when reaching the same accuracy. (4) On the dataset Msong, PQx8 and OPQx8 have normal accuracy while PQx4fs and OPQx4fs have disastrous accuracy. It indicates that the reasonable accuracy of the conventional quantization methods with $k = 8$ does not indicate its normal performance with $k = 4$ . Thus,it is not always feasible to speed up a conventional quantization method with the fast SIMD-based implementation [5]. On the other hand,the efficiency of the conventional quantization methods with $k = 8$ is hardly comparable with those with $k = 4$ on the other datasets. It indicates that the recent success of PQ in the in-memory ANN is largely attributed to the fast SIMD-based implementation. Thus, it is not a choice to replace the fast SIMD-based implementation with the original one in pursuit of the stability. (5) Except for the dataset SIFT and DEEP, PQx4fs and OPQx4fs have their maximum relative error of around 100%. It indicates that PQ and OPQ do not robustly produce high-accuracy estimated distances even on the datasets they perform well in general. As a comparison, our method has its maximum relative error at most ${40}\%$ on all the tested datasets.

根据图3中的结果，我们有以下观察。（1）LSQ的性能稳定性远不如PQ和OPQ。除了SIFT和DEEP数据集外，LSQx4fs的精度比PQx4fs和OPQx4fs差。（2）比较实线曲线，我们发现，在默认的比特数设置下（对应于红色和橙色实线曲线的最后一个点以及绿色实线曲线的第一个点），我们的方法在所有测试数据集上始终比PQ和OPQ具有更好的精度，同时效率相当。我们强调，在默认设置下，我们方法的量化码长度仅约为PQ和OPQ的一半（即$D$对比2D）。（3）比较虚线曲线，我们发现，在达到相同精度时，我们的方法比PQ和OPQ具有显著更高的效率。（4）在Msong数据集上，PQx8和OPQx8的精度正常，而PQx4fs和OPQx4fs的精度极差。这表明，使用$k = 8$的传统量化方法的合理精度并不意味着使用$k = 4$时其性能也正常。因此，使用基于单指令多数据（SIMD，Single Instruction Multiple Data）的快速实现来加速传统量化方法并不总是可行的[5]。另一方面，在其他数据集上，使用$k = 8$的传统量化方法的效率很难与使用$k = 4$的方法相媲美。这表明，PQ最近在内存中近似最近邻搜索（ANN，Approximate Nearest Neighbor）中的成功很大程度上归功于基于SIMD的快速实现。因此，为了追求稳定性而用原始实现替换基于SIMD的快速实现并非明智之选。（5）除了SIFT和DEEP数据集外，PQx4fs和OPQx4fs的最大相对误差约为100%。这表明，即使在通常表现良好的数据集上，PQ和OPQ也不能稳定地产生高精度的估计距离。相比之下，我们的方法在所有测试数据集上的最大相对误差至多为${40}\%$。

<!-- Media -->

Table 4. The Indexing Time for the GIST Dataset

表4. GIST数据集的索引时间

<table><tr><td/><td>RaBitQ</td><td>PQ</td><td>OPQ</td><td>LSQ</td></tr><tr><td>Time</td><td>117s</td><td>105s</td><td>291s</td><td>time-out (>24 hours)</td></tr></table>

<table><tbody><tr><td></td><td>拉比特Q（RaBitQ）</td><td>PQ（可根据具体专业领域确定准确译法，此处保留英文）</td><td>OPQ（可根据具体专业领域确定准确译法，此处保留英文）</td><td>LSQ（可根据具体专业领域确定准确译法，此处保留英文）</td></tr><tr><td>时间</td><td>117s</td><td>105s</td><td>291s</td><td>超时（超过24小时）</td></tr></tbody></table>

<!-- Media -->

5.2.2 Time in the Indexing Phase. In Table 4, we report the indexing time of the quantization methods ( $k = 4$ for PQ,OPQ and LSQ) under the default parameter setting on the GIST dataset with 32 threads on CPU. The results show that the indexing time is not a bottleneck for our method, PQ and OPQ since all of them can finish the indexing phase within a few mins. However, for LSQ, it takes more than 24 hours. This is because in LSQ, the step of mapping a data vector to its quantization code is NP-Hard [66, 67]. Although several techniques have been proposed for approximately solving the NP-Hard problem [66, 67], the time cost is still much larger than that of PQ, which largely limits its usage in practice.

5.2.2 索引阶段的时间。在表4中，我们报告了量化方法（乘积量化（PQ）、正交乘积量化（OPQ）和最小平方量化（LSQ）的 $k = 4$ ）在GIST数据集上默认参数设置下，使用CPU的32个线程进行索引的时间。结果表明，对于我们的方法、PQ和OPQ而言，索引时间并非瓶颈，因为它们都能在几分钟内完成索引阶段。然而，对于LSQ，它需要超过24小时。这是因为在LSQ中，将数据向量映射到其量化码的步骤是NP难问题 [66, 67]。尽管已经提出了几种近似解决NP难问题的技术 [66, 67]，但时间成本仍然比PQ大得多，这在很大程度上限制了它在实际中的应用。

5.2.3 Time-Accuracy Trade-Off for ANN Search. We then measure the performance of the algorithms when they are used in combination with the IVF index for ANN search. Considering the results in Section 5.2.1, we only include OPQx4fs-batch and RaBitQ-batch for the comparison as other methods or implementations are in general dominated when the quantization codes are allowed to be packed in batch. As a reference, we also include HNSW for comparison. We then plot the "QPS"-"recall" curve (left panel, upper-right is better) and the "QPS"-"average distance ratio" curve (right panel, upper-left is better) by varying the number of buckets to probe in the IVF index for the quantization methods in Figure 4. The curves for HNSW are plotted by varying a parameter named efSearch which controls the QPS-recall tradeoff of HNSW. For OPQ, we show three curves which correspond to three different numbers of candidates for re-ranking. Based on Figure 4, we have the following observations. (1) On all the tested datasets, our method has consistently better performance than OPQ regardless of the re-ranking parameter. We emphasize that it has been reported that on the datasets SIFT, DEEP and GIST, OPQx4fs has good empirical performance [5]. Our method also consistently outperforms HNSW on all the tested datasets. (2) On the dataset MSong, the performance of OPQ is disastrous even with re-ranking applied. In particular, as the IVF index probes more buckets, the recall abnormally decreases because OPQ introduces too much error on the estimated distances. The poor accuracy shown in Figure 3 can explain the disastrous failure. (3) No single re-ranking parameter for OPQ works well across all the datasets. On SIFT, DEEP and GIST, 1,000 of candidates for re-ranking suffice to produce a nearly perfect recall while on Image and Word2Vec, a larger number of candidates for re-ranking is needed. We note that the tuning of the re-ranking parameter is often exhaustive as the parameters are intertwined with many factors such as the datasets and the other parameters. Prior to the testing, there is no reliable way to predict the optimal setting of parameters in practice. In contrast, recall that as is discussed in Section 3.2.2 and Section 3.3.1, in our method, the theoretical analysis provides explicit suggestions on the parameters. Thus, our method requires no tuning.

5.2.3 近似最近邻（ANN）搜索的时间 - 准确率权衡。接下来，我们测量这些算法与倒排文件（IVF）索引结合用于ANN搜索时的性能。考虑到5.2.1节的结果，由于在允许批量打包量化码时，其他方法或实现通常表现较差，因此我们仅将正交乘积量化x4快速扫描批量版（OPQx4fs - batch）和快速自适应比特量化批量版（RaBitQ - batch）纳入比较。作为参考，我们还纳入了分层可导航小世界图（HNSW）进行比较。然后，我们通过改变IVF索引中用于量化方法的探测桶数量，绘制了“每秒查询数（QPS）”-“召回率”曲线（左图，右上角为优）和“QPS”-“平均距离比”曲线（右图，左上角为优），如图4所示。HNSW的曲线是通过改变一个名为efSearch的参数绘制的，该参数控制着HNSW的QPS - 召回率权衡。对于OPQ，我们展示了三条曲线，分别对应三种不同的重排序候选数量。基于图4，我们有以下观察结果。（1）在所有测试数据集上，无论重排序参数如何，我们的方法始终比OPQ表现更好。我们强调，已有报告指出，在SIFT、DEEP和GIST数据集上，OPQx4fs具有良好的实证性能 [5]。我们的方法在所有测试数据集上也始终优于HNSW。（2）在MSong数据集上，即使应用了重排序，OPQ的性能也非常糟糕。特别是，随着IVF索引探测更多的桶，召回率异常下降，因为OPQ在估计距离时引入了过多的误差。图3中显示的低准确率可以解释这种糟糕的表现。（3）OPQ没有一个单一的重排序参数能在所有数据集上都表现良好。在SIFT、DEEP和GIST数据集上，1000个重排序候选足以产生近乎完美的召回率，而在Image和Word2Vec数据集上，则需要更多的重排序候选。我们注意到，重排序参数的调优通常是一项详尽的工作，因为这些参数与许多因素相互交织，如数据集和其他参数。在测试之前，实际上没有可靠的方法来预测参数的最优设置。相比之下，回顾3.2.2节和3.3.1节的讨论，在我们的方法中，理论分析为参数提供了明确的建议。因此，我们的方法无需调优。

<!-- Media -->

<!-- figureText: IVF-OPQx4fs (rerank=500) IVF-OPQx4fs (rerank=1000) IVF-OPQx4fs (rerank=2500) IVF-RaBitQ HNSW GIST ${10}^{3} \times$ $6 \times$ $2 \times$ 1.000 $\begin{array}{llll} {1.001} & {1.002} & {1.003} & {1.004} \end{array}$ Recall(%) Average Distance Ratio SIFT SIFT $6 \times$ $2 \times$ $2 \times$ ${10}^{3}$ ${10}^{3}$ 80 85 95 100 1.000 1.010 Recall(%) Average Distance Ratio Word2Vec Word2Vec $6 \times$ $4 \times$ $2 \times$ 80 85 90 95 100 1.000 1.005 Recall(%) Average Distance Ratio Image $4 \times$ $2 \times$ $2 \times$ QPS $4 \times$ $2 \times$ $2 \times$ Recall(%) Average Distance Ratio MSong MSong $4 \times$ $4 \times$ 2 x QPS ${10}^{3}$ QPS 8× $4 \times$ 0 20 40 60 100 1.2 1.3 1.4 1.5 Recall(%) Average Distance Ratio DEEP DEEP $6 \times$ $2 \times$ $2 \times$ QPS $8 \times$ 8 * 6 × $6 \times$ $4 \times$ 80 90 95 100 1.000 1.002 1,004 1.006 1.008 Recall(%) Average Distance Ratio -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_18.jpg?x=145&y=270&w=1274&h=719&r=0"/>

Fig. 4. Time-Accuracy Trade-Off for ANN Search. The parameter rerank represents the number of candidates for re-ranking.

图4. 近似最近邻（ANN）搜索的时间 - 准确率权衡。参数rerank表示重排序的候选数量。

<!-- figureText: SIFT (D=128) GIST (D=960) 100 80 Recall (%) 40 20 0 1 3 ${\varepsilon }_{0}$ 100 80 Recall (%) 40 20 0 2 4 ${\varepsilon }_{0}$ -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_18.jpg?x=343&y=1669&w=871&h=310&r=0"/>

Fig. 5. Verification Study on ${\epsilon }_{0}$ .

图5. 对 ${\epsilon }_{0}$ 的验证研究。

<!-- Media -->

5.2.4 Results for Verifying the Statement about ${\epsilon }_{0}.{\epsilon }_{0}$ is a parameter which controls the confidence interval of the error bound (see Section 3.2.2). When the RaBitQ method is applied in ANN search, it further controls the probability that we correctly send the NN to re-ranking (see Section 4). In particular,to make sure the failure probability be no greater than $\delta$ ,the theoretical analysis in Section 3.2.2 suggests to set ${\epsilon }_{0} = \Theta \left( \sqrt{\log \left( {1/\delta }\right) }\right)$ . We emphasize that the statement is independent of any other factors such as the datasets or the setting of other parameters. This is the reason that the parameter needs no tuning. In Figure 5, we provide the empirical verification on the statement. In particular,we plot the "recall"-" ${\epsilon }_{0}$ " curve by varying ${\epsilon }_{0}$ from 0.0 to 4.0 . The recall is measured by estimating the distances for all the data vectors and decide the vectors to be re-ranked based on the strategy in Section 4 (note that if a true nearest neighbor is not re-ranked, it will be missed). Thus, the factors (other than the error of quantization methods) which may affect the recall are eliminated. Figure 5 shows that on two different datasets, both curves show highly similar trends that it achieves nearly perfect recall at around ${\epsilon }_{0} = {1.9}$ .

5.2.4 关于 ${\epsilon }_{0}.{\epsilon }_{0}$ 陈述的验证结果 ${\epsilon }_{0}.{\epsilon }_{0}$ 是一个控制误差界置信区间的参数（见3.2.2节）。当RaBitQ方法应用于人工神经网络（ANN）搜索时，它还控制着我们正确发送最近邻（NN）进行重排序的概率（见第4节）。特别地，为确保失败概率不大于 $\delta$ ，3.2.2节的理论分析建议将 ${\epsilon }_{0} = \Theta \left( \sqrt{\log \left( {1/\delta }\right) }\right)$ 。我们强调，该陈述与任何其他因素（如数据集或其他参数的设置）无关。这就是该参数无需调整的原因。在图5中，我们对该陈述进行了实证验证。特别地，我们通过将 ${\epsilon }_{0}$ 从0.0变化到4.0来绘制“召回率”-“ ${\epsilon }_{0}$ ”曲线。召回率是通过估计所有数据向量的距离并根据第4节中的策略确定要进行重排序的向量来衡量的（请注意，如果一个真正的最近邻没有被重排序，它将被遗漏）。因此，消除了可能影响召回率的因素（除了量化方法的误差）。图5显示，在两个不同的数据集上，两条曲线都呈现出高度相似的趋势，即在 ${\epsilon }_{0} = {1.9}$ 附近实现了近乎完美的召回率。

<!-- Media -->

<!-- figureText: Average Relative Error (%) SIFT $\left( {\mathrm{D} = {128}}\right)$ Average Relative Error (%) GIST (D=960) 15 10 2 8 ${B}_{q}$ 15 10 0 2 4 6 8 ${B}_{q}$ -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_19.jpg?x=340&y=726&w=874&h=312&r=0"/>

Fig. 6. Verification Study on ${B}_{q}$ .

图6. 关于 ${B}_{q}$ 的验证研究。

<!-- Media -->

5.2.5 Results for Verifying the Statement about ${B}_{q}.{B}_{q}$ is a parameter which controls the error introduced in the computation of $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ . Due to our analysis in Section 3.3.1, ${B}_{q} = \Theta \left( {\log \log D}\right)$ suffices to make sure that the error introduced in the computation of $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ is much smaller than the error of the estimator. We note that $\Theta \left( {\log \log D}\right)$ varies extremely slowly with respect to $D$ , and thus, it can be viewed as a constant when the dimensionality does not vary largely. In Figure 6, we provide the empirical verification on the statement. In particular, we plot the "average relative error"-" ${B}_{q}$ " curve by varying ${B}_{q}$ from 1 to 8 . Figure 6 shows that on two different datasets,both curves show highly similar trends that the error converges quickly at around ${B}_{q} = 4$ . On the other hand,we would also like to highlight that further reducing ${B}_{q}$ would produce unignorable error in the computation of $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ . In particular,when ${B}_{q} = 1$ ,i.e.,both query and data vectors are quantized into binary strings,the error is much larger than the error when ${B}_{q} = 4$ . This result may help to explain why the binary hashing methods cannot achieve good empirical performance. 5.2.6 Results for Verifying the Unbiasedness. In Figure 7, we verify the unbiasedness of our method and show the biasedness of OPQ. We collect ${10}^{7}$ pairs of the estimated squared distances and the true squared distances between the query and data vectors (i.e., the first 10 query vectors in the query set and the ${10}^{6}$ data vectors in the full dataset of GIST) to verify the unbiasedness. The values of the distances are normalized by the maximum true squared distances. We fit the ${10}^{7}$ pairs with linear regression and plot the result with the black dashed line. We note that if a method is unbiased, the result of the linear regression should have the slope of 1 and the y-axis intercept of 0 (the green dashed line as a reference). Figure 7 clearly shows that our method is unbiased, which verifies the theoretical analysis in Section 3.2.2. On the other hand, the estimated distances produced by OPQ is clearly biased.

5.2.5 关于 ${B}_{q}.{B}_{q}$ 陈述的验证结果 ${B}_{q}.{B}_{q}$ 是一个控制在计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 时引入的误差的参数。根据我们在3.3.1节的分析， ${B}_{q} = \Theta \left( {\log \log D}\right)$ 足以确保在计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 时引入的误差远小于估计器的误差。我们注意到， $\Theta \left( {\log \log D}\right)$ 相对于 $D$ 的变化极其缓慢，因此，当维度变化不大时，可以将其视为一个常数。在图6中，我们对该陈述进行了实证验证。特别地，我们通过将 ${B}_{q}$ 从1变化到8来绘制“平均相对误差”-“ ${B}_{q}$ ”曲线。图6显示，在两个不同的数据集上，两条曲线都呈现出高度相似的趋势，即误差在 ${B}_{q} = 4$ 附近迅速收敛。另一方面，我们还想强调的是，进一步减小 ${B}_{q}$ 会在计算 $\langle \overline{\mathbf{o}},\mathbf{q}\rangle$ 时产生不可忽视的误差。特别地，当 ${B}_{q} = 1$ 时，即查询向量和数据向量都被量化为二进制字符串时，误差比 ${B}_{q} = 4$ 时的误差大得多。这一结果可能有助于解释为什么二进制哈希方法无法取得良好的实证性能。5.2.6 无偏性验证结果。在图7中，我们验证了我们方法的无偏性，并展示了OPQ（正交投影量化）的有偏性。我们收集了 ${10}^{7}$ 对查询向量和数据向量之间的估计平方距离和真实平方距离（即查询集中的前10个查询向量和GIST完整数据集中的 ${10}^{6}$ 个数据向量）来验证无偏性。距离值通过最大真实平方距离进行归一化。我们使用线性回归拟合这 ${10}^{7}$ 对数据，并使用黑色虚线绘制结果。我们注意到，如果一种方法是无偏的，线性回归的结果应该具有斜率为1和y轴截距为0（以绿色虚线作为参考）。图7清楚地表明我们的方法是无偏的，这验证了3.2.2节中的理论分析。另一方面，OPQ产生的估计距离显然是有偏的。

<!-- Media -->

<!-- figureText: GIST (D=960), RaBitQ GIST (D=960), OPQx4fs 1.0 Slope $= {1.00}$ ,Intercept $= {0.00}$ (Reference) Estimated Squared Distance 0.8 0.6 0.4 0.2 0.0 0.2 0.4 0.6 0.8 1.0 True Squared Distance 1.0 Slope $= {1.00}$ ,Intercept $= {0.00}$ (Reference) Estimated Squared Distance 0.8 0.6 0.4 0.2 0.0 0.2 0.4 0.6 0.8 1.0 True Squared Distance -->

<img src="https://cdn.noedgeai.com/0195c74b-1ef1-7261-a362-fef190163528_19.jpg?x=341&y=1627&w=874&h=425&r=0"/>

Fig. 7. Verification Study for Unbiasedness.

图7. 无偏性验证研究。

<!-- Media -->

## 6 RELATED WORK

## 6 相关工作

Approximate Nearest Neighbor Search. Existing studies on ANN search are usually categorized into four types: (1) graph-based methods [29, 30, 58, 64, 65, 73], (2) quantization-based methods $\left\lbrack  {8,{34},{36},{37},{45},{66},{67},{72},{94}}\right\rbrack$ ,(3) tree-based methods $\left\lbrack  {{10},{15},{17},{70}}\right\rbrack$ and (4) hashing-based methods $\left\lbrack  {{18},{31},{35},{38},{39},{54},{56},{63},{77} - {79},{96}}\right\rbrack$ . We refer readers to recent tutorials [23,74] and benchmarks/surveys [6, 7, 19, 58, 85, 87] for a comprehensive review. We note that recently, many studies design algorithms or systems by jointly considering different types of methods so that a method can enjoy the merits of both sides $\left\lbrack  {1,{12},{13},{21},{43},{62},{95}}\right\rbrack$ . Our work proposes a quantization method which provides a sharp error bound and good empirical performance at the same time. Just like the conventional quantization methods, it can work as a component in an integrated algorithm or system. Our method has two additional advantages: (1) it involves no parameter tuning and (2) it supports efficient distance estimation for a single quantization code. These advantages may further smoothen its combination with other types of methods. Recently, there are a thread of methods which apply machine learning (ML) on ANN [9, 20, 26, 55, 57, 86, 90].

近似最近邻搜索。现有的近似最近邻（ANN）搜索研究通常分为四类：（1）基于图的方法 [29, 30, 58, 64, 65, 73]；（2）基于量化的方法 $\left\lbrack  {8,{34},{36},{37},{45},{66},{67},{72},{94}}\right\rbrack$；（3）基于树的方法 $\left\lbrack  {{10},{15},{17},{70}}\right\rbrack$；（4）基于哈希的方法 $\left\lbrack  {{18},{31},{35},{38},{39},{54},{56},{63},{77} - {79},{96}}\right\rbrack$。我们建议读者参考近期的教程 [23,74] 和基准测试/综述 [6, 7, 19, 58, 85, 87] 以进行全面回顾。我们注意到，最近许多研究通过联合考虑不同类型的方法来设计算法或系统，以便一种方法能够兼具双方的优点 $\left\lbrack  {1,{12},{13},{21},{43},{62},{95}}\right\rbrack$。我们的工作提出了一种量化方法，该方法同时提供了严格的误差界和良好的实证性能。与传统的量化方法一样，它可以作为集成算法或系统中的一个组件。我们的方法有两个额外的优点：（1）它无需参数调整；（2）它支持对单个量化码进行高效的距离估计。这些优点可能会进一步促进它与其他类型方法的结合。最近，有一系列方法将机器学习（ML）应用于近似最近邻搜索 [9, 20, 26, 55, 57, 86, 90]。

Quantization. There is a vast literature about the quantization of high-dimensional vectors from different communities including machine learning, computer vision and data management [8, ${27},{34},{36},{45},{66},{67},{72},{80},{89},{94}\rbrack$ . We refer readers to comprehensive surveys $\left\lbrack  {{24},{68},{82},{84}}\right\rbrack$ and reference books [75, 93]. It is worth of mentioning that in [80], a quantization method called Split-VQ was mentioned. The method covers the major idea of PQ, i.e., it splits the vectors into subsegments, constructs sub-codebooks for each sub-segment and forms the codebook with Cartesian product. Besides PQ and its variants, there are other types of quantization methods, e.g., scalar quantization $\left\lbrack  {1,{27},{89}}\right\rbrack$ . These methods quantize the scalar values of each dimension separately, which often adopt more moderate compression rates than PQ in exchange for better accuracy. In particular, VA+ File [27], a scalar quantization method, has shown leading performance on the similarity search of data series according to a recent benchmark [24]. Besides the studies on the quantization algorithms, we note that hardware-aware optimization (with SIMD, GPU, FPGA, etc) also makes significant contributions to the performance of these methods [4, 5, 47, 48, 61]. To inherit the merits of the well-developed hardware-aware optimization, in this work, we reduce our computation to the computation of PQ (Section 3.3). However, RaBitQ, in its nature, can be implemented with much simpler bitwise operations (which is not possible for PQ and its variants). It remains to be an interesting question whether dedicated hardware-aware optimization can further improve the performance of RaBitQ.

量化。来自机器学习、计算机视觉和数据管理等不同领域的关于高维向量量化的文献非常丰富 [8, ${27},{34},{36},{45},{66},{67},{72},{80},{89},{94}\rbrack$。我们建议读者参考全面的综述 $\left\lbrack  {{24},{68},{82},{84}}\right\rbrack$ 和参考书 [75, 93]。值得一提的是，在文献 [80] 中提到了一种名为 Split - VQ 的量化方法。该方法涵盖了乘积量化（PQ）的主要思想，即它将向量分割成子段，为每个子段构建子码本，并通过笛卡尔积形成码本。除了乘积量化及其变体之外，还有其他类型的量化方法，例如标量量化 $\left\lbrack  {1,{27},{89}}\right\rbrack$。这些方法分别对每个维度的标量值进行量化，与乘积量化相比，它们通常采用更适中的压缩率以换取更高的精度。特别是，标量量化方法 VA + 文件 [27]，根据最近的一项基准测试 [24]，在数据序列的相似性搜索方面表现出色。除了对量化算法的研究之外，我们注意到硬件感知优化（使用单指令多数据（SIMD）、图形处理器（GPU）、现场可编程门阵列（FPGA）等）也对这些方法的性能做出了重要贡献 [4, 5, 47, 48, 61]。为了继承成熟的硬件感知优化的优点，在这项工作中，我们将计算简化为乘积量化的计算（第 3.3 节）。然而，RaBitQ 本质上可以通过更简单的按位运算来实现（这对于乘积量化及其变体是不可能的）。专用的硬件感知优化是否能进一步提高 RaBitQ 的性能仍是一个有趣的问题。

Theoretical Studies on High-Dimensional Vectors. The theoretical studies on high-dimensional vectors are primarily about the seminal Johnson-Lindenstrauss (JL) Lemma [49]. It presents that reducing the dimensionality of a vector to $O\left( {{\epsilon }^{-2}\log \left( {1/\delta }\right) }\right)$ suffices to guarantee the error bound of $\epsilon$ . Recent advances improve the JL Lemma in different aspects. For example, [53] proves the optimality of the JL Lemma. [2, 50] propose fast algorithms to do the dimension reduction. We refer readers to a comprehensive survey [28]. Our method fits into a recent line of studies [3, 40, 41, 71], which target to improve the JL Lemma by compressing high-dimensional vectors into short codes. As a comparison,to guarantee an error bound of $\epsilon$ ,the JL Lemma requires a vector of $O\left( {{\epsilon }^{-2}\log \left( {1/\delta }\right) }\right)$ dimensions while these studies prove that a short code with $O\left( {{\epsilon }^{-2}\log \left( {1/\delta }\right) }\right)$ bits would be sufficient. In practical terms, we note that although the existing studies achieve the improvement in theory in terms of the space complexity (i.e., the minimum number of bits needed for guaranteeing a certain error bound), they care less about the improvement in efficiency. In particular, these methods do not suit the in-memory ANN search because, for estimating the distance during the query phase, they need decompress the short codes and compute the distances with the decompressed vectors, which degrades to the brute force in efficiency. For this reason, these methods have not been adopted in practice. In contrast, our method supports practically efficient computation as is specified in Section 3.3.

高维向量的理论研究。高维向量的理论研究主要围绕着具有开创性的约翰逊 - 林登斯特劳斯（JL）引理 [49]。该引理表明，将向量的维度降至 $O\left( {{\epsilon }^{-2}\log \left( {1/\delta }\right) }\right)$ 足以保证 $\epsilon$ 的误差界。近期的研究在不同方面对 JL 引理进行了改进。例如，[53] 证明了 JL 引理的最优性。[2, 50] 提出了用于降维的快速算法。我们建议读者参考一篇全面的综述 [28]。我们的方法属于近期的一系列研究 [3, 40, 41, 71]，这些研究旨在通过将高维向量压缩为短码来改进 JL 引理。作为对比，为了保证 $\epsilon$ 的误差界，JL 引理要求向量具有 $O\left( {{\epsilon }^{-2}\log \left( {1/\delta }\right) }\right)$ 维，而这些研究证明，一个具有 $O\left( {{\epsilon }^{-2}\log \left( {1/\delta }\right) }\right)$ 位的短码就足够了。实际上，我们注意到，尽管现有研究在空间复杂度（即保证一定误差界所需的最小位数）方面在理论上取得了改进，但它们较少关注效率的提升。特别是，这些方法不适合内存中的近似最近邻（ANN）搜索，因为在查询阶段估计距离时，它们需要对短码进行解压缩，并使用解压缩后的向量计算距离，这在效率上会退化为暴力搜索。因此，这些方法在实践中并未被采用。相比之下，我们的方法支持如第 3.3 节所述的实际高效计算。

Signed Random Projection. We note that there are a line of studies named signed random projection (SRP) which generate a short code for estimating the angular values between vectors via binarizing the vectors after randomization [11, 22, 46, 51]. We note that our method is different from these studies in the following aspects. (1) Problem-wise, SRP targets to unbiasedly estimate the angular value while RaBitQ targets to unbiasedly estimate the inner product (and further, the squared distances). Note that the relationship between the angular value and the inner product is non-linear. The unbiased estimator for one does not trivially derive an unbiased estimator for the other. (2) Theory-wise, RaBitQ has a stronger type of guarantee than SRP. In particular, RaBitQ guarantees that every data vector has its distance within the bounds with high probability. In contrast, SRP only bounds the variance, i.e., the "average" squared error, and it does not provide a bound for every estimated value. Thus, it cannot help with the re-ranking in similarity search. (3) Technique-wise, in SRP the bit strings are viewed as some hashing codes while in RaBitQ, the bit strings are the binary representations of bi-valued vectors. Moreover, SRP maps both the data and query vectors to bit strings, which introduces error from both sides. In contrast, RaBitQ quantizes the data vectors to be bit strings and the query vectors to be vectors of 4-bit unsigned integers. Theorem 3.3 proves that quantizing the query vectors only introduces negligible error. Thus, RaBitQ only introduces the error from the side of the data vector.

带符号随机投影。我们注意到有一系列名为带符号随机投影（SRP）的研究，这些研究通过对随机化后的向量进行二值化来生成短码，以估计向量之间的角度值 [11, 22, 46, 51]。我们注意到我们的方法在以下方面与这些研究不同。（1）从问题角度来看，SRP 的目标是无偏估计角度值，而 RaBitQ 的目标是无偏估计内积（进而估计平方距离）。请注意，角度值和内积之间的关系是非线性的。一个的无偏估计量并不能轻易推导出另一个的无偏估计量。（2）从理论角度来看，RaBitQ 比 SRP 具有更强的保证类型。特别是，RaBitQ 保证每个数据向量的距离以高概率处于一定范围内。相比之下，SRP 仅限制方差，即“平均”平方误差，并且它没有为每个估计值提供界限。因此，它无法帮助在相似性搜索中进行重排序。（3）从技术角度来看，在 SRP 中，位串被视为某种哈希码，而在 RaBitQ 中，位串是双值向量的二进制表示。此外，SRP 将数据向量和查询向量都映射到位串，这会从双方引入误差。相比之下，RaBitQ 将数据向量量化为位串，将查询向量量化为 4 位无符号整数向量。定理 3.3 证明，对查询向量进行量化只会引入可忽略的误差。因此，RaBitQ 仅从数据向量一侧引入误差。

## 7 CONCLUSION

## 7 结论

In conclusion, we propose a novel randomized quantization method RaBitQ which has clear advantages in both empirical accuracy and rigorous theoretical error bound over PQ and its variants. The proposed efficient implementations based on simple bitwise operations or fast SIMD-based operations further make it stand out in terms of the time-accuracy trade-off for the in-memory ANN search. Extensive experiments on real-world datasets verify both (1) the empirical superiority of our method in terms of the time-accuracy trade-off and (2) the alignment of the empirical performance with the theoretical analysis. Some interesting research directions include applying our method in other scenarios of ANN search (e.g., with graph-based indexes or on other storage devices $\left\lbrack  {{14},{42},{44},{59}}\right\rbrack  )$ . Besides,RaBitQ can also be trivially applied to unbiasedly estimate cosine similarity and inner product ${}^{8}$ ,which further implies its potential in maximum inner product search and neural network quantization.

总之，我们提出了一种新颖的随机量化方法 RaBitQ，与乘积量化（PQ）及其变体相比，该方法在经验准确性和严格的理论误差界方面都具有明显优势。基于简单的按位运算或基于单指令多数据（SIMD）的快速运算所提出的高效实现，进一步使其在内存中近似最近邻（ANN）搜索的时间 - 准确性权衡方面脱颖而出。在真实世界数据集上进行的大量实验验证了（1）我们的方法在时间 - 准确性权衡方面的经验优越性，以及（2）经验性能与理论分析的一致性。一些有趣的研究方向包括将我们的方法应用于 ANN 搜索的其他场景（例如，使用基于图的索引或在其他存储设备上 $\left\lbrack  {{14},{42},{44},{59}}\right\rbrack  )$）。此外，RaBitQ 还可以轻松地用于无偏估计余弦相似度和内积 ${}^{8}$，这进一步暗示了其在最大内积搜索和神经网络量化方面的潜力。

---

<!-- Footnote -->

${}^{8}$ The cosine similarity of two vectors exactly equals to the inner product of their unit vectors. The inner product of $\mathbf{o}$ and $\mathbf{q}$ can be expressed as $\langle \mathbf{o},\mathbf{q}\rangle  = \langle \mathbf{o} - \mathbf{c} + \mathbf{c},\mathbf{q} - \mathbf{c} + \mathbf{c}\rangle  = \parallel \mathbf{o} - \mathbf{c}\parallel  \cdot  \parallel \mathbf{q} - \mathbf{c}\parallel  \cdot  \langle \left( {\mathbf{o} - \mathbf{c}}\right) /\parallel \mathbf{o} - \mathbf{c}\parallel ,\left( {\mathbf{q} - \mathbf{c}}\right) /\parallel \mathbf{q} - \mathbf{c}\parallel \rangle  + \langle \mathbf{o},\mathbf{c}\rangle  +$ $\langle \mathbf{q},\mathbf{c}\rangle  - \parallel \mathbf{c}{\parallel }^{2}$ ,where $\mathbf{c}$ is the centroid of the data vectors,and it reduces to the estimation of inner product between unit vectors as we do in Section 3.1.1.

${}^{8}$ 两个向量的余弦相似度恰好等于它们单位向量的内积。$\mathbf{o}$ 和 $\mathbf{q}$ 的内积可以表示为 $\langle \mathbf{o},\mathbf{q}\rangle  = \langle \mathbf{o} - \mathbf{c} + \mathbf{c},\mathbf{q} - \mathbf{c} + \mathbf{c}\rangle  = \parallel \mathbf{o} - \mathbf{c}\parallel  \cdot  \parallel \mathbf{q} - \mathbf{c}\parallel  \cdot  \langle \left( {\mathbf{o} - \mathbf{c}}\right) /\parallel \mathbf{o} - \mathbf{c}\parallel ,\left( {\mathbf{q} - \mathbf{c}}\right) /\parallel \mathbf{q} - \mathbf{c}\parallel \rangle  + \langle \mathbf{o},\mathbf{c}\rangle  +$ $\langle \mathbf{q},\mathbf{c}\rangle  - \parallel \mathbf{c}{\parallel }^{2}$，其中 $\mathbf{c}$ 是数据向量的质心（centroid），并且正如我们在 3.1.1 节中所做的那样，它简化为单位向量之间内积的估计。

<!-- Footnote -->

---

## ACKNOWLEDGEMENTS

## 致谢

We would like to thank the anonymous reviewers for providing constructive feedback and valuable suggestions. This research is supported by the Ministry of Education, Singapore, under its Academic Research Fund (Tier 2 Award MOE-T2EP20221-0013, Tier 2 Award MOE-T2EP20220-0011, and Tier 1 Award (RG77/21)). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of the Ministry of Education, Singapore.

我们要感谢匿名审稿人提供的建设性反馈和宝贵建议。本研究得到了新加坡教育部学术研究基金的支持（二级奖励 MOE - T2EP20221 - 0013、二级奖励 MOE - T2EP20220 - 0011 和一级奖励 (RG77/21)）。本材料中表达的任何观点、研究结果、结论或建议均为作者个人观点，并不反映新加坡教育部的意见。

## REFERENCES

## 参考文献

[1] Cecilia Aguerrebere, Ishwar Singh Bhati, Mark Hildebrand, Mariano Tepper, and Theodore Willke. 2023. Similarity Search in the Blink of an Eye with Compressed Indices. Proc. VLDB Endow. 16, 11 (aug 2023), 3433-3446. https: //doi.org/10.14778/3611479.3611537

[2] Nir Ailon and Bernard Chazelle. 2009. The Fast Johnson-Lindenstrauss Transform and Approximate Nearest Neighbors. SIAM J. Comput. 39, 1 (2009), 302-322. https://doi.org/10.1137/060673096 arXiv:https://doi.org/10.1137/060673096

[3] Noga Alon and Bo'az Klartag. 2017. Optimal Compression of Approximate Inner Products and Dimension Reduction. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS). 639-650. https://doi.org/10.1109/ FOCS.2017.65

[4] Fabien André, Anne-Marie Kermarrec, and Nicolas Le Scouarnec. 2015. Cache Locality is Not Enough: High-Performance Nearest Neighbor Search with Product Quantization Fast Scan. Proc. VLDB Endow. 9, 4 (dec 2015), 288-299. https://doi.org/10.14778/2856318.2856324

[5] Fabien André, Anne-Marie Kermarrec, and Nicolas Le Scouarnec. 2017. Accelerated Nearest Neighbor Search with Quick ADC. In Proceedings of the 2017 ACM on International Conference on Multimedia Retrieval (Bucharest, Romania) (ICMR '17). Association for Computing Machinery, New York, NY, USA, 159-166. https://doi.org/10.1145/3078971.3078992

[6] Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms. Inf. Syst. 87, C (jan 2020), 13 pages. https://doi.org/10.1016/j.is.2019.02.006

[7] Martin Aumüller and Matteo Ceccarello. 2023. Recent Approaches and Trends in Approximate Nearest Neighbor Search, with Remarks on Benchmarking. Data Engineering (2023), 89.

[8] Artem Babenko and Victor Lempitsky. 2014. Additive Quantization for Extreme Vector Compression. In 2014 IEEE Conference on Computer Vision and Pattern Recognition. 931-938. https://doi.org/10.1109/CVPR.2014.124

[9] Dmitry Baranchuk, Dmitry Persiyanov, Anton Sinitsin, and Artem Babenko. 2019. Learning to Route in Similarity Graphs. In Proceedings of the 36th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 97), Kamalika Chaudhuri and Ruslan Salakhutdinov (Eds.). PMLR, 475-484. https://proceedings.mlr.press/v97/baranchuk19a.html

[10] Alina Beygelzimer, Sham Kakade, and John Langford. 2006. Cover trees for nearest neighbor. In Proceedings of the 23rd international conference on Machine learning. 97-104.

[11] Moses S. Charikar. 2002. Similarity Estimation Techniques from Rounding Algorithms. In Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing (Montreal, Quebec, Canada) (STOC '02). Association for Computing Machinery, New York, NY, USA, 380-388. https://doi.org/10.1145/509907.509965

[12] Patrick H. Chen, Wei-Cheng Chang, Jyun-Yu Jiang, Hsiang-Fu Yu, Inderjit S. Dhillon, and Cho-Jui Hsieh. 2023. FINGER: Fast inference for graph-based approximate nearest neighbor search. In The Web Conference 2023. https: //www.amazon.science/publications/finger-fast-inference-for-graph-based-approximate-nearest-neighbor-search

[13] Qi Chen, Haidong Wang, Mingqin Li, Gang Ren, Scarlett Li, Jeffery Zhu, Jason Li, Chuanjie Liu, Lintao Zhang, and Jingdong Wang. 2018. SPTAG: A library for fast approximate nearest neighbor search. https://github.com/Microsoft/ SPTAG

[14] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li, Mao Yang, and Jingdong Wang. 2021. SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search. In 35th Conference on Neural Information Processing Systems (NeurIPS 2021).

[15] Paolo Ciaccia, Marco Patella, and Pavel Zezula. 1997. M-Tree: An Efficient Access Method for Similarity Search in Metric Spaces. In Proceedings of the 23rd International Conference on Very Large Data Bases (VLDB '97). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 426-435.

[16] T. Cover and P. Hart. 1967. Nearest neighbor pattern classification. IEEE Transactions on Information Theory 13, 1 (1967), 21-27. https://doi.org/10.1109/TIT.1967.1053964

[17] Sanjoy Dasgupta and Yoav Freund. 2008. Random projection trees and low dimensional manifolds. In Proceedings of the fortieth annual ACM symposium on Theory of computing. 537-546.

[18] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. 2004. Locality-sensitive hashing scheme based on p-stable distributions. In Proceedings of the twentieth annual symposium on Computational geometry. 253-262.

[19] Magdalen Dobson, Zheqi Shen, Guy E Blelloch, Laxman Dhulipala, Yan Gu, Harsha Vardhan Simhadri, and Yihan Sun. 2023. Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets: A Comparative Analysis. arXiv preprint arXiv:2305.04359 (2023).

[20] Yihe Dong, Piotr Indyk, Ilya Razenshteyn, and Tal Wagner. 2020. Learning Space Partitions for Nearest Neighbor Search. In International Conference on Learning Representations. https://openreview.net/forum?id=rkenmREFDr

[21] Matthijs Douze, Alexandre Sablayrolles, and Hervé Jégou. 2018. Link and Code: Fast Indexing with Graphs and Compact Regression Codes. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3646-3654. https://doi.org/10.1109/CVPR.2018.00384

[22] Punit Pankaj Dubey, Bhisham Dev Verma, Rameshwar Pratap, and Keegan Kang. 2022. Improving sign-random-projection via count sketch. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence (Proceedings of Machine Learning Research, Vol. 180), James Cussens and Kun Zhang (Eds.). PMLR, 599-609. https: //proceedings.mlr.press/v180/dubey22a.html

[23] Karima Echihabi, Kostas Zoumpatianos, and Themis Palpanas. 2021. New Trends in High-D Vector Similarity Search: Al-Driven, Progressive, and Distributed. Proc. VLDB Endow. 14, 12 (jul 2021), 3198-3201. https://doi.org/10.14778/ 3476311.3476407

[24] Karima Echihabi, Kostas Zoumpatianos, Themis Palpanas, and Houda Benbrahim. 2018. The Lernaean Hydra of Data Series Similarity Search: An Experimental Evaluation of the State of the Art. Proc. VLDB Endow. 12, 2 (oct 2018), 112-127. https://doi.org/10.14778/3282495.3282498

[25] Faiss. 2023. Faiss. https://github.com/facebookresearch/faiss.

[26] Chao Feng, Defu Lian, Xiting Wang, Zheng Liu, Xing Xie, and Enhong Chen. 2022. Reinforcement Routing on Proximity Graph for Efficient Recommendation. ACM Trans. Inf. Syst. (jan 2022). https://doi.org/10.1145/3512767 Just Accepted.

[27] Hakan Ferhatosmanoglu, Ertem Tuncel, Divyakant Agrawal, and Amr El Abbadi. 2000. Vector Approximation Based Indexing for Non-Uniform High Dimensional Data Sets. In Proceedings of the Ninth International Conference on Information and Knowledge Management (McLean, Virginia, USA) (CIKM '00). Association for Computing Machinery, New York, NY, USA, 202-209. https://doi.org/10.1145/354756.354820

[28] Casper Benjamin Freksen. 2021. An Introduction to Johnson-Lindenstrauss Transforms. CoRR abs/2103.00564 (2021). arXiv:2103.00564 https://arxiv.org/abs/2103.00564

[29] Cong Fu, Changxu Wang, and Deng Cai. 2021. High dimensional similarity search with satellite system graph: Efficiency, scalability, and unindexed query compatibility. IEEE Transactions on Pattern Analysis and Machine Intelligence (2021).

[30] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast Approximate Nearest Neighbor Search with the Navigating Spreading-out Graph. Proc. VLDB Endow. 12, 5 (jan 2019), 461-474. https://doi.org/10.14778/3303753.3303754

[31] Junhao Gan, Jianlin Feng, Qiong Fang, and Wilfred Ng. 2012. Locality-Sensitive Hashing Scheme Based on Dynamic Collision Counting. In Proceedings of the 2012 ACM SIGMOD International Conference on Management of Data (Scottsdale, Arizona, USA) (SIGMOD '12). Association for Computing Machinery, New York, NY, USA, 541-552. https://doi.org/10.1145/2213836.2213898

[32] Jianyang Gao and Cheng Long. 2023. High-Dimensional Approximate Nearest Neighbor Search: With Reliable and Efficient Distance Comparison Operations. Proc. ACM Manag. Data 1, 2, Article 137 (jun 2023), 27 pages. https: //doi.org/10.1145/3589282

[33] Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with Theoretical Error Bound for Approximate Nearest Neighbor Search (Technical Report). https://github.com/gaoj0017/RaBitQ/technical_report.pdf.

[34] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization for approximate nearest neighbor search. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2946-2953.

[35] Long Gong, Huayi Wang, Mitsunori Ogihara, and Jun Xu. 2020. IDEC: Indexable Distance Estimating Codes for Approximate Nearest Neighbor Search. Proc. VLDB Endow. 13, 9 (may 2020), 1483-1497. https://doi.org/10.14778/ 3397230.3397243

[36] Yunchao Gong, Svetlana Lazebnik, Albert Gordo, and Florent Perronnin. 2013. Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-Scale Image Retrieval. IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 12 (2013), 2916-2929. https://doi.org/10.1109/TPAMI.2012.193

[37] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating Large-Scale Inference with Anisotropic Vector Quantization. In Proceedings of the 37th International Conference on Machine Learning (ICML'20). JMLR.org, Article 364, 10 pages.

[38] Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong Fang, and Wilfred Ng. 2015. Query-aware locality-sensitive hashing for approximate nearest neighbor search. Proceedings of the VLDB Endowment 9, 1 (2015), 1-12.

[39] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing. 604-613.

RaBitQ: Quantizing High-Dim. Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search 167:25

RaBitQ：为近似最近邻搜索对具有理论误差界的高维向量进行量化 167:25

[40] Piotr Indyk, Ilya Razenshteyn, and Tal Wagner. 2017. Practical Data-Dependent Metric Compression with Provable Guarantees. In Proceedings of the 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 2614-2623.

[41] Piotr Indyk and Tal Wagner. 2022. Optimal (Euclidean) Metric Compression. SIAM J. Comput. 51, 3 (2022), 467-491. https://doi.org/10.1137/20M1371324 arXiv:https://doi.org/10.1137/20M1371324

[42] Junhyeok Jang, Hanjin Choi, Hanyeoreum Bae, Seungjun Lee, Miryeong Kwon, and Myoungsoo Jung. 2023. CXL-ANNS: Software-Hardware Collaborative Memory Disaggregation and Computation for Billion-Scale Approximate Nearest Neighbor Search. In 2023 USENIX Annual Technical Conference (USENIX ATC 23). USENIX Association, Boston, MA, 585-600. https://www.usenix.org/conference/atc23/presentation/jang

[43] Yahoo Japan. 2022. NGT-QG. https://github.com/yahoojapan/NGT.

[44] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi. 2019. DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node. In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32. Curran Associates, Inc. https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6- Paper.pdf

[45] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117-128.

[46] Jianqiu Ji, Jianmin Li, Shuicheng Yan, Bo Zhang, and Qi Tian. 2012. Super-Bit Locality-Sensitive Hashing. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 (Lake Tahoe, Nevada) (NIPS'12). Curran Associates Inc., Red Hook, NY, USA, 108-116.

[47] Wenqi Jiang, Shigang Li, Yu Zhu, Johannes De Fine Licht, Zhenhao He, Runbin Shi, Cedric Renggli, Shuai Zhang, Theodoros Rekatsinas, Torsten Hoefler, and Gustavo Alonso. 2023. Co-design Hardware and Algorithm for Vector Search. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (Denver, CO, USA) (SC '23). Association for Computing Machinery, New York, NY, USA, Article 87, 15 pages. https://doi.org/10.1145/3581784.3607045

[48] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535-547.

[49] William B Johnson and Joram Lindenstrauss. 1984. Extensions of Lipschitz mappings into a Hilbert space 26. Contemporary mathematics 26 (1984), 28.

[50] Daniel M. Kane and Jelani Nelson. 2014. Sparser Johnson-Lindenstrauss Transforms. J. ACM 61, 1, Article 4 (jan 2014), 23 pages. https://doi.org/10.1145/2559902

[51] Keegan Kang and Weipin Wong. 2018. Improving Sign Random Projections With Additional Information. In Proceedings of the 35th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 80), Jennifer Dy and Andreas Krause (Eds.). PMLR, 2479-2487. https://proceedings.mlr.press/v80/kang18b.html

[52] V. I. Khokhlov. 2006. The Uniform Distribution on a Sphere in ${\mathbf{R}}^{\mathcal{S}}$ . Properties of Projections. I. Theory of Probability & Its Applications 50, 3 (2006), 386-399. https://doi.org/10.1137/S0040585X97981846 arXiv:https://doi.org/10.1137/S0040585X97981846

[53] Kasper Green Larsen and Jelani Nelson. 2017. Optimality of the Johnson-Lindenstrauss lemma. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 633-638.

[54] Yifan Lei, Qiang Huang, Mohan Kankanhalli, and Anthony K. H. Tung. 2020. Locality-Sensitive Hashing Scheme Based on Longest Circular Co-Substring. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (Portland, OR, USA) (SIGMOD '20). Association for Computing Machinery, New York, NY, USA, 2589-2599. https://doi.org/10.1145/3318464.3389778

[55] Conglong Li, Minjia Zhang, David G. Andersen, and Yuxiong He. 2020. Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (Portland, OR, USA). Association for Computing Machinery, New York, NY, USA, 2539-2554. https://doi.org/10.1145/3318464.3380600

[56] Jinfeng Li, Xiao Yan, Jian Zhang, An Xu, James Cheng, Jie Liu, Kelvin K. W. Ng, and Ti-chung Cheng. 2018. A General and Efficient Querying Method for Learning to Hash. In Proceedings of the 2018 International Conference on Management of Data (Houston, TX, USA) (SIGMOD '18). Association for Computing Machinery, New York, NY, USA, 1333-1347. https://doi.org/10.1145/3183713.3183750

[57] Mingjie Li, Yuan-Gen Wang, Peng Zhang, Hanpin Wang, Lisheng Fan, Enxia Li, and Wei Wang. 2023. Deep Learning for Approximate Nearest Neighbour Search: A Survey and Future Directions. IEEE Transactions on Knowledge and Data Engineering 35, 9 (2023), 8997-9018. https://doi.org/10.1109/TKDE.2022.3220683

[58] Wen Li, Ying Zhang, Yifang Sun, Wei Wang, Mingjie Li, Wenjie Zhang, and Xuemin Lin. 2019. Approximate nearest neighbor search on high dimensional data-experiments, analyses, and improvement. IEEE Transactions on Knowledge and Data Engineering 32, 8 (2019), 1475-1488.

[59] Yingfan Liu, Hong Cheng, and Jiangtao Cui. 2017. PQBF: I/O-Efficient Approximate Nearest Neighbor Search by Product Quantization. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (Singapore, Singapore) (CIKM '17). Association for Computing Machinery, New York, NY, USA, 667-676. https: //doi.org/10.1145/3132847.3132901

[60] Ying Liu, Dengsheng Zhang, Guojun Lu, and Wei-Ying Ma. 2007. A survey of content-based image retrieval with high-level semantics. Pattern Recognition 40, 1 (2007), 262-282. https://doi.org/10.1016/j.patcog.2006.04.045

[61] Zihan Liu, Wentao Ni, Jingwen Leng, Yu Feng, Cong Guo, Quan Chen, Chao Li, Minyi Guo, and Yuhao Zhu. 2023. JUNO: Optimizing High-Dimensional Approximate Nearest Neighbour Search with Sparsity-Aware Algorithm and Ray-Tracing Core Mapping. arXiv:2312.01712 [cs.DC]

[62] Kejing Lu, Mineichi Kudo, Chuan Xiao, and Yoshiharu Ishikawa. 2021. HVS: Hierarchical Graph Structure Based on Voronoi Diagrams for Solving Approximate Nearest Neighbor Search. Proc. VLDB Endow. 15, 2 (oct 2021), 246-258. https://doi.org/10.14778/3489496.3489506

[63] Kejing Lu, Hongya Wang, Wei Wang, and Mineichi Kudo. 2020. VHP: approximate nearest neighbor search via virtual hypersphere partitioning. Proceedings of the VLDB Endowment 13, 9 (2020), 1443-1455.

[64] Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and Vladimir Krylov. 2014. Approximate nearest neighbor algorithm based on navigable small world graphs. Information Systems 45 (2014), 61-68. https://doi.org/10.1016/j.is.2013.10.006

[65] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence 42, 4 (2020), 824-836. https://doi.org/10.1109/TPAMI.2018.2889473

[66] Julieta Martinez, Joris Clement, Holger H. Hoos, and James J. Little. 2016. Revisiting Additive Quantization. In Computer Vision - ECCV 2016, Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (Eds.). Springer International Publishing, Cham, 137-153.

[67] Julieta Martinez, Shobhit Zakhmi, Holger H. Hoos, and James J. Little. 2018. LSQ++: Lower Running Time and Higher Recall in Multi-Codebook Quantization. In Computer Vision - ECCV 2018: 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part XVI (Munich, Germany). Springer-Verlag, Berlin, Heidelberg, 508-523. https://doi.org/10.1007/978-3-030-01270-0_30

[68] Yusuke Matsui, Yusuke Uchida, Hervé Jégou, and Shin'ichi Satoh. 2018. A Survey of Product Quantization. ITE Transactions on Media Technology and Applications 6, 1 (2018), 2-10.

[69] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Ali Mousavi, Ihab F. Ilyas, Umar Farooq Minhas, Jeffrey Pound, and Theodoros Rekatsinas. 2023. High-Throughput Vector Similarity Search in Knowledge Graphs. Proc. ACM Manag. Data 1, 2, Article 197 (jun 2023), 25 pages. https://doi.org/10.1145/3589777

[70] Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms for high dimensional data. IEEE transactions on pattern analysis and machine intelligence 36, 11 (2014), 2227-2240.

[71] Rasmus Pagh and Johan Sivertsen. 2020. The Space Complexity of Inner Product Filters. In 23rd International Conference on Database Theory (ICDT 2020) (Leibniz International Proceedings in Informatics (LIPIcs), Vol. 155), Carsten Lutz and Jean Christoph Jung (Eds.). Schloss Dagstuhl-Leibniz-Zentrum für Informatik, Dagstuhl, Germany, 22:1-22:14. https://doi.org/10.4230/LIPIcs.ICDT.2020.22

[72] John Paparrizos, Ikraduya Edian, Chunwei Liu, Aaron J. Elmore, and Michael J. Franklin. 2022. Fast Adaptive Similarity Search through Variance-Aware Quantization. In 2022 IEEE 38th International Conference on Data Engineering (ICDE). 2969-2983. https://doi.org/10.1109/ICDE53745.2022.00268

[73] Yun Peng, Byron Choi, Tsz Nam Chan, Jianye Yang, and Jianliang Xu. 2023. Efficient Approximate Nearest Neighbor Search in Multi-Dimensional Databases. Proc. ACM Manag. Data 1, 1, Article 54 (may 2023), 27 pages. https: //doi.org/10.1145/3588908

[74] Jianbin Qin, Wei Wang, Chuan Xiao, Ying Zhang, and Yaoshu Wang. 2021. High-Dimensional Similarity Query Processing for Data Science. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Virtual Event, Singapore) (KDD ’21). Association for Computing Machinery, New York, NY, USA, 4062-4063. https://doi.org/10.1145/3447548.3470811

[75] Hanan Samet. 2005. Foundations of Multidimensional and Metric Data Structures (The Morgan Kaufmann Series in Computer Graphics and Geometric Modeling). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.

[76] J. Ben Schafer, Dan Frankowski, Jon Herlocker, and Shilad Sen. 2007. Collaborative Filtering Recommender Systems. Springer Berlin Heidelberg, Berlin, Heidelberg, 291-324. https://doi.org/10.1007/978-3-540-72079-9_9

[77] Yifang Sun, Wei Wang, Jianbin Qin, Ying Zhang, and Xuemin Lin. 2014. SRS: solving c-approximate nearest neighbor queries in high dimensional euclidean space with a tiny index. Proceedings of the VLDB Endowment (2014).

[78] Yufei Tao, Ke Yi, Cheng Sheng, and Panos Kalnis. 2010. Efficient and accurate nearest neighbor and closest pair search in high-dimensional space. ACM Transactions on Database Systems (TODS) 35, 3 (2010), 1-46.

RaBitQ: Quantizing High-Dim. Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search 167:27

RaBitQ：为近似最近邻搜索对具有理论误差界的高维向量进行量化 167:27

[79] Y. Tian, X. Zhao, and X. Zhou. 2022. DB-LSH: Locality-Sensitive Hashing with Query-based Dynamic Bucketing. In 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE Computer Society, Los Alamitos, CA, USA, 2250-2262. https://doi.org/10.1109/ICDE53745.2022.00214

[80] Ertem Tuncel, Hakan Ferhatosmanoglu, and Kenneth Rose. 2002. VQ-Index: An Index Structure for Similarity Searching in Multimedia Databases. In Proceedings of the Tenth ACM International Conference on Multimedia (Juan-les-Pins, France) (MULTIMEDIA '02). Association for Computing Machinery, New York, NY, USA, 543-552. https: //doi.org/10.1145/641007.641117

[81] Roman Vershynin. 2018. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press. https://doi.org/10.1017/9781108231596

[82] Jun Wang, Wei Liu, Sanjiv Kumar, and Shih-Fu Chang. 2016. Learning to Hash for Indexing Big Data - A Survey. Proc. IEEE 104, 1 (2016), 34-57. https://doi.org/10.1109/JPROC.2015.2487976

[83] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan, Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua Mo, Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Milvus: A Purpose-Built Vector Data Management System. In Proceedings of the 2021 International Conference on Management of Data (Virtual Event, China) (SIGMOD '21). Association for Computing Machinery, New York, NY, USA, 2614-2627. https://doi.org/10.1145/3448016.3457550

[84] Jingdong Wang, Ting Zhang, jingkuan song, Nicu Sebe, and Heng Tao Shen. 2018. A Survey on Learning to Hash. IEEE Transactions on Pattern Analysis and Machine Intelligence 40, 4 (2018), 769-790. https://doi.org/10.1109/TPAMI.2017.2699960

[85] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021. A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search. Proc. VLDB Endow. 14, 11 (jul 2021), 1964-1978. https://doi.org/10.14778/3476249.3476255

[86] Yifan Wang, Haodi Ma, and Daisy Zhe Wang. 2022. LIDER: An Efficient High-Dimensional Learned Index for Large-Scale Dense Passage Retrieval. Proc. VLDB Endow. 16, 2 (oct 2022), 154-166. https://doi.org/10.14778/3565816.3565819

[87] Zeyu Wang, Peng Wang, Themis Palpanas, and Wei Wang. 2023. Graph-and Tree-based Indexes for High-dimensional Vector Similarity Search: Analyses, Comparisons, and Future Directions. Data Engineering (2023), 3-21.

[88] Zeyu Wang, Qitong Wang, Peng Wang, Themis Palpanas, and Wei Wang. 2023. Dumpy: A compact and adaptive index for large data series collections. Proceedings of the ACM on Management of Data 1, 1 (2023), 1-27.

[89] Roger Weber, Hans-Jörg Schek, and Stephen Blott. 1998. A Quantitative Analysis and Performance Study for Similarity-Search Methods in High-Dimensional Spaces. In Proceedings of the 24rd International Conference on Very Large Data Bases (VLDB '98). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 194-205.

[90] Shitao Xiao, Zheng Liu, Weihao Han, Jianjin Zhang, Defu Lian, Yeyun Gong, Qi Chen, Fan Yang, Hao Sun, Yingxia Shao, and Xing Xie. 2022. Distill-VQ: Learning Retrieval Oriented Vector Quantization By Distilling Knowledge from Dense Embeddings. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR '22). ACM, New York, NY, USA, 1513-1523. https://doi.org/10.1145/3477495.3531799

[91] Wen Yang, Tao Li, Gai Fang, and Hong Wei. 2020. PASE: PostgreSQL Ultra-High-Dimensional Approximate Nearest Neighbor Search Extension. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (Portland, OR, USA) (SIGMOD '20). Association for Computing Machinery, New York, NY, USA, 2241-2253. https://doi.org/10.1145/3318464.3386131

[92] R. Zamir and M. Feder. 1992. On universal quantization by randomized uniform/lattice quantizers. IEEE Transactions on Information Theory 38, 2 (1992), 428-436. https://doi.org/10.1109/18.119699

[93] Pavel Zezula, Giuseppe Amato, Vlastislav Dohnal, and Michal Batko. 2010. Similarity Search: The Metric Space Approach (1st ed.). Springer Publishing Company, Incorporated.

[94] Ting Zhang, Chao Du, and Jingdong Wang. 2014. Composite Quantization for Approximate Nearest Neighbor Search. In Proceedings of the 31st International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 32), Eric P. Xing and Tony Jebara (Eds.). PMLR, Bejing, China, 838-846. https://proceedings.mlr.press/v32/zhangd14.html

[95] Xi Zhao, Yao Tian, Kai Huang, Bolong Zheng, and Xiaofang Zhou. 2023. Towards Efficient Index Construction and Approximate Nearest Neighbor Search in High-Dimensional Spaces. Proc. VLDB Endow. 16, 8 (jun 2023), 1979-1991. https://doi.org/10.14778/3594512.3594527

[96] Bolong Zheng, Zhao Xi, Lianggui Weng, Nguyen Quoc Viet Hung, Hang Liu, and Christian S Jensen. 2020. PM-LSH: A fast and accurate LSH framework for high-dimensional approximate NN search. Proceedings of the VLDB Endowment 13,5 (2020), 643-655.

Received October 2023; revised January 2024; accepted February 2024

2023 年 10 月收到；2024 年 1 月修订；2024 年 2 月接受