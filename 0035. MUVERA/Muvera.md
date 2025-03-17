# MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings

# MUVERA：基于固定维度编码的多向量检索

Laxman Dhulipala

拉克什曼·杜利帕拉

Google Research and UMD

谷歌研究院和马里兰大学学院市分校

Majid Hadian

马吉德·哈迪安

Google DeepMind

谷歌深度思维

Rajesh Jayaram*

拉杰什·贾亚拉姆*

Google Research

谷歌研究院

Jason Lee

杰森·李

Google Research

谷歌研究院

Vahab Mirrokni

瓦哈布·米罗克尼

Google Research

谷歌研究院

## Abstract

## 摘要

Neural embedding models have become a fundamental component of modern information retrieval (IR) pipelines. These models produce a single embedding $x \in  {\mathbb{R}}^{d}$ per data-point,allowing for fast retrieval via highly optimized maximum inner product search (MIPS) algorithms. Recently, beginning with the landmark ColBERT paper, multi-vector models, which produce a set of embedding per data point, have achieved markedly superior performance for IR tasks. Unfortunately, using these models for IR is computationally expensive due to the increased complexity of multi-vector retrieval and scoring.

神经嵌入模型已成为现代信息检索（IR）流程的基本组成部分。这些模型为每个数据点生成一个单一嵌入向量 $x \in  {\mathbb{R}}^{d}$，从而可以通过高度优化的最大内积搜索（MIPS）算法实现快速检索。最近，从具有里程碑意义的 ColBERT 论文开始，多向量模型（为每个数据点生成一组嵌入向量）在信息检索任务中取得了显著更优的性能。不幸的是，由于多向量检索和评分的复杂性增加，使用这些模型进行信息检索的计算成本很高。

In this paper, we introduce MUVERA (Multi-Vector Retrieval Algorithm), a retrieval mechanism which reduces multi-vector similarity search to single-vector similarity search. This enables the usage of off-the-shelf MIPS solvers for multi-vector retrieval. MUVERA asymmetrically generates Fixed Dimensional Encodings (FDEs) of queries and documents, which are vectors whose inner product approximates multi-vector similarity. We prove that FDEs give high-quality $\varepsilon$ - approximations, thus providing the first single-vector proxy for multi-vector similarity with theoretical guarantees. Empirically, we find that FDEs achieve the same recall as prior state-of-the-art heuristics while retrieving $2 - 5 \times$ fewer candidates. Compared to prior state of the art implementations, MUVERA achieves consistently good end-to-end recall and latency across a diverse set of the BEIR retrieval datasets,achieving an average of ${10}\%$ improved recall with ${90}\%$ lower latency.

在本文中，我们介绍了 MUVERA（多向量检索算法），这是一种将多向量相似性搜索简化为单向量相似性搜索的检索机制。这使得可以使用现成的 MIPS 求解器进行多向量检索。MUVERA 以非对称方式生成查询和文档的固定维度编码（FDE），这些向量的内积近似于多向量相似性。我们证明了 FDE 能提供高质量的 $\varepsilon$ - 近似，从而首次为多向量相似性提供了具有理论保证的单向量代理。从经验上看，我们发现 FDE 在检索 $2 - 5 \times$ 个更少候选对象的情况下，能达到与先前最先进启发式方法相同的召回率。与先前最先进的实现相比，MUVERA 在各种 BEIR 检索数据集上始终能实现良好的端到端召回率和低延迟，平均召回率提高了 ${10}\%$，延迟降低了 ${90}\%$。

## 1 Introduction

## 1 引言

Over the past decade, the use of neural embeddings for representing data has become a central tool for information retrieval (IR) [56], among many other tasks such as clustering and classification [39]. Recently, multi-vector (MV) representations, introduced by the late-interaction framework in ColBERT [29], have been shown to deliver significantly improved performance on popular IR benchmarks. ColBERT and its variants [17, 21, 32, 35, 42, 44, 49, 54] produce multiple embeddings per query or document by generating one embedding per token. The query-document similarity is then scored via the Chamfer Similarity (§1.1), also known as the MaxSim operation, between the two sets of vectors. These multi-vector representations have many advantages over single-vector (SV) representations,such as better interpretability $\left\lbrack  {{15},{50}}\right\rbrack$ and generalization $\left\lbrack  {{16},{36},{51},{55}}\right\rbrack$ .

在过去十年中，使用神经嵌入来表示数据已成为信息检索（IR）[56]的核心工具，同时也用于聚类和分类等许多其他任务[39]。最近，由 ColBERT [29] 中的后期交互框架引入的多向量（MV）表示，已被证明在流行的信息检索基准测试中显著提高了性能。ColBERT 及其变体 [17, 21, 32, 35, 42, 44, 49, 54] 通过为每个标记生成一个嵌入向量，为每个查询或文档生成多个嵌入向量。然后，通过两组向量之间的倒角相似性（§1.1）（也称为 MaxSim 操作）对查询 - 文档相似性进行评分。与单向量（SV）表示相比，这些多向量表示具有许多优势，例如更好的可解释性 $\left\lbrack  {{15},{50}}\right\rbrack$ 和泛化能力 $\left\lbrack  {{16},{36},{51},{55}}\right\rbrack$。

Despite these advantages, multi-vector retrieval is inherently more expensive than single-vector retrieval. Firstly, producing one embedding per token increases the number of embeddings in a dataset by orders of magnitude. Moreover, due to the non-linear Chamfer similarity scoring, there is a lack of optimized systems for multi-vector retrieval. Specifically, single-vector retrieval is generally accomplished via Maximum Inner Product Search (MIPS) algorithms, which have been highly-optimized over the past few decades [18]. However, SV MIPS alone cannot be used for MV retrieval. This is because the MV similarity is the sum of the SV similarities of each embedding in a query to the nearest embedding in a document. Thus, a document containing a token with high similarity to a single query token may not be very similar to the query overall. Thus, in an effort to close the gap between SV and MV retrieval, there has been considerable work in recent years to design custom MV retrieval algorithms with improved efficiency [12, 21, 42, 43].

尽管有这些优势，但多向量检索本质上比单向量检索成本更高。首先，为每个标记生成一个嵌入会使数据集中的嵌入数量增加几个数量级。此外，由于非线性的倒角相似度评分，缺乏针对多向量检索的优化系统。具体而言，单向量检索通常通过最大内积搜索（Maximum Inner Product Search，MIPS）算法完成，这些算法在过去几十年中已经得到了高度优化 [18]。然而，仅靠单向量 MIPS 无法用于多向量检索。这是因为多向量相似度是查询中的每个嵌入与文档中最近嵌入的单向量相似度之和。因此，包含与单个查询标记高度相似的标记的文档可能与整个查询的相似度并不高。因此，为了缩小单向量和多向量检索之间的差距，近年来有大量工作致力于设计效率更高的自定义多向量检索算法 [12, 21, 42, 43]。

---

<!-- Footnote -->

*Corresponding Author: rkjayaram@google.com

*通讯作者：rkjayaram@google.com

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: MUVERA Stage 2: Centroid Embeddings FDEs Query FDE Chamfer Reranking (DiskANN) Documents Stage 1: Construct FDEs Stage 2: Chamfer and Query MIPS Reranking -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_1.jpg?x=318&y=163&w=1164&h=371&r=0"/>

Figure 1: MUVERA's two-step retrieval process, comapred to PLAID's multi-stage retrieval process. Diagram on the right from Santhanam et. al. [43] with permission.

图 1：MUVERA 的两步检索过程，与 PLAID 的多阶段检索过程对比。右侧图表经许可取自 Santhanam 等人 [43]。

<!-- Media -->

The most prominent approach to MV retrieval is to employ a multi-stage pipeline beginning with single-vector MIPS. The basic version of this approach is as follows: in the initial stage, the most similar document tokens are found for each of the query tokens using SV MIPS. Then the corresponding documents containing these tokens are gathered together and rescored with the original Chamfer similarity. We refer to this method as the single-vector heuristic. ColBERTv2 [44] and its optimized retrieval engine PLAID [43] are based on this approach, with the addition of several intermediate stages of pruning. In particular, PLAID employs a complex four-stage retrieval and pruning process to gradually reduce the number of final candidates to be scored (Figure 1). Unfortunately, as described above, employing SV MIPS on individual query embeddings can fail to find the true MV nearest neighbors. Additionally, this process is expensive, since it requires querying a significantly larger MIPS index for every query embedding (larger because there are multiple embeddings per document). Finally, these multi-stage pipelines are complex and highly sensitive to parameter setting, as recently demonstrated in a reproducibility study [37], making them difficult to tune. To address these challenges and bridge the gap between single and multi-vector retrieval, in this paper we seek to design faster and simplified MV retrieval algorithms.

多向量检索最突出的方法是采用一个从单向量 MIPS 开始的多阶段管道。这种方法的基本版本如下：在初始阶段，使用单向量 MIPS 为每个查询标记找到最相似的文档标记。然后将包含这些标记的相应文档收集起来，并使用原始的倒角相似度重新评分。我们将这种方法称为单向量启发式方法。ColBERTv2 [44] 及其优化的检索引擎 PLAID [43] 就是基于这种方法，并增加了几个中间修剪阶段。特别是，PLAID 采用了一个复杂的四阶段检索和修剪过程，以逐步减少最终需要评分的候选文档数量（图 1）。不幸的是，如上所述，对单个查询嵌入使用单向量 MIPS 可能无法找到真正的多向量最近邻。此外，这个过程成本很高，因为它需要为每个查询嵌入查询一个大得多的 MIPS 索引（之所以更大，是因为每个文档有多个嵌入）。最后，这些多阶段管道很复杂，并且对参数设置非常敏感，正如最近一项可重复性研究 [37] 所表明的那样，这使得它们难以调整。为了应对这些挑战并缩小单向量和多向量检索之间的差距，在本文中，我们试图设计更快、更简化的多向量检索算法。

Contributions. We propose MUVERA: a multi-vector retrieval mechanism based on a light-weight and provably correct reduction to single-vector MIPS. MUVERA employs a fast, data-oblivious transformation from a set of vectors to a single vector, allowing for retrieval via highly-optimized MIPS solvers before a single stage of re-ranking. Specifically, MUVERA transforms query and document MV sets $Q,P \subset  {\mathbb{R}}^{d}$ into single fixed-dimensional vectors $\overrightarrow{q},\overrightarrow{p}$ ,called Fixed Dimensional Encodings (FDEs),such that the the dot product $\overrightarrow{q} \cdot  \overrightarrow{p}$ approximates the multi-vector similarity between $Q,P\left( {§2}\right)$ . Empirically,we show that retrieving with respect to the FDE dot product significantly outperforms the single vector heuristic at recovering the Chamfer nearest neighbors (§3.1). For instance,on MS MARCO,our FDEs Recall@N surpasses the Recall@2-5N achieved by the SV heuristic while scanning a similar total number of floats in the search.

贡献。我们提出了 MUVERA：一种基于轻量级且可证明正确地简化为单向量 MIPS 的多向量检索机制。MUVERA 采用了一种快速、与数据无关的转换方法，将一组向量转换为单个向量，从而允许在单阶段重排序之前通过高度优化的 MIPS 求解器进行检索。具体而言，MUVERA 将查询和文档的多向量集 $Q,P \subset  {\mathbb{R}}^{d}$ 转换为单个固定维度的向量 $\overrightarrow{q},\overrightarrow{p}$，称为固定维度编码（Fixed Dimensional Encodings，FDEs），使得点积 $\overrightarrow{q} \cdot  \overrightarrow{p}$ 近似于 $Q,P\left( {§2}\right)$ 之间的多向量相似度。根据经验，我们表明，相对于 FDE 点积进行检索在恢复倒角最近邻方面明显优于单向量启发式方法（§3.1）。例如，在 MS MARCO 数据集上，我们的 FDEs 的 Recall@N 超过了单向量启发式方法在扫描相似数量的浮点数时所达到的 Recall@2 - 5N。

We prove in (§2.1) that our FDEs have strong approximation guarantees; specifically, the FDE dot product gives an $\varepsilon$ -approximation to the true MV similarity. This gives the first algorithm with provable guarantees for Chamfer similarity search with strictly faster than brute-force runtime (Theorem 2.2). Thus, MUVERA provides the first principled method for MV retrieval via a SV proxy.

我们在（§2.1）中证明了我们的 FDEs 具有很强的近似保证；具体而言，FDE 点积对真正的多向量相似度给出了 $\varepsilon$ 近似。这为倒角相似度搜索提供了第一个具有可证明保证且运行时间严格快于暴力搜索的算法（定理 2.2）。因此，MUVERA 提供了第一种通过单向量代理进行多向量检索的原则性方法。

We compare the end-to-end retrieval performance of MUVERA to PLAID on several of the BEIR IR datasets, including the well-studied MS MARCO dataset. We find MUVERA to be a robust and efficient retrieval mechanism; across the datasets we evaluated, MUVERA obtains an average of 10% higher recall,while requiring ${90}\%$ lower latency on average compared with PLAID. Additionally, MUVERA crucially incorporates a vector compression technique called product quantization that enables us to compress the FDEs by ${32} \times$ (i.e.,storing 10240 dimensional FDEs using 1280 bytes) while incurring negligible quality loss, resulting in a significantly smaller memory footprint.

我们在几个 BEIR 信息检索（IR）数据集上，包括经过充分研究的 MS MARCO 数据集，比较了 MUVERA 和 PLAID 的端到端检索性能。我们发现 MUVERA 是一种稳健且高效的检索机制；在我们评估的数据集上，与 PLAID 相比，MUVERA 的平均召回率提高了 10%，同时平均延迟降低了 ${90}\%$。此外，MUVERA 关键地采用了一种称为乘积量化的向量压缩技术，该技术使我们能够将 FDEs 压缩 ${32} \times$（即，使用 1280 字节存储 10240 维的 FDEs），同时质量损失可以忽略不计，从而显著减少了内存占用。

### 1.1 Chamfer Similarity and the Multi-Vector Retrieval Problem

### 1.1 倒角相似度与多向量检索问题

Given two sets of vectors $Q,P \subset  {\mathbb{R}}^{d}$ ,the Chamfer Similarity is given by

给定两组向量 $Q,P \subset  {\mathbb{R}}^{d}$ ，倒角相似度（Chamfer Similarity）的计算公式如下

$$
\operatorname{CHAMFER}\left( {Q,P}\right)  = \mathop{\sum }\limits_{{q \in  Q}}\mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle 
$$

where $\langle  \cdot  , \cdot  \rangle$ is the standard vector inner product. Chamfer similarity is the default method of MV similarity used in the late-interaction architecture of ColBERT, which includes systems like ColBERTv2 [44], Baleen [28], Hindsight [41], DrDecr [34], and XTR [32], among many others. These models encode queries and documents as sets $Q,P \subset  {\mathbb{R}}^{d}$ (respectively),where the query-document similarity is given by $\operatorname{CHAMFER}\left( {Q,P}\right)$ . We note that Chamfer Similarity (and its distance variant) itself has a long history of study in the computer vision (e.g., [4, 6, 14, 27, 45]) and graphics [33] communities, and had been previously used in the ML literature to compare sets of embeddings $\left\lbrack  {3,5,{30},{48}}\right\rbrack$ . In these works, Chamfer is also referred to as MaxSim or the relaxed earth mover distance; we choose the terminology Chamfer due to its historical precedence [6].

其中 $\langle  \cdot  , \cdot  \rangle$ 是标准向量内积。倒角相似度是ColBERT后期交互架构中使用的多向量（MV）相似度的默认方法，采用该架构的系统包括ColBERTv2 [44]、Baleen [28]、Hindsight [41]、DrDecr [34] 和XTR [32] 等。这些模型将查询和文档分别编码为集合 $Q,P \subset  {\mathbb{R}}^{d}$ ，其中查询 - 文档相似度由 $\operatorname{CHAMFER}\left( {Q,P}\right)$ 给出。我们注意到，倒角相似度（及其距离变体）本身在计算机视觉（例如 [4, 6, 14, 27, 45]）和图形学 [33] 领域有着悠久的研究历史，并且之前在机器学习文献中已被用于比较嵌入集合 $\left\lbrack  {3,5,{30},{48}}\right\rbrack$ 。在这些研究中，倒角相似度也被称为最大相似度（MaxSim）或松弛的推土机距离；由于其历史优先性 [6]，我们选择使用“倒角相似度”这一术语。

In this paper, we study the problem of Nearest Neighbor Search (NNS) with respect to the Chamfer Similarity. Specifically,we are given a dataset $D = \left\{  {{P}_{1},\ldots ,{P}_{n}}\right\}$ where each ${P}_{i} \subset  {\mathbb{R}}^{d}$ is a set of vectors. Given a query subset $Q \subset  {\mathbb{R}}^{d}$ ,the goal is to quickly recover the nearest neighbor ${P}^{ * } \in  D$ ,

在本文中，我们研究了关于倒角相似度的最近邻搜索（NNS）问题。具体来说，给定一个数据集 $D = \left\{  {{P}_{1},\ldots ,{P}_{n}}\right\}$ ，其中每个 ${P}_{i} \subset  {\mathbb{R}}^{d}$ 是一组向量。给定一个查询子集 $Q \subset  {\mathbb{R}}^{d}$ ，目标是快速找到最近邻 ${P}^{ * } \in  D$ ，

namely:

即：

$$
{P}^{ * } = \arg \mathop{\max }\limits_{{{P}_{i} \in  D}}\operatorname{CHAMFER}\left( {Q,{P}_{i}}\right) 
$$

For the retrieval system to be scalable, this must be achieved in time significantly faster than brute-force scoring each of the $n$ similarities $\operatorname{CHAMFER}\left( {Q,{P}_{i}}\right)$ .

为了使检索系统具有可扩展性，必须在比暴力计算 $n$ 个相似度 $\operatorname{CHAMFER}\left( {Q,{P}_{i}}\right)$ 显著更快的时间内实现这一目标。

### 1.2 Our Approach: Reducing Multi-Vector Search to Single-Vector MIPS

### 1.2 我们的方法：将多向量搜索简化为单向量最大内积搜索（MIPS）

MUVERA is a streamlined procedure that directly reduces the Chamfer Similarity Search to MIPS. For a pre-specified target dimension ${d}_{\mathrm{{FDE}}}$ ,MUVERA produces randomized mappings ${\mathbf{F}}_{\mathrm{q}} : {2}^{{\mathbb{R}}^{d}} \rightarrow  {\mathbb{R}}^{{d}_{\mathrm{{FDE}}}}$ (for queries) and ${\mathbf{F}}_{\text{doc }} : {2}^{{\mathbb{R}}^{d}} \rightarrow  {\mathbb{R}}^{{d}_{\mathrm{{FDE}}}}$ (for documents) such that,for all query and document multivector representations $Q,P \subset  {\mathbb{R}}^{d}$ ,we have:

MUVERA是一种简化的过程，它直接将倒角相似度搜索简化为最大内积搜索（MIPS）。对于预先指定的目标维度 ${d}_{\mathrm{{FDE}}}$ ，MUVERA生成随机映射 ${\mathbf{F}}_{\mathrm{q}} : {2}^{{\mathbb{R}}^{d}} \rightarrow  {\mathbb{R}}^{{d}_{\mathrm{{FDE}}}}$ （用于查询）和 ${\mathbf{F}}_{\text{doc }} : {2}^{{\mathbb{R}}^{d}} \rightarrow  {\mathbb{R}}^{{d}_{\mathrm{{FDE}}}}$ （用于文档），使得对于所有查询和文档的多向量表示 $Q,P \subset  {\mathbb{R}}^{d}$ ，我们有：

$$
\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   \approx  \operatorname{CHAMFER}\left( {Q,P}\right) 
$$

We refer to the vectors ${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\text{doc }}\left( P\right)$ as Fixed Dimensional Encodings (FDEs). MUVERA first applies ${\mathbf{F}}_{\text{doc }}$ to each document representation $P \in  D$ ,and indexes the set ${\left\{  {\mathbf{F}}_{\text{doc }}\left( P\right) \right\}  }_{P \in  D}$ into a MIPS solver. Given a query $Q \subset  {\mathbb{R}}^{d}$ ,MUVERA quickly computes ${\mathbf{F}}_{\mathfrak{q}}\left( Q\right)$ and feeds it to the MIPS solver to recover top- $k$ most similar document FDE’s ${\mathbf{F}}_{\text{doc }}\left( P\right)$ . Finally,we re-rank these candidates by the original Chamfer similarity. See Figure 1 for an overview. We remark that one important advantage of the FDEs is that the functions ${\mathbf{F}}_{\mathrm{q}},{\mathbf{F}}_{\text{doc }}$ are data-oblivious,making them robust to distribution shifts, and easily usable in streaming settings.

我们将向量 ${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\text{doc }}\left( P\right)$ 称为固定维度编码（FDEs）。MUVERA首先将 ${\mathbf{F}}_{\text{doc }}$ 应用于每个文档表示 $P \in  D$ ，并将集合 ${\left\{  {\mathbf{F}}_{\text{doc }}\left( P\right) \right\}  }_{P \in  D}$ 索引到一个最大内积搜索求解器中。给定一个查询 $Q \subset  {\mathbb{R}}^{d}$ ，MUVERA快速计算 ${\mathbf{F}}_{\mathfrak{q}}\left( Q\right)$ 并将其输入到最大内积搜索求解器中，以找回前 $k$ 个最相似的文档固定维度编码 ${\mathbf{F}}_{\text{doc }}\left( P\right)$ 。最后，我们根据原始的倒角相似度对这些候选文档进行重新排序。总体流程如图1所示。我们要指出的是，固定维度编码的一个重要优点是函数 ${\mathbf{F}}_{\mathrm{q}},{\mathbf{F}}_{\text{doc }}$ 与数据无关，这使得它们对分布变化具有鲁棒性，并且易于在流式设置中使用。

### 1.3 Related Work on Multi-Vector Retrieval

### 1.3 多向量检索的相关工作

The early multi-vector retrieval systems, such as ColBERT [29], all implement optimizations of the previously described SV heuristic, where the initial set of candidates is found by querying a MIPS index for every query token $q \in  Q$ . In ColBERTv2 [44],the document token embeddings are first clustered via k-means, and the first round of scoring using cluster centroids instead of the original token. This technique was further optimized in PLAID [43] by employing a four-stage pipeline to progressively prune candidates before a final reranking (Figure 1).

早期的多向量检索系统，如ColBERT [29]，都实现了前文所述的单向量（SV）启发式方法的优化，即通过为每个查询词元 $q \in  Q$ 查询最大内积搜索（MIPS）索引来找到初始候选集。在ColBERTv2 [44] 中，文档词元嵌入首先通过k - 均值聚类，第一轮评分使用聚类中心而非原始词元。PLAID [43] 通过采用一个四阶段的流水线在最终重排序之前逐步修剪候选集，进一步优化了该技术（图1）。

An alternative approach with proposed in DESSERT [12], whose authors also pointed out the limitations of the SV heuristic, and proposed an algorithm based on Locality Sensitive Hashing (LSH) [20]. They prove that their algorithm recovers $\varepsilon$ -approximate nearest neighbors in time $\widetilde{O}\left( {n\left| Q\right| T}\right)$ , where $T$ is roughly the maximum number of document tokens $p \in  {P}_{i}$ that are similar to any query token $q \in  Q$ ,which can be as large as $\mathop{\max }\limits_{i}\left| {P}_{i}\right|$ . Thus,in the worst case,their algorithm runs no faster than brute-force. Conversely,our algorithm recovers $\varepsilon$ -approximate nearest neighbors and always runs in time $\widetilde{O}\left( {n\left| Q\right| }\right)$ . Experimentally,DESSERT is 2-5 $\times$ faster than PLAID,but attains worse recall (e.g. 2-2.5% R@1000 on MS MARCO). Conversely, we match and sometimes strongly exceed PLAID’s recall with up to ${5.7} \times$ lower latency. Additionally,DESSERT still employs an initial filtering stage based on $k$ -means clustering of individual query token embeddings (in the manner of ColBERTv2), thus they do not truly avoid the aforementioned limitations of the SV heuristic.

DESSERT [12] 提出了一种替代方法，其作者也指出了单向量（SV）启发式方法的局限性，并提出了一种基于局部敏感哈希（LSH） [20] 的算法。他们证明了其算法能在时间 $\widetilde{O}\left( {n\left| Q\right| T}\right)$ 内恢复 $\varepsilon$ -近似最近邻，其中 $T$ 大致是与任何查询词元 $q \in  Q$ 相似的文档词元 $p \in  {P}_{i}$ 的最大数量，该数量可能高达 $\mathop{\max }\limits_{i}\left| {P}_{i}\right|$ 。因此，在最坏情况下，他们的算法运行速度并不比暴力搜索快。相反，我们的算法能恢复 $\varepsilon$ -近似最近邻，且始终能在时间 $\widetilde{O}\left( {n\left| Q\right| }\right)$ 内运行。实验表明，DESSERT比PLAID快2 - 5 $\times$ ，但召回率较差（例如在MS MARCO数据集上R@1000为2 - 2.5%）。相反，我们的召回率与PLAID相当，有时还远超PLAID，且延迟最多降低 ${5.7} \times$ 。此外，DESSERT仍然采用了基于单个查询词元嵌入的 $k$ -均值聚类的初始过滤阶段（采用ColBERTv2的方式），因此他们并没有真正避免上述单向量（SV）启发式方法的局限性。

## 2 Fixed Dimensional Encodings

## 2 固定维度编码

We now describe our process for generating FDEs. Our transformation is reminiscent of the technique of probabilistic tree embeddings $\left\lbrack  {1,7,{10},{13}}\right\rbrack$ ,which can be used to transform a set of vectors into a single vector. For instance, they have been used to embed the Earth Mover's Distance into the ${\ell }_{1}$ metric $\left\lbrack  {1,{10},{22},{24}}\right\rbrack$ ,and to embed the weight of a Euclidean MST of a set of vectors into the Hamming metric $\left\lbrack  {9,{22},{23}}\right\rbrack$ . However,since we are working with inner products,which are not metrics,instead of ${\ell }_{p}$ distances,an alternative approach for our transformation will be needed.

我们现在描述生成固定维度编码（FDE）的过程。我们的变换让人想起概率树嵌入技术 $\left\lbrack  {1,7,{10},{13}}\right\rbrack$ ，该技术可用于将一组向量转换为单个向量。例如，它已被用于将推土机距离嵌入到 ${\ell }_{1}$ 度量 $\left\lbrack  {1,{10},{22},{24}}\right\rbrack$ 中，以及将一组向量的欧几里得最小生成树（MST）的权重嵌入到汉明度量 $\left\lbrack  {9,{22},{23}}\right\rbrack$ 中。然而，由于我们处理的是内积，而内积不是度量，而非 ${\ell }_{p}$ 距离，因此我们的变换需要一种替代方法。

The intuition behind our transformation is as follows. Hypothetically, for two MV representations $Q,P \subset  {\mathbb{R}}^{d}$ ,if we knew the optimal mapping $\pi  : Q \rightarrow  P$ in which to match them,then we could create vectors $\overrightarrow{q},\overrightarrow{p}$ by concatenating all the vectors in $Q$ and their corresponding images in $P$ together, so that $\langle \overrightarrow{q},\overrightarrow{p}\rangle  = \mathop{\sum }\limits_{{q \in  Q}}\langle q,\pi \left( q\right) \rangle  = \operatorname{CHAMFER}\left( {Q,P}\right)$ . However,since we do not know $\pi$ in advance, and since different query-document pairs have different optimal mappings, this simple concatenation clearly will not work. Instead,our goal is to find a randomized ordering over all the points in ${\mathbb{R}}^{d}$ so that,after clustering close points together,the dot product of any query-document pair $Q,P \subset  {\mathbb{R}}^{d}$ concatenated into a single vector under this ordering will approximate the Chamfer similarity.

我们变换背后的直觉如下。假设对于两个多向量（MV）表示 $Q,P \subset  {\mathbb{R}}^{d}$ ，如果我们知道匹配它们的最优映射 $\pi  : Q \rightarrow  P$ ，那么我们可以通过将 $Q$ 中的所有向量及其在 $P$ 中的对应映射连接在一起创建向量 $\overrightarrow{q},\overrightarrow{p}$ ，使得 $\langle \overrightarrow{q},\overrightarrow{p}\rangle  = \mathop{\sum }\limits_{{q \in  Q}}\langle q,\pi \left( q\right) \rangle  = \operatorname{CHAMFER}\left( {Q,P}\right)$ 。然而，由于我们事先不知道 $\pi$ ，并且不同的查询 - 文档对有不同的最优映射，这种简单的连接显然行不通。相反，我们的目标是在 ${\mathbb{R}}^{d}$ 中的所有点上找到一种随机排序，使得在将相近的点聚类在一起后，在这种排序下连接成单个向量的任何查询 - 文档对 $Q,P \subset  {\mathbb{R}}^{d}$ 的点积将近似于倒角相似度。

The first step is to partition the latent space ${\mathbb{R}}^{d}$ into $B$ clusters so that vectors that are closer are more are more likely to land in the same cluster. Let $\varphi  : {\mathbb{R}}^{d} \rightarrow  \left\lbrack  B\right\rbrack$ be such a partition; $\varphi$ can be implemented via Locality Sensitive Hashing (LSH) [20], $k$ -means,or other methods; we discuss choices for $\varphi$ later in this section. After partitioning via $\varphi$ ,the hope is that for each $q \in  Q$ ,the closest $p \in  P$ lands in the same cluster (i.e. $\varphi \left( q\right)  = \varphi \left( p\right)$ ). Hypothetically,if this were to occur,then:

第一步是将潜在空间 ${\mathbb{R}}^{d}$ 划分为 $B$ 个聚类，以便距离更近的向量更有可能落入同一个聚类中。设 $\varphi  : {\mathbb{R}}^{d} \rightarrow  \left\lbrack  B\right\rbrack$ 为这样的一个划分；$\varphi$ 可以通过局部敏感哈希（Locality Sensitive Hashing，LSH） [20]、$k$ -均值法或其他方法来实现；我们将在本节后面讨论 $\varphi$ 的选择。通过 $\varphi$ 进行划分后，我们希望对于每个 $q \in  Q$，最近的 $p \in  P$ 落入同一个聚类中（即 $\varphi \left( q\right)  = \varphi \left( p\right)$）。假设这种情况发生，那么：

$$
\operatorname{CHAMFER}\left( {Q,P}\right)  = \mathop{\sum }\limits_{{k = 1}}^{B}\mathop{\sum }\limits_{\substack{{q \in  Q} \\  {\varphi \left( q\right)  = k} }}\mathop{\max }\limits_{\substack{{p \in  P} \\  {\varphi \left( p\right)  = k} }}\langle q,p\rangle  \tag{1}
$$

If $p$ is the only point in $P$ that collides with $q$ ,then (1) can be realized as a dot product between two vectors $\overrightarrow{q},\overrightarrow{p}$ by creating one block of $d$ coordinates in $\overrightarrow{q},\overrightarrow{p}$ for each cluster $k \in  \left\lbrack  B\right\rbrack$ (call these blocks ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) } \in  {\mathbb{R}}^{d}$ ),and setting ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) }$ to be the sum of all $q \in  Q$ (resp. $p \in  P$ ) that land in the $k$ -th cluster under $\varphi$ . However,if multiple ${p}^{\prime } \in  P$ collide with $q$ ,then $\langle \overrightarrow{q},\overrightarrow{p}\rangle$ will differ from (1),since every ${p}^{\prime }$ with $\varphi \left( {p}^{\prime }\right)  = \varphi \left( q\right)$ will contribute at least $\left\langle  {q,{p}^{\prime }}\right\rangle$ to $\langle \overrightarrow{q},\overrightarrow{p}\rangle$ . To resolve this,we set ${\overrightarrow{p}}_{\left( k\right) }$ to be the centroid of the $p \in  P$ ’s with $\varphi \left( p\right)  = \varphi \left( q\right)$ . Formally,for $k = 1,\ldots ,B$ ,we define

如果 $p$ 是 $P$ 中唯一与 $q$ 发生碰撞的点，那么可以通过为每个聚类 $k \in  \left\lbrack  B\right\rbrack$ 在 $\overrightarrow{q},\overrightarrow{p}$ 中创建一个包含 $d$ 个坐标的块（称这些块为 ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) } \in  {\mathbb{R}}^{d}$），并将 ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) }$ 设置为在 $\varphi$ 下落入第 $k$ 个聚类的所有 $q \in  Q$（分别对应 $p \in  P$）的总和，从而将 (1) 实现为两个向量 $\overrightarrow{q},\overrightarrow{p}$ 之间的点积。然而，如果多个 ${p}^{\prime } \in  P$ 与 $q$ 发生碰撞，那么 $\langle \overrightarrow{q},\overrightarrow{p}\rangle$ 将与 (1) 不同，因为每个满足 $\varphi \left( {p}^{\prime }\right)  = \varphi \left( q\right)$ 的 ${p}^{\prime }$ 至少会对 $\langle \overrightarrow{q},\overrightarrow{p}\rangle$ 贡献 $\left\langle  {q,{p}^{\prime }}\right\rangle$。为了解决这个问题，我们将 ${\overrightarrow{p}}_{\left( k\right) }$ 设置为满足 $\varphi \left( p\right)  = \varphi \left( q\right)$ 的 $p \in  P$ 的质心。形式上，对于 $k = 1,\ldots ,B$，我们定义

$$
{\overrightarrow{q}}_{\left( k\right) } = \mathop{\sum }\limits_{\substack{{q \in  Q} \\  {\varphi \left( q\right)  = k} }}q,\;{\overrightarrow{p}}_{\left( k\right) } = \frac{1}{\left| P \cap  {\varphi }^{-1}\left( k\right) \right| }\mathop{\sum }\limits_{\substack{{p \in  P} \\  {\varphi \left( p\right)  = k} }}p \tag{2}
$$

Setting $\overrightarrow{q} = \left( {{\overrightarrow{q}}_{\left( 1\right) },\ldots ,{\overrightarrow{q}}_{\left( B\right) }}\right)$ and $\overrightarrow{p} = \left( {{\overrightarrow{p}}_{\left( 1\right) },\ldots ,{\overrightarrow{p}}_{\left( B\right) }}\right)$ ,then we have

设置 $\overrightarrow{q} = \left( {{\overrightarrow{q}}_{\left( 1\right) },\ldots ,{\overrightarrow{q}}_{\left( B\right) }}\right)$ 和 $\overrightarrow{p} = \left( {{\overrightarrow{p}}_{\left( 1\right) },\ldots ,{\overrightarrow{p}}_{\left( B\right) }}\right)$，那么我们有

$$
\langle \overrightarrow{q},\overrightarrow{p}\rangle  = \mathop{\sum }\limits_{{k = 1}}^{B}\mathop{\sum }\limits_{\substack{{q \in  Q} \\  {\varphi \left( q\right)  = k} }}\frac{1}{\left| P \cap  {\varphi }^{-1}\left( k\right) \right| }\mathop{\sum }\limits_{\substack{{p \in  P} \\  {\varphi \left( p\right)  = k} }}\langle q,p\rangle  \tag{3}
$$

Note that the resulting dimension of the vectors $\overrightarrow{q},\overrightarrow{p}$ is ${dB}$ . To reduce the dependency on $d$ ,we can apply a random linear projection $\psi  : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}_{\text{proj }}}$ to each block ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) }$ ,where ${d}_{\text{proj }} < d$ . Specifically,we define $\psi \left( x\right)  = \left( {1/\sqrt{{d}_{\text{proj }}}}\right) {Sx}$ ,where $S \in  {\mathbb{R}}^{{d}_{\text{proj }} \times  d}$ is a random matrix with uniformly distributed $\pm  1$ entries. We can then define ${\overrightarrow{q}}_{\left( k\right) ,\psi } = \psi \left( {\overrightarrow{q}}_{\left( k\right) }\right)$ and ${\overrightarrow{p}}_{\left( k\right) ,\psi } = \psi \left( {\overrightarrow{p}}_{\left( k\right) }\right)$ ,and define the FDE’s with inner projection as ${\overrightarrow{q}}_{\psi } = \left( {{\overrightarrow{q}}_{\left( 1\right) ,\psi },\ldots ,{\overrightarrow{q}}_{\left( B\right) ,\psi }}\right)$ and ${\overrightarrow{p}}_{\psi } = \left( {{\overrightarrow{p}}_{\left( 1\right) ,\psi },\ldots ,{\overrightarrow{p}}_{\left( B\right) ,\psi }}\right)$ . When $d = {d}_{\text{proj }}$ ,we simply define $\psi$ to be the identity mapping,in which case ${\overrightarrow{q}}_{\psi },{\overrightarrow{p}}_{\psi }$ are identical to $\overrightarrow{q},\overrightarrow{p}$ . To increase accuracy of (3) in approximating (1),we repeat the above process ${R}_{\text{reps }} \geq  1$ times independently,using different randomized partitions ${\varphi }_{1},\ldots ,{\varphi }_{{R}_{\text{reps }}}$ and projections ${\psi }_{1},\ldots ,{\psi }_{{R}_{\text{reps }}}$ . We denote the vectors resulting from $i$ -th repetition by ${\overrightarrow{q}}_{i,\psi },{\overrightarrow{p}}_{i,\psi }$ . Finally,we concatenate these ${R}_{\text{reps }}$ vectors together,so that our final FDEs are defined as ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)  = \left( {{\overrightarrow{q}}_{1,\psi },\ldots ,{\overrightarrow{q}}_{{R}_{\text{reps }},\psi }}\right)$ and ${\mathbf{F}}_{\text{doc }}\left( P\right)  = \left( {{\overrightarrow{p}}_{1,\psi },\ldots ,{\overrightarrow{p}}_{{R}_{\text{reps }},\psi }}\right)$ . Observe that a complete FDE mapping is specified by the three parameters $\left( {B,{d}_{\text{proj }},{R}_{\text{reps }}}\right)$ ,resulting in a final dimension of ${d}_{\mathrm{{FDE}}} = B \cdot  {d}_{\text{proj }} \cdot  {R}_{\text{reps }}$ .

请注意，向量 $\overrightarrow{q},\overrightarrow{p}$ 的最终维度为 ${dB}$。为了减少对 $d$ 的依赖，我们可以对每个块 ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) }$ 应用随机线性投影 $\psi  : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}_{\text{proj }}}$，其中 ${d}_{\text{proj }} < d$。具体而言，我们定义 $\psi \left( x\right)  = \left( {1/\sqrt{{d}_{\text{proj }}}}\right) {Sx}$，其中 $S \in  {\mathbb{R}}^{{d}_{\text{proj }} \times  d}$ 是一个元素均匀分布在 $\pm  1$ 上的随机矩阵。然后我们可以定义 ${\overrightarrow{q}}_{\left( k\right) ,\psi } = \psi \left( {\overrightarrow{q}}_{\left( k\right) }\right)$ 和 ${\overrightarrow{p}}_{\left( k\right) ,\psi } = \psi \left( {\overrightarrow{p}}_{\left( k\right) }\right)$，并将带有内投影的FDE（全微分方程，Full Differential Equation）定义为 ${\overrightarrow{q}}_{\psi } = \left( {{\overrightarrow{q}}_{\left( 1\right) ,\psi },\ldots ,{\overrightarrow{q}}_{\left( B\right) ,\psi }}\right)$ 和 ${\overrightarrow{p}}_{\psi } = \left( {{\overrightarrow{p}}_{\left( 1\right) ,\psi },\ldots ,{\overrightarrow{p}}_{\left( B\right) ,\psi }}\right)$。当 $d = {d}_{\text{proj }}$ 时，我们简单地将 $\psi$ 定义为恒等映射，在这种情况下，${\overrightarrow{q}}_{\psi },{\overrightarrow{p}}_{\psi }$ 与 $\overrightarrow{q},\overrightarrow{p}$ 相同。为了提高 (3) 式对 (1) 式的近似精度，我们独立地重复上述过程 ${R}_{\text{reps }} \geq  1$ 次，使用不同的随机划分 ${\varphi }_{1},\ldots ,{\varphi }_{{R}_{\text{reps }}}$ 和投影 ${\psi }_{1},\ldots ,{\psi }_{{R}_{\text{reps }}}$。我们用 ${\overrightarrow{q}}_{i,\psi },{\overrightarrow{p}}_{i,\psi }$ 表示第 $i$ 次重复得到的向量。最后，我们将这 ${R}_{\text{reps }}$ 个向量连接起来，这样我们最终的FDE就定义为 ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)  = \left( {{\overrightarrow{q}}_{1,\psi },\ldots ,{\overrightarrow{q}}_{{R}_{\text{reps }},\psi }}\right)$ 和 ${\mathbf{F}}_{\text{doc }}\left( P\right)  = \left( {{\overrightarrow{p}}_{1,\psi },\ldots ,{\overrightarrow{p}}_{{R}_{\text{reps }},\psi }}\right)$。可以看到，一个完整的FDE映射由三个参数 $\left( {B,{d}_{\text{proj }},{R}_{\text{reps }}}\right)$ 指定，最终维度为 ${d}_{\mathrm{{FDE}}} = B \cdot  {d}_{\text{proj }} \cdot  {R}_{\text{reps }}$。

<!-- Media -->

<!-- figureText: := Query Embeddings := Doc Embeddings -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_4.jpg?x=334&y=166&w=1092&h=398&r=0"/>

Figure 2: FDE Generation Process. Three SimHashes $\left( {{k}_{\text{sim }} = 3}\right)$ split space into six regions labelled $A - F$ (in high-dimensions $B = {2}^{{k}_{\text{sim }}}$ ,but $B = 6$ here since $d = 2$ ). ${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right)$ are shown as $B \times  d$ matrices, where the $k$ -th row is ${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) }$ . The actual FDEs are flattened versions of these matrices. Not shown: inner projections, repetitions, and fill_empty_clusters.

图2：FDE生成过程。三个SimHash函数$\left( {{k}_{\text{sim }} = 3}\right)$将空间划分为六个区域，标记为$A - F$（在高维空间中为$B = {2}^{{k}_{\text{sim }}}$，但这里是$B = 6$，因为$d = 2$）。${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right)$表示为$B \times  d$矩阵，其中第$k$行是${\overrightarrow{q}}_{\left( k\right) },{\overrightarrow{p}}_{\left( k\right) }$。实际的FDE是这些矩阵的扁平化版本。未展示：内部投影、重复项和填充空簇。

<!-- Media -->

Choice of Space Partition When choosing the partition function $\varphi$ ,the desired property is that points are more likely to collide (i.e. $\varphi \left( x\right)  = \varphi \left( y\right)$ ) the closer they are to each other. Such functions with this property exist, and are known as locality-sensitive hash functions (LSH) (see [20]). When the vectors are normalized, as they are for those produced by ColBERT-style models, SimHash [8] is the standard choice of LSH. Specifically,for any ${k}_{\text{sim }} \geq  1$ ,we sample random Gaussian vectors ${g}_{1},\ldots ,{g}_{{k}_{\operatorname{sim}}} \in  {\mathbb{R}}^{d}$ ,and set $\mathbf{\varphi }\left( x\right)  = \left( {\mathbf{1}\left( {\left\langle  {{g}_{1},x}\right\rangle   > 0}\right) ,\ldots ,\mathbf{1}\left( {\left\langle  {{g}_{{k}_{\operatorname{sim}}},x}\right\rangle   > 0}\right) }\right)$ ,where $\mathbf{1}\left( \cdot \right)  \in  \{ 0,1\}$ is the indicator function. Converting the bit-string to decimal, $\varphi \left( x\right)$ gives a mapping from ${\mathbb{R}}^{d}$ to $\left\lbrack  B\right\rbrack$ where $B = {2}^{{k}_{\text{sim }}}$ . In other words,SimHash partitions ${\mathbb{R}}^{d}$ by drawing ${k}_{\text{sim }}$ random half-spaces,and each of the the ${2}^{{k}_{\text{sim }}}$ clusters is formed by the ${k}_{\text{sim }}$ -wise intersection of each of these halfspaces or their complement. Another natural approach is to choose ${k}_{\text{CENTER }} \geq  1$ centers from the collection of all token embeddings ${ \cup  }_{i = 1}^{n}{P}_{i}$ ,either randomly or via $k$ -means,and set $\varphi \left( x\right)  \in  \left\lbrack  {k}_{\text{CENTER }}\right\rbrack$ to be the index of the center nearest to $x$ . We compare this method to SimHash in (§3.1).

空间划分的选择 在选择划分函数$\varphi$时，理想的特性是，彼此距离越近的点越有可能发生碰撞（即$\varphi \left( x\right)  = \varphi \left( y\right)$）。具有这种特性的函数是存在的，它们被称为局部敏感哈希函数（LSH）（参见[20]）。当向量被归一化时，就像ColBERT风格模型生成的向量那样，SimHash [8]是LSH的标准选择。具体来说，对于任何${k}_{\text{sim }} \geq  1$，我们采样随机高斯向量${g}_{1},\ldots ,{g}_{{k}_{\operatorname{sim}}} \in  {\mathbb{R}}^{d}$，并设置$\mathbf{\varphi }\left( x\right)  = \left( {\mathbf{1}\left( {\left\langle  {{g}_{1},x}\right\rangle   > 0}\right) ,\ldots ,\mathbf{1}\left( {\left\langle  {{g}_{{k}_{\operatorname{sim}}},x}\right\rangle   > 0}\right) }\right)$，其中$\mathbf{1}\left( \cdot \right)  \in  \{ 0,1\}$是指示函数。将位串转换为十进制，$\varphi \left( x\right)$给出了从${\mathbb{R}}^{d}$到$\left\lbrack  B\right\rbrack$的映射，其中$B = {2}^{{k}_{\text{sim }}}$。换句话说，SimHash通过绘制${k}_{\text{sim }}$个随机半空间对${\mathbb{R}}^{d}$进行划分，并且每个${2}^{{k}_{\text{sim }}}$簇是由这些半空间或其补集的${k}_{\text{sim }}$维交集形成的。另一种自然的方法是从所有词元嵌入${ \cup  }_{i = 1}^{n}{P}_{i}$的集合中选择${k}_{\text{CENTER }} \geq  1$个中心，可以随机选择，也可以通过$k$ -均值算法选择，并将$\varphi \left( x\right)  \in  \left\lbrack  {k}_{\text{CENTER }}\right\rbrack$设置为最接近$x$的中心的索引。我们在（§3.1）中比较了这种方法与SimHash。

Filling Empty Clusters. A key source of error in the FDE's approximation is when the nearest vector $p \in  P$ to a given query embedding $q \in  Q$ maps to a different cluster,namely $\varphi \left( p\right)  \neq  \varphi \left( q\right)  = k$ . This can be made less likely by decreasing $B$ ,at the cost of making it more likely for other ${p}^{\prime } \in  P$ to also map to the same cluster,moving the centroid ${\overrightarrow{p}}_{\left( k\right) }$ farther from $p$ . If we increase $B$ too much,it is possible that no $p \in  P$ collides with $\varphi \left( q\right)$ . To avoid this trade-off,we directly ensure that if no $p \in  P$ maps to a cluster $k$ ,then instead of setting ${\overrightarrow{p}}_{\left( k\right) } = 0$ we set ${\overrightarrow{p}}_{\left( k\right) }$ to the point $p$ that is closest to cluster $k$ . As a result,increasing $B$ will result in a more accurate estimator,as this results in smaller clusters. Formally,for any cluster $k$ with $P \cap  {\varphi }^{-1}\left( k\right)  = \varnothing$ ,if fill_empty_clusters is enabled,we set ${\overrightarrow{p}}_{\left( k\right) } = p$ where $p \in  P$ is the point for which $\varphi \left( p\right)$ has the fewest number of disagreeing bits with $k$ (both thought of as binary strings),with ties broken arbitrarily. We do not enable this for query FDEs,as doing so would result in a given $q \in  Q$ contributing to the dot product multiple times.

填充空簇。FDE（快速距离估计，Fast Distance Estimation）近似误差的一个关键来源是，给定查询嵌入 $q \in  Q$ 的最近向量 $p \in  P$ 映射到了不同的簇，即 $\varphi \left( p\right)  \neq  \varphi \left( q\right)  = k$ 。可以通过减小 $B$ 来降低这种可能性，但代价是其他 ${p}^{\prime } \in  P$ 也更有可能映射到同一个簇，从而使质心 ${\overrightarrow{p}}_{\left( k\right) }$ 离 $p$ 更远。如果我们过度增大 $B$ ，可能会出现没有 $p \in  P$ 与 $\varphi \left( q\right)$ 发生碰撞的情况。为了避免这种权衡，我们直接确保如果没有 $p \in  P$ 映射到簇 $k$ ，那么我们不设置 ${\overrightarrow{p}}_{\left( k\right) } = 0$ ，而是将 ${\overrightarrow{p}}_{\left( k\right) }$ 设置为最接近簇 $k$ 的点 $p$ 。因此，增大 $B$ 将得到更准确的估计器，因为这会使簇更小。形式上，对于任何满足 $P \cap  {\varphi }^{-1}\left( k\right)  = \varnothing$ 的簇 $k$ ，如果启用了填充空簇（fill_empty_clusters），我们设置 ${\overrightarrow{p}}_{\left( k\right) } = p$ ，其中 $p \in  P$ 是使得 $\varphi \left( p\right)$ 与 $k$ （都视为二进制字符串）的不同比特位数最少的点，若有平局则任意打破。我们不会为查询 FDE 启用此功能，因为这样做会导致给定的 $q \in  Q$ 多次对 dot 积产生贡献。

Final Projections. A natural approach to reducing the dimensionality is to apply a final projection ${\psi }^{\prime } : {\mathbb{R}}^{{d}_{\text{FDE }}} \rightarrow  {\mathbb{R}}^{{d}_{\text{final }}}$ (also implemented via multiplication by a random $\pm  1$ matrix) to the FDE’s, reducing the final dimensionality to any ${d}_{\text{final }} < {d}_{\mathrm{{FDE}}}$ . Experimentally,we find that final projections can provides small but non-trivial 1-2% recall boosts for a fixed dimension (see §C.2).

最终投影。降低维度的一种自然方法是对 FDE 应用最终投影 ${\psi }^{\prime } : {\mathbb{R}}^{{d}_{\text{FDE }}} \rightarrow  {\mathbb{R}}^{{d}_{\text{final }}}$ （也通过与随机 $\pm  1$ 矩阵相乘来实现），将最终维度降低到任意 ${d}_{\text{final }} < {d}_{\mathrm{{FDE}}}$ 。通过实验，我们发现对于固定维度，最终投影可以提供小但显著的 1 - 2% 的召回率提升（见§C.2）。

### 2.1 Theoretical Guarantees for FDEs

### 2.1 FDE 的理论保证

We now state our theoretical guarantees for our FDE construction. For clarity, we state our results in terms of normalized Chamfer similarity $\operatorname{NCHAMFER}\left( {Q,P}\right)  = \frac{1}{\left| Q\right| }\operatorname{CHAMFER}\left( {Q,P}\right)$ . This ensures NCHAMFER $\left( {Q,P}\right)  \in  \left\lbrack  {-1,1}\right\rbrack$ whenever the vectors in $Q,P$ are normalized. Note that this factor of $1/\left| Q\right|$ does not affect the relative scoring of documents for a fixed query. In what follows,we assume that all token embeddings are normalized (i.e. $\parallel q{\parallel }_{2} = \parallel p{\parallel }_{2} = 1$ for all $q \in  Q,p \in  P$ ). Note that ColBERT-style late interaction MV models indeed produce normalized token embeddings. We will always use the fill_empty_clusters method for document FDEs, but never for queries.

我们现在陈述我们对 FDE 构造的理论保证。为了清晰起见，我们用归一化的 Chamfer 相似度 $\operatorname{NCHAMFER}\left( {Q,P}\right)  = \frac{1}{\left| Q\right| }\operatorname{CHAMFER}\left( {Q,P}\right)$ 来陈述我们的结果。这确保了只要 $Q,P$ 中的向量是归一化的，就有 NCHAMFER $\left( {Q,P}\right)  \in  \left\lbrack  {-1,1}\right\rbrack$ 。请注意，这个 $1/\left| Q\right|$ 因子不会影响固定查询下文档的相对评分。在接下来的内容中，我们假设所有的词元嵌入都是归一化的（即对于所有的 $q \in  Q,p \in  P$ ，有 $\parallel q{\parallel }_{2} = \parallel p{\parallel }_{2} = 1$ ）。请注意，ColBERT 风格的后期交互 MV 模型确实会产生归一化的词元嵌入。我们将始终对文档 FDE 使用填充空簇（fill_empty_clusters）方法，但从不用于查询。

Our main result is that FDEs give $\varepsilon$ -additive approximations of the Chamfer similarity. The proof uses the properties of LSH (SimHash) to show that for each query point $q \in  Q$ ,the point $q$ gets mapped to a cluster $\varphi \left( q\right)$ that only contains points $p \in  P$ that are close to $q$ (within $\varepsilon$ of the closest point to $q$ ); the fact that at least one point collides with $q$ uses the fill_empty_partitions method.

我们的主要结果是，FDE 给出了 Chamfer 相似度的 $\varepsilon$ 加性近似。证明使用了 LSH（局部敏感哈希，Locality - Sensitive Hashing）（SimHash）的性质，以表明对于每个查询点 $q \in  Q$ ，点 $q$ 被映射到一个簇 $\varphi \left( q\right)$ ，该簇只包含接近 $q$ 的点 $p \in  P$ （在离 $q$ 最近的点的 $\varepsilon$ 范围内）；至少有一个点与 $q$ 发生碰撞这一事实使用了填充空分区（fill_empty_partitions）方法。

Theorem 2.1 (FDE Approximation). Fix any $\varepsilon ,\delta  > 0$ ,and sets $Q,P \subset  {\mathbb{R}}^{d}$ of unit vectors,and let $m = \left| Q\right|  + \left| P\right|$ . Then setting ${k}_{sim} = O\left( \frac{\log \left( {m{\delta }^{-1}}\right) }{\varepsilon }\right) ,{d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( \frac{m}{\varepsilon \delta }\right) }\right) ,{R}_{reps} = 1$ ,so that ${d}_{FDE} = {\left( m/\delta \right) }^{O\left( {1/\varepsilon }\right) }$ ,then in expectation and with probability at least $1 - \delta$ we have

定理2.1（FDE近似）。固定任意$\varepsilon ,\delta  > 0$，以及单位向量集合$Q,P \subset  {\mathbb{R}}^{d}$，并设$m = \left| Q\right|  + \left| P\right|$。然后设${k}_{sim} = O\left( \frac{\log \left( {m{\delta }^{-1}}\right) }{\varepsilon }\right) ,{d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( \frac{m}{\varepsilon \delta }\right) }\right) ,{R}_{reps} = 1$，使得${d}_{FDE} = {\left( m/\delta \right) }^{O\left( {1/\varepsilon }\right) }$，那么在期望意义下且以至少$1 - \delta$的概率，我们有

$$
\operatorname{NCHAMFER}\left( {Q,P}\right)  - \varepsilon  \leq  \frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( P\right) }\right\rangle   \leq  \operatorname{NCHAMFER}\left( {Q,P}\right)  + \varepsilon 
$$

Finally,we show that our FDE’s give an $\varepsilon$ -approximate solution to Chamfer similarity search,using FDE dimension that depends only logarithmically on the size of the dataset $n$ . Using the fact that our query FDEs are sparse (Lemma A.1),one can run exact MIPS over the FDEs in time $\widetilde{O}\left( {\left| Q\right|  \cdot  n}\right)$ , improving on the brute-force runtime of $O\left( {\left| Q\right| \mathop{\max }\limits_{i}\left| {P}_{i}\right| n}\right)$ for Chamfer similarity search.

最后，我们证明我们的FDE（快速数据嵌入，Fast Data Embedding）为倒角相似度搜索（Chamfer similarity search）提供了一个$\varepsilon$ - 近似解，所使用的FDE维度仅与数据集$n$的大小呈对数关系。利用我们的查询FDE是稀疏的这一事实（引理A.1），可以在时间$\widetilde{O}\left( {\left| Q\right|  \cdot  n}\right)$内对FDE进行精确的最大内积搜索（MIPS，Maximum Inner Product Search），这比倒角相似度搜索的暴力运行时间$O\left( {\left| Q\right| \mathop{\max }\limits_{i}\left| {P}_{i}\right| n}\right)$有所改进。

Theorem 2.2. Fix any $\varepsilon  > 0$ ,query $Q$ ,and dataset $P = \left\{  {{P}_{1},\ldots ,{P}_{n}}\right\}$ ,where $Q \subset  {\mathbb{R}}^{d}$ and each ${P}_{i} \subset  {\mathbb{R}}^{d}$ is a set of unit vectors. Let $m = \left| Q\right|  + \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left| {P}_{i}\right|$ . Let ${k}_{sim} = O\left( \frac{\log m}{\varepsilon }\right)$ , ${d}_{\text{proj }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\varepsilon }\right) }\right)$ and ${R}_{\text{reps }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log n}\right)$ so that ${d}_{FDE} = {m}^{O\left( {1/\varepsilon }\right) } \cdot  \log n$ . Then if ${i}^{ * } = \arg \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( {P}_{i}\right) }\right\rangle$ ,with high probability (i.e. $1 - 1/\operatorname{poly}\left( n\right)$ ) we have:

定理2.2。固定任意$\varepsilon  > 0$、查询$Q$和数据集$P = \left\{  {{P}_{1},\ldots ,{P}_{n}}\right\}$，其中$Q \subset  {\mathbb{R}}^{d}$且每个${P}_{i} \subset  {\mathbb{R}}^{d}$是一个单位向量集合。设$m = \left| Q\right|  + \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left| {P}_{i}\right|$。设${k}_{sim} = O\left( \frac{\log m}{\varepsilon }\right)$、${d}_{\text{proj }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\varepsilon }\right) }\right)$和${R}_{\text{reps }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log n}\right)$，使得${d}_{FDE} = {m}^{O\left( {1/\varepsilon }\right) } \cdot  \log n$。那么如果${i}^{ * } = \arg \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( {P}_{i}\right) }\right\rangle$，以高概率（即$1 - 1/\operatorname{poly}\left( n\right)$）我们有：

$$
\operatorname{NCHAMFER}\left( {Q,{P}_{{i}^{ * }}}\right)  \geq  \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\operatorname{NCHAMFER}\left( {Q,{P}_{i}}\right)  - \varepsilon 
$$

Given the query $Q$ ,the document ${P}^{ * }$ can be recovered in time $O\left( {\left| Q\right| \max \{ d,n\} \frac{1}{{\varepsilon }^{4}}\log \left( \frac{m}{\varepsilon }\right) \log n}\right)$ .

给定查询$Q$，文档${P}^{ * }$可以在时间$O\left( {\left| Q\right| \max \{ d,n\} \frac{1}{{\varepsilon }^{4}}\log \left( \frac{m}{\varepsilon }\right) \log n}\right)$内被恢复。

## 3 Evaluation

## 3 评估

In this section, we evaluate our FDEs as a method for MV retrieval. First, we evaluate the FDEs themselves (offline) as a proxy for Chamfer similarity (§3.1). In (§3.2), we discuss the implementation of MUVERA, as well as several optimizations made in the search. Then we evaluate the latency of MUVERA compared to PLAID, and study the effects of the aforementioned optimizations.

在本节中，我们评估将我们的FDE作为多向量（MV，Multi - Vector）检索方法的效果。首先，我们（离线地）评估FDE本身作为倒角相似度的代理（§3.1）。在（§3.2）中，我们讨论MUVERA的实现，以及在搜索中进行的若干优化。然后我们评估MUVERA与PLAID相比的延迟，并研究上述优化的效果。

Datasets. Our evaluation includes results from six of the well-studied BEIR [46] information retrieval datasets: MS MARCO [40] (CC BY-SA 4.0), HotpotQA (CC BY-SA 4.0) [53], NQ (Apache-2.0) [31], Quora (Apache-2.0) [46], SciDocs (CC BY 4.0) [11], and ArguAna (Apache-2.0) [47]. These datasets were selected for varying corpus size (8K-8.8M) and average number of document tokens (18-165); see (§B) for further dataset statistics. Following [43], we use the development set for our experiments on MS MARCO, and use the test set on the other datasets.

数据集。我们的评估包含了六个经过充分研究的BEIR [46]信息检索数据集的结果：MS MARCO [40]（知识共享署名-相同方式共享4.0国际许可协议）、HotpotQA（知识共享署名-相同方式共享4.0国际许可协议） [53]、NQ（Apache-2.0许可协议） [31]、Quora（Apache-2.0许可协议） [46]、SciDocs（知识共享署名4.0国际许可协议） [11]和ArguAna（Apache-2.0许可协议） [47]。选择这些数据集是因为它们的语料库大小（8K - 880万）和文档标记的平均数量（18 - 165）各不相同；更多数据集统计信息见（§B）。遵循文献[43]，我们在MS MARCO上使用开发集进行实验，在其他数据集上使用测试集。

MV Model, MV Embedding Sizes, and FDE Dimensionality. We compute our FDEs on the MV embeddings produced by the ColBERTv2 model [44] (MIT License), which have a dimension of $d = {128}$ and a fixed number $\left| Q\right|  = {32}$ of embeddings per query. The number of document embeddings is variable, ranging from an average of 18.3 on Quora to 165 on Scidocs. This results in 2,300-21,000 floats per document on average (e.g. 10,087 for MS MARCO). Thus, when constructing our FDEs we consider a comparable range of dimensions ${d}_{\mathrm{{FDE}}}$ between 1,000-20,000. Furthermore, using product quantization,we show in (§3.2) that the FDEs can be significantly compressed by ${32} \times$ with minimal quality loss, further increasing the practicality of FDEs.

MV模型、MV嵌入大小和FDE维度。我们在ColBERTv2模型 [44]（麻省理工学院许可协议）生成的MV嵌入上计算我们的FDE，其维度为$d = {128}$，每个查询的嵌入数量固定为$\left| Q\right|  = {32}$。文档嵌入的数量是可变的，从Quora上的平均18.3个到Scidocs上的165个不等。这导致每个文档平均有2300 - 21000个浮点数（例如，MS MARCO为10087个）。因此，在构建我们的FDE时，我们考虑了1000 - 20000之间的可比维度范围${d}_{\mathrm{{FDE}}}$。此外，使用乘积量化，我们在（§3.2）中表明，FDE可以通过${32} \times$显著压缩，同时质量损失最小，进一步提高了FDE的实用性。

### 3.1 Offline Evaluation of FDE Quality

### 3.1 FDE质量的离线评估

We evaluate the quality of our FDEs as a proxy for the Chamfer similarity, without any re-ranking and using exact (offline) search. We first demonstrate that FDE recall quality improves dependably as the dimension ${d}_{\mathrm{{FDE}}}$ increases,making our method relatively easy to tune. We then show that FDEs are a more effective method of retrieval than the SV heuristic. Specifically, the FDE method achieves Recall $@N$ exceeding the Recall $@2 - 4\mathrm{\;N}$ of the SV heuristic,while in principle scanning a similar number of floats in the search. This suggests that the success of the SV heuristic is largely due to the significant effort put towards optimizing it (as supported by [37]), and similar effort for FDEs may result in even bigger efficiency gains. Additional plots can be found in ( $§\mathrm{C}$ ). All recall curves use a single FDE instantiation,since in (§C.1) we show the variance of FDE recall is negligible.

我们将FDE的质量作为倒角相似度的代理进行评估，不进行任何重新排序，并使用精确（离线）搜索。我们首先证明，随着维度${d}_{\mathrm{{FDE}}}$的增加，FDE召回质量可靠地提高，这使得我们的方法相对容易调整。然后我们表明，FDE是一种比SV启发式更有效的检索方法。具体来说，FDE方法实现的召回率$@N$超过了SV启发式的召回率$@2 - 4\mathrm{\;N}$，而在搜索中原则上扫描的浮点数数量相似。这表明SV启发式的成功在很大程度上归功于为优化它所付出的巨大努力（如文献[37]所支持的），而为FDE付出类似的努力可能会带来更大的效率提升。更多图表见（ $§\mathrm{C}$ ）。所有召回曲线都使用单个FDE实例，因为在（§C.1）中我们表明FDE召回的方差可以忽略不计。

FDE Quality vs. Dimensionality. We study how the retrieval quality of FDE's improves as a function of the dimension ${d}_{\mathrm{{FDE}}}$ . We perform a grid search over FDE parameters ${R}_{\text{reps }} \in$ $\{ 1,5,{10},{15},{20}\} ,{k}_{\text{sim }} \in  \{ 2,3,4,5,6\} ,{d}_{\text{proj }} \in  \{ 8,{16},{32},{64}\}$ ,and compute recall on MS MARCO (Figure 3). We find that Pareto optimal parameters are generally achieved by larger ${R}_{\text{reps }}$ , with ${k}_{\text{sim }},{d}_{\text{proj }}$ playing a lesser role in improving quality. Specifically, $\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)  \in$

FDE质量与维度的关系。我们研究了FDE的检索质量如何随维度${d}_{\mathrm{{FDE}}}$的变化而提高。我们对FDE参数${R}_{\text{reps }} \in$ $\{ 1,5,{10},{15},{20}\} ,{k}_{\text{sim }} \in  \{ 2,3,4,5,6\} ,{d}_{\text{proj }} \in  \{ 8,{16},{32},{64}\}$进行网格搜索，并在MS MARCO上计算召回率（图3）。我们发现，帕累托最优参数通常通过更大的${R}_{\text{reps }}$实现，而${k}_{\text{sim }},{d}_{\text{proj }}$在提高质量方面的作用较小。具体来说，$\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)  \in$

<!-- Media -->

<!-- figureText: Exact Chamfer R@25 FDE SimHash Partitioning R@1000 Exact Chamfer R@250 15000 20000 Dimensior 20000 -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_6.jpg?x=312&y=210&w=1176&h=316&r=0"/>

Figure 3: FDE recall vs dimension for varying FDE parameters on MS MARCO. Plots show FDE Recall@100,1k,10k left to right. Recalls@N for exact Chamfer scoring is shown by dotted lines.

图3：MS MARCO上不同FDE参数下FDE召回率与维度的关系。图表从左到右显示了FDE在召回率@100、1k、10k的情况。精确倒角评分的召回率@N用虚线表示。

<!-- figureText: 1.0 0.9 Recall@N Recall@N 0.9 0.8 100 Recall@N Recall@N -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_6.jpg?x=311&y=635&w=1173&h=259&r=0"/>

Figure 4: Comparison of FDE recall versus brute-force search over Chamfer similarity.

图4：FDE召回率与基于倒角相似度的暴力搜索的比较。

<!-- Media -->

$\{ \left( {{20},3,8}\right) ,\left( {{20},4,8}\right) \left( {{20},5,8}\right) ,\left( {{20},5,{16}}\right) \}$ were all Pareto optimal for their respective dimensions (namely ${R}_{\text{reps }} \cdot  {2}^{{k}_{\text{sim }}} \cdot  {d}_{\text{proj }}$ ). While there are small variations depending on the parameter choice,the FDE quality is tightly linked to dimensionality; increase in dimensionality will generally result in quality gains. We also evaluate using $k$ -means as a method of partitioning instead of SimHash. Specifically,we cluster the document embeddings with $k$ -means and set $\varphi \left( x\right)$ to be the index of the nearest centroid to $x$ . We perform a grid search over the same parameters (but with $k \in  \{ 4,8,{16},{32},{64}\}$ to match $\left. {B = {2}^{{k}_{\text{sim }}}}\right)$ . We find that $k$ -means partitioning offers no quality gains on the Pareto Frontier over SimHash,and is often worse. Moreover,FDE construction with $k$ -means is no longer data oblivious. Thus, SimHash is chosen as the preferred method for partitioning for the remainder of our experiments.

$\{ \left( {{20},3,8}\right) ,\left( {{20},4,8}\right) \left( {{20},5,8}\right) ,\left( {{20},5,{16}}\right) \}$ 在各自的维度上都是帕累托最优的（即 ${R}_{\text{reps }} \cdot  {2}^{{k}_{\text{sim }}} \cdot  {d}_{\text{proj }}$ ）。虽然根据参数选择会有一些小的变化，但FDE（特征分布嵌入，Feature Distribution Embedding）质量与维度紧密相关；维度的增加通常会带来质量的提升。我们还评估了使用 $k$ -均值作为划分方法，而非SimHash（相似哈希）。具体来说，我们使用 $k$ -均值对文档嵌入进行聚类，并将 $\varphi \left( x\right)$ 设置为距离 $x$ 最近的质心的索引。我们对相同的参数进行网格搜索（但 $k \in  \{ 4,8,{16},{32},{64}\}$ 要与 $\left. {B = {2}^{{k}_{\text{sim }}}}\right)$ 匹配）。我们发现，与SimHash相比， $k$ -均值划分在帕累托前沿上没有带来质量提升，而且通常更差。此外，使用 $k$ -均值构建FDE不再是数据无关的。因此，在我们后续的实验中，SimHash被选为首选的划分方法。

In Figure 4, we evaluate the FDE retrieval quality with respect to the Chamfer similarity (instead of labelled ground truth data). We compute $1\mathrm{{Recall}}@N$ ,which is the fraction of queries for which the Chamfer 1-nearest neighbor is among the top- $N$ most similar in FDE dot product. We choose FDE parameters which are Pareto optimal for the dimension from the above grid search. We find that FDE's with fewer dimensions that the original MV representations achieve significantly good recall across multiple BEIR retrieval datasets. For instance,on MS MARCO (where $d \cdot  {m}_{avg} \approx  {10}\mathrm{K}$ ) we achieve ${95}\%$ recall while retrieving only 75 candidates using ${d}_{\mathrm{{FDE}}} = {5120}$ .

在图4中，我们根据倒角相似度（而非标记的真实数据）评估FDE的检索质量。我们计算 $1\mathrm{{Recall}}@N$ ，即倒角1 - 最近邻在FDE点积中最相似的前 $N$ 个结果中的查询比例。我们从上述网格搜索中选择在该维度上帕累托最优的FDE参数。我们发现，与原始的多向量（MV，Multi - Vector）表示相比，维度更少的FDE在多个BEIR（基准信息检索，Benchmarking Information Retrieval）检索数据集上实现了显著良好的召回率。例如，在MS MARCO（其中 $d \cdot  {m}_{avg} \approx  {10}\mathrm{K}$ ）上，使用 ${d}_{\mathrm{{FDE}}} = {5120}$ 仅检索75个候选结果时，我们实现了 ${95}\%$ 的召回率。

Single Vector Heuristic vs. FDE retrieval. We compare the quality of FDEs as a proxy for retrieval against the previously described SV heuristic, which is the method underpinning PLAID. Recall that in this method,for each of the $i = 1,\ldots ,{32}$ query vectors ${q}_{i}$ we compute the $k$ nearest neighbors ${p}_{1,i},\ldots ,{p}_{k,i}$ from the set ${ \cup  }_{i}{P}_{i}$ of all documents token embeddings. To compute Recall $@N$ ,we create an ordered list ${\ell }_{1,1},\ldots ,{\ell }_{1,{32}},{\ell }_{2,1},\ldots$ ,where ${\ell }_{i,j}$ is the document ID containing ${p}_{i,j}$ ,consisting of the 1-nearest neighbors of the queries, then the 2-nearest neighbors, and so on. When re-ranking, one firsts removes duplicate document IDs from this list. Since duplicates cannot be detected while performing the initial 32 SV MIPS queries, the SV heuristic needs to over-retrieve to reach a desired number of unique candidates. Thus, we note that the true recall curve of implementations of the SV heuristic (e.g. PLAID) is somewhere between the case of no deduplication and full deduplication; we compare to both in Figure 5.

单向量启发式方法与FDE检索的比较。我们将FDE作为检索代理的质量与之前描述的SV（单向量，Single Vector）启发式方法进行比较，该方法是PLAID的基础。回想一下，在这种方法中，对于 $i = 1,\ldots ,{32}$ 个查询向量 ${q}_{i}$ 中的每一个，我们从所有文档词嵌入的集合 ${ \cup  }_{i}{P}_{i}$ 中计算 $k$ 个最近邻 ${p}_{1,i},\ldots ,{p}_{k,i}$ 。为了计算召回率 $@N$ ，我们创建一个有序列表 ${\ell }_{1,1},\ldots ,{\ell }_{1,{32}},{\ell }_{2,1},\ldots$ ，其中 ${\ell }_{i,j}$ 是包含 ${p}_{i,j}$ 的文档ID，该列表由查询的1 - 最近邻、2 - 最近邻等组成。在重新排序时，首先要从该列表中去除重复的文档ID。由于在执行初始的32个SV最大内积搜索（MIPS，Maximum Inner Product Search）查询时无法检测到重复项，SV启发式方法需要过度检索才能达到所需数量的唯一候选结果。因此，我们注意到SV启发式方法（如PLAID）实现的真实召回曲线介于不进行去重和完全去重这两种情况之间；我们在图5中对这两种情况都进行了比较。

To compare the cost of the SV heuristic to running MIPS over the FDEs, we consider the total number of floats scanned by both using a brute force search. The FDE method must scan $n \cdot  {d}_{\mathrm{{FDE}}}$ floats to compute the $k$ -nearest neighbors. For the SV heuristic,one runs 32 brute force scans over $n \cdot  {m}_{avg}$ vectors in 128 dimensions,where ${m}_{avg}$ is the average number embeddings per document (see $§\mathrm{B}$ for values of ${m}_{avg}$ ). For MS MARCO,where ${m}_{avg} = {78.8}$ ,the SV heuristic searches through ${32} \cdot  {128} \cdot  {78.8} \cdot  n$ floats. This allows for an FDE dimension of ${d}_{\mathrm{{FDE}}} = {322},{764}$ to have comparable cost! We can extend this comparison to fast approximate search - suppose that approximate MIPS over $n$ vectors can be accomplished in sublinear ${n}^{\varepsilon }$ time,for some $\varepsilon  \in  \left( {0,1}\right)$ . Then even in the unrealistic case of $\varepsilon  = 0$ ,we can still afford an FDE dimension of ${d}_{\mathrm{{FDE}}} = {32} \cdot  {128} = {4096}$ .

为了将SV启发式算法的成本与在FDE（特征分布嵌入）上运行MIPS（最大内积搜索）的成本进行比较，我们考虑使用暴力搜索时两者扫描的浮点数总数。FDE方法必须扫描$n \cdot  {d}_{\mathrm{{FDE}}}$个浮点数来计算$k$近邻。对于SV启发式算法，需要在128维的$n \cdot  {m}_{avg}$个向量上进行32次暴力扫描，其中${m}_{avg}$是每个文档的平均嵌入数（${m}_{avg}$的值见$§\mathrm{B}$）。对于MS MARCO数据集，当${m}_{avg} = {78.8}$时，SV启发式算法要搜索${32} \cdot  {128} \cdot  {78.8} \cdot  n$个浮点数。这使得维度为${d}_{\mathrm{{FDE}}} = {322},{764}$的FDE具有相当的成本！我们可以将这种比较扩展到快速近似搜索——假设在$n$个向量上进行近似MIPS可以在次线性${n}^{\varepsilon }$时间内完成，其中$\varepsilon  \in  \left( {0,1}\right)$为某个值。那么，即使在不切实际的$\varepsilon  = 0$情况下，我们仍然可以承受维度为${d}_{\mathrm{{FDE}}} = {32} \cdot  {128} = {4096}$的FDE。

<!-- Media -->

<!-- figureText: 1.0 1.00 1.0 0.8 0.7 SV w/ Dedup 0.5 250 750 1000 0.8 0.95 0.90 0.85 0.5 0.80 1000 250 1000 100 -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_7.jpg?x=310&y=203&w=1175&h=260&r=0"/>

Figure 5: FDE retrieval vs SV Heuristic, both with and without document id deduplication.

图5：FDE检索与SV启发式算法对比，包括有无文档ID去重的情况。

<!-- Media -->

The results can be found in Figure 5. We build FDEs once for each dimension,using ${R}_{\text{reps }} =$ ${40},{k}_{\text{sim }} = 6,{d}_{\text{proj }} = d = {128}$ ,and then applying a final projection to reduce to the target dimension (see C.2 for experiments on the impact of final projections). On MS MARCO, even the 4096- dimensional FDEs match the recall of the (deduplicated) SV heuristic while retrieving 1.75-3.75 X fewer candidates (our Recall@N matches the Recall@1.75-3.75N of the SV heuristic), and 10.5-15 $\times$ fewer than to the non-deduplicated SV heuristic. For our 10240-dimension FDEs, these numbers are ${2.6} - 5 \times$ and ${20} - {22.5} \times$ fewer,respectively. For instance,we achieve ${80}\%$ recall with 60 candidates when ${d}_{\mathrm{{FDE}}} = {10240}$ and 80 candidates when ${d}_{\mathrm{{FDE}}} = {4096}$ ,but the SV heuristic requires 300 and 1200 candidates (for dedup and non-dedup respectively). See Table 1 for further comparisons.

结果如图5所示。我们针对每个维度构建一次FDE，使用${R}_{\text{reps }} =$ ${40},{k}_{\text{sim }} = 6,{d}_{\text{proj }} = d = {128}$，然后进行最终投影以降至目标维度（最终投影的影响实验见C.2）。在MS MARCO数据集上，即使是4096维的FDE在召回率上也能与（去重后的）SV启发式算法相匹配，同时检索到的候选数量减少了1.75 - 3.75倍（我们的Recall@N与SV启发式算法的Recall@1.75 - 3.75N相匹配），并且比未去重的SV启发式算法少检索10.5 - 15 $\times$个候选。对于我们的10240维FDE，这些数字分别减少了${2.6} - 5 \times$和${20} - {22.5} \times$。例如，当${d}_{\mathrm{{FDE}}} = {10240}$时，我们用60个候选就能达到${80}\%$的召回率，当${d}_{\mathrm{{FDE}}} = {4096}$时用80个候选，而SV启发式算法（去重和未去重情况下）分别需要300和1200个候选。更多比较见表格1。

Variance. Note that although the FDE generation is a randomized process, we show in (§C.1) that the variance of the FDE Recall is essentially negligible; for instance, the standard deviation Recall@1000 is at most 0.08-0.16% for FDEs with 2-10k dimensions.

方差。请注意，尽管FDE生成是一个随机过程，但我们在（§C.1）中表明，FDE召回率的方差实际上可以忽略不计；例如，对于维度在2 - 10k的FDE，Recall@1000的标准差最多为0.08 - 0.16%。

### 3.2 Online Implementation and End-to-End Evaluation

### 3.2 在线实现与端到端评估

We implemented MUVERA, an FDE generation and end-to-end retrieval engine in C++. We discussed FDE generation and various optimizations and their tradeoffs in (§3.1). Next, we discuss how we perform retrieval over the FDEs, and additional optimizations.

我们用C++实现了MUVERA，这是一个FDE生成和端到端检索引擎。我们在（§3.1）中讨论了FDE生成、各种优化及其权衡。接下来，我们将讨论如何在FDE上进行检索以及额外的优化措施。

Single-Vector MIPS Retrieval using DiskANN Our single-vector retrieval engine uses a scalable implementation [38] of DiskANN [25] (MIT License), a state-of-the-art graph-based ANNS algorithm. We build DiskANN indices by using the uncompressed document FDEs with a maximum degree of 200 and a build beam-width of 600 . Our retrieval works by querying the DiskANN index using beam search with beam-width $W$ ,and subsequently reranking the retrieved candidates with Chamfer similarity. The only tuning knob in our system is $W$ ; increasing $W$ increases the number of candidates retrieved by MUVERA, which improves the recall.

使用DiskANN进行单向量MIPS检索 我们的单向量检索引擎使用了DiskANN [25]（MIT许可）的可扩展实现[38]，DiskANN是一种先进的基于图的近似最近邻搜索（ANNS）算法。我们使用未压缩的文档FDE构建DiskANN索引，最大度为200，构建波束宽度为600。我们的检索过程是使用波束宽度为$W$的波束搜索查询DiskANN索引，然后使用倒角相似度对检索到的候选进行重新排序。我们系统中唯一的调优参数是$W$；增大$W$会增加MUVERA检索到的候选数量，从而提高召回率。

Ball Carving. To improve re-ranking speed, we reduce the number of query embeddings by clustering them via a ball carving method and replacing the embeddings in each cluster with their sum. This speeds up reranking without decreasing recall; we provide further details in (§C.3).

球分割。为了提高重新排序的速度，我们通过球分割方法对查询嵌入进行聚类，并将每个聚类中的嵌入替换为它们的和，从而减少查询嵌入的数量。这在不降低召回率的情况下加快了重新排序的速度；我们在（§C.3）中提供了更多细节。

Product Quantization (PQ). To further improve the memory usage of MUVERA, we use a textbook vector compression technique called product quantization (PQ) with asymmetric querying [19, 26] on the FDEs. We refer to product quantization with $C$ centers per group of $G$ dimensions as PQ- $C - G$ . For example, PQ-256-8, which we find to provide the best tradeoff between quality and compression in our experiments, compresses every consecutive set of 8 dimensions to one of 256 centers. Thus PQ-256-8 provides ${32} \times$ compression over storing each dimension using a single float,since each block of 8 floats is represented by a single byte. See (§C.4) for further experiments and details on PQ.

乘积量化（Product Quantization，PQ）。为了进一步提高MUVERA的内存使用效率，我们在FDEs（特征描述符）上采用了一种经典的向量压缩技术，即带有不对称查询的乘积量化（PQ）[19, 26]。我们将每组$G$维使用$C$个质心的乘积量化称为PQ - $C - G$。例如，PQ - 256 - 8在我们的实验中被证明能在质量和压缩率之间提供最佳平衡，它将每连续的8维压缩为256个质心之一。因此，与使用单个浮点数存储每一维相比，PQ - 256 - 8实现了${32} \times$的压缩率，因为每8个浮点数块由一个字节表示。有关PQ的更多实验和详细信息，请参阅（§C.4）。

Experimental Setup We run our online experiments on an Intel Sapphire Rapids machine on Google Cloud (c3-standard-176). The machine supports up to 176 hyper-threads. We run latency experiments using a single thread, and run our QPS experiments on all 176 threads.

实验设置 我们在谷歌云（c3 - standard - 176）的英特尔至强可扩展处理器Sapphire Rapids机器上进行在线实验。该机器最多支持176个超线程。我们使用单线程进行延迟实验，并在所有176个线程上进行每秒查询数（QPS）实验。

QPS vs. Recall A useful metric for retrieval is the number of queries per second (QPS) a system can serve at a given recall; evaluating the QPS of a system tries to fully utilize the system resources (e.g., the bandwidth of multiple memory channels and caches), and deployments where machines serve many queries simultaneously. Figure 6 shows the QPS vs. Recall@100 for MUVERA on a subset of the BEIR datasets, using different PQ schemes over the FDEs. We show results for additional datasets, as well as Recall@1000, in the Appendix. Using PQ-256-8 not only reduces the space usage of the FDEs by ${32} \times$ ,but also improves the QPS at the same query beamwidth by up to ${20} \times$ , while incurring a minimal loss in end-to-end recall. Our method has a relatively small dependence on the dataset size, which is consistent with prior studies on graph-based ANNS data structures, since the number of distance comparisons made during beam search grows roughly logarithmically with increasing dataset size [25, 38]. We tried to include QPS numbers for PLAID [43], but unfortunately their implementation does not support running multiple queries in parallel, and is optimized for measuring latency.

QPS与召回率 检索的一个有用指标是系统在给定召回率下每秒可以处理的查询数（QPS）；评估系统的QPS旨在充分利用系统资源（例如，多个内存通道和缓存的带宽），适用于机器同时处理大量查询的部署场景。图6展示了在BEIR数据集的一个子集上，MUVERA使用不同的PQ方案对FDEs进行处理时的QPS与Recall@100的关系。我们在附录中展示了其他数据集以及Recall@1000的结果。使用PQ - 256 - 8不仅将FDEs的空间使用减少了${32} \times$，还在相同查询束宽下将QPS提高了多达${20} \times$，同时在端到端召回率上的损失极小。我们的方法对数据集大小的依赖性相对较小，这与之前基于图的近似最近邻搜索（ANNS）数据结构的研究结果一致，因为在束搜索过程中进行的距离比较次数大致随数据集大小的增加呈对数增长[25, 38]。我们试图纳入PLAID [43]的QPS数据，但遗憾的是，他们的实现不支持并行运行多个查询，并且是为测量延迟而优化的。

<!-- Media -->

<!-- figureText: Uncompressed PQ-256-5 PQ-256-8 0.95 0.98 -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_8.jpg?x=311&y=179&w=1166&h=283&r=0"/>

Figure 6: Plots showing the QPS vs. Recall @100 for MUVERA on a subset of the BEIR datasets. The different curves are obtained by using different PQ methods on 10240-dimensional FDEs.

图6：展示了在BEIR数据集的一个子集上，MUVERA的QPS与Recall@100的关系图。不同的曲线是通过在10240维FDEs上使用不同的PQ方法得到的。

<!-- figureText: 0.60 PLAID PLAID -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_8.jpg?x=441&y=560&w=925&h=764&r=0"/>

Figure 7: Bar plots showing the latency and Recall@k of MUVERA vs PLAID on a subset of the BEIR datasets. The x-tick labels are formatted as dataset- $k$ ,i.e.,optimizing for Recall $@k$ on the given dataset.

图7：柱状图展示了在BEIR数据集的一个子集上，MUVERA与PLAID的延迟和Recall@k的对比。x轴刻度标签格式为数据集 - $k$，即在给定数据集上针对Recall $@k$进行优化。

<!-- Media -->

Latency and Recall Results vs. PLAID [43] We evaluated MUVERA and PLAID [43] on the 6 datasets from the BEIR benchmark described earlier in (§3); Figure 7 shows that MUVERA achieves essentially equivalent Recall $@k$ as PLAID (within 0.4%) on MS MARCO,while obtaining up to ${1.56} \times$ higher recall on other datasets (e.g. HotpotQA). We ran PLAID using the recommended settings for their system, which reproduced their recall results for MS MARCO. Compared with PLAID, on average over all 6 datasets and $k \in  \{ {100},{1000}\}$ ,MUVERA achieves ${10}\%$ higher Recall $@k$ (up to ${56}\%$ higher),and ${90}\%$ lower latency (up to ${5.7} \times$ lower).

延迟和召回率结果与PLAID [43]的对比 我们在前面（§3）描述的BEIR基准的6个数据集上评估了MUVERA和PLAID [43]；图7显示，在MS MARCO数据集上，MUVERA实现了与PLAID基本相当的Recall $@k$（误差在0.4%以内），而在其他数据集（如HotpotQA）上的召回率最高可提高${1.56} \times$。我们按照PLAID系统的推荐设置运行，重现了他们在MS MARCO上的召回率结果。与PLAID相比，在所有6个数据集和$k \in  \{ {100},{1000}\}$上平均而言，MUVERA的Recall $@k$提高了${10}\%$（最高提高${56}\%$），延迟降低了${90}\%$（最高降低${5.7} \times$）。

Importantly, MUVERA has consistently high recall and low latency across all of the datasets that we measure, and our method does not require costly parameter tuning to achieve this-all of our results use the same 10240-dimensional FDEs that are compressed using PQ with PQ-256-8; the only tuning in our system was to pick the first query beam-width over the $k$ that we rerank to that obtained recall matching that of PLAID. As Figure 7 shows, in cases like NQ and HotpotQA, MUVERA obtains much higher recall while obtaining lower latency. Given these results, we believe a distinguishing feature of MUVERA compared to prior multi-vector retrieval systems is that it achieves consistently high recall and low latency across a wide variety of datasets with minimal tuning effort.

重要的是，MUVERA在我们测量的所有数据集上都始终保持高召回率和低延迟，并且我们的方法无需进行昂贵的参数调整即可实现这一点——我们所有的结果都使用相同的10240维FDEs，并使用PQ - 256 - 8进行压缩；我们系统中唯一的调整是在我们重新排序的$k$中选择第一个查询束宽，以获得与PLAID相当的召回率。如图7所示，在NQ和HotpotQA等情况下，MUVERA在获得更低延迟的同时，召回率要高得多。鉴于这些结果，我们认为与之前的多向量检索系统相比，MUVERA的一个显著特点是，它只需进行极少的调整，就能在各种数据集上始终保持高召回率和低延迟。

## 4 Conclusion

## 4 结论

In this paper, we presented MUVERA: a principled and practical MV retrieval algorithm which reduces MV similarity to SV similarity by constructing Fixed Dimensional Encoding (FDEs) of a MV representation. We prove that FDE dot products give high-quality approximations to Chamfer similarity (§2.1). Experimentally, we show that FDEs are a much more effective proxy for MV similarity,since they require retrieving $2 - 4 \times$ fewer candidates to achieve the same recall as the SV Heuristic (§3.1). We complement these results with an end-to-end evaluation of MUVERA, showing that it achieves an average of ${10}\%$ improved recall with ${90}\%$ lower latency compared with PLAID. Moreover, despite the extensive optimizations made by PLAID to the SV Heuristic, we still achieve significantly better latency on 5 out of 6 BEIR datasets we consider (§3). Given their retrieval efficiency compared to the SV heuristic, we believe that there are still significant gains to be obtained by optimizing the FDE method, and leave further exploration of this to future work.

在本文中，我们提出了MUVERA：一种原则性且实用的多向量（MV）检索算法，该算法通过构建多向量表示的固定维度编码（Fixed Dimensional Encoding，FDE）将多向量相似度转化为单向量（SV）相似度。我们证明了FDE点积能够对倒角相似度（Chamfer similarity）进行高质量的近似（§2.1）。实验表明，FDE是一种更有效的多向量相似度代理，因为与单向量启发式方法相比，它们只需检索$2 - 4 \times$更少的候选对象就能达到相同的召回率（§3.1）。我们通过对MUVERA进行端到端评估来补充这些结果，结果显示，与PLAID相比，它的平均召回率提高了${10}\%$，延迟降低了${90}\%$。此外，尽管PLAID对单向量启发式方法进行了大量优化，但在我们考虑的6个BEIR数据集中，有5个数据集上我们的方法在延迟方面仍显著更优（§3）。鉴于与单向量启发式方法相比，FDE方法具有更高的检索效率，我们认为通过优化FDE方法仍可获得显著的收益，并将对这方面的进一步探索留待未来工作。

Broader Impacts and Limitations: While retrieval is an important component of LLMs, which themselves have broader societal impacts, these impacts are unlikely to result from our retrieval algorithm. Our contribution simply improves the efficiency of retrieval, without enabling any fundamentally new capabilities. As for limitations, while we outperformed PLAID, sometimes significantly, on 5 out of the 6 datasets we studied, we did not outperform PLAID on MS MARCO, possibly due to their system having been carefully tuned for MS MARCO given its prevalence. Additionally,we did not study the effect that the average number of embeddings ${m}_{avg}$ per document has on retrieval quality of FDEs; this is an interesting direction for future work.

更广泛的影响和局限性：虽然检索是大语言模型（LLMs）的一个重要组成部分，而大语言模型本身会产生更广泛的社会影响，但这些影响不太可能源于我们的检索算法。我们的贡献仅仅是提高了检索效率，并没有带来任何根本性的新能力。至于局限性，虽然在我们研究的6个数据集中，有5个数据集上我们的方法有时显著优于PLAID，但在MS MARCO数据集上我们没有超过PLAID，这可能是因为考虑到MS MARCO的普遍性，他们的系统针对该数据集进行了精心调整。此外，我们没有研究每个文档的平均嵌入数量${m}_{avg}$对FDE检索质量的影响；这是未来工作的一个有趣方向。

## References

## 参考文献

[1] Alexandr Andoni, Piotr Indyk, and Robert Krauthgamer. Earth mover distance over high-dimensional spaces. In Proceedings of the 19th ACM-SIAM Symposium on Discrete Algorithms (SODA '2008), pages 343-352, 2008.

[1] Alexandr Andoni、Piotr Indyk和Robert Krauthgamer。高维空间上的地球移动距离。见《第19届ACM - SIAM离散算法研讨会论文集（SODA '2008）》，第343 - 352页，2008年。

[2] Rosa I Arriaga and Santosh Vempala. An algorithmic theory of learning: Robust concepts and random projection. Machine learning, 63:161-182, 2006.

[2] Rosa I Arriaga和Santosh Vempala。学习的算法理论：鲁棒概念和随机投影。《机器学习》，63:161 - 182，2006年。

[3] Kubilay Atasu and Thomas Mittelholzer. Linear-complexity data-parallel earth mover's distance approximations. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 364-373. PMLR, 09-15 Jun 2019.

[3] Kubilay Atasu和Thomas Mittelholzer。线性复杂度的数据并行地球移动距离近似方法。见Kamalika Chaudhuri和Ruslan Salakhutdinov编，《第36届国际机器学习会议论文集》，《机器学习研究会议录》第97卷，第364 - 373页。机器学习研究会议录（PMLR），2019年6月9 - 15日。

[4] Vassilis Athitsos and Stan Sclaroff. Estimating 3d hand pose from a cluttered image. In 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings., volume 2, pages II-432. IEEE, 2003.

[4] Vassilis Athitsos和Stan Sclaroff。从杂乱图像中估计三维手部姿势。见《2003年IEEE计算机协会计算机视觉与模式识别会议论文集》，2003年。第2卷，第II - 432页。IEEE，2003年。

[5] Ainesh Bakshi, Piotr Indyk, Rajesh Jayaram, Sandeep Silwal, and Erik Waingarten. Near-linear time algorithm for the chamfer distance. Advances in Neural Information Processing Systems, 36, 2024.

[5] Ainesh Bakshi、Piotr Indyk、Rajesh Jayaram、Sandeep Silwal和Erik Waingarten。倒角距离的近似线性时间算法。《神经信息处理系统进展》，36，2024年。

[6] Harry G Barrow, Jay M Tenenbaum, Robert C Bolles, and Helen C Wolf. Parametric correspondence and chamfer matching: Two new techniques for image matching. In Proceedings: Image Understanding Workshop, pages 21-27. Science Applications, Inc, 1977.

[6] Harry G Barrow、Jay M Tenenbaum、Robert C Bolles和Helen C Wolf。参数对应和倒角匹配：两种新的图像匹配技术。见《图像理解研讨会论文集》，第21 - 27页。科学应用公司，1977年。

[7] Yair Bartal. Probabilistic approximation of metric spaces and its algorithmic applications. In Proceedings of the 37th Annual IEEE Symposium on Foundations of Computer Science (FOCS '1996), 1996.

[7] Yair Bartal。度量空间的概率近似及其算法应用。见《第37届IEEE计算机科学基础研讨会论文集（FOCS '1996）》，1996年。

[8] Moses S Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing, pages 380-388, 2002.

[8] Moses S Charikar。基于舍入算法的相似度估计技术。见《第34届ACM计算理论研讨会论文集》，第380 - 388页，2002年。

[9] Xi Chen, Vincent Cohen-Addad, Rajesh Jayaram, Amit Levi, and Erik Waingarten. Streaming euclidean mst to a constant factor. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, STOC 2023, page 156-169, New York, NY, USA, 2023. Association for Computing Machinery.

[9] Xi Chen、Vincent Cohen - Addad、Rajesh Jayaram、Amit Levi和Erik Waingarten。流式欧几里得最小生成树的常数因子近似。见《第55届ACM计算理论研讨会论文集，STOC 2023》，第156 - 169页，美国纽约州纽约市，2023年。美国计算机协会。

[10] Xi Chen, Rajesh Jayaram, Amit Levi, and Erik Waingarten. New streaming algorithms for high dimensional emd and mst. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing, pages 222-233, 2022.

[10] Xi Chen、Rajesh Jayaram、Amit Levi和Erik Waingarten。用于高维地球移动距离和最小生成树的新型流式算法。见《第54届ACM SIGACT计算理论研讨会论文集》，第222 - 233页，2022年。

[11] Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. Specter: Document-level representation learning using citation-informed transformers. arXiv preprint arXiv:2004.07180, 2020.

[11] 阿尔曼·科汉（Arman Cohan）、谢尔盖·费尔德曼（Sergey Feldman）、伊兹·贝尔塔吉（Iz Beltagy）、道格·唐尼（Doug Downey）和丹尼尔·S·韦尔德（Daniel S Weld）。Specter：使用引用感知的Transformer进行文档级表示学习。预印本arXiv:2004.07180，2020年。

[12] Joshua Engels, Benjamin Coleman, Vihan Lakshman, and Anshumali Shrivastava. Dessert: An efficient algorithm for vector set search with vector set queries. Advances in Neural Information Processing Systems, 36, 2024.

[12] 约书亚·恩格尔斯（Joshua Engels）、本杰明·科尔曼（Benjamin Coleman）、维汉·拉克什曼（Vihan Lakshman）和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。Dessert：一种用于向量集查询的向量集搜索高效算法。《神经信息处理系统进展》，第36卷，2024年。

[13] Jittat Fakcharoenphol, Satish Rao, and Kunal Talwar. A tight bound on approximating arbitrary metrics by tree metrics. Journal of Computer and System Sciences, 69(3):485-497, 2004.

[13] 吉塔特·法查伦波尔（Jittat Fakcharoenphol）、萨蒂什·拉奥（Satish Rao）和库纳尔·塔尔瓦尔（Kunal Talwar）。用树度量近似任意度量的紧界。《计算机与系统科学杂志》（Journal of Computer and System Sciences），69(3):485 - 497，2004年。

[14] Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 605-613, 2017.

[14] 樊浩强（Haoqiang Fan）、苏浩（Hao Su）和利奥尼达斯·J·吉巴斯（Leonidas J Guibas）。用于从单张图像进行3D物体重建的点集生成网络。《电气与电子工程师协会计算机视觉与模式识别会议论文集》（Proceedings of the IEEE conference on computer vision and pattern recognition），第605 - 613页，2017年。

[15] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. A white box analysis of colbert. In Advances in Information Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28-April 1, 2021, Proceedings, Part II 43, pages 257-263. Springer, 2021.

[15] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。对COLBERT的白盒分析。《信息检索进展：第43届欧洲信息检索研究会议（ECIR 2021），虚拟会议，2021年3月28日 - 4月1日，会议录，第二部分43》（Advances in Information Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28 - April 1, 2021, Proceedings, Part II 43），第257 - 263页。施普林格出版社（Springer），2021年。

[16] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. Match your words! a study of lexical matching in neural information retrieval. In European Conference on Information Retrieval, pages 120-127. Springer, 2022.

[16] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。匹配你的词汇！神经信息检索中的词汇匹配研究。《欧洲信息检索会议》（European Conference on Information Retrieval），第120 - 127页。施普林格出版社（Springer），2022年。

[17] Luyu Gao, Zhuyun Dai, and Jamie Callan. Coil: Revisit exact lexical match in information retrieval with contextualized inverted list. arXiv preprint arXiv:2104.07186, 2021.

[17] 高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。COIL：利用上下文倒排列表重新审视信息检索中的精确词汇匹配。预印本arXiv:2104.07186，2021年。

[18] Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, and David Simcha. Quantization based fast inner product search. In Artificial intelligence and statistics, pages 482-490. PMLR, 2016.

[18] 郭瑞琪（Ruiqi Guo）、桑吉夫·库马尔（Sanjiv Kumar）、克日什托夫·乔罗曼斯基（Krzysztof Choromanski）和大卫·西姆查（David Simcha）。基于量化的快速内积搜索。《人工智能与统计学》（Artificial intelligence and statistics），第482 - 490页。机器学习研究会议录（PMLR），2016年。

[19] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pages 3887-3896. PMLR, 2020.

[19] 郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿全（Quan Geng）、大卫·西姆查（David Simcha）、费利克斯·陈（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。利用各向异性向量量化加速大规模推理。《国际机器学习会议》（International Conference on Machine Learning），第3887 - 3896页。机器学习研究会议录（PMLR），2020年。

[20] Sariel Har-Peled, Piotr Indyk, and Rajeev Motwani. Approximate nearest neighbor: Towards removing the curse of dimensionality. Theory of Computing, 8(1):321-350, 2012.

[20] 萨里尔·哈 - 佩雷德（Sariel Har - Peled）、彼得·因迪克（Piotr Indyk）和拉杰夫·莫特瓦尼（Rajeev Motwani）。近似最近邻：消除维度灾难。《计算理论》（Theory of Computing），8(1):321 - 350，2012年。

[21] Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. Introducing neural bag of whole-words with colberter: Contextualized late interactions using enhanced reduction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 737-747, 2022.

[21] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·哈塔布（Omar Khattab）、索菲娅·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。用COLBERTer引入全词神经词袋：使用增强约简的上下文后期交互。《第31届美国计算机协会信息与知识管理国际会议论文集》（Proceedings of the 31st ACM International Conference on Information & Knowledge Management），第737 - 747页，2022年。

[22] Piotr Indyk. Algorithms for dynamic geometric problems over data streams. In Proceedings of the 36th ACM Symposium on the Theory of Computing (STOC '2004), pages 373-380, 2004.

[22] 彼得·因迪克（Piotr Indyk）。数据流上动态几何问题的算法。《第36届美国计算机协会计算理论研讨会（STOC '2004）论文集》（Proceedings of the 36th ACM Symposium on the Theory of Computing (STOC '2004)），第373 - 380页，2004年。

[23] Rajesh Jayaram, Vahab Mirrokni, Shyam Narayanan, and Peilin Zhong. Massively parallel algorithms for high-dimensional euclidean minimum spanning tree. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 3960-3996. SIAM, 2024.

[23] 拉杰什·贾亚拉姆（Rajesh Jayaram）、瓦哈布·米罗克尼（Vahab Mirrokni）、希亚姆·纳拉亚南（Shyam Narayanan）和钟培林（Peilin Zhong）。高维欧几里得最小生成树的大规模并行算法。《2024年美国计算机协会 - 工业与应用数学学会离散算法年会（SODA）论文集》（Proceedings of the 2024 Annual ACM - SIAM Symposium on Discrete Algorithms (SODA)），第3960 - 3996页。工业与应用数学学会（SIAM），2024年。

[24] Rajesh Jayaram, Erik Waingarten, and Tian Zhang. Data-dependent Ish for the earth mover's distance. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, 2024.

[24] 拉杰什·贾亚拉姆（Rajesh Jayaram）、埃里克·温加滕（Erik Waingarten）和田章（Tian Zhang）。用于推土机距离的数据相关Ish。《第56届美国计算机协会计算理论年会论文集》（Proceedings of the 56th Annual ACM Symposium on Theory of Computing），2024年。

[25] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi. Diskann: Fast accurate billion-point nearest neighbor search on a single node. Advances in Neural Information Processing Systems, 32, 2019.

[25] 苏哈斯·贾亚拉姆·苏布拉马尼亚（Suhas Jayaram Subramanya）、富努·德夫里特（Fnu Devvrit）、哈沙·瓦尔丹·西姆哈德里（Harsha Vardhan Simhadri）、拉维尚卡尔·克里什纳瓦米（Ravishankar Krishnawamy）和罗汉·卡德科迪（Rohan Kadekodi）。DISKANN：单节点上快速准确的十亿点最近邻搜索。《神经信息处理系统进展》（Advances in Neural Information Processing Systems），32，2019年。

[26] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128, 2010.

[26] 埃尔韦·热古（Herve Jegou）、马蒂亚斯·杜泽（Matthijs Douze）和科迪莉亚·施密德（Cordelia Schmid）。用于最近邻搜索的乘积量化。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE transactions on pattern analysis and machine intelligence），33(1):117 - 128，2010年。

[27] Li Jiang, Shaoshuai Shi, Xiaojuan Qi, and Jiaya Jia. Gal: Geometric adversarial loss for single-view 3d-object reconstruction. In Proceedings of the European conference on computer vision (ECCV), pages 802-816, 2018.

[27] 江力（Li Jiang）、施少帅（Shaoshuai Shi）、齐晓娟（Xiaojuan Qi）和贾佳亚（Jiaya Jia）。GAL：用于单视图3D物体重建的几何对抗损失。《欧洲计算机视觉会议论文集》（Proceedings of the European conference on computer vision (ECCV)），第802 - 816页，2018年。

[28] Omar Khattab, Christopher Potts, and Matei Zaharia. Baleen: Robust multi-hop reasoning at scale via condensed retrieval. Advances in Neural Information Processing Systems, 34:27670- 27682, 2021.

[28] 奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。BALEEN：通过浓缩检索实现大规模鲁棒多跳推理。《神经信息处理系统进展》（Advances in Neural Information Processing Systems），34:27670 - 27682，2021年。

[29] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextual-ized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39-48, 2020.

[29] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。《科尔伯特：通过基于BERT的上下文后期交互实现高效有效的段落搜索》。收录于《第43届ACM SIGIR国际信息检索研究与发展会议论文集》，第39 - 48页，2020年。

[30] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From word embeddings to document distances. In International conference on machine learning, pages 957-966. PMLR, 2015.

[30] 马特·库斯纳（Matt Kusner）、孙宇（Yu Sun）、尼古拉斯·科尔金（Nicholas Kolkin）和基利安·温伯格（Kilian Weinberger）。《从词嵌入到文档距离》。收录于《国际机器学习会议》，第957 - 966页。机器学习研究会议录（PMLR），2015年。

[31] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, et al. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 2019.

[31] 汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔伯蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏金（Illia Polosukhin）、马修·凯尔西（Matthew Kelcey）、雅各布·德夫林（Jacob Devlin）等。《自然问题：问答研究的基准》。《计算语言学协会汇刊》，2019年。

[32] Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Zhao. Rethinking the role of token retrieval in multi-vector retrieval. Advances in Neural Information Processing Systems, 36, 2024.

[32] 李镇赫（Jinhyuk Lee）、戴珠云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming - Wei Chang）和文森特·赵（Vincent Zhao）。《重新思考多向量检索中词元检索的作用》。《神经信息处理系统进展》，第36卷，2024年。

[33] Chun-Liang Li, Tomas Simon, Jason Saragih, Barnabás Póczos, and Yaser Sheikh. Lbs autoencoder: Self-supervised fitting of articulated meshes to point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11967-11976, 2019.

[34] 李春良（Chun - Liang Li）、托马斯·西蒙（Tomas Simon）、杰森·萨拉吉（Jason Saragih）、巴尔纳巴斯·波乔斯（Barnabás Póczos）和亚西尔·谢赫（Yaser Sheikh）。《基于局部二值模式的自编码器：将铰接网格自监督拟合到点云》。收录于《IEEE/CVF计算机视觉与模式识别会议论文集》，第11967 - 11976页，2019年。

[34] Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. Learning cross-lingual ir from an english retriever. arXiv preprint arXiv:2112.08185, 2021.

[34] 李玉龙（Yulong Li）、马丁·弗朗茨（Martin Franz）、穆罕默德·阿拉法特·苏丹（Md Arafat Sultan）、巴瓦尼·伊耶尔（Bhavani Iyer）、李英锡（Young - Suk Lee）和阿维鲁普·西尔（Avirup Sil）。《从英语检索器学习跨语言信息检索》。预印本arXiv:2112.08185，2021年。

[35] Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. Fine-grained late-interaction multi-modal retrieval for retrieval augmented visual question answering. Advances in Neural Information Processing Systems, 36, 2024.

[35] 林伟哲（Weizhe Lin）、陈静红（Jinghong Chen）、梅景标（Jingbiao Mei）、亚历山德鲁·科卡（Alexandru Coca）和比尔·伯恩（Bill Byrne）。《用于检索增强视觉问答的细粒度后期交互多模态检索》。《神经信息处理系统进展》，第36卷，2024年。

[36] Simon Lupart, Thibault Formal, and Stéphane Clinchant. Ms-shift: An analysis of msmarco distribution shifts on neural retrieval. In European Conference on Information Retrieval, pages 636-652. Springer, 2023.

[36] 西蒙·卢帕特（Simon Lupart）、蒂博·福尔马尔（Thibault Formal）和斯特凡·克兰尚（Stéphane Clinchant）。《MS - 偏移：对MSMarco神经检索分布偏移的分析》。收录于《欧洲信息检索会议》，第636 - 652页。施普林格出版社，2023年。

[37] Sean MacAvaney and Nicola Tonellotto. A reproducibility study of plaid. arXiv preprint arXiv:2404.14989, 2024.

[37] 肖恩·麦卡瓦尼（Sean MacAvaney）和尼古拉·托内洛托（Nicola Tonellotto）。《对格子算法（Plaid）的可重复性研究》。预印本arXiv:2404.14989，2024年。

[38] Magdalen Dobson Manohar, Zheqi Shen, Guy Blelloch, Laxman Dhulipala, Yan Gu, Har-sha Vardhan Simhadri, and Yihan Sun. Parlayann: Scalable and deterministic parallel graph-based approximate nearest neighbor search algorithms. In Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming, pages 270-285, 2024.

[38] 玛格德琳·多布森·马诺哈尔（Magdalen Dobson Manohar）、沈哲奇（Zheqi Shen）、盖伊·布莱洛克（Guy Blelloch）、拉克什曼·杜利帕拉（Laxman Dhulipala）、顾燕（Yan Gu）、哈沙·瓦尔丹·西姆哈德里（Harsha Vardhan Simhadri）和孙一涵（Yihan Sun）。《Parlayann：可扩展且确定性的基于并行图的近似最近邻搜索算法》。收录于《第29届ACM SIGPLAN并行编程原理与实践年度研讨会论文集》，第270 - 285页，2024年。

[39] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316, 2022.

[39] 尼克拉斯·米宁霍夫（Niklas Muennighoff）、努阿曼·塔齐（Nouamane Tazi）、洛里克·马涅（Loïc Magne）和尼尔斯·赖默斯（Nils Reimers）。《MTEB：大规模文本嵌入基准》。预印本arXiv:2210.07316，2022年。

[40] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. Ms marco: A human-generated machine reading comprehension dataset. 2016.

[40] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李登（Li Deng）。《MS Marco：一个人工生成的机器阅读理解数据集》。2016年。

[41] Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia, and Christopher D Manning. Hindsight: Posterior-guided training of retrievers for improved open-ended generation. arXiv preprint arXiv:2110.07752, 2021.

[41] 阿什温·帕兰贾佩（Ashwin Paranjape）、奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）、马特·扎哈里亚（Matei Zaharia）和克里斯托弗·D·曼宁（Christopher D Manning）。《后见之明：通过后验引导训练检索器以改进开放式生成》。预印本arXiv:2110.07752，2021年。

[42] Yujie Qian, Jinhyuk Lee, Sai Meher Karthik Duddu, Zhuyun Dai, Siddhartha Brahma, Iftekhar Naim, Tao Lei, and Vincent Y Zhao. Multi-vector retrieval as sparse alignment. arXiv preprint arXiv:2211.01267, 2022.

[42] 钱玉洁（Yujie Qian）、李镇赫（Jinhyuk Lee）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、戴珠云（Zhuyun Dai）、悉达多·布拉马（Siddhartha Brahma）、伊夫特哈尔·奈姆（Iftekhar Naim）、雷涛（Tao Lei）和文森特·Y·赵（Vincent Y Zhao）。《多向量检索作为稀疏对齐》。预印本arXiv:2211.01267，2022年。

[43] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. Plaid: an efficient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 1747-1756, 2022.

[43] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。《格子算法（Plaid）：一种高效的后期交互检索引擎》。收录于《第31届ACM国际信息与知识管理会议论文集》，第1747 - 1756页，2022年。

[44] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488, 2021.

[44] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。《科尔伯特v2：通过轻量级后期交互实现高效有效的检索》。预印本arXiv:2112.01488，2021年。

[45] Erik B Sudderth, Michael I Mandel, William T Freeman, and Alan S Willsky. Visual hand tracking using nonparametric belief propagation. In 2004 Conference on Computer Vision and Pattern Recognition Workshop, pages 189-189. IEEE, 2004.

[45] 埃里克·B·萨德思（Erik B Sudderth）、迈克尔·I·曼德尔（Michael I Mandel）、威廉·T·弗里曼（William T Freeman）和艾伦·S·威尔斯基（Alan S Willsky）。使用非参数置信传播进行视觉手部跟踪。见2004年计算机视觉与模式识别研讨会会议论文集，第189 - 189页。电气与电子工程师协会（IEEE），2004年。

[46] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021.

[46] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。BEIR：信息检索模型零样本评估的异构基准。预印本arXiv:2104.08663，2021年。

[47] Henning Wachsmuth, Shahbaz Syed, and Benno Stein. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 241-251, 2018.

[47] 亨宁·瓦克斯穆特（Henning Wachsmuth）、沙赫巴兹·赛义德（Shahbaz Syed）和本诺·斯坦（Benno Stein）。在没有先验主题知识的情况下检索最佳反驳论点。见第56届计算语言学协会年会会议论文集（第1卷：长论文），第241 - 251页，2018年。

[48] Ziyu Wan, Dongdong Chen, Yan Li, Xingguang Yan, Junge Zhang, Yizhou Yu, and Jing Liao. Transductive zero-shot learning with visual structure constraint. Advances in neural information processing systems, 32, 2019.

[48] 万子玉、陈冬冬、李岩、闫星光、张军革、余一舟和廖静。具有视觉结构约束的直推式零样本学习。《神经信息处理系统进展》，32，2019年。

[49] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. Pseudo-relevance feedback for multiple representation dense retrieval. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval, pages 297-306, 2021.

[49] 王晓、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。多表示密集检索的伪相关反馈。见2021年美国计算机协会信息检索理论国际会议会议论文集，第297 - 306页，2021年。

[50] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. Reproducibility, replicability, and insights into dense multi-representation retrieval models: from colbert to col. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2552-2561, 2023.

[50] 王晓、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。密集多表示检索模型的可重复性、可复现性及见解：从ColBERT到Col。见第46届美国计算机协会信息检索研究与发展国际会议会议论文集，第2552 - 2561页，2023年。

[51] Orion Weller, Dawn Lawrie, and Benjamin Van Durme. Nevir: Negation in neural information retrieval. arXiv preprint arXiv:2305.07614, 2023.

[51] 奥赖恩·韦勒（Orion Weller）、道恩·劳里（Dawn Lawrie）和本杰明·范·杜尔姆（Benjamin Van Durme）。NEVIR：神经信息检索中的否定。预印本arXiv:2305.07614，2023年。

[52] David P Woodruff et al. Sketching as a tool for numerical linear algebra. Foundations and Trends® in Theoretical Computer Science, 10(1-2):1-157, 2014.

[52] 大卫·P·伍德拉夫（David P Woodruff）等人。草图法作为数值线性代数的工具。《理论计算机科学基础与趋势》，10(1 - 2)：1 - 157，2014年。

[53] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhut-dinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

[53] 杨植麟、齐鹏、张赛增、约书亚·本吉奥（Yoshua Bengio）、威廉·W·科恩（William W Cohen）、鲁斯兰·萨拉胡季诺夫（Ruslan Salakhut - dinov）和克里斯托弗·D·曼宁（Christopher D Manning）。HotpotQA：一个用于多样化、可解释多跳问答的数据集。预印本arXiv:1809.09600，2018年。

[54] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.

[54] 姚乐伟、黄润辉、侯璐、卢冠松、牛敏哲、徐航、梁小丹、李震国、蒋鑫和徐春静。FILIP：细粒度交互式语言 - 图像预训练。预印本arXiv:2111.07783，2021年。

[55] Jingtao Zhan, Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. Evaluating interpolation and extrapolation performance of neural retrieval models. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 2486-2496, 2022.

[55] 詹景涛、谢晓辉、毛佳欣、刘奕群、郭佳峰、张敏和马少平。评估神经检索模型的内插和外推性能。见第31届美国计算机协会信息与知识管理国际会议会议论文集，第2486 - 2496页，2022年。

[56] Ye Zhang, Md Mustafizur Rahman, Alex Braylan, Brandon Dang, Heng-Lu Chang, Henna Kim, Quinten McNamara, Aaron Angert, Edward Banner, Vivek Khetan, et al. Neural information retrieval: A literature review. arXiv preprint arXiv:1611.06792, 2016.

[56] 张烨、穆斯塔菲祖尔·拉赫曼（Md Mustafizur Rahman）、亚历克斯·布雷兰（Alex Braylan）、布兰登·当（Brandon Dang）、张恒禄、金亨娜（Henna Kim）、昆汀·麦克纳马拉（Quinten McNamara）、亚伦·安格特（Aaron Angert）、爱德华·班纳（Edward Banner）、维韦克·凯坦（Vivek Khetan）等。神经信息检索：文献综述。预印本arXiv:1611.06792，2016年。

## A Missing Proofs from Section 2.1

## 2.1节缺失的证明

In this section, we provide the missing proofs in Section 2.1. For convenience, we also reproduce theorem statements as they appear in the main text before the proofs. We begin by analyzing the runtime to compute query and document FDEs, as well as the sparsity of the queries.

在本节中，我们提供2.1节中缺失的证明。为方便起见，我们还在证明之前重现正文中出现的定理陈述。我们首先分析计算查询和文档FDE的运行时间，以及查询的稀疏性。

Lemma A.1. For any FDE parameters ${k}_{sim},{d}_{proj},{R}_{reps} \geq$ and sets $Q,P \subset  {\mathbb{R}}^{d}$ ,we can compute ${\mathbf{F}}_{q}\left( Q\right)$ in time ${T}_{q} \mathrel{\text{:=}} O\left( {{R}_{\text{reps }}\left| Q\right| d\left( {{d}_{\text{proj }} + {k}_{\text{sim }}}\right) }\right)$ ,and ${\mathbf{F}}_{q}\left( P\right)$ in time $O\left( {{T}_{q} + {R}_{\text{reps }}\left| P\right| {2}^{{k}_{\text{sim }}}{k}_{\text{sim }}}\right)$ . Moreover, ${\mathbf{F}}_{q}\left( Q\right)$ has at most $O\left( {\left| Q\right| {d}_{\text{proj }}{R}_{\text{reps }}}\right)$ non-zero entries.

引理A.1。对于任意的FDE参数${k}_{sim},{d}_{proj},{R}_{reps} \geq$和集合$Q,P \subset  {\mathbb{R}}^{d}$，我们可以在时间${T}_{q} \mathrel{\text{:=}} O\left( {{R}_{\text{reps }}\left| Q\right| d\left( {{d}_{\text{proj }} + {k}_{\text{sim }}}\right) }\right)$内计算出${\mathbf{F}}_{q}\left( Q\right)$，并在时间$O\left( {{T}_{q} + {R}_{\text{reps }}\left| P\right| {2}^{{k}_{\text{sim }}}{k}_{\text{sim }}}\right)$内计算出${\mathbf{F}}_{q}\left( P\right)$。此外，${\mathbf{F}}_{q}\left( Q\right)$最多有$O\left( {\left| Q\right| {d}_{\text{proj }}{R}_{\text{reps }}}\right)$个非零元素。

Proof. We first consider the queries. To generate the queries,we must first project each of the $\left| Q\right|$ queries via the inner random linear productions ${\psi }_{i} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}_{\text{proj }}}$ ,which requires $O\left( {\left| Q\right| d{d}_{\text{proj }}{R}_{\text{reps }}}\right)$ time to perform the matrix-query products for all repetitions. Next,we must compute ${\varphi }_{i}\left( q\right)$ for each $q \in  Q$ and repetition $i \in  \left\lbrack  {R}_{\text{reps }}\right\rbrack$ ,Each such value can be compute in $d \cdot  {k}_{\text{sim }}$ time to multiply the $q \in  {\mathbb{R}}^{d}$ by the ${k}_{\text{sim }}$ Gaussian vectors. Thus the total running time for this step i $O\left( {{R}_{\text{reps }}\left| Q\right| d{k}_{\text{sim }}}\right)$ . Finally,summing the relevant values into the FDE once ${\varphi }_{i}\left( q\right) ,{\mathbf{\psi }}_{i}\left( q\right)$ are computed can be done in $O\left( {\left| Q\right| {d}_{\text{proj }}}\right)$ time. For sparsity,note that only the coordinate blocks in the FDE corresponding to clusters $k$ in a repetition $i$ with at least one $q \in  \left| Q\right|$ with ${\varphi }_{i}\left( q\right)  = k$ are non-zero,and there can be at most $O\left( {{R}_{\text{reps }}\left| Q\right| }\right)$ of these blocks,each of which has $O\left( {d}_{\text{proj }}\right)$ coordinates.

证明。我们首先考虑查询。为了生成查询，我们必须首先通过内部随机线性投影${\psi }_{i} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}_{\text{proj }}}$对$\left| Q\right|$个查询中的每一个进行投影，这需要$O\left( {\left| Q\right| d{d}_{\text{proj }}{R}_{\text{reps }}}\right)$的时间来对所有重复操作执行矩阵 - 查询乘积。接下来，我们必须为每个$q \in  Q$和重复操作$i \in  \left\lbrack  {R}_{\text{reps }}\right\rbrack$计算${\varphi }_{i}\left( q\right)$，每个这样的值可以在$d \cdot  {k}_{\text{sim }}$的时间内通过将$q \in  {\mathbb{R}}^{d}$与${k}_{\text{sim }}$个高斯向量相乘得到。因此，此步骤的总运行时间为$O\left( {{R}_{\text{reps }}\left| Q\right| d{k}_{\text{sim }}}\right)$。最后，一旦计算出${\varphi }_{i}\left( q\right) ,{\mathbf{\psi }}_{i}\left( q\right)$，将相关值累加到FDE中可以在$O\left( {\left| Q\right| {d}_{\text{proj }}}\right)$的时间内完成。关于稀疏性，请注意，只有在重复操作$i$中对应于簇$k$且至少有一个满足${\varphi }_{i}\left( q\right)  = k$的$q \in  \left| Q\right|$的FDE中的坐标块是非零的，并且这些块最多有$O\left( {{R}_{\text{reps }}\left| Q\right| }\right)$个，每个块有$O\left( {d}_{\text{proj }}\right)$个坐标。

The document runtime is similar, except with the additional complexity required to carry out the fill_empty_clusters option. For each repetition,the runtime required to find the closest $p \in  P$ to a give cluster $k$ is $O\left( {\left| P\right|  \cdot  {k}_{\text{sim }}}\right)$ ,since we need to run over all $\left| p\right|$ values of $\varphi \left( p\right)$ and check how many bits disagree with $k$ . Thus,the total runtime is $O\left( {{R}_{\text{reps }}\left| P\right| B{k}_{\text{sim }}}\right)  = O\left( {{R}_{\text{reps }}\left| P\right| {2}^{{k}_{\text{sim }}}{k}_{\text{sim }}}\right)$ .

文档运行时间与之类似，只是执行fill_empty_clusters选项需要额外的复杂度。对于每次重复操作，找到给定簇$k$的最近邻$p \in  P$所需的运行时间为$O\left( {\left| P\right|  \cdot  {k}_{\text{sim }}}\right)$，因为我们需要遍历$\varphi \left( p\right)$的所有$\left| p\right|$个值，并检查与$k$有多少位不同。因此，总运行时间为$O\left( {{R}_{\text{reps }}\left| P\right| B{k}_{\text{sim }}}\right)  = O\left( {{R}_{\text{reps }}\left| P\right| {2}^{{k}_{\text{sim }}}{k}_{\text{sim }}}\right)$。

In what follows, we will need the following standard fact that random projections approximately preserve dot products. The proof is relatively standard, and can be found in [2], or see results on approximate matrix product [52] for more general bounds.

在接下来的内容中，我们需要用到以下标准事实：随机投影近似保留点积。证明相对标准，可在文献[2]中找到，或者查看关于近似矩阵乘积的结果[52]以获取更通用的界。

Fact A.2 ([2]). Fix $\varepsilon ,\delta  > 0$ . For any $d \geq  1$ and $x,y \in  {\mathbb{R}}^{d}$ ,let $S \in  {\mathbb{R}}^{t \times  d}$ by a matrix of independent entries distributed uniformly over $\{ 1, - 1\}$ ,where $t = O\left( {1/{\varepsilon }^{2} \cdot  \log {\delta }^{-1}}\right)$ . Then we have $\mathbb{E}\left\lbrack  {\langle {Sx},{Sy}\rangle }\right\rbrack   = \langle x,y\rangle$ ,and moreover with probability at least $1 - \delta$ we have

事实A.2（[2]）。固定$\varepsilon ,\delta  > 0$ 。对于任意的$d \geq  1$ 和$x,y \in  {\mathbb{R}}^{d}$ ，设$S \in  {\mathbb{R}}^{t \times  d}$ 是一个元素相互独立且在$\{ 1, - 1\}$ 上均匀分布的矩阵，其中$t = O\left( {1/{\varepsilon }^{2} \cdot  \log {\delta }^{-1}}\right)$ 。那么我们有$\mathbb{E}\left\lbrack  {\langle {Sx},{Sy}\rangle }\right\rbrack   = \langle x,y\rangle$ ，此外，至少以概率$1 - \delta$ 我们有

$$
\left| {\langle {Sx},{Sy}\rangle -\langle x,y\rangle }\right|  \leq  \varepsilon \parallel x{\parallel }_{2}\parallel y{\parallel }_{2}
$$

To anaylze the approximations of our FDEs, we begin by proving an upper bound on the value of the FDE dot product. In fact, we prove a stronger result: we show that our FDEs have the desirable property of being one-sided estimators - namely, they never overestimate the true Chamfer similarity. This is summarized in the following Lemma.

为了分析我们的全差分方程（FDEs）的近似情况，我们首先证明全差分方程点积值的一个上界。事实上，我们证明了一个更强的结果：我们表明我们的全差分方程具有单侧估计器（one - sided estimators）的理想性质——即，它们从不高估真实的倒角相似度（Chamfer similarity）。这总结在以下引理中。

Lemma A. 3 (One-Sided Error Estimator). Fix any sets $Q,P \subset  {\mathbb{R}}^{d}$ of unit vectors with $\left| Q\right|  + \left| P\right|  = m$ . Then if $d = {d}_{\text{proj }}$ ,we always have

引理A. 3（单侧误差估计器）。固定任意单位向量集$Q,P \subset  {\mathbb{R}}^{d}$ ，其中$\left| Q\right|  + \left| P\right|  = m$ 。那么如果$d = {d}_{\text{proj }}$ ，我们总是有

$$
\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( P\right) }\right\rangle   \leq  \operatorname{NCHAMFER}\left( {Q,P}\right) 
$$

Furthermore,for any $\delta  > 0$ ,if we set ${d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\delta }\right) }\right)$ ,then we have $\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( P\right) }\right\rangle   \leq  \mathrm{{NCHAMFER}}\left( {Q,P}\right)  + \varepsilon$ in expectation and with probability at least $1 - \delta$ .

此外，对于任意的$\delta  > 0$ ，如果我们设${d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\delta }\right) }\right)$ ，那么我们期望有$\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( P\right) }\right\rangle   \leq  \mathrm{{NCHAMFER}}\left( {Q,P}\right)  + \varepsilon$ ，并且至少以概率$1 - \delta$ 成立。

Proof. First claim simply follows from the fact that the average of a subset of a set of numbers can't be bigger than the maximum number in that set. More formally, we have:

证明。第一个断言直接源于这样一个事实：一组数的子集的平均值不可能大于该集合中的最大数。更正式地，我们有：

$$
\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{k = 1}}^{B}\mathop{\sum }\limits_{\substack{{q \in  Q} \\  {\varphi \left( q\right)  = k} }}\frac{1}{\left| P \cap  {\varphi }^{-1}\left( k\right) \right| }\mathop{\sum }\limits_{\substack{{p \in  P} \\  {\varphi \left( p\right)  = k} }}\langle q,p\rangle 
$$

$$
 \leq  \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{k = 1}}^{B}\mathop{\sum }\limits_{\substack{{q \in  Q} \\  {\varphi \left( q\right)  = k} }}\frac{1}{\left| P \cap  {\varphi }^{-1}\left( k\right) \right| }\mathop{\sum }\limits_{\substack{{p \in  P} \\  {\varphi \left( p\right)  = k} }}\mathop{\max }\limits_{{{p}^{\prime } \in  P}}\left\langle  {q,{p}^{\prime }}\right\rangle   \tag{4}
$$

$$
 = \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{k = 1}}^{B}\mathop{\sum }\limits_{\substack{{q \in  Q} \\  {\varphi \left( q\right)  = k} }}\mathop{\max }\limits_{{{p}^{\prime } \in  p}}\langle q,p\rangle  = \operatorname{NCHAMFER}\left( {Q,P}\right) 
$$

Which completes the first part of the lemma. For the second part,to analyze the case of ${d}_{\text{proj }} < d$ , when inner random projections are used,by applying Fact A.2,firstly we have $\mathbb{E}\left\lbrack  {\langle \mathbf{\psi }\left( p\right) ,\mathbf{\psi }\left( q\right) }\right\rbrack   =$ $\langle q,p\rangle$ for any $q \in  Q,p \in  P$ ,and secondly,after a union bound we over $\left| P\right|  \cdot  \left| Q\right|  \leq  {m}^{2}$ pairs,we have $\langle q,p\rangle  = \langle \mathbf{\psi }\left( p\right) ,\mathbf{\psi }\left( q\right) \rangle  \pm  \varepsilon$ simultaneously for all $q \in  Q,p \in  P$ ,with probability $1 - \delta$ ,for any constant $C > 1$ . The second part of the Lemma then follows similarly as above.

这就完成了引理的第一部分。对于第二部分，为了分析使用内部随机投影时${d}_{\text{proj }} < d$ 的情况，通过应用事实A.2，首先对于任意的$q \in  Q,p \in  P$ 我们有$\mathbb{E}\left\lbrack  {\langle \mathbf{\psi }\left( p\right) ,\mathbf{\psi }\left( q\right) }\right\rbrack   =$ $\langle q,p\rangle$ ，其次，在对$\left| P\right|  \cdot  \left| Q\right|  \leq  {m}^{2}$ 对应用联合界之后，对于任意常数$C > 1$ ，至少以概率$1 - \delta$ ，对于所有的$q \in  Q,p \in  P$ 同时有$\langle q,p\rangle  = \langle \mathbf{\psi }\left( p\right) ,\mathbf{\psi }\left( q\right) \rangle  \pm  \varepsilon$ 。引理的第二部分随后类似地得证。

We are now ready to give the proof of our main FDE approximation theorem.

我们现在准备给出我们主要的全差分方程近似定理的证明。

Theorem 2.1 (FDE Approximation). Fix any $\varepsilon ,\delta  > 0$ ,and sets $Q,P \subset  {\mathbb{R}}^{d}$ of unit vectors,and let $m = \left| Q\right|  + \left| P\right|$ . Then setting ${k}_{sim} = O\left( \frac{\log \left( {m{\delta }^{-1}}\right) }{\varepsilon }\right) ,{d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( \frac{m}{\varepsilon \delta }\right) }\right) ,{R}_{reps} = 1$ ,so that ${d}_{FDE} = {\left( m/\delta \right) }^{O\left( {1/\varepsilon }\right) }$ ,we have

定理2.1（FDE近似）。固定任意$\varepsilon ,\delta  > 0$，以及单位向量集合$Q,P \subset  {\mathbb{R}}^{d}$，并设$m = \left| Q\right|  + \left| P\right|$。然后令${k}_{sim} = O\left( \frac{\log \left( {m{\delta }^{-1}}\right) }{\varepsilon }\right) ,{d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( \frac{m}{\varepsilon \delta }\right) }\right) ,{R}_{reps} = 1$，使得${d}_{FDE} = {\left( m/\delta \right) }^{O\left( {1/\varepsilon }\right) }$，我们有

$$
\operatorname{NCHAMFER}\left( {Q,P}\right)  - \varepsilon  \leq  \frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( P\right) }\right\rangle   \leq  \operatorname{NCHAMFER}\left( {Q,P}\right)  + \varepsilon 
$$

in expectation,and with probability at least $1 - \delta$ .

在期望意义下，且概率至少为$1 - \delta$。

Proof of Theorem 2.1. The upper bound follows from Lemma A.3, so it will suffice to prove the lower bound. We first prove the result in the case when there are no random projections $\psi$ ,and remove this assumption at the end of the proof. Note that,by construction, ${\mathbf{F}}_{\mathrm{q}}$ is a linear mapping so that ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)  = \mathop{\sum }\limits_{{q \in  Q}}\mathbf{F}\left( q\right)$ ,thus

定理2.1的证明。上界由引理A.3得出，因此只需证明下界。我们首先证明不存在随机投影$\psi$时的结果，并在证明结尾处去掉这一假设。注意，根据构造，${\mathbf{F}}_{\mathrm{q}}$是一个线性映射，使得${\mathbf{F}}_{\mathrm{q}}\left( Q\right)  = \mathop{\sum }\limits_{{q \in  Q}}\mathbf{F}\left( q\right)$，因此

$$
\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \mathop{\sum }\limits_{{q \in  Q}}\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle  
$$

So it will suffice to prove that

因此，只需证明

$$
\Pr \left\lbrack  {\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   \geq  \mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle  - \varepsilon }\right\rbrack   \geq  1 - {\varepsilon \delta }/\left| Q\right|  \tag{5}
$$

for all $q \in  Q$ ,since then,by a union bound 5 will hold for all over all $q \in  Q$ with probability at least $1 - {\varepsilon \delta }$ ,in which case we will have

对于所有$q \in  Q$成立，因为这样一来，根据联合界，不等式5将以至少$1 - {\varepsilon \delta }$的概率对所有$q \in  Q$成立，在这种情况下我们将有

$$
\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   \geq  \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{q \in  Q}}\left( {\mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle  - \varepsilon }\right)  \tag{6}
$$

$$
 = \text{NCHAMFER}\left( {Q,P}\right)  - \varepsilon 
$$

which will complete the theorem.

这将完成定理的证明。

In what follows,for any $x,y \in  {\mathbb{R}}^{d}$ let $\theta \left( {x,y}\right)  \in  \left\lbrack  {0,\pi }\right\rbrack$ be the angle between $x,y$ . Now fix any $q \in  Q$ , and let ${p}^{ * } = \arg \mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle$ ,and let ${\theta }^{ * } = \theta \left( {q,{p}^{ * }}\right)$ . By construction,there always exists some set of points $S \subset  P$ such that

接下来，对于任意$x,y \in  {\mathbb{R}}^{d}$，设$\theta \left( {x,y}\right)  \in  \left\lbrack  {0,\pi }\right\rbrack$为$x,y$之间的夹角。现在固定任意$q \in  Q$，并设${p}^{ * } = \arg \mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle$，再设${\theta }^{ * } = \theta \left( {q,{p}^{ * }}\right)$。根据构造，总是存在某组点$S \subset  P$使得

$$
\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \left\langle  {q,\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right\rangle  
$$

Moreover, the RHS of the above equation is always bounded by 1 in magnitude, since it is an average of dot products of normalized vectors $q,p \in  {\mathbb{S}}^{d - 1}$ . In particular,there are two cases. In case (A) $S$ is the set of points $p$ with $\varphi \left( p\right)  = \varphi \left( q\right)$ ,and in case (B) $S$ is the single point $\arg \mathop{\min }\limits_{{p \in  P}}\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0}$ ,where $\parallel x - y{\parallel }_{0}$ denotes the hamming distance between any two bit-strings $x,y \in  \{ 0,1{\} }^{{k}_{\text{sim }}}$ ,and we are interpreting $\mathbf{\varphi }\left( p\right) ,\mathbf{\varphi }\left( q\right)  \in  \{ 0,1{\} }^{{k}_{\text{sim }}}$ as such bit-strings. Also let ${g}_{1},\ldots ,{g}_{{k}_{\operatorname{sim}}} \in  {\mathbb{R}}^{d}$ be the random Gaussian vectors that were drawn to define the partition function $\varphi$ . To analyze $S$ ,we first prove the following:

此外，上述方程的右边的模总是有界于1，因为它是归一化向量$q,p \in  {\mathbb{S}}^{d - 1}$的点积的平均值。特别地，有两种情况。在情况(A)中，$S$是满足$\varphi \left( p\right)  = \varphi \left( q\right)$的点集$p$，在情况(B)中，$S$是单点$\arg \mathop{\min }\limits_{{p \in  P}}\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0}$，其中$\parallel x - y{\parallel }_{0}$表示任意两个比特串$x,y \in  \{ 0,1{\} }^{{k}_{\text{sim }}}$之间的汉明距离（Hamming distance），并且我们将$\mathbf{\varphi }\left( p\right) ,\mathbf{\varphi }\left( q\right)  \in  \{ 0,1{\} }^{{k}_{\text{sim }}}$解释为这样的比特串。同时，设${g}_{1},\ldots ,{g}_{{k}_{\operatorname{sim}}} \in  {\mathbb{R}}^{d}$是为定义配分函数$\varphi$而抽取的随机高斯向量。为了分析$S$，我们首先证明以下内容：

Claim A.4. For any $q \in  Q$ and $p \in  P$ ,we have

断言A.4。对于任意$q \in  Q$和$p \in  P$，我们有

$$
\Pr \left\lbrack  {\left| {\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0} - {k}_{\text{sim }} \cdot  \frac{\theta \left( {q,p}\right) }{\pi }}\right|  > \sqrt{\varepsilon }{k}_{\text{sim }}}\right\rbrack   \leq  \left( \frac{\varepsilon \delta }{{m}^{2}}\right) 
$$

Proof. Fix any such $p$ ,and for $i \in  \left\lbrack  {k}_{\text{sim }}\right\rbrack$ let ${Z}_{i}$ be an indicator random variable that indicates the event that $\mathbf{1}\left( {\left\langle  {{g}_{i},p}\right\rangle   > 0}\right)  \neq  \mathbf{1}\left( {\left\langle  {{g}_{i},q}\right\rangle   > 0}\right)$ . First then note that $\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0} =$ $\mathop{\sum }\limits_{{i = 1}}^{{k}_{\operatorname{aim}}}{Z}_{i}$ . Now by rotational invariance of Gaussians,for a Gaussian vector $g \in  {\mathbb{R}}^{d}$ we have $\Pr \left\lbrack  {\mathbf{1}\left( {\langle g,x\rangle  > 0}\right)  \neq  \mathbf{1}\left( {\langle g,y\rangle  > 0}\right) }\right\rbrack   = \frac{\theta \left( {x,y}\right) }{\pi }$ for any two vectors $x,y \in  {\mathbb{R}}^{d}$ . It follows that ${Z}_{i}$ is a Bernoulli random variable with $\mathbb{E}\left\lbrack  {Z}_{i}\right\rbrack   = \frac{\theta \left( {x,y}\right) }{\pi }$ . By a simple application of Hoeffding’s inequality, we have

证明。固定任意这样的 $p$，并且对于 $i \in  \left\lbrack  {k}_{\text{sim }}\right\rbrack$，令 ${Z}_{i}$ 为一个指示随机变量，用于指示事件 $\mathbf{1}\left( {\left\langle  {{g}_{i},p}\right\rangle   > 0}\right)  \neq  \mathbf{1}\left( {\left\langle  {{g}_{i},q}\right\rangle   > 0}\right)$ 发生。首先注意到 $\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0} =$ $\mathop{\sum }\limits_{{i = 1}}^{{k}_{\operatorname{aim}}}{Z}_{i}$。现在，由于高斯分布的旋转不变性，对于高斯向量 $g \in  {\mathbb{R}}^{d}$，我们有对于任意两个向量 $x,y \in  {\mathbb{R}}^{d}$ 都有 $\Pr \left\lbrack  {\mathbf{1}\left( {\langle g,x\rangle  > 0}\right)  \neq  \mathbf{1}\left( {\langle g,y\rangle  > 0}\right) }\right\rbrack   = \frac{\theta \left( {x,y}\right) }{\pi }$。由此可知，${Z}_{i}$ 是一个伯努利随机变量（Bernoulli random variable），其参数为 $\mathbb{E}\left\lbrack  {Z}_{i}\right\rbrack   = \frac{\theta \left( {x,y}\right) }{\pi }$。通过简单应用霍夫丁不等式（Hoeffding’s inequality），我们有

$$
\Pr \left\lbrack  {\left| {\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0} - {k}_{\text{sim }} \cdot  \frac{\theta \left( {q,p}\right) }{\pi }}\right|  > \sqrt{\varepsilon }{k}_{\text{sim }}}\right\rbrack   = \Pr \left\lbrack  {\left| {\mathop{\sum }\limits_{{i = 1}}^{{k}_{\text{sim }}}{Z}_{i} - \mathbb{E}\left\lbrack  {\mathop{\sum }\limits_{{i = 1}}^{{k}_{\text{sim }}}{Z}_{i}}\right\rbrack  }\right|  > \sqrt{\varepsilon }{k}_{\text{sim }}}\right\rbrack  
$$

$$
 \leq  \exp \left( {-{2\varepsilon }{k}_{\text{sim }}}\right) 
$$

$$
 \leq  \left( \frac{\varepsilon \delta }{{m}^{2}}\right) 
$$

(7)

where we took ${k}_{\text{sim }} \geq  1/2 \cdot  \log \left( \frac{{m}^{2}}{\varepsilon \delta }\right) /\varepsilon$ ,which completes the proof.

这里我们取 ${k}_{\text{sim }} \geq  1/2 \cdot  \log \left( \frac{{m}^{2}}{\varepsilon \delta }\right) /\varepsilon$，至此证明完毕。

We now condition on the event in Claim A. 4 occurring for all $p \in  P$ ,which holds with probability at least $1 - \left| P\right|  \cdot  \left( \frac{\varepsilon \delta }{{m}^{2}}\right)  > 1 - \left( \frac{\varepsilon \delta }{m}\right)$ by a union bound. Call this event $\mathcal{E}$ ,and condition on it in what follows.

现在，我们假设命题 A.4 中的事件对于所有 $p \in  P$ 都发生，根据联合界（union bound），该事件发生的概率至少为 $1 - \left| P\right|  \cdot  \left( \frac{\varepsilon \delta }{{m}^{2}}\right)  > 1 - \left( \frac{\varepsilon \delta }{m}\right)$。将此事件记为 $\mathcal{E}$，并在接下来的讨论中以此为条件。

Now first suppose that we are in case (B),and the set $S$ of points which map to the cluster $\varphi \left( q\right)$ is given by $S = \left\{  {p}^{\prime }\right\}$ where ${p}^{\prime } = \arg \mathop{\min }\limits_{{p \in  P}}\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0}$ . Firstly,if ${p}^{\prime } = {p}^{ * }$ ,then we are done as $\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \left\langle  {q,{p}^{ * }}\right\rangle$ ,and 5 follows. Otherwise,by Claim A. 4 we must have had $\left| {\theta \left( {q,{p}^{\prime }}\right)  - \theta \left( {q,{p}^{ * }}\right) }\right|  \leq  \pi  \cdot  \sqrt{\varepsilon }$ . Using that the Taylor expansion of cosine is $\cos \left( x\right)  = 1 - {x}^{2}/2 + O\left( {x}^{4}\right)$ , we have

现在首先假设我们处于情况 (B)，并且映射到聚类 $\varphi \left( q\right)$ 的点集 $S$ 由 $S = \left\{  {p}^{\prime }\right\}$ 给出，其中 ${p}^{\prime } = \arg \mathop{\min }\limits_{{p \in  P}}\parallel \varphi \left( p\right)  - \varphi \left( q\right) {\parallel }_{0}$。首先，如果 ${p}^{\prime } = {p}^{ * }$，那么我们就完成了证明，因为 $\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \left\langle  {q,{p}^{ * }}\right\rangle$，并且结论 5 成立。否则，根据命题 A.4，我们必定有 $\left| {\theta \left( {q,{p}^{\prime }}\right)  - \theta \left( {q,{p}^{ * }}\right) }\right|  \leq  \pi  \cdot  \sqrt{\varepsilon }$。利用余弦函数的泰勒展开式 $\cos \left( x\right)  = 1 - {x}^{2}/2 + O\left( {x}^{4}\right)$，我们有

$$
\left| {\cos \left( {\theta \left( {q,{p}^{\prime }}\right) }\right)  - \cos \left( {\theta \left( {q,{p}^{ * }}\right) }\right) }\right|  \leq  O\left( \varepsilon \right) 
$$

Thus

因此

$$
\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \left\langle  {q,{p}^{\prime }}\right\rangle  
$$

$$
 = \cos \left( {\theta \left( {q,{p}^{\prime }}\right) }\right) 
$$

$$
 \geq  \cos \left( {\theta \left( {q,{p}^{ * }}\right) }\right)  - O\left( \varepsilon \right)  \tag{8}
$$

$$
 = \mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle  - O\left( \varepsilon \right) 
$$

which proves the desired statement 5 after a constant factor rescaling of $\varepsilon$ .

在对 $\varepsilon$ 进行常数因子缩放后，这就证明了所需的陈述 5。

Next,suppose we are in case (A) where $S = \left\{  {p \in  {P}^{\prime } \mid  \varphi \left( p\right)  = \varphi \left( q\right) }\right\}$ is non-empty. In this case, $S$ consists of the set of points $p$ with $\parallel \mathbf{\varphi }\left( p\right)  - \mathbf{\varphi }\left( q\right) {\parallel }_{0} = 0$ . From this,it follows again by Claim A. 4

接下来，假设我们处于情况 (A)，其中 $S = \left\{  {p \in  {P}^{\prime } \mid  \varphi \left( p\right)  = \varphi \left( q\right) }\right\}$ 非空。在这种情况下，$S$ 由满足 $\parallel \mathbf{\varphi }\left( p\right)  - \mathbf{\varphi }\left( q\right) {\parallel }_{0} = 0$ 的点集 $p$ 组成。由此，根据命题 A. 4 再次可得

that $\theta \left( {q,p}\right)  \leq  \sqrt{\varepsilon }\pi$ for any $p \in  S$ . Thus,by the same reasoning as above,we have

对于任意 $p \in  S$，有 $\theta \left( {q,p}\right)  \leq  \sqrt{\varepsilon }\pi$。因此，通过与上述相同的推理，我们有

$$
\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   = \frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}\cos \left( {\theta \left( {q,{p}^{\prime }}\right) }\right) 
$$

$$
 \geq  \frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}\left( {1 - O\left( \varepsilon \right) }\right)  \tag{9}
$$

$$
 \geq  \frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}\left( {\left\langle  {q,{p}^{ * }}\right\rangle   - O\left( \varepsilon \right) }\right) 
$$

$$
 = \mathop{\max }\limits_{{p \in  P}}\langle q,p\rangle  - O\left( \varepsilon \right) 
$$

which again proves the desired statement 5 in case (A), thereby completing the full proof in the case where there are no random projections.

这再次证明了情况 (A) 中所需的陈述 5，从而在没有随机投影的情况下完成了完整的证明。

To analyze the expectation,note that using the fact that $\left| \left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle  \right|  \leq  1$ deterministically,the small $O\left( {\varepsilon \delta }\right)$ probability of failure (i.e. the event that $\mathcal{E}$ does not hold) above can introduce at most a $O\left( {\varepsilon \delta }\right)  \leq  \varepsilon$ additive error into the expectation,which is acceptable after a constant factor rescaling of $\varepsilon$ .

为了分析期望，注意到利用 $\left| \left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle  \right|  \leq  1$ 是确定性的这一事实，上述小的 $O\left( {\varepsilon \delta }\right)$ 失败概率（即 $\mathcal{E}$ 不成立的事件）最多会给期望引入一个 $O\left( {\varepsilon \delta }\right)  \leq  \varepsilon$ 的加性误差，在对 $\varepsilon$ 进行常数因子重新缩放后，这是可以接受的。

Finally, to incorporate projections, by standard consequences of the Johnson Lindenstrauss Lemma (Fact A.2) setting ${d}_{\text{proj }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \frac{m}{\varepsilon }}\right)$ and projecting via a random Gaussian or $\pm  1$ matrix from $\psi$ : ${\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}_{\text{proj }}}$ ,for any set $S \subset  P$ we have that $\mathbb{E}\left\lbrack  \left\langle  {\psi \left( q\right) ,\psi \left( {\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right) }\right\rangle  \right\rbrack   = \left\langle  {q,\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right\rangle$ ,and moreover that $\left\langle  {q,\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right\rangle   = \left\langle  {\psi \left( q\right) ,\psi \left( {\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right) }\right\rangle  {\begin{Vmatrix}q\end{Vmatrix}}_{2}{\begin{Vmatrix}\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p\end{Vmatrix}}_{2} \pm  \varepsilon$ for all $q \in  Q,p \in$ $P$ with probability at least $1 - {\varepsilon \delta }$ . Note that $\parallel q{\parallel }_{2} = 1$ ,and by triangle inequality ${\begin{Vmatrix}\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p\end{Vmatrix}}_{2} \leq$ $\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}\parallel p{\parallel }_{2} = 1$ . Thus,letting ${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right)$ be the FDE values without the inner projection $\psi$ and ${\mathbf{F}}_{\mathrm{q}}^{\psi }\left( Q\right) ,{\mathbf{F}}_{\text{doc }}^{\psi }\left( P\right)$ be the FDE values with the inner projection $\psi$ ,conditioned on the above it follows that

最后，为了引入投影，根据约翰逊 - 林登斯特劳斯引理（事实A.2）的标准推论，设置${d}_{\text{proj }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \frac{m}{\varepsilon }}\right)$并通过一个来自$\psi$的随机高斯矩阵或$\pm  1$矩阵进行投影：${\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}_{\text{proj }}}$，对于任意集合$S \subset  P$，我们有$\mathbb{E}\left\lbrack  \left\langle  {\psi \left( q\right) ,\psi \left( {\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right) }\right\rangle  \right\rbrack   = \left\langle  {q,\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right\rangle$，此外，对于所有$q \in  Q,p \in$ $P$，至少以$1 - {\varepsilon \delta }$的概率有$\left\langle  {q,\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right\rangle   = \left\langle  {\psi \left( q\right) ,\psi \left( {\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p}\right) }\right\rangle  {\begin{Vmatrix}q\end{Vmatrix}}_{2}{\begin{Vmatrix}\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p\end{Vmatrix}}_{2} \pm  \varepsilon$。注意到$\parallel q{\parallel }_{2} = 1$，并且根据三角不等式有${\begin{Vmatrix}\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}p\end{Vmatrix}}_{2} \leq$ $\frac{1}{\left| S\right| }\mathop{\sum }\limits_{{p \in  S}}\parallel p{\parallel }_{2} = 1$。因此，设${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right)$为不进行内部投影$\psi$时的FDE（全变差距离估计，Full - Variation Distance Estimation）值，${\mathbf{F}}_{\mathrm{q}}^{\psi }\left( Q\right) ,{\mathbf{F}}_{\text{doc }}^{\psi }\left( P\right)$为进行内部投影$\psi$时的FDE值，基于上述条件可得

$$
\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{\psi }\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{\psi }\left( P\right) }\right\rangle   = \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{q \in  Q}}\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{\psi }\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{\psi }\left( P\right) }\right\rangle  
$$

$$
 = \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{q \in  Q}}\left( {\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   \pm  \varepsilon }\right)  \tag{10}
$$

$$
 = \frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle   \pm  \varepsilon 
$$

Finally, to analyze the expectation, note that since

最后，为了分析期望，注意到由于

$$
\left| {\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle  }\right|  \leq  \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{q \in  Q}}\left| \left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( P\right) }\right\rangle  \right|  \leq  1
$$

as before conditioning on this small probability event changes the expectation of 5 by at most a $\varepsilon$ additive factor,which completes the proof of the Theorem after a constant factor rescaling of $\varepsilon$ .

如前所述，对这个小概率事件进行条件限定最多会使5的期望增加一个$\varepsilon$的加性因子，在对$\varepsilon$进行常数因子重新缩放后，这就完成了定理的证明。

Equipped with Theorem 2.1, as well as the sparsity bounds from Lemma A.1, we are now prepared to prove our main theorem on approximate nearest neighbor search under the Chamfer Similarity.

有了定理2.1以及引理A.1中的稀疏性界，我们现在准备证明关于倒角相似度（Chamfer Similarity）下近似最近邻搜索的主要定理。

Theorem 2.2. Fix any $\varepsilon  > 0$ ,query $Q$ ,and dataset $P = \left\{  {{P}_{1},\ldots ,{P}_{n}}\right\}$ ,where $Q \subset  {\mathbb{R}}^{d}$ and each ${P}_{i} \subset  {\mathbb{R}}^{d}$ is a set of unit vectors. Let $m = \left| Q\right|  + \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left| {P}_{i}\right|$ . Then setting ${k}_{sim} = O\left( \frac{\log m}{\varepsilon }\right)$ , ${d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\varepsilon }\right) }\right)$ and ${R}_{reps} = O\left( {\frac{1}{{\varepsilon }^{2}}\log n}\right)$ so that ${d}_{FDE} = {m}^{O\left( {1/\varepsilon }\right) } \cdot  \log n$ . Then setting ${i}^{ * } = \arg \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( {P}_{i}\right) }\right\rangle$ ,with high probability (i.e. $1 - 1/\operatorname{poly}\left( n\right)$ ) we have:

定理2.2。固定任意$\varepsilon  > 0$、查询$Q$和数据集$P = \left\{  {{P}_{1},\ldots ,{P}_{n}}\right\}$，其中$Q \subset  {\mathbb{R}}^{d}$且每个${P}_{i} \subset  {\mathbb{R}}^{d}$是一组单位向量。设$m = \left| Q\right|  + \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left| {P}_{i}\right|$。然后设置${k}_{sim} = O\left( \frac{\log m}{\varepsilon }\right)$、${d}_{proj} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\varepsilon }\right) }\right)$和${R}_{reps} = O\left( {\frac{1}{{\varepsilon }^{2}}\log n}\right)$，使得${d}_{FDE} = {m}^{O\left( {1/\varepsilon }\right) } \cdot  \log n$。接着设置${i}^{ * } = \arg \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\left\langle  {{\mathbf{F}}_{q}\left( Q\right) ,{\mathbf{F}}_{doc}\left( {P}_{i}\right) }\right\rangle$，在大概率情况下（即$1 - 1/\operatorname{poly}\left( n\right)$），我们有：

$$
\operatorname{NCHAMFER}\left( {Q,{P}_{{i}^{ * }}}\right)  \geq  \mathop{\max }\limits_{{i \in  \left\lbrack  n\right\rbrack  }}\operatorname{NCHAMFER}\left( {Q,{P}_{i}}\right)  - \varepsilon 
$$

Given the query $Q$ ,the document ${P}^{ * }$ can be recovered in time $O\left( {\left| Q\right| \max \{ d,n\} \frac{1}{{\varepsilon }^{4}}\log \left( \frac{m}{\varepsilon }\right) \log n}\right)$ .

给定查询$Q$，文档${P}^{ * }$可以在时间$O\left( {\left| Q\right| \max \{ d,n\} \frac{1}{{\varepsilon }^{4}}\log \left( \frac{m}{\varepsilon }\right) \log n}\right)$内恢复。

Proof of Theorem 2.2. First note,for a single repetition,for any subset ${P}_{j} \in  D$ ,by Theorem 2.1 we have

定理2.2的证明。首先注意，对于单次重复，对于任意子集${P}_{j} \in  D$，根据定理2.1我们有

$$
\mathbb{E}\left\lbrack  \left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right) }\right\rangle  \right\rbrack   = \operatorname{NCHAMFER}\left( {Q,P}\right)  \pm  \varepsilon 
$$

Moreover,as demonsrated in the proof of Theorem 2.1,setting $\delta  = 1/{10}$ ,we have

此外，如定理2.1的证明中所示，设置$\delta  = 1/{10}$，我们有

$$
\left| {\frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right) }\right\rangle  }\right|  \leq  \frac{1}{\left| Q\right| }\mathop{\sum }\limits_{{q \in  Q}}\left| \left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right) }\right\rangle  \right|  \leq  1
$$

It follows that for each repetition $i \in  \left\lbrack  {R}_{\text{reps }}\right\rbrack$ ,letting ${\mathbf{F}}_{\mathrm{q}}{\left( Q\right) }^{i},{\mathbf{F}}_{\text{doc }}{\left( {P}_{j}\right) }^{i}$ be the coordinates in the final FDE vectors coordeesponding to that repetition,the random variable ${X}_{i} = \frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{i}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{i}\left( {P}_{j}\right) }\right\rangle$ is bounded in $\left\lbrack  {-1,1}\right\rbrack$ and has expectation NCHAMFER $\left( {Q,{P}_{j}}\right)  \pm  \varepsilon$ . By Chernoff bounds,averaging over ${R}_{\text{reps }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( n\right) }\right)$ repetitions,we have

由此可知，对于每次重复$i \in  \left\lbrack  {R}_{\text{reps }}\right\rbrack$，设${\mathbf{F}}_{\mathrm{q}}{\left( Q\right) }^{i},{\mathbf{F}}_{\text{doc }}{\left( {P}_{j}\right) }^{i}$为与该重复对应的最终FDE向量中的坐标，随机变量${X}_{i} = \frac{1}{\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{i}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{i}\left( {P}_{j}\right) }\right\rangle$在$\left\lbrack  {-1,1}\right\rbrack$内有界，且期望为NCHAMFER $\left( {Q,{P}_{j}}\right)  \pm  \varepsilon$。根据切尔诺夫界，对${R}_{\text{reps }} = O\left( {\frac{1}{{\varepsilon }^{2}}\log \left( n\right) }\right)$次重复取平均值，我们有

$$
\left| {\mathop{\sum }\limits_{{i = 1}}^{{R}_{\text{reps }}}\frac{1}{{R}_{\text{reps }}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{i}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{i}\left( {P}_{j}\right) }\right\rangle   - \operatorname{NCHAMFER}\left( {Q,{P}_{j}}\right) }\right|  \leq  {2\varepsilon } \tag{11}
$$

with probability $1 - 1/{n}^{C}$ for any arbitrarily large constant $C > 1$ . Note also that $\mathop{\sum }\limits_{{i = 1}}^{{R}_{\mathrm{{reps}}}}\frac{1}{{R}_{\mathrm{{reps}}}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{i}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{i}\left( {P}_{j}\right) }\right\rangle   = \frac{1}{{R}_{\mathrm{{reps}}}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right) }\right\rangle$ ,where ${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right)$ are the final FDEs. We can then condition on (11) holding for all documents $j \in  \left\lbrack  n\right\rbrack$ ,which holds with probability with probability $1 - 1/{n}^{C - 1}$ by a union bound. Conditioned on this,we have

以概率 $1 - 1/{n}^{C}$ 成立，其中 $C > 1$ 为任意大的常数。另请注意，$\mathop{\sum }\limits_{{i = 1}}^{{R}_{\mathrm{{reps}}}}\frac{1}{{R}_{\mathrm{{reps}}}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}^{i}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}^{i}\left( {P}_{j}\right) }\right\rangle   = \frac{1}{{R}_{\mathrm{{reps}}}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right) }\right\rangle$ 成立，其中 ${\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right)$ 是最终的全维嵌入（FDE，Final Dense Embeddings）。然后，我们可以基于所有文档 $j \in  \left\lbrack  n\right\rbrack$ 都满足式 (11) 这一条件进行分析，根据联合界（union bound），此条件成立的概率为 $1 - 1/{n}^{C - 1}$。在此条件下，我们有

$$
\operatorname{NCHAMFER}\left( {Q,{P}_{{i}^{ * }}}\right)  \geq  \frac{1}{{R}_{\text{reps }}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{{i}^{ * }}\right) }\right\rangle   - {2\varepsilon }
$$

$$
 = \mathop{\max }\limits_{{j \in  \left\lbrack  n\right\rbrack  }}\frac{1}{{R}_{\text{reps }}\left| Q\right| }\left\langle  {{\mathbf{F}}_{\mathrm{q}}\left( Q\right) ,{\mathbf{F}}_{\mathrm{{doc}}}\left( {P}_{j}\right) }\right\rangle   - {2\varepsilon } \tag{12}
$$

$$
 \geq  \mathop{\max }\limits_{{j \in  \left\lbrack  n\right\rbrack  }}\operatorname{NCHAMFER}\left( {Q,{P}_{j}}\right)  - {6\varepsilon }
$$

which completes the proof of the approximation after a constant factor scaling of $\varepsilon$ . The runtime bound follows from the runtime required to compute ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)$ ,which is $O\left( {\left| Q\right| {R}_{\text{reps }}d\left( {{d}_{\text{proj }} + {k}_{\text{sim }}}\right) }\right)  =$ $O\left( {\left| Q\right| \frac{\log n}{{\varepsilon }^{2}}d\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\varepsilon }\right)  + \frac{1}{\varepsilon }\log m}\right) \text{,plus the runtime required to brute force search for the nearest}}\right)$ dot product. Specifically,note that each of the $n$ FDE dot products can be computed in time proportional to the sparsity of ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)$ ,which is at most $O\left( {\left| Q\right| {d}_{\text{proj }}{R}_{\text{reps }}}\right)  = O\left( {\left| Q\right| \frac{1}{{\varepsilon }^{4}}\log \left( {m/\varepsilon }\right) \log n}\right)$ . Adding these two bounds together yields the desired runtime.

在对 $\varepsilon$ 进行常数因子缩放后，这就完成了近似性的证明。运行时间界（runtime bound）可由计算 ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)$ 所需的运行时间得出，即 $O\left( {\left| Q\right| {R}_{\text{reps }}d\left( {{d}_{\text{proj }} + {k}_{\text{sim }}}\right) }\right)  =$ $O\left( {\left| Q\right| \frac{\log n}{{\varepsilon }^{2}}d\left( {\frac{1}{{\varepsilon }^{2}}\log \left( {m/\varepsilon }\right)  + \frac{1}{\varepsilon }\log m}\right) \text{,plus the runtime required to brute force search for the nearest}}\right)$ 点积运算的时间。具体而言，请注意 $n$ 个全维嵌入（FDE）点积中的每一个都可以在与 ${\mathbf{F}}_{\mathrm{q}}\left( Q\right)$ 的稀疏度成正比的时间内计算完成，而其稀疏度至多为 $O\left( {\left| Q\right| {d}_{\text{proj }}{R}_{\text{reps }}}\right)  = O\left( {\left| Q\right| \frac{1}{{\varepsilon }^{4}}\log \left( {m/\varepsilon }\right) \log n}\right)$。将这两个界相加即可得到所需的运行时间。

## B Additional Dataset Information

## B 额外的数据集信息

In Table 8 we provide further dataset-specific information on the BEIR retrieval datasets used in this paper. Specifically, we state the sizes of the query and corpuses used, as well as the average number of embeddings produced by the ColBERTv2 model per document. Specifically, we consider the six BEIR retrieval datasets MS MARCO [40], NQ [31], HotpotQA [53], ArguAna [47], SciDocs [11], and Quora [46], Note that the MV corpus (after generating MV embeddings on all documents in a corpus) will have a total of $\#$ Corpus $\times$ (Avg # Embeddings per Doc) token embeddings. For even further details, see the BEIR paper [46].

在表 8 中，我们提供了本文所使用的 BEIR 检索数据集的更多特定于数据集的信息。具体来说，我们列出了所使用的查询集和语料库的大小，以及 ColBERTv2 模型为每个文档生成的嵌入的平均数量。具体而言，我们考虑了六个 BEIR 检索数据集：MS MARCO [40]、NQ [31]、HotpotQA [53]、ArguAna [47]、SciDocs [11] 和 Quora [46]。请注意，MV 语料库（在对语料库中的所有文档生成 MV 嵌入之后）将总共包含 $\#$ 个语料库 $\times$（每个文档的平均嵌入数量） 词元嵌入。如需更多详细信息，请参阅 BEIR 论文 [46]。

<!-- Media -->

<table><tr><td/><td>MS MARCO</td><td>HotpotQA</td><td>NQ</td><td>Quora</td><td>SciDocs</td><td>ArguAna</td></tr><tr><td>#Queries</td><td>6,980</td><td>7,405</td><td>3,452</td><td>10,000</td><td>1,000</td><td>1,406</td></tr><tr><td>#Corpus</td><td>8.84M</td><td>5.23M</td><td>2.68M</td><td>523K</td><td>25.6K</td><td>8.6K</td></tr><tr><td>Avg #Embeddings per Doc</td><td>78.8</td><td>68.65</td><td>100.3</td><td>18.28</td><td>165.05</td><td>154.72</td></tr></table>

<table><tbody><tr><td></td><td>微软机器阅读理解数据集（MS MARCO）</td><td>火锅问答数据集（HotpotQA）</td><td>自然问题数据集（NQ）</td><td>奎若问答平台（Quora）</td><td>科学文献数据集（SciDocs）</td><td>论证分析数据集（ArguAna）</td></tr><tr><td>查询数量（#Queries）</td><td>6,980</td><td>7,405</td><td>3,452</td><td>10,000</td><td>1,000</td><td>1,406</td></tr><tr><td>语料库数量（#Corpus）</td><td>8.84M</td><td>5.23M</td><td>2.68M</td><td>523K</td><td>25.6K</td><td>8.6K</td></tr><tr><td>每份文档的平均嵌入数量（Avg #Embeddings per Doc）</td><td>78.8</td><td>68.65</td><td>100.3</td><td>18.28</td><td>165.05</td><td>154.72</td></tr></tbody></table>

Figure 8: Dataset Specific Statistics for the BEIR datasets considered in this paper.

图8：本文所考虑的BEIR数据集的特定数据集统计信息。

<!-- Media -->

## C Additional Experiments and Plots

## C 额外实验与图表

In this Section, we provide additional plots to support the experimental results from Section 3. We providing plots for all six of the datasets and additional ranges of the $x$ -axis for our experiments in Section (§3.1), as well as additional experimental results, such as an evaluation of variance, and of the quality of final projections in the FDEs.

在本节中，我们提供额外的图表以支持第3节的实验结果。我们为所有六个数据集提供图表，并为第（§3.1）节中的实验提供$x$轴的额外范围，以及额外的实验结果，如方差评估和FDE中最终投影质量的评估。

FDE vs. SV Heuristic Experiments. In Figures 9 and 10, we show further datasets and an expanded recall range for the comparison of the SV Heuristic to retrieval via FDEs. We find that our $4\mathrm{\;k} +$ dimensional FDE methods outperform even the deduplciated SV heuristic (whose cost is somewhat unrealistic, since the SV heuristic must over-retrieve to handle duplicates) on most datasets, especially in lower recall regimes. In Table 1, we compare how many candidates must be retrieved by the SV heuristic, both with and without the deduplciation step, as well as by our FDE methods, in order to exceed a given recall threshold.

FDE与SV启发式实验。在图9和图10中，我们展示了更多的数据集以及SV启发式与通过FDE进行检索比较的扩展召回范围。我们发现，在大多数数据集上，尤其是在较低召回率的情况下，我们的$4\mathrm{\;k} +$维FDE方法甚至优于去重后的SV启发式（其成本有些不切实际，因为SV启发式必须过度检索以处理重复项）。在表1中，我们比较了为了超过给定的召回阈值，SV启发式在有和没有去重步骤的情况下以及我们的FDE方法必须检索的候选数量。

<!-- Media -->

<table><tr><td>Recall Threshold</td><td>SV non-dedup</td><td>SV dedup</td><td>20k FDE</td><td>10k FDE</td><td>4k FDE</td><td>2k FDE</td></tr><tr><td>80%</td><td>1200</td><td>300</td><td>60</td><td>60</td><td>80</td><td>200</td></tr><tr><td>85%</td><td>2100</td><td>400</td><td>90</td><td>100</td><td>200</td><td>300</td></tr><tr><td>90%</td><td>4500</td><td>800</td><td>200</td><td>200</td><td>300</td><td>800</td></tr><tr><td>95%</td><td>>10000</td><td>2100</td><td>700</td><td>800</td><td>1200</td><td>5600</td></tr></table>

<table><tbody><tr><td>召回阈值</td><td>结构变异未去重（SV non-dedup）</td><td>结构变异去重（SV dedup）</td><td>20k全数据集误差（20k FDE）</td><td>10k全数据集误差（10k FDE）</td><td>4k全数据集误差（4k FDE）</td><td>2k全数据集误差（2k FDE）</td></tr><tr><td>80%</td><td>1200</td><td>300</td><td>60</td><td>60</td><td>80</td><td>200</td></tr><tr><td>85%</td><td>2100</td><td>400</td><td>90</td><td>100</td><td>200</td><td>300</td></tr><tr><td>90%</td><td>4500</td><td>800</td><td>200</td><td>200</td><td>300</td><td>800</td></tr><tr><td>95%</td><td>>10000</td><td>2100</td><td>700</td><td>800</td><td>1200</td><td>5600</td></tr></tbody></table>

<!-- Media -->

Table 1: FDE retrieval vs SV Heuristic: number of candidates that must be retrieved by each method to exceed a given recall on MS MARCO. The first two columns are for the SV non-deduplicated and deduplicated heuristics, respectively, and the remaining four columns are for the FDE retrieved candidates with FDE dimensions $\{ {20480},{10240},{4096},{2048}\}$ ,respectively. Recall $@N$ values were computed in increments of 10 between 10-100 , and in increments of 100 between 100-10000, and were not computed above $N > {10000}$ .

表1：FDE检索与SV启发式方法对比：在MS MARCO数据集上，每种方法为达到给定召回率必须检索的候选数量。前两列分别对应SV未去重和去重启发式方法，其余四列分别对应FDE维度为$\{ {20480},{10240},{4096},{2048}\}$时FDE检索到的候选。召回率$@N$的值在10 - 100之间以10为增量计算，在100 - 10000之间以100为增量计算，且不计算高于$N > {10000}$的值。

Retrieval quality with respect to exact Chamfer. In Figure 11, we display the full plots for FDE Recall with respects to recovering the 1-nearest neighbor under Chamfer Similarity for all six BEIR datasets that we consider, including the two omitted from the main text (namely, SciDocs and ArguAna).

关于精确倒角距离（Chamfer）的检索质量。在图11中，我们展示了在倒角相似度（Chamfer Similarity）下恢复1 - 最近邻的FDE召回率的完整曲线图，涉及我们考虑的所有六个BEIR数据集，包括正文中省略的两个数据集（即SciDocs和ArguAna）。

### C.1 Variance of FDEs.

### C.1 FDE的方差

Since the FDE generation is a randomized process, one natural concern is whether there is large variance in the recall quality across different random seeds. Fortunately, we show that this is not the case, and the variance of the recall of FDE is essentially negligible, and can be easily accounted for via minor extra retrieval. To evaluate this,we chose four sets of FDE parameters $\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)$ which were Pareto optimal for their respective dimensionalities, generated 10 independent copies of the query and document FDEs for the entire MS MARCO dataset, and computed the average recall@100 and 1000 and standard deviation of these recalls. The results are shown in Table 2, where for all of the experiments the standard deviation was between 0.08-0.3% of a recall point, compared to the ${80} - {95}\%$ range of recall values. Note that Recall@1000 had roughly twice as small standard deviation as Recall@100.

由于FDE生成是一个随机过程，一个自然的担忧是不同随机种子下召回质量是否存在较大差异。幸运的是，我们证明情况并非如此，FDE召回率的方差实际上可以忽略不计，并且可以通过少量额外检索轻松解决。为了评估这一点，我们选择了四组FDE参数$\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)$，它们在各自的维度上是帕累托最优的，为整个MS MARCO数据集生成了10个独立的查询和文档FDE副本，并计算了召回率@100和@1000的平均值以及这些召回率的标准差。结果如表2所示，在所有实验中，与召回率值的${80} - {95}\%$范围相比，标准差为召回率点的0.08 - 0.3%。请注意，召回率@1000的标准差大约是召回率@100的一半。

<!-- Media -->

<table><tr><td>FDE params $\left( {{R}_{\mathrm{{reps}}},{k}_{\mathrm{{sim}}},{d}_{\mathrm{{proj}}}}\right)$</td><td>(20, 5, 32)</td><td>(20, 5, 16)</td><td>(20, 4, 16)</td><td>(20, 4, 8)</td></tr><tr><td>FDE Dimension</td><td>20480</td><td>10240</td><td>5120</td><td>2560</td></tr><tr><td>Recall@100</td><td>83.68</td><td>82.82</td><td>80.46</td><td>77.75</td></tr><tr><td>Standard Deviation</td><td>0.19</td><td>0.27</td><td>0.29</td><td>0.17</td></tr><tr><td>Recall@1000</td><td>95.37</td><td>94.88</td><td>93.67</td><td>91.85</td></tr><tr><td>Standard Deviation</td><td>0.08</td><td>0.11</td><td>0.16</td><td>0.12</td></tr></table>

<table><tbody><tr><td>分数阶微分方程（FDE）参数 $\left( {{R}_{\mathrm{{reps}}},{k}_{\mathrm{{sim}}},{d}_{\mathrm{{proj}}}}\right)$</td><td>(20, 5, 32)</td><td>(20, 5, 16)</td><td>(20, 4, 16)</td><td>(20, 4, 8)</td></tr><tr><td>分数阶微分方程（FDE）维度</td><td>20480</td><td>10240</td><td>5120</td><td>2560</td></tr><tr><td>前100召回率</td><td>83.68</td><td>82.82</td><td>80.46</td><td>77.75</td></tr><tr><td>标准差</td><td>0.19</td><td>0.27</td><td>0.29</td><td>0.17</td></tr><tr><td>前1000召回率</td><td>95.37</td><td>94.88</td><td>93.67</td><td>91.85</td></tr><tr><td>标准差</td><td>0.08</td><td>0.11</td><td>0.16</td><td>0.12</td></tr></tbody></table>

Table 2: Variance of FDE Recall Quality on MS MARCO.

表2：MS MARCO上FDE召回质量的方差

<!-- figureText: 1.0 0.9 0.95 0.90 0.85 SV $\begin{array}{lllll} {1000} & {2000} & {3000} & {4000} & {5000} \end{array}$ SV w/ Dedup FDE 20480 ArguAna 1.0 0.9 0.8 Recall@N 0.8 0.7 0.6 0.5 $\begin{array}{lllll} {1000} & {2000} & {3000} & {4000} & {5000} \end{array}$ $\begin{array}{lllll} {1000} & {2000} & {3000} & {4000} & {5000} \end{array}$ NQ SciDocs 0.8 0.7 0.6 0.5 0.4 0.7 0.3 0.6 0.2 Recall@N Recall@N -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_20.jpg?x=306&y=202&w=1180&h=671&r=0"/>

Figure 9: FDE retrieval vs SV Heuristic, Recall@100-5000

图9：FDE检索与SV启发式方法对比，召回率@100 - 5000

<!-- figureText: 1.0 0.9 1.00 0.95 0.90 0.85 0.80 0.75 SV SV w/ Dedup 100 200 300 FDE 20480 ArguAna FDE 10240 1.0 FDE 4096 0.9 0.8 0.6 0.5 0.4 Recall@N 0.9 0.8 0.7 0.6 0.5 0.5 0.4 NQ SciDocs 0.5 0.9 0.4 0.3 0.2 0.5 0.0 Recall@N -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_20.jpg?x=306&y=964&w=1179&h=672&r=0"/>

Figure 10: FDE retrieval vs SV Heuristic, Recall@5-500

图10：FDE检索与SV启发式方法对比，召回率@5 - 500

<table><tr><td>Experiment</td><td>w/o projection</td><td>w/ projection</td><td>w/o projection</td><td>w/ projection</td></tr><tr><td>Dimension</td><td>2460</td><td>2460</td><td>5120</td><td>5120</td></tr><tr><td>Recall@100</td><td>77.71</td><td>78.82</td><td>80.37</td><td>83.35</td></tr><tr><td>Recall@1000</td><td>91.91</td><td>91.62</td><td>93.55</td><td>94.83</td></tr><tr><td>Recall@10000</td><td>97.52</td><td>96.64</td><td>98.07</td><td>98.33</td></tr></table>

<table><tbody><tr><td>实验</td><td>无投影</td><td>有投影</td><td>无投影</td><td>有投影</td></tr><tr><td>维度</td><td>2460</td><td>2460</td><td>5120</td><td>5120</td></tr><tr><td>前100召回率</td><td>77.71</td><td>78.82</td><td>80.37</td><td>83.35</td></tr><tr><td>前1000召回率</td><td>91.91</td><td>91.62</td><td>93.55</td><td>94.83</td></tr><tr><td>前10000召回率</td><td>97.52</td><td>96.64</td><td>98.07</td><td>98.33</td></tr></tbody></table>

Table 3: Recall Quality of Final Projection based FDEs with ${d}_{\mathrm{{FDE}}} \in  \{ {2460},{5120}\}$

表3：基于最终投影的FDE（特征描述符）在${d}_{\mathrm{{FDE}}} \in  \{ {2460},{5120}\}$下的召回质量

<!-- figureText: MSMarco 1.0 0.9 0.8 0.7 0.6 400 500 200 300 400 500 ArguAna 1.00 0.99 0.98 0.97 400 500 300 400 500 0.9 0.8 0.7 0.7 0.6 300 500 200 300 Quora NQ 1.0 0.9 0.95 0.8 0.7 0.90 100 400 500 300 Recall@N -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_21.jpg?x=309&y=204&w=1177&h=662&r=0"/>

Figure 11: Comparison of FDE recall with respect to the most similar point under Chamfer.

图11：在倒角距离（Chamfer）下FDE召回率与最相似点的比较

<table><tr><td>Experiment</td><td>w/o projection</td><td>w/ projection</td><td>w/o projection</td><td>w/ projection</td></tr><tr><td>Dimension</td><td>10240</td><td>10240</td><td>20480</td><td>20480</td></tr><tr><td>Recall@100</td><td>82.31</td><td>85.15</td><td>83.36</td><td>86.00</td></tr><tr><td>Recall@1000</td><td>94.91</td><td>95.68</td><td>95.58</td><td>95.95</td></tr><tr><td>Recall@10000</td><td>98.76</td><td>98.93</td><td>98.95</td><td>99.17</td></tr></table>

<table><tbody><tr><td>实验</td><td>无投影</td><td>有投影</td><td>无投影</td><td>有投影</td></tr><tr><td>维度</td><td>10240</td><td>10240</td><td>20480</td><td>20480</td></tr><tr><td>前100召回率</td><td>82.31</td><td>85.15</td><td>83.36</td><td>86.00</td></tr><tr><td>前1000召回率</td><td>94.91</td><td>95.68</td><td>95.58</td><td>95.95</td></tr><tr><td>前10000召回率</td><td>98.76</td><td>98.93</td><td>98.95</td><td>99.17</td></tr></tbody></table>

Table 4: Recall Quality of Final Projection based FDEs with ${d}_{\mathrm{{FDE}}} \in  \{ {10240},{20480}\}$

表4：基于最终投影的FDE（特征描述符）在${d}_{\mathrm{{FDE}}} \in  \{ {10240},{20480}\}$下的召回质量

<!-- Media -->

### C.2 Comparison to Final Projections.

### C.2 与最终投影的比较

We now show the effect of employing final projections to reduce the target dimensionality of the FDE’s. For all experiments,the final projection ${\psi }^{\prime }$ is implemented in the same way as inner projections are: namely,via multiplication by a random $\pm  1$ matrix. We choose four target dimensions, ${d}_{\text{FDE }} \in  \{ {2460},{5120},{10240},{20480}\}$ ,and choose the Pareto optimal parameters $\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)$ from the grid search without final projections in Section 3.1,which are $\left( {{20},4,8}\right) ,\left( {{20},5,8}\right) ,\left( {{20},5,{16}}\right) ,\left( {{20},5,{32}}\right)$ . We then build a large dimensional FDE with the parameters $\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)  = \left( {{40},6,{128}}\right)$ . Here,since $d = {d}_{\text{proj }}$ ,we do not use any inner productions when constructing the FDE. We then use a single random final projection to reduce the dimensionality of this FDE from ${R}_{\text{reps }} \cdot  {2}^{{k}_{\text{sim }}} \cdot  {d}_{\text{proj }} = {327680}$ down to each of the above target dimensions ${d}_{\text{FDE }}$ . The results are show in Tables 3 and 4. Notice that incorporating final projections can have a non-trivial impact on recall,especially for Recall@100, where it can increase by around 3%. In particular, FDEs with the final projections are often better than FDEs with twice the dimensionality without final projections. The one exception is the 2460-dimensional FDE, where the Recall@100 only improved by 1.1%, and the Recall@1000 was actually lower bound 0.3%.

我们现在展示采用最终投影来降低FDE（特征描述符）目标维度的效果。对于所有实验，最终投影${\psi }^{\prime }$的实现方式与内部投影相同：即通过与一个随机的$\pm  1$矩阵相乘。我们选择四个目标维度${d}_{\text{FDE }} \in  \{ {2460},{5120},{10240},{20480}\}$，并从3.1节中无最终投影的网格搜索中选择帕累托最优参数$\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)$，这些参数为$\left( {{20},4,8}\right) ,\left( {{20},5,8}\right) ,\left( {{20},5,{16}}\right) ,\left( {{20},5,{32}}\right)$。然后，我们使用参数$\left( {{R}_{\text{reps }},{k}_{\text{sim }},{d}_{\text{proj }}}\right)  = \left( {{40},6,{128}}\right)$构建一个高维FDE。这里，由于$d = {d}_{\text{proj }}$，我们在构建FDE时不使用任何内部积。接着，我们使用单个随机最终投影将该FDE的维度从${R}_{\text{reps }} \cdot  {2}^{{k}_{\text{sim }}} \cdot  {d}_{\text{proj }} = {327680}$降至上述每个目标维度${d}_{\text{FDE }}$。结果显示在表3和表4中。注意，纳入最终投影对召回率可能有显著影响，特别是对于Recall@100，其可能提高约3%。特别是，带有最终投影的FDE通常比维度是其两倍但无最终投影的FDE更好。唯一的例外是2460维的FDE，其Recall@100仅提高了1.1%，而Recall@1000实际上降低了0.3%。

### C.3 Ball Carving

### C.3 球雕刻

We now provide further details on the ball carving technique described in Section 3.2 that is used in our online experiments. Specifically, to improve rescoring latency, we reduce the number of query embeddings by a pre-clustering stage. Specifically,we group the queries $Q$ into clusters ${C}_{1},\ldots ,{C}_{k}$ , set ${c}_{i} = \mathop{\sum }\limits_{{q \in  {C}_{i}}}q$ and ${Q}_{C} = \left\{  {{c}_{1},\ldots ,{c}_{k}}\right\}$ . Then,after retrieving a set of candidate documents with the FDEs,instead of rescoring via $\operatorname{CHAMFER}\left( {Q,P}\right)$ for each candidate $P$ ,we rescore via $\operatorname{CHAMFER}\left( {{Q}_{C},P}\right)$ ,which runs in time $O\left( {\left| {Q}_{C}\right|  \cdot  \left| P\right| }\right)$ ,offering speed-ups when the number of clusters is small. Instead of fixing $k$ ,we perform a greedy ball-carving procedure to allow $k$ to adapt to $Q$ . Specifically,given a threshold $\tau$ ,we select an arbitrary point $q \in  Q$ ,cluster it with all other points ${q}^{\prime } \in  Q$ with $\left\langle  {q,{q}^{\prime }}\right\rangle   \geq  \tau$ ,remove the clustered points and repeat until all points are clustered.

我们现在详细介绍3.2节中描述的用于在线实验的球雕刻技术。具体来说，为了提高重排序延迟，我们通过预聚类阶段减少查询嵌入的数量。具体而言，我们将查询$Q$分组到簇${C}_{1},\ldots ,{C}_{k}$中，设置${c}_{i} = \mathop{\sum }\limits_{{q \in  {C}_{i}}}q$和${Q}_{C} = \left\{  {{c}_{1},\ldots ,{c}_{k}}\right\}$。然后，在使用FDE检索到一组候选文档后，我们不是通过$\operatorname{CHAMFER}\left( {Q,P}\right)$对每个候选$P$进行重排序，而是通过$\operatorname{CHAMFER}\left( {{Q}_{C},P}\right)$进行重排序，其运行时间为$O\left( {\left| {Q}_{C}\right|  \cdot  \left| P\right| }\right)$，当簇的数量较少时可提高速度。我们不固定$k$，而是执行贪婪球雕刻过程，使$k$适应$Q$。具体来说，给定一个阈值$\tau$，我们选择一个任意点$q \in  Q$，将其与所有满足$\left\langle  {q,{q}^{\prime }}\right\rangle   \geq  \tau$的其他点${q}^{\prime } \in  Q$聚类，移除已聚类的点，然后重复该过程，直到所有点都被聚类。

<!-- Media -->

<!-- figureText: NQ MSMarco 100.0 0.92 1000.0 Ball Carving Threshold Ball Carving Threshold 0.80 Ball Carving Threshold -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_22.jpg?x=318&y=237&w=1174&h=294&r=0"/>

Figure 12: Plots showing the trade-off between the threshold used for ball carving and the end-to-end recall.

图12：展示球雕刻阈值与端到端召回率之间权衡关系的图表

<!-- figureText: 50000 Chamfer Throughput on MS MARCO - - Sequential Ball Carving Threshold 30000 15000 -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_22.jpg?x=632&y=670&w=528&h=398&r=0"/>

Figure 13: Per-Core Re-ranking QPS versus Ball Carving Threshold, on MS MARCO dataset.

图13：在MS MARCO数据集上，每核心重排序每秒查询数（QPS）与球雕刻阈值的关系

<!-- Media -->

In Figure 12,we show the the trade-off between end-to-end Recall@k of MUVERA and the ball carving threshold used. Notice that for both $k = {100}$ and $k = {1000}$ ,the Recall curves flatten dramatically after a threshold of $\tau  = {0.6}$ ,and for all datasets they are essentially flat after $\tau  \geq  {0.7}$ . Thus, for such thresholds we incur essentially no quality loss by the ball carving. For this reason, we choose the value of $\tau  = {0.7}$ in our end-to-end experiments.

在图12中，我们展示了MUVERA的端到端Recall@k与所使用的球分割阈值之间的权衡关系。请注意，对于$k = {100}$和$k = {1000}$，在阈值达到$\tau  = {0.6}$之后，召回率曲线急剧趋于平缓，而对于所有数据集，在$\tau  \geq  {0.7}$之后它们基本保持平稳。因此，对于这样的阈值，球分割基本上不会导致质量损失。出于这个原因，我们在端到端实验中选择了$\tau  = {0.7}$这个值。

On the other hand, we show that ball-carving at this threshold of 0.7 gives non-trivial efficiency gains. Specifically, in Figure 13, we plot the per-core queries-per-second of re-ranking (i.e. computing CHAMFER $\left( {{Q}_{C},P}\right)$ ) against varying ball carving thresholds for the MS MARCO dataset. For sequential re-ranking,ball carving at a $\tau  = {0.7}$ threshold provides a ${25}\%$ QPS improvement,and when re-ranking is being done in parallel (over all cores simultaneously) it yields a ${20}\%$ QPS improvement. Moreover,with a threshold of $\tau  = {0.7}$ ,there were an average of 5.9 clusters created per query on MS Marco. This reduces the number of embeddings per query by ${5.4} \times$ ,down from the initial fixed setting of $\left| Q\right|  = {32}$ . This suggests that pre-clustering the queries before re-ranking gives non-trivial runtime improvements with negligible quality loss. This also suggests that a fixed setting of $\left| Q\right|  = {32}$ query embeddings per model is likely excessive for MV similarity quality,and that fewer queries could achieve a similar performance.

另一方面，我们表明在0.7这个阈值下进行球分割能带来显著的效率提升。具体来说，在图13中，我们针对MS MARCO数据集，绘制了重排序（即计算CHAMFER $\left( {{Q}_{C},P}\right)$）的每核心每秒查询数（QPS）与不同球分割阈值的关系图。对于顺序重排序，在$\tau  = {0.7}$阈值下进行球分割可使QPS提高${25}\%$，而当重排序并行进行（同时在所有核心上）时，QPS可提高${20}\%$。此外，在$\tau  = {0.7}$的阈值下，在MS Marco数据集上每个查询平均创建5.9个聚类。这使得每个查询的嵌入数量从初始的固定设置$\left| Q\right|  = {32}$减少了${5.4} \times$。这表明在重排序之前对查询进行预聚类可以在质量损失可忽略不计的情况下显著提高运行时间。这也表明，每个模型固定设置$\left| Q\right|  = {32}$个查询嵌入对于MV相似度质量来说可能过多，较少的查询也可以实现类似的性能。

### C.4 Product Quantization

### C.4 乘积量化

PQ Details We implemented our product quantizers using a simple "textbook" $k$ -means based quantizer. Recall that $\mathrm{{AH}} - C - G$ means that each consecutive group of $G$ dimensions is represented by $C$ centers. We train the quantizer by: (1) taking for each group of dimensions the coordinates of a sample of at most 100,000 vectors from the dataset,and (2) running $k$ -means on this sample using $k = C = {256}$ centers until convergence. Given a vector $x \in  {\mathbb{R}}^{d}$ ,we can split $x$ into $d/G$ blocks of coordinates ${x}_{\left( 1\right) },\ldots ,{x}_{\left( d/G\right) } \in  {\mathbb{R}}^{G}$ each of size $G$ . The block ${x}_{\left( i\right) }$ can be compressed by representing ${x}_{\left( i\right) }$ by the index of the centroid from the $i$ -th group that is nearest to ${x}_{\left( i\right) }$ . Since there are 256 centroids per group,each block ${x}_{\left( i\right) }$ can then be represented by a single byte.

PQ细节 我们使用基于简单的“教科书式”$k$ -均值量化器实现了我们的乘积量化器。回顾一下，$\mathrm{{AH}} - C - G$表示每连续的$G$个维度组由$C$个中心表示。我们通过以下步骤训练量化器：(1) 对于每个维度组，从数据集中选取最多100,000个向量样本的坐标；(2) 对这个样本使用$k = C = {256}$个中心运行$k$ -均值算法，直到收敛。给定一个向量$x \in  {\mathbb{R}}^{d}$，我们可以将$x$分割成$d/G$个坐标块${x}_{\left( 1\right) },\ldots ,{x}_{\left( d/G\right) } \in  {\mathbb{R}}^{G}$，每个块的大小为$G$。块${x}_{\left( i\right) }$可以通过用距离${x}_{\left( i\right) }$最近的第$i$组质心的索引来表示${x}_{\left( i\right) }$进行压缩。由于每组有256个质心，因此每个块${x}_{\left( i\right) }$可以用一个字节表示。

<!-- Media -->

<!-- figureText: ArguAna Quora Uncompressed PQ-256-5 PQ-256-16 PQ-256-4 PQ-256-16 Recall@100 Recall@100 MS MARCC PQ-256-2 PQ-256-8 PQ-256-16 PQ-256-5 0.80 0.82 0.88 0.90 Recall@100 Recall@100 Uncompressed PQ-256-5 PQ-256-4 PQ-256-16 PQ-256-4 Recall@100 NQ ${10}^{3}$ ${10}^{2}$ Uncompressed PQ-256-5 - Uncompressed ${10}^{1}$ PQ-256-4 PQ-256-16 PQ-256-4 ${10}^{1}$ Recall@100 -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_23.jpg?x=309&y=234&w=1165&h=618&r=0"/>

Figure 14: Plots showing the QPS vs. Recall@100 for MUVERA on the BEIR datasets we evaluate in this paper. The different curves are obtained by using different PQ methods on 10240-dimensional FDEs.

图14：展示了本文评估的BEIR数据集上MUVERA的QPS与Recall@100的关系图。不同的曲线是通过对10240维FDE使用不同的PQ方法得到的。

<!-- figureText: ArguAna SCIDOCS Quora PQ-256-8 PQ-256-2 PQ-256-8 0.45 0.50 0.55 0.96 0.97 0.98 0.99 HotpotQA MS MARCC ...1- PQ-256-5 PQ-256-2 PQ-256-8 _____ PQ-256-8 PQ-256-4 PQ-256-16 0.7 0.8 0.80 0.85 0.00 0.95 Recall ${10}^{4}$ PQ-256-2 PQ-256-8 PQ-256-2 0.90 0.94 0.96 0.35 0.40 ${10}^{3}$ Uncompressed ...1- PQ-256-5 Uncompressed PQ-256-2 ... PQ-256-8 PQ-256-2 ${10}^{1}$ ${10}^{1}$ 0.80 0.85 0.90 0.95 0.5 0.6 Recall -->

<img src="https://cdn.noedgeai.com/01959aa6-d103-724b-80b8-d1794f133272_23.jpg?x=309&y=1025&w=1165&h=614&r=0"/>

Figure 15: Plots showing the QPS vs. Recall@1000 for MUVERA on the BEIR datasets we evaluate in this paper. The different curves are obtained by using different PQ methods on 10240-dimensional FDEs.

图15：展示了本文评估的BEIR数据集上MUVERA的QPS与Recall@1000的关系图。不同的曲线是通过对10240维FDE使用不同的PQ方法得到的。

<!-- Media -->

Results In Figures 14 and 15 we show the full set of results for our QPS experiments from Section 3.2 on all of the BEIR datasets that we evaluated in this paper. We include results for both Recall@100 (Figure 14) and Recall@1000 (Figure 15).

结果 在图14和图15中，我们展示了第3.2节中对本文评估的所有BEIR数据集进行的QPS实验的完整结果集。我们同时包括了Recall@100（图14）和Recall@1000（图15）的结果。

We find that PQ-256-8 is consistently the best performing PQ codec across all of the datasets that we tested. Not using PQ at all results in significantly worse results (worse by at least $5 \times$ compared to using PQ) at the same beam width for the beam; however, the recall loss due to using PQ-256-8 is minimal, and usually only a fraction of a percent. Since our retrieval engine works by over-retrieving with respect to the FDEs and then reranking using Chamfer similarity, the loss due to approximating the FDEs using PQ can be handled by simply over-retrieving slightly more candidates.

我们发现，在我们测试的所有数据集上，PQ - 256 - 8始终是性能最佳的乘积量化（Product Quantization，PQ）编解码器。在相同的束宽下，完全不使用PQ会导致显著更差的结果（与使用PQ相比，至少差$5 \times$）；然而，使用PQ - 256 - 8导致的召回率损失极小，通常仅为百分之几。由于我们的检索引擎的工作方式是相对于特征描述符（Feature Descriptors，FDEs）进行过度检索，然后使用倒角相似度进行重新排序，因此通过稍微多检索一些候选对象，就可以处理因使用PQ近似FDEs而导致的损失。

We also observe that the difference between different PQ codecs is much more pronounced in the lower-recall regime when searching for the top 1000 candidates for a query. For example, most of the plots in Figure 15 show significant stratification in the QPS achieved in lower recall regimes, with PQ-256-16 (the most compressed and memory-efficient format) usually outperforming all others; however, for achieving higher recall, PQ-256-16 actually does much worse than slightly less compressed formats like PQ-256-8 and PQ-256-4.

我们还观察到，在为查询搜索前1000个候选对象时，不同PQ编解码器之间的差异在低召回率情况下更为明显。例如，图15中的大多数图表显示，在低召回率情况下实现的每秒查询率（Queries Per Second，QPS）存在显著分层，其中PQ - 256 - 16（压缩率最高且内存效率最高的格式）通常优于其他所有格式；然而，为了实现更高的召回率，PQ - 256 - 16的表现实际上比PQ - 256 - 8和PQ - 256 - 4等压缩率稍低的格式要差得多。